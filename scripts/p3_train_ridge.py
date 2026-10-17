"""P3-03: train-only Ridge, frozen validation selection and actual development MPC."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
from time import perf_counter
import tomllib
import traceback
import numpy as np
import torch
from push_align import InitialSpec, PlanarPushEnv
from push_align.closed_loop import ENGINE_GUARDS, consume_actual_step, model_planning_scope
from push_align.config import ROOT
from push_align.data import DatasetWriter, EpisodeRecorder, digest, iter_dataset, make_contract
from push_align.learning_data import load_learning_data
from push_align.planning import LearnedPredictor, PlannerContext, TaskSpec, plan
from push_align.preparation import SPLIT_CODES
from push_align.prediction import evaluate_predictions, evaluate_state_hold
from push_align.ridge import fit_ridge, save_ridge, load_ridge
from push_align.task import fail_task, handoff_metrics, pose_in_goal, start_task
from p2_collect_pilot import save_trace
from p2_closed_loop_pilot import case_plan, compact_plan, causal_prefix, holding_fixture
from p2_episode_contract_check import source_bundle
from p2_train_pilot import save, save_evaluation
from p3_geometry_pilot import verify_artifacts

DEFAULT_LEARNING = ROOT/'runs/p3-02-20261009-01/learning_data'


EXPECTED_SETTINGS_SHA256 = '25015ce983055e4d0860c87946ad42cbfc6fc3fcf152394bfcdcd4bf01f5a55c'
EXPECTED_CLOSED_SETTINGS_SHA256 = '98b490e873c64302a2380e090623f944f611edf7cdfc3c09deea774cea1a8738'


def validate_settings(settings):
    encoded=json.dumps(settings,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != EXPECTED_SETTINGS_SHA256:
        raise ValueError('P3-03 requires the predeclared settings; change version before new experiments')
    return settings


def fit_train_ridge(prepared, normalizer, alpha):
    a = prepared.arrays
    rows = np.flatnonzero(a['split_code'] == SPLIT_CODES['train'])
    if not len(rows):
        raise ValueError('No train rows to fit')
    model, diagnostics = fit_ridge(a['features'][rows].copy(), a['deltas'][rows].copy(), normalizer, alpha)
    return model, diagnostics, rows


def choose_alpha(candidates):
    eligible = [c for c in candidates if c['selection_status'] == 'eligible'
                and isinstance(c['selection_score'], (int, float)) and not isinstance(c['selection_score'], bool)
                and np.isfinite(c['selection_score']) and c['selection_score'] >= 0]
    if not eligible:
        raise ValueError('No eligible K5 validation candidate; preserve diagnostics without selected model')
    return min(eligible, key=lambda c: (c['selection_score'], c['alpha']))


def array_hash(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def development(out, bundle, prepared, normalizer, settings, config, selection_hash):
    out.mkdir(parents=True, exist_ok=False)
    began = perf_counter()
    profile = tomllib.loads((ROOT/'configs/acceptance.v1.toml').read_text())
    predictor = LearnedPredictor(bundle.model, normalizer)
    planner_config = deepcopy(config)
    planner_config['planner']['command_speed_norm_max_m_s'] = settings['candidate_speed_norm_max_m_s']
    task = TaskSpec.from_config(config)
    env, control_env = PlanarPushEnv(config), PlanarPushEnv(config)
    contract = make_contract(env)
    if contract != prepared.metadata['contract']:
        raise ValueError('Execution environment differs from the unchanged main-data physical contract')
    cases = case_plan(settings, config)
    weights = {k: v.clone() for k, v in bundle.model.state_dict().items()}
    save(out/'manifest.json', {'task_id':'P3-03-development', 'cases':cases, 'settings':settings,
         'resolved_planning_config':planner_config, 'ridge_hash':bundle.metadata['ridge_hash'],
         'selection_sha256':selection_hash, 'normalizer_hash':normalizer.metadata['normalizer_hash'],
         'case_source':'configs/p2_closed_loop.v1.toml; previously used development cases',
         'scope':'actual same-interface MPC development; no final-test or model-superiority claim'})
    fixture_checks = holding_fixture(out, control_env, config, task, profile)
    writer = DatasetWriter(out/'data'/'p3-ridge-development-v1', contract, purpose='pilot')
    train_features = prepared.arrays['features'][prepared.arrays['split_code'] == SPLIT_CODES['train']]
    guards = dict.fromkeys(ENGINE_GUARDS, 0)
    ledger, planning_times, execution_times, causal = [], [], [], []
    for case in cases:
        folder = out/'episodes'/case['episode_id']
        folder.mkdir(parents=True)
        clock = perf_counter()
        recorder, records = None, []
        try:
            observation = env.reset(InitialSpec(**case['initial_spec']))
            status = start_task(observation, task, env.action_dt)
            context = PlannerContext.initial()
            rng = np.random.default_rng(case['candidate_seed'])
            recorder = EpisodeRecorder(env, episode_id=case['episode_id'], episode_index=case['episode_index'],
                                       family_id=case['family_id'], family_definition={'base_initial_spec': case['initial_spec'],
                                       'variant_rule': 'one fixed development reset; no outcome-dependent replacement'},
                                       split='smoke-test', collection_policy={'kind': 'learned_mpc_closed_loop',
                                       'purpose': 'pilot', 'closed_loop_id': 'p3-ridge-development-v1', 'ridge_hash': bundle.metadata['ridge_hash']},
                                       seed=case['candidate_seed'], contract=contract)
            for index in range(settings['max_action_intervals']):
                if status.phase in {'ready', 'failed'}:
                    break
                holding = pose_in_goal(observation.state, task)
                planned, predicted_next, plan_info = None, None, None
                if holding:
                    action = np.zeros(2, np.float64)
                    selected_index, source = None, 'holding'
                else:
                    before_calls, before_rows = predictor.calls, predictor.rows
                    with model_planning_scope(guards):
                        planned = plan(observation.state.copy(), task, context, rng, predictor, planner_config)
                    selected_index, source = planned.selected_index, 'model'
                    identity = {'case': case, 'control_step': index, 'current_sim_time': observation.sim_time}
                    plan_info = compact_plan(folder/f'plans/step-{index:03d}', planned, identity,
                                             predictor.calls-before_calls, predictor.rows-before_rows, train_features)
                    planning_times.append(planned.diagnostics['wall_time_s'])
                    if planned.reason == 'planner_invalid':
                        status = fail_task(status, 'planner_invalid')
                        save(folder/'unexecuted_final_plan.json', {'step': index, 'reason': planned.reason, 'no_extra_zero_step': True})
                        break
                    action, context = planned.action, planned.context
                    predicted_next = planned.predictions.states[selected_index, 1].copy()
                before = observation.state.copy()
                tick = perf_counter()
                step = recorder.step(action)
                execution_s = perf_counter()-tick
                execution_times.append(execution_s)
                geometry = recorder.geometry[-1]
                status, context = consume_actual_step(status, step, task, env.action_dt, context,
                                                     holding=holding, geometry=geometry)
                observation = step.observation
                metrics = handoff_metrics(observation.state, task) if np.isfinite(observation.state).all() else None
                error = None
                if predicted_next is not None and np.isfinite(observation.state).all():
                    difference = observation.state-predicted_next
                    difference[2] = np.arctan2(np.sin(difference[2]), np.cos(difference[2]))
                    error = {'object_position_m': float(np.linalg.norm(difference[:2])),
                             'object_yaw_rad': float(abs(difference[2])), 'state_error': difference.tolist()}
                record = {'index': index, 'source': source, 'selected_index': selected_index,
                          'requested_action': action.tolist(), 'applied_action': step.applied_action.tolist(),
                          'state_before': before.tolist(), 'state_after': observation.state.tolist(),
                          'actual_dt_s': step.actual_dt, 'sim_time': observation.sim_time,
                          'task': asdict(status), 'metrics': metrics, 'geometry': geometry,
                          'physical_valid': bool(step.transition_valid), 'prediction_error': error,
                          'plan': plan_info, 'execution_wall_time_s': execution_s}
                # Strict JSON is validated before each line; raw nonfinite physics
                # is retained by EpisodeRecorder even if summary serialization fails.
                from push_align.data import _encode_nonfinite
                encoded = json.dumps(_encode_nonfinite(record), ensure_ascii=False, allow_nan=False)
                with (folder/'steps.jsonl').open('a', encoding='utf-8') as file:
                    file.write(encoded+'\n')
                records.append(record)
                if (index+1) % 25 == 0:
                    print(json.dumps({'progress': 'closed_loop', 'episode': case['episode_id'],
                                      'steps': index+1, 'elapsed_sim_s': status.elapsed_s, 'phase': status.phase}), flush=True)
            if status.phase not in {'ready', 'failed'}:
                raise ValueError('Declared execution budget ended without terminal outcome')
            if recorder.results:
                episode = recorder.finish(status.terminal_reason)
                episode.metadata['task_outcome'] = status.phase
                episode.metadata['supervised_task_status'] = asdict(status)
                writer.append(episode)
                prefix = causal_prefix(folder/'causal_prefix', episode, records, recorder.snapshots[0],
                                       control_env, profile, settings['causal_prefix_intervals'])
                causal.append({'episode_id': case['episode_id'], **prefix})
                raw_terminal = episode.metadata['terminal_reason']
                transitions, excluded = episode.transition_count, int((~episode.arrays['transition_valid']).sum())
                del episode
            else:
                raw_terminal, transitions, excluded = None, 0, 0
                save_trace(folder, 'empty_execution', recorder.snapshots[0], [])
                save(folder/'empty_execution.json', {'terminal_reason': status.terminal_reason,
                                                   'scope': 'finished task without physical interval; not a dataset episode'})
                causal.append({'episode_id': case['episode_id'], 'pass': False, 'reason': 'no_executed_interval'})
            entry = {'episode_id': case['episode_id'], 'status': 'completed_execution', 'task': asdict(status),
                     'raw_terminal_reason': raw_terminal, 'transitions': transitions, 'excluded_transitions': excluded,
                     'model_selected_applied_intervals': sum(r['source']=='model' for r in records),
                     'holding_intervals': sum(r['source']=='holding' for r in records),
                     'final_metrics': handoff_metrics(observation.state, task) if np.isfinite(observation.state).all() else None,
                     'wall_time_s': perf_counter()-clock}
            save(folder/'summary.json', entry)
            ledger.append(entry)
            print(json.dumps({'progress': 'episode_complete', 'episode': case['episode_id'],
                              'reason': status.terminal_reason, 'steps': transitions,
                              'causal_prefix_pass': causal[-1]['pass']}), flush=True)
            del recorder, records
            gc.collect()
        except Exception as error:
            save(folder/'failure.json', {'case': case, 'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
            if recorder is not None:
                save_trace(folder, 'failed_recorded_prefix', recorder.snapshots[0], recorder.results)
            ledger.append({'episode_id': case['episode_id'], 'status': 'infrastructure_error', 'error': str(error)})
            print(json.dumps({'progress': 'episode_error', 'episode': case['episode_id'], 'error': str(error)}), flush=True)
    save(out/'ledger.json', ledger)
    save(out/'causal_summary.json', causal)
    completed_dataset = writer.finish() if writer.entries else None
    reviews = []
    if completed_dataset is not None:
        for episode in iter_dataset(writer.path, contract):
            reviews.append({'episode_id': episode.metadata['episode_id'], 'transitions': episode.transition_count,
                            'valid': int(episode.arrays['transition_valid'].sum())})
            del episode
    save(out/'dataset_review.json', {'episodes': reviews, 'empty_executions_are_separate': True})
    codes = Counter(row['task']['terminal_reason'] for row in ledger if 'task' in row)
    unresolved = [row['episode_id'] for row in ledger if row['status'] != 'completed_execution'
                  or row.get('task', {}).get('terminal_reason') in settings['unresolved_physics_codes']]
    checks = {
        'all_ten_declared_executions_accounted':len(ledger)==10 and [r['episode_id'] for r in ledger]==[r['episode_id'] for r in cases],
        'no_unfinished_infrastructure_or_unresolved_physics':not unresolved,
        'model_used_in_shared_planner':predictor.calls>0,
        'future_engine_calls_in_planning_zero':sum(guards.values())==0,
        'every_recorded_episode_audited':len(reviews)==len(writer.entries),
        'model_weights_unchanged':all(torch.equal(v,bundle.model.state_dict()[k]) for k,v in weights.items()),
        'selection_frozen_before_and_after_execution':digest(out.parent/'selection.json')==selection_hash,
        **fixture_checks,
    }
    p = np.asarray(planning_times)
    resources = {'total_wall_time_s':perf_counter()-began,
        'main_peak_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024),
        'plan_count':len(p), 'planning_mean_s':float(p.mean()) if len(p) else None,
        'planning_p95_s':float(np.percentile(p,95)) if len(p) else None,
        'execution_intervals':len(execution_times),
        'scope':'CPU with Torch one thread; BLAS thread count unmeasured; includes recorder/audit; no physical-device real-time validation'}
    summary = {'task_id':'P3-03-development', 'pass':all(checks.values()), 'checks':checks,
        'ledger':ledger, 'terminal_counts':dict(codes), 'unresolved_items':unresolved,
        'causal_prefix_pass_count':sum(r['pass'] for r in causal),
        'causal_scope':'all first prefixes retained; measured sequence contribution diagnostic, not an alpha-selection rule or Ridge completion gate',
        'dataset_recorded_episodes':len(reviews), 'dataset_transitions':sum(r['transitions'] for r in reviews),
        'future_engine_calls_during_planning':guards, 'actual_model_calls':predictor.calls,
        'alignment_ready_count':codes.get('ready',0),
        'alignment_outcome':'CASE_DEMONSTRATED' if codes.get('ready',0) else 'NOT_DEMONSTRATED',
        'performance_scope':'same ten previously used development cases; different training data from pilot MLP; not a fair model-superiority comparison',
        'resources':resources, 'user_understanding':'NOT_ASSESSED'}
    save(out/'summary.json',summary)
    return summary


def toy_regression(normalizer):
    x = np.zeros((3,11),np.float64); x[:,2]=[-1,0,1]
    y = np.zeros((3,10),np.float64); y[:,0]=[1,3,5]
    physical_x=x*normalizer.arrays['input_scale']+normalizer.arrays['input_mean']
    physical_y=y*normalizer.arrays['delta_scale']+normalizer.arrays['delta_mean']
    predictions={}; checks={}
    for alpha,expected in [(0.,[1.,3.,5.]),(2.,[2.,3.,4.])]:
        model,diag=fit_ridge(physical_x,physical_y,normalizer,alpha)
        with torch.no_grad(): p=model(torch.from_numpy(normalizer.normalize_features(physical_x))).numpy()[:,0]
        predictions[str(alpha)]=p.tolist()
        checks[str(alpha)]=bool(np.allclose(p,expected,atol=2e-5,rtol=0))
    return {'checks':checks,'pass':all(checks.values()),'predicted_normalized_first_output':predictions,
            'expected':{'0.0':[1.,3.,5.],'2.0':[2.,3.,4.]},
            'scope':'analytical rank-one artificial fixture, not robot prediction performance'}


def run(args,out):
    began=perf_counter(); settings=validate_settings(tomllib.loads(args.ridge_config.read_text()))
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    learning=args.learning_data.resolve()
    if learning != DEFAULT_LEARNING.resolve(): raise ValueError('Main dataset identity is frozen for this runner version')
    upstream_before=verify_artifacts(learning.parent)
    m=json.loads((learning/'normalizer/normalizer.json').read_text())['normalizer']
    if m['normalizer_hash']!=settings['normalizer_hash'] or m['dataset_identity']['manifest_sha256']!=settings['dataset_manifest_sha256']:
        raise ValueError('Frozen main dataset/normalizer changed')
    prepared,normalizer=load_learning_data(learning,m['schema'],m['dataset_identity'],m['normalizer_hash'])
    a=prepared.arrays; config=prepared.metadata['contract']['resolved_config']
    train=np.flatnonzero(a['split_code']==SPLIT_CODES['train']); val=np.flatnonzero(a['split_code']==SPLIT_CODES['validation'])
    if len(train)!=settings['fit_count'] or len(val)!=settings['validation_count'] or len(train)+len(val)!=4800:
        raise ValueError('Frozen row counts/split changed')
    closed_path=ROOT/settings['closed_loop_config']
    if digest(closed_path)!=EXPECTED_CLOSED_SETTINGS_SHA256:
        raise ValueError('The original P2-06 development cases/seeds/time budget changed')
    closed_settings=tomllib.loads(closed_path.read_text())
    if len(closed_settings['cases'])!=10 or closed_settings['candidate_speed_norm_max_m_s']!=.03 or config['planner']['candidate_count']!=256 or config['planner']['horizon']!=10:
        raise ValueError('Same-interface development plan settings differ')
    provenance={'learning_manifest_sha256':digest(learning/'manifest.json'),'dataset_identity':m['dataset_identity'],
                'validation_split':'validation','task_scope':'P3-03-main-ridge'}
    sources=source_bundle(out)
    manifest={'task_id':'P3-03','created_at':datetime.now().astimezone().isoformat(),'argv':sys.argv,
        'executable':sys.executable,'python':platform.python_version(),'platform':platform.platform(),
        'numpy':str(np.__version__),'torch':str(torch.__version__),'settings':settings,
        'ridge_config_sha256':digest(args.ridge_config),'source_hashes':sources,'provenance':provenance,
        'normalizer_hash':m['normalizer_hash'],'stats_hash':m['stats_hash'],
        'original_development_settings_sha256':EXPECTED_CLOSED_SETTINGS_SHA256,
        'train_rows':train.tolist(),'validation_rows':val.tolist(),'upstream_artifacts':upstream_before,
        'original_physical_config':config,'resolved_development_planning_config':{**deepcopy(config['planner']),'command_speed_norm_max_m_s':.03},
        'validation_window_hashes':{str(k):array_hash(w[a['split_code'][w[:,0]]==1]) for k,w in prepared.windows.items()},
        'normalization_scope':'uses unchanged stored statistics; loader independently recomputes train-only statistics for audit, without replacing or writing them',
        'scope':settings['scope'],'user_understanding':'NOT_ASSESSED'}
    save(out/'manifest.json',manifest); manifest_hash=digest(out/'manifest.json')
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,capture_output=True,text=True)
    log=tests.stdout+tests.stderr; (out/'tests.txt').write_text(log)
    match=re.search(r'Ran (\d+) tests',log); count=int(match.group(1)) if match else 0
    if tests.returncode or count<=0 or 'skipped=' in log: raise ValueError('Mandatory regression checks failed')
    print(json.dumps({'progress':'regression_complete','tests':count}),flush=True)
    toy=toy_regression(normalizer); save(out/'toy_regression.json',toy)
    if not toy['pass']: raise ValueError('Analytical toy regression failed')
    save_evaluation(out/'state_hold_validation',evaluate_state_hold(prepared))
    candidates=[]; models={}; fit_times=[]
    for index,alpha in enumerate(settings['alpha_candidates']):
        folder=out/'candidates'/f'alpha-{index:02d}'; folder.mkdir(parents=True)
        tick=perf_counter(); model,diag,rows=fit_train_ridge(prepared,normalizer,alpha); fit_s=perf_counter()-tick;fit_times.append(fit_s)
        evaluation=evaluate_predictions(model,normalizer,prepared)
        save_evaluation(folder/'validation',evaluation)
        metadata=save_ridge(folder/'model',model,normalizer,config,provenance,alpha=alpha,fit_diagnostics=diag,selection_score=evaluation.summary['selection_score'])
        restored=load_ridge(folder/'model',normalizer,config,provenance)
        repeat=evaluate_predictions(restored.model,normalizer,prepared)
        restored_equal=all(np.array_equal(evaluation.traces[k],repeat.traces[k],equal_nan=True) if np.issubdtype(evaluation.traces[k].dtype,np.floating) else np.array_equal(evaluation.traces[k],repeat.traces[k]) for k in evaluation.traces)
        if not restored_equal or evaluation.summary!=repeat.summary:raise ValueError('Saved Ridge prediction replay differs')
        row={'index':index,'alpha':alpha,'selection_score':evaluation.summary['selection_score'],
             'selection_status':evaluation.summary['selection_status'],'model_path':str((folder/'model').relative_to(out)),
             'model_hash':restored.metadata['ridge_hash'],'fit_diagnostics':diag,'fit_time_s':fit_s,
             'restored_all_prediction_traces_identical':restored_equal,
             'K5_attempted_windows':evaluation.summary['horizons']['5']['attempted_windows'],
             'K5_invalid_windows':evaluation.summary['horizons']['5']['invalid_windows'],
             'validation_window_hashes':{str(k):array_hash(evaluation.traces[f'h{k}_window_rows']) for k in (1,5,10)}}
        save(folder/'summary.json',row);candidates.append(row);models[index]=model
        print(json.dumps({'progress':'ridge_candidate','alpha':alpha,'score':row['selection_score'],'status':row['selection_status']}),flush=True)
    selected=choose_alpha(candidates)
    save(out/'selection.json',{'candidates':candidates,'selected':selected,'rule':settings['selection_rule'],
         'scope':'frozen before development outcomes; validation selection score is not independent held-out performance',
         'final_test_used':False})
    selection_hash=digest(out/'selection.json')
    bundle=load_ridge(out/selected['model_path'],normalizer,config,provenance)
    # In-memory leakage probe; original artifact and normalization are untouched.
    mutated=deepcopy(prepared);mutated.arrays['features'][val]+=1e4;mutated.arrays['deltas'][val]-=1e4
    control,_,control_rows=fit_train_ridge(mutated,normalizer,selected['alpha'])
    no_leak=np.array_equal(control_rows,train) and all(torch.equal(v,control.state_dict()[k]) for k,v in bundle.model.state_dict().items())
    save(out/'train_only_probe.json',{'validation_rows_modified_in_memory':len(val),'original_train_rows_fit':len(train),
        'model_weights_unchanged':no_leak,'source_files_mutated':False,
        'scope':'fit-mask experiment; original loaded prepared data is validated separately'})
    if not no_leak:raise ValueError('Validation values entered the Ridge fit')
    development_summary=development(out/'development',bundle,prepared,normalizer,closed_settings,config,selection_hash)
    upstream_after=verify_artifacts(learning.parent)
    checks={'regression_tests_pass':True,'analytical_rank_one_ridge_and_ols':toy['pass'],
        'fit_uses_original_3200_train_rows':all(c['fit_diagnostics']['fit_count']==3200 for c in candidates),
        'validation_fit_mask_invariance':no_leak,'all_eight_candidates_preserved':len(candidates)==8,
        'selection_matches_predeclared_rule':choose_alpha(candidates)==selected,
        'same_validation_windows_for_all_candidates':all(c['validation_window_hashes']==manifest['validation_window_hashes'] for c in candidates),
        'selected_has_all_K5_windows_valid':selected['selection_status']=='eligible' and selected['K5_invalid_windows']==0,
        'all_saved_models_and_free_predictions_reload_identically':all(c['restored_all_prediction_traces_identical'] for c in candidates),
        'same_model_planning_interface':development_summary['checks']['model_used_in_shared_planner'],
        'development_all_required_checks':development_summary['pass'],
        'upstream_raw_data_statistics_and_manifests_unchanged':upstream_before==upstream_after,
        'manifest_and_selected_alpha_frozen':digest(out/'manifest.json')==manifest_hash and digest(out/'selection.json')==selection_hash,
        'source_files_unchanged_during_run':all(digest(ROOT/name)==sha for name,sha in sources.items())}
    resources={'total_wall_time_s':perf_counter()-began,'candidate_fit_total_s':sum(fit_times),
        'main_peak_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024),
        'development':development_summary['resources'],'scope':'CPU with Torch one thread; BLAS thread count unmeasured; includes regression, artifact verification and development; no MLP training or physical real-time claim'}
    summary={'task_id':'P3-03','pass':all(checks.values()),'stage_verdict':'PASS' if all(checks.values()) else 'FAIL',
        'checks':checks,'test_count':count,'selected':selected,'candidates':candidates,'resources':resources,
        'development':development_summary,'next_task':'P3-04','user_understanding':'NOT_ASSESSED',
        'model_claim':'trained linear function of fixed nonlinear angle features, not linear true contact physics or best analytical model',
        'comparison_claim':'no fair superiority claim versus the pilot-data MLP; common-main-data MLP remains next task'}
    save(out/'resources.json',resources);save(out/'summary.json',summary)
    save(out/'artifact_hashes.json',{'sha256':{str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    if not all(checks.values()):raise ValueError('P3-03 gate failed; preserve original diagnostics')
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    parser.add_argument('--learning-data',type=Path,default=DEFAULT_LEARNING)
    parser.add_argument('--ridge-config',type=Path,default=ROOT/'configs/p3_ridge.v1.toml')
    args=parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',args.run_id):raise ValueError('Unsafe run ID')
    out=ROOT/'runs'/args.run_id;out.mkdir(parents=True,exist_ok=False)
    try:result=run(args,out)
    except Exception as error:
        save(out/'failure.json',{'task_id':'P3-03','pass':False,'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        print(json.dumps({'run':str(out),'pass':False,'error':str(error)}),flush=True);return 1
    print(json.dumps({'run':str(out),'pass':result['pass'],'alpha':result['selected']['alpha']}),flush=True);return 0

if __name__=='__main__':raise SystemExit(main())
