"""P3-05 bounded development selection and measured resource preparation.

Every attempted trial and executed physical prefix is retained. Five-second
screens are explicitly development_budget, never task timeout or final test.
The common MLPs here are diagnostic precursors, not the later S1/SK branches.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import gc
import gzip
import hashlib
import json
from pathlib import Path
import platform
import re
import resource
import shutil
import subprocess
import sys
from time import perf_counter
import tomllib
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import torch
from push_align import InitialSpec, PlanarPushEnv
from push_align.closed_loop import ENGINE_GUARDS, consume_actual_step, model_planning_scope
from push_align.config import ROOT
from push_align.contact_geometry import check_contact_geometry
from push_align.data import _encode_nonfinite, digest, make_contract
from push_align.dynamics import free_rollout
from push_align.learning_data import load_learning_data
from push_align.planning import LearnedPredictor, PlannerContext, TaskSpec, plan
from push_align.planning_protocol import GeometryProposalAdapter, keyed_planner_rng
from push_align.preparation import SPLIT_CODES
from push_align.representation import state_delta
from push_align.ridge import load_ridge
from push_align.task import fail_task, handoff_metrics, pose_in_goal, start_task
from push_align.training import one_step_loss
from p2_closed_loop_pilot import case_plan
from p2_collect_pilot import save_trace
from p2_episode_contract_check import source_bundle
from p2_train_pilot import save
from p3_geometry_pilot import verify_artifacts
from p3_train_common_mlp import load_common_checkpoint
from p3_train_ridge import array_hash

SETTINGS_SHA256 = '81f6e2bf2af422a63c6603a4c0a6281f160c4dcc1abe706fbc53d57dd375c8e4'
LEARNING = ROOT/'runs/p3-02-20261009-01/learning_data'
COMMON = ROOT/'runs/p3-04-20261009-01'
RIDGE = ROOT/'runs/p3-03-20261009-01'


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_settings(value):
    if canonical_hash(value) != SETTINGS_SHA256:
        raise ValueError('P3-05 requires the predeclared settings fingerprint; version changes precede experiments')
    return value


def trial_score(rows, expected_ids):
    """No survivor-only selection; declared case x model cells all count."""
    if [r['cell_id'] for r in rows] != expected_ids or not rows:
        raise ValueError('Missing, reordered or extra development cell')
    eligible = all(r['execution_status'] == 'completed' and r['termination'] not in
                   {'physics_error', 'geometry_invalid', 'infrastructure_error', 'controller_infeasible'} for r in rows)
    losses = [r.get('terminal_pose_loss') for r in rows]
    eligible = eligible and all(type(v) in (int, float) and np.isfinite(v) and v >= 0 for v in losses)
    return {'eligible': bool(eligible), 'ready': sum(r['termination']=='ready' for r in rows),
            'task_failures':sum(r['termination'] not in {'ready','development_budget','timeout'} for r in rows),
            'mean_terminal_pose_loss': float(np.mean(losses)) if eligible else None,
            'attempted': len(rows), 'expected': len(expected_ids)}


def select_trial(trials):
    eligible = [t for t in trials if t['score']['eligible']]
    if not eligible:
        raise ValueError('No eligible setting; preserve all trials and revisit shared contracts')
    # Python min keeps declared order on exact ties; wall time is descriptive.
    return min(eligible, key=lambda t: (-t['score']['ready'], t['score']['task_failures'], t['score']['mean_terminal_pose_loss']))


def resolved_planner(config, trial, settings):
    c = deepcopy(config)
    c['planner'].update({k: trial[k] for k in ('candidate_count', 'horizon', 'block_lengths')})
    c['planner']['geometry_proposal_enabled'] = settings['geometry_proposal_enabled']
    c['planner']['command_speed_norm_max_m_s'] = settings['candidate_speed_norm_max_m_s']
    return c


def statistic(values):
    a = np.asarray(values, np.float64)
    return {'count': len(a), 'first_s': float(a[0]) if len(a) else None,
            'mean_s': float(a.mean()) if len(a) else None,
            'p95_s': float(np.percentile(a, 95)) if len(a) else None,
            'max_s': float(a.max()) if len(a) else None,
            'over_action_dt_0_1s': int((a > .1).sum())}


def retain_plan(folder, result, identity, *, full):
    """Lossless executed raw is separate; detailed predictions are sampled.

    The compact record includes enough input/context/provenance to regenerate
    every candidate and prediction with the archived model and source.
    """
    i = result.selected_index
    row = {**identity, 'selected_index': i, 'reason': result.reason,
           'candidate_hash': array_hash(result.candidates.actions),
           'prediction_hash': array_hash(result.predictions.states),
           'delta_hash': array_hash(result.predictions.raw_deltas),
           'valid_predictions': int(result.predictions.valid.sum()),
           'eligible_costs': int((result.predictions.valid & np.isfinite(result.costs.total)).sum()),
           'invalid_reasons': dict(Counter(r for r in result.predictions.reasons if r)),
           'candidate_generation': result.candidates.diagnostics,
           'selected_cost': None if i is None else float(result.costs.total[i]),
           'selected_cost_components': None if i is None else {k: float(v[i]) for k,v in result.costs.components.items()},
           'selected_actions': None if i is None else result.candidates.actions[i].tolist(),
           'full_trace': bool(full), 'planning_core_wall_time_s': result.diagnostics['wall_time_s']}
    if full:
        folder.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(folder/'trace.npz', candidates=result.candidates.actions,
                            predicted_states=result.predictions.states, raw_deltas=result.predictions.raw_deltas,
                            valid=result.predictions.valid, failure_steps=result.predictions.failure_steps,
                            total_cost=result.costs.total, reasons=np.array(result.predictions.reasons),
                            **{'cost_'+k:v for k,v in result.costs.components.items()})
        save(folder/'identity.json', row)
    return row


def run_cell(out, case, snapshot, model_id, bundle, normalizer, physical, planner,
             geometry_parameters, transition, settings, *, intervals, resource_case=False):
    out.mkdir(parents=True, exist_ok=False)
    clock = perf_counter(); env = PlanarPushEnv(physical)
    observed = env.restore(snapshot); restored = env.snapshot()
    # Check full integration state, not only the ten observed coordinates.
    initial_equal = (np.array_equal(restored.integration_state, snapshot.integration_state)
                     and canonical_hash(_encode_nonfinite(asdict(restored))) == canonical_hash(_encode_nonfinite(asdict(snapshot))))
    if not initial_equal:
        raise ValueError('Initial full snapshot failed exact restoration')
    task = TaskSpec.from_config(physical); status = start_task(observed, task, env.action_dt)
    context = PlannerContext.initial(); adapter = GeometryProposalAdapter(physical, geometry_parameters, transition)
    gctx = adapter.initial_context(); guards = dict.fromkeys(ENGINE_GUARDS, 0)
    predictor = None if bundle is None else LearnedPredictor(bundle.model, normalizer)
    before_weights = None if bundle is None else {k:v.clone() for k,v in bundle.model.state_dict().items()}
    profile = tomllib.loads((ROOT/'configs/acceptance.v1.toml').read_text())
    results, records, gtimes, coretimes, decisiontimes, steptimes, writetimes = [], [], [], [], [], [], []
    full_ticks = settings['full_trace_ticks_resource' if resource_case else 'full_trace_ticks_screen']
    frozen = {'case':case, 'model_id':model_id, 'model_metadata':None if bundle is None else bundle.metadata,
              'physical_contract':make_contract(env), 'planner_config':planner,
              'effective_geometry_parameters':adapter.parameters, 'transition_parameters':transition,
              'initial_snapshot_hash':canonical_hash(_encode_nonfinite(asdict(snapshot))),
              'initial_state':observed.state.tolist(), 'max_intervals':intervals,
              'scope':'development resource execution' if resource_case else 'five-second development screen'}
    save(out/'manifest.json', frozen)
    termination = None; execution_status = 'completed'; first_candidate = None; replay_checks = []
    try:
        for index in range(intervals):
            if status.phase in ('ready','failed'):
                break
            if index % 25 == 0 and shutil.disk_usage(ROOT).free < settings['minimum_free_disk_bytes']:
                raise ValueError('Predeclared free-disk reserve reached; preserve executed prefix')
            holding = pose_in_goal(observed.state, task); before = observed.state.copy()
            previous = {'previous_applied_action':context.previous_applied_action.tolist(),
                        'optimal_actions':None if context.optimal_actions is None else context.optimal_actions.tolist(),
                        'control_step':context.control_step}
            gc_before = asdict(gctx); tick = perf_counter()
            with model_planning_scope(guards):
                begin_g = perf_counter()
                proposal = adapter.propose(before.copy(), task, gctx, status.elapsed_s, control_tick=index)
                gtimes.append(perf_counter()-begin_g)
                if bundle is None:
                    planned = None
                    action = np.zeros(2,np.float64) if holding else proposal.geometry_action
                elif holding:
                    planned = None; action = np.zeros(2,np.float64)
                else:
                    rng = keyed_planner_rng(settings['candidate_seed_base'], case['episode_index'], index, settings['candidate_stream_id'])
                    rng_before = deepcopy(rng.bit_generator.state)
                    planned = plan(before.copy(), task, context, rng, predictor, planner,
                                   geometry_action=proposal.geometry_action, tangent_direction=proposal.tangent_direction)
                    action = planned.action; coretimes.append(planned.diagnostics['wall_time_s'])
            decisiontimes.append(perf_counter()-tick)
            # Context records actual observed phase transitions, never G action execution.
            gctx = adapter.commit(proposal, actual_control_tick=index)
            info = None
            if planned is not None:
                identity = {'state':before.tolist(), 'task_status':asdict(status), 'task_spec':asdict(task),
                            'planner_context_before':previous, 'geometry_context_before':gc_before,
                            'geometry_proposal':proposal.diagnostics, 'geometry_context_after':asdict(gctx),
                            'rng_keys':[settings['candidate_seed_base'],case['episode_index'],index,settings['candidate_stream_id']],
                            'rng_state_before':rng_before, 'model_id':model_id,
                            'model_hash':bundle.metadata.get('checkpoint_hash',bundle.metadata.get('ridge_hash')),
                            'planner_config_hash':canonical_hash(planner), 'tick':index,
                            'source_manifest_sha256':digest(out/'manifest.json')}
                if index in full_ticks or planned.reason=='planner_invalid':
                    identity['physical_snapshot']=asdict(env.snapshot())
                saved = perf_counter()
                info = retain_plan(out/f'plans/step-{index:03d}', planned, _encode_nonfinite(identity),
                                   full=index in full_ticks or planned.reason=='planner_invalid')
                writetimes.append(perf_counter()-saved)
                if first_candidate is None:
                    first_candidate = info['candidate_hash']
                    # Independent freshly keyed stream, same frozen inputs/context.
                    with model_planning_scope(guards):
                        again = plan(before.copy(), task, context,
                                     keyed_planner_rng(*identity['rng_keys']), predictor, planner,
                                     geometry_action=proposal.geometry_action, tangent_direction=proposal.tangent_direction)
                    replay_checks.append(np.array_equal(again.candidates.actions, planned.candidates.actions)
                                         and np.array_equal(again.predictions.states, planned.predictions.states, equal_nan=True)
                                         and np.array_equal(again.costs.total, planned.costs.total)
                                         and again.selected_index==planned.selected_index)
                if planned.reason=='planner_invalid':
                    status = fail_task(status,'planner_invalid'); termination=status.terminal_reason
                    save(out/'unexecuted_final_plan.json',info); break
                context = planned.context
            elif action is None:
                status = fail_task(status,'controller_infeasible'); termination=status.terminal_reason
                save(out/'unexecuted_final_G.json',proposal.diagnostics); break
            step_clock = perf_counter(); step = env.step(action); steptimes.append(perf_counter()-step_clock)
            results.append(step)
            geometry = check_contact_geometry(step.diagnostics['samples'],profile['physics_suite']['actual_contact_penetration_max_m'])
            status, context = consume_actual_step(status,step,task,env.action_dt,context,holding=holding,geometry=geometry)
            observed = step.observation
            row = {'index':index,'source':'holding' if holding else 'geometry' if bundle is None else 'model',
                   'requested_action':action.tolist(),'applied_action':step.applied_action.tolist(),
                   'state_before':before.tolist(),'state_after':observed.state.tolist(),'actual_dt_s':step.actual_dt,
                   'physical_valid':bool(step.transition_valid),'geometry':geometry,'task':asdict(status),
                   'G_proposal':proposal.diagnostics,'G_context_committed':asdict(gctx),'plan':info,
                   'execution_wall_time_s':steptimes[-1]}
            with (out/'steps.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(_encode_nonfinite(row),ensure_ascii=False,allow_nan=False)+'\n')
            records.append(row)
        termination = status.terminal_reason or ('development_budget' if not resource_case else 'unfinished_budget')
        if resource_case and termination=='unfinished_budget':
            raise ValueError('Full 300-tick execution missing terminal task outcome')
    except Exception as error:
        execution_status='infrastructure_error';termination='infrastructure_error'
        save(out/'failure.json',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
    finally:
        raw_clock=perf_counter(); save_trace(out,'raw_execution',snapshot,results); raw_save_s=perf_counter()-raw_clock
    # Verify compressed raw against the original in-memory fields, not merely
    # that a gzip reader accepts it. Nonfinite paths/values stay separately.
    paths,values=[],[]
    expected_diagnostics=_encode_nonfinite([r.diagnostics for r in results],paths=paths,values=values)
    with gzip.open(out/'raw_execution.json.gz','rt',encoding='utf-8') as f:
        raw_json=json.load(f)
    with np.load(out/'raw_execution.npz',allow_pickle=False) as raw:
        raw_roundtrip=(raw_json['diagnostics']==expected_diagnostics
            and raw_json['initial_snapshot']==_encode_nonfinite(asdict(snapshot))
            and raw_json['diagnostic_nonfinite_paths']==paths
            and np.array_equal(raw['diagnostic_nonfinite_values'],np.asarray(values,np.float64),equal_nan=True)
            and np.array_equal(raw['state'],np.asarray([r.observation.state for r in results],np.float64).reshape(-1,10),equal_nan=True)
            and np.array_equal(raw['action_applied'],np.asarray([r.applied_action for r in results],np.float64).reshape(-1,2),equal_nan=True)
            and np.array_equal(raw['substep_states'],np.concatenate([r.substep_states for r in results]) if results else np.empty((0,10)),equal_nan=True))
    metrics = handoff_metrics(observed.state,task) if np.isfinite(observed.state).all() else None
    loss = None if metrics is None else (metrics['e_pos_m']/task.position_tolerance_m)**2+(metrics['e_yaw_rad']/task.yaw_tolerance_rad)**2
    unchanged = bundle is None or all(torch.equal(v,bundle.model.state_dict()[k]) for k,v in before_weights.items())
    resources = {'wall_time_s':perf_counter()-clock,'G_proposal':statistic(gtimes),'model_plan_core':statistic(coretimes),
                 'whole_decision':statistic(decisiontimes),'physics_step_with_diagnostics':statistic(steptimes),
                 'plan_trace_save':statistic(writetimes),'raw_compression_save_s':raw_save_s,
                 'stored_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
                 'raw_bytes':sum(p.stat().st_size for p in out.glob('raw_execution.*')),
                 'scope':'single sequential process; per-model peak RSS not compared; all raw diagnostics preserved compressed'}
    row = {'cell_id':f'{model_id}/{case["episode_id"]}','case_id':case['episode_id'],'model_id':model_id,
           'execution_status':execution_status,'termination':termination,'task':asdict(status),'final_metrics':metrics,
           'terminal_pose_loss':loss,'transitions':len(results),'valid_transitions':sum(r.transition_valid for r in results),
           'holding_intervals':sum(r['source']=='holding' for r in records),'initial_full_snapshot_equal':initial_equal,
           'first_candidate_hash':first_candidate,'first_frozen_plan_replay_pass':all(replay_checks) if predictor else True,
           'model_weights_unchanged':unchanged,'future_engine_calls':guards,'resources':resources}
    row['compressed_raw_roundtrip_exact']=bool(raw_roundtrip)
    save(out/'summary.json',row)
    print(json.dumps({'progress':'cell_complete','cell':row['cell_id'],'terminal':termination,'steps':len(results),
                      'pose_loss':loss,'wall_s':resources['wall_time_s']}),flush=True)
    del results,records;gc.collect()
    return row


def training_cost_probe(out, prepared, normalizer, config, provenance, training_settings, selected, settings):
    """Disposable S1/K5 timing; full gradient/term tests remain P4-01."""
    out.mkdir();a=prepared.arrays
    windows=prepared.windows[5];windows=windows[a['split_code'][windows[:,0]]==SPLIT_CODES['train']]
    rng=np.random.default_rng(5711)
    batches=[windows[rng.integers(len(windows),size=settings['resource_training_probe_batch_windows'])] for _ in range(settings['resource_training_probe_updates'])]
    save(out/'batch_rows.json',{'train_windows':len(windows),'batches':[b.tolist() for b in batches],
                              'scope':'timing diagnostic only; P4 formal sampler/order frozen separately'})
    outputs=[]
    for kind in ('S1-timing','K5-timing'):
        bundle=load_common_checkpoint(COMMON/selected['selected_checkpoint'],normalizer,config,provenance,training_settings,11,resume=True)
        before={k:v.clone() for k,v in bundle.model.state_dict().items()};times=[];losses=[]
        for b in batches:
            clock=perf_counter();bundle.model.train();bundle.optimizer.zero_grad(set_to_none=True)
            f=torch.from_numpy(a['features'][b].reshape(-1,11).copy())
            d=torch.from_numpy(a['deltas'][b].reshape(-1,10).copy())
            loss=one_step_loss(bundle.model,normalizer,f,d);one=float(loss.detach());roll=None
            if kind=='K5-timing':
                initial=torch.from_numpy(a['current_state'][b[:,0]].copy())
                actions=torch.from_numpy(a['applied_action'][b].copy())
                predicted=free_rollout(bundle.model,normalizer,initial,actions)[:,1:]
                truth=torch.from_numpy(a['next_state'][b].copy())
                error=state_delta(truth,predicted)
                scale=torch.from_numpy(normalizer.arrays['delta_scale'].copy())
                roll=((error/scale)**2).mean();loss=loss+settings['resource_training_probe_lambda']*roll
            if not torch.isfinite(loss):raise ValueError('Nonfinite timing-probe loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(bundle.model.parameters(),config['training']['gradient_clip_norm'],error_if_nonfinite=True)
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in bundle.model.parameters()):
                raise ValueError('Missing/nonfinite timing-probe gradient')
            bundle.optimizer.step();times.append(perf_counter()-clock)
            losses.append({'one_step':one,'rollout':None if roll is None else float(roll.detach()),'gradient_norm':float(norm)})
        output={'kind':kind,'updates':len(times),'batch_windows':settings['resource_training_probe_batch_windows'],
                'time':statistic(times),'total_update_s':sum(times),'losses':losses,
                'weights_changed':any(not torch.equal(v,bundle.model.state_dict()[k]) for k,v in before.items()),
                'diagnostic_only':True,'formal_branch_checkpoint_saved':False}
        save(out/(kind+'.json'),output);outputs.append(output)
    return outputs


def run(args,out):
    clock=perf_counter();settings=validate_settings(json.loads(args.settings.read_text()))
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    upstream=[LEARNING.parent,RIDGE,COMMON]
    prior={str(p.relative_to(ROOT)):verify_artifacts(p) for p in upstream}
    m=json.loads((LEARNING/'normalizer/normalizer.json').read_text())['normalizer']
    prepared,normalizer=load_learning_data(LEARNING,m['schema'],m['dataset_identity'],m['normalizer_hash'])
    config=prepared.metadata['contract']['resolved_config']
    mainmanifest=json.loads((COMMON/'manifest.json').read_text());main=json.loads((COMMON/'summary.json').read_text())
    models={}
    selection=json.loads((RIDGE/'selection.json').read_text())
    ridge_path=RIDGE/selection['selected']['model_path']
    models['R']=load_ridge(ridge_path,normalizer,config,json.loads((RIDGE/'manifest.json').read_text())['provenance'])
    for s in main['seeds']:
        models[f'MLP-{s["seed"]}']=load_common_checkpoint(COMMON/s['selected_checkpoint'],normalizer,config,
              mainmanifest['provenance'],mainmanifest['settings'],s['seed'])
    base_geometry=json.loads((ROOT/'configs/p3_geometry_selected.v2.json').read_text())
    transition=json.loads((ROOT/'configs/p3_geometry_transition.v2.json').read_text())
    closed=tomllib.loads((ROOT/'configs/p2_closed_loop.v1.toml').read_text());allcases=case_plan(closed,config)
    cases=[allcases[i] for i in settings['development_case_indices']]
    env=PlanarPushEnv(config);snapshots={}
    for case in cases:
        env.reset(InitialSpec(**case['initial_spec']));snapshots[case['episode_id']]=env.snapshot()
    sources=source_bundle(out)
    for p in (args.settings,ROOT/'configs/p3_geometry_selected.v2.json',ROOT/'configs/p3_geometry_transition.v2.json'):
        dest=out/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);sources[str(p.relative_to(ROOT))]=digest(p)
    manifest={'task_id':'P3-05','created_at':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),'argv':sys.argv,
              'python':platform.python_version(),'platform':platform.platform(),'settings':settings,
              'settings_sha256':canonical_hash(settings),'source_hashes':sources,'cases':cases,
              'snapshots':_encode_nonfinite({k:asdict(v) for k,v in snapshots.items()}),
              'physical_config':config,'upstream_artifacts':prior,'normalizer_hash':m['normalizer_hash'],
              'model_hashes':{k:v.metadata.get('checkpoint_hash',v.metadata.get('ridge_hash')) for k,v in models.items()},
              'final_test_used':False,'user_understanding':'NOT_ASSESSED'}
    save(out/'manifest.json',manifest);manifest_sha=digest(out/'manifest.json')
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,capture_output=True,text=True)
    log=tests.stdout+tests.stderr;(out/'tests.txt').write_text(log);match=re.search(r'Ran (\d+) tests',log);n=int(match.group(1)) if match else 0
    if tests.returncode or not n or 'skipped=' in log:raise ValueError('Mandatory regression failed')
    print(json.dumps({'progress':'regression_complete','tests':n}),flush=True)
    geometry_trials=[]
    for g in settings['geometry_trials']:
        patch={**base_geometry,'position_gain_per_s':g['position_gain_per_s']}
        rows=[run_cell(out/'geometry_trials'/g['id']/case['episode_id'],case,snapshots[case['episode_id']],
                       'G',None,normalizer,config,config,patch,transition,settings,intervals=settings['screen_intervals']) for case in cases]
        trial={'id':g['id'],'parameters':patch,'rows':rows,'score':trial_score(rows,[f'G/{c["episode_id"]}' for c in cases])}
        geometry_trials.append(trial);save(out/'geometry_trials'/g['id']/'summary.json',trial)
    chosen_g=select_trial(geometry_trials);save(out/'geometry_selection.json',chosen_g)
    planning_trials=[]
    for p in settings['planning_trials']:
        planner=resolved_planner(config,p,settings);rows=[]
        for model_id,bundle in models.items():
            for case in cases:
                rows.append(run_cell(out/'planning_trials'/p['id']/model_id/case['episode_id'],case,snapshots[case['episode_id']],
                                     model_id,bundle,normalizer,config,planner,chosen_g['parameters'],transition,settings,intervals=settings['screen_intervals']))
        trial={'id':p['id'],'planner':planner,'rows':rows,'score':trial_score(rows,[f'{model}/{c["episode_id"]}' for model in models for c in cases])}
        planning_trials.append(trial);save(out/'planning_trials'/p['id']/'summary.json',trial)
    chosen_p=select_trial(planning_trials);save(out/'planning_selection.json',chosen_p)
    resource_rows=[];case=allcases[settings['full_resource_case_index']]
    snapshot=snapshots[case['episode_id']]
    for model_id,bundle in {'G':None,**models}.items():
        resource_rows.append(run_cell(out/'full_resource'/model_id,case,snapshot,model_id,bundle,normalizer,config,chosen_p['planner'],
                            chosen_g['parameters'],transition,settings,intervals=settings['full_resource_max_intervals'],resource_case=True))
    probes=training_cost_probe(out/'training_cost_probe',prepared,normalizer,config,mainmanifest['provenance'],
                               mainmanifest['settings'],main['seeds'][0],settings)
    cells=[r for t in geometry_trials+planning_trials for r in t['rows']]+resource_rows
    hashes_shared=all(len({r['first_candidate_hash'] for r in t['rows'] if r['case_id']==c['episode_id']})==1 for t in planning_trials for c in cases)
    checks={'regression_tests_pass':True,'all_30_screen_cells_and_5_resource_cells':len(cells)==35,
            'all_initial_full_snapshots_exact':all(r['initial_full_snapshot_equal'] for r in cells),
            'same_first_candidates_all_four_models':hashes_shared,
            'every_first_frozen_plan_replays_exactly':all(r['first_frozen_plan_replay_pass'] for r in cells),
            'all_model_weights_unchanged':all(r['model_weights_unchanged'] for r in cells),
            'every_compressed_raw_roundtrip_exact':all(r['compressed_raw_roundtrip_exact'] for r in cells),
            'future_engine_calls_zero':all(sum(r['future_engine_calls'].values())==0 for r in cells),
            'no_unresolved_physics_or_infrastructure':all(r['execution_status']=='completed' and r['termination'] not in settings['unresolved_physics_codes'] for r in cells),
            'both_geometry_and_common_planner_selected':chosen_g['score']['eligible'] and chosen_p['score']['eligible'],
            'two_disposable_training_cost_probes_complete':len(probes)==2 and all(p['updates']==settings['resource_training_probe_updates'] and p['weights_changed'] for p in probes),
            'upstream_raw_stats_models_unchanged':all(verify_artifacts(p)==prior[str(p.relative_to(ROOT))] for p in upstream),
            'sources_and_manifest_frozen':digest(out/'manifest.json')==manifest_sha and all(digest(ROOT/p)==sha for p,sha in sources.items()),
            'final_test_not_used':settings['final_test_used'] is False}
    summary={'task_id':'P3-05','execution_checks_pass':all(checks.values()),'checks':checks,'test_count':n,
             'geometry_selection':{'id':chosen_g['id'],'parameters':chosen_g['parameters'],'score':chosen_g['score']},
             'planning_selection':{'id':chosen_p['id'],'planner':chosen_p['planner'],'score':chosen_p['score']},
             'geometry_trial_scores':[{k:t[k] for k in ('id','score')} for t in geometry_trials],
             'planning_trial_scores':[{k:t[k] for k in ('id','score')} for t in planning_trials],
             'full_resource_rows':resource_rows,'screen_count':30,'resource_count':5,
             'resources':{'total_wall_time_s':perf_counter()-clock,'main_peak_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024),
                          'stored_bytes_before_final_index':sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
                          'training_cost_probe':probes,'scope':'sequential local CPU; main cumulative peak only, regression child excluded; BLAS threads unmeasured'},
             'stage_verdict':'PENDING_SCOPE_FREEZE','final_test_used':False,'formal_S1_SK':'NOT_RUN','user_understanding':'NOT_ASSESSED'}
    save(out/'summary.json',summary)
    if not summary['execution_checks_pass']:raise ValueError('P3-05 execution preparation checks incomplete')
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True)
    p.add_argument('--settings',type=Path,default=ROOT/'configs/p3_planning_trials.v1.json');args=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id):raise ValueError('Unsafe run ID')
    out=ROOT/'runs'/args.run_id;out.mkdir(parents=True,exist_ok=False)
    try:run(args,out)
    except Exception as error:
        save(out/'failure.json',{'task_id':'P3-05','type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()});raise
    finally:
        save(out/'artifact_hashes.json',{'sha256':{str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'}})


if __name__=='__main__':main()
