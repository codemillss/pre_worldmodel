"""P4-03: frozen development inputs, model-only choice, shared physical labels.

No fitting, retuning, final test or closed-loop alignment experiment. All
model choices are stored before any candidate future is physically executed.
"""
from __future__ import annotations

import argparse
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
import shutil
import subprocess
import sys
from time import perf_counter
import tomllib
import traceback
import unittest
from zoneinfo import ZoneInfo

import numpy as np
import torch

from push_align.candidate_diagnostics import compare_candidates
from push_align.candidate_replay import replay_candidate, snapshots_exact
from push_align.candidates import generate_candidates
from push_align.closed_loop import ENGINE_GUARDS, model_planning_scope
from push_align.config import ROOT
from push_align.data import _encode_nonfinite, digest
from push_align.fork_checkpoint import load_fork_checkpoint
from push_align.learning_data import load_learning_data
from push_align.planning import LearnedPredictor, choose, rollout, score
from push_align.planning_protocol import GeometryProposalAdapter, keyed_planner_rng
from push_align.ridge import load_ridge
from p2_collect_pilot import save_trace
from p2_train_pilot import save as strict_save
from p3_planning_settings import canonical_hash
from p3_train_ridge import array_hash
from p4_development_inputs import load_development_snapshot
from p4_train_forks import all_upstream, byte_count, tree_hash
from p4_verify_sequence_loss import snapshot_sources, inventory

REGISTRY = ROOT/'configs/p4_development_e1.v1.json'
REGISTRY_SHA = '1a66021cd059d7e33afa7d04e6344a31bf08b950484bd9cd12f7b871463fa50f'
FORKS = ROOT/'runs/p4-02-20261012-01'
RIDGE = ROOT/'runs/p3-03-20261009-01'
LEARNING = ROOT/'runs/p3-02-20261009-01/learning_data'
GIB = 1024**3


def save(path,value):
    """Preserve NumPy scalar flags as ordinary JSON, retaining NPZ raw values."""
    strict_save(path,_encode_nonfinite(value))


def validate_test_inventory(names,prior_log):
    # A known passing test emits a warning between its ID and the final "ok".
    # The immutable prior summary confirms success; preserve every qualified ID.
    prior=re.findall(r'^test_\S+ \(([^)]+)\)',prior_log,re.MULTILINE)
    expected={'test_p4_candidate_replay':17,'test_p4_candidate_diagnostics':31,'test_p4_development_inputs':7}
    counts={module:sum(name.split('.')[0]==module for name in names) for module in expected}
    if len(prior)!=714 or len(set(prior))!=714 or not set(prior)<=set(names) or counts!=expected or len(set(names))!=len(names):
        raise ValueError('Inherited714 test IDs or required17/31/7 new module populations changed')
    return {'inherited_test_count':714,'required_new_module_counts':counts,'all_inherited_ids_preserved':True}


def guard(out):
    stored, free = byte_count(out), shutil.disk_usage(ROOT).free
    if stored > GIB or free < 2*GIB:
        raise RuntimeError('Stage storage/reserve reached; preserve prefix and reestimate')
    return {'stored_bytes':stored,'free_bytes':free}


def upstream():
    from p3_geometry_pilot import verify_artifacts
    result = all_upstream()
    result[str(FORKS.relative_to(ROOT))] = {'registered_artifacts':verify_artifacts(FORKS),'file_inventory':inventory(FORKS)}
    return result


def tests_in(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):yield from tests_in(test)
        else:yield test.id()


def load_models(normalizer, physical, registry):
    summary = json.loads((FORKS/'summary.json').read_text())
    manifest = json.loads((FORKS/'manifest.json').read_text())
    if not summary['pass'] or not all(summary['checks'].values()) or summary['lambda_selection']['selected_lambda'] != registry['selected_lambda']:
        raise ValueError('P4-02 matching and global lambda prerequisite changed')
    budget = json.loads((ROOT/'configs/p4_fork_budget.v2.json').read_text())
    ridge_selection = json.loads((RIDGE/'selection.json').read_text())['selected']
    ridge_manifest = json.loads((RIDGE/'manifest.json').read_text())
    models = {'R':load_ridge(RIDGE/ridge_selection['model_path'],normalizer,physical,ridge_manifest['provenance'])}
    identities = {'R':{'path':str((RIDGE/ridge_selection['model_path']).relative_to(ROOT)),
        'model_hash':models['R'].metadata['ridge_hash']}}
    for kind in ('S1','SK'):
        for seed in (11,23,47):
            branch = kind if kind=='S1' else f"SK-lambda{registry['selected_lambda']}"
            row = next(r for r in summary['branches'] if r['seed']==seed and r['branch']==branch)
            path = FORKS/row['final_checkpoint']
            bundle = load_fork_checkpoint(path,normalizer,physical,manifest['provenance'],budget,
                seed=seed,branch=branch,expected_checkpoint_hash=row['final_checkpoint_hash'],
                expected_run_manifest_sha256=digest(FORKS/'manifest.json'))
            models[f'{kind}-{seed}'] = bundle
            identities[f'{kind}-{seed}'] = {'path':str(path.relative_to(ROOT)),
                'model_hash':bundle.metadata['checkpoint_hash'],'additional_updates':bundle.metadata['additional_updates'],
                'total_updates':bundle.metadata['total_updates'],'lambda':bundle.metadata['rollout_weight']}
    if list(models)!=registry['models']:raise ValueError('Frozen seven-model population differs')
    return models,identities


def prediction_trace(folder,predictions,costs,candidates):
    folder.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(folder/'trace.npz',predicted_states=predictions.states,
        raw_deltas=predictions.raw_deltas,valid=predictions.valid,reasons=np.array(predictions.reasons),
        failure_steps=predictions.failure_steps,candidates=candidates.actions,total_cost=costs.total,
        **{'cost_'+k:v for k,v in costs.components.items()})


def endpoint_errors(predicted,actual,mask):
    """Conditional same-candidate endpoint physical errors; 2pi model yaw."""
    rows = np.flatnonzero(mask)
    if not len(rows):return {'denominator':0,'XY_mean_m':None,'yaw_absolute_mean_rad':None}
    error = predicted[rows,-1]-actual[rows,0,-1]
    angle = np.arctan2(np.sin(error[:,2]),np.cos(error[:,2]))
    return {'denominator':len(rows),'XY_mean_m':float(np.hypot(error[:,0],error[:,1]).mean()),
        'yaw_absolute_mean_rad':float(np.abs(angle).mean()),'actual_reference':'repeat0, only stable valid candidates; repeats equality reported separately'}


def run(args,out):
    began=perf_counter();free_before=shutil.disk_usage(ROOT).free
    if free_before < 3*GIB:raise RuntimeError('Need stage1GiB plus free reserve2GiB')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if digest(REGISTRY)!=REGISTRY_SHA:raise ValueError('Predeclared development inputs changed')
    registry=json.loads(REGISTRY.read_text())
    for path,sha in registry['source_files_sha256'].items():
        if digest(ROOT/path)!=sha:raise ValueError('Predeclared existing source changed')
    for path,key in [('configs/p3_planning.v1.json','planning_protocol_sha256'),
            ('configs/p3_evaluation_scope.v3.json','scope_v3_sha256'),
            ('configs/acceptance.v1.toml','profile_sha256'),
            ('runs/p4-02-20261012-01/summary.json','prerequisite_P4_02_summary_sha256')]:
        if digest(ROOT/path)!=registry[key]:raise ValueError('Frozen prerequisite/contract changed')
    before=upstream();save(out/'upstream_before.json',before)
    protocol=json.loads((ROOT/'configs/p3_planning.v1.json').read_text())
    profile=tomllib.loads((ROOT/'configs/acceptance.v1.toml').read_text())
    criteria=profile['counterfactual_check']
    if (criteria['replays_per_candidate']!=registry['physical_repeats'] or
            criteria['cost_padding_atol']!=registry['cost_padding_atol'] or
            criteria['cost_padding_rtol']!=registry['cost_padding_rtol']):raise ValueError('Cost replay criterion changed')
    nm=json.loads((LEARNING/'normalizer/normalizer.json').read_text())['normalizer']
    prepared,normalizer=load_learning_data(LEARNING,nm['schema'],nm['dataset_identity'],nm['normalizer_hash'])
    physical=prepared.metadata['contract']['resolved_config'];del prepared
    planner=deepcopy(physical);planner['planner']=deepcopy(protocol['planner']);planner['cost']=deepcopy(protocol['cost'])
    planner['planner']['candidate_count']=registry['candidates']
    if planner['planner']['horizon']!=registry['horizon'] or planner['task']!=protocol['task']:
        raise ValueError('Horizon/task differs from frozen common planning')
    models,identities=load_models(normalizer,physical,registry)
    weights_before={name:tree_hash(b.model.state_dict()) for name,b in models.items()}
    sources=snapshot_sources(out)
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    test_names=list(tests_in(suite));new_names=[n for n in test_names if any(p in n for p in
        ('test_p4_candidate_replay','test_p4_candidate_diagnostics','test_p4_development_inputs','test_p4_candidate_runner'))]
    test_population=validate_test_inventory(test_names,(FORKS/'tests.txt').read_text())
    manifest={'task_id':'P4-03','created_at':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        'registry_sha256':REGISTRY_SHA,'registry':registry,'source_hashes':sources,
        'physical_contract':physical,'diagnostic_planner_config':planner,'model_identities':identities,
        'normalizer_hash':normalizer.metadata['normalizer_hash'],'stats_hash':normalizer.metadata['stats_hash'],
        'test_ids':test_names,'new_test_ids':new_names,'test_population':test_population,'python':platform.python_version(),
        'torch':str(torch.__version__),'numpy':str(np.__version__),'platform':platform.platform(),
        'torch_threads':1,'deterministic_algorithms':True,'free_before_bytes':free_before,
        'selection_order':'all42 model choices stored before first physical candidate future',
        'final_test_used':False,'learning_updates':0,'new_closed_loop_alignment':'NOT_RUN',
        'personal_understanding':'NOT_ASSESSED'}
    save(out/'manifest.json',manifest);manifest_sha=digest(out/'manifest.json')
    print(json.dumps({'progress':'regressions_started','registered_tests':len(test_names)}),flush=True)
    process=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,
        capture_output=True,text=True,env={**__import__('os').environ,'PYTHONDONTWRITEBYTECODE':'1'})
    log=process.stdout+process.stderr;(out/'tests.txt').write_text(log)
    found=re.search(r'Ran (\d+) tests',log)
    executed_ids=re.findall(r'^test_\S+ \(([^)]+)\)',log,re.MULTILINE)
    if process.returncode or not found or int(found.group(1))!=len(test_names) or set(executed_ids)!=set(test_names) or len(executed_ids)!=len(test_names) or 'skipped=' in log or ' ... skipped ' in log:
        raise ValueError('Full frozen regression inventory failed or skipped')
    save(out/'regression_result.json',{'pass':True,'test_count':len(test_names),'new_test_count':len(new_names),'no_skips':True})
    print(json.dumps({'progress':'regressions_complete','tests':len(test_names),'new_tests':len(new_names)}),flush=True)
    guards=dict.fromkeys(ENGINE_GUARDS,0);inputs=[];frozen_choices=[];prediction_hashes={}
    source_prefix_intervals=0;prediction_seconds=0.
    # Model choices for every input are frozen before actual candidate futures.
    for spec in registry['snapshots']:
        data=load_development_snapshot(ROOT/spec['source_folder'],spec['source_tick'],physical,profile)
        if data.proof['source_regime']!=spec['source_regime']:raise ValueError('Source regime selection changed')
        source_prefix_intervals+=data.proof['prefix_restore_intervals']
        folder=out/'snapshots'/spec['id'];folder.mkdir(parents=True,exist_ok=False)
        save(folder/'physical_snapshot.json',_encode_nonfinite(asdict(data.snapshot)))
        save(folder/'input.json',{'spec':spec,'observed_state':data.state.tolist(),'task':_encode_nonfinite(asdict(data.task)),
            'task_status':asdict(data.status),'planner_context':_encode_nonfinite(asdict(data.context)),
            'geometry_context':asdict(data.gctx),'proof':data.proof,'source_files_sha256':data.source_files_sha256})
        input_hash=canonical_hash(_encode_nonfinite({'state':data.state.tolist(),'snapshot':asdict(data.snapshot),
            'task':asdict(data.task),'status':asdict(data.status),'planner':asdict(data.context),'G':asdict(data.gctx)}))
        adapter=GeometryProposalAdapter(physical,protocol['geometry_selected_overrides'],protocol['geometry_transition'])
        with model_planning_scope(guards):
            proposal=adapter.propose(data.state.copy(),data.task,data.gctx,data.status.elapsed_s,control_tick=spec['source_tick'])
            if proposal.diagnostics!=data.original_plan['geometry_proposal'] or asdict(proposal.context)!=data.original_plan['geometry_context_after']:
                raise ValueError('Shared current-state G proposal differs from original context')
            rng=keyed_planner_rng(registry['candidate_rng_base'],data.case['episode_index'],spec['source_tick'],registry['candidate_stream_id'])
            before_rng=deepcopy(rng.bit_generator.state)
            candidates=generate_candidates(planner,rng,previous_actions=data.context.optimal_actions,
                geometry_action=proposal.geometry_action,tangent_direction=proposal.tangent_direction)
            independent=generate_candidates(planner,keyed_planner_rng(registry['candidate_rng_base'],data.case['episode_index'],spec['source_tick'],registry['candidate_stream_id']),
                previous_actions=data.context.optimal_actions,geometry_action=proposal.geometry_action,tangent_direction=proposal.tangent_direction)
            if not np.array_equal(candidates.actions,independent.actions):raise ValueError('Candidate replay differs')
            if candidates.diagnostics['random_group_counts']!={'global':4,'local_geometry':2,'local_warm':2}:
                raise ValueError('N16 groups differ from registered diagnostic set')
            np.savez_compressed(folder/'candidates.npz',actions=candidates.actions,sources=np.array(candidates.sources))
            save(folder/'candidate_identity.json',{'input_hash':input_hash,'candidate_hash':array_hash(candidates.actions),
                'rng_before':before_rng,'rng_after':rng.bit_generator.state,'geometry_proposal':proposal.diagnostics,
                'geometry_context_after':asdict(proposal.context),'generation':candidates.diagnostics})
            entries=[]
            for name,bundle in models.items():
                predictor=LearnedPredictor(bundle.model,normalizer);tick=perf_counter()
                pred=rollout(np.broadcast_to(data.state,(16,10)).copy(),candidates.actions,predictor,planner)
                costs=score(pred,candidates.actions,data.task,data.context.previous_applied_action,planner)
                index,action,reason=choose(costs,pred,candidates.actions)
                elapsed=perf_counter()-tick;prediction_seconds+=elapsed
                dest=folder/'models'/name;prediction_trace(dest,pred,costs,candidates)
                choice={'snapshot_id':spec['id'],'model':name,'selected_index':index,'selected_action':action.tolist(),
                    'reason':reason,'input_hash':input_hash,'candidate_hash':array_hash(candidates.actions),
                    'prediction_hash':array_hash(pred.states),'cost_hash':array_hash(costs.total),
                    'previous_applied_action_hash':array_hash(data.context.previous_applied_action),
                    'prediction_seconds':elapsed,'predictor_calls':predictor.calls}
                save(dest/'choice_before_physics.json',choice);frozen_choices.append(choice)
                prediction_hashes[str((dest/'trace.npz').relative_to(out))]=digest(dest/'trace.npz')
                entries.append((name,pred,costs,choice))
        inputs.append((spec,data,candidates,entries,input_hash))
    if len(frozen_choices)!=42 or source_prefix_intervals!=registry['source_prefix_restore_intervals']:
        raise ValueError('Input/choice population or recorded prefix restoration differs')
    save(out/'choices_before_any_physical_future.json',frozen_choices)
    choice_file_sha=digest(out/'choices_before_any_physical_future.json')
    save(out/'prediction_files_before_any_physical_future.json',prediction_hashes)
    guard(out);print(json.dumps({'progress':'all42_choices_frozen','candidate_future_intervals':0,'source_prefix_restoration_intervals':source_prefix_intervals}),flush=True)
    ledger=[];snapshots=[];physical_s=raw_save_s=0.;replay_attempts=actual_attempted=completed=0
    for spec,data,candidates,entries,input_hash in inputs:
        folder=out/'snapshots'/spec['id'];actual_states=np.full((16,3,6,10),np.nan,np.float64)
        actual_actions=np.full((16,3,5,2),np.nan,np.float64);actual_cost=np.full((16,3),np.inf,np.float64)
        actual_valid=np.zeros((16,3),bool);replays=[];final_hashes=[]
        component_names=('stage_position','stage_yaw','stage_action','stage_action_change','stage_gap','terminal_position','terminal_yaw','terminal_speed','terminal_angular_speed')
        component_values={key:np.full((16,3),np.nan,np.float64) for key in component_names}
        for i,actions in enumerate(candidates.actions):
            cand_finals=[]
            for repeat in range(3):
                target=folder/'physical'/f'candidate-{i:02d}'/f'repeat-{repeat}';target.mkdir(parents=True,exist_ok=False)
                tick=perf_counter();replay_attempts+=1
                try:
                    result=replay_candidate(physical,planner,data.snapshot,actions.copy(),data.task,data.context.previous_applied_action,profile)
                except Exception as error:
                    save(target/'infrastructure_error.json',{'candidate':i,'repeat':repeat,'error':str(error),'traceback':traceback.format_exc()})
                    raise
                physical_s+=perf_counter()-tick;actual_attempted+=len(result.results);completed+=result.completed_intervals
                actual_states[i,repeat]=result.states;actual_actions[i,repeat]=result.applied_actions
                actual_cost[i,repeat]=result.cost;actual_valid[i,repeat]=result.valid
                for key in component_names:component_values[key][i,repeat]=result.cost_components[key]
                tick=perf_counter();save_trace(target,'raw_execution',data.snapshot,result.results);raw_save_s+=perf_counter()-tick
                final=canonical_hash(_encode_nonfinite(asdict(result.final_snapshot)));cand_finals.append(final)
                row={'candidate':i,'repeat':repeat,'physical_valid':result.valid,'cost':None if not result.valid else result.cost,
                    'cost_components':_encode_nonfinite(result.cost_components),'attempted_intervals':len(result.results),
                    'completed_intervals':result.completed_intervals,'initial_snapshot_exact':result.initial_snapshot_exact,
                    'final_snapshot_hash':final,'reasons':list(result.reasons),'state_hash':array_hash(result.states),
                    'applied_action_hash':array_hash(result.applied_actions)}
                save(target/'summary.json',row);replays.append(row);del result;gc.collect()
            final_hashes.append(cand_finals)
            guard(out)
        exact=all(np.array_equal(actual_states[i,0],actual_states[i,r],equal_nan=True)
            and np.array_equal(actual_actions[i,0],actual_actions[i,r],equal_nan=True)
            and actual_cost[i,0]==actual_cost[i,r] and actual_valid[i,0]==actual_valid[i,r]
            and final_hashes[i][0]==final_hashes[i][r] for i in range(16) for r in (1,2))
        np.savez_compressed(folder/'actual_futures.npz',states=actual_states,applied_actions=actual_actions,
            total_cost=actual_cost,valid=actual_valid,**{'cost_'+k:v for k,v in component_values.items()})
        save(folder/'physical_replays.json',{'attempts':replays,'all3_repeats_exact':exact})
        stable=actual_valid.all(axis=1);common=stable.copy()
        for _,pred,costs,_ in entries:common &= pred.valid & np.isfinite(costs.total)
        snap_rows=[]
        for name,pred,costs,choice in entries:
            eligible=pred.valid & np.isfinite(costs.total)
            diagnostic=compare_candidates(costs.total.copy(),eligible,actual_cost.copy(),actual_valid.copy(),
                atol=criteria['cost_padding_atol'],rtol=criteria['cost_padding_rtol'])
            if diagnostic['selected_index']!=choice['selected_index']:raise ValueError('Physical labels changed model choice')
            diagnostic.update({'snapshot_id':spec['id'],'model':name,'source_regime':spec['source_regime'],
                'source_case':data.case['episode_id'],'source_initial_conditions':2,
                'all3_physical_repeats_exact':exact,'input_hash':input_hash,'candidate_hash':choice['candidate_hash'],
                'trajectory_valid_candidate_count':int(pred.valid.sum()),
                'trajectory_valid_but_cost_nonfinite_indices':np.flatnonzero(pred.valid & ~np.isfinite(costs.total)).tolist(),
                'conditional_endpoint_errors':endpoint_errors(pred.states,actual_states,stable & pred.valid & np.isfinite(costs.total)),
                'all7_common_valid_endpoint_errors':endpoint_errors(pred.states,actual_states,common),
                'all7_common_valid_cost_error':{'denominator':int(common.sum()),'MAE':float(np.abs(costs.total[common]-actual_cost[common].mean(axis=1)).mean()) if common.any() else None}})
            save(folder/'models'/name/'diagnostics.json',diagnostic);ledger.append(diagnostic);snap_rows.append(diagnostic)
        snapshot_summary={'snapshot_id':spec['id'],'source_case':data.case['episode_id'],'source_regime':spec['source_regime'],
            'candidate_count':16,'physical_replays':48,'actual_valid_candidates':int(stable.sum()),
            'all7_common_valid':int(common.sum()),'all3_repeats_exact':exact,
            'any_validity_changed':bool((actual_valid.any(axis=1)!=stable).any()),'models':snap_rows}
        save(folder/'summary.json',snapshot_summary);snapshots.append(snapshot_summary)
        print(json.dumps({'progress':'snapshot_complete','snapshot':spec['id'],'actual_valid':int(stable.sum()),'all3_exact':exact,'stored_bytes':byte_count(out)}),flush=True)
    save(out/'comparison_ledger.json',ledger)
    aggregates=[]
    for name in models:
        rows=[r for r in ledger if r['model']==name];losses=[r['selection_loss'] for r in rows if r['selection_loss'] is not None]
        aggregates.append({'model':name,'snapshot_attempts':len(rows),'conditional_finite_loss_count':len(losses),
            'conditional_mean_selection_loss':float(np.mean(losses)) if losses else None,
            'model_all_invalid_snapshots':sum(r['model_all_invalid'] for r in rows),
            'actual_all_invalid_snapshots':sum(r['actual_all_invalid'] for r in rows),
            'selected_actual_invalid_snapshots':sum(r['selected_actual_status']=='STABLE_INVALID' for r in rows),
            'selected_actual_invalid_with_valid_alternative':sum(r['selected_actual_status']=='STABLE_INVALID' and r['counts']['actual_stable_valid']>0 for r in rows),
            'candidate_count':sum(r['counts']['candidate_count'] for r in rows),
            'predicted_eligible_candidate_count':sum(r['counts']['predicted_valid'] for r in rows),
            'false_valid_candidate_count':sum(r['counts']['false_valid'] for r in rows),
            'false_invalid_candidate_count':sum(r['counts']['false_invalid'] for r in rows),
            'unstable_actual_candidate_count':sum(r['counts']['actual_validity_changed'] for r in rows),
            'rank_pairs_actual_inconclusive':sum(r['pairwise']['actual_inconclusive'] for r in rows),
            'rank_pairs_common_valid':sum(r['pairwise']['common_valid_pairs'] for r in rows),
            'rank_pairs_all_candidates':sum(r['pairwise']['total_pairs'] for r in rows),
            'rank_pairs_excluded_noncommon':sum(r['pairwise']['excluded_noncommon_pairs'] for r in rows),
            'rank_pairs_predicted_tie':sum(r['pairwise']['predicted_tie'] for r in rows),
            'rank_pairs_concordant':sum(r['pairwise']['concordant'] for r in rows),
            'rank_pairs_discordant':sum(r['pairwise']['discordant'] for r in rows),
            'rank_pairs_strict_comparison':sum(r['pairwise']['strict_comparison_pairs'] for r in rows),
            'selected_cost_ordering_inconclusive_snapshots':sum(r['selection_loss_ordering']=='INCONCLUSIVE' for r in rows),
            'scope':'descriptive six related snapshots from two development initialconditions; not independent trial success or final test'})
    save(out/'model_aggregate.json',aggregates)
    after=upstream();save(out/'upstream_after.json',after);store=guard(out)
    checks={'entire_frozen_regression_inventory_passed':True,'six_declared_snapshot_inputs_from_two_original_cases':len(inputs)==6,
        'all42_model_selections_frozen_before_candidate_physics':len(frozen_choices)==42 and digest(out/'choices_before_any_physical_future.json')==choice_file_sha,
        'same_original_context_and_N16_candidates_in_every_model':all(len({r['input_hash'] for r in s['models']})==1 and len({r['candidate_hash'] for r in s['models']})==1 for s in snapshots),
        'all96_candidates_times3_physical_attempts':replay_attempts==288 and len(snapshots)==6,
        'no_candidate_future_budget_exceeded':actual_attempted<=1440 and completed<=1440,
        'all3_repeats_complete_snapshot_states_actions_cost_exact':all(s['all3_repeats_exact'] for s in snapshots),
        'no_repeat_validity_changes':not any(s['any_validity_changed'] for s in snapshots),
        'all42_diagnostics_include_invalid_and_rank_denominators':len(ledger)==42 and all(r['counts']['candidate_count']==16 and r['pairwise']['total_pairs']==120 for r in ledger),
        'predictions_and_choices_unchanged_after_actual_labels':all(digest(out/path)==sha for path,sha in prediction_hashes.items()),
        'future_engine_calls_during_model_prediction_zero':sum(guards.values())==0,
        'all_model_weights_original_statistics_and_upstream_unchanged':weights_before=={n:tree_hash(b.model.state_dict()) for n,b in models.items()} and before==after,
        'training_sources_and_manifest_frozen':digest(out/'manifest.json')==manifest_sha and all(digest(ROOT/p)==sha and digest(out/'source'/p)==sha for p,sha in sources.items()),
        'within1GiB_stage_and2GiB_free_reserve':store['stored_bytes']<=GIB and store['free_bytes']>=2*GIB,
        'no_new_alignment_test_or_learning':True}
    resources={'total_wall_time_s':perf_counter()-began,'prediction_scoring_s':prediction_seconds,'physical_candidate_replay_s':physical_s,
        'raw_save_s':raw_save_s,'peak_main_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024),
        'peak_regression_child_rss_MiB':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024),
        'stored_bytes_before_summary_index':store['stored_bytes'],'free_before_bytes':free_before,'free_after_bytes':store['free_bytes'],
        'candidate_replays_attempted':replay_attempts,'candidate_intervals_attempted':actual_attempted,'candidate_intervals_completed':completed,
        'source_prefix_restoration_intervals':source_prefix_intervals,'learning_updates':0,'additional_spend_KRW':0,
        'scope':'local CPU Torch1thread; raw candidate prefixes preserved; observed peak sums are not simultaneous RSS'}
    summary={'task_id':'P4-03','pass':all(checks.values()),'stage_verdict':'PASS' if all(checks.values()) else 'FAIL',
        'checks':checks,'test_count':len(test_names),'new_test_count':len(new_names),'snapshots':snapshots,
        'model_aggregates':aggregates,'resources':resources,'future_engine_counters':guards,
        'final_test_used':False,'new_closed_loop_alignment':'NOT_RUN','personal_understanding':'NOT_ASSESSED',
        'next_task':'P4-04','scope':'offline N16/H5 development counterfactual; actual minimum only within this finite set; related snapshots from2 reused cases; not closed-loop alignment, independent generalization or personal understanding'}
    save(out/'resources.json',resources);save(out/'summary.json',summary)
    if not summary['pass']:raise ValueError('P4-03 contract/repeat/evidence incomplete')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True);args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id):raise ValueError('Unsafe run ID')
    out=ROOT/'runs'/args.run_id;out.mkdir(parents=True,exist_ok=False)
    try:
        result=run(args,out)
    except Exception as error:
        save(out/'failure.json',{'task_id':'P4-03','pass':False,'error':str(error),'traceback':traceback.format_exc()})
        print(json.dumps({'pass':False,'error':str(error),'run':str(out)}),flush=True);return 1
    finally:
        save(out/'artifact_hashes.json',{'sha256':{str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'}})
    print(json.dumps({'pass':result['pass'],'run':str(out)}),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
