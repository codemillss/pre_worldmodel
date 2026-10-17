"""P4-04 actual failure reproduction and preregistered loss-only contrast.

All diagnostic choices precede new physical futures. Counterfactual labels and
teacher-forced current states remain offline; the unchanged online controller
uses only the frozen learned predictor. A completed experiment is distinct
from successful handoff, identified root cause, or general superiority.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import gc
import gzip
import json
from pathlib import Path
import re
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
from push_align.contact_geometry import check_contact_geometry
from push_align.data import digest, _encode_nonfinite
from push_align.planning import LearnedPredictor, rollout, score, choose
from push_align.planning_protocol import GeometryProposalAdapter, keyed_planner_rng
from push_align.learning_data import load_learning_data
from push_align.preparation import SPLIT_CODES
from push_align.task import handoff_metrics
from p2_collect_pilot import save_trace
from p3_geometry_pilot import verify_artifacts
from p3_planning_settings import run_cell, canonical_hash
from p3_train_common_mlp import load_common_checkpoint
from p3_train_ridge import array_hash
from p4_compare_candidates import save, load_models, prediction_trace, tests_in, upstream
from p4_development_inputs import load_development_snapshot
from p4_train_forks import byte_count, tree_hash
from p4_verify_sequence_loss import snapshot_sources, inventory

REGISTRY = ROOT/'configs/p4_failure_analysis.v1.json'
REGISTRY_SHA = 'e88cf4aead154ec71a532dc9f1be8d005e9ddebfd07065f7d4f01bba2ea07cef'
SOURCE = ROOT/'runs/p3-05-20261010-01/full_resource/MLP-11'
LEARNING = ROOT/'runs/p3-02-20261009-01/learning_data'
COMMON = ROOT/'runs/p3-04-20261009-01'
GIB = 1024**3


def select_diagnostic_subset(selected_indices, candidate_count=128, size=16, reserved_count=8):
    for name, value in [('candidate_count',candidate_count),('size',size),('reserved_count',reserved_count)]:
        if type(value) is not int or value < 1: raise ValueError(f'Invalid {name}')
    if not reserved_count <= size <= candidate_count: raise ValueError('Subset population inconsistent')
    picked = set(range(reserved_count))
    for index in selected_indices:
        if index is None: continue
        if type(index) is not int or not 0 <= index < candidate_count: raise ValueError('Invalid selected index')
        picked.add(index)
    if len(picked) > size: raise ValueError('Required choices exceed declared subset')
    for index in range(candidate_count):
        if len(picked) == size: break
        picked.add(index)
    return sorted(picked)


def compare_closed_loop_reproduction(source_folder, replay_folder):
    """Require raw and decision equality; wall-clock/provenance labels can differ."""
    source_folder, replay_folder = Path(source_folder), Path(replay_folder)
    checks = {}
    with np.load(source_folder/'raw_execution.npz',allow_pickle=False) as a, np.load(replay_folder/'raw_execution.npz',allow_pickle=False) as b:
        checks['raw_array_fields'] = set(a.files)==set(b.files)
        for key in a.files:
            checks['raw_'+key] = key in b and a[key].dtype==b[key].dtype and np.array_equal(a[key],b[key],equal_nan=True)
    def raw(folder):
        with gzip.open(folder/'raw_execution.json.gz','rt') as f: return json.load(f)
    a,b=raw(source_folder),raw(replay_folder)
    for key in ('initial_snapshot','diagnostics','diagnostic_nonfinite_paths'):
        checks['raw_json_'+key] = a.get(key)==b.get(key)
    def steps(folder): return [json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
    a,b=steps(source_folder),steps(replay_folder)
    checks['all_actual_tick_count'] = len(a)==len(b) and len(a)>0
    step_keys=('index','source','requested_action','applied_action','state_before','state_after','actual_dt_s','physical_valid','geometry','task','G_proposal','G_context_committed')
    plan_keys=('selected_index','reason','candidate_hash','prediction_hash','delta_hash','selected_cost','selected_cost_components','selected_actions','planner_context_before','tick')
    for key in step_keys: checks['actual_'+key] = len(a)==len(b) and all(x.get(key)==y.get(key) for x,y in zip(a,b))
    for key in plan_keys:
        checks['plan_'+key] = len(a)==len(b) and all((x['plan'] is None and y['plan'] is None) or (isinstance(x['plan'],dict) and isinstance(y['plan'],dict) and x['plan'].get(key)==y['plan'].get(key)) for x,y in zip(a,b))
    a=json.loads((source_folder/'summary.json').read_text());b=json.loads((replay_folder/'summary.json').read_text())
    for key in ('execution_status','termination','task','final_metrics','transitions','valid_transitions','holding_intervals','first_candidate_hash'):
        checks['summary_'+key]=a.get(key)==b.get(key)
    return {'pass':bool(all(checks.values())),'checks':checks}


def validate_matched_pair(left, right):
    for meta in (left,right):
        for key in ('seed','additional_updates','total_updates'):
            if type(meta.get(key)) is not int:raise ValueError('Strict integer matched counters required')
    equal=('seed','additional_updates','total_updates','common_origin','config','learning_schema','model_spec','normalizer_hash','fork_budget_sha256','training_provenance','run_manifest_sha256')
    checks={key:left.get(key)==right.get(key) and key in left and key in right for key in equal}
    checks['single_loss_change']=(left.get('branch')=='S1' and left.get('rollout_weight')==0. and right.get('branch')=='SK-lambda0.1' and right.get('rollout_weight')==.1)
    checks['declared_seed_updates']=(left.get('seed')==11 and left.get('additional_updates')==1000 and left.get('total_updates')==2300)
    if not all(checks.values()): raise ValueError('Matched S1/SK loss-only prerequisite failed')
    return {'pass':True,'checks':checks,'contrast':'S1 vs SK only; Common vs either also changes additional training'}


def validate_budget(registry, free_bytes, prior_consumed_bytes):
    b=registry['budget']
    if type(free_bytes) is not int or type(prior_consumed_bytes) is not int or min(free_bytes,prior_consumed_bytes)<0:
        raise ValueError('Strict nonnegative observed resources required')
    for key in ('closed_loop_cells','closed_loop_max_intervals','additional_E1_max_intervals','P4_pool_limit_bytes','estimated_stage_bytes','prior_consumed_bytes'):
        if type(b.get(key)) is not int or b[key]<0:raise ValueError('Strict integer resource field required')
    if (registry['source_ticks']!=[49,99] or any(type(t) is not int for t in registry['source_ticks'])
            or any(type(registry[k]) is not int for k in ('N128_horizon','physical_repeats','subset_size'))
            or registry['N128_horizon']!=5 or registry['physical_repeats']!=3 or registry['subset_size']!=16
            or registry['models']!=['MLP-11','S1-11','SK-11'] or prior_consumed_bytes!=b['prior_consumed_bytes']):
        raise ValueError('Preregistered experiment population changed')
    checks={'three_full_cells':b['closed_loop_cells']==3 and b['closed_loop_max_intervals']==900,
        'additional_E1':b['additional_E1_max_intervals']==480,
        'unchanged_1GiB_pool':b['P4_pool_limit_bytes']==GIB,
        'free_reserve':free_bytes>=b['estimated_stage_bytes']+2*GIB,
        'remaining_pool':prior_consumed_bytes+b['estimated_stage_bytes']<=GIB}
    if not all(checks.values()):raise ValueError('P4-04 resource revision does not fit pool/reserve')
    return {'pass':True,'checks':checks,'free_before_bytes':free_bytes,'prior_consumed_bytes':prior_consumed_bytes,**b}


def guard(out, prior):
    if prior+byte_count(out)>GIB or shutil.disk_usage(ROOT).free<2*GIB:
        raise RuntimeError('P4 shared pool/free reserve reached; retain prefix and reestimate')


def upstream_all():
    result=upstream()
    for name in ('p4-03-20261012-01','p4-03-20261012-02'):
        folder=ROOT/'runs'/name
        result[str(folder.relative_to(ROOT))]={'registered_artifacts':verify_artifacts(folder),'file_inventory':inventory(folder)}
    return result


def hypotheses_for_snapshot(data, actions, actual_states, actual_costs, actual_valid, entries, local):
    position_eps=data.task.position_tolerance_m*.001
    initial=handoff_metrics(data.state,data.task)
    metrics=[handoff_metrics(state,data.task) if np.isfinite(state).all() else None for state in actual_states[:,0,-1]]
    pose_progress=[i for i,m in enumerate(metrics) if m and actual_valid[i].all() and m['e_pos_m']<initial['e_pos_m']-position_eps and m['e_yaw_rad']<=initial['e_yaw_rad']+data.task.yaw_tolerance_rad*.001]
    pose_in_goal=[i for i,m in enumerate(metrics) if m and actual_valid[i].all() and m['qualifies']]
    rows=[]
    for name,pred,costs,choice in entries:
        d=compare_candidates(costs.total,pred.valid&np.isfinite(costs.total),actual_costs,actual_valid,atol=1e-6,rtol=1e-5)
        idx=choice['selected_subset_index']; best=d['actual_best_index']
        if idx!=d['selected_index']:raise ValueError('Subset dropped or changed the original model choice')
        rank='INCONCLUSIVE'
        if idx is not None and best is not None and actual_valid[idx].all():
            a=actual_costs[idx];b=actual_costs[best]
            ea=1e-6+1e-5*max(abs(a));eb=1e-6+1e-5*max(abs(b))
            rank='SUPPORTED_WITHIN_SUBSET' if min(a)-ea>max(b)+eb else 'NO_DETECTED_SELECTION_LOSS' if idx==best else 'INCONCLUSIVE'
        traces=local[name]['traces'];error=traces['teacher_next_states']-actual_states[:,0,1:]
        detected_components={'object_XY':bool(np.any(traces['teacher_object_xy_error_m']>position_eps)),
            'pusher_XY':bool(np.any(traces['teacher_pusher_xy_error_m']>position_eps)),
            'object_yaw':bool(np.any(np.abs(traces['teacher_object_yaw_learning_2pi_error_rad'])>data.task.yaw_tolerance_rad*.001)),
            'object_linear_speed':bool(np.any(np.linalg.norm(error[...,3:5],axis=-1)>data.task.linear_speed_tolerance_m_s*.001)),
            'pusher_linear_speed':bool(np.any(np.linalg.norm(error[...,8:10],axis=-1)>data.task.linear_speed_tolerance_m_s*.001)),
            'object_angular_speed':bool(np.any(np.abs(error[...,5])>data.task.angular_speed_tolerance_rad_s*.001))}
        detects=any(detected_components.values())
        measured=bool(np.any(traces['teacher_inference_valid_mask']))
        rows.append({'model':name,'H1_misranking':rank,'H3_recursion_only':'INCONCLUSIVE_NO_VALID_INFERENCE' if not measured else 'REJECTED_AS_COMPLETE_EXPLANATION' if detects else 'NOT_REJECTED_BY_LOCAL_DIAGNOSTIC',
            'numeric_detection_xy_m':position_eps,'detected_local_error_components':detected_components,'not_model_acceptance_threshold':True,'comparison':d})
    return {'source_tick':data.proof['source_tick'],'initial_metrics':initial,'endpoint_metrics':metrics,
        'pose_progress_subset_indices':pose_progress,'pose_and_speed_endpoint_subset_indices':pose_in_goal,
        'H2_candidate_limitation':'INCONCLUSIVE_NO_VALID_ENDPOINT' if not actual_valid.all(axis=1).any() else 'NOT_SUPPORTED_WITHIN_SUBSET' if pose_progress else 'SUPPORTED_WITHIN_SUBSET_AND_H05',
        'H2_scope':'Position improvement with nonworse yaw; no continuous hold or global reachability inference. Low J alone is not handoff.',
        'model_hypotheses':rows}


def run(args,out):
    from p4_failure_model_diagnostics import make_feature_support, diagnose_model_sequence
    start=perf_counter();registry=json.loads(REGISTRY.read_text());registry_sha=digest(REGISTRY)
    if registry_sha!=REGISTRY_SHA:raise ValueError('Preregistered analysis version changed')
    for path,sha in registry['bindings_sha256'].items():
        if digest(ROOT/path)!=sha:raise ValueError('Preregistered source/config/checkpoint binding changed')
    prior=registry['budget']['prior_consumed_bytes'];resource_check=validate_budget(registry,shutil.disk_usage(ROOT).free,prior)
    save(out/'resource_preflight.json',resource_check)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    nm=json.loads((LEARNING/'normalizer/normalizer.json').read_text())['normalizer']
    prepared,normalizer=load_learning_data(LEARNING,nm['schema'],nm['dataset_identity'],nm['normalizer_hash'])
    physical=prepared.metadata['contract']['resolved_config'];features=prepared.arrays['features'][prepared.arrays['split_code']==SPLIT_CODES['train']].copy()
    support=make_feature_support(normalizer,features);del prepared
    protocol=json.loads((ROOT/'configs/p3_planning.v1.json').read_text());planner=deepcopy(physical)
    planner['planner']=deepcopy(protocol['planner']);planner['cost']=deepcopy(protocol['cost'])
    profile=tomllib.loads((ROOT/'configs/acceptance.v1.toml').read_text())
    physical_dt_tolerance=json.loads((SOURCE/'manifest.json').read_text())['physical_contract']['dt_tolerance_s']
    action_dt=physical['physics']['timestep_s']*physical['physics']['frame_skip']
    models_all,identities_all=load_models(normalizer,physical,json.loads((ROOT/'configs/p4_development_e1.v1.json').read_text()))
    cm=json.loads((COMMON/'manifest.json').read_text());cs=json.loads((COMMON/'summary.json').read_text())
    selected=next(r for r in cs['seeds'] if r['seed']==11)
    common=load_common_checkpoint(COMMON/selected['selected_checkpoint'],normalizer,physical,cm['provenance'],cm['settings'],11)
    models={'MLP-11':common,'S1-11':models_all['S1-11'],'SK-11':models_all['SK-11']}
    identities={'MLP-11':{'path':str((COMMON/selected['selected_checkpoint']).relative_to(ROOT)),'model_hash':common.metadata['checkpoint_hash']},**{name:identities_all[name] for name in ('S1-11','SK-11')}}
    matched=validate_matched_pair(models['S1-11'].metadata,models['SK-11'].metadata)
    fs=json.loads((ROOT/'runs/p4-02-20261012-01/summary.json').read_text())
    if fs['checks'].get('three_seed_matched_start_and_entire_batch_data_order') is not True:
        raise ValueError('Original same-window/optimizer/RNG comparison missing')
    matched['same_batch_stream_evidence']={'source':'runs/p4-02-20261012-01/summary.json','sha256':digest(ROOT/'runs/p4-02-20261012-01/summary.json'),'reused_original_check':'three_seed_matched_start_and_entire_batch_data_order','new_training_or_sampler_replay':False}
    if common.metadata['checkpoint_hash']!=json.loads((SOURCE/'manifest.json').read_text())['model_metadata']['checkpoint_hash']:raise ValueError('Reproduction checkpoint differs')
    before_weights={name:tree_hash(b.model.state_dict()) for name,b in models.items()};rng_before=torch.get_rng_state().clone()
    before=upstream_all();save(out/'upstream_before.json',before)
    sources=snapshot_sources(out)
    manifest={'task_id':'P4-04','created_at':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),'registry_sha256':registry_sha,'registry':registry,
        'source_hashes':sources,'models':identities,'matched_pair':matched,'physical':physical,'planner':planner,
        'normalizer_hash':normalizer.metadata['normalizer_hash'],'learning_updates':0,'final_test_used':False,
        'personal_understanding':'NOT_ASSESSED','resource_preflight':resource_check}
    save(out/'manifest.json',manifest)
    manifest_sha=digest(out/'manifest.json')
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'));names=list(tests_in(suite))
    old=re.findall(r'^test_\S+ \(([^)]+)\)',(ROOT/'runs/p4-03-20261012-02/tests.txt').read_text(),re.MULTILINE)
    if len(old)!=772 or not set(old)<=set(names):raise ValueError('Inherited772 test identities changed')
    proc=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,capture_output=True,text=True)
    log=proc.stdout+proc.stderr;(out/'tests.txt').write_text(log)
    found=re.search(r'Ran (\d+) tests',log);executed=re.findall(r'^test_\S+ \(([^)]+)\)',log,re.MULTILINE)
    if proc.returncode or not found or int(found.group(1))!=len(names) or set(executed)!=set(names) or len(executed)!=len(names) or 'skipped=' in log:raise ValueError('Required regression population failed')
    save(out/'regression_result.json',{'pass':True,'test_count':len(names),'inherited772':True,'test_ids':names,'no_skips':True})
    print(json.dumps({'progress':'regressions_complete','tests':len(names)}),flush=True)
    guards=dict.fromkeys(ENGINE_GUARDS,0);inputs=[];choices=[];frozen_hashes={}
    initial=load_development_snapshot(SOURCE,0,physical,profile)
    for tick in registry['source_ticks']:
        data=load_development_snapshot(SOURCE,tick,physical,profile)
        if data.proof['prefix_restore_intervals']!=0:raise ValueError('P4-04 requires archived whole snapshots without physical prefix')
        folder=out/'snapshots'/f'tick-{tick:03d}';folder.mkdir(parents=True,exist_ok=False)
        save(folder/'physical_snapshot.json',asdict(data.snapshot));save(folder/'input.json',{'proof':data.proof,'source_files_sha256':data.source_files_sha256,'state':data.state.tolist(),'task':asdict(data.task),'status':asdict(data.status),'context':asdict(data.context),'G_context':asdict(data.gctx)})
        adapter=GeometryProposalAdapter(physical,protocol['geometry_selected_overrides'],protocol['geometry_transition'])
        entries=[]
        with model_planning_scope(guards):
            proposal=adapter.propose(data.state.copy(),data.task,data.gctx,data.status.elapsed_s,control_tick=tick)
            if _encode_nonfinite(proposal.diagnostics)!=data.original_plan['geometry_proposal']:raise ValueError('Original G context proposal changed')
            candidates=generate_candidates(planner,keyed_planner_rng(4711,data.case['episode_index'],tick,0),previous_actions=data.context.optimal_actions,geometry_action=proposal.geometry_action,tangent_direction=proposal.tangent_direction)
            if array_hash(candidates.actions)!=data.original_plan['candidate_hash']:raise ValueError('N128 source candidates not exactly reproduced')
            for name,bundle in models.items():
                predictor=LearnedPredictor(bundle.model,normalizer)
                pred=rollout(np.broadcast_to(data.state,(128,10)).copy(),candidates.actions,predictor,planner)
                costs=score(pred,candidates.actions,data.task,data.context.previous_applied_action,planner)
                idx,action,reason=choose(costs,pred,candidates.actions)
                if name=='MLP-11' and (idx!=data.original_plan['selected_index'] or array_hash(pred.states)!=data.original_plan['prediction_hash']):raise ValueError('Source model prediction/selection not exactly reproduced')
                entry={'model':name,'tick':tick,'selected_N128_index':idx,'reason':reason,'candidate_hash':array_hash(candidates.actions),'prediction_hash':array_hash(pred.states),'cost_hash':array_hash(costs.total),'selected_action':action.tolist()}
                prediction_trace(folder/'models'/name,pred,costs,candidates)
                entries.append((name,pred,costs,entry));choices.append(entry)
        indices=select_diagnostic_subset([e[3]['selected_N128_index'] for e in entries])
        actions=candidates.actions[indices].copy()
        for e in entries:e[3]['selected_subset_index']=None if e[3]['selected_N128_index'] is None else indices.index(e[3]['selected_N128_index'])
        np.savez_compressed(folder/'candidates.npz',all_actions=candidates.actions,subset_actions=actions,original_indices=np.asarray(indices,np.int64))
        save(folder/'subset_before_physics.json',{'indices':indices,'selection_rule':registry['subset_rule'],'choices':[e[3] for e in entries]})
        # Keep N128 predictions for provenance, subset for shared actual comparison.
        from push_align.planning import PredictionBatch,CostBatch
        subset_entries=[]
        for name,p,c,e in entries:
            sub=PredictionBatch(p.states[indices],p.raw_deltas[indices],p.valid[indices],tuple(p.reasons[i] for i in indices),p.failure_steps[indices])
            co=CostBatch(c.total[indices],{key:value[indices] for key,value in c.components.items()})
            subset_entries.append((name,sub,co,e))
        inputs.append((data,folder,indices,actions,subset_entries))
    save(out/'choices_before_any_new_physical_future.json',choices)
    for p in (out/'snapshots').rglob('*'):
        if p.is_file():frozen_hashes[str(p.relative_to(out))]=digest(p)
    frozen_hashes['choices_before_any_new_physical_future.json']=digest(out/'choices_before_any_new_physical_future.json')
    save(out/'frozen_prediction_hashes.json',frozen_hashes)
    print(json.dumps({'progress':'all6_N128_choices_and_subsets_frozen','new_physics_intervals':0}),flush=True)
    settings=deepcopy(json.loads((ROOT/'runs/p3-05-20261010-01/manifest.json').read_text())['settings'])
    settings['full_trace_ticks_resource']=[0,49,99,199,299]
    # Guard shared P4 allocation before each cell; old cell runner also checks free disk every25 ticks.
    cells=[];guard(out,prior)
    cell=run_cell(out/'closed_loop'/'MLP-11',initial.case,initial.snapshot,'MLP-11',common,normalizer,physical,planner,protocol['geometry_selected_overrides'],protocol['geometry_transition'],settings,intervals=300,resource_case=True)
    cells.append(cell)
    reproduction=compare_closed_loop_reproduction(SOURCE,out/'closed_loop'/'MLP-11');save(out/'failure_reproduction.json',reproduction)
    if not reproduction['pass'] or cell['execution_status']!='completed' or cell['termination']!='timeout' or cell['transitions']!=300:raise ValueError('Original actual failure not exactly reproduced; do not substitute')
    e1=[];replay_count=interval_count=0;replay_ledger=[]
    for data,folder,indices,actions,entries in inputs:
        states=np.full((16,3,6,10),np.nan,np.float64);applied=np.full((16,3,5,2),np.nan,np.float64);costs=np.full((16,3),np.inf,np.float64);valid=np.zeros((16,3),np.bool_);interval_valid=np.zeros((16,3,5),np.bool_)
        for i,action in enumerate(actions):
            reference=None
            for repeat in range(3):
                guard(out,prior);result=replay_candidate(physical,planner,data.snapshot,action,data.task,data.context.previous_applied_action,profile)
                dest=folder/'actual'/f'candidate-{indices[i]:03d}'/f'repeat-{repeat}';dest.mkdir(parents=True,exist_ok=False)
                save_trace(dest,'raw_execution',data.snapshot,result.results)
                states[i,repeat]=result.states;applied[i,repeat]=result.applied_actions;costs[i,repeat]=result.cost;valid[i,repeat]=result.valid
                prefix=True
                for j,step in enumerate(result.results):
                    prefix=(prefix and step.transition_valid and step.observation.observation_valid
                        and abs(step.actual_dt-action_dt)<=physical_dt_tolerance
                        and np.array_equal(step.applied_action,action[j])
                        and check_contact_geometry(step.diagnostics['samples'],profile['physics_suite']['actual_contact_penetration_max_m'])['pass'])
                    interval_valid[i,repeat,j]=prefix
                save(dest/'final_snapshot.json',asdict(result.final_snapshot));save(dest/'result.json',{'valid':result.valid,'cost':result.cost,'cost_components':result.cost_components,'completed_intervals':result.completed_intervals,'initial_exact':result.initial_snapshot_exact,'reasons':result.reasons})
                replay_ledger.append({'tick':data.proof['source_tick'],'original_index':indices[i],'subset_index':i,'repeat':repeat,'valid':result.valid,'attempted_intervals':len(result.results),'completed_intervals':result.completed_intervals,'valid_prefix_intervals':int(interval_valid[i,repeat].sum()),'reasons':result.reasons})
                if reference is not None and not (np.array_equal(result.states,reference[0],equal_nan=True) and np.array_equal(result.applied_actions,reference[1],equal_nan=True) and result.cost==reference[2] and result.valid==reference[3] and snapshots_exact(result.final_snapshot,reference[4])):raise ValueError('Repeat state/cost/validity/full final snapshot changes; hypotheses unresolved')
                reference=(result.states.copy(),result.applied_actions.copy(),result.cost,result.valid,result.final_snapshot)
                replay_count+=1;interval_count+=len(result.results);del result;gc.collect()
        np.savez_compressed(folder/'actual_labels.npz',states=states,applied_actions=applied,total_cost=costs,valid=valid,interval_valid=interval_valid)
        local={}
        with model_planning_scope(guards):
            for name,bundle in models.items():
                report=diagnose_model_sequence(LearnedPredictor(bundle.model,normalizer),support,states[:,0],applied[:,0],actual_valid_mask=interval_valid[:,0].copy())
                local[name]={'summary':report.summary,'traces':report.traces}
                save(folder/'models'/name/'local_diagnostics.json',report.summary)
                np.savez_compressed(folder/'models'/name/'local_diagnostics.npz',**report.traces)
        row=hypotheses_for_snapshot(data,actions,states,costs,valid,entries,local);row['original_indices']=indices
        save(folder/'hypothesis_results.json',row);e1.append(row)
        print(json.dumps({'progress':'failure_path_E1_complete','tick':data.proof['source_tick'],'actual_valid':int(valid.all(axis=1).sum()),'H2':row['H2_candidate_limitation']}),flush=True)
    for name in ('S1-11','SK-11'):
        guard(out,prior)
        row=run_cell(out/'closed_loop'/name,initial.case,initial.snapshot,name,models[name],normalizer,physical,planner,protocol['geometry_selected_overrides'],protocol['geometry_transition'],settings,intervals=300,resource_case=True)
        cells.append(row)
        if row['execution_status']!='completed':raise ValueError('Incomplete cell is infrastructure failure, not task result')
    after=upstream_all();save(out/'upstream_after.json',after)
    save(out/'replay_ledger.json',replay_ledger)
    checks={'regressions':True,'actual_failure_exact':reproduction['pass'],'matched_loss_only':matched['pass'],
        'all_choices_before_physics':len(choices)==6,'two_archived_failure_path_snapshots':len(e1)==2,
        'all_candidate_attempts':replay_count==96 and interval_count<=480 and all(r['completed_intervals']==5 if r['valid'] else 0<=r['completed_intervals']<=5 and 1<=r['attempted_intervals']<=5 for r in replay_ledger),
        'three_repeats_exact':True,'all_three_closed_loop_cells':len(cells)==3 and all(r['execution_status']=='completed' for r in cells),
        'raw_roundtrip':all(r['compressed_raw_roundtrip_exact'] for r in cells),
        'weights_unchanged':all(before_weights[n]==tree_hash(b.model.state_dict()) for n,b in models.items()),
        'torch_rng_unchanged':torch.equal(rng_before,torch.get_rng_state()),'future_engine_calls_zero':not any(guards.values()) and all(not any(r['future_engine_calls'].values()) for r in cells),
        'upstream_unchanged':before==after,'source_unchanged':all(digest(ROOT/p)==h and digest(out/'source'/p)==h for p,h in sources.items()),
        'manifest_unchanged':digest(out/'manifest.json')==manifest_sha,
        'choices_unchanged':all(digest(out/p)==h for p,h in frozen_hashes.items()),
        'pool_and_reserve':prior+byte_count(out)<GIB and shutil.disk_usage(ROOT).free>=2*GIB}
    distinguished=any(r['H3_recursion_only']=='REJECTED_AS_COMPLETE_EXPLANATION' or r['H1_misranking']=='SUPPORTED_WITHIN_SUBSET' for x in e1 for r in x['model_hypotheses']) or any(x['H2_candidate_limitation'] in ('NOT_SUPPORTED_WITHIN_SUBSET','SUPPORTED_WITHIN_SUBSET_AND_H05') for x in e1)
    resources={'wall_seconds':perf_counter()-start,'closed_loop_intervals':sum(r['transitions'] for r in cells),'additional_E1_intervals':interval_count,'E1_replays':replay_count,'learning_updates':0,'estimated_stage_bytes':registry['budget']['estimated_stage_bytes'],'stored_before_summary_bytes':byte_count(out),'prior_P4_pool_bytes':prior,'free_after_bytes':shutil.disk_usage(ROOT).free,'extra_spending_KRW':0}
    summary={'task_id':'P4-04','pass':bool(all(checks.values()) and distinguished),'execution_pass':bool(all(checks.values())),
        'phase_criterion_pass':bool(distinguished),'root_cause_status':'PARTIALLY_DISTINGUISHED_NOT_FULLY_ATTRIBUTED' if distinguished else 'INCONCLUSIVE',
        'checks':checks,'cells':cells,'E1':e1,'resources':resources,'model_planning_guard_counts':guards,
        'identified':'Limited failure-path local prediction / candidate diagnostic; see each hypothesis status. No complete root-cause attribution.',
        'loss_contrast':'Frozen S1-11 vs SK-11 only; same initial condition, learned loss change; divergent later states/candidates are policy effects.',
        'remaining':'Single reused case and seed; model/data/contact/candidate/horizon contributions not causally isolated by all diagnostics. No final-test generalization, field effect, or personal understanding claim.',
        'final_test_used':False,'personal_understanding':'NOT_ASSESSED','learning_updates':0,'next_task':'P5-01 only after reviewing unresolved hypotheses and freezing final scope'}
    save(out/'resources.json',resources);save(out/'summary.json',summary)
    if not summary['execution_pass']:raise ValueError('P4-04 execution/evidence incomplete; no automatic DONE')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True);args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id):raise ValueError('Unsafe run ID')
    out=ROOT/'runs'/args.run_id;out.mkdir(parents=True,exist_ok=False)
    try:result=run(args,out)
    except Exception as error:
        save(out/'failure.json',{'task_id':'P4-04','pass':False,'error':str(error),'traceback':traceback.format_exc()})
        print(json.dumps({'pass':False,'error':str(error),'run':str(out)}),flush=True);return 1
    finally:save(out/'artifact_hashes.json',{'sha256':{str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'}})
    print(json.dumps({'pass':result['pass'],'run':str(out)}),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
