"""P4 development correction: paired learned forecasts, three evaluation rules.

H5 proposals, models, observed state/history, and the physical task stay fixed.
Every proposal is forecast with a common five-zero stopping suffix. H5 J is a
reference; H10 J versus H10 handoff rank is the cost-only contrast. First choices
are frozen before new physics. Actual substep handoff is never a model input.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
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
from zoneinfo import ZoneInfo

import numpy as np
import torch

from push_align.candidate_replay import replay_candidate, snapshots_exact
from push_align.candidates import generate_candidates
from push_align.closed_loop import ENGINE_GUARDS, model_planning_scope, consume_actual_step
from push_align.config import ROOT
from push_align.contact_geometry import check_contact_geometry
from push_align.data import digest, _encode_nonfinite
from push_align.environment import PlanarPushEnv, Observation, StepResult
from push_align.handoff_cost import score_handoff, choose_handoff
from push_align.learning_data import load_learning_data
from push_align.planning import LearnedPredictor, PredictionBatch, PlannerContext, rollout, score, choose
from push_align.planning_protocol import GeometryProposalAdapter, keyed_planner_rng
from push_align.task import handoff_metrics, pose_in_goal, fail_task
from p2_collect_pilot import save_trace
from p3_geometry_pilot import verify_artifacts
from p3_train_common_mlp import load_common_checkpoint
from p3_train_ridge import array_hash
from p4_compare_candidates import save, load_models, tests_in
from p4_development_inputs import load_development_snapshot
from p4_failure_analysis import upstream_all, validate_matched_pair
from p4_stop_task_summary import summarize_task_prefix
from p4_train_forks import byte_count, tree_hash
from p4_verify_sequence_loss import snapshot_sources, inventory

REGISTRY = ROOT/'configs/p4_handoff_cost.v1.json'
SOURCE = ROOT/'runs/p3-05-20261010-01/full_resource/MLP-11'
PREVIOUS = ROOT/'runs/p4-04-20261013-01'
STOP = ROOT/'runs/p4-04-stop-20261013-01'
LEARNING = ROOT/'runs/p3-02-20261009-01/learning_data'
COMMON = ROOT/'runs/p3-04-20261009-01'
ARMS = ('J_H5', 'J_H10_stop', 'TASK_H10_stop')
GIB = 1024**3


def prefix_prediction(predictions, horizon=5):
    """H10 failures after H5 must not exclude an originally valid H5 reference."""
    if type(horizon) is not int or horizon != 5:
        raise ValueError('This registered comparison requires H5 prefix')
    valid = (predictions.failure_steps < 0) | (predictions.failure_steps > horizon)
    return PredictionBatch(predictions.states[:, :horizon+1].copy(),
        predictions.raw_deltas[:, :horizon].copy(), valid,
        tuple('' if v else r for v,r in zip(valid,predictions.reasons)),
        np.where(valid,-1,predictions.failure_steps).astype(np.int64))


def shared_subset(choices, count=128, size=16):
    """Four reserved slots, all nine choices, prior11/60, then ascending fill."""
    if count != 128 or size != 16:
        raise ValueError('Frozen candidate and subset counts required')
    picked = {0,1,2,3,11,60}
    for row in choices:
        i=row['selected_index']
        if i is not None:
            if type(i) is not int or not 0 <= i < count:raise ValueError('Bad choice')
            picked.add(i)
    if len(picked)>size:raise ValueError('Required choices exceed registered subset')
    for i in range(count):
        if len(picked)==size:break
        picked.add(i)
    return sorted(picked)


def extended_config(planner):
    result=deepcopy(planner)
    result['planner']['horizon']=10
    # The generator still receives original H5 / two blocks; this config is
    # only for model rollout and cost of the fixed appended zero commands.
    result['planner']['block_lengths']=[2,3,5]
    return result


def forecast(head, state, task, context, status, predictor, planner):
    full=np.concatenate((head,np.zeros((len(head),5,2),np.float64)),axis=1)
    config10=extended_config(planner)
    predictions=rollout(np.broadcast_to(state,(len(head),10)).copy(),full,predictor,config10)
    short=prefix_prediction(predictions)
    j5=score(short,head,task,context.previous_applied_action,planner)
    j10=score(predictions,full,task,context.previous_applied_action,config10)
    handoff=score_handoff(predictions,task,j10,.1,
        initial_held_time_s=status.held_time_s,
        initial_previous_qualifies=status.previous_qualifies)
    selections=[choose(j5,short,head),choose(j10,predictions,full),choose_handoff(handoff,full)]
    choices=[]
    for arm,(idx,action,reason) in zip(ARMS,selections):
        choices.append({'arm':arm,'selected_index':idx,'first_action':action.tolist(),'reason':reason,
            'predicted_surrogate_ready':False if idx is None else bool(handoff.ready[idx]),
            'first_surrogate_ready_step':None if idx is None or handoff.first_ready_step[idx]<0 else int(handoff.first_ready_step[idx]),
            'selected_J_H5':None if idx is None else float(j5.total[idx]),
            'selected_J_H10':None if idx is None else float(j10.total[idx]),
            'selected_task_rank':None if idx is None else handoff.rank_keys[idx].tolist()})
    return {'full_actions':full,'predictions':predictions,'short':short,'J5':j5,'J10':j10,
            'handoff':handoff,'choices':choices}


def save_forecast(folder, head, result):
    folder.mkdir(parents=True,exist_ok=False)
    p=result['predictions'];s=result['handoff']
    np.savez_compressed(folder/'predictions.npz',head_actions=head,full_actions=result['full_actions'],
        states=p.states,raw_deltas=p.raw_deltas,valid=p.valid,reasons=np.asarray(p.reasons),failure_steps=p.failure_steps,
        prefix_valid=result['short'].valid,J_H5=result['J5'].total,J_H10=result['J10'].total,
        task_ready=s.ready,task_first_ready_step=s.first_ready_step,task_eligible=s.eligible,
        task_ratios=s.ratios,task_held_time_s=s.held_time_s,task_rank=s.rank_keys,
        **{'J5_'+k:v for k,v in result['J5'].components.items()},
        **{'J10_'+k:v for k,v in result['J10'].components.items()})
    save(folder/'choices.json',result['choices'])


def resource_guard(out, prior):
    if prior+byte_count(out)>GIB or shutil.disk_usage(ROOT).free<2*GIB:
        raise RuntimeError('P4 shared pool/free reserve reached; retain prefix')


def canonical_status_equal(left, right):
    """JSON list vs dataclass tuple is serialization, not a task discrepancy."""
    return _encode_nonfinite(left)==_encode_nonfinite(right)


def recover_recorded_cell(folder, data, profile):
    """Finish summary of an entirely recorded valid cell without any physics.

    Only the serialization-comparison interruption is eligible for recovery.
    Every original row must have valid physics/geometry, raw boundaries/actions
    and full diagnostic arrays must match, and raw supervision must reproduce
    every logged task status. Other failures are not silently reconstructed.
    """
    records=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
    with gzip.open(folder/'raw_execution.json.gz','rt') as stream:raw=json.load(stream)
    with np.load(folder/'raw_execution.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    if len(records)!=20 or len(raw['diagnostics'])!=20 or not all(r['physical_valid'] and r['geometry']['pass'] for r in records):raise ValueError('Recovery requires all20 complete valid recorded intervals')
    skip=len(arrays['substep_times'])//len(records);steps=[]
    for i,(record,diag) in enumerate(zip(records,raw['diagnostics'])):
        sl=slice(i*skip,(i+1)*skip)
        if not (record['global_tick']==99+i and np.array_equal(arrays['state'][i],record['state_after'])
                and np.array_equal(arrays['action_applied'][i],record['applied_action'])
                and np.array_equal(arrays['action_requested'][i],record['requested_action'])):raise ValueError('Recovery raw/record mismatch')
        steps.append(StepResult(arrays['action_requested'][i],arrays['action_applied'][i],
            Observation(arrays['state'][i],float(arrays['substep_times'][sl][-1]),'push-align-state-v1',True),
            float(arrays['actual_dt'][i]),True,diag,arrays['substep_states'][sl],arrays['substep_times'][sl]))
    checked=summarize_task_prefix(steps,deepcopy(data.status),data.task,.1,profile)
    if not canonical_status_equal(checked['status'],records[-1]['task']):raise ValueError('Recovered final raw task differs')
    for record,row in zip(records,checked['timeline']):
        if record['task']['phase']!=row['phase'] or record['task']['held_time_s']!=row['held_time_s']:raise ValueError('Recovered task timeline differs')
    manifest=json.loads((folder/'manifest.json').read_text());status=checked['status']
    model=manifest['model'];arm=manifest['arm']
    return {'model':model,'arm':arm,'phase':status['phase'],'termination':status['terminal_reason'] or 'development_budget',
        'intervals':20,'elapsed_from_source_s':status['last_sim_time']-status['initial_sim_time']-data.status.elapsed_s,
        'total_original_task_elapsed_s':status['last_sim_time']-status['initial_sim_time'],'task':status,'task_prefix':checked,
        'final_metrics':handoff_metrics(arrays['state'][-1],data.task),'first_plan':records[0]['plan'],
        'valid_intervals':20,'raw_roundtrip':True,'future_engine_calls':dict.fromkeys(ENGINE_GUARDS,0),
        'guard_proof':'All20 guarded planning decisions completed without forbidden engine exception; serialized status comparison was the sole interruption.',
        'predictor_calls':None,
        'predictor_rows':'Recovery uses archived prediction arrays; no new inference for this cell',
        'recovery':'Recorded20 intervals reused; raw/record/task timeline checked without advancing physics'}


def bounded_cell(out, model_name, arm, data, bundle, normalizer, physical, planner, protocol, profile, budget, prior):
    """Replan from source99 history for at most2s; never reset the task clock."""
    out.mkdir(parents=True,exist_ok=False)
    env=PlanarPushEnv(deepcopy(physical));observed=env.restore(deepcopy(data.snapshot))
    if not snapshots_exact(data.snapshot,env.snapshot()):raise ValueError('Full source restoration differs')
    status=deepcopy(data.status);context=deepcopy(data.context);gctx=deepcopy(data.gctx)
    adapter=GeometryProposalAdapter(physical,protocol['geometry_selected_overrides'],protocol['geometry_transition'])
    predictor=LearnedPredictor(bundle.model,normalizer);guards=dict.fromkeys(ENGINE_GUARDS,0)
    manifest={'model':model_name,'arm':arm,'source_tick':99,'source_task_status':asdict(status),
        'max_intervals':budget,'task':asdict(data.task),'physical':physical,'head_planner':planner,
        'tail':'five fixed zeros only in forecasts; execute first0.1 then replan',
        'holding':'unchanged observed pose-in-goal zero override','learning_updates':0,'final_test_used':False}
    save(out/'manifest.json',manifest)
    results=[];records=[];first_plan=None
    for local in range(budget):
        if status.phase in ('ready','failed'):break
        resource_guard(out,prior)
        tick=99+local;holding=pose_in_goal(observed.state,data.task);before=observed.state.copy()
        old_context=deepcopy(context);old_gctx=deepcopy(gctx);decision=None
        with model_planning_scope(guards):
            proposal=adapter.propose(before.copy(),data.task,gctx,status.elapsed_s,control_tick=tick)
            if holding:
                action=np.zeros(2,np.float64)
            else:
                rng=keyed_planner_rng(4711,data.case['episode_index'],tick,0)
                candidates=generate_candidates(planner,rng,previous_actions=context.optimal_actions,
                    geometry_action=proposal.geometry_action,tangent_direction=proposal.tangent_direction)
                f=forecast(candidates.actions,before,data.task,context,status,predictor,planner)
                selected=next(r for r in f['choices'] if r['arm']==arm)
                action=np.asarray(selected['first_action'],np.float64)
                idx=selected['selected_index']
                context=PlannerContext(action.copy(),None if idx is None else candidates.actions[idx].copy(),context.control_step+1)
                decision={**selected,'all_three_choices':f['choices'],'candidate_hash':array_hash(candidates.actions),
                    'full_prediction_hash':array_hash(f['predictions'].states),'global_tick':tick,
                    'task_status_before':asdict(status),'planner_context_before':asdict(old_context)}
                save_forecast(out/f'plans/step-{local:03d}',candidates.actions,f)
                if first_plan is None:first_plan=deepcopy(decision)
        gctx=adapter.commit(proposal,actual_control_tick=tick)
        if decision is not None and decision['selected_index'] is None:
            status=fail_task(status,'planner_invalid');save(out/'unexecuted_plan.json',decision);break
        step=env.step(action.copy());results.append(step)
        geometry=check_contact_geometry(step.diagnostics['samples'],profile['physics_suite']['actual_contact_penetration_max_m'])
        status,context=consume_actual_step(status,step,data.task,.1,context,holding=holding,geometry=geometry)
        observed=step.observation
        row={'index':local,'global_tick':tick,'source':'holding' if holding else 'model',
            'requested_action':action.tolist(),'applied_action':step.applied_action.tolist(),
            'state_before':before.tolist(),'state_after':observed.state.tolist(),'actual_dt_s':step.actual_dt,
            'physical_valid':bool(step.transition_valid),'geometry':geometry,'task':asdict(status),
            'G_context_before':asdict(old_gctx),'G_context_after':asdict(gctx),'plan':decision}
        records.append(row)
        with (out/'steps.jsonl').open('a') as stream:stream.write(json.dumps(_encode_nonfinite(row),ensure_ascii=False,allow_nan=False)+'\n')
    save_trace(out,'raw_execution',data.snapshot,results)
    save(out/'final_snapshot.json',asdict(env.snapshot()))
    if not results:raise ValueError('No valid physical interval in bounded cell')
    task_check=summarize_task_prefix(results,deepcopy(data.status),data.task,.1,profile)
    if not canonical_status_equal(task_check['status'],asdict(status)):raise ValueError('Task-prefix summary differs')
    with gzip.open(out/'raw_execution.json.gz','rt') as stream:raw=json.load(stream)
    with np.load(out/'raw_execution.npz',allow_pickle=False) as z:
        roundtrip=np.array_equal(z['state'],np.asarray([r.observation.state for r in results])) and raw['diagnostics']==_encode_nonfinite([r.diagnostics for r in results])
    row={'model':model_name,'arm':arm,'phase':status.phase,'termination':status.terminal_reason or 'development_budget',
        'intervals':len(results),'elapsed_from_source_s':status.elapsed_s-data.status.elapsed_s,
        'total_original_task_elapsed_s':status.elapsed_s,'task':asdict(status),'task_prefix':task_check,
        'final_metrics':handoff_metrics(observed.state,data.task),'first_plan':first_plan,
        'valid_intervals':sum(bool(r.transition_valid) and records[i]['geometry']['pass'] for i,r in enumerate(results)),
        'raw_roundtrip':bool(roundtrip),'future_engine_calls':guards,'predictor_calls':predictor.calls,'predictor_rows':predictor.rows}
    save(out/'summary.json',row)
    print(json.dumps({'progress':'bounded_cell_complete','model':model_name,'arm':arm,'phase':status.phase,'intervals':len(results),'position_mm':row['final_metrics']['e_pos_m']*1000}),flush=True)
    return row


def run(out, registry_sha, registry_path):
    began=perf_counter();registry=json.loads(registry_path.read_text())
    if digest(registry_path)!=registry_sha:raise ValueError('Preregistration hash differs')
    for p,h in registry['bindings_sha256'].items():
        if digest(ROOT/p)!=h:raise ValueError('Frozen binding differs: '+p)
    prior=registry['budget']['prior_pool_bytes'];free=shutil.disk_usage(ROOT).free
    if prior+registry['budget']['estimated_run_bytes']>GIB or free<registry['budget']['estimated_run_bytes']+2*GIB:raise ValueError('Budget/reserve does not fit')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    nm=json.loads((LEARNING/'normalizer/normalizer.json').read_text())['normalizer']
    prepared,normalizer=load_learning_data(LEARNING,nm['schema'],nm['dataset_identity'],nm['normalizer_hash'])
    physical=prepared.metadata['contract']['resolved_config'];del prepared
    protocol=json.loads((ROOT/'configs/p3_planning.v1.json').read_text());planner=deepcopy(physical)
    planner['planner']=deepcopy(protocol['planner']);planner['cost']=deepcopy(protocol['cost'])
    profile=tomllib.loads((ROOT/'configs/acceptance.v1.toml').read_text())
    all_models,identities_all=load_models(normalizer,physical,json.loads((ROOT/'configs/p4_development_e1.v1.json').read_text()))
    cm=json.loads((COMMON/'manifest.json').read_text());cs=json.loads((COMMON/'summary.json').read_text())
    sel=next(r for r in cs['seeds'] if r['seed']==11)
    common=load_common_checkpoint(COMMON/sel['selected_checkpoint'],normalizer,physical,cm['provenance'],cm['settings'],11)
    models={'MLP-11':common,'S1-11':all_models['S1-11'],'SK-11':all_models['SK-11']}
    identities={'MLP-11':{'path':str((COMMON/sel['selected_checkpoint']).relative_to(ROOT)),'model_hash':common.metadata['checkpoint_hash']},**{k:identities_all[k] for k in ('S1-11','SK-11')}}
    weights_before={k:tree_hash(v.model.state_dict()) for k,v in models.items()};rng_before=torch.get_rng_state().clone()
    matched=validate_matched_pair(models['S1-11'].metadata,models['SK-11'].metadata)
    before=upstream_all()
    for folder in (PREVIOUS,STOP):before[str(folder.relative_to(ROOT))]={'registered_artifacts':verify_artifacts(folder),'file_inventory':inventory(folder)}
    reuse=None
    if registry.get('reuse_failed_run'):
        reuse=ROOT/registry['reuse_failed_run']['path']
        failed=json.loads((reuse/'failure.json').read_text())
        if failed['error']!='Task-prefix summary differs':raise ValueError('Unapproved interrupted run kind')
        before[str(reuse.relative_to(ROOT))]={'registered_artifacts':verify_artifacts(reuse),'file_inventory':inventory(reuse)}
    save(out/'upstream_before.json',before);sources=snapshot_sources(out)
    manifest={'task_id':'P4-development-cost-correction','created_at':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        'registry_sha256':registry_sha,'registry':registry,'source_hashes':sources,'model_identities':identities,
        'matched_S1_SK':matched,'physical':physical,'head_planner':planner,'evaluation_planner':extended_config(planner),
        'learning_updates':0,'final_test_used':False,'personal_understanding':'NOT_ASSESSED','free_before_bytes':free}
    save(out/'manifest.json',manifest)
    import unittest
    names=list(tests_in(unittest.defaultTestLoader.discover(str(ROOT/'tests'))))
    old=json.loads((PREVIOUS/'regression_result.json').read_text())['test_ids']
    if len(old)!=826 or not set(old)<=set(names):raise ValueError('Previous826 test IDs changed')
    proc=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,capture_output=True,text=True)
    log=proc.stdout+proc.stderr;(out/'tests.txt').write_text(log)
    found=re.search(r'Ran (\d+) tests',log);executed=re.findall(r'^test_\S+ \(([^)]+)\)',log,re.MULTILINE)
    if proc.returncode or not found or int(found.group(1))!=len(names) or set(executed)!=set(names) or len(executed)!=len(names) or 'skipped=' in log:raise ValueError('Full regression failed/skipped')
    save(out/'regression_result.json',{'pass':True,'test_count':len(names),'test_ids':names,'inherited826_ids_preserved':True,'no_skips':True})
    print(json.dumps({'progress':'regressions_complete','tests':len(names)}),flush=True)
    data=load_development_snapshot(SOURCE,99,physical,profile)
    if data.proof['prefix_restore_intervals']!=0:raise ValueError('Expected stored whole source snapshot')
    save(out/'input.json',{'state':data.state.tolist(),'task':asdict(data.task),'status':asdict(data.status),'context':asdict(data.context),'G_context':asdict(data.gctx),'proof':data.proof})
    save(out/'physical_snapshot.json',asdict(data.snapshot))
    adapter=GeometryProposalAdapter(physical,protocol['geometry_selected_overrides'],protocol['geometry_transition'])
    guards=dict.fromkeys(ENGINE_GUARDS,0);initial={};choices=[]
    with model_planning_scope(guards):
        proposal=adapter.propose(data.state.copy(),data.task,data.gctx,data.status.elapsed_s,control_tick=99)
        candidates=generate_candidates(planner,keyed_planner_rng(4711,data.case['episode_index'],99,0),previous_actions=data.context.optimal_actions,geometry_action=proposal.geometry_action,tangent_direction=proposal.tangent_direction)
        if array_hash(candidates.actions)!=data.original_plan['candidate_hash']:raise ValueError('Original N128 differs')
        for name,bundle in models.items():
            predictor=LearnedPredictor(bundle.model,normalizer)
            f=forecast(candidates.actions,data.state,data.task,data.context,data.status,predictor,planner)
            if name=='MLP-11' and (f['choices'][0]['selected_index']!=data.original_plan['selected_index'] or array_hash(f['short'].states)!=data.original_plan['prediction_hash']):raise ValueError('Original Common H5 prediction/choice differs')
            for row in f['choices']:row['model']=name
            save_forecast(out/'initial_models'/name,candidates.actions,f)
            initial[name]=f;choices.extend(deepcopy(f['choices']))
    indices=shared_subset(choices);actions=np.concatenate((candidates.actions[indices],np.zeros((16,5,2),np.float64)),axis=1)
    np.savez_compressed(out/'candidates.npz',all_head_actions=candidates.actions,subset_full_actions=actions,original_indices=np.asarray(indices,np.int64))
    save(out/'choices_before_new_physics.json',{'choices':choices,'subset':indices,'new_physics_intervals':0,'candidate_hash':array_hash(candidates.actions),'predictions':{k:array_hash(v['predictions'].states) for k,v in initial.items()}})
    if reuse is not None:
        if json.loads((reuse/'choices_before_new_physics.json').read_text())!=json.loads((out/'choices_before_new_physics.json').read_text()):raise ValueError('Reused frozen choices/predictions differ')
        with np.load(reuse/'candidates.npz',allow_pickle=False) as a,np.load(out/'candidates.npz',allow_pickle=False) as b:
            if set(a.files)!=set(b.files) or not all(np.array_equal(a[k],b[k]) for k in a.files):raise ValueError('Reused candidate arrays differ')
        save(out/'reuse_proof_before_physics.json',{'pass':True,'source_run':str(reuse.relative_to(ROOT)),
            'source_index_sha256':digest(reuse/'artifact_hashes.json'),'all9_frozen_choices_and_predictions_exact':True,
            'candidate_arrays_exact':True,'prior_executed_intervals':500,'new_budget_intervals':160,
            'reason':'Only status serialization equivalence changed; old48 actual replays and complete20-interval first cell are reused.'})
    frozen={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}
    save(out/'frozen_before_physics.json',frozen)
    print(json.dumps({'progress':'all9_choices_frozen','subset':indices,'new_physics_intervals':0}),flush=True)
    actual_rows=[];actual_lookup={};physical_intervals=0
    for index,seq in zip(indices,actions):
        reference=None
        for repeat in range(3):
            resource_guard(out,prior);dest=out/f'actual/candidate-{index:03d}/repeat-{repeat}';dest.mkdir(parents=True,exist_ok=False)
            if reuse is not None:
                src=reuse/f'actual/candidate-{index:03d}/repeat-{repeat}'
                if not src.exists():raise ValueError('Registered reused candidate missing')
                shutil.copytree(src,dest,dirs_exist_ok=True)
                row=json.loads((dest/'result.json').read_text())
                if row['original_index']!=index or row['repeat']!=repeat:raise ValueError('Reused row index changed')
                actual_rows.append(row)
                if repeat==0:actual_lookup[index]=row
                continue
            replay=replay_candidate(physical,extended_config(planner),data.snapshot,seq,data.task,data.context.previous_applied_action,profile)
            save_trace(dest,'raw_execution',data.snapshot,replay.results);save(dest/'final_snapshot.json',asdict(replay.final_snapshot))
            actualpred=PredictionBatch(replay.states[None],np.diff(replay.states,axis=0)[None],np.asarray([replay.valid]),((';'.join(replay.reasons)),),np.asarray([-1 if replay.valid else len(replay.results)],np.int64))
            j10=score(actualpred,seq[None],data.task,data.context.previous_applied_action,extended_config(planner))
            hs=score_handoff(actualpred,data.task,j10,.1,initial_held_time_s=data.status.held_time_s,initial_previous_qualifies=data.status.previous_qualifies)
            task=summarize_task_prefix(replay.results,deepcopy(data.status),data.task,.1,profile)
            row={'original_index':index,'repeat':repeat,'physical_valid':replay.valid,'completed_intervals':replay.completed_intervals,
                'actual_J_H10':replay.cost,'actual_J_components':replay.cost_components,
                'actual_endpoint_proxy_ready':bool(hs.ready[0]),'actual_endpoint_proxy_first_step':int(hs.first_ready_step[0]),
                'actual_task_rank':hs.rank_keys[0].tolist(),'raw_task':task,'final_metrics':handoff_metrics(replay.states[-1],data.task) if replay.valid else None,'reasons':replay.reasons}
            with gzip.open(dest/'raw_execution.json.gz','rt') as stream:raw=json.load(stream)
            if reference is not None and not (np.array_equal(replay.states,reference[0],equal_nan=True) and replay.cost==reference[1] and snapshots_exact(replay.final_snapshot,reference[2]) and row['raw_task']==reference[3] and raw['diagnostics']==reference[4]):raise ValueError('Physical repeats differ; retain and investigate')
            reference=(replay.states.copy(),replay.cost,replay.final_snapshot,task,raw['diagnostics'])
            save(dest/'result.json',row);actual_rows.append(row);physical_intervals+=len(replay.results)
            if repeat==0:actual_lookup[index]=row
    save(out/'actual_ledger.json',actual_rows)
    effects=[]
    for row in choices:
        idx=row['selected_index'];actual=None if idx is None else actual_lookup[idx]
        effects.append({**row,'actual':actual})
    pair={'candidate11':actual_lookup[11],'candidate60':actual_lookup[60],
        'new_actual_rank_prefers11':tuple(actual_lookup[11]['actual_task_rank'])<tuple(actual_lookup[60]['actual_task_rank']),
        'legacy_actual_J_prefers60':actual_lookup[60]['actual_J_H10']<actual_lookup[11]['actual_J_H10'],
        'scope':'Known post-hoc pair: objective compatibility check, not unseen validation'}
    save(out/'selection_effects.json',{'initial_choices':effects,'known_pair':pair})
    cells=[]
    for name,bundle in models.items():
        for arm in ARMS:
            dest=out/'bounded_closed_loop'/name/arm
            if reuse is not None and name=='MLP-11' and arm=='J_H5':
                shutil.copytree(reuse/'bounded_closed_loop'/name/arm,dest)
                cell=recover_recorded_cell(dest,data,profile);save(dest/'summary.json',cell)
            else:
                cell=bounded_cell(dest,name,arm,data,bundle,normalizer,physical,planner,protocol,profile,registry['budget']['closed_loop_intervals_per_cell'],prior+byte_count(out))
                physical_intervals+=cell['intervals']
            initial_choice=next(r for r in choices if r['model']==name and r['arm']==arm)
            if cell['first_plan'] is None or cell['first_plan']['selected_index']!=initial_choice['selected_index'] or cell['first_plan']['candidate_hash']!=array_hash(candidates.actions):raise ValueError('Bounded first choice differs from prephysics freeze')
            cells.append(cell)
    after=upstream_all()
    for folder in (PREVIOUS,STOP):after[str(folder.relative_to(ROOT))]={'registered_artifacts':verify_artifacts(folder),'file_inventory':inventory(folder)}
    if reuse is not None:after[str(reuse.relative_to(ROOT))]={'registered_artifacts':verify_artifacts(reuse),'file_inventory':inventory(reuse)}
    save(out/'upstream_after.json',after)
    checks={'all9_initial_choices':len(choices)==9,'all16_candidates_retained':len(actual_lookup)==16,
        'all48_physical_replays':len(actual_rows)==48,'three_repeats_exact':True,
        'all9_bounded_cells':len(cells)==9,'bounded_task_clock_preserved':all(c['total_original_task_elapsed_s']>=data.status.elapsed_s for c in cells),
        'all_raw_roundtrips':all(c['raw_roundtrip'] for c in cells),'no_future_engine_calls':not any(guards.values()) and not any(v for c in cells for v in c['future_engine_calls'].values()),
        'model_weights_unchanged':all(weights_before[k]==tree_hash(v.model.state_dict()) for k,v in models.items()),'torch_rng_unchanged':torch.equal(rng_before,torch.get_rng_state()),
        'upstream_unchanged':before==after,'frozen_files_unchanged':all(digest(out/p)==h for p,h in frozen.items()),
        'source_bundle_unchanged':all(digest(ROOT/p)==h and digest(out/'source'/p)==h for p,h in sources.items()),
        'max_registered_intervals':physical_intervals<=registry['budget'].get('max_new_physical_intervals',registry['budget']['max_physical_intervals']),
        'pool_reserve':prior+byte_count(out)<=GIB and shutil.disk_usage(ROOT).free>=2*GIB}
    summary={'task_id':'P4-development-cost-correction','pass':bool(all(checks.values())),'checks':checks,
        'initial_choices':effects,'known_pair':pair,'bounded_cells':cells,'new_physical_intervals':physical_intervals,
        'reused_physical_intervals':500 if reuse is not None else 0,
        'total_distinct_execution_intervals_this_protocol':physical_intervals+(500 if reuse is not None else 0),
        'counterfactual_intervals':sum(r['completed_intervals'] for r in actual_rows),'bounded_closed_loop_intervals':sum(c['intervals'] for c in cells),
        'regression_tests':len(names),'future_engine_calls_initial':guards,'learning_updates':0,'final_test_used':False,
        'personal_understanding':'NOT_ASSESSED','resources':{'wall_time_s':perf_counter()-began,'bytes_before_final_summary':byte_count(out),'free_after_bytes':shutil.disk_usage(ROOT).free,'prior_pool_bytes':prior},
        'scope':'One reused source99 development state, seed11, N128 paired model-only selections; N16 actual subset, deterministic3repeats; nine bounded2s suffix policies. No from-start or independent test/generalization claim.',
        'P5_01_freeze':'HOLD_PENDING_REVIEW'}
    save(out/'summary.json',summary)
    if not summary['pass']:raise ValueError('Evidence checks failed; preserve all outcomes')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True);parser.add_argument('--registry-sha256',required=True);parser.add_argument('--registry',default=str(REGISTRY))
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id) or not re.fullmatch(r'[0-9a-f]{64}',args.registry_sha256):raise ValueError('Unsafe run id/hash')
    out=ROOT/'runs'/args.run_id;out.mkdir(parents=True,exist_ok=False)
    registry_path=Path(args.registry).resolve()
    if registry_path.parent!=ROOT/'configs':raise ValueError('Registry must remain in project configs')
    try:result=run(out,args.registry_sha256,registry_path)
    except Exception as e:
        save(out/'failure.json',{'pass':False,'error':str(e),'traceback':traceback.format_exc()});print(json.dumps({'pass':False,'error':str(e)}),flush=True);return 1
    finally:save(out/'artifact_hashes.json',{'sha256':{str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'}})
    print(json.dumps({'pass':result['pass'],'run':str(out),'new_physical_intervals':result['new_physical_intervals']}),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
