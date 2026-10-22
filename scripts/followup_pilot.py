#!/usr/bin/env python3
"""Run only predeclared D2 after exact D1 gates; preserve every physical prefix."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from push_align.closed_loop import ENGINE_GUARDS, model_planning_scope
from push_align.final_driver_v2 import exact_comparison,read_full_trace,typed_digest
from push_align.final_execution import array_hash
from push_align.final_recording import file_sha256
from push_align import final_recording_columns as codec
from push_align.followup_probe import run_followup_probe
from push_align.fork_checkpoint import load_fork_checkpoint
from push_align.normalization import load_normalizer
from push_align.planning import LearnedPredictor,rollout,score,choose,contact_distance
from push_align.ridge import load_ridge
from followup_restore import write_json


def load_models():
    folder=ROOT/'artifacts/learning/normalizer';meta=json.loads((folder/'normalizer.json').read_text())['normalizer']
    norm=load_normalizer(folder,meta['schema'],meta['dataset_identity'],meta['normalizer_hash'])
    models={};bindings={}
    for path in sorted(folder.iterdir()):
        if path.is_file():bindings[path.relative_to(ROOT).as_posix()]=file_sha256(path)
    for folder in sorted((ROOT/'artifacts/models').iterdir()):
        if folder.name=='R':
            m=json.loads((folder/'ridge.json').read_text());b=load_ridge(folder,norm,m['config'],m['training_provenance'])
        else:
            m=json.loads((folder/'checkpoint.json').read_text())
            b=load_fork_checkpoint(folder,norm,m['config'],m['training_provenance'],m['fork_budget'],seed=m['seed'],branch=m['branch'],resume=False,expected_checkpoint_hash=m['checkpoint_hash'],expected_run_manifest_sha256=m['run_manifest_sha256'])
        models[folder.name]=LearnedPredictor(b.model,norm)
        for path in folder.iterdir():
            if path.is_file():bindings[path.relative_to(ROOT).as_posix()]=file_sha256(path)
    return models,bindings


def get_input(run,state):
    folder=run/'D1'/state;report=json.loads((folder/'restoration.json').read_text())
    assert report['status']=='PASS_EXACT_RESTORATION_GATE'
    tick=254 if state=='F-O0' else 78
    capture=next(c for c in report['captures'] if c['tick']==tick)
    receipt={'files_sha256':{str(folder/k):v for k,v in capture['trace_files_sha256'].items()}}
    data,_,_=read_full_trace(receipt)
    bank=folder/capture['bank'];assert file_sha256(bank)==capture['bank_sha256']
    with np.load(bank,allow_pickle=False) as source:actions=source['actions'].copy();labels=source['sources'].copy()
    assert array_hash(actions)==capture['candidate_hash'] and actions.shape==(128,5,2)
    return data,actions,labels,report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--original-root',type=Path,required=True);ap.add_argument('--run',type=Path,required=True)
    a=ap.parse_args();original=a.original_root.resolve();run=a.run.resolve();out=run/'D2';out.mkdir(exist_ok=False)
    start=time.monotonic();design_path=ROOT/'configs/followup_candidate_diagnostic.v1.json';design=json.loads(design_path.read_text())
    limits=design['resource_limits'];started=datetime.now(timezone.utc).isoformat()
    entry={'schema':'followup-d2-pilot-v1','started_at_utc':started,'design_sha256':file_sha256(design_path),
        'states':['N-T','F-O0'],'candidate_original_indices':list(range(16)),'panels':['A','B'],'physical_repeats':3,
        'model_ids':design['models'],'training_updates':0,'all_128_or_extension_authorized':False,
        'source_git_base':'6d6d1fde2793676708cdb32e1843f9162de01c85',
        'implementation_sha256':{p:file_sha256(ROOT/p) for p in ['scripts/followup_restore.py','scripts/followup_pilot.py','src/push_align/followup_probe.py']},
        'resource_caps':limits,'free_bytes_before':shutil.disk_usage(run).free,
        'scope':'Two posthoc source states and first16 original candidates. Completion gate, not a full-bank scientific conclusion. Prefix durable at candidate completion/error; not per-substep power-loss recovery.'}
    rows=[];guards=dict.fromkeys(ENGINE_GUARDS,0);predictors={};model_bindings={};max_bytes_per_interval=0.;max_seconds_per_interval=0.;actual_intervals=0
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        manifest=original/'runs/p5-final-20261017-02/manifest.json'
        assert file_sha256(manifest)==design['original_manifest_sha256']
        frozen=json.loads(manifest.read_text());sources={};inputs={}
        for state in entry['states']:
            data,bank,labels,d1=get_input(run,state);inputs[state]=(data,bank,labels)
            assert d1['design_sha256']==entry['design_sha256']
            sources.update(d1['input_sha256'])
        for rel,sha in sources.items():assert file_sha256(original/rel)==sha,'Changed original source '+rel
        entry['original_source_sha256']=sources
        assert entry['free_bytes_before']>=limits['free_space_reserve_bytes']+256*1024**2
        predictors,model_bindings=load_models();entry['model_and_normalizer_sha256']=model_bindings
        assert sorted(predictors)==sorted(design['models']),'Exact seven model identities required'
        provenance={r['public_path']:r for r in json.loads((ROOT/'artifacts/provenance/publication_files.json').read_text())['records']}
        for rel,sha in model_bindings.items():
            record=provenance[rel];source=record['source_path']
            assert record['transformation']=='byte_identical' and sha==record['public_sha256']==record['source_sha256']==frozen['runtime_source_sha256'][source]==file_sha256(original/source),'Loaded public model differs from frozen original: '+rel
        entry['loaded_public_copies_bound_to_frozen_originals']=len(model_bindings)
        entry['runtime']={'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'mujoco':__import__('mujoco').__version__,'device':'CPU','deterministic_algorithms':True,'torch_threads':1,'physical_state_dtype':'float64'}
        write_json(out/'entry.json',entry)
        for state,(data,bank,labels) in inputs.items():
            folder=out/state;folder.mkdir();actions=bank[:16].copy();snap=data['snapshot'];task=data['task_spec'];prev=data['planner_context'].previous_applied_action
            with (folder/'bank.npz').open('xb') as f:np.savez_compressed(f,actions=actions,original_ids=np.arange(16,dtype=np.int64),sources=labels[:16])
            write_json(folder/'source.json',{'full_original_bank_hash':array_hash(bank),'pilot_bank_hash':array_hash(actions),'input_typed_sha256':typed_digest(data),'elapsed_s':data['task_status'].elapsed_s,'source_controller':data['controller_id'],'tick':data['tick']})
            # Freeze model-only H5 forecasts before evaluating any actual candidate futures.
            pmeta={}
            for name,predictor in predictors.items():
                with model_planning_scope(guards):
                    pred=rollout(np.broadcast_to(data['observation'].state,(16,10)).copy(),actions,predictor,data['planner_config'])
                    costs=score(pred,actions,task,prev,data['planner_config']);idx,_,reason=choose(costs,pred,actions)
                    gap=contact_distance(pred.states,data['planner_config'])
                with (folder/f'prediction-{name}.npz').open('xb') as f:np.savez_compressed(f,states=pred.states,raw_deltas=pred.raw_deltas,valid=pred.valid,reasons=np.asarray(pred.reasons),failure_steps=pred.failure_steps,total_cost=costs.total,geometric_gap_proxy=gap,**{'cost_'+k:v for k,v in costs.components.items()})
                pmeta[name]={'valid':int(pred.valid.sum()),'selected_original_candidate_id':idx,'reason':reason,'selection_scope':'first16 only; not original128-bank policy'}
            write_json(folder/'predictions.json',pmeta)
            first_A={}
            for index in range(16):
                for panel in ('A','B'):
                    reference=None
                    for repeat in range(3):
                        free=shutil.disk_usage(run).free;used=sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
                        if free<limits['free_space_reserve_bytes']+16*1024**2 or used>limits['incremental_storage_cap_bytes'] or time.monotonic()-start>limits['wall_time_cap_s']:
                            raise RuntimeError('NOT_RUN_RESOURCE: stop before next candidate')
                        trial=folder/f'candidate-{index:03d}'/panel/f'repeat-{repeat}';trial.mkdir(parents=True,exist_ok=False)
                        write_json(trial/'entry.json',{'state':state,'candidate_id':index,'panel':panel,'repeat':repeat,'candidate_hash':array_hash(actions[index]),'started_at_utc':datetime.now(timezone.utc).isoformat()})
                        prefix=[];t=time.monotonic();result=None;failure=None
                        try:
                            result=run_followup_probe(data['physical_config'],data['planner_config'],snap,task,data['task_status'],actions[index],prev,frozen['profile'],panel=panel,on_step=lambda i,s,c:prefix.append(s))
                        except Exception as error:
                            failure={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
                        actual_intervals+=len(prefix)
                        # Seal the returned physical prefix even if validation raised.
                        rawreceipt=codec.write_raw(trial/'raw',snap,prefix,xz_preset=3)
                        write_json(trial/'raw_receipt.json',rawreceipt)
                        loaded_snapshot,loaded=codec.read_raw(trial/'raw',expected_sha256=rawreceipt['files_sha256'])
                        comp=exact_comparison({'snapshot':snap,'steps':prefix},{'snapshot':loaded_snapshot,'steps':loaded})
                        assert comp['status']=='PASS_EXACT','Raw recording round-trip mismatch'
                        if failure:
                            write_json(trial/'incomplete.json',failure);raise RuntimeError('Probe infrastructure failure; prefix preserved: '+str(trial.relative_to(run)))
                        repeat_compare='FIRST_REPEAT'
                        signature=result  # Full dataclass: raw substeps/contacts/statuses/overrides included.
                        if reference is None:reference=signature
                        else:
                            repeat_compare=exact_comparison(reference,signature)['status'];assert repeat_compare=='PASS_EXACT','UNSTABLE_REPLAY'
                        with (trial/'diagnostic.npz').open('xb') as f:np.savez_compressed(f,states=result.states,boundary_times=result.boundary_times,nominal_actions=result.nominal_actions,requested_actions=result.requested_actions,applied_actions=result.applied_actions,supervisor_overrides=result.supervisor_overrides,hold_s=result.diagnostic_hold_s,qualifies=result.diagnostic_qualifies)
                        wall=time.monotonic()-t;size=sum(p.stat().st_size for p in trial.iterdir() if p.is_file())
                        row={'state':state,'candidate_id':index,'panel':panel,'repeat':repeat,'physical_valid':result.physical_valid,'horizon_completed':result.horizon_completed,'intervals':len(result.results),'termination':result.termination,'reasons':list(result.reasons),'first_ready_interval':result.first_ready_interval,'ready_prefix':result.ready_prefix,'ready':result.ready,'ready_eligible':result.ready_eligible,'cost':result.cost if result.cost is not None and math.isfinite(result.cost) else None,'cost_finite':result.cost is not None and math.isfinite(result.cost),'cost_components':{k:v if math.isfinite(v) else None for k,v in result.cost_components.items()},'raw_round_trip':comp['status'],'repeat_comparison':repeat_compare,'wall_seconds':wall,'bytes_before_summary':size,'initial_snapshot_exact':result.initial_snapshot_exact}
                        write_json(trial/'summary.json',row);rows.append(row)
                        max_bytes_per_interval=max(max_bytes_per_interval,size/max(1,len(prefix)));max_seconds_per_interval=max(max_seconds_per_interval,wall/max(1,len(prefix)))
                        if panel=='A' and repeat==0:first_A[index]=result
                    print(f'{state} candidate{index:02d} {panel}: '+result.termination+f', ready={result.ready_eligible}',flush=True)
            # Teacher-forced one-step predictions on first-repeat actual A boundaries.
            starts=[];truth=[];cmd=[];identifiers=[];actual_valid=[]
            for index,r in first_A.items():
                for k,step in enumerate(r.results):
                    if np.isfinite(r.states[k]).all():
                        starts.append(r.states[k]);truth.append(step.observation.state);cmd.append(step.applied_action);identifiers.append((index,k));actual_valid.append(bool(step.transition_valid and step.observation.observation_valid and r.geometries[k]['pass'] and abs(step.actual_dt-.1)<=1e-9))
            starts=np.asarray(starts,np.float64);truth=np.asarray(truth,np.float64);cmd=np.asarray(cmd,np.float64)
            for name,predictor in predictors.items():
                with model_planning_scope(guards):teacher=rollout(starts,cmd[:,None,:],predictor,data['planner_config'])
                with (folder/f'teacher-{name}.npz').open('xb') as f:np.savez_compressed(f,predicted=teacher.states[:,1],raw_deltas=teacher.raw_deltas[:,0],prediction_valid=teacher.valid,reasons=np.asarray(teacher.reasons),actual=truth,actual_interval_valid=np.asarray(actual_valid,np.bool_),start=starts,applied=cmd,candidate_interval=np.asarray(identifiers,np.int64))
        for rel,sha in sources.items():assert file_sha256(original/rel)==sha,'Original changed during D2'
        for rel,sha in model_bindings.items():assert file_sha256(ROOT/rel)==sha,'Model/normalizer file changed'
        assert all(v==0 for v in guards.values())
        # D2 only estimates D3/D4 resource needs; it never starts them.
        projected_intervals=limits['all_planned_max_intervals_with_extension_restore_separate_pilot']
        remaining_intervals=max(0,projected_intervals-254-78-actual_intervals)
        used=sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
        projection={'safety_factor':1.5,'max_measured_trial_bytes_per_returned_interval':max_bytes_per_interval,'max_measured_trial_wall_seconds_per_returned_interval':max_seconds_per_interval,'conservative_remaining_intervals':remaining_intervals,'incremental_projected_total_bytes':int(used+1.5*remaining_intervals*max_bytes_per_interval),'projected_remaining_wall_seconds':1.5*remaining_intervals*max_seconds_per_interval,'free_bytes_now':shutil.disk_usage(run).free,'current_run_bytes':used,'includes_no_future_execution_authorization':True}
        projection['storage_cap_fit']=projection['incremental_projected_total_bytes']<=limits['incremental_storage_cap_bytes'] and projection['free_bytes_now']>=1.5*remaining_intervals*max_bytes_per_interval+limits['free_space_reserve_bytes']
        projection['wall_cap_fit']=projection['projected_remaining_wall_seconds']+(time.monotonic()-start)<=limits['wall_time_cap_s']
        write_json(out/'resource_projection.json',projection)
        summary={'schema':'followup-d2-pilot-summary-v1','status':'PASS_PILOT_RECORDING_AND_REPRODUCTION_GATES','rows':rows,'actual_physics_intervals':actual_intervals,'counterfactual_trajectories':len(rows),'training_updates':0,'prediction_calls':{k:{'calls':v.calls,'rows':v.rows} for k,v in predictors.items()},'future_engine_calls_during_prediction':guards,'all_original_source_bindings_rechecked':len(sources),'all_model_normalizer_files_unchanged':len(model_bindings),'wall_seconds':time.monotonic()-start,'peak_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'resource_projection':projection,'interpretation':'Pilot does not diagnose full128-bank cost optimum or search coverage; no D3 or extension executed.'}
        write_json(out/'summary.json',summary)
        print(json.dumps({k:summary[k] for k in ['status','actual_physics_intervals','counterfactual_trajectories','wall_seconds']},ensure_ascii=False),flush=True)
    except Exception as error:
        failure={'status':'STOP_INCOMPLETE_NO_AUTOMATIC_RETRY','failure_category':'repeat_fidelity' if 'UNSTABLE_REPLAY' in str(error) else 'infrastructure_or_contract','error':{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()},'completed_rows':rows,'actual_physics_intervals_returned':actual_intervals,'wall_seconds':time.monotonic()-start,'training_updates':0}
        write_json(out/'incomplete.json',failure);print(json.dumps({k:failure[k] for k in ['status','error','actual_physics_intervals_returned']},ensure_ascii=False),flush=True);sys.exit(2)

if __name__=='__main__':main()
