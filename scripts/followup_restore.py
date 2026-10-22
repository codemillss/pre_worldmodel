#!/usr/bin/env python3
"""Exact recorded-command restoration gate for separate post-freeze diagnostics.

No model is loaded. Original selected indices are replay inputs, not fresh
policy decisions. Prefix physics and present-state candidate/context generation
are reproduced; any mismatch stops before counterfactual candidate futures.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from push_align.candidates import generate_candidates
from push_align.closed_loop import ENGINE_GUARDS, consume_actual_step, model_planning_scope
from push_align.contact_geometry import check_contact_geometry
from push_align.data import _encode_nonfinite
from push_align.environment import PlanarPushEnv
from push_align.final_driver_v2 import exact_comparison, read_full_trace, write_full_trace, typed_digest
from push_align.final_execution import array_hash
from push_align.final_recording import file_sha256
from push_align.final_recording_columns import read_raw
from push_align.planning import PlannerContext
from push_align.planning_protocol import GeometryProposalAdapter, keyed_planner_rng
from push_align.task import pose_in_goal, start_task

STATES = {'F-O0':('p5-final-offset-00','S1-23',254,(246,)),
          'N-T':('p5-final-translation-03','S1-11',78,())}


def write_json(path, value):
    with Path(path).open('x',encoding='utf-8') as out:
        json.dump(value,out,ensure_ascii=False,indent=2,allow_nan=False)
        out.write('\n');out.flush();os.fsync(out.fileno())


def portable(path, original):
    p=Path(path).resolve()
    return unicodedata.normalize('NFC',p.relative_to(original).as_posix())


def checked_equal(label,left,right):
    c=exact_comparison(left,right)
    if c['status'] != 'PASS_EXACT':
        raise RuntimeError(label+': '+json.dumps(c))
    return c


def load_trace(cell, manifest, tick):
    receipt=next(v for v in manifest['full_traces'] if v['tick']==tick)
    return read_full_trace(receipt)


def regenerate(captured, observed, status, context, gctx, adapter, tick, guards):
    keys=[captured['candidate_seed_base'],captured['episode_index'],tick,captured['candidate_stream_id']]
    with model_planning_scope(guards):
        proposal=adapter.propose(observed.state.copy(),captured['task_spec'],gctx,status.elapsed_s,control_tick=tick)
        bank=generate_candidates(captured['planner_config'],keyed_planner_rng(*keys),
            previous_actions=context.optimal_actions,geometry_action=proposal.geometry_action,
            tangent_direction=proposal.tangent_direction)
    return proposal,bank,keys


def restored_input(template,env,observed,status,context,gctx,tick):
    result=deepcopy(template)
    result.update(tick=tick,snapshot=env.snapshot(),observation=deepcopy(observed),
                  task_status=deepcopy(status),planner_context=deepcopy(context),geometry_context=deepcopy(gctx),
                  rng_keys=[template['candidate_seed_base'],template['episode_index'],tick,template['candidate_stream_id']])
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--original-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--state',choices=STATES,default='F-O0')
    args=ap.parse_args(); original=args.original_root.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic(); report={'schema':'followup-exact-restoration-v1','state':args.state,
        'status':'INCOMPLETE','started_at_utc':datetime.now(timezone.utc).isoformat(),
        'comparison_policy':'Existing typed bit-exact digest AND recursive equality; no tolerance relaxation.',
        'policy_scope':'Recorded commands and selected indices; current-state geometry, banks and contexts regenerated. Not fresh learned policy reproduction.',
        'physics_interval_attempts':0,'physics_intervals_exact':0,'new_model_calls':0,'training_updates':0,
        'counterfactual_future_intervals':0,'future_engine_calls_during_generation':dict.fromkeys(ENGINE_GUARDS,0)}
    bound={}; checks={};captures=[]
    try:
        design=ROOT/'configs/followup_candidate_diagnostic.v1.json'
        mpath=original/'runs/p5-final-20261017-02/manifest.json'
        protocol=json.loads(design.read_text()); m=json.loads(mpath.read_text())
        assert file_sha256(mpath)==protocol['original_manifest_sha256'],'Frozen manifest SHA'
        family,controller,target,extra=STATES[args.state]
        cell=mpath.parent/'E2'/family/controller
        cm=json.loads((cell/'manifest.json').read_text())
        # Bound scientific/runtime originals and the immutable public code copy.
        for rel,sha in m['runtime_source_sha256'].items():
            path=original/rel
            assert file_sha256(path)==sha,'Original runtime binding: '+rel
            bound[portable(path,original)]=sha
            if rel.startswith('src/'):
                assert file_sha256(ROOT/rel)==sha,'Public runtime source differs: '+rel
        for path,sha in cm['files_sha256'].items():
            assert file_sha256(path)==sha,'Original cell binding'
            bound[portable(path,original)]=sha
        bound[portable(mpath,original)]=file_sha256(mpath)
        bound[portable(cell/'manifest.json',original)]=file_sha256(cell/'manifest.json')
        witness=json.load(gzip.open(cell/'witness.json.gz','rt'))
        bound[portable(cell/'witness.json.gz',original)]=file_sha256(cell/'witness.json.gz')
        initial,_,_=load_trace(cell,cm,0)
        expected,expected_arrays,expected_meta=load_trace(cell,cm,target) if args.state=='F-O0' else (None,None,None)
        raw_initial,raw=read_raw(cm['raw']['prefix'],expected_sha256=cm['raw']['files_sha256'])
        assert len(raw)>=target
        checks['initial_raw_full_trace']=checked_equal('initial snapshot raw/fulltrace',initial['snapshot'],raw_initial)
        assert shutil.disk_usage(out).free>=protocol['resource_limits']['free_space_reserve_bytes']+256*1024**2
        report['input_sha256']=bound.copy();report['design_sha256']=file_sha256(design)
        report['implementation_sha256']=file_sha256(Path(__file__))
        report['runtime']={'python':sys.version,'numpy':np.__version__,'mujoco':__import__('mujoco').__version__,'device':'CPU','physics_and_recording_precision':'float64'}
        write_json(out/'entry.json',report)
        env=PlanarPushEnv(deepcopy(initial['physical_config']))
        observed=env.restore(deepcopy(raw_initial))
        checks['initial_restore']=checked_equal('initial restore snapshot',raw_initial,env.snapshot())
        checked_equal('initial observed',initial['observation'],observed)
        task=initial['task_spec'];dt=initial['physical_config']['physics']['action_dt_s']
        status=start_task(observed,task,dt);context=PlannerContext.initial()
        adapter=GeometryProposalAdapter(initial['physical_config'],initial['geometry_parameters'],initial['transition_parameters'])
        gctx=adapter.initial_context()
        capture_ticks=set(extra)|{target}
        with (out/'prefix_comparisons.jsonl').open('x') as log:
            for tick in range(target+1):
                d=witness['decisions'][tick]
                assert d['index']==tick,'Witness decision index'
                checked_equal('task_before '+str(tick),_encode_nonfinite(asdict(status)),d['task_status'])
                checked_equal('planner_before '+str(tick),_encode_nonfinite(asdict(context)),d['planner_context_before'])
                checked_equal('geometry_before '+str(tick),_encode_nonfinite(asdict(gctx)),d['geometry_context_before'])
                checked_equal('state_before '+str(tick),observed.state,np.asarray(d['state'],np.float64))
                proposal,bank,keys=regenerate(initial,observed,status,context,gctx,adapter,tick,report['future_engine_calls_during_generation'])
                checked_equal('geometry_proposal '+str(tick),_encode_nonfinite(proposal.diagnostics),d['geometry_proposal'])
                if d['plan_identity'] is not None:
                    assert keys==d['rng_keys'] and array_hash(bank.actions)==d['plan_identity']['candidate_hash'], 'Bank mismatch tick '+str(tick)
                if tick in capture_ticks:
                    captured=restored_input(initial,env,observed,status,context,gctx,tick)
                    receipt=write_full_trace(out/f'tick-{tick:03d}',tick=tick,captured_input=captured)
                    with (out/f'bank-{tick:03d}.npz').open('xb') as f:np.savez_compressed(f,actions=bank.actions,sources=np.asarray(bank.sources))
                    captures.append({'tick':tick,'candidate_hash':array_hash(bank.actions),'input_typed_sha256':typed_digest(captured),'trace':f'tick-{tick:03d}','bank':f'bank-{tick:03d}.npz','bank_sha256':file_sha256(out/f'bank-{tick:03d}.npz'),'trace_files_sha256':{Path(k).relative_to(out).as_posix():v for k,v in receipt['files_sha256'].items()}})
                if tick==target:
                    checks['final_observation_witness']=checked_equal('final witness observation',observed.state,np.asarray(d['state'],np.float64))
                    report['candidate_hash']=array_hash(bank.actions);report['rng_keys']=keys
                    if expected is not None:
                        for key in ('snapshot','observation','task_status','planner_context','geometry_context'):
                            checks[key]=checked_equal('target '+key,captured[key],expected[key])
                        checks['complete_typed_input']=checked_equal('target complete typed input',captured,expected)
                        checks['candidate_array']=checked_equal('target candidates',bank.actions,expected_arrays['candidates'])
                        assert report['candidate_hash']==d['plan_identity']['candidate_hash']
                        report['original_input_typed_sha256']=expected_meta['input_exact_sha256']
                    else:
                        report['full_snapshot_scope']='Newly reconstructed internal snapshot; no original full snapshot exists at this tick. Raw StepResults and all available witness/context/bank identities checked exactly.'
                    break
                record=witness['intervals'][tick]
                assert record['index']==tick,'Witness interval index'
                holding=pose_in_goal(observed.state,task)
                assert holding is d['holding']
                if holding:
                    action=np.zeros(2,np.float64);planned_context=context
                else:
                    index=d['plan_identity']['selected_index'];assert index is not None
                    warm=bank.actions[index].copy();action=warm[0].copy()
                    checked_equal('selected sequence '+str(tick),warm,np.asarray(d['plan_identity']['selected_actions'],np.float64))
                    planned_context=PlannerContext(action.copy(),warm,context.control_step+1)
                checked_equal('command '+str(tick),action,raw[tick].applied_action)
                report['physics_interval_attempts']+=1
                fresh=env.step(action.copy())
                comp=checked_equal('raw_step '+str(tick),fresh,raw[tick])
                report['physics_intervals_exact']+=1
                geom=check_contact_geometry(fresh.diagnostics['samples'],initial['penetration_budget_m'])
                checked_equal('geometry result '+str(tick),geom,record['geometry'])
                gctx=adapter.commit(proposal,actual_control_tick=tick)
                status,context=consume_actual_step(status,fresh,task,dt,planned_context,holding=holding,geometry=geom)
                observed=fresh.observation
                for label,left,right in [('task',_encode_nonfinite(asdict(status)),record['task_after']),('planner',_encode_nonfinite(asdict(context)),record['planner_context_after']),('geometry',_encode_nonfinite(asdict(gctx)),record['geometry_context_after'])]:
                    checked_equal(label+' after '+str(tick),left,right)
                log.write(json.dumps({'tick':tick,'step_typed_sha256':comp['left_sha256'],'candidate_hash':array_hash(bank.actions),'context_task_geometry':'PASS_EXACT'})+'\n');log.flush()
                if (tick+1)%50==0:print(f'{args.state}: {tick+1}/{target} exact raw intervals',flush=True)
        for rel,sha in bound.items(): assert file_sha256(original/rel)==sha,'Source changed during restoration'
        report.update(status='PASS_EXACT_RESTORATION_GATE',checks=checks,captures=captures,
            original_files_rehashed_unchanged=len(bound),raw_substeps_checked=sum(len(r.substep_times) for r in raw[:target]),
            future_engine_generation_guard_pass=all(v==0 for v in report['future_engine_calls_during_generation'].values()))
    except Exception as error:
        report.update(status='STOP_RESTORATION_OR_INFRASTRUCTURE_MISMATCH',checks=checks,captures=captures,
                      error={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
    report['wall_seconds']=time.monotonic()-start
    report['free_bytes_after']=shutil.disk_usage(out).free
    write_json(out/'restoration.json',report)
    print(json.dumps({k:report.get(k) for k in ('state','status','physics_intervals_exact','raw_substeps_checked','candidate_hash','wall_seconds','error')},ensure_ascii=False),flush=True)
    if report['status']!='PASS_EXACT_RESTORATION_GATE':sys.exit(2)

if __name__=='__main__':main()
