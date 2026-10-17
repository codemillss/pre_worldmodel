"""P3-01: frozen four-case geometry control development and acceptance.

No learned model is used or modified. All failures and tuning attempts remain
separate immutable runs. Forced withdrawal is a scheduled command-only recovery
fixture, not evidence of naturally occurring field disturbances.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import gc
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
import numpy as np
from push_align import InitialSpec, PlanarPushEnv
from push_align.closed_loop import ENGINE_GUARDS, model_planning_scope
from push_align.config import ROOT, load_pilot_config
from push_align.contact_geometry import check_contact_geometry
from push_align.coverage import _contact_events
from push_align.data import DatasetWriter, EpisodeRecorder, _encode_nonfinite, contract_hash, digest, iter_dataset, make_contract
from push_align.geometry_control import GeometryController
from push_align.geometry_control_smooth import SmoothGeometryController
from push_align.planning import TaskSpec
from push_align.task import consume_step, fail_task, handoff_metrics, pose_in_goal, start_task
from p2_collect_pilot import save_trace
from p2_episode_contract_check import source_bundle


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(_encode_nonfinite(value), f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write('\n')


def verify_artifacts(folder):
    hashes=json.loads((folder/'artifact_hashes.json').read_text())['sha256']
    for name, sha in hashes.items():
        if digest(folder/name) != sha:
            raise ValueError(f'Previous artifact changed: {folder/name}')
    return {'count':len(hashes),'manifest_sha256':digest(folder/'artifact_hashes.json')}


def case_plan(settings, config):
    expected=['g-translation','g-yaw-positive','g-yaw-negative','g-recontact']
    if len(settings['cases'])!=4 or [c['id'] for c in settings['cases']] != expected:
        raise ValueError('Frozen four-case IDs/order required')
    result=[]
    for index,c in enumerate(settings['cases']):
        yaw=float(np.deg2rad(c['object_yaw_deg']))
        normal=np.array([np.cos(yaw),np.sin(yaw)])
        xy=np.array([c['object_x'],c['object_y']],np.float64)
        distance=config['object']['size_full_xyz_m'][0]/2+config['pusher']['radius_m']+settings['initial_pusher_surface_gap_m']
        p=xy-distance*normal
        initial=InitialSpec(*xy,yaw,*p)
        result.append({'episode_id':c['id'],'episode_index':index,'family_id':f'g-development-family-{index:02d}',
                       'initial_spec':asdict(initial),'require_recontact':c['require_recontact']})
    return result


def body_coordinates(state):
    angle=.5*np.arctan2(np.sin(2*state[2]),np.cos(2*state[2]))
    d=np.array([np.cos(angle),np.sin(angle)],np.float64)
    t=np.array([-d[1],d[0]],np.float64)
    relative=state[6:8]-state[:2]
    return d,float(relative@d),float(relative@t)


def recovery_evidence(results, records, settings, config):
    e=config['evaluation']
    events=_contact_events([r.diagnostics for r in results],e['contact_active_normal_force_min_N'],1e-9,
                           inactive_min_s=e['recontact_absence_min_s'],active_min_s=e['recontact_presence_min_s'])
    withdrawal=[r for r in records if r['source']=='scheduled_withdrawal']
    matched=[]
    if withdrawal:
        first=withdrawal[0]['interval_start_time'];last=withdrawal[-1]['sim_time']
        for event in events['qualified_recontacts']:
            if first-1e-9<=event['contact_loss_time_s']<=last+1e-9 and event['contact_resume_time_s']>=last-1e-9:
                matched.append(event)
    resume_samples=[]
    face_budget=0.001  # predeclared observed rear-centerline classification tolerance
    # This budget is a named geometric classification tolerance only. Actual
    # penetration remains supervised by the unchanged physical acceptance.
    for event in matched:
        for step in results:
            for sample in step.diagnostics['samples']:
                if event['contact_resume_time_s']-1e-9<=sample['sim_time']<=event['contact_resume_time_s']+e['recontact_presence_min_s']+1e-9:
                    active=any(c['pair']=='pusher_object' and c['normal_force_N']>e['contact_active_normal_force_min_N'] for c in sample['contacts'])
                    if active:
                        _,sx,sy=body_coordinates(np.asarray(sample['state'],np.float64))
                        resume_samples.append({'time':sample['sim_time'],'qx_m':sx,'qy_m':sy})
    resumed_on_rear_face=bool(resume_samples) and all(abs(x['qx_m']+config['object']['size_full_xyz_m'][0]/2+config['pusher']['radius_m'])<=face_budget and abs(x['qy_m'])<=config['object']['size_full_xyz_m'][1]/2 for x in resume_samples)
    same_face=all(r['body_qx_before_m']<0 and abs(r['body_qy_before_m'])<=config['object']['size_full_xyz_m'][1]/2
                  for r in records)
    return {'withdrawal_intervals':len(withdrawal),'scheduled_withdrawal_complete':len(withdrawal)==settings['withdrawal_intervals'],
            'matching_qualified_recontacts':matched,'same_rear_face_at_all_action_starts':same_face,
            'resumed_active_samples_on_rear_face':resumed_on_rear_face,'resume_face_samples':resume_samples,
            'rear_face_classification_tolerance_m':face_budget,
            'events':events,'scope':settings['recontact_scope']}


def run_case(folder, env, controller, case, task, profile, settings, contract, writer, guards):
    folder.mkdir(parents=True)
    observed=env.reset(InitialSpec(**case['initial_spec']));snapshot=env.snapshot()
    status=start_task(observed,task,env.action_dt);ctx=controller.initial_context()
    recorder=EpisodeRecorder(env,episode_id=case['episode_id'],episode_index=case['episode_index'],family_id=case['family_id'],
              family_definition={'base_initial_spec':case['initial_spec'],'variant_rule':'fixed development case; all tuning repetitions same family'},
              split='smoke-test',collection_policy={'kind':'geometry_feedback','purpose':'pilot','controller':'G',
              'recovery_fixture':settings['recontact_scope'] if case['require_recontact'] else None,
              'controller_parameters':controller.parameters},seed=0,contract=contract) if writer else None
    records=[];results=[];push_count=0;withdrawn=False;withdrawal_remaining=0;withdrawal_command=None
    planning_times=[];clock=perf_counter()
    for index in range(settings['max_action_intervals']):
        if status.phase in ('ready','failed'):break
        holding=pose_in_goal(observed.state,task)
        before=observed.state.copy();normal,qx,qy=body_coordinates(before)
        info=None;tick=perf_counter()
        if withdrawal_remaining:
            source='scheduled_withdrawal';command=withdrawal_command.copy();withdrawal_remaining-=1
        elif case['require_recontact'] and not withdrawn and push_count>=settings['recontact_trigger_push_intervals']:
            withdrawn=True;withdrawal_command=-settings['withdrawal_speed_m_s']*normal
            withdrawal_remaining=settings['withdrawal_intervals']-1
            source='scheduled_withdrawal';command=withdrawal_command.copy()
        elif holding:
            source='holding';command=np.zeros(2,np.float64);ctx=controller.initial_context()
        else:
            source='geometry'
            with model_planning_scope(guards):decision=controller.propose(before.copy(),task,ctx,status.elapsed_s)
            info=decision.diagnostics;ctx=decision.context;command=decision.action
            if not info['feasible']:
                status=fail_task(status,'controller_infeasible');save(folder/'unexecuted_final_decision.json',info);break
        planning_times.append(perf_counter()-tick)
        step=recorder.step(command) if recorder else env.step(command)
        results.append(step)
        geometry=recorder.geometry[-1] if recorder else check_contact_geometry(step.diagnostics['samples'],profile['physics_suite']['actual_contact_penetration_max_m'])
        status=consume_step(status,step,task,env.action_dt)
        if not geometry['pass'] and status.terminal_reason!='physics_error':status=fail_task(status,'geometry_invalid')
        observed=step.observation
        if source=='geometry' and ctx.mode=='PUSH':push_count+=1
        if source=='scheduled_withdrawal' and not withdrawal_remaining:ctx=controller.initial_context()
        row={'index':index,'source':source,'mode':ctx.mode,'controller':info,'context':asdict(ctx),
             'requested_action':command.tolist(),'applied_action':step.applied_action.tolist(),'state_before':before.tolist(),
             'state_after':observed.state.tolist(),'interval_start_time':step.observation.sim_time-step.actual_dt,
             'sim_time':step.observation.sim_time,'actual_dt_s':step.actual_dt,'physical_valid':bool(step.transition_valid),
             'geometry':geometry,'task':asdict(status),'metrics':handoff_metrics(observed.state,task) if np.isfinite(observed.state).all() else None,
             'body_qx_before_m':qx,'body_qy_before_m':qy,'decision_wall_time_s':planning_times[-1]}
        records.append(row)
        with (folder/'steps.jsonl').open('a') as f:f.write(json.dumps(_encode_nonfinite(row),ensure_ascii=False,allow_nan=False)+'\n')
    if status.phase not in ('ready','failed'):raise ValueError('Action budget ended without terminal task outcome')
    recovery=recovery_evidence(results,records,settings,env.config)
    if recorder and results:
        episode=recorder.finish(status.terminal_reason);episode.metadata['supervised_task_status']=asdict(status);episode.metadata['task_outcome']=status.phase
        writer.append(episode);valid_count=int(episode.arrays['transition_valid'].sum());del episode
    else:
        save_trace(folder,'raw_execution',snapshot,results);valid_count=sum(r.transition_valid for r in results)
    complete=status.phase=='ready' and (not case['require_recontact'] or (recovery['scheduled_withdrawal_complete'] and recovery['matching_qualified_recontacts'] and recovery['same_rear_face_at_all_action_starts'] and recovery['resumed_active_samples_on_rear_face']))
    entry={'episode_id':case['episode_id'],'task':asdict(status),'final_metrics':handoff_metrics(observed.state,task),
           'transitions':len(results),'valid_transitions':valid_count,'holding_intervals':sum(r['source']=='holding' for r in records),
           'recovery':recovery,'case_complete':bool(complete),'wall_time_s':perf_counter()-clock,'decision_count':len(planning_times),
           'decision_mean_s':float(np.mean(planning_times)) if planning_times else None,'mode_counts':dict(Counter(r['mode'] for r in records))}
    save(folder/'summary.json',entry);gc.collect()
    return entry,planning_times


def run(args,out):
    started=perf_counter();settings=tomllib.loads(args.cases.read_text());config=load_pilot_config()
    frozen=json.loads((ROOT/'reports/p3_01_cases_frozen.json').read_text())
    if digest(args.cases)!=frozen['case_file_sha256'] or settings!=frozen['settings']:raise ValueError('Frozen cases changed')
    controller_config=deepcopy(config)
    parameters=json.loads(args.parameters.read_text()) if args.parameters else {}
    if set(parameters)-set(config['geometry_baseline']):raise ValueError('Only explicit G parameters may be tuned')
    controller_config['geometry_baseline'].update(parameters)
    transition_path=getattr(args,'transition_parameters',None)
    transition_parameters=json.loads(transition_path.read_text()) if transition_path else None
    controller=SmoothGeometryController(controller_config,transition_parameters) if transition_parameters else GeometryController(controller_config)
    task=TaskSpec.from_config(config);env=PlanarPushEnv(config);contract=make_contract(env)
    profile=tomllib.loads((ROOT/'configs/acceptance.v1.toml').read_text());cases=case_plan(settings,config)
    prior_dir=ROOT/'runs/p2-06-20261008-01';prior=verify_artifacts(prior_dir) if not args.development else {'scope':'previous run not touched; formal runner verifies all artifacts'}
    source_hashes=source_bundle(out)
    for extra in (args.parameters,transition_path):
        if extra:
            rel=extra.resolve().relative_to(ROOT);dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(extra,dest);source_hashes[str(rel)]=digest(extra)
    manifest={'task_id':'P3-01','mode':'development_tuning' if args.development else 'formal_four_case_acceptance',
              'created_at':datetime.now().astimezone().isoformat(),'argv':sys.argv,'python':platform.python_version(),'platform':platform.platform(),
              'settings':settings,'cases':cases,'cases_sha256':digest(args.cases),'case_freeze_sha256':digest(ROOT/'reports/p3_01_cases_frozen.json'),
              'physical_config':config,'resolved_controller_parameters':controller_config['geometry_baseline'],
              'requested_parameter_file_sha256':digest(args.parameters) if args.parameters else None,
              'transition_parameters':transition_parameters,'transition_parameters_sha256':digest(transition_path) if transition_path else None,
              'change_reason':args.reason,'previous_attempt':args.previous_attempt,'source_hashes':source_hashes,
              'profile_sha256':digest(ROOT/'configs/acceptance.v1.toml'),'contract_hash':contract_hash(contract),
              'previous_p2_06_artifacts':prior,'scope':'four fixed development feasibility cases; no final-test or comparative superiority result',
              'information_access':'state10/task/own mode and elapsed only; contact diagnostics and friction not controller inputs'}
    save(out/'cases_frozen.json',frozen)
    save(out/'controller_parameters_requested.json',parameters)
    freeze_sha=digest(ROOT/'reports/p3_01_cases_frozen.json')
    save(out/'manifest.json',manifest);manifest_sha=digest(out/'manifest.json')
    tests_count=0;tests_pass=False
    if not args.development:
        tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,capture_output=True,text=True)
        text=tests.stdout+tests.stderr;(out/'tests.txt').write_text(text);m=re.search(r'Ran (\d+) tests',text)
        tests_count=int(m.group(1)) if m else 0;tests_pass=tests.returncode==0 and tests_count>0 and 'skipped=' not in text
        if not tests_pass:raise ValueError('Mandatory regression tests failed')
    writer=None if args.development else DatasetWriter(out/'data'/'geometry-feasibility-v1',contract,purpose='pilot')
    guards=dict.fromkeys(ENGINE_GUARDS,0);ledger=[];times=[]
    for case in cases:
        row,t=run_case(out/'episodes'/case['episode_id'],env,controller,case,task,profile,settings,contract,writer,guards)
        ledger.append(row);times+=t;print(json.dumps({'episode_id':row['episode_id'],'case_complete':row['case_complete'],'terminal':row['task']['terminal_reason'],
           'e_pos_mm':row['final_metrics']['e_pos_m']*1000,'e_yaw_deg':np.rad2deg(row['final_metrics']['e_yaw_rad']),'intervals':row['transitions']}),flush=True)
    reviews=[]
    if writer:
        writer.finish()
        for episode in iter_dataset(writer.path,contract):reviews.append({'episode_id':episode.metadata['episode_id'],'transitions':episode.transition_count,'valid':int(episode.arrays['transition_valid'].sum())})
    checks={'four_fixed_cases_accounted':[r['episode_id'] for r in ledger]==[c['episode_id'] for c in cases],
            'four_cases_ready':all(r['task']['phase']=='ready' for r in ledger),'four_case_requirements':all(r['case_complete'] for r in ledger),
            'future_engine_queries_zero':sum(guards.values())==0,'frozen_cases_unchanged':digest(args.cases)==frozen['case_file_sha256'],
            'case_freeze_record_unchanged':digest(ROOT/'reports/p3_01_cases_frozen.json')==freeze_sha,
            'transition_parameters_unchanged':not transition_path or digest(transition_path)==manifest['transition_parameters_sha256'],
            'requested_parameters_unchanged':not args.parameters or digest(args.parameters)==manifest['requested_parameter_file_sha256'],
            'manifest_unchanged':digest(out/'manifest.json')==manifest_sha,
            'controller_source_frozen':digest(ROOT/'src/push_align/geometry_control.py')==source_hashes['src/push_align/geometry_control.py'],
            'no_unresolved_physics':not any(r['task']['terminal_reason'] in ('physics_error','geometry_invalid') for r in ledger)}
    if not args.development:
        checks.update({'regression_tests_pass':tests_pass,'four_raw_episodes_audited':len(reviews)==4,
                       'source_frozen':all(digest(ROOT/p)==h for p,h in source_hashes.items()),'prior_artifacts_unchanged':verify_artifacts(prior_dir)==prior})
    resources={'total_wall_time_s':perf_counter()-started,'main_peak_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024),
               'decision_count':len(times),'decision_mean_s':float(np.mean(times)),'decision_p95_s':float(np.percentile(times,95)),
               'scope':'local CPU; proposal/holding/fixture branches separately in trace; subprocess test memory excluded; not real-device real-time guarantee'}
    summary={'task_id':'P3-01','mode':manifest['mode'],'pass':all(checks.values()),'checks':checks,'tests_count':tests_count,'ledger':ledger,
             'terminal_counts':dict(Counter(r['task']['terminal_reason'] for r in ledger)),'future_engine_calls':guards,'dataset_review':reviews,
             'resources':resources,'alignment_scope':'four fixed G development feasibility cases; learned-model outcome unchanged','user_understanding':'NOT_ASSESSED'}
    save(out/'summary.json',summary);save(out/'resources.json',resources)
    save(out/'artifact_hashes.json',{'sha256':{str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    if not args.development and not summary['pass']:raise ValueError('P3-01 acceptance incomplete; immutable results preserved')
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--development',action='store_true')
    p.add_argument('--cases',type=Path,default=ROOT/'configs/p3_geometry_cases.v1.toml');p.add_argument('--parameters',type=Path)
    p.add_argument('--transition-parameters',type=Path);p.add_argument('--reason',required=True);p.add_argument('--previous-attempt',default=None);args=p.parse_args()
    if not re.fullmatch(r'p3-01(?:-tuning)?-[0-9]{8}-[0-9]{2}',args.run_id):raise ValueError('Invalid run ID')
    out=ROOT/'runs'/args.run_id;out.mkdir(parents=True,exist_ok=False)
    try:s=run(args,out)
    except Exception as error:
        save(out/'failure.json',{'error_type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()});raise
    print(json.dumps({'run_id':args.run_id,'pass':s['pass']}),flush=True)

if __name__=='__main__':main()
