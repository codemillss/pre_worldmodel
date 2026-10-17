"""Bounded development follow-up: original H5 commands then five zero commands.

An offline fixed-sequence experiment, never an oracle controller or H10 model.
It tests settling/continuous handoff after a previously observed near-goal
candidate. Selection is post-hoc development and does not use final test.
"""
from copy import deepcopy
from dataclasses import asdict
import argparse
import gzip
import json
import re
import shutil
import subprocess
import sys
import tomllib
import traceback
import numpy as np
from push_align.config import ROOT
from push_align.data import digest, _encode_nonfinite
from push_align.candidate_replay import replay_candidate, snapshots_exact
from p2_collect_pilot import save_trace
from p3_geometry_pilot import verify_artifacts
from p4_compare_candidates import save
from p4_development_inputs import load_development_snapshot
from p4_train_forks import byte_count
from p4_verify_sequence_loss import snapshot_sources

REGISTRY=ROOT/'configs/p4_stop_diagnostic.v1.json'
REGISTRY_SHA='2a85a481c4fb07459f52db9a9b12950992c3f56446ce3ac80dcf5c76f5037acd'
MAIN=ROOT/'runs/p4-04-20261013-01'
SOURCE=ROOT/'runs/p3-05-20261010-01/full_resource/MLP-11'


def prefix_exact(previous, current):
    with gzip.open(previous/'raw_execution.json.gz','rt') as f:a=json.load(f)
    with gzip.open(current/'raw_execution.json.gz','rt') as f:b=json.load(f)
    with np.load(previous/'raw_execution.npz',allow_pickle=False) as x,np.load(current/'raw_execution.npz',allow_pickle=False) as y:
        checks={key:np.array_equal(x[key],y[key][:len(x[key])],equal_nan=True) for key in ('state','actual_dt','action_requested','action_applied','substep_states','substep_times')}
        checks['initial_vector']=np.array_equal(x['initial_integration_state'],y['initial_integration_state'])
    checks['whole_initial_snapshot']=a['initial_snapshot']==b['initial_snapshot']
    checks['all_raw_first5_diagnostics']=a['diagnostics']==b['diagnostics'][:5]
    return {'pass':bool(all(checks.values())),'checks':checks}


def run(out):
    from p4_stop_task_summary import summarize_task_prefix
    registry=json.loads(REGISTRY.read_text())
    if digest(REGISTRY)!=REGISTRY_SHA:raise ValueError('Supplement preregistration version changed')
    for p,h in registry['bindings_sha256'].items():
        if digest(ROOT/p)!=h:raise ValueError('Supplement preregistration binding changed')
    if not json.loads((MAIN/'summary.json').read_text())['pass']:raise ValueError('Main failure analysis incomplete')
    before=verify_artifacts(MAIN)
    budget=registry['budget'];free=shutil.disk_usage(ROOT).free
    if free<budget['estimated_bytes']+2*1024**3 or budget['prior_pool_bytes']+budget['estimated_bytes']>1024**3:raise ValueError('Supplement does not fit unchanged pool/free reserve')
    physical=json.loads((MAIN/'manifest.json').read_text())['physical'];planner=deepcopy(json.loads((MAIN/'manifest.json').read_text())['planner'])
    # Only cost evaluation needs its horizon to agree with the fixed sequence.
    # No candidate generator or learned planning calls use this diagnostic config.
    planner['planner']['horizon']=10;planner['planner']['block_lengths']=[2,3,5]
    profile=tomllib.loads((ROOT/'configs/acceptance.v1.toml').read_text())
    data=load_development_snapshot(SOURCE,99,physical,profile)
    with np.load(MAIN/'snapshots/tick-099/candidates.npz',allow_pickle=False) as z:base=z['all_actions'].copy()
    sequences={i:np.concatenate((base[i],np.zeros((5,2),np.float64))) for i in registry['original_candidates']}
    sources=snapshot_sources(out)
    save(out/'manifest.json',{'task_id':'P4-04-supplement','registry':registry,'registry_sha256':digest(REGISTRY),'source_hashes':sources,
        'fixed_actions':{str(i):a.tolist() for i,a in sequences.items()},'physical':physical,'diagnostic_cost_config':planner,
        'context':{'status':asdict(data.status),'previous_applied_action':data.context.previous_applied_action.tolist()},
        'new_online_controller':False,'model_inference_calls':0,'learning_updates':0,'final_test_used':False,'personal_understanding':'NOT_ASSESSED'})
    proc=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_p4_stop_task_summary.py','-v'],cwd=ROOT,capture_output=True,text=True)
    log=proc.stdout+proc.stderr;(out/'tests.txt').write_text(log);match=re.search(r'Ran (\d+) tests',log)
    if proc.returncode or not match or int(match.group(1))==0 or 'skipped=' in log:raise ValueError('Supplement pure-supervision tests failed')
    rows=[];physical_intervals=0
    for index,actions in sequences.items():
        reference=None
        for repeat in range(3):
            if shutil.disk_usage(ROOT).free<2*1024**3 or budget['prior_pool_bytes']+byte_count(out)>1024**3:raise ValueError('Supplement reserve/pool reached')
            dest=out/f'candidate-{index:03d}'/f'repeat-{repeat}';dest.mkdir(parents=True,exist_ok=False)
            result=replay_candidate(physical,planner,data.snapshot,actions,data.task,data.context.previous_applied_action,profile)
            save_trace(dest,'raw_execution',data.snapshot,result.results)
            save(dest/'final_snapshot.json',asdict(result.final_snapshot))
            task=summarize_task_prefix(result.results,data.status,data.task,.1,profile)
            proof=prefix_exact(MAIN/'snapshots/tick-099/actual'/f'candidate-{index:03d}'/f'repeat-{repeat}',dest)
            if not proof['pass']:raise ValueError('First5 original actions/raw do not reproduce main E1')
            raw_json=[]
            with gzip.open(dest/'raw_execution.json.gz','rt') as f:raw_json=json.load(f)['diagnostics']
            with np.load(dest/'raw_execution.npz',allow_pickle=False) as z:
                roundtrip=(np.array_equal(z['state'],np.asarray([r.observation.state for r in result.results],np.float64))
                    and np.array_equal(z['substep_states'],np.concatenate([r.substep_states for r in result.results]))
                    and raw_json==_encode_nonfinite([r.diagnostics for r in result.results]))
            row={'original_index':index,'repeat':repeat,'physical_valid':result.valid,'completed_intervals':result.completed_intervals,
                'first5_exact':proof,'raw_arrays_roundtrip':bool(roundtrip),'task':task,'cost_H10':result.cost,'cost_components_H10':result.cost_components,
                'cost_scope':'H10 diagnostic only; raw numeric J is not compared to the H5 objective','reasons':result.reasons}
            if reference is not None and not (np.array_equal(result.states,reference[0],equal_nan=True) and result.cost==reference[1] and snapshots_exact(result.final_snapshot,reference[2]) and task==reference[3] and raw_json==reference[4]):raise ValueError('Supplement repeats differ; preserve and investigate')
            reference=(result.states.copy(),result.cost,result.final_snapshot,deepcopy(task),raw_json)
            physical_intervals+=len(result.results);save(dest/'result.json',row);rows.append(row)
    checks={'six_declared_attempts':len(rows)==6,'all_prefixes_exact':all(r['first5_exact']['pass'] for r in rows),
        'raw_roundtrip':all(r['raw_arrays_roundtrip'] for r in rows),'three_repeats_exact':True,
        'pool_free_reserve':budget['prior_pool_bytes']+byte_count(out)<1024**3 and shutil.disk_usage(ROOT).free>=2*1024**3,
        'main_run_unchanged':before==verify_artifacts(MAIN),'sources_unchanged':all(digest(ROOT/p)==h and digest(out/'source'/p)==h for p,h in sources.items())}
    result={'task_id':'P4-04-supplement','pass':bool(all(checks.values())),'checks':checks,'results':rows,'targeted_tests':int(match.group(1)),
        'executed_physical_intervals':physical_intervals,'reexecuted_original_prefix_intervals':sum(min(5,r['completed_intervals']) for r in rows),
        'new_zero_suffix_intervals':sum(max(0,r['completed_intervals']-5) for r in rows),'model_inference_calls':0,'learning_updates':0,
        'scope':'Post-hoc development candidate11 vs original model choice60; fixed H5+zero5, no H10 learned planning/closed-loop update/generalization claim',
        'final_test_used':False,'personal_understanding':'NOT_ASSESSED'}
    save(out/'summary.json',result)
    if not result['pass']:raise ValueError('Supplement evidence incomplete')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True);a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.run_id):raise ValueError('Unsafe run id')
    out=ROOT/'runs'/a.run_id;out.mkdir(parents=True,exist_ok=False)
    try:r=run(out)
    except Exception as e:
        save(out/'failure.json',{'pass':False,'error':str(e),'traceback':traceback.format_exc()});print(json.dumps({'pass':False,'error':str(e)}),flush=True);return 1
    finally:save(out/'artifact_hashes.json',{'sha256':{str(f.relative_to(out)):digest(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='artifact_hashes.json'}})
    print(json.dumps({'pass':r['pass'],'run':str(out),'results':[(x['original_index'],x['repeat'],x['task']['first_ready_interval']) for x in r['results']]}),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
