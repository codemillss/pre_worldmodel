"""Record a scoped pilot adoption after physics and boundary prerequisites.

The candidate's original TOML is retained byte-for-byte. This sidecar is the
dated adoption decision; it does not authorize final evaluation or main data.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
from zoneinfo import ZoneInfo

import numpy as np
from push_align import InitialSpec, PlanarPushEnv
from push_align.config import CONFIG_PATH, ROOT, load_pilot_config
from p1_physics_pilot import digest, save


def read(path): return json.loads(path.read_text())


def artifact_check(run):
    checks=read(run/'artifact_hashes.json')
    return {'checked':len(checks),'mismatches':[name for name,h in checks.items() if digest(run/name)!=h]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--physics-run',type=Path,default=ROOT/'runs/p1-06-20261007-06')
    parser.add_argument('--boundary-run',type=Path,default=ROOT/'runs/p1-06-boundary-20261007-07')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=False)
    physical,boundary=read(args.physics_run/'summary.json'),read(args.boundary_run/'summary.json')
    pm,bm=read(args.physics_run/'manifest.json'),read(args.boundary_run/'manifest.json')
    config=load_pilot_config()
    profile_path=ROOT/'configs/acceptance.v1.toml'
    acceptance=tomllib.loads(profile_path.read_text())
    artifacts={'physics':artifact_check(args.physics_run),'boundary':artifact_check(args.boundary_run)}
    checks={'physics_all_required_pass':physical['stage_verdict']=='PASS' and all(v=='PASS' for v in physical['task_verdicts'].values()),
        'geometry_complete':physical['geometry_verdict']=='PASS' and len(physical['contact_geometry_checks'])==physical['expected_geometry_checks'],
        'boundary_all_declared_pass':boundary['boundary_probe_verdict']=='PASS' and boundary['passed_cases']==boundary['predeclared_cases']==24,
        'same_config':config==pm['config']==bm['config'], 'same_profile':acceptance==pm['acceptance']==bm['acceptance'],
        'original_candidate_hash_unchanged':digest(CONFIG_PATH)==pm['source_hashes'][str(CONFIG_PATH.relative_to(ROOT))],
        'source_artifacts_intact':not any(v['mismatches'] for v in artifacts.values())}
    paths=[p for folder in ('src','scripts','tests','configs') for p in (ROOT/folder).rglob('*') if p.suffix in ('.py','.toml')]
    paths += [ROOT/'assets/push_scene.xml',ROOT/'requirements.lock.txt']
    for p in paths:
        target=out/'source'/p.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,target)
    save(out/'manifest.json',{'created_at':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),'argv':sys.argv,
        'config':config,'acceptance':acceptance,
        'physical_manifest_sha256':digest(args.physics_run/'manifest.json'),
        'boundary_manifest_sha256':digest(args.boundary_run/'manifest.json'),
        'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in paths},
        'change_scope':'default config/profile dispatch and version-independent test fixtures; physical v4 config and stepping unchanged'})
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,text=True,capture_output=True)
    (out/'tests.txt').write_text(tests.stdout+tests.stderr)
    checks['current_default_regressions']=tests.returncode==0
    runtime=subprocess.run([sys.executable,str(ROOT/'scripts/p1_runtime_check.py'),'--output',str(out.resolve()/'runtime.json')],cwd=ROOT,text=True,capture_output=True)
    (out/'runtime_stdout.txt').write_text(runtime.stdout+runtime.stderr)
    checks['default_runtime']=runtime.returncode==0
    default,explicit=PlanarPushEnv(),PlanarPushEnv(config)
    checks['default_matches_explicit_compiled_model']=default._compiled_hash==explicit._compiled_hash and default.model_hash==explicit.model_hash and default.environment_hash==explicit.environment_hash
    default.reset(InitialSpec(object_yaw=np.deg2rad(10)))
    explicit.restore(default.snapshot())
    trace=[]
    action=.03*np.array([np.cos(np.deg2rad(10)),np.sin(np.deg2rad(10))])
    for _ in range(10):
        a,b=default.step(action),explicit.step(action)
        same=bool(np.array_equal(a.substep_states,b.substep_states) and np.array_equal(a.substep_times,b.substep_times)
                  and np.array_equal(default.snapshot().integration_state,explicit.snapshot().integration_state))
        trace.append({'time_s':a.observation.sim_time,'valid':a.transition_valid and b.transition_valid,'identical_microsteps_and_integration_state':same})
        if not a.transition_valid or not b.transition_valid: break
    checks['default_matches_explicit_one_second']=len(trace)==10 and all(t['valid'] and t['identical_microsteps_and_integration_state'] for t in trace)
    passed=all(checks.values())
    save(out/'summary.json',{'adoption_id':out.name,'status':'PASS' if passed else 'FAIL','created_at':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        'checks':checks,'source_artifact_checks':artifacts,'default_dispatch_trace':trace,
        'adopted_config_path':str(CONFIG_PATH),'adopted_config_id':config['config_id'],'config_sha256':digest(CONFIG_PATH),
        'model_hash':default.model_hash,'environment_hash':default.environment_hash,
        'acceptance_id':acceptance['profile_id'],'acceptance_sha256':digest(profile_path),
        'physics_run':str(args.physics_run.resolve()),'boundary_run':str(args.boundary_run.resolve()),
        'pilot_data_collection_allowed':passed,'main_data_collection_authorized':False,'final_evaluation_allowed':False,
        'scope':'15 short contrast cases and 24 selected corner pushes; current default runtime and regression checks',
        'continuous_initial_range_validity':'NOT_ASSESSED','thirty_second_episode_validity':'NOT_ASSESSED',
        'alignment_outcome':'NOT_EVALUATED','learner_understanding':'NOT_ASSESSED',
        'design_file_status_note':'Original draft status in immutable candidate/profile files records their creation state; this dated sidecar records scoped adoption.'})
    save(out/'artifact_hashes.json',{p.name:digest(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({'out':str(out),'status':'PASS' if passed else 'FAIL','checks':checks},ensure_ascii=False,indent=2))
    return 0 if passed else 1


if __name__=='__main__': raise SystemExit(main())
