"""Record regression output and current resource evidence without final work.

This only invokes explicitly named synthetic/read-only unit suites. It never
calls the final coordinator CLI, creates final families or grants admission.
Each invocation needs a new output stem so failed evidence is retained.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from push_align.resource_budget import estimate_storage,capacity_decision

MODULES=[
    'test_p5_final_recording_columns','test_p5_final_driver','test_p5_final_execution',
    'test_p5_final_result_review','test_p5_final_e1','test_p5_final_e1_adoption',
    'test_p5_final_visuals','test_p5_final_manifest','test_p5_final_statistics',
    'test_p5_final_e1_statistics','test_p5_final_e1_statistics_review',
    'test_p5_final_prediction','test_p5_final_stage_review','test_p5_final_orchestration',
    'test_p5_g3_e1_integration_audit','test_p5_g3_scope_audit','test_p5_g3_ledger_audit',
    'test_p5_g3_scope_followup','test_p5_resource_budget',
    'test_p5_final_fixed_budget','test_p5_fixed_budget_integration','test_p5_g3_e1_fixed_spending',
    'test_p5_fixed_budget_scope_audit',
]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def save(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stem',required=True,type=Path)
    args=parser.parse_args();stem=(ROOT/args.stem).resolve()
    if stem.parent!=ROOT/'reports':raise ValueError('A new reports/ output stem is required')
    report=Path(str(stem)+'.json');log=Path(str(stem)+'.log')
    if report.exists() or log.exists():raise FileExistsError('Never replace earlier verification evidence')
    files=sorted({*list((ROOT/'src').rglob('*.py')),*list((ROOT/'scripts').glob('*.py')),
        *list((ROOT/'tools').glob('p5*.py')),*[ROOT/'tests'/(m+'.py') for m in MODULES]})
    before={str(p.relative_to(ROOT)):sha(p) for p in files}
    original_maps={
        'scientific':read(ROOT/'configs/p4_first_moment.v1.json')['bindings_sha256'],
        'candidate':read(ROOT/'configs/p5_freeze_candidate.v1.json')['source_sha256'],
    }
    preservation={name:{'checked':len(bindings),
        'all_unchanged':all((ROOT/p).is_file() and sha(ROOT/p)==digest for p,digest in bindings.items())}
        for name,bindings in original_maps.items()}
    unique={p:digest for bindings in original_maps.values() for p,digest in bindings.items()}
    started=time.perf_counter();command=[sys.executable,'-m','unittest',*MODULES,'-q']
    env=dict(os.environ);env['PYTHONPATH']='src:scripts:tests'
    with log.open('xb') as stream:
        result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    wall=time.perf_counter()-started;output=log.read_text()
    match=re.search(r'Ran (\d+) tests in ([\d.]+)s',output)
    after={str(p.relative_to(ROOT)):sha(p) for p in files}
    plan_path=ROOT/'configs/p5_g2_resource_plan.v3.json';plan=read(plan_path)
    estimate=estimate_storage(plan['counts'],plan['units_bytes'],plan['fixed_allowances_bytes'])
    capacity=capacity_decision(shutil.disk_usage(ROOT).free,estimate['required_bytes_including_reserve'])
    checks={'test_process_exit_zero':result.returncode==0,'unittest_summary_present':match is not None,
        'reviewed_sources_unchanged_during_tests':before==after,
        'originals_unchanged':all(row['all_unchanged'] for row in preservation.values()),
        'resource_arithmetic_exact':estimate==plan['estimate'],
        'no_actual_final_manifest':not (ROOT/'runs/p5-final-20261016-01/manifest.json').exists()}
    receipt={'schema':'p5-preparation-verification-v2','checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'PASS_REGRESSION_RESOURCE_ADMISSION_SEPARATE' if all(checks.values()) else 'HOLD',
        'checks':checks,'tests':{'command':command,'modules':MODULES,'returncode':result.returncode,
        'count':int(match[1]) if match else None,'unittest_wall_s':float(match[2]) if match else None,
        'process_wall_s':wall,'output_path':str(log.relative_to(ROOT)),'output_sha256':sha(log)},
        'preservation':preservation,'original_unique_files':len(unique),
        'resource_plan_path':str(plan_path.relative_to(ROOT)),'resource_plan_sha256':sha(plan_path),
        'estimate':estimate,'capacity_snapshot':capacity,'source_sha256':before,
        'scope':'Synthetic and saved-data checks; no trained-model inference, actual physical execution, new learning, or real final family generation. Some tiny fixtures exercise frozen-file APIs with fabricated IDs.',
        'final_execution_authorized':False,'P5_01':'HOLD','overall_done':'23/28',
        'personal_understanding':'NOT_ASSESSED'}
    save(report,receipt)
    print(json.dumps({'report':str(report),'status':receipt['status'],'tests':receipt['tests']['count'],
        'capacity':capacity,'checks':checks},ensure_ascii=False))
    return 0 if all(checks.values()) else 2

if __name__=='__main__':raise SystemExit(main())
