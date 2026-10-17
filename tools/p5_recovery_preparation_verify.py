"""Recheck original and V2 synthetic contracts without final work or mutation."""
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from push_align.resource_budget import estimate_storage,capacity_decision

spec=importlib.util.spec_from_file_location('original_preparation_tool',ROOT/'tools/p5_preparation_verify.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
MODULES=original.MODULES+['test_p5_final_driver_v2','test_p5_final_recovery_manifest',
                        'test_p5_final_recovery_orchestration']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stem',required=True,type=Path)
    stem=(ROOT/parser.parse_args().stem).resolve()
    if stem.parent!=ROOT/'reports':raise ValueError('New report stem required')
    report=Path(str(stem)+'.json');log=Path(str(stem)+'.log')
    if report.exists() or log.exists():raise FileExistsError('Keep every earlier check')
    files=sorted({*list((ROOT/'src').rglob('*.py')),*list((ROOT/'scripts').glob('*.py')),
        *list((ROOT/'tools').glob('p5*.py')),*[ROOT/'tests'/(m+'.py') for m in MODULES]})
    before={str(p.relative_to(ROOT)):sha(p) for p in files}
    auth=read(ROOT/'configs/p5_infrastructure_recovery.v1.json')
    parent=ROOT/auth['parent_run'];preservation=read(ROOT/auth['parent_preservation_receipt'])
    originals={**read(ROOT/'configs/p4_first_moment.v1.json')['bindings_sha256'],
               **read(ROOT/'configs/p5_freeze_candidate.v1.json')['source_sha256']}
    all_protected={**originals,**preservation['files_sha256']}
    protected_changed=[p for p,h in all_protected.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    parent_source=read(parent/'manifest.json')
    parent_bindings={**parent_source['source_sha256'],**parent_source['runtime_source_sha256']}
    parent_bound_changed=[p for p,h in parent_bindings.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    inventory=read(ROOT/'reports/p5_runtime_environment_inventory.v1.json')
    installed={r['name']:importlib.metadata.version(r['name']) for r in inventory['installed_distributions']}
    packages_match=all(installed[r['name']]==r['version'] for r in inventory['installed_distributions'])
    command=[sys.executable,'-B','-m','unittest',*MODULES,'-q']
    env=dict(os.environ);env['PYTHONPATH']='src:scripts:tests'
    with log.open('xb') as stream:
        completed=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    output=log.read_text();match=re.search(r'Ran (\d+) tests in ([\d.]+)s',output)
    after={str(p.relative_to(ROOT)):sha(p) for p in files}
    plan=read(ROOT/'configs/p5_g2_resource_plan.v3.json')
    estimate=estimate_storage(plan['counts'],plan['units_bytes'],plan['fixed_allowances_bytes'])
    capacity=capacity_decision(shutil.disk_usage(ROOT).free,estimate['required_bytes_including_reserve'])
    protected_after=[p for p,h in all_protected.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    checks={'regression_exit_zero':completed.returncode==0,'test_summary_present':match is not None,
        'reviewed_sources_unchanged':before==after,'originals_and_parent_unchanged':not (protected_changed or protected_after),
        'parent_frozen_graph_still_valid':not parent_bound_changed,
        'python_os_packages_match':packages_match and sys.version==inventory['python'] and platform.machine()==inventory['machine'] and platform.system()==inventory['system'],
        'resource_arithmetic_unchanged':estimate==plan['estimate'],
        'new_final_run_not_created':not (ROOT/auth['new_run']).exists()}
    record={'schema':'p5-recovery-preparation-verification-v1','checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'PASS_SYNTHETIC_RECOVERY_PREPARATION' if all(checks.values()) else 'HOLD','checks':checks,
        'tests':{'command':command,'modules':MODULES,'count':int(match[1]) if match else None,
                 'wall_s':float(match[2]) if match else None,'returncode':completed.returncode,
                 'log_path':str(log.relative_to(ROOT)),'log_sha256':sha(log)},
        'original_unique_paths':len(originals),'parent_file_count':len(preservation['files_sha256']),
        'protected_changed':protected_changed,'protected_changed_after':protected_after,
        'parent_source_changed':parent_bound_changed,'source_sha256':before,
        'installed_distributions':installed,'capacity_snapshot':capacity,
        'resource_plan_sha256':sha(ROOT/'configs/p5_g2_resource_plan.v3.json'),
        'scope':'Original and V2 pure/synthetic tests; zero new trained-model calls, physics, learning or final family generation.',
        'new_final_execution_authorized':False,'personal_understanding':'NOT_ASSESSED'}
    with report.open('x') as stream:json.dump(record,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'report':str(report),'status':record['status'],'checks':checks,
                      'tests':record['tests']['count'],'capacity':capacity},ensure_ascii=False))
    return 0 if all(checks.values()) else 2

if __name__=='__main__':raise SystemExit(main())
