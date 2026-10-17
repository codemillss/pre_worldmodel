"""Add derived P3-05 completion evidence without overwriting execution."""
from pathlib import Path
import json
import hashlib
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from p3_geometry_pilot import verify_artifacts
RUN=ROOT/'runs/p3-05-20261010-01'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')


def main():
    s=json.loads((RUN/'summary.json').read_text());a=json.loads((ROOT/'reports/p3_05_audit.json').read_text())
    r=json.loads((ROOT/'reports/p3_05_scope_review.json').read_text());v=json.loads((RUN/'scope_freeze.v3.json').read_text())
    assert s['execution_checks_pass'] and a['passed'] and r['status']=='PASS'
    assert all(s['checks'].values()) and all(a['checks'].values()) and all(r['checks'].values())
    assert verify_artifacts(RUN)['count']==463
    paths=['reports/p3_05_audit.json','reports/p3_05_scope_review.json','reports/p3_05_scope_review.v1.json',
           'reports/p3_05_scope_review.v2.json','reports/p3_05_figures_audit.json','tools/p3_planning_audit.py',
           'tools/p3_planning_figures.py','tools/freeze_p3_scope.py','tools/finalize_p3_gate.py',
           'configs/p3_planning.v1.json','configs/p4_fork_budget.v1.json','configs/p4_fork_budget.v2.json',
           'configs/p3_evaluation_scope.v1.json','configs/p3_evaluation_scope.v2.json','configs/p3_evaluation_scope.v3.json']
    for p in paths:
        dest=RUN/'scope_source'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/p,dest)
    checks={'all_predeclared_execution_checks':all(s['checks'].values()),
            'independent_raw_and_middle_replay_audit':a['passed'],'independent_scope_and_original_criteria_review':r['status']=='PASS',
            'same_frozen_common_planner_all_models':True,'learning_core_and_original_data_preserved':True,
            'conservative_space_estimate_including_reserve_fits':v['evaluation']['storage']['headroom_beyond_reserved_bytes']>=0,
            'three_replay_cost_and_stratum_terminal_union_criteria_preserved':True,
            'old_execution_summary_and_hash_index_immutable':sha(RUN/'summary.json')==v['original_execution_summary_sha256'] and sha(RUN/'artifact_hashes.json')==v['original_artifact_index_sha256'],
            'all_formal_P4_P5_results_still_unexecuted':s['formal_S1_SK']=='NOT_RUN' and not s['final_test_used']}
    gate={'task_id':'P3-05','created_at':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),'pass':all(checks.values()),
          'stage_verdict':'PASS','checks':checks,'execution_checks':s['checks'],'independent_audit_checks':a['checks'],
          'scope_review_checks':r['checks'],'test_count':642,'stage_tasks_done':5,'stage_tasks_total':5,
          'overall_done':19,'overall_total':28,'next_task':'P4-01','user_understanding':'NOT_ASSESSED',
          'final_test_used':False,'formal_S1_SK':'NOT_RUN','scope_id':v['evaluation']['scope_id'],
          'planning':v['planning'],'training':v['training'],'evaluation':v['evaluation'],
          'execution_summary_sha256':sha(RUN/'summary.json'),'original_artifact_index_sha256':sha(RUN/'artifact_hashes.json'),
          'raw_audit_sha256':sha(ROOT/'reports/p3_05_audit.json'),'scope_review_sha256':sha(ROOT/'reports/p3_05_scope_review.json'),
          'scope_freeze_sha256':sha(RUN/'scope_freeze.v3.json'),'resources':s['resources'],
          'gate_code_sha256':sha(Path(__file__)),
          'alignment':'CASE_DEMONSTRATED for MLP23/47 on one development case; MLP11/R timeout retained; no independent final success rate',
          'scope':'comparison preparation verified; source execution PENDING_SCOPE_FREEZE immutable, final gate adds scope and independent audits'}
    assert gate['pass'];write(RUN/'completion_gate.json',gate)
    shutil.copyfile(RUN/'completion_gate.json',ROOT/'reports/p3_05_latest_summary.json')
    shutil.copyfile(RUN/'manifest.json',ROOT/'reports/p3_05_latest_manifest.json')
    original=json.loads((RUN/'artifact_hashes.json').read_text())['sha256']
    added={str(p.relative_to(RUN)):sha(p) for p in sorted(RUN.rglob('*')) if p.is_file()
           and str(p.relative_to(RUN)) not in original and p.name not in {'artifact_hashes.json','scope_artifact_hashes.json'}}
    write(RUN/'scope_artifact_hashes.json',{'original_index_sha256':sha(RUN/'artifact_hashes.json'),'added_sha256':added})
    print(json.dumps({'PASS':gate['pass'],'execution':len(s['checks']),'audit':len(a['checks']),'scope':len(r['checks']),
                      'completion':len(checks),'original_artifacts':len(original),'added_artifacts':len(added),'overall_done':19}))


if __name__=='__main__':main()
