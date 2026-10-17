"""Record completed preparation checks without granting final execution."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from push_align.resource_budget import capacity_decision

def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def save(p,v):
    with (ROOT/p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
    plan=json.loads((ROOT/'configs/p5_g2_resource_plan.v3.json').read_text())
    source=['scripts/p5_final_run.py','scripts/p5_final_e1.py','scripts/p5_final_e1_adoption.py','src/push_align/final_manifest.py','src/push_align/final_driver.py','src/push_align/final_statistics.py','src/push_align/final_prediction.py','src/push_align/final_result_review.py','src/push_align/final_recording_columns.py','tools/p5_final_visuals.py','tools/p5_closeout_snapshot.py']
    source += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'tests').glob('test_p5_final_*.py'))]
    modules=['recording_columns','driver','execution','result_review','e1','e1_adoption','visuals','manifest','statistics','prediction','orchestration']
    tests={'command':'PYTHONPATH=src:scripts:tests .venv/bin/python -m unittest '+' '.join('test_p5_final_'+x for x in modules)+' -q',
        'passed':206,'failed':0,'wall_time_s':57.484,'source':'Completed terminal session11724; unittest terminal result inspected',
        'scope':'Synthetic/read-only regression and actual codec fixture render; no actual final-test outcome'}
    progress={'schema':'p5-g3-preparation-progress-v1','checked_at_utc':datetime.now(timezone.utc).isoformat(),'as_of_client_date':'2026-10-16',
        'status':'CODE_TESTED_FINAL_INDEPENDENT_ORCHESTRATION_REVIEW_AND_CAPACITY_ADMISSION_PENDING','tests':tests,
        'implemented':['sealed exact-cell adoption and raw/task review','duplicate/unknown progress rejection','fixed replay selection from complete population','cumulative mismatch and component/metadata budget checks','E1 predictions before all futures; repeat/stage seals and pure cost regeneration','active raw write reservation credit and hard-reserve preservation retries','same settled E2/application G snapshots','K1/5/10 actual applied-action prediction masks','independent family-stratified statistics and retained-array/source seals'],
        'native_driver_report':'reports/p5_driver_native_preflight_review.json','combined_shadow_memory_report':'reports/p5_review_combined_memory_probe.json',
        'limits':['No final family generated; no actual final evaluation or actual E1 shared futures run','Synthetic206 PASS is not complete independent orchestration review','Repeated free-space snapshots below9,242,369,106B prohibit unqualified batch admission','No runtime source freeze or final manifest has been issued'],
        'remaining':['Independent final orchestration integration audit including resource ledger and receipt handling','Current complete remaining-budget admission; preserve any stopped prefix without automatic retry','Outcomes-blind actual12-family+application generation/source/model manifest freeze only after both gates pass'],
        'actual_final_manifest_generated':False,'actual_final_outcomes':0,'personal_understanding':'NOT_ASSESSED','P5_01':'HOLD','overall_done':'23/28','source_sha256':{p:sha(p) for p in source}}
    save('reports/p5_g3_preparation_progress.json',progress)
    capacity=capacity_decision(shutil.disk_usage(ROOT).free,plan['estimate']['required_bytes_including_reserve'])
    references=['configs/p5_g2_resource_plan.v3.json','reports/p5_g2_resource_closure.v4.json','reports/p5_g2_resource_independent_review.v3.json','reports/p5_review_combined_memory_probe.json','reports/p5_g3_preparation_progress.json','output/pdf/프로젝트_통합보고서.pdf','output/pdf/프로젝트_통합보고서.manifest.json','output/pdf/프로젝트_통합보고서.verification.json','tools/p5_closeout_snapshot.py']
    snapshot={'schema':'p5-g2-closeout-snapshot-v5','checked_at_utc':datetime.now(timezone.utc).isoformat(),'status':'BATCH_HOLD_CAPACITY_VARIABILITY_AND_G3_PENDING','capacity_snapshot':capacity,
        'capacity_observed_range_bytes':[7741440000,9707520000],'range_scope':'Read-only observations during this closeout; not a claimed future bound',
        'actual_final_manifest_generated':(ROOT/'runs/p5-final-20261016-01/manifest.json').exists(),'overall_done':'23/28','P5_01':'HOLD',
        'original_preservation':'Scientific748 and candidate97 SHA verified by v3 closure; originals retained','source_sha256':{p:sha(p) for p in references}}
    save('reports/p5_g2_closeout_snapshot.v5.json',snapshot)
    print(json.dumps({'tests':206,'capacity':capacity,'actual_final_generated':snapshot['actual_final_manifest_generated'],'pdf_sha256':sha('output/pdf/프로젝트_통합보고서.pdf')},ensure_ascii=False))
if __name__=='__main__':main()
