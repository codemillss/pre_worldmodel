"""Close G2 measured budget separately from the unfinished G3 freeze gate."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from push_align.resource_budget import estimate_storage,capacity_decision

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()
def read(p):return json.loads((ROOT/p).read_text())
def main():
    r=read('configs/p5_g2_resource_plan.v3.json');native=read('reports/p5_driver_native_preflight_review.json')
    scientific=read('configs/p4_first_moment.v1.json')['bindings_sha256']
    candidate=read('configs/p5_freeze_candidate.v1.json')['source_sha256']
    groups={'scientific':scientific,'candidate':candidate,'resource_source':r['source_sha256']}
    preservation={name:{'checked':len(values),'all_unchanged':all((ROOT/p).is_file() and sha(ROOT/p)==h for p,h in values.items())} for name,values in groups.items()}
    estimate=estimate_storage(r['counts'],r['units_bytes'],r['fixed_allowances_bytes'])
    capacity=capacity_decision(shutil.disk_usage(ROOT).free,estimate['required_bytes_including_reserve'])
    checks={'same_scope_counts':r['counts']==read('configs/p5_g2_resource_plan.v1.json')['counts'],
      'exact_storage_arithmetic':estimate==r['estimate'],'all_original_bindings_unchanged':all(x['all_unchanged'] for x in preservation.values()),
      'native600_actual_integration_all_checks':all(native['checks'].values()),'current_full_budget_fits':capacity['status']=='FIT_ESTIMATE_ONLY',
      'reserve_2GiB_and_mismatch256MiB':r['fixed_allowances_bytes']['hard_free_reserve']==2*1024**3 and r['fixed_allowances_bytes']['mismatch_prefix_and_diagnostics']==256*1024**2,
      'no_new_training':native['learning_updates']==0,'final_manifest_not_generated':not (ROOT/'runs/p5-final-20261016-01/manifest.json').exists()}
    result={'schema':'p5-g2-closure-v3','as_of_client_date':'2026-10-16','checked_at_utc':datetime.now(timezone.utc).isoformat(),
      'status':'G2_MEASURED_BUDGET_READY_G3_HOLD' if all(checks.values()) else 'HOLD','checks':checks,'capacity':capacity,
      'estimate':estimate,'preservation':preservation,'native_report':'reports/p5_driver_native_preflight_review.json',
      'resource_plan':'configs/p5_g2_resource_plan.v3.json','resource_plan_sha256':sha(ROOT/'configs/p5_g2_resource_plan.v3.json'),
      'combined_peak_rss_bytes':native['combined_peak_rss_bytes'],'new_actual_development_action_intervals':native['total_new_action_intervals_returned'],
      'new_model_adapter_calls':sum(x['ledger']['model_calls'] for x in native['summaries'].values()),
      'new_model_rows':sum(x['ledger']['model_rows'] for x in native['summaries'].values()),'new_training_updates':0,
      'final_comparison_started':False,'P5_01':'HOLD_G3_ORCHESTRATION_AUDIT_AND_ACTUAL_MANIFEST','overall_done':'23/28',
      'performance_improvement':'NOT_EVALUATED','personal_understanding':'NOT_ASSESSED',
      'remaining':'Complete E1 adoption, raw/task review, orchestration regression/independent audit; then preregister actual final families/source manifest. Recheck live capacity before every stage/cell/action/write.',
      'old_capacity_hold_retained':'reports/p5_g2_capacity_hold.json','new_spending_KRW':0,
      'source_sha256':{'configs/p5_g2_resource_plan.v3.json':sha(ROOT/'configs/p5_g2_resource_plan.v3.json'),'reports/p5_driver_native_preflight_review.json':sha(ROOT/'reports/p5_driver_native_preflight_review.json'),'tools/p5_g2_resource_close.py':sha(Path(__file__))}}
    target=ROOT/'reports/p5_g2_resource_closure.v3.json'
    with target.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'status':result['status'],'checks':checks,'capacity':capacity,'preservation':preservation},ensure_ascii=False))
if __name__=='__main__':main()
