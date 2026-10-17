"""Reprice unchanged P5 work; preserve all earlier estimates and measurements."""
from pathlib import Path
from datetime import datetime,timezone
from copy import deepcopy
import hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from push_align.resource_budget import estimate_storage,capacity_decision

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()
def main():
    old=json.loads((ROOT/'configs/p5_g2_resource_plan.v2.json').read_text());r=deepcopy(old)
    native=json.loads((ROOT/'reports/p5_driver_native_preflight_review.json').read_text())
    storage=json.loads((ROOT/'reports/p5_storage_feasibility.native_codec.json').read_text())
    assert all(native['checks'].values())
    r['schema']='p5-g2-resource-plan-v3';r['measured_at_utc']=datetime.now(timezone.utc).isoformat()
    r['fixed_allowances_bytes'].update(E1_prediction_analysis=256*1024**2,P5_boundary_analysis=192*1024**2)
    r['estimate']=estimate_storage(r['counts'],r['units_bytes'],r['fixed_allowances_bytes'])
    r['capacity']=capacity_decision(shutil.disk_usage(ROOT).free,r['estimate']['required_bytes_including_reserve'])
    r['RAM'].update(combined_final_driver_and_native300_codec_peak=native['combined_peak_rss_bytes'],
        combined_peak_scope='Measured sequential R development original and independent actual rerun; all final regimes/devices remain unmeasured',
        gate='Native combined integration passed; one worker per cell, per-action 6GiB RSS checkpoint, no parallel final physics/training')
    r['unit_evidence_limits']={
      'raw':'19 preserved regimes plus empty prefixes roundtripped in typed columnar XZ6; max fixed7443B and max average incremental101478B/interval. Not a proven worst-case bound.',
      'witness':'All26 G1 witnesses plus actual300-interval original/fresh; maximum fixed42596B, perinterval1627B, perdecision2308B. Additive independent gzip members.',
      'plan_index':'Original40 plan arrays/22archives and actual300-plan R original/fresh; perindex typed arrays directly equal; keep larger pre-existing13697B/record unit.',
      'full_trace':'Native fulltrace NPZ max117503B and fullinput JSON max47713B; envelope adds2017B input-only NPZ. Learned input NPZ intentionally counted again, conservatively. G uses envelope only.',
      'multiplier':'1.5 empirical planning allowance, not confidence or worst-case guarantee; observed component totals beyond allowance stop with original prefix preserved.',
      'working_disk':'Final writer writes directly to exclusive destination, no whole raw disk staging copy;32MiB scratch retained. Combined native process RSS1.40GiB measured separately.',
      'analysis_allowances':'E1/fixed-G K1/5/10 arrays/reasons/source metadata256MiB; P5 cell manifests/review receipts/progress/bootstrap192MiB. These are conservative operational allowances, not observed final output sizes.'}
    r['allowance_revision_reason']={
      'E1_prediction_analysis':'Includes all retained K1/5/10 truth/prediction/delta/mask/reason arrays at max3600 start opportunities and7 models, plus E1 84 choices/576 shared futures. Old102.4MiB was not a measured bound; increase before outcomes rather than assume compression.',
      'P5_boundary_analysis':'Includes up to200 native cell manifests (measured maximum319022B/cell), independent raw/task review receipts, row receipts, progress and paired bootstrap. Increase before outcomes; no sample/field removal.'}
    r['time']['native_original_and_fresh_wall_s']=native['wall_time_s']
    r['time']['new_codec_measurement_scope']='New typed columnar codec actual native300; old73.54s JSON-XZ measurement above retained as conservative historical supplement, not mislabeled as new codec speed.'
    r['new_model_calls']={k:{'calls':v['ledger']['model_calls'],'rows':v['ledger']['model_rows']} for k,v in native['summaries'].items()}
    r['new_model_calls_scope']='See native report adapter call counters; action intervals and engine calls are different denominators.'
    r['supersedes_without_modifying']='configs/p5_g2_resource_plan.v1.json and v2.json; capacity HOLD history retained'
    r['source_sha256'].update({'configs/p5_g2_resource_plan.v2.json':sha(ROOT/'configs/p5_g2_resource_plan.v2.json'),
        'tools/p5_resource_finalize.py':sha(Path(__file__))})
    for rel,h in r['source_sha256'].items():
        if sha(ROOT/rel)!=h:raise RuntimeError('Source changed: '+rel)
    target=ROOT/'configs/p5_g2_resource_plan.v3.json'
    with target.open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'path':str(target.relative_to(ROOT)),'estimate':r['estimate'],'capacity':r['capacity']},ensure_ascii=False))
if __name__=='__main__':main()
