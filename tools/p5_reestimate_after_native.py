#!/usr/bin/env python3
"""Preserve v1, price the measured columnar codec and actual driver overhead."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from push_align.resource_budget import estimate_storage,capacity_decision


def main():
    read=lambda name:json.loads((ROOT/name).read_text())
    old=read('configs/p5_g2_resource_plan.v1.json')
    storage=read('reports/p5_storage_feasibility.native_codec.json')
    native=read('reports/p5_driver_native_preflight_review.json')
    assert storage['status'].startswith('PASS_19_') and all(native['checks'].values())
    units=dict(old['units_bytes'])
    measures=storage['summary']
    units.update(raw_fixed=measures['max_fixed_bytes'],raw_interval=math.ceil(measures['max_average_increment_bytes_per_interval']))
    attempts=list(native['summaries'].values())
    units['witness_fixed']=max(units['witness_fixed'],*[r['storage']['witness']['fixed_bytes'] for r in attempts])
    units['witness_interval']=max(units['witness_interval'],*[max(r['storage']['witness']['interval_member_bytes'],default=0) for r in attempts])
    units['witness_decision']=max(units['witness_decision'],*[max(r['storage']['witness']['decision_member_bytes'],default=0) for r in attempts])
    arrays_max=metadata_max=input_only_max=0
    reference=native['summaries']['reference']
    cell=read(str(Path(reference['manifest_path']).relative_to(ROOT)))
    for receipt in cell['full_traces']:
        npz=next(Path(p) for p in receipt['files_sha256'] if p.endswith('.npz'))
        meta=next(Path(p) for p in receipt['files_sha256'] if p.endswith('.gz'))
        arrays_max=max(arrays_max,npz.stat().st_size);metadata_max=max(metadata_max,meta.stat().st_size)
        with np.load(npz,allow_pickle=False) as source:
            buf=io.BytesIO();np.savez_compressed(buf,**{k:source[k] for k in source.files if k.startswith('input_')})
            input_only_max=max(input_only_max,len(buf.getvalue()))
    # New full-input JSON replaces the old separate identity JSON. Input-only
    # NPZ is additionally included in the envelope, conservatively double
    # counted for learned full traces and also covering G input-only captures.
    units.update(full_trace_npz=arrays_max,full_trace_identity=0,
                 snapshot_envelope=metadata_max+input_only_max)
    estimate=estimate_storage(old['counts'],units,old['fixed_allowances_bytes'])
    free=shutil.disk_usage(ROOT).free
    files=['configs/p5_g2_resource_plan.v1.json','configs/p5_recording_policy.v1.json',
        'reports/p5_storage_feasibility.native_codec.json','reports/p5_driver_native_preflight_review.json',
        'configs/p5_driver_native_preflight.design.v1.json','src/push_align/final_recording_columns.py',
        'src/push_align/final_driver.py','src/push_align/resource_budget.py',
        'tests/test_p5_final_recording_columns.py','tests/test_p5_final_driver.py',
        'scripts/p5_driver_native_preflight.py','tools/p5_reestimate_after_native.py']
    result={**old,'schema':'p5-g2-resource-plan-v2','as_of_client_date':'2026-10-16',
        'measured_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'MEASURED_COLUMNAR_AND_NATIVE_INTEGRATION_RESOURCE_REVIEW',
        'supersedes_without_modifying':'configs/p5_g2_resource_plan.v1.json; prior capacity HOLD preserved',
        'units_bytes':units,'estimate':estimate,
        'capacity':capacity_decision(free,estimate['required_bytes_including_reserve']),
        'integration':{'status':'PASS','checks':native['checks'],'report':'reports/p5_driver_native_preflight_review.json',
            'actual_development_action_intervals':native['total_new_action_intervals_returned'],
            'combined_peak_rss_bytes':native['combined_peak_rss_bytes'],
            'scope':'One fixed R development condition and its independent actual rerun; not final task performance or all-device bound'},
        'recording_policy_delta':{'raw_codec':'push_align.final_recording_columns',
            'all_original_and_StepResult_values_retained':True,
            'serialization':'typed structural columns plus numeric vectors, lossless XZ6; same raw array SHA and full typed JSON roundtrip19 regimes',
            'full_input_metadata_max_bytes':metadata_max,'input_only_npz_max_bytes':input_only_max,
            'all_other_fields':'v1 policy retains index/witness/fulltrace slots and source/validity/failure evidence unchanged'},
        'no_sample_model_task_criterion_change':True,
        'same_multiplier':'1.5; empirical allowance, not worst-case proof',
        'G2_capacity_hold_resolution':'A new format with broader measured evidence and an actual driver roundtrip reprices all unchanged work. The old insufficient-capacity measurement is retained; it is not erased by one free-space rebound.',
        'remaining_gates':['Freeze complete outcomes-blind final manifest and immutable source bundle before final outcomes',
                           'Validate final orchestration, E1/shared futures/statistics/prediction/replay wiring before its production freeze',
                           'Recheck live remaining budget per stage/cell; output growth or capacity loss stops with preserved prefix'],
        'new_model_calls':None,'new_physics_calls':native['total_new_action_intervals_returned'],
        'new_physics_count_unit':'actual action intervals, not engine API calls; setup engine calls recorded in native report',
        'new_training_updates':0,'actual_final_test_manifest_generated':False,'P5_01':'HOLD',
        'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    target=ROOT/'configs/p5_g2_resource_plan.v2.json'
    with target.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'capacity':result['capacity'],'units':units,'required':estimate['required_bytes_including_reserve'],'file':str(target.relative_to(ROOT))}))


if __name__=='__main__':main()
