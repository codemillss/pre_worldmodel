"""Read-only seven-model plus saved300-raw/review memory probe.

A reconstructed shadow is explicitly labelled; no controller is rerun and no
new fresh-substep verification is claimed. Every output is a small receipt.
"""
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import shutil
import sys
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch
import tomllib

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import mujoco
import torch
from push_align.final_manifest import frozen_source_contract
from push_align.learning_data import load_learning_data
from push_align.final_recording import file_sha256
from push_align.final_recording_columns import read_raw
from push_align.final_result_review import review_recording
from p4_compare_candidates import load_models

OUT=ROOT/'tmp/p5-review-combined-memory-probe-20261016-01'
REPORT=ROOT/'reports/p5_review_combined_memory_probe.json'
MIB=1024**2
GIB=1024**3
CASES={label:ROOT/'runs/p5-native-driver-20261016-01'/label/'episode/manifest.json' for label in ('reference','fresh')}


def rss():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value)*(1 if sys.platform=='darwin' else 1024)


def used():
    total=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) if OUT.exists() else 0
    return total+(REPORT.stat().st_size if REPORT.exists() else 0)


def guard():
    assert rss()<=6*GIB,'Preserve outputs:6GiB RSS checkpoint reached'
    assert used()<=2*MIB,'Preserve outputs:2MiB receipt ceiling reached'
    assert shutil.disk_usage(ROOT).free>=2*GIB,'Preserve outputs:2GiB free reserve reached'


def save(path,value):
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def run():
    assert not OUT.exists() and not REPORT.exists(),'Never overwrite evidence'
    guard();OUT.mkdir()
    contract=frozen_source_contract(ROOT)
    source_paths=set(contract['source_sha256'])
    source_paths.update({'src/push_align/final_manifest.py','src/push_align/final_driver.py',
        'src/push_align/final_result_review.py','src/push_align/final_recording_columns.py',
        'src/push_align/learning_data.py','src/push_align/task.py','src/push_align/contact_geometry.py',
        'src/push_align/reproduction.py','src/push_align/environment.py','src/push_align/closed_loop.py',
        'scripts/p4_compare_candidates.py','tests/test_p5_final_result_review.py',
        'tools/p5_review_combined_memory_probe.py'})
    for path in CASES.values():
        m=json.loads(path.read_text());source_paths.add(str(path.relative_to(ROOT)))
        for p in list(m['files_sha256'])+list(m['source_bindings']):
            p=Path(p).resolve();assert p.is_relative_to(ROOT);source_paths.add(str(p.relative_to(ROOT)))
    before={p:file_sha256(ROOT/p) for p in sorted(source_paths)}
    measurements={'after_import_rss_bytes':rss()};counts={name:0 for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','model_forward','adam_step')}
    def forbidden(name):
        def call(*a,**k):counts[name]+=1;raise AssertionError('Forbidden new execution:'+name)
        return call
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    with ExitStack() as traps:
        for name in ('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse'):
            traps.enter_context(patch.object(mujoco,name,side_effect=forbidden(name)))
        traps.enter_context(patch.object(torch.nn.Module,'__call__',side_effect=forbidden('model_forward')))
        traps.enter_context(patch.object(torch.optim.Adam,'step',side_effect=forbidden('adam_step')))
        started=perf_counter()
        metadata=json.loads((ROOT/'runs/p3-02-20261009-01/learning_data/normalizer/normalizer.json').read_text())['normalizer']
        learning_data,normalizer=load_learning_data(ROOT/'runs/p3-02-20261009-01/learning_data',metadata['schema'],metadata['dataset_identity'],metadata['normalizer_hash'])
        models,identities=load_models(normalizer,contract['physical_config'],json.loads((ROOT/'configs/p4_development_e1.v1.json').read_text()))
        assert set(models)==set(contract['model_identities'])
        for cid,identity in identities.items():
            assert identity['path']==contract['model_identities'][cid]['path']
            assert identity['model_hash']==contract['model_identities'][cid]['model_hash']
        measurements.update(model_load_wall_s=perf_counter()-started,after_seven_models_rss_bytes=rss())
        guard()
        original=json.loads(CASES['reference'].read_text());started=perf_counter()
        snapshot,steps=read_raw(original['raw']['prefix'],expected_sha256=original['raw']['files_sha256'])
        assert len(steps)==300
        shadow=SimpleNamespace(initial_snapshot=snapshot,results=steps)
        measurements.update(shadow_raw_read_wall_s=perf_counter()-started,after_shadow300_rss_bytes=rss())
        guard()
        profile=tomllib.load((ROOT/'configs/acceptance.v1.toml').open('rb'))
        rows=[]
        for label,path in CASES.items():
            started=perf_counter()
            resolved=review_recording(path,physical_config=contract['physical_config'],profile=profile,
                expected_manifest_sha256=file_sha256(path),expected_run_scope='development_preflight',
                raw_execution=shadow,raw_execution_origin='reconstructed_saved_raw')
            assert resolved['accepted'],resolved['issues']
            assert resolved['fresh_substeps_independently_rechecked'] is False
            assert resolved['provided_raw_execution_origin']=='reconstructed_saved_raw'
            receipt_path=OUT/(label+'-review.json');save(receipt_path,resolved)
            rows.append({'case':label,'status':resolved['status'],'termination':resolved['termination'],
                'review_wall_s':perf_counter()-started,'process_peak_rss_bytes_so_far':rss(),
                'metrics':{k:resolved['metrics'][k] for k in ('success','elapsed_s','final_position_error_m','metric_state_time_s')},
                'raw_provenance':resolved['raw_provenance'],'fresh_substeps_independently_rechecked':False,
                'receipt_path':str(receipt_path.relative_to(ROOT)),'receipt_sha256':file_sha256(receipt_path)})
            guard()
        assert len(models)==7 and len(shadow.results)==300 and learning_data is not None
        assert not any(counts.values())
    after={p:file_sha256(ROOT/p) for p in before};assert before==after
    guard()
    report={'schema':'p5-review-combined-memory-probe-v1','status':'PASS_READONLY_RESIDENCY_PROBE_NOT_NEW_CONTROLLER_RUN',
        'as_of_client_date':'2026-10-16','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'resident_seven_models':list(models),'resident_saved_raw_intervals':len(steps),
        'measurements':measurements,'review_rows':rows,'max_process_rss_bytes':rss(),'max_process_rss_MiB':rss()/MIB,
        'limits_bytes':{'RSS':6*GIB,'new_outputs':2*MIB,'free_reserve':2*GIB},
        'new_execution_counts':counts,'all_original_sources_unchanged':True,'source_before_sha256':before,
        'source_after_sha256':after,'output_bytes_before_report':used(),'free_bytes_after':shutil.disk_usage(ROOT).free,
        'scope':['Seven actual frozen model bundles and normalizer are loaded but never called or updated.',
            'The supplied SimpleNamespace contains reconstructed saved native300 raw. It is a residency shadow, not a new actual fresh controller execution.',
            'Both original and compact-fresh native records are reviewed against retained source/raw/task/boundary evidence. No new fresh-substep verification is claimed.',
            'RSS covers resident model bundles+saved300 shadow+independent review in one process. It does not emulate every live executor capture or prove all future condition peaks.',
            'No final-family generation, simulation, model forward, optimizer update, original deletion or rewriting.']}
    save(REPORT,report);guard()
    print(json.dumps({'report':str(REPORT.relative_to(ROOT)),'sha256':file_sha256(REPORT),
        'status':report['status'],'models':len(models),'RSS_MiB':report['max_process_rss_MiB'],
        'new_execution_counts':counts,'output_bytes':used()},ensure_ascii=False))


if __name__=='__main__':run()
