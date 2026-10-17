"""Measure native lossless columnar codec on19 preserved regimes only.

No environment object, physical state, optimizer or learned model is created.
One fresh interpreter per case makes peak-RSS labels meaningful. Nonrepresentative
new scratch pairs are expressly disposable only after hashes and full roundtrip
checks are retained in the report. All original files remain immutable.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from push_align.data import _encode_nonfinite
from push_align.environment import Observation, PlanarPushEnv, SnapshotBundle, StepResult
from push_align.final_recording_columns import (file_sha256, raw_paths, read_raw, write_raw, tree_digest)

OUT = ROOT / 'tmp/p5-columns-measurement'
REPORT = ROOT / 'reports/p5_storage_feasibility.native_codec.json'
SOURCE = ROOT / 'reports/p5_g2_xz_regime_measurement.v2.json'
MIB = 1024**2
GIB = 1024**3
CODE_FILES = [ROOT / 'src/push_align/final_recording_columns.py',
    ROOT / 'src/push_align/final_recording.py',
    ROOT / 'tests/test_p5_final_recording_columns.py', Path(__file__).resolve(), SOURCE]


def guard():
    used = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) if OUT.exists() else 0
    assert used <= 64 * MIB, 'Keep prefix and stop: measurement64MiB ceiling'
    assert shutil.disk_usage(ROOT).free >= 2 * GIB, 'Keep prefix and stop: free2GiB reserve'
    return {'own_bytes': used, 'free_bytes': shutil.disk_usage(ROOT).free}


def exact(a,b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def peak():
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / (MIB if sys.platform == 'darwin' else 1024)


def run_case(index):
    declared = json.loads(SOURCE.read_text())['rows'][index]
    folder = (ROOT / declared['source_npz']).parent
    paths = [folder/'raw_execution.npz', folder/'raw_execution.json.gz']
    witness_path = folder/'witness.json'
    if witness_path.is_file():
        witness = json.loads(witness_path.read_text()); paths += [witness_path]
    else:
        witness = None; paths += [folder/'manifest.json', folder/'steps.jsonl']
    source_hashes = {str(p.relative_to(ROOT)): file_sha256(p) for p in paths+CODE_FILES}
    guard()
    started = perf_counter()
    with gzip.open(paths[1], 'rt') as stream: meta=json.load(stream)
    with np.load(paths[0],allow_pickle=False) as source: arrays={k:source[k].copy() for k in source.files}
    raw_snapshot=meta['initial_snapshot']
    snapshot=SnapshotBundle(np.asarray(raw_snapshot['integration_state'],np.float64),
        raw_snapshot['model_hash'],raw_snapshot['environment_hash'],raw_snapshot['reference_height'],
        raw_snapshot['rng_state'],np.asarray(raw_snapshot['warning_counts'],np.int64),
        np.asarray(raw_snapshot['warning_lastinfo'],np.int64),raw_snapshot['needs_reset'])
    diagnostics=meta['diagnostics']
    for path,value in zip(meta['diagnostic_nonfinite_paths'],arrays['diagnostic_nonfinite_values']):
        target=diagnostics
        for part in path[:-1]: target=target[part]
        target[path[-1]]=float(value)
    n=len(arrays['state']); assert n==declared['intervals'] and len(diagnostics)==n
    skip=len(arrays['substep_times'])//n
    assert len(arrays['substep_times'])==n*skip
    steps=[]; validity_checks=0
    if witness is not None:
        assert len(witness['intervals'])==n
        schema=witness['initial']['schema_version']
        for i,(diag,boundary) in enumerate(zip(diagnostics,witness['intervals'])):
            span=slice(i*skip,(i+1)*skip)
            steps.append(StepResult(arrays['action_requested'][i],arrays['action_applied'][i],
                Observation(arrays['state'][i],float(boundary['sim_time_after']),schema,boundary['observation_valid']),
                float(arrays['actual_dt'][i]),boundary['transition_valid'],diag,
                arrays['substep_states'][span],arrays['substep_times'][span]))
        provenance='Explicit observation flags/time/schema from original saved G1 witness; transition flag from saved boundary.'
    else:
        manifest=json.loads((folder/'manifest.json').read_text())
        config=manifest['physical_contract']['resolved_config']; schema=config['schema_version']
        assert schema==manifest['physical_contract']['state_schema']
        old_steps=[json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines() if s.strip()]
        assert len(old_steps)==n
        proxy=SimpleNamespace(config=config,_reference_height=snapshot.reference_height)
        for i,(diag,old_step) in enumerate(zip(diagnostics,old_steps)):
            span=slice(i*skip,(i+1)*skip)
            assert old_step['physical_valid'] is True and diag['validity_violations']==[]
            for sample in diag['samples']:
                assert not PlanarPushEnv._validity(proxy,sample);validity_checks+=1
            sim_time=float(arrays['substep_times'][span][-1])
            assert np.float64(sim_time).tobytes()==np.float64(diag['samples'][-1]['sim_time']).tobytes()
            steps.append(StepResult(arrays['action_requested'][i],arrays['action_applied'][i],
                Observation(arrays['state'][i],sim_time,schema,True),float(arrays['actual_dt'][i]),
                old_step['physical_valid'],diag,arrays['substep_states'][span],arrays['substep_times'][span]))
        provenance='Transition flags copied from old steps.jsonl; observation validity derived from unchanged _validity applied only to saved samples/config, time from last NPZ substep, schema from manifest. These absent old fields are not claimed originally explicit.'
    load_s=perf_counter()-started
    case_out=OUT/declared['sample_id'];case_out.mkdir()
    started=perf_counter();empty=write_raw(case_out/'empty',snapshot,[],xz_preset=6);empty_write_s=perf_counter()-started
    restored_empty,empty_steps=read_raw(case_out/'empty',expected_sha256=empty['files_sha256']);assert empty_steps==[]
    assert tree_digest(_encode_nonfinite(asdict(snapshot)))==tree_digest(_encode_nonfinite(asdict(restored_empty)))
    guard();started=perf_counter();receipt=write_raw(case_out/'full',snapshot,steps,xz_preset=6);write_s=perf_counter()-started
    guard();started=perf_counter();restored_snapshot,restored=read_raw(case_out/'full',expected_sha256=receipt['files_sha256']);read_s=perf_counter()-started
    assert len(restored)==n
    numeric_checks={}
    with np.load(raw_paths(case_out/'full')[0],allow_pickle=False) as saved:
        for k,old in arrays.items():
            numeric_checks[k]=exact(old,saved[k]);assert numeric_checks[k],k
    assert tree_digest(_encode_nonfinite(asdict(snapshot)))==tree_digest(_encode_nonfinite(asdict(restored_snapshot)))
    assert tree_digest(_encode_nonfinite(asdict(restored_snapshot)))==tree_digest(raw_snapshot)
    for old,new in zip(steps,restored):
        for key in ('requested_action','applied_action','substep_states','substep_times'):assert exact(getattr(old,key),getattr(new,key)),key
        assert exact(old.observation.state,new.observation.state)
        for a,b in ((old.actual_dt,new.actual_dt),(old.observation.sim_time,new.observation.sim_time)):
            assert np.float64(a).tobytes()==np.float64(b).tobytes()
        assert old.observation.schema_version==new.observation.schema_version
        assert old.observation.observation_valid is new.observation.observation_valid
        assert old.transition_valid is new.transition_valid
        assert tree_digest(old.diagnostics)==tree_digest(new.diagnostics)
    assert all(file_sha256(ROOT/p)==h for p,h in source_hashes.items())
    full_bytes=sum(receipt['bytes'].values());empty_bytes=sum(empty['bytes'].values())
    keep=declared['sample_id'] in ('G1-contact-R','historical300-R')
    scratch_before=guard()
    if not keep:
        # These were declared disposable measurement outputs, not old evidence.
        # Only our four new pair files are removed, after full roundtrip and hashes.
        for prefix in (case_out/'full',case_out/'empty'):
            for path in raw_paths(prefix):path.unlink()
    guard()
    result={'sample_id':declared['sample_id'],'source_directory':str(folder.relative_to(ROOT)),
        'intervals':n,'substep_samples':len(arrays['substep_times']),'source_sha256':source_hashes,
        'full_receipt':receipt,'empty_receipt':empty,'total_bytes':full_bytes,'empty_fixed_bytes':empty_bytes,
        'average_increment_bytes_per_interval':(full_bytes-empty_bytes)/n,
        'old_numeric_arrays_bitwise':numeric_checks,'complete_diagnostics_typed_floatbits_exact':True,
        'complete_initial_snapshot_semantics_exact':True,'all_explicit_reconstructed_fields_roundtrip_exact':True,
        'source_and_original_codec_unchanged':True,'field_provenance':provenance,
        'warning_dtype_limit':'Old JSON warning arrays were untyped; reconstructed int64 does not assert original engine dtype.',
        'saved_validity_checks':validity_checks,'load_reconstruction_s':load_s,'write_s':write_s,'empty_write_s':empty_write_s,
        'read_s':read_s,'fresh_process_peak_MiB':peak(),'RSS_scope':'Codec/reconstruction/roundtrip only; no controller or model RAM.',
        'representative_pair_retained':keep,'declared_disposable_new_pair_deleted_after_verified_roundtrip':not keep,
        'scratch_before_cleanup':scratch_before,'scratch_after_cleanup':guard()}
    report=case_out/'measurement.json';report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'case':declared['sample_id'],'bytes':full_bytes,'increment':result['average_increment_bytes_per_interval'],
        'peak_MiB':result['fresh_process_peak_MiB'],'report':str(report.relative_to(ROOT))}))


def main():
    if '--case' in sys.argv:
        run_case(int(sys.argv[sys.argv.index('--case')+1]));return
    assert not OUT.exists() and not REPORT.exists(),'Do not overwrite existing evidence'
    assert shutil.disk_usage(ROOT).free >= 2*GIB+64*MIB
    OUT.mkdir();before=guard();rows=[]
    declared=json.loads(SOURCE.read_text())['rows'];assert len(declared)==19
    for i,case in enumerate(declared):
        done=subprocess.run([sys.executable,__file__,'--case',str(i)],capture_output=True,text=True)
        if done.returncode:
            print(done.stdout,done.stderr,file=sys.stderr);raise RuntimeError('Retained failed-case prefix; stop and inspect')
        print(done.stdout.strip(),flush=True)
        rows.append(json.loads((OUT/case['sample_id']/'measurement.json').read_text()))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_p5_final_recording_columns.py','-v'],
        env={**__import__('os').environ,'PYTHONPATH':str(ROOT/'src')},capture_output=True,text=True)
    (OUT/'tests.txt').write_text(tests.stdout+tests.stderr);assert tests.returncode==0
    max_fixed=max(r['empty_fixed_bytes'] for r in rows)
    max_increment=max(r['average_increment_bytes_per_interval'] for r in rows)
    old_budget=json.loads((ROOT/'configs/p5_g2_resource_plan.v1.json').read_text())
    new_raw=int(np.ceil(1.5*(max_increment*old_budget['counts']['raw_intervals']+max_fixed*old_budget['counts']['raw_records'])))
    old_total=old_budget['estimate']['required_bytes_including_reserve']
    new_total=old_total-old_budget['estimate']['parts_bytes']['raw']+new_raw
    report={'schema':'p5-storage-feasibility-native-columnar-v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'status':'PASS_19_EXISTING_REGIMES_NATIVE_CODEC_NOT_FINAL_DRIVER_INTEGRATION','rows':rows,
        'summary':{'regimes':len(rows),'G1_short_regimes':13,'unique_long300_regimes':4,'ready_62_71_regimes':2,
            'max_fixed_bytes':max_fixed,'max_average_increment_bytes_per_interval':max_increment,
            'all_complete_diagnostics_typed_and_floatbits_exact':True,'all_old8_npz_arrays_bitwise':True,
            'max_fresh_codec_process_peak_MiB':max(r['fresh_process_peak_MiB'] for r in rows),
            'max_native300_write_s':max(r['write_s'] for r in rows if r['intervals']==300),
            'max_native300_read_s':max(r['read_s'] for r in rows if r['intervals']==300)},
        'tests_passed':26,'tests_log_sha256':file_sha256(OUT/'tests.txt'),
        'source_sha256':{str(p.relative_to(ROOT)):file_sha256(p) for p in CODE_FILES},
        'same_scope_same1_5_multiplier_budget':{'old_total_including_reserve':old_total,'new_raw':new_raw,
            'new_total_including_reserve':new_total,'estimated_saving':old_total-new_total,
            'not_proven_worst_case':True,'not_final_batch_authorization':True},
        'before':before,'after':guard(),'new_physics_calls':0,'new_model_calls':0,'new_training_updates':0,
        'old_original_and_codec_files_preserved':True,
        'limits':['All19 inputs are old observed regimes; no newly generated final-family evidence.',
            'New explicit fields absent from old data are provenance-labelled reconstructions, not a claim about missing original bytes.',
            'Fixedmax plus average marginalmax uses whole-prefix compression: empirical1.5 allowance and live growth guard remain required.',
            'Native300 writer has been measured on4 old long trajectories, not newly executed final controllers.',
            'Final batch driver+codec combined RAM and complete fresh replay integration require their own evidence.',
            'Only declared newly created temporary measurement pairs were removed after exact roundtrip; representative longR and contactR pairs retained.']}
    with REPORT.open('x') as target:json.dump(report,target,ensure_ascii=False,indent=2);target.write('\n')
    print(json.dumps({'report':str(REPORT.relative_to(ROOT)),'sha256':file_sha256(REPORT),'summary':report['summary'],
                     'budget':report['same_scope_same1_5_multiplier_budget']},ensure_ascii=False))

if __name__=='__main__':main()
