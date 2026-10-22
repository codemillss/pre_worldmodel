#!/usr/bin/env python3
"""Publish selected unchanged D1/D2 arrays; leave full raw in the original run."""
from __future__ import annotations
import argparse,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from push_align.final_driver_v2 import read_full_trace
from push_align.final_recording import file_sha256
from push_align.data import _encode_nonfinite
from dataclasses import asdict


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();run=a.run.resolve()
    summary=json.loads((run/'D2/summary.json').read_text());assert summary['status']=='PASS_PILOT_RECORDING_AND_REPRODUCTION_GATES'
    dest=ROOT/'artifacts/followup/20261022-d1-d2';dest.mkdir(parents=True,exist_ok=False);records=[]
    def copy(source,relative):
        target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        sha=file_sha256(source);assert sha==file_sha256(target)
        records.append({'source_run_path':Path(source).relative_to(run).as_posix(),'public_path':relative,'sha256':sha,'bytes':target.stat().st_size,'transformation':'byte_identical'})
    for relative in ['summary.json','entry.json','resource_projection.json']:copy(run/'D2'/relative,relative)
    raw_inventory=[]
    for state in ['N-T','F-O0']:
        folder=run/'D1'/state;restore=json.loads((folder/'restoration.json').read_text());assert restore['status']=='PASS_EXACT_RESTORATION_GATE'
        copy(folder/'restoration.json',f'restoration-{state}.json');copy(folder/'prefix_comparisons.jsonl',f'prefix-{state}.jsonl')
        tick=78 if state=='N-T' else 254;cap=next(v for v in restore['captures'] if v['tick']==tick)
        data,_,_=read_full_trace({'files_sha256':{str(folder/k):v for k,v in cap['trace_files_sha256'].items()}})
        public_state=dest/'states'/state;public_state.mkdir(parents=True)
        snap=data['snapshot']
        with (public_state/'snapshot.npz').open('xb') as out:np.savez_compressed(out,integration_state=snap.integration_state,warning_counts=snap.warning_counts,warning_lastinfo=snap.warning_lastinfo,reference_height=np.asarray(snap.reference_height),needs_reset=np.asarray(snap.needs_reset),observed_state10=data['observation'].state)
        meta={'scope':'Portable derived subset of authenticated full typed input; original full input remains in local run. This JSON is not asserted byte-identical to the original full input.','source_input_typed_sha256':cap['input_typed_sha256'],'tick':tick,'model_hash':snap.model_hash,'environment_hash':snap.environment_hash,'rng_state':snap.rng_state,'physical_config':data['physical_config'],'planner_config':data['planner_config'],'task_spec':asdict(data['task_spec']),'task_status':asdict(data['task_status']),'planner_context':asdict(data['planner_context']),'geometry_context':asdict(data['geometry_context'])}
        (public_state/'input_parameters.json').write_text(json.dumps(_encode_nonfinite(meta),ensure_ascii=False,indent=2)+'\n')
        for source in sorted((run/'D2'/state).glob('*')):
            if source.is_file():copy(source,f'states/{state}/{source.name}')
        for i in range(16):
            for panel in ['A','B']:
                trial=run/'D2'/state/f'candidate-{i:03d}'/panel/'repeat-0'
                copy(trial/'diagnostic.npz',f'states/{state}/actual/candidate-{i:03d}-{panel}.npz')
                copy(trial/'summary.json',f'states/{state}/actual/candidate-{i:03d}-{panel}.summary.json')
                for repeat in range(3):
                    path=trial.parent/f'repeat-{repeat}'/'raw_receipt.json';receipt=json.loads(path.read_text())
                    files=[]
                    for path,sha in receipt['files_sha256'].items():
                        assert file_sha256(path)==sha
                        files.append({'local_run_path':Path(path).relative_to(run).as_posix(),'sha256':sha,'bytes':Path(path).stat().st_size,'included_in_public_repository':False})
                    raw_inventory.append({'state':state,'candidate_id':i,'panel':panel,'repeat':repeat,'raw_files':files})
    (dest/'raw_inventory.json').write_text(json.dumps({'scope':'Full lossless raw is preserved locally and omitted from public subset. Hashes identify original records, not downloadable copies.','records':raw_inventory},ensure_ascii=False,indent=2)+'\n')
    files=[{'path':p.relative_to(dest).as_posix(),'bytes':p.stat().st_size,'sha256':file_sha256(p)} for p in sorted(dest.rglob('*')) if p.is_file()]
    manifest={'schema':'followup-pilot-selected-public-evidence-v1','original_run_relative_path':'runs/'+run.name,'scope':'D1 restoration evidence, seven-model first16 forecasts and teacher-forced arrays; first repeat boundary/hold arrays for all64 state-candidate-panel cases. All192 raw recordings remain local; no physical experiment re-executed by export.','scientific_copy_records':records,'files':files,'files_excluding_self':len(files),'bytes_excluding_self':sum(f['bytes'] for f in files),'source_copies_byte_identical':True}
    (dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'public_files':len(files),'bytes':manifest['bytes_excluding_self']}))
if __name__=='__main__':main()
