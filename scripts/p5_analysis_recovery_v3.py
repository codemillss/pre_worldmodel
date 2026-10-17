"""Additive analysis-only continuation after the frozen E2 reduction-order defect.

Logical stage=statistics; output=statistics/recovery-v2. Existing frozen files,
root statistics outputs and the physical treatment are never replaced. Physical
prediction/replay/application commands remain owned by the frozen v2 executable
and are run separately by the caller. This file starts no worker or engine.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
VERSION='p5-analysis-recovery-v3'
OUTPUT_RELATIVE='statistics/recovery-v2'
RECEIPT_SCHEMA='p5-analysis-recovery-statistics-receipt-v3'
COMPLETION_SCHEMA='p5-analysis-recovery-execution-complete-v3'


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def need(condition,message):
    if not condition:raise RuntimeError(message)
def payload(value):return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def exact_json(left,right):return json.dumps(left,sort_keys=True,allow_nan=False)==json.dumps(right,sort_keys=True,allow_nan=False)
def tree_files(folder):return {str(p.resolve()):sha(p) for p in Path(folder).rglob('*') if p.is_file()}


def load_original():
    path=ROOT/'scripts/p5_final_run_v2.py'
    spec=importlib.util.spec_from_file_location('p5_frozen_v2_for_analysis_recovery',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class StatisticsOutputView:
    """Only the statistics output path changes; actual source paths stay real."""
    def __init__(self,run,output):self.run=Path(run);self.output=Path(output)
    def __truediv__(self,name):return self.output if name=='statistics' else self.run/name


class VerifiedRows:
    """One original deep96 pass, then exact full SHA graph + sealed JSON reloads.

    No original module globals/functions are patched. Every reuse hashes all
    original raw,witness,trace,index,source and metadata bytes again. Reuse is
    bound to an explicit output receipt; it is never a stat-only PASS shortcut.
    """
    def __init__(self,original,run,manifest):
        self.original=original;self.run=Path(run);self.manifest=manifest
        self.graph={};self.deep_calls=0;self.reloads=0;self.graph_hash_passes=0
    def bind(self,path,digest):
        path=str(Path(path).resolve())
        need(path not in self.graph or self.graph[path]==digest,'Conflicting evidence SHA bindings: '+path)
        self.graph[path]=digest
    def verify_graph(self):
        for path,digest in self.graph.items():need(sha(path)==digest,'Verified E2 input graph changed: '+path)
        self.graph_hash_passes+=1
    def initialize(self):
        need(not self.graph,'A full original96 review pass is never repeated implicitly')
        for relative,digest in {**self.manifest['source_sha256'],**self.manifest['runtime_source_sha256']}.items():
            self.bind(self.original.ROOT/relative,digest)
        for name in ('manifest.json','freeze_receipt.json'):self.bind(self.run/name,sha(self.run/name))
        for cell in self.manifest['E2']:
            result=self.original.verified_result(self.run,'E2',cell['cell_id'],self.manifest,deep=True)
            self.deep_calls+=1;folder=Path(result['record_manifest']).parent;record=read(result['record_manifest'])
            for name,digest in (('worker_result.json',read(folder/'worker_result_receipt.json')['sha256']),
                    ('worker_result_receipt.json',sha(folder/'worker_result_receipt.json')),
                    ('manifest.json',result['record_manifest_sha256']),('review.json',result['review_sha256']),
                    ('component_budget_review.json',result['component_review_sha256'])):
                self.bind(folder/name,digest)
            for path,digest in {**record['files_sha256'],**record['source_bindings']}.items():self.bind(path,digest)
        need(self.deep_calls==96,'Original96 deep reviews incomplete')
        return self.reload(self.manifest['E2'])
    def reload(self,order):
        need(bool(self.graph),'A source-bound original raw review is required first')
        self.original.frozen(self.run);self.verify_graph();rows=[]
        for cell in order:
            folder=self.run/'E2'/cell['cell_id'];path=folder/'worker_result.json';result=read(path)
            need(sha(path)==read(folder/'worker_result_receipt.json')['sha256'],'Sealed worker JSON changed')
            need(result['phase']=='E2' and result['cell']==cell
                 and all(result['row'].get(k)==v for k,v in cell.items()),'Freshly reloaded E2 identity differs')
            rows.append(result['row'])
        self.reloads+=1;return rows
    def summary(self):return {'schema':'p5-one-deep-pass-exact-sha-reload-v3',
        'deep_original_reviews':self.deep_calls,'fresh_sealed_JSON_reloads':self.reloads,
        'full_evidence_graph_SHA_passes':self.graph_hash_passes,'bound_evidence_files':len(self.graph),
        'frozen_globals_or_functions_patched':False,'stat_only_cache_used':False,
        'reused_scope':'Original raw/task semantics reviewed once; all raw/witness/trace/index/source/metadata SHA bytes rehashed before each reordered JSON reload.'}


class StatisticsWriter:
    """Original capacity/allowances, counting old root plus nested new bytes."""
    def __init__(self,original,run):
        self.original=original;self.run=Path(run);self.output=self.run/OUTPUT_RELATIVE;self.written=0
        need(not self.output.exists(),'A recovery output is never rerun or overwritten: '+str(self.output))
    def write(self,name,data):
        need(Path(name).name==name and not name.startswith('.'),'Explicit flat analysis output filename required')
        used=self.original.derived_pool_bytes(self.run,'statistics');limit=self.original.derived_limit(self.run,'statistics')
        need(used+len(data)<=limit,'Old plus new statistics exceed original P5_boundary_analysis allowance')
        self.original.capacity_check(self.run,next_write_bytes=len(data),active_written_credit=self.written)
        target=self.output/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        self.written+=len(data);return target
    def json(self,name,value):return self.write(name,payload(value))
    def npz(self,name,arrays):
        import numpy as np
        stream=io.BytesIO();np.savez_compressed(stream,**arrays);return self.write(name,stream.getvalue())


def analysis_sources(root=ROOT):
    names=['scripts/p5_analysis_recovery_v3.py','src/push_align/final_statistics_v2.py',
           'tests/test_p5_analysis_recovery_v3.py','tests/test_p5_final_statistics_v2.py']
    return {name:sha(Path(root)/name) for name in names}


def original_numeric_review(run,output,manifest,rows):
    # Extract the untouched reviewer's population, source/file graph and numeric
    # stages. Its public API canonicalizes Path(run), so a narrow path view is
    # required rather than fabricating a run or patching global pathlib.
    from push_align.final_stage_review import _population_rows,_stage_contract,_statistics
    view=StatisticsOutputView(run,output)
    checked=_population_rows(manifest,lambda:rows)
    receipt,_=_stage_contract(view,'statistics',manifest)
    checks=_statistics(view,manifest,checked)
    return {'schema':'p5-derived-scientific-review-output-adapter-v3','status':'PASS','stage':'statistics',
        'frozen_manifest_sha256':sha(Path(run)/'manifest.json'),'statistics_root':str(output),
        'reviewer_source_sha256':sha(ROOT/'src/push_align/final_stage_review.py'),
        'original_functions':['_population_rows','_stage_contract','_statistics'],
        'files_sha256':receipt['files_sha256'],'source_sha256':receipt['source_sha256'],'checks':checks,
        'new_model_forwards':0,'new_physical_calls':0,'new_training_updates':0,
        'limit':'Exact original statistical estimator and raw-review source contract with output-path adaptation; not an independent estimator.'}


def verify_recovery(original,run,manifest):
    output=run/OUTPUT_RELATIVE;receipt=read(output/'analysis_recovery_receipt.json')
    need(sha(output/'analysis_recovery_receipt.json')==read(output/'analysis_recovery_receipt.sha256.json')['sha256'],'Recovery receipt changed')
    need(receipt['schema']==RECEIPT_SCHEMA and receipt['frozen_manifest_sha256']==sha(run/'manifest.json')
         and receipt['statistics_root']==str(output) and receipt['logical_statistics_stage']=='statistics','Recovery receipt identity differs')
    need(receipt['frozen_source_sha256']==manifest['source_sha256'],'Original frozen source map changed')
    need(receipt['analysis_source_sha256']==analysis_sources(),'Additional analysis source version changed')
    for p,h in {**receipt['outputs_sha256'],**receipt['old_partial_statistics_files_sha256']}.items():need(sha(p)==h,'Recovery or old partial output changed: '+p)
    need(receipt['canonical_forward_exact'] is True and receipt['bootstrap_exact_original_forward'] is True
         and receipt['order_reproduction_status']=='PASS_EXACT_FORWARD_REVERSE_SHUFFLE' and receipt['validated_E2_cells']==96,'Canonical equivalence unresolved')
    seal=read(output/'stage_receipt.json')
    need(sha(output/'stage_receipt.json')==read(output/'stage_receipt.sha256.json')['sha256'],'Statistics seal changed')
    verified=VerifiedRows(original,run,manifest)
    verified.graph=dict(receipt['validated_E2_evidence_sha256'])
    rows=verified.reload(manifest['E2'])
    return receipt,original_numeric_review(run,output,manifest,rows)


def statistics(original,run,continuation_record):
    import numpy as np
    from push_align.final_statistics import analyze_E2 as frozen_analyze
    from push_align.final_statistics_v2 import analyze_E2,CANONICAL_ORDER,ANALYSIS_INFRASTRUCTURE_VERSION
    manifest=original.frozen(run);need(len(manifest['E2'])==96,'Original96 E2 population required')
    record=Path(continuation_record).resolve();need(record.is_file(),'Explicit saved analysis-continuation record required')
    writer=StatisticsWriter(original,run)
    old_files=tree_files(run/'statistics')
    need(not any(Path(p).is_relative_to(writer.output) for p in old_files),'Recovery namespace already exists')
    before_sources={**manifest['source_sha256'],**analysis_sources()}
    verified=VerifiedRows(original,run,manifest)
    # Existing partial analysis consumes the same fixed allowance but is
    # explicitly not promoted to a complete scientific stage.
    if old_files:original.register_fixed_files(run,manifest,'E2_statistics',old_files)
    forward=verified.initialize();reverse=verified.reload(list(reversed(manifest['E2'])))
    shuffled=list(manifest['E2']);random.Random(7711).shuffle(shuffled);shuffle=verified.reload(shuffled)
    frozen_summary,frozen_arrays=frozen_analyze(forward,manifest['E2'])
    summary,arrays=analyze_E2(forward,manifest['E2']);back,back_arrays=analyze_E2(reverse,manifest['E2']);mixed,mixed_arrays=analyze_E2(shuffle,manifest['E2'])
    def arrays_exact(left,right):return set(left)==set(right) and all(left[k].dtype==right[k].dtype and left[k].shape==right[k].shape and left[k].tobytes()==right[k].tobytes() for k in left)
    need(exact_json(summary,frozen_summary) and exact_json(summary,back) and exact_json(summary,mixed),'Canonical/original-forward summary exact reproduction failed')
    need(arrays_exact(arrays,frozen_arrays) and arrays_exact(arrays,back_arrays) and arrays_exact(arrays,mixed_arrays),'Canonical bootstrap exact reproduction failed')
    writer.json('summary.json',summary);writer.npz('bootstrap.npz',arrays)
    writer.json('reproduction.json',{'status':'PASS_EXACT_REGENERATION','source_rows':96,'families':12,
        'summary_sha256':sha(writer.output/'summary.json'),'bootstrap_sha256':sha(writer.output/'bootstrap.npz')})
    sources={str(run/'E2'/c['cell_id']/'worker_result.json'):sha(run/'E2'/c['cell_id']/'worker_result.json') for c in manifest['E2']}
    stage_files={str(writer.output/name):sha(writer.output/name) for name in ('summary.json','bootstrap.npz','reproduction.json')}
    writer.json('stage_receipt.json',{'schema':'p5-derived-stage-seal-v1','stage':'statistics',
        'frozen_manifest_sha256':sha(run/'manifest.json'),'files_sha256':stage_files,'source_sha256':sources})
    writer.json('stage_receipt.sha256.json',{'sha256':sha(writer.output/'stage_receipt.json')})
    review=original_numeric_review(run,writer.output,manifest,forward)
    need(old_files=={p:h for p,h in tree_files(run/'statistics').items() if not Path(p).is_relative_to(writer.output)},'Old partial statistics were changed')
    original.verify(before_sources)
    receipt={'schema':RECEIPT_SCHEMA,'infrastructure_version':VERSION,'logical_statistics_stage':'statistics',
        'statistics_root':str(writer.output),'frozen_manifest_sha256':sha(run/'manifest.json'),
        'analysis_source_sha256':analysis_sources(),'frozen_source_sha256':manifest['source_sha256'],
        'continuation_record':{'path':str(record),'sha256':sha(record)},'analysis_infrastructure_version':ANALYSIS_INFRASTRUCTURE_VERSION,
        'canonical_order':CANONICAL_ORDER,'canonical_forward_exact':True,'bootstrap_exact_original_forward':True,
        'order_reproduction_status':'PASS_EXACT_FORWARD_REVERSE_SHUFFLE','validated_E2_cells':96,
        'bootstrap_array_count':len(arrays),'original_scientific_review':review,
        'old_partial_statistics_files_sha256':old_files,'old_partial_charged_without_scientific_adoption':True,
        'outputs_sha256':tree_files(writer.output),'validation_reuse':verified.summary(),
        'validated_E2_evidence_sha256':dict(verified.graph),'validated_E2_raw_once':True,
        'execution':{'new_physics':0,'new_model_calls':0,'new_training':0,'physical_cell_retries':0,'frozen_source_edits':0},
        'scope':'Post-freeze report-reduction infrastructure only. Original forward numbers, bootstrap,96 raw outcomes and frozen scientific declarations unchanged.'}
    writer.json('analysis_recovery_receipt.json',receipt)
    writer.json('analysis_recovery_receipt.sha256.json',{'sha256':sha(writer.output/'analysis_recovery_receipt.json')})
    original.register_fixed_files(run,manifest,'E2_statistics',tree_files(run/'statistics'))
    return {'status':'PASS_CANONICAL_STATISTICS_RECOVERY','statistics_root':str(writer.output),
        'analysis_recovery_receipt':str(writer.output/'analysis_recovery_receipt.json'),'validation_reuse':verified.summary()}

def complete(original,run):
    manifest=original.frozen(run);target=run/'execution_complete.analysis-v3.json'
    need(not target.exists() and not target.with_name('execution_complete.analysis-v3.sha256.json').exists(),'Completion receipt is never replaced')
    need(not (run/'execution_complete.json').exists(),'Original completion exists; review rather than silently supersede it')
    receipt,review=verify_recovery(original,run,manifest)
    original.fixed_spending(run,manifest,deep=True)
    # E1/statistics were already adopted through the original frozen code.
    # Preserve their exact original stage seals without repeating576 reads.
    for name in ('e1_statistics','fixed_prediction'):
        folder=run/name;seal=read(folder/'stage_receipt.json')
        need(sha(folder/'stage_receipt.json')==read(folder/'stage_receipt.sha256.json')['sha256']
             and seal['stage']==name and seal['frozen_manifest_sha256']==sha(run/'manifest.json'),'Original derived-stage seal unresolved: '+name)
        for p,h in {**seal['files_sha256'],**seal['source_sha256']}.items():need(sha(p)==h,'Original derived source/output changed: '+p)
    e1=read(run/'E1/stage_receipt.json');need(len(e1['completed_repeat_ids'])==576 and e1['frozen_manifest_sha256']==sha(run/'manifest.json'),'Original E1 full denominator unresolved')
    state=original.validated_progress(run,manifest)
    need(len(state['E2_done'])==96 and len(state['E1_done'])==576 and len(state['application_done'])==8
         and state['replay_selection'] is not None and set(state['replay_done'])=={r['cell_id'] for r in state['replay_selection']},'Full remaining stage population unresolved')
    for phase,key in (('replay','replay_done'),('application','application_done')):
        for cell in state[key]:original.verified_result(run,phase,cell,manifest,deep=True)
    original.capacity_check(run,next_write_bytes=256*1024)
    result={'schema':COMPLETION_SCHEMA,'status':'ALL_DECLARED_EXECUTION_STAGES_RECORDED','infrastructure_version':VERSION,
        'at_utc':datetime.now(timezone.utc).isoformat(),'frozen_manifest_sha256':sha(run/'manifest.json'),
        'logical_statistics_stage':'statistics','statistics_root':str(run/OUTPUT_RELATIVE),
        'analysis_recovery_receipt':{'path':str(run/OUTPUT_RELATIVE/'analysis_recovery_receipt.json'),
            'sha256':sha(run/OUTPUT_RELATIVE/'analysis_recovery_receipt.json')},
        'analysis_source_sha256':analysis_sources(),'frozen_source_sha256':manifest['source_sha256'],
        'remaining':original.remaining_estimate(run),'E2_reused_original_raw_review_by_exact_SHA_graph':True,
        'original_scientific_review':review,'personal_understanding':'NOT_ASSESSED',
        'scope':'Original frozen physical prediction/replay/application commands completed separately; original root statistics,manifest and source bundle remain untouched. Report/visual/claim acceptance remains separate.'}
    for path,value in ((target,result),(target.with_name('execution_complete.analysis-v3.sha256.json'),None)):
        data=payload(value if value is not None else {'sha256':sha(target)})
        original.fixed_write_guard(run,manifest,len(data))
        with path.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    return {'status':result['status'],'execution_receipt':str(target),'sha256':sha(target)}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['statistics','complete'])
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--continuation-record',type=Path)
    args=parser.parse_args(argv);run=args.run.resolve()
    need(run.parent==ROOT/'runs' and run.name.startswith('p5-final-'),'Explicit managed final run required')
    original=load_original()
    if args.command=='statistics':need(args.continuation_record is not None,'--continuation-record is required');result=statistics(original,run,args.continuation_record)
    else:result=complete(original,run)
    print(json.dumps(result,ensure_ascii=False));return 0


if __name__=='__main__':raise SystemExit(main())
