#!/usr/bin/env python3
"""Idle-only supplemental P5 accounting; frozen code and ledgers stay unchanged.

기존 보고 원장의 보존 검사를 인용한다. 원시는 metadata만 조사하고 재해시하지
않는다. 전체 run *.log는 SHA를 기록하며 캐시·root stdout은 scratch, 형제 worker
로그 등 P5 운영 누락은 원래 P5 한도로 구분한다. --self-test는 합성만 사용한다.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import unittest

SCRATCH = 32*1024**2
P5_LIMIT = 201326592
REPORT = 64*1024**2
TAIL = 1024**2
RESERVE = 2*1024**3
SCOPES = ('E2','application','replays')
SOURCE_ROOTS = ('src','scripts','tools','tests')


def need(okay, message):
    if not okay: raise ValueError(message)


def encode(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def parse(payload):
    def unique(pairs):
        result={}
        for key,value in pairs:
            need(key not in result,'JSON 중복 key 금지: '+key);result[key]=value
        return result
    return json.loads(payload,object_pairs_hook=unique,parse_constant=lambda value:
        (_ for _ in ()).throw(ValueError('비유한 JSON 금지: '+value)))


class Scan:
    def __init__(self,root):self.root=root;self.seen={};self.bindings={};self.scopes={}
    def path(self,value):
        path=Path(value);path=path if path.is_absolute() else self.root/path
        path=Path(os.path.abspath(path));need(path.is_relative_to(self.root),'프로젝트 외 입력 금지: '+str(path))
        for part in (path,*path.parents):
            need(not part.is_symlink(),'심볼릭 링크 금지: '+str(part))
            if part==self.root:break
        need(path.resolve()==path,'별칭 경로 금지: '+str(path));return path
    def file(self,value):
        path=self.path(value);info=path.stat()
        need(stat.S_ISREG(info.st_mode) and info.st_nlink==1,'정규 단일-link 파일 필요: '+str(path))
        signature=(info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
        need(str(path) not in self.seen or self.seen[str(path)]==signature,'검사 중 파일 변경: '+str(path))
        self.seen[str(path)]=signature
        return {'path':str(path),'size_bytes':info.st_size,'mtime_ns':info.st_mtime_ns}
    def sha(self,value):
        row=self.file(value);path=row['path'];value=digest(path);self.file(path)
        need(path not in self.bindings or self.bindings[path]==value,'입력 SHA 변경: '+path)
        self.bindings[path]=value;return value
    def read(self,value):
        expected=self.sha(value);data=parse(Path(value if Path(value).is_absolute() else self.root/value).read_text())
        need(self.sha(value)==expected,'JSON read 중 변경');return data
    def tree(self,value,kind='all'):
        folder=self.path(value);result={}
        if folder.exists():
            need(folder.is_dir(),'조사 root는 디렉터리여야 함: '+str(folder))
            for current,dirs,files in os.walk(folder,followlinks=False):
                for name in dirs:need(not (Path(current)/name).is_symlink(),'디렉터리 symlink 금지')
                for name in files:
                    path=Path(current)/name
                    need(not path.is_symlink(),'파일 symlink 금지')
                    if kind=='pyc' and path.suffix!='.pyc':continue
                    row=self.file(path);result[str(path)]=row
        return result
    def scope(self,value,kind='all'):
        path=self.path(value);rows=self.tree(path,kind);self.scopes[str(path),kind]=rows;return rows
    def guard(self):
        for (path,kind),before in self.scopes.items():
            need(self.tree(path,kind)==before,'조사 중 목록/크기/시각 변경: '+path)
        for path in list(self.seen):self.file(path)
        for path,expected in self.bindings.items():need(digest(path)==expected,'최종 입력 SHA 변경: '+path)


def nbytes(rows):return sum(row['size_bytes'] for row in rows.values())
def inside(path,folder):return Path(path).is_relative_to(folder)


def build(run,base_ledger,out,*,idle_confirmed=False,free_bytes=None,_before_guard=None):
    need(idle_confirmed,'모든 관련 writer 종료 후 --confirm-idle 명시 필요; 실행 중 조사 금지')
    run_arg=Path(run);need(not run_arg.is_symlink(),'run symlink 금지')
    run=run_arg.resolve();root=run.parent.parent;scan=Scan(root);run=scan.path(run)
    need(run.parent.name=='runs','프로젝트 runs/<run> 경로 필요')
    out=scan.path(out);base_ledger=scan.path(base_ledger)
    need(out.suffix=='.json' and not out.exists() and out!=base_ledger
         and out.relative_to(root).parts[0] in ('reports','output'),'별도 reports/output 새 JSON 필요')
    base=scan.read(base_ledger);manifest=scan.read(run/'manifest.json');freeze=scan.read(run/'freeze_receipt.json')
    need(base.get('schema')=='p5-final-report-resource-ledger-v1' and base.get('accepted') is True
         and base.get('status')=='PASS' and base.get('run')==str(run),'동일 run의 PASS 기존 원장 필요')
    need(manifest.get('schema')=='p5-frozen-final-manifest-v1'
         and manifest.get('status')=='FROZEN_BEFORE_FINAL_OUTCOMES'
         and len(manifest.get('source_sha256',{}))==524
         and freeze.get('sha256')==scan.sha(run/'manifest.json')==base['frozen_manifest']['sha256']
         and base['freeze_receipt']['sha256']==scan.sha(run/'freeze_receipt.json'),'원래 frozen identity 불일치')
    integrity=base.get('integrity',{})
    for name,count in (('source524',524),('parent275',275)):
        proof=integrity.get(name,{})
        need(proof.get('all_unchanged') is True and proof.get('expected_file_count')==count
             and proof.get('unchanged_file_count')==count,'기존 보존 검사 불충족: '+name)
    need(integrity['parent275'].get('inventory_exact') is True
         and integrity.get('old_pdf',{}).get('all_unchanged') is True,'parent/PDF 보존 검사 불충족')
    plan=scan.read(scan.path(manifest['resource_plan_path']))
    allowance=plan.get('fixed_allowances_bytes',{})
    need(allowance.get('P5_boundary_analysis')==P5_LIMIT and allowance.get('working_disk_scratch')==SCRATCH,
         '원래 P5 201326592B 및 scratch32MiB 한도를 변경할 수 없음')
    totals=base['totals'];base_files={}
    for row in base['files']:
        path=str(scan.path(row['path']));need(path not in base_files,'기존 원장 중복 경로')
        need(scan.file(path)['size_bytes']==row['size_bytes'],'기존 원장 크기 변경: '+path)
        base_files[path]=row
        if Path(path).suffix=='.py':need(scan.sha(path)==row['sha256'],'기존 새 source SHA 변경: '+path)
    base_scratch={p:r for p,r in base_files.items() if 'working_disk_scratch' in r['pools']}
    base_report={p:r for p,r in base_files.items() if 'future_report_archive_QA' in r['pools']}
    need(totals['working_disk_scratch']['actual_bytes']==nbytes(base_scratch)
         and totals['working_disk_scratch']['limit_bytes']==SCRATCH,'기존 scratch 분모/합계 불일치')
    own=base['ledger_self']
    need(str(scan.path(own['path']))==str(base_ledger) and own['size_bytes']==scan.file(base_ledger)['size_bytes']
         and totals['future_report_archive_QA']['actual_bytes']==nbytes(base_report)+own['size_bytes']
         and totals['future_report_archive_QA']['limit_bytes']==REPORT
         and totals['future_report_archive_QA']['reserved_tail_bytes']==TAIL,'기존 보고 원장 자기크기/합계 불일치')
    need(nbytes(base_scratch)<=SCRATCH
         and totals['future_report_archive_QA']['actual_bytes']+TAIL<=REPORT,
         '기존 PASS 표기와 실제 한도/예약 tail 계산이 불일치')
    generator=base.get('generator',{})
    need(scan.sha(generator['path'])==generator['sha256'],'기존 원장 generator source 변경')
    tool=Path(__file__).resolve();tool_sha=digest(tool)
    if not manifest.get('fixture_only'):
        need(str(tool) in base_scratch and scan.sha(tool)==base_scratch[str(tool)]['sha256'],
             '보완 도구 source까지 포함한 기존 원장을 먼저 생성해야 함')
    inventory=scan.scope(run)
    pyc={}
    for name in SOURCE_ROOTS:pyc.update(scan.scope(root/name,'pyc'))
    state=scan.read(run/'progress.json');progress_seal=scan.read(run/'progress_receipt.json')
    need(progress_seal.get('sha256')==scan.sha(run/'progress.json'),'progress SHA receipt 불일치')
    operational=state.get('operational_bytes')
    need(type(operational) is int and 0<=operational<=P5_LIMIT,'원래 operational counter 불일치')
    statistics={p:r for p,r in inventory.items() if inside(p,run/'statistics')}
    fixed_paths={p:r for p,r in inventory.items() if Path(p).parent==run
                 and Path(p).match('fixed_spend*.json')}
    frozen_bindings={str(root/p):h for p,h in {**manifest['source_sha256'],
        **manifest.get('runtime_source_sha256',{})}.items()}
    frozen_bindings[str(run/'manifest.json')]=scan.bindings[str(run/'manifest.json')]
    registered={}
    for path in fixed_paths:
        value=scan.read(path)
        for entry in value.get('entries',[]):
            target=str(scan.path(run/entry['path']))
            need(target not in registered,'fixed-spend 중복 파일 경로');registered[target]=entry
    folders=[];measured=set();component_reviews=[]
    for path in sorted(inventory):
        p=Path(path)
        if p.name!='component_budget_review.json' or p.relative_to(run).parts[0] not in SCOPES:continue
        folder=p.parent;review=scan.read(p)
        need(review.get('status')=='PASS_MEASURED_COMPONENT_ALLOWANCES'
             and type(review.get('measured_component_paths')) is list,'검증된 component path 목록 필요')
        need(all(not folder.is_relative_to(other) and not other.is_relative_to(folder) for other in folders),
             '중첩/중복 cell 폴더 금지')
        phase={'E2':'E2','replays':'replay','application':'application'}[p.relative_to(run).parts[0]]
        worker_path=folder/'worker_result.json';worker=scan.read(worker_path)
        receipt=scan.read(folder/'worker_result_receipt.json')
        need(receipt.get('sha256')==scan.sha(worker_path)
             and worker.get('component_review_sha256')==scan.sha(p),'worker/component SHA 봉인 불일치')
        cell=worker.get('cell',{});family=cell.get('family_id');controller=cell.get('controller_id')
        need(worker.get('phase')==phase and folder==run/('replays' if phase=='replay' else phase)/str(family)/str(controller)
             and worker.get('status') in ('ACCEPT_COMPLETED_RECORDING','ACCEPT_COMPLETED_TASK_FAILURE')
             and worker.get('row',{}).get('execution_status')=='completed'
             and all(worker['row'].get(k)==v for k,v in cell.items()),'worker phase/cell/완료 identity 불일치')
        if phase=='application':
            f=manifest['application_family'];expected_cell={'family_id':f['family_id'],
                'episode_id':f.get('episode_id',f['family_id']),'episode_index':12,
                'stratum':f.get('stratum','application'),'controller_id':controller,
                'cell_id':f['family_id']+'/'+str(controller)}
        else:
            matches=[r for r in manifest['E2'] if r['cell_id']==cell.get('cell_id')]
            need(len(matches)==1,'원래 E2 identity 하나 필요');expected_cell=matches[0]
        need(cell==expected_cell,'원래 동결 cell identity 불일치')
        record_path=scan.path(worker['record_manifest'])
        need(record_path==folder/'manifest.json' and scan.sha(record_path)==worker.get('record_manifest_sha256'),
             'record manifest 경로/SHA 불일치')
        record=scan.read(record_path)
        expected_identity={'study':'P5-final','family_id':family,'controller_id':controller,
            'model':None if controller=='G' else manifest['model_identities'][controller],
            'normalizer':None if controller=='G' else manifest['normalizer_identity'],
            'final_manifest_sha256':scan.bindings[str(run/'manifest.json')]}
        need(record.get('record_kind')==('fresh_controller_replay' if phase=='replay' else 'reference_controller_execution')
             and record.get('identity')==expected_identity and record.get('source_bindings')==frozen_bindings,
             'record의 동결 identity/source metadata 불일치')
        declared=set(record.get('raw',{}).get('files_sha256',{})) if record.get('raw') else set()
        declared.add(record['witness_path'])
        if record.get('plan_index') and controller!='G':declared.update(record['plan_index']['files_sha256'])
        for trace in record.get('full_traces',[]):declared.update(trace['files_sha256'])
        need(set(review['measured_component_paths'])==declared,'component 제외 목록이 봉인 record와 다름')
        folders.append(folder);local=[]
        for raw in review['measured_component_paths']:
            target=str(scan.path(raw));need(target in inventory and inside(target,folder),'component path 범위/존재 불일치')
            need(target not in measured,'중복 component path');measured.add(target);local.append(target)
        key={'E2':'E2_done','replay':'replay_done','application':'application_done'}[phase]
        identity=worker.get('cell',{}).get('controller_id' if phase=='application' else 'cell_id')
        need(identity in state.get(key,[]),'아직 채택되지 않은 cell은 idle 보완 대상이 아님')
        component_reviews.append({'path':str(p),'sha256':scan.bindings[str(p)],'cell_folder':str(folder),
             'component_path_count':len(local),'component_path_list_sha256':hashlib.sha256(encode(sorted(local))).hexdigest()})
    cell_all={p:r for p,r in inventory.items() if any(inside(p,folder) for folder in folders)}
    cell_operations={p:r for p,r in cell_all.items() if p not in measured}
    scope_files={p:r for p,r in inventory.items() if Path(p).relative_to(run).parts[0] in SCOPES}
    outside_cells={p:r for p,r in scope_files.items() if p not in cell_all}
    # Root control receipts have no demonstrated counter attribution. Count them
    # conservatively again, except the fixed JSON files already in the formula.
    root_uncertain={p:r for p,r in inventory.items() if Path(p).parent==run
                    and Path(p).suffix!='.log' and p not in fixed_paths}
    p5_extras={**outside_cells,**root_uncertain}
    scratch_add={};logs=[]
    for path,row in pyc.items():
        if path not in base_scratch:scratch_add[path]={**row,'reason':'scoped_all_pyc_conservative'}
    for path,row in sorted(inventory.items()):
        if Path(path).suffix!='.log':continue
        if Path(path).parent==run:
            category='scratch_root_stdout'
            if path not in base_scratch:scratch_add[path]={**row,'reason':category}
        elif path in statistics:category='already_P5_statistics_tree'
        elif path in cell_operations:category='already_P5_cell_operational_counter_assumption'
        elif path in outside_cells:category='additional_P5_sibling_or_unowned_operational'
        elif Path(path).relative_to(run).parts[0] in ('E1','e1_statistics','fixed_prediction'):
            category='already_E1_tree_formula'
        elif path in base_files:category='already_base_report_or_scratch'
        elif inside(path,run/'source') and path in frozen_bindings and scan.sha(path)==frozen_bindings[path]:
            category='already_frozen_source_bundle'
        else:
            category='scratch_uncertain_log_conservative'
            if path not in base_scratch:scratch_add[path]={**row,'reason':category}
        logs.append({**row,'sha256':scan.sha(path),'accounting_class':category,
            'registered_fixed_pool':registered.get(path,{}).get('pool'),
            'already_base_scratch':path in base_scratch})
    current_cell=nbytes(cell_operations);cell_growth=max(0,current_cell-operational)
    original_p5=operational+nbytes(statistics)+nbytes(fixed_paths)
    supplemented_p5=original_p5+cell_growth+nbytes(p5_extras)
    scratch_total=nbytes(base_scratch)+nbytes(scratch_add)
    free=shutil.disk_usage(root).free if free_bytes is None else free_bytes
    need(type(free) is int and free>=0,'유효 free bytes 필요')
    if _before_guard:_before_guard()
    scan.guard();need(digest(tool)==tool_sha,'보완 도구 source 변경')
    value={'schema':'p5-idle-operational-resource-supplement-v1','status':'HOLD','accepted':False,
        'checked_at_utc':datetime.now(timezone.utc).isoformat(),'fixture_only':bool(manifest.get('fixture_only')),
        'run':str(run),'generator':{'path':str(tool),'sha256':tool_sha},
        'base_ledger':{'path':str(base_ledger),'sha256':scan.bindings[str(base_ledger)]},
        'preservation_checks_reused':{'scope':'기존 원장의 source524/parent275/oldPDF 보존 PASS 인용; 원시/부모/모델 재해시 없음',
             'integrity':integrity},
        'idle':{'caller_confirmed':True,'pre_post_census_guard':'PASS','process_liveness_proven':False},
        'working_disk_scratch':{'limit_bytes':SCRATCH,'base_actual_bytes':nbytes(base_scratch),
            'scoped_all_pyc_bytes':nbytes(pyc),'scoped_all_pyc_count':len(pyc),
            'additional_unique_bytes':nbytes(scratch_add),'actual_bytes':scratch_total,
            'remaining_bytes':SCRATCH-scratch_total,'fits':scratch_total<=SCRATCH,
            'additional_files':sorted(scratch_add.values(),key=lambda r:r['path']),
            'pyc_inventory':[{**r,'already_base_scratch':p in base_scratch} for p,r in sorted(pyc.items())]},
        'P5_boundary_analysis':{'limit_bytes':allowance['P5_boundary_analysis'],
            'original_formula':'progress.operational_bytes + statistics recursive bytes + fixed_spend*.json bytes',
            'original_operational_counter_bytes':operational,'statistics_tree_bytes':nbytes(statistics),
            'fixed_ledger_and_receipt_bytes':nbytes(fixed_paths),'original_formula_bytes':original_p5,
            'current_cell_noncomponent_bytes':current_cell,'positive_cell_growth_conservative_bytes':cell_growth,
            'counter_exceeds_current_cell_bytes':max(0,operational-current_cell),
            'additional_unique_operational_bytes':nbytes(p5_extras),
            'actual_conservative_bytes':supplemented_p5,'remaining_bytes':P5_LIMIT-supplemented_p5,
            'fits':supplemented_p5<=P5_LIMIT,'component_reviews':component_reviews,
            'additional_files':[{**r,'reason':'outside_known_cell' if p in outside_cells else 'uncertain_root_control_metadata',
                'registered_fixed_pool':registered.get(p,{}).get('pool'),
                'conservative_duplicate_possible':p in registered or p in root_uncertain}
                for p,r in sorted(p5_extras.items())]},
        'all_run_log_inventory':logs,'input_metadata_and_log_sha256':dict(sorted(scan.bindings.items())),
        'inventory':{'run_file_count':len(inventory),'run_metadata_list_sha256':hashlib.sha256(encode(inventory)).hexdigest(),
             'raw_file_contents_hashed':0,'log_count':len(logs),'log_total_bytes':sum(r['size_bytes'] for r in logs)},
        'scope_limits':['외부 보완 회계이며 frozen fixed-spend scope/credit를 변경하지 않음.',
            'P5 cell 기존분은 progress counter를 유지하고 현재 증가분만 보수 추가; 감소분을 credit로 쓰지 않음.',
            'component_review 명시 경로만 unit-priced 원시로 제외; 새 partial/미확인 파일은 제외하지 않음.',
            'root 기타metadata는 attribution 미확인으로 P5에 보수 중복; root stdout은 scratch에 구분.',
            'E1/source/model/unit-priced budget의 전체 재검증이 아님; 기존 과학/보존 원장과 최종 독립 검토를 따름.',
            '호출자 idle 선언+pre/post snapshot이며 OS lock/원자적 디스크 예약/미래 증가 상한이 아님.'],
        'supplement_self':{'path':str(out),'size_bytes':0,'sha256':None,
             'hash_unavailable_reason':'자기 SHA는 외부 최종 receipt로 바인딩; 기존 보고 원장1MiB tail에서 실제 크기 차감'},
        'future_report_archive_QA':{},'disk':{},'new_model_calls':0,'new_physics_steps':0,'frozen_or_ledger_edits':0}
    for _ in range(32):
        size=value['supplement_self']['size_bytes'];report_total=totals['future_report_archive_QA']['actual_bytes']+size
        value['future_report_archive_QA']={'base_actual_bytes':totals['future_report_archive_QA']['actual_bytes'],
            'supplement_actual_bytes':size,'actual_bytes_after_supplement':report_total,'limit_bytes':REPORT,
            'reserved_tail_remaining_bytes':TAIL-size,'fits':report_total<=REPORT and size<=TAIL}
        value['disk']={'free_bytes_before_write':free,'hard_reserve_bytes':RESERVE,
            'predicted_free_after_write':free-size,'fits':free-size>=RESERVE}
        value['accepted']=all((scratch_total<=SCRATCH,supplemented_p5<=P5_LIMIT,
            value['future_report_archive_QA']['fits'],value['disk']['fits']))
        value['status']='PASS' if value['accepted'] else 'HOLD'
        payload=encode(value)
        if len(payload)==size:return value,payload,scan
        value['supplement_self']['size_bytes']=len(payload)
    raise RuntimeError('보완 원장 자기크기 fixedpoint 수렴 실패')


def write(run,base_ledger,out,*,idle_confirmed=False):
    value,payload,scan=build(run,base_ledger,out,idle_confirmed=idle_confirmed)
    need(len(payload)<=TAIL,'보완 원장1MiB 상한; 기존 파일 보존 후 범위 검토 필요')
    scan.guard();need(shutil.disk_usage(scan.root).free>=RESERVE+len(payload),'쓰기 직전2GiB reserve 부족')
    out=scan.path(out);need(not out.exists(),'이전 출력 보존; 새 버전 경로 필요')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
    need(out.stat().st_size==value['supplement_self']['size_bytes'],'보완 원장 실측 크기 불일치')
    return value


def fixture(root,*,precount_cache=False):
    run=root/'runs/p5-final-fixture-02'
    def put(path,data=b'x'):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);return path
    generator=put(root/'tools/base.py',b'# fixture source')
    plan=put(root/'reports/plan.json',encode({'fixed_allowances_bytes':{'P5_boundary_analysis':P5_LIMIT,'working_disk_scratch':SCRATCH}}))
    cell_id={'family_id':'family','cell_id':'family/G','controller_id':'G'}
    manifest=put(run/'manifest.json',encode({'schema':'p5-frozen-final-manifest-v1','status':'FROZEN_BEFORE_FINAL_OUTCOMES',
        'fixture_only':True,'source_sha256':{f'fixture-{i}':'0'*64 for i in range(524)},
        'E2':[cell_id],'resource_plan_path':str(plan.relative_to(root))}))
    freeze=put(run/'freeze_receipt.json',encode({'sha256':digest(manifest)}))
    cell=run/'E2/family/G';raw=put(cell/'raw.bin',b'SYNTHETIC RAW - NEVER HASH')
    witness=put(cell/'witness.bin',b'SYNTHETIC WITNESS - NEVER HASH')
    bound={str(root/f'fixture-{i}'):'0'*64 for i in range(524)};bound[str(manifest)]=digest(manifest)
    record=put(cell/'manifest.json',encode({'record_kind':'reference_controller_execution',
        'identity':{'study':'P5-final','family_id':'family','controller_id':'G','model':None,'normalizer':None,
            'final_manifest_sha256':digest(manifest)},'source_bindings':bound,
        'raw':{'files_sha256':{str(raw):'0'*64}},'witness_path':str(witness),'full_traces':[]}))
    component=put(cell/'component_budget_review.json',encode({'status':'PASS_MEASURED_COMPONENT_ALLOWANCES',
        'measured_component_paths':[str(raw),str(witness)]}))
    worker=put(cell/'worker_result.json',encode({'cell':cell_id,'phase':'E2','status':'ACCEPT_COMPLETED_RECORDING',
        'row':{**cell_id,'execution_status':'completed'},'record_manifest':str(record),
        'record_manifest_sha256':digest(record),'component_review_sha256':digest(component)}))
    put(cell/'worker_result_receipt.json',encode({'sha256':digest(worker)}))
    ops=sum(p.stat().st_size for p in cell.iterdir() if p not in (raw,witness))
    progress=put(run/'progress.json',encode({'operational_bytes':ops,'E2_done':['family/G']}))
    put(run/'progress_receipt.json',encode({'sha256':digest(progress)}))
    put(run/'E2/family/G.worker.log',b'worker log')
    put(run/'replay-v2.log',b'root stdout')
    put(run/'statistics/recovery-v2/summary.json',b'{}')
    put(run/'fixed_spend.json',encode({'entries':[]}));put(run/'fixed_spend.receipt.json',b'{}')
    cache=put(root/'tools/__pycache__/cache.pyc',b'cache123')
    base_path=root/'reports/base-ledger.json';scratch=[{'path':str(generator),'size_bytes':generator.stat().st_size,
        'sha256':digest(generator),'pools':['working_disk_scratch']}]
    if precount_cache:scratch.append({'path':str(cache),'size_bytes':cache.stat().st_size,
        'sha256':digest(cache),'pools':['working_disk_scratch']})
    value={'schema':'p5-final-report-resource-ledger-v1','accepted':True,'status':'PASS','run':str(run),
        'frozen_manifest':{'sha256':digest(manifest)},'freeze_receipt':{'sha256':digest(freeze)},'files':scratch,
        'generator':{'path':str(generator),'sha256':digest(generator)},
        'integrity':{'source524':{'all_unchanged':True,'expected_file_count':524,'unchanged_file_count':524},
            'parent275':{'all_unchanged':True,'expected_file_count':275,'unchanged_file_count':275,'inventory_exact':True},
            'old_pdf':{'all_unchanged':True}},'ledger_self':{'path':str(base_path),'size_bytes':0},'totals':{}}
    for _ in range(32):
        size=value['ledger_self']['size_bytes'];value['totals']={'working_disk_scratch':{'actual_bytes':nbytes({r['path']:r for r in scratch}),'limit_bytes':SCRATCH},
            'future_report_archive_QA':{'actual_bytes':size,'limit_bytes':REPORT,'reserved_tail_bytes':TAIL}}
        data=encode(value)
        if len(data)==size:break
        value['ledger_self']['size_bytes']=len(data)
    put(base_path,data)
    return run,base_path,root/'reports/supplement.json',cache,raw


class SyntheticTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.args=fixture(self.root)
    def build(self,**kw):return build(*self.args[:3],idle_confirmed=True,free_bytes=8*RESERVE,**kw)
    def test_split_dedup_and_raw_not_hashed(self):
        value,payload,_=self.build();self.assertTrue(value['accepted'])
        self.assertEqual(value['supplement_self']['size_bytes'],len(payload))
        self.assertNotIn(str(self.args[4]),value['input_metadata_and_log_sha256'])
        logs={Path(r['path']).name:r for r in value['all_run_log_inventory']}
        self.assertEqual(logs['replay-v2.log']['accounting_class'],'scratch_root_stdout')
        self.assertEqual(logs['G.worker.log']['accounting_class'],'additional_P5_sibling_or_unowned_operational')
        self.assertEqual(value['working_disk_scratch']['additional_unique_bytes'],len(b'cache123')+len(b'root stdout'))
        self.assertEqual(value['P5_boundary_analysis']['positive_cell_growth_conservative_bytes'],0)
        already=fixture(self.root/'precount',precount_cache=True)
        second,_,_=build(*already[:3],idle_confirmed=True,free_bytes=8*RESERVE)
        self.assertEqual(second['working_disk_scratch']['additional_unique_bytes'],len(b'root stdout'))
        self.assertTrue(second['working_disk_scratch']['pyc_inventory'][0]['already_base_scratch'])
        unknown=self.args[0]/'unclassified/diagnostic.log';unknown.parent.mkdir();unknown.write_bytes(b'unknown')
        (self.args[0]/'fixed_spend.json').write_bytes(encode({'entries':[{
            'path':str(unknown.relative_to(self.args[0])),'pool':'source_and_model_bundle'}]}))
        third,_,_=self.build()
        row=next(r for r in third['all_run_log_inventory'] if r['path']==str(unknown))
        self.assertEqual(row['accounting_class'],'scratch_uncertain_log_conservative')
        self.assertEqual(third['working_disk_scratch']['additional_unique_bytes'],len(b'cache123root stdoutunknown'))
    def test_unknown_new_cell_file_included_not_suffix_excluded(self):
        (self.args[4].parent/'new-partial.npz').write_bytes(b'unknown partial')
        value,_,_=self.build()
        self.assertEqual(value['P5_boundary_analysis']['positive_cell_growth_conservative_bytes'],len(b'unknown partial'))
        component=self.args[4].parent/'component_budget_review.json';review=parse(component.read_text())
        review['measured_component_paths'].append(str(self.args[4].parent/'new-partial.npz'));component.write_bytes(encode(review))
        # Even a resealed worker cannot silently relabel an undeclared file as a paid component.
        worker_path=component.parent/'worker_result.json';worker=parse(worker_path.read_text())
        worker['component_review_sha256']=digest(component);worker_path.write_bytes(encode(worker))
        (component.parent/'worker_result_receipt.json').write_bytes(encode({'sha256':digest(worker_path)}))
        with self.assertRaises(ValueError):self.build()
    def test_mutation_and_symlink_rejected(self):
        with self.assertRaises(ValueError):self.build(_before_guard=lambda:(self.args[0]/'replay-v2.log').write_bytes(b'grew'))
        self.args[3].unlink();self.args[3].symlink_to(self.args[4])
        with self.assertRaises(ValueError):self.build()
    def test_scratch_and_P5_limits_hold(self):
        low,_,_=build(*self.args[:3],idle_confirmed=True,free_bytes=RESERVE)
        self.assertFalse(low['disk']['fits']);self.assertFalse(low['accepted'])
        large=self.root/'src/__pycache__/large.pyc';large.parent.mkdir(parents=True,exist_ok=True)
        with large.open('wb') as stream:stream.truncate(SCRATCH+1)
        value,_,_=self.build();self.assertFalse(value['working_disk_scratch']['fits'])
        large.unlink()
        with (self.args[0]/'E2/family/unknown.partial').open('wb') as stream:stream.truncate(P5_LIMIT+1)
        value,_,_=self.build();self.assertFalse(value['P5_boundary_analysis']['fits'])
    def test_idle_preservation_and_exclusive_write(self):
        with self.assertRaises(ValueError):build(*self.args[:3])
        write(*self.args[:3],idle_confirmed=True)
        with self.assertRaises(ValueError):write(*self.args[:3],idle_confirmed=True)
        base=parse(self.args[1].read_text());base['integrity']['source524']['all_unchanged']=False
        self.args[1].write_bytes(encode(base))
        with self.assertRaises(ValueError):build(self.args[0],self.args[1],self.root/'reports/new.json',idle_confirmed=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path);parser.add_argument('--base-ledger',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--confirm-idle',action='store_true');parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        outcome=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticTests))
        return 0 if outcome.wasSuccessful() else 2
    need(all((args.run,args.base_ledger,args.out)),'--run --base-ledger --out 필요')
    value=write(args.run,args.base_ledger,args.out,idle_confirmed=args.confirm_idle)
    print(json.dumps({'status':value['status'],'accepted':value['accepted'],
        'scratch_bytes':value['working_disk_scratch']['actual_bytes'],
        'P5_conservative_bytes':value['P5_boundary_analysis']['actual_conservative_bytes']},ensure_ascii=False))
    return 0 if value['accepted'] else 2


if __name__=='__main__':raise SystemExit(main())
