#!/usr/bin/env python3
"""P5 보고 산출물64MiB·작업 scratch32MiB 독립 원장.

기존 실험 코드를 import하지 않는다. 파일 logical size와 SHA256을 기록한다.
원본 보고서 중 frozen source graph와 SHA까지 같은 파일만 역사 보존으로 제외한다.
실제 결과는 root가 명시한 --extra 전체 목록과 함께 검사한다. --self-test는 합성만 사용한다.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import unittest

REPORT_LIMIT=64*1024**2
SCRATCH_LIMIT=32*1024**2
TAIL=1024**2
HARD_RESERVE=2*1024**3
SOURCE_COUNT=524
PARENT_COUNT=275
SOURCE_ROOTS=('tools','tests','scripts','src')


def need(okay,message):
    if not okay:raise ValueError(message)


def encode(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def read(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda text:
        (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+text)))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


class Scan:
    def __init__(self,root):self.root=root;self.cache={}
    def path(self,value):
        path=Path(value);path=path if path.is_absolute() else self.root/path
        need(not path.is_symlink(),'심볼릭 링크는 별도 실파일 경로로 명시해야 함: '+str(path))
        path=path.resolve();need(path.is_relative_to(self.root),'프로젝트 밖 입력 금지: '+str(path));return path
    def file(self,value):
        path=self.path(value);need(path.is_file(),'필수 파일 누락: '+str(path))
        stat=path.stat();need(stat.st_nlink==1,'하드 링크 회계는 허용하지 않음: '+str(path))
        key=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        prior=self.cache.get(str(path))
        if prior is None or prior[0]!=key:
            sha=digest(path);after=path.stat()
            need(key==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),
                 '검사 중 파일 변경: '+str(path))
            self.cache[str(path)]=(key,{'path':str(path),'project_relative_path':str(path.relative_to(self.root)),
                'size_bytes':stat.st_size,'allocated_bytes_observed':getattr(stat,'st_blocks',0)*512,'sha256':sha})
        return dict(self.cache[str(path)][1])
    def files(self,value,*,required=False):
        path=self.path(value)
        if not path.exists():
            need(not required,'명시한 --extra 누락: '+str(path));return []
        if path.is_file():return [path]
        need(path.is_dir(),'정규 파일/디렉터리만 허용: '+str(path))
        output=[]
        for p in sorted(path.rglob('*')):
            need(not p.is_symlink(),'디렉터리 안 심볼릭 링크 금지: '+str(p))
            if p.is_file():output.append(self.path(p))
            else:need(p.is_dir(),'비정규 파일은 누락 없이 별도 검토해야 함: '+str(p))
        return output
    def bindings(self,values):
        need(type(values) is dict,'원본 SHA binding 사전 필요')
        changed=[];missing=[];verified=0
        for path,expected in values.items():
            p=self.path(path)
            if not p.is_file():missing.append(str(p));continue
            record=self.file(p)
            if type(expected) is not str or record['sha256']!=expected:changed.append(str(p))
            else:verified+=1
        return {'expected_file_count':len(values),'unchanged_file_count':verified,
                'changed_paths':sorted(changed),'missing_paths':sorted(missing),
                'all_unchanged':not changed and not missing}


def build(run,out,extras,old_pdf,old_pdf_sha256,*,free_bytes=None):
    run=Path(run).resolve();root=run.parent.parent;scan=Scan(root);out=scan.path(out)
    match=re.fullmatch(r'p5-final-(\d{8})-\d+',run.name)
    need(run.parent.name=='runs' and match,'관리 대상 final run 경로 필요')
    need(out.suffix=='.json' and not out.exists(),'이전 원장은 보존하고 새 JSON 출력 필요')
    need(out.relative_to(root).parts[0] in {'reports','tmp','output'},'원장 출력은 reports/tmp/output 안 새 파일')
    manifest_path=run/'manifest.json';manifest=read(manifest_path);freeze=read(run/'freeze_receipt.json')
    manifest_file=scan.file(manifest_path);freeze_file=scan.file(run/'freeze_receipt.json')
    need(manifest.get('schema')=='p5-frozen-final-manifest-v1'
         and manifest.get('status')=='FROZEN_BEFORE_FINAL_OUTCOMES','원래 동결 manifest 필요')
    need(freeze.get('sha256')==manifest_file['sha256'],'동결 manifest SHA 불일치')
    original=manifest.get('source_sha256');need(type(original) is dict and len(original)==SOURCE_COUNT,'원본 source524 필요')
    original_check=scan.bindings(original)
    original_by_abs={str(scan.path(p)):h for p,h in original.items()}
    preservation=root/'reports/p5_recovery_parent_preservation.v1.json'
    need(original_by_abs.get(str(preservation))==scan.file(preservation)['sha256'],
         '부모 보존 receipt 자체가 원래 frozen source에 묶여 있지 않음')
    parent_before=read(preservation);parent=scan.path(manifest['infrastructure_recovery']['parent_run'])
    parent_map=parent_before.get('files_sha256')
    need(parent_before.get('file_count')==PARENT_COUNT and type(parent_map) is dict and len(parent_map)==PARENT_COUNT,
         '부모 보존275 기준 필요')
    need(all(scan.path(p).is_relative_to(parent) for p in parent_map),'부모 파일 binding 범위 불일치')
    parent_check=scan.bindings(parent_map)
    observed_scopes={}
    def listed(value,*,required=False,python_only=False):
        scope=scan.path(value);paths=scan.files(scope,required=required)
        paths=[p for p in paths if p!=out and (not python_only or p.suffix=='.py')]
        observed_scopes[(str(scope),python_only)]=(required,tuple(str(p) for p in paths))
        return paths
    parent_actual={str(p) for p in listed(parent,required=True)}
    parent_expected={str(scan.path(p)) for p in parent_map}
    parent_check['unbound_added_paths']=sorted(parent_actual-parent_expected)
    parent_check['inventory_exact']=parent_actual==parent_expected
    parent_check['all_unchanged']=parent_check['all_unchanged'] and parent_check['inventory_exact']
    need(type(old_pdf_sha256) is str and re.fullmatch('[0-9a-f]{64}',old_pdf_sha256),
         '이전 기록에서 가져온 원본 PDF SHA256을 명시해야 함')
    need(scan.path(old_pdf).suffix.lower()=='.pdf','구PDF 실파일 경로 필요')
    old_pdf_file=scan.file(old_pdf)
    old_pdf_check={**old_pdf_file,'expected_sha256':old_pdf_sha256,
                   'all_unchanged':old_pdf_file['sha256']==old_pdf_sha256}
    selected={};history=[];extra_registration=[]
    def include(path,pool,reason):
        path=scan.path(path)
        if path==out:return
        record=scan.file(path);key=str(path)
        if key not in selected:selected[key]={**record,'pools':[],'reasons':[]}
        if pool not in selected[key]['pools']:selected[key]['pools'].append(pool)
        if reason not in selected[key]['reasons']:selected[key]['reasons'].append(reason)
    for path in listed(root/'reports'):
        if path==out:continue
        record=scan.file(path)
        if original_by_abs.get(str(path))==record['sha256']:history.append(record)
        else:include(path,'future_report_archive_QA','reports_not_exact_frozen_history')
    for folder,reason in ((root/'output'/run.name,'final_output_tree'),(run/'visuals-v2','final_visuals_tree')):
        for path in listed(folder):include(path,'future_report_archive_QA',reason)
    for raw in extras:
        path=scan.path(raw)
        if path==out:
            extra_registration.append({'input_as_supplied':str(raw),'resolved_path':str(path),
                'recursive':False,'expanded_paths':[],'excluded_reason':'현재 원장은 ledger_self actual bytes로만 포함'})
            continue
        paths=listed(path,required=True)
        need(not path.is_relative_to(run) or path.is_relative_to(run/'visuals-v2'),
             '--extra로 원시/로그/모델 run 영역을 보고 pool에 혼합할 수 없음: '+str(path))
        for p in paths:
            if p.is_relative_to(root/'reports') and original_by_abs.get(str(p))==scan.file(p)['sha256']:
                continue  # frozen 역사 보고서는 --extra 디렉터리로 다시 포함하지 않음.
            include(p,'future_report_archive_QA','explicit_extra')
        extra_registration.append({'input_as_supplied':str(raw),'resolved_path':str(path),
                                   'recursive':path.is_dir(),'expanded_paths':[str(p) for p in paths if p!=out]})
    scratch=root/'tmp'/('p5-final-report-'+match[1])
    for path in listed(scratch):include(path,'working_disk_scratch','final_report_working_tree')
    for name in SOURCE_ROOTS:
        for path in listed(root/name,python_only=True):
            if original_by_abs.get(str(path))!=scan.file(path)['sha256']:
                include(path,'working_disk_scratch','python_source_not_exact_frozen_history')
    for (scope,python_only),(required,before) in observed_scopes.items():
        after=tuple(str(p) for p in scan.files(scope,required=required)
                    if p!=out and (not python_only or p.suffix=='.py'))
        need(before==after,'집계 중 파일 목록 변경: '+scope)
    # Re-stat every observed input: no stale size/hash cache can silently pass a
    # concurrent writer. This is a checkpoint, not an OS disk reservation.
    for path,(_,before) in list(scan.cache.items()):
        need(scan.file(path)==before,'집계 중 입력 크기/SHA 변경: '+path)
    entries=sorted(selected.values(),key=lambda row:row['path'])
    for row in entries:row['pools'].sort();row['reasons'].sort()
    report_bytes=sum(r['size_bytes'] for r in entries if 'future_report_archive_QA' in r['pools'])
    scratch_bytes=sum(r['size_bytes'] for r in entries if 'working_disk_scratch' in r['pools'])
    free=shutil.disk_usage(root).free if free_bytes is None else free_bytes
    need(type(free) is int and free>=0,'실측 free bytes는 음이 아닌 정수')
    integrity=original_check['all_unchanged'] and parent_check['all_unchanged'] and old_pdf_check['all_unchanged']
    value={'schema':'p5-final-report-resource-ledger-v1','status':'HOLD','accepted':False,
      'checked_at_utc':datetime.now(timezone.utc).isoformat(),'run':str(run),
      'generator':{'path':str(Path(__file__).resolve()),'sha256':digest(__file__),
                   'scope':'독립 파일 회계만 수행; 모델/물리/통계 재실행 없음'},
      'frozen_manifest':manifest_file,'freeze_receipt':freeze_file,'files':entries,'excluded_exact_frozen_reports':history,
      'explicit_extra_inputs':extra_registration,
      'coverage_inventory_roots':[{'path':scope,'python_only':py,'file_count':len(paths),
          'path_list_sha256':hashlib.sha256(encode(list(paths))).hexdigest()}
          for (scope,py),(_,paths) in observed_scopes.items()],
      'integrity':{'source524':original_check,'parent275':parent_check,'old_pdf':old_pdf_check,
                   'parent_preservation_receipt':scan.file(preservation)},
      'policy':{'byte_unit':'파일 logical size; allocated_bytes는 참고치이며 예산 합산에 사용하지 않음',
        'dedup':'각 pool 안 resolved absolute path 합집합; 심볼릭/하드링크 거부',
        'cross_pool_overlap':'명시적 extra가 scratch와 겹치면 양쪽 한도에 보수적으로 각각 포함',
        'extra_scope':'정의된 자동 root 이외 보고/PDF/QA 결과는 호출자가 --extra로 빠짐없이 명시해야 함; 이 목록을 독립 검토할 것',
        'raw_log_model_runs':'run/visuals-v2 외 run 원시·로그·모델은 기존 fixed/cell 원장 영역; 이 도구가 재분류하거나 삭제하지 않음',
        'historical_rule':'reports 아래 파일이 frozen manifest source SHA와 같을 때만 역사보존 제외',
        'disk_check_scope':'출력 직전 free checkpoint; 외부 동시 writer에 대한 원자적 예약 아님'},
      'claim_limit':'PASS는 현재 선언한 파일 범위의 회계와 원본 무변경 검사이며 프로젝트 완성 판정이 아니다. extra 목록과 최종 산출물 목록은 독립 대조해야 한다.',
      'ledger_self':{'path':str(out),'size_bytes':0,'sha256':None,
        'hash_unavailable_reason':'자기참조 SHA는 계산 불가. 외부 후속 receipt에서 해시할 것. files 목록에서는 제외하되 JSON 실측 크기는 report actual에 포함.'},
      'totals':{},'new_model_calls':0,'new_physics_steps':0,'source_or_original_edits':0}
    for _ in range(32):
        size=value['ledger_self']['size_bytes'];actual=report_bytes+size
        report_ok=actual+TAIL<=REPORT_LIMIT;scratch_ok=scratch_bytes<=SCRATCH_LIMIT
        disk_ok=free-size>=HARD_RESERVE
        value['totals']={'future_report_archive_QA':{'limit_bytes':REPORT_LIMIT,
            'file_bytes_excluding_this_ledger':report_bytes,'ledger_actual_bytes':size,'actual_bytes':actual,
            'reserved_tail_bytes':TAIL,'actual_plus_reserved_tail_bytes':actual+TAIL,
            'remaining_after_reserved_tail_bytes':REPORT_LIMIT-actual-TAIL,
            'actual_fits':actual<=REPORT_LIMIT,'fits_with_reserved_tail':report_ok},
          'working_disk_scratch':{'limit_bytes':SCRATCH_LIMIT,'actual_bytes':scratch_bytes,
            'remaining_bytes':SCRATCH_LIMIT-scratch_bytes,'fits':scratch_ok},
          'unique_included_file_bytes':sum(r['size_bytes'] for r in entries),
          'cross_pool_overlap_paths':[r['path'] for r in entries if len(r['pools'])>1],
          'disk':{'free_bytes_before_ledger':free,'hard_reserve_bytes':HARD_RESERVE,
             'predicted_free_after_ledger_bytes':free-size,'fits_after_ledger':disk_ok}}
        value['accepted']=bool(integrity and report_ok and scratch_ok and disk_ok)
        value['status']='PASS' if value['accepted'] else 'HOLD'
        payload=encode(value)
        if len(payload)==size:return value,payload
        value['ledger_self']['size_bytes']=len(payload)
    raise RuntimeError('원장 JSON 크기 fixed-point가 수렴하지 않음')


def write_ledger(run,out,extras,old_pdf,old_pdf_sha256):
    value,payload=build(run,out,extras,old_pdf,old_pdf_sha256)
    need(len(payload)<=TAIL,'원장 자체가 후속 metadata1MiB 예약을 초과')
    need(shutil.disk_usage(Path(run).resolve()).free>=HARD_RESERVE+len(payload),
         '실제 출력 직전2GiB reserve 부족; 파일을 쓰지 않고 중단')
    target=Path(out);need(not target.is_symlink() and not target.exists(),'원장 덮어쓰기 금지')
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
    need(target.stat().st_size==value['ledger_self']['size_bytes'],'원장 실측 크기 불일치')
    return value


def fixture(root):
    run=root/'runs/p5-final-20261017-02';parent=root/'runs/p5-final-20261017-01'
    def put(path,data=b'x'):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);return path
    originals={}
    for i in range(PARENT_COUNT):
        p=put(parent/f'original-{i:03d}.bin',str(i).encode());originals[str(p.relative_to(root))]=digest(p)
    preservation=put(root/'reports/p5_recovery_parent_preservation.v1.json',encode({'file_count':PARENT_COUNT,'files_sha256':originals}))
    history=put(root/'reports/historical.md',b'original report')
    source={str(p.relative_to(root)):digest(p) for p in (preservation,history)}
    for i in range(SOURCE_COUNT-2):
        p=put(root/f'baseline/source-{i:03d}.txt');source[str(p.relative_to(root))]=digest(p)
    m={'schema':'p5-frozen-final-manifest-v1','status':'FROZEN_BEFORE_FINAL_OUTCOMES','source_sha256':source,
       'infrastructure_recovery':{'parent_run':str(parent.relative_to(root))}}
    put(run/'manifest.json',encode(m));put(run/'freeze_receipt.json',encode({'sha256':digest(run/'manifest.json')}))
    put(root/'reports/new.md',b'new');put(root/'output'/run.name/'figure.png',b'figure')
    put(run/'visuals-v2/clip.gif',b'clip');put(root/'tmp/p5-final-report-20261017/draft.md',b'draft')
    put(root/'tools/new_tool.py',b'# new source');extra=put(root/'output/pdf/new.pdf',b'new PDF')
    old=put(root/'output/pdf/old.pdf',b'old PDF');put(run/'E2/raw.bin',b'not a reporting artifact')
    return run,root/'reports/ledger.json',extra,old,digest(old)


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.args=fixture(self.root)
    def build(self,extras=None,**kw):
        run,out,extra,old,sha=self.args
        return build(run,out,[extra] if extras is None else extras,old,sha,
                     free_bytes=kw.get('free_bytes',8*HARD_RESERVE))
    def test_dedup_history_ownership_and_fixedpoint(self):
        run,out,extra,old,sha=self.args
        value,payload=self.build(extras=[extra,extra.parent/extra.name,self.root/'output'/run.name,self.root/'reports',out])
        self.assertTrue(value['accepted']);self.assertEqual(value['ledger_self']['size_bytes'],len(payload))
        self.assertEqual(len(value['files']),len({r['path'] for r in value['files']}))
        self.assertEqual(value['totals']['future_report_archive_QA']['actual_bytes'],
            sum(r['size_bytes'] for r in value['files'] if 'future_report_archive_QA' in r['pools'])+len(payload))
        self.assertFalse(any('/E2/raw.bin' in r['path'] for r in value['files']))
        self.assertEqual(len(value['excluded_exact_frozen_reports']),2)
        self.assertEqual(value['integrity']['source524']['unchanged_file_count'],524)
        self.assertEqual(value['integrity']['parent275']['unchanged_file_count'],275)
        out.write_bytes(payload);self.assertEqual(out.stat().st_size,value['ledger_self']['size_bytes'])
    def test_changed_frozen_report_is_counted_and_original_change_blocks_pass(self):
        (self.root/'reports/historical.md').write_text('changed original')
        value,_=self.build();self.assertFalse(value['accepted'])
        self.assertTrue(value['integrity']['source524']['changed_paths'])
        self.assertTrue(any(r['path'].endswith('historical.md') for r in value['files']))
    def test_tail_scratch_and_free_reserve_are_actual_stop_conditions(self):
        value,_=self.build(free_bytes=HARD_RESERVE);self.assertFalse(value['totals']['disk']['fits_after_ledger'])
        large=self.root/'reports/large.bin'
        with large.open('wb') as stream:stream.truncate(REPORT_LIMIT-TAIL)
        value,_=self.build();self.assertFalse(value['totals']['future_report_archive_QA']['fits_with_reserved_tail'])
        large.unlink()
        with (self.root/'tmp/p5-final-report-20261017/large.bin').open('wb') as stream:stream.truncate(SCRATCH_LIMIT+1)
        value,_=self.build();self.assertFalse(value['totals']['working_disk_scratch']['fits'])
    def test_parent_and_old_pdf_changes_are_not_hidden(self):
        (self.root/'runs/p5-final-20261017-01/extra.bin').write_bytes(b'extra')
        self.args[3].write_bytes(b'changed old PDF')
        value,_=self.build();self.assertFalse(value['accepted'])
        self.assertFalse(value['integrity']['parent275']['inventory_exact'])
        self.assertFalse(value['integrity']['old_pdf']['all_unchanged'])
    def test_explicit_extra_missing_and_symlink_are_rejected(self):
        with self.assertRaises(ValueError):self.build(extras=[self.root/'missing.pdf'])
        link=self.root/'reports/alias.md';link.symlink_to(self.root/'reports/new.md')
        with self.assertRaises(ValueError):self.build()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--extra',type=Path,action='append',default=[])
    p.add_argument('--old-pdf',type=Path);p.add_argument('--old-pdf-sha256')
    p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:
        r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FixtureTests))
        return 0 if r.wasSuccessful() else 2
    need(all(v is not None for v in (a.run,a.out,a.old_pdf,a.old_pdf_sha256)),
         '--run --out --old-pdf --old-pdf-sha256 명시 필요; 기타 모든 보고/QA 산출물은 --extra로 등록')
    value=write_ledger(a.run,a.out,a.extra,a.old_pdf,a.old_pdf_sha256)
    print(json.dumps({'status':value['status'],'accepted':value['accepted'],'out':str(a.out),
                     'totals':value['totals']},ensure_ascii=False))
    return 0 if value['accepted'] else 2


if __name__=='__main__':raise SystemExit(main())
