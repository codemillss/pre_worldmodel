"""Sequential P5 orchestration with an immutable study manifest and live budget.

Importing this file never generates final initial conditions. Every physical
cell runs in a new child interpreter; completed immutable outputs are adopted
on resume, never silently rerun or replaced. Task failures retain denominators.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def save(path,value,*,replace=False):
    from push_align.data import _encode_nonfinite
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    target=path.with_name(path.name+'.next') if replace else path
    with target.open('x',encoding='utf-8') as stream:
        json.dump(_encode_nonfinite(value),stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    if replace:os.replace(target,path)
def append(path,value):
    with Path(path).open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
def tree_bytes(folder):return sum(p.stat().st_size for p in Path(folder).rglob('*') if p.is_file())
def verify(bindings):
    bad=[p for p,h in bindings.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    if bad:raise RuntimeError('Frozen source/evidence changed: '+str(bad[:4]))
def frozen(run):
    receipt=read(run/'freeze_receipt.json')
    if sha(run/'manifest.json')!=receipt['sha256']:raise RuntimeError('Frozen manifest changed')
    result=read(run/'manifest.json');verify(result['source_sha256']);verify(result['runtime_source_sha256']);return result
def progress(run):
    state=read(run/'progress.json')
    seal=run/'progress_receipt.json'
    if not seal.is_file():raise RuntimeError('Missing durable progress integrity receipt; review required')
    if sha(run/'progress.json')!=read(seal)['sha256']:
        raise RuntimeError('Progress ledger changed or interrupted between durable commits; review required')
    return state
def save_progress(run,state):
    save(run/'progress.json',state,replace=(run/'progress.json').exists())
    save(run/'progress_receipt.json',{'sha256':sha(run/'progress.json')},replace=(run/'progress_receipt.json').exists())

def validated_progress(run,m):
    from push_align.final_manifest import CONTROLLERS,replay_union
    s=progress(run)
    for name,limit in (('operational_bytes',192*1024**2),('mismatch_used_bytes',256*1024**2)):
        value=s.get(name,0)
        if type(value) is not int or not 0<=value<=limit:raise RuntimeError('Invalid cumulative resource counter: '+name)
    expected_E1={row['snapshot_id']+f'/candidate-{candidate:02d}/repeat-{repeat}'
        for row in m['E1'] for candidate in range(16) for repeat in range(3)}
    allowed={'E2_done':{row['cell_id'] for row in m['E2']},'E1_done':expected_E1,
             'application_done':set(CONTROLLERS)}
    selection=s.get('replay_selection')
    if selection is not None:
        if set(s['E2_done'])!=allowed['E2_done']:raise RuntimeError('Replay selection precedes complete E2 population')
        cache_key=(str(run.resolve()),sha(run/'manifest.json'),sha(run/'progress.json'),sha(run/'replay_selection.json'))
        expected=_SELECTION_CACHE.get(cache_key)
        if expected is None:
            rows=[verified_result(run,'E2',cell['cell_id'],m,deep=False)['row'] for cell in m['E2']]
            expected=replay_union(rows)
            if selection!=expected or read(run/'replay_selection.json')!=expected:
                raise RuntimeError('Replay selection differs from verified fixed population rule')
            _SELECTION_CACHE.clear();_SELECTION_CACHE[cache_key]=expected
        allowed['replay_done']={row['cell_id'] for row in expected}
    else:allowed['replay_done']=set()
    for key,permitted in allowed.items():
        values=s.get(key)
        if type(values) is not list or any(type(v) is not str for v in values) or len(set(values))!=len(values) or not set(values)<=permitted:
            raise RuntimeError('Invalid, duplicate or unknown completed ledger IDs: '+key)
    proof_key=(str(run.resolve()),sha(run/'manifest.json'),sha(run/'progress.json'))
    if proof_key not in _COMPLETION_PROOF_CACHE:
        for phase,key in (('E2','E2_done'),('application','application_done'),('replay','replay_done')):
            for cell_id in s[key]:verified_result(run,phase,cell_id,m,deep=False)
    if s['E1_done'] and proof_key not in _COMPLETION_PROOF_CACHE:
        receipts=s.get('E1_receipts',{})
        if set(receipts)!=set(s['E1_done']):raise RuntimeError('E1 completed IDs lack exact external repeat receipts')
        cache_key=(str(run.resolve()),sha(run/'progress.json'))
        if cache_key not in _E1_RECEIPT_CACHE:
            for repeat_id,binding in receipts.items():
                path=run/'E1/receipts/repeats'/(hashlib.sha256(repeat_id.encode()).hexdigest()+'.json')
                if str(path)!=binding['path'] or sha(path)!=binding['sha256'] or read(path)['repeat_id']!=repeat_id:
                    raise RuntimeError('E1 repeat integrity receipt changed')
                receipt=read(path)
                snapshot,suffix=repeat_id.rsplit('/candidate-',1)
                summary_path=run/'E1/snapshots'/snapshot/'physical'/('candidate-'+suffix)/'summary.json'
                if Path(receipt['summary_path'])!=summary_path or sha(summary_path)!=receipt['summary_sha256']:
                    raise RuntimeError('E1 completed repeat summary missing or changed')
                summary=read(summary_path);raw=summary['raw_receipt']
                if summary['status']!='COMPLETE_RECORDED' or summary['repeat_id']!=repeat_id or raw['files_sha256']!=receipt['raw_files_sha256']:
                    raise RuntimeError('E1 repeat summary/raw binding differs')
                for p in receipt['raw_files_sha256']:
                    if not Path(p).is_file() or Path(p).stat().st_size!=raw['bytes'][p]:
                        raise RuntimeError('E1 completed raw missing or size changed')
            _E1_RECEIPT_CACHE.clear();_E1_RECEIPT_CACHE.add(cache_key)
    _COMPLETION_PROOF_CACHE.clear();_COMPLETION_PROOF_CACHE.add(proof_key)
    return s


# Process-local cache lasts only between sealed cell progress commits. Full
# evidence is independently checked at every stage entry and cell adoption.
_SELECTION_CACHE={}
_E1_RECEIPT_CACHE=set()
_COMPLETION_PROOF_CACHE=set()
_FIXED_RECEIPTS={}
_DERIVED_ACTIVE_BYTES={}


def fixed_budget_contract(run,m):
    allowances=read(ROOT/m['resource_plan_path'])['fixed_allowances_bytes']
    scopes={
        'source_bundle':{'pool':'source_and_model_bundle','prefixes':['source'],'exact_paths':[]},
        'cell_operational':{'pool':'P5_boundary_analysis','prefixes':['E2','application','replays'],'exact_paths':[]},
        'E2_statistics':{'pool':'P5_boundary_analysis','prefixes':['statistics'],'exact_paths':[]},
        'E1_analysis':{'pool':'E1_prediction_analysis','prefixes':['E1'],'exact_paths':[]},
        'E1_statistics':{'pool':'E1_prediction_analysis','prefixes':['e1_statistics'],'exact_paths':[]},
        'fixed_prediction':{'pool':'E1_prediction_analysis','prefixes':['fixed_prediction'],'exact_paths':[]},
        'mismatch_extra':{'pool':'mismatch_prefix_and_diagnostics','prefixes':['replays'],'exact_paths':[]},
    }
    return allowances,{k:v for k,v in scopes.items() if v['pool'] in allowances}


def fixed_ledger_bytes(run):
    return sum(p.stat().st_size for p in run.glob('fixed_spend*.json') if p.is_file())


def fixed_write_guard(run,m,nbytes):
    if type(nbytes) is not int or nbytes<0:raise ValueError('Invalid ledger write size')
    free=shutil.disk_usage(ROOT).free
    if free<2*1024**3+nbytes:raise RuntimeError('Fixed-spend receipt write would violate hard reserve')
    allowance=read(ROOT/m['resource_plan_path'])['fixed_allowances_bytes']['P5_boundary_analysis']
    used=progress(run).get('operational_bytes',0)+tree_bytes(run/'statistics')+fixed_ledger_bytes(run)
    if used+nbytes>allowance:raise RuntimeError('Fixed-spend receipt metadata exceeds shared analysis allowance')
    return True


def fixed_spending(run,m,*,deep=False):
    """Full immutable-file check at stage/cell entry, cached inside one cell.

    A capacity subtraction never creates evidence. Every new expenditure is
    registered only by a semantic adopter below. Other stages/processes may not
    mutate these append-only receipts while this single worker is active.
    """
    from push_align.final_fixed_budget import read_fixed_spend
    key=(str(run.resolve()),sha(run/'manifest.json'))
    if deep or key not in _FIXED_RECEIPTS:
        allowances,scopes=fixed_budget_contract(run,m)
        _FIXED_RECEIPTS[key]=read_fixed_spend(run,manifest_sha256=key[1],allowances=allowances,scopes=scopes,deep=True)
    return _FIXED_RECEIPTS[key]


def register_fixed_files(run,m,category,files):
    """Idempotent union of newly verified spending; no task-completion credit."""
    from push_align.final_fixed_budget import add_fixed_spend
    receipt=fixed_spending(run,m,deep=True);new={}
    for p,h in files.items():
        old=receipt['files'].get(str(Path(p).resolve()))
        if old is not None:
            if old['category']!=category or old['sha256']!=h:
                raise RuntimeError('A fixed expenditure changed category/hash')
        else:new[str(Path(p).resolve())]=h
    if new:
        allowances,scopes=fixed_budget_contract(run,m)
        receipt=add_fixed_spend(run,manifest_sha256=sha(run/'manifest.json'),allowances=allowances,
            scopes=scopes,category=category,files_sha256=new,deep=False,
            before_write_bytes=lambda n:fixed_write_guard(run,m,n))
        _FIXED_RECEIPTS[(str(run.resolve()),sha(run/'manifest.json'))]=receipt
    return receipt


def register_source_bundle(run,m):
    folder=(run/'source').resolve()
    files={str((ROOT/p).resolve()):h for p,h in m.get('source_sha256',{}).items()
        if (ROOT/p).resolve().is_relative_to(folder)}
    if files:
        actual={str(p.resolve()) for p in folder.rglob('*') if p.is_file()}
        if set(files)!=actual:raise RuntimeError('Source bundle has missing/unbound files')
        register_fixed_files(run,m,'source_bundle',files)


def register_cell_spending(run,m,phase,result):
    record=read(result['record_manifest']);folder=Path(result['record_manifest']).parent
    measured=set(read(folder/'component_budget_review.json')['measured_component_paths'])
    operational={str(p):sha(p) for p in folder.rglob('*') if p.is_file() and str(p) not in measured}
    register_fixed_files(run,m,'cell_operational',operational)
    # Base counts already pay for raw/index/traces of originals, plus every
    # fresh witness and the three normal G captures. Only extra mismatch bytes
    # belong to the fixed diagnostic pool, never the whole fresh folder.
    extra={}
    if phase=='replay' and record.get('resources',{}).get('mismatch_charged'):
        G_base_ticks={0,29,299}
        if result['cell']['controller_id']=='G':
            from push_align.final_recording import read_json_gzip
            witness=read_json_gzip(record['witness_path'],expected_sha256=record['files_sha256'][record['witness_path']])
            terminal=witness['unexecuted_plan']
            if terminal is not None:G_base_ticks.add(terminal['index'])
        if record.get('raw'):extra.update(record['raw']['files_sha256'])
        if result['cell']['controller_id']!='G' and record.get('plan_index'):
            extra.update(record['plan_index']['files_sha256'])
        for trace in record['full_traces']:
            if result['cell']['controller_id']!='G' or trace['tick'] not in G_base_ticks:
                extra.update(trace['files_sha256'])
    if extra:register_fixed_files(run,m,'mismatch_extra',extra)


def register_complete_E1_spending(run,m):
    from p5_final_e1_adoption import adopt_complete_e1
    adopted=adopt_complete_e1(run,m)
    if adopted is None:raise RuntimeError('E1 complete spending requires fully adopted source')
    seal=read(run/'E1/stage_receipt.json');raw=set()
    for repeat_id in adopted['completed_repeat_ids']:
        snapshot,suffix=repeat_id.rsplit('/candidate-',1)
        row=read(run/'E1/snapshots'/snapshot/'physical'/('candidate-'+suffix)/'summary.json')
        raw.update(row['raw_receipt']['files_sha256'])
    files={p:h for p,h in seal['files_sha256'].items() if p not in raw}
    files[str(run/'E1/stage_receipt.json')]=sha(run/'E1/stage_receipt.json')
    register_fixed_files(run,m,'E1_analysis',files)
    return adopted

def commit_progress(run,event,kind=None,key=None,*,updates=None):
    state=progress(run)
    if kind is not None:
        if key in state[kind]:return state
        state[kind].append(key)
    if updates:state.update(updates)
    append(run/'progress_events.jsonl',{'at_utc':datetime.now(timezone.utc).isoformat(),'event':event,'kind':kind,'key':key})
    save_progress(run,state);return state


def remaining_estimate(run):
    from push_align.resource_budget import estimate_storage
    m=read(run/'manifest.json');plan=read(ROOT/m['resource_plan_path']);s=validated_progress(run,m)
    counts=dict(plan['counts'])
    def deduct_original(controller):
        counts['raw_records']-=1;counts['raw_intervals']-=300
        counts['witness_records']-=1;counts['witness_intervals']-=300;counts['witness_decisions']-=301
        if controller=='G':counts['G_captures']-=3
        else:counts['plan_archives']-=1;counts['plan_records']-=301;counts['full_traces']-=4
    for cell in m['E2']:
        if cell['cell_id'] in s['E2_done']:deduct_original(cell['controller_id'])
    for controller in s['application_done']:deduct_original(controller)
    counts['raw_records']-=len(s['E1_done']);counts['raw_intervals']-=5*len(s['E1_done'])
    selection=s.get('replay_selection')
    if selection is not None:
        unused=96-len(selection)
        counts['witness_records']-=unused;counts['witness_intervals']-=300*unused;counts['witness_decisions']-=301*unused
        G_selected=sum(cell['controller_id']=='G' for cell in selection)
        counts['G_captures']-=3*(12-G_selected)
    for cell_id in s['replay_done']:
        counts['witness_records']-=1;counts['witness_intervals']-=300;counts['witness_decisions']-=301
        controller=next(row['controller_id'] for row in selection if row['cell_id']==cell_id)
        if controller=='G':counts['G_captures']-=3
    if any(value<0 for value in counts.values()):raise RuntimeError('Remaining ledger overcounts completed work')
    spending=fixed_spending(run,m)
    result=estimate_storage(counts,plan['units_bytes'],spending['remaining_allowances_bytes'])
    return {'counts':counts,'fixed_spent_bytes':spending['total_fixed_spent_bytes'],
            'fixed_spent_by_pool':spending['spent_bytes_by_pool'],**result}


def capacity_check(run,*,next_write_bytes=0,preserve=False,active_written_credit=0):
    estimate=remaining_estimate(run);free=shutil.disk_usage(ROOT).free;reserve=2*1024**3
    if type(active_written_credit) is not int or active_written_credit<0:raise ValueError('Nonnegative active reservation credit required')
    required=max(reserve,estimate['required_bytes_including_reserve']-active_written_credit)
    if (not preserve and free<required) or free<reserve+next_write_bytes:
        raise RuntimeError(f'Capacity STOP: free={free} remaining={required} next={next_write_bytes}; preserve prefix')
    return {'status':'CONTINUE_RESOURCE_CHECK_ONLY','free_bytes':free,'remaining_required_bytes':required}


def measured_component_review(record,plan):
    """Check actual file totals against the same empirical 1.5 allowances.

    A nonempty raw pair measures a total, not a separate fixed/interval rate.
    Growth stops after preserving the executed prefix, before another cell.
    Operational manifests/reviews/receipts are charged separately to the
    declared boundary-analysis allowance, never silently discarded.
    """
    u=plan['units_bytes'];n=record['execution']['counts']['physics_intervals_returned']
    d=record['execution']['counts']['controller_decisions'];rows={};measured_paths=set()
    def component(name,paths,base):
        actual=sum(Path(p).stat().st_size for p in paths)
        allowed=math.ceil(1.5*base)
        rows[name]={'actual_bytes':actual,'allowed_bytes':allowed,'within_allowance':actual<=allowed}
        measured_paths.update(str(p) for p in paths)
    if record.get('raw'):
        component('raw',record['raw']['files_sha256'],u['raw_fixed']+n*u['raw_interval'])
    component('witness',[record['witness_path']],u['witness_fixed']+n*u['witness_interval']+d*u['witness_decision'])
    if record.get('plan_index') and record['identity']['controller_id']!='G':
        component('plan_index',record['plan_index']['files_sha256'],u['plan_index_fixed']+d*u['plan_index_record'])
    traces=record.get('full_traces',[])
    per_trace=(u['snapshot_envelope'] if record['identity']['controller_id']=='G'
        else u['full_trace_npz']+u['full_trace_identity']+u['snapshot_envelope'])
    if traces:component('full_traces_and_inputs',[p for r in traces for p in r['files_sha256']],len(traces)*per_trace)
    mismatch=bool(record.get('resources',{}).get('mismatch_charged'))
    return {'status':'PASS_MEASURED_COMPONENT_ALLOWANCES' if all(r['within_allowance'] for r in rows.values()) else 'STOP_REESTIMATE_PRESERVED_PREFIX',
            'components':rows,'mismatch_charged':mismatch,'measured_component_paths':sorted(measured_paths),
            'note':'Empirical component allowances, not a worst-case size proof; all newly observed totals retained.'}


def runtime(manifest=None):
    import torch
    from push_align.learning_data import load_learning_data
    from p4_compare_candidates import load_models
    from push_align.final_manifest import frozen_source_contract
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    contract=frozen_source_contract(ROOT,source_archives=(manifest or {}).get('source_archives'))
    metadata=read(ROOT/'runs/p3-02-20261009-01/learning_data/normalizer/normalizer.json')['normalizer']
    _,normalizer=load_learning_data(ROOT/'runs/p3-02-20261009-01/learning_data',metadata['schema'],metadata['dataset_identity'],metadata['normalizer_hash'])
    models,identities=load_models(normalizer,contract['physical_config'],read(ROOT/'configs/p4_development_e1.v1.json'))
    return contract,normalizer,models,identities


def expected_cell(m,phase,cell_id):
    from push_align.final_manifest import CONTROLLERS
    if phase=='application':
        if cell_id not in CONTROLLERS:raise ValueError('Unknown application controller')
        family=m['application_family']
        cell={'family_id':family['family_id'],'episode_id':family.get('episode_id',family['family_id']),
              'episode_index':12,'stratum':family.get('stratum','application'),
              'controller_id':cell_id,'cell_id':family['family_id']+'/'+cell_id}
    elif phase in ('E2','replay'):
        matches=[row for row in m['E2'] if row['cell_id']==cell_id]
        if len(matches)!=1:raise ValueError('Unknown or duplicate E2/replay cell')
        cell=matches[0]
    else:raise ValueError('Unknown phase')
    return cell


def expected_recording_contract(run,m,cell):
    controller=cell['controller_id']
    identity={'study':'P5-final','family_id':cell['family_id'],'controller_id':controller,
        'model':None if controller=='G' else m['model_identities'][controller],
        'normalizer':None if controller=='G' else m['normalizer_identity'],
        'final_manifest_sha256':sha(run/'manifest.json')}
    bindings={str(ROOT/p):h for p,h in {**m.get('source_sha256',{}),**m.get('runtime_source_sha256',{})}.items()}
    bindings[str(run/'manifest.json')]=sha(run/'manifest.json')
    return identity,bindings


def verified_result(run,phase,cell_id,m,*,deep=True):
    """Adopt only the requested identity, sealed metrics and complete evidence.

    The light form pins already-reviewed row/recording bytes for per-action
    capacity arithmetic. Stage entry/adoption uses the deep file/source check;
    immutable receipts never authorize replacing missing/corrupt evidence.
    """
    cell=expected_cell(m,phase,cell_id)
    folder=run/({'E2':'E2','replay':'replays','application':'application'}[phase])/cell['family_id']/cell['controller_id']
    path=folder/'worker_result.json';result=read(path)
    if not (folder/'worker_result_receipt.json').is_file():raise RuntimeError('Missing worker result integrity receipt')
    if sha(path)!=read(folder/'worker_result_receipt.json')['sha256']:raise RuntimeError('Worker result changed')
    if result.get('phase')!=phase or result.get('cell')!=cell or any(result.get('row',{}).get(k)!=v for k,v in cell.items()):
        raise RuntimeError('Saved outcome is for another phase/cell')
    row=result['row']
    if result['status'] not in ('ACCEPT_COMPLETED_RECORDING','ACCEPT_COMPLETED_TASK_FAILURE') or row.get('execution_status')!='completed':
        raise RuntimeError('Cannot adopt incomplete infrastructure outcome')
    if not row.get('future_engine_calls') or any(type(v) is not int or v!=0 for v in row['future_engine_calls'].values()):
        raise RuntimeError('Oracle guard failed or missing')
    record_path=Path(result['record_manifest']).resolve()
    if record_path!=folder.resolve()/'manifest.json' or sha(record_path)!=result['record_manifest_sha256']:
        raise RuntimeError('Recording manifest identity/hash changed')
    record=read(record_path)
    expected_kind='fresh_controller_replay' if phase=='replay' else 'reference_controller_execution'
    if record.get('record_kind')!=expected_kind:raise RuntimeError('Recording kind differs from expected original/fresh phase')
    expected_identity,expected_bindings=expected_recording_contract(run,m,cell)
    if record['identity']!=expected_identity or record['source_bindings']!=expected_bindings:
        raise RuntimeError('Recording differs from current frozen study/model/source contract')
    if any(row.get(k)!=record['execution'].get(k) for k in ('execution_status','termination','counts','future_engine_calls')):
        raise RuntimeError('Outcome row differs from recorded execution')
    review_path=folder/'review.json'
    if sha(review_path)!=result['review_sha256']:raise RuntimeError('Independent review receipt changed')
    review=read(review_path)
    if review.get('status')!=result['status'] or review.get('accepted') is not True or review.get('manifest_sha256')!=result['record_manifest_sha256']:
        raise RuntimeError('Review resolution or current recording binding differs')
    if any(row.get(k)!=v for k,v in review['metrics'].items()):raise RuntimeError('Metrics differ from independently reviewed raw outcome')
    component_path=folder/'component_budget_review.json'
    if not component_path.is_file() or sha(component_path)!=result.get('component_review_sha256'):
        raise RuntimeError('Missing or changed component budget receipt')
    component=read(component_path)
    if component['status']!='PASS_MEASURED_COMPONENT_ALLOWANCES':raise RuntimeError('Component growth remains unresolved')
    for p in record['files_sha256']:
        if not Path(p).is_file():raise RuntimeError('Completed output is missing: '+p)
    if deep:
        for p,h in record['files_sha256'].items():
            if not Path(p).is_file() or sha(p)!=h:raise RuntimeError('Recorded evidence missing or changed: '+p)
        for p,h in record['source_bindings'].items():
            if not Path(p).is_file() or sha(p)!=h:raise RuntimeError('Recording source changed: '+p)
        from push_align.final_result_review import review_recording
        again=review_recording(record_path,physical_config=m['physical_config'],profile=m['profile'],
            expected_manifest_sha256=result['record_manifest_sha256'],expected_identity=expected_identity,
            expected_source_bindings=expected_bindings)
        if again['status']!=review['status'] or again['metrics']!=review['metrics']:
            raise RuntimeError('Independent saved-raw outcome regeneration differs')
        if measured_component_review(record,read(ROOT/m['resource_plan_path']))!=component:
            raise RuntimeError('Measured component budget differs from saved receipt')
    return result


def _metric_row(execution,cell,*,planning_wall_s,execution_wall_s):
    import numpy as np
    from push_align.task import handoff_metrics
    from push_align.coverage import _contact_events
    initial=execution.initial_observation
    last=initial if initial is not None and initial.observation_valid and np.isfinite(initial.state).all() else None
    origin='last_valid_observation'
    for i,step in enumerate(execution.results):
        geometry=execution.records[i]['geometry']['pass'] if i<len(execution.records) else False
        if step.observation.observation_valid and step.transition_valid and geometry and np.isfinite(step.observation.state).all():
            last=step.observation
    endpoint=execution.results[-1].observation if execution.results else initial
    available=last is endpoint and last is not None and last.observation_valid
    if available:origin='terminal_observation'
    measured=handoff_metrics(last.state,execution.task_spec) if last is not None else None
    contacts=_contact_events([step.diagnostics for step in execution.results],.001,1e-9)
    return {**cell,'execution_status':execution.execution_status,'termination':execution.termination,
        'success':execution.termination=='ready','elapsed_s':float(execution.final_task.elapsed_s),
        'final_position_error_m':None if measured is None else measured['e_pos_m'],
        'final_yaw_error_rad':None if measured is None else measured['e_yaw_rad'],
        'final_linear_speed_m_s':None if measured is None else measured['linear_speed_m_s'],
        'final_angular_speed_rad_s':None if measured is None else measured['angular_speed_rad_s'],
        'final_observation_available':bool(available),'metric_state_origin':origin if last is not None else 'unavailable',
        'metric_state_time_s':None if last is None else float(last.sim_time-execution.initial_task.initial_sim_time),
        'unavailable_reason':None if last is not None else 'no valid observation retained',
        'recontact_count':len(contacts['qualified_recontacts']),'contact_events':contacts,
        'planning_wall_s':planning_wall_s,'execution_wall_s':execution_wall_s,
        'planning_time_scope':'sum of unchanged G proposal and learned plan calls; recording/source checks excluded',
        'counts':execution.counts,'future_engine_calls':execution.future_engine_calls}


def record_call_timing(timings,name,elapsed):
    if not math.isfinite(elapsed) or elapsed<0:raise ValueError('Invalid elapsed wall time')
    if timings[name+'_calls']==0:timings[name+'_first_call_s']=elapsed
    else:timings[name+'_repeat_wall_s']+=elapsed;timings[name+'_repeat_calls']+=1
    timings[name]+=elapsed;timings[name+'_calls']+=1


def worker(run,phase,cell_id):
    worker_started=time.perf_counter()
    from unittest.mock import patch
    import numpy as np
    from push_align import final_execution as core,final_recording_columns as codec
    from push_align.environment import InitialSpec,PlanarPushEnv
    from push_align.final_driver import LiveResourceGuard,run_recorded_episode,read_full_trace,exact_comparison
    from push_align.planning import LearnedPredictor
    from push_align.planning_protocol import GeometryProposalAdapter
    m=frozen(run);fixed_spending(run,m,deep=True);register_source_bundle(run,m)
    capacity_check(run);contract,normalizer,models,_=runtime(m)
    for key in ('model_identities','normalizer_identity','physical_config','planner_config','profile'):
        if contract[key]!=m[key]:raise RuntimeError('Runtime differs from frozen source contract: '+key)
    if phase=='application':
        family=m['application_family'];controller=cell_id
        cell={'family_id':family['family_id'],'episode_id':family['episode_id'],'episode_index':12,
              'stratum':family['stratum'],'controller_id':controller,'cell_id':family['family_id']+'/'+controller}
    else:
        cell=next(row for row in m['E2'] if row['cell_id']==cell_id);controller=cell['controller_id']
        family=next(row for row in m['families'] if row['family_id']==cell['family_id'])
    folder=run/({'E2':'E2','replay':'replays','application':'application'}[phase])/cell['family_id']/controller
    folder.parent.mkdir(parents=True,exist_ok=True)
    if folder.exists():raise FileExistsError('Worker will not replace prior attempt: '+str(folder))
    identity,bindings=expected_recording_contract(run,m,cell)
    guard=LiveResourceGuard(ROOT,remaining_required_bytes=lambda:remaining_estimate(run)['required_bytes_including_reserve'],
        cell_write_allowance_bytes=64*1024**2,mismatch_used_bytes=progress(run).get('mismatch_used_bytes',0))
    original_before=guard.before_action
    def bounded_action():
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform!='darwin':rss*=1024
        if rss>6*1024**3:raise RuntimeError('Per-action RSS checkpoint reached; preserve current prefix')
        return original_before()
    guard.before_action=bounded_action
    predictor=None if controller=='G' else LearnedPredictor(models[controller].model,normalizer)
    env=PlanarPushEnv(m['physical_config']);observed=env.reset(InitialSpec(**family['initial_spec']));snapshot=env.snapshot()
    # G runs first; all methods must share its complete settled initial state.
    if phase in ('E2','application') and controller!='G':
        G_manifest=read(run/phase/cell['family_id']/'G'/'manifest.json')
        G_input,_,_=read_full_trace(next(r for r in G_manifest['full_traces'] if r['tick']==0))
        if exact_comparison(snapshot,G_input['snapshot'])['status']!='PASS_EXACT':raise RuntimeError('Common settled family snapshot differs')
    timings={'plan':0.,'geometry':0.,'plan_calls':0,'geometry_calls':0,
        'plan_first_call_s':None,'geometry_first_call_s':None,
        'plan_repeat_wall_s':0.,'geometry_repeat_wall_s':0.,'plan_repeat_calls':0,'geometry_repeat_calls':0,
        'initialization_wall_s':time.perf_counter()-worker_started,
        'initialization_scope':'worker entry through imports, source/capacity checks, model loading and physical reset/common snapshot check; process startup excluded',
        'warmup_performed':False,'dedicated_warmup_calls':0,
        'first_call_scope':'first actual task call; included in total, never relabeled as warmup'}
    original_plan,original_propose=core.plan,GeometryProposalAdapter.propose
    def timed_plan(*a,**k):
        t=time.perf_counter()
        try:return original_plan(*a,**k)
        finally:record_call_timing(timings,'plan',time.perf_counter()-t)
    def timed_propose(*a,**k):
        t=time.perf_counter()
        try:return original_propose(*a,**k)
        finally:record_call_timing(timings,'geometry',time.perf_counter()-t)
    started=time.perf_counter()
    reference=run/'E2'/cell['family_id']/controller/'manifest.json' if phase=='replay' else None
    with patch.object(core,'plan',timed_plan),patch.object(GeometryProposalAdapter,'propose',timed_propose):
        recorded=run_recorded_episode(env,snapshot,folder=folder,identity=identity,source_bindings=bindings,
            profile=m['profile'],guard=guard,reference_manifest=reference,raw_codec=codec,
            case_id=cell['family_id'],episode_index=cell['episode_index'],controller_id=controller,
            execution_id=run.name+'/'+phase+'/'+cell['cell_id'],predictor=predictor,
            physical_config=m['physical_config'],planner_config=m['planner_config'],
            geometry_parameters=read(ROOT/'configs/p3_geometry_selected.v2.json'),
            transition_parameters=read(ROOT/'configs/p3_geometry_transition.v2.json'),
            candidate_seed_base=m['candidate_generation']['base_seed'],candidate_stream_id=0,
            max_intervals=300,run_scope='final_evaluation',penetration_budget_m=m['profile']['physics_suite']['actual_contact_penetration_max_m'],
            origin={'family':family,'frozen_manifest_sha256':sha(run/'manifest.json')})
    wall=time.perf_counter()-started
    row=_metric_row(recorded.execution,cell,planning_wall_s=timings['plan']+timings['geometry'],execution_wall_s=wall)
    from push_align.final_result_review import review_recording
    review=review_recording(recorded.manifest_path,physical_config=m['physical_config'],profile=m['profile'],
        expected_manifest_sha256=sha(recorded.manifest_path),expected_identity=identity,
        expected_source_bindings=bindings,raw_execution=recorded.execution)
    save(folder/'review.json',review)
    if review['accepted']:row.update(review['metrics'])
    status=review['status']
    component_review=measured_component_review(recorded.manifest,read(ROOT/m['resource_plan_path']))
    save(folder/'component_budget_review.json',component_review)
    result={'status':status,'phase':phase,'cell':cell,'row':row,'timings':timings,
            'record_manifest':str(recorded.manifest_path),'record_manifest_sha256':sha(recorded.manifest_path) if recorded.manifest_path else None,
            'comparison':recorded.comparison,'review_sha256':sha(folder/'review.json'),
            'recording_status':recorded.status,'component_review_sha256':sha(folder/'component_budget_review.json'),
            'cell_bytes':tree_bytes(folder),
            'process_peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024),
            'P5_task_success':row['success'],'learning_updates':0}
    save(folder/'worker_result.json',result)
    save(folder/'worker_result_receipt.json',{'sha256':sha(folder/'worker_result.json')})
    if not review['accepted']:raise RuntimeError('Recorded task needs review before continuing: '+str(review['issues']))
    if component_review['status']!='PASS_MEASURED_COMPONENT_ALLOWANCES':raise RuntimeError('Component output grew beyond budget; prefix retained, reestimate required')
    if row['execution_status']!='completed' or any(row['future_engine_calls'].values()):raise RuntimeError('Incomplete outcome or oracle query')
    print(json.dumps({'phase':phase,'cell':cell['cell_id'],'termination':row['termination'],'success':row['success'],'bytes':result['cell_bytes']}),flush=True)


def execute_cells(run,phase):
    m=frozen(run);fixed_spending(run,m,deep=True);register_source_bundle(run,m);s=validated_progress(run,m)
    if phase=='E2':items=[row['cell_id'] for row in m['E2']];kind='E2_done'
    elif phase=='application':
        from push_align.final_manifest import CONTROLLERS
        items=list(CONTROLLERS);kind='application_done'
    else:
        if s.get('replay_selection') is None:
            from push_align.final_manifest import replay_union
            rows=[verified_result(run,'E2',cell['cell_id'],m)['row'] for cell in m['E2']]
            selection=replay_union(rows);s['replay_selection']=selection
            save(run/'replay_selection.json',selection);save_progress(run,s)
        items=[row['cell_id'] for row in progress(run)['replay_selection']];kind='replay_done'
    for cell_id in items:
        if cell_id in progress(run)[kind]:
            result=verified_result(run,phase,cell_id,m);register_cell_spending(run,m,phase,result);continue
        fixed_spending(run,m,deep=True)
        capacity_check(run,next_write_bytes=64*1024**2)
        if phase=='application':folder=run/'application'/m['application_family']['family_id']/cell_id
        else:folder=run/('E2' if phase=='E2' else 'replays')/cell_id
        if not (folder/'worker_result.json').exists():
            if folder.exists():raise RuntimeError('Incomplete attempt retained; needs explicit review: '+str(folder))
            folder.parent.mkdir(parents=True,exist_ok=True)
            log=folder.parent/(folder.name+'.worker.log')
            with log.open('x') as stream:
                completed=subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker','--run',str(run),
                    '--phase',phase,'--cell',cell_id],stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT)
            if completed.returncode:raise RuntimeError('Worker stopped; retained log '+str(log))
        result=verified_result(run,phase,cell_id,m)
        record=read(result['record_manifest']);review=read(folder/'component_budget_review.json')
        if sha(folder/'component_budget_review.json')!=result['component_review_sha256'] or review['status']!='PASS_MEASURED_COMPONENT_ALLOWANCES':
            raise RuntimeError('Unresolved component budget growth')
        state=progress(run)
        charged=sum(Path(p).stat().st_size for p in review['measured_component_paths'])
        operational=max(0,tree_bytes(folder)-charged)
        if cell_id not in state[kind]:
            state['operational_bytes']=state.get('operational_bytes',0)+operational
            if review['mismatch_charged']:state['mismatch_used_bytes']=state.get('mismatch_used_bytes',0)+tree_bytes(folder)
            if state['operational_bytes']+tree_bytes(run/'statistics')+fixed_ledger_bytes(run)>read(ROOT/m['resource_plan_path'])['fixed_allowances_bytes']['P5_boundary_analysis']:
                raise RuntimeError('Operational metadata/analysis allowance exhausted; preserve and reestimate')
            if state.get('mismatch_used_bytes',0)>256*1024**2:raise RuntimeError('Cumulative mismatch allowance exhausted')
        register_cell_spending(run,m,phase,result)
        if state.get('operational_bytes',0)+tree_bytes(run/'statistics')+fixed_ledger_bytes(run)>read(ROOT/m['resource_plan_path'])['fixed_allowances_bytes']['P5_boundary_analysis']:
            raise RuntimeError('New receipt metadata exhausted shared analysis allowance; prefix retained')
        commit_progress(run,'completed_'+phase,kind,cell_id,updates={k:state[k] for k in ('operational_bytes','mismatch_used_bytes') if k in state})
        print(json.dumps({'phase':phase,'completed':len(progress(run)[kind]),'expected':len(items),'cell':cell_id,'termination':result['row']['termination']}),flush=True)


def statistics(run):
    import numpy as np
    from push_align.final_statistics import analyze_E2
    m=frozen(run);begin_derived_stage(run,'statistics')
    rows=[verified_result(run,'E2',cell['cell_id'],m)['row'] for cell in m['E2']]
    summary,arrays=analyze_E2(rows,m['E2'])
    save_derived_json(run,'statistics','summary.json',summary)
    save_derived_npz(run,'statistics','bootstrap.npz',arrays)
    # A second, independent load/generation is compared before declaring table reproduction.
    regenerated=[verified_result(run,'E2',cell['cell_id'],m)['row'] for cell in reversed(m['E2'])]
    again,other=analyze_E2(regenerated,m['E2'])
    if summary!=again or any(not np.array_equal(arrays[k],other[k],equal_nan=True) for k in arrays):
        raise RuntimeError('Raw-result table/bootstrap regeneration differs')
    save_derived_json(run,'statistics','reproduction.json',{'status':'PASS_EXACT_REGENERATION','source_rows':96,'families':12,
         'summary_sha256':sha(run/'statistics'/'summary.json'),'bootstrap_sha256':sha(run/'statistics'/'bootstrap.npz')})
    bounded_seal_derived_stage(run,'statistics',{str(run/'E2'/c['cell_id']/'worker_result.json'):sha(run/'E2'/c['cell_id']/'worker_result.json') for c in m['E2']})
    adopt_derived_stage(run,'statistics')


def derived_pool_bytes(run,name):
    """Charge analysis to the same fixed pools, apart from separately priced raw.

    E1 raw is recognized from its retained receipt, never from arbitrary suffixes.
    Files with no completed raw receipt remain charged to analysis conservatively.
    """
    if name=='statistics':
        return progress(run).get('operational_bytes',0)+tree_bytes(run/'statistics')+fixed_ledger_bytes(run)
    raw_paths=set()
    for path in (run/'E1/snapshots').rglob('summary.json'):
        receipt=read(path).get('raw_receipt')
        if receipt:raw_paths.update(receipt['files_sha256'])
    e1_bytes=sum(p.stat().st_size for p in (run/'E1').rglob('*') if p.is_file() and str(p) not in raw_paths)
    return e1_bytes+tree_bytes(run/'fixed_prediction')+tree_bytes(run/'e1_statistics')


def derived_limit(run,name):
    m=read(run/'manifest.json');plan=read(ROOT/m['resource_plan_path'])
    return plan['fixed_allowances_bytes']['P5_boundary_analysis' if name=='statistics' else 'E1_prediction_analysis']


def begin_derived_stage(run,name):
    if (run/name).exists():raise RuntimeError('Partial or complete derived stage retained; adopt or review before rerun: '+name)
    fixed_spending(run,read(run/'manifest.json'),deep=True)
    capacity_check(run,next_write_bytes=max(0,derived_limit(run,name)-derived_pool_bytes(run,name)))
    _DERIVED_ACTIVE_BYTES[(str(run.resolve()),name)]=0


def save_derived_bytes(run,name,filename,payload):
    if derived_pool_bytes(run,name)+len(payload)>derived_limit(run,name):
        raise RuntimeError('Analysis allowance exceeded before write; outputs retained, reestimate required')
    key=(str(run.resolve()),name)
    if key not in _DERIVED_ACTIVE_BYTES:raise RuntimeError('Derived write lacks a live stage reservation')
    capacity_check(run,next_write_bytes=len(payload),active_written_credit=_DERIVED_ACTIVE_BYTES[key])
    path=run/name/filename;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
    _DERIVED_ACTIVE_BYTES[key]+=len(payload)


def save_derived_json(run,name,filename,value):
    from push_align.data import _encode_nonfinite
    payload=(json.dumps(_encode_nonfinite(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    save_derived_bytes(run,name,filename,payload)


def save_derived_npz(run,name,filename,arrays):
    import numpy as np
    stream=io.BytesIO();np.savez_compressed(stream,**arrays)
    save_derived_bytes(run,name,filename,stream.getvalue())


def bounded_seal_derived_stage(run,name,sources):
    folder=run/name
    files={str(p):sha(p) for p in folder.rglob('*') if p.is_file()}
    save_derived_json(run,name,'stage_receipt.json',{'schema':'p5-derived-stage-seal-v1','stage':name,
        'frozen_manifest_sha256':sha(run/'manifest.json'),'files_sha256':files,'source_sha256':sources})
    save_derived_json(run,name,'stage_receipt.sha256.json',{'sha256':sha(folder/'stage_receipt.json')})


def seal_derived_stage(run,name,sources):
    folder=run/name
    files={str(p):sha(p) for p in folder.rglob('*') if p.is_file()}
    save(folder/'stage_receipt.json',{'schema':'p5-derived-stage-seal-v1','stage':name,
        'frozen_manifest_sha256':sha(run/'manifest.json'),'files_sha256':files,'source_sha256':sources})
    save(folder/'stage_receipt.sha256.json',{'sha256':sha(folder/'stage_receipt.json')})


def adopt_derived_stage(run,name):
    folder=run/name
    if not folder.exists():return False
    if not (folder/'stage_receipt.json').is_file() or not (folder/'stage_receipt.sha256.json').is_file():
        raise RuntimeError('Incomplete derived stage retained; no automatic rerun: '+name)
    required={'statistics':{'summary.json','bootstrap.npz','reproduction.json'},
              'e1_statistics':{'summary.json','bootstrap.npz','reproduction.json'},
              'fixed_prediction':{'summary.json','traces.npz'}}
    if name not in required or any(not (folder/f).is_file() for f in required[name]):
        raise RuntimeError('Missing required derived output/summary: '+name)
    receipt=read(folder/'stage_receipt.json')
    if sha(folder/'stage_receipt.json')!=read(folder/'stage_receipt.sha256.json')['sha256'] or receipt['frozen_manifest_sha256']!=sha(run/'manifest.json') or receipt['stage']!=name:
        raise RuntimeError('Derived stage identity/seal changed: '+name)
    for p,h in {**receipt['files_sha256'],**receipt['source_sha256']}.items():
        if not Path(p).is_file() or sha(p)!=h:raise RuntimeError('Derived stage/source changed: '+p)
    m=frozen(run)
    if name=='e1_statistics':
        from push_align.final_e1_statistics_review import validate_e1_statistics_stage
        validate_e1_statistics_stage(run,m)
    else:
        from push_align.final_stage_review import validate_derived_stage
        validate_derived_stage(run,name,m,
            load_e2_rows=lambda:[verified_result(run,'E2',cell['cell_id'],m)['row'] for cell in m['E2']])
    category={'statistics':'E2_statistics','e1_statistics':'E1_statistics','fixed_prediction':'fixed_prediction'}[name]
    files={**receipt['files_sha256'],str(folder/'stage_receipt.json'):sha(folder/'stage_receipt.json'),
        str(folder/'stage_receipt.sha256.json'):sha(folder/'stage_receipt.sha256.json')}
    register_fixed_files(run,m,category,files)
    _DERIVED_ACTIVE_BYTES.pop((str(run.resolve()),name),None)
    return True


def freeze(run):
    from push_align.final_manifest import (frozen_source_contract,prior_initial_inventory,
        family_records,expected_E2,expected_E1,freeze_manifest)
    from push_align.resource_budget import capacity_decision
    if run.exists():raise FileExistsError('A frozen or partial run is never silently replaced')
    if not run.is_relative_to(ROOT/'runs') or not run.name.startswith('p5-final-'):raise ValueError('Managed final run path required')
    native=read(ROOT/'reports/p5_driver_native_preflight_review.json')
    if native['status']!='PASS_BOUNDED_NATIVE_DRIVER_PREFLIGHT' or not all(native['checks'].values()):raise RuntimeError('Native integration gate unresolved')
    readiness=read(ROOT/'reports/p5_final_preparation_review.json')
    if readiness['status']!='PASS' or not all(readiness['checks'].values()):raise RuntimeError('Final orchestration preparation audit unresolved')
    if not readiness.get('source_sha256'):raise RuntimeError('Preparation audit lacks exact reviewed source bindings')
    verify(readiness['source_sha256'])
    archives=readiness.get('source_archives',{})
    contract=frozen_source_contract(ROOT,source_archives=archives);resource_plan=read(ROOT/'configs/p5_g2_resource_plan.v3.json')
    run.mkdir()
    if archives:
        remapped={}
        for original,prepared in archives.items():
            target=run/'source'/original;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/prepared,target)
            if sha(target)!=sha(ROOT/prepared):raise RuntimeError('Archived historical document copy differs')
            remapped[original]=str(target.relative_to(ROOT))
        archives=remapped;contract=frozen_source_contract(ROOT,source_archives=archives)
    sources=dict(contract['source_sha256'])
    runtime_paths=sorted({*list((ROOT/'src').rglob('*.py')),*list((ROOT/'scripts').glob('*.py')),
        *[ROOT/p for p in readiness['source_sha256']],
        ROOT/'configs/p5_g2_resource_plan.v3.json',ROOT/'reports/p5_final_preparation_review.json',
        ROOT/'reports/p5_driver_native_preflight_review.json'})
    runtime_sources={str(p.relative_to(ROOT)):sha(p) for p in runtime_paths}
    sources.update(runtime_sources)
    for rel,digest in list(sources.items()):
        if (ROOT/rel).is_relative_to(run/'source'):continue
        target=run/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,target)
        if sha(target)!=digest:raise RuntimeError('Source snapshot copy differs')
        sources[str(target.relative_to(ROOT))]=digest
    if tree_bytes(run/'source')>32*1024**2:raise RuntimeError('Source/model bundle allowance exceeded; preserve partial bundle')
    inventory=prior_initial_inventory(ROOT,exclude_final_run=run)
    if inventory['unresolved']:raise RuntimeError('Unresolved previous initial identities')
    # These copied bytes already reduced free. They own the source-bundle
    # allowance and must not be requested a second time before family drawing.
    copied={str(p):{'sha256':sha(p),'size_bytes':p.stat().st_size}
        for p in (run/'source').rglob('*') if p.is_file()}
    copied_bytes=sum(row['size_bytes'] for row in copied.values())
    expected_copies={str((ROOT/p).resolve()):h for p,h in sources.items()
        if (ROOT/p).resolve().is_relative_to((run/'source').resolve())}
    if {str(Path(p).resolve()):r['sha256'] for p,r in copied.items()}!=expected_copies:
        raise RuntimeError('Prefreeze copied-source spending has an unbound file')
    initial_required=resource_plan['estimate']['required_bytes_including_reserve']
    capacity=capacity_decision(shutil.disk_usage(ROOT).free,max(2*1024**3,initial_required-copied_bytes))
    save(run/'resource_before_freeze.json',{'capacity':capacity,'resource_plan_sha256':sha(ROOT/'configs/p5_g2_resource_plan.v3.json'),
         'checked_at_utc':datetime.now(timezone.utc).isoformat(),'originals_preserved':True,
         'static_initial_required_bytes':initial_required,'verified_copied_source_bytes':copied_bytes,
         'copied_source_files':copied,'credit_scope':'only current-run hash-verified copied source; reserve unchanged'})
    if capacity['status']!='FIT_ESTIMATE_ONLY':raise RuntimeError('Current full budget does not fit; incomplete source bundle and capacity receipt retained; final families not generated')
    families,application=family_records(contract['physical_config'],seed=6711,new_family_seed=6911,
        initial_id_prefix='p5-final',forbidden_identities=inventory['canonical_identities'])
    save(run/'integration_receipt.json',{'status':'PASS','checks':native['checks'],
        'source_report':'reports/p5_driver_native_preflight_review.json','source_report_sha256':sha(ROOT/'reports/p5_driver_native_preflight_review.json'),
        'preparation_review_sha256':sha(ROOT/'reports/p5_final_preparation_review.json')})
    for rel in ('resource_before_freeze.json','integration_receipt.json'):
        sources[str((run/rel).relative_to(ROOT))]=sha(run/rel)
    # All cell inputs bind the same source graph; the manifest's own SHA is
    # added externally by each worker, avoiding a circular self-checksum.
    runtime_sources.update(contract['source_sha256'])
    document={**contract,'source_archives':archives,'source_sha256':sources,'runtime_source_sha256':runtime_sources,
        'source_bundle_map':{**{rel:str((run/'source'/rel).relative_to(ROOT)) for rel in sources if (run/'source'/rel).is_file()},**archives},
        'family_generation':{'recipe_version':'p5-family-strata-v1','seed':6711,'application_seed':6911,'initial_id_prefix':'p5-final'},
        'candidate_generation':{'base_seed':6811,'E2_stream_id':0,'E1_stream_id':1,
            'rule':'PCG64 SeedSequence([base_seed,episode_index,actual_control_tick,stream_id]); no model/learning seed key'},
        'families':families,'application_family':application,'E2':expected_E2(families),'E1':expected_E1(families),
        'prior_initial_inventory':inventory,'statistics':{'replicates':2000,'seed':7711,'unit':'family_within_stratum_seed_mean_first'},
        'prediction_evaluation':{'source':'all_E2_G_raw_fixed_applied_actions','horizons':[1,5,10],
            'validity':'recorded transition+observation+actual contact geometry; model invalid kept separately','regime':'first interval measured contact pattern'},
        'failure_policy':{'task_failures':'retain every original denominator and actual prefix','infrastructure':'no unresolved or missing cases; stop before next cell',
                          'replacement':'none','automatic_retries':0,'mismatch':'retain fresh full raw/index/trace;256MiB then stop/reestimate'},
        'replay_selection_rule':'per controller/learning seed: minimum episode ID per stratum UNION minimum ID per observed termination type; deduplicate',
        'minimum_meaningful_difference':contract['profile']['final_review']['minimum_meaningful_improvement'],
        'visualization_selection':'minimum family ID per stratum, all8controllers; raw-pose replay with actual timestamps, no outcome-based choice',
        'visualization':{'selection_rule':'smallest_family_id_per_stratum_all_eight_controllers',
            'selected_families':{stratum:min(f['family_id'] for f in families if f['stratum']==stratum)
                for stratum in ('translation','rotation','offset')},
            'controllers':['G','R','S1-11','S1-23','S1-47','SK-11','SK-23','SK-47'],
            'fps':5,'postterminal_tail_s':1.0},
        'resource_plan_path':'configs/p5_g2_resource_plan.v3.json',
        'resource_receipt':{'path':str((run/'resource_before_freeze.json').relative_to(ROOT)),'sha256':sha(run/'resource_before_freeze.json')},
        'integration_receipt':{'path':str((run/'integration_receipt.json').relative_to(ROOT)),'sha256':sha(run/'integration_receipt.json')},
        'personal_understanding':'NOT_ASSESSED','actual_final_outcomes_used_for_selection':False}
    receipt=freeze_manifest(run/'manifest.json',document,integration_pass=True,resource_capacity=capacity,source_root=ROOT)
    save(run/'freeze_receipt.json',receipt)
    save_progress(run,{'E2_done':[],'E1_done':[],'application_done':[],'replay_done':[],'replay_selection':None})
    append(run/'progress_events.jsonl',{'at_utc':datetime.now(timezone.utc).isoformat(),'event':'FROZEN_BEFORE_FINAL_OUTCOMES','manifest_sha256':receipt['sha256']})
    print(json.dumps({'frozen':str(run),'manifest_sha256':receipt['sha256'],'families':12,'expected_E2':96,'expected_E1_snapshots':12,'capacity':capacity}),flush=True)


def E1(run):
    from p5_final_e1 import run_e1
    from p5_final_e1_adoption import (adopt_complete_e1,seal_prediction_bundle,
        seal_repeat_receipt,seal_complete_e1)
    m=frozen(run)
    fixed_spending(run,m,deep=True);register_source_bundle(run,m)
    adopted=adopt_complete_e1(run,m)
    if adopted is not None:
        state=validated_progress(run,m)
        if set(state['E1_done'])!=set(adopted['completed_repeat_ids']):raise RuntimeError('Complete E1 differs from completed resource ledger')
        register_complete_E1_spending(run,m)
        return adopted
    if (run/'E1').exists():raise RuntimeError('Partial E1 retained; no automatic replacement or repeat')
    active={'repeat_id':None,'written_credit':0}
    raw_units=read(ROOT/m['resource_plan_path'])['units_bytes']
    raw_write_allowance=math.ceil(1.5*(raw_units['raw_fixed']+5*raw_units['raw_interval']))
    def callback(event):
        kind=event['event'];preserve=event.get('must_preserve',False)
        if kind in ('predictions_frozen','before_repeat','snapshot_complete','stage_complete'):
            if derived_pool_bytes(run,'e1_statistics')>derived_limit(run,'e1_statistics'):
                raise RuntimeError('Shared E1/prediction analysis allowance exhausted; prefix retained, reestimate required')
        if kind=='before_repeat':
            if active['repeat_id'] is not None:raise RuntimeError('Previous E1 reservation remains open')
            active.update(repeat_id=event['repeat_id'],written_credit=0)
        if kind=='after_raw_write':active['written_credit']=event['bytes_written']
        if kind=='after_raw_write':
            units=read(ROOT/read(run/'manifest.json')['resource_plan_path'])['units_bytes']
            allowed=math.ceil(1.5*(units['raw_fixed']+event['actual_returned_intervals']*units['raw_interval']))
            if event['bytes_written']>allowed:raise RuntimeError('E1 raw growth exceeds empirical component allowance; saved prefix retained')
        if kind=='predictions_frozen':
            receipt=seal_prediction_bundle(run,event['choices_sha256'])
            register_fixed_files(run,m,'E1_analysis',{**receipt['files_sha256'],receipt['path']:receipt['sha256']})
        if kind=='repeat_complete':
            if active['repeat_id']!=event['repeat_id']:raise RuntimeError('Completed E1 differs from its active reservation')
            folder=run/'E1/snapshots'/event['repeat_id'].rsplit('/candidate-',1)[0]/'physical'/('candidate-'+event['repeat_id'].rsplit('/candidate-',1)[1])
            receipt=seal_repeat_receipt(run,event['repeat_id'],sha(folder/'summary.json'))
            register_fixed_files(run,m,'E1_analysis',{
                receipt['summary_path']:receipt['summary_sha256'],receipt['path']:receipt['sha256']})
            receipts=dict(progress(run).get('E1_receipts',{}));receipts[event['repeat_id']]={k:receipt[k] for k in ('path','sha256')}
            commit_progress(run,'completed_E1','E1_done',event['repeat_id'],updates={'E1_receipts':receipts})
            active.update(repeat_id=None,written_credit=0)
        if kind=='stage_complete':
            seal_complete_e1(run,event['summary_sha256'])
            register_complete_E1_spending(run,m)
        next_write=raw_write_allowance if kind=='before_repeat' else 0
        if kind in ('before_raw_write','preserve_raw_write'):
            n=event['actual_returned_intervals']
            if type(n) is not int or not 0<=n<=5:raise RuntimeError('Invalid E1 prefix reservation')
            next_write=math.ceil(1.5*(raw_units['raw_fixed']+n*raw_units['raw_interval']))
        capacity_check(run,next_write_bytes=next_write,
            preserve=preserve,active_written_credit=active['written_credit'])
        return {'status':'CONTINUE_RESOURCE_CHECK_ONLY'}
    result=run_e1(run,m,callback)
    if result['status']!='COMPLETE_RECORDED':raise RuntimeError('E1 incomplete: '+str(result))
    return adopt_complete_e1(run,m)


def prediction(run):
    import numpy as np
    from push_align.final_driver import read_full_trace
    from push_align.final_prediction import evaluate_final_predictions,TRUTH_MASK_PROVENANCE
    from push_align import final_recording_columns as codec
    from push_align.planning import LearnedPredictor
    m=frozen(run);begin_derived_stage(run,'fixed_prediction');_,normalizer,models,_=runtime(m)
    def trajectories():
        for family in m['families']:
            folder=run/'E2'/family['family_id']/'G';record=read(folder/'manifest.json')
            initial,_,_=read_full_trace(next(row for row in record['full_traces'] if row['tick']==0))
            _,steps=codec.read_raw(record['raw']['prefix'],expected_sha256=record['raw']['files_sha256'])
            from push_align.final_recording import read_json_gzip
            witness=read_json_gzip(record['witness_path'],expected_sha256=record['files_sha256'][record['witness_path']])
            intervals=witness['intervals']
            mask=np.array([bool(step.transition_valid and step.observation.observation_valid and intervals[i]['geometry']['pass']) for i,step in enumerate(steps)],bool)
            yield {'family_id':family['family_id'],'stratum':family['stratum'],'controller_id':'G',
                'initial_state10':initial['observation'].state.copy(),'initial_sim_time':initial['observation'].sim_time,
                'results':steps,'interval_valid':mask,
                'truth_mask_provenance':TRUTH_MASK_PROVENANCE}
            del initial,steps,witness,intervals,mask;gc.collect()
    summary,arrays=evaluate_final_predictions(trajectories(),{key:LearnedPredictor(value.model,normalizer) for key,value in models.items()},m['physical_config'],m['planner_config'])
    save_derived_json(run,'fixed_prediction','summary.json',summary)
    save_derived_npz(run,'fixed_prediction','traces.npz',arrays)
    sources={str(run/'E2'/family['family_id']/'G/manifest.json'):sha(run/'E2'/family['family_id']/'G/manifest.json') for family in m['families']}
    for family in m['families']:
        record=read(run/'E2'/family['family_id']/'G/manifest.json');sources.update(record['files_sha256'])
    bounded_seal_derived_stage(run,'fixed_prediction',sources)
    adopt_derived_stage(run,'fixed_prediction')


def E1_statistics(run):
    import numpy as np
    from p5_final_e1_adoption import adopt_complete_e1
    from push_align.final_e1_statistics import analyze_E1
    m=frozen(run);begin_derived_stage(run,'e1_statistics')
    if adopt_complete_e1(run,m) is None:raise RuntimeError('E1 statistics requires complete verified shared futures')
    ledger=run/'E1/comparison_ledger.json'
    summary,arrays=analyze_E1(read(ledger),m['E1'],replicates=2000,seed=7711)
    save_derived_json(run,'e1_statistics','summary.json',summary)
    save_derived_npz(run,'e1_statistics','bootstrap.npz',arrays)
    again,other=analyze_E1(list(reversed(read(ledger))),m['E1'],replicates=2000,seed=7711)
    if summary!=again or set(arrays)!=set(other) or any(not np.array_equal(arrays[k],other[k],equal_nan=True) for k in arrays):
        raise RuntimeError('E1 family statistics regeneration differs')
    save_derived_json(run,'e1_statistics','reproduction.json',{'status':'PASS_EXACT_REGENERATION','source_rows':84,'snapshots':12,'families':6,
        'summary_sha256':sha(run/'e1_statistics/summary.json'),'bootstrap_sha256':sha(run/'e1_statistics/bootstrap.npz')})
    sources={str(p):sha(p) for p in (run/'E1/summary.json',run/'E1/stage_receipt.json',ledger,run/'manifest.json')}
    bounded_seal_derived_stage(run,'e1_statistics',sources)
    adopt_derived_stage(run,'e1_statistics')


def all_stages(run):
    if not run.exists():freeze(run)
    execute_cells(run,'E2')
    E1(run)
    if not adopt_derived_stage(run,'e1_statistics'):E1_statistics(run)
    if not adopt_derived_stage(run,'statistics'):statistics(run)
    if not adopt_derived_stage(run,'fixed_prediction'):prediction(run)
    execute_cells(run,'replay');execute_cells(run,'application')
    save(run/'execution_complete.json',{'status':'ALL_DECLARED_EXECUTION_STAGES_RECORDED',
        'at_utc':datetime.now(timezone.utc).isoformat(),'remaining':remaining_estimate(run),
        'scope':'Final report/claim-to-evidence bundle, visual review and completion audit still required',
        'personal_understanding':'NOT_ASSESSED'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','worker','e2','e1','e1-statistics','prediction','application','replay','statistics','all'])
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--phase',choices=['E2','replay','application']);parser.add_argument('--cell')
    a=parser.parse_args();run=a.run.resolve()
    if a.command=='worker':worker(run,a.phase,a.cell)
    elif a.command=='freeze':freeze(run)
    elif a.command=='all':all_stages(run)
    elif a.command=='e1':E1(run)
    elif a.command=='e1-statistics':E1_statistics(run)
    elif a.command=='prediction':prediction(run)
    elif a.command=='statistics':statistics(run)
    else:execute_cells(run,{'e2':'E2'}.get(a.command,a.command))


if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();sys.exit(2)
