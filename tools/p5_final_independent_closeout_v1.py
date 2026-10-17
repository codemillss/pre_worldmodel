#!/usr/bin/env python3
"""Post-execution, read-only saved-evidence closeout (an unfrozen extra auditor).

No project modules are imported. No model, worker, engine or training is run.
Independent arithmetic checks the recorded denominators, conditional estimands
and bootstrap arrays; it does not independently recreate physical observations.
The sole write is an exclusive new --out JSON, including failure diagnostics.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
import zipfile

CONTROLLERS = ('G','R','S1-11','S1-23','S1-47','SK-11','SK-23','SK-47')
STRATA = ('translation','rotation','offset')
GROUPS = {'G':('G',),'R':('R',),'S1':('S1-11','S1-23','S1-47'),'SK':('SK-11','SK-23','SK-47')}
METRICS = ('success','elapsed_s','final_position_error_m','final_yaw_error_rad',
 'final_linear_speed_m_s','final_angular_speed_rad_s','recontact_count','planning_wall_s','execution_wall_s')
STATE_METRICS = METRICS[2:6]
INDICATORS = ('model_all_invalid','no_selection','selected_actual_invalid',
 'selected_actual_validity_changed','actual_all_invalid','no_stable_actual_valid','any_validity_changed','common_valid_empty')
E1_METRICS = ('selection_loss',*INDICATORS)
ENGINE_GUARDS = {'mj_step','mj_step1','mj_step2','mj_forward','mj_inverse'}
TERMINALS = {'ready','timeout','planner_invalid','controller_infeasible','physics_error',
 'geometry_invalid','workspace_exit','planarity_violation'}
MAX_EXPANDED_NPZ = 512*1024**2


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda text: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+text)))


def need(condition, message):
    if not condition:
        raise ValueError(message)


def same(a,b):
    return json.dumps(a,sort_keys=True,allow_nan=False,separators=(',',':')) == json.dumps(b,sort_keys=True,allow_nan=False,separators=(',',':'))


def close(a,b,label):
    if a is None or b is None:
        need(a is b,label+': null population differs')
    else:
        need(type(a) in (int,float,bool) and type(b) in (int,float,bool)
             and math.isfinite(a) and math.isfinite(b)
             and math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12),label+': numeric difference')


def mean(values):
    values=[float(v) for v in values if v is not None]
    if not values:return None
    scale=max(map(abs,values))
    return 0. if scale==0 else math.fsum(v/scale for v in values)/len(values)*scale


def quantiles(values,percentiles):
    values=sorted(float(v) for v in values if v is not None and math.isfinite(v))
    if not values:return None
    out=[]
    for p in percentiles:
        location=(len(values)-1)*p/100;lower=int(math.floor(location));upper=int(math.ceil(location))
        fraction=location-lower
        # Linear interpolation, scaled to avoid overflow for large finite data.
        scale=max(abs(values[lower]),abs(values[upper]))
        out.append(0. if scale==0 else ((1-fraction)*(values[lower]/scale)+fraction*(values[upper]/scale))*scale)
    return out


class Audit:
    def __init__(self,run):
        self.run=run;self.root=run.parent.parent;self.sections={};self.errors=[];self.hash_cache={}
        self.array_checks=[];self.files_hashed=0;self.bytes_hashed=0;self.manifest=None;self.e2_rows=[]

    def section(self,name,function):
        try:self.sections[name]={'status':'PASS_WITH_STATED_SCOPE','details':function()}
        except Exception as error:
            self.errors.append({'section':name,'type':type(error).__name__,'message':str(error),
                                'traceback':traceback.format_exc()})
            self.sections[name]={'status':'FAIL_RETAIN_DIAGNOSTICS','error':str(error)}

    def path(self,value,scope=None):
        p=Path(value);p=(p if p.is_absolute() else self.root/p).resolve()
        need(p.is_relative_to(self.root),'Evidence path escapes project: '+str(p))
        if scope is not None:need(p.is_relative_to(scope),'Evidence path escapes expected scope: '+str(p))
        need(p.is_file(),'Missing evidence: '+str(p));return p

    def sha(self,path):
        p=self.path(path);s=p.stat();key=(str(p),s.st_size,s.st_mtime_ns)
        if key not in self.hash_cache:
            h=hashlib.sha256()
            with p.open('rb') as stream:
                for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
            self.hash_cache[key]=h.hexdigest();self.files_hashed+=1;self.bytes_hashed+=s.st_size
        return self.hash_cache[key]

    def hashed(self,path,digest,scope=None):
        p=self.path(path,scope)
        need(type(digest) is str and len(digest)==64 and self.sha(p)==digest,'SHA256 mismatch: '+str(p))
        return p

    def bindings(self,values,scope=None):
        need(type(values) is dict and bool(values),'Nonempty file SHA bindings required')
        for p,h in values.items():self.hashed(p,h,scope)
        return len(values)

    def sealed_stage(self,name,required,expected_sources=None):
        folder=self.run/name;seal=folder/'stage_receipt.json'
        self.hashed(seal,read(folder/'stage_receipt.sha256.json')['sha256'],folder)
        receipt=read(seal)
        need(receipt.get('schema')=='p5-derived-stage-seal-v1' and receipt.get('stage')==name
             and receipt.get('frozen_manifest_sha256')==self.sha(self.run/'manifest.json'),'Derived stage identity differs: '+name)
        need(set(receipt['files_sha256'])=={str(folder/p) for p in required},'Derived file population differs: '+name)
        self.bindings(receipt['files_sha256'],folder);self.bindings(receipt['source_sha256'])
        if expected_sources is not None:need(receipt['source_sha256']==expected_sources,'Derived sources differ: '+name)
        return receipt

    def npz(self,path):
        import numpy as np
        with zipfile.ZipFile(path) as archive:
            rows=archive.infolist();names=[r.filename for r in rows]
            need(len(names)==len(set(names)) and all(n.endswith('.npy') and '/' not in n for n in names),'Malformed NPZ member population')
            need(sum(r.file_size for r in rows)<=MAX_EXPANDED_NPZ,'NPZ expanded bytes exceed independent audit limit')
        return np.load(path,allow_pickle=False)

    def array_equal(self,actual,expected,label,*,numeric=False):
        import numpy as np
        need(actual.shape==expected.shape and actual.dtype==expected.dtype,label+': dtype/shape differ')
        okay=np.allclose(actual,expected,rtol=1e-12,atol=1e-12,equal_nan=True) if numeric else np.array_equal(actual,expected,equal_nan=True)
        need(bool(okay),label+': array values differ')
        self.array_checks.append({'key':label,'shape':list(actual.shape),'dtype':str(actual.dtype),
            'sha256_array_bytes':hashlib.sha256(actual.tobytes()).hexdigest(),
            'comparison':'independent numeric arithmetic rtol=atol=1e-12' if numeric else 'exact values/dtype/shape'})

    def freeze_and_sources(self):
        receipt=read(self.run/'freeze_receipt.json');self.hashed(self.run/'manifest.json',receipt['sha256'])
        need(receipt.get('bytes')==(self.run/'manifest.json').stat().st_size,'Frozen manifest byte length differs')
        m=read(self.run/'manifest.json');self.manifest=m
        need(m.get('schema')=='p5-frozen-final-manifest-v1' and m.get('status')=='FROZEN_BEFORE_FINAL_OUTCOMES','Frozen manifest schema/status differs')
        need(len(m['source_sha256'])==524 and len(m['runtime_source_sha256'])==251,'Expected recovery source524/runtime251 population differs')
        self.bindings(m['source_sha256']);self.bindings(m['runtime_source_sha256'])
        need(all(m['source_sha256'].get(p)==h for p,h in m['runtime_source_sha256'].items()),'Runtime graph not source-bound')
        recovery=m['infrastructure_recovery'];parent=self.root/recovery['parent_run']
        self.hashed(parent/'manifest.json',recovery['parent_manifest_sha256'])
        self.hashed(parent/'freeze_receipt.json',recovery['parent_freeze_receipt_sha256'])
        need(recovery.get('parent_outcomes_observed') is True and m['recovery_metadata']['freeze_scope']=='before_new_run_outcomes_only','Restart observation scope omitted')
        config=read(self.root/'configs/p5_infrastructure_recovery.v1.json')
        preservation=self.hashed(config['parent_preservation_receipt'],config['parent_preservation_sha256'])
        need(m['source_sha256'].get(config['parent_preservation_receipt'])==config['parent_preservation_sha256'],'Parent preservation receipt not source-bound')
        before=read(preservation);need(before.get('file_count')==275 and len(before['files_sha256'])==275,'Expected parent275 preservation population differs')
        actual={str(p.relative_to(self.root)) for p in parent.rglob('*') if p.is_file()}
        need(actual==set(before['files_sha256']),'Parent file inventory added/removed files after preservation baseline')
        self.bindings(before['files_sha256'],parent)
        return {'manifest_sha256':receipt['sha256'],'source_files':524,'runtime_files':251,'parent_files':275,
            'parent_outcomes_already_observed':True,'same_initial_conditions_reused':True,
            'extra_auditor_not_in_frozen_runtime':str(Path(__file__).resolve().relative_to(self.root)) not in m['runtime_source_sha256']}

    def component(self,record,component):
        plan=read(self.root/self.manifest['resource_plan_path']);units=plan['units_bytes']
        n=record['execution']['counts']['physics_intervals_returned'];d=record['execution']['counts']['controller_decisions']
        expected={};paths=set()
        def add(name,values,base):
            values=list(values);actual=sum(self.path(p).stat().st_size for p in values);allowed=math.ceil(1.5*base)
            expected[name]={'actual_bytes':actual,'allowed_bytes':allowed,'within_allowance':actual<=allowed};paths.update(values)
        if record.get('raw'):add('raw',record['raw']['files_sha256'],units['raw_fixed']+n*units['raw_interval'])
        add('witness',[record['witness_path']],units['witness_fixed']+n*units['witness_interval']+d*units['witness_decision'])
        if record.get('plan_index') and record['identity']['controller_id']!='G':
            add('plan_index',record['plan_index']['files_sha256'],units['plan_index_fixed']+d*units['plan_index_record'])
        traces=record.get('full_traces',[])
        per=units['snapshot_envelope'] if record['identity']['controller_id']=='G' else units['full_trace_npz']+units['full_trace_identity']+units['snapshot_envelope']
        if traces:add('full_traces_and_inputs',[p for r in traces for p in r['files_sha256']],len(traces)*per)
        need(component.get('status')=='PASS_MEASURED_COMPONENT_ALLOWANCES' and same(component['components'],expected)
             and set(component['measured_component_paths'])==paths
             and component['mismatch_charged']==bool(record.get('resources',{}).get('mismatch_charged'))
             and all(r['within_allowance'] for r in expected.values()),'Independent measured component allowances differ')

    def cell(self,phase,cell):
        folder=self.run/({'E2':'E2','replay':'replays','application':'application'}[phase])/cell['family_id']/cell['controller_id']
        result=read(folder/'worker_result.json');self.hashed(folder/'worker_result.json',read(folder/'worker_result_receipt.json')['sha256'],folder)
        row=result['row'];need(result.get('phase')==phase and same(result.get('cell'),cell),'Worker phase/cell identity differs')
        need(all(row.get(k)==v for k,v in cell.items()),'Worker row identity differs')
        need(result['status'] in {'ACCEPT_COMPLETED_RECORDING','ACCEPT_COMPLETED_TASK_FAILURE'} and row['execution_status']=='completed','Unresolved cell infrastructure')
        need(row['termination'] in TERMINALS and type(row['success']) is bool and row['success']==(row['termination']=='ready') and result['P5_task_success']==row['success'],'Ready denominator differs')
        need(result.get('learning_updates')==0 and type(result.get('learning_updates')) is int,'Unexpected learning update')
        need(result.get('recording_guard_phase_after_return')=='CLOSED','Cell guard was not CLOSED')
        history=[r['phase'] for r in result['recording_guard_phase_history']]
        need(history==['CONTROL','SEALING','CLOSED'],'Unexpected guard phase transition history')
        guards=row.get('future_engine_calls');need(type(guards) is dict and set(guards)==ENGINE_GUARDS and all(type(v) is int and v==0 for v in guards.values()),'Missing/nonzero future-physics guard')
        record_path=self.hashed(result['record_manifest'],result['record_manifest_sha256'],folder)
        need(record_path==folder/'manifest.json','Recording path differs from cell')
        record=read(record_path);execution=record['execution']
        need(record['identity']['family_id']==cell['family_id'] and record['identity']['controller_id']==cell['controller_id']
             and record['identity']['final_manifest_sha256']==self.sha(self.run/'manifest.json'),'Recorded identity/frozen study differs')
        need(record['record_kind']==('fresh_controller_replay' if phase=='replay' else 'reference_controller_execution'),'Wrong original/replay record kind')
        need(all(same(row[k],execution[k]) for k in ('execution_status','termination','counts','future_engine_calls')),'Row differs from recorded execution')
        need(execution['error'] is None and execution['initial_full_snapshot_equal'] is True,'Execution error or initial-snapshot mismatch')
        need(record['source_binding_check_after_execution'] is True and record['missing_required_input_ticks']==[]
             and record['witness_validation_issues']==[],'Recording source/witness/input defect')
        self.bindings(record['files_sha256'],folder);self.bindings(record['source_bindings'])
        review=read(self.hashed(folder/'review.json',result['review_sha256'],folder))
        need(review['accepted'] is True and review['status']==result['status'] and review['manifest_sha256']==result['record_manifest_sha256'],'Recording review identity/status differs')
        need(all(same(row.get(k),v) for k,v in review['metrics'].items()),'Row metrics differ from sealed saved-raw review')
        task=execution['task'];need(task['terminal_reason']==row['termination'],'Terminal task reason differs')
        close(row['elapsed_s'],task['last_sim_time']-task['initial_sim_time'],'Task elapsed seconds')
        need(0<=row['elapsed_s']<=30.+1e-7,'Original30s task duration exceeded')
        if row['success']:
            specification=self.manifest['physical_config']['task']
            need(task['phase']=='ready' and task['previous_qualifies'] is True and task['held_time_s']+1e-9>=specification['hold_time_s']
                 and row['final_observation_available'] is True,'Ready task has insufficient recorded hold/final observation')
            for metric,threshold in (('final_position_error_m','position_tolerance_m'),('final_yaw_error_rad','yaw_tolerance_rad'),
                    ('final_linear_speed_m_s','linear_speed_tolerance_m_s'),('final_angular_speed_rad_s','angular_speed_tolerance_rad_s')):
                need(row[metric] is not None and row[metric]<=specification[threshold]+1e-9,'Ready endpoint exceeds '+threshold)
        else:need(task['phase']=='failed','Failed termination lacks failed task phase')
        if row['termination']=='timeout':close(row['elapsed_s'],30.,'Timeout duration')
        component=read(self.hashed(folder/'component_budget_review.json',result['component_review_sha256'],folder));self.component(record,component)
        if phase=='replay':
            need(result['comparison']['boundary']['status']=='PASS_REPRODUCTION'
                 and result['comparison']['raw_exact']['status']=='PASS_EXACT'
                 and result['comparison']['plan_index_exact']['status']=='PASS_EXACT','Fresh replay boundary/raw/index attestation failed')
        return row

    def e2(self):
        m=self.manifest;need(m is not None,'Manifest check must pass first')
        expected=m['E2'];need(len(expected)==96 and len({c['cell_id'] for c in expected})==96,'Expected96 population changed')
        families={c['family_id'] for c in expected};need(len(families)==12,'Expected12 family population changed')
        rows=[self.cell('E2',c) for c in expected];self.e2_rows=rows
        need({r['controller_id'] for r in rows}==set(CONTROLLERS),'Controller population differs')
        sources={str(self.run/'E2'/c['cell_id']/'worker_result.json'):self.sha(self.run/'E2'/c['cell_id']/'worker_result.json') for c in expected}
        self.sealed_stage('statistics',{'summary.json','bootstrap.npz','reproduction.json'},sources)
        summary=read(self.run/'statistics/summary.json')
        need(summary['status']=='COMPLETE_EXPECTED_E2_DENOMINATORS' and summary['episode_count']==96 and summary['family_count']==12 and summary['controller_count']==8,'E2 summary population differs')
        def summary_part(part,saved):
            counts=Counter(r['termination'] for r in part);success=sum(r['success'] for r in part)
            need(saved['episode_count']==len(part) and saved['family_count']==len({r['family_id'] for r in part})
                 and saved['success_count']==success and saved['timeout_count']==counts.get('timeout',0)
                 and saved['termination_counts']==dict(counts),'Independent E2 terminal counts differ')
            close(saved['success_rate'],success/len(part),'success_rate');close(saved['timeout_rate'],counts.get('timeout',0)/len(part),'timeout_rate')
            successes=[r for r in part if r['success']];time=saved['success_conditional_time_s']
            need(time['denominator']==len(successes),'Successful-only time denominator differs')
            close(time['mean'],mean(r['elapsed_s'] for r in successes),'Successful-only time mean')
            q=quantiles([r['elapsed_s'] for r in successes],[50]);close(time['median'],q[0] if q else None,'Successful-only time median')
            need(saved['terminal_observation_unavailable_count']==sum(not r['final_observation_available'] for r in part)
                 and saved['state_observation_populations']==dict(Counter(r['metric_state_origin'] for r in part)),'State origin/missing population differs')
            for metric in METRICS:
                need(saved['metric_denominators'][metric]==sum(r[metric] is not None for r in part),'Metric denominator differs: '+metric)
                close(saved['all_episode_means'][metric],mean(r[metric] for r in part),'Metric mean '+metric)
        for controller in CONTROLLERS:
            part=[r for r in rows if r['controller_id']==controller];need(len(part)==12,'Controller must have12 families')
            summary_part(part,summary['controllers'][controller])
            for stratum in STRATA:
                p=[r for r in part if r['stratum']==stratum];need(len(p)==4,'Each stratum needs4 families');summary_part(p,summary['controllers'][controller]['by_stratum'][stratum])
        for group,members in GROUPS.items():summary_part([r for r in rows if r['controller_id'] in members],summary['seed_group_summary'][group])
        reproduction=read(self.run/'statistics/reproduction.json')
        need(reproduction['status']=='PASS_EXACT_REGENERATION' and reproduction['source_rows']==96 and reproduction['families']==12,'E2 reproduction counts differ')
        self.hashed(self.run/'statistics/summary.json',reproduction['summary_sha256']);self.hashed(self.run/'statistics/bootstrap.npz',reproduction['bootstrap_sha256'])
        self.bootstrap(summary,self.run/'statistics/bootstrap.npz',rows,12,4,METRICS,e1=False)
        return {'rows':96,'families':12,'controllers':{c:{'success':summary['controllers'][c]['success_count'],
            'termination_counts':summary['controllers'][c]['termination_counts']} for c in CONTROLLERS},
            'independent_checks':'terminal/ready/stratum/seed/family-first means and bootstrap arithmetic from saved rows; raw physical trajectory not independently reconstructed here'}

    def bootstrap(self,summary,path,rows,family_count,per_stratum,metrics,*,e1):
        import numpy as np
        families=sorted({r['family_id'] for r in rows});strata={s:sorted({r['family_id'] for r in rows if r['stratum']==s}) for s in STRATA}
        need(len(families)==family_count and all(len(v)==per_stratum for v in strata.values()),'Independent resampling family population differs')
        settings=summary['bootstrap'];need(settings['seed']==7711 and settings['replicates']==2000 and settings['families_in_order']==families and settings['strata']==strata,'Bootstrap frozen seed/order/strata differ')
        rng=np.random.Generator(np.random.PCG64(7711));draws=np.concatenate([rng.choice([families.index(f) for f in strata[s]],size=(2000,per_stratum),replace=True) for s in STRATA],axis=1).astype(np.int64)
        groups={k:v for k,v in GROUPS.items() if not e1 or k!='G'};methods={**groups,**{c:(c,) for c in CONTROLLERS if c not in groups and (not e1 or c!='G')}}
        def value(row,metric):
            if not e1:return row[metric]
            if metric=='no_selection':return float(row['selected_index'] is None)
            if metric in ('selected_actual_invalid','selected_actual_validity_changed'):return float(row['selection_status']==metric)
            if metric=='any_validity_changed':return float(bool(row['validity_flip_indices']))
            return row[metric]
        by={(r['family_id'],r.get('model',r.get('controller_id')),r.get('tick')):r for r in rows}
        values={}
        for method,members in methods.items():
            for metric in metrics:
                per_family=[]
                for family in families:
                    per_seed=[]
                    for member in members:
                        points=[value(by[(family,member,t)],metric) for t in ((0,29) if e1 else (None,))]
                        per_seed.append(mean(points) if all(v is not None for v in points) else None)
                    per_family.append(mean(per_seed) if all(v is not None for v in per_seed) else math.nan)
                values[(method,metric)]=np.asarray(per_family,np.float64)
        contrasts=([('S1','R'),('SK','R'),('SK','S1')] if e1 else
                   [('R','G'),('S1','G'),('SK','G'),('S1','R'),('SK','R'),('SK','S1')])
        contrasts += [(f'SK-{seed}',f'S1-{seed}') for seed in (11,23,47)]
        need(len(summary['comparisons'])==len(contrasts)
             and {(c['left'],c['right']) for c in summary['comparisons']}==set(contrasts),'Missing/extra predeclared paired contrasts')
        expected_keys={'family_draw_indices'}
        for left,right in contrasts:
            for metric in metrics:
                key=f'{left}_minus_{right}_{metric}'
                expected_keys.update((key,key+'_valid'))
                if e1:expected_keys.update((key+'_eligible_resampled_positions',key+'_eligible_family'))
        if e1:
            for method,metric in values:expected_keys.update((f'family_{method}_{metric}',f'family_{method}_{metric}_defined'))
        with self.npz(path) as saved:
            need(set(saved.files)==expected_keys,'Bootstrap NPZ key population differs')
            self.array_equal(saved['family_draw_indices'],draws,('E1' if e1 else 'E2')+' family_draw_indices')
            for comparison in summary['comparisons']:
                left,right=comparison['left'],comparison['right']
                need(set(comparison['metrics'])==set(metrics),'Comparison metric population differs')
                for metric in metrics:
                    differences=values[(left,metric)]-values[(right,metric)];eligible=np.isfinite(differences)
                    stat=comparison['metrics'][metric];need(stat['eligible_family_denominator']==int(eligible.sum()),'Paired eligible family denominator differs')
                    close(stat['difference_mean'],mean(differences[eligible]),'Paired family difference '+metric)
                    need(len(stat['family_differences'])==family_count,'Paired family differences count differs')
                    for actual,expected in zip(stat['family_differences'],differences):close(actual,float(expected) if math.isfinite(expected) else None,'Family difference '+metric)
                    sampled=differences[draws];counts=np.isfinite(sampled).sum(axis=1).astype(np.int64)
                    boot=np.asarray([mean([v for v in row if math.isfinite(v)]) for row in sampled],dtype=np.float64)
                    valid=np.isfinite(boot);key=f'{left}_minus_{right}_{metric}'
                    self.array_equal(saved[key],boot,('E1' if e1 else 'E2')+' '+key,numeric=True);self.array_equal(saved[key+'_valid'],valid,key+'_valid')
                    need(stat['bootstrap_valid_replicates']==int(valid.sum()),'Bootstrap valid replicate denominator differs')
                    interval=quantiles(boot[valid],[2.5,97.5]);need((stat['percentile95_interval'] is None)==(interval is None),'Undefined percentile interval differs')
                    if interval:
                        for a,b in zip(stat['percentile95_interval'],interval):close(a,b,'Manual percentile interval')
                        need(stat['zero_in_interval']==(interval[0]<=0<=interval[1]),'Interval sign classification differs')
                    if e1:
                        need(stat['original_family_denominator']==6 and stat['bootstrap_replicate_denominator']==2000 and stat['bootstrap_unavailable_replicates']==int((~valid).sum()) and stat['interval_conditions_on_nonempty_eligible_resample'] is True,'E1 conditional replicate accounting differs')
                        self.array_equal(saved[key+'_eligible_resampled_positions'],counts,key+'_eligible_resampled_positions');self.array_equal(saved[key+'_eligible_family'],eligible,key+'_eligible_family')
            if e1:
                for (method,metric),array in values.items():
                    self.array_equal(saved[f'family_{method}_{metric}'],array,f'family_{method}_{metric}',numeric=True)
                    self.array_equal(saved[f'family_{method}_{metric}_defined'],np.isfinite(array),f'family_{method}_{metric}_defined')

    def e1(self):
        m=self.manifest;need(m is not None,'Manifest check must pass first');folder=self.run/'E1'
        summary=read(folder/'summary.json');seal=read(folder/'stage_receipt.json')
        need(summary['status']=='COMPLETE_RECORDED' and same(summary['expected_snapshots'],m['E1']) and len(m['E1'])==12,'E1 frozen snapshot population incomplete')
        for key,value in [('completed_snapshots',12),('expected_prediction_sets',84),('completed_prediction_sets',84),('diagnostic_count',84),('expected_physical_repeats',576),('attempted_physical_repeats',576),('completed_physical_repeats',576),('new_learning_updates',0),('new_closed_loop_cells',0)]:need(summary[key]==value,'E1 count differs: '+key)
        need(summary['all_sources_and_model_weights_unchanged'] is True and set(summary['future_engine_calls_during_model_prediction'])==ENGINE_GUARDS and all(type(v) is int and v==0 for v in summary['future_engine_calls_during_model_prediction'].values()),'E1 source/oracle guard differs')
        need(seal['schema']=='p5-E1-stage-seal-v1' and seal['frozen_manifest_sha256']==self.sha(self.run/'manifest.json') and seal['all576_raw_prefixes_and84_base_diagnostics_regenerated'] is True,'E1 original adoption seal missing')
        self.hashed(folder/'summary.json',seal['summary_sha256']);self.bindings(seal['files_sha256'],folder)
        expected_ids={spec['snapshot_id']+f'/candidate-{candidate:02d}/repeat-{repeat}' for spec in m['E1'] for candidate in range(16) for repeat in range(3)}
        need(len(seal['completed_repeat_ids'])==576 and set(seal['completed_repeat_ids'])==expected_ids,'E1 repeat IDs differ')
        prediction=read(folder/'receipts/predictions.json');need(prediction['schema']=='p5-E1-prediction-seal-v1' and prediction['frozen_manifest_sha256']==self.sha(self.run/'manifest.json') and prediction['prediction_sets']==84 and prediction['physical_future_started'] is False,'Before-future prediction seal differs')
        self.bindings(prediction['files_sha256'],folder)
        for repeat_id in sorted(expected_ids):
            p=folder/'receipts/repeats'/(hashlib.sha256(repeat_id.encode()).hexdigest()+'.json');r=read(p)
            need(r['repeat_id']==repeat_id and r['schema']=='p5-E1-repeat-seal-v1' and r['frozen_manifest_sha256']==self.sha(self.run/'manifest.json'),'E1 repeat receipt differs')
            self.hashed(folder/'receipts/predictions.json',r['prediction_seal_sha256']);self.hashed(r['summary_path'],r['summary_sha256'],folder);self.bindings(r['raw_files_sha256'],folder)
        ledger=read(folder/'comparison_ledger.json');desired={(s['snapshot_id'],model) for s in m['E1'] for model in CONTROLLERS[1:]}
        need(len(ledger)==84 and len({(r['snapshot_id'],r['model']) for r in ledger})==84 and {(r['snapshot_id'],r['model']) for r in ledger}==desired,'E1 ledger84 population differs')
        for row in ledger:
            actual=row['actual_records'];need(len(actual)==16 and {r['index'] for r in actual}==set(range(16)),'E1 actual candidate denominator differs')
            for candidate in actual:
                need(len(candidate['repeated_valid'])==len(candidate['repeated_costs'])==3
                     and all(type(v) is bool for v in candidate['repeated_valid']),'E1 three-repeat validity population differs')
                if all(candidate['repeated_valid']):
                    need(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in candidate['repeated_costs']),'Stable-valid actual candidate lacks three finite costs')
            stable={r['index']:mean(r['repeated_costs']) for r in actual if all(r['repeated_valid'])}
            valid_repeats=sum(sum(r['repeated_valid']) for r in actual)
            need(row['counts']['actual_repeat_attempts']==48 and row['counts']['actual_repeat_valid']==valid_repeats
                 and row['counts']['actual_repeat_invalid']==48-valid_repeats
                 and row['actual_all_invalid']==(valid_repeats==0)
                 and row['no_stable_actual_valid']==(not stable),'E1 actual validity denominator/flags differ')
            selected=row['selected_index']
            need(selected is None or type(selected) is int and 0<=selected<16,'E1 selected candidate identity differs')
            loss=(stable[selected]-min(stable.values())) if selected in stable else None
            close(row['selection_loss'],loss,'Independent three-repeat selection loss')
            need(row['selection_loss_denominator']==int(loss is not None),'Conditional loss denominator differs')
            if loss is not None:need(row['selection_loss_ordering'] in {'INCONCLUSIVE','SELECTED_WORSE'},'Conditional ordering label differs')
        for spec in m['E1']:
            part=[r for r in ledger if r['snapshot_id']==spec['snapshot_id']]
            need(all(same(r['actual_records'],part[0]['actual_records']) and r['input_hash']==part[0]['input_hash'] and r['candidate_hash']==part[0]['candidate_hash'] for r in part),'E1 models do not share actual labels/input/candidates')
        sources={str(self.run/p):self.sha(self.run/p) for p in ('manifest.json','E1/summary.json','E1/stage_receipt.json','E1/comparison_ledger.json')}
        self.sealed_stage('e1_statistics',{'summary.json','bootstrap.npz','reproduction.json'},sources)
        stats=read(self.run/'e1_statistics/summary.json')
        need(stats['status']=='COMPLETE_EXPECTED_E1_DENOMINATORS' and stats['model_set_count']==84 and stats['snapshot_count']==12 and stats['independent_family_count']==6 and stats['shared_physical_repeat_attempts']==576,'E1 statistics denominators differ')
        def indicator(row,metric):
            if metric=='no_selection':return row['selected_index'] is None
            if metric in ('selected_actual_invalid','selected_actual_validity_changed'):return row['selection_status']==metric
            if metric=='any_validity_changed':return bool(row['validity_flip_indices'])
            return row[metric]
        for metric in INDICATORS:
            entry=stats['original_model_set_populations'][metric]
            need(entry['denominator']==84 and entry['numerator']==sum(bool(indicator(r,metric)) for r in ledger),'E1 original indicator denominator/numerator differs')
        reproduction=read(self.run/'e1_statistics/reproduction.json');need(reproduction['status']=='PASS_EXACT_REGENERATION' and reproduction['source_rows']==84 and reproduction['snapshots']==12 and reproduction['families']==6,'E1 statistics reproduction population differs')
        self.hashed(self.run/'e1_statistics/summary.json',reproduction['summary_sha256']);self.hashed(self.run/'e1_statistics/bootstrap.npz',reproduction['bootstrap_sha256'])
        self.bootstrap(stats,self.run/'e1_statistics/bootstrap.npz',ledger,6,2,E1_METRICS,e1=True)
        return {'snapshots':12,'families':6,'models':7,'model_sets':84,'shared_repeats':576,
            'independent_arithmetic':'selected loss from stored three-repeat costs, original indicator rates, family-first paired values/bootstrap',
            'not_repeated_here':'Raw state/action/cost generation and model prediction; original stage source-seal attestation is checked but not independently recreated'}

    def fixed_prediction(self):
        import numpy as np
        folder=self.run/'fixed_prediction';self.sealed_stage('fixed_prediction',{'summary.json','traces.npz'})
        summary=read(folder/'summary.json');need(summary['schema']=='p5-common-G-fixed-trace-prediction-v1' and summary['status']=='COMPLETE_DECLARED_RAW_EVALUATION' and summary['family_count']==12 and set(summary['horizons'])=={'1','5','10'},'Fixed prediction population/status differs')
        need(summary['additional_physics_intervals']==0 and summary['online_settings_mutated'] is False and set(summary['future_engine_calls'])==ENGINE_GUARDS and all(type(v) is int and v==0 for v in summary['future_engine_calls'].values()),'Prediction physics/settings guard differs')
        need(set(summary['family_ids'])=={r['family_id'] for r in self.manifest['families']} and set(summary['inference'])==set(CONTROLLERS[1:]),'Fixed G family/model set differs')
        metrics=('object_xy_error_m','object_yaw_error_rad_2pi','pusher_xy_error_m','object_linear_velocity_error_m_s','object_angular_velocity_error_rad_s','pusher_velocity_error_m_s')
        with self.npz(folder/'traces.npz') as arrays:
            for horizon in (1,5,10):
                prefix=f'K{horizon}__';truth=arrays[prefix+'truth_valid'];family=arrays[prefix+'family_id'];count=len(truth);h=summary['horizons'][str(horizon)]
                need(truth.dtype==np.bool_ and h['recorded_start_opportunities']==count and h['truth_eligible_windows']==int(truth.sum()),'Truth-window denominator differs')
                need(h['boundary_excluded_windows']+h['full_horizon_windows']==count and h['raw_invalid_full_windows']+h['truth_eligible_windows']==h['full_horizon_windows'],'Truth boundary/invalid denominator differs')
                for model in CONTROLLERS[1:]:
                    pre=prefix+model+'__';valid=arrays[pre+'valid'];mask=truth&valid;s=h['models'][model]
                    need(s['recorded_start_opportunities']==count and s['truth_eligible_windows']==int(truth.sum()) and s['predicted_valid_windows']==int(mask.sum()) and s['predicted_invalid_windows']==int((truth&~valid).sum()),'Model prediction denominator differs')
                    for metric in metrics:
                        values=arrays[pre+metric];saved=s['metrics_conditional_on_valid_prediction_and_truth'][metric];selected=values[mask]
                        need(saved['denominator']==len(selected),'Prediction metric denominator differs');close(saved['mean'],mean(selected),'Prediction metric mean')
                        q=quantiles(selected,[50,95]);close(saved['median'],q[0] if q else None,'Prediction metric median');close(saved['p95'],q[1] if q else None,'Prediction metric p95');close(saved['max'],max(selected) if len(selected) else None,'Prediction metric max')
                        family_means=[]
                        for identity in summary['family_ids']:
                            vals=values[mask&(family==identity)];family_means.append(mean(vals))
                        f=s['family_first_means_conditional_on_available_families'][metric]
                        need(f['family_denominator']==sum(x is not None for x in family_means),'Prediction family-first denominator differs');close(f['mean'],mean(family_means),'Prediction family-first mean')
        return {'families':12,'models':7,'horizons':[1,5,10],
            'scope':'Independent arithmetic on saved validity masks and saved endpoint-error arrays; source hashes verified. Does not independently reconstruct truth windows, infer model outputs or integrate stored deltas.'}

    def replay_and_application(self):
        need(len(self.e2_rows)==96,'Replay selection requires all independently checked96 rows')
        chosen=[]
        for controller in CONTROLLERS:
            part=[r for r in self.e2_rows if r['controller_id']==controller]
            ids={min((r for r in part if r['stratum']==s),key=lambda r:r['episode_id'])['cell_id'] for s in STRATA}
            for terminal in {r['termination'] for r in part}:ids.add(min((r for r in part if r['termination']==terminal),key=lambda r:r['episode_id'])['cell_id'])
            chosen.extend(r for r in sorted(part,key=lambda r:r['episode_id']) if r['cell_id'] in ids)
        need(same(read(self.run/'replay_selection.json'),chosen),'Replay union population differs from independent stratum/terminal rule')
        by={r['cell_id']:r for r in self.manifest['E2']}
        replays=[self.cell('replay',by[r['cell_id']]) for r in chosen]
        family=self.manifest['application_family'];application=[]
        for controller in CONTROLLERS:
            cell={'family_id':family['family_id'],'episode_id':family.get('episode_id',family['family_id']),
                  'episode_index':12,'stratum':family.get('stratum','application'),'controller_id':controller,'cell_id':family['family_id']+'/'+controller}
            application.append(self.cell('application',cell))
        state=read(self.run/'progress.json');self.hashed(self.run/'progress.json',read(self.run/'progress_receipt.json')['sha256'])
        for key,expected in [('E2_done',set(by)),('E1_done',{s['snapshot_id']+f'/candidate-{c:02d}/repeat-{r}' for s in self.manifest['E1'] for c in range(16) for r in range(3)}),('replay_done',{r['cell_id'] for r in chosen}),('application_done',set(CONTROLLERS))]:
            need(len(state[key])==len(expected) and set(state[key])==expected,'Final progress population differs: '+key)
        return {'independently_selected_replays':len(replays),'application_rows':8,'application_families':1,
            'application_successes':sum(r['success'] for r in application),'replay_interpretation':'Saved fresh-controller boundary PASS checked; identical simulator internals/substep bytes are not asserted by this auditor.'}

    def result(self):
        complete=self.run/'execution_complete.json'
        self.section('execution_complete_entry',lambda: self.entry(complete))
        if not self.errors:
            self.section('freeze_sources_parent_preservation',self.freeze_and_sources)
            self.section('E2_independent_arithmetic',self.e2)
            self.section('E1_saved_ledger_and_independent_arithmetic',self.e1)
            self.section('fixed_prediction_saved_error_arithmetic',self.fixed_prediction)
            self.section('replay_and_application_complete_population',self.replay_and_application)
        return {'schema':'p5-extra-independent-saved-evidence-closeout-v1',
            'status':'PASS_INDEPENDENT_SAVED_EVIDENCE_CLOSEOUT' if not self.errors else 'FAIL_RETAIN_DIAGNOSTICS',
            'accepted':not self.errors,'run':str(self.run),'checked_at_utc':datetime.now(timezone.utc).isoformat(),
            'sections':self.sections,'errors':self.errors,'bootstrap_array_checks':self.array_checks,
            'io':{'unique_stat_version_files_hashed':self.files_hashed,'bytes_hashed':self.bytes_hashed},
            'extra_unfrozen_auditor':{'path':str(Path(__file__).resolve()),'sha256':self.sha(Path(__file__).resolve()),'frozen_runtime_modified':False},
            'execution':{'workers_started':0,'model_loads':0,'model_forward_calls':0,'physics_steps':0,'training_updates':0,'source_or_result_edits':0},
            'claim_limits':['This is an additional auditor written after freeze, not frozen experimental treatment or a newly independent statistical estimator.',
                'Counts, family-first contrasts and stored-bootstrap arithmetic are independently implemented; physical outcomes are checked against saved sealed reviews, not re-simulated.',
                'E1 actual costs and fixed prediction errors are read from stored evidence; raw physical truth, model inference and all original diagnostic algorithms are not independently regenerated here.',
                'Numeric arithmetic uses rtol=atol=1e-12; file SHA and identity/population/boolean checks are exact. NPZ arrays retain independent byte digests.',
                'PASS does not prove model superiority, equivalence, industrial value, real-time hardware performance, generalization or personal understanding.',
                'Parent outcomes were observed and the same initial conditions restarted; parent and run02 are not pooled or selected by best outcome.',
                'Visual/manual acceptance, reporting and personal understanding remain separate completion questions.']}

    def entry(self,path):
        value=read(path);need(value.get('status')=='ALL_DECLARED_EXECUTION_STAGES_RECORDED','Full execution_complete prerequisite is absent/unresolved')
        return {'path':str(path),'sha256':self.sha(path),'status':value['status']}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv);run=args.run.resolve();out=args.out.resolve()
    need(run.parent.name=='runs' and run.name.startswith('p5-final-'),'Managed frozen final run required')
    need(not out.exists(),'Output must not exist; prior diagnostic report is never replaced')
    need(out.is_relative_to(run.parent.parent) and out.suffix=='.json','Audit output must be a new project JSON artifact')
    need(out.relative_to(run.parent.parent).parts[0] in {'reports','tmp'},'Audit output must belong to reports/ or tmp/, never source or experiment paths')
    audit=Audit(run)
    try:result=audit.result()
    except Exception as error:
        result={'schema':'p5-extra-independent-saved-evidence-closeout-v1','status':'FAIL_RETAIN_DIAGNOSTICS','accepted':False,
                'run':str(run),'error':str(error),'traceback':traceback.format_exc(),'source_or_result_edits':0}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as stream:json.dump(result,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':result['status'],'accepted':result['accepted'],'out':str(out)},ensure_ascii=False))
    return 0 if result['accepted'] else 2


if __name__=='__main__':
    raise SystemExit(main())
