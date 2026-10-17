"""Saved KEEP/ZERO_M noncontact addends evaluated on the same final-R1 stop batch."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import shutil
import sys
from time import perf_counter
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from push_align.first_moment_probe import first_moment_probe
from push_align.cross_batch_probe import array_sha256
from push_align.data_intervention_training import tree_hash
from push_align.normalization import load_normalizer
from p4_finite_step_probe import guards, load_frozen_snapshot

ROOT = Path(__file__).resolve().parents[1]
GRAD = ROOT / 'runs/p4-gradient-probe-20261015-01'
PARENT = ROOT / 'runs/p4-relative-loss-20261014-01'
NORMALIZER = ROOT / 'runs/p3-02-20261009-01/learning_data/normalizer'
SNAPSHOT, SOURCE, TARGET = 'R1_u1000', 'D11_noncontact', 'D11_zero5'
MODES, ALPHAS = ('KEEP', 'ZERO_M'), (0., .25, .5, 1.)
RELATIONS = ('INCREASE', 'DECREASE', 'NUMERIC_TIE', 'NOT_INTERPRETED')


def require(condition, message):
    if not condition: raise ValueError(message)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def save(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
def arrays(path):
    with np.load(path, allow_pickle=False) as data: return {k:data[k].copy() for k in data.files}


def validate(path, expected):
    require(sha(path) == expected, 'Execution registration changed')
    p = read(path)
    design = read(ROOT / p['design_path'])
    require(sha(ROOT / p['design_path']) == p['design_sha256'], 'Frozen design changed')
    require(p['run_id'] == 'p4-first-moment-20261015-01' and p['cases'] == design['cases'] and len(p['cases']) == 2
            and p['snapshots'] == design['snapshots'] and len(p['snapshots']) == 1
            and p['snapshots'][0]['snapshot_id'] == SNAPSHOT
            and [(c['snapshot_id'],c['source_probe_name'],c['target_probe_name'],c['source_mode'],c['source_direction_index']) for c in p['cases']]
               == [(SNAPSHOT,SOURCE,TARGET,'KEEP',3),(SNAPSHOT,SOURCE,TARGET,'ZERO_M',4)]
            and p['alphas'] == design['alphas'] == list(ALPHAS), 'Snapshot/source/mode/target/alpha scope changed')
    for name, digest in p['bindings_sha256'].items():
        require((ROOT / name).is_file() and sha(ROOT / name) == digest, 'Frozen source changed: '+name)
    require(str(torch.__version__) == p['runtime']['torch_version'] and np.__version__ == p['runtime']['numpy_version']
            and platform.python_version() == p['runtime']['python_version'], 'Runtime changed')
    for name,digest in p['runtime']['source_sha256'].items(): require(sha(name)==digest, 'Functional runtime source changed')
    return p


def global_controls(rows):
    keep = [r for r in rows if r['source_mode'] == 'KEEP']
    checks = {
        'exactly_keep_zero_m': len(rows)==2 and [r['source_mode'] for r in rows]==list(MODES)
            and all((r['snapshot_id'],r['source_probe_name'],r['target_probe_name'])==(SNAPSHOT,SOURCE,TARGET) for r in rows),
        'two_alpha0_previous_stop_exact': len(rows)==2 and all(r['controls']['alpha0_matches_previous_stop'] for r in rows),
        'two_alpha0_equal': len(rows)==2 and all(r['controls']['alpha0_matches_keep'] for r in rows),
        'keep_previous_cross_exact': len(keep)==1 and keep[0]['controls']['keep_previous_cross'] is not None and keep[0]['controls']['keep_previous_cross']['pass'],
    }
    return {'pass':all(checks.values()),'checks':checks,'scope':'Exact same-output path controls, not independent replication.'}


def primary_counts(rows, controls_pass):
    if not controls_pass: return None
    def group(selected):
        values = {}
        for i,alpha in enumerate(ALPHAS[1:],1):
            counts = {k:0 for k in RELATIONS}
            for r in selected:
                e=r['evaluation']; relation=e['evaluations'][i]['total_change_relation'] if e['status']=='PASS' else 'NOT_INTERPRETED'
                require(relation in RELATIONS, 'Unknown endpoint relation'); counts[relation]+=1
            values[str(alpha)] = counts
        return {'case_count':len(selected),'planned_positive_endpoints':3*len(selected),'by_alpha':values}
    by_mode = {m:group([r for r in rows if r['source_mode']==m]) for m in MODES}
    zero = next(r['evaluation'] for r in rows if r['source_mode']=='ZERO_M')
    return {'all':group(rows),'by_mode':by_mode,
        'zero_m_increase_persists':{str(a):None if zero['status']!='PASS' else zero['evaluations'][i]['total_change_relation']=='INCREASE' for i,a in enumerate(ALPHAS[1:],1)},
        'interpretation_gate':'PASS' if all(r['evaluation']['status']=='PASS' for r in rows) else 'HOLD_CASE_FAILURE'}


def analysis_status(rows, controls_pass):
    if not controls_pass: return 'HOLD_CONTROL_FAILURE'
    return 'DESCRIPTIVE_COMPLETE' if all(r['evaluation']['status']=='PASS' for r in rows) else 'HOLD_CASE_FAILURE'


def storage(out,p):
    size=sum(f.stat().st_size for f in out.rglob('*') if f.is_file()); free=shutil.disk_usage(ROOT).free
    require(size<=p['maximum_run_bytes'] and free>=p['minimum_free_reserve_bytes'], 'Storage guard exceeded')
    return {'run_bytes_before_index':size,'free_bytes':free}


def run(protocol_path,expected):
    began=perf_counter(); p=validate(protocol_path,expected); out=ROOT/p['run_directory']
    require(not out.exists(),'Never overwrite a run')
    require(shutil.disk_usage(ROOT).free>=sum(p[k] for k in ('minimum_free_reserve_bytes','maximum_run_bytes','document_reserve_bytes')),'Insufficient reserve')
    out.mkdir(parents=True); save(out/'registration.json',p)
    forbidden,originals=guards(); old_threads=torch.get_num_threads(); old_det=torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); rng=torch.get_rng_state().clone()
    counts={'snapshots':0,'cases':0,'attempted_evaluations':0,'completed_evaluations':0,'forward_calls':0,'forward_rows':0}
    try:
        meta=read(NORMALIZER/'normalizer.json')['normalizer']
        normalizer=load_normalizer(NORMALIZER,meta['schema'],meta['dataset_identity'],meta['normalizer_hash'])
        parent=read(PARENT/'manifest.json'); config=parent['physical_config']
        require(normalizer.metadata['normalizer_hash']==parent['normalizer_hash'],'Normalizer mismatch')
        batches=arrays(GRAD/'probe_batches.npz'); target_arrays={k:batches[TARGET+'_'+k] for k in ('states','actions','mask')}
        require(target_arrays['states'].shape==(24,6,10) and target_arrays['actions'].shape==(24,5,2)
                and target_arrays['mask'].shape==(24,5) and target_arrays['mask'].all(),'Stop target differs')
        spec=p['snapshots'][0]; model=load_frozen_snapshot(spec,normalizer,config); model_hash=tree_hash(model.state_dict())
        require(model_hash==spec['model_tree_hash'] and sum(v.numel() for v in model.parameters())==19338,'Source model differs')
        optimizer=torch.load(ROOT/spec['checkpoint_path']/'training_state.pt',map_location='cpu',weights_only=True)['optimizer']; opt_hash=tree_hash(optimizer)
        save(out/'manifest.json',{'schema':'p4-first-moment-manifest-v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'protocol_sha256':expected,'model_snapshots':p['snapshots'],'source_probe_name':SOURCE,'target_probe_name':TARGET,'source_modes':list(MODES),
            'normalizer_hash':normalizer.metadata['normalizer_hash'],'physical_config':config,'target_batch_windows':24,
            'target_array_sha256':{k:array_sha256(v) for k,v in target_arrays.items()},'probe_batches_sha256':sha(GRAD/'probe_batches.npz'),
            'runtime':p['runtime'],'alphas':list(ALPHAS),'scope':'Only saved past first moment m removed in ZERO_M; native norms, saved v/t and current clipped source gradient retained. No training.'})
        rows=[]; keep0=None
        for case in p['cases']:
            mode=case['source_mode']; index=case['source_direction_index']; d=read(ROOT/case['source_direction_summary']); t=read(ROOT/case['target_gradient_summary']); order=t['parameter_order']
            require(d['parameter_order']==order and d['snapshot_id']==t['snapshot_id']==SNAPSHOT and d['probe_name']==SOURCE and t['probe_name']==TARGET
                    and d['source_weights_tree_hash']==t['before_weights_hash']==model_hash and d['source_optimizer_tree_hash']==opt_hash,'Source/target coordinates differ')
            record=d['directions'][index]
            require(record['objective']=='total' and record['mode']==mode and record['saved_step']==2300 and record['next_step']==2301,'Direction mode/step differs')
            require(d['adam']['learning_rate']==.001 and d['adam']['betas']==[.9,.999] and d['adam']['epsilon']==1e-8 and d['adam']['weight_decay']==0 and not d['adam']['amsgrad'],'Adam constants differ')
            delta=arrays(ROOT/case['source_direction_npz'])['directions'][index].copy(); gradient=arrays(ROOT/case['target_gradient_npz'])['gradients'][4].copy()
            states,actions,mask=(torch.from_numpy(target_arrays[k].copy()) for k in ('states','actions','mask'))
            common={'snapshot_id':SNAPSHOT,'model_tree_hash':model_hash,'parameter_order':order}
            source_binding={**common,'probe_name':SOURCE,'objective':'total','mode':mode,'direction_index':index,'delta_sha256':array_sha256(delta)}
            target_binding={**common,'probe_name':TARGET,'batch_windows':24,**{k+'_sha256':array_sha256(v) for k,v in target_arrays.items()}}
            gradient_binding={**common,'probe_name':TARGET,'component':'total','gradient_sha256':array_sha256(gradient)}
            loss_binding={**common,'probe_name':TARGET,'loss_values':t['loss_values']}
            trace=[]
            def count(active_model,inputs):
                size=int(inputs[0].shape[0]); counts['forward_calls']+=1; counts['forward_rows']+=size
                require(counts['forward_calls']<=48 and counts['forward_rows']<=1920,'Forward budget exceeded')
                trace.append({'rows':size,'parameter_tree_hash':tree_hash(active_model.state_dict())})
            handle=model.register_forward_pre_hook(count)
            try:
                result=first_moment_probe(model,normalizer,states,actions,mask,delta,source_binding=source_binding,target_binding=target_binding,
                    target_gradient_binding=gradient_binding,target_loss_binding=loss_binding,target_stop_gradient=gradient,target_stop_reference=t['loss_values'],parameter_order=order)
            finally: handle.remove()
            e=result['summary']; cursor=0
            require(len(trace)==e['counters']['root_model_forward_calls'] and sum(v['rows'] for v in trace)==e['counters']['root_model_forward_rows'],'Proxy and actual counts differ')
            for i,endpoint in enumerate(e['evaluations']):
                params={v['name']:torch.from_numpy(result['evaluated_parameters'][i,v['start']:v['end']].reshape(v['shape']).copy()) for v in order}; h=tree_hash(params)
                actual=trace[cursor:cursor+endpoint['forward_calls']]
                require([v['rows'] for v in actual]==endpoint['forward_rows'] and all(v['parameter_tree_hash']==h for v in actual),'Actual parameter map differs');cursor+=endpoint['forward_calls']
            require(cursor==len(trace),'Unassigned forwards')
            previous=read(ROOT/case['prevcross_summary'])['evaluation']; old_params=arrays(ROOT/case['prevcross_parameters']); old_preds=arrays(ROOT/case['prevcross_predictions'])
            current0={'losses':e['evaluations'][0]['loss_values'],'teacher':result['teacher_predictions'][0].copy(),'free':result['free_predicted_states'][0].copy()}
            previous0=current0['losses']==previous['evaluations'][0]['loss_values'] and np.array_equal(current0['teacher'],old_preds['teacher_predictions'][0]) and np.array_equal(current0['free'],old_preds['free_predicted_states'][0])
            if mode=='KEEP': keep0=current0
            same0=current0['losses']==keep0['losses'] and np.array_equal(current0['teacher'],keep0['teacher']) and np.array_equal(current0['free'],keep0['free'])
            keep_control=None
            if mode=='KEEP':
                checks={'all_four_loss_values_statuses_exact':all(n['loss_values']==o['loss_values'] and n['status']==o['status'] for n,o in zip(e['evaluations'],previous['evaluations'])),
                    'parameter_maps_exact':all(result[k].dtype==old_params[k].dtype and np.array_equal(result[k],old_params[k]) for k in ('alphas','evaluated_parameters','rounded_parameter_differences')),
                    'teacher_predictions_exact':np.array_equal(result['teacher_predictions'],old_preds['teacher_predictions']),
                    'free_predictions_exact':np.array_equal(result['free_predicted_states'],old_preds['free_predicted_states'])}
                keep_control={'pass':all(checks.values()),'checks':checks}
            controls={'alpha0_matches_previous_stop':bool(previous0),'alpha0_matches_keep':bool(same0),'keep_previous_cross':keep_control}
            folder=out/'cases'/SNAPSHOT/mode; folder.mkdir(parents=True)
            np.savez_compressed(folder/'parameter_displacements.npz',**{k:result[k] for k in ('alphas','evaluated_parameters','rounded_parameter_differences')})
            np.savez_compressed(folder/'predictions.npz',**{k:result[k] for k in ('teacher_predictions','free_predicted_states')})
            saved={'snapshot_id':SNAPSHOT,'source_probe_name':SOURCE,'target_probe_name':TARGET,'source_mode':mode,'source':case,
                'parameter_order':order,'evaluation':e,'controls':controls,'source_weights_tree_hash':model_hash,'source_optimizer_tree_hash':opt_hash,
                'source_forward_trace':trace,'parameter_arrays_sha256':sha(folder/'parameter_displacements.npz'),'prediction_arrays_sha256':sha(folder/'predictions.npz')}
            require(tree_hash(model.state_dict())==model_hash and tree_hash(optimizer)==opt_hash,'Original source mutated'); save(folder/'summary.json',saved)
            rows.append({k:saved[k] for k in ('snapshot_id','source_probe_name','target_probe_name','source_mode','evaluation','controls')});rows[-1]['summary_path']=str((folder/'summary.json').relative_to(out))
            counts['cases']+=1;counts['attempted_evaluations']+=4;counts['completed_evaluations']+=e['counters']['successful_evaluations'];storage(out,p)
        counts['snapshots']=1
        require(counts['cases']==2 and counts['attempted_evaluations']==8,'Denominator changed')
        if counts['completed_evaluations']==8: require(counts==p['expected_if_all_evaluations_complete'],'Complete compute budget changed')
        require(not any(forbidden.values()) and torch.equal(rng,torch.get_rng_state()),'Forbidden call or RNG change');validate(protocol_path,expected)
        controls=global_controls(rows);effects=primary_counts(rows,controls['pass'])
        save(out/'aggregate.json',{'schema':'p4-first-moment-aggregate-v1','row_count':2,'rows':rows,'global_controls':controls,'primary_counts':effects})
        summary={'schema':'p4-first-moment-summary-v1','execution_status':'PASS','analysis_status':analysis_status(rows,controls['pass']),
            'protocol_sha256':expected,'manifest_sha256':sha(out/'manifest.json'),'counters':counts,'global_controls':controls,'primary_counts':effects,
            'forbidden_execution_counts':forbidden,'source_tensor_checks':[{'snapshot_id':SNAPSHOT,'weights_unchanged':tree_hash(model.state_dict())==model_hash,'optimizer_unchanged':tree_hash(optimizer)==opt_hash,'weight_tree_hash':model_hash,'optimizer_tree_hash':opt_hash}],
            'global_rng_unchanged':True,'new_derivative_traversals':0,'new_optimizer_updates':0,'new_physics_steps':0,'new_checkpoint_writes':0,
            'new_data':False,'normalizer_refit':False,'validation_or_development_probes_used':False,'final_test_used':False,'historical_trajectory_replayed':False,
            'model_selection':'NONE','learning_rate_adoption':'NONE','optimizer_change_adoption':'NONE','alignment_improvement':'NOT_EVALUATED','personal_understanding':'NOT_ASSESSED','P5_01':'HOLD',
            'wall_time_runner_only_s':perf_counter()-began,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),'resources':storage(out,p)}
        save(out/'summary.json',summary);save(out/'artifact_hashes.json',{'sha256':{str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()}});return summary
    except Exception as error:
        save(out/'failure.json',{'status':'FAIL','type':type(error).__name__,'error':str(error),'counters':counts,'forbidden_execution_counts':forbidden,'traceback':traceback.format_exc(),'automatic_retry':False});raise
    finally:
        for obj,name,function in reversed(originals): setattr(obj,name,function)
        torch.set_num_threads(old_threads);torch.use_deterministic_algorithms(old_det)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--protocol-sha256',required=True)
    args=parser.parse_args();print(json.dumps(run(args.protocol,args.protocol_sha256),ensure_ascii=False),flush=True)
