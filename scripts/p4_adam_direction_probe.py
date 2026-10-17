"""Registered saved-gradient Adam addends: never apply an update to a model."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import platform
import resource
import shutil
import sys
from time import perf_counter
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import mujoco
import numpy as np
import torch
from push_align.adam_direction_probe import virtual_directions
from push_align.adam_history import summarize_histories
from push_align.data_intervention_training import tree_hash
ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'runs/p4-gradient-probe-20261015-01'
PARENT=ROOT/'runs/p4-relative-loss-20261014-01'

def require(c,message):
    if not c:raise ValueError(message)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,x):Path(path).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def validate(path,expected):
    require(sha(path)==expected,'Protocol identity changed')
    p=read(path)
    require(p['run_id']=='p4-adam-probe-20261015-01' and p['virtual_direction_count']==168
        and p['objective_order']==['base','total'] and p['mode_order']==['KEEP','ZERO_M','COLD'], 'Unregistered scope')
    for n,h in p['bindings_sha256'].items():
        require((ROOT/n).is_file() and sha(ROOT/n)==h,'Frozen source changed: '+n)
    adam=importlib.import_module('torch.optim.adam')
    require(sha(inspect.getsourcefile(adam))==p['runtime']['torch_adam_source_sha256']
        and str(torch.__version__)==p['runtime']['torch_version'],'Adam runtime changed')
    return p

def install_guards():
    entries=[(mujoco,n,'physics.'+n) for n in ('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse')]
    entries += [(torch.nn.Module,'__init__','model_instances'),(torch.nn.Module,'_call_impl','model_forwards'),
        (torch.optim.Adam,'__init__','Adam_instances'),(torch.optim.Adam,'step','Adam.step'),
        (torch.optim.Optimizer,'step','Optimizer.step'),(torch.autograd,'grad','autograd.grad'),
        (torch.Tensor,'backward','Tensor.backward')]
    counts={label:0 for _,_,label in entries};old=[]
    def blocked(label):
        def call(*a,**kw):counts[label]+=1;raise RuntimeError('Saved-state diagnostic forbids '+label)
        return call
    for obj,name,label in entries:
        old.append((obj,name,getattr(obj,name)));setattr(obj,name,blocked(label))
    return counts,old

def storage(out,p):
    b=sum(f.stat().st_size for f in out.rglob('*') if f.is_file());free=shutil.disk_usage(ROOT).free
    require(b<=p['maximum_run_bytes'] and free>=p['minimum_free_reserve_bytes'],'Storage budget exceeded')
    return {'run_bytes_before_index':b,'free_bytes':free}

def run(protocol_path,expected):
    start=perf_counter();p=validate(protocol_path,expected);out=ROOT/p['run_directory']
    require(not out.exists(),'Never overwrite a diagnostic')
    require(shutil.disk_usage(ROOT).free>=p['minimum_free_reserve_bytes']+p['maximum_run_bytes']+p['document_reserve_bytes'],'Insufficient reserve')
    out.mkdir(parents=True);save(out/'registration.json',p)
    counts,originals=install_guards();rng=torch.get_rng_state().clone();torch.set_num_threads(1)
    ledger={'snapshots':0,'probes':0,'virtual_directions':0}
    try:
        previous=read(PRIOR/'manifest.json');probe_rows=read(PRIOR/'aggregate.json')['rows']
        require(previous['model_snapshots']==p['snapshots'],'Snapshot list changed')
        history=summarize_histories(PARENT/'arms/C0/training_trace.npz',PARENT/'arms/R1/training_trace.npz',PRIOR/'aggregate.json')
        save(out/'history_summary.json',history)
        manifest={'schema':'p4-adam-probe-manifest-v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'protocol_sha256':expected,'gradient_run':str(PRIOR.relative_to(ROOT)),
            'gradient_run_index_sha256':sha(PRIOR/'artifact_hashes.json'),
            'model_snapshots':p['snapshots'],'objective_order':p['objective_order'],'mode_order':p['mode_order'],
            'runtime':p['runtime'],'history_summary_sha256':sha(out/'history_summary.json'),
            'scope':'Prospective addends for frozen weights and fixed saved batches, not historical next-update replay or finite loss evaluation.'}
        save(out/'manifest.json',manifest);aggregate=[];source_checks=[]
        for spec in p['snapshots']:
            checkpoint=ROOT/spec['checkpoint_path']
            weights=torch.load(checkpoint/'weights.pt',map_location='cpu',weights_only=True)
            state=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=True)
            opt=state['optimizer'];wb=tree_hash(weights);ob=tree_hash(opt)
            require(wb==spec['model_tree_hash'],'Weight identity differs')
            require(all(float(v['step'])==1300+spec['additional_updates'] for v in opt['state'].values()),'Saved Adam age differs')
            rows=[r for r in probe_rows if r['snapshot_id']==spec['snapshot_id']]
            require([r['probe_name'] for r in rows]==p['probe_names'],'Fixed probe order differs')
            for row in rows:
                source=PRIOR/row['summary_path'];record=read(source);order=record['parameter_order']
                require([v['name'] for v in order]==list(weights) and all(list(weights[v['name']].shape)==v['shape'] for v in order),'Weight/gradient coordinate mapping differs')
                with np.load(source.parent/'gradients.npz',allow_pickle=False) as z:
                    raw=z['gradients'].copy();base=z['base'].copy()
                rb,bb=raw.copy(),base.copy()
                result=virtual_directions(raw,base,order,opt)
                require(np.array_equal(raw,rb) and np.array_equal(base,bb),'Saved gradient input mutated')
                require(result['directions'].shape==(6,19338) and result['directions'].dtype==np.float32
                    and result['clipped_gradients'].shape==(2,19338) and result['clipped_gradients'].dtype==np.float32,'Unregistered output geometry')
                folder=out/'probes'/spec['snapshot_id']/row['probe_name'];folder.mkdir(parents=True)
                np.savez_compressed(folder/'directions.npz',directions=result['directions'],clipped_gradients=result['clipped_gradients'])
                summary={**result['summary'],'snapshot_id':spec['snapshot_id'],'probe_name':row['probe_name'],
                    'source_summary_path':str(source.relative_to(ROOT)),'source_summary_sha256':sha(source),
                    'source_gradients_sha256':sha(source.parent/'gradients.npz'),'source_optimizer_tree_hash':ob,
                    'source_weights_tree_hash':wb,'no_weights_or_moments_changed':tree_hash(weights)==wb and tree_hash(opt)==ob}
                require(summary['no_weights_or_moments_changed'],'Frozen tensors mutated')
                save(folder/'summary.json',summary)
                aggregate.append({'snapshot_id':spec['snapshot_id'],'probe_name':row['probe_name'],
                    'summary_path':str((folder/'summary.json').relative_to(out)), 'summary':summary})
                ledger['probes']+=1;ledger['virtual_directions']+=6;storage(out,p)
            source_checks.append({'snapshot_id':spec['snapshot_id'],'weights_unchanged':tree_hash(weights)==wb,
                'optimizer_unchanged':tree_hash(opt)==ob,'weight_tree_hash':wb,'optimizer_tree_hash':ob})
            ledger['snapshots']+=1
            print(json.dumps({'snapshot_completed':spec['snapshot_id'],'virtual_directions':ledger['virtual_directions']}),flush=True)
        require(ledger=={'snapshots':7,'probes':28,'virtual_directions':168},'Registered count differs')
        require(not any(counts.values()) and torch.equal(rng,torch.get_rng_state()),'Forbidden execution or RNG mutation')
        validate(protocol_path,expected)
        save(out/'aggregate.json',{'schema':'p4-adam-probe-aggregate-v1','rows':aggregate,'row_count':28,
            'scope':'All selected snapshot/probe appearances and all six independent addends retained; no independent-trial interpretation.'})
        result={'schema':'p4-adam-probe-summary-v1','execution_status':'PASS','analysis_status':'DESCRIPTIVE_COMPLETE',
            'protocol_sha256':expected,'manifest_sha256':sha(out/'manifest.json'),'counters':ledger,
            'forbidden_execution_counts':counts,'source_tensor_checks':source_checks,'global_rng_unchanged':True,
            'new_model_forwards':0,'new_derivative_traversals':0,'new_optimizer_updates':0,'new_physics_steps':0,
            'new_checkpoint_writes':0,'new_data':False,'normalizer_refit':False,'final_test_used':False,
            'finite_step_loss_evaluated':False,'historical_trajectory_replayed':False,'model_selection':'NONE',
            'alignment_improvement':'NOT_EVALUATED','personal_understanding':'NOT_ASSESSED','P5_01':'HOLD',
            'wall_time_runner_only_s':perf_counter()-start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
            'resources':storage(out,p)}
        save(out/'summary.json',result)
        save(out/'artifact_hashes.json',{'sha256':{str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()}})
        return result
    except Exception as e:
        save(out/'failure.json',{'status':'FAIL','type':type(e).__name__,'error':str(e),'counters':ledger,'forbidden_execution_counts':counts,'traceback':traceback.format_exc()})
        raise
    finally:
        for obj,name,fn in reversed(originals):setattr(obj,name,fn)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--protocol-sha256',required=True)
    a=ap.parse_args();print(json.dumps(run(a.protocol,a.protocol_sha256),ensure_ascii=False),flush=True)
