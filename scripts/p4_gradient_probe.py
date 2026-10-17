"""Registered fixed-training-batch derivatives; no optimizer step or physics."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import resource
import shutil
import sys
from time import perf_counter
import tomllib
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import mujoco
import numpy as np
import torch
from push_align.config import ROOT
from push_align.data import digest
from push_align.data_intervention_training import read_json,require,tree_hash,data_hash
from push_align.gradient_probe_data import build_probe_batches,PROBE_NAMES
from push_align.gradient_probe import gradient_components
from push_align.learning_data import load_learning_data
from push_align.relative_loss_training import load_relative_checkpoint,relative_scale
from p3_train_common_mlp import load_common_checkpoint,validate_settings

PARENT=ROOT/'runs/p4-relative-loss-20261014-01'
LEARNING=ROOT/'runs/p3-02-20261009-01/learning_data'
GIB=1024**3

def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def validate_protocol(path,expected):
    require(digest(path)==expected,'Gradient protocol changed')
    p=read_json(path)
    require(p['run_id']=='p4-gradient-probe-20261015-01' and p['allow_optimizer_updates'] is False
        and p['loss_weights']==[1.,.1,.2,.02] and p['snapshot_count']==7 and p['probe_count']==4,
        'Unexpected objective/budget')
    for rel,sha in p['bindings_sha256'].items():
        require((ROOT/rel).is_file() and digest(ROOT/rel)==sha,'Frozen source changed: '+rel)
    return p

def guards():
    counts={name:0 for name in ['mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','Adam.step','Optimizer.step','Tensor.backward']}
    def blocked(name):
        def call(*args,**kwargs):
            counts[name]+=1;raise RuntimeError('Gradient diagnostic forbids '+name)
        return call
    for name in ['mj_step','mj_step1','mj_step2','mj_forward','mj_inverse']:
        setattr(mujoco,name,blocked(name))
    torch.optim.Adam.step=blocked('Adam.step')
    torch.optim.Optimizer.step=blocked('Optimizer.step')
    torch.Tensor.backward=blocked('Tensor.backward')
    return counts

def storage(out,protocol):
    size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file());free=shutil.disk_usage(ROOT).free
    require(size<=protocol['maximum_run_bytes'] and free>=2*GIB,'Output/reserve guard exceeded')
    return {'run_bytes':size,'free_bytes':free}

def load_snapshot(spec,normalizer,config,parent_manifest):
    checkpoint=ROOT/spec['checkpoint_path']
    if spec['snapshot_id']=='common_initial':
        common=read_json(ROOT/'runs/p3-04-20261009-01/manifest.json')
        settings=validate_settings(tomllib.loads((ROOT/'configs/p3_training.v1.toml').read_text()))
        b=load_common_checkpoint(checkpoint,normalizer,config,common['provenance'],settings,11,epoch=100,resume=False)
    else:
        b=load_relative_checkpoint(checkpoint,normalizer,config,expected_manifest_sha256=digest(PARENT/'manifest.json'),
            expected_branch=spec['branch'],expected_dataset_hash=parent_manifest['training_provenance']['D11']['data_hash'],
            expected_initial_checkpoint_hash=parent_manifest['initial_checkpoint']['checkpoint_hash'],resume=False)
    require(b.metadata['checkpoint_hash']==spec['checkpoint_hash'],'Snapshot checkpoint identity differs')
    require(b.optimizer is None,'Diagnostic should not restore a live optimizer')
    require(all(p.grad is None for p in b.model.parameters()),'Fresh model has accumulated gradients')
    return b.model

def run(protocol_path,expected):
    began=perf_counter();protocol=validate_protocol(protocol_path,expected)
    out=ROOT/protocol['run_directory'];require(not out.exists(),'Never overwrite a diagnostic')
    require(shutil.disk_usage(ROOT).free>=2*GIB+protocol['maximum_run_bytes']+protocol['document_reserve_bytes'],'Insufficient reserve')
    out.mkdir(parents=True);save(out/'registration.json',protocol)
    guard=guards();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    rng_initial=torch.get_rng_state().clone()
    counters={'forward_calls':0,'forward_rows':0,'derivative_traversals':0,'completed_probes':0}
    original_grad=torch.autograd.grad
    def counted_grad(*args,**kwargs):
        counters['derivative_traversals']+=1
        return original_grad(*args,**kwargs)
    torch.autograd.grad=counted_grad
    try:
        batches=build_probe_batches(ROOT)
        require(list(batches)==list(PROBE_NAMES),'Probe order changed')
        require({name:b['slots'] for name,b in batches.items()}==protocol['batch_selections'],'Registered train selection changed')
        for name,b in batches.items():
            require(b['counts']==protocol['batch_counts'][name],'Registered probe counts changed')
        save(out/'batch_manifest.json',batches)
        parent_manifest=read_json(PARENT/'manifest.json')
        with np.load(PARENT/'prepared_training.npz',allow_pickle=False) as archive:
            tensors={k:torch.from_numpy(archive[k].copy()) for k in ('states','actions','mask')}
        require(data_hash(*(tensors[k] for k in ('states','actions','mask')))==parent_manifest['training_provenance']['D11']['data_hash'],
            'D11 tensor identity changed')
        probe_arrays={}
        for name,batch in batches.items():
            slots=np.asarray(batch['slots'],dtype=np.int64);probe_arrays[name+'_slots']=slots
            for key in ('states','actions','mask'):probe_arrays[name+'_'+key]=tensors[key][slots].numpy().copy()
        np.savez_compressed(out/'probe_batches.npz',**probe_arrays)
        nm=read_json(LEARNING/'normalizer/normalizer.json')['normalizer']
        prepared,normalizer=load_learning_data(LEARNING,nm['schema'],nm['dataset_identity'],nm['normalizer_hash'])
        config=prepared.metadata['contract']['resolved_config']
        require(config==parent_manifest['physical_config'] and normalizer.metadata['normalizer_hash']==parent_manifest['normalizer_hash'],
            'Model config/normalizer changed')
        model_specs=[]
        for spec in protocol['snapshots']:
            checkpoint=ROOT/spec['checkpoint_path']
            meta=read_json(checkpoint/'checkpoint.json')
            require(meta['checkpoint_hash']==spec['checkpoint_hash'],'Registered snapshot metadata changed')
            model_specs.append({**spec,'model_tree_hash':tree_hash(torch.load(checkpoint/'weights.pt',map_location='cpu',weights_only=True)),
                'checkpoint_json_sha256':digest(checkpoint/'checkpoint.json'),
                'files_sha256':{name:digest(checkpoint/name) for name in ('weights.pt','training_state.pt')}})
        manifest={'schema':'p4-gradient-probe-manifest-v1','protocol_sha256':expected,'run_id':protocol['run_id'],
            'created_at_utc':datetime.now(timezone.utc).isoformat(),'physical_config':config,
            'normalizer_hash':normalizer.metadata['normalizer_hash'],'dataset_hash':parent_manifest['training_provenance']['D11']['data_hash'],
            'batch_manifest_sha256':digest(out/'batch_manifest.json'),'probe_batches_sha256':digest(out/'probe_batches.npz'),
            'model_snapshots':model_specs,'parameter_count':19338,'torch_version':str(torch.__version__),
            'numpy_version':np.__version__,'python_version':platform.python_version(),'torch_threads':1,
            'scope':'Same diagnostic objectives at each frozen snapshot; not historical update replay or actual Adam steps',
            'optimizer_updates':0,'new_physics_steps':0,'final_test_used':False,'P5_01':'HOLD'}
        save(out/'manifest.json',manifest)
        aggregate=[]
        for spec in model_specs:
            model=load_snapshot(spec,normalizer,config,parent_manifest)
            require(sum(p.numel() for p in model.parameters())==19338,'Unexpected parameter count')
            snapshot_identity=tree_hash(model.state_dict())
            require(snapshot_identity==spec['model_tree_hash'],'Loaded weights differ from bound tensor identity')
            for name,batch in batches.items():
                folder=out/'probes'/spec['snapshot_id']/name;folder.mkdir(parents=True)
                p=np.asarray(batch['slots'],dtype=np.int64)
                states,actions,mask=(tensors[k][p] for k in ('states','actions','mask'))
                before=tree_hash(model.state_dict());rng=torch.get_rng_state().clone();mode=model.training
                before_calls={k:counters[k] for k in ('forward_calls','forward_rows','derivative_traversals')}
                def count_forward(_model,inputs):
                    counters['forward_calls']+=1;counters['forward_rows']+=int(inputs[0].shape[0])
                handle=model.register_forward_pre_hook(count_forward)
                try:
                    result=gradient_components(model,normalizer,states,actions,mask)
                finally:
                    handle.remove()
                require(counters['forward_calls']-before_calls['forward_calls']==result['forward_calls']
                    and counters['forward_rows']-before_calls['forward_rows']==result['forward_total_rows']
                    and counters['derivative_traversals']-before_calls['derivative_traversals']==result['derivative_traversals'],
                    'External call counter and component ledger differ')
                after=tree_hash(model.state_dict())
                require(before==after==snapshot_identity and torch.equal(rng,torch.get_rng_state())
                    and model.training==mode and all(v.grad is None for v in model.parameters()),'Probe mutated model/RNG/gradient buffers')
                raw=result.pop('gradients');derived=result.pop('derived_gradients')
                errors={k:result.pop(k) for k in ('teacher_xy_errors','free_xy_errors')}
                require(raw.shape==(5,19338) and raw.dtype==np.float32,'Gradient vector shape/dtype differs')
                np.savez_compressed(folder/'gradients.npz',gradients=raw,**derived)
                np.savez_compressed(folder/'loss_residuals.npz',**errors,relative_scale_m=relative_scale(normalizer).numpy())
                checks={'weights_unchanged':before==after,'rng_unchanged':torch.equal(rng,torch.get_rng_state()),
                    'grad_buffers_absent':all(v.grad is None for v in model.parameters()),'model_mode_unchanged':model.training==mode}
                record={**result,'snapshot_id':spec['snapshot_id'],'probe_name':name,
                    'trained_objective':spec['trained_objective'],'diagnostic_objective':protocol['diagnostic_objective'],
                    'before_weights_hash':before,'after_weights_hash':after,'model_mode_before':mode,'model_mode_after':model.training,
                    'checks':checks,'batch_count':len(p),'interpretation_scope':'Local derivative compatibility on fixed train data, not Adam or historical trajectory causality'}
                save(folder/'summary.json',record)
                counters['completed_probes']+=1
                aggregate.append({'snapshot_id':spec['snapshot_id'],'probe_name':name,'trained_objective':spec['trained_objective'],
                    'batch_count':len(p),'summary_path':(folder/'summary.json').relative_to(out).as_posix(),
                    'loss_values':result['loss_values'],'vector_summary':result['vector_summary']})
                storage(out,protocol)
            require(tree_hash(model.state_dict())==snapshot_identity,'Snapshot weights mutated across probes')
            print(json.dumps({'snapshot_completed':spec['snapshot_id'],'probes_completed':counters['completed_probes']}),flush=True)
        require(counters==protocol['expected_counters'],'Registered derivative/forward budget differs')
        require(not any(guard.values()) and torch.equal(rng_initial,torch.get_rng_state()),'Forbidden operation or global RNG mutation')
        validate_protocol(protocol_path,expected)
        require(digest(out/'batch_manifest.json')==manifest['batch_manifest_sha256'] and digest(out/'probe_batches.npz')==manifest['probe_batches_sha256'],
            'Probe inputs changed')
        save(out/'aggregate.json',{'schema':'p4-gradient-probe-aggregate-v1','rows':aggregate,'row_count':len(aggregate),
            'scope':'All28 snapshot/probe appearances retained; no CI, optimizer step, model selection or efficacy verdict'})
        result={'schema':'p4-gradient-probe-result-v1','execution_status':'PASS','analysis_status':'DESCRIPTIVE_COMPLETE',
            'counters':counters,'forbidden_execution_counts':guard,'model_weights_unchanged':True,'gradient_buffers_unchanged':True,
            'global_rng_unchanged':True,'model_mode_unchanged':True,'new_optimizer_updates':0,'new_physics_steps':0,
            'new_checkpoint_writes':0,'normalizer_refit':False,'development_or_validation_probe_used':False,'final_test_used':False,
            'new_closed_loop_execution':False,'model_selection':'NONE','alignment_improvement':'NOT_EVALUATED',
            'P5_01':'HOLD','personal_understanding':'NOT_ASSESSED','protocol_sha256':expected,
            'manifest_sha256':digest(out/'manifest.json'),'wall_time_s':perf_counter()-began,'resources':storage(out,protocol),
            'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)}
        save(out/'summary.json',result)
        save(out/'artifact_hashes.json',{'sha256':{p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*')) if p.is_file()}})
        return result
    except Exception as error:
        save(out/'failure.json',{'execution_status':'FAIL','error':str(error),'traceback':traceback.format_exc(),
            'counters':counters,'forbidden_execution_counts':guard,'automatic_retry':False})
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--protocol-sha256',required=True);args=parser.parse_args()
    print(json.dumps(run(args.protocol.resolve(),args.protocol_sha256),ensure_ascii=False))
