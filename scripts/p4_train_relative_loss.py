"""Registered matched D11 objective contrast; one fixed auxiliary coefficient."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
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
from push_align.data_intervention_training import data_hash,read_json,require,tree_hash
from push_align.fork_sampler import WindowSampler
from push_align.learning_data import load_learning_data
from push_align.relative_loss_training import (BRANCHES,FORMULA,relative_scale,
    relative_sequence_update,save_relative_checkpoint,load_relative_checkpoint)
from p3_train_common_mlp import validate_settings
from p4_train_forks import restore_common,start_identity

OLD=ROOT/'runs/p4-data-retrain-20261013-02'
HIST=OLD/'arms/D11'
COMMON=ROOT/'runs/p3-04-20261009-01'
LEARNING=ROOT/'runs/p3-02-20261009-01/learning_data'
GIB=1024**3

def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def verify_registration(path,expected):
    require(digest(path)==expected,'Registered protocol hash differs')
    reg=read_json(path)
    for rel,sha in reg['bindings_sha256'].items():
        require((ROOT/rel).is_file() and digest(ROOT/rel)==sha,'Frozen binding changed: '+rel)
    return reg

def storage(out,reg):
    size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    free=shutil.disk_usage(ROOT).free
    require(size<=reg['training_maximum_bytes'] and free>=2*GIB,'Storage/reserve guard exceeded')
    return {'run_bytes':size,'free_bytes':free}

def forbid_engine():
    names=('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse')
    counts=dict.fromkeys(names,0)
    def wrapper(name):
        def blocked(*args,**kwargs):
            counts[name]+=1
            raise RuntimeError('New physics prohibited: '+name)
        return blocked
    for name in names: setattr(mujoco,name,wrapper(name))
    return counts

def compare_checkpoint(path,model,optimizer,sampler):
    meta=read_json(path/'checkpoint.json')
    for name,sha in meta['files'].items(): require(digest(path/name)==sha,'Historical checkpoint changed')
    old_weights=torch.load(path/'weights.pt',map_location='cpu',weights_only=True)
    old_state=torch.load(path/'training_state.pt',map_location='cpu',weights_only=True)
    checks={'weights':tree_hash(model.state_dict())==tree_hash(old_weights),
        'Adam':tree_hash(optimizer.state_dict())==tree_hash(old_state['optimizer']),
        'sampler':tree_hash(sampler.state_dict())==tree_hash(old_state['sampler']),
        'Torch_RNG':torch.equal(torch.get_rng_state(),old_state['torch_rng_state'])}
    require(all(checks.values()),'Control does not exactly reproduce D11 checkpoint')
    return checks

def resume_probe(path,normalizer,config,manifest_sha,branch,tensors,counters):
    rng=torch.get_rng_state().clone()
    states=[]
    with torch.random.fork_rng(devices=[]):
        for _ in range(2):
            bundle=load_relative_checkpoint(path,normalizer,config,
                expected_manifest_sha256=manifest_sha,expected_branch=branch,resume=True)
            torch.set_rng_state(bundle.torch_rng_state)
            records=[]
            for _ in range(2):
                p=bundle.sampler.next_batch()
                counters['disposable_attempts']+=1
                result=relative_sequence_update(bundle.model,bundle.optimizer,normalizer,
                    *(t[p] for t in tensors),relative_weight=BRANCHES[branch])
                counters['disposable_updates']+=1
                records.append({k:v for k,v in result.items() if k not in ('teacher_xy_errors','free_xy_errors')})
            states.append({'weights':tree_hash(bundle.model.state_dict()),'Adam':tree_hash(bundle.optimizer.state_dict()),
                'sampler':tree_hash(bundle.sampler.state_dict()),'rng':tree_hash(torch.get_rng_state()),'records':records})
    require(states[0]==states[1] and torch.equal(rng,torch.get_rng_state()),'Resume proof failed or mutated official RNG')
    return {'status':'PASS','discarded_updates':4,'copies_after_two_updates_exact':True,
        'official_rng_unchanged':True,'copy_identity':states[0]}

def train(branch,tensors,entry,normalizer,config,common,settings,historical_trace,manifest_sha,out,reg,counters):
    folder=out/'arms'/branch;folder.mkdir(parents=True)
    bundle=restore_common(entry,normalizer,config,common,settings)
    torch.set_rng_state(bundle.torch_rng_state)
    sampler=WindowSampler(2960,64,bundle.shuffle_generator,max_updates=1000,horizon=5)
    start=start_identity(bundle,sampler)
    require(start==read_json(HIST/'start_identity.json'),'D11 initial state changed')
    save(folder/'start_identity.json',start)
    losses=[];positions=[];offsets=[0];hashes=[];errors_teacher=[];errors_free=[];checkpoints=[];times=[]
    tick=perf_counter()
    try:
        for update in range(1,1001):
            p=sampler.next_batch();a,b=historical_trace['batch_offsets'][update-1:update+1]
            require(np.array_equal(p.numpy(),historical_trace['window_positions'][a:b]),'Historical schedule changed')
            current=tuple(t[p] for t in tensors);h=data_hash(*current)
            require(h==historical_trace['batch_data_hashes'][update-1],'D11 batch data changed')
            counters['formal_attempts']+=1;t=perf_counter()
            d=relative_sequence_update(bundle.model,bundle.optimizer,normalizer,*current,
                relative_weight=BRANCHES[branch],rollout_weight=.1,clip_norm=1.)
            counters['formal_updates']+=1;times.append(perf_counter()-t)
            require(d['valid_transitions']==len(p)*5 and d['scalar_terms']==len(p)*50,'Base denominator changed')
            values=[d['one_step_loss'],d['rollout_loss'],d['relative_teacher_loss'] or 0.,
                d['relative_rollout_loss'] or 0.,d['loss_before_update'],d['gradient_norm_before_clip']]
            if branch=='C0':
                require(np.array_equal(np.array(values)[[0,1,4,5]],historical_trace['losses'][update-1]),
                    'Control history mismatch at '+str(update))
            else:
                require(d['relative_scalar_terms']==len(p)*10,'Relative denominator changed')
                errors_teacher.append(d['teacher_xy_errors']);errors_free.append(d['free_xy_errors'])
            losses.append(values);positions.append(p.numpy().copy());offsets.append(offsets[-1]+len(p));hashes.append(h)
            if update%100==0:
                path=folder/'checkpoints'/f'update-{update:04d}'
                meta=save_relative_checkpoint(path,bundle.model,bundle.optimizer,normalizer,config,
                    branch=branch,sampler=sampler,manifest_sha256=manifest_sha,dataset_hash=data_hash(*tensors),
                    initial_checkpoint_hash=entry['checkpoint_hash'])
                exact=compare_checkpoint(HIST/'checkpoints'/f'update-{update:04d}',bundle.model,bundle.optimizer,sampler) if branch=='C0' else None
                if update==100:
                    save(folder/'resume-0100.json',resume_probe(path,normalizer,config,manifest_sha,branch,tensors,counters))
                checkpoints.append({'update':update,'checkpoint':path.relative_to(out).as_posix(),
                    'checkpoint_hash':meta['checkpoint_hash'],'historical_exact':exact})
                storage(out,reg)
                print(json.dumps({'branch':branch,'update':update,'historical_exact':exact}),flush=True)
        require(sampler.window_occurrences==62992 and sampler.transition_occurrences==314960,'Exposure budget changed')
        require(sampler.updates==1000,'Formal update count changed')
        # Final reload validates the full continuation format without updating.
        loaded=load_relative_checkpoint(folder/'checkpoints/update-1000',normalizer,config,
            expected_manifest_sha256=manifest_sha,expected_branch=branch,expected_dataset_hash=data_hash(*tensors))
        require(tree_hash(loaded.model.state_dict())==tree_hash(bundle.model.state_dict()),'Final checkpoint reload differs')
        result={'branch':branch,'relative_weight':BRANCHES[branch],'status':'PASS','additional_updates':1000,
            'total_Adam_steps':2300,'dataset_hash':data_hash(*tensors),'start_identity':start,
            'window_occurrences':sampler.window_occurrences,'transition_occurrences':sampler.transition_occurrences,
            'final_checkpoint':checkpoints[-1]['checkpoint'],'final_checkpoint_hash':checkpoints[-1]['checkpoint_hash'],
            'checkpoint_selection':'final1000_only','checkpoints':checkpoints,
            'clipped_updates':sum(v[5]>1. for v in losses),'wall_seconds':perf_counter()-tick}
        save(folder/'summary.json',result)
        return result
    except Exception as error:
        torch.save({'weights':bundle.model.state_dict(),'optimizer':bundle.optimizer.state_dict(),
            'sampler':sampler.state_dict(),'rng':torch.get_rng_state()},folder/'failed_prefix.pt')
        save(folder/'failure.json',{'completed_updates':len(losses),'error':str(error),'counters':counters})
        raise
    finally:
        np.savez_compressed(folder/'training_trace.npz',window_positions=np.concatenate(positions) if positions else np.empty(0,dtype=np.int64),
            batch_offsets=np.array(offsets,dtype=np.int64),batch_data_hashes=np.array(hashes,dtype='U64'),
            losses=np.array(losses,dtype=np.float64),loss_columns=np.array(['one_step','rollout','relative_teacher','relative_free','total','gradient_norm']),
            update_wall_time_s=np.array(times,dtype=np.float64))
        if errors_teacher:
            np.savez_compressed(folder/'relative_residual_trace.npz',
                teacher_xy_errors=np.concatenate(errors_teacher),free_xy_errors=np.concatenate(errors_free),
                transition_offsets=np.array(offsets,dtype=np.int64)*5,
                relative_scale_m=relative_scale(normalizer).numpy())

def run(path,expected):
    start=perf_counter();reg=verify_registration(path,expected)
    require(reg['arms']=={'C0':0.,'R1':.2} and reg['training']['updates_per_arm']==1000,'Unregistered objectives/budget')
    require(reg['loss']['formula']==FORMULA and reg['loss']['rollout_weight']==.1
            and reg['loss']['relative_weight']==.2 and reg['loss']['relative_denominator']=='2N per teacher/free term'
            and reg['training']['formal_updates']==2000 and reg['training']['disposable_update_cap']==8,
            'Registered arithmetic or update cap differs')
    out=ROOT/reg['training_run'];require(not out.exists(),'Never overwrite a run')
    require(shutil.disk_usage(ROOT).free>=2*GIB+reg['training_maximum_bytes']+reg['prediction_evaluation_maximum_bytes'],'Full storage reserve unavailable')
    out.mkdir(parents=True);save(out/'registration.json',reg)
    counters=dict(formal_attempts=0,formal_updates=0,disposable_attempts=0,disposable_updates=0)
    engine=forbid_engine();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    try:
        old_manifest=read_json(OLD/'manifest.json')
        old_summary=read_json(OLD/'summary.json');require(old_summary['status']=='PASS','Old D11 did not complete')
        old_index=read_json(OLD/'training_artifact_hashes.json')['sha256']
        required=['manifest.json','dataset_bindings.json','prepared_training.npz']+[p.relative_to(OLD).as_posix() for p in HIST.rglob('*') if p.is_file()]
        for rel in required: require(digest(OLD/rel)==old_index[rel],'Historical D11 artifact changed: '+rel)
        nm=read_json(LEARNING/'normalizer/normalizer.json')['normalizer']
        prepared,normalizer=load_learning_data(LEARNING,nm['schema'],nm['dataset_identity'],nm['normalizer_hash'])
        config=prepared.metadata['contract']['resolved_config']
        require(config==old_manifest['physical_config'],'Physical config changed')
        require(reg['loss']['relative_scale_m']==relative_scale(normalizer).tolist(),'Registered fixedscale differs')
        with np.load(OLD/'prepared_training.npz',allow_pickle=False) as archive:
            tensors=tuple(torch.from_numpy(archive['D11_'+name].copy()) for name in ('states','actions','mask'))
        dataset_hash=data_hash(*tensors)
        require(dataset_hash==old_manifest['training_provenance']['arms']['D11']['data_hash'],'D11 training population changed')
        require(tensors[0].shape==(2960,6,10) and tensors[1].shape==(2960,5,2) and bool(tensors[2].all()),'D11 population contract changed')
        save(out/'dataset_bindings.json',{'data_hash':dataset_hash,'source':(OLD/'prepared_training.npz').relative_to(ROOT).as_posix(),
            'source_sha256':digest(OLD/'prepared_training.npz'),'original_training_provenance':old_manifest['training_provenance'],
            'development_episodes_loaded_for_training':0,'warmup_episodes_loaded_for_training':0,'normalizer_refit':False})
        np.savez_compressed(out/'prepared_training.npz',states=tensors[0].numpy(),actions=tensors[1].numpy(),mask=tensors[2].numpy())
        common=read_json(COMMON/'manifest.json');entry=old_manifest['initial_checkpoint']
        settings=validate_settings(tomllib.loads((ROOT/'configs/p3_training.v1.toml').read_text()))
        provenance={'arms':{branch:{'data_hash':dataset_hash} for branch in BRANCHES},'D11':old_manifest['training_provenance']['arms']['D11']}
        manifest={'schema':'p4-relative-loss-training-v1','run_id':reg['run_id'],
            'registered_before_training':datetime.now(timezone.utc).isoformat(),'registration_sha256':expected,
            'physical_config':config,'normalizer_hash':normalizer.metadata['normalizer_hash'],'initial_checkpoint':entry,
            'training_provenance':provenance,'dataset_bindings_sha256':digest(out/'dataset_bindings.json'),
            'prepared_training_sha256':digest(out/'prepared_training.npz'),'loss':reg['loss'],
            'formal_update_budget':2000,'disposable_update_cap':8,'historical_D11':HIST.relative_to(ROOT).as_posix(),
            'torch_version':str(torch.__version__),'torch_threads':1,'deterministic_algorithms':True,
            'final_test_used':False,'new_closed_loop_execution':False,'P5_01':'HOLD'}
        save(out/'manifest.json',manifest);manifest_sha=digest(out/'manifest.json')
        with np.load(HIST/'training_trace.npz',allow_pickle=False) as archive:
            historical_trace={k:archive[k].copy() for k in archive.files}
        arms=[train(branch,tensors,entry,normalizer,config,common,settings,historical_trace,manifest_sha,out,reg,counters) for branch in BRANCHES]
        verify_registration(path,expected)
        require(counters==dict(formal_attempts=2000,formal_updates=2000,disposable_attempts=8,disposable_updates=8),'Update accounting differs')
        require(not any(engine.values()),'Future engine call occurred')
        summary={'schema':'p4-relative-loss-training-result-v1','status':'PASS','arms':arms,'counters':counters,
            'future_engine_calls':engine,'manifest_sha256':manifest_sha,'wall_seconds':perf_counter()-start,
            'resources':storage(out,reg),'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
            'final_test_used':False,'new_closed_loop_execution':False,'task_success':'NOT_EVALUATED',
            'P5_01':'HOLD','personal_understanding':'NOT_ASSESSED','prediction_comparison':'NOT_RUN'}
        save(out/'summary.json',summary)
        save(out/'training_artifact_hashes.json',{'sha256':{p.relative_to(out).as_posix():digest(p) for p in sorted(out.rglob('*')) if p.is_file()}})
        return summary
    except Exception as error:
        save(out/'failure.json',{'status':'FAIL','error':str(error),'traceback':traceback.format_exc(),'counters':counters,'future_engine_calls':engine})
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--registration-sha256',required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.registration.resolve(),args.registration_sha256),ensure_ascii=False))
