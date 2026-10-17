"""Registered stored-trace diagnosis with no learner or engine execution."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, importlib.util, inspect, json, shutil, sys, time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def dump(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def folder_bytes(path):return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
def module(path,label):
 spec=importlib.util.spec_from_file_location(label,path);m=importlib.util.module_from_spec(spec);sys.modules[label]=m;spec.loader.exec_module(m);return m

def main(folder):
 import numpy as np
 import torch
 import mujoco
 folder=Path(folder);registration=read(folder/'registration.json');expected=registration['bindings_sha256']
 for rel,h in expected.items():
  if sha(ROOT/rel)!=h:raise ValueError('Registered input/source changed: '+rel)
 protocol=read(ROOT/registration['protocol']);budget=protocol['budget']
 if shutil.disk_usage(ROOT).free-budget['run_bytes_maximum']<budget['minimum_free_reserve_bytes']:raise RuntimeError('Free reserve below registered bound')
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);rng_before=torch.get_rng_state().clone()
 counters={k:0 for k in ['mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','learned_module_call','torch_load','backward']}
 replacements=[]
 def deny(name):
  def call(*args,**kwargs):counters[name]+=1;raise RuntimeError('Forbidden in stored diagnosis: '+name)
  return call
 for name in ['mj_step','mj_step1','mj_step2','mj_forward','mj_inverse']:
  replacements.append((mujoco,name,getattr(mujoco,name)));setattr(mujoco,name,deny(name))
 for obj,name,label in [(torch.nn.Module,'__call__','learned_module_call'),(torch,'load','torch_load'),(torch.Tensor,'backward','backward')]:
  replacements.append((obj,name,getattr(obj,name)));setattr(obj,name,deny(label))
 start=time.perf_counter()
 try:
  jobs=[('tools/diagnose_p4_contact_geometry.py','contact_geometry.json'),('tools/diagnose_p4_stop_error_propagation.py','stop_propagation.json'),('tools/diagnose_p4_contact_input_support.py','input_support.json')]
  for n,(source,out) in enumerate(jobs):
   m=module(ROOT/source,'stored_mechanism_'+str(n));kwargs={}
   if 'expected_sources' in inspect.signature(m.run).parameters:kwargs['expected_sources']=expected
   m.run(folder/out,**kwargs)
   if folder_bytes(folder)>budget['run_bytes_maximum'] or shutil.disk_usage(ROOT).free<budget['minimum_free_reserve_bytes']:raise RuntimeError('Registered resource reserve exceeded')
  for rel,h in expected.items():
   if sha(ROOT/rel)!=h:raise ValueError('Original changed during diagnosis: '+rel)
  if any(counters.values()) or not torch.equal(rng_before,torch.get_rng_state()):raise ValueError('Execution/RNG boundary changed')
  result={'schema':'p4-stored-error-mechanism-run-v1','execution_status':'PASS','analysis_status':'DESCRIPTIVE_COMPLETE',
   'mechanism_status':'PARTIALLY_DISTINGUISHED_NOT_CAUSALLY_IDENTIFIED','registered_before_diagnostic_execution':True,
   'registration_sha256':sha(folder/'registration.json'),'protocol_sha256':sha(ROOT/registration['protocol']),
   'immutable_bindings_verified':len(expected),'cases':protocol['denominators'],
   'execution_guard_counts':counters,'torch_rng_unchanged':True,'new_learned_forwards':0,'learning_updates':0,'new_physical_intervals':0,
   'new_closed_loop_execution':False,'final_test_used':False,'primary_effect_status':'HOLD_INVALID_PRIMARY','P5_01':'HOLD',
   'personal_understanding':'NOT_ASSESSED','wall_seconds':time.perf_counter()-start,
   'resources':{'free_bytes':shutil.disk_usage(ROOT).free,'run_bytes_before_summary':folder_bytes(folder)},
   'scope':'Selected existing development failures only. Component recombinations are geometric diagnostics, not new model/physical intervention or primary metric replacement.'}
  dump(folder/'summary.json',result)
  dump(folder/'artifact_hashes.json',{'sha256':{str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'}})
  print(json.dumps({'execution_status':'PASS','run':str(folder.relative_to(ROOT)),'run_bytes':folder_bytes(folder),'wall_seconds':result['wall_seconds']}))
 except Exception as error:
  dump(folder/'failure.json',{'status':'FAIL','error':str(error),'type':type(error).__name__,'guard_counts':counters,'wall_seconds':time.perf_counter()-start,'registered_before_execution':True,'learning_updates':0})
  raise
 finally:
  for obj,name,original in reversed(replacements):setattr(obj,name,original)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
