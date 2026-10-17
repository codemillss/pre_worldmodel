"""Post-evaluation geometry diagnostic of saved invalid predictions; no model rerun."""
from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/p4-data-retrain-20261013-02'
EVAL=RUN/'prediction_evaluation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def gap(s,c):
 d=s[...,6:8]-s[...,:2];co=np.cos(s[...,2]);si=np.sin(s[...,2])
 local=np.stack([co*d[...,0]+si*d[...,1],-si*d[...,0]+co*d[...,1]],-1)
 q=np.abs(local)-np.array(c['object']['size_full_xyz_m'][:2])/2
 return np.hypot(np.maximum(q[...,0],0),np.maximum(q[...,1],0))+np.minimum(np.maximum(q[...,0],q[...,1]),0)-c['pusher']['radius_m']
def run():
 c=json.loads((RUN/'manifest.json').read_text())['physical_config'];events=[];arms={}
 for arm in ['R','D00','D10','D01','D11']:
  result=json.loads((EVAL/arm/'development.json').read_text())
  with np.load(EVAL/arm/'development_traces.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
  ids=result['core_H10_free']['identities'];actual=a['actual_states'];h=a['head_indices'];s=a['stop_indices'];arm_events=[]
  mappings={'head':[(int(i),int(k)) for i in h for k in range(5)],'stop':[(int(i),10) for i in s],'stop_local':[(int(i),int(k)) for i in s for k in range(10,15)],'full_H10':[(i,0) for i in range(48)]}
  for mode,lookup in mappings.items():
   for row in np.flatnonzero(~a[mode+'_valid']):
    step=int(a[mode+'_failure_steps'][row]);episode,start=lookup[row];pred=a[mode+'_states'][row,step];truth=actual[episode,start+step]
    event={'arm':arm,'mode':mode,'row':int(row),'episode':ids[episode],'failure_step':step,'actual_core_boundary':start+step,'reason':str(a[mode+'_failure_reasons'][row]),'predicted_gap_mm':float(gap(pred,c)*1000),'actual_gap_mm':float(gap(truth,c)*1000),'object_xy_error_mm':float(np.linalg.norm(pred[:2]-truth[:2])*1000),'pusher_xy_error_mm':float(np.linalg.norm(pred[6:8]-truth[6:8])*1000),'actual_transition_regime':str(a['actual_regimes'][episode,max(0,start+step-1)])}
    arm_events.append(event)
  arms[arm]={'invalid_by_mode':dict(Counter(e['mode'] for e in arm_events)),'invalid_reasons':dict(Counter(e['reason'] for e in arm_events)),'invalid_initial_state_count':sum(e['failure_step']==0 for e in arm_events),'actual_boundary_minimum_gap_mm':float(gap(actual,c).min()*1000),'primary_complete_families':{k:sum(all(v is not None for v in x.values()) for x in result['primary'][k].values()) for k in ['Y_primary','S_primary']}}
  events.extend(arm_events)
 assert all(e['failure_step']>0 and e['predicted_gap_mm']<-1 and e['actual_gap_mm']>=-1 for e in events)
 report={'status':'DESCRIPTIVE_ONLY','source_sha256':sha(Path(__file__)),'evaluation_index_sha256':sha(EVAL/'artifact_hashes.json'),'prediction_summary_sha256':sha(EVAL/'summary.json'),'predicted_penetration_tolerance_mm':1.,'arms':arms,'events':events,'event_count':len(events),'all_invalids_are_future_deep_penetration':True,'all_corresponding_actual_boundaries_within_existing_penetration_budget':True,'scope':'Post hoc decomposition of stored invalid predictions. No rerun, threshold change, imputation, primary replacement or extra optimization. Repeated related trajectories are not independent failure cases.','model_forwards':0,'physics_steps':0,'learning_updates':0}
 p=ROOT/'reports/p4_data_retrain_failure_analysis.json';p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'events':len(events),'arms':arms},ensure_ascii=False))
if __name__=='__main__':run()
