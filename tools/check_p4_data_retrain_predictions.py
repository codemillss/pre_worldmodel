"""Independently recompute saved primary estimands and family contrasts; no model execution."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np

ARMS=('R','D00','D10','D01','D11')
ENDPOINTS={'object_xy_m':slice(0,2),'pusher_xy_m':slice(6,8)}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def check_run(folder,output):
    checks=[]
    def check(label,ok):
        checks.append({'name':label,'pass':bool(ok)})
        if not ok:raise ValueError(label)
    index=read(folder/'artifact_hashes.json')['sha256']
    for rel,h in index.items():check('immutable:'+rel,sha(folder/rel)==h)
    all_primary={}; truths=None
    for arm in ARMS:
        result=read(folder/arm/'development.json')
        with np.load(folder/arm/'development_traces.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
        actual=a['actual_states']; hidx=a['head_indices']; sidx=a['stop_indices']
        check(arm+':48_complete_actuals',actual.shape==(48,21,10))
        if truths is None:truths=(actual,a['actual_actions'],a['actual_regimes'])
        else:check(arm+':same_actuals',all(np.array_equal(x,y) for x,y in zip(truths,(actual,a['actual_actions'],a['actual_regimes']))))
        ids=result['core_H10_free']['identities']; families=result['families']
        check(arm+':12families',len(families)==12 and len(set(families))==12)
        all_primary[arm]={}
        for phase in ('Y_primary','S_primary'):
            values={}
            for fam in families:
                if phase=='Y_primary':
                    j=[n for n,k in enumerate(hidx) if ids[k]['family_id']==fam and ids[k]['y_condition']=='Y1']
                    check(arm+fam+':one_deduplicated_Y1_head',len(j)==1)
                    positions=np.arange(j[0]*5,j[0]*5+5)
                    predicted=a['head_states'][positions,-1]; truth=actual[hidx[j[0]],1:6];valid=a['head_valid'][positions]
                else:
                    positions=np.array([n for n,k in enumerate(sidx) if ids[k]['family_id']==fam])
                    check(arm+fam+':two_Y_stop',len(positions)==2 and {ids[sidx[j]]['y_condition'] for j in positions}=={'Y0','Y1'})
                    predicted=a['stop_states'][positions,-1];truth=actual[sidx[positions],15];valid=a['stop_valid'][positions]
                values[fam]={}
                for key,sl in ENDPOINTS.items():
                    value=float(np.hypot(*(predicted[:,sl]-truth[:,sl]).T).mean()) if valid.all() else None
                    recorded=result['primary'][phase][fam][key]
                    check(arm+fam+phase+key,recorded is None if value is None else recorded is not None and np.isclose(value,recorded,rtol=2e-15,atol=1e-18))
                    values[fam][key]=value
            all_primary[arm][phase]=values
        for name,prefix in [('Y_heads_teacher_conditioned','head'),('S1_free_stop','stop'),('S1_teacher_conditioned_stop','stop_local'),('core_H10_free','full_H10')]:
            r=result[name]; valid=a[prefix+'_valid']; failure=a[prefix+'_failure_steps']
            check(arm+':'+name+':denominators',r['attempted']==len(valid) and r['valid']==int(valid.sum()) and r['invalid']==int((~valid).sum()))
            check(arm+':'+name+':failure_mask',np.array_equal(valid,failure==-1))
    comparison=read(folder/'comparison.json')
    families=comparison['family_ids']
    draws=np.random.default_rng(2026101304).integers(0,12,size=(10000,12),dtype=np.int64)
    with np.load(folder/'bootstrap_family_draws.npz',allow_pickle=False) as f:check('registered_draws',np.array_equal(draws,f['draws']))
    formulas={'D10-D00':{'D10':1,'D00':-1},'D11-D01':{'D11':1,'D01':-1},'D01-D00':{'D01':1,'D00':-1},'D11-D10':{'D11':1,'D10':-1},'interaction':{'D11':1,'D10':-1,'D01':-1,'D00':1},**{a+'-R':{a:1,'R':-1} for a in ARMS[1:]}}
    independent={}
    for phase in ('Y_primary','S_primary'):
        independent[phase]={}
        for key in ENDPOINTS:
            independent[phase][key]={}
            for name,coef in formulas.items():
                values=[]
                for fam in families:
                    v=[all_primary[a][phase][fam][key] for a in coef]
                    values.append(None if any(x is None for x in v) else sum(w*x for w,x in zip(coef.values(),v)))
                rec=comparison['contrasts'][phase][key][name]
                check(phase+key+name+':retained12',rec['attempted_families']==12 and rec['family_ids']==families)
                for i,(v,r) in enumerate(zip(values,rec['family_differences'])):check(phase+key+name+str(i),r is None if v is None else r is not None and np.isclose(v,r,rtol=2e-13,atol=1e-17))
                if any(v is None for v in values):
                    check(phase+key+name+':held',rec['status']=='HOLD_INVALID_PRIMARY' and rec['mean_difference'] is None)
                    independent[phase][key][name]=None
                else:
                    x=np.asarray(values);ci=np.quantile(x[draws].mean(1),[.025,.975])
                    check(phase+key+name+':mean_CI',np.isclose(x.mean(),rec['mean_difference'],rtol=2e-13,atol=1e-17) and np.allclose(ci,rec['bootstrap_percentile_95'],rtol=2e-13,atol=1e-17))
                    check(phase+key+name+':counts',(int((x<0).sum()),int((x>0).sum()),int((x==0).sum()))==(rec['improved'],rec['worsened'],rec['exact_float64_ties']))
                    independent[phase][key][name]={'mean':float(x.mean()),'ci':ci.tolist()}
    for factor,phase,names in [('Y','Y_primary',['D10-D00','D11-D01']),('S','S_primary',['D01-D00','D11-D10'])]:
        vals=[independent[phase][key][n] for key in ENDPOINTS for n in names]
        if any(v is None for v in vals):expected='HOLD_INVALID_PRIMARY'
        elif all(v['ci'][1]<0 for v in vals):expected='DIRECTIONAL_SUPPORT'
        elif any(v['mean']<0 for v in vals) and any(v['mean']>0 for v in vals):expected='MIXED'
        else:expected='INCONCLUSIVE'
        check(factor+':decision',comparison['directional_support'][factor]['status']==expected)
    report={'status':'PASS','checks':checks,'check_count':len(checks),'prediction_summary_sha256':sha(folder/'summary.json'),'comparison_sha256':sha(folder/'comparison.json'),'source_sha256':sha(Path(__file__)),'scope':'Independent NumPy arithmetic from saved prediction/truth, actual sample identity, complete denominators and family bootstrap. Not a rerun of learned inference or physical validity solver.','model_forwards':0,'physics_steps':0,'learning_updates':0}
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'PASS','checks':len(checks)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('output',type=Path);a=p.parse_args();check_run(a.folder,a.output)
