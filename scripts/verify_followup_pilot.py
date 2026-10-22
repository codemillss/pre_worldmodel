#!/usr/bin/env python3
"""Verify selected published pilot evidence with NumPy; no simulator/model calls."""
from __future__ import annotations
import argparse,hashlib,json,math
from itertools import product
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/followup/20261022-d1-d2'


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify():
    manifest=json.loads((BASE/'manifest.json').read_text())
    for f in manifest['files']:
        p=BASE/f['path'];assert p.stat().st_size==f['bytes'] and digest(p)==f['sha256'],str(p)
    x=json.loads((BASE/'summary.json').read_text());rows=x['rows']
    expected=set(product(['N-T','F-O0'],range(16),['A','B'],range(3)))
    assert len(rows)==192 and {(r['state'],r['candidate_id'],r['panel'],r['repeat']) for r in rows}==expected
    assert sum(r['intervals'] for r in rows)==x['actual_physics_intervals']==1242
    stats={}
    for state in ['N-T','F-O0']:
        actual_valid={};summary={}
        for panel in ['A','B']:
            group=[r for r in rows if r['state']==state and r['panel']==panel]
            for index in range(16):
                runs=sorted([r for r in group if r['candidate_id']==index],key=lambda r:r['repeat'])
                assert all(r['raw_round_trip']=='PASS_EXACT' for r in runs)
                assert runs[0]['repeat_comparison']=='FIRST_REPEAT' and all(r['repeat_comparison']=='PASS_EXACT' for r in runs[1:])
                keys=['physical_valid','ready','ready_eligible','first_ready_interval','ready_prefix','cost','cost_finite','intervals','termination']
                assert all({k:r[k] for k in keys}=={k:runs[0][k] for k in keys} for r in runs)
                a=np.load(BASE/f'states/{state}/actual/candidate-{index:03d}-{panel}.npz',allow_pickle=False)
                assert a['states'].shape==((6 if panel=='A' else 9),10)
                assert a['applied_actions'].shape==((5 if panel=='A' else 8),2)
                if panel=='A':
                    actual_valid[index]=all(r['physical_valid'] and r['horizon_completed'] and r['cost_finite'] for r in runs)
                    assert np.array_equal(a['applied_actions'],a['nominal_actions'])
                    assert all(math.isclose(sum(r['cost_components'].values()),r['cost'],rel_tol=1e-12,abs_tol=1e-12) for r in runs)
                if runs[0]['ready']:
                    step=runs[0]['first_ready_interval'];assert a['hold_s'][step]+1e-9>=.3
            first=[r for r in group if r['repeat']==0]
            bank=np.load(BASE/f'states/{state}/bank.npz',allow_pickle=False)['actions']
            summary[panel]={'candidate_id_count':16,'unique_action_sequences':len(np.unique(bank.reshape(16,-1),axis=0)),'stable_physical_valid':sum(r['physical_valid'] for r in first),'ready_prefix_candidates':[r['candidate_id'] for r in first if r['ready_prefix']],'ready_eligible_candidates':[r['candidate_id'] for r in first if r['ready_eligible']]}
        model_stats={}
        predictions=json.loads((BASE/f'states/{state}/predictions.json').read_text())
        for name,meta in predictions.items():
            with np.load(BASE/f'states/{state}/prediction-{name}.npz',allow_pickle=False) as p:
                assert p['states'].shape==(16,6,10)
                assert int(p['valid'].sum())==meta['valid']
                false_rejected=[i for i in range(16) if not p['valid'][i] and actual_valid[i]]
                eligible=np.flatnonzero(p['valid'] & np.isfinite(p['total_cost']))
                selected=None if not len(eligible) else int(eligible[np.argmin(p['total_cost'][eligible])])
                assert selected==meta['selected_original_candidate_id']
                model_stats[name]={'predicted_valid':meta['valid'],'actual_A_valid_but_predicted_invalid_ids':false_rejected,'selected_id_among_first16_only':selected}
        stats[state]={'actual':summary,'models':model_stats,'cost_vs_ready_A':'NOT_IDENTIFIABLE_NO_READY_A','full128_search_coverage':'NOT_EVALUATED'}
    assert stats['N-T']['actual']['B']['ready_eligible_candidates']==[9]
    assert not stats['N-T']['actual']['B']['ready_prefix_candidates']
    for state in stats:assert not stats[state]['actual']['A']['ready_eligible_candidates']
    assert all(v==0 for v in x['future_engine_calls_during_prediction'].values())
    assert x['training_updates']==0
    limits=json.loads((ROOT/'configs/followup_candidate_diagnostic.v1.json').read_text())['resource_limits'];projection=x['resource_projection']
    # Include D1 time and separate-pilot upper allowance; no optimistic reuse assumption.
    d1_time=sum(json.loads((BASE/f'restoration-{s}.json').read_text())['wall_seconds'] for s in stats)
    base_remaining=limits['base_total_max_intervals']+limits['restore_prefix_intervals']+limits['pilot_max_additional_intervals_if_not_reused']-254-78-1242
    resources={}
    for name,count in [('base_D3_without_extension',base_remaining),('maximum_with_conditional_extensions',projection['conservative_remaining_intervals'])]:
        size=int(projection['current_run_bytes']+1.5*count*projection['max_measured_trial_bytes_per_returned_interval'])
        remaining=1.5*count*projection['max_measured_trial_wall_seconds_per_returned_interval']
        resources[name]={'remaining_interval_upper_bound':count,'projected_total_bytes':size,'remaining_wall_seconds_upper_bound':remaining,'wall_seconds_including_D1_D2_already_spent':remaining+d1_time+x['wall_seconds'],'storage_cap_fit':size<=limits['incremental_storage_cap_bytes'],'wall_cap_fit':remaining+d1_time+x['wall_seconds']<=limits['wall_time_cap_s']}
    return {'schema':'selected-followup-pilot-verification-v1','status':'PASS_SELECTED_FILES_AND_PILOT_ARITHMETIC','verifier_sha256':digest(__file__),'dataset_manifest_sha256':digest(BASE/'manifest.json'),'verified_selected_files':len(manifest['files']),'physical_trajectories_reported':192,'actual_intervals_reported':1242,'statistics':stats,'resources':resources,'new_physics_calls':0,'new_model_calls':0,'training_updates':0,'scope':'Selected hashes, recorded counts, first-repeat boundary arrays and forecast masks. Does not independently restore omitted original raw; separate execution audit covers local raw verification. Resource values are conservative projections, not measured full-run time.'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path);a=ap.parse_args();v=verify()
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':v['status'],'files':v['verified_selected_files'],'actual':{k:x['actual'] for k,x in v['statistics'].items()},'resources':v['resources']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
