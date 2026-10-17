"""Bounded, model-free preflight for the frozen Y/stop data intervention.

Writes ordinary validated episode datasets, including warmup and diagnostic
replays. Every family is smoke-test/preflight only; no training manifest is made.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import tomllib
import traceback

import numpy as np

from push_align.config import ROOT, CONFIG_PATH
from push_align.data import (digest, iter_dataset, make_contract, sequence_windows,
                             EpisodeRecorder, snapshot_from_episode, write_dataset,
                             _encode_nonfinite, validate_family_splits, Episode)
from push_align.data_intervention import build_preflight_plan
from push_align.environment import InitialSpec, PlanarPushEnv, wrap_angle
from push_align.planning import contact_distance


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def tree_bytes(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def exact_paths(left, right):
    """Compare physical values, not differing provenance/terminal metadata."""
    def joined(episodes, key):
        boundary = key in {'state', 'sim_time', 'raw_integration_state',
                          'raw_reference_height', 'raw_warning_counts',
                          'raw_warning_lastinfo', 'raw_needs_reset', 'observation_valid'}
        return np.concatenate([e.arrays[key] if i == 0 or not boundary else e.arrays[key][1:]
                               for i, e in enumerate(episodes)])
    keys = ['state', 'sim_time', 'raw_integration_state', 'raw_reference_height',
            'raw_warning_counts', 'raw_warning_lastinfo', 'raw_needs_reset',
            'observation_valid', 'action_requested', 'action_applied', 'actual_dt',
            'step_valid', 'geometry_valid', 'transition_valid', 'substep_states',
            'substep_times', 'contact_summary', 'validity_summary']
    checks = {k: bool(np.array_equal(joined(left, k), joined(right, k))) for k in keys}
    boundary_keys=['state','sim_time','raw_integration_state','raw_reference_height',
                   'raw_warning_counts','raw_warning_lastinfo','raw_needs_reset','observation_valid']
    checks['continuous_episode_seams'] = all(
        all(np.array_equal(a.arrays[k][-1],b.arrays[k][0]) for k in boundary_keys)
        and a.metadata['snapshot_rng_states'][-1] == b.metadata['snapshot_rng_states'][0]
        for episodes in (left,right) for a,b in zip(episodes,episodes[1:]))
    for key in ['diagnostics', 'geometry_checks']:
        checks[key] = [x for e in left for x in e.metadata[key]] == [x for e in right for x in e.metadata[key]]
    checks['snapshot_rng_states'] = ([x for i,e in enumerate(left) for x in e.metadata['snapshot_rng_states'][int(i>0):]]
                                     == [x for i,e in enumerate(right) for x in e.metadata['snapshot_rng_states'][int(i>0):]])
    return {'pass': all(checks.values()), 'checks': checks}


def compare_dt(warmup, core, fine, profile, config):
    a = np.vstack([warmup.arrays['state'], core.arrays['state'][1:]])
    b = fine.arrays['state']
    times = np.r_[warmup.arrays['sim_time'], core.arrays['sim_time'][1:]]
    fraction = profile['numeric']['timestep_difference_fraction_of_task_tolerance']
    budgets = {'object_xy_m': fraction*config['task']['position_tolerance_m'],
               'object_yaw_rad': fraction*config['task']['yaw_tolerance_rad']}
    same_length = a.shape == b.shape
    n = min(len(a), len(b)); delta = a[:n]-b[:n]
    metrics = {'object_xy_m': float(np.linalg.norm(delta[:,:2], axis=1).max()),
               'object_yaw_rad': float(np.abs(wrap_angle(delta[:,2])).max()),
               'pusher_xy_m': float(np.linalg.norm(delta[:,6:8], axis=1).max()),
               'object_velocity_xy_m_s': float(np.linalg.norm(delta[:,3:5], axis=1).max()),
               'object_wz_rad_s': float(np.abs(delta[:,5]).max()),
               'pusher_velocity_xy_m_s': float(np.linalg.norm(delta[:,8:10], axis=1).max())}
    checks = {'complete_22': same_length and fine.transition_count == 22,
              'same_initial_integration_state': bool(np.array_equal(warmup.arrays['raw_integration_state'][0], fine.arrays['raw_integration_state'][0])),
              'same_action_boundaries': same_length and bool(np.allclose(times, fine.arrays['sim_time'], rtol=0, atol=config['validity']['dt_match_tolerance_s'])),
              'same_requested_actions': bool(np.array_equal(np.vstack([warmup.arrays['action_requested'], core.arrays['action_requested']]), fine.arrays['action_requested'])),
              'all_valid': all(bool(e.arrays['transition_valid'].all()) for e in (warmup,core,fine)),
              'object_xy_budget': metrics['object_xy_m'] <= budgets['object_xy_m'],
              'object_yaw_budget': metrics['object_yaw_rad'] <= budgets['object_yaw_rad']}
    return {'pass': all(checks.values()), 'checks': checks, 'max_action_boundary_differences': metrics,
            'budgets': budgets, 'contact_boundary_disagreements': int(np.count_nonzero(np.vstack([warmup.arrays['contact_summary'], core.arrays['contact_summary']])[:fine.transition_count,1] != fine.arrays['contact_summary'][:,1])),
            'scope': 'fixed S1 commands; pusher/velocity/contact differences descriptive, not new acceptance thresholds'}


def head_exact(s0, s1):
    checks = {}
    for key in ['state', 'raw_integration_state', 'sim_time']:
        checks[key] = bool(np.array_equal(s0.arrays[key][:6], s1.arrays[key][:6]))
    for key in ['action_requested', 'action_applied', 'contact_summary', 'transition_valid']:
        checks[key] = bool(np.array_equal(s0.arrays[key][:5], s1.arrays[key][:5]))
    checks['all_raw_diagnostics'] = s0.metadata['diagnostics'][:5] == s1.metadata['diagnostics'][:5]
    checks['full_start_snapshot'] = exact_snapshot(snapshot_from_episode(s0,0), snapshot_from_episode(s1,0))
    return {'pass': all(checks.values()), 'checks': checks}


def exact_snapshot(a, b):
    return bool(a.model_hash == b.model_hash and a.environment_hash == b.environment_hash
                and a.reference_height == b.reference_height and a.rng_state == b.rng_state
                and a.needs_reset == b.needs_reset and all(np.array_equal(getattr(a,k),getattr(b,k))
                    for k in ['integration_state','warning_counts','warning_lastinfo']))


def measurements(episode, config, core_offset=0):
    a = episode.arrays; states = a['state']; diagnostics = episode.metadata['diagnostics']
    threshold = config['evaluation']['contact_active_normal_force_min_N']
    raw = [diagnostics[0]['samples'][0], *[s for d in diagnostics for s in d['samples'][1:]]]
    contact = [any(c['pair']=='pusher_object' and c['normal_force_N']>threshold for c in s['contacts']) for s in raw]
    zero = np.all(a['action_applied']==0, axis=1)
    blocks=[]; start=None
    for i, z in enumerate([*zero, False]):
        if z and start is None: start=i
        if not z and start is not None:
            end=i; sample_states=np.asarray([diagnostics[start]['samples'][0]['state'], *[s['state'] for d in diagnostics[start:end] for s in d['samples'][1:]]])
            blocks.append({'start_action':start, 'end_action_exclusive':end,
                           'core_start_time_s': (start-core_offset)*config['physics']['action_dt_s'],
                           'duration_s':float(a['actual_dt'][start:end].sum()),
                           'boundary_states':states[start:end+1].tolist(),
                           'first_substep_pusher_vy_m_s':float(sample_states[1,9]),
                           'pusher_displacement_m':float(np.linalg.norm(states[end,6:8]-states[start,6:8])),
                           'object_displacement_m':float(np.linalg.norm(states[end,:2]-states[start,:2])),
                           'object_yaw_change_rad':float(wrap_angle(states[end,2]-states[start,2])),
                           'pusher_max_speed_m_s':float(np.linalg.norm(sample_states[:,8:10],axis=1).max()),
                           'object_max_speed_m_s':float(np.linalg.norm(sample_states[:,3:5],axis=1).max()),
                           'object_max_abs_wz_rad_s':float(np.abs(sample_states[:,5]).max()),
                           'active_contact_samples':sum(any(c['pair']=='pusher_object' and c['normal_force_N']>threshold for c in s['contacts']) for d in diagnostics[start:end] for s in d['samples'][1:])})
            start=None
    windows = list(sequence_windows(episode,5,episode.metadata['contract'])) if episode.transition_count >= 5 else []
    allzero = [i for i in range(max(0,len(zero)-4)) if zero[i:i+5].all() and a['transition_valid'][i:i+5].all()]
    return {'transition_count':episode.transition_count, 'valid_transitions':int(a['transition_valid'].sum()),
            'requested_applied_exact':bool(np.array_equal(a['action_requested'],a['action_applied'])),
            'start_state':states[0].tolist(), 'end_state':states[-1].tolist(),
            'pusher_vy_boundary_range_m_s':[float(states[:,9].min()),float(states[:,9].max())],
            'active_contact_samples':sum(contact), 'unique_time_samples':len(raw),
            'first_active_contact_elapsed_s':next((float(s['sim_time']-raw[0]['sim_time']) for s,c in zip(raw,contact) if c),None),
            'active_contact_transitions':int(np.count_nonzero(a['contact_summary'][:,2]>0)),
            'noncontact_transitions':int(np.count_nonzero(a['contact_summary'][:,2]==0)),
            'start_side_gap_m':float(contact_distance(states[0],config)),
            'max_penetration_m':max(g['max_finite_penetration_m'] for g in episode.metadata['geometry_checks']),
            'zero_action_count':int(zero.sum()), 'zero_blocks':blocks,
            'valid_K5_windows':len(windows), 'all_zero_K5_starts':allzero,
            'zero_occurrences_in_all_K5':sum(int(zero[i:i+5].sum()) for i in range(max(0,len(zero)-4)))}


class PreflightFailure(RuntimeError):
    pass


class Runner:
    def __init__(self, out, design):
        self.out=out; self.design=design; self.counts={'action_calls':0,'completed_intervals':0,'physics_substeps':0,'by_role':{}}
        self.episodes=[]; self.settling=[]; self.lineage=[]

    def resources(self):
        used=tree_bytes(self.out); free=shutil.disk_usage(self.out).free
        limit=self.design['resources']['preflight_output_limit_MiB']*1024**2
        reserve=self.design['resources']['minimum_free_reserve_GiB']*1024**3
        # Reserve a conservative complete episode before starting, not after disk fills.
        if used+32*1024**2>limit or free<reserve+32*1024**2:
            raise PreflightFailure('output_cap_or_free_reserve')
        return {'output_bytes':used,'free_bytes':free}

    def record(self, env, plan, label, actions, contract, role):
        self.current_recorder=None
        try:
            return self._record(env,plan,label,actions,contract,role)
        except Exception:
            # Keep the unfinished prefix even when serialization/contract code fails.
            recorder=self.current_recorder
            snap=env.snapshot()
            folder=self.out/'failure_raw'; folder.mkdir(exist_ok=True)
            numeric={'current_integration_state':snap.integration_state,
                     'current_warning_counts':snap.warning_counts,
                     'current_warning_lastinfo':snap.warning_lastinfo}
            raw={'label':label,'plan':plan,'current_sample':env._sample(),
                 'current_snapshot_metadata':{'model_hash':snap.model_hash,'environment_hash':snap.environment_hash,
                     'reference_height':snap.reference_height,'rng_state':snap.rng_state,'needs_reset':snap.needs_reset},
                 'recorded_results':[asdict(r) for r in recorder.results] if recorder else [],
                 'recorded_metadata':recorder.metadata if recorder else None,
                 'scope':'available prefix and current physical state; software exceptions may prevent capturing the failed step diagnostics'}
            if recorder:
                numeric['recorded_integration_states']=np.asarray([s.integration_state for s in recorder.snapshots])
            paths=[]; values=[]; safe=_encode_nonfinite(raw,paths=paths,values=values)
            safe['nonfinite_paths']=paths; numeric['nonfinite_values']=np.asarray(values,dtype=np.float64)
            with (folder/f'{label}.npz').open('xb') as stream: np.savez_compressed(stream,**numeric)
            save(folder/f'{label}.json',safe)
            raise

    def _record(self, env, plan, label, actions, contract, role):
        self.resources()
        recorder=EpisodeRecorder(env,episode_id=label,episode_index=len(self.episodes),
            family_id=plan['family_id'],family_definition={'base_initial_spec':{k:plan['initial_spec'][k] for k in ('object_x','object_y','object_yaw')},
                'variant_rule':'All Y/S/warmup/dt/replays stay in preflight; excluded from both training and development evaluation'},
            split='smoke-test', collection_policy={'kind':'p4-data-preflight','role':role,'predeclared_plan':plan,
                'training_eligible':False,'development_evaluation_eligible':False},seed=self.design['collection']['preflight_initial_seed'],contract=contract)
        self.current_recorder=recorder
        failure=None
        for action in actions:
            if self.counts['action_calls'] >= self.design['collection']['preflight_intervals_upper_bound']:
                failure='action_budget'; break
            self.counts['action_calls']+=1
            self.counts['by_role'][role]=self.counts['by_role'].get(role,0)+1
            result=recorder.step(action)
            self.counts['physics_substeps']+=len(result.substep_times)
            self.counts['completed_intervals']+=int(abs(result.actual_dt-env.action_dt)<=env.config['validity']['dt_match_tolerance_s'])
            if recorder.invalid or not np.array_equal(result.requested_action,result.applied_action):
                failure='physical_geometry_or_clipping'; break
        if not recorder.results:
            raise PreflightFailure(failure or 'empty_episode')
        episode=recorder.finish(failure or 'collection_complete')
        folder=self.out/'episodes'/label
        manifest=write_dataset(folder,[episode],purpose='contract-check')
        # Independently reload the serialized raw arrays/diagnostics and validate again.
        loaded,=list(iter_dataset(folder,contract))
        identity={k:deepcopy(loaded.metadata[k]) for k in ('episode_id','episode_index','family_id','family_definition','initial_record','split')}
        self.lineage.append(Episode({},identity))
        validate_family_splits(self.lineage)
        roundtrip=exact_paths([episode],[loaded])
        row={'label':label,'role':role,'family_id':plan['family_id'],'geometry':plan['geometry'],
             'Y':plan['y_condition'],'S':plan['stop_condition'], 'dataset':str(folder.relative_to(self.out)),
             'manifest_sha256':digest(folder/'manifest.json'),'roundtrip_exact':roundtrip['pass'],
             'measurements':measurements(loaded,env.config,2 if role in ('dt_half','exact_replay') else 0),
             'terminal_reason':loaded.metadata['terminal_reason']}
        self.episodes.append(row)
        save(self.out/'progress'/f'{label}.json',row)
        print(json.dumps({'episode':label,'intervals':loaded.transition_count,'valid':row['measurements']['valid_transitions'],'contact_intervals':row['measurements']['active_contact_transitions'],'total_action_calls':self.counts['action_calls']}),flush=True)
        if failure or not roundtrip['pass']:
            raise PreflightFailure(failure or 'serialization_not_exact')
        return loaded

    def reset_log(self, env, role, initial=None):
        if initial is not None: env.reset(initial)
        self.settling.append({'role':role,'physics_dt_s':float(env.model.opt.timestep),
                              'settling_substeps':round(env.initial_record['settle_time_s']/env.model.opt.timestep),
                              'initial_record':deepcopy(env.initial_record)})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(); out=args.output_dir.resolve()
    design_path=ROOT/'configs/p4_data_intervention.design.v1.json'
    design=json.loads(design_path.read_text()); config=tomllib.loads(CONFIG_PATH.read_text())
    profile_path=ROOT/'configs/acceptance.v1.toml'; profile=tomllib.loads(profile_path.read_text())
    plans=build_preflight_plan(design,config)
    if shutil.disk_usage(out.parent).free < design['resources']['minimum_free_reserve_GiB']*1024**3 + 512*1024**2:
        raise PreflightFailure('insufficient_free_disk_before_registration')
    out.mkdir(parents=True,exist_ok=False); (out/'progress').mkdir(); (out/'episodes').mkdir()
    half=deepcopy(config); half['physics']['timestep_s']/=2; half['physics']['frame_skip']*=2
    half_text=CONFIG_PATH.read_text().replace('timestep_s = 0.0005','timestep_s = 0.00025').replace('frame_skip = 200','frame_skip = 400')
    assert tomllib.loads(half_text)==half
    half_path=out/'dt_half.toml'; half_path.write_text(half_text)
    source_paths=[design_path,CONFIG_PATH,profile_path,Path(__file__).resolve(),ROOT/'src/push_align/data_intervention.py',
                  ROOT/'src/push_align/environment.py',ROOT/'src/push_align/data.py',ROOT/'src/push_align/contact_geometry.py',ROOT/'src/push_align/planning.py',ROOT/'src/push_align/config.py',ROOT/'assets/push_scene.xml']
    source_hashes={str(p.relative_to(ROOT)):digest(p) for p in source_paths}
    save(out/'registration.json',{'schema':'p4-data-preflight-v1','status':'REGISTERED_BEFORE_PHYSICS','plans':plans,
        'source_sha256':source_hashes,'design':design,'scope':'3 preflight-only families; no model forward, learning, main collection or task-success evaluation',
        'execution':'6 physical warmups shared by paired S cores; all 6 S1 dt/2 and nominal exact replays; stop on first failed gate',
        'dt_protocol':'identical settled integration state; existing import_dt_comparison_state zeros only solver warmstart on both branches before warmup; actual measured velocities unchanged',
        'expected_action_intervals':516,'budget_action_intervals':528,'constructor_and_explicit_reset_settling':'logged separately',
        'physical_config':config,'half_config':half,'acceptance':profile,'free_bytes_before':shutil.disk_usage(out).free,
        'data_storage':'existing episode-v1 NPZ + complete raw JSON, smoke-test/contract-check; outer registration marks all ineligible for learning'} )
    runner=Runner(out,design); checks=[]; start=time.monotonic(); failure=None
    try:
        base=PlanarPushEnv(config); runner.reset_log(base,'nominal_constructor')
        fine=PlanarPushEnv(half); runner.reset_log(fine,'half_constructor')
        contract=make_contract(base,CONFIG_PATH,profile_path); fine_contract=make_contract(fine,half_path,profile_path)
        for i in range(0,len(plans),2):
            p0,p1=plans[i:i+2]; prefix=p0['family_id']+'-'+p0['y_condition']
            runner.reset_log(base,prefix,InitialSpec(**p0['initial_spec']))
            fine.import_dt_comparison_state(base); base.import_dt_comparison_state(base)
            fine.initial_record=deepcopy(base.initial_record)
            initial=base.snapshot(); fine_initial=fine.snapshot()
            warm=runner.record(base,p0,prefix+'-warmup',p0['warmup_actions'],contract,'warmup')
            core_start=base.snapshot(); vy=float(base.observe().state[9])
            c=design['collection']; ypass=vy>=c['measured_high_Y_gate_m_s'] if p0['y_condition']=='Y1' else vy<=c['measured_low_Y_ceiling_m_s']
            check={'cell':prefix,'kind':'measured_Y','vy_m_s':vy,'pass':ypass}; checks.append(check)
            if not ypass: raise PreflightFailure(f'{prefix}:measured_Y_gate')
            s0=runner.record(base,p0,prefix+'-S0',p0['core_actions'],contract,'core')
            base.restore(core_start)
            s1=runner.record(base,p1,prefix+'-S1',p1['core_actions'],contract,'core')
            pair=head_exact(s0,s1); pair.update(cell=prefix,kind='S_pair_first5_exact'); checks.append(pair)
            if not pair['pass']: raise PreflightFailure(f'{prefix}:S_pair_first5')
            for p,e in [(p0,s0),(p1,s1)]:
                m=measurements(e,config)
                ok=m['valid_K5_windows']==16 and m['zero_action_count']==5 and m['zero_occurrences_in_all_K5']==25 and len(m['all_zero_K5_starts'])==int(p['stop_condition']=='S1')
                checks.append({'cell':prefix,'kind':p['stop_condition']+'_window_contract','pass':ok})
                if not ok: raise PreflightFailure(f'{prefix}:window_contract')
            actions=p1['warmup_actions']+p1['core_actions']
            base.restore(initial)
            replay=runner.record(base,p1,prefix+'-replay',actions,contract,'exact_replay')
            exact=exact_paths([warm,s1],[replay]); exact.update(cell=prefix,kind='nominal_exact_replay'); checks.append(exact)
            if not exact['pass']: raise PreflightFailure(f'{prefix}:exact_replay')
            fine.restore(fine_initial)
            refined=runner.record(fine,p1,prefix+'-half',actions,fine_contract,'dt_half')
            dt=compare_dt(warm,s1,refined,profile,config); dt.update(cell=prefix,kind='dt_half'); checks.append(dt)
            save(out/'progress'/f'{prefix}-checks.json',checks[-6:])
            print(json.dumps({'cell':prefix,'Y_actual_m_s':vy,'dt_pass':dt['pass'],'dt_differences':dt['max_action_boundary_differences']}),flush=True)
            if not dt['pass']: raise PreflightFailure(f'{prefix}:dt_half_budget')
    except Exception as error:
        failure={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
    unchanged=all(digest(ROOT/p)==h for p,h in source_hashes.items())
    accounting={'action_calls':runner.counts['action_calls']==516,
                'completed_intervals':runner.counts['completed_intervals']==516,
                'physics_substeps':runner.counts['physics_substeps']==129600,
                'episodes':len(runner.episodes)==30,
                'roles':runner.counts['by_role']=={'warmup':12,'core':240,'exact_replay':132,'dt_half':132}}
    resources={'output_under_cap':tree_bytes(out)+1024**2<=design['resources']['preflight_output_limit_MiB']*1024**2,
               'free_reserve':shutil.disk_usage(out).free-1024**2>=design['resources']['minimum_free_reserve_GiB']*1024**3}
    passed=failure is None and len(checks)==36 and all(c['pass'] for c in checks) and unchanged and all(accounting.values()) and all(resources.values())
    summary={'schema':'p4-data-preflight-result-v1','status':'PASS' if passed else 'FAIL',
        'failure':failure,'checks':checks,'accounting_checks':accounting,'resource_checks':resources,
        'episodes':runner.episodes,'counts':runner.counts,'settling':runner.settling,
        'settling_substeps_total':sum(r['settling_substeps'] for r in runner.settling),
        'output_bytes_before_summary':tree_bytes(out),'free_bytes_after':shutil.disk_usage(out).free,
        'wall_seconds':time.monotonic()-start,'source_unchanged':unchanged,
        'main_collection':'NOT_RUN','learning_updates':0,'model_forwards':0,'P5_01':'HOLD',
        'task_success':'NOT_EVALUATED','personal_understanding':'NOT_ASSESSED',
        'scope':'collector/data manipulation and 6 representative physics checks only; no data-effect or alignment-improvement claim'}
    save(out/'summary.json',summary)
    save(out/'artifact_hashes.json',{'sha256':{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}})
    print(json.dumps({'status':summary['status'],'failure':failure,'counts':runner.counts,'wall_seconds':summary['wall_seconds']}),flush=True)
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
