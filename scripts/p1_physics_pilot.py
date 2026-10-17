"""P1 prerequisite recheck, paired dt sensitivity, and bounded resource audit.

All thresholds come from the pre-existing acceptance profile. A failed run
is saved and exits nonzero; output directories are never overwritten.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import hashlib
import gzip
import json
import platform
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter
import tomllib

import mujoco
import numpy as np
from push_align import PlanarPushEnv
from push_align.config import CONFIG_PATH, ROOT, load_pilot_config
from push_align.environment import task_angle_error, wrap_angle
from p1_contact_probe import condition_initial


def serializable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def save(path, data):
    if isinstance(data, dict) and 'samples' in data:
        # Detailed traces are retained losslessly without very large indented JSON files.
        with gzip.open(str(path)+'.gz', 'xt') as file:
            json.dump(data, file, default=serializable, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        return
    with path.open('x') as file:
        json.dump(data, file, default=serializable, ensure_ascii=False, allow_nan=False, indent=2)
        file.write('\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(env, action, ticks):
    start = env.observe()
    states, times, samples, intervals = [start.state], [0.], [], []
    tick_ms = []
    for _ in range(ticks):
        clock = perf_counter()
        result = env.step(action)
        tick_ms.append((perf_counter()-clock)*1000)
        states.append(result.observation.state)
        times.append(result.observation.sim_time-start.sim_time)
        # Keep each boundary only once across adjacent intervals.
        samples.extend(result.diagnostics['samples'] if not samples else result.diagnostics['samples'][1:])
        intervals.append({'requested': result.requested_action, 'applied': result.applied_action,
                          'actual_dt': result.actual_dt, 'valid': result.transition_valid,
                          'violations': result.diagnostics['validity_violations']})
        if not result.transition_valid:
            break
    return {'states': np.asarray(states), 'times': np.asarray(times), 'samples': samples,
            'intervals': intervals, 'tick_ms': tick_ms,
            'completed': len(intervals)==ticks and all(i['valid'] for i in intervals),
            'terminal_type': 'duration_complete' if len(intervals)==ticks and all(i['valid'] for i in intervals) else 'invalid'}


def state_difference(a, b):
    difference = np.abs(a-b)
    difference[:, 2] = np.abs(wrap_angle(a[:, 2]-b[:, 2]))
    return {'object_position_m': float(np.max(np.linalg.norm(a[:, :2]-b[:, :2], axis=1))),
            'object_yaw_rad': float(np.max(difference[:, 2])),
            'object_velocity_m_s': float(np.max(np.linalg.norm(a[:, 3:5]-b[:, 3:5], axis=1))),
            'object_angular_velocity_rad_s': float(np.max(difference[:, 5])),
            'pusher_position_m': float(np.max(np.linalg.norm(a[:, 6:8]-b[:, 6:8], axis=1))),
            'pusher_velocity_m_s': float(np.max(np.linalg.norm(a[:, 8:10]-b[:, 8:10], axis=1)))}


def replay_limits(config, fraction):
    t = config['task']
    return {'object_position_m': t['position_tolerance_m']*fraction,
            'object_yaw_rad': t['yaw_tolerance_rad']*fraction,
            'object_velocity_m_s': t['linear_speed_tolerance_m_s']*fraction,
            'object_angular_velocity_rad_s': t['angular_speed_tolerance_rad_s']*fraction,
            'pusher_position_m': t['position_tolerance_m']*fraction,
            'pusher_velocity_m_s': t['linear_speed_tolerance_m_s']*fraction}


def compiled_properties(env):
    m = env.model
    return {'timestep_s': float(m.opt.timestep), 'gravity': m.opt.gravity.tolist(),
            'integrator': int(m.opt.integrator), 'solver': int(m.opt.solver),
            'friction_cone': 'elliptic' if m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC else 'pyramidal',
            'native_ccd_enabled': not bool(m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)),
            'multi_ccd_enabled': not bool(m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)),
            'solver_iterations': int(m.opt.iterations), 'solver_tolerance': float(m.opt.tolerance),
            'object_size': m.geom_size[env._object_geom].tolist(), 'object_mass': float(m.body_mass[env._object_body]),
            'pusher_size': m.geom_size[env._pusher_geom].tolist(), 'pusher_mass': float(m.body_mass[env._pusher_body]),
            'pusher_position': m.body_pos[env._pusher_body].tolist(), 'joint_axes': m.jnt_axis.tolist(),
            'actuator_gear': m.actuator_gear.tolist(), 'actuator_gain': m.actuator_gainprm.tolist(),
            'actuator_bias': m.actuator_biasprm.tolist(), 'actuator_force_limits': m.actuator_forcerange.tolist(),
            'pair_friction': m.pair_friction.tolist(), 'pair_dim': m.pair_dim.tolist(),
            'pair_solref': m.pair_solref.tolist(), 'pair_solimp': m.pair_solimp.tolist()}


def compiled_matches_config(env):
    m, c = env.model, env.config
    checks = {
        'timestep': np.isclose(m.opt.timestep,c['physics']['timestep_s'],rtol=0,atol=1e-15),
        'solver_iterations': m.opt.iterations == c['physics']['solver_iterations'],
        'solver_tolerance': np.isclose(m.opt.tolerance,c['physics']['solver_tolerance'],rtol=0,atol=1e-16),
        'native_ccd': (not bool(m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD))) == c['physics'].get('native_ccd',True),
        'friction_cone': m.opt.cone == (mujoco.mjtCone.mjCONE_ELLIPTIC if c['physics'].get('friction_cone','pyramidal') == 'elliptic' else mujoco.mjtCone.mjCONE_PYRAMIDAL),
        'gravity': np.allclose(m.opt.gravity,c['physics']['gravity_m_s2']),
        'object_half_size': np.allclose(m.geom_size[env._object_geom],np.asarray(c['object']['size_full_xyz_m'])/2),
        'object_mass': np.isclose(m.body_mass[env._object_body],c['object']['mass_kg']),
        'pusher_radius_half_height': np.allclose(m.geom_size[env._pusher_geom,:2],[c['pusher']['radius_m'],c['pusher']['height_full_m']/2]),
        'pusher_mass': np.isclose(m.body_mass[env._pusher_body],c['pusher']['mass_kg']),
        'pusher_height': np.isclose(m.body_pos[env._pusher_body,2],c['pusher']['center_z_m']),
        'axes': np.allclose(m.jnt_axis[-2:],[[1,0,0],[0,1,0]]),
        'gear': np.allclose(m.actuator_gear[:,0],c['pusher']['gear']),
        'velocity_gain': np.allclose(m.actuator_gainprm[:,0],c['pusher']['velocity_gain_N_s_m']),
        'velocity_bias': np.allclose(m.actuator_biasprm[:,2],-c['pusher']['velocity_gain_N_s_m']),
        'force_limits': np.allclose(m.actuator_forcerange,[-c['pusher']['actuator_force_limit_abs_N'],c['pusher']['actuator_force_limit_abs_N']]),
        'pair_friction': np.allclose(m.pair_friction[:,:2],[c['contact']['floor_object_friction'][:2],c['contact']['pusher_object_friction'][:2]]),
        'pair_dimension': np.all(m.pair_dim==c['contact']['condim']),
        'solref': np.allclose(m.pair_solref,c['contact']['solref']),
        'solimp': np.allclose(m.pair_solimp,c['contact']['solimp']),
    }
    return {k:bool(v) for k,v in checks.items()}


def contact_geometry_check(case, label, trace, penetration_budget):
    contacts = [c for s in trace['samples'] for c in s['contacts']]
    side = [c for c in contacts if c['pair']=='pusher_object']
    nonfinite = [c for c in contacts if not np.isfinite([c['distance_m'],c['normal_force_N'],*c['normal_world']]).all()]
    reverse = [c for c in side if c['normal_center_cosine'] is None or not np.isfinite(c['normal_center_cosine']) or c['normal_center_cosine'] <= 0.]
    deep = [c for c in contacts if -c['distance_m'] > penetration_budget]
    return {'case':case,'dt':label,'nonfinite_contacts':len(nonfinite),'reversed_side_contact_normals':len(reverse),
            'over_budget_penetration_contacts':len(deep),'max_penetration_m':max([max(0.,-c['distance_m']) for c in contacts] or [0.]),
            'pass':not nonfinite and not reverse and not deep}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    parser.add_argument('--acceptance', type=Path, default=ROOT/'configs/acceptance.v1.toml')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    with args.config.open('rb') as file:
        config = tomllib.load(file)
    with args.acceptance.open('rb') as file:
        acceptance = tomllib.load(file)
    suite, numeric = acceptance['physics_suite'], acceptance['numeric']
    half_config = deepcopy(config)
    half_config['physics']['timestep_s'] *= suite['physics_dt_comparison_scale']
    half_config['physics']['frame_skip'] = round(config['physics']['action_dt_s']/half_config['physics']['timestep_s'])
    ticks = round(suite['observation_duration_s']/config['physics']['action_dt_s'])
    if not np.isclose(ticks*config['physics']['action_dt_s'], suite['observation_duration_s']):
        raise ValueError('Probe duration must be an integer number of actions')
    source_paths = [p for folder in ('src', 'scripts', 'tests') for p in (ROOT/folder).rglob('*.py')]
    source_paths += list((ROOT/'configs').glob('*.toml')) + [ROOT/'assets/push_scene.xml', ROOT/'requirements.lock.txt']
    save(out/'manifest.json', {'created_at': datetime.now().astimezone().isoformat(), 'argv': sys.argv,
        'executable': sys.executable, 'platform': platform.platform(), 'versions': {'numpy': np.__version__, 'mujoco': mujoco.__version__},
        'config': config, 'half_config': half_config, 'acceptance': acceptance,
        'evaluation_rule': 'max at all common action boundaries; identical physical initial state; warmstart zero in both dt branches',
        'active_snapshot_rule': 'first valid action boundary within 1 second with side contact force > configured active threshold',
        'source_hashes': {str(p.relative_to(ROOT)):digest(p) for p in source_paths}})
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=ROOT, text=True, capture_output=True)
    (out/'tests.txt').write_text(tests.stdout+tests.stderr)
    runtime = subprocess.run([sys.executable, str(ROOT/'scripts/p1_runtime_check.py'), '--config', str(args.config.resolve()), '--output', str(out.resolve()/'runtime.json')], cwd=ROOT, text=True, capture_output=True)
    (out/'runtime_stdout.txt').write_text(runtime.stdout+runtime.stderr)
    env, finer = PlanarPushEnv(config), PlanarPushEnv(half_config)
    (out/'resolved_base.xml').write_text(env.resolved_xml)
    (out/'resolved_half.xml').write_text(finer.resolved_xml)
    compiled = [compiled_properties(e) for e in (env, finer)]
    compiled_checks = [compiled_matches_config(e) for e in (env, finer)]
    save(out/'compiled.json', {'actual_properties':compiled,'checks':compiled_checks})
    limits = replay_limits(config, numeric['replay_tolerance_fraction_of_task_tolerance'])
    dt_limits = replay_limits(config, numeric['timestep_difference_fraction_of_task_tolerance'])
    cases, replay, resource_times, friction, geometry_checks = [], [], [], [], []
    started = perf_counter()
    for yaw_deg in suite['initial_yaw_candidates_deg']:
        yaw = float(np.deg2rad(yaw_deg))
        direction = np.array([np.cos(yaw), np.sin(yaw)])
        for condition in suite['conditions']:
            case_id = f"yaw{yaw_deg:+g}_{condition}"
            env.reset(condition_initial(yaw, condition, env, suite['probe_offset_abs_m']))
            reset_record = deepcopy(env.initial_record)
            action = np.zeros(2) if condition=='zero' else direction*suite['probe_command_speed_m_s']
            # Source is the already settled base state; no independent reset drift enters the comparison.
            finer.import_dt_comparison_state(env)
            env.import_dt_comparison_state(env)
            snapshot = env.snapshot()
            save(out/f'{case_id}_snapshot.json', asdict(snapshot))
            initial_match = bool(np.array_equal(env.data.qpos, finer.data.qpos) and np.array_equal(env.data.qvel, finer.data.qvel) and env.data.time==finer.data.time)
            repeated = []
            for repeat in range(suite['replays_per_snapshot']):
                env.restore(snapshot)
                result = execute(env, action, ticks)
                repeated.append(result)
                save(out/f'{case_id}_replay{repeat}.json', result)
            baseline = repeated[0]
            fine = execute(finer, action, ticks)
            save(out/f'{case_id}_half.json', fine)
            resource_times.extend(baseline['tick_ms'])
            for r in repeated[1:]:
                same_shape = r['states'].shape==baseline['states'].shape
                difference = state_difference(r['states'], baseline['states']) if same_shape else None
                passed = same_shape and r['terminal_type']==baseline['terminal_type'] and all(difference[k]<=limits[k] for k in limits)
                replay.append({'case':case_id, 'pass':bool(passed), 'max_differences':difference})
            comparable = baseline['completed'] and fine['completed'] and initial_match and baseline['states'].shape==fine['states'].shape and np.allclose(baseline['times'], fine['times'], rtol=0, atol=1e-9)
            difference = state_difference(baseline['states'], fine['states']) if comparable else None
            delta = baseline['states'][-1]-baseline['states'][0]
            fine_delta = fine['states'][-1]-fine['states'][0]
            # Only detectable rotations have a meaningful sign; tiny central drift has no imposed sign.
            angle_signal = numeric['signal_min_fraction_of_task_tolerance']*config['task']['yaw_tolerance_rad']
            sign_match = bool((np.sign(delta[2])==np.sign(fine_delta[2])) if max(abs(delta[2]), abs(fine_delta[2]))>=angle_signal else True)
            validity_pass = baseline['completed'] and fine['completed']
            case = {'case_id':case_id, 'yaw_deg':yaw_deg, 'condition':condition, 'reset': reset_record,
                'identical_initial_physical_state':initial_match, 'baseline_completed':baseline['completed'], 'half_completed':fine['completed'],
                'baseline_terminal_type':baseline['terminal_type'], 'half_terminal_type':fine['terminal_type'],
                'baseline_duration_s':float(baseline['times'][-1]), 'half_duration_s':float(fine['times'][-1]),
                'delta_xy_m':delta[:2].tolist(), 'forward_displacement_m':float(delta[:2]@direction),
                'yaw_delta_rad':float(wrap_angle(delta[2])), 'half_yaw_delta_rad':float(wrap_angle(fine_delta[2])),
                'max_position_drift_m':float(np.max(np.linalg.norm(baseline['states'][:, :2]-baseline['states'][0, :2], axis=1))),
                'max_yaw_drift_rad':float(np.max(np.abs(wrap_angle(baseline['states'][:, 2]-baseline['states'][0, 2])))),
                'dt_max_differences':difference, 'rotation_sign_match':sign_match,
                'dt_pass':bool(comparable and sign_match and difference['object_position_m']<=dt_limits['object_position_m'] and difference['object_yaw_rad']<=dt_limits['object_yaw_rad']),
                'validity_pass':validity_pass,
                'violations':[i['violations'] for r in (baseline, fine) for i in r['intervals'] if not i['valid']]}
            for label, result in [('base',baseline), ('half',fine)]:
                contacts=[c for s in result['samples'] for c in s['contacts']]
                for pair, key in [('floor_object','floor_object_friction'), ('pusher_object','pusher_object_friction')]:
                    seen=[c for c in contacts if c['pair']==pair]
                    # condim=3 uses the first two sliding friction coefficients; others are logged but inactive.
                    values=sorted({tuple(c['friction']) for c in seen})
                    friction.append({'case':case_id,'dt':label,'pair':pair,'samples':len(seen),'actual_coefficients':values,
                        'active_coefficients_match':bool(seen) and all(np.allclose(c['friction'][:2],config['contact'][key][:2],rtol=0,atol=1e-12) and c['dimension']==3 for c in seen)})
                case[label+'_max_tilt_rad']=max(s['tilt_rad'] for s in result['samples'])
                case[label+'_max_height_deviation_m']=max(abs(s['object_height_m']-snapshot.reference_height) for s in result['samples'])
                case[label+'_warning_count']=max(sum(s['warning_counts']) for s in result['samples'])
                case[label+'_max_force_N']=max(max(abs(f) for f in s['actuator_force_N'] + s.get('integration_actuator_force_N',[])) for s in result['samples'])
                case[label+'_max_penetration_m']=max([max(0.,-c['distance_m']) for c in contacts] or [0.])
                if suite.get('check_contact_geometry',False):
                    geometry_checks.append(contact_geometry_check(case_id,label,result,suite['actual_contact_penetration_max_m']))
            cases.append(case)
    # Restore while contact is active, then compare the next full second at every action boundary.
    env.reset(condition_initial(0., 'central', env, suite['probe_offset_abs_m']))
    active_selection=[]
    active_contact=False
    active_elapsed=0.
    for _ in range(ticks):
        probe=env.step([suite['probe_command_speed_m_s'],0])
        active_elapsed += probe.actual_dt
        force=max([c['normal_force_N'] for c in env._sample()['contacts'] if c['pair']=='pusher_object'] or [0.])
        active_contact=probe.transition_valid and force>config['evaluation']['contact_active_normal_force_min_N']
        active_selection.append({'elapsed_s':active_elapsed,'valid':probe.transition_valid,'side_normal_force_N':force,'selected':active_contact})
        if active_contact or not probe.transition_valid: break
    active_snapshot=env.snapshot()
    save(out/'active_contact_snapshot.json',asdict(active_snapshot))
    active_repeats=[]
    for repeat in range(3):
        env.restore(active_snapshot)
        result=execute(env,np.array([.03,0]),ticks)
        active_repeats.append(result)
        save(out/f'active_contact_replay{repeat}.json',result)
        if suite.get('check_contact_geometry',False):
            geometry_checks.append(contact_geometry_check('active_contact',f'replay{repeat}',result,suite['actual_contact_penetration_max_m']))
    for r in active_repeats[1:]:
        d=state_difference(r['states'],active_repeats[0]['states']) if r['states'].shape==active_repeats[0]['states'].shape else None
        replay.append({'case':'active_contact','pass':bool(active_contact and all(a['completed'] and len(a['intervals'])==ticks and abs(a['times'][-1]-suite['observation_duration_s'])<=1e-9 for a in active_repeats) and d is not None and r['terminal_type']==active_repeats[0]['terminal_type'] and all(d[k]<=limits[k] for k in limits)), 'max_differences':d})
    contrast_checks=[]
    for yaw in suite['initial_yaw_candidates_deg']:
        by={c['condition']:c for c in cases if c['yaw_deg']==yaw}
        controls=[by['zero'],by['noncontact']]
        drift=max(c['max_position_drift_m'] for c in controls)
        angle_drift=max(c['max_yaw_drift_rad'] for c in controls)
        min_move=max(numeric['signal_to_control_drift_min_ratio']*drift,numeric['signal_min_fraction_of_task_tolerance']*config['task']['position_tolerance_m'])
        minimum_angle=max(numeric['signal_to_control_drift_min_ratio']*angle_drift,numeric['signal_min_fraction_of_task_tolerance']*config['task']['yaw_tolerance_rad'])
        check={'yaw_deg':yaw,'drift_pass':drift<=config['task']['position_tolerance_m']*numeric['physical_drift_fraction_of_task_tolerance'] and angle_drift<=config['task']['yaw_tolerance_rad']*numeric['physical_drift_fraction_of_task_tolerance'],
            'central_signal_pass':by['central']['forward_displacement_m']>=min_move, 'minimum_translation_m':min_move,
            'minimum_rotation_rad':minimum_angle,'offset_sign_pass':by['positive_offset']['yaw_delta_rad']<0<by['negative_offset']['yaw_delta_rad'],
            'offset_signal_pass':min(abs(by['positive_offset']['yaw_delta_rad']),abs(by['negative_offset']['yaw_delta_rad']))>=minimum_angle}
        contrast_checks.append(check)
    # Bare engine and complete logging path measured separately. No controller/training estimate implied.
    raw = PlanarPushEnv(config)
    for _ in range(100): mujoco.mj_step(raw.model,raw.data)
    clock=perf_counter()
    for _ in range(1000): mujoco.mj_step(raw.model,raw.data)
    raw_ms=(perf_counter()-clock)*1000
    clock=perf_counter()
    payload=json.dumps(active_repeats[0],default=serializable,allow_nan=False).encode()
    serialize_ms=(perf_counter()-clock)*1000
    full_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform!='darwin': peak*=1024
    resource_report={'bare_engine_1000_steps_ms':raw_ms,'logged_action_intervals':len(resource_times),
        'logged_action_median_ms':float(np.median(resource_times)),'logged_action_p95_ms':float(np.percentile(resource_times,95)),
        'logged_action_max_ms':max(resource_times),'serialization_1s_record_ms':serialize_ms,'compact_json_1s_record_bytes':len(payload),
        'allocated_run_bytes_before_summary':full_bytes,'process_peak_rss_bytes':peak,'process_peak_scope':'this combined audit process, including retained diagnostic records; excludes separate torch subprocess',
        'measurement_scope':'single local machine; physics and logging only; no learned planner latency claim',
        'suite_wall_s':perf_counter()-started}
    geometry_assessed = suite.get('check_contact_geometry',False)
    expected_geometry_count = 2*len(cases)+len(active_repeats)
    geometry_pass = len(geometry_checks)==expected_geometry_count and all(g['pass'] for g in geometry_checks)
    dt_pass=all(c['dt_pass'] and c['validity_pass'] and c['base_warning_count']==0 and c['half_warning_count']==0 for c in cases)
    friction_pass=all(f['active_coefficients_match'] for f in friction if f['pair']=='floor_object' or not (f['case'].endswith('_zero') or f['case'].endswith('_noncontact')))
    contrast_pass=all(c['drift_pass'] and c['central_signal_pass'] and c['offset_sign_pass'] and c['offset_signal_pass'] for c in contrast_checks) and all(c['baseline_completed'] for c in cases)
    verdicts={'P1-01':'PASS' if runtime.returncode==0 else 'FAIL',
        'P1-02':'PASS' if tests.returncode==0 and all(all(c.values()) for c in compiled_checks) and all(c['reset']['stable_time_s']+1e-12>=config['reset']['required_stable_time_s'] for c in cases) else 'FAIL',
        'P1-03':'PASS' if tests.returncode==0 and all(r['completed'] and len(r['intervals'])==ticks and all(i['valid'] and abs(i['actual_dt']-config['physics']['action_dt_s'])<=config['validity']['dt_match_tolerance_s'] for i in r['intervals']) for r in active_repeats) else 'FAIL',
        'P1-04':'PASS' if contrast_pass else 'FAIL', 'P1-05':'PASS' if tests.returncode==0 and all(r['pass'] for r in replay) else 'FAIL',
        'P1-06':'PASS' if dt_pass and friction_pass and (geometry_pass or not geometry_assessed) else 'FAIL'}
    overall=all(v=='PASS' for v in verdicts.values())
    save(out/'summary.json',{'run_id':out.name, 'stage_verdict':'PASS' if overall else 'FAIL', 'task_verdicts':verdicts,
        'model_hash':env.model_hash,'environment_hash':env.environment_hash,'dt_limits':dt_limits,'replay_limits':limits,
        'cases':cases,'contrast_checks':contrast_checks,'replay_checks':replay,'contact_coefficients':friction,
        'contact_geometry_checks':geometry_checks,'geometry_verdict':('PASS' if geometry_pass else 'FAIL') if geometry_assessed else 'NOT_ASSESSED',
        'expected_geometry_checks':expected_geometry_count if geometry_assessed else 0,
        'active_contact_selection':active_selection,'active_contact_snapshot_found':bool(active_contact),
        'geometry_check_scope':'separated convex cylinder-box side contacts: geom1 to geom2 normal; all contacts penetration budget; paired probes and all active replays',
        'resources':resource_report,'runtime_exit_code':runtime.returncode,'tests_exit_code':tests.returncode,
        'limited_next_collection_probe_allowed':overall,'main_data_collection_authorized':False,
        'unit_test_scope':'shared implementation regressions at explicitly chosen configurations; candidate compiled properties and trajectories checked separately',
        'alignment_outcome':'NOT_EVALUATED','learner_understanding':'NOT_ASSESSED',
        'scope':'15 declared short probes; broader reachability and task success remain unverified'})
    save(out/'artifact_hashes.json',{p.name:digest(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({'out':str(out),'task_verdicts':verdicts,'failed_dt_cases':[c['case_id'] for c in cases if not c['dt_pass']], 'resources':resource_report},ensure_ascii=False,indent=2))
    return 0 if overall else 1


if __name__=='__main__':
    raise SystemExit(main())
