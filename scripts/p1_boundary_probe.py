"""Paired timestep probes at eight declared initial-range corners.

The 24 short cases assess selected boundary poses, not the continuous range,
30-second alignment, or general reachability of the fixed goal.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import gzip
import itertools
import json
from pathlib import Path
import platform
import sys
import tomllib
import traceback

import mujoco
import numpy as np

from push_align import InitialSpec, PlanarPushEnv
from push_align.config import ROOT
from push_align.environment import task_angle_error, wrap_angle
from p1_contact_probe import condition_initial
from p1_physics_pilot import (compiled_matches_config, compiled_properties,
                              contact_geometry_check, digest, execute,
                              replay_limits, save, serializable, state_difference)


def raw_save(path, value):
    with gzip.open(path, 'xt') as file:
        json.dump(value, file, default=serializable, allow_nan=False, separators=(',', ':'))


def conditions(config):
    ranges = config['initial_conditions']
    bounds = [ranges[key] for key in ('object_x_range_m', 'object_y_range_m', 'object_yaw_range_rad')]
    if any(len(b)!=2 or not np.isfinite(b).all() or b[0]>=b[1] for b in bounds):
        raise ValueError('Three finite, ordered and nondegenerate initial ranges are required')
    cases = []
    for corner, (x, y, yaw) in enumerate(itertools.product(*bounds), start=1):
        for condition in ('central', 'positive_offset', 'negative_offset'):
            cases.append({'case_id': f'corner{corner:02d}_{condition}', 'corner': corner,
                          'object_x_m': float(x), 'object_y_m': float(y),
                          'object_yaw_rad': float(yaw), 'condition': condition})
    return cases


def shifted_initial(case, env, offset):
    centered = condition_initial(case['object_yaw_rad'], case['condition'], env, offset)
    return InitialSpec(object_x=case['object_x_m'], object_y=case['object_y_m'],
                       object_yaw=case['object_yaw_rad'],
                       pusher_x=centered.pusher_x+case['object_x_m'],
                       pusher_y=centered.pusher_y+case['object_y_m'])


def initial_goal_geometry(config, observation):
    s = observation.state
    inward = np.array([np.cos(s[2]), np.sin(s[2])])
    tangent = np.array([-inward[1], inward[0]])
    delta = np.asarray(config['task']['goal_xy_m']) - s[:2]
    projection = float(delta @ inward)
    return {'fixed_goal_xy_m': config['task']['goal_xy_m'],
            'settled_object_xy_m': s[:2], 'settled_object_yaw_rad': float(s[2]),
            'initial_rear_face_inward_normal_world_xy': inward,
            'goal_forward_halfspace_signed_projection_m': projection,
            'goal_lateral_projection_m': float(delta @ tangent),
            'goal_is_in_initial_forward_halfspace': projection>0.,
            'initial_task_yaw_error_rad': float(task_angle_error(config['task']['goal_yaw_rad']-s[2])),
            'scope': 'Positive projection supports forward progress along this initial inward normal. It is not sufficient for alignment and is not a necessary condition for a controller allowed to recontact other faces.',
            'full_policy_reachability': 'NOT_ASSESSED'}


def warning_count(trace):
    return max([sum(s['warning_counts']) for s in trace['samples']] or [0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/pilot.v4.toml')
    parser.add_argument('--acceptance', type=Path, default=ROOT/'configs/acceptance.v1.toml')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    with args.config.open('rb') as file:
        config = tomllib.load(file)
    with args.acceptance.open('rb') as file:
        acceptance = tomllib.load(file)
    suite, numeric = acceptance['physics_suite'], acceptance['numeric']
    if not suite.get('check_contact_geometry',False):
        raise ValueError('Boundary prerequisite requires an explicit contact-geometry profile')
    # These are the deliberately small probes authorized for this check.
    expected = [config['physics']['timestep_s'], config['physics']['action_dt_s'],
                suite['physics_dt_comparison_scale'], suite['observation_duration_s'],
                suite['probe_command_speed_m_s'], suite['probe_offset_abs_m']]
    if not np.allclose(expected, [.0005, .1, .5, 1., .03, .015], rtol=0, atol=1e-15):
        raise ValueError('This diagnostic specifies 0.5/0.25 ms, 0.1 s actions, 1 s duration, 0.03 m/s and 0.015 m offsets')
    cases = conditions(config)
    finer_config = deepcopy(config)
    finer_config['physics']['timestep_s'] *= suite['physics_dt_comparison_scale']
    finer_config['physics']['frame_skip'] = round(config['physics']['action_dt_s']/finer_config['physics']['timestep_s'])
    ticks = round(suite['observation_duration_s']/config['physics']['action_dt_s'])
    source_paths = [ROOT/'scripts/p1_boundary_probe.py', ROOT/'scripts/p1_physics_pilot.py',
                    ROOT/'scripts/p1_contact_probe.py', ROOT/'src/push_align/environment.py',
                    ROOT/'src/push_align/config.py', ROOT/'src/push_align/__init__.py',
                    ROOT/'assets/push_scene.xml', ROOT/'requirements.lock.txt',
                    args.config.resolve(), args.acceptance.resolve()]
    source_hashes = {str(p):digest(p) for p in source_paths}
    dt_limits = replay_limits(config, numeric['timestep_difference_fraction_of_task_tolerance'])
    save(out/'manifest.json', {'created_at': datetime.now().astimezone().isoformat(),
        'argv':sys.argv, 'executable':sys.executable, 'platform':platform.platform(),
        'versions':{'mujoco':mujoco.__version__, 'numpy':np.__version__},
        'config':config, 'finer_config':finer_config, 'acceptance':acceptance,
        'predeclared_cases':cases, 'case_count':len(cases), 'dt_limits':dt_limits,
        'source_hashes':source_hashes,
        'comparison_rule':'Identical settled integration state; zero solver warmstart in both dt branches; max object XY and wrapped yaw difference at every shared 0.1 s boundary; both branches must complete 1 s without physical or geometry failure and encounter pusher contact above the existing active-force threshold.',
        'scope':'24 short probes at eight range corners. Continuous-range validity, 30 s episodes and goal alignment are not assessed.'})
    env, finer = PlanarPushEnv(config), PlanarPushEnv(finer_config)
    with (out/'resolved_base.xml').open('x') as file:
        file.write(env.resolved_xml)
    with (out/'resolved_half.xml').open('x') as file:
        file.write(finer.resolved_xml)
    compiled_checks = [compiled_matches_config(e) for e in (env,finer)]
    save(out/'compiled.json', {'properties':[compiled_properties(e) for e in (env,finer)],
                               'checks':compiled_checks})
    results, all_geometry = [], []
    for case in cases:
        phase = 'reset'
        try:
            initial = shifted_initial(case, env, suite['probe_offset_abs_m'])
            env.reset(initial)
            reset_record = deepcopy(env.initial_record)
            phase = 'diagnostic_state_transfer'
            finer.import_dt_comparison_state(env)
            env.import_dt_comparison_state(env)
            snapshot = env.snapshot()
            initial_observation = env.observe()
            initial_match = bool(np.array_equal(env.data.qpos,finer.data.qpos)
                and np.array_equal(env.data.qvel,finer.data.qvel)
                and np.array_equal(env.data.ctrl,finer.data.ctrl)
                and env.data.time==finer.data.time
                and np.array_equal(env.data.qacc_warmstart,finer.data.qacc_warmstart)
                and np.all(env.data.qacc_warmstart==0.))
            raw_save(out/f"{case['case_id']}_initial.json.gz", {
                'requested_initial':asdict(initial), 'reset_record':reset_record,
                'snapshot':asdict(snapshot), 'identical_physical_initial_state':initial_match,
                'goal_geometry':initial_goal_geometry(config, initial_observation)})
            yaw = case['object_yaw_rad']
            action = suite['probe_command_speed_m_s']*np.array([np.cos(yaw),np.sin(yaw)])
            phase = 'execute_base'
            base = execute(env,action,ticks)
            raw_save(out/f"{case['case_id']}_base.json.gz",base)
            phase = 'execute_half'
            half = execute(finer,action,ticks)
            raw_save(out/f"{case['case_id']}_half.json.gz",half)
            phase = 'compare'
            expected_times = np.arange(ticks+1)*env.action_dt
            boundaries_aligned = (base['states'].shape==half['states'].shape==(ticks+1,10)
                and np.allclose(base['times'],expected_times,rtol=0,atol=config['validity']['dt_match_tolerance_s'])
                and np.allclose(half['times'],expected_times,rtol=0,atol=config['validity']['dt_match_tolerance_s']))
            comparable = base['completed'] and half['completed'] and initial_match and boundaries_aligned
            difference = state_difference(base['states'],half['states']) if comparable else None
            final_difference = state_difference(base['states'][-1:],half['states'][-1:]) if comparable else None
            boundary_errors = []
            if comparable:
                boundary_errors = [{'time_s':float(t),'object_position_m':float(np.linalg.norm(a[:2]-b[:2])),
                                    'object_yaw_rad':float(abs(wrap_angle(a[2]-b[2])))}
                                   for t,a,b in zip(expected_times,base['states'],half['states'])]
            geometry = [contact_geometry_check(case['case_id'],label,trace,suite['actual_contact_penetration_max_m'])
                        for label,trace in [('base',base),('half',half)]]
            all_geometry.extend(geometry)
            warnings = {'base':warning_count(base),'half':warning_count(half)}
            active_force = config['evaluation']['contact_active_normal_force_min_N']
            engaged = {label:any(c['pair']=='pusher_object' and c['normal_force_N']>active_force
                        for sample in trace['samples'] for c in sample['contacts'])
                       for label,trace in [('base',base),('half',half)]}
            dt_pass = bool(comparable and difference['object_position_m']<=dt_limits['object_position_m']
                           and difference['object_yaw_rad']<=dt_limits['object_yaw_rad'])
            # The yaw tolerance is a matched-time angular difference budget.
            # Tiny drift in a central probe does not impose a rotation sign.
            dt_validity = bool(base['completed'] and half['completed'] and all(v==0 for v in warnings.values()))
            geometry_pass = all(g['pass'] for g in geometry)
            result = {**case,'identical_physical_initial_state':initial_match,
                'boundaries_aligned':bool(boundaries_aligned),'base_completed':base['completed'],
                'half_completed':half['completed'],'base_terminal_type':base['terminal_type'],
                'half_terminal_type':half['terminal_type'],'base_duration_s':float(base['times'][-1]),
                'half_duration_s':float(half['times'][-1]),'warnings':warnings,
                'active_pusher_contact_seen':engaged,'contact_active_force_threshold_N':active_force,
                'dt_max_differences':difference,'dt_final_differences':final_difference,
                'boundary_errors':boundary_errors,'dt_pass':dt_pass,'physical_validity_pass':dt_validity,
                'geometry_pass':geometry_pass,'goal_geometry':initial_goal_geometry(config,initial_observation),
                'validity_violations':[i['violations'] for trace in (base,half) for i in trace['intervals'] if not i['valid']],
                'pass':bool(initial_match and dt_pass and dt_validity and geometry_pass and all(engaged.values()))}
            results.append(result)
        except Exception as error:
            failure = {**case,'pass':False,'phase':phase,'error_type':type(error).__name__,
                       'error':str(error),'traceback':traceback.format_exc()}
            raw_save(out/f"{case['case_id']}_error.json.gz",failure)
            results.append(failure)
    sources_unchanged = all(digest(Path(p))==h for p,h in source_hashes.items())
    compiled_pass = all(all(check.values()) for check in compiled_checks)
    complete_coverage = len(results)==len(cases)==24 and len(all_geometry)==48
    overall = sources_unchanged and compiled_pass and complete_coverage and all(c['pass'] for c in results)
    comparable_results = [c for c in results if c.get('dt_max_differences') is not None]
    summary = {'run_id':out.name,'boundary_probe_verdict':'PASS' if overall else 'FAIL',
        'predeclared_cases':len(cases),'recorded_cases':len(results),'passed_cases':sum(c['pass'] for c in results),
        'compiled_configuration_pass':compiled_pass,'complete_geometry_coverage':complete_coverage,
        'sources_unchanged_during_run':sources_unchanged,'dt_limits':dt_limits,
        'max_object_position_difference_m':max([c['dt_max_differences']['object_position_m'] for c in comparable_results],default=None),
        'max_object_yaw_difference_rad':max([c['dt_max_differences']['object_yaw_rad'] for c in comparable_results],default=None),
        'cases':results,'contact_geometry_checks':all_geometry,
        'main_data_collection_authorized':False,'continuous_initial_range_validity':'NOT_ASSESSED',
        'thirty_second_episode_validity':'NOT_ASSESSED','alignment_goal_reachability':'NOT_ASSESSED',
        'task_alignment_success':'NOT_EVALUATED','learner_understanding':'NOT_ASSESSED',
        'scope':'Selected eight initial-range corners, three straight 1 s pushes per corner, two dt values. This test does not demonstrate uniform validity inside the range or full task success.'}
    save(out/'summary.json',summary)
    save(out/'artifact_hashes.json',{p.name:digest(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({'out':str(out),'boundary_probe_verdict':summary['boundary_probe_verdict'],
        'passed_cases':summary['passed_cases'],'total_cases':summary['recorded_cases'],
        'max_object_position_difference_m':summary['max_object_position_difference_m'],
        'max_object_yaw_difference_rad':summary['max_object_yaw_difference_rad'],
        'failed_cases':[c['case_id'] for c in results if not c['pass']]},ensure_ascii=False))
    return 0 if overall else 1


if __name__=='__main__':
    raise SystemExit(main())
