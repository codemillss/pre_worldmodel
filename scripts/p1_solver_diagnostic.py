"""Bounded P1-06 solver diagnosis; does not authorize data collection.

The baseline uses the archived settled initial snapshots. Candidate transfers
permit explicitly declared physical option changes only, before any action.
The normal environment's restore guards and stepping path remain in use.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import json
import platform
import shutil
import sys
import tomllib
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import mujoco
import numpy as np

from push_align import PlanarPushEnv
from push_align.config import ROOT
from push_align.environment import SnapshotBundle
from p1_physics_pilot import digest, save, serializable, state_difference

CASES = ('yaw+10_positive_offset', 'yaw+10_negative_offset', 'yaw+0_central')
PERMITTED = {'solver_iterations', 'solver_tolerance', 'native_ccd', 'friction_cone'}
XML_OPTIONS = {'solver_iterations': 'iterations', 'solver_tolerance': 'tolerance', 'friction_cone': 'cone'}


def load_snapshot(path):
    data = json.loads(path.read_text())
    for name in ('integration_state', 'warning_counts', 'warning_lastinfo'):
        data[name] = np.asarray(data[name])
    return SnapshotBundle(**data)


def candidate_transfer(target, source, changes):
    """Explicit experiment initialization, never an action or ordinary restore."""
    if not changes <= PERMITTED:
        raise ValueError('Undeclared candidate changes')
    a, b = deepcopy(source.config), deepcopy(target.config)
    for config in (a, b):
        config.pop('config_id')
        for field in changes | {'timestep_s', 'frame_skip'}:
            config['physics'].pop(field, None)
    if a != b:
        raise ValueError('Candidate differs outside declared options')
    xml = []
    for env in (source, target):
        if env._compiled_fingerprint() != env._compiled_hash:
            raise ValueError('Mutated compiled model')
        tree = ET.fromstring(env.resolved_xml)
        for attribute in {'timestep'} | {XML_OPTIONS[k] for k in changes if k in XML_OPTIONS}:
            tree.find('option').attrib.pop(attribute, None)
        if 'native_ccd' in changes:
            flag = tree.find('option/flag')
            if flag is not None:
                flag.attrib.pop('nativeccd', None)
                if not flag.attrib:
                    tree.find('option').remove(flag)
        xml.append(ET.tostring(tree))
    if xml[0] != xml[1]:
        raise ValueError('Candidate scene differs outside declared options')
    if source.model.nq != target.model.nq or source.model.nv != target.model.nv:
        raise ValueError('Different state layout')
    snap = source.snapshot()
    mujoco.mj_setState(target.model, target.data, snap.integration_state, target._state_spec)
    target._reference_height = snap.reference_height
    target._needs_reset = snap.needs_reset
    target.data.warning.number[:] = snap.warning_counts
    target.data.warning.lastinfo[:] = snap.warning_lastinfo
    target.rng.bit_generator.state = deepcopy(snap.rng_state)
    target.data.qacc_warmstart[:] = 0.
    mujoco.mj_forward(target.model, target.data)
    target.data.qacc_warmstart[:] = 0.


def execute(env, action, ticks):
    start = env.observe()
    states, times, intervals, samples = [start.state], [0.], [], []
    for _ in range(ticks):
        result = env.step(action, record_solver=True)
        states.append(result.observation.state)
        times.append(result.observation.sim_time - start.sim_time)
        intervals.append({'valid': result.transition_valid, 'actual_dt': result.actual_dt,
                          'violations': result.diagnostics['validity_violations']})
        samples.extend(result.diagnostics['samples'][1:])
        if not result.transition_valid:
            break
    return {'states': np.asarray(states), 'times': np.asarray(times), 'intervals': intervals,
            'samples': samples, 'completed': len(intervals) == ticks and all(i['valid'] for i in intervals)}


def statistics(trace):
    solve = [s['integration_solver'] for s in trace['samples']]
    islands = [i for s in solve for i in s['islands']]
    terminal = [i['statistics'][-1] for i in islands]
    contacts = [c for s in solve for c in s['contacts']]
    return {'substeps': len(solve), 'max_iterations': max([i['iterations'] for i in islands] or [0]),
            'island_solves_at_iteration_limit': sum(i['at_iteration_limit'] for i in islands),
            'final_gradient_max': max([s['gradient'] for s in terminal] or [0.]) if terminal else None,
            'final_improvement_max': max([s['improvement'] for s in terminal] or [0.]) if terminal else None,
            'max_line_evaluations': max([s['neval'] for i in islands for s in i['statistics']] or [0.]),
            'max_contact_count': max([len(s['contacts']) for s in solve] or [0]),
            'contact_count_changes': sum(len(a['contacts']) != len(b['contacts']) for a,b in zip(solve,solve[1:])),
            'max_integration_penetration_m': max([max(0.,-c['distance_m']) for c in contacts] or [0.]),
            'max_horizontal_angular_speed_rad_s': max(np.linalg.norm(s['object_velocity_world'][:2]) for s in trace['samples']),
            'warning_count': max(sum(s['warning_counts']) for s in trace['samples']),
            'violations': sorted({v for i in trace['intervals'] for v in i['violations']})}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT/'configs/pilot.v2.toml')
    parser.add_argument('--reference-run', type=Path, default=ROOT/'runs/p1-06-20260928-05')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--solver-iterations', type=int)
    parser.add_argument('--solver-tolerance', type=float)
    parser.add_argument('--candidate-config', type=Path)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    reference = json.loads((args.reference_run/'manifest.json').read_text())
    acceptance = reference['acceptance']
    if config != reference['config']:
        raise ValueError('Reference config mismatch')
    variant = deepcopy(config)
    if args.candidate_config:
        variant = tomllib.loads(args.candidate_config.read_text())
    if args.solver_iterations is not None:
        if args.solver_iterations < 1: raise ValueError('Positive iterations required')
        variant['physics']['solver_iterations'] = args.solver_iterations
    if args.solver_tolerance is not None:
        if not np.isfinite(args.solver_tolerance) or args.solver_tolerance <= 0: raise ValueError('Positive finite tolerance required')
        variant['physics']['solver_tolerance'] = args.solver_tolerance
    changes = {k for k in config['physics'].keys() | variant['physics'].keys() if config['physics'].get(k) != variant['physics'].get(k)}
    if not changes <= PERMITTED or len(changes) > 1:
        raise ValueError('Change exactly one declared physical option per diagnostic')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    paths = [p for folder in ('src','scripts','tests','configs') for p in (ROOT/folder).rglob('*') if p.suffix in ('.py','.toml')]
    paths += [ROOT/'assets/push_scene.xml',ROOT/'requirements.lock.txt']
    source = out/'source'
    for p in paths:
        dest = source/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest)
    save(out/'manifest.json', {'created_at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        'argv':sys.argv, 'platform':platform.platform(), 'versions':{'mujoco':mujoco.__version__,'numpy':np.__version__},
        'baseline_config':config,'candidate_config':variant,'changed_physical_fields':sorted(changes),
        'reference_run':str(args.reference_run.resolve()), 'reference_manifest_sha256':digest(args.reference_run/'manifest.json'),
        'reference_snapshots':{c:digest(args.reference_run/f'{c}_snapshot.json') for c in CASES},
        'acceptance':acceptance, 'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in paths},
        'question':'declared option change vs dt sensitivity and contact geometry, using integration solve statistics',
        'geometry_timestamp':'pre-integration; observations are after integration and mj_forward',
        'comparison':'same archived physical start state, warmstart zero, common action boundaries, 3 representative cases only'})
    base=PlanarPushEnv(config)
    variants=[('base',config)] + ([('candidate',variant)] if changes else [])
    cases=[]
    ticks=round(acceptance['physics_suite']['observation_duration_s']/base.action_dt)
    limits={'object_position_m':config['task']['position_tolerance_m']*acceptance['numeric']['timestep_difference_fraction_of_task_tolerance'],
            'object_yaw_rad':config['task']['yaw_tolerance_rad']*acceptance['numeric']['timestep_difference_fraction_of_task_tolerance']}
    for case_id in CASES:
        base.restore(load_snapshot(args.reference_run/f'{case_id}_snapshot.json'))
        action=np.array([np.cos(np.deg2rad(10 if '+10_' in case_id else 0)),np.sin(np.deg2rad(10 if '+10_' in case_id else 0))])*acceptance['physics_suite']['probe_command_speed_m_s']
        variants_result={}
        for label,c in variants:
            traces=[]
            for dt_label,scale in [('dt',1.),('half',.5)]:
                candidate=deepcopy(c)
                candidate['physics']['timestep_s']*=scale
                candidate['physics']['frame_skip']=round(candidate['physics']['action_dt_s']/candidate['physics']['timestep_s'])
                env=PlanarPushEnv(candidate)
                candidate_transfer(env,base,changes if label=='candidate' else set())
                initial_match=np.array_equal(env.data.qpos,base.data.qpos) and np.array_equal(env.data.qvel,base.data.qvel) and env.data.time==base.data.time
                if not initial_match: raise ValueError('Initial state mismatch')
                save(out/f'{case_id}_{label}_{dt_label}_initial.json',asdict(env.snapshot()))
                (out/f'{label}_{dt_label}_resolved.xml').write_text(env.resolved_xml)
                trace=execute(env,action,ticks)
                traces.append(trace)
                save(out/f'{case_id}_{label}_{dt_label}.json',trace)
            comparable=all(t['completed'] for t in traces) and np.allclose(traces[0]['times'],traces[1]['times'],rtol=0,atol=1e-9)
            diff=state_difference(traces[0]['states'],traces[1]['states']) if comparable else None
            variants_result[label]={'completed':all(t['completed'] for t in traces),'dt_max_differences':diff,
                'dt_pass':bool(comparable and all(diff[k]<=v for k,v in limits.items())),
                'dt_statistics':statistics(traces[0]),'half_statistics':statistics(traces[1]),
                'dt_yaw_deg':np.rad2deg(traces[0]['states'][:,2]).tolist(), 'half_yaw_deg':np.rad2deg(traces[1]['states'][:,2]).tolist(),
                'boundary_times_s':traces[0]['times'].tolist()}
        cases.append({'case_id':case_id,'variants':variants_result})
    save(out/'summary.json',{'run_id':out.name,'changed_physical_fields':sorted(changes),'limits':limits,'cases':cases,
        'scope':'3 representative conditions; diagnostic only, not full P1 acceptance',
        'main_data_collection_authorized':False,'alignment_outcome':'NOT_EVALUATED','learner_understanding':'NOT_ASSESSED'})
    save(out/'artifact_hashes.json',{p.name:digest(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({'run_id':out.name,'changed_physical_fields':sorted(changes),'cases':cases},default=serializable,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
