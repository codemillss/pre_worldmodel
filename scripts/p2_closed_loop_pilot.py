"""P2-06 learned choice -> actual command/contact -> observation -> replan.

Every declared smoke outcome is retained. Physics failures block the phase
unless resolved; alignment success is reported separately from connection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import gc
import json
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
from time import perf_counter
import tomllib
import traceback

import numpy as np
import torch

from push_align import InitialSpec, PlanarPushEnv
from push_align.checkpoint import load_checkpoint
from push_align.closed_loop import ENGINE_GUARDS, consume_actual_step, model_planning_scope
from push_align.config import ROOT
from push_align.contact_geometry import check_contact_geometry
from push_align.data import DatasetWriter, EpisodeRecorder, contract_hash, digest, iter_dataset, make_contract
from push_align.learning_data import load_learning_data
from push_align.planning import LearnedPredictor, PlannerContext, TaskSpec, plan
from push_align.preparation import SPLIT_CODES
from push_align.task import fail_task, handoff_metrics, pose_in_goal, start_task
from p2_collect_pilot import save_trace
from p2_episode_contract_check import source_bundle
from p2_plan_pilot import DEFAULT_LEARNING, DEFAULT_TRAINING, support_diagnostic, verify_artifacts, write_trace
from p2_train_pilot import save


def case_plan(settings, config):
    cases = []
    for index, spec in enumerate(settings['cases']):
        if set(spec) != {'id', 'object_x', 'object_y', 'object_yaw_deg', 'tangent_offset_m'}:
            raise ValueError('Incomplete smoke initial case')
        if spec['id'] != f'smoke-{index:02d}':
            raise ValueError('Smoke cases must preserve declared order/IDs')
        yaw = float(np.deg2rad(spec['object_yaw_deg']))
        xy = np.array([spec['object_x'], spec['object_y']], np.float64)
        normal, tangent = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
        distance = config['object']['size_full_xyz_m'][0]/2+config['pusher']['radius_m']+config['reset']['pusher_surface_gap_m']
        pusher = xy-distance*normal+spec['tangent_offset_m']*tangent
        initial = InitialSpec(*xy, yaw, *pusher)
        if not np.isfinite([*xy, yaw, *pusher]).all():
            raise ValueError('Smoke initial values must be finite')
        cases.append({'episode_id': spec['id'], 'episode_index': index, 'family_id': f'closed-family-{index:02d}',
                      'initial_spec': asdict(initial), 'candidate_seed': settings['candidate_seed_base']+index,
                      'tangent_offset_m': spec['tangent_offset_m']})
    return cases


def compact_plan(folder, result, identity, calls, rows, train_features):
    """Keep all arrays for every plan, and a small selection record per step."""
    summary = write_trace(folder, identity, result, calls, rows,
                          support_diagnostic(result, train_features))
    return summary


def causal_prefix(folder, episode, records, snapshot, control_env, profile, intervals):
    """Compare stored learned commands with zero from the same full snapshot.

    This is post-execution ablation of a recorded prefix, never a planning input.
    It establishes a command-sequence contribution, not an optimal action.
    """
    folder.mkdir(parents=True, exist_ok=False)
    a = episode.arrays
    valid = a['transition_valid']
    prefix = 0
    for good in valid[:intervals]:
        if not good:
            break
        prefix += 1
    if not prefix:
        result = {'pass': False, 'reason': 'no_valid_recorded_prefix', 'interval_count': 0}
        save(folder/'summary.json', result)
        return result
    actions = a['action_applied'][:prefix].copy()
    results = {}
    for label, commands in [('selected_prefix', actions), ('zero_control', np.zeros_like(actions))]:
        control_env.restore(snapshot)
        trajectory = [control_env.observe().state.copy()]
        step_results = []
        geometry_ok = True
        for action in commands:
            step = control_env.step(action)
            step_results.append(step)
            trajectory.append(step.observation.state.copy())
            geometry_ok = geometry_ok and check_contact_geometry(step.diagnostics['samples'],
                                                                 profile['physics_suite']['actual_contact_penetration_max_m'])['pass']
            if not step.transition_valid or not geometry_ok:
                break
        save_trace(folder, label, snapshot, step_results)
        results[label] = {'states': np.asarray(trajectory, np.float64), 'steps': step_results,
                          'valid': len(step_results) == prefix and all(s.transition_valid for s in step_results) and geometry_ok}
    chosen, zero = results['selected_prefix'], results['zero_control']
    if not (chosen['valid'] and zero['valid']):
        result = {'pass': False, 'reason': 'prefix_replay_invalid', 'interval_count': prefix,
                  'selected_valid': chosen['valid'], 'zero_valid': zero['valid']}
        save(folder/'summary.json', result)
        return result
    truth = a['state'][:prefix+1]
    replay_xy = float(np.max(np.linalg.norm(chosen['states'][:, :2]-truth[:, :2], axis=1)))
    angle = chosen['states'][:, 2]-truth[:, 2]
    replay_yaw = float(np.max(np.abs(np.arctan2(np.sin(angle), np.cos(angle)))))
    task = episode.metadata['contract']['resolved_config']['task']
    replay_fraction = profile['numeric']['replay_tolerance_fraction_of_task_tolerance']
    state_error = chosen['states']-truth
    state_error[:, 2] = np.arctan2(np.sin(state_error[:, 2]), np.cos(state_error[:, 2]))
    scales = np.array([task['position_tolerance_m']]*2+[task['yaw_tolerance_rad']]
                      +[task['linear_speed_tolerance_m_s']]*2+[task['angular_speed_tolerance_rad_s']]
                      +[task['position_tolerance_m']]*2+[task['linear_speed_tolerance_m_s']]*2)
    replay_all_state = bool((np.max(np.abs(state_error), axis=0) <= scales*replay_fraction).all())
    replay_times = np.asarray([s.observation.sim_time for s in chosen['steps']])
    time_matches = bool(np.allclose(replay_times, a['sim_time'][1:prefix+1], rtol=0,
                                   atol=episode.metadata['contract']['dt_tolerance_s']))
    applied_matches = bool(np.array_equal(np.asarray([s.applied_action for s in chosen['steps']]), actions))
    terminal_matches = all(s.diagnostics['terminal_type'] == 'completed_interval' for s in chosen['steps'])
    replay_matches = (replay_xy <= task['position_tolerance_m']*replay_fraction
                      and replay_yaw <= task['yaw_tolerance_rad']*replay_fraction
                      and replay_all_state and time_matches and applied_matches and terminal_matches)
    initial = truth[0]
    zero_xy = float(np.max(np.linalg.norm(zero['states'][:, :2]-initial[:2], axis=1)))
    zero_angle = zero['states'][:, 2]-initial[2]
    zero_yaw = float(np.max(np.abs(np.arctan2(np.sin(zero_angle), np.cos(zero_angle)))))
    active_force = episode.metadata['contract']['active_contact_force_min_N']
    qualifying_steps = []
    first_contact = None
    for index, (record, step) in enumerate(zip(records[:prefix], chosen['steps'])):
        active = [s['sim_time'] for s in step.diagnostics['samples']
                  if any(c['pair'] == 'pusher_object' and c['normal_force_N'] > active_force for c in s['contacts'])]
        if record['source'] == 'model' and record['selected_index'] is not None and np.any(actions[index] != 0) and active:
            qualifying_steps.append(index)
            first_contact = min(active) if first_contact is None else min(first_contact, min(active))
    boundary_times = a['sim_time'][:prefix+1]
    after_contact = np.zeros(prefix+1, bool) if first_contact is None else boundary_times >= first_contact
    difference_xy = float(np.max(np.linalg.norm(chosen['states'][after_contact, :2]-zero['states'][after_contact, :2], axis=1))) if after_contact.any() else 0.
    yaw_diff = chosen['states'][after_contact, 2]-zero['states'][after_contact, 2]
    difference_yaw = float(np.max(np.abs(np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))))) if len(yaw_diff) else 0.
    numeric = profile['numeric']
    position_threshold = max(numeric['signal_to_control_drift_min_ratio']*zero_xy,
                             numeric['signal_min_fraction_of_task_tolerance']*task['position_tolerance_m'])
    yaw_threshold = max(numeric['signal_to_control_drift_min_ratio']*zero_yaw,
                        numeric['signal_min_fraction_of_task_tolerance']*task['yaw_tolerance_rad'])
    measurable = difference_xy >= position_threshold or difference_yaw >= yaw_threshold
    result = {'pass': bool(replay_matches and qualifying_steps and measurable),
              'interval_count': prefix, 'selected_prefix_replay_matches_recorded': bool(replay_matches),
              'replay_position_max_error_m': replay_xy, 'replay_yaw_max_error_rad': replay_yaw,
              'replay_all_state_within_tolerance': replay_all_state, 'replay_times_match': time_matches,
              'replay_applied_actions_match': applied_matches, 'replay_terminal_types_match': terminal_matches,
              'model_selected_active_contact_intervals': qualifying_steps, 'first_contact_sim_time': first_contact,
              'zero_control_drift_m': zero_xy, 'zero_control_drift_rad': zero_yaw,
              'position_effect_after_contact_m': difference_xy, 'yaw_effect_after_contact_rad': difference_yaw,
              'position_effect_threshold_m': position_threshold, 'yaw_effect_threshold_rad': yaw_threshold,
              'scope': 'recorded learned-selected prefix versus zero from identical engine snapshot; sequence contribution, not candidate quality'}
    save(folder/'summary.json', result)
    return result


def holding_fixture(out, env, config, task, profile):
    folder = out/'holding_fixture'
    folder.mkdir()
    initial = InitialSpec(*task.goal)
    observation = env.reset(initial)
    snapshot = env.snapshot()
    status = start_task(observation, task, env.action_dt)
    context = PlannerContext.initial()
    steps, phases = [], [status.phase]
    for _ in range(3):
        step = env.step(np.zeros(2, np.float64))
        steps.append(step)
        geometry = check_contact_geometry(step.diagnostics['samples'], profile['physics_suite']['actual_contact_penetration_max_m'])
        status, context = consume_actual_step(status, step, task, env.action_dt, context, holding=True, geometry=geometry)
        phases.append(status.phase)
    save_trace(folder, 'raw_holding', snapshot, steps)
    checks = {'ready_after_three_actual_intervals': status.phase == 'ready',
              'continuous_sampled_hold_at_least_point_three': status.held_time_s+1e-9 >= task.hold_time_s,
              'settling_excluded': abs(status.elapsed_s-.3) <= 1e-9,
              'holding_zero_clears_warm': context.optimal_actions is None and bool(np.array_equal(context.previous_applied_action, np.zeros(2)))}
    save(folder/'summary.json', {'checks': checks, 'phases': phases, 'final_task': asdict(status),
                                'scope': 'separate task-rule physics fixture started already at goal; not learned alignment success or a smoke outcome'})
    return checks


def run(args, out):
    began = perf_counter()
    settings = tomllib.loads(args.closed_loop_config.read_text())
    required = {'device': 'cpu', 'torch_threads': 1, 'deterministic_algorithms': True,
                'episode_count': 10, 'candidate_seed_base': 3711, 'candidate_speed_norm_max_m_s': .03,
                'max_episode_time_s': 30., 'action_dt_s': .1, 'max_action_intervals': 300,
                'causal_prefix_intervals': 10, 'minimum_causal_contact_traces': 1,
                'future_engine_guards': list(ENGINE_GUARDS), 'unresolved_physics_codes': ['physics_error', 'geometry_invalid']}
    if any(type(settings.get(key)) is not type(value) or settings[key] != value for key, value in required.items()) or len(settings['cases']) != 10:
        raise ValueError('Runner requires the declared P2-06 smoke configuration')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    prior_paths = (args.learning_data.parent.resolve(), args.training_run.resolve(), ROOT/'runs/p2-05-20261008-01')
    prior = {str(p): {'count': verify_artifacts(p), 'manifest_sha256': digest(p/'artifact_hashes.json')} for p in prior_paths}
    selection = json.loads((args.training_run/'selection.json').read_text())
    checkpoint = args.training_run/selection['selected_checkpoint']
    meta = json.loads((checkpoint/'checkpoint.json').read_text())
    config = meta['config']
    profile = tomllib.loads((ROOT/'configs/acceptance.v1.toml').read_text())
    if profile['closed_loop_smoke']['episode_count'] != 10 or config['task']['max_episode_time_s'] != settings['max_episode_time_s']:
        raise ValueError('Task/profile budget differs from declared smoke')
    m = json.loads((args.learning_data/'normalizer/normalizer.json').read_text())['normalizer']
    prepared, normalizer = load_learning_data(args.learning_data, m['schema'], m['dataset_identity'], meta['normalizer_hash'])
    provenance = {**meta['training_provenance'], 'learning_manifest_sha256': digest(args.learning_data/'manifest.json')}
    bundle = load_checkpoint(checkpoint, normalizer, config, provenance)
    if bundle.metadata['checkpoint_hash'] != selection['selected_checkpoint_hash']:
        raise ValueError('Selected model identity changed')
    weights = {key: tensor.clone() for key, tensor in bundle.model.state_dict().items()}
    predictor = LearnedPredictor(bundle.model, normalizer)
    planner_config = deepcopy(config)
    planner_config['planner']['command_speed_norm_max_m_s'] = settings['candidate_speed_norm_max_m_s']
    task = TaskSpec.from_config(config)
    env, control_env = PlanarPushEnv(config), PlanarPushEnv(config)
    contract = make_contract(env)
    cases = case_plan(settings, config)
    hashes = source_bundle(out)
    manifest = {'task_id': 'P2-06', 'created_at': datetime.now().astimezone().isoformat(), 'argv': sys.argv,
                'executable': sys.executable, 'python': platform.python_version(), 'platform': platform.platform(),
                'settings': settings, 'settings_sha256': digest(args.closed_loop_config), 'cases': cases,
                'checkpoint_hash': meta['checkpoint_hash'], 'weights_sha256': digest(checkpoint/'weights.pt'),
                'normalizer_hash': meta['normalizer_hash'], 'original_config': config, 'resolved_planning_config': planner_config,
                'learning_manifest_sha256': digest(args.learning_data/'manifest.json'), 'profile_sha256': digest(ROOT/'configs/acceptance.v1.toml'),
                'contract_hash': contract_hash(contract), 'environment_hash': contract['environment_hash'],
                'source_hashes': hashes, 'prior_artifacts': prior,
                'scope': settings['scope'], 'task_success_not_phase_gate': True}
    save(out/'manifest.json', manifest)
    manifest_hash = digest(out/'manifest.json')
    print(json.dumps({'progress': 'manifest_frozen', 'episodes': len(cases)}), flush=True)
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=ROOT, capture_output=True, text=True)
    (out/'tests.txt').write_text(tests.stdout+tests.stderr)
    match = re.search(r'Ran (\d+) tests', tests.stderr+tests.stdout)
    count = int(match.group(1)) if match else 0
    if tests.returncode or not count or 'skipped=' in tests.stderr:
        raise ValueError('Mandatory regression tests did not all pass')
    fixture_checks = holding_fixture(out, control_env, config, task, profile)
    if not all(fixture_checks.values()):
        raise ValueError('Actual holding fixture failed')
    writer = DatasetWriter(out/'data'/'p2-closed-loop-smoke-v1', contract, purpose='pilot')
    train_features = prepared.arrays['features'][prepared.arrays['split_code'] == SPLIT_CODES['train']]
    guards = dict.fromkeys(ENGINE_GUARDS, 0)
    ledger, planning_times, execution_times, causal = [], [], [], []
    for case in cases:
        folder = out/'episodes'/case['episode_id']
        folder.mkdir(parents=True)
        clock = perf_counter()
        recorder, records = None, []
        try:
            observation = env.reset(InitialSpec(**case['initial_spec']))
            status = start_task(observation, task, env.action_dt)
            context = PlannerContext.initial()
            rng = np.random.default_rng(case['candidate_seed'])
            recorder = EpisodeRecorder(env, episode_id=case['episode_id'], episode_index=case['episode_index'],
                                       family_id=case['family_id'], family_definition={'base_initial_spec': case['initial_spec'],
                                       'variant_rule': 'one fixed development reset; no outcome-dependent replacement'},
                                       split='smoke-test', collection_policy={'kind': 'learned_mpc_closed_loop',
                                       'purpose': 'pilot', 'closed_loop_id': settings['closed_loop_id'], 'checkpoint_hash': meta['checkpoint_hash']},
                                       seed=case['candidate_seed'], contract=contract)
            for index in range(settings['max_action_intervals']):
                if status.phase in {'ready', 'failed'}:
                    break
                holding = pose_in_goal(observation.state, task)
                planned, predicted_next, plan_info = None, None, None
                if holding:
                    action = np.zeros(2, np.float64)
                    selected_index, source = None, 'holding'
                else:
                    before_calls, before_rows = predictor.calls, predictor.rows
                    with model_planning_scope(guards):
                        planned = plan(observation.state.copy(), task, context, rng, predictor, planner_config)
                    selected_index, source = planned.selected_index, 'model'
                    identity = {'case': case, 'control_step': index, 'current_sim_time': observation.sim_time}
                    plan_info = compact_plan(folder/f'plans/step-{index:03d}', planned, identity,
                                             predictor.calls-before_calls, predictor.rows-before_rows, train_features)
                    planning_times.append(planned.diagnostics['wall_time_s'])
                    if planned.reason == 'planner_invalid':
                        status = fail_task(status, 'planner_invalid')
                        save(folder/'unexecuted_final_plan.json', {'step': index, 'reason': planned.reason, 'no_extra_zero_step': True})
                        break
                    action, context = planned.action, planned.context
                    predicted_next = planned.predictions.states[selected_index, 1].copy()
                before = observation.state.copy()
                tick = perf_counter()
                step = recorder.step(action)
                execution_s = perf_counter()-tick
                execution_times.append(execution_s)
                geometry = recorder.geometry[-1]
                status, context = consume_actual_step(status, step, task, env.action_dt, context,
                                                     holding=holding, geometry=geometry)
                observation = step.observation
                metrics = handoff_metrics(observation.state, task) if np.isfinite(observation.state).all() else None
                error = None
                if predicted_next is not None and np.isfinite(observation.state).all():
                    difference = observation.state-predicted_next
                    difference[2] = np.arctan2(np.sin(difference[2]), np.cos(difference[2]))
                    error = {'object_position_m': float(np.linalg.norm(difference[:2])),
                             'object_yaw_rad': float(abs(difference[2])), 'state_error': difference.tolist()}
                record = {'index': index, 'source': source, 'selected_index': selected_index,
                          'requested_action': action.tolist(), 'applied_action': step.applied_action.tolist(),
                          'state_before': before.tolist(), 'state_after': observation.state.tolist(),
                          'actual_dt_s': step.actual_dt, 'sim_time': observation.sim_time,
                          'task': asdict(status), 'metrics': metrics, 'geometry': geometry,
                          'physical_valid': bool(step.transition_valid), 'prediction_error': error,
                          'plan': plan_info, 'execution_wall_time_s': execution_s}
                # Strict JSON is validated before each line; raw nonfinite physics
                # is retained by EpisodeRecorder even if summary serialization fails.
                from push_align.data import _encode_nonfinite
                encoded = json.dumps(_encode_nonfinite(record), ensure_ascii=False, allow_nan=False)
                with (folder/'steps.jsonl').open('a', encoding='utf-8') as file:
                    file.write(encoded+'\n')
                records.append(record)
                if (index+1) % 25 == 0:
                    print(json.dumps({'progress': 'closed_loop', 'episode': case['episode_id'],
                                      'steps': index+1, 'elapsed_sim_s': status.elapsed_s, 'phase': status.phase}), flush=True)
            if status.phase not in {'ready', 'failed'}:
                raise ValueError('Declared execution budget ended without terminal outcome')
            if recorder.results:
                episode = recorder.finish(status.terminal_reason)
                episode.metadata['task_outcome'] = status.phase
                episode.metadata['supervised_task_status'] = asdict(status)
                writer.append(episode)
                prefix = causal_prefix(folder/'causal_prefix', episode, records, recorder.snapshots[0],
                                       control_env, profile, settings['causal_prefix_intervals'])
                causal.append({'episode_id': case['episode_id'], **prefix})
                raw_terminal = episode.metadata['terminal_reason']
                transitions, excluded = episode.transition_count, int((~episode.arrays['transition_valid']).sum())
                del episode
            else:
                raw_terminal, transitions, excluded = None, 0, 0
                save_trace(folder, 'empty_execution', recorder.snapshots[0], [])
                save(folder/'empty_execution.json', {'terminal_reason': status.terminal_reason,
                                                   'scope': 'finished task without physical interval; not a dataset episode'})
                causal.append({'episode_id': case['episode_id'], 'pass': False, 'reason': 'no_executed_interval'})
            entry = {'episode_id': case['episode_id'], 'status': 'completed_execution', 'task': asdict(status),
                     'raw_terminal_reason': raw_terminal, 'transitions': transitions, 'excluded_transitions': excluded,
                     'model_selected_applied_intervals': sum(r['source']=='model' for r in records),
                     'holding_intervals': sum(r['source']=='holding' for r in records),
                     'final_metrics': handoff_metrics(observation.state, task) if np.isfinite(observation.state).all() else None,
                     'wall_time_s': perf_counter()-clock}
            save(folder/'summary.json', entry)
            ledger.append(entry)
            print(json.dumps({'progress': 'episode_complete', 'episode': case['episode_id'],
                              'reason': status.terminal_reason, 'steps': transitions,
                              'causal_prefix_pass': causal[-1]['pass']}), flush=True)
            del recorder, records
            gc.collect()
        except Exception as error:
            save(folder/'failure.json', {'case': case, 'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
            if recorder is not None:
                save_trace(folder, 'failed_recorded_prefix', recorder.snapshots[0], recorder.results)
            ledger.append({'episode_id': case['episode_id'], 'status': 'infrastructure_error', 'error': str(error)})
            print(json.dumps({'progress': 'episode_error', 'episode': case['episode_id'], 'error': str(error)}), flush=True)
    save(out/'ledger.json', ledger)
    save(out/'causal_summary.json', causal)
    completed_dataset = writer.finish() if writer.entries else None
    reviews = []
    if completed_dataset is not None:
        for episode in iter_dataset(writer.path, contract):
            reviews.append({'episode_id': episode.metadata['episode_id'], 'transitions': episode.transition_count,
                            'valid': int(episode.arrays['transition_valid'].sum())})
            del episode
    save(out/'dataset_review.json', {'episodes': reviews, 'empty_executions_are_separate': True})
    codes = Counter(row['task']['terminal_reason'] for row in ledger if 'task' in row)
    unresolved = [row['episode_id'] for row in ledger if row['status'] != 'completed_execution'
                  or row.get('task', {}).get('terminal_reason') in settings['unresolved_physics_codes']]
    prior_after = {str(p): {'count': verify_artifacts(p), 'manifest_sha256': digest(p/'artifact_hashes.json')} for p in prior_paths}
    checks = {'regression_tests_pass': tests.returncode == 0 and count > 0,
              'all_ten_declared_executions_accounted': len(ledger) == 10 and [r['episode_id'] for r in ledger] == [r['episode_id'] for r in cases],
              'no_unfinished_infrastructure_or_unresolved_physics': not unresolved,
              'at_least_one_controlled_causal_contact_prefix': sum(r['pass'] for r in causal) >= settings['minimum_causal_contact_traces'],
              'model_used_to_choose_applied_commands': predictor.calls > 0 and any(r.get('model_selected_applied_intervals',0)>0 for r in ledger),
              'future_engine_calls_in_planning_zero': sum(guards.values()) == 0,
              'every_recorded_episode_audited': len(reviews) == len(writer.entries),
              'manifest_frozen': digest(out/'manifest.json') == manifest_hash,
              'model_weights_unchanged': all(torch.equal(value, bundle.model.state_dict()[key]) for key,value in weights.items()),
              'source_frozen_during_run': all(digest(ROOT/name) == sha for name,sha in hashes.items()),
              'prior_artifacts_and_manifests_unchanged': prior_after == prior,
              **fixture_checks}
    p = np.asarray(planning_times)
    wall = {'total_wall_time_s': perf_counter()-began,
            'main_peak_rss_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024),
            'plan_count': len(p), 'planning_mean_s': float(p.mean()) if len(p) else None,
            'planning_p95_s': float(np.percentile(p,95)) if len(p) else None,
            'planning_max_s': float(p.max()) if len(p) else None,
            'execution_intervals': len(execution_times), 'execution_mean_s': float(np.mean(execution_times)) if execution_times else None,
            'scope': 'local CPU one thread; recorder/validation included in overall time; test subprocess memory excluded; not physical-device real-time validation'}
    save(out/'resources.json', wall)
    save(out/'summary.json', {'task_id':'P2-06', 'pass':all(checks.values()), 'checks':checks,
                             'test_count':count, 'declared_episodes':10, 'ledger':ledger, 'terminal_counts':dict(codes),
                             'unresolved_items':unresolved, 'causal_prefix_pass_count':sum(r['pass'] for r in causal),
                             'dataset_recorded_episodes':len(reviews), 'dataset_transitions':sum(r['transitions'] for r in reviews),
                             'future_engine_calls_during_planning':guards, 'actual_model_calls':predictor.calls,
                             'resources':wall, 'alignment_ready_count':codes.get('ready',0),
                             'alignment_outcome':'CASE_DEMONSTRATED' if codes.get('ready',0) else 'NOT_DEMONSTRATED',
                             'performance_scope':'ten fixed development smoke initial conditions, not final/generalization success rate',
                             'user_understanding':'NOT_ASSESSED'})
    save(out/'artifact_hashes.json', {'sha256':{str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    if not all(checks.values()):
        raise ValueError('P2-06 phase gate not met; preserve all outcomes and diagnose before DONE')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--learning-data',type=Path,default=DEFAULT_LEARNING)
    parser.add_argument('--training-run',type=Path,default=DEFAULT_TRAINING)
    parser.add_argument('--closed-loop-config',type=Path,default=ROOT/'configs/p2_closed_loop.v1.toml')
    args=parser.parse_args()
    if not re.fullmatch(r'p2-06-[0-9]{8}-[0-9]{2}',args.run_id):raise ValueError('Invalid run ID')
    out=ROOT/'runs'/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    try:run(args,out)
    except Exception as error:
        save(out/'failure.json',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        raise
    print(json.dumps({'run_id':args.run_id,'pass':True}),flush=True)


if __name__=='__main__':main()
