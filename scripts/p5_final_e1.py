"""Frozen final E1:84 model choices, then576 shared physical candidate replays.

Importing this module performs no model inference, physics or final generation.
The coordinator calls run_e1 only after final E2 G records exist. A missing fixed
G tick blocks completion without replacement. Each physical repeat is guarded,
exclusive and attempts to retain the entire returned raw prefix, including
failures. A hard-reserve or I/O block remains an explicit incomplete record.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
from time import perf_counter
import traceback
from unittest.mock import patch

import numpy as np

from push_align import final_recording_columns as raw_codec
from push_align.candidate_diagnostics import compare_candidates
from push_align.candidate_replay import replay_candidate
from push_align.candidates import generate_candidates
from push_align.closed_loop import ENGINE_GUARDS, model_planning_scope
from push_align.config import ROOT
from push_align.environment import Observation, PlanarPushEnv, SnapshotBundle
from push_align.final_driver import exact_comparison, read_full_trace, typed_digest
from push_align.final_manifest import CONTROLLERS, expected_E1, frozen_source_contract
from push_align.final_recording import file_sha256, read_json_gzip
from push_align.geometry_control import GeometryContext
from push_align.planning import LearnedPredictor, PlannerContext, TaskSpec, choose, contact_distance, rollout, score
from push_align.planning_protocol import GeometryProposalAdapter, keyed_planner_rng
from push_align.reproduction import json_safe
from push_align.task import TaskStatus

MODELS = tuple(CONTROLLERS[1:])
CANDIDATE_SEED = 6811
CANDIDATE_STREAM = 1
N, H, REPEATS = 16, 5, 3
COMPONENT_NAMES = ('stage_position', 'stage_yaw', 'stage_action', 'stage_action_change', 'stage_gap',
                   'terminal_position', 'terminal_yaw', 'terminal_speed', 'terminal_angular_speed')


class E1Stop(RuntimeError):
    pass


def _read(path):
    return json.loads(Path(path).read_text())


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(json_safe(value), stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def _save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    return {'path': str(path), 'sha256': file_sha256(path), 'bytes': path.stat().st_size}


def _check_bindings(bindings):
    if not bindings:
        raise E1Stop('Nonempty frozen source bindings required')
    changed = [str(path) for path, expected in bindings.items()
               if not Path(path).is_file() or file_sha256(path) != expected]
    if changed:
        raise E1Stop(f'Frozen source/original changed: {changed[:5]}')
    return {'count': len(bindings), 'all_unchanged': True}


def _notify(callback, event, **data):
    """Callback gets one dict; returning None/True means checks passed.

    Dict statuses CONTINUE/CONTINUE_RESOURCE_CHECK_ONLY/FIT_ESTIMATE_ONLY also
    continue. False, any other status, or an exception stops before new work.
    repeat_complete consumes reserved_raw_records=1/reserved_intervals=5 even
    when physical invalidity naturally shortened the actual returned prefix.
    before_raw_write(must_preserve=True) uses the already admitted reservation.
    If it stops, preserve_raw_write asks once more to retain only that prefix;
    both preservation callbacks must enforce the hard free-space reserve.
    """
    result = callback({'stage': 'E1', 'event': event, **data})
    allowed = result is None or result is True or isinstance(result, dict) and result.get('status') in {
        'CONTINUE', 'CONTINUE_RESOURCE_CHECK_ONLY', 'FIT_ESTIMATE_ONLY'}
    if not allowed:
        raise E1Stop(f'Resource callback stopped {event}: {result}')
    return result


def _safe_snapshot_folder(stage, snapshot_id):
    folder = stage/'snapshots'/snapshot_id
    if Path(snapshot_id).is_absolute() or '..' in Path(snapshot_id).parts or not folder.resolve().is_relative_to(stage.resolve()):
        raise E1Stop('Unsafe snapshot ID')
    return folder


def _frozen_bindings(manifest):
    bindings = {}
    for name in ('source_sha256', 'runtime_source_sha256'):
        if type(manifest.get(name)) is not dict or not manifest[name]:
            raise E1Stop(f'Frozen manifest requires {name}')
        for relative, expected in manifest[name].items():
            path = (ROOT/relative).resolve()
            if Path(relative).is_absolute() or '..' in Path(relative).parts or not path.is_relative_to(ROOT.resolve()):
                raise E1Stop('Frozen runtime source path escapes project')
            if str(path) in bindings and bindings[str(path)] != expected:
                raise E1Stop("Conflicting frozen/runtime source hashes")
            bindings[str(path)] = expected
    return bindings


def _verify_completed_failed_G(source, saved, manifest, bindings):
    """Accept a failure prefix only through its sealed independent resolution.

    This mirrors the coordinator's deep adoption checks without importing the
    coordinator (which imports this writer). It never changes STOP_REVIEW or
    substitutes an unavailable predecision tick, and never executes a model or
    physical engine.
    """
    from push_align.final_result_review import review_recording
    try:
        folder = source.parent.resolve()
        worker_path = folder/'worker_result.json'
        worker = _read(worker_path)
        receipt = _read(folder/'worker_result_receipt.json')
        record_sha = file_sha256(source)
        matches = [cell for cell in manifest.get('E2', [])
                   if cell.get('family_id') == folder.parent.name and cell.get('controller_id') == 'G']
        if (len(matches) != 1 or file_sha256(worker_path) != receipt.get('sha256')
                or worker.get('phase') != 'E2' or worker.get('cell') != matches[0]
                or worker.get('status') != 'ACCEPT_COMPLETED_TASK_FAILURE'
                or worker.get('recording_status') != 'STOP_REVIEW'
                or Path(worker['record_manifest']).resolve() != source.resolve()
                or worker.get('record_manifest_sha256') != record_sha):
            raise E1Stop('G failure is not the sealed completed E2 identity')
        row = worker['row']
        calls = row.get('future_engine_calls')
        if (any(row.get(key) != value for key, value in matches[0].items())
                or row.get('execution_status') != 'completed'
                or not isinstance(calls, dict) or not calls
                or any(type(value) is not int or value != 0 for value in calls.values())
                or any(row.get(key) != saved['execution'].get(key) for key in
                       ('execution_status', 'termination', 'counts', 'future_engine_calls'))
                or saved['identity'].get('family_id') != matches[0]['family_id']
                or saved['identity'].get('controller_id') != 'G'
                or any(saved['source_bindings'].get(path) != digest for path, digest in bindings.items())):
            raise E1Stop('G failure has incomplete execution, oracle access or different bindings')
        _check_bindings(saved['source_bindings'])
        review_path = folder/'review.json'
        if file_sha256(review_path) != worker.get('review_sha256'):
            raise E1Stop('G independent failure review changed')
        review = _read(review_path)
        if (review.get('accepted') is not True or review.get('status') != worker['status']
                or review.get('manifest_sha256') != record_sha
                or not isinstance(review.get('metrics'), dict)
                or any(row.get(key) != value for key, value in review['metrics'].items())):
            raise E1Stop('G failure review does not match its sealed result')
        again = review_recording(source, physical_config=manifest['physical_config'], profile=manifest['profile'],
            expected_manifest_sha256=record_sha, expected_identity=saved['identity'],
            expected_source_bindings=saved['source_bindings'], expected_run_scope='final_evaluation')
        if (again.get('accepted') is not True or again.get('status') != 'ACCEPT_COMPLETED_TASK_FAILURE'
                or again.get('manifest_sha256') != record_sha or again.get('metrics') != review['metrics']):
            raise E1Stop('G completed task failure does not regenerate from independent saved-raw review')
    except E1Stop:
        raise
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise E1Stop('Missing or malformed completed G failure evidence') from error


def _load_inputs(run, entries, manifest, bindings):
    inputs, missing, originals = [], [], {}
    for spec in entries:
        source = run/'E2'/spec['family_id']/'G'/'manifest.json'
        if not source.is_file():
            missing.append({'snapshot_id': spec['snapshot_id'], 'reason': 'missing_G_episode_manifest', 'path': str(source)})
            continue
        key = str(source)
        if key not in originals:
            saved = _read(source)
            if (saved.get('schema') != 'p5-recorded-episode-v1'
                    or saved.get('record_kind') != 'reference_controller_execution'
                    or saved.get('recording_status') not in {'COMPLETE_RECORDED', 'STOP_REVIEW'}
                    or saved.get('execution', {}).get('controller_id') != 'G'
                    or saved.get('execution', {}).get('run_scope') != 'final_evaluation'):
                raise E1Stop('E1 inputs require completed original final G episodes')
            if saved['recording_status'] == 'STOP_REVIEW':
                _verify_completed_failed_G(source, saved, manifest, bindings)
            _check_bindings(saved['files_sha256'])
            originals[key] = saved
        saved = originals[key]
        traces = [row for row in saved['full_traces'] if row['tick'] == spec['tick']]
        if len(traces) != 1:
            missing.append({'snapshot_id': spec['snapshot_id'], 'reason': 'missing_or_duplicate_exact_G_tick',
                            'tick': spec['tick'], 'available_ticks': [row['tick'] for row in saved['full_traces']]})
            continue
        capture, arrays, metadata = read_full_trace(traces[0])
        required = {'snapshot': SnapshotBundle, 'observation': Observation, 'task_status': TaskStatus,
                    'planner_context': PlannerContext, 'geometry_context': GeometryContext, 'task_spec': TaskSpec}
        if any(not isinstance(capture.get(name), cls) for name, cls in required.items()):
            raise E1Stop('Missing typed G predecision physical/task/planner input')
        if (arrays or metadata['plan'] is not None or capture['tick'] != spec['tick']
                or capture['episode_index'] != spec['episode_index'] or capture['controller_id'] != 'G'
                or capture.get('case_id') != spec['family_id']
                or capture['planner_context'].control_step != spec['tick']
                or capture['observation'].observation_valid is not True
                or capture['task_status'].last_sim_time != capture['observation'].sim_time
                or not np.isfinite(capture['observation'].state).all()):
            raise E1Stop('G snapshot identity/time/schema/context mismatch')
        if (capture['physical_config'] != manifest['physical_config']
                or capture['planner_config'] != manifest['planner_config']
                or exact_comparison(capture['task_spec'], TaskSpec.from_config(manifest['physical_config']))['status'] != 'PASS_EXACT'):
            raise E1Stop('G input differs from frozen task/physical/planner contract')
        if any(capture.get('source_bindings', {}).get(path) != expected for path, expected in bindings.items()):
            raise E1Stop('G input lacks the final frozen runtime/source binding')
        witness = read_json_gzip(saved['witness_path'], expected_sha256=saved['files_sha256'][saved['witness_path']])
        decisions = [row for row in witness['decisions'] if row['index'] == spec['tick']]
        if len(decisions) != 1 or decisions[0]['state'] != capture['observation'].state.tolist():
            raise E1Stop('Captured G input lacks its actual predecision witness')
        # This is the explicit E1 intervention, shared by every model. The
        # original actual previous command is retained; no warm sequence used.
        shared_context = PlannerContext(capture['planner_context'].previous_applied_action.copy(), None, spec['tick'])
        inputs.append({'spec': deepcopy(spec), 'capture': capture, 'shared_context': shared_context,
            'source_manifest': str(source), 'source_manifest_sha256': file_sha256(source),
            'full_input_receipt': traces[0], 'G_decision': decisions[0]})
    return inputs, missing


def _load_runtime(manifest):
    # Loading validates the same original data/normalizer and seven checkpoints;
    # no fit, gradient or optimizer update exists in this E1 path.
    import torch
    from push_align.learning_data import load_learning_data
    from p4_compare_candidates import load_models
    contract = frozen_source_contract(ROOT, source_archives=manifest.get('source_archives'))
    for key in ('model_identities', 'normalizer_identity', 'physical_config', 'planner_config', 'profile'):
        if contract[key] != manifest[key]:
            raise E1Stop(f'Frozen original model/config binding changed: {key}')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    normalizer_folder = ROOT/manifest['normalizer_identity']['path']
    metadata = _read(normalizer_folder/'normalizer.json')['normalizer']
    prepared, normalizer = load_learning_data(normalizer_folder.parent, metadata['schema'],
        metadata['dataset_identity'], metadata['normalizer_hash'])
    del prepared
    gc.collect()
    models, identities = load_models(normalizer, manifest['physical_config'],
                                     _read(ROOT/'configs/p4_development_e1.v1.json'))
    if tuple(models) != MODELS:
        raise E1Stop('Original seven-model order changed')
    for name, identity in identities.items():
        if any(identity[key] != manifest['model_identities'][name][key] for key in ('path', 'model_hash')):
            raise E1Stop('Loaded model is not the frozen instance')
    return {'models': models, 'normalizer': normalizer,
            'geometry': _read(ROOT/'configs/p3_geometry_selected.v2.json'),
            'transition': _read(ROOT/'configs/p3_geometry_transition.v2.json')}


def _weight_hashes(runtime):
    from p4_train_forks import tree_hash
    return {name: tree_hash(bundle.model.state_dict()) for name, bundle in runtime['models'].items()}


def _freeze_predictions(stage, inputs, manifest, runtime, callback):
    physical, planner = manifest['physical_config'], deepcopy(manifest['planner_config'])
    planner['planner']['candidate_count'] = N
    if planner['planner']['horizon'] != H or planner['planner']['block_lengths'] != [2, 3]:
        raise E1Stop('Frozen diagnostic N16/H5/[2,3] required')
    guards = dict.fromkeys(ENGINE_GUARDS, 0)
    adapter = GeometryProposalAdapter(physical, runtime['geometry'], runtime['transition'])
    choices, prediction_files = [], {}
    for data in inputs:
        spec, capture, context = data['spec'], data['capture'], data['shared_context']
        folder = _safe_snapshot_folder(stage, spec['snapshot_id'])
        folder.mkdir(parents=True, exist_ok=False)
        state, task = capture['observation'].state, capture['task_spec']
        _notify(callback, 'before_prediction', snapshot_id=spec['snapshot_id'],
                expected_prediction_sets=84, completed_prediction_sets=len(choices))
        with model_planning_scope(guards):
            proposal = adapter.propose(state.copy(), task, capture['geometry_context'],
                capture['task_status'].elapsed_s, control_tick=spec['tick'])
            if json_safe(proposal.diagnostics) != data['G_decision']['geometry_proposal']:
                raise E1Stop('Current-state G proposal differs from actual source witness')
            rng = keyed_planner_rng(CANDIDATE_SEED, spec['episode_index'], spec['tick'], CANDIDATE_STREAM)
            rng_before = deepcopy(rng.bit_generator.state)
            candidates = generate_candidates(planner, rng, previous_actions=None,
                geometry_action=proposal.geometry_action, tangent_direction=proposal.tangent_direction)
            independent = generate_candidates(planner,
                keyed_planner_rng(CANDIDATE_SEED, spec['episode_index'], spec['tick'], CANDIDATE_STREAM),
                previous_actions=None, geometry_action=proposal.geometry_action, tangent_direction=proposal.tangent_direction)
            if exact_comparison(candidates.actions, independent.actions)['status'] != 'PASS_EXACT':
                raise E1Stop('Candidate keyed-RNG regeneration changed')
            if candidates.diagnostics['random_group_counts'] != {'global': 4, 'local_geometry': 2, 'local_warm': 2}:
                raise E1Stop('N16 reserved/global/local allocation changed')
            input_hash = typed_digest({'full_actual_G_capture': capture, 'shared_context': context})
            _save_npz(folder/'candidates.npz', actions=candidates.actions, sources=np.asarray(candidates.sources))
            _save(folder/'input.json', {'spec': spec, 'source_manifest': data['source_manifest'],
                'source_manifest_sha256': data['source_manifest_sha256'], 'source_full_input': data['full_input_receipt'],
                'input_hash': input_hash, 'previous_applied_action': context.previous_applied_action,
                'warm_sequence': None, 'candidate_rng_before': rng_before, 'candidate_rng_after': rng.bit_generator.state,
                'candidate_hash': typed_digest(candidates.actions), 'generation': candidates.diagnostics,
                'observed_signed_geometric_gap_m': float(contact_distance(state[None], planner)[0]),
                'regime_scope': 'Current observed geometry only; no assumption of actual future contact/force.'})
            entries = []
            for name in MODELS:
                predictor = LearnedPredictor(runtime['models'][name].model, runtime['normalizer'])
                predicted = rollout(np.broadcast_to(state, (N, 10)).copy(), candidates.actions, predictor, planner)
                costs = score(predicted, candidates.actions, task, context.previous_applied_action, planner)
                selected, action, reason = choose(costs, predicted, candidates.actions)
                destination = folder/'models'/name
                receipt = _save_npz(destination/'trace.npz', predicted_states=predicted.states,
                    raw_deltas=predicted.raw_deltas, valid=predicted.valid, reasons=np.asarray(predicted.reasons),
                    failure_steps=predicted.failure_steps, candidates=candidates.actions,
                    total_cost=costs.total, **{'cost_'+key: value for key, value in costs.components.items()})
                choice = {'snapshot_id': spec['snapshot_id'], 'family_id': spec['family_id'], 'stratum': spec['stratum'],
                    'model': name, 'selected_index': selected, 'selected_action': action, 'reason': reason,
                    'input_hash': input_hash, 'candidate_hash': typed_digest(candidates.actions),
                    'prediction_hash': typed_digest(predicted.states), 'cost_hash': typed_digest(costs.total),
                    'previous_applied_action_hash': typed_digest(context.previous_applied_action),
                    'warm_sequence': None, 'candidate_rng_keys': [CANDIDATE_SEED, spec['episode_index'], spec['tick'], CANDIDATE_STREAM],
                    'predictor_calls': predictor.calls, 'before_any_physical_future': True}
                _save(destination/'choice_before_physics.json', choice)
                choices.append(json_safe(choice))
                prediction_files[receipt['path']] = receipt['sha256']
                prediction_files[str(destination/'choice_before_physics.json')] = file_sha256(destination/'choice_before_physics.json')
                entries.append((name, predicted, costs, choice))
        data.update(candidates=candidates, entries=entries, input_hash=input_hash, planner=planner, folder=folder)
    if len(choices) != 84 or sum(guards.values()):
        raise E1Stop('All84 model choices and zero future engine calls required before actual futures')
    _save(stage/'choices_before_any_physical_future.json', choices)
    prediction_files[str(stage/'choices_before_any_physical_future.json')] = file_sha256(stage/'choices_before_any_physical_future.json')
    _save(stage/'prediction_files_before_any_physical_future.json', prediction_files)
    _notify(callback, 'predictions_frozen', completed_prediction_sets=len(choices), candidate_future_intervals=0,
            choices_sha256=file_sha256(stage/'choices_before_any_physical_future.json'))
    return choices, prediction_files, guards


def _endpoint(predicted, actual, mask):
    rows = np.flatnonzero(mask)
    if not len(rows):
        return {'denominator': 0, 'XY_mean_m': None, 'yaw_absolute_mean_rad': None}
    delta = predicted[rows, -1]-actual[rows, 0, -1]
    yaw = np.arctan2(np.sin(delta[:, 2]), np.cos(delta[:, 2]))
    return {'denominator': len(rows), 'XY_mean_m': float(np.hypot(delta[:, 0], delta[:, 1]).mean()),
            'yaw_absolute_mean_rad': float(np.abs(yaw).mean()),
            'actual_reference': 'repeat0, stable valid candidates only; conditional on this mask'}


def _actual_arrays():
    return {'states': np.full((N, REPEATS, H+1, 10), np.nan, np.float64),
            'applied_actions': np.full((N, REPEATS, H, 2), np.nan, np.float64),
            'total_cost': np.full((N, REPEATS), np.inf, np.float64), 'valid': np.zeros((N, REPEATS), bool),
            **{'cost_'+key: np.full((N, REPEATS), np.nan, np.float64) for key in COMPONENT_NAMES}}


def _repeat(stage, data, candidate, repeat, manifest, callback):
    spec, capture = data['spec'], data['capture']
    repeat_id = spec['snapshot_id']+f'/candidate-{candidate:02d}/repeat-{repeat}'
    details = dict(snapshot_id=spec['snapshot_id'], family_id=spec['family_id'], repeat_id=repeat_id,
                   candidate_index=candidate, repeat_index=repeat, reserved_raw_records=1, reserved_intervals=H)
    _notify(callback, 'before_repeat', **details)
    target = data['folder']/'physical'/f'candidate-{candidate:02d}'/f'repeat-{repeat}'
    target.mkdir(parents=True, exist_ok=False)
    prefix, outcome, error = [], None, None
    original_step = PlanarPushEnv.step

    def preserve_returned(env, action):
        if len(prefix) >= H:
            raise E1Stop('Physical repeat cannot exceed its five reserved intervals')
        _notify(callback, 'before_interval', **details, interval_index=len(prefix),
                remaining_reserved_intervals=H-len(prefix))
        result = original_step(env, action)
        prefix.append(result)
        return result

    try:
        _notify(callback, 'repeat_started', **details)
        with patch.object(PlanarPushEnv, 'step', preserve_returned):
            outcome = replay_candidate(manifest['physical_config'], data['planner'], capture['snapshot'],
                data['candidates'].actions[candidate].copy(), capture['task_spec'],
                data['shared_context'].previous_applied_action.copy(), manifest['profile'])
        if not prefix:
            # Pure test doubles return a complete prefix without an env.step.
            # Real production replay always went through preserve_returned.
            prefix = outcome.results
        if (outcome.initial_snapshot_exact is not True or len(outcome.results) > H
                or outcome.states.dtype != np.float64 or outcome.states.shape != (H+1, 10)
                or outcome.applied_actions.dtype != np.float64 or outcome.applied_actions.shape != (H, 2)):
            raise E1Stop('Physical replay restore/shape/work contract is incomplete')
        if exact_comparison(prefix, outcome.results)['status'] != 'PASS_EXACT':
            raise E1Stop('Replay result differs from immediately retained physical prefix')
    except Exception as failure:
        error = {'type': type(failure).__name__, 'message': str(failure), 'traceback': traceback.format_exc()}
    receipt, raw_write_admitted = None, True
    try:
        _notify(callback, 'before_raw_write', **details, must_preserve=True,
                actual_returned_intervals=len(prefix), infrastructure_error=error is not None)
    except Exception as failure:
        error = {'type': type(failure).__name__, 'message': str(failure), 'traceback': traceback.format_exc(),
                 'previous_physical_error': error}
        raw_write_admitted = False
        try:
            # A whole-future guard stop may still permit the already reserved
            # prefix. This second callback must retain the hard free reserve.
            _notify(callback, 'preserve_raw_write', **details, must_preserve=True,
                already_reserved_prefix=True, actual_returned_intervals=len(prefix), infrastructure_error=True)
            raw_write_admitted = True
        except Exception as preservation_failure:
            error = {'type': type(preservation_failure).__name__, 'message': str(preservation_failure),
                'traceback': traceback.format_exc(), 'previous_physical_error': error,
                'raw_preservation_not_admitted': True}
    try:
        if raw_write_admitted:
            receipt = raw_codec.write_raw(target/'raw_execution', capture['snapshot'], prefix, xz_preset=6)
            total_bytes = sum(receipt['bytes'].values())
            _notify(callback, 'after_raw_write', **details, must_preserve=True,
                actual_returned_intervals=len(prefix), bytes_written=total_bytes,
                units_observed={'raw_total_bytes': total_bytes, 'raw_records': 1, 'raw_intervals': len(prefix),
                    'exact_raw_fixed_bytes_if_empty': total_bytes if not prefix else None,
                    'average_bytes_per_interval_including_fixed': total_bytes/len(prefix) if prefix else None,
                    'note': 'Pair total is measured; nonempty-prefix fixed/increment split is not independently measured.'},
                files_sha256=receipt['files_sha256'], raw_schema=receipt['schema'])
    except Exception as failure:
        error = {'type': type(failure).__name__, 'message': str(failure), 'traceback': traceback.format_exc(),
                 'previous_physical_error': error}
    if error is not None:
        row = {**details, 'status': 'INCOMPLETE_INFRASTRUCTURE', 'error': error,
            'actual_returned_intervals': len(prefix), 'raw_receipt': receipt,
            'raw_prefix_preservation': 'SAVED_LOSSLESS' if receipt is not None else 'INCOMPLETE_NOT_SAVED',
            'raw_write_admitted': raw_write_admitted,
            'partial_files_sha256': {str(path): file_sha256(path) for path in target.rglob('*') if path.is_file()}}
        _save(target/'infrastructure_error.json', row)
        _notify(callback, 'repeat_failed', **details, actual_returned_intervals=len(prefix),
                error=error, raw_receipt=receipt, must_preserve=True)
        raise E1Stop(f'Physical replay incomplete: {repeat_id}; raw prefix {row["raw_prefix_preservation"]}')
    row = {**details, 'status': 'COMPLETE_RECORDED', 'physical_valid': bool(outcome.valid),
        'cost': float(outcome.cost) if outcome.valid else None, 'cost_components': outcome.cost_components,
        'actual_returned_intervals': len(prefix), 'completed_intervals': outcome.completed_intervals,
        'initial_snapshot_exact': outcome.initial_snapshot_exact, 'reasons': outcome.reasons,
        'final_snapshot_hash': typed_digest(outcome.final_snapshot), 'state_hash': typed_digest(outcome.states),
        'applied_action_hash': typed_digest(outcome.applied_actions), 'raw_receipt': receipt}
    _save(target/'summary.json', row)
    _notify(callback, 'repeat_complete', **details, completed_id=repeat_id,
        actual_returned_intervals=len(prefix), completed_intervals=outcome.completed_intervals,
        bytes_written=sum(receipt['bytes'].values()), physical_valid=bool(outcome.valid),
        remove_reserved_raw_records=1, remove_reserved_intervals=H)
    return outcome, json_safe(row)


def _diagnostics(data, arrays, repeats, profile):
    stable = arrays['valid'].all(axis=1)
    common = stable.copy()
    for _, predicted, costs, _ in data['entries']:
        common &= predicted.valid & np.isfinite(costs.total)
    final_hashes = {(row['candidate_index'], row['repeat_index']): row['final_snapshot_hash'] for row in repeats}
    exact = all(exact_comparison(arrays['states'][index, 0], arrays['states'][index, repeat])['status'] == 'PASS_EXACT'
        and exact_comparison(arrays['applied_actions'][index, 0], arrays['applied_actions'][index, repeat])['status'] == 'PASS_EXACT'
        and arrays['total_cost'][index, 0] == arrays['total_cost'][index, repeat]
        and arrays['valid'][index, 0] == arrays['valid'][index, repeat]
        and final_hashes[(index, 0)] == final_hashes[(index, repeat)]
        for index in range(N) for repeat in (1, 2))
    rows, criteria = [], profile['counterfactual_check']
    for name, predicted, costs, choice in data['entries']:
        eligible = predicted.valid & np.isfinite(costs.total)
        diagnostic = compare_candidates(costs.total.copy(), eligible, arrays['total_cost'].copy(), arrays['valid'].copy(),
            atol=criteria['cost_padding_atol'], rtol=criteria['cost_padding_rtol'])
        if diagnostic['selected_index'] != choice['selected_index']:
            raise E1Stop('Actual labels altered the model-only choice')
        diagnostic.update({'snapshot_id': data['spec']['snapshot_id'], 'family_id': data['spec']['family_id'],
            'stratum': data['spec']['stratum'], 'model': name, 'tick': data['spec']['tick'],
            'input_hash': data['input_hash'], 'candidate_hash': choice['candidate_hash'],
            'all3_physical_repeats_exact': exact,
            'trajectory_valid_candidate_count': int(predicted.valid.sum()),
            'trajectory_valid_but_cost_nonfinite_indices': np.flatnonzero(predicted.valid & ~np.isfinite(costs.total)).tolist(),
            'conditional_endpoint_errors': _endpoint(predicted.states, arrays['states'], stable & eligible),
            'all7_common_valid_endpoint_errors': _endpoint(predicted.states, arrays['states'], common),
            'all7_common_valid_cost_error': {'denominator': int(common.sum()),
                'MAE': float(np.abs(costs.total[common]-arrays['total_cost'][common].mean(axis=1)).mean()) if common.any() else None},
            'statistical_unit': 'six independent families; paired ticks within each family, related training seeds'})
        _save(data['folder']/'models'/name/'diagnostics.json', diagnostic)
        rows.append(diagnostic)
    snapshot = {'snapshot_id': data['spec']['snapshot_id'], 'family_id': data['spec']['family_id'],
        'stratum': data['spec']['stratum'], 'tick': data['spec']['tick'], 'candidate_count': N,
        'physical_replays': N*REPEATS, 'actual_stable_valid_candidates': int(stable.sum()),
        'all7_common_valid': int(common.sum()), 'all3_repeats_exact': exact,
        'validity_changed_candidates': np.flatnonzero(arrays['valid'].any(axis=1) != stable).tolist()}
    return rows, snapshot


def run_e1(run_folder, frozen_manifest, remaining_budget_check_callback):
    """Run exactly the frozen12 snapshots/84 predictions/576 shared repeats.

    ``run_folder`` is the complete final run root (E2 inputs, E1 output).
    frozen_manifest is its already frozen dict, not a recipe to generate cases.
    Existing E1 output is refused; incomplete attempts are never overwritten or
    silently resumed. See _notify for the coordinator resource-ledger contract.
    """
    if not callable(remaining_budget_check_callback):
        raise TypeError('An active coordinator budget callback is required')
    manifest = deepcopy(frozen_manifest)
    if manifest.get('schema') != 'p5-frozen-final-manifest-v1' or manifest.get('status') != 'FROZEN_BEFORE_FINAL_OUTCOMES':
        raise E1Stop('A frozen final manifest is required')
    entries = manifest.get('E1')
    if entries != expected_E1(manifest['families']) or len(entries) != 12:
        raise E1Stop('Exact12 fixed first-two-family G ticks0/29 entries required')
    if any(row['candidates'] != N or row['horizon'] != H or row['physical_repeats'] != REPEATS for row in entries):
        raise E1Stop('Original N16/H5/3-repeat diagnostic scope required')
    if tuple(manifest['model_identities']) != MODELS:
        # Canonical saved JSON may sort keys; the fixed public order is used for
        # iteration, so key equality, rather than JSON dictionary order, matters.
        if set(manifest['model_identities']) != set(MODELS):
            raise E1Stop('All seven original model instances are required')
    started = perf_counter()
    run = Path(run_folder).resolve()
    stage = run/'E1'
    if stage.exists():
        raise FileExistsError('Preserve existing E1 attempts; no automatic retry/overwrite')
    progress = {'started_repeats': 0, 'returned_by_repeat': {}}
    def callback(event):
        if event['event'] == 'repeat_started':
            progress['started_repeats'] += 1
        if event['event'] in {'after_raw_write', 'repeat_complete', 'repeat_failed'}:
            progress['returned_by_repeat'][event['repeat_id']] = event['actual_returned_intervals']
        return remaining_budget_check_callback(event)
    _notify(callback, 'before_stage', expected_snapshots=12, expected_prediction_sets=84,
        expected_physical_repeats=576, maximum_future_intervals=2880)
    bindings = _frozen_bindings(manifest)
    _check_bindings(bindings)
    inputs, missing = _load_inputs(run, entries, manifest, bindings)
    stage.mkdir(parents=True, exist_ok=False)
    if missing:
        summary = {'schema': 'p5-final-E1-summary-v1', 'status': 'HOLD_MISSING_FIXED_G_INPUT',
            'expected_snapshots': entries, 'available_snapshot_count': len(inputs), 'missing': missing,
            'substitution_permitted': False, 'expected_prediction_sets': 84, 'completed_prediction_sets': 0,
            'expected_physical_repeats': 576, 'completed_physical_repeats': 0,
            'actual_future_intervals': 0, 'new_learning_updates': 0}
        _save(stage/'summary.json', summary)
        _notify(callback, 'stage_hold_missing_inputs', missing=missing)
        return summary
    runtime, rows, snapshots, repeat_rows = None, [], [], []
    weights_before = None
    actual_intervals, completed_intervals, attempted_repeats = 0, 0, 0
    current_arrays, current_data = None, None
    try:
        runtime = _load_runtime(manifest)
        weights_before = _weight_hashes(runtime)
        _save(stage/'manifest.json', {'schema': 'p5-final-E1-execution-v1', 'frozen_contract_hash': typed_digest(manifest),
            'frozen_E1_entries': entries, 'model_identities': manifest['model_identities'],
            'normalizer_identity': manifest['normalizer_identity'], 'source_bindings': bindings,
            'candidate_seed_base': CANDIDATE_SEED, 'candidate_stream_id': CANDIDATE_STREAM,
            'warm_policy': None, 'previous_action': 'actual G predecision command retained',
            'selection_order': 'all84 model choices durably frozen before first physical future',
            'physical_rule': '576 shared repeats, not576 per model', 'created_at_utc': datetime.now(timezone.utc).isoformat()})
        choices, prediction_files, guards = _freeze_predictions(stage, inputs, manifest, runtime, callback)
        _check_bindings(bindings)
        if weights_before != _weight_hashes(runtime):
            raise E1Stop('Model weights changed while freezing predictions; no physical future allowed')
        for data in inputs:
            arrays, local = _actual_arrays(), []
            current_arrays, current_data = arrays, data
            for candidate in range(N):
                for repeat in range(REPEATS):
                    outcome, record = _repeat(stage, data, candidate, repeat, manifest, callback)
                    arrays['states'][candidate, repeat] = outcome.states
                    arrays['applied_actions'][candidate, repeat] = outcome.applied_actions
                    arrays['total_cost'][candidate, repeat] = outcome.cost
                    arrays['valid'][candidate, repeat] = outcome.valid
                    for key in COMPONENT_NAMES:
                        arrays['cost_'+key][candidate, repeat] = outcome.cost_components[key]
                    actual_intervals += len(outcome.results)
                    completed_intervals += outcome.completed_intervals
                    local.append(record)
                    repeat_rows.append(record)
                    del outcome
                    gc.collect()
            _save_npz(data['folder']/'actual_futures.npz', **arrays)
            _save(data['folder']/'physical_replays.json', {'attempts': local, 'expected_attempts': 48})
            _check_bindings(prediction_files)
            diagnostics, snapshot = _diagnostics(data, arrays, local, manifest['profile'])
            rows.extend(diagnostics)
            snapshots.append(snapshot)
            _save(data['folder']/'summary.json', snapshot)
            _notify(callback, 'snapshot_complete', snapshot_id=data['spec']['snapshot_id'], completed_snapshots=len(snapshots),
                    completed_physical_repeats=len(repeat_rows), actual_future_intervals=actual_intervals)
            current_arrays, current_data = None, None
        _check_bindings(bindings)
        _check_bindings(prediction_files)
        weights_after = _weight_hashes(runtime)
        if weights_before != weights_after:
            raise E1Stop('Model weights changed during offline evaluation')
        attempted_repeats = progress['started_repeats']
        if attempted_repeats != 576 or len(repeat_rows) != 576 or len(rows) != 84 or actual_intervals > 2880:
            raise E1Stop('E1 full denominator/work bound incomplete')
        _save(stage/'comparison_ledger.json', rows)
        summary = {'schema': 'p5-final-E1-summary-v1', 'status': 'COMPLETE_RECORDED',
            'expected_snapshots': entries, 'completed_snapshots': len(snapshots), 'snapshots': snapshots,
            'expected_prediction_sets': 84, 'completed_prediction_sets': len(choices), 'diagnostic_count': len(rows),
            'expected_physical_repeats': 576, 'attempted_physical_repeats': attempted_repeats,
            'completed_physical_repeats': len(repeat_rows), 'actual_future_intervals': actual_intervals,
            'completed_full_duration_intervals': completed_intervals, 'maximum_future_intervals': 2880,
            'future_engine_calls_during_model_prediction': guards, 'all_sources_and_model_weights_unchanged': True,
            'repeat_validity_change_snapshot_ids': [row['snapshot_id'] for row in snapshots if row['validity_changed_candidates']],
            'conditional_selection_loss_denominator': sum(row['selection_loss_denominator'] for row in rows),
            'conditional_selection_loss_expected_rows': 84,
            'rank_inconclusive_pair_count': sum(row['pairwise']['actual_inconclusive'] for row in rows),
            'all_invalid_model_snapshot_count': sum(row['model_all_invalid'] for row in rows),
            'all_invalid_actual_snapshot_count': sum(row['actual_all_invalid'] for row in rows),
            'new_learning_updates': 0, 'new_closed_loop_cells': 0,
            'wall_time_s': perf_counter()-started, 'scope': 'Final offline N16/H5 diagnostic;6 families/12 related snapshots, not closed-loop success or global optimality.'}
        _save(stage/'summary.json', summary)
        _notify(callback, 'stage_complete', summary_path=str(stage/'summary.json'), summary_sha256=file_sha256(stage/'summary.json'),
                completed_physical_repeats=576, actual_future_intervals=actual_intervals)
        return summary
    except Exception as error:
        if current_arrays is not None and current_data is not None:
            path = current_data['folder']/'actual_futures.incomplete.npz'
            if not path.exists():
                _save_npz(path, **current_arrays)
        # The coordinator may stop immediately after a durable repeat receipt.
        # Reconcile saved rows so that a callback failure never erases it from
        # the attempted/completed denominator or reported physical work.
        durable_rows = []
        for saved_path in stage.glob('snapshots/**/physical/candidate-*/repeat-*/summary.json'):
            saved_row = _read(saved_path)
            if saved_row.get('status') == 'COMPLETE_RECORDED':
                durable_rows.append(saved_row)
        completed_ids = {row['repeat_id'] for row in durable_rows}
        failed_rows = [_read(path) for path in stage.glob('snapshots/**/physical/candidate-*/repeat-*/infrastructure_error.json')]
        source_check = None
        try:
            source_check = _check_bindings(bindings)
        except Exception as source_error:
            source_check = {'all_unchanged': False, 'error': str(source_error)}
        weights_check = None
        if runtime is not None and weights_before is not None:
            weights_check = weights_before == _weight_hashes(runtime)
        failure = {'schema': 'p5-final-E1-summary-v1', 'status': 'HOLD_INCOMPLETE_E1',
            'error': {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()},
            'expected_snapshots': entries, 'completed_snapshots': len(snapshots),
            'expected_prediction_sets': 84, 'expected_physical_repeats': 576,
            'attempted_physical_repeats': progress['started_repeats'], 'completed_physical_repeats': len(durable_rows),
            'actual_future_intervals_in_completed_repeats': sum(row['actual_returned_intervals'] for row in durable_rows),
            'actual_returned_intervals_including_failed_prefixes': (
                sum(row['actual_returned_intervals'] for row in durable_rows)
                +sum(row['actual_returned_intervals'] for row in failed_rows if row['repeat_id'] not in completed_ids)),
            'completed_repeat_ids': [row['repeat_id'] for row in durable_rows],
            'all_returned_raw_prefixes_saved': all(row.get('raw_receipt') is not None for row in failed_rows),
            'unsaved_raw_prefix_repeat_ids': [row['repeat_id'] for row in failed_rows if row.get('raw_receipt') is None],
            'source_check_after_stop': source_check, 'model_weights_unchanged_after_stop': weights_check,
            'denominator_preserved': True, 'automatic_retry': False,
            'partially_failed_repeat': 'See every infrastructure_error.json for actual returned count, raw receipt and explicit preservation status.',
            'new_learning_updates': 0}
        if not (stage/'summary.json').exists():
            _save(stage/'summary.json', failure)
        else:
            _save(stage/'post_summary_failure.json', failure)
        return failure
    finally:
        if runtime is not None:
            del runtime
        inputs.clear()
        gc.collect()
