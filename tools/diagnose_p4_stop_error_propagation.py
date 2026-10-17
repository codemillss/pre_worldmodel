"""Offline algebra on two frozen D10 stop traces; no models or physics.

The saved free/teacher increment difference includes batch-shape numerical
variation. Geometry recombinations are diagnostic coordinate hybrids, not
executable states or causal interventions. Missing predictions stay missing.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT/'runs/p4-data-retrain-20261013-02/prediction_evaluation'
NORM = ROOT/'runs/p3-02-20261009-01/learning_data/normalizer'
STATE_FIELDS = ('obj_x', 'obj_y', 'obj_yaw', 'obj_vx', 'obj_vy', 'obj_wz',
                'pusher_x', 'pusher_y', 'pusher_vx', 'pusher_vy')
STATE_UNITS = ('m', 'm', 'rad', 'm/s', 'm/s', 'rad/s', 'm', 'm', 'm/s', 'm/s')
FEATURE_FIELDS = ('relative_pusher_x', 'relative_pusher_y', 'sin_2yaw', 'cos_2yaw',
                  'obj_vx', 'obj_vy', 'obj_wz', 'pusher_vx', 'pusher_vy', 'applied_vx', 'applied_vy')
BLOCKS = {'object_xy': (0, 1), 'object_yaw': (2,), 'pusher_xy': (6, 7)}
CLOSURE_ATOL = 1e-12  # Numerical identity check only; never a task threshold.


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition: raise ValueError(message)


def finite_list(value):
    """Strict JSON with nulls for missing/nonfinite values, not imputation."""
    if isinstance(value, np.ndarray): return finite_list(value.tolist())
    if isinstance(value, dict): return {str(k): finite_list(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [finite_list(v) for v in value]
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)): return bool(value)
    if isinstance(value, (int, np.integer)): return int(value)
    return value


def wrap(value): return (value+np.pi) % (2*np.pi)-np.pi


def difference(predicted, actual):
    delta = np.asarray(predicted, np.float64)-np.asarray(actual, np.float64)
    delta = delta.copy(); delta[..., 2] = wrap(delta[..., 2])
    return delta


def integrated(state, delta):
    result = np.asarray(state)+np.asarray(delta)
    result = result.copy(); result[..., 2] = wrap(result[..., 2])
    return result


def encode(state, action):
    return np.concatenate((state[..., 6:8]-state[..., :2],
        np.sin(2*state[..., 2])[..., None], np.cos(2*state[..., 2])[..., None],
        state[..., 3:6], state[..., 8:10], action), axis=-1)


def signed_gap(state, half_xy, radius):
    relative = np.asarray(state)[6:8]-np.asarray(state)[:2]
    yaw = state[2]; c, s = np.cos(yaw), np.sin(yaw)
    local = np.asarray([c*relative[0]+s*relative[1], -s*relative[0]+c*relative[1]])
    q = np.abs(local)-np.asarray(half_xy)
    return float(np.hypot(*np.maximum(q, 0))+min(np.max(q), 0)-radius)


def geometry_decomposition(actual, predicted, half_xy, radius):
    """All 8 three-block hybrids, exact Mobius interactions and Shapley sums."""
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        return {'available': False, 'reason': 'missing_or_nonfinite_state'}
    names = tuple(BLOCKS)
    base = signed_gap(actual, half_xy, radius)
    values, interactions, combinations = {}, {}, []
    for mask in range(8):
        hybrid = np.asarray(actual).copy()
        chosen = [name for b, name in enumerate(names) if mask & (1 << b)]
        for name in chosen: hybrid[list(BLOCKS[name])] = predicted[list(BLOCKS[name])]
        gap = signed_gap(hybrid, half_xy, radius)
        values[mask] = gap-base
        submasks = [sub for sub in range(mask) if sub & mask == sub]
        interactions[mask] = values[mask]-sum(interactions[sub] for sub in submasks)
        combinations.append({'mask': mask, 'predicted_blocks': chosen,
                             'gap_m': gap, 'difference_from_actual_gap_m': values[mask],
                             'mobius_interaction_m': interactions[mask],
                             'hybrid_geometry_coordinates': hybrid[[0, 1, 2, 6, 7]].tolist()})
    shapley = {name: sum(interactions[m]/m.bit_count() for m in range(1, 8) if m & (1 << i))
               for i, name in enumerate(names)}
    closure = sum(shapley.values())-values[7]
    require(abs(closure) <= CLOSURE_ATOL, 'Geometry contribution identity failed')
    # Independent average over all six replacement orders verifies allocation.
    direct = dict.fromkeys(names, 0.)
    for order in itertools.permutations(range(3)):
        mask = 0
        for block in order:
            next_mask = mask | (1 << block)
            direct[names[block]] += (values[next_mask]-values[mask])/6
            mask = next_mask
    require(all(abs(direct[k]-shapley[k]) <= CLOSURE_ATOL for k in names), 'Shapley order average differs')
    return {'available': True, 'actual_gap_m': base, 'predicted_gap_m': base+values[7],
            'total_gap_error_m': values[7], 'combinations': combinations,
            'shapley_gap_error_m': shapley, 'six_order_average_m': direct,
            'sum_contribution_residual_m': closure,
            'second_order_interactions_m': {str(m): interactions[m] for m in (3, 5, 6)},
            'third_order_interaction_m': interactions[7],
            'block_order_for_masks': list(names),
            'scope': 'Algebraic allocation for this actual/predicted geometry pair; hybrids are not physical states or causal interventions. Velocities have no direct argument in gap; their effect on subsequent model outputs is not identified.'}


def diagnose_case(actual, actions, free_states, free_delta, teacher_states, teacher_delta,
                  *, failure_step, free_reason, teacher_valid, teacher_failure_steps,
                  input_scale, half_xy, radius, penetration_tolerance):
    """Decompose stored increments and states without invoking any predictor."""
    actual, actions, free_states, free_delta, teacher_states, teacher_delta = (
        np.asarray(x, np.float64) for x in (actual, actions, free_states, free_delta, teacher_states, teacher_delta))
    k = len(actions)
    require(actual.shape == free_states.shape == (k+1, 10)
            and actions.shape == (k, 2) and free_delta.shape == teacher_delta.shape == (k, 10)
            and teacher_states.shape == (k, 2, 10), 'Stored sequence shape differs')
    require(np.isfinite(actual).all() and np.isfinite(actions).all(), 'Actual labels/commands must be complete')
    require(np.array_equal(teacher_states[:, 0], actual[:-1]), 'Teacher inputs are not the recorded actual states')
    require(np.array_equal(free_states[0], actual[0]), 'Free prediction did not begin at actual stop state')
    require(failure_step == -1 or 0 <= failure_step <= k, 'Failure step outside horizon')
    input_scale = np.asarray(input_scale, np.float64)
    require(input_scale.shape == (11,) and np.isfinite(input_scale).all() and (input_scale > 0).all(), 'Fixed input scales invalid')
    require(np.asarray(teacher_valid).shape == np.asarray(teacher_failure_steps).shape == (k,), 'Teacher masks differ')
    actual_delta = difference(actual[1:], actual[:-1])
    rows, states = [], []
    for t in range(k+1):
        available = (failure_step < 0 or t <= failure_step) and np.isfinite(free_states[t]).all()
        state_error = difference(free_states[t], actual[t]) if available else np.full(10, np.nan)
        states.append({'step': t, 'available': available,
                      'prediction_valid': available and (failure_step < 0 or t < failure_step),
                      'finite_invalid_failure_state': available and t == failure_step,
                      'state_error': state_error,
                      'geometry': geometry_decomposition(actual[t], free_states[t], half_xy, radius) if available
                                  else {'available': False, 'reason': 'prediction_missing_after_failure'}})
    for t in range(k):
        eligible = failure_step < 0 or t+1 <= failure_step
        available = eligible and all(np.isfinite(x).all() for x in (
            free_states[t:t+2], free_delta[t], teacher_states[t], teacher_delta[t]))
        true_input = encode(actual[t], actions[t])
        free_input = encode(free_states[t], actions[t]) if eligible and np.isfinite(free_states[t]).all() else np.full(11, np.nan)
        row = {'step': t+1, 'available': available,
               'free_prediction_valid': bool(available and (failure_step < 0 or t+1 < failure_step)),
               'free_finite_invalid_failure_step': bool(available and t+1 == failure_step),
               'teacher_prediction_valid': bool(teacher_valid[t]),
               'actual_input': actual[t], 'predicted_input': free_states[t] if eligible else np.full(10, np.nan),
               'action': actions[t], 'actual_next': actual[t+1],
               'free_next': free_states[t+1] if eligible else np.full(10, np.nan),
               'teacher_next': teacher_states[t, 1],
               'teacher_next_state_error': difference(teacher_states[t, 1], actual[t+1]),
               'teacher_endpoint_geometry': geometry_decomposition(actual[t+1], teacher_states[t, 1], half_xy, radius),
               'actual_features': true_input, 'free_features': free_input,
               'feature_difference': free_input-true_input,
               'feature_difference_in_original_input_scale_units': (free_input-true_input)/input_scale,
               'actual_increment': actual_delta[t],
               'saved_free_increment': free_delta[t] if eligible else np.full(10, np.nan),
               'saved_teacher_increment': teacher_delta[t]}
        if available:
            previous = difference(free_states[t], actual[t])
            local = teacher_delta[t]-actual_delta[t]
            saved_difference = free_delta[t]-teacher_delta[t]
            observed = difference(free_states[t+1], actual[t+1])
            reconstructed = previous+local+saved_difference
            reconstructed[2] = wrap(reconstructed[2])
            closure = difference(reconstructed, observed)
            free_consistency = difference(integrated(free_states[t], free_delta[t]), free_states[t+1])
            teacher_consistency = difference(integrated(actual[t], teacher_delta[t]), teacher_states[t, 1])
            require(max(np.max(np.abs(closure)), np.max(np.abs(free_consistency)),
                        np.max(np.abs(teacher_consistency))) <= CLOSURE_ATOL,
                    'Saved state/increment algebra does not close')
            require(abs(free_delta[t, 2]) < np.pi and abs(teacher_delta[t, 2]) < np.pi,
                    'Finite stored diagnostic has an inadmissible raw yaw increment')
            row.update(previous_state_error=previous, local_increment_error=local,
                       saved_increment_difference=saved_difference, next_state_error=observed,
                       reconstructed_next_error=reconstructed, closure_residual=closure,
                       free_saved_integration_residual=free_consistency,
                       teacher_saved_integration_residual=teacher_consistency)
        else:
            row.update({key: np.full(10, np.nan) for key in (
                'previous_state_error', 'local_increment_error', 'saved_increment_difference',
                'next_state_error', 'reconstructed_next_error', 'closure_residual',
                'free_saved_integration_residual', 'teacher_saved_integration_residual')})
        rows.append(row)
    available_rows = [r for r in rows if r['available']]
    first = rows[0]
    maximum = max((float(np.max(np.abs(r['closure_residual']))) for r in available_rows), default=None)
    # Report all per-step values; exact changes are not effect-size thresholds.
    growth = {}
    for i, field in enumerate(STATE_FIELDS):
        trajectory = [abs(s['state_error'][i]) if s['available'] else None for s in states]
        growth[field] = {'absolute_error_by_state': trajectory,
                         'first_exact_nonzero_state': next((t for t, v in enumerate(trajectory) if v is not None and v != 0), None),
                         'strictly_increasing_error_steps': [t for t in range(1, k+1)
                            if trajectory[t] is not None and trajectory[t-1] is not None and trajectory[t] > trajectory[t-1]],
                         'unit': STATE_UNITS[i],
                         'scope': 'Exact descriptive ordering; a nonzero float or increase is not a meaningful-effect threshold.'}
    return finite_list({'schema': 'p4-stop-propagation-case-v1', 'horizon': k,
        'states': states, 'steps': rows, 'state_error_growth': growth,
        'failure_step': failure_step, 'failure_reason': free_reason,
        'first_input_exact': np.array_equal(free_states[0], actual[0]),
        'first_saved_increment_difference': first['saved_increment_difference'],
        'first_difference_interpretation': 'Free stop was batched24 and teacher stop batched120; any saved difference at identical first input cannot establish recursive-input influence. Later differences combine changed inputs and possible numerical batch variation.',
        'formula': 'e_next = wrap_yaw(e_previous + (teacher_delta - actual_delta) + (free_delta - teacher_delta))',
        'yaw': 'Actual/state errors use principal 2pi wrap. Local and saved-increment differences retain raw signed increment subtraction; wrap the reconstructed sum and closure residual only. This is a circular identity, not proof of physical winding.',
        'closure_atol': CLOSURE_ATOL, 'maximum_closure_absolute_error': maximum,
        'available_diagnostic_steps': len(available_rows), 'attempted_steps': k,
        'original_penetration_tolerance_m_unchanged': penetration_tolerance,
        'scope': 'Recorded offline algebra, not unique causal attribution. Finite failure-step predictions remain INVALID; missing suffix is not continued or interpolated.'})


def required_source_bindings():
    paths = [Path(__file__).resolve(), ROOT/'tests/test_p4_stop_error_propagation.py',
             EVAL/'artifact_hashes.json', EVAL/'registration.json', EVAL/'summary.json',
             EVAL/'D10/development.json', EVAL/'D10/development_traces.npz',
             EVAL.parent/'manifest.json', EVAL.parent/'training_artifact_hashes.json',
             NORM/'normalizer.json', NORM/'statistics.npz',
             ROOT/'src/push_align/representation.py', ROOT/'src/push_align/planning.py',
             ROOT/'src/push_align/data_intervention_evaluation.py']
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def run(output, expected_sources=None):
    output = Path(output)
    require(not output.exists(), 'Never overwrite an existing diagnostic result')
    bindings = required_source_bindings()
    if expected_sources is not None:
        require(all(expected_sources.get(p) == s for p, s in bindings.items()), 'Stop diagnostic registered sources changed')
    index = json.loads((EVAL/'artifact_hashes.json').read_text())['sha256']
    for name in ('summary.json', 'registration.json', 'D10/development.json', 'D10/development_traces.npz'):
        require(sha(EVAL/name) == index[name], 'Frozen evaluation input hash differs')
    train_index = json.loads((EVAL.parent/'training_artifact_hashes.json').read_text())['sha256']
    require(sha(EVAL.parent/'manifest.json') == train_index['manifest.json'], 'Training physical contract changed')
    norm = json.loads((NORM/'normalizer.json').read_text())
    require(sha(NORM/norm['npz']) == norm['npz_sha256'], 'Original normalization statistics changed')
    manifest = json.loads((EVAL.parent/'manifest.json').read_text())
    require(norm['normalizer']['normalizer_hash'] == manifest['normalizer_hash'], 'Normalizer identity differs')
    config = manifest['physical_config']
    with np.load(NORM/norm['npz'], allow_pickle=False) as n: scales = n['input_scale'].copy()
    summary = json.loads((EVAL/'D10/development.json').read_text())
    with np.load(EVAL/'D10/development_traces.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    results = []
    for rank in (1, 10):
        matches = [(i, row) for i, row in enumerate(summary['S1_free_stop']['identities'])
                   if row['family_rank'] == rank and row['y_condition'] == 'Y0' and row['stop_condition'] == 'S1']
        require(len(matches) == 1, 'Registered stop case is not unique')
        i, identity = matches[0]
        canonical = f'p4-data-development-2026101302-f{rank:03d}'
        require(identity['family_id'] == canonical and identity['episode_id'] == canonical+'-Y0-S1',
                'Canonical development family or episode ID differs')
        n = int(arrays['stop_indices'][i]); local = slice(i*5, (i+1)*5)
        require(not arrays['stop_valid'][i] and arrays['stop_failure_steps'][i] == 5
                and arrays['stop_failure_reasons'][i] == 'predicted_deep_penetration', 'Frozen failure identity differs')
        actions = arrays['actual_actions'][n, 10:15]
        require(np.array_equal(actions, np.zeros((5, 2))), 'Stop commands differ')
        result = diagnose_case(arrays['actual_states'][n, 10:16], actions,
            arrays['stop_states'][i], arrays['stop_raw_deltas'][i],
            arrays['stop_local_states'][local], arrays['stop_local_raw_deltas'][local, 0],
            failure_step=int(arrays['stop_failure_steps'][i]), free_reason=str(arrays['stop_failure_reasons'][i]),
            teacher_valid=arrays['stop_local_valid'][local], teacher_failure_steps=arrays['stop_local_failure_steps'][local],
            input_scale=scales, half_xy=np.asarray(config['object']['size_full_xyz_m'][:2])/2,
            radius=config['pusher']['radius_m'], penetration_tolerance=config['validity']['predicted_penetration_tolerance_m'])
        results.append({'identity': identity, 'stop_row': i, 'actual_row': n,
                        'actual_contact_regimes': arrays['actual_regimes'][n, 10:15].tolist(), **result})
    require(all(sha(ROOT/p) == s for p, s in bindings.items()), 'Diagnostic source changed during read')
    result = {'schema': 'p4-stop-propagation-diagnostic-v1', 'status': 'DESCRIPTIVE_COMPLETE',
              'source_sha256': bindings, 'cases': results, 'model': 'D10',
              'state_fields': list(STATE_FIELDS), 'state_units': list(STATE_UNITS),
              'feature_fields': list(FEATURE_FIELDS), 'original_input_scale': scales.tolist(),
              'model_loads': 0, 'model_forwards': 0, 'learning_updates': 0, 'physics_calls': 0,
              'new_closed_loop_execution': False, 'threshold_changes': False,
              'scope': 'Two previously observed failures; stored arrays only. No effects claimed for unobserved states, no new performance comparison.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream: json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    return {'path': str(output), 'sha256': sha(output), 'cases': 2, 'new_forwards': 0, 'new_physics': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--registration', type=Path)
    parser.add_argument('--registration-sha256')
    args = parser.parse_args()
    expected = None
    if args.registration is not None:
        require(args.registration_sha256 is not None and sha(args.registration) == args.registration_sha256,
                'Registration SHA differs')
        registration = json.loads(args.registration.read_text())
        expected = registration['bindings_sha256']
    elif args.registration_sha256 is not None:
        raise ValueError('Registration path missing')
    print(json.dumps(run(args.output, expected)))
