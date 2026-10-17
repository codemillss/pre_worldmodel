"""Independent saved-output audit; never imports project evaluators or models.

The geometry oracle measures distance to four world-space line segments.
Numerical tolerances below only compare algebraic recomputations; they do not
change physical validity, task success, neighbor ranks or effect decisions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'runs/p4-data-retrain-20261013-02'
DEFAULT_RUN = ROOT/'runs/p4-mechanism-20261014-01'
ARMS = ('R', 'D00', 'D10', 'D01', 'D11')
BLOCKS = {'object_xy': (0, 1), 'pusher_xy': (6, 7), 'object_yaw': (2,)}
ATOL = RTOL = 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wrap(angle):
    # Preserve the registered [-pi,pi) endpoint and its exact float ordering.
    return np.remainder(angle+np.pi, 2*np.pi)-np.pi


def difference(predicted, actual):
    value = np.asarray(predicted, float)-np.asarray(actual, float)
    value = value.copy()
    value[..., 2] = wrap(value[..., 2])
    return value


def features(state, action):
    state, action = np.asarray(state), np.asarray(action)
    return np.concatenate((state[..., 6:8]-state[..., :2],
        np.sin(2*state[..., 2])[..., None], np.cos(2*state[..., 2])[..., None],
        state[..., 3:6], state[..., 8:10], action), axis=-1)


def segment_geometry(state, config):
    """Signed min-distance to four segments, independent of the q-box formula."""
    state = np.asarray(state, float)
    if state.shape != (10,) or not np.isfinite(state).all():
        raise ValueError('Finite ten-dimensional diagnostic state required')
    hx, hy = np.asarray(config['object']['size_full_xyz_m'][:2])/2
    angle = state[2]
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    vertices = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]]) @ rotation.T + state[:2]
    point = state[6:8]
    edges = np.roll(vertices, -1, axis=0)-vertices
    point_offsets = point-vertices
    fractions = np.einsum('ij,ij->i', point_offsets, edges)/np.einsum('ij,ij->i', edges, edges)
    closest = vertices+np.clip(fractions, 0, 1)[:, None]*edges
    distances = np.linalg.norm(point-closest, axis=1)
    cross = edges[:, 0]*point_offsets[:, 1]-edges[:, 1]*point_offsets[:, 0]
    inside = bool(np.all(cross >= 0))
    index = int(np.argmin(distances))
    distance = float(distances[index])
    signed = -distance if inside else distance
    normal = None
    if distance > 0:
        normal = (point-closest[index])/distance * (-1 if inside else 1)
    return {'gap': signed-float(config['pusher']['radius_m']), 'normal': normal,
            'closest': closest[index], 'inside': inside, 'segment_distances': distances}


def hybrids(actual, predicted, config, names):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    states, gaps = {}, {}
    for mask in range(8):
        state = actual.copy()
        for index, name in enumerate(names):
            if mask & (1 << index):
                state[list(BLOCKS[name])] = predicted[list(BLOCKS[name])]
        states[mask] = state
        gaps[mask] = segment_geometry(state, config)['gap']
    # Average marginal effects over all replacement orders, rather than
    # obtaining Shapley allocations from production Mobius coefficients.
    allocations = dict.fromkeys(names, 0.)
    for order in itertools.permutations(range(3)):
        mask = 0
        for index in order:
            next_mask = mask | (1 << index)
            allocations[names[index]] += (gaps[next_mask]-gaps[mask])/6
            mask = next_mask
    coefficients = {0: gaps[0]}
    for size in range(1, 4):
        for chosen in itertools.combinations(range(3), size):
            mask = sum(1 << i for i in chosen)
            coefficients[mask] = gaps[mask]-math.fsum(
                coefficients[sub] for sub in coefficients if sub != mask and sub & mask == sub)
    return states, gaps, allocations, coefficients


class Audit:
    def __init__(self):
        self.checks = {}
        self.failures = []
        self.bindings = {}

    def check(self, name, condition, details=None):
        if name in self.checks:
            raise ValueError('Duplicate audit name: '+name)
        self.checks[name] = bool(condition)
        if not condition:
            self.failures.append({'check': name, 'details': details})

    def equal(self, name, actual, expected, *, exact=False):
        a, b = np.asarray(actual), np.asarray(expected)
        same_shape = a.shape == b.shape
        okay = same_shape and (np.array_equal(a, b, equal_nan=True) if exact
            else np.allclose(a.astype(float), b.astype(float), atol=ATOL, rtol=RTOL, equal_nan=True))
        self.check(name, okay)

    def bind(self, path, expected=None):
        path = Path(path)
        digest = sha(path)
        relative = str(path.relative_to(ROOT))
        if relative in self.bindings and self.bindings[relative] != digest:
            raise ValueError('Source changed during audit: '+relative)
        self.bindings[relative] = digest
        if expected is not None:
            name = 'binding:'+relative
            if name not in self.checks:
                self.check(name, digest == expected)
            elif digest != expected:
                raise ValueError('Inconsistent expected binding: '+relative)
        return path

    def read(self, path, expected=None):
        return json.loads(self.bind(path, expected).read_text())


def check_decomposition(audit, name, result, actual, predicted, config, *, stop=False):
    names = tuple(result['block_order_for_masks']) if stop else tuple(BLOCKS)
    states, gaps, allocations, coefficients = hybrids(actual, predicted, config, names)
    values = {int(row['mask']): row for row in result['combinations']} if stop else {
        int(mask): row for mask, row in result['hybrids'].items()}
    audit.check(name+':eight_combinations', sorted(values) == list(range(8)))
    for mask in range(8):
        row = values[mask]
        audit.equal(f'{name}:gap:{mask}', row['gap_m'], gaps[mask])
        expected_blocks = [block for index, block in enumerate(names) if mask & (1 << index)]
        audit.check(f'{name}:blocks:{mask}', row['predicted_blocks'] == expected_blocks)
        if stop:
            audit.equal(f'{name}:hybrid_coordinates:{mask}', row['hybrid_geometry_coordinates'], states[mask][[0, 1, 2, 6, 7]], exact=True)
            audit.equal(f'{name}:mobius:{mask}', row['mobius_interaction_m'], 0 if mask == 0 else coefficients[mask])
        else:
            relative = states[mask][6:8]-states[mask][:2]
            angle = states[mask][2]
            local = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]]) @ relative
            audit.equal(f'{name}:world_relative:{mask}', row['world_relative_xy_m'], relative)
            audit.equal(f'{name}:body_relative:{mask}', row['body_relative_xy_m'], local)
            audit.equal(f'{name}:q:{mask}', row['box_offset_q_m'], np.abs(local)-np.asarray(config['object']['size_full_xyz_m'][:2])/2)
    audit.equal(name+':actual_gap', result['actual_gap_m'], gaps[0])
    audit.equal(name+':predicted_gap', result['predicted_gap_m'], gaps[7])
    total = gaps[7]-gaps[0]
    audit.equal(name+':total', result['total_gap_error_m' if stop else 'gap_error_m'], total)
    for block, allocation in allocations.items():
        audit.equal(name+':shapley:'+block, result['shapley_gap_error_m'][block], allocation)
        if stop:
            audit.equal(name+':six_order:'+block, result['six_order_average_m'][block], allocation)
    audit.equal(name+':allocation_sum', sum(result['shapley_gap_error_m'].values()), total)
    if not stop:
        for mask in range(1, 8):
            key = '+'.join(block for index, block in enumerate(names) if mask & (1 << index))
            audit.equal(name+':interaction:'+key, result['inclusion_exclusion_m'][key], coefficients[mask])
        err = difference(predicted, actual)
        audit.equal(name+':state_error', result['state_error'], err)
        local_result = result['local_linearization']
        if local_result['status'] == 'DEFINED':
            normal = segment_geometry(actual, config)['normal']
            relative = np.asarray(actual)[6:8]-np.asarray(actual)[:2]
            rotation_derivative = normal[0]*relative[1]-normal[1]*relative[0]
            terms = {'object_xy': -normal @ err[:2], 'pusher_xy': normal @ err[6:8],
                     'object_yaw': rotation_derivative*err[2]}
            for block in names:
                audit.equal(name+':linear:'+block, local_result['terms_m'][block], terms[block])
            audit.equal(name+':linear_remainder', local_result['nonlinear_remainder_m'], total-sum(terms.values()))
    else:
        for mask in (3, 5, 6):
            audit.equal(name+':interaction:'+str(mask), result['second_order_interactions_m'][str(mask)], coefficients[mask])
        audit.equal(name+':interaction:7', result['third_order_interaction_m'], coefficients[7])
    translated = np.asarray(predicted).copy()
    translated[:2] += [.13, -.17]
    translated[6:8] += [.13, -.17]
    turned = np.asarray(predicted).copy()
    turned[2] += np.pi
    audit.equal(name+':translation_invariant', segment_geometry(translated, config)['gap'], gaps[7])
    audit.equal(name+':pi_geometry_invariant', segment_geometry(turned, config)['gap'], gaps[7])


def load_saved(audit):
    evaluation = PARENT/'prediction_evaluation'
    index = audit.read(evaluation/'artifact_hashes.json')['sha256']
    training = audit.read(PARENT/'training_artifact_hashes.json')['sha256']
    manifest = audit.read(PARENT/'manifest.json', training['manifest.json'])
    saved = {}
    for arm in ARMS:
        summary = audit.read(evaluation/arm/'development.json', index[arm+'/development.json'])
        path = audit.bind(evaluation/arm/'development_traces.npz', index[arm+'/development_traces.npz'])
        with np.load(path, allow_pickle=False) as archive:
            saved[arm] = (summary, {k: archive[k].copy() for k in archive.files})
    return manifest, saved


def check_contact(audit, result, saved, config):
    audit.check('contact:denominators', result['target_rows'] == 5 and result['target_development_families'] == 1
                and result['S1_duplicate_heads_excluded'] is True and len(result['rows']) == 5)
    audit.check('contact:arm_order', [row['arm'] for row in result['rows']] == list(ARMS))
    for row in result['rows']:
        arm = row['arm']; name = 'contact:'+arm
        summary, arrays = saved[arm]
        ids = summary['core_H10_free']['identities']
        matching = [i for i, identity in enumerate(ids) if identity['family_rank'] == 1
                    and identity['y_condition'] == 'Y1' and identity['stop_condition'] == 'S0']
        audit.check(name+':unique_identity', len(matching) == 1)
        episode = matching[0]
        head_row = list(arrays['head_indices']).index(episode)*5+1
        current, actual = arrays['actual_states'][episode, 1:3]
        predicted = arrays['head_states'][head_row, 1]
        for key, value in ids[episode].items():
            audit.check(name+':identity:'+key, row[key] == value)
        audit.check(name+':mapping', row['teacher_head_row'] == head_row and row['core_current_state'] == 1 and row['core_next_state'] == 2)
        for field, expected in [('actual_start_state', current), ('actual_end_state', actual),
                                ('predicted_end_state', predicted), ('stored_raw_delta', arrays['head_raw_deltas'][head_row, 0]),
                                ('action', arrays['actual_actions'][episode, 1])]:
            audit.equal(name+':raw:'+field, row[field], expected, exact=True)
        integrated = current+arrays['head_raw_deltas'][head_row, 0]
        audit.equal(name+':delta_closure', difference(integrated, predicted), np.zeros(10))
        audit.check(name+':invalid_retained', row['valid'] is False and not arrays['head_valid'][head_row]
            and row['failure_reason'] == str(arrays['head_failure_reasons'][head_row]) == 'predicted_deep_penetration'
            and row['failure_step'] == arrays['head_failure_steps'][head_row] == 1)
        audit.equal(name+':start_geometry', row['actual_start_geometry']['gap_m'], segment_geometry(current, config)['gap'])
        audit.check(name+':measured_regime', row['actual_recorded_transition_regime'] == str(arrays['actual_regimes'][episode, 1]))
        check_decomposition(audit, name, row, actual, predicted, config)


def normalizer_arrays(audit):
    folder = ROOT/'runs/p3-02-20261009-01/learning_data/normalizer'
    metadata = audit.read(folder/'normalizer.json')
    path = audit.bind(folder/metadata['npz'], metadata['npz_sha256'])
    with np.load(path, allow_pickle=False) as archive:
        return metadata, {key: archive[key].copy() for key in archive.files}


def check_stop(audit, result, saved, config):
    summary, arrays = saved['D10']
    _, norm = normalizer_arrays(audit)
    audit.equal('stop:input_scale', result['original_input_scale'], norm['input_scale'], exact=True)
    audit.check('stop:two_cases', len(result['cases']) == 2 and result['model'] == 'D10')
    audit.check('stop:case_ranks', [c['identity']['family_rank'] for c in result['cases']] == [1, 10])
    for case in result['cases']:
        identity = case['identity']; name = 'stop:'+identity['family_id']
        matching = [i for i, item in enumerate(summary['S1_free_stop']['identities']) if item == identity]
        audit.check(name+':unique_identity', len(matching) == 1)
        index = matching[0]; episode = int(arrays['stop_indices'][index])
        audit.check(name+':mapping', case['stop_row'] == index and case['actual_row'] == episode
                    and identity['y_condition'] == 'Y0' and identity['stop_condition'] == 'S1')
        actual = arrays['actual_states'][episode, 10:16]
        actions = arrays['actual_actions'][episode, 10:15]
        free = arrays['stop_states'][index]
        free_delta = arrays['stop_raw_deltas'][index]
        teacher = arrays['stop_local_states'][index*5:(index+1)*5]
        teacher_delta = arrays['stop_local_raw_deltas'][index*5:(index+1)*5, 0]
        teacher_valid = arrays['stop_local_valid'][index*5:(index+1)*5]
        failure = int(arrays['stop_failure_steps'][index])
        audit.check(name+':denominators', case['horizon'] == case['attempted_steps'] == 5
                    and len(case['states']) == 6 and len(case['steps']) == 5)
        audit.check(name+':failure', case['failure_step'] == failure == 5
                    and case['failure_reason'] == str(arrays['stop_failure_reasons'][index]) == 'predicted_deep_penetration')
        audit.equal(name+':zero_commands', actions, np.zeros((5, 2)), exact=True)
        audit.equal(name+':same_start', free[0], actual[0], exact=True)
        audit.equal(name+':teacher_inputs', teacher[:, 0], actual[:-1], exact=True)
        audit.check(name+':contact_labels', case['actual_contact_regimes'] == arrays['actual_regimes'][episode, 10:15].tolist())
        audit.equal(name+':unchanged_tolerance', case['original_penetration_tolerance_m_unchanged'], config['validity']['predicted_penetration_tolerance_m'], exact=True)
        true_delta = difference(actual[1:], actual[:-1])
        unwrapped_cumulative = np.cumsum(free_delta-true_delta, axis=0)
        for k, state in enumerate(case['states']):
            point = f'{name}:state{k}'
            available = (failure < 0 or k <= failure) and np.isfinite(free[k]).all()
            audit.check(point+':mask', type(state['available']) is bool and state['available'] == available
                and state['prediction_valid'] == (available and (failure < 0 or k < failure))
                and state['finite_invalid_failure_state'] == (available and k == failure))
            if available:
                audit.equal(point+':error', state['state_error'], difference(free[k], actual[k]))
                check_decomposition(audit, point, state['geometry'], actual[k], free[k], config, stop=True)
        for t, row in enumerate(case['steps']):
            point = f'{name}:step{t+1}'
            audit.check(point+':masks', row['step'] == t+1 and row['available'] is True
                and row['free_prediction_valid'] == (t+1 < failure)
                and row['free_finite_invalid_failure_step'] == (t+1 == failure)
                and row['teacher_prediction_valid'] == bool(teacher_valid[t]))
            for field, expected in [('actual_input', actual[t]), ('predicted_input', free[t]),
                                    ('action', actions[t]), ('actual_next', actual[t+1]), ('free_next', free[t+1]),
                                    ('teacher_next', teacher[t, 1]), ('saved_free_increment', free_delta[t]),
                                    ('saved_teacher_increment', teacher_delta[t])]:
                audit.equal(point+':raw:'+field, row[field], expected, exact=True)
            audit.equal(point+':actual_increment', row['actual_increment'], true_delta[t])
            local_error = teacher_delta[t]-true_delta[t]
            shift = free_delta[t]-teacher_delta[t]
            prev_error = difference(free[t], actual[t])
            next_error = difference(free[t+1], actual[t+1])
            reconstructed = prev_error+local_error+shift
            reconstructed[2] = wrap(reconstructed[2])
            for field, expected in [('local_increment_error', local_error), ('saved_increment_difference', shift),
                                    ('previous_state_error', prev_error), ('next_state_error', next_error),
                                    ('reconstructed_next_error', reconstructed),
                                    ('teacher_next_state_error', difference(teacher[t, 1], actual[t+1]))]:
                audit.equal(point+':algebra:'+field, row[field], expected)
            audit.equal(point+':reconstruction', difference(reconstructed, next_error), np.zeros(10))
            audit.equal(point+':cumulative_closure', difference(unwrapped_cumulative[t], next_error), np.zeros(10))
            for field, expected in [('free_saved_integration_residual', difference(free[t]+free_delta[t], free[t+1])),
                                    ('teacher_saved_integration_residual', difference(actual[t]+teacher_delta[t], teacher[t, 1])),
                                    ('closure_residual', difference(reconstructed, next_error))]:
                audit.equal(point+':residual:'+field, row[field], expected)
                audit.equal(point+':zero:'+field, expected, np.zeros(10))
            audit.check(point+':raw_yaw_limits', abs(free_delta[t, 2]) < np.pi and abs(teacher_delta[t, 2]) < np.pi)
            true_feature, free_feature = features(actual[t], actions[t]), features(free[t], actions[t])
            for field, expected in [('actual_features', true_feature), ('free_features', free_feature),
                                    ('feature_difference', free_feature-true_feature),
                                    ('feature_difference_in_original_input_scale_units', (free_feature-true_feature)/norm['input_scale'])]:
                audit.equal(point+':feature:'+field, row[field], expected)
            check_decomposition(audit, point+':teacher_geometry', row['teacher_endpoint_geometry'], actual[t+1], teacher[t, 1], config, stop=True)
        for i, field in enumerate(result['state_fields']):
            trajectory = np.abs(difference(free, actual)[:, i])
            growth = case['state_error_growth'][field]
            audit.equal(name+':growth:'+field, growth['absolute_error_by_state'], trajectory)
            audit.check(name+':growth_order:'+field, growth['strictly_increasing_error_steps'] == [k for k in range(1, 6) if trajectory[k] > trajectory[k-1]])


def load_npz(audit, path, expected=None):
    with np.load(audit.bind(path, expected), allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def check_support(audit, result, saved):
    manifest = audit.read(PARENT/'manifest.json')
    train_index = audit.read(PARENT/'training_artifact_hashes.json')['sha256']
    prepared = load_npz(audit, PARENT/'prepared_training.npz', manifest['prepared_training_sha256'])
    bindings = audit.read(PARENT/'dataset_bindings.json', manifest['dataset_bindings_sha256'])
    main = ROOT/'runs/p4-data-main-20261013-01'
    collection = audit.read(main/'registration.json', bindings['collection_registration_sha256'])
    slot_document = audit.read(main/'slots_bound.json', bindings['slots_bound_sha256'])
    slot_map = {row['slot']: row for row in slot_document['entries']}
    plans = {row['episode_id']: row for row in collection['plans']}
    old_folder = ROOT/'runs/p3-02-20261009-01/learning_data'
    old_meta = audit.read(old_folder/'manifest.json')
    old = load_npz(audit, old_folder/old_meta['npz'], old_meta['npz_sha256'])
    old_episodes = {row['episode_index']: row for row in old_meta['preparation']['episodes']}
    norm_meta, norm = normalizer_arrays(audit)
    audit.check('support:normalizer_fixed', result['metric']['normalizer_hash'] == manifest['normalizer_hash']
                == norm_meta['normalizer']['normalizer_hash'] and result['metric']['normalizer_refit'] is False)
    audit.check('support:ranking_contract', result['metric']['top_k'] == 10
                and result['metric']['tie_break'] == 'lexicographic stable transition ID'
                and result['metric']['absolute_world_xy_in_model_features'] is False
                and result['metric']['body_frame_or_gap_used_for_ranking'] is False)
    audit.equal('support:replacement_slots', prepared['replacement_slots'], sorted(slot_map), exact=True)
    query = result['target']
    summary, arrays = saved['R']
    qepisode = next(i for i, identity in enumerate(summary['core_H10_free']['identities'])
        if identity['family_rank'] == 1 and identity['y_condition'] == 'Y1' and identity['stop_condition'] == 'S0')
    qs, qt = arrays['actual_states'][qepisode, 1:3]
    qa = arrays['actual_actions'][qepisode, 1]
    for field, expected in [('state', qs), ('next_state', qt), ('action_m_s', qa)]:
        audit.equal('support:query:'+field, query[field], expected, exact=True)
    qfeatures = features(qs, qa)
    q32 = ((qfeatures-norm['input_mean'])/norm['input_scale']).astype(np.float32)
    audit.equal('support:query_features', query['features_physical'], qfeatures)
    audit.equal('support:query_float32', query['features_normalized_float32'], q32, exact=True)
    audit.equal('support:query_delta', query['true_target_delta'], difference(qt, qs))
    for arm in ARMS:
        arm_summary, arm_arrays = saved[arm]
        episode = next(i for i, identity in enumerate(arm_summary['core_H10_free']['identities'])
            if identity['family_rank'] == 1 and identity['y_condition'] == 'Y1' and identity['stop_condition'] == 'S0')
        index = list(arm_arrays['head_indices']).index(episode)*5+1
        prediction = query['saved_predictions'][arm]
        audit.check('support:saved:'+arm+':identity', prediction['saved_head_row'] == index
                    and prediction['valid'] is False and prediction['failure_reason'] == str(arm_arrays['head_failure_reasons'][index]))
        audit.equal('support:saved:'+arm+':state', prediction['predicted_next_state'], arm_arrays['head_states'][index, 1], exact=True)
        audit.equal('support:saved:'+arm+':delta', prediction['saved_predicted_delta'], arm_arrays['head_raw_deltas'][index, 0], exact=True)
        audit.equal('support:saved:'+arm+':gap', prediction['predicted_geometry']['boundary_gap_m'], segment_geometry(arm_arrays['head_states'][index, 1], manifest['physical_config'])['gap'])
    raw_cache = {}
    def raw_core(episode):
        if episode not in raw_cache:
            folder = main/'episodes'/episode
            relative = str((folder/'manifest.json').relative_to(main))
            m = audit.read(folder/'manifest.json', bindings['raw_core_artifact_sha256'][relative])
            npz_path = folder/m['episodes'][0]['npz']
            data = load_npz(audit, npz_path, bindings['raw_core_artifact_sha256'][str(npz_path.relative_to(main))])
            plan = plans[episode]
            audit.check('support:raw_train:'+episode, plan['semantic_split'] == 'train'
                and plan['training_eligible'] is True and plan['warmup_training_eligible'] is False
                and data['transition_valid'].all())
            raw_cache[episode] = data
        return raw_cache[episode]
    all_rankings, top_union = {}, set()
    groups = {'relative_position': [0, 1], 'orientation': [2, 3],
              'object_velocity': [4, 5, 6], 'pusher_velocity': [7, 8], 'command': [9, 10]}
    regimes = ['noncontact', 'sustained_contact', 'transition']
    for arm in ARMS:
        name = 'support:'+arm
        trace_path = PARENT/'arms'/arm/'training_trace.npz'
        trace = load_npz(audit, trace_path, train_index[str(trace_path.relative_to(PARENT))])
        positions, offsets = trace['window_positions'], trace['batch_offsets']
        sizes = np.diff(offsets)
        audit.check(name+':stream_shape', positions.shape == (62992,) and offsets.shape == (1001,)
                    and offsets[0] == 0 and offsets[-1] == len(positions) and set(sizes) == {16, 64})
        exposure = np.bincount(positions, minlength=2960)
        mass = np.bincount(positions, weights=np.repeat(1/(sizes*5), sizes), minlength=2960)
        current = prepared[arm+'_states'][:, :-1].reshape(-1, 10)
        following = prepared[arm+'_states'][:, 1:].reshape(-1, 10)
        actions = prepared[arm+'_actions'].reshape(-1, 2)
        identities, metadata = [], []
        raw_consistent = True
        for slot in range(2960):
            for depth in range(5):
                flat = slot*5+depth
                if arm != 'R' and slot in slot_map:
                    entry = slot_map[slot]; episode = entry['arm_episodes'][arm]
                    t = entry['window_start']+depth
                    raw = raw_core(episode)
                    fraction = raw['contact_summary'][t, 2]
                    regime = 'noncontact' if fraction == 0 else 'sustained_contact' if fraction == 1 else 'transition'
                    identity = f'new:{episode}:{t:03d}'
                    source = {'source': 'new_core', 'episode_id': episode, 'family_id': entry['family_id'],
                              'transition_index': t, 'regime': regime}
                    expected = (raw['state'][t], raw['state'][t+1], raw['action_applied'][t])
                else:
                    index = int(prepared['old_window_rows'][slot, depth])
                    ep = old_episodes[int(old['episode_index'][index])]
                    identity = f'old:prepared-row:{index:05d}'
                    source = {'source': 'old_train', 'episode_id': ep['episode_id'], 'family_id': ep['family_id'],
                              'transition_index': int(old['transition_index'][index]), 'regime': regimes[int(old['regime_code'][index])]}
                    expected = (old['current_state'][index], old['next_state'][index], old['applied_action'][index])
                    raw_consistent &= old['split_code'][index] == 0 and ep['split'] == 'train'
                raw_consistent &= all(np.array_equal(x, y) for x, y in zip((current[flat], following[flat], actions[flat]), expected))
                identities.append(identity); metadata.append(source)
        audit.check(name+':all_window_rows_match_train_sources', raw_consistent and prepared[arm+'_mask'].all())
        ids, first, inverse, counts = np.unique(identities, return_index=True, return_inverse=True, return_counts=True)
        states, targets, commands = current[first], following[first], actions[first]
        info = [metadata[i] for i in first]
        audit.check(name+':duplicate_tensors_match', np.array_equal(current, states[inverse])
                    and np.array_equal(following, targets[inverse]) and np.array_equal(actions, commands[inverse]))
        encoded = features(states, commands)
        model_input = ((encoded-norm['input_mean'])/norm['input_scale']).astype(np.float32).astype(float)
        squared = (model_input-q32.astype(float))**2
        distance = np.linalg.norm(model_input-q32.astype(float), axis=1)
        order = sorted(range(len(ids)), key=lambda i: (distance[i], ids[i]))
        ranks = np.empty(len(ids), int); ranks[order] = np.arange(1, len(ids)+1)
        teacher_occurrences = np.bincount(inverse, weights=np.repeat(exposure, 5), minlength=len(ids)).astype(int)
        initial_occurrences = np.bincount(inverse[::5], weights=exposure, minlength=len(ids)).astype(int)
        per_depth = np.array([np.bincount(inverse[d::5], weights=exposure, minlength=len(ids)) for d in range(5)], int).T
        teacher_mass = np.bincount(inverse, weights=np.repeat(mass, 5), minlength=len(ids))
        row = result['arms'][arm]
        audit.check(name+':unique_denominators', row['unique_train_transitions'] == len(ids)
                    and row['unique_train_families'] == len({m['family_id'] for m in info})
                    and row['counts_by_source'] == dict(Counter(m['source'] for m in info))
                    and row['counts_by_actual_regime'] == dict(Counter(m['regime'] for m in info)))
        audit.check(name+':exposure_totals', int(teacher_occurrences.sum()) == row['exposure_totals']['teacher_transition_occurrences'] == 314960
                    and int(initial_occurrences.sum()) == row['exposure_totals']['rollout_initial_state_occurrences'] == 62992
                    and int(counts.sum()) == 14800)
        audit.equal(name+':teacher_mass_total', row['exposure_totals']['nominal_teacher_batch_mean_mass'], teacher_mass.sum())
        audit.check(name+':top10_order', [r['transition_id'] for r in row['top10']] == [str(ids[i]) for i in order[:10]])
        top_info = [info[i] for i in order[:10]]
        audit.check(name+':top10_denominators', row['top10_unique_train_families'] == len({m['family_id'] for m in top_info})
                    and row['top10_families'] == sorted({m['family_id'] for m in top_info})
                    and row['top10_counts_by_source'] == dict(Counter(m['source'] for m in top_info))
                    and row['top10_counts_by_actual_regime'] == dict(Counter(m['regime'] for m in top_info)))
        for key, feature_row in row['feature_univariate_ranges'].items():
            index = result['metric']['feature_fields'].index(key)
            audit.equal(name+':range:'+key, [feature_row['query_physical'], feature_row['minimum'], feature_row['maximum']],
                        [qfeatures[index], encoded[:, index].min(), encoded[:, index].max()])
            audit.check(name+':inside_range:'+key, feature_row['inside_univariate_observed_range'] == bool(encoded[:, index].min() <= qfeatures[index] <= encoded[:, index].max()))
        for source, selected in row['nearest_by_source'].items():
            relevant = [i for i in order if info[i]['source'] == source]
            if not relevant:
                audit.check(name+':no_source:'+source, selected is None)
            else:
                i = relevant[0]
                audit.check(name+':source_rank:'+source, selected['transition_id'] == ids[i] and selected['rank'] == ranks[i])
                audit.equal(name+':source_distance:'+source, selected['distance_normalized_L2'], distance[i])
        for rank, neighbor in enumerate(row['top10'], 1):
            i = order[rank-1]; point = f'{name}:neighbor{rank}'
            memberships = [[int(flat//5), int(flat%5)] for flat in np.flatnonzero(inverse == i)]
            audit.check(point+':metadata', all(neighbor[k] == value for k, value in info[i].items()))
            audit.check(point+':membership', neighbor['window_memberships'] == memberships
                        and neighbor['window_membership_count'] == len(memberships))
            audit.check(point+':exposure', neighbor['teacher_transition_occurrences'] == teacher_occurrences[i]
                        and neighbor['rollout_initial_state_occurrences'] == initial_occurrences[i]
                        and neighbor['rollout_target_occurrences_by_depth'] == per_depth[i].tolist()
                        and neighbor['training_exposed'] == bool(teacher_occurrences[i] > 0))
            audit.equal(point+':mass', neighbor['nominal_teacher_batch_mean_mass'], teacher_mass[i])
            audit.equal(point+':distance', neighbor['distance_normalized_L2'], distance[i])
            audit.equal(point+':coordinate_distance', [neighbor['squared_distance_by_feature'][key] for key in result['metric']['feature_fields']], squared[i])
            for group, indices in groups.items():
                audit.equal(point+':group:'+group, neighbor['squared_distance_by_group'][group], squared[i, indices].sum())
            for field, expected in [('physical_state', states[i]), ('physical_action_m_s', commands[i]), ('true_next_state', targets[i])]:
                audit.equal(point+':raw:'+field, neighbor[field], expected, exact=True)
            delta = difference(targets[i], states[i])
            audit.equal(point+':target_delta', neighbor['true_target_delta'], delta)
            audit.equal(point+':target_difference', neighbor['true_target_delta_minus_query_delta'], delta-difference(qt, qs))
            for key, state in [('current_geometry', states[i]), ('next_geometry', targets[i])]:
                audit.equal(point+':gap:'+key, neighbor[key]['boundary_gap_m'], segment_geometry(state, manifest['physical_config'])['gap'])
        top_union.update(str(ids[i]) for i in order[:10])
        all_rankings[arm] = {str(identity): {'rank': int(ranks[i]), 'distance_normalized_L2': float(distance[i]),
                'teacher_transition_occurrences': int(teacher_occurrences[i]), 'rollout_initial_state_occurrences': int(initial_occurrences[i]),
                'window_membership_count': int(counts[i])} for i, identity in enumerate(ids)}
    cross = result['top10_union_cross_arm_ranks_and_exposure']
    audit.check('support:cross_arm_union', [row['transition_id'] for row in cross] == sorted(top_union))
    for row in cross:
        identity = row['transition_id']
        for arm in ARMS:
            expected = all_rankings[arm].get(identity); stored = row['arms'][arm]
            name = 'support:cross:'+identity+':'+arm
            if expected is None:
                audit.check(name+':absent', stored is None)
            else:
                audit.check(name+':counts', all(stored[k] == v for k, v in expected.items() if k != 'distance_normalized_L2'))
                audit.equal(name+':distance', stored['distance_normalized_L2'], expected['distance_normalized_L2'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/p4_error_mechanism_independent_review.json')
    args = parser.parse_args()
    args.run = args.run.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError('Preserve previous independent review')
    audit = Audit()
    audit.bind(Path(__file__).resolve())
    registration = audit.read(args.run/'registration.json')
    for relative, expected in registration['bindings_sha256'].items():
        audit.bind(ROOT/relative, expected)
    manifest, saved = load_saved(audit)
    results = {name: audit.read(args.run/(name+'.json')) for name in ('contact_geometry', 'stop_propagation', 'input_support')}
    check_contact(audit, results['contact_geometry'], saved, manifest['physical_config'])
    check_stop(audit, results['stop_propagation'], saved, manifest['physical_config'])
    check_support(audit, results['input_support'], saved)
    for name, result in results.items():
        audit.check(name+':descriptive_status', result['status'] == 'DESCRIPTIVE_COMPLETE')
        if name == 'input_support':
            audit.check(name+':zero_execution', all(result['execution'].get(key) == 0 for key in (
                'model_constructions', 'model_forwards', 'optimizer_updates', 'physics_steps', 'normalizer_refits')))
            audit.check(name+':registration', result['registered_sources_checked'] is True)
        elif name == 'stop_propagation':
            audit.check(name+':zero_execution', all(result.get(key) == 0 for key in (
                'model_loads', 'model_forwards', 'learning_updates', 'physics_calls'))
                and result['threshold_changes'] is False and result['new_closed_loop_execution'] is False)
        else:
            audit.check(name+':zero_execution', all(result.get(key) == 0 for key in (
                'new_model_forwards', 'learning_updates', 'physics_steps')))
        for relative, expected in result['source_sha256'].items():
            path = ROOT/relative
            if str(path.relative_to(ROOT)) not in audit.bindings:
                audit.bind(path, expected)
            else:
                audit.check(name+':source:'+relative, sha(path) == expected)
    for relative, expected in tuple(audit.bindings.items()):
        audit.check('unchanged:'+relative, sha(ROOT/relative) == expected)
    output = {'schema': 'p4-error-mechanism-independent-review-v1',
        'status': 'PASS' if not audit.failures else 'FAIL', 'check_count': len(audit.checks),
        'checks': audit.checks, 'failures': audit.failures, 'source_sha256': audit.bindings,
        'numerical_identity_tolerance': {'atol': ATOL, 'rtol': RTOL,
            'scope': 'Arithmetic comparison only, never validity/effect/neighbor-rank tolerance'},
        'new_model_forwards': 0, 'physics_steps': 0, 'learning_updates': 0,
        'diagnostic_denominators': {'contact_endpoint_rows': 5, 'stop_endpoint_rows_free_plus_teacher': 20,
            'additional_identical_initial_state_geometry_rows': 2, 'hybrid_geometry_checks': 216,
            'selected_development_families': 2, 'independent_trial_count_claimed': False},
        'independence': {'imports': 'NumPy and Python standard library only',
            'geometry_oracle': 'Signed nearest distance to four world-space rectangle line segments minus pusher radius',
            'shapley_oracle': 'Average marginal gap over all six block replacement orders',
            'support_ranking': 'Reconstruct unique training transitions from frozen raw/prepared records; independently encode and normalize with NumPy, cast float32, distance float64, exact lexicographic ID ties',
            'not_recomputed': 'Torch-vs-NumPy precast sensitivity diagnostic; raw contact-force traces were previously audited and current support annotations use their frozen summary fractions'},
        'scope': 'Independent saved-array and diagnostic-output audit. Selected development cases are descriptive, not independent trials or causal/model-improvement evidence.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': output['status'], 'checks': output['check_count'], 'failures': audit.failures}))
    if audit.failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
