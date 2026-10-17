"""Frozen developmental data-intervention estimands, never model selection.

All arms see identical recorded states/actions. Family, not overlapping window,
is the uncertainty unit. Invalid predictions remain in every denominator and
prevent the corresponding directional conclusion. No simulator step or fitting
is performed by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .coverage import transition_regimes
from .data import digest, iter_dataset
from .planning import contact_distance, rollout, _state_reasons
from .representation import state_delta

ARMS = ('R', 'D00', 'D10', 'D01', 'D11')
ENDPOINTS = ('object_xy_m', 'pusher_xy_m')


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class DevelopmentData:
    identities: tuple[dict, ...]
    states: np.ndarray
    actions: np.ndarray
    regimes: np.ndarray
    source_sha256: dict


def load_development(run):
    """Validate all 48 registered development core episodes, retaining no train.

    Raw reader snapshot validation may use mj_forward to reconstruct a recorded
    snapshot. No future integration is executed. Model inference is guarded by
    the caller after this offline read.
    """
    run = Path(run)
    registration = json.loads((run/'registration.json').read_text())
    hashes = json.loads((run/'artifact_hashes.json').read_text())['sha256']
    require(digest(run/'registration.json') == hashes['registration.json'], 'Collection registration differs')
    plans = [p for p in registration['plans'] if p['semantic_split'] == 'development']
    require(len(plans) == 48 and len({p['family_id'] for p in plans}) == 12, 'Development family denominator differs')
    identities, states, actions, regimes, bindings = [], [], [], [], {}
    for plan in plans:
        label = plan['episode_id']
        folder = run/'episodes'/label
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                rel = str(path.relative_to(run))
                require(rel in hashes and digest(path) == hashes[rel], 'Raw development binding differs: '+rel)
                bindings[rel] = hashes[rel]
        manifest = json.loads((folder/'manifest.json').read_text())
        episode, = list(iter_dataset(folder, manifest['contract']))
        a, m = episode.arrays, episode.metadata
        policy = m['collection_policy']
        require(m['family_id'] == plan['family_id'] and m['episode_id'] == label
                and policy['predeclared_plan'] == plan and policy['role'] == 'core'
                and policy['training_eligible'] is False and policy['development_evaluation_eligible'] is True,
                'Development eligibility/provenance mismatch')
        require(a['state'].shape == (21, 10) and a['action_applied'].shape == (20, 2)
                and a['transition_valid'].all()
                and np.array_equal(a['action_requested'], a['action_applied'])
                and np.array_equal(a['action_applied'], np.asarray(plan['core_actions'], np.float64)),
                'Incomplete or altered development core')
        require(m['contract']['resolved_config'] == registration['physical_config'], 'Core physical contract differs')
        identities.append({k: plan[k] for k in ('episode_id', 'family_id', 'family_rank', 'geometry', 'y_condition', 'stop_condition')})
        states.append(a['state'].copy()); actions.append(a['action_applied'].copy())
        regimes.append(np.asarray(transition_regimes(episode)))
    data = DevelopmentData(tuple(identities), np.stack(states), np.stack(actions), np.stack(regimes), bindings)
    validate_development(data)
    return data


def validate_development(data):
    require(data.states.dtype == np.float64 and data.states.shape == (48, 21, 10)
            and data.actions.dtype == np.float64 and data.actions.shape == (48, 20, 2)
            and data.regimes.shape == (48, 20), 'Development arrays differ')
    require(np.isfinite(data.states).all() and np.isfinite(data.actions).all(), 'Nonfinite actual data')
    families = sorted({i['family_id'] for i in data.identities})
    require(len(families) == 12, 'Exactly 12 development families required')
    lookup = {(i['family_id'], i['y_condition'], i['stop_condition']): n for n, i in enumerate(data.identities)}
    require(len(lookup) == 48, 'Duplicated development identity')
    for family in families:
        for y in ('Y0', 'Y1'):
            a, b = (lookup[(family, y, s)] for s in ('S0', 'S1'))
            require(np.array_equal(data.states[a, :6], data.states[b, :6])
                    and np.array_equal(data.actions[a, :5], data.actions[b, :5])
                    and np.array_equal(data.regimes[a, :5], data.regimes[b, :5]), 'Duplicated S heads disagree')
            require(np.count_nonzero(np.all(data.actions[b, 10:15] == 0, axis=1)) == 5,
                    'Actual S1 stop is not five zero actions at state10')
    return families


def metric_rows(predicted, actual, config):
    """Physical error components and signed gap, preserving NaN missing rows."""
    require(predicted.shape == actual.shape and predicted.shape[-1] == 10, 'Metric state shape differs')
    error = predicted - actual
    error[..., 2] = (error[..., 2]+np.pi) % (2*np.pi)-np.pi
    return {
        'object_xy_m': np.linalg.norm(error[..., :2], axis=-1),
        'pusher_xy_m': np.linalg.norm(error[..., 6:8], axis=-1),
        'object_yaw_rad': np.abs(error[..., 2]),
        'object_velocity_m_s': np.linalg.norm(error[..., 3:5], axis=-1),
        'object_wz_rad_s': np.abs(error[..., 5]),
        'pusher_velocity_m_s': np.linalg.norm(error[..., 8:10], axis=-1),
        'pusher_vy_absolute_m_s': np.abs(error[..., 9]),
        'pusher_vy_signed_m_s': error[..., 9].copy(),
        'gap_absolute_m': np.abs(contact_distance(predicted, config)-contact_distance(actual, config)),
        'predicted_object_speed_m_s': np.linalg.norm(predicted[..., 3:5], axis=-1),
        'actual_object_speed_m_s': np.linalg.norm(actual[..., 3:5], axis=-1),
        'predicted_pusher_speed_m_s': np.linalg.norm(predicted[..., 8:10], axis=-1),
        'actual_pusher_speed_m_s': np.linalg.norm(actual[..., 8:10], axis=-1),
        'predicted_object_wz_rad_s': predicted[..., 5].copy(),
        'actual_object_wz_rad_s': actual[..., 5].copy(),
    }


def summarize_rows(metrics, valid, mask=None):
    valid = np.asarray(valid, bool)
    selected = np.ones(valid.shape, bool) if mask is None else np.asarray(mask, bool)
    require(selected.shape == valid.shape, 'Metric denominator shape differs')
    usable = valid & selected
    return {'attempted': int(selected.sum()), 'valid': int(usable.sum()),
            'invalid': int((selected & ~valid).sum()),
            'metrics_valid_only': {k: {'mean': float(np.mean(v[usable])) if usable.any() else None,
                                       'maximum_absolute': float(np.max(np.abs(v[usable]))) if usable.any() else None}
                                   for k, v in metrics.items()}}


def _json_metrics(metrics, valid):
    return {k: [float(x) if ok and np.isfinite(x) else None for x, ok in zip(v, valid)]
            for k, v in metrics.items()}


def evaluate_development(predictor, data, config):
    """Evaluate fixed Y heads, S1 zero suffix, and full head H10 separately."""
    families = validate_development(data)
    head_idx = np.asarray([n for n, i in enumerate(data.identities) if i['stop_condition'] == 'S0'])
    stop_idx = np.asarray([n for n, i in enumerate(data.identities) if i['stop_condition'] == 'S1'])
    require(len(head_idx) == len(stop_idx) == 24, 'Head/stop denominator differs')
    head_truth = data.states[head_idx, 1:6].reshape(-1, 10)
    head = rollout(data.states[head_idx, :5].reshape(-1, 10),
                   data.actions[head_idx, :5].reshape(-1, 1, 2), predictor, config)
    stop = rollout(data.states[stop_idx, 10], data.actions[stop_idx, 10:15], predictor, config)
    stop_local = rollout(data.states[stop_idx, 10:15].reshape(-1, 10),
                        data.actions[stop_idx, 10:15].reshape(-1, 1, 2), predictor, config)
    full = rollout(data.states[:, 0], data.actions[:, :10], predictor, config)
    head_metrics = metric_rows(head.states[:, -1], head_truth, config)
    stop_local_metrics = metric_rows(stop_local.states[:, -1], data.states[stop_idx, 11:16].reshape(-1, 10), config)
    stop_metrics = metric_rows(stop.states[:, -1], data.states[stop_idx, 15], config)
    full_metrics = metric_rows(full.states[:, -1], data.states[:, 10], config)
    head_valid = head.valid.reshape(24, 5)
    primary = {'Y_primary': {}, 'S_primary': {}}
    for family in families:
        h = [j for j, n in enumerate(head_idx) if data.identities[n]['family_id'] == family and data.identities[n]['y_condition'] == 'Y1']
        s = [j for j, n in enumerate(stop_idx) if data.identities[n]['family_id'] == family]
        require(len(h) == 1 and len(s) == 2, 'Family estimand membership differs')
        primary['Y_primary'][family] = {key: float(head_metrics[key].reshape(24, 5)[h].mean())
            if head_valid[h].all() else None for key in ENDPOINTS}
        primary['S_primary'][family] = {key: float(stop_metrics[key][s].mean())
            if stop.valid[s].all() else None for key in ENDPOINTS}
    head_ids = [{**data.identities[n], 'core_step': step+1} for n in head_idx for step in range(5)]
    stop_ids = [data.identities[n] for n in stop_idx]
    head_y = np.asarray([i['y_condition'] for i in head_ids])
    head_regime = data.regimes[head_idx, :5].reshape(-1)
    local_step = np.tile(np.arange(5), 24)
    steps = {}
    for k in range(1, 6):
        # Earlier valid steps remain visible even if a later step failed.
        valid = (stop.failure_steps == -1) | (stop.failure_steps > k)
        metrics = metric_rows(stop.states[:, k], data.states[stop_idx, 10+k], config)
        steps[str(k)] = {**summarize_rows(metrics, valid), 'row_metrics': _json_metrics(metrics, valid)}
    summaries = {
        'schema': 'p4-data-intervention-development-v1',
        'families': families, 'family_count': 12, 'model_seed_count': 1,
        'primary': primary,
        'Y_heads_teacher_conditioned': {
            **summarize_rows(head_metrics, head.valid), 'identities': head_ids,
            'valid_mask': head.valid.tolist(), 'failure_reasons': list(head.reasons),
            'row_metrics': _json_metrics(head_metrics, head.valid),
            'Y_conditions': {y: summarize_rows(head_metrics, head.valid, head_y == y) for y in ('Y0', 'Y1')},
            'actual_contact_regimes': {y: {r: summarize_rows(head_metrics, head.valid, (head_y == y) & (head_regime == r))
                for r in ('noncontact', 'sustained_contact', 'transition')} for y in ('Y0', 'Y1')},
            'duplicate_S1_heads_excluded': True,
        },
        'S1_free_stop': {**summarize_rows(stop_metrics, stop.valid), 'identities': stop_ids,
            'valid_mask': stop.valid.tolist(), 'failure_reasons': list(stop.reasons),
            'failure_steps': stop.failure_steps.tolist(), 'row_metrics': _json_metrics(stop_metrics, stop.valid),
            'free_K1_to_K5': steps, 'start_core_state_index': 10, 'endpoint_core_state_index': 15,
            'Y_conditions': {y: summarize_rows(stop_metrics, stop.valid, np.asarray([i['y_condition'] == y for i in stop_ids])) for y in ('Y0', 'Y1')},
        },
        'S1_teacher_conditioned_stop': {**summarize_rows(stop_local_metrics, stop_local.valid),
            'identities': [{**data.identities[n], 'core_step': step+11} for n in stop_idx for step in range(5)],
            'valid_mask': stop_local.valid.tolist(), 'failure_reasons': list(stop_local.reasons),
            'row_metrics': _json_metrics(stop_local_metrics, stop_local.valid),
            'first_zero': summarize_rows(stop_local_metrics, stop_local.valid, local_step == 0),
            'remaining_four_zero': summarize_rows(stop_local_metrics, stop_local.valid, local_step > 0)},
        'core_H10_free': {**summarize_rows(full_metrics, full.valid), 'identities': list(data.identities),
            'valid_mask': full.valid.tolist(), 'failure_reasons': list(full.reasons),
            'failure_steps': full.failure_steps.tolist(), 'row_metrics': _json_metrics(full_metrics, full.valid)},
        'scope': 'Fixed development families; state10-conditioned stop and core0 H10 are separate estimands. No task success or model selection.',
    }
    arrays = {'actual_states': data.states, 'actual_actions': data.actions, 'actual_regimes': data.regimes,
              'head_indices': head_idx, 'stop_indices': stop_idx}
    for name, result in (('head', head), ('stop', stop), ('stop_local', stop_local), ('full_H10', full)):
        arrays.update({name+'_'+k: v for k, v in {'states': result.states, 'raw_deltas': result.raw_deltas,
                       'valid': result.valid, 'failure_steps': result.failure_steps,
                       'failure_reasons': np.asarray(result.reasons)}.items()})
    return summaries, arrays


def paired_comparison(values, families, draws):
    """Same indexed family bootstrap for every contrast; never drop invalids."""
    require(len(values) == len(families) == 12, 'Pairing requires all 12 families')
    require(draws.shape == (10000, 12) and np.issubdtype(draws.dtype, np.integer)
            and np.all((draws >= 0) & (draws < 12)), 'Bootstrap registration differs')
    missing = [f for f, v in zip(families, values) if v is None or not np.isfinite(v)]
    result = {'family_ids': list(families), 'family_differences': values,
              'attempted_families': 12, 'valid_families': 12-len(missing), 'invalid_family_ids': missing,
              'mean_difference': None, 'bootstrap_percentile_95': [None, None],
              'improved': None, 'worsened': None, 'exact_float64_ties': None}
    if missing:
        result['status'] = 'HOLD_INVALID_PRIMARY'
        return result
    x = np.asarray(values, np.float64)
    result.update(status='COMPLETE', mean_difference=float(x.mean()),
                  bootstrap_percentile_95=np.percentile(x[draws].mean(axis=1), [2.5, 97.5]).tolist(),
                  improved=int((x < 0).sum()), worsened=int((x > 0).sum()), exact_float64_ties=int((x == 0).sum()))
    return result


def compare_arms(results):
    require(set(results) == set(ARMS), 'All five final arms are mandatory')
    families = results['R']['families']
    require(len(families) == 12 and all(r['families'] == families for r in results.values()), 'Cross-arm pairing differs')
    draws = np.random.default_rng(2026101304).integers(0, 12, size=(10000, 12), dtype=np.int64)
    formulas = {'D10-D00': {'D10': 1, 'D00': -1}, 'D11-D01': {'D11': 1, 'D01': -1},
                'D01-D00': {'D01': 1, 'D00': -1}, 'D11-D10': {'D11': 1, 'D10': -1},
                'interaction': {'D11': 1, 'D10': -1, 'D01': -1, 'D00': 1},
                **{arm+'-R': {arm: 1, 'R': -1} for arm in ARMS[1:]}}
    contrasts = {}
    for phase in ('Y_primary', 'S_primary'):
        contrasts[phase] = {}
        for metric in ENDPOINTS:
            contrasts[phase][metric] = {}
            for name, coefficients in formulas.items():
                values = []
                for family in families:
                    x = [results[arm]['primary'][phase][family][metric] for arm in coefficients]
                    values.append(float(sum(coefficients[a]*v for a, v in zip(coefficients, x)))
                                  if all(v is not None and np.isfinite(v) for v in x) else None)
                contrasts[phase][metric][name] = paired_comparison(values, families, draws)
    directional = {}
    for factor, phase, names in (('Y', 'Y_primary', ('D10-D00', 'D11-D01')),
                                 ('S', 'S_primary', ('D01-D00', 'D11-D10'))):
        rows = [contrasts[phase][metric][name] for metric in ENDPOINTS for name in names]
        if any(r['status'] != 'COMPLETE' for r in rows):
            decision = 'HOLD_INVALID_PRIMARY'
        elif all(r['bootstrap_percentile_95'][1] < 0 for r in rows):
            decision = 'DIRECTIONAL_SUPPORT'
        elif any(r['mean_difference'] < 0 for r in rows) and any(r['mean_difference'] > 0 for r in rows):
            decision = 'MIXED'
        else:
            decision = 'INCONCLUSIVE'
        directional[factor] = {'status': decision, 'phase': phase, 'contrasts': list(names),
                               'endpoints': list(ENDPOINTS), 'required_complete_conditions': 4}
    return {'schema': 'p4-data-intervention-paired-comparison-v1', 'family_ids': families,
            'bootstrap': {'unit': 'family', 'samples': 10000, 'seed': 2026101304,
                          'interval_percentiles': [2.5, 97.5], 'same_draws_all_contrasts': True},
            'contrasts': contrasts, 'directional_support': directional,
            'primary_invalid_policy': 'Any missing corresponding family retains denominator12 and holds directional conclusion',
            'units': {'object_xy_m': 'm', 'pusher_xy_m': 'm'},
            'scope': 'Descriptive fixed-model development uncertainty; one seed; no final test or task-success claim'}, draws


def regression_geometry(traces, config):
    """Supplement unchanged historical prediction metrics with geometry counts."""
    output = {}
    for k in (1, 5, 10):
        predicted = traces[f'h{k}_predicted_states']
        first_failure = np.full(len(predicted), -1, np.int64)
        reasons = np.full(len(predicted), '', dtype='U200')
        for step in range(k+1):
            rows = np.flatnonzero(first_failure == -1)
            if not len(rows): break
            r = _state_reasons(predicted[rows, step], config)
            bad = r != ''
            first_failure[rows[bad]] = step
            reasons[rows[bad]] = r[bad]
        valid = first_failure == -1
        regimes = traces[f'h{k}_regime_code']
        output[str(k)] = {'attempted_windows': len(predicted), 'geometry_valid_windows': int(valid.sum()),
            'geometry_invalid_windows': int((~valid).sum()),
            'regimes': {name: {'attempted_windows': int((regimes == code).sum()),
                         'geometry_invalid_windows': int(((regimes == code) & ~valid).sum())}
                        for name, code in (('noncontact', 0), ('sustained_contact', 1), ('transition', 2))},
            'failure_details': [{'window_index': int(i), 'failure_step': int(first_failure[i]), 'reason': str(reasons[i])}
                                for i in np.flatnonzero(~valid)],
            'scope': 'Supplement only; historical evaluate_predictions numeric metrics and denominator are unchanged'}
    return output
