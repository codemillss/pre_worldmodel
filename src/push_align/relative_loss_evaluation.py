"""Fixed-D11 two-arm prediction comparison, without fitting or physics.

This module adds relative-position/gap estimands to the unchanged development
evaluation. Finite invalid states are visible only as diagnostic rows. They
never recover a missing primary family or enter a valid-only effect claim.
"""
from __future__ import annotations

import numpy as np

from .data_intervention_evaluation import (evaluate_development, metric_rows,
    paired_comparison, require, validate_development)
from .planning import contact_distance

ARMS = ('C0', 'R1')
PRIMARY_ENDPOINTS = ('relative_xy_m', 'gap_absolute_m')
SECONDARY_ENDPOINTS = ('object_xy_m', 'pusher_xy_m')
BOOTSTRAP_SEED = 2026101401
CONTACT_FAMILY = 'p4-data-development-2026101302-f001'
STOP_FAMILIES = (CONTACT_FAMILY, 'p4-data-development-2026101302-f010')


def finite_json(value):
    """Represent missing numeric values as null without filling or clipping."""
    if isinstance(value, np.ndarray): return finite_json(value.tolist())
    if isinstance(value, dict): return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [finite_json(v) for v in value]
    if isinstance(value, (bool, np.bool_)): return bool(value)
    if isinstance(value, (int, np.integer)): return int(value)
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    return value


def geometry_metrics(predicted, actual, config):
    require(predicted.shape == actual.shape and predicted.shape[-1] == 10,
            'Matched ten-component states required')
    result = metric_rows(predicted, actual, config)
    relative_error = (predicted[..., 6:8]-predicted[..., :2])-(actual[..., 6:8]-actual[..., :2])
    result.update(relative_xy_m=np.linalg.norm(relative_error, axis=-1),
                  relative_x_signed_m=relative_error[..., 0],
                  relative_y_signed_m=relative_error[..., 1],
                  predicted_gap_m=contact_distance(predicted, config),
                  actual_gap_m=contact_distance(actual, config))
    result['gap_signed_error_m'] = result['predicted_gap_m']-result['actual_gap_m']
    return result


def geometry_rows(predicted, actual, valid, identities, config):
    """Every attempted row stays visible; invalid finite errors are diagnostic."""
    valid = np.asarray(valid, bool)
    require(predicted.shape == actual.shape == (len(valid), 10)
            and len(identities) == len(valid), 'Geometry row denominator differs')
    metrics = geometry_metrics(predicted, actual, config)
    finite = np.isfinite(predicted).all(axis=1) & np.isfinite(actual).all(axis=1)
    require(np.all(~valid | finite), 'Valid geometry row is nonfinite')
    rows = [{**identity, 'prediction_valid': bool(ok), 'finite_state_available': bool(has),
             'finite_invalid_diagnostic_only': bool(has and not ok),
             'metrics': {name: float(values[n]) if has else None for name, values in metrics.items()}}
            for n, (identity, ok, has) in enumerate(zip(identities, valid, finite))]
    return {'attempted': len(valid), 'valid': int(valid.sum()), 'invalid': int((~valid).sum()),
            'finite_diagnostic_rows': int(finite.sum()),
            'finite_invalid_diagnostic_rows': int((finite & ~valid).sum()),
            'missing_rows': int((~finite).sum()), 'rows': rows,
            'metrics_valid_only_descriptive': {name: float(values[valid].mean()) if valid.any() else None
                                             for name, values in metrics.items()},
            'metrics_finite_including_invalid_diagnostic_only': {
                name: float(values[finite].mean()) if finite.any() else None for name, values in metrics.items()},
            'scope': 'All attempted rows retained; finite-invalid means describe failure and are not primary effects'}


def supplement_development(data, legacy, arrays, config, *, selected_cases=True):
    """Derive paired endpoints and chosen prior-failure diagnostics from traces."""
    families = validate_development(data)
    require(legacy['families'] == families, 'Development family order differs')
    head_idx, stop_idx = arrays['head_indices'], arrays['stop_indices']
    require(np.array_equal(arrays['actual_states'], data.states)
            and np.array_equal(arrays['actual_actions'], data.actions), 'Actual sample traces differ')
    head_ids = [{**identity, 'actual_contact_regime': str(regime)} for identity,regime in zip(
        legacy['Y_heads_teacher_conditioned']['identities'], data.regimes[head_idx,:5].reshape(-1))]
    stop_ids = legacy['S1_free_stop']['identities']
    local_ids = legacy['S1_teacher_conditioned_stop']['identities']
    head_truth = data.states[head_idx, 1:6].reshape(-1, 10)
    head_pred = arrays['head_states'][:, 1]
    stop_truth, stop_pred = data.states[stop_idx, 15], arrays['stop_states'][:, 5]
    head_valid, stop_valid = arrays['head_valid'], arrays['stop_valid']
    head_m, stop_m = geometry_metrics(head_pred, head_truth, config), geometry_metrics(stop_pred, stop_truth, config)
    primary, membership = {}, {}
    for phase, metrics, valid, identities in (
            ('contact_Y1_teacher_first5', head_m, head_valid, head_ids),
            ('stop_S1_free_K5', stop_m, stop_valid, stop_ids)):
        primary[phase], membership[phase] = {}, {}
        for family in families:
            indices = [n for n, i in enumerate(identities) if i['family_id'] == family
                       and (phase != 'contact_Y1_teacher_first5' or i['y_condition'] == 'Y1')]
            require(len(indices) == (5 if phase == 'contact_Y1_teacher_first5' else 2),
                    'Primary family membership differs')
            is_valid = bool(np.asarray(valid)[indices].all())
            primary[phase][family] = {key: float(metrics[key][indices].mean()) if is_valid else None
                for key in PRIMARY_ENDPOINTS+SECONDARY_ENDPOINTS}
            membership[phase][family] = {'attempted_rows': len(indices),
                'valid_rows': int(np.asarray(valid)[indices].sum()), 'all_rows_valid': is_valid,
                'invalid_row_identities': [identities[n] for n in indices if not valid[n]]}
    local_pred = arrays['stop_local_states'][:, 1]
    local_truth = data.states[stop_idx, 11:16].reshape(-1, 10)
    steps = {}
    for k in range(1, 6):
        valid = (arrays['stop_failure_steps'] == -1) | (arrays['stop_failure_steps'] > k)
        step_ids = [{**identity, 'free_step': k, 'core_current_state': 9+k,
                     'core_next_state': 10+k, 'actual_contact_regime': str(data.regimes[n,9+k])}
                    for identity,n in zip(stop_ids,stop_idx)]
        steps[str(k)] = geometry_rows(arrays['stop_states'][:, k], data.states[stop_idx, 10+k],
                                     valid, step_ids, config)
    summary = {'schema': 'p4-relative-loss-development-v1', 'families': families,
        'family_count': 12, 'model_seed_count': 1, 'primary': primary,
        'primary_membership': membership,
        'primary_phase_definitions': {
            'contact_Y1_teacher_first5': 'Y1 first five recorded transitions, including noncontact, sustained-contact and transition regimes; not a contact-only estimand',
            'stop_S1_free_K5': 'Actual-state10-conditioned five-step free stop endpoint; equal Y0/Y1 weight within family'},
        'Y_heads_teacher_conditioned': geometry_rows(head_pred, head_truth, head_valid, head_ids, config),
        'S1_free_stop': geometry_rows(stop_pred, stop_truth, stop_valid, stop_ids, config),
        'S1_free_stop_K1_to_K5': steps,
        'S1_teacher_conditioned_stop': geometry_rows(local_pred, local_truth, arrays['stop_local_valid'], local_ids, config),
        'invalid_masks': {'head_valid': head_valid.tolist(), 'stop_valid': stop_valid.tolist(),
            'head_failure_reasons': arrays['head_failure_reasons'].tolist(),
            'stop_failure_reasons': arrays['stop_failure_reasons'].tolist(),
            'stop_failure_steps': arrays['stop_failure_steps'].tolist()},
        'duplicate_S1_heads_excluded': True,
        'scope': 'Same known development families and fixed final checkpoints; no independent final test, task success or causal mechanism identification'}
    groups = {}
    for y in ('Y0','Y1'):
        groups[y] = {}
        for regime in ('noncontact','sustained_contact','transition'):
            selected = np.asarray([i['y_condition']==y and i['actual_contact_regime']==regime for i in head_ids])
            ids = [i for i,keep in zip(head_ids,selected) if keep]
            grouped = geometry_rows(head_pred[selected],head_truth[selected],head_valid[selected],ids,config)
            grouped['scope'] = 'Recorded-regime subgroup diagnosis only; valid-only and finite-invalid means do not replace the twelve-family primary estimand or establish superiority; no subgroup CI'
            groups[y][regime] = grouped
    summary['Y_heads_teacher_conditioned']['actual_contact_regimes'] = groups
    if selected_cases:
        contact = [n for n, i in enumerate(head_ids) if i['family_id'] == CONTACT_FAMILY
                   and i['y_condition'] == 'Y1' and i['stop_condition'] == 'S0' and i['core_step'] == 2]
        require(len(contact) == 1, 'Canonical f001 Y1 state1-to2 contact query absent')
        selected_stop = {}
        for family in STOP_FAMILIES:
            free = [n for n, i in enumerate(stop_ids) if i['family_id'] == family and i['y_condition'] == 'Y0']
            teacher = [n for n, i in enumerate(local_ids) if i['family_id'] == family and i['y_condition'] == 'Y0']
            require(len(free) == 1 and len(teacher) == 5, 'Canonical Y0 S1 stop query absent')
            f = free[0]
            require(np.array_equal(arrays['stop_states'][f, 0], data.states[stop_idx[f], 10]),
                    'Chosen stop is not actual-state10 conditioned')
            selected_stop[family] = {'identity': stop_ids[f],
                'failure_step': int(arrays['stop_failure_steps'][f]),
                'failure_reason': str(arrays['stop_failure_reasons'][f]),
                'free': [steps[str(k)]['rows'][f] for k in range(1, 6)],
                'teacher': [summary['S1_teacher_conditioned_stop']['rows'][n] for n in teacher],
                'scope': 'Previously selected failure diagnostic, not independent five-step trials'}
        contact_row = {**summary['Y_heads_teacher_conditioned']['rows'][contact[0]],
                       'failure_reason': str(arrays['head_failure_reasons'][contact[0]]),
                       'core_current_state': 1, 'core_next_state': 2}
        summary['selected_prior_failures'] = {'contact_state1_to2': contact_row,
            'stop_state10_to15': selected_stop,
            'selection_basis': 'Contact: prior common five-arm f001 Y1 state1-to2 failure. Stop: prior D10 f001/f010 Y0 S1 K5 failures; historical D11 stop predictions were already valid in these two cases.',
            'scope': 'Selected after inspecting historical results; not blind independent evaluation or proof that D11 had these stop failures'}
    return summary


def evaluate_relative_development(predictor, data, config):
    legacy, arrays = evaluate_development(predictor, data, config)
    return supplement_development(data, legacy, arrays, config), legacy, arrays


def compare_relative_arms(results, *, bootstrap_seed=BOOTSTRAP_SEED):
    require(set(results) == set(ARMS), 'Both C0/R1 final arms required')
    families = results['C0']['families']
    require(len(families) == 12 and results['R1']['families'] == families, 'Cross-arm family pairing differs')
    draws = np.random.default_rng(bootstrap_seed).integers(0, 12, (10000, 12), dtype=np.int64)
    contrasts, decisions, validity = {}, {}, {}
    for phase in ('contact_Y1_teacher_first5', 'stop_S1_free_K5'):
        contrasts[phase] = {}
        for metric in PRIMARY_ENDPOINTS+SECONDARY_ENDPOINTS:
            values = []
            for family in families:
                a, b = (results[arm]['primary'][phase][family][metric] for arm in ARMS)
                values.append(float(b-a) if a is not None and b is not None and np.isfinite([a,b]).all() else None)
            contrasts[phase][metric] = paired_comparison(values, families, draws)
        primary = [contrasts[phase][m] for m in PRIMARY_ENDPOINTS]
        if any(r['status'] != 'COMPLETE' for r in primary): decision = 'HOLD_INVALID_PRIMARY'
        elif all(r['bootstrap_percentile_95'][1] < 0 for r in primary): decision = 'DIRECTIONAL_SUPPORT'
        elif all(r['bootstrap_percentile_95'][0] > 0 for r in primary): decision = 'DIRECTIONAL_WORSENING'
        elif any(r['mean_difference'] < 0 for r in primary) and any(r['mean_difference'] > 0 for r in primary): decision = 'MIXED'
        else: decision = 'INCONCLUSIVE'
        decisions[phase] = {'status': decision, 'primary_endpoints': list(PRIMARY_ENDPOINTS)}
        groups = {'both_valid': [], 'C0_invalid_R1_valid': [], 'C0_valid_R1_invalid': [], 'both_invalid': []}
        for family in families:
            a, b = (results[arm]['primary_membership'][phase][family]['all_rows_valid'] for arm in ARMS)
            groups[('both_valid' if b else 'C0_valid_R1_invalid') if a else ('C0_invalid_R1_valid' if b else 'both_invalid')].append(family)
        validity[phase] = {'attempted_families': 12, **{key: {'count': len(rows), 'family_ids': rows} for key, rows in groups.items()},
                           'scope': 'Descriptive validity changes; not a substitute for withheld continuous primary contrast'}
    return {'schema': 'p4-relative-loss-paired-comparison-v1', 'arm_order': list(ARMS),
        'difference': 'R1 minus C0; negative numeric error difference favours relative loss',
        'family_ids': families, 'bootstrap': {'unit': 'family', 'samples': 10000, 'seed': bootstrap_seed,
            'interval_percentiles': [2.5,97.5], 'same_draws_all_contrasts': True},
        'contrasts': contrasts, 'effect_decisions': decisions, 'validity_changes': validity,
        'primary_invalid_policy': 'Any invalid family in either arm retains denominator12 and holds continuous endpoint/phase conclusions; finite invalid data stays diagnostic only',
        'scope': 'One fixed seed and previously inspected development conditions; no multiple-testing correction, generalization, alignment-success or causal-mechanism claim'}, draws
