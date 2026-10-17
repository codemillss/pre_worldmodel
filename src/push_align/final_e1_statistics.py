"""Aggregate the fixed E1 ledger without changing candidate diagnostics.

Scope v3 statistics and docs05/06 require common stratified family bootstrap,
with ticks and training seeds averaged inside each family. E1 has six families,
not 12 independent snapshots, 84 independent model sets or 576 independent
physical repeats. This module only reads diagnostics; it runs no model/physics.
"""
from __future__ import annotations

from collections import Counter
import json
import math

import numpy as np

from .final_manifest import CONTROLLERS, STRATA

MODELS = tuple(CONTROLLERS[1:])
GROUPS = {'R': ('R',), 'S1': ('S1-11', 'S1-23', 'S1-47'),
          'SK': ('SK-11', 'SK-23', 'SK-47')}
INDICATORS = ('model_all_invalid', 'no_selection', 'selected_actual_invalid',
              'selected_actual_validity_changed', 'actual_all_invalid',
              'no_stable_actual_valid', 'any_validity_changed', 'common_valid_empty')
METRICS = ('selection_loss', *INDICATORS)
SELECTIONS = {'no_valid_prediction': None, 'selected_actual_valid': 'STABLE_VALID',
              'selected_actual_invalid': 'STABLE_INVALID',
              'selected_actual_validity_changed': 'VALIDITY_CHANGED'}
DESCRIPTIVE = ('cost_MAE', 'cost_RMSE', 'rank_rho', 'all7_common_cost_MAE')


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, lower, upper, name):
    _need(type(value) is int and lower <= value <= upper, 'Invalid integer '+name)


def _number(value, name, *, nonnegative=True):
    _need(type(value) in (int, float) and math.isfinite(value)
          and (not nonnegative or value >= 0), 'Invalid finite metric '+name)


def _mean(values):
    values = np.asarray(values, np.float64)
    if not len(values):
        return None
    _need(bool(np.isfinite(values).all()), 'A mean requires its complete finite population')
    scale = float(np.max(np.abs(values)))
    return 0. if scale == 0 else float((math.fsum(float(value)/scale for value in values)/len(values))*scale)


def _boot_mean(values):
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    scale = np.max(np.where(finite, np.abs(values), 0.), axis=1, keepdims=True)
    scale = np.where(scale == 0, 1., scale)
    normalized = np.divide(values, scale, out=np.zeros_like(values), where=finite)
    means = (normalized.sum(axis=1)/np.maximum(count, 1))*scale[:, 0]
    means[count == 0] = np.nan
    return means, count.astype(np.int64)


def _interval(values):
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None
    scale = float(np.max(np.abs(finite)))
    return [0., 0.] if scale == 0 else (np.percentile(finite/scale, [2.5, 97.5])*scale).tolist()


def _expected(expected):
    _need(type(expected) is list and len(expected) == 12, 'Exact twelve frozen E1 specs required')
    by, family_specs = {}, {}
    for spec in expected:
        _need(type(spec) is dict and type(spec.get('snapshot_id')) is str
              and type(spec.get('family_id')) is str and bool(spec['family_id']), 'Malformed frozen E1 identity')
        _need(spec['snapshot_id'] not in by, 'Duplicate frozen E1 snapshot')
        _integer(spec.get('tick'), 0, 29, 'tick')
        _integer(spec.get('episode_index'), 0, 11, 'episode_index')
        _need(spec['tick'] in (0, 29) and spec.get('stratum') in STRATA
              and spec.get('controller_id') == 'G' and type(spec.get('candidates')) is int and spec['candidates'] == 16
              and type(spec.get('horizon')) is int and spec['horizon'] == 5
              and type(spec.get('physical_repeats')) is int and spec['physical_repeats'] == 3
              and spec['snapshot_id'] == spec['family_id']+f'/G/tick-{spec["tick"]:03d}',
              'E1 must retain G ticks0/29 and N16/H5/three shared repeats')
        by[spec['snapshot_id']] = spec
        family_specs.setdefault(spec['family_id'], []).append(spec)
    _need(len(family_specs) == 6, 'E1 requires six independent families')
    for family, specs in family_specs.items():
        _need(len(specs) == 2 and {item['tick'] for item in specs} == {0, 29}
              and len({item['stratum'] for item in specs}) == 1
              and len({item['episode_index'] for item in specs}) == 1,
              'Each E1 family needs exactly its paired ticks and one identity')
    _need(len({specs[0]['episode_index'] for specs in family_specs.values()}) == 6,
          'E1 independent families must have distinct episode indices')
    strata = {stratum: sorted(family for family, specs in family_specs.items() if specs[0]['stratum'] == stratum)
              for stratum in STRATA}
    _need(all(len(families) == 2 for families in strata.values()), 'E1 requires exactly two families per stratum')
    return by, sorted(family_specs), strata


def _validate_row(row, spec):
    _need(row.get('schema') == 'push-align-offline-candidate-diagnostics-v1'
          and all(row.get(key) == spec[key] for key in ('snapshot_id', 'family_id', 'stratum', 'tick')),
          'E1 diagnostic does not match its frozen snapshot identity')
    counts = row['counts']
    for key, upper in {'candidate_count':16, 'predicted_valid':16, 'predicted_invalid':16,
            'actual_stable_valid':16, 'actual_stable_invalid':16, 'actual_validity_changed':16,
            'common_valid':16, 'actual_repeat_attempts':48, 'actual_repeat_valid':48, 'actual_repeat_invalid':48}.items():
        _integer(counts.get(key), 0, upper, key)
    _need(counts['candidate_count'] == 16 and counts['actual_repeat_attempts'] == 48
          and counts['predicted_valid']+counts['predicted_invalid'] == 16
          and sum(counts[key] for key in ('actual_stable_valid','actual_stable_invalid','actual_validity_changed')) == 16
          and counts['actual_repeat_valid']+counts['actual_repeat_invalid'] == 48
          and counts['common_valid'] <= min(counts['predicted_valid'], counts['actual_stable_valid']),
          'Changed E1 candidate/repeat/validity denominators')
    for key in ('model_all_invalid','actual_all_invalid','no_stable_actual_valid','common_valid_empty'):
        _need(type(row.get(key)) is bool, 'Explicit boolean diagnostic flag required: '+key)
    _need(row['model_all_invalid'] == (counts['predicted_valid'] == 0)
          and row['actual_all_invalid'] == (counts['actual_repeat_valid'] == 0)
          and row['no_stable_actual_valid'] == (counts['actual_stable_valid'] == 0)
          and row['common_valid_empty'] == (counts['common_valid'] == 0), 'Diagnostic flags differ from original counts')
    selected, status = row['selected_index'], row['selection_status']
    _need(status in SELECTIONS and row['selected_actual_status'] == SELECTIONS[status], 'Unknown/inconsistent selection status')
    if selected is None:
        _need(status == 'no_valid_prediction' and row['model_all_invalid'], 'Missing selection is not a valid prediction')
    else:
        _integer(selected, 0, 15, 'selected_index')
        _need(status != 'no_valid_prediction' and not row['model_all_invalid'], 'Selected index contradicts model validity')
    loss = row['selection_loss']
    _integer(row['selection_loss_denominator'], 0, 1, 'selection_loss_denominator')
    _need(row['selection_loss_denominator'] == int(loss is not None)
          and (loss is not None) == (status == 'selected_actual_valid'), 'Conditional selection-loss denominator differs')
    if loss is not None:
        _number(loss, 'selection_loss')
        _need(row['selection_loss_ordering'] in {'INCONCLUSIVE','SELECTED_WORSE'}, 'Invalid loss ordering')
    else:
        _need(row['selection_loss_ordering'] is None, 'Undefined loss cannot have an ordering')
    flips = row['validity_flip_indices']
    _need(type(flips) is list and len(set(flips)) == len(flips)
          and len(flips) == counts['actual_validity_changed'], 'Validity-changed population differs')
    for index in flips:
        _integer(index, 0, 15, 'validity_flip_index')
    for group, keys in (('error_metrics', ('MAE','RMSE')), ('rank_correlation', ('value',)),
                        ('all7_common_valid_cost_error', ('MAE',))):
        metric = row[group]
        _integer(metric['denominator'], 0, 16, group+'.denominator')
        if group != 'all7_common_valid_cost_error':
            _need(metric['denominator'] == counts['common_valid'], 'Error/rank common-valid mask count differs')
        else:
            _need(metric['denominator'] <= counts['common_valid'], 'All-seven common mask is not a subset')
        for key in keys:
            value = metric[key]
            if value is not None:
                _number(value, group+'.'+key, nonnegative=group != 'rank_correlation')
                _need(metric['denominator'] > 0, 'Defined diagnostic has no eligible candidates')
                if group == 'rank_correlation':
                    _need(-1 <= value <= 1 and metric['denominator'] >= 2, 'Invalid rank correlation')
            elif group != 'rank_correlation':
                _need(metric['denominator'] == 0, 'Missing finite error despite nonempty common mask')
    _need(row['pairwise']['total_pairs'] == 120, 'Original sixteen-candidate pair denominator required')
    _integer(row['pairwise']['actual_inconclusive'], 0, 120, 'actual_inconclusive')
    for name in ('input_hash','candidate_hash'):
        _need(type(row.get(name)) is str and len(row[name]) == 64, 'Missing shared input/candidate hash')
    # Preserved diagnostic content must already be JSON-safe. No invalid/range
    # value is repaired here or reclassified from an undefined rho/order.
    json.dumps(row, allow_nan=False)


def _value(row, metric):
    if metric == 'selection_loss':
        return row['selection_loss']
    if metric == 'no_selection':
        return float(row['selected_index'] is None)
    if metric in ('selected_actual_invalid', 'selected_actual_validity_changed'):
        return float(row['selection_status'] == metric)
    if metric == 'any_validity_changed':
        return float(bool(row['validity_flip_indices']))
    return float(row[metric])


def _descriptive(row, metric):
    if metric in ('cost_MAE','cost_RMSE'):
        return row['error_metrics'][metric.removeprefix('cost_')], row['error_metrics']['denominator']
    if metric == 'rank_rho':
        return row['rank_correlation']['value'], row['rank_correlation']['denominator']
    return row['all7_common_valid_cost_error']['MAE'], row['all7_common_valid_cost_error']['denominator']


def _family_values(by, families, members, metric, value_function):
    result = []
    for family in families:
        per_seed = []
        for model in members:
            values = [value_function(by[(family, tick, model)], metric) for tick in (0, 29)]
            if any(value is None for value in values):
                break
            per_seed.append(_mean(values))
        result.append(_mean(per_seed) if len(per_seed) == len(members) else np.nan)
    return np.array(result, np.float64)


def _population(part, families, members, family_arrays):
    conditions = {metric: {'numerator': sum(bool(_value(row, metric)) for row in part),
                          'model_set_denominator': len(part)} for metric in INDICATORS}
    finite_loss = [row['selection_loss'] for row in part if row['selection_loss'] is not None]
    loss_values = family_arrays['selection_loss']
    descriptive = {}
    for metric in DESCRIPTIVE:
        observations = [(row['snapshot_id']+'/'+row['model'], *_descriptive(row, metric)) for row in part]
        values = [value for _, value, _ in observations if value is not None]
        descriptive[metric] = {'defined_model_sets': len(values), 'model_set_denominator': len(part),
            'mean_of_defined_model_set_values': _mean(values),
            'candidate_denominator_by_model_set': {identity: count for identity, _, count in observations},
            'scope': ('same all-seven predicted-valid AND all-three-actual-valid mask within each snapshot; snapshot sizes can differ'
                      if metric == 'all7_common_cost_MAE' else
                      'each model uses its own predicted-valid AND all-three-actual-valid candidates; masks may differ across models/snapshots; descriptive only'),
            'not_a_common_population_superiority_test': True}
    return {'members': list(members), 'model_set_count': len(part),
        'snapshot_count': len({row['snapshot_id'] for row in part}), 'independent_family_count': len(families),
        'conditions_on_original_model_set_population': conditions,
        'selection_status_counts': dict(sorted(Counter(row['selection_status'] for row in part).items())),
        'selection_loss_ordering_counts': dict(sorted(Counter(row['selection_loss_ordering'] or 'UNDEFINED' for row in part).items())),
        'selection_loss': {'defined_model_sets': len(finite_loss), 'undefined_model_sets': len(part)-len(finite_loss),
            'defined_model_set_mean_descriptive': _mean(finite_loss),
            'complete_family_denominator': int(np.isfinite(loss_values).sum()),
            'complete_family_mean': _mean(loss_values[np.isfinite(loss_values)]),
            'family_values': [float(value) if np.isfinite(value) else None for value in loss_values],
            'eligibility': 'all two ticks and all member training seeds require defined selection loss before family averaging'},
        'pairwise_actual_inconclusive_count': sum(row['pairwise']['actual_inconclusive'] for row in part),
        'pairwise_original_candidate_pairs_per_model_set': 120,
        'descriptive_cost_and_rank': descriptive,
        'rank_undefined_is_not_model_or_physics_failure': True}


def analyze_E1(rows84, expected12spec, replicates=2000, seed=7711):
    """Return a JSON-safe summary and object-free arrays for a frozen E1 ledger.

    Undefined losses exclude the entire family from that paired estimand when
    any tick/member seed on either side is undefined. All six families remain
    in the shared stratified draws; per-replicate eligible positions and zero-
    eligible replicates are retained. Failure/population rates keep all84 sets.
    """
    _integer(replicates, 2, 1000000, 'bootstrap_replicates')
    _integer(seed, 0, 2**64-1, 'bootstrap_seed')
    expected, families, strata = _expected(expected12spec)
    _need(type(rows84) is list and len(rows84) == 84, 'Exact84 original E1 model sets required')
    _need(all(type(row) is dict for row in rows84), 'E1 diagnostics must be dictionaries')
    desired = {(snapshot, model) for snapshot in expected for model in MODELS}
    pairs = [(row.get('snapshot_id'), row.get('model')) for row in rows84]
    _need(len(set(pairs)) == 84 and set(pairs) == desired, 'Missing/extra/duplicate E1 model/snapshot identity')
    by = {}
    for row in rows84:
        try:
            _validate_row(row, expected[row['snapshot_id']])
        except (KeyError, TypeError) as error:
            raise ValueError('Missing or malformed E1 diagnostic fields') from error
        by[(row['family_id'], row['tick'], row['model'])] = row
    for snapshot in expected:
        part = [row for row in rows84 if row['snapshot_id'] == snapshot]
        first = part[0]
        for row in part[1:]:
            _need(all(row[key] == first[key] for key in ('input_hash','candidate_hash','actual_records'))
                  and row['all7_common_valid_cost_error']['denominator'] == first['all7_common_valid_cost_error']['denominator']
                  and all(row['counts'][key] == first['counts'][key] for key in
                          ('actual_stable_valid','actual_stable_invalid','actual_validity_changed','actual_repeat_valid','actual_repeat_invalid')),
                  'E1 models do not share the same actual candidate labels/input/common-seven mask')
    methods = {**GROUPS, **{model: (model,) for model in MODELS if model not in GROUPS}}
    family_arrays = {name: {metric: _family_values(by, families, members, metric, _value) for metric in METRICS}
                     for name, members in methods.items()}
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.concatenate([rng.choice([families.index(family) for family in strata[stratum]],
        size=(replicates, 2), replace=True) for stratum in STRATA], axis=1).astype(np.int64)
    # Identity order is explicit in the JSON summary; keep NPZ arrays purely
    # numeric/boolean for exact regeneration with equal_nan=True.
    arrays = {'family_draw_indices': draws}
    for method, values in family_arrays.items():
        for metric, value in values.items():
            arrays[f'family_{method}_{metric}'] = value
            arrays[f'family_{method}_{metric}_defined'] = np.isfinite(value)
    contrasts = [('S1','R'), ('SK','R'), ('SK','S1')]
    contrasts += [(f'SK-{seed_id}', f'S1-{seed_id}') for seed_id in (11, 23, 47)]
    comparisons = []
    for left, right in contrasts:
        stats = {}
        for metric in METRICS:
            values = family_arrays[left][metric]-family_arrays[right][metric]
            boot, positions = _boot_mean(values[draws])
            valid = np.isfinite(boot)
            interval = _interval(boot)
            key = f'{left}_minus_{right}_{metric}'
            arrays[key], arrays[key+'_valid'] = boot, valid
            arrays[key+'_eligible_resampled_positions'] = positions
            arrays[key+'_eligible_family'] = np.isfinite(values)
            stats[metric] = {'difference_mean': _mean(values[np.isfinite(values)]),
                'family_differences': [float(value) if np.isfinite(value) else None for value in values],
                'eligible_family_ids': [family for family, value in zip(families, values) if np.isfinite(value)],
                'eligible_family_denominator': int(np.isfinite(values).sum()), 'original_family_denominator': 6,
                'bootstrap_valid_replicates': int(valid.sum()), 'bootstrap_unavailable_replicates': int((~valid).sum()),
                'bootstrap_replicate_denominator': replicates, 'percentile95_interval': interval,
                'interval_conditions_on_nonempty_eligible_resample': True,
                'zero_in_interval': bool(interval[0] <= 0 <= interval[1]) if interval else None,
                'estimand': ('conditional family mean loss difference: every tick and member seed on both sides must have defined loss'
                             if metric == 'selection_loss' else 'family mean difference in indicator fractions; all original model sets retained'),
                'claim': 'exploratory six-family paired comparison; no equivalence/global-optimality/industrial or seed-population claim'}
        comparisons.append({'left': left, 'right': right, 'metrics': stats})
    summary_by_method = {name: _population([row for row in rows84 if row['model'] in members], families, members, family_arrays[name])
                         for name, members in methods.items()}
    representatives = [next(row for row in rows84 if row['snapshot_id'] == snapshot) for snapshot in sorted(expected)]
    summary = {'schema': 'p5-final-E1-family-statistics-v1', 'status': 'COMPLETE_EXPECTED_E1_DENOMINATORS',
        'model_set_count': 84, 'snapshot_count': 12, 'independent_family_count': 6, 'model_count': 7,
        'shared_physical_repeat_attempts': 576, 'physical_repeats_are_not_resampling_units': True,
        'original_model_set_populations': {metric: {'numerator': sum(bool(_value(row, metric)) for row in rows84),
            'denominator':84} for metric in INDICATORS},
        'shared_actual_snapshot_populations': {metric: {'numerator': sum(bool(_value(row, metric)) for row in representatives),
            'denominator':12} for metric in ('actual_all_invalid','no_stable_actual_valid','any_validity_changed')},
        'models': {model: summary_by_method[model] for model in MODELS},
        'seed_groups': {name: summary_by_method[name] for name in GROUPS}, 'comparisons': comparisons,
        'bootstrap': {'seed': seed, 'replicates': replicates, 'families_in_order': families, 'strata': strata,
            'unit': 'family within stratum; two draws per stratum; common draws across every model/contrast',
            'averaging_order': 'mean paired ticks0/29 within each model/family; mean training seeds within that family; then compare/resample families',
            'missing_policy': 'complete tick/seed family eligibility per estimand; preserve full draws, eligible position counts and zero-eligible replicate masks',
            'limitations': 'Two families per E1 stratum. Candidate costs, three physical repeats, related ticks and learning seeds are not independent family samples. Exploratory uncertainty; training-seed population variation is not fully covered.'},
        'cost_rank_scope': 'Existing diagnostic MAE/RMSE/rho are descriptive on their stored eligible masks; model-specific mask changes prevent a common-population superiority interpretation. Undefined rho or INCONCLUSIVE order is not reclassified as failure.',
        'minimum_meaningful_difference': 'NOT_SPECIFIED; no post-result threshold',
        'personal_understanding': 'NOT_ASSESSED'}
    json.dumps(summary, allow_nan=False)
    _need(all(value.dtype != object for value in arrays.values()), 'Unsafe object NPZ array')
    return summary, arrays
