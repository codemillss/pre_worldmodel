"""Offline shared-candidate diagnostics, with explicit validity denominators.

Actual future costs are evaluation labels only. They never determine the
predicted-cost selection. Repeated-cost intervals are observed numerical
ranges with a tolerance allowance, not confidence intervals or error bounds.
"""
from __future__ import annotations

import itertools
import json
import math

import numpy as np


class DiagnosticError(ValueError):
    """Malformed candidate costs, masks, or unrepresentable finite metrics."""


def _array(value, dtype, shape, name):
    if not isinstance(value, np.ndarray) or value.dtype != dtype or value.shape != shape:
        raise DiagnosticError(f"{name} must be a NumPy {np.dtype(dtype)} array with shape {shape}")
    return np.asarray(value)


def _costs(value, valid, name):
    if np.isnan(value).any() or np.isneginf(value).any() or (value < 0).any():
        raise DiagnosticError(f"{name} requires nonnegative costs; only invalid costs may be +inf")
    if not np.isfinite(value[valid]).all():
        raise DiagnosticError(f"{name} contains a nonfinite valid cost")


def _tolerance(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise DiagnosticError(f"{name} must be a finite nonnegative number")
    return float(value)


def _mean(values):
    """Finite signed mean without an overflowing sum."""
    peak = max(abs(float(value)) for value in values)
    return 0. if peak == 0 else (math.fsum(float(value) / peak for value in values) / len(values)) * peak


def _interval_order(first, second):
    """-1 means first is strictly cheaper; touching endpoints overlap."""
    if first[1] < second[0]:
        return -1
    if second[1] < first[0]:
        return 1
    return 0


def _finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _average_ranks(values):
    """One-based ranks; only exact equal costs share an average rank."""
    values = np.asarray(values, np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2
        start = stop
    return ranks


def _spearman(prediction, actual):
    if len(prediction) < 2:
        return None
    first, second = _average_ranks(prediction), _average_ranks(actual)
    first -= first.mean()
    second -= second.mean()
    norms = math.sqrt(float(first @ first)) * math.sqrt(float(second @ second))
    if norms == 0:
        return None
    return min(1., max(-1., float(first @ second) / norms))


def compare_candidates(predicted_costs, predicted_valid, actual_costs, actual_valid,
                       *, atol=1e-6, rtol=1e-5):
    """Compare predictions to three shared physical replays of every candidate.

    False-valid/false-invalid counts use only stable actual validity; changed
    validity is a separate investigation population. Error/ranking metrics
    use the common predicted-valid and three-replays-valid population. Loss is
    conditional on the chosen candidate being valid in every physical replay.
    All candidates remain represented, including all-invalid populations.
    """
    if not isinstance(predicted_costs, np.ndarray) or predicted_costs.ndim != 1 or not len(predicted_costs):
        raise DiagnosticError("Predicted costs must contain at least one candidate")
    count = len(predicted_costs)
    prediction = _array(predicted_costs, np.float64, (count,), "predicted_costs")
    prediction_valid = _array(predicted_valid, np.bool_, (count,), "predicted_valid")
    actual = _array(actual_costs, np.float64, (count, 3), "actual_costs")
    repeated_valid = _array(actual_valid, np.bool_, (count, 3), "actual_valid")
    atol, rtol = _tolerance(atol, "atol"), _tolerance(rtol, "rtol")
    _costs(prediction, prediction_valid, "predicted_costs")
    _costs(actual, repeated_valid, "actual_costs")

    stable_valid = repeated_valid.all(axis=1)
    stable_invalid = (~repeated_valid).all(axis=1)
    changed = ~(stable_valid | stable_invalid)
    common = prediction_valid & stable_valid
    means, intervals = {}, {}
    actual_records = []
    for index in range(count):
        status = "STABLE_VALID" if stable_valid[index] else ("STABLE_INVALID" if stable_invalid[index] else "VALIDITY_CHANGED")
        mean, interval, epsilon, raw_range, raw_width = None, None, None, None, None
        if stable_valid[index]:
            mean = _mean(actual[index])
            epsilon = atol + rtol * float(np.max(np.abs(actual[index])))
            raw_range = [float(np.min(actual[index])), float(np.max(actual[index]))]
            raw_width = raw_range[1] - raw_range[0]
            interval = [raw_range[0] - epsilon, raw_range[1] + epsilon]
            if not all(math.isfinite(value) for value in [mean, epsilon, *interval]):
                raise DiagnosticError("Actual-cost tolerance interval overflows finite representation")
            means[index], intervals[index] = mean, interval
        actual_records.append({"index": index, "status": status,
                               "repeated_valid": [bool(value) for value in repeated_valid[index]],
                               "repeated_costs": [_finite_or_none(value) for value in actual[index]],
                               "mean_cost": mean, "cost_interval": interval, "epsilon": epsilon,
                               "observed_cost_range": raw_range, "observed_cost_range_width": raw_width})

    predicted_indices = np.flatnonzero(prediction_valid)
    actual_indices = np.flatnonzero(stable_valid)
    selected = (int(predicted_indices[np.argmin(prediction[predicted_indices])])
                if len(predicted_indices) else None)
    best = min((int(index) for index in actual_indices), key=lambda index: (means[index], index)) if len(actual_indices) else None
    actual_by_mean = sorted(map(int, actual_indices), key=lambda index: (means[index], index))
    second_best = actual_by_mean[1] if len(actual_by_mean) >= 2 else None
    selected_status = actual_records[selected]["status"] if selected is not None else None
    loss, loss_ordering = None, None
    if selected is not None and stable_valid[selected]:
        loss = means[selected] - means[best]
        # The self-comparison has overlapping intervals as well: preserve raw
        # zero and the same-candidate flag without claiming a separated order.
        loss_ordering = "SELECTED_WORSE" if _interval_order(intervals[selected], intervals[best]) == 1 else "INCONCLUSIVE"

    common_indices = np.flatnonzero(common)
    error_values = [float(prediction[index]) - means[int(index)] for index in common_indices]
    mae = rmse = None
    if error_values:
        absolute = [abs(value) for value in error_values]
        mae = _mean(absolute)
        peak = max(absolute)
        rmse = 0. if peak == 0 else math.sqrt(_mean([(value / peak) ** 2 for value in absolute])) * peak
    errors = {"denominator": len(error_values), "MAE": mae, "RMSE": rmse}
    correlation = _spearman([float(prediction[index]) for index in common_indices],
                            [means[int(index)] for index in common_indices])

    pair_counts = {"actual_inconclusive": 0, "predicted_tie": 0, "concordant": 0, "discordant": 0}
    pair_records = []
    order_names = {-1: "FIRST_LOWER", 0: "INCONCLUSIVE", 1: "SECOND_LOWER"}
    for first, second in itertools.combinations(map(int, common_indices), 2):
        physical_order = _interval_order(intervals[first], intervals[second])
        predicted_order = (-1 if prediction[first] < prediction[second] else
                           (1 if prediction[first] > prediction[second] else 0))
        category = ("actual_inconclusive" if physical_order == 0 else
                    ("predicted_tie" if predicted_order == 0 else
                     ("concordant" if physical_order == predicted_order else "discordant")))
        pair_counts[category] += 1
        pair_records.append({"indices": [first, second],
                             "actual_order": order_names[physical_order],
                             "predicted_order": "TIE" if predicted_order == 0 else order_names[predicted_order],
                             "classification": category})
    strict_pairs = pair_counts["concordant"] + pair_counts["discordant"]
    total_pairs = count * (count - 1) // 2
    pairs = {"total_pairs": total_pairs, "common_valid_pairs": len(pair_records),
             "excluded_noncommon_pairs": total_pairs - len(pair_records),
             **pair_counts, "strict_comparison_pairs": strict_pairs,
             "concordance_rate": pair_counts["concordant"] / strict_pairs if strict_pairs else None,
             "records": pair_records}
    flips = np.flatnonzero(changed).tolist()
    result = {
        "schema": "push-align-offline-candidate-diagnostics-v1",
        "selected_index": selected, "selected_actual_status": selected_status,
        "selection_status": ("no_valid_prediction" if selected is None else
                             ("selected_actual_valid" if selected_status == "STABLE_VALID" else
                              ("selected_actual_invalid" if selected_status == "STABLE_INVALID" else
                               "selected_actual_validity_changed"))),
        "actual_best_index": best, "selection_loss": loss,
        "actual_second_index": second_best,
        "actual_best_mean_cost": means[best] if best is not None else None,
        "actual_second_mean_cost": means[second_best] if second_best is not None else None,
        "actual_best_second_mean_gap": means[second_best] - means[best] if second_best is not None else None,
        "actual_best_second_gap_ordering": ("SECOND_WORSE" if _interval_order(intervals[best], intervals[second_best]) == -1 else "INCONCLUSIVE") if second_best is not None else None,
        "selection_loss_denominator": int(loss is not None),
        "selection_loss_ordering": loss_ordering,
        "selected_actual_cost_interval": intervals.get(selected),
        "actual_best_cost_interval": intervals.get(best),
        "selected_is_actual_mean_best": selected is not None and selected == best,
        "model_all_invalid": not bool(prediction_valid.any()),
        "actual_all_invalid": not bool(repeated_valid.any()),
        "no_stable_actual_valid": not bool(stable_valid.any()),
        "actual_status": ("no_valid_candidate" if not repeated_valid.any() else
                          ("no_stable_valid_candidate" if not stable_valid.any() else "has_stable_valid_candidate")),
        "common_valid_empty": not bool(common.any()),
        "validity_flip_indices": flips,
        "infrastructure_status": "VALIDITY_CHANGED_REQUIRES_INVESTIGATION" if flips else "CONSISTENT_REPEATS",
        "counts": {"candidate_count": count,
                   "predicted_valid": int(prediction_valid.sum()),
                   "predicted_invalid": int((~prediction_valid).sum()),
                   "actual_stable_valid": int(stable_valid.sum()),
                   "actual_stable_invalid": int(stable_invalid.sum()),
                   "actual_validity_changed": int(changed.sum()),
                   "common_valid": int(common.sum()),
                   "false_valid": int((prediction_valid & stable_invalid).sum()),
                   "false_invalid": int((~prediction_valid & stable_valid).sum()),
                   "predicted_valid_actual_unstable": int((prediction_valid & changed).sum()),
                   "predicted_invalid_actual_unstable": int((~prediction_valid & changed).sum()),
                   "actual_repeat_attempts": count * 3,
                   "actual_repeat_valid": int(repeated_valid.sum()),
                   "actual_repeat_invalid": int((~repeated_valid).sum())},
        "error_metrics": errors,
        "rank_correlation": {"method": "Spearman_average_exact_ties", "denominator": len(common_indices),
                             "value": correlation,
                             "scope": "Descriptive correlation of predicted costs with actual means on common-valid candidates. Numerical-range pairwise ambiguity remains separate."},
        "pairwise": pairs,
        "actual_records": actual_records,
        "predicted_records": [{"index": index, "valid": bool(prediction_valid[index]),
                               "cost": _finite_or_none(prediction[index])} for index in range(count)],
        "tolerances": {"atol": atol, "rtol": rtol},
        "scope": "Actual futures are offline labels only; conditional errors/ranking/loss retain invalid and changed-validity denominators. Repeated intervals are observed numerical ranges, not confidence intervals.",
    }
    # Fail before returning an object with accidental NumPy values, NaN, or inf.
    json.dumps(result, allow_nan=False)
    return result
