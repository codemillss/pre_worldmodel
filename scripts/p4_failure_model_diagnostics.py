"""Read-only P4-04 prediction diagnostics on already measured sequences.

This module never simulates, trains, clips commands, or changes a policy.
Teacher forcing supplies recorded current states *only for diagnosis*. Free
prediction sees the initial state and recorded applied commands; no later
physical state is supplied to it. Both paths use the same physical-delta
predictor interface as LearnedPredictor.

The inference masks below establish only finite/shape/raw-yaw integration
validity. They do not establish planner eligibility, contact geometry,
workspace validity, or task success. A caller must retain its separate
planner guards and physical validity evidence. Feature-support distances and
marginal ranges are descriptive, never a new rejection gate or OOD proof.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math

import numpy as np

from push_align.normalization import Normalizer
from push_align.representation import (FEATURE_FIELDS, FEATURE_UNITS, STATE_FIELDS,
                                       STATE_UNITS, RepresentationError, encode,
                                       integrate, state_delta)


class ModelDiagnosticError(ValueError):
    """The recorded sequence or frozen feature support violates its contract."""


def _require(condition, message):
    if not condition:
        raise ModelDiagnosticError(message)


def _array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.astype(value.dtype.newbyteorder("<"), copy=False).tobytes())
    return digest.hexdigest()


def _readonly(value):
    result = np.array(value, copy=True)
    result.flags.writeable = False
    return result


def _nearest(queries, reference, *, exclude_same_rows=False):
    """Direct 11D Euclidean distances; avoid a large all-pairs allocation.

    The subtraction form avoids cancellation in ||x||²+||y||²-2x·y. Ties
    select the first original train row. Self-exclusion is used only for
    descriptive train leave-one-row-out distances, not evaluation queries.
    """
    distances = np.full(len(queries), np.nan, dtype=np.float64)
    indices = np.full(len(queries), -1, dtype=np.int64)
    if len(reference) == 1 and exclude_same_rows:
        return distances, indices
    for start in range(0, len(queries), 64):
        rows = queries[start:start+64]
        difference = rows[:, None, :] - reference[None, :, :]
        squared = np.einsum("ijk,ijk->ij", difference, difference)
        if exclude_same_rows:
            local = np.arange(len(rows))
            squared[local, start+local] = np.inf
        nearest = np.argmin(squared, axis=1)
        distances[start:start+len(rows)] = np.sqrt(squared[np.arange(len(rows)), nearest])
        indices[start:start+len(rows)] = nearest
    return distances, indices


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "mean": None, "min": None, "q50": None,
                "q90": None, "q95": None, "q99": None, "max": None}
    maximum = float(np.max(np.abs(values)))
    mean = 0. if maximum == 0. else float(np.mean(values/maximum)*maximum)
    return {"count": len(values), "mean": mean if math.isfinite(mean) else None,
            "min": float(values.min()), "q50": float(np.quantile(values, .5)),
            "q90": float(np.quantile(values, .9)), "q95": float(np.quantile(values, .95)),
            "q99": float(np.quantile(values, .99)), "max": float(values.max())}


@dataclass(frozen=True)
class FeatureSupportIndex:
    normalizer: Normalizer
    train_features: np.ndarray
    normalized_train_features: np.ndarray
    physical_min: np.ndarray
    physical_max: np.ndarray
    marginal_tolerance: np.ndarray
    train_leave_one_out_distances: np.ndarray
    metadata: dict

    def validate(self):
        self.normalizer.validate()
        _require(self.normalizer.metadata["normalizer_hash"] == self.metadata["normalizer_hash"],
                 "Feature support normalizer identity changed")
        for key, value in (("train_features_sha256", self.train_features),
                           ("normalized_train_features_sha256", self.normalized_train_features),
                           ("physical_min_sha256", self.physical_min),
                           ("physical_max_sha256", self.physical_max),
                           ("marginal_tolerance_sha256", self.marginal_tolerance),
                           ("train_leave_one_out_distances_sha256", self.train_leave_one_out_distances)):
            _require(_array_hash(value) == self.metadata[key], "Feature support content changed")


def make_feature_support(normalizer, train_features, *, marginal_atol=1e-8,
                         marginal_rtol=1e-6):
    """Index original train features using existing stats, without fitting.

    The caller supplies each original valid train row once in original order,
    not overlapping windows. Count and train feature statistics are checked;
    source manifest/row selection remain the caller's provenance responsibility.
    Tolerance is atol + rtol*max(abs(train_min),abs(train_max)) per physical
    feature. It suppresses numeric boundary noise, not meaningful deviations.
    """
    _require(isinstance(normalizer, Normalizer), "A frozen Normalizer is required")
    normalizer.validate()
    _require(isinstance(train_features, np.ndarray) and train_features.dtype == np.float64
             and train_features.ndim == 2 and train_features.shape[1] == 11
             and len(train_features) >= 1 and np.isfinite(train_features).all(),
             "Train support requires finite float64 [T,11] original rows")
    _require(len(train_features) == normalizer.metadata["fit_count"],
             "Support count differs from original train fit count")
    for name, value in (("marginal_atol", marginal_atol), ("marginal_rtol", marginal_rtol)):
        _require(type(value) in (float, int) and math.isfinite(value) and value >= 0,
                 f"{name} requires a finite nonnegative number")
    means, stds = train_features.mean(axis=0), train_features.std(axis=0, ddof=0)
    means[2:4] = 0.
    _require(np.array_equal(means, normalizer.arrays["input_mean"])
             and np.array_equal(stds, normalizer.arrays["input_std"]),
             "Support features do not reproduce the frozen train input statistics")
    physical = _readonly(train_features)
    # Match the model's deliberate float32 input boundary, then measure in f64.
    normalized = _readonly(normalizer.normalize_features(physical).astype(np.float64))
    minimum, maximum = _readonly(physical.min(axis=0)), _readonly(physical.max(axis=0))
    tolerance = _readonly(float(marginal_atol)+float(marginal_rtol)*
                          np.maximum(np.abs(minimum), np.abs(maximum)))
    loo, _ = _nearest(normalized, normalized, exclude_same_rows=True)
    loo = _readonly(loo)
    metadata = {
        "format": "p4-frozen-feature-support-v1", "original_train_rows": len(physical),
        "normalizer_hash": normalizer.metadata["normalizer_hash"],
        "stats_hash": normalizer.metadata["stats_hash"],
        "dataset_identity": deepcopy(normalizer.metadata["dataset_identity"]),
        "feature_fields": list(FEATURE_FIELDS), "feature_units": list(FEATURE_UNITS),
        "marginal_atol": float(marginal_atol), "marginal_rtol": float(marginal_rtol),
        "nearest_distance": "Euclidean 11D frozen-normalized float32 model inputs, evaluated in float64",
        "nearest_index": "zero-based original train row, first row on exact tie",
        "support_scope": "Descriptive train support; no rejection gate, density test, independent validation, or cause proof",
        "train_leave_one_row_out_distance": _distribution(loo),
        "leave_one_out_scope": "Excludes own row only; adjacent correlated rows and duplicates remain",
    }
    for key, value in (("train_features_sha256", physical),
                       ("normalized_train_features_sha256", normalized),
                       ("physical_min_sha256", minimum), ("physical_max_sha256", maximum),
                       ("marginal_tolerance_sha256", tolerance),
                       ("train_leave_one_out_distances_sha256", loo)):
        metadata[key] = _array_hash(value)
    return FeatureSupportIndex(normalizer, physical, normalized, minimum, maximum,
                               tolerance, loo, metadata)


@dataclass(frozen=True)
class ModelSequenceDiagnostics:
    summary: dict
    traces: dict[str, np.ndarray]


def _validate_sequences(actual_states, applied_actions, actual_valid_mask):
    _require(isinstance(actual_states, np.ndarray) and actual_states.dtype == np.float64
             and actual_states.ndim == 3 and actual_states.shape[2] == 10
             and actual_states.shape[0] >= 1 and actual_states.shape[1] >= 2,
             "Actual states require physical float64 [B,H+1,10], B,H>=1")
    batch, horizon = actual_states.shape[0], actual_states.shape[1]-1
    _require(isinstance(applied_actions, np.ndarray) and applied_actions.dtype == np.float64
             and applied_actions.shape == (batch, horizon, 2),
             "Applied commands require physical float64 [B,H,2]")
    if actual_valid_mask is None:
        mask = np.ones((batch, horizon), dtype=np.bool_)
    else:
        _require(isinstance(actual_valid_mask, np.ndarray) and actual_valid_mask.dtype == np.bool_
                 and actual_valid_mask.shape == (batch, horizon), "Invalid actual validity mask")
        mask = actual_valid_mask.copy()
    _require(not np.any(mask[:, 1:] & ~mask[:, :-1]), "Actual mask must be a contiguous prefix")
    _require(np.isfinite(actual_states[:, 0]).all()
             and np.isfinite(actual_states[:, :-1][mask]).all()
             and np.isfinite(actual_states[:, 1:][mask]).all()
             and np.isfinite(applied_actions[mask]).all(),
             "Initial states and valid measured transitions must be finite")
    # Endpoints alone cannot prove physical yaw interval validity; supplied
    # actual_valid_mask must come from the caller's recorded physics checks.
    deltas = state_delta(actual_states[:, :-1][mask], actual_states[:, 1:][mask])
    _require((np.abs(deltas[:, 2]) < math.pi).all(),
             "Valid actual endpoints have an ambiguous exactly-pi yaw increment")
    return mask


def _attempt_next(predictor, states, actions):
    """Retain raw failures; batch rejection does not hide valid peer rows."""
    count = len(states)
    deltas = np.full((count, 10), np.nan, dtype=np.float64)
    next_states = np.full_like(deltas, np.nan)
    reasons = np.full(count, "", dtype="U256")

    def call(current, command):
        # Independent copies protect source and recurrence even from a callable
        # which mutates its arguments. Unexpected infrastructure exceptions
        # propagate; contract/numerical prediction failures are diagnostic.
        output = predictor(current.copy(), command.copy())
        _require(isinstance(output, np.ndarray) and output.dtype == np.float64
                 and output.shape == current.shape,
                 "Physical predictor requires float64 [N,10] deltas")
        return output.copy()

    try:
        deltas[:] = call(states, actions)
    except (ValueError, FloatingPointError):
        for row in range(count):
            try:
                deltas[row] = call(states[row:row+1], actions[row:row+1])[0]
            except (ValueError, FloatingPointError) as error:
                reasons[row] = f"inference_rejected:{type(error).__name__}:{error}"[:256]
    nonfinite = ~np.isfinite(deltas).all(axis=1)
    reasons[nonfinite & (reasons == "")] = "nonfinite_physical_delta"
    yaw_invalid = ~nonfinite & (np.abs(deltas[:, 2]) >= math.pi)
    reasons[yaw_invalid & (reasons == "")] = "raw_delta_yaw_abs_ge_pi"
    selected = np.flatnonzero(reasons == "")
    if len(selected):
        with np.errstate(over="ignore", invalid="ignore"):
            summed = states[selected]+deltas[selected]
        finite_sum = np.isfinite(summed).all(axis=1)
        reasons[selected[~finite_sum]] = "nonfinite_integrated_state"
        selected = selected[finite_sum]
        if len(selected):
            next_states[selected] = integrate(states[selected], deltas[selected])
    return deltas, next_states, reasons


def _support_traces(features, mask, support):
    shape = mask.shape
    normalized = np.full((*shape, 11), np.nan, dtype=np.float64)
    distances = np.full(shape, np.nan, dtype=np.float64)
    indices = np.full(shape, -1, dtype=np.int64)
    lower = np.zeros((*shape, 11), dtype=np.bool_)
    upper = np.zeros_like(lower)
    if mask.any():
        values = features[mask]
        normalized[mask] = support.normalizer.normalize_features(values).astype(np.float64)
        distances[mask], indices[mask] = _nearest(normalized[mask], support.normalized_train_features)
        lower[mask] = values < support.physical_min-support.marginal_tolerance
        upper[mask] = values > support.physical_max+support.marginal_tolerance
    return {"features": features, "normalized_features": normalized,
            "feature_valid_mask": mask, "nearest_train_distance": distances,
            "nearest_train_row": indices, "feature_below_train_range": lower,
            "feature_above_train_range": upper,
            "feature_outside_train_range": lower | upper}


def _errors(predicted, actual_next, mask, delta_scale):
    errors = np.full_like(actual_next, np.nan)
    errors[mask] = state_delta(actual_next[mask], predicted[mask])
    metrics = {
        "object_xy_error_m": np.hypot(errors[..., 0], errors[..., 1]),
        "object_yaw_learning_2pi_error_rad": np.abs(errors[..., 2]),
        "object_yaw_task_pi_error_rad": np.abs((errors[..., 2]+math.pi/2) % math.pi-math.pi/2),
        "object_linear_velocity_error_m_s": np.hypot(errors[..., 3], errors[..., 4]),
        "object_angular_velocity_error_rad_s": np.abs(errors[..., 5]),
        "pusher_xy_error_m": np.hypot(errors[..., 6], errors[..., 7]),
        "pusher_linear_velocity_error_m_s": np.hypot(errors[..., 8], errors[..., 9]),
    }
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = errors/delta_scale
        metrics["normalized_state_rmse"] = np.sqrt(np.mean(scaled**2, axis=-1))
    return {"state_errors": errors, **metrics}


def diagnose_model_sequence(predictor, support, actual_states, applied_actions, *,
                            actual_valid_mask=None):
    """Compare independent one-step and recurrent predictions on fixed truth.

    This is a post-execution diagnostic, not a controller. A false actual
    validity entry ends that measured prefix; suffix arrays may contain NaN.
    Invalid inference is preserved with raw deltas/reasons and honest per-step
    denominators. Teacher forcing can recover at a later measured step; a
    free branch cannot restart after its failed predicted prefix.
    """
    _require(callable(predictor), "A physical-delta predictor callable is required")
    _require(isinstance(support, FeatureSupportIndex), "A frozen FeatureSupportIndex is required")
    support.validate()
    mask = _validate_sequences(actual_states, applied_actions, actual_valid_mask)
    states, actions = actual_states.copy(), applied_actions.copy()
    batch, horizon = mask.shape
    actual_deltas = np.full((batch, horizon, 10), np.nan, dtype=np.float64)
    actual_deltas[mask] = state_delta(states[:, :-1][mask], states[:, 1:][mask])
    actual_features = np.full((batch, horizon, 11), np.nan, dtype=np.float64)
    actual_features[mask] = encode(states[:, :-1][mask], actions[mask])
    free_features = np.full_like(actual_features, np.nan)
    free_feature_mask = np.zeros_like(mask)
    teacher_deltas, free_deltas = np.full_like(actual_deltas, np.nan), np.full_like(actual_deltas, np.nan)
    teacher_next = np.full_like(actual_deltas, np.nan)
    free_states = np.full_like(states, np.nan)
    free_states[:, 0] = states[:, 0]
    teacher_valid, free_valid = np.zeros_like(mask), np.zeros_like(mask)
    teacher_reasons, free_reasons = np.full(mask.shape, "not_measured", dtype="U256"), np.full(mask.shape, "not_measured", dtype="U256")
    alive = np.ones(batch, dtype=np.bool_)
    for step in range(horizon):
        actual_rows = np.flatnonzero(mask[:, step])
        if len(actual_rows):
            deltas, next_states, reasons = _attempt_next(predictor, states[actual_rows, step], actions[actual_rows, step])
            teacher_deltas[actual_rows, step] = deltas
            teacher_next[actual_rows, step] = next_states
            teacher_reasons[actual_rows, step] = reasons
            teacher_valid[actual_rows, step] = reasons == ""
        free_reasons[mask[:, step] & ~alive, step] = "prior_inference_failure"
        free_rows = np.flatnonzero(mask[:, step] & alive)
        if len(free_rows):
            free_features[free_rows, step] = encode(free_states[free_rows, step], actions[free_rows, step])
            free_feature_mask[free_rows, step] = True
            deltas, next_states, reasons = _attempt_next(predictor, free_states[free_rows, step], actions[free_rows, step])
            free_deltas[free_rows, step] = deltas
            free_states[free_rows, step+1] = next_states
            free_reasons[free_rows, step] = reasons
            free_valid[free_rows, step] = reasons == ""
            alive[free_rows[reasons != ""]] = False
    support_traces = {"actual_input": _support_traces(actual_features, mask.copy(), support),
                      "free_input": _support_traces(free_features, free_feature_mask, support)}
    metrics = {"teacher": _errors(teacher_next, states[:, 1:], teacher_valid, support.normalizer.arrays["delta_scale"]),
               "free": _errors(free_states[:, 1:], states[:, 1:], free_valid, support.normalizer.arrays["delta_scale"])}
    # Teacher prediction changes are defined from actual current states. Free
    # predicted deltas are from their own recurrent current state; subtracting
    # them from actual deltas is diagnostic and not itself endpoint accuracy.
    delta_errors = {}
    for name, predicted, valid in (("teacher", teacher_deltas, teacher_valid),
                                    ("free", free_deltas, free_valid)):
        error = np.full_like(actual_deltas, np.nan)
        error[valid] = state_delta(actual_deltas[valid], predicted[valid])
        delta_errors[name] = error
    traces = {"actual_states": states, "applied_actions": actions, "actual_valid_mask": mask,
              "actual_deltas": actual_deltas, "teacher_predicted_deltas": teacher_deltas,
              "free_predicted_deltas": free_deltas, "teacher_next_states": teacher_next,
              "free_states": free_states, "teacher_inference_valid_mask": teacher_valid,
              "free_inference_valid_mask": free_valid, "teacher_failure_reason": teacher_reasons,
              "free_failure_reason": free_reasons, "teacher_delta_errors": delta_errors["teacher"],
              "free_delta_errors": delta_errors["free"]}
    for prefix, columns in {**support_traces, **metrics}.items():
        traces.update({f"{prefix}_{key}": value for key, value in columns.items()})
    by_step = []
    for step in range(horizon):
        record = {"step_1based": step+1, "actual_valid_transitions": int(mask[:, step].sum())}
        for name, valid in (("teacher", teacher_valid), ("free", free_valid)):
            record[name] = {
                "valid_inference_transitions": int(valid[:, step].sum()),
                "invalid_or_unavailable_inference_transitions": int((mask[:, step] & ~valid[:, step]).sum()),
                "metrics_valid_inference_only": {key: _distribution(value[valid[:, step], step])
                    for key, value in metrics[name].items() if key != "state_errors"}}
        for name, columns in support_traces.items():
            input_valid = columns["feature_valid_mask"][:, step]
            record[name] = {
                "available_input_transitions": int(input_valid.sum()),
                "nearest_train_distance": _distribution(columns["nearest_train_distance"][input_valid, step]),
                "outside_marginal_range_by_feature": {
                    field: int(columns["feature_outside_train_range"][input_valid, step, index].sum())
                    for index, field in enumerate(FEATURE_FIELDS)}}
        by_step.append(record)
    invalid_details = {}
    for name, valid, reasons in (("teacher", teacher_valid, teacher_reasons), ("free", free_valid, free_reasons)):
        invalid_details[name] = [{"candidate_index": int(row), "step_1based": int(step+1),
                                  "reason": str(reasons[row, step])}
                                 for row, step in np.argwhere(mask & ~valid)]
    summary = {
        "format": "p4-fixed-sequence-model-diagnostic-v1", "candidate_count": batch, "horizon": horizon,
        "actual_valid_transitions": int(mask.sum()), "actual_unmeasured_transitions": int((~mask).sum()),
        "state_fields": list(STATE_FIELDS), "state_units": list(STATE_UNITS),
        "teacher_forcing": "Recorded current state + recorded applied command -> one predicted next state",
        "free_prediction": "Initial recorded state + recurrent predicted states + same applied commands; no later truth input",
        "yaw_primary": "2pi wrapped learning state and delta error",
        "yaw_secondary": "pi-symmetric task yaw error; does not replace 2pi learning error",
        "validity_scope": "Raw inference/integration only. No planner geometry/workspace guard or task-success evaluation.",
        "support": deepcopy(support.metadata),
        "by_step": by_step, "invalid_inference_details": invalid_details,
        "claims": "Post-execution descriptive diagnostics; no causal attribution, held-out generalization, policy improvement, or task-effect claim",
    }
    support.validate()
    return ModelSequenceDiagnostics(summary, traces)
