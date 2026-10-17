"""Held-out free prediction diagnostics, without simulator futures or clipping.

Each window starts from its recorded first state and thereafter feeds the
model's own state back in. Only the recorded applied actions are supplied.
The checkpoint score uses the K=5 endpoint, three equally weighted first-step
contact regimes, and 2*pi yaw error. An invalid window is retained in counts
and traces and makes selection ineligible; valid-only error summaries are
explicit diagnostics, never a substitute selection population.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .data import DataContractError
from .preparation import SPLIT_CODES, REGIME_CODES
from .representation import STATE_FIELDS, STATE_UNITS, RepresentationError, integrate


@dataclass(frozen=True)
class PredictionEvaluation:
    summary: dict
    traces: dict[str, np.ndarray]


def _require(condition, message):
    if not condition:
        raise DataContractError(message)


def _finite_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _mean(values):
    """Avoid overflowing a sum when each contributing value is finite."""
    if not len(values):
        return None
    maximum = float(np.max(np.abs(values)))
    if maximum == 0:
        return 0.0
    return _finite_number(np.mean(values/maximum)*maximum)


def _statistics(values, *, signed=False):
    if not len(values):
        return {"mean": None, "rmse": None, "p95_absolute": None,
                **({"mae": None} if signed else {})}
    absolute = np.abs(values)
    maximum = float(absolute.max())
    rms = 0.0 if maximum == 0 else _finite_number(maximum*np.sqrt(np.mean((values/maximum)**2)))
    result = {"mean": _mean(values), "rmse": rms,
              "p95_absolute": _finite_number(np.percentile(absolute, 95))}
    if signed:
        result["mae"] = _mean(absolute)
    return result


def _metrics(errors, delta_scale):
    components = {field: {"unit": unit, **_statistics(errors[:, index], signed=True)}
                  for index, (field, unit) in enumerate(zip(STATE_FIELDS, STATE_UNITS))}
    groups = {
        "object_xy_distance_m": np.hypot(errors[:, 0], errors[:, 1]),
        "object_yaw_absolute_rad": np.abs(errors[:, 2]),
        "object_linear_velocity_distance_m_s": np.hypot(errors[:, 3], errors[:, 4]),
        "object_angular_velocity_absolute_rad_s": np.abs(errors[:, 5]),
        "pusher_xy_distance_m": np.hypot(errors[:, 6], errors[:, 7]),
        "pusher_linear_velocity_distance_m_s": np.hypot(errors[:, 8], errors[:, 9]),
    }
    result = {"state_components": components,
              **{name: _statistics(values) for name, values in groups.items()}}
    result["normalized_state_rmse"] = None
    if delta_scale is not None and len(errors):
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = errors/delta_scale
        if np.isfinite(scaled).all():
            result["normalized_state_rmse"] = _statistics(scaled.reshape(-1))["rmse"]
    return result


def _windows(prepared, split, horizon):
    _require(split in SPLIT_CODES, "Unknown evaluation split")
    _require(horizon in prepared.windows, "Requested prediction horizon is absent")
    arrays = prepared.arrays
    n = len(arrays["current_state"])
    for key, width in (("current_state", 10), ("next_state", 10), ("applied_action", 2)):
        value = arrays[key]
        _require(isinstance(value, np.ndarray) and value.dtype == np.float64
                 and value.shape == (n, width) and np.isfinite(value).all(),
                 f"Invalid physical evaluation column {key}")
    for key in ("split_code", "regime_code", "episode_index", "transition_index"):
        _require(isinstance(arrays[key], np.ndarray) and arrays[key].shape == (n,)
                 and np.issubdtype(arrays[key].dtype, np.integer), f"Invalid evaluation identity {key}")
    _require(np.isin(arrays["split_code"], list(SPLIT_CODES.values())).all()
             and np.isin(arrays["regime_code"], list(REGIME_CODES.values())).all(),
             "Evaluation split or regime code is invalid")
    windows = prepared.windows[horizon]
    _require(isinstance(windows, np.ndarray) and windows.dtype == np.int64
             and windows.ndim == 2 and windows.shape[1] == horizon,
             "Invalid window table shape or dtype")
    _require(np.all((windows >= 0) & (windows < n)), "Window row is outside the preparation")
    _require(len(set(map(tuple, windows))) == len(windows), "Duplicated evaluation window")
    if len(windows):
        first = windows[:, 0]
        _require(np.all(arrays["episode_index"][windows] == arrays["episode_index"][first, None])
                 and np.all(arrays["split_code"][windows] == arrays["split_code"][first, None])
                 and np.all(np.diff(arrays["transition_index"][windows], axis=1) == 1),
                 "Window crosses an episode, split or excluded transition gap")
        if horizon > 1:
            _require(np.array_equal(arrays["next_state"][windows[:, :-1]],
                                    arrays["current_state"][windows[:, 1:]]),
                     "Window truth states are not continuous")
        windows = windows[arrays["split_code"][first] == SPLIT_CODES[split]].copy()
    return windows


def _physical_delta(model, normalizer, state, action):
    """Inference boundary: f64 physical NumPy -> f64 physical CPU Torch."""
    import torch
    from .dynamics import predict_delta
    with torch.no_grad():
        output = predict_delta(model, normalizer, torch.from_numpy(state.copy()),
                               torch.from_numpy(action.copy()))
    return output.detach().cpu().numpy()


def _attempt_deltas(predictor, state, action):
    """A rejected batch is diagnosed row by row, retaining its valid peers."""
    count = len(state)
    deltas = np.full((count, 10), np.nan, dtype=np.float64)
    reasons = np.full(count, "", dtype="U160")
    try:
        result = np.asarray(predictor(state, action))
        _require(result.shape == (count, 10) and result.dtype == np.float64,
                 "Physical predictor returned an incompatible shape or dtype")
        deltas[:] = result
    except (RepresentationError, ValueError, FloatingPointError) as error:
        for index in range(count):
            try:
                result = np.asarray(predictor(state[index:index+1], action[index:index+1]))
                _require(result.shape == (1, 10) and result.dtype == np.float64,
                         "Physical predictor returned an incompatible shape or dtype")
                deltas[index] = result[0]
            except (RepresentationError, ValueError, FloatingPointError) as row_error:
                reasons[index] = f"inference_rejected:{type(row_error).__name__}:{row_error}"[:160]
    nonfinite = ~np.isfinite(deltas).all(axis=1)
    reasons[nonfinite & (reasons == "")] = "nonfinite_physical_delta"
    yaw_invalid = ~nonfinite & (np.abs(deltas[:, 2]) >= math.pi)
    reasons[yaw_invalid & (reasons == "")] = "raw_delta_yaw_abs_ge_pi"
    return deltas, reasons


def _evaluate(prepared, split, horizons, predictor, delta_scale, kind):
    horizons = tuple(horizons)
    _require(bool(horizons) and all(type(k) is int and k > 0 for k in horizons)
             and len(set(horizons)) == len(horizons), "Horizons must be distinct positive integers")
    task = prepared.metadata["contract"]["resolved_config"]["task"]
    position_tolerance, yaw_tolerance = task["position_tolerance_m"], task["yaw_tolerance_rad"]
    _require(np.isfinite([position_tolerance, yaw_tolerance]).all()
             and min(position_tolerance, yaw_tolerance) > 0, "Invalid checkpoint task tolerances")
    if delta_scale is not None:
        _require(isinstance(delta_scale, np.ndarray) and delta_scale.shape == (10,)
                 and np.isfinite(delta_scale).all() and (delta_scale > 0).all(), "Invalid rollout error scale")
    arrays, traces, horizon_summaries = prepared.arrays, {}, {}
    episode_lookup = {episode["episode_index"]: episode for episode in prepared.metadata["episodes"]}
    for horizon in horizons:
        rows = _windows(prepared, split, horizon)
        count = len(rows)
        first = rows[:, 0]
        regimes = arrays["regime_code"][first].copy()
        episode_indices = arrays["episode_index"][first].copy()
        _require(all(int(index) in episode_lookup for index in episode_indices), "Window episode identity is absent")
        truth = np.concatenate((arrays["current_state"][first, None], arrays["next_state"][rows]), axis=1)
        predicted = np.full((count, horizon+1, 10), np.nan, dtype=np.float64)
        predicted[:, 0] = truth[:, 0]
        valid = np.ones(count, dtype=np.bool_)
        reasons = np.full(count, "", dtype="U160")
        failure_steps = np.full(count, -1, dtype=np.int64)
        for step in range(horizon):
            active = np.flatnonzero(valid)
            if not len(active):
                break
            states = predicted[active, step]
            actions = arrays["applied_action"][rows[active, step]]
            deltas, step_reasons = _attempt_deltas(predictor, states, actions)
            okay = step_reasons == ""
            if okay.any():
                selected = np.flatnonzero(okay)
                # integrate's numerical overflow rejection is batch-global;
                # precheck additions so a single extreme row cannot hide peers.
                with np.errstate(over="ignore", invalid="ignore"):
                    summed = states[selected]+deltas[selected]
                finite_sum = np.isfinite(summed).all(axis=1)
                step_reasons[selected[~finite_sum]] = "nonfinite_integrated_state"
                selected = selected[finite_sum]
                if len(selected):
                    predicted[active[selected], step+1] = integrate(states[selected], deltas[selected])
            failed = step_reasons != ""
            valid[active[failed]] = False
            reasons[active[failed]] = step_reasons[failed]
            failure_steps[active[failed]] = step+1
        errors = np.full((count, 10), np.nan, dtype=np.float64)
        active = np.flatnonzero(valid)
        if len(active):
            with np.errstate(over="ignore", invalid="ignore"):
                errors[active] = predicted[active, -1]-truth[active, -1]
                errors[active, 2] = (errors[active, 2]+math.pi) % (2*math.pi)-math.pi
            finite_error = np.isfinite(errors[active]).all(axis=1)
            failed = active[~finite_error]
            valid[failed] = False
            errors[failed] = np.nan
            reasons[failed] = "nonfinite_endpoint_error"
            failure_steps[failed] = horizon
        group_summaries = {}
        for name, code in REGIME_CODES.items():
            attempted = regimes == code
            selected = attempted & valid
            e = errors[selected]
            with np.errstate(over="ignore", invalid="ignore"):
                selection_terms = np.hypot(e[:, 0], e[:, 1])/position_tolerance+np.abs(e[:, 2])/yaw_tolerance
            selection_mean = _mean(selection_terms) if np.isfinite(selection_terms).all() else None
            group_summaries[name] = {
                "attempted_windows": int(attempted.sum()), "valid_windows": int(selected.sum()),
                "invalid_windows": int((attempted & ~valid).sum()),
                "metrics_valid_windows_only": _metrics(e, delta_scale),
                "selection_error_mean_valid_windows_only": selection_mean,
            }
        invalid_details = [{"window_index": int(index), "episode_index": int(episode_indices[index]),
                            "episode_id": episode_lookup[int(episode_indices[index])]["episode_id"],
                            "start_transition_index": int(arrays["transition_index"][first[index]]),
                            "regime": next(name for name, code in REGIME_CODES.items() if code == int(regimes[index])),
                            "failure_step": int(failure_steps[index]), "reason": str(reasons[index])}
                           for index in np.flatnonzero(~valid)]
        horizon_summaries[str(horizon)] = {
            "attempted_windows": count, "valid_windows": int(valid.sum()),
            "invalid_windows": int((~valid).sum()), "invalid_details": invalid_details,
            "metrics_valid_windows_only": _metrics(errors[valid], delta_scale), "regimes": group_summaries,
        }
        prefix = f"h{horizon}_"
        traces.update({prefix+key: value for key, value in {
            "window_rows": rows, "predicted_states": predicted, "truth_states": truth,
            "endpoint_errors": errors, "valid": valid, "regime_code": regimes,
            "episode_index": episode_indices,
            "episode_id": np.asarray([episode_lookup[int(e)]["episode_id"] for e in episode_indices], dtype=np.str_),
            "family_id": np.asarray([episode_lookup[int(e)]["family_id"] for e in episode_indices], dtype=np.str_),
            "start_transition_index": arrays["transition_index"][first].copy(),
            "failure_step": failure_steps, "failure_reason": reasons,
        }.items()})
    score, status = None, "missing_selection_horizon"
    if "5" in horizon_summaries:
        selected = horizon_summaries["5"]
        if selected["invalid_windows"]:
            status = "invalid_selection_windows"
        elif any(group["attempted_windows"] == 0 for group in selected["regimes"].values()):
            status = "missing_selection_regime"
        else:
            values = [group["selection_error_mean_valid_windows_only"] for group in selected["regimes"].values()]
            if all(value is not None for value in values):
                score = _mean(np.asarray(values, dtype=np.float64))
                status = "eligible" if score is not None else "nonfinite_selection_metric"
            else:
                status = "nonfinite_selection_metric"
    summary = {
        "schema": "push-align-prediction-evaluation-v1", "kind": kind, "split": split,
        "horizons": horizon_summaries, "selection_score": score, "selection_status": status,
        "selection_horizon": 5, "time_aggregation": "endpoint", "yaw_error": "wrap_2pi",
        "regime_basis": "first_transition_substep_contact_pattern",
        "selection_definition": "equal mean of three regime means of object XY distance / task position tolerance + absolute 2pi yaw error / task yaw tolerance",
        "position_tolerance_m": float(position_tolerance), "yaw_tolerance_rad": float(yaw_tolerance),
        "invalid_window_policy": "retain every attempted window; any invalid K5 window prevents checkpoint selection",
        "window_dependence": "overlapping windows are descriptive samples, not independent trials",
        "scope": "recorded-action free prediction; task alignment and action selection are not assessed",
    }
    return PredictionEvaluation(summary, traces)


def evaluate_predictions(model, normalizer, prepared, split="validation", horizons=(1, 5, 10)):
    """Evaluate a learned model and restore its incoming train/eval mode."""
    normalizer.validate()
    training = model.training
    model.eval()
    try:
        return _evaluate(prepared, split, horizons,
                         lambda state, action: _physical_delta(model, normalizer, state, action),
                         normalizer.arrays["delta_scale"], "learned_dynamics")
    finally:
        model.train(training)


def evaluate_state_hold(prepared, split="validation", horizons=(1, 5, 10)):
    """No-response diagnostic: hold all ten initial state values constant.

    This is a prediction reference only, not a tuned geometric controller or
    a Ridge dynamics baseline, and does not establish task-control superiority.
    """
    return _evaluate(prepared, split, horizons, lambda state, action: np.zeros_like(state),
                     None, "state_hold_diagnostic")
