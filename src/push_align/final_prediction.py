"""Offline K1/5/10 prediction on predeclared, common E2 G raw trajectories.

Recorded applied actions drive model-only recursive predictions. All raw start
positions are enumerated before inference, including incomplete suffixes and
invalid truth windows. These overlapping windows are not independent families.
The caller binds the twelve G records to the frozen manifest and raw hashes.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
import json

import numpy as np

from .closed_loop import ENGINE_GUARDS, model_planning_scope
from .environment import STATE_FIELDS, StepResult, wrap_angle
from .final_manifest import CONTROLLERS, STRATA
from .planning import rollout

MODELS = CONTROLLERS[1:]
REGIMES = ("noncontact", "sustained_contact", "transition", "unknown")
ERROR_METRICS = (
    "object_xy_error_m", "object_yaw_error_rad_2pi", "pusher_xy_error_m",
    "object_linear_velocity_error_m_s", "object_angular_velocity_error_rad_s",
    "pusher_velocity_error_m_s",
)
TRUTH_MASK_PROVENANCE = "E2 recorded transition+observation+actual_contact_geometry"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _array(value, shape, name, *, finite=True):
    _require(isinstance(value, np.ndarray) and value.dtype == np.float64
             and value.shape == shape, name + " requires float64 " + str(shape))
    if finite:
        _require(bool(np.isfinite(value).all()), name + " requires finite values")
    return value


def _strings(values):
    return np.asarray(values, dtype=f"U{max([1, *(len(str(x)) for x in values)])}")


def _mean(values):
    values = np.asarray(values, np.float64)
    if not values.size:
        return None
    _require(bool(np.isfinite(values).all()), "Nonfinite metric cannot enter a mean")
    scale = float(np.max(np.abs(values)))
    return 0. if scale == 0 else float(np.mean(values / scale) * scale)


def _metric_summary(values):
    values = np.asarray(values, np.float64)
    if not len(values):
        return {"denominator": 0, "mean": None, "median": None,
                "p95": None, "max": None}
    _require(bool(np.isfinite(values).all()), "Metric summary requires finite values")
    scale = float(np.max(np.abs(values)))
    percentiles = np.percentile(values / scale, [50, 95]) * scale if scale else np.zeros(2)
    return {"denominator": len(values), "mean": _mean(values),
            "median": float(percentiles[0]), "p95": float(percentiles[1]),
            "max": float(np.max(values))}


def _interval(step, before, previous_time, physical, search_cap):
    """Validate serialized raw wiring; retain real invalid physics as exclusions."""
    _require(isinstance(step, StepResult), "Truth results require typed StepResult")
    _array(step.requested_action, (2,), "requested_action")
    action = _array(step.applied_action, (2,), "applied_action")
    _require(np.hypot(*action) <= physical["pusher"]["command_speed_norm_max_m_s"] + 1e-15,
             "Recorded applied action exceeds the physical norm cap")
    _require(np.hypot(*action) <= search_cap + 1e-15,
             "Recorded applied action exceeds bound G/planner cap; never clip diagnostic actions")
    _require(type(step.transition_valid) is bool
             and type(step.observation.observation_valid) is bool,
             "Explicit raw validity flags required")
    _require(step.observation.schema_version == physical["schema_version"],
             "Raw observation schema differs from physical contract")
    endpoint = _array(step.observation.state, (10,), "observation.state", finite=False)
    _require(type(step.actual_dt) in (int, float) and math.isfinite(step.actual_dt)
             and step.actual_dt >= 0, "Finite nonnegative actual interval required")
    _require(isinstance(step.substep_states, np.ndarray) and step.substep_states.ndim == 2
             and step.substep_states.shape[1] == 10, "Raw substeps require [n,10]")
    n = len(step.substep_states)
    _array(step.substep_states, (n, 10), "substep_states", finite=False)
    _array(step.substep_times, (n,), "substep_times", finite=False)
    diagnostic = step.diagnostics
    _require(isinstance(diagnostic, dict)
             and tuple(diagnostic.get("state_fields", ())) == STATE_FIELDS,
             "Raw diagnostic field order required")
    samples = diagnostic.get("samples")
    _require(isinstance(samples, list) and len(samples) == n + 1,
             "Raw diagnostics must contain start plus every substep")
    states = np.asarray([sample["state"] for sample in samples], np.float64)
    times = np.asarray([sample["sim_time"] for sample in samples], np.float64)
    _require(states.shape == (n + 1, 10), "Diagnostic raw state shape mismatch")
    _require(np.array_equal(states[0], before, equal_nan=True),
             "Raw start state differs from actual previous boundary")
    _require(np.array_equal(states[1:], step.substep_states, equal_nan=True)
             and np.array_equal(times[1:], step.substep_times, equal_nan=True)
             and np.array_equal(states[-1], endpoint, equal_nan=True),
             "Raw diagnostics/substeps/observation disagree")
    _require(math.isfinite(float(step.observation.sim_time))
             and np.isfinite(times).all()
             and abs(times[-1] - step.observation.sim_time) <= 1e-9,
             "Raw timestamp must be finite and match observation")
    _require((previous_time is None or abs(times[0] - previous_time) <= 1e-9)
             and (n == 0 or bool((np.diff(times) > 0).all()))
             and abs(times[-1] - times[0] - step.actual_dt) <= 1e-9,
             "Raw actual interval times are discontinuous")
    active = []
    for sample in samples:
        contacts = sample.get("contacts")
        _require(isinstance(contacts, (list, tuple)), "Raw contacts required for regime labels")
        side_forces = []
        for contact in contacts:
            _require(isinstance(contact, dict) and isinstance(contact.get("pair"), str)
                     and type(contact.get("normal_force_N")) in (int, float),
                     "Contact pair and numeric normal force required")
            if contact["pair"] == "pusher_object":
                side_forces.append(contact["normal_force_N"])
        active.append(None if not np.isfinite(side_forces).all() else
                      any(force > physical["evaluation"]["contact_active_normal_force_min_N"]
                          for force in side_forces))
    regime = ("unknown" if any(value is None for value in active) else
              "sustained_contact" if all(active) else
              "transition" if any(active) else "noncontact")
    failures = []
    if not step.transition_valid: failures.append("transition_invalid")
    if not step.observation.observation_valid: failures.append("observation_invalid")
    if not np.isfinite(states).all(): failures.append("nonfinite_raw_state")
    if abs(step.actual_dt - physical["physics"]["action_dt_s"]) > 1e-9:
        failures.append("incomplete_action_interval")
    if regime == "unknown": failures.append("nonfinite_contact_force")
    if diagnostic.get("validity_violations"):
        failures.append("recorded_validity_violation")
    _require(not (step.transition_valid and failures),
             "Claimed-valid raw interval contradicts its values: " + str(failures))
    return endpoint.copy(), action.copy(), regime, "|".join(failures), float(times[-1])


def _population(trajectories, physical, planner):
    # A native 300-step diagnostic record may occupy gigabytes as Python
    # objects. Never materialize all twelve raw rows from a caller's generator.
    # Each row is reduced to small numeric arrays before requesting the next.
    rows = iter(trajectories.values() if isinstance(trajectories, dict) else trajectories)
    cap = planner["planner"].get("command_speed_norm_max_m_s",
                                  physical["pusher"]["command_speed_norm_max_m_s"])
    prepared, seen = [], set()
    for row in rows:
        _require(isinstance(row["family_id"], str) and row["family_id"]
                 and row.get("controller_id", "G") == "G", "Only G source trajectories allowed")
        _require(row["family_id"] not in seen and len(prepared) < 12,
                 "Exactly twelve distinct final G families required")
        _require(row["stratum"] in STRATA, "Unknown final G stratum")
        seen.add(row["family_id"])
        before = _array(row["initial_state10"], (10,), "initial_state10").copy()
        states, actions, regimes, reasons = [before], [], [], []
        _require(isinstance(row["results"], (list, tuple)), "Explicit raw result sequence required")
        mask = row.get("interval_valid")
        _require(isinstance(mask, np.ndarray) and mask.dtype == np.bool_
                 and mask.shape == (len(row["results"]),),
                 "Explicit interval_valid bool ndarray[T] required")
        _require(row.get("truth_mask_provenance") == TRUTH_MASK_PROVENANCE,
                 "Recorded transition/observation/actual-contact-geometry provenance required")
        previous_time = row.get("initial_sim_time")
        for index, step in enumerate(row["results"]):
            _require(isinstance(step, StepResult), "Truth results require typed StepResult")
            _require(not mask[index] or (step.transition_valid and step.observation.observation_valid),
                     "Execution validity mask cannot promote invalid raw flags")
            before, action, regime, reason, previous_time = _interval(
                step, before, previous_time, physical, cap)
            if not mask[index]:
                reason = "|".join(filter(None, (reason, "recorded_execution_validity_mask_false")))
            states.append(before); actions.append(action); regimes.append(regime); reasons.append(reason)
        prepared.append({"family_id": row["family_id"], "stratum": row["stratum"],
                         "states": np.asarray(states, np.float64),
                         "actions": np.asarray(actions, np.float64).reshape(-1, 2),
                         "regimes": regimes, "interval_reasons": reasons})
        if row["results"]:
            del step
        del row, mask
    _require(len(prepared) == 12, "Exactly twelve distinct final G families required")
    _require(all(sum(row["stratum"] == s for row in prepared) == 4 for s in STRATA),
             "Exactly four final G families per stratum required")
    return sorted(prepared, key=lambda row: row["family_id"])


def _windows(prepared, horizon):
    ids, families, strata, starts, regimes, reasons, truth, actions = ([] for _ in range(8))
    for row in prepared:
        n = len(row["actions"])
        for start in range(n):
            ids.append(f"{row['family_id']}/G/start-{start:03d}/K{horizon}")
            families.append(row["family_id"]); strata.append(row["stratum"]); starts.append(start)
            regimes.append(REGIMES.index(row["regimes"][start]))
            observed = np.full((horizon + 1, 10), np.nan, np.float64)
            applied = np.full((horizon, 2), np.nan, np.float64)
            remaining = min(horizon, n - start)
            observed[:remaining + 1] = row["states"][start:start + remaining + 1]
            applied[:remaining] = row["actions"][start:start + remaining]
            truth.append(observed); actions.append(applied)
            if remaining < horizon:
                reasons.append("insufficient_remaining_intervals")
            else:
                bad = [f"offset{offset}:{reason}" for offset, reason in
                       enumerate(row["interval_reasons"][start:start + horizon]) if reason]
                if not np.isfinite(observed).all(): bad.append("nonfinite_truth_boundary")
                reasons.append(";".join(bad))
    count = len(ids)
    return {"window_id": _strings(ids), "family_id": _strings(families),
            "stratum": _strings(strata), "start_tick": np.asarray(starts, np.int64),
            "regime_code": np.asarray(regimes, np.int64), "truth_reason": _strings(reasons),
            "truth_valid": np.asarray([not value for value in reasons], np.bool_),
            "truth_states": np.asarray(truth, np.float64).reshape(count, horizon + 1, 10),
            "applied_actions": np.asarray(actions, np.float64).reshape(count, horizon, 2)}


def _group_summary(windows, valid, reasons, metrics, mask):
    truth = mask & windows["truth_valid"]
    evaluated = truth & valid
    return {"recorded_start_opportunities": int(mask.sum()), "truth_eligible_windows": int(truth.sum()),
            "predicted_valid_windows": int(evaluated.sum()),
            "predicted_invalid_windows": int((truth & ~valid).sum()),
            "prediction_invalid_reasons": dict(Counter(str(value) for value in reasons[truth & ~valid])),
            "metrics_conditional_on_valid_prediction_and_truth": {
                name: _metric_summary(values[evaluated]) for name, values in metrics.items()}}


def evaluate_final_predictions(trajectories, predictors, physical_config, planner_config,
                               *, horizons=(1, 5, 10)):
    """Return JSON summary and non-object NPZ arrays; no new physical execution.

    A trajectory is a mapping with family_id, stratum, initial_state10 and
    results (typed StepResults), interval_valid (bool ndarray[T]), and
    truth_mask_provenance. The mask must include the original actual contact
    geometry check, not just the StepResult validity flags. An optional initial_sim_time binds the first
    raw timestamp; optional controller_id, if supplied, must be G. Source-file
    identities and hashes remain caller-owned frozen-manifest evidence. Pass
    a generator which releases each raw row after yield for bounded memory;
    this function retains compact numeric arrays, never prior StepResults.
    """
    _require(tuple(horizons) == (1, 5, 10), "Frozen final horizons K1/5/10 required")
    _require(isinstance(predictors, dict) and set(predictors) == set(MODELS)
             and all(callable(value) for value in predictors.values()),
             "Exactly seven frozen physical-delta predictor callables required")
    _require(planner_config["planner"]["horizon"] == 5,
             "Online planner must retain frozen H5")
    prepared = _population(trajectories, physical_config, planner_config)
    guards = dict.fromkeys(ENGINE_GUARDS, 0)
    calls = {name: {"calls": 0, "rows": 0} for name in MODELS}
    summary, arrays = {}, {}
    for horizon in horizons:
        window = _windows(prepared, horizon)
        count = len(window["window_id"])
        for name, values in window.items(): arrays[f"K{horizon}__{name}"] = values
        eligible = np.flatnonzero(window["truth_valid"])
        diagnostic = deepcopy(planner_config)
        diagnostic["planner"]["horizon"] = horizon
        models = {}
        for name in MODELS:
            predicted = np.full((count, horizon + 1, 10), np.nan, np.float64)
            deltas = np.full((count, horizon, 10), np.nan, np.float64)
            valid = np.zeros(count, np.bool_)
            reasons = np.full(count, "not_evaluated_truth_ineligible", dtype="U200")
            failure_steps = np.full(count, -2, np.int64)
            if len(eligible):
                def counted(state, action, *, model_name=name):
                    calls[model_name]["calls"] += 1; calls[model_name]["rows"] += len(state)
                    return predictors[model_name](state, action)
                with model_planning_scope(guards):
                    result = rollout(window["truth_states"][eligible, 0].copy(),
                                     window["applied_actions"][eligible].copy(), counted, diagnostic)
                predicted[eligible] = result.states; deltas[eligible] = result.raw_deltas
                valid[eligible] = result.valid; reasons[eligible] = result.reasons
                failure_steps[eligible] = result.failure_steps
            errors = np.full((count, 10), np.nan, np.float64)
            active = np.flatnonzero(valid)
            if len(active):
                with np.errstate(over="ignore", invalid="ignore"):
                    errors[active] = predicted[active, -1] - window["truth_states"][active, -1]
                    errors[active, 2] = wrap_angle(errors[active, 2])
                finite = np.isfinite(errors[active]).all(axis=1)
                invalid = active[~finite]
                valid[invalid] = False; reasons[invalid] = "nonfinite_endpoint_error"
                failure_steps[invalid] = horizon; errors[invalid] = np.nan
            with np.errstate(over="ignore", invalid="ignore"):
                metrics = dict(zip(ERROR_METRICS, (
                    np.hypot(errors[:, 0], errors[:, 1]), np.abs(errors[:, 2]),
                    np.hypot(errors[:, 6], errors[:, 7]), np.hypot(errors[:, 3], errors[:, 4]),
                    np.abs(errors[:, 5]), np.hypot(errors[:, 8], errors[:, 9]))))
            metric_finite = np.logical_and.reduce([np.isfinite(value) for value in metrics.values()])
            invalid = valid & ~metric_finite
            valid[invalid] = False; reasons[invalid] = "nonfinite_endpoint_metric"
            failure_steps[invalid] = horizon; errors[invalid] = np.nan
            for values in metrics.values(): values[invalid] = np.nan
            prefix = f"K{horizon}__{name}__"
            for key, values in {"predicted_states": predicted, "raw_deltas": deltas,
                                "valid": valid, "reasons": reasons, "failure_steps": failure_steps,
                                "endpoint_state_error": errors, **metrics}.items(): arrays[prefix + key] = values
            families = {row["family_id"]: {"stratum": row["stratum"],
                        **_group_summary(window, valid, reasons, metrics,
                        window["family_id"] == row["family_id"])} for row in prepared}
            family_means = {metric: [family["metrics_conditional_on_valid_prediction_and_truth"][metric]["mean"]
                                    for family in families.values()] for metric in ERROR_METRICS}
            models[name] = {**_group_summary(window, valid, reasons, metrics, np.ones(count, np.bool_)),
                "by_first_interval_regime": {regime: _group_summary(window, valid, reasons, metrics,
                    window["regime_code"] == code) for code, regime in enumerate(REGIMES)},
                "by_family": families,
                "family_first_means_conditional_on_available_families": {
                    metric: {"family_denominator": sum(value is not None for value in values),
                             "mean": _mean([value for value in values if value is not None])}
                    for metric, values in family_means.items()}}
        boundary = window["truth_reason"] == "insufficient_remaining_intervals"
        summary[str(horizon)] = {
            "recorded_start_opportunities": count, "boundary_excluded_windows": int(boundary.sum()),
            "full_horizon_windows": int((~boundary).sum()),
            "raw_invalid_full_windows": int((~boundary & ~window["truth_valid"]).sum()),
            "truth_eligible_windows": len(eligible),
            "prediction_comparison_status": "EVALUATED_ON_ELIGIBLE_WINDOWS" if len(eligible) else "NO_ELIGIBLE_GROUND_TRUTH_WINDOWS",
            "truth_exclusion_reasons": dict(Counter(str(value) for value in window["truth_reason"][~window["truth_valid"]])),
            "models": models}
    result = {"schema": "p5-common-G-fixed-trace-prediction-v1", "status": "COMPLETE_DECLARED_RAW_EVALUATION",
            "family_count": 12, "family_ids": [row["family_id"] for row in prepared],
            "source": "all_E2_G_raw_fixed_applied_actions", "horizons": summary,
            "regime_codes": dict(enumerate(REGIMES)), "state_fields": list(STATE_FIELDS),
            "yaw_error": "signed endpoint residual wrapped2pi; absolute scalar metric, not pi-symmetric task angle",
            "recorded_intervals_by_family": {row["family_id"]: len(row["actions"]) for row in prepared},
            "source_interval_populations": {row["family_id"]: {"stratum": row["stratum"],
                "recorded_intervals": len(row["actions"]),
                "valid_intervals": sum(not reason for reason in row["interval_reasons"]),
                "invalid_intervals": sum(bool(reason) for reason in row["interval_reasons"]),
                "invalid_reasons": dict(Counter(reason for reason in row["interval_reasons"] if reason))}
                for row in prepared},
            "truth_mask_provenance": TRUTH_MASK_PROVENANCE,
            "failure_step_codes": {"truth_not_evaluated": -2, "valid_prediction": -1,
                "invalid_initial_predicted_state": 0, "prediction_transition": "1 through K"},
            "inference": calls, "future_engine_calls": guards,
            "online_settings_mutated": False, "additional_physics_intervals": 0,
            "limitations": "Overlapping windows are not independent samples; descriptive valid-only errors retain truth/model invalid counts. Different models may have different valid windows: paired effects require common-window/family eligibility and invalid rates. Zero eligible windows do not prove model quality. All predictions are offline and never fed to online control. Raw/source provenance and actual E2 completion are caller-owned."}
    json.dumps(result, allow_nan=False)
    return result, arrays
