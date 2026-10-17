"""Pure action-boundary witnesses for fresh controller reproduction.

A witness is not an executor: the driver must run the controller again, never
feed recorded actions to it, and bind that run's identity and source hashes.
The original lossless raw trace remains the evidence for physical substeps and
continuous holding. This compact witness only compares action boundaries.
Planner hashes are deliberately exact for the same runtime; numeric boundary
comparisons retain the existing acceptance profile's 0.001 task-scale budget.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

SCHEMA = "p5-controller-boundary-witness-v1"
ENGINE_GUARDS = ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse")
COUNT_FIELDS = ("controller_decisions", "physics_interval_attempts",
                "physics_intervals_returned", "completed_boundary_records")
BOUNDARY_FIELDS = (
    "index", "local_index", "source", "state_before", "state_after", "sim_time_before", "sim_time_after",
    "requested_action", "applied_action", "actual_dt_s", "holding",
    "task_before", "task_after", "geometry", "observation_valid", "transition_valid",
    "planner_context_before", "planner_context_after", "geometry_context_before",
    "geometry_context_after", "plan_identity", "rng_keys",
)
HASH_FIELDS = ("candidate_hash", "prediction_hash", "choice_hash")


def json_safe(value):
    """Return lossless JSON-compatible scalar/list values, tagging nonfinites."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, Path):
        return str(value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"Unsupported witness value: {type(value).__name__}")


def value_hash(value):
    return hashlib.sha256(json.dumps(json_safe(value), ensure_ascii=False,
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def raw_reference(paths, executed_intervals):
    """Bind saved complete raw files; content/schema validation is driver-owned."""
    if type(executed_intervals) is not int or executed_intervals < 0:
        raise ValueError("executed_intervals must be a nonnegative integer")
    files = {str(Path(path)): _sha(path) for path in paths}
    if not files:
        raise ValueError("A raw reference requires saved files")
    return {"files": files, "executed_intervals": executed_intervals,
            "scope": "complete_raw_execution_prefix"}


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def make_boundary(record):
    """Copy an executor interval, preserving omissions for INCOMPLETE checks."""
    return {key: json_safe(record[key]) for key in BOUNDARY_FIELDS if key in record}


def build_witness(execution, *, identity, source_bindings, raw_reference,
                  attempt_id, original_raw_reference=None):
    """Bind fresh executor output without replacing absent evidence by defaults.

    ``raw_reference`` may be None only for a normal fresh reproduction which
    references the original complete raw. On mismatch/physical failure the
    driver saves its new full executed prefix and attaches that reference.
    """
    observation = _get(execution, "initial_observation")
    counts = _get(execution, "counts")
    if counts is None:
        counts = {key: _get(execution, key) for key in COUNT_FIELDS}
    witness = {
        "schema": SCHEMA, "identity": identity, "source_bindings": source_bindings,
        "attempt_id": attempt_id, "execution_id": _get(execution, "execution_id"),
        "record_kind": ("fresh_controller_replay" if original_raw_reference is not None
                        else "reference_controller_execution"),
        "origin": _get(execution, "origin", _get(execution, "initial_input_origin")),
        "controller_id": _get(execution, "controller_id"),
        "case_id": _get(execution, "case_id"), "run_scope": _get(execution, "run_scope"),
        "initial_full_snapshot_equal": _get(execution, "initial_full_snapshot_equal"),
        "initial": {"state": _get(observation, "state"),
                    "sim_time": _get(observation, "sim_time"),
                    "observation_valid": _get(observation, "observation_valid"),
                    "schema_version": _get(observation, "schema_version"),
                    "snapshot_hash": value_hash(_get(execution, "initial_snapshot")),
                    "task": _get(execution, "initial_task"),
                    "planner_context": _get(execution, "initial_context"),
                    "geometry_context": _get(execution, "initial_gctx")},
        "task_spec": _get(execution, "task_spec"),
        "intervals": [make_boundary(record) for record in _get(execution, "records", [])],
        "decisions": _get(execution, "decisions"),
        "unexecuted_plan": _get(execution, "unexecuted_plan"),
        "final_task": _get(execution, "final_task"),
        "execution_status": _get(execution, "execution_status"),
        "termination": _get(execution, "termination"), "error": _get(execution, "error"),
        "counts": counts, "future_engine_calls": _get(execution, "future_engine_calls"),
        "returned_results": len(_get(execution, "results", [])),
        "raw_reference": raw_reference, "original_raw_reference": original_raw_reference,
        "scope": "fresh controller action-boundary comparison; not substep/raw-byte invariance",
    }
    return json_safe(witness)


def _hashes(value):
    return isinstance(value, dict) and bool(value) and all(
        isinstance(key, str) and key and isinstance(item, str)
        and re.fullmatch(r"[0-9a-f]{64}", item) for key, item in value.items())


def _raw_issues(ref, expected, *, verify_files, label):
    if not isinstance(ref, dict):
        return [f"{label}:missing_raw_reference"]
    issues = []
    if ref.get("scope") != "complete_raw_execution_prefix" or not _hashes(ref.get("files")):
        return [f"{label}:invalid_raw_reference"]
    if type(ref.get("executed_intervals")) is not int or ref["executed_intervals"] != expected:
        issues.append(f"{label}:raw_interval_count_mismatch")
    if verify_files:
        for path, expected_sha in ref["files"].items():
            if not Path(path).is_file() or _sha(path) != expected_sha:
                issues.append(f"{label}:raw_file_missing_or_changed:{path}")
    return issues


def _array(value, width):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (width,):
        raise ValueError(f"Expected vector width {width}")
    return result


def _task_shape(value):
    required = {"phase", "initial_sim_time", "last_sim_time", "held_time_s",
                "previous_qualifies", "terminal_reason", "violations"}
    if not isinstance(value, dict) or set(value) != required:
        return False
    try:
        clocks = [value[key] for key in ("initial_sim_time", "last_sim_time", "held_time_s")]
        return (all(type(item) in (int, float) and math.isfinite(item) and item >= 0 for item in clocks)
                and value["last_sim_time"] >= value["initial_sim_time"]
                and value["held_time_s"] <= value["last_sim_time"]-value["initial_sim_time"]+1e-9
                and type(value["previous_qualifies"]) is bool
                and value["phase"] in ("running", "holding", "ready", "failed")
                and isinstance(value["violations"], list)
                and all(isinstance(item, str) and item for item in value["violations"]))
    except (KeyError, TypeError, ValueError):
        return False


def _task_match(a, b, tolerance):
    """Task labels are exact; its observed clocks use the existing time budget."""
    if not isinstance(a, dict) or not isinstance(b, dict) or set(a) != set(b):
        return False
    clocks = {"initial_sim_time", "last_sim_time", "held_time_s"}
    try:
        return all((math.isfinite(a[key]) and math.isfinite(b[key])
                    and abs(a[key]-b[key]) <= tolerance) if key in clocks
                   else a[key] == b[key] for key in a)
    except (TypeError, ValueError):
        return False


def _validate(witness, label, time_tolerance_s):
    issues = []
    if not isinstance(witness, dict) or witness.get("schema") != SCHEMA:
        return [f"{label}:unsupported_schema"]
    required = ("identity", "source_bindings", "attempt_id", "execution_id", "origin",
                "controller_id", "case_id", "initial", "task_spec", "intervals", "decisions",
                "unexecuted_plan", "final_task", "execution_status", "termination", "error",
                "counts", "future_engine_calls", "returned_results", "raw_reference",
                "original_raw_reference", "record_kind", "run_scope", "initial_full_snapshot_equal")
    issues.extend(f"{label}:missing:{key}" for key in required if key not in witness)
    for key in ("attempt_id", "execution_id", "controller_id", "case_id"):
        if not isinstance(witness.get(key), str) or not witness[key]:
            issues.append(f"{label}:invalid:{key}")
    if not isinstance(witness.get("identity"), dict) or not witness.get("origin"):
        issues.append(f"{label}:missing_identity_or_origin")
    if not _hashes(witness.get("source_bindings")):
        issues.append(f"{label}:missing_source_bindings")
    if witness.get("initial_full_snapshot_equal") is not True:
        issues.append(f"{label}:initial_snapshot_restore_not_exact")
    guards = witness.get("future_engine_calls")
    if not isinstance(guards, dict) or set(guards) != set(ENGINE_GUARDS) or any(
            type(n) is not int or n != 0 for n in guards.values()):
        issues.append(f"{label}:future_engine_guard_violation_or_missing")
    records, counts = witness.get("intervals"), witness.get("counts")
    decisions = witness.get("decisions")
    if not isinstance(records, list) or not isinstance(decisions, list):
        return issues + [f"{label}:missing_records_or_decisions"]
    if not isinstance(counts, dict) or any(type(counts.get(key)) is not int
            or counts[key] < 0 for key in COUNT_FIELDS):
        return issues + [f"{label}:missing_counts"]
    if not (counts["physics_interval_attempts"] == counts["physics_intervals_returned"]
            == counts["completed_boundary_records"] == witness.get("returned_results") == len(records)):
        issues.append(f"{label}:physical_result_boundary_count_mismatch")
    if counts["controller_decisions"] != len(decisions) or len(decisions) < len(records):
        issues.append(f"{label}:decision_count_mismatch")
    expected_decisions = len(records) + int(witness.get("unexecuted_plan") is not None)
    if len(decisions) != expected_decisions or any(not isinstance(row, dict)
            or row.get("executed") is not (index < len(records))
            for index, row in enumerate(decisions)):
        issues.append(f"{label}:decision_execution_record_mismatch")
    if witness.get("execution_status") != "completed" or witness.get("error"):
        issues.append(f"{label}:execution_incomplete")
    initial, final = witness.get("initial"), witness.get("final_task")
    try:
        _array(initial["state"], 10)
        if (initial["schema_version"] != "push-align-state-v1"
                or type(initial["observation_valid"]) is not bool
                or not math.isfinite(initial["sim_time"])
                or not isinstance(initial.get("planner_context"), dict)
                or not isinstance(initial.get("geometry_context"), dict)):
            raise ValueError("invalid initial observation")
        if not _task_shape(initial.get("task")) or not _task_shape(final):
            issues.append(f"{label}:malformed_task_status")
        if initial["observation_valid"] and not np.isfinite(_array(initial["state"], 10)).all():
            issues.append(f"{label}:nonfinite_valid_initial_observation")
        development_prefix = witness.get("run_scope") == "development_preflight" \
            and witness.get("termination") == "development_budget"
        if not isinstance(final, dict) or (development_prefix and final.get("phase") not in ("running", "holding")) or (not development_prefix and (
                final.get("phase") not in ("ready", "failed")
                or not isinstance(final.get("terminal_reason"), str) or not final["terminal_reason"])):
            issues.append(f"{label}:missing_terminal_result")
        previous_state, previous_time, previous_task = initial["state"], initial["sim_time"], initial["task"]
        start_index = records[0].get("index") if records else None
        for index, row in enumerate(records):
            if not isinstance(row, dict) or any(key not in row for key in BOUNDARY_FIELDS):
                issues.append(f"{label}:missing_boundary_fields:{index}")
                continue
            decision = decisions[index] if index < len(decisions) else None
            links = {"index": "index", "local_index": "local_index", "source": "source",
                     "state": "state_before", "sim_time": "sim_time_before", "holding": "holding",
                     "task_status": "task_before", "planner_context_before": "planner_context_before",
                     "geometry_context_before": "geometry_context_before", "plan_identity": "plan_identity",
                     "rng_keys": "rng_keys"}
            if not isinstance(decision, dict) or any(key not in decision
                    or value_hash(decision[key]) != value_hash(row[target]) for key, target in links.items()):
                issues.append(f"{label}:decision_boundary_mismatch:{index}")
            _array(row["state_before"], 10); _array(row["state_after"], 10)
            _array(row["requested_action"], 2); _array(row["applied_action"], 2)
            if not _task_shape(row["task_before"]) or not _task_shape(row["task_after"]):
                issues.append(f"{label}:malformed_boundary_task:{index}")
            if (not np.isfinite(_array(row["requested_action"], 2)).all()
                    or not np.isfinite(_array(row["applied_action"], 2)).all()
                    or (row["observation_valid"] and not np.isfinite(_array(row["state_after"], 10)).all())):
                issues.append(f"{label}:nonfinite_valid_boundary_or_command:{index}")
            if (type(start_index) is not int or row["index"] != start_index+index
                    or row["local_index"] != index or not math.isfinite(row["sim_time_before"])
                    or not math.isfinite(row["sim_time_after"]) or not math.isfinite(row["actual_dt_s"])
                    or row["actual_dt_s"] < 0
                    or abs(row["sim_time_before"]-previous_time) > time_tolerance_s
                    or abs(row["sim_time_after"]-row["sim_time_before"]-row["actual_dt_s"]) > time_tolerance_s
                    or value_hash(row["state_before"]) != value_hash(previous_state)
                    or value_hash(row["task_before"]) != value_hash(previous_task)):
                issues.append(f"{label}:discontinuous_boundary:{index}")
            if previous_task.get("phase") in ("ready", "failed"):
                issues.append(f"{label}:interval_after_termination:{index}")
            if any(type(row[key]) is not bool for key in ("holding", "observation_valid", "transition_valid")):
                issues.append(f"{label}:invalid_validity_flags:{index}")
            if not isinstance(row["geometry"], dict) or type(row["geometry"].get("pass")) is not bool:
                issues.append(f"{label}:missing_geometry:{index}")
            if (row["source"] == "model" and (not isinstance(row["plan_identity"], dict) or any(
                    not isinstance(row["plan_identity"].get(key), str)
                    or re.fullmatch(r"[0-9a-f]{64}", row["plan_identity"][key]) is None
                    for key in HASH_FIELDS))) or (
                    row["source"] != "model" and row["plan_identity"] is not None):
                issues.append(f"{label}:missing_plan_hashes:{index}")
            if row["source"] not in ("model", "geometry", "holding"):
                issues.append(f"{label}:invalid_controller_source:{index}")
            if row["holding"] and (row["source"] != "holding" or
                    np.any(_array(row["applied_action"], 2) != 0.) or
                    np.any(_array(row["requested_action"], 2) != 0.)):
                issues.append(f"{label}:nonzero_or_inconsistent_holding:{index}")
            previous_state, previous_time, previous_task = row["state_after"], row["sim_time_after"], row["task_after"]
        # A nonexecuted all-invalid decision may fail at the last physical
        # boundary, so compare task clocks separately from its changed phase.
        if abs(final["last_sim_time"]-previous_time) > time_tolerance_s:
            issues.append(f"{label}:final_time_mismatch")
        if final != previous_task and witness.get("unexecuted_plan") is None:
            issues.append(f"{label}:final_task_without_boundary_or_unexecuted_plan")
        if witness.get("unexecuted_plan") is not None and (not decisions or
                value_hash(witness["unexecuted_plan"]) != value_hash(decisions[-1])):
            issues.append(f"{label}:unexecuted_plan_decision_mismatch")
        if not development_prefix and witness.get("termination") != final.get("terminal_reason"):
            issues.append(f"{label}:terminal_label_mismatch")
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        issues.append(f"{label}:malformed_boundary_or_task")
    return issues


def compare_witness(reference, fresh, task, profile, *, time_tolerance_s=1e-9,
                    verify_files=True):
    """Compare independent controller runs, retaining missing evidence as such.

    PASS_REPRODUCTION means matching declared boundaries for this runtime and
    source. It neither means alignment success nor validates every substep.
    """
    task.validate()
    fraction = profile["controller_replay"]["tolerance_fraction_of_task_tolerance"]
    if not (type(fraction) in (int, float) and math.isfinite(fraction) and fraction > 0
            and type(time_tolerance_s) in (int, float) and math.isfinite(time_tolerance_s)
            and time_tolerance_s > 0):
        raise ValueError("Positive finite replay/time tolerances are required")
    incomplete = _validate(reference, "reference", time_tolerance_s) + _validate(fresh, "fresh", time_tolerance_s)
    failures = []
    budgets = {"position_m": task.position_tolerance_m*fraction,
               "yaw_rad": task.yaw_tolerance_rad*fraction,
               "linear_speed_m_s": task.linear_speed_tolerance_m_s*fraction,
               "angular_speed_rad_s": task.angular_speed_tolerance_rad_s*fraction,
               "time_s": time_tolerance_s}
    maxima = {"state_components": [0.]*10, "requested_action_components": [0.]*2,
              "applied_action_components": [0.]*2, "requested_action_norm_m_s": 0.,
              "applied_action_norm_m_s": 0., "position_m": 0., "pusher_position_m": 0.,
              "yaw_rad": 0., "linear_speed_m_s": 0., "pusher_speed_m_s": 0.,
              "angular_speed_rad_s": 0., "time_s": 0.}
    same_nonfinite_pattern = True
    if isinstance(reference, dict) and isinstance(fresh, dict):
        if reference.get("execution_id") == fresh.get("execution_id") \
                or reference.get("attempt_id") == fresh.get("attempt_id"):
            incomplete.append("fresh:independent_execution_identity_required")
        if reference.get("record_kind") != "reference_controller_execution" \
                or fresh.get("record_kind") != "fresh_controller_replay":
            incomplete.append("fresh:reference_and_replay_roles_required")
        for key in ("identity", "source_bindings", "origin", "controller_id", "case_id", "task_spec", "run_scope"):
            if reference.get(key) != fresh.get(key):
                failures.append(f"identity_or_context_mismatch:{key}")
        if reference.get("task_spec") != json_safe(task):
            incomplete.append("comparison_task_differs_from_bound_task_spec")
        original_raw = reference.get("raw_reference")
        incomplete += _raw_issues(original_raw, reference.get("returned_results"),
                                  verify_files=verify_files, label="reference")
        if fresh.get("original_raw_reference") != original_raw:
            incomplete.append("fresh:original_raw_reference_mismatch")
        if not incomplete:
            for key in ("snapshot_hash", "planner_context", "geometry_context", "observation_valid", "schema_version"):
                if reference["initial"].get(key) != fresh["initial"].get(key):
                    failures.append(f"initial_context_mismatch:{key}")
            a_rows, b_rows = reference["intervals"], fresh["intervals"]
            if len(a_rows) != len(b_rows):
                failures.append("reproduced_interval_count_mismatch")

            def difference(a, b, *, state=False):
                nonlocal same_nonfinite_pattern
                a, b = np.asarray(a, float), np.asarray(b, float)
                same_nonfinite_pattern = same_nonfinite_pattern and bool(np.array_equal(
                    np.isfinite(a), np.isfinite(b)) and np.array_equal(np.isnan(a), np.isnan(b))
                    and np.array_equal(np.isposinf(a), np.isposinf(b))
                    and np.array_equal(np.isneginf(a), np.isneginf(b)))
                finite = np.isfinite(a) & np.isfinite(b)
                delta = np.zeros(a.shape, float)
                np.subtract(b, a, out=delta, where=finite)
                if state:
                    # Physical orientation reproduction uses 2*pi, unlike
                    # the task's pi-symmetric acceptance region.
                    delta[2] = math.atan2(math.sin(delta[2]), math.cos(delta[2]))
                return np.abs(delta)

            def state_difference(a, b):
                delta = difference(a, b, state=True)
                maxima["state_components"] = np.maximum(maxima["state_components"], delta).tolist()
                for axes, key in (((0, 1), "position_m"), ((6, 7), "pusher_position_m"),
                                  ((3, 4), "linear_speed_m_s"), ((8, 9), "pusher_speed_m_s")):
                    maxima[key] = max(maxima[key], float(np.linalg.norm(delta[list(axes)])))
                maxima["yaw_rad"] = max(maxima["yaw_rad"], float(delta[2]))
                maxima["angular_speed_rad_s"] = max(maxima["angular_speed_rad_s"], float(delta[5]))

            state_difference(reference["initial"]["state"], fresh["initial"]["state"])
            maxima["time_s"] = abs(reference["initial"]["sim_time"]-fresh["initial"]["sim_time"])
            for index, (a, b) in enumerate(zip(a_rows, b_rows)):
                state_difference(a["state_before"], b["state_before"])
                state_difference(a["state_after"], b["state_after"])
                for key in ("requested_action", "applied_action"):
                    delta = difference(a[key], b[key])
                    maxima[f"{key}_components"] = np.maximum(maxima[f"{key}_components"],
                        delta).tolist()
                    maxima[f"{key}_norm_m_s"] = max(maxima[f"{key}_norm_m_s"], float(np.linalg.norm(delta)))
                for key in ("sim_time_before", "sim_time_after", "actual_dt_s"):
                    maxima["time_s"] = max(maxima["time_s"], abs(a[key]-b[key]))
                for key in ("holding", "observation_valid", "transition_valid", "geometry",
                            "source", "rng_keys", "planner_context_before", "planner_context_after",
                            "geometry_context_before", "geometry_context_after", "plan_identity"):
                    if a[key] != b[key]:
                        failures.append(f"boundary_metadata_or_hash_mismatch:{index}:{key}")
                for key in ("task_before", "task_after"):
                    if not _task_match(a[key], b[key], time_tolerance_s):
                        failures.append(f"boundary_task_mismatch:{index}:{key}")
            if not _task_match(reference["final_task"], fresh["final_task"], time_tolerance_s):
                failures.append("final_task_mismatch")
            for key in ("termination", "counts", "decisions", "unexecuted_plan"):
                if reference[key] != fresh[key]:
                    failures.append(f"final_or_decision_mismatch:{key}")
            for key, budget_key in (("position_m", "position_m"), ("pusher_position_m", "position_m"),
                                    ("yaw_rad", "yaw_rad"), ("linear_speed_m_s", "linear_speed_m_s"),
                                    ("pusher_speed_m_s", "linear_speed_m_s"),
                                    ("angular_speed_rad_s", "angular_speed_rad_s"), ("time_s", "time_s")):
                if maxima[key] > budgets[budget_key]:
                    failures.append(f"numeric_budget_exceeded:{key}")
            for key in ("requested_action_components", "applied_action_components"):
                if maxima[key.removesuffix("_components")+"_norm_m_s"] > budgets["linear_speed_m_s"]:
                    failures.append(f"numeric_budget_exceeded:{key}")
            if not same_nonfinite_pattern:
                failures.append("nonfinite_boundary_pattern_mismatch")
    def has_physical_failure(witness):
        if not isinstance(witness, dict):
            return False
        if witness.get("termination") in ("physics_error", "geometry_invalid"):
            return True
        rows = witness.get("intervals")
        return isinstance(rows, list) and any(isinstance(row, dict) and (
            not row.get("transition_valid", False) or not row.get("observation_valid", False)
            or not isinstance(row.get("geometry"), dict)
            or not row["geometry"].get("pass", False)) for row in rows)

    physical_failure = has_physical_failure(reference) or has_physical_failure(fresh)
    preservation_required = bool(failures or incomplete or physical_failure)
    fresh_raw = fresh.get("raw_reference") if isinstance(fresh, dict) else None
    preservation_issues = []
    if preservation_required or fresh_raw is not None:
        preservation_issues = _raw_issues(fresh_raw, fresh.get("returned_results") if isinstance(fresh, dict) else None,
                                        verify_files=verify_files, label="fresh_prefix")
    incomplete += preservation_issues
    return {"schema": "p5-controller-reproduction-check-v1",
            "status": "INCOMPLETE" if incomplete else ("FAIL" if failures else "PASS_REPRODUCTION"),
            "comparison_matches": not incomplete and not failures,
            "incomplete_reasons": list(dict.fromkeys(incomplete)),
            "mismatch_reasons": list(dict.fromkeys(failures)),
            "numeric_budgets": budgets, "maximum_differences": maxima,
            "same_nonfinite_pattern": same_nonfinite_pattern,
            "strict_same_runtime_planner_hashes": True,
            "physical_failure_observed": physical_failure,
            "fresh_full_raw_prefix_required": preservation_required,
            "fresh_full_raw_prefix_present_and_verified": fresh_raw is not None and not preservation_issues,
            "task_success_claim": "not_assessed_by_boundary_reproduction",
            "scope": "fresh same-runtime controller boundaries only; raw substep/continuous-hold validation is separate"}
