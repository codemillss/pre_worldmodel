"""Read-only P3-05 raw-field audit and predeclared tick-17 model replay.

Run after the parent's official EXIT0 signal. No physics, fitting, model
selection or original write occurs. Model-only replay is not a timing probe.
All 35 declared cells remain; G's seven cells are N/A for model replay and
every one of the 28 model cells must contain compact tick 17 without replacing
it by a successful/full-trace/other-tick example.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import gc
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/p3_05_audit.json"
TICK = 17
USED = {}
INDEXES = {}
FAILURES = []


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def canonical_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.dtype.str.encode() + str(value.shape).encode()
                          + value.tobytes()).hexdigest()


def record(path, expected=None):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Audit inputs must remain in this project")
    actual = sha(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Declared hash differs: {path.relative_to(ROOT)}")
    for folder, index in INDEXES.items():
        if path.is_relative_to(folder) and path.name != "artifact_hashes.json":
            key = str(path.relative_to(folder))
            if index.get(key) != actual:
                raise ValueError(f"Immutable artifact index differs: {path.relative_to(ROOT)}")
    key = str(path.relative_to(ROOT))
    if key in USED and USED[key] != actual:
        raise ValueError(f"Input changed during audit: {key}")
    USED[key] = actual
    return path


def read(path, expected=None):
    return json.loads(record(path, expected).read_text())


def register_index(folder):
    folder = Path(folder).resolve()
    value = read(folder / "artifact_hashes.json")
    INDEXES[folder] = value["sha256"]
    return len(value["sha256"])


def npz(path):
    with np.load(record(path), allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("Duplicate NPZ member")
        return {key: archive[key].copy() for key in archive.files}


def checked(checks, name, passed, location):
    checks[name] = bool(passed)
    if not passed:
        FAILURES.append({"location": location, "check": name})
    return bool(passed)


def close(a, b, tolerance=1e-12):
    if a is None or b is None:
        return a is b
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    return math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=tolerance)


def encoded(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {key: encoded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encoded(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def restored_diagnostics(raw, arrays):
    result = raw["diagnostics"]
    paths, values = raw["diagnostic_nonfinite_paths"], arrays["diagnostic_nonfinite_values"]
    if (len(paths) != len(values) or np.isfinite(values).any()
            or len({json.dumps(p) for p in paths}) != len(paths)):
        raise ValueError("Invalid preserved nonfinite mapping")
    for path, value in zip(paths, values):
        if not isinstance(path, list) or len(path) < 2:
            raise ValueError("Invalid diagnostic nonfinite path")
        node = result
        for key in path[:-1]:
            node = node[key]
        if node[path[-1]] is not None:
            raise ValueError("Nonfinite mapping does not point to null")
        node[path[-1]] = float(value)
    return result


def metrics(state, config):
    state = np.asarray(state, np.float64)
    if state.shape != (10,) or not np.isfinite(state).all():
        return None
    t = config["task"]
    position = float(np.hypot(*(state[:2] - np.asarray(t["goal_xy_m"]))))
    angle = float(state[2] - t["goal_yaw_rad"])
    yaw = abs(.5 * math.atan2(math.sin(2 * angle), math.cos(2 * angle)))
    speed = float(np.hypot(state[3], state[4]))
    angular = abs(float(state[5]))
    pose = position <= t["position_tolerance_m"] and yaw <= t["yaw_tolerance_rad"]
    return {"e_pos_m": position, "e_yaw_rad": yaw, "linear_speed_m_s": speed,
            "angular_speed_rad_s": angular, "pose_within": pose,
            "qualifies": pose and speed <= t["linear_speed_tolerance_m_s"]
                         and angular <= t["angular_speed_tolerance_rad_s"]}


def geometry(samples, budget):
    nonfinite = reversed_side = deep = contacts = sides = 0
    maximum = 0.; all_finite = True
    for sample in samples:
        for contact in sample["contacts"]:
            contacts += 1
            distance, force = contact["distance_m"], contact["normal_force_N"]
            normal = np.asarray(contact["normal_world"], np.float64)
            if not (math.isfinite(distance) and math.isfinite(force) and np.isfinite(normal).all()):
                nonfinite += 1
            if math.isfinite(distance):
                maximum = max(maximum, max(0., -distance))
            else:
                all_finite = False
            deep += int(-distance > budget)
            if contact["pair"] == "pusher_object":
                sides += 1
                cosine = contact["normal_center_cosine"]
                reversed_side += int(cosine is None or not math.isfinite(cosine) or cosine <= 0.)
    reasons = (["nonfinite_contact_geometry"] if nonfinite else [])
    reasons += ["reversed_side_contact_normal"] if reversed_side else []
    reasons += ["contact_penetration_exceeded"] if deep else []
    return {"pass": not reasons, "sample_count": len(samples), "contact_count": contacts,
            "side_contact_count": sides, "nonfinite_contacts": nonfinite,
            "reversed_side_contact_normals": reversed_side,
            "over_budget_penetration_contacts": deep,
            "max_penetration_m": maximum if all_finite else None,
            "max_finite_penetration_m": maximum, "penetration_budget_m": budget,
            "reasons": reasons}


def task_start(state, time, config):
    m = metrics(state, config)
    return {"phase": "holding" if m["pose_within"] else "running",
            "initial_sim_time": time, "last_sim_time": time, "held_time_s": 0.,
            "previous_qualifies": m["qualifies"], "terminal_reason": None, "violations": []}


def task_after(before, diagnostic, actual_dt, config, geom):
    t = config["task"]; samples = diagnostic["samples"]
    reasons = list(diagnostic["validity_violations"])
    held, previous, last = before["held_time_s"], before["previous_qualifies"], before["last_sim_time"]
    numerical = False
    for sample in samples[1:]:
        m = metrics(sample["state"], config)
        qualified = m is not None and m["qualifies"]
        if m is None:
            numerical = True
        held = held + float(sample["sim_time"] - last) if qualified and previous else 0.
        previous, last = qualified, float(sample["sim_time"])
    if numerical and "nonfinite_state" not in reasons:
        reasons.append("nonfinite_state")
    if abs(actual_dt - config["physics"]["action_dt_s"]) > 1e-9 or len(samples) < 2:
        if "partial_transition" not in reasons:
            reasons.append("partial_transition")
    end = float(samples[-1]["sim_time"])
    after = {"phase": "running", "initial_sim_time": before["initial_sim_time"],
             "last_sim_time": end, "held_time_s": held, "previous_qualifies": previous,
             "terminal_reason": None, "violations": reasons}
    if reasons:
        reason_set = set(reasons)
        if numerical or reason_set & {"nonfinite_state", "physics_warning", "physics_time_discontinuity", "observation_invalid", "task_metrics_nonfinite"}:
            failure = "physics_error"
        elif reason_set & {"object_corners_outside_workspace", "pusher_extent_outside_workspace"}:
            failure = "workspace_exit"
        elif reason_set & {"height_deviation", "tilt_exceeded", "vertical_speed_exceeded", "horizontal_angular_speed_exceeded"}:
            failure = "planarity_violation"
        else:
            failure = "physics_error"
        after.update(phase="failed", terminal_reason=failure)
    elif held + 1e-9 >= t["hold_time_s"]:
        after.update(phase="ready", terminal_reason="ready")
    elif end - before["initial_sim_time"] + 1e-9 >= t["max_episode_time_s"]:
        after.update(phase="failed", terminal_reason="timeout")
    else:
        m = metrics(samples[-1]["state"], config)
        after["phase"] = "holding" if m["pose_within"] else "running"
    if not geom["pass"] and after["terminal_reason"] != "physics_error":
        after.update(phase="failed", terminal_reason="geometry_invalid",
                     violations=list(dict.fromkeys([*reasons, *geom["reasons"]])))
    return after


def status_matches(actual, expected):
    return (set(actual) == set(expected) and all(
        close(actual[key], value) if key.endswith("_time") or key == "held_time_s"
        else actual[key] == value for key, value in expected.items()))


def cell_audit(folder, label, case, config, root_manifest, resource_cell=False):
    summary = read(folder / "summary.json"); manifest = read(folder / "manifest.json")
    records = ([json.loads(line) for line in record(folder / "steps.jsonl").read_text().splitlines() if line]
               if (folder / "steps.jsonl").exists() else [])
    arrays = npz(folder / "raw_execution.npz")
    with gzip.open(record(folder / "raw_execution.json.gz"), "rt", encoding="utf-8") as stream:
        raw = json.load(stream)
    diagnostics = restored_diagnostics(raw, arrays)
    checks = {}; count = len(records)
    checked(checks, "cell_case_identity_and_physical_config_bound", manifest["case"] == case
            and summary["cell_id"] == manifest["model_id"] + "/" + case["episode_id"]
            and summary["case_id"] == case["episode_id"]
            and manifest["physical_contract"]["resolved_config"] == config, label)
    required = {"state", "actual_dt", "action_requested", "action_applied", "substep_states",
                "substep_times", "initial_integration_state", "diagnostic_nonfinite_values"}
    checked(checks, "exact_raw_array_population", set(arrays) == required, label)
    checked(checks, "all_raw_arrays_float64", all(a.dtype == np.float64 for a in arrays.values()), label)
    checked(checks, "raw_records_and_diagnostics_counts", count == len(diagnostics) == len(arrays["state"])
            == len(arrays["actual_dt"]) == len(arrays["action_requested"]) == len(arrays["action_applied"])
            == summary["transitions"], label)
    checked(checks, "recorded_tick_order", [r["index"] for r in records] == list(range(count)), label)
    snapshot = root_manifest["snapshots"][case["episode_id"]]
    checked(checks, "gzip_initial_snapshot_matches_frozen_manifest", raw["initial_snapshot"] == snapshot
            and canonical_hash(snapshot) == manifest["initial_snapshot_hash"], label)
    checked(checks, "npz_initial_integration_matches_manifest", np.array_equal(
        arrays["initial_integration_state"], np.asarray(snapshot["integration_state"], np.float64), equal_nan=True), label)
    requests = np.asarray([r["requested_action"] for r in records], np.float64).reshape(-1, 2)
    applied = np.asarray([r["applied_action"] for r in records], np.float64).reshape(-1, 2)
    actual_dt = np.asarray([r["actual_dt_s"] for r in records], np.float64)
    checked(checks, "all_requested_commands_exact", np.array_equal(arrays["action_requested"], requests, equal_nan=True), label)
    checked(checks, "all_applied_commands_exact", np.array_equal(arrays["action_applied"], applied, equal_nan=True), label)
    checked(checks, "all_actual_dt_exact", np.array_equal(arrays["actual_dt"], actual_dt, equal_nan=True), label)
    checked(checks, "all_raw_states_match_step_JSON", encoded(arrays["state"]) == [r["state_after"] for r in records], label)
    substates, subtimes = [], []; geometry_match = task_match = continuity = validity = True
    identity_match = True; previous_applied = np.zeros(2, np.float64); previous_warm = None
    previous_gctx = {"mode": "RETRACT", "locked_offset_m": None, "retreat_offset_m": None,
                     "phase_started_s": 0., "last_elapsed_s": 0.,
                     "reassessment_count": 0, "saturation_count": 0}
    prior_state = np.asarray(manifest["initial_state"], np.float64)
    initial_time = float(diagnostics[0]["samples"][0]["sim_time"]) if count else summary["task"]["initial_sim_time"]
    status = task_start(prior_state, initial_time, config)
    valid_count = active_contact_intervals = 0
    for index, (row, diag) in enumerate(zip(records, diagnostics)):
        if row["plan"] is not None:
            p = row["plan"]; prior = p["planner_context_before"]
            identity_match &= (p["tick"] == index and p["state"] == row["state_before"]
                and status_matches(p["task_status"], status)
                and prior["previous_applied_action"] == previous_applied.tolist()
                and prior["optimal_actions"] == previous_warm and prior["control_step"] == index
                and p["geometry_context_before"] == previous_gctx
                and p["geometry_context_after"] == row["G_context_committed"]
                and p["geometry_proposal"] == row["G_proposal"]
                and p["rng_keys"] == [root_manifest["settings"]["candidate_seed_base"], case["episode_index"],
                                     index, root_manifest["settings"]["candidate_stream_id"]]
                and p["source_manifest_sha256"] == sha(folder / "manifest.json")
                and p["planner_config_hash"] == canonical_hash(manifest["planner_config"]))
            previous_warm = p["selected_actions"]
            identity_match &= p["selected_actions"] is not None and p["selected_actions"][0] == row["requested_action"]
        elif row["source"] == "holding":
            previous_warm = None
        samples = diag["samples"]
        states = np.asarray([s["state"] for s in samples], np.float64)
        times = np.asarray([s["sim_time"] for s in samples], np.float64)
        substates.append(states[1:]); subtimes.append(times[1:])
        continuity &= (np.array_equal(prior_state, states[0], equal_nan=True)
                       and encoded(states[0]) == row["state_before"]
                       and np.array_equal(states[-1], arrays["state"][index], equal_nan=True)
                       and abs(times[0] - status["last_sim_time"]) <= 1e-9
                       and bool((np.diff(times) > 0).all())
                       and arrays["actual_dt"][index] == float(times[-1] - times[0]))
        geom = geometry(samples, row["geometry"]["penetration_budget_m"])
        geometry_match &= all(close(row["geometry"][k], v) if isinstance(v, (float, int)) and not isinstance(v, bool)
                              else row["geometry"][k] == v for k, v in geom.items())
        physical_valid = not diag["validity_violations"] and diag["terminal_type"] == "completed_interval"
        validity &= physical_valid == row["physical_valid"]
        valid_count += int(physical_valid)
        active_contact_intervals += int(any(any(c["pair"] == "pusher_object"
            and c["normal_force_N"] > config["evaluation"]["contact_active_normal_force_min_N"]
            for c in s["contacts"]) for s in samples))
        status = task_after(status, diag, actual_dt[index], config, geom)
        task_match &= status_matches(row["task"], status)
        prior_state = states[-1]
        previous_applied = applied[index]
        previous_gctx = row["G_context_committed"]
    substates = np.concatenate(substates) if substates else np.empty((0, 10), np.float64)
    subtimes = np.concatenate(subtimes) if subtimes else np.empty(0, np.float64)
    checked(checks, "all_substep_states_exact_from_raw_diagnostics", np.array_equal(arrays["substep_states"], substates, equal_nan=True), label)
    checked(checks, "all_substep_times_exact_from_raw_diagnostics", np.array_equal(arrays["substep_times"], subtimes, equal_nan=True), label)
    checked(checks, "all_boundaries_and_actual_elapsed_match", continuity, label)
    checked(checks, "all_geometry_counts_and_reasons_independent", geometry_match, label)
    checked(checks, "every_task_hold_and_terminal_status_independent", task_match, label)
    checked(checks, "physical_validity_flags_and_counts_match", validity and valid_count == summary["valid_transitions"], label)
    checked(checks, "all_compact_identities_bound_to_actual_pre_state_task_G_warm_and_tick", identity_match, label)
    checked(checks, "holding_count_and_zero_commands_match", sum(r["source"] == "holding" for r in records)
            == summary["holding_intervals"] and all(np.array_equal(applied[i], np.zeros(2))
                 for i, r in enumerate(records) if r["source"] == "holding"), label)
    # Unexecuted final decisions keep the same last physical boundary.
    for file, reason in (("unexecuted_final_plan.json", "planner_invalid"),
                         ("unexecuted_final_G.json", "controller_infeasible")):
        if (folder / file).exists():
            read(folder / file)
            status.update(phase="failed", terminal_reason=reason)
    checked(checks, "final_task_matches_all_actual_samples", status_matches(summary["task"], status), label)
    termination = status["terminal_reason"] or ("unfinished_budget" if resource_cell else "development_budget")
    checked(checks, "termination_budget_and_denominator_preserved", termination == summary["termination"], label)
    final = metrics(prior_state, config)
    checked(checks, "terminal_metrics_independent", final is not None and all(
        close(summary["final_metrics"][k], v) for k, v in final.items()), label)
    loss = None if final is None else (final["e_pos_m"] / config["task"]["position_tolerance_m"]) ** 2 \
        + (final["e_yaw_rad"] / config["task"]["yaw_tolerance_rad"]) ** 2
    checked(checks, "terminal_normalized_pose_loss_independent", close(summary["terminal_pose_loss"], loss), label)
    compact = next((r["plan"] for r in records if r["index"] == TICK), None)
    output = {"cell": label, "case_id": case["episode_id"], "model_id": manifest["model_id"],
              "passed": all(checks.values()), "checks": checks, "transitions": count,
              "substep_count": len(subtimes), "valid_transitions": valid_count,
              "active_contact_intervals": active_contact_intervals,
              "termination": termination, "terminal_pose_loss": loss,
              "primary_execution_status": summary["execution_status"]}
    return output, compact, manifest, summary


def replay(compact, cell_manifest, bundle, normalizer, config, label):
    from push_align.closed_loop import ENGINE_GUARDS, model_planning_scope
    from push_align.geometry_control import GeometryContext
    from push_align.planning import LearnedPredictor, PlannerContext, TaskSpec, plan
    from push_align.planning_protocol import GeometryProposalAdapter, keyed_planner_rng
    checks = {}
    if compact is None:
        checked(checks, "predeclared_tick_17_present_no_substitution", False, label)
        return {"cell": label, "passed": False, "checks": checks, "tick": TICK, "substitution": False}
    checked(checks, "predeclared_tick_17_present_no_substitution", compact["tick"] == TICK, label)
    checked(checks, "tick_17_is_compact_without_saved_prediction_arrays", compact["full_trace"] is False, label)
    before = deepcopy(compact); task_data = deepcopy(compact["task_spec"])
    task_data["goal"] = np.asarray(task_data["goal"], np.float64)
    task = TaskSpec(**task_data); state = np.asarray(compact["state"], np.float64)
    prior = compact["planner_context_before"]
    context = PlannerContext(np.asarray(prior["previous_applied_action"], np.float64),
        None if prior["optimal_actions"] is None else np.asarray(prior["optimal_actions"], np.float64), prior["control_step"])
    gctx = GeometryContext(**compact["geometry_context_before"])
    parameters = cell_manifest["effective_geometry_parameters"]
    selected = {k: parameters[k] for k in ("position_gain_per_s", "yaw_creep_speed_m_s", "yaw_offset_gain_m_rad", "speed_max_m_s")}
    adapter = GeometryProposalAdapter(config, selected, cell_manifest["transition_parameters"])
    elapsed = compact["task_status"]["last_sim_time"] - compact["task_status"]["initial_sim_time"]
    rng = keyed_planner_rng(*compact["rng_keys"]); guards = dict.fromkeys(ENGINE_GUARDS, 0)
    checked(checks, "keyed_stream_matches_saved_PC64_state", rng.bit_generator.state == compact["rng_state_before"], label)
    checked(checks, "planner_config_hash_and_effective_G_values_match",
            canonical_hash(cell_manifest["planner_config"]) == compact["planner_config_hash"]
            and adapter.parameters == parameters, label)
    checked(checks, "saved_model_identity_matches_loaded_artifact",
            compact["model_hash"] == bundle.metadata.get("checkpoint_hash", bundle.metadata.get("ridge_hash")), label)
    predictor = LearnedPredictor(bundle.model, normalizer)
    with model_planning_scope(guards):
        proposal = adapter.propose(state.copy(), task, gctx, elapsed, control_tick=TICK)
        result = plan(state.copy(), task, context, rng, predictor, cell_manifest["planner_config"],
                      geometry_action=proposal.geometry_action, tangent_direction=proposal.tangent_direction)
    checked(checks, "G_proposal_from_actual_state_history_elapsed_task_exact", encoded(proposal.diagnostics) == compact["geometry_proposal"]
            and asdict(adapter.commit(proposal, actual_control_tick=TICK)) == compact["geometry_context_after"], label)
    for name, value in (("candidate_hash", result.candidates.actions),
                        ("prediction_hash", result.predictions.states), ("delta_hash", result.predictions.raw_deltas)):
        checked(checks, "exact_" + name, array_hash(value) == compact[name], label)
    i = result.selected_index
    checked(checks, "selected_index_reason_and_full_sequence_exact", i == compact["selected_index"]
            and result.reason == compact["reason"]
            and (None if i is None else result.candidates.actions[i].tolist()) == compact["selected_actions"], label)
    checked(checks, "selected_cost_and_each_component_exact", (None if i is None else float(result.costs.total[i])) == compact["selected_cost"]
            and (None if i is None else {k: float(v[i]) for k, v in result.costs.components.items()}) == compact["selected_cost_components"], label)
    checked(checks, "validity_eligibility_reasons_and_generator_diagnostics_exact", int(result.predictions.valid.sum()) == compact["valid_predictions"]
            and int((result.predictions.valid & np.isfinite(result.costs.total)).sum()) == compact["eligible_costs"]
            and dict(Counter(r for r in result.predictions.reasons if r)) == compact["invalid_reasons"]
            and result.candidates.diagnostics == compact["candidate_generation"], label)
    checked(checks, "future_engine_guards_all_zero", all(v == 0 for v in guards.values()), label)
    checked(checks, "compact_original_input_unchanged", compact == before, label)
    return {"cell": label, "passed": all(checks.values()), "checks": checks, "tick": TICK,
            "full_trace_used_as_input": False, "physical_execution": False,
            "future_engine_calls": guards, "predictor_adapter_calls": predictor.calls,
            "previous_applied_action_nonzero": bool(np.any(context.previous_applied_action != 0)),
            "nonempty_warm_sequence": context.optimal_actions is not None,
            "G_mode_before": gctx.mode, "raw_elapsed_s": elapsed}


def score(rows):
    losses = [r["terminal_pose_loss"] for r in rows]
    eligible = all(r["primary_execution_status"] == "completed" and r["termination"] not in
        {"physics_error", "geometry_invalid", "infrastructure_error", "controller_infeasible"} for r in rows)
    eligible &= all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in losses)
    return {"eligible": bool(eligible), "ready": sum(r["termination"] == "ready" for r in rows),
            "task_failures": sum(r["termination"] not in {"ready", "development_budget", "timeout"} for r in rows),
            "mean_terminal_pose_loss": float(np.mean(losses)) if eligible else None,
            "attempted": len(rows), "expected": len(rows)}


def safe_replay(compact, cell_manifest, bundle, normalizer, config, label):
    try:
        return replay(compact, cell_manifest, bundle, normalizer, config, label)
    except Exception as error:
        FAILURES.append({"location": label, "replay_error": type(error).__name__ + ": " + str(error)})
        return {"cell": label, "passed": False, "tick": TICK, "substitution": False}


def audit(run_dir):
    sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
    import torch
    from push_align.normalization import load_normalizer
    from push_align.ridge import load_ridge
    from p3_train_common_mlp import load_common_checkpoint
    USED.clear(); INDEXES.clear(); FAILURES.clear()
    run_dir = Path(run_dir).resolve()
    if not (run_dir / "summary.json").exists() or not (run_dir / "artifact_hashes.json").exists():
        raise ValueError("Wait for completed official EXIT0 summary and artifact index")
    register_index(run_dir)
    manifest = read(run_dir / "manifest.json"); primary = read(run_dir / "summary.json")
    if not primary["execution_checks_pass"]:
        raise ValueError("Official execution checks did not pass; preserve and investigate that run")
    settings = manifest["settings"]; config = manifest["physical_config"]
    root_checks = {}
    checked(root_checks, "settings_fingerprint_bound", canonical_hash(settings) == manifest["settings_sha256"], "run")
    for relative, expected in manifest["source_hashes"].items():
        record(run_dir / "source" / relative, expected)
        if relative.startswith(("src/", "scripts/")):
            record(ROOT / relative, expected)
    torch.set_num_threads(settings["torch_threads"]); torch.use_deterministic_algorithms(True)
    global_rng = torch.get_rng_state().clone()
    learning = ROOT / "runs/p3-02-20261009-01/learning_data"
    ridge = ROOT / "runs/p3-03-20261009-01"; common = ROOT / "runs/p3-04-20261009-01"
    for folder in (learning.parent, ridge, common):
        register_index(folder)
    nm = read(learning / "normalizer/normalizer.json")
    record(learning / "normalizer" / nm["npz"], nm["npz_sha256"])
    normalizer = load_normalizer(learning / "normalizer", nm["normalizer"]["schema"],
                                nm["normalizer"]["dataset_identity"], manifest["normalizer_hash"])
    models = {}; rs = read(ridge / "selection.json"); rm = read(ridge / "manifest.json")
    ridge_path = ridge / rs["selected"]["model_path"]
    rmeta = read(ridge_path / "ridge.json"); record(ridge_path / "weights.npz", rmeta["files"]["weights.npz"])
    models["R"] = load_ridge(ridge_path, normalizer, config, rm["provenance"])
    main = read(common / "summary.json"); cm = read(common / "manifest.json")
    for row in main["seeds"]:
        path = common / row["selected_checkpoint"]
        meta = read(path / "checkpoint.json"); record(path / "weights.pt", meta["files"]["weights.pt"])
        models[f'MLP-{row["seed"]}'] = load_common_checkpoint(path, normalizer, config,
            cm["provenance"], cm["settings"], row["seed"], epoch=row["selected_epoch"])
    before_weights = {name: {key: value.clone() for key, value in bundle.model.state_dict().items()}
                      for name, bundle in models.items()}
    checked(root_checks, "all_four_loaded_models_match_run_frozen_identities", {k: v.metadata.get("checkpoint_hash", v.metadata.get("ridge_hash"))
            for k, v in models.items()} == manifest["model_hashes"], "run")
    cases = manifest["cases"]; cells = []; replays = []; trials = {"geometry": [], "planning": []}
    source_selections = {"geometry": read(run_dir / "geometry_selection.json"),
                         "planning": read(run_dir / "planning_selection.json")}
    for kind, declared in (("geometry", settings["geometry_trials"]), ("planning", settings["planning_trials"])):
        for trial in declared:
            trial_folder = run_dir / (kind + "_trials") / trial["id"]
            saved_trial = read(trial_folder / "summary.json"); rows = []
            for model in (["G"] if kind == "geometry" else list(models)):
                for case in cases:
                    folder = trial_folder / case["episode_id"] if model == "G" else trial_folder / model / case["episode_id"]
                    label = str(folder.relative_to(run_dir))
                    try:
                        measured, compact, cell_manifest, saved = cell_audit(folder, label, case, config, manifest)
                        rows.append(measured); cells.append(measured)
                        if model != "G":
                            replays.append(safe_replay(compact, cell_manifest, models[model], normalizer, config, label))
                    except Exception as error:
                        FAILURES.append({"location": label, "error": type(error).__name__ + ": " + str(error)})
                        failed = {"cell": label, "model_id": model, "case_id": case["episode_id"],
                                  "passed": False, "terminal_pose_loss": None,
                                  "termination": "audit_read_or_replay_error", "primary_execution_status": "audit_error"}
                        cells.append(failed); rows.append(failed)
                        if model != "G":
                            replays.append({"cell": label, "passed": False, "tick": TICK, "substitution": False})
            calculated = score(rows)
            matches = all(close(saved_trial["score"][key], value) for key, value in calculated.items())
            expected_ids = [r["model_id"] + "/" + r["case_id"] for r in rows]
            checked(root_checks, f"{kind}_{trial['id']}_declared_cells_and_order_preserved",
                    [r["cell_id"] for r in saved_trial["rows"]] == expected_ids, "selection")
            checked(root_checks, f"{kind}_{trial['id']}_all_cell_score_independent", matches, "selection")
            trials[kind].append({"id": trial["id"], "score": calculated,
                                 "cell_ids": [r["model_id"] + "/" + r["case_id"] for r in rows]})
        eligible = [t for t in trials[kind] if t["score"]["eligible"]]
        selected = min(eligible, key=lambda t: (-t["score"]["ready"], t["score"]["task_failures"],
                                                t["score"]["mean_terminal_pose_loss"])) if eligible else None
        checked(root_checks, kind + "_selection_exact_argmin_declared_order", selected is not None
                and selected["id"] == source_selections[kind]["id"], "selection")
    case = next(c for c in cases if c["episode_index"] == settings["full_resource_case_index"])
    for model in ["G", *models]:
        folder = run_dir / "full_resource" / model; label = str(folder.relative_to(run_dir))
        try:
            measured, compact, cell_manifest, saved = cell_audit(folder, label, case, config, manifest, resource_cell=True)
            cells.append(measured)
            if model != "G":
                replays.append(safe_replay(compact, cell_manifest, models[model], normalizer, config, label))
        except Exception as error:
            FAILURES.append({"location": label, "error": type(error).__name__ + ": " + str(error)})
            cells.append({"cell": label, "passed": False, "model_id": model, "case_id": case["episode_id"]})
            if model != "G":
                replays.append({"cell": label, "passed": False, "tick": TICK, "substitution": False})
    checked(root_checks, "all_35_cells_retained_no_survivor_filter", len(cells) == 35, "run")
    checked(root_checks, "all_28_predeclared_middle_replays_pass", len(replays) == 28 and all(r["passed"] for r in replays), "run")
    checked(root_checks, "all_35_raw_and_task_geometry_audits_pass", all(r["passed"] for r in cells), "run")
    checked(root_checks, "all_loaded_model_tensors_unchanged", all(torch.equal(value, models[name].model.state_dict()[key])
            for name, tree in before_weights.items() for key, value in tree.items()), "run")
    checked(root_checks, "global_Torch_RNG_unchanged_after_load_and_model_replay", torch.equal(global_rng, torch.get_rng_state()), "run")
    checked(root_checks, "all_used_inputs_unchanged_after_audit", all(sha(ROOT / p) == digest for p, digest in USED.items()), "run")
    report = {"schema": "p3-planning-independent-audit-v1", "run": str(run_dir.relative_to(ROOT)),
              "created_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
              "passed": all(root_checks.values()) and not FAILURES, "checks": root_checks,
              "raw_cells": cells, "middle_replays": replays, "independent_trial_scores": trials,
              "denominators": {"raw_cells": 35, "model_middle_replay_cells": 28, "G_model_replay_NA": 7,
                               "tick": TICK, "substitution_allowed": False},
              "middle_replay_coverage": {
                  "nonzero_previous_applied_action": sum(r.get("previous_applied_action_nonzero", False) for r in replays),
                  "nonempty_warm_sequence": sum(r.get("nonempty_warm_sequence", False) for r in replays),
                  "G_mode_before_counts": dict(Counter(r.get("G_mode_before", "AUDIT_MISSING") for r in replays)),
                  "scope": "All 28 prescribed tick17 inputs have RETRACT G history; other actual G phases are not claimed by this replay."},
              "failures": FAILURES, "analysis_code_sha256": sha(__file__),
              "used_input_sha256": dict(sorted(USED.items())),
              "original_index_verification": {str(p.relative_to(ROOT)): len(v) for p, v in INDEXES.items()},
              "limitations": [
                  "Exact comparisons audit preserved archive contents and duplicated independent data routes; original live objects are no longer present.",
                  "Physical validity counts are cross-checked against recorded violation/terminal flags; this is not a new full threshold or timestep physics acceptance suite.",
                  "Geometry and continuous-hold/task metrics are independently recalculated from all preserved samples.",
                  "Tick17 replay uses actual observed inputs, task/G/planner history, raw elapsed, keyed RNG and frozen model/config; no full prediction arrays or future physics are used.",
                  "The primary 28 first-plan checks concern mostly fresh initial warm/G contexts. The additional 28 tick17 checks broaden that scope but do not prove every tick was replayed.",
                  "All tick17 G histories are RETRACT. Nonzero previous actions and warm sequences are covered; actual middle-plan replay across SLIDE/APPROACH/PUSH is not established here.",
                  "No training updates, physics simulation, final-test tuning, model reselection or new time/memory-performance measurement occurs.",
                  "Previously used development cases, overlapping cells and the resource case do not form independent generalization or model-superiority evidence.",
              ]}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "raw_cells": len(cells), "middle_replays": len(replays),
                      "replay_pass": sum(r["passed"] for r in replays), "failures": FAILURES,
                      "used_inputs": len(USED), "report_sha256": sha(OUTPUT)}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/p3-05-20261010-01")
    args = parser.parse_args()
    audit(args.run_dir)
