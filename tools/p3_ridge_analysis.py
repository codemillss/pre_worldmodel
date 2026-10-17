"""Independent, read-only derived audit of the frozen P3-03 Ridge run.

Use --analyze --selection-only while development is running, then --analyze
after its final summary exists. --plot consumes only the derived report and
needs Matplotlib; numerical analysis uses the project NumPy runtime. No model
is fitted and no simulator is invoked. Original raw-main validity is the
upstream gate's evidence, not repeated or newly claimed by this analysis.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/p3_03_analysis.json"
FIGURES = ROOT / "reports/figures"
HORIZONS = (1, 5, 10)
REGIMES = ("noncontact", "sustained_contact", "transition")
STATIONARY_XY_M = STATIONARY_YAW_RAD = 1e-9
ENGINE_GUARDS = {"mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse"}
USED = {}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def record(path, expected=None):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Analysis input must remain in this project")
    actual = sha(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Input hash mismatch: {path.relative_to(ROOT)}")
    relative = str(path.relative_to(ROOT))
    if relative in USED and USED[relative] != actual:
        raise ValueError(f"Input changed during analysis: {relative}")
    USED[relative] = actual
    return path


def read(path, expected=None):
    return json.loads(record(path, expected).read_text(encoding="utf-8"))


def arrays(path, expected=None):
    with np.load(record(path, expected), allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("Duplicate NPZ member")
        return {name: archive[name].copy() for name in archive.files}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.dtype.str.encode() + str(value.shape).encode()
                          + value.tobytes()).hexdigest()


def close(actual, expected, name, atol=1e-12):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=atol):
        raise ValueError(f"Independent measurement differs: {name}")


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"n": 0, "mean": None, "rmse": None, "p95": None, "max": None}
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite value in a descriptive metric")
    return {"n": len(values), "mean": float(values.mean()),
            "rmse": float(np.sqrt(np.mean(values ** 2))),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def physical_metrics(errors):
    return {"object_position_error_mm": describe(np.linalg.norm(errors[:, :2], axis=1) * 1000),
            "object_yaw_error_deg": describe(np.rad2deg(np.abs(errors[:, 2])))}


def wrapped(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def stationary(current, future):
    return ((np.linalg.norm(future[:, :2] - current[:, :2], axis=1) < STATIONARY_XY_M)
            & (np.abs(wrapped(future[:, 2] - current[:, 2])) < STATIONARY_YAW_RAD))


def source_audit(folder, manifest):
    for relative, expected in manifest["source_hashes"].items():
        record(folder / "source" / relative, expected)
    return {"archived_source_files": len(manifest["source_hashes"]),
            "all_archived_source_hashes_match": True,
            "scope": "run snapshot verified; current project documentation may later advance"}


def load_main(manifest):
    learning = ROOT / "runs/p3-02-20261009-01/learning_data"
    lm = read(learning / "manifest.json", manifest["provenance"]["learning_manifest_sha256"])
    nm = read(learning / "normalizer/normalizer.json", lm["normalizer_manifest_sha256"])
    norm = nm["normalizer"]
    stats = arrays(learning / "normalizer" / nm["npz"], nm["npz_sha256"])
    prepared = arrays(learning / lm["npz"], lm["npz_sha256"])
    upstream = read(learning.parent / "summary.json")
    if upstream["stage_verdict"] != "PASS" or not all(upstream["checks"].values()):
        raise ValueError("Main upstream data gate has not passed")
    if (norm["normalizer_hash"] != manifest["normalizer_hash"]
            or norm["stats_hash"] != manifest["stats_hash"]
            or norm["dataset_identity"] != manifest["provenance"]["dataset_identity"]
            or norm["fit_split"] != "train"):
        raise ValueError("Run normalization identity differs from original stored statistics")
    train = np.flatnonzero(prepared["split_code"] == 0)
    validation = np.flatnonzero(prepared["split_code"] == 1)
    if (len(train) != 3200 or len(validation) != 1600
            or train.tolist() != manifest["train_rows"]
            or validation.tolist() != manifest["validation_rows"]
            or norm["fit_count"] != len(train)):
        raise ValueError("Train/validation mask or original fit count differs")
    x = ((prepared["features"][train] - stats["input_mean"])
         / stats["input_scale"]).astype(np.float32).astype(np.float64)
    y = ((prepared["deltas"][train] - stats["delta_mean"])
         / stats["delta_scale"]).astype(np.float32).astype(np.float64)
    if (not np.array_equal(x.astype(np.float32), prepared["normalized_features"][train])
            or not np.array_equal(y.astype(np.float32), prepared["normalized_deltas"][train])):
        raise ValueError("Cached normalized train rows differ from stored-statistics boundary")
    train_hashes = {"normalized_features_sha256": hashlib.sha256(x.astype("<f8").tobytes()).hexdigest(),
                    "normalized_targets_sha256": hashlib.sha256(y.astype("<f8").tobytes()).hexdigest()}
    identities = {entry["episode_index"]: entry for entry in lm["preparation"]["episodes"]}
    for key in ("family_id", "episode_id"):
        prepared["_" + key] = np.asarray([identities[int(index)][key]
                                          for index in prepared["episode_index"]], dtype=np.str_)
    still = stationary(prepared["current_state"], prepared["next_state"])
    pusher_moves = (np.linalg.norm(prepared["next_state"][:, 6:8]
                                  - prepared["current_state"][:, 6:8], axis=1) >= 1e-9)
    station_counts = {}
    for split, rows in (("train", train), ("validation", validation)):
        station_counts[split] = {
            "original_valid_rows": len(rows), "object_pose_stationary": int(still[rows].sum()),
            "object_stationary_pusher_position_changes": int((still[rows] & pusher_moves[rows]).sum()),
            "object_and_pusher_positions_stationary": int((still[rows] & ~pusher_moves[rows]).sum()),
            "by_regime": {name: int((still[rows] & (prepared["regime_code"][rows] == code)).sum())
                          for code, name in enumerate(REGIMES)},
        }
    return prepared, lm, norm, x, y, train_hashes, {
        "upstream_gate": upstream["stage_verdict"], "upstream_checks": upstream["checks"],
        "train_rows": len(train), "validation_rows": len(validation),
        "normalizer_hash": norm["normalizer_hash"], "statistics_hash": norm["stats_hash"],
        "physical_contract_hash": norm["schema"]["data_contract_hash"],
        "critical_learning_and_statistics_artifact_hashes_match": True,
        "fit_mask_matches_actual_prepared_splits": True,
        "stationary_row_diagnostic": station_counts,
        "scope": "prepared NPZ and stored statistics verified; full raw-main physics/mask audit belongs to upstream 27 checks, not repeated here",
    }


def window_analysis(folder, prepared, expected_hashes, *, state_hold=False):
    summary = read(folder / "summary.json")
    trace = arrays(folder / "traces.npz")
    result, scores = {}, {}
    still = stationary(prepared["current_state"], prepared["next_state"])
    for k in HORIZONS:
        prefix = f"h{k}_"
        windows = prepared[f"windows_{k}"]
        windows = windows[prepared["split_code"][windows[:, 0]] == 1]
        if (not np.array_equal(trace[prefix + "window_rows"], windows)
                or array_hash(windows) != expected_hashes[str(k)]
                or not np.all(prepared["split_code"][windows] == 1)):
            raise ValueError("Candidate changed validation windows, order or split")
        truth = np.concatenate((prepared["current_state"][windows[:, 0], None],
                                prepared["next_state"][windows]), axis=1)
        if not np.array_equal(truth, trace[prefix + "truth_states"]):
            raise ValueError("Prediction trace truth differs from original prepared states")
        prediction = trace[prefix + "predicted_states"]
        valid = trace[prefix + "valid"]
        errors = prediction[:, -1] - truth[:, -1]
        errors[:, 2] = wrapped(errors[:, 2])
        if not np.array_equal(errors[valid], trace[prefix + "endpoint_errors"][valid]):
            raise ValueError("Logged endpoint error differs from predicted minus measured state")
        if state_hold and not np.array_equal(prediction, np.repeat(truth[:, :1], k + 1, axis=1)):
            raise ValueError("State-hold trace does not hold the initial state")
        first = windows[:, 0]
        regimes = prepared["regime_code"][first]
        if not np.array_equal(trace[prefix + "regime_code"], regimes):
            raise ValueError("Regime labels differ from first original transition")
        if (not np.array_equal(trace[prefix + "family_id"], prepared["_family_id"][first])
                or not np.array_equal(trace[prefix + "episode_id"], prepared["_episode_id"][first])
                or not np.array_equal(trace[prefix + "episode_index"], prepared["episode_index"][first])):
            raise ValueError("Prediction family/episode identity differs from original prepared rows")
        primary = summary["horizons"][str(k)]
        if (len(windows) != primary["attempted_windows"]
                or int((~valid).sum()) != primary["invalid_windows"]):
            raise ValueError("Attempted/invalid window accounting differs")
        measured = physical_metrics(errors[valid])
        close(measured["object_position_error_mm"]["mean"] / 1000,
              primary["metrics_valid_windows_only"]["object_xy_distance_m"]["mean"], "mean position")
        close(np.deg2rad(measured["object_yaw_error_deg"]["mean"]),
              primary["metrics_valid_windows_only"]["object_yaw_absolute_rad"]["mean"], "mean yaw")
        all_still = still[windows].all(axis=1)
        groups, regime_means = {}, []
        for code, name in enumerate(REGIMES):
            attempted = regimes == code
            eligible = attempted & valid
            term = (np.linalg.norm(errors[eligible, :2], axis=1) / summary["position_tolerance_m"]
                    + np.abs(errors[eligible, 2]) / summary["yaw_tolerance_rad"])
            families = trace[prefix + "family_id"][attempted]
            per_family = Counter(map(str, families))
            groups[name] = {
                "attempted_windows": int(attempted.sum()), "valid_windows": int(eligible.sum()),
                "distinct_families": len(per_family),
                "windows_per_family_min": min(per_family.values()) if per_family else None,
                "windows_per_family_max": max(per_family.values()) if per_family else None,
                "metrics_valid_windows_only": physical_metrics(errors[eligible]),
                "selection_error_mean_valid_windows_only": float(term.mean()) if len(term) else None,
            }
            regime_means.append(groups[name]["selection_error_mean_valid_windows_only"])
            close(groups[name]["selection_error_mean_valid_windows_only"],
                  primary["regimes"][name]["selection_error_mean_valid_windows_only"], name + " score")
        motion = {}
        for name, mask in (("all_original_intervals_object_pose_stationary", all_still),
                           ("at_least_one_original_interval_object_pose_changes", ~all_still)):
            motion[name] = {"attempted_windows": int(mask.sum()),
                            "valid_windows": int((mask & valid).sum()),
                            "metrics_valid_windows_only": physical_metrics(errors[mask & valid])}
        result[str(k)] = {"attempted_windows": len(windows), "valid_windows": int(valid.sum()),
                          "invalid_windows": int((~valid).sum()),
                          "window_rows_sha256": array_hash(windows),
                          "distinct_families": len(set(map(str, trace[prefix + "family_id"]))),
                          "metrics_valid_windows_only": measured, "regimes": groups,
                          "first_interval_stationary_windows": int(still[first].sum()),
                          "motion_groups": motion}
        scores[k] = (float(np.mean(regime_means))
                     if valid.all() and all(value is not None for value in regime_means) else None)
    if summary["selection_status"] == "eligible":
        close(scores[5], summary["selection_score"], "equal-regime K5 score")
    return summary, trace, result, scores[5]


def candidates_analysis(folder, manifest, selection, prepared, norm, x, y, train_hashes):
    candidates = []
    selected_result = None
    for index, chosen in enumerate(selection["candidates"]):
        candidate = folder / "candidates" / f"alpha-{index:02d}"
        primary = read(candidate / "summary.json")
        if primary != chosen or primary["alpha"] != manifest["settings"]["alpha_candidates"][index]:
            raise ValueError("Candidate identity differs from frozen selection/settings")
        model_folder = folder / primary["model_path"]
        if model_folder.resolve() != (candidate / "model").resolve():
            raise ValueError("Candidate model points outside its own immutable artifact folder")
        metadata = read(model_folder / "ridge.json")
        computed_hash = hashlib.sha256(canonical({key: value for key, value in metadata.items()
                                                 if key != "ridge_hash"})).hexdigest()
        if (computed_hash != metadata["ridge_hash"] or metadata["ridge_hash"] != primary["model_hash"]
                or metadata["training_provenance"] != manifest["provenance"]
                or metadata["normalizer_hash"] != norm["normalizer_hash"]
                or metadata["normalizer_stats_hash"] != norm["stats_hash"]
                or metadata["config"] != manifest["original_physical_config"]):
            raise ValueError("Candidate artifact identity/config/normalization differs")
        weights = arrays(model_folder / "weights.npz", metadata["files"]["weights.npz"])
        if (set(weights) != {"weight", "bias"}
                or weights["weight"].shape != (10, 11) or weights["bias"].shape != (10,)
                or any(value.dtype != np.float32 or not np.isfinite(value).all()
                       for value in weights.values())):
            raise ValueError("Unexpected model weight members")
        fingerprint = hashlib.sha256(b"".join(np.ascontiguousarray(weights[k], dtype="<f4").tobytes()
                                               for k in ("weight", "bias"))).hexdigest()
        d = metadata["fit_diagnostics"]
        if (d != primary["fit_diagnostics"] or d["fit_count"] != 3200
                or any(d[key] != value for key, value in train_hashes.items())
                or d["model_weights_sha256"] != fingerprint):
            raise ValueError("Fit diagnostics do not bind actual train rows and stored model")
        residual = x @ weights["weight"].T.astype(np.float64) + weights["bias"].astype(np.float64) - y
        rss = float(np.sum(residual ** 2))
        close(rss, d["stored_model_residual_sum_squares"], "stored train RSS", atol=1e-8)
        summary, _, measured, score = window_analysis(candidate / "validation", prepared,
                                                     manifest["validation_window_hashes"])
        if primary["validation_window_hashes"] != manifest["validation_window_hashes"]:
            raise ValueError("Candidate's declared window hashes differ")
        close(score, primary["selection_score"], "candidate score")
        candidates.append({"index": index, "alpha": primary["alpha"],
                           "selection_score": score, "stored_selection_score": primary["selection_score"],
                           "selection_status": summary["selection_status"],
                           "model_path": primary["model_path"], "ridge_hash": metadata["ridge_hash"],
                           "weight_file_sha256": metadata["files"]["weights.npz"],
                           "fit_count": d["fit_count"], "rank": d["rank"],
                           "condition_estimate": d["condition_estimate"],
                           "stored_model_normalized_residual_mse": d["stored_model_normalized_residual_mse"],
                           "same_original_train_hashes": True,
                           "same_validation_windows_and_order": True,
                           "restored_prediction_equality_primary_check": primary["restored_all_prediction_traces_identical"],
                           "horizons": measured})
        if primary["index"] == selection["selected"]["index"]:
            selected_result = measured
    if len(candidates) != 8:
        raise ValueError("Eight predeclared candidates must remain")
    eligible = [c for c in candidates if c["selection_status"] == "eligible"
                and c["selection_score"] is not None]
    selected = min(eligible, key=lambda c: (c["selection_score"], c["alpha"]))
    if selected["index"] != selection["selected"]["index"]:
        raise ValueError("Exact independent argmin differs from chosen alpha")
    runner_probe = read(folder / "train_only_probe.json")
    if (runner_probe["validation_rows_modified_in_memory"] != 1600
            or runner_probe["original_train_rows_fit"] != 3200
            or runner_probe["model_weights_unchanged"] is not True
            or runner_probe["source_files_mutated"] is not False):
        raise ValueError("Runner's validation fit-mask probe failed")
    _, _, state_hold, _ = window_analysis(folder / "state_hold_validation", prepared,
                                         manifest["validation_window_hashes"], state_hold=True)
    return {"candidates": candidates, "selected": selected,
            "runner_fit_mask_probe": runner_probe, "selected_horizons": selected_result,
            "state_hold_horizons": state_hold,
            "second_best_score_gap": sorted(c["selection_score"] for c in eligible)[1] - selected["selection_score"],
            "selection_exact_argmin_confirmed": True,
            "selection_scope": "validation selected the model; these validation errors are not independent final-test performance",
            "score_definition": "equal mean of three first-transition regime means of XY_error/0.005m + abs(wrap_2pi_yaw_error)/(3deg); endpoint K5",
            "interpretation": "alpha zero is the eligible centered OLS candidate. A tiny exact score gap does not establish robust preference or that regularization is generally unnecessary."}


def development_analysis(folder, selection_hash):
    dev = folder / "development"
    if not (dev / "summary.json").exists() or not (folder / "summary.json").exists():
        return {"status": "PENDING", "reason": "final development and stage summaries have not both completed; missing files are not failures"}
    manifest = read(dev / "manifest.json")
    summary = read(dev / "summary.json")
    ledger = read(dev / "ledger.json")
    causal = read(dev / "causal_summary.json")
    review = read(dev / "dataset_review.json")
    reviewed = {row["episode_id"]: row for row in review["episodes"]}
    guards = summary["future_engine_calls_during_planning"]
    if set(guards) != ENGINE_GUARDS or sum(guards.values()) != 0:
        raise ValueError("Planning future-engine guard accounting differs or is nonzero")
    if manifest["selection_sha256"] != selection_hash:
        raise ValueError("Development changed the preselected alpha record")
    ids = [case["episode_id"] for case in manifest["cases"]]
    if ids != [f"smoke-{i:02d}" for i in range(10)] or ids != [row["episode_id"] for row in ledger]:
        raise ValueError("Declared development executions were dropped, reordered or replaced")
    if ids != [row["episode_id"] for row in causal]:
        raise ValueError("Causal diagnostics do not retain every declared case")
    terminal = Counter(row["task"]["terminal_reason"] for row in ledger if "task" in row)
    if dict(terminal) != summary["terminal_counts"]:
        raise ValueError("Terminal outcomes disagree with all-case ledger")
    case_results, planning_times, one_step_errors = [], [], []
    model_calls = 0
    model_intervals = holding_intervals = full_model_intervals = 0
    for entry, c in zip(ledger, causal):
        case_folder = dev / "episodes" / entry["episode_id"]
        primary = read(case_folder / "summary.json")
        if primary != entry:
            raise ValueError("Individual development outcome differs from preserved ledger")
        rows = [json.loads(line) for line in record(case_folder / "steps.jsonl").read_text().splitlines() if line]
        if [row["index"] for row in rows] != list(range(len(rows))) or len(rows) != entry["transitions"]:
            raise ValueError("Executed control interval accounting differs")
        case_model = [row for row in rows if row["source"] == "model"]
        case_holding = [row for row in rows if row["source"] == "holding"]
        if (len(case_model) != entry["model_selected_applied_intervals"]
                or len(case_holding) != entry["holding_intervals"]):
            raise ValueError("Model/holding action counts differ")
        times, errors, mismatched_horizon = [], [], []
        for row in case_model:
            p = row["plan"]
            if (p["selected_index"] != row["selected_index"]
                    or p["first_action"] != row["requested_action"]
                    or p["case"]["control_step"] != row["index"]
                    or not 1 <= p["model_calls"] <= 10):
                raise ValueError("Stored shared planner/action contract differs")
            times.append(p["wall_time_s"])
            model_calls += p["model_calls"]
            if row["prediction_error"] is not None:
                e = np.asarray(row["prediction_error"]["state_error"], dtype=np.float64)
                if e.shape != (10,) or not np.isfinite(e).all():
                    raise ValueError("Invalid logged actual one-step prediction error")
                close(np.linalg.norm(e[:2]), row["prediction_error"]["object_position_m"], "logged actual position error")
                close(abs(e[2]), row["prediction_error"]["object_yaw_rad"], "logged actual yaw error")
                if (row["physical_valid"] and row["geometry"]["pass"]
                        and abs(row["actual_dt_s"] - .1) <= 1e-9):
                    errors.append(e)
                else:
                    mismatched_horizon.append({"control_index": row["index"],
                                                "actual_dt_s": row["actual_dt_s"],
                                                "raw_prediction_error": row["prediction_error"]})
        if any(row["applied_action"] != [0., 0.] for row in case_holding):
            raise ValueError("Holding did not use the declared common zero command")
        # The full trace is large; verify the first and last actually applied
        # model plans of every case rather than silently claiming every trace
        # has been independently reconstructed from its arrays.
        trace_samples = []
        for row in {r["index"]: r for r in case_model[:1] + case_model[-1:]}.values():
            plan_folder = case_folder / "plans" / f"step-{row['index']:03d}"
            selection = read(plan_folder / "selection.json")
            trace = arrays(plan_folder / "trace.npz")
            index = row["selected_index"]
            if (selection["physics_executed"] is not False
                    or selection["diagnostics"]["candidate_count"] != 256
                    or selection["diagnostics"]["horizon"] != 10
                    or selection["selected_index"] != index
                    or trace["predicted_states"].shape != (256, 11, 10)
                    or not np.array_equal(trace["chosen_first_action"], row["requested_action"])):
                raise ValueError("Representative saved planning trace differs from applied plan")
            error = np.asarray(row["state_after"], dtype=np.float64) - trace["predicted_states"][index, 1]
            error[2] = math.atan2(math.sin(error[2]), math.cos(error[2]))
            if not np.allclose(error, row["prediction_error"]["state_error"], rtol=0, atol=1e-12):
                raise ValueError("Representative actual error differs from selected forecast and observation")
            trace_samples.append(row["index"])
        unexecuted = []
        if (case_folder / "unexecuted_final_plan.json").exists():
            failure = read(case_folder / "unexecuted_final_plan.json")
            p = read(case_folder / "plans" / f"step-{failure['step']:03d}" / "selection.json")
            if failure["no_extra_zero_step"] is not True or p["diagnostics"]["reason"] != "planner_invalid":
                raise ValueError("Unexecuted invalid final plan was not preserved")
            times.append(p["diagnostics"]["wall_time_s"])
            model_calls += p["model_calls"]
            unexecuted.append(failure)
        cp = read(case_folder / "causal_prefix/summary.json")
        if cp != {key: value for key, value in c.items() if key != "episode_id"}:
            raise ValueError("Individual causal summary differs from final aggregate")
        if "selected_prefix_replay_matches_recorded" in cp:
            replay_flags = ("replay_all_state_within_tolerance", "replay_times_match",
                            "replay_applied_actions_match", "replay_terminal_types_match")
            if cp["selected_prefix_replay_matches_recorded"] and not all(cp[key] for key in replay_flags):
                raise ValueError("Causal replay aggregate contradicts detailed checks")
            for index in cp["model_selected_active_contact_intervals"]:
                if (not 0 <= index < cp["interval_count"] or rows[index]["source"] != "model"
                        or not any(value != 0 for value in rows[index]["applied_action"])):
                    raise ValueError("Causal qualifying action does not occur in measured prefix")
            expected_causal = (cp["selected_prefix_replay_matches_recorded"]
                               and bool(cp["model_selected_active_contact_intervals"])
                               and (cp["position_effect_after_contact_m"] >= cp["position_effect_threshold_m"]
                                    or cp["yaw_effect_after_contact_rad"] >= cp["yaw_effect_threshold_rad"]))
            if cp["pass"] != expected_causal:
                raise ValueError("Causal PASS differs from replay/contact/effect conjunction")
        task = entry["task"]
        elapsed = sum(row["actual_dt_s"] for row in rows)
        close(elapsed, task["last_sim_time"] - task["initial_sim_time"], "actual task elapsed", atol=1e-8)
        physical_task = manifest["resolved_planning_config"]["task"]
        final_state = np.asarray(rows[-1]["state_after"], dtype=np.float64)
        position_error = float(np.linalg.norm(final_state[:2] - physical_task["goal_xy_m"]))
        delta_yaw = final_state[2] - physical_task["goal_yaw_rad"]
        task_yaw_error = abs(.5 * math.atan2(math.sin(2 * delta_yaw), math.cos(2 * delta_yaw)))
        close(position_error, entry["final_metrics"]["e_pos_m"], "final observed goal position error")
        close(task_yaw_error, entry["final_metrics"]["e_yaw_rad"], "final observed pi-symmetric goal yaw error")
        qualifies = (position_error <= physical_task["position_tolerance_m"]
                     and task_yaw_error <= physical_task["yaw_tolerance_rad"]
                     and np.linalg.norm(final_state[3:5]) <= physical_task["linear_speed_tolerance_m_s"]
                     and abs(final_state[5]) <= physical_task["angular_speed_tolerance_rad_s"])
        if bool(qualifies) != entry["final_metrics"]["qualifies"]:
            raise ValueError("Final qualifying status differs from observed state")
        if task["phase"] == "ready" and not (qualifies and task["held_time_s"] >= physical_task["hold_time_s"]):
            raise ValueError("Ready case lacks final pose/speed or reported hold threshold")
        valid_log_count = sum(row["physical_valid"] and row["geometry"]["pass"]
                              and abs(row["actual_dt_s"] - .1) <= 1e-9 for row in rows)
        if (entry["episode_id"] not in reviewed
                or reviewed[entry["episode_id"]]["transitions"] != len(rows)
                or reviewed[entry["episode_id"]]["valid"] != valid_log_count):
            raise ValueError("Compact observed validity differs from primary completed raw dataset review")
        model_intervals += len(case_model)
        holding_intervals += len(case_holding)
        full_model_intervals += len(errors)
        planning_times.extend(times)
        one_step_errors.extend(errors)
        support = [row["plan"]["support"] for row in case_model]
        selected_outside = [s["selected_candidate_outside_steps"] for s in support
                            if s["selected_candidate_outside_steps"] is not None]
        pairs = sum(s["evaluated_finite_input_pairs"] for s in support)
        support_measurement = {
            "all_candidate_finite_predicted_input_pairs": pairs,
            "all_candidate_pairs_outside_any_train_marginal": sum(s["outside_any_train_marginal_bound"] for s in support),
            "selected_predicted_input_steps": len(selected_outside) * 10,
            "selected_predicted_input_steps_outside_any_train_marginal": sum(selected_outside),
            "selected_plans_with_at_least_one_marginal_exceedance": sum(value > 0 for value in selected_outside),
            "scope": "predicted input features versus empirical separate train-feature ranges; not joint support, actual future observations, independent trials or proof of an OOD failure cause",
        }
        case_results.append({"episode_id": entry["episode_id"], "outcome": task["phase"],
                             "terminal_reason": task["terminal_reason"], "violations": task["violations"],
                             "executed_intervals": len(rows), "model_applied_intervals": len(case_model),
                             "holding_intervals": len(case_holding), "elapsed_sim_s": elapsed,
                             "reported_continuous_held_time_s": task["held_time_s"],
                             "valid_completed_intervals": valid_log_count,
                             "final_position_error_mm": entry["final_metrics"]["e_pos_m"] * 1000,
                             "final_task_yaw_error_deg": math.degrees(entry["final_metrics"]["e_yaw_rad"]),
                             "final_qualifies": entry["final_metrics"]["qualifies"],
                             "planning_time_s": describe(times),
                             "representative_saved_plan_trace_indices_verified": trace_samples,
                             "unexecuted_final_plans_retained": unexecuted,
                             "train_marginal_range_diagnostic": support_measurement,
                             "actual_full_horizon_one_step_errors": physical_metrics(np.asarray(errors).reshape(-1, 10)),
                             "partial_or_invalid_interval_errors_retained": mismatched_horizon,
                             "causal_prefix": cp})
    if (len(planning_times) != summary["resources"]["plan_count"]
            or model_intervals + holding_intervals != summary["resources"]["execution_intervals"]
            or model_calls != summary["actual_model_calls"]
            or sum(row["pass"] for row in causal) != summary["causal_prefix_pass_count"]):
        raise ValueError("Development count or causal count differs")
    close(np.mean(planning_times), summary["resources"]["planning_mean_s"], "planning mean")
    close(np.percentile(planning_times, 95), summary["resources"]["planning_p95_s"], "planning p95")
    return {"status": "COMPLETE", "primary_gate_pass": summary["pass"], "primary_checks": summary["checks"],
            "all_ten_original_cases_accounted": True, "terminal_counts": dict(terminal),
            "ready_cases": terminal.get("ready", 0), "cases": case_results,
            "future_engine_calls_in_planning": guards, "future_engine_call_total": sum(guards.values()),
            "planning_time_s": describe(planning_times), "model_applied_intervals": model_intervals,
            "actual_model_calls": model_calls,
            "planning_contract": {"candidate_count": 256, "horizon": 10,
                                  "candidate_speed_norm_max_m_s": manifest["settings"]["candidate_speed_norm_max_m_s"],
                                  "max_action_intervals": manifest["settings"]["max_action_intervals"],
                                  "geometry_proposal_enabled": manifest["resolved_planning_config"]["planner"]["geometry_proposal_enabled"]},
            "holding_intervals": holding_intervals,
            "recorded_intervals": sum(row["transitions"] for row in review["episodes"]),
            "valid_intervals": sum(row["valid"] for row in review["episodes"]),
            "excluded_intervals": sum(row["transitions"] - row["valid"] for row in review["episodes"]),
            "primary_raw_review_matches_compact_observed_validity": True,
            "actual_full_horizon_one_step_error_intervals": full_model_intervals,
            "actual_one_step_errors": physical_metrics(np.asarray(one_step_errors).reshape(-1, 10)),
            "causal_prefix_pass_count": sum(row["pass"] for row in causal),
            "resources_primary": summary["resources"],
            "causal_scope": "selected measured prefix vs zero after actual execution; contribution only, not candidate optimality, model superiority or alpha selection",
            "scope": "same ten previously used development cases, no independent final test; error groups retain every attempted outcome and explicit partial/invalid records"}


def analyze(run, selection_only=False):
    USED.clear()
    folder = ROOT / "runs" / run
    manifest = read(folder / "manifest.json")
    selection = read(folder / "selection.json")
    source = source_audit(folder, manifest)
    prepared, lm, norm, x, y, train_hashes, main = load_main(manifest)
    selected = candidates_analysis(folder, manifest, selection, prepared, norm, x, y, train_hashes)
    development = ({"status": "PENDING", "reason": "selection-only analysis requested during development"}
                   if selection_only else development_analysis(folder, sha(folder / "selection.json")))
    stage = {"status": "PENDING"}
    if not selection_only and (folder / "summary.json").exists():
        primary = read(folder / "summary.json")
        stage = {"stage_verdict": primary["stage_verdict"], "primary_checks": primary["checks"],
                 "test_count": primary["test_count"], "resources": primary["resources"]}
        artifact_index = read(folder / "artifact_hashes.json")
        for relative, actual in USED.items():
            p = ROOT / relative
            if p.is_relative_to(folder) and p.name != "artifact_hashes.json":
                inner = str(p.relative_to(folder))
                if inner not in artifact_index["sha256"] or actual != artifact_index["sha256"][inner]:
                    raise ValueError("Read run artifact differs from final artifact index")
        stage["all_used_run_inputs_match_final_artifact_index"] = True
    output = {
        "schema": "p3-ridge-derived-analysis-v1", "run_id": run,
        "created_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "stage": stage, "source_verification": source, "main_learning_context": main,
        "selection": selected, "development": development,
        "metric_labels_ko": {"object_position_error_mm": "예측 종료 시 물체 위치 오차(mm)",
                              "object_yaw_error_deg": "예측 종료 시 물체 각도 오차(도, 2π 차이)",
                              "final_task_yaw_error_deg": "실제 종료 시 목표 각도 오차(도, π 대칭)"},
        "stationary_definition": {"original_interval": "norm(delta object XY)<1e-9 m and abs(wrap2pi(delta object yaw))<1e-9 rad",
                                  "window_groups": "all K original intervals stationary vs at least one changing interval; endpoint return-to-start alone is not called stationary",
                                  "scope": "object pose only; does not mean the pusher, velocities or commands are identical, useless data, duplicated rows or full learning support"},
        "limitations": [
            "Validation selected alpha; its errors are descriptive development evidence, not independent held-out performance.",
            "Overlapping K windows share rows and families; window counts are not independent trials. No confidence interval or statistical superiority is claimed.",
            "State-hold is a no-response prediction diagnostic, not a tuned task controller or a fair main-data MLP comparison.",
            "Both stationary and moving windows remain; neither clipping predictions nor deleting weak outcomes is used.",
            "A lower K5 selection score does not ensure closed-loop readiness, physical real-time behavior or generalization.",
            "The fit-mask perturbation probe checks that particular leakage path; data provenance, unit checks and actual train-row digests provide separate evidence.",
            "This analysis does not rerun physics, refit Ridge, refit normalization or repeat the complete raw-main validity audit.",
            "Development uses all compact logs for accounting and error summaries; only each case's first/last applied model traces are separately checked against selected forecasts.",
            "The user understanding and real-world usefulness of the project are not assessed by these measurements.",
        ],
        "analysis_code_sha256": sha(__file__), "numpy_version": str(np.__version__),
        "used_input_sha256": dict(sorted(USED.items())),
    }
    output["all_used_inputs_unchanged_after_analysis"] = all(sha(ROOT / name) == value for name, value in USED.items())
    if not output["all_used_inputs_unchanged_after_analysis"]:
        raise ValueError("Inputs changed during derived analysis")
    REPORT.write_text(json.dumps(output, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selection_alpha": selected["selected"]["alpha"],
                      "selection_score": selected["selected"]["selection_score"],
                      "development_status": development["status"],
                      "used_inputs": len(USED), "report_sha256": sha(REPORT)}))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    data = json.loads(REPORT.read_text(encoding="utf-8"))
    if data["analysis_code_sha256"] != sha(__file__):
        raise ValueError("Plot code differs from the analyzed code; analyze again first")
    plt.rcParams.update({"font.size": 10.5, "axes.labelsize": 11., "axes.titlesize": 13.,
                         "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
                         "legend.fontsize": 10.5, "svg.fonttype": "none"})
    outputs, svg_font_audit = {}, {}

    def save(fig, name):
        for suffix in ("png", "svg"):
            path = FIGURES / f"p3_ridge_{name}.{suffix}"
            fig.savefig(path, dpi=180)
            outputs[str(path.relative_to(ROOT))] = sha(path)
            if suffix == "svg":
                sizes = [float(value) for value in re.findall(
                    r"font-size:\s*([0-9.]+)px", path.read_text(encoding="utf-8"))]
                if not sizes or min(sizes) < 10.5:
                    raise ValueError("Exported SVG contains a text font smaller than 10.5px")
                svg_font_audit[str(path.relative_to(ROOT))] = {
                    "actual_font_sizes_px": sorted(set(sizes)),
                    "actual_minimum_font_size_px": min(sizes),
                    "all_actual_text_fonts_at_least_10_5px": True,
                    "scope": "exported font-size declarations, including MathText tspan sizes; actual PDF placement checked separately",
                }
        plt.close(fig)

    candidates = data["selection"]["candidates"]
    score = [c["selection_score"] for c in candidates]
    alphas = [c["alpha"] for c in candidates]
    fig = plt.figure(figsize=(9., 7.5))
    grid = fig.add_gridspec(2, 2, width_ratios=(1, 5), left=.11, right=.97, bottom=.15,
                          top=.9, hspace=.4, wspace=.05)
    zero, positive = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    zero.scatter([0], score[:1], s=65, color="#257d69", label="Selected OLS")
    zero.set_xticks([0], ["alpha = 0"])
    zero.set_xlim(-.6, .6)
    zero.set_ylabel("Equal-regime K5 endpoint score")
    positive.plot(alphas[1:], score[1:], "o-", color="#46647f")
    positive.set_xscale("log")
    def plain_scientific(value, _position):
        coefficient, exponent = f"{value:.0e}".split("e")
        return f"{coefficient}e{int(exponent)}"
    positive.xaxis.set_major_formatter(FuncFormatter(plain_scientific))
    positive.set_xlabel("Positive alpha (log scale; SUM-loss penalty)")
    for ax in (zero, positive):
        ax.set_ylim(min(score) - .001, max(score) + .001)
        ax.grid(axis="y", alpha=.22)
    positive.tick_params(labelleft=False)
    zero.set_title("OLS", pad=12)
    positive.set_title("All eight candidates use identical train rows and validation windows", pad=12)
    lower = fig.add_subplot(grid[1, :])
    gaps = np.asarray(score[1:]) - score[0]
    lower.plot(alphas[1:], gaps, "o-", color="#b88c43")
    lower.set_xscale("log")
    lower.set_yscale("log")
    lower.xaxis.set_major_formatter(FuncFormatter(plain_scientific))
    lower.yaxis.set_major_formatter(FuncFormatter(plain_scientific))
    lower.set_ylabel("Score minus alpha=0 score (log scale)")
    lower.set_xlabel("Positive alpha")
    lower.grid(alpha=.22)
    lower.set_title(f"Exact minimum selects alpha=0; next gap is {data['selection']['second_best_score_gap']:.3g}")
    fig.text(.5, .035, "No rounding ties or closed-loop reselection. A tiny validation gap is not robust superiority.", ha="center", fontsize=10.5)
    save(fig, "alpha_selection")

    selected = data["selection"]["selected_horizons"]
    held = data["selection"]["state_hold_horizons"]
    fig, axes = plt.subplots(2, 2, figsize=(9., 7.5))
    fig.subplots_adjust(left=.1, right=.97, bottom=.14, top=.9, hspace=.4, wspace=.27)
    metric_labels = (("object_position_error_mm", "Mean object position error (mm)"),
                     ("object_yaw_error_deg", "Mean object yaw error (deg, 2pi)"))
    for col, (metric, label) in enumerate(metric_labels):
        for name, result, color in (("Selected Ridge", selected, "#257d69"),
                                    ("State hold", held, "#b88c43")):
            axes[0, col].plot(HORIZONS, [result[str(k)]["metrics_valid_windows_only"][metric]["mean"] for k in HORIZONS],
                              "o-", label=name, color=color)
        axes[0, col].set_xticks(HORIZONS)
        axes[0, col].set_xlabel("Recorded-action free prediction horizon K")
        axes[0, col].set_ylabel(label)
        axes[0, col].legend(loc="upper left")
        xaxis = np.arange(3)
        for name, result, offset, color in (("Selected Ridge", selected, -.18, "#257d69"),
                                           ("State hold", held, .18, "#b88c43")):
            axes[1, col].bar(xaxis + offset, [result["5"]["regimes"][r]["metrics_valid_windows_only"][metric]["mean"] for r in REGIMES],
                            width=.36, color=color, label=name)
        axes[1, col].set_xticks(xaxis, [f"{label}\nn={selected['5']['regimes'][regime]['attempted_windows']}"
                                      for label, regime in zip(("No contact", "Sustained", "Transition"), REGIMES)])
        axes[1, col].set_ylabel(label)
        axes[1, col].set_title("K5; regime of first measured interval")
        for ax in axes[:, col]:
            ax.grid(axis="y", alpha=.22)
            ax.set_axisbelow(True)
    fig.suptitle("Selected Ridge vs no-response diagnostic on the same validation windows", fontsize=13.)
    fig.text(.5, .035, "Validation selects alpha. Overlapping windows share families; no final-test or task-success claim.", ha="center", fontsize=10.5)
    save(fig, "prediction_errors")

    development = data["development"]
    if development["status"] == "COMPLETE":
        cases = development["cases"]
        fig, axes = plt.subplots(2, 1, figsize=(9., 7.5))
        fig.subplots_adjust(left=.1, right=.97, bottom=.21, top=.88, hspace=.37)
        colors = ["#257d69" if row["terminal_reason"] == "ready" else "#b88c43"
                  if row["terminal_reason"] == "timeout" else "#b85b52" for row in cases]
        xaxis = np.arange(10)
        for ax, key, ylabel, tolerance in ((axes[0], "final_position_error_mm", "Final position error to goal (mm)", 5.),
                                            (axes[1], "final_task_yaw_error_deg", "Final goal yaw error (deg, pi symmetry)", 3.)):
            values = [row[key] for row in cases]
            ax.bar(xaxis, values, color=colors)
            ax.axhline(tolerance, color="#b85b52", linestyle="--", label=f"Pose limit: {tolerance:g}")
            ax.set_ylim(0, max(max(values) * 1.22, tolerance * 1.8))
            ax.set_xticks(xaxis, [f"{i:02d}" for i in range(10)])
            ax.set_xlabel("Original development case smoke-00 .. smoke-09")
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=.22)
            ax.set_axisbelow(True)
            ax.legend(loc="upper right")
        fig.suptitle(f"All ten outcomes retained: {development['ready_cases']} ready / "
                     f"{development['terminal_counts'].get('timeout', 0)} timeout", fontsize=13.)
        fig.legend(handles=[Patch(color="#257d69", label="Ready"),
                            Patch(color="#b88c43", label="Timeout")],
                   loc="upper center", bbox_to_anchor=(.5, .955), ncol=2)
        fig.text(.5, .095, f"Plan mean {development['planning_time_s']['mean']:.4f}s; p95 {development['planning_time_s']['p95']:.4f}s. "
                 f"Planning future-engine calls: {development['future_engine_call_total']}.", ha="center", fontsize=10.5)
        fig.text(.5, .05, f"Causal prefixes: {development['causal_prefix_pass_count']}/10. Contribution does not establish goal success.\n"
                 "Previously used development cases; no final-test, MLP-superiority or device real-time claim.", ha="center", fontsize=10.5)
        save(fig, "development")
    audit = {"analysis_report_sha256": sha(REPORT), "analysis_code_sha256": sha(__file__),
             "figures": outputs, "minimum_label_font_pt": 10.5, "figure_size_inches": [9., 7.5],
             "actual_svg_font_audit": svg_font_audit,
             "all_actual_svg_text_fonts_at_least_10_5px": True,
             "aspect_ratio": 1.2, "anticipated_pdf_width_pt": 480.,
             "minimum_label_pt_at_pdf_width": 10.5 * 480. / (9. * 72.),
             "scope": "new P3-03 files only; actual exported PNG and PDF placement need visual QA"}
    (ROOT / "reports/p3_03_figures_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="p3-03-20261009-01")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run):
        raise ValueError("Unsafe run ID")
    if args.analyze:
        analyze(args.run, args.selection_only)
    if args.plot:
        plot()
