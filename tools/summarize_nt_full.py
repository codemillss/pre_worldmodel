#!/usr/bin/env python3
"""Summarize saved N-T candidate diagnostics without physics/model imports.

Raw/full-probe exact comparisons are executor evidence, not repeated here. This
reader authenticates their receipts, requires all three exact probe hashes for
stable groups, and applies the predeclared cost/validity/readiness rules. A
partial run remains partial, with missing candidates rather than exclusions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("R", "S1-11", "S1-23", "S1-47", "SK-11", "SK-23", "SK-47")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.dtype.str.encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def passed(value):
    return isinstance(value, dict) and value.get("status") == "PASS_EXACT"


def group_result(candidate, panel, rows):
    result = {"candidate_id": candidate, "panel": panel,
              "present_repeats": sorted(row["repeat"] for row in rows),
              "status": "MISSING_OR_PARTIAL", "stable": False}
    if result["present_repeats"] != [0, 1, 2]:
        return result
    signature_fields = ("physical_valid", "horizon_completed", "ready", "ready_prefix",
                        "ready_eligible", "first_ready_interval", "termination", "cost_finite")
    exact = (len({row["whole_probe_typed_sha256"] for row in rows}) == 1
             and all(passed(row["raw_roundtrip"]) and row["initial_snapshot_exact"] for row in rows)
             and passed(rows[0]["full_probe_roundtrip"])
             and all(passed(row["full_probe_repeat_exact"]) for row in rows[1:]))
    agrees = all(all(row[key] == rows[0][key] for key in signature_fields) for row in rows)
    if not exact or not agrees:
        result.update(status="UNSTABLE_REPLAY", exact_full_probe_repeat=exact,
                      validity_ready_agreement=agrees)
        return result
    first = rows[0]
    physical = bool(first["physical_valid"])
    h5_valid = physical and bool(first["horizon_completed"]) and first["intervals"] == 5 if panel == "A" else None
    finite = all(row["cost_finite"] and isinstance(row["cost"], (int, float))
                 and math.isfinite(row["cost"]) for row in rows) if panel == "A" else None
    rankable = h5_valid and finite if panel == "A" else False
    result.update(status="STABLE", stable=True, physical_valid=physical,
                  horizon_completed=bool(first["horizon_completed"]),
                  H5_physical_valid=h5_valid, cost_finite=finite, rankable_A=rankable,
                  ready=bool(first["ready"]), ready_prefix=bool(first["ready_prefix"]),
                  ready_eligible=bool(first["ready_eligible"]),
                  first_ready_interval=first["first_ready_interval"],
                  intervals=first["intervals"], termination=first["termination"],
                  reasons=first["reasons"], whole_probe_typed_sha256=first["whole_probe_typed_sha256"],
                  cost_L=min(row["cost"] for row in rows) if rankable else None,
                  cost_U=max(row["cost"] for row in rows) if rankable else None,
                  cost_components=first["cost_components"] if rankable else None)
    if first.get("first_repeat_complete_probe_reference") is not None:
        result["first_repeat_complete_probe_reference"] = first["first_repeat_complete_probe_reference"]
    require(not first["ready_eligible"] or (physical and first["ready"]
            and (panel == "B" or h5_valid)), "Impossible ready_eligible receipt")
    if panel == "B":
        require(all(row["cost"] is None and row["cost_components"] == {} for row in rows),
                "B must never be ranked with the H5 cost")
    return result


def cost_assessment(groups, *, complete):
    valid = [group for group in groups if group.get("stable") and group.get("rankable_A")]
    if not valid:
        return {"status": "NO_VALID_ACTUAL", "epsilon": None,
                "possible_cost_optimum_ids": [], "C": {"status": "UNRESOLVED_NO_VALID_ACTUAL"}}
    maximum = max(abs(group[key]) for group in valid for key in ("cost_L", "cost_U"))
    epsilon = 1e-6 + 1e-5 * maximum
    min_upper = min(group["cost_U"] for group in valid)
    possible = [group["candidate_id"] for group in valid if group["cost_L"] <= min_upper + epsilon]
    ready = [group for group in valid if group["ready_eligible"]]
    nonready = [group for group in valid if not group["ready_eligible"]]
    result = {"status": "FULL_BANK_RANKING" if complete else "OBSERVED_SUBSET_RANKING_ONLY",
              "epsilon": epsilon, "epsilon_max_abs_finite_cost": maximum,
              "rankable_candidate_ids": [group["candidate_id"] for group in valid],
              "best_upper_cost": min_upper,
              "best_lower_cost": min(group["cost_L"] for group in valid),
              "possible_cost_optimum_ids": possible,
              "stable_ready_A_ids": [group["candidate_id"] for group in ready]}
    if not complete:
        conclusion = {"status": "UNRESOLVED_INCOMPLETE_OR_UNSTABLE_BANK"}
    elif not ready:
        conclusion = {"status": "UNRESOLVED_NO_READY_A",
                      "meaning": "No stable eligible H5 readiness in this bank; cost/readiness mismatch cannot be identified."}
    else:
        ready_ids = {group["candidate_id"] for group in ready}
        optimum_ready = sorted(ready_ids.intersection(possible))
        all_nonready = not optimum_ready
        nonready_upper = min((group["cost_U"] for group in nonready), default=None)
        ready_lower = min(group["cost_L"] for group in ready)
        strict = nonready_upper is not None and nonready_upper + epsilon < ready_lower
        conclusion = {"status": "SUPPORTED_LOCAL_COST_READINESS_MISMATCH" if all_nonready and strict
                      else "UNRESOLVED_POSSIBLE_OPTIMUM_INCLUDES_READY_OR_NO_STRICT_GAP",
                      "all_possible_optimum_nonready": all_nonready,
                      "possible_optimum_ready_ids": optimum_ready,
                      "best_nonready_U": nonready_upper, "min_ready_L": ready_lower,
                      "strict_nonready_U_plus_epsilon_below_ready_L": strict}
    result["C"] = conclusion
    return result


def model_assessment(meta, forecast, groups, costs, *, complete):
    eligible = np.flatnonzero(forecast["valid"] & np.isfinite(forecast["total_cost"]))
    selected = None if not len(eligible) else int(eligible[np.argmin(forecast["total_cost"][eligible])])
    require(selected == meta["selected_original_candidate_id"], "Selected ID differs from sealed costs")
    require(int(forecast["valid"].sum()) == meta["valid_count"], "Predicted validity count mismatch")
    false_invalid = [group["candidate_id"] for group in groups
                     if group.get("stable") and group.get("H5_physical_valid")
                     and not bool(forecast["valid"][group["candidate_id"]])]
    false_valid = [group["candidate_id"] for group in groups
                   if group.get("stable") and not group.get("H5_physical_valid")
                   and bool(forecast["valid"][group["candidate_id"]])]
    report = {"predicted_valid_count": int(forecast["valid"].sum()),
              "predicted_valid_finite_cost_count": len(eligible),
              "selected_original_candidate_id": selected,
              "M1_false_invalid_actual_A_valid_ids": false_invalid,
              "M1_status": "SUPPORTED_LOCAL" if false_invalid else
                  ("NOT_OBSERVED_FULL_BANK" if complete else "UNRESOLVED_INCOMPLETE_BANK"),
              "predicted_valid_actual_A_invalid_ids": false_valid,
              "invalid_failure_step_counts": {str(key): value for key, value in
                  sorted(Counter(int(value) for value in forecast["failure_steps"][~forecast["valid"]]).items())}}
    if selected is None:
        report["M2"] = {"status": "NO_VALID_PREDICTION", "regret_interval": None}
        return report
    group = groups[selected]
    report["selected_actual_A"] = {key: group.get(key) for key in
        ("status", "H5_physical_valid", "ready_eligible", "cost_L", "cost_U")}
    if not group.get("stable"):
        report["M2"] = {"status": "UNRESOLVED_SELECTED_ACTUAL_MISSING_OR_UNSTABLE", "regret_interval": None}
    elif not group["H5_physical_valid"]:
        report["M2"] = {"status": "SUPPORTED_SELECTED_ACTUAL_A_INVALID", "regret_interval": None}
    elif not group["rankable_A"] or costs["epsilon"] is None:
        report["M2"] = {"status": "UNRESOLVED_NONFINITE_COST", "regret_interval": None}
    else:
        epsilon = costs["epsilon"]
        strict = group["cost_L"] > costs["best_upper_cost"] + epsilon
        report["M2"] = {"status": "SUPPORTED_STRICT_LOCAL_RANKING_ERROR" if strict else
            ("NOT_ESTABLISHED_BY_STRICT_RULE" if complete else "NOT_ESTABLISHED_ON_OBSERVED_SUBSET"),
            "strict_selected_L_gt_best_U_plus_epsilon": strict,
            "selected_L_minus_best_U": group["cost_L"] - costs["best_upper_cost"],
            "cost_regret_interval_against_observed_valid_subset": [
                max(0., group["cost_L"] - costs["best_upper_cost"]),
                max(0., group["cost_U"] - costs["best_lower_cost"])],
            "scope": "Exact fixed H5 cost among stable physically valid finite-cost candidates; not handoff regret."}
    return report


def teacher_assessment(run, source):
    folder = run / "teacher"
    path = folder / "summary.json"
    if not path.exists():
        return {"status": "NOT_AVAILABLE_NOT_768_COMPLETE" if source.get("completed_trajectories") != 768
                else "NOT_AVAILABLE_OR_INCOMPLETE_TEACHER_STAGE",
                "source_row_count": None, "model_count": 0,
                "partial_teacher_files": sorted(item.name for item in folder.iterdir() if item.is_file()) if folder.exists() else [],
                "interpretation": "No fresh full-bank teacher result claimed; no prior repeated timing benchmark is substituted."}
    metadata = load(path)
    require(source.get("teacher_metadata") == metadata, "Teacher top-level metadata mismatch")
    require(metadata["all_actual_futures_finished_before_teacher"] is True
            and source["completed_trajectories"] == 768, "Teacher stage cannot precede full768 actuals")
    count = metadata["rows"]
    require(0 < count <= 640, "Invalid fresh teacher row count")
    source_path = folder / "source-fresh-A-first-repeat.npz"
    require(sha(source_path) == metadata["source"]["sha256"]
            and passed(metadata["source"]["roundtrip"]), "Teacher source receipt mismatch")
    masks = {}
    with np.load(source_path, allow_pickle=False) as values:
        require(values["start"].shape == values["actual"].shape == (count, 10)
                and values["applied"].shape == (count, 2), "Teacher source array shape mismatch")
        identifiers = values["candidate_interval"]
        require(identifiers.shape == (count, 2) and len(set(map(tuple, identifiers))) == count,
                "Teacher candidate/interval rows are not unique")
        require(np.all((identifiers[:, 0] >= 0) & (identifiers[:, 0] < 128))
                and np.all((identifiers[:, 1] >= 0) & (identifiers[:, 1] < 5)), "Bad teacher identifiers")
        for key in values.files:
            if key.startswith("actual_"):
                require(values[key].shape == (count,) and values[key].dtype == np.bool_, "Teacher validity mask mismatch")
                masks[key] = {"true": int(values[key].sum()), "false": int((~values[key]).sum())}
    require(set(metadata["models"]) == set(MODELS), "Seven complete teacher model receipts required")
    models = {}
    for name in MODELS:
        record = metadata["models"][name]
        file = folder / f"one-step-{name}.npz"
        require(record["rows"] == count and sha(file) == record["receipt"]["sha256"]
                and passed(record["receipt"]["roundtrip"]), "Teacher model receipt mismatch")
        with np.load(file, allow_pickle=False) as values:
            require(values["predicted"].shape == values["raw_deltas"].shape == (count, 10)
                    and values["valid"].shape == (count,) and values["valid"].dtype == np.bool_,
                    "Teacher model prediction shape mismatch")
            models[name] = {"file_sha256": sha(file), "rows": count, "predicted_valid": int(values["valid"].sum())}
    return {"status": "AUTHENTICATED_FRESH_A_FIRST_REPEAT_TEACHER", "source_row_count": count,
            "unique_candidate_interval_rows": count, "source_sha256": sha(source_path),
            "metadata_sha256": sha(path), "actual_masks": masks, "model_count": 7, "models": models,
            "scope": "Saved unique fresh transition rows and complete masks; no new inference or training."}


def summarize(run):
    started = time.perf_counter()
    run = run.resolve()
    entry_path, seal_path = run / "entry.json", run / "prospective_forecast_seal.json"
    entry, seal = load(entry_path), load(seal_path)
    config_path = ROOT / "configs/followup_candidate_diagnostic.v1.json"
    design = load(config_path)
    full_config_path = ROOT / "configs/followup_nt_full.v1.json"
    full_config = load(full_config_path)
    require(design["models"] == list(MODELS), "Design model identity mismatch")
    require(sha(full_config_path) == entry["config_sha256"], "Executed full-bank config changed")
    require(full_config["original_design_sha256"] == sha(config_path), "Original design binding changed")
    require(full_config["original_manifest_sha256"] == design["original_manifest_sha256"], "Original manifest identity mismatch")
    require(entry["full_input_typed_sha256"] == full_config["expected_input_typed_sha256"]
            and entry["bank_hash"] == full_config["expected_candidate_hash"], "Configured input/bank mismatch")
    require(entry["model_and_normalizer_sha256"] == full_config["expected_model_normalizer_sha256"]
            and len(entry["model_and_normalizer_sha256"]) == 22, "Configured model/normalizer map mismatch")
    for relative, expected in entry["model_and_normalizer_sha256"].items():
        require(sha(ROOT / relative) == expected, "Loaded model/normalizer changed")
    require(seal["before_any_new_actual_future"] is True, "Missing prospective forecast seal")
    require(all(value == 0 for value in seal["future_engine_calls_during_prediction"].values()),
            "Nonzero future engine counter")
    for relative, expected in seal["forecasts"].items():
        path = (run / relative).resolve()
        require(path.is_relative_to(run) and sha(path) == expected, "Sealed forecast differs: " + relative)
    metadata = load(run / "forecasts/predictions.json")
    require(set(metadata) == set(MODELS), "Seven forecast files required")
    with np.load(run / "bank.npz", allow_pickle=False) as saved:
        bank = saved["actions"].copy()
        require(bank.dtype == np.float64 and bank.shape == (128, 5, 2), "Invalid candidate shape/dtype")
        require(np.array_equal(saved["original_ids"], np.arange(128)), "Original IDs changed")
    require(array_hash(bank) == entry["bank_hash"], "Bank identity differs from entry")
    rows, files, incomplete_trials = [], {}, []
    for candidate, panel, repeat in product(range(128), ("A", "B"), range(3)):
        trial = run / "actual/N-T" / f"candidate-{candidate:03d}" / panel / f"repeat-{repeat}"
        path = trial / "summary.json"
        if not path.exists():
            if trial.exists():
                incomplete_trials.append({"candidate_id": candidate, "panel": panel, "repeat": repeat,
                                          "incomplete_receipt_present": (trial / "incomplete.json").exists()})
            continue
        row = load(path)
        require((row["candidate_id"], row["panel"], row["repeat"]) == (candidate, panel, repeat), "Trial identity mismatch")
        require(row["state"] == "N-T" and row["candidate_hash"] == array_hash(bank[candidate]), "Trial candidate hash mismatch")
        require(type(row["intervals"]) is int and 0 <= row["intervals"] <= (5 if panel == "A" else 8), "Bad interval count")
        require(row["ready_prefix"] == (row["ready"] and row["first_ready_interval"] <= 5), "Prefix readiness mismatch")
        receipt = row["diagnostic_receipt"]
        require(sha(trial / "diagnostic.npz") == receipt["sha256"], "Diagnostic hash mismatch")
        require(passed(receipt["roundtrip"]), "Diagnostic roundtrip did not pass")
        if panel == "A" and row["cost_finite"]:
            require(math.isclose(sum(row["cost_components"].values()), row["cost"], rel_tol=1e-12, abs_tol=1e-8), "Nine-term cost sum mismatch")
        # The trial summary is written before its own final serialization
        # duration is known; the top-level row has the final timing receipt.
        final_timing = trial / "timing_final.json"
        if final_timing.exists():
            row["timings_ns"] = load(final_timing)["timings_ns"]
        trial_entry = load(trial / "entry.json")
        require(trial_entry["prospective_seal_sha256"] == sha(seal_path), "Trial not bound to forecast seal")
        rows.append(row)
        files[path.relative_to(run).as_posix()] = sha(path)
        files[(trial / "diagnostic.npz").relative_to(run).as_posix()] = receipt["sha256"]
        if repeat == 0:
            full_receipt_path = trial / "complete_probe_receipt.json"
            full_receipt = load(full_receipt_path)
            files[full_receipt_path.relative_to(run).as_posix()] = sha(full_receipt_path)
            portable_files = {}
            for filename, expected in full_receipt["files_sha256"].items():
                full_path = Path(filename).resolve()
                require(full_path.is_relative_to(run), "Complete-probe receipt outside this run")
                require(sha(full_path) == expected, "Complete-probe saved file hash mismatch")
                relative = full_path.relative_to(run).as_posix()
                portable_files[relative] = expected
                files[relative] = expected
            row["first_repeat_complete_probe_reference"] = {
                "receipt": full_receipt_path.relative_to(run).as_posix(),
                "receipt_sha256": files[full_receipt_path.relative_to(run).as_posix()],
                "files_sha256": portable_files}
    durable_row_count = len(rows)
    source = None
    source_path = None
    for name in ("summary.json", "incomplete.json"):
        if (run / name).exists():
            source_path, source = run / name, load(run / name)
            break
    require(source is not None, "Run must have a complete or stopped receipt before analysis")
    uncommitted_rows = []
    if "rows" in source:
        index = {(row["candidate_id"], row["panel"], row["repeat"]): row for row in source["rows"]}
        require(len(index) == len(source["rows"]), "Duplicate top-level committed keys")
        durable_index = {(row["candidate_id"], row["panel"], row["repeat"]): row for row in rows}
        require(set(index).issubset(durable_index), "Committed row missing durable trial summary")
        uncommitted_rows = [{"candidate_id": key[0], "panel": key[1], "repeat": key[2],
                            "summary_sha256": files[f"actual/N-T/candidate-{key[0]:03d}/{key[1]}/repeat-{key[2]}/summary.json"],
                            "scope": "Durable trial summary not committed in final top-level rows; excluded from scientific ranking without deletion."}
                           for key in sorted(set(durable_index) - set(index))]
        rows = [durable_index[key] for key in sorted(index)]
        require(all(index[(row["candidate_id"], row["panel"], row["repeat"])] ==
                    {key: value for key, value in row.items() if key != "first_repeat_complete_probe_reference"}
                    for row in rows), "Top-level trial content differs")
    groups = {panel: [group_result(candidate, panel,
        [row for row in rows if row["candidate_id"] == candidate and row["panel"] == panel])
        for candidate in range(128)] for panel in ("A", "B")}
    complete_groups = all(group["stable"] for panel in groups.values() for group in panel)
    row_complete = len(rows) == 768 and not incomplete_trials and not uncommitted_rows
    complete = row_complete and complete_groups and source.get("status") == "COMPLETE_NT_FULLBANK_EXACT_768"
    costs = cost_assessment(groups["A"], complete=complete)
    model_reports = {}
    for name in MODELS:
        path = run / "forecasts" / f"prediction-{name}.npz"
        require(sha(path) == metadata[name]["receipt"]["sha256"], "Forecast receipt mismatch")
        with np.load(path, allow_pickle=False) as stored:
            forecast = {key: stored[key].copy() for key in ("valid", "total_cost", "failure_steps")}
            require(stored["states"].shape == (128, 6, 10), "Forecast states mismatch")
        require(forecast["valid"].shape == (128,) and forecast["valid"].dtype == np.bool_, "Bad validity mask")
        model_reports[name] = model_assessment(metadata[name], forecast, groups["A"], costs, complete=complete)
    pairs = []
    for candidate in range(128):
        a, b = groups["A"][candidate], groups["B"][candidate]
        if not a["stable"] or not b["stable"]:
            continue
        base = run / "actual/N-T" / f"candidate-{candidate:03d}"
        with np.load(base / "A/repeat-0/diagnostic.npz", allow_pickle=False) as za, np.load(base / "B/repeat-0/diagnostic.npz", allow_pickle=False) as zb:
            count = min(5, a["intervals"], b["intervals"])
            changed = np.flatnonzero(np.any(za["applied_actions"][:count] != zb["applied_actions"][:count], axis=1)).tolist()
        pairs.append({"candidate_id": candidate, "A_H5_physical_valid": a["H5_physical_valid"],
            "A_ready_first_hit_by_H5": a["ready_prefix"], "A_ready_eligible": a["ready_eligible"],
            "B_physical_valid": b["physical_valid"], "B_ready_prefix": b["ready_prefix"],
            "B_ready_eligible": b["ready_eligible"], "B_first_ready_interval": b["first_ready_interval"],
            "compared_common_command_intervals": count,
            "different_applied_command_intervals_1based": [value + 1 for value in changed],
            "B_ready_after_H5": b["ready_eligible"] and not b["ready_prefix"],
            "A_post_ready_invalid_B_absorbing_ready": a["ready"] and not a["H5_physical_valid"] and b["ready_eligible"]})
    counters = {}
    for panel in ("A", "B"):
        stable = [group for group in groups[panel] if group["stable"]]
        counters[panel] = {"stable_group_count": len(stable),
            "physical_valid_ids": [group["candidate_id"] for group in stable if group["physical_valid"]],
            "ready_first_hit_ids": [group["candidate_id"] for group in stable if group["ready"]],
            "ready_prefix_first_hit_ids": [group["candidate_id"] for group in stable if group["ready_prefix"]],
            "ready_eligible_ids": [group["candidate_id"] for group in stable if group["ready_eligible"]],
            "missing_or_unstable_ids": [group["candidate_id"] for group in groups[panel] if not group["stable"]]}
    preservation = {key: source.get(key) for key in ("unchanged_source_run_files", "unchanged_original_files",
                    "unchanged_model_normalizer_files", "original_source_models_code_config_unchanged",
                    "future_engine_calls_during_prediction", "training_updates")}
    preservation["bound_original_file_count"] = len(entry.get("original_sha256", {}))
    preservation["bound_source_run_file_count"] = len(entry.get("source_run_sha256", {}))
    preservation["bound_model_normalizer_file_count"] = len(entry.get("model_and_normalizer_sha256", {}))
    return {"schema": "followup-nt-fullbank-saved-result-summary-v1",
        "status": "COMPLETE_768_STABLE_SAVED_EVIDENCE" if complete else "PARTIAL_OR_UNSTABLE_NO_FULL_BANK_CLAIM",
        "source_run_name": run.name, "source_run_status": source.get("status"),
        "bindings": {"summarizer_sha256": sha(__file__), "design_sha256": sha(config_path),
            "executed_full_config_sha256": sha(full_config_path),
            "entry_sha256": sha(entry_path), "prospective_forecast_seal_sha256": sha(seal_path),
            "run_receipt_file": source_path.name, "run_receipt_sha256": sha(source_path),
            "bank_file_sha256": sha(run / "bank.npz"), "candidate_bank_hash": entry["bank_hash"],
            "full_input_typed_sha256": entry["full_input_typed_sha256"],
            "original_manifest_sha256": design["original_manifest_sha256"],
            "model_and_normalizer_sha256": entry["model_and_normalizer_sha256"]},
        "dimensions": {"state_count": 1, "state": "N-T", "models": list(MODELS),
            "candidate_ids": 128, "unique_command_sequences": len({array_hash(row) for row in bank}),
            "planned_actual_trajectories": 768, "recorded_complete_trajectories": len(rows),
            "durable_trial_summary_count": durable_row_count,
            "recorded_complete_intervals": sum(row["intervals"] for row in rows),
            "stable_groups": sum(group["stable"] for panel in groups.values() for group in panel),
            "all_rows_complete": row_complete, "all_groups_exact_3_repeats": complete_groups},
        "incomplete_trials": incomplete_trials, "durable_uncommitted_rows": uncommitted_rows,
        "executor_active_cursor": {"key": source.get("active_unsealed_result_key"),
            "also_present_in_committed_rows": source.get("active_unsealed_result_key") is not None and
                tuple(source["active_unsealed_result_key"]) in {(row["candidate_id"], row["panel"], row["repeat"]) for row in rows},
            "scope": "Cursor alone is not a missing-result verdict; use committed keys and durable receipts."},
        "actual_candidate_results": groups,
        "actual_counters": counters, "cost_assessment_A_only": costs,
        "model_hypotheses": model_reports,
        "I_A_B_comparison": {"paired_stable_groups": len(pairs), "candidate_pairs": pairs,
            "B_full_ready_after_H5_ids": [pair["candidate_id"] for pair in pairs if pair["B_ready_after_H5"]],
            "prefix_first_hit_difference_ids": [pair["candidate_id"] for pair in pairs if pair["A_ready_first_hit_by_H5"] != pair["B_ready_prefix"]],
            "interpretation": "B full vs A is a joint current-pose supervisor, horizon and termination-convention contrast. No B H8 cost or A-forecast-versus-B error is computed."},
        "S": {"status": "UNRESOLVED_NO_EXTRA_BANK_EXECUTED", "conditional_extension": "NOT_RUN",
              "interpretation": "Even an empty complete base bank would not establish physical unreachability."},
        "source_preservation_reported_by_executor": preservation,
        "fresh_teacher": teacher_assessment(run, source),
        "local_evidence_sha256": files,
        "analysis_seconds": time.perf_counter() - started, "new_physics_calls": 0, "new_model_calls": 0,
        "limits": ["One posthoc S1 source snapshot; deterministic repeats are exact reproduction checks, not independent successes or population evidence.",
            "Full raw/full-probe equality verdicts are authenticated saved executor evidence; this NumPy-only tool does not decode or independently recompare complete raw dataclasses.",
            "Possible optimum and regret use only physically valid H5 finite-cost candidates; invalid/missing candidates never receive zero cost or regret.",
            "No original frozen result or denominator is replaced; no whole-policy handoff improvement or model-family superiority is established."]}


def self_test():
    def group(index, value, ready):
        return {"candidate_id": index, "stable": True, "rankable_A": True,
                "cost_L": value, "cost_U": value, "ready_eligible": ready}
    result = cost_assessment([group(0, 1., False), group(1, 2., True)], complete=True)
    require(result["C"]["status"] == "SUPPORTED_LOCAL_COST_READINESS_MISMATCH", "Strict C test failed")
    result = cost_assessment([group(0, 1., False), group(1, 1., True)], complete=True)
    require(result["possible_cost_optimum_ids"] == [0, 1] and result["C"]["status"].startswith("UNRESOLVED"), "Mixed C test failed")
    result = cost_assessment([group(0, 1., False)], complete=True)
    require(result["C"]["status"] == "UNRESOLVED_NO_READY_A", "No ready test failed")
    result = cost_assessment([group(0, 1., False), group(1, 2., True)], complete=False)
    require(result["C"]["status"] == "UNRESOLVED_INCOMPLETE_OR_UNSTABLE_BANK", "Partial C test failed")
    result = cost_assessment([], complete=False)
    require(result["epsilon"] is None and result["status"] == "NO_VALID_ACTUAL", "No valid test failed")
    print("PASS 5 cost-rule edge-case checks; no experiment/model/physics executed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        require(arguments.run is None and arguments.output is None, "Self-test cannot analyze a run")
        self_test()
    else:
        require(arguments.run is not None and arguments.output is not None, "--run and --output are required")
        value = summarize(arguments.run)
        with arguments.output.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": value["status"], "dimensions": value["dimensions"],
            "C": value["cost_assessment_A_only"]["C"], "analysis_seconds": value["analysis_seconds"]}, indent=2))
