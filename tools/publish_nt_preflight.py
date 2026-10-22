#!/usr/bin/env python3
"""Publish a portable N-T preflight subset and conservatively price a next run.

This tool uses the standard library only. It authenticates saved evidence and
never loads a model, steps physics, changes an original, or launches full128.
Complete raw, full inputs and complete first-repeat probes remain local.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("R", "S1-11", "S1-23", "S1-47", "SK-11", "SK-23", "SK-47")
IDS = (0, 9, 127)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def tree_bytes(folder):
    return sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())


def portable_path(path):
    return unicodedata.normalize("NFC", Path(path).as_posix())


def sanitization(roots):
    mappings = sorted(((unicodedata.normalize("NFC", str(path.resolve())), alias) for path, alias in roots),
                      key=lambda pair: len(pair[0]), reverse=True)
    substitutions = {}
    def text(value):
        if not isinstance(value, str):
            return value
        result = unicodedata.normalize("NFC", value)
        for prefix, alias in mappings:
            if prefix in result:
                result = result.replace(prefix, alias)
        if result != value:
            substitutions[value] = result
        return unicodedata.normalize("NFC", result)
    def transform(value):
        if isinstance(value, dict):
            return {text(key) if isinstance(key, str) else key: transform(item) for key, item in value.items()}
        if isinstance(value, list):
            return [transform(item) for item in value]
        return text(value)
    return transform, substitutions


def assert_portable(value):
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    require(not any(pattern in encoded for pattern in ("/Users/", "/private/var/", "/var/folders/", "file://")),
            "Private absolute path remains in public data")


def timing_from_row(run, row):
    trial = run / "actual/N-T" / f"candidate-{row['candidate_id']:03d}" / row["panel"] / f"repeat-{row['repeat']}"
    final = load(trial / "timing_final.json")["timings_ns"]
    require(final == row["timings_ns"], "Final timing receipt differs from summary row")
    require(final["trial_total_through_summary_ns"] >= final["trial_total_before_summary_ns"], "Invalid final trial time")
    return trial, final


def process_wall(run, path, summary):
    candidates = []
    if path is not None:
        candidates.append(path.resolve())
    elif (run / "process_wall_receipt.json").is_file():
        candidates.append(run / "process_wall_receipt.json")
    measured = summary["wall_seconds"]
    receipt = None
    if candidates:
        path = candidates[0]
        data = load(path)
        values = [data[key] for key in ("process_wall_seconds", "wall_seconds", "elapsed_seconds")
                  if key in data and type(data[key]) in (int, float)]
        require(values and all(math.isfinite(value) and value >= 0 for value in values), "Invalid external wall receipt")
        measured = max(measured, *values)
        receipt = {"path": path.name, "sha256": sha256(path), "record": data}
    return measured, receipt


def resource_projection(run, config, summary, measured_wall, trial_details, *, recovery_bytes=0):
    """Price fresh full768, with separate per-group persistence and fixed work."""
    limits = config["limits"]
    factor = limits["safety_factor"]
    rates, byte_rates, complete_seconds, complete_bytes, rows = [], [], [], [], []
    trial_total_ns = 0
    trial_bytes = 0
    for row, trial, timing in trial_details:
        intervals = row["intervals"]
        require(type(intervals) is int and intervals > 0, "Cannot price an empty physical trial")
        complete_ns = timing["first_repeat_complete_probe_durable_roundtrip_ns"]
        # Zero-repeat branch contains only the measured conditional check; no
        # complete evidence is removed from non-first repeat baseline costs.
        if row["repeat"] != 0:
            complete_ns = 0
        baseline_ns = timing["trial_total_through_summary_ns"] - complete_ns
        require(baseline_ns > 0, "Nonpositive baseline trial time")
        size = tree_bytes(trial)
        full_size = row["first_repeat_complete_probe_bytes"]
        require(0 <= full_size < size, "Invalid complete-probe byte partition")
        rate, byte_rate = baseline_ns / 1e9 / intervals, (size - full_size) / intervals
        rates.append(rate)
        byte_rates.append(byte_rate)
        if row["repeat"] == 0:
            complete_seconds.append(complete_ns / 1e9)
            complete_bytes.append(full_size)
        trial_total_ns += timing["trial_total_through_summary_ns"]
        trial_bytes += size
        rows.append({"candidate_id": row["candidate_id"], "panel": row["panel"], "repeat": row["repeat"],
                     "intervals": intervals, "trial_total_through_summary_seconds": timing["trial_total_through_summary_ns"] / 1e9,
                     "complete_probe_addition_seconds": complete_ns / 1e9,
                     "baseline_seconds_per_interval": rate, "trial_bytes": size,
                     "complete_probe_addition_bytes": full_size, "baseline_bytes_per_interval": byte_rate})
    require(len(complete_seconds) == 6, "All six complete-probe first repeats required")
    selected_rate = max(limits["prior_worst_seconds_per_interval"], max(rates))
    selected_byte_rate = max(limits["prior_worst_bytes_per_interval"], max(byte_rates))
    actual_stage_ns = summary["stages_ns"]["actual_preflight_trials_and_all_verification_ns"]
    residual_ns = actual_stage_ns - trial_total_ns
    require(residual_ns >= 0, "Trial-loop overhead partition is negative")
    residual_per_trial_s = residual_ns / 1e9 / 18
    teacher_inputs_s = summary["stages_ns"]["original_D2_teacher_input_authentication_ns"] / 1e9
    # Fixed-before already contains one80-row input authentication and the full
    # 640-row throughput benchmark. Seven more input-authentication blocks
    # conservatively price 640 fresh actual boundaries instead of80.
    fixed_measured_s = (summary["fixed_before_any_actual_seconds"] + summary["fixed_after_actual_seconds"]
                        + 7 * teacher_inputs_s)
    fixed_future_s = factor * fixed_measured_s
    report_reserve_s = limits["final_report_reserve_seconds"]
    baseline_future_s = factor * 4992 * selected_rate
    persistence_future_s = factor * 256 * max(complete_seconds)
    loop_future_s = factor * 768 * residual_per_trial_s
    future_s = baseline_future_s + persistence_future_s + loop_future_s + fixed_future_s + report_reserve_s
    spent_s = config["cumulative_prior_wall_seconds"] + measured_wall
    current_run_bytes = tree_bytes(run)
    fixed_current_bytes = current_run_bytes - trial_bytes
    require(fixed_current_bytes >= 0, "Negative fixed output bytes")
    baseline_future_bytes = math.ceil(factor * 4992 * selected_byte_rate)
    persistence_future_bytes = math.ceil(factor * 256 * max(complete_bytes))
    fixed_future_bytes = math.ceil(factor * fixed_current_bytes)
    future_bytes = baseline_future_bytes + persistence_future_bytes + fixed_future_bytes
    prior_bytes = config["existing_run_bytes_before_preflight"]
    total_bytes = prior_bytes + current_run_bytes + future_bytes
    free = shutil.disk_usage(run).free
    time_fit = spent_s + future_s <= limits["wall_time_cap_s"]
    storage_fit = total_bytes <= limits["incremental_storage_cap_bytes"]
    reserve_fit = free >= future_bytes + limits["free_space_reserve_bytes"]
    return {"schema": "followup-nt-fullbank-fresh-run-resource-projection-v1",
        "status": "RESOURCE_FIT_PENDING_FULL_EXECUTOR_ENTRY" if time_fit and storage_fit and reserve_fit else "NOT_RUN_RESOURCE_HOLD",
        "next_actual_execution": "NOT_RUN", "safety_factor": factor,
        "dimensions": {"states": 1, "candidates": 128, "models": list(MODELS), "panels": ["A", "B"],
            "physical_repeats": 3, "physical_trajectories": 768, "maximum_returned_intervals": 4992,
            "first_repeat_complete_probe_groups": 256, "D2_or_preflight_row_reuse": False,
            "conditional_extension": False},
        "measured_preflight_wall_seconds": summary["wall_seconds"],
        "charged_process_wall_seconds": measured_wall,
        "prior_experiment_process_wall_seconds": config["cumulative_prior_wall_seconds"],
        "cumulative_spent_wall_seconds": spent_s,
        "measured_fixed_before_actual_seconds": summary["fixed_before_any_actual_seconds"],
        "measured_fixed_after_actual_seconds": summary["fixed_after_actual_seconds"],
        "measured_teacher_input_authentication_80row_seconds": teacher_inputs_s,
        "extra_teacher_input_authentication_blocks": 7,
        "fixed_future_before_safety_seconds": fixed_measured_s,
        "full640_teacher_benchmark_already_in_measured_fixed": True,
        "prior_baseline_seconds_per_interval_floor": limits["prior_worst_seconds_per_interval"],
        "max_measured_baseline_seconds_per_interval": max(rates),
        "selected_baseline_seconds_per_interval": selected_rate,
        "max_first_repeat_complete_probe_addition_seconds": max(complete_seconds),
        "measured_actual_loop_seconds": actual_stage_ns / 1e9,
        "sum_measured_trial_through_summary_seconds": trial_total_ns / 1e9,
        "measured_loop_overhead_seconds": residual_ns / 1e9,
        "selected_loop_overhead_seconds_per_trial": residual_per_trial_s,
        "future_seconds": {"physical_raw_validation_baseline": baseline_future_s,
            "complete_probe_first_repeat_persistence_and_exact": persistence_future_s,
            "trial_loop_entry_timing_receipt_overhead": loop_future_s,
            "fixed_forecast_models_bindings_teacher_and_final_rehash": fixed_future_s,
            "unmeasured_report_and_shared_adapter_reserve": report_reserve_s,
            "total": future_s},
        "projected_cumulative_wall_seconds": spent_s + future_s,
        "wall_cap_seconds": limits["wall_time_cap_s"], "wall_cap_fit": time_fit,
        "prior_run_bytes": prior_bytes, "base_existing_run_bytes": config["existing_run_bytes_before_preflight"],
        "preserved_failed_preflight_bytes_already_in_prior": recovery_bytes, "entire_preflight_run_bytes": current_run_bytes,
        "measured_preflight_trial_tree_bytes": trial_bytes, "measured_preflight_nontrial_bytes": fixed_current_bytes,
        "prior_baseline_bytes_per_interval_floor": limits["prior_worst_bytes_per_interval"],
        "max_measured_baseline_bytes_per_interval": max(byte_rates),
        "selected_baseline_bytes_per_interval": selected_byte_rate,
        "max_first_repeat_complete_probe_addition_bytes": max(complete_bytes),
        "future_bytes": {"physical_raw_validation_baseline": baseline_future_bytes,
            "complete_probe_first_repeat_persistence": persistence_future_bytes,
            "fixed_nontrial_evidence": fixed_future_bytes, "total": future_bytes},
        "projected_cumulative_bytes": total_bytes, "storage_cap_bytes": limits["incremental_storage_cap_bytes"],
        "storage_cap_fit": storage_fit, "free_bytes_at_projection": free,
        "free_space_reserve_bytes": limits["free_space_reserve_bytes"], "free_space_reserve_fit": reserve_fit,
        "per_trial_rate_inputs": rows,
        "accounting": ["All fixed forecasting, loading, SHA checks and teacher640 benchmark costs remain included.",
            "Source-D2 comparison extra remains in baseline conservatively; no repeat/raw check is subtracted.",
            "New complete-probe first-repeat serialization/rehydration scales256 groups, not768 trials.",
            "Any larger external process-wall receipt is charged; human planning/calendar time is outside the experiment-process clock.",
            "120s report/shared-adapter reserve is a planning allowance, not a measured upper bound."],
        "limitations": ["Measured maxima plus1.5 safety are resource estimates, not universal execution-time bounds.",
            "New full mode is not implemented by this preflight-only driver; shared full adapter still needs identity and live-resource entry checks.",
            "No scientific rows reused; full128 physical outcomes and conditional extension remain unexecuted."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/followup_nt_preflight.v2.json")
    parser.add_argument("--process-wall-receipt", type=Path)
    args = parser.parse_args()
    run, output, config_path = args.run.resolve(), args.output.resolve(), args.config.resolve()
    require(run.is_dir() and not output.exists() and run != output and run not in output.parents,
            "Export requires an existing source run and exclusive separate output")
    original = run.parent.parent
    source_run = original / "runs/followup-near-goal-20261022-01"
    config, entry, summary = load(config_path), load(run / "entry.json"), load(run / "summary.json")
    require(summary["status"] == "PASS_NT_EXECUTOR_PREFLIGHT", "Preflight did not pass")
    require(entry["config_sha256"] == sha256(config_path), "Executed config changed")
    require(summary["model_ids"] == list(MODELS) and summary["forecast_candidate_count"] == 128, "Forecast model/bank dimensions differ")
    require(entry["full_input_typed_sha256"] == config["expected_input_typed_sha256"] and entry["bank_hash"] == config["expected_candidate_hash"], "Executed input or bank identity differs")
    require(sha256(run / "bank.npz") == entry["bank_receipt"]["sha256"], "Executed bank artifact changed")
    require(summary["full128_actual_execution"] == "NOT_RUN" and summary["actual_trajectories"] == 18,
            "Wrong execution scope")
    require(summary["actual_physics_intervals"] == sum(row["intervals"] for row in summary["rows"]), "Interval sum mismatch")
    expected = {(index, panel, repeat) for index in IDS for panel in ("A", "B") for repeat in range(3)}
    require({(row["candidate_id"], row["panel"], row["repeat"]) for row in summary["rows"]} == expected,
            "Preflight row Cartesian product incomplete")
    for relative, digest in entry["code_sha256"].items():
        require(sha256(ROOT / relative) == digest, "Execution code changed: " + relative)
    require(entry["model_and_normalizer_sha256"] == config["expected_model_normalizer_sha256"], "Loaded model file map differs from configured frozen map")
    for relative, digest in entry["model_and_normalizer_sha256"].items():
        require(sha256(ROOT / relative) == digest, "Model/normalizer changed: " + relative)
    for relative, digest in entry["original_sha256"].items():
        require(sha256(original / relative) == digest, "Original source changed: " + relative)
    for relative, digest in entry["source_run_sha256"].items():
        require(sha256(source_run / relative) == digest, "Source D1/D2 run changed: " + relative)
    require(sha256(ROOT / config["resource_ledger_path"]) == config["resource_ledger_sha256"], "Resource ledger changed")
    seal = load(run / "prospective_forecast_seal.json")
    require(seal["before_any_new_actual_future"] and all(v == 0 for v in seal["future_engine_calls_during_prediction"].values()),
            "Prospective forecast seal scope invalid")
    require(all(v == 0 for v in summary["future_engine_calls_during_prediction"].values()), "Prediction engine guard nonzero")
    for relative, digest in {**seal["forecasts"], **seal["teacher"]}.items():
        require(sha256(run / relative) == digest, "Prospectively frozen prediction/teacher file changed")
    details = [(row, *timing_from_row(run, row)) for row in summary["rows"]]
    measured_wall, wall_receipt = process_wall(run, args.process_wall_receipt, summary)
    recovery_records, recovery_bytes = [], 0
    for attempt in config.get("recovery_attempts", []):
        failure_path = original / attempt["failure_record_path"]
        require(sha256(failure_path) == attempt["failure_record_sha256"], "Preserved failure record changed")
        failure = load(failure_path)
        require(failure["wall_seconds"] == attempt["wall_seconds"] and
                failure["actual_physics_intervals_returned"] == attempt["actual_physics_intervals_returned"],
                "Preserved failure resource record differs")
        size = tree_bytes(failure_path.parent)
        require(size == attempt["preserved_run_bytes"], "Preserved failed-run byte ledger changed")
        recovery_bytes += size
        recovery_records.append({"run_id": attempt["run_id"], "failure_record_sha256": attempt["failure_record_sha256"],
            "failure": failure, "preserved_run_bytes": size,
            "wall_seconds_already_charged_in_config_prior": attempt["wall_seconds"],
            "time_charged_again": False, "reason": attempt["reason"]})
    require(config["cumulative_prior_wall_seconds"] == config.get("base_reviewed_prior_wall_seconds", config["cumulative_prior_wall_seconds"])
            + sum(attempt["wall_seconds"] for attempt in config.get("recovery_attempts", [])),
            "Recovery wall ledger is inconsistent")
    budget = resource_projection(run, config, summary, measured_wall, details, recovery_bytes=recovery_bytes)
    transform, substitutions = sanitization(((run, "local_run"), (source_run, "source_D1_D2_run"),
                                            (ROOT, "repository_root"), (original, "project_root"),
                                            *((original / "runs" / attempt["run_id"], "preserved_failed_run/" + attempt["run_id"])
                                              for attempt in config.get("recovery_attempts", []))))
    output.mkdir(parents=True, exist_ok=False)
    records = []
    def publish(relative, *, target=None, as_json=False):
        source, destination = run / relative, output / (target or relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256(source)
        if as_json:
            value = transform(load(source))
            assert_portable(value)
            dump(destination, value)
            action = "derived_json_paths_mapped_and_unicode_NFC"
        else:
            shutil.copyfile(source, destination)
            require(sha256(destination) == digest, "Array byte copy changed")
            action = "byte_identical"
        records.append({"source_run_path": portable_path(relative), "source_sha256": digest,
            "public_path": portable_path(destination.relative_to(output)), "public_sha256": sha256(destination),
            "bytes": destination.stat().st_size, "transformation": action})
    for relative in ("bank.npz",):
        publish(relative)
    for relative in ("entry.json", "summary.json", "prospective_forecast_seal.json"):
        publish(relative, as_json=True)
    for folder in ("forecasts", "teacher"):
        for source in sorted((run / folder).iterdir()):
            if source.is_file():
                publish(source.relative_to(run), as_json=source.suffix == ".json")
    inventories = []
    for row, trial, timing in details:
        relative = trial.relative_to(run)
        require(row["raw_roundtrip"]["status"] == "PASS_EXACT" and row["initial_snapshot_exact"], "Raw preflight verdict failed")
        require(row["full_probe_repeat_exact"] is None if row["repeat"] == 0 else row["full_probe_repeat_exact"]["status"] == "PASS_EXACT",
                "Full-probe repeat verdict failed")
        for name in ("entry.json", "summary.json", "timing_final.json", "raw_receipt.json"):
            publish(relative / name, as_json=True)
        raw = load(trial / "raw_receipt.json")
        raw_files = []
        for path, digest in raw["files_sha256"].items():
            path = Path(path)
            require(path.is_relative_to(run) and sha256(path) == digest, "Raw inventory file changed")
            raw_files.append({"path": portable_path(path.relative_to(run)), "sha256": digest,
                              "bytes": path.stat().st_size, "included_in_public_repository": False})
        require(len(raw_files) == 2, "Expected one complete raw pair")
        complete_files = []
        if row["repeat"] == 0:
            require(row["full_probe_roundtrip"]["status"] == "PASS_EXACT", "Complete probe rehydration failed")
            complete = load(trial / "complete_probe_receipt.json")
            for path, digest in complete["files_sha256"].items():
                path = Path(path)
                require(path.is_relative_to(run) and sha256(path) == digest, "Complete probe inventory changed")
                complete_files.append({"path": portable_path(path.relative_to(run)), "sha256": digest,
                                       "bytes": path.stat().st_size, "included_in_public_repository": False})
            publish(relative / "complete_probe_receipt.json", as_json=True)
            publish(relative / "diagnostic.npz")
        inventories.append({"state": "N-T", "candidate_id": row["candidate_id"], "panel": row["panel"],
            "repeat": row["repeat"], "whole_probe_typed_sha256": row["whole_probe_typed_sha256"],
            "raw_files": raw_files, "complete_probe_files": complete_files})
    input_inventory = []
    for path, digest in entry["full_input_receipt"]["files_sha256"].items():
        path = Path(path)
        require(path.is_relative_to(run) and sha256(path) == digest, "Local full-input file changed")
        input_inventory.append({"path": portable_path(path.relative_to(run)), "sha256": digest,
                                "bytes": path.stat().st_size, "included_in_public_repository": False})
    inventory_record = {"schema": "followup-nt-preflight-local-evidence-inventory-v1",
        "scope": "Hash and byte inventory only. All18 complete raw pairs, six first-repeat complete probes and fulltyped input remain local and are omitted from this public subset.",
        "physical_trajectories": 18, "raw_pair_count": 18, "complete_probe_group_count": 6,
        "records": inventories, "full_input_files": input_inventory}
    dump(output / "local_evidence_inventory.json", inventory_record)
    dump(output / "resource_projection.json", budget)
    if recovery_records:
        recovery_public = transform({"schema": "followup-nt-preserved-recovery-attempts-v1", "records": recovery_records,
            "scope": "Failed attempts preserved separately; their wall is already in config prior and charged only once. Their actual directory byte counts are verified and already included in config prior storage; no duplicate charge."})
        assert_portable(recovery_public)
        dump(output / "preserved_recovery_attempts.json", recovery_public)
    if wall_receipt is not None:
        portable_wall = transform(wall_receipt)
        assert_portable(portable_wall)
        dump(output / "process_wall_receipt.json", portable_wall)
    mapping_record = {"schema": "followup-nt-preflight-public-path-mapping-v1",
        "scope": "JSON source files are derived path-portable views, not byte-identical copies. Array files are exact copies. Original JSON SHA values remain in manifest copy records.",
        "prefixes": {"local_run": "new N-T preflight run root", "source_D1_D2_run": "preserved D1/D2 source run root",
                     "repository_root": "public repository root", "project_root": "original local project root"},
        "substitution_count": len(substitutions), "private_original_paths_not_published": True}
    dump(output / "path_mapping.json", mapping_record)
    for path in output.rglob("*.json"):
        assert_portable(load(path))
    files = [{"path": portable_path(path.relative_to(output)), "sha256": sha256(path), "bytes": path.stat().st_size}
             for path in sorted(output.rglob("*")) if path.is_file()]
    manifest = {"schema": "followup-nt-preflight-selected-public-evidence-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_run_relative_path": "runs/" + run.name,
        "status": "PASS_NT_EXECUTOR_PREFLIGHT", "full128_actual_execution": "NOT_RUN",
        "scope": "Complete128-bank seven-model forecasts, original-D2 teacher inputs and640-row repeated throughput benchmark,18-trial preflight receipts and six first-repeat boundary diagnostics. No physics, inference, training or full batch executed by publication.",
        "source_entry_sha256": sha256(run / "entry.json"), "source_summary_sha256": sha256(run / "summary.json"),
        "executed_config_sha256": sha256(config_path), "publisher_sha256": sha256(Path(__file__)),
        "source_copy_records": records, "files": files, "files_excluding_self": len(files),
        "bytes_excluding_self": sum(item["bytes"] for item in files),
        "public_limits": "Raw/fullprobe/input hashes identify omitted local evidence; public files alone cannot independently reread those complete local payloads.",
        "resource_status": budget["status"]}
    dump(output / "manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "public_files": len(files),
        "public_bytes": manifest["bytes_excluding_self"], "resource_status": budget["status"],
        "projected_cumulative_wall_seconds": budget["projected_cumulative_wall_seconds"],
        "projected_cumulative_bytes": budget["projected_cumulative_bytes"]}))


if __name__ == "__main__":
    main()
