#!/usr/bin/env python3
"""Separate original-D2-reference comparison cost by its actual scope.

The immutable v1 projection intentionally extrapolated that comparison to all
768 fresh trials. Original D2 has candidate IDs0..15 only. This separate v2
budget charges all96 applicable comparisons, preserves every fresh raw/repeat
check, and makes no change to measured files, safety factors or experiment
scope. No physics, model inference, training or full execution occurs here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["R", "S1-11", "S1-23", "S1-47", "SK-11", "SK-23", "SK-47"]


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
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


def calculate(config, summary, v1):
    """Pure arithmetic: baseline all4992 intervals; reference check96 trials."""
    limits = config["limits"]
    require(limits["safety_factor"] == 1.5 and limits["wall_time_cap_s"] == 7200,
            "Safety factor or cumulative wall cap changed")
    require(limits["prior_worst_seconds_per_interval"] == 0.48116094996221365,
            "Reviewed prior baseline floor changed")
    factor = limits["safety_factor"]
    rows = summary["rows"]
    expected = {(index, panel, repeat) for index in (0, 9, 127)
                for panel in ("A", "B") for repeat in range(3)}
    require(len(rows) == 18 and {(r["candidate_id"], r["panel"], r["repeat"]) for r in rows} == expected,
            "Preflight Cartesian product differs")
    require(v1["dimensions"] == {"states": 1, "candidates": 128, "models": MODELS, "panels": ["A", "B"],
        "physical_repeats": 3, "physical_trajectories": 768, "maximum_returned_intervals": 4992,
        "first_repeat_complete_probe_groups": 256, "D2_or_preflight_row_reuse": False, "conditional_extension": False},
        "Immutable v1 prospective dimensions changed")
    rates, applicable_costs, inputs = [], [], []
    full_persistence_seconds = []
    for row in rows:
        timing = row["timings_ns"]
        total_ns = timing["trial_total_through_summary_ns"]
        first_persistence_ns = timing["first_repeat_complete_probe_durable_roundtrip_ns"] if row["repeat"] == 0 else 0
        applicable = 0 <= row["candidate_id"] < 16
        comparison_ns = timing["source_D2_raw_read_and_exact_ns"] if applicable else 0
        require(type(row["intervals"]) is int and row["intervals"] > 0, "Cannot price zero intervals")
        require(total_ns > first_persistence_ns + comparison_ns >= 0, "Timing partition invalid")
        require(row["raw_roundtrip"]["status"] == "PASS_EXACT", "Fresh raw exact check failed")
        if row["repeat"] == 0:
            require(row["full_probe_roundtrip"]["status"] == "PASS_EXACT", "First complete-probe check failed")
            full_persistence_seconds.append(first_persistence_ns / 1e9)
        else:
            require(row["full_probe_repeat_exact"]["status"] == "PASS_EXACT", "Fresh full-probe repeat check failed")
        if applicable:
            require(row["source_D2_raw_exact"]["status"] == "PASS_EXACT", "Applicable original-D2 comparison failed")
            applicable_costs.append(comparison_ns / 1e9)
        else:
            require(row["source_D2_raw_exact"] is None, "Candidate127 must have no original-D2 future")
        baseline_ns = total_ns - first_persistence_ns - comparison_ns
        rate = baseline_ns / 1e9 / row["intervals"]
        rates.append(rate)
        inputs.append({"candidate_id": row["candidate_id"], "panel": row["panel"], "repeat": row["repeat"],
            "intervals": row["intervals"], "total_through_summary_ns": total_ns,
            "first_repeat_complete_probe_ns": first_persistence_ns,
            "source_D2_comparison_applicable": applicable, "source_D2_comparison_ns_separated": comparison_ns,
            "all_fresh_evidence_baseline_ns": baseline_ns, "all_fresh_evidence_baseline_seconds_per_interval": rate})
    require(len(applicable_costs) == 12 and len(full_persistence_seconds) == 6,
            "Expected12 measured original-reference cases and6 first-repeat complete probes")
    selected_rate = max(limits["prior_worst_seconds_per_interval"], max(rates))
    source_comparison_s = factor * 96 * max(applicable_costs)
    baseline_s = factor * 4992 * selected_rate
    full_persistence_s = factor * 256 * max(full_persistence_seconds)
    require(math.isclose(full_persistence_s,
        v1["future_seconds"]["complete_probe_first_repeat_persistence_and_exact"], rel_tol=0, abs_tol=1e-9),
        "Complete-probe persistence term changed")
    preserved = {key: v1["future_seconds"][key] for key in (
        "trial_loop_entry_timing_receipt_overhead", "fixed_forecast_models_bindings_teacher_and_final_rehash",
        "unmeasured_report_and_shared_adapter_reserve")}
    require(preserved["unmeasured_report_and_shared_adapter_reserve"] == 120,
            "Unmeasured report/adapter reserve changed")
    teacher_input_s = summary["stages_ns"]["original_D2_teacher_input_authentication_ns"] / 1e9
    fixed_recomputed = factor * (summary["fixed_before_any_actual_seconds"] + summary["fixed_after_actual_seconds"] + 7 * teacher_input_s)
    require(math.isclose(fixed_recomputed, preserved["fixed_forecast_models_bindings_teacher_and_final_rehash"], rel_tol=0, abs_tol=1e-9),
        "Fixed forecast/model/source/teacher term differs from measured8-block accounting")
    total_trial_ns = sum(row["timings_ns"]["trial_total_through_summary_ns"] for row in rows)
    loop_residual_ns = summary["stages_ns"]["actual_preflight_trials_and_all_verification_ns"] - total_trial_ns
    require(loop_residual_ns >= 0, "Negative loop overhead")
    loop_recomputed = factor * 768 * (loop_residual_ns / 1e9 / 18)
    require(math.isclose(loop_recomputed, preserved["trial_loop_entry_timing_receipt_overhead"], rel_tol=0, abs_tol=1e-9),
        "Loop overhead differs from immutable v1")
    components = {"physical_raw_validation_baseline_all4992_intervals": baseline_s,
        "original_D2_reference_read_and_exact_all96_applicable_trials": source_comparison_s,
        "complete_probe_first_repeat_persistence_and_exact_all256_groups": full_persistence_s, **preserved}
    future_s = math.fsum(components.values())
    spent_s = v1["cumulative_spent_wall_seconds"]
    require(math.isclose(spent_s, config["cumulative_prior_wall_seconds"] + v1["charged_process_wall_seconds"], rel_tol=0, abs_tol=1e-9),
        "Cumulative prior/preflight clock changed")
    projected_s = spent_s + future_s
    time_fit = projected_s <= limits["wall_time_cap_s"]
    storage_fit = v1["storage_cap_fit"] and v1["free_space_reserve_fit"]
    return {"schema": "followup-nt-fullbank-resource-budget-v2",
        "status": "RESOURCE_FIT_CONDITIONAL_ON_SCOPED_FULL_ADAPTER" if time_fit and storage_fit else "NOT_RUN_RESOURCE_HOLD",
        "full128_actual_execution": "NOT_RUN", "next_actual_execution": "NOT_RUN", "shared_full_adapter": "NOT_IMPLEMENTED_ENTRY_REQUIRED",
        "interpretation": "Scope-correct resource estimate. v1 blanket upper projection remains unchanged; this is not a measured speedup or removal of validation.",
        "dimensions": {**v1["dimensions"], "original_D2_reference_candidate_ids": list(range(16)),
            "original_D2_reference_trials": 96, "reference_check_measured_preflight_trials": 12},
        "comparison_with_v1": {"v1_status": v1["status"],
            "v1_blanket_projected_cumulative_wall_seconds": v1["projected_cumulative_wall_seconds"],
            "v1_blanket_baseline_seconds_per_interval": v1["selected_baseline_seconds_per_interval"],
            "v2_scoped_projected_cumulative_wall_seconds": projected_s,
            "projection_difference_seconds": v1["projected_cumulative_wall_seconds"] - projected_s,
            "reason": "v1 includes original-D2 reference comparison inside a worst interval rate extrapolated to768 trials; v2 retains it explicitly for all96 trials whose original references exist."},
        "safety_factor": factor, "prior_baseline_seconds_per_interval_floor": limits["prior_worst_seconds_per_interval"],
        "max_measured_baseline_without_source_reference_seconds_per_interval": max(rates),
        "selected_baseline_seconds_per_interval": selected_rate,
        "max_measured_original_D2_reference_seconds_per_trial": max(applicable_costs),
        "max_source_D2_comparison_seconds_per_trial": max(applicable_costs),
        "source_D2_reference_comparison_count": 96,
        "max_measured_first_complete_probe_seconds_per_group": max(full_persistence_seconds),
        "cumulative_spent_wall_seconds": spent_s, "prior_experiment_process_wall_seconds": config["cumulative_prior_wall_seconds"],
        "charged_current_process_wall_seconds": v1["charged_process_wall_seconds"],
        "future_seconds": {**components, "total": future_s},
        "projected_cumulative_wall_seconds": projected_s,
        "projected_cumulative_minutes": projected_s / 60,
        "wall_cap_seconds": limits["wall_time_cap_s"], "wall_cap_fit": time_fit,
        "wall_headroom_seconds": limits["wall_time_cap_s"] - projected_s,
        "projected_cumulative_bytes": v1["projected_cumulative_bytes"],
        "storage_cap_bytes": v1["storage_cap_bytes"], "storage_cap_fit": v1["storage_cap_fit"],
        "free_space_reserve_fit": v1["free_space_reserve_fit"],
        "storage_from_unchanged_v1": {key: v1[key] for key in (
            "prior_run_bytes", "base_existing_run_bytes", "preserved_failed_preflight_bytes_already_in_prior",
            "entire_preflight_run_bytes", "future_bytes", "projected_cumulative_bytes", "storage_cap_bytes", "storage_cap_fit",
            "free_bytes_at_projection", "free_space_reserve_bytes", "free_space_reserve_fit")},
        "component_proof": {"future_component_sum_seconds": math.fsum(components.values()),
            "reported_future_seconds": future_s, "cumulative_component_sum_seconds": spent_s + future_s,
            "reported_projected_cumulative_wall_seconds": projected_s,
            "equalities_exact": future_s == math.fsum(components.values()) and projected_s == spent_s + future_s,
            "fixed_includes_8_teacher_input_blocks": True, "teacher640_model_benchmark_still_in_fixed": True,
            "first_persistence_scales256_not768": True, "all768_fresh_trials_still_required": True},
        "required_execution_policy": [
            "Freshly execute all128 original candidates A/B with3 physical repeats:768 trajectories, at most4992 intervals; no D2/preflight row reuse.",
            "For all original IDs0..15, both panels and all3 repeats, read the authenticated original-D2 raw and compare the fresh complete raw exactly:96 comparisons.",
            "For IDs16..127 there is no source-D2 raw future to compare; retain all fresh raw fsync/read/hash/schema/bit-exact roundtrip checks and whole FollowupProbe repeat checks.",
            "Persist and rehydrate complete first-repeat FollowupProbe fields for all256 candidate-panel groups; retain whole-probe typed hashes and durable diagnostic arrays.",
            "Use the exact7 models and same full128 candidate/input hashes; freeze model forecasts before physical futures; original data/config/weights remain unchanged.",
            "Full adapter must pass identity, scope, live-resource and interruption/prefix-preservation checks before execution. Current preflight driver has no full mode.",
            "Keep1.5 safety, reviewed prior floor, all cumulative elapsed costs and120s unmeasured report/adapter reserve. No conditional extension."],
        "limits": ["Measured maxima plus safety remain estimates, not a universal runtime upper bound.",
            "Resource fit is conditional on the explicit96-reference-check policy and must be rechecked with current free space at full-adapter entry.",
            "No scientific success, model/cost superiority or full128 actual result follows from this accounting refinement."],
        "per_trial_scope_partition": inputs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-preflight", type=Path, default=ROOT / "artifacts/followup/20261022-nt-preflight")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/followup_nt_preflight.v2.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/validation/nt_fullbank_resource_budget.v2.json")
    args = parser.parse_args()
    public, config_path, output = args.public_preflight.resolve(), args.config.resolve(), args.output.resolve()
    require(not output.exists() and public not in output.parents, "New independent output required; immutable v1 artifact must not be edited")
    manifest_path = public / "manifest.json"
    manifest, config = load(manifest_path), load(config_path)
    bindings = {"public_manifest_sha256": sha256(manifest_path), "config_sha256": sha256(config_path)}
    require(manifest["executed_config_sha256"] == bindings["config_sha256"], "Executed config SHA mismatch")
    for record in manifest["files"]:
        require(sha256(public / record["path"]) == record["sha256"], "Immutable v1 public artifact changed: " + record["path"])
    entry, summary, v1 = (load(public / name) for name in ("entry.json", "summary.json", "resource_projection.json"))
    require(summary["status"] == "PASS_NT_EXECUTOR_PREFLIGHT" and summary["model_ids"] == MODELS,
            "Preflight status/models invalid")
    require(entry["config_sha256"] == bindings["config_sha256"], "Entry config differs")
    require(entry["bank_hash"] == config["expected_candidate_hash"] and entry["full_input_typed_sha256"] == config["expected_input_typed_sha256"],
            "Input/bank identity differs")
    for relative, digest in entry["code_sha256"].items():
        require(sha256(ROOT / relative) == digest, "Executed code changed: " + relative)
    bindings.update({"public_entry_sha256": sha256(public / "entry.json"),
        "public_summary_sha256": sha256(public / "summary.json"),
        "immutable_v1_resource_projection_sha256": sha256(public / "resource_projection.json"),
        "executed_driver_sha256": entry["code_sha256"]["scripts/followup_nt_executor.py"],
        "refinement_tool_sha256": sha256(Path(__file__))})
    result = calculate(config, summary, v1)
    result["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["bindings"] = bindings
    result["source_paths"] = {"immutable_public_preflight": public.relative_to(ROOT).as_posix(),
        "executed_config": config_path.relative_to(ROOT).as_posix(),
        "driver": "scripts/followup_nt_executor.py", "refinement_tool": "tools/refine_nt_resource_scope.py"}
    # Recheck immutable files after arithmetic, before writing this new receipt.
    require(sha256(manifest_path) == bindings["public_manifest_sha256"], "v1 manifest changed during arithmetic")
    for record in manifest["files"]:
        require(sha256(public / record["path"]) == record["sha256"], "v1 artifact changed during arithmetic")
    require(sha256(config_path) == bindings["config_sha256"], "Config changed during arithmetic")
    dump(output, result)
    print(json.dumps({key: result[key] for key in ("status", "projected_cumulative_wall_seconds",
        "projected_cumulative_minutes", "wall_cap_fit", "wall_headroom_seconds")}))


if __name__ == "__main__":
    main()
