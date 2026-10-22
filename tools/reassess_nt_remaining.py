#!/usr/bin/env python3
"""Pure, conservative resource reassessment; never run physics or predictors.

The original STOP_RESOURCE is immutable. This tool assesses the missing Cartesian
keys using authenticated saved records and only raises the historical rates.
The current profile's outer process clock is charged once; its inclusive and
exclusive child timers explain that clock and are never added to it.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

PROGRAM_START = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
FULL_RUN = ROOT.parents[1] / "runs/followup-nt-full-20261022-01"
PUBLIC_BASE = ROOT / "artifacts/followup/20261022-nt-full"
DELIVERY = ROOT / "artifacts/validation/nt_full_delivery_review.v1.json"
PRIOR_SECONDS = 1736.5930288732052
FIXED_SHA = {
    "summary.json": "8f45091686dc3b6a8c0b45b48d5036b303a0fa8c1b459c479dcb4e1ed1970df7",
    "entry.json": "a06fe85be6f2b7a54947f9338d3433ab2f91bf8e91c462666c90d78e4d482889",
    "config": "6c610918926f076cc09e950512252187a071147cede367e473bc5a2e9920b58b",
    "driver": "021faeff2e5a46cc3d0ece5b308ee729153b66c81f266efd23d54484079c645a",
    "delivery": "813a2b143e92d79589f40056d23d5a145f9f454a007fee35750f2348d8346571",
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def nonnegative(value, name):
    need(type(value) in (int, float) and math.isfinite(value) and value >= 0,
         "Invalid nonnegative quantity: " + name)
    return value


def remaining_work(completed):
    expected = {(i, panel, repeat) for i in range(128) for panel in ("A", "B") for repeat in range(3)}
    keys = list(completed)
    need(len(keys) == len(set(keys)) and set(keys) <= expected, "Duplicate or out-of-design completed keys")
    missing = expected - set(keys)
    work = {"trials": len(missing), "intervals": sum(5 if p == "A" else 8 for _, p, _ in missing),
            "source_comparisons": sum(i < 16 for i, _, _ in missing),
            "complete_probe_groups": sum(repeat == 0 for _, _, repeat in missing)}
    return work, [{"candidate_id": i, "panel": p, "repeat": r} for i, p, r in sorted(missing)]


def assess(*, prior_seconds, process_seconds, elapsed_postwork_seconds, work, rates,
           preserved_bytes, free_bytes, profile_bundle_seconds=0., profile_bundle_bytes=0.,
           assessment_seconds=0., wall_cap_seconds=7200., safety_factor=1.5,
           final_report_reserve_seconds=600., sealing_reserve_seconds=60.,
           storage_cap_bytes=5*1024**3, free_reserve_bytes=2*1024**3,
           sealing_reserve_bytes=16*1024**2, fixed_future_seconds=12.124927558461845,
           fixed_future_bytes=1988750):
    """Calculate only; arguments represent mutually exclusive outer clock scopes."""
    for name, value in locals().copy().items():
        if name not in {"work", "rates"}:
            nonnegative(value, name)
    for name, value in work.items():
        need(type(value) is int and value >= 0, "Invalid remaining-work count: " + name)
    for name, value in rates.items():
        nonnegative(value, name)
    spent = prior_seconds + process_seconds + elapsed_postwork_seconds + assessment_seconds
    base = dict(rates)
    updated = dict(base)
    updated["complete_probe_seconds"] = max(base["complete_probe_seconds"], profile_bundle_seconds)
    updated["complete_probe_bytes"] = max(base["complete_probe_bytes"], profile_bundle_bytes)

    def scenario(unit_rates):
        seconds = {
            "interval_baseline": safety_factor*work["intervals"]*unit_rates["seconds_per_interval"],
            "source_D2_comparisons": safety_factor*work["source_comparisons"]*unit_rates["source_comparison_seconds"],
            "first_repeat_complete_probe": safety_factor*work["complete_probe_groups"]*unit_rates["complete_probe_seconds"],
            "trial_loop": safety_factor*work["trials"]*unit_rates["loop_seconds_per_trial"],
            "fixed_future": fixed_future_seconds,
            "report_reserve": final_report_reserve_seconds,
        }
        byte_parts = {
            "interval_baseline": math.ceil(safety_factor*work["intervals"]*unit_rates["bytes_per_interval"]),
            "first_repeat_complete_probe": math.ceil(safety_factor*work["complete_probe_groups"]*unit_rates["complete_probe_bytes"]),
            "fixed_future": fixed_future_bytes,
        }
        future_seconds, future_bytes = sum(seconds.values()), sum(byte_parts.values())
        projected_seconds, projected_bytes = spent+future_seconds, preserved_bytes+future_bytes
        failures = []
        if projected_seconds > wall_cap_seconds:
            failures.append("cumulative_wall_cap")
        if projected_bytes > storage_cap_bytes:
            failures.append("cumulative_storage_cap")
        if free_bytes < free_reserve_bytes+future_bytes:
            failures.append("free_space_reserve")
        non_bundle = spent+future_seconds-seconds["first_repeat_complete_probe"]
        divisor = safety_factor*work["complete_probe_groups"]
        maximum_bundle = (wall_cap_seconds-non_bundle)/divisor if divisor else None
        return {"status": "HOLD_RESOURCE" if failures else "FITS_ESTIMATE_ONLY_NOT_AUTHORIZATION",
                "failures": failures, "unit_rates": unit_rates, "future_seconds": seconds,
                "future_bytes": byte_parts, "projected_cumulative_seconds": projected_seconds,
                "projected_cumulative_minutes": projected_seconds/60,
                "projected_cumulative_bytes": projected_bytes,
                "wall_margin_seconds": wall_cap_seconds-projected_seconds,
                "maximum_bundle_seconds_that_would_fit_if_other_costs_fixed": maximum_bundle,
                "crossover_is_rate_target_not_permission_to_lower_observed_maximum": True}

    return {
        "time_accounting": {"prior_delivery_conservative_seconds": prior_seconds,
            "current_profile_outer_process_seconds": process_seconds,
            "nonoverlapping_current_task_additional_work_seconds": elapsed_postwork_seconds,
            "this_assessment_elapsed_seconds": assessment_seconds, "cumulative_spent_seconds": spent,
            "clock_policy": "Sum these four non-overlapping scopes once. Nested trial/stage/exclusive timers are explanatory subsets, never added."},
        "remaining": work, "frozen_rate_scenario": scenario(base), "profile_only_raise_scenario": scenario(updated),
        "limits": {"wall_cap_seconds": wall_cap_seconds, "safety_factor": safety_factor,
            "final_report_reserve_seconds": final_report_reserve_seconds,
            "sealing_reserve_seconds": sealing_reserve_seconds, "storage_cap_bytes": storage_cap_bytes,
            "free_reserve_bytes": free_reserve_bytes, "sealing_reserve_bytes": sealing_reserve_bytes},
        "step_guard_envelope": {"needed_seconds": safety_factor*updated["seconds_per_interval"]+sealing_reserve_seconds+final_report_reserve_seconds,
            "fits_now": spent+safety_factor*updated["seconds_per_interval"]+sealing_reserve_seconds+final_report_reserve_seconds <= wall_cap_seconds,
            "scope": "Original separate returned-step sealing/report reserve; not added again to the whole-workflow forecast."},
        "authorization": "NONE: original STOP_RESOURCE unchanged; no continuation, retry or new physics authorized by this arithmetic.",
    }


def authenticate_frozen(full_run=FULL_RUN):
    full_run = Path(full_run).resolve()
    for name in ("summary.json", "entry.json"):
        need(sha(full_run/name) == FIXED_SHA[name], "Frozen full-run identity changed: " + name)
    config_path = ROOT/"configs/followup_nt_full.v1.json"
    need(sha(config_path) == FIXED_SHA["config"], "Executed config changed")
    need(sha(ROOT/"scripts/followup_nt_full.py") == FIXED_SHA["driver"], "Executed driver changed")
    need(sha(DELIVERY) == FIXED_SHA["delivery"], "Prior delivery ledger changed")
    summary, entry, config, ledger = load(full_run/"summary.json"), load(full_run/"entry.json"), load(config_path), load(DELIVERY)
    need(ledger["time_accounting"]["conservative_total_including_publication_reserve_seconds"] == PRIOR_SECONDS, "Prior delivery clock differs")
    need(summary["status"] == "STOP_RESOURCE" and summary["completed_trajectories"] == 70, "Frozen partial status changed")
    need(entry["config_sha256"] == FIXED_SHA["config"], "Entry/config identity mismatch")
    for relative, digest in entry["code_sha256"].items():
        need(sha(ROOT/relative) == digest, "Frozen execution source changed: " + relative)
    manifest = load(PUBLIC_BASE/"manifest.json")
    need(manifest["source_summary_sha256"] == FIXED_SHA["summary.json"] and manifest["source_entry_sha256"] == FIXED_SHA["entry.json"], "Published source identities differ")
    copies = {r["source_path"]: r for r in manifest["source_copy_records"] if r["source_scope"] == "run"}
    row_hashes = {}
    keys = []
    for row in summary["rows"]:
        key = (row["candidate_id"], row["panel"], row["repeat"])
        keys.append(key)
        folder = f"actual/N-T/candidate-{key[0]:03d}/{key[1]}/repeat-{key[2]}"
        for filename in ("summary.json", "timing_final.json"):
            relative = folder+"/"+filename
            digest = sha(full_run/relative)
            need(digest == copies[relative]["source_sha256"], "Frozen row file hash changed: " + relative)
            row_hashes[relative] = digest
        saved = load(full_run/folder/"summary.json")
        need({k: v for k, v in saved.items() if k != "timings_ns"} == {k: v for k, v in row.items() if k != "timings_ns"}, "Saved row differs from bound full summary")
        final_timings = load(full_run/folder/"timing_final.json")["timings_ns"]
        need(final_timings == row["timings_ns"] and all(final_timings[k] == v for k, v in saved["timings_ns"].items()), "Saved row timing mismatch")
    work, missing = remaining_work(keys)
    need(work == {"trials": 698, "intervals": 4540, "source_comparisons": 26, "complete_probe_groups": 232}, "Unexpected frozen remaining counts")
    need(missing == summary["missing_row_keys"], "Missing Cartesian keys differ")
    need(work == summary["final_resource_decision"]["remaining"], "Live-decision remaining counts differ")
    return summary, config, {"full_run_summary_sha256": FIXED_SHA["summary.json"], "full_run_entry_sha256": FIXED_SHA["entry.json"],
        "executed_config_sha256": FIXED_SHA["config"], "executed_driver_sha256": FIXED_SHA["driver"],
        "prior_delivery_sha256": FIXED_SHA["delivery"], "frozen_code_sha256": entry["code_sha256"], "saved_row_file_sha256": row_hashes}, work


def profile_maxima(profile):
    """Use one inclusive bundle per trial, never sum nested child timers."""
    timed = profile.get("trials", profile.get("records", []))
    calibration = profile.get("reference_calibration", [])
    need(isinstance(timed, list) and timed and isinstance(calibration, list), "Profile needs nonempty trials/records and a calibration list")
    trials = timed+calibration
    values = []
    for trial in trials:
        total = trial.get("total_ns", trial.get("bundle_total_ns"))
        byte_count = trial.get("output_bytes", trial.get("first_repeat_complete_probe_bytes"))
        nonnegative(total, "profile bundle total_ns")
        nonnegative(byte_count, "profile bundle output_bytes")
        need(type(total) is int and type(byte_count) is int, "Profile totals must be integer ns/bytes")
        values.append((total/1e9, byte_count))
    return max(v[0] for v in values), max(v[1] for v in values), len(values)


def logical_bytes(folder):
    return sum(p.stat().st_size for p in Path(folder).rglob("*") if p.is_file())


def self_test():
    completed = [(i, p, r) for i in range(11) for p in ("A", "B") for r in range(3)]
    completed += [(11, "A", r) for r in range(3)]+[(11, "B", 0)]
    work, missing = remaining_work(completed)
    need(work == {"trials": 698, "intervals": 4540, "source_comparisons": 26, "complete_probe_groups": 232} and len(missing) == 698, "Remaining-work self-test")
    rates = {"seconds_per_interval": .51612885, "bytes_per_interval": 89074.4,
        "source_comparison_seconds": 1.35706625, "complete_probe_seconds": 6.1589925,
        "complete_probe_bytes": 1036225, "loop_seconds_per_trial": .026950667}
    args = dict(prior_seconds=PRIOR_SECONDS, process_seconds=200., elapsed_postwork_seconds=100.,
        assessment_seconds=10., work=work, rates=rates, preserved_bytes=150000000, free_bytes=9*1024**3)
    a = assess(**args, profile_bundle_seconds=1., profile_bundle_bytes=1)
    need(a["time_accounting"]["cumulative_spent_seconds"] == PRIOR_SECONDS+310., "Clock scopes double counted")
    need(a["profile_only_raise_scenario"]["unit_rates"] == rates, "A faster profile lowered the frozen maximum")
    b = assess(**args, profile_bundle_seconds=7., profile_bundle_bytes=1100000)
    need(b["profile_only_raise_scenario"]["unit_rates"]["complete_probe_seconds"] == 7. and b["profile_only_raise_scenario"]["unit_rates"]["complete_probe_bytes"] == 1100000, "Larger profile not charged")
    need(b["frozen_rate_scenario"] == a["frozen_rate_scenario"], "Frozen scenario mutated")
    need(b["profile_only_raise_scenario"]["status"] == "HOLD_RESOURCE", "Infeasible continuation passed")
    trial = {"total_ns": 7_000_000_000, "output_bytes": 1000, "stages": [{"total_ns": 6_000_000_000}]}
    need(profile_maxima({"trials": [trial]}) == (7., 1000, 1), "Inclusive child time double counted")
    need(profile_maxima({"trials": [trial], "reference_calibration": [{"total_ns": 8_000_000_000, "output_bytes": 1200}]}) == (8., 1200, 2), "Slower original calibration maximum discarded")
    for malformed in (completed+[completed[0]], [(128, "A", 0)]):
        try:
            remaining_work(malformed)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal Cartesian keys accepted")
    bad = dict(args, elapsed_postwork_seconds=-1)
    try:
        assess(**bad)
    except ValueError:
        pass
    else:
        raise AssertionError("Negative clock accepted")
    return {"status": "PASS", "checks": 9, "scope": "Pure synthetic arithmetic; no experiment, physics, model or profiling execution."}


def authenticate_profile(profile_run):
    """Authenticate completed saved profiling receipts; no codec/model/physics call."""
    profile_run = Path(profile_run).resolve()
    profile_path, process_path = profile_run/"summary.json", profile_run/"process_wall_receipt.json"
    profile, process = load(profile_path), load(process_path)
    need(process["exit_code"] == 0 and nonnegative(process["process_wall_seconds"], "profile outer process seconds") >= profile.get("wall_seconds", 0.), "Incomplete or inconsistent profile process")
    need(str(profile.get("status", "")).startswith(("PASS", "COMPLETE")), "Profile must complete and pass")
    need(profile.get("schema") == "nt-storage-profile-summary-v1", "Reviewed profile schema required")
    need(profile.get("training_updates") == 0 and profile.get("new_actual_trajectories") == 0, "New learning/actual execution reported")
    guards = profile.get("forbidden_compute_calls")
    expected_guards = {"mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse",
        "environment.__init__", "environment.reset", "environment.restore", "environment.step", "model.__call__"}
    need(isinstance(guards, dict) and set(guards) == expected_guards and all(type(v) is int and v == 0 for v in guards.values()), "Profile forbidden-compute guards missing/nonzero")
    for name in ("source_runs_unchanged", "source_code_unchanged"):
        values = profile.get(name)
        need(isinstance(values, dict) and values and all(v is True for v in values.values()), "Profile preservation check missing/false: " + name)
    profile_entry_path = profile_run/"entry.json"
    need(sha(profile_entry_path) == profile["entry_sha256"], "Profile entry receipt changed")
    profile_entry = load(profile_entry_path)
    profile_config_path = ROOT/"configs/nt_storage_profile.v1.json"
    need(sha(profile_config_path) == profile_entry["config_sha256"], "Executed profile config changed")
    profile_config = load(profile_config_path)
    need(profile_entry["source_code_sha256"] == profile_config["source_code_sha256"], "Profile code map differs from executed design")
    need(set(profile["source_code_unchanged"]) == set(profile_entry["source_code_sha256"]), "Profile final code inventory omitted source")
    for relative, digest in profile_entry["source_code_sha256"].items():
        need(sha(ROOT/relative) == digest, "Executed profile source changed: " + relative)
    need(profile_config["prior_conservative_seconds"] == PRIOR_SECONDS, "Profile prior conservative clock differs")
    need(profile.get("bindings") == profile_entry["bindings"] == profile_config["bindings"], "Profile inherited bindings differ")
    need(set(profile["source_runs_unchanged"]) == set(profile_config["source_runs"]), "Profile final source inventory omitted run")
    inherited = profile.get("bindings", profile)
    for aliases, expected in ((('source_summary_sha256', 'source_run_summary_sha256', 'full_run_summary_sha256'), FIXED_SHA['summary.json']),
                              (('source_entry_sha256', 'full_run_entry_sha256'), FIXED_SHA['entry.json']),
                              (('executed_config_sha256',), FIXED_SHA['config']),
                              (('prior_delivery_review_sha256',), FIXED_SHA['delivery'])):
        actual = next((inherited[a] for a in aliases if a in inherited), None)
        need(actual == expected, "Profile inherited source binding missing/different: " + aliases[0])
    maximum_s, maximum_b, count = profile_maxima(profile)
    timed_trials = profile["trials"]
    calibration = profile["reference_calibration"]
    need(len(timed_trials) == len(profile_config["cases"])*profile_config["repeats"] == 18 and len(calibration) == 3, "Incomplete reviewed profile Cartesian trials/calibration")
    expected_case_keys = {(c["id"], repeat) for c in profile_config["cases"] for repeat in range(3)}
    observed_case_keys = [(r["case_id"], r["repeat"]) for r in timed_trials]
    need(len(observed_case_keys) == len(set(observed_case_keys)) and set(observed_case_keys) == expected_case_keys, "Profile case keys omitted/duplicated")
    need({(r["case_id"], r["repeat"]) for r in calibration} == {(profile_config["calibration_case_id"], repeat) for repeat in range(3)}, "Original-function calibration cases differ")
    for trial in timed_trials:
        stages = trial["stages"]
        need(stages["bundle"]["inclusive_ns"] == trial["total_ns"] and
             all(row["exclusive_ns"] >= 0 for row in stages.values()) and
             sum(row["exclusive_ns"] for row in stages.values()) == trial["total_ns"], "Profile nested wall durations inconsistent")
        need(sum(row["exclusive_cpu_ns"] for row in stages.values()) == stages["bundle"]["inclusive_cpu_ns"] and
             all(row["exclusive_cpu_ns"] >= 0 for row in stages.values()), "Profile nested CPU durations inconsistent")
    for trial in timed_trials+calibration:
        need(trial["tree_exact"]["status"] == trial["whole_probe_exact"]["status"] == "PASS_EXACT", "Profile complete evidence comparison failed")
    for reference in calibration:
        matched = next(t for t in timed_trials if (t["case_id"], t["repeat"]) == (reference["case_id"], reference["repeat"]))
        need(reference["payload_sha256"] == matched["payload_sha256"] and matched["calibration"]["payload_bytes_identical"] is True, "Original/instrumented payload calibration differs")
    return profile, process, {"profile_summary_sha256": sha(profile_path),
        "profile_process_receipt_sha256": sha(process_path), "profile_entry_sha256": sha(profile_entry_path),
        "profile_config_sha256": sha(profile_config_path), "profile_source_sha256": profile_entry["source_code_sha256"],
        "inclusive_bundle_max_seconds": maximum_s, "inclusive_bundle_max_bytes": maximum_b,
        "bundle_count": count, "instrumented_bundle_count": len(timed_trials), "original_calibration_bundle_count": len(calibration)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--elapsed-postwork-seconds", "--additional-work-seconds", dest="elapsed_postwork_seconds", type=float, default=0., help="Nonoverlapping current-task pre/post-profile elapsed work, excluding the profile outer process, this assessment and the earlier fixed delivery ledger. Charge an inclusive caller scope once; do not add contained subprocess timers again.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        need(args.profile_run is None and args.output is None, "Self-test cannot use experiment paths")
        print(json.dumps(self_test(), ensure_ascii=False))
        return
    need(args.profile_run is not None and args.output is not None, "--profile-run and --output required")
    profile_run, output = args.profile_run.resolve(), args.output.resolve()
    need(not output.exists() and output.parent.is_dir(), "Exclusive output with existing parent required")
    need(output != FULL_RUN.resolve() and FULL_RUN.resolve() not in output.parents, "Output cannot alter frozen source run")
    summary, config, bindings, work = authenticate_frozen()
    profile, process, profile_auth = authenticate_profile(profile_run)
    maximum_s, maximum_b, count = profile_auth["inclusive_bundle_max_seconds"], profile_auth["inclusive_bundle_max_bytes"], profile_auth["bundle_count"]
    timed_trials, calibration = profile["trials"], profile["reference_calibration"]
    profile_path, process_path = profile_run/"summary.json", profile_run/"process_wall_receipt.json"
    current_bytes = config["existing_run_bytes_before_full"]+logical_bytes(FULL_RUN)+logical_bytes(profile_run)
    result = assess(prior_seconds=PRIOR_SECONDS, process_seconds=process["process_wall_seconds"],
        elapsed_postwork_seconds=args.elapsed_postwork_seconds, assessment_seconds=time.perf_counter()-PROGRAM_START,
        work=work, rates=summary["observed_unit_maxima"], profile_bundle_seconds=maximum_s,
        profile_bundle_bytes=maximum_b, preserved_bytes=current_bytes, free_bytes=shutil.disk_usage(profile_run).free)
    result.update({"schema": "nt-remaining-resource-reassessment-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_bindings": bindings, "profile_summary_sha256": sha(profile_path), "profile_process_receipt_sha256": sha(process_path),
        "assessment_tool_sha256": sha(Path(__file__)), "profile_inclusive_bundle_max_seconds": maximum_s,
        "profile_inclusive_bundle_max_bytes": maximum_b, "profile_bundle_count": count,
        "profile_instrumented_bundle_count": len(timed_trials), "profile_original_calibration_bundle_count": len(calibration),
        "preserved_local_experiment_bytes_at_assessment": current_bytes,
        "limits_of_evidence": ["Same-code warm-filesystem observations cannot prove a cold-cache/OS upper bound or retrospectively identify the original slowdown.",
            "Historical maxima never decrease. No size-conditioned extrapolation, optimization speedup, row substitution or count relaxation is assumed.",
            "The conservative previous publication reserve remains charged once; constituent earlier process/audit timers are not added again.",
            "This assessment's final JSON write/process shutdown and subsequent postwork require an external elapsed ledger charge; final reserve is prospective."]})
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": result["profile_only_raise_scenario"]["status"],
        "remaining": work, "projected_cumulative_seconds": result["profile_only_raise_scenario"]["projected_cumulative_seconds"]}))


if __name__ == "__main__":
    main()
