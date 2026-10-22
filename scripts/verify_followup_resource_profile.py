#!/usr/bin/env python3
"""Recompute public timing partitions and resource arithmetic; no simulator."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/followup/20261022-resource-profile"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9), (actual, expected)


def verify():
    manifest = load(BASE / "manifest.json")
    for row in manifest["files"]:
        path = (BASE / row["path"]).resolve()
        assert path.is_relative_to(BASE.resolve())
        assert path.stat().st_size == row["bytes"] and sha(path) == row["sha256"]
    assert len(manifest["files"]) == manifest["file_count_excluding_manifest"] == 6
    x = load(BASE / "resource_summary.json")
    compute, io = load(BASE / "compute/summary.json"), load(BASE / "io/summary.json")
    assert compute["status"] == "PASS_TIMED_COMPUTE_AND_FULL_FIDELITY"
    assert io["status"] == "PASS_UNCHANGED_CODEC_AND_EXACT_SAVED_EVIDENCE_PROFILE"
    assert len(compute["rows"]) == len(io["measurements"]) == 27
    assert sum(r["intervals"] for r in compute["rows"]) == compute["actual_intervals"] == 174
    full, exact, intervals, partitions = 0, 0, 0, 0
    keys = set()
    for r in compute["rows"]:
        key = (r["state"], r["panel"], r["candidate_id"], r["repeat"])
        assert key not in keys
        keys.add(key)
        p = r["phases_ns"]
        assert all(v >= 0 for v in p.values())
        assert p["probe_total"] == p["probe_other_validation_task_cost"] + sum(v for k, v in p.items() if k.startswith("env_"))
        original_keys = ["probe_total", "write_raw_total", "read_raw_total", "roundtrip_exact",
                         "full_probe_repeat_exact", "diagnostic_npz", "trial_other_receipt_checks"]
        t = sum(p[k] for k in original_keys)
        assert t == p["trial_total"] - p["new_raw_vs_original_exact"]
        full += t
        exact += p["roundtrip_exact"] + p["full_probe_repeat_exact"]
        intervals += r["intervals"]
        assert r["original_exact"] == r["roundtrip_exact"] == "PASS_EXACT"
        assert r["full_probe_repeat_exact"] == ("FIRST_REPEAT" if r["repeat"] == 0 else "PASS_EXACT")
    assert len({k[:3] for k in keys}) == 9
    assert full == x["compute_pipeline_weighted"]["pipeline_equivalent_ns"]
    close(100 * exact / full, x["compute_pipeline_weighted"]["grouped_percent"]["exact_verification"])
    for r in io["measurements"]:
        assert all(r["raw_bytes_identical_to_source"].values())
        assert r["raw_roundtrip"]["status"] == "PASS_EXACT"
        assert all(v["status"] == "PASS_EXACT" for v in r["source_repeat_checks"])
        assert r["sum_stage_ns"] == sum(p["total_ns"] for p in r["phases"])
        for p in r["phases"]:
            assert all(v >= 0 for v in p["exclusive_phase_ns"].values())
            assert p["total_ns"] == p["sum_exclusive_ns"] == sum(p["exclusive_phase_ns"].values())
            partitions += 1
    assert partitions == 135
    b, binding = x["budget"], x["source_bindings"]
    for kind in ("compute", "io"):
        entry = load(BASE / kind / "entry.json")
        code = entry.get("code_sha256", entry.get("implementation_sha256"))
        for rel, digest in code.items():
            assert sha(ROOT / rel) == digest
    assert sha(ROOT / "tools/build_followup_resource_profile.py") == binding["builder_sha256"]
    assert sha(ROOT / "configs/followup_candidate_diagnostic.v1.json") == binding["frozen_design_sha256"]
    prior = ROOT / "artifacts/followup/20261022-d1-d2"
    assert sha(prior / "summary.json") == binding["D2_summary_sha256"]
    for state, digest in binding["D1_restoration_sha256"].items():
        assert sha(prior / f"restoration-{state}.json") == digest
    measured_rate = max((r["phases_ns"]["trial_total"]-r["phases_ns"]["new_raw_vs_original_exact"])/1e9/r["intervals"] for r in compute["rows"])
    close(measured_rate, b["new_compute_max_pipeline_seconds_per_interval"])
    close(b["selected_trial_seconds_per_interval"], max(measured_rate, b["original_D2_max_seconds_per_interval_floor"]))
    close(b["selected_bytes_per_interval"], max(b["original_D2_max_bytes_per_interval_floor"], b["new_compute_max_full_trial_directory_bytes_per_interval"]))
    close(sum(b["already_spent_seconds_by_phase"].values()), b["already_spent_seconds"])
    assert b["wall_cap_seconds_cumulative"] == 7200 and b["incremental_storage_cap_bytes"] == 5*1024**3
    outcomes = {}
    for s in b["scenarios"]:
        n = len(s["states"])
        assert s["model_count"] == 7 and s["physical_repeats"] == 3 and s["candidate_ids_per_state"] == 128
        assert s["panels"] == ["A", "B"] and s["D2_rows_reused"] is False and s["extension_included"] is False
        assert s["physical_trajectories"] == n*128*3*2
        assert s["future_probe_interval_upper_bound"] == n*128*3*13
        assert s["free_prediction_transition_rows"] == s["teacher_forced_transition_rows_upper_bound"] == n*128*5*7
        future = b["safety_factor"] * (s["future_probe_interval_upper_bound"]*b["selected_trial_seconds_per_interval"]
            + s["future_restore_interval_upper_bound"]*b["max_D1_restoration_seconds_per_interval"]) + n*b["fixed_allowance_seconds_per_state"]
        close(s["future_total_seconds_upper_planning_estimate"], future)
        close(s["cumulative_seconds_upper_planning_estimate"], b["already_spent_seconds"]+future)
        size = math.ceil(b["safety_factor"]*(s["future_probe_interval_upper_bound"]+s["future_restore_interval_upper_bound"])*b["selected_bytes_per_interval"])
        assert size == s["future_incremental_bytes_upper_planning_estimate"]
        assert s["cumulative_incremental_bytes_upper_planning_estimate"] == size+b["existing_D1_D2_run_bytes"]+b["profile_run_bytes"]
        assert s["wall_cap_fit"] == (b["already_spent_seconds"]+future <= b["wall_cap_seconds_cumulative"])
        assert s["storage_cap_fit"] == (s["cumulative_incremental_bytes_upper_planning_estimate"] <= b["incremental_storage_cap_bytes"])
        assert s["free_space_reserve_fit"] == (size+b["free_space_reserve_bytes"] <= b["free_bytes_at_projection"])
        assert s["execution_status"] == "NOT_RUN"
        outcomes[s["id"]] = {"cumulative_minutes": s["cumulative_seconds_upper_planning_estimate"]/60,
            "cumulative_GiB": s["cumulative_incremental_bytes_upper_planning_estimate"]/1024**3,
            "wall_cap_fit": s["wall_cap_fit"]}
    assert [s["wall_cap_fit"] for s in b["scenarios"]] == [True, False, False]
    inventory = load(BASE / "local_raw_inventory.json")
    assert inventory["pair_count"] == len(inventory["records"]) == 54
    assert inventory["file_count"] == sum(len(r["files"]) for r in inventory["records"]) == 108
    assert inventory["raw_bytes"] == sum(f["bytes"] for r in inventory["records"] for f in r["files"])
    assert x["new_experiment_execution_status"] == "NOT_RUN"
    return {"schema": "followup-resource-profile-public-verification-v1", "status": "PASS_PUBLIC_PARTITIONS_AND_RESOURCE_ARITHMETIC",
        "verifier_sha256": sha(__file__), "manifest_sha256": sha(BASE / "manifest.json"),
        "verified_public_files": 6, "compute_trials": 27, "compute_intervals": intervals,
        "io_phase_partitions": partitions, "exact_verification_percent": 100*exact/full, "scenarios": outcomes,
        "new_physics_calls": 0, "new_model_calls": 0,
        "scope": "Authenticates selected public receipts and redoes timing/budget arithmetic. Raw files remain local; their hash inventory is not independent replay evidence. Fixed allowance is a planning value and does not certify the next driver."}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    result = verify()
    if a.output:
        with a.output.open("x") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
