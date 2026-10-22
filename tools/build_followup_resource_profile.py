#!/usr/bin/env python3
"""Publish portable profiling receipts and a conservative, non-executing budget.

Only the standard library is imported. The source measurement runs, original
scientific files, model weights and frozen v1 design are never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_RATE_FLOOR = 0.48116094996221365
ORIGINAL_BYTES_FLOOR = 88934.8
SAFETY = 1.5
WALL_CAP = 7200
STORAGE_CAP = 5 * 1024**3
FREE_RESERVE = 2 * 1024**3
FIXED_ALLOWANCE_PER_STATE = 120


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def tree_bytes(folder):
    return sum(path.stat().st_size for path in Path(folder).rglob("*") if path.is_file())


def assert_portable(value):
    text = json.dumps(value, ensure_ascii=False, allow_nan=False)
    for forbidden in ("/Users/", "/private/var/", "/var/folders/", "file://"):
        if forbidden in text:
            raise ValueError("Private absolute path in public profiling material")


def weighted_compute(rows):
    phases = Counter()
    for row in rows:
        p = row["phases_ns"]
        env_other = sum(value for key, value in p.items()
                        if key.startswith("env_") and key != "env_step")
        components = {
            "env_step_including_contact_diagnostics": p.get("env_step", 0),
            "env_creation_restore_observe_snapshot": env_other,
            "probe_task_geometry_cost_other": p["probe_other_validation_task_cost"],
            "codec_write_and_read": p["write_raw_total"] + p["read_raw_total"],
            "raw_roundtrip_exact": p["roundtrip_exact"],
            "full_probe_repeat_exact": p["full_probe_repeat_exact"],
            "diagnostic_npz": p["diagnostic_npz"],
            "receipt_and_check_other": p["trial_other_receipt_checks"],
        }
        equivalent = p["trial_total"] - p["new_raw_vs_original_exact"]
        if sum(components.values()) != equivalent or equivalent <= 0:
            raise ValueError("Compute timing partition is not additive")
        phases.update(components)
    total = sum(phases.values())
    return {
        "trial_count": len(rows),
        "returned_intervals": sum(row["intervals"] for row in rows),
        "pipeline_equivalent_ns": total,
        "pipeline_equivalent_seconds": total / 1e9,
        "additional_original_comparison_excluded_ns": sum(
            row["phases_ns"]["new_raw_vs_original_exact"] for row in rows),
        "method": "Sum phase nanoseconds across trials, then divide by their common total; not an average of per-trial percentages.",
        "phases": {key: {"ns": value, "seconds": value / 1e9,
                          "percent": 100 * value / total}
                   for key, value in sorted(phases.items())},
        "grouped_percent": {
            "exact_verification": 100 * (phases["raw_roundtrip_exact"] + phases["full_probe_repeat_exact"]) / total,
            "codec_write_and_read": 100 * phases["codec_write_and_read"] / total,
            "env_step_including_contact_diagnostics": 100 * phases["env_step_including_contact_diagnostics"] / total,
            "remaining_setup_task_diagnostic_and_receipts": 100 * (
                total - phases["raw_roundtrip_exact"] - phases["full_probe_repeat_exact"]
                - phases["codec_write_and_read"] - phases["env_step_including_contact_diagnostics"]) / total,
        },
        "scope": "The original-comparison benchmark extra is excluded; all original raw roundtrip and full FollowupProbe repeat exact checks remain. env_step is not engine-only time.",
    }


def weighted_io(measurements):
    stages, phases = Counter(), Counter()
    for row in measurements:
        if row["sum_stage_ns"] != sum(stage["total_ns"] for stage in row["phases"]):
            raise ValueError("IO stage sum mismatch")
        for stage in row["phases"]:
            exclusive = stage["exclusive_phase_ns"]
            if sum(exclusive.values()) != stage["total_ns"]:
                raise ValueError("IO exclusive timing partition mismatch")
            stages[stage["stage"]] += stage["total_ns"]
            phases.update(exclusive)
    total = sum(stages.values())
    if total <= 0 or total != sum(phases.values()):
        raise ValueError("Invalid IO aggregate")
    def table(values):
        return {key: {"ns": value, "seconds": value / 1e9,
                      "percent": 100 * value / total}
                for key, value in sorted(values.items())}
    return {"measurement_count": len(measurements), "timed_stage_ns": total,
            "stages": table(stages), "exclusive_subphases": table(phases),
            "scope": "IO-only warm-filesystem microbenchmarks. Each measurement performs two saved-evidence repeat comparisons; this composition is not the full original trial pipeline or the full FollowupProbe comparator.",
            "not_used_as_future_pipeline_rate": True}


def raw_inventory(profile, compute, io):
    records = []
    for row in compute["rows"]:
        folder = (profile / "compute" / row["state"] / f"candidate-{row['candidate_id']:03d}"
                  / row["panel"] / f"repeat-{row['repeat']}")
        receipt = load(folder / "raw_receipt.json")
        if sha256(folder / "raw_receipt.json") != row["raw_receipt_sha256"]:
            raise ValueError("Compute raw receipt binding changed")
        expected = {Path(path).name: digest for path, digest in receipt["files_sha256"].items()}
        files = []
        for name, digest in sorted(expected.items()):
            path = folder / name
            if sha256(path) != digest:
                raise ValueError("Compute raw file binding changed")
            files.append({"path": path.relative_to(profile).as_posix(), "sha256": digest,
                          "bytes": path.stat().st_size})
        if len(files) != 2 or sum(item["bytes"] for item in files) != row["raw_bytes"]:
            raise ValueError("Compute raw pair count/size mismatch")
        records.append({"measurement": "compute", "state": row["state"],
                        "candidate_id": row["candidate_id"], "panel": row["panel"],
                        "repeat": row["repeat"], "files": files})
    for row in io["measurements"]:
        files = []
        for rel, digest in sorted(row["raw_receipt"]["files_sha256"].items()):
            path = (profile / "io" / rel).resolve()
            if not path.is_relative_to(profile / "io") or sha256(path) != digest:
                raise ValueError("IO raw file binding changed or escaped profile")
            if path.stat().st_size != row["raw_receipt"]["bytes"][rel]:
                raise ValueError("IO raw size mismatch")
            files.append({"path": path.relative_to(profile).as_posix(), "sha256": digest,
                          "bytes": path.stat().st_size})
        if len(files) != 2 or not all(row["raw_bytes_identical_to_source"].values()):
            raise ValueError("IO raw pair/identity mismatch")
        records.append({"measurement": "io", "state": row["state"],
                        "candidate_id": row["candidate_id"], "panel": row["panel"],
                        "pass": row["pass"], "files": files})
    return {"schema": "followup-profile-local-raw-inventory-v1",
            "scope": "Hashes and byte counts only. Complete new raw pairs remain in the local measurement run and are not included in this public artifact.",
            "pair_count": len(records), "file_count": sum(len(row["files"]) for row in records),
            "raw_bytes": sum(item["bytes"] for row in records for item in row["files"]),
            "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original, profile, out = args.original_root.resolve(), args.profile.resolve(), args.output.resolve()
    if out == profile or profile in out.parents or out == original:
        raise ValueError("Public output must be separate from source measurement runs")
    source = original / "runs/followup-near-goal-20261022-01"
    files = {f"{name}/{kind}.json": profile / name / f"{kind}.json"
             for name in ("compute", "io") for kind in ("entry", "summary")}
    copies = {rel: path.read_bytes() for rel, path in files.items()}
    documents = {rel: json.loads(value) for rel, value in copies.items()}
    compute, io = documents["compute/summary.json"], documents["io/summary.json"]
    if compute["status"] != "PASS_TIMED_COMPUTE_AND_FULL_FIDELITY":
        raise ValueError("Compute measurement not complete")
    if io["status"] != "PASS_UNCHANGED_CODEC_AND_EXACT_SAVED_EVIDENCE_PROFILE":
        raise ValueError("IO measurement not complete")
    for value in documents.values():
        assert_portable(value)
    if compute["model_calls"] or compute["training_updates"] or io["physics_calls"] or io["model_calls"] or io["training_updates"]:
        raise ValueError("Unexpected model/training/IO physics execution")
    design_path = ROOT / "configs/followup_candidate_diagnostic.v1.json"
    design = load(design_path)
    if len(design["models"]) != 7 or design["bank"]["count"] != 128 or design["panel_A"]["physical_repeats"] != 3 or design["panel_B"]["physical_repeats"] != 3:
        raise ValueError("Frozen design dimensions changed")
    pilot = load(source / "D2/summary.json")
    d1 = {name: load(source / "D1" / name / "restoration.json") for name in ("N-T", "F-O0")}
    if pilot["status"] != "PASS_PILOT_RECORDING_AND_REPRODUCTION_GATES" or any(
            value["status"] != "PASS_EXACT_RESTORATION_GATE" for value in d1.values()):
        raise ValueError("Prior execution gate not passed")
    measured_rate = max((row["phases_ns"]["trial_total"] - row["phases_ns"]["new_raw_vs_original_exact"])
                        / 1e9 / row["intervals"] for row in compute["rows"])
    rate_floor = pilot["resource_projection"]["max_measured_trial_wall_seconds_per_returned_interval"]
    byte_floor = pilot["resource_projection"]["max_measured_trial_bytes_per_returned_interval"]
    if rate_floor != ORIGINAL_RATE_FLOOR or byte_floor != ORIGINAL_BYTES_FLOOR:
        raise ValueError("Original conservative floor differs from expected frozen receipt")
    measured_bytes = max(tree_bytes(profile / "compute" / row["state"]
        / f"candidate-{row['candidate_id']:03d}" / row["panel"] / f"repeat-{row['repeat']}")
        / row["intervals"] for row in compute["rows"])
    rate, bytes_rate = max(rate_floor, measured_rate), max(byte_floor, measured_bytes)
    past_seconds = {"D1_N-T": d1["N-T"]["wall_seconds"],
                    "D1_F-O0": d1["F-O0"]["wall_seconds"],
                    "D2": pilot["wall_seconds"], "compute_profile": compute["wall_seconds"],
                    "io_profile": io["total_wall_seconds"]}
    spent = sum(past_seconds.values())
    existing_bytes, profile_bytes = tree_bytes(source), tree_bytes(profile)
    free = shutil.disk_usage(profile).free
    d1_rate = max(value["wall_seconds"] / value["physics_intervals_exact"] for value in d1.values())
    scopes = []
    for scope_id, states, restore_intervals in (
            ("N-T_complete_base128", ["N-T"], 0),
            ("N-T_F-O0_complete_base128", ["N-T", "F-O0"], 0),
            ("original_six_state_D3_base", [value["id"] for value in design["states"]], 685)):
        n = len(states)
        intervals, trajectories = n * 128 * 3 * (5 + 8), n * 128 * 3 * 2
        trial_seconds = SAFETY * intervals * rate
        restore_seconds = SAFETY * restore_intervals * d1_rate
        allowance = n * FIXED_ALLOWANCE_PER_STATE
        future = trial_seconds + restore_seconds + allowance
        future_bytes = math.ceil(SAFETY * (intervals + restore_intervals) * bytes_rate)
        total_bytes = existing_bytes + profile_bytes + future_bytes
        scopes.append({"id": scope_id, "states": states, "model_count": 7,
            "physical_repeats": 3, "panels": ["A", "B"], "candidate_ids_per_state": 128,
            "physical_trajectories": trajectories, "future_probe_interval_upper_bound": intervals,
            "future_restore_interval_upper_bound": restore_intervals,
            "free_prediction_transition_rows": n * 128 * 5 * 7,
            "teacher_forced_transition_rows_upper_bound": n * 128 * 5 * 7,
            "D2_rows_reused": False, "extension_included": False,
            "future_trial_seconds_upper_planning_estimate": trial_seconds,
            "future_restoration_seconds_upper_planning_estimate": restore_seconds,
            "fixed_hash_inference_teacher_report_allowance_seconds": allowance,
            "future_total_seconds_upper_planning_estimate": future,
            "cumulative_seconds_upper_planning_estimate": spent + future,
            "future_incremental_bytes_upper_planning_estimate": future_bytes,
            "cumulative_incremental_bytes_upper_planning_estimate": total_bytes,
            "wall_cap_fit": spent + future <= WALL_CAP,
            "storage_cap_fit": total_bytes <= STORAGE_CAP,
            "free_space_reserve_fit": free >= future_bytes + FREE_RESERVE,
            "execution_status": "NOT_RUN"})
    complete = weighted_compute(compute["rows"])
    per_group = {}
    for state, panel in sorted({(row["state"], row["panel"]) for row in compute["rows"]}):
        per_group[f"{state}/{panel}"] = weighted_compute([
            row for row in compute["rows"] if (row["state"], row["panel"]) == (state, panel)])
    inventory = raw_inventory(profile, compute, io)
    fits = all(scopes[0][key] for key in ("wall_cap_fit", "storage_cap_fit", "free_space_reserve_fit"))
    summary = {"schema": "followup-resource-profile-and-budget-v1",
        "status": "READY_FOR_DRIVER_PREFLIGHT_ONLY" if fits else "HOLD_RESOURCE_CAP",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "new_experiment_execution_status": "NOT_RUN",
        "original_full_D3_status": "NOT_RUN_RESOURCE_HOLD",
        "recommended_proposed_scope": scopes[0]["id"] if fits else None,
        "recommendation_scope": "A proposal for a separate complete-state diagnostic. The next driver needs a new scope/config, identity preflight and resource gate; this report does not authorize or execute it and does not mark original six-state D3 complete.",
        "measurement_order": "compute then IO, sequential; no overlapping benchmarks",
        "compute_pipeline_weighted": complete, "compute_state_panel_weighted": per_group,
        "io_microbenchmark_weighted": weighted_io(io["measurements"]),
        "budget": {"wall_cap_seconds_cumulative": WALL_CAP, "incremental_storage_cap_bytes": STORAGE_CAP,
            "free_space_reserve_bytes": FREE_RESERVE, "safety_factor": SAFETY,
            "already_spent_seconds_by_phase": past_seconds, "already_spent_seconds": spent,
            "existing_D1_D2_run_bytes": existing_bytes, "profile_run_bytes": profile_bytes,
            "free_bytes_at_projection": free, "original_D2_max_seconds_per_interval_floor": rate_floor,
            "new_compute_max_pipeline_seconds_per_interval": measured_rate,
            "selected_trial_seconds_per_interval": rate,
            "original_D2_max_bytes_per_interval_floor": byte_floor,
            "new_compute_max_full_trial_directory_bytes_per_interval": measured_bytes,
            "selected_bytes_per_interval": bytes_rate,
            "max_D1_restoration_seconds_per_interval": d1_rate,
            "fixed_allowance_seconds_per_state": FIXED_ALLOWANCE_PER_STATE,
            "fixed_allowance_status": "Explicit planning allowance for hashes/inference/teacher/report; not a measured upper bound or guarantee. Preflight must measure and stop if actual budget exceeds cap.",
            "scenarios": scopes},
        "fidelity": {"source_D2_preserved": True,
            "new_compute_groups": compute["groups"], "new_compute_trajectories": compute["trajectories"],
            "new_compute_intervals": compute["actual_intervals"], "new_compute_complete_result_repeats": 3,
            "new_compute_original_raw_and_roundtrip": "PASS_EXACT",
            "new_compute_full_FollowupProbe_repeat": "PASS_EXACT after first repeat",
            "io_measurements": len(io["measurements"]),
            "io_new_raw_bytes_identical_to_source": True,
            "codec_preset": 3, "fsync_hash_schema_and_bit_exact_checks_preserved": True,
            "full_new_raw_local_only": True, "new_model_calls": 0, "training_updates": 0},
        "source_bindings": {"frozen_design_sha256": sha256(design_path),
            "D2_summary_sha256": sha256(source / "D2/summary.json"),
            "D1_restoration_sha256": {name: sha256(source / "D1" / name / "restoration.json") for name in d1},
            "measurement_files_sha256": {rel: hashlib.sha256(value).hexdigest() for rel, value in copies.items()},
            "builder_sha256": sha256(Path(__file__))},
        "limitations": [
            "Measurements cover nine selected saved candidate groups, not all future states; the original D2 maximum floor and factor 1.5 are retained.",
            "env_step includes per-substep environment/contact diagnostics, not pure engine compute. Other probe time includes task, geometry and cost.",
            "Additional comparison with authenticated old raw is benchmark overhead; original roundtrip and full Probe repeat checks are not removed.",
            "IO uses warm filesystem/cache state. CPU profiler overhead is excluded from timed stage composition; pstats cumulative time overlaps and must not be summed.",
            "IO repeat comparison covers saved raw, diagnostic arrays and scientific summaries, not unsaved final integration state; original and fresh compute full Probe checks remain separate.",
            "No algorithm, codec, cost, model, original criterion or frozen design was changed. No speedup or cold-cache/power-loss guarantee is claimed.",
            "No D2 scientific-row reuse, extension, reduced repeat count or reduced model set is assumed. Physical futures are shared across all seven models.",
            "Budgets are conservative planning estimates, not proved upper bounds. All accumulated D1/D2/measurement time is charged; no silent clock reset.",
            "Raw hashes document omitted local files; public summaries alone do not independently replay those raw data or verify all underlying fidelity checks.",
        ]}
    assert_portable(summary)
    assert_portable(inventory)
    out.mkdir(parents=True, exist_ok=False)
    for rel, content in copies.items():
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content)
        if target.read_bytes() != content:
            raise ValueError("Public summary copy differs")
    dump(out / "local_raw_inventory.json", inventory)
    dump(out / "resource_summary.json", summary)
    records = [{"path": path.relative_to(out).as_posix(), "sha256": sha256(path),
                "bytes": path.stat().st_size}
               for path in sorted(out.rglob("*")) if path.is_file()]
    dump(out / "manifest.json", {"schema": "followup-resource-profile-public-files-v1",
        "scope": "Byte-identical measurement entry/summary copies, derived weighted budget, and raw hashes; no raw payloads.",
        "file_count_excluding_manifest": len(records),
        "bytes_excluding_manifest": sum(row["bytes"] for row in records), "files": records})
    for rel, path in files.items():
        if path.read_bytes() != copies[rel]:
            raise ValueError("Source measurement changed during publication")
    print(json.dumps({"status": summary["status"], "output": out.relative_to(ROOT).as_posix(),
        "public_files_excluding_manifest": len(records), "local_raw_pairs": inventory["pair_count"],
        "already_spent_seconds": spent, "selected_seconds_per_interval": rate,
        "recommended_scope": summary["recommended_proposed_scope"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
