#!/usr/bin/env python3
"""Verify portable N-T full-bank COMPLETE/STOP evidence with NumPy only.

Checks public hashes, dimensions, selected IDs, recorded comparison verdicts,
omitted-evidence inventories and live-resource arithmetic. Omitted complete raw
and full-probe semantic audit remains a separate local verification.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/followup/20261022-nt-full"
MODELS = ("R", "S1-11", "S1-23", "S1-47", "SK-11", "SK-23", "SK-47")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.dtype.str.encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def close(left, right):
    require(math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-8), "Arithmetic mismatch")


def verify(base=BASE):
    started = time.perf_counter()
    base = base.resolve()
    manifest_path = base / "manifest.json"
    manifest = load(manifest_path)
    require(manifest["schema"] == "followup-nt-full-selected-public-evidence-v1", "Wrong public manifest schema")
    files = manifest["files"]
    require(len(files) == manifest["files_excluding_self"], "Manifest count mismatch")
    require(sum(row["bytes"] for row in files) == manifest["bytes_excluding_self"], "Manifest byte count mismatch")
    file_paths = {row["path"] for row in files}
    require(len(file_paths) == len(files), "Duplicate public path")
    for row in files:
        path = (base / row["path"]).resolve()
        require(path.is_relative_to(base) and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Public file hash/size mismatch")
    require(manifest["publisher_sha256"] == sha(ROOT / "tools/publish_nt_full.py"), "Publisher source identity mismatch")
    for path in base.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".jsonl"):
            payload = path.read_text(encoding="utf-8")
            require(not any(pattern in payload for pattern in ("/Users/", "/private/var/", "/var/folders/", "file://")), "Private absolute path in public text")
    copies = {row["public_path"]: row for row in manifest["source_copy_records"]}
    for relative, row in copies.items():
        require(sha(base / relative) == row["public_sha256"], "Published copy record mismatch")
        if row["transformation"] == "byte_identical":
            require(row["public_sha256"] == row["source_sha256"], "Byte-identical source mismatch")
    entry, summary, analysis, config, seal = [load(base / name) for name in
        ("entry.json", "summary.json", "analysis.json", "executed_config.json", "prospective_forecast_seal.json")]
    require(summary["status"] == manifest["source_run_status"] == analysis["source_run_status"], "Run status mismatch")
    require(copies["entry.json"]["source_sha256"] == manifest["source_entry_sha256"] == analysis["bindings"]["entry_sha256"], "Entry provenance mismatch")
    require(copies["summary.json"]["source_sha256"] == manifest["source_summary_sha256"] == analysis["bindings"]["run_receipt_sha256"], "Summary provenance mismatch")
    require(copies["analysis.json"]["source_sha256"] == manifest["source_analysis_sha256"], "Analysis provenance mismatch")
    require(copies["executed_config.json"]["source_sha256"] == entry["config_sha256"] == manifest["executed_config_sha256"], "Config provenance mismatch")
    require(sha(ROOT / "tools/summarize_nt_full.py") == analysis["bindings"]["summarizer_sha256"], "Analysis source identity mismatch")
    require(entry["full_input_typed_sha256"] == config["expected_input_typed_sha256"]
            and entry["bank_hash"] == config["expected_candidate_hash"], "Configured input/bank mismatch")
    require(entry["model_and_normalizer_sha256"] == config["expected_model_normalizer_sha256"]
            and len(entry["model_and_normalizer_sha256"]) == 22, "Frozen model/normalizer identity mismatch")
    with np.load(base / "bank.npz", allow_pickle=False) as saved:
        bank = saved["actions"].copy()
        require(bank.dtype == np.float64 and bank.shape == (128, 5, 2)
                and np.array_equal(saved["original_ids"], np.arange(128)) and array_hash(bank) == entry["bank_hash"], "Bank dimensions/identity mismatch")
    require(seal["before_any_new_actual_future"] is True and summary["training_updates"] == 0,
            "Prospective seal/training state mismatch")
    require(all(value == 0 for value in seal["future_engine_calls_during_prediction"].values())
            and all(value == 0 for value in summary["future_engine_calls_during_prediction"].values()), "Prediction guard nonzero")
    for relative, digest in seal["forecasts"].items():
        require(copies[relative]["source_sha256"] == digest, "Forecast seal/source mismatch")
    prediction_metadata = load(base / "forecasts/predictions.json")
    require(tuple(prediction_metadata) == MODELS and summary["model_ids"] == list(MODELS), "Seven-model identity mismatch")
    for name in MODELS:
        with np.load(base / "forecasts" / f"prediction-{name}.npz", allow_pickle=False) as saved:
            require(saved["states"].shape == (128, 6, 10) and saved["valid"].dtype == np.bool_, "Forecast dimensions mismatch")
            eligible = np.flatnonzero(saved["valid"] & np.isfinite(saved["total_cost"]))
            selected = None if not len(eligible) else int(eligible[np.argmin(saved["total_cost"][eligible])])
            require(selected == prediction_metadata[name]["selected_original_candidate_id"]
                    == analysis["model_hypotheses"][name]["selected_original_candidate_id"], "Forecast-selected ID mismatch")
            require(int(saved["valid"].sum()) == prediction_metadata[name]["valid_count"], "Forecast validity count mismatch")
    rows, index = summary["rows"], load(base / "trial_index.json")
    keys = {(row["candidate_id"], row["panel"], row["repeat"]) for row in rows}
    expected = set(product(range(128), ("A", "B"), range(3)))
    missing = {(row["candidate_id"], row["panel"], row["repeat"]) for row in index["missing_keys"]}
    require(len(keys) == len(rows) == summary["completed_trajectories"] == analysis["dimensions"]["recorded_complete_trajectories"], "Scientific row count mismatch")
    require(keys.issubset(expected) and missing == expected - keys, "Missing Cartesian key set mismatch")
    require(sum(row["intervals"] for row in rows) == analysis["dimensions"]["recorded_complete_intervals"], "Recorded interval sum mismatch")
    for row in rows:
        relative = f"actual/N-T/candidate-{row['candidate_id']:03d}/{row['panel']}/repeat-{row['repeat']}"
        require(row["candidate_hash"] == array_hash(bank[row["candidate_id"]])
                and row["initial_snapshot_exact"] and row["raw_roundtrip"]["status"] == "PASS_EXACT", "Scientific row identity/raw verdict mismatch")
        require(load(base / relative / "entry.json")["prospective_seal_sha256"] == copies["prospective_forecast_seal.json"]["source_sha256"], "Trial prospective seal mismatch")
        require(load(base / relative / "timing_final.json")["timings_ns"] == row["timings_ns"], "Final row timing receipt mismatch")
        receipt = row["full_probe_roundtrip"] if row["repeat"] == 0 else row["full_probe_repeat_exact"]
        require(receipt and receipt["status"] == "PASS_EXACT", "Full-probe recorded exact verdict mismatch")
        if row["panel"] == "B":
            require(row["cost"] is None and row["cost_components"] == {}, "B was given H5 cost")
        elif row["cost_finite"]:
            close(sum(row["cost_components"].values()), row["cost"])
        if row["repeat"] == 0:
            diagnostic = base / relative / "diagnostic.npz"
            require(sha(diagnostic) == row["diagnostic_receipt"]["sha256"], "First-repeat diagnostic identity mismatch")
    inventory = load(base / "local_evidence_inventory.json")
    raw_pairs = [row for row in inventory["records"] if row["raw_files"]]
    full_groups = [row for row in inventory["records"] if row["complete_probe_files"]]
    require(len(raw_pairs) == inventory["raw_pair_count"] == manifest["local_raw_pair_count"], "Raw inventory count mismatch")
    require(len(full_groups) == inventory["complete_probe_group_count"] == manifest["local_complete_probe_group_count"], "Complete-probe inventory count mismatch")
    for row in inventory["records"]:
        for record in row["raw_files"] + row["complete_probe_files"]:
            require(record["included_in_public_repository"] is False and record["path"] not in file_paths, "Local complete payload accidentally published")
    decisions = [json.loads(line) for line in (base / "resource_decisions.jsonl").read_text().splitlines()]
    for decision in decisions:
        close(decision["projected_cumulative_seconds"], decision["cumulative_spent_seconds"] + sum(decision["future_seconds"].values()))
        require(decision["limits"]["wall_cap_seconds"] == 7200, "Wall cap changed")
    require(decisions[-1] == summary["final_resource_decision"], "Final resource decision not last ledger record")
    if summary["status"] == "STOP_RESOURCE":
        require(summary["final_resource_decision"]["decision"] == "STOP_RESOURCE"
                and "cumulative_wall_cap" in summary["final_resource_decision"]["failures"], "STOP resource reason mismatch")
        require(analysis["status"] == "PARTIAL_OR_UNSTABLE_NO_FULL_BANK_CLAIM"
                and analysis["cost_assessment_A_only"]["C"]["status"].startswith("UNRESOLVED"), "Partial run makes full-bank scientific claim")
    process = load(base / "process_wall_receipt.json")
    cumulative = config["cumulative_prior_wall_seconds"] + process["process_wall_seconds"]
    close(load(base / "observed_resources.json")["full_process_seconds_including_shutdown"], process["process_wall_seconds"])
    require(cumulative <= 7200, "Observed cumulative process time exceeded cap")
    return {"schema": "followup-nt-full-public-verification-v1", "status": "PASS_PUBLIC_HASHES_DIMENSIONS_AND_PARTIAL_SCOPE",
        "verifier_sha256": sha(__file__), "manifest_sha256": sha(manifest_path),
        "source_run_status": summary["status"], "public_files": len(files), "public_bytes": manifest["bytes_excluding_self"],
        "model_count": 7, "model_normalizer_files": 22, "forecast_candidates": 128,
        "sealed_scientific_rows": len(rows), "missing_scientific_rows": len(missing),
        "recorded_intervals": sum(row["intervals"] for row in rows),
        "local_raw_pairs_reported": len(raw_pairs), "local_complete_probes_reported": len(full_groups),
        "actual_process_cumulative_seconds": cumulative,
        "private_absolute_path_hits": 0, "new_physics_calls": 0, "new_model_calls": 0,
        "verification_seconds": time.perf_counter() - started,
        "limits": "Public receipt/hash/dimension/arithmetic check only. Omitted full raw/complete-probe semantic reproduction audit is separate."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.base)
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
