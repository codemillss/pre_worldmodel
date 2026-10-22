#!/usr/bin/env python3
"""Check public N-T preflight evidence using NumPy, without model or physics."""
from __future__ import annotations
import argparse
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/followup/20261022-nt-preflight"


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.dtype.str.encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-8), (a, b)


def verify():
    manifest = load(BASE / "manifest.json")
    for row in manifest["files"]:
        p = (BASE / row["path"]).resolve()
        assert p.is_relative_to(BASE.resolve())
        assert p.stat().st_size == row["bytes"] and sha(p) == row["sha256"]
    assert manifest["files_excluding_self"] == len(manifest["files"])
    assert manifest["publisher_sha256"] == sha(ROOT / "tools/publish_nt_preflight.py")
    copies = {r["source_run_path"]: r for r in manifest["source_copy_records"]}
    entry, x, seal = [load(BASE / p) for p in ("entry.json", "summary.json", "prospective_forecast_seal.json")]
    config = load(ROOT / "configs/followup_nt_preflight.v2.json")
    assert sha(ROOT / "configs/followup_nt_preflight.v2.json") == entry["config_sha256"] == manifest["executed_config_sha256"]
    source_map = load(ROOT / "artifacts/followup/20261022-d1-d2/entry.json")["model_and_normalizer_sha256"]
    assert entry["model_and_normalizer_sha256"] == config["expected_model_normalizer_sha256"] == source_map
    for path, digest in {**entry["code_sha256"], **source_map}.items():
        assert sha(ROOT / path) == digest
    assert x["status"] == "PASS_NT_EXECUTOR_PREFLIGHT" and x["full128_actual_execution"] == "NOT_RUN"
    assert entry["full_input_typed_sha256"] == config["expected_input_typed_sha256"]
    assert entry["bank_hash"] == config["expected_candidate_hash"]
    with np.load(BASE / "bank.npz", allow_pickle=False) as z:
        bank = z["actions"].copy()
        assert bank.shape == (128, 5, 2) and bank.dtype == np.float64
        assert array_hash(bank) == entry["bank_hash"]
        assert np.array_equal(z["original_ids"], np.arange(128)) and z["sources"].shape == (128,)
    assert len(np.unique(bank.reshape(128, -1), axis=0)) == entry["unique_candidate_sequences"] == 128
    assert set(x["model_ids"]) == set(config["models"]) and len(config["models"]) == 7
    for name, meta in x["forecast_metadata"].items():
        with np.load(BASE / "forecasts" / f"prediction-{name}.npz", allow_pickle=False) as p:
            assert p["states"].shape == (128, 6, 10) and p["valid"].dtype == bool
            assert int(p["valid"].sum()) == meta["valid_count"]
            eligible = np.flatnonzero(p["valid"] & np.isfinite(p["total_cost"]))
            selected = None if not len(eligible) else int(eligible[np.argmin(p["total_cost"][eligible])])
            assert selected == meta["selected_original_candidate_id"]
    for rel, digest in {**seal["forecasts"], **seal["teacher"]}.items():
        assert copies[rel]["source_sha256"] == digest
        if rel.endswith(".npz"):
            assert sha(BASE / rel) == digest
    assert seal["before_any_new_actual_future"] is True
    assert all(v == 0 for v in seal["future_engine_calls_during_prediction"].values())
    assert all(v == 0 for v in x["future_engine_calls_during_prediction"].values())
    rows = x["rows"]
    expected = set(product([0, 9, 127], ["A", "B"], range(3)))
    assert len(rows) == 18 and {(r["candidate_id"], r["panel"], r["repeat"]) for r in rows} == expected
    assert sum(r["intervals"] for r in rows) == x["actual_physics_intervals"] == 111
    assert x["actual_physics_substeps"] == 22200
    for r in rows:
        assert r["candidate_hash"] == array_hash(bank[r["candidate_id"]])
        assert r["raw_roundtrip"]["status"] == "PASS_EXACT" and r["initial_snapshot_exact"]
        if r["repeat"] == 0:
            assert r["full_probe_roundtrip"]["status"] == "PASS_EXACT"
        else:
            assert r["full_probe_repeat_exact"]["status"] == "PASS_EXACT"
        if r["candidate_id"] < 16:
            assert r["source_D2_raw_exact"]["status"] == "PASS_EXACT"
        trial = BASE / "actual/N-T" / f"candidate-{r['candidate_id']:03d}" / r["panel"] / f"repeat-{r['repeat']}"
        assert load(trial / "timing_final.json")["timings_ns"] == r["timings_ns"]
        assert load(trial / "entry.json")["prospective_seal_sha256"] == copies["prospective_forecast_seal.json"]["source_sha256"]
        if r["panel"] == "B":
            assert r["cost"] is None and r["cost_components"] == {}
        elif r["cost_finite"]:
            close(sum(r["cost_components"].values()), r["cost"])
    for i, panel in product([0, 9, 127], ["A", "B"]):
        group = [r for r in rows if r["candidate_id"] == i and r["panel"] == panel]
        assert len({r["whole_probe_typed_sha256"] for r in group}) == 1
    with np.load(BASE / "teacher/source-original-D2.npz", allow_pickle=False) as source, np.load(BASE / "teacher/benchmark-repeated-prefix.npz", allow_pickle=False) as benchmark:
        assert source["start"].shape == (80, 10) and benchmark["start"].shape == (640, 10)
        indices = benchmark["source_row_index"]
        assert np.array_equal(indices, np.arange(640) % 80)
        for key in source.files:
            a, b = np.ascontiguousarray(source[key][indices]), np.ascontiguousarray(benchmark[key])
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    inventory = load(BASE / "local_evidence_inventory.json")
    assert inventory["raw_pair_count"] == len(inventory["records"]) == 18
    assert sum(len(r["raw_files"]) for r in inventory["records"]) == 36
    assert sum(bool(r["complete_probe_files"]) for r in inventory["records"]) == 6
    projection = load(BASE / "resource_projection.json")
    f = projection["future_seconds"]
    close(f["total"], sum(v for k, v in f.items() if k != "total"))
    close(projection["projected_cumulative_wall_seconds"], projection["cumulative_spent_wall_seconds"]+f["total"])
    close(projection["prior_experiment_process_wall_seconds"], config["base_reviewed_prior_wall_seconds"]+sum(r["wall_seconds"] for r in config["recovery_attempts"]))
    assert projection["wall_cap_fit"] is False and projection["next_actual_execution"] == "NOT_RUN"
    refined_path = ROOT / "artifacts/validation/nt_fullbank_resource_budget.v2.json"
    refined = load(refined_path)
    rates, reference_seconds, persistence_seconds = [], [], []
    for r in rows:
        t = r["timings_ns"]
        persist = t["first_repeat_complete_probe_durable_roundtrip_ns"]/1e9 if r["repeat"] == 0 else 0
        reference = t["source_D2_raw_read_and_exact_ns"]/1e9 if r["candidate_id"] < 16 else 0
        rates.append((t["trial_total_through_summary_ns"]/1e9-persist-reference)/r["intervals"])
        if r["candidate_id"] < 16:
            reference_seconds.append(reference)
        if r["repeat"] == 0:
            persistence_seconds.append(persist)
    factor = config["limits"]["safety_factor"]
    rate = max(config["limits"]["prior_worst_seconds_per_interval"], max(rates))
    loop = (x["stages_ns"]["actual_preflight_trials_and_all_verification_ns"]
        - sum(r["timings_ns"]["trial_total_through_summary_ns"] for r in rows))/1e9/18
    fixed = x["fixed_before_any_actual_seconds"]+x["fixed_after_actual_seconds"]+7*x["stages_ns"]["original_D2_teacher_input_authentication_ns"]/1e9
    future = factor*(4992*rate+96*max(reference_seconds)+256*max(persistence_seconds)+768*loop+fixed)+120
    expected_total = projection["cumulative_spent_wall_seconds"]+future
    close(refined["projected_cumulative_wall_seconds"], expected_total)
    close(refined["future_seconds"]["total"], future)
    assert refined["source_D2_reference_comparison_count"] == 96
    assert refined["projected_cumulative_bytes"] == projection["projected_cumulative_bytes"]
    assert refined["bindings"]["immutable_v1_resource_projection_sha256"] == sha(BASE / "resource_projection.json")
    assert refined["bindings"]["refinement_tool_sha256"] == sha(ROOT / "tools/refine_nt_resource_scope.py")
    assert refined["wall_cap_fit"] == (expected_total <= 7200) is True
    assert refined["next_actual_execution"] == "NOT_RUN"
    return {"schema": "nt-preflight-public-checks-v2", "status": "PASS_PUBLIC_HASHES_DIMENSIONS_AND_ARITHMETIC",
        "verifier_sha256": sha(__file__), "manifest_sha256": sha(BASE / "manifest.json"),
        "public_files": len(manifest["files"]), "forecast_candidates": 128,
        "model_count": 7, "model_normalizer_file_count": len(source_map), "actual_trials": 18, "actual_intervals": 111,
        "original_D2_exact_references_reported": 12, "complete_probe_groups_reported": 6,
        "teacher_source_rows": 80, "teacher_timing_benchmark_rows": 640,
        "scoped_resource_status": refined["status"], "scoped_cumulative_minutes": expected_total/60,
        "scoped_resource_budget_sha256": sha(refined_path),
        "new_physics_calls": 0, "new_model_calls": 0,
        "scope": "Public hashes and recorded exact verdicts/dimensions/selection/timing arithmetic only; local raw/fullProbe audit is separate. Original coarse budget remains preserved; refined applicability budget has its own bound receipt."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    result = verify()
    if a.output:
        with a.output.open("x") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
