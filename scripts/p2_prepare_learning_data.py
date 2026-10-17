"""P2-03: verify representation and train-only preparation on frozen pilot data."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import re
import resource
import shutil
import subprocess
import sys
from time import perf_counter
import tomllib
import traceback

import numpy as np

from push_align.config import ROOT
from push_align.data import _read_json, digest
from push_align.learning_data import load_learning_data, save_learning_data, validate_prepared
from push_align.normalization import fit_normalizer, learning_schema, load_normalizer
from push_align.preparation import prepare_dataset, SPLIT_CODES
from push_align.representation import encode, integrate, state_delta
from p2_episode_contract_check import source_bundle, save

DEFAULT_DATASET = ROOT/"runs/p2-02-20261007-01/data/p2-pilot-v1"
EXPECTED_DATASET_SHA = "e7f98e33ad46ce22c2fea648820f7c76a6bf27d216e0b403c22975fafe6695db"


def fit(prepared, *, features=None, deltas=None, identity=None):
    a, m = prepared.arrays, prepared.metadata
    return fit_normalizer(a["features"] if features is None else features,
                          a["deltas"] if deltas is None else deltas, a["split_code"], learning_schema(m["contract"]),
                          {k: m[k] for k in ("dataset_id", "manifest_sha256", "data_contract_hash")} if identity is None else identity,
                          a["episode_index"], a["transition_index"])


def mutation_checks(prepared, normalizer):
    a = prepared.arrays
    identity = deepcopy(normalizer.metadata["dataset_identity"])
    validation = a["split_code"] == SPLIT_CODES["validation"]
    if not validation.any():
        raise ValueError("Validation mutation check needs actual held-out rows")
    current, future = a["current_state"].copy(), a["next_state"].copy()
    current[validation, 0] += .0002
    future[validation, 0] += .0003
    features = encode(current, a["applied_action"]); deltas = state_delta(current, future)
    identity["manifest_sha256"] = hashlib.sha256(features.tobytes()+deltas.tobytes()).hexdigest()
    changed = fit(prepared, features=features, deltas=deltas, identity=identity)
    current2 = a["current_state"].copy()
    index = int(np.flatnonzero(a["split_code"] == 0)[0]); current2[index, 0] += .0002
    train_changed = fit(prepared, features=encode(current2, a["applied_action"]), deltas=state_delta(current2, a["next_state"]))
    return {"validation_rows_changed": int(validation.sum()), "train_rows_unchanged": True,
            "statistics_arrays_identical": all(np.array_equal(normalizer.arrays[k], changed.arrays[k]) for k in normalizer.arrays),
            "train_row_hash_identical": normalizer.metadata["train_rows_sha256"] == changed.metadata["train_rows_sha256"],
            "stats_hash_identical": normalizer.metadata["stats_hash"] == changed.metadata["stats_hash"],
            "full_provenance_hash_changes": normalizer.metadata["normalizer_hash"] != changed.metadata["normalizer_hash"],
            "train_mutation_positive_control_changes_stats_hash": normalizer.metadata["stats_hash"] != train_changed.metadata["stats_hash"],
            "scope": "in-memory counterfactual numeric fixture; source files unchanged; changed provenance is a synthetic digest, not another collected dataset"}


def rejection_checks(out, schema, identity, normalizer_hash):
    source = out/"learning_data/normalizer"; folder = out/"error_injection"; folder.mkdir()
    rows = []
    cases = []
    for name, key, value in (("column_order", "feature_fields", [schema["feature_fields"][1], schema["feature_fields"][0], *schema["feature_fields"][2:]]),
                             ("unit", "feature_units", ["cm", *schema["feature_units"][1:]]),
                             ("action_dt", "action_dt_s", .2), ("environment", "environment_hash", "0"*64)):
        changed = deepcopy(schema); changed[key] = value
        cases.append((name, source, changed, identity, normalizer_hash))
    changed = deepcopy(identity); changed["manifest_sha256"] = "0"*64
    cases.extend((("dataset_manifest", source, schema, changed, normalizer_hash),
                  ("expected_normalizer_hash", source, schema, identity, "0"*64)))
    for name in ("statistics_bytes", "metadata_statistics_hash"):
        target = folder/name; shutil.copytree(source, target)
        if name == "statistics_bytes":
            with (target/"statistics.npz").open("ab") as file: file.write(b"damaged")
        else:
            m = _read_json(target/"normalizer.json"); m["normalizer"]["stats_hash"] = "0"*64
            (target/"normalizer.json").write_text(json.dumps(m, allow_nan=False), encoding="utf-8")
        cases.append((name, target, schema, identity, normalizer_hash))
    for name, path, expected_schema, expected_identity, expected_hash in cases:
        try:
            load_normalizer(path, expected_schema, expected_identity, expected_hash)
        except ValueError as error:
            rows.append({"case": name, "rejected": True, "error_type": type(error).__name__, "message": str(error)})
        else:
            rows.append({"case": name, "rejected": False})
    save(folder/"results.json", rows)
    return rows


def run(args, out):
    started = perf_counter()
    dataset = args.dataset.resolve()
    dataset_sha = digest(dataset/"manifest.json")
    if dataset == DEFAULT_DATASET.resolve() and dataset_sha != EXPECTED_DATASET_SHA:
        raise ValueError("Previously adopted P2-02 dataset manifest changed")
    source_hashes = source_bundle(out)
    profile = tomllib.loads((ROOT/"configs/acceptance.v1.toml").read_text())
    numeric = profile["numeric"]
    save(out/"manifest.json", {"task_id": "P2-03", "created_at": datetime.now().astimezone().isoformat(),
                              "argv": sys.argv, "executable": sys.executable, "python": sys.version, "platform": platform.platform(),
                              "dataset_path": str(dataset), "dataset_manifest_sha256": dataset_sha,
                              "source_hashes": source_hashes, "acceptance_profile_sha256": digest(ROOT/"configs/acceptance.v1.toml"),
                              "predeclared_checks": ["all regression tests", "fully validated dataset stream", "train-only original-row statistics",
                                  "normalized float32 roundtrip", "float64 yaw reconstruction", "validation mutation invariance + train positive control",
                                  "stored pair reload", "incompatible schema/hash rejection", "source data unchanged"],
                              "numeric_tolerances": numeric, "dynamics_training": "NOT_RUN", "MPC": "NOT_RUN"})
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT, text=True, capture_output=True)
    (out/"tests.txt").write_text(tests.stdout+tests.stderr, encoding="utf-8")
    count = re.search(r"Ran (\d+) tests", tests.stdout+tests.stderr)
    if tests.returncode: raise ValueError("Regression checks failed; dataset transform not released")
    print(json.dumps({"progress": "regression_complete", "tests": int(count.group(1)) if count else None}), flush=True)
    tick = perf_counter(); prepared = prepare_dataset(dataset); preparation_s = perf_counter()-tick
    normalizer = fit(prepared); a = prepared.arrays
    save_learning_data(out/"learning_data", prepared, normalizer)
    schema, identity = normalizer.metadata["schema"], normalizer.metadata["dataset_identity"]
    loaded, restored = load_learning_data(out/"learning_data", schema, identity, normalizer.metadata["normalizer_hash"])
    roundtrips = {}
    for prefix, raw, normalize, denormalize in (("features", a["features"], normalizer.normalize_features, normalizer.denormalize_features),
                                              ("deltas", a["deltas"], normalizer.normalize_delta, normalizer.denormalize_delta)):
        p = "input" if prefix == "features" else "delta"
        reference = (raw-normalizer.arrays[f"{p}_mean"])/normalizer.arrays[f"{p}_scale"]
        if p == "input": reference[:, 2:4] = raw[:, 2:4]
        normalized = normalize(raw); decoded = denormalize(normalized); roundtrip = normalize(decoded)
        roundtrips[prefix] = {"raw_shape": list(raw.shape), "normalized_dtype": str(normalized.dtype),
                              "normalized_formula_allclose": bool(np.allclose(normalized, reference, atol=numeric["float32_atol"], rtol=numeric["float32_rtol"])),
                              "normalized_roundtrip_allclose": bool(np.allclose(roundtrip, normalized, atol=numeric["float32_atol"], rtol=numeric["float32_rtol"])),
                              "max_normalized_formula_abs_error": float(np.max(np.abs(normalized-reference))),
                              "max_physical_roundtrip_abs_error_by_column": np.max(np.abs(decoded-raw), axis=0).tolist(),
                              "max_physical_error_divided_by_train_scale": float(np.max(np.abs(decoded-raw)/normalizer.arrays[f"{p}_scale"]))}
    reconstructed = integrate(a["current_state"], a["deltas"])
    errors = state_delta(reconstructed, a["next_state"])
    angle = float(np.max(np.abs(errors[:, 2])))
    decoded_next = integrate(a["current_state"], normalizer.denormalize_delta(normalizer.normalize_delta(a["deltas"])))
    decoded_errors = state_delta(decoded_next, a["next_state"])
    mutations = mutation_checks(prepared, normalizer)
    rejections = rejection_checks(out, schema, identity, normalizer.metadata["normalizer_hash"])
    bundle_equal = all(np.array_equal(a[k], loaded.arrays[k]) for k in a) and all(np.array_equal(v, loaded.windows[k]) for k, v in prepared.windows.items()) and restored.metadata == normalizer.metadata
    parent_hash_path = dataset.parents[1]/"artifact_hashes.json"
    source_data_unchanged = digest(dataset/"manifest.json") == dataset_sha
    verified_source_artifacts = None
    if dataset == DEFAULT_DATASET.resolve():
        old = _read_json(parent_hash_path)["sha256"]
        verified_source_artifacts = len(old)
        source_data_unchanged &= all(digest(parent_hash_path.parent/path) == sha for path, sha in old.items())
    source_frozen = all(digest(ROOT/path) == sha for path, sha in source_hashes.items())
    checks = {"regression_tests": tests.returncode == 0, "prepared_pair_validated": bool(validate_prepared(prepared, normalizer)),
              "train_only_fit_count": normalizer.metadata["fit_count"] == int((a["split_code"] == 0).sum()),
              "fixed_angle_features": bool(np.array_equal(normalizer.arrays["input_mean"][2:4], [0., 0.]) and np.array_equal(normalizer.arrays["input_scale"][2:4], [1., 1.])),
              "normalized_formula_and_roundtrip": all(r["normalized_formula_allclose"] and r["normalized_roundtrip_allclose"] for r in roundtrips.values()),
              "float64_angle_reconstruction": angle <= numeric["angle_formula_atol_rad_float64"],
              "validation_stats_invariance": all(mutations[k] for k in ("statistics_arrays_identical", "train_row_hash_identical", "stats_hash_identical", "full_provenance_hash_changes", "train_mutation_positive_control_changes_stats_hash")),
              "stored_bundle_reload_identical": bundle_equal, "schema_hash_errors_rejected": all(r["rejected"] for r in rejections),
              "source_data_unchanged": bool(source_data_unchanged), "source_frozen": source_frozen}
    resources = {"device": "cpu", "model_training_performed": False, "preparation_wall_time_s": preparation_s,
                 "total_wall_time_s": perf_counter()-started, "self_peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                 "self_peak_rss_scope": "main process only; excludes regression child peak", "output_learning_bytes": sum(p.stat().st_size for p in (out/"learning_data").rglob("*") if p.is_file()),
                 "scope": "raw dataset validation and transformation; not neural training/inference timing"}
    save(out/"numeric_checks.json", {"roundtrips": roundtrips, "float64_reconstruction_max_error_by_column": np.max(np.abs(errors), axis=0).tolist(),
                                    "float32_decoded_state_max_error_by_column": np.max(np.abs(decoded_errors), axis=0).tolist(),
                                    "numeric_tolerances": numeric, "validation_mutations": mutations, "rejections": rejections})
    save(out/"resources.json", resources)
    summary = {"task_id": "P2-03", "stage_verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
               "tests_run": int(count.group(1)) if count else None, "episode_count": prepared.metadata["episode_count"],
               "valid_transitions": len(a["features"]), "train_fit_count": normalizer.metadata["fit_count"],
               "split_counts": prepared.metadata["split_counts"], "window_counts": prepared.metadata["window_counts"],
               "schema_hash": normalizer.metadata["schema_hash"], "train_rows_sha256": normalizer.metadata["train_rows_sha256"],
               "stats_hash": normalizer.metadata["stats_hash"], "normalizer_hash": normalizer.metadata["normalizer_hash"],
               "dataset_manifest_sha256": dataset_sha, "verified_source_artifacts": verified_source_artifacts,
               "float64_angle_reconstruction_max_rad": angle, "validation_mutations": mutations, "resources": resources,
               "alignment_outcome": "NOT_EVALUATED", "learner_understanding": "NOT_ASSESSED", "dynamics_training": "NOT_RUN", "MPC": "NOT_RUN",
               "next_task": "P2-04 one-step MLP training + free rollout", "scope": "P2-03 transforms and provenance only; no model accuracy or task success claim"}
    save(out/"summary.json", summary)
    print(json.dumps({"verdict": summary["stage_verdict"], "tests": summary["tests_run"], "transitions": len(a["features"]), "fit_train": normalizer.metadata["fit_count"]}), flush=True)
    return all(checks.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    try:
        passed = run(args, out)
    except Exception as error:
        save(out/"failure.json", {"exception": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        if not (out/"summary.json").exists():
            save(out/"summary.json", {"task_id": "P2-03", "stage_verdict": "ERROR", "reason": str(error), "dynamics_training": "NOT_RUN", "MPC": "NOT_RUN"})
        print(json.dumps({"verdict": "ERROR", "reason": str(error)}), flush=True); passed = False
    save(out/"artifact_hashes.json", {"sha256": {str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()}})
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__": main()
