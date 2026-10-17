"""Read-only audit of frozen data wiring, equal exposure and saved training state.

This checker independently reconstructs tensors and the sampler stream. It never
imports the training runner, instantiates a model, calls physics, or updates an
optimizer. Logged disposable-update proofs are checked, not re-executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("R", "D00", "D10", "D01", "D11")
MAIN = ROOT / "runs/p4-data-main-20261013-01"
OLD = ROOT / "runs/p4-02-20261012-01"
HIST = OLD / "seeds/seed-11/SK-lambda0.1"
LEARNING = ROOT / "runs/p3-02-20261009-01/learning_data"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def tree_hash(value):
    """Independently implement the historical typed identity byte format."""
    h = hashlib.sha256()
    def feed(item):
        if torch.is_tensor(item):
            h.update(b"tensor" + str(item.dtype).encode() + json.dumps(list(item.shape)).encode())
            h.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            h.update(b"dict")
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))): feed(key); feed(item[key])
        elif isinstance(item, (tuple, list)):
            h.update(type(item).__name__.encode())
            for element in item: feed(element)
        else:
            h.update(type(item).__name__.encode() + json.dumps(item, sort_keys=True, allow_nan=False).encode())
    feed(value)
    return h.hexdigest()


def array_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind in "fc")


def npz(path):
    with np.load(path, allow_pickle=False) as f: return {k: f[k].copy() for k in f.files}


def tensors(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def audit(run):
    checks = []
    def check(name, passed, **details):
        row = {"name": name, "pass": bool(passed), **details}; checks.append(row)
        if not passed: raise ValueError(name + ": " + str(details))

    result = {"schema": "p4-data-retrain-training-review-v1", "status": "FAIL",
              "run": run.relative_to(ROOT).as_posix(), "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "review_source_sha256": sha(__file__), "checks": checks,
              "execution_during_review": {"model_instances": 0, "model_forwards": 0,
                                           "optimizer_updates": 0, "physics_steps": 0},
              "scope": "Data wiring, saved-state and recorded-budget audit; no prediction-effect or task-success conclusion"}
    try:
        summary, manifest = read(run / "summary.json"), read(run / "manifest.json")
        index = read(run / "training_artifact_hashes.json")["sha256"]
        check("completed_training_pass", summary["status"] == "PASS" and len(summary["arms"]) == 5)
        mismatches = [rel for rel, value in index.items() if not (run / rel).is_file() or sha(run / rel) != value]
        check("all_saved_training_artifacts_hash_exact", not mismatches, files=len(index), mismatches=mismatches)
        check("manifest_bound_in_summary", summary["manifest_sha256"] == sha(run / "manifest.json"))
        registration = read(ROOT / manifest["evaluation_registration"])
        check("execution_registration_hash", sha(ROOT / manifest["evaluation_registration"]) == manifest["evaluation_registration_sha256"])
        all_bound = {**registration["bindings_sha256"], **manifest["source_sha256"], **manifest["historical_reference_sha256"]}
        mismatch = [rel for rel, value in all_bound.items() if sha(ROOT / rel) != value]
        check("all_source_and_upstream_hashes_exact", not mismatch, files=len(all_bound), mismatches=mismatch)
        check("registered_budget_and_model_unchanged", all(canonical(registration[k]) == canonical(manifest["design"][k])
              for k in ("model", "training", "evaluation")))
        bindings = read(run / "dataset_bindings.json")
        check("prepared_and_bindings_hashes", sha(run / "dataset_bindings.json") == manifest["dataset_bindings_sha256"]
              and sha(run / "prepared_training.npz") == manifest["prepared_training_sha256"])
        check("manifest_matches_actual_bindings", canonical(bindings) == canonical(manifest["training_provenance"]))
        check("no_development_or_warmup_loaded", bindings["train_core_episodes_loaded"] == 96
              and bindings["development_episodes_loaded"] == bindings["warmup_episodes_loaded"] == 0)
        mismatch = [rel for rel, value in bindings["raw_core_artifact_sha256"].items() if sha(MAIN / rel) != value]
        check("all_bound_train_raw_hashes_exact", not mismatch, files=len(bindings["raw_core_artifact_sha256"]), mismatches=mismatch)
        matrix = npz(run / "prepared_training.npz")
        original = npz(LEARNING / "prepared.npz")
        original_manifest = read(LEARNING / "manifest.json")
        check("original_prepared_hash", sha(LEARNING / "prepared.npz") == original_manifest["npz_sha256"])
        old_rows = npz(OLD / "train_windows.npz")["window_rows"]
        check("original_window_rows_exact", array_equal(matrix["old_window_rows"], old_rows) and old_rows.shape == (2960, 5))
        check("original_window_rows_train_only", np.all(original["split_code"][old_rows] == 0))
        expected_states = np.concatenate((original["current_state"][old_rows[:, 0], None], original["next_state"][old_rows]), axis=1)
        expected_actions = original["applied_action"][old_rows]
        check("R_all_original_tensor_values_exact", array_equal(matrix["R_states"], expected_states)
              and array_equal(matrix["R_actions"], expected_actions))
        slots_document = read(MAIN / "slots_bound.json")
        slots = np.asarray([row["slot"] for row in slots_document["entries"]], dtype=np.int64)
        unchanged = np.setdiff1d(np.arange(2960), slots)
        check("384_frozen_slots_and2576_old_slots", array_equal(slots, matrix["replacement_slots"])
              and len(slots) == len(np.unique(slots)) == 384 and len(unchanged) == 2576)
        plans = {p["episode_id"]: p for p in read(MAIN / "registration.json")["plans"]}
        raw_rows = {r["label"]: r for r in read(MAIN / "summary.json")["episodes"]}
        raw_cache = {}
        for arm in ARMS:
            for key, shape, dtype in (("states", (2960, 6, 10), np.float64), ("actions", (2960, 5, 2), np.float64),
                                      ("mask", (2960, 5), np.bool_)):
                value = matrix[arm + "_" + key]
                check(arm + "_" + key + "_shape_finite", value.shape == shape and value.dtype == dtype and np.isfinite(value).all())
            check(arm + "_all_recorded_transitions_in_denominator", matrix[arm + "_mask"].all())
            values = {"states": torch.from_numpy(matrix[arm + "_states"]),
                      "applied_actions": torch.from_numpy(matrix[arm + "_actions"]),
                      "mask": torch.from_numpy(matrix[arm + "_mask"])}
            check(arm + "_actual_tensor_hash", tree_hash(values) == bindings["arms"][arm]["data_hash"])
            if arm == "R": continue
            check(arm + "_2576_old_slots_exact", array_equal(matrix[arm + "_states"][unchanged], expected_states[unchanged])
                  and array_equal(matrix[arm + "_actions"][unchanged], expected_actions[unchanged]))
            for rank, entry in enumerate(slots_document["entries"]):
                episode = entry["arm_episodes"][arm]; plan = plans[episode]
                check(f"{arm}_slot{entry['slot']}_train_family_and_start", plan["split"] == plan["semantic_split"] == "train"
                      and plan["training_eligible"] is True and not plan["warmup_training_eligible"]
                      and plan["family_rank"] == rank // 16 and entry["window_start"] == rank % 16
                      and plan["family_id"] == entry["family_id"] and raw_rows[episode]["role"] == "core"
                      and plan["y_condition"] == "Y" + arm[1] and plan["stop_condition"] == "S" + arm[2])
                if episode not in raw_cache:
                    folder = MAIN / raw_rows[episode]["dataset"]
                    raw_entry = read(folder / "manifest.json")["episodes"][0]
                    with np.load(folder / raw_entry["npz"], allow_pickle=False) as f:
                        raw_cache[episode] = {k: f[k].copy() for k in ("state", "action_applied", "transition_valid")}
                raw = raw_cache[episode]; start, slot = entry["window_start"], entry["slot"]
                check(f"{arm}_slot{slot}_exact_raw_core_window", array_equal(matrix[arm + "_states"][slot], raw["state"][start:start+6])
                      and array_equal(matrix[arm + "_actions"][slot], raw["action_applied"][start:start+5])
                      and raw["transition_valid"][start:start+5].all())
        check("only24_train_families96_core_episodes", len(raw_cache) == 96
              and len({plans[e]["family_id"] for e in raw_cache}) == 24)

        historical = npz(HIST / "training_trace.npz")
        common_path = ROOT / manifest["initial_checkpoint"]["path"]
        common_state = tensors(common_path / "training_state.pt")
        common_weights = tensors(common_path / "weights.pt")
        generator = torch.Generator(device="cpu"); generator.set_state(common_state["shuffle_rng_state"])
        permutation, cursor, epoch = torch.randperm(2960, generator=generator), 0, 1
        expected_positions, offsets, sampler_boundaries = [], [0], {}
        for update in range(1, 1001):
            if cursor == 2960:
                permutation = torch.randperm(2960, generator=generator); cursor = 0; epoch += 1
            stop = min(cursor + 64, 2960); positions = permutation[cursor:stop].numpy().copy()
            expected_positions.append(positions); offsets.append(offsets[-1] + len(positions)); cursor = stop
            if update % 100 == 0:
                sampler_boundaries[update] = {"epoch": epoch, "cursor": cursor, "permutation": permutation.clone(),
                                              "shuffle_rng_state": generator.get_state().clone(), "occurrences": offsets[-1]}
        flat = np.concatenate(expected_positions); offsets = np.asarray(offsets, dtype=np.int64)
        check("independently_regenerated_historical_shuffle", array_equal(flat, historical["window_positions"])
              and array_equal(offsets, historical["batch_offsets"]))
        exposures = np.bincount(flat, minlength=2960)
        partial = np.concatenate([p for p in expected_positions if len(p) == 16])
        check("new_slots21_occurrences_and_no_partial_batches", np.all(exposures[slots] == 21) and not np.isin(slots, partial).any())
        check("whole_exposure_979full21partial62992windows314960transitions", len(flat) == 62992
              and int((np.diff(offsets) == 64).sum()) == 979 and int((np.diff(offsets) == 16).sum()) == 21)
        proofs, checkpoints = 0, 0
        arm_evidence = []
        historical_start = read(HIST / "start_identity.json")
        for arm in ARMS:
            folder = run / "arms" / arm; arm_summary = read(folder / "summary.json")
            check(arm + "_same_start_identity", read(folder / "start_identity.json") == historical_start == arm_summary["start_identity"])
            trace = npz(folder / "training_trace.npz")
            check(arm + "_entire_batch_order_exact", array_equal(trace["window_positions"], flat)
                  and array_equal(trace["batch_offsets"], offsets))
            check(arm + "_losses1000x4finite", trace["losses"].shape == (1000, 4) and np.isfinite(trace["losses"]).all())
            actual_hashes = []
            for p in expected_positions:
                values = {"states": torch.from_numpy(matrix[arm + "_states"][p]),
                          "applied_actions": torch.from_numpy(matrix[arm + "_actions"][p]),
                          "mask": torch.from_numpy(matrix[arm + "_mask"][p])}
                actual_hashes.append(tree_hash(values))
            check(arm + "_every_actual_batch_hash_recomputed", np.array_equal(trace["batch_data_hashes"], actual_hashes))
            if arm == "R":
                check("R1000_losses_gradients_exact_historical", array_equal(trace["losses"], historical["losses"]))
                check("R1000_batch_data_hashes_exact_historical", array_equal(trace["batch_data_hashes"], historical["batch_data_hashes"]))
            for update in range(100, 1001, 100):
                checkpoint = folder / "checkpoints" / f"update-{update:04d}"
                meta = read(checkpoint / "checkpoint.json"); state = tensors(checkpoint / "training_state.pt")
                weights = tensors(checkpoint / "weights.pt"); sam = state["sampler"]; expected = sampler_boundaries[update]
                check(f"{arm}_{update}_checkpoint_bindings", meta["arm"] == arm and meta["additional_updates"] == update
                      and meta["total_updates"] == 1300+update and meta["manifest_sha256"] == sha(run / "manifest.json")
                      and meta["dataset_hash"] == bindings["arms"][arm]["data_hash"]
                      and meta["initial_checkpoint_hash"] == manifest["initial_checkpoint"]["checkpoint_hash"]
                      and meta["normalizer_hash"] == manifest["normalizer_hash"] and meta["seed"] == 11 and meta["rollout_weight"] == .1)
                check(f"{arm}_{update}_checkpoint_hash", meta["checkpoint_hash"] == hashlib.sha256(canonical({k:v for k,v in meta.items() if k != "checkpoint_hash"})).hexdigest()
                      and all(sha(checkpoint / name) == value for name, value in meta["files"].items()))
                check(f"{arm}_{update}_architecture_dtype_finite", weights.keys() == common_weights.keys()
                      and all(weights[k].shape == common_weights[k].shape and weights[k].dtype == torch.float32
                              and bool(torch.isfinite(weights[k]).all()) for k in weights))
                check(f"{arm}_{update}_optimizer_full_state", state["optimizer"]["param_groups"] == common_state["optimizer"]["param_groups"]
                      and state["optimizer"]["state"].keys() == common_state["optimizer"]["state"].keys()
                      and all(float(v["step"]) == 1300+update and v["step"].dtype == torch.float32
                              and all(v[k].dtype == torch.float32 and bool(torch.isfinite(v[k]).all()) for k in ("exp_avg", "exp_avg_sq"))
                              and bool((v["exp_avg_sq"] >= 0).all()) for v in state["optimizer"]["state"].values()))
                check(f"{arm}_{update}_sampler_reconstructed", sam["updates"] == update and sam["window_count"] == 2960
                      and sam["batch_windows"] == 64 and sam["max_updates"] == 1000 and sam["horizon"] == 5
                      and sam["drop_last"] is False and sam["epoch"] == expected["epoch"] and sam["cursor"] == expected["cursor"]
                      and sam["window_occurrences"] == expected["occurrences"] and sam["transition_occurrences"] == expected["occurrences"]*5
                      and torch.equal(sam["permutation"], expected["permutation"])
                      and torch.equal(sam["shuffle_rng_state"], expected["shuffle_rng_state"]))
                check(f"{arm}_{update}_torch_rng", torch.equal(state["torch_rng_state"], common_state["torch_rng_state"]))
                if arm == "R":
                    old_cp = HIST / "checkpoints" / f"update-{update:04d}"
                    check(f"R_{update}_weights_adam_sampler_rng_exact_history", tree_hash(weights) == tree_hash(tensors(old_cp / "weights.pt"))
                          and tree_hash(state) == tree_hash(tensors(old_cp / "training_state.pt")))
                proof = read(folder / f"resume-{update:04d}.json")
                check(f"{arm}_{update}_recorded_resume_proof", proof["pass"] and proof["disposable_updates"] == 3
                      and proof["formal_state_unchanged"] and proof["next_model_adam_rng_sampler_exact"])
                proofs += 3; checkpoints += 1
            check(arm + "_final1000_only", arm_summary["status"] == "PASS" and arm_summary["additional_updates"] == 1000
                  and arm_summary["total_Adam_steps"] == 2300 and arm_summary["checkpoint_selection"] == "final1000_only"
                  and len(arm_summary["checkpoints"]) == 10)
            arm_evidence.append({"arm": arm, "data_hash": bindings["arms"][arm]["data_hash"],
                                 "updates": 1000, "window_occurrences": len(flat),
                                 "new_window_occurrences": 0 if arm == "R" else int(exposures[slots].sum()),
                                 "final_checkpoint_hash": arm_summary["final_checkpoint_hash"]})
        reproduction = run / "arms/R/historical_validation_reproduction"
        reproduced, historical_prediction = npz(reproduction / "traces.npz"), npz(HIST / "evaluations/update-1000/traces.npz")
        check("R_all_saved_original_prediction_traces_exact", reproduced.keys() == historical_prediction.keys()
              and all(array_equal(reproduced[k], historical_prediction[k]) for k in reproduced))
        check("R_saved_original_prediction_summary_exact", canonical(read(reproduction / "summary.json"))
              == canonical(read(HIST / "evaluations/update-1000/summary.json")))
        probe = read(run / "train_only_probe.json")
        check("recorded_train_only_and_positive_control9", probe["status"] == "PASS" and all(probe["checks"].values())
              and probe["disposable_updates"] == 9)
        check("formal5000_disposable159_and50checkpoints", summary["counters"] == {
            "formal_updates": 5000, "formal_update_attempts": 5000,
            "disposable_updates": 159, "disposable_update_attempts": 159} and checkpoints == 50 and proofs+9 == 159)
        check("no_physics_final_test_or_task_claim", summary["physics"]["attempted_physics_calls"] == 0
              and summary["final_test_used"] is False and summary["P5_01"] == "HOLD" and summary["task_success"] == "NOT_EVALUATED")
        training_bytes = sum((run / rel).stat().st_size for rel in index) + (run / "training_artifact_hashes.json").stat().st_size
        check("training_output_below256MiB", training_bytes < 256*1024**2, bytes=training_bytes)
        result.update(status="PASS", arm_evidence=arm_evidence, observed={"formal_updates": 5000,
                      "recorded_disposable_updates": 159, "checkpoints": checkpoints, "training_bytes": training_bytes,
                      "raw_train_episodes": len(raw_cache), "source_and_upstream_files": len(all_bound)},
                      manifest_sha256=sha(run / "manifest.json"), summary_sha256=sha(run / "summary.json"),
                      training_index_sha256=sha(run / "training_artifact_hashes.json"),
                      limitations=["Disposable updates were not re-executed; stored proofs and counters were verified.",
                                   "No prediction was generated in this review; R stored traces were compared exactly."])
    except Exception as error:
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
    result["check_count"] = len(checks)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); result = audit(args.run.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f: json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"status": result["status"], "checks": result["check_count"], "failure": result.get("failure")}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__": sys.exit(main())
