"""Execute the frozen five-arm, same-model, same-budget data intervention.

R must exactly reproduce historical weights/Adam/RNG/sampling/loss/predictions
before any intervention arm starts. Every failure preserves its prefix; no
automatic retries, selection, new physical rollout, or data normalization.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import resource
import shutil
import sys
from time import perf_counter
import tomllib
import traceback

import mujoco
import numpy as np
import torch

from push_align.checkpoint import make_optimizer
from push_align.config import ROOT
from push_align.data import digest
from push_align.data_intervention_training import (
    ARMS, data_hash, json_hash, load_intervention_checkpoint, materialize_arms,
    read_json, require, save_intervention_checkpoint, tree_hash)
from push_align.fork_checkpoint import load_fork_checkpoint
from push_align.fork_sampler import WindowSampler
from push_align.fork_training import train_sequences
from push_align.learning_data import load_learning_data
from push_align.prediction import evaluate_predictions
from push_align.sequence_training import sequence_update
from p2_train_pilot import save, save_evaluation, tensor_tree_equal
from p3_train_common_mlp import validate_settings
from p4_train_forks import restore_common, start_identity

MAIN = ROOT / "runs/p4-data-main-20261013-01"
DESIGN = ROOT / "configs/p4_data_intervention.design.v1.json"
LEARNING = ROOT / "runs/p3-02-20261009-01/learning_data"
COMMON = ROOT / "runs/p3-04-20261009-01"
HISTORICAL_RUN = ROOT / "runs/p4-02-20261012-01"
HISTORICAL = HISTORICAL_RUN / "seeds/seed-11/SK-lambda0.1"
MIB, GIB = 1024**2, 1024**3


def storage_guard(out):
    stored = sum(p.stat().st_size for p in Path(out).rglob("*") if p.is_file())
    free = shutil.disk_usage(ROOT).free
    require(stored <= 256*MIB and free >= 2*GIB, "256MiB training/evaluation or 2GiB reserve guard exceeded")
    return {"run_bytes": stored, "free_bytes": free}


def check_registration(path, expected_sha256, run_id):
    require(digest(path) == expected_sha256, "Execution registration hash differs")
    registration = read_json(path); design = read_json(DESIGN)
    require(registration["run_id"] == run_id and registration["design_sha256"] == digest(DESIGN),
            "Execution run/design identity differs")
    require(registration["collection_registration_sha256"] == digest(MAIN / "registration.json")
            and registration["slots_bound_sha256"] == digest(MAIN / "slots_bound.json"), "Collection binding differs")
    for field in ("model", "training", "evaluation"):
        require(json_hash(registration[field]) == json_hash(design[field]), "Execution changes frozen " + field)
    for rel, expected in registration["bindings_sha256"].items():
        require(digest(ROOT / rel) == expected, "Execution bound file changed: " + rel)
    return registration, design


def source_hashes():
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if file:
            path = Path(file).resolve()
            # Torch exposes virtual modules with synthetic relative __file__
            # names (for example _classes.py), which are not source artifacts.
            if path.is_file() and path.suffix == ".py" and path.is_relative_to(ROOT) and ".venv" not in path.parts:
                paths.add(path)
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def forbid_physics():
    counts = {"attempted_physics_calls": 0}
    def forbidden(*args, **kwargs):
        counts["attempted_physics_calls"] += 1
        raise RuntimeError("Training/evaluation forbids future physics integration")
    for name in ("mj_step", "mj_step1", "mj_step2"):
        setattr(mujoco, name, forbidden)
    return counts


def resume_proof(checkpoint, model, optimizer, sampler, normalizer, config, manifest_sha, tensors, counters):
    """Three discarded next updates, never assigned back to the formal arm."""
    before = {"weights": tree_hash(model.state_dict()), "Adam": tree_hash(optimizer.state_dict()),
              "sampler": tree_hash(sampler.state_dict()), "rng": tree_hash(torch.get_rng_state())}
    copies = [load_intervention_checkpoint(checkpoint, normalizer, config,
               expected_manifest_sha256=manifest_sha, resume=True) for _ in range(2)]
    require(all(tree_hash(c.model.state_dict()) == before["weights"]
                and tree_hash(c.optimizer.state_dict()) == before["Adam"]
                and tree_hash(c.sampler.state_dict()) == before["sampler"]
                and tree_hash(c.torch_rng_state) == before["rng"] for c in copies), "Checkpoint reload differs")
    diagnostics, results = [], []
    with torch.random.fork_rng(devices=[]):
        reference = deepcopy(model)
        refopt = make_optimizer(reference, config); refopt.load_state_dict(deepcopy(optimizer.state_dict()))
        samplers = [sampler.clone_for_resume_probe(), *(c.sampler.clone_for_resume_probe() for c in copies)]
        positions = [s.next_batch() for s in samplers]
        require(all(torch.equal(positions[0], p) for p in positions[1:]), "Resume next batch differs")
        states, actions, mask = tensors
        second_before = (tree_hash(copies[1].model.state_dict()), tree_hash(copies[1].optimizer.state_dict()))
        for index, (current, opt, rng) in enumerate([(reference, refopt, torch.get_rng_state().clone()),
                (copies[0].model, copies[0].optimizer, copies[0].torch_rng_state),
                (copies[1].model, copies[1].optimizer, copies[1].torch_rng_state)]):
            torch.set_rng_state(rng)
            counters["disposable_update_attempts"] += 1
            require(counters["disposable_update_attempts"] <= 160, "Disposable update cap exceeded")
            p = positions[index]
            diagnostics.append(sequence_update(current, opt, normalizer, states[p], actions[p], mask[p], .1, 1.))
            counters["disposable_updates"] += 1
            results.append({"weights": tree_hash(current.state_dict()), "Adam": tree_hash(opt.state_dict()),
                            "rng": tree_hash(torch.get_rng_state()), "sampler": tree_hash(samplers[index].state_dict())})
            if index == 1:
                require(second_before == (tree_hash(copies[1].model.state_dict()), tree_hash(copies[1].optimizer.state_dict())),
                        "Restored model copies share mutable state")
    after = {"weights": tree_hash(model.state_dict()), "Adam": tree_hash(optimizer.state_dict()),
             "sampler": tree_hash(sampler.state_dict()), "rng": tree_hash(torch.get_rng_state())}
    require(results[0] == results[1] == results[2] and diagnostics[0] == diagnostics[1] == diagnostics[2]
            and before == after, "Disposable resume proof failed or mutated formal state")
    return {"pass": True, "disposable_updates": 3, "next_loss": diagnostics[0],
            "formal_state_unchanged": True, "next_model_adam_rng_sampler_exact": True}


def historical_checkpoint_comparison(update, model, optimizer, sampler, normalizer, config, manifest):
    old = load_fork_checkpoint(HISTORICAL / "checkpoints" / f"update-{update:04d}", normalizer,
        config, manifest["provenance"], manifest["budget"], seed=11, branch="SK-lambda0.1", resume=True,
        expected_run_manifest_sha256=digest(HISTORICAL_RUN / "manifest.json"))
    checks = {"model": tensor_tree_equal(model.state_dict(), old.model.state_dict()),
              "Adam": tensor_tree_equal(optimizer.state_dict(), old.optimizer.state_dict()),
              "sampler": tensor_tree_equal(sampler.state_dict(), old.sampler.state_dict()),
              "Torch_RNG": torch.equal(torch.get_rng_state(), old.torch_rng_state)}
    require(all(checks.values()), f"R does not exactly reproduce historical checkpoint {update}: {checks}")
    return checks


def train_only_probe(datasets, slots, prepared, historical_trace, entry, normalizer, config,
                     common_manifest, settings, counters):
    """Nine disposable updates: validation perturbation versus train-positive control."""
    changed = deepcopy(prepared)
    validation = changed.arrays["split_code"] == 1
    for field in ("current_state", "next_state"):
        changed.arrays[field][validation, 0] += .01
    rows, states, actions, mask = train_sequences(changed)
    val_changed, _, _ = materialize_arms(MAIN, rows, states, actions, mask,
        historical_trace["window_positions"], historical_trace["batch_offsets"])
    positive = tuple(t.clone() for t in datasets["D11"])
    positive[0][slots, :, 6] += .002
    results = []
    with torch.random.fork_rng(devices=[]):
        for tensors in (datasets["D11"], val_changed["D11"], positive):
            bundle = restore_common(entry, normalizer, config, common_manifest, settings)
            torch.set_rng_state(bundle.torch_rng_state)
            sampler = WindowSampler(2960, 64, bundle.shuffle_generator, max_updates=1000, horizon=5)
            diagnostics, positions_all = [], []
            for _ in range(3):
                positions = sampler.next_batch(); positions_all.append(positions.tolist())
                counters["disposable_update_attempts"] += 1
                diagnostics.append(sequence_update(bundle.model, bundle.optimizer, normalizer,
                    *(t[positions] for t in tensors), .1, 1.))
                counters["disposable_updates"] += 1
            results.append({"data_hash": data_hash(*tensors), "model": tree_hash(bundle.model.state_dict()),
                            "Adam": tree_hash(bundle.optimizer.state_dict()), "diagnostics": diagnostics,
                            "positions": positions_all})
    checks = {"validation_only_change_train_tensor_and_updates_exact": results[0] == results[1],
              "new_train_data_change_changes_tensor_and_weights": results[0]["data_hash"] != results[2]["data_hash"]
                  and results[0]["model"] != results[2]["model"],
              "positive_control_preserves_sampler": results[0]["positions"] == results[2]["positions"]}
    require(all(checks.values()), "Training-only invariance or positive control failed")
    return {"status": "PASS", "checks": checks, "disposable_updates": 9,
            "scope": "Original validation-only perturbation cannot change D11 train data/updates; train core perturbation does"}


def train_arm(arm, tensors, entry, normalizer, config, common_manifest, settings,
              historical_manifest, historical_trace, manifest_sha, dataset_hash, out, counters, prepared):
    folder = out / "arms" / arm; folder.mkdir(parents=True, exist_ok=False)
    bundle = restore_common(entry, normalizer, config, common_manifest, settings)
    model, optimizer = bundle.model, bundle.optimizer
    torch.set_rng_state(bundle.torch_rng_state)
    sampler = WindowSampler(2960, 64, bundle.shuffle_generator, max_updates=1000, horizon=5)
    start = start_identity(bundle, sampler); save(folder / "start_identity.json", start)
    require(start == read_json(HISTORICAL / "start_identity.json"), "Common start identity differs")
    states, actions, mask = tensors
    losses, times, hashes, positions_all, offsets, checkpoints = [], [], [], [], [0], []
    tick = perf_counter()
    try:
        for update in range(1, 1001):
            positions = sampler.next_batch()
            a, b = historical_trace["batch_offsets"][update-1:update+1]
            require(np.array_equal(positions.numpy(), historical_trace["window_positions"][a:b]), "Historical batch order differs")
            current = (states[positions], actions[positions], mask[positions])
            batch_hash = data_hash(*current)
            if arm == "R": require(batch_hash == historical_trace["batch_data_hashes"][update-1], "R historical batch data differ")
            began = perf_counter(); counters["formal_update_attempts"] += 1
            diagnostic = sequence_update(model, optimizer, normalizer, *current, .1, 1.)
            counters["formal_updates"] += 1
            values = [diagnostic[k] for k in ("one_step_loss", "rollout_loss", "loss_before_update", "gradient_norm_before_clip")]
            losses.append(values); times.append(perf_counter()-began); hashes.append(batch_hash)
            positions_all.append(positions.numpy().copy()); offsets.append(offsets[-1]+len(positions))
            require(diagnostic["valid_transitions"] == len(positions)*5 and diagnostic["scalar_terms"] == len(positions)*50,
                    "Training denominator differs")
            if arm == "R": require(np.array_equal(values, historical_trace["losses"][update-1]), f"R loss differs at update {update}")
            if update % 100 == 0:
                checkpoint = folder / "checkpoints" / f"update-{update:04d}"
                metadata = save_intervention_checkpoint(checkpoint, model, optimizer, normalizer, config,
                    arm=arm, sampler=sampler, manifest_sha256=manifest_sha, dataset_hash=dataset_hash,
                    initial_checkpoint_hash=entry["checkpoint_hash"])
                history = (historical_checkpoint_comparison(update, model, optimizer, sampler, normalizer, config,
                           historical_manifest) if arm == "R" else None)
                proof = resume_proof(checkpoint, model, optimizer, sampler, normalizer, config, manifest_sha, tensors, counters)
                save(folder / f"resume-{update:04d}.json", proof)
                checkpoints.append({"update": update, "checkpoint": checkpoint.relative_to(out).as_posix(),
                                    "checkpoint_hash": metadata["checkpoint_hash"], "historical_exact": history,
                                    "resume_pass": proof["pass"]})
                storage_guard(out)
                print(json.dumps({"progress": "checkpoint", "arm": arm, "update": update}), flush=True)
        prediction_exact = None
        if arm == "R":
            evaluation = evaluate_predictions(model, normalizer, prepared, horizons=(1, 5, 10))
            save_evaluation(folder / "historical_validation_reproduction", evaluation)
            with np.load(HISTORICAL / "evaluations/update-1000/traces.npz", allow_pickle=False) as old:
                prediction_exact = set(old.files) == set(evaluation.traces) and all(
                    np.array_equal(old[k], evaluation.traces[k], equal_nan=old[k].dtype.kind in "fc") for k in old.files)
            # The exact summary uses the same original validation denominator.
            summary_exact = json_hash(evaluation.summary) == json_hash(read_json(HISTORICAL / "evaluations/update-1000/summary.json"))
            require(prediction_exact and summary_exact, "R historical prediction traces/summary differ")
        result = {"arm": arm, "status": "PASS", "additional_updates": 1000,
                  "total_Adam_steps": 2300, "checkpoints": checkpoints,
                  "final_checkpoint": (folder / "checkpoints/update-1000").relative_to(out).as_posix(),
                  "final_checkpoint_hash": checkpoints[-1]["checkpoint_hash"],
                  "dataset_hash": dataset_hash, "start_identity": start,
                  "window_occurrences": sampler.window_occurrences,
                  "transition_occurrences": sampler.transition_occurrences,
                  "historical_prediction_exact": prediction_exact,
                  "wall_seconds": perf_counter()-tick, "checkpoint_selection": "final1000_only"}
        require(sampler.window_occurrences == 62992 and sampler.transition_occurrences == 314960,
                "Final exposure budget differs")
        save(folder / "summary.json", result)
        return result
    except Exception as error:
        torch.save({"weights": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "sampler": sampler.state_dict(), "rng": torch.get_rng_state()}, folder / "failed_prefix.pt")
        save(folder / "failure.json", {"arm": arm, "completed_updates": len(losses), "error": str(error),
             "traceback": traceback.format_exc(), "counters": counters})
        raise
    finally:
        np.savez_compressed(folder / "training_trace.npz",
            window_positions=np.concatenate(positions_all) if positions_all else np.empty(0, dtype=np.int64),
            batch_offsets=np.asarray(offsets, dtype=np.int64), batch_data_hashes=np.asarray(hashes, dtype="U64"),
            losses=np.asarray(losses, dtype=np.float64), update_wall_time_s=np.asarray(times, dtype=np.float64))


def run(args, out, counters):
    began = perf_counter(); physics = forbid_physics()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    registration, design = check_registration(args.evaluation_registration, args.evaluation_registration_sha256, args.run_id)
    require(shutil.disk_usage(ROOT).free >= 2*GIB + 256*MIB, "Need full output reserve before training")
    audit = read_json(ROOT / "reports/p4_data_main_independent_review.json")
    require(audit["status"] == "PASS" and audit["registration_sha256"] == digest(MAIN / "registration.json")
            and audit["summary_sha256"] == digest(MAIN / "summary.json"), "Independent collection audit does not bind current data")
    common_manifest = read_json(COMMON / "manifest.json")
    historical_manifest = read_json(HISTORICAL_RUN / "manifest.json")
    old_inventory = read_json(HISTORICAL_RUN / "artifact_hashes.json")["sha256"]
    historical_bound = {}
    for path in [HISTORICAL_RUN / "manifest.json", *[p for p in HISTORICAL.rglob("*") if p.is_file()]]:
        rel = path.relative_to(HISTORICAL_RUN).as_posix()
        require(digest(path) == old_inventory[rel], "Historical reference changed: " + rel)
        historical_bound[path.relative_to(ROOT).as_posix()] = digest(path)
    settings = validate_settings(tomllib.loads((ROOT / "configs/p3_training.v1.toml").read_text()))
    normalizer_meta = read_json(LEARNING / "normalizer/normalizer.json")["normalizer"]
    prepared, normalizer = load_learning_data(LEARNING, normalizer_meta["schema"],
        normalizer_meta["dataset_identity"], design["model"]["normalizer_hash"])
    config = prepared.metadata["contract"]["resolved_config"]
    require(config["training"]["gradient_clip_norm"] == 1.
            and config["training"]["learning_rate"] == .001
            and config["training"]["weight_decay"] == 0, "Frozen optimizer settings differ")
    require(normalizer.metadata["stats_hash"] == common_manifest["stats_hash"]
            and digest(LEARNING / "manifest.json") == common_manifest["provenance"]["learning_manifest_sha256"],
            "Original learning provenance changed")
    rows, states, actions, mask = train_sequences(prepared)
    with np.load(HISTORICAL / "training_trace.npz", allow_pickle=False) as archive:
        historical_trace = {k: archive[k].copy() for k in archive.files}
    datasets, slots, bindings = materialize_arms(MAIN, rows, states, actions, mask,
        historical_trace["window_positions"], historical_trace["batch_offsets"])
    require(bindings["arms"]["R"]["data_hash"] == historical_manifest["train_sequence_hash"],
            "R entire training tensor differs before updates")
    save(out / "dataset_bindings.json", bindings)
    arrays = {"old_window_rows": rows, "replacement_slots": slots}
    for arm, tensors in datasets.items():
        arrays.update({arm + "_" + name: value.numpy() for name, value in zip(("states", "actions", "mask"), tensors)})
    np.savez_compressed(out / "prepared_training.npz", **arrays)
    sources = source_hashes()
    entry = next(e for e in historical_manifest["budget"]["common_checkpoints"] if e["seed"] == 11)
    require(entry["path"] == design["model"]["initial_checkpoint"] and entry["checkpoint_hash"] == design["model"]["initial_checkpoint_hash"], "Common origin differs")
    manifest = {"schema": "p4-data-intervention-training-v1", "run_id": args.run_id,
        "registered_before_training": datetime.now(timezone.utc).isoformat(),
        "evaluation_registration": str(args.evaluation_registration.relative_to(ROOT)),
        "evaluation_registration_sha256": args.evaluation_registration_sha256,
        "physical_config": config, "normalizer_hash": normalizer.metadata["normalizer_hash"],
        "stats_hash": normalizer.metadata["stats_hash"], "initial_checkpoint": entry,
        "design": design, "training_provenance": bindings,
        "dataset_bindings_sha256": digest(out / "dataset_bindings.json"),
        "prepared_training_sha256": digest(out / "prepared_training.npz"),
        "source_sha256": sources, "historical_reference_sha256": historical_bound,
        "python": platform.python_version(), "torch": str(torch.__version__), "numpy": str(np.__version__),
        "torch_threads": 1, "deterministic_algorithms": True,
        "formal_update_budget": 5000, "disposable_update_cap": 160,
        "planned_disposable_updates": 159, "run_storage_limit_bytes": 256*MIB,
        "free_reserve_bytes": 2*GIB, "final_test_used": False,
        "new_closed_loop_execution": False, "P5_01": "HOLD"}
    save(out / "manifest.json", manifest); manifest_sha = digest(out / "manifest.json")
    probe = train_only_probe(datasets, slots, prepared, historical_trace, entry, normalizer, config,
                            common_manifest, settings, counters)
    save(out / "train_only_probe.json", probe)
    results = []
    for arm in ARMS:
        result = train_arm(arm, datasets[arm], entry, normalizer, config, common_manifest, settings,
            historical_manifest, historical_trace, manifest_sha, bindings["arms"][arm]["data_hash"], out, counters, prepared)
        results.append(result)
        print(json.dumps({"progress": "arm_complete", "arm": arm, "status": result["status"]}), flush=True)
    check_registration(args.evaluation_registration, args.evaluation_registration_sha256, args.run_id)
    require(all(digest(ROOT / rel) == sha for rel, sha in sources.items()), "Training source changed during execution")
    require(all(digest(ROOT / rel) == sha for rel, sha in historical_bound.items()), "Historical reference changed during training")
    require(all(digest(MAIN / rel) == sha for rel, sha in bindings["raw_core_artifact_sha256"].items()),
            "Bound train core artifacts changed during training")
    require(digest(out / "manifest.json") == manifest_sha
            and digest(out / "dataset_bindings.json") == manifest["dataset_bindings_sha256"]
            and digest(out / "prepared_training.npz") == manifest["prepared_training_sha256"], "Registered training inputs changed")
    require(counters["formal_updates"] == counters["formal_update_attempts"] == 5000
            and counters["disposable_updates"] == counters["disposable_update_attempts"] == 159,
            "Run update accounting differs")
    store = storage_guard(out)
    result = {"schema": "p4-data-intervention-training-result-v1", "status": "PASS", "arms": results,
              "counters": counters, "physics": physics, "manifest_sha256": manifest_sha,
              "wall_seconds": perf_counter()-began, "resources": store,
              "peak_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
              "prediction_comparison": "NOT_RUN", "final_test_used": False, "P5_01": "HOLD",
              "task_success": "NOT_EVALUATED", "personal_understanding": "NOT_ASSESSED"}
    save(out / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evaluation-registration", type=Path, required=True)
    parser.add_argument("--evaluation-registration-sha256", required=True)
    args = parser.parse_args(); args.evaluation_registration = args.evaluation_registration.resolve()
    require(re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id), "Unsafe run ID")
    out = ROOT / "runs" / args.run_id; out.mkdir(parents=True, exist_ok=False)
    counters = {"formal_updates": 0, "formal_update_attempts": 0,
                "disposable_updates": 0, "disposable_update_attempts": 0}
    try:
        result = run(args, out, counters)
        print(json.dumps({"run": str(out), "status": result["status"], "counters": counters}), flush=True)
        return 0
    except Exception as error:
        save(out / "failure.json", {"status": "FAIL", "error": str(error), "traceback": traceback.format_exc(),
                                   "counters": counters, "automatic_retry": False})
        print(json.dumps({"run": str(out), "status": "FAIL", "error": str(error), "counters": counters}), flush=True)
        return 1
    finally:
        save(out / "training_artifact_hashes.json", {"sha256": {p.relative_to(out).as_posix(): digest(p)
            for p in sorted(out.rglob("*")) if p.is_file() and p.name != "training_artifact_hashes.json"}})


if __name__ == "__main__": raise SystemExit(main())
