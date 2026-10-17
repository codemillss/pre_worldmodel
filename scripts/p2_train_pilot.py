"""P2-04 actual one-step learning, held-out free prediction and resume proof."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
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
import torch

from push_align.config import ROOT
from push_align.data import _read_json, digest
from push_align.checkpoint import load_checkpoint, make_optimizer, save_checkpoint
from push_align.dynamics import DeltaMLP, model_spec, predict_delta
from push_align.learning_data import load_learning_data
from push_align.prediction import evaluate_predictions, evaluate_state_hold
from push_align.training import one_step_loss, select_tiny_subset, training_update
from p2_episode_contract_check import source_bundle

DEFAULT_LEARNING = ROOT/"runs/p2-03-20261007-01/learning_data"
EXPECTED_NORMALIZER_HASH = "7df6a0ec11299bccc7925ae7c3c5c76e3435fb863b83520292637e154d8c09b0"


def save(path, value):
    # Validate the complete strict JSON before opening an exclusive output.
    # A serialization error must not leave a partial apparent PASS summary.
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+"\n"
    with Path(path).open("x", encoding="utf-8") as file:
        file.write(encoded)


def parameter_vector(model):
    return torch.cat([p.detach().reshape(-1).cpu() for p in model.parameters()]).clone()


def infer_normalized(model, normalizer, features):
    model.eval()
    with torch.no_grad():
        return model(normalizer.normalize_features(features).to(torch.float32)).clone()


def full_loss(model, normalizer, features, targets):
    model.eval()
    with torch.no_grad():
        return float(one_step_loss(model, normalizer, features, targets))


def save_evaluation(folder, evaluation):
    folder.mkdir(parents=True, exist_ok=False)
    save(folder/"summary.json", evaluation.summary)
    with (folder/"traces.npz").open("xb") as file:
        np.savez_compressed(file, **evaluation.traces)


def tensor_tree_equal(a, b):
    if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict): return isinstance(b, dict) and a.keys() == b.keys() and all(tensor_tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return type(a) is type(b) and len(a) == len(b) and all(tensor_tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def verify_prior_run(folder):
    hashes = _read_json(folder/"artifact_hashes.json")["sha256"]
    if not all(digest(folder/path) == sha for path, sha in hashes.items()):
        raise ValueError("Prior P2-03 artifact changed")
    return len(hashes)


def run(args, out):
    started = perf_counter(); settings = tomllib.loads(args.training_config.read_text())
    if settings["seed"] != 11 or settings["shuffle_seed"] != 1011 or settings["tiny_updates"] != 500 or settings["main_epochs"] != 30 or settings["batch_size"] != 256 or settings["drop_last"] is not False or settings["evaluation_horizons"] != [1, 5, 10] or settings["device"] != "cpu" or settings["torch_threads"] != 1 or settings["deterministic_algorithms"] is not True:
        raise ValueError("Runner requires the predeclared P2 one-seed pilot budget")
    # No paid backend, GPU or change to the existing project environment.
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    learning = args.learning_data.resolve()
    verified_prior = verify_prior_run(learning.parent) if learning == DEFAULT_LEARNING.resolve() else None
    normalizer_manifest = _read_json(learning/"normalizer/normalizer.json")["normalizer"]
    if learning == DEFAULT_LEARNING.resolve() and normalizer_manifest["normalizer_hash"] != EXPECTED_NORMALIZER_HASH:
        raise ValueError("Adopted P2-03 normalization changed")
    prepared, normalizer = load_learning_data(learning, normalizer_manifest["schema"], normalizer_manifest["dataset_identity"], normalizer_manifest["normalizer_hash"])
    config = prepared.metadata["contract"]["resolved_config"]
    if config["training"]["pilot_max_epochs"] != settings["main_epochs"] or config["training"]["batch_size"] != settings["batch_size"] or config["training"]["seed_candidates"][0] != settings["seed"]:
        raise ValueError("Separate runner budget disagrees with registered pilot configuration")
    provenance = {"learning_manifest_sha256": digest(learning/"manifest.json"), "dataset_identity": normalizer.metadata["dataset_identity"],
                  "validation_split": "validation", "task_scope": "P2-04-pilot"}
    a = prepared.arrays; tiny_rows = select_tiny_subset(prepared)
    train_rows = np.flatnonzero(a["split_code"] == 0)
    features, targets = torch.from_numpy(a["features"].copy()), torch.from_numpy(a["deltas"].copy())
    profile = tomllib.loads((ROOT/"configs/acceptance.v1.toml").read_text()); numeric = profile["numeric"]; lc = profile["learning_checks"]
    if len(tiny_rows) != lc["tiny_subset_transition_count"] or settings["tiny_updates"] != lc["tiny_subset_max_updates"]:
        raise ValueError("Tiny probe differs from unchanged acceptance profile")
    sources = source_bundle(out)
    save(out/"manifest.json", {"task_id": "P2-04", "created_at": datetime.now().astimezone().isoformat(), "argv": sys.argv,
                              "executable": sys.executable, "python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__),
                              "settings": settings, "training_config_sha256": digest(args.training_config), "config": config,
                              "model_spec": model_spec(config), "provenance": provenance, "normalizer_hash": normalizer.metadata["normalizer_hash"],
                              "source_hashes": sources, "tiny_rows": tiny_rows.tolist(), "tiny_original_identities": [{"episode_index": int(a["episode_index"][i]), "transition_index": int(a["transition_index"][i]), "regime_code": int(a["regime_code"][i])} for i in tiny_rows],
                              "train_rows": train_rows.tolist(), "acceptance_profile_sha256": digest(ROOT/"configs/acceptance.v1.toml"),
                              "numeric_tolerances": numeric, "learning_checks": lc, "MPC": "NOT_RUN", "alignment_outcome": "NOT_EVALUATED"})
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT, text=True, capture_output=True)
    (out/"tests.txt").write_text(tests.stdout+tests.stderr)
    count = re.search(r"Ran (\d+) tests", tests.stdout+tests.stderr)
    if tests.returncode: raise ValueError("Regression tests failed; official training not started")
    print(json.dumps({"progress": "regression_complete", "tests": int(count.group(1))}), flush=True)
    seed, clip = settings["seed"], config["training"]["gradient_clip_norm"]
    torch.manual_seed(seed); tiny = DeltaMLP(config); tiny_optimizer = make_optimizer(tiny, config)
    tiny_start = parameter_vector(tiny); initial_hash = hashlib.sha256(tiny_start.numpy().tobytes()).hexdigest()
    tx, ty = features[tiny_rows], targets[tiny_rows]
    before_loss = full_loss(tiny, normalizer, tx, ty); before_prediction = infer_normalized(tiny, normalizer, tx)
    tiny_history = []; tick = perf_counter()
    for update in range(settings["tiny_updates"]):
        row = training_update(tiny, tiny_optimizer, normalizer, tx, ty, clip)
        tiny_history.append({"update": update+1, **row})
    tiny_s = perf_counter()-tick
    after_loss = full_loss(tiny, normalizer, tx, ty); after_prediction = infer_normalized(tiny, normalizer, tx)
    tiny_parameter_change = float((parameter_vector(tiny)-tiny_start).abs().max())
    relative_decrease = (before_loss-after_loss)/before_loss if before_loss > 0 else None
    tiny_checks = {"initial_loss_not_numeric_floor": before_loss > numeric["float32_atol"],
                   "loss_finite": bool(np.isfinite([before_loss, after_loss]).all()), "gradient_finite_and_nonzero": all(np.isfinite(r["gradient_norm_before_clip"]) and r["gradient_norm_before_clip"] > 0 for r in tiny_history),
                   "parameter_changed": tiny_parameter_change > 0,
                   "prediction_change_above_tolerance": not bool(torch.allclose(before_prediction, after_prediction, atol=numeric["float32_atol"], rtol=numeric["float32_rtol"])),
                   "loss_decrease_at_least_one_percent": relative_decrease is not None and relative_decrease >= lc["tiny_subset_loss_relative_decrease_min"],
                   "loss_change_above_tolerance": not bool(np.isclose(before_loss, after_loss, atol=numeric["float32_atol"], rtol=numeric["float32_rtol"]))}
    save(out/"tiny_learning.json", {"checks": tiny_checks, "initial_loss": before_loss, "final_loss": after_loss, "relative_decrease": relative_decrease,
                                    "parameter_max_abs_change": tiny_parameter_change, "normalized_prediction_max_abs_change": float((after_prediction-before_prediction).abs().max()),
                                    "rows": tiny_rows.tolist(), "updates": settings["tiny_updates"], "wall_time_s": tiny_s, "history": tiny_history,
                                    "scope": "deliberately high-motion train-contact subset optimization probe; not held-out performance"})
    np.savez_compressed(out/"tiny_predictions.npz", before=before_prediction.numpy(), after=after_prediction.numpy(), physical_targets=ty.numpy(), rows=tiny_rows)
    if not all(tiny_checks.values()): raise ValueError("Tiny-learning connection checks failed; retain evidence and diagnose before main learning")
    print(json.dumps({"progress": "tiny_learning_complete", "initial_loss": before_loss, "final_loss": after_loss}), flush=True)
    # Tiny-fit weights never enter the main pilot or the validation selection.
    torch.manual_seed(seed); model = DeltaMLP(config); optimizer = make_optimizer(model, config)
    main_start = parameter_vector(model)
    fresh_matches = torch.equal(main_start, tiny_start)
    shuffle = torch.Generator(device="cpu").manual_seed(settings["shuffle_seed"])
    initial_evaluation = evaluate_predictions(model, normalizer, prepared)
    hold = evaluate_state_hold(prepared)
    save_evaluation(out/"evaluations/initial_untrained", initial_evaluation); save_evaluation(out/"evaluations/state_hold", hold)
    initial_train_loss = full_loss(model, normalizer, features[train_rows], targets[train_rows])
    before_main_prediction = infer_normalized(model, normalizer, features[train_rows])
    best_score, best_path, best_epoch, history, updates = None, None, None, [], 0
    train_s, validation_s = 0., 0.
    for epoch in range(1, settings["main_epochs"]+1):
        tick = perf_counter(); order = train_rows[torch.randperm(len(train_rows), generator=shuffle).numpy()]
        batches = []
        for start in range(0, len(order), settings["batch_size"]):
            rows = order[start:start+settings["batch_size"]]
            batches.append(training_update(model, optimizer, normalizer, features[rows], targets[rows], clip)); updates += 1
        elapsed = perf_counter()-tick; train_s += elapsed
        post_loss = full_loss(model, normalizer, features[train_rows], targets[train_rows])
        tick = perf_counter(); validation = evaluate_predictions(model, normalizer, prepared, horizons=(5,)); val_elapsed = perf_counter()-tick; validation_s += val_elapsed
        score = validation.summary["selection_score"]
        selected = score is not None and (best_score is None or score < best_score)
        checkpoint_path = out/f"checkpoints/epoch-{epoch:03d}"
        if selected or epoch == settings["main_epochs"]:
            save_checkpoint(checkpoint_path, model, optimizer, normalizer, config, provenance, epoch=epoch, updates=updates, seed=seed, shuffle_generator=shuffle, selection_score=score)
        if selected: best_score, best_path, best_epoch = score, checkpoint_path, epoch
        row = {"epoch": epoch, "updates": updates, "weighted_preupdate_batch_loss": sum(b["loss_before_update"]*b["batch_size"] for b in batches)/len(train_rows),
               "post_epoch_train_loss": post_loss, "batch_sizes": [b["batch_size"] for b in batches], "batch_diagnostics": batches,
               "validation_score": score, "selection_status": validation.summary["selection_status"], "selected_new_best": selected,
               "validation_regimes": validation.summary["horizons"]["5"]["regimes"], "train_wall_time_s": elapsed, "validation_wall_time_s": val_elapsed,
               "shuffle_original_rows": order.tolist()}
        history.append(row)
        with (out/"history.jsonl").open("a", encoding="utf-8") as file: file.write(json.dumps(row, allow_nan=False)+"\n")
        if epoch % 5 == 0: print(json.dumps({"progress": "main_learning", "epoch": epoch, "updates": updates, "train_loss": post_loss, "validation_score": score}), flush=True)
    final_path = out/f"checkpoints/epoch-{settings['main_epochs']:03d}"
    final = load_checkpoint(final_path, normalizer, config, provenance, resume=True)
    chosen = load_checkpoint(best_path if best_path is not None else final_path, normalizer, config, provenance)
    selected_evaluation = evaluate_predictions(chosen.model, normalizer, prepared)
    save_evaluation(out/"evaluations/selected_validation", selected_evaluation)
    last_evaluation = evaluate_predictions(final.model, normalizer, prepared)
    save_evaluation(out/"evaluations/final_validation", last_evaluation)
    after_main_prediction = infer_normalized(model, normalizer, features[train_rows])
    final_train_loss = full_loss(model, normalizer, features[train_rows], targets[train_rows])
    prediction_reload = torch.allclose(after_main_prediction, infer_normalized(final.model, normalizer, features[train_rows]), atol=numeric["float32_atol"], rtol=numeric["float32_rtol"])
    with torch.no_grad():
        physical_inputs = torch.from_numpy(a["current_state"][train_rows].copy())
        physical_actions = torch.from_numpy(a["applied_action"][train_rows].copy())
        physical_reload = torch.equal(predict_delta(model, normalizer, physical_inputs, physical_actions), predict_delta(final.model, normalizer, physical_inputs, physical_actions))
    main_parameter_change = float((parameter_vector(model)-main_start).abs().max())
    global_rng_matches = torch.equal(torch.get_rng_state(), final.torch_rng_state)
    # Two copies take one DIAGNOSTIC next update, outside the 60-update model
    # budget. Both checkpoint files and selection results remain untouched.
    order1 = train_rows[torch.randperm(len(train_rows), generator=shuffle).numpy()]
    order2 = train_rows[torch.randperm(len(train_rows), generator=final.shuffle_generator).numpy()]
    rows = order1[:settings["batch_size"]]
    training_update(model, optimizer, normalizer, features[rows], targets[rows], clip)
    training_update(final.model, final.optimizer, normalizer, features[order2[:len(rows)]], targets[order2[:len(rows)]], clip)
    resume_checks = {"saved_global_torch_rng_matches": global_rng_matches, "next_shuffle_identical": bool(np.array_equal(order1, order2)), "next_parameters_identical": torch.equal(parameter_vector(model), parameter_vector(final.model)),
                     "next_adam_state_identical": tensor_tree_equal(optimizer.state_dict(), final.optimizer.state_dict()), "diagnostic_updates_per_copy": 1,
                     "scope": "next update equivalence after saved epoch30; diagnostic copies only; selected/final checkpoint training budgets unchanged"}
    checks = {"regression_tests": tests.returncode == 0, "tiny_learning": all(tiny_checks.values()), "fresh_main_initialization": fresh_matches,
              "main_updates_exactly_60": updates == 60, "main_parameter_changed": main_parameter_change > 0,
              "main_prediction_changed": not bool(torch.allclose(before_main_prediction, after_main_prediction, atol=numeric["float32_atol"], rtol=numeric["float32_rtol"])),
              "trained_checkpoint_selected": best_path is not None, "selection_score_reproduced": best_score is not None and bool(np.isclose(best_score, selected_evaluation.summary["selection_score"], atol=numeric["float32_atol"], rtol=numeric["float32_rtol"])),
              "all_validation_horizons_recorded": set(selected_evaluation.summary["horizons"]) == {"1", "5", "10"},
              "stored_prediction_reload": bool(prediction_reload), "physical_prediction_reload_identical": physical_reload,
              "next_update_resume": all(resume_checks[k] for k in ("saved_global_torch_rng_matches", "next_shuffle_identical", "next_parameters_identical", "next_adam_state_identical")),
              "source_frozen": all(digest(ROOT/path) == sha for path, sha in sources.items()),
              "source_learning_data_unchanged": digest(learning/"manifest.json") == provenance["learning_manifest_sha256"]}
    if learning == DEFAULT_LEARNING.resolve(): checks["prior_artifacts_unchanged"] = verify_prior_run(learning.parent) == verified_prior
    selection = {"selected_epoch": best_epoch, "selected_updates": chosen.metadata["updates"], "selected_score": best_score,
                 "selected_checkpoint": str((best_path if best_path else final_path).relative_to(out)), "selected_checkpoint_hash": chosen.metadata["checkpoint_hash"],
                 "initial_untrained_score": initial_evaluation.summary["selection_score"], "final_epoch_score": last_evaluation.summary["selection_score"],
                 "state_hold_score": hold.summary["selection_score"], "initial_model_eligible_for_selection": False,
                 "scope": "trained epochs1..30 validation selection only; not P3 common checkpoint, final evaluation or G/R comparison"}
    save(out/"selection.json", selection); save(out/"resume_check.json", resume_checks)
    resources = {"device": "cpu", "torch_threads": 1, "tiny_training_wall_time_s": tiny_s, "main_updates_wall_time_s": train_s,
                 "epoch_validation_wall_time_s": validation_s, "total_before_plot_wall_time_s": perf_counter()-started,
                 "self_peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2, "scope": "main process only; regression/plot child memory excluded; no MPC inference measurement"}
    save(out/"training_outcome.json", {"initial_parameter_sha256": initial_hash, "initial_main_train_loss": initial_train_loss, "final_main_train_loss": final_train_loss,
                                       "main_parameter_max_abs_change": main_parameter_change, "main_normalized_prediction_max_abs_change": float((after_main_prediction-before_main_prediction).abs().max()),
                                       "main_seed_count": 1, "main_epochs": len(history), "main_updates": updates})
    env = os.environ.copy(); env["PYTHONPATH"] = str(args.plotting_packages); env["MPLCONFIGDIR"] = str(out/"plot_cache")
    plotting = subprocess.run([sys.executable, str(ROOT/"scripts/p2_plot_learning.py"), str(out)], text=True, capture_output=True, env=env)
    (out/"plotting.txt").write_text(plotting.stdout+plotting.stderr)
    checks["prediction_figures_created"] = plotting.returncode == 0
    checks["source_frozen"] &= all(digest(ROOT/path) == sha for path, sha in sources.items())
    resources["total_wall_time_s"] = perf_counter()-started
    save(out/"resources.json", resources)
    summary = {"task_id": "P2-04", "stage_verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
               "tests_run": int(count.group(1)), "tiny_initial_loss": before_loss, "tiny_final_loss": after_loss, "tiny_relative_decrease": relative_decrease,
               "main_training": {"seed": seed, "epochs": len(history), "updates": updates, "initial_train_loss": initial_train_loss, "final_train_loss": final_train_loss},
               "selection": selection, "validation": selected_evaluation.summary, "resources": resources, "verified_prior_artifacts": verified_prior,
               "alignment_outcome": "NOT_EVALUATED", "learner_understanding": "NOT_ASSESSED", "MPC": "NOT_RUN", "next_task": "P2-05 candidate/cost/model-rollout connection",
               "scope": "one-seed pilot dynamics training and recorded-action prediction; not main model, control improvement or alignment success"}
    save(out/"summary.json", summary)
    print(json.dumps({"verdict": summary["stage_verdict"], "tests": summary["tests_run"], "selected_epoch": best_epoch, "validation_K5_score": best_score}), flush=True)
    return all(checks.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--learning-data", type=Path, default=DEFAULT_LEARNING)
    parser.add_argument("--training-config", type=Path, default=ROOT/"configs/p2_training.v1.toml")
    parser.add_argument("--plotting-packages", type=Path, required=True, help="isolated plotting-only Python packages directory")
    args = parser.parse_args(); out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    try:
        passed = run(args, out)
    except Exception as error:
        save(out/"failure.json", {"exception": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        if not (out/"summary.json").exists(): save(out/"summary.json", {"task_id": "P2-04", "stage_verdict": "ERROR", "reason": str(error), "alignment_outcome": "NOT_EVALUATED", "MPC": "NOT_RUN"})
        print(json.dumps({"verdict": "ERROR", "reason": str(error)}), flush=True); passed = False
    save(out/"artifact_hashes.json", {"sha256": {str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()}})
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__": main()
