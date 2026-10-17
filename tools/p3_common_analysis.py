"""Independent P3-04 accounting, prediction diagnostics and readable figures.

Run only after the official summary and final artifact index have completed.
The audit performs no training update, normalization fit, simulator call or
original write. Saved live witnesses, checkpoint tensors, all 300 K5 traces
and original train-row permutations are checked directly. The runner's
disposable next-update experiment remains separately identified evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import numpy as np

import p3_ridge_analysis as r

ROOT = r.ROOT
REPORT = ROOT / "reports/p3_04_analysis.json"
FIGURES = ROOT / "reports/figures"
SEEDS = (11, 23, 47)
SHUFFLE_SEEDS = (1011, 1023, 1047)
HORIZONS = (1, 5, 10)
PARAMETER_SHAPES = {
    "network.0.weight": (128, 11), "network.0.bias": (128,),
    "network.2.weight": (128, 128), "network.2.bias": (128,),
    "network.4.weight": (10, 128), "network.4.bias": (10,),
}


def equal_tree(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.device == b.device and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def tensor_file(path, expected=None):
    import torch
    with r.record(path, expected).open("rb") as source:
        return torch.load(source, map_location="cpu", weights_only=True)


def checkpoint(folder, manifest, normalizer, seed, epoch, generator):
    import torch
    metadata = r.read(folder / "checkpoint.json")
    bound_hash = hashlib.sha256(r.canonical(manifest["settings"])).hexdigest()
    outer = hashlib.sha256(r.canonical({k: v for k, v in metadata.items() if k != "checkpoint_hash"})).hexdigest()
    if (metadata["checkpoint_hash"] != outer or metadata["seed"] != seed
            or metadata["epoch"] != epoch or metadata["updates"] != epoch * 13
            or metadata["training_settings"] != manifest["settings"]
            or metadata["training_settings_sha256"] != bound_hash
            or metadata["normalizer_hash"] != normalizer["normalizer_hash"]
            or metadata["learning_schema"] != normalizer["schema"]
            or metadata["training_provenance"] != manifest["provenance"]
            or metadata["config"] != manifest["physical_config"]):
        raise ValueError("Checkpoint counters/settings/data/physical contract differ")
    weights = tensor_file(folder / "weights.pt", metadata["files"]["weights.pt"])
    state = tensor_file(folder / "training_state.pt", metadata["files"]["training_state.pt"])
    if set(weights) != set(PARAMETER_SHAPES) or set(state) != {"optimizer", "torch_rng_state", "shuffle_rng_state"}:
        raise ValueError("Unexpected checkpoint tensor population")
    for name, shape in PARAMETER_SHAPES.items():
        value = weights[name]
        if value.shape != shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError("Checkpoint MLP shape/dtype/finite boundary differs")
    adam = state["optimizer"]
    if set(adam["state"]) != set(range(6)) or len(adam["param_groups"]) != 1:
        raise ValueError("Adam does not cover all six model tensors")
    for index, shape in enumerate(PARAMETER_SHAPES.values()):
        entry = adam["state"][index]
        if set(entry) != {"step", "exp_avg", "exp_avg_sq"} or float(entry["step"]) != epoch * 13:
            raise ValueError("Saved Adam step differs from official update boundary")
        for name in ("exp_avg", "exp_avg_sq"):
            value = entry[name]
            if value.shape != shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError("Invalid saved Adam moment")
        if (entry["exp_avg_sq"] < 0).any():
            raise ValueError("Negative saved Adam squared moment")
    for name in ("torch_rng_state", "shuffle_rng_state"):
        state_rng = state[name]
        if state_rng.dtype != torch.uint8 or state_rng.ndim != 1:
            raise ValueError("Invalid saved CPU RNG format")
        torch.Generator(device="cpu").set_state(state_rng)
    if not torch.equal(state["shuffle_rng_state"], generator.get_state()):
        raise ValueError("Checkpoint shuffle RNG does not follow the recorded original-row permutation")
    return metadata, weights, state


def epoch_k5(folder, original, window_hash):
    primary = r.read(folder / "summary.json")
    trace = r.arrays(folder / "traces.npz")
    rows = original["windows_5"]
    rows = rows[original["split_code"][rows[:, 0]] == 1]
    if (not np.array_equal(trace["h5_window_rows"], rows)
            or r.array_hash(rows) != window_hash
            or not np.all(original["split_code"][rows] == 1)):
        raise ValueError("Epoch validation K5 windows or original split/order differ")
    truth = np.concatenate((original["current_state"][rows[:, 0], None], original["next_state"][rows]), axis=1)
    if not np.array_equal(trace["h5_truth_states"], truth):
        raise ValueError("Epoch K5 truth differs from original prepared rows")
    first = rows[:, 0]
    for tag in ("family_id", "episode_id"):
        if not np.array_equal(trace["h5_" + tag], original["_" + tag][first]):
            raise ValueError("Epoch K5 family/episode lineage differs")
    regimes = original["regime_code"][first]
    if not np.array_equal(trace["h5_regime_code"], regimes):
        raise ValueError("Epoch regime differs from first original transition")
    valid = trace["h5_valid"]
    errors = trace["h5_predicted_states"][:, -1] - truth[:, -1]
    errors[:, 2] = r.wrapped(errors[:, 2])
    if not np.array_equal(errors[valid], trace["h5_endpoint_errors"][valid]):
        raise ValueError("Epoch endpoint errors differ from stored states")
    summary = primary["horizons"]["5"]
    if len(rows) != summary["attempted_windows"] or int((~valid).sum()) != summary["invalid_windows"]:
        raise ValueError("Epoch invalid/attempted-window accounting differs")
    group_scores = []
    for code, name in enumerate(r.REGIMES):
        mask = (regimes == code) & valid
        terms = (np.hypot(errors[mask, 0], errors[mask, 1]) / primary["position_tolerance_m"]
                 + np.abs(errors[mask, 2]) / primary["yaw_tolerance_rad"])
        measured = float(terms.mean()) if len(terms) else None
        saved = summary["regimes"][name]["selection_error_mean_valid_windows_only"]
        if measured is None or saved is None:
            if measured != saved:
                raise ValueError("Missing epoch regime differs")
        else:
            r.close(measured, saved, "independent epoch regime score")
        group_scores.append(measured)
    score = float(np.mean(group_scores)) if valid.all() and all(v is not None for v in group_scores) else None
    if score is None:
        if primary["selection_status"] == "eligible" or primary["selection_score"] is not None:
            raise ValueError("Invalid K5 epoch was allowed to select")
    else:
        if primary["selection_status"] != "eligible":
            raise ValueError("Complete finite K5 epoch incorrectly marked ineligible")
        r.close(score, primary["selection_score"], "independent equal-regime epoch score")
    return primary, score, int((~valid).sum())


def reference_prediction(manifest, original, normalizer):
    folder = ROOT / "runs/p3-03-20261009-01"
    ridge_manifest = r.read(folder / "manifest.json")
    choice = r.read(folder / "selection.json")
    if (ridge_manifest["normalizer_hash"] != manifest["normalizer_hash"]
            or ridge_manifest["provenance"]["dataset_identity"] != manifest["provenance"]["dataset_identity"]
            or ridge_manifest["validation_window_hashes"] != manifest["validation_window_hashes"]):
        raise ValueError("Ridge does not use the same main dataset, statistics and validation windows")
    model_path = folder / choice["selected"]["model_path"]
    metadata = r.read(model_path / "ridge.json")
    if metadata["ridge_hash"] != choice["selected"]["model_hash"]:
        raise ValueError("Selected Ridge model identity differs")
    r.record(model_path / "weights.npz", metadata["files"]["weights.npz"])
    _, _, measured, score = r.window_analysis(model_path.parent / "validation", original,
                                            manifest["validation_window_hashes"])
    return {"selected_alpha": choice["selected"]["alpha"], "K5_score": score,
            "ridge_hash": metadata["ridge_hash"], "horizons": measured,
            "selection_attempts": len(choice["candidates"]),
            "scope": "same main data/statistics/windows; selected on validation; 8 alpha candidates versus 100 epochs per each of 3 MLP seeds"}


def seed_analysis(folder, manifest, original, normalizer, seed, shuffle_seed):
    import torch
    primary = r.read(folder / "summary.json")
    selection = r.read(folder / "selection.json")
    proof = r.read(folder / "selected_resume_proof.json")
    history = [json.loads(line) for line in r.record(folder / "history.jsonl").read_text().splitlines() if line]
    if len(history) != 100 or [e["epoch"] for e in history] != list(range(1, 101)):
        raise ValueError("Every trained epoch must be retained in order")
    train = np.flatnonzero(original["split_code"] == 0)
    generator = torch.Generator(device="cpu").manual_seed(shuffle_seed)
    compact = []
    selected_tensors = None
    first_global_rng = None
    for epoch, entry in enumerate(history, 1):
        if entry["updates"] != epoch * 13:
            raise ValueError("Epoch update counter differs from 13-batch boundary")
        order = train[torch.randperm(len(train), generator=generator).numpy()]
        if entry["original_shuffle_rows"] != order.tolist() or sorted(order.tolist()) != train.tolist():
            raise ValueError("Epoch train permutation differs from declared shuffle RNG or original train rows")
        sizes = [b["batch_size"] for b in entry["batch_diagnostics"]]
        if sizes != [256] * 12 + [128]:
            raise ValueError("Dropped, repeated or differently sized train batch")
        if not all(np.isfinite([b["loss_before_update"], b["gradient_norm_before_clip"]]).all()
                   for b in entry["batch_diagnostics"]):
            raise ValueError("Nonfinite original batch diagnostic")
        weighted = sum(b["loss_before_update"] * b["batch_size"] for b in entry["batch_diagnostics"]) / 3200
        r.close(weighted, entry["weighted_batch_preupdate_loss"], "weighted batch denominator")
        checkpoint_folder = folder / entry["checkpoint"]
        metadata, weights, state = checkpoint(checkpoint_folder, manifest, normalizer, seed, epoch, generator)
        if metadata["checkpoint_hash"] != entry["checkpoint_hash"]:
            raise ValueError("History and actual checkpoint identity differ")
        if first_global_rng is None:
            first_global_rng = state["torch_rng_state"].clone()
        elif not torch.equal(first_global_rng, state["torch_rng_state"]):
            raise ValueError("Unexpected global RNG consumption in deterministic one-step epochs")
        summary, score, invalid = epoch_k5(folder / "evaluations" / f"epoch-{epoch:03d}",
                                           original, manifest["validation_window_hashes"]["5"])
        if summary != entry["validation_summary"] or invalid != entry["K5_invalid_windows"]:
            raise ValueError("History and actual saved K5 evaluation differ")
        if score is None or entry["selection_score"] is None:
            if score != entry["selection_score"]:
                raise ValueError("Epoch eligibility/score differs")
        else:
            r.close(score, entry["selection_score"], "history score")
        if metadata["selection_score"] != entry["selection_score"]:
            raise ValueError("Checkpoint score differs from original trained epoch")
        compact.append({"epoch": epoch, "updates": epoch * 13,
                        "post_epoch_train_loss": entry["post_epoch_train_loss"],
                        "weighted_batch_preupdate_loss": weighted, "selection_score": score,
                        "stored_selection_score": entry["selection_score"],
                        "selection_status": summary["selection_status"], "K5_invalid_windows": invalid,
                        "checkpoint_hash": metadata["checkpoint_hash"],
                        "original_train_shuffle_sha256": r.array_hash(order)})
        if epoch == selection["selected_epoch"]:
            selected_tensors = (metadata, weights, state)
    eligible = [e for e in compact if e["selection_status"] == "eligible" and e["selection_score"] is not None]
    selected = min(eligible, key=lambda e: (e["selection_score"], e["epoch"]))
    if (selected["epoch"] != selection["selected_epoch"] or selected["updates"] != selection["selected_updates"]
            or selected["checkpoint_hash"] != selection["selected_checkpoint_hash"]
            or primary["selected_epoch"] != selected["epoch"] or primary["seed"] != seed
            or primary["shuffle_seed"] != shuffle_seed):
        raise ValueError("Selected trained epoch differs from exact independent K5 argmin/earliest tie")
    r.close(selected["selection_score"], selection["selected_score"], "selected score")
    if primary["epochs"] != 100 or primary["updates"] != 1300:
        raise ValueError("Official per-seed training budget differs")
    witness = tensor_file(folder / "selected_live_witness.pt", primary["selected_live_witness_sha256"])
    metadata, weights, state = selected_tensors
    if (not equal_tree(witness["weights"], weights) or not equal_tree(witness["optimizer"], state["optimizer"])
            or not torch.equal(witness["torch_rng"], state["torch_rng_state"])
            or not torch.equal(witness["shuffle_rng"], state["shuffle_rng_state"])):
        raise ValueError("Persisted live selected witness differs from checkpoint weights/Adam/RNG")
    next_shuffle = torch.Generator(device="cpu").set_state(witness["shuffle_rng"])
    next_rows = train[torch.randperm(len(train), generator=next_shuffle).numpy()][:256].tolist()
    if (next_rows != proof["next_original_batch_rows"]
            or proof["selected_epoch"] != selected["epoch"] or proof["selected_updates"] != selected["updates"]
            or proof["diagnostic_updates_per_copy"] != 1 or proof["original_training_budget_changed"] is not False
            or proof["pass"] is not True or not all(proof["checks"].values())):
        raise ValueError("Recorded next-update experiment or independently recovered next batch differs")
    witness_rng = torch.Generator(device="cpu").set_state(witness["torch_rng"])
    checkpoint_rng = torch.Generator(device="cpu").set_state(state["torch_rng_state"])
    if not torch.equal(torch.rand(9, generator=witness_rng), torch.rand(9, generator=checkpoint_rng)):
        raise ValueError("Saved live and restored Torch RNG streams differ")
    _, selected_trace, selected_metrics, selected_score = r.window_analysis(
        folder / "selected_validation", original, manifest["validation_window_hashes"])
    r.close(selected_score, selected["selection_score"], "selected full evaluation score")
    final_summary, _, final_metrics, _ = r.window_analysis(
        folder / "final_validation", original, manifest["validation_window_hashes"])
    # Reload-vs-live values are an official experiment; artifacts and actual
    # original-row/RNG/counter/witness boundaries above are independently read.
    return {"seed": seed, "shuffle_seed": shuffle_seed, "epochs": 100, "main_updates": 1300,
            "selected_epoch": selected["epoch"], "selected_updates": selected["updates"],
            "selected_score": selected_score, "selected_checkpoint_hash": selected["checkpoint_hash"],
            "selected_live_witness_sha256": primary["selected_live_witness_sha256"],
            "all_100_checkpoint_tensor_counters_and_hashes_match": True,
            "all_original_train_permutations_and_saved_shuffle_rng_match": True,
            "global_rng_snapshots_constant_across_deterministic_epochs": True,
            "persisted_live_witness_matches_selected_weights_adam_and_both_rng": True,
            "independently_recovered_next_original_batch_matches": True,
            "saved_torch_rng_next_draws_match": True,
            "primary_live_restore_and_next_update_checks": primary["checks"],
            "primary_next_update_experiment": proof,
            "selected_horizons": selected_metrics, "final_horizons": final_metrics,
            "history": compact, "resources": primary["resources"],
            "initial_train_loss": primary["initial_train_loss"], "final_train_loss": primary["final_train_loss"],
            "MPC": primary["MPC"], "alignment_outcome": primary["alignment_outcome"]}


def analyze(run):
    r.USED.clear()
    folder = ROOT / "runs" / run
    if not (folder / "summary.json").exists() or not (folder / "artifact_hashes.json").exists():
        raise ValueError("Wait for the completed official summary and artifact index before analysis")
    primary = r.read(folder / "summary.json")
    manifest = r.read(folder / "manifest.json")
    if primary["stage_verdict"] != "PASS" or not all(primary["checks"].values()):
        raise ValueError("This completed-run audit requires the official PASS signal")
    settings = manifest["settings"]
    if (settings["seeds"] != list(SEEDS) or settings["shuffle_seeds"] != list(SHUFFLE_SEEDS)
            or settings["main_epochs"] != 100 or settings["updates_per_seed"] != 1300
            or settings["total_updates"] != 3900
            or hashlib.sha256(r.canonical(settings)).hexdigest() != manifest["training_settings_sha256"]):
        raise ValueError("Predeclared seed/settings/budget contract differs")
    source = r.source_audit(folder, manifest)
    r.record(ROOT / "tools/p3_ridge_analysis.py")
    original, learning_manifest, normalizer, _, _, _, main = r.load_main(manifest)
    probe = r.read(folder / "train_only_probe.json")
    if (probe["validation_rows_modified_in_memory"] != 1600
            or probe["diagnostic_updates_per_copy"] != 3 or not all(probe["checks"].values())):
        raise ValueError("Fixed-update train-only probe differs")
    seeds = [seed_analysis(folder / "seeds" / f"seed-{seed:02d}", manifest, original, normalizer,
                           seed, shuffle_seed) for seed, shuffle_seed in zip(SEEDS, SHUFFLE_SEEDS)]
    reference = reference_prediction(manifest, original, normalizer)
    _, _, held, _ = r.window_analysis(folder / "state_hold_validation", original,
                                     manifest["validation_window_hashes"], state_hold=True)
    budgets = {"official_main_updates_per_seed": 1300, "official_main_updates_total": sum(s["main_updates"] for s in seeds),
               "official_saved_trained_epoch_boundaries": len(seeds) * 100,
               "leakage_probe_updates": 2 * probe["diagnostic_updates_per_copy"],
               "selected_next_update_witness_and_two_copies": sum(3 * s["primary_next_update_experiment"]["diagnostic_updates_per_copy"] for s in seeds),
               "fresh_adam_counterfactual_updates": len(seeds),
               "diagnostic_updates_total": 2 * 3 + 3 * 3 + 3,
               "scope": "3900 original main updates; 18 disposable-copy diagnostics (6 leakage + 9 continuation + 3 fresh-Adam), not added to the saved common checkpoint training budget"}
    if budgets["official_main_updates_total"] != 3900:
        raise ValueError("Official update budget differs")
    index = r.read(folder / "artifact_hashes.json")
    for relative, actual in r.USED.items():
        path = ROOT / relative
        if path.is_relative_to(folder) and path.name != "artifact_hashes.json":
            inner = str(path.relative_to(folder))
            if inner not in index["sha256"] or index["sha256"][inner] != actual:
                raise ValueError("Read run artifact differs from completed immutable index")
    output = {
        "schema": "p3-common-derived-analysis-v1", "run_id": run,
        "created_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "stage": {"stage_verdict": primary["stage_verdict"], "primary_checks": primary["checks"],
                  "test_count": primary["test_count"], "resources": primary["resources"],
                  "all_used_run_inputs_match_final_artifact_index": True},
        "source_verification": source, "main_learning_context": main, "budgets": budgets,
        "train_only_fixed_update_probe": probe, "seeds": seeds,
        "reference_ridge": reference, "state_hold_horizons": held,
        "comparison_scope": {"same_main_dataset_statistics_windows_and_order": True,
                             "Ridge_selection_attempts": 8, "MLP_epoch_selections_per_seed": 100,
                             "MLP_predeclared_seeds": 3, "validation_used_for_model_selection": True,
                             "independent_final_test": False, "MPC": "NOT_RUN", "alignment_outcome": "NOT_EVALUATED"},
        "metric_labels_ko": {"object_position_error_mm": "예측 종료 시 물체 위치 오차(mm)",
                              "object_yaw_error_deg": "예측 종료 시 물체 각도 오차(도, 2π 차이)"},
        "limitations": [
            "Same validation windows permit descriptive prediction comparison; both model types were selected on that validation set.",
            "Selection opportunity differs: 8 Ridge alphas versus 100 trained epochs for each of 3 MLP seeds. This is not equal-search-budget superiority.",
            "Overlapping windows share rows and 30 validation families. Regime family counts overlap and cannot be summed into independent trials.",
            "Three seeds show these three runs, not a reliable confidence interval, population stability or unseen-condition generalization.",
            "Stationary K5 means all five original object-pose intervals satisfy the 1e-9 thresholds; pusher movement and state velocities are not called stationary.",
            "All stationary and moving windows and K10 directions remain. State hold can beat a learned model on stationary drift without solving control.",
            "The saved live witness is independently compared with checkpoint tensors/RNG; actual next-update equality is the runner's separately recorded disposable-copy experiment.",
            "The analysis performs no additional updates, normalization fit, simulator call or full raw-main validity audit.",
            "Common checkpoints are starting states for later S1/SK work. MPC, task success, final test and personal understanding are not assessed here.",
        ],
        "analysis_code_sha256": r.sha(__file__), "reused_prediction_helper_sha256": r.sha(ROOT / "tools/p3_ridge_analysis.py"),
        "used_input_sha256": dict(sorted(r.USED.items())),
    }
    output["all_used_inputs_unchanged_after_analysis"] = all(r.sha(ROOT / name) == value for name, value in r.USED.items())
    if not output["all_used_inputs_unchanged_after_analysis"]:
        raise ValueError("Original inputs changed during derived analysis")
    REPORT.write_text(json.dumps(output, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    print(json.dumps({"seeds": [{"seed": s["seed"], "selected_epoch": s["selected_epoch"], "score": s["selected_score"]} for s in seeds],
                      "official_updates": budgets["official_main_updates_total"], "diagnostic_updates": budgets["diagnostic_updates_total"],
                      "used_inputs": len(r.USED), "report_sha256": r.sha(REPORT)}))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    data = json.loads(REPORT.read_text())
    if data["analysis_code_sha256"] != r.sha(__file__):
        raise ValueError("Analyze the current code before plotting")
    plt.rcParams.update({"font.size": 10.5, "axes.labelsize": 11., "axes.titlesize": 12.,
                         "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
                         "legend.fontsize": 10.5, "svg.fonttype": "none"})
    colors = ("#287c6c", "#477397", "#ad8040")
    outputs, font_audit = {}, {}

    def save(fig, name):
        for suffix in ("png", "svg"):
            path = FIGURES / f"p3_common_{name}.{suffix}"
            fig.savefig(path, dpi=180)
            outputs[str(path.relative_to(ROOT))] = r.sha(path)
            if suffix == "svg":
                sizes = [float(v) for v in re.findall(r"font-size:\s*([0-9.]+)px", path.read_text())]
                if not sizes or min(sizes) < 10.5:
                    raise ValueError("Actual SVG font below 10.5px, including a MathText exponent")
                font_audit[str(path.relative_to(ROOT))] = {"actual_font_sizes_px": sorted(set(sizes)),
                                                          "actual_minimum_font_size_px": min(sizes)}
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9., 7.5))
    fig.subplots_adjust(left=.12, right=.97, top=.91, bottom=.17, hspace=.36)
    for s, color in zip(data["seeds"], colors):
        epochs = [e["epoch"] for e in s["history"]]
        axes[0].plot(epochs, [e["post_epoch_train_loss"] for e in s["history"]], color=color, label=f"Seed {s['seed']}")
        axes[1].plot(epochs, [e["selection_score"] for e in s["history"]], color=color, label=f"Seed {s['seed']}")
        axes[1].scatter([s["selected_epoch"]], [s["selected_score"]], color=color, marker="*", s=120, zorder=3)
    axes[0].set_yscale("log")
    def plain_scientific(value, _position):
        coefficient, exponent = f"{value:.0e}".split("e")
        return f"{coefficient}e{int(exponent)}"
    axes[0].yaxis.set_major_formatter(FuncFormatter(plain_scientific))
    axes[0].yaxis.set_major_locator(LogLocator(base=10., subs=(1., 2., 5.)))
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel("Normalized one-step train loss")
    axes[0].legend(loc="upper right", ncol=3)
    axes[1].set_ylabel("Equal-regime validation K5 score")
    axes[1].set_title("Stars: exact best trained epoch for each seed; earliest exact tie")
    for ax in axes:
        ax.set_xlabel("Trained epoch (1 .. 100)")
        ax.grid(alpha=.22)
        ax.set_xlim(1, 102)
    fig.suptitle("All three seeds retained: 100 epochs and 1300 main updates per seed", fontsize=13.)
    fig.text(.5, .08, "3900 official updates; 18 disposable-copy diagnostics are separate. Epoch 0 cannot select.", ha="center", fontsize=10.5)
    fig.text(.5, .04, "Selected validation is not independent final-test performance. MPC and S1/SK are not run.", ha="center", fontsize=10.5)
    save(fig, "learning")

    references = [(f"MLP seed {s['seed']}", s["selected_horizons"], color, "-")
                  for s, color in zip(data["seeds"], colors)]
    references += [("Ridge alpha 0", data["reference_ridge"]["horizons"], "#404040", "--"),
                   ("State hold", data["state_hold_horizons"], "#929292", ":")]
    fig, axes = plt.subplots(2, 1, figsize=(9., 7.5))
    fig.subplots_adjust(left=.12, right=.97, top=.91, bottom=.18, hspace=.36)
    for ax, metric, ylabel in ((axes[0], "object_position_error_mm", "Mean XY error (mm)"),
                               (axes[1], "object_yaw_error_deg", "Mean yaw error (deg, 2pi)")):
        maximum = 0.
        for name, results, color, style in references:
            values = [results[str(k)]["metrics_valid_windows_only"][metric]["mean"] for k in HORIZONS]
            maximum = max(maximum, max(values))
            ax.plot(HORIZONS, values, "o", linestyle=style, color=color, label=name)
        ax.set_ylim(0, maximum * 1.5)
        ax.set_ylabel(ylabel)
        ax.set_xticks(HORIZONS, [f"K{k}\nn={data['state_hold_horizons'][str(k)]['attempted_windows']}" for k in HORIZONS])
        ax.legend(loc="upper left", ncol=3)
        ax.grid(alpha=.22)
    fig.suptitle("Object endpoint prediction: common MLPs and Ridge on identical validation windows", fontsize=13.)
    fig.text(.5, .085, "Same data and normalization; 8 Ridge alpha trials vs 100 epoch selections per MLP seed.", ha="center", fontsize=10.5)
    fig.text(.5, .035, "Descriptive selected-validation comparison. Three seeds and overlapping windows do not prove superiority.", ha="center", fontsize=10.5)
    save(fig, "prediction")

    fig, axes = plt.subplots(2, 2, figsize=(9., 7.5))
    fig.subplots_adjust(left=.1, right=.97, top=.84, bottom=.17, hspace=.44, wspace=.28)
    motion_names = ("all_original_intervals_object_pose_stationary", "at_least_one_original_interval_object_pose_changes")
    for col, (metric, ylabel) in enumerate((("object_position_error_mm", "Mean K5 position error (mm)"),
                                           ("object_yaw_error_deg", "Mean K5 yaw error (deg, 2pi)"))):
        for index, (name, results, color, _) in enumerate(references):
            offset = (index - 2) * .15
            axes[0, col].bar(np.arange(3) + offset, [results["5"]["regimes"][group]["metrics_valid_windows_only"][metric]["mean"] for group in r.REGIMES],
                             width=.15, color=color, label=name)
            axes[1, col].bar(np.arange(2) + offset, [results["5"]["motion_groups"][group]["metrics_valid_windows_only"][metric]["mean"] for group in motion_names],
                             width=.15, color=color, label=name)
        axes[0, col].set_xticks(range(3), ["No contact\nn=889", "Sustained\nn=474", "Transition\nn=117"])
        axes[1, col].set_xticks(range(2), ["All 5 stationary\nn=702", "Any interval moves\nn=778"])
        axes[0, col].set_title("First measured interval contact regime")
        axes[1, col].set_title("All original intervals retained")
        for ax in axes[:, col]:
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=.22)
            ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), ncol=5)
    fig.suptitle("K5 gains and stationary drift remain separate diagnostics", fontsize=13.)
    fig.text(.5, .07, "Stationary means object pose within 1e-9 per interval; the pusher and velocities may change.", ha="center", fontsize=10.5)
    fig.text(.5, .03, "Windows share 30 validation families. Contact-regime family counts overlap; no independent-trial claim.", ha="center", fontsize=10.5)
    save(fig, "regimes")
    audit = {"analysis_report_sha256": r.sha(REPORT), "analysis_code_sha256": r.sha(__file__),
             "reused_prediction_helper_sha256": r.sha(ROOT / "tools/p3_ridge_analysis.py"),
             "figures": outputs, "actual_svg_font_audit": font_audit,
             "all_actual_svg_text_fonts_at_least_10_5px": True,
             "figure_size_inches": [9., 7.5], "aspect_ratio": 1.2,
             "minimum_label_font_pt": 10.5, "anticipated_pdf_width_pt": 480.,
             "minimum_label_pt_at_pdf_width": 10.5 * 480. / (9. * 72.),
             "scope": "new P3-04 derived figures only; exported PNG and actual PDF placement require visual QA"}
    (ROOT / "reports/p3_04_figures_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="p3-04-20261009-01")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run):
        raise ValueError("Unsafe run ID")
    if args.analyze:
        analyze(args.run)
    if args.plot:
        plot()
