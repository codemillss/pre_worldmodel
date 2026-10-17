"""Standalone Matplotlib figures from recorded learning/evaluation artifacts."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def main():
    run = Path(sys.argv[1]); figures = run/"figures"; figures.mkdir(exist_ok=False)
    tiny = read(run/"tiny_learning.json")
    history = [json.loads(line) for line in (run/"history.jsonl").read_text().splitlines()]
    selection = read(run/"selection.json")
    initial = read(run/"evaluations/initial_untrained/summary.json")
    learned = read(run/"evaluations/selected_validation/summary.json")
    hold = read(run/"evaluations/state_hold/summary.json")
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .2})
    colors = {"noncontact": "#58789a", "sustained_contact": "#2b9278", "transition": "#d58931"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    axes[0].plot([r["update"] for r in tiny["history"]], [r["loss_before_update"] for r in tiny["history"]], color="#2b9278")
    axes[0].set(xlabel="Tiny-subset update", ylabel="Scaled one-step loss", title="16 high-motion train rows; optimization probe", yscale="log")
    axes[1].plot([r["epoch"] for r in history], [r["post_epoch_train_loss"] for r in history], color="#2b9278")
    axes[1].set(xlabel="Epoch (2 updates / epoch)", ylabel="Full-train scaled one-step loss", title="Fresh model; 384 train rows", yscale="log")
    epochs = [0]+[r["epoch"] for r in history]
    axes[2].plot(epochs, [initial["selection_score"]]+[r["validation_score"] for r in history], color="#333333", linewidth=2, label="Equal-regime score")
    for regime, color in colors.items():
        values = [initial["horizons"]["5"]["regimes"][regime]["selection_error_mean_valid_windows_only"]]+[r["validation_regimes"][regime]["selection_error_mean_valid_windows_only"] for r in history]
        axes[2].plot(epochs, values, color=color, alpha=.8, label=regime.replace("_", " "))
    if selection["selected_epoch"] is not None:
        axes[2].axvline(selection["selected_epoch"], color="#b34153", linestyle="--", label=f"Selected epoch {selection['selected_epoch']}")
    axes[2].set(xlabel="Epoch (0: diagnostic only)", ylabel="K5 endpoint score (lower is better)", title="128 held-out transitions / 96 K5 windows")
    axes[2].legend(fontsize=7)
    fig.savefig(figures/"learning_curves.png", dpi=160); fig.savefig(figures/"learning_curves.svg"); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    for column, regime in enumerate(colors):
        for row, (metric, factor, label) in enumerate((("object_xy_distance_m", 1000, "Mean object XY error (mm)"),
                                                    ("object_yaw_absolute_rad", 180/np.pi, "Mean object yaw error (deg; 2pi)"))):
            x = np.arange(3); ax = axes[row, column]
            for shift, evaluation, color, method in ((-.18, learned, "#2b9278", "Selected MLP"), (.18, hold, "#a9b3c0", "State hold")):
                values = [evaluation["horizons"][str(k)]["regimes"][regime]["metrics_valid_windows_only"][metric]["mean"] for k in (1, 5, 10)]
                ax.bar(x+shift, [np.nan if v is None else v*factor for v in values], width=.34, color=color, label=method)
            ax.set(xticks=x, xticklabels=["K1 / 0.1s", "K5 / 0.5s", "K10 / 1.0s"], ylabel=label if column == 0 else "", title=regime.replace("_", " ") if row == 0 else "")
            if row == column == 0: ax.legend(fontsize=8)
    fig.suptitle("Validation endpoint errors by FIRST transition regime; state hold is a prediction diagnostic\nOverlapping windows are descriptive samples; this is not a control comparison", fontsize=11)
    fig.savefig(figures/"validation_endpoint_errors.png", dpi=160); fig.savefig(figures/"validation_endpoint_errors.svg"); plt.close(fig)
    with np.load(run/"evaluations/selected_validation/traces.npz", allow_pickle=False) as archive:
        traces = {k: archive[k].copy() for k in archive.files}
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), layout="constrained")
    examples = []
    for row, (regime, color) in enumerate(colors.items()):
        code = ("noncontact", "sustained_contact", "transition").index(regime)
        candidates = np.flatnonzero((traces["h10_regime_code"] == code) & traces["h10_valid"])
        if not len(candidates):
            for ax in axes[row]: ax.text(.5, .5, "No valid example", ha="center", transform=ax.transAxes)
            continue
        index = min(candidates, key=lambda i: (int(traces["h10_episode_index"][i]), int(traces["h10_start_transition_index"][i])))
        truth, prediction = traces["h10_truth_states"][index], traces["h10_predicted_states"][index]
        label = f"{regime.replace('_',' ')}\n{traces['h10_episode_id'][index]} / start {traces['h10_start_transition_index'][index]}"
        for column, pair in ((0, (0, 1)), (2, (6, 7))):
            ax = axes[row, column]
            ax.plot(truth[:, pair[0]]*1000, truth[:, pair[1]]*1000, "o-", markersize=3, color="#394556", label="Recorded truth")
            ax.plot(prediction[:, pair[0]]*1000, prediction[:, pair[1]]*1000, "x--", markersize=4, color=color, label="Free prediction")
            ax.scatter(truth[0, pair[0]]*1000, truth[0, pair[1]]*1000, color="#b34153", s=30, marker="s", label="Initial state")
            ax.set(xlabel="World X (mm)", ylabel="World Y (mm)", title=label if column == 0 else "Pusher XY")
        t = np.arange(11)*.1; ax = axes[row, 1]
        ax.plot(t, np.unwrap(truth[:, 2])*180/np.pi, "o-", markersize=3, color="#394556", label="Recorded truth")
        ax.plot(t, np.unwrap(prediction[:, 2])*180/np.pi, "x--", markersize=4, color=color, label="Free prediction")
        ax.set(xlabel="Time from window start (s)", ylabel="Object yaw (deg)", title="Object yaw")
        if row == 0: axes[row, 0].legend(fontsize=7)
        examples.append({"regime": regime, "episode_id": str(traces["h10_episode_id"][index]), "start_transition_index": int(traces["h10_start_transition_index"][index])})
    fig.suptitle("K10 examples: lowest episode/start identity per first-transition regime\nEvery future model input uses predicted object AND pusher states", fontsize=11)
    fig.savefig(figures/"validation_traces.png", dpi=160); fig.savefig(figures/"validation_traces.svg"); plt.close(fig)
    versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions(path=[os.environ["PYTHONPATH"]])}
    metadata = {"matplotlib": str(matplotlib.__version__), "numpy": str(np.__version__), "python": sys.version,
                "plotting_packages": os.environ["PYTHONPATH"], "plotting_package_versions": versions,
                "sample_selection": "lowest episode_index then original start_transition_index among valid K10 windows per first regime", "examples": examples,
                "plot_scope": "learning and recorded-action validation; state hold diagnostic; no alignment claim"}
    (run/"plot_runtime.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"figures": 3, "matplotlib": str(matplotlib.__version__), "examples": examples}))


if __name__ == "__main__": main()
