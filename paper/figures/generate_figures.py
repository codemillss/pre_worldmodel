#!/usr/bin/env python3
"""Render paper figures from preserved public artifacts; no simulation or training.

Run after the paper directory is created and the parent has recorded the PDF
operation marker. Example (from repository root):
PYTHONPATH=tmp/doc-deps MPLCONFIGDIR=tmp/paper-mpl \
  /path/to/bundled/python paper/figures/generate_figures.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Circle
import numpy as np


BLUE = "#174f84"
RED = "#b23a36"
INK = "#222222"
GRAY = "#666666"
LIGHT = "#f5f6f7"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 8.2,
        "axes.labelsize": 8.5, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def save(fig, out: Path, name: str) -> dict:
    paths = {}
    for extension in ("pdf", "png"):
        path = out / f"{name}.{extension}"
        if path.exists():
            raise FileExistsError(f"Refusing to replace existing figure: {path}")
        fig.savefig(path, format=extension, dpi=220,
                    metadata={"Creator": "Matplotlib; preserved experiment artifacts"})
        paths[extension] = {"file": path.name, "sha256": digest(path),
                            "bytes": path.stat().st_size}
    plt.close(fig)
    return paths


def arrow(ax, start, end, *, color=INK, dashed=False, mutation=9):
    patch = FancyArrowPatch(start, end, arrowstyle="-|>",
        mutation_scale=mutation, linewidth=0.9, color=color,
        linestyle="--" if dashed else "-", shrinkA=2, shrinkB=2)
    ax.add_patch(patch)


def box(ax, xy, width, height, title, detail, *, learned=False):
    x, y = xy
    ax.add_patch(FancyBboxPatch(xy, width, height,
        boxstyle="round,pad=0.035,rounding_size=0.09",
        facecolor="#edf4fb" if learned else LIGHT,
        edgecolor=BLUE if learned else GRAY, linewidth=0.85))
    ax.text(x + width / 2, y + height * .70, title, ha="center", va="center",
            fontsize=8.4, color=BLUE if learned else INK, fontweight="bold")
    ax.text(x + width / 2, y + height * .31, detail, ha="center", va="center",
            fontsize=7.8, linespacing=1.25, color=INK)


def method_system(out: Path) -> dict:
    fig = plt.figure(figsize=(7.16, 3.65))
    task = fig.add_axes([.012, .21, .272, .68])
    task.set_xlim(-.1, 2.2); task.set_ylim(-.2, 2.15)
    task.set_aspect("equal"); task.axis("off")
    fig.text(.012, .945, "(a) Planar alignment task", fontsize=9.2, fontweight="bold")
    # Geometry is schematic: no scene dimensions or measured trajectory implied.
    goal_center = np.array([1.35, 1.25]); object_center = np.array([.8, .70])
    local = np.array([[-.42, -.24], [.42, -.24], [.42, .24], [-.42, .24]])
    task.add_patch(Polygon(local + goal_center, closed=True, facecolor="#fff8d7",
                           edgecolor="#967400", linewidth=1.15, linestyle=(0, (4, 2))))
    angle = np.deg2rad(21)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    task.add_patch(Polygon(local @ rot.T + object_center, closed=True,
                           facecolor="#dbeaf8", edgecolor=BLUE, linewidth=1.1))
    task.plot(*object_center, marker="+", ms=5.5, color=BLUE)
    task.plot(*goal_center, marker="+", ms=5.5, color="#967400")
    task.add_patch(Circle((.23, .53), .11, facecolor="#d0d0d0", edgecolor=INK, lw=.9))
    arrow(task, (.10, .27), (.51, .42), color=RED, mutation=10)
    task.text(.00, .09, r"Velocity command $a_t$", fontsize=8)
    task.text(.25, 1.12, r"Part $(x,y,\theta)$", fontsize=8)
    task.text(.95, 1.66, r"Goal pose", fontsize=8)
    arrow(task, (1.79, -.09), (2.11, -.09), color=GRAY)
    arrow(task, (1.79, -.09), (1.79, .23), color=GRAY)
    task.text(2.12, -.10, "$x$", fontsize=8)
    task.text(1.76, .25, "$y$", fontsize=8)
    task.text(.02, -.20, "Schematic; not to scale", fontsize=7.5, color=GRAY)

    ax = fig.add_axes([.321, .21, .670, .68])
    ax.set_xlim(-.15, 10.05); ax.set_ylim(-.35, 5.10); ax.axis("off")
    fig.text(.321, .945, "(b) S1 / SK learned-dynamics planning loop", fontsize=9.2, fontweight="bold")
    ytop, ybot, h = 3.00, .12, 1.33
    box(ax, (0, ytop), 2.45, h, "Observe", "Current 10-D state\nfrom simulator")
    box(ax, (3.18, ytop), 2.45, h, "Propose", "128 sequences\n5 steps; 2-D actions")
    box(ax, (6.36, ytop), 3.35, h, "Learned rollout", "$11\\!\\to\\!128\\!\\to\\!128\\!\\to\\!10$ MLP\nPredict $\\Delta s$; recurse 5 steps", learned=True)
    box(ax, (6.36, ybot), 3.35, h, "Score and select", "Fixed 9-term cost + checks\nSelect first action $a^*_0$")
    box(ax, (3.18, ybot), 2.45, h, "Pose supervisor", "If pose in goal: zero\nOtherwise: $a^*_0$")
    box(ax, (0, ybot), 2.45, h, "Execute", "0.1 s interval\nHandoff check")
    arrow(ax, (2.45, 3.66), (3.18, 3.66))
    arrow(ax, (5.63, 3.66), (6.36, 3.66))
    arrow(ax, (8.04, ytop), (8.04, ybot+h))
    ax.text(8.22, 2.24, "Predictions", fontsize=7.6, va="center")
    arrow(ax, (6.36, .78), (5.63, .78))
    arrow(ax, (3.18, .78), (2.45, .78))
    arrow(ax, (1.22, ybot+h), (1.22, ytop))
    ax.text(.99, 2.22, "Reobserve", rotation=90, fontsize=7.5, ha="right", va="center")
    # Actual current state drives the fixed supervisor; this is not learned stopping.
    ax.plot([2.45, 4.4], [3.27, 2.24], color=GRAY, lw=.8, ls="--")
    arrow(ax, (4.4, 2.24), (4.4, ybot+h), color=GRAY, dashed=True)
    ax.text(4.55, 2.18, "Pose in goal:\nskip planning", fontsize=7.4,
            color=GRAY, linespacing=1.25, va="center")
    ax.text(4.36, 4.45, "Geometric proposal + seeded sampling", fontsize=6.85,
            ha="center", color=GRAY)
    fig.text(.015, .128,
        "MLP: multilayer perceptron. S1: one-step loss; SK: one-step + 0.1 × 5-step free-rollout loss.",
        fontsize=8.05, color=INK)
    fig.text(.015, .078,
        "R: feature-linear model (α = 0) replaces the MLP.  G: direct geometric feedback; no candidate search.",
        fontsize=8.05, color=INK)
    fig.text(.015, .030,
        "128 candidates; 5 × 0.1 s prediction steps; execute only the first command. Supervisor and contact physics are not learned.",
        fontsize=7.6, color=GRAY)
    result = save(fig, out, "method_system")
    result["caption"] = (
        "Task and S1/SK controller schematic (not to scale). S1 is a one-step-trained multilayer perceptron (MLP); "
        "SK has the same architecture and one-step plus 0.1 times five-step free-rollout training loss. "
        "R replaces the MLP with feature-linear dynamics (selected regularization alpha=0); G is direct geometric feedback without candidate search. "
        "A rectangular part is pushed toward a target pose. "
        "At each 0.1 s control interval, the current 10-D state and candidate commands form the 11-D model input; "
        "a learned state-delta model recursively evaluates 128 five-step action sequences. "
        "When the current pose is outside tolerance, a fixed cost and validity checks select the first command. "
        "When the actual position and orientation are within tolerance, planning is bypassed and a common, nonlearned supervisor "
        "applies zero command; low actual speed and "
        "continuous 0.3 s qualification are checked at physics substeps. Future simulator states are not queried by planning.")
    result["source_fields"] = [
        "src/push_align/representation.py:STATE_FIELDS,FEATURE_FIELDS,DELTA_FIELDS",
        "src/push_align/planning.py:rollout,score,plan",
        "src/push_align/final_execution.py:holding=pose_in_goal; zero override before env.step",
        "src/push_align/task.py:handoff_metrics,pose_in_goal",
    ]
    result["nature"] = "Explanatory schematic, not empirical scene image or measured trajectory"
    return result


def fixed_prediction(repo: Path, out: Path) -> dict:
    source = repo / "artifacts/results/fixed_prediction/summary.json"
    data = json.loads(source.read_text())
    assert data["family_count"] == 12
    assert data["additional_physics_intervals"] == 0
    assert all(value == 0 for value in data["future_engine_calls"].values())
    exact = {"source": str(source.relative_to(repo)), "source_sha256": digest(source),
             "metric": "object_xy_error_m", "unit_plotted": "mm",
             "aggregation": "Unweighted mean of 12 per-family mean endpoint errors",
             "family_denominator": 12, "seeds": {}, "horizons": [1, 5, 10],
             "limitations": data["limitations"],
             "selection": "Seeds 23 and 47 have all eligible windows valid for both S1 and SK at all plotted horizons; seed 11 is excluded from this common-valid illustrative plot because its validity masks differ.",
             "confidence_intervals": "Not computed for this descriptive figure; windows overlap and are not independent samples."}
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.48), sharey=True)
    for ax, seed, label in zip(axes, (23, 47), ("a", "b")):
        exact["seeds"][str(seed)] = {}
        for method, color, marker, linestyle in (("S1", BLUE, "o", "-"), ("SK", RED, "^", "--")):
            model = f"{method}-{seed}"
            values, rows = [], []
            for horizon in (1, 5, 10):
                hdata = data["horizons"][str(horizon)]
                record = hdata["models"][model]
                eligible = hdata["truth_eligible_windows"]
                assert record["truth_eligible_windows"] == eligible
                assert record["predicted_valid_windows"] == eligible
                assert record["predicted_invalid_windows"] == 0
                assert len(record["by_family"]) == 12
                means = []
                for family in data["family_ids"]:
                    fr = record["by_family"][family]
                    fm = fr["metrics_conditional_on_valid_prediction_and_truth"]["object_xy_error_m"]
                    assert fr["predicted_valid_windows"] == fr["truth_eligible_windows"]
                    assert fr["predicted_invalid_windows"] == 0
                    assert fm["denominator"] == fr["truth_eligible_windows"] > 0
                    means.append(fm["mean"])
                computed = float(np.mean(means))
                stored = record["family_first_means_conditional_on_available_families"]["object_xy_error_m"]
                assert stored["family_denominator"] == 12
                assert abs(computed - stored["mean"]) < 1e-18
                mm = computed * 1000
                values.append(mm)
                rows.append({"horizon": horizon, "seconds": horizon * .1,
                             "family_first_mean_m": computed, "family_first_mean_mm": mm,
                             "truth_eligible_windows": eligible,
                             "valid_prediction_windows": record["predicted_valid_windows"],
                             "invalid_prediction_windows": record["predicted_invalid_windows"],
                             "family_means_m": dict(zip(data["family_ids"], means)),
                             "source_field": f"horizons.{horizon}.models.{model}.family_first_means_conditional_on_available_families.object_xy_error_m"})
            exact["seeds"][str(seed)][method] = rows
            legend = ("S1: one-step multilayer perceptron (MLP)" if method == "S1" else
                      "SK: same MLP, +0.1 × 5-step rollout loss")
            ax.plot((1, 5, 10), values, color=color, marker=marker, linestyle=linestyle,
                    markersize=4.5, lw=1.45, label=legend, zorder=3)
        ax.set_title(f"({label}) Training seed {seed}", loc="left", fontweight="bold", pad=6)
        ax.set_xticks([1, 5, 10]); ax.set_xlim(.5, 10.5)
        ax.set_ylim(0, .185); ax.set_yticks([0, .04, .08, .12, .16])
        ax.set_xlabel("Prediction horizon $K$ (0.1 s per step)")
        ax.grid(axis="y", color="#d7d7d7", lw=.55, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Part-position endpoint error (mm)")
    fig.text(.10, .975, "Fixed-action prediction on geometric-feedback (G) trajectories",
             fontsize=9, fontweight="bold", va="top")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.535, .932),
               ncol=2, frameon=False, handlelength=2.4, columnspacing=1.5, fontsize=8)
    fig.subplots_adjust(left=.10, right=.985, top=.745, bottom=.335, wspace=.17)
    footnotes = [
        "Each marker: mean of 12 equally weighted initial-condition means. Lines guide the eye; no confidence intervals.",
        "K = future 0.1 s steps; all 2,656 / 2,608 / 2,548 eligible windows valid at K = 1 / 5 / 10 for both plotted seeds.",
        "Seed: training randomness setting. Seed 11 omitted: validity masks differ. Overlapping windows are not independent.",
    ]
    for y, note in zip((.158, .108, .058), footnotes):
        fig.text(.015, y, note, fontsize=7.75, color=GRAY)
    result = save(fig, out, "fixed_prediction")
    exact_path = out / "fixed_prediction.values.json"
    if exact_path.exists():
        raise FileExistsError(f"Refusing to replace values: {exact_path}")
    exact_path.write_text(json.dumps(exact, indent=2, ensure_ascii=False) + "\n")
    result["values"] = {"file": exact_path.name, "sha256": digest(exact_path),
                         "bytes": exact_path.stat().st_size}
    result["source"] = exact["source"]
    result["source_sha256"] = exact["source_sha256"]
    result["caption"] = (
        "Offline fixed-action position prediction on the 12 geometric-feedback (G) controller traces. "
        "S1 is the one-step-trained MLP; SK is the same MLP trained with an additional 0.1 times five-step free-rollout loss. "
        "Seed identifies the training random-number-generator setting, not a new initial-condition family. "
        "K counts future 0.1 s prediction steps; training uses five-step windows and online planning has a five-step horizon. "
        "Each point is the mean of 12 equally weighted family means, not the pooled-window mean. "
        "At K=1,5,10, respectively, all 2,656, 2,608, and 2,548 truth-eligible windows are valid "
        "for both methods in each plotted seed. Seed 11 is omitted because validity masks differ. "
        "Connecting lines are visual guides and confidence intervals are not shown. "
        "Overlapping windows are not independent; these descriptive prediction errors do not establish closed-loop handoff improvement.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if not args.out.is_dir():
        raise FileNotFoundError("Parent must create paper/figures before generation")
    style()
    report = {"schema": "paper-figures-v2", "presentation_revision": "2026-10-22-standalone-method-definitions",
              "method_system": method_system(args.out),
              "fixed_prediction": fixed_prediction(args.repo, args.out),
              "visual_review": "PENDING_DIRECT_VIEW_CURRENT_REVISION",
              "scientific_data_changed": False, "additional_physics_intervals": 0,
              "generator": {"file": "generate_figures.py", "sha256": digest(Path(__file__))}}
    target = args.out / "figures.manifest.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"manifest": str(target), "manifest_sha256": digest(target)}, indent=2))


if __name__ == "__main__":
    main()
