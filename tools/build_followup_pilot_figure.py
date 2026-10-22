#!/usr/bin/env python3
"""Plot selected D2 cases from saved derived arrays; no simulator/model calls.

The two cases serve different diagnostic questions, not a population estimate.
The prediction curve ends at its first invalid boundary; no suffix is filled.
Example: PYTHONPATH=tmp/doc-deps MPLCONFIGDIR=tmp/mplconfig python this_file.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, BLUE, RED, GREEN = "#172B3A", "#516671", "#174F84", "#B23A36", "#26786A"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def gap_proxy(states, config):
    # The same signed rectangle-distance expression as planning.contact_distance.
    relative = states[..., 6:8]-states[..., :2]
    yaw = states[..., 2]
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.stack((c*relative[..., 0]+s*relative[..., 1],
                      -s*relative[..., 0]+c*relative[..., 1]), axis=-1)
    half = np.asarray(config["object"]["size_full_xyz_m"][:2], np.float64)/2
    q = np.abs(local)-half
    return np.hypot(np.maximum(q[..., 0], 0), np.maximum(q[..., 1], 0)) \
        + np.minimum(np.max(q, axis=-1), 0)-config["pusher"]["radius_m"]


def position_mm(states, goal):
    return np.hypot(states[..., 0]-goal[0], states[..., 1]-goal[1])*1000


def actual_curve(data):
    mask = np.isfinite(data["boundary_times"]) & np.isfinite(data["states"]).all(axis=1)
    indices = np.flatnonzero(mask)
    if not np.array_equal(indices, np.arange(len(indices))):
        raise ValueError("Actual boundary observations must be an unfilled contiguous prefix")
    return data["boundary_times"][mask]-data["boundary_times"][0], data["states"][mask], mask


def format_ax(ax, ylabel, *, xmax):
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.set_xlabel("Time since restored state (s)", fontsize=10)
    ax.set_xlim(-.012, xmax+.012)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9.5)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        default=ROOT/"artifacts/followup/20261022-d1-d2")
    parser.add_argument("--output", type=Path, default=ROOT/"assets/figures/followup_pilot.png")
    args = parser.parse_args()
    base = args.source_root.resolve()
    paths = {
        "prediction": base/"states/F-O0/prediction-S1-23.npz",
        "F-A": base/"states/F-O0/actual/candidate-000-A.npz",
        "N-A": base/"states/N-T/actual/candidate-009-A.npz",
        "N-B": base/"states/N-T/actual/candidate-009-B.npz",
        "F-A-meta": base/"states/F-O0/actual/candidate-000-A.summary.json",
        "N-A-meta": base/"states/N-T/actual/candidate-009-A.summary.json",
        "N-B-meta": base/"states/N-T/actual/candidate-009-B.summary.json",
        "summary": base/"summary.json",
        "frozen": ROOT/"artifacts/provenance/frozen_manifest.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Wait for completed derived public data; missing: "+", ".join(missing))
    pred = read_npz(paths["prediction"])
    actual = {key: read_npz(paths[key]) for key in ("F-A", "N-A", "N-B")}
    meta = {key: json.loads(paths[key+"-meta"].read_text()) for key in actual}
    summary = json.loads(paths["summary"].read_text())
    frozen = json.loads(paths["frozen"].read_text())
    config = frozen["planner_config"]
    goal = np.asarray(config["task"]["goal_xy_m"], np.float64)
    dt = config["physics"]["action_dt_s"]
    hold_threshold = config["task"]["hold_time_s"]
    position_threshold = config["task"]["position_tolerance_m"]*1000
    if summary.get("status") != "PASS_PILOT_RECORDING_AND_REPRODUCTION_GATES":
        raise ValueError("Figure requires completed pilot recording/reproduction gates")
    if not np.array_equal(gap_proxy(pred["states"], config), pred["geometric_gap_proxy"], equal_nan=True):
        raise ValueError("Recomputed geometric proxy differs from preserved prediction array")
    if bool(pred["valid"][0]) or not meta["F-A"]["physical_valid"] or not meta["F-A"]["horizon_completed"]:
        raise ValueError("The prespecified rejection case does not satisfy its displayed claim")
    first_invalid = int(pred["failure_steps"][0])
    if str(pred["reasons"][0]) != "predicted_deep_penetration":
        raise ValueError("This case caption requires the recorded deep-penetration rejection reason")
    if not 0 <= first_invalid <= 5:
        raise ValueError("Prediction failure boundary is not in H5")
    pstate = pred["states"][0, :first_invalid+1]
    finite = np.isfinite(pstate).all(axis=1)
    if not finite.all():
        raise ValueError("Rejected predicted boundary is nonfinite; this figure needs a different visual treatment")
    ptime = np.arange(len(pstate), dtype=np.float64)*dt
    if (meta["N-B"]["first_ready_interval"] != 6 or meta["N-B"]["ready_prefix"]
            or not meta["N-B"]["ready_eligible"] or meta["N-A"]["ready_eligible"]):
        raise ValueError("The prespecified A/B case no longer matches its recorded explanation")
    # Verify the case statement for all 3 repeats, not just the plotted repeat0.
    for state_id, cid, panel in (("F-O0", 0, "A"), ("N-T", 9, "A"), ("N-T", 9, "B")):
        rows = [row for row in summary["rows"] if row["state"] == state_id
                and row["candidate_id"] == cid and row["panel"] == panel]
        if len(rows) != 3 or {row["repeat"] for row in rows} != {0, 1, 2}:
            raise ValueError("Selected case lacks all 3 preserved repeats")
        for row in rows:
            if row["physical_valid"] is not True:
                raise ValueError("Selected case has an invalid actual repeat")
            if row["ready_eligible"] != meta[("F-A" if state_id == "F-O0" else "N-"+panel)]["ready_eligible"]:
                raise ValueError("Selected case has unstable readiness")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelcolor": INK, "text.color": INK,
                         "axes.edgecolor": "#9FAEB6", "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(12, 8.4), facecolor="white")
    axes = [fig.add_axes(rect) for rect in (
        [.080, .575, .380, .220], [.580, .575, .380, .220],
        [.080, .170, .380, .220], [.580, .170, .380, .220])]
    fig.text(.080, .962, "A valid physical candidate can be rejected by the learned model",
             fontsize=17, fontweight="bold", va="top")
    fig.text(.080, .917, "Selected post-freeze pilot cases · first repeat plotted · 3 repeats agree · no training or frozen-result replacement",
             fontsize=10.8, color=MUTED, va="top")
    fig.text(.080, .858, "1   OFFSET CASE  |  offset-00 · source S1-23 · tick 254 · candidate 0",
             fontsize=12.2, fontweight="bold")
    fig.text(.080, .824, "Same nominal commands, same 0.5 s: prediction stops at its first rejected boundary; actual A remains physically valid.",
             fontsize=10.5, color=MUTED)
    tF, sF, _ = actual_curve(actual["F-A"])
    values = ((position_mm(sF, goal), position_mm(pstate, goal)),
              (gap_proxy(sF, config)*1000, pred["geometric_gap_proxy"][0, :first_invalid+1]*1000))
    for index, (ax, (physical, predicted)) in enumerate(zip(axes[:2], values)):
        ax.plot(tF, physical, "o-", color=BLUE, label="Actual A", markersize=4, linewidth=2)
        ax.plot(ptime, predicted, "--", color=RED, label="One-step MLP prediction", linewidth=2)
        ax.scatter(ptime[-1], predicted[-1], color=RED, marker="x", s=72, linewidth=2, zorder=4)
        ax.annotate("first rejected\nboundary", (ptime[-1], predicted[-1]),
                    xytext=(-10, 23 if index == 0 else 25), textcoords="offset points",
                    fontsize=9.1, color=RED, ha="right", va="bottom",
                    arrowprops={"arrowstyle": "-", "color": RED, "lw": .8})
        format_ax(ax, "Position error (mm)" if index == 0 else "Signed gap proxy (mm)", xmax=.5)
        if index == 1:
            ax.axhline(0, color="#7A8990", ls=":", lw=1)
            rejection_mm = -config["validity"]["predicted_penetration_tolerance_m"]*1000
            ax.axhline(rejection_mm, color="#8C6F24", ls=":", lw=1)
            ax.text(.98, .20, f"Model rejection below {rejection_mm:g} mm",
                    transform=ax.transAxes, ha="right", fontsize=9, color="#8C6F24")
    axes[0].legend(loc="best", frameon=False, fontsize=9.2)
    fig.text(.580, .512, "Negative gap = geometric overlap; not measured contact force.",
             fontsize=9.2, color=MUTED)
    fig.add_artist(plt.Line2D([.08, .96], [.486, .486], transform=fig.transFigure, color="#DAE3E8", lw=1))
    fig.text(.080, .451, "2   TRANSLATION CASE  |  translation-03 · source S1-11 · tick 78 · candidate 9",
             fontsize=12.2, fontweight="bold")
    fig.text(.080, .416, "B reaches ready at 0.6 s, after the 0.5 s prefix: this contrast combines a stop rule and additional zero-command time.",
             fontsize=10.5, color=MUTED)
    for key, color, marker, label in (("N-A", BLUE, "o", "A: fixed nominal 0.5 s"),
                                      ("N-B", GREEN, "s", "B: pose-zero rule + zero tail")):
        time, states, mask = actual_curve(actual[key])
        axes[2].plot(time, position_mm(states, goal), marker+"-", color=color,
                     markersize=4, label=label, linewidth=2)
        axes[3].plot(time, actual[key]["hold_s"][mask], marker+"-", color=color,
                     markersize=4, label=label, linewidth=2)
    axes[2].axhline(position_threshold, color="#8C6F24", lw=1.1, ls=":", label="5 mm pose threshold")
    axes[3].axhline(hold_threshold, color="#8C6F24", lw=1.1, ls=":")
    axes[3].text(.02, .92, "0.3 s continuous qualification required", transform=axes[3].transAxes,
                 fontsize=9, color="#8C6F24")
    for ax, ylabel in zip(axes[2:], ("Position error (mm)", "Continuous qualifying hold (s)")):
        format_ax(ax, ylabel, xmax=.8)
        ax.axvline(.5, color="#A6B2B9", ls="--", lw=1)
    axes[2].legend(loc="best", frameon=False, fontsize=8.9)
    tb, sb, mb = actual_curve(actual["N-B"])
    ready_time = dt*meta["N-B"]["first_ready_interval"]
    axes[3].annotate("first ready at 0.6 s", (ready_time, actual["N-B"]["hold_s"][mb][-1]),
                     xytext=(-110, -40), textcoords="offset points", fontsize=9.5, color=GREEN,
                     arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1})
    axes[3].set_ylim(-.013, max(.34, float(np.nanmax(actual["N-B"]["hold_s"]))+.025))
    fig.text(.080, .104, "S1 = one-step-trained multilayer perceptron (MLP); suffixes 11/23 = training seeds. Both sources are S1 failure paths.",
             fontsize=10, color=MUTED)
    fig.text(.080, .071, "A = fixed 5 commands, no future stop override. B = current-pose zero rule (not latched), then up to 0.3 s of zero commands.",
             fontsize=10, color=MUTED)
    fig.text(.080, .038, "Readiness also requires yaw and low linear/angular speed continuously. Local cases do not establish full-policy success or population effects.",
             fontsize=9.7, color=MUTED)
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError("Refusing to overwrite an existing diagnostic figure: "+str(out))
    fig.savefig(out, dpi=190, facecolor="white", metadata={"Description": "Saved post-freeze pilot case diagnostics; no new physical/model execution"})
    plt.close(fig)
    receipt = {
        "schema": "followup-pilot-figure-v1",
        "generator": str(Path(__file__).relative_to(ROOT)), "generator_sha256": sha(Path(__file__)),
        "sources": {str(path.relative_to(ROOT)): sha(path) for path in paths.values()},
        "figure": str(out.relative_to(ROOT)), "figure_sha256": sha(out),
        "selected_case_rule": "Prespecified F-O0 candidate0 S1-23 false rejection and N-T candidate9 A/B contrast; first repeat plotted; all3 repeats checked",
        "F_prediction_first_invalid_boundary": first_invalid,
        "F_actual_A_valid": meta["F-A"]["physical_valid"],
        "N_B_first_ready_relative_s": ready_time,
        "N_B_ready_prefix": meta["N-B"]["ready_prefix"],
        "prediction_suffix_fill": False, "additional_physics_or_inference_calls": 0,
        "visual_review": "PENDING_DIRECT_VIEW",
        "interpretation": "A compares same command/horizon. B contrast jointly changes supervisor and horizon; no population or full-policy claim.",
    }
    manifest = out.with_suffix(".manifest.json")
    if manifest.exists():
        raise FileExistsError("Refusing to overwrite figure manifest")
    manifest.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n")
    print(out)


if __name__ == "__main__":
    main()
