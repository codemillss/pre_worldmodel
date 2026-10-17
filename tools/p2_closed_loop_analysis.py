"""Derived P2-06 audit/figures; never changes original runs or source files."""
from pathlib import Path
import hashlib
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime

os.environ.setdefault("MPLCONFIGDIR", "${LOCAL_TEMP}/push-align-p2-06-analysis/mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np

ROOT = Path(sys.argv[1]).resolve()
RUN = ROOT / "runs/p2-06-20261008-01"
REPORT = ROOT / "reports/p2_06_prediction_audit.json"
FIGURES = ROOT / "reports/figures"
DT, TOL = .1, 1e-9
read_hashes = {}


def read_text(path):
    content = path.read_bytes()
    read_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(content).hexdigest()
    return content.decode("utf-8")


def read_json(path):
    return json.loads(read_text(path))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values):
    values = np.asarray(values, np.float64)
    if not len(values):
        return {"n": 0, "mean": None, "p95": None, "max": None}
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite value in descriptive aggregate")
    return {"n": len(values), "mean": float(values.mean()),
            "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def error_stats(rows):
    position = [r["prediction_error"]["object_position_m"] for r in rows]
    yaw = [r["prediction_error"]["object_yaw_rad"] for r in rows]
    return {"object_position_m": describe(position), "object_yaw_rad": describe(yaw),
            "object_position_mm": describe(np.asarray(position)*1000),
            "object_yaw_deg": describe(np.degrees(yaw))}


def support_stats(rows, horizon):
    plans = [r["plan"] for r in rows if r["source"] == "model" and r.get("plan")]
    supports = [p["support"] for p in plans]
    pairs = sum(s["evaluated_finite_input_pairs"] for s in supports)
    outside = sum(s["outside_any_train_marginal_bound"] for s in supports)
    selected = [s["selected_candidate_outside_steps"] for s in supports
                if s["selected_candidate_outside_steps"] is not None]
    selected_outside = sum(selected)
    return {
        "model_applied_planning_steps": len(plans),
        "planning_steps_with_selected_horizon_outside_any_marginal": sum(x > 0 for x in selected),
        "planning_steps_with_selected_horizon_outside_ratio":
            sum(x > 0 for x in selected)/len(selected) if selected else None,
        "selected_predicted_input_steps": len(selected)*horizon,
        "selected_predicted_input_steps_outside_any_marginal": selected_outside,
        "selected_predicted_input_steps_outside_ratio":
            selected_outside/(len(selected)*horizon) if selected else None,
        "all_candidate_evaluated_input_pairs": pairs,
        "all_candidate_input_pairs_outside_any_marginal": outside,
        "all_candidate_input_pairs_outside_ratio": outside/pairs if pairs else None,
        "scope": "Empirical per-feature training bounds. Predicted inputs, not actual future states. "
                 "Neither joint support, independent observations, nor an OOD failure guarantee.",
    }


manifest = read_json(RUN / "manifest.json")
summary = read_json(RUN / "summary.json")
ledger = read_json(RUN / "ledger.json")
resources = read_json(RUN / "resources.json")
artifact_manifest = read_json(RUN / "artifact_hashes.json")
declared = manifest["cases"]
ids = [c["episode_id"] for c in declared]
if len(ids) != 10 or ids != [r["episode_id"] for r in ledger]:
    raise ValueError("All ten declared outcomes must remain in their original order")
config = manifest["resolved_planning_config"]
task = config["task"]
horizon = config["planner"]["horizon"]
episodes, all_rows, eligible_rows, exclusions = [], [], [], []
for identity in ids:
    folder = RUN / "episodes" / identity
    case_summary = read_json(folder / "summary.json")
    rows = [json.loads(line) for line in read_text(folder / "steps.jsonl").splitlines() if line]
    if [r["index"] for r in rows] != list(range(len(rows))):
        raise ValueError("Noncontiguous original control indices")
    case_eligible, case_exclusions = [], []
    for row in rows:
        reasons = []
        if row["source"] != "model":
            reasons.append("no_model_prediction_holding")
        if row["prediction_error"] is None:
            reasons.append("prediction_error_missing")
        if row["physical_valid"] is not True:
            reasons.append("physical_invalid")
        if row["geometry"]["pass"] is not True:
            reasons.append("geometry_invalid")
        if abs(row["actual_dt_s"] - DT) > TOL:
            reasons.append("horizon_mismatch_partial_interval")
        if not reasons:
            error = row["prediction_error"]
            delta = np.asarray(error["state_error"], np.float64)
            if (delta.shape != (10,) or not np.isfinite(delta).all()
                    or abs(math.hypot(*delta[:2])-error["object_position_m"]) > 1e-12
                    or abs(abs(delta[2])-error["object_yaw_rad"]) > 1e-12):
                raise ValueError("Logged prediction error does not match the state difference")
            case_eligible.append(row)
        else:
            case_exclusions.append({"episode_id": identity, "control_index": row["index"],
                                    "reasons": reasons, "actual_dt_s": row["actual_dt_s"],
                                    "raw_prediction_error": row["prediction_error"],
                                    "task_terminal_reason": row["task"]["terminal_reason"],
                                    "task_violations": row["task"]["violations"]})
    episodes.append({"episode_id": identity, "summary": case_summary, "rows": rows,
                     "eligible": case_eligible, "exclusions": case_exclusions})
    all_rows.extend(rows)
    eligible_rows.extend(case_eligible)
    exclusions.extend(case_exclusions)

outcomes = Counter(r["task"]["terminal_reason"] for r in ledger)
audit = {
    "run_id": RUN.name, "generated_at": datetime.now().astimezone().isoformat(),
    "scope": "Derived descriptive analysis of all ten fixed development smoke executions. "
             "Not final test, statistical generalization, superiority, or user-understanding evidence.",
    "accuracy_inclusion": {
        "source": "model", "physical_valid": True, "geometry_pass": True,
        "prediction_error_present": True, "expected_actual_dt_s": DT,
        "actual_dt_tolerance_s": TOL,
        "meaning": "One-step prediction and measured endpoint at the same 0.1 s horizon.",
        "partial_interval_policy": "Raw prediction error retained, excluded from accuracy: "
                                   "the 0.1 s forecast and shorter execution do not share a horizon.",
    },
    "quantile_definition": "NumPy percentile 95 with its default linear interpolation.",
    "metric_definitions": {
        "prediction_position": "Euclidean norm of measured XY minus predicted XY, in world metres.",
        "prediction_yaw": "Absolute wrapped measured-minus-predicted yaw difference in [-pi,pi], radians.",
        "task_yaw": "Goal-relative pi-symmetric angle; distinct from the recorded physical prediction yaw metric.",
    },
    "analysis_environment": {"python": sys.version, "numpy": np.__version__,
                             "matplotlib": matplotlib.__version__,
                             "script_sha256": sha(Path(__file__)),
                             "script_path": str(Path(__file__).resolve())},
    "statistical_unit": "Ten declared episodes. Frame/step aggregates below are descriptive; "
                        "consecutive within-episode samples are not independent trials.",
    "task_outcomes": {"denominator_declared_episodes": 10, "terminal_counts": dict(outcomes),
                      "ready_count": outcomes.get("ready", 0),
                      "ready_fraction": outcomes.get("ready", 0)/10,
                      "missing_or_replaced_episodes": [], "all_original_endpoints_retained": True},
    "overall": {"recorded_execution_intervals": len(all_rows),
                "model_source_intervals": sum(r["source"] == "model" for r in all_rows),
                "holding_source_intervals": sum(r["source"] == "holding" for r in all_rows),
                "included_equal_horizon_predictions": len(eligible_rows),
                "excluded_from_accuracy": len(exclusions),
                "exclusion_reason_counts": dict(Counter(reason for e in exclusions for reason in e["reasons"])),
                "prediction_error": error_stats(eligible_rows),
                "train_marginal_support": support_stats(all_rows, horizon)},
    "per_episode": [], "excluded_records": exclusions, "resources_original": resources,
    "original_phase_gate_pass": summary["pass"],
    "original_artifact_manifest_sha256": read_hashes[str((RUN/"artifact_hashes.json").relative_to(ROOT))],
    "figure_selection_rule": "All ten declared episodes in original order; no successful-case selection.",
}
for episode in episodes:
    rows, case_summary = episode["rows"], episode["summary"]
    audit["per_episode"].append({
        "episode_id": episode["episode_id"], "recorded_execution_intervals": len(rows),
        "source_counts": dict(Counter(r["source"] for r in rows)),
        "included_equal_horizon_predictions": len(episode["eligible"]),
        "excluded_from_accuracy": len(episode["exclusions"]),
        "prediction_error": error_stats(episode["eligible"]),
        "train_marginal_support": support_stats(rows, horizon),
        "terminal_phase": case_summary["task"]["phase"],
        "terminal_reason": case_summary["task"]["terminal_reason"],
        "terminal_violations": case_summary["task"]["violations"],
        "elapsed_sim_s": case_summary["task"]["last_sim_time"]-case_summary["task"]["initial_sim_time"],
        "final_metrics_original": case_summary["final_metrics"],
        "final_observed_state": rows[-1]["state_after"] if rows else None,
        "final_actual_dt_s": rows[-1]["actual_dt_s"] if rows else None,
    })

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                     "axes.titlesize": 12, "axes.labelsize": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white",
                     "svg.fonttype": "none"})
colors = plt.get_cmap("tab10").colors
handles = [Line2D([0], [0], color=colors[i], lw=2, label=identity)
           for i, identity in enumerate(ids)]


def common_figure(title, subtitle, footer):
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.6))
    fig.subplots_adjust(left=.07, right=.97, top=.80, bottom=.12, wspace=.27, hspace=.38)
    fig.suptitle(title, x=.07, y=.985, ha="left", fontsize=19, fontweight="bold")
    fig.text(.07, .944, subtitle, ha="left", fontsize=10.5, color="#444444")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, .913),
               ncol=5, frameon=False, columnspacing=1.8, handlelength=2.5)
    fig.text(.07, .057, footer, ha="left", va="top", fontsize=9.7, color="#444444")
    for ax in axes.flat:
        ax.grid(True, alpha=.22)
    return fig, axes


def save_figure(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES/f"{name}.png", dpi=180)
    fig.savefig(FIGURES/f"{name}.svg")
    plt.close(fig)


goal_mm = np.asarray(task["goal_xy_m"])*1000
position_tolerance_mm = task["position_tolerance_m"]*1000
yaw_tolerance_deg = math.degrees(task["yaw_tolerance_rad"])
fig, axes = common_figure(
    "P2-06 | Closed-loop behavior", 
    "Ten fixed development executions | " + ", ".join(f"{key}: {count}/10" for key, count in outcomes.items()),
    "All raw endpoints retained, including failed/partial terminal intervals. Open circle: start; star: ready; x: other endpoint.\n"
    "Readiness also requires low linear/angular speed for 0.3 s; this plot does not establish generalization or baseline superiority.")
xy, final, position, yaw = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
for index, episode in enumerate(episodes):
    rows = episode["rows"]
    if not rows:
        continue
    states = np.asarray([rows[0]["state_before"], *[r["state_after"] for r in rows]], np.float64)
    start_time = episode["summary"]["task"]["initial_sim_time"]
    times = np.asarray([0., *[r["sim_time"]-start_time for r in rows]])
    xym = states[:, :2]*1000
    marker = "*" if episode["summary"]["task"]["terminal_reason"] == "ready" else "x"
    xy.plot(xym[:, 0], xym[:, 1], color=colors[index], lw=1.5, alpha=.85)
    xy.scatter(*xym[0], s=20, marker="o", facecolors="white", edgecolors=colors[index], zorder=4)
    xy.scatter(*xym[-1], s=85 if marker == "*" else 36, marker=marker, color=colors[index], zorder=5)
    final_metrics = episode["summary"]["final_metrics"]
    fm_pos = final_metrics["e_pos_m"]*1000
    fm_yaw = math.degrees(final_metrics["e_yaw_rad"])
    final.scatter(fm_pos, fm_yaw, marker=marker, s=100 if marker == "*" else 42,
                  color=colors[index], zorder=4)
    offsets = {1: (-24, 12), 4: (12, -12), 6: (-24, 3), 9: (14, 17)}
    offset = offsets.get(index, (5, 5 if index % 2 else -11))
    final.annotate(episode["episode_id"][-2:], (fm_pos, fm_yaw),
                   xytext=offset, textcoords="offset points", fontsize=9,
                   arrowprops={"arrowstyle": "-", "color": colors[index], "lw": .65}
                   if index in offsets else None)
    position_error = np.linalg.norm(states[:, :2]-goal_mm/1000, axis=1)*1000
    angular_error = np.abs(.5*np.arctan2(np.sin(2*(states[:, 2]-task["goal_yaw_rad"])),
                                       np.cos(2*(states[:, 2]-task["goal_yaw_rad"]))))
    position.plot(times, position_error, color=colors[index], lw=1.3, alpha=.85)
    yaw.plot(times, np.degrees(angular_error), color=colors[index], lw=1.3, alpha=.85)
xy.add_patch(Circle(goal_mm, position_tolerance_mm, facecolor="#dfefe7", edgecolor="#333333", lw=1.2, zorder=1))
xy.scatter(*goal_mm, marker="+", color="#222222", s=90, zorder=5)
xy.set(title="A. Object XY trajectories", xlabel="Object x (mm)", ylabel="Object y (mm)")
xy.set_aspect("equal", adjustable="datalim")
final.add_patch(Rectangle((0, 0), position_tolerance_mm, yaw_tolerance_deg,
                          facecolor="#dfefe7", edgecolor="none", zorder=0))
final.axvline(position_tolerance_mm, color="#444444", lw=1, ls="--")
final.axhline(yaw_tolerance_deg, color="#444444", lw=1, ls="--")
final.set(title="B. Final observed pose error (all ten)", xlabel="Position error (mm)", ylabel="Pi-symmetric yaw error (deg)")
position.axhline(position_tolerance_mm, color="#444444", lw=1, ls="--")
yaw.axhline(yaw_tolerance_deg, color="#444444", lw=1, ls="--")
position.set(title="C. Observed position error", xlabel="Elapsed simulation time (s)", ylabel="Position error (mm)")
yaw.set(title="D. Observed orientation error", xlabel="Elapsed simulation time (s)", ylabel="Pi-symmetric yaw error (deg)")
save_figure(fig, "p2_06_closed_loop_overview")

fig, axes = common_figure(
    "P2-06 | Prediction and planning diagnostics",
    f"0.1 s matched-horizon predictions: {len(eligible_rows)} | excluded execution intervals: {len(exclusions)} | all ten episodes shown",
    "Accuracy excludes partial/invalid/holding intervals; raw records and all ten task outcomes remain in the audit.\n"
    "Step aggregates are descriptive, not independent trials. Planning time excludes execution/recording; no real-device real-time claim.")
pos_ax, yaw_ax, bar_ax, time_ax = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
for index, episode in enumerate(episodes):
    start_time = episode["summary"]["task"]["initial_sim_time"]
    selected = episode["eligible"]
    times = [r["sim_time"]-start_time for r in selected]
    pos_ax.plot(times, [r["prediction_error"]["object_position_m"]*1000 for r in selected],
                color=colors[index], lw=1.1, alpha=.8)
    yaw_ax.plot(times, [math.degrees(r["prediction_error"]["object_yaw_rad"]) for r in selected],
                color=colors[index], lw=1.1, alpha=.8)
    plans = [r for r in episode["rows"] if r["source"] == "model" and r["plan"]]
    time_ax.plot([r["sim_time"]-r["actual_dt_s"]-start_time for r in plans],
                 [r["plan"]["wall_time_s"]*1000 for r in plans], color=colors[index], lw=1.1, alpha=.8)
pos_ax.set(title="A. One-step position prediction error", xlabel="Elapsed simulation time (s)", ylabel="Prediction error (mm)")
yaw_ax.set(title="B. One-step yaw prediction error", xlabel="Elapsed simulation time (s)", ylabel="Wrapped yaw prediction error (deg)")
x = np.arange(10)
means = [p["prediction_error"]["object_position_mm"]["mean"] for p in audit["per_episode"]]
p95s = [p["prediction_error"]["object_position_mm"]["p95"] for p in audit["per_episode"]]
bar_ax.bar(x-.16, means, width=.32, color=colors, alpha=.55)
bar_ax.bar(x+.16, p95s, width=.32, color=colors, alpha=1.)
bar_ax.set_xticks(x, [identity[-2:] for identity in ids])
bar_ax.set(title="C. Position error: mean (light) / p95 (solid)", xlabel="Smoke episode ID", ylabel="Prediction error (mm)")
time_ax.axhline(100., color="#444444", lw=1, ls="--")
time_ax.set(title="D. Model planning wall time", xlabel="Elapsed simulation time at plan start (s)", ylabel="Planning time (ms)")
save_figure(fig, "p2_06_prediction_errors")

for relative, digest in read_hashes.items():
    if sha(ROOT/relative) != digest:
        raise ValueError(f"Original input changed during analysis: {relative}")
audit["input_sha256"] = read_hashes
audit["original_inputs_unchanged_during_analysis"] = True
audit["figure_outputs"] = {
    str(path.relative_to(ROOT)): sha(path)
    for name in ("p2_06_closed_loop_overview", "p2_06_prediction_errors")
    for extension in ("png", "svg") for path in [FIGURES/f"{name}.{extension}"]
}
REPORT.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
print(json.dumps({"output": str(REPORT), "intervals": len(all_rows), "included": len(eligible_rows),
                  "excluded": len(exclusions), "terminal_counts": dict(outcomes),
                  "errors": audit["overall"]["prediction_error"],
                  "support": audit["overall"]["train_marginal_support"]}, ensure_ascii=False))
