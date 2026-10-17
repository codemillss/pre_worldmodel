"""Read-only analysis of independent-feedback timestep gate traces.

Usage: PYTHONPATH=src .venv/bin/python tools/p3_02_gate_analysis.py RUN_DIR
The JSON result is printed to stdout. Raw run artifacts, source/configuration,
and acceptance budgets are never modified. No simulator is constructed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_above(values, threshold, elapsed, *, tick_divisor=None):
    indices = np.flatnonzero(values > threshold)
    if not len(indices):
        return None
    index = int(indices[0])
    result = {"index": index, "elapsed_s": float(elapsed[index]),
              "difference": float(values[index]), "threshold": float(threshold)}
    if tick_divisor:
        result["action_tick"] = index // tick_divisor
    return result


def yaw_difference(first, second):
    # Physical orientation comparison, not the pi-symmetric handoff metric.
    angle = first - second
    return np.abs(np.arctan2(np.sin(angle), np.cos(angle)))


def inferred_locks(decisions):
    """Reconstruct locks from documented G reassessment increments.

    A lock is not saved directly in these diagnostic traces. Each increment
    follows relock=True, which copies b_target; annotate the inference rather
    than presenting it as a directly stored context value.
    """
    previous, count = None, 0
    rows = []
    for decision in decisions:
        before = previous
        if decision["geometry_context_reset"]:
            previous, count = None, 0
        elif decision["reassessment_count"] != count:
            previous = decision["b_target_m"]
            count = decision["reassessment_count"]
        rows.append({"before_m": before, "after_m": previous})
    return rows


def trace_changes(decisions):
    result = []
    for index, decision in enumerate(decisions):
        if index == 0 or any(decision[key] != decisions[index-1][key]
                             for key in ("mode_after", "holding")):
            result.append({"tick": index, "elapsed_s": decision["elapsed_s"],
                           "mode": decision["mode_after"], "holding": decision["holding"],
                           "transition_reason": decision["transition_reason"]})
    return result


def analyze_mode(folder, case, base_dt):
    mode = case["mode"]
    paths = [folder/f"{mode}_{leg}{suffix}" for leg in ("base", "half")
             for suffix in (".npz", "_policy.json")]
    before_hashes = {str(path): digest(path) for path in paths}
    with np.load(paths[0], allow_pickle=False) as archive:
        base = {key: archive[key].copy() for key in archive.files}
    with np.load(paths[2], allow_pickle=False) as archive:
        half = {key: archive[key].copy() for key in archive.files}
    bd = json.loads(paths[1].read_text())["decisions"]
    hd = json.loads(paths[3].read_text())["decisions"]
    count = len(bd)
    if count != len(hd) or base["state"].shape != half["state"].shape:
        raise ValueError("Cannot compare unequal feedback trace lengths")
    position = np.linalg.norm(base["state"][:, :2]-half["state"][:, :2], axis=1)
    yaw = yaw_difference(base["state"][:, 2], half["state"][:, 2])
    commands = np.linalg.norm(base["action_requested"]-half["action_requested"], axis=1)
    starts = np.asarray([row["elapsed_s"] for row in bd], np.float64)
    ends = np.cumsum(base["actual_dt"])
    # The half-dt trace records twice as many substeps; every second half
    # substep corresponds to one base-dt substep at the same physical time.
    half_shared = half["substep_states"][1::2]
    if half_shared.shape != base["substep_states"].shape:
        raise ValueError("Shared substep shape does not match a dt/2 comparison")
    time_difference = np.max(np.abs(base["substep_times"]-half["substep_times"][1::2]))
    if time_difference > 1e-8:
        raise ValueError("Substeps are not aligned in physical time")
    substep_elapsed = base["substep_times"]-base["substep_times"][0]+base_dt
    sub_position = np.linalg.norm(base["substep_states"][:, :2]-half_shared[:, :2], axis=1)
    sub_yaw = yaw_difference(base["substep_states"][:, 2], half_shared[:, 2])
    substeps_per_action = round(.1/base_dt)
    discrete = [index for index, (b, h) in enumerate(zip(bd, hd))
                if b["mode_after"] != h["mode_after"] or b["holding"] != h["holding"]]
    first_discrete = discrete[0] if discrete else None
    budget = case["budgets"]
    result = {
        "mode": mode, "episode_id": case["episode_id"], "recorded_gate_pass": case["pass"],
        "full_equal_execution_lengths": [len(bd), len(hd)],
        "initial_integration_state_identical": bool(np.array_equal(
            base["initial_integration_state"], half["initial_integration_state"])),
        "maximum_shared_time_difference_s": float(time_difference),
        "first_boundary_position_budget_exceeded": first_above(position, budget["position_m"], ends),
        "first_boundary_yaw_budget_exceeded": first_above(yaw, budget["yaw_rad"], ends),
        "first_shared_substep_position_budget_exceeded": first_above(
            sub_position, budget["position_m"], substep_elapsed, tick_divisor=substeps_per_action),
        "first_shared_substep_yaw_budget_exceeded": first_above(
            sub_yaw, budget["yaw_rad"], substep_elapsed, tick_divisor=substeps_per_action),
        "first_exact_command_difference": first_above(commands, 0., starts),
        "first_command_difference_above_1e_minus_6_m_s": first_above(commands, 1e-6, starts),
        "first_command_difference_above_1e_minus_3_m_s": first_above(commands, 1e-3, starts),
        "command_display_thresholds_are_not_acceptance_gates": True,
        "max_position_difference_mm": float(position.max()*1000),
        "max_yaw_difference_deg": float(np.rad2deg(yaw.max())),
        "max_command_difference_m_s": float(commands.max()),
        "any_holding_base_or_half": any(row["holding"] for row in bd+hd),
        "base_phase_changes": trace_changes(bd), "half_phase_changes": trace_changes(hd),
    }
    for field in ("mode_after", "holding", "noise_applied", "transition_reason"):
        indices = [index for index, (b, h) in enumerate(zip(bd, hd)) if b[field] != h[field]]
        result[f"first_{field}_difference"] = None if not indices else {
            "tick": indices[0], "elapsed_s": float(starts[indices[0]]),
            "base": bd[indices[0]][field], "half": hd[indices[0]][field],
            "different_tick_count": len(indices),
        }
    if first_discrete is not None:
        index = first_discrete
        before_position = 0. if not index else float(position[index-1])
        before_yaw = 0. if not index else float(yaw[index-1])
        result["first_discrete_branch"] = {
            "tick": index, "start_elapsed_s": float(starts[index]),
            "input_object_position_difference_mm": before_position*1000,
            "input_object_yaw_difference_deg": float(np.rad2deg(before_yaw)),
            "max_prebranch_boundary_position_difference_mm": float(position[:index].max()*1000),
            "max_prebranch_boundary_yaw_difference_deg": float(np.rad2deg(yaw[:index].max())),
            "base_decision": bd[index], "half_decision": hd[index],
            "base_locked_offset_inference": inferred_locks(bd)[index],
            "half_locked_offset_inference": inferred_locks(hd)[index],
            "lock_basis": "G reassessment increments imply relock to recorded b_target; inferred, not stored context",
        }
    result["boundary_profile"] = [{
        "tick": index, "start_elapsed_s": float(starts[index]), "end_elapsed_s": float(ends[index]),
        "object_position_difference_mm": float(position[index]*1000),
        "object_yaw_difference_deg": float(np.rad2deg(yaw[index])),
        "requested_command_difference_m_s": float(commands[index]),
        "base_mode": bd[index]["mode_after"], "half_mode": hd[index]["mode_after"],
        "base_noise_applied": bd[index]["noise_applied"],
        "half_noise_applied": hd[index]["noise_applied"],
    } for index in range(count)]
    result["input_sha256"] = before_hashes
    result["inputs_unchanged"] = all(digest(Path(path)) == hash_ for path, hash_ in before_hashes.items())
    return result


def analyze_replay(folder, case, gate_folder, base_dt):
    mode = case["mode"]
    paths = [folder/f"{mode}_{leg}.npz" for leg in ("base", "half")]
    hashes = {str(path): digest(path) for path in paths}
    with np.load(paths[0], allow_pickle=False) as archive:
        base = {key: archive[key].copy() for key in archive.files}
    with np.load(paths[1], allow_pickle=False) as archive:
        half = {key: archive[key].copy() for key in archive.files}
    with np.load(gate_folder/f"{mode}_base.npz", allow_pickle=False) as original:
        exact_nominal = all(np.array_equal(base[key], original[key]) for key in
                            ("state", "action_requested", "action_applied", "actual_dt"))
    if base["state"].shape != half["state"].shape:
        raise ValueError("Replay length mismatch")
    requested_identical = bool(np.array_equal(base["action_requested"], half["action_requested"]))
    if not requested_identical:
        raise ValueError("Fixed-command diagnostic did not preserve requested commands")
    position = np.linalg.norm(base["state"][:, :2]-half["state"][:, :2], axis=1)
    yaw = yaw_difference(base["state"][:, 2], half["state"][:, 2])
    ends = np.cumsum(base["actual_dt"])
    sub_position = np.linalg.norm(base["substep_states"][:, :2]-half["substep_states"][1::2, :2], axis=1)
    sub_yaw = yaw_difference(base["substep_states"][:, 2], half["substep_states"][1::2, 2])
    sub_times = base["substep_times"]-base["substep_times"][0]+base_dt
    budget = case["budgets"]
    return {
        "mode": mode, "episode_id": case["episode_id"], "recorded_diagnostic_pass": case["pass"],
        "nominal_arrays_reproduced_exactly": bool(exact_nominal),
        "requested_commands_identical": requested_identical,
        "applied_commands_identical": bool(np.array_equal(base["action_applied"], half["action_applied"])),
        "max_position_difference_mm": float(position.max()*1000),
        "max_yaw_difference_deg": float(np.rad2deg(yaw.max())),
        "first_boundary_position_budget_exceeded": first_above(position, budget["position_m"], ends),
        "first_boundary_yaw_budget_exceeded": first_above(yaw, budget["yaw_rad"], ends),
        "first_shared_substep_position_budget_exceeded": first_above(sub_position, budget["position_m"], sub_times),
        "first_shared_substep_yaw_budget_exceeded": first_above(sub_yaw, budget["yaw_rad"], sub_times),
        "boundary_profile": [{"end_elapsed_s": float(ends[index]),
                              "object_position_difference_mm": float(position[index]*1000),
                              "object_yaw_difference_deg": float(np.rad2deg(yaw[index]))}
                             for index in range(len(ends))],
        "input_sha256": hashes,
        "inputs_unchanged": all(digest(Path(path)) == hash_ for path, hash_ in hashes.items()),
    }


def figures(output, gate_summary, figure_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.titlesize": 13, "axes.labelsize": 12,
                         "svg.fonttype": "none", "figure.facecolor": "white"})
    figure_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    replay = {row["mode"]: row for row in output.get("fixed_command_replay", [])}
    gate_rows = gate_summary["cases"]
    labels = ["Noncontact", "Central", "Eccentric", "Contact\ntransition",
              "G\nfeedback", "G + noise"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 8.5))
    fig.suptitle("P3-02 physics gate FAIL | main collection NOT_RUN", fontweight="bold", fontsize=14)
    x = np.arange(len(gate_rows))
    for axis, key, scale, title, ylabel, budget in (
            (axes[0], "max_position_difference_m", 1000., "Maximum XY difference", "XY difference (mm)", .5),
            (axes[1], "max_yaw_difference_rad", 180/np.pi, "Maximum physical-yaw difference", "Yaw difference (degrees)", .3)):
        values = [row[key]*scale for row in gate_rows]
        colors = ["#237ba3" if row["pass"] else "#c84345" for row in gate_rows]
        axis.bar(x, values, width=.56, color=colors)
        for index, value in enumerate(values):
            axis.text(index, value+.045*max(values), f"{value:.3g}", ha="center", fontsize=11)
        for index, row in enumerate(gate_rows):
            if row["mode"] not in replay:
                continue
            record = replay[row["mode"]]
            value = record["max_position_difference_mm"] if scale == 1000. else record["max_yaw_difference_deg"]
            axis.bar(index+.32, value, width=.23, color="#e1a42a", hatch="///", edgecolor="#7a5b18")
            axis.text(index+.32, value+.015*max(values), f"{value:.3g}", ha="center", fontsize=11)
        axis.axhline(budget, color="#a22326", ls="--", lw=1.5, label=f"Unchanged budget: {budget:g}")
        axis.set_xticks(x, labels)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.set_ylim(0, max(values)*1.24)
        axis.grid(axis="y", alpha=.2)
        axis.legend(loc="upper left", fontsize=12)
    handles = [Patch(color="#237ba3", label="Original timestep gate: PASS"),
               Patch(color="#c84345", label="Original timestep gate: FAIL"),
               Patch(facecolor="#e1a42a", hatch="///", edgecolor="#7a5b18", label="Separate fixed-command replay diagnostic")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .014), ncol=2, fontsize=12)
    fig.subplots_adjust(left=.12, right=.98, top=.91, bottom=.225, hspace=.43)
    fig.text(.5, .117, "Maxima at 0.1 s action boundaries. Six frozen representatives.\n"
             "Four open-loop cases: 1.6 s; two feedback cases: 12.8 s. No full-domain validation.",
             ha="center", fontsize=11)
    for suffix in ("png", "svg"):
        path = figure_dir/f"p3_02_physics_gate.{suffix}"
        fig.savefig(path, dpi=160)
        artifacts[str(path)] = digest(path)
    plt.close(fig)
    for case in output["cases"]:
        fig, axes = plt.subplots(3, 1, figsize=(9, 8.4))
        profile = case["boundary_profile"]
        t = np.array([row["end_elapsed_s"] for row in profile])
        start = np.array([row["start_elapsed_s"] for row in profile])
        branch = case["first_discrete_branch"]["start_elapsed_s"]
        replay_case = replay.get(case["mode"])
        name = "G feedback | main-episode-008" if case["mode"] == "geometry_feedback" else "G + noise | main-episode-009"
        fig.suptitle(name+"\nIndependent feedback and fixed-command replay", fontweight="bold", fontsize=14)
        for row, key, ylabel, budget in (
                (0, "object_position_difference_mm", "XY difference (mm)", .5),
                (1, "object_yaw_difference_deg", "Yaw difference (deg)", .3)):
            axis = axes[row]
            axis.plot(t, [row[key] for row in profile], color="#c84345", lw=1.8, label="Independent feedback")
            if replay_case:
                axis.plot([row["end_elapsed_s"] for row in replay_case["boundary_profile"]],
                          [row[key] for row in replay_case["boundary_profile"]], color="#58717c", ls="--", lw=1.5,
                          label="Same stored commands")
            axis.axhline(budget, color="#a22326", ls=":", lw=1.5, label=f"Budget {budget:g}")
            axis.axvline(branch, color="#8b57a3", ls="-.", lw=1.2, label=f"First phase branch {branch:.1f} s")
            axis.set_ylabel(ylabel)
            axis.set_xlabel("Episode time after reset settling (s)")
            axis.set_xlim(0, 12.8)
            axis.set_ylim(bottom=0)
            axis.grid(alpha=.2)
            axis.text(.18, budget+.02*axis.get_ylim()[1],
                      f"Unchanged budget: {budget:g} {'mm' if row == 0 else 'deg'}", fontsize=11)
        axis = axes[2]
        mode_values = {"RETRACT": 0, "SLIDE": 1, "APPROACH": 2, "PUSH": 3}
        axis.step(start, [mode_values[row["base_mode"]] for row in profile], where="post", lw=2,
                  color="#237ba3", label="dt = 0.5 ms")
        axis.step(start, [mode_values[row["half_mode"]] for row in profile], where="post", lw=1.7,
                  color="#e1a42a", ls="--", label="dt = 0.25 ms")
        axis.axvline(branch, color="#8b57a3", ls="-.", lw=1.2)
        axis.set_yticks(range(4), ["RETRACT", "SLIDE", "APPROACH", "PUSH"])
        axis.set_ylim(-1.05, 3.5)
        axis.set_xlim(0, 12.8)
        axis.set_title(f"Observed-state phase branch at {branch:.1f} s")
        axis.set_xlabel("Decision time after reset settling (s)")
        axis.grid(alpha=.2)
        axis.legend(fontsize=12, loc="lower left", ncol=2)
        handles = [Line2D([0], [0], color="#c84345", lw=1.8, label="Independent feedback"),
                   Line2D([0], [0], color="#58717c", ls="--", lw=1.5, label="Same stored commands"),
                   Line2D([0], [0], color="#a22326", ls=":", lw=1.5, label="Unchanged budget"),
                   Line2D([0], [0], color="#8b57a3", ls="-.", lw=1.2, label=f"First phase branch: {branch:.1f} s")]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.57, .917), ncol=2, fontsize=12)
        fig.subplots_adjust(left=.16, right=.98, top=.825, bottom=.135, hspace=.47)
        fig.text(.5, .018, "No holding. Differences compare dt to dt/2, not goal error.\n"
                 "Original gate remains FAIL; fixed-command replay is a separate diagnosis.",
                 ha="center", fontsize=11)
        stem = "p3_02_feedback_branch" if case["mode"] == "geometry_feedback" else "p3_02_perturbed_branch"
        for suffix in ("png", "svg"):
            path = figure_dir/f"{stem}.{suffix}"
            fig.savefig(path, dpi=160)
            artifacts[str(path)] = digest(path)
        plt.close(fig)
    output["figures_sha256"] = artifacts
    output["analysis_environment"] = {"python": sys.version, "numpy": np.__version__,
                                      "matplotlib": matplotlib.__version__}
    output["figure_readability"] = {
        "source_figure_width_in": 9.0,
        "minimum_source_text_pt": 11.0,
        "legend_source_text_pt": 12.0,
        "assumed_pdf_body_width_in": 6.5,
        "minimum_text_pt_at_assumed_pdf_width": 11.0*6.5/9.0,
        "legend_text_pt_at_assumed_pdf_width": 12.0*6.5/9.0,
        "scope": "width scaling estimate; final PDF layout and visual inspection remain separate",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--replay-run", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--figures-dir", type=Path)
    args = parser.parse_args()
    folder = args.run_dir/"physics_extension"
    summary_path = folder/"summary.json"
    summary_hash = digest(summary_path)
    summary = json.loads(summary_path.read_text())
    run_summary_path = args.run_dir/"summary.json"
    run_summary_hash = digest(run_summary_path)
    run_summary = json.loads(run_summary_path.read_text())
    base_dt = float(summary["half_config"]["physics"]["timestep_s"])*2
    cases = [case for case in summary["cases"]
             if case["mode"] in {"geometry_feedback", "geometry_perturbed"}]
    output = {
        "run_dir": str(args.run_dir), "summary_sha256": summary_hash,
        "run_summary_sha256": run_summary_hash,
        "original_stage_verdict": run_summary["stage_verdict"],
        "main_collection": run_summary["main_collection"],
        "original_gate_counts": {"pass": sum(case["pass"] for case in summary["cases"]),
                                 "fail": sum(not case["pass"] for case in summary["cases"]),
                                 "total": len(summary["cases"])},
        "original_six_cases": summary["cases"],
        "analysis_script_sha256": digest(Path(__file__)),
        "scope": "independent-feedback dt/2 diagnosis; original gate remains failed; fixed-command replay is separate, bounded evidence",
        "physical_yaw_comparison": "wrapped 2pi; handoff pi-symmetry is not used",
        "cases": [analyze_mode(folder, case, base_dt) for case in cases],
        "summary_unchanged": digest(summary_path) == summary_hash,
    }
    if args.replay_run:
        replay_summary_path = args.replay_run/"summary.json"
        replay_hash = digest(replay_summary_path)
        replay_summary = json.loads(replay_summary_path.read_text())
        output["fixed_command_replay_summary_sha256"] = replay_hash
        output["fixed_command_replay"] = [analyze_replay(args.replay_run, row, folder, base_dt)
                                           for row in replay_summary["rows"]]
        output["replay_summary_unchanged"] = digest(replay_summary_path) == replay_hash
        output["interpretation"] = {
            "geometry_feedback": "Small prebranch differences cross offset-reassessment threshold; identical-command replay stays within budgets. Evidence supports branch amplification for this stored command sequence, not complete physical validity.",
            "geometry_perturbed": "Recovery and PUSH eligibility branch at different ticks; identical-command replay also exceeds both budgets. Branch amplification alone does not explain all differences; physical timestep sensitivity remains unresolved for this command sequence.",
            "acceptance": "Original 4/6 gate PASS, 2/6 FAIL; main collection NOT_RUN. Replay diagnosis does not override original verdict.",
        }
    if args.figures_dir:
        figures(output, summary, args.figures_dir)
    output["summary_unchanged"] = digest(summary_path) == summary_hash
    output["run_summary_unchanged"] = digest(run_summary_path) == run_summary_hash
    text = json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output_json:
        args.output_json.write_text(text+"\n")
        print(json.dumps({"output_json": str(args.output_json),
                          "original_gate": "FAIL", "figure_count": len(output.get("figures_sha256", {}))}))
    else:
        print(text)


if __name__ == "__main__":
    main()
