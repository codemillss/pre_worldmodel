"""Derived comparison of the frozen G transition attempts; never run physics.

Analyze with the project Python (--analyze). Plot the resulting JSON with a
Matplotlib runtime (--plot). Raw executions, source bundles and criteria are
read-only. Descriptive contact windows are not independent trials or training
admission, and no newly invented performance threshold is used here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import gc
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/"reports/p3_transition_analysis.json"
G_RUNS = ("p3-01-20261008-01", "p3-01-tuning-20261009-01", "p3-01-tuning-20261009-02")
GATE_RUNS = ("p3-02-gate-20261008-02", "p3-02-gate-20261009-01", "p3-02-gate-20261009-02")
CASE_IDS = ("g-translation", "g-yaw-positive", "g-yaw-negative", "g-recontact")
REGIMES = ("noncontact", "sustained_contact", "transition")
INPUTS = {}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    path = Path(path)
    INPUTS[str(path.relative_to(ROOT))] = sha(path)
    return path


def read(path):
    return json.loads(record(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def source_check(folder, manifest):
    mismatches = []
    for relative, expected in manifest.get("source_hashes", {}).items():
        file = folder/"source"/relative
        if record(file) and INPUTS[str(file.relative_to(ROOT))] != expected:
            mismatches.append(relative)
    if mismatches:
        raise ValueError(f"Frozen source changed: {folder.name}: {mismatches}")
    return {"matched_source_files": len(manifest.get("source_hashes", {})),
            "all_match_manifest": True}


def restore(raw, values):
    diagnostics = raw["diagnostics"]
    paths = raw["diagnostic_nonfinite_paths"]
    if len(paths) != len(values):
        raise ValueError("Raw nonfinite mapping differs")
    for path, value in zip(paths, values):
        target = diagnostics
        for key in path[:-1]:
            target = target[key]
        if target[path[-1]] is not None or np.isfinite(value):
            raise ValueError("Invalid raw nonfinite placeholder")
        target[path[-1]] = float(value)
    return diagnostics


def contact_measurements(diagnostics, threshold, tolerance):
    from push_align.coverage import _contact_events, _regimes
    regimes = _regimes(diagnostics, threshold)
    events = _contact_events(diagnostics, threshold, tolerance)
    times, labels = [], []
    for diagnostic in diagnostics:
        for sample in diagnostic["samples"]:
            forces = [c["normal_force_N"] for c in sample["contacts"] if c["pair"] == "pusher_object"]
            active = any(force > threshold for force in forces) if np.isfinite(forces).all() else None
            time = float(sample["sim_time"])
            if times and abs(time-times[-1]) <= tolerance:
                if active != labels[-1]:
                    raise ValueError("Raw duplicate boundary contact differs")
                continue
            if times and time <= times[-1]:
                raise ValueError("Raw contact timeline is not increasing")
            times.append(time); labels.append(active)
    # Conservative observed span, not dwell-time interpolation through flips.
    observed_active = sum(b-a for a, b, left, right in
                          zip(times, times[1:], labels, labels[1:]) if left is True and right is True)
    return regimes, {"force_threshold_N": threshold, "strictly_greater": True,
        "deduplicated_raw_samples": len(labels), "active_samples": sum(x is True for x in labels),
        "unknown_samples": sum(x is None for x in labels),
        "active_observed_span_s": float(observed_active),
        "active_span_rule": "sum intervals whose two deduplicated boundary samples are active; no extension over force flips",
        "qualified_recontacts": events["qualified_recontacts"],
        "raw_crossing_count": events["raw_crossing_count"]}


def measure(states, commands, actual_dt, valid, decisions, diagnostics, cfg):
    count = len(commands)
    if not (states.shape == (count+1, 10) and valid.shape == (count,) and
            len(decisions) == len(diagnostics) == count):
        raise ValueError("State/command/raw/policy lengths differ")
    if not np.isfinite(states).all() or not np.isfinite(commands).all():
        raise ValueError("These derived bounded-support comparisons require finite recorded states")
    task = cfg["task"]
    goal = np.asarray(task["goal_pose_xy_yaw"], np.float64) if "goal_pose_xy_yaw" in task else None
    if goal is None:
        goal = np.asarray([*task["goal_xy_m"], task["goal_yaw_rad"]], np.float64)
    before = states[:-1]
    position = np.linalg.norm(before[:, :2]-goal[:2], axis=1)
    yaw_error = .5*np.arctan2(np.sin(2*(before[:, 2]-goal[2])), np.cos(2*(before[:, 2]-goal[2])))
    pose = (position <= task["position_tolerance_m"]) & (np.abs(yaw_error) <= task["yaw_tolerance_rad"])
    dyaw = np.arctan2(np.sin(np.diff(states[:, 2])), np.cos(np.diff(states[:, 2])))
    stationary = valid & (np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1)<1e-9) & (np.abs(dyaw)<1e-9)
    zero = np.linalg.norm(commands, axis=1) <= 1e-12
    holding = np.array([bool(d.get("holding", d.get("reason")=="holding")) for d in decisions])
    modes = [d.get("mode_after") for d in decisions]
    regimes, contact = contact_measurements(diagnostics,
        cfg["evaluation"]["contact_active_normal_force_min_N"], cfg["validity"]["dt_match_tolerance_s"])
    window_starts = [i for i in range(count-4) if valid[i:i+5].all()]
    eligible = [bool(d.get("collection_mode")=="geometry_perturbed" and d.get("feasible", True)
                     and d.get("mode_after")=="PUSH" and not holding[i]) for i, d in enumerate(decisions)]
    weighted, applied_delta, flags = [], [], []
    eligible_gains, weighted_norms, delta_norms = [], [], []
    for index, diagnostic in enumerate(decisions):
        vector = np.asarray(diagnostic.get("predeclared_noise_m_s", [0., 0.]), float)
        gain = float(diagnostic.get("noise_gain", 1.)) if eligible[index] else 0.
        delta = np.asarray(diagnostic.get("applied_noise_delta_m_s", [0., 0.]), float)
        weighted.append(bool(np.linalg.norm(gain*vector)>1e-12))
        applied_delta.append(bool(np.linalg.norm(delta)>1e-12))
        flags.append(bool(diagnostic.get("noise_applied", False)))
        if eligible[index]:
            eligible_gains.append(gain)
            weighted_norms.append(float(np.linalg.norm(gain*vector)))
            delta_norms.append(float(np.linalg.norm(delta)))
    jumps = np.linalg.norm(np.diff(commands, axis=0), axis=1)
    transitions = Counter(t["reason"] for d in decisions for t in d.get("transitions", []))
    output = {"recorded_intervals": count, "valid_intervals": int(valid.sum()),
        "invalid_or_partial_intervals": int((~valid).sum()), "elapsed_s": float(actual_dt.sum()),
        "holding_intervals": int(holding.sum()), "zero_command_intervals": int(zero.sum()),
        "valid_zero_outside_goal_pose": int((valid & zero & ~pose).sum()),
        "valid_stationary_intervals": int(stationary.sum()),
        "valid_stationary_holding": int((stationary & holding).sum()),
        "valid_stationary_outside_goal_pose": int((stationary & ~pose).sum()),
        "stationary_rule": "valid; delta XY norm<1e-9m and abs(wrapped delta yaw)<1e-9rad; same existing main-audit rule",
        "zero_command_rule": "applied command norm<=1e-12m/s; descriptive numerical-zero diagnostic",
        "G_modes": dict(Counter(m for m in modes if m is not None)),
        "phase_transition_reasons": dict(transitions),
        "reassessment_max": max((d.get("reassessment_count", 0) for d in decisions), default=0),
        "retreat_reanchor_intervals": sum(bool(d.get("retreat_route_reanchored")) for d in decisions),
        "command_jump_count": len(jumps), "command_jump_max_m_s": float(jumps.max()) if len(jumps) else None,
        "command_jump_p95_m_s": float(np.quantile(jumps, .95)) if len(jumps) else None,
        "regime_counts_valid": {r: sum(bool(v) and g==r for v, g in zip(valid, regimes)) for r in REGIMES},
        "K5": {"total": len(window_starts), "by_first_regime": {
            r: sum(regimes[i]==r for i in window_starts) for r in REGIMES},
            "all_five_sustained_contact": sum(all(g=="sustained_contact" for g in regimes[i:i+5]) for i in window_starts),
            "nonstationary_window_count": sum(not stationary[i:i+5].all() for i in window_starts)},
        "actual_contact": contact, "noise": {"eligible_intervals": sum(eligible),
            "noise_applied_flag_intervals": sum(flags),
            "nonzero_weighted_candidate_intervals": sum(weighted),
            "nonzero_actual_delta_intervals": sum(applied_delta),
            "zero_gain_eligible_intervals": sum(e and not w for e, w in zip(eligible, weighted)),
            "eligible_gain_min_mean_max": [float(min(eligible_gains)), float(np.mean(eligible_gains)), float(max(eligible_gains))] if eligible_gains else None,
            "eligible_weighted_candidate_norm_mean_m_s": float(np.mean(weighted_norms)) if weighted_norms else None,
            "eligible_actual_delta_norm_mean_m_s": float(np.mean(delta_norms)) if delta_norms else None,
            "note": "eligible/flag may be true at zero gain; actual delta includes post-addition norm clipping"},
        "series": {"elapsed_s": np.cumsum(actual_dt).tolist(),
            "position_error_after_mm": (np.linalg.norm(states[1:, :2]-goal[:2], axis=1)*1000).tolist(),
            "mode": modes, "regime": regimes, "applied_norm_m_s": np.linalg.norm(commands, axis=1).tolist()}}
    return output


def g_case(run, case, cfg):
    folder = ROOT/"runs"/run/"episodes"/case
    rows = [json.loads(line) for line in record(folder/"steps.jsonl").read_text().splitlines()]
    summary = read(folder/"summary.json")
    states = np.asarray([r["state_before"] for r in rows]+[rows[-1]["state_after"]], float)
    for a, b in zip(rows, rows[1:]):
        if a["state_after"] != b["state_before"]:
            raise ValueError("G raw boundary states do not join")
    commands = np.asarray([r["applied_action"] for r in rows], float)
    actual_dt = np.asarray([r["actual_dt_s"] for r in rows], float)
    valid = np.array([r["physical_valid"] and r["geometry"]["pass"] and
        abs(r["actual_dt_s"]-.1)<=cfg["validity"]["dt_match_tolerance_s"] for r in rows])
    if (folder/"raw_execution.json.gz").exists():
        raw = json.loads(gzip.decompress(record(folder/"raw_execution.json.gz").read_bytes()))
        npz = record(folder/"raw_execution.npz")
        with np.load(npz, allow_pickle=False) as archive:
            if not np.array_equal(archive["state"], states[1:]) or not np.array_equal(archive["action_applied"], commands):
                raise ValueError("G per-step ledger differs from saved physical trace")
            diagnostics = restore(raw, archive["diagnostic_nonfinite_values"])
    else:
        metadata = read(ROOT/"runs"/run/"data/geometry-feasibility-v1/episodes"/(case+".json"))
        npz = record(ROOT/"runs"/run/"data/geometry-feasibility-v1/episodes"/(case+".npz"))
        if sha(npz) != metadata["npz_sha256"]:
            raise ValueError("Frozen G dataset NPZ hash mismatch")
        with np.load(npz, allow_pickle=False) as archive:
            if not np.array_equal(archive["state"], states) or not np.array_equal(archive["transition_valid"], valid):
                raise ValueError("G dataset and ledger masks differ")
            diagnostics = restore(metadata, archive["diagnostic_nonfinite_values"])
    decisions = [r["controller"] if r["controller"] is not None else
        {"holding": r["source"]=="holding", "reason": r["source"], "mode_after": None}
        for r in rows]
    result = measure(states, commands, actual_dt, valid, decisions, diagnostics, cfg)
    result.update(case_id=case, terminal_reason=summary["task"]["terminal_reason"],
        ready=summary["task"]["phase"]=="ready", final_metrics=summary["final_metrics"],
        held_time_s=summary["task"]["held_time_s"],
        scheduled_recovery=summary["recovery"],
        source_branch_counts=dict(Counter(r["source"] for r in rows)))
    unexecuted = folder/"unexecuted_final_decision.json"
    if unexecuted.exists():
        result["unexecuted_terminal_decision"] = read(unexecuted)
    gc.collect()
    return result


def gate_case(run, row, label, cfg):
    from push_align.contact_geometry import check_contact_geometry
    prefix = ROOT/"runs"/run/"physics_extension"/(row["mode"]+"_"+label)
    raw = json.loads(gzip.decompress(record(prefix.with_suffix(".json.gz")).read_bytes()))
    with np.load(record(prefix.with_suffix(".npz")), allow_pickle=False) as archive:
        post = archive["state"].copy(); commands = archive["action_applied"].copy()
        actual_dt = archive["actual_dt"].copy()
        diagnostics = restore(raw, archive["diagnostic_nonfinite_values"])
    policy = read(prefix.parent/(prefix.name+"_policy.json"))
    count = len(commands)
    states = np.vstack((np.asarray(diagnostics[0]["samples"][0]["state"], float), post))
    for i, diagnostic in enumerate(diagnostics):
        if diagnostic["samples"][0]["state"] != states[i].tolist() or diagnostic["samples"][-1]["state"] != states[i+1].tolist():
            raise ValueError("Gate raw samples and boundary trace differ")
    valid = np.array([not d["validity_violations"] and d["terminal_type"]=="completed_interval" and
        abs(dt-.1)<=cfg["validity"]["dt_match_tolerance_s"] and check_contact_geometry(d["samples"], .001)["pass"]
        for d, dt in zip(diagnostics, actual_dt)])
    result = measure(states, commands, actual_dt, valid, policy["decisions"][:count], diagnostics, cfg)
    result.update(terminal_reason=policy["terminal_reason"], episode_id=row["episode_id"], dt_label=label)
    return result


def analyze():
    sys.path.insert(0, str(ROOT/"src"))
    output = {"schema": "p3-transition-derived-v1", "created_at": datetime.now().astimezone().isoformat(),
        "scope": "three G4 versions and three gate attempts; same fixed conditions; no final performance, full-domain or new-data coverage claim",
        "interpretation": ["Lower command jumps cannot establish improvement when a run stops before ready.",
            "G4 success remains limited to four development cases; a failed six-representative dt gate blocks main collection.",
            "PUSH counts are policy modes; measured contact regimes require actual force>0.001N.",
            "K5 windows overlap and are not independent trials; first-regime counts do not assert all five intervals remain in that regime.",
            "Weighted noise changes the executed policy. Candidate vectors and family assignment remain fixed; support must be measured again."],
        "geometry_runs": {}, "gate_runs": {}}
    manifests = {}
    for run in G_RUNS:
        folder = ROOT/"runs"/run; manifest = read(folder/"manifest.json")
        manifests[run] = manifest
        summary = read(folder/"summary.json")
        provenance = source_check(folder, manifest)
        cases = [g_case(run, case, manifest["physical_config"]) for case in CASE_IDS]
        output["geometry_runs"][run] = {"pass": summary["pass"], "checks": summary["checks"],
            "source_verification": provenance, "cases": cases, "ready_cases": sum(c["ready"] for c in cases),
            "holding_intervals": sum(c["holding_intervals"] for c in cases),
            "recorded_intervals": sum(c["recorded_intervals"] for c in cases)}
    for run in GATE_RUNS:
        folder = ROOT/"runs"/run; manifest = read(folder/"manifest.json")
        manifests[run] = manifest
        summary = read(folder/"summary.json")
        result = {"stage_verdict": summary["stage_verdict"], "source_verification": source_check(folder, manifest)}
        if (folder/"physics_extension/summary.json").exists():
            extension = read(folder/"physics_extension/summary.json")
            result.update(pass_value=extension["pass"], cases=[])
            for row in extension["cases"]:
                derived = dict(row)
                derived["behavior"] = {label: gate_case(run, row, label,
                    manifest["contract"]["resolved_config"]) for label in ("base", "half")}
                result["cases"].append(derived)
        else:
            result.update(physics_executed=False, reason=summary.get("reason"))
            if (folder/"failure.json").exists():
                result["failure"] = read(folder/"failure.json")
            record(folder/"tests.txt")
        output["gate_runs"][run] = result
    old, new = manifests[GATE_RUNS[0]], manifests[GATE_RUNS[2]]
    output["comparison_checks"] = {
        "G_four_cases_identical": all(manifests[r]["cases"]==manifests[G_RUNS[0]]["cases"] for r in G_RUNS),
        "G_physical_configs_identical": all(manifests[r]["physical_config"]==manifests[G_RUNS[0]]["physical_config"] for r in G_RUNS),
        "G_sixteen_selected_parameters_identical": all(manifests[r]["resolved_controller_parameters"]==manifests[G_RUNS[0]]["resolved_controller_parameters"] for r in G_RUNS),
        "gate_ninety_predeclared_plans_identical": old["predeclared_episodes"]==new["predeclared_episodes"],
        "gate_family_assignments_identical": old["family_assignments"]==new["family_assignments"],
        "gate_raw_contract_identical": old["contract"]==new["contract"],
        "gate_budgets_identical": all(a["budgets"]==b["budgets"] for a, b in zip(
            output["gate_runs"][GATE_RUNS[0]]["cases"], output["gate_runs"][GATE_RUNS[2]]["cases"])),
        "scope": "transition command/noise policy deliberately differs; this is a compound candidate, not isolation of each component"}
    output["analysis_code_sha256"] = sha(Path(__file__))
    output["canonical_helper_sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in
        (ROOT/"src/push_align/coverage.py", ROOT/"src/push_align/contact_geometry.py")}
    output["input_sha256"] = INPUTS
    output["all_used_inputs_unchanged_after_analysis"] = all(sha(ROOT/p)==expected for p, expected in INPUTS.items())
    if not output["all_used_inputs_unchanged_after_analysis"]:
        raise ValueError("Frozen inputs changed during analysis")
    write(REPORT, output)
    print(json.dumps({"report": str(REPORT), "inputs_hashed": len(INPUTS),
        "G_ready": {r: v["ready_cases"] for r, v in output["geometry_runs"].items()},
        "gate_verdicts": {r: v["stage_verdict"] for r, v in output["gate_runs"].items()}}))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = json.loads(REPORT.read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12,
        "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "svg.fonttype": "none", "figure.dpi": 180})
    folder = ROOT/"reports/figures"; folder.mkdir(exist_ok=True)
    files = []

    def save(fig, name):
        for extension in ("png", "svg"):
            path = folder/(name+"."+extension)
            fig.savefig(path, dpi=180, facecolor="white")
            files.append(path)
        plt.close(fig)

    colors = ["#466480", "#bd554d", "#277b69"]
    labels = ["Legacy G / 4 ready", "Smooth01 / 4 failed", "Smooth02 / 4 ready"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.3)); x = np.arange(4)
    for i, run in enumerate(G_RUNS):
        rows = data["geometry_runs"][run]["cases"]
        values = [r["elapsed_s"] for r in rows]
        axes[0].bar(x+(i-1)*.23, values, width=.23, label=labels[i], color=colors[i])
        for j, row in enumerate(rows):
            axes[0].text(x[j]+(i-1)*.23, values[j]+.4, f"{values[j]:.1f}", ha="center", fontsize=10)
        axes[1].bar(x+(i-1)*.23, [r["final_metrics"]["e_pos_m"]*1000 for r in rows], width=.23, color=colors[i])
    axes[0].axhline(30, color="#555", linestyle="--", linewidth=1)
    axes[0].set_ylim(0, 33); axes[0].set_ylabel("Recorded duration (s)")
    axes[0].legend(loc="upper left", ncol=1)
    axes[0].set_title("Same four task cases: restored readiness requires extra time")
    axes[1].axhline(5, color="#555", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Final position error (mm)"); axes[1].set_title("30 s and 5 mm unchanged; readiness also checks yaw, speed and hold")
    for ax in axes:
        ax.set_xticks(x, ["Translation", "+10deg yaw", "-10deg yaw", "Forced recontact"])
        ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    fig.text(.08, .015, "Red bars end at controller-infeasible: shorter duration means failure, not faster completion.", fontsize=11)
    fig.tight_layout(rect=(0, .045, 1, 1)); save(fig, "p3_transition_task_cases")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7.0)); x = np.arange(6)
    for index, run in enumerate((GATE_RUNS[0], GATE_RUNS[2])):
        rows = data["gate_runs"][run]["cases"]
        for axis, key, scale in ((axes[0], "max_position_difference_m", 1000),
                                (axes[1], "max_yaw_difference_rad", 180/math.pi)):
            axis.bar(x+(index-.5)*.32, [r[key]*scale for r in rows], .32,
                     color=colors[index*2], label="Legacy G" if index==0 else "Smooth02")
    axes[0].axhline(.5, color="#bd554d", linestyle="--", label="Unchanged budget: 0.5 mm")
    axes[1].axhline(.3, color="#bd554d", linestyle="--", label="Unchanged budget: 0.3 deg")
    for ax in axes:
        ax.set_xticks(x, ["No contact", "Central", "Eccentric", "Contact\ntransition", "G feedback", "G+noise"])
        ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True); ax.legend(loc="upper left", ncol=3)
    axes[0].set_ylabel("Max dt position difference (mm)"); axes[0].set_title("Same six frozen representatives: independent feedback at each dt")
    axes[1].set_ylabel("Max dt yaw difference (deg)")
    fig.text(.07, .015, "Both G representatives remain FAIL. Full 16/128 intervals; same cases and budgets.", fontsize=11)
    fig.tight_layout(rect=(0, .045, 1, 1)); save(fig, "p3_transition_dt_gate")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7.2)); x = np.arange(4)
    chosen = [(run, mode) for mode in ("geometry_feedback", "geometry_perturbed") for run in (GATE_RUNS[0], GATE_RUNS[2])]
    rows = [next(r for r in data["gate_runs"][run]["cases"] if r["mode"]==mode)["behavior"]["base"] for run, mode in chosen]
    bottom = np.zeros(4)
    for regime, color in zip(REGIMES, ("#bac6d2", "#277b69", "#d3a05b")):
        values = np.array([r["regime_counts_valid"][regime] for r in rows])
        axes[0].bar(x, values, bottom=bottom, color=color, label=regime.replace("_", " "))
        for i, value in enumerate(values):
            if value:
                axes[0].text(i, bottom[i]+value/2, str(value), ha="center", va="center", fontsize=11)
        bottom += values
    axes[0].set_ylabel("Valid 0.1 s intervals"); axes[0].set_ylim(0, 147); axes[0].legend(loc="upper center", ncol=3)
    axes[0].set_title("Actual force regimes differ from controller PUSH mode")
    width = .2
    fields = [("K5 windows", [r["K5"]["by_first_regime"]["sustained_contact"] for r in rows]),
              ("Noise eligible", [r["noise"]["eligible_intervals"] for r in rows]),
              ("Nonzero noise delta", [r["noise"]["nonzero_actual_delta_intervals"] for r in rows])]
    for i, (label, values) in enumerate(fields):
        axes[1].bar(x+(i-1)*width, values, width, label=label)
        for j, value in enumerate(values):
            axes[1].text(x[j]+(i-1)*width, value+.7, str(value), ha="center", fontsize=10)
    axes[1].set_ylabel("Descriptive count"); axes[1].legend(loc="upper center", ncol=3); axes[1].set_ylim(0, 145)
    axes[1].set_title("K5: valid window whose first interval is sustained contact")
    for ax in axes:
        ax.set_xticks(x, ["Legacy G", "Smooth02 G", "Legacy G+noise", "Smooth02 G+noise"])
        ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    fig.text(.055, .015, "Base dt only, one representative each; overlapping windows and rows are not independent samples.", fontsize=11)
    fig.tight_layout(rect=(0, .045, 1, 1)); save(fig, "p3_transition_contact_support")
    audit = {"analysis_report_sha256": sha(REPORT), "analysis_code_sha256": sha(Path(__file__)),
        "figures": {str(p.relative_to(ROOT)): sha(p) for p in files},
        "figure_width_inches": 9., "minimum_label_font_pt": 10.,
        "anticipated_pdf_width_pt": 480., "minimum_label_pt_at_pdf_width": 10*480/(9*72),
        "scope": "scientific figures derived from the cited JSON; visual QA required separately"}
    write(ROOT/"reports/p3_transition_figures_audit.json", audit)
    print(json.dumps(audit))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--plot", action="store_true"); args = parser.parse_args()
    if not args.analyze and not args.plot:
        parser.error("Choose --analyze and/or --plot")
    if args.analyze:
        analyze()
    if args.plot:
        plot()
