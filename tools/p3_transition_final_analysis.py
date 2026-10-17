"""Separate derived audit for clock03, speed-cap04, formal G and optional main.

The earlier 01/02 audit stays immutable. Run --analyze after the formal G
summary exists; add --main-run only after its summary completes. Plot with a
Matplotlib runtime using --plot. This tool reads raw records and never invokes
physics or alters source, acceptance criteria, policies, or learned models.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import gc
import gzip
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

import p3_transition_analysis as h

ROOT = h.ROOT
REPORT = ROOT/"reports/p3_transition_final_analysis.json"
G_RUNS = ("p3-01-tuning-20261009-03", "p3-01-tuning-20261009-04", "p3-01-20261009-01")
GATE_RUNS = ("p3-02-gate-20261009-03", "p3-02-gate-20261009-04")


def historical_snapshot():
    report = ROOT/"reports/p3_transition_analysis.json"
    history = h.read(report)
    audit = h.read(ROOT/"reports/p3_transition_figures_audit.json")
    if h.sha(report) != audit["analysis_report_sha256"]:
        raise ValueError("Historical analysis and figures have inconsistent provenance")
    for relative, expected in {**history["input_sha256"], **audit["figures"]}.items():
        path = h.record(ROOT/relative)
        if h.sha(path) != expected:
            raise ValueError(f"Historical frozen input/output changed: {relative}")
    if h.sha(ROOT/"tools/p3_transition_analysis.py") != history["analysis_code_sha256"]:
        raise ValueError("Historical analysis source changed")
    for relative, expected in history["canonical_helper_sha256"].items():
        path = h.record(ROOT/relative)
        if h.sha(path) != expected:
            raise ValueError(f"Runtime contact/geometry measurement helper changed: {relative}")
    return history, {"path": str(report.relative_to(ROOT)), "sha256": h.sha(report),
        "preserved_input_files": len(history["input_sha256"]),
        "all_01_02_inputs_and_six_figure_outputs_unchanged": True,
        "attempt01_failure_preserved": history["geometry_runs"]["p3-01-tuning-20261009-01"]["ready_cases"]==0
            and history["gate_runs"]["p3-02-gate-20261009-01"]["stage_verdict"]=="ERROR"}


def phase_audit(run, mode):
    folder = ROOT/"runs"/run/"physics_extension"
    policies = {label: h.read(folder/(mode+"_"+label+"_policy.json"))["decisions"] for label in ("base", "half")}
    common = min(map(len, policies.values()))
    deadlines = {label: [i for i, d in enumerate(rows) if any(
        t["reason"]=="phase_time_reassessment" for t in d.get("transitions", []))]
        for label, rows in policies.items()}
    mode_divergence = [i for i in range(common) if policies["base"][i].get("mode_after") != policies["half"][i].get("mode_after")]
    noise_divergence = [i for i in range(common) if policies["base"][i].get("noise_applied", False) != policies["half"][i].get("noise_applied", False)]
    deadline_divergence = sorted(set(deadlines["base"]) ^ set(deadlines["half"]))
    chosen = mode_divergence[0] if mode_divergence else None
    return {"common_decisions": common, "first_mode_divergence_tick": chosen,
        "deadline_reassessment_ticks": deadlines, "deadline_tick_disagreements": deadline_divergence,
        "noise_eligibility_disagreement_ticks": noise_divergence,
        "scheduling_clock_snapped_intervals": {label: sum(bool(d.get("clock_snapped")) for d in rows) for label, rows in policies.items()},
        "first_mode_divergence_details": {label: {k: policies[label][chosen].get(k) for k in (
            "mode_before", "mode_after", "transition_reason", "phase_age_s", "raw_elapsed_s", "scheduling_elapsed_s",
            "context_phase_start_raw_s", "context_phase_start_scheduling_s", "b_target_m", "noise_gain")}
            for label in policies} if chosen is not None else None,
        "scope": "0-based policy ticks; phase metadata may differ while smoothed commands remain close; no full-domain stability assertion"}


def main_analysis(run):
    folder = ROOT/"runs"/run
    if not folder.resolve().is_relative_to((ROOT/"runs").resolve()):
        raise ValueError("Main run must be under project runs")
    summary = h.read(folder/"summary.json")
    manifest = h.read(folder/"manifest.json")
    if "dataset_manifest" not in summary:
        return {"run": run, "stage_verdict": summary["stage_verdict"], "data_collected": False,
                "reason": summary.get("reason"), "source_verification": h.source_check(folder, manifest)}
    data_manifest_path = (folder/summary["dataset_manifest"]).resolve()
    if not data_manifest_path.is_relative_to(folder.resolve()):
        raise ValueError("Dataset manifest escapes main run")
    dataset = data_manifest_path.parent; metadata = h.read(data_manifest_path)
    if h.sha(data_manifest_path) != summary["dataset_manifest_sha256"]:
        raise ValueError("Main dataset manifest differs from completed summary")
    coverage = h.read(folder/"coverage.json"); support = h.read(folder/"input_support.json")
    preparation = h.read(folder/"preparation.json")
    if metadata["contract"] != manifest["contract"]:
        raise ValueError("Main dataset contract differs from frozen run contract")
    plans = {p["episode_id"]: p for p in manifest["predeclared_episodes"]}
    expected_coverage = {p["episode_id"]: p for p in coverage["episodes"]}
    behavior = []
    for entry in metadata["episodes"]:
        episode = entry["episode_id"]
        plan = plans[episode]
        npz_path, json_path = dataset/entry["npz"], dataset/entry["json"]
        for path, expected in ((npz_path, entry["npz_sha256"]), (json_path, entry["json_sha256"])):
            if not path.resolve().is_relative_to(dataset.resolve()) or h.sha(h.record(path)) != expected:
                raise ValueError("Main episode artifact hash/path mismatch")
        raw = json.loads(json_path.read_text())
        if raw["contract"] != metadata["contract"]:
            raise ValueError("Main episode contract differs from dataset")
        if any(raw[key] != entry[key] or raw[key] != plan[key] for key in ("episode_id", "family_id", "split")):
            raise ValueError("Main family/split differs from frozen plan")
        if raw["family_definition"] != plan["family_definition"]:
            raise ValueError("Main lineage definition differs from frozen plan")
        with np.load(npz_path, allow_pickle=False) as archive:
            states = archive["state"].copy(); commands = archive["action_applied"].copy()
            dt = archive["actual_dt"].copy(); valid = archive["transition_valid"].copy()
            diagnostics = h.restore(raw, archive["diagnostic_nonfinite_values"])
            complete = np.diff(archive["substep_offsets"])==metadata["contract"]["frame_skip"]
            complete &= np.abs(dt-metadata["contract"]["action_dt_s"])<=metadata["contract"]["dt_tolerance_s"]
            composite_mask = (archive["step_valid"] & archive["geometry_valid"] &
                archive["observation_valid"][:-1] & archive["observation_valid"][1:] & complete)
            if not np.array_equal(valid, composite_mask):
                raise ValueError("Main stored learning mask differs from complete stored validity components")
        if np.any(np.linalg.norm(commands, axis=1) > manifest["collection_settings"]["commands"]["command_support_norm_max_m_s"]+1e-15):
            raise ValueError("Main actual action exceeds declared collection support")
        policy = h.read(folder/"policy"/(episode+".json"))
        executed = [p for p in policy["records"] if p["executed"]]
        if len(executed) != len(commands):
            raise ValueError("Main policy/execution lengths differ")
        for index, p in enumerate(executed):
            if p["applied_action"] != commands[index].tolist():
                raise ValueError("Main policy and stored actual action differ")
            if plan["mode"]=="geometry_perturbed" and p["diagnostics"]["predeclared_noise_m_s"] != plan["perturbation_vectors_m_s"][index]:
                raise ValueError("Main predeclared noise changed or redrawn")
        derived = h.measure(states, commands, dt, valid, [p["diagnostics"] for p in executed],
                            diagnostics, metadata["contract"]["resolved_config"])
        primary = expected_coverage[episode]
        if (derived["recorded_intervals"] != entry["review"]["transitions"] or
                derived["valid_intervals"] != entry["review"]["learning_transitions"] or
                derived["regime_counts_valid"] != primary["regime_counts_valid"] or
                derived["K5"]["total"] != primary["windows"]["5"]["total"] or
                derived["K5"]["by_first_regime"] != primary["windows"]["5"]["by_first_regime"]):
            raise ValueError("Independently derived main contact/windows disagree with collection audit")
        derived.update(episode_id=episode, family_id=entry["family_id"], split=entry["split"],
                       mode=plan["mode"], terminal_reason=raw["terminal_reason"])
        behavior.append(derived)
        del diagnostics, raw, states, commands, dt, valid
        gc.collect()
    if len(behavior) != metadata["episode_count"] or len(behavior) != len(plans):
        raise ValueError("Not all frozen main episodes represented")
    splits = {}
    for split in ("train", "validation"):
        part = [r for r in behavior if r["split"]==split]
        families = {regime: len({r["family_id"] for r in part if r["K5"]["by_first_regime"][regime]>0}) for regime in h.REGIMES}
        if families != coverage["splits"][split]["windows"]["5"]["contributing_families"]:
            raise ValueError("Main distinct-family coverage differs")
        splits[split] = {"episodes": len(part), "families": len({r["family_id"] for r in part}),
            "valid_intervals": sum(r["valid_intervals"] for r in part),
            "holding_intervals": sum(r["holding_intervals"] for r in part),
            "stationary_intervals": sum(r["valid_stationary_intervals"] for r in part),
            "stationary_outside_goal_pose": sum(r["valid_stationary_outside_goal_pose"] for r in part),
            "zero_outside_goal_pose": sum(r["valid_zero_outside_goal_pose"] for r in part),
            "K5_distinct_families_by_first_regime": families,
            "K5_windows": sum(r["K5"]["total"] for r in part),
            "K5_all_five_sustained_contact": sum(r["K5"]["all_five_sustained_contact"] for r in part),
            "regimes_valid": {g: sum(r["regime_counts_valid"][g] for r in part) for g in h.REGIMES},
            "noise_eligible_intervals": sum(r["noise"]["eligible_intervals"] for r in part),
            "nonzero_noise_delta_intervals": sum(r["noise"]["nonzero_actual_delta_intervals"] for r in part)}
    if summary["train_fit_count"] != splits["train"]["valid_intervals"]:
        raise ValueError("Train-only fit count differs from raw valid train rows")
    totals = {"recorded": sum(r["recorded_intervals"] for r in behavior),
              "valid": sum(r["valid_intervals"] for r in behavior),
              "excluded": sum(r["invalid_or_partial_intervals"] for r in behavior)}
    if (totals["recorded"] != summary["recorded_transitions"] or totals["valid"] != summary["valid_transitions"]
            or totals["excluded"] != summary["excluded_transitions"]):
        raise ValueError("Main raw/summary totals differ")
    return {"run": run, "stage_verdict": summary["stage_verdict"], "source_verification": h.source_check(folder, manifest),
        "all_episode_hashes_and_independent_raw_contact_K5_counts_match": True,
        "summary_checks": summary["checks"], "totals": totals, "splits": splits,
        "terminal_counts": dict(Counter(r["terminal_reason"] for r in behavior)),
        "normalizer_hash": summary["normalizer_hash"], "train_fit_count": summary["train_fit_count"],
        "normalizer_preparation_checks": preparation["checks"], "resources": summary["resources"],
        "marginal_input_support": support["splits"], "episodes": behavior,
        "alignment_outcome": summary["alignment_outcome"], "dynamics_training": summary["dynamics_training"],
        "scope": "independent raw-mask/contact/K5/lineage accounting; minimum support only; overlapping rows/windows are not independent trials"}


def analyze(main_run=None):
    sys.path.insert(0, str(ROOT/"src"))
    history, preserved = historical_snapshot()
    data = {"schema": "p3-transition-final-derived-v1", "created_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "historical_snapshot": preserved, "geometry_runs": {}, "gate_runs": {},
        "main_data": {"status": "NOT_ANALYZED", "reason": "No completed main run supplied"},
        "interpretation": ["Clock03 and cap04 are sequential development changes, not a factorial ablation.",
            "The selected G cap is0.03m/s; the prior G cap was0.04m/s. Physical cap0.06 and collection post-noise/open-loop cap0.04 stay unchanged.",
            "Successful four-case readiness and a six-case dt pass establish these bounded gates, not field value or generalization.",
            "Collection does not evaluate alignment success. Learning and closed-loop model comparison remain separate tasks.",
            "Do not pool independently observed frames/windows as independent statistical trials."]}
    manifests = {}
    for run in G_RUNS:
        folder = ROOT/"runs"/run; manifest = h.read(folder/"manifest.json"); manifests[run] = manifest
        summary = h.read(folder/"summary.json")
        cases = [h.g_case(run, case, manifest["physical_config"]) for case in h.CASE_IDS]
        data["geometry_runs"][run] = {"pass": summary["pass"], "checks": summary["checks"],
            "source_verification": h.source_check(folder, manifest), "cases": cases,
            "ready_cases": sum(c["ready"] for c in cases),
            "valid_intervals": sum(c["valid_intervals"] for c in cases),
            "holding_intervals": sum(c["holding_intervals"] for c in cases),
            "zero_outside_goal_pose": sum(c["valid_zero_outside_goal_pose"] for c in cases),
            "selected_base_parameters": manifest["resolved_controller_parameters"],
            "transition_parameters": manifest["transition_parameters"]}
    for run in GATE_RUNS:
        folder = ROOT/"runs"/run; manifest = h.read(folder/"manifest.json"); manifests[run] = manifest
        summary = h.read(folder/"summary.json"); extension = h.read(folder/"physics_extension/summary.json")
        result = {"stage_verdict": summary["stage_verdict"], "pass": extension["pass"],
            "source_verification": h.source_check(folder, manifest), "cases": []}
        for row in extension["cases"]:
            derived = dict(row)
            derived["behavior"] = {label: h.gate_case(run, row, label, manifest["contract"]["resolved_config"]) for label in ("base", "half")}
            derived["phase_clock_audit"] = phase_audit(run, row["mode"])
            result["cases"].append(derived)
        data["gate_runs"][run] = result
    base_manifest = h.read(ROOT/"runs/p3-01-20261008-01/manifest.json")
    old_gate_manifest = h.read(ROOT/"runs/p3-02-gate-20261008-02/manifest.json")
    selected = manifests[G_RUNS[2]]["resolved_controller_parameters"]
    prior = base_manifest["resolved_controller_parameters"]
    changes = {key: {"before": prior[key], "after": selected[key]} for key in prior if prior[key]!=selected[key]}
    if changes != {"speed_max_m_s": {"before": .04, "after": .03}}:
        raise ValueError("Unexpected selected base-G changes")
    data["comparison_checks"] = {"G_base_parameter_changes_vs_original": changes,
        "other_fifteen_G_parameters_identical": True,
        "G_physical_config_and_task_identical": all(manifests[r]["physical_config"]==base_manifest["physical_config"] for r in G_RUNS),
        "G_four_cases_identical": all(manifests[r]["cases"]==base_manifest["cases"] for r in G_RUNS),
        "candidate04_vs_formal_parameters_identical": manifests[G_RUNS[1]]["resolved_controller_parameters"]==selected,
        "candidate04_vs_formal_transition_identical": manifests[G_RUNS[1]]["transition_parameters"]==manifests[G_RUNS[2]]["transition_parameters"],
        "gate_ninety_plans_and_families_identical": all(manifests[r]["predeclared_episodes"]==old_gate_manifest["predeclared_episodes"]
            and manifests[r]["family_assignments"]==old_gate_manifest["family_assignments"] for r in GATE_RUNS),
        "gate_raw_physical_contract_identical": all(manifests[r]["contract"]==old_gate_manifest["contract"] for r in GATE_RUNS),
        "gate_budgets_identical": all(row["budgets"]==old["budgets"] for r in GATE_RUNS for row, old in zip(
            data["gate_runs"][r]["cases"], history["gate_runs"]["p3-02-gate-20261008-02"]["cases"]))}
    if main_run:
        data["main_data"] = main_analysis(main_run)
    data["analysis_code_sha256"] = h.sha(Path(__file__))
    data["reused_analysis_helper_sha256"] = h.sha(ROOT/"tools/p3_transition_analysis.py")
    data["input_sha256"] = dict(h.INPUTS)
    data["all_used_inputs_unchanged_after_analysis"] = all(h.sha(ROOT/p)==expected for p, expected in h.INPUTS.items())
    if not data["all_used_inputs_unchanged_after_analysis"]:
        raise ValueError("Inputs changed during final analysis")
    h.write(REPORT, data)
    print(json.dumps({"report": str(REPORT), "inputs_hashed": len(h.INPUTS),
        "G_ready": {r: v["ready_cases"] for r, v in data["geometry_runs"].items()},
        "gate_verdicts": {r: v["stage_verdict"] for r, v in data["gate_runs"].items()},
        "main_status": data["main_data"].get("stage_verdict", data["main_data"].get("status"))}))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = json.loads(REPORT.read_text())
    history = json.loads((ROOT/"reports/p3_transition_analysis.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12,
        "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
        "svg.fonttype": "none", "figure.dpi": 180})
    folder = ROOT/"reports/figures"; files = []

    def save(fig, name):
        for extension in ("png", "svg"):
            path = folder/(name+"."+extension); fig.savefig(path, dpi=180, facecolor="white"); files.append(path)
        plt.close(fig)

    colors = ("#466480", "#bd8b43", "#277b69")
    versions = [history["geometry_runs"]["p3-01-20261008-01"], data["geometry_runs"][G_RUNS[0]], data["geometry_runs"][G_RUNS[2]]]
    labels = ("Legacy G (0.04 m/s)", "Clock03 G (0.04 m/s)", "Formal G (0.03 m/s)")
    x = np.arange(4); fig, axes = plt.subplots(2, 1, figsize=(9, 7.3))
    for i, (version, label) in enumerate(zip(versions, labels)):
        rows = version["cases"]
        durations = [r["elapsed_s"] for r in rows]
        axes[0].bar(x+(i-1)*.23, durations, .23, label=label, color=colors[i])
        for j, value in enumerate(durations):
            axes[0].text(x[j]+(i-1)*.23, value+.3, f"{value:.1f}", ha="center", fontsize=10)
        axes[1].bar(x+(i-1)*.23, [r["valid_zero_outside_goal_pose"] for r in rows], .23, color=colors[i])
    axes[0].axhline(30, color="#555", linestyle="--"); axes[0].set_ylim(0, 33)
    axes[0].legend(loc="upper left", fontsize=10); axes[0].set_ylabel("Ready completion time (s)")
    axes[0].set_title("All four cases reach ready under the same 30 s task budget")
    axes[1].set_ylabel("Zero commands outside goal pose"); axes[1].set_title("Entry ramps add pauses; lower speed does not establish higher productivity")
    for ax in axes:
        ax.set_xticks(x, ["Translation", "+10 deg yaw", "-10 deg yaw", "Forced recontact"])
        ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    fig.text(.06, .015, "Same cases and task criteria; changed control policy. Clock03 and cap04 are sequential tuning.", fontsize=11)
    fig.tight_layout(rect=(0, .045, 1, 1)); save(fig, "p3_final_transition_task_cases")

    old = history["gate_runs"]["p3-02-gate-20261008-02"]["cases"]
    variants = (old, data["gate_runs"][GATE_RUNS[0]]["cases"], data["gate_runs"][GATE_RUNS[1]]["cases"])
    x = np.arange(6); fig, axes = plt.subplots(2, 1, figsize=(9, 7.0))
    for i, rows in enumerate(variants):
        for ax, key, multiplier in ((axes[0], "max_position_difference_m", 1000),
                                   (axes[1], "max_yaw_difference_rad", 180/math.pi)):
            ax.bar(x+(i-1)*.23, [r[key]*multiplier for r in rows], .23,
                   label=("Legacy", "Clock03", "Selected04")[i], color=colors[i])
    for ax, budget, unit in ((axes[0], .5, "mm"), (axes[1], .3, "deg")):
        ax.axhline(budget, color="#bd554d", linestyle="--", label=f"Budget: {budget} {unit}")
        ax.legend(loc="upper left", ncol=4, fontsize=10)
        ax.set_xticks(x, ["No contact", "Central", "Eccentric", "Contact\ntransition", "G feedback", "G+noise"])
        ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    axes[0].set_ylabel("Max dt position difference (mm)"); axes[1].set_ylabel("Max dt yaw difference (deg)")
    axes[0].set_title("Selected04 passes the same six full-length dt representatives")
    fig.text(.075, .015, "Same physical dt, frozen inputs and budgets; base G cap 0.04->0.03 m/s is a policy change.", fontsize=11)
    fig.tight_layout(rect=(0, .045, 1, 1)); save(fig, "p3_final_transition_dt_gate")

    rows = [next(r for r in variant if r["mode"]==mode)["behavior"]["base"] for mode in
            ("geometry_feedback", "geometry_perturbed") for variant in (old, variants[2])]
    x = np.arange(4); fig, axes = plt.subplots(2, 1, figsize=(9, 7.2)); bottom = np.zeros(4)
    for regime, color in zip(h.REGIMES, ("#bac6d2", "#277b69", "#d3a05b")):
        values = np.array([r["regime_counts_valid"][regime] for r in rows])
        axes[0].bar(x, values, bottom=bottom, color=color, label=regime.replace("_", " "))
        for i, value in enumerate(values):
            if value:
                axes[0].text(i, bottom[i]+value/2, str(value), ha="center", va="center", fontsize=11)
        bottom += values
    axes[0].set_xticks(x, ["Legacy G", "Selected G", "Legacy G+noise", "Selected G+noise"])
    axes[0].set_ylim(0, 147); axes[0].legend(loc="upper center", ncol=3)
    axes[0].set_ylabel("Valid 0.1 s intervals"); axes[0].set_title("Actual force regimes: same 128-interval base-dt representatives")
    main = data["main_data"]
    if "splits" in main:
        families = main["splits"]["validation"]["K5_distinct_families_by_first_regime"]
        counts = [families[g] for g in h.REGIMES]
        axes[1].bar(np.arange(3), counts, color=colors[2])
        for i, value in enumerate(counts):
            axes[1].text(i, value+.25, str(value), ha="center", fontsize=11)
        axes[1].axhline(5, color="#bd554d", linestyle="--", label="Required minimum: 5 families")
        axes[1].set_xticks(np.arange(3), ["Noncontact", "Sustained contact", "Transition"])
        axes[1].set_ylim(0, max(counts+[5])+6); axes[1].legend(loc="upper right")
        axes[1].set_ylabel("Distinct validation families")
        axes[1].set_title(f"Main data: {main['totals']['valid']} valid rows; K5 uses first-interval regime")
        caption = "Main minimum coverage is not statistical adequacy, task success or learned-model performance."
    else:
        for i, (field, label) in enumerate((("eligible_intervals", "Noise eligible"),
                                          ("nonzero_actual_delta_intervals", "Nonzero applied noise"))):
            values = [r["noise"][field] for r in rows]
            axes[1].bar(x+(i-.5)*.3, values, .3, label=label)
            for j, value in enumerate(values):
                axes[1].text(x[j]+(i-.5)*.3, value+.4, str(value), ha="center", fontsize=10)
        axes[1].set_xticks(x, ["Legacy G", "Selected G", "Legacy G+noise", "Selected G+noise"])
        axes[1].set_ylabel("Descriptive interval count"); axes[1].legend(loc="upper left")
        axes[1].set_title("Representative behavior only; completed main dataset not yet included")
        caption = "A dt representative does not establish main-dataset coverage. Rows/windows are not independent trials."
    for ax in axes:
        ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    fig.text(.045, .015, caption, fontsize=11)
    fig.tight_layout(rect=(0, .045, 1, 1)); save(fig, "p3_final_transition_data_support")
    audit = {"analysis_report_sha256": h.sha(REPORT), "analysis_code_sha256": h.sha(Path(__file__)),
        "reused_helper_sha256": h.sha(ROOT/"tools/p3_transition_analysis.py"),
        "figures": {str(p.relative_to(ROOT)): h.sha(p) for p in files},
        "minimum_label_font_pt": 10., "figure_width_inches": 9., "anticipated_pdf_width_pt": 480.,
        "minimum_label_pt_at_pdf_width": 10*480/(9*72),
        "scope": "new files only; historical01/02 figures unchanged; PDF placement requires visual QA"}
    h.write(ROOT/"reports/p3_transition_final_figures_audit.json", audit)
    print(json.dumps(audit))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--plot", action="store_true"); parser.add_argument("--main-run")
    args = parser.parse_args()
    if not args.analyze and not args.plot:
        parser.error("Choose --analyze and/or --plot")
    if args.analyze:
        analyze(args.main_run)
    if args.plot:
        plot()
