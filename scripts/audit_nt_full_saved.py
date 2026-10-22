#!/usr/bin/env python3
"""Independent saved-only audit of a complete or resource-stopped N-T run.

All present rows and file hashes are checked. Prespecified complete-probe
groups and their raw repeats are decoded when available, with omissions named.
No environment is created, predictor loaded/called, training performed, or
scientific source written. The full executor already performs every raw and
complete-probe roundtrip; this audit describes its separate sampling scope.
"""
from __future__ import annotations

import time
START = time.monotonic()
import argparse
from collections import defaultdict
from dataclasses import fields
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from push_align.contact_geometry import check_contact_geometry
from push_align.environment import StepResult
from push_align.final_driver_v2 import exact_comparison, read_full_trace, typed_digest
from push_align import final_recording_columns as codec
from push_align.followup_probe import FollowupProbe, _physical_reasons
from push_align.planning import PredictionBatch, score
from types import SimpleNamespace

MODELS = ("R", "S1-11", "S1-23", "S1-47", "SK-11", "SK-23", "SK-47")
SAMPLE_IDS = (0, 9, 127)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def ah(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.dtype.str.encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def passed(value):
    return isinstance(value, dict) and value.get("status") == "PASS_EXACT" and value["left_sha256"] == value["right_sha256"]


def same(a, b, message):
    result = exact_comparison(a, b)
    need(result["status"] == "PASS_EXACT", message)
    return result


def close(a, b):
    need(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-8), f"Numeric mismatch {a!r} != {b!r}")


def auth_receipt(receipt, run):
    count = 0
    for filename, expected in receipt["files_sha256"].items():
        path = Path(filename).resolve()
        need(path.is_relative_to(run) and sha(path) == expected, "Run receipt file mismatch: " + str(path))
        if "bytes" in receipt and isinstance(receipt["bytes"], dict):
            need(path.stat().st_size == receipt["bytes"][filename], "Receipt byte size mismatch")
        count += 1
    return count


def reconstruct(tree):
    need(tree["schema"] == "followup-complete-probe-tree-v1" and tree["class"] == "FollowupProbe", "Full probe schema")
    values = dict(tree["fields"])
    need(set(values) == set(FollowupProbe.__dataclass_fields__), "Complete field set changed")
    values["results"] = tuple(StepResult(**step) for step in values["results"])
    return FollowupProbe(**values)


def audit(run, original, analysis_path, prior_postwork_seconds):
    entry, summary, seal = (load(run / p) for p in ("entry.json", "summary.json", "prospective_forecast_seal.json"))
    wall = load(run / "process_wall_receipt.json")
    config = load(ROOT / "configs/followup_nt_full.v1.json")
    expected = list(product(range(128), ("A", "B"), range(3)))
    complete = summary["status"] == "COMPLETE_NT_FULLBANK_EXACT_768"
    need(complete or summary["status"] == "STOP_RESOURCE", "Expected complete or explicitly resource-stopped evidence")
    need(not summary["scientific_row_reuse"] and summary["conditional_extension"] == "NOT_RUN", "Reused/extended rows")
    need(summary["training_updates"] == 0 and summary["original_source_models_code_config_unchanged"], "Frozen evidence not verified")
    need(wall["exit_code"] == (0 if complete else 2) and wall["process_wall_seconds"] >= summary["wall_seconds"], "External wall receipt invalid")
    need(config["cumulative_prior_wall_seconds"] == summary["cumulative_prior_wall_seconds"] == 662.5052578353789, "Prior clock drift")
    need(sha(ROOT / "configs/followup_nt_full.v1.json") == entry["config_sha256"], "Executed config mismatch")
    for relative, digest in {**entry["code_sha256"], **entry["model_and_normalizer_sha256"], **entry["reviewed_resource_sha256"]}.items():
        need(sha(ROOT / relative) == digest, "Frozen publication code/model/resource changed: " + relative)
    need(len(entry["model_and_normalizer_sha256"]) == 22 and entry["model_and_normalizer_sha256"] == config["expected_model_normalizer_sha256"], "Model file map differs")
    for relative, digest in {**entry["original_sha256"], **entry["authenticated_prior_receipts"]}.items():
        need(sha(original / relative) == digest, "Original/prior receipt changed: " + relative)
    frozen = load(original / "runs/p5-final-20261017-02/manifest.json")
    need(sha(original / "runs/p5-final-20261017-02/manifest.json") == config["original_manifest_sha256"], "Frozen manifest changed")
    for relative, digest in frozen["runtime_source_sha256"].items():
        need(sha(original / relative) == digest, "Frozen original runtime changed: " + relative)
    source = original / "runs/followup-near-goal-20261022-01"
    for relative, digest in entry["source_run_sha256"].items():
        need(sha(source / relative) == digest, "Preserved D1/D2 source changed: " + relative)
    need(seal["before_any_new_actual_future"] and all(v == 0 for v in seal["future_engine_calls_during_prediction"].values()) and all(v == 0 for v in summary["future_engine_calls_during_prediction"].values()), "Prediction futures guard")
    for relative, digest in seal["forecasts"].items():
        need(sha(run / relative) == digest, "Prospective forecast changed")
    seal_sha = sha(run / "prospective_forecast_seal.json")
    with np.load(run / "bank.npz", allow_pickle=False) as z:
        bank = z["actions"].copy()
        need(bank.shape == (128, 5, 2) and bank.dtype == np.float64 and np.array_equal(z["original_ids"], np.arange(128)), "Bank format")
    need(ah(bank) == config["expected_candidate_hash"] == entry["bank_hash"], "Whole candidate hash")
    data, _, _ = read_full_trace(entry["full_input_receipt"])
    need(typed_digest(data) == entry["full_input_typed_sha256"] == config["expected_input_typed_sha256"], "Full restored input typed identity")
    rows = summary["rows"]
    keys = [(row["candidate_id"], row["panel"], row["repeat"]) for row in rows]
    need(keys == expected[:len(keys)] and len(keys) == summary["completed_trajectories"], "Cartesian execution prefix/order/identity mismatch")
    missing = [{"candidate_id":i,"panel":p,"repeat":r} for i,p,r in expected[len(keys):]]
    need(summary["missing_row_keys"] == missing, "Missing row inventory mismatch")
    need((len(keys) == 768) == complete, "Complete status inconsistent with row count")
    groups, file_checks, original_checks, repeated, persisted = defaultdict(list), 0, 0, 0, 0
    for row in rows:
        i, panel, repeat = row["candidate_id"], row["panel"], row["repeat"]
        trial = run / "actual/N-T" / f"candidate-{i:03d}" / panel / f"repeat-{repeat}"
        saved = load(trial / "summary.json")
        saved["timings_ns"] = load(trial / "timing_final.json")["timings_ns"]
        need(saved == row, "Top/trial summary mismatch")
        tr_entry = load(trial / "entry.json")
        need(tr_entry["prospective_seal_sha256"] == seal_sha and tr_entry["full_input_typed_sha256"] == entry["full_input_typed_sha256"], "Trial prospective/input binding")
        need(row["candidate_hash"] == tr_entry["candidate_hash"] == ah(bank[i]), "Trial candidate hash")
        need(passed(row["raw_roundtrip"]) and row["initial_snapshot_exact"], "Raw roundtrip or restore failed")
        raw = load(trial / "raw_receipt.json")
        need(raw["interval_count"] == row["intervals"], "Raw interval count")
        file_checks += auth_receipt(raw, run)
        diagnostic = row["diagnostic_receipt"]
        need(sha(trial / "diagnostic.npz") == diagnostic["sha256"] and passed(diagnostic["roundtrip"]), "Diagnostic authentication")
        file_checks += 1
        if repeat == 0:
            need(passed(row["full_probe_roundtrip"]), "First complete probe roundtrip")
            file_checks += auth_receipt(load(trial / "complete_probe_receipt.json"), run)
            persisted += 1
        else:
            need(passed(row["full_probe_repeat_exact"]) and row["full_probe_repeat_exact"]["left_sha256"] == row["whole_probe_typed_sha256"], "Whole repeat equality")
            repeated += 1
        if i < 16:
            need(passed(row["source_D2_raw_exact"]), "Original D2 reference equality")
            need(row["source_D2_raw_exact"]["right_sha256"] == row["raw_roundtrip"]["left_sha256"], "Original/fresh raw typed-digest linkage")
            original_checks += 1
        else:
            need(row["source_D2_raw_exact"] is None, "Unexpected original reference")
        if panel == "B":
            need(row["cost"] is None and row["cost_components"] == {}, "B must have no H5 cost")
        elif row["cost_finite"]:
            close(sum(row["cost_components"].values()), row["cost"])
        need(row["ready_prefix"] == (row["ready"] and row["first_ready_interval"] <= 5), "Ready prefix convention")
        groups[(i, panel)].append(row)
    need((persisted, repeated, original_checks) == (sum(r[2]==0 for r in keys), sum(r[2]!=0 for r in keys), sum(r[0]<16 for r in keys)), "Verification cardinalities")
    for group in groups.values():
        need(len({r["whole_probe_typed_sha256"] for r in group}) == 1, "Unstable complete probe group")
    teacher_keys = []
    teacher_path = run / "teacher/source-fresh-A-first-repeat.npz"
    if complete:
        need(teacher_path.exists(), "Complete run missing fresh teacher")
    else:
        need(not teacher_path.exists() and summary["teacher_metadata"] is None, "Stopped pre-teacher run must not claim fresh teacher completion")
    teacher = None
    if teacher_path.exists():
        teacher_keys, teacher = check_teacher(run, groups)
    sample_verdicts = []
    skipped_sample_ids = []
    contact_limit = frozen["profile"]["physics_suite"]["actual_contact_penetration_max_m"]
    for i, panel in product(SAMPLE_IDS, ("A", "B")):
        if len(groups.get((i,panel),[])) != 3:
            skipped_sample_ids.append({"candidate_id":i,"panel":panel,"reason":"NOT_EXECUTED_OR_GROUP_INCOMPLETE"})
            continue
        trial0 = run / "actual/N-T" / f"candidate-{i:03d}" / panel / "repeat-0"
        tree, _, _ = read_full_trace(load(trial0 / "complete_probe_receipt.json"))
        probe = reconstruct(tree)
        need(typed_digest(probe) == groups[(i, panel)][0]["whole_probe_typed_sha256"], "Decoded sample whole-probe digest")
        for repeat in range(3):
            trial = trial0.parent / f"repeat-{repeat}"
            receipt = load(trial / "raw_receipt.json")
            snapshot, steps = codec.read_raw(trial / "raw", expected_sha256=receipt["files_sha256"])
            verdict = same({"snapshot": snapshot, "steps": steps}, {"snapshot": data["snapshot"], "steps": list(probe.results)}, "Decoded sample raw/full-probe differs")
            need(verdict["left_sha256"] == groups[(i, panel)][repeat]["raw_roundtrip"]["left_sha256"], "Decoded sample raw digest linkage")
        if panel == "A":
            prediction = PredictionBatch(probe.states[None], np.diff(probe.states, axis=0)[None], np.array([probe.physical_valid and probe.horizon_completed]), ("",), np.array([-1], np.int64))
            costs = score(prediction, bank[i:i+1], data["task_spec"], data["planner_context"].previous_applied_action, data["planner_config"])
            close(float(costs.total[0]), probe.cost)
            if teacher is not None:
                check_sample_teacher(teacher, teacher_keys, i, probe, data, bank, contact_limit)
        sample_verdicts.append({"candidate_id": i, "panel": panel, "decoded_raw_repeats": 3, "decoded_complete_probe_typed_sha256": typed_digest(probe)})
    return finish_rules_and_budget(run, analysis_path, prior_postwork_seconds, entry, summary, wall, groups, complete,
        file_checks, persisted, repeated, original_checks, teacher_keys, sample_verdicts, skipped_sample_ids)


def check_teacher(run, groups):
    teacher_path = run / "teacher/source-fresh-A-first-repeat.npz"
    teacher_meta = load(run / "teacher/summary.json")
    need(sha(teacher_path) == teacher_meta["source"]["sha256"], "Fresh teacher source hash")
    with np.load(teacher_path, allow_pickle=False) as z:
        teacher = {k: z[k].copy() for k in z.files}
    teacher_keys = list(map(tuple, teacher["candidate_interval"]))
    expected_teacher_keys = [(i, step) for i in range(128) for step in range(groups[(i, "A")][0]["intervals"])]
    need(teacher_keys == expected_teacher_keys and len(set(teacher_keys)) == len(teacher_keys) <= 640, "Teacher prefix/unique-row scope")
    need(len(teacher_keys) == teacher_meta["rows"], "Teacher row count")
    bool_keys = ("actual_transition_valid", "actual_observation_valid", "actual_contact_geometry_valid", "actual_dt_match", "actual_endpoint_finite", "actual_physical_valid")
    need(all(teacher[k].dtype == np.bool_ for k in bool_keys), "Teacher mask dtype")
    need(set(teacher_meta["models"]) == set(MODELS), "Seven teacher models")
    for name in MODELS:
        meta = teacher_meta["models"][name]
        need(meta["rows"] == len(teacher_keys) and sha(run / "teacher" / f"one-step-{name}.npz") == meta["receipt"]["sha256"], "Teacher model receipt")
    return teacher_keys, teacher


def check_sample_teacher(teacher, teacher_keys, i, probe, data, bank, contact_limit):
    current = data["observation"].state
    for step_index, step in enumerate(probe.results):
        at = teacher_keys.index((i, step_index))
        for key, value in (("start", current), ("actual", step.observation.state), ("applied", step.applied_action)):
            need(np.array_equal(teacher[key][at], value, equal_nan=True), "Teacher sample transition binding")
        geometry = check_contact_geometry(step.diagnostics["samples"], contact_limit)
        reasons = _physical_reasons(step, bank[i, step_index], SimpleNamespace(action_dt=data["physical_config"]["physics"]["action_dt_s"]), data["physical_config"]["validity"]["dt_match_tolerance_s"], geometry)
        need(teacher["actual_physical_valid"][at] == (not reasons) and teacher["actual_contact_geometry_valid"][at] == geometry["pass"], "Teacher sample effective physical mask")
        current = step.observation.state


def finish_rules_and_budget(run, analysis_path, prior_postwork_seconds, entry, summary, wall, groups, complete,
        file_checks, persisted, repeated, original_checks, teacher_keys, sample_verdicts, skipped_sample_ids):
    # Independent decision-rule arithmetic; does not import the summarizer.
    ranking = {i: (min(r["cost"] for r in g), max(r["cost"] for r in g), bool(g[0]["ready_eligible"])) for (i, p), g in groups.items() if p == "A" and len(g)==3 and all(r["physical_valid"] and r["horizon_completed"] and r["intervals"] == 5 and r["cost_finite"] for r in g)}
    analysis = load(analysis_path)
    report_cost = analysis["cost_assessment_A_only"]
    eps = None if not ranking else 1e-6 + 1e-5 * max(abs(v) for pair in ranking.values() for v in pair[:2])
    possible = [] if not ranking else [i for i,(lo,hi,_) in ranking.items() if lo <= min(p[1] for p in ranking.values()) + eps]
    need(report_cost["possible_cost_optimum_ids"] == possible, "Independent possible-cost-optimum set")
    if eps is not None:
        close(eps, report_cost["epsilon"])
    ready = [i for i,p in ranking.items() if p[2]]
    mismatch = complete and bool(ready) and not set(possible).intersection(ready) and any(not p[2] and p[1] + eps < min(ranking[j][0] for j in ready) for p in ranking.values())
    need((report_cost["C"]["status"] == "SUPPORTED_LOCAL_COST_READINESS_MISMATCH") == mismatch, "Independent C rule")
    for name in MODELS:
        with np.load(run / "forecasts" / f"prediction-{name}.npz", allow_pickle=False) as z:
            valid, cost = z["valid"], z["total_cost"]
            eligible = np.flatnonzero(valid & np.isfinite(cost))
            chosen = None if not len(eligible) else int(eligible[np.argmin(cost[eligible])])
            false_invalid = [i for i in range(128) if len(groups.get((i,"A"),[]))==3 and groups[(i,"A")][0]["physical_valid"] and groups[(i,"A")][0]["horizon_completed"] and groups[(i,"A")][0]["intervals"] == 5 and not valid[i]]
        reported = analysis["model_hypotheses"][name]
        need(chosen == reported["selected_original_candidate_id"] and false_invalid == reported["M1_false_invalid_actual_A_valid_ids"], "Independent model selection/M1")
        if chosen is None:
            need(reported["M2"]["status"] == "NO_VALID_PREDICTION", "No valid prediction must stay separate")
        elif chosen in ranking:
            strict = ranking[chosen][0] > min(p[1] for p in ranking.values()) + eps
            need(reported["M2"]["strict_selected_L_gt_best_U_plus_epsilon"] == strict, "Independent M2 strict ranking rule")
    # Inspect every recorded whole-workflow gate and independently recompute units.
    decisions = [json.loads(line) for line in (run / "resource_decisions.jsonl").read_text().splitlines()]
    previous = None
    stop_decisions=[]
    for d in decisions:
        if d["decision"] == "STOP_RESOURCE":
            need(not complete and d["failures"], "Unacknowledged resource failure")
            stop_decisions.append(d["stage"])
        else:
            need(d["decision"] == "PASS_RESOURCE" and not d["failures"], "Malformed resource gate")
        rates, work, lim = d["observed_rates"], d["remaining"], d["limits"]
        need(lim["report_reserve_seconds"] == 600 and lim["wall_cap_seconds"] == 7200 and lim["safety_factor"] == 1.5, "Live resource limits relaxed")
        if previous:
            need(all(rates[k] >= previous[k] for k in rates), "Resource rate decreased")
        previous = rates
        close(d["future_seconds"]["interval_baseline"], 1.5 * work["intervals"] * rates["seconds_per_interval"])
        close(d["future_seconds"]["source_D2_comparisons"], 1.5 * work["source_comparisons"] * rates["source_comparison_seconds"])
        close(d["future_seconds"]["first_repeat_complete_probe"], 1.5 * work["complete_probe_groups"] * rates["complete_probe_seconds"])
        close(d["projected_cumulative_seconds"], lim["prior_seconds"] + d["process_seconds"] + sum(d["future_seconds"].values()))
        if d["decision"] == "PASS_RESOURCE":
            need(d["projected_cumulative_seconds"] <= 7200 and d["projected_cumulative_bytes"] <= 5 * 1024**3 and d["free_bytes"] >= 2 * 1024**3 + sum(d["future_bytes"].values()), "Passing resource projection exceeds cap")
        else:
            need((d["projected_cumulative_seconds"] > 7200) == ("cumulative_wall_cap" in d["failures"]), "Resource stop arithmetic")
    measured_audit_seconds = time.monotonic() - START
    cumulative_through_audit = summary["cumulative_prior_wall_seconds"] + wall["process_wall_seconds"] + prior_postwork_seconds + measured_audit_seconds
    need(cumulative_through_audit <= 7200, "Actual cumulative wall through this audit exceeds cap")
    cursor = summary["active_unsealed_result_key"]
    cursor_sealed = cursor is not None and any((r["candidate_id"],r["panel"],r["repeat"]) == tuple(cursor) for r in summary["rows"])
    return {
        "schema":"followup-nt-full-saved-independent-audit-v1",
        "status":"PASS_SAVED_EVIDENCE_AND_INDEPENDENT_RULE_CHECKS",
        "source_run_status":summary["status"], "full_execution_complete":complete,
        "scope":{"planned_rows":768,"all_present_rows":len(summary["rows"]),
            "missing_rows":len(summary["missing_row_keys"]),
            "all_present_raw_pair_hashes":len(summary["rows"]),
            "complete_probe_group_hashes":persisted,"whole_probe_repeat_receipts":repeated,
            "source_D2_receipts":original_checks,"authenticated_run_files":file_checks,
            "independently_decoded_raw_repeats":sum(s["decoded_raw_repeats"] for s in sample_verdicts),
            "independently_decoded_complete_groups":len(sample_verdicts),
            "prespecified_sample_candidate_ids":list(SAMPLE_IDS),
            "prediction_model_calls":0,"physics_calls":0},
        "executor_cursor":{"key":cursor,"already_present_in_sealed_rows":cursor_sealed,
            "interpretation":"The cursor is metadata; its present sealed row is retained and is not a missing-result verdict."},
        "independent_samples":sample_verdicts,"skipped_prespecified_samples":skipped_sample_ids,
        "independent_cost_rules":{"epsilon":eps,"possible_cost_optimum_ids":possible,"C_supported":mismatch,
            "scope":"Full finite bank" if complete else "Observed stable subset only; C remains unresolved because bank incomplete"},
        "resource_decisions_checked":len(decisions),"expected_resource_stop_stages":stop_decisions,
        "teacher_unique_rows":len(teacher_keys),"fresh_teacher_status":"VERIFIED" if teacher_keys else "NOT_RUN_BECAUSE_768_ACTUAL_ROWS_INCOMPLETE",
        "analysis_sha256":sha(analysis_path),"entry_sha256":sha(run/"entry.json"),
        "summary_sha256":sha(run/"summary.json"),"process_wall_receipt_sha256":sha(run/"process_wall_receipt.json"),
        "script_sha256":sha(Path(__file__)),
        "time_accounting":{"prior_seconds":summary["cumulative_prior_wall_seconds"],
            "full_process_wall_seconds":wall["process_wall_seconds"],
            "charged_postwork_before_audit_seconds":prior_postwork_seconds,
            "audit_wall_seconds_through_last_check":measured_audit_seconds,
            "cumulative_seconds_through_this_audit":cumulative_through_audit,"cap_seconds":7200,
            "final_publication_and_shutdown_not_yet_charged":True},
        "limits":[
            "Every present raw/full-probe roundtrip was performed by the executor. Independent semantic decoding covers the declared available prespecified sample; every present raw/full file is authenticated.",
            "This receipt establishes cumulative compliance through its measured audit only. Subsequent publication/report work and shutdown require an added measured ledger charge.",
            "Resource stopping retains missing rows. No unexecuted candidate, fresh teacher completion, population/generalization/full-MPC or sole-cause claim is made."]}



def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--original-root",type=Path,required=True)
    p.add_argument("--analysis",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--prior-postwork-seconds",type=float,default=0.)
    a=p.parse_args()
    need(a.prior_postwork_seconds >= 0 and math.isfinite(a.prior_postwork_seconds),"Invalid postwork charge")
    result=audit(a.run.resolve(),a.original_root.resolve(),a.analysis.resolve(),a.prior_postwork_seconds)
    with a.output.open("x",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")
    print(json.dumps({k:result[k] for k in ("status","scope","teacher_unique_rows","time_accounting")}))


if __name__ == "__main__":
    main()
