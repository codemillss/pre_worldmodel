"""Read-only independent audit of the frozen P4 main data collection.

The ordinary episode reader validates saved raw samples and reconstructs
snapshots using mj_forward. This checker never constructs an environment,
calls mj_step, executes a learned model, or trains. Missing/invalid episodes
remain explicit failures; a partial collection cannot receive PASS.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from push_align.config import ROOT
from push_align.data import Episode, digest, iter_dataset, validate_family_splits
from push_align.data_intervention import build_preflight_plan
from push_align.data_intervention_main import build_main_plan
from check_p4_data_preflight import BOUNDARY_FIELDS, zero_blocks


ARMS = {"D00": ("Y0", "S0"), "D10": ("Y1", "S0"),
        "D01": ("Y0", "S1"), "D11": ("Y1", "S1")}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def slot_design_from_schedule(positions, offsets):
    """Recompute SHA-selected slots using only saved integer schedule arrays."""
    sizes = np.diff(offsets)
    if (positions.shape != (62992,) or offsets.shape != (1001,)
            or offsets[0] != 0 or offsets[-1] != len(positions)
            or np.any((positions < 0) | (positions >= 2960))
            or int((sizes == 64).sum()) != 979 or int((sizes == 16).sum()) != 21
            or np.any((sizes != 64) & (sizes != 16))):
        raise ValueError("Historical schedule shape, range or batch counts changed")
    counts = np.bincount(positions, minlength=2960)
    partial = np.concatenate([positions[offsets[i]:offsets[i + 1]]
                              for i, n in enumerate(sizes) if n == 16])
    partial_counts = np.bincount(partial, minlength=2960)
    eligible = np.flatnonzero((counts == 21) & (partial_counts == 0)).tolist()
    if len(eligible) != 1897:
        raise ValueError("Historical eligible slot count changed")
    chosen = sorted(sorted(eligible, key=lambda i: hashlib.sha256(
        f"p4-data-v1-slot:{i}".encode()).hexdigest())[:384])
    return [{"slot": slot, "family_rank": i // 16, "window_start": i % 16,
             "occurrences": int(counts[slot])} for i, slot in enumerate(chosen)]


def contact_transition_mask(episode, threshold):
    """A contact transition has any raw pusher/object force strictly > threshold."""
    return np.asarray([any(contact["pair"] == "pusher_object"
                           and contact["normal_force_N"] > threshold
                           for sample in diagnostic["samples"] for contact in sample["contacts"])
                       for diagnostic in episode.metadata["diagnostics"]], dtype=bool)


def core_window_statistics(episode):
    a = episode.arrays
    zero = np.all(a["action_applied"] == 0, axis=1)
    starts = [i for i in range(max(0, episode.transition_count - 4))
              if a["transition_valid"][i:i + 5].all()]
    return {"valid_starts": starts, "zero_command_count": int(zero.sum()),
            "zero_occurrences": sum(int(zero[i:i + 5].sum()) for i in starts),
            "all_zero_starts": [i for i in starts if zero[i:i + 5].all()]}


def audit(run):
    registration = read(run / "registration.json")
    summary = read(run / "summary.json")
    checks, cells, episode_rows, identities = [], [], [], []

    def check(name, passed, **detail):
        checks.append({"name": name, "pass": bool(passed), **detail})

    hashes = read(run / "artifact_hashes.json")["sha256"]
    mismatches = [p for p, sha in hashes.items()
                  if not (run / p).is_file() or digest(run / p) != sha]
    actual_files = {str(p.relative_to(run)) for p in run.rglob("*") if p.is_file()}
    check("all_registered_run_artifact_hashes", not mismatches, mismatches=mismatches)
    check("all_run_files_indexed", actual_files == set(hashes) | {"artifact_hashes.json"},
          unindexed=sorted(actual_files - set(hashes) - {"artifact_hashes.json"}))
    for key in ("source_sha256", "immutable_source_sha256"):
        bound = registration.get(key, {})
        mismatches = [p for p, sha in bound.items()
                      if not (ROOT / p).is_file() or digest(ROOT / p) != sha]
        check(key + "_unchanged", bool(bound) and not mismatches, mismatches=mismatches)
    config, design = registration["physical_config"], registration["design"]
    plans = registration["plans"]
    check("registered_plans_reproduce", build_main_plan(design, config) == plans)
    check("144_core_plans_24_train_12_development_families",
          len(plans) == len({p["episode_id"] for p in plans}) == 144
          and Counter(p["semantic_split"] for p in plans) == {"train": 96, "development": 48}
          and Counter({p["family_id"]: p["semantic_split"] for p in plans}.values())
          == {"train": 24, "development": 12})
    preflight = build_preflight_plan(design, config)
    main_families = {p["family_id"] for p in plans}
    object_key = lambda p: tuple(p["initial_spec"][k] for k in ("object_x", "object_y", "object_yaw"))
    check("preflight_families_and_base_poses_excluded",
          not main_families.intersection(p["family_id"] for p in preflight)
          and not {object_key(p) for p in plans}.intersection(object_key(p) for p in preflight))
    with (run / "initial_conditions.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    csv_expected = [{**{k: str(p[k]) for k in ("episode_id", "family_id", "semantic_split", "geometry")},
                     "Y": p["y_condition"], "S": p["stop_condition"],
                     **{k: str(v) for k, v in p["initial_spec"].items()},
                     "command_vx": str(p["moving_action"][0]), "command_vy": str(p["moving_action"][1])}
                    for p in plans]
    check("frozen_readable_initial_condition_list", csv_rows == csv_expected
          and digest(run / "initial_conditions.csv") == registration.get("initial_conditions_csv_sha256"))
    preflight_run = ROOT / registration["preflight_run"]
    preflight_hashes = read(preflight_run / "artifact_hashes.json")["sha256"]
    preflight_mismatch = [p for p, sha in preflight_hashes.items()
                         if not (preflight_run / p).is_file() or digest(preflight_run / p) != sha]
    check("preflight_evidence_still_unchanged", not preflight_mismatch, mismatches=preflight_mismatch)
    check("summary_complete_without_failure", summary.get("status") == "PASS"
          and summary.get("failure") is None)
    check("no_learning_model_final_test_or_handoff_claim",
          summary.get("learning_updates") == summary.get("model_forwards") == 0
          and summary.get("P5_01") == "HOLD" and summary.get("task_success") == "NOT_EVALUATED")
    summary_labels = {row["label"]: row for row in summary.get("episodes", [])}
    expected_labels, loaded_labels, core_labels = set(), set(), set()
    counts, window_counts = Counter(), Counter()
    coverage = defaultdict(lambda: {"contact": set(), "noncontact": set()})
    threshold = config["evaluation"]["contact_active_normal_force_min_N"]
    dt_tolerance = config["validity"]["dt_match_tolerance_s"]

    for i in range(0, len(plans), 2):
        p0, p1 = plans[i:i + 2]
        cell = p0["family_id"] + "-" + p0["y_condition"]
        roles = {"warmup": ("warmup", p0, p0["warmup_actions"]),
                 "S0": ("core", p0, p0["core_actions"]),
                 "S1": ("core", p1, p1["core_actions"])}
        episodes = {}
        for suffix, (role, plan, actions) in roles.items():
            label = cell + "-" + suffix
            expected_labels.add(label)
            folder = run / "episodes" / label
            if not (folder / "manifest.json").is_file():
                check(label + ":present", False, execution_state="NOT_RUN_OR_INCOMPLETE")
                continue
            try:
                manifest = read(folder / "manifest.json")
                episode, = list(iter_dataset(folder, manifest["contract"]))
                episodes[suffix] = episode
                loaded_labels.add(label)
                a, m = episode.arrays, episode.metadata
                identity = {key: m[key] for key in ("episode_id", "episode_index", "family_id",
                            "family_definition", "initial_record", "split")}
                identities.append(Episode({}, identity))
                c, policy = m["contract"], m["collection_policy"]
                check(label + ":contract_split_and_commands",
                      manifest["purpose"] == "main" and m["split"] == plan["split"]
                      and m["episode_id"] == label and m["family_id"] == plan["family_id"]
                      and policy.get("semantic_split") == plan["semantic_split"]
                      and policy["role"] == role and policy["predeclared_plan"] == plan
                      and policy["training_eligible"] is (role == "core" and plan["semantic_split"] == "train")
                      and policy["development_evaluation_eligible"] is (role == "core" and plan["semantic_split"] == "development")
                      and m["initial_record"]["initial_spec"] == plan["initial_spec"]
                      and c["resolved_config"] == config
                      and np.array_equal(a["action_requested"], np.asarray(actions))
                      and np.array_equal(a["action_requested"], a["action_applied"])
                      and bool((np.linalg.norm(a["action_applied"], axis=1)
                                <= design["collection"]["command_norm_limit_m_s"]).all()))
                check(label + ":physical_geometry_and_unassessed_handoff",
                      bool(a["transition_valid"].all()) and all(g["pass"] for g in m["geometry_checks"])
                      and np.all(np.abs(a["actual_dt"] - .1) <= dt_tolerance)
                      and len(a["substep_times"]) == len(actions) * c["frame_skip"]
                      and m["task_outcome"] == "not_evaluated"
                      and m["terminal_reason"] == "collection_complete")
                srow = summary_labels.get(label, {})
                check(label + ":summary_manifest_binding",
                      srow.get("manifest_sha256") == digest(folder / "manifest.json")
                      and srow.get("dataset") == str(folder.relative_to(run))
                      and srow.get("semantic_split") == plan["semantic_split"]
                      and srow.get("split") == plan["split"]
                      and srow.get("measurements", {}).get("transition_count") == episode.transition_count)
                contact = contact_transition_mask(episode, threshold)
                check(label + ":raw_contact_label", np.array_equal(contact, a["contact_summary"][:, 2] > 0))
                counts[role] += episode.transition_count
                counts["physics_substeps"] += len(a["substep_times"])
                counts["episodes"] += 1
                row = {"label": label, "role": role, "split": plan["split"],
                       "semantic_split": plan["semantic_split"], "family_id": plan["family_id"],
                       "transitions": episode.transition_count, "physics_substeps": len(a["substep_times"]),
                       "active_contact_transitions": int(contact.sum()),
                       "noncontact_transitions": int((~contact).sum())}
                if role == "core":
                    core_labels.add(label)
                    coverage_key = plan["semantic_split"] + ":" + plan["y_condition"]
                    if contact.any(): coverage[coverage_key]["contact"].add(plan["family_id"])
                    if (~contact).any(): coverage[coverage_key]["noncontact"].add(plan["family_id"])
                    stats = core_window_statistics(episode)
                    check(label + ":K5_contract", stats == {
                        "valid_starts": list(range(16)), "zero_command_count": 5,
                        "zero_occurrences": 25,
                        "all_zero_starts": [10] if suffix == "S1" else []})
                    arm = next(k for k, v in ARMS.items() if v == (plan["y_condition"], suffix))
                    window_counts[plan["semantic_split"] + ":" + arm] += len(stats["valid_starts"])
                    row["window_statistics"] = stats
                episode_rows.append(row)
            except Exception as error:
                check(label + ":ordinary_episode_validation", False,
                      error=f"{type(error).__name__}: {error}")
        if len(episodes) != 3:
            continue
        warm, s0, s1 = [episodes[key] for key in ("warmup", "S0", "S1")]
        vy = float(warm.arrays["state"][-1, 9])
        high = p0["y_condition"] == "Y1"
        check(cell + ":measured_Y", vy >= design["collection"]["measured_high_Y_gate_m_s"] if high
              else vy <= design["collection"]["measured_low_Y_ceiling_m_s"], measured_vy_m_s=vy)
        shared = all(np.array_equal(warm.arrays[key][-1], e.arrays[key][0])
                     for key in BOUNDARY_FIELDS for e in (s0, s1))
        shared = shared and all(warm.metadata["snapshot_rng_states"][-1] == e.metadata["snapshot_rng_states"][0]
                                for e in (s0, s1))
        check(cell + ":shared_exact_warmup_snapshot", shared)
        head = all(np.array_equal(s0.arrays[key][:6], s1.arrays[key][:6]) for key in BOUNDARY_FIELDS)
        head = head and all(np.array_equal(s0.arrays[key][:5], s1.arrays[key][:5])
                            for key in ("action_requested", "action_applied", "actual_dt", "step_valid",
                                        "geometry_valid", "transition_valid", "contact_summary", "validity_summary"))
        head = head and all(s0.metadata[key][:5] == s1.metadata[key][:5]
                            for key in ("diagnostics", "geometry_checks"))
        head = head and s0.metadata["snapshot_rng_states"][:6] == s1.metadata["snapshot_rng_states"][:6]
        prefix_substeps = 5 * config["physics"]["frame_skip"]
        head = head and all(np.array_equal(s0.arrays[key][:prefix_substeps], s1.arrays[key][:prefix_substeps])
                            for key in ("substep_states", "substep_times"))
        check(cell + ":same_speed_first5_exact", head)
        stops = {}
        for suffix, episode in (("S0", s0), ("S1", s1)):
            stops[suffix] = zero_blocks(episode, config)
            lengths = [5] if suffix == "S1" else [1] * 5
            check(cell + ":" + suffix + "_recorded_stop_intervals",
                  [r["end_action_exclusive"] - r["start_action"] for r in stops[suffix]] == lengths
                  and all(r["zero_requested_and_applied"] and r["finite_states"]
                          and r["substep_count"] == n * config["physics"]["frame_skip"]
                          and abs(r["actual_duration_s"] - n * .1) <= dt_tolerance
                          for r, n in zip(stops[suffix], lengths)))
        cells.append({"cell": cell, "semantic_split": p0["semantic_split"],
                      "geometry_intent": p0["geometry"], "Y": p0["y_condition"],
                      "warmup_end_measured_vy_m_s": vy, "stop_measurements": stops})
        print(json.dumps({"audited_cell": cell, "measured_vy_m_s": vy}), flush=True)
    try:
        assignments = validate_family_splits(identities)
        check("all_episode_family_lineage", Counter(assignments.values()) == {"train": 24, "validation": 12})
    except Exception as error:
        check("all_episode_family_lineage", False, error=f"{type(error).__name__}: {error}")
    actual_labels = {p.name for p in (run / "episodes").iterdir() if p.is_dir()}
    check("all216_expected_episodes_without_replacement", expected_labels == actual_labels == loaded_labels
          == set(summary_labels) and len(summary.get("episodes", [])) == len(summary_labels) == 216,
          missing_or_unvalidated=sorted(expected_labels - loaded_labels), extra=sorted(actual_labels - expected_labels))
    expected_counts = {"warmup": 144, "core": 2880, "physics_substeps": 604800, "episodes": 216}
    check("saved_transition_and_substep_accounting", dict(counts) == expected_counts,
          observed=dict(counts), expected=expected_counts)
    execution = summary.get("counts", {})
    check("execution_accounting_matches_saved_evidence", execution.get("action_calls") == 3024
          and execution.get("completed_intervals") == 3024
          and execution.get("physics_substeps") == counts["physics_substeps"]
          and execution.get("by_role") == {"warmup": 144, "core": 2880})
    settling = summary.get("settling", [])
    check("settling_separately_accounted", len(settling) == 73
          and summary.get("settling_substeps_total") == sum(r["settling_substeps"] for r in settling)
          and all(abs(r["settling_substeps"] * r["physics_dt_s"] - r["initial_record"]["settle_time_s"])
                  <= dt_tolerance for r in settling))
    coverage_rows = []
    for split, minimum in (("train", 3), ("development", 2)):
        for y in ("Y0", "Y1"):
            key = split + ":" + y
            row = {"semantic_split": split, "Y": y, "minimum_unique_families_per_regime": minimum,
                   **{regime + "_families": sorted(coverage[key][regime]) for regime in ("contact", "noncontact")}}
            check(key + ":measured_contact_coverage", all(len(coverage[key][regime]) >= minimum
                  for regime in ("contact", "noncontact")), **row)
            coverage_rows.append(row)
    expected_windows = {split + ":" + arm: n for split, n in (("train", 384), ("development", 192)) for arm in ARMS}
    check("core_only_window_counts_per_arm", dict(window_counts) == expected_windows,
          observed=dict(window_counts), expected=expected_windows)
    try:
        _audit_slots(run, registration, plans, core_labels, check)
    except Exception as error:
        check("slot_binding_audit_completed", False, error=f"{type(error).__name__}: {error}")
    output_bytes = sum(p.stat().st_size for p in run.rglob("*") if p.is_file())
    free = shutil.disk_usage(run).free
    check("actual_free_reserve", free >= design["resources"]["minimum_free_reserve_GiB"] * 1024**3,
          output_bytes=output_bytes, free_bytes=free)
    resource = registration["resource_plan"]
    preflight_bytes = sum(p.stat().st_size for p in preflight_run.rglob("*") if p.is_file())
    check("full_experiment_budget_preserves_future_reserve",
          preflight_bytes == resource["preflight_bytes"]
          and output_bytes <= resource["main_output_cap_bytes"]
          and preflight_bytes + resource["main_output_cap_bytes"] + resource["future_training_docs_reserve_bytes"]
              <= design["resources"]["future_full_experiment_output_limit_GiB"] * 1024**3
          and resource["free_bytes_before"] >= resource["minimum_free_bytes"] + resource["main_output_cap_bytes"],
          preflight_bytes=preflight_bytes, actual_main_bytes=output_bytes,
          future_reserved_bytes=resource["future_training_docs_reserve_bytes"])
    resource_checks = summary.get("resource_checks", {})
    check("collector_resource_checks_pass", bool(resource_checks) and all(resource_checks.values()),
          collector_checks=resource_checks)
    return {"schema": "p4-data-main-independent-review-v1",
            "status": "PASS" if all(c["pass"] for c in checks) else "FAIL",
            "run": str(run.relative_to(ROOT)), "registration_sha256": digest(run / "registration.json"),
            "summary_sha256": digest(run / "summary.json"), "audit_source_sha256": digest(__file__),
            "helper_source_sha256": digest(Path(__file__).with_name("check_p4_data_preflight.py")),
            "check_count": len(checks), "checks": checks, "cells": cells, "episodes": episode_rows,
            "coverage": coverage_rows, "window_counts": dict(window_counts), "observed_counts": dict(counts),
            "missing_or_unvalidated_episodes": sorted(expected_labels - loaded_labels),
            "execution_during_review": {"physical_action_intervals": 0, "model_forwards": 0, "learning_updates": 0,
                "snapshot_reconstruction": "ordinary episode reader uses mj_forward without mj_step"},
            "scope": "Saved main collection consistency, frozen split/slots and measured coverage; no learning effect, handoff success, independent final test or all-space physical validity claim"}


def _audit_slots(run, registration, plans, core_labels, check):
    bound = read(run / "slots_bound.json")
    design_slots = read(ROOT / "reports/p4_data_intervention_slots.design.json")
    schedule_path = ROOT / registration["design"]["training"]["historical_schedule"]
    with np.load(schedule_path, allow_pickle=False) as archive:
        regenerated = slot_design_from_schedule(archive["window_positions"], archive["batch_offsets"])
    check("historical_slot_rule_and_hash", regenerated == design_slots["entries"]
          and digest(schedule_path) == design_slots["source_schedule_sha256"])
    check("slot_binding_frozen_before_physics", registration.get("slots_bound_sha256") == digest(run / "slots_bound.json"))
    lookup = {(p["family_rank"], p["y_condition"], p["stop_condition"]): p
              for p in plans if p["semantic_split"] == "train"}
    original_path = ROOT / "runs/p4-02-20261012-01/train_windows.npz"
    with np.load(original_path, allow_pickle=False) as archive:
        rows = archive["window_rows"]
    with np.load(ROOT / "runs/p3-02-20261009-01/learning_data/prepared.npz", allow_pickle=False) as archive:
        prepared = archive["windows_5"]
        original_train_rows = prepared[archive["split_code"][prepared[:, 0]] == 0]
    check("original2960_training_rows_match_prepared", rows.shape == (2960, 5)
          and np.array_equal(rows, original_train_rows)
          and registration["immutable_source_sha256"].get(str(original_path.relative_to(ROOT))) == digest(original_path))
    expected_entries = []
    for row in regenerated:
        expected_entries.append({**row, "family_id": lookup[row["family_rank"], "Y0", "S0"]["family_id"],
                                 "arm_episodes": {arm: lookup[row["family_rank"], *conditions]["episode_id"]
                                                  for arm, conditions in ARMS.items()},
                                 "replaced_original_prepared_rows": rows[row["slot"]].tolist()})
    check("384_actual_family_window_bindings", bound.get("entries") == expected_entries)
    check("all_new_slot_episodes_saved_core_only", all(episode in core_labels
          for row in bound.get("entries", []) for episode in row["arm_episodes"].values()))
    selected = {row["slot"] for row in regenerated}
    old_slots = [i for i in range(2960) if i not in selected]
    old = bound.get("unchanged_old_slots")
    # Accept a direct integer list or explicit original row records. Both must
    # contain exactly the same original 2,576 slot positions, in original order.
    if old and isinstance(old[0], dict):
        check("2576_original_slots_unchanged", [row["slot"] for row in old] == old_slots)
        check("unchanged_original_window_rows", all(row.get("window_rows") == rows[row["slot"]].tolist() for row in old))
    else:
        check("2576_original_slots_unchanged", old == old_slots)
    check("slot_no_model_execution", summary_not_training(bound))


def summary_not_training(bound):
    """Do not infer completed training from a planning-only slot manifest."""
    return bound.get("learning_updates", 0) == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not all((args.run_dir / name).is_file() for name in ("summary.json", "artifact_hashes.json")):
        parser.error("The run must be finalized before independent review")
    report = audit(args.run_dir.resolve())
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "checks": report["check_count"],
                      "failures": [c["name"] for c in report["checks"] if not c["pass"]]}), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
