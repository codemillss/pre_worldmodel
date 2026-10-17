"""Describe finalized P4 data collection without simulation, models, or training.

Every raw NPZ is bound to its episode manifest and the finalized run inventory.
This is a descriptive account of recorded motion, not a model-effect test or a
second physical-validity checker. The run is opened read-only throughout.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DISPLAY_VELOCITY_TOLERANCE_M_S = 1e-9


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON token: {value}")
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject_constant)


def relative_to_run(run, path):
    path = Path(path).resolve()
    require(path.is_relative_to(run), f"Source escaped run: {path}")
    return path.relative_to(run).as_posix()


def bound_episode(run, row, inventory):
    folder = (run / row["dataset"]).resolve()
    manifest_path = folder / "manifest.json"
    manifest_rel = relative_to_run(run, manifest_path)
    manifest_hash = sha256(manifest_path)
    require(manifest_hash == row["manifest_sha256"] == inventory[manifest_rel],
            f"Manifest hash mismatch: {row['label']}")
    manifest = read_json(manifest_path)
    require(len(manifest["episodes"]) == 1, "Each collection dataset must contain one episode")
    entry = manifest["episodes"][0]
    require(entry["episode_id"] == row["label"] and entry["family_id"] == row["family_id"],
            "Summary and manifest episode identity differ")
    require(entry["split"] == row["split"], "Summary and manifest split differ")
    bindings = {manifest_rel: manifest_hash}
    for name, hash_key in (("npz", "npz_sha256"), ("json", "json_sha256")):
        source = (folder / entry[name]).resolve()
        require(source.parent == folder / "episodes", "Unexpected episode artifact directory")
        rel = relative_to_run(run, source)
        actual_hash = sha256(source)
        require(actual_hash == entry[hash_key] == inventory[rel], f"Hash mismatch: {rel}")
        bindings[rel] = actual_hash
    with np.load(folder / entry["npz"], allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in (
            "state", "sim_time", "action_requested", "action_applied", "actual_dt",
            "transition_valid", "substep_offsets", "substep_states", "substep_times",
            "contact_summary")}
    require(np.isfinite(arrays["state"]).all() and np.isfinite(arrays["substep_states"]).all(),
            f"Non-finite states need a separate failure analysis: {row['label']}")
    require(len(arrays["action_applied"]) == row["measurements"]["transition_count"],
            "Raw and summary transition count differ")
    return arrays, bindings


def motion_segment(states):
    """Net displacement and path length differ if motion reverses within the block."""
    result = {}
    for label, columns in (("pusher", slice(6, 8)), ("object", slice(0, 2))):
        xy = states[:, columns]
        delta = xy[-1] - xy[0]
        result[f"{label}_net_xy_m"] = delta.tolist()
        result[f"{label}_net_distance_m"] = float(np.linalg.norm(delta))
        result[f"{label}_path_length_m"] = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    return result


def stop_description(row, arrays):
    """S1's predeclared core actions 10..14 form one measured half-second block."""
    require(len(arrays["action_applied"]) == 20, "Stop analysis requires a complete core")
    require(np.array_equal(arrays["action_applied"][10:15], np.zeros((5, 2))),
            "Registered S1 stop block is not five applied zero commands")
    require(np.array_equal(arrays["action_requested"][10:15], np.zeros((5, 2))),
            "Registered S1 stop block is not five requested zero commands")
    offsets = arrays["substep_offsets"]
    begin, middle, end = (int(offsets[i]) for i in (10, 11, 15))
    raw = np.vstack([arrays["state"][10], arrays["substep_states"][begin:end]])
    middle_index = middle - begin
    require(middle_index > 0 and len(raw) > middle_index, "Stop block has no physical substeps")
    require(np.array_equal(raw[middle_index], arrays["state"][11])
            and np.array_equal(raw[-1], arrays["state"][15]), "Substeps and stop boundaries disagree")
    duration = float(arrays["actual_dt"][10:15].sum())
    require(abs(duration - .5) < 1e-10, "Unexpected stop duration")
    tol = DISPLAY_VELOCITY_TOLERANCE_M_S
    vy = raw[:, 9]
    pusher_speed = np.linalg.norm(raw[:, 8:10], axis=1)
    object_speed = np.linalg.norm(raw[:, 3:5], axis=1)
    initial_direction = np.sign(vy[0]) if abs(vy[0]) > tol else 0.
    checkpoints = {}
    for label, index in (("start", 0), ("first_substep", 1),
                         ("after_0_1s", middle_index), ("end_0_5s", len(raw) - 1)):
        state = raw[index]
        checkpoints[label] = {
            "pusher_vx_m_s": float(state[8]), "pusher_vy_m_s": float(state[9]),
            "pusher_speed_m_s": float(pusher_speed[index]),
            "object_vx_m_s": float(state[3]), "object_vy_m_s": float(state[4]),
            "object_speed_m_s": float(object_speed[index]), "object_wz_rad_s": float(state[5]),
        }
    return {
        "episode_id": row["label"], "family_id": row["family_id"],
        "semantic_split": row["semantic_split"], "Y": row["Y"], "geometry": row["geometry"],
        "core_action_start": 10, "core_action_end_exclusive": 15,
        "duration_s": duration, "raw_state_samples_including_start": len(raw),
        "checkpoints": checkpoints,
        "min_signed_pusher_vy_m_s": float(vy.min()),
        "max_signed_pusher_vy_m_s": float(vy.max()),
        "max_pusher_speed_m_s": float(pusher_speed.max()),
        "max_object_speed_m_s": float(object_speed.max()),
        "max_abs_object_wz_rad_s": float(np.abs(raw[:, 5]).max()),
        "end_abs_object_wz_rad_s": float(abs(raw[-1, 5])),
        "initial_pusher_speed_nonzero_for_display": bool(pusher_speed[0] > tol),
        "initial_pusher_vy_nonzero_for_display": bool(abs(vy[0]) > tol),
        "negative_pusher_vy_observed_for_display": bool(np.any(vy < -tol)),
        "pusher_vy_reversed_from_initial_direction_for_display": bool(
            initial_direction != 0 and np.any(vy * initial_direction < -tol)),
        "all_0_5s": motion_segment(raw),
        "first_0_1s": motion_segment(raw[:middle_index + 1]),
        "remaining_0_4s": motion_segment(raw[middle_index:]),
    }


def numeric_summary(rows, accessor):
    """Preserve both extreme episode IDs; ties stay visible without rounding."""
    if not rows:
        return {"count": 0, "minimum": None, "maximum": None, "mean": None,
                "minimum_episode_ids": [], "maximum_episode_ids": []}
    values = np.asarray([accessor(row) for row in rows], dtype=np.float64)
    require(np.isfinite(values).all(), "Non-finite descriptive metric")
    lo, hi = float(values.min()), float(values.max())
    return {"count": len(rows), "minimum": lo, "maximum": hi, "mean": float(values.mean()),
            "minimum_episode_ids": [row["episode_id"] for row, value in zip(rows, values) if value == lo],
            "maximum_episode_ids": [row["episode_id"] for row, value in zip(rows, values) if value == hi]}


def stop_group(rows, label):
    metric_paths = [
        "min_signed_pusher_vy_m_s", "max_signed_pusher_vy_m_s", "max_pusher_speed_m_s",
        "max_object_speed_m_s", "max_abs_object_wz_rad_s", "end_abs_object_wz_rad_s",
    ]
    metric_paths += [f"checkpoints.{point}.{field}" for point in
                     ("start", "first_substep", "after_0_1s", "end_0_5s")
                     for field in ("pusher_vy_m_s", "pusher_speed_m_s", "object_speed_m_s", "object_wz_rad_s")]
    metric_paths += [f"{period}.{entity}_{quantity}" for period in
                     ("all_0_5s", "first_0_1s", "remaining_0_4s")
                     for entity in ("pusher", "object")
                     for quantity in ("net_distance_m", "path_length_m")]
    metrics = {}
    for path in metric_paths:
        def get(row, components=path.split(".")):
            for key in components:
                row = row[key]
            return row
        metrics[path] = numeric_summary(rows, get)
    flags = ["initial_pusher_speed_nonzero_for_display", "initial_pusher_vy_nonzero_for_display",
             "negative_pusher_vy_observed_for_display",
             "pusher_vy_reversed_from_initial_direction_for_display"]
    return {"group": label, "episodes": len(rows),
            "unique_families": len({row["family_id"] for row in rows}),
            "flag_counts": {key: sum(row[key] for row in rows) for key in flags},
            "flag_episode_ids": {key: [row["episode_id"] for row in rows if row[key]] for key in flags},
            "metrics": metrics}


def contact_groups(core_rows, by_geometry=False):
    groups = defaultdict(list)
    for row in core_rows:
        key = (row["semantic_split"], row["Y"], row["S"])
        if by_geometry:
            key += (row["geometry"],)
        groups[key].append(row)
    result = []
    for key, rows in sorted(groups.items()):
        contact = sorted({r["family_id"] for r in rows if r["contact_transitions"] > 0})
        noncontact = sorted({r["family_id"] for r in rows if r["noncontact_transitions"] > 0})
        result.append({"semantic_split": key[0], "Y": key[1], "S": key[2],
            **({"geometry": key[3]} if by_geometry else {}),
            "episodes": len(rows), "families": len({r["family_id"] for r in rows}),
            "transitions": sum(r["transitions"] for r in rows),
            "valid_transitions": sum(r["valid_transitions"] for r in rows),
            "contact_transitions": sum(r["contact_transitions"] for r in rows),
            "noncontact_transitions": sum(r["noncontact_transitions"] for r in rows),
            "contact_families": contact, "noncontact_families": noncontact,
            "contact_family_count": len(contact), "noncontact_family_count": len(noncontact),
            "families_with_both": sorted(set(contact) & set(noncontact))})
    return result


def summarize(run):
    run = Path(run).resolve()
    registration = read_json(run / "registration.json")
    summary = read_json(run / "summary.json")
    inventory_path = run / "artifact_hashes.json"
    inventory = read_json(inventory_path)["sha256"]
    source_hashes = {name: sha256(run / name) for name in ("registration.json", "summary.json")}
    for name, digest in source_hashes.items():
        require(digest == inventory[name], f"Finalized inventory mismatch: {name}")
    require(source_hashes["registration.json"] == summary["registration_sha256"],
            "Summary is bound to another registration")
    source_hashes["artifact_hashes.json"] = sha256(inventory_path)
    plans = {p["episode_id"]: p for p in registration["plans"]}
    warmup_rows, core_rows, stop_rows, unavailable_stop = [], [], [], []
    for row in summary["episodes"]:
        arrays, bindings = bound_episode(run, row, inventory)
        source_hashes.update(bindings)
        if row["role"] == "warmup":
            warmup_rows.append({"episode_id": row["label"], "family_id": row["family_id"],
                "semantic_split": row["semantic_split"], "Y": row["Y"], "geometry": row["geometry"],
                "measured_end_pusher_vy_m_s": float(arrays["state"][-1, 9]),
                "transitions": len(arrays["action_applied"])})
            continue
        require(row["role"] == "core", "Unrecognized main collection role")
        p = plans[row["label"]]
        require(p["family_id"] == row["family_id"] and p["semantic_split"] == row["semantic_split"],
                "Recorded core differs from registered family/split")
        contact = int(np.count_nonzero(arrays["contact_summary"][:, 2] > 0))
        noncontact = int(np.count_nonzero(arrays["contact_summary"][:, 2] == 0))
        require(contact == row["measurements"]["active_contact_transitions"]
                and noncontact == row["measurements"]["noncontact_transitions"],
                "Raw and summary contact counts disagree")
        core_rows.append({"episode_id": row["label"], "family_id": row["family_id"],
            "semantic_split": row["semantic_split"], "Y": row["Y"], "S": row["S"],
            "geometry": row["geometry"], "transitions": len(arrays["action_applied"]),
            "valid_transitions": int(arrays["transition_valid"].sum()),
            "contact_transitions": contact, "noncontact_transitions": noncontact})
        if row["S"] == "S1":
            if len(arrays["action_applied"]) == 20 and bool(arrays["transition_valid"].all()):
                stop_rows.append(stop_description(row, arrays))
            else:
                unavailable_stop.append({"episode_id": row["label"],
                                         "reason": "incomplete_or_invalid_core"})
    warmup_groups = []
    for semantic in ("train", "development"):
        for y in ("Y0", "Y1"):
            rows = [r for r in warmup_rows if r["semantic_split"] == semantic and r["Y"] == y]
            warmup_groups.append({"semantic_split": semantic, "Y": y,
                "cells": len(rows), "unique_families": len({r["family_id"] for r in rows}),
                "measured_end_pusher_vy_m_s": numeric_summary(rows, lambda r: r["measured_end_pusher_vy_m_s"])})
    groups = [stop_group(stop_rows, {"selection": "all_recorded_valid_S1"})]
    for semantic in ("train", "development"):
        for y in ("Y0", "Y1"):
            rows = [r for r in stop_rows if r["semantic_split"] == semantic and r["Y"] == y]
            groups.append(stop_group(rows, {"semantic_split": semantic, "Y": y}))
    recorded = {row["episode_id"] for row in core_rows}
    missing = [p["episode_id"] for p in registration["plans"] if p["episode_id"] not in recorded]
    require(sum(r["contact_transitions"] + r["noncontact_transitions"] for r in core_rows)
            == sum(r["transitions"] for r in core_rows), "Contact categories do not exhaust raw intervals")
    require(all(row["all_0_5s"][f"{entity}_path_length_m"] + 1e-15
                >= row["all_0_5s"][f"{entity}_net_distance_m"]
                for row in stop_rows for entity in ("object", "pusher")), "Path length cannot be below net displacement")
    return {
        "schema": "p4-data-main-descriptive-v1", "status": "DESCRIPTIVE_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "run": str(run),
        "source_collection_status": summary["status"],
        "source_sha256_relative_to_run": source_hashes,
        "analysis_source_sha256": sha256(Path(__file__)),
        "counts": {"registered_core_episodes": len(registration["plans"]),
            "recorded_core_episodes": len(core_rows), "physical_warmups": len(warmup_rows),
            "S1_stops_analyzed": len(stop_rows), "planned_S1_stops": 72,
            "core_transitions": sum(r["transitions"] for r in core_rows),
            "warmup_transitions": sum(r["transitions"] for r in warmup_rows)},
        "incomplete_scope": {"missing_core_ids": missing, "unavailable_S1_stops": unavailable_stop},
        "definitions": {
            "state_units": "SI; world XY; signed world Y velocity; object angular velocity in rad/s",
            "contact": "transition has at least one pusher-object active-contact sample in the recorded contact summary; source threshold is normal force > 0.001 N",
            "family_counts": "unique family IDs; contact and noncontact sets can overlap; separate S groups do not inflate aggregate family counts",
            "stop": "S1 core action indices 10..14, using state10 and every recorded physics substep through state15",
            "movement": "net distance is norm(end XY - start XY); path length sums substep XY distances and includes reversals",
            "display_velocity_tolerance_m_s": DISPLAY_VELOCITY_TOLERANCE_M_S,
            "display_tolerance_scope": "Only descriptive nonzero/negative/reversal counts; not a physical validity, task success, or collection acceptance threshold",
            "reversal": "Initial signed pusher Y is above display tolerance in magnitude, then a later sample has opposite signed Y above tolerance",
            "source_validation_scope": "Finalized inventory, registration/summary, manifests and NPZ/metadata hashes checked; array arithmetic checked; no physical rerun or full independent dataset-contract revalidation",
        },
        "warmup_actual_Y_by_split": warmup_groups, "warmup_cells": warmup_rows,
        "core_contact_by_split_Y_S": contact_groups(core_rows),
        "core_contact_by_split_Y_S_geometry": contact_groups(core_rows, by_geometry=True),
        "core_episodes": core_rows, "S1_stop_groups": groups, "S1_stop_episodes": stop_rows,
        "scope": {"physics_intervals": 0, "model_forwards": 0, "learning_updates": 0,
            "data_effect_causal_inference": "NOT_EVALUATED", "closed_loop_alignment_effect": "NOT_EVALUATED",
            "task_success": "NOT_EVALUATED", "generalization": "NOT_EVALUATED", "P5_01": "HOLD",
            "claim": "Describes what was collected; this does not establish that data enrichment improves the learned model or planning"},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/p4_data_main_descriptive.json")
    args = parser.parse_args()
    require(not args.output.resolve().is_relative_to(args.run_dir.resolve()), "Analysis output must not modify the source run")
    result = summarize(args.run_dir)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "counts": result["counts"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
