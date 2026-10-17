"""Independent read-only audit of saved P4 preflight evidence.

No environment is instantiated and no mj_step/model call is made. The ordinary
episode reader recomputes raw geometry and reconstructs saved snapshots with
mj_forward; this is evidence validation, not a new physical trajectory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np

from push_align.config import ROOT
from push_align.data import Episode, digest, iter_dataset, validate_family_splits
from push_align.data_intervention import build_preflight_plan


BOUNDARY_FIELDS = {
    "state", "sim_time", "raw_integration_state", "raw_reference_height",
    "raw_warning_counts", "raw_warning_lastinfo", "raw_needs_reset", "observation_valid",
}
EXACT_FIELDS = BOUNDARY_FIELDS | {
    "action_requested", "action_applied", "actual_dt", "step_valid", "geometry_valid",
    "transition_valid", "substep_states", "substep_times", "contact_summary", "validity_summary",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def joined(episodes, field):
    return np.concatenate([
        e.arrays[field][1:] if i and field in BOUNDARY_FIELDS else e.arrays[field]
        for i, e in enumerate(episodes)
    ])


def raw_exact(left, right):
    details = {key: bool(np.array_equal(joined(left, key), joined(right, key)))
               for key in sorted(EXACT_FIELDS)}
    for key in ("diagnostics", "geometry_checks"):
        details[key] = ([d for e in left for d in e.metadata[key]] ==
                        [d for e in right for d in e.metadata[key]])
    details["snapshot_rng_states"] = (
        [v for i, e in enumerate(left) for v in e.metadata["snapshot_rng_states"][int(i > 0):]] ==
        [v for i, e in enumerate(right) for v in e.metadata["snapshot_rng_states"][int(i > 0):]])
    return details


def zero_blocks(episode, config):
    """Measure the command interval and the physical response separately."""
    a, d = episode.arrays, episode.metadata["diagnostics"]
    zero = np.all(a["action_applied"] == 0, axis=1)
    blocks, first = [], None
    threshold = config["evaluation"]["contact_active_normal_force_min_N"]
    for stop, is_zero in enumerate([*zero, False]):
        if is_zero and first is None:
            first = stop
        if is_zero or first is None:
            continue
        samples = [d[first]["samples"][0]] + [s for item in d[first:stop] for s in item["samples"][1:]]
        states = np.asarray([s["state"] for s in samples])
        elapsed = np.asarray([s["sim_time"] for s in samples]) - samples[0]["sim_time"]
        speed = np.linalg.norm(states[:, 8:10], axis=1)
        speed_limit = config["reset"]["linear_speed_tolerance_m_s"]
        below = speed <= speed_limit
        # Descriptive settling criterion only, not the task handoff predicate.
        suffix_below = np.logical_and.accumulate(below[::-1])[::-1]
        first_settled = np.flatnonzero(suffix_below)
        row = {
            "start_action": first, "end_action_exclusive": stop,
            "actual_duration_s": float(a["actual_dt"][first:stop].sum()),
            "substep_count": len(samples) - 1,
            "zero_requested_and_applied": bool(np.all(a["action_requested"][first:stop] == 0)
                                               and np.all(a["action_applied"][first:stop] == 0)),
            "finite_states": bool(np.isfinite(states).all()),
            "initial_pusher_velocity_xy_m_s": states[0, 8:10].tolist(),
            "first_substep_pusher_velocity_xy_m_s": states[1, 8:10].tolist(),
            "final_pusher_velocity_xy_m_s": states[-1, 8:10].tolist(),
            "final_object_velocity_xy_m_s": states[-1, 3:5].tolist(),
            "final_object_wz_rad_s": float(states[-1, 5]),
            "pusher_displacement_xy_m": (states[-1, 6:8] - states[0, 6:8]).tolist(),
            "object_displacement_xy_m": (states[-1, :2] - states[0, :2]).tolist(),
            "maximum_object_speed_m_s": float(np.linalg.norm(states[:, 3:5], axis=1).max()),
            "maximum_pusher_speed_m_s": float(speed.max()),
            "first_time_pusher_stays_below_0_001_m_s": float(elapsed[first_settled[0]]) if len(first_settled) else None,
            "settling_speed_threshold_m_s": speed_limit,
            "active_contact_samples": sum(any(c["pair"] == "pusher_object" and c["normal_force_N"] > threshold
                                                for c in s["contacts"]) for s in samples),
            "pusher_vy_at_times": [],
        }
        for query in (.0005, .01, .05, .1, .5):
            if query <= elapsed[-1] + 1e-9:
                index = int(np.argmin(np.abs(elapsed - query)))
                row["pusher_vy_at_times"].append({"requested_time_s": query,
                    "recorded_time_s": float(elapsed[index]), "vy_m_s": float(states[index, 9])})
        blocks.append(row)
        first = None
    return blocks


def audit(run):
    registration = read(run / "registration.json")
    summary = read(run / "summary.json")
    checks, cells, episode_rows, identities = [], [], [], []

    def check(name, passed, **detail):
        checks.append({"name": name, "pass": bool(passed), **detail})

    hash_index = read(run / "artifact_hashes.json")["sha256"]
    mismatch = [p for p, expected in hash_index.items()
                if not (run / p).is_file() or digest(run / p) != expected]
    actual_files = {str(p.relative_to(run)) for p in run.rglob("*") if p.is_file()}
    check("all_registered_run_artifact_hashes", not mismatch, mismatches=mismatch)
    check("all_run_files_indexed", actual_files == set(hash_index) | {"artifact_hashes.json"},
          unindexed=sorted(actual_files - set(hash_index) - {"artifact_hashes.json"}))
    source_mismatch = [p for p, expected in registration["source_sha256"].items()
                       if not (ROOT / p).is_file() or digest(ROOT / p) != expected]
    check("registered_sources_unchanged", not source_mismatch, mismatches=source_mismatch)
    config, design = registration["physical_config"], registration["design"]
    check("registered_plans_reproduce", build_preflight_plan(design, config) == registration["plans"])
    check("summary_complete_without_failure", summary["status"] == "PASS" and summary["failure"] is None)
    check("no_main_training_or_model_execution", summary["main_collection"] == "NOT_RUN"
          and summary["learning_updates"] == summary["model_forwards"] == 0
          and summary["P5_01"] == "HOLD")
    summary_labels = {row["label"]: row for row in summary["episodes"]}
    expected_labels = set()
    counts = Counter()
    dt_tolerance = config["validity"]["dt_match_tolerance_s"]

    for i in range(0, len(registration["plans"]), 2):
        p0, p1 = registration["plans"][i:i + 2]
        cell = p0["family_id"] + "-" + p0["y_condition"]
        roles = {"warmup": ("warmup", p0, p0["warmup_actions"]),
                 "S0": ("core", p0, p0["core_actions"]),
                 "S1": ("core", p1, p1["core_actions"]),
                 "replay": ("exact_replay", p1, p1["warmup_actions"] + p1["core_actions"]),
                 "half": ("dt_half", p1, p1["warmup_actions"] + p1["core_actions"])}
        episodes = {}
        for suffix, (role, plan, actions) in roles.items():
            label = cell + "-" + suffix
            expected_labels.add(label)
            folder = run / "episodes" / label
            if not (folder / "manifest.json").is_file():
                check(label + ":present", False)
                continue
            try:
                manifest = read(folder / "manifest.json")
                episode, = list(iter_dataset(folder, manifest["contract"]))
                episodes[suffix] = episode
                a, m = episode.arrays, episode.metadata
                identity = {key: m[key] for key in ("episode_id", "episode_index", "family_id",
                            "family_definition", "initial_record", "split")}
                identities.append(Episode({}, identity))
                c = m["contract"]
                check(label + ":contract_and_commands", m["split"] == "smoke-test"
                      and m["family_id"] == plan["family_id"]
                      and m["collection_policy"]["role"] == role
                      and m["collection_policy"]["predeclared_plan"] == plan
                      and m["collection_policy"]["training_eligible"] is False
                      and m["collection_policy"]["development_evaluation_eligible"] is False
                      and m["initial_record"]["initial_spec"] == plan["initial_spec"]
                      and c["resolved_config"] == (registration["half_config"] if suffix == "half" else config)
                      and np.array_equal(a["action_requested"], np.asarray(actions))
                      and np.array_equal(a["action_requested"], a["action_applied"]))
                check(label + ":physical_and_geometry_valid", bool(a["transition_valid"].all())
                      and all(g["pass"] for g in m["geometry_checks"])
                      and np.all(np.abs(a["actual_dt"] - .1) <= dt_tolerance)
                      and len(a["substep_times"]) == len(actions) * c["frame_skip"])
                srow = summary_labels.get(label, {})
                check(label + ":summary_manifest_binding", srow.get("manifest_sha256") == digest(folder / "manifest.json")
                      and srow.get("dataset") == str(folder.relative_to(run))
                      and srow.get("measurements", {}).get("transition_count") == episode.transition_count)
                counts[role] += episode.transition_count
                counts["physics_substeps"] += len(a["substep_times"])
                counts["episodes"] += 1
                episode_rows.append({"label": label, "role": role, "transitions": episode.transition_count,
                                     "physics_substeps": len(a["substep_times"]),
                                     "active_contact_transitions": int(np.count_nonzero(a["contact_summary"][:, 2] > 0))})
            except Exception as error:
                check(label + ":ordinary_episode_validation", False,
                      error=f"{type(error).__name__}: {error}")
        if len(episodes) != 5:
            continue
        warm, s0, s1, replay, half = [episodes[k] for k in ("warmup", "S0", "S1", "replay", "half")]
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
                            for key in ("action_requested", "action_applied", "contact_summary", "transition_valid"))
        head = head and s0.metadata["diagnostics"][:5] == s1.metadata["diagnostics"][:5]
        check(cell + ":same_speed_first5_exact", head)
        stops = {}
        for key, episode in (("S0", s0), ("S1", s1)):
            a = episode.arrays
            zeros = np.all(a["action_applied"] == 0, axis=1)
            valid_starts = [j for j in range(episode.transition_count - 4) if a["transition_valid"][j:j + 5].all()]
            zero_starts = [j for j in valid_starts if zeros[j:j + 5].all()]
            check(cell + ":" + key + "_K5_contract", valid_starts == list(range(16))
                  and int(zeros.sum()) == 5 and sum(int(zeros[j:j + 5].sum()) for j in valid_starts) == 25
                  and zero_starts == ([10] if key == "S1" else []))
            stops[key] = zero_blocks(episode, config)
            expected_lengths = [5] if key == "S1" else [1] * 5
            check(cell + ":" + key + "_recorded_stop_intervals",
                  [r["end_action_exclusive"] - r["start_action"] for r in stops[key]] == expected_lengths
                  and all(r["zero_requested_and_applied"] and r["finite_states"]
                          and r["substep_count"] == length * config["physics"]["frame_skip"]
                          and abs(r["actual_duration_s"] - length * .1) <= dt_tolerance
                          for r, length in zip(stops[key], expected_lengths)))
        exact = raw_exact([warm, s1], [replay])
        check(cell + ":nominal_exact_replay", all(exact.values()), fields=exact)
        nominal_states = joined([warm, s1], "state")
        times = joined([warm, s1], "sim_time")
        delta = nominal_states - half.arrays["state"]
        xy = float(np.linalg.norm(delta[:, :2], axis=1).max())
        yaw = float(np.abs((delta[:, 2] + np.pi) % (2 * np.pi) - np.pi).max())
        frac = registration["acceptance"]["numeric"]["timestep_difference_fraction_of_task_tolerance"]
        check(cell + ":dt_half_all_23_boundaries", len(nominal_states) == 23
              and np.array_equal(warm.arrays["raw_integration_state"][0], half.arrays["raw_integration_state"][0])
              and np.allclose(times, half.arrays["sim_time"], rtol=0, atol=dt_tolerance)
              and xy <= frac * config["task"]["position_tolerance_m"]
              and yaw <= frac * config["task"]["yaw_tolerance_rad"],
              max_object_xy_m=xy, max_object_yaw_rad=yaw,
              max_pusher_xy_m=float(np.linalg.norm(delta[:, 6:8], axis=1).max()),
              max_object_velocity_xy_m_s=float(np.linalg.norm(delta[:, 3:5], axis=1).max()),
              max_pusher_velocity_xy_m_s=float(np.linalg.norm(delta[:, 8:10], axis=1).max()))
        cells.append({"cell": cell, "geometry_intent": p0["geometry"], "Y": p0["y_condition"],
                      "warmup_end_measured_vy_m_s": vy, "stop_measurements": stops})
        print(json.dumps({"audited_cell": cell, "measured_vy_m_s": vy, "dt_max_xy_m": xy}), flush=True)
    try:
        assignments = validate_family_splits(identities)
        check("all_episode_family_lineage", len(assignments) == 3 and set(assignments.values()) == {"smoke-test"})
    except Exception as error:
        check("all_episode_family_lineage", False, error=f"{type(error).__name__}: {error}")
    actual_labels = {p.name for p in (run / "episodes").iterdir() if p.is_dir()}
    check("exactly_all_30_expected_episodes", expected_labels == actual_labels == set(summary_labels)
          and len(summary["episodes"]) == len(summary_labels) == 30)
    expected_counts = {"warmup": 12, "core": 240, "exact_replay": 132, "dt_half": 132,
                       "physics_substeps": 129600, "episodes": 30}
    check("saved_transition_and_substep_accounting", dict(counts) == expected_counts,
          observed=dict(counts), expected=expected_counts)
    check("execution_accounting_matches_saved_evidence", summary["counts"]["action_calls"] == 516
          and summary["counts"]["completed_intervals"] == 516
          and summary["counts"]["physics_substeps"] == counts["physics_substeps"]
          and summary["counts"]["by_role"] == {k: expected_counts[k] for k in ("warmup", "core", "exact_replay", "dt_half")})
    check("settling_separately_accounted", len(summary["settling"]) == 8
          and summary["settling_substeps_total"] == sum(r["settling_substeps"] for r in summary["settling"])
          and all(abs(r["settling_substeps"] * r["physics_dt_s"] - r["initial_record"]["settle_time_s"]) <= dt_tolerance
                  for r in summary["settling"]))
    output_bytes = sum(p.stat().st_size for p in run.rglob("*") if p.is_file())
    free = shutil.disk_usage(run).free
    check("actual_output_and_free_reserve", output_bytes <= design["resources"]["preflight_output_limit_MiB"] * 1024**2
          and free >= design["resources"]["minimum_free_reserve_GiB"] * 1024**3,
          output_bytes=output_bytes, free_bytes=free)
    return {"schema": "p4-data-preflight-independent-review-v1",
            "status": "PASS" if all(c["pass"] for c in checks) else "FAIL",
            "run": str(run.relative_to(ROOT)), "registration_sha256": digest(run / "registration.json"),
            "summary_sha256": digest(run / "summary.json"), "audit_source_sha256": digest(__file__),
            "check_count": len(checks), "checks": checks, "cells": cells, "episodes": episode_rows,
            "execution_during_review": {"physical_action_intervals": 0, "model_forwards": 0, "learning_updates": 0,
                "snapshot_reconstruction": "ordinary episode reader uses mj_forward without mj_step"},
            "scope": "Saved evidence consistency, collector manipulation and six representative physical comparisons; no training effect, alignment success or general physical validity claim"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (args.run_dir / "summary.json").is_file() or not (args.run_dir / "artifact_hashes.json").is_file():
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
