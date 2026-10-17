"""Collect and audit a small, separated, bounded dynamics pilot dataset."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import gzip
import json
from pathlib import Path
import platform
import re
import resource
import shutil
import subprocess
import sys
from time import perf_counter
import tomllib
import traceback

import numpy as np

from push_align import InitialSpec, PlanarPushEnv
from push_align.collection import MODES, build_collection_plan
from push_align.config import ROOT
from push_align.contact_geometry import check_contact_geometry
from push_align.coverage import REGIMES, episode_coverage
from push_align.data import (DatasetWriter, EpisodeRecorder, _encode_nonfinite,
                             contract_hash, digest, iter_dataset, make_contract)
from push_align.environment import wrap_angle
from p2_episode_contract_check import source_bundle


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as file:
        json.dump(_encode_nonfinite(value), file, ensure_ascii=False, allow_nan=False, indent=2)
        file.write("\n")


def save_trace(folder, label, initial_snapshot, results):
    states = np.array([r.observation.state for r in results], dtype=np.float64).reshape(-1, 10)
    paths, values = [], []
    diagnostics = _encode_nonfinite([r.diagnostics for r in results], paths=paths, values=values)
    np.savez_compressed(folder/f"{label}.npz", state=states,
                        actual_dt=np.array([r.actual_dt for r in results]),
                        action_requested=np.array([r.requested_action for r in results]).reshape(-1, 2),
                        action_applied=np.array([r.applied_action for r in results]).reshape(-1, 2),
                        substep_states=np.concatenate([r.substep_states for r in results]) if results else np.empty((0, 10)),
                        substep_times=np.concatenate([r.substep_times for r in results]) if results else np.empty(0),
                        initial_integration_state=initial_snapshot.integration_state,
                        diagnostic_nonfinite_values=np.array(values, dtype=np.float64))
    with gzip.open(folder/f"{label}.json.gz", "xt", encoding="utf-8") as file:
        json.dump({"initial_snapshot": _encode_nonfinite(asdict(initial_snapshot)), "diagnostics": diagnostics,
                   "diagnostic_nonfinite_paths": paths}, file, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def compare_physics_extension(out, plans, config, contract):
    folder = out/"physics_extension"; folder.mkdir()
    finer_config = deepcopy(config)
    finer_config["physics"]["timestep_s"] *= .5
    finer_config["physics"]["frame_skip"] *= 2
    base, fine = PlanarPushEnv(config), PlanarPushEnv(finer_config)
    profile = tomllib.loads(contract["profile_toml"])
    fraction = profile["numeric"]["timestep_difference_fraction_of_task_tolerance"]
    budgets = {"position_m": fraction*config["task"]["position_tolerance_m"],
               "yaw_rad": fraction*config["task"]["yaw_tolerance_rad"]}
    rows = []
    for mode in MODES:
        plan = next(p for p in plans if p["mode"] == mode)
        base.reset(InitialSpec(**plan["initial_spec"]))
        fine.import_dt_comparison_state(base); base.import_dt_comparison_state(base)
        starts, snapshots, results, curves, geometry = [], [], [], [], []
        for label, env in (("base", base), ("half", fine)):
            starts.append(env.observe())
            snap = env.snapshot(); snapshots.append(snap); rs = []
            for action in plan["actions"]:
                r = env.step(action); rs.append(r)
                if not r.transition_valid: break
            save_trace(folder, f"{mode}_{label}", snap, rs)
            results.append(rs)
            curves.append(np.vstack([starts[-1].state, *[r.observation.state for r in rs]]))
            geometry.append([check_contact_geometry(r.diagnostics["samples"], contract["penetration_budget_m"]) for r in rs])
        complete = all(len(rs) == len(plan["actions"]) and all(r.transition_valid for r in rs) for rs in results)
        initial_match = bool(np.array_equal(starts[0].state, starts[1].state) and starts[0].sim_time == starts[1].sim_time and np.array_equal(snapshots[0].integration_state, snapshots[1].integration_state))
        aligned = complete and initial_match and np.allclose([r.observation.sim_time for r in results[0]], [r.observation.sim_time for r in results[1]], rtol=0, atol=contract["dt_tolerance_s"])
        xy = float(np.max(np.linalg.norm(curves[0][:, :2]-curves[1][:, :2], axis=1))) if aligned else None
        yaw = float(np.max(np.abs(wrap_angle(curves[0][:, 2]-curves[1][:, 2])))) if aligned else None
        geometry_pass = all(g["pass"] for gs in geometry for g in gs)
        row = {"mode": mode, "episode_id": plan["episode_id"], "complete_both": complete,
               "identical_initial_observation_time_and_integration_state": initial_match, "same_action_boundaries": bool(aligned),
               "max_position_difference_m": xy, "max_yaw_difference_rad": yaw,
               "geometry_checks": geometry, "budgets": budgets,
               "pass": bool(aligned and geometry_pass and xy <= budgets["position_m"] and yaw <= budgets["yaw_rad"])}
        rows.append(row)
    result = {"cases": rows, "pass": all(r["pass"] for r in rows), "half_config": finer_config,
              "scope": "four predeclared 1.6-second sequences; no claim for all collected trajectories or continuous state/action space"}
    save(folder/"summary.json", result)
    return result


def aggregate_coverage(rows, settings):
    splits = {}
    for split in ("train", "validation", "smoke-test"):
        part = [r for r in rows if r["split"] == split]
        windows = {}
        for h in settings["coverage"]["horizons"]:
            windows[str(h)] = {"total": sum(r["windows"][str(h)]["total"] for r in part),
                              "by_first_regime": {g: sum(r["windows"][str(h)]["by_first_regime"][g] for r in part) for g in REGIMES},
                              "contributing_families": {g: len({r["family_id"] for r in part if r["windows"][str(h)]["by_first_regime"][g]}) for g in REGIMES}}
        splits[split] = {"episodes": len(part), "families": len({r["family_id"] for r in part}),
                         "modes": {m: sum(r["mode"] == m for r in part) for m in MODES},
                         "recorded_transitions": sum(r["recorded_transitions"] for r in part),
                         "valid_transitions": sum(r["valid_transitions"] for r in part),
                         "excluded_transitions": sum(r["excluded_transitions"] for r in part),
                         "regimes_all": {g: sum(r["regime_counts_all"][g] for r in part) for g in REGIMES},
                         "regimes_valid": {g: sum(r["regime_counts_valid"][g] for r in part) for g in REGIMES},
                         "windows": windows,
                         "translation_signal_episodes": sum(r["translation_signal"] for r in part),
                         "rotation_signal_episodes": sum(r["rotation_signal"] for r in part),
                         "qualified_recontact_episodes": sum(r["mode"] == "contact_transition" and r["qualified_recontact_count"] > 0 for r in part)}
    return {"splits": splits, "episodes": rows,
            "scope": "pilot coverage, not statistical power; windows overlap; contact regimes are measured and modes are prescribed"}


def validate_runner_settings(settings, contract):
    c = settings["coverage"]; cfg = contract["resolved_config"]; p = tomllib.loads(contract["profile_toml"])
    if c["horizons"] != [1, 5, 10] or c["required_validation_horizon"] != 5 or c["required_validation_regimes"] != list(REGIMES):
        raise ValueError("v1 requires 1/5/10 reporting and all three K=5 validation regimes")
    if c["require_signal_per_split"] is not True or c["require_qualified_recontact_per_split"] is not True:
        raise ValueError("Do not silently remove registered coverage checks")
    if c["window_regime"] != "first_transition_only" or c["active_contact_rule"] != "pusher_object normal_force strictly greater than environment threshold; start plus every substep":
        raise ValueError("Declared regime definition differs from implemented contract")
    fraction = p["numeric"]["signal_min_fraction_of_task_tolerance"]
    if not np.isclose(c["position_signal_min_m"], fraction*cfg["task"]["position_tolerance_m"], rtol=0, atol=1e-15) or not np.isclose(c["yaw_signal_min_rad"], fraction*cfg["task"]["yaw_tolerance_rad"], rtol=0, atol=1e-15):
        raise ValueError("Signal budgets differ from registered task-based rule")
    if settings["physics_extension"]["physics_dt_scale"] != .5 or settings["physics_extension"]["representative_rule"] != "lowest_episode_id_per_mode":
        raise ValueError("Unsupported physics extension rule")
    if settings["physics_extension"]["budgets"] != "unchanged acceptance-v1 dt position/yaw budgets" or settings["physics_extension"]["comparison"] != "identical_stabilized_initial_snapshot; both warmstarts zero; max difference at all shared action boundaries":
        raise ValueError("Declared comparison differs from implemented physics check")


def run(args, out):
    started = perf_counter()
    settings = tomllib.loads(args.collection_config.read_text(encoding="utf-8"))
    env = PlanarPushEnv(); contract = make_contract(env)
    plans, assignments = build_collection_plan(settings, env.config)
    validate_runner_settings(settings, contract)
    sources = source_bundle(out); shutil.copyfile(args.collection_config, out/"collection.toml")
    adoption_path = ROOT/"reports/p1_environment_adoption.json"
    adoption = json.loads(adoption_path.read_text())
    prerequisites = {"P1_pilot_adopted": adoption["status"] == "PASS" and adoption["pilot_data_collection_allowed"],
                     "config_matches_adoption": contract["config_sha256"] == adoption["config_sha256"],
                     "profile_matches_adoption": contract["profile_sha256"] == adoption["acceptance_sha256"],
                     "environment_matches_adoption": contract["environment_hash"] == adoption["environment_hash"]}
    save(out/"manifest.json", {"task_id": "P2-02", "created_at": datetime.now().astimezone().isoformat(),
                              "argv": sys.argv, "executable": sys.executable, "python": sys.version, "platform": platform.platform(),
                              "contract": contract, "contract_hash": contract_hash(contract), "collection_settings": settings,
                              "collection_config_sha256": digest(args.collection_config), "predeclared_episodes": plans,
                              "family_assignments": assignments, "source_hashes": sources, "P1_adoption_sha256": digest(adoption_path)})
    if not all(prerequisites.values()): raise ValueError(f"P1 prerequisite mismatch: {prerequisites}")
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT, text=True, capture_output=True)
    (out/"tests.txt").write_text(tests.stdout+tests.stderr, encoding="utf-8")
    count = re.search(r"Ran (\d+) tests", tests.stdout+tests.stderr)
    if tests.returncode: raise ValueError("Regression tests failed; full collection not started")
    extension = compare_physics_extension(out, plans, env.config, contract)
    if not extension["pass"]:
        save(out/"summary.json", {"task_id": "P2-02", "stage_verdict": "FAIL", "prerequisites": prerequisites,
                                  "physics_extension": extension, "planned_episodes": len(plans), "not_started_episodes": len(plans),
                                  "alignment_outcome": "NOT_EVALUATED", "learner_understanding": "NOT_ASSESSED"})
        return False
    dataset = out/"data"/"p2-pilot-v1"
    writer = DatasetWriter(dataset, contract, purpose="pilot", split_seed=settings["split_seed"])
    ledger = []; step_times = []; collection_start = perf_counter()
    for plan in plans:
        clock = perf_counter(); recorder = None
        try:
            env.reset(InitialSpec(**plan["initial_spec"]))
            recorder = EpisodeRecorder(env, episode_id=plan["episode_id"], episode_index=plan["episode_index"],
                                       family_id=plan["family_id"], family_definition=plan["family_definition"],
                                       split=plan["split"], collection_policy={"kind": plan["mode"], "purpose": "pilot",
                                       "collection_id": settings["collection_id"], "planned_actions": plan["actions"],
                                       "phases": plan["phases"], "pusher_tangent_offset_m": plan["pusher_tangent_offset_m"],
                                       "action_rng_key": plan["action_rng_key"]}, seed=settings["initial_seed"], contract=contract)
            for action in plan["actions"]:
                tick = perf_counter(); recorder.step(action); step_times.append(perf_counter()-tick)
                if recorder.invalid: break
            episode = recorder.finish(); writer.append(episode)
            ledger.append({"episode_id": plan["episode_id"], "status": "valid_complete" if not recorder.invalid else "physical_or_geometry_failure",
                           "terminal_reason": episode.metadata["terminal_reason"], "recorded_transitions": episode.transition_count,
                           "valid_transitions": int(episode.arrays["transition_valid"].sum()), "wall_time_s": perf_counter()-clock})
            del episode, recorder
        except Exception as error:
            folder = out/"collection_errors"/plan["episode_id"]; folder.mkdir(parents=True, exist_ok=False)
            save(folder/"failure.json", {"plan": plan, "exception": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
            if recorder is not None:
                save_trace(folder, "recorded_prefix", recorder.snapshots[0], recorder.results)
                np.savez_compressed(folder/"last_raw_state.npz", integration_state=env.snapshot().integration_state)
            ledger.append({"episode_id": plan["episode_id"], "status": "collection_error", "message": str(error), "wall_time_s": perf_counter()-clock})
        if len(ledger) % 4 == 0:
            print(json.dumps({"progress": "collection", "attempted_episodes": len(ledger), "planned": len(plans)}), flush=True)
    collection_time = perf_counter()-collection_start
    save(out/"collection_ledger.json", ledger)
    if any(r["status"] == "collection_error" for r in ledger):
        save(out/"incomplete_dataset.json", {"entries": writer.entries, "reason": "collection errors; no completed dataset manifest"})
        raise ValueError("Collection errors retained; dataset audit cannot be completed")
    dataset_manifest = writer.finish()
    audit_start = perf_counter(); rows, support = [], []
    for episode in iter_dataset(dataset, contract):
        row = episode_coverage(episode, contract, position_signal_m=settings["coverage"]["position_signal_min_m"],
                               yaw_signal_rad=settings["coverage"]["yaw_signal_min_rad"], horizons=settings["coverage"]["horizons"])
        rows.append(row)
        s = episode.arrays["state"][:-1]; a = episode.arrays["action_applied"]; valid = episode.arrays["transition_valid"]
        support.append(np.column_stack([s[:, 6:8]-s[:, :2], s[:, 2], s[:, 3:6], s[:, 8:10], a])[valid])
        plan = plans[episode.metadata["episode_index"]]
        if episode.metadata["episode_id"] != plan["episode_id"] or episode.metadata["split"] != plan["split"] or episode.metadata["collection_policy"]["kind"] != plan["mode"]:
            raise ValueError("Stored episode differs from predeclared plan")
        t = episode.transition_count
        if not np.array_equal(episode.arrays["action_requested"], np.asarray(plan["actions"][:t])) or episode.metadata["initial_record"]["initial_spec"] != plan["initial_spec"]:
            raise ValueError("Applied recording is not connected to registered initial/action sequence")
        if len(rows) % 8 == 0: print(json.dumps({"progress": "audit", "episodes": len(rows)}), flush=True)
    audit_time = perf_counter()-audit_start
    coverage = aggregate_coverage(rows, settings); save(out/"coverage.json", coverage)
    features = np.concatenate(support)
    support_report = {"fields": ["relative_pusher_x", "relative_pusher_y", "obj_yaw", "obj_vx", "obj_vy", "obj_wz", "pusher_vx", "pusher_vy", "applied_vx", "applied_vy"],
                      "min": features.min(axis=0).tolist(), "max": features.max(axis=0).tolist(),
                      "command_norm_max_m_s": float(np.max(np.linalg.norm(features[:, -2:], axis=1))),
                      "scope": "observed raw input support; not a learned encoder, full 2D coverage, or extrapolation guarantee"}
    save(out/"input_support.json", support_report)
    required = settings["coverage"]["required_validation_regimes"]
    validation = coverage["splits"]["validation"]["windows"]["5"]["by_first_regime"]
    checks = {**prerequisites, "regression_tests": True, "representative_physics_extension": extension["pass"],
              "planned_and_actual_episodes_match": len(ledger) == len(rows) == len(plans) == dataset_manifest["episode_count"],
              "all_split_counts_match": all(coverage["splits"][s]["episodes"] == n for s, n in settings["split_counts"].items()),
              "no_collection_errors": all(r["status"] != "collection_error" for r in ledger),
              "all_files_contract_checked": len(rows) == len(plans),
              "validation_three_K5_regimes": all(validation[g] > 0 for g in required),
              "translation_and_rotation_signal_per_split": all(r["translation_signal_episodes"] > 0 and r["rotation_signal_episodes"] > 0 for r in coverage["splits"].values()),
              "qualified_recontact_per_split": all(r["qualified_recontact_episodes"] > 0 for r in coverage["splits"].values()),
              "source_files_unchanged_during_run": all(digest(ROOT/path) == sha for path, sha in sources.items())}
    memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    resources = {"collection_wall_time_s": collection_time, "read_and_coverage_audit_wall_time_s": audit_time,
                 "total_wall_time_s": perf_counter()-started,
                 "step_with_raw_recording_median_ms": float(np.median(step_times)*1000),
                 "step_with_raw_recording_p95_ms": float(np.percentile(step_times, 95)*1000),
                 "peak_self_RSS_MiB": float(memory/(1024**2 if sys.platform == "darwin" else 1024)),
                 "dataset_bytes": sum(p.stat().st_size for p in dataset.rglob("*") if p.is_file()),
                 "scope": "collection and raw storage/audit; test subprocess memory excluded; no dynamics training or MPC latency measured"}
    save(out/"resources.json", resources)
    summary = {"task_id": "P2-02", "stage_verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
               "tests_run": int(count[1]) if count else None, "planned_episodes": len(plans),
               "valid_complete_episodes": sum(r["status"] == "valid_complete" for r in ledger),
               "physical_or_geometry_failure_episodes": sum(r["status"] == "physical_or_geometry_failure" for r in ledger),
               "collection_error_episodes": sum(r["status"] == "collection_error" for r in ledger),
               "recorded_transitions": dataset_manifest["transition_count"], "valid_transitions": dataset_manifest["learning_transition_count"],
               "excluded_transitions": dataset_manifest["excluded_transition_count"],
               "dataset_manifest": str(dataset.relative_to(out)/"manifest.json"),
               "dataset_manifest_sha256": digest(dataset/"manifest.json"), "coverage_by_split": coverage["splits"],
               "physics_extension": {"pass": extension["pass"], "cases": [{k: v for k, v in r.items() if k != "geometry_checks"} for r in extension["cases"]]},
               "resources": resources, "alignment_outcome": "NOT_EVALUATED", "learner_understanding": "NOT_ASSESSED",
               "P2-03": "NOT_RUN", "dynamics_training": "NOT_RUN", "MPC": "NOT_RUN",
               "scope": "small model-connection data only; balanced sampling is not field frequency, power, generalization or task performance"}
    save(out/"summary.json", summary)
    print(json.dumps({"verdict": summary["stage_verdict"], "episodes": len(rows), "transitions": summary["recorded_transitions"],
                      "excluded": summary["excluded_transitions"], "validation_K5": validation, "tests": summary["tests_run"]}), flush=True)
    return all(checks.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--collection-config", type=Path, default=ROOT/"configs/p2_collection.v1.toml")
    args = parser.parse_args(); out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    try:
        passed = run(args, out)
    except Exception as error:
        save(out/"failure.json", {"exception": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        if not (out/"summary.json").exists():
            save(out/"summary.json", {"task_id": "P2-02", "stage_verdict": "ERROR", "reason": str(error), "alignment_outcome": "NOT_EVALUATED", "learner_understanding": "NOT_ASSESSED"})
        passed = False
        print(json.dumps({"verdict": "ERROR", "reason": str(error)}), flush=True)
    hashes = {str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()}
    save(out/"artifact_hashes.json", {"sha256": hashes})
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__": main()
