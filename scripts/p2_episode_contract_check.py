"""P2-01 storage/split/window proof with small, predeclared physical traces.

This is contract verification data, not the P2-02 learning dataset or task
performance evaluation. The workspace-edge case deliberately creates a
partial interval to verify preservation and exclusion.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

import mujoco
import numpy as np

from push_align import InitialSpec, PlanarPushEnv
from push_align.config import ROOT
from push_align.data import (EpisodeRecorder, contract_hash, digest, load_dataset,
                             make_contract, sequence_windows, snapshot_from_episode,
                             split_families, validate_episode, write_dataset)


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, allow_nan=False, indent=2)
        file.write("\n")


def source_bundle(out):
    paths = [p for folder in ("src", "scripts", "tests") for p in (ROOT/folder).rglob("*.py")]
    paths += list((ROOT/"configs").glob("*.toml"))
    paths += [ROOT/"assets/push_scene.xml", ROOT/"requirements.lock.txt",
              ROOT/"docs/05_구현계약과_설정명세.md", ROOT/"docs/06_단계별_완료판정.md"]
    hashes = {}
    for path in paths:
        rel = path.relative_to(ROOT); target = out/"source"/rel
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
        hashes[str(rel)] = digest(path)
    return hashes


def generation_regression(out, env):
    """Compare physical outputs against archived pre-generation environment."""
    path = ROOT/"runs/p1-06-adoption-20261007-08/source/src/push_align/environment.py"
    expected_sha = "41ad48083f7ffb0d37e98aef690bf44672067599f3bc1cbe12c4885663cd9bfd"
    if digest(path) != expected_sha:
        raise ValueError("Archived P1 environment reference changed")
    reference = out/"reference_environment.py"; shutil.copyfile(path, reference)
    name = "push_align._p2_reference_environment"
    spec = importlib.util.spec_from_file_location(name, reference)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    old = module.PlanarPushEnv()
    env.reset(); old.reset()
    current = [env.step([.03, 0.]) for _ in range(10)]
    previous = [old.step([.03, 0.]) for _ in range(10)]
    state_a, state_b = np.concatenate([r.substep_states for r in current]), np.concatenate([r.substep_states for r in previous])
    time_a, time_b = np.concatenate([r.substep_times for r in current]), np.concatenate([r.substep_times for r in previous])
    raw_a, raw_b = env.snapshot().integration_state, old.snapshot().integration_state
    np.savez_compressed(out/"generation_physics_regression.npz", current_states=state_a, reference_states=state_b,
                        current_times=time_a, reference_times=time_b, current_integration=raw_a, reference_integration=raw_b)
    return {"reference_source_sha256": expected_sha, "microstep_count": len(state_a),
            "same_microstep_states": bool(np.array_equal(state_a, state_b)),
            "same_microstep_times": bool(np.array_equal(time_a, time_b)),
            "same_final_integration_state": bool(np.array_equal(raw_a, raw_b)),
            "scope": "one predeclared 1-second central push; lifecycle token regression, not new task evaluation"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    sources = source_bundle(out)
    adoption = json.loads((ROOT/"reports/p1_environment_adoption.json").read_text())
    env = PlanarPushEnv(); contract = make_contract(env)
    prerequisites = {
        "adoption_pass": adoption["status"] == "PASS",
        "pilot_allowed": adoption["pilot_data_collection_allowed"] is True,
        "config_matches_adoption": contract["config_sha256"] == adoption["config_sha256"],
        "profile_matches_adoption": contract["profile_sha256"] == adoption["acceptance_sha256"],
        "environment_matches_adoption": contract["environment_hash"] == adoption["environment_hash"],
    }
    plans = [
        {"episode_id": "episode-000", "family_id": "family-000", "initial": {"object_x": -.01}, "mode": "central", "ticks": 10},
        {"episode_id": "episode-001", "family_id": "family-000", "initial": {"object_x": -.01}, "mode": "central", "ticks": 10},
        {"episode_id": "episode-002", "family_id": "family-001", "initial": {"object_yaw": float(np.deg2rad(5.))}, "mode": "central", "ticks": 10},
        {"episode_id": "episode-003", "family_id": "family-002", "initial": {"object_x": .01, "object_yaw": float(np.deg2rad(-5.))}, "mode": "clipped_noncontact", "ticks": 10},
        {"episode_id": "episode-004", "family_id": "family-003", "initial": {"object_x": .14}, "mode": "partial_contract_fixture", "ticks": 10},
    ]
    seed = 20261007
    assignments = split_families(sorted({p["family_id"] for p in plans}), {"train": 2, "validation": 1, "smoke-test": 1}, seed)
    for index, plan in enumerate(plans):
        plan.update({"episode_index": index, "split": assignments[plan["family_id"]]})
    save(out/"manifest.json", {"created_at": datetime.now().astimezone().isoformat(), "task_id": "P2-01",
                              "argv": sys.argv, "executable": sys.executable, "python": sys.version,
                              "platform": platform.platform(), "versions": contract["versions"],
                              "purpose": "contract-check", "contract": contract, "contract_hash": contract_hash(contract),
                              "predeclared_episodes": plans, "split_seed": seed, "family_assignments": assignments,
                              "source_hashes": sources, "p1_adoption_sha256": digest(ROOT/"reports/p1_environment_adoption.json"),
                              "failure_fixture_scope": "x=0.14 m outside nominal initial range, deliberately approaches workspace boundary; not main data or field failure evidence"})
    if not all(prerequisites.values()):
        save(out/"summary.json", {"stage_verdict": "FAIL", "prerequisites": prerequisites})
        raise SystemExit(1)
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                           cwd=ROOT, text=True, capture_output=True)
    (out/"tests.txt").write_text(tests.stdout+tests.stderr, encoding="utf-8")
    count = re.search(r"Ran (\d+) tests", tests.stdout+tests.stderr)
    episodes = []
    for plan in plans:
        initial = InitialSpec(**plan["initial"]); env.reset(initial)
        family_definition = {"base_initial_spec": asdict(initial), "variant_rule": "exact same reset specification; repeated actions remain in one family"}
        recorder = EpisodeRecorder(env, episode_id=plan["episode_id"], episode_index=plan["episode_index"],
                                   family_id=plan["family_id"], family_definition=family_definition,
                                   split=plan["split"], collection_policy={"kind": plan["mode"], "planned_ticks": plan["ticks"], "purpose": "contract-check"},
                                   seed=seed, contract=contract)
        yaw = initial.object_yaw
        for tick in range(plan["ticks"]):
            if plan["mode"] == "central": action = .03*np.array([np.cos(yaw), np.sin(yaw)])
            elif plan["mode"] == "clipped_noncontact": action = np.array([-.09, .09]) if tick == 0 else np.array([-.02, 0.])
            else: action = np.array([.06, 0.])
            recorder.step(action)
            if recorder.invalid: break
        episodes.append(recorder.finish())
    dataset = out/"data"/"p2-contract-check"
    written = write_dataset(dataset, episodes, purpose="contract-check", split_seed=seed)
    loaded, manifest = load_dataset(dataset, contract)
    reviews = []
    roundtrip = True
    for original, restored in zip(episodes, loaded):
        roundtrip &= set(original.arrays) == set(restored.arrays) and all(np.array_equal(original.arrays[k], restored.arrays[k], equal_nan=True) for k in original.arrays)
        restored_ticks = []
        for i in range(restored.transition_count):
            if not restored.arrays["transition_valid"][i]: continue
            env.restore(snapshot_from_episode(restored, i))
            result = env.step(restored.arrays["action_requested"][i])
            equal = np.array_equal(result.observation.state, restored.arrays["state"][i+1]) and np.array_equal(env.snapshot().integration_state, restored.arrays["raw_integration_state"][i+1])
            lo, hi = restored.arrays["substep_offsets"][i:i+2]
            equal &= np.array_equal(result.substep_states, restored.arrays["substep_states"][lo:hi])
            restored_ticks.append(bool(equal))
        review = validate_episode(restored, contract)
        review.update({"episode_id": restored.metadata["episode_id"], "family_id": restored.metadata["family_id"], "split": restored.metadata["split"],
                       "sequence_windows": {str(k): len(sequence_windows(restored, k, contract)) for k in (1, 5, 10)},
                       "replayed_valid_intervals": len(restored_ticks), "all_replayed_intervals_match": all(restored_ticks),
                       "last_actual_dt_s": float(restored.arrays["actual_dt"][-1]),
                       "terminal_raw_retained": bool(len(restored.arrays["state"]) == restored.transition_count+1)})
        reviews.append(review)
    generation = generation_regression(out, env)
    unchanged = all(digest(ROOT/path) == sha for path, sha in sources.items())
    checks = {**prerequisites, "regression_tests": tests.returncode == 0,
              "all_declared_episodes_recorded": len(loaded) == len(plans),
              "npz_raw_roundtrip": bool(roundtrip), "dataset_manifest_roundtrip": written == manifest,
              "all_valid_intervals_replay": all(r["all_replayed_intervals_match"] for r in reviews),
              "partial_interval_preserved_excluded": reviews[-1]["excluded_transitions"] == 1 and reviews[-1]["last_actual_dt_s"] < contract["action_dt_s"],
              "generation_does_not_change_physics": all(generation[k] for k in ("same_microstep_states", "same_microstep_times", "same_final_integration_state")),
              "sources_unchanged_during_run": unchanged}
    summary = {"task_id": "P2-01", "stage_verdict": "PASS" if all(checks.values()) else "FAIL",
               "checks": checks, "tests_run": int(count[1]) if count else None,
               "dataset_manifest": str(dataset.relative_to(out)/"manifest.json"),
               "episode_count": manifest["episode_count"], "family_count": len(assignments),
               "transition_count": manifest["transition_count"], "learning_transition_count": manifest["learning_transition_count"],
               "excluded_transition_count": manifest["excluded_transition_count"],
               "episode_reviews": reviews, "generation_regression": generation,
               "alignment_outcome": "NOT_EVALUATED", "learner_understanding": "NOT_ASSESSED",
               "P2-02": "NOT_RUN", "dynamics_training": "NOT_RUN", "MPC": "NOT_RUN",
               "scope": "episode/split/window contracts only; deliberate boundary fixture is not nominal data coverage or task performance"}
    save(out/"summary.json", summary)
    artifacts = {str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()}
    save(out/"artifact_hashes.json", {"sha256": artifacts})
    print(json.dumps({"run": str(out), "verdict": summary["stage_verdict"], "episodes": len(loaded), "transitions": manifest["transition_count"], "excluded": manifest["excluded_transition_count"], "tests": summary["tests_run"]}, ensure_ascii=False))
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
