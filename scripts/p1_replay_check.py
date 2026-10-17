"""P1-05: measure repeated transitions from identical snapshots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib
import numpy as np

from push_align import InitialSpec, PlanarPushEnv
from push_align.config import ROOT
from push_align.environment import task_angle_error


def wrap(value: float) -> float:
    return float((value + np.pi) % (2 * np.pi) - np.pi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (ROOT / "configs" / "acceptance.v0.toml").open("rb") as file:
        acceptance = tomllib.load(file)
    env = PlanarPushEnv()
    outcomes = []
    for yaw in np.deg2rad(acceptance["physics_suite"]["initial_yaw_candidates_deg"]):
        env.reset(InitialSpec(object_yaw=float(yaw)))
        snap = env.snapshot()
        repeated = []
        for _ in range(acceptance["physics_suite"]["replays_per_snapshot"]):
            env.restore(snap)
            repeated.append(env.step(np.array([0.03 * np.cos(yaw), 0.03 * np.sin(yaw)])).observation.state)
        stacked = np.asarray(repeated)
        max_abs = np.max(np.abs(stacked - stacked[0]), axis=0)
        outcomes.append({"initial_yaw_rad": float(yaw), "replays": len(repeated), "max_abs_state_difference": max_abs.tolist()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output_file:
        output_file.write(json.dumps({
        "run_kind": "P1-05 snapshot replay",
        "scope": "legacy one-transition smoke only; full P1-05 acceptance uses p1_physics_pilot.py",
        "full_P1_05_verdict": "NOT_ASSESSED",
        "config_id": env.config["config_id"],
        "replay_cases": outcomes,
        "angle_formula_checks": {
            "wrap_pi_minus_epsilon": wrap(np.pi - 1e-12),
            "wrap_minus_pi_plus_epsilon": wrap(-np.pi + 1e-12),
            "symmetric_pi_error_rad": float(task_angle_error(np.pi)),
            "symmetric_pi_check": abs(float(task_angle_error(np.pi))) <= 1e-10,
        },
        }, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(args.output)

if __name__ == "__main__": main()
