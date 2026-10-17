"""P1-04: paired planar-pushing contrast experiments.

This script records physical outcomes. It does not train a model or claim
industrial performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib

import numpy as np

from push_align import InitialSpec, PlanarPushEnv
from push_align.config import ROOT


def angle_delta(end: float, start: float) -> float:
    return float((end - start + np.pi) % (2 * np.pi) - np.pi)


def condition_initial(yaw: float, condition: str, env: PlanarPushEnv, offset_abs: float) -> InitialSpec:
    half_x = env.config["object"]["size_full_xyz_m"][0] / 2
    radius = env.config["pusher"]["radius_m"]
    gap = env.config["reset"]["pusher_surface_gap_m"]
    tangent = np.array([-np.sin(yaw), np.cos(yaw)])
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    offset = offset_abs if condition == "positive_offset" else -offset_abs if condition == "negative_offset" else 0.0
    if condition == "noncontact":
        offset = 0.08
    pusher_xy = -forward * (half_x + radius + gap) + tangent * offset
    return InitialSpec(object_yaw=yaw, pusher_x=float(pusher_xy[0]), pusher_y=float(pusher_xy[1]))


def run_condition(env: PlanarPushEnv, yaw: float, condition: str, settings: dict) -> dict:
    initial = env.reset(condition_initial(yaw, condition, env, settings["probe_offset_abs_m"]))
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    action = np.zeros(2) if condition == "zero" else forward * settings["probe_command_speed_m_s"]
    contacts, trace, transitions = [], [], []
    steps = round(settings['observation_duration_s'] / env.action_dt)
    if abs(steps * env.action_dt - settings['observation_duration_s']) > 1e-9:
        raise ValueError('Probe duration must be a multiple of action_dt')
    for _ in range(steps):
        result = env.step(action)
        contacts.extend(result.diagnostics["contact_samples"])
        trace.append(result.observation.state.tolist())
        transitions.append({'actual_dt': result.actual_dt, 'transition_valid': result.transition_valid,
                            'violations': result.diagnostics['validity_violations']})
        if not result.transition_valid:
            break
    final = env._observation().state
    return {
        "condition": condition,
        "initial_yaw_rad": yaw,
        "action_m_s": action.tolist(),
        "initial_state": initial.state.tolist(),
        "final_state": final.tolist(),
        "delta_object_xy_m": (final[:2] - initial.state[:2]).tolist(),
        "delta_object_yaw_rad": angle_delta(final[2], initial.state[2]),
        "contact_sample_count": sum(item["pusher_object_contact"] for item in contacts),
        "max_normal_force_N": max((item["normal_force_N"] for item in contacts), default=0.0),
        "transition_valid": len(transitions) == steps and all(t['transition_valid'] for t in transitions),
        "transitions": transitions,
        "actual_duration_s": sum(t['actual_dt'] for t in transitions),
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    env = PlanarPushEnv()
    with (ROOT / "configs" / "acceptance.v0.toml").open("rb") as file:
        suite = tomllib.load(file)["physics_suite"]
    cases = []
    for yaw in np.deg2rad(suite["initial_yaw_candidates_deg"]):
        for condition in suite["conditions"]:
            cases.append(run_condition(env, float(yaw), condition, suite))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output_file:
        output_file.write(json.dumps({
        "run_kind": "P1-04 physical contrast probe",
        "config_id": env.config["config_id"],
        "conditions": suite["conditions"],
        "duration_s": suite["observation_duration_s"],
        "cases": cases,
        }, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__": main()
