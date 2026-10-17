"""Predeclared bounded action sequences; no learned/feedback policy."""
from __future__ import annotations

from dataclasses import asdict
import numpy as np

from .data import DataContractError, split_families
from .environment import InitialSpec

MODES = ("noncontact", "central", "eccentric", "contact_transition")


def _require(condition, message):
    if not condition:
        raise DataContractError(message)


def build_collection_plan(settings, environment_config):
    try:
        return _build_collection_plan(settings, environment_config)
    except (KeyError, TypeError, IndexError) as error:
        raise DataContractError(f"Malformed collection configuration: {error}") from error


def _build_collection_plan(settings, environment_config):
    """Split lineage before assigning policies or observing physical results."""
    _require(settings["purpose"] == "pilot" and settings["modes"] == list(MODES), "Unsupported pilot collection")
    _require(settings["actions_per_episode"] == 16, "v1 templates require 16 actions")
    counts = settings["split_counts"]
    _require(set(counts) == {"train", "validation", "smoke-test"} and all(type(n) is int and n > 0 and n % 4 == 0 for n in counts.values()), "Positive split counts divisible by four required")
    c, init = settings["commands"], environment_config["initial_conditions"]
    fixed = {
        "initial_sampling": "independent_uniform_within_nominal_xy_yaw_ranges",
        "family_rule": "one independently generated reset per family; any future repeats/variants retain that family",
        "mode_assignment": "after_global_family_split_before_physics; sorted_ids_cycle_four_modes_within_each_split",
    }
    _require(all(settings[k] == v for k, v in fixed.items()), "Unsupported declared generation policy")
    command_fixed = {
        "blocks": "four_steps_per_speed_draw; constant_world_commands; no feedback or learned controller",
        "eccentric_sign_rule": "alternate_within_each_split",
        "noncontact_phases": ["zero:2", "retreat:4", "lateral:4", "partial_return:4", "zero:2"],
        "contact_transition_phases": ["forward:5", "retreat:3", "reapproach:8"],
    }
    _require(all(c[k] == v for k, v in command_fixed.items()), "Declared action policy differs from implemented v1")
    for key in ("forward_speed_min_m_s", "forward_speed_max_m_s", "retreat_speed_m_s", "noncontact_lateral_speed_m_s", "noncontact_return_speed_m_s", "central_tangent_abs_max_m_s", "eccentric_tangent_abs_max_m_s", "command_support_norm_max_m_s", "eccentric_offset_abs_min_m", "eccentric_offset_abs_max_m"):
        _require(not isinstance(c[key], bool) and np.isfinite(c[key]) and c[key] > 0, f"Positive finite {key} required")
    for key in ("initial_seed", "split_seed", "action_seed"):
        _require(type(settings[key]) is int and settings[key] >= 0, "Nonnegative RNG seeds required")
    _require(settings["limits"]["feedback_collection_enabled"] is False and settings["limits"]["includes_p2_01_stress_fixture"] is False and settings["limits"]["final_evaluation_allowed"] is False, "v1 is a bounded open-loop pilot")
    _require(0 < c["forward_speed_min_m_s"] <= c["forward_speed_max_m_s"] <= c["command_support_norm_max_m_s"] <= environment_config["pusher"]["command_speed_norm_max_m_s"], "Command support exceeds environment")
    _require(0 < c["eccentric_offset_abs_min_m"] <= c["eccentric_offset_abs_max_m"] < environment_config["object"]["size_full_xyz_m"][1]/2, "Offset outside chosen rear face")
    families = [f"family-{i:03d}" for i in range(sum(counts.values()))]
    assignments = split_families(families, counts, settings["split_seed"])
    rng = np.random.Generator(np.random.PCG64(settings["initial_seed"]))
    positions = [{"object_x": float(rng.uniform(*init["object_x_range_m"])),
                  "object_y": float(rng.uniform(*init["object_y_range_m"])),
                  "object_yaw": float(rng.uniform(*init["object_yaw_range_rad"]))} for _ in families]
    mode_rank = {}
    for split in ("train", "validation", "smoke-test"):
        members = sorted(f for f in families if assignments[f] == split)
        for rank, family in enumerate(members):
            mode_rank[family] = (MODES[rank % 4], rank//4)
    plans = []
    for index, family in enumerate(families):
        mode, rank = mode_rank[family]
        action_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([settings["action_seed"], index])))
        pos = positions[index]; yaw = pos["object_yaw"]
        direction = np.array([np.cos(yaw), np.sin(yaw)]); tangent = np.array([-direction[1], direction[0]])
        offset = (1 if rank % 2 == 0 else -1)*float(action_rng.uniform(c["eccentric_offset_abs_min_m"], c["eccentric_offset_abs_max_m"])) if mode == "eccentric" else 0.
        distance = environment_config["object"]["size_full_xyz_m"][0]/2 + environment_config["pusher"]["radius_m"] + environment_config["reset"]["pusher_surface_gap_m"]
        pusher = np.array([pos["object_x"], pos["object_y"]])-distance*direction+offset*tangent
        initial = InitialSpec(**pos, pusher_x=float(pusher[0]), pusher_y=float(pusher[1]))
        actions, phases = [], []
        def append(normal, lateral, ticks, label):
            command = normal*direction+lateral*tangent
            norm = float(np.linalg.norm(command))
            if norm > c["command_support_norm_max_m_s"]:
                command *= c["command_support_norm_max_m_s"]/norm
            actions.extend([command.tolist() for _ in range(ticks)]); phases.extend([label]*ticks)
        if mode == "noncontact":
            append(0, 0, 2, "zero")
            append(-c["retreat_speed_m_s"], 0, 4, "retreat")
            append(0, (1 if rank % 2 == 0 else -1)*c["noncontact_lateral_speed_m_s"], 4, "lateral")
            append(c["noncontact_return_speed_m_s"], 0, 4, "partial_return")
            append(0, 0, 2, "zero")
        elif mode == "contact_transition":
            speed = float(action_rng.uniform(c["forward_speed_min_m_s"], c["forward_speed_max_m_s"]))
            append(speed, 0, 5, "forward")
            append(-c["retreat_speed_m_s"], 0, 3, "retreat")
            append(speed, 0, 8, "reapproach")
        else:
            lateral_limit = c["central_tangent_abs_max_m_s"] if mode == "central" else c["eccentric_tangent_abs_max_m_s"]
            for block in range(4):
                append(float(action_rng.uniform(c["forward_speed_min_m_s"], c["forward_speed_max_m_s"])),
                       float(action_rng.uniform(-lateral_limit, lateral_limit)), 4, f"forward_block_{block}")
        _require(len(actions) == settings["actions_per_episode"], "Wrong action template length")
        plans.append({"episode_id": f"episode-{index:03d}", "episode_index": index, "family_id": family,
                      "family_definition": {"base_initial_spec": asdict(initial), "variant_rule": settings["family_rule"]},
                      "split": assignments[family], "mode": mode, "initial_spec": asdict(initial),
                      "pusher_tangent_offset_m": offset, "actions": actions, "phases": phases,
                      "action_rng_key": [settings["action_seed"], index]})
    return plans, assignments
