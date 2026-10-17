"""Pure, predeclared P4 data-intervention preflight plans.

This module samples poses and constructs commands only. It does not instantiate
an environment, assign physical velocities, collect data, or train a model.
S0/S1 plans request a shared warmup snapshot; the runner must verify that claim
against physical state and recorded transitions.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math

import numpy as np

from .data import DataContractError
from .environment import InitialSpec


GEOMETRIES = ("side_pass", "central_approach", "eccentric_approach")
PATTERNS = {"S0": "MMMMMZMZMZMZMZMMMMMM", "S1": "MMMMMMMMMMZZZZZMMMMM"}
DRAW_NAMES = ("object_x", "object_y", "object_yaw", "gap", "eccentric_abs_offset")
_RNG_RULE = (
    "numpy PCG64(SeedSequence([split_initial_seed, family_rank])); draw object_x, "
    "object_y, object_yaw, gap, eccentric_abs_offset in that order, including unused draws"
)
_RANK_RULE = (
    "geometry=rank%3; within_geometry_rank=rank//3; Y pair index=within_geometry_rank%2; "
    "X index=within_geometry_rank%3; eccentric sign positive on even within_geometry_rank"
)


def _require(condition, message):
    if not condition:
        raise DataContractError(message)


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _validate(design, config):
    _require(design["schema"] == "p4-data-intervention-design-v1", "Unsupported design schema")
    _require(design["executable_runner_config"] is False,
             "This builder consumes the declared design, not a generic runner configuration")
    c = design["collection"]
    fixed = {
        "action_dt_s": .1,
        "core_ticks": 20,
        "warmup_ticks": 2,
        "warmup_in_training_windows": False,
        "moving_command_count": 15,
        "zero_command_count": 5,
        "preflight_families_excluded_from_both": 3,
        "variants_per_family": 4,
        "geometry_intents": list(GEOMETRIES),
        "initial_rng": _RNG_RULE,
        "family_rank_rule": _RANK_RULE,
        "physical_velocity_assignment_forbidden": True,
        "stop_variants_share_exact_warmup_snapshot": True,
        "same_speed_pair_core_first5_exact": True,
        "no_drop_replace_or_success_filter": True,
        "settling_logged_separately": True,
        "preflight_Y0_m_s": .014,
        "preflight_Y1_m_s": .029,
        "matched_world_X_m_s": [-.003, 0., .003],
        "command_norm_limit_m_s": .03,
        "side_pass_gap_m": .015,
        "approach_gap_range_m": [.002, .004],
        "eccentric_abs_offset_range_m": [.010, .014],
        "measured_high_Y_gate_m_s": .018,
        "measured_low_Y_ceiling_m_s": .0149908461,
        "preflight_base_intervals": 264,
        "preflight_dt_half_intervals": 132,
        "preflight_exact_replay_intervals": 132,
        "preflight_intervals_upper_bound": 528,
    }
    fixed.update({f"{name}_pattern": pattern for name, pattern in PATTERNS.items()})
    for key, expected in fixed.items():
        # Reject bool masquerading as a numeric count and vice versa.
        _require(type(c[key]) is type(expected) and c[key] == expected,
                 f"Unsupported collection contract: {key}")
    _require(design["model"]["K"] == 5 and type(design["model"]["K"]) is int,
             "The declared exposure balance requires K=5")
    _require(design["training"]["new_window_starts"] == list(range(16)),
             "Only the sixteen core K5 window starts are supported")
    _require(c["reset_pusher_rule"] ==
             "rotate body-frame anchor then translate by object_xy; subtract0.2*world_command; "
             "use actual settling and warmup, never assign velocity", "Unsupported reset rule")
    _require(c["body_frame_pusher_center_anchors"] == {
        "side_pass": "[-(0.04+0.008+0.015),0]",
        "central_approach": "[0,-(0.03+0.008+gap)]",
        "eccentric_approach": "[signed_offset,-(0.03+0.008+gap)]",
    }, "Unsupported body-frame center anchors")
    seeds = [c[key] for key in ("train_initial_seed", "development_initial_seed",
                               "preflight_initial_seed")]
    _require(all(type(seed) is int and seed >= 0 for seed in seeds) and len(set(seeds)) == 3,
             "Preflight, training, and development require distinct nonnegative seed keys")
    _require(config["object"]["size_full_xyz_m"] == [.08, .06, .02]
             and config["pusher"]["radius_m"] == .008,
             "Anchor design requires the fixed 80x60x20 mm object and 8 mm pusher radius")
    _require(config["physics"]["action_dt_s"] == c["action_dt_s"],
             "Preflight action duration differs from the environment")
    _require(_finite_number(config["pusher"]["command_speed_norm_max_m_s"])
             and config["pusher"]["command_speed_norm_max_m_s"] >= c["command_norm_limit_m_s"],
             "Environment cannot support all planned commands without clipping")
    for key in ("object_x_range_m", "object_y_range_m", "object_yaw_range_rad"):
        bounds = c[key]
        _require(type(bounds) is list and len(bounds) == 2
                 and all(_finite_number(x) for x in bounds) and bounds[0] < bounds[1]
                 and bounds == config["initial_conditions"][key],
                 f"Design and nominal environment range differ: {key}")
    return c


def build_preflight_plan(design, env_config):
    """Return twelve JSON-safe plans, grouped by family, Y, then stop condition.

    The family is the independent randomization unit. Warmup and S0/S1 variants
    share the family split, and the preflight split is excluded from all later
    training/development. Geometry is an intent, not a measured contact label.
    """
    try:
        return _build_preflight_plan(design, env_config)
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise DataContractError(f"Malformed preflight design: {error}") from error


def _build_preflight_plan(design, env_config):
    c = _validate(design, env_config)
    ranges = [c[key] for key in ("object_x_range_m", "object_y_range_m",
                                "object_yaw_range_rad", "approach_gap_range_m",
                                "eccentric_abs_offset_range_m")]
    seed = c["preflight_initial_seed"]
    warmup_duration = c["warmup_ticks"] * c["action_dt_s"]
    half_x, half_y = np.asarray(env_config["object"]["size_full_xyz_m"][:2]) / 2
    radius = env_config["pusher"]["radius_m"]
    plans = []
    for rank in range(c["preflight_families_excluded_from_both"]):
        rng_key = [seed, rank]
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(rng_key)))
        before_draws = deepcopy(rng.bit_generator.state)
        # All five draws occur even when a geometry does not use the last two.
        draws = {name: float(rng.uniform(*bounds)) for name, bounds in zip(DRAW_NAMES, ranges)}
        geometry = GEOMETRIES[rank % 3]
        within_rank = rank // 3
        offset_sign = 1 if within_rank % 2 == 0 else -1
        offset = offset_sign * draws["eccentric_abs_offset"] if geometry == "eccentric_approach" else 0.
        gap = c["side_pass_gap_m"] if geometry == "side_pass" else draws["gap"]
        anchor_body = ([-float(half_x + radius + gap), 0.] if geometry == "side_pass"
                       else [offset, -float(half_y + radius + gap)])
        yaw = draws["object_yaw"]
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        anchor_world = np.array([draws["object_x"], draws["object_y"]]) + rotation @ anchor_body
        family = f"p4-data-preflight-{seed}-f{rank:03d}"
        rng_metadata = {
            "bit_generator": "PCG64", "seed_sequence_entropy": rng_key,
            "draw_order": list(DRAW_NAMES), "draws": draws,
            "state_before_draws": before_draws,
            "state_after_draws": deepcopy(rng.bit_generator.state),
        }
        for y_index in range(2):
            y_label = f"Y{y_index}"
            command = [float(c["matched_world_X_m_s"][within_rank % 3]),
                       float(c[f"preflight_{y_label}_m_s"])]
            _require(np.linalg.norm(command) <= c["command_norm_limit_m_s"],
                     "Planned command norm exceeds the declared limit; clipping is forbidden")
            reset_pusher = anchor_world - warmup_duration * np.asarray(command)
            initial = InitialSpec(object_x=draws["object_x"], object_y=draws["object_y"],
                                  object_yaw=yaw, pusher_x=float(reset_pusher[0]),
                                  pusher_y=float(reset_pusher[1]))
            for stop_label, pattern in PATTERNS.items():
                plan = {
                    "episode_id": f"{family}-{y_label}-{stop_label}",
                    "episode_index": len(plans), "family_id": family, "family_rank": rank,
                    "within_geometry_rank": within_rank, "geometry": geometry,
                    "split": "preflight", "training_eligible": False,
                    "development_evaluation_eligible": False,
                    "y_condition": y_label, "stop_condition": stop_label,
                    "initial_spec": asdict(initial), "moving_action": command,
                    "warmup_actions": [list(command) for _ in range(c["warmup_ticks"])],
                    "core_actions": [list(command) if phase == "M" else [0., 0.] for phase in pattern],
                    "core_phases": list(pattern), "action_dt_s": c["action_dt_s"],
                    "warmup_duration_s": warmup_duration,
                    "warmup_snapshot_group": f"{family}-{y_label}",
                    "required_same_speed_prefix_ticks": 5,
                    "body_frame_pusher_center_anchor_m": anchor_body,
                    "nominal_world_pusher_center_anchor_m": anchor_world.tolist(),
                    "nominal_surface_gap_m": gap, "signed_eccentric_offset_m": offset,
                    "rng_metadata": rng_metadata,
                    "core_window_starts": list(range(16)),
                    "model_K": 5,
                }
                # Plans do not alias one another; a recorder can enrich one safely.
                plans.append(deepcopy(plan))
    return plans
