"""Pure, frozen family registration for the P4 main data intervention.

This module constructs the training/development initial-condition list before
physics. It does not collect transitions, set physical velocities, filter
outcomes, or authorize model learning. A plan's training eligibility applies
only to its core; warmup and any diagnostic replays remain ineligible.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

import numpy as np

from .data import DataContractError
from .data_intervention import DRAW_NAMES, GEOMETRIES, PATTERNS, _require, _validate
from .environment import InitialSpec


_SPLITS = (
    ("train", "train", "train_families", "train_initial_seed", "train"),
    ("development", "validation", "development_evaluation_families",
     "development_initial_seed", "evaluation"),
)


def _validate_main(design, env_config):
    """Require the frozen main contract in addition to the common geometry rules."""
    c = _validate(design, env_config)
    fixed = {
        "train_families": 24,
        "development_evaluation_families": 12,
        "train_initial_seed": 2026101301,
        "development_initial_seed": 2026101302,
        "preflight_initial_seed": 2026101303,
        "train_Y0_m_s": [.010, .014],
        "train_Y1_m_s": [.022, .029],
        "evaluation_Y0_m_s": [.011, .013],
        "evaluation_Y1_m_s": [.024, .027],
        "family_split": "register family IDs before physics; all four variants and every warmup/repeat stay in the same family split",
        "main_core_intervals": 2880,
        "main_warmup_intervals_upper_bound": 288,
        "main_intervals_upper_bound": 3168,
        "total_action_intervals_upper_bound": 3696,
    }
    for key, expected in fixed.items():
        _require(type(c[key]) is type(expected) and c[key] == expected,
                 f"Unsupported frozen main collection contract: {key}")
    return c


def build_main_plan(design, env_config):
    """Return the registered 24 train + 12 development families, four variants each.

    All family draws use their declared split seed and local family rank. The
    serialized data split is ``validation`` for held-out development episodes;
    ``semantic_split`` makes their developmental (not final-test) role explicit.
    Same-family S0/S1 variants request one exact shared warmup snapshot for each
    Y condition. The collector must demonstrate that equality in recorded data.
    """
    try:
        return _build_main_plan(design, env_config)
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise DataContractError(f"Malformed main data-intervention design: {error}") from error


def _build_main_plan(design, env_config):
    c = _validate_main(design, env_config)
    ranges = [c[key] for key in ("object_x_range_m", "object_y_range_m",
                                "object_yaw_range_rad", "approach_gap_range_m",
                                "eccentric_abs_offset_range_m")]
    warmup_duration = c["warmup_ticks"] * c["action_dt_s"]
    half_x, half_y = np.asarray(env_config["object"]["size_full_xyz_m"][:2]) / 2
    radius = env_config["pusher"]["radius_m"]
    plans = []
    for semantic_split, split, count_key, seed_key, speed_prefix in _SPLITS:
        seed = c[seed_key]
        is_train = semantic_split == "train"
        for rank in range(c[count_key]):
            rng_key = [seed, rank]
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(rng_key)))
            before_draws = deepcopy(rng.bit_generator.state)
            # Even unused geometry draws are consumed, in the registered order.
            draws = {name: float(rng.uniform(*bounds))
                     for name, bounds in zip(DRAW_NAMES, ranges)}
            geometry = GEOMETRIES[rank % 3]
            within_rank = rank // 3
            pair_index = within_rank % 2
            offset_sign = 1 if within_rank % 2 == 0 else -1
            offset = (offset_sign * draws["eccentric_abs_offset"]
                      if geometry == "eccentric_approach" else 0.)
            gap = c["side_pass_gap_m"] if geometry == "side_pass" else draws["gap"]
            anchor_body = ([-float(half_x + radius + gap), 0.] if geometry == "side_pass"
                           else [offset, -float(half_y + radius + gap)])
            yaw = draws["object_yaw"]
            rotation = np.array([[np.cos(yaw), -np.sin(yaw)],
                                 [np.sin(yaw), np.cos(yaw)]])
            anchor_world = (np.array([draws["object_x"], draws["object_y"]])
                            + rotation @ anchor_body)
            family = f"p4-data-{semantic_split}-{seed}-f{rank:03d}"
            rng_metadata = {
                "bit_generator": "PCG64", "seed_sequence_entropy": rng_key,
                "draw_order": list(DRAW_NAMES), "draws": draws,
                "state_before_draws": before_draws,
                "state_after_draws": deepcopy(rng.bit_generator.state),
            }
            for y_index in range(2):
                y_label = f"Y{y_index}"
                command = [float(c["matched_world_X_m_s"][within_rank % 3]),
                           float(c[f"{speed_prefix}_{y_label}_m_s"][pair_index])]
                _require(np.linalg.norm(command) <= c["command_norm_limit_m_s"],
                         "Planned command exceeds the declared norm limit; clipping is forbidden")
                reset_pusher = anchor_world - warmup_duration * np.asarray(command)
                initial = InitialSpec(object_x=draws["object_x"], object_y=draws["object_y"],
                                      object_yaw=yaw, pusher_x=float(reset_pusher[0]),
                                      pusher_y=float(reset_pusher[1]))
                for stop_label, pattern in PATTERNS.items():
                    plan = {
                        "episode_id": f"{family}-{y_label}-{stop_label}",
                        "episode_index": len(plans), "family_id": family,
                        "family_rank": rank, "within_geometry_rank": within_rank,
                        "y_speed_pair_index": pair_index, "geometry": geometry,
                        "split": split, "semantic_split": semantic_split,
                        "training_eligible": is_train,
                        "core_training_eligible": is_train,
                        "warmup_training_eligible": False,
                        "development_evaluation_eligible": not is_train,
                        "y_condition": y_label, "stop_condition": stop_label,
                        "initial_spec": asdict(initial), "moving_action": command,
                        "warmup_actions": [list(command) for _ in range(c["warmup_ticks"])],
                        "core_actions": [list(command) if phase == "M" else [0., 0.]
                                         for phase in pattern],
                        "core_phases": list(pattern), "action_dt_s": c["action_dt_s"],
                        "warmup_duration_s": warmup_duration,
                        "warmup_snapshot_group": f"{family}-{y_label}",
                        "required_same_speed_prefix_ticks": 5,
                        "body_frame_pusher_center_anchor_m": anchor_body,
                        "nominal_world_pusher_center_anchor_m": anchor_world.tolist(),
                        "nominal_surface_gap_m": gap, "signed_eccentric_offset_m": offset,
                        "rng_metadata": rng_metadata,
                        "core_window_starts": list(range(16)), "model_K": 5,
                    }
                    plans.append(deepcopy(plan))
    return plans
