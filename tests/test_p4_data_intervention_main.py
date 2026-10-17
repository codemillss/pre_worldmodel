"""Registration tests only: no physical integration or model operations."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from push_align.config import load_pilot_config
from push_align.data import DataContractError
from push_align.data_intervention import build_preflight_plan
from push_align.data_intervention_main import build_main_plan


ROOT = Path(__file__).resolve().parents[1]


class MainInterventionPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = json.loads((ROOT / "configs/p4_data_intervention.design.v1.json").read_text())
        cls.config = load_pilot_config()
        cls.plans = build_main_plan(cls.design, cls.config)

    def test_all_families_have_all_four_variants_and_one_split(self):
        self.assertEqual(len(self.plans), 144)
        self.assertEqual(len({p["episode_id"] for p in self.plans}), 144)
        self.assertEqual([p["episode_index"] for p in self.plans], list(range(144)))
        self.assertEqual(Counter(p["semantic_split"] for p in self.plans),
                         {"train": 96, "development": 48})
        families = {p["family_id"] for p in self.plans}
        self.assertEqual(len(families), 36)
        for family in families:
            group = [p for p in self.plans if p["family_id"] == family]
            self.assertEqual(len(group), 4)
            self.assertEqual({(p["y_condition"], p["stop_condition"]) for p in group},
                             {("Y0", "S0"), ("Y0", "S1"), ("Y1", "S0"), ("Y1", "S1")})
            self.assertEqual(len({p["split"] for p in group}), 1)
            self.assertEqual(len({json.dumps(p["rng_metadata"], sort_keys=True) for p in group}), 1)
        for semantic, count, data_split in (("train", 24, "train"),
                                             ("development", 12, "validation")):
            group = [p for p in self.plans[::4] if p["semantic_split"] == semantic]
            self.assertEqual([p["family_rank"] for p in group], list(range(count)))
            self.assertEqual({p["split"] for p in group}, {data_split})

    def test_core_eligibility_and_warmup_exclusion_are_explicit(self):
        for p in self.plans:
            train = p["semantic_split"] == "train"
            self.assertIs(p["training_eligible"], train)
            self.assertIs(p["core_training_eligible"], train)
            self.assertIs(p["development_evaluation_eligible"], not train)
            self.assertIs(p["warmup_training_eligible"], False)
            self.assertEqual(p["core_window_starts"], list(range(16)))
        per_arm = Counter((p["y_condition"], p["stop_condition"])
                          for p in self.plans if p["training_eligible"])
        self.assertEqual(set(per_arm.values()), {24})
        self.assertEqual({key: value * 16 for key, value in per_arm.items()},
                         {("Y0", "S0"): 384, ("Y0", "S1"): 384,
                          ("Y1", "S0"): 384, ("Y1", "S1"): 384})

    def test_pcg64_five_draw_order_and_split_seed_are_exact(self):
        c = self.design["collection"]
        names = ["object_x", "object_y", "object_yaw", "gap", "eccentric_abs_offset"]
        ranges = [c[key] for key in ("object_x_range_m", "object_y_range_m",
                  "object_yaw_range_rad", "approach_gap_range_m", "eccentric_abs_offset_range_m")]
        for p in self.plans[::4]:
            seed = 2026101301 if p["semantic_split"] == "train" else 2026101302
            key = [seed, p["family_rank"]]
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(key)))
            self.assertEqual(p["rng_metadata"]["state_before_draws"], rng.bit_generator.state)
            expected = [float(rng.uniform(*bounds)) for bounds in ranges]
            self.assertEqual(list(p["rng_metadata"]["draws"]), names)
            self.assertEqual(list(p["rng_metadata"]["draws"].values()), expected)
            self.assertEqual(p["rng_metadata"]["seed_sequence_entropy"], key)
            self.assertEqual(p["rng_metadata"]["state_after_draws"], rng.bit_generator.state)

    def test_preflight_train_and_development_ids_keys_and_poses_are_disjoint(self):
        groups = [build_preflight_plan(self.design, self.config)] + [
            [p for p in self.plans if p["semantic_split"] == semantic]
            for semantic in ("train", "development")]
        for i, left in enumerate(groups):
            for right in groups[i + 1:]:
                for key in ("family_id", "episode_id", "warmup_snapshot_group"):
                    self.assertTrue({p[key] for p in left}.isdisjoint(p[key] for p in right))
                left_keys = {tuple(p["rng_metadata"]["seed_sequence_entropy"]) for p in left}
                self.assertTrue(left_keys.isdisjoint(tuple(p["rng_metadata"]["seed_sequence_entropy"])
                                                     for p in right))
                def object_poses(plans):
                    return {tuple(p["initial_spec"][k] for k in ("object_x", "object_y", "object_yaw"))
                            for p in plans}
                self.assertTrue(object_poses(left).isdisjoint(object_poses(right)))

    def test_world_commands_follow_rank_not_random_draws(self):
        expected_speeds = {"train": {"Y0": [.010, .014], "Y1": [.022, .029]},
                           "development": {"Y0": [.011, .013], "Y1": [.024, .027]}}
        for p in self.plans:
            within = p["family_rank"] // 3
            pair = within % 2
            expected = [[-.003, 0., .003][within % 3],
                        expected_speeds[p["semantic_split"]][p["y_condition"]][pair]]
            self.assertEqual(p["within_geometry_rank"], within)
            self.assertEqual(p["y_speed_pair_index"], pair)
            self.assertEqual(p["moving_action"], expected)
            self.assertLessEqual(float(np.linalg.norm(expected)), .03)
            self.assertEqual(p["warmup_actions"], [expected, expected])
        for semantic in ("train", "development"):
            for y_label in ("Y0", "Y1"):
                values = {p["moving_action"][1] for p in self.plans
                          if p["semantic_split"] == semantic and p["y_condition"] == y_label}
                self.assertEqual(values, set(expected_speeds[semantic][y_label]))

    def test_center_geometry_radius_sign_and_world_warmup_compensation(self):
        geometries = ("side_pass", "central_approach", "eccentric_approach")
        for p in self.plans:
            self.assertEqual(p["geometry"], geometries[p["family_rank"] % 3])
            initial = p["initial_spec"]
            draws = p["rng_metadata"]["draws"]
            theta = initial["object_yaw"]
            rotation = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])
            actual_anchor = (np.array([initial["pusher_x"], initial["pusher_y"]])
                             + .2 * np.asarray(p["moving_action"]))
            local = rotation.T @ (actual_anchor - [initial["object_x"], initial["object_y"]])
            if p["geometry"] == "side_pass":
                expected = [-.063, 0.]
            else:
                sign = 1 if (p["family_rank"] // 3) % 2 == 0 else -1
                offset = sign * draws["eccentric_abs_offset"] if p["geometry"] == "eccentric_approach" else 0.
                expected = [offset, -(.03 + .008 + draws["gap"])]
            np.testing.assert_allclose(local, expected, rtol=0, atol=3e-17)
            np.testing.assert_allclose(actual_anchor, p["nominal_world_pusher_center_anchor_m"], rtol=0, atol=1e-17)
            self.assertEqual(p["body_frame_pusher_center_anchor_m"], expected)
            self.assertEqual(set(initial), {"object_x", "object_y", "object_yaw", "pusher_x", "pusher_y"})

    def test_shared_warmup_and_prefix_request_is_same_only_within_y_cell(self):
        for left, right in zip(self.plans[::2], self.plans[1::2]):
            self.assertEqual((left["stop_condition"], right["stop_condition"]), ("S0", "S1"))
            for key in ("initial_spec", "family_id", "moving_action", "warmup_actions",
                        "warmup_snapshot_group", "split", "semantic_split"):
                self.assertEqual(left[key], right[key])
            self.assertEqual(left["core_actions"][:5], right["core_actions"][:5])
        for low, high in zip(self.plans[::4], self.plans[2::4]):
            self.assertNotEqual(low["warmup_snapshot_group"], high["warmup_snapshot_group"])
            self.assertNotEqual(low["initial_spec"]["pusher_y"], high["initial_spec"]["pusher_y"])
            for key in ("object_x", "object_y", "object_yaw", "pusher_x"):
                self.assertEqual(low["initial_spec"][key], high["initial_spec"][key])

    def test_fixed_zero_exposure_and_contiguous_zero_window_are_preserved(self):
        for p in self.plans:
            actions = np.asarray(p["core_actions"])
            self.assertEqual(actions.shape, (20, 2))
            self.assertTrue(np.isfinite(actions).all())
            zero = np.all(actions == 0, axis=1)
            self.assertEqual(int(zero.sum()), 5)
            self.assertEqual(sum(int(zero[start:start + 5].sum()) for start in range(16)), 25)
            all_zero = [start for start in range(16) if zero[start:start + 5].all()]
            self.assertEqual(all_zero, [10] if p["stop_condition"] == "S1" else [])
            self.assertEqual(p["core_phases"], ["Z" if flag else "M" for flag in zero])
            np.testing.assert_array_equal(actions[~zero], np.tile(p["moving_action"], (15, 1)))
        self.assertEqual(sum(len(p["core_actions"]) for p in self.plans), 2880)
        self.assertEqual(sum(len(p["warmup_actions"]) for p in self.plans[::2]), 144)

    def test_pure_deterministic_json_safe_unaliased_and_preflight_unchanged(self):
        design, config = deepcopy(self.design), deepcopy(self.config)
        before = deepcopy((design, config))
        original_preflight = build_preflight_plan(design, config)
        with patch("push_align.environment.PlanarPushEnv", side_effect=AssertionError("No physics")):
            plans = build_main_plan(design, config)
        self.assertEqual(plans, self.plans)
        self.assertEqual((design, config), before)
        self.assertEqual(build_preflight_plan(design, config), original_preflight)
        self.assertEqual(json.loads(json.dumps(plans, allow_nan=False)), plans)
        plans[0]["core_actions"][0][0] = 999.
        plans[0]["rng_metadata"]["draws"]["object_x"] = 999.
        self.assertEqual(plans[1], self.plans[1])
        self.assertEqual(plans[0]["core_actions"][1], self.plans[0]["core_actions"][1])

    def test_changed_main_contract_is_rejected_not_silently_resampled(self):
        changes = [
            ("train_families", 25), ("development_evaluation_families", 11),
            ("train_initial_seed", 2026101304), ("development_initial_seed", 2026101305),
            ("preflight_initial_seed", 2026101306), ("train_families", True),
            ("train_Y0_m_s", [.009, .014]), ("train_Y1_m_s", [.022, .028]),
            ("evaluation_Y0_m_s", [.010, .013]), ("evaluation_Y1_m_s", [.024, .028]),
            ("train_Y0_m_s", [.01, float("nan")]),
            ("matched_world_X_m_s", [0., 0., 0.]),
            ("family_split", "random episode split"), ("main_core_intervals", 2881),
            ("main_warmup_intervals_upper_bound", 287), ("main_intervals_upper_bound", 3169),
            ("total_action_intervals_upper_bound", 3697),
            ("warmup_in_training_windows", True), ("no_drop_replace_or_success_filter", False),
            ("physical_velocity_assignment_forbidden", False),
        ]
        for key, value in changes:
            with self.subTest(key=key):
                design = deepcopy(self.design)
                design["collection"][key] = value
                with self.assertRaises(DataContractError):
                    build_main_plan(design, self.config)
        for bad_design in ({}, {"schema": "unknown"}):
            with self.assertRaises(DataContractError):
                build_main_plan(bad_design, self.config)


if __name__ == "__main__":
    unittest.main()
