"""Pure tests: no environment construction, physical steps or model learning."""
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


ROOT = Path(__file__).resolve().parents[1]


class PreflightPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = json.loads((ROOT / "configs/p4_data_intervention.design.v1.json").read_text())
        cls.config = load_pilot_config()
        cls.plans = build_preflight_plan(cls.design, cls.config)

    def test_twelve_variants_are_registered_as_three_excluded_families(self):
        self.assertEqual(len(self.plans), 12)
        self.assertEqual(len({p["episode_id"] for p in self.plans}), 12)
        self.assertEqual([p["episode_index"] for p in self.plans], list(range(12)))
        families = {p["family_id"] for p in self.plans}
        self.assertEqual(len(families), 3)
        self.assertEqual(Counter(p["family_id"] for p in self.plans), {f: 4 for f in families})
        for family in families:
            group = [p for p in self.plans if p["family_id"] == family]
            self.assertEqual({(p["y_condition"], p["stop_condition"]) for p in group},
                             {("Y0", "S0"), ("Y0", "S1"), ("Y1", "S0"), ("Y1", "S1")})
            self.assertEqual(len({json.dumps(p["rng_metadata"]) for p in group}), 1)
        self.assertEqual({p["split"] for p in self.plans}, {"preflight"})
        self.assertTrue(all(not p["training_eligible"] and not p["development_evaluation_eligible"]
                            for p in self.plans))

    def test_all_five_independent_draws_match_declared_pcg64_keys(self):
        c = self.design["collection"]
        draw_names = ["object_x", "object_y", "object_yaw", "gap", "eccentric_abs_offset"]
        ranges = [c[k] for k in ("object_x_range_m", "object_y_range_m", "object_yaw_range_rad",
                                 "approach_gap_range_m", "eccentric_abs_offset_range_m")]
        for p in self.plans[::4]:
            key = [c["preflight_initial_seed"], p["family_rank"]]
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(key)))
            expected = [float(rng.uniform(*bounds)) for bounds in ranges]
            self.assertEqual(p["rng_metadata"]["draw_order"], draw_names)
            self.assertEqual(list(p["rng_metadata"]["draws"].values()), expected)
            self.assertEqual(p["rng_metadata"]["state_after_draws"], rng.bit_generator.state)
            self.assertEqual(p["rng_metadata"]["seed_sequence_entropy"], key)
            for other_seed in (c["train_initial_seed"], c["development_initial_seed"]):
                other_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([other_seed, p["family_rank"]])))
                self.assertNotEqual(expected, [float(other_rng.uniform(*bounds)) for bounds in ranges])

    def test_body_center_geometry_includes_radius_and_warmup_compensation(self):
        self.assertEqual([p["geometry"] for p in self.plans[::4]],
                         ["side_pass", "central_approach", "eccentric_approach"])
        for p in self.plans:
            initial = p["initial_spec"]
            theta = initial["object_yaw"]
            rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            reset_xy = np.array([initial["pusher_x"], initial["pusher_y"]])
            compensated = reset_xy + .2 * np.asarray(p["moving_action"])
            local = rotation.T @ (compensated - [initial["object_x"], initial["object_y"]])
            if p["geometry"] == "side_pass":
                expected = [-.063, 0.]
            else:
                offset = p["rng_metadata"]["draws"]["eccentric_abs_offset"] if p["geometry"] == "eccentric_approach" else 0.
                expected = [offset, -(.03 + .008 + p["rng_metadata"]["draws"]["gap"])]
            np.testing.assert_allclose(local, expected, rtol=0, atol=2e-17)
            np.testing.assert_allclose(compensated, p["nominal_world_pusher_center_anchor_m"], rtol=0, atol=1e-17)
            self.assertEqual(p["body_frame_pusher_center_anchor_m"], expected)
            # The designed preflight uses within-geometry rank zero for all three.
            self.assertEqual(p["within_geometry_rank"], 0)
            self.assertEqual(p["moving_action"][0], -.003)
            self.assertEqual(set(initial), {"object_x", "object_y", "object_yaw", "pusher_x", "pusher_y"})

    def test_actions_have_fixed_exposure_and_only_s1_has_contiguous_zero_k5(self):
        for p in self.plans:
            actions = np.asarray(p["core_actions"])
            self.assertEqual(actions.shape, (20, 2))
            self.assertEqual(p["warmup_actions"], [p["moving_action"]] * 2)
            np.testing.assert_array_equal(actions[:5], np.tile(p["moving_action"], (5, 1)))
            self.assertTrue(np.all(np.linalg.norm(actions, axis=1) <= .03))
            self.assertTrue(np.isfinite(actions).all())
            zero = np.all(actions == 0, axis=1)
            self.assertEqual(int(zero.sum()), 5)
            self.assertEqual(sum(int(zero[start:start + 5].sum()) for start in range(16)), 25)
            self.assertEqual(sum(int((~zero[start:start + 5]).sum()) for start in range(16)), 55)
            all_zero = [start for start in range(16) if zero[start:start + 5].all()]
            self.assertEqual(all_zero, [10] if p["stop_condition"] == "S1" else [])
            expected_zero = [10, 11, 12, 13, 14] if p["stop_condition"] == "S1" else [5, 7, 9, 11, 13]
            self.assertEqual(np.flatnonzero(zero).tolist(), expected_zero)
            self.assertEqual(p["core_phases"], ["Z" if v else "M" for v in zero])
            self.assertEqual(p["core_window_starts"], list(range(16)))

    def test_same_y_pair_shares_initial_warmup_and_first_five_core_actions(self):
        for left, right in zip(self.plans[::2], self.plans[1::2]):
            self.assertEqual(left["stop_condition"], "S0")
            self.assertEqual(right["stop_condition"], "S1")
            for key in ("family_id", "initial_spec", "moving_action", "warmup_actions", "warmup_snapshot_group"):
                self.assertEqual(left[key], right[key])
            self.assertEqual(left["core_actions"][:5], right["core_actions"][:5])
        for low, high in zip(self.plans[::4], self.plans[2::4]):
            self.assertEqual(low["family_id"], high["family_id"])
            for key in ("object_x", "object_y", "object_yaw", "pusher_x"):
                self.assertEqual(low["initial_spec"][key], high["initial_spec"][key])
            self.assertNotEqual(low["initial_spec"]["pusher_y"], high["initial_spec"]["pusher_y"])
            self.assertNotEqual(low["warmup_snapshot_group"], high["warmup_snapshot_group"])

    def test_plan_is_pure_deterministic_json_safe_and_not_aliased(self):
        design, config = deepcopy(self.design), deepcopy(self.config)
        before = deepcopy((design, config))
        with patch("push_align.environment.PlanarPushEnv", side_effect=AssertionError("No physics allowed")):
            plans = build_preflight_plan(design, config)
        self.assertEqual(plans, self.plans)
        self.assertEqual((design, config), before)
        self.assertEqual(json.loads(json.dumps(plans, allow_nan=False)), plans)
        plans[0]["core_actions"][0][0] = 123.
        plans[0]["rng_metadata"]["draws"]["object_x"] = 999.
        self.assertEqual(plans[1], self.plans[1])
        self.assertEqual(plans[0]["core_actions"][1], self.plans[0]["core_actions"][1])

    def test_changed_seed_changes_all_poses_without_changing_actions(self):
        design = deepcopy(self.design)
        design["collection"]["preflight_initial_seed"] += 10
        changed = build_preflight_plan(design, self.config)
        for old, new in zip(self.plans, changed):
            self.assertNotEqual(old["family_id"], new["family_id"])
            self.assertNotEqual(old["initial_spec"], new["initial_spec"])
            for key in ("geometry", "moving_action", "core_actions", "core_phases"):
                self.assertEqual(old[key], new[key])

    def test_incompatible_contracts_fail_before_collection(self):
        changes = [
            ("core_ticks", 15), ("warmup_ticks", 0), ("S0_pattern", "M" * 15 + "Z" * 5),
            ("variants_per_family", 5), ("preflight_families_excluded_from_both", 4),
            ("physical_velocity_assignment_forbidden", False),
            ("stop_variants_share_exact_warmup_snapshot", False),
            ("preflight_Y1_m_s", .04), ("matched_world_X_m_s", [0., 0., 0.]),
            ("preflight_initial_seed", self.design["collection"]["train_initial_seed"]),
            ("preflight_initial_seed", True), ("object_x_range_m", [float("nan"), .02]),
            ("preflight_intervals_upper_bound", 529),
        ]
        for key, value in changes:
            with self.subTest(key=key):
                design = deepcopy(self.design)
                design["collection"][key] = value
                with self.assertRaises(DataContractError):
                    build_preflight_plan(design, self.config)
        for section, key, value in (("object", "size_full_xyz_m", [.08, .07, .02]),
                                    ("pusher", "radius_m", .004),
                                    ("physics", "action_dt_s", .2),
                                    ("pusher", "command_speed_norm_max_m_s", .02)):
            with self.subTest(config_key=key):
                config = deepcopy(self.config)
                config[section][key] = value
                with self.assertRaises(DataContractError):
                    build_preflight_plan(self.design, config)
        with self.assertRaises(DataContractError):
            build_preflight_plan({}, self.config)


if __name__ == "__main__":
    unittest.main()
