"""P2-02 predeclared sampling/action contracts and two physical witnesses.

The two recordings test plan-to-P2-01 wiring and intended contact changes;
they do not replace collection-wide validity, coverage, or dt comparison.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import tomllib
import unittest

import numpy as np

from push_align import InitialSpec, PlanarPushEnv
from push_align.collection import MODES, build_collection_plan
from push_align.data import (
    DataContractError, EpisodeRecorder, make_contract, sequence_windows,
    validate_episode,
)


ROOT = Path(__file__).resolve().parents[1]


def object_pose(plan):
    return np.array([plan["initial_spec"][key]
                     for key in ("object_x", "object_y", "object_yaw")])


def local_commands(plan):
    yaw = plan["initial_spec"]["object_yaw"]
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)],
                         [np.sin(yaw), np.cos(yaw)]])
    return np.asarray(plan["actions"]) @ rotation


class P2CollectionPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "configs/p2_collection.v1.toml").open("rb") as file:
            cls.settings = tomllib.load(file)
        cls.config = PlanarPushEnv().config
        cls.plans, cls.assignments = build_collection_plan(cls.settings, cls.config)

    def test_forty_families_are_split_before_any_physics_without_final_test(self):
        self.assertEqual(len(self.plans), 40)
        self.assertEqual(len(self.assignments), 40)
        self.assertEqual(Counter(self.assignments.values()),
                         {"train": 24, "validation": 8, "smoke-test": 8})
        self.assertEqual({p["family_id"] for p in self.plans}, set(self.assignments))
        self.assertEqual(len({p["episode_id"] for p in self.plans}), 40)
        self.assertEqual([p["episode_index"] for p in self.plans], list(range(40)))
        self.assertNotIn("test", self.assignments.values())
        for p in self.plans:
            self.assertEqual(p["split"], self.assignments[p["family_id"]])
            self.assertEqual(p["family_definition"]["base_initial_spec"],
                             p["initial_spec"])
            self.assertEqual(p["family_definition"]["variant_rule"],
                             self.settings["family_rule"])

    def test_each_split_has_balanced_modes_and_both_eccentric_signs(self):
        for split, count in self.settings["split_counts"].items():
            with self.subTest(split=split):
                members = [p for p in self.plans if p["split"] == split]
                self.assertEqual(Counter(p["mode"] for p in members),
                                 {mode: count // 4 for mode in MODES})
                eccentric = [p for p in members if p["mode"] == "eccentric"]
                self.assertEqual(Counter(np.sign(p["pusher_tangent_offset_m"])
                                         for p in eccentric),
                                 {1.: count // 8, -1.: count // 8})
                for p in eccentric:
                    offset = abs(p["pusher_tangent_offset_m"])
                    self.assertGreaterEqual(offset, .010)
                    self.assertLessEqual(offset, .015)
                self.assertTrue(all(p["pusher_tangent_offset_m"] == 0
                                    for p in members if p["mode"] != "eccentric"))

    def test_nominal_initial_ranges_and_rear_face_geometry_are_respected(self):
        nominal = self.config["initial_conditions"]
        half_length = self.config["object"]["size_full_xyz_m"][0] / 2
        half_width = self.config["object"]["size_full_xyz_m"][1] / 2
        radius = self.config["pusher"]["radius_m"]
        gap = self.config["reset"]["pusher_surface_gap_m"]
        for p in self.plans:
            with self.subTest(episode=p["episode_id"]):
                s = p["initial_spec"]
                for key, ranges in (("object_x", "object_x_range_m"),
                                    ("object_y", "object_y_range_m"),
                                    ("object_yaw", "object_yaw_range_rad")):
                    self.assertGreaterEqual(s[key], nominal[ranges][0])
                    self.assertLessEqual(s[key], nominal[ranges][1])
                yaw = s["object_yaw"]
                normal = np.array([np.cos(yaw), np.sin(yaw)])
                tangent = np.array([-np.sin(yaw), np.cos(yaw)])
                displacement = np.array([s["pusher_x"] - s["object_x"],
                                         s["pusher_y"] - s["object_y"]])
                self.assertAlmostEqual(displacement @ normal,
                                       -(half_length + radius + gap), places=14)
                self.assertAlmostEqual(displacement @ tangent,
                                       p["pusher_tangent_offset_m"], places=14)
                self.assertLess(abs(displacement @ tangent), half_width)
                for coordinate, bounds in (("pusher_x", "x_bounds_m"),
                                           ("pusher_y", "y_bounds_m")):
                    low, high = self.config["workspace"][bounds]
                    self.assertGreaterEqual(s[coordinate] - radius, low)
                    self.assertLessEqual(s[coordinate] + radius, high)

    def test_command_norm_support_and_sixteen_action_phase_semantics(self):
        for p in self.plans:
            with self.subTest(episode=p["episode_id"]):
                commands = np.asarray(p["actions"])
                self.assertEqual(commands.shape, (16, 2))
                self.assertTrue(np.isfinite(commands).all())
                self.assertLessEqual(float(np.linalg.norm(commands, axis=1).max()),
                                     .03 + 1e-15)
                local = local_commands(p)
                if p["mode"] == "noncontact":
                    self.assertEqual(p["phases"], ["zero"] * 2 + ["retreat"] * 4
                                     + ["lateral"] * 4 + ["partial_return"] * 4
                                     + ["zero"] * 2)
                    np.testing.assert_allclose(local[:2], 0, atol=1e-16)
                    np.testing.assert_allclose(local[-2:], 0, atol=1e-16)
                    self.assertTrue(np.all(local[2:6, 0] < 0))
                    np.testing.assert_allclose(local[6:10, 0], 0, atol=1e-16)
                    self.assertTrue(np.all(local[10:14, 0] > 0))
                    # Its partial return cannot undo the prior retreat.
                    self.assertLess(float(local[:, 0].sum()), 0)
                elif p["mode"] == "contact_transition":
                    self.assertEqual(p["phases"], ["forward"] * 5 + ["retreat"] * 3
                                     + ["reapproach"] * 8)
                    self.assertTrue(np.all(local[:5, 0] > 0))
                    self.assertTrue(np.all(local[5:8, 0] < 0))
                    self.assertTrue(np.all(local[8:, 0] > 0))
                    np.testing.assert_allclose(local[:, 1], 0, atol=1e-16)
                    np.testing.assert_array_equal(commands[:5],
                                                  np.repeat(commands[:1], 5, axis=0))
                    np.testing.assert_array_equal(commands[8:],
                                                  np.repeat(commands[:1], 8, axis=0))
                else:
                    self.assertEqual(p["phases"], sum(([f"forward_block_{i}"] * 4
                                                      for i in range(4)), []))
                    self.assertTrue(np.all(local[:, 0] > 0))
                    for block in range(4):
                        chunk = commands[4 * block:4 * block + 4]
                        np.testing.assert_array_equal(chunk,
                                                      np.repeat(chunk[:1], 4, axis=0))

    def test_repeated_plan_is_identical_without_mutating_inputs(self):
        settings, config = deepcopy(self.settings), deepcopy(self.config)
        before_settings, before_config = deepcopy(settings), deepcopy(config)
        first = build_collection_plan(settings, config)
        second = build_collection_plan(settings, config)
        self.assertEqual(first, second)
        self.assertEqual(settings, before_settings)
        self.assertEqual(config, before_config)

    def test_initial_seed_changes_pose_but_not_split_or_local_action_draws(self):
        changed = deepcopy(self.settings); changed["initial_seed"] += 1
        plans, assignments = build_collection_plan(changed, self.config)
        self.assertEqual(assignments, self.assignments)
        self.assertTrue(any(not np.array_equal(object_pose(a), object_pose(b))
                            for a, b in zip(self.plans, plans)))
        for original, other in zip(self.plans, plans):
            self.assertEqual(original["mode"], other["mode"])
            self.assertEqual(original["pusher_tangent_offset_m"],
                             other["pusher_tangent_offset_m"])
            self.assertEqual(original["action_rng_key"], other["action_rng_key"])
            # World commands rotate with yaw; compare the independent local draws.
            np.testing.assert_allclose(local_commands(original), local_commands(other),
                                       rtol=0, atol=2e-17)

    def test_action_seed_changes_action_draws_without_changing_object_pose_or_split(self):
        changed = deepcopy(self.settings); changed["action_seed"] += 1
        plans, assignments = build_collection_plan(changed, self.config)
        self.assertEqual(assignments, self.assignments)
        for original, other in zip(self.plans, plans):
            np.testing.assert_array_equal(object_pose(original), object_pose(other))
            self.assertEqual(original["mode"], other["mode"])
            self.assertEqual(other["action_rng_key"],
                             [changed["action_seed"], other["episode_index"]])
        self.assertTrue(any(a["actions"] != b["actions"]
                            for a, b in zip(self.plans, plans)))

    def test_split_seed_changes_membership_without_changing_sampled_object_poses(self):
        changed = deepcopy(self.settings); changed["split_seed"] += 1
        plans, assignments = build_collection_plan(changed, self.config)
        self.assertNotEqual(assignments, self.assignments)
        self.assertEqual(Counter(assignments.values()), Counter(self.assignments.values()))
        for original, other in zip(self.plans, plans):
            np.testing.assert_array_equal(object_pose(original), object_pose(other))
        # Modes are assigned after splitting, so changed family membership may
        # intentionally change mode/action policy while preserving initial draws.
        self.assertTrue(any(a["mode"] != b["mode"]
                            for a, b in zip(self.plans, plans)))

    def test_rejected_supported_parameter_errors_cannot_produce_a_plan(self):
        invalid = []
        for key, value in (("purpose", "final-evaluation"),
                           ("modes", list(reversed(MODES))),
                           ("actions_per_episode", 15)):
            settings = deepcopy(self.settings); settings[key] = value
            invalid.append(settings)
        for key, value in (("train", 0), ("validation", 6), ("test", 8)):
            settings = deepcopy(self.settings)
            settings["split_counts"][key] = value
            invalid.append(settings)
        for key, value in (("forward_speed_min_m_s", -.01),
                           ("forward_speed_max_m_s", .04),
                           ("eccentric_offset_abs_min_m", 0),
                           ("eccentric_offset_abs_max_m", .03)):
            settings = deepcopy(self.settings); settings["commands"][key] = value
            invalid.append(settings)
        for index, settings in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(DataContractError):
                build_collection_plan(settings, self.config)

    def test_action_numbers_and_rng_seeds_must_be_finite_and_well_typed(self):
        for key in ("retreat_speed_m_s", "noncontact_lateral_speed_m_s",
                    "noncontact_return_speed_m_s", "central_tangent_abs_max_m_s",
                    "eccentric_tangent_abs_max_m_s"):
            for value in (float("nan"), float("inf"), -.01, 0., True):
                with self.subTest(parameter=key, value=value):
                    settings = deepcopy(self.settings); settings["commands"][key] = value
                    with self.assertRaises(DataContractError):
                        build_collection_plan(settings, self.config)
        for key in ("initial_seed", "split_seed", "action_seed"):
            for value in (-1, 1.5, True, "20261007"):
                with self.subTest(parameter=key, value=value):
                    settings = deepcopy(self.settings); settings[key] = value
                    with self.assertRaises(DataContractError):
                        build_collection_plan(settings, self.config)

    def test_policy_declarations_cannot_silently_disagree_with_generation(self):
        changes = [(None, "initial_sampling", "fixed_origin"),
                   (None, "family_rule", "each repetition is a new family"),
                   (None, "mode_assignment", "assign after observing failures"),
                   ("commands", "blocks", "feedback each step"),
                   ("commands", "eccentric_sign_rule", "always_positive"),
                   ("commands", "noncontact_phases", ["forward:16"]),
                   ("commands", "contact_transition_phases", ["forward:16"]),
                   ("limits", "feedback_collection_enabled", True),
                   ("limits", "includes_p2_01_stress_fixture", True),
                   ("limits", "final_evaluation_allowed", True)]
        for group, key, value in changes:
            with self.subTest(group=group, parameter=key):
                settings = deepcopy(self.settings)
                target = settings if group is None else settings[group]
                target[key] = value
                with self.assertRaises(DataContractError):
                    build_collection_plan(settings, self.config)

    def test_missing_generation_definition_is_a_contract_error(self):
        for group, key in ((None, "initial_seed"), (None, "mode_assignment"),
                           (None, "split_counts"), (None, "family_rule"),
                           ("commands", "retreat_speed_m_s"),
                           ("commands", "contact_transition_phases"),
                           ("limits", "final_evaluation_allowed")):
            with self.subTest(group=group, parameter=key):
                settings = deepcopy(self.settings)
                target = settings if group is None else settings[group]
                del target[key]
                with self.assertRaises(DataContractError):
                    build_collection_plan(settings, self.config)

    def test_two_real_recordings_preserve_all_actions_and_qualify_contact_changes(self):
        env = PlanarPushEnv(); contract = make_contract(env)
        for mode in ("central", "contact_transition"):
            with self.subTest(mode=mode):
                p = next(plan for plan in self.plans if plan["mode"] == mode)
                env.reset(InitialSpec(**p["initial_spec"]))
                recorder = EpisodeRecorder(
                    env, episode_id=p["episode_id"], episode_index=p["episode_index"],
                    family_id=p["family_id"], family_definition=p["family_definition"],
                    split=p["split"], collection_policy={"name": "plan-test", "mode": mode},
                    seed=self.settings["action_seed"], contract=contract)
                for action in p["actions"]:
                    result = recorder.step(action)
                    self.assertTrue(result.transition_valid)
                    self.assertFalse(recorder.invalid)
                episode = recorder.finish()
                review = validate_episode(episode, contract)
                self.assertEqual(review["transitions"], 16)
                self.assertEqual(review["learning_transitions"], 16)
                self.assertEqual(review["excluded_transitions"], 0)
                self.assertEqual(episode.arrays["state"].shape, (17, 10))
                np.testing.assert_array_equal(episode.arrays["action_requested"], p["actions"])
                np.testing.assert_array_equal(episode.arrays["action_applied"], p["actions"])
                np.testing.assert_allclose(episode.arrays["actual_dt"], .1,
                                           rtol=0, atol=contract["dt_tolerance_s"])
                self.assertEqual(np.flatnonzero(episode.arrays["terminal"]).tolist(), [15])
                for horizon in (1, 5, 10):
                    windows = sequence_windows(episode, horizon, contract)
                    self.assertEqual(len(windows), 17 - horizon)
                    # The last action is retained in learning, with its next state.
                    np.testing.assert_array_equal(windows[-1].actions[-1], p["actions"][-1])
                    np.testing.assert_array_equal(windows[-1].states[-1], episode.arrays["state"][-1])
                contact = episode.arrays["contact_summary"]
                self.assertEqual(contact[0, 0], 0)
                self.assertTrue(np.any(contact[:5, 1] == 1))
                if mode == "contact_transition":
                    self.assertTrue(np.any(contact[5:8, 2] == 0))
                    self.assertTrue(np.any(contact[8:, 0] == 0))
                    self.assertEqual(contact[-1, 0], 1)
                    self.assertEqual(contact[-1, 1], 1)
                else:
                    self.assertTrue(np.all(contact[2:, 2] == 1))


if __name__ == "__main__":
    unittest.main()
