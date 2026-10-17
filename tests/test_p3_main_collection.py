"""Independent P3-02 sampling, lineage and applied-policy contracts.

Pure plan tests do not establish contact coverage or physical validity. Policy
witnesses check what the collector sends, without claiming alignment success.
"""
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import json
import tomllib
import unittest
from unittest.mock import patch

import numpy as np
import mujoco

from push_align.config import ROOT, load_pilot_config
from push_align.data import DataContractError, split_families
from push_align.geometry_control import GeometryContext, GeometryControlError, GeometryDecision
from push_align.main_collection import (
    MODES, MainCollectionPolicy, build_main_collection_plan,
)


OPEN_MODES = ("noncontact", "central", "eccentric", "contact_transition")
FEEDBACK_MODES = ("geometry_feedback", "geometry_perturbed")


def pose(plan):
    return np.asarray([plan["initial_spec"][key]
                       for key in ("object_x", "object_y", "object_yaw")])


def local_commands(plan):
    yaw = plan["initial_spec"]["object_yaw"]
    normal = np.asarray([np.cos(yaw), np.sin(yaw)])
    tangent = np.asarray([-np.sin(yaw), np.cos(yaw)])
    return np.asarray(plan["actions"]) @ np.column_stack([normal, tangent])


class MainCollectionPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = tomllib.loads((ROOT/"configs/p3_collection.v1.toml").read_text())
        cls.config = load_pilot_config()
        cls.plans, cls.assignments = build_main_collection_plan(cls.settings, cls.config)

    def test_main_development_contains_ninety_unique_families_and_no_test(self):
        self.assertEqual(tuple(MODES), OPEN_MODES + FEEDBACK_MODES)
        self.assertEqual(len(self.plans), 90)
        self.assertEqual(Counter(self.assignments.values()), {"train": 60, "validation": 30})
        self.assertEqual([p["episode_index"] for p in self.plans], list(range(90)))
        self.assertEqual(len({p["episode_id"] for p in self.plans}), 90)
        self.assertEqual({p["family_id"] for p in self.plans}, set(self.assignments))
        self.assertTrue(all(p["split"] in ("train", "validation") for p in self.plans))
        self.assertEqual(len({tuple(pose(p)) for p in self.plans}), 90)

    def test_family_lineage_and_seeded_assignment_agree_before_physics(self):
        expected = split_families([p["family_id"] for p in self.plans],
                                  {"train": 60, "validation": 30}, self.settings["split_seed"])
        self.assertEqual(self.assignments, expected)
        for p in self.plans:
            self.assertEqual(p["split"], expected[p["family_id"]])
            self.assertEqual(p["family_definition"]["base_initial_spec"], p["initial_spec"])
            self.assertEqual(p["family_definition"]["variant_rule"], self.settings["family_rule"])

    def test_each_split_balances_all_six_modes_before_observation(self):
        for split, n in (("train", 10), ("validation", 5)):
            members = sorted((p for p in self.plans if p["split"] == split),
                             key=lambda p: p["family_id"])
            self.assertEqual(Counter(p["mode"] for p in members), {mode: n for mode in MODES})
            self.assertEqual([p["mode"] for p in members], list(MODES)*n)

    def test_eccentric_offsets_cover_both_signs_and_chosen_rear_face(self):
        for split in ("train", "validation"):
            offsets = [p["pusher_tangent_offset_m"] for p in self.plans
                       if p["split"] == split and p["mode"] == "eccentric"]
            self.assertEqual(Counter(np.sign(offsets)),
                             {1.: 5, -1.: 5} if split == "train" else {1.: 3, -1.: 2})
            self.assertTrue(all(.010 <= abs(value) <= .015 for value in offsets))
        self.assertTrue(all(p["pusher_tangent_offset_m"] == 0 for p in self.plans
                            if p["mode"] != "eccentric"))

    def test_initial_ranges_and_rear_geometry_use_world_rotation(self):
        bounds = self.config["initial_conditions"]
        # v4 fixture: half length .04, cylinder radius .008, reset gap .003.
        expected_rear_distance = .051
        for p in self.plans:
            s = p["initial_spec"]
            for key, interval in (("object_x", "object_x_range_m"),
                                  ("object_y", "object_y_range_m"),
                                  ("object_yaw", "object_yaw_range_rad")):
                self.assertGreaterEqual(s[key], bounds[interval][0])
                self.assertLessEqual(s[key], bounds[interval][1])
            angle = s["object_yaw"]
            d = np.array([np.cos(angle), np.sin(angle)])
            t = np.array([-np.sin(angle), np.cos(angle)])
            relative = np.array([s["pusher_x"]-s["object_x"], s["pusher_y"]-s["object_y"]])
            self.assertAlmostEqual(relative @ d, -expected_rear_distance, places=14)
            self.assertAlmostEqual(relative @ t, p["pusher_tangent_offset_m"], places=14)

    def test_open_commands_have_sixteen_ticks_and_point_zero_four_norm_cap(self):
        for p in self.plans:
            if p["mode"] not in OPEN_MODES:
                continue
            commands = np.asarray(p["actions"])
            self.assertEqual(commands.shape, (16, 2))
            self.assertEqual(p["max_ticks"], 16)
            self.assertEqual(len(p["phases"]), 16)
            self.assertTrue(np.isfinite(commands).all())
            self.assertLessEqual(float(np.linalg.norm(commands, axis=1).max()), .04 + 1e-15)
            self.assertEqual(p["perturbation_vectors_m_s"], [])

    def test_open_phase_order_has_retreat_and_true_partial_return(self):
        for p in self.plans:
            if p["mode"] not in OPEN_MODES:
                continue
            local = local_commands(p)
            if p["mode"] == "noncontact":
                self.assertEqual(p["phases"], ["zero"]*2 + ["retreat"]*4 + ["lateral"]*4
                                 + ["partial_return"]*4 + ["zero"]*2)
                np.testing.assert_allclose(local[[0, 1, 14, 15]], 0, atol=1e-16)
                self.assertTrue(np.all(local[2:6, 0] < 0))
                np.testing.assert_allclose(local[6:10, 0], 0, atol=1e-16)
                self.assertTrue(np.all(local[10:14, 0] > 0))
                self.assertLess(local[:, 0].sum(), 0)
            elif p["mode"] == "contact_transition":
                self.assertEqual(p["phases"], ["forward"]*5 + ["retreat"]*3 + ["reapproach"]*8)
                self.assertTrue(np.all(local[:5, 0] > 0))
                self.assertTrue(np.all(local[5:8, 0] < 0))
                self.assertTrue(np.all(local[8:, 0] > 0))
                np.testing.assert_allclose(local[:, 1], 0, atol=1e-16)
                np.testing.assert_array_equal(p["actions"][:5], [p["actions"][0]]*5)
                np.testing.assert_array_equal(p["actions"][8:], [p["actions"][0]]*8)
            else:
                self.assertTrue(np.all(local[:, 0] > 0))
                for start in (0, 4, 8, 12):
                    np.testing.assert_array_equal(p["actions"][start:start+4],
                                                  [p["actions"][start]]*4)

    def test_feedback_commands_are_unresolved_until_current_observation(self):
        for p in self.plans:
            if p["mode"] in FEEDBACK_MODES:
                self.assertEqual(p["max_ticks"], 128)
                self.assertEqual(p["actions"], [])
                self.assertEqual(p["phases"], [])
            if p["mode"] != "geometry_perturbed":
                self.assertEqual(p["perturbation_vectors_m_s"], [])

    def test_perturbation_schedule_is_frozen_world_xy_four_tick_blocks(self):
        for p in self.plans:
            if p["mode"] != "geometry_perturbed":
                continue
            values = np.asarray(p["perturbation_vectors_m_s"])
            self.assertEqual(values.shape, (128, 2))
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue(np.any(values != 0))
            for start in range(0, 128, 4):
                np.testing.assert_array_equal(values[start:start+4],
                                              np.repeat(values[start:start+1], 4, axis=0))
            self.assertGreater(len({tuple(v) for v in values}), 1)

    def test_noise_rng_key_and_first_blocks_reproduce_declared_gaussian_stream(self):
        for p in self.plans:
            self.assertEqual(p["action_rng_key"], [2026100803, p["episode_index"]])
            if p["mode"] != "geometry_perturbed":
                continue
            # Contract-level known key; the expected stream is independent of
            # policy eligibility and of any number of prior executed episodes.
            rng = np.random.Generator(np.random.PCG64([2026100803, p["episode_index"]]))
            expected = rng.normal(0., .003, size=(2, 2))
            values = np.asarray(p["perturbation_vectors_m_s"])
            np.testing.assert_array_equal(values[[0, 4]], expected)

    def test_plan_ids_never_alias_pilot_lineage(self):
        self.assertEqual([p["episode_id"] for p in self.plans],
                         [f"main-episode-{i:03d}" for i in range(90)])
        self.assertEqual([p["family_id"] for p in self.plans],
                         [f"main-family-{i:03d}" for i in range(90)])

    def test_plan_building_is_reproducible_without_mutating_original_contracts(self):
        settings, config = deepcopy(self.settings), deepcopy(self.config)
        before_settings, before_config = deepcopy(settings), deepcopy(config)
        self.assertEqual(build_main_collection_plan(settings, config), (self.plans, self.assignments))
        self.assertEqual(build_main_collection_plan(settings, config), (self.plans, self.assignments))
        self.assertEqual(settings, before_settings)
        self.assertEqual(config, before_config)
        self.assertEqual(self.config["pusher"]["command_speed_norm_max_m_s"], .06)

    def test_initial_seed_changes_pose_without_changing_local_commands_or_noise(self):
        settings = deepcopy(self.settings); settings["initial_seed"] += 1
        other, assignments = build_main_collection_plan(settings, self.config)
        self.assertEqual(assignments, self.assignments)
        self.assertTrue(any(not np.array_equal(pose(a), pose(b)) for a, b in zip(self.plans, other)))
        for a, b in zip(self.plans, other):
            self.assertEqual(a["mode"], b["mode"])
            self.assertEqual(a["pusher_tangent_offset_m"], b["pusher_tangent_offset_m"])
            self.assertEqual(a["perturbation_vectors_m_s"], b["perturbation_vectors_m_s"])
            if a["mode"] in OPEN_MODES:
                np.testing.assert_allclose(local_commands(a), local_commands(b), atol=2e-17, rtol=0)

    def test_action_seed_changes_commands_without_changing_initial_object_pose_or_split(self):
        settings = deepcopy(self.settings); settings["action_seed"] += 1
        other, assignments = build_main_collection_plan(settings, self.config)
        self.assertEqual(assignments, self.assignments)
        for a, b in zip(self.plans, other):
            np.testing.assert_array_equal(pose(a), pose(b))
            self.assertEqual(a["mode"], b["mode"])
        self.assertTrue(any(a["actions"] != b["actions"] for a, b in zip(self.plans, other)))
        self.assertTrue(any(a["perturbation_vectors_m_s"] != b["perturbation_vectors_m_s"]
                            for a, b in zip(self.plans, other)))

    def test_split_seed_changes_membership_but_never_regenerates_object_poses(self):
        settings = deepcopy(self.settings); settings["split_seed"] += 1
        other, assignments = build_main_collection_plan(settings, self.config)
        self.assertNotEqual(assignments, self.assignments)
        self.assertEqual(Counter(assignments.values()), {"train": 60, "validation": 30})
        for a, b in zip(self.plans, other):
            np.testing.assert_array_equal(pose(a), pose(b))

    def test_final_test_and_wrong_counts_are_rejected_instead_of_silent_reassignment(self):
        for counts in ({"train": 60, "validation": 30, "test": 1},
                       {"train": 60, "validation": 24}, {"train": 0, "validation": 30},
                       {"train": True, "validation": 30}):
            settings = deepcopy(self.settings); settings["split_counts"] = counts
            with self.subTest(counts=counts), self.assertRaises(DataContractError):
                build_main_collection_plan(settings, self.config)

    def test_declared_modes_and_fixed_tick_lengths_cannot_silently_drift(self):
        for key, value in (("modes", list(reversed(MODES))),
                           ("purpose", "final-evaluation"), ("actions_per_episode", 15),
                           ("feedback_max_ticks", 127), ("initial_seed", True),
                           ("action_seed", -1), ("split_seed", 1.5)):
            settings = deepcopy(self.settings); settings[key] = value
            with self.subTest(key=key), self.assertRaises(DataContractError):
                build_main_collection_plan(settings, self.config)

    def test_invalid_commands_offsets_and_perturbation_numbers_are_rejected(self):
        for section, key, values in (
                ("commands", "retreat_speed_m_s", (True, float("nan"), -.01)),
                ("commands", "forward_speed_min_m_s", (.05, 0.)),
                ("commands", "forward_speed_max_m_s", (.061, float("inf"))),
                ("commands", "command_support_norm_max_m_s", (.06, .03)),
                ("commands", "eccentric_offset_abs_max_m", (.04,)),
                ("perturbation", "block_steps", (True, 3, 4.0)),
                ("perturbation", "world_xy_std_m_s", (.004, True, float("nan"))),
                ("perturbation", "holding_noise_enabled", (True,)),
                ("perturbation", "apply_mode", ("all_modes",))):
            for value in values:
                settings = deepcopy(self.settings); settings[section][key] = value
                with self.subTest(section=section, key=key, value=value), self.assertRaises(DataContractError):
                    build_main_collection_plan(settings, self.config)

    def test_false_noise_coordinate_or_generation_declaration_is_rejected(self):
        for key, value in (("frame", "object_body_xy"),
                           ("generation", "random_draw_during_each_push")):
            settings = deepcopy(self.settings); settings["perturbation"][key] = value
            with self.subTest(key=key), self.assertRaises(DataContractError):
                build_main_collection_plan(settings, self.config)

    def test_missing_lineage_and_feedback_or_final_test_policy_drift_are_rejected(self):
        for section, key, value in ((None, "family_rule", "new_family_for_each_repeat"),
                                    ("limits", "feedback_collection_enabled", False),
                                    ("limits", "final_evaluation_allowed", True),
                                    ("limits", "includes_p2_01_stress_fixture", True)):
            settings = deepcopy(self.settings)
            if section:
                settings[section][key] = value
            else:
                settings[key] = value
            with self.subTest(section=section, key=key), self.assertRaises(DataContractError):
                build_main_collection_plan(settings, self.config)
        for key in ("initial_sampling", "family_rule", "mode_assignment", "perturbation"):
            settings = deepcopy(self.settings); settings.pop(key)
            with self.subTest(missing=key), self.assertRaises(DataContractError):
                build_main_collection_plan(settings, self.config)


class MainCollectionPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = tomllib.loads((ROOT/"configs/p3_collection.v1.toml").read_text())
        cls.config = load_pilot_config()
        cls.plans, _ = build_main_collection_plan(cls.settings, cls.config)
        cls.selected = json.loads((ROOT/"configs/p3_geometry_selected.v1.json").read_text())

    def policy(self, mode="geometry_perturbed", *, noise=None):
        plan = deepcopy(next(p for p in self.plans if p["mode"] == mode))
        if noise is not None:
            plan["perturbation_vectors_m_s"] = [list(noise)]*128
        return MainCollectionPolicy(self.config, self.settings, plan, self.selected)

    def state(self, *, goal=False):
        return np.array([.08 if goal else 0., 0., 0., 0., 0., 0.,
                         .029 if goal else -.051, 0., 0., 0.], np.float64)

    def fake_g(self, *, mode="PUSH", action=(.03, 0.), feasible=True):
        context = GeometryContext(mode=mode, locked_offset_m=0., retreat_offset_m=0.)
        return GeometryDecision(np.array(action, np.float64), context,
                                {"feasible": feasible, "reason": "command",
                                 "mode_before": "APPROACH", "mode_after": mode,
                                 "raw_contact_labels_used": False, "future_physics_queries": 0})

    def test_only_feasible_push_adds_predeclared_world_noise(self):
        policy = self.policy(noise=(.003, .004))
        state = self.state(); state[2] = .3
        with patch.object(policy.controller, "propose", return_value=self.fake_g()):
            decision = policy.propose(state, policy.initial_context(), 0., 0)
        # World-axis addition: no rotation by observed object yaw.
        np.testing.assert_allclose(decision.action, [.033, .004], atol=1e-16)
        self.assertTrue(decision.diagnostics["noise_applied"])
        self.assertFalse(decision.diagnostics["saturated"])

    def test_noise_limit_scales_the_vector_rather_than_clipping_each_component(self):
        policy = self.policy(noise=(.03, .04))
        with patch.object(policy.controller, "propose", return_value=self.fake_g(action=(0., 0.))):
            decision = policy.propose(self.state(), policy.initial_context(), 0., 0)
        # 3-4-5 triangle: .05 -> .04 keeps direction and yields .024/.032.
        np.testing.assert_allclose(decision.action, [.024, .032], atol=1e-16)
        self.assertAlmostEqual(np.linalg.norm(decision.action), .04, places=14)
        self.assertTrue(decision.diagnostics["saturated"])
        np.testing.assert_allclose(decision.diagnostics["applied_noise_delta_m_s"],
                                   [.024, .032], atol=1e-16)
        self.assertFalse(np.array_equal(decision.diagnostics["applied_noise_delta_m_s"],
                                        decision.diagnostics["predeclared_noise_m_s"]))

    def test_retract_slide_and_approach_are_not_perturbed(self):
        policy = self.policy(noise=(.003, .004))
        for mode in ("RETRACT", "SLIDE", "APPROACH"):
            with self.subTest(mode=mode), patch.object(policy.controller, "propose",
                                                      return_value=self.fake_g(mode=mode)):
                decision = policy.propose(self.state(), policy.initial_context(), 0., 0)
                np.testing.assert_array_equal(decision.action, [.03, 0.])
                self.assertFalse(decision.diagnostics["noise_applied"])
                np.testing.assert_array_equal(decision.diagnostics["applied_noise_delta_m_s"], [0., 0.])
                np.testing.assert_array_equal(decision.diagnostics["predeclared_noise_m_s"], [.003, .004])

    def test_infeasible_push_is_not_made_nonzero_by_noise(self):
        policy = self.policy(noise=(.003, .004))
        with patch.object(policy.controller, "propose",
                          return_value=self.fake_g(action=(0., 0.), feasible=False)):
            decision = policy.propose(self.state(), policy.initial_context(), 0., 0)
        np.testing.assert_array_equal(decision.action, [0., 0.])
        self.assertFalse(decision.diagnostics["noise_applied"])
        self.assertFalse(decision.diagnostics["feasible"])

    def test_unperturbed_feedback_passes_g_command_unchanged(self):
        policy = self.policy("geometry_feedback")
        with patch.object(policy.controller, "propose", return_value=self.fake_g(action=(.01, -.02))):
            decision = policy.propose(self.state(), policy.initial_context(), 0., 0)
        np.testing.assert_array_equal(decision.action, [.01, -.02])
        self.assertFalse(decision.diagnostics["noise_applied"])

    def test_holding_clears_g_contact_memory_and_commands_zero_despite_velocity(self):
        policy = self.policy(noise=(.003, .004))
        old = replace(policy.initial_context(), geometry_context=GeometryContext(
            mode="PUSH", locked_offset_m=.01, retreat_offset_m=.01,
            reassessment_count=2, saturation_count=1))
        value = self.state(goal=True); value[3] = .05; value[5] = .2
        decision = policy.propose(value, old, .4, 4)
        np.testing.assert_array_equal(decision.action, [0., 0.])
        self.assertEqual(decision.context.geometry_context.mode, "RETRACT")
        self.assertIsNone(decision.context.geometry_context.locked_offset_m)
        self.assertIsNone(decision.context.geometry_context.retreat_offset_m)
        self.assertTrue(decision.diagnostics["holding"])
        self.assertFalse(decision.diagnostics["noise_applied"])
        np.testing.assert_array_equal(decision.diagnostics["predeclared_noise_m_s"], [.003, .004])
        np.testing.assert_array_equal(decision.diagnostics["applied_noise_delta_m_s"], [0., 0.])
        self.assertEqual(old.geometry_context.mode, "PUSH")
        # Holding request alone does not claim physical readiness.
        self.assertNotIn("ready", decision.diagnostics)

    def test_holding_cannot_hide_backwards_time_or_invalid_g_context(self):
        policy = self.policy()
        value = self.state(goal=True)
        for geometry in (GeometryContext(mode="PUSH", locked_offset_m=0., retreat_offset_m=0.,
                                         phase_started_s=.2, last_elapsed_s=.2),
                         GeometryContext(mode="PUSH", locked_offset_m=.2, retreat_offset_m=0.)):
            context = replace(policy.initial_context(), geometry_context=geometry)
            with self.subTest(geometry=geometry), self.assertRaises(GeometryControlError):
                policy.propose(value, context, .1, 1)

    def test_noise_uses_absolute_tick_even_after_earlier_ineligible_recovery(self):
        policy = self.policy()
        first = np.asarray(policy.plan["perturbation_vectors_m_s"][0])
        later = np.asarray(policy.plan["perturbation_vectors_m_s"][4])
        self.assertFalse(np.array_equal(first, later))
        with patch.object(policy.controller, "propose", return_value=self.fake_g(mode="RETRACT")):
            policy.propose(self.state(), policy.initial_context(), 0., 0)
        with patch.object(policy.controller, "propose", return_value=self.fake_g(action=(.01, 0.))):
            decision = policy.propose(self.state(), policy.initial_context(), .4, 4)
        np.testing.assert_allclose(decision.action, [.01, 0.] + later, atol=1e-16)
        np.testing.assert_array_equal(decision.diagnostics["predeclared_noise_m_s"], later)

    def test_context_keeps_actual_previous_command_until_runner_replaces_it(self):
        policy = self.policy("central")
        actual = np.array([.006, -.008], np.float64)
        context = replace(policy.initial_context(), previous_applied_action=actual)
        state = self.state(); original = state.copy()
        decision = policy.propose(state, context, 0., 0)
        np.testing.assert_array_equal(decision.context.previous_applied_action, actual)
        np.testing.assert_array_equal(decision.diagnostics["previous_applied_action"], actual)
        self.assertFalse(np.array_equal(decision.action, actual))
        np.testing.assert_array_equal(state, original)
        np.testing.assert_array_equal(context.previous_applied_action, actual)
        self.assertFalse(decision.action.flags.writeable)
        self.assertFalse(decision.context.previous_applied_action.flags.writeable)
        self.assertFalse(np.shares_memory(actual, decision.context.previous_applied_action))

    def test_g_gets_current_state_task_own_context_and_elapsed_only(self):
        policy = self.policy("geometry_feedback")
        state = self.state(); context = replace(policy.initial_context(),
                    previous_applied_action=np.array([.059, 0.], np.float64))
        with patch.object(policy.controller, "propose", return_value=self.fake_g()) as callback:
            policy.propose(state, context, .2, 2)
        args, kwargs = callback.call_args
        self.assertEqual(len(args), 4)
        self.assertIs(args[0], state)
        self.assertIs(args[1], policy.task)
        self.assertIs(args[2], context.geometry_context)
        self.assertEqual(args[3], .2)
        self.assertEqual(kwargs, {})

    def test_real_feedback_computes_without_engine_or_rng_access(self):
        policy = self.policy()
        with ExitStack() as stack:
            for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse"):
                stack.enter_context(patch.object(mujoco, name, side_effect=AssertionError("Future physics accessed")))
            stack.enter_context(patch.object(np.random, "Generator", side_effect=AssertionError("Execution drew RNG")))
            decision = policy.propose(self.state(), policy.initial_context(), 0., 0)
        self.assertFalse(decision.diagnostics["raw_contact_labels_used"])
        self.assertEqual(decision.diagnostics["future_physics_queries"], 0)

    def test_constructor_copies_plan_settings_physics_and_selected_parameters(self):
        settings, config, selected = deepcopy(self.settings), deepcopy(self.config), deepcopy(self.selected)
        plan = deepcopy(next(p for p in self.plans if p["mode"] == "geometry_feedback"))
        original = deepcopy((settings, config, selected, plan))
        policy = MainCollectionPolicy(config, settings, plan, selected)
        self.assertEqual((settings, config, selected, plan), original)
        first = policy.propose(self.state(), policy.initial_context(), 0., 0)
        settings["commands"]["command_support_norm_max_m_s"] = .001
        config["geometry_baseline"]["position_gain_per_s"] = .01
        selected["position_gain_per_s"] = .01
        plan["mode"] = "central"
        again = policy.propose(self.state(), policy.initial_context(), 0., 0)
        np.testing.assert_array_equal(first.action, again.action)

    def test_open_loop_returns_frozen_command_by_tick_without_holding_substitution(self):
        policy = self.policy("central")
        # Open-loop excitation may deliberately continue at a goal pose; only
        # feedback modes implement holding. This is data, not task evaluation.
        decision = policy.propose(self.state(goal=True), policy.initial_context(), .4, 4)
        np.testing.assert_array_equal(decision.action, policy.plan["actions"][4])
        self.assertFalse(decision.diagnostics["holding"])
        self.assertEqual(decision.diagnostics["phase"], "forward_block_1")

    def test_bad_state_tick_elapsed_or_previous_applied_command_is_rejected(self):
        policy = self.policy()
        context = policy.initial_context()
        cases = ((np.zeros(9, np.float64), context, 0., 0),
                 (self.state().astype(np.float32), context, 0., 0),
                 (self.state(), context, -1., 0), (self.state(), context, True, 0),
                 (self.state(), context, 0., True), (self.state(), context, 0., -1),
                 (self.state(), context, 0., 128),
                 (self.state(), replace(context, previous_applied_action=np.array([.07, 0.], np.float64)), 0., 0))
        for state, ctx, elapsed, tick in cases:
            with self.subTest(elapsed=elapsed, tick=tick), self.assertRaises(DataContractError):
                policy.propose(state, ctx, elapsed, tick)

    def test_malformed_action_or_noise_array_is_rejected_before_execution(self):
        for mode, key, value in (("central", "actions", [[.05, 0.]]*16),
                                 ("central", "phases", []),
                                 ("geometry_feedback", "actions", [[0., 0.]]),
                                 ("geometry_perturbed", "perturbation_vectors_m_s", [[0., 0.]]*127)):
            plan = deepcopy(next(p for p in self.plans if p["mode"] == mode)); plan[key] = value
            with self.subTest(mode=mode, key=key), self.assertRaises(DataContractError):
                MainCollectionPolicy(self.config, self.settings, plan, self.selected)
        plan = deepcopy(next(p for p in self.plans if p["mode"] == "geometry_perturbed"))
        plan["perturbation_vectors_m_s"][1][0] += .001
        with self.assertRaises(DataContractError):
            MainCollectionPolicy(self.config, self.settings, plan, self.selected)


if __name__ == "__main__":
    unittest.main()
