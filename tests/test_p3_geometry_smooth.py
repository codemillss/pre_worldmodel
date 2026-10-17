"""Independent command and information-boundary checks for the opt-in G policy.

These fixtures verify prescribed units, limiting behaviour and current-state
geometry. They cannot establish numerical convergence of physical execution;
the six frozen dt representatives and four task cases remain separate gates.
"""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import math
import json
import tomllib
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from push_align.config import ROOT, load_pilot_config
from push_align.data import DataContractError
from push_align.geometry_control import (
    GeometryContext, GeometryControlError, GeometryController, GeometryDecision,
    PARAMETER_NAMES,
)
from push_align.geometry_control_smooth import SmoothGeometryController
from push_align.planning import TaskSpec
from push_align.main_collection import MainCollectionPolicy, build_main_collection_plan


SPEC = {
    "schema_version": "geometry-transition-v1",
    "offset_taper_start_m": .002,
    "phase_end_taper_s": .4,
    "phase_entry_ramp_s": .4,
    "slide_blend_outer_tolerance_multiplier": 2.,
    "free_motion_speed_max_m_s": .01,
}


NOISE_WEIGHTING = (
    "same predeclared PUSH-only vector times continuous phase entry and "
    "pre-reassessment gain; zero in holding/free modes"
)

SPEC_V2 = {**SPEC, "schema_version": "geometry-transition-v2", "clock_snap_tolerance_s": 1e-9}


# Numerically frozen final observed states/context from the four unsuccessful
# p3-01-tuning-20261009-01 cases. The legacy rejection is retained as evidence;
# this unit reproduction checks route preparation, not a successful episode.
STALE_RETRACT_CASES = (
    ("g-translation", [
        .05403532487812617, -3.047449009634678e-08, 3.586340819428102e-06,
        -2.8371887348563293e-06, 3.751342359415268e-11, -1.5297568910636393e-10,
        .0060353296007492644, 4.438879720144088e-09,
        -3.173102452567125e-05, -1.670585208064499e-10],
        dict(mode="RETRACT", locked_offset_m=4.19067988234951e-07,
             retreat_offset_m=2.071048752989222e-07,
             phase_started_s=3.9000000000003685, last_elapsed_s=3.9000000000003685,
             reassessment_count=2, saturation_count=26), 4.0000000000003135),
    ("g-yaw-positive", [
        .04136671276359119, -.0007956974146242943, .09563197813067514,
        -2.5625915861034675e-07, 2.084985001692397e-07, -2.7763272454085757e-06,
        -.007568245264368556, .006656351966931674,
        -9.696739600102878e-06, -7.091505908411154e-07],
        dict(mode="RETRACT", locked_offset_m=.011611971464054639,
             retreat_offset_m=.012090759687342141,
             phase_started_s=5.299999999999593, last_elapsed_s=5.299999999999593,
             reassessment_count=2, saturation_count=70), 5.399999999999538),
    ("g-yaw-negative", [
        .04136659086013217, .0007957050043237907, -.09563205389355334,
        -2.5626850016117053e-07, -2.0850456397262096e-07, 2.776393079690346e-06,
        -.007568367804650773, -.00665634149775662,
        -9.696930491228388e-06, 7.091724314141938e-07],
        dict(mode="RETRACT", locked_offset_m=-.011611976012625095,
             retreat_offset_m=-.012090760527477248,
             phase_started_s=5.299999999999593, last_elapsed_s=5.299999999999593,
             reassessment_count=2, saturation_count=70), 5.399999999999538),
    ("g-recontact", [
        .07241361763302984, -3.4931149908162908e-06, -4.575419460906005e-05,
        -1.9904983407586867e-06, 1.8391106410091494e-09, -1.3070795967092055e-08,
        .02441378004480502, 1.956465623587451e-05,
        -2.8152912749772056e-05, 3.614645307028455e-09],
        dict(mode="RETRACT", locked_offset_m=-2.4400954526143368e-05,
             retreat_offset_m=2.0861200533786238e-05,
             phase_started_s=10.200000000001392, last_elapsed_s=10.200000000001392,
             reassessment_count=3, saturation_count=52), 10.300000000001514),
)


def state(*, x=0., y=0., yaw=0., wz=0., qx=-.049, qy=0., vx=0., vy=0.):
    """Body-frame rear-side positions; no contact label is part of state10."""
    angle = .5*math.atan2(math.sin(2*yaw), math.cos(2*yaw))
    d = np.array([math.cos(angle), math.sin(angle)], np.float64)
    t = np.array([-d[1], d[0]], np.float64)
    pusher = np.array([x, y]) + qx*d + qy*t
    return np.array([x, y, yaw, 0., 0., wz, *pusher, vx, vy], np.float64)


def active(mode="PUSH", *, offset=0., phase=0., last=0.):
    return GeometryContext(mode=mode, locked_offset_m=offset,
                           retreat_offset_m=0., phase_started_s=phase,
                           last_elapsed_s=last)


class SmoothGeometryContractTest(unittest.TestCase):
    def setUp(self):
        self.config = load_pilot_config()
        self.config["geometry_baseline"].update(
            position_gain_per_s=4., yaw_creep_speed_m_s=.02,
            yaw_offset_gain_m_rad=.09)
        self.spec = deepcopy(SPEC)
        self.task = TaskSpec.from_config(self.config)
        self.control = SmoothGeometryController(self.config, self.spec)

    def propose(self, value=None, *, context=None, elapsed=.6, control=None):
        controller = self.control if control is None else control
        return controller.propose(state() if value is None else value, self.task,
                                  active() if context is None else context, elapsed)

    def test_legacy_parameter_registry_remains_sixteen_and_requires_no_new_keys(self):
        self.assertEqual(len(PARAMETER_NAMES), 16)
        self.assertEqual(set(self.config["geometry_baseline"]), set(PARAMETER_NAMES))
        legacy = GeometryController(self.config)
        decision = legacy.propose(state(wz=1.), self.task, active(), .1)
        self.assertEqual(decision.context.mode, "RETRACT")
        # Original immediate outward release survives in the legacy controller.
        np.testing.assert_allclose(decision.action, [-.033, 0.], atol=1e-14)
        self.assertNotIn("noise_gain", decision.diagnostics)

    def test_opt_in_initial_retract_starts_at_zero_without_altering_legacy(self):
        value = state(qx=-.048)
        new = self.control.propose(value, self.task, self.control.initial_context(), 0.)
        old = GeometryController(self.config).propose(
            value, self.task, GeometryContext(), 0.)
        self.assertEqual(new.context.mode, "RETRACT")
        np.testing.assert_array_equal(new.action, [0., 0.])
        np.testing.assert_allclose(old.action, [-.036, 0.], atol=1e-14)

    def test_full_gain_push_retains_selected_geometric_command(self):
        decision = self.propose()
        np.testing.assert_allclose(decision.action, [.04, 0.], atol=1e-14)
        self.assertAlmostEqual(decision.diagnostics["noise_gain"], 1.)
        self.assertEqual(decision.context.mode, "PUSH")

    def test_offset_three_millimetres_half_tapers_push_and_noise(self):
        # At zero yaw/direction compensation, .01 * .3rad/s = .003m.
        decision = self.propose(state(wz=.3))
        self.assertAlmostEqual(decision.diagnostics["b_target_m"], .003, places=14)
        np.testing.assert_allclose(decision.action, [.02, 0.], atol=1e-14)
        self.assertAlmostEqual(decision.diagnostics["noise_gain"], .5, places=14)
        self.assertEqual(decision.context.mode, "PUSH")

    def test_offset_taper_start_retains_full_gain(self):
        decision = self.propose(state(wz=.2))
        np.testing.assert_allclose(decision.action, [.04, 0.], atol=1e-14)
        self.assertAlmostEqual(decision.diagnostics["noise_gain"], 1., places=14)

    def test_offset_end_zero_precedes_replanning_on_strictly_larger_change(self):
        at = self.propose(state(wz=.4))
        beyond = self.propose(state(wz=.4001))
        self.assertEqual(at.context.mode, "PUSH")
        self.assertEqual(beyond.context.mode, "RETRACT")
        np.testing.assert_array_equal(at.action, [0., 0.])
        np.testing.assert_array_equal(beyond.action, [0., 0.])
        self.assertEqual(at.diagnostics["noise_gain"], 0.)
        self.assertEqual(beyond.diagnostics["noise_gain"], 0.)

    def test_offset_threshold_epsilon_pair_cannot_change_zero_into_large_motion(self):
        eps = 1e-8
        before = self.propose(state(wz=(.004-eps)/.01))
        after = self.propose(state(wz=(.004+eps)/.01))
        self.assertLess(np.linalg.norm(before.action-after.action), 1e-9)
        self.assertLess(abs(before.diagnostics["noise_gain"]
                            - after.diagnostics["noise_gain"]), 1e-9)

    def test_original_offset_failure_pair_is_attenuated_without_hiding_transition(self):
        inside = self.propose(state(wz=.003751953/.01))
        outside = self.propose(state(wz=.004084656/.01))
        self.assertEqual(inside.context.mode, "PUSH")
        self.assertEqual(outside.context.mode, "RETRACT")
        self.assertLess(np.linalg.norm(inside.action-outside.action), .002)
        self.assertGreater(outside.context.reassessment_count, 0)

    def test_phase_entry_halfway_is_half_push(self):
        decision = self.propose(elapsed=.2)
        np.testing.assert_allclose(decision.action, [.02, 0.], atol=1e-14)
        self.assertAlmostEqual(decision.diagnostics["noise_gain"], .5, places=14)

    def test_entry_and_offset_half_gains_multiply_instead_of_adding(self):
        decision = self.propose(state(wz=.3), elapsed=.2)
        np.testing.assert_allclose(decision.action, [.01, 0.], atol=1e-14)
        self.assertAlmostEqual(decision.diagnostics["noise_gain"], .25, places=14)

    def test_phase_end_halfway_is_half_push(self):
        decision = self.propose(elapsed=1.8)
        np.testing.assert_allclose(decision.action, [.02, 0.], atol=1e-14)
        self.assertAlmostEqual(decision.diagnostics["noise_gain"], .5, places=14)

    def test_phase_timeout_retracts_at_zero_and_records_real_reassessment(self):
        decision = self.propose(elapsed=2.)
        self.assertEqual(decision.context.mode, "RETRACT")
        np.testing.assert_array_equal(decision.action, [0., 0.])
        self.assertGreater(decision.context.reassessment_count, 0)
        self.assertAlmostEqual(decision.context.phase_started_s, 2.)
        self.assertTrue(decision.diagnostics["feasible"])

    def test_phase_timeout_epsilon_pair_has_no_command_jump(self):
        before = self.propose(elapsed=2.-1e-8)
        after = self.propose(elapsed=2.+1e-8)
        self.assertLess(np.linalg.norm(before.action-after.action), 1e-9)

    def test_retract_half_entry_is_half_outward_release(self):
        context = active("RETRACT")
        decision = self.propose(state(qx=-.049), context=context, elapsed=.2)
        np.testing.assert_allclose(decision.action, [-.0165, 0.], atol=1e-14)
        self.assertEqual(decision.diagnostics["noise_gain"], 0.)

    def test_large_contact_change_keeps_tangent_offset_during_release(self):
        context = replace(active(), locked_offset_m=.004, retreat_offset_m=.004)
        value = state(qx=-.048, qy=.004, wz=-1.)
        first = self.propose(value, context=context, elapsed=.6)
        self.assertEqual(first.context.mode, "RETRACT")
        np.testing.assert_array_equal(first.action, [0., 0.])
        later = self.propose(value, context=first.context, elapsed=1.)
        self.assertLess(later.action[0], 0.)
        self.assertAlmostEqual(later.action[1], 0., places=14)
        self.assertAlmostEqual(later.context.retreat_offset_m, .004, places=14)

    def test_slide_and_approach_share_the_same_command_for_same_observation(self):
        # Tangent error .00225m lies halfway in the 1.5--3mm band.
        value = state(qx=-.060, qy=.00225)
        slide = self.propose(value, context=active("SLIDE"))
        approach = self.propose(value, context=active("APPROACH"))
        np.testing.assert_array_equal(slide.action, approach.action)
        expected_waypoint = np.array([-.0545, 0.])
        np.testing.assert_allclose(slide.diagnostics["waypoint_local_xy_m"],
                                   expected_waypoint, atol=1e-14)
        # 3/s feedback on displacement [5.5,-2.25]mm is [.0165,-.00675]
        # m/s before the .01m/s vector-norm cap. Direction is preserved.
        expected = .01*np.array([.0165, -.00675])/math.hypot(.0165, .00675)
        np.testing.assert_allclose(slide.action, expected, atol=1e-14)
        self.assertLessEqual(np.linalg.norm(slide.action), .01+1e-15)
        self.assertEqual(slide.diagnostics["noise_gain"], 0.)

    def test_slide_small_error_uses_approach_and_large_error_retreat_waypoint(self):
        near = self.propose(state(qx=-.060, qy=.001), context=active("SLIDE"))
        far = self.propose(state(qx=-.060, qy=.0035), context=active("SLIDE"))
        np.testing.assert_allclose(near.diagnostics["waypoint_local_xy_m"],
                                   [-.049, 0.], atol=1e-14)
        np.testing.assert_allclose(far.diagnostics["waypoint_local_xy_m"],
                                   [-.060, 0.], atol=1e-14)
        self.assertAlmostEqual(far.action[0], 0., places=14)
        self.assertLess(far.action[1], 0.)

    def test_slide_transition_epsilon_pair_is_command_continuous(self):
        eps = 1e-8
        inside = self.propose(state(qx=-.060, qy=.0015-eps), context=active("SLIDE"))
        outside = self.propose(state(qx=-.060, qy=.0015+eps), context=active("SLIDE"))
        self.assertEqual(inside.context.mode, "APPROACH")
        self.assertEqual(outside.context.mode, "SLIDE")
        self.assertLess(np.linalg.norm(inside.action-outside.action), 1e-7)

    def test_original_tangent_failure_pair_has_small_commands_in_both_phases(self):
        inside = self.propose(state(qx=-.060, qy=.001458944), context=active("SLIDE"))
        outside = self.propose(state(qx=-.060, qy=.001534326), context=active("SLIDE"))
        self.assertEqual(inside.context.mode, "APPROACH")
        self.assertEqual(outside.context.mode, "SLIDE")
        self.assertLess(np.linalg.norm(inside.action-outside.action), .0001)

    def test_holding_zero_has_priority_over_entry_gain_and_velocity_damping(self):
        value = state(x=.08, wz=.2, vx=.05, vy=-.04)
        decision = self.propose(value, context=active(offset=.01), elapsed=.2)
        np.testing.assert_array_equal(decision.action, [0., 0.])
        self.assertEqual(decision.diagnostics["reason"], "holding")
        self.assertEqual(decision.diagnostics["noise_gain"], 0.)
        self.assertEqual(decision.context.mode, "RETRACT")
        self.assertIsNone(decision.context.locked_offset_m)
        self.assertIsNone(decision.context.retreat_offset_m)

    def test_infeasible_outward_goal_cannot_be_made_valid_by_smoothing(self):
        value = state(x=.1)
        decision = self.propose(value)
        np.testing.assert_array_equal(decision.action, [0., 0.])
        self.assertFalse(decision.diagnostics["feasible"])
        self.assertIn("insufficient_inward", decision.diagnostics["reason"])
        self.assertEqual(decision.diagnostics["noise_gain"], 0.)

    def test_front_side_position_is_still_explicitly_infeasible(self):
        decision = self.propose(state(qx=.07), context=GeometryContext())
        self.assertFalse(decision.diagnostics["feasible"])
        np.testing.assert_array_equal(decision.action, [0., 0.])

    def test_free_motion_commands_are_bounded_and_noiseless(self):
        for mode, qx, qy in (("RETRACT", -.049, .004),
                             ("SLIDE", -.060, .004),
                             ("APPROACH", -.055, .001)):
            with self.subTest(mode=mode):
                decision = self.propose(state(qx=qx, qy=qy), context=active(mode))
                bound = .04 if decision.context.mode == "RETRACT" else .01
                self.assertLessEqual(np.linalg.norm(decision.action), bound+1e-15)
                self.assertEqual(decision.diagnostics["noise_gain"], 0.)

    def test_current_command_segment_cannot_cross_expanded_body_from_rear(self):
        # A large outward measured velocity makes velocity damping ask for a
        # forward step, despite the rear waypoint. A safe waypoint alone is
        # therefore insufficient. At x=-.049 the expanded body begins -.048.
        value = state(qx=-.049, qy=.002, vx=-.3)
        decision = self.propose(value, context=active("RETRACT"))
        self.assertFalse(decision.diagnostics["feasible"])
        self.assertFalse(decision.diagnostics["command_endpoint_path_safe"])
        self.assertIn("unsafe_continuous_free_command", decision.diagnostics["reason"])
        np.testing.assert_array_equal(decision.action, [0., 0.])

    def test_shallow_contact_release_discards_inward_and_tangential_damping(self):
        value = state(qx=-.047, qy=.004, vx=-.2, vy=.1)
        context = replace(active("RETRACT"), retreat_offset_m=.004)
        decision = self.propose(value, context=context)
        self.assertTrue(decision.diagnostics["feasible"])
        self.assertTrue(decision.diagnostics["smoothing_outward_tangent_projection"])
        self.assertTrue(decision.diagnostics["command_endpoint_outward_release"])
        self.assertLessEqual(decision.action[0], 0.)
        self.assertAlmostEqual(decision.action[1], 0., places=14)

    def test_approach_entry_starts_push_at_zero_with_a_bounded_approach_command_drop(self):
        # The two states differ by 20nm around the same legacy .001+.0015m
        # approach threshold. One transitions to PUSH but starts its ramp at0.
        before = self.propose(state(qx=-.0505-1e-8), context=active("APPROACH"))
        after = self.propose(state(qx=-.0505+1e-8), context=active("APPROACH"))
        self.assertEqual(before.context.mode, "APPROACH")
        self.assertEqual(after.context.mode, "PUSH")
        # The last approach speed need not be zero. Bound this residual by
        # the physical .01m/s approach cap; never claim whole policy continuity.
        self.assertLessEqual(np.linalg.norm(before.action-after.action), .01+1e-15)
        self.assertEqual(before.diagnostics["noise_gain"], 0.)
        self.assertEqual(after.diagnostics["noise_gain"], 0.)

    def test_controller_does_not_require_contact_or_engine_parameter_metadata(self):
        config = deepcopy(self.config)
        del config["contact"]
        config["physics"] = {"action_dt_s": .1}
        controller = SmoothGeometryController(config, SPEC)
        decision = self.propose(control=controller)
        np.testing.assert_allclose(decision.action, [.04, 0.], atol=1e-14)

    def test_four_archived_stale_retract_routes_remain_rejected_by_legacy(self):
        legacy = GeometryController(self.config)
        for label, values, old_context, elapsed in STALE_RETRACT_CASES:
            with self.subTest(case=label):
                value = np.array(values, np.float64)
                context = GeometryContext(**old_context)
                decision = legacy.propose(value, self.task, context, elapsed)
                self.assertFalse(decision.diagnostics["feasible"])
                self.assertEqual(decision.diagnostics["reason"],
                                 "controller_infeasible:unsafe_same_face_waypoint")
                np.testing.assert_array_equal(decision.action, [0., 0.])

    def test_four_archived_stale_retract_routes_use_current_outward_tangent(self):
        for label, values, old_context, elapsed in STALE_RETRACT_CASES:
            with self.subTest(case=label):
                value = np.array(values, np.float64)
                context = GeometryContext(**old_context)
                previous = value.copy()
                decision = self.propose(value, context=context, elapsed=elapsed)
                theta = .5*math.atan2(math.sin(2*value[2]), math.cos(2*value[2]))
                normal = np.array([math.cos(theta), math.sin(theta)])
                tangent = np.array([-normal[1], normal[0]])
                local_y = float((value[6:8]-value[:2]) @ tangent)
                self.assertTrue(decision.diagnostics["feasible"])
                self.assertTrue(decision.diagnostics["retreat_route_reanchored"])
                self.assertTrue(decision.diagnostics["command_endpoint_checked"])
                self.assertTrue(decision.diagnostics["command_endpoint_path_safe"])
                self.assertTrue(decision.diagnostics["command_endpoint_outward_release"])
                self.assertAlmostEqual(decision.context.retreat_offset_m, local_y, places=14)
                self.assertLess(float(decision.action @ normal), 0.)
                self.assertAlmostEqual(float(decision.action @ tangent), 0., places=14)
                self.assertEqual(decision.context.locked_offset_m, context.locked_offset_m)
                self.assertEqual(decision.context.phase_started_s, context.phase_started_s)
                self.assertEqual(decision.context.last_elapsed_s, elapsed)
                self.assertEqual(decision.context.reassessment_count, context.reassessment_count)
                np.testing.assert_array_equal(value, previous)
                self.assertEqual(context.retreat_offset_m, old_context["retreat_offset_m"])

    def test_shallow_reanchor_preserves_locked_target_and_does_not_resample_context(self):
        context = replace(active("RETRACT", offset=.003), retreat_offset_m=.0030001,
                          reassessment_count=7, saturation_count=5)
        decision = self.propose(state(qx=-.0479, qy=.003), context=context)
        self.assertTrue(decision.diagnostics["retreat_route_reanchored"])
        self.assertEqual(decision.context.locked_offset_m, .003)
        self.assertAlmostEqual(decision.context.retreat_offset_m, .003, places=14)
        self.assertEqual(decision.context.reassessment_count, 7)
        self.assertEqual(decision.context.phase_started_s, context.phase_started_s)
        self.assertEqual(decision.diagnostics["noise_gain"], 0.)

    def test_reanchor_is_not_applied_to_other_modes_or_uninitialized_history(self):
        for context in (active("PUSH", offset=.003), active("APPROACH", offset=.003),
                        active("SLIDE", offset=.003), GeometryContext()):
            with self.subTest(mode=context.mode, initialized=context.locked_offset_m is not None):
                decision = self.propose(state(qx=-.0479, qy=.003), context=context)
                self.assertFalse(decision.diagnostics["retreat_route_reanchored"])

    def test_reanchor_is_not_applied_outside_observed_shallow_rear_overlap(self):
        context = replace(active("RETRACT"), retreat_offset_m=.0020001)
        for qx in (-.049, -.060):
            with self.subTest(qx=qx):
                decision = self.propose(state(qx=qx, qy=.002), context=context)
                self.assertFalse(decision.diagnostics["retreat_route_reanchored"])

    def test_reanchor_cannot_rescue_beyond_recovery_front_or_unsafe_workspace(self):
        context = replace(active("RETRACT"), retreat_offset_m=.0020001)
        values = (state(qx=-.044, qy=.002), state(qx=.07, qy=.002),
                  state(x=-.24, qx=-.0479, qy=.002))
        for value in values:
            with self.subTest(value=value):
                decision = self.propose(value, context=context)
                self.assertFalse(decision.diagnostics["feasible"])
                np.testing.assert_array_equal(decision.action, [0., 0.])
                self.assertEqual(decision.diagnostics["noise_gain"], 0.)

    def test_holding_still_clears_shallow_retract_route_and_commands_zero(self):
        context = replace(active("RETRACT", offset=.003), retreat_offset_m=.0030001)
        decision = self.propose(state(x=.08, qx=-.0479, qy=.003, vx=.05), context=context)
        np.testing.assert_array_equal(decision.action, [0., 0.])
        self.assertEqual(decision.diagnostics["reason"], "holding")
        self.assertIsNone(decision.context.locked_offset_m)
        self.assertIsNone(decision.context.retreat_offset_m)
        self.assertEqual(decision.diagnostics["noise_gain"], 0.)

    def test_shallow_route_preparation_cannot_hide_malformed_state_or_context(self):
        good = state(qx=-.0479, qy=.003)
        malformed = replace(active("RETRACT", offset=.003), mode="UNKNOWN")
        with self.assertRaises(GeometryControlError):
            self.propose(good, context=malformed)
        for value in (good.astype(np.float32), np.zeros(9, np.float64)):
            with self.subTest(shape=value.shape), self.assertRaises(GeometryControlError):
                self.propose(value, context=active("RETRACT", offset=.003))

    def test_control_copies_configuration_and_transition_specification(self):
        value = state(wz=.3)
        first = self.propose(value)
        self.config["geometry_baseline"]["yaw_offset_gain_m_rad"] = 30.
        self.spec["offset_taper_start_m"] = .0035
        second = self.propose(value)
        np.testing.assert_array_equal(first.action, second.action)
        self.assertEqual(first.diagnostics, second.diagnostics)

    def test_state_task_context_and_output_do_not_alias_or_mutate(self):
        value = state(wz=.3)
        previous = value.copy()
        goal = self.task.goal.copy()
        context = active()
        decision = self.propose(value, context=context)
        np.testing.assert_array_equal(value, previous)
        np.testing.assert_array_equal(self.task.goal, goal)
        self.assertEqual(context, active())
        self.assertFalse(decision.action.flags.writeable)
        self.assertEqual(decision.action.dtype, np.float64)
        self.assertEqual(decision.action.shape, (2,))
        with self.assertRaises(FrozenInstanceError):
            decision.context.mode = "UNKNOWN"

    def test_pi_symmetric_state_preserves_smooth_action(self):
        first = self.propose(state(yaw=.1, qx=-.060, qy=.01), context=active("SLIDE"))
        second = self.propose(state(yaw=.1+math.pi, qx=-.060, qy=.01), context=active("SLIDE"))
        np.testing.assert_allclose(first.action, second.action, atol=1e-14)
        self.assertAlmostEqual(first.diagnostics["noise_gain"], second.diagnostics["noise_gain"])

    def test_no_engine_or_execution_rng_can_be_used_by_smooth_proposal(self):
        with ExitStack() as stack:
            callbacks = [stack.enter_context(patch.object(mujoco, key, side_effect=AssertionError(key)))
                         for key in ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse")]
            stack.enter_context(patch.object(np.random, "Generator", side_effect=AssertionError("RNG")))
            decision = self.propose(state(wz=.3))
        for callback in callbacks:
            callback.assert_not_called()
        self.assertEqual(decision.diagnostics["future_physics_queries"], 0)
        self.assertFalse(decision.diagnostics["raw_contact_labels_used"])

    def test_holding_does_not_hide_invalid_context_or_backward_time(self):
        value = state(x=.08)
        for context, elapsed in ((active(last=1.), .5),
                                 (active(offset=.1), 1.)):
            with self.subTest(context=context), self.assertRaises(GeometryControlError):
                self.propose(value, context=context, elapsed=elapsed)

    def test_transition_specification_rejects_unknown_schema_or_unregistered_key(self):
        for changes in ({"schema_version": "future-v2"}, {"hidden_gain": 1.}):
            spec = deepcopy(SPEC); spec.update(changes)
            with self.subTest(changes=changes), self.assertRaises(GeometryControlError):
                SmoothGeometryController(self.config, spec)

    def test_transition_scales_reject_boolean_nonfinite_and_nonpositive_values(self):
        for key in ("offset_taper_start_m", "phase_end_taper_s", "phase_entry_ramp_s",
                    "slide_blend_outer_tolerance_multiplier", "free_motion_speed_max_m_s"):
            for bad in (True, 0., -.1, math.nan, math.inf):
                spec = deepcopy(SPEC); spec[key] = bad
                with self.subTest(key=key, bad=bad), self.assertRaises(GeometryControlError):
                    SmoothGeometryController(self.config, spec)

    def test_taper_cannot_start_after_replan_or_consume_whole_phase(self):
        for changes in ({"offset_taper_start_m": .004},
                        {"phase_end_taper_s": 2.},
                        {"phase_entry_ramp_s": 2.},
                        {"slide_blend_outer_tolerance_multiplier": 1.}):
            spec = deepcopy(SPEC); spec.update(changes)
            with self.subTest(changes=changes), self.assertRaises(GeometryControlError):
                SmoothGeometryController(self.config, spec)


class SmoothMainCollectionIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_pilot_config()
        cls.legacy_settings = tomllib.loads((ROOT/"configs/p3_collection.v1.toml").read_text())
        cls.smooth_settings = deepcopy(cls.legacy_settings)
        cls.smooth_settings["geometry_transition"] = deepcopy(SPEC)
        cls.smooth_settings["geometry_transition"]["noise_weighting"] = NOISE_WEIGHTING
        cls.override = json.loads((ROOT/"configs/p3_geometry_selected.v1.json").read_text())
        cls.plans, _ = build_main_collection_plan(cls.legacy_settings, cls.config)

    def policy(self, mode="geometry_perturbed", *, noise=(.003, .004), legacy=False):
        plan = deepcopy(next(p for p in self.plans if p["mode"] == mode))
        if mode == "geometry_perturbed":
            plan["perturbation_vectors_m_s"] = [list(noise) for _ in range(128)]
        return MainCollectionPolicy(self.config,
            self.legacy_settings if legacy else self.smooth_settings, plan, self.override)

    def test_legacy_configuration_selects_legacy_controller_without_hidden_opt_in(self):
        old = self.policy(legacy=True)
        new = self.policy()
        self.assertIs(type(old.controller), GeometryController)
        self.assertIs(type(new.controller), SmoothGeometryController)
        self.assertNotIn("geometry_transition", self.legacy_settings)

    def test_transition_choice_does_not_redraw_initial_conditions_split_or_noise(self):
        plans, assignments = build_main_collection_plan(self.smooth_settings, self.config)
        self.assertEqual(plans, self.plans)
        _, legacy_assignments = build_main_collection_plan(self.legacy_settings, self.config)
        self.assertEqual(assignments, legacy_assignments)

    def test_noise_gain_scales_same_predeclared_world_vector_without_redraw(self):
        policy = self.policy()
        actual = np.array([.006, -.008], np.float64)
        context = replace(policy.initial_context(), geometry_context=active(),
                          previous_applied_action=actual)
        decision = policy.propose(state(wz=.3), context, .6, 6)
        # Taper .5 gives base .02m/s and adds half [.003,.004] in world XY.
        np.testing.assert_allclose(decision.action, [.0215, .002], atol=1e-14)
        np.testing.assert_allclose(decision.diagnostics["base_action_m_s"], [.02, 0.], atol=1e-14)
        np.testing.assert_allclose(decision.diagnostics["applied_noise_delta_m_s"], [.0015, .002], atol=1e-14)
        np.testing.assert_array_equal(decision.diagnostics["predeclared_noise_m_s"], [.003, .004])
        self.assertAlmostEqual(decision.diagnostics["noise_gain"], .5, places=14)
        np.testing.assert_array_equal(decision.context.previous_applied_action, actual)
        self.assertFalse(np.shares_memory(decision.context.previous_applied_action, actual))
        self.assertFalse(decision.context.previous_applied_action.flags.writeable)

    def test_post_noise_norm_limit_preserves_direction_and_previous_actual_command(self):
        policy = self.policy(noise=(.02, .06))
        actual = np.array([.006, -.008], np.float64)
        context = replace(policy.initial_context(), previous_applied_action=actual)
        proposal = GeometryDecision(np.array([.032, 0.], np.float64), active(),
            {"feasible": True, "reason": "command", "mode_after": "PUSH", "noise_gain": .5})
        with patch.object(policy.controller, "propose", return_value=proposal):
            decision = policy.propose(state(), context, .6, 6)
        raw = np.array([.042, .03])
        expected = .04*raw/math.hypot(*raw)
        np.testing.assert_allclose(decision.action, expected, atol=1e-14)
        self.assertTrue(decision.diagnostics["post_noise_norm_saturated"])
        np.testing.assert_allclose(decision.diagnostics["applied_noise_delta_m_s"],
                                   expected-[.032, 0.], atol=1e-14)
        np.testing.assert_array_equal(decision.context.previous_applied_action, actual)
        np.testing.assert_array_equal(context.previous_applied_action, actual)

    def test_push_entry_zero_gain_leaves_candidate_vector_unapplied(self):
        policy = self.policy()
        context = replace(policy.initial_context(), geometry_context=active())
        decision = policy.propose(state(), context, 0., 0)
        self.assertEqual(decision.context.geometry_context.mode, "PUSH")
        self.assertEqual(decision.diagnostics["noise_gain"], 0.)
        np.testing.assert_array_equal(decision.action, [0., 0.])
        np.testing.assert_array_equal(decision.diagnostics["applied_noise_delta_m_s"], [0., 0.])
        np.testing.assert_array_equal(decision.diagnostics["predeclared_noise_m_s"], [.003, .004])

    def test_free_modes_receive_no_noise_even_when_candidate_vector_is_nonzero(self):
        policy = self.policy()
        for mode, qx, qy in (("RETRACT", -.049, .004), ("SLIDE", -.060, .004),
                             ("APPROACH", -.055, .001)):
            context = replace(policy.initial_context(), geometry_context=active(mode))
            with self.subTest(mode=mode):
                decision = policy.propose(state(qx=qx, qy=qy), context, .6, 6)
                self.assertFalse(decision.diagnostics["noise_applied"])
                self.assertEqual(decision.diagnostics["noise_gain"], 0.)
                np.testing.assert_array_equal(decision.diagnostics["applied_noise_delta_m_s"], [0., 0.])

    def test_holding_zero_reset_is_not_a_readiness_claim_and_retains_actual_context(self):
        policy = self.policy()
        actual = np.array([.02, .01], np.float64)
        context = replace(policy.initial_context(), geometry_context=active(offset=.01),
                          previous_applied_action=actual)
        decision = policy.propose(state(x=.08, wz=.2, vx=.05), context, .2, 2)
        np.testing.assert_array_equal(decision.action, [0., 0.])
        np.testing.assert_array_equal(decision.context.previous_applied_action, actual)
        self.assertEqual(decision.context.geometry_context.mode, "RETRACT")
        self.assertIsNone(decision.context.geometry_context.locked_offset_m)
        self.assertTrue(decision.diagnostics["holding"])
        self.assertFalse(decision.diagnostics["noise_applied"])
        self.assertNotIn("ready", decision.diagnostics)

    def test_infeasible_decision_stays_zero_and_unperturbed(self):
        policy = self.policy()
        decision = policy.propose(state(x=.1), policy.initial_context(), 0., 0)
        self.assertFalse(decision.diagnostics["feasible"])
        self.assertFalse(decision.diagnostics["noise_applied"])
        np.testing.assert_array_equal(decision.action, [0., 0.])

    def test_corrupt_noise_gain_is_rejected_before_physical_execution(self):
        policy = self.policy()
        for bad in (-.1, 1.1, math.nan, math.inf):
            proposal = GeometryDecision(np.array([.01, 0.], np.float64), active(),
                {"feasible": True, "reason": "command", "mode_after": "PUSH", "noise_gain": bad})
            with self.subTest(bad=bad), patch.object(policy.controller, "propose", return_value=proposal), \
                    self.assertRaises(DataContractError):
                policy.propose(state(), policy.initial_context(), .6, 6)

    def test_missing_or_changed_noise_weighting_rule_is_rejected_explicitly(self):
        for value in (None, "unweighted", False):
            settings = deepcopy(self.smooth_settings)
            if value is None:
                del settings["geometry_transition"]["noise_weighting"]
            else:
                settings["geometry_transition"]["noise_weighting"] = value
            plan = next(p for p in self.plans if p["mode"] == "geometry_perturbed")
            with self.subTest(value=value), self.assertRaises(DataContractError):
                MainCollectionPolicy(self.config, settings, plan, self.override)


class SmoothSchedulingClockTest(unittest.TestCase):
    def setUp(self):
        self.config = load_pilot_config()
        self.config["geometry_baseline"].update(
            position_gain_per_s=4., yaw_creep_speed_m_s=.02,
            yaw_offset_gain_m_rad=.09)
        self.task = TaskSpec.from_config(self.config)
        self.control = SmoothGeometryController(self.config, SPEC_V2)

    def propose(self, elapsed, *, context=None, value=None, control=None):
        return (self.control if control is None else control).propose(
            state() if value is None else value, self.task,
            active(last=1.9) if context is None else context, elapsed)

    def test_two_second_roundoff_pair_has_same_phase_and_zero_entry_command(self):
        decisions = [self.propose(elapsed) for elapsed in (2.-5e-12, 2.+5e-12)]
        for elapsed, decision in zip((2.-5e-12, 2.+5e-12), decisions):
            self.assertEqual(decision.context.mode, "RETRACT")
            self.assertEqual(decision.context.phase_started_s, 2.)
            np.testing.assert_array_equal(decision.action, [0., 0.])
            self.assertEqual(decision.diagnostics["raw_elapsed_s"], elapsed)
            self.assertEqual(decision.diagnostics["scheduling_elapsed_s"], 2.)
            self.assertTrue(decision.diagnostics["clock_snapped"])
            self.assertTrue(decision.diagnostics["clock_scheduling_only"])
        self.assertEqual(decisions[0].context, decisions[1].context)

    def test_v1_and_legacy_keep_the_recorded_phase_boundary_difference(self):
        context = active(last=1.9)
        for controller in (GeometryController(self.config), SmoothGeometryController(self.config, SPEC)):
            with self.subTest(controller=type(controller).__name__):
                before = self.propose(2.-5e-12, context=context, control=controller)
                exact = self.propose(2., context=context, control=controller)
                self.assertEqual(before.context.mode, "PUSH")
                self.assertEqual(exact.context.mode, "RETRACT")
                if isinstance(controller, SmoothGeometryController):
                    self.assertFalse(before.diagnostics["clock_snapped"])
                    self.assertEqual(before.diagnostics["scheduling_elapsed_s"], 2.-5e-12)

    def test_off_grid_elapsed_and_context_times_are_preserved(self):
        context = active(phase=.00321, last=.12)
        decision = self.propose(.12345, context=context)
        self.assertEqual(decision.context.phase_started_s, .00321)
        self.assertEqual(decision.context.last_elapsed_s, .12345)
        self.assertEqual(decision.diagnostics["scheduling_elapsed_s"], .12345)
        self.assertEqual(decision.diagnostics["context_phase_start_scheduling_s"], .00321)
        self.assertFalse(decision.diagnostics["clock_snapped"])

    def test_near_grid_value_outside_tolerance_is_not_snapped(self):
        elapsed = 2.-2e-9
        decision = self.propose(elapsed, context=active(last=1.5))
        self.assertEqual(decision.context.mode, "PUSH")
        self.assertEqual(decision.diagnostics["scheduling_elapsed_s"], elapsed)
        self.assertFalse(decision.diagnostics["clock_snapped"])

    def test_context_and_elapsed_clocks_prepare_together_without_mutating_inputs(self):
        context = active(phase=1.+5e-12, last=2.9+5e-12)
        value = state(); previous = value.copy(); goal = self.task.goal.copy()
        decision = self.propose(3.+5e-12, context=context, value=value)
        self.assertEqual(decision.context.mode, "RETRACT")
        self.assertEqual(decision.context.phase_started_s, 3.)
        self.assertEqual(decision.diagnostics["context_phase_start_raw_s"], context.phase_started_s)
        self.assertEqual(decision.diagnostics["context_phase_start_scheduling_s"], 1.)
        self.assertAlmostEqual(decision.diagnostics["context_last_elapsed_scheduling_s"], 2.9, places=14)
        self.assertEqual(context.phase_started_s, 1.+5e-12)
        self.assertEqual(context.last_elapsed_s, 2.9+5e-12)
        np.testing.assert_array_equal(value, previous)
        np.testing.assert_array_equal(self.task.goal, goal)

    def test_clock_prepare_and_reanchor_keep_the_same_locked_target(self):
        context = replace(active("RETRACT", offset=.003, phase=.5+5e-12, last=.5+5e-12),
                          retreat_offset_m=.0030001)
        decision = self.propose(.6+5e-12, context=context, value=state(qx=-.0479, qy=.003))
        self.assertTrue(decision.diagnostics["clock_snapped"])
        self.assertTrue(decision.diagnostics["retreat_route_reanchored"])
        self.assertEqual(decision.context.locked_offset_m, .003)
        self.assertAlmostEqual(decision.context.retreat_offset_m, .003, places=14)
        self.assertEqual(decision.context.phase_started_s, .5)
        self.assertAlmostEqual(decision.context.last_elapsed_s, .6, places=14)

    def test_raw_invalid_or_backwards_time_is_rejected_before_clock_preparation(self):
        for elapsed in (-.1, True, math.nan, math.inf, .5):
            with self.subTest(elapsed=elapsed), self.assertRaises(GeometryControlError):
                self.propose(elapsed, context=active(last=1.))
        for context in (replace(active(), phase_started_s=1., last_elapsed_s=.5),
                        replace(active(), mode="UNKNOWN")):
            with self.subTest(context=context), self.assertRaises(GeometryControlError):
                self.propose(2., context=context)

    def test_raw_bad_state_and_task_cannot_be_hidden_by_grid_time(self):
        for value in (np.zeros(9, np.float64), state().astype(np.float32)):
            with self.subTest(shape=value.shape), self.assertRaises(GeometryControlError):
                self.propose(2.+5e-12, value=value)
        task = replace(self.task, position_tolerance_m=0.)
        with self.assertRaises(GeometryControlError):
            self.control.propose(state(), task, active(last=1.9), 2.+5e-12)

    def test_v2_holding_changes_no_physical_state_or_supervisor_timer(self):
        value = state(x=.08, qx=-.0479, qy=.003, vx=.02)
        before = value.copy()
        context = replace(active("RETRACT", offset=.003, phase=.5+5e-12, last=.5+5e-12),
                          retreat_offset_m=.0030001)
        with patch("push_align.task.consume_step", side_effect=AssertionError("Policy touched supervisor")):
            decision = self.propose(.6+5e-12, context=context, value=value)
        np.testing.assert_array_equal(value, before)
        np.testing.assert_array_equal(decision.action, [0., 0.])
        self.assertEqual(decision.diagnostics["reason"], "holding")
        self.assertEqual(decision.diagnostics["noise_gain"], 0.)
        self.assertFalse(decision.diagnostics["retreat_route_reanchored"])
        self.assertIsNone(decision.context.locked_offset_m)
        self.assertIsNone(decision.context.retreat_offset_m)
        self.assertEqual(decision.diagnostics["raw_elapsed_s"], .6+5e-12)

    def test_clock_tolerance_has_units_and_strict_finite_positive_bound(self):
        for bad in (True, 0., -.1, math.nan, math.inf, 2e-7):
            spec = deepcopy(SPEC_V2); spec["clock_snap_tolerance_s"] = bad
            with self.subTest(bad=bad), self.assertRaises(GeometryControlError):
                SmoothGeometryController(self.config, spec)
        spec = deepcopy(SPEC_V2); spec["clock_snap_tolerance_s"] = 1e-7
        SmoothGeometryController(self.config, spec)

    def test_schema_v1_cannot_silently_enable_clock_snap_and_v2_requires_it(self):
        wrong_v1 = {**SPEC, "clock_snap_tolerance_s": 1e-9}
        missing_v2 = {**SPEC, "schema_version": "geometry-transition-v2"}
        for spec in (wrong_v1, missing_v2):
            with self.subTest(spec=spec), self.assertRaises(GeometryControlError):
                SmoothGeometryController(self.config, spec)

    def test_v2_controller_copies_clock_tolerance_before_execution(self):
        spec = deepcopy(SPEC_V2)
        controller = SmoothGeometryController(self.config, spec)
        spec["clock_snap_tolerance_s"] = 1e-15
        decision = self.propose(2.-5e-12, control=controller)
        self.assertEqual(decision.context.mode, "RETRACT")
        self.assertEqual(decision.diagnostics["scheduling_elapsed_s"], 2.)


if __name__ == "__main__":
    unittest.main()
