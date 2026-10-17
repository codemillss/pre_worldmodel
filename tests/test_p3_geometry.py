"""Independent geometric-controller contract and hand-calculation tests.

State fixtures establish command geometry, units, event handling and input
contracts. They do not establish applied contact forces or robot task success;
those require the separately recorded P3-01 physical executions.
"""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import math
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from push_align.config import load_pilot_config
from push_align.geometry_control import (GeometryContext, GeometryControlError,
                                        GeometryController)
from push_align.planning import TaskSpec


def state(*, x=0., y=0., yaw=0., wz=0., qx=-.051, qy=0.,
          pusher_vx=0., pusher_vy=0.):
    """Place the pusher in canonical local object coordinates, in metres."""
    theta_bar = .5 * math.atan2(math.sin(2*yaw), math.cos(2*yaw))
    d = np.array([math.cos(theta_bar), math.sin(theta_bar)])
    t = np.array([-math.sin(theta_bar), math.cos(theta_bar)])
    pusher = np.array([x, y]) + qx*d + qy*t
    return np.array([x, y, yaw, 0., 0., wz, *pusher,
                     pusher_vx, pusher_vy], dtype=np.float64)


class GeometryControllerTest(unittest.TestCase):
    def setUp(self):
        self.config = load_pilot_config()
        self.controller = GeometryController(self.config)
        self.task = TaskSpec.from_config(self.config)

    def propose(self, value=None, *, controller=None, task=None, context=None,
                elapsed=0.):
        control = self.controller if controller is None else controller
        return control.propose(state() if value is None else value,
                               self.task if task is None else task,
                               control.initial_context() if context is None else context,
                               elapsed)

    def test_initial_context_is_frozen_and_has_no_locked_contact_offset(self):
        context = self.controller.initial_context()
        self.assertIsInstance(context, GeometryContext)
        self.assertEqual(context.mode, 'RETRACT')
        self.assertIsNone(context.locked_offset_m)
        self.assertEqual(context.reassessment_count, 0)
        with self.assertRaises(FrozenInstanceError):
            context.mode = 'PUSH'

    def test_hand_calculated_yaw_offset_has_correct_rotation_sign(self):
        # theta=+0.1, horizontal u, hx=.04: b=.03*.1+.04*tan(.1).
        expected = .003 + .04*math.tan(.1)
        positive = self.propose(state(yaw=.1))
        negative = self.propose(state(yaw=-.1))
        self.assertAlmostEqual(positive.diagnostics['b_target_m'], expected, places=13)
        self.assertAlmostEqual(negative.diagnostics['b_target_m'], -expected, places=13)
        # F along +world X gives torque -hx*(t.u)-b*(d.u).
        torque = .04*math.sin(.1)-expected*math.cos(.1)
        self.assertLess(torque, 0.)

    def test_angular_velocity_damping_adds_a_metre_valued_offset(self):
        decision = self.propose(state(yaw=.1, wz=.2))
        expected = .003 + .002 + .04*math.tan(.1)
        self.assertAlmostEqual(decision.diagnostics['b_target_m'], expected, places=13)

    def test_lateral_translation_compensation_is_hand_calculated(self):
        spec = replace(self.task, goal=np.array([.08, .016, 0.], np.float64))
        decision = self.propose(task=spec)
        # t.u / d.u = .016/.08=.2, independent of command magnitude.
        self.assertAlmostEqual(decision.diagnostics['b_target_m'], -.008, places=13)
        np.testing.assert_allclose(decision.diagnostics['direction_unit'],
                                   [5/math.sqrt(26), 1/math.sqrt(26)], atol=1e-14)

    def test_offset_uses_unit_direction_independently_of_command_speed(self):
        changed = deepcopy(self.config)
        changed['geometry_baseline']['position_gain_per_s'] = .08
        slow = GeometryController(changed)
        value = state(yaw=.1)
        fast_decision = self.propose(value)
        slow_decision = self.propose(value, controller=slow)
        self.assertAlmostEqual(fast_decision.diagnostics['b_target_m'],
                               slow_decision.diagnostics['b_target_m'], places=14)
        self.assertAlmostEqual(np.linalg.norm(fast_decision.diagnostics['direction_unit']),
                               1., places=14)

    def test_contact_offset_saturates_symmetrically_without_changing_units(self):
        for sign in (-1., 1.):
            with self.subTest(sign=sign):
                decision = self.propose(state(wz=sign*10.))
                self.assertAlmostEqual(decision.diagnostics['b_target_m'], sign*.015)
                self.assertGreater(abs(decision.diagnostics['b_unclipped_m']), .015)
                self.assertGreater(decision.context.saturation_count, 0)

    def test_direction_cosine_threshold_is_applied_before_velocity_scaling(self):
        spec = replace(self.task, goal=np.array([.02, math.sqrt(.08**2-.02**2), 0.],
                                               np.float64))
        decision = self.propose(task=spec)
        self.assertAlmostEqual(decision.diagnostics['direction_cosine'], .25, places=13)
        self.assertTrue(decision.diagnostics['feasible'])
        self.assertTrue(np.isfinite(decision.action).all())

    def test_outward_direction_returns_explicit_infeasible_without_division(self):
        below = .249
        for target in ([0., .08, 0.], [-.08, 0., 0.],
                       [.08*below, .08*math.sqrt(1-below**2), 0.]):
            with self.subTest(target=target):
                spec = replace(self.task, goal=np.array(target, np.float64))
                with np.errstate(divide='raise', invalid='raise'):
                    decision = self.propose(task=spec)
                self.assertFalse(decision.diagnostics['feasible'])
                self.assertIn('controller_infeasible', decision.diagnostics['reason'])
                np.testing.assert_array_equal(decision.action, np.zeros(2))

    def test_pi_symmetric_rectangle_gives_the_same_command_and_offset(self):
        first = self.propose(state(yaw=.1))
        second = self.propose(state(yaw=.1+math.pi))
        np.testing.assert_allclose(first.action, second.action, atol=1e-14)
        self.assertAlmostEqual(first.diagnostics['b_target_m'],
                               second.diagnostics['b_target_m'], places=14)
        self.assertAlmostEqual(second.diagnostics['yaw_error_rad'], .1, places=13)

    def test_goal_yaw_is_used_in_symmetric_orientation_error(self):
        spec = replace(self.task, goal=np.array([.08, 0., .08], np.float64))
        decision = self.propose(state(yaw=.1), task=spec)
        expected = .03*.02 + .04*math.tan(.1)
        self.assertAlmostEqual(decision.diagnostics['yaw_error_rad'], .02, places=13)
        self.assertAlmostEqual(decision.diagnostics['b_target_m'], expected, places=13)

    def test_yaw_only_uses_inward_creep_and_never_normalizes_zero_translation(self):
        value = state(x=.08, yaw=.1, qx=-.049, qy=.003)
        context = replace(self.controller.initial_context(), mode='PUSH',
                          locked_offset_m=.003, retreat_offset_m=.003)
        with np.errstate(divide='raise', invalid='raise'):
            decision = self.propose(value, context=context)
        np.testing.assert_allclose(decision.diagnostics['direction_unit'],
                                   [math.cos(.1), math.sin(.1)], atol=1e-14)
        np.testing.assert_allclose(decision.action,
                                   .005*np.array([math.cos(.1), math.sin(.1)]),
                                   atol=1e-14)

    def test_goal_pose_requests_zero_even_if_velocity_has_not_settled(self):
        value = state(x=.08, wz=.2, pusher_vx=.01)
        with np.errstate(divide='raise', invalid='raise'):
            decision = self.propose(value)
        np.testing.assert_array_equal(decision.action, np.zeros(2))
        self.assertTrue(decision.diagnostics['feasible'])
        # Task supervisor, rather than G, owns the speed/holding completion gate.

    def test_retreat_preserves_tangent_offset_instead_of_crossing_the_object(self):
        value = state(qx=-.048, qy=.005, wz=-1.)
        decision = self.propose(value)
        self.assertEqual(decision.context.mode, 'RETRACT')
        waypoint = np.asarray(decision.diagnostics['waypoint_local_xy_m'])
        np.testing.assert_allclose(waypoint, [-.060, .005], atol=1e-14)
        self.assertLess(decision.action[0], 0.)
        self.assertAlmostEqual(decision.action[1], 0., places=14)
        self.assertTrue(decision.diagnostics['path_safe'])

    def test_slide_occurs_on_the_rear_clearance_line(self):
        value = state(qx=-.060, qy=0., wz=1.)
        decision = self.propose(value)
        self.assertEqual(decision.context.mode, 'SLIDE')
        np.testing.assert_allclose(decision.diagnostics['waypoint_local_xy_m'],
                                   [-.060, .010], atol=1e-14)
        self.assertAlmostEqual(decision.action[0], 0., places=14)
        self.assertGreater(decision.action[1], 0.)
        self.assertTrue(decision.diagnostics['path_safe'])

    def test_front_side_retreat_cannot_cut_through_the_expanded_object(self):
        # A straight retreat from +X through the centre would cross the part.
        decision = self.propose(state(qx=.07))
        self.assertFalse(decision.diagnostics['feasible'])
        self.assertIn('controller_infeasible', decision.diagnostics['reason'])
        np.testing.assert_array_equal(decision.action, np.zeros(2))

    def test_slide_then_approach_keeps_target_tangent_coordinate(self):
        value = state(qx=-.060, qy=.010, wz=1.)
        context = replace(self.controller.initial_context(), mode='SLIDE',
                          locked_offset_m=.010, retreat_offset_m=0.)
        decision = self.propose(value, context=context, elapsed=.1)
        self.assertEqual(decision.context.mode, 'APPROACH')
        np.testing.assert_allclose(decision.diagnostics['waypoint_local_xy_m'],
                                   [-.049, .010], atol=1e-14)
        self.assertGreater(decision.action[0], 0.)
        self.assertAlmostEqual(decision.action[1], 0., places=14)
        self.assertLessEqual(np.linalg.norm(decision.action), .01+1e-15)

    def test_near_geometry_changes_approach_to_push_without_contact_label(self):
        value = state(qx=-.049)
        context = replace(self.controller.initial_context(), mode='APPROACH',
                          locked_offset_m=0., retreat_offset_m=0.)
        decision = self.propose(value, context=context, elapsed=.1)
        self.assertEqual(decision.context.mode, 'PUSH')
        np.testing.assert_allclose(decision.action, [.04, 0.], atol=1e-14)
        self.assertTrue(decision.diagnostics['intentional_contact_motion'])

    def test_geometric_contact_loss_retracts_instead_of_continuing_push(self):
        value = state(qx=-.053)
        context = replace(self.controller.initial_context(), mode='PUSH',
                          locked_offset_m=0., retreat_offset_m=0.)
        decision = self.propose(value, context=context, elapsed=.1)
        self.assertEqual(decision.context.mode, 'RETRACT')
        self.assertLess(decision.action[0], 0.)
        self.assertIn('loss', decision.diagnostics['transition_reason'])

    def test_large_contact_offset_change_retracts_before_tangential_relocation(self):
        value = state(qx=-.049, wz=1.)
        context = replace(self.controller.initial_context(), mode='PUSH',
                          locked_offset_m=0., retreat_offset_m=0.)
        decision = self.propose(value, context=context, elapsed=.1)
        self.assertEqual(decision.context.mode, 'RETRACT')
        self.assertLess(decision.action[0], 0.)
        self.assertAlmostEqual(decision.action[1], 0., places=14)
        self.assertIn('offset', decision.diagnostics['transition_reason'])

    def test_phase_timeout_reassesses_without_ending_the_task(self):
        context = replace(self.controller.initial_context(), mode='PUSH',
                          locked_offset_m=0., retreat_offset_m=0.)
        decision = self.propose(state(qx=-.049), context=context, elapsed=2.1)
        self.assertTrue(decision.diagnostics['feasible'])
        self.assertGreater(decision.context.reassessment_count, 0)
        self.assertNotIn('timeout', decision.diagnostics['reason'])

    def test_waypoint_velocity_damping_reduces_motion_toward_waypoint(self):
        still = self.propose(state(qx=-.055))
        moving = self.propose(state(qx=-.055, pusher_vx=-.005))
        self.assertLess(still.action[0], 0.)
        self.assertLess(abs(moving.action[0]), abs(still.action[0]))

    def test_all_phase_commands_respect_world_speed_norm_and_float64_shape(self):
        cases = [(state(qx=-.048), 'RETRACT'),
                 (state(qx=-.060, wz=1.), 'SLIDE'),
                 (state(qx=-.060, qy=.010, wz=1.), 'APPROACH'),
                 (state(qx=-.049), 'PUSH')]
        for value, mode in cases:
            with self.subTest(mode=mode):
                context = replace(self.controller.initial_context(), mode=mode,
                                  locked_offset_m=(.010 if mode=='APPROACH' else 0.),
                                  retreat_offset_m=0.)
                decision = self.propose(value, context=context, elapsed=.1)
                self.assertEqual(decision.action.dtype, np.float64)
                self.assertEqual(decision.action.shape, (2,))
                self.assertTrue(np.isfinite(decision.action).all())
                self.assertLessEqual(np.linalg.norm(decision.action), .04+1e-15)

    def test_input_state_goal_context_and_configuration_remain_unchanged(self):
        value = state(yaw=.1)
        before = value.copy()
        goal_before = self.task.goal.copy()
        config_before = deepcopy(self.config)
        context = self.controller.initial_context()
        decision = self.propose(value, context=context)
        np.testing.assert_array_equal(value, before)
        np.testing.assert_array_equal(self.task.goal, goal_before)
        self.assertEqual(self.config, config_before)
        self.assertEqual(context, self.controller.initial_context())
        self.assertIsNot(decision.context, context)

    def test_controller_copies_settings_instead_of_reading_later_mutations(self):
        original = self.propose(state(yaw=.1))
        self.config['geometry_baseline']['yaw_offset_gain_m_rad'] = 30.
        after = self.propose(state(yaw=.1))
        self.assertAlmostEqual(original.diagnostics['b_target_m'],
                               after.diagnostics['b_target_m'], places=14)
        np.testing.assert_array_equal(original.action, after.action)

    def test_repeated_current_state_and_context_produce_identical_command(self):
        value = state(yaw=.1)
        context = self.controller.initial_context()
        first = self.propose(value, context=context)
        second = self.propose(value.copy(), context=context)
        np.testing.assert_array_equal(first.action, second.action)
        self.assertEqual(first.context, second.context)
        self.assertEqual(first.diagnostics, second.diagnostics)

    def test_controller_never_queries_future_physics_or_builds_engine_state(self):
        forbidden = ('mj_step', 'mj_step1', 'mj_step2', 'mj_forward', 'mj_inverse')
        with ExitStack() as stack:
            mocks = [stack.enter_context(patch.object(mujoco, name,
                                                      side_effect=AssertionError(name)))
                     for name in forbidden]
            self.propose(state(yaw=.1))
            for mock in mocks:
                mock.assert_not_called()

    def test_nonfinite_wrong_shape_or_non_float64_state_is_rejected(self):
        for value in (np.zeros(9, np.float64), np.zeros((1, 10), np.float64),
                      np.zeros(10, np.float32), np.zeros(10, np.int64),
                      state().tolist()):
            with self.subTest(value_type=type(value), shape=getattr(value, 'shape', None)):
                with self.assertRaises(GeometryControlError):
                    self.propose(value)
        for index in (0, 2, 5, 6, 9):
            for bad in (math.nan, math.inf):
                value = state()
                value[index] = bad
                with self.subTest(index=index, bad=bad):
                    with self.assertRaises(GeometryControlError):
                        self.propose(value)

    def test_nonfinite_or_backward_elapsed_and_invalid_context_are_rejected(self):
        for elapsed in (-.1, math.nan, math.inf, True):
            with self.subTest(elapsed=elapsed):
                with self.assertRaises(GeometryControlError):
                    self.propose(elapsed=elapsed)
        context = replace(self.controller.initial_context(), last_elapsed_s=1.)
        with self.assertRaises(GeometryControlError):
            self.propose(context=context, elapsed=.9)
        context = replace(self.controller.initial_context(), mode='UNKNOWN')
        with self.assertRaises(GeometryControlError):
            self.propose(context=context)

    def test_nonfinite_goal_or_nonpositive_task_tolerance_is_rejected(self):
        bad_goal = replace(self.task, goal=np.array([math.nan, 0., 0.], np.float64))
        bad_tolerance = replace(self.task, position_tolerance_m=0.)
        for spec in (bad_goal, bad_tolerance):
            with self.subTest(spec=spec):
                with self.assertRaises(GeometryControlError):
                    self.propose(task=spec)

    def test_missing_required_geometry_setting_is_rejected_before_execution(self):
        keys = ('waypoint_velocity_damping', 'approach_speed_max_m_s',
                'yaw_creep_speed_m_s', 'near_contact_loss_gap_m',
                'contact_offset_replan_threshold_m')
        for key in keys:
            config = deepcopy(self.config)
            del config['geometry_baseline'][key]
            with self.subTest(key=key):
                with self.assertRaises(GeometryControlError):
                    GeometryController(config)

    def test_nonpositive_or_nonfinite_required_geometry_scale_is_rejected(self):
        for key in ('position_gain_per_s', 'speed_max_m_s', 'waypoint_gain_per_s',
                    'approach_speed_max_m_s', 'yaw_creep_speed_m_s',
                    'contact_mode_max_elapsed_s'):
            for value in (0., -.1, math.nan, math.inf, True):
                config = deepcopy(self.config)
                config['geometry_baseline'][key] = value
                with self.subTest(key=key, value=value):
                    with self.assertRaises(GeometryControlError):
                        GeometryController(config)

    def test_state_schema_version_or_field_order_change_is_rejected(self):
        for change in ('version', 'order', 'action_semantics'):
            config = deepcopy(self.config)
            if change == 'version':
                config['schema_version'] = 'push-align-state-v2'
            elif change == 'order':
                config['schema']['state_fields'][0:2] = ['obj_y', 'obj_x']
            else:
                config['schema']['action_semantics'] = 'target_position'
            with self.subTest(change=change):
                with self.assertRaises(GeometryControlError):
                    GeometryController(config)


if __name__ == '__main__':
    unittest.main()
