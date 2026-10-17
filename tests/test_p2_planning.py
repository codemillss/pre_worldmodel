"""Independent P2-05 checks of predictions, costs and candidate decisions.

These controlled predictors establish arithmetic and wiring, not learned-model
accuracy, alignment success or optimality outside the supplied candidate set.
"""
from copy import deepcopy
import math
import unittest
from unittest.mock import patch

import numpy as np

from push_align.config import load_pilot_config
from push_align.planning import (CostBatch, PlanningError, TaskSpec, choose,
                                 rollout, score)


def fixture(candidate_count=2, horizon=2):
    config = deepcopy(load_pilot_config())
    states = np.zeros((candidate_count, 10), dtype=np.float64)
    states[:, 6] = -.05
    actions = np.zeros((candidate_count, horizon, 2), dtype=np.float64)
    return config, states, actions


def hold(state, action):
    return np.zeros_like(state)


def position_only(config):
    config = deepcopy(config)
    for key in config["cost"]:
        if key.startswith(("stage_", "terminal_")):
            config["cost"][key] = 0.
    config["cost"]["stage_position"] = 1.
    config["cost"]["terminal_position"] = 5.
    return config


def costs(predictions, actions, config, previous=None):
    if previous is None:
        previous = np.zeros(2, dtype=np.float64)
    return score(predictions, actions, TaskSpec.from_config(config), previous, config)


class P2PlanningTest(unittest.TestCase):
    def test_task_spec_records_goal_and_all_task_thresholds(self):
        config, _, _ = fixture()
        task = TaskSpec.from_config(config)
        np.testing.assert_array_equal(task.goal, [.08, 0., 0.])
        for name in ("position_tolerance_m", "yaw_tolerance_rad",
                     "linear_speed_tolerance_m_s", "angular_speed_tolerance_rad_s",
                     "hold_time_s", "max_episode_time_s"):
            self.assertEqual(getattr(task, name), config["task"][name])
        config["task"]["goal_xy_m"][0] = -.08
        np.testing.assert_array_equal(task.goal, [.08, 0., 0.])

    def test_task_spec_rejects_nonfinite_goal_and_nonpositive_thresholds(self):
        config, _, _ = fixture()
        for key, value in (("goal_xy_m", [float("nan"), 0.]),
                           ("goal_xy_m", [0.]),
                           ("goal_yaw_rad", float("inf")),
                           ("position_tolerance_m", 0.),
                           ("yaw_tolerance_rad", -.01),
                           ("linear_speed_tolerance_m_s", 0.),
                           ("angular_speed_tolerance_rad_s", 0.),
                           ("hold_time_s", 0.),
                           ("max_episode_time_s", 0.)):
            changed = deepcopy(config)
            changed["task"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(PlanningError):
                TaskSpec.from_config(changed)

    def test_recursive_predictions_use_entire_predicted_state_and_stored_actions(self):
        config, states, actions = fixture(2, 3)
        states[:, 0] = [.001, .002]
        actions[0, :, 0] = [.001, .002, .003]
        actions[1, :, 1] = [-.001, -.002, -.003]
        observed = []

        def doubling(state, action):
            observed.append((state.copy(), action.copy()))
            delta = np.zeros_like(state)
            delta[:, 0] = state[:, 0]
            delta[:, 3] = .1 + state[:, 3]
            return delta

        before_states, before_actions = states.copy(), actions.copy()
        # Any attempt to obtain a future physics answer would fail this check.
        with patch("push_align.environment.PlanarPushEnv.step",
                   side_effect=AssertionError("future simulator call forbidden")):
            result = rollout(states, actions, doubling, config)
        np.testing.assert_array_equal(result.valid, [True, True])
        np.testing.assert_allclose(result.states[:, :, 0],
                                   [[.001, .002, .004, .008], [.002, .004, .008, .016]])
        np.testing.assert_allclose(result.states[:, :, 3], np.tile([0., .1, .3, .7], (2, 1)))
        self.assertEqual(len(observed), 3)
        for step, (seen_state, seen_action) in enumerate(observed):
            np.testing.assert_array_equal(seen_state, result.states[:, step])
            np.testing.assert_array_equal(seen_action, actions[:, step])
        np.testing.assert_array_equal(states, before_states)
        np.testing.assert_array_equal(actions, before_actions)
        self.assertEqual(result.states.dtype, np.float64)
        self.assertEqual(result.raw_deltas.dtype, np.float64)
        self.assertEqual(result.raw_deltas.shape, (2, 3, 10))
        np.testing.assert_array_equal(result.failure_steps, [-1, -1])

    def test_two_candidates_known_costs_and_prediction_flip_change_selection(self):
        config, states, actions = fixture(2, 1)
        config = position_only(config)
        config["task"]["goal_xy_m"] = [.01, 0.]
        actions[:, 0, 0] = [.01, -.01]

        def predictor(sign):
            def delta(state, action):
                result = np.zeros_like(state)
                result[:, 0] = sign * .5 * action[:, 0]
                result[:, 6] = result[:, 0]
                return result
            return delta

        first = rollout(states, actions, predictor(1.), config)
        first_cost = costs(first, actions, config)
        # Distances .005 and .015, tolerance .005, dt .1, terminal weight 5.
        np.testing.assert_allclose(first_cost.total, [5.1, 45.9], rtol=0, atol=1e-12)
        index, command, reason = choose(first_cost, first, actions)
        self.assertEqual((index, reason), (0, "selected"))
        np.testing.assert_array_equal(command, [.01, 0.])
        second = rollout(states, actions, predictor(-1.), config)
        second_cost = costs(second, actions, config)
        np.testing.assert_allclose(second_cost.total, [45.9, 5.1], rtol=0, atol=1e-12)
        index, command, reason = choose(second_cost, second, actions)
        self.assertEqual((index, reason), (1, "selected"))
        np.testing.assert_array_equal(command, [-.01, 0.])

    def test_stage_uses_post_action_states_dt_and_previous_applied_action(self):
        config, states, actions = fixture(1, 2)
        config["cost"]["stage_gap"] = 0.
        actions[0] = [[.03, 0.], [0., .03]]
        previous = np.array([-.03, 0.], dtype=np.float64)
        known_delta = np.array([.002, .001, .01, .003, .004, .02,
                                .002, .001, 0., 0.], dtype=np.float64)
        predictions = rollout(states, actions,
                              lambda state, action: np.broadcast_to(known_delta, state.shape).copy(), config)
        result = costs(predictions, actions, config, previous)
        # Both post-action endpoints, not the initial state or terminal alone.
        stage_position = ((.078**2 + .001**2) + (.076**2 + .002**2)) / .005**2
        stage_yaw = (.01**2 + .02**2) / config["task"]["yaw_tolerance_rad"]**2
        # action squared magnitudes .25+.25; changes 1+.5, norm limit .06.
        stage = stage_position + stage_yaw + .02*.5 + .02*1.5
        terminal = (5.*(.076**2 + .002**2)/.005**2
                    + 5.*.02**2/config["task"]["yaw_tolerance_rad"]**2
                    + .1*(.01/.01)**2 + .1*(.04/.05)**2)
        self.assertAlmostEqual(float(result.total[0]), .1*stage + terminal, places=10)
        self.assertTrue(result.components)
        for name, values in result.components.items():
            with self.subTest(component=name):
                self.assertEqual(values.shape, (1,))
                self.assertTrue(np.isfinite(values).all())

    def test_action_change_uses_previous_command_once_then_neighboring_commands(self):
        config, states, actions = fixture(1, 3)
        for key in config["cost"]:
            if key.startswith(("stage_", "terminal_")):
                config["cost"][key] = 0.
        config["cost"]["stage_action_change"] = 1.
        actions[0] = [[.03, 0.], [.03, 0.], [0., .03]]
        predictions = rollout(states, actions, hold, config)
        result = costs(predictions, actions, config, np.array([-.03, 0.], np.float64))
        self.assertAlmostEqual(float(result.total[0]), .1*(1. + 0. + .5), places=14)

    def test_goal_yaw_cost_treats_pi_rotation_as_same_rectangular_pose(self):
        config, states, actions = fixture(2, 1)
        for key in config["cost"]:
            if key.startswith(("stage_", "terminal_")):
                config["cost"][key] = 0.
        config["cost"]["stage_yaw"] = 1.
        config["cost"]["terminal_yaw"] = 5.
        states[:, 2] = [0., math.pi]
        result = costs(rollout(states, actions, hold, config), actions, config)
        np.testing.assert_allclose(result.total, 0., atol=1e-26, rtol=0)

    def test_gap_cost_accounts_for_box_frame_and_positive_contact_gap(self):
        config, states, actions = fixture(2, 1)
        for key in config["cost"]:
            if key.startswith(("stage_", "terminal_")):
                config["cost"][key] = 0.
        config["cost"]["stage_gap"] = 1.
        # The same .002 surface gap on the two differently oriented bodies.
        states[1, 2] = math.pi/2
        states[1, 6:8] = [0., -.05]
        predictions = rollout(states, actions, hold, config)
        np.testing.assert_array_equal(predictions.valid, [True, True])
        result = costs(predictions, actions, config)
        np.testing.assert_allclose(result.total, np.full(2, .1*(.002/.08)**2), atol=1e-15, rtol=0)

    def test_equal_cost_uses_lowest_candidate_index_and_returns_a_copy(self):
        config, states, actions = fixture(3, 2)
        predictions = rollout(states, actions, hold, config)
        result = costs(predictions, actions, config)
        index, command, reason = choose(result, predictions, actions)
        self.assertEqual((index, reason), (0, "selected"))
        command[0] = .123
        self.assertEqual(actions[0, 0, 0], 0.)

    def test_one_nonfinite_prediction_keeps_its_finite_peer_selectable(self):
        config, states, actions = fixture(2, 2)
        actions[:, :, 1] = [[.01, .01], [-.01, -.01]]

        def one_nan(state, action):
            delta = np.zeros_like(state)
            delta[action[:, 1] > 0., 0] = np.nan
            return delta

        result = rollout(states, actions, one_nan, config)
        np.testing.assert_array_equal(result.valid, [False, True])
        self.assertEqual(result.reasons[0], "nonfinite_physical_delta")
        np.testing.assert_array_equal(result.failure_steps, [1, -1])
        self.assertTrue(np.isfinite(result.states[1]).all())
        scored = costs(result, actions, config)
        self.assertTrue(np.isinf(scored.total[0]))
        self.assertTrue(np.isfinite(scored.total[1]))
        index, command, reason = choose(scored, result, actions)
        self.assertEqual((index, reason), (1, "selected"))
        np.testing.assert_array_equal(command, [0., -.01])

    def test_batch_inference_rejection_is_isolated_to_rejected_candidate(self):
        config, states, actions = fixture(3, 2)
        actions[:, :, 1] = [[-.01, -.01], [.01, .01], [0., 0.]]

        def rejects_one(state, action):
            if (action[:, 1] > 0.).any():
                raise ValueError("controlled rejected candidate")
            return np.zeros_like(state)

        result = rollout(states, actions, rejects_one, config)
        np.testing.assert_array_equal(result.valid, [True, False, True])
        self.assertIn("inference_rejected:ValueError", result.reasons[1])
        np.testing.assert_array_equal(result.failure_steps, [-1, 1, -1])
        self.assertTrue(np.isfinite(result.states[[0, 2]]).all())

    def test_predicted_geometry_failures_preserve_other_candidates(self):
        for field, delta_value, reason in (
            (0, .19, "object_corners_outside_workspace"),
            (6, .243, "pusher_extent_outside_workspace"),
            (6, .05, "predicted_deep_penetration"),
        ):
            config, states, actions = fixture(2, 2)
            actions[0, :, 1] = .01

            def invalid_geometry(state, action):
                delta = np.zeros_like(state)
                delta[action[:, 1] > 0., field] = delta_value
                return delta

            with self.subTest(reason=reason):
                result = rollout(states, actions, invalid_geometry, config)
                np.testing.assert_array_equal(result.valid, [False, True])
                self.assertEqual(result.reasons[0], reason)
                self.assertEqual(int(result.failure_steps[0]), 1)
                self.assertEqual(choose(costs(result, actions, config), result, actions)[0], 1)

    def test_raw_yaw_increment_is_checked_before_wrap_at_both_pi_boundaries(self):
        for sign in (-1., 1.):
            config, states, actions = fixture(2, 1)
            actions[0, :, 1] = .01

            def invalid_rotation(state, action):
                delta = np.zeros_like(state)
                delta[action[:, 1] > 0., 2] = sign*math.pi
                return delta

            with self.subTest(sign=sign):
                result = rollout(states, actions, invalid_rotation, config)
                np.testing.assert_array_equal(result.valid, [False, True])
                self.assertEqual(result.reasons[0], "raw_delta_yaw_abs_ge_pi")
                self.assertEqual(float(result.raw_deltas[0, 0, 2]), sign*math.pi)

    def test_small_yaw_increment_wraps_two_pi_without_being_invalid(self):
        config, states, actions = fixture(1, 1)
        states[0, 2] = math.pi-.01

        def rotate(state, action):
            delta = np.zeros_like(state)
            delta[:, 2] = .02
            return delta

        result = rollout(states, actions, rotate, config)
        self.assertTrue(bool(result.valid[0]))
        self.assertAlmostEqual(float(result.states[0, 1, 2]), -math.pi+.01, places=14)
        self.assertEqual(float(result.raw_deltas[0, 0, 2]), .02)

    def test_failure_at_later_step_preserves_earlier_predictions_and_peer(self):
        config, states, actions = fixture(2, 3)
        actions[0, 1:, 1] = .01

        def delayed_nan(state, action):
            delta = np.zeros_like(state)
            delta[:, 0] = .001
            delta[:, 6] = .001
            delta[action[:, 1] > 0., 3] = np.nan
            return delta

        result = rollout(states, actions, delayed_nan, config)
        np.testing.assert_array_equal(result.valid, [False, True])
        np.testing.assert_array_equal(result.failure_steps, [2, -1])
        np.testing.assert_allclose(result.states[0, :2, 0], [0., .001])
        self.assertTrue(np.isnan(result.states[0, 2:]).all())
        np.testing.assert_allclose(result.states[1, :, 0], [0., .001, .002, .003])

    def test_invalid_initial_geometry_is_not_sent_to_predictor(self):
        config, states, actions = fixture(2, 2)
        states[0, 6] = 0.  # Pusher centre inside the rectangular object.
        observed = []

        def record(state, action):
            observed.append(state.copy())
            return np.zeros_like(state)

        result = rollout(states, actions, record, config)
        np.testing.assert_array_equal(result.valid, [False, True])
        np.testing.assert_array_equal(result.failure_steps, [0, -1])
        self.assertEqual(result.reasons[0], "predicted_deep_penetration")
        self.assertEqual(len(observed), 2)
        self.assertTrue(all(batch.shape == (1, 10) for batch in observed))
        self.assertTrue(all(batch[0, 6] == -.05 for batch in observed))

    def test_predictor_cannot_mutate_caller_state_actions_or_rollout_history(self):
        config, states, actions = fixture(2, 2)
        before_states, before_actions = states.copy(), actions.copy()

        def mutates(state, action):
            delta = np.zeros_like(state)
            state[:] = 123.
            action[:] = 456.
            return delta

        result = rollout(states, actions, mutates, config)
        np.testing.assert_array_equal(result.valid, [True, True])
        np.testing.assert_array_equal(states, before_states)
        np.testing.assert_array_equal(actions, before_actions)
        np.testing.assert_array_equal(result.states, np.repeat(before_states[:, None], 3, axis=1))

    def test_simulator_prediction_configuration_is_rejected(self):
        config, states, actions = fixture(2, 2)
        config["planner"]["use_simulator_for_candidate_prediction"] = True
        with self.assertRaises(PlanningError):
            rollout(states, actions, hold, config)

    def test_finite_high_predicted_velocity_has_no_undocumented_physical_cap(self):
        config, states, actions = fixture(1, 1)

        def fast(state, action):
            delta = np.zeros_like(state)
            delta[:, 3:6] = [1.e3, -1.e3, 1.e3]
            delta[:, 8:10] = [1.e3, -1.e3]
            return delta

        result = rollout(states, actions, fast, config)
        self.assertTrue(bool(result.valid[0]))
        np.testing.assert_array_equal(result.states[0, 1, 3:6], [1.e3, -1.e3, 1.e3])

    def test_all_invalid_candidates_request_zero_and_explicit_failure(self):
        config, states, actions = fixture(2, 2)

        def invalid(state, action):
            return np.full_like(state, np.nan)

        predictions = rollout(states, actions, invalid, config)
        scored = costs(predictions, actions, config)
        np.testing.assert_array_equal(predictions.valid, [False, False])
        self.assertTrue(np.isinf(scored.total).all())
        index, command, reason = choose(scored, predictions, actions)
        self.assertIsNone(index)
        self.assertEqual(reason, "planner_invalid")
        np.testing.assert_array_equal(command, np.zeros(2))
        self.assertEqual(command.dtype, np.float64)

    def test_rollout_rejects_bad_shape_dtype_nonfinite_and_norm_excess(self):
        config, states, actions = fixture(2, 2)
        bad_inputs = (
            (states.astype(np.float32), actions),
            (states, actions.astype(np.float32)),
            (states[0], actions),
            (states, actions[0]),
            (states, actions[:, :, :1]),
            (states, actions[:1]),
            (states, np.zeros((2, 0, 2), dtype=np.float64)),
            (np.full_like(states, np.nan), actions),
            (states, np.full_like(actions, np.inf)),
            (states, np.full_like(actions, .06)),
        )
        for state, action in bad_inputs:
            with self.subTest(state_shape=state.shape, action_shape=action.shape,
                              state_dtype=state.dtype, action_dtype=action.dtype), self.assertRaises(PlanningError):
                rollout(state, action, hold, config)

    def test_command_norm_limit_is_inclusive_without_axiswise_clipping(self):
        config, states, actions = fixture(1, 1)
        actions[0, 0] = [config["pusher"]["command_speed_norm_max_m_s"], 0.]
        np.testing.assert_array_equal(rollout(states, actions, hold, config).valid, [True])
        actions[0, 0] = [.05, .05]
        with self.assertRaises(PlanningError):
            rollout(states, actions, hold, config)

    def test_score_rejects_bad_previous_action_or_candidate_shape(self):
        config, states, actions = fixture(2, 2)
        predictions = rollout(states, actions, hold, config)
        for previous in (np.zeros(2, np.float32), np.zeros((1, 2), np.float64),
                         np.array([np.nan, 0.], np.float64), np.array([.06, .06], np.float64)):
            with self.subTest(previous=previous), self.assertRaises(PlanningError):
                costs(predictions, actions, config, previous)
        with self.assertRaises(PlanningError):
            costs(predictions, actions[:1], config)

    def test_choose_rejects_nan_or_negative_infinity_cost(self):
        config, states, actions = fixture(2, 2)
        predictions = rollout(states, actions, hold, config)
        for value in (np.nan, -np.inf):
            invalid_cost = CostBatch(np.array([value, 1.], np.float64), {})
            with self.subTest(value=value), self.assertRaises(PlanningError):
                choose(invalid_cost, predictions, actions)

    def test_positive_infinite_cost_cannot_win_and_all_infinite_costs_fail(self):
        config, states, actions = fixture(2, 2)
        predictions = rollout(states, actions, hold, config)
        scored = CostBatch(np.array([np.inf, 1.], np.float64), {})
        self.assertEqual(choose(scored, predictions, actions)[0], 1)
        all_infinite = CostBatch(np.full(2, np.inf, np.float64), {})
        index, command, reason = choose(all_infinite, predictions, actions)
        self.assertIsNone(index)
        self.assertEqual(reason, "planner_invalid")
        np.testing.assert_array_equal(command, np.zeros(2))

    def test_cost_overflow_preserves_valid_prediction_but_excludes_it_from_selection(self):
        config, states, actions = fixture(2, 1)
        actions[0, 0, 1] = .01

        def finite_extreme_speed(state, action):
            delta = np.zeros_like(state)
            delta[action[:, 1] > 0., 3] = 1.e308
            return delta

        predictions = rollout(states, actions, finite_extreme_speed, config)
        np.testing.assert_array_equal(predictions.valid, [True, True])
        result = costs(predictions, actions, config)
        self.assertEqual(float(result.total[0]), np.inf)
        self.assertTrue(np.isfinite(result.total[1]))
        self.assertEqual(choose(result, predictions, actions)[0], 1)

    def test_finite_delta_integration_overflow_is_isolated_to_its_candidate(self):
        config, states, actions = fixture(2, 1)
        states[0, 3] = 1.e308
        actions[0, 0, 1] = .01

        def overflowing_sum(state, action):
            delta = np.zeros_like(state)
            delta[action[:, 1] > 0., 3] = 1.e308
            return delta

        predictions = rollout(states, actions, overflowing_sum, config)
        np.testing.assert_array_equal(predictions.valid, [False, True])
        self.assertEqual(predictions.reasons[0], "nonfinite_predicted_state")
        self.assertTrue(np.isfinite(predictions.raw_deltas).all())
        self.assertEqual(choose(costs(predictions, actions, config), predictions, actions)[0], 1)


if __name__ == "__main__":
    unittest.main()
