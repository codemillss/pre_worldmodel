"""Independent task-boundary cases for pure learned-endpoint H10 ranking.

Controlled prediction arrays verify ranking and endpoint-duration arithmetic,
not actual physical holding, learned accuracy or closed-loop task success.
"""
from dataclasses import replace
import math
import unittest

import numpy as np

from push_align.handoff_cost import HandoffCostError, choose_handoff, score_handoff
from push_align.planning import CostBatch, PredictionBatch, TaskSpec


def fixture(n=2):
    task = TaskSpec(np.zeros(3, np.float64), .005, math.pi/60, .01, .05, .3, 30.)
    states = np.zeros((n, 11, 10), np.float64)
    states[:, :, 0] = .006
    states[:, :, 6] = -.05
    prediction = PredictionBatch(states, np.zeros((n, 10, 10), np.float64),
        np.ones(n, np.bool_), tuple('' for _ in range(n)), np.full(n, -1, np.int64))
    legacy = CostBatch(np.ones(n, np.float64), {})
    actions = np.zeros((n, 10, 2), np.float64)
    return task, prediction, legacy, actions


def evaluate(prediction, task, legacy, **kwargs):
    return score_handoff(prediction, task, legacy, .1, **kwargs)


class HandoffCostTest(unittest.TestCase):
    def test_nonqualifying_pose_has_explicit_ratios_and_excess(self):
        task, prediction, legacy, _ = fixture(1)
        scores = evaluate(prediction, task, legacy)
        np.testing.assert_allclose(scores.ratios[0, :, 0], 1.2, atol=1e-15, rtol=0)
        np.testing.assert_array_equal(scores.ratios[0, :, 1:], 0.)
        self.assertFalse(scores.ready[0])
        self.assertEqual(scores.first_ready_step[0], -1)
        self.assertAlmostEqual(scores.terminal_max_excess[0], .2)
        self.assertAlmostEqual(scores.integrated_max_excess[0], .2)

    def test_three_new_qualifying_endpoints_do_not_cover_three_intervals(self):
        task, prediction, legacy, _ = fixture(1)
        prediction.states[0, 8:, 0] = 0.
        scores = evaluate(prediction, task, legacy)
        self.assertFalse(scores.ready[0])
        np.testing.assert_allclose(scores.held_time_s[0, 8:], [0., .1, .2], atol=1e-15, rtol=0)

    def test_four_new_qualifying_endpoints_reach_exact_three_intervals(self):
        task, prediction, legacy, _ = fixture(1)
        prediction.states[0, 7:, 0] = 0.
        scores = evaluate(prediction, task, legacy)
        self.assertTrue(scores.ready[0])
        self.assertEqual(scores.first_ready_step[0], 10)
        self.assertAlmostEqual(scores.held_time_s[0, 10], .3)

    def test_qualified_initial_endpoint_counts_first_interval_without_initial_hold(self):
        task, prediction, legacy, _ = fixture(1)
        prediction.states[0, :, 0] = 0.
        scores = evaluate(prediction, task, legacy, initial_previous_qualifies=True)
        self.assertEqual(scores.first_ready_step[0], 3)
        self.assertEqual(scores.held_time_s[0, 0], 0.)

    def test_measured_initial_hold_can_complete_at_first_future_boundary(self):
        task, prediction, legacy, _ = fixture(1)
        prediction.states[0, :, 0] = 0.
        scores = evaluate(prediction, task, legacy, initial_held_time_s=.25,
                          initial_previous_qualifies=True)
        self.assertEqual(scores.first_ready_step[0], 1)
        self.assertAlmostEqual(scores.held_time_s[0, 1], .35)

    def test_interruption_resets_hold_before_new_run(self):
        task, prediction, legacy, _ = fixture(1)
        prediction.states[0, 1:4, 0] = 0.
        prediction.states[0, 5:9, 0] = 0.
        scores = evaluate(prediction, task, legacy)
        self.assertEqual(scores.first_ready_step[0], 8)
        np.testing.assert_allclose(scores.held_time_s[0, :9],
            [0., 0., .1, .2, 0., 0., .1, .2, .3], atol=1e-15, rtol=0)

    def test_first_ready_event_truncates_new_terms_even_if_later_endpoints_leave(self):
        task, prediction, legacy, _ = fixture(1)
        prediction.states[0, 1:5, 0] = 0.
        prediction.states[0, 5:, 0] = .015
        legacy.total[0] = 123.
        scores = evaluate(prediction, task, legacy)
        self.assertEqual(scores.first_ready_step[0], 4)
        self.assertEqual(scores.terminal_max_excess[0], 0.)
        self.assertEqual(scores.integrated_max_excess[0], 0.)
        self.assertEqual(scores.held_time_s[0, 5], 0.)
        np.testing.assert_array_equal(scores.rank_keys[0], [0., 0., 0., 123.])

    def test_integral_uses_post_action_endpoints_through_first_ready(self):
        task, prediction, legacy, _ = fixture(1)
        prediction.states[0, 1, 0] = .015
        prediction.states[0, 2:6, 0] = 0.
        scores = evaluate(prediction, task, legacy)
        self.assertEqual(scores.first_ready_step[0], 5)
        self.assertAlmostEqual(scores.integrated_max_excess[0], .2)
        self.assertEqual(scores.terminal_max_excess[0], 0.)

    def test_pi_symmetric_yaw_treats_pi_as_same_orientation(self):
        task, prediction, legacy, _ = fixture(1)
        prediction.states[0, 1:, 0] = 0.
        prediction.states[0, 1:, 2] = math.pi
        scores = evaluate(prediction, task, legacy)
        self.assertTrue(scores.ready[0])
        np.testing.assert_allclose(scores.ratios[0, 1:, 1], 0., atol=1e-14, rtol=0)

    def test_yaw_outside_tolerance_blocks_hold_while_position_is_inside(self):
        task, prediction, legacy, _ = fixture(1)
        prediction.states[0, 1:, 0] = 0.
        prediction.states[0, 1:, 2] = 2*task.yaw_tolerance_rad
        scores = evaluate(prediction, task, legacy)
        self.assertFalse(scores.ready[0])
        self.assertAlmostEqual(scores.terminal_max_excess[0], 1.)

    def test_linear_speed_alone_blocks_pose_holding(self):
        task, prediction, legacy, _ = fixture(1)
        prediction.states[0, 1:, 0] = 0.
        prediction.states[0, 1:, 3:5] = [.012, .016]
        scores = evaluate(prediction, task, legacy)
        self.assertFalse(scores.ready[0])
        np.testing.assert_allclose(scores.ratios[0, 1:, 2], 2., atol=1e-15, rtol=0)
        self.assertEqual(scores.terminal_max_excess[0], 1.)

    def test_absolute_angular_speed_alone_blocks_pose_holding(self):
        task, prediction, legacy, _ = fixture(1)
        prediction.states[0, 1:, 0] = 0.
        prediction.states[0, 1:, 5] = -.06
        scores = evaluate(prediction, task, legacy)
        self.assertFalse(scores.ready[0])
        self.assertAlmostEqual(scores.terminal_max_excess[0], .2)

    def test_exact_linear_and_angular_speed_thresholds_qualify(self):
        task, prediction, legacy, _ = fixture(1)
        prediction.states[0, 1:, 0] = 0.
        prediction.states[0, 1:, 3] = .01
        prediction.states[0, 1:, 5] = -.05
        scores = evaluate(prediction, task, legacy)
        self.assertTrue(scores.ready[0])
        np.testing.assert_array_equal(scores.ratios[0, 1:, 2:], 1.)

    def test_exact_position_and_yaw_thresholds_qualify(self):
        task, prediction, legacy, _ = fixture(1)
        prediction.states[0, 1:, 0] = task.position_tolerance_m
        prediction.states[0, 1:, 2] = task.yaw_tolerance_rad
        scores = evaluate(prediction, task, legacy)
        self.assertTrue(scores.ready[0])
        np.testing.assert_array_equal(scores.ratios[0, 1:, :2], 1.)

    def test_maximum_excess_does_not_sum_or_reward_within_tolerance_channels(self):
        task, prediction, legacy, _ = fixture(1)
        prediction.states[0, 1:, 0] = .01
        prediction.states[0, 1:, 5] = .125
        scores = evaluate(prediction, task, legacy)
        self.assertAlmostEqual(scores.terminal_max_excess[0], 1.5)
        self.assertAlmostEqual(scores.integrated_max_excess[0], 1.5)

    def test_surrogate_ready_has_priority_over_better_legacy_cost(self):
        task, prediction, legacy, actions = fixture()
        prediction.states[0, 1:, 0] = 0.
        legacy.total[:] = [1000., 0.]
        index, _, reason = choose_handoff(evaluate(prediction, task, legacy), actions)
        self.assertEqual((index, reason), (0, 'selected'))

    def test_final_excess_precedes_integrated_excess_and_original_J(self):
        task, prediction, legacy, actions = fixture()
        prediction.states[0, 1:10, 0] = .1
        prediction.states[0, 10, 0] = .0055
        legacy.total[:] = [100., 0.]
        self.assertEqual(choose_handoff(evaluate(prediction, task, legacy), actions)[0], 0)

    def test_integrated_excess_breaks_equal_terminal_excess_before_original_J(self):
        task, prediction, legacy, actions = fixture()
        prediction.states[0, 1:10, 0] = .01
        legacy.total[:] = [0., 100.]
        self.assertEqual(choose_handoff(evaluate(prediction, task, legacy), actions)[0], 1)

    def test_legacy_whole_horizon_J_breaks_equal_task_keys(self):
        task, prediction, legacy, actions = fixture()
        legacy.total[:] = [2., 1.]
        self.assertEqual(choose_handoff(evaluate(prediction, task, legacy), actions)[0], 1)

    def test_exact_tie_uses_original_index_and_returns_action_copy(self):
        task, prediction, legacy, actions = fixture(3)
        actions[:, 0, 0] = [.01, .02, .03]
        index, selected, reason = choose_handoff(evaluate(prediction, task, legacy), actions)
        self.assertEqual((index, reason), (0, 'selected'))
        selected[0] = 10.
        self.assertEqual(actions[0, 0, 0], .01)

    def test_dimensionally_scaled_task_states_preserve_ratios_and_choice(self):
        task, prediction, legacy, actions = fixture()
        prediction.states[1, 1:, 0] = .0055
        first = evaluate(prediction, task, legacy)
        scaled = replace(task, goal=task.goal*1000, position_tolerance_m=5.,
                         linear_speed_tolerance_m_s=10.)
        changed = prediction.states.copy()
        changed[:, :, [0, 1, 3, 4, 6, 7, 8, 9]] *= 1000.
        second = evaluate(replace(prediction, states=changed), scaled, legacy)
        np.testing.assert_allclose(first.ratios, second.ratios, atol=1e-15, rtol=0)
        self.assertEqual(choose_handoff(first, actions)[0], choose_handoff(second, actions)[0])

    def test_invalid_full_horizon_cannot_win_after_apparent_early_event(self):
        task, prediction, legacy, actions = fixture()
        prediction.states[0, 1:5, 0] = 0.
        changed = replace(prediction, valid=np.array([False, True]),
                          reasons=('workspace_exit', ''), failure_steps=np.array([8, -1], np.int64))
        scores = evaluate(changed, task, legacy)
        self.assertFalse(scores.eligible[0])
        self.assertFalse(scores.ready[0])
        self.assertEqual(scores.first_ready_step[0], -1)
        self.assertEqual(scores.reasons[0], 'prediction_invalid:workspace_exit')
        self.assertEqual(choose_handoff(scores, actions)[0], 1)

    def test_nonfinite_state_excludes_only_that_candidate(self):
        task, prediction, legacy, actions = fixture()
        prediction.states[0, 7, 0] = float('nan')
        scores = evaluate(prediction, task, legacy)
        np.testing.assert_array_equal(scores.eligible, [False, True])
        self.assertEqual(scores.reasons[0], 'prediction_nonfinite')
        self.assertEqual(choose_handoff(scores, actions)[0], 1)

    def test_nonfinite_raw_delta_excludes_only_that_candidate(self):
        task, prediction, legacy, _ = fixture()
        prediction.raw_deltas[0, 2, 0] = float('inf')
        scores = evaluate(prediction, task, legacy)
        np.testing.assert_array_equal(scores.eligible, [False, True])
        self.assertEqual(scores.reasons[0], 'prediction_nonfinite')

    def test_positive_infinite_legacy_J_remains_ineligible(self):
        task, prediction, legacy, actions = fixture()
        legacy.total[0] = float('inf')
        scores = evaluate(prediction, task, legacy)
        self.assertEqual(scores.reasons[0], 'legacy_cost_nonfinite')
        self.assertEqual(choose_handoff(scores, actions)[0], 1)

    def test_all_ineligible_returns_unexecuted_zero_and_invalid_reason(self):
        task, prediction, legacy, actions = fixture()
        legacy.total[:] = float('inf')
        index, selected, reason = choose_handoff(evaluate(prediction, task, legacy), actions)
        self.assertEqual((index, reason), (None, 'planner_invalid'))
        np.testing.assert_array_equal(selected, np.zeros(2, np.float64))

    def test_finite_physical_metric_normalization_overflow_is_not_repaired(self):
        task, prediction, legacy, _ = fixture()
        prediction.states[0, 1:, 0] = 1e308
        scores = evaluate(prediction, task, legacy)
        np.testing.assert_array_equal(scores.eligible, [False, True])
        self.assertEqual(scores.reasons[0], 'handoff_metrics_overflow')
        self.assertTrue(np.isinf(scores.ratios[0, 1:, 0]).all())

    def test_finite_ratios_accumulation_overflow_is_not_repaired(self):
        task, prediction, legacy, _ = fixture()
        prediction.states[0, 1:, 0] = 5e305
        scores = evaluate(prediction, task, legacy)
        self.assertFalse(scores.eligible[0])
        self.assertEqual(scores.reasons[0], 'handoff_score_overflow')
        self.assertTrue(np.isfinite(scores.ratios[0]).all())

    def test_initial_observed_qualification_must_match_endpoint(self):
        task, prediction, legacy, _ = fixture(1)
        with self.assertRaises(HandoffCostError):
            evaluate(prediction, task, legacy, initial_previous_qualifies=True)
        prediction.states[0, 0, 0] = 0.
        with self.assertRaises(HandoffCostError):
            evaluate(prediction, task, legacy)

    def test_invalid_initial_hold_values_and_flag_are_contract_errors(self):
        task, prediction, legacy, _ = fixture(1)
        for value in (True, -.01, float('inf'), float('nan'), .3):
            with self.subTest(value=value), self.assertRaises(HandoffCostError):
                evaluate(prediction, task, legacy, initial_held_time_s=value)
        with self.assertRaises(HandoffCostError):
            evaluate(prediction, task, legacy, initial_held_time_s=.1)
        with self.assertRaises(HandoffCostError):
            evaluate(prediction, task, legacy, initial_previous_qualifies=0)

    def test_wrong_dt_and_nonfinite_dt_are_contract_errors(self):
        task, prediction, legacy, _ = fixture(1)
        for value in (True, 0., -.1, .05, float('inf'), float('nan')):
            with self.subTest(value=value), self.assertRaises(HandoffCostError):
                score_handoff(prediction, task, legacy, value)

    def test_state_shape_and_dtype_are_explicit_H10_contract(self):
        task, prediction, legacy, _ = fixture(1)
        for value in (prediction.states.astype(np.float32), prediction.states[:, :-1],
                      prediction.states[:, :, :-1], np.zeros((0, 11, 10), np.float64)):
            with self.subTest(shape=value.shape), self.assertRaises(HandoffCostError):
                evaluate(replace(prediction, states=value), task, legacy)

    def test_delta_mask_reason_and_failure_step_contracts_are_checked(self):
        task, prediction, legacy, _ = fixture(1)
        changes = (
            {'raw_deltas': prediction.raw_deltas.astype(np.float32)},
            {'valid': np.ones(1, np.int64)}, {'valid': np.ones(2, np.bool_)},
            {'reasons': ['']}, {'reasons': ('bad',)}, {'failure_steps': np.array([0], np.int64)},
            {'failure_steps': np.array([-1], np.int32)},
        )
        for change in changes:
            with self.subTest(change=tuple(change)), self.assertRaises(HandoffCostError):
                evaluate(replace(prediction, **change), task, legacy)

    def test_invalid_task_contract_is_rejected(self):
        task, prediction, legacy, _ = fixture(1)
        for changed in (None, replace(task, goal=np.zeros(3, np.float32)),
                        replace(task, position_tolerance_m=0.), replace(task, hold_time_s=float('inf'))):
            with self.subTest(task=changed), self.assertRaises(HandoffCostError):
                evaluate(prediction, changed, legacy)

    def test_legacy_cost_shape_dtype_nan_negative_and_components_are_checked(self):
        task, prediction, legacy, _ = fixture(1)
        for changed in (CostBatch(np.ones(2, np.float64), {}),
                        CostBatch(np.ones(1, np.float32), {}),
                        CostBatch(np.array([float('nan')]), {}),
                        CostBatch(np.array([-1.]), {}), CostBatch(np.array([-float('inf')]), {}),
                        CostBatch(np.ones(1), {'component': np.ones(2)})):
            with self.subTest(cost=changed), self.assertRaises(HandoffCostError):
                evaluate(prediction, task, changed)

    def test_selection_rejects_malformed_action_shape_dtype_or_nonfinite(self):
        task, prediction, legacy, actions = fixture(1)
        scores = evaluate(prediction, task, legacy)
        bad_nan = actions.copy(); bad_nan[0, 0, 0] = float('nan')
        for changed in (actions.astype(np.float32), actions[:, :-1], bad_nan):
            with self.subTest(shape=changed.shape), self.assertRaises(HandoffCostError):
                choose_handoff(scores, changed)

    def test_selection_rejects_forged_nonfinite_keys_or_inconsistent_ready_key(self):
        task, prediction, legacy, actions = fixture(1)
        scores = evaluate(prediction, task, legacy)
        for key in (float('nan'), float('inf'), -1., 0.):
            changed = scores.rank_keys.copy(); changed[0, 0] = key
            with self.subTest(key=key), self.assertRaises(HandoffCostError):
                choose_handoff(replace(scores, rank_keys=changed), actions)

    def test_selection_rejects_nonfinite_or_negative_eligible_diagnostics(self):
        task, prediction, legacy, actions = fixture(1)
        scores = evaluate(prediction, task, legacy)
        for field, coordinate, value in (('ratios', (0, 2, 1), float('nan')),
                ('ratios', (0, 2, 1), -1.), ('held_time_s', (0, 2), float('inf')),
                ('held_time_s', (0, 2), -.1)):
            changed = getattr(scores, field).copy(); changed[coordinate] = value
            with self.subTest(field=field, value=value), self.assertRaises(HandoffCostError):
                choose_handoff(replace(scores, **{field: changed}), actions)

    def test_selection_rejects_claimed_ready_on_ineligible_row(self):
        task, prediction, legacy, actions = fixture(1)
        legacy.total[0] = float('inf')
        scores = evaluate(prediction, task, legacy)
        with self.assertRaises(HandoffCostError):
            choose_handoff(replace(scores, ready=np.array([True]),
                                    first_ready_step=np.array([3], np.int64)), actions)

    def test_inputs_are_retained_without_mutation_or_regeneration(self):
        task, prediction, legacy, actions = fixture()
        snapshots = [array.copy() for array in (task.goal, prediction.states,
            prediction.raw_deltas, prediction.valid, prediction.failure_steps, legacy.total, actions)]
        choose_handoff(evaluate(prediction, task, legacy), actions)
        for before, after in zip(snapshots, (task.goal, prediction.states,
            prediction.raw_deltas, prediction.valid, prediction.failure_steps, legacy.total, actions)):
            np.testing.assert_array_equal(before, after)


if __name__ == '__main__':
    unittest.main()
