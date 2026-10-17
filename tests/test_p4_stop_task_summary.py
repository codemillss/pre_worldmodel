"""Recorded-state checks only; no simulator, model, or training execution."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tomllib
import unittest
from unittest.mock import patch

import numpy as np

from push_align.config import ROOT
from push_align.task import consume_step, start_task

scripts_path = str(ROOT / 'scripts')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from p4_stop_task_summary import summarize_task_prefix
from test_p2_task import ACTION_DT, observation, state, step, task


def recorded_step(start, initial=None, **kwargs):
    value = step(start, initial, **kwargs)
    # The original task fixture has no contact fields. Add an explicit recorded
    # noncontact observation without changing that old fixture or its tests.
    for sample in value.diagnostics['samples']:
        sample['contacts'] = []
    return value


class StopTaskSummaryTest(unittest.TestCase):
    def setUp(self):
        self.task = task()
        self.profile = tomllib.loads((ROOT / 'configs/acceptance.v1.toml').read_text())
        self.status = start_task(observation(sim_time=12.5), self.task, ACTION_DT)

    def rows(self, count, *, start=12.5, value=None):
        return [recorded_step(start + index * ACTION_DT, value)
                for index in range(count)]

    def summarize(self, rows, status=None):
        return summarize_task_prefix(rows, self.status if status is None else status,
                                     self.task, ACTION_DT, self.profile)

    def test_qualifying_endpoint_without_full_hold_is_not_ready(self):
        result = self.summarize(self.rows(2))
        self.assertTrue(result['qualifies_at_end'])
        self.assertIsNone(result['first_ready_interval'])
        self.assertEqual(result['status']['phase'], 'holding')
        self.assertAlmostEqual(result['status']['held_time_s'], .2)

    def test_first_ready_is_one_based_and_postterminal_raw_intervals_are_not_consumed(self):
        rows = self.rows(5)
        with patch('p4_stop_task_summary.consume_step', wraps=consume_step) as consume:
            result = self.summarize(rows)
        self.assertEqual(consume.call_count, 3)
        self.assertEqual(result['first_ready_interval'], 3)
        self.assertEqual([row['index'] for row in result['timeline']], [1, 2, 3, 4, 5])
        self.assertEqual([row['task_consumed'] for row in result['timeline']],
                         [True, True, True, False, False])
        self.assertAlmostEqual(result['status']['last_sim_time'], 12.8)
        self.assertAlmostEqual(result['status']['held_time_s'], .3)

    def test_existing_source_hold_and_elapsed_clock_are_preserved(self):
        # A source at tick99 already has a clock and .2 s of qualifying hold.
        source = replace(self.status, initial_sim_time=2.5, last_sim_time=12.5,
                         held_time_s=.2, previous_qualifies=True)
        before = asdict(source)
        result = self.summarize(self.rows(2), source)
        self.assertEqual(result['first_ready_interval'], 1)
        self.assertEqual(result['status']['initial_sim_time'], 2.5)
        self.assertAlmostEqual(result['status']['last_sim_time'], 12.6)
        self.assertEqual(asdict(source), before)

    def test_interval_start_entry_does_not_credit_time_before_first_qualifying_sample(self):
        outside = state(x=.02)
        source = start_task(observation(outside, 12.5), self.task, ACTION_DT)
        first = recorded_step(12.5, outside, values=[state(), state()])
        result = self.summarize([first, *self.rows(2, start=12.6)], source)
        self.assertIsNone(result['first_ready_interval'])
        self.assertAlmostEqual(result['status']['held_time_s'], .25)
        self.assertTrue(result['qualifies_at_end'])

    def test_interior_speed_exit_breaks_hold_even_with_good_endpoints(self):
        rows = self.rows(3)
        rows[1] = recorded_step(12.6, values=[state(vx=.02), state()])
        result = self.summarize(rows)
        self.assertTrue(result['qualifies_at_end'])
        self.assertIsNone(result['first_ready_interval'])
        self.assertAlmostEqual(result['status']['held_time_s'], .1)

    def test_later_raw_pose_exit_remains_visible_without_overwriting_earlier_ready(self):
        rows = self.rows(3)
        rows.append(recorded_step(12.8, values=[state(x=.02), state(x=.02)]))
        result = self.summarize(rows)
        self.assertEqual(result['first_ready_interval'], 3)
        self.assertEqual(result['status']['phase'], 'ready')
        self.assertFalse(result['qualifies_at_end'])
        self.assertFalse(result['timeline'][-1]['task_consumed'])
        self.assertFalse(result['timeline'][-1]['metrics']['qualifies'])

    def test_geometry_failure_at_ready_boundary_prevents_success(self):
        rows = self.rows(3)
        rows[-1].diagnostics['samples'][-1]['contacts'] = [{
            'pair': 'pusher_object', 'distance_m': -.002,
            'normal_force_N': 1., 'normal_world': [1., 0., 0.],
            'normal_center_cosine': 1.}]
        result = self.summarize(rows)
        self.assertIsNone(result['first_ready_interval'])
        self.assertEqual(result['status']['terminal_reason'], 'geometry_invalid')
        self.assertIn('contact_penetration_exceeded', result['status']['violations'])
        self.assertTrue(result['qualifies_at_end'])
        self.assertFalse(result['timeline'][-1]['geometry_pass'])

    def test_physics_failure_outranks_geometry_and_is_not_consumed_again(self):
        rows = self.rows(3)
        rows[0] = recorded_step(12.5, valid=False, violations=['physics_warning'])
        rows[0].diagnostics['samples'][1]['contacts'] = [{
            'pair': 'pusher_object', 'distance_m': -.002,
            'normal_force_N': 1., 'normal_world': [1., 0., 0.],
            'normal_center_cosine': 1.}]
        with patch('p4_stop_task_summary.consume_step', wraps=consume_step) as consume:
            result = self.summarize(rows)
        self.assertEqual(consume.call_count, 1)
        self.assertEqual(result['status']['terminal_reason'], 'physics_error')
        self.assertIsNone(result['first_ready_interval'])
        self.assertEqual(len(result['timeline']), 3)
        self.assertEqual([row['task_consumed'] for row in result['timeline']], [True, False, False])

    def test_supplied_failed_source_never_restarts_task_or_receives_hold_credit(self):
        failed = replace(self.status, phase='failed', terminal_reason='timeout')
        with patch('p4_stop_task_summary.consume_step', wraps=consume_step) as consume:
            result = self.summarize(self.rows(4), failed)
        consume.assert_not_called()
        self.assertEqual(result['status'], asdict(failed))
        self.assertIsNone(result['first_ready_interval'])
        self.assertTrue(result['qualifies_at_end'])

    def test_near_deadline_source_times_out_without_resetting_elapsed_time(self):
        source = replace(self.status, initial_sim_time=2.5, last_sim_time=32.4,
                         held_time_s=0., previous_qualifies=False)
        rows = [recorded_step(32.4, state(x=.02), values=[state(), state()]),
                recorded_step(32.5)]
        result = self.summarize(rows, source)
        self.assertEqual(result['status']['terminal_reason'], 'timeout')
        self.assertAlmostEqual(result['status']['last_sim_time'], 32.5)
        self.assertEqual([row['task_consumed'] for row in result['timeline']], [True, False])
        self.assertTrue(result['qualifies_at_end'])

    def test_nonfinite_endpoint_is_preserved_as_invalid_not_a_success_or_json_nan(self):
        bad = state(); bad[0] = np.nan
        rows = [recorded_step(12.5, values=[bad, bad], valid=False,
                              violations=['nonfinite_state'])]
        result = self.summarize(rows)
        self.assertFalse(result['qualifies_at_end'])
        self.assertIsNone(result['timeline'][0]['metrics'])
        self.assertEqual(result['status']['terminal_reason'], 'physics_error')
        json.dumps(result, allow_nan=False)
        self.assertTrue(np.isnan(rows[0].observation.state[0]))

    def test_input_task_status_raw_records_and_profile_remain_unchanged(self):
        rows = self.rows(4)
        before = [(row.substep_states.copy(), deepcopy(row.diagnostics)) for row in rows]
        status_before, goal_before, profile_before = asdict(self.status), self.task.goal.copy(), deepcopy(self.profile)
        result = self.summarize(rows)
        result['status']['violations'] = ('external_change',)
        for row, (states_before, diagnostics_before) in zip(rows, before):
            np.testing.assert_array_equal(row.substep_states, states_before)
            self.assertEqual(row.diagnostics, diagnostics_before)
        self.assertEqual(asdict(self.status), status_before)
        np.testing.assert_array_equal(self.task.goal, goal_before)
        self.assertEqual(self.profile, profile_before)

    def test_absent_geometry_records_or_disabled_profile_cannot_look_like_a_pass(self):
        rows = self.rows(1)
        del rows[0].diagnostics['samples'][0]['contacts']
        with self.assertRaises(ValueError):
            self.summarize(rows)
        profile = deepcopy(self.profile); profile['physics_suite']['check_contact_geometry'] = False
        with self.assertRaises(ValueError):
            summarize_task_prefix(self.rows(1), self.status, self.task, ACTION_DT, profile)

    def test_empty_prefix_and_wrong_endpoint_dtype_are_rejected(self):
        with self.assertRaises(ValueError):
            self.summarize([])
        rows = self.rows(1)
        rows[0].observation.state = rows[0].observation.state.astype(np.float32)
        with self.assertRaises(ValueError):
            self.summarize(rows)


if __name__ == '__main__':
    unittest.main()
