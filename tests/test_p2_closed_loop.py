"""Execution-boundary checks separate from task math and task performance."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

from push_align.closed_loop import ENGINE_GUARDS, consume_actual_step, model_planning_scope
from push_align.config import load_pilot_config
from push_align.planning import PlannerContext, TaskSpec
from push_align.task import start_task
from push_align.representation import STATE_FIELDS


class ExecutionBoundaryTest(unittest.TestCase):
    def fixture(self):
        config = load_pilot_config()
        task = TaskSpec.from_config(config)
        state = np.zeros(10, np.float64)
        obs = SimpleNamespace(state=state, sim_time=2., observation_valid=True,
                              schema_version=config['schema_version'])
        status = start_task(obs, task, .1)
        end = SimpleNamespace(state=state.copy(), sim_time=2.1, observation_valid=True,
                              schema_version=config['schema_version'])
        result = SimpleNamespace(applied_action=np.array([.03, 0.], np.float64),
                                 requested_action=np.array([.06, 0.], np.float64),
                                 observation=end, actual_dt=.1, transition_valid=True,
                                 diagnostics={'validity_violations': [], 'state_fields': STATE_FIELDS,
                                              'samples': [{'state': state.tolist(), 'sim_time': t}
                                                          for t in (2., 2.05, 2.1)]},
                                 substep_states=np.stack([state, state]),
                                 substep_times=np.array([2.05, 2.1], np.float64))
        context = PlannerContext(np.array([.06, 0.]), np.full((10, 2), .001), 1)
        return task, status, result, context

    def test_guard_blocks_and_counts_engine_access_then_restores_function(self):
        original = mujoco.mj_step
        counts = dict.fromkeys(ENGINE_GUARDS, 0)
        with model_planning_scope(counts):
            with self.assertRaises(AssertionError):
                mujoco.mj_step(None, None)
        self.assertIs(mujoco.mj_step, original)
        self.assertEqual(counts['mj_step'], 1)
        self.assertEqual(sum(counts.values()), 1)

    def test_missing_guard_is_rejected(self):
        with self.assertRaises(ValueError):
            with model_planning_scope({'mj_step': 0}):
                pass

    def test_actual_command_replaces_proposal_without_double_shifting_warm(self):
        task, status, result, context = self.fixture()
        _, after = consume_actual_step(status, result, task, .1, context,
                                      holding=False, geometry={'pass': True, 'reasons': []})
        np.testing.assert_array_equal(after.previous_applied_action, [.03, 0.])
        np.testing.assert_array_equal(after.optimal_actions, context.optimal_actions)
        self.assertEqual(after.control_step, 1)

    def test_holding_zero_updates_applied_context_and_clears_stale_warm(self):
        task, status, result, context = self.fixture()
        result.applied_action = np.zeros(2, np.float64)
        _, after = consume_actual_step(status, result, task, .1, context,
                                      holding=True, geometry={'pass': True, 'reasons': []})
        np.testing.assert_array_equal(after.previous_applied_action, [0., 0.])
        self.assertIsNone(after.optimal_actions)
        self.assertEqual(after.control_step, 2)

    def test_geometry_failure_is_not_hidden_by_valid_physical_step(self):
        task, status, result, context = self.fixture()
        after, _ = consume_actual_step(status, result, task, .1, context, holding=False,
                                      geometry={'pass': False, 'reasons': ['contact_penetration_exceeded']})
        self.assertEqual(after.phase, 'failed')
        self.assertEqual(after.terminal_reason, 'geometry_invalid')
        self.assertIn('contact_penetration_exceeded', after.violations)

    def test_holding_cannot_apply_nonzero_command(self):
        task, status, result, context = self.fixture()
        with self.assertRaises(ValueError):
            consume_actual_step(status, result, task, .1, context, holding=True,
                                geometry={'pass': True, 'reasons': []})


if __name__ == '__main__':
    unittest.main()
