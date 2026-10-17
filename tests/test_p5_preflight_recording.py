"""Regression for preserving invalid and mixed candidate arrays, no physics."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from push_align.config import load_pilot_config
from push_align.planning import PlannerContext, TaskSpec, plan
from p5_execution_preflight import CountedPredictor, save_plan_traces


class PreflightRecordingTest(unittest.TestCase):
    def record(self, fault):
        config = load_pilot_config()
        config['planner'].update(candidate_count=128, horizon=5, block_lengths=[2, 3])
        state = np.zeros(10, np.float64)
        state[6] = -.051
        predicted = CountedPredictor(lambda s,a: np.zeros_like(s), fault)
        result = plan(state, TaskSpec.from_config(config), PlannerContext.initial(),
                      np.random.default_rng(4711), predicted, config)
        decision = {'index': 0, 'selected_index': result.selected_index, 'executed': fault != 'all_nonfinite_prediction'}
        with TemporaryDirectory() as folder:
            save_plan_traces(Path(folder), SimpleNamespace(plan_results={0: result}, decisions=[decision]))
            path = Path(folder) / 'plans/step-000'
            with np.load(path / 'trace.npz', allow_pickle=False) as z:
                self.assertEqual(z['candidates'].shape, (128, 5, 2))
                np.testing.assert_array_equal(z['valid'], result.predictions.valid)
                np.testing.assert_array_equal(z['total_cost'], result.costs.total)
                np.testing.assert_array_equal(z['predicted_states'], result.predictions.states)
                self.assertTrue(np.isposinf(z['total_cost'][~z['valid']]).all())
            self.assertEqual(json.loads((path / 'identity.json').read_text()), decision)
        return result

    def test_all_invalid_unexecuted_trace_preserves_all128_slots(self):
        result = self.record('all_nonfinite_prediction')
        self.assertIsNone(result.selected_index)
        self.assertFalse(result.predictions.valid.any())

    def test_mixed_invalid_trace_keeps_invalid_slot_and_valid_choice(self):
        result = self.record('mixed_nonfinite_prediction')
        self.assertFalse(result.predictions.valid[0])
        self.assertIsNotNone(result.selected_index)
        self.assertTrue(result.predictions.valid[result.selected_index])


if __name__ == '__main__':
    unittest.main()
