"""Synthetic algebra/geometry tests; no real models, labels or simulator calls."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import diagnose_p4_stop_error_propagation as diagnostic


def fixture(k=5):
    actual = np.zeros((k+1, 10), np.float64); actual[:, 6] = -.06
    action = np.zeros((k, 2), np.float64)
    teacher_delta = np.zeros((k, 10), np.float64); teacher_delta[:, 7] = .001
    free_delta = teacher_delta.copy(); free_delta[:, 7] += np.arange(k)*.0001
    teacher = np.stack([np.stack((actual[t], diagnostic.integrated(actual[t], teacher_delta[t]))) for t in range(k)])
    free = np.empty_like(actual); free[0] = actual[0]
    for t in range(k): free[t+1] = diagnostic.integrated(free[t], free_delta[t])
    return actual, action, free, free_delta, teacher, teacher_delta


def diagnose(args, failure=-1, **kwargs):
    k = len(args[1])
    return diagnostic.diagnose_case(*args, failure_step=failure,
        free_reason='' if failure < 0 else 'predicted_deep_penetration',
        teacher_valid=np.ones(k, bool), teacher_failure_steps=np.full(k, -1),
        input_scale=np.ones(11), half_xy=np.array([.04, .03]), radius=.008,
        penetration_tolerance=.001, **kwargs)


class StopErrorPropagationTest(unittest.TestCase):
    def test_increment_decomposition_closes_in_all_ten_components(self):
        args = list(fixture())
        # Nontrivial velocity and angular components must not be discarded.
        args[3][:, 3:6] = np.arange(15).reshape(5, 3)*1e-4
        args[5][:, 3:6] = np.arange(15).reshape(5, 3)*2e-5
        for t in range(5):
            args[2][t+1] = diagnostic.integrated(args[2][t], args[3][t])
            args[4][t, 1] = diagnostic.integrated(args[0][t], args[5][t])
        result = diagnose(args)
        self.assertEqual(result['available_diagnostic_steps'], 5)
        self.assertLess(result['maximum_closure_absolute_error'], 1e-12)
        last = result['steps'][-1]
        self.assertAlmostEqual(last['next_state_error'][7], .006)
        self.assertAlmostEqual(last['previous_state_error'][7], .0046)
        self.assertAlmostEqual(last['local_increment_error'][7], .001)
        self.assertAlmostEqual(last['saved_increment_difference'][7], .0004)
        self.assertTrue(all(v is not None for v in last['closure_residual']))

    def test_yaw_wrap_boundary_closes_without_raw_rotation_misattribution(self):
        args = list(fixture(1))
        args[0][0, 2] = np.pi-.01; args[0][1, 2] = -np.pi+.01
        args[2][0] = args[0][0]
        args[3][0, 2] = .04; args[5][0, 2] = .03
        args[2][1] = diagnostic.integrated(args[2][0], args[3][0])
        args[4][0, 0] = args[0][0]
        args[4][0, 1] = diagnostic.integrated(args[0][0], args[5][0])
        result = diagnose(args); row = result['steps'][0]
        self.assertAlmostEqual(row['actual_increment'][2], .02)
        self.assertAlmostEqual(row['local_increment_error'][2], .01)
        self.assertAlmostEqual(row['saved_increment_difference'][2], .01)
        self.assertAlmostEqual(row['next_state_error'][2], .02)
        self.assertLess(result['maximum_closure_absolute_error'], 1e-12)

    def test_finite_failure_state_kept_as_invalid_diagnostic(self):
        result = diagnose(fixture(), failure=5)
        self.assertTrue(result['steps'][4]['available'])
        self.assertTrue(result['steps'][4]['free_finite_invalid_failure_step'])
        self.assertIs(result['steps'][4]['free_prediction_valid'], False)
        self.assertTrue(result['states'][5]['geometry']['available'])
        self.assertEqual(result['available_diagnostic_steps'], 5)

    def test_after_failure_is_missing_even_if_source_has_stale_finite_suffix(self):
        result = diagnose(fixture(), failure=3)
        self.assertEqual(result['available_diagnostic_steps'], 3)
        for row in result['steps'][3:]:
            self.assertIs(row['available'], False)
            self.assertEqual(row['free_next'], [None]*10)
            self.assertEqual(row['saved_free_increment'], [None]*10)
            self.assertEqual(row['closure_residual'], [None]*10)
        self.assertIs(result['states'][4]['geometry']['available'], False)

    def test_nonfinite_failure_prediction_not_interpolated(self):
        args = list(fixture()); args[2][3:] = np.nan; args[3][2:] = np.nan
        result = diagnose(args, failure=3)
        self.assertEqual(result['available_diagnostic_steps'], 2)
        self.assertEqual(result['steps'][2]['next_state_error'], [None]*10)
        self.assertIs(result['states'][3]['available'], False)

    def test_saved_teacher_must_have_actual_inputs(self):
        args = list(fixture()); args[4][1, 0, 0] += .001
        with self.assertRaisesRegex(ValueError, 'Teacher inputs'):
            diagnose(args)

    def test_saved_state_delta_mismatch_is_not_silently_explained(self):
        args = list(fixture()); args[2][2, 0] += .01
        with self.assertRaisesRegex(ValueError, 'does not close'):
            diagnose(args)

    def test_identical_first_input_saved_batch_difference_is_visible(self):
        args = list(fixture()); args[3][0, 7] += 1e-10
        for t in range(5): args[2][t+1] = diagnostic.integrated(args[2][t], args[3][t])
        result = diagnose(args)
        self.assertIs(result['first_input_exact'], True)
        self.assertEqual(result['steps'][0]['feature_difference'], [0.]*11)
        self.assertAlmostEqual(result['first_saved_increment_difference'][7], 1e-10, places=16)
        self.assertIn('batch', result['first_difference_interpretation'])

    def test_feature_difference_includes_geometry_velocity_and_same_commands(self):
        actual, action, free, *_ = fixture()
        free[1, 0] = .002; free[1, 2] = .1; free[1, 3] = .03
        a = diagnostic.encode(actual[1], action[1]); b = diagnostic.encode(free[1], action[1])
        self.assertAlmostEqual(b[0]-a[0], -.002)
        self.assertAlmostEqual(b[2]-a[2], np.sin(.2))
        self.assertAlmostEqual(b[4]-a[4], .03)
        self.assertEqual((b[-2:]-a[-2:]).tolist(), [0., 0.])

    def test_geometry_linear_single_pusher_block_has_no_spurious_interaction(self):
        actual = np.zeros(10); actual[6] = -.06
        predicted = actual.copy(); predicted[6] += .003
        g = diagnostic.geometry_decomposition(actual, predicted, [.04, .03], .008)
        self.assertAlmostEqual(g['total_gap_error_m'], -.003)
        self.assertAlmostEqual(g['shapley_gap_error_m']['pusher_xy'], -.003)
        self.assertAlmostEqual(g['shapley_gap_error_m']['object_xy'], 0.)
        self.assertAlmostEqual(g['shapley_gap_error_m']['object_yaw'], 0.)
        self.assertEqual(len(g['combinations']), 8)
        self.assertTrue(all(abs(v) < 1e-12 for v in g['second_order_interactions_m'].values()))
        self.assertLess(abs(g['third_order_interaction_m']), 1e-12)

    def test_nonlinear_geometry_interactions_and_six_orders_reconstruct(self):
        actual = np.zeros(10); actual[6:8] = [-.055, -.046]
        predicted = actual.copy(); predicted[:3] = [.003, -.002, .3]; predicted[6:8] = [-.045, -.037]
        g = diagnostic.geometry_decomposition(actual, predicted, [.04, .03], .008)
        self.assertTrue(any(abs(v) > 1e-8 for v in g['second_order_interactions_m'].values()))
        interactions = [r['mobius_interaction_m'] for r in g['combinations']]
        self.assertAlmostEqual(sum(interactions), g['total_gap_error_m'])
        self.assertAlmostEqual(sum(g['shapley_gap_error_m'].values()), g['total_gap_error_m'])
        for name, value in g['shapley_gap_error_m'].items():
            self.assertAlmostEqual(value, g['six_order_average_m'][name])

    def test_velocity_has_no_direct_geometric_contribution(self):
        actual = np.zeros(10); actual[6] = -.06
        predicted = actual.copy(); predicted[[3, 4, 5, 8, 9]] = [100, -100, 50, -200, 200]
        g = diagnostic.geometry_decomposition(actual, predicted, [.04, .03], .008)
        self.assertEqual(g['total_gap_error_m'], 0.)
        self.assertTrue(all(v == 0 for v in g['shapley_gap_error_m'].values()))
        self.assertIn('not identified', g['scope'])

    def test_teacher_geometry_is_conditioned_on_each_actual_endpoint(self):
        args = list(fixture())
        args[5][:, 6] = .001
        for t in range(5): args[4][t, 1] = diagnostic.integrated(args[0][t], args[5][t])
        result = diagnose(args, failure=3)
        for row in result['steps']:
            self.assertAlmostEqual(row['teacher_endpoint_geometry']['total_gap_error_m'], -.001)
            self.assertTrue(row['teacher_endpoint_geometry']['available'])
        self.assertIs(result['steps'][3]['available'], False)
        self.assertTrue(result['steps'][3]['teacher_prediction_valid'])

    def test_strict_json_preserves_boolean_and_missing(self):
        import json
        value = diagnostic.finite_list({'b':True,'n':float('nan'),'a':np.array([1.,np.inf]),'i':np.int64(5)})
        self.assertIs(value['b'], True)
        self.assertIsNone(value['n'])
        self.assertEqual(value['a'], [1.,None])
        json.dumps(value, allow_nan=False)


if __name__ == '__main__': unittest.main()
