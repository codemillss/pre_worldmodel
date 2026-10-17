"""Paired estimand/invalid-denominator tests with no learned model or physics."""
from copy import deepcopy
import json
import unittest

import numpy as np

from push_align.config import load_pilot_config
from push_align.data_intervention_evaluation import (
    ARMS, DevelopmentData, ENDPOINTS, compare_arms, evaluate_development,
    paired_comparison, regression_geometry, validate_development,
)


def fixture():
    identities, states, actions = [], [], []
    for f in range(12):
        for y in range(2):
            for s in range(2):
                ids = {'episode_id': f'f{f}-Y{y}-S{s}', 'family_id': f'f{f:02}',
                       'family_rank': f, 'geometry': 'side_pass', 'y_condition': f'Y{y}', 'stop_condition': f'S{s}'}
                a = np.tile([0., .01+.01*y], (20, 1))
                a[[5, 7, 9, 11, 13] if not s else [10, 11, 12, 13, 14]] = 0
                x = np.zeros((21, 10), np.float64); x[:, 6] = -.08
                x[0, 9] = .01+.01*y
                for t in range(20):
                    x[t+1] = x[t]
                    x[t+1, 6:8] += .1*a[t]
                    x[t+1, 8:10] = a[t]
                identities.append(ids); states.append(x); actions.append(a)
    return DevelopmentData(tuple(identities), np.stack(states), np.stack(actions),
                           np.full((48, 20), 'noncontact'), {})


def perfect(state, action):
    delta = np.zeros_like(state)
    delta[:, 6:8] = .1*action
    delta[:, 8:10] = action-state[:, 8:10]
    return delta


def result_fixture(values=None):
    values = values or {'R': 10., 'D00': 10., 'D10': 8., 'D01': 9., 'D11': 7.}
    families = [f'f{i:02}' for i in range(12)]
    return {arm: {'families': families.copy(), 'primary': {phase: {f: {endpoint: value for endpoint in ENDPOINTS}
              for f in families} for phase in ('Y_primary', 'S_primary')}} for arm, value in values.items()}


class DataInterventionEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.config = load_pilot_config()
        self.data = fixture()

    def test_teacher_heads_deduplicate_s_and_stop_uses_actual_state10(self):
        calls = []
        def capture(state, action):
            calls.append((state.copy(), action.copy()))
            return perfect(state, action)
        summary, arrays = evaluate_development(capture, self.data, self.config)
        self.assertEqual(summary['Y_heads_teacher_conditioned']['attempted'], 120)
        self.assertEqual(summary['Y_heads_teacher_conditioned']['valid'], 120)
        self.assertEqual(summary['Y_heads_teacher_conditioned']['Y_conditions']['Y1']['attempted'], 60)
        self.assertTrue(np.array_equal(calls[0][0], self.data.states[arrays['head_indices'], :5].reshape(-1, 10)))
        self.assertTrue(np.array_equal(calls[1][0], self.data.states[arrays['stop_indices'], 10]))
        self.assertEqual(summary['S1_free_stop']['attempted'], 24)
        self.assertEqual(summary['core_H10_free']['attempted'], 48)
        for phase in summary['primary'].values():
            for row in phase.values():
                for error in row.values(): self.assertLess(error, 1e-15)

    def test_free_error_accumulates_but_teacher_error_does_not(self):
        def biased(state, action):
            delta = perfect(state, action); delta[:, 7] += .0001
            return delta
        summary, _ = evaluate_development(biased, self.data, self.config)
        self.assertAlmostEqual(summary['primary']['Y_primary']['f00']['pusher_xy_m'], .0001)
        self.assertAlmostEqual(summary['primary']['S_primary']['f00']['pusher_xy_m'], .0005)
        self.assertAlmostEqual(summary['core_H10_free']['metrics_valid_only']['pusher_xy_m']['mean'], .001)

    def test_initial_geometry_invalid_preserves_whole_family_and_denominator(self):
        states = self.data.states.copy(); states[:4, :, 6] = 1.
        bad = DevelopmentData(self.data.identities, states, self.data.actions, self.data.regimes, {})
        summary, arrays = evaluate_development(perfect, bad, self.config)
        self.assertIsNone(summary['primary']['Y_primary']['f00']['pusher_xy_m'])
        self.assertIsNone(summary['primary']['S_primary']['f00']['object_xy_m'])
        self.assertEqual(summary['Y_heads_teacher_conditioned']['attempted'], 120)
        self.assertEqual(summary['Y_heads_teacher_conditioned']['invalid'], 10)
        self.assertEqual(summary['S1_free_stop']['free_K1_to_K5']['1']['invalid'], 2)
        self.assertTrue((arrays['stop_failure_steps'][:2] == 0).all())
        json.dumps(summary, allow_nan=False)

    def test_later_free_failure_does_not_hide_earlier_valid_step(self):
        count = 0
        def late(state, action):
            nonlocal count
            count += 1
            delta = perfect(state, action)
            # Call1 teacher head; calls2/3/4 free stop k1/k2/k3.
            if count == 4: delta[0] = np.nan
            return delta
        summary, arrays = evaluate_development(late, self.data, self.config)
        self.assertEqual(arrays['stop_failure_steps'][0], 3)
        self.assertEqual(summary['S1_free_stop']['free_K1_to_K5']['2']['valid'], 24)
        self.assertEqual(summary['S1_free_stop']['free_K1_to_K5']['3']['valid'], 23)
        self.assertEqual(summary['S1_free_stop']['invalid'], 1)
        self.assertIsNone(summary['primary']['S_primary']['f00']['object_xy_m'])

    def test_s1_head_cannot_be_silently_counted_twice_if_different(self):
        states = self.data.states.copy(); states[1, 1, 0] += .001
        with self.assertRaisesRegex(ValueError, 'S heads disagree'):
            validate_development(DevelopmentData(self.data.identities, states, self.data.actions, self.data.regimes, {}))

    def test_stop_index_must_be_exact_registered_five_zero_actions(self):
        actions = self.data.actions.copy(); actions[1, 10, 1] = .01
        with self.assertRaisesRegex(ValueError, 'five zero'):
            validate_development(DevelopmentData(self.data.identities, self.data.states, actions, self.data.regimes, {}))

    def test_direction_requires_both_contrasts_and_both_xy_endpoints(self):
        results = result_fixture()
        comparison, draws = compare_arms(results)
        self.assertEqual(draws.shape, (10000, 12))
        self.assertEqual(comparison['directional_support']['Y']['status'], 'DIRECTIONAL_SUPPORT')
        self.assertEqual(comparison['directional_support']['S']['status'], 'DIRECTIONAL_SUPPORT')
        for family in results['R']['families']:
            results['D11']['primary']['Y_primary'][family]['pusher_xy_m'] = 10.
        comparison, _ = compare_arms(results)
        self.assertEqual(comparison['directional_support']['Y']['status'], 'MIXED')

    def test_exact_ties_are_inconclusive_not_negative_by_epsilon(self):
        comparison, _ = compare_arms(result_fixture(dict.fromkeys(ARMS, 1.)))
        row = comparison['contrasts']['Y_primary']['object_xy_m']['D10-D00']
        self.assertEqual(row['exact_float64_ties'], 12)
        self.assertEqual(row['bootstrap_percentile_95'], [0., 0.])
        self.assertEqual(comparison['directional_support']['Y']['status'], 'INCONCLUSIVE')

    def test_one_invalid_family_holds_inference_without_complete_case_bootstrap(self):
        results = result_fixture()
        results['D10']['primary']['Y_primary']['f00']['object_xy_m'] = None
        comparison, _ = compare_arms(results)
        self.assertEqual(comparison['directional_support']['Y']['status'], 'HOLD_INVALID_PRIMARY')
        row = comparison['contrasts']['Y_primary']['object_xy_m']['D10-D00']
        self.assertEqual((row['attempted_families'], row['valid_families']), (12, 11))
        self.assertIsNone(row['mean_difference'])
        self.assertEqual(len(row['family_differences']), 12)
        self.assertEqual(comparison['directional_support']['S']['status'], 'DIRECTIONAL_SUPPORT')
        json.dumps(comparison, allow_nan=False)

    def test_bootstrap_matches_manual_paired_family_resampling(self):
        values = np.arange(12, dtype=float).tolist()
        draws = np.random.default_rng(2026101304).integers(0, 12, size=(10000, 12), dtype=np.int64)
        result = paired_comparison(values, [str(i) for i in range(12)], draws)
        expected = np.percentile(np.asarray(values)[draws].mean(axis=1), [2.5, 97.5]).tolist()
        self.assertEqual(result['bootstrap_percentile_95'], expected)
        self.assertEqual(result['mean_difference'], 5.5)

    def test_cross_arm_family_reordering_rejected(self):
        results = result_fixture(); results['D11']['families'].reverse()
        with self.assertRaisesRegex(ValueError, 'pairing'):
            compare_arms(results)

    def test_interaction_and_r_comparisons_preserved_separately(self):
        comparison, _ = compare_arms(result_fixture())
        for phase in ('Y_primary', 'S_primary'):
            for metric in ENDPOINTS:
                rows = comparison['contrasts'][phase][metric]
                self.assertEqual(rows['interaction']['family_differences'], [0.]*12)
                self.assertEqual(rows['D11-R']['family_differences'], [-3.]*12)

    def test_old_validation_geometry_supplement_keeps_regime_denominators(self):
        traces = {}
        for k in (1, 5, 10):
            x = np.tile(self.data.states[0, 0], (3, k+1, 1)); x[1, k, 6] = 1.
            traces[f'h{k}_predicted_states'] = x
            traces[f'h{k}_regime_code'] = np.arange(3)
        summary = regression_geometry(traces, self.config)
        for k in ('1', '5', '10'):
            self.assertEqual(summary[k]['attempted_windows'], 3)
            self.assertEqual(summary[k]['geometry_invalid_windows'], 1)
            self.assertEqual(summary[k]['regimes']['sustained_contact']['geometry_invalid_windows'], 1)


if __name__ == '__main__':
    unittest.main()
