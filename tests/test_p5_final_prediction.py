"""Pure recorded-trajectory fixtures; no environment reset or physical step."""
from copy import deepcopy
from dataclasses import replace
import json
import math
import gc
from pathlib import Path
import tempfile
import unittest
import weakref

import numpy as np

from push_align.config import load_pilot_config
from push_align.environment import Observation, STATE_FIELDS, StepResult, wrap_angle
from push_align.final_prediction import evaluate_final_predictions, MODELS, TRUTH_MASK_PROVENANCE


def predictor(state, action):
    delta = np.zeros_like(state)
    delta[:, 6:8] = action * .1
    delta[:, 8:10] = action - state[:, 8:10]
    return delta


def fixtures(length=12, *, yaw_wrap=False):
    config = deepcopy(load_pilot_config())
    config['planner']['horizon'] = 5
    config['planner']['command_speed_norm_max_m_s'] = .03
    rows = []
    for family in range(12):
        state = np.zeros(10, np.float64)
        state[0] = (family + 1) * .00001
        state[6] = -.1
        if yaw_wrap: state[2] = math.pi - .01
        initial = state.copy(); results = []; time = .126
        for tick in range(length):
            before = state.copy(); before_time = time
            requested = np.array([.02, 0.], np.float64)
            applied = np.array([.001, 0.], np.float64)
            state = state + predictor(state[None], applied[None])[0]
            if yaw_wrap: state[2] = float(wrap_angle(state[2] + .02))
            time += .1
            middle = (state + before) / 2
            pattern = ((False, False, False), (True, True, True), (False, True, True))[tick % 3]
            samples = []
            for sample_state, sample_time, active in zip(
                    (before, middle, state), (before_time, before_time + .05, time), pattern):
                contacts = [{'pair': 'pusher_object', 'normal_force_N': .002}] if active else []
                samples.append(dict(state=sample_state.tolist(), sim_time=sample_time, contacts=contacts))
            results.append(StepResult(requested, applied,
                Observation(state.copy(), time, config['schema_version'], True), .1, True,
                {'samples': samples, 'state_fields': STATE_FIELDS, 'validity_violations': []},
                np.asarray([middle, state], np.float64), np.asarray([before_time + .05, time], np.float64)))
        rows.append(dict(family_id=f'fixture-{family:02d}',
            stratum=('translation', 'rotation', 'offset')[family // 4], controller_id='G',
            initial_state10=initial, initial_sim_time=.126, results=results,
            interval_valid=np.ones(length, np.bool_), truth_mask_provenance=TRUTH_MASK_PROVENANCE))
    return rows, {name: predictor for name in MODELS}, config


class FinalPredictionTests(unittest.TestCase):
    def evaluate(self, rows=None, models=None, config=None):
        if rows is None: rows, models, config = fixtures()
        return evaluate_final_predictions(rows, models, config, config)

    def test_known_perfect_model_and_common_horizon_denominators(self):
        report, arrays = self.evaluate()
        expected = {1: (144, 0), 5: (96, 48), 10: (36, 108)}
        self.assertEqual(report['additional_physics_intervals'], 0)
        self.assertTrue(all(value == 0 for value in report['future_engine_calls'].values()))
        for horizon, (eligible, boundary) in expected.items():
            group = report['horizons'][str(horizon)]
            self.assertEqual(group['recorded_start_opportunities'], 144)
            self.assertEqual(group['truth_eligible_windows'], eligible)
            self.assertEqual(group['boundary_excluded_windows'], boundary)
            for model in MODELS:
                summary = group['models'][model]
                self.assertEqual(summary['predicted_valid_windows'], eligible)
                self.assertEqual(summary['predicted_invalid_windows'], 0)
                self.assertLess(summary['metrics_conditional_on_valid_prediction_and_truth']['pusher_xy_error_m']['max'], 1e-15)
                self.assertEqual(summary['family_first_means_conditional_on_available_families']['object_xy_error_m']['family_denominator'], 12)
                self.assertEqual(arrays[f'K{horizon}__{model}__valid'].dtype, np.bool_)
                self.assertEqual(arrays[f'K{horizon}__{model}__failure_steps'].dtype, np.int64)
            self.assertEqual(arrays[f'K{horizon}__window_id'].shape, (144,))
        self.assertTrue(all(value.dtype.kind != 'O' for value in arrays.values()))
        json.dumps(report, allow_nan=False)

    def test_actual_applied_actions_and_diagnostic_horizon_do_not_change_online_config(self):
        rows, models, config = fixtures(); original = deepcopy(config)
        report, arrays = self.evaluate(rows, models, config)
        self.assertEqual(config, original)
        self.assertFalse(report['online_settings_mutated'])
        np.testing.assert_array_equal(arrays['K1__applied_actions'][:, 0, 0], np.full(144, .001))
        self.assertNotEqual(rows[0]['results'][0].requested_action[0], arrays['K1__applied_actions'][0, 0, 0])

    def test_recursive_state_is_not_replaced_by_future_truth(self):
        rows, models, config = fixtures()
        def growing(state, action):
            delta = predictor(state, action); delta[:, 0] = state[:, 0]
            return delta
        models['R'] = growing
        _, arrays = self.evaluate(rows, models, config)
        self.assertAlmostEqual(arrays['K10__R__predicted_states'][0, -1, 0], .00001 * 2**10)
        self.assertAlmostEqual(arrays['K10__truth_states'][0, -1, 0], .00001)

    def test_invalid_truth_interval_is_retained_and_never_sent_to_model(self):
        rows, models, config = fixtures(); step = rows[0]['results'][-1]
        diagnostic = deepcopy(step.diagnostics); diagnostic['validity_violations'] = ['fixture_physics_invalid']
        rows[0]['results'][-1] = replace(step, transition_valid=False,
            observation=replace(step.observation, observation_valid=False), diagnostics=diagnostic)
        rows[0]['interval_valid'][-1] = False
        report, arrays = self.evaluate(rows, models, config)
        for horizon, eligible in ((1, 143), (5, 95), (10, 35)):
            group = report['horizons'][str(horizon)]
            self.assertEqual(group['raw_invalid_full_windows'], 1)
            self.assertEqual(group['truth_eligible_windows'], eligible)
            self.assertEqual(group['models']['R']['predicted_invalid_windows'], 0)
            not_attempted = ~arrays[f'K{horizon}__truth_valid']
            np.testing.assert_array_equal(arrays[f'K{horizon}__R__failure_steps'][not_attempted], -2)

    def test_model_invalid_does_not_disappear_or_change_other_models_population(self):
        rows, models, config = fixtures()
        models['R'] = lambda state, action: np.full(state.shape, np.nan, np.float64)
        report, arrays = self.evaluate(rows, models, config)
        for horizon, count in ((1, 144), (5, 96), (10, 36)):
            group = report['horizons'][str(horizon)]
            self.assertEqual(group['models']['R']['predicted_invalid_windows'], count)
            self.assertEqual(group['models']['S1-11']['predicted_valid_windows'], count)
            self.assertEqual(group['models']['R']['family_first_means_conditional_on_available_families']['object_xy_error_m']['family_denominator'], 0)
            eligible = arrays[f'K{horizon}__truth_valid']
            self.assertTrue(all(value == 'nonfinite_physical_delta' for value in arrays[f'K{horizon}__R__reasons'][eligible]))
            np.testing.assert_array_equal(arrays[f'K{horizon}__R__failure_steps'][eligible], 1)

    def test_regimes_are_from_first_actual_interval_including_threshold_pattern(self):
        report, _ = self.evaluate()
        groups = report['horizons']['1']['models']['R']['by_first_interval_regime']
        for name in ('noncontact', 'sustained_contact', 'transition'):
            self.assertEqual(groups[name]['recorded_start_opportunities'], 48)
        self.assertEqual(groups['unknown']['recorded_start_opportunities'], 0)

    def test_yaw_residual_uses_2pi_wrap_at_discontinuity(self):
        rows, models, config = fixtures(1, yaw_wrap=True)
        def yaw_predictor(state, action):
            delta = predictor(state, action); delta[:, 2] = .02
            return delta
        models = {name: yaw_predictor for name in MODELS}
        report, arrays = self.evaluate(rows, models, config)
        self.assertLess(report['horizons']['1']['models']['R']['metrics_conditional_on_valid_prediction_and_truth']['object_yaw_error_rad_2pi']['max'], 1e-14)
        np.testing.assert_allclose(arrays['K1__R__endpoint_state_error'][:, 2], 0., atol=1e-14)

    def test_zero_transition_family_population_and_missing_suffixes_are_explicit(self):
        rows, models, config = fixtures(0)
        report, arrays = self.evaluate(rows, models, config)
        self.assertEqual(report['family_count'], 12)
        for horizon in (1, 5, 10):
            self.assertEqual(report['horizons'][str(horizon)]['recorded_start_opportunities'], 0)
            self.assertEqual(report['horizons'][str(horizon)]['models']['R']['predicted_valid_windows'], 0)
            self.assertEqual(arrays[f'K{horizon}__truth_states'].shape, (0, horizon + 1, 10))
        self.assertTrue(all(value['calls'] == 0 for value in report['inference'].values()))

    def test_missing_duplicate_wrong_source_and_malformed_raw_are_rejected(self):
        rows, models, config = fixtures()
        for changed in (rows[:-1], rows[:-1] + [rows[0]]):
            with self.assertRaises(ValueError): self.evaluate(changed, models, config)
        changed = deepcopy(rows); changed[0]['controller_id'] = 'S1-11'
        with self.assertRaises(ValueError): self.evaluate(changed, models, config)
        changed = deepcopy(rows); step = changed[0]['results'][0]
        bad = step.observation.state.copy(); bad[0] += .01
        changed[0]['results'][0] = replace(step, observation=replace(step.observation, state=bad))
        with self.assertRaises(ValueError): self.evaluate(changed, models, config)

    def test_future_engine_access_in_predictor_is_blocked(self):
        import mujoco
        rows, models, config = fixtures(1)
        def forbidden(state, action):
            mujoco.mj_forward(None, None)
            return np.zeros_like(state)
        models['R'] = forbidden
        with self.assertRaisesRegex(AssertionError, 'Future engine call'):
            self.evaluate(rows, models, config)

    def test_geometry_mask_excludes_real_interval_even_when_raw_flags_are_true(self):
        rows, models, config = fixtures()
        rows[0]['interval_valid'][-1] = False
        report, _ = self.evaluate(rows, models, config)
        self.assertTrue(rows[0]['results'][-1].transition_valid)
        self.assertEqual(report['horizons']['5']['raw_invalid_full_windows'], 1)
        self.assertEqual(report['source_interval_populations']['fixture-00']['invalid_intervals'], 1)

    def test_missing_or_wrong_mask_provenance_and_promoted_invalid_flags_rejected(self):
        rows, models, config = fixtures(1)
        for key in ('interval_valid', 'truth_mask_provenance'):
            changed = deepcopy(rows); del changed[0][key]
            with self.assertRaises(ValueError): self.evaluate(changed, models, config)
        changed = deepcopy(rows); changed[0]['interval_valid'] = np.ones(1, np.int64)
        with self.assertRaises(ValueError): self.evaluate(changed, models, config)
        changed = deepcopy(rows); step = changed[0]['results'][0]
        changed[0]['results'][0] = replace(step, transition_valid=False)
        with self.assertRaisesRegex(ValueError, 'cannot promote'): self.evaluate(changed, models, config)

    def test_pi_task_symmetry_does_not_erase_physical_prediction_yaw_error(self):
        rows, models, config = fixtures()
        def turning(state, action):
            delta = predictor(state, action); delta[:, 2] = math.pi / 10
            return delta
        models['R'] = turning
        report, _ = self.evaluate(rows, models, config)
        self.assertAlmostEqual(report['horizons']['10']['models']['R']['metrics_conditional_on_valid_prediction_and_truth']['object_yaw_error_rad_2pi']['mean'], math.pi)

    def test_typed_npz_arrays_roundtrip_without_pickle_or_value_loss(self):
        report, arrays = self.evaluate()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.npz'
            np.savez_compressed(path, **arrays)
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(set(saved.files), set(arrays))
                for name, expected in arrays.items():
                    self.assertEqual(saved[name].dtype, expected.dtype)
                    self.assertEqual(saved[name].shape, expected.shape)
                    self.assertEqual(saved[name].tobytes(), expected.tobytes())

    def test_streamed_raw_steps_are_released_before_requesting_next_family(self):
        rows, models, config = fixtures(1)
        original_ids = [row['family_id'] for row in rows]
        released = []
        def stream():
            while rows:
                row = rows.pop(0)
                reference = weakref.ref(row['results'][0])
                family = row['family_id']
                yield row
                del row
                gc.collect()
                self.assertIsNone(reference(), 'Previous fullraw StepResult retained while requesting next family')
                released.append(family)
        report, _ = self.evaluate(stream(), models, config)
        self.assertEqual(released, original_ids)
        self.assertEqual(report['family_ids'], sorted(original_ids))

    def test_invalid_middle_interval_never_bridges_a_truth_window(self):
        rows, models, config = fixtures()
        rows[0]['interval_valid'][5] = False
        report, _ = self.evaluate(rows, models, config)
        self.assertEqual(report['horizons']['1']['raw_invalid_full_windows'], 1)
        self.assertEqual(report['horizons']['5']['raw_invalid_full_windows'], 5)
        self.assertEqual(report['horizons']['10']['raw_invalid_full_windows'], 3)

    def test_family_first_mean_does_not_weight_longer_G_trajectories_as_more_families(self):
        rows, models, config = fixtures()
        rows[0]['results'] = rows[0]['results'][:2]
        rows[0]['interval_valid'] = rows[0]['interval_valid'][:2]
        def biased(state, action):
            delta = predictor(state, action)
            delta[:, 0] = np.where(state[:, 0] < .000015, .001, 0.)
            return delta
        models['R'] = biased
        report, _ = self.evaluate(rows, models, config)
        model = report['horizons']['1']['models']['R']
        pooled = model['metrics_conditional_on_valid_prediction_and_truth']['object_xy_error_m']['mean']
        equal_family = model['family_first_means_conditional_on_available_families']['object_xy_error_m']['mean']
        self.assertAlmostEqual(pooled, .002 / 134)
        self.assertAlmostEqual(equal_family, .001 / 12)
        self.assertNotEqual(pooled, equal_family)

    def test_applied_norm_exceeding_bound_cap_is_never_clipped(self):
        rows, models, config = fixtures(1); step = rows[0]['results'][0]
        rows[0]['results'][0] = replace(step, applied_action=np.array([.04, 0.], np.float64))
        with self.assertRaisesRegex(ValueError, 'bound G/planner cap'):
            self.evaluate(rows, models, config)


if __name__ == '__main__': unittest.main()
