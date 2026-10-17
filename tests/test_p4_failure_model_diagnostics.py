"""Algebraic P4-04 diagnostics; no dynamics training or physics advancement."""
from copy import deepcopy
import json
import math
import unittest

import numpy as np
import torch

from push_align import PlanarPushEnv
from push_align.data import contract_hash, make_contract
from push_align.dynamics import DeltaMLP
from push_align.normalization import fit_normalizer, learning_schema
from push_align.planning import LearnedPredictor
from p4_failure_model_diagnostics import (ModelDiagnosticError,
                                          diagnose_model_sequence,
                                          make_feature_support)


def fixture_normalizer(features):
    contract = make_contract(PlanarPushEnv())
    schema = learning_schema(contract)
    identity = {"dataset_id": "p4-read-only-analytic-diagnostic", "manifest_sha256": "b"*64,
                "data_contract_hash": contract_hash(contract)}
    count = len(features)
    # A synthetic statistical fixture, not a physical measured trajectory.
    deltas = np.repeat(np.linspace(-1., 1., count)[:, None], 10, axis=1)
    normalizer = fit_normalizer(features, deltas, np.zeros(count, dtype=np.int8), schema,
                                identity, np.zeros(count, dtype=np.int64),
                                np.arange(count, dtype=np.int64))
    return normalizer, contract["resolved_config"]


def sequence(batch=1, horizon=3):
    states = np.zeros((batch, horizon+1, 10), dtype=np.float64)
    states[..., 6] = 1.
    actions = np.zeros((batch, horizon, 2), dtype=np.float64)
    return states, actions


class RelativePredictor:
    def __init__(self):
        self.inputs = []

    def __call__(self, states, actions):
        self.inputs.append((states.copy(), actions.copy()))
        delta = np.zeros_like(states)
        delta[:, 0] = .25*(states[:, 6]-states[:, 0])
        delta[:, 6:8] = .1*actions
        return delta


def constant_delta(columns):
    def predictor(states, actions):
        result = np.zeros_like(states)
        for key, value in columns.items():
            result[:, key] = value
        return result
    return predictor


class P4FailureModelDiagnosticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.train = np.repeat(np.array([[-1.], [1.]], dtype=np.float64), 11, axis=1)
        cls.normalizer, cls.config = fixture_normalizer(cls.train)
        cls.support = make_feature_support(cls.normalizer, cls.train)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def diagnose(self, predictor, states=None, actions=None, **kwargs):
        if states is None:
            states, actions = sequence()
        return diagnose_model_sequence(predictor, self.support, states, actions, **kwargs)

    def test_independent_one_step_and_free_recurrence_have_distinct_exact_answers(self):
        result = self.diagnose(RelativePredictor())
        np.testing.assert_array_equal(result.traces["free_states"][0, :, 0], [0., .25, .4375, .578125])
        np.testing.assert_array_equal(result.traces["teacher_next_states"][0, :, 0], [.25]*3)
        np.testing.assert_array_equal(result.traces["teacher_predicted_deltas"][0, :, 0], [.25]*3)
        np.testing.assert_array_equal(result.traces["free_predicted_deltas"][0, :, 0], [.25, .1875, .140625])
        np.testing.assert_array_equal(result.traces["teacher_object_xy_error_m"][0], [.25]*3)
        np.testing.assert_array_equal(result.traces["free_object_xy_error_m"][0], [.25, .4375, .578125])
        self.assertEqual(result.summary["actual_valid_transitions"], 3)
        self.assertIn("No planner", result.summary["validity_scope"])

    def test_later_physical_truth_never_enters_free_predictor(self):
        states, actions = sequence()
        changed = states.copy()
        changed[0, 1:, 0] = [.8, -.4, .9]
        first = self.diagnose(RelativePredictor(), states, actions)
        second = self.diagnose(RelativePredictor(), changed, actions)
        np.testing.assert_array_equal(first.traces["free_states"], second.traces["free_states"])
        np.testing.assert_array_equal(first.traces["free_input_features"], second.traces["free_input_features"])
        self.assertFalse(np.array_equal(first.traces["teacher_next_states"], second.traces["teacher_next_states"]))
        self.assertFalse(np.array_equal(first.traces["free_state_errors"], second.traces["free_state_errors"]))

    def test_applied_command_distinct_from_measured_pusher_velocity_is_used(self):
        states, actions = sequence(horizon=2)
        actions[0] = [[.03, -.02], [.01, .04]]
        states[..., 8:10] = [-.11, .17]
        result = self.diagnose(RelativePredictor(), states, actions)
        np.testing.assert_array_equal(result.traces["actual_input_features"][..., -2:], actions)
        np.testing.assert_array_equal(result.traces["actual_input_features"][..., 7:9], states[:, :-1, 8:10])
        np.testing.assert_array_equal(result.traces["free_predicted_deltas"][..., 6:8], .1*actions)

    def test_learning_2pi_yaw_primary_is_not_replaced_by_task_pi_symmetry(self):
        states, actions = sequence(horizon=1)
        states[0, 1, 2] = math.pi
        result = self.diagnose(constant_delta({2: 0.}), states, actions,
                               actual_valid_mask=np.zeros((1, 1), dtype=np.bool_))
        self.assertEqual(result.summary["actual_valid_transitions"], 0)
        # Avoid an exactly-pi actual displacement; both endpoints themselves
        # carry the same yaw, while the prediction changes by almost pi.
        states[:, :, 2] = math.pi-.2
        result = self.diagnose(constant_delta({2: math.pi-.01}), states, actions)
        self.assertAlmostEqual(result.traces["free_object_yaw_learning_2pi_error_rad"][0, 0], math.pi-.01)
        self.assertAlmostEqual(result.traces["free_object_yaw_task_pi_error_rad"][0, 0], .01)
        self.assertAlmostEqual(result.traces["free_delta_errors"][0, 0, 2], math.pi-.01)

    def test_crossing_pi_boundary_integrates_and_compares_with_2pi_wrap(self):
        states, actions = sequence(horizon=1)
        states[0, 0, 2] = math.pi-.02
        states[0, 1, 2] = -math.pi+.03
        result = self.diagnose(constant_delta({2: .04}), states, actions)
        self.assertAlmostEqual(result.traces["actual_deltas"][0, 0, 2], .05)
        self.assertAlmostEqual(result.traces["free_states"][0, 1, 2], -math.pi+.02)
        self.assertAlmostEqual(result.traces["free_object_yaw_learning_2pi_error_rad"][0, 0], .01)

    def test_invalid_actual_suffix_is_never_used_and_denominators_are_per_step(self):
        states, actions = sequence(batch=3)
        mask = np.array([[1, 1, 1], [1, 0, 0], [0, 0, 0]], dtype=np.bool_)
        states[1, 2:] = np.nan
        states[2, 1:] = np.nan
        actions[~mask] = np.nan
        model = RelativePredictor()
        result = self.diagnose(model, states, actions, actual_valid_mask=mask)
        self.assertEqual(result.summary["actual_valid_transitions"], 4)
        self.assertEqual([x["actual_valid_transitions"] for x in result.summary["by_step"]], [2, 1, 1])
        self.assertEqual(sum(len(s) for s, _ in model.inputs), 8)
        self.assertTrue(np.isnan(result.traces["free_states"][1, 2:]).all())
        self.assertTrue(np.isnan(result.traces["free_states"][2, 1:]).all())
        np.testing.assert_array_equal(result.traces["free_inference_valid_mask"], mask)
        self.assertEqual(result.traces["free_failure_reason"][1, 1], "not_measured")

    def test_all_invalid_measured_prefix_preserves_empty_diagnostic_without_inference(self):
        states, actions = sequence()
        states[:, 1:] = np.nan
        actions[:] = np.nan
        model = RelativePredictor()
        result = self.diagnose(model, states, actions,
                               actual_valid_mask=np.zeros((1, 3), dtype=np.bool_))
        self.assertEqual(model.inputs, [])
        self.assertEqual(result.summary["by_step"][0]["free"]["metrics_valid_inference_only"]["object_xy_error_m"]["count"], 0)
        json.dumps(result.summary, allow_nan=False)

    def test_raw_yaw_failure_preserves_delta_and_free_prefix_cannot_restart(self):
        states, actions = sequence(batch=2)
        states[0, 0, 0] = -1.

        def predictor(current, command):
            delta = np.zeros_like(current)
            delta[current[:, 0] < 0, 2] = math.pi
            return delta

        result = self.diagnose(predictor, states, actions)
        self.assertEqual(result.traces["free_predicted_deltas"][0, 0, 2], math.pi)
        self.assertEqual(result.traces["free_failure_reason"][0, 0], "raw_delta_yaw_abs_ge_pi")
        self.assertEqual(result.traces["free_failure_reason"][0, 1], "prior_inference_failure")
        self.assertTrue(np.isnan(result.traces["free_states"][0, 1:]).all())
        np.testing.assert_array_equal(result.traces["teacher_inference_valid_mask"][0], [False, True, True])
        np.testing.assert_array_equal(result.traces["free_inference_valid_mask"][1], [True]*3)
        self.assertEqual(len(result.summary["invalid_inference_details"]["free"]), 3)

    def test_batch_rejection_does_not_hide_a_valid_peer(self):
        states, actions = sequence(batch=2, horizon=2)
        states[0, :, 0] = -1.

        def predictor(current, command):
            if np.any(current[:, 0] < 0):
                raise ValueError("bad negative fixture row")
            return np.zeros_like(current)

        result = self.diagnose(predictor, states, actions)
        self.assertIn("inference_rejected:ValueError", result.traces["free_failure_reason"][0, 0])
        np.testing.assert_array_equal(result.traces["free_inference_valid_mask"][1], [True, True])
        self.assertEqual(result.summary["by_step"][1]["free"]["invalid_or_unavailable_inference_transitions"], 1)

    def test_nonfinite_and_wrong_dtype_prediction_are_retained_as_failures(self):
        for predictor, expected in ((constant_delta({0: np.nan}), "nonfinite_physical_delta"),
                                     (lambda s, a: np.zeros_like(s, dtype=np.float32), "inference_rejected:ModelDiagnosticError")):
            with self.subTest(expected=expected):
                result = self.diagnose(predictor)
                self.assertIn(expected, result.traces["free_failure_reason"][0, 0])
                self.assertFalse(result.traces["free_inference_valid_mask"].any())
                json.dumps(result.summary, allow_nan=False)

    def test_infrastructure_exception_is_not_relabelled_as_model_failure(self):
        def predictor(s, a):
            raise RuntimeError("missing external resource")
        with self.assertRaisesRegex(RuntimeError, "missing external resource"):
            self.diagnose(predictor)

    def test_support_nn_distance_has_exact_fixture_answer_and_duplicate_semantics(self):
        support = self.support
        expected = math.sqrt(44.)
        np.testing.assert_array_equal(support.train_leave_one_out_distances, [expected]*2)
        states, actions = sequence(horizon=1)
        result = self.diagnose(constant_delta({}), states, actions)
        features = result.traces["actual_input_normalized_features"][0, 0]
        brute = np.linalg.norm(support.normalized_train_features-features, axis=1)
        self.assertEqual(result.traces["actual_input_nearest_train_row"][0, 0], int(brute.argmin()))
        self.assertAlmostEqual(result.traces["actual_input_nearest_train_distance"][0, 0], float(brute.min()))
        duplicate = np.repeat(self.train, 2, axis=0)
        normalizer, _ = fixture_normalizer(duplicate)
        duplicate_support = make_feature_support(normalizer, duplicate)
        np.testing.assert_array_equal(duplicate_support.train_leave_one_out_distances, np.zeros(4))

    def test_numerical_cos_boundary_is_not_a_material_range_violation(self):
        train = self.train.copy()
        train[:, 3] = .999999997
        normalizer, _ = fixture_normalizer(train)
        support = make_feature_support(normalizer, train)
        states, actions = sequence(horizon=1)
        actions[0, 0, 1] = 1.01
        result = diagnose_model_sequence(constant_delta({}), support, states, actions)
        self.assertEqual(result.traces["actual_input_features"][0, 0, 3], 1.)
        self.assertFalse(result.traces["actual_input_feature_outside_train_range"][0, 0, 3])
        self.assertTrue(result.traces["actual_input_feature_outside_train_range"][0, 0, 10])
        self.assertTrue(result.traces["actual_input_feature_above_train_range"][0, 0, 10])
        # A descriptive range finding never rejects an otherwise valid rollout.
        self.assertTrue(result.traces["free_inference_valid_mask"].all())

    def test_source_arrays_statistics_and_global_rng_are_unchanged(self):
        states, actions = sequence()
        original_states, original_actions = states.copy(), actions.copy()
        original_metadata = deepcopy(self.normalizer.metadata)
        arrays = {k: v.copy() for k, v in self.normalizer.arrays.items()}
        numpy_rng = deepcopy(np.random.get_state())
        torch_rng = torch.get_rng_state().clone()

        def mutating_predictor(current, command):
            current[:] = 100.
            command[:] = -100.
            return np.zeros_like(current)

        result = self.diagnose(mutating_predictor, states, actions)
        np.testing.assert_array_equal(states, original_states)
        np.testing.assert_array_equal(actions, original_actions)
        np.testing.assert_array_equal(result.traces["free_states"], original_states)
        self.assertEqual(self.normalizer.metadata, original_metadata)
        for key, value in arrays.items():
            np.testing.assert_array_equal(self.normalizer.arrays[key], value)
        self.assertEqual(np.random.get_state()[0], numpy_rng[0])
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
        self.assertEqual(np.random.get_state()[2:], numpy_rng[2:])
        torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
        self.assertFalse(self.support.train_features.flags.writeable)
        self.assertFalse(self.support.normalized_train_features.flags.writeable)

    def test_registered_mlp_adapter_uses_real_normalization_without_weight_or_mode_change(self):
        model = DeltaMLP(self.config)
        model.train()
        weights = {k: v.clone() for k, v in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        states, actions = sequence(batch=2, horizon=2)
        result = self.diagnose(LearnedPredictor(model, self.normalizer), states, actions)
        self.assertTrue(result.traces["free_inference_valid_mask"].all())
        self.assertTrue(model.training)
        for key, value in weights.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_input_contract_rejects_restart_wrong_shape_dtype_and_nonfinite_valid_truth(self):
        states, actions = sequence()
        cases = [(states.astype(np.float32), actions, None),
                 (states, actions[:, :, :1], None),
                 (states, actions.astype(np.float32), None),
                 (states, actions, np.array([[1, 0, 1]], dtype=np.bool_)),
                 (states, actions, np.ones((1, 3), dtype=np.int8))]
        bad_states = states.copy()
        bad_states[0, 1, 0] = np.nan
        cases.append((bad_states, actions, None))
        for s, a, mask in cases:
            with self.subTest(shape=s.shape, action_dtype=a.dtype):
                with self.assertRaises(ModelDiagnosticError):
                    self.diagnose(RelativePredictor(), s, a, actual_valid_mask=mask)

    def test_support_rejects_refit_or_mismatched_rows_and_invalid_tolerance(self):
        bad = self.train.copy()
        bad[0, 0] += .1
        with self.assertRaisesRegex(ModelDiagnosticError, "statistics"):
            make_feature_support(self.normalizer, bad)
        with self.assertRaisesRegex(ModelDiagnosticError, "count"):
            make_feature_support(self.normalizer, self.train[:1])
        for value in (True, -1., np.nan, np.inf):
            with self.assertRaises(ModelDiagnosticError):
                make_feature_support(self.normalizer, self.train, marginal_atol=value)
        support = make_feature_support(self.normalizer, self.train)
        support.metadata["train_features_sha256"] = "0"*64
        with self.assertRaisesRegex(ModelDiagnosticError, "content changed"):
            diagnose_model_sequence(RelativePredictor(), support, *sequence())


if __name__ == "__main__":
    unittest.main()
