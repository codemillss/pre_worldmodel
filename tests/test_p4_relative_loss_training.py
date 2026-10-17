"""Independent synthetic checks of the single relative-XY objective change.

No simulator, stored training windows, or registered checkpoint is used. The
small optimizer updates here are algebraic unit checks, not formal experiment
updates. The reference arithmetic is written in physical coordinates rather
than importing the production auxiliary-loss helper.
"""
from copy import deepcopy
import math
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from push_align.normalization import fit_normalizer
from push_align.representation import (DELTA_FIELDS, DELTA_UNITS, FEATURE_FIELDS,
                                       FEATURE_UNITS, STATE_FIELDS, STATE_UNITS)
from push_align.relative_loss_training import (relative_scale,
                                               relative_sequence_loss,
                                               relative_sequence_update)
from push_align.sequence_training import sequence_loss, sequence_update
from push_align.training import TrainingError


def synthetic_normalizer(delta_scale=None):
    """A legal train-only normalizer constructed solely from two synthetic rows."""
    scale = np.ones(10) if delta_scale is None else np.asarray(delta_scale, np.float64)
    schema = {
        "schema_version": "push-align-learning-v1", "coordinates": "world_SI",
        "feature_yaw": "sin_2yaw_cos_2yaw_pi_symmetry",
        "delta_yaw": "wrap_2pi; interval absolute displacement strictly below pi",
        "action_semantics": "applied constant target world XY velocity, not measured response",
        "state_fields": list(STATE_FIELDS), "state_units": list(STATE_UNITS),
        "feature_fields": list(FEATURE_FIELDS), "feature_units": list(FEATURE_UNITS),
        "delta_fields": list(DELTA_FIELDS), "delta_units": list(DELTA_UNITS),
        "physical_dtype": "float64", "model_dtype": "float32", "action_dt_s": .1,
        "normalization": {"fit_split": "train", "input_fixed_unit_indices": [2, 3],
                          "rollout_error_scale": "train_delta_scale",
                          "input_scale_floor": [1.] * 11,
                          "delta_scale_floor": [1.] * 10},
    }
    for key in ("data_contract_hash", "environment_hash", "compiled_model_hash",
                "config_sha256", "profile_sha256"):
        schema[key] = "a" * 64
    identity = {"dataset_id": "relative-objective-analytic-fixture",
                "manifest_sha256": "b" * 64, "data_contract_hash": "a" * 64}
    features = np.repeat(np.array([[-1.], [1.]], np.float64), 11, axis=1)
    return fit_normalizer(features, np.stack((-scale, scale)), np.zeros(2, np.int8),
                          schema, identity, np.zeros(2, np.int64),
                          np.arange(2, dtype=np.int64))


class ConstantDelta(nn.Module):
    """A learnable float32 normalized delta, with observable forward calls."""
    def __init__(self, normalized_delta=None):
        super().__init__()
        value = np.zeros(10) if normalized_delta is None else normalized_delta
        self.delta = nn.Parameter(torch.tensor(value, dtype=torch.float32))
        self.inputs = []
        self.outputs = []

    def forward(self, features):
        self.inputs.append(features)
        output = self.delta.expand(*features.shape[:-1], 10)
        self.outputs.append(output)
        return output


class RecurrentToy(nn.Module):
    """o_x(k+1)=(1-a)o_x(k)+a*p_x(k); p_x remains unchanged."""
    def __init__(self, a=.25):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        self.inputs = []
        self.outputs = []

    def forward(self, features):
        self.inputs.append(features)
        zero = features[..., 0] * 0
        output = torch.stack([self.a * features[..., 0]] + [zero] * 9, dim=-1)
        self.outputs.append(output)
        return output


def batch(lengths=(3,), horizon=3):
    states = torch.zeros((len(lengths), horizon + 1, 10), dtype=torch.float64)
    states[..., 6] = 1.
    actions = torch.zeros((len(lengths), horizon, 2), dtype=torch.float64)
    mask = torch.arange(horizon)[None, :] < torch.tensor(lengths)[:, None]
    return states, actions, mask


def physical_model(delta, normalizer):
    return ConstantDelta(np.asarray(delta) / normalizer.arrays["delta_scale"])


def exact_tree(test, first, second):
    if isinstance(first, torch.Tensor):
        test.assertEqual(first.dtype, second.dtype)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        test.assertEqual(first.keys(), second.keys())
        for key in first:
            exact_tree(test, first[key], second[key])
    elif isinstance(first, (list, tuple)):
        test.assertEqual(len(first), len(second))
        for one, two in zip(first, second):
            exact_tree(test, one, two)
    else:
        test.assertEqual(first, second)


class RelativeLossTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.normalizer = synthetic_normalizer()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_fixed_rss_scale_uses_object_and_pusher_delta_coordinates(self):
        scales = np.ones(10); scales[[0, 1, 6, 7]] = [3., 5., 4., 12.]
        normalizer = synthetic_normalizer(scales)
        before = deepcopy(normalizer.arrays)
        result = relative_scale(normalizer)
        torch.testing.assert_close(result, torch.tensor([5., 13.], dtype=torch.float64), rtol=0, atol=0)
        self.assertEqual(result.dtype, torch.float64)
        self.assertEqual(result.device.type, "cpu")
        for key, value in before.items():
            np.testing.assert_array_equal(normalizer.arrays[key], value)

    def test_hand_computed_teacher_and_free_relative_losses_and_total(self):
        states, actions, mask = batch()
        delta = np.zeros(10); delta[0] = 1.; delta[6] = -2.
        result = relative_sequence_loss(physical_model(delta, self.normalizer), self.normalizer,
                                        states, actions, mask, .2, .1)
        # Relative residual is -3 per step; its free endpoint value is -3*k.
        # RSS scale squared=2, coordinate denominator=2, and N=3.
        teacher = 9. / 4.
        free = 9. * sum(k*k for k in (1, 2, 3)) / (4.*3)
        base_teacher = 5. / 10.
        base_free = 5. * sum(k*k for k in (1, 2, 3)) / (10.*3)
        self.assertAlmostEqual(float(result.relative_teacher.detach()), teacher, places=12)
        self.assertAlmostEqual(float(result.relative_rollout.detach()), free, places=12)
        self.assertAlmostEqual(float(result.total.detach()), base_teacher + .1*base_free + .2*(teacher + .1*free), places=12)
        self.assertEqual((result.valid_transitions, result.scalar_terms, result.relative_scalar_terms), (3, 30, 6))
        np.testing.assert_array_equal(result.teacher_xy_errors.detach().numpy(), [[1., 0., -2., 0.]]*3)
        np.testing.assert_array_equal(result.free_xy_errors.detach().numpy(),
                                      [[1., 0., -2., 0.], [2., 0., -4., 0.], [3., 0., -6., 0.]])

    def test_anisotropic_scale_matches_physical_coordinate_reference(self):
        scales = np.ones(10); scales[[0, 1, 6, 7]] = [3., 5., 4., 12.]
        norm = synthetic_normalizer(scales)
        states, actions, mask = batch((1,), 1)
        delta = np.zeros(10); delta[[0, 1, 6, 7]] = [3., 5., 8., 24.]
        result = relative_sequence_loss(physical_model(delta, norm), norm, states, actions, mask, .2, 0.)
        expected = ((5./5.)**2 + (19./13.)**2) / 2.
        self.assertAlmostEqual(float(result.relative_teacher.detach()), expected, places=12)
        self.assertIsNone(result.relative_rollout)

    def test_next_relative_position_equals_teacher_relative_delta_residual(self):
        states, actions, mask = batch()
        states[0, :, 0] = torch.tensor([.25, .5, 1., .75], dtype=torch.float64)
        states[0, :, 6] = torch.tensor([1., 1.5, 1.75, 2.], dtype=torch.float64)
        delta = np.zeros(10); delta[0] = .125; delta[6] = -.25
        result = relative_sequence_loss(physical_model(delta, self.normalizer), self.normalizer,
                                        states, actions, mask, .2, 0.)
        current, target = states[:, :-1][mask].numpy(), states[:, 1:][mask].numpy()
        predicted = current + delta
        expected = (predicted[:, 6:8] - predicted[:, :2]) - (target[:, 6:8] - target[:, :2])
        errors = result.teacher_xy_errors.detach().numpy()
        np.testing.assert_array_equal(errors[:, 2:] - errors[:, :2], expected)

    def test_equal_common_position_error_has_zero_auxiliary_but_nonzero_base(self):
        states, actions, mask = batch()
        delta = np.zeros(10); delta[[0, 1, 6, 7]] = [.5, -.25, .5, -.25]
        result = relative_sequence_loss(physical_model(delta, self.normalizer), self.normalizer,
                                        states, actions, mask)
        self.assertEqual(float(result.relative_teacher.detach()), 0.)
        self.assertEqual(float(result.relative_rollout.detach()), 0.)
        self.assertGreater(float(result.one_step.detach()), 0.)
        self.assertGreater(float(result.rollout.detach()), 0.)

    def test_opposed_errors_increase_relative_penalty_by_four(self):
        states, actions, mask = batch((1,), 1)
        single = np.zeros(10); single[0] = .5
        opposed = single.copy(); opposed[6] = -.5
        one = relative_sequence_loss(physical_model(single, self.normalizer), self.normalizer, states, actions, mask, .2, 0.)
        two = relative_sequence_loss(physical_model(opposed, self.normalizer), self.normalizer, states, actions, mask, .2, 0.)
        self.assertAlmostEqual(float(two.relative_teacher.detach()), 4.*float(one.relative_teacher.detach()), places=12)

    def test_auxiliary_gradient_couples_object_and_pusher_with_opposite_signs(self):
        states, actions, mask = batch((1,), 1)
        delta = np.zeros(10); delta[0] = 1.; delta[6] = -2.
        model = physical_model(delta, self.normalizer)
        result = relative_sequence_loss(model, self.normalizer, states, actions, mask, .2, 0.)
        gradient = torch.autograd.grad(result.relative_teacher, model.delta)[0]
        expected = torch.zeros(10); expected[0] = 1.5; expected[6] = -1.5
        torch.testing.assert_close(gradient, expected, rtol=0, atol=0)

    def test_auxiliary_is_xy_only_while_original_loss_keeps_other_components(self):
        states, actions, mask = batch()
        delta = np.zeros(10); delta[[2, 3, 4, 5, 8, 9]] = [.125, 2., -3., 1., 4., -5.]
        model = physical_model(delta, self.normalizer)
        result = relative_sequence_loss(model, self.normalizer, states, actions, mask)
        self.assertEqual(float(result.relative_teacher.detach()), 0.)
        self.assertEqual(float(result.relative_rollout.detach()), 0.)
        self.assertGreater(float(result.total.detach()), 0.)
        gradient = torch.autograd.grad(result.relative_teacher, model.delta)[0]
        torch.testing.assert_close(gradient, torch.zeros(10), rtol=0, atol=0)

    def test_common_world_translation_preserves_losses_predictions_and_gradients(self):
        states, actions, mask = batch()
        shifted = states.clone(); shifted[..., [0, 1, 6, 7]] += torch.tensor([8., -4., 8., -4.], dtype=torch.float64)
        first_model, second_model = RecurrentToy(), RecurrentToy()
        one = relative_sequence_loss(first_model, self.normalizer, states, actions, mask)
        two = relative_sequence_loss(second_model, self.normalizer, shifted, actions, mask)
        for name in ("one_step", "rollout", "relative_teacher", "relative_rollout", "total"):
            torch.testing.assert_close(getattr(one, name), getattr(two, name), rtol=0, atol=0)
        for first, second in zip(first_model.inputs, second_model.inputs):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        one.total.backward(); two.total.backward()
        torch.testing.assert_close(first_model.a.grad, second_model.a.grad, rtol=0, atol=0)

    def test_prefix_mask_denominators_use_recorded_valid_steps_only(self):
        states, actions, mask = batch((3, 1, 0))
        delta = np.zeros(10); delta[0] = 1.; delta[6] = -2.
        model = physical_model(delta, self.normalizer)
        result = relative_sequence_loss(model, self.normalizer, states, actions, mask)
        self.assertEqual((result.valid_transitions, result.scalar_terms, result.relative_scalar_terms), (4, 40, 8))
        expected = 9.*(1.+4.+9.+1.)/(4.*4)
        self.assertAlmostEqual(float(result.relative_rollout.detach()), expected, places=12)
        self.assertEqual(result.teacher_xy_errors.shape, (4, 4))
        self.assertEqual(result.free_xy_errors.shape, (4, 4))
        self.assertEqual([len(value) for value in model.inputs], [4, 2, 1, 1])

    def test_nonfinite_suffix_padding_is_excluded_without_changing_graph(self):
        states, actions, mask = batch((3, 1, 0))
        dirty_states, dirty_actions = states.clone(), actions.clone()
        for row, length in enumerate((3, 1, 0)):
            dirty_states[row, length+1:] = float("nan")
            dirty_actions[row, length:] = float("nan")
        first_model, second_model = RecurrentToy(), RecurrentToy()
        one = relative_sequence_loss(first_model, self.normalizer, states, actions, mask)
        two = relative_sequence_loss(second_model, self.normalizer, dirty_states, dirty_actions, mask)
        torch.testing.assert_close(one.total, two.total, rtol=0, atol=0)
        torch.testing.assert_close(one.predicted_states, two.predicted_states, rtol=0, atol=0)
        one.total.backward(); two.total.backward()
        torch.testing.assert_close(first_model.a.grad, second_model.a.grad, rtol=0, atol=0)

    def test_padding_inputs_and_targets_receive_no_gradient(self):
        states, actions, mask = batch((3, 1, 0))
        states.requires_grad_(); actions.requires_grad_()
        result = relative_sequence_loss(RecurrentToy(), self.normalizer, states, actions, mask)
        state_gradient, action_gradient = torch.autograd.grad(result.total, (states, actions))
        torch.testing.assert_close(state_gradient[1, 2:], torch.zeros_like(state_gradient[1, 2:]), rtol=0, atol=0)
        torch.testing.assert_close(state_gradient[2], torch.zeros_like(state_gradient[2]), rtol=0, atol=0)
        torch.testing.assert_close(action_gradient[~mask], torch.zeros_like(action_gradient[~mask]), rtol=0, atol=0)
        self.assertGreater(float(state_gradient[0, 1:].abs().sum()), 0.)

    def test_invalid_mask_restart_and_empty_population_are_rejected(self):
        states, actions, mask = batch()
        for invalid in (torch.tensor([[True, False, True]]), torch.zeros_like(mask)):
            with self.subTest(mask=invalid.tolist()), self.assertRaises(TrainingError):
                relative_sequence_loss(RecurrentToy(), self.normalizer, states, actions, invalid)

    def test_zero_weight_loss_has_exact_original_values_gradients_and_forwards(self):
        states, actions, mask = batch((5, 5), 5)
        first_model, second_model = RecurrentToy(), RecurrentToy()
        before_rng = torch.get_rng_state().clone()
        old = sequence_loss(first_model, self.normalizer, states, actions, mask, .1)
        result = relative_sequence_loss(second_model, self.normalizer, states, actions, mask, 0., .1)
        for name in ("one_step", "rollout", "total", "predicted_states"):
            torch.testing.assert_close(getattr(old, name), getattr(result, name), rtol=0, atol=0)
        old.total.backward(); result.total.backward()
        torch.testing.assert_close(first_model.a.grad, second_model.a.grad, rtol=0, atol=0)
        self.assertEqual([len(x) for x in first_model.inputs], [10, 2, 2, 2, 2, 2])
        for first, second in zip(first_model.inputs, second_model.inputs):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        for name in ("relative_teacher", "relative_rollout", "teacher_xy_errors", "free_xy_errors"):
            self.assertIsNone(getattr(result, name))
        self.assertEqual(result.relative_scalar_terms, 0)
        torch.testing.assert_close(before_rng, torch.get_rng_state(), rtol=0, atol=0)

    def test_zero_weight_update_exact_adam_state_parameters_and_rng_after_warmup(self):
        states, actions, mask = batch()
        delta = np.zeros(10); delta[[0, 6]] = [.25, -.125]
        first_model, second_model = ConstantDelta(delta), ConstantDelta(delta)
        first_optimizer = torch.optim.Adam(first_model.parameters(), lr=.001)
        second_optimizer = torch.optim.Adam(second_model.parameters(), lr=.001)
        # Existing Adam moments must remain equal, not merely uninitialized states.
        sequence_update(first_model, first_optimizer, self.normalizer, states, actions, mask, .1, 1.)
        sequence_update(second_model, second_optimizer, self.normalizer, states, actions, mask, .1, 1.)
        before_rng = torch.get_rng_state().clone()
        for _ in range(3):
            one = sequence_update(first_model, first_optimizer, self.normalizer, states, actions, mask, .1, 1.)
            two = relative_sequence_update(second_model, second_optimizer, self.normalizer, states, actions, mask,
                                           relative_weight=0., rollout_weight=.1, clip_norm=1.)
            self.assertEqual(one, {key: two[key] for key in one})
            exact_tree(self, first_model.state_dict(), second_model.state_dict())
            exact_tree(self, first_optimizer.state_dict(), second_optimizer.state_dict())
            self.assertIsNone(two["relative_teacher_loss"])
            self.assertIsNone(two["relative_rollout_loss"])
            self.assertEqual(two["relative_scalar_terms"], 0)
        torch.testing.assert_close(before_rng, torch.get_rng_state(), rtol=0, atol=0)

    def test_positive_weight_reuses_exactly_one_teacher_and_five_free_forwards(self):
        states, actions, mask = batch((5, 5), 5)
        model = RecurrentToy()
        result = relative_sequence_loss(model, self.normalizer, states, actions, mask, .2, .1)
        self.assertEqual([len(x) for x in model.inputs], [10, 2, 2, 2, 2, 2])
        self.assertEqual(len(model.outputs), 6)
        self.assertEqual(len(model._forward_hooks), 0)
        self.assertEqual(result.teacher_xy_errors.shape, (10, 4))
        self.assertEqual(result.free_xy_errors.shape, (10, 4))

    def test_zero_rollout_weight_uses_only_teacher_forward(self):
        states, actions, mask = batch()
        model = RecurrentToy()
        result = relative_sequence_loss(model, self.normalizer, states, actions, mask, .2, 0.)
        self.assertEqual(len(model.inputs), 1)
        for name in ("rollout", "relative_rollout", "predicted_states", "free_xy_errors"):
            self.assertIsNone(getattr(result, name))
        torch.testing.assert_close(result.total, result.one_step+.2*result.relative_teacher, rtol=0, atol=0)

    def test_free_auxiliary_gradient_reaches_early_output_and_matches_closed_form(self):
        states, actions, mask = batch()
        model = RecurrentToy()
        result = relative_sequence_loss(model, self.normalizer, states, actions, mask)
        expected_x = np.array([.25, .4375, .578125])
        derivatives = np.array([1., 1.5, 1.6875])
        expected = np.square(expected_x).sum() / 12.
        self.assertAlmostEqual(float(result.relative_rollout.detach()), expected, places=12)
        gradient = torch.autograd.grad(result.relative_rollout, model.a, retain_graph=True)[0]
        # The physical objective is float64, while accumulated parameter
        # gradients cross the registered float32 model boundary.
        self.assertAlmostEqual(float(gradient), float(2.*np.dot(expected_x, derivatives)/12.), delta=5e-8)
        early_gradient = torch.autograd.grad(result.predicted_states[0, -1, 0], model.outputs[1], retain_graph=True)[0]
        self.assertAlmostEqual(float(early_gradient[0, 0]), .75**2, places=7)
        # A terminal-only relative penalty still must reach the first free output.
        terminal_relative = result.free_xy_errors[-1, 2]-result.free_xy_errors[-1, 0]
        terminal_gradient = torch.autograd.grad(terminal_relative, model.outputs[1])[0]
        self.assertAlmostEqual(float(terminal_gradient[0, 0]), -.75**2, places=7)

    def test_free_auxiliary_gradient_matches_independent_finite_difference(self):
        states, actions, mask = batch()
        model = RecurrentToy()
        center = relative_sequence_loss(model, self.normalizer, states, actions, mask)
        gradient = float(torch.autograd.grad(center.relative_rollout, model.a)[0])
        epsilon = 1./1024.
        plus = relative_sequence_loss(RecurrentToy(.25+epsilon), self.normalizer,
                                      states, actions, mask).relative_rollout
        minus = relative_sequence_loss(RecurrentToy(.25-epsilon), self.normalizer,
                                       states, actions, mask).relative_rollout
        numerical = float((plus-minus).detach())/(2.*epsilon)
        self.assertAlmostEqual(gradient, numerical, delta=1e-6)

    def test_future_truth_changes_labels_without_changing_free_inputs(self):
        states, actions, mask = batch()
        changed = states.clone(); changed[0, 1:, 0] = torch.tensor([.5, -.25, .75], dtype=torch.float64)
        one_model, two_model = RecurrentToy(), RecurrentToy()
        one = relative_sequence_loss(one_model, self.normalizer, states, actions, mask)
        two = relative_sequence_loss(two_model, self.normalizer, changed, actions, mask)
        torch.testing.assert_close(one.predicted_states, two.predicted_states, rtol=0, atol=0)
        for first, second in zip(one_model.inputs[1:], two_model.inputs[1:]):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertFalse(torch.equal(one.teacher_xy_errors, two.teacher_xy_errors))
        self.assertFalse(torch.equal(one.free_xy_errors, two.free_xy_errors))

    def test_nonfinite_model_output_fails_and_always_removes_temporary_hook(self):
        states, actions, mask = batch()
        delta = np.zeros(10); delta[0] = math.nan
        model = ConstantDelta(delta)
        with self.assertRaises(TrainingError):
            relative_sequence_loss(model, self.normalizer, states, actions, mask)
        self.assertEqual(len(model._forward_hooks), 0)

    def test_finite_forbidden_raw_yaw_fails_without_masking_recorded_steps(self):
        states, actions, mask = batch()
        delta = np.zeros(10); delta[2] = 4.
        model = ConstantDelta(delta)
        with self.assertRaisesRegex(TrainingError, "delta_yaw"):
            relative_sequence_loss(model, self.normalizer, states, actions, mask)
        self.assertEqual(len(model._forward_hooks), 0)
        self.assertTrue(mask.all())

    def test_update_diagnostics_preserve_independent_physical_error_arrays(self):
        states, actions, mask = batch()
        delta = np.zeros(10); delta[[0, 6]] = [.25, -.5]
        model = ConstantDelta(delta); optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        before = model.delta.detach().clone()
        result = relative_sequence_update(model, optimizer, self.normalizer, states, actions, mask)
        self.assertEqual((result["valid_transitions"], result["scalar_terms"], result["relative_scalar_terms"]), (3, 30, 6))
        np.testing.assert_array_equal(result["teacher_xy_errors"], [[.25, 0., -.5, 0.]]*3)
        np.testing.assert_array_equal(result["free_xy_errors"], [[.25, 0., -.5, 0.], [.5, 0., -1., 0.], [.75, 0., -1.5, 0.]])
        self.assertTrue(np.isfinite(result["gradient_norm_before_clip"]))
        self.assertFalse(torch.equal(before, model.delta))
        self.assertAlmostEqual(result["loss_before_update"], result["one_step_loss"]+.1*result["rollout_loss"]+.2*(result["relative_teacher_loss"]+.1*result["relative_rollout_loss"]), places=12)

    def test_invalid_weights_scales_and_clipping_are_rejected(self):
        states, actions, mask = batch()
        for weight in (True, False, -.1, math.nan, math.inf, "0.2"):
            with self.subTest(relative_weight=weight), self.assertRaises(TrainingError):
                relative_sequence_loss(RecurrentToy(), self.normalizer, states, actions, mask, weight, .1)
        for weight in (True, -.1, math.nan, math.inf):
            with self.subTest(rollout_weight=weight), self.assertRaises(TrainingError):
                relative_sequence_loss(RecurrentToy(), self.normalizer, states, actions, mask, .2, weight)
        for scale in (np.ones(9), np.zeros(10), -np.ones(10), np.full(10, math.nan)):
            with self.subTest(scale=scale.tolist()), self.assertRaises(TrainingError):
                relative_scale(SimpleNamespace(arrays={"delta_scale": scale}))
        for value in (True, 0., -1., math.nan, math.inf):
            model = RecurrentToy(); optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            with self.subTest(clip_norm=value), self.assertRaises(TrainingError):
                relative_sequence_update(model, optimizer, self.normalizer, states, actions, mask, clip_norm=value)


if __name__ == "__main__":
    unittest.main()
