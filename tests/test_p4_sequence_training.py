"""Independent P4 recurrence, mask, denominator, and optimizer checks.

The algebraic fixtures are not contact physics or task-performance evidence.
Their purpose is to make wrong teacher forcing, detached recurrence, and loss
denominators distinguishable from the registered training definition.
"""
from copy import deepcopy
import math
import unittest

import numpy as np
import torch
from torch import nn

from push_align import PlanarPushEnv
from push_align.data import contract_hash, make_contract
from push_align.dynamics import predict_next
from push_align.normalization import fit_normalizer, learning_schema
from push_align.representation import encode, state_delta
from push_align.sequence_training import (masked_free_rollout, sequence_loss,
                                         sequence_update)
from push_align.training import TrainingError, one_step_loss, training_update


class RelativeToy(nn.Module):
    """x_next=(1-a)*x+a*pusher_x; other states stay unchanged.

    A unit normalizer makes the normalized model output physical delta.
    Keeping the model float32 exercises the real recurrent dtype boundary.
    """
    def __init__(self, a=.25, action_moves_pusher=False):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
        self.action_moves_pusher = action_moves_pusher
        self.inputs = []
        self.outputs = []

    def forward(self, features):
        self.inputs.append(features)
        zero = features[..., 0]*0
        columns = [self.a*features[..., 0]]+[zero]*9
        if self.action_moves_pusher:
            columns[6] = .1*features[..., 9]
        result = torch.stack(columns, dim=-1)
        self.outputs.append(result)
        return result


class ConstantToy(nn.Module):
    def __init__(self, delta=None, dtype=torch.float32):
        super().__init__()
        value = torch.zeros(10, dtype=dtype) if delta is None else torch.tensor(delta, dtype=dtype)
        self.delta = nn.Parameter(value)

    def forward(self, features):
        return self.delta.expand(*features.shape[:-1], 10)


def toy_batch(lengths=(3,), horizon=3):
    states = torch.zeros((len(lengths), horizon+1, 10), dtype=torch.float64)
    states[..., 6] = 1.
    actions = torch.zeros((len(lengths), horizon, 2), dtype=torch.float64)
    mask = torch.arange(horizon)[None, :] < torch.tensor(lengths)[:, None]
    return states, actions, mask


def optimizer_equal(test, first, second):
    one, two = first.state_dict(), second.state_dict()
    test.assertEqual(one["param_groups"], two["param_groups"])
    test.assertEqual(one["state"].keys(), two["state"].keys())
    for parameter, values in one["state"].items():
        test.assertEqual(values.keys(), two["state"][parameter].keys())
        for field, value in values.items():
            other = two["state"][parameter][field]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, other, rtol=0, atol=0)
            else:
                test.assertEqual(value, other)


class P4SequenceTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        contract = make_contract(PlanarPushEnv())
        schema = learning_schema(contract)
        identity = {"dataset_id": "p4-analytic-fixture", "manifest_sha256": "b"*64,
                    "data_contract_hash": contract_hash(contract)}
        features = np.repeat(np.array([[-1.], [1.]], dtype=np.float64), 11, axis=1)
        deltas = np.repeat(np.array([[-1.], [1.]], dtype=np.float64), 10, axis=1)
        cls.normalizer = fit_normalizer(features, deltas, np.zeros(2, dtype=np.int8),
                                        schema, identity, np.zeros(2, dtype=np.int64),
                                        np.arange(2, dtype=np.int64))
        np.testing.assert_array_equal(cls.normalizer.arrays["input_mean"], np.zeros(11))
        np.testing.assert_array_equal(cls.normalizer.arrays["input_scale"], np.ones(11))
        np.testing.assert_array_equal(cls.normalizer.arrays["delta_mean"], np.zeros(10))
        np.testing.assert_array_equal(cls.normalizer.arrays["delta_scale"], np.ones(10))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_closed_form_k1_to_k3_losses_and_gradient(self):
        states, actions, mask = toy_batch()
        model = RelativeToy()
        result = sequence_loss(model, self.normalizer, states, actions, mask, rollout_weight=.1)
        expected_x = np.array([.25, .4375, .578125])
        derivatives = np.array([1., 1.5, 1.6875])
        expected_rollout = np.square(expected_x).sum()/30
        expected_gradient = 2*(expected_x*derivatives).sum()/30
        self.assertEqual(result.valid_transitions, 3)
        self.assertEqual(result.scalar_terms, 30)
        self.assertAlmostEqual(float(result.one_step.detach()), .25**2/10, places=12)
        self.assertAlmostEqual(float(result.rollout.detach()), expected_rollout, places=12)
        self.assertAlmostEqual(float(result.total.detach()), .25**2/10+.1*expected_rollout, places=12)
        torch.testing.assert_close(result.predicted_states[0, 1:, 0],
                                   torch.tensor(expected_x), rtol=0, atol=0)
        grad = torch.autograd.grad(result.rollout, model.a, retain_graph=True)[0]
        self.assertAlmostEqual(float(grad), expected_gradient, delta=1e-8)
        result.total.backward()
        self.assertAlmostEqual(float(model.a.grad), .05+.1*expected_gradient, delta=1e-8)

    def test_terminal_gradient_reaches_first_rollout_output_and_detach_is_detectable(self):
        states, actions, mask = toy_batch()
        model = RelativeToy()
        result = sequence_loss(model, self.normalizer, states, actions, mask, rollout_weight=1.)
        # Identify the recurrent calls by their one-row batch, independent of
        # whether the common real-state one-step term is evaluated first.
        self.assertEqual(sorted(len(value) for value in model.inputs), [1, 1, 1, 3])
        terminal = result.predicted_states[0, -1, 0]
        first_recurrent_output = next(output for inputs, output in zip(model.inputs, model.outputs)
                                      if len(inputs) == 1)
        early_gradient = torch.autograd.grad(terminal, first_recurrent_output,
                                             retain_graph=True)[0]
        self.assertAlmostEqual(float(early_gradient[0, 0]), .75**2, places=7)
        full_gradient = float(torch.autograd.grad(result.rollout, model.a)[0])
        detached = RelativeToy()
        current = states[:, 0]
        errors = []
        for index in range(3):
            current = predict_next(detached, self.normalizer, current.detach(), actions[:, index])
            errors.append(current[0, 0].square())
        detached_loss = torch.stack(errors).sum()/30
        detached_gradient = float(torch.autograd.grad(detached_loss, detached.a)[0])
        expected_detached = 2*(.25+.4375*.75+.578125*.75**2)/30
        self.assertAlmostEqual(detached_gradient, expected_detached, delta=1e-8)
        self.assertGreater(full_gradient-detached_gradient, .06)

    def test_rollout_gradient_agrees_with_independent_finite_difference(self):
        states, actions, mask = toy_batch()
        center_model = RelativeToy()
        center = sequence_loss(center_model, self.normalizer, states, actions, mask, rollout_weight=1.)
        gradient = float(torch.autograd.grad(center.rollout, center_model.a)[0])
        epsilon = 1/1024
        plus = sequence_loss(RelativeToy(.25+epsilon), self.normalizer, states, actions, mask,
                             rollout_weight=1.).rollout
        minus = sequence_loss(RelativeToy(.25-epsilon), self.normalizer, states, actions, mask,
                              rollout_weight=1.).rollout
        finite_difference = float((plus-minus).detach())/(2*epsilon)
        self.assertAlmostEqual(gradient, finite_difference, delta=5e-7)

    def test_future_truth_changes_targets_and_one_step_inputs_but_not_rollout(self):
        states, actions, mask = toy_batch()
        changed = states.clone()
        changed[:, 1:, 0] = torch.tensor([.8, -.4, .9], dtype=torch.float64)
        first_model, second_model = RelativeToy(), RelativeToy()
        first = sequence_loss(first_model, self.normalizer, states, actions, mask, rollout_weight=.1)
        second = sequence_loss(second_model, self.normalizer, changed, actions, mask, rollout_weight=.1)
        torch.testing.assert_close(first.predicted_states, second.predicted_states, rtol=0, atol=0)
        first_recurrent = [value for value in first_model.inputs if len(value) == 1]
        second_recurrent = [value for value in second_model.inputs if len(value) == 1]
        for one, two in zip(first_recurrent, second_recurrent):
            torch.testing.assert_close(one, two, rtol=0, atol=0)
        first_real = next(value for value in first_model.inputs if len(value) == 3)
        second_real = next(value for value in second_model.inputs if len(value) == 3)
        self.assertFalse(torch.equal(first_real, second_real))
        self.assertNotEqual(float(first.one_step.detach()), float(second.one_step.detach()))
        self.assertNotEqual(float(first.rollout.detach()), float(second.rollout.detach()))

    def test_same_one_step_definition_matches_original_loss_on_all_valid_real_transitions(self):
        states, actions, mask = toy_batch((3, 1))
        states[0, 1:, 0] = torch.tensor([.1, .2, .15], dtype=torch.float64)
        states[1, 1, 0] = .2
        s1_model, sk_model = RelativeToy(), RelativeToy()
        s1 = sequence_loss(s1_model, self.normalizer, states, actions, mask, rollout_weight=0.)
        sk = sequence_loss(sk_model, self.normalizer, states, actions, mask, rollout_weight=.1)
        actual_features = encode(states[:, :-1][mask], actions[mask])
        actual_deltas = state_delta(states[:, :-1][mask], states[:, 1:][mask])
        expected = one_step_loss(RelativeToy(), self.normalizer, actual_features, actual_deltas)
        torch.testing.assert_close(s1.one_step, expected, rtol=0, atol=0)
        torch.testing.assert_close(sk.one_step, expected, rtol=0, atol=0)
        self.assertIsNone(s1.rollout)
        self.assertIsNone(s1.predicted_states)
        self.assertEqual(len(s1_model.inputs), 1)
        self.assertEqual(s1.scalar_terms, 40)

    def test_actual_mask_denominator_excludes_k0_and_invalid_suffix(self):
        states, actions, mask = toy_batch((3, 1))
        result = sequence_loss(RelativeToy(), self.normalizer, states, actions, mask, rollout_weight=1.)
        expected = (.25**2+.4375**2+.578125**2+.25**2)/40
        self.assertEqual(result.valid_transitions, 4)
        self.assertEqual(result.scalar_terms, 40)
        self.assertAlmostEqual(float(result.rollout.detach()), expected, places=12)
        self.assertAlmostEqual(float(result.one_step.detach()), .25**2/10, places=12)
        torch.testing.assert_close(result.predicted_states[1, 1:, 0],
                                   torch.full((3,), .25, dtype=torch.float64), rtol=0, atol=0)

    def test_nan_padding_and_zero_valid_row_do_not_affect_loss_gradient_or_forward_count(self):
        states, actions, mask = toy_batch((3, 1, 0))
        clean_model = RelativeToy()
        clean = sequence_loss(clean_model, self.normalizer, states, actions, mask, rollout_weight=.1)
        poison_states, poison_actions = states.clone(), actions.clone()
        for row, length in enumerate((3, 1, 0)):
            poison_states[row, length+1:] = float("nan")
            poison_actions[row, length:] = float("nan")
        poison_model = RelativeToy()
        poison = sequence_loss(poison_model, self.normalizer, poison_states, poison_actions, mask,
                               rollout_weight=.1)
        torch.testing.assert_close(clean.total, poison.total, rtol=0, atol=0)
        torch.testing.assert_close(clean.predicted_states, poison.predicted_states, rtol=0, atol=0)
        clean.total.backward(); poison.total.backward()
        torch.testing.assert_close(clean_model.a.grad, poison_model.a.grad, rtol=0, atol=0)
        self.assertEqual(sorted(len(value) for value in poison_model.inputs), [1, 1, 2, 4])
        torch.testing.assert_close(poison.predicted_states[2], states[2, :1].expand(4, 10), rtol=0, atol=0)

    def test_invalid_padding_has_zero_input_and_target_gradients(self):
        states, actions, mask = toy_batch((3, 1, 0))
        states.requires_grad_(); actions.requires_grad_()
        result = sequence_loss(RelativeToy(), self.normalizer, states, actions, mask, rollout_weight=.1)
        state_gradient, action_gradient = torch.autograd.grad(result.total, (states, actions))
        torch.testing.assert_close(state_gradient[1, 2:], torch.zeros_like(state_gradient[1, 2:]),
                                   rtol=0, atol=0)
        torch.testing.assert_close(state_gradient[2], torch.zeros_like(state_gradient[2]), rtol=0, atol=0)
        torch.testing.assert_close(action_gradient[~mask], torch.zeros_like(action_gradient[~mask]),
                                   rtol=0, atol=0)
        self.assertGreater(float(state_gradient[0, 1:].abs().sum()), 0.)

    def test_masked_rollout_preserves_mode_inputs_and_reaches_early_action(self):
        states, actions, mask = toy_batch((3, 1))
        actions[:, :, 0] = .5
        actions.requires_grad_()
        model = RelativeToy(action_moves_pusher=True).eval()
        before_states, before_actions, before_mask = states.clone(), actions.detach().clone(), mask.clone()
        predicted = masked_free_rollout(model, self.normalizer, states[:, 0], actions, mask)
        predicted[0, -1, [0, 6]].sum().backward()
        expected = torch.tensor([.14375, .125, .1], dtype=torch.float64)
        torch.testing.assert_close(actions.grad[0, :, 0], expected, rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(actions.grad[1], torch.zeros((3, 2), dtype=torch.float64), rtol=0, atol=0)
        self.assertFalse(model.training)
        torch.testing.assert_close(states, before_states, rtol=0, atol=0)
        torch.testing.assert_close(actions.detach(), before_actions, rtol=0, atol=0)
        torch.testing.assert_close(mask, before_mask, rtol=0, atol=0)
        self.assertEqual(predicted.dtype, torch.float64)

    def test_k1_rollout_and_delta_loss_match_with_identical_physical_truth(self):
        states, actions, mask = toy_batch((1,), horizon=1)
        states[0, 1, 0] = .1
        result = sequence_loss(RelativeToy(), self.normalizer, states, actions, mask, rollout_weight=1.)
        torch.testing.assert_close(result.one_step, result.rollout, rtol=0, atol=0)
        self.assertEqual(result.scalar_terms, 10)

    def test_state_yaw_uses_two_pi_wrap_at_boundary_and_not_object_pi_symmetry(self):
        states, actions, mask = toy_batch((1,), horizon=1)
        states[:, :, 6] = 0
        states[0, 0, 2] = math.pi-.01
        states[0, 1, 2] = math.pi-.02
        delta = np.zeros(10); delta[2] = .02
        model = ConstantToy(delta)
        result = sequence_loss(model, self.normalizer, states, actions, mask, rollout_weight=1.)
        predicted_delta = float(model.delta[2].detach())
        expected = (predicted_delta+.01)**2/10
        self.assertAlmostEqual(float(result.rollout.detach()), expected, delta=1e-12)
        self.assertAlmostEqual(float(result.one_step.detach()), expected, delta=1e-12)
        self.assertLess(float(result.predicted_states[0, 1, 2].detach()), -3.)
        half_turn = torch.zeros_like(states); half_turn[0, 1, 2] = math.pi
        half = sequence_loss(ConstantToy(), self.normalizer, half_turn, actions, mask, rollout_weight=1.)
        self.assertAlmostEqual(float(half.rollout.detach()), math.pi**2/10, places=12)

    def test_lambda_zero_update_is_exactly_original_one_step_update_on_shared_windows(self):
        states, actions, mask = toy_batch((3, 1))
        first, second = RelativeToy(), RelativeToy()
        first_optimizer = torch.optim.Adam(first.parameters(), lr=.001)
        second_optimizer = torch.optim.Adam(second.parameters(), lr=.001)
        features = encode(states[:, :-1][mask], actions[mask])
        targets = state_delta(states[:, :-1][mask], states[:, 1:][mask])
        expected = training_update(first, first_optimizer, self.normalizer, features, targets, 1.)
        actual = sequence_update(second, second_optimizer, self.normalizer, states, actions, mask,
                                 rollout_weight=0., clip_norm=1.)
        torch.testing.assert_close(first.a, second.a, rtol=0, atol=0)
        optimizer_equal(self, first_optimizer, second_optimizer)
        self.assertEqual(actual["loss_before_update"], expected["loss_before_update"])
        self.assertEqual(actual["gradient_norm_before_clip"], expected["gradient_norm_before_clip"])
        self.assertEqual(actual["window_count"], 2)
        self.assertEqual(actual["valid_transitions"], 4)
        self.assertEqual(actual["scalar_terms"], 40)
        self.assertIsNone(actual["rollout_loss"])

    def test_sk_update_reports_additional_term_and_changes_parameter(self):
        states, actions, mask = toy_batch()
        model = RelativeToy()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        expected = sequence_loss(RelativeToy(), self.normalizer, states, actions, mask, rollout_weight=.1)
        record = sequence_update(model, optimizer, self.normalizer, states, actions, mask,
                                 rollout_weight=.1, clip_norm=1.)
        self.assertEqual(record["loss_before_update"], float(expected.total.detach()))
        self.assertEqual(record["one_step_loss"], float(expected.one_step.detach()))
        self.assertEqual(record["rollout_loss"], float(expected.rollout.detach()))
        self.assertLess(float(model.a.detach()), .25)
        self.assertTrue(model.training)

    def test_invalid_mask_shapes_types_holes_and_all_empty_are_rejected(self):
        states, actions, mask = toy_batch()
        bad_masks = [mask.float(), mask.numpy(), mask[:, :2], mask[0],
                     torch.tensor([[True, False, True]]), torch.zeros_like(mask)]
        for bad in bad_masks:
            with self.subTest(mask=repr(bad)), self.assertRaises(TrainingError):
                sequence_loss(RelativeToy(), self.normalizer, states, actions, bad, rollout_weight=.1)

    def test_invalid_valid_tensors_and_nonfinite_initial_state_are_rejected(self):
        states, actions, mask = toy_batch()
        bad_states = states.clone(); bad_states[0, 1, 0] = float("nan")
        bad_initial = states.clone(); bad_initial[0, 0, 0] = float("nan")
        bad_actions = actions.clone(); bad_actions[0, 0, 0] = float("nan")
        cases = [(states.float(), actions, mask), (states, actions.float(), mask),
                 (states.numpy(), actions, mask), (states[:, :-1], actions, mask),
                 (states[:, :, :9], actions, mask), (states, actions[:, :, :1], mask),
                 (states[:0], actions[:0], mask[:0]), (bad_states, actions, mask),
                 (bad_initial, actions, mask), (states, bad_actions, mask)]
        for state, action, valid in cases:
            with self.subTest(state_shape=getattr(state, "shape", None)), self.assertRaises(TrainingError):
                sequence_loss(RelativeToy(), self.normalizer, state, action, valid, rollout_weight=.1)
        empty_states, empty_actions, empty_mask = toy_batch((0,), horizon=0)
        with self.assertRaises(TrainingError):
            sequence_loss(RelativeToy(), self.normalizer, empty_states, empty_actions, empty_mask,
                          rollout_weight=.1)
        states, actions, mask = toy_batch((1, 0), horizon=1)
        states[1, 0, 0] = float("nan")
        with self.assertRaises(TrainingError):
            sequence_loss(RelativeToy(), self.normalizer, states, actions, mask, rollout_weight=.1)

    def test_lambda_and_clip_guards_reject_nonfinite_negative_bool_and_text(self):
        states, actions, mask = toy_batch()
        for weight in [-1., float("nan"), float("inf"), True, "0.1"]:
            with self.subTest(weight=weight), self.assertRaises(TrainingError):
                sequence_loss(RelativeToy(), self.normalizer, states, actions, mask, rollout_weight=weight)
        for clip in [0., -1., float("nan"), float("inf"), True, "1"]:
            model = RelativeToy(); optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            with self.subTest(clip=clip), self.assertRaises(TrainingError):
                sequence_update(model, optimizer, self.normalizer, states, actions, mask,
                                rollout_weight=.1, clip_norm=clip)

    def test_nonfinite_gradient_is_rejected_before_optimizer_parameter_mutation(self):
        states, actions, mask = toy_batch()
        model = RelativeToy(); optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        parameter_before = model.a.detach().clone()
        optimizer_before = deepcopy(optimizer.state_dict())
        handle = model.a.register_hook(lambda grad: torch.full_like(grad, float("nan")))
        try:
            with self.assertRaises(TrainingError):
                sequence_update(model, optimizer, self.normalizer, states, actions, mask,
                                rollout_weight=.1, clip_norm=1.)
        finally:
            handle.remove()
        torch.testing.assert_close(model.a.detach(), parameter_before, rtol=0, atol=0)
        self.assertEqual(optimizer.state_dict(), optimizer_before)

    def test_nonfinite_loss_and_invalid_raw_yaw_are_rejected_before_update(self):
        states, actions, mask = toy_batch()
        model = ConstantToy(); optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        with torch.no_grad():
            model.delta[0] = float("nan")
        before = model.delta.detach().clone()
        with self.assertRaises(TrainingError):
            sequence_update(model, optimizer, self.normalizer, states, actions, mask,
                            rollout_weight=.1, clip_norm=1.)
        torch.testing.assert_close(model.delta.detach(), before, equal_nan=True, rtol=0, atol=0)
        self.assertEqual(optimizer.state_dict()["state"], {})
        invalid_yaw = np.zeros(10); invalid_yaw[2] = math.pi+1e-3
        with self.assertRaises(TrainingError):
            sequence_loss(ConstantToy(invalid_yaw), self.normalizer, states, actions, mask,
                          rollout_weight=.1)

    def test_autocast_and_float64_model_are_rejected_without_changing_mode(self):
        states, actions, mask = toy_batch()
        with torch.autocast("cpu", dtype=torch.bfloat16), self.assertRaises(TrainingError):
            sequence_loss(RelativeToy(), self.normalizer, states, actions, mask, rollout_weight=.1)
        with self.assertRaises(TrainingError):
            sequence_loss(ConstantToy(dtype=torch.float64), self.normalizer, states, actions, mask,
                          rollout_weight=.1)
        model = RelativeToy().eval()
        sequence_loss(model, self.normalizer, states, actions, mask, rollout_weight=.1)
        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
