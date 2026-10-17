"""Known delta, recurrent graph, and explicit model-boundary checks."""
from copy import deepcopy
import math
import unittest

import numpy as np
import torch
from torch import nn

from push_align import PlanarPushEnv
from push_align.config import load_pilot_config
from push_align.data import contract_hash, make_contract
from push_align.dynamics import (DeltaMLP, DynamicModelError, free_rollout,
                                 model_spec, predict_delta, predict_next)
from push_align.normalization import fit_normalizer, learning_schema
from push_align.representation import encode


class ControlledModel(nn.Module):
    """A known linear normalized mapping, independent of MLP internals."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((10, 11), dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(10, dtype=torch.float32))

    def forward(self, features):
        return features @ self.weight.T + self.bias


class P2DynamicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_pilot_config()
        contract = make_contract(PlanarPushEnv())
        schema = learning_schema(contract)
        identity = {"dataset_id": "known-dynamics-fixture", "manifest_sha256": "a" * 64,
                    "data_contract_hash": contract_hash(contract)}
        states = np.zeros((2, 10), dtype=np.float64)
        states[:, 6] = [-.001, .001]
        features = encode(states, np.zeros((2, 2), dtype=np.float64))
        cls.normalizer = fit_normalizer(
            features, np.zeros((2, 10), dtype=np.float64), np.zeros(2, dtype=np.int8),
            schema, identity, np.zeros(2, dtype=np.int64), np.arange(2, dtype=np.int64))

    def setUp(self):
        self.state = torch.zeros(10, dtype=torch.float64)
        self.state[6] = .01
        self.action = torch.tensor([.02, -.01], dtype=torch.float64)

    def test_exact_architecture_count_and_caller_seed_reproducibility(self):
        torch.manual_seed(11)
        first = DeltaMLP(self.config)
        torch.manual_seed(11)
        second = DeltaMLP(self.config)
        self.assertEqual(model_spec(self.config)["parameter_count"], 19338)
        self.assertEqual(sum(p.numel() for p in first.parameters()), 19338)
        self.assertEqual([type(layer) for layer in first.network],
                         [nn.Linear, nn.SiLU, nn.Linear, nn.SiLU, nn.Linear])
        self.assertEqual([(m.in_features, m.out_features) for m in first.network if isinstance(m, nn.Linear)],
                         [(11, 128), (128, 128), (128, 10)])
        for one, two in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(one, two, rtol=0, atol=0)

    def test_constructor_explicit_float32_under_global_float64_default(self):
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            model = DeltaMLP(self.config)
            self.assertTrue(all(p.dtype == torch.float32 and p.device.type == "cpu" for p in model.parameters()))
            self.assertEqual(torch.get_default_dtype(), torch.float64)
        finally:
            torch.set_default_dtype(original)

    def test_model_config_rejects_semantic_architecture_or_runtime_changes(self):
        cases = [("model", "hidden_widths", [64, 64]), ("model", "activation", "relu"),
                 ("model", "learned_inputs", 10), ("model", "learned_outputs", 10.),
                 ("runtime", "device", "mps"), ("runtime", "model_dtype", "float64")]
        for section, field, value in cases:
            cfg = deepcopy(self.config); cfg[section][field] = value
            with self.subTest(field=field), self.assertRaises(DynamicModelError):
                DeltaMLP(cfg)
        with self.assertRaises(DynamicModelError):
            model_spec(None)

    def test_forward_preserves_arbitrary_leading_shape_and_float32(self):
        model = DeltaMLP(self.config)
        for shape in [(11,), (3, 11), (2, 3, 11)]:
            result = model(torch.zeros(shape, dtype=torch.float32))
            self.assertEqual(result.shape, (*shape[:-1], 10))
            self.assertEqual(result.dtype, torch.float32)

    def test_forward_rejects_wrong_width_dtype_backend_nonfinite_and_weights(self):
        model = DeltaMLP(self.config)
        cases = [np.zeros(11, dtype=np.float32), torch.zeros(10), torch.zeros(11, dtype=torch.float64),
                 torch.full((11,), float("nan")), torch.full((11,), float("inf"))]
        for value in cases:
            with self.subTest(value=type(value).__name__), self.assertRaises(DynamicModelError):
                model(value)
        model.double()
        with self.assertRaises(DynamicModelError):
            model(torch.zeros(11, dtype=torch.float32))

    def test_shared_normalization_produces_known_physical_delta_without_mutation(self):
        model = ControlledModel()
        with torch.no_grad():
            model.weight[0, 0] = 5.  # physical dx = .5 * relative_pusher_x
            model.weight[6, 9] = 1.  # physical dpx = .1 * applied_vx
            model.bias[2] = 30.      # physical dyaw = .03 rad
        before_state, before_action = self.state.clone(), self.action.clone()
        delta = predict_delta(model, self.normalizer, self.state, self.action)
        expected = torch.zeros(10, dtype=torch.float64)
        expected[[0, 2, 6]] = torch.tensor([.005, .03, .002], dtype=torch.float64)
        torch.testing.assert_close(delta, expected, atol=1e-10, rtol=0)
        self.assertEqual(delta.dtype, torch.float64)
        torch.testing.assert_close(self.state, before_state, rtol=0, atol=0)
        torch.testing.assert_close(self.action, before_action, rtol=0, atol=0)

    def test_yaw_crosses_two_pi_boundary_using_physical_delta(self):
        model = ControlledModel()
        with torch.no_grad():
            model.bias[2] = 30.
        self.state[2] = math.pi - .01
        result = predict_next(model, self.normalizer, self.state, self.action)
        self.assertAlmostEqual(float(result[2].detach()), -math.pi + .02, delta=1e-10)

    def test_free_rollout_matches_recursive_closed_form_without_future_truth(self):
        model = ControlledModel()
        with torch.no_grad():
            model.weight[0, 0] = 5.
        states = free_rollout(model, self.normalizer, self.state,
                              self.action.repeat(5, 1))
        expected = .01 * (1 - .5 ** torch.arange(6, dtype=torch.float64))
        torch.testing.assert_close(states[:, 0], expected, atol=1e-10, rtol=0)
        self.assertEqual(states.shape, (6, 10))
        # Reusing the initial state at each step would incorrectly give .005.
        self.assertGreater(float(states[-1, 0].detach()), .009)
        batch = free_rollout(model, self.normalizer, self.state.repeat(2, 3, 1),
                             self.action.repeat(2, 3, 5, 1))
        self.assertEqual(batch.shape, (2, 3, 6, 10))

    def test_gradients_span_all_future_steps_actions_and_parameters(self):
        model = ControlledModel()
        with torch.no_grad():
            model.weight[0, 0] = 5.
            model.weight[6, 9] = 1.
        initial = self.state.clone().requires_grad_()
        actions = self.action.repeat(5, 1).requires_grad_()
        result = free_rollout(model, self.normalizer, initial, actions)
        result[-1, [0, 6]].sum().backward()
        self.assertAlmostEqual(float(initial.grad[0]), .5 ** 5, delta=1e-7)
        self.assertTrue(bool((actions.grad[:, 0].abs() > 0).all()))
        self.assertTrue(bool(torch.isfinite(model.weight.grad).all()))
        self.assertGreater(float(model.weight.grad.abs().sum()), 0.)

    def test_model_parameters_can_update_without_adapter_mode_change(self):
        torch.manual_seed(11)
        model = DeltaMLP(self.config)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        before = next(model.parameters()).detach().clone()
        features = torch.randn((4, 11), dtype=torch.float32)
        loss = (model(features) - torch.ones((4, 10), dtype=torch.float32)).square().mean()
        loss.backward(); optimizer.step()
        self.assertFalse(torch.equal(next(model.parameters()), before))
        model.eval()
        predict_next(model, self.normalizer, self.state, self.action)
        self.assertFalse(model.training)
        model.train()
        free_rollout(model, self.normalizer, self.state, self.action.repeat(2, 1))
        self.assertTrue(model.training)

    def test_physical_api_rejects_numpy_float32_broadcasting_and_bad_horizons(self):
        model = ControlledModel()
        pairs = [(self.state.numpy(), self.action), (self.state.float(), self.action),
                 (self.state, self.action.float()), (self.state.repeat(2, 1), self.action),
                 (torch.full((10,), float("nan"), dtype=torch.float64), self.action)]
        for state, action in pairs:
            with self.subTest(state_type=type(state).__name__), self.assertRaises(DynamicModelError):
                predict_delta(model, self.normalizer, state, action)
        for actions in [self.action, torch.zeros((0, 2), dtype=torch.float64),
                        self.action.repeat(2, 3, 1)]:
            with self.subTest(shape=actions.shape), self.assertRaises(DynamicModelError):
                free_rollout(model, self.normalizer, self.state, actions)
        with self.assertRaises(DynamicModelError):
            predict_delta(model, object(), self.state, self.action)

    def test_prediction_rejects_changed_output_shape_dtype_and_nonfinite(self):
        class BadOutput(nn.Module):
            def __init__(self, value):
                super().__init__(); self.value = value
            def forward(self, features):
                return self.value
        for result in [torch.zeros(9), torch.zeros(10, dtype=torch.float64),
                       torch.zeros((1, 10)), torch.full((10,), float("nan"))]:
            with self.subTest(shape=result.shape), self.assertRaises(DynamicModelError):
                predict_delta(BadOutput(result), self.normalizer, self.state, self.action)

    def test_raw_yaw_limit_is_rejected_and_other_deltas_are_not_clipped(self):
        model = ControlledModel()
        with torch.no_grad():
            model.bias[2] = 4000.  # 4 rad, invalid before wrapping
        delta = predict_delta(model, self.normalizer, self.state, self.action)
        self.assertEqual(float(delta[2].detach()), 4.)
        with self.assertRaises(DynamicModelError):
            predict_next(model, self.normalizer, self.state, self.action)
        with self.assertRaises(DynamicModelError):
            free_rollout(model, self.normalizer, self.state, self.action.repeat(2, 1))
        with torch.no_grad():
            model.bias[2] = 0.; model.bias[0] = 10000.
        result = predict_next(model, self.normalizer, self.state, self.action)
        self.assertEqual(float(result[0].detach()), 1.)  # workspace validity is evaluated separately

    def test_autocast_is_rejected_before_any_implicit_numeric_change(self):
        model = DeltaMLP(self.config)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with self.assertRaises(DynamicModelError):
                model(torch.zeros(11, dtype=torch.float32))
            with self.assertRaises(DynamicModelError):
                predict_delta(model, self.normalizer, self.state, self.action)


if __name__ == "__main__":
    unittest.main()
