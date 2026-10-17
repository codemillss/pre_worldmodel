"""Synthetic-only finite-step checks; no checkpoint or stored probe is read."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from push_align.finite_step_probe import (
    ALPHAS, FiniteStepProbeError, classify_total_change, finite_step_probe,
)
from test_p4_relative_loss_training import batch, synthetic_normalizer


class PureConstant(nn.Module):
    def __init__(self, value=None):
        super().__init__()
        self.delta = nn.Parameter(torch.tensor(np.zeros(10) if value is None else value, dtype=torch.float32))

    def forward(self, features):
        return self.delta.expand(len(features), 10)


class BufferedConstant(PureConstant):
    def __init__(self, value=None):
        super().__init__(value)
        self.register_buffer("scale", torch.ones(10, dtype=torch.float32))

    def forward(self, features):
        return (self.delta * self.scale).expand(len(features), 10)


class RandomZeroConstant(PureConstant):
    def forward(self, features):
        return (self.delta + torch.rand(()) * 0.).expand(len(features), 10)


class NonfiniteConstant(PureConstant):
    def forward(self, features):
        return (self.delta * float("nan")).expand(len(features), 10)


class MutatingDisposableConstant(PureConstant):
    def forward(self, features):
        self.delta.add_(1.)
        return self.delta.expand(len(features), 10)


def analytic_losses(delta):
    """Independent constant-delta K5 formula, unit delta scales and zero truth."""
    delta = np.asarray(delta, np.float64)
    squared = float(delta @ delta)
    relative = delta[6:8] - delta[:2]
    relative_squared = float(relative @ relative)
    one, rollout = squared / 10., squared * 11. / 10.
    teacher, free = relative_squared / 4., relative_squared * 11. / 4.
    return {"one_step": one, "rollout": rollout, "relative_teacher": teacher,
            "relative_rollout": free, "total": one + .1 * rollout + .2 * (teacher + .1 * free)}


def analytic_gradient(delta):
    value = np.asarray(delta, np.float64)
    gradient = .42 * value
    relative = value[6:8] - value[:2]
    gradient[:2] -= .21 * relative
    gradient[6:8] += .21 * relative
    return gradient.astype(np.float32)


def arguments(model=None, direction=None, windows=2):
    if model is None:
        delta = np.zeros(10, np.float32)
        delta[[0, 6]] = [1., -2.]
        model = PureConstant(delta)
    model.eval()
    original = model.delta.detach().numpy().copy()
    states, actions, mask = batch((5,) * windows, 5)
    delta = (-3. * original).astype(np.float32) if direction is None else np.asarray(direction, np.float32)
    return (model, synthetic_normalizer(), states, actions, mask, delta), {
        "raw_total_gradient": analytic_gradient(original), "reference_losses": analytic_losses(original)}


class FiniteStepProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_analytic_quadratic_retains_all_four_alphas_and_detects_scoped_overshoot(self):
        args, kwargs = arguments()
        result = finite_step_probe(*args, **kwargs)
        summary = result["summary"]
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["decision"], "NOMINAL_INCREASE_SMALLER_DECREASE")
        self.assertEqual([row["total_change_relation"] for row in summary["evaluations"]],
                         ["NUMERIC_TIE", "DECREASE", "DECREASE", "INCREASE"])
        self.assertEqual(result["alphas"].dtype, np.float64)
        np.testing.assert_array_equal(result["alphas"], ALPHAS)
        for index, row in enumerate(summary["evaluations"]):
            expected = analytic_losses(result["evaluated_parameters"][index])
            self.assertEqual(row["status"], "PASS")
            for name, value in expected.items():
                self.assertAlmostEqual(row["loss_values"][name], value, places=12)
            self.assertAlmostEqual(row["loss_values"]["base"], expected["one_step"] + .1 * expected["rollout"], places=12)
            self.assertAlmostEqual(row["loss_values"]["weighted_aux"], .2 * (expected["relative_teacher"] + .1 * expected["relative_rollout"]), places=12)
        json.dumps(summary, allow_nan=False)

    def test_parameter_maps_are_independent_float32_multiply_then_add_not_chained(self):
        args, kwargs = arguments()
        original = args[0].delta.detach().numpy().copy()
        delta = args[5].copy()
        result = finite_step_probe(*args, **kwargs)
        self.assertEqual(result["evaluated_parameters"].dtype, np.float32)
        self.assertEqual(result["rounded_parameter_differences"].dtype, np.float32)
        for index, alpha in enumerate(ALPHAS):
            expected = np.add(original, np.multiply(delta, np.float32(alpha), dtype=np.float32), dtype=np.float32)
            np.testing.assert_array_equal(result["evaluated_parameters"][index], expected)
            np.testing.assert_array_equal(result["rounded_parameter_differences"][index],
                                          np.subtract(expected, original, dtype=np.float32))
        np.testing.assert_array_equal(result["evaluated_parameters"][0], original)

    def test_saved_teacher_and_free_predictions_match_independent_physical_rollout(self):
        args, kwargs = arguments(windows=3)
        result = finite_step_probe(*args, **kwargs)
        self.assertEqual(result["teacher_predictions"].shape, (4, 15, 10))
        self.assertEqual(result["teacher_predictions"].dtype, np.float32)
        self.assertEqual(result["free_predicted_states"].shape, (4, 3, 6, 10))
        self.assertEqual(result["free_predicted_states"].dtype, np.float64)
        for index in range(4):
            delta = result["evaluated_parameters"][index]
            np.testing.assert_array_equal(result["teacher_predictions"][index], np.tile(delta, (15, 1)))
            expected = args[2][:, :1].numpy() + np.arange(6, dtype=np.float64)[None, :, None] * delta.astype(np.float64)[None, None, :]
            np.testing.assert_array_equal(result["free_predicted_states"][index], expected)

    def test_source_root_pre_hook_sees_candidate_once_each_and_proxy_capture_does_not_double_calls(self):
        args, kwargs = arguments(windows=2)
        model = args[0]
        actual_parameters, rows = [], []
        def inspect(module, inputs):
            rows.append(len(inputs[0]))
            actual_parameters.append(torch.cat([parameter.detach().reshape(-1) for parameter in module.parameters()]).numpy().copy())
        handle = model.register_forward_pre_hook(inspect)
        try:
            result = finite_step_probe(*args, **kwargs)
            self.assertEqual(len(rows), 24)
            self.assertEqual(rows, [10, 2, 2, 2, 2, 2] * 4)
            for index, actual in enumerate(actual_parameters):
                np.testing.assert_array_equal(actual, result["evaluated_parameters"][index // 6])
            for record in result["summary"]["evaluations"]:
                self.assertEqual(record["forward_calls"], 6)
                self.assertEqual(record["forward_rows"], [10, 2, 2, 2, 2, 2])
                self.assertEqual(record["forward_total_rows"], 20)
                self.assertEqual(record["denominators"], {"batch_windows": 2, "valid_transitions": 10,
                                                          "base_scalar_terms": 100, "relative_scalar_terms": 20})
            self.assertIn(handle.id, model._forward_pre_hooks)
            self.assertEqual(len(model._forward_hooks), 0)
        finally:
            handle.remove()

    def test_no_new_gradient_or_optimizer_calls_and_all_original_inputs_preserved(self):
        value = np.zeros(10, np.float32)
        value[[0, 6]] = [.25, -.5]
        args, kwargs = arguments(BufferedConstant(value))
        model, normalizer, states, actions, mask, delta = args
        model.delta.grad = torch.arange(10, dtype=torch.float32)
        states.requires_grad_()
        states.grad = torch.ones_like(states)
        before = deepcopy(model.state_dict())
        parameter_id, buffer_id, grad_id = id(model.delta), id(model.scale), id(model.delta.grad)
        input_grad_id, modes = id(states.grad), [module.training for module in model.modules()]
        rng = torch.get_rng_state().clone()
        delta_before, gradient_before = delta.copy(), kwargs["raw_total_gradient"].copy()
        metadata_before = deepcopy(normalizer.metadata)
        with patch("torch.autograd.grad", side_effect=AssertionError("No derivatives")), \
                patch.object(torch.Tensor, "backward", side_effect=AssertionError("No backward")), \
                patch("torch.optim.Adam", side_effect=AssertionError("No optimizer")):
            result = finite_step_probe(*args, **kwargs)
        for key, expected in before.items():
            torch.testing.assert_close(model.state_dict()[key], expected, atol=0, rtol=0)
        self.assertEqual((id(model.delta), id(model.scale), id(model.delta.grad)), (parameter_id, buffer_id, grad_id))
        self.assertEqual(id(states.grad), input_grad_id)
        torch.testing.assert_close(states.grad, torch.ones_like(states), atol=0, rtol=0)
        torch.testing.assert_close(model.delta.grad, torch.arange(10, dtype=torch.float32), atol=0, rtol=0)
        self.assertEqual([module.training for module in model.modules()], modes)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(normalizer.metadata, metadata_before)
        np.testing.assert_array_equal(delta, delta_before)
        np.testing.assert_array_equal(kwargs["raw_total_gradient"], gradient_before)
        self.assertTrue(all(result["summary"]["checks"].values()))
        self.assertEqual(result["summary"]["source_state_hash_before"], result["summary"]["source_state_hash_after"])
        counters = result["summary"]["counters"]
        self.assertEqual(counters["successful_evaluations"], 4)
        self.assertEqual(counters["root_model_forward_calls"], 24)
        self.assertEqual(counters["root_model_forward_rows"], 80)
        self.assertTrue(all(counters[key] == 0 for key in
                            ("new_autograd_traversals", "new_optimizer_steps", "new_training_updates", "new_physics_steps")))

    def test_same_source_rng_is_restored_for_each_independent_alpha(self):
        args, kwargs = arguments(RandomZeroConstant(np.zeros(10)), direction=np.zeros(10))
        before = torch.get_rng_state().clone()
        result = finite_step_probe(*args, **kwargs)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for index in range(1, 4):
            np.testing.assert_array_equal(result["teacher_predictions"][index], result["teacher_predictions"][0])

    def test_primary_float32_subtraction_and_true_endpoint_sensitivity_remain_separate(self):
        original = np.zeros(10, np.float32)
        original[0] = 1e-8
        direction = np.zeros(10, np.float32)
        direction[0] = 1.
        args, kwargs = arguments(PureConstant(original), direction)
        kwargs["raw_total_gradient"][:] = 1.
        result = finite_step_probe(*args, **kwargs)
        for index, row in enumerate(result["summary"]["evaluations"]):
            primary = float(kwargs["raw_total_gradient"].astype(np.float64) @ result["rounded_parameter_differences"][index].astype(np.float64))
            endpoint = float(kwargs["raw_total_gradient"].astype(np.float64) @
                             (result["evaluated_parameters"][index].astype(np.float64) - original.astype(np.float64)))
            self.assertEqual(row["taylor"]["rounded_directional_change"], primary)
            self.assertEqual(row["taylor"]["exact_endpoint_directional_change"], endpoint)
        self.assertNotEqual(result["summary"]["evaluations"][3]["taylor"]["rounded_directional_change"],
                            result["summary"]["evaluations"][3]["taylor"]["exact_endpoint_directional_change"])

    def test_alpha0_reference_failure_keeps_four_evaluations_without_interpretation(self):
        args, kwargs = arguments()
        kwargs["reference_losses"]["one_step"] += .01
        result = finite_step_probe(*args, **kwargs)
        summary = result["summary"]
        self.assertEqual(summary["status"], "REPRODUCTION_FAILURE")
        self.assertEqual(summary["decision"], "REPRODUCTION_FAILURE")
        self.assertEqual(summary["alpha0_reproduction"]["status"], "REPRODUCTION_FAILURE")
        self.assertEqual(summary["retained_alpha_count"], 4)
        self.assertEqual(summary["counters"]["successful_evaluations"], 4)
        self.assertTrue(all(row["total_change_relation"] == "NOT_INTERPRETED" for row in summary["evaluations"]))

    def test_reproduction_uses_registered_one_sided_tolerance_on_all_five_losses(self):
        args, kwargs = arguments()
        result = finite_step_probe(*args, **kwargs)
        reproduction = result["summary"]["alpha0_reproduction"]
        self.assertEqual((reproduction["rtol"], reproduction["atol"]), (1e-10, 1e-12))
        self.assertEqual(set(reproduction["per_loss"]), {"one_step", "rollout", "relative_teacher", "relative_rollout", "total"})
        for name, row in reproduction["per_loss"].items():
            self.assertEqual(row["allowed_error"], 1e-12 + 1e-10 * abs(kwargs["reference_losses"][name]))
            self.assertTrue(row["pass"])

    def test_finite_numerical_failures_preserve_all_rows_and_prediction_placeholders(self):
        direction = np.zeros(10, np.float32)
        direction[2] = 10.
        args, kwargs = arguments(PureConstant(), direction)
        result = finite_step_probe(*args, **kwargs)
        rows, summary = result["summary"]["evaluations"], result["summary"]
        self.assertEqual(summary["status"], "NUMERICAL_FAILURE")
        self.assertEqual(summary["decision"], "NUMERICAL_FAILURE")
        self.assertEqual([row["status"] for row in rows], ["PASS", "PASS", "NUMERICAL_FAILURE", "NUMERICAL_FAILURE"])
        self.assertEqual(summary["retained_alpha_count"], 4)
        self.assertEqual(summary["case_denominator"], 1)
        self.assertTrue(np.isfinite(result["evaluated_parameters"]).all())
        self.assertTrue(np.isfinite(result["teacher_predictions"]).all())
        self.assertTrue(np.isnan(result["free_predicted_states"][2:]).all())
        for row in rows[2:]:
            self.assertTrue(all(value is None for value in row["loss_values"].values()))
            self.assertEqual(row["denominators"], summary["denominators"])
            self.assertEqual(row["forward_rows"], [10, 2])
        json.dumps(summary, allow_nan=False)

    def test_numerical_failure_has_priority_over_separate_alpha0_reproduction_failure(self):
        direction = np.zeros(10, np.float32)
        direction[2] = 10.
        args, kwargs = arguments(PureConstant(), direction)
        kwargs["reference_losses"]["one_step"] = .01
        summary = finite_step_probe(*args, **kwargs)["summary"]
        self.assertEqual(summary["alpha0_reproduction"]["status"], "REPRODUCTION_FAILURE")
        self.assertEqual(summary["status"], "NUMERICAL_FAILURE")
        self.assertEqual(summary["decision"], "NUMERICAL_FAILURE")

    def test_nonfinite_teacher_output_retains_failed_alpha0_and_other_endpoints(self):
        args, kwargs = arguments(NonfiniteConstant(), np.zeros(10))
        result = finite_step_probe(*args, **kwargs)
        summary = result["summary"]
        self.assertEqual(summary["status"], "NUMERICAL_FAILURE")
        self.assertEqual(summary["alpha0_reproduction"]["status"], "NUMERICAL_FAILURE")
        self.assertEqual(summary["counters"]["successful_evaluations"], 0)
        self.assertEqual(summary["counters"]["root_model_forward_calls"], 4)
        self.assertEqual(len(summary["evaluations"]), 4)
        self.assertTrue(np.isnan(result["teacher_predictions"]).all())
        self.assertTrue(np.isnan(result["free_predicted_states"]).all())
        json.dumps(summary, allow_nan=False)

    def test_zero_direction_is_numeric_tie_and_not_claimed_overshoot_or_improvement(self):
        args, kwargs = arguments(direction=np.zeros(10, np.float32))
        result = finite_step_probe(*args, **kwargs)
        self.assertEqual(result["summary"]["decision"], "INCONCLUSIVE")
        self.assertTrue(all(row["total_change_relation"] == "NUMERIC_TIE" for row in result["summary"]["evaluations"]))
        self.assertEqual(result["summary"]["neutral_band"], 1e-6 * max(1., abs(result["summary"]["evaluations"][0]["loss_values"]["common_total"])))
        np.testing.assert_array_equal(result["rounded_parameter_differences"], np.zeros((4, 10), np.float32))

    def test_all_positive_alphas_decrease_only_supports_no_overshoot_on_this_grid(self):
        args, kwargs = arguments()
        args = (*args[:5], (-.5 * args[0].delta.detach().numpy()).astype(np.float32))
        summary = finite_step_probe(*args, **kwargs)["summary"]
        self.assertEqual(summary["decision"], "NO_ALPHA1_OVERSHOOT_OBSERVED_ON_GRID")
        self.assertTrue(all(row["total_change_relation"] == "DECREASE" for row in summary["evaluations"][1:]))

    def test_numeric_tie_boundary_and_signs_are_explicit(self):
        for value in (-1e-6, 0., 1e-6):
            self.assertEqual(classify_total_change(value, 1e-6), "NUMERIC_TIE")
        self.assertEqual(classify_total_change(-1.0001e-6, 1e-6), "DECREASE")
        self.assertEqual(classify_total_change(1.0001e-6, 1e-6), "INCREASE")
        with self.assertRaises(FiniteStepProbeError):
            classify_total_change(float("nan"), 1e-6)

    def test_contract_rejections_happen_before_any_forward(self):
        args, kwargs = arguments()
        rows = []
        handle = args[0].register_forward_pre_hook(lambda _m, inputs: rows.append(len(inputs[0])))
        try:
            broken_mask = args[4].clone()
            broken_mask[0, -1] = False
            variants = [(args[:4] + (broken_mask, args[5]), kwargs),
                        (args[:5] + (args[5].astype(np.float64),), kwargs),
                        (args, {**kwargs, "raw_total_gradient": np.zeros(9, np.float32)}),
                        (args, {**kwargs, "reference_losses": {"total": 0.}}),
                        (args, {**kwargs, "parameter_order": []})]
            for positional, keyword in variants:
                with self.assertRaises(FiniteStepProbeError):
                    finite_step_probe(*positional, **keyword)
            args[0].train()
            with self.assertRaisesRegex(FiniteStepProbeError, "eval mode"):
                finite_step_probe(*args, **kwargs)
            self.assertEqual(rows, [])
        finally:
            handle.remove()

    def test_disposable_map_mutation_is_a_contract_failure_but_source_is_preserved(self):
        args, kwargs = arguments(MutatingDisposableConstant(), np.zeros(10))
        original = args[0].delta.detach().clone()
        with self.assertRaisesRegex(FiniteStepProbeError, "mutated a disposable"):
            finite_step_probe(*args, **kwargs)
        torch.testing.assert_close(args[0].delta, original, atol=0, rtol=0)
        self.assertFalse(args[0].training)
        self.assertEqual(len(args[0]._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
