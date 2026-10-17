"""Synthetic first-moment routing checks; no project checkpoint is loaded."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from push_align.cross_batch_probe import array_sha256
from push_align.finite_step_probe import finite_step_probe
from push_align.first_moment_probe import FirstMomentProbeError, first_moment_probe
from test_p4_cross_batch_probe import route_arguments
from test_p4_finite_step_probe import PureConstant


def arguments(mode="KEEP", **kwargs):
    args, bindings = route_arguments(**kwargs)
    bindings["source_binding"].update(mode=mode, direction_index={"KEEP": 3, "ZERO_M": 4}[mode])
    return args, bindings


class FirstMomentRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def reject(self, args, kwargs, message=None):
        with patch("push_align.first_moment_probe.finite_step_probe", side_effect=AssertionError("No evaluation")) as evaluator:
            with self.assertRaisesRegex(FirstMomentProbeError, message or "."):
                first_moment_probe(*args, **kwargs)
        evaluator.assert_not_called()

    def test_both_modes_match_unchanged_finite_helper_arrays_and_summary_exactly(self):
        for mode, index in (("KEEP", 3), ("ZERO_M", 4)):
            args, kwargs = arguments(mode)
            expected = finite_step_probe(*args, raw_total_gradient=kwargs["target_stop_gradient"],
                                         reference_losses=kwargs["target_stop_reference"], parameter_order=kwargs["parameter_order"])
            actual = first_moment_probe(*args, **kwargs)
            with self.subTest(mode=mode):
                self.assertEqual(set(actual), set(expected))
                for key in set(actual) - {"summary"}:
                    np.testing.assert_array_equal(actual[key], expected[key])
                summary = deepcopy(actual["summary"])
                routing = summary.pop("first_moment_routing")
                self.assertEqual(summary, expected["summary"])
                self.assertEqual(routing["source_mode"], mode)
                self.assertEqual(routing["source_direction"]["direction_index"], index)
                self.assertFalse(routing["source_direction"]["norm_normalized"])
                self.assertTrue(all(routing["routing_checks"].values()))
                json.dumps(actual["summary"], allow_nan=False)

    def test_only_r1_final_snapshot_is_accepted_in_every_binding(self):
        for key in ("source_binding", "target_binding", "target_gradient_binding", "target_loss_binding"):
            args, kwargs = arguments()
            kwargs[key]["snapshot_id"] = "R1_u0500"
            with self.subTest(binding=key):
                self.reject(args, kwargs, "same R1_u1000")
        args, kwargs = arguments()
        for key in ("source_binding", "target_binding", "target_gradient_binding", "target_loss_binding"):
            kwargs[key]["snapshot_id"] = "C0_u1000"
        self.reject(args, kwargs, "same R1_u1000")

    def test_source_target_gradient_and_loss_names_are_fixed(self):
        for key, name in (("source_binding", "D11_zero5"), ("target_binding", "D11_noncontact"),
                          ("target_gradient_binding", "D11_noncontact"), ("target_loss_binding", "D11_noncontact")):
            args, kwargs = arguments()
            kwargs[key]["probe_name"] = name
            with self.subTest(binding=key):
                self.reject(args, kwargs, "Only D11_noncontact source")

    def test_only_total_keep3_or_zero_m4_exact_indices_are_accepted(self):
        for mode, index in (("COLD", 5), ("ZERO_M", 3), ("KEEP", 4), ("KEEP", True), ("KEEP", 3.)):
            args, kwargs = arguments()
            kwargs["source_binding"].update(mode=mode, direction_index=index)
            with self.subTest(mode=mode, index=index):
                self.reject(args, kwargs, "Only saved TOTAL")
        args, kwargs = arguments()
        kwargs["source_binding"]["objective"] = "base"
        self.reject(args, kwargs, "Only saved TOTAL")

    def test_source_typed_hash_rejects_direction_substitution_and_normalization(self):
        args, kwargs = arguments("ZERO_M")
        normalized = (args[5] / np.linalg.norm(args[5])).astype(np.float32)
        self.reject((*args[:5], normalized), kwargs, "normalization or substitution")
        args, kwargs = arguments()
        kwargs["source_binding"]["delta_sha256"] = "0" * 64
        self.reject(args, kwargs, "Source direction typed hash")

    def test_target_raw_gradient_cannot_be_replaced_by_source_gradient(self):
        args, kwargs = arguments()
        kwargs["target_stop_gradient"] = (-2. * kwargs["target_stop_gradient"]).astype(np.float32)
        self.reject(args, kwargs, "Target gradient typed hash")
        args, kwargs = arguments()
        kwargs["target_gradient_binding"]["component"] = "one_step"
        self.reject(args, kwargs, "raw total component")

    def test_target_reference_raw5_values_cannot_be_substituted(self):
        args, kwargs = arguments()
        kwargs["target_stop_reference"] = {**kwargs["target_stop_reference"], "total": 123.}
        self.reject(args, kwargs, "bound target-stop raw5")

    def test_same_label_wrong_actual_model_or_parameter_map_is_rejected(self):
        args, kwargs = arguments()
        with torch.no_grad():
            args[0].delta.add_(.01)
        self.reject(args, kwargs, "actual frozen model")
        args, kwargs = arguments()
        kwargs["target_gradient_binding"]["parameter_order"][0]["name"] = "wrong"
        self.reject(args, kwargs, "parameter orders")

    def test_bound_target_arrays_and_window_count_are_enforced(self):
        for field, value in (("batch_windows", 24), ("states_sha256", "0" * 64),
                             ("actions_sha256", "0" * 64), ("mask_sha256", "0" * 64)):
            args, kwargs = arguments()
            kwargs["target_binding"][field] = value
            with self.subTest(field=field):
                self.reject(args, kwargs)

    def test_no_extra_calls_and_target_gradient_source_delta_passed_verbatim(self):
        for mode in ("KEEP", "ZERO_M"):
            args, kwargs = arguments(mode)
            calls = []
            handle = args[0].register_forward_pre_hook(lambda module, inputs: calls.append(len(inputs[0])))
            try:
                with patch("push_align.first_moment_probe.finite_step_probe", wraps=finite_step_probe) as evaluator:
                    result = first_moment_probe(*args, **kwargs)
            finally:
                handle.remove()
            self.assertEqual(evaluator.call_count, 1)
            positional, keyword = evaluator.call_args
            for index in (2, 3, 4, 5):
                self.assertIs(positional[index], args[index])
            self.assertIs(keyword["raw_total_gradient"], kwargs["target_stop_gradient"])
            self.assertIs(keyword["reference_losses"], kwargs["target_stop_reference"])
            self.assertEqual(calls, [10, 2, 2, 2, 2, 2] * 4)
            self.assertEqual(result["summary"]["counters"]["root_model_forward_calls"], 24)

    def test_alphas_are_independent_and_original_model_grad_rng_inputs_bindings_unchanged(self):
        args, kwargs = arguments("ZERO_M")
        model = args[0]
        model.delta.grad = torch.ones_like(model.delta)
        original = model.delta.detach().numpy().copy()
        before = deepcopy(kwargs)
        state = deepcopy(model.state_dict())
        delta, gradient = args[5].copy(), kwargs["target_stop_gradient"].copy()
        inputs = [value.clone() for value in args[2:5]]
        parameter_id, gradient_id = id(model.delta), id(model.delta.grad)
        rng = torch.get_rng_state().clone()
        with patch("torch.autograd.grad", side_effect=AssertionError("No derivative")), \
                patch.object(torch.Tensor, "backward", side_effect=AssertionError("No backward")), \
                patch("torch.optim.Adam", side_effect=AssertionError("No optimizer")):
            result = first_moment_probe(*args, **kwargs)
        self.assertEqual((id(model.delta), id(model.delta.grad)), (parameter_id, gradient_id))
        torch.testing.assert_close(model.delta.grad, torch.ones_like(model.delta), atol=0, rtol=0)
        self.assertFalse(model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in state.items():
            torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
        for value, expected in zip(args[2:5], inputs):
            torch.testing.assert_close(value, expected, atol=0, rtol=0)
        np.testing.assert_array_equal(args[5], delta)
        np.testing.assert_array_equal(kwargs["target_stop_gradient"], gradient)
        for key in set(kwargs) - {"target_stop_gradient"}:
            self.assertEqual(kwargs[key], before[key])
        for index, alpha in enumerate((0., .25, .5, 1.)):
            expected = np.add(original, np.multiply(delta, np.float32(alpha), dtype=np.float32), dtype=np.float32)
            np.testing.assert_array_equal(result["evaluated_parameters"][index], expected)
        self.assertEqual(array_sha256(args[5]), kwargs["source_binding"]["delta_sha256"])

    def test_failed_rows_retain_all_four_alpha_denominators_and_original_arrays(self):
        delta = np.zeros(10, np.float32)
        delta[2] = 10.
        args, kwargs = arguments("ZERO_M", model=PureConstant(), direction=delta)
        result = first_moment_probe(*args, **kwargs)
        summary = result["summary"]
        self.assertEqual(summary["status"], "NUMERICAL_FAILURE")
        self.assertEqual(summary["retained_alpha_count"], 4)
        self.assertEqual(summary["case_denominator"], 1)
        self.assertEqual([row["status"] for row in summary["evaluations"]],
                         ["PASS", "PASS", "NUMERICAL_FAILURE", "NUMERICAL_FAILURE"])
        for row in summary["evaluations"][2:]:
            self.assertTrue(all(value is None for value in row["loss_values"].values()))
            self.assertEqual(row["denominators"], summary["denominators"])
        self.assertTrue(np.isnan(result["free_predicted_states"][2:]).all())
        self.assertEqual(summary["first_moment_routing"]["secondary_finite_pattern_decision"], "NUMERICAL_FAILURE")
        json.dumps(summary, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
