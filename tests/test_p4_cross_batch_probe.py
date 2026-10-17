"""Synthetic routing checks; no project checkpoint or recorded probe is read."""
from copy import deepcopy
import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from push_align.cross_batch_probe import CrossBatchProbeError, array_sha256, cross_batch_probe
from push_align.data_intervention_training import tree_hash
from push_align.finite_step_probe import finite_step_probe
from test_p4_finite_step_probe import PureConstant, arguments


def route_arguments(source_probe="D11_noncontact", model=None, direction=None):
    args, finite_kwargs = arguments(model=model, direction=direction, windows=2)
    model, _, states, actions, mask, delta = args
    cursor, order = 0, []
    for name, parameter in model.named_parameters():
        order.append({"name": name, "shape": list(parameter.shape), "numel": parameter.numel(),
                      "start": cursor, "end": cursor + parameter.numel()})
        cursor += parameter.numel()
    common = {"snapshot_id": "R1_u1000", "model_tree_hash": tree_hash(model.state_dict()),
              "parameter_order": order}
    source = {**deepcopy(common), "probe_name": source_probe, "objective": "total", "mode": "KEEP",
              "direction_index": 3, "delta_sha256": array_sha256(delta)}
    target = {**deepcopy(common), "probe_name": "D11_zero5", "batch_windows": len(states),
              "states_sha256": array_sha256(states), "actions_sha256": array_sha256(actions),
              "mask_sha256": array_sha256(mask)}
    target_gradient = {**deepcopy(common), "probe_name": "D11_zero5", "component": "total",
                       "gradient_sha256": array_sha256(finite_kwargs["raw_total_gradient"])}
    target_loss = {**deepcopy(common), "probe_name": "D11_zero5",
                   "loss_values": deepcopy(finite_kwargs["reference_losses"])}
    kwargs = {"source_binding": source, "target_binding": target,
              "target_gradient_binding": target_gradient, "target_loss_binding": target_loss,
              "target_stop_gradient": finite_kwargs["raw_total_gradient"],
              "target_stop_reference": finite_kwargs["reference_losses"], "parameter_order": order}
    return args, kwargs


class CrossBatchRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_typed_hash_has_independent_canonical_definition_and_no_stride_dependency(self):
        value = np.arange(12, dtype=np.float32).reshape(3, 4)
        selected = value[:, ::2]
        descriptor = json.dumps({"dtype": "float32", "shape": [3, 2]}, sort_keys=True,
                                separators=(",", ":")).encode("utf-8")
        expected = hashlib.sha256(descriptor + b"\0" + selected.tobytes(order="C")).hexdigest()
        self.assertEqual(array_sha256(selected), expected)
        self.assertEqual(array_sha256(torch.from_numpy(selected.copy())), expected)
        self.assertNotEqual(array_sha256(selected.astype(np.float64)), expected)
        self.assertNotEqual(array_sha256(selected.reshape(-1)), expected)

    def test_same_snapshot_stop_evaluation_matches_unchanged_helper_exactly(self):
        args, kwargs = route_arguments()
        expected = finite_step_probe(*args, raw_total_gradient=kwargs["target_stop_gradient"],
                                     reference_losses=kwargs["target_stop_reference"], parameter_order=kwargs["parameter_order"])
        actual = cross_batch_probe(*args, **kwargs)
        self.assertEqual(set(actual), set(expected))
        for key in set(actual) - {"summary"}:
            np.testing.assert_array_equal(actual[key], expected[key])
        summary = deepcopy(actual["summary"])
        routing = summary.pop("cross_batch_routing")
        self.assertEqual(summary, expected["summary"])
        self.assertEqual(routing["source_probe_name"], "D11_noncontact")
        self.assertEqual(routing["target_probe_name"], "D11_zero5")
        self.assertFalse(routing["is_self_stop_control"])
        self.assertFalse(routing["source_direction"]["norm_normalized"])
        self.assertEqual(routing["target_gradient"]["gradient_sha256"], array_sha256(kwargs["target_stop_gradient"]))
        self.assertTrue(all(routing["routing_checks"].values()))
        json.dumps(actual["summary"], allow_nan=False)

    def test_self_stop_control_is_explicit_and_preserves_native_same_batch_result(self):
        args, kwargs = route_arguments(source_probe="D11_zero5")
        result = cross_batch_probe(*args, **kwargs)
        routing = result["summary"]["cross_batch_routing"]
        self.assertTrue(routing["is_self_stop_control"])
        self.assertEqual(routing["source_probe_name"], routing["target_probe_name"])
        self.assertEqual(routing["secondary_finite_pattern_decision"], result["summary"]["decision"])
        self.assertEqual(result["summary"]["retained_alpha_count"], 4)
        self.assertEqual(result["summary"]["counters"]["root_model_forward_calls"], 24)

    def test_wrapper_passes_original_source_direction_and_target_gradient_without_normalization(self):
        args, kwargs = route_arguments()
        with patch("push_align.cross_batch_probe.finite_step_probe", wraps=finite_step_probe) as evaluator:
            result = cross_batch_probe(*args, **kwargs)
        self.assertEqual(evaluator.call_count, 1)
        positional, keyword = evaluator.call_args
        self.assertIs(positional[2], args[2])
        self.assertIs(positional[3], args[3])
        self.assertIs(positional[4], args[4])
        self.assertIs(positional[5], args[5])
        self.assertIs(keyword["raw_total_gradient"], kwargs["target_stop_gradient"])
        self.assertIs(keyword["reference_losses"], kwargs["target_stop_reference"])
        original = args[0].delta.detach().numpy()
        for index, alpha in enumerate((0., .25, .5, 1.)):
            expected = np.add(original, np.multiply(args[5], np.float32(alpha), dtype=np.float32), dtype=np.float32)
            np.testing.assert_array_equal(result["evaluated_parameters"][index], expected)

    def test_source_gradient_substitution_is_rejected_by_target_typed_hash_before_forward(self):
        args, kwargs = route_arguments()
        kwargs["target_stop_gradient"] = (-2. * kwargs["target_stop_gradient"]).astype(np.float32)
        with patch("push_align.cross_batch_probe.finite_step_probe", side_effect=AssertionError("No evaluation")):
            with self.assertRaisesRegex(CrossBatchProbeError, "Target gradient typed hash"):
                cross_batch_probe(*args, **kwargs)

    def test_source_gradient_identity_cannot_be_relabeled_as_target_in_one_binding(self):
        args, kwargs = route_arguments()
        kwargs["target_gradient_binding"]["probe_name"] = "D11_noncontact"
        with patch("push_align.cross_batch_probe.finite_step_probe", side_effect=AssertionError("No evaluation")):
            with self.assertRaisesRegex(CrossBatchProbeError, "all be D11_zero5"):
                cross_batch_probe(*args, **kwargs)

    def test_source_reference_loss_substitution_is_rejected_before_forward(self):
        args, kwargs = route_arguments()
        kwargs["target_stop_reference"] = {**kwargs["target_stop_reference"], "total": 123.}
        with patch("push_align.cross_batch_probe.finite_step_probe", side_effect=AssertionError("No evaluation")):
            with self.assertRaisesRegex(CrossBatchProbeError, "bound target-stop raw5"):
                cross_batch_probe(*args, **kwargs)

    def test_source_target_snapshot_mismatch_rejected_before_forward(self):
        for name in ("target_binding", "target_gradient_binding", "target_loss_binding"):
            args, kwargs = route_arguments()
            kwargs[name]["snapshot_id"] = "C0_u1000"
            with self.subTest(binding=name), patch("push_align.cross_batch_probe.finite_step_probe", side_effect=AssertionError("No evaluation")):
                with self.assertRaisesRegex(CrossBatchProbeError, "same snapshot"):
                    cross_batch_probe(*args, **kwargs)

    def test_same_label_but_wrong_actual_model_or_parameter_order_is_rejected(self):
        args, kwargs = route_arguments()
        with torch.no_grad():
            args[0].delta.add_(.01)
        with self.assertRaisesRegex(CrossBatchProbeError, "current same-snapshot model"):
            cross_batch_probe(*args, **kwargs)
        args, kwargs = route_arguments()
        kwargs["target_gradient_binding"]["parameter_order"][0]["name"] = "wrong"
        with self.assertRaisesRegex(CrossBatchProbeError, "parameter orders"):
            cross_batch_probe(*args, **kwargs)

    def test_only_saved_total_keep_native_direction_is_accepted(self):
        for field, value in (("probe_name", "held_out"), ("objective", "base"),
                             ("mode", "COLD"), ("direction_index", 0), ("direction_index", True)):
            args, kwargs = route_arguments()
            kwargs["source_binding"][field] = value
            with self.subTest(field=field), patch("push_align.cross_batch_probe.finite_step_probe", side_effect=AssertionError("No evaluation")):
                with self.assertRaises(CrossBatchProbeError):
                    cross_batch_probe(*args, **kwargs)
        args, kwargs = route_arguments()
        normalized = args[5] / np.linalg.norm(args[5])
        with self.assertRaisesRegex(CrossBatchProbeError, "normalization or substitution"):
            cross_batch_probe(*args[:5], normalized.astype(np.float32), **kwargs)

    def test_target_batch_name_values_and_window_count_must_match_all_bound_inputs(self):
        for field, value in (("probe_name", "D11_transition"), ("batch_windows", 24),
                             ("states_sha256", "0" * 64), ("actions_sha256", "0" * 64),
                             ("mask_sha256", "0" * 64)):
            args, kwargs = route_arguments()
            kwargs["target_binding"][field] = value
            with self.subTest(field=field), patch("push_align.cross_batch_probe.finite_step_probe", side_effect=AssertionError("No evaluation")):
                with self.assertRaises(CrossBatchProbeError):
                    cross_batch_probe(*args, **kwargs)

    def test_target_gradient_component_and_loss_identity_must_be_target_total(self):
        args, kwargs = route_arguments()
        kwargs["target_gradient_binding"]["component"] = "one_step"
        with self.assertRaisesRegex(CrossBatchProbeError, "raw total component"):
            cross_batch_probe(*args, **kwargs)
        args, kwargs = route_arguments()
        kwargs["target_loss_binding"]["probe_name"] = "historical_update1"
        with self.assertRaisesRegex(CrossBatchProbeError, "all be D11_zero5"):
            cross_batch_probe(*args, **kwargs)

    def test_source_weights_grad_rng_native_direction_and_bindings_stay_unchanged(self):
        args, kwargs = route_arguments()
        model = args[0]
        model.delta.grad = torch.ones_like(model.delta)
        before = deepcopy(model.state_dict())
        delta_before = args[5].copy()
        gradient_before = kwargs["target_stop_gradient"].copy()
        bindings_before = deepcopy({key: value for key, value in kwargs.items() if key not in ("target_stop_gradient",)})
        gradient_id, parameter_id = id(model.delta.grad), id(model.delta)
        rng = torch.get_rng_state().clone()
        with patch("torch.autograd.grad", side_effect=AssertionError("No new derivative")), \
                patch.object(torch.Tensor, "backward", side_effect=AssertionError("No backward")), \
                patch("torch.optim.Adam", side_effect=AssertionError("No optimizer")):
            result = cross_batch_probe(*args, **kwargs)
        self.assertEqual(id(model.delta.grad), gradient_id)
        self.assertEqual(id(model.delta), parameter_id)
        self.assertFalse(model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in before.items():
            torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
        np.testing.assert_array_equal(args[5], delta_before)
        np.testing.assert_array_equal(kwargs["target_stop_gradient"], gradient_before)
        self.assertEqual({key: value for key, value in kwargs.items() if key not in ("target_stop_gradient",)}, bindings_before)
        self.assertTrue(all(result["summary"]["checks"].values()))

    def test_failed_endpoints_are_preserved_with_original_denominators_and_null_values(self):
        delta = np.zeros(10, np.float32)
        delta[2] = 10.
        args, kwargs = route_arguments(model=PureConstant(), direction=delta)
        result = cross_batch_probe(*args, **kwargs)
        summary = result["summary"]
        self.assertEqual(summary["status"], "NUMERICAL_FAILURE")
        self.assertEqual(summary["retained_alpha_count"], 4)
        self.assertEqual(summary["case_denominator"], 1)
        self.assertEqual([row["status"] for row in summary["evaluations"]],
                         ["PASS", "PASS", "NUMERICAL_FAILURE", "NUMERICAL_FAILURE"])
        for row in summary["evaluations"][2:]:
            self.assertTrue(all(value is None for value in row["loss_values"].values()))
            self.assertEqual(row["denominators"], summary["denominators"])
        self.assertEqual(summary["cross_batch_routing"]["secondary_finite_pattern_decision"], "NUMERICAL_FAILURE")
        self.assertTrue(np.isnan(result["free_predicted_states"][2:]).all())
        json.dumps(summary, allow_nan=False)

    def test_primary_target_relations_are_retained_and_inconclusive_is_not_no_cross_harm(self):
        args, kwargs = route_arguments(direction=np.zeros(10, np.float32))
        result = cross_batch_probe(*args, **kwargs)
        routing = result["summary"]["cross_batch_routing"]
        self.assertEqual(result["summary"]["decision"], "INCONCLUSIVE")
        self.assertTrue(all(row["total_change_relation"] == "NUMERIC_TIE" for row in result["summary"]["evaluations"]))
        self.assertIn("INCONCLUSIVE does not mean absence", routing["secondary_decision_limit"])
        self.assertIn("Per-alpha", routing["primary_target_effect"])

    def test_all_positive_target_increases_remain_primary_despite_secondary_inconclusive(self):
        direction = np.zeros(10, np.float32)
        direction[[0, 6]] = [.5, -1.]
        args, kwargs = route_arguments(direction=direction)
        result = cross_batch_probe(*args, **kwargs)
        summary = result["summary"]
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["decision"], "INCONCLUSIVE")
        self.assertEqual(summary["cross_batch_routing"]["secondary_finite_pattern_decision"], "INCONCLUSIVE")
        self.assertTrue(all(row["total_change_relation"] == "INCREASE" for row in summary["evaluations"][1:]))
        self.assertTrue(all(row["loss_differences"]["common_total"] > summary["neutral_band"]
                            for row in summary["evaluations"][1:]))


if __name__ == "__main__":
    unittest.main()
