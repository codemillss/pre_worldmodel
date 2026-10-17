"""Synthetic virtual arithmetic and disposable zero-parameter Adam oracles.

No saved scientific model, checkpoint, probe gradient or simulator is used.
The optimizer steps below belong only to disposable synthetic test fixtures.
"""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from push_align.adam_direction_probe import (
    AdamDirectionProbeError, MODE_ORDER, direction_contrast, loss_geometry,
    virtual_directions,
)


def synthetic_inputs(shapes=((2,), ()), step=1700):
    order, cursor = [], 0
    for index, shape in enumerate(shapes):
        count = int(np.prod(shape))
        order.append({"name": f"synthetic.{index}", "shape": list(shape),
                      "numel": count, "start": cursor, "end": cursor + count})
        cursor += count
    coordinate = np.arange(1, cursor + 1, dtype=np.float32)
    raw = np.stack((coordinate * .4, coordinate[::-1].copy() * -.3,
                    np.where(np.arange(cursor) % 2 == 0, -.2, .5).astype(np.float32),
                    coordinate * .1, coordinate * -.25)).astype(np.float32)
    base = (raw[0].astype(np.float64) + .1 * raw[1].astype(np.float64)).astype(np.float32)
    states = {}
    for index, item in enumerate(order):
        selected = coordinate[item["start"]:item["end"]]
        states[index] = {
            "step": torch.tensor(float(step), dtype=torch.float32),
            "exp_avg": torch.from_numpy((selected * -.03).copy()).reshape(item["shape"]),
            "exp_avg_sq": torch.from_numpy((selected * .002 + .001).copy()).reshape(item["shape"]),
        }
    optimizer = {"state": states, "param_groups": [{
        "lr": .001, "betas": (.9, .999), "eps": 1e-8, "weight_decay": 0.,
        "amsgrad": False, "maximize": False, "foreach": None,
        "capturable": False, "differentiable": False, "fused": None,
        "decoupled_weight_decay": False, "params": list(range(len(order))),
    }]}
    return raw, base, order, optimizer


def disposable_adam_oracle(clipped, order, saved, mode):
    """Run installed Adam only on newly created synthetic zero parameters."""
    parameters = [torch.nn.Parameter(torch.zeros(item["shape"], dtype=torch.float32))
                  for item in order]
    optimizer = torch.optim.Adam(parameters, lr=.001)
    state = deepcopy(saved)
    for entry in state["state"].values():
        if mode in ("ZERO_M", "COLD"):
            entry["exp_avg"].zero_()
        if mode == "COLD":
            entry["exp_avg_sq"].zero_()
            entry["step"].zero_()
    optimizer.load_state_dict(state)
    for parameter, item in zip(parameters, order):
        parameter.grad = torch.from_numpy(clipped[item["start"]:item["end"]].copy()).reshape(item["shape"])
    optimizer.step()
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters]).numpy().copy()


class AdamDirectionProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_tree_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            self.assertEqual(first.dtype, second.dtype)
            self.assertEqual(first.shape, second.shape)
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first:
                self.assert_tree_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(type(first), type(second))
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(first, second)

    def test_all_six_addends_exactly_match_disposable_installed_adam(self):
        raw, base, order, state = synthetic_inputs(((2, 3), (3,), (2, 2), (2,), (2, 3), ()))
        result = virtual_directions(raw, base, order, state)
        self.assertEqual(result["directions"].shape, (6, len(base)))
        self.assertEqual(result["directions"].dtype, np.float32)
        self.assertEqual(result["clipped_gradients"].dtype, np.float32)
        for objective in range(2):
            for mode_index, mode in enumerate(MODE_ORDER):
                expected = disposable_adam_oracle(result["clipped_gradients"][objective], order, state, mode)
                np.testing.assert_array_equal(result["directions"][3 * objective + mode_index], expected)

    def test_clipping_matches_installed_global_parameterwise_clipping(self):
        raw, base, order, state = synthetic_inputs()
        base[:] = [3., 4., 12.]
        raw[4] = [.1, .2, .3]
        result = virtual_directions(raw, base, order, state)
        for index, flat in enumerate((base, raw[4])):
            parameters = [torch.nn.Parameter(torch.zeros(item["shape"], dtype=torch.float32)) for item in order]
            for parameter, item in zip(parameters, order):
                parameter.grad = torch.from_numpy(flat[item["start"]:item["end"]].copy()).reshape(item["shape"])
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            expected = torch.cat([parameter.grad.reshape(-1) for parameter in parameters]).numpy()
            np.testing.assert_array_equal(result["clipped_gradients"][index], expected)
            clip = result["summary"]["clips"][("base", "total")[index]]
            self.assertEqual(clip["parameterwise_float32_norm"], float(norm))
            self.assertEqual(clip["coefficient"], float(torch.clamp(1. / (norm + 1e-6), max=1.)))
        self.assertTrue(result["summary"]["clips"]["base"]["would_scale"])
        self.assertFalse(result["summary"]["clips"]["total"]["would_scale"])

    def test_inputs_state_rng_and_global_grad_mode_remain_unchanged(self):
        raw, base, order, state = synthetic_inputs()
        original_raw, original_base = raw.copy(), base.copy()
        original_order, original_state = deepcopy(order), deepcopy(state)
        rng, enabled = torch.get_rng_state().clone(), torch.is_grad_enabled()
        with patch("torch.optim.Adam", side_effect=AssertionError("No optimizer construction")), \
                patch("torch.autograd.grad", side_effect=AssertionError("No derivative")), \
                patch.object(torch.Tensor, "backward", side_effect=AssertionError("No backward")):
            first = virtual_directions(raw, base, order, state)
            second = virtual_directions(raw, base, order, state)
        np.testing.assert_array_equal(raw, original_raw)
        np.testing.assert_array_equal(base, original_base)
        self.assertEqual(order, original_order)
        self.assert_tree_equal(state, original_state)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(enabled, torch.is_grad_enabled())
        np.testing.assert_array_equal(first["directions"], second["directions"])

    def test_saved_momentum_can_reverse_but_zero_m_descends_own_gradient(self):
        raw, base, order, state = synthetic_inputs(((2,),), step=11)
        base[:] = [.2, -.1]
        raw[4] = base
        state["state"][0]["exp_avg"][:] = torch.tensor([-1., .5])
        state["state"][0]["exp_avg_sq"][:] = .01
        result = virtual_directions(raw, base, order, state)
        records = result["summary"]["directions"]
        for offset in (0, 3):
            objective = ("base", "total")[offset // 3]
            self.assertEqual(records[offset]["loss_geometry"][objective]["relation"], "ASCENT")
            self.assertEqual(records[offset + 1]["loss_geometry"][objective]["relation"], "DESCENT")
            self.assertEqual(records[offset + 2]["loss_geometry"][objective]["relation"], "DESCENT")
        self.assertLess(records[0]["loss_geometry"]["base"]["descent_cosine"], 0.)
        self.assertGreater(records[1]["loss_geometry"]["base"]["descent_cosine"], 0.)

    def test_zero_m_retains_saved_second_moment_and_age_cold_resets_both(self):
        raw, base, order, state = synthetic_inputs(((2,),), step=2300)
        result = virtual_directions(raw, base, order, state)
        records = result["summary"]["directions"]
        self.assertEqual([record["next_step"] for record in records], [2301, 2301, 1] * 2)
        self.assertTrue(all(record["saved_step"] == 2300 for record in records))
        self.assertFalse(np.array_equal(result["directions"][1], result["directions"][2]))
        changed = deepcopy(state)
        changed["state"][0]["exp_avg_sq"].mul_(9.)
        after = virtual_directions(raw, base, order, changed)
        self.assertFalse(np.array_equal(result["directions"][1], after["directions"][1]))
        np.testing.assert_array_equal(result["directions"][[2, 5]], after["directions"][[2, 5]])

    def test_saved_base_and_independent_total_are_used_without_recombination(self):
        raw, base, order, state = synthetic_inputs(((2,),))
        base[:] = [-.7, .4]
        raw[4] = [.3, .8]
        result = virtual_directions(raw, base, order, state)
        aux = .2 * (raw[2].astype(np.float64) + .1 * raw[3].astype(np.float64))
        for record, delta in zip(result["summary"]["directions"], result["directions"]):
            for key, gradient in (("base", base), ("aux", aux), ("total", raw[4])):
                expected = loss_geometry(gradient, delta)
                self.assertEqual(record["loss_geometry"][key], expected)
        np.testing.assert_array_equal(result["clipped_gradients"][0], base)
        np.testing.assert_array_equal(result["clipped_gradients"][1], raw[4])
        self.assertIn("not historical exact base backward", result["summary"]["reference_gradients"]["base"])

    def test_contrasts_are_computed_between_independent_cases_in_declared_order(self):
        result = virtual_directions(*synthetic_inputs())
        summary, directions = result["summary"], result["directions"]
        self.assertEqual(summary["objective_order"], ["base", "total"])
        self.assertEqual(summary["mode_order"], ["KEEP", "ZERO_M", "COLD"])
        self.assertEqual([(record["objective"], record["mode"]) for record in summary["directions"]],
                         [(objective, mode) for objective in ("base", "total") for mode in MODE_ORDER])
        self.assertEqual(len(summary["objective_contrasts"]), 3)
        self.assertEqual(len(summary["state_contrasts"]), 4)
        for contrast in summary["objective_contrasts"] + summary["state_contrasts"]:
            expected = direction_contrast(directions[contrast["first_index"]], directions[contrast["second_index"]])
            for key, value in expected.items():
                self.assertEqual(contrast[key], value)
        json.dumps(summary, allow_nan=False)

    def test_parameter_and_output_row_subsets_use_displacement_coordinates(self):
        raw, base, order, state = synthetic_inputs(((2, 2), (2,), (2, 2), (2,), (10, 2), (10,)))
        order[-2]["name"], order[-1]["name"] = "network.4.weight", "network.4.bias"
        result = virtual_directions(raw, base, order, state)
        references = {"base": base.astype(np.float64),
                      "aux": .2 * (raw[2].astype(np.float64) + .1 * raw[3].astype(np.float64)),
                      "total": raw[4].astype(np.float64)}
        for record, delta in zip(result["summary"]["directions"], result["directions"]):
            self.assertEqual(len(record["parameter_blocks"]), 6)
            for item, block in zip(order, record["parameter_blocks"]):
                selected = slice(item["start"], item["end"])
                for name, gradient in references.items():
                    self.assertEqual(block["loss_geometry"][name], loss_geometry(gradient[selected], delta[selected]))
            head = record["output_head_groups"]
            self.assertEqual(head["status"], "APPLICABLE")
            self.assertEqual(len(head["groups"]), 5)
            self.assertEqual(sum(group["parameter_count"] for group in head["groups"].values()), 30)
            weight, bias = order[-2:]
            for group in head["groups"].values():
                ids = [weight["start"] + row * 2 + col for row in group["output_rows"] for col in range(2)]
                ids += [bias["start"] + row for row in group["output_rows"]]
                for name, gradient in references.items():
                    self.assertEqual(group["loss_geometry"][name], loss_geometry(gradient[ids], delta[ids]))
        generic = virtual_directions(*synthetic_inputs())["summary"]["directions"][0]
        self.assertEqual(generic["output_head_groups"]["status"], "NOT_APPLICABLE")

    def test_near_zero_and_near_orthogonal_geometry_never_become_descent(self):
        nearzero = loss_geometry(np.array([1e-13, 0.]), np.array([1., 0.]))
        self.assertIsNone(nearzero["descent_cosine"])
        self.assertEqual(nearzero["relation"], "UNDEFINED_NEAR_ZERO_NORM")
        nearorthogonal = loss_geometry(np.array([1., 0.]), np.array([-5e-7, 1.]))
        self.assertEqual(nearorthogonal["relation"], "NEAR_ORTHOGONAL")
        self.assertLess(nearorthogonal["dot"], 0.)
        self.assertEqual(loss_geometry(np.array([1., 0.]), np.array([-1e-4, 1.]))["relation"], "DESCENT")
        self.assertEqual(loss_geometry(np.array([1., 0.]), np.array([1e-4, 1.]))["relation"], "ASCENT")
        zero_pair = direction_contrast(np.zeros(2), np.ones(2))
        self.assertIsNone(zero_pair["cosine"])
        self.assertIsNone(zero_pair["second_to_first_norm_ratio"])

    def test_zero_gradients_and_moments_have_finite_undefined_directions(self):
        raw, base, order, state = synthetic_inputs()
        raw[:] = 0.
        base[:] = 0.
        for entry in state["state"].values():
            entry["exp_avg"].zero_()
            entry["exp_avg_sq"].zero_()
        result = virtual_directions(raw, base, order, state)
        np.testing.assert_array_equal(result["directions"], np.zeros((6, len(base)), np.float32))
        for record in result["summary"]["directions"]:
            self.assertTrue(all(metric["relation"] == "UNDEFINED_NEAR_ZERO_NORM"
                                for metric in record["loss_geometry"].values()))
        json.dumps(result["summary"], allow_nan=False)

    def test_invalid_raw_or_base_vectors_rejected(self):
        raw, base, order, state = synthetic_inputs()
        for broken in (raw.astype(np.float64), raw[:4], raw[:, :0], np.full_like(raw, np.nan)):
            with self.subTest(shape=broken.shape), self.assertRaises(AdamDirectionProbeError):
                virtual_directions(broken, base, order, state)
        for broken in (base.astype(np.float64), base[:-1], np.full_like(base, np.inf)):
            with self.subTest(base=broken), self.assertRaises(AdamDirectionProbeError):
                virtual_directions(raw, broken, order, state)

    def test_parameter_mapping_gaps_overlap_duplicates_and_shape_errors_rejected(self):
        raw, base, order, state = synthetic_inputs()
        bad_orders = []
        for field, value in (("start", 1), ("end", 1), ("numel", 1), ("shape", [3]), ("shape", [True])):
            broken = deepcopy(order)
            broken[0][field] = value
            bad_orders.append(broken)
        duplicate = deepcopy(order)
        duplicate[1]["name"] = duplicate[0]["name"]
        bad_orders.append(duplicate)
        for broken in bad_orders:
            with self.subTest(order=broken), self.assertRaises(AdamDirectionProbeError):
                virtual_directions(raw, base, broken, state)
        broken = deepcopy(state)
        broken["param_groups"][0]["params"] = [1, 0]
        with self.assertRaisesRegex(AdamDirectionProbeError, "parameter ids"):
            virtual_directions(raw, base, order, broken)

    def test_unsupported_adam_options_and_missing_state_rejected(self):
        raw, base, order, state = synthetic_inputs()
        for name, value in (("lr", .01), ("betas", (.8, .999)), ("eps", 1e-7),
                            ("weight_decay", .1), ("amsgrad", True), ("maximize", True),
                            ("foreach", True), ("fused", True), ("capturable", True),
                            ("differentiable", True), ("decoupled_weight_decay", True),
                            ("unregistered_option", False)):
            broken = deepcopy(state)
            broken["param_groups"][0][name] = value
            with self.subTest(option=name), self.assertRaisesRegex(AdamDirectionProbeError, "Unsupported"):
                virtual_directions(raw, base, order, broken)
        broken = deepcopy(state)
        del broken["state"][0]
        with self.assertRaisesRegex(AdamDirectionProbeError, "cover every"):
            virtual_directions(raw, base, order, broken)

    def test_nonfinite_negative_mismatched_and_nonintegral_moments_rejected(self):
        raw, base, order, state = synthetic_inputs()
        for field, value in (("exp_avg", torch.tensor([float("nan"), 0.])),
                             ("exp_avg_sq", torch.tensor([-.1, .2])),
                             ("exp_avg", torch.zeros(3)),
                             ("exp_avg_sq", torch.zeros(2, dtype=torch.float64)),
                             ("step", torch.tensor(2.5)), ("step", torch.tensor(float("inf"))),
                             ("step", torch.tensor(float(2**24)))):
            broken = deepcopy(state)
            broken["state"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(AdamDirectionProbeError):
                virtual_directions(raw, base, order, broken)
        broken = deepcopy(state)
        broken["state"][1]["step"].add_(1)
        with self.assertRaisesRegex(AdamDirectionProbeError, "differs across"):
            virtual_directions(raw, base, order, broken)


if __name__ == "__main__":
    unittest.main()
