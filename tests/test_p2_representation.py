"""Hand-computed and adversarial P2-03 representation contract checks."""
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from push_align.representation import (
    DELTA_FIELDS, DELTA_UNITS, FEATURE_FIELDS, FEATURE_UNITS,
    STATE_FIELDS, STATE_UNITS, RepresentationError, encode, integrate,
    state_delta,
)


def example(dtype=np.float64):
    return np.array([.2, -.1, math.pi / 12, .03, -.02, .04, .25, -.08, .01, -.015], dtype=dtype)


class NumpyRepresentationTest(unittest.TestCase):
    def test_feature_columns_match_hand_computed_applied_and_measured_velocities(self):
        state, action = example(), np.array([.06, -.03])
        expected = [.05, .02, .5, math.sqrt(3)/2, .03, -.02, .04, .01, -.015, .06, -.03]
        np.testing.assert_allclose(encode(state, action), expected, atol=1e-15, rtol=0)
        self.assertEqual(len(STATE_FIELDS), 10)
        self.assertEqual(len(FEATURE_FIELDS), 11)
        self.assertEqual(DELTA_FIELDS, tuple("delta_" + f for f in STATE_FIELDS))
        self.assertEqual(DELTA_UNITS, STATE_UNITS)
        self.assertEqual(FEATURE_UNITS[2:4], ("1", "1"))

    def test_numpy_float32_is_promoted_to_float64_for_all_transforms(self):
        state = example(np.float32)
        action = np.array([.03, 0], dtype=np.float32)
        delta = np.full(10, .001, dtype=np.float32)
        for result in (encode(state, action), state_delta(state, state + delta), integrate(state, delta)):
            self.assertEqual(result.dtype, np.float64)

    def test_single_batch_sequence_and_empty_shapes_are_preserved(self):
        for leading in ((), (1,), (2, 3), (0,)):
            with self.subTest(leading=leading):
                state = np.broadcast_to(example(), leading + (10,)).copy()
                action = np.zeros(leading + (2,))
                delta = np.zeros_like(state)
                self.assertEqual(encode(state, action).shape, leading + (11,))
                self.assertEqual(state_delta(state, state).shape, state.shape)
                self.assertEqual(integrate(state, delta).shape, state.shape)

    def test_translation_invariance_and_pi_symmetry_hold_in_feature_space(self):
        state, action = example(), np.array([.03, -.01])
        translated = state.copy()
        translated[[0, 1, 6, 7]] += [1.2, -.7, 1.2, -.7]
        symmetric = state.copy()
        symmetric[2] -= math.pi
        np.testing.assert_allclose(encode(state, action), encode(translated, action), atol=3e-16, rtol=0)
        np.testing.assert_allclose(encode(state, action), encode(symmetric, action), atol=1e-15, rtol=0)

    def test_hand_computed_delta_and_integration_use_addition_for_all_non_yaw_columns(self):
        state = example()
        expected_delta = np.array([.004, -.002, .01, -.01, .03, -.02, .005, -.003, .02, .01])
        next_state = state + expected_delta
        np.testing.assert_allclose(state_delta(state, next_state), expected_delta, atol=3e-16, rtol=0)
        np.testing.assert_allclose(integrate(state, expected_delta), next_state, atol=3e-16, rtol=0)

    def test_yaw_boundary_retains_small_signed_two_pi_displacements(self):
        for start, end, expected in ((math.pi-.02, -math.pi+.03, .05),
                                     (-math.pi+.02, math.pi-.03, -.05)):
            with self.subTest(start=start):
                state, next_state = example(), example()
                state[2], next_state[2] = start, end
                delta = state_delta(state, next_state)
                self.assertAlmostEqual(delta[2], expected, places=14)
                np.testing.assert_allclose(integrate(state, delta), next_state, atol=2e-15, rtol=0)

    def test_transition_yaw_is_not_task_pi_symmetry(self):
        state, next_state = example(), example()
        next_state[2] += .75 * math.pi
        self.assertAlmostEqual(state_delta(state, next_state)[2], .75 * math.pi)

    def test_endpoints_do_not_certify_winding_number_or_large_rotation(self):
        state, next_state = example(), example()
        # The same endpoint can result from .1 or 2*pi+.1 radians of rotation.
        next_state[2] += .1
        principal = state_delta(state, next_state)[2]
        self.assertAlmostEqual(principal, .1)
        next_state[2] += 2 * math.pi
        self.assertAlmostEqual(state_delta(state, next_state)[2], principal)
        next_state[2] = state[2] + math.pi
        ambiguous = state_delta(state, next_state)
        self.assertAlmostEqual(ambiguous[2], -math.pi)
        with self.assertRaisesRegex(RepresentationError, "Raw physical delta_yaw"):
            integrate(state, ambiguous)

    def test_raw_predicted_large_yaw_rejected_before_wrapping(self):
        for angle in (math.pi, -math.pi, 2*math.pi+.1, -4*math.pi):
            with self.subTest(angle=angle):
                delta = np.zeros(10)
                delta[2] = angle
                with self.assertRaisesRegex(RepresentationError, "Raw physical delta_yaw"):
                    integrate(example(), delta)
        delta[2] = np.nextafter(math.pi, 0)
        self.assertTrue(np.isfinite(integrate(example(), delta)).all())

    def test_invalid_shapes_cannot_broadcast_or_implicitly_add_batch_axis(self):
        for state, action in ((np.zeros(9), np.zeros(2)), (np.zeros(10), np.zeros(3)),
                              (np.zeros((2, 10)), np.zeros(2)),
                              (np.zeros((2, 1, 10)), np.zeros((2, 2))),
                              (np.array(0.), np.zeros(2))):
            with self.subTest(shapes=(state.shape, action.shape)):
                with self.assertRaises(RepresentationError):
                    encode(state, action)
        for function in (state_delta, integrate):
            with self.assertRaisesRegex(RepresentationError, "leading shapes"):
                function(np.zeros((2, 10)), np.zeros(10))

    def test_invalid_dtype_non_arrays_and_mixed_dtypes_are_rejected(self):
        for dtype in (np.int64, np.bool_, np.complex128, np.float16):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(RepresentationError, "dtype"):
                    encode(np.zeros(10, dtype=dtype), np.zeros(2, dtype=dtype))
        with self.assertRaisesRegex(RepresentationError, "NumPy array or Torch tensor"):
            encode([0.] * 10, np.zeros(2))
        with self.assertRaisesRegex(RepresentationError, "identical dtype"):
            encode(np.zeros(10, dtype=np.float32), np.zeros(2, dtype=np.float64))

    def test_nonfinite_values_in_each_input_and_output_are_rejected(self):
        for invalid in (np.nan, np.inf, -np.inf):
            for function, width in ((encode, 2), (state_delta, 10), (integrate, 10)):
                for input_index in (0, 1):
                    with self.subTest(function=function.__name__, invalid=invalid, input=input_index):
                        a, b = np.zeros(10), np.zeros(width)
                        (a if input_index == 0 else b)[0] = invalid
                        with self.assertRaisesRegex(RepresentationError, "finite"):
                            function(a, b)
        a, b = np.zeros(10), np.zeros(10)
        a[0], b[0] = -np.finfo(np.float64).max, np.finfo(np.float64).max
        with self.assertRaisesRegex(RepresentationError, "finite"):
            state_delta(a, b)
        with self.assertRaisesRegex(RepresentationError, "finite"):
            integrate(b, b)
        a[6], a[0] = np.finfo(np.float64).max, -np.finfo(np.float64).max
        with self.assertRaisesRegex(RepresentationError, "finite"):
            encode(a, np.zeros(2))

    def test_inputs_unchanged_and_outputs_are_independent_arrays(self):
        state, action, delta = example(), np.array([.03, 0]), np.full(10, .001)
        before = [value.copy() for value in (state, action, delta)]
        for result in (encode(state, action), state_delta(state, state+delta), integrate(state, delta)):
            result[...] = 0
        for value, expected in zip((state, action, delta), before):
            np.testing.assert_array_equal(value, expected)

    def test_numpy_import_and_transforms_do_not_import_torch(self):
        source = Path(__file__).resolve().parents[1] / "src/push_align/representation.py"
        code = ("import importlib.util, sys, numpy as np; "
                f"spec = importlib.util.spec_from_file_location('pure_representation', {str(source)!r}); "
                "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
                "module.encode(np.zeros(10), np.zeros(2)); "
                "module.integrate(np.zeros(10), np.zeros(10)); "
                "assert 'torch' not in sys.modules")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Torch not installed")
class TorchRepresentationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch

    def test_dtype_device_shapes_and_numpy_agreement(self):
        torch = self.torch
        for dtype in (torch.float32, torch.float64):
            for leading in ((), (2,), (2, 3), (0,)):
                with self.subTest(dtype=dtype, leading=leading):
                    state = torch.tensor(example(), dtype=dtype).expand(leading + (10,)).clone()
                    action = torch.zeros(leading + (2,), dtype=dtype)
                    delta = torch.full_like(state, .001)
                    outputs = (encode(state, action), state_delta(state, state+delta), integrate(state, delta))
                    expected = (encode(state.numpy(), action.numpy()),
                                state_delta(state.numpy(), (state+delta).numpy()),
                                integrate(state.numpy(), delta.numpy()))
                    for actual, reference in zip(outputs, expected):
                        self.assertEqual(actual.dtype, dtype)
                        self.assertEqual(actual.device, state.device)
                        self.assertEqual(actual.shape, reference.shape)
                        np.testing.assert_allclose(actual.numpy(), reference, atol=4e-7, rtol=1e-5)

    def test_encode_float32_gradient_has_hand_computed_columns_and_signs(self):
        torch = self.torch
        state = torch.tensor(example(), dtype=torch.float32, requires_grad=True)
        action = torch.tensor([.03, -.02], dtype=torch.float32, requires_grad=True)
        weights = torch.arange(1., 12.)
        (encode(state, action) * weights).sum().backward()
        expected = np.array([-1, -2, 6*math.cos(math.pi/6)-8*math.sin(math.pi/6),
                             5, 6, 7, 1, 2, 8, 9])
        np.testing.assert_allclose(state.grad.numpy(), expected, atol=2e-6, rtol=1e-5)
        np.testing.assert_array_equal(action.grad.numpy(), [10, 11])

    def test_delta_and_integrate_preserve_gradients_across_yaw_boundary(self):
        torch = self.torch
        current = torch.tensor(example(), dtype=torch.float32, requires_grad=True)
        future_values = example(); future_values[2] = -math.pi+.03
        with torch.no_grad():
            current[2] = math.pi-.02
        future = torch.tensor(future_values, dtype=torch.float32, requires_grad=True)
        delta = state_delta(current, future)
        self.assertAlmostEqual(float(delta[2].detach()), .05, places=5)
        delta.sum().backward()
        np.testing.assert_array_equal(current.grad.numpy(), -np.ones(10))
        np.testing.assert_array_equal(future.grad.numpy(), np.ones(10))
        base = current.detach().clone().requires_grad_()
        predicted = delta.detach().clone().requires_grad_()
        integrated = integrate(base, predicted)
        integrated.sum().backward()
        np.testing.assert_array_equal(base.grad.numpy(), np.ones(10))
        np.testing.assert_array_equal(predicted.grad.numpy(), np.ones(10))

    def test_recursive_encode_integrate_has_nonzero_earlier_step_gradient(self):
        torch = self.torch
        state = torch.tensor(example(), dtype=torch.float32, requires_grad=True)
        action = torch.tensor([.03, -.02], dtype=torch.float32, requires_grad=True)
        # A deterministic differentiable fixture substitutes for a future model.
        first_delta = torch.cat((encode(state, action)[:2]*.01,
                                 torch.zeros(8, dtype=state.dtype)))
        first_state = integrate(state, first_delta)
        second_features = encode(first_state, action)
        second_features[0].backward()
        self.assertAlmostEqual(float(state.grad[0]), -.99, places=6)
        self.assertAlmostEqual(float(state.grad[6]), .99, places=6)

    def test_torch_inputs_remain_unchanged(self):
        torch = self.torch
        state = torch.tensor(example(), dtype=torch.float32, requires_grad=True)
        action = torch.tensor([.03, 0], dtype=torch.float32, requires_grad=True)
        delta = torch.full((10,), .001, dtype=torch.float32, requires_grad=True)
        before = [value.detach().clone() for value in (state, action, delta)]
        encode(state, action); state_delta(state, state+delta); integrate(state, delta)
        for value, expected in zip((state, action, delta), before):
            self.assertTrue(torch.equal(value, expected))

    def test_torch_dtype_backend_device_and_dense_layout_contracts(self):
        torch = self.torch
        for dtype in (torch.int64, torch.bool, torch.float16, torch.complex64):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(RepresentationError, "dtype"):
                    encode(torch.zeros(10, dtype=dtype), torch.zeros(2, dtype=dtype))
        with self.assertRaisesRegex(RepresentationError, "same backend"):
            encode(torch.zeros(10), np.zeros(2, dtype=np.float32))
        with self.assertRaisesRegex(RepresentationError, "identical dtype"):
            integrate(torch.zeros(10), torch.zeros(10, dtype=torch.float64))
        with self.assertRaisesRegex(RepresentationError, "same device"):
            integrate(torch.zeros(10), torch.zeros(10, device="meta"))
        with self.assertRaisesRegex(RepresentationError, "materialized"):
            integrate(torch.zeros(10, device="meta"), torch.zeros(10, device="meta"))
        with self.assertRaisesRegex(RepresentationError, "dense strided"):
            encode(torch.zeros(10).to_sparse(), torch.zeros(2))

    def test_torch_nonfinite_and_raw_large_yaw_are_rejected(self):
        torch = self.torch
        for function, width in ((encode, 2), (state_delta, 10), (integrate, 10)):
            for index in (0, 1):
                a, b = torch.zeros(10), torch.zeros(width)
                (a if index == 0 else b)[0] = float("nan")
                with self.assertRaisesRegex(RepresentationError, "finite"):
                    function(a, b)
        for angle in (math.pi, -math.pi, 2*math.pi+.1):
            delta = torch.zeros(10); delta[2] = angle
            with self.assertRaisesRegex(RepresentationError, "Raw physical delta_yaw"):
                integrate(torch.zeros(10), delta)


if __name__ == "__main__":
    unittest.main()
