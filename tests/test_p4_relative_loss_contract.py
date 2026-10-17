"""Independent analytic contract examples; no learned model or optimizer."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from check_p4_relative_loss import (combined_loss, masked_relative_loss,
                                    moments_loss, relative_moments,
                                    relative_residual, relative_scale)


class RelativeLossOracleTests(unittest.TestCase):
    def setUp(self):
        self.target = np.zeros((2, 3, 10), np.float64)
        self.predicted = self.target.copy()
        self.mask = np.ones((2, 3), np.bool_)
        self.scale = np.asarray([2., 3.], np.float64)

    def test_equal_object_pusher_translation_cancels(self):
        self.predicted[..., [0, 1, 6, 7]] = np.asarray([.8, -.3, .8, -.3])
        self.assertEqual(masked_relative_loss(self.predicted, self.target, self.mask, self.scale), 0.)

    def test_opposite_errors_amplify_and_two_scalar_denominator(self):
        self.predicted[..., [0, 1, 6, 7]] = np.asarray([-1., -1.5, 1., 1.5])
        self.assertEqual(masked_relative_loss(self.predicted, self.target, self.mask, self.scale), 1.)

    def test_partially_masked_suffix_does_not_enter_denominator(self):
        self.mask[:, 1:] = False
        self.predicted[:, 0, 6] = 2.
        self.predicted[:, 1:] = np.nan
        self.assertEqual(masked_relative_loss(self.predicted, self.target, self.mask, self.scale), .5)

    def test_invalid_predicted_valid_row_is_failure(self):
        self.predicted[0, 0, 6] = np.nan
        with self.assertRaises(ValueError):
            masked_relative_loss(self.predicted, self.target, self.mask, self.scale)

    def test_empty_denominator_is_failure(self):
        with self.assertRaises(ValueError):
            masked_relative_loss(self.predicted, self.target, np.zeros_like(self.mask), self.scale)

    def test_relative_residual_sign(self):
        self.predicted[..., 0] = 3.
        self.predicted[..., 6] = 1.
        self.assertTrue(np.array_equal(relative_residual(self.predicted, self.target)[..., 0],
                                       np.full((2, 3), -2.)))

    def test_yaw_velocity_errors_are_outside_xy_term(self):
        self.predicted[..., [2, 3, 4, 5, 8, 9]] = 10.
        self.assertEqual(masked_relative_loss(self.predicted, self.target, self.mask, self.scale), 0.)

    def test_scale_uses_only_original_marginal_delta_scales(self):
        delta = np.ones(10, np.float64)
        delta[[0, 1, 6, 7]] = [3., 5., 4., 12.]
        self.assertTrue(np.array_equal(relative_scale(delta), np.asarray([5., 13.])))

    def test_scale_rejects_invalid_and_float32(self):
        for value in (np.ones(10, np.float32), np.zeros(10, np.float64), np.ones(9, np.float64)):
            with self.assertRaises(ValueError):
                relative_scale(value)

    def test_independent_direct_and_cross_term_moment_identity(self):
        obj = np.asarray([[1., 2.], [-3., 4.], [.5, -1.]], np.float64)
        push = np.asarray([[2., -2.], [3., 2.], [-.5, 1.]], np.float64)
        moments = relative_moments(obj, push, self.scale)
        direct = float(np.square((push - obj) / self.scale).sum() / 6)
        self.assertAlmostEqual(moments_loss(moments, 3), direct, places=14)

    def test_correlation_is_not_two_independent_position_losses(self):
        obj = np.asarray([[1., 1.]], np.float64)
        same, opposite = obj.copy(), -obj
        self.assertEqual(moments_loss(relative_moments(obj, same, self.scale), 1), 0.)
        self.assertGreater(moments_loss(relative_moments(obj, opposite, self.scale), 1), 0.)

    def test_historical_and_augmented_loss_coefficients(self):
        self.assertAlmostEqual(combined_loss(2., 3., 4., 5., .1, .25), 3.425)
        self.assertAlmostEqual(combined_loss(2., 3., 4., 5., .1, 0.), 2.3)

    def test_weight_validation(self):
        for coefficient in (True, -.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                combined_loss(1., 2., 3., 4., .1, coefficient)

    def test_count_is_exact_integer_not_bool(self):
        for count in (True, 0, 1., -2):
            with self.assertRaises(ValueError):
                moments_loss(np.ones(3, np.float64), count)


if __name__ == "__main__":
    unittest.main()
