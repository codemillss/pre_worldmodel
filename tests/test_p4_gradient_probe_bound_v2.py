"""Regression: exact producer-bound rounding never weakens the ideal64 guard."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from check_p4_gradient_probe_v2 import (CLOSURE_ABSOLUTE_FLOOR, EPS32,
    dual_closure_comparison, producer_float32_closure_bound)


class GradientBoundCompatibilityTests(unittest.TestCase):
    def test_first_failed_record_float32_bound_is_exactly_reproduced(self):
        norm_sum = 61.38405156873924
        self.assertEqual(producer_float32_closure_bound(norm_sum), .0004683231527451426)
        ideal = 64. * EPS32 * norm_sum + 1e-12
        self.assertNotEqual(producer_float32_closure_bound(norm_sum), ideal)

    def test_explicit_rounding_matches_numpy2_weak_scalar_expression(self):
        for norm_sum in (0., 1e-12, .1, 1., 61.38405156873924, 1e6):
            observed = float(64. * np.finfo(np.float32).eps * norm_sum + 1e-12)
            self.assertEqual(producer_float32_closure_bound(norm_sum), observed)

    def test_actual_failed_record_closure_passes_both_bounds(self):
        row = dual_closure_comparison(1.3230118530405963e-5, 61.38405156873924,
                                      .0004683231527451426)
        self.assertTrue(row["recorded_matches_float32_exactly"])
        self.assertTrue(row["both_bounds_pass"])
        self.assertGreater(row["ideal_bound_margin"], 0.)

    def test_between_bounds_residual_is_not_promoted_to_pass(self):
        norm_sum = 61.38405156873924
        producer = producer_float32_closure_bound(norm_sum)
        ideal = 64. * EPS32 * norm_sum + CLOSURE_ABSOLUTE_FLOOR
        error = (producer + ideal) / 2
        row = dual_closure_comparison(error, norm_sum, producer)
        self.assertTrue(row["passes_recorded_float32_bound"])
        self.assertFalse(row["passes_ideal_float64_bound"])
        self.assertFalse(row["both_bounds_pass"])

    def test_changed_reported_bound_is_detected_exactly(self):
        producer = producer_float32_closure_bound(1.)
        row = dual_closure_comparison(0., 1., float(np.nextafter(producer, np.inf)))
        self.assertFalse(row["recorded_matches_float32_exactly"])

    def test_zero_components_preserve_floor_and_undefined_direction_policy(self):
        bound = producer_float32_closure_bound(0.)
        self.assertEqual(bound, float(np.float32(1e-12)))
        row = dual_closure_comparison(0., 0., bound)
        self.assertTrue(row["both_bounds_pass"])
        self.assertEqual(row["ideal_float64_bound"], 1e-12)

    def test_residual_above_both_bounds_is_rejected(self):
        row = dual_closure_comparison(.01, 1., producer_float32_closure_bound(1.))
        self.assertFalse(row["passes_recorded_float32_bound"])
        self.assertFalse(row["passes_ideal_float64_bound"])
        self.assertFalse(row["both_bounds_pass"])

    def test_invalid_inputs_are_rejected(self):
        for value in (True, -.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError): producer_float32_closure_bound(value)
        for error, bound in ((True, 1.), (-.1, 1.), (.1, 0.), (.1, float("nan"))):
            with self.assertRaises(ValueError): dual_closure_comparison(error, 1., bound)


if __name__ == "__main__": unittest.main()
