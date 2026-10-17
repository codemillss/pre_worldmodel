"""Analytic gradient identities, independent of models and autograd."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from check_p4_gradient_probe import (algebraic_clip, component_geometry, cosine, cosine_relation,
                                   dot, head_partition_indices, norm, pairwise)


class GradientProbeOracleTests(unittest.TestCase):
    def raw(self, one, rollout, teacher, free):
        values = np.asarray([one, rollout, teacher, free], np.float32)
        total = values[0] + np.float32(.1) * values[1] + np.float32(.2) * (
            values[2] + np.float32(.1) * values[3])
        return np.concatenate((values, total[None]), axis=0)

    def test_hand_computed_euclidean_norm_and_dot(self):
        self.assertEqual(norm(np.asarray([3., 4.])), 5.)
        self.assertEqual(dot(np.asarray([1., 2.]), np.asarray([3., -4.])), -5.)

    def test_orthogonal_aligned_opposed_cosines(self):
        first = np.asarray([1., 0.])
        self.assertEqual(cosine(first, np.asarray([0., 1.])), 0.)
        self.assertEqual(cosine(first, 3 * first), 1.)
        self.assertEqual(cosine(first, -first), -1.)

    def test_zero_norm_cosine_is_undefined(self):
        self.assertIsNone(cosine(np.zeros(3), np.ones(3)))
        self.assertIsNone(cosine(np.zeros(3), np.zeros(3)))

    def test_numerically_tiny_norm_cosine_is_undefined(self):
        self.assertIsNone(cosine(np.asarray([1e-13, 0.]), np.ones(2)))
        self.assertIsNone(cosine(np.asarray([1e-12, 0.]), np.ones(2)))
        self.assertIsNotNone(cosine(np.asarray([2e-12, 0.]), np.ones(2)))

    def test_near_orthogonal_relation_uses_registered_tolerance(self):
        self.assertEqual(cosine_relation(None), "UNDEFINED_NEAR_ZERO_NORM")
        self.assertEqual(cosine_relation(-1e-7), "NEAR_ORTHOGONAL")
        self.assertEqual(cosine_relation(1e-6), "NEAR_ORTHOGONAL")
        self.assertEqual(cosine_relation(-2e-6), "OPPOSED")
        self.assertEqual(cosine_relation(2e-6), "ALIGNED")

    def test_raw_component_weights_are_applied_once(self):
        raw = self.raw([1., 0.], [0., 10.], [5., 0.], [0., 50.])
        result = component_geometry(raw)
        np.testing.assert_allclose(result["base"], [1., 1.], rtol=0, atol=0)
        np.testing.assert_allclose(result["auxiliary"], [1., 1.], rtol=0, atol=0)
        self.assertTrue(result["closure"]["pass"])
        self.assertAlmostEqual(result["triangle_norm_ratio"], 1.)

    def test_opposing_components_cancel_without_zero_cosine(self):
        raw = self.raw([1., 0.], [0., 0.], [-5., 0.], [0., 0.])
        result = component_geometry(raw)
        self.assertEqual(result["base_auxiliary_cosine"], -1.)
        self.assertEqual(result["triangle_norm_ratio"], 0.)
        self.assertTrue(result["closure"]["pass"])

    def test_zero_base_ratio_is_undefined(self):
        raw = self.raw([0., 0.], [0., 0.], [1., 0.], [0., 0.])
        self.assertIsNone(component_geometry(raw)["auxiliary_to_base_norm_ratio"])

    def test_all_zero_ratio_and_cosine_are_undefined(self):
        result = component_geometry(np.zeros((5, 3), np.float32))
        self.assertIsNone(result["triangle_norm_ratio"])
        self.assertIsNone(result["base_auxiliary_cosine"])
        self.assertTrue(result["closure"]["pass"])

    def test_substantial_total_corruption_fails_closure(self):
        raw = self.raw([1., 2.], [3., 4.], [2., 4.], [3., 1.])
        raw[4, 0] += .1
        self.assertFalse(component_geometry(raw)["closure"]["pass"])

    def test_closure_bound_uses_component_scale_under_cancellation(self):
        raw = self.raw([1000., 0.], [0., 0.], [-5000., 0.], [0., 0.])
        result = component_geometry(raw)
        self.assertEqual(result["total_norm"], 0.)
        self.assertGreater(result["closure"]["l2_bound"], .01)

    def test_pairwise_symmetry_and_norm_identity(self):
        vectors = np.asarray([[1., 2., 3.], [2., -1., 0.]], np.float32)
        result = pairwise(vectors)
        self.assertEqual(result["dot"][0][1], result["dot"][1][0])
        self.assertAlmostEqual(result["dot"][0][0], result["norm"][0] ** 2)

    def test_nonfinite_or_wrong_dtype_gradients_are_rejected(self):
        for raw in (np.zeros((5, 2), np.float64), np.zeros((4, 2), np.float32),
                    np.full((5, 2), np.nan, np.float32)):
            with self.assertRaises(ValueError): component_geometry(raw)

    def test_vector_broadcast_is_rejected(self):
        with self.assertRaises(ValueError): dot(np.ones(2), np.ones(3))

    def test_clip_uses_epsilon_and_preserves_direction(self):
        result = algebraic_clip(np.asarray([3., 4.]))
        self.assertAlmostEqual(result["coefficient"], 1 / 5.000001)
        self.assertLess(result["clipped_norm"], 1.)
        self.assertAlmostEqual(result["direction_cosine"], 1.)

    def test_small_gradient_is_not_clipped(self):
        result = algebraic_clip(np.asarray([.1, .2]))
        self.assertEqual(result["coefficient"], 1.)

    def test_zero_clipped_gradient_does_not_get_an_angle(self):
        result = algebraic_clip(np.zeros(2))
        self.assertEqual(result["coefficient"], 1.)
        self.assertIsNone(result["direction_cosine"])

    def test_invalid_clip_configuration_is_rejected(self):
        for maximum, epsilon in ((True, 1e-6), (0., 1e-6), (1., -1e-6), (np.inf, 1e-6)):
            with self.assertRaises(ValueError): algebraic_clip(np.ones(2), maximum, epsilon)

    def test_output_row_indices_include_corresponding_bias(self):
        order = [{"name": "network.4.weight", "shape": [10, 2], "start": 3, "end": 23},
                 {"name": "network.4.bias", "shape": [10], "start": 23, "end": 33}]
        groups = head_partition_indices(order)
        self.assertEqual(groups["object_xy"]["indices"].tolist(), [3, 4, 5, 6, 23, 24])
        self.assertEqual(groups["pusher_xy"]["indices"].tolist(), [15, 16, 17, 18, 29, 30])

    def test_output_partition_exactly_covers_head_excluding_hidden(self):
        order = [{"name": "network.4.weight", "shape": [10, 2], "start": 3, "end": 23},
                 {"name": "network.4.bias", "shape": [10], "start": 23, "end": 33}]
        indices = [int(index) for group in head_partition_indices(order).values() for index in group["indices"]]
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(sorted(indices), list(range(3, 33)))

    def test_output_partition_rejects_incompatible_head(self):
        with self.assertRaises(ValueError):
            head_partition_indices([{"name": "network.4.weight", "shape": [9, 2], "start": 0, "end": 18},
                                    {"name": "network.4.bias", "shape": [9], "start": 18, "end": 27}])


if __name__ == "__main__":
    unittest.main()
