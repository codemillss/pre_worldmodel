"""Candidate distribution, warm-start and input contracts for P2-05."""
from collections import Counter
from copy import deepcopy
import math
import unittest

import numpy as np

from push_align.candidates import (
    CandidateError, RESERVED_CANDIDATES, generate_candidates,
)
from push_align.config import load_pilot_config


class P2CandidateTest(unittest.TestCase):
    def setUp(self):
        self.config = load_pilot_config()

    def generate(self, config=None, seed=23, **kwargs):
        return generate_candidates(self.config if config is None else config,
                                   np.random.default_rng(seed), **kwargs)

    def test_default_counts_shape_constant_blocks_and_sources(self):
        batch = self.generate()
        self.assertEqual(batch.actions.shape, (256, 10, 2))
        self.assertEqual(batch.actions.dtype, np.float64)
        self.assertEqual(batch.sources[:8], RESERVED_CANDIDATES)
        self.assertEqual(Counter(batch.sources[8:]),
                         {"random_global": 124, "random_local_geometry": 62,
                          "random_local_warm": 62})
        np.testing.assert_array_equal(batch.actions[:, :5],
                                      np.repeat(batch.actions[:, :1], 5, axis=1))
        np.testing.assert_array_equal(batch.actions[:, 5:],
                                      np.repeat(batch.actions[:, 5:6], 5, axis=1))
        self.assertFalse(batch.actions.flags.writeable)

    def test_small_odd_remainder_allocated_without_deduplication(self):
        config = deepcopy(self.config)
        config["planner"].update(candidate_count=15, horizon=5, block_lengths=[2, 3])
        batch = self.generate(config)
        self.assertEqual(batch.actions.shape, (15, 5, 2))
        self.assertEqual(Counter(batch.sources[8:]),
                         {"random_global": 3, "random_local_geometry": 1,
                          "random_local_warm": 3})
        # The fixed zero/warm/disabled-G slots deliberately remain duplicates.
        np.testing.assert_array_equal(batch.actions[:8], 0)
        self.assertEqual(batch.diagnostics["duplicates_removed"], 0)
        self.assertEqual(batch.diagnostics["replacement_draws"], 0)

    def test_warm_shift_append_zero_then_block_mean(self):
        previous = np.array([[.002 * i, -.001 * i] for i in range(10)], dtype=np.float64)
        original = previous.copy()
        batch = self.generate(previous_actions=previous)
        np.testing.assert_allclose(batch.actions[1, :5],
                                   np.tile(previous[1:6].mean(axis=0), (5, 1)))
        np.testing.assert_allclose(batch.actions[1, 5:],
                                   np.tile(previous[6:].sum(axis=0) / 5, (5, 1)))
        np.testing.assert_array_equal(previous, original)
        self.assertTrue(batch.diagnostics["warm_start_available"])

    def test_disabled_geometry_cannot_perturb_fixed_slots(self):
        batch = self.generate(geometry_action=np.array([.02, .01]),
                              tangent_direction=np.array([1., 0.]))
        np.testing.assert_array_equal(batch.actions[2:8], 0)
        self.assertEqual(batch.diagnostics["geometry_proposal_status"], "disabled_zero")
        self.assertFalse(batch.diagnostics["geometry_tangent_perturbation_applied"])
        # Disabled G still has an explicitly zero-centered exploration group.
        self.assertTrue(np.any(batch.actions[132:194] != 0))

    def test_enabled_unavailable_geometry_falls_back_to_zero(self):
        config = deepcopy(self.config)
        config["planner"]["geometry_proposal_enabled"] = True
        batch = self.generate(config)
        np.testing.assert_array_equal(batch.actions[2:8], 0)
        self.assertEqual(batch.diagnostics["geometry_proposal_status"], "unavailable_zero")

    def test_enabled_geometry_fixed_proposals_use_current_unit_tangent(self):
        config = deepcopy(self.config)
        config["planner"]["geometry_proposal_enabled"] = True
        g = np.array([.015, .005])
        tangent = np.array([-.6, .8])
        batch = self.generate(config, geometry_action=g, tangent_direction=tangent)
        expected = np.array([[g, g], [g, [0., 0.]], [[0., 0.], g],
                             [.5 * g, .5 * g], [g + .01 * tangent, g],
                             [g - .01 * tangent, g]])
        np.testing.assert_allclose(batch.actions[2:8, 0], expected[:, 0])
        np.testing.assert_allclose(batch.actions[2:8, 5], expected[:, 1])
        self.assertEqual(batch.diagnostics["geometry_proposal_status"], "available")

    def test_radial_norm_projection_and_lower_planner_support_limit(self):
        config = deepcopy(self.config)
        config["planner"]["command_speed_norm_max_m_s"] = .03
        config["planner"]["geometry_proposal_enabled"] = True
        g = np.array([.1, .1])
        batch = self.generate(config, previous_actions=np.full((10, 2), .2),
                              geometry_action=g, tangent_direction=np.array([0., 1.]))
        norms = np.linalg.norm(batch.actions, axis=-1)
        self.assertLessEqual(float(norms.max()), .03 + 1e-15)
        np.testing.assert_allclose(batch.actions[2, 0],
                                   np.array([1., 1.]) * (.03 / math.sqrt(2)))
        self.assertEqual(config["pusher"]["command_speed_norm_max_m_s"], .06)
        self.assertTrue(batch.diagnostics["planner_support_limit_overridden"])

    def test_exact_global_and_correlated_local_draw_equations(self):
        config = deepcopy(self.config)
        config["planner"].update(candidate_count=16, random_local_noise_std_m_s=.001)
        batch = self.generate(config, seed=47)
        reference = np.random.default_rng(47)

        def disk(count):
            radius = .06 * np.sqrt(reference.random(count))
            angle = reference.uniform(0, 2 * math.pi, count)
            return radius[:, None] * np.stack((np.cos(angle), np.sin(angle)), axis=-1)

        first, independent = disk(4), disk(4)
        np.testing.assert_array_equal(batch.actions[8:12, 0], first)
        np.testing.assert_array_equal(batch.actions[8:12, 5], .7 * first + (1 - .7) * independent)
        for start in (12, 14):
            e1, e2 = reference.standard_normal((2, 2)), reference.standard_normal((2, 2))
            np.testing.assert_array_equal(batch.actions[start:start + 2, 0], .001 * e1)
            np.testing.assert_array_equal(batch.actions[start:start + 2, 5],
                                          .001 * (.7 * e1 + math.sqrt(1 - .7**2) * e2))

    def test_uniform_disk_sampling_statistical_witness(self):
        config = deepcopy(self.config)
        config["planner"]["candidate_count"] = 40008
        batch = self.generate(config, seed=101)
        first = batch.actions[8:20008, 0]
        scaled_radius_squared = np.sum(first * first, axis=1) / .06**2
        # This distinguishes area-uniform sqrt(U) from radius-uniform U.
        self.assertAlmostEqual(float(scaled_radius_squared.mean()), .5, delta=.012)
        self.assertAlmostEqual(float((scaled_radius_squared < .25).mean()), .25, delta=.012)
        np.testing.assert_allclose(first.mean(axis=0), 0, atol=.0007)

    def test_projection_is_after_half_geometry_not_before(self):
        config = deepcopy(self.config)
        config["planner"]["geometry_proposal_enabled"] = True
        batch = self.generate(config, geometry_action=np.array([.08, 0.]),
                              tangent_direction=np.array([0., 1.]))
        np.testing.assert_allclose(batch.actions[2, 0], [.06, 0.])
        np.testing.assert_allclose(batch.actions[5, 0], [.04, 0.])

    def test_finite_large_previous_command_can_be_projected_without_overflow(self):
        previous = np.full((10, 2), 1e308)
        batch = self.generate(previous_actions=previous)
        self.assertTrue(np.isfinite(batch.actions).all())
        self.assertLessEqual(float(np.linalg.norm(batch.actions, axis=-1).max()), .06 + 1e-15)
        np.testing.assert_allclose(batch.actions[1, 0], [.06 / math.sqrt(2)] * 2)

    def test_zero_noise_rho_endpoints_and_no_random_draw_when_reserved_only(self):
        config = deepcopy(self.config)
        config["planner"].update(candidate_count=8, random_local_noise_std_m_s=0.,
                                 second_block_correlation=1.)
        rng = np.random.default_rng(5)
        before = deepcopy(rng.bit_generator.state)
        batch = generate_candidates(config, rng)
        self.assertEqual(before, rng.bit_generator.state)
        np.testing.assert_array_equal(batch.actions, 0)
        for rho in (0., 1.):
            config["planner"].update(candidate_count=16, second_block_correlation=rho)
            batch = self.generate(config)
            if rho == 1:
                np.testing.assert_array_equal(batch.actions[8:12, 0], batch.actions[8:12, 5])
            np.testing.assert_array_equal(batch.actions[12:], 0)

    def test_seed_replay_and_config_input_immutability(self):
        before = deepcopy(self.config)
        first, second = self.generate(seed=11), self.generate(seed=11)
        np.testing.assert_array_equal(first.actions, second.actions)
        self.assertEqual(first.sources, second.sources)
        self.assertEqual(first.diagnostics, second.diagnostics)
        self.assertFalse(np.array_equal(first.actions, self.generate(seed=12).actions))
        self.assertEqual(before, self.config)

    def test_malformed_settings_rejected_before_rng_consumption(self):
        cases = [
            ("candidate_count", 7), ("candidate_count", True), ("candidate_count", 8.5),
            ("horizon", 0), ("horizon", True), ("block_lengths", [4, 5]),
            ("block_lengths", [0, 10]), ("block_lengths", [5, 3, 2]),
            ("block_lengths", [True, 9]), ("kind", "cem"),
            ("reserved_candidates", list(reversed(RESERVED_CANDIDATES))),
            ("geometry_proposal_enabled", 0), ("geometry_tangent_perturbation_m_s", -.01),
            ("random_local_noise_std_m_s", float("nan")),
            ("random_local_noise_std_m_s", -.1), ("second_block_correlation", -.1),
            ("second_block_correlation", 1.1), ("random_global_fraction", .4),
            ("command_speed_norm_max_m_s", .07), ("command_speed_norm_max_m_s", 0),
            ("command_speed_norm_max_m_s", float("inf")),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                config = deepcopy(self.config)
                config["planner"][field] = value
                rng = np.random.default_rng(1)
                before = deepcopy(rng.bit_generator.state)
                with self.assertRaises(CandidateError):
                    generate_candidates(config, rng)
                self.assertEqual(before, rng.bit_generator.state)

    def test_invalid_arrays_and_missing_tangent_rejected(self):
        cases = [
            {"previous_actions": [[0., 0.]] * 10},
            {"previous_actions": np.zeros((9, 2))},
            {"previous_actions": np.zeros((10, 2), dtype=np.int64)},
            {"previous_actions": np.full((10, 2), np.nan)},
            {"geometry_action": np.ones((1, 2))},
            {"geometry_action": np.array([np.inf, 0.])},
            {"tangent_direction": np.array([0., 0.])},
            {"tangent_direction": np.array([2., 0.])},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                rng = np.random.default_rng(1)
                before = deepcopy(rng.bit_generator.state)
                with self.assertRaises(CandidateError):
                    generate_candidates(self.config, rng, **kwargs)
                self.assertEqual(before, rng.bit_generator.state)
        config = deepcopy(self.config)
        config["planner"]["geometry_proposal_enabled"] = True
        with self.assertRaisesRegex(CandidateError, "requires tangent"):
            self.generate(config, geometry_action=np.array([.01, 0.]))

    def test_rng_and_physical_limit_contract_rejections(self):
        for rng in (None, 42, np.random.RandomState(1),
                    np.random.Generator(np.random.MT19937(1))):
            with self.subTest(rng=type(rng).__name__), self.assertRaises(CandidateError):
                generate_candidates(self.config, rng)
        config = deepcopy(self.config)
        config["reproducibility"]["rng"] = "numpy_MT19937"
        with self.assertRaises(CandidateError):
            self.generate(config)
        for value in (0., -.01, float("inf"), True):
            config = deepcopy(self.config)
            config["pusher"]["command_speed_norm_max_m_s"] = value
            with self.subTest(value=value), self.assertRaises(CandidateError):
                self.generate(config)
        with self.assertRaises(CandidateError):
            generate_candidates({}, np.random.default_rng(1))


if __name__ == "__main__":
    unittest.main()
