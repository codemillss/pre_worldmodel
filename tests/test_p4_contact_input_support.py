"""Input/metric/provenance arithmetic only: no model, fitting, or physics."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from diagnose_p4_contact_input_support import (
    LEARNING, FEATURE_FIELDS, add_occurrence, contact_regimes, exposure_by_window,
    geometry, load_normalizer, model_features, ranked_distance, read, summarize_neighbor,
)


class InputBoundaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = LEARNING / "normalizer"
        metadata = read(folder / "normalizer.json")["normalizer"]
        cls.normalizer = load_normalizer(folder, metadata["schema"], metadata["dataset_identity"], metadata["normalizer_hash"])

    def test_actual_boundary_is_float64_encode_normalize_then_float32(self):
        state = np.array([.25, -.125, .234, .01, -.02, .03, .3125, -.0625, .025, -.0125])
        action = np.array([.04, -.03])
        yaw = torch.tensor(.234, dtype=torch.float64)
        manual = torch.tensor([.0625, .0625, 0., 0., .01, -.02, .03, .025, -.0125, .04, -.03], dtype=torch.float64)
        manual[2], manual[3] = torch.sin(2*yaw), torch.cos(2*yaw)
        mean = torch.tensor(self.normalizer.arrays["input_mean"], dtype=torch.float64)
        scale = torch.tensor(self.normalizer.arrays["input_scale"], dtype=torch.float64)
        with patch.object(torch.nn.Module, "__call__", side_effect=AssertionError("No model call permitted")):
            physical, values32, values64, _ = model_features(state, action, self.normalizer)
        np.testing.assert_array_equal(physical, manual.numpy())
        np.testing.assert_array_equal(values32, ((manual-mean)/scale).to(torch.float32).numpy())
        np.testing.assert_array_equal(values64, ((manual-mean)/scale).numpy())
        self.assertEqual(values32.dtype, np.float32)
        self.assertEqual(values64.dtype, np.float64)

    def test_absolute_translation_does_not_change_model_input(self):
        states = np.zeros((2, 10)); states[:, 6:8] = [.0625, .03125]
        states[1, [0, 6]] += .25; states[1, [1, 7]] -= .125
        _, values, _, _ = model_features(states, np.zeros((2, 2)), self.normalizer)
        np.testing.assert_array_equal(values[0], values[1])

    def test_measured_pusher_velocity_and_command_are_separate(self):
        state = np.zeros(10); state[8:10] = [.01, -.02]
        physical, _, _, _ = model_features(state, np.array([-.03, .04]), self.normalizer)
        np.testing.assert_array_equal(physical[7:9], [.01, -.02])
        np.testing.assert_array_equal(physical[9:11], [-.03, .04])


class RankedMetricTest(unittest.TestCase):
    def test_stable_id_ties_and_coordinate_decomposition(self):
        values = np.array([[3, 4], [-3, -4], [0, 0]], np.float32)
        order, ranks, distance, squared = ranked_distance(values, np.zeros(2, np.float32), ["z", "a", "q"])
        np.testing.assert_array_equal(order, [2, 1, 0])
        np.testing.assert_array_equal(ranks, [3, 2, 1])
        np.testing.assert_array_equal(distance, [5, 5, 0])
        np.testing.assert_array_equal(squared.sum(axis=1), distance**2)

    def test_float32_values_are_promoted_before_distance_arithmetic(self):
        values = np.array([[.1, 1e-8], [.2, 2e-8]], np.float32)
        query = np.array([.03, -3e-8], np.float32)
        _, _, _, squared = ranked_distance(values, query, ["a", "b"])
        np.testing.assert_array_equal(squared, (values.astype(np.float64)-query.astype(np.float64))**2)
        self.assertEqual(squared.dtype, np.float64)

    def test_duplicate_or_nonfinite_values_rejected(self):
        for values, ids in ((np.zeros((2, 2)), ["a", "a"]), (np.array([[0., np.nan], [0., 0.]]), ["a", "b"])):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                ranked_distance(values, np.zeros(2), ids)


class ExposureProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.source = {"source": "old_train", "episode_id": "ep", "family_id": "family",
                       "transition_index": 3, "regime": "transition"}
        self.state = np.zeros(10); self.action = np.array([.01, 0.]); self.next_state = np.zeros(10)
        self.next_state[0] = .001

    def add(self, population, slot=1, depth=0, exposure=21, **overrides):
        args = {"source": self.source, "state": self.state, "action": self.action, "next_state": self.next_state}
        args.update(overrides)
        add_occurrence(population, "same_physical_transition", slot=slot, depth=depth, exposure=exposure, mass=.1, **args)

    def test_overlap_dedup_separates_teacher_and_rollout_initialization(self):
        population = {}; self.add(population, slot=1, depth=0); self.add(population, slot=2, depth=3)
        self.assertEqual(len(population), 1)
        row = population["same_physical_transition"]
        self.assertEqual(row["teacher_transition_occurrences"], 42)
        self.assertEqual(row["rollout_initial_state_occurrences"], 21)
        self.assertEqual(row["rollout_target_occurrences_by_depth"], [21, 0, 0, 21, 0])
        self.assertEqual(row["window_memberships"], [[1, 0], [2, 3]])

    def test_repeated_identity_with_different_tensor_or_provenance_rejected(self):
        changed_state = self.state.copy(); changed_state[0] += 1e-6
        for change in ({"state": changed_state}, {"source": {**self.source, "family_id": "other"}}):
            population = {}; self.add(population)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.add(population, slot=2, **change)

    def test_same_window_depth_not_double_counted(self):
        population = {}; self.add(population)
        with self.assertRaises(ValueError): self.add(population)

    def test_frozen_batch_exposure_and_mass_reconcile(self):
        sizes = np.array([64]*979+[16]*21)
        positions = np.arange(sizes.sum(), dtype=np.int64) % 2960
        trace = {"window_positions": positions, "batch_offsets": np.r_[0, np.cumsum(sizes)]}
        counts, mass = exposure_by_window(trace)
        self.assertEqual(counts.sum(), 62992)
        self.assertAlmostEqual(mass.sum()*5, 1000., places=10)
        # Each window exposes five actual teacher inputs and one real rollout start.
        self.assertEqual(counts.sum()*5, 314960)
        trace["batch_offsets"][1] = 63
        with self.assertRaises(ValueError): exposure_by_window(trace)

    def test_neighbor_reports_target_and_disjoint_squared_groups(self):
        population = {}; self.add(population)
        squared = np.arange(11, dtype=np.float64)
        config = {"object": {"size_full_xyz_m": [.08, .06, .02]}, "pusher": {"radius_m": .008}}
        row = summarize_neighbor(population["same_physical_transition"], 1, np.sqrt(squared.sum()), squared,
                                 self.state, self.next_state, config)
        self.assertEqual(set(row["squared_distance_by_feature"]), set(FEATURE_FIELDS))
        self.assertAlmostEqual(sum(row["squared_distance_by_group"].values()), squared.sum())
        np.testing.assert_array_equal(row["true_target_delta_minus_query_delta"], np.zeros(10))
        self.assertTrue(row["training_exposed"])


class ContactGeometryTest(unittest.TestCase):
    def test_frozen_contact_fraction_types(self):
        summary = np.zeros((20, 4)); summary[1, 2] = 1.; summary[2, 2] = .001
        self.assertEqual(contact_regimes(summary)[:3].tolist(), ["noncontact", "sustained_contact", "transition"])
        summary[3, 2] = 1.001
        with self.assertRaises(ValueError): contact_regimes(summary)

    def test_rotated_body_geometry_is_annotation_only(self):
        config = {"object": {"size_full_xyz_m": [.08, .06, .02]}, "pusher": {"radius_m": .008}}
        state = np.zeros(10); state[2] = np.pi/2; state[7] = .052
        value = geometry(state, config)
        np.testing.assert_allclose(value["relative_body_xy_m"], [.052, 0.], atol=1e-16)
        self.assertAlmostEqual(value["boundary_gap_m"], .004, places=15)


if __name__ == "__main__": unittest.main()
