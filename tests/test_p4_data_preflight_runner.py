"""Pure regression tests for preflight evidence comparisons, using fake episodes.

These fixtures exercise helper semantics only. They do not claim to be collected
physical episodes, and no environment, model, or recorder is constructed here.
"""
from copy import deepcopy
from pathlib import Path
import sys
import tomllib
import unittest

import numpy as np

from push_align.config import load_pilot_config
from push_align.data import Episode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from p4_data_preflight import compare_dt, exact_paths  # noqa: E402


BOUNDARY_KEYS = {
    "state", "sim_time", "raw_integration_state", "raw_reference_height",
    "raw_warning_counts", "raw_warning_lastinfo", "raw_needs_reset", "observation_valid",
}


def fake_episode():
    """A 22-action timeline with two synthetic substeps per action."""
    ticks = 22
    times = np.arange(ticks + 1, dtype=float) * .1
    states = np.zeros((ticks + 1, 10))
    states[:, 7] = times * .014
    states[:, 9] = .014
    sub_times = np.arange(1, 2 * ticks + 1, dtype=float) * .05
    sub_states = np.zeros((2 * ticks, 10))
    sub_states[:, 7] = sub_times * .014
    sub_states[:, 9] = .014
    raw = np.column_stack([times, states[:, 0], states[:, 7], np.ones(ticks + 1)])
    actions = np.tile([-.003, .014], (ticks, 1))
    arrays = {
        "state": states, "sim_time": times,
        "raw_integration_state": raw, "raw_reference_height": np.full(ticks + 1, .01),
        "raw_warning_counts": np.zeros((ticks + 1, 2), dtype=int),
        "raw_warning_lastinfo": np.zeros((ticks + 1, 2), dtype=int),
        "raw_needs_reset": np.zeros(ticks + 1, dtype=bool),
        "observation_valid": np.ones(ticks + 1, dtype=bool),
        "action_requested": actions.copy(), "action_applied": actions.copy(),
        "actual_dt": np.full(ticks, .1), "step_valid": np.ones(ticks, dtype=bool),
        "geometry_valid": np.ones(ticks, dtype=bool),
        "transition_valid": np.ones(ticks, dtype=bool),
        "substep_states": sub_states, "substep_times": sub_times,
        "substep_offsets": np.arange(ticks + 1, dtype=int) * 2,
        "contact_summary": np.zeros((ticks, 4)), "validity_summary": np.zeros((ticks, 5)),
    }
    metadata = {
        "diagnostics": [{"samples": [{"contacts": [{"pair": "pusher_object",
                            "normal_force_N": 0., "distance_m": .003}],
                            "sim_time": float(times[i])}]} for i in range(ticks)],
        "geometry_checks": [{"pass": True, "max_finite_penetration_m": 0.} for _ in range(ticks)],
        "snapshot_rng_states": [{"fixture_rng_state": 111 + i} for i in range(ticks + 1)],
        "terminal_reason": "collection_complete",
    }
    return Episode(arrays, metadata)


def slice_episode(episode, first, last):
    """Select action indices [first,last), retaining both boundary endpoints."""
    arrays = {}
    for key, values in episode.arrays.items():
        if key in BOUNDARY_KEYS:
            arrays[key] = values[first:last + 1].copy()
        elif key in ("substep_states", "substep_times"):
            arrays[key] = values[2 * first:2 * last].copy()
        elif key == "substep_offsets":
            arrays[key] = (values[first:last + 1] - values[first]).copy()
        else:
            arrays[key] = values[first:last].copy()
    metadata = deepcopy(episode.metadata)
    metadata["diagnostics"] = deepcopy(episode.metadata["diagnostics"][first:last])
    metadata["geometry_checks"] = deepcopy(episode.metadata["geometry_checks"][first:last])
    metadata["snapshot_rng_states"] = deepcopy(episode.metadata["snapshot_rng_states"][first:last + 1])
    return Episode(arrays, metadata)


class ExactPathTests(unittest.TestCase):
    def setUp(self):
        self.full = fake_episode()
        self.warm = slice_episode(self.full, 0, 2)
        self.core = slice_episode(self.full, 2, 22)

    def test_combined_path_equals_segmented_path_without_double_counting_seam(self):
        result = exact_paths([self.warm, self.core], [self.full])
        self.assertTrue(result["pass"], result)
        # Episode bookkeeping differs legitimately after segmentation.
        self.core.metadata["terminal_reason"] = "different_provenance_only"
        self.assertTrue(exact_paths([self.warm, self.core], [self.full])["pass"])

    def test_raw_contact_change_fails_even_if_summary_and_endpoints_match(self):
        changed = deepcopy(self.full)
        changed.metadata["diagnostics"][8]["samples"][0]["contacts"][0]["normal_force_N"] = .003
        np.testing.assert_array_equal(changed.arrays["state"][[0, -1]], self.full.arrays["state"][[0, -1]])
        np.testing.assert_array_equal(changed.arrays["contact_summary"], self.full.arrays["contact_summary"])
        result = exact_paths([self.warm, self.core], [changed])
        self.assertFalse(result["pass"])
        self.assertFalse(result["checks"]["diagnostics"])

    def test_substep_excursion_fails_even_if_every_action_boundary_matches(self):
        changed = deepcopy(self.full)
        changed.arrays["substep_states"][11, 0] += .002
        np.testing.assert_array_equal(changed.arrays["state"], self.full.arrays["state"])
        result = exact_paths([self.warm, self.core], [changed])
        self.assertFalse(result["pass"])
        self.assertFalse(result["checks"]["substep_states"])

    def test_snapshot_internal_state_warning_and_rng_changes_fail(self):
        for key in ("raw_integration_state", "raw_warning_counts", "raw_warning_lastinfo"):
            with self.subTest(key=key):
                changed = deepcopy(self.full)
                changed.arrays[key][8, 0] += 1
                result = exact_paths([self.warm, self.core], [changed])
                self.assertFalse(result["pass"])
                self.assertFalse(result["checks"][key])
        changed = deepcopy(self.full)
        changed.metadata["snapshot_rng_states"][8]["fixture_rng_state"] += 1
        result = exact_paths([self.warm, self.core], [changed])
        self.assertFalse(result["pass"])
        self.assertFalse(result["checks"]["snapshot_rng_states"])

    def test_time_applied_action_and_validity_mismatch_fail(self):
        for key, index, value in (("substep_times", 11, .123),
                                   ("sim_time", 8, .123),
                                   ("action_applied", (8, 1), .012),
                                   ("transition_valid", 8, False)):
            with self.subTest(key=key):
                changed = deepcopy(self.full)
                changed.arrays[key][index] = value
                result = exact_paths([self.warm, self.core], [changed])
                self.assertFalse(result["pass"])
                self.assertFalse(result["checks"][key])

    def test_discontinuous_segment_start_snapshot_is_not_dropped_silently(self):
        for field in ("raw_integration_state", "rng"):
            with self.subTest(field=field):
                changed_core = deepcopy(self.core)
                if field == "rng":
                    changed_core.metadata["snapshot_rng_states"][0]["fixture_rng_state"] += 1
                else:
                    changed_core.arrays[field][0, 0] += 1
                result = exact_paths([self.warm, changed_core], [self.full])
                self.assertFalse(result["pass"], "A changed seam snapshot must not disappear during concatenation")


class DtComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_pilot_config()
        cls.profile = tomllib.loads((ROOT / "configs/acceptance.v1.toml").read_text())

    def setUp(self):
        self.fine = fake_episode()
        self.warm = slice_episode(self.fine, 0, 2)
        self.core = slice_episode(self.fine, 2, 22)

    def compare(self):
        return compare_dt(self.warm, self.core, self.fine, self.profile, self.config)

    def test_matching_complete_trajectory_passes_original_thresholds(self):
        result = self.compare()
        self.assertTrue(result["pass"], result)
        self.assertEqual(result["budgets"]["object_xy_m"], .0005)
        self.assertAlmostEqual(result["budgets"]["object_yaw_rad"], np.deg2rad(.3))

    def test_interior_action_boundary_excursion_fails_even_if_endpoint_matches(self):
        endpoint = self.fine.arrays["state"][-1].copy()
        self.fine.arrays["state"][9, 0] = .0006
        np.testing.assert_array_equal(endpoint, self.fine.arrays["state"][-1])
        result = self.compare()
        self.assertFalse(result["pass"])
        self.assertFalse(result["checks"]["object_xy_budget"])
        self.assertEqual(result["max_action_boundary_differences"]["object_xy_m"], .0006)

    def test_interior_yaw_excursion_fails_and_wrapped_equivalence_passes(self):
        self.fine.arrays["state"][9, 2] = np.deg2rad(.31)
        result = self.compare()
        self.assertFalse(result["pass"])
        self.assertFalse(result["checks"]["object_yaw_budget"])
        self.fine.arrays["state"][9, 2] = 2 * np.pi
        self.assertTrue(self.compare()["pass"])

    def test_different_starting_integration_state_fails(self):
        self.fine.arrays["raw_integration_state"][0, 3] += 1
        result = self.compare()
        self.assertFalse(result["pass"])
        self.assertFalse(result["checks"]["same_initial_integration_state"])

    def test_action_boundary_time_mismatch_fails(self):
        self.fine.arrays["sim_time"][8] += 1e-6
        result = self.compare()
        self.assertFalse(result["pass"])
        self.assertFalse(result["checks"]["same_action_boundaries"])

    def test_requested_action_mismatch_fails_even_if_states_match(self):
        self.fine.arrays["action_requested"][8, 1] += .001
        result = self.compare()
        self.assertFalse(result["pass"])
        self.assertFalse(result["checks"]["same_requested_actions"])

    def test_invalid_transition_in_any_branch_fails(self):
        for branch in ("warm", "core", "fine"):
            with self.subTest(branch=branch):
                episode = getattr(self, branch)
                episode.arrays["transition_valid"][0] = False
                result = self.compare()
                self.assertFalse(result["pass"])
                self.assertFalse(result["checks"]["all_valid"])
                episode.arrays["transition_valid"][0] = True

    def test_incomplete_fine_trajectory_fails(self):
        self.fine = slice_episode(self.fine, 0, 21)
        result = self.compare()
        self.assertFalse(result["pass"])
        self.assertFalse(result["checks"]["complete_22"])

    def test_contact_and_pusher_differences_are_reported_without_inventing_thresholds(self):
        self.fine.arrays["contact_summary"][8, 1] = 1.
        self.fine.arrays["state"][8, 6] += .003
        self.fine.arrays["state"][8, 9] += .002
        result = self.compare()
        self.assertTrue(result["pass"], result)
        self.assertEqual(result["contact_boundary_disagreements"], 1)
        self.assertEqual(result["max_action_boundary_differences"]["pusher_xy_m"], .003)
        self.assertAlmostEqual(result["max_action_boundary_differences"]["pusher_velocity_xy_m_s"], .002)


if __name__ == "__main__":
    unittest.main()
