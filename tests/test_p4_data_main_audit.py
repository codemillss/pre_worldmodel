"""Adversarial checks of audit arithmetic; no environment or model is executed."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

from push_align.data import Episode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from check_p4_data_main import (contact_transition_mask, core_window_statistics,
                                slot_design_from_schedule)


class MainAuditTest(unittest.TestCase):
    def test_historical_schedule_regenerates_frozen384_slots(self):
        with np.load(ROOT / "runs/p4-02-20261012-01/seeds/seed-11/SK-lambda0.1/training_trace.npz",
                     allow_pickle=False) as archive:
            actual = slot_design_from_schedule(archive["window_positions"], archive["batch_offsets"])
        expected = json.loads((ROOT / "reports/p4_data_intervention_slots.design.json").read_text())["entries"]
        self.assertEqual(actual, expected)

    def test_malformed_schedule_denominators_cannot_pass(self):
        with np.load(ROOT / "runs/p4-02-20261012-01/seeds/seed-11/SK-lambda0.1/training_trace.npz",
                     allow_pickle=False) as archive:
            positions, offsets = archive["window_positions"], archive["batch_offsets"]
        broken_offsets = offsets.copy(); broken_offsets[3] += 1
        with self.assertRaises(ValueError):
            slot_design_from_schedule(positions, broken_offsets)
        negative = positions.copy(); negative[0] = -1
        with self.assertRaises(ValueError):
            slot_design_from_schedule(negative, offsets)

    def test_contact_requires_correct_pair_and_strict_threshold(self):
        def sample(pair, force):
            return {"contacts": [{"pair": pair, "normal_force_N": force}]}
        episode = Episode({}, {"diagnostics": [
            {"samples": [sample("floor_object", 10.), sample("pusher_object", .001)]},
            {"samples": [sample("pusher_object", 0.), sample("pusher_object", .001000001)]},
            {"samples": [{"contacts": []}]}]})
        np.testing.assert_array_equal(contact_transition_mask(episode, .001), [False, True, False])

    def test_overlap_zero_mass_matches_but_continuous_windows_differ(self):
        for pattern, starts in (("MMMMMZMZMZMZMZMMMMMM", []), ("MMMMMMMMMMZZZZZMMMMM", [10])):
            episode = Episode({"action_applied": np.asarray([[0., .02] if c == "M" else [0., 0.] for c in pattern]),
                               "actual_dt": np.full(20, .1), "transition_valid": np.ones(20, dtype=bool)}, {})
            stats = core_window_statistics(episode)
            self.assertEqual(stats, {"valid_starts": list(range(16)), "zero_command_count": 5,
                                     "zero_occurrences": 25, "all_zero_starts": starts})

    def test_invalid_stop_transition_removes_overlap_windows(self):
        pattern = "MMMMMMMMMMZZZZZMMMMM"
        valid = np.ones(20, dtype=bool); valid[12] = False
        episode = Episode({"action_applied": np.asarray([[0., .02] if c == "M" else [0., 0.] for c in pattern]),
                           "actual_dt": np.full(20, .1), "transition_valid": valid}, {})
        stats = core_window_statistics(episode)
        self.assertEqual(stats["all_zero_starts"], [])
        self.assertEqual(len(stats["valid_starts"]), 11)
        self.assertFalse(set(range(8, 13)) & set(stats["valid_starts"]))


if __name__ == "__main__":
    unittest.main()
