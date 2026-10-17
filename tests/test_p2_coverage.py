"""Coverage semantics, failure masking, and real-physics contract integration."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np

from push_align import InitialSpec, PlanarPushEnv
from push_align.coverage import REGIMES, _contact_events, episode_coverage, transition_regimes
from push_align.data import DataContractError, Episode, EpisodeRecorder, make_contract


def sample(time, active, *, pair="pusher_object", force=None):
    return {"sim_time": time, "contacts": [] if active is None else [
        {"pair": pair, "normal_force_N": (0.002 if active else 0.0) if force is None else force}]}


def fixture(patterns, *, valid=None, states=None):
    """Calculator-only fixture; full validator integration is tested separately."""
    count = len(patterns)
    diagnostics = [{"samples": [sample(index+offset/(len(pattern)-1), active)
                                for offset, active in enumerate(pattern)]}
                   for index, pattern in enumerate(patterns)]
    return Episode({"actual_dt": np.ones(count),
                    "transition_valid": np.array(valid if valid is not None else [True]*count),
                    "state": np.zeros((count+1, 10)) if states is None else states,
                    "diagnostic_nonfinite_values": np.array([], dtype=np.float64)},
                   {"episode_id": "calculator", "family_id": "family-calculator", "split": "validation",
                    "collection_policy": {"kind": "synthetic-semantics"}, "terminal_reason": "timeout",
                    "contract": {"active_contact_force_min_N": .001, "dt_tolerance_s": 1e-9,
                                 "resolved_config": {"evaluation": {
                                     "recontact_absence_min_s": .04,
                                     "recontact_presence_min_s": .01}}},
                    "diagnostics": diagnostics, "diagnostic_nonfinite_paths": []})


def summarize_calculator(episode, **kwargs):
    valid = episode.arrays["transition_valid"]
    review = {"transitions": len(valid), "learning_transitions": int(valid.sum()),
              "excluded_transitions": int(len(valid)-valid.sum())}
    with patch("push_align.coverage.validate_episode", return_value=review) as validation:
        result = episode_coverage(episode, {}, position_signal_m=.001,
                                  yaw_signal_rad=.01, **kwargs)
        validation.assert_called_once_with(episode, {})
    return result


class CoverageSemanticsTest(unittest.TestCase):
    def test_regimes_use_designated_pair_strict_threshold_and_raw_pattern(self):
        episode = fixture([[False, False], [True, True], [False, True]])
        self.assertEqual(transition_regimes(episode), list(REGIMES))
        for contact_sample in episode.metadata["diagnostics"][0]["samples"]:
            contact_sample["contacts"] = [{"pair": "floor_object", "normal_force_N": 99.},
                                          {"pair": "pusher_object", "normal_force_N": .001}]
        self.assertEqual(transition_regimes(episode)[0], "noncontact")

    def test_raw_nonfinite_value_is_restored_and_not_a_false_crossing(self):
        episode = fixture([[False, False]])
        episode.metadata["diagnostics"][0]["samples"][1]["contacts"][0]["normal_force_N"] = None
        episode.metadata["diagnostic_nonfinite_paths"] = [[0, "samples", 1, "contacts", 0, "normal_force_N"]]
        episode.arrays["diagnostic_nonfinite_values"] = np.array([np.nan])
        self.assertEqual(transition_regimes(episode), ["noncontact"])
        result = summarize_calculator(episode)
        self.assertEqual(result["contact_events"]["unknown_force_sample_count"], 1)
        self.assertEqual(result["contact_events"]["raw_crossing_count"], 0)

    def test_windows_are_contiguous_and_first_label_does_not_describe_future(self):
        episode = fixture([[False, False], [False, True], [True, True],
                           [True, True], [True, False]], valid=[True, True, True, True, False])
        result = summarize_calculator(episode, horizons=(1, 2, 5, 10))
        self.assertEqual(result["recorded_transitions"], 5)
        self.assertEqual(result["valid_transitions"], 4)
        self.assertEqual(result["excluded_transitions"], 1)
        self.assertEqual(result["regime_counts_all"], {"noncontact": 1, "sustained_contact": 2, "transition": 2})
        self.assertEqual(result["regime_counts_valid"], {"noncontact": 1, "sustained_contact": 2, "transition": 1})
        self.assertEqual(result["windows"]["2"]["start_indices"], [0, 1, 2])
        self.assertEqual(result["windows"]["2"]["start_indices_by_first_regime"],
                         {"noncontact": [0], "sustained_contact": [2], "transition": [1]})
        self.assertEqual(result["windows"]["5"]["total"], 0)
        self.assertEqual(result["windows"]["10"]["total"], 0)
        # Calculator must not join data across a hole even if given such a mask.
        # The real episode validator additionally rejects interior invalid rows.
        episode.arrays["transition_valid"] = np.array([True, False, True, True, False])
        self.assertEqual(summarize_calculator(episode, horizons=(2,))["windows"]["2"]["start_indices"], [2])

    def test_motion_uses_valid_boundaries_and_2pi_wrap_not_task_symmetry(self):
        states = np.zeros((4, 10))
        states[:, 0] = [0., .002, .003, 100.]
        states[:, 2] = [np.pi-.005, -np.pi+.005, -np.pi+.015, 0.]
        episode = fixture([[False, True], [True, True], [True, False]],
                          valid=[True, True, False], states=states)
        result = summarize_calculator(episode)
        self.assertAlmostEqual(result["max_position_excursion_m"], .003)
        self.assertAlmostEqual(result["max_yaw_excursion_rad"], .02)
        self.assertEqual(result["valid_boundary_indices"], [0, 1, 2])
        self.assertTrue(result["translation_signal"])
        self.assertTrue(result["rotation_signal"])

    def test_motion_without_active_valid_transition_does_not_claim_contact_signal(self):
        states = np.zeros((3, 10)); states[1, 0] = .1; states[1, 2] = .2
        episode = fixture([[False, False], [False, True]], valid=[True, False], states=states)
        result = summarize_calculator(episode)
        self.assertGreater(result["max_position_excursion_m"], .001)
        self.assertFalse(result["valid_contact_present"])
        self.assertFalse(result["translation_signal"])
        self.assertFalse(result["rotation_signal"])

    def test_shared_boundary_is_deduplicated_and_short_chatter_is_not_recontact(self):
        diagnostics = [{"samples": [sample(.00, True), sample(.01, False), sample(.02, True)]},
                       {"samples": [sample(.02, True), sample(.03, True), sample(.04, False)]}]
        result = _contact_events(diagnostics, .001, 1e-9)
        self.assertEqual(result["deduplicated_raw_sample_count"], 5)
        self.assertEqual(result["entry_times_s"], [.02])
        self.assertEqual(result["exit_times_s"], [.01, .04])
        self.assertEqual(result["raw_crossing_count"], 3)
        self.assertEqual(result["qualified_recontacts"], [])

    def test_recontact_requires_observed_inactive_and_resumed_active_spans(self):
        samples = [sample(.00, True), sample(.01, False), sample(.03, False),
                   sample(.05, False), sample(.06, True), sample(.07, True)]
        result = _contact_events([{"samples": samples}], .001, 1e-9)
        self.assertEqual(len(result["qualified_recontacts"]), 1)
        event = result["qualified_recontacts"][0]
        self.assertAlmostEqual(event["observed_inactive_span_s"], .04)
        self.assertAlmostEqual(event["observed_resumed_active_span_s"], .01)
        # Ending immediately after resumption does not establish active dwell.
        self.assertEqual(_contact_events([{"samples": samples[:-1]}], .001, 1e-9)["qualified_recontacts"], [])

    def test_unknown_force_breaks_recontact_chain(self):
        samples = [sample(.00, True), sample(.01, False), sample(.05, False),
                   sample(.055, True, force=np.inf), sample(.06, True), sample(.07, True)]
        result = _contact_events([{"samples": samples}], .001, 1e-9)
        self.assertEqual(result["qualified_recontacts"], [])
        self.assertEqual(result["entry_times_s"], [])
        self.assertEqual(result["exit_times_s"], [.01])

    def test_episode_recontact_thresholds_come_from_contract_and_are_validated(self):
        episode = fixture([[True, False, False], [False, False, True], [True, True, True]])
        result = summarize_calculator(episode)
        self.assertEqual(result["qualified_recontact_count"], 1)
        evaluation = episode.metadata["contract"]["resolved_config"]["evaluation"]
        for key, output in (("recontact_absence_min_s", "inactive_min_s"),
                            ("recontact_presence_min_s", "resumed_active_min_s")):
            previous = evaluation[key]
            evaluation[key] = 1.1
            result = summarize_calculator(episode)
            self.assertEqual(result["qualified_recontact_count"], 0)
            self.assertEqual(result["contact_events"][output], 1.1)
            for invalid in (0., -1., np.nan, np.inf, True):
                evaluation[key] = invalid
                with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                    summarize_calculator(episode)
            evaluation[key] = previous

    def test_bad_horizons_and_signal_thresholds_are_rejected(self):
        episode = fixture([[False, False]])
        for horizons in ((), (0,), (1, 1), (True,), (1.5,)):
            with self.subTest(horizons=horizons), self.assertRaises(ValueError):
                episode_coverage(episode, {}, position_signal_m=.001,
                                 yaw_signal_rad=.01, horizons=horizons)
        for threshold in (0., -1., np.nan, np.inf, True):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                episode_coverage(episode, {}, position_signal_m=threshold,
                                 yaw_signal_rad=.01)


class CoveragePhysicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = PlanarPushEnv(); cls.contract = make_contract(env)
        recorder = EpisodeRecorder(env, episode_id="coverage-real", episode_index=0,
                                   family_id="family-real", split="validation",
                                   family_definition={"base_initial_spec": {"object_x": 0.},
                                                      "variant_rule": "exact physics fixture"},
                                   collection_policy={"kind": "central_push"}, seed=13,
                                   contract=cls.contract)
        for _ in range(12):
            recorder.step([.03, 0.])
            if recorder.invalid:
                raise AssertionError("Coverage physical fixture unexpectedly invalid")
        cls.episode = recorder.finish()

    def test_real_episode_labels_windows_motion_and_strict_json(self):
        before = self.episode.arrays["state"].copy()
        result = episode_coverage(self.episode, self.contract,
                                  position_signal_m=.001, yaw_signal_rad=.01)
        self.assertEqual(result["mode"], "central_push")
        self.assertEqual(result["recorded_transitions"], 12)
        self.assertEqual(result["valid_transitions"], 12)
        self.assertEqual(result["windows"]["5"]["total"], 8)
        self.assertEqual(result["windows"]["10"]["total"], 3)
        self.assertTrue(all(count > 0 for count in result["regime_counts_valid"].values()))
        self.assertTrue(result["translation_signal"])
        self.assertFalse(result["rotation_signal"])
        self.assertEqual(result["contact_events"]["deduplicated_raw_sample_count"], 2401)
        self.assertEqual(result["contact_events"]["unknown_force_sample_count"], 0)
        json.dumps(result, allow_nan=False)
        np.testing.assert_array_equal(self.episode.arrays["state"], before)

    def test_stored_mask_cannot_skip_raw_validation(self):
        episode = Episode({key: value.copy() for key, value in self.episode.arrays.items()},
                          deepcopy(self.episode.metadata))
        episode.arrays["transition_valid"][0] = False
        with self.assertRaisesRegex(DataContractError, "Learning mask"):
            episode_coverage(episode, self.contract,
                             position_signal_m=.001, yaw_signal_rad=.01)


if __name__ == "__main__":
    unittest.main()
