"""Train-only materialization and global final-update selection safeguards."""
from copy import deepcopy
import json
import unittest

import numpy as np
import torch

from push_align.data import DataContractError
from push_align.fork_training import choose_lambda, train_sequences
from push_align.preparation import PreparedData


def prepared_fixture():
    current, future, actions, windows = [], [], [], []
    for episode in range(2):
        trajectory = np.arange(11, dtype=np.float64)[:, None] * np.arange(1, 11, dtype=np.float64)[None, :] / 1000 + episode / 10
        current.extend(trajectory[:-1])
        future.extend(trajectory[1:])
        actions.extend(np.column_stack((np.arange(10) / 1000, -np.arange(10) / 2000)))
        windows.extend(np.arange(episode * 10 + start, episode * 10 + start + 5) for start in range(6))
    arrays = {"current_state": np.array(current, np.float64),
              "next_state": np.array(future, np.float64),
              "applied_action": np.array(actions, np.float64),
              "split_code": np.repeat([0, 1], 10).astype(np.int8),
              "regime_code": np.tile(np.arange(10) % 3, 2).astype(np.int8),
              "episode_index": np.repeat([0, 1], 10).astype(np.int64),
              "transition_index": np.tile(np.arange(10), 2).astype(np.int64)}
    return PreparedData(arrays, {}, {5: np.array(windows, np.int64)})


def budget():
    return {"seeds": [11, 23, 47], "lambda_candidates": [.1, 1.],
            "additional_updates_per_branch": 1000}


def records(small=(1., 1., 1.), large=(2., 2., 2.)):
    return [{"seed": seed, "lambda": weight, "additional_updates": 1000,
             "selection_status": "eligible", "selection_score": score,
             "K5_invalid_windows": 0}
            for weight, scores in ((.1, small), (1., large))
            for seed, score in zip((11, 23, 47), scores)]


class P4ForkTrainingTest(unittest.TestCase):
    def test_train_sequences_use_recorded_applied_actions_and_initial_plus_five_next_states(self):
        prepared = prepared_fixture()
        windows, states, actions, mask = train_sequences(prepared)
        np.testing.assert_array_equal(windows, np.array([np.arange(start, start + 5) for start in range(6)]))
        self.assertEqual(states.shape, (6, 6, 10))
        self.assertEqual(actions.shape, (6, 5, 2))
        self.assertEqual(mask.shape, (6, 5))
        self.assertEqual((states.dtype, actions.dtype, mask.dtype), (torch.float64, torch.float64, torch.bool))
        self.assertEqual((states.device.type, actions.device.type, mask.device.type), ("cpu", "cpu", "cpu"))
        self.assertTrue(mask.all())
        for index, window in enumerate(windows):
            np.testing.assert_array_equal(states[index, 0].numpy(), prepared.arrays["current_state"][window[0]])
            np.testing.assert_array_equal(states[index, 1:].numpy(), prepared.arrays["next_state"][window])
            np.testing.assert_array_equal(actions[index].numpy(), prepared.arrays["applied_action"][window])

    def test_finite_validation_perturbation_cannot_change_training_sequences(self):
        original = prepared_fixture()
        modified = deepcopy(original)
        for key in ("current_state", "next_state", "applied_action"):
            modified.arrays[key][10:] = modified.arrays[key][10:] * 11 + .25
        first, second = train_sequences(original), train_sequences(modified)
        np.testing.assert_array_equal(first[0], second[0])
        for a, b in zip(first[1:], second[1:]):
            self.assertTrue(torch.equal(a, b))

    def test_train_perturbation_positive_control_changes_only_expected_outputs(self):
        original = prepared_fixture()
        modified = deepcopy(original)
        modified.arrays["current_state"][:10] += .25
        modified.arrays["next_state"][:10] += .25
        modified.arrays["applied_action"][:10] += .01
        first, second = train_sequences(original), train_sequences(modified)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertTrue(torch.equal(first[3], second[3]))
        torch.testing.assert_close(second[1] - first[1], torch.full_like(first[1], .25), rtol=0, atol=1e-16)
        torch.testing.assert_close(second[2] - first[2], torch.full_like(first[2], .01), rtol=0, atol=1e-17)

    def test_returned_training_sequences_never_alias_prepared_columns_or_windows(self):
        prepared = prepared_fixture()
        untouched = deepcopy(prepared)
        windows, states, actions, mask = train_sequences(prepared)
        windows.fill(-1)
        states.fill_(999)
        actions.fill_(999)
        mask.fill_(False)
        for key in prepared.arrays:
            np.testing.assert_array_equal(prepared.arrays[key], untouched.arrays[key])
        np.testing.assert_array_equal(prepared.windows[5], untouched.windows[5])

    def test_invalid_time_gap_split_crossing_and_truth_discontinuity_rejected(self):
        for alteration in ("gap", "split", "truth", "duplicate"):
            prepared = prepared_fixture()
            if alteration == "gap":
                prepared.arrays["transition_index"][2] += 1
            elif alteration == "split":
                prepared.arrays["split_code"][2] = 1
            elif alteration == "truth":
                prepared.arrays["next_state"][1, 0] += .01
            else:
                prepared.windows[5][1] = prepared.windows[5][0]
            with self.subTest(alteration=alteration), self.assertRaises(DataContractError):
                train_sequences(prepared)

    def test_no_original_train_windows_rejected(self):
        prepared = prepared_fixture()
        prepared.arrays["split_code"][:] = 1
        with self.assertRaises(ValueError):
            train_sequences(prepared)

    def test_global_seed_mean_prevents_best_seed_selection(self):
        result = choose_lambda(records((0., 12., 12.), (4., 4., 4.)), budget())
        self.assertTrue(result["pass"])
        self.assertEqual(result["selected_lambda"], 1.)
        self.assertEqual([candidate["mean_final_validation_K5_score"] for candidate in result["candidates"]], [8., 4.])

    def test_zero_score_is_valid_and_not_confused_with_missing_score(self):
        result = choose_lambda(records((0., 0., 0.), (1., 1., 1.)), budget())
        self.assertTrue(result["pass"])
        self.assertEqual(result["selected_lambda"], .1)
        self.assertTrue(result["candidates"][0]["eligible"])
        self.assertEqual(result["candidates"][0]["mean_final_validation_K5_score"], 0.)

    def test_one_invalid_seed_disqualifies_entire_lambda(self):
        data = records((0., 0., 0.), (5., 5., 5.))
        data[2]["K5_invalid_windows"] = 1
        result = choose_lambda(data, budget())
        self.assertEqual(result["selected_lambda"], 1.)
        candidate = result["candidates"][0]
        self.assertFalse(candidate["eligible"])
        self.assertIsNone(candidate["mean_final_validation_K5_score"])
        self.assertEqual(candidate["reasons"], [{"seed": 47, "reason": "invalid_K5_windows"}])

    def test_all_invalid_lambdas_fail_without_fallback_or_best_seed(self):
        data = records()
        data[0]["selection_status"] = "invalid_selection_windows"
        data[4]["K5_invalid_windows"] = 1
        result = choose_lambda(data, budget())
        self.assertFalse(result["pass"])
        self.assertIsNone(result["selected_lambda"])
        self.assertTrue(all(not candidate["eligible"] for candidate in result["candidates"]))

    def test_exact_tie_selects_smaller_lambda_independent_of_record_order(self):
        data = records((1., 2., 3.), (3., 2., 1.))
        data.reverse()
        settings = budget()
        settings["lambda_candidates"].reverse()
        result = choose_lambda(data, settings)
        self.assertEqual(result["selected_lambda"], .1)
        self.assertEqual([candidate["mean_final_validation_K5_score"] for candidate in result["candidates"]], [2., 2.])

    def test_partial_or_excess_updates_cannot_replace_registered_final_weights(self):
        for update in (0, 999, 1001):
            data = records((0., 0., 0.), (2., 2., 2.))
            data[0].update(additional_updates=update, checkpoint_epoch=1, intermediate_best=True)
            with self.subTest(update=update):
                result = choose_lambda(data, budget())
                self.assertEqual(result["selected_lambda"], 1.)
                self.assertIn({"seed": 11, "reason": "not_registered_final_additional_update"}, result["candidates"][0]["reasons"])

    def test_absent_nonfinite_negative_or_boolean_score_makes_lambda_ineligible(self):
        for score in (None, float("nan"), float("inf"), -1., True, "0", 10**1000):
            data = records((0., 0., 0.), (2., 2., 2.))
            data[0]["selection_score"] = score
            with self.subTest(score_type=type(score).__name__):
                result = choose_lambda(data, budget())
                self.assertEqual(result["selected_lambda"], 1.)
                self.assertIsNone(result["candidates"][0]["seed_records"][0]["selection_score"])
                json.dumps(result, allow_nan=False)
        data = records()
        del data[0]["selection_score"]
        self.assertEqual(choose_lambda(data, budget())["selected_lambda"], 1.)

    def test_missing_duplicate_extra_and_unknown_seed_lambda_records_rejected(self):
        cases = []
        data = records(); cases.append(data[:-1])
        data = records(); cases.append(data + [deepcopy(data[0])])
        data = records(); data[0]["seed"] = 12; cases.append(data)
        data = records(); data[0]["lambda"] = .5; cases.append(data)
        for data in cases:
            with self.assertRaises(ValueError):
                choose_lambda(data, budget())

    def test_identifier_and_counter_wrong_types_rejected(self):
        for field, value in (("seed", "11"), ("seed", 11.), ("seed", True),
                             ("lambda", "0.1"), ("lambda", 1), ("lambda", True),
                             ("additional_updates", 1000.), ("additional_updates", True),
                             ("K5_invalid_windows", 0.), ("K5_invalid_windows", True),
                             ("K5_invalid_windows", -1)):
            data = records(); data[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                choose_lambda(data, budget())

    def test_finite_large_scores_do_not_overflow_global_mean(self):
        maximum = np.finfo(np.float64).max.item()
        result = choose_lambda(records((maximum,) * 3, (maximum / 2,) * 3), budget())
        self.assertEqual(result["selected_lambda"], 1.)
        self.assertTrue(all(np.isfinite(candidate["mean_final_validation_K5_score"]) for candidate in result["candidates"]))

    def test_selection_does_not_mutate_records_or_frozen_budget(self):
        data, settings = records(), budget()
        before_data, before_settings = deepcopy(data), deepcopy(settings)
        result = choose_lambda(data, settings)
        result["candidates"][0]["seed_records"][0]["seed"] = 999
        self.assertEqual(data, before_data)
        self.assertEqual(settings, before_settings)

    def test_frozen_population_configuration_errors_rejected(self):
        for replacement in ({"seeds": [11, 11, 47]}, {"seeds": [11., 23, 47]},
                            {"lambda_candidates": [.1, .1]}, {"lambda_candidates": [.1, 1]},
                            {"lambda_candidates": [.1, float("nan")]},
                            {"additional_updates_per_branch": True}):
            settings = budget(); settings.update(replacement)
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                choose_lambda(records(), settings)


if __name__ == "__main__":
    unittest.main()
