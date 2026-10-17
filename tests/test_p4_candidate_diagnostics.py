"""Known candidate orders, numerical ambiguity, and invalid-selection traces."""
from copy import deepcopy
import json
import math
import unittest

import numpy as np

from push_align.candidate_diagnostics import DiagnosticError, compare_candidates


def fixture(prediction=(1., 2., 3.), truth=(1., 2., 3.)):
    predicted = np.array(prediction, np.float64)
    actual = np.repeat(np.array(truth, np.float64)[:, None], 3, axis=1)
    return predicted, np.ones(len(predicted), np.bool_), actual, np.ones(actual.shape, np.bool_)


def compare(*args, **kwargs):
    return compare_candidates(*args, atol=0., rtol=0., **kwargs)


class P4CandidateDiagnosticsTest(unittest.TestCase):
    def test_known_cost_errors_selection_and_concordant_order(self):
        result = compare(*fixture((0., 1., 3.), (0., 1., 2.)))
        self.assertEqual(result["selected_index"], 0)
        self.assertEqual(result["actual_best_index"], 0)
        self.assertEqual(result["selection_loss"], 0.)
        self.assertEqual(result["selection_loss_denominator"], 1)
        self.assertTrue(result["selected_is_actual_mean_best"])
        self.assertAlmostEqual(result["error_metrics"]["MAE"], 1/3)
        self.assertAlmostEqual(result["error_metrics"]["RMSE"], math.sqrt(1/3))
        self.assertEqual(result["error_metrics"]["denominator"], 3)
        self.assertEqual(result["pairwise"]["concordant"], 3)
        self.assertEqual(result["pairwise"]["concordance_rate"], 1.)

    def test_known_single_inversion_retains_all_pair_denominators(self):
        result = compare(*fixture((2., 1., 3., 4.), (1., 2., 3., 4.)))
        self.assertEqual(result["selected_index"], 1)
        self.assertEqual(result["selection_loss"], 1.)
        self.assertEqual(result["selection_loss_ordering"], "SELECTED_WORSE")
        pairs = result["pairwise"]
        self.assertEqual((pairs["total_pairs"], pairs["common_valid_pairs"], pairs["concordant"], pairs["discordant"]), (6, 6, 5, 1))
        self.assertEqual(pairs["concordance_rate"], 5/6)

    def test_changing_actual_futures_never_changes_predicted_selection(self):
        inputs = fixture((2., 1., 3.), (1., 2., 3.))
        first = compare(*inputs)
        changed = list(deepcopy(inputs))
        changed[2] = np.repeat(np.array([3., .1, 2.])[:, None], 3, axis=1)
        second = compare(*changed)
        self.assertEqual(first["selected_index"], second["selected_index"])
        self.assertNotEqual(first["actual_best_index"], second["actual_best_index"])
        self.assertNotEqual(first["selection_loss"], second["selection_loss"])

    def test_minimum_prediction_tie_selects_lowest_original_index(self):
        inputs = list(fixture((0., 1., 1., 3.), (0., 2., 1., 3.)))
        inputs[1][0] = False
        result = compare(*inputs)
        self.assertEqual(result["selected_index"], 1)
        self.assertEqual(result["actual_best_index"], 0)
        self.assertEqual(result["selection_loss"], 2.)

    def test_predicted_tie_is_not_concordance_when_actual_order_is_separated(self):
        result = compare(*fixture((3., 3.), (1., 2.)))
        pairs = result["pairwise"]
        self.assertEqual(pairs["predicted_tie"], 1)
        self.assertEqual(pairs["strict_comparison_pairs"], 0)
        self.assertIsNone(pairs["concordance_rate"])
        self.assertEqual(result["selected_index"], 0)

    def test_actual_equal_cost_tie_is_inconclusive_even_if_prediction_ties(self):
        result = compare(*fixture((1., 1.), (2., 2.)))
        self.assertEqual(result["pairwise"]["actual_inconclusive"], 1)
        self.assertEqual(result["pairwise"]["predicted_tie"], 0)
        self.assertEqual(result["actual_best_index"], 0)
        self.assertEqual(result["selection_loss"], 0.)

    def test_close_apparent_improvement_is_inconclusive_but_raw_loss_survives(self):
        inputs = list(fixture((2., 1.), (1., 1.15)))
        inputs[2] = np.array([[.9, 1., 1.1], [1.05, 1.15, 1.25]], np.float64)
        result = compare(*inputs)
        self.assertEqual(result["selected_index"], 1)
        self.assertAlmostEqual(result["selection_loss"], .15)
        self.assertEqual(result["selection_loss_ordering"], "INCONCLUSIVE")
        self.assertEqual(result["pairwise"]["actual_inconclusive"], 1)
        self.assertEqual(result["pairwise"]["discordant"], 0)

    def test_touching_intervals_overlap_and_slightly_separated_intervals_order(self):
        result = compare_candidates(*fixture((2., 1.), (1., 2.)), atol=.5, rtol=0.)
        self.assertEqual(result["actual_records"][0]["cost_interval"], [.5, 1.5])
        self.assertEqual(result["actual_records"][1]["cost_interval"], [1.5, 2.5])
        self.assertEqual(result["pairwise"]["actual_inconclusive"], 1)
        self.assertEqual(result["selection_loss_ordering"], "INCONCLUSIVE")
        separated = compare_candidates(*fixture((2., 1.), (1., 2.000001)), atol=.5, rtol=0.)
        self.assertEqual(separated["pairwise"]["discordant"], 1)
        self.assertEqual(separated["selection_loss_ordering"], "SELECTED_WORSE")

    def test_registered_relative_tolerance_uses_maximum_absolute_replay_cost(self):
        inputs = list(fixture((2.,), (2.,)))
        inputs[2][0] = [2., 3., 4.]
        result = compare_candidates(*inputs, atol=.25, rtol=.1)
        record = result["actual_records"][0]
        self.assertEqual(record["epsilon"], .65)
        self.assertEqual(record["cost_interval"], [1.35, 4.65])
        self.assertEqual(record["mean_cost"], 3.)
        self.assertEqual(record["observed_cost_range"], [2., 4.])
        self.assertEqual(record["observed_cost_range_width"], 2.)

    def test_spearman_uses_average_exact_tie_ranks_with_known_correlation(self):
        result = compare(*fixture((1., 1., 2.), (1., 2., 3.)))
        self.assertEqual(result["rank_correlation"]["denominator"], 3)
        self.assertAlmostEqual(result["rank_correlation"]["value"], math.sqrt(3)/2)
        tied_both = compare(*fixture((1., 1., 4., 4.), (1., 2., 2., 3.)))
        self.assertAlmostEqual(tied_both["rank_correlation"]["value"], 1/math.sqrt(2))

    def test_spearman_strict_orders_and_near_costs_do_not_create_ties(self):
        ascending = compare(*fixture((1., 1. + 1e-10, 2.), (1., 2., 3.)))
        descending = compare(*fixture((3., 2., 1.), (1., 2., 3.)))
        self.assertAlmostEqual(ascending["rank_correlation"]["value"], 1.)
        self.assertAlmostEqual(descending["rank_correlation"]["value"], -1.)

    def test_spearman_constant_or_less_than_two_common_candidates_is_undefined(self):
        for data in (fixture((1., 1.), (1., 2.)), fixture((1., 2.), (1., 1.)), fixture((1.,), (1.,))):
            self.assertIsNone(compare(*data)["rank_correlation"]["value"])
        data = list(fixture())
        data[1][:] = False
        result = compare(*data)
        self.assertEqual(result["rank_correlation"]["denominator"], 0)
        self.assertIsNone(result["rank_correlation"]["value"])

    def test_spearman_only_uses_common_valid_and_is_invariant_to_shared_permutation(self):
        data = list(fixture((1., 2., 3., 0.), (1., 2., 3., 20.)))
        data[1][3] = False
        first = compare(*data)
        order = [2, 0, 3, 1]
        second = compare(*(column[order] for column in data))
        self.assertEqual(first["rank_correlation"]["denominator"], 3)
        self.assertAlmostEqual(first["rank_correlation"]["value"], 1.)
        self.assertEqual(first["rank_correlation"], second["rank_correlation"])

    def test_perfect_raw_mean_rank_correlation_cannot_resolve_overlapping_ranges(self):
        data = list(fixture((1., 2.), (10., 15.)))
        data[2] = np.array([[0., 10., 20.], [5., 15., 25.]], np.float64)
        result = compare(*data)
        self.assertAlmostEqual(result["rank_correlation"]["value"], 1.)
        self.assertEqual(result["pairwise"]["actual_inconclusive"], 1)
        self.assertEqual(result["pairwise"]["strict_comparison_pairs"], 0)
        self.assertEqual(result["actual_best_second_mean_gap"], 5.)
        self.assertEqual(result["actual_best_second_gap_ordering"], "INCONCLUSIVE")

    def test_best_second_mean_gap_and_selection_intervals_are_conditional(self):
        result = compare_candidates(*fixture((2., 1., 3.), (1., 2., 3.)), atol=.1, rtol=0.)
        self.assertEqual(result["actual_best_index"], 0)
        self.assertEqual(result["actual_second_index"], 1)
        self.assertEqual(result["actual_best_second_mean_gap"], 1.)
        self.assertEqual(result["actual_best_second_gap_ordering"], "SECOND_WORSE")
        self.assertEqual(result["selected_actual_cost_interval"], [1.9, 2.1])
        self.assertEqual(result["actual_best_cost_interval"], [.9, 1.1])
        single = compare(*fixture((1.,), (1.,)))
        self.assertIsNone(single["actual_second_index"])
        self.assertIsNone(single["actual_best_second_mean_gap"])
        invalid = list(fixture()); invalid[3][:] = False; invalid[2][:] = np.inf
        none = compare(*invalid)
        self.assertEqual(none["actual_status"], "no_valid_candidate")
        self.assertEqual(none["selection_status"], "selected_actual_invalid")
        self.assertIsNone(none["actual_best_cost_interval"])
        self.assertIsNone(none["selected_actual_cost_interval"])

    def test_selected_physical_invalid_candidate_is_retained_not_filtered_or_replaced(self):
        inputs = list(fixture((.1, 1., 2.), (1., 2., 3.)))
        inputs[2][0] = np.inf
        inputs[3][0] = False
        result = compare(*inputs)
        self.assertEqual(result["selected_index"], 0)
        self.assertEqual(result["selected_actual_status"], "STABLE_INVALID")
        self.assertEqual(result["actual_best_index"], 1)
        self.assertIsNone(result["selection_loss"])
        self.assertEqual(result["selection_loss_denominator"], 0)
        self.assertEqual(result["counts"]["false_valid"], 1)
        self.assertEqual(result["counts"]["candidate_count"], 3)
        self.assertEqual(len(result["actual_records"]), 3)
        self.assertEqual(result["actual_records"][0]["repeated_costs"], [None] * 3)

    def test_false_valid_false_invalid_and_common_valid_are_separate_populations(self):
        inputs = list(fixture((.1, 1., 2., 3.), (1., 2., 3., 4.)))
        inputs[1][1] = False
        inputs[2][0] = np.inf; inputs[3][0] = False
        result = compare(*inputs)
        counts = result["counts"]
        self.assertEqual((counts["candidate_count"], counts["false_valid"], counts["false_invalid"], counts["common_valid"]), (4, 1, 1, 2))
        self.assertEqual(result["error_metrics"]["denominator"], 2)
        self.assertEqual(result["pairwise"]["common_valid_pairs"], 1)
        self.assertEqual(result["pairwise"]["excluded_noncommon_pairs"], 5)

    def test_repeated_validity_flip_requires_investigation_not_false_valid_relabeling(self):
        inputs = list(fixture((.1, 1., 2.), (1., 2., 3.)))
        inputs[3][0] = [True, False, True]
        inputs[2][0, 1] = np.inf
        result = compare(*inputs)
        self.assertEqual(result["validity_flip_indices"], [0])
        self.assertEqual(result["infrastructure_status"], "VALIDITY_CHANGED_REQUIRES_INVESTIGATION")
        self.assertEqual(result["selected_actual_status"], "VALIDITY_CHANGED")
        self.assertIsNone(result["selection_loss"])
        self.assertEqual(result["counts"]["actual_validity_changed"], 1)
        self.assertEqual(result["counts"]["false_valid"], 0)
        self.assertEqual(result["counts"]["predicted_valid_actual_unstable"], 1)
        self.assertEqual(result["counts"]["actual_repeat_attempts"], 9)
        self.assertEqual(result["counts"]["actual_repeat_valid"], 8)
        self.assertEqual(result["counts"]["actual_repeat_invalid"], 1)

    def test_all_model_invalid_keeps_actual_best_and_false_invalid_denominator(self):
        inputs = list(fixture())
        inputs[0][:] = np.inf; inputs[1][:] = False
        result = compare(*inputs)
        self.assertTrue(result["model_all_invalid"])
        self.assertFalse(result["actual_all_invalid"])
        self.assertTrue(result["common_valid_empty"])
        self.assertIsNone(result["selected_index"])
        self.assertIsNone(result["selection_loss"])
        self.assertEqual(result["actual_best_index"], 0)
        self.assertEqual(result["counts"]["false_invalid"], 3)
        self.assertEqual(result["pairwise"]["total_pairs"], 3)
        self.assertEqual(result["pairwise"]["excluded_noncommon_pairs"], 3)
        self.assertEqual(result["error_metrics"], {"denominator": 0, "MAE": None, "RMSE": None})

    def test_all_actual_invalid_does_not_turn_into_model_all_invalid(self):
        inputs = list(fixture())
        inputs[2][:] = np.inf; inputs[3][:] = False
        result = compare(*inputs)
        self.assertFalse(result["model_all_invalid"])
        self.assertTrue(result["actual_all_invalid"])
        self.assertTrue(result["no_stable_actual_valid"])
        self.assertTrue(result["common_valid_empty"])
        self.assertEqual(result["selected_index"], 0)
        self.assertIsNone(result["actual_best_index"])
        self.assertIsNone(result["selection_loss"])
        self.assertEqual(result["counts"]["false_valid"], 3)

    def test_all_unstable_is_not_all_actual_invalid(self):
        inputs = list(fixture((1., 2.), (1., 2.)))
        inputs[3][:] = [True, False, True]
        inputs[2][:, 1] = np.inf
        result = compare(*inputs)
        self.assertFalse(result["actual_all_invalid"])
        self.assertTrue(result["no_stable_actual_valid"])
        self.assertEqual(result["validity_flip_indices"], [0, 1])
        self.assertEqual(result["counts"]["false_valid"], 0)
        self.assertIsNone(result["actual_best_index"])

    def test_one_candidate_works_without_inventing_pairwise_evidence(self):
        result = compare(*fixture((0.,), (0.,)))
        self.assertEqual(result["pairwise"]["total_pairs"], 0)
        self.assertEqual(result["pairwise"]["common_valid_pairs"], 0)
        self.assertIsNone(result["pairwise"]["concordance_rate"])
        self.assertEqual(result["selection_loss_ordering"], "INCONCLUSIVE")
        self.assertEqual(result["selection_loss"], 0.)

    def test_every_pair_belongs_to_exactly_one_category_and_original_indices_remain(self):
        inputs = list(fixture((1., 1., 3., 2., 9.), (1., 2., 3., 4., 5.)))
        inputs[1][4] = False
        inputs[2][3] = [2.5, 4., 4.5]
        result = compare(*inputs)
        pair = result["pairwise"]
        self.assertEqual(pair["common_valid_pairs"], 6)
        self.assertEqual(sum(pair[name] for name in ("actual_inconclusive", "predicted_tie", "concordant", "discordant")), 6)
        self.assertEqual(pair["excluded_noncommon_pairs"], 4)
        self.assertEqual({tuple(record["indices"]) for record in pair["records"]}, {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)})

    def test_arrays_and_masks_are_not_mutated_or_retained_by_reference(self):
        inputs = fixture()
        original = deepcopy(inputs)
        result = compare(*inputs)
        result["actual_records"][0]["repeated_valid"][0] = False
        result["predicted_records"][0]["cost"] = 99.
        for first, second in zip(inputs, original):
            np.testing.assert_array_equal(first, second)

    def test_json_is_finite_even_for_invalid_infinite_costs_and_empty_metrics(self):
        inputs = list(fixture())
        inputs[0][:] = np.inf; inputs[1][:] = False
        inputs[2][:] = np.inf; inputs[3][:] = False
        result = compare(*inputs)
        json.dumps(result, allow_nan=False)
        self.assertEqual(len(result["actual_records"]), 3)
        self.assertEqual(len(result["predicted_records"]), 3)

    def test_shape_and_dtype_contract_is_strict(self):
        original = fixture()
        mutations = [(0, np.array([1., 2., 3.], np.float32)),
                     (0, np.array([[1., 2., 3.]], np.float64)),
                     (0, [1., 2., 3.]), (0, np.array([], np.float64)),
                     (1, np.ones(3, np.int8)), (1, np.ones((3, 1), np.bool_)),
                     (2, np.ones((3, 2), np.float64)), (2, np.ones((3, 3), np.float32)),
                     (3, np.ones((3, 3), np.int8)), (3, np.ones((2, 3), np.bool_))]
        for index, value in mutations:
            args = list(deepcopy(original)); args[index] = value
            with self.subTest(index=index, shape=getattr(value, "shape", None)), self.assertRaises(DiagnosticError):
                compare(*args)

    def test_nonfinite_valid_costs_are_rejected(self):
        for index in (0, 2):
            for value in (np.inf, -np.inf, np.nan):
                args = list(fixture())
                args[index].flat[0] = value
                with self.subTest(index=index, value=value), self.assertRaises(DiagnosticError):
                    compare(*args)

    def test_invalid_costs_allow_positive_infinity_but_not_negative_or_nan(self):
        for index, mask_index in ((0, 1), (2, 3)):
            for value in (-.1, -np.inf, np.nan):
                args = list(fixture())
                args[index].flat[0] = value
                args[mask_index].flat[0] = False
                with self.subTest(index=index, value=value), self.assertRaises(DiagnosticError):
                    compare(*args)
        args = list(fixture()); args[0][0] = np.inf; args[1][0] = False
        self.assertEqual(compare(*args)["selected_index"], 1)

    def test_negative_valid_costs_and_invalid_tolerances_are_rejected(self):
        for index in (0, 2):
            args = list(fixture()); args[index].flat[0] = -.1
            with self.assertRaises(DiagnosticError):
                compare(*args)
        for name in ("atol", "rtol"):
            for value in (True, -.1, float("inf"), float("nan"), "0"):
                with self.subTest(name=name, value=value), self.assertRaises(DiagnosticError):
                    compare_candidates(*fixture(), **{name: value})

    def test_large_finite_means_and_rmse_avoid_sum_square_overflow(self):
        maximum = float(np.finfo(np.float64).max)
        result = compare(*fixture((0., 0.), (maximum, maximum / 2)))
        self.assertTrue(math.isfinite(result["error_metrics"]["MAE"]))
        self.assertTrue(math.isfinite(result["error_metrics"]["RMSE"]))
        self.assertEqual(result["actual_records"][0]["mean_cost"], maximum)
        json.dumps(result, allow_nan=False)

    def test_unrepresentable_interval_rejected_instead_of_returning_infinite_json(self):
        maximum = float(np.finfo(np.float64).max)
        with self.assertRaises(DiagnosticError):
            compare_candidates(*fixture((1.,), (maximum,)))


if __name__ == "__main__":
    unittest.main()
