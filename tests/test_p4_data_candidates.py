"""Failure/denominator and fixed-candidate checks for data-regression metrics."""
from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from evaluate_p4_data_candidates import (confusion, error_summary, selected_actual,
    task_comparison, repeated_lex_order, unique_groups, main)


def row(key, ready=False, valid=True):
    return dict(physical_valid=valid, actual_task_rank=list(key), actual_J_H10=float(key[-1]),
        actual_endpoint_proxy_ready=ready, raw_task={'status': {'phase': 'ready' if ready else 'running'}})


class CandidateDataEvaluationTests(unittest.TestCase):
    def test_duplicate_actions_count_once_preserves_original_order(self):
        actions = np.zeros((8, 10, 2), np.float64)
        actions[3, 0, 1] = .001
        actions[7] = actions[3]
        self.assertEqual(unique_groups(actions, [7, 3, 2, 0]), [[0, 2], [3, 7]])

    def test_invalid_prediction_is_not_a_false_negative(self):
        result = confusion([False, True, True], [False, True, True], [True, False, True], [True, True, False])
        self.assertEqual(result['FN'], 0)
        self.assertEqual(result['prediction_invalid'], 1)
        self.assertEqual(result['actual_invalid'], 1)
        self.assertEqual(result['FP'], 1)
        self.assertEqual(result['classified_denominator'], 1)

    def test_all_confusion_cells(self):
        result = confusion([1, 1, 0, 0], [1]*4, [1, 0, 0, 1], [1]*4)
        self.assertEqual([result[k] for k in ('TP', 'FP', 'TN', 'FN')], [1]*4)

    def test_full_candidate_unmeasured_stays_unknown(self):
        actual = {11: [row([0, 0, .1, 5], True)]*3}
        self.assertEqual(selected_actual(77, actual)['status'], 'UNKNOWN')
        self.assertEqual(selected_actual(None, actual)['status'], 'NO_VALID_PREDICTION')
        self.assertTrue(selected_actual(11, actual)['actual_raw_substep_ready'])

    def test_invalid_error_keeps_denominator_and_no_all_case_mean(self):
        actual = np.zeros((2, 10), np.float64)
        predicted = actual.copy(); predicted[0, 0] = .002
        predicted[1] = np.nan
        result = error_summary(actual, predicted, [True, False])
        self.assertEqual((result['total'], result['valid'], result['invalid']), (2, 1, 1))
        self.assertIsNone(result['all_case_mean']['object_xy_mm'])
        self.assertEqual(result['conditional_valid_mean']['object_xy_mm'], 2)
        self.assertIsNone(result['rows'][1]['object_xy_mm'])

    def test_lexicographic_ready_precedes_larger_J(self):
        a, b = [[0, 0, .1, 100]]*3, [[1, 0, 0, 1]]*3
        self.assertEqual(repeated_lex_order(a, b), -1)

    def test_small_distinct_key_keeps_exact_order_without_tolerance_claim(self):
        a, b = [[1, .1, 1, 1]]*3, [[1, .1000000001, 100, 100]]*3
        self.assertEqual(repeated_lex_order(a, b), -1)

    def test_exact_equal_prefix_permits_later_key(self):
        self.assertEqual(repeated_lex_order([[1, .1, 1, 4]]*3, [[1, .1, 3, 1]]*3), -1)

    def test_task_regret_is_lexicographic_vector_not_sum(self):
        actual = {0: [row([1, .1, .1, 1])]*3, 2: [row([0, 0, .2, 5], True)]*3}
        keys = np.asarray([[0, 0, 0, 1], [9]*4, [1, 1, 1, 5]], np.float64)
        result = task_comparison([0, 2], keys, np.asarray([True, False, True]), actual)
        self.assertEqual(result['predicted_selected_index'], 0)
        self.assertEqual(result['actual_best_index'], 2)
        self.assertEqual(result['selection_loss']['lexicographic_rank_gap'], 1)
        self.assertEqual(result['selection_loss']['first_differing_component'], 0)
        self.assertEqual(result['pairwise']['discordant'], 1)

    def test_all_invalid_task_predictions_retain_full_population(self):
        actual = {0: [row([1, .1, .1, 1])]*3}
        result = task_comparison([0], np.full((1, 4), np.inf), np.asarray([False]), actual)
        self.assertIsNone(result['predicted_selected_index'])
        self.assertIsNone(result['selection_loss'])
        self.assertEqual(result['counts'], dict(total=1, predicted_valid=0, actual_stable_valid=1, common_valid=0))

    def test_task_best_and_exact_tied_best_are_distinct(self):
        keys = np.asarray([[0, 0, .1, 5], [0, 0, .1, 5]], np.float64)
        actual = {0: [row(keys[0])]*3, 1: [row(keys[1])]*3}
        tied = task_comparison([0, 1], keys, np.asarray([True, True]), actual)
        single = task_comparison([0], keys, np.asarray([True, True]), actual)
        self.assertEqual(tied['selection_loss']['ordering'], 'TIED_EXACT_BEST')
        self.assertEqual(single['selection_loss']['ordering'], 'EXACT_BEST')
        self.assertEqual(tied['numerical_robustness'], 'NOT_EVALUATED')

    def test_changed_actual_repeats_are_not_given_an_exact_order(self):
        self.assertIsNone(repeated_lex_order([[1, .1, .1, 1], [1, .2, .1, 1], [1, .1, .1, 1]],
                                           [[1, .3, .1, 1]]*3))

    def test_cli_failure_preserves_new_partial_artifacts_and_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'result'
            def fail(*args):
                out.mkdir(); (out / 'partial.json').write_text('{}')
                raise ValueError('synthetic failure')
            argv = ['candidate', '--training-run', tmp, '--registration', tmp+'/registry.json',
                    '--registration-sha256', 'a'*64, '--output', str(out)]
            with patch.object(sys, 'argv', argv), patch('evaluate_p4_data_candidates.evaluate', side_effect=fail), redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 1)
            self.assertEqual(json.loads((out / 'status.json').read_text())['status'], 'FAIL')
            self.assertIn('synthetic failure', json.loads((out / 'failure.json').read_text())['traceback'])
            index = json.loads((out / 'artifact_hashes.json').read_text())['sha256']
            self.assertEqual(set(index), {'partial.json', 'status.json', 'failure.json'})

    def test_cli_failure_does_not_write_into_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'result'; out.mkdir()
            (out / 'sentinel.txt').write_text('original')
            argv = ['candidate', '--training-run', tmp, '--registration', tmp+'/registry.json',
                    '--registration-sha256', 'a'*64, '--output', str(out)]
            with patch.object(sys, 'argv', argv), patch('evaluate_p4_data_candidates.evaluate', side_effect=ValueError('exists')), redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 1)
            self.assertEqual([p.name for p in out.iterdir()], ['sentinel.txt'])
            self.assertEqual((out / 'sentinel.txt').read_text(), 'original')


if __name__ == '__main__':
    unittest.main()
