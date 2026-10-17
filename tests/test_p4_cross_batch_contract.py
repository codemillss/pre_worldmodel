"""Synthetic denominator, control-gate and registration checks; no model calls."""
from copy import deepcopy
import json
from pathlib import Path
import platform
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import p4_cross_batch_probe as runner


def fixture_rows():
    rows = []
    sources = ('historical_update1', 'D11_zero5', 'D11_transition', 'D11_noncontact')
    for snapshot in range(7):
        for source in sources:
            relations = ['NUMERIC_TIE', 'INCREASE', 'DECREASE', 'NUMERIC_TIE']
            rows.append({'snapshot_id': str(snapshot), 'source_probe_name': source,
                'evaluation': {'status': 'PASS', 'evaluations': [{'total_change_relation': value} for value in relations]},
                'controls': {'alpha0_matches_previous_stop': True, 'alpha0_matches_snapshot_first_source': True,
                             'self_stop_previous': {'pass': True} if source == 'D11_zero5' else None}})
    return rows


class CrossBatchContractTests(unittest.TestCase):
    def test_all84_positive_endpoints_are_partitioned_into7self21other_cases(self):
        rows = fixture_rows()
        result = runner.primary_counts(rows, True)
        self.assertEqual(result['all']['case_count'], 28)
        self.assertEqual(result['all']['planned_positive_endpoints'], 84)
        self.assertEqual(result['self_stop']['planned_positive_endpoints'], 21)
        self.assertEqual(result['other_sources']['planned_positive_endpoints'], 63)
        self.assertEqual(result['all']['by_alpha']['0.25']['INCREASE'], 28)
        self.assertEqual(result['other_sources']['by_alpha']['0.5']['DECREASE'], 21)
        self.assertTrue(runner.global_controls(rows)['pass'])

    def test_one_failed_case_holds_all_three_endpoints_without_denominator_removal(self):
        rows = fixture_rows()
        rows[0]['evaluation']['status'] = 'NUMERICAL_FAILURE'
        result = runner.primary_counts(rows, True)
        self.assertEqual(result['all']['case_count'], 28)
        self.assertEqual(result['all']['planned_positive_endpoints'], 84)
        for alpha in ('0.25', '0.5', '1.0'):
            self.assertEqual(result['all']['by_alpha'][alpha]['NOT_INTERPRETED'], 1)
            self.assertEqual(sum(result['all']['by_alpha'][alpha].values()), 28)
            self.assertEqual(result['other_sources']['by_alpha'][alpha]['NOT_INTERPRETED'], 1)

    def test_any_self_control_failure_gates_all_primary_counts(self):
        rows = fixture_rows()
        rows[1]['controls']['self_stop_previous']['pass'] = False
        controls = runner.global_controls(rows)
        self.assertFalse(controls['pass'])
        self.assertIsNone(runner.primary_counts(rows, controls['pass']))
        self.assertEqual(len(rows), 28)

    def test_alpha0_previous_or_cross_source_mismatch_gates_global_interpretation(self):
        for key in ('alpha0_matches_previous_stop', 'alpha0_matches_snapshot_first_source'):
            with self.subTest(key=key):
                rows = fixture_rows()
                rows[0]['controls'][key] = False
                self.assertFalse(runner.global_controls(rows)['pass'])

    def test_missing_or_duplicate_self_control_cannot_pass_vacuously(self):
        rows = fixture_rows()
        self.assertFalse(runner.global_controls([row for row in rows if row['source_probe_name'] != 'D11_zero5'])['pass'])
        rows[-1] = deepcopy(rows[0])
        self.assertFalse(runner.global_controls(rows)['pass'])

    def registration(self, root):
        cases = [{'synthetic': index} for index in range(28)]
        design = {'cases': cases, 'target_probe_name': 'D11_zero5', 'alphas': [0., .25, .5, 1.]}
        (root / 'design.json').write_text(json.dumps(design))
        (root / 'source.txt').write_text('fixed')
        p = {'run_id': 'p4-cross-batch-20261015-01', 'design_path': 'design.json',
             'design_sha256': runner.sha(root / 'design.json'), **design,
             'bindings_sha256': {'source.txt': runner.sha(root / 'source.txt'), 'design.json': runner.sha(root / 'design.json')},
             'runtime': {'torch_version': str(torch.__version__), 'numpy_version': np.__version__,
                         'python_version': platform.python_version(), 'source_sha256': {}}}
        path = root / 'registration.json'
        path.write_text(json.dumps(p))
        return path, p

    def test_registration_fixes_source_target_grid_and_input_hash_before_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, p = self.registration(root)
            with patch.object(runner, 'ROOT', root):
                self.assertEqual(runner.validate(path, runner.sha(path)), p)
                (root / 'source.txt').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'Frozen source'):
                    runner.validate(path, runner.sha(path))

    def test_target_relabel_or_new_alpha_is_rejected_before_model_calls(self):
        for field, value in [('target_probe_name', 'D11_noncontact'), ('alphas', [0., .1, .5, 1.])]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path, p = self.registration(root)
                p[field] = value
                path.write_text(json.dumps(p))
                with patch.object(runner, 'ROOT', root), self.assertRaisesRegex(ValueError, 'scope changed'):
                    runner.validate(path, runner.sha(path))


if __name__ == '__main__':
    unittest.main()
