"""Independent selection, split and frozen-development runner witnesses.

Small interleaved tables and synthetic failed plans are unit fixtures. They
do not establish robot accuracy, successful control or contact physics.
"""
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import tomllib
import unittest
from unittest.mock import patch

import numpy as np
import torch

from push_align.config import ROOT, load_pilot_config
from push_align.data import digest


DECLARED_SETTINGS = {
    'ridge_id': 'p3-main-ridge-v1', 'device': 'cpu', 'torch_threads': 1,
    'deterministic_algorithms': True,
    'alpha_candidates': [0., .0001, .001, .01, .1, 1., 10., 100.],
    'objective': 'sum_squared_normalized_residuals_plus_alpha_coefficient_frobenius_squared',
    'intercept': 'unpenalized', 'zero_alpha': 'centered_svd_minimum_norm_ols',
    'fit_split': 'train', 'fit_count': 3200, 'validation_count': 1600,
    'evaluation_horizons': [1, 5, 10], 'selection_horizon': 5,
    'selection_rule': 'eligible_minimum_equal_three_regime_endpoint_score_then_smallest_alpha',
    'selection_before_closed_loop': True,
    'closed_loop_config': 'configs/p2_closed_loop.v1.toml',
    'closed_loop_scope': 'reuse_the_same_ten_previously_used_development_cases_not_final_test',
    'normalizer_hash': '3ea38c6a9b5d17a751de10b77e557838d3f2fac6b6bafa6f3728d7de69753c22',
    'dataset_manifest_sha256': '9faa2f64a2eaf302f3a68fb06fc9d6340118b9ff703e213e9ba6abeb56a5df4b',
    'scope': 'main-data learned linear feature-map dynamics; implementation, free prediction and fixed development MPC; no fair comparison against pilot-data MLP',
}


def candidate(alpha, score, status='eligible'):
    return {'alpha': alpha, 'selection_score': score, 'selection_status': status}


def interleaved_prepared():
    # Physical values identify their source row. Cached normalized values are
    # deliberately different: passing those as physical input would be wrong.
    return SimpleNamespace(arrays={
        'split_code': np.array([1, 0, 1, 0, 0, 2], np.int8),
        'features': np.arange(66, dtype=np.float64).reshape(6, 11),
        'deltas': (100+np.arange(60, dtype=np.float64)).reshape(6, 10),
        'normalized_features': np.full((6, 11), -777., np.float32),
        'normalized_deltas': np.full((6, 10), 888., np.float32),
    })


class RidgeRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = str(ROOT/'scripts')
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        spec = importlib.util.spec_from_file_location('p3_ridge_runner_test_module',
                                                     ROOT/'scripts/p3_train_ridge.py')
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def test_minimum_eligible_score_wins_before_alpha_size(self):
        low_alpha = candidate(0., 9.)
        winner = candidate(100., 2.)
        self.assertIs(self.runner.choose_alpha([low_alpha, winner]), winner)

    def test_exact_score_tie_selects_smaller_alpha_regardless_of_order(self):
        left, right = candidate(10., 2.), candidate(.001, 2.)
        for rows in ([left, right], [right, left]):
            self.assertIs(self.runner.choose_alpha(rows), right)

    def test_near_tie_is_not_rounded_into_exact_tie(self):
        small_alpha = candidate(0., 1.000000000000001)
        lower_score = candidate(100., 1.)
        self.assertIs(self.runner.choose_alpha([small_alpha, lower_score]), lower_score)

    def test_invalid_scores_and_ineligible_better_scores_cannot_win(self):
        winner = candidate(1., 7.)
        for bad in (None, float('nan'), float('inf'), -float('inf'), -1.,
                    True, False, '0'):
            with self.subTest(score=bad):
                self.assertIs(self.runner.choose_alpha([candidate(0., bad), winner]), winner)
        for status in ('invalid_selection_windows', 'missing_selection_regime', 'ineligible'):
            with self.subTest(status=status):
                self.assertIs(self.runner.choose_alpha([candidate(0., 0., status), winner]), winner)

    def test_all_ineligible_fails_explicitly_without_fallback_model(self):
        for rows in ([], [candidate(0., None)],
                     [candidate(0., 0., 'invalid_selection_windows')]):
            with self.subTest(rows=rows):
                with self.assertRaisesRegex(ValueError, 'No eligible K5 validation candidate'):
                    self.runner.choose_alpha(rows)

    def test_selection_does_not_remove_or_rewrite_invalid_candidate_records(self):
        rows = [candidate(.1, None, 'invalid_selection_windows'), candidate(1., 3.)]
        before = deepcopy(rows)
        self.runner.choose_alpha(rows)
        self.assertEqual(rows, before)

    def test_fit_receives_only_interleaved_original_train_rows(self):
        prepared, normalizer = interleaved_prepared(), object()
        model, diagnostics = object(), {'fit_count': 3}
        with patch.object(self.runner, 'fit_ridge', return_value=(model, diagnostics)) as fit:
            actual_model, actual_diag, rows = self.runner.fit_train_ridge(prepared, normalizer, .1)
        self.assertIs(actual_model, model)
        self.assertIs(actual_diag, diagnostics)
        np.testing.assert_array_equal(rows, [1, 3, 4])
        x, y, passed_normalizer, alpha = fit.call_args.args
        np.testing.assert_array_equal(x[:, 0], [11., 33., 44.])
        np.testing.assert_array_equal(y[:, 0], [110., 130., 140.])
        np.testing.assert_array_equal(x, prepared.arrays['features'][rows])
        np.testing.assert_array_equal(y, prepared.arrays['deltas'][rows])
        self.assertEqual((x.dtype, y.dtype), (np.dtype('float64'), np.dtype('float64')))
        self.assertIs(passed_normalizer, normalizer)
        self.assertEqual(alpha, .1)

    def test_extreme_validation_and_smoke_values_do_not_enter_fit(self):
        prepared = interleaved_prepared()
        captured = []
        def witness(x, y, normalizer, alpha):
            captured.append((x.copy(), y.copy()))
            return object(), {}
        with patch.object(self.runner, 'fit_ridge', side_effect=witness):
            _, _, before_rows = self.runner.fit_train_ridge(prepared, object(), 1.)
            prepared.arrays['features'][[0, 2, 5]] = 1e100
            prepared.arrays['deltas'][[0, 2, 5]] = -1e100
            _, _, after_rows = self.runner.fit_train_ridge(prepared, object(), 1.)
        np.testing.assert_array_equal(before_rows, after_rows)
        for before, after in zip(captured[0], captured[1]):
            np.testing.assert_array_equal(before, after)

    def test_fit_input_copies_protect_prepared_source_from_core_mutation(self):
        prepared = interleaved_prepared()
        before = {k: v.copy() for k, v in prepared.arrays.items()}
        def mutating_core(x, y, normalizer, alpha):
            self.assertFalse(np.shares_memory(x, prepared.arrays['features']))
            self.assertFalse(np.shares_memory(y, prepared.arrays['deltas']))
            x[:] = -1.; y[:] = -2.
            return object(), {}
        with patch.object(self.runner, 'fit_ridge', side_effect=mutating_core):
            self.runner.fit_train_ridge(prepared, object(), 0.)
        for key in before:
            np.testing.assert_array_equal(prepared.arrays[key], before[key])

    def test_no_train_rows_fail_before_core_fit(self):
        prepared = interleaved_prepared()
        prepared.arrays['split_code'][:] = 1
        with patch.object(self.runner, 'fit_ridge') as fit:
            with self.assertRaisesRegex(ValueError, 'No train rows'):
                self.runner.fit_train_ridge(prepared, object(), 1.)
        fit.assert_not_called()

    def test_fixed_settings_are_independent_of_current_config_file(self):
        self.runner.validate_settings(deepcopy(DECLARED_SETTINGS))
        raw = (ROOT/'configs/p3_ridge.v1.toml').read_text()
        with tempfile.TemporaryDirectory() as tmp:
            fake_root = Path(tmp)
            (fake_root/'configs').mkdir()
            changed_raw = raw.replace('fit_count = 3200', 'fit_count = 3199')
            self.assertNotEqual(raw, changed_raw)
            (fake_root/'configs/p3_ridge.v1.toml').write_text(changed_raw)
            changed = tomllib.loads(changed_raw)
            with patch.object(self.runner, 'ROOT', fake_root):
                with self.assertRaises(ValueError):
                    self.runner.validate_settings(changed)

    def test_settings_reject_protocol_and_numeric_type_changes(self):
        changes = {
            'alpha_candidates': [0., 1.], 'evaluation_horizons': [1, 5],
            'selection_horizon': 1, 'selection_before_closed_loop': False,
            'fit_split': 'validation', 'fit_count': 3200., 'torch_threads': True,
            'normalizer_hash': '0'*64, 'closed_loop_config': 'configs/new_cases.toml',
        }
        for key, value in changes.items():
            with self.subTest(key=key):
                settings = deepcopy(DECLARED_SETTINGS)
                settings[key] = value
                with self.assertRaises(ValueError):
                    self.runner.validate_settings(settings)

    def test_frozen_case_ids_seeds_and_offset_signs_are_not_outcome_selected(self):
        settings = tomllib.loads((ROOT/'configs/p2_closed_loop.v1.toml').read_text())
        config = {'object': {'size_full_xyz_m': [.08, .04, .01]},
                  'pusher': {'radius_m': .008}, 'reset': {'pusher_surface_gap_m': .003}}
        cases = self.runner.case_plan(settings, config)
        self.assertEqual([c['episode_id'] for c in cases], [f'smoke-{i:02d}' for i in range(10)])
        self.assertEqual([c['candidate_seed'] for c in cases], list(range(3711, 3721)))
        self.assertEqual([c['episode_index'] for c in cases], list(range(10)))
        for index, xy in ((0, [-.051, 0.]), (7, [-.051, .015]), (8, [-.051, -.015])):
            state = cases[index]['initial_spec']
            np.testing.assert_allclose([state['pusher_x'], state['pusher_y']], xy,
                                       atol=1e-15, rtol=0.)
        self.assertEqual(cases[9]['initial_spec']['object_x'], -.01)
        self.assertEqual(cases[9]['initial_spec']['object_y'], .005)

    def development_witness(self, *, tamper_selection=False):
        runner = self.runner
        config = load_pilot_config()
        contract = {'unit_fixture_contract': True}
        prepared = interleaved_prepared()
        prepared.metadata = {'contract': contract}
        normalizer = SimpleNamespace(metadata={'normalizer_hash': 'normalizer-fixture'})
        bundle = SimpleNamespace(model=torch.nn.Linear(11, 10),
                                 metadata={'ridge_hash': 'ridge-fixture'})
        settings = tomllib.loads((ROOT/'configs/p2_closed_loop.v1.toml').read_text())
        predictor = SimpleNamespace(calls=0, rows=0)
        class Env:
            action_dt = .1
            def __init__(self, config):
                pass
            def reset(self, initial):
                return SimpleNamespace(state=np.zeros(10, np.float64), sim_time=0.,
                                       schema_version='push-align-state-v1', observation_valid=True)
        class Recorder:
            def __init__(self, *args, **kwargs):
                self.results, self.snapshots = [], ['synthetic-initial-snapshot']
        class Writer:
            def __init__(self, path, *args, **kwargs):
                self.entries, self.path = [], path
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            selection = parent/'selection.json'
            selection.write_text(json.dumps({'selected': {'alpha': 1.}}))
            selection_hash = digest(selection)
            def failed_plan(*args, **kwargs):
                predictor.calls += 1
                predictor.rows += 256
                if tamper_selection:
                    selection.write_text(json.dumps({'selected': {'alpha': 100.}}))
                return SimpleNamespace(reason='planner_invalid', selected_index=None,
                                       diagnostics={'wall_time_s': .001})
            with ExitStack() as stack:
                for name, value in (
                    ('PlanarPushEnv', Env), ('make_contract', lambda env: contract),
                    ('LearnedPredictor', lambda *args: predictor), ('EpisodeRecorder', Recorder),
                    ('DatasetWriter', Writer), ('holding_fixture', lambda *args: {'fixture_pass': True}),
                    ('compact_plan', lambda *args: {}), ('plan', failed_plan),
                    ('save_trace', lambda *args: None)):
                    stack.enter_context(patch.object(runner, name, value))
                choose = stack.enter_context(patch.object(runner, 'choose_alpha',
                                                         side_effect=AssertionError('No reselection allowed')))
                result = runner.development(parent/'development', bundle, prepared, normalizer,
                                            settings, config, selection_hash)
                choose.assert_not_called()
            return result

    def test_development_preserves_all_failed_cases_without_alpha_reselection(self):
        result = self.development_witness()
        self.assertEqual(len(result['ledger']), 10)
        self.assertEqual([row['episode_id'] for row in result['ledger']],
                         [f'smoke-{i:02d}' for i in range(10)])
        self.assertEqual(result['terminal_counts'], {'planner_invalid': 10})
        self.assertTrue(result['checks']['selection_frozen_before_and_after_execution'])
        self.assertEqual(result['actual_model_calls'], 10)
        self.assertEqual(sum(result['future_engine_calls_during_planning'].values()), 0)
        self.assertEqual(result['alignment_outcome'], 'NOT_DEMONSTRATED')
        self.assertEqual(result['user_understanding'], 'NOT_ASSESSED')

    def test_development_detects_selected_alpha_file_change(self):
        result = self.development_witness(tamper_selection=True)
        self.assertFalse(result['checks']['selection_frozen_before_and_after_execution'])
        self.assertFalse(result['pass'])


if __name__ == '__main__':
    unittest.main()
