"""Independent trained-epoch selection and original-train-row witnesses.

Synthetic tables test data routing and deterministic updates, not predictive
accuracy or robot behavior. Checkpoint/RNG continuation has separate tests.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
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

from push_align import PlanarPushEnv
from push_align.config import ROOT
from push_align.data import make_contract
from push_align.normalization import fit_normalizer, learning_schema
from push_align.representation import encode, state_delta


def epoch_candidate(epoch, score, status='eligible'):
    return {'epoch': epoch, 'selection_score': score, 'selection_status': status}


def interleaved_table():
    return SimpleNamespace(arrays={
        'split_code': np.array([1, 0, 1, 0, 0, 2], np.int8),
        'features': np.arange(66, dtype=np.float64).reshape(6, 11),
        'deltas': (100+np.arange(60, dtype=np.float64)).reshape(6, 10),
        'normalized_features': np.full((6, 11), -777., np.float32),
        'normalized_deltas': np.full((6, 10), 888., np.float32),
    })


class CommonRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = str(ROOT/'scripts')
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        spec = importlib.util.spec_from_file_location('p3_common_runner_test_module',
                                                     ROOT/'scripts/p3_train_common_mlp.py')
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)
        cls.settings = tomllib.loads((ROOT/'configs/p3_training.v1.toml').read_text())
        torch.set_num_threads(1)

    def test_epoch_zero_untrained_diagnostic_cannot_win(self):
        untrained = epoch_candidate(0, 0.)
        trained = epoch_candidate(9, 2.)
        self.assertIs(self.runner.choose_epoch([untrained, trained]), trained)

    def test_best_trained_score_precedes_epoch_order(self):
        early, winner = epoch_candidate(1, 3.), epoch_candidate(100, .1)
        self.assertIs(self.runner.choose_epoch([early, winner]), winner)

    def test_exact_tie_chooses_earliest_trained_epoch_without_rounding(self):
        late, early = epoch_candidate(100, 1.), epoch_candidate(3, 1.)
        for rows in ([late, early], [early, late]):
            self.assertIs(self.runner.choose_epoch(rows), early)
        lower_score = epoch_candidate(100, 1.)
        near_tie = epoch_candidate(1, 1.000000000000001)
        self.assertIs(self.runner.choose_epoch([near_tie, lower_score]), lower_score)

    def test_invalid_metrics_or_epoch_identity_cannot_win(self):
        winner = epoch_candidate(7, 5.)
        for score in (None, float('nan'), float('inf'), -1., True, False, '0'):
            with self.subTest(score=score):
                self.assertIs(self.runner.choose_epoch([epoch_candidate(1, score), winner]), winner)
        for epoch in (-1, 0, 101, True, False, 1.5, '1'):
            with self.subTest(epoch=epoch):
                self.assertIs(self.runner.choose_epoch([epoch_candidate(epoch, 0.), winner]), winner)
        for status in ('invalid_selection_windows', 'missing_selection_regime', 'ineligible'):
            with self.subTest(status=status):
                self.assertIs(self.runner.choose_epoch([epoch_candidate(1, 0., status), winner]), winner)

    def test_all_invalid_or_untrained_candidates_have_no_fallback(self):
        for rows in ([], [epoch_candidate(0, 0.)],
                     [epoch_candidate(1, None)],
                     [epoch_candidate(100, 0., 'invalid_selection_windows')]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.runner.choose_epoch(rows)

    def test_selection_preserves_all_recorded_candidates(self):
        candidates = [epoch_candidate(1, None, 'invalid_selection_windows'),
                      epoch_candidate(2, 3.), epoch_candidate(3, 2.)]
        before = deepcopy(candidates)
        self.runner.choose_epoch(candidates)
        self.assertEqual(candidates, before)

    def test_train_tensors_use_source_split_mask_and_physical_columns(self):
        prepared = interleaved_table()
        rows, features, targets = self.runner.train_tensors(prepared)
        np.testing.assert_array_equal(rows, [1, 3, 4])
        self.assertEqual(rows.dtype, np.dtype('int64'))
        for value, shape in ((features, (3, 11)), (targets, (3, 10))):
            self.assertIsInstance(value, torch.Tensor)
            self.assertEqual(tuple(value.shape), shape)
            self.assertEqual(value.dtype, torch.float64)
            self.assertEqual(value.device.type, 'cpu')
        np.testing.assert_array_equal(features.numpy()[:, 0], [11., 33., 44.])
        np.testing.assert_array_equal(targets.numpy()[:, 0], [110., 130., 140.])
        np.testing.assert_array_equal(features.numpy(), prepared.arrays['features'][rows])
        np.testing.assert_array_equal(targets.numpy(), prepared.arrays['deltas'][rows])

    def test_train_tensor_mutation_does_not_edit_original_rows_or_cached_data(self):
        prepared = interleaved_table()
        before = {key: value.copy() for key, value in prepared.arrays.items()}
        rows, features, targets = self.runner.train_tensors(prepared)
        features.fill_(-1.); targets.fill_(-2.)
        rows[:] = -1
        for key, expected in before.items():
            np.testing.assert_array_equal(prepared.arrays[key], expected)

    def test_validation_and_smoke_extremes_do_not_change_extracted_train(self):
        prepared = interleaved_table()
        before = self.runner.train_tensors(prepared)
        prepared.arrays['features'][[0, 2, 5]] = 1e100
        prepared.arrays['deltas'][[0, 2, 5]] = -1e100
        after = self.runner.train_tensors(prepared)
        np.testing.assert_array_equal(before[0], after[0])
        self.assertTrue(torch.equal(before[1], after[1]))
        self.assertTrue(torch.equal(before[2], after[2]))

    def test_empty_train_mask_is_rejected(self):
        prepared = interleaved_table()
        prepared.arrays['split_code'][:] = 1
        with self.assertRaises(ValueError):
            self.runner.train_tensors(prepared)

    def evaluation_fixture(self):
        return SimpleNamespace(summary={'selection_score': 1., 'selection_status': 'eligible'},
                               traces={'states': np.array([[1., np.nan]], np.float64),
                                       'valid': np.array([False]),
                                       'reason': np.array(['invalid_window'])})

    def test_reloaded_evaluation_matches_nan_float_and_nonfloat_traces(self):
        left = self.evaluation_fixture()
        self.assertTrue(self.runner.evaluation_equal(left, deepcopy(left)))

    def test_reload_comparison_requires_full_summary_and_trace_key_population(self):
        left = self.evaluation_fixture()
        for kind in ('summary', 'missing', 'extra', 'value', 'reason'):
            with self.subTest(kind=kind):
                right = deepcopy(left)
                if kind == 'summary':
                    right.summary['selection_score'] = 2.
                elif kind == 'missing':
                    del right.traces['states']
                elif kind == 'extra':
                    right.traces['extra'] = np.array([1.])
                elif kind == 'value':
                    right.traces['states'][0, 1] = 0.
                else:
                    right.traces['reason'][0] = 'changed_reason'
                self.assertFalse(self.runner.evaluation_equal(left, right))

    def test_registered_seed_mapping_and_full_batch_budget(self):
        self.assertEqual(self.settings['seeds'], [11, 23, 47])
        self.assertEqual(self.settings['shuffle_seeds'], [1011, 1023, 1047])
        self.assertEqual(self.settings['main_epochs'], 100)
        self.assertEqual(self.settings['batch_size'], 256)
        self.assertFalse(self.settings['drop_last'])
        self.assertEqual(self.settings['fit_count'], 3200)
        self.assertEqual(self.settings['validation_count'], 1600)
        self.assertEqual(self.settings['updates_per_epoch'], 13)
        self.assertEqual(self.settings['updates_per_seed'], 1300)
        self.assertEqual(self.settings['total_updates'], 3900)

    def test_settings_fingerprint_rejects_protocol_types_and_simultaneous_file_change(self):
        self.runner.validate_settings(deepcopy(self.settings))
        changes = {
            'seeds': [11, 23], 'shuffle_seeds': [1023, 1011, 1047],
            'main_epochs': 99, 'batch_size': 320, 'drop_last': True,
            'updates_per_epoch': 12, 'updates_per_seed': 1299,
            'fit_count': 3200., 'torch_threads': True, 'leakage_probe_updates': 0,
        }
        for key, value in changes.items():
            with self.subTest(key=key):
                changed = deepcopy(self.settings)
                changed[key] = value
                with self.assertRaises(ValueError):
                    self.runner.validate_settings(changed)
        raw = (ROOT/'configs/p3_training.v1.toml').read_text()
        changed_raw = raw.replace('main_epochs = 100', 'main_epochs = 99')
        self.assertNotEqual(raw, changed_raw)
        with tempfile.TemporaryDirectory() as tmp:
            fake_root = Path(tmp)
            (fake_root/'configs').mkdir()
            (fake_root/'configs/p3_training.v1.toml').write_text(changed_raw)
            with patch.object(self.runner, 'ROOT', fake_root), self.assertRaises(ValueError):
                self.runner.validate_settings(tomllib.loads(changed_raw))

    def test_epoch_visits_3200_interleaved_train_rows_once_with_final_128_batch(self):
        split = np.tile(np.array([0, 1, 0], np.int8), 1600)
        physical_x = np.zeros((4800, 11), np.float64)
        physical_y = np.zeros((4800, 10), np.float64)
        physical_x[:, 0] = np.arange(4800)
        physical_y[:, 0] = 10000+np.arange(4800)
        prepared = SimpleNamespace(arrays={'split_code': split, 'features': physical_x,
                                          'deltas': physical_y})
        rows, x, y = self.runner.train_tensors(prepared)
        shuffled = torch.Generator(device='cpu').manual_seed(1011)
        expected = torch.randperm(3200, generator=torch.Generator().manual_seed(1011))
        observed = []
        def update_witness(model, optimizer, normalizer, batch_x, batch_y, clip):
            self.assertEqual(batch_x.dtype, torch.float64)
            self.assertEqual(batch_y.dtype, torch.float64)
            observed.extend(batch_x[:, 0].numpy().astype(np.int64).tolist())
            np.testing.assert_array_equal(batch_y[:, 0].numpy(), batch_x[:, 0].numpy()+10000)
            return {'batch_size': len(batch_x), 'loss_before_update': 1.,
                    'gradient_norm_before_clip': .1}
        with patch.object(self.runner, 'training_update', side_effect=update_witness) as update:
            result = self.runner.train_epoch(object(), object(), object(), x, y, rows,
                                             shuffled, 256, 1.)
        self.assertEqual(update.call_count, 13)
        self.assertEqual([b['batch_size'] for b in result['batches']], [256]*12+[128])
        self.assertEqual(result['order'], rows[expected.numpy()].tolist())
        self.assertEqual(observed, result['order'])
        self.assertEqual(sorted(observed), rows.tolist())
        self.assertTrue(all(split[row] == 0 for row in observed))

    def normalized_training_fixture(self):
        contract = make_contract(PlanarPushEnv())
        config = contract['resolved_config']
        state = np.zeros((6, 10), np.float64)
        state[:, 0] = np.arange(6)*.0003
        state[:, 2] = np.arange(6)*.03
        state[:, 3] = np.linspace(.001, .004, 6)
        state[:, 6] = state[:, 0]-.052
        state[:, 7] = np.linspace(-.008, .008, 6)
        action = np.zeros((6, 2), np.float64)
        action[:, 0] = np.linspace(.01, .02, 6)
        future = state.copy()
        future[:, 0] += np.linspace(.0005, .0012, 6)
        future[:, 2] += np.linspace(-.002, .002, 6)
        future[:, 6] += .001
        features, deltas = encode(state, action), state_delta(state, future)
        split = np.array([1, 0, 1, 0, 0, 2], np.int8)
        schema = learning_schema(contract)
        identity = {'dataset_id': 'common-runner-fixture', 'manifest_sha256': 'a'*64,
                    'data_contract_hash': schema['data_contract_hash']}
        normalizer = fit_normalizer(features, deltas, split, schema,
                                    identity, np.arange(6, dtype=np.int64),
                                    np.zeros(6, np.int64))
        prepared = SimpleNamespace(arrays={'features': features, 'deltas': deltas,
                                          'split_code': split})
        return prepared, normalizer, config

    def test_fixed_update_validation_mutation_preserves_weights_adam_and_shuffle(self):
        prepared, normalizer, config = self.normalized_training_fixture()
        mutated = deepcopy(prepared)
        mutated.arrays['features'][[0, 2, 5]] += 1e4
        mutated.arrays['deltas'][[0, 2, 5]] -= 1e4
        before = {key: value.copy() for key, value in prepared.arrays.items()}
        normalizer_before = deepcopy(normalizer)
        torch_rng = torch.get_rng_state().clone()
        with patch.object(self.runner, 'evaluate_predictions',
                          side_effect=AssertionError('Fixed updates must not select using validation')):
            left = self.runner.fit_fixed_updates(prepared, normalizer, config, 11, 1011,
                                                updates=3, batch_size=2)
            right = self.runner.fit_fixed_updates(mutated, normalizer, config, 11, 1011,
                                                 updates=3, batch_size=2)
        np.testing.assert_array_equal(left['rows'], [1, 3, 4])
        np.testing.assert_array_equal(left['rows'], right['rows'])
        self.assertTrue(self.runner.tensor_tree_equal(left['model'].state_dict(), right['model'].state_dict()))
        self.assertTrue(self.runner.tensor_tree_equal(left['optimizer'].state_dict(), right['optimizer'].state_dict()))
        self.assertEqual(left['order_history'], right['order_history'])
        self.assertTrue(torch.equal(left['shuffle'].get_state(), right['shuffle'].get_state()))
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        self.assertTrue(all(int(v['step']) == 3 for v in left['optimizer'].state.values()))
        for key, expected in before.items():
            np.testing.assert_array_equal(prepared.arrays[key], expected)
        self.assertEqual(normalizer.metadata, normalizer_before.metadata)
        for key in normalizer.arrays:
            np.testing.assert_array_equal(normalizer.arrays[key], normalizer_before.arrays[key])
        # Deliberately no selected-epoch assertion: changed validation may
        # legitimately select another checkpoint after identical updates.

    def test_common_checkpoint_budget_and_seed_binding(self):
        good = SimpleNamespace(metadata={'seed': 11, 'epoch': 1, 'updates': 13})
        with patch.object(self.runner, 'load_checkpoint', return_value=good) as load:
            result = self.runner.load_common_checkpoint(Path('fixture'), None, None, None,
                                                        self.settings, 11, epoch=1, resume=True)
        self.assertIs(result, good)
        self.assertTrue(load.call_args.kwargs['resume'])
        self.assertEqual(load.call_args.kwargs['training_settings'], self.settings)
        defects = ({'seed': 23}, {'epoch': 0, 'updates': 0},
                   {'epoch': 101, 'updates': 1313}, {'updates': 12})
        for defect in defects:
            with self.subTest(defect=defect):
                bad = SimpleNamespace(metadata={**good.metadata, **defect})
                with patch.object(self.runner, 'load_checkpoint', return_value=bad), self.assertRaises(ValueError):
                    self.runner.load_common_checkpoint(Path('fixture'), None, None, None,
                                                        self.settings, 11)
        for seed in (True, 0, 12):
            with self.subTest(seed=seed), patch.object(self.runner, 'load_checkpoint', return_value=good), self.assertRaises(ValueError):
                self.runner.load_common_checkpoint(Path('fixture'), None, None, None,
                                                    self.settings, seed)
        for epoch in (2, True):
            with self.subTest(epoch=epoch), patch.object(self.runner, 'load_checkpoint', return_value=good), self.assertRaises(ValueError):
                self.runner.load_common_checkpoint(Path('fixture'), None, None, None,
                                                    self.settings, 11, epoch=epoch)

    def test_strict_json_rejects_nonfinite_before_creation_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'summary.json'
            with self.assertRaises(ValueError):
                self.runner.save(path, {'pass': True, 'loss': float('nan')})
            self.assertFalse(path.exists())
            self.runner.save(path, {'pass': False, 'reason': 'preserved'})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                self.runner.save(path, {'pass': True})
            self.assertEqual(path.read_bytes(), before)

    def test_failed_run_preserves_partial_seed_artifacts_and_explicit_failure(self):
        def failing_run(args, out):
            (out/'partial-seed.txt').write_text('seed47 interrupted; preserve prior work')
            raise RuntimeError('seed47 interrupted')
        with tempfile.TemporaryDirectory() as tmp:
            fake_root = Path(tmp)
            with patch.object(self.runner, 'ROOT', fake_root), \
                 patch.object(self.runner, 'run', side_effect=failing_run), \
                 patch.object(sys, 'argv', ['p3_train_common_mlp.py', '--run-id', 'unit-failure']), \
                 redirect_stdout(io.StringIO()):
                result = self.runner.main()
            out = fake_root/'runs/unit-failure'
            self.assertEqual(result, 1)
            self.assertEqual((out/'partial-seed.txt').read_text(), 'seed47 interrupted; preserve prior work')
            failure = json.loads((out/'failure.json').read_text())
            self.assertFalse(failure['pass'])
            self.assertEqual(failure['task_id'], 'P3-04')
            self.assertIn('seed47 interrupted', failure['message'])
            self.assertFalse((out/'summary.json').exists())
            artifact_index = json.loads((out/'artifact_hashes.json').read_text())['sha256']
            self.assertIn('partial-seed.txt', artifact_index)
            self.assertIn('failure.json', artifact_index)


if __name__ == '__main__':
    unittest.main()
