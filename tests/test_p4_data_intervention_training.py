"""Data-bound corruption/split/sampler/checkpoint checks; no optimizer updates."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import sys

import numpy as np
import torch

from push_align.config import ROOT
from push_align.data import digest
from push_align.data_intervention_training import (
    checked_slots, load_intervention_checkpoint, read_json, save_intervention_checkpoint,
    tree_hash, validate_core)
from push_align.fork_checkpoint import load_fork_checkpoint
from push_align.learning_data import load_learning_data


class DataInterventionSourceInventoryTest(unittest.TestCase):
    def test_virtual_torch_module_is_not_a_filesystem_source(self):
        from p4_train_data_intervention import source_hashes
        fake = SimpleNamespace(__file__='_codex_nonexistent_virtual_classes.py')
        with patch.dict(sys.modules, {'audit_virtual_module': fake}):
            inventory = source_hashes()
        self.assertIn('scripts/p4_train_data_intervention.py', inventory)
        self.assertNotIn('_codex_nonexistent_virtual_classes.py', inventory)
        self.assertTrue(all((ROOT / rel).is_file() and digest(ROOT / rel) == value for rel, value in inventory.items()))


class DataInterventionBindingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        main = ROOT / 'runs/p4-data-main-20261013-01'
        cls.bindings = read_json(main / 'slots_bound.json')
        with np.load(ROOT / 'runs/p4-02-20261012-01/train_windows.npz') as f:
            cls.old_rows = f['window_rows'].copy()
        with np.load(ROOT / 'runs/p4-02-20261012-01/seeds/seed-11/SK-lambda0.1/training_trace.npz') as f:
            cls.positions, cls.offsets = f['window_positions'].copy(), f['batch_offsets'].copy()
        cls.plan = read_json(main / 'registration.json')['plans'][0]
        cls.row = next(r for r in read_json(main / 'summary.json')['episodes'] if r['label'] == cls.plan['episode_id'])
        folder = main / cls.row['dataset']
        entry = read_json(folder / 'manifest.json')['episodes'][0]
        with np.load(folder / entry['npz']) as f:
            cls.arrays = {k: f[k].copy() for k in ('state', 'action_applied', 'action_requested', 'transition_valid', 'actual_dt')}

    def test_frozen_slots_and_exposure(self):
        slots, unchanged = checked_slots(self.bindings, self.old_rows, self.positions, self.offsets)
        self.assertEqual((len(slots), len(unchanged)), (384, 2576))

    def test_duplicate_slot_rejected(self):
        changed = deepcopy(self.bindings); changed['entries'][1]['slot'] = changed['entries'][0]['slot']
        with self.assertRaises(ValueError): checked_slots(changed, self.old_rows, self.positions, self.offsets)

    def test_changed_original_row_rejected(self):
        changed = self.old_rows.copy(); changed[self.bindings['entries'][0]['slot'], 0] += 1
        with self.assertRaises(ValueError): checked_slots(self.bindings, changed, self.positions, self.offsets)

    def test_changed_family_order_rejected(self):
        changed = deepcopy(self.bindings); changed['entries'][20]['family_rank'] = 0
        with self.assertRaises(ValueError): checked_slots(changed, self.old_rows, self.positions, self.offsets)

    def test_train_core_accepted(self):
        validate_core(self.plan, self.row, self.arrays)

    def test_development_rejected(self):
        changed = deepcopy(self.plan); changed['semantic_split'] = 'development'
        with self.assertRaises(ValueError): validate_core(changed, self.row, self.arrays)

    def test_warmup_rejected(self):
        changed = deepcopy(self.row); changed['role'] = 'warmup'
        with self.assertRaises(ValueError): validate_core(self.plan, changed, self.arrays)

    def test_invalid_transition_not_filtered(self):
        changed = deepcopy(self.arrays); changed['transition_valid'][8] = False
        with self.assertRaises(ValueError): validate_core(self.plan, self.row, changed)

    def test_changed_command_rejected(self):
        changed = deepcopy(self.arrays); changed['action_applied'][0, 1] += .001
        with self.assertRaises(ValueError): validate_core(self.plan, self.row, changed)


class DataInterventionCheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        learning = ROOT / 'runs/p3-02-20261009-01/learning_data'
        n = read_json(learning / 'normalizer/normalizer.json')['normalizer']
        prepared, cls.normalizer = load_learning_data(learning, n['schema'], n['dataset_identity'], n['normalizer_hash'])
        cls.config = prepared.metadata['contract']['resolved_config']
        oldrun = ROOT / 'runs/p4-02-20261012-01'
        manifest = read_json(oldrun / 'manifest.json')
        cls.bundle = load_fork_checkpoint(oldrun / 'seeds/seed-11/SK-lambda0.1/checkpoints/update-0100',
            cls.normalizer, cls.config, manifest['provenance'], manifest['budget'], seed=11,
            branch='SK-lambda0.1', resume=True, expected_run_manifest_sha256=digest(oldrun / 'manifest.json'))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'checkpoint'
        b = self.bundle
        self.metadata = save_intervention_checkpoint(self.path, b.model, b.optimizer,
            self.normalizer, self.config, arm='D11', sampler=b.sampler,
            manifest_sha256='a'*64, dataset_hash='b'*64, initial_checkpoint_hash='c'*64)

    def test_full_owned_roundtrip_without_rng_change(self):
        before = torch.get_rng_state().clone()
        loaded = load_intervention_checkpoint(self.path, self.normalizer, self.config, resume=True,
            expected_manifest_sha256='a'*64, expected_arm='D11', expected_dataset_hash='b'*64,
            expected_initial_checkpoint_hash='c'*64)
        self.assertEqual(tree_hash(loaded.model.state_dict()), tree_hash(self.bundle.model.state_dict()))
        self.assertEqual(tree_hash(loaded.optimizer.state_dict()), tree_hash(self.bundle.optimizer.state_dict()))
        self.assertEqual(tree_hash(loaded.sampler.state_dict()), tree_hash(self.bundle.sampler.state_dict()))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        next(iter(loaded.model.parameters())).data.add_(1)
        self.assertNotEqual(tree_hash(loaded.model.state_dict()), tree_hash(self.bundle.model.state_dict()))

    def test_wrong_arm_data_origin_manifest_rejected(self):
        for kw in ({'expected_arm': 'D00'}, {'expected_dataset_hash': 'e'*64},
                   {'expected_initial_checkpoint_hash': 'e'*64}, {'expected_manifest_sha256': 'e'*64}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                load_intervention_checkpoint(self.path, self.normalizer, self.config, **kw)

    def test_tensor_corruption_rejected(self):
        with (self.path / 'weights.pt').open('ab') as f: f.write(b'corrupt')
        with self.assertRaises(ValueError): load_intervention_checkpoint(self.path, self.normalizer, self.config)


if __name__ == '__main__': unittest.main()
