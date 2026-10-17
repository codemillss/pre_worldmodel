"""Synthetic execution-registration and forbidden-operation contract checks."""
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import p4_finite_step_probe as runner


class FiniteStepContractTests(unittest.TestCase):
    def write(self, root, name, value):
        path = root / name
        path.write_text(json.dumps(value))
        return path

    def registration(self, root):
        cases = [{'synthetic_case': index} for index in range(28)]
        design = {'alphas': [0., .25, .5, 1.], 'cases': cases}
        self.write(root, 'design.json', design)
        source = root / 'source.txt'
        source.write_text('fixed source')
        p = {'run_id': 'p4-finite-step-20261015-01', 'design_path': 'design.json',
             'alphas': design['alphas'], 'cases': cases,
             'bindings_sha256': {'source.txt': runner.sha(source), 'design.json': runner.sha(root / 'design.json')},
             'runtime': {'torch_version': str(torch.__version__), 'numpy_version': np.__version__,
                         'python_version': platform.python_version(), 'source_sha256': {str(source): runner.sha(source)}}}
        path = self.write(root, 'registration.json', p)
        return path, p

    def test_registration_binds_exact_case_order_alpha_design_and_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, p = self.registration(root)
            with patch.object(runner, 'ROOT', root):
                self.assertEqual(runner.validate(path, runner.sha(path)), p)

    def test_source_mutation_rejected_before_any_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = self.registration(root)
            (root / 'source.txt').write_text('changed')
            with patch.object(runner, 'ROOT', root), self.assertRaisesRegex(ValueError, 'Frozen source'):
                runner.validate(path, runner.sha(path))

    def test_alpha_grid_change_rejected_even_if_new_registration_digest_given(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, p = self.registration(root)
            p['alphas'] = [0., .1, .5, 1.]
            self.write(root, path.name, p)
            with patch.object(runner, 'ROOT', root), self.assertRaisesRegex(ValueError, 'Unregistered'):
                runner.validate(path, runner.sha(path))

    def test_runtime_change_rejected_without_model_or_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, p = self.registration(root)
            p['runtime']['torch_version'] = 'wrong'
            self.write(root, path.name, p)
            with patch.object(runner, 'ROOT', root), self.assertRaisesRegex(ValueError, 'Runtime version'):
                runner.validate(path, runner.sha(path))

    def test_guards_block_autograd_optimizer_and_checkpoint_writes_and_restore(self):
        original_save = torch.save
        counts, originals = runner.guards()
        try:
            for label, call in [('autograd.grad', lambda: torch.autograd.grad(None, None)),
                                ('Adam_instances', lambda: torch.optim.Adam([])),
                                ('torch.save', lambda: torch.save({}, 'forbidden.pt'))]:
                with self.assertRaisesRegex(RuntimeError, 'forbids'):
                    call()
                self.assertEqual(counts[label], 1)
            self.assertEqual(counts['physics.mj_step'], 0)
        finally:
            for obj, name, function in reversed(originals):
                setattr(obj, name, function)
        self.assertIs(torch.save, original_save)

    def frozen_fixture(self, root):
        class SyntheticModel(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.layer = torch.nn.Linear(11, 10, dtype=torch.float32)
        path = root / 'synthetic'
        path.mkdir()
        config = {'synthetic': True}
        normalizer = SimpleNamespace(metadata={'normalizer_hash': 'f' * 64})
        with torch.random.fork_rng(devices=[]):
            model = SyntheticModel(config)
        weights = model.state_dict()
        torch.save(weights, path / 'weights.pt')
        torch.save({'optimizer': {}}, path / 'training_state.pt')
        files = {name: runner.sha(path / name) for name in ('weights.pt', 'training_state.pt')}
        metadata = {'checkpoint_hash': 'a' * 64, 'config': config,
                    'normalizer_hash': normalizer.metadata['normalizer_hash']}
        self.write(path, 'checkpoint.json', metadata)
        spec = {'checkpoint_path': 'synthetic', 'checkpoint_hash': metadata['checkpoint_hash'],
                'checkpoint_json_sha256': runner.sha(path / 'checkpoint.json'),
                'files_sha256': files, 'model_tree_hash': runner.tree_hash(weights)}
        return SyntheticModel, spec, normalizer, config

    def test_narrow_weight_loader_never_instantiates_optimizer_or_changes_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic, spec, normalizer, config = self.frozen_fixture(root)
            rng = torch.get_rng_state().clone()
            counts, originals = runner.guards()
            try:
                with patch.object(runner, 'ROOT', root), patch.object(runner, 'DeltaMLP', synthetic):
                    model = runner.load_frozen_snapshot(spec, normalizer, config)
                self.assertFalse(model.training)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertEqual(runner.tree_hash(model.state_dict()), spec['model_tree_hash'])
                self.assertFalse(any(counts.values()))
            finally:
                for obj, name, function in reversed(originals):
                    setattr(obj, name, function)

    def test_narrow_weight_loader_rejects_changed_file_before_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic, spec, normalizer, config = self.frozen_fixture(root)
            (root / 'synthetic/weights.pt').write_bytes(b'changed weights')
            with patch.object(runner, 'ROOT', root), patch.object(runner, 'DeltaMLP', side_effect=AssertionError('Constructed before validating files')):
                with self.assertRaisesRegex(ValueError, 'checkpoint files'):
                    runner.load_frozen_snapshot(spec, normalizer, config)


if __name__ == '__main__':
    unittest.main()
