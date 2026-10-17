"""Real archived G diagnostics and immutable regression-inventory guards.

Inventory discovery reads test IDs only; it never runs the discovered suite.
The tick-zero source is restored without executing any physical command.
"""
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
import tomllib
import unittest
from unittest.mock import patch

import numpy as np

from push_align.config import ROOT
from push_align.environment import PlanarPushEnv
from push_align.planning_protocol import GeometryProposalAdapter

scripts_path = str(ROOT / 'scripts')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from p4_compare_candidates import FORKS, save, tests_in, validate_test_inventory
from p4_development_inputs import load_development_snapshot


class CandidateRunnerIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior_log = (FORKS / 'tests.txt').read_text(encoding='utf-8')
        cls.inherited_ids = re.findall(
            r'^test_\S+ \(([^)]+)\)', cls.prior_log, re.MULTILINE)
        # Discovering only IDs cannot recursively execute this test class.
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
        cls.current_ids = list(tests_in(suite))
        if any(name.startswith('unittest.loader._FailedTest.')
               for name in cls.current_ids):
            raise RuntimeError('Current inventory contains an import failure')
        validate_test_inventory(cls.current_ids, cls.prior_log)

    def test_real_tick_zero_numpy_bool_proposal_saves_and_reloads_without_physics(self):
        physical = json.loads((FORKS / 'manifest.json').read_text())['physical_config']
        profile = tomllib.loads((ROOT / 'configs/acceptance.v1.toml').read_text())
        protocol = json.loads((ROOT / 'configs/p3_planning.v1.json').read_text())
        source = ROOT / 'runs/p3-05-20261010-01/planning_trials/N128-H5/R/smoke-00'
        # A stored tick-zero snapshot requires no old prefix or new future step.
        with patch.object(PlanarPushEnv, 'step',
                          side_effect=AssertionError('Physical commands are forbidden')) as step:
            data = load_development_snapshot(source, 0, physical, profile)
            adapter = GeometryProposalAdapter(physical,
                protocol['geometry_selected_overrides'], protocol['geometry_transition'])
            proposal = adapter.propose(data.state.copy(), data.task, data.gctx,
                data.status.elapsed_s, control_tick=0)
        step.assert_not_called()
        self.assertEqual(data.proof['prefix_restore_intervals'], 0)
        self.assertEqual(data.proof['candidate_future_intervals'], 0)
        self.assertEqual(proposal.diagnostics['future_physics_queries'], 0)
        self.assertIsInstance(proposal.diagnostics['offset_saturated'], np.bool_)
        self.assertEqual(proposal.diagnostics, data.original_plan['geometry_proposal'])
        payload = {'geometry_proposal': proposal.diagnostics,
                   'geometry_context_after': asdict(proposal.context)}
        # This negative control reproduces the run01 failure using real values.
        with self.assertRaises(TypeError):
            json.dumps(payload, allow_nan=False)
        with TemporaryDirectory(prefix='p4-candidate-json-') as directory:
            path = Path(directory) / 'candidate_identity.json'
            save(path, payload)
            def reject_nonfinite(value):
                raise AssertionError(f'Nonstandard JSON constant: {value}')
            restored = json.loads(path.read_text(encoding='utf-8'),
                                  parse_constant=reject_nonfinite)
        self.assertEqual(restored, payload)
        self.assertIs(type(restored['geometry_proposal']['offset_saturated']), bool)
        self.assertEqual(restored['geometry_proposal']['offset_saturated'],
                         bool(proposal.diagnostics['offset_saturated']))

    def test_same_total_cannot_replace_one_inherited_test_with_dummy(self):
        baseline = validate_test_inventory(self.current_ids, self.prior_log)
        self.assertEqual(baseline['inherited_test_count'], 714)
        self.assertEqual(len(set(self.inherited_ids)), 714)
        removed = sorted(self.inherited_ids)[0]
        changed = self.current_ids.copy()
        changed[changed.index(removed)] = (
            'test_p4_candidate_runner.DummyReplacement.test_same_total_old_removed')
        self.assertEqual(len(changed), len(self.current_ids))
        self.assertEqual(len(set(changed)), len(changed))
        self.assertNotIn(removed, changed)
        with self.assertRaisesRegex(ValueError, 'Inherited714'):
            validate_test_inventory(changed, self.prior_log)

    def test_same_total_cannot_replace_one_diagnostic_test_with_dummy(self):
        baseline = validate_test_inventory(self.current_ids, self.prior_log)
        self.assertEqual(baseline['required_new_module_counts'], {
            'test_p4_candidate_replay': 17,
            'test_p4_candidate_diagnostics': 31,
            'test_p4_development_inputs': 7,
        })
        diagnostic_ids = [name for name in self.current_ids
                          if name.split('.')[0] == 'test_p4_candidate_diagnostics']
        self.assertEqual(len(diagnostic_ids), 31)
        changed = self.current_ids.copy()
        changed[changed.index(sorted(diagnostic_ids)[0])] = (
            'test_p4_candidate_runner.DummyReplacement.test_same_total_new_removed')
        self.assertEqual(len(changed), len(self.current_ids))
        self.assertEqual(len(set(changed)), len(changed))
        self.assertTrue(set(self.inherited_ids) <= set(changed))
        with self.assertRaisesRegex(ValueError, '17/31/7'):
            validate_test_inventory(changed, self.prior_log)


if __name__ == '__main__':
    unittest.main()
