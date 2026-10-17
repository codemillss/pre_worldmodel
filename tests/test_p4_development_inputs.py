"""Archived-source wiring and adverse pre-decision context checks.

These tests are restoration diagnostics, not new candidate outcomes. The one
transition test replays two old applied-command intervals on a disposable env.
"""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tomllib
import unittest
from unittest.mock import patch

import numpy as np

from push_align.candidate_replay import snapshots_exact
from push_align.config import ROOT
from push_align.environment import PlanarPushEnv
from p4_development_inputs import DevelopmentInputError, load_development_snapshot, _read_steps


class DevelopmentInputsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = ROOT / 'runs/p3-05-20261010-01/planning_trials/N128-H5/R/smoke-00'
        cls.physical = json.loads((ROOT / 'runs/p4-02-20261012-01/manifest.json').read_text())['physical_config']
        cls.profile = tomllib.loads((ROOT / 'configs/acceptance.v1.toml').read_text())

    def load(self, tick=0, physical=None, profile=None):
        return load_development_snapshot(self.source, tick,
            self.physical if physical is None else physical,
            self.profile if profile is None else profile)

    def test_stored_initial_snapshot_task_and_actual_previous_command_match(self):
        original_physical, original_profile = deepcopy(self.physical), deepcopy(self.profile)
        data = self.load()
        self.assertTrue(data.proof['pass'])
        self.assertEqual(data.proof['snapshot_kind'], 'stored_complete')
        self.assertEqual(data.proof['source_regime'], 'noncontact')
        self.assertEqual(data.proof['source_sample_count'], 201)
        self.assertEqual(data.proof['source_active_samples'], 0)
        self.assertEqual(data.proof['prefix_restore_intervals'], 0)
        self.assertEqual(data.state.dtype, np.float64)
        self.assertEqual(data.state.shape, (10,))
        self.assertTrue(np.array_equal(data.context.previous_applied_action, np.zeros(2)))
        self.assertEqual(data.context.control_step, 0)
        self.assertEqual(asdict(data.gctx), data.original_plan['geometry_context_before'])
        env = PlanarPushEnv(self.physical); observed = env.restore(data.snapshot)
        self.assertTrue(snapshots_exact(data.snapshot, env.snapshot()))
        self.assertTrue(np.array_equal(observed.state, data.state))
        self.assertEqual(self.physical, original_physical)
        self.assertEqual(self.profile, original_profile)

    def test_transition_uses_two_recorded_intervals_and_restores_derived_complete_state(self):
        data = self.load(2)
        self.assertEqual(data.proof['snapshot_kind'], 'derived_checked_applied_prefix')
        self.assertEqual(data.proof['prefix_restore_intervals'], 2)
        self.assertEqual(data.proof['source_regime'], 'transition')
        self.assertEqual(data.proof['source_active_samples'], 108)
        self.assertTrue(data.proof['source_observed_state_exact'])
        self.assertTrue(all(p['raw_state_exact'] and p['raw_time_exact'] and p['geometry']['pass']
                            for p in data.proof['prefix']))
        self.assertTrue(np.array_equal(data.context.previous_applied_action,
                        data.original_plan['planner_context_before']['previous_applied_action']))
        self.assertEqual(data.proof['candidate_future_intervals'], 0)
        self.assertEqual(data.proof['formal_updates'], 0)

    def test_bool_negative_and_absent_ticks_are_rejected(self):
        for tick in (True, -1, 50):
            with self.subTest(tick=tick), self.assertRaises(DevelopmentInputError): self.load(tick)

    def test_different_requested_physics_is_rejected_before_new_execution(self):
        different = deepcopy(self.physical); different['contact']['pusher_object_friction'][0] += .01
        with self.assertRaisesRegex(DevelopmentInputError, 'physical config'):
            self.load(physical=different)

    def test_different_profile_cannot_relax_source_acceptance(self):
        different = deepcopy(self.profile); different['numeric']['replay_tolerance_fraction_of_task_tolerance'] *= 2
        with self.assertRaisesRegex(DevelopmentInputError, 'acceptance profile'):
            self.load(profile=different)

    def test_corrupted_stored_engine_payload_is_not_accepted_from_compact_plan(self):
        rows = _read_steps(self.source / 'steps.jsonl')
        rows[0]['plan']['physical_snapshot']['warning_counts'][0] += 1
        with patch('p4_development_inputs._read_steps', return_value=rows):
            with self.assertRaisesRegex(DevelopmentInputError, 'Stored full identity'):
                self.load()

    def test_proposed_or_warm_action_cannot_replace_actual_previous_command(self):
        rows = _read_steps(self.source / 'steps.jsonl')
        rows[0]['plan']['planner_context_before']['previous_applied_action'][0] = .01
        with patch('p4_development_inputs._read_steps', return_value=rows):
            with self.assertRaisesRegex(DevelopmentInputError, 'actual previously applied'):
                self.load()


if __name__ == '__main__': unittest.main()
