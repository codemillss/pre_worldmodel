"""Independent collection-gate witnesses; synthetic paths are not physics evidence.

These counterexamples distinguish complete numerical comparisons from matching
prefixes, independent feedback from command replay, and families from repeated
windows. Actual contact and sampled-domain validity still require the real run.
"""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import tomllib
import unittest
from unittest.mock import patch

import numpy as np

from push_align import PlanarPushEnv
from push_align.config import ROOT, load_pilot_config
from push_align.data import DataContractError, make_contract
from push_align.main_collection import MainCollectionContext, build_main_collection_plan
from push_align.preparation import PreparedData
from push_align.representation import encode, state_delta


def observation(x=0., yaw=0., time=0.):
    state = np.zeros(10, np.float64)
    state[0] = x
    state[2] = yaw
    state[6] = x-.05
    return SimpleNamespace(state=state, sim_time=time)


def path(count=3, *, x=0., yaw=0., time_offset=0., invalid=()):
    return [SimpleNamespace(observation=observation(x, yaw, .1*(i+1)+time_offset),
                            transition_valid=i not in invalid) for i in range(count)]


def coverage_row(family, *, split='validation', first_regimes=None,
                 mode='central', windows=50, valid_contact=100):
    first_regimes = ('noncontact', 'sustained_contact', 'transition') \
        if first_regimes is None else first_regimes
    return {
        'split': split, 'family_id': family, 'mode': mode,
        'windows': {str(h): {'total': windows*len(first_regimes),
                            'by_first_regime': {g: windows if g in first_regimes else 0
                                                for g in ('noncontact', 'sustained_contact', 'transition')}}
                    for h in (1, 5, 10)},
        'recorded_transitions': valid_contact, 'valid_transitions': valid_contact,
        'excluded_transitions': 0,
        'regime_counts_valid': {'noncontact': 0, 'sustained_contact': valid_contact, 'transition': 0},
        'translation_signal': True, 'rotation_signal': True, 'qualified_recontact_count': 0,
    }


class MainRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = str(ROOT/'scripts')
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        spec = importlib.util.spec_from_file_location('p3_main_runner_test_module',
                                                     ROOT/'scripts/p3_collect_main.py')
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)
        cls.settings = tomllib.loads((ROOT/'configs/p3_collection.v1.toml').read_text())
        cls.config = load_pilot_config()
        cls.contract = make_contract(PlanarPushEnv(cls.config))

    def compare(self, paths=None, starts=None, geometries=None, maximum=3):
        paths = [path(), path()] if paths is None else paths
        starts = [observation(), observation()] if starts is None else starts
        geometries = [[{'pass': True} for _ in p] for p in paths] \
            if geometries is None else geometries
        return self.runner.compare_paths(starts, paths, geometries, maximum,
                                         {'position_m': .0005, 'yaw_rad': math.pi/600}, 1e-9)

    def test_full_equal_paths_with_all_geometry_records_pass(self):
        result = self.compare()
        self.assertTrue(result['pass'])
        self.assertTrue(result['complete_both'])
        self.assertEqual(result['executed_lengths'], [3, 3])
        self.assertEqual(result['max_position_difference_m'], 0.)

    def test_identical_prefix_or_equal_short_execution_is_not_full_gate_pass(self):
        for counts in ((3, 2), (2, 2), (0, 0)):
            with self.subTest(counts=counts):
                result = self.compare(paths=[path(counts[0]), path(counts[1])])
                self.assertFalse(result['pass'])
                self.assertFalse(result['complete_both'])
                self.assertEqual(result['max_position_difference_m'], 0.)
                self.assertEqual(result['shared_intervals'], min(counts))

    def test_invalid_terminal_interval_is_not_complete_valid_comparison(self):
        result = self.compare(paths=[path(), path(invalid=(2,))])
        self.assertFalse(result['complete_both'])
        self.assertFalse(result['pass'])

    def test_geometry_fail_or_missing_checks_cannot_pass(self):
        for geometries in ([[], []], [[{'pass': True}]*3],
                           [[{'pass': True}]*3, [{'pass': True}]*2],
                           [[{'pass': True}]*3, [{'pass': True}, {'pass': False}, {'pass': True}]]):
            with self.subTest(geometries=geometries):
                result = self.compare(geometries=geometries)
                self.assertFalse(result['geometry_pass'])
                self.assertFalse(result['pass'])

    def test_initial_state_or_timestamp_difference_rejects_comparison(self):
        for changed in (observation(x=1e-12), observation(time=1e-12)):
            with self.subTest(changed=changed):
                result = self.compare(starts=[observation(), changed])
                self.assertFalse(result['identical_initial_observation_and_time'])
                self.assertFalse(result['pass'])

    def test_action_timestamp_alignment_is_separate_from_position_agreement(self):
        result = self.compare(paths=[path(), path(time_offset=1e-7)])
        self.assertEqual(result['max_position_difference_m'], 0.)
        self.assertFalse(result['same_shared_action_boundaries'])
        self.assertFalse(result['pass'])

    def test_maximum_interior_position_and_yaw_error_cannot_hide_at_endpoint(self):
        for coordinate, value in ((0, .000501), (2, math.radians(.301))):
            with self.subTest(coordinate=coordinate):
                modified = path()
                modified[1].observation.state[coordinate] = value
                self.assertFalse(self.compare(paths=[path(), modified])['pass'])

    def test_yaw_dt_difference_uses_two_pi_wrap(self):
        left, right = path(), path()
        left[0].observation.state[2] = math.pi-.0001
        right[0].observation.state[2] = -math.pi+.0001
        result = self.compare(paths=[left, right])
        self.assertAlmostEqual(result['max_yaw_difference_rad'], .0002, places=12)
        self.assertTrue(result['pass'])

    def test_four_families_cannot_become_five_by_repeated_windows_or_episodes(self):
        rows = [coverage_row(f'f{i}') for i in range(4)]
        rows += [coverage_row('f0', windows=100000) for _ in range(6)]
        result = self.runner.aggregate(rows, self.settings)['splits']['validation']
        self.assertEqual(result['episodes'], 10)
        self.assertEqual(result['families'], 4)
        self.assertEqual(result['windows']['5']['contributing_families'],
                         {'noncontact': 4, 'sustained_contact': 4, 'transition': 4})
        self.assertTrue(all(n < self.settings['coverage']['minimum_distinct_validation_families_per_regime']
                            for n in result['windows']['5']['contributing_families'].values()))

    def test_fifth_family_contributes_only_to_regime_with_valid_k5_window(self):
        rows = [coverage_row(f'f{i}') for i in range(4)]
        rows += [coverage_row('f4', first_regimes=('noncontact',), mode='contact_transition')]
        result = self.runner.aggregate(rows, self.settings)['splits']['validation']
        self.assertEqual(result['windows']['5']['contributing_families'],
                         {'noncontact': 5, 'sustained_contact': 4, 'transition': 4})
        # Prescribed transition mode and abundant active transitions do not
        # establish any valid K5 transition-regime window.
        self.assertEqual(result['modes']['contact_transition'], 1)

    def test_train_families_do_not_fill_validation_regime_gate(self):
        rows = [coverage_row(f'f{i}', split='train') for i in range(5)]
        result = self.runner.aggregate(rows, self.settings)['splits']['validation']
        self.assertEqual(result['episodes'], 0)
        self.assertEqual(result['windows']['5']['contributing_families'],
                         {'noncontact': 0, 'sustained_contact': 0, 'transition': 0})

    def test_actual_returned_command_replaces_proposal_context_without_alias(self):
        previous = np.zeros(2, np.float64)
        context = MainCollectionContext(None, previous)
        decision = SimpleNamespace(context=context, action=np.array([.04, 0.]))
        actual = np.array([.02, -.001], np.float64)
        changed = self.runner.next_context(decision, SimpleNamespace(applied_action=actual))
        np.testing.assert_array_equal(changed.previous_applied_action, [.02, -.001])
        self.assertFalse(changed.previous_applied_action.flags.writeable)
        actual[:] = 0.
        np.testing.assert_array_equal(changed.previous_applied_action, [.02, -.001])
        np.testing.assert_array_equal(context.previous_applied_action, [0., 0.])

    def test_registered_thresholds_purpose_and_failure_policy_cannot_be_relaxed(self):
        self.runner.validate_settings(deepcopy(self.settings), self.contract)
        changes = (
            ('coverage', 'minimum_distinct_validation_families_per_regime', 4),
            ('coverage', 'required_validation_horizon', 1),
            ('coverage', 'position_signal_min_m', .0005),
            ('coverage', 'window_regime', 'prescribed_mode'),
            ('physics_extension', 'require_full_equal_execution_lengths', False),
            ('physics_extension', 'comparison', 'replay nominal commands at both dt'),
            ('physics_extension', 'on_failure', 'replace failed representative'),
            ('limits', 'final_evaluation_allowed', True),
            ('limits', 'alignment_success_evaluation', True),
            ('limits', 'physical_command_limit_m_s', .04),
            ('limits', 'termination', 'terminate on pose and call it success'),
            ('resources', 'maximum_planned_transitions', 4799),
        )
        for group, field, value in changes:
            with self.subTest(field=field):
                settings = deepcopy(self.settings)
                settings[group][field] = value
                with self.assertRaises(ValueError):
                    self.runner.validate_settings(settings, self.contract)

    def gate_fixture(self, *, invalid_fine=False):
        plans, _ = build_main_collection_plan(self.settings, self.config)
        # Reverse source order and include all ninety plans. Selection must
        # use declared minimum ID rather than first iteration or best outcome.
        plans = list(reversed(plans))
        policy_instances = []
        saved = []
        class FakeEnv:
            def __init__(self, config):
                self.config = deepcopy(config)
                self.current = observation()
                self.ticks = 0
            def reset(self, initial):
                self.current = observation(initial.object_x, initial.object_yaw)
                self.ticks = 0
            def import_dt_comparison_state(self, other):
                self.current = deepcopy(other.current)
                self.ticks = 0
            def observe(self):
                return deepcopy(self.current)
            def snapshot(self):
                return SimpleNamespace(integration_state=self.current.state.copy())
            def step(self, action):
                before = self.current.state.copy()
                # Tiny dt-dependent differences cause different *observed*
                # feedback actions, while remaining under the position budget.
                self.current.state[0] += .0001+self.config['physics']['timestep_s']*.001
                self.current.sim_time += .1
                self.ticks += 1
                valid = not (invalid_fine and self.config['physics']['timestep_s'] == .00025
                             and self.ticks == 2)
                return SimpleNamespace(observation=self.observe(), transition_valid=valid,
                                       requested_action=action.copy(), applied_action=action.copy()*.5,
                                       diagnostics={'samples': []}, before_state=before)
        class FakePolicy:
            def __init__(self, config, settings, plan, selected):
                self.dt = config['physics']['timestep_s']
                self.plan = plan
                self.calls = []
                policy_instances.append(self)
            def initial_context(self):
                return MainCollectionContext(None, np.zeros(2, np.float64))
            def propose(self, state, context, elapsed, tick):
                action = np.array([state[0]+.001, .0], np.float64)
                self.calls.append((state.copy(), context.previous_applied_action.copy(), tick))
                return SimpleNamespace(action=action, context=context,
                                       diagnostics={'feasible': True, 'tick': tick})
        def save_trace(folder, label, snapshot, results):
            saved.append((label, len(results)))
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            stack.enter_context(patch.object(self.runner, 'PlanarPushEnv', FakeEnv))
            stack.enter_context(patch.object(self.runner, 'MainCollectionPolicy', FakePolicy))
            stack.enter_context(patch.object(self.runner, 'check_contact_geometry', return_value={'pass': True}))
            stack.enter_context(patch.object(self.runner, 'save_trace', side_effect=save_trace))
            stack.enter_context(patch.object(self.runner, 'gc', SimpleNamespace(collect=lambda: None)))
            result = self.runner.physics_gate(Path(tmp), plans, self.settings, self.config,
                                              self.contract, {})
        return result, policy_instances, saved

    def test_gate_uses_six_minimum_id_representatives_and_independent_observed_feedback(self):
        result, instances, saved = self.gate_fixture()
        self.assertTrue(result['pass'])
        self.assertEqual(len(result['cases']), 6)
        plans, _ = build_main_collection_plan(self.settings, self.config)
        self.assertEqual([r['episode_id'] for r in result['cases']],
                         [min(p['episode_id'] for p in plans if p['mode'] == mode)
                          for mode in ('noncontact', 'central', 'eccentric', 'contact_transition',
                                       'geometry_feedback', 'geometry_perturbed')])
        self.assertEqual([r['executed_lengths'] for r in result['cases']],
                         [[16, 16]]*4+[[128, 128]]*2)
        self.assertEqual(len(instances), 12)
        self.assertEqual(len(saved), 12)
        self.assertTrue(all(not r['requested_actions_identical'] for r in result['cases']))
        for i in range(0, len(instances), 2):
            base, fine = instances[i:i+2]
            self.assertIsNot(base, fine)
            self.assertEqual((base.dt, fine.dt), (.0005, .00025))
            np.testing.assert_array_equal(base.calls[0][0], fine.calls[0][0])
            self.assertNotEqual(base.calls[1][0][0], fine.calls[1][0][0])
            for policy in (base, fine):
                self.assertEqual(len(policy.calls), policy.plan['max_ticks'])
                for prior, current in zip(policy.calls, policy.calls[1:]):
                    # Returned actual command differs from proposal in this
                    # fixture. Replaying requested command context would fail.
                    np.testing.assert_array_equal(current[1], [(prior[0][0]+.001)*.5, 0.])

    def test_gate_retains_failed_short_fine_prefix_without_replacing_representatives(self):
        result, instances, saved = self.gate_fixture(invalid_fine=True)
        self.assertFalse(result['pass'])
        self.assertEqual(len(result['cases']), 6)
        self.assertTrue(all(r['executed_lengths'][1] == 2 for r in result['cases']))
        self.assertTrue(all(not r['complete_both'] and not r['pass'] for r in result['cases']))
        self.assertTrue(all(r['terminal_reasons'][1] == 'physical_or_geometry_failure'
                            for r in result['cases']))
        self.assertEqual(len(instances), 12)
        self.assertEqual([n for label, n in saved if label.endswith('_half')], [2]*6)

    def test_prepare_calls_train_only_fit_with_validation_mutation_positive_control(self):
        current = np.zeros((8, 10), np.float64)
        current[:, 0] = np.arange(8)*.001
        current[:4, 6] = current[:4, 0]-.05
        current[4:, 6] = current[4:, 0]+.2
        future = current.copy()
        future[:, 0] += .001
        actions = np.full((8, 2), .01, np.float64)
        arrays = {'current_state': current, 'next_state': future, 'applied_action': actions,
                  'features': encode(current, actions), 'deltas': state_delta(current, future),
                  'split_code': np.array([0]*4+[1]*4, np.int8),
                  'episode_index': np.array([0]*4+[1]*4, np.int64),
                  'transition_index': np.tile(np.arange(4, dtype=np.int64), 2)}
        prepared = PreparedData(arrays, {'dataset_id': 'synthetic-normalizer-witness',
                                        'manifest_sha256': 'a'*64,
                                        'data_contract_hash': self.runner.contract_hash(self.contract),
                                        'contract': self.contract}, {})
        stored = {}
        def store(folder, data, normalizer):
            stored.update(data=deepcopy(data), normalizer=normalizer)
        def restore(*args):
            return stored['data'], stored['normalizer']
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(self.runner, 'prepare_dataset', return_value=prepared), \
             patch.object(self.runner, 'save_learning_data', side_effect=store), \
             patch.object(self.runner, 'load_learning_data', side_effect=restore):
            checks, metadata = self.runner.prepare(Path(tmp), Path(tmp)/'synthetic')
        self.assertEqual(metadata['fit_count'], 4)
        self.assertAlmostEqual(stored['normalizer'].arrays['input_mean'][0], -.05, places=14)
        self.assertTrue(checks['train_only_fit_count'])
        self.assertTrue(checks['validation_stats_invariance'])

    def audit_fixture(self, *, change=None, late_failure=False):
        """Synthetic policy trace; raw-loader and contact physics are mocked."""
        plans, episodes, stored_records, policy_calls = [], [], {}, []
        for i, split in enumerate(('train', 'validation')):
            plan = {'episode_index': i, 'episode_id': f'fixture-{i}',
                    'family_id': f'fixture-family-{i}', 'split': split,
                    'mode': 'geometry_feedback', 'initial_spec': {'object_x': .001*i}}
            plans.append(plan)
            current = observation(x=.001*i).state
            states = np.vstack([current, current.copy()])
            meta = {'episode_index': i, 'episode_id': plan['episode_id'],
                    'family_id': plan['family_id'], 'split': split,
                    'collection_policy': {'predeclared_plan': plan},
                    'initial_record': {'initial_spec': plan['initial_spec']},
                    'terminal_reason': 'controller_infeasible'}
            arrays = {'state': states, 'sim_time': np.array([0., .1], np.float64),
                      'action_requested': np.array([[.04, 0.]], np.float64),
                      'action_applied': np.array([[.02, 0.]], np.float64),
                      'transition_valid': np.array([True])}
            episodes.append(SimpleNamespace(metadata=meta, arrays=arrays, transition_count=1))
            def diagnostic(tick):
                return {'tick': tick, 'feasible': tick == 0, 'holding': False,
                        'noise_applied': False, 'mode_after': 'PUSH'}
            records = [
                {'executed': True, 'diagnostics': diagnostic(0),
                 'requested_action': [.04, 0.], 'applied_action': [.02, 0.]},
                {'executed': False, 'diagnostics': diagnostic(1),
                 'requested_action': [0., 0.]},
            ]
            if change == 'drop_tail':
                records.pop()
            elif change == 'requested_action':
                records[0]['requested_action'] = [0., 0.]
            elif change == 'infeasible_diagnostics':
                records[-1]['diagnostics']['feasible'] = True
            elif change == 'infeasible_action':
                records[-1]['requested_action'] = [.01, 0.]
            stored_records[plan['episode_id']] = {'records': records}
        class Policy:
            def __init__(self, config, settings, plan, selected):
                self.plan = plan
            def initial_context(self):
                return MainCollectionContext(None, np.zeros(2, np.float64))
            def propose(self, state, context, elapsed, tick):
                policy_calls.append((self.plan['episode_id'], tick,
                                     context.previous_applied_action.copy()))
                return SimpleNamespace(action=np.array([.04, 0.] if tick == 0 else [0., 0.]),
                                       context=context, diagnostics=diagnostic(tick))
        def stream(*args):
            yield from episodes
            if late_failure:
                raise DataContractError('late global family/manifest failure')
        def measured(episode, *args, **kwargs):
            return coverage_row(episode.metadata['family_id'], split=episode.metadata['split'],
                                mode='geometry_feedback')
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            out = Path(tmp)
            (out/'policy').mkdir()
            for name, data in stored_records.items():
                (out/'policy'/f'{name}.json').write_text(json.dumps(data))
            stack.enter_context(patch.object(self.runner, 'iter_dataset', side_effect=stream))
            stack.enter_context(patch.object(self.runner, 'MainCollectionPolicy', Policy))
            stack.enter_context(patch.object(self.runner, 'episode_coverage', side_effect=measured))
            result = self.runner.audit(out, out/'synthetic', plans, self.settings,
                                       self.contract, {})
        return result, policy_calls

    def test_infeasible_terminal_proposal_is_rechecked_from_last_state_and_actual_context(self):
        result, calls = self.audit_fixture()
        self.assertTrue(result[2])
        self.assertEqual([(episode, tick) for episode, tick, _ in calls],
                         [('fixture-0', 0), ('fixture-0', 1),
                          ('fixture-1', 0), ('fixture-1', 1)])
        for _, tick, previous in calls:
            np.testing.assert_array_equal(previous, [0., 0.] if tick == 0 else [.02, 0.])

    def test_missing_terminal_infeasible_proposal_does_not_silently_pass_replay(self):
        with self.assertRaises(ValueError):
            self.audit_fixture(change='drop_tail')

    def test_executed_requested_action_field_and_infeasible_tail_are_bound_to_replay(self):
        for change in ('requested_action', 'infeasible_diagnostics', 'infeasible_action'):
            with self.subTest(change=change):
                result, _ = self.audit_fixture(change=change)
                self.assertFalse(result[2])

    def test_audit_exhausts_stream_and_late_global_failure_cannot_return_partial_result(self):
        with self.assertRaisesRegex(DataContractError, 'late global family/manifest failure'):
            self.audit_fixture(late_failure=True)


if __name__ == '__main__':
    unittest.main()
