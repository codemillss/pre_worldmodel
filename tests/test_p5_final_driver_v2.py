"""Driver integration with scripted physics and synthetic predictions only.

No MuJoCo model/environment or trained network is constructed or stepped.
The unchanged actual executor is used with the same synthetic environment as
its fault tests, so these checks cover wiring as well as pure serialization.
"""
from copy import deepcopy
from dataclasses import replace
import ast
import json
import os
from pathlib import Path
import tempfile
import tomllib
import tracemalloc
import unittest
from unittest.mock import patch

import numpy as np

from push_align import final_driver_v2, final_recording
from push_align.config import ROOT, load_pilot_config
from push_align.final_driver_v2 import (DriverStop, LiveResourceGuard, exact_comparison,
    plan_index_records, read_full_trace, run_recorded_episode, write_full_trace)
from push_align.final_execution import execute_episode
from push_align.final_recording import file_sha256, read_plan_index, read_raw
from test_p5_final_execution import SyntheticEnv, state_at_goal, stationary_predictor


GIB = 1024**3


class FinalDriverV2Test(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'frozen_source.txt'
        self.source.write_text('synthetic immutable source')
        self.bindings = {str(self.source): file_sha256(self.source)}
        self.identity = {'model': {'synthetic': 'stationary'}, 'normalizer': {'synthetic': 'identity'}}
        self.config = load_pilot_config()
        self.planner = deepcopy(self.config)
        self.planner['planner'].update(candidate_count=16, horizon=5, block_lengths=[2, 3])
        self.geometry = json.loads((ROOT/'configs/p3_geometry_selected.v2.json').read_text())
        self.transition = json.loads((ROOT/'configs/p3_geometry_transition.v2.json').read_text())
        self.profile = tomllib.loads((ROOT/'configs/acceptance.v1.toml').read_text())
        self.counter = 0

    def guard(self, *, free_bytes=None, **overrides):
        args = dict(remaining_required_bytes=lambda: 3*GIB,
            cell_write_allowance_bytes=2*1024**2, free_bytes=free_bytes or (lambda: 10*GIB))
        args.update(overrides)
        return LiveResourceGuard(self.root, **args)

    def run_cell(self, folder='original', *, env=None, guard=None, reference=None, **overrides):
        self.counter += 1
        env = env or SyntheticEnv(state_at_goal())
        args = dict(case_id='synthetic-known', episode_index=3, controller_id='R',
            execution_id=f'synthetic-{self.counter}', predictor=stationary_predictor,
            physical_config=self.config, planner_config=self.planner,
            geometry_parameters=self.geometry, transition_parameters=self.transition,
            candidate_seed_base=4711, candidate_stream_id=0, max_intervals=2,
            run_scope='development_preflight', penetration_budget_m=.0005,
            origin={'source': 'synthetic-known-input', 'not_final_test': True})
        args.update(overrides)
        identity = self.identity if args['predictor'] is not None else {'model': None, 'normalizer': None}
        return run_recorded_episode(env, env.snapshot(), folder=self.root/folder,
            identity=identity, source_bindings=self.bindings, profile=self.profile,
            guard=guard or self.guard(), reference_manifest=reference, **args)

    def test_original_saves_all_raw_index_and_only_declared_full_ticks(self):
        result = self.run_cell(max_intervals=3)
        self.assertEqual(result.status, 'COMPLETE_RECORDED')
        self.assertEqual(len(result.execution.results), 3)
        self.assertEqual([row['tick'] for row in result.manifest['full_traces']], [0])
        self.assertEqual(result.manifest['missing_predeclared_ticks'], [29, 299])
        original, raw = read_raw(result.manifest['raw']['prefix'],
            expected_sha256=result.manifest['raw']['files_sha256'])
        self.assertEqual(exact_comparison(raw, result.execution.results)['status'], 'PASS_EXACT')
        records = read_plan_index(result.manifest['plan_index']['path'])
        self.assertEqual([row['tick'] for row in records], [0, 1, 2])
        self.assertEqual(exact_comparison(records, plan_index_records(result.execution))['status'], 'PASS_EXACT')
        self.assertTrue(all(phase in {row['phase'] for row in result.manifest['resources']['guard_checks']}
                            for phase in ('before_cell', 'before_action', 'before_write')))

    def test_full_trace_reconstructs_complete_typed_input_and_all_model_arrays(self):
        result = self.run_cell()
        full_input, arrays, metadata = read_full_trace(result.manifest['full_traces'][0])
        self.assertEqual(full_input['episode_index'], 3)
        self.assertEqual(full_input['candidate_seed_base'], 4711)
        self.assertEqual(full_input['candidate_stream_id'], 0)
        self.assertEqual(full_input['rng_keys'], [4711, 3, 0, 0])
        self.assertEqual(full_input['identity'], self.identity)
        self.assertEqual(full_input['source_bindings'], self.bindings)
        self.assertEqual(full_input['physical_config'], self.config)
        self.assertEqual(full_input['planner_config'], self.planner)
        self.assertEqual(full_input['planner_context'].control_step, 0)
        self.assertEqual(full_input['observation'].schema_version, 'push-align-state-v1')
        self.assertEqual(exact_comparison(full_input['snapshot'], result.execution.initial_snapshot)['status'], 'PASS_EXACT')
        self.assertEqual(exact_comparison(arrays['predicted_states'],
            result.execution.plan_results[0].predictions.states)['status'], 'PASS_EXACT')
        self.assertEqual(exact_comparison(arrays['candidates'],
            result.execution.plan_results[0].candidates.actions)['status'], 'PASS_EXACT')
        self.assertIn('task_status', full_input)
        self.assertIn('task_spec', full_input)
        self.assertIn('initial_observation', full_input)
        self.assertIn('geometry_context', full_input)

    def test_fresh_calls_controller_again_before_exactly_reusing_original(self):
        original = self.run_cell()
        env = SyntheticEnv(state_at_goal())
        calls = []
        def synthetic_predictor(state, action):
            calls.append(action.copy())
            return np.zeros_like(state)
        fresh = self.run_cell('fresh', env=env, reference=original.manifest_path,
                              predictor=synthetic_predictor)
        self.assertEqual(env.calls, 2)
        self.assertTrue(calls)
        self.assertEqual(fresh.status, 'COMPLETE_RECORDED')
        self.assertIsNone(fresh.manifest['raw'])
        self.assertIsNone(fresh.manifest['plan_index'])
        self.assertEqual(fresh.comparison['boundary']['status'], 'PASS_REPRODUCTION')
        self.assertEqual(fresh.comparison['raw_exact']['status'], 'PASS_EXACT')
        self.assertEqual(fresh.comparison['plan_index_exact']['status'], 'PASS_EXACT')
        self.assertEqual(fresh.manifest['raw_reference_reuse']['prefix'], original.manifest['raw']['prefix'])
        self.assertFalse(list((self.root/'fresh').glob('raw_execution*')))
        self.assertEqual(fresh.witness['original_raw_reference'], original.witness['raw_reference'])

    def test_per_index_component_difference_at_nonstandard_tick_preserves_full_input_and_stops(self):
        original = self.run_cell()
        def altered(*args, **kwargs):
            output = execute_episode(*args, **kwargs)
            component = next(iter(output.plan_results[1].costs.components.values()))
            component[0] += .125
            return output
        guard = self.guard()
        with patch('push_align.final_driver_v2.execute_episode', side_effect=altered):
            fresh = self.run_cell('fresh', reference=original.manifest_path, guard=guard)
        self.assertEqual(fresh.status, 'STOP_REVIEW')
        self.assertEqual(fresh.comparison['raw_exact']['status'], 'PASS_EXACT')
        self.assertEqual(fresh.comparison['plan_index_exact']['status'], 'FAIL_EXACT')
        self.assertIn(1, fresh.comparison['differing_ticks'])
        self.assertEqual([row['tick'] for row in fresh.manifest['full_traces']], [0, 1])
        self.assertIsNotNone(fresh.manifest['raw'])
        self.assertIsNotNone(fresh.manifest['plan_index'])
        self.assertGreater(guard.mismatch_used_bytes, 0)
        self.assertFalse(guard.active)

    def test_raw_substep_difference_is_detected_even_when_boundary_passes(self):
        original = self.run_cell()
        def altered(*args, **kwargs):
            output = execute_episode(*args, **kwargs)
            output.results[1].substep_states[0, 4] = -.0
            return output
        with patch('push_align.final_driver_v2.execute_episode', side_effect=altered):
            fresh = self.run_cell('fresh', reference=original.manifest_path)
        self.assertEqual(fresh.comparison['boundary']['status'], 'PASS_REPRODUCTION')
        self.assertEqual(fresh.comparison['raw_exact']['status'], 'FAIL_EXACT')
        self.assertEqual(fresh.status, 'STOP_REVIEW')
        self.assertIsNotNone(fresh.manifest['raw'])
        self.assertIn(1, fresh.comparison['differing_ticks'])

    def test_all_invalid_terminal_tick_preserves_all_slots_and_full_current_input(self):
        calls = []
        def becomes_invalid(state, action):
            calls.append(None)
            return np.zeros_like(state) if len(calls) <= 5 else np.full_like(state, np.nan)
        env = SyntheticEnv(state_at_goal())
        result = self.run_cell(predictor=becomes_invalid, env=env, max_intervals=3)
        self.assertEqual(result.execution.termination, 'planner_invalid')
        self.assertEqual(env.calls, 1)
        self.assertEqual(result.status, 'COMPLETE_RECORDED')
        self.assertEqual([row['tick'] for row in result.manifest['full_traces']], [0, 1])
        rows = read_plan_index(result.manifest['plan_index']['path'])
        self.assertFalse(rows[-1]['arrays']['valid'].any())
        self.assertEqual(rows[-1]['arrays']['valid'].shape, (16,))
        self.assertTrue(np.isposinf(rows[-1]['arrays']['total_cost']).all())
        full, arrays, metadata = read_full_trace(result.manifest['full_traces'][-1])
        self.assertEqual(full['task_status'].phase, 'running')
        self.assertEqual(full['planner_context'].control_step, 1)
        self.assertEqual(full['tick'], 1)
        self.assertEqual(full['rng_keys'], [4711, 3, 1, 0])
        self.assertEqual(metadata['plan']['selected_index'], None)

    def test_partial_invalid_slots_are_not_filtered_or_renumbered(self):
        def mixed(state, action):
            output = np.zeros_like(state)
            output[action[:, 0] > .0001] = np.nan
            return output
        result = self.run_cell(predictor=mixed)
        records = read_plan_index(result.manifest['plan_index']['path'])
        for row in records:
            arrays = row['arrays']
            self.assertEqual(arrays['valid'].shape, (16,))
            self.assertGreater(arrays['valid'].sum(), 0)
            self.assertLess(arrays['valid'].sum(), 16)
            self.assertTrue(np.isposinf(arrays['total_cost'][~arrays['valid']]).all())
            self.assertTrue((arrays['reasons'][~arrays['valid']] != '').all())

    def test_G_fresh_retains_predeclared_actual_input_capture(self):
        original = self.run_cell(predictor=None, controller_id='G')
        fresh = self.run_cell('fresh', reference=original.manifest_path, predictor=None, controller_id='G')
        self.assertEqual(fresh.status, 'COMPLETE_RECORDED')
        self.assertEqual([row['tick'] for row in fresh.manifest['full_traces']], [0])
        full, arrays, metadata = read_full_trace(fresh.manifest['full_traces'][0])
        self.assertEqual(arrays, {})
        self.assertIsNone(metadata['plan'])
        self.assertEqual(full['identity'], {'model': None, 'normalizer': None})

    def test_columnar_codec_is_injectable_with_same_exact_fresh_contract(self):
        from push_align import final_recording_columns
        original = self.run_cell(raw_codec=final_recording_columns)
        self.assertEqual(original.status, 'COMPLETE_RECORDED')
        self.assertEqual(original.manifest['raw']['schema'], final_recording_columns.SCHEMA)
        self.assertTrue(any(path.endswith('.raw.columns.xz')
                            for path in original.manifest['raw']['files_sha256']))
        fresh = self.run_cell('fresh', reference=original.manifest_path, raw_codec=final_recording_columns)
        self.assertEqual(fresh.status, 'COMPLETE_RECORDED')
        self.assertEqual(fresh.comparison['raw_exact']['status'], 'PASS_EXACT')
        self.assertEqual(fresh.comparison['plan_index_exact']['status'], 'PASS_EXACT')
        self.assertEqual(fresh.comparison['boundary']['status'], 'PASS_REPRODUCTION')

    def test_whole_budget_prevents_cell_before_restore_or_directory_creation(self):
        env = SyntheticEnv(state_at_goal())
        with patch.object(env, 'restore', wraps=env.restore) as restore:
            with self.assertRaises(DriverStop):
                self.run_cell(env=env, guard=self.guard(free_bytes=lambda: 2*GIB))
            restore.assert_not_called()
        self.assertFalse((self.root/'original').exists())

    def test_capacity_decline_stops_before_next_action_but_keeps_actual_prefix(self):
        values = iter([10*GIB, 10*GIB, int(2.5*GIB)])
        last = [int(2.5*GIB)]
        def free():
            last[0] = next(values, last[0])
            return last[0]
        env = SyntheticEnv(state_at_goal())
        result = self.run_cell(env=env, guard=self.guard(free_bytes=free))
        self.assertEqual(env.calls, 1)
        self.assertEqual(result.execution.execution_status, 'infrastructure_error')
        self.assertEqual(result.status, 'STOP_REVIEW')
        self.assertEqual(len(read_raw(result.manifest['raw']['prefix'])[1]), 1)
        self.assertEqual(result.execution.error['type'], 'DriverStop')

    def test_source_mismatch_stops_before_controller_and_original_unchanged(self):
        self.source.write_text('changed source')
        env = SyntheticEnv(state_at_goal())
        with self.assertRaises(DriverStop):
            self.run_cell(env=env)
        self.assertEqual(env.calls, 0)
        self.assertEqual(self.source.read_text(), 'changed source')
        self.assertFalse((self.root/'original').exists())

    def test_missing_original_stops_before_new_execution(self):
        original = self.run_cell()
        Path(original.manifest['plan_index']['path']).unlink()
        env = SyntheticEnv(state_at_goal())
        with self.assertRaisesRegex(ValueError, 'Missing or changed original'):
            self.run_cell('fresh', env=env, reference=original.manifest_path)
        self.assertEqual(env.calls, 0)
        self.assertFalse((self.root/'fresh').exists())

    def test_missing_original_during_comparison_preserves_fresh_prefix_as_incomplete(self):
        original = self.run_cell()
        with patch('push_align.final_recording.read_raw', side_effect=OSError('synthetic vanished original')):
            fresh = self.run_cell('fresh', reference=original.manifest_path)
        self.assertEqual(fresh.status, 'STOP_REVIEW')
        self.assertEqual(fresh.comparison['status'], 'INCOMPLETE')
        self.assertIsNotNone(fresh.manifest['raw'])
        self.assertEqual(fresh.manifest['raw']['interval_count'], 2)
        self.assertIsNotNone(fresh.manifest['plan_index'])

    def test_failed_raw_write_retains_partial_file_and_reports_incomplete(self):
        def fail(prefix, *args, **kwargs):
            Path(str(prefix)+'.raw.npz').write_bytes(b'synthetic partial')
            raise OSError('synthetic writer failure')
        with patch('push_align.final_recording.write_raw', side_effect=fail):
            result = self.run_cell()
        self.assertEqual(result.status, 'STOP_INCOMPLETE_RECORDING')
        partial = self.root/'original/raw_execution.raw.npz'
        self.assertEqual(partial.read_bytes(), b'synthetic partial')
        self.assertIn(str(partial.resolve()), result.manifest['partial_files_sha256'])
        self.assertTrue(result.manifest_path.is_file())
        self.assertEqual(result.manifest['recording_error']['type'], 'OSError')

    def test_existing_episode_path_is_never_overwritten(self):
        original = self.run_cell()
        digest = file_sha256(original.manifest_path)
        with self.assertRaises(FileExistsError):
            self.run_cell()
        self.assertEqual(file_sha256(original.manifest_path), digest)

    def test_budget_mismatch_allowance_checked_before_fresh_execution(self):
        original = self.run_cell()
        env = SyntheticEnv(state_at_goal())
        with self.assertRaises(DriverStop):
            self.run_cell('fresh', env=env, reference=original.manifest_path,
                guard=self.guard(mismatch_limit_bytes=1))
        self.assertEqual(env.calls, 0)

    def test_plan_record_missing_decision_is_not_silently_filled(self):
        result = self.run_cell()
        result.execution.decisions.pop()
        with self.assertRaises(ValueError):
            plan_index_records(result.execution)

    def test_exact_compare_covers_dtype_shape_signedzero_nan_payload_and_diagnostics(self):
        self.assertEqual(exact_comparison(np.array([0.]), np.array([-0.]))['status'], 'FAIL_EXACT')
        self.assertEqual(exact_comparison(np.zeros(1, np.float64), np.zeros(1, np.float32))['status'], 'FAIL_EXACT')
        self.assertEqual(exact_comparison(np.zeros((1, 1)), np.zeros(1))['status'], 'FAIL_EXACT')
        a = np.array([0x7ff8000000000123], np.uint64).view(np.float64)[0]
        b = np.array([0x7ff8000000000124], np.uint64).view(np.float64)[0]
        self.assertEqual(exact_comparison({'nested': float(a)}, {'nested': float(b)})['status'], 'FAIL_EXACT')
        self.assertEqual(exact_comparison({'nested': float(a)}, {'nested': float(a)})['status'], 'PASS_EXACT')

    def test_generic_full_input_codec_preserves_dataclasses_scalarbits_and_numpy_types(self):
        env = SyntheticEnv(state_at_goal())
        value = {'snapshot': env.snapshot(), 'observation': env.observation(),
            'scalar': np.int32(17), 'float32': np.float32(-0.), 'plain': -0.,
            'nested': (np.array([1, 2], np.int16), None, 2**140)}
        receipt = write_full_trace(self.root/'full', tick=7, captured_input=value)
        restored, arrays, metadata = read_full_trace(receipt)
        self.assertEqual(exact_comparison(value, restored)['status'], 'PASS_EXACT')
        self.assertEqual(arrays, {})

    def test_exact_comparison_streams_without_json_or_mirrored_diagnostic_trees(self):
        left = [{'state': [float(index), -0., .01], 'flags': [True, False], 'index': index}
                for index in range(6000)]
        right = deepcopy(left)
        tracemalloc.start()
        try:
            with patch('push_align.final_driver_v2.json.dumps', side_effect=AssertionError('giant JSON forbidden')):
                result = exact_comparison(left, right)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertEqual(result['status'], 'PASS_EXACT')
        self.assertLess(peak, 1024**2, f'auxiliary Python allocations grew to {peak} bytes')

    def test_direct_equality_rejects_difference_even_if_hashes_collide(self):
        with patch('push_align.final_driver_v2.typed_digest', return_value='a'*64):
            self.assertEqual(exact_comparison(np.array([0.]), np.array([-0.]))['status'], 'FAIL_EXACT')
            self.assertEqual(exact_comparison({'value': True}, {'value': 1})['status'], 'FAIL_EXACT')
            self.assertEqual(exact_comparison((1, 2), [1, 2])['status'], 'FAIL_EXACT')

    def test_diagnostic_json_conversion_is_local_and_preserves_float_payloads(self):
        env = SyntheticEnv(state_at_goal())
        left = env.step(np.zeros(2, np.float64))
        right = replace(left, diagnostics=deepcopy(left.diagnostics))
        right.diagnostics['state_fields'] = list(right.diagnostics['state_fields'])
        self.assertEqual(exact_comparison(left, right)['status'], 'PASS_EXACT')
        right.diagnostics['signedzero'] = -0.
        self.assertEqual(exact_comparison(left, right)['status'], 'FAIL_EXACT')
        left.diagnostics['signedzero'] = 0.
        self.assertEqual(exact_comparison(left, right)['status'], 'FAIL_EXACT')
        left.diagnostics['signedzero'] = -0.
        self.assertEqual(exact_comparison(left, right)['status'], 'PASS_EXACT')

    def test_changed_full_trace_receipt_is_rejected(self):
        result = self.run_cell()
        receipt = result.manifest['full_traces'][0]
        path = Path(next(iter(receipt['files_sha256'])))
        with path.open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'Missing or changed full trace'):
            read_full_trace(receipt)

    def test_unsupported_codec_and_missing_identity_rejected_before_execution(self):
        with self.assertRaises(TypeError):
            run_recorded_episode(None, None, folder=self.root/'never', identity=self.identity,
                source_bindings=self.bindings, profile=self.profile, guard=self.guard(), raw_codec=object())
        with self.assertRaises(ValueError):
            run_recorded_episode(None, None, folder=self.root/'never', identity={},
                source_bindings=self.bindings, profile=self.profile, guard=self.guard())

    def test_percell_source_check_is_not_repeated_by_action_guards(self):
        with patch('push_align.final_driver_v2.bindings_match', wraps=final_driver_v2.bindings_match) as check:
            result = self.run_cell(max_intervals=3)
        self.assertEqual(result.status, 'COMPLETE_RECORDED')
        self.assertEqual(check.call_count, 2)

    def test_missing_actual_required_input_is_incomplete_and_not_substituted(self):
        def missing(*args, **kwargs):
            output = execute_episode(*args, **kwargs)
            output.captured_inputs.pop(0)
            return output
        with patch('push_align.final_driver_v2.execute_episode', side_effect=missing):
            result = self.run_cell()
        self.assertEqual(result.status, 'STOP_REVIEW')
        self.assertEqual(result.manifest['missing_required_input_ticks'], [0])
        self.assertEqual(result.manifest['full_traces'], [])
        self.assertIsNotNone(result.manifest['raw'])

    def test_missing_actual_boundary_is_not_a_complete_recorded_original(self):
        def missing(*args, **kwargs):
            output = execute_episode(*args, **kwargs)
            output.records.pop()
            return output
        with patch('push_align.final_driver_v2.execute_episode', side_effect=missing):
            result = self.run_cell()
        self.assertEqual(result.status, 'STOP_REVIEW')
        self.assertTrue(result.manifest['witness_validation_issues'])
        self.assertEqual(result.manifest['raw']['interval_count'], 2)

    def test_guard_cell_history_does_not_accumulate_prior_episodes(self):
        guard = self.guard()
        first = self.run_cell(guard=guard)
        second = self.run_cell('second', guard=guard)
        self.assertEqual(len(first.manifest['resources']['guard_checks']),
                         len(second.manifest['resources']['guard_checks']))

    def test_failed_cell_latches_stop_until_caller_reviews_and_reestimates(self):
        guard = self.guard()
        env = SyntheticEnv(state_at_goal(), raise_at=1)
        result = self.run_cell(env=env, guard=guard)
        self.assertEqual(result.status, 'STOP_REVIEW')
        self.assertTrue(guard.stopped)
        other = SyntheticEnv(state_at_goal())
        with self.assertRaisesRegex(DriverStop, 'prior_cell_requires_review'):
            self.run_cell('another', env=other, guard=guard)
        self.assertEqual(other.calls, 0)

    def test_source_changed_during_execution_saves_full_prefix_and_stops(self):
        def changed(*args, **kwargs):
            output = execute_episode(*args, **kwargs)
            self.source.write_text('synthetic concurrent source change')
            return output
        with patch('push_align.final_driver_v2.execute_episode', side_effect=changed):
            result = self.run_cell()
        self.assertEqual(result.status, 'STOP_REVIEW')
        self.assertFalse(result.manifest['source_binding_check_after_execution'])
        self.assertIsNotNone(result.manifest['raw'])

    def test_invalid_executor_invocation_releases_cell_and_latches_stop(self):
        guard = self.guard()
        with self.assertRaises(ValueError):
            self.run_cell(guard=guard, max_intervals=0)
        self.assertFalse(guard.active)
        self.assertTrue(guard.stopped)

    def test_oversize_output_is_preserved_as_stop_with_no_next_cell_authority(self):
        result = self.run_cell(guard=self.guard(cell_write_allowance_bytes=1))
        self.assertEqual(result.status, 'STOP_INCOMPLETE_RECORDING')
        self.assertTrue(list((self.root/'original').glob('raw_execution*')))
        self.assertIn('measured_unit_exceeds_allowance_reestimate',
            result.manifest['recording_error']['message'])


class SealingReservationV2Test(unittest.TestCase):
    """Synthetic pressure tests; no MuJoCo construction or model loading."""

    setUp = FinalDriverV2Test.setUp
    guard = FinalDriverV2Test.guard
    run_cell = FinalDriverV2Test.run_cell

    def enter_sealing(self, guard, folder=None):
        folder = folder or self.root/'guard-cell'
        guard.begin_cell(self.bindings, folder=folder)
        folder.mkdir()
        token = guard._start_control_execution()
        execution = execute_episode(SyntheticEnv(state_at_goal()),
            SyntheticEnv(state_at_goal()).snapshot(), case_id='synthetic', episode_index=0,
            controller_id='R', execution_id='synthetic', predictor=stationary_predictor,
            physical_config=self.config, planner_config=self.planner,
            geometry_parameters=self.geometry, transition_parameters=self.transition,
            candidate_seed_base=4711, max_intervals=1, run_scope='development_preflight',
            penetration_budget_m=.0005)
        guard._control_returned(execution, token)
        return folder, execution

    def compare_science(self, first, second):
        self.assertEqual(exact_comparison(first.execution.results,
            second.execution.results)['status'], 'PASS_EXACT')
        self.assertEqual(exact_comparison(first.execution.decisions,
            second.execution.decisions)['status'], 'PASS_EXACT')
        self.assertEqual(exact_comparison(first.execution.plan_results,
            second.execution.plan_results)['status'], 'PASS_EXACT')
        self.assertEqual(exact_comparison(read_raw(first.manifest['raw']['prefix']),
            read_raw(second.manifest['raw']['prefix']))['status'], 'PASS_EXACT')
        self.assertEqual(exact_comparison(read_plan_index(first.manifest['plan_index']['path']),
            read_plan_index(second.manifest['plan_index']['path']))['status'], 'PASS_EXACT')
        self.assertEqual(first.execution.summary(), second.execution.summary())
        self.assertEqual(first.execution.termination, 'timeout')
        self.assertEqual(first.execution.execution_status, 'completed')
        self.assertEqual(first.status, 'COMPLETE_RECORDED')
        self.assertEqual(second.status, 'COMPLETE_RECORDED')
        self.assertNotEqual(first.execution.final_task.phase, 'ready')
        self.assertNotEqual(second.execution.final_task.phase, 'ready')
        for a, b in zip(first.manifest['full_traces'], second.manifest['full_traces']):
            left, arrays_left, _ = read_full_trace(a)
            right, arrays_right, _ = read_full_trace(b)
            self.assertEqual(exact_comparison(left, right)['status'], 'PASS_EXACT')
            self.assertEqual(exact_comparison(arrays_left, arrays_right)['status'], 'PASS_EXACT')

    @patch('push_align.planning.perf_counter', return_value=1.)
    def pressure_pair(self, _clock, *, after_raw):
        self.config['task']['max_episode_time_s'] = .2
        first = self.run_cell('abundant', execution_id='identical-science')
        folder = self.root/'pressured'
        calls = [0]
        def free():
            calls[0] += 1
            ready = bool(list(folder.glob('raw_execution*'))) if after_raw else calls[0] > 3
            return int(2.5*GIB) if ready else 10*GIB
        guard = self.guard(free_bytes=free)
        second = self.run_cell('pressured', guard=guard, execution_id='identical-science')
        self.compare_science(first, second)
        self.assertTrue(guard.requires_reentry)
        self.assertFalse(guard.stopped)
        self.assertEqual(guard.phase, 'CLOSED')
        events = second.manifest['resources']['guard_checks']
        self.assertTrue(any(row['sealing_reservation_only'] and not row['full_future_budget_fits']
                            for row in events))
        self.assertTrue(all(row['timestamp_utc'] for row in events))
        self.assertEqual([row['phase'] for row in guard.phase_history],
                         ['CONTROL', 'SEALING', 'CLOSED'])
        with self.assertRaises(DriverStop):
            guard.before_stage()
        with self.assertRaises(DriverStop):
            self.run_cell('next-cell', guard=guard)
        self.assertFalse((self.root/'next-cell').exists())
        return first, second

    def test_normal_timeout_full_budget_loss_before_first_raw_still_seals_same_science(self):
        self.pressure_pair(after_raw=False)

    def test_observed_failure_after_raw_now_saves_index_traces_witness_and_manifest(self):
        self.pressure_pair(after_raw=True)

    def test_sealing_is_irreversible_and_rejects_step_restore_reset_even_after_capacity_recovers(self):
        guard = self.guard()
        _, execution = self.enter_sealing(guard)
        wrapped = final_driver_v2._GuardedEnvironment(SyntheticEnv(state_at_goal()), guard)
        for call in (lambda: guard.before_action(), lambda: wrapped.step(np.zeros(2)),
                     lambda: wrapped.restore(execution.initial_snapshot), lambda: wrapped.reset()):
            with self.assertRaises(DriverStop):
                call()
        guard.finish_cell()
        with self.assertRaises(DriverStop):
            guard.before_action()
        with self.assertRaises(RuntimeError):
            guard.finish_cell()

    def test_no_returned_executor_token_cannot_open_sealing_or_write(self):
        guard = self.guard()
        with self.assertRaises(DriverStop):
            guard._control_returned(None, None)
        folder = self.root/'guard-cell'
        guard.begin_cell(self.bindings, folder=folder)
        with self.assertRaises(RuntimeError):
            guard.before_write(preserving=True)
        with self.assertRaises(RuntimeError):
            guard._control_returned(None, object())
        self.assertFalse(folder.exists())

    def test_sealing_reserve_equality_passes_one_byte_less_fails(self):
        free = [10*GIB]
        guard = self.guard(free_bytes=lambda: free[0])
        self.enter_sealing(guard)
        free[0] = guard.hard_reserve_bytes + guard.cell_write_allowance_bytes
        self.assertEqual(guard.before_write()['status'], 'CONTINUE_RESOURCE_CHECK_ONLY')
        free[0] -= 1
        with self.assertRaises(DriverStop) as caught:
            guard.before_write(preserving=True)
        self.assertIn('next_write_crosses_hard_reserve', caught.exception.decision['reasons'])

    def test_written_credit_is_owned_monotonic_and_allowance_not_enlarged(self):
        guard = self.guard(cell_write_allowance_bytes=16)
        folder, _ = self.enter_sealing(guard)
        (folder/'raw').write_bytes(b'x'*16)
        self.assertEqual(guard.observe(folder), 16)
        guard.before_write()
        (folder/'raw').write_bytes(b'x'*17)
        with self.assertRaises(DriverStop) as caught:
            guard.observe(folder)
        self.assertIn('measured_unit_exceeds_allowance_reestimate', caught.exception.decision['reasons'])
        self.assertEqual(guard.cell_write_allowance_bytes, 16)
        with self.assertRaises(ValueError):
            guard.observe(self.root)
        (folder/'raw').write_bytes(b'x')
        with self.assertRaisesRegex(ValueError, 'decreased'):
            guard.observe(folder)

    def test_hardlinks_and_symlinks_never_supply_write_credit(self):
        for name, link in [('symlink', lambda a, b: b.symlink_to(a)),
                           ('hardlink', lambda a, b: os.link(a, b))]:
            with self.subTest(name=name):
                guard = self.guard()
                folder, _ = self.enter_sealing(guard, self.root/name)
                link(self.source, folder/'linked')
                with self.assertRaises(ValueError):
                    guard.observe(folder)

    def test_sealing_keeps_mismatch_limit_and_charges_exactly_once(self):
        allowance = 1024
        guard = self.guard(cell_write_allowance_bytes=allowance,
                           mismatch_used_bytes=255*1024**2,
                           mismatch_limit_bytes=255*1024**2+allowance)
        folder = self.root/'mismatch'
        guard.begin_cell(self.bindings, fresh=True, folder=folder)
        folder.mkdir()
        token = guard._start_control_execution()
        execution = execute_episode(SyntheticEnv(state_at_goal()),
            SyntheticEnv(state_at_goal()).snapshot(), case_id='synthetic', episode_index=0,
            controller_id='R', execution_id='synthetic', predictor=stationary_predictor,
            physical_config=self.config, planner_config=self.planner,
            geometry_parameters=self.geometry, transition_parameters=self.transition,
            candidate_seed_base=4711, max_intervals=1, run_scope='development_preflight',
            penetration_budget_m=.0005)
        guard._control_returned(execution, token)
        guard.before_write(preserving=True)
        guard.mismatch_limit_bytes -= 1
        with self.assertRaises(DriverStop) as caught:
            guard.before_write(preserving=True)
        self.assertIn('mismatch_allowance_exceeded', caught.exception.decision['reasons'])
        guard.mismatch_limit_bytes += 1
        (folder/'retained').write_bytes(b'raw')
        guard.observe(folder, mismatch=True)
        before = guard.mismatch_used_bytes
        guard.finish_cell(mismatch=True, stop=True)
        self.assertEqual(guard.mismatch_used_bytes, before+3)
        with self.assertRaises(RuntimeError):
            guard.finish_cell(mismatch=True)
        self.assertEqual(guard.mismatch_used_bytes, before+3)

    def test_failed_writer_receipt_preserves_first_resource_reason_and_timestamps(self):
        def fail(prefix, *args, **kwargs):
            Path(str(prefix)+'.raw.npz').write_bytes(b'partial exact prefix')
            raise OSError('synthetic codec failure')
        with patch('push_align.final_recording.write_raw', side_effect=fail):
            result = self.run_cell()
        saved = json.loads(result.manifest_path.read_text())
        self.assertEqual(saved['recording_error']['type'], 'OSError')
        self.assertIn('synthetic codec failure', saved['recording_error']['message'])
        self.assertTrue(saved['recording_error']['timestamp_utc'])
        self.assertEqual(saved['recording_error']['cell_phase'], 'SEALING')
        self.assertTrue(saved['recording_error']['guard_history'])
        self.assertTrue(saved['failure_receipt_resources']['guard_last_decision'])
        self.assertTrue(saved['failure_manifest_persisted'])
        self.assertTrue(saved['partial_files_sha256'])

    def test_reserve_failure_retains_inmemory_guard_reason_without_claiming_disk_receipt(self):
        folder = self.root/'original'
        def free():
            return 2*GIB if list(folder.glob('raw_execution*')) else 10*GIB
        result = self.run_cell(guard=self.guard(free_bytes=free))
        self.assertEqual(result.status, 'STOP_INCOMPLETE_RECORDING')
        self.assertIsNone(result.manifest_path)
        self.assertFalse(result.manifest['failure_manifest_persisted'])
        error = result.manifest['recording_error']
        self.assertEqual(error['type'], 'DriverStop')
        self.assertIn('next_write_crosses_hard_reserve', error['guard_last_decision']['reasons'])
        self.assertEqual(result.manifest['failure_receipt_error']['type'], 'DriverStop')
        self.assertTrue(list(folder.glob('raw_execution*')))

    def test_source_or_missing_input_under_capacity_loss_remains_stop_review(self):
        for kind in ('source', 'input'):
            with self.subTest(kind=kind):
                self.source.write_text('synthetic immutable source')
                folder = self.root/kind
                def altered(*args, **kwargs):
                    output = execute_episode(*args, **kwargs)
                    if kind == 'source':
                        self.source.write_text('changed after actual synthetic control')
                    else:
                        output.captured_inputs.pop(0)
                    return output
                def free():
                    return int(2.5*GIB) if list(folder.glob('raw_execution*')) else 10*GIB
                guard = self.guard(free_bytes=free)
                with patch('push_align.final_driver_v2.execute_episode', side_effect=altered):
                    result = self.run_cell(kind, guard=guard)
                self.assertEqual(result.status, 'STOP_REVIEW')
                self.assertIsNotNone(result.manifest['raw'])
                self.assertIsNotNone(result.manifest['plan_index'])
                self.assertTrue(guard.stopped)
                self.assertTrue(guard.requires_reentry)
                if kind == 'source':
                    self.assertFalse(result.manifest['source_binding_check_after_execution'])
                else:
                    self.assertEqual(result.manifest['missing_required_input_ticks'], [0])

    def test_fresh_plan_mismatch_remains_review_under_sealing_capacity_loss(self):
        original = self.run_cell()
        folder = self.root/'fresh-pressure'
        def altered(*args, **kwargs):
            output = execute_episode(*args, **kwargs)
            next(iter(output.plan_results[1].costs.components.values()))[0] += .125
            return output
        def free():
            return int(2.5*GIB) if list(folder.glob('raw_execution*')) else 10*GIB
        guard = self.guard(free_bytes=free)
        with patch('push_align.final_driver_v2.execute_episode', side_effect=altered):
            result = self.run_cell('fresh-pressure', guard=guard, reference=original.manifest_path)
        self.assertEqual(result.status, 'STOP_REVIEW')
        self.assertEqual(result.comparison['raw_exact']['status'], 'PASS_EXACT')
        self.assertEqual(result.comparison['plan_index_exact']['status'], 'FAIL_EXACT')
        self.assertIn(1, result.comparison['differing_ticks'])
        self.assertGreater(guard.mismatch_used_bytes, 0)
        self.assertTrue(guard.requires_reentry)
        self.assertTrue(guard.stopped)

    def test_fresh_guard_cannot_turn_sealed_prefix_into_new_full_budget_authority(self):
        self.pressure_pair(after_raw=True)
        env = SyntheticEnv(state_at_goal())
        guard = self.guard(free_bytes=lambda: int(2.5*GIB))
        with self.assertRaises(DriverStop):
            self.run_cell('new-process', env=env, guard=guard)
        self.assertEqual(env.calls, 0)
        self.assertFalse((self.root/'new-process').exists())

    def test_closed_guard_has_no_prior_written_credit_for_next_stage(self):
        free = [10*GIB]
        guard = self.guard(free_bytes=lambda: free[0])
        folder, _ = self.enter_sealing(guard)
        (folder/'retained').write_bytes(b'x'*100)
        guard.observe(folder)
        guard.finish_cell()
        free[0] = 3*GIB-1
        with self.assertRaises(DriverStop) as caught:
            guard.before_stage()
        self.assertEqual(caught.exception.decision['full_future_required_bytes'], 3*GIB)
        free[0] = 10*GIB
        guard.before_stage()
        guard.begin_cell(self.bindings, folder=self.root/'reentered')
        self.assertEqual(guard.phase, 'CONTROL')
        self.assertFalse(guard.requires_reentry)
        self.assertEqual(guard.written_bytes, 0)

    def test_readwrite_science_helpers_are_exact_copies_of_frozen_driver(self):
        old = (ROOT/'src/push_align/final_driver.py').read_text()
        new = (ROOT/'src/push_align/final_driver_v2.py').read_text()
        self.assertEqual(old[old.index('def _bytes_view'):old.index('def run_recorded_episode')],
                         new[new.index('def _bytes_view'):new.index('def run_recorded_episode')])
        science_names = {'execute_episode', '_capture_terminal_input', 'plan_index_records',
            'write_plan_index', 'write_full_trace', '_full_input', 'build_witness',
            'compare_witness', '_validate', 'write_witness_gzip'}
        def science_calls(source):
            function = next(node for node in ast.parse(source).body
                            if isinstance(node, ast.FunctionDef) and node.name == 'run_recorded_episode')
            return [ast.dump(node, include_attributes=False) for node in ast.walk(function)
                    if isinstance(node, ast.Call) and
                    ((isinstance(node.func, ast.Name) and node.func.id in science_names)
                     or (isinstance(node.func, ast.Attribute) and node.func.attr in {'write_raw', 'read_raw'}))]
        self.assertEqual(science_calls(old), science_calls(new))


if __name__ == '__main__':
    unittest.main()
