"""Pure synthetic E1 orchestration tests; real candidate physics is forbidden."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import tomllib
import unittest
from unittest.mock import patch

import numpy as np

import p5_final_e1 as e1
from push_align.candidate_replay import PhysicalCandidateReplay
from push_align.config import ROOT, load_pilot_config
from push_align.environment import Observation, SnapshotBundle, StepResult
from push_align.final_driver import typed_digest, write_full_trace
from push_align.final_manifest import expected_E1
from push_align.final_recording import file_sha256, write_witness_gzip
from push_align.geometry_control import GeometryContext
from push_align.planning import CostBatch, PlannerContext, PredictionBatch, TaskSpec
from push_align.planning_protocol import GeometryProposalAdapter
from push_align.reproduction import json_safe
from push_align.task import start_task


class FakeModel:
    def __init__(self, value=0):
        self.value = value

    def state_dict(self):
        return {'synthetic_weight': self.value}


class FakePredictor:
    calls_total = 0

    def __init__(self, model, normalizer):
        self.model, self.calls = model, 0

    def __call__(self, state, action):
        self.calls += 1
        FakePredictor.calls_total += 1
        return np.full_like(state, np.nan) if self.model.value == -1 else np.zeros_like(state)


class FinalE1Test(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT/'tmp'
        temporary_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='p5-e1-unit-', dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name).resolve()
        self.run = self.work/'run'
        self.run.mkdir()
        self.source = self.work/'frozen_synthetic_source.txt'
        self.runtime_source = self.work/'synthetic_runtime_source.txt'
        self.source.write_text('synthetic immutable original')
        self.runtime_source.write_text('synthetic runtime code')
        self.bindings = {str(path): file_sha256(path) for path in (self.source, self.runtime_source)}
        self.physical = load_pilot_config()
        self.planner = deepcopy(self.physical)
        self.planner['planner'].update(candidate_count=128, horizon=5, block_lengths=[2, 3])
        self.profile = tomllib.loads((ROOT/'configs/acceptance.v1.toml').read_text())
        self.geometry = json.loads((ROOT/'configs/p3_geometry_selected.v2.json').read_text())
        self.transition = json.loads((ROOT/'configs/p3_geometry_transition.v2.json').read_text())
        families = []
        for stratum in ('translation', 'rotation', 'offset'):
            for index in range(4):
                families.append({'family_id': f'synthetic-{stratum}-{index}',
                    'episode_index': len(families), 'stratum': stratum})
        self.manifest = {'schema': 'p5-frozen-final-manifest-v1', 'status': 'FROZEN_BEFORE_FINAL_OUTCOMES',
            'families': families, 'E1': expected_E1(families),
            'physical_config': self.physical, 'planner_config': self.planner, 'profile': self.profile,
            'model_identities': {name: {'path': f'synthetic/{name}', 'model_hash': str(index)*64}
                for index, name in enumerate(e1.MODELS)},
            'normalizer_identity': {'synthetic': True},
            'source_sha256': {str(self.source.relative_to(ROOT.resolve())): self.bindings[str(self.source)]},
            'runtime_source_sha256': {str(self.runtime_source.relative_to(ROOT.resolve())): self.bindings[str(self.runtime_source)]}}
        self.runtime = {'models': {name: SimpleNamespace(model=FakeModel()) for name in e1.MODELS},
            'normalizer': object(), 'geometry': self.geometry, 'transition': self.transition}
        self.events = []
        self.replays = 0
        FakePredictor.calls_total = 0
        self.make_G_inputs()

    def make_G_inputs(self, *, omit_snapshot=None):
        grouped = {}
        for spec in self.manifest['E1']:
            grouped.setdefault(spec['family_id'], []).append(spec)
        adapter = GeometryProposalAdapter(self.physical, self.geometry, self.transition)
        for family, specs in grouped.items():
            folder = self.run/'E2'/family/'G'
            folder.mkdir(parents=True, exist_ok=True)
            receipts, decisions, files = [], [], {}
            for spec in specs:
                if spec['snapshot_id'] == omit_snapshot:
                    continue
                state = np.zeros(10, np.float64)
                state[6] = -.051
                observed = Observation(state, 2.+spec['tick']*.1, 'push-align-state-v1', True)
                task = TaskSpec.from_config(self.physical)
                status = replace(start_task(observed, task, .1), initial_sim_time=2.)
                snapshot = SnapshotBundle(np.r_[observed.sim_time, state], 'synthetic-model',
                    'synthetic-environment', .01, {'state': 17}, np.zeros(1, np.int64), np.zeros(1, np.int64), False)
                context = PlannerContext(np.array([.003, -.002], np.float64),
                    np.full((5, 2), .007, np.float64), spec['tick'])
                gctx = GeometryContext()
                proposal = adapter.propose(state.copy(), task, gctx, status.elapsed_s, control_tick=spec['tick'])
                capture = {'tick': spec['tick'], 'snapshot': snapshot, 'observation': observed,
                    'initial_observation': observed, 'task_status': status, 'task_spec': task,
                    'planner_context': context, 'geometry_context': gctx, 'episode_index': spec['episode_index'],
                    'controller_id': 'G', 'case_id': family, 'identity': {'model': None, 'normalizer': None},
                    'physical_config': self.physical, 'planner_config': self.planner,
                    'source_bindings': self.bindings}
                receipt = write_full_trace(folder/'plans'/f"step-{spec['tick']:03d}", tick=spec['tick'],
                                           captured_input=capture)
                receipts.append(receipt)
                files.update(receipt['files_sha256'])
                decisions.append({'index': spec['tick'], 'state': state.tolist(),
                                  'geometry_proposal': json_safe(proposal.diagnostics)})
            witness_path = folder/'witness.json.gz'
            witness_receipt = write_witness_gzip(witness_path, {'intervals': [], 'decisions': decisions})
            files.update(witness_receipt['files_sha256'])
            saved = {'schema': 'p5-recorded-episode-v1', 'record_kind': 'reference_controller_execution',
                'recording_status': 'COMPLETE_RECORDED', 'execution': {'controller_id': 'G', 'run_scope': 'final_evaluation'},
                'full_traces': receipts, 'files_sha256': files, 'witness_path': str(witness_path)}
            (folder/'manifest.json').write_text(json.dumps(saved))

    def callback(self, event):
        self.events.append(event)
        return None

    def sealed_failed_G(self, *, omit_tick=None, execution_status='completed'):
        """Synthetic seals only; saved-raw deep review is mocked by each test."""
        spec = self.manifest['E1'][0]
        cell = {'family_id': spec['family_id'], 'controller_id': 'G',
                'episode_index': spec['episode_index'], 'cell_id': spec['family_id']+'/G'}
        self.manifest['E2'] = [cell]
        folder = self.run/'E2'/spec['family_id']/'G'
        path = folder/'manifest.json'
        saved = json.loads(path.read_text())
        saved['recording_status'] = 'STOP_REVIEW'
        saved['identity'] = {'family_id': spec['family_id'], 'controller_id': 'G',
                             'model': None, 'normalizer': None}
        saved['source_bindings'] = self.bindings
        saved['execution'].update(execution_status=execution_status, termination='physics_error',
            counts={'executed_intervals': 31}, future_engine_calls={'mj_step': 0, 'mj_forward': 0})
        if omit_tick is not None:
            saved['full_traces'] = [row for row in saved['full_traces'] if row['tick'] != omit_tick]
        path.write_text(json.dumps(saved))
        metrics = {'success': False, 'termination': 'physics_error'}
        review = {'schema': 'p5-independent-recording-review-v1',
            'status': 'ACCEPT_COMPLETED_TASK_FAILURE', 'accepted': True, 'metrics': metrics,
            'manifest_sha256': file_sha256(path)}
        (folder/'review.json').write_text(json.dumps(review))
        worker = {'phase': 'E2', 'cell': cell, 'status': review['status'],
            'recording_status': 'STOP_REVIEW', 'record_manifest': str(path),
            'record_manifest_sha256': file_sha256(path),
            'review_sha256': file_sha256(folder/'review.json'),
            'row': {**cell, **saved['execution'], **metrics}}
        (folder/'worker_result.json').write_text(json.dumps(worker))
        (folder/'worker_result_receipt.json').write_text(json.dumps({
            'sha256': file_sha256(folder/'worker_result.json')}))
        return folder, review

    def fake_replay(self, physical, planner, snapshot, actions, task, previous, profile):
        # Actual futures may not start until every model choice is on disk.
        choices = json.loads((self.run/'E1/choices_before_any_physical_future.json').read_text())
        self.assertEqual(len(choices), 84)
        self.assertTrue(all(row['before_any_physical_future'] for row in choices))
        self.assertTrue(all(row['warm_sequence'] is None for row in choices))
        self.assertTrue(all(row['candidate_rng_keys'][0] == 6811 and row['candidate_rng_keys'][-1] == 1 for row in choices))
        self.assertEqual(FakePredictor.calls_total, 420)
        self.assertEqual(actions.shape, (5, 2))
        self.assertEqual(previous.tolist(), [.003, -.002])
        self.assertEqual(planner['planner']['candidate_count'], 16)
        self.replays += 1
        initial = snapshot.integration_state[1:].copy()
        states = np.repeat(initial[None], 6, axis=0)
        results = []
        for index, action in enumerate(actions):
            time = float(snapshot.integration_state[0]+.1*(index+1))
            results.append(StepResult(action.copy(), action.copy(),
                Observation(states[index+1].copy(), time, 'push-align-state-v1', True), .1, True,
                {'samples': [{'state': states[index].tolist(), 'sim_time': time-.1, 'contacts': []},
                             {'state': states[index+1].tolist(), 'sim_time': time, 'contacts': []}],
                 'validity_violations': []}, states[index+1:index+2].copy(), np.array([time], np.float64)))
        cost = float(1.+np.abs(actions).sum())
        return PhysicalCandidateReplay(states, actions.copy(), True, cost,
            {name: 0. for name in e1.COMPONENT_NAMES}, 5, results, True, snapshot, ())

    def patches(self, *, replay=None):
        import contextlib
        stack = contextlib.ExitStack()
        stack.enter_context(patch('p5_final_e1.gc.collect', return_value=0))
        stack.enter_context(patch('p5_final_e1._load_runtime', return_value=self.runtime))
        stack.enter_context(patch('p5_final_e1.LearnedPredictor', FakePredictor))
        stack.enter_context(patch('p5_final_e1.replay_candidate', side_effect=replay or self.fake_replay))
        return stack

    def test_complete_scope_freezes84_choices_then576_shared_repeats_with_all_denominators(self):
        with self.patches():
            result = e1.run_e1(self.run, self.manifest, self.callback)
        self.assertEqual(result['status'], 'COMPLETE_RECORDED', result.get('error'))
        self.assertEqual(self.replays, 576)
        self.assertEqual(result['actual_future_intervals'], 2880)
        self.assertEqual(result['diagnostic_count'], 84)
        self.assertEqual(len(result['snapshots']), 12)
        self.assertTrue(all(row['physical_replays'] == 48 for row in result['snapshots']))
        self.assertTrue(all(row['all7_common_valid'] == 16 for row in result['snapshots']))
        self.assertEqual(result['conditional_selection_loss_denominator'], 84)
        self.assertEqual(sum(result['future_engine_calls_during_model_prediction'].values()), 0)
        self.assertTrue(result['all_sources_and_model_weights_unchanged'])
        ledger = json.loads((self.run/'E1/comparison_ledger.json').read_text())
        self.assertEqual(len(ledger), 84)
        self.assertTrue(all(row['counts']['candidate_count'] == 16 for row in ledger))
        self.assertTrue(all(row['pairwise']['total_pairs'] == 120 for row in ledger))
        self.assertTrue(all(row['all7_common_valid_cost_error']['denominator'] == 16 for row in ledger))
        events = [row['event'] for row in self.events]
        self.assertEqual(events.count('repeat_complete'), 576)
        self.assertLess(events.index('predictions_frozen'), events.index('repeat_started'))
        self.assertTrue(all(row['remove_reserved_intervals'] == 5 for row in self.events if row['event'] == 'repeat_complete'))
        for path in (self.run/'E1').rglob('actual_futures.npz'):
            with np.load(path, allow_pickle=False) as arrays:
                self.assertEqual(arrays['states'].shape, (16, 3, 6, 10))
                self.assertEqual(arrays['total_cost'].shape, (16, 3))
        self.assertEqual(len(list((self.run/'E1').rglob('*.raw.columns.xz'))), 576)
        for path in (self.run/'E1').rglob('candidates.npz'):
            with np.load(path, allow_pickle=False) as arrays:
                # The source deliberately had a nonzero warm sequence. E1
                # must reset warm for every model while keeping actual prev.
                self.assertTrue(np.array_equal(arrays['actions'][1], np.zeros((5, 2))))

    def test_missing_fixed_tick_holds_before_any_model_or_future_without_substitution(self):
        missing = self.manifest['E1'][1]
        manifest_path = self.run/'E2'/missing['family_id']/'G/manifest.json'
        saved = json.loads(manifest_path.read_text())
        saved['full_traces'] = [row for row in saved['full_traces'] if row['tick'] != 29]
        manifest_path.write_text(json.dumps(saved))
        with self.patches() as stack:
            with patch('p5_final_e1._load_runtime') as loader:
                result = e1.run_e1(self.run, self.manifest, self.callback)
                loader.assert_not_called()
        self.assertEqual(result['status'], 'HOLD_MISSING_FIXED_G_INPUT')
        self.assertEqual(result['available_snapshot_count'], 11)
        self.assertEqual(result['missing'][0]['snapshot_id'], missing['snapshot_id'])
        self.assertFalse(result['substitution_permitted'])
        self.assertEqual(self.replays, 0)

    def test_sealed_completed_physical_G_failure_keeps_its_valid_predecision_inputs(self):
        folder, review = self.sealed_failed_G()
        with patch('push_align.final_result_review.review_recording', return_value=review) as independent:
            inputs, missing = e1._load_inputs(self.run, self.manifest['E1'], self.manifest, self.bindings)
        self.assertFalse(missing)
        self.assertEqual(len(inputs), 12)
        independent.assert_called_once()
        self.assertEqual(independent.call_args.kwargs['expected_run_scope'], 'final_evaluation')
        self.assertEqual(json.loads((folder/'manifest.json').read_text())['recording_status'], 'STOP_REVIEW')
        self.assertEqual(self.replays, 0)
        self.assertEqual(FakePredictor.calls_total, 0)

    def test_completed_G_failure_never_substitutes_a_missing_fixed_tick(self):
        _, review = self.sealed_failed_G(omit_tick=29)
        with (patch('push_align.final_result_review.review_recording', return_value=review),
              patch('p5_final_e1._load_runtime') as loader):
            result = e1.run_e1(self.run, self.manifest, self.callback)
        self.assertEqual(result['status'], 'HOLD_MISSING_FIXED_G_INPUT')
        self.assertEqual(result['available_snapshot_count'], 11)
        self.assertFalse(result['substitution_permitted'])
        loader.assert_not_called()
        self.assertEqual(self.replays, 0)

    def test_G_failure_missing_or_changed_seals_and_incomplete_infrastructure_are_rejected(self):
        for fault in ('no_worker_receipt', 'changed_worker', 'changed_review', 'incomplete', 'oracle'):
            folder, review = self.sealed_failed_G(execution_status='error' if fault == 'incomplete' else 'completed')
            if fault == 'no_worker_receipt':
                (folder/'worker_result_receipt.json').unlink()
            elif fault == 'changed_worker':
                with (folder/'worker_result.json').open('a') as stream:
                    stream.write(' ')
            elif fault == 'changed_review':
                with (folder/'review.json').open('a') as stream:
                    stream.write(' ')
            elif fault == 'oracle':
                worker_path = folder/'worker_result.json'
                worker = json.loads(worker_path.read_text())
                worker['row']['future_engine_calls']['mj_step'] = 1
                worker_path.write_text(json.dumps(worker))
                (folder/'worker_result_receipt.json').write_text(json.dumps({'sha256': file_sha256(worker_path)}))
            with self.subTest(fault=fault), patch('push_align.final_result_review.review_recording', return_value=review) as independent:
                with self.assertRaises(e1.E1Stop):
                    e1._load_inputs(self.run, self.manifest['E1'], self.manifest, self.bindings)
                independent.assert_not_called()

    def test_G_failure_sealed_acceptance_alone_does_not_replace_fresh_deep_review(self):
        _, review = self.sealed_failed_G()
        for altered in ({**review, 'accepted': False}, {**review, 'metrics': {'success': True}},
                        {**review, 'status': 'ACCEPT_COMPLETED_RECORDING'}):
            with self.subTest(altered=altered), patch('push_align.final_result_review.review_recording', return_value=altered):
                with self.assertRaises(e1.E1Stop):
                    e1._load_inputs(self.run, self.manifest['E1'], self.manifest, self.bindings)

    def test_resource_stop_before_first_repeat_preserves_frozen_predictions_and_zero_physics(self):
        def stop(event):
            self.events.append(event)
            return {'status': 'STOP_PRESERVE_PREFIX'} if event['event'] == 'before_repeat' else None
        with self.patches():
            result = e1.run_e1(self.run, self.manifest, stop)
        self.assertEqual(result['status'], 'HOLD_INCOMPLETE_E1')
        self.assertEqual(self.replays, 0)
        self.assertEqual(len(json.loads((self.run/'E1/choices_before_any_physical_future.json').read_text())), 84)
        self.assertTrue(result['denominator_preserved'])
        self.assertEqual(result['completed_physical_repeats'], 0)

    def test_before_raw_guard_stop_preserves_reserved_prefix_only_after_second_admission(self):
        def stop(event):
            self.events.append(event)
            return False if event['event'] == 'before_raw_write' else None
        with self.patches():
            result = e1.run_e1(self.run, self.manifest, stop)
        self.assertEqual(result['status'], 'HOLD_INCOMPLETE_E1')
        self.assertEqual(self.replays, 1)
        self.assertEqual(result['completed_physical_repeats'], 0)
        self.assertEqual(result['actual_returned_intervals_including_failed_prefixes'], 5)
        errors = list((self.run/'E1').rglob('infrastructure_error.json'))
        self.assertEqual(len(errors), 1)
        row = json.loads(errors[0].read_text())
        from push_align import final_recording_columns as codec
        _, steps = codec.read_raw(errors[0].parent/'raw_execution',
            expected_sha256=row['raw_receipt']['files_sha256'])
        self.assertEqual(len(steps), 5)
        self.assertEqual(row['status'], 'INCOMPLETE_INFRASTRUCTURE')
        self.assertEqual(row['raw_prefix_preservation'], 'SAVED_LOSSLESS')
        self.assertTrue(result['all_returned_raw_prefixes_saved'])
        self.assertIn('preserve_raw_write', [event['event'] for event in self.events])
        self.assertFalse(list((self.run/'E1').rglob('stage_receipt.json')))

    def test_denied_preservation_admission_keeps_hard_guard_and_reports_unsaved_prefix(self):
        def stop(event):
            self.events.append(event)
            return False if event['event'] in {'before_raw_write', 'preserve_raw_write'} else None
        with self.patches():
            result = e1.run_e1(self.run, self.manifest, stop)
        self.assertEqual(result['status'], 'HOLD_INCOMPLETE_E1')
        self.assertEqual(self.replays, 1)
        self.assertEqual(result['completed_physical_repeats'], 0)
        errors = list((self.run/'E1').rglob('infrastructure_error.json'))
        self.assertEqual(len(errors), 1)
        row = json.loads(errors[0].read_text())
        self.assertEqual(row['actual_returned_intervals'], 5)
        self.assertEqual(row['raw_prefix_preservation'], 'INCOMPLETE_NOT_SAVED')
        self.assertFalse(row['raw_write_admitted'])
        self.assertIsNone(row['raw_receipt'])
        self.assertFalse(result['all_returned_raw_prefixes_saved'])
        self.assertEqual(result['unsaved_raw_prefix_repeat_ids'], [row['repeat_id']])
        self.assertTrue(row['error']['raw_preservation_not_admitted'])
        self.assertFalse(list((self.run/'E1').rglob('*.raw.columns.xz')))

    def test_infrastructure_failure_keeps_raw_zero_prefix_and_failure_denominator(self):
        with self.patches(replay=lambda *args: (_ for _ in ()).throw(RuntimeError('synthetic transport failure'))):
            result = e1.run_e1(self.run, self.manifest, self.callback)
        self.assertEqual(result['status'], 'HOLD_INCOMPLETE_E1')
        failures = list((self.run/'E1').rglob('infrastructure_error.json'))
        self.assertEqual(len(failures), 1)
        failure = json.loads(failures[0].read_text())
        self.assertEqual(failure['actual_returned_intervals'], 0)
        self.assertIsNotNone(failure['raw_receipt'])
        self.assertEqual(failure['raw_receipt']['interval_count'], 0)
        self.assertEqual(result['attempted_physical_repeats'], 1)
        self.assertEqual(result['completed_physical_repeats'], 0)
        self.assertFalse(result['automatic_retry'])
        self.assertEqual(len(list((self.run/'E1').rglob('actual_futures.incomplete.npz'))), 1)

    def test_returned_physical_prefix_survives_exception_after_first_synthetic_step(self):
        def synthetic_step(env, action):
            state = np.zeros(10, np.float64)
            return StepResult(action.copy(), action.copy(),
                Observation(state, 2.1, 'push-align-state-v1', True), .1, True,
                {'samples': [{'state': state.tolist(), 'sim_time': 2.1, 'contacts': []}],
                 'validity_violations': []}, state[None], np.array([2.1], np.float64))
        def failed_replay(*args):
            e1.PlanarPushEnv.step(object(), np.zeros(2, np.float64))
            raise RuntimeError('synthetic failure after an actual-return boundary')
        with self.patches(replay=failed_replay), patch('p5_final_e1.PlanarPushEnv.step', side_effect=synthetic_step):
            result = e1.run_e1(self.run, self.manifest, self.callback)
        self.assertEqual(result['status'], 'HOLD_INCOMPLETE_E1')
        failure_path = next((self.run/'E1').rglob('infrastructure_error.json'))
        failure = json.loads(failure_path.read_text())
        self.assertEqual(failure['actual_returned_intervals'], 1)
        self.assertEqual(failure['raw_receipt']['interval_count'], 1)
        self.assertEqual(result['actual_returned_intervals_including_failed_prefixes'], 1)
        _, prefix = e1.raw_codec.read_raw(failure_path.parent/'raw_execution',
                                       expected_sha256=failure['raw_receipt']['files_sha256'])
        self.assertEqual(len(prefix), 1)

    def test_all_invalid_and_changed_validity_keep_rank_loss_and_common_denominators(self):
        state = np.zeros((16, 6, 10), np.float64)
        entries = []
        for name in e1.MODELS:
            valid = np.zeros(16, bool) if name == 'S1-23' else np.ones(16, bool)
            costs = np.ones(16, np.float64)
            costs[~valid] = np.inf
            predicted = PredictionBatch(state.copy(), np.zeros((16, 5, 10), np.float64), valid,
                tuple('invalid' if not value else '' for value in valid), np.where(valid, -1, 1).astype(np.int64))
            entries.append((name, predicted, CostBatch(costs, {}),
                {'selected_index': None if name == 'S1-23' else 0, 'candidate_hash': 'a'*64}))
        arrays = e1._actual_arrays()
        arrays['states'][:] = 0.
        arrays['applied_actions'][:] = 0.
        arrays['total_cost'][:] = 1.
        arrays['valid'][:] = True
        arrays['valid'][0, 1] = False
        arrays['total_cost'][0, 1] = np.inf
        repeats = [{'candidate_index': index, 'repeat_index': repeat, 'final_snapshot_hash': 'same'}
                   for index in range(16) for repeat in range(3)]
        data = {'entries': entries, 'spec': self.manifest['E1'][0], 'input_hash': 'b'*64,
                'folder': self.work/'diagnostic-only'}
        rows, snapshot = e1._diagnostics(data, arrays, repeats, self.profile)
        self.assertEqual(snapshot['all7_common_valid'], 0)
        self.assertEqual(snapshot['validity_changed_candidates'], [0])
        invalid = next(row for row in rows if row['model'] == 'S1-23')
        self.assertTrue(invalid['model_all_invalid'])
        self.assertEqual(invalid['selection_loss_denominator'], 0)
        self.assertEqual(invalid['counts']['candidate_count'], 16)
        self.assertEqual(invalid['counts']['actual_validity_changed'], 1)
        for row in rows:
            self.assertEqual(row['all7_common_valid_cost_error']['denominator'], 0)
            self.assertEqual(row['pairwise']['total_pairs'], 120)
            self.assertIn(row['selection_loss_ordering'], (None, 'INCONCLUSIVE'))

    def test_raw_write_failure_keeps_partial_file_and_blocks_next_repeat(self):
        def partial(prefix, *args, **kwargs):
            Path(str(prefix)+'.raw.npz').write_bytes(b'synthetic partial raw')
            raise OSError('synthetic codec failure')
        with self.patches(), patch('p5_final_e1.raw_codec.write_raw', side_effect=partial):
            result = e1.run_e1(self.run, self.manifest, self.callback)
        self.assertEqual(result['status'], 'HOLD_INCOMPLETE_E1')
        self.assertEqual(self.replays, 1)
        failures = list((self.run/'E1').rglob('infrastructure_error.json'))
        failure = json.loads(failures[0].read_text())
        self.assertEqual(failure['actual_returned_intervals'], 5)
        self.assertIsNone(failure['raw_receipt'])
        self.assertTrue(failure['partial_files_sha256'])
        self.assertEqual(next((self.run/'E1').rglob('*.raw.npz')).read_bytes(), b'synthetic partial raw')

    def test_all_invalid_model_choices_are_preserved_before_any_actual_futures(self):
        self.runtime['models']['S1-23'].model.value = -1
        def stop(event):
            return False if event['event'] == 'predictions_frozen' else None
        with self.patches():
            result = e1.run_e1(self.run, self.manifest, stop)
        choices = json.loads((self.run/'E1/choices_before_any_physical_future.json').read_text())
        rows = [row for row in choices if row['model'] == 'S1-23']
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(row['selected_index'] is None and row['reason'] == 'planner_invalid' for row in rows))
        self.assertEqual(self.replays, 0)
        self.assertEqual(result['status'], 'HOLD_INCOMPLETE_E1')
        for path in (self.run/'E1').glob('snapshots/*/G/*/models/S1-23/trace.npz'):
            with np.load(path, allow_pickle=False) as arrays:
                self.assertFalse(arrays['valid'].any())
                self.assertEqual(arrays['valid'].shape, (16,))
                self.assertTrue(np.isposinf(arrays['total_cost']).all())

    def test_stop_after_durable_repeat_does_not_erase_completed_denominator(self):
        def stop(event):
            return False if event['event'] == 'repeat_complete' else None
        with self.patches():
            result = e1.run_e1(self.run, self.manifest, stop)
        self.assertEqual(result['status'], 'HOLD_INCOMPLETE_E1')
        self.assertEqual(self.replays, 1)
        self.assertEqual(result['attempted_physical_repeats'], 1)
        self.assertEqual(result['completed_physical_repeats'], 1)
        self.assertEqual(result['actual_returned_intervals_including_failed_prefixes'], 5)
        self.assertEqual(len(result['completed_repeat_ids']), 1)
        self.assertTrue(result['model_weights_unchanged_after_stop'])

    def test_naturally_invalid_short_repeat_remains_completed_evidence_and_consumes_reserved5(self):
        inputs, missing = e1._load_inputs(self.run, self.manifest['E1'], self.manifest, self.bindings)
        self.assertFalse(missing)
        data = inputs[0]
        data['folder'] = self.work/'short-invalid'
        data['folder'].mkdir()
        data['planner'] = deepcopy(self.planner)
        data['planner']['planner']['candidate_count'] = 16
        data['candidates'] = SimpleNamespace(actions=np.zeros((16, 5, 2), np.float64))
        state = data['capture']['observation'].state.copy()
        observation = Observation(state, 2.2, 'push-align-state-v1', False)
        raw = StepResult(np.zeros(2), np.zeros(2), observation, .05, False,
            {'samples': [{'state': state.tolist(), 'sim_time': 2.2, 'contacts': []}],
             'validity_violations': ['synthetic partial']}, state[None], np.array([2.2], np.float64))
        states = np.full((6, 10), np.nan, np.float64)
        states[0] = states[1] = state
        outcome = PhysicalCandidateReplay(states, np.full((5, 2), np.nan, np.float64), False,
            float('inf'), {name: float('inf') for name in e1.COMPONENT_NAMES}, 0, [raw], True,
            data['capture']['snapshot'], ('partial_physical_interval',))
        with patch('p5_final_e1.replay_candidate', return_value=outcome):
            recorded, row = e1._repeat(self.work, data, 0, 0, self.manifest, self.callback)
        self.assertEqual(row['status'], 'COMPLETE_RECORDED')
        self.assertFalse(row['physical_valid'])
        self.assertEqual(row['raw_receipt']['interval_count'], 1)
        self.assertEqual(row['actual_returned_intervals'], 1)
        complete = next(event for event in self.events if event['event'] == 'repeat_complete')
        self.assertEqual(complete['remove_reserved_intervals'], 5)
        self.assertEqual(complete['actual_returned_intervals'], 1)
        self.assertFalse(list(data['folder'].rglob('infrastructure_error.json')))

    def test_budget_callback_requires_an_explicit_continue_contract(self):
        for result in (False, 'STOP', 0, [], {'status': 'HOLD_CAPACITY'}):
            with self.subTest(result=result), self.assertRaises(e1.E1Stop):
                e1._notify(lambda event: result, 'before_repeat')
        for result in (None, True, {'status': 'CONTINUE'}, {'status': 'CONTINUE_RESOURCE_CHECK_ONLY'},
                       {'status': 'FIT_ESTIMATE_ONLY'}):
            e1._notify(lambda event: result, 'before_repeat')

    def test_callback_can_stop_stage_before_loading_inputs_or_creating_output(self):
        with self.assertRaises(e1.E1Stop):
            e1.run_e1(self.run, self.manifest, lambda event: False)
        self.assertFalse((self.run/'E1').exists())
        self.assertEqual(self.replays, 0)

    def test_unfrozen_manifest_wrong_scope_missing_model_and_escape_are_rejected(self):
        for fault in ('unfrozen', 'scope', 'model', 'escape'):
            manifest = deepcopy(self.manifest)
            if fault == 'unfrozen':
                manifest['status'] = 'NOT_FROZEN'
            elif fault == 'scope':
                manifest['E1'][0]['candidates'] = 32
            elif fault == 'model':
                manifest['model_identities'].pop('SK-47')
            else:
                manifest['runtime_source_sha256'] = {'../outside.py': '0'*64}
            with self.subTest(fault=fault), self.assertRaises(e1.E1Stop):
                e1.run_e1(self.run, manifest, self.callback)
        self.assertFalse((self.run/'E1').exists())

    def test_wrong_G_runtime_hash_and_changed_G_input_are_not_silently_used(self):
        spec = self.manifest['E1'][0]
        folder = self.run/'E2'/spec['family_id']/'G'
        metadata = next((folder/'plans').rglob('input.json.gz'))
        with metadata.open('ab') as stream:
            stream.write(b'synthetic corruption')
        with self.assertRaises(e1.E1Stop):
            e1.run_e1(self.run, self.manifest, self.callback)
        self.assertEqual(self.replays, 0)

    def test_original_source_changed_prevents_E1_before_model_work(self):
        self.source.write_text('changed synthetic source')
        with self.assertRaises(e1.E1Stop):
            e1.run_e1(self.run, self.manifest, self.callback)
        self.assertEqual(FakePredictor.calls_total, 0)
        self.assertEqual(self.replays, 0)

    def test_existing_stage_is_preserved_and_not_automatically_retried(self):
        (self.run/'E1').mkdir()
        (self.run/'E1/earlier_failure.json').write_text('preserve')
        with self.assertRaises(FileExistsError):
            e1.run_e1(self.run, self.manifest, self.callback)
        self.assertEqual((self.run/'E1/earlier_failure.json').read_text(), 'preserve')

    def test_model_and_sources_are_checked_after_all_physical_work(self):
        # Stop at the final repeat's callback after mutating an original. No
        # additional replay may mask this change or silently regenerate choices.
        def event(row):
            if row['event'] == 'repeat_complete' and self.replays == 576:
                self.source.write_text('changed during synthetic evaluation')
        with self.patches():
            result = e1.run_e1(self.run, self.manifest, event)
        self.assertEqual(result['status'], 'HOLD_INCOMPLETE_E1')
        self.assertIn('Frozen source/original changed', result['error']['message'])
        self.assertEqual(self.replays, 576)


if __name__ == '__main__':
    unittest.main()
