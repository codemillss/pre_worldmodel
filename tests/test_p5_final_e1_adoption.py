"""Sealed E1 adoption checks on one wholly synthetic576-repeat fixture."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

import p5_final_e1 as e1
import p5_final_e1_adoption as adoption
import test_p5_final_e1 as fixtures
from push_align.final_recording import file_sha256
from push_align import final_recording_columns as codec
from push_align.planning import PredictionBatch, score
from push_align.reproduction import json_safe


class FinalE1AdoptionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = fixtures.FinalE1Test(methodName='runTest')
        cls.fixture.setUp()
        cls.final_run = cls.fixture.run
        cls.manifest = cls.fixture.manifest
        manifest_path = cls.final_run/'manifest.json'
        manifest_path.write_text(json.dumps(json_safe(cls.manifest)))
        (cls.final_run/'freeze_receipt.json').write_text(json.dumps({'sha256': file_sha256(manifest_path)}))
        def consistent_replay(physical, planner, snapshot, actions, task, previous, profile):
            result = cls.fixture.fake_replay(physical, planner, snapshot, actions, task, previous, profile)
            predictions = PredictionBatch(result.states[None], np.diff(result.states, axis=0)[None],
                np.array([True]), ('',), np.array([-1], np.int64))
            costs = score(predictions, actions[None], task, previous, planner)
            return replace(result, cost=float(costs.total[0]),
                           cost_components={key: float(value[0]) for key, value in costs.components.items()})
        def callback(event):
            cls.fixture.events.append(event)
            if event['event'] == 'predictions_frozen':
                adoption.seal_prediction_bundle(cls.final_run, event['choices_sha256'])
            elif event['event'] == 'repeat_complete':
                spec, candidate, repeat, folder = adoption._repeat(cls.final_run, cls.manifest, event['repeat_id'])
                adoption.seal_repeat_receipt(cls.final_run, event['repeat_id'], file_sha256(folder/'summary.json'))
            elif event['event'] == 'stage_complete':
                adoption.seal_complete_e1(cls.final_run, event['summary_sha256'])
        with cls.fixture.patches(replay=consistent_replay):
            cls.result = e1.run_e1(cls.final_run, cls.manifest, callback)
        if cls.result['status'] != 'COMPLETE_RECORDED':
            raise AssertionError(cls.result)
        cls.first_id = cls.manifest['E1'][0]['snapshot_id']+'/candidate-00/repeat-0'
        cls.stage_receipt = cls.final_run/'E1/stage_receipt.json'

    @classmethod
    def tearDownClass(cls):
        cls.fixture.doCleanups()

    def edit_temporarily(self, path, edit):
        path = Path(path)
        original = path.read_bytes()
        self.addCleanup(path.write_bytes, original)
        edit(path)

    def test_complete_sealed_scope_adopts_without_model_or_physical_execution(self):
        with (patch('p5_final_e1.replay_candidate', side_effect=AssertionError('no new physics')),
              patch('p5_final_e1.LearnedPredictor', side_effect=AssertionError('no model inference'))):
            result = adoption.adopt_complete_e1(self.final_run, self.manifest)
        self.assertEqual(result['status'], 'ADOPT_COMPLETE_VERIFIED_E1')
        self.assertEqual(len(result['completed_repeat_ids']), 576)
        self.assertEqual(result['summary']['diagnostic_count'], 84)
        self.assertEqual(result['summary']['actual_future_intervals'], 2880)
        self.assertGreater(result['actual_bytes'], 0)

    def test_repeat_deep_and_shallow_agree_on_identity_and_byte_counts(self):
        deep = adoption.verified_repeat_receipt(self.final_run, self.first_id, deep=True)
        shallow = adoption.verified_repeat_receipt(self.final_run, self.first_id, deep=False)
        self.assertEqual(deep['summary_sha256'], shallow['summary_sha256'])
        self.assertEqual(deep['actual_bytes'], shallow['actual_bytes'])
        self.assertEqual(deep['raw_regeneration']['intervals'], 5)
        self.assertIn('not rehashed', shallow['verification'])

    def test_unknown_and_out_of_range_repeat_ids_are_never_accepted(self):
        for repeat_id in ('unknown', self.first_id.replace('candidate-00', 'candidate-16'),
                          self.first_id.replace('repeat-0', 'repeat-3')):
            with self.subTest(repeat_id=repeat_id), self.assertRaises(adoption.E1AdoptionError):
                adoption.verified_repeat_receipt(self.final_run, repeat_id)

    def test_missing_repeat_seal_is_not_created_during_adoption(self):
        path = adoption._repeat_seal_path(self.final_run, self.first_id)
        self.edit_temporarily(path, lambda p: p.unlink())
        with self.assertRaises(adoption.E1AdoptionError):
            adoption.verified_repeat_receipt(self.final_run, self.first_id)
        self.assertFalse(path.exists())

    def test_changed_repeat_summary_is_rejected_before_raw_reuse(self):
        _, _, _, folder = adoption._repeat(self.final_run, self.manifest, self.first_id)
        def change(path):
            row = json.loads(path.read_text())
            row['actual_returned_intervals'] = 4
            path.write_text(json.dumps(row))
        self.edit_temporarily(folder/'summary.json', change)
        with self.assertRaises(adoption.E1AdoptionError):
            adoption.verified_repeat_receipt(self.final_run, self.first_id, deep=False)

    def test_raw_hash_detects_changed_bytes_even_at_same_file_size(self):
        _, _, _, folder = adoption._repeat(self.final_run, self.manifest, self.first_id)
        path = next(folder.glob('*.raw.npz'))
        def change(path):
            data = bytearray(path.read_bytes())
            data[-1] ^= 1
            path.write_bytes(data)
        self.edit_temporarily(path, change)
        with self.assertRaises(adoption.E1AdoptionError):
            adoption.verified_repeat_receipt(self.final_run, self.first_id, deep=True)

    def test_missing_complete_stage_seal_is_not_invented_from_summary(self):
        self.edit_temporarily(self.stage_receipt, lambda p: p.unlink())
        with self.assertRaises(adoption.E1AdoptionError):
            adoption.adopt_complete_e1(self.final_run, self.manifest)
        self.assertFalse(self.stage_receipt.exists())

    def test_actual_future_tensor_corruption_blocks_adoption(self):
        spec = self.manifest['E1'][0]
        path = self.final_run/'E1/snapshots'/spec['snapshot_id']/'actual_futures.npz'
        self.edit_temporarily(path, lambda p: p.write_bytes(p.read_bytes()+b'changed'))
        with self.assertRaises(adoption.E1AdoptionError):
            adoption.adopt_complete_e1(self.final_run, self.manifest)

    def test_changed_prediction_or_diagnostic_file_blocks_adoption(self):
        spec = self.manifest['E1'][0]
        path = self.final_run/'E1/snapshots'/spec['snapshot_id']/'models/R/diagnostics.json'
        def change(path):
            row = json.loads(path.read_text())
            row['selection_loss_denominator'] = 0
            path.write_text(json.dumps(row))
        self.edit_temporarily(path, change)
        with self.assertRaises(adoption.E1AdoptionError):
            adoption.adopt_complete_e1(self.final_run, self.manifest)

    def test_complete_summary_with_post_summary_failure_is_never_adopted(self):
        path = self.final_run/'E1/post_summary_failure.json'
        path.write_text('{}')
        self.addCleanup(path.unlink)
        with self.assertRaises(adoption.E1AdoptionError):
            adoption.adopt_complete_e1(self.final_run, self.manifest)

    def test_absent_or_explicit_partial_summary_returns_none_without_retry(self):
        path = self.final_run/'E1/summary.json'
        self.edit_temporarily(path, lambda p: p.unlink())
        self.assertIsNone(adoption.adopt_complete_e1(self.final_run, self.manifest))
        path.write_text(json.dumps({'status': 'HOLD_INCOMPLETE_E1'}))
        self.assertIsNone(adoption.adopt_complete_e1(self.final_run, self.manifest))

    def test_wrong_caller_manifest_and_changed_freeze_are_rejected(self):
        altered = deepcopy(self.manifest)
        altered['status'] = 'NOT_FROZEN'
        with self.assertRaises(adoption.E1AdoptionError):
            adoption.adopt_complete_e1(self.final_run, altered)
        self.edit_temporarily(self.final_run/'manifest.json', lambda p: p.write_bytes(p.read_bytes()+b' '))
        with self.assertRaises(adoption.E1AdoptionError):
            adoption.verified_repeat_receipt(self.final_run, self.first_id)

    def test_prediction_seal_cannot_be_backdated_after_physical_attempts(self):
        path = self.final_run/'E1/receipts/predictions.json'
        self.edit_temporarily(path, lambda p: p.unlink())
        with self.assertRaisesRegex(adoption.E1AdoptionError, 'Cannot backdate'):
            adoption.seal_prediction_bundle(self.final_run,
                file_sha256(self.final_run/'E1/choices_before_any_physical_future.json'))
        self.assertFalse(path.exists())

    def test_fresh_summary_seal_still_rejects_cost_that_does_not_regenerate_from_raw(self):
        _, _, _, folder = adoption._repeat(self.final_run, self.manifest, self.first_id)
        seal = adoption._repeat_seal_path(self.final_run, self.first_id)
        self.edit_temporarily(seal, lambda p: p.unlink())
        def change(path):
            row = json.loads(path.read_text())
            row['cost'] += .01
            path.write_text(json.dumps(row))
        self.edit_temporarily(folder/'summary.json', change)
        with self.assertRaisesRegex(adoption.E1AdoptionError, 'does not regenerate'):
            adoption.seal_repeat_receipt(self.final_run, self.first_id, file_sha256(folder/'summary.json'))
        self.assertFalse(seal.exists())

    def test_physically_invalid_complete_prefix_preserves_infinite_cost_and_denominator(self):
        _, _, _, folder = adoption._repeat(self.final_run, self.manifest, self.first_id)
        summary_path = folder/'summary.json'
        seal_path = adoption._repeat_seal_path(self.final_run, self.first_id)
        raw_paths = codec.raw_paths(folder/'raw_execution')
        backups = {path: path.read_bytes() for path in (*raw_paths, summary_path, seal_path)}
        def restore():
            for path, payload in backups.items():
                path.write_bytes(payload)
        self.addCleanup(restore)
        row = json.loads(summary_path.read_text())
        initial, results = codec.read_raw(folder/'raw_execution', expected_sha256=row['raw_receipt']['files_sha256'])
        results[-1] = replace(results[-1], transition_valid=False)
        for path in raw_paths:
            path.unlink()
        receipt = codec.write_raw(folder/'raw_execution', initial, results)
        row.update(physical_valid=False, cost=None,
            cost_components={name: 'NaN' for name in e1.COMPONENT_NAMES}, raw_receipt=receipt,
            reasons=['synthetic invalid transition'])
        summary_path.write_text(json.dumps(row))
        seal_path.unlink()
        adoption.seal_repeat_receipt(self.final_run, self.first_id, file_sha256(summary_path))
        verified = adoption.verified_repeat_receipt(self.final_run, self.first_id)
        self.assertFalse(verified['raw_regeneration']['valid'])
        self.assertTrue(np.isposinf(verified['raw_regeneration']['cost']))
        self.assertEqual(verified['raw_regeneration']['intervals'], 5)
        self.assertEqual(verified['summary']['status'], 'COMPLETE_RECORDED')

    def test_existing_seals_are_never_overwritten(self):
        digest = file_sha256(self.stage_receipt)
        with self.assertRaises(FileExistsError):
            adoption.seal_complete_e1(self.final_run, file_sha256(self.final_run/'E1/summary.json'))
        self.assertEqual(file_sha256(self.stage_receipt), digest)


if __name__ == '__main__':
    unittest.main()
