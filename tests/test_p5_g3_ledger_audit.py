"""Independent G3 admission/resume counterexamples using only tiny fake files.

No frozen family generation, predictor construction, model forward or physical
engine call occurs.  Rejected outcomes are asserted as the required behavior;
failures document the coordinator source hash named by the audit report.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('p5_g3_ledger_under_test', PROJECT/'scripts/p5_final_run.py')
coordinator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coordinator)


class G3LedgerAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='p5-g3-ledger-fixture-', dir=PROJECT/'tmp')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root/'run'
        self.run.mkdir()
        self.plan = json.loads((PROJECT/'configs/p5_g2_resource_plan.v3.json').read_text())
        self.write(self.root/'configs/plan.json', self.plan)
        # Handwritten labels exercise the ledger only; these are no final poses.
        self.cell = {'family_id':'fixture-family','episode_id':'fixture-family',
            'episode_index':0,'stratum':'translation','controller_id':'G','cell_id':'fixture-family/G'}
        self.entry = {'snapshot_id':'fixture-family/G/tick-000'}
        self.manifest = {'families':[], 'E2':[self.cell], 'E1':[self.entry],
            'resource_plan_path':'configs/plan.json', 'physical_config':{}, 'profile':{},
            'planner_config':{}, 'model_identities':{}, 'normalizer_identity':{},
            'application_family':{'family_id':'fixture-application'}}
        self.initial = {'E2_done':[], 'E1_done':[], 'replay_done':[],
            'application_done':[], 'replay_selection':None}
        self.write(self.run/'manifest.json', self.manifest)
        coordinator.save_progress(self.run, self.initial)
        self.root_patch = patch.object(coordinator, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        coordinator._SELECTION_CACHE.clear()
        coordinator._E1_RECEIPT_CACHE.clear()
        coordinator._COMPLETION_PROOF_CACHE.clear()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def completed_fixture(self, *, wrong_contract=False):
        folder = self.run/'E2'/self.cell['cell_id']
        folder.mkdir(parents=True)
        raw = folder/'tiny.raw'
        raw.write_bytes(b'preserved tiny raw fixture')
        witness = folder/'tiny.witness'
        witness.write_bytes(b'preserved tiny witness fixture')
        identity, bindings = coordinator.expected_recording_contract(self.run,self.manifest,self.cell)
        if wrong_contract:
            identity.update(study='another-study', model={'model_hash':'another-model'},
                normalizer={'normalizer_hash':'another-normalizer'},final_manifest_sha256='0'*64)
        execution = {'execution_status':'completed','termination':'timeout',
            'counts':{'physics_intervals_returned':1,'controller_decisions':1},'future_engine_calls':{'mj_step':0}}
        record = {'identity':identity,'record_kind':'reference_controller_execution','execution':execution,'files_sha256':{str(raw):coordinator.sha(raw)},
            'source_bindings':bindings,'recording_status':'COMPLETE_RECORDED',
            'raw':{'files_sha256':{str(raw):coordinator.sha(raw)}},'witness_path':str(witness),
            'full_traces':[],'resources':{'mismatch_charged':False}}
        record['files_sha256'][str(witness)] = coordinator.sha(witness)
        self.write(folder/'manifest.json', record)
        metrics = {'success':False}
        review = {'status':'ACCEPT_COMPLETED_TASK_FAILURE','accepted':True,
            'manifest_sha256':coordinator.sha(folder/'manifest.json'),'metrics':metrics}
        self.write(folder/'review.json', review)
        component = coordinator.measured_component_review(record,self.plan)
        self.write(folder/'component_budget_review.json',component)
        result = {'phase':'E2','cell':self.cell,'row':{**self.cell,**execution,**metrics},
            'status':review['status'],'record_manifest':str(folder/'manifest.json'),
            'record_manifest_sha256':coordinator.sha(folder/'manifest.json'),
            'review_sha256':coordinator.sha(folder/'review.json'),
            'component_review_sha256':coordinator.sha(folder/'component_budget_review.json')}
        self.write(folder/'worker_result.json', result)
        self.write(folder/'worker_result_receipt.json',{'sha256':coordinator.sha(folder/'worker_result.json')})
        return folder, review

    def test_missing_completed_E2_result_cannot_release_a_cell_reservation(self):
        state = deepcopy(self.initial)
        state['E2_done'] = [self.cell['cell_id']]
        coordinator.save_progress(self.run, state)
        with self.assertRaises((ValueError, RuntimeError, FileNotFoundError)):
            coordinator.remaining_estimate(self.run)

    def test_E1_repeat_receipt_requires_its_summary_and_raw_before_release(self):
        repeat_id = self.entry['snapshot_id']+'/candidate-00/repeat-0'
        receipt = self.run/'E1/receipts/repeats'/(hashlib.sha256(repeat_id.encode()).hexdigest()+'.json')
        self.write(receipt, {'repeat_id':repeat_id, 'summary_path':str(self.run/'E1/missing-summary.json'),
            'summary_sha256':'0'*64,'raw_files_sha256':{str(self.run/'E1/missing.raw'):'0'*64}})
        state = deepcopy(self.initial)
        state.update(E1_done=[repeat_id], E1_receipts={repeat_id:{'path':str(receipt),'sha256':coordinator.sha(receipt)}})
        coordinator.save_progress(self.run, state)
        with self.assertRaises((ValueError, RuntimeError, FileNotFoundError)):
            coordinator.remaining_estimate(self.run)

    def test_done_cell_still_requires_its_component_budget_receipt(self):
        folder, review = self.completed_fixture()
        with patch('push_align.final_result_review.review_recording',return_value=review):
            self.assertEqual(coordinator.verified_result(self.run,'E2',self.cell['cell_id'],self.manifest)['status'],
                'ACCEPT_COMPLETED_TASK_FAILURE')
        (folder/'component_budget_review.json').unlink()
        state = deepcopy(self.initial)
        state['E2_done'] = [self.cell['cell_id']]
        coordinator.save_progress(self.run, state)
        # Stub only the separate scientific raw evaluator.  Real coordinator
        # file seals/identity checks still run; no component file exists.
        with patch.object(coordinator, 'frozen', return_value=self.manifest), \
             patch('push_align.final_result_review.review_recording', return_value=review):
            with self.assertRaisesRegex(RuntimeError,'component budget'):
                coordinator.execute_cells(self.run, 'E2')
        self.assertFalse((folder/'component_budget_review.json').exists())

    def test_deep_adoption_requires_the_full_frozen_recording_identity(self):
        _, review = self.completed_fixture(wrong_contract=True)
        with patch('push_align.final_result_review.review_recording', return_value=review):
            with self.assertRaises((ValueError, RuntimeError)):
                coordinator.verified_result(self.run,'E2',self.cell['cell_id'],self.manifest)

    def test_valid_completed_baseline_has_full_contract_and_component_receipt(self):
        _, review = self.completed_fixture()
        with patch('push_align.final_result_review.review_recording',return_value=review):
            result = coordinator.verified_result(self.run,'E2',self.cell['cell_id'],self.manifest)
        self.assertEqual(result['status'],'ACCEPT_COMPLETED_TASK_FAILURE')

    def test_statistics_admission_stops_before_analysis_or_writes(self):
        self.write(self.run/'E2'/self.cell['cell_id']/'worker_result.json', {'fixture':True})
        with patch.object(coordinator,'frozen',return_value=self.manifest), \
             patch.object(coordinator,'capacity_check',side_effect=RuntimeError('Capacity STOP')) as admission, \
             patch.object(coordinator,'verified_result',return_value={'row':self.cell}), \
             patch('push_align.final_statistics.analyze_E2',return_value=({'fixture':True},{'tiny':np.zeros(1)})) as analyze, \
             patch.object(coordinator,'seal_derived_stage'):
            with self.assertRaisesRegex(RuntimeError,'Capacity STOP'):
                coordinator.statistics(self.run)
            admission.assert_called()
            analyze.assert_not_called()
        self.assertFalse((self.run/'statistics').exists())

    def test_prediction_admission_stops_before_runtime_or_writes(self):
        with patch.object(coordinator,'frozen',return_value=self.manifest), \
             patch.object(coordinator,'capacity_check',side_effect=RuntimeError('Capacity STOP')) as admission, \
             patch.object(coordinator,'runtime',return_value=({},None,{},{})) as runtime, \
             patch('push_align.final_prediction.evaluate_final_predictions',return_value=({'fixture':True},{'tiny':np.zeros(1)})) as evaluate:
            with self.assertRaisesRegex(RuntimeError,'Capacity STOP'):
                coordinator.prediction(self.run)
            admission.assert_called()
            runtime.assert_not_called()
            evaluate.assert_not_called()
        self.assertFalse((self.run/'fixed_prediction').exists())

    def test_capacity_still_rejects_no_credit_shortage(self):
        from types import SimpleNamespace
        reserve = 2*1024**3
        with patch.object(coordinator,'remaining_estimate',return_value={'required_bytes_including_reserve':reserve+1000}), \
             patch.object(coordinator.shutil,'disk_usage',return_value=SimpleNamespace(free=reserve+999)):
            with self.assertRaisesRegex(RuntimeError,'Capacity STOP'):
                coordinator.capacity_check(self.run)


if __name__ == '__main__':
    unittest.main()
