"""V2 wiring against the existing pure orchestration contract; no physics."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_p5_final_orchestration as legacy

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('p5_recovery_orchestration_under_test', PROJECT/'scripts/p5_final_run_v2.py')
coordinator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coordinator)


class V2OrchestrationContractTests(legacy.OrchestrationPureTests):
    def setUp(self):
        patch.object(legacy, 'coordinator', coordinator).start()
        super().setUp()

    def test_incomplete_recording_stops_before_witness_or_component_adoption(self):
        folder = self.run/'incomplete'
        folder.mkdir()
        failure = folder/'recording_failure.json'
        self.write(failure, {'recording_error': {'type': 'DriverStop'}})
        recorded = SimpleNamespace(status='STOP_INCOMPLETE_RECORDING',
            manifest_path=failure, manifest={'recording_error': {'type': 'DriverStop', 'message': 'remaining_budget_exceeds_free'}},
            execution=SimpleNamespace(summary=lambda: {'termination':'timeout','execution_status':'completed'}))
        with patch.object(coordinator,'capacity_check',return_value={}) as capacity:
            with self.assertRaisesRegex(RuntimeError,'Incomplete recording retained'):
                coordinator.reject_incomplete_recording(recorded,run=self.run,folder=folder,
                    cell={'cell_id':'synthetic/R'},phase='E2',
                    guard=SimpleNamespace(last_decision={'status':'STOP_PRESERVE_PREFIX'}),timings={})
        stop=json.loads((folder/'worker_stop.json').read_text())
        self.assertFalse(stop['completed_for_comparison'])
        self.assertFalse(stop['automatic_retry'])
        self.assertEqual(stop['execution_summary']['termination'],'timeout')
        self.assertEqual(stop['recording_error']['type'],'DriverStop')
        self.assertTrue(capacity.call_args.kwargs['preserve'])
        self.assertEqual(coordinator.progress(self.run),self.initial)
        self.assertFalse((folder/'worker_result.json').exists())

    def test_complete_recording_passes_through_without_new_metadata(self):
        self.assertFalse(coordinator.reject_incomplete_recording(
            SimpleNamespace(status='COMPLETE_RECORDED'),run=self.run,
            folder=self.run/'absent',cell={},phase='E2',guard=None,timings={}))
        self.assertFalse((self.run/'absent').exists())

    def test_recording_stop_metadata_cannot_spend_the_reserve(self):
        recorded=SimpleNamespace(status='STOP_INCOMPLETE_RECORDING',manifest_path=None,
            manifest={'recording_error':{}},execution=SimpleNamespace(summary=lambda:{}))
        with patch.object(coordinator,'capacity_check',side_effect=RuntimeError('reserve stop')):
            with self.assertRaisesRegex(RuntimeError,'reserve stop'):
                coordinator.reject_incomplete_recording(recorded,run=self.run,folder=self.run,
                    cell={},phase='E2',guard=SimpleNamespace(last_decision={}),timings={})
        self.assertFalse((self.run/'worker_stop.json').exists())


class RecoveryScienceSourceTests(unittest.TestCase):
    def test_scientific_execution_and_statistics_functions_are_unchanged(self):
        original=ast.parse((PROJECT/'scripts/p5_final_run.py').read_text())
        updated=ast.parse((PROJECT/'scripts/p5_final_run_v2.py').read_text())
        functions=lambda tree:{node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
        old,new=functions(original),functions(updated)
        for name in ('runtime','_metric_row','record_call_timing','expected_cell','expected_recording_contract',
                     'measured_component_review','statistics','E1_statistics','prediction','E1',
                     'execute_cells','remaining_estimate','capacity_check'):
            # Only the selected version of format-compatible recording helpers changes.
            left=ast.dump(old[name],include_attributes=False)
            right=ast.dump(new[name],include_attributes=False).replace('push_align.final_driver_v2','push_align.final_driver')
            self.assertEqual(left,right,name)


if __name__=='__main__':
    unittest.main()
