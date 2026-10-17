"""Coordinator -> fixed-spend registration spies, never real budget credit.

The fixed-budget module's path/hash/size/union guards have their own tests.
Here real progress/callback wiring is exercised with synthetic sealed metadata;
model/physics/writer/source-adoption work is explicitly replaced by spies.
"""
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PROJECT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('p5_fixed_spending_wiring',PROJECT/'scripts/p5_final_run.py')
coordinator=importlib.util.module_from_spec(spec);spec.loader.exec_module(coordinator)


class E1FixedSpendingWiringTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='p5-e1-fixed-wiring-',dir=PROJECT/'tmp')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.run=self.root/'run';self.run.mkdir()
        self.manifest={'resource_plan_path':'configs/synthetic.json'}
        self.write(self.root/'configs/synthetic.json',{'units_bytes':{'raw_fixed':100,'raw_interval':10},
            'fixed_allowances_bytes':{'E1_prediction_analysis':1024**2}})
        self.write(self.run/'manifest.json',self.manifest)
        coordinator.save_progress(self.run,{'E1_done':[],'E1_receipts':{}})
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(coordinator,'ROOT',self.root))
        self.stack.enter_context(patch.object(coordinator,'frozen',return_value=self.manifest))
        self.stack.enter_context(patch.object(coordinator,'fixed_spending',return_value={'files':{}}))
        self.stack.enter_context(patch.object(coordinator,'register_source_bundle',return_value=None))
        self.stack.enter_context(patch.object(coordinator,'derived_pool_bytes',return_value=0))
        self.capacity=self.stack.enter_context(patch.object(coordinator,'capacity_check',return_value={'status':'CONTINUE_RESOURCE_CHECK_ONLY'}))
        self.register=self.stack.enter_context(patch.object(coordinator,'register_fixed_files',return_value={'synthetic_accounting_spy':True}))
        self.complete=self.stack.enter_context(patch.object(coordinator,'register_complete_E1_spending',return_value={'synthetic_complete_spy':True}))
        # E1 imports these into its callback closure at construction time.
        self.prediction_seal=self.stack.enter_context(patch('p5_final_e1_adoption.seal_prediction_bundle'))
        self.repeat_seal=self.stack.enter_context(patch('p5_final_e1_adoption.seal_repeat_receipt'))
        self.stage_seal=self.stack.enter_context(patch('p5_final_e1_adoption.seal_complete_e1'))

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

    def callback(self):
        holder={}
        def fake_writer(run,m,callback):
            holder['callback']=callback
            return {'status':'HOLD_SYNTHETIC_NO_WORK'}
        with (patch('p5_final_e1_adoption.adopt_complete_e1',return_value=None),
              patch('p5_final_e1.run_e1',side_effect=fake_writer)):
            with self.assertRaisesRegex(RuntimeError,'E1 incomplete'):coordinator.E1(self.run)
        self.capacity.reset_mock();self.register.reset_mock();self.complete.reset_mock()
        return holder['callback']

    def test_prediction_registers_only_the_sealed_prediction_graph_plus_its_receipt(self):
        callback=self.callback();trace=[]
        receipt={'files_sha256':{str(self.run/'E1/models/synthetic.npz'):'a'*64},
                 'path':str(self.run/'E1/receipts/predictions.json'),'sha256':'b'*64}
        def sealed(*args):trace.append('sealed');return receipt
        self.register.side_effect=lambda *args:trace.append('registered')
        self.capacity.side_effect=lambda *args,**kwargs:trace.append('capacity')
        self.prediction_seal.side_effect=sealed
        callback({'event':'predictions_frozen','choices_sha256':'synthetic-choices'})
        expected={**receipt['files_sha256'],receipt['path']:receipt['sha256']}
        self.register.assert_called_once_with(self.run,self.manifest,'E1_analysis',expected)
        self.assertEqual(trace,['sealed','registered','capacity'])
        self.assertEqual(coordinator.progress(self.run)['E1_done'],[])
        self.complete.assert_not_called()

    def test_prediction_seal_rejection_cannot_register_an_unverified_graph(self):
        callback=self.callback()
        self.prediction_seal.side_effect=ValueError('unverified prediction')
        with self.assertRaisesRegex(ValueError,'unverified prediction'):
            callback({'event':'predictions_frozen','choices_sha256':'synthetic'})
        self.register.assert_not_called();self.complete.assert_not_called()

    def test_repeat_registers_summary_and_receipt_but_never_raw_or_actual_bytes(self):
        callback=self.callback()
        repeat_id='synthetic/G/tick-000/candidate-00/repeat-0'
        summary=self.run/'E1/snapshots/synthetic/G/tick-000/physical/candidate-00/repeat-0/summary.json'
        self.write(summary,{'status':'COMPLETE_RECORDED','repeat_id':repeat_id})
        receipt={'path':str(self.run/'E1/receipts/repeats/synthetic.json'),'sha256':'b'*64,
            'summary_path':str(summary),'summary_sha256':coordinator.sha(summary),
            'raw_files_sha256':{str(summary.parent/'raw_execution.raw.npz'):'c'*64,
                                str(summary.parent/'raw_execution.raw.columns.xz'):'d'*64},
            'raw_bytes':125,'actual_bytes':5000}
        callback({'event':'before_repeat','repeat_id':repeat_id})
        callback({'event':'after_raw_write','repeat_id':repeat_id,'actual_returned_intervals':5,'bytes_written':125,'must_preserve':True})
        self.register.assert_not_called()
        self.assertEqual(self.capacity.call_args.kwargs['active_written_credit'],125)
        self.repeat_seal.return_value=receipt
        callback({'event':'repeat_complete','repeat_id':repeat_id})
        expected={str(summary):receipt['summary_sha256'],receipt['path']:receipt['sha256']}
        self.register.assert_called_once_with(self.run,self.manifest,'E1_analysis',expected)
        self.assertTrue(set(expected).isdisjoint(receipt['raw_files_sha256']))
        self.assertEqual(self.capacity.call_args.kwargs['active_written_credit'],0)
        self.assertEqual(coordinator.progress(self.run)['E1_done'],[repeat_id])
        self.complete.assert_not_called()

    def test_no_credit_for_snapshot_or_unfinished_failed_prefix_events(self):
        callback=self.callback()
        for event in ({'event':'snapshot_complete'},
                      {'event':'before_raw_write','actual_returned_intervals':2,'must_preserve':True},
                      {'event':'preserve_raw_write','actual_returned_intervals':2,'must_preserve':True},
                      {'event':'repeat_failed','must_preserve':True}):
            callback(event)
        self.register.assert_not_called();self.complete.assert_not_called()
        self.assertEqual(coordinator.progress(self.run)['E1_done'],[])

    def test_stage_complete_registers_only_after_the_complete_stage_seal(self):
        callback=self.callback();trace=[]
        self.complete.side_effect=lambda *args:trace.append('adopt_and_register_complete')
        self.stage_seal.side_effect=lambda *args:trace.append('stage_seal')
        callback({'event':'stage_complete','summary_sha256':'synthetic-summary'})
        self.assertEqual(trace,['stage_seal','adopt_and_register_complete'])
        self.complete.assert_called_once_with(self.run,self.manifest)
        self.register.assert_not_called()

    def test_failed_stage_seal_does_not_register_its_incomplete_outputs(self):
        callback=self.callback()
        self.stage_seal.side_effect=ValueError('stage incomplete')
        with self.assertRaisesRegex(ValueError,'stage incomplete'):
            callback({'event':'stage_complete','summary_sha256':'synthetic'})
        self.complete.assert_not_called();self.register.assert_not_called()

    def test_resume_complete_adoption_recovers_fixed_spending_without_rerunning_writer(self):
        adopted={'status':'ADOPT_COMPLETE_VERIFIED_E1','completed_repeat_ids':['synthetic-complete']}
        with (patch('p5_final_e1_adoption.adopt_complete_e1',return_value=adopted),
              patch.object(coordinator,'validated_progress',return_value={'E1_done':['synthetic-complete']}),
              patch('p5_final_e1.run_e1') as writer):
            result=coordinator.E1(self.run)
        self.assertIs(result,adopted)
        self.complete.assert_called_once_with(self.run,self.manifest)
        self.register.assert_not_called();writer.assert_not_called()

    def test_repeat_accounting_rejection_stops_before_done_commit(self):
        callback=self.callback();repeat_id='synthetic/G/tick-000/candidate-00/repeat-0'
        summary=self.run/'E1/snapshots/synthetic/G/tick-000/physical/candidate-00/repeat-0/summary.json'
        self.write(summary,{'synthetic':'complete-row-spy'})
        self.repeat_seal.return_value={'path':str(self.run/'E1/receipts/repeats/synthetic.json'),'sha256':'a'*64,
            'summary_path':str(summary),'summary_sha256':coordinator.sha(summary)}
        callback({'event':'before_repeat','repeat_id':repeat_id})
        self.register.side_effect=RuntimeError('fixed proof invalid')
        with self.assertRaisesRegex(RuntimeError,'fixed proof invalid'):
            callback({'event':'repeat_complete','repeat_id':repeat_id})
        self.assertEqual(coordinator.progress(self.run)['E1_done'],[])


class CompleteE1SpendingSelectionTests(unittest.TestCase):
    def test_actual_helper_adopts_first_then_excludes_raw_by_receipt_paths(self):
        # Two mocked completed IDs isolate file classification; the real
        # adoption contract checks576 and is independently tested elsewhere.
        with tempfile.TemporaryDirectory(prefix='p5-e1-complete-map-',dir=PROJECT/'tmp') as temporary:
            run=Path(temporary).resolve();manifest={'synthetic':'upstream-adoption-spy'}
            repeats=['synthetic/G/tick-000/candidate-00/repeat-0','synthetic/G/tick-000/candidate-00/repeat-1']
            raw,nonraw={},{}
            for repeat_id in repeats:
                snapshot,suffix=repeat_id.rsplit('/candidate-',1)
                folder=run/'E1/snapshots'/snapshot/'physical'/('candidate-'+suffix);folder.mkdir(parents=True)
                row={'raw_receipt':{'files_sha256':{str(folder/'raw.a'):'a'*64,str(folder/'raw.b'):'b'*64}}}
                path=folder/'summary.json';path.write_text(json.dumps(row))
                nonraw[str(path)]=coordinator.sha(path);raw.update(row['raw_receipt']['files_sha256'])
            extra=run/'E1/comparison_ledger.json';extra.write_text('synthetic complete ledger')
            nonraw[str(extra)]=coordinator.sha(extra)
            seal=run/'E1/stage_receipt.json';seal.write_text(json.dumps({'files_sha256':{**raw,**nonraw}}))
            adopted={'status':'ADOPT_COMPLETE_VERIFIED_E1','completed_repeat_ids':repeats}
            with (patch('p5_final_e1_adoption.adopt_complete_e1',return_value=adopted) as adoption,
                  patch.object(coordinator,'register_fixed_files') as register):
                result=coordinator.register_complete_E1_spending(run,manifest)
            adoption.assert_called_once_with(run,manifest)
            register.assert_called_once_with(run,manifest,'E1_analysis',{**nonraw,str(seal):coordinator.sha(seal)})
            self.assertIs(result,adopted)
            self.assertTrue(set(register.call_args.args[3]).isdisjoint(raw))

    def test_incomplete_original_E1_never_registers_any_spending(self):
        with tempfile.TemporaryDirectory(prefix='p5-e1-incomplete-map-',dir=PROJECT/'tmp') as temporary:
            with (patch('p5_final_e1_adoption.adopt_complete_e1',return_value=None),
                  patch.object(coordinator,'register_fixed_files') as register):
                with self.assertRaisesRegex(RuntimeError,'fully adopted source'):
                    coordinator.register_complete_E1_spending(Path(temporary),{})
                register.assert_not_called()


if __name__=='__main__':unittest.main()
