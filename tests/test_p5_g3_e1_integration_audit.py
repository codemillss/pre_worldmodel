"""Read-only coordinator wiring audit using inert synthetic callbacks.

The E1 library seals/writer are spies here: this suite tests the coordinator's
calls and persistence, not that library's own implementation. No final initial
condition generator, model runtime, controller, or physical engine may run.
"""
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('p5_coordinator_integration_audit', PROJECT/'scripts/p5_final_run.py')
coordinator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coordinator)
RESERVE = 2*1024**3


class CoordinatorE1WiringAudit(unittest.TestCase):
    def setUp(self):
        (PROJECT/'tmp').mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix='p5-g3-e1-wiring-', dir=PROJECT/'tmp')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.run = self.root/'run'
        self.run.mkdir()
        self.manifest = {'resource_plan_path': 'configs/synthetic_plan.json'}
        self.write(self.root/'configs/synthetic_plan.json', {'units_bytes': {'raw_fixed': 100, 'raw_interval': 10},
            'fixed_allowances_bytes': {'E1_prediction_analysis': 32*1024**2,'P5_boundary_analysis':32*1024**2}})
        self.write(self.run/'manifest.json', self.manifest)
        coordinator.save_progress(self.run, {'E1_done': [], 'E1_receipts': {}})
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(coordinator, 'ROOT', self.root))
        self.stack.enter_context(patch.object(coordinator, 'frozen', return_value=self.manifest))
        # Fixed-credit safety is tested separately. This older suite isolates
        # progress/capacity wiring and never publishes real spending receipts.
        self.stack.enter_context(patch.object(coordinator,'register_fixed_files',return_value={}))
        self.stack.enter_context(patch.object(coordinator,'register_complete_E1_spending',return_value={}))
        self.stack.enter_context(patch.object(coordinator,'fixed_spending',return_value={'files':{}}))
        self.stack.enter_context(patch.object(coordinator,'register_source_bundle',return_value=None))
        self.free = RESERVE+2000
        self.stack.enter_context(patch.object(coordinator.shutil, 'disk_usage', side_effect=lambda path: SimpleNamespace(free=self.free)))
        def remaining(run):
            # Whole-reservation accounting: the completed row releases its
            # fixed+five-interval allowance exactly once, after its seal.
            done = len(coordinator.progress(run)['E1_done'])
            return {'required_bytes_including_reserve': RESERVE+1000-225*done}
        self.stack.enter_context(patch.object(coordinator, 'remaining_estimate', side_effect=remaining))
        self.capacity = []
        original_capacity = coordinator.capacity_check
        def check(*args, **kwargs):
            result = original_capacity(*args, **kwargs)
            self.capacity.append((deepcopy(kwargs), deepcopy(result)))
            return result
        self.stack.enter_context(patch.object(coordinator, 'capacity_check', side_effect=check))

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def capture_callback(self):
        holder = {}
        def inert_writer(run, manifest, callback):
            holder['callback'] = callback
            return {'status': 'HOLD_SYNTHETIC_NO_WORK'}
        with (patch('p5_final_e1_adoption.adopt_complete_e1', return_value=None),
              patch('p5_final_e1.run_e1', side_effect=inert_writer)):
            with self.assertRaisesRegex(RuntimeError, 'E1 incomplete'):
                coordinator.E1(self.run)
        return holder['callback']

    def test_three_seals_are_called_in_order_and_progress_commit_follows_repeat_seal(self):
        trace, adopted = [], {'status': 'SYNTHETIC_ADOPTED'}
        repeat_id = 'synthetic/G/tick-00/candidate-00/repeat-0'
        summary = self.run/'E1/snapshots/synthetic/G/tick-00/physical/candidate-00/repeat-0/summary.json'
        def seal_prediction(run, digest):
            self.assertEqual(digest, 'synthetic-prediction-digest')
            self.assertEqual(coordinator.progress(run)['E1_done'], [])
            trace.append('prediction_seal')
            return {'files_sha256':{},'path':str(run/'E1/receipts/predictions.json'),'sha256':'0'*64}
        def seal_repeat(run, key, digest):
            self.assertEqual(key, repeat_id)
            self.assertEqual(digest, coordinator.sha(summary))
            self.assertEqual(coordinator.progress(run)['E1_done'], [])
            trace.append('repeat_seal')
            receipt = run/'E1/receipts/repeats/synthetic.json'
            self.write(receipt, {'repeat_id': key})
            return {'path': str(receipt), 'sha256': coordinator.sha(receipt),
                    'summary_path':str(summary),'summary_sha256':coordinator.sha(summary)}
        def seal_stage(run, digest):
            self.assertEqual(digest, 'synthetic-summary-digest')
            self.assertEqual(coordinator.progress(run)['E1_done'], [repeat_id])
            trace.append('stage_seal')
        def inert_writer(run, manifest, callback):
            callback({'event': 'predictions_frozen', 'choices_sha256': 'synthetic-prediction-digest'})
            callback({'event': 'before_repeat', 'repeat_id': repeat_id})
            self.assertEqual(self.capacity[-1][0]['next_write_bytes'], 225)
            callback({'event': 'before_raw_write', 'actual_returned_intervals': 3, 'must_preserve': True})
            callback({'event': 'after_raw_write', 'bytes_written': 125, 'actual_returned_intervals': 3, 'must_preserve': True})
            self.assertEqual(self.capacity[-1][0]['active_written_credit'], 125)
            self.assertEqual(self.capacity[-1][1]['remaining_required_bytes'], RESERVE+875)
            self.write(summary, {'status': 'SYNTHETIC', 'repeat_id': repeat_id})
            callback({'event': 'repeat_complete', 'repeat_id': repeat_id})
            self.assertEqual(self.capacity[-1][0]['active_written_credit'], 0)
            self.assertEqual(self.capacity[-1][1]['remaining_required_bytes'], RESERVE+775)
            callback({'event': 'stage_complete', 'summary_sha256': 'synthetic-summary-digest'})
            return {'status': 'COMPLETE_RECORDED'}
        with (patch('p5_final_e1_adoption.adopt_complete_e1', side_effect=[None, adopted]),
              patch('p5_final_e1_adoption.seal_prediction_bundle', side_effect=seal_prediction),
              patch('p5_final_e1_adoption.seal_repeat_receipt', side_effect=seal_repeat),
              patch('p5_final_e1_adoption.seal_complete_e1', side_effect=seal_stage),
              patch('p5_final_e1.run_e1', side_effect=inert_writer)):
            result = coordinator.E1(self.run)
        self.assertIs(result, adopted)
        self.assertEqual(trace, ['prediction_seal', 'repeat_seal', 'stage_seal'])
        state = coordinator.progress(self.run)
        self.assertEqual(state['E1_done'], [repeat_id])
        self.assertEqual(set(state['E1_receipts']), {repeat_id})
        self.assertEqual(len((self.run/'progress_events.jsonl').read_text().splitlines()), 1)

    def test_preservation_bypasses_whole_future_budget_but_keeps_hard_reserve_and_n_bound(self):
        callback = self.capture_callback()
        self.free = RESERVE+195
        callback({'event': 'preserve_raw_write', 'actual_returned_intervals': 3, 'must_preserve': True})
        self.assertEqual(self.capacity[-1][0]['next_write_bytes'], 195)
        self.assertTrue(self.capacity[-1][0]['preserve'])
        self.free = RESERVE+194
        with self.assertRaisesRegex(RuntimeError, 'Capacity STOP'):
            callback({'event': 'preserve_raw_write', 'actual_returned_intervals': 3, 'must_preserve': True})
        for n in (-1, 6, True, 1.5):
            with self.subTest(n=n), self.assertRaisesRegex(RuntimeError, 'Invalid E1 prefix reservation'):
                callback({'event': 'preserve_raw_write', 'actual_returned_intervals': n, 'must_preserve': True})

    def test_initial_repeat_admission_keeps_the_whole_budget_and_max5_write_reservation(self):
        callback = self.capture_callback()
        self.free = RESERVE+999
        with self.assertRaisesRegex(RuntimeError, 'Capacity STOP'):
            callback({'event': 'before_repeat', 'repeat_id': 'synthetic-repeat'})
        self.assertEqual(coordinator.progress(self.run)['E1_done'], [])

    def test_raw_growth_is_rejected_without_committing_or_releasing_the_repeat(self):
        callback = self.capture_callback()
        callback({'event': 'before_repeat', 'repeat_id': 'synthetic-repeat'})
        with self.assertRaisesRegex(RuntimeError, 'raw growth'):
            callback({'event': 'after_raw_write', 'bytes_written': 196,
                      'actual_returned_intervals': 3, 'must_preserve': True})
        self.assertEqual(coordinator.progress(self.run)['E1_done'], [])

    def test_complete_adoption_avoids_writer_and_rejects_different_resource_ids(self):
        adopted = {'completed_repeat_ids': ['synthetic-repeat']}
        with (patch('p5_final_e1_adoption.adopt_complete_e1', return_value=adopted),
              patch.object(coordinator, 'validated_progress', return_value={'E1_done': ['synthetic-repeat']}),
              patch('p5_final_e1.run_e1') as writer):
            self.assertIs(coordinator.E1(self.run), adopted)
            writer.assert_not_called()
        with (patch('p5_final_e1_adoption.adopt_complete_e1', return_value=adopted),
              patch.object(coordinator, 'validated_progress', return_value={'E1_done': []}),
              patch('p5_final_e1.run_e1') as writer):
            with self.assertRaisesRegex(RuntimeError, 'completed resource ledger'):
                coordinator.E1(self.run)
            writer.assert_not_called()

    def test_partial_e1_is_never_automatically_restarted(self):
        (self.run/'E1').mkdir()
        with (patch('p5_final_e1_adoption.adopt_complete_e1', return_value=None),
              patch('p5_final_e1.run_e1') as writer):
            with self.assertRaisesRegex(RuntimeError, 'Partial E1 retained'):
                coordinator.E1(self.run)
            writer.assert_not_called()

    def test_shared_analysis_pool_stops_all_four_checkpoints_before_seals_or_new_work(self):
        callback=self.capture_callback()
        with (patch.object(coordinator,'derived_pool_bytes',return_value=32*1024**2+1),
              patch('p5_final_e1_adoption.seal_prediction_bundle') as prediction_seal,
              patch('p5_final_e1_adoption.seal_complete_e1') as stage_seal):
            for event in ('predictions_frozen','before_repeat','snapshot_complete','stage_complete'):
                with self.subTest(event=event),self.assertRaisesRegex(RuntimeError,'Shared E1/prediction analysis allowance'):
                    callback({'event':event,'repeat_id':'synthetic-repeat'})
            prediction_seal.assert_not_called();stage_seal.assert_not_called()
            self.assertEqual(coordinator.progress(self.run)['E1_done'],[])
            # Already returned raw preservation still uses its own admitted
            # raw allowance plus hard reserve; this starts no future work.
            callback({'event':'preserve_raw_write','actual_returned_intervals':0,'must_preserve':True})
            self.assertTrue(self.capacity[-1][0]['preserve'])


class CoordinatorPersistenceAudit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='p5-g3-persistence-', dir=PROJECT/'tmp')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(coordinator, 'ROOT', self.root))
        # This suite verifies actual semantic stage adoption, while fixed-file
        # credit itself belongs to the separate accounting safety suite.
        self.stack.enter_context(patch.object(coordinator,'register_fixed_files',return_value={}))
        self.stack.enter_context(patch.object(coordinator,'register_source_bundle',return_value=None))

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def derived_fixture(self):
        from push_align.final_manifest import expected_E2
        from push_align.final_statistics import analyze_E2, METRICS
        run = self.root/'synthetic-run'
        families = [{'family_id':f'audit-{stratum}-{rank}', 'episode_id':f'audit-{stratum}-{rank}',
            'episode_index':stratum_index*4+rank, 'stratum':stratum}
            for stratum_index,stratum in enumerate(('translation','rotation','offset')) for rank in range(4)]
        expected = expected_E2(families)
        manifest = {'families':families, 'E2':expected, 'statistics':{'replicates':2000,'seed':7711}}
        self.write(run/'manifest.json', manifest)
        rows, sources = [], {}
        for cell in expected:
            row = {**cell, 'execution_status':'completed', 'termination':'timeout',
                **{metric:1. for metric in METRICS}, 'success':False, 'elapsed_s':30., 'recontact_count':1,
                'final_observation_available':True, 'metric_state_origin':'terminal_observation','metric_state_time_s':30.}
            rows.append(row)
            path=run/'E2'/cell['cell_id']/'worker_result.json'
            self.write(path,{'row':row,'synthetic_only':True})
            sources[str(path)]=coordinator.sha(path)
        summary,arrays=analyze_E2(rows,expected)
        self.write(run/'statistics/summary.json',summary)
        with (run/'statistics/bootstrap.npz').open('wb') as stream:np.savez_compressed(stream,**arrays)
        self.write(run/'statistics/reproduction.json',{'status':'PASS_EXACT_REGENERATION','source_rows':96,'families':12,
            'summary_sha256':coordinator.sha(run/'statistics/summary.json'),
            'bootstrap_sha256':coordinator.sha(run/'statistics/bootstrap.npz')})
        coordinator.seal_derived_stage(run,'statistics',sources)
        self.stack.enter_context(patch.object(coordinator,'frozen',return_value=manifest))
        by={row['cell_id']:row for row in rows}
        # Physical-source adoption is an explicit spy here. This test exercises
        # coordinator -> semantic table reviewer wiring and real recomputation
        # on96 fabricated complete rows, not576/96 actual physical executions.
        self.verified_rows=self.stack.enter_context(patch.object(coordinator,'verified_result',
            side_effect=lambda run,phase,key,m:{'row':deepcopy(by[key])}))
        return run,Path(next(iter(sources)))

    def test_derived_stage_adoption_binds_outputs_and_upstream_sources(self):
        run, raw = self.derived_fixture()
        self.assertTrue(coordinator.adopt_derived_stage(run, 'statistics'))
        self.assertEqual(self.verified_rows.call_count,96)
        raw.write_text('changed upstream row')
        with self.assertRaisesRegex(RuntimeError, 'Derived stage/source changed'):
            coordinator.adopt_derived_stage(run, 'statistics')

    def test_derived_stage_output_and_receipt_corruption_and_partial_directory_are_rejected(self):
        run, _ = self.derived_fixture()
        output = run/'statistics/summary.json'
        payload = output.read_bytes()
        output.write_bytes(payload+b' ')
        with self.assertRaisesRegex(RuntimeError, 'Derived stage/source changed'):
            coordinator.adopt_derived_stage(run, 'statistics')
        output.write_bytes(payload)
        receipt = run/'statistics/stage_receipt.json'
        receipt.write_bytes(receipt.read_bytes()+b' ')
        with self.assertRaisesRegex(RuntimeError, 'Derived stage identity/seal changed'):
            coordinator.adopt_derived_stage(run, 'statistics')
        (run/'fixed_prediction').mkdir()
        with self.assertRaisesRegex(RuntimeError, 'Incomplete derived stage'):
            coordinator.adopt_derived_stage(run, 'fixed_prediction')
        self.assertFalse(coordinator.adopt_derived_stage(run, 'absent'))

    def test_empty_and_HOLD_summary_only_derived_stages_are_no_longer_adopted(self):
        run=self.root/'synthetic-incomplete'
        self.write(run/'manifest.json',{'synthetic':'not a final manifest'})
        (run/'statistics').mkdir()
        coordinator.seal_derived_stage(run,'statistics',{})
        with self.assertRaisesRegex(RuntimeError,'Missing required derived'):
            coordinator.adopt_derived_stage(run,'statistics')
        self.write(run/'fixed_prediction/summary.json',{'status':'HOLD_INCOMPLETE','prediction_sets':0})
        coordinator.seal_derived_stage(run,'fixed_prediction',{})
        with self.assertRaisesRegex(RuntimeError,'Missing required derived'):
            coordinator.adopt_derived_stage(run,'fixed_prediction')

    def test_all_files_and_fresh_hashes_do_not_make_a_HOLD_summary_complete(self):
        run,_=self.derived_fixture()
        self.write(run/'statistics/summary.json',{'status':'HOLD_INCOMPLETE','episode_count':0})
        reproduction=coordinator.read(run/'statistics/reproduction.json')
        reproduction['summary_sha256']=coordinator.sha(run/'statistics/summary.json')
        self.write(run/'statistics/reproduction.json',reproduction)
        sources=coordinator.read(run/'statistics/stage_receipt.json')['source_sha256']
        (run/'statistics/stage_receipt.json').unlink()
        (run/'statistics/stage_receipt.sha256.json').unlink()
        coordinator.seal_derived_stage(run,'statistics',sources)
        with self.assertRaisesRegex(RuntimeError,'summary does not regenerate'):
            coordinator.adopt_derived_stage(run,'statistics')

    def freeze_fixture(self, *, free):
        run = self.root/'runs/p5-final-synthetic-audit'
        run.parent.mkdir()
        for relative, payload in {'docs/historical.md': 'current document',
                'prepared/historical.md': 'historical document', 'src/synthetic.py': '# inert\n',
                'tools/plot.py':'# inert standalone plotting helper\n'}.items():
            path = self.root/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload)
        self.write(self.root/'reports/p5_driver_native_preflight_review.json',
            {'status': 'PASS_BOUNDED_NATIVE_DRIVER_PREFLIGHT', 'checks': {'synthetic_gate': True}})
        self.write(self.root/'reports/p5_final_preparation_review.json',
            {'status': 'PASS', 'checks': {'synthetic_gate': True},
             'source_sha256':{name:coordinator.sha(self.root/name) for name in ('src/synthetic.py','tools/plot.py')},
             'source_archives': {'docs/historical.md': 'prepared/historical.md'}})
        self.write(self.root/'configs/p5_g2_resource_plan.v3.json', {'estimate': {'required_bytes_including_reserve': RESERVE+1000},
            'fixed_allowances_bytes':{'source_and_model_bundle':32*1024**2}})
        def contract(root, *, source_archives=None):
            document = (source_archives or {}).get('docs/historical.md', 'docs/historical.md')
            return {'source_sha256': {document: coordinator.sha(root/document),
                                     'src/synthetic.py': coordinator.sha(root/'src/synthetic.py')},
                'physical_config': {'synthetic': True},
                'profile': {'final_review': {'minimum_meaningful_improvement': 'not_specified'}},
                'model_identities': {}, 'normalizer_identity': {}, 'planner_config': {}}
        self.stack.enter_context(patch.object(coordinator.shutil, 'disk_usage', return_value=SimpleNamespace(free=free)))
        self.stack.enter_context(patch('push_align.final_manifest.frozen_source_contract', side_effect=contract))
        self.stack.enter_context(patch('push_align.final_manifest.prior_initial_inventory',
            return_value={'unresolved': [], 'canonical_identities': []}))
        # No real generation: fixed fabricated identities have no physical poses.
        families = [{'family_id': 'synthetic-'+stratum, 'stratum': stratum}
                    for stratum in ('translation', 'rotation', 'offset')]
        generator = self.stack.enter_context(patch('push_align.final_manifest.family_records',
            return_value=(families, {'family_id': 'synthetic-application'})))
        self.stack.enter_context(patch('push_align.final_manifest.expected_E2', return_value=[]))
        self.stack.enter_context(patch('push_align.final_manifest.expected_E1', return_value=[]))
        captured = {}
        def seal(path, document, **kwargs):
            captured.update(document=deepcopy(document), kwargs=kwargs)
            self.write(path, document)
            return {'sha256': coordinator.sha(path)}
        self.stack.enter_context(patch('push_align.final_manifest.freeze_manifest', side_effect=seal))
        return run, generator, captured

    def test_freeze_copies_historical_and_runtime_sources_and_binds_persistent_bundle(self):
        run, generator, captured = self.freeze_fixture(free=RESERVE+2000)
        coordinator.freeze(run)
        generator.assert_called_once()
        document = captured['document']
        target = run/'source/docs/historical.md'
        self.assertEqual(target.read_text(), 'historical document')
        self.assertEqual((self.root/'docs/historical.md').read_text(), 'current document')
        archive = str(target.relative_to(self.root))
        self.assertEqual(document['source_archives'], {'docs/historical.md': archive})
        self.assertEqual(document['source_bundle_map']['docs/historical.md'], archive)
        self.assertEqual(document['source_sha256'][archive], coordinator.sha(target))
        self.assertEqual((run/'source/src/synthetic.py').read_bytes(), (self.root/'src/synthetic.py').read_bytes())
        tool_copy=run/'source/tools/plot.py'
        tool_relative=str(tool_copy.relative_to(self.root))
        self.assertEqual(tool_copy.read_bytes(),(self.root/'tools/plot.py').read_bytes())
        self.assertEqual(document['source_sha256']['tools/plot.py'],coordinator.sha(self.root/'tools/plot.py'))
        self.assertEqual(document['source_sha256'][tool_relative],coordinator.sha(tool_copy))
        self.assertEqual(document['source_bundle_map']['tools/plot.py'],tool_relative)
        coordinator.verify(document['source_sha256'])
        self.assertEqual(coordinator.sha(run/'manifest.json'), coordinator.read(run/'freeze_receipt.json')['sha256'])
        self.assertTrue((run/'progress_receipt.json').is_file())
        capacity=coordinator.read(run/'resource_before_freeze.json')['capacity']
        self.assertEqual(capacity['required_bytes'],max(RESERVE,RESERVE+1000-coordinator.tree_bytes(run/'source')))
        target.write_text('changed bundle')
        with self.assertRaisesRegex(RuntimeError, 'Frozen source/evidence changed'):
            coordinator.verify(document['source_sha256'])

    def test_freeze_capacity_hold_precedes_any_family_generation_and_preserves_receipt(self):
        run, generator, captured = self.freeze_fixture(free=RESERVE-1)
        with self.assertRaisesRegex(RuntimeError, 'final families not generated'):
            coordinator.freeze(run)
        generator.assert_not_called()
        self.assertFalse(captured)
        self.assertTrue((run/'source/docs/historical.md').is_file())
        receipt = coordinator.read(run/'resource_before_freeze.json')
        self.assertEqual(receipt['capacity']['status'], 'HOLD_CAPACITY')
        self.assertFalse((run/'manifest.json').exists())

    def test_stale_PASS_preparation_source_stops_before_run_creation_or_family_generation(self):
        run,generator,captured=self.freeze_fixture(free=RESERVE+2000)
        (self.root/'src/synthetic.py').write_text('# changed after preparation PASS\n')
        with self.assertRaisesRegex(RuntimeError,'Frozen source/evidence changed'):
            coordinator.freeze(run)
        generator.assert_not_called()
        self.assertFalse(captured)
        self.assertFalse(run.exists())

    def test_PASS_preparation_without_source_bindings_stops_before_run_creation_or_family_generation(self):
        run,generator,captured=self.freeze_fixture(free=RESERVE+2000)
        path=self.root/'reports/p5_final_preparation_review.json'
        readiness=coordinator.read(path);readiness.pop('source_sha256')
        self.write(path,readiness)
        with self.assertRaisesRegex(RuntimeError,'lacks exact reviewed source bindings'):
            coordinator.freeze(run)
        generator.assert_not_called()
        self.assertFalse(captured)
        self.assertFalse(run.exists())


if __name__ == '__main__':
    unittest.main()
