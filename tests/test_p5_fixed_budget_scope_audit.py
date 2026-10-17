"""Independent tiny-file accounting probes; no final inputs/model/physics.

Scientific stage reviewers are stubbed only in the overlay handoff probe.
That probe tests bookkeeping order, never scientific stage acceptance.
"""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from push_align.final_fixed_budget import POOLS, add_fixed_spend, read_fixed_spend
from push_align.final_recording import write_json_gzip


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('scope_audit_fixed_coordinator', PROJECT/'scripts/p5_final_run.py')
coordinator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coordinator)


class FixedBudgetScopeAudit(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='p5-fixed-scope-audit-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.run = self.root/'run'
        self.run.mkdir()
        self.plan = json.loads((PROJECT/'configs/p5_g2_resource_plan.v3.json').read_text())
        self.write(self.root/'configs/plan.json', self.plan)
        self.m = {'E2':[], 'E1':[], 'families':[], 'resource_plan_path':'configs/plan.json',
                  'application_family':{'family_id':'fixture-only'}, 'source_sha256':{}}
        self.write(self.run/'manifest.json', self.m)
        coordinator.save_progress(self.run, {'E2_done':[], 'E1_done':[], 'application_done':[],
            'replay_done':[], 'replay_selection':None})
        root_patch = patch.object(coordinator, 'ROOT', self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def file(self, relative, payload):
        path = self.run/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    def source(self, payload=b'1234567'):
        path = self.file('source/fixture.py', payload)
        self.m['source_sha256'][str(path.relative_to(self.root))] = coordinator.sha(path)
        self.write(self.run/'manifest.json', self.m)
        return path

    def cell(self, terminal=4):
        folder = self.run/'replays/fixture/G'
        raw = self.file('replays/fixture/G/raw.fixture', b'raw')
        witness = folder/'witness.json.gz'
        write_json_gzip(witness, {'unexecuted_plan':{'index':terminal}})
        traces = []
        for tick in (0, 4, 77):
            path = self.file(f'replays/fixture/G/trace-{tick}.fixture', bytes([tick]))
            traces.append({'tick':tick, 'files_sha256':{str(path):coordinator.sha(path)}})
        measured = [str(raw), str(witness), *[next(iter(t['files_sha256'])) for t in traces]]
        record = {'resources':{'mismatch_charged':True}, 'witness_path':str(witness),
                  'files_sha256':{str(witness):coordinator.sha(witness)},
                  'raw':{'files_sha256':{str(raw):coordinator.sha(raw)}}, 'full_traces':traces}
        self.write(folder/'manifest.json', record)
        self.write(folder/'component_budget_review.json', {'measured_component_paths':measured})
        return {'record_manifest':str(folder/'manifest.json'), 'cell':{'controller_id':'G'}}, raw, witness, traces

    def test_terminal_input_is_base_paid_but_other_differing_tick_is_extra(self):
        result, raw, _, traces = self.cell()
        calls = []
        with patch.object(coordinator, 'register_fixed_files', side_effect=lambda run,m,c,f:calls.append((c,f))):
            coordinator.register_cell_spending(self.run, self.m, 'replay', result)
        extra = dict(calls)['mismatch_extra']
        self.assertEqual(set(extra), {str(raw), next(iter(traces[2]['files_sha256']))})
        self.assertNotIn(next(iter(traces[1]['files_sha256'])), extra)

    def test_terminal_tick_cannot_be_changed_after_witness_sha_binding(self):
        result, _, witness, _ = self.cell()
        witness.unlink()  # Deliberate corruption of disposable fixture only.
        write_json_gzip(witness, {'unexecuted_plan':{'index':77}})
        with patch.object(coordinator, 'register_fixed_files'):
            with self.assertRaisesRegex((RuntimeError, ValueError), 'SHA|sha|hash'):
                coordinator.register_cell_spending(self.run, self.m, 'replay', result)

    def test_managed_payload_and_receipt_aliases_fail_before_another_revision(self):
        # Separate tiny runs avoid leaving one intentionally invalid graph in
        # the next subcase. Both aliases used to evade name-only exclusion.
        for name in ('fixed_spend.json', 'fixed_spend.receipt.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix='p5-managed-alias-') as temp:
                run = Path(temp).resolve()
                (run/'manifest.json').write_bytes(b'{"fixture_seed":123}')
                first = run/'source/a'; first.parent.mkdir(); first.write_bytes(b'x')
                scopes = {'source':{'pool':'source_and_model_bundle','prefixes':['source'],'exact_paths':[]},
                          'E1':{'pool':'E1_prediction_analysis','prefixes':['E1'],'exact_paths':[]}}
                kwargs = {'manifest_sha256':coordinator.sha(run/'manifest.json'),
                          'allowances':{pool:(2*1024**3 if pool=='hard_free_reserve' else 100000) for pool in POOLS},
                          'scopes':scopes}
                add_fixed_spend(run, **kwargs, category='source', files_sha256={str(first):coordinator.sha(first)})
                alias = run/'E1/alias'; alias.parent.mkdir(); os.link(run/name, alias)
                with self.assertRaisesRegex(RuntimeError, 'hardlink|alias'):
                    add_fixed_spend(run, **kwargs, category='E1', files_sha256={str(alias):coordinator.sha(alias)})
                self.assertFalse((run/'fixed_spend.000001.json').exists())
                with self.assertRaisesRegex(RuntimeError, 'hardlink|alias'):
                    read_fixed_spend(run, **kwargs, deep=True)

    def test_fixed_source_credit_is_exact_no_multiplier_and_keeps_base_counts(self):
        source = self.source(b'x'*103)
        coordinator.register_source_bundle(self.run, self.m)
        actual = coordinator.remaining_estimate(self.run)
        self.assertEqual(actual['counts'], self.plan['counts'])
        self.assertEqual(actual['fixed_spent_bytes'], source.stat().st_size)
        self.assertEqual(actual['required_bytes_including_reserve'],
                         self.plan['estimate']['required_bytes_including_reserve']-103)
        self.assertEqual(actual['parts_bytes']['hard_free_reserve'], 2*1024**3)

    def test_new_stage_rehashes_evidence_even_after_process_cache_exists(self):
        source = self.source(b'abcd')
        coordinator.register_source_bundle(self.run, self.m)
        self.assertEqual(coordinator.fixed_spending(self.run, self.m)['total_fixed_spent_bytes'], 4)
        source.write_bytes(b'efgh')
        with self.assertRaisesRegex(RuntimeError, 'SHA changed'):
            coordinator.begin_derived_stage(self.run, 'statistics')
        self.assertNotIn((str(self.run.resolve()),'statistics'), coordinator._DERIVED_ACTIVE_BYTES)

    def test_derived_overlay_handoff_credits_each_written_file_once_then_clears(self):
        self.source(b'abc')
        coordinator.register_source_bundle(self.run, self.m)
        with patch.object(coordinator, 'capacity_check', return_value={}):
            coordinator.begin_derived_stage(self.run, 'statistics')
            for filename in ('summary.json', 'bootstrap.npz', 'reproduction.json'):
                coordinator.save_derived_bytes(self.run, 'statistics', filename, b'fixture')
            coordinator.bounded_seal_derived_stage(self.run, 'statistics', {})
        key = (str(self.run.resolve()), 'statistics')
        actual_stage_bytes = sum(path.stat().st_size for path in (self.run/'statistics').iterdir())
        self.assertEqual(coordinator._DERIVED_ACTIVE_BYTES[key], actual_stage_bytes)
        self.assertEqual(coordinator.fixed_spending(self.run, self.m)['total_fixed_spent_bytes'], 3)
        with patch.object(coordinator, 'frozen', return_value=self.m), \
             patch('push_align.final_stage_review.validate_derived_stage', return_value={'fixture_only':True}):
            self.assertTrue(coordinator.adopt_derived_stage(self.run, 'statistics'))
        self.assertNotIn(key, coordinator._DERIVED_ACTIVE_BYTES)
        spending = coordinator.fixed_spending(self.run, self.m, deep=True)
        self.assertEqual(spending['spent_bytes_by_pool']['P5_boundary_analysis'], actual_stage_bytes)
        self.assertEqual(spending['total_fixed_spent_bytes'], actual_stage_bytes+3)
        self.assertTrue(set(spending['ledger_files_sha256']).isdisjoint(spending['files']))
        remaining = coordinator.remaining_estimate(self.run)
        self.assertEqual(remaining['required_bytes_including_reserve'],
            self.plan['estimate']['required_bytes_including_reserve']-actual_stage_bytes-3)


if __name__ == '__main__':
    unittest.main()
