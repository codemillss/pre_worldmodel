"""Tiny-file coordinator accounting; no final poses, model or physics work."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PROJECT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('fixed_budget_coordinator',PROJECT/'scripts/p5_final_run.py')
coordinator=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(coordinator)


class FixedBudgetIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='p5-fixed-integration-',dir=PROJECT/'tmp')
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.run=self.root/'run';self.run.mkdir()
        self.plan=json.loads((PROJECT/'configs/p5_g2_resource_plan.v3.json').read_text())
        self.write(self.root/'configs/plan.json',self.plan)
        self.m={'E2':[],'E1':[],'families':[],'resource_plan_path':'configs/plan.json',
                'application_family':{'family_id':'fixture-application'},'source_sha256':{}}
        self.write(self.run/'manifest.json',self.m)
        coordinator.save_progress(self.run,{'E2_done':[],'E1_done':[],'application_done':[],
            'replay_done':[],'replay_selection':None})
        self.stack=patch.object(coordinator,'ROOT',self.root);self.stack.start();self.addCleanup(self.stack.stop)

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

    def source(self,content=b'current copied source'):
        p=self.run/'source/src/copied.py';p.parent.mkdir(parents=True);p.write_bytes(content)
        self.m['source_sha256'][str(p.relative_to(self.root))]=coordinator.sha(p)
        self.write(self.run/'manifest.json',self.m)
        return p

    def test_initial_budget_is_not_lowered_by_old_development_files(self):
        old=self.root/'old-development/raw';old.parent.mkdir();old.write_bytes(b'old preserved data')
        self.m['source_sha256'][str(old.relative_to(self.root))]=coordinator.sha(old)
        self.write(self.run/'manifest.json',self.m)
        coordinator.register_source_bundle(self.run,self.m)
        result=coordinator.remaining_estimate(self.run)
        self.assertEqual(result['fixed_spent_bytes'],0)
        self.assertEqual(result['required_bytes_including_reserve'],self.plan['estimate']['required_bytes_including_reserve'])

    def test_current_copied_source_releases_exact_bytes_once_without_multiplier(self):
        source=self.source(b'x'*101)
        coordinator.register_source_bundle(self.run,self.m)
        first=coordinator.fixed_spending(self.run,self.m,deep=True)
        coordinator.register_source_bundle(self.run,self.m)
        again=coordinator.fixed_spending(self.run,self.m,deep=True)
        self.assertEqual(first['revision'],again['revision'])
        result=coordinator.remaining_estimate(self.run)
        self.assertEqual(result['fixed_spent_bytes'],101)
        self.assertEqual(result['required_bytes_including_reserve'],self.plan['estimate']['required_bytes_including_reserve']-101)
        self.assertEqual(result['parts_bytes']['hard_free_reserve'],2*1024**3)
        self.assertEqual(first['files'][str(source)]['category'],'source_bundle')

    def test_unbound_source_file_cannot_create_a_credit(self):
        self.source();(self.run/'source/unbound').write_bytes(b'unbound')
        with self.assertRaisesRegex(RuntimeError,'unbound'):coordinator.register_source_bundle(self.run,self.m)
        self.assertFalse((self.run/'fixed_spend.json').exists())

    def test_stage_entry_detects_same_length_changed_source_or_missing_file(self):
        source=self.source(b'abcd');coordinator.register_source_bundle(self.run,self.m)
        source.write_bytes(b'efgh')
        with self.assertRaisesRegex(RuntimeError,'SHA changed'):coordinator.fixed_spending(self.run,self.m,deep=True)
        source.unlink()
        with self.assertRaises(RuntimeError):coordinator.fixed_spending(self.run,self.m,deep=True)

    def test_receipt_write_keeps_the_hard_reserve(self):
        source=self.source()
        with patch.object(coordinator.shutil,'disk_usage',return_value=SimpleNamespace(free=2*1024**3+1)):
            with self.assertRaisesRegex(RuntimeError,'hard reserve'):
                coordinator.register_source_bundle(self.run,self.m)
        self.assertTrue(source.exists());self.assertFalse((self.run/'fixed_spend.json').exists())

    def test_receipt_metadata_is_charged_to_shared_P5_pool(self):
        self.plan['fixed_allowances_bytes']['P5_boundary_analysis']=5
        self.write(self.root/'configs/plan.json',self.plan)
        self.source()
        with self.assertRaisesRegex(RuntimeError,'shared analysis'):
            coordinator.register_source_bundle(self.run,self.m)
        self.assertFalse((self.run/'fixed_spend.json').exists())

    def cell(self,controller='G',phase='replay',terminal_tick=None):
        folder=self.run/('replays' if phase=='replay' else 'E2')/'fixture'/controller;folder.mkdir(parents=True)
        paths={}
        for name in ('raw','index','witness','normal-trace','extra-trace','metadata'):
            p=folder/name;p.write_bytes(name.encode());paths[name]={str(p):coordinator.sha(p)}
        from push_align.final_recording import write_json_gzip
        witness_path=folder/'witness.json.gz'
        write_json_gzip(witness_path,{'unexecuted_plan':None if terminal_tick is None else {'index':terminal_tick}})
        (folder/'witness').unlink()
        paths['witness']={str(witness_path):coordinator.sha(witness_path)}
        measured={p for name,files in paths.items() if name!='metadata' for p in files}
        if controller=='G':measured-=set(paths['index'])
        record={'resources':{'mismatch_charged':phase=='replay'},'raw':{'files_sha256':paths['raw']},
                'witness_path':next(iter(paths['witness'])),'plan_index':{'files_sha256':paths['index']},
                'full_traces':[{'tick':29,'files_sha256':paths['normal-trace']},
                               {'tick':77,'files_sha256':paths['extra-trace']}]}
        record['files_sha256']={p:h for files in paths.values() for p,h in files.items()}
        self.write(folder/'manifest.json',record)
        self.write(folder/'component_budget_review.json',{'measured_component_paths':sorted(measured)})
        return {'record_manifest':str(folder/'manifest.json'),'cell':{'controller_id':controller}},paths

    def test_G_mismatch_credit_excludes_base_witness_and_normal_G_capture(self):
        result,paths=self.cell();calls=[]
        with patch.object(coordinator,'register_fixed_files',side_effect=lambda run,m,c,f:calls.append((c,f))):
            coordinator.register_cell_spending(self.run,self.m,'replay',result)
        extra=dict(calls)['mismatch_extra']
        self.assertEqual(set(extra),set(paths['raw'])|set(paths['extra-trace']))
        self.assertFalse(set(extra)&(set(paths['witness'])|set(paths['normal-trace'])|set(paths['index'])))
        self.assertTrue(set(paths['index'])<=set(dict(calls)['cell_operational']))

    def test_G_unexecuted_terminal_capture_is_base_even_outside_predeclared_ticks(self):
        result,paths=self.cell(terminal_tick=77);calls=[]
        with patch.object(coordinator,'register_fixed_files',side_effect=lambda run,m,c,f:calls.append((c,f))):
            coordinator.register_cell_spending(self.run,self.m,'replay',result)
        self.assertEqual(set(dict(calls)['mismatch_extra']),set(paths['raw']))

    def test_learned_mismatch_credit_contains_only_extra_own_raw_index_traces(self):
        result,paths=self.cell('S1-11');calls=[]
        with patch.object(coordinator,'register_fixed_files',side_effect=lambda run,m,c,f:calls.append((c,f))):
            coordinator.register_cell_spending(self.run,self.m,'replay',result)
        extra=dict(calls)['mismatch_extra']
        expected=set().union(*(paths[key] for key in ('raw','index','normal-trace','extra-trace')))
        self.assertEqual(set(extra),expected);self.assertFalse(set(extra)&set(paths['witness']))

    def test_original_unit_files_never_receive_fixed_credit(self):
        result,paths=self.cell('R','E2');calls=[]
        with patch.object(coordinator,'register_fixed_files',side_effect=lambda run,m,c,f:calls.append((c,f))):
            coordinator.register_cell_spending(self.run,self.m,'E2',result)
        self.assertEqual([c for c,_ in calls],['cell_operational'])
        self.assertFalse(set(calls[0][1])&set().union(*(paths[k] for k in ('raw','index','witness','normal-trace','extra-trace'))))

    def test_transient_derived_overlay_counts_only_writes_in_this_process(self):
        with patch.object(coordinator,'capacity_check',return_value={}) as capacity:
            coordinator.begin_derived_stage(self.run,'statistics')
            coordinator.save_derived_bytes(self.run,'statistics','one',b'abc')
            self.assertEqual(capacity.call_args.kwargs['active_written_credit'],0)
            coordinator.save_derived_bytes(self.run,'statistics','two',b'de')
            self.assertEqual(capacity.call_args.kwargs['active_written_credit'],3)
        self.assertEqual(coordinator._DERIVED_ACTIVE_BYTES[str(self.run.resolve()),'statistics'],5)
        self.assertEqual(coordinator.fixed_spending(self.run,self.m,deep=True)['total_fixed_spent_bytes'],0)

    def test_unreserved_derived_write_stops_before_creating_output(self):
        with self.assertRaisesRegex(RuntimeError,'live stage reservation'):
            coordinator.save_derived_bytes(self.run,'statistics','bad',b'x')
        self.assertFalse((self.run/'statistics').exists())


if __name__=='__main__':unittest.main()
