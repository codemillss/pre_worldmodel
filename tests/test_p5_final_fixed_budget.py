"""Pure file/receipt graph tests; no model, final poses or physical execution."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from push_align.final_fixed_budget import POOLS, read_fixed_spend, add_fixed_spend
from push_align import final_fixed_budget as budget


class FinalFixedBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='p5-fixed-budget-fixture-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.run = self.base/'fixture-run'
        self.run.mkdir()
        (self.run/'manifest.json').write_text('{"fixture":123}')
        self.manifest_sha = self.sha(self.run/'manifest.json')
        self.allowances = {pool:1000 for pool in POOLS}
        self.allowances['hard_free_reserve'] = 2*1024**3
        self.scopes = {
            'source_bundle':{'pool':'source_and_model_bundle','prefixes':['source/'],'exact_paths':[]},
            'E1_analysis':{'pool':'E1_prediction_analysis','prefixes':['E1/'],'exact_paths':[]},
            'fixed_prediction':{'pool':'E1_prediction_analysis','prefixes':['fixed_prediction/'],'exact_paths':[]},
            'cell_operationals':{'pool':'P5_boundary_analysis','prefixes':['E2/','application/','replays/'],'exact_paths':[]},
            'E2_statistics':{'pool':'P5_boundary_analysis','prefixes':['statistics/'],'exact_paths':[]},
            'mismatch_extra':{'pool':'mismatch_prefix_and_diagnostics','prefixes':['replays/'],'exact_paths':[]},
            'explicit_report':{'pool':'future_report_archive_QA','prefixes':[],'exact_paths':['report/summary.txt']},
            'ledger_self_forbidden':{'pool':'working_disk_scratch','prefixes':[],'exact_paths':['fixed_spend.json']}}

    def sha(self,path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def file(self,name,data=b'known bytes'):
        path=self.run/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        return path

    def read(self,**kwargs):
        values=dict(manifest_sha256=self.manifest_sha,allowances=self.allowances,scopes=self.scopes)
        values.update(kwargs)
        return read_fixed_spend(self.run,**values)

    def add(self,category,*paths,**kwargs):
        values=dict(manifest_sha256=self.manifest_sha,allowances=self.allowances,scopes=self.scopes,
            category=category,files_sha256={str(path):self.sha(path) for path in paths})
        values.update(kwargs)
        return add_fixed_spend(self.run,**values)

    def rewrite_snapshot(self,revision,change):
        path,receipt=budget._paths(self.run,revision)
        value=json.loads(path.read_text());change(value);path.write_text(json.dumps(value))
        seal=json.loads(receipt.read_text());seal['sha256']=self.sha(path);receipt.write_text(json.dumps(seal))

    def test_absent_ledger_returns_zero_credit_without_writing(self):
        result=self.read()
        self.assertIsNone(result['revision'])
        self.assertEqual(result['remaining_allowances_bytes'],self.allowances)
        self.assertEqual(result['total_fixed_spent_bytes'],0)
        self.assertEqual(list(self.run.glob('fixed_spend.*')),[])

    def test_exact_stat_bytes_are_credited_and_original_reserve_is_preserved(self):
        path=self.file('source/model.bin',b'1234567')
        result=self.add('source_bundle',path)
        self.assertEqual(result['revision'],0)
        self.assertEqual(result['spent_bytes_by_pool']['source_and_model_bundle'],7)
        self.assertEqual(result['remaining_allowances_bytes']['source_and_model_bundle'],993)
        self.assertEqual(result['remaining_allowances_bytes']['hard_free_reserve'],2*1024**3)
        self.assertEqual(result['total_fixed_spent_bytes'],7)
        self.assertEqual(result['total_fixed_remaining_bytes'],sum(self.allowances.values())-7)
        self.assertEqual(result['files'][str(path)]['sha256'],self.sha(path))

    def test_categories_share_one_pool_and_delta_snapshots_remain_linear(self):
        first=self.file('E1/predictions.npz',b'a'*11)
        second=self.file('fixed_prediction/traces.npz',b'b'*13)
        self.add('E1_analysis',first)
        initial=(self.run/'fixed_spend.json').read_bytes()
        result=self.add('fixed_prediction',second)
        self.assertEqual(result['spent_bytes_by_pool']['E1_prediction_analysis'],24)
        self.assertEqual(result['remaining_allowances_bytes']['E1_prediction_analysis'],976)
        self.assertEqual((self.run/'fixed_spend.json').read_bytes(),initial)
        newest=json.loads((self.run/'fixed_spend.000001.json').read_text())
        self.assertEqual(len(newest['entries']),1)
        self.assertEqual(newest['entries'][0]['path'],'fixed_prediction/traces.npz')
        self.assertEqual(len(result['ledger_files_sha256']),4)

    def test_relative_and_absolute_files_cannot_be_double_credited(self):
        path=self.file('source/model.bin')
        self.add('source_bundle',path)
        with self.assertRaisesRegex(RuntimeError,'duplicate'):
            self.add('source_bundle',files_sha256={'source/model.bin':self.sha(path)})
        self.assertEqual(self.read()['revision'],0)

    def test_duplicate_same_file_across_different_pools_is_rejected(self):
        path=self.file('replays/family/receipt.json')
        self.add('cell_operationals',path)
        with self.assertRaisesRegex(RuntimeError,'duplicate'):
            self.add('mismatch_extra',path)
        self.assertEqual(self.read()['spent_bytes_by_pool']['mismatch_prefix_and_diagnostics'],0)

    def test_duplicate_same_file_across_categories_of_one_pool_is_rejected(self):
        self.scopes['E1_alias']={'pool':'E1_prediction_analysis','prefixes':['E1/'],'exact_paths':[]}
        path=self.file('E1/choices.json');self.add('E1_analysis',path)
        with self.assertRaisesRegex(RuntimeError,'duplicate'):
            self.add('E1_alias',path)

    def test_hardlink_alias_is_not_a_second_actual_file_credit(self):
        first=self.file('source/first.bin');second=self.run/'source/second.bin';os.link(first,second)
        with self.assertRaisesRegex(RuntimeError,'duplicate'):
            self.add('source_bundle',first,second)
        self.assertIsNone(self.read()['revision'])

    def test_distinct_physical_copies_with_identical_content_are_separate_files(self):
        first=self.file('source/first.bin',b'identical')
        second=self.file('source/second.bin',b'identical')
        self.assertEqual(self.sha(first),self.sha(second))
        self.assertNotEqual(first.stat().st_ino,second.stat().st_ino)
        self.assertEqual(self.add('source_bundle',first,second)['total_fixed_spent_bytes'],18)

    def test_ledger_hardlink_inside_an_allowed_evidence_scope_cannot_credit_itself(self):
        self.allowances['E1_prediction_analysis']=100000
        original=self.file('source/first.bin',b'x');self.add('source_bundle',original)
        alias=self.run/'E1/ledger_alias.json';alias.parent.mkdir()
        os.link(self.run/'fixed_spend.json',alias)
        with self.assertRaisesRegex(RuntimeError,'hardlink|alias'):
            self.add('E1_analysis',alias)
        self.assertFalse((self.run/'fixed_spend.000001.json').exists())

    def test_external_receipt_hardlink_is_also_not_eligible_evidence(self):
        self.allowances['E1_prediction_analysis']=100000
        original=self.file('source/first.bin',b'x');self.add('source_bundle',original)
        alias=self.run/'E1/receipt_alias.json';alias.parent.mkdir()
        os.link(self.run/'fixed_spend.receipt.json',alias)
        with self.assertRaisesRegex(RuntimeError,'hardlink|alias'):
            self.add('E1_analysis',alias)
        self.assertFalse((self.run/'fixed_spend.000001.json').exists())

    def test_reserve_cannot_be_a_category_even_without_any_credit(self):
        self.scopes['illegal']={'pool':'hard_free_reserve','prefixes':['source/'],'exact_paths':[]}
        with self.assertRaisesRegex(RuntimeError,'reserve'):
            self.read()

    def test_pool_overspend_is_rejected_before_any_snapshot_mutation(self):
        self.allowances['source_and_model_bundle']=5
        path=self.file('source/model.bin',b'123456')
        with self.assertRaisesRegex(RuntimeError,'overspent'):
            self.add('source_bundle',path)
        self.assertFalse((self.run/'fixed_spend.json').exists())

    def test_second_add_overspend_keeps_the_previous_graph_intact(self):
        self.allowances['source_and_model_bundle']=10
        first=self.file('source/one',b'a'*6);second=self.file('source/two',b'b'*5)
        self.add('source_bundle',first);prior=self.read()['ledger_sha256']
        with self.assertRaisesRegex(RuntimeError,'overspent'):
            self.add('source_bundle',second)
        self.assertEqual(self.read()['ledger_sha256'],prior)
        self.assertEqual(self.read()['total_fixed_spent_bytes'],6)

    def test_other_run_same_content_is_not_eligible(self):
        path=self.base/'another-run.bin';path.write_bytes(b'known bytes')
        with self.assertRaisesRegex(RuntimeError,'another run'):
            self.add('source_bundle',path)

    def test_wrong_expected_or_changed_manifest_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'manifest'):
            self.read(manifest_sha256='0'*64)
        self.add('source_bundle',self.file('source/model.bin'))
        (self.run/'manifest.json').write_text('{"fixture":456}')
        with self.assertRaisesRegex(RuntimeError,'manifest'):
            self.read()

    def test_copied_ledger_is_bound_to_its_original_run(self):
        self.add('source_bundle',self.file('source/model.bin'))
        another=self.base/'copied-run';another.mkdir();(another/'manifest.json').write_bytes((self.run/'manifest.json').read_bytes())
        for source in self.run.glob('fixed_spend.*'):(another/source.name).write_bytes(source.read_bytes())
        with self.assertRaisesRegex(RuntimeError,'run/manifest'):
            read_fixed_spend(another,manifest_sha256=self.manifest_sha,allowances=self.allowances,scopes=self.scopes)

    def test_manifest_symlink_to_another_run_is_not_owned_frozen_identity(self):
        saved=self.base/'another-manifest.json';saved.write_bytes((self.run/'manifest.json').read_bytes())
        (self.run/'manifest.json').unlink();(self.run/'manifest.json').symlink_to(saved)
        with self.assertRaisesRegex(RuntimeError,'manifest'):
            self.read()

    def test_files_must_match_exact_or_prefix_scopes(self):
        allowed=self.file('report/summary.txt');self.add('explicit_report',allowed)
        with self.assertRaisesRegex(RuntimeError,'scope'):
            self.add('explicit_report',self.file('report/another.txt'))
        with self.assertRaisesRegex(RuntimeError,'scope'):
            self.add('source_bundle',self.file('source-neighbor/model.bin'))

    def test_symlink_alias_and_scope_escape_are_rejected(self):
        original=self.file('source/original');link=self.run/'source/alias';link.symlink_to(original)
        with self.assertRaisesRegex(RuntimeError,'alias|symlink'):
            self.add('source_bundle',link)
        with self.assertRaisesRegex(RuntimeError,'noncanonical'):
            self.add('source_bundle',files_sha256={'source/../manifest.json':self.manifest_sha})

    def test_removed_and_resized_files_stop_both_read_modes(self):
        path=self.file('source/model.bin');self.add('source_bundle',path);path.write_bytes(b'larger altered bytes')
        for deep in (False,True):
            with self.assertRaisesRegex(RuntimeError,'size'):
                self.read(deep=deep)
        path.unlink()
        for deep in (False,True):
            with self.assertRaises(RuntimeError):self.read(deep=deep)

    def test_light_read_scope_is_explicit_and_deep_catches_same_size_changes(self):
        path=self.file('source/model.bin',b'original');self.add('source_bundle',path);path.write_bytes(b'modified')
        result=self.read(deep=False)
        self.assertIn('existence_size',result['verification'])
        with self.assertRaisesRegex(RuntimeError,'SHA changed'):
            self.read(deep=True)

    def test_new_add_always_hashes_even_when_existing_graph_is_light(self):
        path=self.file('source/model.bin')
        with self.assertRaisesRegex(RuntimeError,'SHA changed'):
            self.add('source_bundle',files_sha256={str(path):'0'*64},deep=False)

    def test_original_allowances_and_scope_sets_are_immutable(self):
        self.add('source_bundle',self.file('source/model.bin'))
        changed=deepcopy(self.allowances);changed['source_and_model_bundle']+=1
        with self.assertRaisesRegex(RuntimeError,'contract'):
            self.read(allowances=changed)
        changed=deepcopy(self.scopes);changed['source_bundle']['prefixes'].append('other/')
        with self.assertRaisesRegex(RuntimeError,'contract'):
            self.read(scopes=changed)

    def test_scope_path_order_is_a_set_not_a_different_policy(self):
        self.add('cell_operationals',self.file('E2/one/receipt.json'))
        changed=deepcopy(self.scopes);changed['cell_operationals']['prefixes'].reverse()
        self.assertEqual(self.read(scopes=changed)['total_fixed_spent_bytes'],11)

    def test_ledger_payload_and_external_receipt_are_each_verified_in_light_mode(self):
        self.add('source_bundle',self.file('source/model.bin'))
        path=self.run/'fixed_spend.json';path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(RuntimeError,'external ledger SHA'):
            self.read(deep=False)

    def test_missing_external_receipt_fails_closed_without_repair(self):
        self.add('source_bundle',self.file('source/model.bin'));receipt=self.run/'fixed_spend.receipt.json';receipt.unlink()
        with self.assertRaisesRegex(RuntimeError,'incomplete'):
            self.read()
        self.assertFalse(receipt.exists())

    def test_crash_between_payload_and_external_receipt_retains_partial_prefix(self):
        path=self.file('source/model.bin');original=budget._exclusive_json
        def interrupted(target,value,**kwargs):
            if str(target).endswith('.receipt.json'):raise OSError('injected receipt write failure')
            return original(target,value,**kwargs)
        with patch.object(budget,'_exclusive_json',side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError,'injected'):
                self.add('source_bundle',path)
        self.assertTrue((self.run/'fixed_spend.json').is_file())
        self.assertTrue(path.is_file())
        with self.assertRaisesRegex(RuntimeError,'incomplete'):
            self.read()

    def test_revision_gap_or_unknown_partial_name_fails_closed(self):
        self.add('source_bundle',self.file('source/one'))
        (self.run/'fixed_spend.000002.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'gap'):
            self.read()

    def test_unknown_partial_snapshot_filename_is_not_ignored(self):
        (self.run/'fixed_spend.json.next').write_text('partial write')
        with self.assertRaisesRegex(RuntimeError,'partial fixed-spend'):
            self.read()

    def test_wrong_previous_chain_sha_is_rejected_even_with_updated_outer_sha(self):
        self.add('source_bundle',self.file('source/one'))
        self.add('source_bundle',self.file('source/two'))
        self.rewrite_snapshot(1,lambda value:value.update(previous_snapshot_sha256='0'*64))
        with self.assertRaisesRegex(RuntimeError,'chain'):
            self.read()

    def test_forged_entry_pool_and_counter_fields_are_rejected(self):
        self.add('source_bundle',self.file('source/one'))
        self.rewrite_snapshot(0,lambda value:value['entries'][0].update(pool='hard_free_reserve'))
        with self.assertRaisesRegex(RuntimeError,'category/pool'):
            self.read()

    def test_arbitrary_counter_cannot_replace_a_file_receipt(self):
        with self.assertRaisesRegex(RuntimeError,'path/SHA'):
            self.add('source_bundle',files_sha256=123)
        path=self.file('source/one')
        with self.assertRaisesRegex(RuntimeError,'receipt'):
            self.add('source_bundle',files_sha256={str(path):{'sha256':self.sha(path),'size_bytes':999}})

    def test_ledger_and_receipt_overhead_never_credit_themselves(self):
        self.add('source_bundle',self.file('source/model.bin'))
        with self.assertRaisesRegex(RuntimeError,'credit itself'):
            self.add('ledger_self_forbidden',self.run/'fixed_spend.json')
        self.assertEqual(self.read()['spent_bytes_by_pool']['working_disk_scratch'],0)

    def test_duplicate_json_keys_are_not_silently_accepted(self):
        self.add('source_bundle',self.file('source/model.bin'))
        path=self.run/'fixed_spend.receipt.json';text=path.read_text()
        text=text.replace('"schema":', '"schema":"duplicate", "schema":',1);path.write_text(text)
        with self.assertRaisesRegex(RuntimeError,'duplicate JSON'):
            self.read()

    def test_prewrite_callback_gets_exact_serialized_utf8_bytes_for_both_files(self):
        path=self.file('source/한글파일.bin')
        observed=[]
        self.add('source_bundle',path,before_write_bytes=lambda count:observed.append(count))
        self.assertEqual(observed,[(self.run/'fixed_spend.json').stat().st_size,
            (self.run/'fixed_spend.receipt.json').stat().st_size])
        self.assertEqual(self.read()['total_fixed_spent_bytes'],path.stat().st_size)

    def test_first_prewrite_resource_denial_preserves_sources_without_a_ledger(self):
        path=self.file('source/model.bin')
        with self.assertRaisesRegex(RuntimeError,'resource admission'):
            self.add('source_bundle',path,before_write_bytes=lambda count:False)
        self.assertEqual(list(self.run.glob('fixed_spend.*')),[])
        self.assertTrue(path.is_file())

    def test_second_prewrite_denial_retains_a_fail_closed_partial_revision(self):
        self.add('source_bundle',self.file('source/one'))
        first=(self.run/'fixed_spend.json').read_bytes()
        seen=[]
        def admit(count):
            seen.append(count)
            return len(seen)==1
        with self.assertRaisesRegex(RuntimeError,'resource admission'):
            self.add('source_bundle',self.file('source/two'),before_write_bytes=admit)
        self.assertEqual((self.run/'fixed_spend.json').read_bytes(),first)
        self.assertTrue((self.run/'fixed_spend.000001.json').is_file())
        self.assertFalse((self.run/'fixed_spend.000001.receipt.json').exists())
        with self.assertRaisesRegex(RuntimeError,'incomplete'):
            self.read()

    def test_stop_dictionary_is_not_resource_approval(self):
        with self.assertRaisesRegex(RuntimeError,'resource admission'):
            self.add('source_bundle',self.file('source/one'),
                before_write_bytes=lambda count:{'status':'STOP_PRESERVE_PREFIX'})

    def test_file_counter_extra_field_is_rejected_after_outer_sha_reseal(self):
        self.add('source_bundle',self.file('source/one'))
        self.rewrite_snapshot(0,lambda value:value['entries'][0].update(credit_bytes=999))
        with self.assertRaisesRegex(RuntimeError,'arbitrary counters'):
            self.read()


if __name__ == '__main__':unittest.main()
