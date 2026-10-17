"""Hermetic analysis continuation tests; no real run/model/physics execution."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_p5_final_statistics_v2 import fixture
from push_align.final_manifest import family_records

MODULE_PATH=Path(__file__).resolve().parents[1]/'scripts/p5_analysis_recovery_v3.py'
spec=importlib.util.spec_from_file_location('analysis_recovery_v3_tested',MODULE_PATH)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class FakeOriginal:
    def __init__(self,root,run,manifest):
        self.ROOT=root;self.run=run;self.m=manifest;self.deep_calls=[];self.capacity=[];self.registrations=[];self.limit=16*1024**2
    def frozen(self,run):
        self.verify(self.m['source_sha256']);return deepcopy(self.m)
    def verify(self,bindings):
        for p,h in bindings.items():
            if module.sha(self.ROOT/p)!=h:raise RuntimeError('Frozen synthetic source changed')
    def verified_result(self,run,phase,cell_id,manifest,*,deep=True):
        if not deep:raise AssertionError('This fixture expects precisely one original deep pass')
        self.deep_calls.append((phase,cell_id))
        return module.read(run/'E2'/cell_id/'worker_result.json')
    def derived_pool_bytes(self,run,name):
        if name!='statistics':raise AssertionError('Wrong original budget pool')
        return 30+sum(p.stat().st_size for p in (run/'statistics').rglob('*') if p.is_file())
    def derived_limit(self,run,name):
        if name!='statistics':raise AssertionError('Wrong original budget limit')
        return self.limit
    def capacity_check(self,run,**kwargs):self.capacity.append(kwargs)
    def register_fixed_files(self,run,manifest,category,files):
        if category!='E2_statistics':raise AssertionError('Wrong original fixed category')
        self.registrations.append(dict(files))


class AnalysisRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='p5-analysis-v3-fixture-');self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name).resolve();self.run=self.root/'runs/p5-final-fixture';self.run.mkdir(parents=True)
        (self.root/'fixture-source.py').write_text('# synthetic immutable source\n')
        rows,expected=fixture()
        physical={'object':{'size_full_xyz_m':[.06,.03,.02]},'pusher':{'radius_m':.015},'reset':{'pusher_surface_gap_m':.006}}
        families,_=family_records(physical,seed=123,new_family_seed=456,initial_id_prefix='decimal-fixture')
        self.m={'source_sha256':{'fixture-source.py':module.sha(self.root/'fixture-source.py')},
            'runtime_source_sha256':{},'E2':expected,'families':families,
            'statistics':{'replicates':2000,'seed':7711,'unit':'family_within_stratum_seed_mean_first'}}
        write(self.run/'manifest.json',self.m);write(self.run/'freeze_receipt.json',{'sha256':module.sha(self.run/'manifest.json')})
        self.source={str(self.root/'fixture-source.py'):module.sha(self.root/'fixture-source.py')}
        for row,cell in zip(rows,expected):
            folder=self.run/'E2'/cell['cell_id'];raw=folder/'raw.bin';raw.parent.mkdir(parents=True)
            raw.write_bytes(b'fixture raw preserved')
            record={'files_sha256':{str(raw):module.sha(raw)},'source_bindings':self.source}
            write(folder/'manifest.json',record);write(folder/'review.json',{'fixture':'accepted'});write(folder/'component_budget_review.json',{'fixture':'within'})
            result={'phase':'E2','cell':cell,'row':row,'record_manifest':str(folder/'manifest.json'),
                'record_manifest_sha256':module.sha(folder/'manifest.json'),'review_sha256':module.sha(folder/'review.json'),
                'component_review_sha256':module.sha(folder/'component_budget_review.json')}
            write(folder/'worker_result.json',result);write(folder/'worker_result_receipt.json',{'sha256':module.sha(folder/'worker_result.json')})
        self.original=FakeOriginal(self.root,self.run,self.m)
        self.continuation=self.root/'continuation.json';write(self.continuation,{'scope':'analysis-only fixture'})

    def sources(self):return dict(self.m['source_sha256'])

    def test_full_original_96_review_once_then_fresh_sha_bound_reloads(self):
        verified=module.VerifiedRows(self.original,self.run,self.m)
        first=verified.initialize();back=verified.reload(list(reversed(self.m['E2'])))
        self.assertEqual(len(first),96);self.assertEqual(first,list(reversed(back)))
        self.assertEqual(len(self.original.deep_calls),96)
        self.assertEqual(verified.summary()['fresh_sealed_JSON_reloads'],2)
        self.assertEqual(len([p for p in verified.graph if p.endswith('fixture-source.py')]),1)
        self.assertFalse(verified.summary()['frozen_globals_or_functions_patched'])

    def test_any_raw_witness_source_or_receipt_change_blocks_reuse(self):
        verified=module.VerifiedRows(self.original,self.run,self.m);verified.initialize()
        raw=self.run/'E2'/self.m['E2'][0]['cell_id']/'raw.bin';old=raw.read_bytes();raw.write_bytes(b'X'+old[1:])
        with self.assertRaises(RuntimeError):verified.reload(self.m['E2'])
        self.assertEqual(len(self.original.deep_calls),96)

    def test_output_adapter_only_changes_statistics_path(self):
        output=self.run/module.OUTPUT_RELATIVE;view=module.StatisticsOutputView(self.run,output)
        self.assertEqual(view/'statistics',output)
        self.assertEqual(view/'E2',self.run/'E2')
        self.assertEqual(view/'manifest.json',self.run/'manifest.json')

    def test_writer_counts_old_and_new_in_same_pool_and_never_overwrites(self):
        old=self.run/'statistics/old-partial.json';write(old,{'old':True})
        writer=module.StatisticsWriter(self.original,self.run);writer.write('new.json',b'123')
        self.assertTrue(old.exists());self.assertEqual(self.original.capacity[0]['active_written_credit'],0)
        writer.write('other.json',b'4567');self.assertEqual(self.original.capacity[1]['active_written_credit'],3)
        with self.assertRaises(FileExistsError):writer.write('new.json',b'789')
        with self.assertRaises(RuntimeError):module.StatisticsWriter(self.original,self.run)
        self.original.limit=self.original.derived_pool_bytes(self.run,'statistics')
        with self.assertRaises(RuntimeError):writer.write('over-budget.json',b'x')
        self.assertFalse((writer.output/'over-budget.json').exists())

    def test_synthetic_full_statistics_preserves_old_files_and_original_numeric_review(self):
        old=self.run/'statistics/old-partial.json';write(old,{'old_partial':'retained'})
        baseline=module.tree_files(self.run/'E2');source=module.sha(self.root/'fixture-source.py');old_sha=module.sha(old)
        with patch.object(module,'analysis_sources',side_effect=self.sources):
            result=module.statistics(self.original,self.run,self.continuation)
            self.assertEqual(result['status'],'PASS_CANONICAL_STATISTICS_RECOVERY')
            receipt=module.read(self.run/module.OUTPUT_RELATIVE/'analysis_recovery_receipt.json')
            self.assertEqual(receipt['schema'],module.RECEIPT_SCHEMA)
            self.assertTrue(receipt['canonical_forward_exact']);self.assertTrue(receipt['bootstrap_exact_original_forward'])
            self.assertEqual(receipt['original_scientific_review']['status'],'PASS')
            self.assertEqual(receipt['original_scientific_review']['checks']['verified_rows'],96)
            self.assertEqual(receipt['validation_reuse']['deep_original_reviews'],96)
            self.assertEqual(receipt['validation_reuse']['fresh_sealed_JSON_reloads'],3)
            accepted,review=module.verify_recovery(self.original,self.run,self.m)
            self.assertEqual(review['status'],'PASS')
            self.assertEqual(len(self.original.deep_calls),96)
            self.assertEqual(receipt['validated_E2_evidence_sha256'],accepted['validated_E2_evidence_sha256'])
            changed=self.run/'E2'/self.m['E2'][0]['cell_id']/'review.json';changed.write_text('{}')
            with self.assertRaises(RuntimeError):module.verify_recovery(self.original,self.run,self.m)
        # Only our intentional corruption changed a source; the writer itself did not.
        self.assertEqual(module.sha(old),old_sha);self.assertEqual(module.sha(self.root/'fixture-source.py'),source)
        final_registration=self.original.registrations[-1]
        self.assertIn(str(old),final_registration)
        self.assertTrue(all(Path(p).is_relative_to(self.run/'statistics') for p in final_registration))
        for path,digest in baseline.items():
            if not path.endswith('/review.json') or Path(path).parent!=changed.parent:self.assertEqual(module.sha(path),digest)

    def test_no_worker_launcher_or_original_global_function_replacement(self):
        source=MODULE_PATH.read_text()
        self.assertNotIn('subprocess',source)
        self.assertNotIn('original.verified_result=',source)
        self.assertNotIn('cached_original',source)


if __name__=='__main__':unittest.main()
