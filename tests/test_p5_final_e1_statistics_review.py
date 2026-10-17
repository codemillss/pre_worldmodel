"""Read-only review fixtures; original576-prefix adoption is explicitly mocked."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from push_align.config import ROOT
from push_align.final_e1_statistics import analyze_E1
from push_align.final_e1_statistics_review import validate_e1_statistics_stage, E1StatisticsReviewError
from push_align.final_recording import file_sha256


class E1StatisticsReviewTests(unittest.TestCase):
    def setUp(self):
        from test_p5_final_e1_statistics import fixture
        self.temporary=tempfile.TemporaryDirectory(prefix='p5-e1-stats-review-',dir=ROOT/'tmp')
        self.addCleanup(self.temporary.cleanup)
        self.run=Path(self.temporary.name).resolve()
        self.folder=self.run/'e1_statistics'
        self.rows,self.specs=fixture()
        self.manifest={'schema':'p5-frozen-final-manifest-v1','status':'FROZEN_BEFORE_FINAL_OUTCOMES',
            'E1':self.specs,'statistics':{'replicates':2000,'seed':7711}}
        self.write(self.run/'manifest.json',self.manifest)
        self.write(self.run/'freeze_receipt.json',{'sha256':file_sha256(self.run/'manifest.json')})
        self.write(self.run/'E1/summary.json',{'status':'COMPLETE_RECORDED','synthetic_adoption_fixture_only':True})
        self.write(self.run/'E1/stage_receipt.json',{'synthetic':'original E1 seal'})
        self.write(self.run/'E1/comparison_ledger.json',self.rows)
        self.produce()
        self.adoption=patch('p5_final_e1_adoption.adopt_complete_e1',side_effect=self.adopted)
        self.spy=self.adoption.start()
        self.addCleanup(self.adoption.stop)

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,allow_nan=False))

    def adopted(self,run,manifest):
        return {'status':'ADOPT_COMPLETE_VERIFIED_E1',
            'completed_repeat_ids':[spec['snapshot_id']+f'/candidate-{candidate:02d}/repeat-{repeat}'
                for spec in self.specs for candidate in range(16) for repeat in range(3)],
            'summary_sha256':file_sha256(self.run/'E1/summary.json'),
            'stage_seal_path':str(self.run/'E1/stage_receipt.json'),
            'stage_seal_sha256':file_sha256(self.run/'E1/stage_receipt.json')}

    def sources(self):
        return {str(self.run/relative):file_sha256(self.run/relative) for relative in
            ('manifest.json','E1/summary.json','E1/stage_receipt.json','E1/comparison_ledger.json')}

    def produce(self):
        summary,arrays=analyze_E1(self.rows,self.specs)
        self.write(self.folder/'summary.json',summary)
        with (self.folder/'bootstrap.npz').open('wb') as stream:np.savez_compressed(stream,**arrays)
        self.reproduction()
        self.seal()

    def reproduction(self):
        self.write(self.folder/'reproduction.json',{'status':'PASS_EXACT_REGENERATION',
            'source_rows':84,'snapshots':12,'families':6,
            'summary_sha256':file_sha256(self.folder/'summary.json'),
            'bootstrap_sha256':file_sha256(self.folder/'bootstrap.npz')})

    def seal(self,*,sources=None):
        self.write(self.folder/'stage_receipt.json',{'schema':'p5-derived-stage-seal-v1','stage':'e1_statistics',
            'frozen_manifest_sha256':file_sha256(self.run/'manifest.json'),
            'files_sha256':{str(self.folder/name):file_sha256(self.folder/name)
                for name in ('summary.json','bootstrap.npz','reproduction.json')},
            'source_sha256':self.sources() if sources is None else sources})
        self.write(self.folder/'stage_receipt.sha256.json',{'sha256':file_sha256(self.folder/'stage_receipt.json')})

    @contextmanager
    def restore_all(self):
        files={path:path.read_bytes() for path in self.run.rglob('*') if path.is_file()}
        try:
            yield
        finally:
            for path in self.run.rglob('*'):
                if path.is_file() and path not in files:path.unlink()
            for path,payload in files.items():path.write_bytes(payload)

    def test_complete_stage_reloads_original_E1_and_is_read_only(self):
        before={str(path):file_sha256(path) for path in self.run.rglob('*') if path.is_file()}
        result=validate_e1_statistics_stage(self.run,self.manifest)
        self.spy.assert_called_once_with(self.run,self.manifest)
        self.assertEqual(result['status'],'PASS_E1_STATISTICS_EXACT_REGENERATION')
        self.assertEqual(result['verified_physical_repeat_ids'],576)
        self.assertEqual(result['verified_model_sets'],84)
        self.assertEqual(result['independent_families'],6)
        self.assertEqual(result['additional_model_forward'],0)
        self.assertEqual(result['additional_physics'],0)
        self.assertEqual(before,{str(path):file_sha256(path) for path in self.run.rglob('*') if path.is_file()})

    def test_missing_mandatory_outputs_receipts_or_original_sources_are_rejected(self):
        names=['e1_statistics/summary.json','e1_statistics/bootstrap.npz','e1_statistics/reproduction.json',
            'e1_statistics/stage_receipt.json','e1_statistics/stage_receipt.sha256.json','freeze_receipt.json',
            'E1/summary.json','E1/stage_receipt.json','E1/comparison_ledger.json']
        for name in names:
            with self.subTest(name=name),self.restore_all():
                (self.run/name).unlink()
                with self.assertRaises(E1StatisticsReviewError):validate_e1_statistics_stage(self.run,self.manifest)

    def test_changed_derived_or_original_bytes_fail_without_resealing(self):
        for name in ('e1_statistics/summary.json','e1_statistics/bootstrap.npz',
                     'e1_statistics/reproduction.json','E1/comparison_ledger.json'):
            with self.subTest(name=name),self.restore_all():
                path=self.run/name;path.write_bytes(path.read_bytes()+b' ')
                with self.assertRaises(E1StatisticsReviewError):validate_e1_statistics_stage(self.run,self.manifest)

    def test_wrong_summary_denominator_is_rejected_even_with_updated_valid_hashes(self):
        path=self.folder/'summary.json';summary=json.loads(path.read_text());summary['independent_family_count']=12
        self.write(path,summary);self.reproduction();self.seal()
        with self.assertRaisesRegex(E1StatisticsReviewError,'Summary does not exactly regenerate'):
            validate_e1_statistics_stage(self.run,self.manifest)

    def test_diagnostic_row_change_is_not_hidden_by_a_fresh_stage_source_hash(self):
        rows=deepcopy(self.rows);rows[0]['selection_loss']+=1.
        self.write(self.run/'E1/comparison_ledger.json',rows);self.seal()
        with self.assertRaisesRegex(E1StatisticsReviewError,'Summary does not exactly regenerate'):
            validate_e1_statistics_stage(self.run,self.manifest)

    def test_bad_diagnostic_population_never_regenerates_as_complete(self):
        rows=deepcopy(self.rows);rows[0]['counts']['candidate_count']=15
        self.write(self.run/'E1/comparison_ledger.json',rows);self.seal()
        with self.assertRaises(E1StatisticsReviewError):validate_e1_statistics_stage(self.run,self.manifest)

    def test_exact_four_original_source_paths_are_mandatory(self):
        for extra in (False,True):
            with self.subTest(extra=extra),self.restore_all():
                sources=self.sources()
                if extra:
                    self.write(self.run/'unrelated.json',{'unrelated':True})
                    sources[str(self.run/'unrelated.json')]=file_sha256(self.run/'unrelated.json')
                else:sources.pop(str(self.run/'E1/stage_receipt.json'))
                self.seal(sources=sources)
                with self.assertRaisesRegex(E1StatisticsReviewError,'Exact four original'):
                    validate_e1_statistics_stage(self.run,self.manifest)

    def test_exact_output_file_map_cannot_be_empty_or_add_an_undeclared_file(self):
        for extra in (False,True):
            with self.subTest(extra=extra),self.restore_all():
                path=self.folder/'stage_receipt.json';receipt=json.loads(path.read_text())
                if extra:
                    self.write(self.folder/'unrelated.json',{})
                    receipt['files_sha256'][str(self.folder/'unrelated.json')]=file_sha256(self.folder/'unrelated.json')
                else:receipt['files_sha256']={}
                self.write(path,receipt)
                self.write(self.folder/'stage_receipt.sha256.json',{'sha256':file_sha256(path)})
                with self.assertRaisesRegex(E1StatisticsReviewError,'Exact summary/bootstrap/reproduction'):
                    validate_e1_statistics_stage(self.run,self.manifest)

    def test_bootstrap_key_dtype_and_value_are_all_recomputed(self):
        for fault in ('key','dtype','value','object'):
            with self.subTest(fault=fault),self.restore_all():
                path=self.folder/'bootstrap.npz'
                with np.load(path,allow_pickle=False) as loaded:arrays={key:loaded[key].copy() for key in loaded.files}
                if fault=='key':arrays.pop('family_draw_indices')
                elif fault=='dtype':arrays['family_draw_indices']=arrays['family_draw_indices'].astype(np.float64)
                elif fault=='object':arrays['family_draw_indices']=np.array([{}],object)
                else:arrays['family_draw_indices'][0,0]=(arrays['family_draw_indices'][0,0]+1)%6
                with path.open('wb') as stream:np.savez_compressed(stream,**arrays)
                self.reproduction();self.seal()
                with self.assertRaises(E1StatisticsReviewError):validate_e1_statistics_stage(self.run,self.manifest)

    def test_reproduction_population_status_and_bound_hashes_are_not_cosmetic(self):
        for key,value in (('source_rows',12),('snapshots',6),('families',12),('families',True),
                          ('status','HOLD'),('summary_sha256','0'*64)):
            with self.subTest(key=key),self.restore_all():
                path=self.folder/'reproduction.json';receipt=json.loads(path.read_text());receipt[key]=value
                self.write(path,receipt);self.seal()
                with self.assertRaisesRegex(E1StatisticsReviewError,'Reproduction receipt'):
                    validate_e1_statistics_stage(self.run,self.manifest)

    def test_fixed2000_and7711_and_same_caller_manifest_are_required(self):
        altered=deepcopy(self.manifest);altered['statistics']['seed']=123
        with self.assertRaises(E1StatisticsReviewError):validate_e1_statistics_stage(self.run,altered)
        for key,value in (('replicates',20),('replicates',True),('seed',123)):
            with self.subTest(key=key),self.restore_all():
                altered=deepcopy(self.manifest);altered['statistics'][key]=value
                self.write(self.run/'manifest.json',altered)
                self.write(self.run/'freeze_receipt.json',{'sha256':file_sha256(self.run/'manifest.json')})
                self.seal()
                with self.assertRaisesRegex(E1StatisticsReviewError,'fixed2000'):
                    validate_e1_statistics_stage(self.run,altered)

    def test_original_E1_adoption_must_be_complete576_and_match_bound_files(self):
        good=self.adopted(self.run,self.manifest)
        wrong_ids={**good,'completed_repeat_ids':good['completed_repeat_ids'][:-1]}
        wrong_hash={**good,'stage_seal_sha256':'0'*64}
        for adopted in (None,{'status':'HOLD'},wrong_ids,wrong_hash):
            with self.subTest(adopted=adopted and adopted.get('status')):
                self.spy.side_effect=None;self.spy.return_value=adopted
                with self.assertRaises(E1StatisticsReviewError):validate_e1_statistics_stage(self.run,self.manifest)

    def test_source_change_during_adoption_cannot_pass_end_of_review_recheck(self):
        def change(run,m):
            adopted=self.adopted(run,m)
            path=self.run/'E1/comparison_ledger.json';path.write_bytes(path.read_bytes()+b' ')
            return adopted
        self.spy.side_effect=change
        with self.assertRaisesRegex(E1StatisticsReviewError,'Missing/changed evidence'):
            validate_e1_statistics_stage(self.run,self.manifest)

    def test_stage_receipt_replacement_during_adoption_is_not_hidden_by_a_new_outer_hash(self):
        def change(run,m):
            adopted=self.adopted(run,m)
            path=self.folder/'stage_receipt.json';path.write_bytes(path.read_bytes()+b' ')
            self.write(self.folder/'stage_receipt.sha256.json',{'sha256':file_sha256(path)})
            return adopted
        self.spy.side_effect=change
        with self.assertRaisesRegex(E1StatisticsReviewError,'Missing/changed evidence'):
            validate_e1_statistics_stage(self.run,self.manifest)


if __name__=='__main__':unittest.main()
