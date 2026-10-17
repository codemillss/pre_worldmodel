"""Synthetic historical-archive and compatibility checks; actual runs are untouched."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def load(name):
    path=ROOT/'tools'/name
    spec=importlib.util.spec_from_file_location(name.removesuffix('.py'),path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
V1=load('p5_report_resource_ledger_v1.py');V2=load('p5_report_resource_ledger_v2.py')


class HistoricalArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.args=V2.fixture(self.root)
        self.parent=self.root/'reports/old_parent.json';self.figure=self.root/'reports/old_figure.svg'
        self.untrusted=self.root/'reports/untrusted.svg';self.raw=self.args[0]/'raw-never-open.bin'
        self.figure.write_text('<svg>historical</svg>');self.untrusted.write_text('<svg>new</svg>');self.raw.write_bytes(b'raw sentinel')
        self.parent.write_bytes(V2.encode({'source_before_sha256':{
            str(self.figure.relative_to(self.root)):V2.digest(self.figure),
            str(self.raw.relative_to(self.root)):'0'*64},
            'ignored_unapproved_map':{str(self.untrusted.relative_to(self.root)):V2.digest(self.untrusted)}}))
        self.anchors=V2.fixture_history(self.root,self.args)
        manifest=self.anchors['historical_manifest'];value=V2.read(manifest)
        value['references']=[{'id':1,'path':str(self.parent.relative_to(self.root)),'sha256':V2.digest(self.parent)}]
        value['chapters']=[{'path':'docs/out-of-archive.md','sha256':'0'*64,'historical':False}]
        manifest.write_bytes(V2.encode(value));self.anchors['historical_manifest_sha256']=V2.digest(manifest)
        observation=self.anchors['baseline_observation'];value=V2.read(observation)
        value['checks']['existing_473page_artifacts']['manifest_sha256']=V2.digest(manifest)
        observation.write_bytes(V2.encode(value));self.anchors['baseline_observation_sha256']=V2.digest(observation)
    def build(self):
        run,out,extra,old,oldsha=self.args
        return V2.build(run,out,[extra],old,oldsha,free_bytes=8*V2.HARD_RESERVE,**self.anchors)
    def test_explicit_transitive_history_only_inside_archive(self):
        original_digest=V2.digest
        def guarded(path):
            self.assertNotEqual(Path(path),self.raw,'out-of-archive raw must not be opened')
            return original_digest(path)
        with patch.object(V2,'digest',guarded):value,payload=self.build()
        self.assertTrue(value['accepted']);self.assertEqual(value['schema'],'p5-final-report-resource-ledger-v2')
        old={r['path']:r for r in value['excluded_anchored_historical_archive']}
        new={r['path'] for r in value['files']}
        self.assertIn(str(self.parent),old);self.assertIn(str(self.figure),old)
        self.assertNotIn(str(self.figure),new);self.assertIn(str(self.untrusted),new)
        self.assertEqual(old[str(self.figure)]['ancestry'][-1]['sha256'],V2.digest(self.parent))
        self.assertIn('source_before_sha256',old[str(self.figure)]['ancestry'][-1]['field'])
        self.assertGreater(value['historical_archive_provenance']['skipped_outside_archive_target_references'],0)
        self.assertEqual(value['ledger_self']['size_bytes'],len(payload))
    def test_changed_historical_figure_is_new_and_mtime_is_not_evidence(self):
        self.figure.write_text('<svg>changed</svg>')
        value,_=self.build();new={r['path'] for r in value['files']}
        self.assertIn(str(self.figure),new)
        self.assertIn(str(self.figure),value['historical_archive_provenance']['changed_historical_candidates'])
        self.assertNotIn(str(self.figure),{r['path'] for r in value['excluded_anchored_historical_archive']})
    def test_changed_json_cannot_authorize_new_child(self):
        self.parent.write_bytes(V2.encode({'source_after_sha256':{str(self.untrusted):V2.digest(self.untrusted)}}))
        value,_=self.build();new={r['path'] for r in value['files']}
        self.assertTrue({str(self.parent),str(self.figure),str(self.untrusted)}<=new)
    def test_historical_manifest_anchor_change_rejected(self):
        self.anchors['historical_manifest'].write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError,'anchor'):self.build()
    def test_earlier_observation_anchor_change_rejected(self):
        self.anchors['baseline_observation'].write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError,'anchor'):self.build()
    def test_unhashed_directory_reference_is_skipped_without_exemption(self):
        manifest=self.anchors['historical_manifest'];value=V2.read(manifest)
        value['references'].extend([{'id':475,'path':'runs/p3-old/full_resource','sha256':None},
            {'id':476,'path':str(self.untrusted.relative_to(self.root)),'sha256':None}])
        manifest.write_bytes(V2.encode(value));self.anchors['historical_manifest_sha256']=V2.digest(manifest)
        observation=self.anchors['baseline_observation'];prior=V2.read(observation)
        prior['checks']['existing_473page_artifacts']['manifest_sha256']=V2.digest(manifest)
        observation.write_bytes(V2.encode(prior));self.anchors['baseline_observation_sha256']=V2.digest(observation)
        actual,_=self.build()
        self.assertEqual(actual['historical_archive_provenance']['skipped_unhashed_reference_count'],2)
        self.assertIn(str(self.untrusted),{r['path'] for r in actual['files']})
        self.assertFalse((self.root/'runs/p3-old/full_resource').exists())
        run,out,_,old,oldsha=self.args
        for forbidden in (self.root,self.root/'runs',self.raw):
            with self.subTest(forbidden=forbidden):
                with self.assertRaises(ValueError):V2.build(run,out,[forbidden],old,oldsha,**self.anchors)
    def test_limits_scanner_and_supplement_only_intended_ast_changes(self):
        for name in ('REPORT_LIMIT','SCRATCH_LIMIT','TAIL','HARD_RESERVE'):
            self.assertEqual(getattr(V1,name),getattr(V2,name))
        def tree(name):return ast.parse((ROOT/'tools'/name).read_text())
        one,two=tree('p5_report_resource_ledger_v1.py'),tree('p5_report_resource_ledger_v2.py')
        scanner=lambda t:next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='Scan')
        self.assertEqual(ast.dump(scanner(one),include_attributes=False),ast.dump(scanner(two),include_attributes=False))
        before=tree('p5_operational_resource_supplement_v1.py');after=tree('p5_operational_resource_supplement_v2.py')
        class Adapt(ast.NodeTransformer):
            changes=0
            def visit_Compare(self,node):
                self.generic_visit(node)
                if len(node.ops)==1 and isinstance(node.ops[0],ast.Eq) and len(node.comparators)==1 \
                        and isinstance(node.comparators[0],ast.Constant) and node.comparators[0].value=='p5-final-report-resource-ledger-v1':
                    self.changes+=1;node.ops=[ast.In()];node.comparators=[ast.Tuple(elts=[ast.Constant('p5-final-report-resource-ledger-v1'),ast.Constant('p5-final-report-resource-ledger-v2')],ctx=ast.Load())]
                return node
            def visit_Constant(self,node):
                if node.value=='p5-idle-operational-resource-supplement-v1':return ast.Constant('p5-idle-operational-resource-supplement-v2')
                return node
        expected=deepcopy(before);expected.body[0]=deepcopy(after.body[0]);transform=Adapt();expected=transform.visit(expected)
        self.assertEqual(transform.changes,1)
        self.assertEqual(ast.dump(expected,include_attributes=False),ast.dump(after,include_attributes=False))


if __name__=='__main__':unittest.main()
