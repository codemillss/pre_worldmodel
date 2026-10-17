"""Synthetic restart fixtures; no real P5 outcomes, model loads or physics."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from push_align.final_manifest import (
    FAMILY_RECIPE_VERSION, expected_E1, expected_E2, family_records, freeze_manifest,
    frozen_source_contract, prior_initial_inventory,
)
from push_align.final_recovery_manifest import (
    PROTOCOL_VERSION, SELECTION_RULE, USER_AUTHORIZATION, freeze_recovery_manifest,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True))


class FinalRecoveryManifestTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='p5-recovery-fixture-', dir=self.ROOT / 'tmp')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        actual_contract = frozen_source_contract(self.ROOT)
        for relative in actual_contract['source_sha256']:
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.ROOT / relative, target)
        self.contract = frozen_source_contract(self.root)
        self.parent_run = self.root / 'runs/p5-final-fixture-parent'
        self.run = self.root / 'runs/p5-final-fixture-recovery'
        self.parent_run.mkdir(parents=True)
        self.run.mkdir(parents=True)
        # The synthetic legacy catalog has 146 identities, as the real parent,
        # but unrelated coordinates and family seeds, without a simulator reset.
        self.old_manifest = self.root / 'runs/legacy-fixture/manifest.json'
        save(self.old_manifest, {'resets': [dict(object_x=10. + i, object_y=0.,
            object_yaw=0., pusher_x=9.9 + i, pusher_y=0.) for i in range(146)]})
        self.runtime = self.root / 'src/frozen_fixture_runtime.py'
        self.runtime.parent.mkdir(parents=True, exist_ok=True)
        self.runtime.write_text('# Original frozen runtime, never modified.\n')
        self.capacity = {'status': 'FIT_ESTIMATE_ONLY', 'free_bytes': 4 * 1024**3,
            'required_bytes': 3 * 1024**3, 'headroom_bytes': 1024**3}
        prior = prior_initial_inventory(self.root, exclude_final_run=self.parent_run)
        families, application = family_records(self.contract['physical_config'], seed=123,
            new_family_seed=456, initial_id_prefix='fixture', forbidden_identities=prior['canonical_identities'])
        parent = deepcopy(self.contract)
        parent.update(families=families, application_family=application,
            E1=expected_E1(families), E2=expected_E2(families),
            statistics={'replicates': 2000, 'seed': 7711, 'unit': 'family_within_stratum_seed_mean_first'},
            prediction_evaluation={'horizons': [1, 5, 10], 'source': 'all_E2_G_raw_fixed_applied_actions'},
            failure_policy={'automatic_retries': 0, 'replacement': 'none'},
            replay_selection_rule='stratum_and_termination_union', prior_initial_inventory=prior,
            family_generation={'recipe_version': FAMILY_RECIPE_VERSION, 'seed': 123,
                'application_seed': 456, 'initial_id_prefix': 'fixture'},
            candidate_generation={'base_seed': 1234, 'E2_stream_id': 0, 'E1_stream_id': 1},
            minimum_meaningful_difference=self.contract['profile']['final_review']['minimum_meaningful_improvement'],
            visualization_selection='smallest family', visualization={'fps': 5},
            personal_understanding='NOT_ASSESSED', actual_final_outcomes_used_for_selection=False,
            source_archives={}, source_bundle_map={},
            runtime_source_sha256={'src/frozen_fixture_runtime.py': digest(self.runtime)})
        parent['source_sha256'].update(parent['runtime_source_sha256'])
        self.admission(parent, self.parent_run)
        receipt = freeze_manifest(self.parent_run / 'manifest.json', parent, integration_pass=True,
            resource_capacity=self.capacity, source_root=self.root)
        save(self.parent_run / 'freeze_receipt.json', receipt)
        self.parent = json.loads((self.parent_run / 'manifest.json').read_text())
        # A genuine additional manifest under the retained parent is inventoried.
        save(self.parent_run / 'E2/fixture/G/manifest.json', {'initial_spec': families[0]['initial_spec']})
        self.document = deepcopy(parent)
        self.document['infrastructure_recovery'] = {
            'parent_run': str(self.parent_run.relative_to(self.root)),
            'parent_manifest_sha256': digest(self.parent_run / 'manifest.json'),
            'parent_freeze_receipt_sha256': digest(self.parent_run / 'freeze_receipt.json'),
            'user_authorization': USER_AUTHORIZATION, 'protocol_version': PROTOCOL_VERSION,
            'selection_rule': SELECTION_RULE, 'parent_outcomes_observed': True}
        self.document['prior_initial_inventory'] = prior_initial_inventory(self.root, exclude_final_run=self.run)
        for name in ('manifest.json', 'freeze_receipt.json'):
            p = self.parent_run / name
            self.document['source_sha256'][str(p.relative_to(self.root))] = digest(p)
        new_runtime = self.root / 'src/repaired_fixture_recorder.py'
        new_runtime.write_text('# Additive recording-only fixture.\n')
        self.document['runtime_source_sha256']['src/repaired_fixture_recorder.py'] = digest(new_runtime)
        self.document['source_sha256'].update(self.document['runtime_source_sha256'])
        for rel, sha in self.document['runtime_source_sha256'].items():
            target = self.run / 'source' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.root / rel, target)
            archived = str(target.relative_to(self.root))
            self.document['source_sha256'][archived] = sha
            self.document['source_bundle_map'][rel] = archived
        self.admission(self.document, self.run)

    def admission(self, document, run):
        for name, value in (('resource', {'capacity': self.capacity}),
                ('integration', {'status': 'PASS', 'checks': {'roundtrip': True, 'replay': True}})):
            p = run / (name + '.json')
            save(p, value)
            relative = str(p.relative_to(self.root))
            document['source_sha256'][relative] = digest(p)
            document[name + '_receipt'] = {'path': relative, 'sha256': digest(p)}

    def freeze(self, document=None, **kwargs):
        return freeze_recovery_manifest(self.run / 'manifest.json', self.document if document is None else document,
            integration_pass=kwargs.get('integration_pass', True),
            resource_capacity=kwargs.get('resource_capacity', self.capacity), source_root=self.root)

    def reject(self, document):
        with self.assertRaises(ValueError):
            self.freeze(document)
        self.assertFalse((self.run / 'manifest.json').exists())

    def test_same_population_freezes_only_new_run_outcomes_and_preserves_parent(self):
        before = {str(p): digest(p) for p in self.parent_run.rglob('*') if p.is_file()}
        receipt = self.freeze()
        saved = json.loads((self.run / 'manifest.json').read_text())
        self.assertEqual(receipt['sha256'], digest(self.run / 'manifest.json'))
        self.assertEqual(receipt['bytes'], (self.run / 'manifest.json').stat().st_size)
        self.assertEqual(saved['schema'], 'p5-frozen-final-manifest-v1')
        self.assertEqual(saved['status'], 'FROZEN_BEFORE_FINAL_OUTCOMES')
        self.assertEqual(saved['recovery_metadata']['freeze_scope'], 'before_new_run_outcomes_only')
        self.assertTrue(saved['recovery_metadata']['parent_outcomes_observed'])
        self.assertEqual(len(saved['recovery_metadata']['original_prior_canonical_identities']), 146)
        self.assertEqual(len(saved['recovery_metadata']['retained_parent_canonical_identities']), 13)
        self.assertEqual(saved['recovery_metadata']['new_run_prior_canonical_count'], 159)
        self.assertEqual(saved['prior_initial_inventory']['source_manifest_count'],
            self.parent['prior_initial_inventory']['source_manifest_count'] + 2)
        self.assertEqual(before, {str(p): digest(p) for p in self.parent_run.rglob('*') if p.is_file()})
        with self.assertRaises(FileExistsError):
            self.freeze()

    def test_authorization_parent_and_restart_rule_are_exact(self):
        original = self.document
        for field, value in [('user_authorization', 'yes'), ('protocol_version', 'old'),
                ('selection_rule', 'best run'), ('parent_outcomes_observed', False),
                ('parent_run', '../outside'), ('parent_manifest_sha256', '0' * 64),
                ('parent_freeze_receipt_sha256', '0' * 64)]:
            with self.subTest(field=field):
                m = deepcopy(original)
                m['infrastructure_recovery'][field] = value
                self.reject(m)
        m = deepcopy(original)
        m['infrastructure_recovery']['extra'] = True
        self.reject(m)
        m = deepcopy(original)
        del m['infrastructure_recovery']
        self.reject(m)

    def test_every_scientific_declaration_must_be_identical(self):
        for field in ('families', 'application_family', 'E2', 'E1', 'model_identities',
                'normalizer_identity', 'physical_config', 'planner_config', 'profile',
                'family_generation', 'candidate_generation', 'statistics', 'prediction_evaluation',
                'failure_policy', 'replay_selection_rule', 'visualization_selection', 'visualization',
                'minimum_meaningful_difference', 'actual_final_outcomes_used_for_selection'):
            with self.subTest(field=field):
                m = deepcopy(self.document)
                m[field] = None
                self.reject(m)
        m = deepcopy(self.document)
        m['hidden_new_scientific_condition'] = True
        self.reject(m)

    def test_current_catalog_cannot_hide_old_poses_parent_or_other_source(self):
        for kind in ('canonical', 'parent_pose', 'old_pose', 'source', 'source_sha', 'unresolved', 'exclusion'):
            with self.subTest(kind=kind):
                m = deepcopy(self.document)
                p = m['prior_initial_inventory']
                if kind == 'canonical': p['canonical_identities'].pop()
                elif kind == 'parent_pose': p['recognized_initials'].pop()
                elif kind == 'old_pose': p['recognized_initials'].pop(0)
                elif kind == 'source': p['source_manifests'].pop(next(iter(p['source_manifests'])))
                elif kind == 'source_sha': p['source_manifests'][next(iter(p['source_manifests']))] = '0' * 64
                elif kind == 'unresolved': p['unresolved'] = [{'reason': 'unknown'}]
                else: p['excluded_final_run'] = str(self.parent_run.relative_to(self.root))
                self.reject(m)
        save(self.root / 'runs/new-development/manifest.json', {'initial_spec': self.document['families'][0]['initial_spec']})
        m = deepcopy(self.document)
        m['prior_initial_inventory'] = prior_initial_inventory(self.root, exclude_final_run=self.run)
        self.reject(m)

    def test_original_manifest_bytes_cannot_be_replaced(self):
        self.old_manifest.write_text(self.old_manifest.read_text() + ' ')
        m = deepcopy(self.document)
        m['prior_initial_inventory'] = prior_initial_inventory(self.root, exclude_final_run=self.run)
        self.reject(m)

    def test_parent_receipt_must_match_real_path_hash_and_length(self):
        receipt_path = self.parent_run / 'freeze_receipt.json'
        original = json.loads(receipt_path.read_text())
        for key, value in (('path', str(self.run / 'manifest.json')), ('sha256', '0' * 64), ('bytes', 1), ('bytes', True)):
            with self.subTest(key=key, value=value):
                r = dict(original)
                r[key] = value
                save(receipt_path, r)
                m = deepcopy(self.document)
                sha = digest(receipt_path)
                m['infrastructure_recovery']['parent_freeze_receipt_sha256'] = sha
                m['source_sha256'][str(receipt_path.relative_to(self.root))] = sha
                self.reject(m)
        save(receipt_path, original)

    def test_parent_manifest_and_every_frozen_source_are_independently_verified(self):
        self.runtime.write_text('# modified parent source\n')
        # Even rebinding the new manifest to changed bytes cannot rewrite parent.
        m = deepcopy(self.document)
        m['runtime_source_sha256']['src/frozen_fixture_runtime.py'] = digest(self.runtime)
        m['source_sha256']['src/frozen_fixture_runtime.py'] = digest(self.runtime)
        self.reject(m)

    def test_source_and_runtime_bindings_cannot_be_removed_or_retargeted(self):
        for kind in ('historical', 'parent_manifest', 'parent_receipt', 'runtime_remove', 'runtime_hash', 'new_runtime_type', 'escape'):
            with self.subTest(kind=kind):
                m = deepcopy(self.document)
                if kind == 'historical': m['source_sha256'].pop(next(iter(self.contract['source_sha256'])))
                elif kind == 'parent_manifest': m['source_sha256'].pop(str((self.parent_run / 'manifest.json').relative_to(self.root)))
                elif kind == 'parent_receipt': m['source_sha256'].pop(str((self.parent_run / 'freeze_receipt.json').relative_to(self.root)))
                elif kind == 'runtime_remove': m['runtime_source_sha256'].pop('src/frozen_fixture_runtime.py')
                elif kind == 'runtime_hash': m['runtime_source_sha256']['src/frozen_fixture_runtime.py'] = '0' * 64
                elif kind == 'new_runtime_type':
                    p = self.root / 'configs/new-science.json'
                    save(p, {})
                    m['runtime_source_sha256']['configs/new-science.json'] = digest(p)
                    m['source_sha256']['configs/new-science.json'] = digest(p)
                else: m['source_sha256']['../outside'] = '0' * 64
                self.reject(m)

    def test_snapshot_mapping_and_snapshot_actual_bytes_are_required(self):
        m = deepcopy(self.document)
        m['source_bundle_map']['src/frozen_fixture_runtime.py'] = 'src/frozen_fixture_runtime.py'
        self.reject(m)
        m = deepcopy(self.document)
        m['source_bundle_map'] = {}
        self.reject(m)
        m = deepcopy(self.document)
        m['source_bundle_map'].pop('src/repaired_fixture_recorder.py')
        self.reject(m)
        snapshot = self.root / self.document['source_bundle_map']['src/repaired_fixture_recorder.py']
        snapshot.write_text('# changed snapshot\n')
        m = deepcopy(self.document)
        m['source_sha256'][str(snapshot.relative_to(self.root))] = digest(snapshot)
        self.reject(m)

    def test_historical_document_archives_keep_the_original_freeze_contract(self):
        m = deepcopy(self.document)
        original = next(p for p in self.contract['source_sha256'] if p.startswith('docs/'))
        target = self.run / 'source' / original
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.root / original, target)
        archived = str(target.relative_to(self.root))
        m['source_archives'] = {original: archived}
        m['source_bundle_map'][original] = archived
        m['source_sha256'][archived] = m['source_sha256'].pop(original)
        self.freeze(m)

    def test_admission_gates_capacity_and_complete_integration_remain_required(self):
        for cap in (None, {'status': 'HOLD'}, {**self.capacity, 'headroom_bytes': 0},
                {**self.capacity, 'free_bytes': True}, {**self.capacity, 'required_bytes': 1},
                {**self.capacity, 'free_bytes': 1}):
            with self.subTest(capacity=cap):
                with self.assertRaises(ValueError):
                    self.freeze(resource_capacity=cap)
        with self.assertRaises(ValueError):
            self.freeze(integration_pass=False)
        path = self.root / self.document['integration_receipt']['path']
        for saved in ({'status': 'PASS', 'checks': {}}, {'status': 'PASS', 'checks': {'roundtrip': 1}},
                {'status': 'HOLD', 'checks': {'roundtrip': True}}):
            save(path, saved)
            m = deepcopy(self.document)
            sha = digest(path)
            m['source_sha256'][str(path.relative_to(self.root))] = sha
            m['integration_receipt']['sha256'] = sha
            self.reject(m)


if __name__ == '__main__':
    unittest.main()
