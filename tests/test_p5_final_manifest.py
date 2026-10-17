import json
import hashlib
from copy import deepcopy
import shutil
from pathlib import Path
import tempfile
import unittest

from push_align.final_manifest import (canonical_initial, extract_prior_initials,
    family_records, expected_E2, expected_E1, replay_union, freeze_manifest, frozen_source_contract, prior_initial_inventory, FAMILY_RECIPE_VERSION)


PHYSICAL = {"object": {"size_full_xyz_m": [.06, .03, .02]},
            "pusher": {"radius_m": .015}, "reset": {"pusher_surface_gap_m": .006}}


class FinalManifestTests(unittest.TestCase):
    def family(self):
        # Fixture seeds are intentionally unrelated to actual final streams.
        return family_records(PHYSICAL, seed=123, new_family_seed=456, initial_id_prefix="fixture")

    def test_fixed_strata_sign_balance_and_disjoint_application(self):
        rows, application = self.family()
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row['canonical_identity'] for row in rows + [application]}), 13)
        for stratum in ('translation', 'rotation', 'offset'):
            part = [row for row in rows if row['stratum'] == stratum]
            self.assertEqual(len(part), 4)
            if stratum == 'rotation':
                self.assertEqual(sum(row['initial_spec']['object_yaw'] > 0 for row in part), 2)
            if stratum == 'offset':
                self.assertEqual(sum(row['tangent_offset_m'] > 0 for row in part), 2)
        self.assertEqual((rows, application), self.family())

    def test_collision_stops_without_replacement(self):
        rows, _ = self.family()
        with self.assertRaises(ValueError):
            family_records(PHYSICAL, seed=123, new_family_seed=456,
                           initial_id_prefix='fixture', forbidden_identities=[rows[0]['canonical_identity']])

    def test_canonical_identity_treats_rectangle_pi_and_signed_zero_as_equal(self):
        a = dict(object_x=0., object_y=-0., object_yaw=0., pusher_x=-.05, pusher_y=.01)
        b = dict(a, object_y=0., object_yaw=3.141592653589793)
        self.assertEqual(canonical_initial(a), canonical_initial(b))
        self.assertEqual(canonical_initial(dict(a,object_yaw=1.5707963267948966)),
                         canonical_initial(dict(a,object_yaw=-1.5707963267948966)))
        with self.assertRaises(ValueError): canonical_initial(dict(a, pusher_y=None))

    def test_prior_recursion_and_initial_observation_exclusion(self):
        rows, _ = self.family(); initial = rows[0]['initial_spec']
        values = {'episodes': [{'initial_spec': initial}], 'initial_state': [initial['object_x'], initial['object_y'], initial['object_yaw'],0.,0.,0.,initial['pusher_x'],initial['pusher_y'],0.,0.]}
        found = extract_prior_initials(values, source='fixture.json')
        self.assertEqual(len(found), 2)
        self.assertEqual({row['canonical_identity'] for row in found}, {rows[0]['canonical_identity']})

    def test_unresolved_prior_reset_is_detected_and_archived_config_can_resolve(self):
        reset=dict(object_x=0.,object_y=0.,object_yaw=0.,pusher_x=None,pusher_y=None)
        with self.assertRaises(ValueError):extract_prior_initials(reset,source='fixture')
        issues=[]
        self.assertEqual(extract_prior_initials(reset,source='fixture',unresolved=issues),[])
        self.assertEqual(len(issues),1)
        physical={**PHYSICAL,'reset':{**PHYSICAL['reset'],'initial_pusher_tangent_offset_m':0.}}
        found=extract_prior_initials(reset,source='fixture',physical=physical)
        self.assertAlmostEqual(found[0]['initial_spec']['pusher_x'],-.051)

    def test_E1_first_two_family_and_exact_ticks_no_post_outcome_choice(self):
        rows, _ = self.family(); e2, e1 = expected_E2(rows), expected_E1(list(reversed(rows)))
        self.assertEqual(len(e2), 96); self.assertEqual(len(e1), 12)
        self.assertTrue(all(row['family_id'].endswith(('00','01')) for row in e1))
        self.assertEqual({row['tick'] for row in e1}, {0,29})

    def test_duplicate_and_three_four_five_strata_are_rejected(self):
        rows,_=self.family();rows[0]['stratum']='offset'
        with self.assertRaises(ValueError):expected_E2(rows)
        rows,_=self.family();rows[0]['family_id']=rows[1]['family_id']
        with self.assertRaises(ValueError):expected_E2(rows)

    def results(self):
        rows, _ = self.family(); records = expected_E2(rows)
        for row in records:
            row.update(execution_status='completed', termination='timeout')
        return records

    def test_replay_union_includes_smallest_stratum_and_all_observed_terminals(self):
        rows = self.results()
        special = next(row for row in rows if row['controller_id'] == 'R' and row['episode_id'].endswith('offset-03'))
        special['termination'] = 'ready'
        chosen = replay_union(rows)
        self.assertEqual(len(chosen), 25)
        self.assertIn(special, chosen)
        self.assertEqual(len({row['cell_id'] for row in chosen}), len(chosen))

    def test_infrastructure_and_missing_population_block_replay(self):
        rows = self.results(); rows[0]['termination'] = 'infrastructure_error'
        with self.assertRaises(ValueError): replay_union(rows)
        with self.assertRaises(ValueError): replay_union(self.results()[:-1])

    def test_freeze_rejects_incomplete_even_when_counts_are_plausible(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'fixture.json'
            cap = {'status': 'FIT_ESTIMATE_ONLY', 'free_bytes': 4*1024**3,
                   'required_bytes': 3*1024**3, 'headroom_bytes': 1024**3}
            with self.assertRaises(ValueError):
                freeze_manifest(p, {}, integration_pass=True, resource_capacity=cap, source_root=d)
            self.assertFalse(p.exists())


class FinalFreezeSourceTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='p5-manifest-fixture-', dir=self.ROOT / 'tmp')
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.output = self.folder / 'manifest.json'
        self.contract = frozen_source_contract(self.ROOT)
        self.capacity = {'status':'FIT_ESTIMATE_ONLY','free_bytes':4*1024**3,
            'required_bytes':3*1024**3,'headroom_bytes':1024**3}

    def fixture(self):
        prior = prior_initial_inventory(self.ROOT)
        families, application = family_records(self.contract['physical_config'], seed=123,
            new_family_seed=456, initial_id_prefix='fixture', forbidden_identities=prior['canonical_identities'])
        manifest = deepcopy(self.contract)
        for name, value in (('resource', {'capacity':self.capacity}),
                            ('integration', {'status':'PASS','checks':{'roundtrip':True,'replay':True}})):
            path = self.folder / (name+'.json'); path.write_text(json.dumps(value))
            relative = str(path.relative_to(self.ROOT))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest['source_sha256'][relative] = digest
            manifest[name+'_receipt'] = {'path':relative,'sha256':digest}
        manifest.update(families=families, application_family=application,
            E1=expected_E1(families), E2=expected_E2(families),
            statistics={'replicates':2000,'seed':7711,'unit':'family_within_stratum_seed_mean_first'},
            prediction_evaluation={'horizons':[1,5,10],'source':'all_E2_G_raw_fixed_applied_actions'},
            failure_policy='retain', replay_selection_rule='stratum_and_termination_union',
            prior_initial_inventory=prior, family_generation={'recipe_version':FAMILY_RECIPE_VERSION,
                'seed':123,'application_seed':456,'initial_id_prefix':'fixture'},
            minimum_meaningful_difference='not_specified')
        return manifest

    def freeze(self, manifest):
        return freeze_manifest(self.output, manifest, integration_pass=True,
            resource_capacity=self.capacity, source_root=self.ROOT)

    def assert_rejected(self, manifest):
        with self.assertRaises(ValueError):
            self.freeze(manifest)
        self.assertFalse(self.output.exists())

    def test_real_historical_contract_fixture_freezes_exclusively(self):
        manifest = self.fixture()
        receipt = self.freeze(manifest)
        self.assertEqual(receipt['bytes'], self.output.stat().st_size)
        self.assertEqual(json.loads(self.output.read_text())['status'], 'FROZEN_BEFORE_FINAL_OUTCOMES')
        with self.assertRaises(FileExistsError):
            self.freeze(manifest)

    def test_old_fixture_only_models_and_empty_inventory_cannot_freeze(self):
        m = self.fixture()
        m.update(model_identities={name:{'fixture_only':True} for name in m['model_identities']},
            normalizer_identity='fixture',planner_config={'fixture':True},profile={'fixture':True},
            prior_initial_inventory={'source_manifest_count':1,'unresolved':[],'canonical_identities':[]})
        self.assert_rejected(m)

    def test_each_selected_model_path_hash_metadata_and_file_binding_is_required(self):
        original = self.fixture()
        for field in ('path','model_hash','metadata_path','metadata_sha256','files_sha256'):
            with self.subTest(field=field):
                m = deepcopy(original); m['model_identities']['R'][field] = {} if field=='files_sha256' else 'tampered'
                self.assert_rejected(m)
        for name in ('R', 'S1-23', 'SK-47'):
            with self.subTest(missing=name):
                m=deepcopy(original);m['model_identities'].pop(name);self.assert_rejected(m)

    def test_every_source_binding_including_candidate_weights_and_profile_is_required(self):
        original=self.fixture()
        paths=['configs/p5_freeze_candidate.v1.json', 'configs/acceptance.v1.toml',
               self.contract['model_identities']['S1-11']['path']+'/weights.pt']
        for path in paths:
            with self.subTest(path=path):
                m=deepcopy(original);m['source_sha256'].pop(path);self.assert_rejected(m)
                m=deepcopy(original);m['source_sha256'][path]='0'*64;self.assert_rejected(m)
        m=deepcopy(original);m['source_sha256']['../outside']='0'*64;self.assert_rejected(m)

    def test_normalizer_physical_planner_and_profile_must_match_actual_contents(self):
        original=self.fixture()
        for key in ('normalizer_identity','physical_config','planner_config','profile'):
            with self.subTest(key=key):
                m=deepcopy(original);m[key]={'fixture':True};self.assert_rejected(m)
        m=deepcopy(original);m['profile']['invariants']['allowed_future_simulator_calls_in_planning']=1;self.assert_rejected(m)
        m=deepcopy(original);m['planner_config']['planner']['candidate_count']=127;self.assert_rejected(m)
        m=deepcopy(original);m['physical_config']['task']['hold_time_s']=.5;self.assert_rejected(m)

    def test_seed_recipe_geometry_strata_and_pose_hash_are_recomputed(self):
        original=self.fixture()
        for field,value in (('seed',456),('application_seed',123),('recipe_version','fixture'),
                            ('initial_id_prefix','other')):
            with self.subTest(field=field):
                m=deepcopy(original);m['family_generation'][field]=value;self.assert_rejected(m)
        for mutation in ('pose','offset','rank','hash','application','duplicate'):
            with self.subTest(mutation=mutation):
                m=deepcopy(original)
                if mutation=='pose':
                    m['families'][0]['initial_spec']['object_x']+=.001
                    m['families'][0]['canonical_identity']=canonical_initial(m['families'][0]['initial_spec'])
                elif mutation=='offset':m['families'][8]['tangent_offset_m']*=-1
                elif mutation=='rank':m['families'][4]['rank_within_stratum']=3
                elif mutation=='hash':m['families'][0]['canonical_identity']='0'*64
                elif mutation=='application':m['application_family']['initial_spec']['pusher_x']+=.001
                else:m['families'][0]=deepcopy(m['families'][1])
                self.assert_rejected(m)

    def test_complete_prior_manifest_paths_hashes_poses_and_unknowns_are_recomputed(self):
        original=self.fixture()
        for kind in ('count','manifest','manifesthash','pose','canonical','unresolved','exclusion'):
            with self.subTest(kind=kind):
                m=deepcopy(original);p=m['prior_initial_inventory']
                if kind=='count':p['source_manifest_count']=1
                elif kind=='manifest':p['source_manifests'].pop(next(iter(p['source_manifests'])))
                elif kind=='manifesthash':p['source_manifests'][next(iter(p['source_manifests']))]='0'*64
                elif kind=='pose':p['recognized_initials'].pop()
                elif kind=='canonical':p['canonical_identities'].pop()
                elif kind=='unresolved':p['unresolved']=[{'initial_spec':None,'reason':'unknown'}]
                else:p['excluded_final_run']='runs'
                self.assert_rejected(m)

    def test_document_archive_preserves_actual_bytes_and_cannot_archive_code(self):
        m=self.fixture();original=next(p for p in m['source_sha256'] if p.startswith('docs/'))
        archived=self.folder/'source'/original;archived.parent.mkdir(parents=True);shutil.copy2(self.ROOT/original,archived)
        relative=str(archived.relative_to(self.ROOT));digest=m['source_sha256'].pop(original)
        m['source_sha256'][relative]=digest;m['source_archives']={original:relative}
        self.freeze(m)
        self.output.unlink()
        archived.write_bytes(archived.read_bytes()+b'changed')
        self.assert_rejected(m)
        m=self.fixture();m['source_archives']={'src/push_align/planning.py':relative};self.assert_rejected(m)

    def test_altered_actual_source_bytes_and_fake_historical_candidate_are_rejected(self):
        clone=self.folder/'source-copy';clone.mkdir()
        for relative in self.contract['source_sha256']:
            dest=clone/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(self.ROOT/relative,dest)
        self.assertEqual(frozen_source_contract(clone)['model_identities'],self.contract['model_identities'])
        original=clone/self.contract['model_identities']['R']['metadata_path'];saved=original.read_bytes()
        original.write_bytes(saved+b' ')
        with self.assertRaises(ValueError):frozen_source_contract(clone)
        original.write_bytes(saved)
        candidate=clone/'configs/p5_freeze_candidate.v1.json';m=json.loads(candidate.read_text());m['learned_models']=[];candidate.write_text(json.dumps(m))
        with self.assertRaises(ValueError):frozen_source_contract(clone)

    def test_inventory_keeps_null_unrecognized_forms_and_lists_every_manifest(self):
        root=self.folder/'inventory';(root/'runs'/'old').mkdir(parents=True)
        (root/'runs'/'old'/'manifest.json').write_text(json.dumps({'initial_spec':dict(object_x=0.,object_y=0.,object_yaw=0.,pusher_x=None,pusher_y=None)}))
        archive=root/'runs'/'old'/'source'/'saved';archive.mkdir(parents=True)
        (archive/'manifest.json').write_text('{}')
        final=root/'runs'/'p5-final-fixture';final.mkdir();(final/'manifest.json').write_text('{}')
        inventory=prior_initial_inventory(root,exclude_final_run='runs/p5-final-fixture')
        self.assertEqual(inventory['source_manifest_count'],1)
        self.assertEqual(len(inventory['excluded_source_manifest_paths']),1)
        self.assertEqual(len(inventory['unresolved']),1)
        self.assertIsNone(inventory['unresolved'][0]['initial_spec']['pusher_x'])
        (root/'runs'/'old'/'manifest.json').write_text(json.dumps({'initial_state':None}))
        self.assertEqual(prior_initial_inventory(root)['unresolved'][0]['reason'],'unsupported_initial_state_representation')
        (root/'runs'/'old'/'manifest.json').write_text(json.dumps({'initial_spec':None}))
        self.assertEqual(prior_initial_inventory(root)['unresolved'][0]['reason'],'unsupported_initial_spec_representation')


if __name__ == '__main__': unittest.main()
