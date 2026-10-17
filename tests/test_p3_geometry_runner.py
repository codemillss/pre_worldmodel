"""Independent recovery-evidence and frozen-case runner tests.

Synthetic recorded contacts test the reporting predicate, rather than contact
physics or controller performance. No fixture here constitutes robot success.
"""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import tomllib
import unittest
from unittest.mock import patch

import numpy as np

from push_align.config import ROOT, load_pilot_config
from push_align.data import DataContractError, digest
from push_align.planning import TaskSpec


def sample(time, *, force=.002, pair='pusher_object', qx=-.048, qy=0., yaw=0.):
    state = np.zeros(10, np.float64)
    state[2] = yaw
    theta = .5*math.atan2(math.sin(2*yaw), math.cos(2*yaw))
    state[6:8] = qx*np.array([math.cos(theta), math.sin(theta)]) \
                    + qy*np.array([-math.sin(theta), math.cos(theta)])
    contacts = [] if force is None else [{'pair': pair, 'normal_force_N': force}]
    return {'sim_time': time, 'contacts': contacts, 'state': state.tolist()}


def result(*samples):
    return SimpleNamespace(diagnostics={'samples': list(samples)})


def withdrawal_records(count=6):
    return [{'source': 'scheduled_withdrawal',
             'interval_start_time': .1 + .1*i, 'sim_time': .2 + .1*i,
             'body_qx_before_m': -.048, 'body_qy_before_m': 0.}
            for i in range(count)]


def valid_recovery(*, resume=.8, duration=.02, force=.002, qx=-.048, qy=0.):
    return [result(sample(0.), sample(.1), sample(.11, force=None),
                   sample(.5, force=None), sample(.7, force=None),
                   sample(resume-.01, force=None),
                   sample(resume, force=force, qx=qx, qy=qy),
                   sample(resume+duration, force=force, qx=qx, qy=qy))]


class GeometryRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = str(ROOT/'scripts')
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        spec = importlib.util.spec_from_file_location('p3_geometry_runner_test_module',
                                                     ROOT/'scripts/p3_geometry_pilot.py')
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)

    def setUp(self):
        self.config = load_pilot_config()
        self.settings = tomllib.loads((ROOT/'configs/p3_geometry_cases.v1.toml').read_text())

    def evidence(self, results=None, records=None):
        return self.runner.recovery_evidence(
            valid_recovery() if results is None else results,
            withdrawal_records() if records is None else records,
            self.settings, self.config)

    def test_recontact_during_withdrawal_is_not_post_intervention_recovery(self):
        # Actual loss .10; qualified resume .21; withdrawal continues through .70.
        results = [result(sample(0.), sample(.09), sample(.1, force=None),
                          sample(.2, force=None), sample(.21), sample(.23), sample(.7))]
        evidence = self.evidence(results)
        self.assertEqual(len(evidence['events']['qualified_recontacts']), 1)
        self.assertEqual(evidence['matching_qualified_recontacts'], [])
        self.assertFalse(evidence['resumed_active_samples_on_rear_face'])

    def test_measured_loss_then_qualified_recontact_after_withdrawal_matches(self):
        evidence = self.evidence()
        self.assertTrue(evidence['scheduled_withdrawal_complete'])
        self.assertEqual(len(evidence['matching_qualified_recontacts']), 1)
        event = evidence['matching_qualified_recontacts'][0]
        self.assertAlmostEqual(event['contact_loss_time_s'], .11)
        self.assertAlmostEqual(event['contact_resume_time_s'], .8)
        self.assertAlmostEqual(event['observed_inactive_span_s'], .68)
        self.assertAlmostEqual(event['observed_resumed_active_span_s'], .02)
        self.assertTrue(evidence['resumed_active_samples_on_rear_face'])

    def test_loss_before_withdrawal_does_not_match_intervention(self):
        results = [result(sample(0.), sample(.04), sample(.05, force=None),
                          sample(.79, force=None), sample(.8), sample(.82))]
        evidence = self.evidence(results)
        self.assertEqual(len(evidence['events']['qualified_recontacts']), 1)
        self.assertEqual(evidence['matching_qualified_recontacts'], [])

    def test_no_scheduled_withdrawal_does_not_claim_matching_recovery(self):
        evidence = self.evidence(records=[])
        self.assertEqual(evidence['withdrawal_intervals'], 0)
        self.assertFalse(evidence['scheduled_withdrawal_complete'])
        self.assertEqual(evidence['matching_qualified_recontacts'], [])
        self.assertFalse(evidence['resumed_active_samples_on_rear_face'])

    def test_incomplete_withdrawal_remains_explicitly_incomplete(self):
        evidence = self.evidence(records=withdrawal_records(5))
        self.assertEqual(evidence['withdrawal_intervals'], 5)
        self.assertFalse(evidence['scheduled_withdrawal_complete'])

    def test_force_at_threshold_is_inactive_and_strictly_above_is_active(self):
        threshold = .001
        equality = self.evidence(valid_recovery(force=threshold))
        above = self.evidence(valid_recovery(force=np.nextafter(threshold, math.inf)))
        self.assertEqual(equality['matching_qualified_recontacts'], [])
        self.assertFalse(equality['resumed_active_samples_on_rear_face'])
        self.assertEqual(len(above['matching_qualified_recontacts']), 1)
        self.assertTrue(above['resumed_active_samples_on_rear_face'])

    def test_other_contact_pair_cannot_create_pusher_recontact(self):
        results = valid_recovery()
        for value in results[0].diagnostics['samples']:
            for contact in value['contacts']:
                contact['pair'] = 'floor_object'
        self.assertEqual(self.evidence(results)['matching_qualified_recontacts'], [])

    def test_shared_boundary_is_deduplicated_and_short_span_is_not_extended(self):
        first = result(sample(0.), sample(.1), sample(.11, force=None),
                       sample(.79, force=None), sample(.8))
        second = result(sample(.8), sample(.809))
        evidence = self.evidence([first, second])
        self.assertEqual(evidence['events']['deduplicated_raw_sample_count'], 6)
        self.assertEqual(evidence['matching_qualified_recontacts'], [])
        self.assertFalse(evidence['resumed_active_samples_on_rear_face'])

    def test_exact_minimum_observed_active_span_qualifies(self):
        evidence = self.evidence(valid_recovery(duration=.01))
        self.assertEqual(len(evidence['matching_qualified_recontacts']), 1)
        self.assertTrue(evidence['resumed_active_samples_on_rear_face'])

    def test_short_inactive_span_does_not_qualify_even_with_long_active_resume(self):
        results = [result(sample(0.), sample(.7), sample(.71, force=None),
                          sample(.739, force=None), sample(.75), sample(.80))]
        # Withdrawal ends .70; loss lies outside the intervention too, but the
        # independent event predicate must first reject the .029 s absence.
        evidence = self.evidence(results)
        self.assertEqual(evidence['events']['qualified_recontacts'], [])

    def test_unknown_force_breaks_chain_instead_of_inventing_recontact(self):
        results = valid_recovery()
        results[0].diagnostics['samples'].insert(6, sample(.795, force=math.nan))
        evidence = self.evidence(results)
        self.assertEqual(evidence['events']['unknown_force_sample_count'], 1)
        self.assertEqual(evidence['matching_qualified_recontacts'], [])

    def test_conflicting_contact_at_shared_boundary_is_rejected(self):
        results = [result(sample(0.), sample(.1)),
                   result(sample(.1, force=None), sample(.2, force=None))]
        with self.assertRaises(DataContractError):
            self.evidence(results)

    def test_rear_half_plane_contact_is_not_necessarily_rear_face_contact(self):
        evidence = self.evidence(valid_recovery(qx=-.02, qy=.03))
        self.assertEqual(len(evidence['matching_qualified_recontacts']), 1)
        self.assertTrue(evidence['same_rear_face_at_all_action_starts'])
        self.assertFalse(evidence['resumed_active_samples_on_rear_face'])

    def test_rear_face_classification_checks_normal_and_tangent_coordinates(self):
        for qx, qy in ((-.0491, 0.), (-.048, .0301)):
            with self.subTest(qx=qx, qy=qy):
                evidence = self.evidence(valid_recovery(qx=qx, qy=qy))
                self.assertEqual(len(evidence['matching_qualified_recontacts']), 1)
                self.assertFalse(evidence['resumed_active_samples_on_rear_face'])

    def test_every_active_sample_in_resume_window_must_be_on_rear_face(self):
        results = valid_recovery()
        results[0].diagnostics['samples'].insert(7, sample(.805, qx=-.02, qy=.03))
        evidence = self.evidence(results)
        self.assertEqual(len(evidence['matching_qualified_recontacts']), 1)
        self.assertFalse(evidence['resumed_active_samples_on_rear_face'])

    def test_body_coordinates_use_pi_symmetric_face_and_relative_position(self):
        value = np.array([.04, -.02, .1, 0., 0., 0., 0., 0., 0., 0.], np.float64)
        d = np.array([math.cos(.1), math.sin(.1)])
        t = np.array([-math.sin(.1), math.cos(.1)])
        value[6:8] = value[:2] - .048*d + .01*t
        normal, qx, qy = self.runner.body_coordinates(value)
        np.testing.assert_allclose(normal, d, atol=1e-14)
        self.assertAlmostEqual(qx, -.048, places=14)
        self.assertAlmostEqual(qy, .01, places=14)
        equivalent = value.copy()
        equivalent[2] += math.pi
        normal2, qx2, qy2 = self.runner.body_coordinates(equivalent)
        np.testing.assert_allclose(normal2, normal, atol=1e-14)
        self.assertAlmostEqual(qx2, qx, places=14)
        self.assertAlmostEqual(qy2, qy, places=14)

    def test_frozen_four_cases_have_exact_ids_order_and_initial_rear_face(self):
        goal_before = TaskSpec.from_config(self.config).goal.copy()
        config_before = deepcopy(self.config)
        settings_before = deepcopy(self.settings)
        plan = self.runner.case_plan(self.settings, self.config)
        self.assertEqual([case['episode_id'] for case in plan],
                         ['g-translation', 'g-yaw-positive', 'g-yaw-negative', 'g-recontact'])
        self.assertEqual([case['episode_index'] for case in plan], [0, 1, 2, 3])
        self.assertEqual([case['require_recontact'] for case in plan], [False, False, False, True])
        self.assertEqual(len({case['family_id'] for case in plan}), 4)
        for case in plan:
            initial = case['initial_spec']
            value = np.array([initial['object_x'], initial['object_y'], initial['object_yaw'],
                              0., 0., 0., initial['pusher_x'], initial['pusher_y'], 0., 0.],
                             np.float64)
            _, qx, qy = self.runner.body_coordinates(value)
            self.assertAlmostEqual(qx, -.051, places=14)
            self.assertAlmostEqual(qy, 0., places=14)
        self.assertAlmostEqual(plan[1]['initial_spec']['object_yaw'], math.radians(10))
        self.assertAlmostEqual(plan[2]['initial_spec']['object_yaw'], math.radians(-10))
        self.assertEqual(self.settings, settings_before)
        self.assertEqual(self.config, config_before)
        np.testing.assert_array_equal(TaskSpec.from_config(self.config).goal, goal_before)

    def test_changed_case_count_id_or_order_is_rejected(self):
        for change in ('count', 'id', 'order'):
            settings = deepcopy(self.settings)
            if change == 'count':
                settings['cases'].pop()
            elif change == 'id':
                settings['cases'][0]['id'] = 'replacement-successful-case'
            else:
                settings['cases'][1:3] = settings['cases'][1:3][::-1]
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    self.runner.case_plan(settings, self.config)

    def test_changed_case_file_hash_or_frozen_settings_is_rejected_before_physics(self):
        original = (ROOT/'configs/p3_geometry_cases.v1.toml').read_text()
        for change in ('hash', 'settings'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root/'reports').mkdir()
                cases = root/'cases.toml'
                cases.write_text(original)
                frozen = {'case_file_sha256': digest(cases), 'settings': deepcopy(self.settings)}
                if change == 'hash':
                    frozen['case_file_sha256'] = '0'*64
                else:
                    frozen['settings']['cases'][0]['object_x'] = -.02
                (root/'reports/p3_01_cases_frozen.json').write_text(json.dumps(frozen))
                args = SimpleNamespace(cases=cases)
                with patch.object(self.runner, 'ROOT', root), \
                     patch.object(self.runner, 'load_pilot_config', return_value=self.config), \
                     patch.object(self.runner, 'PlanarPushEnv',
                                  side_effect=AssertionError('Physics must not start')) as env:
                    with self.assertRaisesRegex(ValueError, 'Frozen cases changed'):
                        self.runner.run(args, root/'unused-output')
                    env.assert_not_called()


if __name__ == '__main__':
    unittest.main()
