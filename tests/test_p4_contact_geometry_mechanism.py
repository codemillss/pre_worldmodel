import importlib.util
from pathlib import Path

import numpy as np
import unittest

spec=importlib.util.spec_from_file_location('contact_mechanism_tool',Path(__file__).parents[1]/'tools/diagnose_p4_contact_geometry.py')
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
CFG={'object':{'size_full_xyz_m':[.08,.06,.02]},'pusher':{'radius_m':.008}}


def state():
    s=np.zeros(10);s[6]=-.05
    return s


class GeometryMechanismTests(unittest.TestCase):
    def test_unambiguous_face_gap_and_derivative(self):
        g=tool.geometry(state(),CFG)
        self.assertAlmostEqual(g['gap_m'],.002)
        np.testing.assert_allclose(g['normal_world'],[-1,0],atol=1e-15)
        self.assertEqual(g['gap_yaw_derivative_m_per_rad'],0)


    def test_corner_gap_uses_euclidean_distance(self):
        s=state();s[6:8]=[.05,.04]
        self.assertAlmostEqual(tool.geometry(s,CFG)['gap_m'],np.hypot(.01,.01)-.008)


    def test_common_translation_and_pi_rotation_preserve_gap(self):
        s=state();g=tool.geometry(s,CFG)['gap_m']
        moved=s.copy();moved[:2]+=[.1,-.03];moved[6:8]+=[.1,-.03]
        self.assertAlmostEqual(tool.geometry(moved,CFG)['gap_m'],g)
        moved[2]+=np.pi
        self.assertAlmostEqual(tool.geometry(moved,CFG)['gap_m'],g)


    def test_pure_face_position_errors_have_signed_contributions(self):
        actual=state();pred=actual.copy();pred[0]=-.0003;pred[6]+=.0007
        r=tool.decompose(actual,pred,CFG)
        np.testing.assert_allclose(list(r['shapley_gap_error_m'].values()),[-.0003,-.0007,0],atol=1e-16)
        self.assertAlmostEqual(r['gap_error_m'],-.001)
        self.assertLess(abs(r['local_linearization']['nonlinear_remainder_m']),1e-16)


    def test_velocity_errors_have_no_instantaneous_gap_contribution(self):
        actual=state();pred=actual.copy();pred[[3,4,5,8,9]]=[100,-100,10,20,30]
        r=tool.decompose(actual,pred,CFG)
        self.assertEqual(r['gap_error_m'],0)
        self.assertEqual(sum(r['shapley_gap_error_m'].values()),0)
        self.assertGreater(np.linalg.norm(r['state_error']),0)


    def test_inside_box_tie_does_not_invent_a_normal(self):
        s=state();s[6:8]=[.01,0]
        g=tool.geometry(s,CFG)
        self.assertEqual(g['derivative_status'],'NOT_DEFINED')
        self.assertIsNone(g['normal_world'])


    def test_large_rotation_and_face_switch_keep_interactions_and_closure(self):
        actual=state();pred=actual.copy();pred[0]=.007;pred[6:8]=[.01,.039];pred[2]=.7
        r=tool.decompose(actual,pred,CFG)
        self.assertEqual(len(r['hybrids']),8)
        self.assertEqual(len(r['inclusion_exclusion_m']),7)
        self.assertLess(abs(sum(r['shapley_gap_error_m'].values())-r['gap_error_m']),1e-14)
        self.assertTrue(any(abs(v)>1e-8 for k,v in r['inclusion_exclusion_m'].items() if '+' in k))


    def test_yaw_error_uses_2pi_and_nonfinite_is_rejected(self):
        actual=state();pred=actual.copy();actual[2]=np.pi-.01;pred[2]=-np.pi+.01
        self.assertAlmostEqual(tool.state_error(pred,actual)[2],.02)
        pred[1]=np.nan
        with self.assertRaises(ValueError):tool.decompose(actual,pred,CFG)

    def test_canonical_family_identity_and_duplicate_heads(self):
        row={'family_id':'p4-data-development-2026101302-f001','family_rank':1,'y_condition':'Y1','stop_condition':'S0'}
        self.assertEqual(tool.selected_contact_episode([{**row,'stop_condition':'S1'},row]),1)
        with self.assertRaises(ValueError):tool.selected_contact_episode([{**row,'family_id':'f001'}])
        with self.assertRaises(ValueError):tool.selected_contact_episode([row,row])

    def test_smooth_switch_guard_is_not_claimed_nondifferentiable(self):
        s=state();s[6:8]=[.04,0]
        g=tool.geometry(s,CFG)
        self.assertEqual(g['derivative_status'],'NEAR_SWITCH_NOT_ASSESSED')


if __name__=='__main__':
    unittest.main()
