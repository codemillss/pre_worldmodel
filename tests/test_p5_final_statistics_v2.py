"""Nonconstant decimal regression fixtures; no model/physics/official output."""
from copy import deepcopy
import json
import math
import random
import unittest

import numpy as np

from push_align.final_manifest import expected_E2, family_records
from push_align.final_statistics import analyze_E2 as frozen_analyze_E2
from push_align.final_statistics_v2 import analyze_E2, ANALYSIS_INFRASTRUCTURE_VERSION


def fixture():
    physical={'object':{'size_full_xyz_m':[.06,.03,.02]},'pusher':{'radius_m':.015},'reset':{'pusher_surface_gap_m':.006}}
    families,_=family_records(physical,seed=123,new_family_seed=456,initial_id_prefix='decimal-fixture')
    expected=expected_E2(families);rows=[]
    for i,identity in enumerate(expected):
        success=(i%5 in (0,2))
        elapsed=(.7+(i%19)*.137+(i%3)*.01) if success else 30.
        row=dict(identity,execution_status='completed',termination='ready' if success else 'timeout',
            success=success,elapsed_s=elapsed,final_position_error_m=(i+1)/713.+(i%3)/107.,
            final_yaw_error_rad=abs(math.sin(i+.17))*.31,
            final_linear_speed_m_s=(i%11+1)*.0013+1e-20,
            final_angular_speed_rad_s=(i%7+1)/997.+(i%2)*.00001,
            recontact_count=i%9,planning_wall_s=(i+1)/17.+(i%4)*.017,
            execution_wall_s=(i+1)/3.7+(i%3)*.013,
            final_observation_available=True,metric_state_origin='terminal_observation',metric_state_time_s=elapsed)
        rows.append(row)
    return rows,expected


def assert_arrays_equal(test,left,right):
    test.assertEqual(set(left),set(right))
    for name in left:
        with test.subTest(array=name):
            test.assertEqual(left[name].dtype,right[name].dtype)
            test.assertEqual(left[name].shape,right[name].shape)
            test.assertEqual(left[name].tobytes(),right[name].tobytes())


def numeric_close_tree(test,left,right):
    test.assertEqual(type(left),type(right))
    if type(left) is dict:
        test.assertEqual(set(left),set(right))
        for key in left:numeric_close_tree(test,left[key],right[key])
    elif type(left) is list:
        test.assertEqual(len(left),len(right))
        for a,b in zip(left,right):numeric_close_tree(test,a,b)
    elif type(left) is float:
        test.assertTrue(math.isclose(left,right,rel_tol=1e-12,abs_tol=1e-12),(left,right))
    else:test.assertEqual(left,right)


class CanonicalE2StatisticsTests(unittest.TestCase):
    def test_nonconstant_decimal_fixture_exposes_original_reduction_order(self):
        rows,expected=fixture()
        original,a=frozen_analyze_E2(rows,expected)
        reverse,b=frozen_analyze_E2(list(reversed(rows)),expected)
        self.assertNotEqual(original,reverse)
        numeric_close_tree(self,original,reverse)
        assert_arrays_equal(self,a,b)

    def test_forward_reverse_and_shuffles_are_exact_with_fixed_expected_sequence(self):
        rows,expected=fixture();saved=deepcopy(rows)
        baseline,arrays=analyze_E2(rows,expected)
        orders=[list(reversed(rows)),sorted(rows,key=lambda row:row['cell_id'])]
        for seed in (1,7,23):
            order=list(rows);random.Random(seed).shuffle(order);orders.append(order)
        for order in orders:
            summary,other=analyze_E2(order,expected)
            self.assertEqual(baseline,summary)
            self.assertEqual(json.dumps(baseline,sort_keys=True),json.dumps(summary,sort_keys=True))
            assert_arrays_equal(self,arrays,other)
        self.assertEqual(rows,saved)
        self.assertEqual(ANALYSIS_INFRASTRUCTURE_VERSION,'p5-E2-canonical-row-reporting-v2')

    def test_original_canonical_result_is_returned_without_estimator_or_schema_changes(self):
        rows,expected=fixture()
        ranks={row['cell_id']:i for i,row in enumerate(expected)}
        canonical=sorted(rows,key=lambda row:ranks[row['cell_id']])
        original,arrays=frozen_analyze_E2(canonical,expected)
        repaired,other=analyze_E2(rows,expected)
        self.assertEqual(original,repaired)
        assert_arrays_equal(self,arrays,other)
        raw,raw_arrays=frozen_analyze_E2(rows,expected)
        numeric_close_tree(self,raw,repaired)
        assert_arrays_equal(self,raw_arrays,other)
        self.assertEqual(repaired['episode_count'],96)
        self.assertEqual(repaired['family_count'],12)
        for controller,value in repaired['controllers'].items():
            self.assertEqual(value['episode_count'],12)
            self.assertEqual(value['termination_counts'],raw['controllers'][controller]['termination_counts'])
            self.assertEqual(value['success_count'],raw['controllers'][controller]['success_count'])
            self.assertEqual(value['metric_denominators'],raw['controllers'][controller]['metric_denominators'])

    def test_fixed_manifest_sequence_preserves_original_forward_bits(self):
        rows,expected=fixture()
        original,arrays=frozen_analyze_E2(rows,expected)
        repaired,other=analyze_E2(list(reversed(rows)),expected)
        self.assertEqual(original,repaired)
        assert_arrays_equal(self,arrays,other)

    def test_conditional_missing_state_population_is_preserved(self):
        rows,expected=fixture();row=rows[0]
        row.update(termination='physics_error',success=False,elapsed_s=3.7,
            final_observation_available=False,metric_state_origin='unavailable',metric_state_time_s=None,
            unavailable_reason='synthetic fixture missing terminal state')
        for metric in ('final_position_error_m','final_yaw_error_rad','final_linear_speed_m_s','final_angular_speed_rad_s'):row[metric]=None
        a,arrays=analyze_E2(rows,expected);b,other=analyze_E2(list(reversed(rows)),expected)
        self.assertEqual(a,b);assert_arrays_equal(self,arrays,other)
        self.assertEqual(a['controllers']['G']['episode_count'],12)
        self.assertEqual(a['controllers']['G']['metric_denominators']['final_position_error_m'],11)
        self.assertEqual(a['controllers']['G']['termination_counts']['physics_error'],1)

    def test_frozen_complete_population_and_invalid_outcome_rejections_remain(self):
        rows,expected=fixture()
        for invalid in (rows[:-1],rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):analyze_E2(invalid,expected)
        for key,value in (('termination','infrastructure_error'),('success',1),
                ('final_position_error_m',float('nan')),('elapsed_s',True),('recontact_count',.5)):
            changed=deepcopy(rows);changed[0][key]=value
            with self.subTest(field=key),self.assertRaises(ValueError):analyze_E2(changed,expected)
        changed=deepcopy(rows);changed[0]['cell_id']=None
        with self.assertRaises(ValueError):analyze_E2(changed,expected)

    def test_explicit_bootstrap_parameters_are_forwarded_unchanged(self):
        rows,expected=fixture()
        a,arrays=analyze_E2(rows,expected,bootstrap_replicates=31,bootstrap_seed=111)
        ranks={row['cell_id']:i for i,row in enumerate(expected)}
        b,other=frozen_analyze_E2(sorted(rows,key=lambda row:ranks[row['cell_id']]),expected,
            bootstrap_replicates=31,bootstrap_seed=111)
        self.assertEqual(a,b);assert_arrays_equal(self,arrays,other)
        self.assertEqual(a['bootstrap']['replicates'],31)
        self.assertEqual(a['bootstrap']['seed'],111)


if __name__=='__main__':unittest.main()
