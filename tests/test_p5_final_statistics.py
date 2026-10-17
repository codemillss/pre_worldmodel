import unittest
import numpy as np
from push_align.final_manifest import expected_E2, family_records
from push_align.final_statistics import analyze_E2, METRICS


class FinalStatisticsTests(unittest.TestCase):
    def data(self):
        physical={'object':{'size_full_xyz_m':[.06,.03,.02]},'pusher':{'radius_m':.015},'reset':{'pusher_surface_gap_m':.006}}
        families,_=family_records(physical,seed=123,new_family_seed=456,initial_id_prefix='fixture')
        expected=expected_E2(families); rows=[]
        for identity in expected:
            row=dict(identity,execution_status='completed',termination='timeout',**{key:1. for key in METRICS})
            row.update(success=False,elapsed_s=30.,recontact_count=1,
                       final_observation_available=True,metric_state_origin='terminal_observation',metric_state_time_s=30.)
            if row['controller_id'].startswith('SK'):
                row.update(termination='ready',success=True,elapsed_s=10.,metric_state_time_s=10.)
            rows.append(row)
        return rows,expected

    def test_known_paired_effect_is_exact_and_all_denominators_present(self):
        rows,expected=self.data();result,arrays=analyze_E2(rows,expected)
        self.assertEqual(result['episode_count'],96)
        contrast=next(x for x in result['comparisons'] if x['left']=='SK' and x['right']=='S1')
        self.assertEqual(contrast['metrics']['success']['percentile95_interval'],[1.,1.])
        self.assertEqual(contrast['metrics']['elapsed_s']['percentile95_interval'],[-20.,-20.])
        self.assertEqual(result['seed_group_summary']['SK']['family_count'],12)
        self.assertEqual(result['seed_group_summary']['SK']['episode_count'],36)
        self.assertEqual(result['controllers']['G']['success_conditional_time_s']['mean'],None)
        self.assertEqual(arrays['family_draw_indices'].shape,(2000,12))

    def test_strata_draws_and_common_seed_resampling_are_reproducible(self):
        rows,expected=self.data();a,arrays=analyze_E2(rows,expected);b,other=analyze_E2(list(reversed(rows)),expected)
        self.assertEqual(a,b)
        self.assertTrue(all(np.array_equal(arrays[k],other[k]) for k in arrays))
        families=a['bootstrap']['families_in_order']
        for start,stratum in zip((0,4,8),('translation','rotation','offset')):
            self.assertTrue(set(families[i] for i in arrays['family_draw_indices'][:,start:start+4].ravel()) <= set(a['bootstrap']['strata'][stratum]))

    def test_duplicate_missing_and_infrastructure_never_become_complete(self):
        rows,expected=self.data()
        for altered in (rows[:-1],rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):analyze_E2(altered,expected)
        rows[0]['termination']='infrastructure_error'
        with self.assertRaises(ValueError):analyze_E2(rows,expected)

    def test_success_is_not_inferred_from_endpoint_score(self):
        rows,expected=self.data();rows[0]['success']=True
        with self.assertRaises(ValueError):analyze_E2(rows,expected)

    def test_nonfinite_metrics_are_not_silently_deleted(self):
        rows,expected=self.data();rows[0]['final_position_error_m']=float('nan')
        with self.assertRaises(ValueError):analyze_E2(rows,expected)

    def test_unknown_terminal_numeric_bool_and_fractional_count_rejected(self):
        for key,value in [('termination','banana'),('elapsed_s',True),('recontact_count',.5)]:
            rows,expected=self.data();rows[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):analyze_E2(rows,expected)

    def test_missing_terminal_state_keeps_failure_and_declares_conditional_denominator(self):
        rows,expected=self.data();row=rows[0]
        row.update(termination='physics_error',final_observation_available=False,
                   metric_state_origin='unavailable',metric_state_time_s=None,unavailable_reason='invalid endpoint without usable fallback')
        for key in ('final_position_error_m','final_yaw_error_rad','final_linear_speed_m_s','final_angular_speed_rad_s'):row[key]=None
        result,arrays=analyze_E2(rows,expected)
        self.assertEqual(result['controllers']['G']['episode_count'],12)
        self.assertEqual(result['controllers']['G']['success_count'],0)
        self.assertEqual(result['controllers']['G']['metric_denominators']['final_position_error_m'],11)
        self.assertEqual(result['controllers']['G']['terminal_observation_unavailable_count'],1)
        contrast=next(x for x in result['comparisons'] if x['left']=='R' and x['right']=='G')
        self.assertEqual(contrast['metrics']['final_position_error_m']['eligible_family_denominator'],11)
        self.assertEqual(contrast['metrics']['success']['eligible_family_denominator'],12)

    def test_large_finite_values_do_not_overflow_aggregates(self):
        rows,expected=self.data()
        for row in rows:row['planning_wall_s']=1e308
        result,arrays=analyze_E2(rows,expected)
        self.assertEqual(result['controllers']['G']['all_episode_means']['planning_wall_s'],1e308)
        self.assertTrue(all(np.isfinite(x).all() for x in arrays.values()))

    def test_terminal_state_time_must_match_elapsed_and_success_needs_observation(self):
        rows,expected=self.data();rows[0]['metric_state_time_s']=0.
        with self.assertRaises(ValueError):analyze_E2(rows,expected)
        rows,expected=self.data();row=next(row for row in rows if row['success'])
        row.update(final_observation_available=False,metric_state_origin='last_valid_observation',metric_state_time_s=9.)
        with self.assertRaises(ValueError):analyze_E2(rows,expected)


if __name__=='__main__':unittest.main()
