"""Small saved-array controls and denominator tests, without model execution."""
from copy import deepcopy
import unittest
import numpy as np
from tools.check_p4_first_moment import saved_controls, primary_counts


def fixture():
    e={'evaluations':[{'loss_values':{'total':v},'status':'PASS'} for v in (1.,1.1,1.2,1.3)]}
    pred={'teacher_predictions':np.arange(8,dtype=np.float32).reshape(4,2),
          'free_predicted_states':np.arange(12,dtype=np.float64).reshape(4,3)}
    params={'alphas':np.array([0.,.25,.5,1.]),'evaluated_parameters':np.arange(8,dtype=np.float32).reshape(4,2),
            'rounded_parameter_differences':np.arange(8,dtype=np.float32).reshape(4,2)}
    zero={'loss':deepcopy(e['evaluations'][0]['loss_values']),**{k:v[0].copy() for k,v in pred.items()}}
    return e,pred,params,deepcopy(e),deepcopy(pred),deepcopy(params),zero


def cases():
    return [{'source_mode':m,'case_status':'PASS','change_relations':['NUMERIC_TIE','INCREASE','INCREASE','INCREASE']} for m in ('KEEP','ZERO_M')]


class AuditTests(unittest.TestCase):
    def test_exact_keep_control_passes(self):
        c=saved_controls('KEEP',*fixture());self.assertTrue(c['keep_previous_cross']['pass']);self.assertTrue(c['alpha0_matches_keep']);self.assertTrue(c['alpha0_matches_previous_stop'])
    def test_keep_nonzero_parameter_substitution_is_not_exact(self):
        f=fixture();f[2]['evaluated_parameters'][2,0]+=1
        self.assertFalse(saved_controls('KEEP',*f)['keep_previous_cross']['pass'])
    def test_keep_dtype_substitution_is_not_exact(self):
        f=fixture();f[1]['teacher_predictions']=f[1]['teacher_predictions'].astype(np.float64)
        self.assertFalse(saved_controls('KEEP',*f)['keep_previous_cross']['pass'])
    def test_keep_nonzero_loss_substitution_is_not_exact(self):
        f=fixture();f[0]['evaluations'][3]['loss_values']['total']+=1
        self.assertFalse(saved_controls('KEEP',*f)['keep_previous_cross']['pass'])
    def test_zero_m_nonzero_maps_need_not_match_keep(self):
        f=fixture();f[2]['evaluated_parameters'][1:]+=1;c=saved_controls('ZERO_M',*f)
        self.assertIsNone(c['keep_previous_cross']);self.assertTrue(c['alpha0_matches_keep'])
    def test_zero_m_alpha0_loss_mismatch_detected(self):
        f=fixture();f[0]['evaluations'][0]['loss_values']['total']+=1;c=saved_controls('ZERO_M',*f)
        self.assertFalse(c['alpha0_matches_keep']);self.assertFalse(c['alpha0_matches_previous_stop'])
    def test_zero_m_alpha0_prediction_mismatch_detected(self):
        f=fixture();f[1]['free_predicted_states'][0,0]+=1
        self.assertFalse(saved_controls('ZERO_M',*f)['alpha0_matches_keep'])
    def test_primary_retains_all_six_endpoints(self):
        p=primary_counts(cases(),True);self.assertEqual(p['all']['planned_positive_endpoints'],6)
        self.assertTrue(all(v['INCREASE']==2 for v in p['all']['by_alpha'].values()));self.assertTrue(all(p['zero_m_increase_persists'].values()))
    def test_failed_case_retains_three_not_interpreted_endpoints(self):
        c=cases();c[1]['case_status']='NUMERICAL_FAILURE';p=primary_counts(c,True)
        self.assertTrue(all(v['NOT_INTERPRETED']==1 for v in p['all']['by_alpha'].values()));self.assertTrue(all(v is None for v in p['zero_m_increase_persists'].values()))
        self.assertEqual(p['interpretation_gate'],'HOLD_CASE_FAILURE')
    def test_global_control_failure_nulls_primary(self):self.assertIsNone(primary_counts(cases(),False))
    def test_decrease_and_tie_do_not_count_as_persisting_increase(self):
        c=cases();c[1]['change_relations'][1:]=['DECREASE','NUMERIC_TIE','INCREASE'];p=primary_counts(c,True)
        self.assertEqual(p['zero_m_increase_persists'],{'0.25':False,'0.5':False,'1.0':True})
    def test_reordered_or_missing_modes_rejected(self):
        for c in (cases()[:1],cases()[::-1]):
            with self.assertRaises(Exception):primary_counts(c,True)


if __name__=='__main__':unittest.main()
