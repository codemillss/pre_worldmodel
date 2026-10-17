"""Synthetic failures must not become a favorable first-moment conclusion."""
from copy import deepcopy
import unittest
from p4_first_moment_probe import global_controls, primary_counts, analysis_status


def rows():
    output=[]
    for mode in ('KEEP','ZERO_M'):
        output.append({'snapshot_id':'R1_u1000','source_probe_name':'D11_noncontact','target_probe_name':'D11_zero5','source_mode':mode,
            'controls':{'alpha0_matches_previous_stop':True,'alpha0_matches_keep':True,'keep_previous_cross':{'pass':True} if mode=='KEEP' else None},
            'evaluation':{'status':'PASS','evaluations':[{'total_change_relation':r} for r in ('NUMERIC_TIE','INCREASE','INCREASE','INCREASE')]}})
    return output


class ContractTests(unittest.TestCase):
    def test_both_increase_retains_six_positive_endpoints(self):
        r=rows(); c=global_controls(r); self.assertTrue(c['pass']); p=primary_counts(r,c['pass'])
        self.assertEqual(p['all']['planned_positive_endpoints'],6)
        self.assertTrue(all(p['zero_m_increase_persists'].values()))
        self.assertTrue(all(v['INCREASE']==2 for v in p['all']['by_alpha'].values()))
    def test_zero_m_case_failure_keeps_three_held_slots(self):
        r=rows();r[1]['evaluation']['status']='NUMERICAL_FAILURE';p=primary_counts(r,True)
        self.assertEqual(p['all']['planned_positive_endpoints'],6)
        self.assertTrue(all(v['NOT_INTERPRETED']==1 for v in p['all']['by_alpha'].values()))
        self.assertTrue(all(v is None for v in p['zero_m_increase_persists'].values()))
        self.assertEqual(p['interpretation_gate'],'HOLD_CASE_FAILURE')
        self.assertEqual(analysis_status(r,True),'HOLD_CASE_FAILURE')
    def test_keep_replay_failure_holds_all_primary_interpretation(self):
        r=rows();r[0]['controls']['keep_previous_cross']['pass']=False
        self.assertFalse(global_controls(r)['pass']);self.assertIsNone(primary_counts(r,False))
        self.assertEqual(analysis_status(r,False),'HOLD_CONTROL_FAILURE')
    def test_zero_m_alpha0_mismatch_holds_all(self):
        r=rows();r[1]['controls']['alpha0_matches_keep']=False
        self.assertFalse(global_controls(r)['pass'])
    def test_missing_duplicate_or_wrong_snapshot_rejected_by_global_controls(self):
        for r in (rows()[:1],[rows()[0],deepcopy(rows()[0])]):self.assertFalse(global_controls(r)['pass'])
        r=rows();r[1]['snapshot_id']='C0_u1000';self.assertFalse(global_controls(r)['pass'])
    def test_scale_specific_decrease_or_tie_is_not_persisting_increase(self):
        r=rows();r[1]['evaluation']['evaluations'][1]['total_change_relation']='DECREASE'
        r[1]['evaluation']['evaluations'][2]['total_change_relation']='NUMERIC_TIE';p=primary_counts(r,True)
        self.assertEqual(p['zero_m_increase_persists'],{'0.25':False,'0.5':False,'1.0':True})


if __name__=='__main__':unittest.main()
