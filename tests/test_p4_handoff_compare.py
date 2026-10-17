"""Cost-only isolation and failure-population contracts for the diagnostic."""
from copy import deepcopy
import unittest

import numpy as np

from push_align.planning import PredictionBatch
from p4_handoff_cost_compare import prefix_prediction, shared_subset, extended_config, canonical_status_equal


class HandoffComparisonContractTests(unittest.TestCase):
    def test_task_status_serialization_equivalence_preserves_real_discrepancies(self):
        left={'violations':(),'phase':'running','held_time_s':0.}
        right={'violations':[],'phase':'running','held_time_s':0.}
        self.assertTrue(canonical_status_equal(left,right))
        self.assertFalse(canonical_status_equal(left,{**right,'held_time_s':.1}))
        self.assertFalse(canonical_status_equal(left,{**right,'phase':'ready'}))

    def prediction(self):
        return PredictionBatch(np.zeros((3,11,10),np.float64),
            np.zeros((3,10,10),np.float64),np.array([True,False,False]),
            ('','after_H5','at_H5'),np.array([-1,6,5],np.int64))

    def test_failure_after_H5_does_not_change_original_H5_eligibility(self):
        p=self.prediction();short=prefix_prediction(p)
        np.testing.assert_array_equal(short.valid,[True,True,False])
        np.testing.assert_array_equal(short.failure_steps,[-1,-1,5])
        self.assertEqual(short.reasons,('','','at_H5'))

    def test_prefix_uses_identical_first_five_states_and_deltas(self):
        p=self.prediction();p.states[:]=np.arange(330).reshape(3,11,10)
        p.raw_deltas[:]=np.arange(300).reshape(3,10,10)
        q=prefix_prediction(p)
        np.testing.assert_array_equal(q.states,p.states[:,:6]);np.testing.assert_array_equal(q.raw_deltas,p.raw_deltas[:,:5])
        q.states[:]=1;self.assertNotEqual(p.states[0,0,0],1)

    def test_failure_at_initial_sample_remains_invalid_in_reference(self):
        p=self.prediction();p.failure_steps[1]=0
        self.assertFalse(prefix_prediction(p).valid[1])

    def test_nonregistered_prefix_is_rejected(self):
        for h in [True,4,6,5.]:
            with self.assertRaises(ValueError):prefix_prediction(self.prediction(),h)

    def test_subset_contains_every_selection_anchors_and_reserved(self):
        choices=[{'selected_index':i} for i in [100,101,102,103,104,105,106,107,108]]
        got=shared_subset(choices)
        self.assertEqual(len(got),16)
        self.assertTrue(set(range(100,109))|{0,1,2,3,11,60} <= set(got))
        self.assertEqual(got,sorted(set(got)))

    def test_invalid_and_duplicate_choices_are_retained_without_replacement(self):
        got=shared_subset([{'selected_index':None},{'selected_index':60},{'selected_index':60}])
        self.assertEqual(len(got),16);self.assertIn(11,got);self.assertIn(60,got)

    def test_bad_selection_index_is_not_silently_removed(self):
        for i in [-1,128,True,1.]:
            with self.assertRaises(ValueError):shared_subset([{'selected_index':i}])

    def test_subset_cannot_drop_required_choices_to_fit_budget(self):
        with self.assertRaises(ValueError):shared_subset([{'selected_index':i} for i in range(30,50)])

    def test_extended_evaluation_keeps_physics_task_cost_and_generator_source(self):
        p={'planner':{'horizon':5,'block_lengths':[2,3],'candidate_count':128},
           'cost':{'stage_position':1.},'task':{'position_tolerance_m':.005},'physics':{'action_dt_s':.1}}
        before=deepcopy(p);q=extended_config(p)
        self.assertEqual(p,before);self.assertEqual(q['planner']['horizon'],10)
        for key in ['physics','task','cost']:self.assertEqual(q[key],p[key])


if __name__=='__main__':unittest.main()
