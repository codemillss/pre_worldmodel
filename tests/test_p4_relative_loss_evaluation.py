"""Relative/gap estimand and validity tests without learned inference/physics."""
from copy import deepcopy
import json
import unittest

import numpy as np

from push_align.config import load_pilot_config
from push_align.data_intervention_evaluation import DevelopmentData, evaluate_development
from push_align.relative_loss_evaluation import (ARMS, PRIMARY_ENDPOINTS,
    compare_relative_arms, geometry_metrics, geometry_rows, supplement_development)


def fixture():
    identities, states, actions = [], [], []
    for f in range(12):
        for y in range(2):
            for s in range(2):
                family = f'p4-data-development-2026101302-f{f:03}'
                identities.append({'episode_id': f'{family}-Y{y}-S{s}', 'family_id': family,
                    'family_rank': f, 'geometry': 'side_pass', 'y_condition': f'Y{y}', 'stop_condition': f'S{s}'})
                a = np.tile([0., .01+.01*y], (20, 1)).astype(np.float64)
                a[[5,7,9,11,13] if s == 0 else [10,11,12,13,14]] = 0
                x = np.zeros((21,10), np.float64); x[:,6] = -.08
                x[0,9] = .01+.01*y
                for t in range(20):
                    x[t+1] = x[t]; x[t+1,6:8] += .1*a[t]; x[t+1,8:10] = a[t]
                states.append(x); actions.append(a)
    return DevelopmentData(tuple(identities), np.stack(states), np.stack(actions),
                           np.full((48,20),'noncontact'), {})


def perfect(state, action):
    delta = np.zeros_like(state); delta[:,6:8] = .1*action; delta[:,8:10] = action-state[:,8:10]
    return delta


class RelativeLossEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.config, self.data = load_pilot_config(), fixture()
        self.legacy, self.arrays = evaluate_development(perfect, self.data, self.config)
        self.result = supplement_development(self.data, self.legacy, self.arrays, self.config)

    def test_relative_metric_is_relative_error_not_sum_individual_norms(self):
        actual = self.data.states[0,:1].copy(); predicted = actual.copy()
        predicted[:,0:2] += [.002,-.003]; predicted[:,6:8] += [.002,-.003]
        m = geometry_metrics(predicted, actual, self.config)
        self.assertAlmostEqual(m['relative_xy_m'][0], 0., places=15)
        self.assertAlmostEqual(m['gap_absolute_m'][0], 0., places=15)
        self.assertGreater(m['object_xy_m'][0], 0)

    def test_gap_and_relative_xy_are_not_interchangeable(self):
        actual = self.data.states[0,:1].copy(); predicted = actual.copy(); predicted[:,2] += .2
        m = geometry_metrics(predicted, actual, self.config)
        self.assertEqual(m['relative_xy_m'][0], 0)
        self.assertGreater(m['gap_absolute_m'][0], 0)

    def test_contact_y1_first5_deduplicates_s_and_retains12families(self):
        phase = self.result['primary_membership']['contact_Y1_teacher_first5']
        self.assertEqual(len(phase), 12)
        self.assertTrue(all(row['attempted_rows'] == 5 for row in phase.values()))
        self.assertEqual(self.result['Y_heads_teacher_conditioned']['attempted'], 120)
        self.assertTrue(self.result['duplicate_S1_heads_excluded'])

    def test_stop_y0y1_is_equal_weighted_with_actual_state10(self):
        phase = self.result['primary_membership']['stop_S1_free_K5']
        self.assertTrue(all(row['attempted_rows'] == 2 for row in phase.values()))
        for metrics in self.result['primary']['stop_S1_free_K5'].values():
            self.assertTrue(all(abs(v) < 1e-15 for v in metrics.values()))
        self.assertEqual(self.result['S1_free_stop']['attempted'], 24)
        self.assertEqual(self.result['S1_teacher_conditioned_stop']['attempted'], 120)

    def test_actual_regime_subgroups_partition_y_heads_without_duplicate_s(self):
        groups=self.result['Y_heads_teacher_conditioned']['actual_contact_regimes']
        for y in ('Y0','Y1'):
            self.assertEqual(sum(g['attempted'] for g in groups[y].values()),60)
            self.assertEqual(groups[y]['noncontact']['valid'],60)
            self.assertEqual(groups[y]['transition']['attempted'],0)
            self.assertIsNone(groups[y]['transition']['metrics_valid_only_descriptive']['gap_absolute_m'])
            self.assertIn('no subgroup CI',groups[y]['noncontact']['scope'])
        definition=self.result['primary_phase_definitions']['contact_Y1_teacher_first5']
        self.assertIn('not a contact-only',definition)

    def test_regime_subgroup_preserves_finite_invalid_diagnosis_without_primary_rescue(self):
        regimes=self.data.regimes.copy(); regimes[6,1]='transition'; regimes[7,1]='transition'
        data=DevelopmentData(self.data.identities,self.data.states,self.data.actions,regimes,{})
        legacy,arrays=evaluate_development(perfect,data,self.config)
        arrays['head_valid'][16]=False; arrays['head_failure_reasons'][16]='test_invalid'
        result=supplement_development(data,legacy,arrays,self.config)
        group=result['Y_heads_teacher_conditioned']['actual_contact_regimes']['Y1']['transition']
        self.assertEqual((group['attempted'],group['valid'],group['finite_invalid_diagnostic_rows']),(1,0,1))
        self.assertIsNone(group['metrics_valid_only_descriptive']['relative_xy_m'])
        self.assertIsNotNone(group['metrics_finite_including_invalid_diagnostic_only']['relative_xy_m'])
        family=self.data.identities[6]['family_id']
        self.assertIsNone(result['primary']['contact_Y1_teacher_first5'][family]['relative_xy_m'])
        json.dumps(result,allow_nan=False)

    def test_selected_known_failures_use_canonical_family_step_and_y(self):
        selected = self.result['selected_prior_failures']
        contact = selected['contact_state1_to2']
        self.assertEqual((contact['family_rank'], contact['y_condition'], contact['core_step']), (1,'Y1',2))
        self.assertEqual(len(selected['stop_state10_to15']), 2)
        for case in selected['stop_state10_to15'].values():
            self.assertEqual(case['identity']['y_condition'],'Y0')
            self.assertEqual(len(case['free']),5); self.assertEqual(len(case['teacher']),5)

    def test_finite_invalid_state_stays_diagnostic_and_invalid_family(self):
        arrays = {k:v.copy() for k,v in self.arrays.items()}
        row = 5*3+1  # f001 Y1 S0 local core state1-to2.
        arrays['head_valid'][row] = False; arrays['head_failure_reasons'][row] = 'test_invalid'
        result = supplement_development(self.data,self.legacy,arrays,self.config)
        family = self.data.identities[6]['family_id']
        self.assertIsNone(result['primary']['contact_Y1_teacher_first5'][family]['relative_xy_m'])
        diagnostic = result['Y_heads_teacher_conditioned']['rows'][row]
        self.assertTrue(diagnostic['finite_invalid_diagnostic_only'])
        self.assertEqual(result['Y_heads_teacher_conditioned']['invalid'],1)
        self.assertIsNotNone(diagnostic['metrics']['relative_xy_m'])

    def test_nonfinite_missing_rows_jsonnull_and_never_imputed(self):
        actual = self.data.states[0,:2].copy(); pred = actual.copy(); pred[1] = np.nan
        result = geometry_rows(pred,actual,np.array([True,False]),[{},{}],self.config)
        self.assertEqual(result['missing_rows'],1)
        self.assertTrue(all(v is None for v in result['rows'][1]['metrics'].values()))
        json.dumps(result,allow_nan=False)

    def test_valid_nonfinite_state_rejected(self):
        actual = self.data.states[0,:1].copy(); pred = actual.copy(); pred[0] = np.nan
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            geometry_rows(pred,actual,np.array([True]),[{}],self.config)

    def test_later_failure_preserves_earlier_stop_validity(self):
        arrays = {k:v.copy() for k,v in self.arrays.items()}
        arrays['stop_valid'][0] = False; arrays['stop_failure_steps'][0] = 3
        arrays['stop_states'][0,4:] = np.nan
        result = supplement_development(self.data,self.legacy,arrays,self.config)
        self.assertEqual(result['S1_free_stop_K1_to_K5']['2']['valid'],24)
        self.assertEqual(result['S1_free_stop_K1_to_K5']['3']['valid'],23)
        self.assertEqual(result['S1_free_stop_K1_to_K5']['3']['finite_invalid_diagnostic_rows'],1)
        self.assertEqual(result['S1_free_stop_K1_to_K5']['4']['missing_rows'],1)

    def test_one_invalid_family_holds_continuous_compare_even_when_fixed(self):
        a,b = deepcopy(self.result),deepcopy(self.result)
        family = a['families'][0]
        a['primary_membership']['contact_Y1_teacher_first5'][family]['all_rows_valid'] = False
        a['primary']['contact_Y1_teacher_first5'][family] = dict.fromkeys(a['primary']['contact_Y1_teacher_first5'][family])
        comparison,_ = compare_relative_arms({'C0':a,'R1':b})
        phase = 'contact_Y1_teacher_first5'
        self.assertEqual(comparison['effect_decisions'][phase]['status'],'HOLD_INVALID_PRIMARY')
        self.assertEqual(comparison['validity_changes'][phase]['C0_invalid_R1_valid']['count'],1)
        for metric in PRIMARY_ENDPOINTS:
            row = comparison['contrasts'][phase][metric]
            self.assertEqual((row['attempted_families'],row['valid_families']),(12,11))
            self.assertIsNone(row['mean_difference']); self.assertEqual(len(row['family_differences']),12)

    def test_exact_ties_inconclusive_not_epsilon_win(self):
        comparison,draws = compare_relative_arms({arm:deepcopy(self.result) for arm in ARMS})
        self.assertEqual(draws.shape,(10000,12))
        for phase in comparison['effect_decisions']:
            self.assertEqual(comparison['effect_decisions'][phase]['status'],'INCONCLUSIVE')
            self.assertEqual(comparison['contrasts'][phase]['relative_xy_m']['exact_float64_ties'],12)
            self.assertEqual(comparison['contrasts'][phase]['relative_xy_m']['bootstrap_percentile_95'],[0,0])

    def test_directional_support_requires_both_primary_metrics(self):
        a,b = deepcopy(self.result),deepcopy(self.result)
        for phase in a['primary']:
            for family in a['families']:
                for metric in PRIMARY_ENDPOINTS:
                    a['primary'][phase][family][metric]=.002; b['primary'][phase][family][metric]=.001
        c,_ = compare_relative_arms({'C0':a,'R1':b})
        self.assertTrue(all(d['status']=='DIRECTIONAL_SUPPORT' for d in c['effect_decisions'].values()))
        for family in a['families']: b['primary']['stop_S1_free_K5'][family]['gap_absolute_m']=.003
        c,_ = compare_relative_arms({'C0':a,'R1':b})
        self.assertEqual(c['effect_decisions']['stop_S1_free_K5']['status'],'MIXED')

    def test_bootstrap_matches_paired_family_resampling(self):
        a,b=deepcopy(self.result),deepcopy(self.result)
        for n,family in enumerate(a['families']): b['primary']['stop_S1_free_K5'][family]['relative_xy_m']=n*.001
        c,draws=compare_relative_arms({'C0':a,'R1':b})
        expected=np.percentile((np.arange(12)*.001)[draws].mean(axis=1),[2.5,97.5]).tolist()
        self.assertEqual(c['contrasts']['stop_S1_free_K5']['relative_xy_m']['bootstrap_percentile_95'],expected)

    def test_arm_missing_and_pair_reorder_rejected(self):
        with self.assertRaisesRegex(ValueError,'Both'):
            compare_relative_arms({'C0':deepcopy(self.result)})
        b=deepcopy(self.result); b['families'].reverse()
        with self.assertRaisesRegex(ValueError,'pairing'):
            compare_relative_arms({'C0':deepcopy(self.result),'R1':b})

    def test_actual_sample_replacement_rejected(self):
        arrays={k:v.copy() for k,v in self.arrays.items()}; arrays['actual_states'][0,0,0] += .01
        with self.assertRaisesRegex(ValueError,'Actual'):
            supplement_development(self.data,self.legacy,arrays,self.config)


if __name__=='__main__': unittest.main()
