"""Synthetic candidate ledgers only: no family generation/model/physics."""
from copy import deepcopy
import io
import json
import unittest
from unittest.mock import patch

import numpy as np

from push_align.candidate_diagnostics import compare_candidates
from push_align.final_e1_statistics import analyze_E1, MODELS, INDICATORS


def fixture(*, actual_change=None, prediction_change=None, large=False):
    specs, rows = [], []
    chosen = {'R':0, 'S1-11':3, 'S1-23':4, 'S1-47':5, 'SK-11':1, 'SK-23':2, 'SK-47':3}
    for stratum_index, stratum in enumerate(('translation','rotation','offset')):
        for rank in range(2):
            family = f'fixture-{stratum}-{rank}'
            family_index = stratum_index*2+rank
            for tick in (0,29):
                spec = {'snapshot_id': family+f'/G/tick-{tick:03d}', 'family_id':family,
                    'episode_index':stratum_index*4+rank, 'stratum':stratum, 'controller_id':'G',
                    'tick':tick, 'candidates':16, 'horizon':5, 'physical_repeats':3}
                specs.append(spec)
                means = np.linspace(0.,1e308,16) if large else (1.+2*np.arange(16,dtype=np.float64))*(1. if tick==0 else 2.)
                actual = np.repeat(means[:,None],3,axis=1)
                valid = np.ones((16,3),bool)
                if actual_change:
                    actual_change(family_index,tick,actual,valid)
                local, predictions = [], []
                for model in MODELS:
                    predicted = means+3.
                    predicted[15 if large else chosen[model]] = 0.
                    predicted_valid = np.ones(16,bool)
                    if prediction_change:
                        prediction_change(model,family_index,tick,predicted,predicted_valid)
                    row = compare_candidates(predicted,predicted_valid,actual,valid)
                    row.update(snapshot_id=spec['snapshot_id'],family_id=family,stratum=stratum,tick=tick,
                        model=model,input_hash='b'*64,candidate_hash='a'*64)
                    predictions.append((predicted,predicted_valid))
                    local.append(row)
                common = valid.all(axis=1)
                for _, eligible in predictions:
                    common &= eligible
                for row,(predicted,_) in zip(local,predictions):
                    row['all7_common_valid_cost_error'] = {'denominator':int(common.sum()),
                        'MAE':float(np.abs(predicted[common]-actual[common].mean(axis=1)).mean()) if common.any() and not large else
                              (0. if common.any() else None)}
                rows.extend(local)
    return rows,specs


def contrast(summary,left='SK',right='S1'):
    return next(row for row in summary['comparisons'] if row['left']==left and row['right']==right)


class FinalE1StatisticsTests(unittest.TestCase):
    def test_known_paired_effect_averages_ticks_then_seed_inside_six_families(self):
        rows,specs=fixture()
        summary,arrays=analyze_E1(rows,specs)
        self.assertEqual(summary['status'],'COMPLETE_EXPECTED_E1_DENOMINATORS')
        self.assertEqual(summary['model_set_count'],84)
        self.assertEqual(summary['snapshot_count'],12)
        self.assertEqual(summary['independent_family_count'],6)
        self.assertEqual(summary['shared_physical_repeat_attempts'],576)
        self.assertAlmostEqual(summary['seed_groups']['S1']['selection_loss']['complete_family_mean'],12.)
        self.assertAlmostEqual(summary['seed_groups']['SK']['selection_loss']['complete_family_mean'],6.)
        metric=contrast(summary)['metrics']['selection_loss']
        self.assertAlmostEqual(metric['difference_mean'],-6.)
        np.testing.assert_allclose(metric['percentile95_interval'],[-6.,-6.],rtol=1e-14)
        self.assertEqual(metric['eligible_family_denominator'],6)
        self.assertEqual(metric['bootstrap_valid_replicates'],2000)
        for seed in (11,23,47):
            paired=contrast(summary,f'SK-{seed}',f'S1-{seed}')['metrics']['selection_loss']
            self.assertAlmostEqual(paired['difference_mean'],-6.)
        self.assertEqual(arrays['family_draw_indices'].shape,(2000,6))
        np.testing.assert_allclose(arrays['family_S1_selection_loss'],12.,rtol=1e-14)
        self.assertEqual(summary['models']['R']['model_set_count'],12)
        self.assertEqual(summary['seed_groups']['S1']['model_set_count'],36)

    def test_common_stratified_draws_reproducible_under_row_and_spec_order(self):
        rows,specs=fixture()
        first,arrays=analyze_E1(rows,specs)
        second,other=analyze_E1(list(reversed(rows)),list(reversed(specs)))
        self.assertEqual(first,second)
        for key in arrays:
            if arrays[key].dtype.kind in 'US':
                self.assertTrue(np.array_equal(arrays[key],other[key]),key)
            else:
                self.assertTrue(np.array_equal(arrays[key],other[key],equal_nan=True),key)
        families=first['bootstrap']['families_in_order']
        for start,stratum in zip((0,2,4),('translation','rotation','offset')):
            drawn={families[index] for index in arrays['family_draw_indices'][:,start:start+2].ravel()}
            self.assertEqual(drawn,set(first['bootstrap']['strata'][stratum]))

    def test_one_missing_tick_in_one_seed_excludes_whole_family_for_its_group_only(self):
        def missing(model,family,tick,predicted,valid):
            if model=='S1-11' and family==0 and tick==29:
                valid[:]=False;predicted[:]=np.inf
        rows,specs=fixture(prediction_change=missing)
        summary,arrays=analyze_E1(rows,specs)
        self.assertEqual(summary['original_model_set_populations']['no_selection'],{'numerator':1,'denominator':84})
        group=summary['seed_groups']['S1']
        self.assertEqual(group['selection_loss']['defined_model_sets'],35)
        self.assertEqual(group['selection_loss']['complete_family_denominator'],5)
        self.assertEqual(contrast(summary)['metrics']['selection_loss']['eligible_family_denominator'],5)
        self.assertEqual(contrast(summary,'SK-11','S1-11')['metrics']['selection_loss']['eligible_family_denominator'],5)
        self.assertEqual(contrast(summary,'SK-23','S1-23')['metrics']['selection_loss']['eligible_family_denominator'],6)
        self.assertEqual(np.isnan(arrays['family_S1_selection_loss']).sum(),1)
        self.assertTrue(np.all(arrays['SK_minus_S1_selection_loss_valid']))
        self.assertEqual(summary['model_set_count'],84)

    def test_zero_eligible_bootstrap_replicates_are_saved_instead_of_filled_or_deleted(self):
        def missing(model,family,tick,predicted,valid):
            if model.startswith('S1') and family!=0:
                valid[:]=False;predicted[:]=np.inf
        rows,specs=fixture(prediction_change=missing)
        summary,arrays=analyze_E1(rows,specs)
        metric=contrast(summary,'S1','R')['metrics']['selection_loss']
        self.assertEqual(metric['eligible_family_denominator'],1)
        self.assertEqual(metric['original_family_denominator'],6)
        self.assertGreater(metric['bootstrap_unavailable_replicates'],0)
        self.assertLess(metric['bootstrap_unavailable_replicates'],2000)
        self.assertEqual(metric['bootstrap_valid_replicates']+metric['bootstrap_unavailable_replicates'],2000)
        valid=arrays['S1_minus_R_selection_loss_valid']
        positions=arrays['S1_minus_R_selection_loss_eligible_resampled_positions']
        self.assertTrue(np.array_equal(valid,positions>0))
        self.assertTrue(np.isnan(arrays['S1_minus_R_selection_loss'][~valid]).all())
        np.testing.assert_allclose(arrays['S1_minus_R_selection_loss'][valid],12.,rtol=1e-14)
        self.assertEqual(summary['original_model_set_populations']['no_selection']['numerator'],30)

    def test_all_undefined_selection_losses_keep_all84_and_explicit_null_intervals(self):
        def missing(model,family,tick,predicted,valid):
            valid[:]=False;predicted[:]=np.inf
        rows,specs=fixture(prediction_change=missing)
        summary,arrays=analyze_E1(rows,specs)
        self.assertEqual(summary['original_model_set_populations']['model_all_invalid'],{'numerator':84,'denominator':84})
        metric=contrast(summary)['metrics']['selection_loss']
        self.assertEqual(metric['eligible_family_denominator'],0)
        self.assertIsNone(metric['difference_mean'])
        self.assertIsNone(metric['percentile95_interval'])
        self.assertEqual(metric['bootstrap_unavailable_replicates'],2000)
        self.assertFalse(arrays['SK_minus_S1_selection_loss_valid'].any())
        json.dumps(summary,allow_nan=False)

    def test_actual_all_invalid_deduplicates_shared12_snapshot_population(self):
        def invalid(family,tick,actual,valid):
            if family==0 and tick==0:
                valid[:]=False;actual[:]=np.inf
        rows,specs=fixture(actual_change=invalid)
        summary,_=analyze_E1(rows,specs)
        self.assertEqual(summary['original_model_set_populations']['actual_all_invalid'],{'numerator':7,'denominator':84})
        self.assertEqual(summary['shared_actual_snapshot_populations']['actual_all_invalid'],{'numerator':1,'denominator':12})
        self.assertEqual(summary['models']['R']['conditions_on_original_model_set_population']['selected_actual_invalid']['numerator'],1)
        self.assertEqual(contrast(summary)['metrics']['selection_loss']['eligible_family_denominator'],5)

    def test_validity_change_and_stable_invalid_selection_are_separate_from_no_selection(self):
        for changed in (False,True):
            def invalid(family,tick,actual,valid):
                if family==0 and tick==0:
                    valid[0]=[True,False,True] if changed else [False,False,False]
                    actual[0,~valid[0]]=np.inf
            rows,specs=fixture(actual_change=invalid)
            summary,_=analyze_E1(rows,specs)
            key='selected_actual_validity_changed' if changed else 'selected_actual_invalid'
            self.assertEqual(summary['models']['R']['conditions_on_original_model_set_population'][key]['numerator'],1)
            self.assertEqual(summary['original_model_set_populations']['no_selection']['numerator'],0)
            self.assertEqual(summary['shared_actual_snapshot_populations']['any_validity_changed']['numerator'],int(changed))
            self.assertEqual(contrast(summary,'S1','R')['metrics']['selection_loss']['eligible_family_denominator'],5)
            self.assertEqual(contrast(summary)['metrics']['selection_loss']['eligible_family_denominator'],6)

    def test_undefined_rank_and_inconclusive_order_are_not_new_failure_labels(self):
        def tied(model,family,tick,predicted,valid):
            predicted[:]=0.
        rows,specs=fixture(prediction_change=tied)
        summary,_=analyze_E1(rows,specs)
        self.assertEqual(summary['original_model_set_populations']['model_all_invalid']['numerator'],0)
        self.assertEqual(summary['original_model_set_populations']['no_selection']['numerator'],0)
        self.assertEqual(summary['models']['R']['selection_loss_ordering_counts'],{'INCONCLUSIVE':12})
        rho=summary['models']['R']['descriptive_cost_and_rank']['rank_rho']
        self.assertEqual(rho['defined_model_sets'],0)
        self.assertIsNone(rho['mean_of_defined_model_set_values'])
        self.assertTrue(all(n==16 for n in rho['candidate_denominator_by_model_set'].values()))

    def test_model_specific_cost_mask_denominators_are_preserved_and_descriptive(self):
        def absent(model,family,tick,predicted,valid):
            if model=='S1-11':
                valid[15]=False;predicted[15]=np.inf
        rows,specs=fixture(prediction_change=absent)
        summary,_=analyze_E1(rows,specs)
        first=summary['models']['S1-11']['descriptive_cost_and_rank']['cost_MAE']
        second=summary['models']['SK-11']['descriptive_cost_and_rank']['cost_MAE']
        self.assertEqual(set(first['candidate_denominator_by_model_set'].values()),{15})
        self.assertEqual(set(second['candidate_denominator_by_model_set'].values()),{16})
        self.assertIn('masks may differ',first['scope'])
        self.assertTrue(first['not_a_common_population_superiority_test'])
        self.assertEqual(set(summary['models']['SK-11']['descriptive_cost_and_rank']['all7_common_cost_MAE']['candidate_denominator_by_model_set'].values()),{15})

    def test_missing_duplicate_unknown_models_and_frozen_specs_are_rejected(self):
        rows,specs=fixture()
        for altered in (rows[:-1],rows[:-1]+[rows[0]], [{**rows[0],'model':'G'},*rows[1:]]):
            with self.assertRaises(ValueError):analyze_E1(altered,specs)
        for key,value in [('tick',1),('candidates',32),('physical_repeats',True),('episode_index',True),('stratum','unknown')]:
            altered=deepcopy(specs);altered[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):analyze_E1(rows,altered)
        with self.assertRaises(ValueError):analyze_E1(rows,specs[:-1]+[specs[0]])

    def test_nonfinite_negative_or_miscounted_diagnostics_are_rejected(self):
        rows,specs=fixture()
        cases=[('selection_loss',float('nan')),('selection_loss',-1.),('selection_loss_denominator',True),
               ('selected_index',False),('selection_status','infrastructure_error'),('model_all_invalid',True)]
        for key,value in cases:
            altered=deepcopy(rows);altered[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):analyze_E1(altered,specs)
        altered=deepcopy(rows);altered[0]['counts']['candidate_count']=15
        with self.assertRaises(ValueError):analyze_E1(altered,specs)

    def test_all_models_must_share_input_candidates_and_actual_future_labels(self):
        rows,specs=fixture()
        for key in ('candidate_hash','input_hash','actual_records'):
            altered=deepcopy(rows)
            if key=='actual_records':
                altered[0][key][0]['mean_cost']+=1.
            else:
                altered[0][key]='c'*64
            with self.subTest(key=key),self.assertRaises(ValueError):analyze_E1(altered,specs)

    def test_json_and_npz_are_safe_and_input_rows_unchanged(self):
        rows,specs=fixture()
        before=json.dumps([rows,specs],sort_keys=True)
        summary,arrays=analyze_E1(rows,specs,replicates=20,seed=123)
        self.assertEqual(before,json.dumps([rows,specs],sort_keys=True))
        json.dumps(summary,allow_nan=False)
        self.assertTrue(all(array.dtype!=object for array in arrays.values()))
        stream=io.BytesIO();np.savez_compressed(stream,**arrays);stream.seek(0)
        with np.load(stream,allow_pickle=False) as restored:
            self.assertEqual(set(restored.files),set(arrays))
            self.assertEqual(restored['family_draw_indices'].shape,(20,6))

    def test_large_finite_losses_do_not_overflow_family_or_bootstrap_means(self):
        rows,specs=fixture(large=True)
        summary,arrays=analyze_E1(rows,specs)
        self.assertEqual(summary['seed_groups']['S1']['selection_loss']['complete_family_mean'],1e308)
        self.assertEqual(contrast(summary)['metrics']['selection_loss']['percentile95_interval'],[0.,0.])
        self.assertTrue(np.isfinite(arrays['family_S1_selection_loss']).all())
        json.dumps(summary,allow_nan=False)

    def test_invalid_bootstrap_arguments_are_rejected(self):
        rows,specs=fixture()
        for replicates,seed in ((True,7711),(1,7711),(20,True),(20,-1)):
            with self.subTest(replicates=replicates,seed=seed),self.assertRaises(ValueError):
                analyze_E1(rows,specs,replicates,seed)

    def test_existing_writer_synthetic_ledger_schema_integrates_without_real_runtime(self):
        from test_p5_final_e1 import FinalE1Test
        import p5_final_e1
        fixture_case=FinalE1Test(methodName='runTest')
        fixture_case.setUp()
        self.addCleanup(fixture_case.doCleanups)
        with (fixture_case.patches(),
              patch('push_align.environment.PlanarPushEnv.__init__',side_effect=AssertionError('no physical environment'))):
            recorded=p5_final_e1.run_e1(fixture_case.run,fixture_case.manifest,fixture_case.callback)
        self.assertEqual(recorded['status'],'COMPLETE_RECORDED')
        ledger=json.loads((fixture_case.run/'E1/comparison_ledger.json').read_text())
        summary,arrays=analyze_E1(ledger,fixture_case.manifest['E1'])
        self.assertEqual(summary['model_set_count'],84)
        self.assertEqual(summary['snapshot_count'],12)
        self.assertEqual(arrays['family_draw_indices'].shape,(2000,6))


if __name__=='__main__':unittest.main()
