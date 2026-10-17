"""Failure-analysis guards, independent of new physical experiments.

The archived 300-interval witness is read, never executed or modified here.
Small copied prefixes test equality and adverse mutations cheaply; a passing
comparison of those prefixes is not evidence of reproducing the full failure.
"""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from push_align.config import ROOT
from push_align.planning import CostBatch, PredictionBatch, TaskSpec

scripts_path = str(ROOT / 'scripts')
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

from p4_failure_analysis import (
    compare_closed_loop_reproduction,
    hypotheses_for_snapshot,
    select_diagnostic_subset,
    validate_budget,
    validate_matched_pair,
)


class DiagnosticSubsetTest(unittest.TestCase):
    def test_reserved_and_all_three_model_choices_survive_deterministic_fill(self):
        selected = [20, 40, 60]
        result = list(select_diagnostic_subset(selected))
        self.assertEqual(result, list(range(13)) + [20, 40, 60])
        self.assertTrue(set(range(8)) <= set(result))
        self.assertTrue(set(selected) <= set(result))
        self.assertEqual(len(set(result)), 16)
        self.assertEqual(selected, [20, 40, 60])

    def test_duplicate_reserved_and_invalid_model_choices_keep_sixteen_distinct_rows(self):
        for choices in ([None, None, None], [1, 1, 1], [9, 9, None]):
            with self.subTest(choices=choices):
                self.assertEqual(list(select_diagnostic_subset(choices)), list(range(16)))

    def test_model_order_does_not_change_population_or_sorted_output(self):
        a = list(select_diagnostic_subset([127, 64, 20]))
        b = list(select_diagnostic_subset([20, 127, 64]))
        self.assertEqual(a, b)
        self.assertEqual(a, sorted(a))
        self.assertEqual(a, list(range(13)) + [20, 64, 127])

    def test_insufficient_size_never_drops_a_model_choice_or_reserved_candidate(self):
        with self.assertRaises(ValueError):
            select_diagnostic_subset([20, 40, 60], size=10)
        self.assertEqual(list(select_diagnostic_subset([20, 40, 60], size=11)),
                         list(range(8)) + [20, 40, 60])

    def test_bool_float_and_out_of_range_model_indices_are_rejected(self):
        for value in (True, False, 3.0, '3', -1, 128, np.int64(3)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                select_diagnostic_subset([0, value, None])

    def test_malformed_population_dimensions_are_not_coerced(self):
        for kwargs in ({'candidate_count': True}, {'candidate_count': 128.0},
                       {'candidate_count': 0}, {'candidate_count': 15},
                       {'size': True}, {'size': 16.0}, {'size': 0},
                       {'reserved_count': True}, {'reserved_count': 8.0},
                       {'reserved_count': -1}, {'reserved_count': 17}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                select_diagnostic_subset([None, None, None], **kwargs)

    def test_subset_is_not_the_entire_original_candidate_population(self):
        # This explicitly preserves the denominator needed for bounded claims.
        result = list(select_diagnostic_subset([20, 40, 60]))
        self.assertEqual(len(result), 16)
        self.assertEqual(len(set(range(128)) - set(result)), 112)
        self.assertNotIn(127, result)


class ClosedLoopReproductionComparisonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archived = ROOT / 'runs/p3-05-20261010-01/full_resource/MLP-11'
        cls.steps = [json.loads(line) for line in
                     (cls.archived / 'steps.jsonl').read_text().splitlines()[:2]]
        cls.summary = json.loads((cls.archived / 'summary.json').read_text())
        with np.load(cls.archived / 'raw_execution.npz', allow_pickle=False) as raw:
            cls.raw = {key: raw[key].copy() for key in raw.files}
        with gzip.open(cls.archived / 'raw_execution.json.gz', 'rt', encoding='utf-8') as f:
            cls.raw_json = json.load(f)
        cls.raw = {key: value[:400] if key.startswith('substep_') else
                   value if key in ('initial_integration_state', 'diagnostic_nonfinite_values') else
                   value[:2] for key, value in cls.raw.items()}
        cls.raw_json['diagnostics'] = cls.raw_json['diagnostics'][:2]
        cls.summary['transitions'] = cls.summary['valid_transitions'] = 2
        cls.summary['termination'] = 'development_budget'
        cls.summary['task'] = deepcopy(cls.steps[-1]['task'])

    def setUp(self):
        self.directory = TemporaryDirectory(prefix='p4-failure-comparison-')
        self.addCleanup(self.directory.cleanup)
        self.source, self.replay = (Path(self.directory.name) / x for x in ('source', 'replay'))
        self.write_folder(self.source)
        self.write_folder(self.replay)

    def write_folder(self, path, *, raw=None, raw_json=None, steps=None, summary=None):
        path.mkdir(exist_ok=True)
        np.savez_compressed(path / 'raw_execution.npz',
                            **(self.raw if raw is None else raw))
        with gzip.open(path / 'raw_execution.json.gz', 'wt', encoding='utf-8') as f:
            json.dump(self.raw_json if raw_json is None else raw_json, f, allow_nan=False)
        (path / 'steps.jsonl').write_text(''.join(json.dumps(row, allow_nan=False) + '\n'
            for row in (self.steps if steps is None else steps)))
        (path / 'summary.json').write_text(json.dumps(
            self.summary if summary is None else summary, allow_nan=False))

    def result(self):
        result = compare_closed_loop_reproduction(self.source, self.replay)
        self.assertIs(type(result['pass']), bool)
        self.assertIsInstance(result['checks'], dict)
        self.assertTrue(result['checks'])
        return result

    def assert_mismatch(self):
        result = self.result()
        self.assertFalse(result['pass'])
        self.assertFalse(all(result['checks'].values()))

    def test_real_archived_full_failure_compares_with_itself_without_execution(self):
        result = compare_closed_loop_reproduction(self.archived, self.archived)
        self.assertTrue(result['pass'])
        self.assertTrue(all(result['checks'].values()))
        self.assertEqual(json.loads((self.archived / 'summary.json').read_text())['termination'],
                         'timeout')

    def test_elapsed_wall_timing_model_labels_and_source_bindings_are_not_behavior(self):
        steps = deepcopy(self.steps)
        for row in steps:
            row['execution_wall_time_s'] += 10.
            row['plan']['planning_core_wall_time_s'] += 20.
            row['plan']['source_manifest_sha256'] = 'f' * 64
            row['plan']['model_id'] = 'replay-label'
            row['plan']['model_hash'] = 'e' * 64
        summary = deepcopy(self.summary)
        summary['model_id'] = 'replay-label'
        summary['resources']['wall_time_s'] += 100.
        self.write_folder(self.replay, steps=steps, summary=summary)
        self.assertTrue(self.result()['pass'])

    def test_every_raw_npz_field_is_exact_not_tolerance_rounded(self):
        for key in self.raw:
            with self.subTest(key=key):
                changed = {name: value.copy() for name, value in self.raw.items()}
                if changed[key].size:
                    flat = changed[key].reshape(-1)
                    flat[0] = np.nextafter(flat[0], np.inf)
                else:
                    changed[key] = np.array([np.nan], np.float64)
                self.write_folder(self.replay, raw=changed)
                self.assert_mismatch()

    def test_missing_raw_field_and_interior_contact_diagnostic_are_not_hidden_by_endpoint(self):
        raw = {key: value.copy() for key, value in self.raw.items()
               if key != 'substep_times'}
        self.write_folder(self.replay, raw=raw)
        self.assert_mismatch()
        raw_json = deepcopy(self.raw_json)
        raw_json['diagnostics'][0]['samples'][1]['state'][0] += .0001
        self.write_folder(self.replay, raw_json=raw_json)
        self.assert_mismatch()

    def test_full_engine_snapshot_and_nonfinite_path_provenance_must_match(self):
        for key in ('initial_snapshot', 'diagnostic_nonfinite_paths'):
            with self.subTest(key=key):
                changed = deepcopy(self.raw_json)
                if key == 'initial_snapshot':
                    changed[key]['warning_counts'][0] += 1
                else:
                    changed[key] = ['/diagnostics/0/missing']
                self.write_folder(self.replay, raw_json=changed)
                self.assert_mismatch()

    def test_requested_applied_and_observed_commands_are_separately_checked(self):
        for key in ('requested_action', 'applied_action', 'state_before', 'state_after'):
            with self.subTest(key=key):
                steps = deepcopy(self.steps)
                steps[1][key][0] += .001
                self.write_folder(self.replay, steps=steps)
                self.assert_mismatch()

    def test_plan_hash_choice_cost_and_causal_context_changes_are_detected(self):
        for key in ('candidate_hash', 'prediction_hash', 'delta_hash', 'selected_index',
                    'selected_actions', 'selected_cost', 'planner_context_before'):
            with self.subTest(key=key):
                steps = deepcopy(self.steps)
                plan = steps[1]['plan']
                if key.endswith('_hash'):
                    plan[key] = '0' * 64
                elif key == 'selected_index':
                    plan[key] = (plan[key] + 1) % 128
                elif key == 'selected_actions':
                    plan[key][0][0] += .001
                elif key == 'selected_cost':
                    plan[key] = np.nextafter(plan[key], np.inf).item()
                else:
                    plan[key]['previous_applied_action'][0] += .001
                self.write_folder(self.replay, steps=steps)
                self.assert_mismatch()

    def test_same_pose_loss_cannot_hide_different_terminal_or_physical_validity(self):
        summary = deepcopy(self.summary)
        summary['termination'] = 'ready'
        summary['task']['phase'] = 'ready'
        self.write_folder(self.replay, summary=summary)
        self.assert_mismatch()
        steps = deepcopy(self.steps)
        steps[1]['physical_valid'] = False
        self.write_folder(self.replay, steps=steps)
        self.assert_mismatch()

    def test_shorter_replay_is_not_equal_even_if_final_summary_claims_same_result(self):
        self.write_folder(self.replay, steps=self.steps[:-1])
        self.assert_mismatch()


class FailureAnalysisBudgetTest(unittest.TestCase):
    def setUp(self):
        self.registry = json.loads((ROOT / 'configs/p4_failure_analysis.v1.json').read_text())
        self.gib = 1024 ** 3
        self.prior = self.registry['budget']['prior_consumed_bytes']
        self.free = self.registry['budget']['estimated_stage_bytes'] + 2 * self.gib

    def test_registered_stage_preserves_shared_pool_and_exact_free_reserve(self):
        original = deepcopy(self.registry)
        result = validate_budget(self.registry, self.free, self.prior)
        self.assertTrue(result['pass'])
        self.assertTrue(all(result['checks'].values()))
        self.assertEqual(self.registry, original)
        self.assertEqual(result['closed_loop_max_intervals'], 900)
        self.assertEqual(result['additional_E1_max_intervals'], 480)

    def test_exact_pool_and_free_boundaries_are_accepted_without_extra_allowance(self):
        self.registry['budget']['estimated_stage_bytes'] = self.gib - self.prior
        self.free = self.registry['budget']['estimated_stage_bytes'] + 2 * self.gib
        self.assertTrue(validate_budget(self.registry, self.free, self.prior)['pass'])

    def test_one_byte_short_reserve_or_over_pool_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_budget(self.registry, self.free - 1, self.prior)
        self.registry['budget']['estimated_stage_bytes'] = self.gib - self.prior + 1
        with self.assertRaises(ValueError):
            validate_budget(self.registry, 4 * self.gib, self.prior)

    def test_changed_snapshot_horizon_repeats_subset_or_model_population_is_rejected(self):
        variants = {'source_ticks': [49, 199], 'N128_horizon': 10,
                    'physical_repeats': 2, 'subset_size': 15,
                    'models': ['MLP-11', 'SK-11', 'S1-11']}
        for key, value in variants.items():
            with self.subTest(key=key):
                changed = deepcopy(self.registry)
                changed[key] = value
                with self.assertRaises(ValueError):
                    validate_budget(changed, self.free, self.prior)

    def test_integer_budget_and_population_fields_cannot_be_floats_or_booleans(self):
        for key in ('closed_loop_cells', 'closed_loop_max_intervals',
                    'additional_E1_max_intervals', 'P4_pool_limit_bytes',
                    'estimated_stage_bytes', 'prior_consumed_bytes'):
            with self.subTest(key=key):
                changed = deepcopy(self.registry)
                changed['budget'][key] = float(changed['budget'][key])
                with self.assertRaises(ValueError):
                    validate_budget(changed, self.free, self.prior)
        for key in ('N128_horizon', 'physical_repeats', 'subset_size'):
            with self.subTest(key=key):
                changed = deepcopy(self.registry)
                changed[key] = float(changed[key])
                with self.assertRaises(ValueError):
                    validate_budget(changed, self.free, self.prior)
        changed = deepcopy(self.registry)
        changed['source_ticks'] = [49., 99.]
        with self.assertRaises(ValueError):
            validate_budget(changed, self.free, self.prior)
        for free, prior in ((float(self.free), self.prior), (self.free, float(self.prior)),
                            (float('inf'), self.prior), (self.free, -1)):
            with self.subTest(free=free, prior=prior), self.assertRaises(ValueError):
                validate_budget(self.registry, free, prior)

    def test_prior_consumption_argument_cannot_disagree_with_bound_ledger(self):
        with self.assertRaises(ValueError):
            validate_budget(self.registry, self.free, self.prior - 1)


class MatchedLossContrastTest(unittest.TestCase):
    def setUp(self):
        folder = ROOT / 'runs/p4-02-20261012-01/seeds/seed-11'
        self.left = json.loads((folder / 'S1/checkpoints/update-1000/checkpoint.json').read_text())
        self.right = json.loads((folder / 'SK-lambda0.1/checkpoints/update-1000/checkpoint.json').read_text())

    def test_actual_final_seed_eleven_metadata_has_only_declared_loss_contrast(self):
        before = deepcopy((self.left, self.right))
        result = validate_matched_pair(self.left, self.right)
        self.assertTrue(result['pass'])
        self.assertTrue(all(result['checks'].values()))
        self.assertEqual((self.left, self.right), before)
        # Distinct final weights are expected, not an equality prerequisite.
        self.assertNotEqual(self.left['checkpoint_hash'], self.right['checkpoint_hash'])

    def test_common_origin_update_normalizer_and_schema_changes_break_matched_contrast(self):
        for key in ('common_origin', 'additional_updates', 'total_updates', 'normalizer_hash',
                    'learning_schema', 'config', 'model_spec', 'fork_budget_sha256',
                    'training_provenance', 'run_manifest_sha256'):
            with self.subTest(key=key):
                changed = deepcopy(self.right)
                changed[key] = {'tampered': True} if isinstance(changed[key], dict) else 'tampered'
                with self.assertRaises(ValueError):
                    validate_matched_pair(self.left, changed)

    def test_same_malformed_identifier_on_both_sides_is_not_equality_evidence(self):
        for key in ('seed', 'additional_updates', 'total_updates'):
            with self.subTest(key=key):
                left, right = deepcopy(self.left), deepcopy(self.right)
                left[key] = right[key] = float(left[key])
                with self.assertRaises(ValueError):
                    validate_matched_pair(left, right)

    def test_lambda_branch_or_s1_rollout_loss_cannot_be_silently_substituted(self):
        for side, key, value in (('left', 'rollout_weight', .1),
                                 ('right', 'rollout_weight', 1.),
                                 ('right', 'branch', 'SK-lambda1.0'),
                                 ('left', 'branch', 'MLP-11')):
            with self.subTest(side=side, key=key):
                left, right = deepcopy(self.left), deepcopy(self.right)
                (left if side == 'left' else right)[key] = value
                with self.assertRaises(ValueError):
                    validate_matched_pair(left, right)

    def test_missing_prerequisite_on_both_sides_is_not_a_match(self):
        for key in ('common_origin', 'normalizer_hash', 'learning_schema', 'training_provenance'):
            with self.subTest(key=key):
                left, right = deepcopy(self.left), deepcopy(self.right)
                del left[key], right[key]
                with self.assertRaises(ValueError):
                    validate_matched_pair(left, right)


class BoundedHypothesisInterpretationTest(unittest.TestCase):
    def setUp(self):
        physical = json.loads((ROOT /
            'runs/p3-05-20261010-01/manifest.json').read_text())['physical_config']
        self.task = TaskSpec.from_config(physical)
        self.state = np.zeros(10, np.float64)
        self.state[6] = -.051
        self.data = SimpleNamespace(task=self.task, state=self.state,
                                    proof={'source_tick': 49})
        self.states = np.broadcast_to(self.state, (2, 3, 6, 10)).copy()
        self.states[0, :, -1, 0] = .079
        self.states[1, :, -1, 0] = .078
        self.actions = np.zeros((2, 5, 2), np.float64)
        self.actual_costs = np.array([[1., 1., 1.], [2., 2., 2.]], np.float64)
        self.actual_valid = np.ones((2, 3), np.bool_)
        self.prediction_valid = np.ones(2, np.bool_)
        self.predicted_costs = np.array([2., 1.], np.float64)
        self.choice = 1
        self.local_error = np.zeros((2, 5), np.float64)
        self.local_state_error = np.zeros((2, 5, 10), np.float64)
        self.teacher_valid = np.ones((2, 5), np.bool_)

    def diagnose(self):
        states = self.states[:, 0].copy()
        predictions = PredictionBatch(states, np.diff(states, axis=1),
            self.prediction_valid, ('', ''), np.full(2, -1, np.int64))
        costs = CostBatch(self.predicted_costs, {})
        entries = [('fixture-model', predictions, costs,
                    {'selected_subset_index': self.choice})]
        local_difference = self.local_state_error.copy()
        local_difference[..., 0] += self.local_error
        local = {'fixture-model': {'traces': {
            'teacher_next_states': self.states[:, 0, 1:] + local_difference,
            'teacher_inference_valid_mask': self.teacher_valid,
            'teacher_object_xy_error_m': np.linalg.norm(local_difference[..., :2], axis=-1),
            'teacher_pusher_xy_error_m': np.linalg.norm(local_difference[..., 6:8], axis=-1),
            'teacher_object_yaw_learning_2pi_error_rad': local_difference[..., 2]}}}
        return hypotheses_for_snapshot(self.data, self.actions, self.states,
            self.actual_costs, self.actual_valid, entries, local)

    def test_disjoint_actual_ranges_support_misranking_only_in_checked_subset(self):
        result = self.diagnose()
        row = result['model_hypotheses'][0]
        self.assertEqual(row['H1_misranking'], 'SUPPORTED_WITHIN_SUBSET')
        self.assertEqual(row['comparison']['selection_loss'], 1.)
        self.assertEqual(result['H2_candidate_limitation'], 'NOT_SUPPORTED_WITHIN_SUBSET')

    def test_tiny_positive_cost_gap_stays_inconclusive_with_raw_loss_retained(self):
        self.actual_costs[1] = 1.000002
        row = self.diagnose()['model_hypotheses'][0]
        self.assertEqual(row['H1_misranking'], 'INCONCLUSIVE')
        self.assertGreater(row['comparison']['selection_loss'], 0.)
        self.assertEqual(row['comparison']['selection_loss_ordering'], 'INCONCLUSIVE')

    def test_detectable_teacher_forced_error_rejects_only_recursion_only_explanation(self):
        row = self.diagnose()['model_hypotheses'][0]
        self.assertEqual(row['H3_recursion_only'], 'NOT_REJECTED_BY_LOCAL_DIAGNOSTIC')
        self.local_error[0, 2] = self.task.position_tolerance_m * .001 * 1.01
        row = self.diagnose()['model_hypotheses'][0]
        self.assertEqual(row['H3_recursion_only'], 'REJECTED_AS_COMPLETE_EXPLANATION')
        self.assertTrue(row['not_model_acceptance_threshold'])

    def test_other_state_components_cannot_be_hidden_by_perfect_object_xy(self):
        for column, magnitude, component in ((2, .001, 'object_yaw'),
                (5, .001, 'object_angular_speed'), (8, .001, 'pusher_linear_speed')):
            with self.subTest(component=component):
                self.local_state_error.fill(0.)
                self.local_state_error[0, 1, column] = magnitude
                row = self.diagnose()['model_hypotheses'][0]
                self.assertEqual(row['H3_recursion_only'], 'REJECTED_AS_COMPLETE_EXPLANATION')
                self.assertFalse(row['detected_local_error_components']['object_XY'])
                self.assertTrue(row['detected_local_error_components'][component])

    def test_absent_valid_teacher_inference_is_not_evidence_against_recursion_only(self):
        self.teacher_valid[:] = False
        self.local_error[:] = .01
        row = self.diagnose()['model_hypotheses'][0]
        self.assertEqual(row['H3_recursion_only'], 'INCONCLUSIVE_NO_VALID_INFERENCE')

    def test_qualifying_endpoint_does_not_create_ready_or_hold_claim(self):
        result = self.diagnose()
        self.assertEqual(result['pose_and_speed_endpoint_subset_indices'], [0, 1])
        self.assertIn('no continuous hold', result['H2_scope'])
        self.assertNotIn('ready', result)
        self.assertNotIn('task_success', result)

    def test_position_progress_with_material_yaw_worsening_is_not_candidate_solution(self):
        self.states[:, :, -1, 2] = .2
        result = self.diagnose()
        self.assertEqual(result['pose_progress_subset_indices'], [])
        self.assertEqual(result['pose_and_speed_endpoint_subset_indices'], [])
        self.assertEqual(result['H2_candidate_limitation'], 'SUPPORTED_WITHIN_SUBSET_AND_H05')

    def test_physically_invalid_model_choice_is_preserved_without_selection_loss(self):
        self.actual_valid[1] = False
        self.actual_costs[1] = np.inf
        row = self.diagnose()['model_hypotheses'][0]
        self.assertEqual(row['H1_misranking'], 'INCONCLUSIVE')
        self.assertEqual(row['comparison']['selection_status'], 'selected_actual_invalid')
        self.assertIsNone(row['comparison']['selection_loss'])
        self.assertEqual(row['comparison']['counts']['false_valid'], 1)

    def test_no_valid_actual_endpoint_cannot_support_global_candidate_absence(self):
        self.actual_valid[:] = False
        self.actual_costs[:] = np.inf
        result = self.diagnose()
        self.assertEqual(result['H2_candidate_limitation'], 'INCONCLUSIVE_NO_VALID_ENDPOINT')
        row = result['model_hypotheses'][0]
        self.assertEqual(row['comparison']['actual_status'], 'no_valid_candidate')
        self.assertEqual(row['comparison']['counts']['actual_stable_invalid'], 2)
        self.assertIsNone(row['comparison']['selection_loss'])

    def test_caller_choice_cannot_disagree_with_model_argmin_on_original_population(self):
        self.choice = 0
        with self.assertRaises(ValueError):
            self.diagnose()


if __name__ == '__main__':
    unittest.main()
