"""Independent review over scripted records, with zero real model/physics calls."""
from copy import deepcopy
from dataclasses import replace
import json
import gzip
from pathlib import Path
from types import SimpleNamespace
import tempfile
import tomllib
import unittest
from unittest.mock import patch
import numpy as np

from push_align.config import ROOT, load_pilot_config
from push_align.environment import Observation
from push_align.final_driver import LiveResourceGuard, run_recorded_episode
from push_align.final_recording import file_sha256
from push_align.final_recording import write_witness_gzip
from push_align import final_recording_columns
from push_align.final_result_review import review_recording
from push_align.planning_protocol import GeometryProposalAdapter
from test_p5_final_execution import SyntheticEnv, state_at_goal, stationary_predictor


class FullSampleFakeEnv(SyntheticEnv):
    """Explicit physical-looking metadata, with scripted samples and no engine."""
    def __init__(self,state,*,failure=None,**kwargs):
        super().__init__(state,**kwargs);self.failure=failure

    def step(self,action):
        if self.calls >= len(self.script):
            self.script.append([self.state.copy() for _ in range(200)])
        step=super().step(action)
        samples=step.diagnostics['samples']
        for i,sample in enumerate(samples):
            state=np.asarray(sample['state'])
            sample.update(engine_finite=bool(np.isfinite(state).all()),warning_counts=[0]*7,
                object_corners_xy=[[state[0]-.04,state[1]-.03],[state[0]+.04,state[1]+.03]],
                object_height_m=.01,tilt_rad=0.,object_velocity_world=[0.,0.,float(state[5]),float(state[3]),float(state[4]),0.],
                pusher_velocity_world=[0.,0.,0.,float(state[8]),float(state[9]),0.],actuator_force_N=[0.,0.])
            if i:sample['integration_actuator_force_N']=[0.,0.]
        reasons=[];observation_valid=True
        if self.failure=='geometry':
            samples[-1]['contacts']=[{'pair':'pusher_object','normal_force_N':.1,'distance_m':0.,
                'normal_world':[1.,0.,0.],'normal_center_cosine':-1.}]
        elif self.failure=='planarity':
            samples[-1]['tilt_rad']=.2;reasons=['tilt_exceeded'];observation_valid=False
        elif self.failure=='force':
            samples[-1]['integration_actuator_force_N']=[3.,0.];reasons=['integration_actuator_force_exceeded']
        elif self.failure=='false_label':
            reasons=['tilt_exceeded'];observation_valid=False
        step.diagnostics['validity_violations']=reasons
        return replace(step,transition_valid=not bool(reasons),
            observation=replace(step.observation,observation_valid=observation_valid))


class FinalResultReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.folder=Path(self.temp.name)
        self.config=load_pilot_config();self.config['task'].update(hold_time_s=.1,max_episode_time_s=.2)
        self.planner=deepcopy(self.config);self.planner['planner'].update(candidate_count=16,horizon=5,block_lengths=[2,3])
        self.geometry=json.loads((ROOT/'configs/p3_geometry_selected.v2.json').read_text())
        self.transition=json.loads((ROOT/'configs/p3_geometry_transition.v2.json').read_text())
        self.profile=tomllib.loads((ROOT/'configs/acceptance.v1.toml').read_text())
        source=self.folder/'source.txt';source.write_text('read-only scripted fixture source')
        self.bindings={str(source):file_sha256(source)}
        self.identity={'model':{'fixture':'stationary'},'normalizer':{'fixture':'identity'}}
        self.counter=0

    def run_cell(self,*,failure=None,goal=False,predictor=stationary_predictor,reference=None,**overrides):
        self.counter+=1;env=FullSampleFakeEnv(state_at_goal(goal),failure=failure)
        guard=LiveResourceGuard(self.folder,remaining_required_bytes=lambda:3*1024**3,
            cell_write_allowance_bytes=8*1024**2,free_bytes=lambda:10*1024**3)
        args=dict(case_id='fixture',episode_index=0,controller_id='R',execution_id=f'fixture-{self.counter}',
            predictor=predictor,physical_config=self.config,planner_config=self.planner,
            geometry_parameters=self.geometry,transition_parameters=self.transition,candidate_seed_base=123,
            candidate_stream_id=0,max_intervals=2,run_scope='final_evaluation',penetration_budget_m=.001,
            origin={'fixture':True,'not_final_test':True})
        args.update(overrides)
        identity=self.identity if predictor is not None else {'model':None,'normalizer':None}
        return run_recorded_episode(env,env.snapshot(),folder=self.folder/f'cell-{self.counter}',
            identity=identity,source_bindings=self.bindings,profile=self.profile,guard=guard,
            reference_manifest=reference,raw_codec=final_recording_columns,**args)

    def review(self,result,**kwargs):
        return review_recording(result.manifest_path,physical_config=self.config,profile=self.profile,
            expected_manifest_sha256=file_sha256(result.manifest_path),expected_identity=result.manifest['identity'],
            expected_source_bindings=self.bindings,**kwargs)

    def test_ready_is_reassessed_from_all_actual_samples(self):
        result=self.run_cell(goal=True)
        reviewed=self.review(result)
        self.assertTrue(reviewed['accepted'],reviewed['issues'])
        self.assertEqual(reviewed['status'],'ACCEPT_COMPLETED_RECORDING')
        self.assertTrue(reviewed['metrics']['success'])
        self.assertAlmostEqual(reviewed['metrics']['elapsed_s'],.1)
        self.assertAlmostEqual(reviewed['metrics']['metric_state_time_s'],.1)

    def test_timeout_retains_failed_denominator_without_success_claim(self):
        result=self.run_cell()
        reviewed=self.review(result)
        self.assertTrue(reviewed['accepted'],reviewed['issues'])
        self.assertEqual(reviewed['status'],'ACCEPT_COMPLETED_TASK_FAILURE')
        self.assertFalse(reviewed['metrics']['success'])
        self.assertEqual(reviewed['termination'],'timeout')
        self.assertAlmostEqual(reviewed['metrics']['elapsed_s'],.2)

    def test_geometry_failure_keeps_STOP_REVIEW_and_gets_separate_resolution(self):
        result=self.run_cell(goal=True,failure='geometry')
        self.assertEqual(result.status,'STOP_REVIEW')
        before=file_sha256(result.manifest_path);reviewed=self.review(result)
        self.assertTrue(reviewed['accepted'],reviewed['issues'])
        self.assertEqual(reviewed['termination'],'geometry_invalid')
        self.assertEqual(reviewed['recording_status_preserved'],'STOP_REVIEW')
        self.assertEqual(file_sha256(result.manifest_path),before)
        self.assertFalse(reviewed['metrics']['success'])

    def test_planarity_and_integration_force_failures_reproduce_independently(self):
        for failure,reason in [('planarity','planarity_violation'),('force','physics_error')]:
            with self.subTest(failure=failure):
                result=self.run_cell(failure=failure);reviewed=self.review(result)
                self.assertTrue(reviewed['accepted'],reviewed['issues'])
                self.assertEqual(reviewed['termination'],reason)
                self.assertEqual(reviewed['metrics']['metric_state_origin'],'last_valid_observation')
                self.assertAlmostEqual(reviewed['metrics']['metric_state_time_s'],0.)

    def test_all_invalid_is_supported_by_complete_zero_execution_and_index(self):
        result=self.run_cell(predictor=lambda state,action:np.full_like(state,np.nan))
        reviewed=self.review(result)
        self.assertTrue(reviewed['accepted'],reviewed['issues'])
        self.assertEqual(reviewed['termination'],'planner_invalid')
        self.assertEqual(reviewed['metrics']['counts']['physics_intervals_returned'],0)

    def test_compact_fresh_review_reports_reference_scope_and_live_raw_option(self):
        original=self.run_cell();fresh=self.run_cell(reference=original.manifest_path)
        self.assertIsNone(fresh.manifest['raw'])
        reviewed=self.review(fresh)
        self.assertTrue(reviewed['accepted'],reviewed['issues'])
        self.assertEqual(reviewed['raw_provenance'],'original_referenced_prefix')
        self.assertFalse(reviewed['fresh_substeps_independently_rechecked'])
        direct=self.review(fresh,raw_execution=fresh.execution)
        self.assertTrue(direct['accepted'],direct['issues'])
        self.assertTrue(direct['fresh_substeps_independently_rechecked'])
        self.assertEqual(reviewed['metrics'],direct['metrics'])
        # Identical original raw bytes cannot be renamed into an independently
        # supplied new execution; the actual fresh execution identity is required.
        self.assertFalse(self.review(fresh,raw_execution=original.execution)['accepted'])
        shadow=SimpleNamespace(initial_snapshot=original.execution.initial_snapshot,results=original.execution.results)
        reconstructed=self.review(fresh,raw_execution=shadow,raw_execution_origin='reconstructed_saved_raw')
        self.assertTrue(reconstructed['accepted'],reconstructed['issues'])
        self.assertFalse(reconstructed['fresh_substeps_independently_rechecked'])

    def test_fresh_physical_failure_requires_and_checks_its_own_raw(self):
        original=self.run_cell(failure='force');fresh=self.run_cell(failure='force',reference=original.manifest_path)
        self.assertIsNotNone(fresh.manifest['raw'])
        reviewed=self.review(fresh,raw_execution=fresh.execution)
        self.assertTrue(reviewed['accepted'],reviewed['issues'])
        self.assertEqual(reviewed['raw_provenance'],'fresh_saved_full_prefix')
        self.assertTrue(self.review(fresh)['fresh_substeps_independently_rechecked'])

    def test_wrong_identity_source_missing_changed_raw_and_manifest_are_rejected(self):
        result=self.run_cell()
        wrong=review_recording(result.manifest_path,physical_config=self.config,profile=self.profile,expected_identity={'wrong':True})
        self.assertFalse(wrong['accepted'])
        wrong=review_recording(result.manifest_path,physical_config=self.config,profile=self.profile,expected_source_bindings={'wrong':'0'*64})
        self.assertFalse(wrong['accepted'])
        raw_path=next(iter(result.manifest['raw']['files_sha256']));Path(raw_path).write_bytes(b'changed')
        self.assertFalse(self.review(result)['accepted'])

    def test_incomplete_prefix_and_oracle_counts_cannot_be_resolved_as_task_failure(self):
        result=self.run_cell(max_intervals=1)
        self.assertFalse(self.review(result)['accepted'])
        result=self.run_cell();manifest=json.loads(result.manifest_path.read_text())
        manifest['execution']['future_engine_calls']['mj_step']=1
        result.manifest_path.write_text(json.dumps(manifest))
        self.assertFalse(self.review(result)['accepted'])

    def test_profile_change_or_resource_limit_does_not_get_accepted(self):
        result=self.run_cell(goal=True,failure='geometry')
        bad=deepcopy(self.profile);bad['physics_suite']['actual_contact_penetration_max_m']=.002
        reviewed=review_recording(result.manifest_path,physical_config=self.config,profile=bad)
        self.assertFalse(reviewed['accepted'])
        with patch('push_align.final_result_review.MAX_RSS_BYTES',0):
            self.assertFalse(self.review(result)['accepted'])

    def forged_copy(self,result,mutate):
        manifest=json.loads(result.manifest_path.read_text())
        with gzip.open(manifest['witness_path'],'rt') as stream:witness=json.load(stream)
        mutate(manifest,witness)
        witness_path=result.manifest_path.parent/'forged-witness.json.gz'
        receipt=write_witness_gzip(witness_path,witness)
        manifest['witness_path']=str(witness_path);manifest['files_sha256'].update(receipt['files_sha256'])
        path=result.manifest_path.parent/'forged-manifest.json';path.write_text(json.dumps(manifest))
        return review_recording(path,physical_config=self.config,profile=self.profile,
            expected_manifest_sha256=file_sha256(path),expected_identity=self.identity,
            expected_source_bindings=self.bindings)

    def test_forged_matching_saved_terminal_labels_still_fail_raw_supervision(self):
        result=self.run_cell()
        def mutate(manifest,witness):
            status=deepcopy(witness['final_task'])
            status.update(phase='ready',terminal_reason='ready',held_time_s=.1,previous_qualifies=True)
            witness['termination']='ready';witness['final_task']=status;witness['intervals'][-1]['task_after']=status
            manifest['execution']['termination']='ready';manifest['execution']['task']=status
        reviewed=self.forged_copy(result,mutate)
        self.assertFalse(reviewed['accepted'])
        self.assertTrue(any('raw_task_after_differs' in issue for issue in reviewed['issues']),reviewed['issues'])

    def test_false_physical_failure_labels_are_not_a_resolution(self):
        result=self.run_cell(failure='false_label');self.assertEqual(result.status,'STOP_REVIEW')
        reviewed=self.review(result)
        self.assertFalse(reviewed['accepted'])
        self.assertTrue(any('independent_physical_violations_differ' in issue for issue in reviewed['issues']),reviewed['issues'])

    def test_unexplained_STOP_REVIEW_missing_full_input_and_count_change_reject(self):
        for change in ('stop','input','count'):
            with self.subTest(change=change):
                result=self.run_cell()
                def mutate(manifest,witness):
                    if change=='stop':manifest['recording_status']='STOP_REVIEW'
                    elif change=='input':manifest['full_traces']=[]
                    else:
                        witness['counts']['physics_interval_attempts']+=1
                        manifest['execution']['counts']['physics_interval_attempts']+=1
                reviewed=self.forged_copy(result,mutate)
                self.assertFalse(reviewed['accepted'])

    def test_fresh_exact_attestation_digest_and_changed_source_are_rejected(self):
        original=self.run_cell();fresh=self.run_cell(reference=original.manifest_path)
        def mutate(manifest,witness):manifest['comparison']['raw_exact']['right_sha256']='0'*64
        self.assertFalse(self.forged_copy(fresh,mutate)['accepted'])
        Path(next(iter(self.bindings))).write_text('changed source')
        self.assertFalse(self.review(original)['accepted'])

    def test_controller_infeasible_requires_independent_G_proposal_check(self):
        original=GeometryProposalAdapter.propose
        def infeasible(*args,**kwargs):
            proposal=original(*args,**kwargs)
            return replace(proposal,geometry_action=None)
        with patch.object(GeometryProposalAdapter,'propose',side_effect=infeasible,autospec=True) as checked:
            result=self.run_cell(predictor=None,controller_id='G')
            reviewed=self.review(result)
            self.assertTrue(reviewed['accepted'],reviewed['issues'])
            self.assertEqual(reviewed['termination'],'controller_infeasible')
            self.assertEqual(checked.call_count,2)
        # The same invented failure is rejected when the pure controller actually
        # proposes an action; review never trusts a saved infeasible label alone.
        self.assertFalse(self.review(result)['accepted'])


if __name__=='__main__':unittest.main()
