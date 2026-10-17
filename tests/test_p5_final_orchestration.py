"""Pure fake-work orchestration invariants; no model or physics construction.

These tests intentionally specify rejection of resume/ledger evidence that the
coordinator must not adopt. Fixtures use fabricated task IDs, never final seeds
or generated final initial conditions. The historical resource plan is read-only.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('p5_orchestration_under_test', PROJECT/'scripts/p5_final_run.py')
coordinator = importlib.util.module_from_spec(spec);spec.loader.exec_module(coordinator)
from push_align.final_manifest import CONTROLLERS, expected_E2, expected_E1
from push_align.environment import Observation, StepResult
from push_align.planning import TaskSpec


class OrchestrationPureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='p5-orchestration-fixture-', dir=PROJECT/'tmp')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name);self.run=self.root/'run';self.run.mkdir()
        (self.root/'configs').mkdir()
        plan=json.loads((PROJECT/'configs/p5_g2_resource_plan.v2.json').read_text())
        self.write(self.root/'configs/plan.json',plan);self.plan=plan
        families=[]
        for sidx,stratum in enumerate(('translation','rotation','offset')):
            for rank in range(4):
                fid=f'fixture-{stratum}-{rank:02d}'
                families.append({'family_id':fid,'episode_id':fid,'episode_index':4*sidx+rank,'stratum':stratum})
        self.manifest={'families':families,'E2':expected_E2(families),'E1':expected_E1(families),
            'resource_plan_path':'configs/plan.json','application_family':{'family_id':'fixture-application-00'}}
        self.write(self.run/'manifest.json',self.manifest)
        self.initial={'E2_done':[],'E1_done':[],'replay_done':[],'application_done':[],'replay_selection':None}
        coordinator.save_progress(self.run,self.initial)
        self.addCleanup(patch.stopall)
        patch.object(coordinator,'ROOT',self.root).start()

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

    def test_one_normal_G_completion_releases_exactly_one_max_cell(self):
        state=deepcopy(self.initial);state['E2_done']=[self.manifest['E2'][0]['cell_id']]
        coordinator.save_progress(self.run,state)
        # Arithmetic only: evidence admission has its own missing/corrupt-file
        # tests. A fabricated ledger is not a positive completion fixture.
        with patch.object(coordinator,'verified_result',return_value={}):
            counts=coordinator.remaining_estimate(self.run)['counts']
        expected=dict(self.plan['counts'])
        for key,value in {'raw_records':1,'raw_intervals':300,'witness_records':1,
            'witness_intervals':300,'witness_decisions':301,'G_captures':3}.items():expected[key]-=value
        self.assertEqual(counts,expected)

    def test_commit_of_same_completed_key_is_idempotent(self):
        key=self.manifest['E2'][0]['cell_id']
        coordinator.commit_progress(self.run,'fixture','E2_done',key)
        coordinator.commit_progress(self.run,'fixture','E2_done',key)
        self.assertEqual(coordinator.progress(self.run)['E2_done'],[key])
        self.assertEqual(len((self.run/'progress_events.jsonl').read_text().splitlines()),1)

    def test_duplicate_application_done_cannot_release_twice(self):
        state=deepcopy(self.initial);state['application_done']=['G','G'];coordinator.save_progress(self.run,state)
        with self.assertRaises((ValueError,RuntimeError)):
            coordinator.remaining_estimate(self.run)

    def test_duplicate_E1_done_cannot_release_twice(self):
        state=deepcopy(self.initial);state['E1_done']=['fixture/physical-000','fixture/physical-000']
        coordinator.save_progress(self.run,state)
        with self.assertRaises((ValueError,RuntimeError)):
            coordinator.remaining_estimate(self.run)

    def test_unknown_E2_progress_is_not_accepted_as_valid_ledger(self):
        state=deepcopy(self.initial);state['E2_done']=['another-study/missing'];coordinator.save_progress(self.run,state)
        with self.assertRaises((ValueError,RuntimeError)):
            coordinator.remaining_estimate(self.run)

    def fixture_result(self,*,wrong_identity=False,bad_raw=False,bad_outcome=False):
        cell=self.manifest['E2'][0]
        coordinator.save_progress(self.run,deepcopy(self.initial))
        folder=self.run/'E2'/cell['cell_id'];folder.mkdir(parents=True,exist_ok=True)
        raw=folder/'raw-fixture.bin';raw.write_bytes(b'preserved raw evidence')
        identity,bindings=coordinator.expected_recording_contract(self.run,self.manifest,cell)
        record={'schema':'p5-recorded-episode-v1','record_kind':'reference_controller_execution','identity':identity,
            'recording_status':'COMPLETE_RECORDED','files_sha256':{str(raw):coordinator.sha(raw)},'source_bindings':bindings,
            'execution':{'execution_status':'completed','termination':'timeout','future_engine_calls':{'mj_step':0},'counts':{}}}
        record_path=folder/'manifest.json';self.write(record_path,record)
        row={**cell,'execution_status':'completed','termination':'timeout','future_engine_calls':{'mj_step':0},'success':False,'counts':{}}
        review={'status':'ACCEPT_COMPLETED_TASK_FAILURE','accepted':True,
            'manifest_sha256':coordinator.sha(record_path),'metrics':{'success':False}}
        self.write(folder/'review.json',review)
        self.write(folder/'component_budget_review.json',{'status':'PASS_MEASURED_COMPONENT_ALLOWANCES'})
        result={'status':'ACCEPT_COMPLETED_TASK_FAILURE','phase':'E2','cell':deepcopy(cell),'row':row,
            'review_sha256':coordinator.sha(folder/'review.json'),
            'component_review_sha256':coordinator.sha(folder/'component_budget_review.json'),
            'record_manifest':str(record_path),'record_manifest_sha256':coordinator.sha(record_path)}
        if wrong_identity:
            result['cell']=deepcopy(self.manifest['E2'][1]);result['row'].update(self.manifest['E2'][1])
        if bad_outcome:
            result['row'].update(execution_status='infrastructure_error',future_engine_calls={'mj_step':1})
        self.write(folder/'worker_result.json',result)
        self.write(folder/'worker_result_receipt.json',{'sha256':coordinator.sha(folder/'worker_result.json')})
        if bad_raw:raw.write_bytes(b'changed after recording')
        patch.object(coordinator,'frozen',return_value=self.manifest).start()
        patch.object(coordinator,'capacity_check',return_value={}).start()
        return cell

    def test_valid_sealed_metadata_can_be_read_without_reloading_raw(self):
        cell=self.fixture_result()
        result=coordinator.verified_result(self.run,'E2',cell['cell_id'],self.manifest,deep=False)
        self.assertEqual(result['row']['termination'],'timeout')

    def test_metrics_cannot_change_after_sealing(self):
        cell=self.fixture_result()
        path=self.run/'E2'/cell['cell_id']/'worker_result.json'
        altered=json.loads(path.read_text());altered['row']['success']=True;self.write(path,altered)
        with self.assertRaisesRegex(RuntimeError,'Worker result changed'):
            coordinator.verified_result(self.run,'E2',cell['cell_id'],self.manifest,deep=False)

    def test_progress_receipt_detects_edit_between_commits(self):
        coordinator.save_progress(self.run,self.initial)
        changed=deepcopy(self.initial);changed['E2_done']=[self.manifest['E2'][0]['cell_id']]
        self.write(self.run/'progress.json',changed)
        with self.assertRaisesRegex(RuntimeError,'Progress ledger changed'):
            coordinator.remaining_estimate(self.run)

    def test_progress_receipt_cannot_be_missing(self):
        (self.run/'progress_receipt.json').unlink()
        with self.assertRaisesRegex(RuntimeError,'Missing durable progress'):
            coordinator.progress(self.run)

    def test_E2_original_cannot_be_adopted_as_fresh_record_kind(self):
        cell=self.fixture_result()
        folder=self.run/'E2'/cell['cell_id']
        record=coordinator.read(folder/'manifest.json');record['record_kind']='fresh_controller_replay'
        self.write(folder/'manifest.json',record)
        result=coordinator.read(folder/'worker_result.json');result['record_manifest_sha256']=coordinator.sha(folder/'manifest.json')
        self.write(folder/'worker_result.json',result)
        self.write(folder/'worker_result_receipt.json',{'sha256':coordinator.sha(folder/'worker_result.json')})
        with self.assertRaisesRegex(RuntimeError,'Recording kind differs'):
            coordinator.verified_result(self.run,'E2',cell['cell_id'],self.manifest,deep=False)

    def test_shortened_replay_selection_cannot_release_reserved_reproductions(self):
        from push_align.final_manifest import replay_union
        rows=[dict(c,execution_status='completed',termination='timeout') for c in self.manifest['E2']]
        chosen=replay_union(rows);state=deepcopy(self.initial)
        state.update(E2_done=[c['cell_id'] for c in rows],replay_selection=chosen[:-1])
        coordinator.save_progress(self.run,state);self.write(self.run/'replay_selection.json',chosen[:-1])
        by={r['cell_id']:r for r in rows}
        with patch.object(coordinator,'verified_result',side_effect=lambda run,phase,key,m,deep=False:{'row':by[key]}):
            with self.assertRaisesRegex(RuntimeError,'Replay selection differs'):
                coordinator.remaining_estimate(self.run)

    def test_component_growth_is_a_stop_and_G_empty_index_is_operational(self):
        folder=self.root/'measured';folder.mkdir();raw=folder/'raw';raw.write_bytes(b'0'*200000)
        witness=folder/'witness';witness.write_bytes(b'0')
        index=folder/'index';index.write_bytes(b'0'*265)
        record={'execution':{'counts':{'physics_intervals_returned':1,'controller_decisions':1}},
            'identity':{'controller_id':'G'},'raw':{'files_sha256':{str(raw):'not used'}},
            'plan_index':{'files_sha256':{str(index):'not used'}},'witness_path':str(witness),'full_traces':[]}
        result=coordinator.measured_component_review(record,self.plan)
        self.assertEqual(result['status'],'STOP_REESTIMATE_PRESERVED_PREFIX')
        self.assertNotIn(str(index),result['measured_component_paths'])

    def test_written_active_prefix_is_not_reserved_twice(self):
        reserve=2*1024**3
        with patch.object(coordinator,'remaining_estimate',return_value={'required_bytes_including_reserve':reserve+1000}), \
                patch.object(coordinator.shutil,'disk_usage',return_value=SimpleNamespace(free=reserve+900)):
            with self.assertRaisesRegex(RuntimeError,'Capacity STOP'):
                coordinator.capacity_check(self.run)
            result=coordinator.capacity_check(self.run,active_written_credit=200)
            self.assertEqual(result['remaining_required_bytes'],reserve+800)

    def test_summary_without_completion_seal_is_not_a_completed_stage(self):
        self.write(self.run/'statistics/summary.json',{'status':'looks complete'})
        with self.assertRaisesRegex(RuntimeError,'Incomplete derived stage'):
            coordinator.adopt_derived_stage(self.run,'statistics')

    def test_derived_stage_seal_checks_retained_arrays_and_sources(self):
        folder=self.run/'statistics';folder.mkdir()
        arrays=folder/'bootstrap.npz';arrays.write_bytes(b'fixture retained arrays')
        self.write(folder/'summary.json',{'scope':'byte seal fixture only'})
        self.write(folder/'reproduction.json',{'scope':'byte seal fixture only'})
        source=self.root/'source';source.write_bytes(b'fixture source')
        coordinator.seal_derived_stage(self.run,'statistics',{str(source):coordinator.sha(source)})
        # This test isolates the byte seal only, not scientific completeness.
        with patch('push_align.final_stage_review.validate_derived_stage',return_value={'status':'PASS'}), \
                patch.object(coordinator,'frozen',return_value=self.manifest):
            self.assertTrue(coordinator.adopt_derived_stage(self.run,'statistics'))
        arrays.write_bytes(b'changed retained arrays')
        with self.assertRaisesRegex(RuntimeError,'Derived stage/source changed'):
            coordinator.adopt_derived_stage(self.run,'statistics')

    def test_resume_rejects_completed_result_for_a_different_cell(self):
        cell=self.fixture_result(wrong_identity=True)
        with self.assertRaises((ValueError,RuntimeError)):
            coordinator.execute_cells(self.run,'E2')
        self.assertNotIn(cell['cell_id'],coordinator.progress(self.run)['E2_done'])

    def test_resume_revalidates_recorded_file_hashes(self):
        cell=self.fixture_result(bad_raw=True)
        with self.assertRaises((ValueError,RuntimeError)):
            coordinator.execute_cells(self.run,'E2')
        self.assertNotIn(cell['cell_id'],coordinator.progress(self.run)['E2_done'])

    def test_resume_cannot_adopt_saved_result_after_worker_postcheck_failed(self):
        cell=self.fixture_result(bad_outcome=True)
        with self.assertRaises((ValueError,RuntimeError)):
            coordinator.execute_cells(self.run,'E2')
        self.assertNotIn(cell['cell_id'],coordinator.progress(self.run)['E2_done'])

    def test_metric_time_is_episode_relative_and_last_valid_is_explicit(self):
        config=json.loads((PROJECT/'runs/p4-04-20261013-01/manifest.json').read_text())['physical']
        task=TaskSpec.from_config(config)
        initial=Observation(np.zeros(10,np.float64),2.0,'fixture-state',True)
        good=np.zeros(10,np.float64);good[0]=.075
        step0=StepResult(np.zeros(2),np.zeros(2),Observation(good,2.1,'fixture-state',True),.1,True,
            {'samples':[]},np.empty((0,10)),np.empty(0))
        step1=StepResult(np.zeros(2),np.zeros(2),Observation(np.full(10,np.nan),2.2,'fixture-state',False),.1,False,
            {'samples':[]},np.empty((0,10)),np.empty(0))
        execution=SimpleNamespace(initial_observation=initial,results=[step0,step1],
            records=[{'geometry':{'pass':True}},{'geometry':{'pass':False}}],task_spec=task,
            execution_status='completed',termination='physics_error',final_task=SimpleNamespace(elapsed_s=.2),
            initial_task=SimpleNamespace(initial_sim_time=2.0),counts={},future_engine_calls={})
        row=coordinator._metric_row(execution,self.manifest['E2'][0],planning_wall_s=.1,execution_wall_s=.2)
        self.assertAlmostEqual(row['metric_state_time_s'],.1)
        self.assertEqual(row['metric_state_origin'],'last_valid_observation')
        self.assertFalse(row['final_observation_available'])
        self.assertAlmostEqual(row['final_position_error_m'],.005)
        self.assertFalse(row['success'])

    def test_call_timing_distinguishes_first_real_call_without_added_warmup(self):
        ledger={'plan':0.,'plan_calls':0,'plan_first_call_s':None,
                'plan_repeat_wall_s':0.,'plan_repeat_calls':0}
        coordinator.record_call_timing(ledger,'plan',.3)
        coordinator.record_call_timing(ledger,'plan',.2)
        coordinator.record_call_timing(ledger,'plan',.1)
        self.assertEqual(ledger['plan_first_call_s'],.3)
        self.assertAlmostEqual(ledger['plan_repeat_wall_s'],.3)
        self.assertAlmostEqual(ledger['plan'],.6)
        self.assertEqual(ledger['plan_calls'],3)
        self.assertEqual(ledger['plan_repeat_calls'],2)

    def test_analysis_write_rejects_growth_before_creating_output(self):
        self.write(self.run/'statistics/preserved.json',{'retained':True})
        with patch.object(coordinator,'derived_limit',return_value=2), \
                patch.object(coordinator,'capacity_check') as capacity:
            with self.assertRaisesRegex(RuntimeError,'Analysis allowance exceeded'):
                coordinator.save_derived_bytes(self.run,'statistics','too-large.bin',b'abc')
            capacity.assert_not_called()
        self.assertFalse((self.run/'statistics/too-large.bin').exists())

    def test_derived_partial_folder_requires_review_before_analysis(self):
        (self.run/'statistics').mkdir()
        with self.assertRaisesRegex(RuntimeError,'retained'):
            coordinator.begin_derived_stage(self.run,'statistics')


if __name__=='__main__':unittest.main()
