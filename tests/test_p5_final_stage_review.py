"""Derived-stage contracts over synthetic files; no model/physics execution."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

import numpy as np

from push_align.config import ROOT, load_pilot_config
from push_align.contact_geometry import check_contact_geometry
from push_align.environment import Observation, SnapshotBundle
from push_align.final_driver import write_full_trace
from push_align.final_manifest import expected_E2
from push_align.final_prediction import evaluate_final_predictions, MODELS
from push_align.final_recording import file_sha256, write_json_gzip
from push_align import final_recording_columns as codec
from push_align.final_stage_review import validate_derived_stage
from push_align.final_statistics import analyze_E2, METRICS
from test_p5_final_prediction import fixtures


class FinalStageReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='p5-stage-review-fixture-',dir=ROOT/'tmp')
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.physical = load_pilot_config()
        self.planner = deepcopy(self.physical)
        self.planner['planner'].update(horizon=5,command_speed_norm_max_m_s=.03)
        self.profile = tomllib.loads((ROOT/'configs/acceptance.v1.toml').read_text())
        self.families = [{'family_id':f'fixture-{index:02d}','episode_id':f'fixture-{index:02d}',
            'episode_index':index,'stratum':('translation','rotation','offset')[index//4]} for index in range(12)]
        self.m = {'families':self.families,'E2':expected_E2(self.families),
            'statistics':{'replicates':2000,'seed':7711},'physical_config':self.physical,
            'planner_config':self.planner,'profile':self.profile}
        self.write(self.run/'manifest.json',self.m)
        self.rows = []
        for cell in self.m['E2']:
            row = {**cell,'execution_status':'completed','termination':'timeout',**{key:1. for key in METRICS}}
            row.update(success=False,elapsed_s=30.,recontact_count=1,
                final_observation_available=True,metric_state_origin='terminal_observation',metric_state_time_s=30.,
                counts={'physics_intervals_returned':12})
            self.rows.append(row)
            self.write(self.run/'E2'/cell['cell_id']/'worker_result.json',{'fixture_verified_row':row})

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,allow_nan=False))

    def seal(self,name,sources):
        folder = self.run/name
        files = {str(path):file_sha256(path) for path in folder.iterdir() if path.is_file() and path.name != 'stage_receipt.json'}
        self.write(folder/'stage_receipt.json',{'schema':'p5-derived-stage-seal-v1','stage':name,
            'frozen_manifest_sha256':file_sha256(self.run/'manifest.json'),'files_sha256':files,'source_sha256':sources})

    def reseal(self,name):
        prior=json.loads((self.run/name/'stage_receipt.json').read_text())
        self.seal(name,prior['source_sha256'])

    def statistics(self):
        summary,arrays=analyze_E2(self.rows,self.m['E2'])
        folder=self.run/'statistics';folder.mkdir()
        self.write(folder/'summary.json',summary)
        np.savez_compressed(folder/'bootstrap.npz',**arrays)
        self.write(folder/'reproduction.json',{'status':'PASS_EXACT_REGENERATION','source_rows':96,'families':12,
            'summary_sha256':file_sha256(folder/'summary.json'),'bootstrap_sha256':file_sha256(folder/'bootstrap.npz')})
        sources={str(self.run/'E2'/row['cell_id']/'worker_result.json'):file_sha256(self.run/'E2'/row['cell_id']/'worker_result.json') for row in self.m['E2']}
        self.seal('statistics',sources)

    def prediction(self,*,length=12,invalid_model=False,invalid_truth=False,model_override=None):
        trajectories,predictors,_=fixtures(length)
        if invalid_model:
            predictors['R']=lambda state,action:np.full_like(state,np.nan)
        if model_override is not None:
            predictors['R']=model_override
        sources={}
        for family,trajectory in zip(self.families,trajectories):
            folder=self.run/'E2'/family['family_id']/'G'
            for step in trajectory['results']:
                for sample in step.diagnostics['samples']:
                    for contact in sample['contacts']:
                        contact.update(distance_m=0.,normal_world=[1.,0.,0.],normal_center_cosine=1.)
            if invalid_truth and trajectory is trajectories[0] and length:
                trajectory['results'][-1].diagnostics['samples'][-1]['contacts']=[{
                    'pair':'pusher_object','normal_force_N':.002,'distance_m':-.01,
                    'normal_world':[1.,0.,0.],'normal_center_cosine':1.}]
                trajectory['interval_valid'][-1]=False
            initial=Observation(trajectory['initial_state10'].copy(),float(trajectory['initial_sim_time']),self.physical['schema_version'],True)
            snapshot=SnapshotBundle(np.r_[initial.sim_time,initial.state].astype(np.float64),'fixture-model','fixture-environment',.01,
                {'fixture_seed':123},np.zeros(1,np.int64),np.zeros(1,np.int64),False)
            raw=codec.write_raw(folder/'raw_execution',snapshot,trajectory['results'])
            raw['prefix']=str(folder/'raw_execution')
            capture={'tick':0,'case_id':family['family_id'],'controller_id':'G','episode_index':family['episode_index'],
                'observation':initial,'snapshot':snapshot}
            trace=write_full_trace(folder/'plans/step-000',tick=0,captured_input=capture)
            witness_path=folder/'witness.json.gz'
            write_json_gzip(witness_path,{'intervals':[{'geometry':check_contact_geometry(step.diagnostics['samples'],
                self.profile['physics_suite']['actual_contact_penetration_max_m'])} for step in trajectory['results']]})
            files={**raw['files_sha256'],**trace['files_sha256'],str(witness_path):file_sha256(witness_path)}
            record={'schema':'p5-recorded-episode-v1','record_kind':'reference_controller_execution','recording_status':'COMPLETE_RECORDED',
                'identity':{'family_id':family['family_id'],'controller_id':'G','model':None,'normalizer':None,
                    'final_manifest_sha256':file_sha256(self.run/'manifest.json')},
                'execution':{'run_scope':'final_evaluation','execution_status':'completed','counts':{'physics_intervals_returned':length}},
                'files_sha256':files,'raw':raw,'raw_codec_module':'push_align.final_recording_columns','full_traces':[trace],
                'witness_path':str(witness_path)}
            path=folder/'manifest.json';self.write(path,record)
            sources[str(path)]=file_sha256(path);sources.update(files)
        for row in self.rows:
            if row['controller_id']=='G':row['counts']['physics_intervals_returned']=length
        summary,arrays=evaluate_final_predictions(trajectories,predictors,self.physical,self.planner)
        folder=self.run/'fixed_prediction';folder.mkdir()
        self.write(folder/'summary.json',summary);np.savez_compressed(folder/'traces.npz',**arrays)
        self.seal('fixed_prediction',sources)

    def review(self,name,rows=None):
        return validate_derived_stage(self.run,name,self.m,load_e2_rows=lambda:self.rows if rows is None else rows)

    def mutate_arrays(self,name,filename,change):
        path=self.run/name/filename
        with np.load(path,allow_pickle=False) as source:arrays={key:source[key].copy() for key in source.files}
        change(arrays);np.savez_compressed(path,**arrays)
        self.reseal(name)

    def test_statistics_regenerates_every_array_and_all96_denominators(self):
        self.statistics();receipt=self.review('statistics')
        self.assertEqual(receipt['status'],'PASS')
        self.assertEqual(receipt['checks']['bootstrap_replicates'],2000)
        self.assertGreater(receipt['checks']['exact_npz_arrays'],100)
        self.assertEqual(receipt['new_model_forwards'],0)

    def test_statistics_rejects_summary_success_claim_and_wrong_source_population(self):
        self.statistics();path=self.run/'statistics/summary.json'
        summary=json.loads(path.read_text());summary['episode_count']=95;self.write(path,summary);self.reseal('statistics')
        with self.assertRaisesRegex(RuntimeError,'summary'):self.review('statistics')
        with self.assertRaisesRegex(RuntimeError,'96'):self.review('statistics',self.rows[:-1]+[self.rows[0]])

    def test_statistics_rejects_missing_arrays_and_wrong_dtypes(self):
        self.statistics()
        self.mutate_arrays('statistics','bootstrap.npz',lambda arrays:arrays.pop('family_draw_indices'))
        with self.assertRaisesRegex(RuntimeError,'key population'):self.review('statistics')

    def test_statistics_rejects_tampered_draws_even_after_file_reseal(self):
        self.statistics()
        def change(arrays):arrays['family_draw_indices'][0,0]=11
        self.mutate_arrays('statistics','bootstrap.npz',change)
        with self.assertRaisesRegex(RuntimeError,'array values'):self.review('statistics')

    def test_statistics_rejects_array_dtype_change(self):
        self.statistics()
        def change(arrays):arrays['family_draw_indices']=arrays['family_draw_indices'].astype(np.int32)
        self.mutate_arrays('statistics','bootstrap.npz',change)
        with self.assertRaisesRegex(RuntimeError,'dtype'):self.review('statistics')

    def test_statistics_rejects_changed_bootstrap_settings(self):
        self.statistics();self.m['statistics']['seed']=123;self.write(self.run/'manifest.json',self.m);self.reseal('statistics')
        with self.assertRaisesRegex(RuntimeError,'seed7711'):self.review('statistics')

    def test_statistics_rejects_missing_reproduction_and_source_graph_omission(self):
        self.statistics();path=self.run/'statistics/stage_receipt.json'
        receipt=json.loads(path.read_text());receipt['source_sha256'].pop(next(iter(receipt['source_sha256'])))
        self.write(path,receipt)
        with self.assertRaisesRegex(RuntimeError,'source graph'):self.review('statistics')
        (self.run/'statistics/reproduction.json').unlink()
        with self.assertRaisesRegex(RuntimeError,'missing or changed'):self.review('statistics')

    def test_prediction_checks_all_horizons_models_arrays_and_raw_sources(self):
        self.prediction();receipt=self.review('fixed_prediction')
        self.assertEqual(receipt['status'],'PASS')
        self.assertEqual(receipt['checks']['exact_npz_arrays'],279)
        self.assertEqual(receipt['checks']['recorded_start_opportunities_per_horizon'],144)

    def test_prediction_retains_model_invalid_and_invalid_truth_windows(self):
        self.prediction(invalid_model=True,invalid_truth=True)
        self.assertEqual(self.review('fixed_prediction')['status'],'PASS')

    def test_prediction_checks_workspace_failure_and_retains_failed_full_state(self):
        def workspace_failure(state,action):
            delta=np.zeros_like(state);delta[:,0]=1.
            return delta
        self.prediction(model_override=workspace_failure)
        self.assertEqual(self.review('fixed_prediction')['status'],'PASS')

    def test_prediction_checks_raw_yaw_limit_without_wrapping_it_into_validity(self):
        def excessive_yaw(state,action):
            delta=np.zeros_like(state);delta[:,2]=np.pi
            return delta
        self.prediction(model_override=excessive_yaw)
        self.assertEqual(self.review('fixed_prediction')['status'],'PASS')

    def test_prediction_retains_inference_rejection_and_checks_fallback_bounds(self):
        def rejected(state,action):
            raise ValueError('scripted rejection; no actual model')
        self.prediction(model_override=rejected)
        self.assertEqual(self.review('fixed_prediction')['status'],'PASS')

    def test_prediction_zero_opportunity_population_is_explicit(self):
        self.prediction(length=0)
        self.assertEqual(self.review('fixed_prediction')['checks']['recorded_start_opportunities_per_horizon'],0)

    def test_prediction_missing_full_states_or_raw_deltas_is_rejected(self):
        self.prediction()
        self.mutate_arrays('fixed_prediction','traces.npz',lambda arrays:arrays.pop('K10__R__raw_deltas'))
        with self.assertRaises(RuntimeError):self.review('fixed_prediction')

    def test_prediction_rejects_changed_truth_mask_even_if_summary_is_unmodified(self):
        self.prediction()
        def change(arrays):arrays['K5__truth_valid'][0]=False
        self.mutate_arrays('fixed_prediction','traces.npz',change)
        with self.assertRaisesRegex(RuntimeError,'truth_valid'):self.review('fixed_prediction')

    def test_prediction_rejects_nonrecursive_states_even_with_saved_deltas(self):
        self.prediction()
        def change(arrays):arrays['K5__R__predicted_states'][0,1,0]+=.01
        self.mutate_arrays('fixed_prediction','traces.npz',change)
        with self.assertRaisesRegex(RuntimeError,'predicted_states'):self.review('fixed_prediction')

    def test_prediction_rejects_missing_error_metric_or_dtype_change(self):
        self.prediction()
        def change(arrays):arrays['K1__R__endpoint_state_error']=arrays['K1__R__endpoint_state_error'].astype(np.float32)
        self.mutate_arrays('fixed_prediction','traces.npz',change)
        with self.assertRaisesRegex(RuntimeError,'dtype'):self.review('fixed_prediction')

    def test_prediction_rejects_forged_scalar_error(self):
        self.prediction()
        def change(arrays):arrays['K1__R__object_xy_error_m'][0]=1.
        self.mutate_arrays('fixed_prediction','traces.npz',change)
        with self.assertRaisesRegex(RuntimeError,'object_xy_error'):self.review('fixed_prediction')

    def test_prediction_rejects_changed_summary_denominators(self):
        self.prediction();path=self.run/'fixed_prediction/summary.json';summary=json.loads(path.read_text())
        summary['horizons']['5']['models']['R']['truth_eligible_windows']+=1
        self.write(path,summary);self.reseal('fixed_prediction')
        with self.assertRaisesRegex(RuntimeError,'denominators'):self.review('fixed_prediction')

    def test_prediction_rejects_nonzero_or_bool_engine_guard(self):
        self.prediction();path=self.run/'fixed_prediction/summary.json';summary=json.loads(path.read_text())
        summary['future_engine_calls']['mj_step']=False;self.write(path,summary);self.reseal('fixed_prediction')
        with self.assertRaisesRegex(RuntimeError,'guard'):self.review('fixed_prediction')

    def test_prediction_rejects_impossible_inference_counts(self):
        self.prediction();path=self.run/'fixed_prediction/summary.json';summary=json.loads(path.read_text())
        summary['inference']['R']['calls']=0;self.write(path,summary);self.reseal('fixed_prediction')
        with self.assertRaisesRegex(RuntimeError,'inference counts'):self.review('fixed_prediction')

    def test_prediction_rejects_raw_source_omission_from_receipt(self):
        self.prediction();path=self.run/'fixed_prediction/stage_receipt.json';receipt=json.loads(path.read_text())
        key=next(key for key in receipt['source_sha256'] if key.endswith('.raw.npz'))
        receipt['source_sha256'].pop(key);self.write(path,receipt)
        with self.assertRaisesRegex(RuntimeError,'source graph'):self.review('fixed_prediction')

    def test_prediction_rejects_changed_raw_bytes(self):
        self.prediction();path=self.run/'E2/fixture-00/G/raw_execution.raw.npz';path.write_bytes(b'changed source')
        with self.assertRaisesRegex(RuntimeError,'source file changed'):self.review('fixed_prediction')

    def test_memory_checkpoint_rejects_before_any_loading(self):
        with patch('push_align.final_stage_review.MAX_RSS_BYTES',0):
            with self.assertRaisesRegex(RuntimeError,'RSS'):self.review('statistics')


if __name__ == '__main__':unittest.main()
