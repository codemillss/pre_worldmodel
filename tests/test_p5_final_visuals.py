"""Presentation fixtures only; no actual final families, models or physics."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from push_align.config import load_pilot_config
from push_align.environment import SnapshotBundle
from push_align.final_manifest import family_records, expected_E2
from push_align.final_recording import write_witness_gzip
from push_align.final_recording_columns import write_raw
from test_p5_final_prediction import fixtures

TOOL_PATH = Path(__file__).resolve().parents[1] / 'tools/p5_final_visuals.py'
spec = importlib.util.spec_from_file_location('p5_final_visuals_fixture_module', TOOL_PATH)
visuals = importlib.util.module_from_spec(spec); spec.loader.exec_module(visuals)


def compact_fixture():
    config = load_pilot_config()
    families, _ = family_records(config, seed=123, new_family_seed=456, initial_id_prefix='fixture')
    expected = expected_E2(families); cells = []
    for index, cell in enumerate(expected):
        initial = [0., 0., 0., 0., 0., 0., -.06, 0., 0., 0.]
        moved = list(initial); moved[0] = .001 * (index % 8 + 1)
        success = index % 3 == 0
        cells.append({**cell, 'termination':'ready' if success else 'timeout', 'success':success,
            'elapsed_s':.2, 'samples':[
                dict(relative_time_s=0.,sim_time_s=.126,state10=initial,observation_valid=True,phase='running',held_time_s=0.),
                dict(relative_time_s=.2,sim_time_s=.326,state10=moved,observation_valid=True,phase='ready' if success else 'failed',held_time_s=.3 if success else 0.)],
            'metrics':{'final_position_error_m':.01,'final_yaw_error_rad':.02,'final_linear_speed_m_s':0.,
                       'final_angular_speed_rad_s':0.,'metric_state_origin':'terminal_observation',
                       'metric_state_time_s':.2,'final_observation_available':True}})
    metrics = ('success','final_position_error_m','final_yaw_error_rad','elapsed_s','planning_wall_s','execution_wall_s')
    comparisons = [{'left':left,'right':right,'metrics':{name:{'difference_mean':-.01,'percentile95_interval':[-.02,.01],
                     'eligible_family_denominator':12} for name in metrics}}
                   for left,right in [('R','G'),('S1','G'),('SK','G'),('S1','R'),('SK','R'),('SK','S1'),
                                     ('SK-11','S1-11'),('SK-23','S1-23'),('SK-47','S1-47')]]
    selection=visuals.select_replay_families(families)
    return {'schema':'p5-visual-compact-actual-boundary-v1','source_bindings_sha256':{},
        'fixture_only':True,
        'source_frozen_manifest_sha256':'fixture_only','families':families,'selected_families':selection,
        'controllers':list(visuals.CONTROLLERS),'physical_config':config,
        'statistics':{'comparisons':comparisons},'cells':cells,
        'visualization':dict(selection_rule=visuals.REPLAY_RULE,selected_families=selection,
                             controllers=list(visuals.CONTROLLERS),fps=5,postterminal_tail_s=1.),
        'render_scope':'Synthetic fixture raw pose replay only; not an executed robot result'}


class FinalVisualsTests(unittest.TestCase):
    def test_selection_is_fixed_by_id_and_never_by_success(self):
        data=compact_fixture(); selected=visuals.select_replay_families(list(reversed(data['families'])))
        self.assertEqual(selected,data['selected_families'])
        self.assertTrue(all(name.endswith('-00') for name in selected.values()))
        with self.assertRaises(ValueError):visuals.select_replay_families(data['families'][:-1])

    def test_sample_selection_is_measurement_hold_and_postterminal_freeze(self):
        cell=compact_fixture()['cells'][0]
        held=visuals.held_sample(cell,.15)
        self.assertEqual(held['selected_index'],0)
        self.assertEqual(held['display']['state10'][0],0.)
        held=visuals.held_sample(cell,1.)
        self.assertEqual(held['selected_index'],1)
        self.assertEqual(held['display']['state10'][0],.001)
        self.assertTrue(held['terminal'])

    def test_nonfinite_measurement_is_marked_and_previous_actual_pose_is_explicit(self):
        cell=compact_fixture()['cells'][0];cell['samples'][1]['state10'][0]=None
        held=visuals.held_sample(cell,.2)
        self.assertEqual(held['selected_index'],1)
        self.assertEqual(held['display_index'],0)
        self.assertTrue(held['nonfinite_selected_observation'])

    def test_budget_is_hard_and_pillow_cannot_bypass_it_with_a_file_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            budget=visuals.Budget(directory,10)
            with budget.file('fixture.bin') as stream:
                stream.write(b'12345678')
                with self.assertRaises(io.UnsupportedOperation):stream.fileno()
                with self.assertRaises(ValueError):stream.write(b'abc')
            self.assertEqual((Path(directory)/'fixture.bin').stat().st_size,8)
            self.assertEqual(budget.used,8)

    def test_real_columnar_codec_extraction_uses_only_recorded_observations(self):
        trajectories,_,config=fixtures(2);source=trajectories[0]
        cell=dict(family_id=source['family_id'],episode_id=source['family_id'],episode_index=0,
                  stratum=source['stratum'],controller_id='G',cell_id=source['family_id']+'/G')
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            snapshot=SnapshotBundle(np.zeros(12,np.float64),'fixture-model','fixture-env',.01,
                {'bit_generator':'PCG64','state':{'state':1,'inc':3},'has_uint32':0,'uinteger':0},
                np.zeros(2,np.int32),np.zeros(2,np.int32),False)
            raw=write_raw(folder/'raw_execution',snapshot,source['results'])
            rows=[]
            for i,step in enumerate(source['results']):
                rows.append(dict(index=i,state_after=step.observation.state.tolist(),sim_time_after=step.observation.sim_time,
                                 geometry={'pass':True},task_after={'phase':'running','held_time_s':0.}))
            witness={'initial':{'sim_time':.126,'state':source['initial_state10'].tolist(),'observation_valid':True,
                                'task':{'phase':'running','held_time_s':0.}},'intervals':rows,'decisions':[],
                     'termination':'timeout','execution_status':'completed'}
            wr=write_witness_gzip(folder/'witness.json.gz',witness)
            manifest={'execution':{'controller_id':'G','case_id':source['family_id']},'raw':{'prefix':str(folder/'raw_execution'),**raw},
                      'raw_codec_module':'push_align.final_recording_columns','witness_path':str(folder/'witness.json.gz'),
                      'files_sha256':wr['files_sha256']}
            (folder/'manifest.json').write_text(json.dumps(manifest))
            row={**cell,'execution_status':'completed','termination':'timeout','success':False,'elapsed_s':.2}
            manifest_hash=visuals.sha(folder/'manifest.json')
            review={'status':'ACCEPT_COMPLETED_TASK_FAILURE','accepted':True,'manifest_sha256':manifest_hash,
                    'metrics':{'success':False,'elapsed_s':.2}}
            (folder/'review.json').write_text(json.dumps(review))
            worker={'status':'ACCEPT_COMPLETED_TASK_FAILURE','row':row,'record_manifest_sha256':manifest_hash,
                    'review_sha256':visuals.sha(folder/'review.json')}
            (folder/'worker_result.json').write_text(json.dumps(worker))
            (folder/'worker_result_receipt.json').write_text(json.dumps({'sha256':visuals.sha(folder/'worker_result.json')}))
            bindings={};compact=visuals._extract_cell(folder,cell,bindings)
            self.assertEqual(len(compact['samples']),3)
            self.assertEqual(compact['samples'][-1]['state10'],source['results'][-1].observation.state.tolist())
            self.assertAlmostEqual(compact['samples'][-1]['relative_time_s'],.2)
            self.assertEqual(len(bindings),7)
            with self.assertRaisesRegex(ValueError,'different frozen study'):
                visuals._extract_cell(folder,cell,{},expected_study_sha256='different-fixture-study')
            (folder/'review.json').write_text('changed independent review')
            with self.assertRaisesRegex(ValueError,'Changed visual source'):
                visuals._extract_cell(folder,cell,{})
            (folder/'worker_result.json').write_text('changed')
            self.assertNotEqual(visuals.sha(folder/'worker_result.json'),bindings[str((folder/'worker_result.json').resolve())])

    def test_bundled_pillow_renders_three_full_controller_gifs_and_bound_figures(self):
        self.assertTrue(visuals.DEFAULT_RENDERER.is_file())
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);source=folder/'compact_visual_source.json'
            source.write_text(json.dumps(compact_fixture()))
            result=subprocess.run([str(visuals.DEFAULT_RENDERER),str(TOOL_PATH),'--render-compact',str(source),
                '--out',str(folder),'--gif'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            index=json.loads((folder/'visual_index.json').read_text())
            self.assertEqual(index['status'],'GENERATED_REQUIRES_DIRECT_VISUAL_REVIEW')
            self.assertEqual(index['direct_visual_review'],'NOT_YET_RECORDED')
            self.assertEqual(len(index['clips']),3)
            self.assertEqual(index['compact_source_sha256'],visuals.sha(source))
            self.assertLess(sum(p.stat().st_size for p in folder.iterdir()),64*1024**2)
            for clip in index['clips']:
                self.assertEqual(clip['controllers'],list(visuals.CONTROLLERS))
                self.assertEqual((folder/clip['path']).read_bytes()[:6],b'GIF89a')
                frames=json.loads((folder/('clip_'+next(s for s,f in index['selected_families'].items() if f==clip['family_id'])+'_frame_index.json')).read_text())
                self.assertFalse(frames['interpolation'])
                self.assertTrue(all(len(frame['raw_measurements'])==8 for frame in frames['frames']))
            self.assertTrue(all((folder/name).is_file() for name in ('all_endpoints.png','all_trajectories.png','bootstrap_intervals.png')))


if __name__=='__main__':unittest.main()
