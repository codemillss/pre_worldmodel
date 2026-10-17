#!/usr/bin/env python3
"""Post-hoc illustrative GIF from an already sealed compact visual export.

No raw recording, model, physics or recovery verifier is opened/executed. The
original renderer is SHA-pinned and reused; only an extra disclosure banner is
added. This optional example never replaces the three preregistered clips.
Use bundled Pillow Python. --self-test uses synthetic compact/index files only.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile
import types
import unittest

RENDERER_SHA = 'c31e71402e90aad0755bdd9fe2c4195fea9cb96cedf69cd80dec12005fa77839'
MAX_BYTES = 6 * 1024**2
LABEL = 'POSTHOC success example - not a rate sample'
RULE = 'lexicographically_smallest_family_id_with_any_S1_or_SK_ready_among_original96'
CONTROLLERS = ('G','R','S1-11','S1-23','S1-47','SK-11','SK-23','SK-47')
STRATA = ('translation','rotation','offset')


def need(okay, message):
    if not okay: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value:
        (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+value)))


def helper():
    path = Path(__file__).resolve().with_name('p5_final_visuals_v2.py')
    source = path.read_bytes()
    need(hashlib.sha256(source).hexdigest() == RENDERER_SHA, 'Pinned v2 renderer source changed')
    # Compile the checked bytes: neither read nor create an unbound .pyc cache.
    module = types.ModuleType('p5_supplementary_bound_renderer')
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module, path


def checked_inputs(compact, visual_index, renderer):
    compact, visual_index = Path(compact).resolve(), Path(visual_index).resolve()
    need(compact.name == 'compact_visual_source.json' and visual_index.name == 'visual_index.json'
         and compact.parent == visual_index.parent, 'Original sibling compact/index paths required')
    compact_hash, index_hash = sha(compact), sha(visual_index)
    index, data = read(visual_index), read(compact)
    need(index.get('schema') == 'p5-final-visual-index-analysis-recovery-v2'
         and index.get('status') == 'GENERATED_REQUIRES_DIRECT_VISUAL_REVIEW'
         and data.get('schema') == 'p5-visual-compact-actual-boundary-analysis-recovery-v2'
         and index.get('derived_after_freeze') is True and data.get('derived_after_freeze') is True,
         'Completed additive-v2 visual export required; not an original frozen product')
    files = index.get('files_sha256')
    need(type(files) is dict and files.get(compact.name) == compact_hash
         == index.get('compact_source_sha256') and index.get('renderer_tool_sha256') == RENDERER_SHA,
         'Compact/index/renderer SHA identities differ')
    bindings = {str(visual_index): index_hash}
    for name, expected in files.items():
        need(type(name) is str and Path(name).name == name and name not in ('.','..','visual_index.json')
             and type(expected) is str and len(expected) == 64, 'Flat SHA-bound original visual files required')
        path = compact.parent / name
        need(path.is_file() and not path.is_symlink() and sha(path) == expected,
             'Original visual file missing or changed: '+name)
        bindings[str(path)] = expected
    need(index.get('original_source_bindings_sha256') == data.get('source_bindings_sha256')
         and type(data.get('source_bindings_sha256')) is dict
         and index.get('analysis_provenance') == data.get('analysis_provenance'),
         'Compact/index upstream identity maps differ')
    if data.get('fixture_only') is not True:
        provenance = data.get('analysis_provenance', {})
        need(provenance.get('schema') == 'p5-explicit-postfreeze-analysis-inputs-v2'
             and provenance.get('source_frozen_manifest_sha256') == data.get('source_frozen_manifest_sha256')
             and type(data.get('source_frozen_manifest_sha256')) is str
             and len(data['source_frozen_manifest_sha256']) == 64,
             'Compact/index frozen-study metadata identity differs')
    # These maps are compared as metadata only. Never traverse/open project raw paths.
    need(data.get('controllers') == list(CONTROLLERS), 'All original eight controllers required')
    families = data.get('families'); cells = data.get('cells')
    need(type(families) is list and type(cells) is list and len(cells) == 96,
         'Original96 compact cells required')
    selected = renderer.select_replay_families(families)
    need(data.get('selected_families') == selected == index.get('selected_families')
         and index.get('selection_rule') == renderer.REPLAY_RULE
         and data.get('visualization', {}).get('selection_rule') == renderer.REPLAY_RULE
         and data['visualization'].get('selected_families') == selected
         and data['visualization'].get('controllers') == list(CONTROLLERS)
         and data['visualization'].get('fps') == 5
         and data['visualization'].get('postterminal_tail_s') == 1., 'Original selection/timing differs')
    clips = index.get('clips')
    need(type(clips) is list and len(clips) == 3, 'Keep all three original clips')
    for stratum, clip in zip(STRATA, clips):
        need(clip.get('family_id') == selected[stratum] and clip.get('controllers') == list(CONTROLLERS)
             and clip.get('fps') == 5 and clip.get('path') in files
             and clip.get('path') in ('clip_'+stratum+'.gif', 'clip_'+stratum+'.mp4')
             and 'clip_'+stratum+'_frame_index.json' in files, 'Original clip/trace bindings incomplete')
    strata = {family['family_id']: family['stratum'] for family in families}
    need(len({cell.get('cell_id') for cell in cells}) == 96
         and {(c.get('family_id'), c.get('controller_id')) for c in cells}
         == {(family, control) for family in strata for control in CONTROLLERS},
         'Duplicate or missing family/controller/cell identity')
    for cell in cells:
        need(cell.get('stratum') == strata[cell['family_id']] and type(cell.get('success')) is bool
             and cell['success'] == (cell.get('termination') == 'ready'), 'Compact outcome identity differs')
        elapsed = cell.get('elapsed_s'); samples = cell.get('samples')
        need(type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed >= 0
             and type(samples) is list and bool(samples), 'Finite recorded clock/samples required')
        times = [s.get('relative_time_s') for s in samples]
        need(all(type(t) in (int,float) and math.isfinite(t) for t in times)
             and times[0] == 0 and all(b>a for a,b in zip(times,times[1:]))
             and abs(times[-1]-elapsed) <= 1e-7, 'Recorded boundary clocks differ')
        for sample in samples:
            state = sample.get('state10')
            need(type(state) is list and len(state) == 10
                 and all(v is None or (type(v) in (int,float) and math.isfinite(v)) for v in state),
                 'Unmodified finite-or-null recorded state10 required')
    candidates = sorted({c['family_id'] for c in cells
        if c['controller_id'] in CONTROLLERS[2:] and c['termination'] == 'ready'})
    need(bool(candidates), 'No S1/SK-ready family: no supplementary success example can be drawn')
    for path, expected in bindings.items(): need(sha(path)==expected,'Input changed during verification: '+path)
    return data, index, bindings, candidates


def banner_wrapper(renderer):
    original = renderer.replay_frame
    def draw(data, cells, label, clock, bounds):
        image, metadata = original(data, cells, 'supplementary', clock, bounds)
        canvas = renderer.Image.new('RGB', (1280, 990), '#fff2d8')
        canvas.paste(image, (0,140)); pen = renderer.ImageDraw.Draw(canvas)
        lines = ((12,LABEL,34), (55,'Selected AFTER outcomes | original 3 clips retained',27),
            (91,'Lex-first NN-ready family; NO rate denominator or superiority claim',24),
            (119,'Recorded poses only; postterminal display hold is not additional readiness evidence',19))
        for y,value,size in lines: renderer.text(pen,(22,y),value,size,'#7a3300')
        if data.get('fixture_only'):
            renderer.text(pen,(1010,12),'FIXTURE',28,'#b64039')
        return canvas, metadata
    return original, draw


def make(compact, visual_index, out, budget_bytes=MAX_BYTES):
    need(type(budget_bytes) is int and 0 < budget_bytes <= MAX_BYTES, 'At most6MiB supplementary output allowed')
    renderer, renderer_path = helper()
    compact, visual_index = Path(compact).resolve(), Path(visual_index).resolve()
    out_arg = Path(out)
    need(not out_arg.is_symlink(), 'Output cannot be a symlink')
    out = out_arg.resolve()
    need(not out.exists() and compact.parent != out and compact.parent not in out.parents,
         'Exclusive new output outside the original visuals directory required')
    data, index, bindings, candidates = checked_inputs(compact,visual_index,renderer)
    bindings[str(renderer_path)] = RENDERER_SHA; bindings[str(Path(__file__).resolve())] = sha(__file__)
    family = candidates[0]
    # Shallow top-level view only: nested original cell/sample objects are never changed.
    view = dict(data); view['selected_families'] = {LABEL:family}
    ready = [c['controller_id'] for c in data['cells'] if c['family_id']==family
             and c['controller_id'] in CONTROLLERS[2:] and c['termination']=='ready']
    before_cells = hashlib.sha256(json.dumps(data['cells'],sort_keys=True,allow_nan=False).encode()).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True); out.mkdir()
    renderer.graphics(); budget = renderer.Budget(out,budget_bytes)
    original, decorated = banner_wrapper(renderer); renderer.replay_frame = decorated
    try: clip = renderer.render_clip(view,LABEL,budget,fps=5,prefer_mp4=False)
    finally: renderer.replay_frame = original
    need(hashlib.sha256(json.dumps(data['cells'],sort_keys=True,allow_nan=False).encode()).hexdigest() == before_cells,
         'Compact cells changed in memory')
    for path, expected in bindings.items(): need(sha(path)==expected,'Input/tool changed while rendering: '+path)
    generated = {p.name:sha(p) for p in out.iterdir() if p.is_file()}
    receipt = {'schema':'p5-posthoc-supplementary-success-clip-v1',
        'status':'GENERATED_REQUIRES_DIRECT_VISUAL_REVIEW', 'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'notice':LABEL, 'derived_after_freeze':True, 'selection_rule':RULE,
        'selection_scope':'Original96 E2 cells; twelve families; any S1/SK seed termination=ready',
        'eligible_family_ids':candidates,'selected_families':view['selected_families'],
        'selected_ready_NN_controllers':sorted(ready),'panels':list(CONTROLLERS),
        'original_three_clips_retained':True,'original_selection_rule':index['selection_rule'],
        'original_selected_families':index['selected_families'],
        'source_compact':{'path':str(compact),'sha256':bindings[str(compact)]},
        'original_visual_index':{'path':str(visual_index),'sha256':bindings[str(visual_index)]},
        'renderer':{'path':str(renderer_path),'sha256':RENDERER_SHA,
            'unchanged_helpers':['render_clip','replay_frame','held_sample','world_bounds'],
            'decoration_only':'140px disclosure banner; original1280x850 frame pasted without scaling'},
        'input_files_unchanged_sha256':bindings,'cell_samples_sha256_before_after':before_cells,
        'clip':clip,'files_sha256':generated,'maximum_output_bytes':budget_bytes,
        'new_physical_calls':0,'new_model_forwards':0,'new_training_updates':0,'raw_recording_reads':0,
        'source_validation_scope':'Sealed compact/index/original visual-file cross-check only; no upstream raw or analysis rerun',
        'direct_visual_review':'NOT_YET_RECORDED','rate_estimation_denominator':None,
        'no_claims':['representative success rate','method superiority','new readiness during held display tail',
                     'fresh control replay','new physics','industrial validity','learner understanding']}
    budget.json('supplementary_manifest.json',receipt)
    need(sum(p.stat().st_size for p in out.iterdir() if p.is_file())<=budget_bytes,'Supplementary actual bytes exceed budget')
    return receipt


def fixture(folder):
    """Entirely invented short trajectories; never imports project data/config."""
    families = [{'family_id':s+f'-{i:02d}','stratum':s} for s in STRATA for i in range(4)]
    selection = {s:s+'-00' for s in STRATA}; cells=[]
    for f in families:
        for control in CONTROLLERS:
            success = f['family_id'] in ('offset-02','translation-01') and control=='SK-23'
            state=[0.,0.,0.,0.,0.,0.,-.03,0.,0.,0.]; moved=list(state);moved[0]=.002
            cells.append({**f,'cell_id':f['family_id']+'/'+control,'controller_id':control,
                'termination':'ready' if success else 'timeout','success':success,'elapsed_s':.2,
                'samples':[{'relative_time_s':t,'sim_time_s':t+.1,'state10':v,'phase':'ready' if success and t else 'running',
                    'held_time_s':.3 if success and t else 0.,'observation_valid':True} for t,v in ((0.,state),(.2,moved))]})
    data={'schema':'p5-visual-compact-actual-boundary-analysis-recovery-v2','derived_after_freeze':True,
        'fixture_only':True,'analysis_provenance':{'fixture_only':True},'source_bindings_sha256':{},
        'families':families,'cells':cells,'controllers':list(CONTROLLERS),'selected_families':selection,
        'visualization':{'selection_rule':'smallest_family_id_per_stratum_all_eight_controllers',
            'selected_families':selection,'controllers':list(CONTROLLERS),'fps':5,'postterminal_tail_s':1.},
        'physical_config':{'object':{'size_full_xyz_m':[.04,.02,.01]},'pusher':{'radius_m':.005},
            'task':{'goal_xy_m':[.08,0.],'goal_yaw_rad':0.,'position_tolerance_m':.005}}}
    compact=folder/'compact_visual_source.json';compact.write_text(json.dumps(data))
    files={compact.name:sha(compact)};clips=[]
    for s in STRATA:
        for name in ('clip_'+s+'.gif','clip_'+s+'_frame_index.json'):
            path=folder/name;path.write_bytes(b'SYNTHETIC INDEX-BINDING FIXTURE ONLY');files[name]=sha(path)
        clips.append({'family_id':selection[s],'controllers':list(CONTROLLERS),'fps':5,'path':'clip_'+s+'.gif'})
    index={'schema':'p5-final-visual-index-analysis-recovery-v2','status':'GENERATED_REQUIRES_DIRECT_VISUAL_REVIEW',
        'derived_after_freeze':True,'compact_source_sha256':sha(compact),'renderer_tool_sha256':RENDERER_SHA,
        'original_source_bindings_sha256':{},'analysis_provenance':data['analysis_provenance'],
        'selected_families':selection,'selection_rule':data['visualization']['selection_rule'],
        'clips':clips,'files_sha256':files}
    path=folder/'visual_index.json';path.write_text(json.dumps(index))
    return compact,path


def self_test():
    class Checks(unittest.TestCase):
        def test_lex_selection_and_input_preservation(self):
            with tempfile.TemporaryDirectory() as temporary:
                base=Path(temporary);source=base/'original';source.mkdir();compact,index=fixture(source)
                renderer,_=helper();data,_,bindings,candidates=checked_inputs(compact,index,renderer)
                self.assertEqual(candidates,['offset-02','translation-01'])
                before={p:sha(p) for p in bindings};result=make(compact,index,base/'new')
                self.assertEqual(result['selected_families'],{LABEL:'offset-02'})
                self.assertEqual(result['panels'],list(CONTROLLERS))
                self.assertEqual(before,{p:sha(p) for p in bindings})
                trace=read(base/'new'/('clip_'+LABEL+'_frame_index.json'))
                self.assertEqual(trace['frame_count'],7);self.assertEqual(trace['fps'],5)
                for row in trace['frames']:
                    self.assertEqual(len(row['raw_measurements']),8)
                    for sample in row['raw_measurements']:
                        self.assertEqual(sample['drawn_raw_index'],0 if row['display_clock_s']<.2 else 1)
                from PIL import Image
                for i in (0,3,6):
                    with Image.open(base/'new'/f'clip_{LABEL}_frame_{i:04d}.png') as image:
                        self.assertEqual(image.size,(1280,990))
                with self.assertRaises(ValueError):make(compact,index,base/'new')
        def test_changed_old_clip_rejected(self):
            with tempfile.TemporaryDirectory() as temporary:
                folder=Path(temporary);compact,index=fixture(folder)
                (folder/'clip_translation.gif').write_bytes(b'changed')
                with self.assertRaises(ValueError):checked_inputs(compact,index,helper()[0])
        def test_no_nn_success_not_replaced_by_G_success(self):
            with tempfile.TemporaryDirectory() as temporary:
                folder=Path(temporary);compact,index=fixture(folder);data=read(compact)
                for c in data['cells']:c['success']=c['controller_id']=='G';c['termination']='ready' if c['success'] else 'timeout'
                compact.write_text(json.dumps(data));record=read(index)
                record['compact_source_sha256']=record['files_sha256'][compact.name]=sha(compact);index.write_text(json.dumps(record))
                with self.assertRaises(ValueError):checked_inputs(compact,index,helper()[0])
        def test_budget_and_exclusive_out(self):
            with tempfile.TemporaryDirectory() as temporary:
                base=Path(temporary);source=base/'original';source.mkdir();compact,index=fixture(source)
                with self.assertRaises(ValueError):make(compact,index,base/'tiny',1024)
                self.assertLessEqual(sum(p.stat().st_size for p in (base/'tiny').iterdir()),1024)
                with self.assertRaises(ValueError):make(compact,index,base/'large',MAX_BYTES+1)
                with self.assertRaises(ValueError):make(compact,index,source/'nested')
    outcome=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if not outcome.wasSuccessful():raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compact');parser.add_argument('--visual-index');parser.add_argument('--out-dir')
    parser.add_argument('--budget-bytes',type=int,default=MAX_BYTES);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test();return
    if not all((args.compact,args.visual_index,args.out_dir)):parser.error('--compact, --visual-index and --out-dir required')
    result=make(args.compact,args.visual_index,args.out_dir,args.budget_bytes)
    print(json.dumps({'status':result['status'],'selected_families':result['selected_families'],
        'notice':result['notice'],'maximum_output_bytes':args.budget_bytes},ensure_ascii=False))


if __name__=='__main__':main()
