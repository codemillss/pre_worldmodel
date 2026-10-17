"""Plot the cached candidate-77 learned predictions and recorded execution.

Read immutable P4 cost diagnostic arrays only; no physics, model inference,
learning, task re-scoring, or PDF authoring. Connecting lines are visual aids
between recorded/predicted action boundaries, not intermediate state samples.
"""
from pathlib import Path
import hashlib
import json
import math
import subprocess

import numpy as np
from reportlab.graphics import renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Line, Path as Curve, Rect, String
from reportlab.lib.colors import HexColor

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/p4-cost-20261013-02'
REUSED = ROOT/'runs/p4-cost-20261013-01'
OUT = ROOT/'reports/figures'
STEM = 'p4_cost_prediction_vs_actual'
INDEX = 77


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load_curves():
    manifest, summary, initial = (read(RUN/name) for name in ('manifest.json', 'summary.json', 'input.json'))
    assert summary['pass'] is True and summary['learning_updates'] == 0
    assert summary['final_test_used'] is False
    task = manifest['evaluation_planner']['task']
    goal = np.asarray(task['goal_xy_m'], np.float64)
    threshold = task['position_tolerance_m']*1000.
    assert threshold == 5. and initial['context']['control_step'] == 99
    initial_state = np.asarray(initial['state'], np.float64)
    paths = ['input.json']
    curves = []
    common_actions = None
    for name, label, color, dash in (
            ('MLP-11', 'Common-11 prediction', '#166A8F', None),
            ('S1-11', 'S1-11 prediction', '#774B9B', [15, 7]),
            ('SK-11', 'SK-11 prediction', '#CF7927', [4, 7])):
        folder = RUN/'initial_models'/name
        with np.load(folder/'predictions.npz', allow_pickle=False) as saved:
            state = saved['states'][INDEX].copy()
            actions = saved['full_actions'][INDEX].copy()
            surrogate_step = int(saved['task_first_ready_step'][INDEX])
            assert saved['valid'][INDEX] and saved['task_eligible'][INDEX] and saved['task_ready'][INDEX]
            assert state.shape == (11, 10) and np.isfinite(state).all()
            assert np.array_equal(state[0], initial_state)
            assert surrogate_step == 7
        if common_actions is not None:
            assert np.array_equal(actions, common_actions)
        common_actions = actions
        assert np.array_equal(actions[5:], np.zeros((5, 2), np.float64))
        choices = read(folder/'choices.json')
        selected = next(row for row in choices if row['arm'] == 'TASK_H10_stop')
        assert selected['selected_index'] == INDEX and selected['predicted_surrogate_ready']
        assert selected['first_surrogate_ready_step'] == surrogate_step
        errors = np.hypot(state[:, 0]-goal[0], state[:, 1]-goal[1])*1000.
        curves.append({'model_id': name, 'label': label, 'kind': 'learned_prediction',
            'color': color, 'dash': dash, 'elapsed_s': np.arange(11, dtype=np.float64)*.1,
            'position_error_mm': errors, 'predicted_surrogate_first_ready_step': surrogate_step,
            'predicted_surrogate_first_ready_s': surrogate_step*.1,
            'last_position_error_mm': float(errors[-1])})
        paths.extend([f'initial_models/{name}/predictions.npz', f'initial_models/{name}/choices.json'])
    folder = RUN/f'actual/candidate-{INDEX:03d}/repeat-0'
    result = read(folder/'result.json')
    assert result['physical_valid'] and result['completed_intervals'] == 10
    assert result['raw_task']['first_ready_interval'] is None
    assert result['raw_task']['status']['phase'] == 'running'
    with np.load(folder/'raw_execution.npz', allow_pickle=False) as saved:
        endpoints = saved['state'].copy()
        dt = saved['actual_dt'].copy()
        raw_state, raw_time = saved['substep_states'].copy(), saved['substep_times'].copy()
        assert endpoints.shape == (10, 10) and np.isfinite(endpoints).all()
        assert np.array_equal(saved['action_applied'], common_actions)
        assert np.array_equal(saved['action_requested'], common_actions)
        assert np.allclose(dt, .1, rtol=0, atol=1e-9)
    skip = len(raw_time)//10
    assert skip == 200 and raw_state.shape == (2000, 10)
    assert np.array_equal(endpoints, raw_state[skip-1::skip])
    elapsed = np.r_[0., raw_time[skip-1::skip]-initial['status']['last_sim_time']]
    assert np.allclose(elapsed[1:], np.cumsum(dt), rtol=0, atol=1e-9)
    assert np.allclose(elapsed, curves[0]['elapsed_s'], rtol=0, atol=1e-9)
    actual_states = np.vstack([initial_state, endpoints])
    actual_errors = np.hypot(actual_states[:, 0]-goal[0], actual_states[:, 1]-goal[1])*1000.
    assert abs(actual_errors[-1]-result['final_metrics']['e_pos_m']*1000.) < 1e-12
    curves.append({'model_id': None, 'label': 'Recorded simulation', 'kind': 'recorded_execution',
        'color': '#172B3A', 'dash': None, 'elapsed_s': elapsed,
        'position_error_mm': actual_errors, 'raw_task_first_ready_interval': None,
        'raw_task_phase_after_1_s': 'running',
        'last_position_error_mm': float(actual_errors[-1])})
    paths.extend([f'actual/candidate-{INDEX:03d}/repeat-0/raw_execution.npz',
                  f'actual/candidate-{INDEX:03d}/repeat-0/result.json'])
    source_index = read(RUN/'artifact_hashes.json')['sha256']
    hashes, reuse = {}, {}
    for relative in paths:
        digest = sha(RUN/relative)
        assert source_index[relative] == digest
        assert sha(REUSED/relative) == digest
        hashes[str((RUN/relative).relative_to(ROOT))] = digest
        reuse[relative] = {'source_sha256': sha(REUSED/relative), 'official_copy_sha256': digest,
                           'byte_exact': True}
    return curves, threshold, goal, hashes, reuse


def main():
    curves, threshold, goal, hashes, reuse = load_curves()
    all_values = np.concatenate([row['position_error_mm'] for row in curves])
    low = math.floor((min(float(all_values.min()), threshold)-.15)/.25)*.25
    high = math.ceil((max(float(all_values.max()), threshold)+.15)/.25)*.25
    assert low < all_values.min() and high > all_values.max()
    width, height = 1600, 960
    drawing = Drawing(width, height)
    c = HexColor
    ink, muted, grid, limit = '#172B3A', '#5D7080', '#D7E1E7', '#A84139'
    drawing.add(Rect(0, 0, width, height, fillColor=c('#FFFFFF'), strokeColor=None))
    drawing.add(String(72, 908, 'Candidate 77: predicted approach, recorded position unchanged',
                       fontSize=36, fillColor=c(ink)))
    drawing.add(String(72, 861, 'Same H5 commands + five zero commands | One reused state at source tick 99',
                       fontSize=28, fillColor=c(muted)))
    for x, row in zip((72, 502, 842, 1166), curves):
        drawing.add(Line(x, 805, x+52, 805, strokeColor=c(row['color']),
                         strokeWidth=5, strokeDashArray=row['dash']))
        drawing.add(String(x+65, 795, row['label'], fontSize=28, fillColor=c(ink)))
    drawing.add(String(145, 745, 'XY goal distance (mm)', fontSize=29, fillColor=c(muted)))
    drawing.add(String(635, 745, 'Predicted endpoint surrogate ready: 0.7 s (all models)',
                       fontSize=29, fillColor=c(ink)))
    x, y, w, h = 145, 272, 1350, 414
    px = lambda value: x+value*w
    py = lambda value: y+(value-low)/(high-low)*h
    drawing.add(Rect(px(.5), y, w/2, h, fillColor=c('#EEF3F7'), strokeColor=None))
    drawing.add(String(px(.25), y+h+20, 'Original H5 commands', textAnchor='middle',
                       fontSize=28, fillColor=c(muted)))
    drawing.add(String(px(.75), y+h+20, 'Zero-command suffix', textAnchor='middle',
                       fontSize=28, fillColor=c(muted)))
    for tick in np.arange(low, high+.125, .25):
        drawing.add(Line(x, py(tick), x+w, py(tick), strokeColor=c(grid), strokeWidth=1))
        drawing.add(String(x-18, py(tick)-10, f'{tick:g}', textAnchor='end',
                           fontSize=29, fillColor=c(muted)))
    for tick in np.arange(0., 1.01, .2):
        drawing.add(Line(px(tick), y, px(tick), y+h, strokeColor=c(grid), strokeWidth=1))
        drawing.add(String(px(tick), y-38, f'{tick:g}', textAnchor='middle',
                           fontSize=29, fillColor=c(muted)))
    drawing.add(Line(px(.5), y, px(.5), y+h, strokeColor=c('#899AAA'), strokeWidth=2))
    drawing.add(Line(px(.7), y, px(.7), y+h, strokeColor=c('#A3ABB5'),
                     strokeWidth=2, strokeDashArray=[4, 7]))
    drawing.add(Line(x, py(threshold), x+w, py(threshold), strokeColor=c(limit),
                     strokeWidth=3, strokeDashArray=[10, 7]))
    drawing.add(String(x+w-14, py(threshold)+15, '5 mm position limit', textAnchor='end',
                       fontSize=28, fillColor=c(limit)))
    drawing.add(Line(x, y, x, y+h, strokeColor=c(ink), strokeWidth=1.5))
    drawing.add(Line(x, y, x+w, y, strokeColor=c(ink), strokeWidth=1.5))
    for row in curves:
        assert row['position_error_mm'].min() >= low and row['position_error_mm'].max() <= high
        curve = Curve(strokeColor=c(row['color']), fillColor=None,
                      strokeWidth=5 if row['kind'] == 'recorded_execution' else 4,
                      strokeDashArray=row['dash'])
        for step, (time, value) in enumerate(zip(row['elapsed_s'], row['position_error_mm'])):
            if step == 0:
                curve.moveTo(px(time), py(value))
            else:
                curve.lineTo(px(time), py(value))
        drawing.add(curve)
    actual = curves[-1]
    drawing.add(Circle(px(1.), py(actual['last_position_error_mm']), 5,
                       fillColor=c('#FFFFFF'), strokeColor=c(ink), strokeWidth=3))
    drawing.add(String(x+w-14, py(actual['last_position_error_mm'])+18,
                       f'Recorded: {actual["last_position_error_mm"]:.3f} mm',
                       textAnchor='end', fontSize=29, fillColor=c(ink)))
    drawing.add(String(x+w/2, 182, 'Time from saved source state (s)', textAnchor='middle',
                       fontSize=30, fillColor=c(ink)))
    drawing.add(String(72, 111, 'All models predict surrogate ready at 0.7 s; raw-task assessment is not ready within 1 s.',
                       fontSize=28, fillColor=c(ink)))
    drawing.add(String(72, 69, 'Position alone is not handoff. Recorded action boundaries; no new inference or physics for this figure.',
                       fontSize=28, fillColor=c(muted)))
    OUT.mkdir(parents=True, exist_ok=True)
    svg, png = OUT/(STEM+'.svg'), OUT/(STEM+'.png')
    renderSVG.drawToFile(drawing, str(svg))
    subprocess.run(['/opt/homebrew/bin/rsvg-convert', '-w', str(width), '-h', str(height),
                    str(svg), '-o', str(png)], check=True)
    for path, digest in hashes.items():
        assert sha(ROOT/path) == digest
    data = {'format': 'p4-cost-candidate77-prediction-vs-actual-position-v1',
        'official_run': str(RUN.relative_to(ROOT)), 'reused_source_run': str(REUSED.relative_to(ROOT)),
        'original_candidate_index': INDEX, 'source_tick': 99, 'model_seed': 11,
        'goal_xy_m': goal.tolist(), 'position_limit_mm': threshold,
        'png_dimensions_px': [width, height], 'minimum_font_size_px': 28,
        'axis': {'elapsed_s': [0., 1.], 'position_error_mm': [low, high]},
        'axis_rule': 'Enclose all four curves and threshold, pad by at least0.15mm and round outward to0.25mm; no clipping',
        'zero_suffix_elapsed_s': [.5, 1.], 'samples_per_curve': 11,
        'recorded_raw_substep_count': 2000,
        'state_sources': 'Initial input observation; three stored learned H10 arrays; recorded ten action endpoints exactly matching every200th raw substep',
        'line_interpolation': 'Connecting action boundaries is a plotting aid; no intermediate predictions or physical samples are fabricated',
        'task_scope': 'Surrogate .1s endpoint hold is distinct from raw-substep actual task. Position alone cannot establish handoff. One reused development state and learning seed.',
        'source_sha256': hashes, 'source_copies_01_to_02_byte_exact': reuse,
        'source_summary_sha256': sha(RUN/'summary.json'),
        'source_manifest_sha256': sha(RUN/'manifest.json'),
        'source_artifact_index_sha256': sha(RUN/'artifact_hashes.json'),
        'plot_script_sha256': sha(__file__), 'png_sha256': sha(png), 'svg_sha256': sha(svg),
        'curves': [{key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in row.items()} for row in curves],
        'new_physics_intervals': 0, 'new_model_inference': 0, 'new_learning_updates': 0}
    json_path = OUT/(STEM+'.data.json')
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'png': str(png), 'axis': data['axis'], 'source_copies_exact': True,
        'last_error_mm': {row['label']:row['last_position_error_mm'] for row in curves},
        'png_sha256': sha(png), 'svg_sha256': sha(svg), 'data_sha256': sha(json_path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
