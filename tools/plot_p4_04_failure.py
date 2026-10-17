"""Actual position-error plot from the three immutable P4-04 main cells.

Read recorded action-boundary states and physical endpoint times, not model
predictions or newly executed physics. SVG/PNG only; no PDF is authored.
"""
from pathlib import Path
import hashlib
import json
import subprocess

import numpy as np
from reportlab.graphics import renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Line, Path as Curve, Rect, String
from reportlab.lib.colors import HexColor

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/p4-04-20261013-01'
OUT = ROOT/'reports/figures'
STEM = 'p4_04_actual_position_error'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load_curves():
    manifest, summary = read(RUN/'manifest.json'), read(RUN/'summary.json')
    assert summary['execution_pass'] is True and summary['learning_updates'] == 0
    goal = np.asarray(manifest['planner']['task']['goal_xy_m'], dtype=np.float64)
    threshold = manifest['planner']['task']['position_tolerance_m']*1000.
    assert threshold == 5.
    curves = []
    for name, label, color, dash in (
            ('MLP-11', 'Common-11', '#166A8F', None),
            ('S1-11', 'S1-11', '#774B9B', [15, 7]),
            ('SK-11', 'SK-11', '#CF7927', [4, 7])):
        folder = RUN/'closed_loop'/name
        cell = read(folder/'summary.json')
        cell_manifest = read(folder/'manifest.json')
        with (folder/'steps.jsonl').open() as file:
            first_step = json.loads(next(file))
        before = np.asarray(first_step['state_before'], dtype=np.float64)
        assert np.array_equal(before, np.asarray(cell_manifest['initial_state'], dtype=np.float64))
        assert cell_manifest['planner_config']['task'] == manifest['planner']['task']
        initial_time = first_step['task']['initial_sim_time']
        skip = cell_manifest['physical_contract']['frame_skip']
        with np.load(folder/'raw_execution.npz', allow_pickle=False) as source:
            state = source['state'].copy()
            all_times = source['substep_times'].copy()
            all_states = source['substep_states'].copy()
            dt = source['actual_dt'].copy()
        assert state.shape == (300, 10) and len(all_times) == len(all_states) == 300*skip
        assert np.array_equal(state, all_states[skip-1::skip])
        assert np.isfinite(state).all() and np.isfinite(all_states).all()
        elapsed = np.r_[0., all_times[skip-1::skip]-initial_time]
        assert np.all(np.diff(elapsed) > 0) and abs(elapsed[-1]-30.) < 1e-8
        assert np.allclose(elapsed[1:], np.cumsum(dt), rtol=0, atol=1e-9)
        positions = np.vstack([before[:2], state[:, :2]])
        error_mm = np.hypot(positions[:, 0]-goal[0], positions[:, 1]-goal[1])*1000.
        all_error_mm = np.hypot(all_states[:, 0]-goal[0], all_states[:, 1]-goal[1])*1000.
        expected_mm = cell['final_metrics']['e_pos_m']*1000.
        assert abs(error_mm[-1]-expected_mm) < 1e-12
        assert cell['termination'] == 'timeout' and cell['transitions'] == 300
        assert cell['task']['phase'] == 'failed' and cell['task']['held_time_s'] == 0.
        curves.append({'model_id': name, 'label': label, 'color': color, 'dash': dash,
            'elapsed_s': elapsed, 'position_error_mm': error_mm,
            'all_recorded_substep_minimum_position_error_mm': float(all_error_mm.min()),
            'raw_substep_count': len(all_times),
            'final_error_mm': float(error_mm[-1]), 'summary_final_error_mm': expected_mm,
            'final_value_difference_mm': abs(float(error_mm[-1])-expected_mm),
            'termination': cell['termination'], 'held_time_s': cell['task']['held_time_s'],
            'source_sha256': {str((folder/path).relative_to(ROOT)): sha(folder/path)
                for path in ('raw_execution.npz', 'steps.jsonl', 'manifest.json', 'summary.json')}})
    assert all(np.array_equal(curves[0]['elapsed_s'], row['elapsed_s']) for row in curves[1:])
    return curves, threshold, goal


def main():
    curves, threshold, goal = load_curves()
    width, height = 1600, 940
    drawing = Drawing(width, height)
    ink, muted, grid, limit = '#172B3A', '#5D7080', '#D7E1E7', '#A84139'
    c = HexColor
    drawing.add(Rect(0, 0, width, height, fillColor=c('#FFFFFF'), strokeColor=None))
    drawing.add(String(72, 892, 'Actual position error in the three closed-loop runs',
                       fontSize=38, fillColor=c(ink)))
    drawing.add(String(72, 846, 'Same initial state | One reused development case | Frozen models, seed 11',
                       fontSize=28, fillColor=c(muted)))
    for x, row in zip((74, 464, 790), curves):
        drawing.add(Line(x, 789, x+58, 789, strokeColor=c(row['color']),
                         strokeWidth=5, strokeDashArray=row['dash']))
        drawing.add(String(x+74, 780, row['label'], fontSize=31, fillColor=c(ink)))
    drawing.add(Line(1150, 789, 1208, 789, strokeColor=c(limit), strokeWidth=3,
                     strokeDashArray=[10, 7]))
    drawing.add(String(1222, 780, '5 mm limit', fontSize=31, fillColor=c(ink)))

    def panel(x, y, w, h, xlimits, ylimits, xticks, yticks, title, show_final):
        left, right = xlimits
        low, high = ylimits
        px = lambda value: x+(value-left)/(right-left)*w
        py = lambda value: y+(value-low)/(high-low)*h
        drawing.add(String(x, y+h+55, title, fontSize=31, fillColor=c(ink)))
        drawing.add(String(x, y+h+14, 'XY goal distance (mm)', fontSize=28, fillColor=c(muted)))
        drawing.add(Rect(x, y, w, py(threshold)-y, fillColor=c('#F1F7F2'), strokeColor=None))
        for value in yticks:
            drawing.add(Line(x, py(value), x+w, py(value), strokeColor=c(grid), strokeWidth=1))
            drawing.add(String(x-17, py(value)-10, str(value), textAnchor='end',
                               fontSize=29, fillColor=c(muted)))
        for value in xticks:
            drawing.add(Line(px(value), y, px(value), y+h, strokeColor=c(grid), strokeWidth=1))
            drawing.add(String(px(value), y-39, str(value), textAnchor='middle',
                               fontSize=29, fillColor=c(muted)))
        drawing.add(Line(x, py(threshold), x+w, py(threshold), strokeColor=c(limit),
                         strokeWidth=3, strokeDashArray=[10, 7]))
        drawing.add(Line(x, y, x, y+h, strokeColor=c(ink), strokeWidth=1.5))
        drawing.add(Line(x, y, x+w, y, strokeColor=c(ink), strokeWidth=1.5))
        for row in curves:
            keep = (row['elapsed_s'] >= left-1e-8) & (row['elapsed_s'] <= right+1e-8)
            times = row['elapsed_s'][keep]
            values = row['position_error_mm'][keep]
            assert values.min() >= low and values.max() <= high
            line = Curve(strokeColor=c(row['color']), fillColor=None,
                         strokeWidth=4, strokeDashArray=row['dash'])
            for index, (time, value) in enumerate(zip(times, values)):
                if index == 0:
                    line.moveTo(px(time), py(value))
                else:
                    line.lineTo(px(time), py(value))
            drawing.add(line)
            if show_final:
                value = row['final_error_mm']
                drawing.add(Circle(px(times[-1]), py(value), 5, fillColor=c('#FFFFFF'),
                                   strokeColor=c(row['color']), strokeWidth=3))
                drawing.add(String(x+w+19, py(value)-10, f'{value:.3f}',
                                   fontSize=30, fillColor=c(row['color'])))
        if show_final:
            drawing.add(String(x+w+16, y+h+14, 'Final', fontSize=28, fillColor=c(muted)))
        drawing.add(String(x+w/2, y-92, 'Elapsed time (s)', textAnchor='middle',
                           fontSize=30, fillColor=c(ink)))

    panel(120, 252, 775, 430, (0., 30.), (0., 85.), (0, 5, 10, 15, 20, 25, 30),
          (0, 20, 40, 60, 80), 'Full 30 s (0-85 mm)', False)
    panel(1070, 252, 330, 430, (20., 30.), (0., 10.), (20, 25, 30),
          (0, 2, 4, 6, 8, 10), 'Last 10 s (0-10 mm)', True)
    drawing.add(String(72, 112, 'All three runs timed out. Position is only one of the handoff conditions.',
                       fontSize=28, fillColor=c(ink)))
    drawing.add(String(72, 70, 'Action boundaries shown. One reused case and seed; no general superiority claim.',
                       fontSize=28, fillColor=c(muted)))
    OUT.mkdir(parents=True, exist_ok=True)
    svg, png = OUT/(STEM+'.svg'), OUT/(STEM+'.png')
    renderSVG.drawToFile(drawing, str(svg))
    subprocess.run(['/opt/homebrew/bin/rsvg-convert', '-w', str(width), '-h', str(height),
                    str(svg), '-o', str(png)], check=True)
    source_index = read(RUN/'artifact_hashes.json')['sha256']
    for row in curves:
        for source, digest in row['source_sha256'].items():
            relative = str((ROOT/source).relative_to(RUN))
            assert source_index[relative] == digest and sha(ROOT/source) == digest
    rows = [{key: value.tolist() if isinstance(value, np.ndarray) else value
             for key, value in row.items()} for row in curves]
    data = {'format': 'p4-04-main-actual-position-plot-v1', 'run': str(RUN.relative_to(ROOT)),
        'source_summary_sha256': sha(RUN/'summary.json'),
        'source_manifest_sha256': sha(RUN/'manifest.json'),
        'source_artifact_index_sha256': sha(RUN/'artifact_hashes.json'),
        'plot_script_sha256': sha(__file__), 'goal_xy_m': goal.tolist(),
        'position_limit_mm': threshold, 'png_dimensions_px': [width, height],
        'minimum_font_size_px': 28, 'full_axis': {'elapsed_s': [0, 30], 'position_error_mm': [0, 85]},
        'late_axis': {'elapsed_s': [20, 30], 'position_error_mm': [0, 10]},
        'samples_per_curve': 301, 'time_source': 'Initial recorded task time, then raw substep endpoints every frame_skip=200',
        'state_source': 'Recorded initial state and raw post-action state; all 300 endpoints exact against raw microstep state',
        'line_interpolation': 'Lines connect recorded action boundaries; the connecting segments are not extra physical samples',
        'numeric_endpoint_checks': 'All final values match summary.e_pos_m within 1e-12 mm',
        'svg_sha256': sha(svg), 'png_sha256': sha(png), 'curves': rows,
        'new_physics_intervals': 0, 'new_model_inference': 0, 'new_learning_updates': 0,
        'scope': 'Main run only, one reused development case and seed. All three timeout. Not a general ranking or proof of handoff.'}
    (OUT/(STEM+'.data.json')).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                                 allow_nan=False)+'\n')
    print(json.dumps({'png': str(png), 'dimensions': [width, height],
        'final_mm': {row['label']: row['final_error_mm'] for row in curves},
        'all_substep_minimum_mm': {row['label']: row['all_recorded_substep_minimum_position_error_mm'] for row in curves},
        'original_source_hashes_preserved': True, 'new_physics_or_learning': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
