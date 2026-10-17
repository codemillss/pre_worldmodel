"""One static figure from frozen post-hoc geometry/stop diagnostics only.

ReportLab -> SVG -> rsvg-convert PNG, with data/code hashes in SVG metadata.
No model, future physics, training, hidden threshold, or repaired prediction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from reportlab.graphics import renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Line, Path as Curve, Rect, String
from reportlab.lib.colors import HexColor

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('R', 'D00', 'D10', 'D01', 'D11')
BLOCKS = ('object_xy', 'pusher_xy', 'object_yaw')
PALETTE = {'object_xy': '#C78035', 'pusher_xy': '#177B9C', 'object_yaw': '#885BA4'}
NAMES = {'object_xy': 'Object XY', 'pusher_xy': 'Pusher XY', 'object_yaw': 'Object yaw'}
INK, MUTED, GRID, INVALID = '#193144', '#516A7B', '#DBE5EC', '#BA443F'
ACTUAL, FREE, TEACHER = '#2D3748', '#C07828', '#197A9B'
WIDTH, HEIGHT = 1600, 1350


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition: raise ValueError(message)


def txt(d, x, y, text, size=18, color=INK, anchor='start'):
    d.add(String(x, y, text, fontName='Helvetica', fontSize=size,
                 fillColor=HexColor(color), textAnchor=anchor))


def line(d, x1, y1, x2, y2, color=GRID, width=1, dash=None):
    d.add(Line(x1, y1, x2, y2, strokeColor=HexColor(color), strokeWidth=width, strokeDashArray=dash))


def dot(d, x, y, color=INK, radius=4, hollow=False):
    d.add(Circle(x, y, radius, fillColor=HexColor('#FFFFFF' if hollow else color),
                 strokeColor=HexColor(color), strokeWidth=2 if hollow else 0))


def cross(d, x, y, color=INVALID, radius=8):
    line(d, x-radius, y-radius, x+radius, y+radius, color, 3)
    line(d, x-radius, y+radius, x+radius, y-radius, color, 3)


def nice_bound(value, upper=True):
    if value == 0: return 0.
    scale = 10**math.floor(math.log10(abs(value)))
    return (math.ceil if upper else math.floor)(value/scale*2)/2*scale


def number(value): return f'{value:.3g}'


def curve(d, points, color, dash=None, width=2.5):
    segment = []
    def draw(values):
        if len(values) < 2: return
        p = Curve(strokeColor=HexColor(color), strokeWidth=width, strokeDashArray=dash, fillColor=None)
        p.moveTo(*values[0])
        for x, y in values[1:]: p.lineTo(x, y)
        d.add(p)
    for point in points:
        if point is None:
            draw(segment); segment = []
        else: segment.append(point)
    draw(segment)


def make_figure(contact, stop):
    require(contact['status'] == stop['status'] == 'DESCRIPTIVE_COMPLETE', 'Diagnostic data incomplete')
    rows = {row['arm']: row for row in contact['rows']}
    require(set(rows) == set(ARMS), 'Five stored contact models required')
    actual_gaps = [rows[a]['actual_gap_m']*1000 for a in ARMS]
    require(max(actual_gaps)-min(actual_gaps) < 1e-9, 'Contact models have different actual target geometry')
    actual_gap = actual_gaps[0]
    threshold = -float(contact['predicted_penetration_tolerance_m'])*1000
    threshold_error = threshold-actual_gap
    cases = sorted(stop['cases'], key=lambda c: c['identity']['family_rank'])
    require([c['identity']['family_rank'] for c in cases] == [1, 10], 'Two intended stop cases required')
    require(all(c['original_penetration_tolerance_m_unchanged'] == contact['predicted_penetration_tolerance_m']
                for c in cases), 'Stored diagnostic thresholds differ')
    d = Drawing(WIDTH, HEIGHT)
    d.add(Rect(0, 0, WIDTH, HEIGHT, fillColor=HexColor('#FFFFFF'), strokeColor=None))
    txt(d, 60, 1290, 'Stored prediction failures: geometry and stop-error propagation', 34)
    txt(d, 60, 1247, 'Post-hoc selected development cases. Geometry allocations are descriptive; no new forecasts, physics or training.', 20, MUTED)
    txt(d, 60, 1190, 'A  Contact transition: f001 / Y1 / core step 2', 25)
    for i, name in enumerate(BLOCKS):
        x = 795+i*235
        d.add(Rect(x, 1185, 20, 17, fillColor=HexColor(PALETTE[name]), strokeColor=None))
        txt(d, x+29, 1187, NAMES[name], 18)
    left, bottom, width, height = 165., 823., 1050., 295.
    contributions = {a: {b: rows[a]['shapley_gap_error_m'][b]*1000 for b in BLOCKS} for a in ARMS}
    lows = [sum(min(v, 0) for v in contributions[a].values()) for a in ARMS]
    highs = [sum(max(v, 0) for v in contributions[a].values()) for a in ARMS]
    lo = nice_bound(min(*lows, threshold_error, 0)*1.08, False)
    hi = nice_bound(max(*highs, 0)*1.08, True)
    if hi-lo <= 0: lo, hi = -1., 1.
    margin = (hi-lo)*.06
    lo -= margin; hi += margin
    def xx(value): return left+width*(value-lo)/(hi-lo)
    for tick in np.linspace(lo, hi, 7):
        line(d, xx(tick), bottom-5, xx(tick), bottom+height, GRID)
        txt(d, xx(tick), bottom-32, number(tick), 17, MUTED, 'middle')
    line(d, xx(0), bottom-5, xx(0), bottom+height+5, INK, 1.8)
    line(d, xx(threshold_error), bottom-5, xx(threshold_error), bottom+height+5, INVALID, 2, [7, 5])
    txt(d, 1260, 1143, 'Predicted gap', 17, MUTED)
    txt(d, 1455, 1143, 'Gap error', 17, MUTED, 'end')
    for i, arm in enumerate(ARMS):
        y = bottom+height*(4.5-i)/5
        txt(d, left-22, y-7, arm, 21, INK, 'end')
        negative, positive = 0., 0.
        for block in BLOCKS:
            value = contributions[arm][block]
            start = positive if value >= 0 else negative
            end = start+value
            if value >= 0: positive = end
            else: negative = end
            d.add(Rect(min(xx(start), xx(end)), y-13, max(abs(xx(end)-xx(start)), .4), 26,
                       fillColor=HexColor(PALETTE[block]), strokeColor=None))
        error = rows[arm]['gap_error_m']*1000
        require(abs(sum(contributions[arm].values())-error) < 1e-9, 'Bar contributions do not sum to stored error')
        dot(d, xx(error), y, INK, 6, True)
        pred_gap = rows[arm]['predicted_gap_m']*1000
        txt(d, 1260, y-7, f'{pred_gap:+.3f} mm', 18, INVALID)
        txt(d, 1455, y-7, f'{error:+.3f} mm', 18, INK, 'end')
    txt(d, 60, 742, 'Bars: three-block Shapley gap-error allocation; interactions retained. Hollow dot: total gap error.', 19, MUTED)
    txt(d, 60, 709, f'Error-axis references: 0 = actual gap {actual_gap:+.3f} mm; red dashed = existing {threshold:+.3f} mm gap boundary ({threshold_error:+.3f} mm error).', 19, MUTED)
    txt(d, 60, 676, 'Hybrids are geometry calculations, not feasible states or causal interventions. Every shown model endpoint remains invalid.', 19, MUTED)
    txt(d, 60, 623, 'B  D10 / Y0 / five zero commands: stored endpoints', 25)
    legend_y = 581
    for x, color, label, dash in ((510, ACTUAL, 'Actual', None), (745, FREE, 'Free rollout', None),
                                  (1030, TEACHER, 'Actual-state-conditioned', [5, 4])):
        line(d, x, legend_y, x+42, legend_y, color, 2.5, dash)
        dot(d, x+21, legend_y, color, 4)
        txt(d, x+53, legend_y-6, label, 18)
    cross(d, 1445, legend_y, INVALID, 6)
    txt(d, 1460, legend_y-6, 'Invalid', 17, INVALID)
    values = [threshold]
    series = []
    for case in cases:
        actual = [s['geometry']['actual_gap_m']*1000 if s['geometry']['available'] else None for s in case['states']]
        free = [s['geometry']['predicted_gap_m']*1000 if s['geometry']['available'] else None for s in case['states']]
        # There is no teacher forecast at step0; do not synthesize a point.
        teacher = [None]+[r['teacher_endpoint_geometry']['predicted_gap_m']*1000
                         if r['teacher_endpoint_geometry']['available'] else None for r in case['steps']]
        values.extend(v for data in (actual, free, teacher) for v in data if v is not None and math.isfinite(v))
        series.append((actual, free, teacher))
    ymin = nice_bound(min(values)-.10, False)
    ymax = nice_bound(max(values)+.10, True)
    for n, (case, data) in enumerate(zip(cases, series)):
        left, bottom, width, height = 115+n*770, 213, 605, 295
        def x(t): return left+width*t/5
        def y(v): return bottom+height*(v-ymin)/(ymax-ymin)
        rank = case['identity']['family_rank']
        txt(d, left, 539, f'f{rank:03d}: actual core state10 -> state15', 21)
        for tick in np.linspace(ymin, ymax, 5):
            line(d, left, y(tick), left+width, y(tick), GRID)
            txt(d, left-12, y(tick)-6, number(tick), 17, MUTED, 'end')
        line(d, left, bottom, left, bottom+height, MUTED)
        line(d, left, y(threshold), left+width, y(threshold), INVALID, 1.7, [7, 5])
        txt(d, left+width-6, y(threshold)+10, 'existing -1 mm boundary', 15, INVALID, 'end')
        for step in range(6): txt(d, x(step), bottom-27, str(step), 17, MUTED, 'middle')
        txt(d, left-18, bottom+height+8, 'gap (mm)', 17, MUTED, 'end')
        for values_, color, dash in zip(data, (ACTUAL, FREE, TEACHER), (None, None, [5, 4])):
            points = [(x(t), y(v)) if v is not None and math.isfinite(v) else None for t, v in enumerate(values_)]
            curve(d, points, color, dash)
            for p in points:
                if p is not None: dot(d, *p, color, 3.8)
        for state in case['states']:
            t = state['step']
            if state['available'] and not state['prediction_valid']:
                value = data[1][t]; cross(d, x(t), y(value), INVALID, 8)
                txt(d, x(t)-14, y(value)-22, f'{value:+.3f} mm (invalid)', 16, INVALID, 'end')
        for t, row in enumerate(case['steps'], 1):
            if row['teacher_endpoint_geometry']['available'] and not row['teacher_prediction_valid']:
                cross(d, x(t), y(data[2][t]), INVALID, 8)
        txt(d, left+width/2, bottom-59, 'Stop step (0.1 s per step)', 18, MUTED, 'middle')
    txt(d, 60, 109, 'Teacher dots use the actual input at each step; the dashed connection is a visual guide, not a recursive trajectory.', 19, MUTED)
    txt(d, 60, 76, 'Finite invalid endpoints are retained and marked X. Missing values stay missing. Shared gap scale across both stop cases.', 19, MUTED)
    txt(d, 60, 43, 'Scope: one contact transition and two previously observed stop failures. This figure does not repair the primary comparison or prove a unique cause.', 18, MUTED)
    return d


def render(output, contact, stop, metadata):
    svg, png = output/'p4_error_mechanism.svg', output/'p4_error_mechanism.png'
    require(not svg.exists() and not png.exists(), 'Do not overwrite an existing mechanism figure')
    d = make_figure(contact, stop)
    output.mkdir(parents=True, exist_ok=True)
    renderSVG.drawToFile(d, str(svg))
    text = svg.read_text(); at = text.index('>', text.index('<svg'))+1
    node = ET.Element('metadata'); node.text = json.dumps(metadata, ensure_ascii=True, sort_keys=True)
    svg.write_text(text[:at]+ET.tostring(node, encoding='unicode')+text[at:])
    subprocess.run(['/opt/homebrew/bin/rsvg-convert', '-w', str(WIDTH), '-h', str(HEIGHT), str(svg), '-o', str(png)], check=True)
    return {'svg': str(svg.relative_to(ROOT)), 'png': str(png.relative_to(ROOT)),
            'svg_sha256': sha(svg), 'png_sha256': sha(png), 'dimensions': [WIDTH, HEIGHT]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, default=ROOT/'reports/figures')
    parser.add_argument('--source-index', type=Path)
    args = parser.parse_args()
    run = args.run.resolve(); output = args.output_directory.resolve()
    names = ('contact_geometry.json', 'stop_propagation.json')
    source = {str((run/name).relative_to(ROOT)): sha(run/name) for name in names}
    if args.source_index is not None:
        index = json.loads(args.source_index.read_text())['sha256']
        for name in names:
            require(index.get(name) == sha(run/name), 'Mechanism result index differs: '+name)
        source[str(args.source_index.resolve().relative_to(ROOT))] = sha(args.source_index)
    contact, stop = (json.loads((run/name).read_text()) for name in names)
    for diagnostic in (contact, stop):
        for path, expected in diagnostic['source_sha256'].items():
            require(sha(ROOT/path) == expected, 'Frozen diagnostic source changed: '+path)
    metadata = {'source_sha256': source, 'plot_source_sha256': sha(Path(__file__)),
                'new_model_forwards': 0, 'new_physics': 0, 'learning_updates': 0,
                'physical_threshold_changed': False, 'geometry_unit': 'mm',
                'scope': 'Post-hoc selected cases and algebraic geometry allocation; no unique causal or task-success claim.'}
    artifact = render(output, contact, stop, metadata)
    require(all(sha(ROOT/p) == s for p, s in source.items()), 'Plot source changed during rendering')
    report = {'status': 'RENDERED_VISUAL_REVIEW_PENDING', **metadata, **artifact}
    path = output/'p4_error_mechanism.figure.json'
    with path.open('x') as stream: json.dump(report, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(report))


if __name__ == '__main__': main()
