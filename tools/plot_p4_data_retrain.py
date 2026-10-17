"""Static paired-family evidence plots from frozen evaluations only.

Reuses the workspace's ReportLab SVG renderer and rsvg-convert. No model,
training, simulation, fitting, selection, or extra confidence calculation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from reportlab.graphics import renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
from reportlab.lib.colors import HexColor

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('R', 'D00', 'D10', 'D01', 'D11')
COLORS = {'R': '#718096', 'D00': '#9C6B30', 'D10': '#197A9B', 'D01': '#8955A4', 'D11': '#18805C'}
INK, MUTED, GRID = '#173042', '#526B7C', '#DCE5EC'
PANELS = (('Y_primary', 'object_xy_m', 'High Y: object local XY error'),
          ('Y_primary', 'pusher_xy_m', 'High Y: pusher local XY error'),
          ('S_primary', 'object_xy_m', 'Stop: object free K5 endpoint error'),
          ('S_primary', 'pusher_xy_m', 'Stop: pusher free K5 endpoint error'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def txt(d, x, y, text, size=18, color=INK, anchor='start'):
    d.add(String(x, y, text, fontName='Helvetica', fontSize=size, fillColor=HexColor(color), textAnchor=anchor))


def line(d, x1, y1, x2, y2, color=GRID, width=1):
    d.add(Line(x1, y1, x2, y2, strokeColor=HexColor(color), strokeWidth=width))


def dot(d, x, y, radius=4, color=MUTED):
    d.add(Circle(x, y, radius, fillColor=HexColor(color), strokeColor=None))


def finite(values):
    return [float(v) for v in values if v is not None and math.isfinite(v)]


def nice_max(value):
    if value <= 0: return 1.
    scale = 10**math.floor(math.log10(value))
    return math.ceil(value/scale*2)/2*scale


def number(value):
    return f'{value:.3g}'


def canvas(title, subtitle):
    d = Drawing(1600, 1200)
    d.add(Rect(0, 0, 1600, 1200, fillColor=HexColor('#FFFFFF'), strokeColor=None))
    txt(d, 60, 1145, title, 34)
    txt(d, 60, 1105, subtitle, 20, MUTED)
    return d


def arm_plot(results):
    d = canvas('Fixed-budget data intervention: all 12 development families',
               'Dots: valid family values; missing families stay in denominator 12. Thick line: mean only with 12/12 valid.')
    for index, (phase, metric, title) in enumerate(PANELS):
        col, row = index % 2, index // 2
        left, bottom, width, height = 115+col*770, 665-row*475, 605, 290
        txt(d, left-35, bottom+height+56, title, 23)
        ys = {arm: [results[arm]['primary'][phase][f][metric] for f in results[arm]['families']] for arm in ARMS}
        ymax = nice_max(max([v*1000 for vals in ys.values() for v in finite(vals)] or [0])*1.12)
        for tick in np.linspace(0, ymax, 5):
            y = bottom+height*tick/ymax
            line(d, left, y, left+width, y)
            txt(d, left-12, y-6, number(tick), 17, MUTED, 'end')
        line(d, left, bottom, left, bottom+height, MUTED)
        txt(d, left-22, bottom+height+15, 'mm', 18, MUTED, 'end')
        for a, arm in enumerate(ARMS):
            x = left+width*(a+.5)/5
            vals = ys[arm]
            valid = [(f, v*1000) for f, v in enumerate(vals) if v is not None and math.isfinite(v)]
            for f, value in valid:
                jitter = (f-5.5)*3.1
                dot(d, x+jitter, bottom+height*value/ymax, 4, COLORS[arm])
            if len(valid) == 12:
                mean = float(np.mean([v for _, v in valid]))
                line(d, x-25, bottom+height*mean/ymax, x+25, bottom+height*mean/ymax, COLORS[arm], 4)
            txt(d, x, bottom-28, arm, 20, COLORS[arm], 'middle')
            txt(d, x, bottom-55, f'{len(valid)}/12 valid', 16, MUTED, 'middle')
            if len(valid) != 12:
                txt(d, x, bottom+height+18, 'MEAN HELD', 15, '#B24B42', 'middle')
        if phase == 'Y_primary':
            txt(d, left, bottom-90, 'Y1 only; mean of 5 actual-state-conditioned predictions; S heads counted once.', 16, MUTED)
        else:
            txt(d, left, bottom-90, 'Actual state10 -> five zero actions; equal Y0/Y1 weight inside each family.', 16, MUTED)
    txt(d, 60, 53, 'One model seed; paired development data. Independent y scales; lower error is better. Missing values are never imputed.', 19, MUTED)
    return d


def contrast_plot(comparison):
    d = canvas('Conditional contrasts: paired differences and frozen bootstrap intervals',
               'Dots: available paired families; denominator stays 12. Bar: mean and 95% family-bootstrap interval only with 12/12 valid.')
    for index, (phase, metric, title) in enumerate(PANELS):
        col, row = index % 2, index // 2
        left, bottom, width, height = 155+col*780, 680-row*465, 550, 250
        txt(d, left-105, bottom+height+72, title, 22)
        names = ('D10-D00', 'D11-D01') if phase == 'Y_primary' else ('D01-D00', 'D11-D10')
        names = (*names, 'interaction')
        rows = [comparison['contrasts'][phase][metric][name] for name in names]
        allv = [v*1000 for r in rows for v in finite(r['family_differences']+r['bootstrap_percentile_95'])]
        limit = nice_max(max([abs(v) for v in allv] or [0])*1.12)
        def x(value): return left+width*(value+limit)/(2*limit)
        for tick in np.linspace(-limit, limit, 5):
            line(d, x(tick), bottom, x(tick), bottom+height, MUTED if tick == 0 else GRID, 2 if tick == 0 else 1)
            txt(d, x(tick), bottom-30, number(tick), 17, MUTED, 'middle')
        for n, (name, result) in enumerate(zip(names, rows)):
            y = bottom+height*(2.5-n)/3
            color = '#197A9B' if n < 2 else '#718096'
            txt(d, left-12, y-6, name, 17, INK, 'end')
            for family, value in enumerate(result['family_differences']):
                if value is not None and math.isfinite(value):
                    dot(d, x(value*1000), y+(family-5.5)*2.5, 3.4, color)
            if result['status'] == 'COMPLETE':
                lo, hi = result['bootstrap_percentile_95']; mean = result['mean_difference']
                by = y-25
                line(d, x(lo*1000), by, x(hi*1000), by, color, 3)
                line(d, x(lo*1000), by-6, x(lo*1000), by+6, color, 2)
                line(d, x(hi*1000), by-6, x(hi*1000), by+6, color, 2)
                dot(d, x(mean*1000), by, 5, color)
            else:
                txt(d, left+width, y-33, f"HELD: {result['valid_families']}/12 valid", 16, '#B24B42', 'end')
        txt(d, left+width, bottom-59, 'difference in XY error (mm)', 18, MUTED, 'end')
        factor = 'Y' if phase == 'Y_primary' else 'S'
        decision = comparison['directional_support'][factor]['status']
        txt(d, left-105, bottom-91, f'{factor} factor decision across both endpoints: {decision}', 18, MUTED)
    txt(d, 60, 62, 'Negative differences = lower error. Interaction = D11 - D10 - D01 + D00. Intervals: 10,000 draws, 2.5 / 97.5 percentiles.', 18, MUTED)
    txt(d, 60, 30, 'Directional support requires BOTH conditional contrasts and BOTH XY endpoints to have upper interval < 0. Missing families hold the decision.', 18, MUTED)
    return d


def render(d, output, stem, metadata):
    svg, png = output/(stem+'.svg'), output/(stem+'.png')
    renderSVG.drawToFile(d, str(svg))
    text = svg.read_text(); where = text.index('>', text.index('<svg'))+1
    node = ET.Element('metadata'); node.text = json.dumps(metadata, sort_keys=True)
    svg.write_text(text[:where]+ET.tostring(node, encoding='unicode')+text[where:])
    subprocess.run(['/opt/homebrew/bin/rsvg-convert', '-w', '1600', '-h', '1200', str(svg), '-o', str(png)], check=True)
    return {'svg': str(svg.relative_to(ROOT)), 'png': str(png.relative_to(ROOT)),
            'svg_sha256': sha(svg), 'png_sha256': sha(png)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/figures')
    args = parser.parse_args()
    run = args.evaluation.resolve(); out = args.output.resolve()
    summary = json.loads((run/'summary.json').read_text())
    if summary['status'] != 'PASS': raise ValueError('Evaluation execution is not complete; do not plot a surviving-arm subset')
    hashes = json.loads((run/'artifact_hashes.json').read_text())['sha256']
    source = ['summary.json', 'comparison.json', *[a+'/development.json' for a in ARMS]]
    for name in source:
        if sha(run/name) != hashes[name]: raise ValueError('Plot source differs: '+name)
    results = {a: json.loads((run/a/'development.json').read_text()) for a in ARMS}
    comparison = json.loads((run/'comparison.json').read_text())
    metadata = {'source_sha256': {str((run/p).relative_to(ROOT)): sha(run/p) for p in source},
                'plot_source_sha256': sha(Path(__file__)), 'units': 'mm', 'family_count': 12, 'new_forwards': 0,
                'new_physics': 0, 'new_updates': 0, 'scope': 'Saved development estimands only; no changed evaluation or thresholds'}
    out.mkdir(parents=True, exist_ok=True)
    artifacts = [render(arm_plot(results), out, 'p4_data_retrain_primary', metadata),
                 render(contrast_plot(comparison), out, 'p4_data_retrain_contrasts', metadata)]
    for p, expected in metadata['source_sha256'].items():
        if sha(ROOT/p) != expected: raise ValueError('Plot source changed during rendering')
    result = {'status': 'RENDERED_VISUAL_REVIEW_PENDING', **metadata, 'artifacts': artifacts}
    (out/'p4_data_retrain_figures.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
