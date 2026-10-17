"""Static evidence plot from immutable P4-03 labels; no new experiment."""
from pathlib import Path
import json
import subprocess

import numpy as np
from reportlab.graphics import renderSVG
from reportlab.graphics.shapes import Drawing, Line, Rect, String

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/p4-03-20261012-02'
OUT = ROOT / 'reports/figures'


def main():
    snapshot = RUN / 'snapshots/dev-R-smoke-00-t002-transition'
    with np.load(snapshot / 'models/SK-23/trace.npz') as prediction, np.load(snapshot / 'actual_futures.npz') as actual:
        indices = (9, 15)
        predicted = [float(prediction['total_cost'][i]) for i in indices]
        physical = [float(actual['total_cost'][i, 0]) for i in indices]
    rows = json.loads((RUN / 'comparison_ledger.json').read_text())
    match = next(r for r in rows if r['snapshot_id'] == snapshot.name and r['model'] == 'SK-23')
    assert match['selected_index'] == 15 and match['actual_best_index'] == 9
    assert match['all3_physical_repeats_exact']
    # A visibly labelled truncated axis makes the ordering comparison readable.
    width, height = 900, 405
    drawing = Drawing(width, height)
    ink, muted, blue, orange, pale = '#172B3A', '#617383', '#166A8F', '#D17C30', '#F1F6F8'
    from reportlab.lib.colors import HexColor
    c = HexColor
    drawing.add(Rect(0, 0, width, height, fillColor=c('#FFFFFF'), strokeColor=None))
    drawing.add(String(42, 373, 'Same candidates: predicted ranking reverses in physical execution', fontSize=19, fillColor=c(ink)))
    drawing.add(String(42, 348, 'SK seed 23 | smoke-00, tick 2 | H5 = 0.5 s | lower fixed cost is better', fontSize=12, fillColor=c(muted)))
    x0, y0, plot_w, plot_h, low, high = 92, 90, 610, 205, 1200, 1450
    py = lambda value: y0 + (value - low) / (high - low) * plot_h
    for value in range(low, high + 1, 50):
        y = py(value)
        drawing.add(Line(x0, y, x0 + plot_w, y, strokeColor=c('#D5DFE5'), strokeWidth=.6))
        drawing.add(String(x0 - 10, y - 4, str(value), textAnchor='end', fontSize=11, fillColor=c(muted)))
    drawing.add(Line(x0, y0, x0, y0 + plot_h, strokeColor=c(ink), strokeWidth=1))
    for group, index in enumerate(indices):
        center = x0 + 150 + group * 290
        for offset, value, color in ((-70, predicted[group], blue), (10, physical[group], orange)):
            x = center + offset
            drawing.add(Rect(x, y0, 60, py(value) - y0, fillColor=c(color), strokeColor=None))
            drawing.add(String(x + 30, py(value) + 10, f'{value:.3f}', textAnchor='middle', fontSize=12, fillColor=c(ink)))
        label = 'Candidate 9: actual minimum' if index == 9 else 'Candidate 15: model choice'
        drawing.add(String(center, y0 - 23, label, textAnchor='middle', fontSize=12, fillColor=c(ink)))
    for y, color, label in ((277, blue, 'Predicted cost'), (249, orange, 'Actual cost')):
        drawing.add(Rect(739, y - 3, 13, 13, fillColor=c(color), strokeColor=None))
        drawing.add(String(759, y, label, fontSize=11, fillColor=c(ink)))
    drawing.add(String(42, 35, 'Axis starts at 1200. Actual values are repeat 0; all 3 repeats were exact.', fontSize=11, fillColor=c(muted)))
    drawing.add(String(42, 17, 'Actual minimum is within N16 only. This is an offline cost diagnostic, not an alignment success result.', fontSize=10, fillColor=c(muted)))
    OUT.mkdir(exist_ok=True)
    svg = OUT / 'p4_03_same_candidate_costs.svg'
    png = OUT / 'p4_03_same_candidate_costs.png'
    renderSVG.drawToFile(drawing, str(svg))
    subprocess.run(['/opt/homebrew/bin/rsvg-convert', '-w', '1800', '-h', '810', str(svg), '-o', str(png)], check=True)
    (OUT / 'p4_03_same_candidate_costs.data.json').write_text(json.dumps({
        'run': str(RUN.relative_to(ROOT)), 'snapshot': snapshot.name, 'model': 'SK-23',
        'candidate_indices': list(indices), 'predicted_cost': predicted, 'actual_repeat0_cost': physical,
        'all3_repeats_exact': True, 'selected_index': 15, 'actual_best_index_within_N16': 9,
        'axis_limits': [low, high], 'new_physics_or_learning': 0,
    }, ensure_ascii=False, indent=2) + '\n')
    print(png)


if __name__ == '__main__':
    main()
