"""Static plot of recorded diagnostic arrays; no inference or physics."""
from pathlib import Path
import hashlib
import json
import subprocess

import numpy as np
from reportlab.graphics import renderSVG
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Path as Curve
from reportlab.lib.colors import HexColor

ROOT = Path(__file__).resolve().parents[1]
STEM = 'p4_local_contact_and_residual'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    source_paths = ['runs/p4-local-20261013-01/summary.json',
                    'reports/p4_local_geometry_diagnostics.json']
    bindings = {p: sha(ROOT/p) for p in source_paths}
    summary = json.loads((ROOT/source_paths[0]).read_text())
    geometry = json.loads((ROOT/source_paths[1]).read_text())
    assert summary['status'] == geometry['status'] == 'PASS'
    width, height = 1800, 1260
    d = Drawing(width, height)
    c = HexColor
    ink, muted, grid = '#172B3A', '#5D7080', '#D7E1E7'
    palette = {'MLP-11': '#166A8F', 'S1-11': '#774B9B', 'SK-11': '#CF7927'}
    d.add(Rect(0, 0, width, height, fillColor=c('#FFFFFF'), strokeColor=None))
    d.add(String(70, 1200, 'Contact transition and local prediction error: candidates 11 and 77',
                 fontSize=36, fillColor=c(ink)))
    d.add(String(70, 1150, 'One saved source state | Original H5 commands + 0.5 s zero-command suffix',
                 fontSize=28, fillColor=c(muted)))
    for x, (name, color) in zip((70, 460, 735), palette.items()):
        d.add(Line(x, 1100, x+46, 1100, strokeColor=c(color), strokeWidth=4))
        d.add(String(x+58, 1090, 'Common-11' if name=='MLP-11' else name, fontSize=28, fillColor=c(ink)))
    d.add(Line(1030, 1100, 1076, 1100, strokeColor=c(ink), strokeWidth=5))
    d.add(String(1088, 1090, 'Recorded geometry', fontSize=28, fillColor=c(ink)))
    for col, candidate in enumerate((11,77)):
        x = 120 + col*880
        w = 700
        d.add(String(x+w/2, 1020, f'Candidate {candidate}', textAnchor='middle', fontSize=33, fillColor=c(ink)))
        for row, (y, h, low, high, label) in enumerate((
                (660, 300, -.5, 7.5, 'Pusher-object endpoint gap (mm)'),
                (210, 300, 0., 2.5, 'Object XY error (mm)'))):
            px = lambda t: x+t*w
            py = lambda value: y+(value-low)/(high-low)*h
            d.add(String(x, y+h+28, label, fontSize=27, fillColor=c(muted)))
            d.add(Rect(px(.5), y, w/2, h, fillColor=c('#EEF3F7'), strokeColor=None))
            if row==0 and candidate==11:
                d.add(Rect(px(.2125), y, (.5505-.2125)*w, h,
                           fillColor=c('#E6F3E9'), strokeColor=None))
            yticks = [0,2,4,6] if row==0 else [0,.5,1.,1.5,2.,2.5]
            for tick in yticks:
                d.add(Line(x, py(tick), x+w, py(tick), strokeColor=c(grid), strokeWidth=1))
                d.add(String(x-16, py(tick)-9, f'{tick:g}', textAnchor='end', fontSize=26, fillColor=c(muted)))
            for tick in [0.,.2,.4,.6,.8,1.]:
                d.add(Line(px(tick), y, px(tick), y+h, strokeColor=c(grid), strokeWidth=1))
                d.add(String(px(tick), y-34, f'{tick:g}', textAnchor='middle', fontSize=26, fillColor=c(muted)))
            d.add(Line(px(.5), y, px(.5), y+h, strokeColor=c('#899AAA'), strokeWidth=2))
            d.add(Line(x, y, x, y+h, strokeColor=c(ink), strokeWidth=1.5))
            d.add(Line(x, y, x+w, y, strokeColor=c(ink), strokeWidth=1.5))
            series = []
            if row==0:
                rec=next(r for r in geometry['results'] if r['candidate']==candidate)
                series.append((np.arange(11)/10, rec['actual_states_endpoint_gap_mm'], ink, None))
                for name,color in palette.items():
                    r=next(r for r in geometry['results'] if r['candidate']==candidate and r['model']==name)
                    series.append((np.arange(11)/10,r['free_states_endpoint_gap_mm'],color,None))
            else:
                for name,color in palette.items():
                    r=next(r for r in summary['results'] if r['candidate']==candidate and r['model']==name)
                    series.append((np.arange(1,11)/10,r['free_endpoint_error_norms']['object_xy_mm'],color,[5,7]))
                    series.append((np.arange(1,11)/10,r['conditioned_one_step_error_norms']['object_xy_mm'],color,None))
            for time,values,color,dash in series:
                assert np.isfinite(values).all() and min(values)>=low and max(values)<=high
                curve=Curve(strokeColor=c(color),strokeWidth=4,strokeDashArray=dash,fillColor=None)
                for i,(t,v) in enumerate(zip(time,values)):
                    if i==0:curve.moveTo(px(t),py(v))
                    else:curve.lineTo(px(t),py(v))
                d.add(curve)
            if row==0:
                note = 'Raw force contact: 0.2125-0.5505 s' if candidate==11 else 'No pusher-object contact in raw 1 s record'
                d.add(String(x,y-80,note,fontSize=26,fillColor=c(ink)))
            else:
                d.add(String(x+w/2,y-80,'Elapsed time from source state (s)',textAnchor='middle',fontSize=27,fillColor=c(ink)))
    d.add(Line(70,95,120,95,strokeColor=c(ink),strokeWidth=4))
    d.add(String(135,85,'Solid: local one-step error, reset to actual state',fontSize=27,fillColor=c(ink)))
    d.add(Line(1050,95,1100,95,strokeColor=c(ink),strokeWidth=4,strokeDashArray=[5,7]))
    d.add(String(1115,85,'Dotted: free endpoint error',fontSize=27,fillColor=c(ink)))
    d.add(String(70,35,'Gray: zero commands. Endpoint gaps do not measure predicted force. Lines connect stored boundaries only.',fontSize=26,fillColor=c(muted)))
    out=ROOT/'reports/figures'
    svg,png=out/(STEM+'.svg'),out/(STEM+'.png')
    renderSVG.drawToFile(d,str(svg))
    subprocess.run(['/opt/homebrew/bin/rsvg-convert','-w',str(width),'-h',str(height),str(svg),'-o',str(png)],check=True)
    assert all(sha(ROOT/p)==h for p,h in bindings.items())
    data={'source_sha256':bindings,'plot_source_sha256':sha(Path(__file__)),
          'png_sha256':sha(png),'svg_sha256':sha(svg),'dimensions_px':[width,height],
          'minimum_font_size_px':26,'axes_enclose_all_values':True,
          'scope':'Offline local reset and free endpoint errors are different diagnostics; curves are not alternative deployed policies. Contact stripe is raw recorded force, gap curves are boundary geometry.',
          'new_physics':0,'new_inference':0,'new_learning':0}
    (out/(STEM+'.data.json')).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'png':str(png),'sha256':sha(png)}))

if __name__=='__main__':main()
