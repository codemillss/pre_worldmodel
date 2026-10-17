"""Static fixed-D11 relative-loss diagnostics from immutable saved records.

No prediction, fitting, simulator calls or altered success/validity thresholds.
The original training/evaluation indexes stay unchanged; this plot owns a
separate source-and-output evidence record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from reportlab.graphics import renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Line, Path as Curve, Rect, String
from reportlab.lib.colors import HexColor

ROOT=Path(__file__).resolve().parents[1]
WIDTH,HEIGHT=1600,1470
INK,MUTED,GRID,INVALID='#193144','#516A7B','#DCE5EC','#B74440'
ACTUAL,C0,R1='#263442','#286DA8','#C47A26'
FAMILIES=('p4-data-development-2026101302-f001','p4-data-development-2026101302-f010')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition,message):
    if not condition: raise ValueError(message)


def txt(d,x,y,text,size=18,color=INK,anchor='start'):
    d.add(String(x,y,text,fontName='Helvetica',fontSize=size,fillColor=HexColor(color),textAnchor=anchor))


def line(d,x1,y1,x2,y2,color=GRID,width=1,dash=None):
    d.add(Line(x1,y1,x2,y2,strokeColor=HexColor(color),strokeWidth=width,strokeDashArray=dash))


def dot(d,x,y,color=INK,radius=4):
    d.add(Circle(x,y,radius,fillColor=HexColor(color),strokeColor=None))


def cross(d,x,y,radius=8):
    line(d,x-radius,y-radius,x+radius,y+radius,INVALID,3)
    line(d,x-radius,y+radius,x+radius,y-radius,INVALID,3)


def curve(d,points,color,width=2.8):
    segment=[]
    def draw(values):
        if len(values)<2:return
        path=Curve(strokeColor=HexColor(color),strokeWidth=width,fillColor=None)
        path.moveTo(*values[0])
        for x,y in values[1:]:path.lineTo(x,y)
        d.add(path)
    for point in points:
        if point is None:draw(segment);segment=[]
        else:segment.append(point)
    draw(segment)


def axis(d,left,bottom,width,height,ymin,ymax,yticks,xlabels):
    def x(n):return left+width*n/max(1,len(xlabels)-1)
    def y(v):return bottom+height*(v-ymin)/(ymax-ymin)
    for value in yticks:
        line(d,left,y(value),left+width,y(value),GRID)
        txt(d,left-14,y(value)-6,f'{value:.2f}',17,MUTED,'end')
    line(d,left,bottom,left,bottom+height,MUTED,1.4)
    for n,label in enumerate(xlabels):txt(d,x(n),bottom-30,str(label),16,MUTED,'middle')
    return x,y


def make_figure(dev,comparison,threshold):
    require(set(dev)=={'C0','R1'},'Both new branches required')
    require(comparison['effect_decisions']['contact_Y1_teacher_first5']['status']=='HOLD_INVALID_PRIMARY',
            'This figure requires the observed withheld contact primary')
    contact={arm:dev[arm]['selected_prior_failures']['contact_state1_to2'] for arm in dev}
    require(all(not contact[a]['prediction_valid'] and contact[a]['finite_invalid_diagnostic_only'] for a in dev),
            'Selected contact must remain finite-invalid in both branches')
    actual_gap=contact['C0']['metrics']['actual_gap_m']*1000
    require(contact['R1']['metrics']['actual_gap_m']==contact['C0']['metrics']['actual_gap_m'],'Contact target differs')
    d=Drawing(WIDTH,HEIGHT);d.add(Rect(0,0,WIDTH,HEIGHT,fillColor=HexColor('#FFFFFF'),strokeColor=None))
    txt(d,60,1418,'Relative-position loss: local contact change and stop regression',32)
    txt(d,60,1380,'Fixed D11 data, architecture, 1,000 updates and seed 11. C0: beta = 0; R1: beta = 0.2. Stored predictions only.',20,MUTED)
    for x,color,label in ((70,ACTUAL,'Recorded actual'),(375,C0,'C0 / existing objective'),(790,R1,'R1 / relative-loss objective')):
        line(d,x,1347,x+40,1347,color,3);dot(d,x+20,1347,color,4);txt(d,x+53,1341,label,18)
    cross(d,1378,1347,7);txt(d,1398,1341,'Invalid endpoint',18,INVALID)

    txt(d,60,1284,'A  Selected contact: f001 / Y1 / state 1 -> 2',24)
    left,bottom,width,height=165.,946.,535.,274.
    xmin,xmax=-1.45,.15
    def xx(v):return left+width*(v-xmin)/(xmax-xmin)
    for v in (-1.4,-1.2,-1.,-.8,-.6,-.4,-.2,0):
        line(d,xx(v),bottom-8,xx(v),bottom+height,GRID)
        txt(d,xx(v),bottom-37,f'{v:.1f}',16,MUTED,'middle')
    line(d,xx(0),bottom-8,xx(0),bottom+height,INK,1.5)
    line(d,xx(threshold),bottom-8,xx(threshold),bottom+height,INVALID,2,[7,5])
    values=(('Actual',actual_gap,ACTUAL,False),('C0',contact['C0']['metrics']['predicted_gap_m']*1000,C0,True),
            ('R1',contact['R1']['metrics']['predicted_gap_m']*1000,R1,True))
    for n,(label,value,color,bad) in enumerate(values):
        y=bottom+height*(2.5-n)/3
        txt(d,left-20,y-7,label,20,INK,'end')
        d.add(Rect(min(xx(value),xx(0)),y-17,abs(xx(value)-xx(0)),34,fillColor=HexColor(color),strokeColor=None))
        if bad:cross(d,xx(value),y,9)
        txt(d,730,y-7,f'{value:+.3f}',18,INVALID if bad else INK,'end')
    txt(d,730,1246,'gap (mm)',18,MUTED,'end')
    txt(d,165,871,'Both predictions violate the existing -1 mm penetration gate.',18,INVALID)
    txt(d,165,842,'Relative XY error: 1.215 -> 1.134 mm; finite-invalid diagnosis.',18,MUTED)
    txt(d,165,813,'Y1 first-five-transition primary: HOLD_INVALID_PRIMARY.',18,INVALID)

    stop_values=[]
    for family in FAMILIES:
        for arm in ('C0','R1'):
            case=dev[arm]['selected_prior_failures']['stop_state10_to15'][family]
            stop_values.extend(r['metrics']['predicted_gap_m']*1000 for r in case['free'] if r['finite_state_available'])
            stop_values.extend(r['metrics']['actual_gap_m']*1000 for r in case['free'])
    ymin,ymax=-.1,max(1.2,float(np.ceil((max(stop_values)+.08)*10)/10))
    for family,panel,title_left,plot_left,base in ((FAMILIES[0],'B',840.,915.,946.),(FAMILIES[1],'C',60.,165.,421.)):
        rank=int(family[-3:]);title_y=base+338
        txt(d,title_left,title_y,f'{panel}  Selected stop: f{rank:03} / Y0 / five zero commands',24)
        x,y=axis(d,plot_left,base,605.,274.,ymin,ymax,np.linspace(0,ymax,5),[1,2,3,4,5])
        txt(d,plot_left-16,base+291,'gap (mm)',18,MUTED,'end')
        c0case=dev['C0']['selected_prior_failures']['stop_state10_to15'][family]
        actual=[r['metrics']['actual_gap_m']*1000 for r in c0case['free']]
        for arm in ('C0','R1'):
            case=dev[arm]['selected_prior_failures']['stop_state10_to15'][family]
            require(actual==[r['metrics']['actual_gap_m']*1000 for r in case['free']],'Stop actual endpoints differ')
        for label,values_,color in [('Actual',actual,ACTUAL)]+[(arm,[r['metrics']['predicted_gap_m']*1000
            if r['finite_state_available'] else None for r in dev[arm]['selected_prior_failures']['stop_state10_to15'][family]['free']],color)
            for arm,color in (('C0',C0),('R1',R1))]:
            points=[(x(t),y(v)) if v is not None else None for t,v in enumerate(values_)]
            curve(d,points,color)
            for point in points:
                if point is not None:dot(d,*point,color,4.3)
            if label!='Actual':
                case=dev[label]['selected_prior_failures']['stop_state10_to15'][family]
                for t,row in enumerate(case['free']):
                    if row['finite_state_available'] and not row['prediction_valid']:cross(d,*points[t])
        txt(d,plot_left+302,base-62,'Free stop step (0.1 s per step)',18,MUTED,'middle')
        c0gap=c0case['free'][-1]['metrics']['predicted_gap_m']*1000
        r1case=dev['R1']['selected_prior_failures']['stop_state10_to15'][family]
        r1gap=r1case['free'][-1]['metrics']['predicted_gap_m']*1000
        txt(d,plot_left,base-94,f'K5 free gap: C0 {c0gap:+.3f} -> R1 {r1gap:+.3f} mm.',18)
        tc0=c0case['teacher'][-1]['metrics']['predicted_gap_m']*1000
        tr1=r1case['teacher'][-1]['metrics']['predicted_gap_m']*1000
        txt(d,plot_left,base-123,f'K5 actual-input one-step: {tc0:+.3f} -> {tr1:+.3f} mm.',17,MUTED)
        txt(d,plot_left,base-151,'All stop points valid; positive gap means separation.',17,MUTED)

    txt(d,840,759,'D  Stop endpoint: all 12 paired development families',24)
    families=comparison['family_ids'];phase='stop_S1_free_K5'
    require(len(families)==12 and all(dev[a]['primary_membership'][phase][f]['all_rows_valid'] for a in dev for f in families),
            'Plot D requires all twelve families valid in both arms; never plot a complete-case effect')
    errors={a:[dev[a]['primary'][phase][f]['relative_xy_m']*1000 for f in families] for a in dev}
    upper=max(2.4,float(np.ceil((max(errors['C0']+errors['R1'])+.15)*10)/10))
    x,y=axis(d,915.,421.,605.,274.,0,upper,np.linspace(0,upper,5),[f[-3:] for f in families])
    txt(d,899,712,'relative XY (mm)',18,MUTED,'end')
    for n,f in enumerate(families):
        a,b=errors['C0'][n],errors['R1'][n]
        line(d,x(n)-6,y(a),x(n)+6,y(b),'#A7B6C0',2)
        dot(d,x(n)-6,y(a),C0,4.4);dot(d,x(n)+6,y(b),R1,4.4)
    txt(d,1217,359,'Family rank (Y0/Y1 equal-weight mean)',18,MUTED,'middle')
    row=comparison['contrasts'][phase]['relative_xy_m'];diff=row['mean_difference']*1000
    lo,hi=[v*1000 for v in row['bootstrap_percentile_95']]
    txt(d,915,327,f'R1 - C0: {diff:+.3f} mm; family-bootstrap 95% [{lo:+.3f}, {hi:+.3f}].',18)
    txt(d,915,298,'Relative XY: 8 worse, 4 better. Gap error: 9 worse, 3 better.',18)
    txt(d,915,269,'Both primary errors worsen here; no alignment-success claim.',18,INVALID)

    line(d,60,205,1540,205,GRID,1.5)
    txt(d,60,164,'Contact primary contains all Y1 first five transitions: 27 noncontact, 25 sustained-contact and 8 transition rows.',19,MUTED)
    txt(d,60,130,'The selected stop cases were prior D10 failures; historical D11 (new C0) was already valid for both. Positive gap is not penetration failure.',19,MUTED)
    txt(d,60,96,'Scope: one seed and previously inspected development conditions. Family intervals are descriptive, not learned-seed uncertainty or final-test evidence.',18,MUTED)
    txt(d,60,62,'Invalid endpoints remain invalid. No model inference, learning or physics was performed to create this figure.',18,MUTED)
    return d


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output-directory',type=Path,default=ROOT/'reports/figures')
    args=parser.parse_args();run=args.run.resolve();evaluation=run/'prediction_evaluation';output=args.output_directory.resolve()
    require(json.loads((run/'summary.json').read_text())['status']=='PASS','Training incomplete')
    require(json.loads((evaluation/'summary.json').read_text())['status']=='PASS','Evaluation incomplete')
    names=['C0/development.json','R1/development.json','comparison.json','summary.json']
    eindex=json.loads((evaluation/'artifact_hashes.json').read_text())['sha256']
    tindex=json.loads((run/'training_artifact_hashes.json').read_text())['sha256']
    sources={str((evaluation/name).relative_to(ROOT)):sha(evaluation/name) for name in names}
    for name in names:require(eindex.get(name)==sha(evaluation/name),'Evaluation binding differs: '+name)
    require(tindex.get('manifest.json')==sha(run/'manifest.json'),'Training manifest binding differs')
    for path in (run/'manifest.json',run/'training_artifact_hashes.json',evaluation/'artifact_hashes.json'):
        sources[str(path.relative_to(ROOT))]=sha(path)
    dev={a:json.loads((evaluation/a/'development.json').read_text()) for a in ('C0','R1')}
    comparison=json.loads((evaluation/'comparison.json').read_text())
    config=json.loads((run/'manifest.json').read_text())['physical_config']
    threshold=-config['validity']['predicted_penetration_tolerance_m']*1000
    metadata={'source_sha256':sources,'plot_source_sha256':sha(Path(__file__)),
        'training_artifact_index_sha256':sha(run/'training_artifact_hashes.json'),
        'evaluation_artifact_index_sha256':sha(evaluation/'artifact_hashes.json'),
        'new_model_forwards':0,'learning_updates':0,'new_physics':0,'physical_threshold_changed':False,
        'geometry_unit':'mm','effect_decisions':comparison['effect_decisions'],
        'scope':'Saved-prediction diagnostics. Contact primary withheld; stop worsening is bounded to known development families, one seed.'}
    output.mkdir(parents=True,exist_ok=True)
    svg,png,record=(output/f'p4_relative_loss.{suffix}' for suffix in ('svg','png','figure.json'))
    require(not any(p.exists() for p in (svg,png,record)),'Do not overwrite a relative-loss figure')
    d=make_figure(dev,comparison,threshold);renderSVG.drawToFile(d,str(svg))
    content=svg.read_text();at=content.index('>',content.index('<svg'))+1
    node=ET.Element('metadata');node.text=json.dumps(metadata,sort_keys=True,ensure_ascii=True)
    svg.write_text(content[:at]+ET.tostring(node,encoding='unicode')+content[at:])
    subprocess.run(['/opt/homebrew/bin/rsvg-convert','-w',str(WIDTH),'-h',str(HEIGHT),str(svg),'-o',str(png)],check=True)
    require(all(sha(ROOT/p)==h for p,h in sources.items()),'Plot sources changed during rendering')
    result={'status':'RENDERED_VISUAL_REVIEW_PENDING',**metadata,'svg':str(svg.relative_to(ROOT)),
        'png':str(png.relative_to(ROOT)),'svg_sha256':sha(svg),'png_sha256':sha(png),'dimensions':[WIDTH,HEIGHT]}
    with record.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':result['status'],'png':result['png'],'source_count':len(sources)}))


if __name__=='__main__':main()
