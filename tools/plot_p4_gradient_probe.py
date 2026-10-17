"""Three-panel deterministic gradient geometry from immutable saved JSON.

No model/gradient computation, optimizer access or physics. Source bindings
are embedded in the supplemental SVG without changing the official run index.
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
from reportlab.graphics.shapes import Circle,Drawing,Line,Rect,String
from reportlab.lib.colors import HexColor

ROOT=Path(__file__).resolve().parents[1]
WIDTH,HEIGHT=1600,1650
INK,MUTED,GRID='#193144','#516A7B','#DBE5EC'
C0,R1='#286DA8','#C47A26'
SHOTS=('common_initial','C0_u0100','C0_u0500','C0_u1000','R1_u0100','R1_u0500','R1_u1000')
BATCHES=('historical_update1','D11_zero5','D11_transition','D11_noncontact')
LABELS=('Schedule anchor (64)','D11 zero5 (24)','D11 transition (16)','D11 noncontact (8)')
HEADS=(('object_xy','Object XY'),('yaw','Object yaw'),('object_velocity','Object velocity'),
       ('pusher_xy','Pusher XY'),('pusher_velocity','Pusher velocity'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def txt(d,x,y,text,size=18,color=INK,anchor='start'):
    d.add(String(x,y,text,fontName='Helvetica',fontSize=size,fillColor=HexColor(color),textAnchor=anchor))


def line(d,x1,y1,x2,y2,color=GRID,width=1,dash=None):
    d.add(Line(x1,y1,x2,y2,strokeColor=HexColor(color),strokeWidth=width,strokeDashArray=dash))


def dot(d,x,y,color,radius=5):
    d.add(Circle(x,y,radius,fillColor=HexColor(color),strokeColor=None))


def mix(color,value):
    rgb=np.array([int(color[i:i+2],16) for i in (1,3,5)])
    result=np.rint(255*(1-value)+rgb*value).astype(int)
    return '#'+''.join(f'{v:02x}' for v in result)


def heatmap(d,rows,title,top,field,color,*,left=340.,width=1085.,legend_left=1470.):
    txt(d,60,top,title,25)
    bottom,height=top-300.,222.
    colw,rowh=width/7,height/4
    for c,label in enumerate(('Common +0','C0 +100','C0 +500','C0 +1000','R1 +100','R1 +500','R1 +1000')):
        txt(d,left+(c+.5)*colw,top-45,label,18,INK,'middle')
    for r,(batch,label) in enumerate(zip(BATCHES,LABELS)):
        cy=bottom+(3.5-r)*rowh
        txt(d,left-22,cy-7,label,20,INK,'end')
        for c,shot in enumerate(SHOTS):
            value=float(rows[shot,batch]['vector_summary']['base_vs_weighted_aux'][field])
            require(math.isfinite(value) and 0<=value<=1,'Heatmap value outside explicit 0..1 display scale')
            d.add(Rect(left+c*colw,bottom+(3-r)*rowh,colw,rowh,fillColor=HexColor(mix(color,value)),
                       strokeColor=HexColor('#FFFFFF'),strokeWidth=3))
            txt(d,left+(c+.5)*colw,cy-8,f'{value:.3f}',23,INK,'middle')
    line(d,left+colw,bottom,left+colw,bottom+height,'#AABCC7',2)
    line(d,left+4*colw,bottom,left+4*colw,bottom+height,'#AABCC7',2)
    lx=legend_left
    for n in range(80):
        d.add(Rect(lx,bottom+n*height/80,22,height/80+1,fillColor=HexColor(mix(color,n/79)),strokeColor=None))
    for value in (0.,.5,1.):txt(d,lx+33,bottom+value*height-5,f'{value:.1f}',16,MUTED)
    return bottom


def make_figure(rows):
    d=Drawing(WIDTH,HEIGHT);d.add(Rect(0,0,WIDTH,HEIGHT,fillColor=HexColor('#FFFFFF'),strokeColor=None))
    txt(d,60,1597,'Fixed-batch gradient geometry across saved model snapshots',32)
    txt(d,60,1559,'base = g(one-step) + 0.1 g(free); weighted aux = 0.2 [g(relative teacher) + 0.1 g(relative free)]',21,MUTED)
    txt(d,60,1523,'Common +0: P3 epoch100 origin. +100 / +500 / +1000: additional updates. Every cell is one fixed model-batch derivative.',19,MUTED)
    a=heatmap(d,rows,'A  Whole-model cosine: base versus weighted auxiliary',1470,'cosine',C0)
    txt(d,340,a-38,'All 28 sampled whole-model cosines are positive. Angle alignment is not evidence of task improvement.',19,MUTED)
    b=heatmap(d,rows,'B  Whole-model norm ratio: ||weighted aux|| / ||base||',1084,'second_to_first_norm_ratio',R1)
    txt(d,340,b-38,'The auxiliary weight 0.2 is already included. Norm ratios describe scale, not improvement or an Adam step.',19,MUTED)
    txt(d,60,687,'C  Final +1000 snapshots / D11 zero5: final-output-layer norm ratios',25)
    txt(d,340,649,'Only the named output weight rows and biases; hidden parameters are excluded.',19,MUTED)
    dot(d,1320,655,C0,5);txt(d,1334,649,'C0',18)
    dot(d,1425,655,R1,5);txt(d,1439,649,'R1',18)
    left,bottom,width,height=340.,315.,925.,276.
    xmin,xmax=-5.,math.log10(2.)
    def x(v):return left+width*(math.log10(v)-xmin)/(xmax-xmin)
    def y(n):return bottom+height*(4.5-n)/5
    for value in (1e-5,1e-4,1e-3,1e-2,.1,1.):
        line(d,x(value),bottom-12,x(value),bottom+height+8,GRID)
        txt(d,x(value),bottom-40,f'{value:g}',17,MUTED,'middle')
    line(d,x(1),bottom-12,x(1),bottom+height+8,'#7D8D98',1.8,[6,5])
    txt(d,1320,614,'C0 / R1 ratio',18,MUTED)
    for n,(key,label) in enumerate(HEADS):
        cy=y(n);txt(d,left-22,cy-7,label,20,INK,'end')
        values=[]
        for arm in ('C0_u1000','R1_u1000'):
            entry=rows[arm,'D11_zero5']['vector_summary']['output_head_groups']
            require(entry['status']=='APPLICABLE','Registered final output groups absent')
            values.append(float(entry['groups'][key]['base_vs_aux']['second_to_first_norm_ratio']))
        require(all(1e-5<=v<=2 for v in values),'Output-row ratio outside explicit log display range')
        line(d,x(values[0]),cy+5,x(values[1]),cy-5,'#AAB9C3',2)
        dot(d,x(values[0]),cy+5,C0,6);dot(d,x(values[1]),cy-5,R1,6)
        txt(d,1320,cy-7,' / '.join(f'{v:.3g}' for v in values),19)
    txt(d,790,bottom-74,'Weighted auxiliary / base norm (logarithmic axis; 1 = equal norms)',19,MUTED,'middle')
    txt(d,60,196,'R1 yaw-row cosine is negative (-0.834), but its weighted norm ratio is only 0.000105; whole-model cosine stays positive.',18,MUTED)
    txt(d,60,160,'C0 and the common origin were not trained with this auxiliary objective; their auxiliary derivatives are prospective diagnostics.',18,MUTED)
    txt(d,60,124,'Same train-only probes across seven snapshots; overlapping families and repeated snapshots are not 28 independent trials. No error bars.',18,MUTED)
    txt(d,60,88,'Parameter subsets are not physical causes. This figure infers no Adam direction, optimizer update, generalization or alignment success.',18,MUTED)
    txt(d,60,52,'Frozen JSON records only. Plot generation performs zero model forwards, derivative traversals, learning updates and physics steps.',18,MUTED)
    return d


def make_ab(rows):
    d=Drawing(1200,1060);d.add(Rect(0,0,1200,1060,fillColor=HexColor('#FFFFFF'),strokeColor=None))
    txt(d,45,1012,'Whole-model gradient geometry: 7 snapshots x 4 train probes',27)
    txt(d,45,976,'base = g(one-step) + 0.1 g(free); aux = 0.2 [g(relative teacher) + 0.1 g(relative free)]',19,MUTED)
    txt(d,45,941,'Common +0 = P3 epoch100. Other labels are additional updates; same four batches at every snapshot.',18,MUTED)
    heatmap(d,rows,'A  Cosine: base versus weighted auxiliary',893,'cosine',C0,left=280.,width=790.,legend_left=1110.)
    txt(d,280,555,'All 28 whole-model cosines are positive; this is local gradient geometry.',18,MUTED)
    heatmap(d,rows,'B  Norm ratio: ||weighted aux|| / ||base||',493,'second_to_first_norm_ratio',R1,left=280.,width=790.,legend_left=1110.)
    txt(d,280,155,'The auxiliary weight 0.2 is already included; ratio is not improvement.',18,MUTED)
    txt(d,45,108,'C0/common auxiliary derivatives are prospective. Repeated snapshots and train families are not independent trials.',17,MUTED)
    txt(d,45,71,'No error bars, optimizer steps, Adam direction, generalization or task-success inference. Saved records only.',17,MUTED)
    return d


def make_heads(rows):
    d=Drawing(1200,900);d.add(Rect(0,0,1200,900,fillColor=HexColor('#FFFFFF'),strokeColor=None))
    txt(d,45,850,'Final +1000 / D11 zero5: final-output-layer gradient scale',27)
    txt(d,45,812,'Only each named output weight row and bias; hidden parameters excluded. Aux weight 0.2 included.',19,MUTED)
    dot(d,965,776,C0,6);txt(d,981,769,'C0',20);dot(d,1060,776,R1,6);txt(d,1076,769,'R1',20)
    left,bottom,width,height=285.,343.,635.,340.
    def x(v):return left+width*(math.log10(v)+5)/(math.log10(2.)+5)
    def y(n):return bottom+height*(4.5-n)/5
    for value in (1e-5,1e-4,1e-3,1e-2,.1,1.):
        line(d,x(value),bottom-12,x(value),bottom+height+8,GRID)
        txt(d,x(value),bottom-40,f'{value:g}',19,MUTED,'middle')
    line(d,x(1),bottom-12,x(1),bottom+height+8,'#7D8D98',1.8,[6,5])
    txt(d,980,715,'C0 / R1 ratio',20,MUTED)
    for n,(key,label) in enumerate(HEADS):
        cy=y(n);txt(d,left-22,cy-7,label,22,INK,'end')
        values=[float(rows[arm,'D11_zero5']['vector_summary']['output_head_groups']['groups'][key]
                      ['base_vs_aux']['second_to_first_norm_ratio']) for arm in ('C0_u1000','R1_u1000')]
        require(all(1e-5<=v<=2 for v in values),'Head ratio outside explicit log axis')
        line(d,x(values[0]),cy+5,x(values[1]),cy-5,'#AAB9C3',2)
        dot(d,x(values[0]),cy+5,C0,7);dot(d,x(values[1]),cy-5,R1,7)
        txt(d,980,cy-7,' / '.join(f'{v:.3g}' for v in values),20)
    txt(d,600,265,'||weighted auxiliary|| / ||base|| (logarithmic axis; 1 = equal norms)',21,MUTED,'middle')
    txt(d,45,197,'R1 yaw cosine is -0.834, but weighted norm ratio is only 0.000105.',20,MUTED)
    txt(d,45,159,'R1 whole-model cosine on this batch is +0.684; one negative subset does not reverse it.',20,MUTED)
    txt(d,45,112,'These are subsets of parameter-gradient coordinates, not input importance or physical causation.',18,MUTED)
    txt(d,45,74,'Fixed train probes and saved models; no Adam-step, performance or generalization claim. Saved records only.',18,MUTED)
    return d


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output-directory',type=Path,default=ROOT/'reports/figures');args=parser.parse_args()
    run=args.run.resolve();output=args.output_directory.resolve()
    indexpath=run/'artifact_hashes.json';index=json.loads(indexpath.read_text())['sha256']
    names=['aggregate.json','summary.json','manifest.json','registration.json']
    source={}
    def read(name):
        p=run/name;require(index.get(name)==sha(p),'Frozen source index differs: '+name)
        source[str(p.relative_to(ROOT))]=sha(p);return json.loads(p.read_text())
    aggregate,summary,manifest,registration=(read(name) for name in names)
    require(summary['execution_status']=='PASS' and summary['analysis_status']=='DESCRIPTIVE_COMPLETE','Diagnostic incomplete')
    require(summary['counters']['completed_probes']==28 and aggregate['row_count']==28,'Expected exactly 28 saved probes')
    rows={(r['snapshot_id'],r['probe_name']):r for r in aggregate['rows']}
    require(set(rows)=={(s,b) for s in SHOTS for b in BATCHES},'Snapshot and batch inventory differs')
    for identity,row in rows.items():
        saved=read(row['summary_path'])
        require(saved['vector_summary']==row['vector_summary'] and saved['loss_values']==row['loss_values'],
                'Aggregate and original saved probe differ: '+str(identity))
    source[str(indexpath.relative_to(ROOT))]=sha(indexpath)
    metadata={'source_sha256':source,'plot_source_sha256':sha(Path(__file__)),
        'run_artifact_index_sha256':sha(indexpath),'snapshot_order':SHOTS,'batch_order':BATCHES,
        'new_model_forwards':0,'new_derivative_traversals':0,'new_optimizer_updates':0,'new_physics_steps':0,
        'uncertainty':'No error bars; deterministic fixed-model, fixed-batch geometry, not independent trials or seed uncertainty',
        'scope':'Supplemental saved-record visualization, outside the immutable official run artifact index; no causal or performance claim'}
    output.mkdir(parents=True,exist_ok=True)
    for suffix,drawing in (('',make_figure(rows)),('_ab',make_ab(rows)),('_heads',make_heads(rows))):
        svg,png=(output/f'p4_gradient_probe_geometry{suffix}.{kind}' for kind in ('svg','png'))
        require(not svg.exists() and not png.exists(),'Never overwrite a gradient-probe figure')
        renderSVG.drawToFile(drawing,str(svg))
        content=svg.read_text();at=content.index('>',content.index('<svg'))+1
        node=ET.Element('metadata');node.text=json.dumps({**metadata,'layout':suffix or 'full'},sort_keys=True,ensure_ascii=True)
        svg.write_text(content[:at]+ET.tostring(node,encoding='unicode')+content[at:])
        subprocess.run(['/opt/homebrew/bin/rsvg-convert','-w',str(int(drawing.width)),'-h',str(int(drawing.height)),str(svg),'-o',str(png)],check=True)
        require(all(sha(ROOT/p)==h for p,h in source.items()),'Frozen figure source changed during rendering')
        print(json.dumps({'status':'RENDERED_VISUAL_PENDING','svg':str(svg.relative_to(ROOT)),'png':str(png.relative_to(ROOT)),
            'svg_sha256':sha(svg),'png_sha256':sha(png),'source_count':len(source),'dimensions':[drawing.width,drawing.height]}))


if __name__=='__main__':main()
