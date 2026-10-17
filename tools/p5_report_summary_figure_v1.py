#!/usr/bin/env python3
"""Additional PDF excerpt from an explicitly supplied, upstream-verified E2 summary.

Uses Pillow only when rendering. No project, model, physics or statistics kernel
is imported. Existing comprehensive figures/clips remain separate and untouched.
Run with bundled document Python; --self-test uses synthetic metadata only.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

SIZE = (1280, 760)
PDF_WIDTH_PT = 450
MIN_FONT_PX = 26
TITLE_FONT_PX = 42
MAX_PAIR_BYTES = 2*1024**2
METHODS = ('G','R','S1','SK')
CONTROLLERS = ('G','R','S1-11','S1-23','S1-47','SK-11','SK-23','SK-47')
PAIRS = (('R','G'),('S1','G'),('SK','G'),('S1','R'),('SK','R'),('SK','S1'))
STRATA = ('translation','rotation','offset')
COLORS = ('#2463a2','#c46d17','#24844b','#7355a5')


def need(okay, message):
    if not okay: raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda text:
        (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+text)))


def number(value, name, lower=-1., upper=1.):
    need(type(value) in (int,float) and math.isfinite(value) and lower<=value<=upper,
         'Finite bounded '+name+' required')
    return float(value)


def validate(summary):
    """Structural excerpt checks; does not regenerate raw outcomes/bootstrap."""
    need(type(summary) is dict and summary.get('status')=='COMPLETE_EXPECTED_E2_DENOMINATORS'
         and (summary.get('episode_count'),summary.get('family_count'),summary.get('controller_count'))==(96,12,8),
         'Complete original96 /12-family /8-controller E2 summary required')
    boot=summary.get('bootstrap',{});families=boot.get('families_in_order');strata=boot.get('strata')
    need(boot.get('replicates')==2000 and boot.get('seed')==7711
         and type(families) is list and len(families)==len(set(families))==12
         and type(strata) is dict and set(strata)==set(STRATA)
         and all(type(v) is list and len(v)==len(set(v))==4 for v in strata.values())
         and set().union(*(set(v) for v in strata.values()))==set(families),
         'Frozen12 independent families /4 per stratum /2000 shared draws required')
    population=summary.get('controllers',{});groups=summary.get('seed_group_summary',{})
    need(set(population)==set(CONTROLLERS) and set(groups)==set(METHODS),'Controller/group population differs')
    def outcome(value,n):
        need(type(value) is dict and value.get('episode_count')==n and value.get('family_count')==12,
             'Pooled episode and independent-family denominators differ')
        count=value.get('success_count')
        need(type(count) is int and 0<=count<=n,'Integer saved success count required')
        rate=number(value.get('success_rate'),'success rate',0.,1.)
        need(rate==count/n,'Stored rate disagrees with saved numerator/denominator')
        return {'count':count,'episodes':n,'rate':rate,'percent':rate*100.}
    individuals={name:outcome(population[name],12) for name in CONTROLLERS}
    pooled={name:outcome(groups[name],12 if name in ('G','R') else 36) for name in METHODS}
    for name in METHODS:
        members=(name,) if name in ('G','R') else tuple(name+'-'+str(seed) for seed in (11,23,47))
        need(pooled[name]['count']==sum(individuals[m]['count'] for m in members),
             'Pooled success count differs from saved controller counts')
    comparisons=summary.get('comparisons')
    all_pairs=PAIRS+tuple(('SK-'+str(seed),'S1-'+str(seed)) for seed in (11,23,47))
    need(type(comparisons) is list and len(comparisons)==9
         and tuple((c.get('left'),c.get('right')) for c in comparisons)==all_pairs,
         'Exact declared contrasts/order required; only first six group contrasts are drawn')
    effects=[]
    for pair,comparison in zip(PAIRS,comparisons[:6]):
        value=comparison.get('metrics',{}).get('success',{})
        need(comparison.get('family_count')==12 and value.get('unit')=='proportion'
             and value.get('eligible_family_denominator')==12
             and value.get('bootstrap_valid_replicates')==2000,
             'Success contrast must retain12 eligible paired families and2000 valid draws')
        mean=number(value.get('difference_mean'),'stored paired success mean')
        interval=value.get('percentile95_interval')
        need(type(interval) is list and len(interval)==2,'Saved95% success interval is missing')
        low,high=(number(x,'stored95% success interval') for x in interval)
        need(low<=high and type(value.get('zero_in_interval')) is bool
             and value['zero_in_interval']==(low<=0<=high),'Stored interval order/sign flag differs')
        need(math.isclose(mean,pooled[pair[0]]['rate']-pooled[pair[1]]['rate'],rel_tol=1e-12,abs_tol=1e-12),
             'Stored contrast direction disagrees with saved balanced pooled rates')
        effects.append({'left':pair[0],'right':pair[1],'mean_percentage_points':mean*100.,
                        'interval95_percentage_points':[low*100.,high*100.]})
    return pooled,effects


def draw_figure(summary):
    from PIL import Image,ImageDraw,ImageFont
    groups,effects=validate(summary)
    image=Image.new('RGB',SIZE,'white');draw=ImageDraw.Draw(image);boxes=[];fonts={}
    font_path=next((p for p in ('/System/Library/Fonts/Supplemental/Arial.ttf',
        '/System/Library/Fonts/Helvetica.ttc','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
        if Path(p).is_file()),None)
    need(font_path is not None,'Scalable TrueType font required; no tiny bitmap fallback')
    def text(x,y,value,size=26,fill='#172b3a',anchor='lt'):
        need(size>=MIN_FONT_PX,'Text below PDF-readable font floor')
        if size not in fonts:fonts[size]=ImageFont.truetype(font_path,size)
        bbox=draw.textbbox((x,y),value,font=fonts[size],anchor=anchor)
        need(0<=bbox[0]<bbox[2]<=SIZE[0] and 0<=bbox[1]<bbox[3]<=SIZE[1],
             'Text leaves image: '+value)
        for previous in boxes:
            a=previous['bbox']
            need(min(a[2],bbox[2])<=max(a[0],bbox[0]) or min(a[3],bbox[3])<=max(a[1],bbox[1]),
                 'Text overlap: '+previous['text']+' / '+value)
        boxes.append({'text':value,'font_px':size,'bbox':list(bbox)})
        draw.text((x,y),value,font=fonts[size],fill=fill,anchor=anchor)
    text(36,20,'E2 success: outcomes and paired uncertainty'+(' | FIXTURE' if summary.get('fixture_only') else ''),42)
    text(36,83,'Saved success rates and counts; recorded summary only',28,fill='#536273')
    text(40,135,'Task success (%)',32)
    text(616,135,'Family-paired contrasts (%p)',32)
    text(765,181,'Dot = mean; line = 95% CI',26,fill='#536273')
    left,right,top,bottom=105,555,230,502
    for tick in (0,25,50,75,100):
        y=bottom-(bottom-top)*tick/100.
        draw.line((left,y,right,y),fill='#d7dee5',width=2)
        text(85,y,str(tick),26,anchor='rm',fill='#536273')
    for index,name in enumerate(METHODS):
        x=150+index*116;value=groups[name];y=bottom-(bottom-top)*value['rate']
        if y<bottom:draw.rectangle((x-34,y,x+34,bottom),fill=COLORS[index])
        else:draw.line((x-34,bottom,x+34,bottom),fill=COLORS[index],width=3)
        text(x,y-34,f"{value['percent']:.1f}%",28,anchor='mt')
        text(x,520,name,30,anchor='mt')
        text(x,557,f"{value['count']}/{value['episodes']}",26,anchor='mt',fill='#536273')
    lo,hi,plot_left,plot_right=-100.,100.,775,1235
    xmap=lambda value:plot_left+(value-lo)/(hi-lo)*(plot_right-plot_left)
    for index,effect in enumerate(effects):
        y=232+index*53;a,b=effect['interval95_percentage_points'];mean=effect['mean_percentage_points']
        # Keep the zero reference out of the numerical labels below each CI.
        draw.line((xmap(0),y-13,xmap(0),y+8),fill='#aab5c1',width=2)
        text(616,y,effect['left']+' - '+effect['right'],28,anchor='lm')
        draw.line((xmap(a),y,xmap(b),y),fill='#2463a2',width=5)
        for point in (a,b):draw.line((xmap(point),y-7,xmap(point),y+7),fill='#2463a2',width=3)
        x=xmap(mean);draw.ellipse((x-6,y-6,x+6,y+6),fill='#c46d17')
        text(1005,y+10,f'{mean:+.1f} [{a:+.1f}, {b:+.1f}]',26,anchor='mt',fill='#536273')
    for tick in (-100,-50,0,50,100):text(xmap(tick),554,f'{tick:+d}' if tick else '0',26,anchor='mt')
    footer=(f"G/R: n={groups['G']['episodes']} each; S1/SK: n={groups['S1']['episodes']} each (3 seeds); shared 12 families.",
        'Independent units: 12 families; 4 per stratum; 2,000 common draws.',
        'Seed means within family; exploratory CIs; no multiplicity correction.',
        'A CI crossing zero does not establish equivalence.')
    for i,line in enumerate(footer):text(36,610+i*34,line,26,fill='#536273')
    return image,{'groups':groups,'contrasts':effects,'text_boxes':boxes,'font_path':font_path,
                  'minimum_font_px':min(row['font_px'] for row in boxes),
                  'minimum_font_at_pdf_450pt':MIN_FONT_PX/SIZE[0]*PDF_WIDTH_PT}


def make(statistics,out):
    statistics=Path(statistics).resolve();out_arg=Path(out)
    need(not out_arg.is_symlink() and not out_arg.with_suffix('.manifest.json').is_symlink(),
         'Output symlink cannot alias prior evidence')
    out=out_arg.resolve();manifest=out.with_suffix('.manifest.json')
    need(statistics.is_file() and out.suffix.lower()=='.png','Existing statistics JSON and new PNG output required')
    need(not out.exists() and not manifest.exists(),'Keep prior PNG/manifest; choose an exclusive new version')
    need(out!=statistics and manifest!=statistics,'Output cannot alias statistics source')
    source_sha=sha(statistics);tool_sha=sha(__file__);summary=read(statistics)
    image,audit=draw_figure(summary);buffer=io.BytesIO();image.save(buffer,format='PNG',optimize=True)
    payload=buffer.getvalue();need(len(payload)<=MAX_PAIR_BYTES,'PNG exceeds2MiB contribution budget')
    need(sha(statistics)==source_sha and sha(__file__)==tool_sha,'Source/tool changed while drawing')
    record={'schema':'p5-additional-PDF-success-summary-figure-v1',
      'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'GENERATED_REQUIRES_DIRECT_VISUAL_REVIEW',
      'source_statistics':{'path':str(statistics),'sha256':source_sha,'status':summary['status']},
      'tool':{'path':str(Path(__file__).resolve()),'sha256':tool_sha},
      'output':{'path':str(out),'sha256':hashlib.sha256(payload).hexdigest(),'size_px':list(SIZE)},
      'purpose':'Additional PDF excerpt; existing comprehensive three PNG figures and three clips remain untouched.',
      'scope':'Uses saved success rates/counts and first six stored family-paired success mean/95% CI contrasts only.',
      'validation_limit':'Structural/count/pairing checks only; caller supplies an upstream-verified summary. No raw/bootstrap regeneration.',
      'fixture_only':summary.get('fixture_only') is True,'new_model_calls':0,'new_physics_steps':0,
      'direct_visual_review':'NOT_ASSESSED','audit':audit}
    metadata=(json.dumps(record,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    need(len(payload)+len(metadata)<=MAX_PAIR_BYTES,'PNG+manifest exceeds2MiB shared-QA contribution')
    out.parent.mkdir(parents=True,exist_ok=True)
    for path,data in ((out,payload),(manifest,metadata)):
        with path.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    return record


def fixture():
    """Synthetic metadata, unrelated to observed project outcomes."""
    counts={'G':7,'R':3,'S1-11':2,'S1-23':4,'S1-47':5,'SK-11':4,'SK-23':6,'SK-47':5}
    population={k:{'episode_count':12,'family_count':12,'success_count':v,'success_rate':v/12} for k,v in counts.items()}
    groups={}
    for name in METHODS:
        members=(name,) if name in ('G','R') else tuple(name+'-'+str(s) for s in (11,23,47))
        n=len(members)*12;c=sum(counts[k] for k in members)
        groups[name]={'episode_count':n,'family_count':12,'success_count':c,'success_rate':c/n}
    comparisons=[]
    for left,right in PAIRS+tuple(('SK-'+str(s),'S1-'+str(s)) for s in (11,23,47)):
        lookup={**population,**groups};mean=lookup[left]['success_rate']-lookup[right]['success_rate']
        a,b=max(-1.,mean-.4),min(1.,mean+.35)
        comparisons.append({'left':left,'right':right,'family_count':12,'metrics':{'success':{
            'difference_mean':mean,'percentile95_interval':[a,b],'unit':'proportion',
            'eligible_family_denominator':12,'bootstrap_valid_replicates':2000,'zero_in_interval':a<=0<=b}}})
    return {'fixture_only':True,'status':'COMPLETE_EXPECTED_E2_DENOMINATORS','episode_count':96,'family_count':12,
        'controller_count':8,'controllers':population,'seed_group_summary':groups,'comparisons':comparisons,
        'bootstrap':{'replicates':2000,'seed':7711,'families_in_order':[f'fixture-{i:02d}' for i in range(12)],
        'strata':{name:[f'fixture-{i:02d}' for i in range(j*4,j*4+4)] for j,name in enumerate(STRATA)}}}


class FixtureTests(unittest.TestCase):
    def test_render_dimensions_font_and_all_nonoverlapping_labels(self):
        image,audit=draw_figure(fixture())
        self.assertEqual(image.size,SIZE);self.assertGreaterEqual(audit['minimum_font_px'],26)
        self.assertGreaterEqual(audit['minimum_font_at_pdf_450pt'],9)
        self.assertEqual(audit['groups']['S1']['episodes'],36)
        self.assertAlmostEqual(audit['groups']['S1']['percent'],11/36*100)
        self.assertTrue(any(row['mean_percentage_points']<0 for row in audit['contrasts']))
        self.assertTrue(any(row['mean_percentage_points']>0 for row in audit['contrasts']))
    def test_missing_value_and_changed_pairing_are_rejected_before_output(self):
        for kind in ('missing','pair'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                summary=fixture()
                if kind=='missing':summary['comparisons'][0]['metrics']['success']['percentile95_interval']=None
                else:summary['comparisons'][0]['left']='G'
                root=Path(directory);source=root/'fixture.json';source.write_text(json.dumps(summary))
                with self.assertRaises(ValueError):make(source,root/'fixture.png')
                self.assertFalse((root/'fixture.png').exists())
    def test_source_hash_pair_is_exclusive_and_stored_values_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'fixture.json';summary=fixture();source.write_text(json.dumps(summary))
            result=make(source,root/'fixture.png');self.assertEqual(result['source_statistics']['sha256'],sha(source))
            value=result['audit']['contrasts'][0]
            self.assertEqual(value['mean_percentage_points'],summary['comparisons'][0]['metrics']['success']['difference_mean']*100)
            self.assertEqual(result['output']['sha256'],sha(root/'fixture.png'))
            self.assertTrue((root/'fixture.manifest.json').is_file())
            with self.assertRaises(ValueError):make(source,root/'fixture.png')
    def test_absolute_counts_and_invalid_denominators_cannot_define_bar_height(self):
        summary=fixture();summary['seed_group_summary']['S1']['success_rate']=11/12
        with self.assertRaises(ValueError):validate(summary)
        summary=fixture();summary['seed_group_summary']['S1']['episode_count']=12
        with self.assertRaises(ValueError):validate(summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--statistics',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FixtureTests))
        return 0 if result.wasSuccessful() else 2
    need(args.statistics is not None and args.out is not None,'Explicit --statistics and --out required')
    result=make(args.statistics,args.out);print(json.dumps({'status':result['status'],'output':result['output']},ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
