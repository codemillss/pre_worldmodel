"""Derived descriptive P3-01 figures; keep all fixed cases/tuning failures."""
from pathlib import Path
import hashlib,json,sys,os
os.environ.setdefault('MPLCONFIGDIR','${LOCAL_TEMP}/push-align-p3-01-analysis/mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
ROOT=Path(sys.argv[1]).resolve(); hashes={}
def read(p):
 b=p.read_bytes();hashes[str(p.relative_to(ROOT))]=hashlib.sha256(b).hexdigest();return b.decode()
def j(p):return json.loads(read(p))
run=ROOT/'runs/p3-01-20261008-01';s=j(run/'summary.json');m=j(run/'manifest.json')
rows={r['episode_id']:[json.loads(x) for x in read(run/'episodes'/r['episode_id']/'steps.jsonl').splitlines()] for r in s['ledger']}
tuning=[j(ROOT/f'runs/p3-01-tuning-20261008-{i:02d}/summary.json') for i in range(1,5)]
colors=['#1479ad','#dd7927','#3a9961','#a451ab'];labels=['translation','yaw +10 deg','yaw -10 deg','forced recovery']
plt.rcParams.update({'font.size':10,'axes.labelsize':10,'axes.titlesize':11,'figure.facecolor':'white','savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
out=ROOT/'reports/figures';out.mkdir(exist_ok=True)
fig,ax=plt.subplots(2,2,figsize=(10.4,8.0),layout='constrained')
fig.suptitle('P3-01 | Geometric feedback feasibility',fontsize=15,fontweight='bold')
for index,entry in enumerate(s['ledger']):
 records=rows[entry['episode_id']];start=entry['task']['initial_sim_time']
 states=np.asarray([records[0]['state_before']]+[r['state_after'] for r in records])
 times=np.asarray([0]+[r['sim_time']-start for r in records]);pos=np.linalg.norm(states[:,:2]-[.08,0],axis=1)*1000
 yaw=np.abs(.5*np.arctan2(np.sin(2*states[:,2]),np.cos(2*states[:,2])))*180/np.pi
 ax[0,0].plot(states[:,0]*1000,states[:,1]*1000,color=colors[index],label=labels[index]);ax[0,0].scatter(*states[0,:2]*1000,color=colors[index],s=22)
 ax[0,0].scatter(*states[-1,:2]*1000,color=colors[index],marker='*',s=100)
 ax[0,1].plot(times,pos,color=colors[index],label=labels[index]);ax[1,0].plot(times,yaw,color=colors[index])
 ax[1,1].bar(index,entry['task']['last_sim_time']-start,color=colors[index]);ax[1,1].text(index,entry['task']['last_sim_time']-start+.2,f"{entry['task']['last_sim_time']-start:.1f}s",ha='center')
ax[0,0].add_patch(Circle((80,0),5,fc='#98cbb744',ec='#3a9961'));ax[0,0].plot(80,0,'k+',ms=10);ax[0,0].set(xlabel='Object X (mm)',ylabel='Object Y (mm)',title='A. All four trajectories | equal XY scale',xlim=(-5,90));ax[0,0].legend(fontsize=8);ax[0,0].set_aspect('equal',adjustable='datalim')
ax[0,1].axhline(5,c='gray',ls='--');ax[0,1].set(xlabel='Elapsed simulation time (s)',ylabel='XY goal error (mm)',title='B. Position error | threshold 5 mm')
ax[1,0].axhline(3,c='gray',ls='--');ax[1,0].set(xlabel='Elapsed simulation time (s)',ylabel='Pi-symmetric yaw error (deg)',title='C. Orientation error | threshold 3 deg')
ax[1,1].set(xticks=range(4),xticklabels=['translation','yaw +10','yaw -10','recovery'],ylabel='Elapsed simulation time (s)',title='D. Ready including measured continuous hold',ylim=(0,15))
for a in ax.flat:a.grid(alpha=.18);a.set_axisbelow(True)
fig.supxlabel('Four frozen development cases after recorded tuning. All ready; this is not a final success rate or a learned-model comparison.',fontsize=9)
fig.savefig(out/'p3_01_geometry_results.png',dpi=180);fig.savefig(out/'p3_01_geometry_results.svg');plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(10.4,5.0),layout='constrained');fig.suptitle('P3-01 | All four sequential tuning attempts',fontsize=15,fontweight='bold')
x=np.arange(4);width=.18
for index in range(4):
 p=[t['ledger'][index]['final_metrics']['e_pos_m']*1000 for t in tuning];y=[np.rad2deg(t['ledger'][index]['final_metrics']['e_yaw_rad']) for t in tuning]
 ax[0].bar(x+(index-1.5)*width,p,width,color=colors[index],label=labels[index]);ax[1].bar(x+(index-1.5)*width,y,width,color=colors[index])
for a in ax:
 a.set(xticks=x,xticklabels=['01: base\n0/4','02: position gain\n2/4','03: yaw creep\n2/4','04: yaw offset\n4/4'],xlabel='Retained cumulative development settings');a.grid(axis='y',alpha=.18);a.set_axisbelow(True)
ax[0].set(ylabel='Final XY error (mm)',title='A. Original terminal states, including failures');ax[0].axhline(5,c='gray',ls='--');ax[0].legend(fontsize=8)
ax[1].set(ylabel='Final pi-symmetric yaw error (deg)',title='B. Pose errors alone do not imply ready');ax[1].axhline(3,c='gray',ls='--')
fig.supxlabel('01/02: timeout failures; 03: yaw cases stopped outside the position window (controller infeasible). No case/tolerance replacement.\nSame four development cases used for tuning; sequential results are not independent validation or isolated global causal effects.',fontsize=9)
fig.savefig(out/'p3_01_tuning_history.png',dpi=180);fig.savefig(out/'p3_01_tuning_history.svg');plt.close(fig)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(ROOT/p)==v for p,v in hashes.items())
audit={'run_id':'p3-01-20261008-01','scope':'descriptive all-four-case trajectories and all-four-attempt tuning; not independent final or model-superiority evidence',
       'official_task_pass':s['pass'],'formal_declared_cases':len(s['ledger']),'formal_ready':sum(r['case_complete'] for r in s['ledger']),
       'formal_valid_intervals':sum(r['valid_transitions'] for r in s['ledger']),'tuning_case_complete_counts':[sum(r['case_complete'] for r in t['ledger']) for t in tuning],
       'input_sha256':hashes,'original_inputs_unchanged':True,'script_sha256':sha(Path(__file__)),'python':sys.version,'numpy':np.__version__,'matplotlib':matplotlib.__version__,
       'figure_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(out.glob('p3_01_*.png'))+sorted(out.glob('p3_01_*.svg'))}}
(ROOT/'reports/p3_01_figure_audit.json').write_text(json.dumps(audit,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
print({'ready':audit['formal_ready'],'tuning_complete':audit['tuning_case_complete_counts'],'figure_files':list(audit['figure_sha256'])})
