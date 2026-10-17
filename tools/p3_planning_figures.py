"""Readable static figures from P3-05 original records; no new experiments."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/p3-05-20261010-01'
OUT=ROOT/'reports/figures'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    summary=json.loads((RUN/'summary.json').read_text())
    if not summary['execution_checks_pass']:raise ValueError('No completed development execution')
    plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.labelsize':11,
                         'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':10.5,
                         'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    OUT.mkdir(exist_ok=True);sources={str((RUN/'summary.json').relative_to(ROOT)):sha(RUN/'summary.json')}
    figures=[]
    labels=['R','MLP-11','MLP-23','MLP-47'];colors=['#d27b32','#346da0','#41836d','#89629c']
    fig,axes=plt.subplots(1,2,figsize=(9.3,4.9),layout='constrained')
    names=['N128-H5','N256-H10'];x=np.arange(len(labels));all_scores=[]
    for i,name in enumerate(names):
        p=RUN/'planning_trials'/name/'summary.json';t=json.loads(p.read_text());sources[str(p.relative_to(ROOT))]=sha(p)
        means=[np.mean([r['terminal_pose_loss'] for r in t['rows'] if r['model_id']==m]) for m in labels]
        all_scores.append(t['score']['mean_terminal_pose_loss'])
        axes[0].bar(x+(i-.5)*.34,means,width=.32,color=['#437baa','#aec2d4'][i],label=name)
    axes[0].set_xticks(x,labels,rotation=18);axes[0].set_ylabel('Mean normalized squared pose error')
    axes[0].set_title('Three development cases at 5 s');axes[0].legend(loc='upper right')
    axes[1].bar(names,all_scores,color=['#437baa','#aec2d4']);axes[1].set_ylabel('Predeclared mean across 12 cells')
    axes[1].set_title('One shared setting, all four models')
    for i,v in enumerate(all_scores):axes[1].text(i,v+.5,f'{v:.3f}',ha='center',fontsize=11)
    axes[1].set_ylim(0,max(all_scores)*1.18)
    for a in axes:a.grid(axis='y',alpha=.18);a.set_axisbelow(True)
    fig.suptitle('Setting selection only: joint N/H change; no final-test superiority',fontsize=12)
    for ext in ('png','svg'):
        p=OUT/f'p3_planning_selection.{ext}';fig.savefig(p,dpi=180);figures.append({'file':str(p.relative_to(ROOT)),'sha256':sha(p)})
    plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(9.3,6.7),layout='constrained')
    for m,color in zip(['G']+labels,['#62666d']+colors):
        p=RUN/'full_resource'/m/'steps.jsonl';records=[json.loads(x) for x in p.read_text().splitlines()];sources[str(p.relative_to(ROOT))]=sha(p)
        time=np.array([r['task']['last_sim_time']-r['task']['initial_sim_time'] for r in records])
        state=np.array([r['state_after'] for r in records]);xy=1000*np.linalg.norm(state[:,:2]-[.08,0.],axis=1)
        yaw=np.rad2deg(np.abs(.5*np.arctan2(np.sin(2*state[:,2]),np.cos(2*state[:,2]))))
        axes[0].plot(time,xy,label=m,color=color,lw=1.6);axes[1].plot(time,yaw,label=m,color=color,lw=1.6)
    axes[0].axhline(5,color='#a53442',ls='--',lw=1,label='5 mm threshold');axes[1].axhline(3,color='#a53442',ls='--',lw=1,label='3 degree threshold')
    axes[0].set_ylabel('Object position error (mm)');axes[1].set_ylabel('Symmetric yaw error (degree)')
    axes[1].set_xlabel('Actual episode simulation time (s)')
    axes[0].set_title('One fixed resource case: trajectories end at their actual termination')
    axes[0].legend(ncol=3,loc='upper right');axes[1].legend(ncol=3,loc='upper right')
    for a in axes:a.set_xlim(left=0);a.set_ylim(bottom=0);a.grid(alpha=.18)
    fig.suptitle('Development only: readiness also requires measured low speeds and 0.3 s hold',fontsize=12)
    for ext in ('png','svg'):
        p=OUT/f'p3_planning_resource_case.{ext}';fig.savefig(p,dpi=180);figures.append({'file':str(p.relative_to(ROOT)),'sha256':sha(p)})
    plt.close(fig)
    output={'scope':'figures from original fixed development data; no interpolation after terminal, no final performance estimate',
            'source_sha256':sources,'code_sha256':sha(Path(__file__)),'figures':figures,
            'minimum_configured_text_pt':10.5,'plot_size_in':[[9.3,4.9],[9.3,6.7]]}
    (ROOT/'reports/p3_05_figures_audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'figures':len(figures),'sources':len(sources)}))


if __name__=='__main__':main()
