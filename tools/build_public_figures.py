"""Create a readable success figure from the published E2 rows (no model calls)."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
rows = json.loads((ROOT/'artifacts/results/E2/rows.json').read_text())
groups = ['G','R','S1','SK']
counts = []
for group in groups:
    selected = [r for r in rows if r['controller_id'] == group or r['controller_id'].startswith(group+'-')]
    counts.append((sum(r['success'] for r in selected),len(selected)))
assert counts == [(8,12),(0,12),(13,36),(12,36)]
fig, ax = plt.subplots(figsize=(8,5.1))
fig.subplots_adjust(left=.10,right=.98,top=.86,bottom=.19)
fig.patch.set_facecolor('white')
ax.bar(groups,[100*a/b for a,b in counts],color=['#276779','#98A9B0','#8C9DCC','#D18B49'],width=.57)
for index,(success,total) in enumerate(counts):
    ax.text(index,100*success/total+3,f'{success}/{total}',ha='center',fontsize=14,fontweight='bold',color='#172B3A')
ax.set_ylim(0,85)
ax.set_ylabel('Handoff-ready executions (%)',fontsize=12)
ax.set_title('Handoff preparation success under fixed conditions',loc='left',fontsize=13,pad=14)
ax.tick_params(axis='x',labelsize=12)
ax.spines[['top','right']].set_visible(False)
ax.yaxis.grid(True,alpha=.18)
ax.set_axisbelow(True)
fig.text(.10,.025,'12 independent initial-condition families. S1/SK: 3 training seeds per family.\nFixed pose, speed and continuous 0.3 s hold criteria; all failures retained.',fontsize=9.5,color='#4F6470',va='bottom')
destination=ROOT/'assets/figures/handoff_success.png'
destination.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(destination,dpi=180,bbox_inches='tight')
plt.close(fig)
print(destination.relative_to(ROOT))
