"""Diagnostic: replay frozen nominal commands at dt and dt/2.

This does not replace the failed independent-feedback acceptance. It changes
only the feedback-vs-replay mechanism to help distinguish competing causes.
"""
from dataclasses import asdict
from pathlib import Path
import argparse,json,sys,tomllib
from copy import deepcopy
import numpy as np
from push_align import InitialSpec, PlanarPushEnv
from push_align.config import ROOT
from push_align.contact_geometry import check_contact_geometry
from push_align.data import digest,make_contract
from p2_collect_pilot import save_trace
from p3_collect_main import save,compare_paths
from p3_geometry_pilot import verify_artifacts


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-run',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    source=args.source_run.resolve();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    before=verify_artifacts(source);manifest=json.loads((source/'manifest.json').read_text());config=manifest['contract']['resolved_config'];contract=manifest['contract'];half=deepcopy(config);half['physics']['timestep_s']*=.5;half['physics']['frame_skip']*=2
    fraction=tomllib.loads(contract['profile_toml'])['numeric']['timestep_difference_fraction_of_task_tolerance'];budgets={'position_m':fraction*config['task']['position_tolerance_m'],'yaw_rad':fraction*config['task']['yaw_tolerance_rad']}
    save(out/'manifest.json',{'task_id':'P3-02.diagnostic','source_run':str(source),'source_artifacts':before,'predeclared_hypotheses':{'H1':'Same fixed commands also exceed original dt budgets: accumulated engine/contact numerical difference is sufficient.','H2':'Same fixed commands remain within budgets while independently recomputed feedback fails: feedback path differences materially amplify initial numerical differences; this does not identify a unique threshold or physical truth.'},'intervention':'hold all nominal command vectors fixed, remove state-dependent command recomputation at both dts; same stabilized start/zero warmstarts','budgets':budgets,'scope':'causal diagnostic for two fixed representatives only; failed original gate remains FAIL','source_sha256':digest(Path(__file__))})
    rows=[]
    for mode in ('geometry_feedback','geometry_perturbed'):
        plan=min((x for x in manifest['predeclared_episodes'] if x['mode']==mode),key=lambda x:x['episode_id'])
        with np.load(source/'physics_extension'/f'{mode}_base.npz',allow_pickle=False) as z:
            actions=z['action_requested'].copy();original_states=z['state'].copy()
        base,fine=PlanarPushEnv(config),PlanarPushEnv(half);base.reset(InitialSpec(**plan['initial_spec']));fine.import_dt_comparison_state(base);base.import_dt_comparison_state(base)
        starts=[];snaps=[];paths=[];gs=[]
        for label,env in (('base',base),('half',fine)):
            starts.append(env.observe());snaps.append(env.snapshot());rs=[];geos=[]
            for action in actions:
                r=env.step(action);rs.append(r);geo=check_contact_geometry(r.diagnostics['samples'],contract['penetration_budget_m']);geos.append(geo)
                if not r.transition_valid or not geo['pass']:break
            save_trace(out,f'{mode}_{label}',snaps[-1],rs);paths.append(rs);gs.append(geos)
        row=compare_paths(starts,paths,gs,len(actions),budgets,contract['dt_tolerance_s']);row.update(mode=mode,episode_id=plan['episode_id'],original_nominal_reproduced_exactly=bool(len(paths[0])==len(original_states) and np.array_equal(np.array([r.observation.state for r in paths[0]]),original_states)),identical_initial_integration_state=bool(np.array_equal(snaps[0].integration_state,snaps[1].integration_state)))
        row['interpretation']='H2 supported within these fixed commands' if row['pass'] and row['original_nominal_reproduced_exactly'] else 'H1 remains possible; inspect trace'
        rows.append(row);print(json.dumps(row),flush=True)
    unchanged=before==verify_artifacts(source)
    save(out/'summary.json',{'task_id':'P3-02.diagnostic','rows':rows,'source_artifacts_unchanged':unchanged,'diagnostic_execution_complete':all(r['complete_both'] and r['original_nominal_reproduced_exactly'] and r['identical_initial_integration_state'] for r in rows) and unchanged,'original_acceptance':'FAIL; main collection remains NOT_RUN','scope':'separate fixed-command replay; not independent-feedback acceptance, tuning success or all-domain physical validity'})
    save(out/'artifact_hashes.json',{'sha256':{str(x.relative_to(out)):digest(x) for x in out.rglob('*') if x.is_file()}})


if __name__=='__main__':main()
