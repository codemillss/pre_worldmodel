"""Predeclared additional fixed-command dt/2 vs dt/4 diagnostic.

Uses the same two original failed representatives, not replacements. It does
not adopt a new environment or alter original failed stage acceptance.
"""
import argparse,json,tomllib
from pathlib import Path
from copy import deepcopy
import numpy as np
from push_align import InitialSpec,PlanarPushEnv
from push_align.data import digest
from push_align.contact_geometry import check_contact_geometry
from p2_collect_pilot import save_trace
from p3_collect_main import save,compare_paths
from p3_geometry_pilot import verify_artifacts


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-run',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    source=args.source_run.resolve();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    prior=verify_artifacts(source);manifest=json.loads((source/'manifest.json').read_text());contract=manifest['contract'];cfg=contract['resolved_config'];half=deepcopy(cfg);half['physics']['timestep_s']*=.5;half['physics']['frame_skip']*=2;quarter=deepcopy(cfg);quarter['physics']['timestep_s']*=.25;quarter['physics']['frame_skip']*=4
    fraction=tomllib.loads(contract['profile_toml'])['numeric']['timestep_difference_fraction_of_task_tolerance'];budgets={'position_m':fraction*cfg['task']['position_tolerance_m'],'yaw_rad':fraction*cfg['task']['yaw_tolerance_rad']}
    save(out/'manifest.json',{'task_id':'P3-02.dt-refinement-diagnostic','source_run':str(source),'source_artifacts':prior,'predeclared_question':'Do the same originally failed nominal command vectors converge within unchanged budgets at dt/2 vs dt/4?','intervention':'halve both already compared timesteps, keep frozen starts/commands/action dt; no policy recomputation','budgets':budgets,'half_config':half,'quarter_config':quarter,'adoption':'NOT_ADOPTED; new configuration would require affected physical/control/data/model validation','source_sha256':digest(Path(__file__))})
    rows=[]
    for mode in ('geometry_feedback','geometry_perturbed'):
        plan=min((p for p in manifest['predeclared_episodes'] if p['mode']==mode),key=lambda p:p['episode_id'])
        with np.load(source/'physics_extension'/f'{mode}_base.npz',allow_pickle=False) as z: actions=z['action_requested'].copy()
        base,fine=PlanarPushEnv(half),PlanarPushEnv(quarter)
        original=PlanarPushEnv(cfg);original.reset(InitialSpec(**plan['initial_spec']));base.import_dt_comparison_state(original);fine.import_dt_comparison_state(original)
        starts=[];snaps=[];paths=[];gs=[]
        for label,env in (('half',base),('quarter',fine)):
            starts.append(env.observe());snaps.append(env.snapshot());rs=[];geos=[]
            for action in actions:
                r=env.step(action);rs.append(r);geo=check_contact_geometry(r.diagnostics['samples'],contract['penetration_budget_m']);geos.append(geo)
                if not r.transition_valid or not geo['pass']:break
            save_trace(out,f'{mode}_{label}',snaps[-1],rs);paths.append(rs);gs.append(geos)
        row=compare_paths(starts,paths,gs,len(actions),budgets,contract['dt_tolerance_s']);row.update(mode=mode,episode_id=plan['episode_id'],identical_initial_integration_state=bool(np.array_equal(snaps[0].integration_state,snaps[1].integration_state)))
        rows.append(row);print(json.dumps(row),flush=True)
    save(out/'summary.json',{'task_id':'P3-02.dt-refinement-diagnostic','rows':rows,'source_artifacts_unchanged':prior==verify_artifacts(source),'scope':'fixed-command diagnostic at finer timesteps only; independent feedback and full P1 domain unverified; no environment adoption or original gate reversal'})
    save(out/'artifact_hashes.json',{'sha256':{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}})


if __name__=='__main__': main()
