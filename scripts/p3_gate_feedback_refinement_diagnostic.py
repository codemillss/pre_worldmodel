"""Predeclared additional observed-feedback dt/2 vs dt/4 diagnostic.

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
from p3_collect_main import save,compare_paths,next_context
from push_align.main_collection import MainCollectionPolicy
from push_align.closed_loop import ENGINE_GUARDS,model_planning_scope
from p3_geometry_pilot import verify_artifacts


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-run',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    source=args.source_run.resolve();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    prior=verify_artifacts(source);manifest=json.loads((source/'manifest.json').read_text());contract=manifest['contract'];cfg=contract['resolved_config'];half=deepcopy(cfg);half['physics']['timestep_s']*=.5;half['physics']['frame_skip']*=2;quarter=deepcopy(cfg);quarter['physics']['timestep_s']*=.25;quarter['physics']['frame_skip']*=4
    fraction=tomllib.loads(contract['profile_toml'])['numeric']['timestep_difference_fraction_of_task_tolerance'];budgets={'position_m':fraction*cfg['task']['position_tolerance_m'],'yaw_rad':fraction*cfg['task']['yaw_tolerance_rad']}
    save(out/'manifest.json',{'task_id':'P3-02.feedback-refinement-diagnostic','source_run':str(source),'source_artifacts':prior,'predeclared_question':'Do the same originally failed representatives converge within unchanged budgets with independently recomputed observed feedback at dt/2 vs dt/4?','intervention':'halve both already compared timesteps; same frozen initial snapshot, G parameters/noise/action dt; recompute commands independently at observed states','budgets':budgets,'half_config':half,'quarter_config':quarter,'adoption':'NOT_ADOPTED; new configuration would require affected physical/control/data/model validation','source_sha256':digest(Path(__file__))})
    rows=[]
    for mode in ('geometry_feedback','geometry_perturbed'):
        plan=min((p for p in manifest['predeclared_episodes'] if p['mode']==mode),key=lambda p:p['episode_id'])
        settings=manifest['collection_settings'];selected=manifest['selected_G_override'];guards={k:0 for k in ENGINE_GUARDS};actions=range(plan['max_ticks'])
        base,fine=PlanarPushEnv(half),PlanarPushEnv(quarter)
        original=PlanarPushEnv(cfg);original.reset(InitialSpec(**plan['initial_spec']));base.import_dt_comparison_state(original);fine.import_dt_comparison_state(original)
        starts=[];snaps=[];paths=[];gs=[]
        for label,env in (('half',base),('quarter',fine)):
            start=env.observe();starts.append(start);snaps.append(env.snapshot());rs=[];geos=[];decisions=[]
            policy=MainCollectionPolicy(env.config,settings,plan,selected);context=policy.initial_context()
            for tick in actions:
                obs=env.observe()
                with model_planning_scope(guards): decision=policy.propose(obs.state,context,obs.sim_time-start.sim_time,tick)
                decisions.append(decision.diagnostics)
                if not decision.diagnostics['feasible']:break
                r=env.step(decision.action);context=next_context(decision,r);rs.append(r);geo=check_contact_geometry(r.diagnostics['samples'],contract['penetration_budget_m']);geos.append(geo)
                if not r.transition_valid or not geo['pass']:break
            save_trace(out,f'{mode}_{label}',snaps[-1],rs);save(out/f'{mode}_{label}_policy.json',decisions);paths.append(rs);gs.append(geos)
        row=compare_paths(starts,paths,gs,len(actions),budgets,contract['dt_tolerance_s']);row.update(mode=mode,episode_id=plan['episode_id'],future_engine_guard_attempts=guards,identical_initial_integration_state=bool(np.array_equal(snaps[0].integration_state,snaps[1].integration_state)))
        rows.append(row);print(json.dumps(row),flush=True)
    save(out/'summary.json',{'task_id':'P3-02.feedback-refinement-diagnostic','rows':rows,'source_artifacts_unchanged':prior==verify_artifacts(source),'scope':'two independent-feedback representatives at finer timesteps only; full P1 domain and other collection modes unverified; no environment adoption or original gate reversal'})
    save(out/'artifact_hashes.json',{'sha256':{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}})


if __name__=='__main__': main()
