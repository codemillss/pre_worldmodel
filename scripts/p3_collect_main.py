"""P3-02: frozen main-development collection, physics extension and preparation.

A collection completion is not handoff success. Feedback acts only on observed
state; raw contact is audited afterwards. Failed representatives are retained
and cannot be replaced to pass the unchanged numerical gate.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime
import gc
import json
from pathlib import Path
import platform
import re
import resource
import shutil
import subprocess
import sys
from time import perf_counter
import tomllib
import traceback
import numpy as np
from push_align import InitialSpec, PlanarPushEnv
from push_align.closed_loop import ENGINE_GUARDS, model_planning_scope
from push_align.config import ROOT
from push_align.contact_geometry import check_contact_geometry
from push_align.coverage import REGIMES, episode_coverage
from push_align.data import DatasetWriter, EpisodeRecorder, _encode_nonfinite, contract_hash, digest, iter_dataset, make_contract
from push_align.environment import wrap_angle
from push_align.learning_data import load_learning_data, save_learning_data
from push_align.main_collection import MODES, FEEDBACK_MODES, MainCollectionPolicy, build_main_collection_plan
from push_align.normalization import learning_schema
from push_align.preparation import prepare_dataset
from p2_collect_pilot import save_trace
from p2_episode_contract_check import source_bundle
from p2_prepare_learning_data import fit, mutation_checks
from p3_geometry_pilot import verify_artifacts


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as file:
        json.dump(_encode_nonfinite(value), file, ensure_ascii=False, allow_nan=False, indent=2)
        file.write('\n')


def validate_settings(settings, contract):
    c=settings['coverage']; cfg=contract['resolved_config']; p=tomllib.loads(contract['profile_toml'])
    if c['horizons']!=[1,5,10] or c['required_validation_horizon']!=5 or c['required_validation_regimes']!=list(REGIMES) or c['minimum_distinct_validation_families_per_regime']!=5:
        raise ValueError('Registered all-regime K5 / five-family gate required')
    if c['require_signal_per_split'] is not True or c['require_qualified_recontact_per_split'] is not True:
        raise ValueError('Registered signal/recontact checks cannot be disabled')
    if c['active_contact_rule']!='pusher_object normal_force strictly greater than environment threshold; start plus every substep' or c['window_regime']!='first_transition_only':
        raise ValueError('Actual raw-contact regime definition changed')
    fraction=p['numeric']['signal_min_fraction_of_task_tolerance']
    if not np.isclose(c['position_signal_min_m'],fraction*cfg['task']['position_tolerance_m'],rtol=0,atol=1e-15) or not np.isclose(c['yaw_signal_min_rad'],fraction*cfg['task']['yaw_tolerance_rad'],rtol=0,atol=1e-15):
        raise ValueError('Task-based signal budgets changed')
    ext=settings['physics_extension']
    if ext['physics_dt_scale']!=.5 or ext['representative_rule']!='lowest_episode_id_per_mode; frozen before physics' or ext['require_full_equal_execution_lengths'] is not True or ext['budgets']!='unchanged acceptance-v1 dt position/yaw budgets':
        raise ValueError('Registered physical extension rule changed')
    if ext['comparison']!='independent feedback at observed states with identical predeclared noise; identical stabilized snapshot and zero warmstarts; all shared action boundaries' or ext['on_failure']!='preserve evidence; block main collection; no replacement or threshold relaxation':
        raise ValueError('Physical gate execution/failure policy changed')
    if settings['limits']['physical_command_limit_m_s']!=cfg['pusher']['command_speed_norm_max_m_s'] or settings['limits']['termination']!='fixed 16 or 128 ticks; pose goal means zero action without declaring handoff success; preserve controller-infeasible and invalid prefixes':
        raise ValueError('Physical limit or episode termination contract changed')
    if settings['limits']['final_evaluation_allowed'] is not False or settings['limits']['alignment_success_evaluation'] is not False or settings['resources']['maximum_planned_transitions']!=4800:
        raise ValueError('Collection purpose/cap changed')


def next_context(decision, result):
    actual=result.applied_action.copy(); actual.flags.writeable=False
    return replace(decision.context, previous_applied_action=actual)


def compare_paths(start_states, paths, geometries, max_ticks, budgets, tolerance):
    initial_match=bool(np.array_equal(start_states[0].state,start_states[1].state) and start_states[0].sim_time==start_states[1].sim_time)
    complete=all(len(rs)==max_ticks and all(r.transition_valid for r in rs) for rs in paths)
    length=min(len(rs) for rs in paths)
    times_match=np.allclose([r.observation.sim_time for r in paths[0][:length]], [r.observation.sim_time for r in paths[1][:length]],rtol=0,atol=tolerance)
    states=[np.vstack([s.state,*[r.observation.state for r in rs[:length]]]) for s,rs in zip(start_states,paths)]
    xy=float(np.max(np.linalg.norm(states[0][:,:2]-states[1][:,:2],axis=1)))
    yaw=float(np.max(np.abs(wrap_angle(states[0][:,2]-states[1][:,2]))))
    geometry_pass=len(geometries)==2 and all(len(gs)==len(rs) for gs,rs in zip(geometries,paths)) and all(g['pass'] for gs in geometries for g in gs)
    return {'complete_both':complete,'executed_lengths':[len(rs) for rs in paths], 'shared_intervals':length,
            'identical_initial_observation_and_time':initial_match,'same_shared_action_boundaries':bool(times_match),
            'max_position_difference_m':xy,'max_yaw_difference_rad':yaw,'geometry_pass':geometry_pass,
            'budgets':budgets,'pass':bool(complete and initial_match and times_match and geometry_pass and xy<=budgets['position_m'] and yaw<=budgets['yaw_rad'])}


def physics_gate(out, plans, settings, config, contract, selected):
    folder=out/'physics_extension'; folder.mkdir()
    half=deepcopy(config); half['physics']['timestep_s']*=.5; half['physics']['frame_skip']*=2
    base,fine=PlanarPushEnv(config),PlanarPushEnv(half)
    fraction=tomllib.loads(contract['profile_toml'])['numeric']['timestep_difference_fraction_of_task_tolerance']
    budgets={'position_m':fraction*config['task']['position_tolerance_m'],'yaw_rad':fraction*config['task']['yaw_tolerance_rad']}
    rows=[]; guards={k:0 for k in ENGINE_GUARDS}
    for mode in MODES:
        plan=min((p for p in plans if p['mode']==mode),key=lambda p:p['episode_id'])
        base.reset(InitialSpec(**plan['initial_spec']))
        fine.import_dt_comparison_state(base); base.import_dt_comparison_state(base)
        starts=[]; snaps=[]; paths=[]; geometries=[]; policy_rows=[]; reasons=[]
        for label,env in (('base',base),('half',fine)):
            start=env.observe(); snap=env.snapshot(); starts.append(start); snaps.append(snap)
            policy=MainCollectionPolicy(env.config,settings,plan,selected); context=policy.initial_context()
            rs=[]; gs=[]; decisions=[]; reason='collection_complete'
            for tick in range(plan['max_ticks']):
                obs=env.observe()
                with model_planning_scope(guards):
                    decision=policy.propose(obs.state,context,obs.sim_time-start.sim_time,tick)
                decisions.append(decision.diagnostics)
                if not decision.diagnostics['feasible']:
                    reason='controller_infeasible'; break
                result=env.step(decision.action); rs.append(result)
                geo=check_contact_geometry(result.diagnostics['samples'],contract['penetration_budget_m']); gs.append(geo)
                context=next_context(decision,result)
                if not result.transition_valid or not geo['pass']:
                    reason='physical_or_geometry_failure'; break
            save_trace(folder,f'{mode}_{label}',snap,rs)
            save(folder/f'{mode}_{label}_policy.json',{'decisions':decisions,'terminal_reason':reason})
            paths.append(rs); geometries.append(gs); policy_rows.append(decisions); reasons.append(reason)
        row=compare_paths(starts,paths,geometries,plan['max_ticks'],budgets,contract['dt_tolerance_s'])
        same_snap=bool(np.array_equal(snaps[0].integration_state,snaps[1].integration_state))
        row.update(mode=mode,episode_id=plan['episode_id'],terminal_reasons=reasons,
                   identical_initial_integration_state=same_snap,
                   requested_actions_identical=bool(len(paths[0])==len(paths[1]) and all(np.array_equal(a.requested_action,b.requested_action) for a,b in zip(*paths))))
        row['pass'] &= same_snap
        rows.append(row)
        print(json.dumps({'progress':'physics_extension','mode':mode,'pass':row['pass'],'position_mm':row['max_position_difference_m']*1000,'yaw_deg':float(np.rad2deg(row['max_yaw_difference_rad'])),'lengths':row['executed_lengths']}),flush=True)
        del paths, geometries, policy_rows, rs; gc.collect()
    result={'pass':all(r['pass'] for r in rows) and sum(guards.values())==0,'cases':rows,'half_config':half,'future_engine_guard_attempts':guards,
            'scope':'six frozen representatives; independently recomputed feedback at each dt, same predeclared noise; not all trajectories or full continuous-domain validation'}
    save(folder/'summary.json',result)
    return result


def aggregate(rows,settings):
    splits={}
    for split in ('train','validation'):
        part=[r for r in rows if r['split']==split]
        windows={str(h):{'total':sum(r['windows'][str(h)]['total'] for r in part),
                         'by_first_regime':{g:sum(r['windows'][str(h)]['by_first_regime'][g] for r in part) for g in REGIMES},
                         'contributing_families':{g:len({r['family_id'] for r in part if r['windows'][str(h)]['by_first_regime'][g]}) for g in REGIMES}}
                 for h in settings['coverage']['horizons']}
        splits[split]={'episodes':len(part),'families':len({r['family_id'] for r in part}),
                       'modes':{m:sum(r['mode']==m for r in part) for m in MODES},
                       'recorded_transitions':sum(r['recorded_transitions'] for r in part),'valid_transitions':sum(r['valid_transitions'] for r in part),
                       'excluded_transitions':sum(r['excluded_transitions'] for r in part),'regimes_valid':{g:sum(r['regime_counts_valid'][g] for r in part) for g in REGIMES},
                       'windows':windows,'translation_signal_episodes':sum(r['translation_signal'] for r in part),
                       'rotation_signal_episodes':sum(r['rotation_signal'] for r in part),
                       'qualified_recontact_episodes':sum(r['qualified_recontact_count']>0 for r in part)}
    return {'splits':splits,'episodes':rows,'scope':'overlapping windows are not independent trials; prescribed modes differ from observed regimes; five families is minimum coverage, not power'}


def collect(out,plans,settings,config,contract,selected):
    dataset=out/'data'/settings['collection_id'].replace('-development',''); writer=DatasetWriter(dataset,contract,purpose='main',split_seed=settings['split_seed'])
    policy_folder=out/'policy'; policy_folder.mkdir(); env=PlanarPushEnv(config)
    ledger=[]; guards={k:0 for k in ENGINE_GUARDS}; step_times=[]
    for plan in plans:
        started=perf_counter(); recorder=None; records=[]; reason='collection_complete'
        try:
            env.reset(InitialSpec(**plan['initial_spec'])); start=env.observe()
            policy=MainCollectionPolicy(config,settings,plan,selected); context=policy.initial_context()
            recorder=EpisodeRecorder(env,episode_id=plan['episode_id'],episode_index=plan['episode_index'],family_id=plan['family_id'],family_definition=plan['family_definition'],split=plan['split'],collection_policy={'kind':plan['mode'],'purpose':'main','collection_id':settings['collection_id'],'predeclared_plan':plan,'selected_G_override':selected,'task_outcome':'NOT_EVALUATED'},seed=settings['initial_seed'],contract=contract)
            for tick in range(plan['max_ticks']):
                obs=env.observe()
                with model_planning_scope(guards):
                    decision=policy.propose(obs.state,context,obs.sim_time-start.sim_time,tick)
                record={'tick':tick,'interval_start_time':obs.sim_time,'diagnostics':decision.diagnostics,'requested_action':decision.action.tolist(),'executed':False}
                records.append(record)
                if not decision.diagnostics['feasible']:
                    reason='controller_infeasible'; break
                clock=perf_counter(); result=recorder.step(decision.action); step_times.append(perf_counter()-clock)
                record.update(executed=True,applied_action=result.applied_action.tolist(),sim_time=result.observation.sim_time)
                context=next_context(decision,result)
                if recorder.invalid: reason='physical_or_geometry_failure'; break
            episode=recorder.finish(reason); writer.append(episode)
            save(policy_folder/f"{plan['episode_id']}.json",{'episode_id':plan['episode_id'],'mode':plan['mode'],'records':records,'terminal_reason':episode.metadata['terminal_reason']})
            ledger.append({'episode_id':plan['episode_id'],'mode':plan['mode'],'split':plan['split'], 'status':reason,'terminal_reason':episode.metadata['terminal_reason'],'planned_ticks':plan['max_ticks'],'recorded_transitions':episode.transition_count,'valid_transitions':int(episode.arrays['transition_valid'].sum()),'wall_time_s':perf_counter()-started})
            del recorder,episode,records; gc.collect()
        except Exception as error:
            folder=out/'collection_errors'/plan['episode_id'];folder.mkdir(parents=True)
            save(folder/'failure.json',{'plan':plan,'message':str(error),'exception':type(error).__name__,'traceback':traceback.format_exc(),'records':records})
            if recorder is not None: save_trace(folder,'recorded_prefix',recorder.snapshots[0],recorder.results)
            ledger.append({'episode_id':plan['episode_id'],'status':'collection_error','message':str(error)})
        if len(ledger)%6==0: print(json.dumps({'progress':'collection','attempted':len(ledger),'planned':len(plans)}),flush=True)
    save(out/'collection_ledger.json',ledger)
    if any(r['status']=='collection_error' for r in ledger):
        save(out/'incomplete_dataset.json',{'entries':writer.entries,'ledger':ledger}); raise ValueError('Collection errors retained; no complete manifest')
    manifest=writer.finish()
    return dataset,manifest,ledger,guards,step_times


def audit(out,dataset,plans,settings,contract,selected):
    rows=[]; supports={s:[] for s in ('train','validation')}; behavior={}; replay_pass=True; guards={k:0 for k in ENGINE_GUARDS}
    for episode in iter_dataset(dataset,contract):
        plan=plans[episode.metadata['episode_index']]
        if episode.metadata['collection_policy']['predeclared_plan']!=plan or episode.metadata['initial_record']['initial_spec']!=plan['initial_spec'] or any(episode.metadata[k]!=plan[k] for k in ('episode_id','family_id','split')):
            raise ValueError('Stored episode mismatches frozen plan')
        record=json.loads((out/'policy'/f"{plan['episode_id']}.json").read_text())['records']
        executed=[r for r in record if r['executed']]
        if len(executed)!=episode.transition_count: raise ValueError('Decision/action length mismatch')
        policy=MainCollectionPolicy(contract['resolved_config'],settings,plan,selected); context=policy.initial_context()
        for tick,r in enumerate(executed):
            with model_planning_scope(guards):
                decision=policy.propose(episode.arrays['state'][tick].copy(),context,float(episode.arrays['sim_time'][tick]-episode.arrays['sim_time'][0]),tick)
            replay_pass &= bool(decision.diagnostics==r['diagnostics'] and np.array_equal(decision.action,episode.arrays['action_requested'][tick]) and np.array_equal(episode.arrays['action_requested'][tick],r['requested_action']) and np.array_equal(episode.arrays['action_applied'][tick],r['applied_action']))
            actual=episode.arrays['action_applied'][tick].copy();actual.flags.writeable=False
            context=replace(decision.context,previous_applied_action=actual)
        if (episode.metadata['terminal_reason']=='controller_infeasible') != (len(record)>len(executed)):
            raise ValueError('Infeasible termination requires exactly one retained unexecuted decision')
        if len(record)>len(executed):
            if len(record)!=len(executed)+1 or record[-1]['executed'] is not False or episode.metadata['terminal_reason']!='controller_infeasible':
                raise ValueError('Only one terminal infeasible proposal may be unexecuted')
            tick=len(executed)
            with model_planning_scope(guards):
                rejected=policy.propose(episode.arrays['state'][-1].copy(),context,float(episode.arrays['sim_time'][-1]-episode.arrays['sim_time'][0]),tick)
            replay_pass &= bool(not rejected.diagnostics['feasible'] and rejected.diagnostics==record[-1]['diagnostics'] and np.array_equal(rejected.action,record[-1]['requested_action']))
        row=episode_coverage(episode,contract,position_signal_m=settings['coverage']['position_signal_min_m'],yaw_signal_rad=settings['coverage']['yaw_signal_min_rad'],horizons=(1,5,10)); rows.append(row)
        s=episode.arrays['state'][:-1]; a=episode.arrays['action_applied']; valid=episode.arrays['transition_valid']
        rel=s[:,6:8]-s[:,:2]; d=np.column_stack((np.cos(s[:,2]),np.sin(s[:,2]))); t=np.column_stack((-d[:,1],d[:,0]))
        supports[plan['split']].append(np.column_stack((rel,s[:,2],s[:,3:6],s[:,8:10],a,(rel*d).sum(axis=1),(rel*t).sum(axis=1)))[valid])
        behavior[plan['episode_id']]={'holding_intervals':sum(r['diagnostics']['holding'] for r in executed), 'noise_applied_intervals':sum(r['diagnostics']['noise_applied'] for r in executed),
                                     'G_modes':dict(Counter(r['diagnostics']['mode_after'] for r in executed if r['diagnostics']['mode_after'] is not None)),
                                     'stationary_intervals':int(np.sum(valid & (np.linalg.norm(episode.arrays['state'][1:,:2]-s[:,:2],axis=1)<1e-9) & (np.abs(wrap_angle(episode.arrays['state'][1:,2]-s[:,2]))<1e-9))),
                                     'terminal_reason':episode.metadata['terminal_reason']}
        if len(rows)%10==0: print(json.dumps({'progress':'audit','episodes':len(rows)}),flush=True)
    coverage=aggregate(rows,settings);save(out/'coverage.json',coverage)
    report={'fields':['relative_pusher_x','relative_pusher_y','obj_yaw','obj_vx','obj_vy','obj_wz','pusher_vx','pusher_vy','applied_vx','applied_vy','rear_normal_coordinate','rear_tangent_coordinate'],'splits':{},'behavior':behavior,'scope':'marginal min/max and bins describe sampled support; not joint coverage, extrapolation safety or final success'}
    for split,parts in supports.items():
        x=np.concatenate(parts);norm=np.linalg.norm(x[:,8:10],axis=1);offset=x[:,-1];yaw=x[:,2]
        report['splits'][split]={'min':x.min(axis=0).tolist(),'max':x.max(axis=0).tolist(),'valid_rows':len(x),'command_norm_max_m_s':float(norm.max()),
                                 'command_norm_bins_edges_m_s':[0.,.01,.02,.03,.04+1e-12],'command_norm_bins_counts':np.histogram(norm,[0.,.01,.02,.03,.04+1e-12])[0].tolist(),
                                 'rear_tangent_bins_edges_m':[-float('inf'),-.015,-.005,.005,.015,float('inf')], 'rear_tangent_bins_counts':np.histogram(offset,[-np.inf,-.015,-.005,.005,.015,np.inf])[0].tolist(),
                                 'yaw_bins_edges_deg':[-float('inf'),-10.,-3.,3.,10.,float('inf')],'yaw_bins_counts':np.histogram(np.rad2deg(yaw),[-np.inf,-10.,-3.,3.,10.,np.inf])[0].tolist(),
                                 'planned_initial_eccentric_offsets_m':[p['pusher_tangent_offset_m'] for p in plans if p['split']==split and p['mode']=='eccentric']}
    save(out/'input_support.json',report)
    return coverage,report,replay_pass,guards


def prepare(out,dataset):
    prepared=prepare_dataset(dataset);normalizer=fit(prepared);save_learning_data(out/'learning_data',prepared,normalizer)
    restored,norm=load_learning_data(out/'learning_data',learning_schema(prepared.metadata['contract']),normalizer.metadata['dataset_identity'],normalizer.metadata['normalizer_hash'])
    equality=all(np.array_equal(prepared.arrays[k],restored.arrays[k]) for k in prepared.arrays) and all(np.array_equal(prepared.windows[k],restored.windows[k]) for k in prepared.windows) and norm.metadata==normalizer.metadata
    mutations=mutation_checks(prepared,normalizer)
    checks={'train_only_fit_count':normalizer.metadata['fit_count']==int((prepared.arrays['split_code']==0).sum()),'stored_bundle_reload_identical':equality,
            'validation_stats_invariance':all(mutations[k] for k in ('statistics_arrays_identical','train_row_hash_identical','stats_hash_identical','full_provenance_hash_changes','train_mutation_positive_control_changes_stats_hash'))}
    save(out/'preparation.json',{'checks':checks,'mutations':mutations,'metadata':prepared.metadata,'normalizer':normalizer.metadata,'features_shape':list(prepared.arrays['features'].shape),'deltas_shape':list(prepared.arrays['deltas'].shape)})
    return checks,normalizer.metadata


def geometry_reference_checks(folder, contract, parameter_path, settings):
    """Bind actual collection to a formal, same-contract G acceptance run."""
    reference=json.loads((folder/'manifest.json').read_text())
    result=json.loads((folder/'summary.json').read_text())
    proposed={k:v for k,v in settings.get('geometry_transition',{}).items() if k!='noise_weighting'} or None
    required=('four_fixed_cases_accounted','four_cases_ready','four_case_requirements',
              'regression_tests_pass','four_raw_episodes_audited','future_engine_queries_zero',
              'no_unresolved_physics','source_frozen','prior_artifacts_unchanged')
    dependencies=('geometry_control.py','environment.py','config.py','planning.py','task.py',
                  'representation.py','contact_geometry.py','closed_loop.py','data.py')
    if proposed: dependencies=(*dependencies,'geometry_control_smooth.py')
    hashes=reference.get('source_hashes',{})
    return {'G_reference_is_formal_acceptance':reference.get('task_id')=='P3-01'
            and reference.get('mode')=='formal_four_case_acceptance'
            and result.get('task_id')=='P3-01' and result.get('mode')=='formal_four_case_acceptance'
            and result.get('pass') is True and all(result.get('checks',{}).get(k) is True for k in required),
            'G_reference_contract_matches':reference.get('contract_hash')==contract_hash(contract)
            and reference.get('profile_sha256')==contract['profile_sha256']
            and reference.get('physical_config')==contract['resolved_config'],
            'G_reference_transition_matches':reference.get('transition_parameters')==proposed,
            'G_reference_selected_parameters_match':reference.get('requested_parameter_file_sha256')==digest(parameter_path),
            'G_reference_runtime_dependencies_match':all(hashes.get('src/push_align/'+name)==digest(ROOT/'src/push_align'/name) for name in dependencies)}


def run(args,out):
    started=perf_counter();settings=tomllib.loads(args.collection_config.read_text());selected=json.loads(args.geometry_parameters.read_text())
    env=PlanarPushEnv();contract=make_contract(env);plans,assignments=build_main_collection_plan(settings,env.config);validate_settings(settings,contract)
    sources=source_bundle(out)
    for path in (args.geometry_parameters,):
        rel=path.resolve().relative_to(ROOT);target=out/'source'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target);sources[str(rel)]=digest(path)
    upstream=getattr(args,'geometry_reference_run',None) or ROOT/'runs/p3-01-20261008-01'; upstream_before=verify_artifacts(upstream)
    adoption=json.loads((ROOT/'reports/p1_environment_adoption.json').read_text())
    prereq={'P1_adopted':adoption['status']=='PASS','environment_matches_adoption':contract['environment_hash']==adoption['environment_hash'],'config_matches_adoption':contract['config_sha256']==adoption['config_sha256'],'profile_matches_adoption':contract['profile_sha256']==adoption['acceptance_sha256'], 'P3_01_pass':json.loads((upstream/'summary.json').read_text())['pass'] is True}
    free=shutil.disk_usage(ROOT).free; estimate=settings['resources']['maximum_planned_transitions']*550000+350000000; reserve=settings['resources']['free_disk_reserve_gib']*1024**3
    save(out/'manifest.json',{'task_id':'P3-02','created_at':datetime.now().astimezone().isoformat(),'argv':sys.argv,'executable':sys.executable,'python':sys.version,'platform':platform.platform(),'contract':contract,'contract_hash':contract_hash(contract),'collection_settings':settings,'collection_config_sha256':digest(args.collection_config),'selected_G_override':selected,'selected_G_sha256':digest(args.geometry_parameters),'predeclared_episodes':plans,'family_assignments':assignments,'source_hashes':sources,'upstream':upstream_before,'disk_preflight':{'free_bytes':free,'estimated_additional_bytes':estimate,'reserve_bytes':reserve}})
    if not args.gate_only:
        prereq.update(geometry_reference_checks(upstream,contract,args.geometry_parameters,settings))
    if not all(prereq.values()) or free<estimate+reserve: raise ValueError('Prerequisites or disk reserve failed')
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,text=True,capture_output=True)
    (out/'tests.txt').write_text(tests.stdout+tests.stderr)
    match=re.search(r'Ran (\d+) tests',tests.stdout+tests.stderr)
    if tests.returncode or match is None or int(match[1])==0 or 'skipped=' in tests.stdout+tests.stderr:
        raise ValueError('Positive complete regression tests required before collection')
    gate=physics_gate(out,plans,settings,env.config,contract,selected)
    if args.gate_only or not gate['pass']:
        save(out/'summary.json',{'task_id':'P3-02','stage_verdict':'GATE_PASS' if gate['pass'] else 'FAIL','tests_run':int(match[1]) if match else None,'physics_extension':gate,'main_collection':'NOT_RUN','planned_episodes':len(plans),'not_started_episodes':len(plans),'alignment_outcome':'NOT_EVALUATED'})
        return gate['pass']
    clock=perf_counter();dataset,manifest,ledger,guards,step_times=collect(out,plans,settings,env.config,contract,selected);collection_s=perf_counter()-clock
    clock=perf_counter();coverage,support,replay_pass,audit_guards=audit(out,dataset,plans,settings,contract,selected);audit_s=perf_counter()-clock
    prep_checks,norm=prepare(out,dataset);upstream_after=verify_artifacts(upstream)
    family_counts=coverage['splits']['validation']['windows']['5']['contributing_families']
    checks={**prereq,'regression_tests':tests.returncode==0,'physics_extension':gate['pass'],'all_90_plans_recorded':len(ledger)==len(coverage['episodes'])==manifest['episode_count']==len(plans),
            'all_split_and_mode_counts_match':all(coverage['splits'][s]['episodes']==n and all(v==n//6 for v in coverage['splits'][s]['modes'].values()) for s,n in settings['split_counts'].items()),
            'zero_integrity_errors':True,'stored_policy_replay':replay_pass,'validation_three_K5_regimes_min5_families':all(family_counts[g]>=5 for g in REGIMES),
            'translation_and_rotation_signal_per_split':all(r['translation_signal_episodes']>0 and r['rotation_signal_episodes']>0 for r in coverage['splits'].values()),
            'qualified_recontact_per_split':all(r['qualified_recontact_episodes']>0 for r in coverage['splits'].values()),
            'no_physical_or_geometry_failures':all(r['status']!='physical_or_geometry_failure' for r in ledger), 'zero_future_engine_access':sum(guards.values())+sum(audit_guards.values())==0,
            'bounded_actual_actions':all(r['command_norm_max_m_s']<=.04+1e-15 for r in support['splits'].values()),
            'upstream_artifacts_unchanged':upstream_before==upstream_after,'source_files_unchanged':all(digest(ROOT/path)==sha for path,sha in sources.items()),**prep_checks}
    resources={'collection_wall_time_s':collection_s,'audit_wall_time_s':audit_s,'total_wall_time_s':perf_counter()-started,'peak_self_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024),'step_with_raw_recording_median_ms':float(np.median(step_times)*1000),'step_with_raw_recording_p95_ms':float(np.percentile(step_times,95)*1000),'dataset_bytes':sum(p.stat().st_size for p in dataset.rglob('*') if p.is_file()),'scope':'CPU collection/storage/audit/preparation; excludes child RSS; not model training or control latency'}
    save(out/'resources.json',resources)
    summary={'task_id':'P3-02','stage_verdict':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'tests_run':int(match[1]) if match else None,'planned_episodes':len(plans),'recorded_transitions':manifest['transition_count'],'valid_transitions':manifest['learning_transition_count'],'excluded_transitions':manifest['excluded_transition_count'],'terminal_counts':dict(Counter(r['status'] for r in ledger)), 'coverage_by_split':coverage['splits'],'physics_extension':gate,'dataset_manifest':str(dataset.relative_to(out)/'manifest.json'),'dataset_manifest_sha256':digest(dataset/'manifest.json'),'normalizer_hash':norm['normalizer_hash'],'stats_hash':norm['stats_hash'],'train_rows_sha256':norm['train_rows_sha256'],'train_fit_count':norm['fit_count'],'resources':resources,'upstream_verified':upstream_before,'alignment_outcome':'NOT_EVALUATED','learner_understanding':'NOT_ASSESSED','dynamics_training':'NOT_RUN','MPC':'NOT_RUN','next_task':'P3-03 Ridge same main data/representation','scope':'bounded main-development v1; minimum actual coverage and data integrity only; not statistical adequacy or task performance'}
    save(out/'summary.json',summary); print(json.dumps({'verdict':summary['stage_verdict'],'tests':summary['tests_run'],'transitions':summary['recorded_transitions'],'validation_K5_families':family_counts}),flush=True)
    return all(checks.values())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--collection-config',type=Path,default=ROOT/'configs/p3_collection.v1.toml');parser.add_argument('--geometry-parameters',type=Path,default=ROOT/'configs/p3_geometry_selected.v1.json');parser.add_argument('--geometry-reference-run',type=Path);parser.add_argument('--gate-only',action='store_true');args=parser.parse_args()
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    try: passed=run(args,out)
    except Exception as error:
        save(out/'failure.json',{'exception':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        if not (out/'summary.json').exists(): save(out/'summary.json',{'task_id':'P3-02','stage_verdict':'ERROR','reason':str(error),'alignment_outcome':'NOT_EVALUATED'})
        print(json.dumps({'verdict':'ERROR','reason':str(error)}),flush=True);passed=False
    hashes={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()};save(out/'artifact_hashes.json',{'sha256':hashes})
    raise SystemExit(0 if passed else 1)


if __name__=='__main__': main()
