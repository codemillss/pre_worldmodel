"""Freeze P3-05 choices and conservative remaining work from actual costs.

Original execution summary and hash index stay immutable. Completion is a
separate gate, requiring the independent raw/middle-plan audit and scope review.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/p3-05-20261010-01'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')


def main():
    s=json.loads((RUN/'summary.json').read_text());m=json.loads((RUN/'manifest.json').read_text())
    if not s['execution_checks_pass']:raise ValueError('Incomplete execution')
    source_index=json.loads((RUN/'artifact_hashes.json').read_text())['sha256']
    if not all(sha(RUN/p)==h for p,h in source_index.items()):raise ValueError('Original execution changed')
    selected=s['planning_selection'];geometry=s['geometry_selection']
    planner={'protocol_id':'p3-common-planning-v1','selection_run':str(RUN.relative_to(ROOT)),
             'selection_summary_sha256':sha(RUN/'summary.json'),'selection_id':selected['id'],
             'planner':selected['planner']['planner'],'cost':selected['planner']['cost'],
             'task':selected['planner']['task'],'physical_config_id':selected['planner']['config_id'],
             'physical_config_rule':'load saved original model/environment config; override planner only; no environment/normalizer mutation',
             'geometry_selected_overrides':geometry['parameters'],
             'geometry_effective_16':{**m['physical_config']['geometry_baseline'],**geometry['parameters']},
             'geometry_transition':json.loads((ROOT/'configs/p3_geometry_transition.v2.json').read_text()),
             'geometry_context_policy':m['settings']['geometry_context_commit'],
             'rng_rule':m['settings']['candidate_seed_keys'],'development_base_seed':4711,'final_base_seed_candidate':6811,
             'selection_scope':'3 previously used cases x 4 model instances; 5s surrogate, MLP seeds jointly have 3/4 cell weight; not independent superiority',
             'holding':'common zero command; previous actual action zero, warm sequence clear; actual control tick includes holding',
             'all_invalid':'planner_invalid; save unexecuted full trace; no extra zero physics step'}
    common=json.loads((ROOT/'runs/p3-04-20261009-01/summary.json').read_text())
    fork={'training_id':'p4-matched-window-budget-v2','seeds':[11,23,47],
          'common_checkpoints':[{'seed':x['seed'],'path':'runs/p3-04-20261009-01/'+x['selected_checkpoint'],
              'checkpoint_hash':x['selected_checkpoint_hash'],'initial_updates':x['selected_updates']} for x in common['seeds']],
          'common_start':'same seed-specific selected weights, Adam moments/step, Torch/shuffle RNG, normalizer; no timing-probe weights',
          'additional_updates_per_branch':1000,'batch_windows':64,'K':5,'drop_last':False,
          'window_count_train':2960,'window_count_validation':1480,
          'window_sampler':'epoch randperm of 2960 original train K5 windows using restored common shuffle generator; 46x64+16 per epoch; same order and partial final batch in every branch',
          'update_budget_arithmetic':'21 complete window epochs x47 +13 updates =1000; 62992 window occurrences, 314960 transition occurrences per branch; overlapping windows not independent samples',
          'lambda_candidates':[0.1,1.0],'branches_per_seed':['S1','SK-lambda0.1','SK-lambda1.0'],
          'total_branches':9,'total_additional_updates':9000,
          'one_step_term':'all five real transitions in same window; original physical-delta angular-aware loss/train delta scale; identical labels/mask/weights in S1 and SK',
          'rollout_term':'SK only; predicted complete state recurrently fed to next step; k=1..5, exclude k=0; state yaw error wrap2pi; stored train_delta_scale; mean over valid window/time/state dimensions',
          'optimizer':'resume Adam; lr0.001/weight_decay0/gradient_clip_norm1; finite loss/grad/parameter guards',
          'final_comparison':'same final additional update1000; intermediate best epochs never replace final weights',
          'lambda_selection':'one global lambda from mean of all three seed final-update validation K5 equal-regime scores; any invalid seed makes lambda ineligible; exact tie smaller lambda; no best-seed selection',
          'checkpoint_every_updates':100,'checkpoint_count_candidate':90,
          'mid_epoch_sampler_checkpoint':'save actual current window permutation, epoch, cursor, restored shuffle RNG and their hashes; checkpoint100/1000 can be mid-epoch; verify exact next batch/update on independent reload',
          'all_lambda_ineligible':'P4-02 FAIL; preserve all branches and diagnostic final weights, never label fallback as selected SK',
          'evaluation':'final each branch K1/K5/K10 and selected lambda all seeds; diagnostics at checkpoint100 multiples; test never used',
          'P4_01_formal_loss_gradient_mask_tests':'NOT_RUN','formal_branches':'NOT_RUN'}
    cells=[]
    for p in (RUN/'planning_trials'/selected['id']).glob('*/smoke-*/summary.json'):cells.append(json.loads(p.read_text()))
    cells+=s['full_resource_rows']
    raw=max(x['resources']['raw_bytes']/x['transitions'] for x in cells)
    nonraw=max((x['resources']['stored_bytes']-x['resources']['raw_bytes'])/x['transitions'] for x in cells)
    gcells=[json.loads(p.read_text()) for p in (RUN/'geometry_trials').glob('*/smoke-*/summary.json')]+[x for x in s['full_resource_rows'] if x['model_id']=='G']
    gmetadata=max((x['resources']['stored_bytes']-x['resources']['raw_bytes'])/x['transitions'] for x in gcells)
    wall=max(x['resources']['wall_time_s']/x['transitions'] for x in s['full_resource_rows'])
    factor=1.5;f=12;h=selected['planner']['planner']['horizon'];gib=2**30
    parts={'E2_raw':f*8*300*raw*factor,'E2_plan_and_cell_metadata':f*7*300*nonraw*factor,
           'E2_G_cell_metadata':f*300*gmetadata*factor,
           'E1_raw':12*16*h*3*raw*factor,'E1_predictions_and_analysis_allowance':.1*gib,
           'learning_checkpoints_validation_allowance':gib,'P4_development_allowance':gib,
           'P5_replay_boundary_witness_learned':12*7*300*nonraw*factor,
           'P5_replay_boundary_witness_G':12*300*gmetadata*factor,
           'P5_boundary_arrays_allowance':.1*gib,
           'P5_one_new_family_raw':8*300*raw*factor,
           'P5_one_new_family_learned_metadata':7*300*nonraw*factor,
           'P5_one_new_family_G_metadata':300*gmetadata*factor,
           'P5_mismatch_full_raw_allowance':.25*gib,
           'free_disk_reserve':2*gib}
    available=shutil.disk_usage(ROOT).free
    probe=s['resources']['training_cost_probe'];training=3000*probe[0]['time']['mean_s']+6000*probe[1]['time']['mean_s']
    scope={'scope_id':'p3-remaining-work-v3','created_at':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
           'supersedes':'v1/v2 configs and scope_freeze records retained; v1 omitted G metadata/three-replay cost rule; v2 corrected E1 but omitted P5 full stratum/terminal union replay count; v3 preserves both original criteria with explicit replay witness storage; online planning and learning core unchanged',
           'physical_contract':'unchanged p3-main-v3 physical v4/acceptance-v1; task goal [0.08,0,0], 5mm/3deg and measured low speeds for >=0.3s, max30s, retries0',
           'data':'unchanged 90 families, train60/validation30, original train3200/validation1600; no new collection required',
           'learning':'three seeds, K5, two lambda candidates, matched1000 additional updates/branch; nine training branches',
           'E2':{'family_count':12,'strata_count':3,'families_per_stratum':4,'controllers_per_family':8,
                 'controllers':['G','R','S1-11','S1-23','S1-47','SK-11','SK-23','SK-47'],'episodes':96,'max_action_intervals':28800,
                 'strata':[{'id':'translation','yaw_deg':0,'tangent_offset_m':0},
                           {'id':'rotation','abs_yaw_deg_range':[5,15],'yaw_sign_balance':[2,2],'tangent_offset_m':0},
                           {'id':'offset','yaw_deg_range':[-15,15],'abs_tangent_offset_m_range':[.01,.015],'offset_sign_balance':[2,2]}],
                 'all_strata_object_x_range_m':[-.02,.02],'all_strata_object_y_range_m':[-.01,.01],
                 'initial_seed_candidate':6711,'candidate_seed_candidate':6811,
                 'test_generation':'P5-01 only; canonical initial pose/pusher identity disjoint from train/validation/previous development; outcomes never guide replacement',
                 'normal_failures':'retain timeout/planner_invalid/workspace and physics failures with original denominator; unresolved infrastructure blocks complete evaluation',
                 'final_manifest':'NOT_GENERATED'},
           'E1':{'snapshots':12,'family_count':6,'families_per_stratum':2,'snapshots_per_family':2,
                 'family_selection':'first two final families by fixed ID in each stratum; selected before results',
                 'ticks':[0,29],'snapshot_source':'G actual E2 path before decision; full physical+task+actual previous action+G context+RNG; warm None shared; initial and later contact-capable prefix, actual regime reported not assumed',
                 'missing_snapshot':'record unavailable and stop E1 completion; no successful-family substitution',
                 'candidates':16,'horizon':h,'candidate_stream_id':1,'physical_repeats_per_candidate':3,
                 'candidate_groups':{'reserved':8,'global':4,'local_geometry':2,'local_warm':2},
                 'actual_future_max_intervals':12*16*h*3,
                 'future_rule':'independent offline physical replay from same complete snapshot three times per candidate, shared by all seven model instances, not repeated separately by model; never fed into online planning',
                 'cost_range_rule':'preserve docs06: [min(J)-epsilon,max(J)+epsilon], epsilon=profile atol+rtol*max(abs(J)); overlap=>ranking INCONCLUSIVE, repeated validity changes=>investigate; keep all attempts/costs/denominators',
                 'limits':'N16 diagnostic set is smaller than v1 N32 and online N128; 12 snapshots are six independent families, no full-N optimality claim','execution':'NOT_RUN'},
           'E3':'same S1/SK raw E1/E2 contrasts with other settings matched; no redundant physics benchmark',
           'statistics':{'bootstrap_replicates':2000,'bootstrap_seed':7711,'unit':'family within stratum, common resample across models; seed averages first within each family plus each seed separately',
                         'limitations':'four families per E2 stratum and two per E1 stratum; exploratory uncertainty, no power/equivalence or industrial population guarantee; intervals do not cover all training-seed variation'},
           'P4_development_resource_allowance':{'closed_loop_cells':8,'closed_loop_max_intervals':2400,
                 'E1_snapshots':6,'E1_candidates':16,'E1_repeats':3,'E1_horizon':h,'E1_max_intervals':6*16*h*3,
                 'E1_selection':'two development snapshots per stratum/regime by fixed source ID/tick declared in P4-03 before physical outcomes; previous development data only, final test not used',
                 'computed_raw_and_plan_bytes':(2400+6*16*h*3)*raw*factor+2400*nonraw*factor,
                 'analysis_allowance_bytes':.1*gib,'total_allowance_bytes':gib,
                 'rule':'failure replay and single-change hypotheses plus development E1; revise estimate before exceeding declared work/space; no future criterion removed'},
           'P5_replay_application_resource_allowance':{
                 'original_selection_rule':'for each method/learning seed, smallest episode ID per stratum plus smallest per observed termination type, union deduplicate; include both success and failure when observed',
                 'existing_replay_episode_upper_bound':96,'new_family_controllers':8,'episode_upper_bound':104,'max_intervals':31200,
                 'existing_replay_record':'all requested/applied commands, actual interval times, pre/post state10, task/holding/terminal/validity, candidate/prediction/selected hashes and maximum component differences at every action boundary; original complete raw+model/source/config hashes referenced, original files retained',
                 'existing_replay_scope':'comparison witness produced by actual controller rerun, not table regeneration; preserve existing profile0.1percent action-boundary max position/yaw/speed tolerance and same termination; do not claim every raw diagnostic byte or substep invariance from this witness',
                 'witness_budget_basis':'conservative observed nonraw/interval max with1.5 allowance; exact new witness schema and actual bytes/interval verified before P5 batch',
                 'mismatch_policy':'keep full raw executed prefix on mismatch/physics error; 0.25GiB initial allowance, if insufficient stop and reestimate without dropping cases/relaxing criteria; P5-04 incomplete until all required reruns resolved',
                 'new_initial_condition':'one separate same-range initial family, full raw for all eight controllers; predeclared expected effect/changes/result, not OOD/new task'},
           'storage':{'observed_raw_bytes_per_interval_max':raw,'observed_nonraw_bytes_per_interval_max':nonraw,
                      'safety_multiplier':factor,'multiplier_is':'planning allowance, not measured worst-case bound',
                      'observed_G_metadata_bytes_per_interval_max':gmetadata,
                      'parts_bytes':parts,'required_bytes_including_reserve':sum(parts.values()),'free_bytes_at_freeze':available,
                      'headroom_beyond_reserved_bytes':available-sum(parts.values()),
                      '24family_same_allowances_required_bytes':sum(parts.values())+parts['E2_raw']+parts['E2_plan_and_cell_metadata']+parts['E2_G_cell_metadata'],
                      '60family_same_allowances_required_bytes':sum(parts.values())+4*parts['E2_raw']+4*parts['E2_plan_and_cell_metadata']+4*parts['E2_G_cell_metadata'],
                      'new_recording':'all original StepResult fields and initial snapshot compressed losslessly, all applied/requested commands and raw substeps retained; every plan compact replay inputs/hash and predeclared full snapshots/invalid trace',
                      'old_records':'all original successes/failures/models/statistics/sources kept unchanged',
                      'guard':'before each stage remeasure free disk and actual bytes/interval; minimum free2GiB; exceed estimate => preserve prefix and revise scope before continuing'},
           'time':{'observed_full_case_wall_s_per_interval_max':wall,'extrapolation_factor':factor,
                   'E2_max_action_work_extrapolated_s':28800*wall*factor,'E1_max_action_work_extrapolated_s':12*16*h*3*wall*factor,
                   'P4_dev_max_action_work_extrapolated_s':(2400+6*16*h*3)*wall*factor,'P5_replay_apply_max_action_work_extrapolated_s':31200*wall*factor,
                   'nine_branch_update_path_mean_extrapolated_s':training,'nine_branch_update_path_with_allowance_s':training*factor,
                   'excluded_or_uncertain':'new E1 implementation/labels, P4 formal validation/checkpoints/audit, setup/load/bootstrap/report and retries; no guaranteed completion date or device real-time claim',
                   'same_updates_not_same_time':True},
           'measured_device_scope':{'platform':m['platform'],'cpu':'local arm64 CPU; Torch1; BLAS threads unmeasured','peak_rss_MiB':s['resources']['main_peak_rss_MiB'],
                                    'other_M1_M4':'not separately measured; one small replay/training timing on the destination device before extrapolating'},
           'budget':'no paid service/cloud/equipment; additional monetary spending0; under100000KRW scope',
           'rationale':'12 family preserves contact/rotation/three seeds/learning-control; 24 and original60 exceed current conservative storage. Sample size changes, learning core retained.',
           'stage_scope':'P3 comparison preparation only; P4/P5 formal work not run'}
    if scope['storage']['headroom_beyond_reserved_bytes']<0:raise ValueError('Chosen scope exceeds available storage including reserve')
    dev=scope['P4_development_resource_allowance']
    if dev['computed_raw_and_plan_bytes']+dev['analysis_allowance_bytes']>dev['total_allowance_bytes']:raise ValueError('P4 development allowance too small')
    if scope['storage']['24family_same_allowances_required_bytes']<=available:raise ValueError('Reconsider scope rationale; 24 now fits')
    if json.loads((ROOT/'configs/p3_planning.v1.json').read_text())!=planner:raise ValueError('Planning selection changed during scope correction')
    if json.loads((ROOT/'configs/p4_fork_budget.v2.json').read_text())!=fork:raise ValueError('Learning budget changed during resource-only correction')
    save(ROOT/'configs/p3_evaluation_scope.v3.json',scope)
    save(RUN/'scope_freeze.v3.json',{'task_id':'P3-05-scope-v3','planning':planner,'training':fork,'evaluation':scope,
                                'original_execution_summary_sha256':sha(RUN/'summary.json'),
                                'original_artifact_index_sha256':sha(RUN/'artifact_hashes.json'),'code_sha256':sha(Path(__file__))})
    print(json.dumps({'required_GiB':sum(parts.values())/gib,'available_GiB':available/gib,'headroom_GiB':(available-sum(parts.values()))/gib,'scope_files':3}))


if __name__=='__main__':main()
