"""Independent NumPy-only audit of stored data-intervention candidate results.

No production evaluator imports, model loading/forward, simulator, or learning.
Numerical allclose below is audit arithmetic roundoff, not a new task threshold.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('R', 'D00', 'D10', 'D01', 'D11')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def archive(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def signed_gap(states, config):
    relative = states[..., 6:8] - states[..., :2]
    c, s = np.cos(states[..., 2]), np.sin(states[..., 2])
    local = np.stack((c*relative[..., 0] + s*relative[..., 1],
                      -s*relative[..., 0] + c*relative[..., 1]), axis=-1)
    q = np.abs(local)-np.asarray(config['object']['size_full_xyz_m'][:2])/2
    return np.hypot(np.maximum(q[..., 0], 0), np.maximum(q[..., 1], 0)) \
        + np.minimum(np.max(q, axis=-1), 0)-config['pusher']['radius_m']


def task_ratios(states, task):
    yaw = states[..., 2]-task['goal'][2]
    return np.stack((np.hypot(states[..., 0]-task['goal'][0], states[..., 1]-task['goal'][1])/task['position_tolerance_m'],
        np.abs(.5*np.arctan2(np.sin(2*yaw), np.cos(2*yaw)))/task['yaw_tolerance_rad'],
        np.hypot(states[..., 3], states[..., 4])/task['linear_speed_tolerance_m_s'],
        np.abs(states[..., 5])/task['angular_speed_tolerance_rad_s']), axis=-1)


def legacy_cost(states, actions, previous, task, config):
    s = states[:, 1:]; w = config['cost']; limit = config['pusher']['command_speed_norm_max_m_s']
    delta = s[..., 2]-task['goal'][2]
    position = np.hypot(s[..., 0]-task['goal'][0], s[..., 1]-task['goal'][1])/task['position_tolerance_m']
    yaw = .5*np.arctan2(np.sin(2*delta), np.cos(2*delta))/task['yaw_tolerance_rad']
    previous_actions = np.concatenate((np.broadcast_to(previous, (len(actions), 1, 2)), actions[:, :-1]), axis=1)
    components = {
        'stage_position': w['stage_position']*.1*np.sum(position**2, axis=1),
        'stage_yaw': w['stage_yaw']*.1*np.sum(yaw**2, axis=1),
        'stage_action': w['stage_action']*.1*np.sum(np.sum((actions/limit)**2, axis=-1), axis=1),
        'stage_action_change': w['stage_action_change']*.1*np.sum(np.sum(((actions-previous_actions)/limit)**2, axis=-1), axis=1),
        'stage_gap': w['stage_gap']*.1*np.sum((np.maximum(signed_gap(s, config), 0)/w['gap_scale_m'])**2, axis=1),
        'terminal_position': w['terminal_position']*position[:, -1]**2,
        'terminal_yaw': w['terminal_yaw']*yaw[:, -1]**2,
        'terminal_speed': w['terminal_speed']*(np.hypot(s[:, -1, 3], s[:, -1, 4])/task['linear_speed_tolerance_m_s'])**2,
        'terminal_angular_speed': w['terminal_angular_speed']*(s[:, -1, 5]/task['angular_speed_tolerance_rad_s'])**2}
    return np.sum(np.stack(list(components.values())), axis=0), components


def endpoint_rank(states, costs, task, initial_status):
    ratios = task_ratios(states, task)
    qualified = (ratios <= 1).all(axis=-1)
    held = np.zeros(qualified.shape, np.float64)
    held[:, 0] = initial_status['held_time_s']
    first = np.full(len(states), -1, np.int64)
    for t in range(1, 11):
        held[:, t] = np.where(qualified[:, t-1] & qualified[:, t], held[:, t-1]+.1, 0.)
        newly = (first == -1) & (held[:, t]+1e-9 >= task['hold_time_s'])
        first[newly] = t
    rank = []
    for i, end in enumerate(np.where(first < 0, 10, first)):
        excess = np.maximum(ratios[i]-1, 0).max(axis=-1)
        rank.append([float(first[i] < 0), float(excess[end]), float(.1*excess[1:end+1].sum()), float(costs[i])])
    return dict(ready=first >= 0, first=first, rank=np.asarray(rank), ratios=ratios, held=held)


def physical_ready(raw, task, status):
    held, previous, last = status['held_time_s'], status['previous_qualifies'], status['last_sim_time']
    qualified = (task_ratios(raw['substep_states'], task) <= 1).all(axis=-1)
    interval = len(qualified)//10
    for tick in range(10):
        for n in range(tick*interval, (tick+1)*interval):
            time = float(raw['substep_times'][n]); current = bool(qualified[n])
            held = held+(time-last) if current and previous else 0.
            previous, last = current, time
        if held+1e-9 >= task['hold_time_s']:
            return True, tick+1, held
    return False, None, held


def errors(truth, prediction):
    error = prediction-truth
    error[:, 2] = (error[:, 2]+np.pi)%(2*np.pi)-np.pi
    return {'object_xy_mm': np.linalg.norm(error[:, :2], axis=-1)*1000,
        'object_yaw_deg': np.rad2deg(np.abs(error[:, 2])),
        'object_velocity_mm_s': np.linalg.norm(error[:, 3:5], axis=-1)*1000,
        'object_wz_rad_s': np.abs(error[:, 5]),
        'pusher_xy_mm': np.linalg.norm(error[:, 6:8], axis=-1)*1000,
        'pusher_velocity_mm_s': np.linalg.norm(error[:, 8:10], axis=-1)*1000}


def check(run, report):
    folder = run/'candidate_evaluation'; summary = read(folder/'summary.json')
    parent = ROOT/summary['source_run']; source = read(parent/'input.json')
    config = read(parent/'manifest.json')['head_planner']; task = source['task']
    status, previous = source['status'], np.asarray(source['context']['previous_applied_action'])
    checks = []
    def record(name, okay, detail=None):
        checks.append({'id': name, 'pass': bool(okay), **({'detail': detail} if detail is not None else {})})
    def same(name, a, b):
        a, b = np.asarray(a), np.asarray(b)
        okay = a.shape == b.shape and np.allclose(a, b, atol=1e-12, rtol=1e-12, equal_nan=True)
        record(name, okay)
    idx = read(folder/'artifact_hashes.json')['sha256']
    for name, expected in idx.items(): record('artifact:'+name, sha(folder/name) == expected)
    registration = read(folder/'registration.json')
    for name, expected in registration['bindings_sha256'].items(): record('registered:'+name, sha(ROOT/name)==expected)
    record('training_index_binding', sha(run/'training_artifact_hashes.json')==summary['training_artifact_index_sha256'])
    for arm, bindings in summary['checkpoint_files_sha256'].items():
        for name, expected in bindings.items(): record('checkpoint:'+name, sha(ROOT/name)==expected)
    candidates = archive(parent/'candidates.npz'); indices = candidates['original_indices'].tolist()
    head = candidates['all_head_actions']; full = np.concatenate((head, np.zeros((128, 5, 2))), axis=1)
    groups = []
    for index in indices:
        existing = next((g for g in groups if np.array_equal(full[index], full[g[0]])), None)
        if existing is None: groups.append([index])
        else: existing.append(index)
    record('sixteen_indices_ten_distinct_actions', len(indices)==16 and len(groups)==10)
    record('summary_groups_exact', groups==summary['exact_action_groups'])
    actual, actual_states, raw_ready = {}, {}, {}
    for i in indices:
        base = parent/f'actual/candidate-{i:03d}'
        actual[i] = read(base/'repeat-0/result.json')
        for repeat in range(1, 3):
            repeated = read(base/f'repeat-{repeat}/result.json')
            record(f'actual_repeat:{i}:{repeat}', all(repeated[k]==actual[i][k] for k in
                ('actual_J_H10', 'physical_valid', 'actual_task_rank', 'raw_task')))
        raw = archive(base/'repeat-0/raw_execution.npz')
        same(f'actual_command:{i}', raw['action_applied'], full[i])
        truth = np.vstack((source['state'], raw['state'])); actual_states[i] = truth
        costs, _ = legacy_cost(truth[None], full[i:i+1], previous, task, config)
        proxy = endpoint_rank(truth[None], costs, task, status)
        same(f'actual_J:{i}', costs[0], actual[i]['actual_J_H10'])
        same(f'actual_task_rank:{i}', proxy['rank'][0], actual[i]['actual_task_rank'])
        raw_ready[i], interval, held = physical_ready(raw, task, status)
        record(f'actual_raw_ready:{i}', raw_ready[i]==(actual[i]['raw_task']['status']['phase']=='ready'))
        record(f'actual_ready_interval:{i}', interval==actual[i]['raw_task']['first_ready_interval'])
        same(f'actual_terminal_hold:{i}', held, actual[i]['raw_task']['status']['held_time_s'])
    for group in groups:
        for i in group[1:]: record(f'duplicate_exact_states:{i}', np.array_equal(actual_states[i], actual_states[group[0]]))
    compact = {}; local_compact = []
    for arm in ARMS:
        pred = archive(folder/arm/'predictions.npz'); local = archive(folder/arm/'local_predictions.npz')
        record(arm+':all128valid', pred['valid'].shape==(128,) and pred['valid'].all())
        record(arm+':source_state_all128', np.array_equal(pred['states'][:, 0], np.broadcast_to(source['state'], (128, 10))))
        same(arm+':head_actions', pred['head_actions'], head); same(arm+':full_actions', pred['full_actions'], full)
        J10, components = legacy_cost(pred['states'], full, previous, task, config)
        J5, components5 = legacy_cost(pred['states'][:, :6], head, previous, task, config)
        same(arm+':J_H10', pred['J_H10'], J10); same(arm+':J_H5', pred['J_H5'], J5)
        for key, value in components.items(): same(arm+':J10_'+key, pred['J10_'+key], value)
        for key, value in components5.items(): same(arm+':J5_'+key, pred['J5_'+key], value)
        task_forecast = endpoint_rank(pred['states'], J10, task, status)
        for name, key in [('task_ready','ready'),('task_first_ready_step','first'),('task_rank','rank'),
                          ('task_ratios','ratios'),('task_held_time_s','held')]: same(arm+':'+name, pred[name], task_forecast[key])
        chosen = {'J_H5': int(np.argmin(pred['J_H5'])), 'J_H10_stop': int(np.argmin(pred['J_H10'])),
                  'TASK_H10_stop': min(range(128), key=lambda i:(*pred['task_rank'][i], i))}
        result = summary['arms'][arm]
        for choice in result['full_N128_choices']:
            selected = chosen[choice['arm']]
            record(arm+':selection:'+choice['arm'], selected==choice['selected_index'])
            expected_status = 'MEASURED' if selected in indices else 'UNKNOWN'
            record(arm+':actual_scope:'+choice['arm'], choice['actual']['status']==expected_status)
            if selected in indices:
                record(arm+':selected_raw_ready:'+choice['arm'], choice['actual']['actual_raw_substep_ready']==raw_ready[selected])
        for name, selected_indices in [('sixteen_original_indices', indices), ('ten_unique_action_sequences', [g[0] for g in groups])]:
            c = result['comparisons'][name]; jc, tc = c['J_H10_stop'], c['TASK_H10_stop']
            record(arm+':indices:'+name, c['original_indices']==selected_indices)
            ps = min(selected_indices, key=lambda i:(pred['J_H10'][i],i))
            best = min(selected_indices, key=lambda i:(actual[i]['actual_J_H10'],i))
            # Historical compare_candidates uses a stable scaled mean of three exact repeats.
            means = {i:float(actual[i]['actual_J_H10']) for i in selected_indices}
            same(arm+':J_selection_loss:'+name, jc['selection_loss'], means[ps]-means[best])
            record(arm+':J_selected:'+name, jc['selected_original_index']==ps)
            record(arm+':J_actual_best:'+name, jc['actual_best_original_index']==best)
            predicted_order = sorted(selected_indices, key=lambda i:(*pred['task_rank'][i],i))
            actual_order = sorted(selected_indices, key=lambda i:(*actual[i]['actual_task_rank'],i))
            record(arm+':task_predicted_order:'+name, tc['predicted_order']==predicted_order)
            record(arm+':task_actual_order:'+name, tc['actual_order']==actual_order)
            selected_task = predicted_order[0]; best_task = actual_order[0]
            gap = sum(tuple(actual[i]['actual_task_rank'])<tuple(actual[selected_task]['actual_task_rank']) for i in selected_indices)
            record(arm+':task_rank_gap:'+name, tc['selection_loss']['lexicographic_rank_gap']==gap)
            same(arm+':task_vector_regret:'+name, tc['selection_loss']['rank_component_difference'],
                 np.asarray(actual[selected_task]['actual_task_rank'])-np.asarray(actual[best_task]['actual_task_rank']))
            for label, actual_boolean in [('handoff_against_raw_substeps',raw_ready),
                ('handoff_against_actual_endpoint_surrogate',{i:actual[i]['actual_endpoint_proxy_ready'] for i in selected_indices})]:
                counts = Counter(('T' if bool(pred['task_ready'][i])==bool(actual_boolean[i]) else 'F')
                                 +('P' if pred['task_ready'][i] else 'N') for i in selected_indices)
                for key in ('TP','FP','TN','FN'): record(f'{arm}:{name}:{label}:{key}', c[label][key]==counts[key])
                record(f'{arm}:{name}:{label}:denominator', c[label]['total']==c[label]['classified_denominator']==len(selected_indices))
            counts = Counter()
            for pair in tc['pairwise']['records']:
                a,b = pair['original_indices']; pa,pb = tuple(pred['task_rank'][a]),tuple(pred['task_rank'][b])
                aa,ab = tuple(actual[a]['actual_task_rank']),tuple(actual[b]['actual_task_rank'])
                po = -1 if pa<pb else 1 if pa>pb else 0
                ao = -1 if aa<ab else 1 if aa>ab else 0
                label = 'actual_exact_tie' if not ao else 'predicted_tie' if not po else 'concordant' if ao==po else 'discordant'
                counts[label]+=1
                record(f'{arm}:{name}:task_pair:{a}:{b}', pair['predicted_order']==po and pair['actual_exact_order']==ao and pair['classification']==label)
            record(f'{arm}:{name}:task_pair_denominator', tc['pairwise']['total_pairs']==len(selected_indices)*(len(selected_indices)-1)//2)
            for key in ('concordant','discordant','predicted_tie','actual_exact_tie'): record(f'{arm}:{name}:task_pair_count:{key}', tc['pairwise'][key]==counts[key])
            jcounts = Counter()
            for pair in jc['pairwise']['records']:
                a,b = (selected_indices[n] for n in pair['indices'])
                epsilon_a,epsilon_b = 1e-6+1e-5*abs(means[a]),1e-6+1e-5*abs(means[b])
                ao = -1 if means[a]+epsilon_a<means[b]-epsilon_b else 1 if means[b]+epsilon_b<means[a]-epsilon_a else 0
                po = -1 if pred['J_H10'][a]<pred['J_H10'][b] else 1 if pred['J_H10'][a]>pred['J_H10'][b] else 0
                label = 'actual_inconclusive' if not ao else 'predicted_tie' if not po else 'concordant' if ao==po else 'discordant'
                jcounts[label]+=1; record(f'{arm}:{name}:J_pair:{a}:{b}',pair['classification']==label)
            for key in ('concordant','discordant','predicted_tie','actual_inconclusive'): record(f'{arm}:{name}:J_pair_count:{key}',jc['pairwise'][key]==jcounts[key])
        for offset,candidate in enumerate((11,77)):
            truth = actual_states[candidate]; subset = slice(10*offset,10*(offset+1))
            stored = next(r for r in summary['local_diagnostics'] if r['arm']==arm and r['candidate']==candidate)
            record(f'{arm}:{candidate}:local_actual_states',np.array_equal(local['actual_states'][offset],truth))
            same(f'{arm}:{candidate}:teacher_inputs',local['teacher_states'][subset,0],truth[:-1])
            same(f'{arm}:{candidate}:stop_start',local['stop_states'][offset,0],truth[5])
            phases = {'teacher_K1':(truth[1:],local['teacher_states'][subset,1]),
                'teacher_head_K1':(truth[1:6],local['teacher_states'][subset,1][:5]),
                'teacher_zero_suffix_K1':(truth[6:],local['teacher_states'][subset,1][5:]),
                'head_H5':(truth[1:6],pred['states'][candidate,1:6]),
                'full_H10':(truth[1:],pred['states'][candidate,1:]),
                'stop_K5_from_actual_state5':(truth[6:],local['stop_states'][offset,1:])}
            for phase,(true,predicted) in phases.items():
                metrics=errors(true,predicted); saved=stored[phase]
                record(f'{arm}:{candidate}:{phase}:denominator', saved['total']==saved['valid']==len(true) and saved['invalid']==0)
                for key,v in metrics.items():
                    same(f'{arm}:{candidate}:{phase}:{key}:rows',[r[key] for r in saved['rows']],v)
                    same(f'{arm}:{candidate}:{phase}:{key}:mean',saved['all_case_mean'][key],v.mean())
            local_compact.append({'arm':arm,'candidate':candidate,
                'teacher_head_mean_mm':{k:stored['teacher_head_K1']['all_case_mean'][k] for k in ('object_xy_mm','pusher_xy_mm')},
                'H10_endpoint_mm':{k:stored['full_H10']['rows'][-1][k] for k in ('object_xy_mm','pusher_xy_mm')},
                'actual_state5_stop_endpoint_mm':{k:stored['stop_K5_from_actual_state5']['rows'][-1][k] for k in ('object_xy_mm','pusher_xy_mm')}})
        compact[arm]={'full_choices':{c['arm']:{'index':c['selected_index'],'actual_status':c['actual']['status']} for c in result['full_N128_choices']},
            'unique_action_confusion':result['comparisons']['ten_unique_action_sequences']['handoff_against_raw_substeps'],
            'unique_action_J_selection_loss':result['comparisons']['ten_unique_action_sequences']['J_H10_stop']['selection_loss'],
            'unique_action_TASK_rank_gap':result['comparisons']['ten_unique_action_sequences']['TASK_H10_stop']['selection_loss']['lexicographic_rank_gap']}
    original=archive(parent/'initial_models/SK-11/predictions.npz'); reproduced=archive(folder/'R/predictions.npz')
    for key in ('states','valid','raw_deltas','J_H10','task_rank'): record('R_exact_oldSK:'+key,np.array_equal(original[key],reproduced[key],equal_nan=True))
    record('explicit_no_new_execution',summary['physical_intervals']==summary['learning_updates']==0 and not any(summary['future_engine_calls'].values())
           and summary['new_closed_loop_execution'] is False and summary['final_test_used'] is False)
    failed=[c for c in checks if not c['pass']]
    result={'schema':'p4-data-retrain-independent-candidate-review-v1','status':'PASS' if not failed else 'FAIL',
        'run':str(run.relative_to(ROOT)),'reviewed_summary_sha256':sha(folder/'summary.json'),
        'reviewed_artifact_index_sha256':sha(folder/'artifact_hashes.json'),'reviewer_source_sha256':sha(Path(__file__)),
        'checks_total':len(checks),'failed_checks':failed,'checks':checks,
        'model_forwards':0,'physics_steps':0,'learning_updates':0,
        'audit_arithmetic_tolerance':{'absolute':1e-12,'relative':1e-12,'scope':'Independent recomputation roundoff only; task/selection thresholds unchanged.'},
        'actual_ready_unique_actions':[g[0] for g in groups if raw_ready[g[0]]],
        'duplicate_groups':groups,'arm_summary':compact,'local_summary':local_compact,
        'scope':'Independent arithmetic over saved raw actual states and stored predictions. No prediction regeneration, new labels, or independent final test.'}
    report.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'checks':len(checks),'failed':failed},ensure_ascii=False))
    return not failed


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();raise SystemExit(0 if check(args.run.resolve(),args.report.resolve()) else 1)
