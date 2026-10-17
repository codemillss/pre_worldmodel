"""Register and collect the frozen 24/12 P4 data families, without models.

The immutable registration precedes even the environment constructor. Raw
warmups are retained separately from core-only arm/window references. The
preflight runner's exception preservation is reused, never its split/budget.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import time
import tomllib
import traceback

import numpy as np

from push_align.config import ROOT, CONFIG_PATH
from push_align.data import (digest, iter_dataset, make_contract, EpisodeRecorder,
                             write_dataset, validate_family_splits, Episode)
from push_align.data_intervention_main import build_main_plan
from push_align.environment import InitialSpec, PlanarPushEnv
from p4_data_preflight import (Runner, PreflightFailure, save, tree_bytes,
                               exact_paths, head_exact, measurements)

PREFLIGHT = ROOT/'runs/p4-data-preflight-20261013-01'
MAIN_CAP = 9 * 1024**3 // 4
FUTURE_RESERVE = 320 * 1024**2
EPISODE_RESERVE = 32 * 1024**2


def host_resources():
    host = {'machine': platform.machine(), 'platform': platform.platform(),
            'python': platform.python_version(), 'logical_cpus': os.cpu_count(),
            'physical_memory_bytes': None, 'physical_memory_status': 'UNAVAILABLE'}
    if platform.system() == 'Darwin':
        result = subprocess.run(['sysctl', '-n', 'hw.memsize'], text=True, capture_output=True)
        if result.returncode == 0 and result.stdout.strip().isdigit():
            host.update(physical_memory_bytes=int(result.stdout), physical_memory_status='READ')
        else:
            host['physical_memory_status'] = 'UNAVAILABLE_SANDBOX_SYSCTL'
    return host


def measured_y_pass(plan, vy, design):
    c = design['collection']
    return bool(np.isfinite(vy) and (vy >= c['measured_high_Y_gate_m_s']
        if plan['y_condition'] == 'Y1' else vy <= c['measured_low_Y_ceiling_m_s']))


def window_pass(plan, m):
    return (m['transition_count'] == m['valid_transitions'] == 20
            and m['valid_K5_windows'] == 16 and m['zero_action_count'] == 5
            and m['zero_occurrences_in_all_K5'] == 25
            and m['all_zero_K5_starts'] == ([10] if plan['stop_condition'] == 'S1' else []))


def contact_coverage(episodes):
    rows = []
    for split, minimum in [('train', 3), ('development', 2)]:
        for y in ['Y0', 'Y1']:
            selected = [r for r in episodes if r['role'] == 'core'
                        and r['semantic_split'] == split and r['Y'] == y]
            contact = sorted({r['family_id'] for r in selected
                              if r['measurements']['active_contact_transitions'] > 0})
            noncontact = sorted({r['family_id'] for r in selected
                                 if r['measurements']['noncontact_transitions'] > 0})
            rows.append({'split': split, 'Y': y, 'minimum_each': minimum,
                         'contact_families': contact, 'noncontact_families': noncontact,
                         'pass': len(contact) >= minimum and len(noncontact) >= minimum})
    return rows


def bind_slots(plans, slots, old_rows):
    if old_rows.shape != (2960, 5):
        raise ValueError('historical window rows must have shape (2960,5)')
    if any(type(x['slot']) is not int or not 0 <= x['slot'] < 2960
           or x['occurrences'] != 21 for x in slots['entries']):
        raise ValueError('replacement slots require valid historical IDs and 21 occurrences')
    lookup = {(p['family_rank'], 'D'+p['y_condition'][1]+p['stop_condition'][1]): p
              for p in plans if p['semantic_split'] == 'train'}
    entries = []
    for row in slots['entries']:
        rank = row['family_rank']
        entry = dict(row, family_id=lookup[rank, 'D00']['family_id'],
                     arm_episodes={arm: lookup[rank, arm]['episode_id']
                                   for arm in ['D00', 'D10', 'D01', 'D11']},
                     replaced_original_prepared_rows=old_rows[row['slot']].tolist())
        entries.append(entry)
    chosen = {x['slot'] for x in entries}
    if len(entries) != 384 or len(chosen) != 384:
        raise ValueError('exactly 384 unique replacement slots required')
    if [(x['family_rank'], x['window_start']) for x in entries] != [(f, w) for f in range(24) for w in range(16)]:
        raise ValueError('replacement ordering must be family rank then window start')
    return {'schema': 'p4-data-main-slot-bindings-v1', 'status': 'REGISTERED_NOT_TRAINED',
            'entries': entries, 'unchanged_old_slots': [i for i in range(2960) if i not in chosen],
            'R_arm': 'all original 2960 slots unchanged',
            'scope': 'core-only references; no warmup, preflight, development or replay in replacement slots'}


def resource_plan(out_parent, design):
    summary = json.loads((PREFLIGHT/'summary.json').read_text())
    if summary['status'] != 'PASS':
        raise PreflightFailure('preflight_not_passed')
    sizes = {role: [tree_bytes(PREFLIGHT/r['dataset']) for r in summary['episodes']
                    if r['role'] == role] for role in ['warmup', 'core']}
    used = tree_bytes(PREFLIGHT)
    full_cap = design['resources']['future_full_experiment_output_limit_GiB']*1024**3
    reserve = design['resources']['minimum_free_reserve_GiB']*1024**3
    free = shutil.disk_usage(out_parent).free
    result = {'preflight_bytes': used, 'nominal_role_sizes_bytes': sizes,
              'scaled_nominal_bytes': 12*sum(sum(v) for v in sizes.values()),
              'observed_role_max_projection_bytes': 72*max(sizes['warmup'])+144*max(sizes['core']),
              'projection_is_not_a_proven_upper_bound': True,
              'main_output_cap_bytes': MAIN_CAP, 'future_training_docs_reserve_bytes': FUTURE_RESERVE,
              'full_experiment_output_cap_bytes': full_cap, 'minimum_free_bytes': reserve,
              'episode_emergency_reserve_bytes': EPISODE_RESERVE, 'free_bytes_before': free,
              'cpu_policy': 'serial simulation and serialization; no model/training worker',
              'host': host_resources(),
              'additional_spend_KRW': 0}
    if used+MAIN_CAP+FUTURE_RESERVE > full_cap or free < reserve+MAIN_CAP:
        raise PreflightFailure('resource_budget_not_feasible')
    return result


class MainRunner(Runner):
    def resources(self):
        used = tree_bytes(self.out)
        free = shutil.disk_usage(self.out).free
        reserve = self.design['resources']['minimum_free_reserve_GiB']*1024**3
        if used+EPISODE_RESERVE > MAIN_CAP or free < reserve+EPISODE_RESERVE:
            raise PreflightFailure('main_output_cap_or_free_reserve')
        return {'output_bytes': used, 'free_bytes': free}

    def _record(self, env, plan, label, actions, contract, role):
        self.resources()
        core = role == 'core'
        recorder = EpisodeRecorder(env, episode_id=label, episode_index=len(self.episodes),
            family_id=plan['family_id'], family_definition={
                'base_initial_spec': {k: plan['initial_spec'][k] for k in ('object_x', 'object_y', 'object_yaw')},
                'variant_rule': 'All Y/S variants and warmups stay in the registered family split; only core contributes learning/evaluation windows'},
            split=plan['split'], collection_policy={'kind': 'p4-data-main', 'role': role,
                'predeclared_plan': plan, 'semantic_split': plan['semantic_split'],
                'training_eligible': core and plan['training_eligible'],
                'development_evaluation_eligible': core and plan['development_evaluation_eligible'],
                'independent_final_test': False},
            seed=plan['rng_metadata']['seed_sequence_entropy'][0], contract=contract)
        self.current_recorder = recorder
        failure = None
        for action in actions:
            if self.counts['action_calls'] >= 3024:
                failure = 'action_budget'; break
            self.counts['action_calls'] += 1
            self.counts['by_role'][role] = self.counts['by_role'].get(role, 0)+1
            result = recorder.step(action)
            self.counts['physics_substeps'] += len(result.substep_times)
            self.counts['completed_intervals'] += int(abs(result.actual_dt-env.action_dt) <= env.config['validity']['dt_match_tolerance_s'])
            if recorder.invalid or not np.array_equal(result.requested_action, result.applied_action):
                failure = 'physical_geometry_or_clipping'; break
        if not recorder.results:
            raise PreflightFailure(failure or 'empty_episode')
        episode = recorder.finish(failure or 'collection_complete')
        folder = self.out/'episodes'/label
        write_dataset(folder, [episode], purpose='main')
        loaded, = list(iter_dataset(folder, contract))
        self.lineage.append(Episode({}, {k: deepcopy(loaded.metadata[k]) for k in
                            ('episode_id', 'episode_index', 'family_id', 'family_definition', 'initial_record', 'split')}))
        validate_family_splits(self.lineage)
        exact = exact_paths([episode], [loaded])['pass']
        row = {'label': label, 'role': role, 'family_id': plan['family_id'],
               'split': plan['split'], 'semantic_split': plan['semantic_split'],
               'geometry': plan['geometry'], 'Y': plan['y_condition'],
               'S': plan['stop_condition'] if core else 'shared',
               'dataset': str(folder.relative_to(self.out)),
               'manifest_sha256': digest(folder/'manifest.json'), 'roundtrip_exact': exact,
               'measurements': measurements(loaded, env.config),
               'terminal_reason': loaded.metadata['terminal_reason']}
        self.episodes.append(row)
        save(self.out/'progress'/f'{label}.json', row)
        print(json.dumps({'episode': label, 'intervals': loaded.transition_count,
                          'valid': row['measurements']['valid_transitions'],
                          'contact_intervals': row['measurements']['active_contact_transitions'],
                          'total_action_calls': self.counts['action_calls']}), flush=True)
        if failure or not exact:
            raise PreflightFailure(failure or 'serialization_not_exact')
        return loaded


def register(out, design, config, plans):
    resources = resource_plan(out.parent, design)
    review_path = ROOT/'reports/p4_data_intervention_design_review.json'
    review = json.loads(review_path.read_text())
    for key, path in [('design_sha256', 'configs/p4_data_intervention.design.v1.json'),
                      ('report_sha256', 'reports/p4_data_intervention_design.md'),
                      ('slot_design_sha256', 'reports/p4_data_intervention_slots.design.json')]:
        if digest(ROOT/path) != review[key]:
            raise PreflightFailure('changed_reviewed_design:'+path)
    immutable = review['immutable_source_bindings'].copy()
    for p, expected in immutable.items():
        if digest(ROOT/p) != expected:
            raise PreflightFailure('changed_original_source:'+p)
    # Carry the design and preflight identities forward before any simulation.
    pre = json.loads((PREFLIGHT/'registration.json').read_text())
    for p, expected in pre['source_sha256'].items():
        if digest(ROOT/p) != expected:
            raise PreflightFailure('changed_preflight_source:'+p)
    assert not {p['family_id'] for p in plans} & {p['family_id'] for p in pre['plans']}
    extra = ['runs/p3-02-20261009-01/learning_data/prepared.npz',
             'runs/p3-02-20261009-01/learning_data/manifest.json',
             'runs/p4-02-20261012-01/train_windows.npz', 'runs/p4-02-20261012-01/manifest.json',
             'reports/p4_data_intervention_design.md', 'reports/p4_data_intervention_slots.design.json',
             'reports/p4_data_preflight_independent_review.json']
    extra += [str(p.relative_to(ROOT)) for p in (ROOT/design['model']['initial_checkpoint']).rglob('*') if p.is_file()]
    extra += [str((PREFLIGHT/p).relative_to(ROOT)) for p in ['registration.json', 'summary.json', 'artifact_hashes.json']]
    immutable.update({p: digest(ROOT/p) for p in extra})
    assert json.loads((ROOT/'reports/p4_data_preflight_independent_review.json').read_text())['status'] == 'PASS'
    slots = json.loads((ROOT/'reports/p4_data_intervention_slots.design.json').read_text())
    assert slots['source_schedule_sha256'] == digest(ROOT/design['training']['historical_schedule'])
    with np.load(ROOT/'runs/p4-02-20261012-01/train_windows.npz', allow_pickle=False) as a:
        bound = bind_slots(plans, slots, a['window_rows'])
    sources = list(pre['source_sha256'])+['src/push_align/data_intervention_main.py',
               'scripts/p4_data_main.py', 'reports/p4_data_intervention_design_review.json']
    source_hashes = {p: digest(ROOT/p) for p in sources}
    out.mkdir(exist_ok=False)
    (out/'episodes').mkdir(); (out/'progress').mkdir()
    save(out/'slots_bound.json', bound)
    with (out/'initial_conditions.csv').open('x', newline='', encoding='utf-8') as stream:
        fields = ['episode_id', 'family_id', 'semantic_split', 'geometry', 'Y', 'S',
                  'object_x', 'object_y', 'object_yaw', 'pusher_x', 'pusher_y', 'command_vx', 'command_vy']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for p in plans:
            writer.writerow({**{k: p[k] for k in ['episode_id', 'family_id', 'semantic_split', 'geometry']},
                'Y': p['y_condition'], 'S': p['stop_condition'], **p['initial_spec'],
                'command_vx': p['moving_action'][0], 'command_vy': p['moving_action'][1]})
    registration = {'schema': 'p4-data-main-registration-v1', 'status': 'REGISTERED_BEFORE_PHYSICS',
        'registered_at_utc': datetime.now(timezone.utc).isoformat(),
        'preflight_run': str(PREFLIGHT.relative_to(ROOT)),
        'plans': plans, 'source_sha256': source_hashes, 'immutable_source_sha256': immutable,
        'design': design, 'physical_config': config, 'resource_plan': resources,
        'slots_bound_sha256': digest(out/'slots_bound.json'),
        'initial_conditions_csv_sha256': digest(out/'initial_conditions.csv'),
        'expected_action_intervals': 3024, 'expected_physics_substeps': 604800,
        'expected_episodes': 216, 'expected_roles': {'warmup': 144, 'core': 2880},
        'execution': '72 physical warmups; paired S0/S1 from identical snapshot; stop at first failed gate; no resampling',
        'dt_protocol': 'same nominal config and preflight solver-warmstart reset; no main dt/2 or replay extension',
        'scope': 'raw data collection only; warmup excluded; development is not independent final test; no model loading, forwards, training or closed loop'}
    save(out/'registration.json', registration)
    return registration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); out = args.output_dir.resolve()
    design = json.loads((ROOT/'configs/p4_data_intervention.design.v1.json').read_text())
    config = tomllib.loads(CONFIG_PATH.read_text())
    plans = build_main_plan(design, config)
    registration = register(out, design, config, plans)
    registration_hash = digest(out/'registration.json')
    runner = MainRunner(out, design); checks = []; start = time.monotonic(); failure = None
    try:
        runner.resources()
        env = PlanarPushEnv(config); runner.reset_log(env, 'nominal_constructor')
        contract = make_contract(env, CONFIG_PATH, ROOT/'configs/acceptance.v1.toml')
        for i in range(0, len(plans), 2):
            p0, p1 = plans[i:i+2]; prefix = p0['warmup_snapshot_group']
            runner.resources()
            runner.reset_log(env, prefix, InitialSpec(**p0['initial_spec']))
            env.import_dt_comparison_state(env)  # solver warmstart only; never physical velocities
            warm = runner.record(env, p0, prefix+'-warmup', p0['warmup_actions'], contract, 'warmup')
            core_start = env.snapshot(); vy = float(env.observe().state[9])
            ypass = measured_y_pass(p0, vy, design)
            checks.append({'cell': prefix, 'kind': 'measured_Y', 'vy_m_s': vy, 'pass': ypass})
            if not ypass: raise PreflightFailure(prefix+':measured_Y_gate')
            s0 = runner.record(env, p0, p0['episode_id'], p0['core_actions'], contract, 'core')
            env.restore(core_start)
            s1 = runner.record(env, p1, p1['episode_id'], p1['core_actions'], contract, 'core')
            pair = dict(head_exact(s0, s1), cell=prefix, kind='S_pair_first5_exact'); checks.append(pair)
            if not pair['pass']: raise PreflightFailure(prefix+':S_pair_first5')
            for p, row in [(p0, runner.episodes[-2]), (p1, runner.episodes[-1])]:
                ok = window_pass(p, row['measurements'])
                checks.append({'cell': prefix, 'kind': p['stop_condition']+'_window_contract', 'pass': ok})
                if not ok: raise PreflightFailure(prefix+':window_contract')
            save(out/'progress'/f'{prefix}-checks.json', checks[-4:])
            print(json.dumps({'cell_complete': prefix, 'cells_completed': (i+2)//2,
                              'cells_planned': 72, 'measured_Y_m_s': vy}), flush=True)
    except Exception as error:
        failure = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
    coverage = contact_coverage(runner.episodes)
    unchanged = (digest(out/'registration.json') == registration_hash
        and all(digest(ROOT/p) == h for p, h in {**registration['source_sha256'], **registration['immutable_source_sha256']}.items())
        and digest(out/'slots_bound.json') == registration['slots_bound_sha256']
        and digest(out/'initial_conditions.csv') == registration['initial_conditions_csv_sha256'])
    accounting = {'action_calls': runner.counts['action_calls'] == 3024,
        'completed_intervals': runner.counts['completed_intervals'] == 3024,
        'physics_substeps': runner.counts['physics_substeps'] == 604800,
        'episodes': len(runner.episodes) == 216, 'roles': runner.counts['by_role'] == {'warmup': 144, 'core': 2880}}
    resources = {'output_under_cap': tree_bytes(out)+4*1024**2 <= MAIN_CAP,
        'free_reserve': shutil.disk_usage(out).free-4*1024**2 >= design['resources']['minimum_free_reserve_GiB']*1024**3}
    passed = (failure is None and len(checks) == 288 and all(x['pass'] for x in checks)
              and all(x['pass'] for x in coverage) and unchanged and all(accounting.values()) and all(resources.values()))
    completed = {r['label'] for r in runner.episodes if r['role'] == 'core' and r['terminal_reason'] == 'collection_complete'}
    summary = {'schema': 'p4-data-main-result-v1', 'status': 'PASS' if passed else 'FAIL',
        'failure': failure, 'registration_sha256': registration_hash, 'checks': checks, 'coverage': coverage,
        'accounting_checks': accounting, 'resource_checks': resources, 'episodes': runner.episodes,
        'counts': runner.counts, 'settling': runner.settling,
        'settling_substeps_total': sum(r['settling_substeps'] for r in runner.settling),
        'core_completion': [{'episode_id': p['episode_id'], 'status': 'RECORDED' if p['episode_id'] in completed else 'NOT_COMPLETED'} for p in plans],
        'output_bytes_before_summary': tree_bytes(out), 'free_bytes_after': shutil.disk_usage(out).free,
        'wall_seconds': time.monotonic()-start, 'source_unchanged': unchanged,
        'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024),
        'learning_updates': 0, 'model_forwards': 0, 'P5_01': 'HOLD', 'task_success': 'NOT_EVALUATED',
        'independent_review': 'PENDING', 'scope': 'collection gates only; data effects and alignment improvement remain untested'}
    save(out/'summary.json', summary)
    save(out/'artifact_hashes.json', {'sha256': {str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file()}})
    print(json.dumps({'status': summary['status'], 'failure': failure, 'counts': runner.counts,
                      'wall_seconds': summary['wall_seconds']}), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
