"""P2-05 reproducible, model-only candidate/cost/selection evidence.

No physical action is executed here. Stored validation states seed independent
planning probes; their stored future states are never supplied to planning.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
from time import perf_counter
import tomllib
import traceback
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from push_align.checkpoint import load_checkpoint
from push_align.config import ROOT
from push_align.data import digest
from push_align.learning_data import load_learning_data
from push_align.planning import (LearnedPredictor, PlannerContext, TaskSpec, choose,
                                 plan, rollout, score)
from push_align.preparation import REGIME_CODES, SPLIT_CODES
from push_align.representation import FEATURE_FIELDS, encode
from p2_episode_contract_check import source_bundle
from p2_train_pilot import save

DEFAULT_LEARNING = ROOT/'runs/p2-03-20261007-01/learning_data'
DEFAULT_TRAINING = ROOT/'runs/p2-04-20261007-02'


def verify_artifacts(folder):
    hashes = json.loads((folder/'artifact_hashes.json').read_text())['sha256']
    if not all(digest(folder/name) == sha for name, sha in hashes.items()):
        raise ValueError(f'Prior artifacts changed: {folder.name}')
    return len(hashes)


def controlled_probe(config):
    """Independent exact two-candidate prediction intervention, not learning."""
    c = deepcopy(config)
    for key in c['cost']:
        if key.startswith(('stage_', 'terminal_')):
            c['cost'][key] = 0.
    c['cost']['stage_position'], c['cost']['terminal_position'] = 1., 5.
    c['task']['goal_xy_m'] = [.002, 0.]
    task = TaskSpec.from_config(c)
    initial = np.zeros((2, 10), np.float64)
    initial[:, 6] = -.05
    actions = np.array([[[.01, 0.]], [[.02, 0.]]], np.float64)
    outcomes = []
    for flipped in (False, True):
        def predictor(state, action):
            delta = np.zeros_like(state)
            delta[:, 0] = .003-action[:, 0]*.1 if flipped else action[:, 0]*.1
            return delta
        predictions = rollout(initial, actions, predictor, c)
        costs = score(predictions, actions, task, np.zeros(2, np.float64), c)
        index, action, reason = choose(costs, predictions, actions)
        outcomes.append({'flipped': flipped, 'costs': costs.total.tolist(),
                         'selected_index': index, 'first_action': action.tolist(), 'reason': reason})
    expected = np.array([[.204, 0.], [0., .204]])
    checks = {'hand_calculated_costs': bool(np.allclose([r['costs'] for r in outcomes], expected, atol=1e-12, rtol=0)),
              'prediction_intervention_changes_expected_choice': [r['selected_index'] for r in outcomes] == [1, 0]}
    return {'checks': checks, 'outcomes': outcomes, 'expected_costs': expected.tolist(),
            'scope': 'synthetic intervention checks prediction-to-choice wiring; no trained-model performance claim'}


def select_cases(prepared, settings):
    a = prepared.arrays
    cases = []
    for regime in settings['trace_regime_order']:
        rows = np.flatnonzero((a['split_code'] == SPLIT_CODES['validation'])
                              & (a['regime_code'] == REGIME_CODES[regime]))
        rows = sorted(map(int, rows), key=lambda row: (int(a['episode_index'][row]), int(a['transition_index'][row])))
        if len(rows) < settings['trace_cases_per_regime']:
            raise ValueError(f'Insufficient validation rows for the declared probe: {regime}')
        for row in rows[:settings['trace_cases_per_regime']]:
            episode = next(e for e in prepared.metadata['episodes'] if e['episode_index'] == int(a['episode_index'][row]))
            cases.append({'case_index': len(cases), 'regime': regime, 'prepared_row': row,
                          'episode_index': int(a['episode_index'][row]), 'episode_id': episode['episode_id'],
                          'family_id': episode['family_id'], 'transition_index': int(a['transition_index'][row]),
                          'candidate_seed': settings['candidate_seed_base']+len(cases)})
    return cases


def support_diagnostic(result, train_features):
    """Marginal bounds only; observations outside are not silently filtered."""
    states = result.predictions.states[:, :-1]
    valid_steps = np.isfinite(states).all(axis=-1)
    # A state that fails initial geometry was never input to the predictor.
    for index, failed in enumerate(result.predictions.failure_steps):
        if failed >= 0:
            valid_steps[index, max(0, int(failed)):] = False
    inputs = encode(states[valid_steps].copy(), result.candidates.actions[valid_steps].copy())
    lower, upper = train_features.min(axis=0), train_features.max(axis=0)
    outside = (inputs < lower-1e-12) | (inputs > upper+1e-12)
    selected_outside = None
    if result.selected_index is not None:
        index = result.selected_index
        selected = encode(states[index].copy(), result.candidates.actions[index].copy())
        selected_outside = int(((selected < lower-1e-12) | (selected > upper+1e-12)).any(axis=1).sum())
    return {'train_original_transitions': len(train_features), 'evaluated_finite_input_pairs': len(inputs),
            'outside_any_train_marginal_bound': int(outside.any(axis=1).sum()),
            'outside_by_feature': {name: int(outside[:, i].sum()) for i, name in enumerate(FEATURE_FIELDS)},
            'selected_candidate_outside_steps': selected_outside,
            'scope': 'per-feature empirical train ranges; neither joint coverage nor an OOD guarantee; diagnostic only'}


def write_trace(folder, case, result, model_calls, model_rows, support):
    folder.mkdir(parents=True, exist_ok=False)
    eligible = result.predictions.valid & np.isfinite(result.costs.total)
    candidate_records = []
    for index in range(len(result.candidates.actions)):
        prediction_reason = result.predictions.reasons[index]
        reason = prediction_reason if prediction_reason else ('' if eligible[index] else 'nonfinite_cost')
        candidate_records.append({'index': index, 'source': result.candidates.sources[index],
                                  'prediction_valid': bool(result.predictions.valid[index]),
                                  'prediction_reason': prediction_reason,
                                  'selection_eligible': bool(eligible[index]), 'invalid': not bool(eligible[index]),
                                  'reason': reason,
                                  'cost': float(result.costs.total[index]) if eligible[index] else None,
                                  'failure_step': int(result.predictions.failure_steps[index])})
    save(folder/'selection.json', {'case': case, 'model_calls': model_calls, 'model_input_rows': model_rows,
                                 'diagnostics': result.diagnostics, 'first_action': result.action.tolist(),
                                 'selected_index': result.selected_index, 'candidates': candidate_records,
                                 'support': support, 'physics_executed': False})
    with (folder/'trace.npz').open('xb') as file:
        np.savez_compressed(file, candidate_actions=result.candidates.actions,
                            predicted_states=result.predictions.states, raw_deltas=result.predictions.raw_deltas,
                            prediction_valid=result.predictions.valid, selection_eligible=eligible,
                            failure_steps=result.predictions.failure_steps, predicted_costs=result.costs.total,
                            chosen_first_action=result.action,
                            **{f'cost_{key}': value for key, value in result.costs.components.items()})
    return {'case': case, 'selected_index': result.selected_index, 'first_action': result.action.tolist(),
            'valid_predictions': int(result.predictions.valid.sum()), 'eligible_candidates': int(eligible.sum()),
            'invalid_reason_counts': dict(Counter(r['reason'] for r in candidate_records if r['invalid'])),
            'selected_cost': None if result.selected_index is None else float(result.costs.total[result.selected_index]),
            'model_calls': model_calls, 'model_input_rows': model_rows,
            'wall_time_s': result.diagnostics['wall_time_s'], 'support': support}


def run(args, out):
    started = perf_counter()
    settings = tomllib.loads(args.planning_config.read_text())
    required = {'device': 'cpu', 'torch_threads': 1, 'deterministic_algorithms': True,
                'candidate_count': 256, 'horizon': 10, 'block_lengths': [5, 5],
                'candidate_speed_norm_max_m_s': .03, 'geometry_proposal_enabled': False,
                'candidate_seed_base': 2711, 'trace_cases_per_regime': 2,
                'trace_regime_order': ['noncontact', 'sustained_contact', 'transition'], 'repeat_case_index': 0,
                'future_engine_guards': ['mj_step', 'mj_step1', 'mj_step2', 'mj_forward', 'mj_inverse'],
                'stage_state_convention': 'post-action predicted states 1..H',
                'cost_time_convention': 'action_dt_weighted_stage_plus_terminal'}
    if any(type(settings.get(key)) is not type(value) or settings[key] != value for key, value in required.items()):
        raise ValueError('Runner requires the predeclared P2-05 connection settings')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    learning, training = args.learning_data.resolve(), args.training_run.resolve()
    before_prior = {str(p): verify_artifacts(p) for p in (learning.parent, training)}
    selected = json.loads((training/'selection.json').read_text())
    checkpoint = training/selected['selected_checkpoint']
    meta = json.loads((checkpoint/'checkpoint.json').read_text())
    config = meta['config']
    normalizer_manifest = json.loads((learning/'normalizer/normalizer.json').read_text())['normalizer']
    prepared, normalizer = load_learning_data(learning, normalizer_manifest['schema'],
                                              normalizer_manifest['dataset_identity'], meta['normalizer_hash'])
    provenance = {**meta['training_provenance'], 'learning_manifest_sha256': digest(learning/'manifest.json')}
    bundle = load_checkpoint(checkpoint, normalizer, config, provenance)
    if selected['selected_checkpoint_hash'] != bundle.metadata['checkpoint_hash']:
        raise ValueError('Selected model differs from the registered P2-04 checkpoint')
    frozen_weights = {key: value.clone() for key, value in bundle.model.state_dict().items()}
    planning_config = deepcopy(config)
    planning_config['planner']['command_speed_norm_max_m_s'] = settings['candidate_speed_norm_max_m_s']
    task = TaskSpec.from_config(config)
    cases = select_cases(prepared, settings)
    hashes = source_bundle(out)
    save(out/'manifest.json', {'task_id': 'P2-05', 'created_at': datetime.now().astimezone().isoformat(),
                              'argv': sys.argv, 'executable': sys.executable, 'python': platform.python_version(),
                              'platform': platform.platform(), 'numpy': np.__version__, 'torch': str(torch.__version__),
                              'settings': settings, 'planning_config_sha256': digest(args.planning_config),
                              'original_checkpoint_config': config, 'resolved_planning_config': planning_config,
                              'cases': cases, 'case_count': len(cases), 'source_hashes': hashes,
                              'checkpoint_path': str(checkpoint), 'checkpoint_hash': meta['checkpoint_hash'],
                              'weights_sha256': digest(checkpoint/'weights.pt'), 'normalizer_hash': meta['normalizer_hash'],
                              'learning_manifest_sha256': digest(learning/'manifest.json'), 'prior_artifact_counts': before_prior,
                              'alignment_outcome': 'NOT_EVALUATED', 'candidate_quality': 'NOT_EVALUATED'})
    print(json.dumps({'progress': 'manifest_saved', 'cases': len(cases)}), flush=True)
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=ROOT, capture_output=True, text=True)
    (out/'tests.txt').write_text(tests.stdout+tests.stderr)
    match = re.search(r'Ran (\d+) tests', tests.stdout+tests.stderr)
    test_count = int(match.group(1)) if match else 0
    if tests.returncode or not test_count or 'skipped=' in tests.stderr:
        raise ValueError('Mandatory regression suite did not pass without skips')
    probe = controlled_probe(planning_config)
    save(out/'controlled_probe.json', probe)
    if not all(probe['checks'].values()):
        raise ValueError('Controlled prediction-to-choice intervention failed')
    print(json.dumps({'progress': 'tests_and_controlled_probe_passed', 'tests': test_count}), flush=True)
    predictor = LearnedPredictor(bundle.model, normalizer)
    engine_calls = {name: 0 for name in settings['future_engine_guards']}
    def forbidden(name):
        def invoke(*unused_args, **unused_kwargs):
            engine_calls[name] += 1
            raise AssertionError(f'Future physics forbidden during planning: {name}')
        return invoke
    trace_results, checks = [], {}
    train_features = prepared.arrays['features'][prepared.arrays['split_code'] == SPLIT_CODES['train']]
    first_result = None
    with ExitStack() as guards:
        for name in engine_calls:
            guards.enter_context(patch.object(mujoco, name, side_effect=forbidden(name)))
        for case in cases:
            initial = prepared.arrays['current_state'][case['prepared_row']].copy()
            before_calls, before_rows = predictor.calls, predictor.rows
            result = plan(initial, task, PlannerContext.initial(), np.random.default_rng(case['candidate_seed']),
                          predictor, planning_config)
            model_calls, model_rows = predictor.calls-before_calls, predictor.rows-before_rows
            support = support_diagnostic(result, train_features)
            record = write_trace(out/f"traces/case-{case['case_index']:02d}", case, result, model_calls, model_rows, support)
            trace_results.append(record)
            eligible = result.predictions.valid & np.isfinite(result.costs.total)
            expected = int(np.flatnonzero(eligible)[np.argmin(result.costs.total[eligible])]) if eligible.any() else None
            checks[f"case_{case['case_index']}_selection_and_first_action"] = bool(
                result.selected_index == expected and (np.array_equal(result.action, np.zeros(2))
                if expected is None else np.array_equal(result.action, result.candidates.actions[expected, 0])))
            checks[f"case_{case['case_index']}_actual_model_invoked"] = model_calls > 0 and model_rows > 0
            if case['case_index'] == settings['repeat_case_index']:
                first_result = result
            print(json.dumps({'progress': 'model_plan_recorded', 'case': case['case_index'],
                              'eligible': int(eligible.sum()), 'selected': result.selected_index}), flush=True)
        repeat_case = cases[settings['repeat_case_index']]
        repeated = plan(prepared.arrays['current_state'][repeat_case['prepared_row']].copy(), task,
                        PlannerContext.initial(), np.random.default_rng(repeat_case['candidate_seed']), predictor, planning_config)
    checks.update(probe['checks'])
    checks.update({'regression_tests_pass': tests.returncode == 0 and test_count > 0,
                   'future_engine_calls_zero': sum(engine_calls.values()) == 0,
                   'candidate_repeat_exact': bool(np.array_equal(first_result.candidates.actions, repeated.candidates.actions)),
                   'prediction_repeat_exact_including_invalid': bool(np.array_equal(first_result.predictions.states, repeated.predictions.states, equal_nan=True)),
                   'cost_repeat_exact': bool(np.array_equal(first_result.costs.total, repeated.costs.total)),
                   'choice_repeat_exact': first_result.selected_index == repeated.selected_index and bool(np.array_equal(first_result.action, repeated.action)),
                   'model_weights_unchanged': all(torch.equal(value, bundle.model.state_dict()[key]) for key, value in frozen_weights.items()),
                   'checkpoint_config_unchanged': config == meta['config'] and 'command_speed_norm_max_m_s' not in config['planner'],
                   'source_frozen': all(digest(ROOT/name) == sha for name, sha in hashes.items()),
                   'prior_artifacts_unchanged': {str(p): verify_artifacts(p) for p in (learning.parent, training)} == before_prior})
    save(out/'future_engine_guard.json', {'calls': engine_calls, 'total_calls': sum(engine_calls.values()),
                                         'scope': 'six separately initialized stored-state planning probes and one repeat; possibly shared families; no physics execution'})
    times = np.asarray([row['wall_time_s'] for row in trace_results])
    resources = {'main_peak_rss_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024),
                 'total_wall_time_s': perf_counter()-started, 'plan_calls_measured': len(times),
                 'planning_wall_time_s_mean': float(times.mean()), 'planning_wall_time_s_max': float(times.max()),
                 'planning_wall_time_s_p95': float(np.percentile(times, 95)), 'action_dt_s': config['physics']['action_dt_s'],
                 'scope': 'local CPU one thread; six separately initialized stored-state probes; subprocess test memory excluded; no real-time guarantee'}
    save(out/'resources.json', resources)
    save(out/'summary.json', {'task_id': 'P2-05', 'pass': all(checks.values()), 'checks': checks,
                             'tests': {'count': test_count, 'returncode': tests.returncode, 'skipped': 0},
                             'model_identity': {'checkpoint_hash': meta['checkpoint_hash'], 'weights_sha256': digest(checkpoint/'weights.pt'),
                                                'normalizer_hash': meta['normalizer_hash'], 'epoch': meta['epoch'], 'updates': meta['updates']},
                             'trace_case_count': len(trace_results), 'trace_results': trace_results,
                             'actual_model_total_calls_including_repeat': predictor.calls,
                             'actual_model_total_input_rows_including_repeat': predictor.rows,
                             'future_engine_calls': engine_calls, 'resources': resources,
                             'candidate_quality': 'NOT_EVALUATED', 'alignment_outcome': 'NOT_EVALUATED',
                             'closed_loop': 'NOT_RUN', 'user_understanding': 'NOT_ASSESSED'})
    save(out/'artifact_hashes.json', {'sha256': {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    if not all(checks.values()):
        raise ValueError('P2-05 checks failed; see immutable summary')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--learning-data', type=Path, default=DEFAULT_LEARNING)
    parser.add_argument('--training-run', type=Path, default=DEFAULT_TRAINING)
    parser.add_argument('--planning-config', type=Path, default=ROOT/'configs/p2_planning.v1.toml')
    args = parser.parse_args()
    if not re.fullmatch(r'p2-05-[0-9]{8}-[0-9]{2}', args.run_id):
        raise ValueError('Invalid run ID')
    out = ROOT/'runs'/args.run_id
    out.mkdir(parents=True, exist_ok=False)
    try:
        run(args, out)
    except Exception as error:
        save(out/'failure.json', {'error_type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
        raise
    print(json.dumps({'run_id': args.run_id, 'pass': True}), flush=True)


if __name__ == '__main__':
    main()
