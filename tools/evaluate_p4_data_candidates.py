"""Frozen source99 candidate regression for the five data-intervention arms.

All actual futures are archived labels. This program never constructs an
environment, advances physics, selects a checkpoint, or updates a model.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path
import sys
from time import perf_counter
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

from push_align.candidate_diagnostics import compare_candidates
from push_align.closed_loop import ENGINE_GUARDS, model_planning_scope
from push_align.data import digest, _encode_nonfinite
from push_align.normalization import load_normalizer
from push_align.planning import LearnedPredictor, PlannerContext, TaskSpec, rollout
from push_align.representation import state_delta
from push_align.task import TaskStatus
from p4_handoff_cost_compare import forecast, save_forecast
from p4_local_prediction_diagnostic import local_error_decomposition, norms
from p4_train_forks import tree_hash

PARENT = ROOT / 'runs/p4-cost-20261013-02'
ARMS = ('R', 'D00', 'D10', 'D01', 'D11')
CANDIDATES = (11, 77)
ATOL, RTOL = 1e-6, 1e-5


def save(path, value):
    path.write_text(json.dumps(_encode_nonfinite(value), ensure_ascii=False,
                               indent=2, allow_nan=False) + '\n')


def required_source_bindings():
    """Return the exact historical/source files the root must preregister."""
    paths = [ROOT / 'configs/p4_data_intervention.design.v1.json',
             ROOT / 'configs/p4_handoff_cost.v1.json', Path(__file__).resolve(),
             ROOT / 'src/push_align/candidate_diagnostics.py',
             ROOT / 'src/push_align/planning.py',
             ROOT / 'src/push_align/handoff_cost.py',
             ROOT / 'scripts/p4_handoff_cost_compare.py',
             ROOT / 'scripts/p4_local_prediction_diagnostic.py']
    paths += [PARENT / name for name in ('manifest.json', 'input.json',
        'physical_snapshot.json', 'candidates.npz', 'artifact_hashes.json',
        'choices_before_new_physics.json', 'summary.json')]
    paths.append(PARENT / 'initial_models/SK-11/predictions.npz')
    with np.load(PARENT / 'candidates.npz', allow_pickle=False) as z:
        indices = z['original_indices'].tolist()
    for index in indices:
        folder = PARENT / f'actual/candidate-{index:03d}'
        paths.append(folder / 'repeat-0/raw_execution.npz')
        paths += [folder / f'repeat-{repeat}/result.json' for repeat in range(3)]
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def unique_groups(actions, indices):
    """Group exact action sequences, retaining lowest original index first."""
    groups = []
    for index in sorted(map(int, indices)):
        group = next((row for row in groups
                      if np.array_equal(actions[index], actions[row[0]])), None)
        if group is None:
            groups.append([index])
        else:
            group.append(index)
    return groups


def confusion(predicted_ready, eligible, actual_ready, actual_valid):
    """Invalid predictions are unknown, never silently treated as negative."""
    result = dict(TP=0, FP=0, TN=0, FN=0, prediction_invalid=0,
                  actual_invalid=0, total=len(predicted_ready))
    for p, e, a, v in zip(predicted_ready, eligible, actual_ready, actual_valid):
        if not v:
            result['actual_invalid'] += 1
        elif not e:
            result['prediction_invalid'] += 1
        else:
            result[('T' if bool(p) == bool(a) else 'F') + ('P' if p else 'N')] += 1
    result['classified_denominator'] = sum(result[k] for k in ('TP', 'FP', 'TN', 'FN'))
    return result


def rank_order(first, second):
    a, b = tuple(map(float, first)), tuple(map(float, second))
    return -1 if a < b else 1 if a > b else 0


def repeated_lex_order(first_repeats, second_repeats):
    """Exact tuple order only; no cross-unit tolerance or robustness claim."""
    first, second = np.asarray(first_repeats), np.asarray(second_repeats)
    if not (np.array_equal(first, np.broadcast_to(first[0], first.shape))
            and np.array_equal(second, np.broadcast_to(second[0], second.shape))):
        return None
    return rank_order(first[0], second[0])


def task_comparison(indices, predicted_keys, eligible, actual):
    indices = list(map(int, indices))
    stable = {i: all(r['physical_valid'] for r in actual[i]) for i in indices}
    predictions = [i for i in indices if eligible[i]]
    actual_valid = [i for i in indices if stable[i]]
    selected = min(predictions, key=lambda i: (*predicted_keys[i], i)) if predictions else None
    keys = {i: [r['actual_task_rank'] for r in actual[i]] for i in actual_valid}
    best = min(actual_valid, key=lambda i: (*keys[i][0], i)) if actual_valid else None
    common = [i for i in predictions if stable[i]]
    counts = dict(concordant=0, discordant=0, predicted_tie=0, actual_exact_tie=0, actual_repeat_inconsistent=0)
    pairs = []
    for a, b in itertools.combinations(common, 2):
        predicted = rank_order(predicted_keys[a], predicted_keys[b])
        observed = repeated_lex_order(keys[a], keys[b])
        category = ('actual_repeat_inconsistent' if observed is None else 'actual_exact_tie'
                    if observed == 0 else 'predicted_tie' if not predicted else 'concordant'
                    if observed == predicted else 'discordant')
        counts[category] += 1
        pairs.append(dict(original_indices=[a, b], predicted_order=predicted,
                          actual_exact_order=observed, classification=category))
    loss = None
    if selected is not None and stable[selected]:
        selected_key, best_key = np.asarray(keys[selected][0]), np.asarray(keys[best][0])
        difference = selected_key - best_key
        unequal = np.flatnonzero(difference)
        loss = {
            'lexicographic_rank_gap': sum(rank_order(keys[i][0], selected_key) < 0 for i in actual_valid),
            'rank_component_difference': difference.tolist(),
            'first_differing_component': int(unequal[0]) if len(unequal) else None,
            'ordering': ('NUMERICAL_SELECTED_WORSE' if rank_order(selected_key, best_key) > 0
                         else 'TIED_EXACT_BEST' if sum(rank_order(keys[i][0], best_key) == 0 for i in actual_valid) > 1
                         else 'EXACT_BEST'),
            'selected_is_exact_rank_best': rank_order(selected_key, best_key) == 0,
            'scope': 'Lexicographic comparison; component differences are not a summed task cost.'}
    return dict(original_indices=indices, predicted_selected_index=selected,
        actual_best_index=best, selection_loss=loss,
        counts=dict(total=len(indices), predicted_valid=len(predictions), actual_stable_valid=len(actual_valid),
                    common_valid=len(common)),
        predicted_order=sorted(predictions, key=lambda i: (*predicted_keys[i], i)),
        actual_order=sorted(actual_valid, key=lambda i: (*keys[i][0], i)),
        pairwise=dict(total_pairs=len(indices)*(len(indices)-1)//2, common_valid_pairs=len(pairs),
                      excluded_noncommon_pairs=len(indices)*(len(indices)-1)//2-len(pairs),
                      **counts, records=pairs),
        rank_comparison_scope='Exact fixed tuple ordering; three-repeat equality is checked separately. Numerical robustness is NOT_EVALUATED.',
        numerical_robustness='NOT_EVALUATED')


def selected_actual(index, actual):
    if index is None:
        return dict(status='NO_VALID_PREDICTION', original_index=None)
    if index not in actual:
        return dict(status='UNKNOWN', original_index=index,
                    reason='Selected original index is outside the frozen measured subset; no new physics.')
    row = actual[index][0]
    return dict(status='MEASURED' if row['physical_valid'] else 'ACTUAL_INVALID',
        original_index=index, actual_J_H10=row['actual_J_H10'],
        actual_endpoint_surrogate_ready=row['actual_endpoint_proxy_ready'],
        actual_raw_substep_ready=row['raw_task']['status']['phase'] == 'ready',
        actual_task_rank=row['actual_task_rank'])


def error_summary(actual, predicted, valid):
    """Retain full denominators; never compute an all-case mean on survivors."""
    actual, predicted, valid = np.asarray(actual), np.asarray(predicted), np.asarray(valid, bool)
    errors = np.full(actual.shape, np.nan, np.float64)
    available = valid & np.isfinite(predicted).all(axis=-1)
    if available.any():
        errors[available] = state_delta(actual[available], predicted[available])
    measures = norms(errors)
    return dict(total=len(valid), valid=int(available.sum()), invalid=int((~available).sum()),
        rows=[{k: float(v[i]) if available[i] else None for k, v in measures.items()}
              for i in range(len(valid))],
        all_case_mean={k: float(np.mean(v)) if available.all() else None for k, v in measures.items()},
        conditional_valid_mean={k: float(np.mean(np.asarray(v)[available])) if available.any() else None
                                for k, v in measures.items()},
        complete=bool(available.all()))


def load_historical():
    manifest = json.loads((PARENT / 'manifest.json').read_text())
    initial = json.loads((PARENT / 'input.json').read_text())
    with np.load(PARENT / 'candidates.npz', allow_pickle=False) as z:
        head, indices = z['all_head_actions'].copy(), z['original_indices'].tolist()
        full = np.concatenate((head, np.zeros((128, 5, 2), np.float64)), axis=1)
        if not np.array_equal(z['subset_full_actions'], full[indices]):
            raise ValueError('Historical candidate subset does not match N128 commands')
    actual, states = {}, {}
    for index in indices:
        folder = PARENT / f'actual/candidate-{index:03d}'
        actual[index] = [json.loads((folder / f'repeat-{r}/result.json').read_text()) for r in range(3)]
        for r, row in enumerate(actual[index]):
            if row['original_index'] != index or row['repeat'] != r:
                raise ValueError('Historical result index differs')
            for key in ('physical_valid', 'actual_J_H10', 'actual_task_rank',
                        'actual_endpoint_proxy_ready', 'raw_task'):
                if row[key] != actual[index][0][key]:
                    raise ValueError('Historical repeat stability no longer matches the frozen evidence')
        with np.load(folder / 'repeat-0/raw_execution.npz', allow_pickle=False) as z:
            if z['state'].shape != (10, 10) or not np.all(np.abs(z['actual_dt']-.1) < 1e-9):
                raise ValueError('Historical recorded intervals differ')
            if not (np.array_equal(z['action_applied'], full[index])
                    and np.array_equal(z['action_requested'], full[index])):
                raise ValueError('Historical candidate actions differ')
            states[index] = np.vstack((np.asarray(initial['state'], np.float64), z['state']))
    groups = unique_groups(full, indices)
    if len(indices) != 16 or len(groups) != 10 or not set(CANDIDATES) <= set(indices):
        raise ValueError('Frozen sixteen-index, ten-sequence subset differs')
    for group in groups:
        for index in group[1:]:
            if not np.array_equal(states[index], states[group[0]]):
                raise ValueError('Identical actions have differing historical trajectories')
    task_data = deepcopy(initial['task']); task_data['goal'] = np.asarray(task_data['goal'], np.float64)
    context = initial['context']
    status = deepcopy(initial['status']); status['violations'] = tuple(status['violations'])
    return manifest, initial, head, full, indices, groups, actual, states, TaskSpec(**task_data), \
        PlannerContext(np.asarray(context['previous_applied_action'], np.float64),
                       np.asarray(context['optimal_actions'], np.float64), context['control_step']), TaskStatus(**status)


def evaluate(training_run, registration_path, registration_sha, output):
    from push_align.data_intervention_training import load_intervention_checkpoint
    began = perf_counter()
    registration = json.loads(registration_path.read_text())
    if digest(registration_path) != registration_sha:
        raise ValueError('Evaluation registration hash differs')
    if str(training_run.relative_to(ROOT)) != registration['training_run']:
        raise ValueError('Training run differs from registration')
    expected_bindings = registration['bindings_sha256']
    for path, sha in required_source_bindings().items():
        if expected_bindings.get(path) != sha:
            raise ValueError('Missing or altered registered source: ' + path)
    if output.exists():
        raise ValueError('Never overwrite a candidate evaluation')
    output.mkdir(parents=True)
    save(output / 'registration.json', registration)
    training_manifest = json.loads((training_run / 'manifest.json').read_text())
    training_manifest_sha = digest(training_run / 'manifest.json')
    if json.loads((training_run / 'summary.json').read_text())['status'] != 'PASS':
        raise ValueError('Training and R reproduction must pass before candidate evaluation')
    training_index_path = training_run / 'training_artifact_hashes.json'
    training_index_sha = digest(training_index_path)
    training_index = json.loads(training_index_path.read_text())['sha256']
    for path, sha in training_index.items():
        if digest(training_run / path) != sha:
            raise ValueError('Training artifact changed: ' + path)
    historical, initial, head, full, indices, groups, actual, states, task, context, status = load_historical()
    physical = training_manifest['physical_config']
    if physical != historical['physical']:
        raise ValueError('Training and frozen candidate physical contracts differ')
    nm_dir = ROOT / 'runs/p3-02-20261009-01/learning_data/normalizer'
    nm = json.loads((nm_dir / 'normalizer.json').read_text())['normalizer']
    normalizer = load_normalizer(nm_dir, nm['schema'], nm['dataset_identity'], nm['normalizer_hash'])
    if normalizer.metadata['normalizer_hash'] != training_manifest['normalizer_hash']:
        raise ValueError('Intervention normalizer differs')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    guards = dict.fromkeys(ENGINE_GUARDS, 0)
    checkpoint_hashes, arm_results, local_results, bridge = {}, {}, [], None
    for arm in ARMS:
        path = training_run / f'arms/{arm}/checkpoints/update-1000'
        bundle = load_intervention_checkpoint(path, normalizer, physical,
            expected_manifest_sha256=training_manifest_sha, expected_arm=arm,
            expected_dataset_hash=training_manifest['training_provenance']['arms'][arm]['data_hash'],
            expected_initial_checkpoint_hash=training_manifest['initial_checkpoint']['checkpoint_hash'])
        if bundle.metadata['additional_updates'] != 1000 or bundle.metadata['total_updates'] != 2300:
            raise ValueError('Only the preregistered final checkpoint may be evaluated')
        checkpoint_hashes[arm] = {str(p.relative_to(ROOT)): digest(p) for p in path.iterdir() if p.is_file()}
        before, rng = tree_hash(bundle.model.state_dict()), torch.get_rng_state().clone()
        predictor = LearnedPredictor(bundle.model, normalizer)
        with model_planning_scope(guards):
            result = forecast(head, np.asarray(initial['state'], np.float64), task, context,
                              status, predictor, historical['head_planner'])
            teacher_states = np.concatenate([states[c][:-1] for c in CANDIDATES])
            teacher_actions = np.concatenate([full[c] for c in CANDIDATES])
            teacher = rollout(teacher_states, teacher_actions[:, None], predictor, historical['head_planner'])
            stop = rollout(np.stack([states[c][5] for c in CANDIDATES]),
                           np.stack([full[c, 5:] for c in CANDIDATES]), predictor, historical['head_planner'])
        save_forecast(output / arm, head, result)
        np.savez_compressed(output / arm / 'local_predictions.npz',
            teacher_states=teacher.states, teacher_valid=teacher.valid, teacher_deltas=teacher.raw_deltas,
            teacher_failure_steps=teacher.failure_steps, teacher_reasons=np.asarray(teacher.reasons),
            stop_states=stop.states, stop_valid=stop.valid, stop_failure_steps=stop.failure_steps,
            stop_reasons=np.asarray(stop.reasons), actual_states=np.stack([states[c] for c in CANDIDATES]))
        for offset, candidate in enumerate(CANDIDATES):
            sl = slice(offset*10, (offset+1)*10)
            pred = result['predictions']; fail = int(pred.failure_steps[candidate])
            free_valid = np.asarray([fail < 0 or fail > k for k in range(1, 11)])
            stop_fail = int(stop.failure_steps[offset])
            stop_valid = np.asarray([stop_fail < 0 or stop_fail > k for k in range(1, 6)])
            free_error = error_summary(states[candidate][1:], pred.states[candidate, 1:], free_valid)
            local = dict(arm=arm, candidate=candidate,
                teacher_K1=error_summary(states[candidate][1:], teacher.states[sl, 1], teacher.valid[sl]),
                teacher_head_K1=error_summary(states[candidate][1:6], teacher.states[sl, 1][:5], teacher.valid[sl][:5]),
                teacher_zero_suffix_K1=error_summary(states[candidate][6:], teacher.states[sl, 1][5:], teacher.valid[sl][5:]),
                head_H5=error_summary(states[candidate][1:6], pred.states[candidate, 1:6], free_valid[:5]),
                full_H10=free_error,
                stop_K5_from_actual_state5=error_summary(states[candidate][6:], stop.states[offset, 1:], stop_valid),
                predicted_full_valid=bool(pred.valid[candidate]), prediction_failure_step=fail,
                prediction_reason=pred.reasons[candidate],
                stop_scope='Restart at recorded actual H5 boundary, then five zeros. Offline diagnostic only.')
            if pred.valid[candidate] and teacher.valid[sl].all():
                decomposition = local_error_decomposition(states[candidate], pred.states[candidate],
                    pred.raw_deltas[candidate], teacher.raw_deltas[sl, 0])
                local['decomposition_closure_maximum'] = float(np.max(np.abs(decomposition['closure_residual'])))
                local['local_error_norms'] = norms(decomposition['local_error'])
                local['recursive_input_effect_norms'] = norms(decomposition['recursive_input_effect'])
            else:
                local['decomposition_status'] = 'INCOMPLETE_PREDICTION_RETAINED'
            local_results.append(local)
        comparisons = {}
        for name, selected_indices in [('sixteen_original_indices', indices), ('ten_unique_action_sequences', [g[0] for g in groups])]:
            ac = np.asarray([[r['actual_J_H10'] for r in actual[i]] for i in selected_indices], np.float64)
            av = np.asarray([[r['physical_valid'] for r in actual[i]] for i in selected_indices], np.bool_)
            j = compare_candidates(result['J10'].total[selected_indices], result['predictions'].valid[selected_indices], ac, av,
                                   atol=ATOL, rtol=RTOL)
            for key in ('selected_index', 'actual_best_index', 'actual_second_index'):
                j[key.replace('_index', '_original_index')] = None if j[key] is None else selected_indices[j[key]]
            valid = [all(r['physical_valid'] for r in actual[i]) for i in selected_indices]
            raw_ready = [actual[i][0]['raw_task']['status']['phase'] == 'ready' for i in selected_indices]
            proxy_ready = [actual[i][0]['actual_endpoint_proxy_ready'] for i in selected_indices]
            comparisons[name] = dict(original_indices=selected_indices, J_H10_stop=j,
                TASK_H10_stop=task_comparison(selected_indices, result['handoff'].rank_keys, result['handoff'].eligible, actual),
                handoff_against_raw_substeps=confusion(result['handoff'].ready[selected_indices],
                    result['handoff'].eligible[selected_indices], raw_ready, valid),
                handoff_against_actual_endpoint_surrogate=confusion(result['handoff'].ready[selected_indices],
                    result['handoff'].eligible[selected_indices], proxy_ready, valid))
        choices = [{**row, 'actual': selected_actual(row['selected_index'], actual)} for row in result['choices']]
        arm_results[arm] = dict(full_N128_choices=choices, comparisons=comparisons,
            prediction_valid=int(result['predictions'].valid.sum()), prediction_total=128,
            predictor_calls=predictor.calls, predictor_rows=predictor.rows,
            nominal_all_valid_forward_rows=1310,
            forward_count_note='Actual counters include any per-row retries after rejected batches.')
        if arm == 'R':
            with np.load(PARENT / 'initial_models/SK-11/predictions.npz', allow_pickle=False) as old:
                difference = np.abs(result['predictions'].states-old['states'])
                bridge = dict(states_exact=np.array_equal(result['predictions'].states, old['states'], equal_nan=True),
                    maximum_absolute_state_difference=float(np.nanmax(difference)),
                    validity_exact=np.array_equal(result['predictions'].valid, old['valid']),
                    J_H10_exact=np.array_equal(result['J10'].total, old['J_H10']),
                    task_rank_exact=np.array_equal(result['handoff'].rank_keys, old['task_rank']),
                    scope='Same historical N128 batch shape and original SK seed11 final checkpoint predictions.')
        if before != tree_hash(bundle.model.state_dict()) or not torch.equal(rng, torch.get_rng_state()):
            raise ValueError('Inference mutated weights or Torch RNG')
        print(json.dumps(dict(progress='candidate_regression_complete', arm=arm,
                              N128_valid=int(result['predictions'].valid.sum()))), flush=True)
    checks = dict(all_five_arms=len(arm_results)==5, all_ten_local_records=len(local_results)==10,
        all128_candidate_commands_reused=full.shape==(128, 10, 2), measured16_unique10=len(indices)==16 and len(groups)==10,
        historical_bindings_unchanged=required_source_bindings()=={p:expected_bindings[p] for p in required_source_bindings()},
        all_checkpoints_unchanged=all(digest(ROOT/p)==sha for bindings in checkpoint_hashes.values() for p,sha in bindings.items()),
        training_index_unchanged=digest(training_index_path)==training_index_sha,
        zero_future_engine_calls=not any(guards.values()), R_original_SK_numerically_exact=all(bridge[k]
            for k in ('states_exact','validity_exact','J_H10_exact','task_rank_exact')))
    summary = dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
        registration_sha256=registration_sha, training_manifest_sha256=training_manifest_sha,
        training_artifact_index_sha256=training_index_sha, training_artifact_count_verified=len(training_index),
        checkpoint_files_sha256=checkpoint_hashes, source_run=str(PARENT.relative_to(ROOT)), source_tick=99,
        indices=indices, exact_action_groups=groups, arms=arm_results, local_diagnostics=local_results,
        R_historical_SK_bridge=bridge, physical_intervals=0, learning_updates=0, future_engine_calls=guards,
        final_test_used=False, new_closed_loop_execution=False, P5_01='HOLD', elapsed_s=perf_counter()-began,
        scope='Already-seen source99 regression. Repeated/duplicate actions are not independent samples. '
              'Full-N128 unmeasured selections stay UNKNOWN; actual futures are offline labels only. '
              'Prediction-boundary readiness remains distinct from archived physical substep readiness.')
    save(output / 'summary.json', summary)
    save(output / 'artifact_hashes.json', {'sha256': {str(p.relative_to(output)):digest(p)
        for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'artifact_hashes.json'}})
    if not all(checks.values()):
        raise ValueError('Candidate regression integrity checks failed; retain outputs')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', type=Path, required=True)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--registration-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    existed_before = output.exists()
    try:
        result = evaluate(args.training_run.resolve(), args.registration.resolve(), args.registration_sha256, output)
        print(json.dumps({'status': result['status'], 'physical_intervals': 0, 'learning_updates': 0}), flush=True)
        return 0
    except Exception as error:
        if not existed_before and output.is_dir():
            save(output / 'failure.json', {'status': 'FAIL', 'error': str(error),
                'traceback': traceback.format_exc(), 'physical_intervals': 0, 'learning_updates': 0,
                'forward_counts': 'INCOMPLETE; inspect preserved arm artifacts',
                'automatic_retry': False})
            save(output / 'status.json', {'status': 'FAIL', 'error': str(error)})
            save(output / 'artifact_hashes.json', {'sha256': {str(p.relative_to(output)): digest(p)
                for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'artifact_hashes.json'}})
        print(json.dumps({'status': 'FAIL', 'error': str(error),
                          'existing_output_untouched': existed_before}), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
