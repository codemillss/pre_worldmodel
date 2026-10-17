"""Cross saved source-batch addends onto a fixed, same-snapshot stop batch."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import shutil
import sys
from time import perf_counter
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from push_align.cross_batch_probe import cross_batch_probe, array_sha256
from push_align.data_intervention_training import tree_hash
from push_align.normalization import load_normalizer
from p4_finite_step_probe import guards, load_frozen_snapshot

ROOT = Path(__file__).resolve().parents[1]
GRAD = ROOT / 'runs/p4-gradient-probe-20261015-01'
PARENT = ROOT / 'runs/p4-relative-loss-20261014-01'
NORMALIZER = ROOT / 'runs/p3-02-20261009-01/learning_data/normalizer'
TARGET = 'D11_zero5'
ALPHAS = (0., .25, .5, 1.)
RELATIONS = ('INCREASE', 'DECREASE', 'NUMERIC_TIE', 'NOT_INTERPRETED')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def arrays(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def validate(path, expected):
    require(sha(path) == expected, 'Execution registration changed')
    p = read(path)
    design = read(ROOT / p['design_path'])
    require(p['run_id'] == 'p4-cross-batch-20261015-01' and p['cases'] == design['cases']
            and len(p['cases']) == 28 and p['target_probe_name'] == design['target_probe_name'] == TARGET
            and p['alphas'] == design['alphas'] == list(ALPHAS), 'Source/target/alpha scope changed')
    require(sha(ROOT / p['design_path']) == p['design_sha256'], 'Frozen design changed')
    for name, digest in p['bindings_sha256'].items():
        require((ROOT / name).is_file() and sha(ROOT / name) == digest, 'Frozen source changed: ' + name)
    require(str(torch.__version__) == p['runtime']['torch_version']
            and np.__version__ == p['runtime']['numpy_version']
            and platform.python_version() == p['runtime']['python_version'], 'Runtime changed')
    for name, digest in p['runtime']['source_sha256'].items():
        require(sha(name) == digest, 'Functional runtime source changed')
    return p


def primary_counts(rows, controls_pass):
    """All planned positive endpoints remain, including whole held cases."""
    if not controls_pass:
        return None
    def group(selected):
        by_alpha = {}
        for index, alpha in enumerate(ALPHAS[1:], start=1):
            counts = {name: 0 for name in RELATIONS}
            for row in selected:
                e = row['evaluation']
                relation = e['evaluations'][index]['total_change_relation'] if e['status'] == 'PASS' else 'NOT_INTERPRETED'
                require(relation in RELATIONS, 'Unknown target relation')
                counts[relation] += 1
            by_alpha[str(alpha)] = counts
        return {'case_count': len(selected), 'planned_positive_endpoints': 3 * len(selected), 'by_alpha': by_alpha}
    names = list(dict.fromkeys(row['source_probe_name'] for row in rows))
    return {'all': group(rows), 'self_stop': group([row for row in rows if row['source_probe_name'] == TARGET]),
            'other_sources': group([row for row in rows if row['source_probe_name'] != TARGET]),
            'by_source_probe': {name: group([row for row in rows if row['source_probe_name'] == name]) for name in names}}


def global_controls(rows):
    self_rows = [row for row in rows if row['source_probe_name'] == TARGET]
    snapshots = {row['snapshot_id'] for row in rows}
    sources = {'historical_update1', TARGET, 'D11_transition', 'D11_noncontact'}
    checks = {
        'seven_snapshots_four_unique_sources': len(snapshots) == 7 and len({(row['snapshot_id'], row['source_probe_name']) for row in rows}) == 28 and all({row['source_probe_name'] for row in rows if row['snapshot_id'] == snapshot} == sources for snapshot in snapshots),
        'seven_self_stop_cases': len(self_rows) == 7,
        'all_self_stop_previous_reproduced': all(row['controls']['self_stop_previous']['pass'] for row in self_rows),
        'all28_alpha0_previous_stop_reproduced': len(rows) == 28 and all(row['controls']['alpha0_matches_previous_stop'] for row in rows),
        'all_sources_alpha0_same_within_snapshot': all(row['controls']['alpha0_matches_snapshot_first_source'] for row in rows),
    }
    return {'pass': all(checks.values()), 'checks': checks,
            'self_control_case_count': len(self_rows), 'alpha0_control_case_count': len(rows),
            'scope': 'Exact saved-output controls, not independent experimental replication.'}


def storage(out, p):
    size = sum(path.stat().st_size for path in out.rglob('*') if path.is_file())
    free = shutil.disk_usage(ROOT).free
    require(size <= p['maximum_run_bytes'] and free >= p['minimum_free_reserve_bytes'], 'Storage guard exceeded')
    return {'run_bytes_before_index': size, 'free_bytes': free}


def run(protocol_path, expected):
    began = perf_counter()
    p = validate(protocol_path, expected)
    out = ROOT / p['run_directory']
    require(not out.exists(), 'Never overwrite a run')
    require(shutil.disk_usage(ROOT).free >= p['minimum_free_reserve_bytes'] + p['maximum_run_bytes'] + p['document_reserve_bytes'], 'Insufficient reserve')
    out.mkdir(parents=True)
    save(out / 'registration.json', p)
    forbidden, originals = guards()
    old_threads, old_deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rng = torch.get_rng_state().clone()
    counts = {'snapshots': 0, 'cases': 0, 'attempted_evaluations': 0, 'completed_evaluations': 0,
              'forward_calls': 0, 'forward_rows': 0}
    try:
        meta = read(NORMALIZER / 'normalizer.json')['normalizer']
        normalizer = load_normalizer(NORMALIZER, meta['schema'], meta['dataset_identity'], meta['normalizer_hash'])
        parent = read(PARENT / 'manifest.json')
        config = parent['physical_config']
        require(normalizer.metadata['normalizer_hash'] == parent['normalizer_hash'], 'Normalizer mismatch')
        probes = arrays(GRAD / 'probe_batches.npz')
        target_arrays = {key: probes[TARGET + '_' + key] for key in ('states', 'actions', 'mask')}
        require(target_arrays['states'].shape == (24, 6, 10) and target_arrays['actions'].shape == (24, 5, 2)
                and target_arrays['mask'].shape == (24, 5) and target_arrays['mask'].all(), 'Fixed target stop batch differs')
        save(out / 'manifest.json', {'schema': 'p4-cross-batch-manifest-v1',
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': expected,
            'model_snapshots': p['snapshots'], 'source_probe_names': p['source_probe_names'], 'target_probe_name': TARGET,
            'normalizer_hash': normalizer.metadata['normalizer_hash'], 'physical_config': config,
            'target_array_sha256': {key: array_sha256(value) for key, value in target_arrays.items()},
            'target_batch_windows': 24, 'probe_batches_sha256': sha(GRAD / 'probe_batches.npz'),
            'runtime': p['runtime'], 'alphas': list(ALPHAS),
            'scope': 'Native saved source addends on same-snapshot fixed stop target; source norm differences retained; no training or historical trajectory replay.'})
        aggregate, source_checks = [], []
        for spec in p['snapshots']:
            model = load_frozen_snapshot(spec, normalizer, config)
            model_hash = tree_hash(model.state_dict())
            require(model_hash == spec['model_tree_hash'] and sum(value.numel() for value in model.parameters()) == 19338, 'Source model identity changed')
            optimizer = torch.load(ROOT / spec['checkpoint_path'] / 'training_state.pt', map_location='cpu', weights_only=True)['optimizer']
            opt_hash = tree_hash(optimizer)
            cases = [row for row in p['cases'] if row['snapshot_id'] == spec['snapshot_id']]
            require([row['source_probe_name'] for row in cases] == p['source_probe_names'], 'Source order changed')
            first_alpha0 = None
            for case in cases:
                require(case['target_probe_name'] == TARGET, 'Target routing changed')
                direction_record = read(ROOT / case['source_direction_summary'])
                target_record = read(ROOT / case['target_gradient_summary'])
                order = target_record['parameter_order']
                require(direction_record['parameter_order'] == order and direction_record['snapshot_id'] == case['snapshot_id']
                        and direction_record['probe_name'] == case['source_probe_name']
                        and target_record['snapshot_id'] == case['snapshot_id'] and target_record['probe_name'] == TARGET
                        and direction_record['source_weights_tree_hash'] == target_record['before_weights_hash'] == model_hash
                        and direction_record['source_optimizer_tree_hash'] == opt_hash,
                        'Bound source and target coordinates/model identities differ')
                require(case['source_direction_index'] == 3 and direction_record['directions'][3]['objective'] == 'total'
                        and direction_record['directions'][3]['mode'] == 'KEEP', 'Not the registered native TOTAL/KEEP addend')
                delta = arrays(ROOT / case['source_direction_npz'])['directions'][3].copy()
                gradient = arrays(ROOT / case['target_gradient_npz'])['gradients'][4].copy()
                states, actions, mask = (torch.from_numpy(target_arrays[key].copy()) for key in ('states', 'actions', 'mask'))
                common = {'snapshot_id': case['snapshot_id'], 'model_tree_hash': model_hash, 'parameter_order': order}
                source_binding = {**common, 'probe_name': case['source_probe_name'], 'objective': 'total',
                                  'mode': 'KEEP', 'direction_index': 3, 'delta_sha256': array_sha256(delta)}
                target_binding = {**common, 'probe_name': TARGET, 'batch_windows': 24,
                                  **{key + '_sha256': array_sha256(value) for key, value in target_arrays.items()}}
                gradient_binding = {**common, 'probe_name': TARGET, 'component': 'total', 'gradient_sha256': array_sha256(gradient)}
                loss_binding = {**common, 'probe_name': TARGET, 'loss_values': target_record['loss_values']}
                trace = []
                def count_source(active_model, inputs):
                    size = int(inputs[0].shape[0])
                    counts['forward_calls'] += 1
                    counts['forward_rows'] += size
                    require(counts['forward_calls'] <= 672 and counts['forward_rows'] <= 26880, 'Forward budget exceeded')
                    trace.append({'rows': size, 'parameter_tree_hash': tree_hash(active_model.state_dict())})
                handle = model.register_forward_pre_hook(count_source)
                try:
                    result = cross_batch_probe(model, normalizer, states, actions, mask, delta,
                        source_binding=source_binding, target_binding=target_binding,
                        target_gradient_binding=gradient_binding, target_loss_binding=loss_binding,
                        target_stop_gradient=gradient, target_stop_reference=target_record['loss_values'], parameter_order=order)
                finally:
                    handle.remove()
                e = result['summary']
                require(len(trace) == e['counters']['root_model_forward_calls']
                        and sum(item['rows'] for item in trace) == e['counters']['root_model_forward_rows'], 'Source and proxy counts differ')
                cursor = 0
                for index, endpoint in enumerate(e['evaluations']):
                    params = {item['name']: torch.from_numpy(result['evaluated_parameters'][index, item['start']:item['end']].reshape(item['shape']).copy()) for item in order}
                    h = tree_hash(params)
                    actual = trace[cursor:cursor + endpoint['forward_calls']]
                    require([item['rows'] for item in actual] == endpoint['forward_rows']
                            and all(item['parameter_tree_hash'] == h for item in actual), 'Actual forward parameter map differs')
                    cursor += endpoint['forward_calls']
                require(cursor == len(trace), 'Unassigned forwards')
                old_source = arrays(ROOT / case['previous_finite_source_parameters'])
                require(all(result[key].dtype == old_source[key].dtype and np.array_equal(result[key], old_source[key])
                            for key in ('alphas', 'evaluated_parameters', 'rounded_parameter_differences')),
                        'Saved native source parameter map changed')
                old_stop = read(ROOT / case['previous_finite_stop_summary'])['evaluation']
                old_predictions = arrays(ROOT / case['previous_finite_stop_predictions'])
                current0 = {'losses': e['evaluations'][0]['loss_values'],
                            'teacher': result['teacher_predictions'][0].copy(), 'free': result['free_predicted_states'][0].copy()}
                alpha0_previous = (current0['losses'] == old_stop['evaluations'][0]['loss_values']
                                   and np.array_equal(current0['teacher'], old_predictions['teacher_predictions'][0])
                                   and np.array_equal(current0['free'], old_predictions['free_predicted_states'][0]))
                if first_alpha0 is None:
                    first_alpha0 = current0
                same0 = (current0['losses'] == first_alpha0['losses']
                         and np.array_equal(current0['teacher'], first_alpha0['teacher']) and np.array_equal(current0['free'], first_alpha0['free']))
                self_control = None
                if case['source_probe_name'] == TARGET:
                    self_checks = {'all_four_loss_values_and_statuses_exact': all(new['loss_values'] == old['loss_values'] and new['status'] == old['status'] for new, old in zip(e['evaluations'], old_stop['evaluations'])),
                                   'teacher_predictions_exact': np.array_equal(result['teacher_predictions'], old_predictions['teacher_predictions']),
                                   'free_predictions_exact': np.array_equal(result['free_predicted_states'], old_predictions['free_predicted_states'])}
                    self_control = {'pass': all(self_checks.values()), 'checks': self_checks}
                controls = {'source_parameter_maps_previous_exact': True, 'alpha0_matches_previous_stop': bool(alpha0_previous),
                            'alpha0_matches_snapshot_first_source': bool(same0), 'self_stop_previous': self_control}
                folder = out / 'cases' / spec['snapshot_id'] / case['source_probe_name']
                folder.mkdir(parents=True)
                np.savez_compressed(folder / 'parameter_displacements.npz', **{key: result[key] for key in ('alphas', 'evaluated_parameters', 'rounded_parameter_differences')})
                np.savez_compressed(folder / 'predictions.npz', **{key: result[key] for key in ('teacher_predictions', 'free_predicted_states')})
                record = {'snapshot_id': spec['snapshot_id'], 'source_probe_name': case['source_probe_name'], 'target_probe_name': TARGET,
                          'source': case, 'parameter_order': order, 'evaluation': e, 'controls': controls,
                          'source_weights_tree_hash': model_hash, 'source_optimizer_tree_hash': opt_hash,
                          'source_forward_trace': trace, 'parameter_arrays_sha256': sha(folder / 'parameter_displacements.npz'),
                          'prediction_arrays_sha256': sha(folder / 'predictions.npz')}
                require(tree_hash(model.state_dict()) == model_hash and tree_hash(optimizer) == opt_hash, 'Original model/optimizer mutated')
                save(folder / 'summary.json', record)
                aggregate.append({'snapshot_id': spec['snapshot_id'], 'source_probe_name': case['source_probe_name'], 'target_probe_name': TARGET,
                                  'summary_path': str((folder / 'summary.json').relative_to(out)), 'evaluation': e, 'controls': controls})
                counts['cases'] += 1
                counts['attempted_evaluations'] += 4
                counts['completed_evaluations'] += e['counters']['successful_evaluations']
                storage(out, p)
            source_checks.append({'snapshot_id': spec['snapshot_id'], 'weights_unchanged': tree_hash(model.state_dict()) == model_hash,
                                  'optimizer_unchanged': tree_hash(optimizer) == opt_hash, 'weight_tree_hash': model_hash, 'optimizer_tree_hash': opt_hash})
            counts['snapshots'] += 1
            print(json.dumps({'snapshot_completed': spec['snapshot_id'], 'counters': counts}), flush=True)
        require(counts['snapshots'] == 7 and counts['cases'] == 28 and counts['attempted_evaluations'] == 112, 'Case denominator differs')
        require(counts['forward_calls'] <= 672 and counts['forward_rows'] <= 26880, 'Forward budget exceeded')
        if counts['completed_evaluations'] == 112:
            require(counts == p['expected_if_all_evaluations_complete'], 'Completed budget differs')
        require(not any(forbidden.values()) and torch.equal(rng, torch.get_rng_state()), 'Forbidden call or CPU RNG mutation')
        validate(protocol_path, expected)
        controls = global_controls(aggregate)
        effects = primary_counts(aggregate, controls['pass'])
        save(out / 'aggregate.json', {'schema': 'p4-cross-batch-aggregate-v1', 'row_count': 28, 'rows': aggregate,
                                     'global_controls': controls, 'primary_counts': effects,
                                     'scope': 'All planned source/target/alpha appearances, raw finite-style decisions secondary.'})
        summary = {'schema': 'p4-cross-batch-summary-v1', 'execution_status': 'PASS',
                   'analysis_status': 'DESCRIPTIVE_COMPLETE' if controls['pass'] else 'HOLD_CONTROL_FAILURE',
                   'protocol_sha256': expected, 'manifest_sha256': sha(out / 'manifest.json'), 'counters': counts,
                   'global_controls': controls, 'primary_counts': effects, 'forbidden_execution_counts': forbidden,
                   'source_tensor_checks': source_checks, 'global_rng_unchanged': True,
                   'new_derivative_traversals': 0, 'new_optimizer_updates': 0, 'new_physics_steps': 0,
                   'new_checkpoint_writes': 0, 'new_data': False, 'normalizer_refit': False,
                   'validation_or_development_probes_used': False, 'final_test_used': False,
                   'historical_trajectory_replayed': False, 'model_selection': 'NONE', 'learning_rate_adoption': 'NONE',
                   'alignment_improvement': 'NOT_EVALUATED', 'personal_understanding': 'NOT_ASSESSED', 'P5_01': 'HOLD',
                   'wall_time_runner_only_s': perf_counter() - began,
                   'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
                   'resources': storage(out, p)}
        save(out / 'summary.json', summary)
        save(out / 'artifact_hashes.json', {'sha256': {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob('*')) if path.is_file()}})
        return summary
    except Exception as error:
        save(out / 'failure.json', {'status': 'FAIL', 'type': type(error).__name__, 'error': str(error), 'counters': counts,
                                  'forbidden_execution_counts': forbidden, 'traceback': traceback.format_exc(), 'automatic_retry': False})
        raise
    finally:
        for obj, name, function in reversed(originals):
            setattr(obj, name, function)
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.protocol, arguments.protocol_sha256), ensure_ascii=False), flush=True)
