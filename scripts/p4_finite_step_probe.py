"""Registered finite loss evaluations on disposable parameter maps, never training."""
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
import mujoco
import numpy as np
import torch
from push_align.data_intervention_training import tree_hash
from push_align.dynamics import DeltaMLP
from push_align.finite_step_probe import finite_step_probe
from push_align.normalization import load_normalizer

ROOT = Path(__file__).resolve().parents[1]
GRAD = ROOT / 'runs/p4-gradient-probe-20261015-01'
ADAM = ROOT / 'runs/p4-adam-probe-20261015-01'
PARENT = ROOT / 'runs/p4-relative-loss-20261014-01'
NORMALIZER = ROOT / 'runs/p3-02-20261009-01/learning_data/normalizer'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def validate(path, expected):
    require(sha(path) == expected, 'Execution registration identity changed')
    p = read(path)
    design = read(ROOT / p['design_path'])
    require(p['run_id'] == 'p4-finite-step-20261015-01'
            and p['alphas'] == design['alphas'] == [0., .25, .5, 1.]
            and p['cases'] == design['cases'] and len(p['cases']) == 28,
            'Unregistered case/direction/alpha change')
    for name, digest in p['bindings_sha256'].items():
        require((ROOT / name).is_file() and sha(ROOT / name) == digest, 'Frozen source changed: ' + name)
    require(str(torch.__version__) == p['runtime']['torch_version']
            and np.__version__ == p['runtime']['numpy_version']
            and platform.python_version() == p['runtime']['python_version'], 'Runtime version changed')
    for name, digest in p['runtime']['source_sha256'].items():
        require(sha(name) == digest, 'Functional runtime changed: ' + name)
    return p


def guards():
    entries = [(mujoco, name, 'physics.' + name) for name in
               ('mj_step', 'mj_step1', 'mj_step2', 'mj_forward', 'mj_inverse')]
    entries += [(torch.optim.Adam, '__init__', 'Adam_instances'),
                (torch.optim.Optimizer, '__init__', 'Optimizer_instances'),
                (torch.optim.Adam, 'step', 'Adam.step'),
                (torch.optim.Optimizer, 'step', 'Optimizer.step'),
                (torch.autograd, 'grad', 'autograd.grad'),
                (torch.Tensor, 'backward', 'Tensor.backward'),
                (torch, 'save', 'torch.save')]
    counts = {label: 0 for _, _, label in entries}
    originals = []
    def blocked(label):
        def fail(*args, **kwargs):
            counts[label] += 1
            raise RuntimeError('Finite-step evaluation forbids ' + label)
        return fail
    for obj, key, label in entries:
        originals.append((obj, key, getattr(obj, key)))
        setattr(obj, key, blocked(label))
    return counts, originals


def storage(out, p):
    size = sum(path.stat().st_size for path in out.rglob('*') if path.is_file())
    free = shutil.disk_usage(ROOT).free
    require(size <= p['maximum_run_bytes'] and free >= p['minimum_free_reserve_bytes'], 'Storage guard exceeded')
    return {'run_bytes_before_index': size, 'free_bytes': free}


def load_frozen_snapshot(spec, normalizer, config):
    """Read the already validated immutable checkpoint, without any optimizer.

    Historical general loaders construct a disposable Adam while validating
    its schema even with resume=False. This narrower path instead binds the
    previously audited file/metadata/tensor identities and loads weights only.
    """
    path = ROOT / spec['checkpoint_path']
    metadata = read(path / 'checkpoint.json')
    require(sha(path / 'checkpoint.json') == spec['checkpoint_json_sha256']
            and metadata['checkpoint_hash'] == spec['checkpoint_hash']
            and metadata['config'] == config
            and metadata['normalizer_hash'] == normalizer.metadata['normalizer_hash'],
            'Frozen checkpoint metadata/config/normalizer identity changed')
    require(all(sha(path / name) == digest for name, digest in spec['files_sha256'].items()),
            'Frozen checkpoint files changed')
    weights = torch.load(path / 'weights.pt', map_location='cpu', weights_only=True)
    require(tree_hash(weights) == spec['model_tree_hash'], 'Frozen weight tensor identity changed')
    require(all(value.dtype == torch.float32 and value.device.type == 'cpu'
                and bool(torch.isfinite(value).all()) for value in weights.values()),
            'Finite CPU float32 checkpoint weights required')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = DeltaMLP(config)
        model.load_state_dict(weights, strict=True)
    model.eval()
    require(tree_hash(model.state_dict()) == spec['model_tree_hash']
            and all(value.grad is None for value in model.parameters()), 'Frozen model load differs')
    return model


def run(protocol_path, expected):
    began = perf_counter()
    p = validate(protocol_path, expected)
    out = ROOT / p['run_directory']
    require(not out.exists(), 'Never overwrite an experiment')
    require(shutil.disk_usage(ROOT).free >= p['minimum_free_reserve_bytes'] + p['maximum_run_bytes'] + p['document_reserve_bytes'], 'Insufficient reserve')
    out.mkdir(parents=True)
    save(out / 'registration.json', p)
    forbidden, originals = guards()
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rng = torch.get_rng_state().clone()
    counts = {'snapshots': 0, 'cases': 0, 'attempted_evaluations': 0,
              'completed_evaluations': 0, 'forward_calls': 0, 'forward_rows': 0}
    try:
        meta = read(NORMALIZER / 'normalizer.json')['normalizer']
        normalizer = load_normalizer(NORMALIZER, meta['schema'], meta['dataset_identity'], meta['normalizer_hash'])
        parent = read(PARENT / 'manifest.json')
        config = parent['physical_config']
        require(normalizer.metadata['normalizer_hash'] == parent['normalizer_hash'], 'Normalizer mismatch')
        with np.load(GRAD / 'probe_batches.npz', allow_pickle=False) as source:
            probes = {key: source[key].copy() for key in source.files}
        grad_manifest = read(GRAD / 'manifest.json')
        require(grad_manifest['model_snapshots'] == p['snapshots'], 'Frozen snapshot identity changed')
        manifest = {'schema': 'p4-finite-step-manifest-v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                    'protocol_sha256': expected, 'model_snapshots': p['snapshots'],
                    'normalizer_hash': normalizer.metadata['normalizer_hash'], 'physical_config': config,
                    'gradient_run_index_sha256': sha(GRAD / 'artifact_hashes.json'),
                    'adam_run_index_sha256': sha(ADAM / 'artifact_hashes.json'),
                    'probe_batches_sha256': sha(GRAD / 'probe_batches.npz'),
                    'runtime': p['runtime'], 'alphas': p['alphas'],
                    'scope': 'Same frozen TOTAL/KEEP addends, independently evaluated parameter copies; no historical replay, training, policy execution or model selection.'}
        save(out / 'manifest.json', manifest)
        aggregate = []
        source_checks = []
        for spec in p['snapshots']:
            model = load_frozen_snapshot(spec, normalizer, config)
            require(not model.training and sum(v.numel() for v in model.parameters()) == 19338, 'Source model contract differs')
            model_hash = tree_hash(model.state_dict())
            require(model_hash == spec['model_tree_hash'], 'Loaded model identity differs')
            optimizer_state = torch.load(ROOT / spec['checkpoint_path'] / 'training_state.pt', map_location='cpu', weights_only=True)['optimizer']
            opt_hash = tree_hash(optimizer_state)
            rows = [row for row in p['cases'] if row['snapshot_id'] == spec['snapshot_id']]
            require(len(rows) == 4, 'Exactly four frozen probes per snapshot required')
            for row in rows:
                name = row['probe_name']
                old = read(ROOT / row['original_gradient_summary'])
                direction_summary = read(Path(ROOT / row['source_direction_npz']).with_name('summary.json'))
                parameter_order = old['parameter_order']
                require(direction_summary['parameter_order'] == parameter_order
                        and direction_summary['directions'][3]['objective'] == 'total'
                        and direction_summary['directions'][3]['mode'] == 'KEEP'
                        and row['source_direction_index'] == 3, 'Direction coordinate or objective changed')
                with np.load(ROOT / row['source_direction_npz'], allow_pickle=False) as z:
                    delta = z['directions'][3].copy()
                with np.load(Path(ROOT / row['original_gradient_summary']).with_name('gradients.npz'), allow_pickle=False) as z:
                    gradient = z['gradients'][4].copy()
                states, actions, mask = (torch.from_numpy(probes[name + '_' + key].copy()) for key in ('states', 'actions', 'mask'))
                call_trace = []
                def count_source(active_model, inputs):
                    count = int(inputs[0].shape[0])
                    counts['forward_calls'] += 1
                    counts['forward_rows'] += count
                    require(counts['forward_calls'] <= 672 and counts['forward_rows'] <= 31360, 'Forward budget exceeded')
                    call_trace.append({'rows': count, 'parameter_tree_hash': tree_hash(active_model.state_dict())})
                handle = model.register_forward_pre_hook(count_source)
                try:
                    result = finite_step_probe(model, normalizer, states, actions, mask, delta,
                                               raw_total_gradient=gradient, reference_losses=old['loss_values'],
                                               parameter_order=parameter_order)
                finally:
                    handle.remove()
                folder = out / 'cases' / spec['snapshot_id'] / name
                folder.mkdir(parents=True)
                np.savez_compressed(folder / 'parameter_displacements.npz',
                                    **{key: result[key] for key in ('alphas', 'evaluated_parameters', 'rounded_parameter_differences')})
                np.savez_compressed(folder / 'predictions.npz',
                                    **{key: result[key] for key in ('teacher_predictions', 'free_predicted_states')})
                evaluation = result['summary']
                require(len(call_trace) == evaluation['counters']['root_model_forward_calls']
                        and sum(item['rows'] for item in call_trace) == evaluation['counters']['root_model_forward_rows'],
                        'Source-model hook and evaluator forward counts differ')
                cursor = 0
                for alpha_index, endpoint in enumerate(evaluation['evaluations']):
                    parameters = {item['name']: torch.from_numpy(result['evaluated_parameters'][alpha_index, item['start']:item['end']].reshape(item['shape']).copy())
                                  for item in parameter_order}
                    expected_hash = tree_hash(parameters)
                    calls = call_trace[cursor:cursor + endpoint['forward_calls']]
                    require([item['rows'] for item in calls] == endpoint['forward_rows']
                            and all(item['parameter_tree_hash'] == expected_hash for item in calls),
                            'Model forward did not use the registered evaluated parameter copy')
                    cursor += endpoint['forward_calls']
                require(cursor == len(call_trace), 'Unassigned source-model forwards')
                record = {'snapshot_id': spec['snapshot_id'], 'probe_name': name,
                          'source': row, 'parameter_order': parameter_order,
                          'source_weights_tree_hash': model_hash, 'source_optimizer_tree_hash': opt_hash,
                          'evaluation': evaluation, 'source_forward_trace': call_trace,
                          'parameter_arrays_sha256': sha(folder / 'parameter_displacements.npz'),
                          'prediction_arrays_sha256': sha(folder / 'predictions.npz')}
                require(tree_hash(model.state_dict()) == model_hash and tree_hash(optimizer_state) == opt_hash,
                        'Original model or saved optimizer tensors mutated')
                save(folder / 'summary.json', record)
                aggregate.append({'snapshot_id': spec['snapshot_id'], 'probe_name': name,
                                  'summary_path': str((folder / 'summary.json').relative_to(out)), 'evaluation': evaluation})
                counts['cases'] += 1
                counts['attempted_evaluations'] += 4
                counts['completed_evaluations'] += evaluation['counters']['successful_evaluations']
                storage(out, p)
            source_checks.append({'snapshot_id': spec['snapshot_id'], 'weights_unchanged': tree_hash(model.state_dict()) == model_hash,
                                  'optimizer_unchanged': tree_hash(optimizer_state) == opt_hash,
                                  'weight_tree_hash': model_hash, 'optimizer_tree_hash': opt_hash})
            counts['snapshots'] += 1
            print(json.dumps({'snapshot_completed': spec['snapshot_id'], 'counters': counts}), flush=True)
        require(counts['snapshots'] == 7 and counts['cases'] == 28 and counts['attempted_evaluations'] == 112, 'Incomplete case denominator')
        require(counts['forward_calls'] <= 672 and counts['forward_rows'] <= 31360, 'Registered forward budget exceeded')
        if counts['completed_evaluations'] == 112:
            require(counts['forward_calls'] == 672 and counts['forward_rows'] == 31360, 'Completed evaluation count differs')
        require(not any(forbidden.values()) and torch.equal(rng, torch.get_rng_state()), 'Forbidden call or global RNG mutation')
        validate(protocol_path, expected)
        save(out / 'aggregate.json', {'schema': 'p4-finite-step-aggregate-v1', 'rows': aggregate, 'row_count': 28,
                                      'scope': 'All frozen overlapping train probes and all four alphas retained, including failed rows.'})
        summary = {'schema': 'p4-finite-step-summary-v1', 'execution_status': 'PASS',
                   'analysis_status': 'DESCRIPTIVE_COMPLETE', 'protocol_sha256': expected,
                   'manifest_sha256': sha(out / 'manifest.json'), 'counters': counts,
                   'forbidden_execution_counts': forbidden, 'source_tensor_checks': source_checks,
                   'global_rng_unchanged': True, 'new_derivative_traversals': 0, 'new_optimizer_updates': 0,
                   'new_physics_steps': 0, 'new_checkpoint_writes': 0, 'new_data': False,
                   'normalizer_refit': False, 'final_test_used': False, 'validation_or_development_probes_used': False,
                   'historical_trajectory_replayed': False, 'model_selection': 'NONE',
                   'alignment_improvement': 'NOT_EVALUATED', 'personal_understanding': 'NOT_ASSESSED', 'P5_01': 'HOLD',
                   'wall_time_runner_only_s': perf_counter() - began,
                   'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
                   'resources': storage(out, p)}
        save(out / 'summary.json', summary)
        save(out / 'artifact_hashes.json', {'sha256': {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob('*')) if path.is_file()}})
        return summary
    except Exception as error:
        save(out / 'failure.json', {'status': 'FAIL', 'type': type(error).__name__, 'error': str(error),
                                  'counters': counts, 'forbidden_execution_counts': forbidden,
                                  'traceback': traceback.format_exc(), 'automatic_retry': False})
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
    args = parser.parse_args()
    print(json.dumps(run(args.protocol, args.protocol_sha256), ensure_ascii=False), flush=True)
