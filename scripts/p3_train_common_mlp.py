"""P3-04: immutable, validation-selected common MLP/Adam/RNG per seed."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
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
from zoneinfo import ZoneInfo

import numpy as np
import torch

from push_align.checkpoint import load_checkpoint, make_optimizer, save_checkpoint
from push_align.config import ROOT
from push_align.data import digest
from push_align.dynamics import DeltaMLP, model_spec, predict_delta
from push_align.learning_data import load_learning_data
from push_align.preparation import SPLIT_CODES
from push_align.prediction import evaluate_predictions, evaluate_state_hold
from push_align.training import training_update
from p2_episode_contract_check import source_bundle
from p2_train_pilot import full_loss, infer_normalized, parameter_vector, save, save_evaluation, tensor_tree_equal
from p3_geometry_pilot import verify_artifacts
from p3_train_ridge import array_hash

DEFAULT_LEARNING = ROOT / 'runs/p3-02-20261009-01/learning_data'
# Independent registered-settings fingerprint, filled before any execution.
EXPECTED_SETTINGS_SHA256 = 'd4e833a53bf8079ac9a994c17aba8256e739441c0c5fc78606fa595f4640819c'


def settings_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_settings(settings):
    if not isinstance(settings, dict) or settings_hash(settings) != EXPECTED_SETTINGS_SHA256:
        raise ValueError('P3-04 requires the independently registered settings; declare a new version for changes')
    return settings


def choose_epoch(candidates):
    eligible = [r for r in candidates if type(r.get('epoch')) is int and 1 <= r['epoch'] <= 100
                and r.get('selection_status') == 'eligible'
                and type(r.get('selection_score')) in (int, float)
                and np.isfinite(r['selection_score']) and r['selection_score'] >= 0]
    if not eligible:
        raise ValueError('No eligible trained K5 checkpoint; a final or untrained fallback is forbidden')
    return min(eligible, key=lambda r: (r['selection_score'], r['epoch']))


def train_tensors(prepared):
    a = prepared.arrays
    rows = np.flatnonzero(a['split_code'] == SPLIT_CODES['train'])
    if not len(rows):
        raise ValueError('No original train rows')
    features = a['features'][rows].copy()
    targets = a['deltas'][rows].copy()
    if features.dtype != np.float64 or targets.dtype != np.float64 or features.shape != (len(rows), 11) or targets.shape != (len(rows), 10):
        raise ValueError('Train tensors require original physical float64 11/10 columns')
    return rows, torch.from_numpy(features), torch.from_numpy(targets)


def train_epoch(model, optimizer, normalizer, features, targets, rows, shuffle, batch_size, clip):
    if len(rows) != len(features) or len(targets) != len(rows) or type(batch_size) is not int or batch_size <= 0:
        raise ValueError('Invalid train epoch batch contract')
    order = torch.randperm(len(rows), generator=shuffle)
    batches = []
    for start in range(0, len(rows), batch_size):
        positions = order[start:start + batch_size]
        batches.append(training_update(model, optimizer, normalizer, features[positions], targets[positions], clip))
    return {'order': rows[order.numpy()].tolist(), 'batches': batches}


def fit_fixed_updates(prepared, normalizer, config, seed, shuffle_seed, updates=3, batch_size=256):
    """Train-only mask diagnostic. No validation selection or original writes."""
    if type(updates) is not int or updates <= 0 or type(batch_size) is not int or batch_size <= 0:
        raise ValueError('Positive diagnostic update/batch budget required')
    rows, features, targets = train_tensors(prepared)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = DeltaMLP(config)
        optimizer = make_optimizer(model, config)
        shuffle = torch.Generator(device='cpu').manual_seed(shuffle_seed)
        orders = []
        completed = 0
        while completed < updates:
            order = torch.randperm(len(rows), generator=shuffle)
            orders.append(rows[order.numpy()].tolist())
            for start in range(0, len(rows), batch_size):
                positions = order[start:start + batch_size]
                training_update(model, optimizer, normalizer, features[positions], targets[positions], config['training']['gradient_clip_norm'])
                completed += 1
                if completed == updates:
                    break
    return {'model': model, 'optimizer': optimizer, 'shuffle': shuffle, 'rows': rows, 'order_history': orders}


def evaluation_equal(a, b):
    return a.summary == b.summary and a.traces.keys() == b.traces.keys() and all(
        np.array_equal(a.traces[k], b.traces[k], equal_nan=True)
        if np.issubdtype(a.traces[k].dtype, np.floating) else np.array_equal(a.traces[k], b.traces[k])
        for k in a.traces)


def witness_state(model, optimizer, shuffle):
    return {'weights': deepcopy(model.state_dict()), 'optimizer': deepcopy(optimizer.state_dict()),
            'torch_rng': torch.get_rng_state().clone(), 'shuffle_rng': shuffle.get_state().clone()}


def load_common_checkpoint(path, normalizer, config, provenance, settings, seed, *, epoch=None, resume=False):
    """The runner binds a seed-specific artifact to its declared epoch budget."""
    validate_settings(settings)
    if type(seed) is not int or seed not in settings['seeds']:
        raise ValueError('Expected training seed is not registered')
    bundle = load_checkpoint(path, normalizer, config, provenance, resume=resume, training_settings=settings)
    m = bundle.metadata
    if m['seed'] != seed or not 1 <= m['epoch'] <= settings['main_epochs'] or m['updates'] != m['epoch'] * settings['updates_per_epoch']:
        raise ValueError('Common checkpoint seed/epoch/update boundary differs from registered budget')
    if epoch is not None and (type(epoch) is not int or m['epoch'] != epoch):
        raise ValueError('Common checkpoint is not the expected selected/final epoch')
    return bundle


def replay_selected_checkpoint(path, normalizer, config, provenance, settings, witness, features, targets, rows, seed):
    """Three disposable copies; the saved model and official budget are untouched."""
    files_before = {str(p): digest(p) for p in path.iterdir() if p.is_file()}
    global_before = torch.get_rng_state().clone()
    copies = [load_common_checkpoint(path, normalizer, config, provenance, settings, seed, resume=True) for _ in range(2)]
    restored = copies[0]
    initial = {'selected_weights_match_live_witness': tensor_tree_equal(restored.model.state_dict(), witness['weights']),
               'selected_adam_matches_live_witness': tensor_tree_equal(restored.optimizer.state_dict(), witness['optimizer']),
               'selected_torch_rng_matches_live_witness': torch.equal(restored.torch_rng_state, witness['torch_rng']),
               'selected_shuffle_rng_matches_live_witness': torch.equal(restored.shuffle_generator.get_state(), witness['shuffle_rng']),
               'two_copies_initially_identical': tensor_tree_equal(copies[0].model.state_dict(), copies[1].model.state_dict()) and tensor_tree_equal(copies[0].optimizer.state_dict(), copies[1].optimizer.state_dict()),
               'loading_does_not_change_global_rng': torch.equal(global_before, torch.get_rng_state())}
    with torch.random.fork_rng(devices=[]):
        reference = DeltaMLP(config)
        reference.load_state_dict(witness['weights'])
        reference_optimizer = make_optimizer(reference, config)
        reference_optimizer.load_state_dict(deepcopy(witness['optimizer']))
        reference_shuffle = torch.Generator(device='cpu').set_state(witness['shuffle_rng'])
        orders = [torch.randperm(len(rows), generator=g) for g in (reference_shuffle, copies[0].shuffle_generator, copies[1].shuffle_generator)]
        positions = orders[0][:settings['batch_size']]
        diagnostics = []
        next_torch_rngs = []
        untouched_copy = deepcopy(copies[1].model.state_dict())
        untouched_adam = deepcopy(copies[1].optimizer.state_dict())
        for model, optimizer, rng in [(reference, reference_optimizer, witness['torch_rng']),
                                      (copies[0].model, copies[0].optimizer, copies[0].torch_rng_state),
                                      (copies[1].model, copies[1].optimizer, copies[1].torch_rng_state)]:
            torch.set_rng_state(rng)
            diagnostics.append(training_update(model, optimizer, normalizer, features[positions], targets[positions], config['training']['gradient_clip_norm']))
            next_torch_rngs.append(torch.get_rng_state().clone())
            if model is copies[0].model:
                independent = tensor_tree_equal(copies[1].model.state_dict(), untouched_copy)
                adam_independent = tensor_tree_equal(copies[1].optimizer.state_dict(), untouched_adam)
        next_checks = {'next_shuffle_order_identical': all(torch.equal(orders[0], o) for o in orders[1:]),
                       'restored_copies_do_not_share_weights': independent,
                       'restored_copies_do_not_share_adam': adam_independent,
                       'next_parameters_identical_to_witness_and_each_other': all(tensor_tree_equal(reference.state_dict(), c.model.state_dict()) for c in copies),
                       'next_adam_identical_to_witness_and_each_other': all(tensor_tree_equal(reference_optimizer.state_dict(), c.optimizer.state_dict()) for c in copies),
                       'next_torch_rng_identical': all(torch.equal(next_torch_rngs[0], rng) for rng in next_torch_rngs[1:]),
                       'next_shuffle_rng_identical': all(torch.equal(reference_shuffle.get_state(), c.shuffle_generator.get_state()) for c in copies),
                       'next_loss_gradient_diagnostics_identical': diagnostics[0] == diagnostics[1] == diagnostics[2]}
        # A fresh Adam is a counterfactual diagnostic, never a P4 branch.
        reset_model = DeltaMLP(config); reset_model.load_state_dict(witness['weights'])
        reset_optimizer = make_optimizer(reset_model, config)
        torch.set_rng_state(witness['torch_rng'])
        training_update(reset_model, reset_optimizer, normalizer, features[positions], targets[positions], config['training']['gradient_clip_norm'])
        reset_difference = float((parameter_vector(reference) - parameter_vector(reset_model)).abs().max())
    checks = {**initial, **next_checks, 'diagnostic_restores_global_rng': torch.equal(global_before, torch.get_rng_state()),
              'saved_selected_files_unchanged': files_before == {str(p): digest(p) for p in path.iterdir() if p.is_file()}}
    return {'checks': checks, 'pass': all(checks.values()), 'next_original_batch_rows': rows[positions.numpy()].tolist(),
            'selected_epoch': restored.metadata['epoch'], 'selected_updates': restored.metadata['updates'],
            'diagnostic_updates_per_copy': 1, 'original_training_budget_changed': False,
            'fresh_adam_counterfactual_max_parameter_difference': reset_difference,
            'fresh_adam_scope': 'diagnostic only; nonzero difference is reported, not a performance or P3 completion requirement',
            'scope': 'same one-step loss resumption proof; S1/SK losses and additional training are not run'}


def run(args, out):
    began = perf_counter()
    settings = validate_settings(tomllib.loads(args.training_config.read_text()))
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    learning = args.learning_data.resolve()
    if learning != DEFAULT_LEARNING.resolve():
        raise ValueError('Main data source is fixed for this runner version')
    prior_folders = [learning.parent, ROOT / 'runs/p3-03-20261009-01']
    upstream_before = {str(p.relative_to(ROOT)): verify_artifacts(p) for p in prior_folders}
    m = json.loads((learning / 'normalizer/normalizer.json').read_text())['normalizer']
    if m['normalizer_hash'] != settings['normalizer_hash'] or m['dataset_identity']['manifest_sha256'] != settings['dataset_manifest_sha256']:
        raise ValueError('Frozen main data or normalizer changed')
    prepared, normalizer = load_learning_data(learning, m['schema'], m['dataset_identity'], m['normalizer_hash'])
    config = prepared.metadata['contract']['resolved_config']
    rows, features, targets = train_tensors(prepared)
    val_rows = np.flatnonzero(prepared.arrays['split_code'] == SPLIT_CODES['validation'])
    if len(rows) != settings['fit_count'] or len(val_rows) != settings['validation_count'] or len(rows) + len(val_rows) != len(prepared.arrays['features']):
        raise ValueError('Registered train/validation rows changed')
    if config['training']['seed_candidates'] != settings['seeds'] or config['training']['main_max_epochs_candidate'] != settings['main_epochs'] or config['training']['batch_size'] != settings['batch_size']:
        raise ValueError('Runner differs from registered main architecture/budget candidates')
    sources = source_bundle(out)
    provenance = {'learning_manifest_sha256': digest(learning / 'manifest.json'), 'dataset_identity': m['dataset_identity'],
                  'validation_split': 'validation', 'task_scope': 'P3-04-main-common'}
    window_hashes = {str(k): array_hash(w[prepared.arrays['split_code'][w[:, 0]] == SPLIT_CODES['validation']]) for k, w in prepared.windows.items()}
    manifest = {'task_id': 'P3-04', 'created_at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
                'argv': sys.argv, 'python': platform.python_version(), 'platform': platform.platform(),
                'torch': str(torch.__version__), 'numpy': str(np.__version__), 'settings': settings,
                'training_settings_sha256': settings_hash(settings), 'training_config_file_sha256': digest(args.training_config),
                'source_hashes': sources, 'provenance': provenance, 'physical_config': config, 'model_spec': model_spec(config),
                'normalizer_hash': m['normalizer_hash'], 'stats_hash': m['stats_hash'], 'train_rows': rows.tolist(),
                'validation_rows': val_rows.tolist(), 'validation_window_hashes': window_hashes, 'upstream_artifacts': upstream_before,
                'normalization_scope': 'unchanged saved normalizer; loader recomputes train statistics for audit without replacing original files',
                'final_test_used': False, 'MPC': 'NOT_RUN', 'user_understanding': 'NOT_ASSESSED'}
    save(out / 'manifest.json', manifest); manifest_hash = digest(out / 'manifest.json')
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=ROOT, capture_output=True, text=True)
    log = tests.stdout + tests.stderr; (out / 'tests.txt').write_text(log)
    match = re.search(r'Ran (\d+) tests', log); test_count = int(match.group(1)) if match else 0
    if tests.returncode or not test_count or 'skipped=' in log:
        raise ValueError('Mandatory regressions failed before official training')
    print(json.dumps({'progress': 'regression_complete', 'tests': test_count}), flush=True)
    mutated = deepcopy(prepared)
    mutated.arrays['features'][val_rows] += 1e4; mutated.arrays['deltas'][val_rows] -= 1e4
    controls = [fit_fixed_updates(p, normalizer, config, settings['seeds'][0], settings['shuffle_seeds'][0], settings['leakage_probe_updates'], settings['batch_size']) for p in (prepared, mutated)]
    leakage = {'original_train_rows_identical': bool(np.array_equal(controls[0]['rows'], controls[1]['rows'])),
               'fixed_update_weights_identical': tensor_tree_equal(controls[0]['model'].state_dict(), controls[1]['model'].state_dict()),
               'fixed_update_adam_identical': tensor_tree_equal(controls[0]['optimizer'].state_dict(), controls[1]['optimizer'].state_dict()),
               'shuffle_order_identical': controls[0]['order_history'] == controls[1]['order_history']}
    save(out / 'train_only_probe.json', {'checks': leakage, 'validation_rows_modified_in_memory': len(val_rows),
         'diagnostic_updates_per_copy': settings['leakage_probe_updates'], 'scope': 'fixed-update training invariance; validation-selected epoch may legitimately change; original files and normalization unchanged'})
    if not all(leakage.values()):
        raise ValueError('Validation rows entered train updates')
    del controls, mutated
    save_evaluation(out / 'state_hold_validation', evaluate_state_hold(prepared))
    seed_summaries = []
    for seed, shuffle_seed in zip(settings['seeds'], settings['shuffle_seeds']):
        seed_began = perf_counter()
        folder = out / 'seeds' / f'seed-{seed:02d}'; folder.mkdir(parents=True)
        torch.manual_seed(seed); model = DeltaMLP(config); optimizer = make_optimizer(model, config)
        shuffle = torch.Generator(device='cpu').manual_seed(shuffle_seed)
        start_parameters = parameter_vector(model)
        initial_loss = full_loss(model, normalizer, features, targets)
        initial_prediction = infer_normalized(model, normalizer, features)
        save_evaluation(folder / 'initial_untrained_validation', evaluate_predictions(model, normalizer, prepared))
        candidates, updates, witness, best_eval, best = [], 0, None, None, None
        train_s = validation_s = checkpoint_s = 0.
        for epoch in range(1, settings['main_epochs'] + 1):
            tick = perf_counter()
            batch = train_epoch(model, optimizer, normalizer, features, targets, rows, shuffle, settings['batch_size'], config['training']['gradient_clip_norm'])
            epoch_train_s = perf_counter() - tick
            train_s += epoch_train_s; updates += len(batch['batches'])
            if len(batch['batches']) != settings['updates_per_epoch'] or sorted(batch['order']) != rows.tolist():
                raise ValueError('Train epoch skipped, duplicated or included held-out rows')
            post_loss = full_loss(model, normalizer, features, targets)
            tick = perf_counter(); evaluation = evaluate_predictions(model, normalizer, prepared, horizons=(5,))
            epoch_validation_s = perf_counter() - tick
            validation_s += epoch_validation_s
            save_evaluation(folder / 'evaluations' / f'epoch-{epoch:03d}', evaluation)
            candidate = {'epoch': epoch, 'updates': updates, 'selection_score': evaluation.summary['selection_score'],
                         'selection_status': evaluation.summary['selection_status'], 'K5_invalid_windows': evaluation.summary['horizons']['5']['invalid_windows'],
                         'checkpoint': f'checkpoints/epoch-{epoch:03d}', 'validation_summary': evaluation.summary,
                         'post_epoch_train_loss': post_loss, 'batch_diagnostics': batch['batches'], 'original_shuffle_rows': batch['order'],
                         'train_updates_s': epoch_train_s, 'validation_s': epoch_validation_s,
                         'K5_window_rows_sha256': array_hash(evaluation.traces['h5_window_rows']),
                         'weighted_batch_preupdate_loss': sum(b['loss_before_update'] * b['batch_size'] for b in batch['batches']) / len(rows)}
            candidates.append(candidate)
            tick = perf_counter()
            metadata = save_checkpoint(folder / candidate['checkpoint'], model, optimizer, normalizer, config, provenance,
                          epoch=epoch, updates=updates, seed=seed, shuffle_generator=shuffle,
                          selection_score=candidate['selection_score'], training_settings=settings)
            checkpoint_s += perf_counter() - tick
            candidate['checkpoint_hash'] = metadata['checkpoint_hash']
            try:
                new_best = choose_epoch(candidates)
            except ValueError:
                new_best = None
            if new_best is candidate:
                best = candidate; witness = witness_state(model, optimizer, shuffle); best_eval = deepcopy(evaluation)
            with (folder / 'history.jsonl').open('a', encoding='utf-8') as file:
                file.write(json.dumps(candidate, allow_nan=False) + '\n')
            if epoch % 10 == 0:
                print(json.dumps({'progress': 'main_learning', 'seed': seed, 'epoch': epoch, 'updates': updates, 'train_loss': post_loss, 'validation_score': candidate['selection_score']}), flush=True)
        selected = choose_epoch(candidates)
        if selected is not best or witness is None:
            raise ValueError('Selected live witness disagrees with exact selection rule')
        witness_path = folder / 'selected_live_witness.pt'
        with witness_path.open('xb') as file:
            torch.save(witness, file)
        with witness_path.open('rb') as file:
            stored_witness = torch.load(file, map_location='cpu', weights_only=True)
        witness_persisted = tensor_tree_equal(witness, stored_witness)
        save(folder / 'selection.json', {'seed': seed, 'shuffle_seed': shuffle_seed, 'selected_epoch': selected['epoch'],
              'selected_updates': selected['updates'], 'selected_score': selected['selection_score'],
              'selected_checkpoint': selected['checkpoint'], 'selected_checkpoint_hash': selected['checkpoint_hash'],
              'initial_epoch_eligible': False, 'final_test_used': False, 'rule': settings['selection'],
              'scope': 'common starting checkpoint for this seed; not final S1/SK model; selected validation is not independent test performance'})
        selection_hash = digest(folder / 'selection.json')
        selected_bundle = load_common_checkpoint(folder / selected['checkpoint'], normalizer, config, provenance, settings, seed, epoch=selected['epoch'], resume=True)
        restored_K5 = evaluate_predictions(selected_bundle.model, normalizer, prepared, horizons=(5,))
        selected_evaluation = evaluate_predictions(selected_bundle.model, normalizer, prepared)
        save_evaluation(folder / 'selected_validation', selected_evaluation)
        repeat = load_common_checkpoint(folder / selected['checkpoint'], normalizer, config, provenance, settings, seed, epoch=selected['epoch'])
        full_reload = evaluation_equal(selected_evaluation, evaluate_predictions(repeat.model, normalizer, prepared))
        final_bundle = load_common_checkpoint(folder / f'checkpoints/epoch-{settings["main_epochs"]:03d}', normalizer, config, provenance, settings, seed, epoch=settings['main_epochs'], resume=True)
        final_eval = evaluate_predictions(final_bundle.model, normalizer, prepared); save_evaluation(folder / 'final_validation', final_eval)
        physical_reload = torch.equal(predict_delta(model, normalizer, torch.from_numpy(prepared.arrays['current_state'][rows].copy()), torch.from_numpy(prepared.arrays['applied_action'][rows].copy())),
                                      predict_delta(final_bundle.model, normalizer, torch.from_numpy(prepared.arrays['current_state'][rows].copy()), torch.from_numpy(prepared.arrays['applied_action'][rows].copy())))
        resume = replay_selected_checkpoint(folder / selected['checkpoint'], normalizer, config, provenance, settings, witness, features, targets, rows, seed)
        save(folder / 'selected_resume_proof.json', resume)
        all_boundary_counters = all(c['updates'] == c['epoch'] * settings['updates_per_epoch'] for c in candidates)
        window_match = all(array_hash(selected_evaluation.traces[f'h{k}_window_rows']) == window_hashes[str(k)] for k in settings['evaluation_horizons'])
        final_loss = full_loss(model, normalizer, features, targets)
        checks = {'fresh_initialized_main_weights_changed': float((parameter_vector(model) - start_parameters).abs().max()) > 0,
                  'main_predictions_changed': not torch.equal(initial_prediction, infer_normalized(model, normalizer, features)),
                  'all_100_epochs_and_1300_updates': len(candidates) == settings['main_epochs'] and updates == settings['updates_per_seed'],
                  'all_epoch_counter_boundaries': all_boundary_counters,
                  'all_train_batches_finite': all(np.isfinite(b['loss_before_update']) and np.isfinite(b['gradient_norm_before_clip']) for c in candidates for b in c['batch_diagnostics']),
                  'all_100_checkpoints_and_K5_traces_saved': all((folder / c['checkpoint'] / 'checkpoint.json').is_file() and (folder / 'evaluations' / f'epoch-{c["epoch"]:03d}' / 'traces.npz').is_file() for c in candidates),
                  'exact_selected_epoch_matches_rule': choose_epoch(candidates) == selected,
                  'selected_K5_valid_and_matches_live_predictions': selected['K5_invalid_windows'] == 0 and evaluation_equal(best_eval, restored_K5),
                  'selected_K1_K5_K10_full_reload_exact': full_reload,
                  'same_validation_windows_as_Ridge_and_manifest': window_match,
                  'every_epoch_uses_identical_K5_windows': all(c['K5_window_rows_sha256'] == window_hashes['5'] for c in candidates),
                  'final_live_and_saved_weights_and_Adam_exact': tensor_tree_equal(model.state_dict(), final_bundle.model.state_dict()) and tensor_tree_equal(optimizer.state_dict(), final_bundle.optimizer.state_dict()),
                  'final_physical_predictions_reload_exact': physical_reload,
                  'actual_selected_checkpoint_next_update_exact': resume['pass'],
                  'selected_live_witness_persisted_exact': witness_persisted,
                  'selection_frozen_after_diagnostic': digest(folder / 'selection.json') == selection_hash}
        row = {'seed': seed, 'shuffle_seed': shuffle_seed, 'pass': all(checks.values()), 'checks': checks,
               'epochs': len(candidates), 'updates': updates, 'initial_train_loss': initial_loss, 'final_train_loss': final_loss,
               'initial_parameter_sha256': hashlib.sha256(start_parameters.numpy().tobytes()).hexdigest(),
               'selected_epoch': selected['epoch'], 'selected_updates': selected['updates'], 'selected_score': selected['selection_score'],
               'selected_checkpoint': str((folder / selected['checkpoint']).relative_to(out)), 'selected_checkpoint_hash': selected['checkpoint_hash'],
               'selected_live_witness_sha256': digest(witness_path),
               'validation': selected_evaluation.summary, 'resources': {'train_updates_s': train_s, 'validation_s': validation_s, 'checkpoint_save_s': checkpoint_s, 'total_seed_s': perf_counter() - seed_began},
               'selected_resume': resume, 'alignment_outcome': 'NOT_EVALUATED', 'MPC': 'NOT_RUN'}
        save(folder / 'summary.json', row); seed_summaries.append(row)
        print(json.dumps({'progress': 'seed_complete', 'seed': seed, 'pass': row['pass'], 'selected_epoch': row['selected_epoch'], 'selected_K5_score': row['selected_score']}), flush=True)
        if not row['pass']:
            raise ValueError(f'Seed {seed} failed; preserve artifacts and do not omit the seed')
    upstream_after = {str(p.relative_to(ROOT)): verify_artifacts(p) for p in prior_folders}
    checks = {'regression_tests_pass': True, 'original_train_3200_and_validation_1600': len(rows) == 3200 and len(val_rows) == 1600,
              'fixed_update_validation_mask_invariance': all(leakage.values()),
              'all_three_predeclared_seeds_completed': [s['seed'] for s in seed_summaries] == settings['seeds'],
              'total_main_updates_exactly_3900': sum(s['updates'] for s in seed_summaries) == settings['total_updates'],
              'every_seed_common_checkpoint_and_resume_pass': all(s['pass'] for s in seed_summaries),
              'source_frozen_during_run': all(digest(ROOT / name) == sha for name, sha in sources.items()),
              'upstream_data_statistics_Ridge_and_artifacts_unchanged': upstream_before == upstream_after,
              'manifest_frozen_during_run': digest(out / 'manifest.json') == manifest_hash,
              'final_test_not_used': True}
    resources = {'total_wall_time_s': perf_counter() - began, 'main_peak_rss_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024),
                 'training_updates_s': sum(s['resources']['train_updates_s'] for s in seed_summaries),
                 'epoch_validation_s': sum(s['resources']['validation_s'] for s in seed_summaries),
                 'checkpoint_save_s': sum(s['resources']['checkpoint_save_s'] for s in seed_summaries),
                 'device': 'cpu', 'torch_threads': 1, 'stored_artifact_bytes_before_final_hash_index': sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
                 'scope': 'main process only; regression child RSS excluded, BLAS thread count unmeasured; no physical-device real-time claim'}
    summary = {'task_id': 'P3-04', 'pass': all(checks.values()), 'stage_verdict': 'PASS' if all(checks.values()) else 'FAIL',
               'checks': checks, 'test_count': test_count, 'seeds': seed_summaries, 'resources': resources,
               'training_settings_sha256': settings_hash(settings), 'normalizer_hash': m['normalizer_hash'],
               'next_task': 'P3-05', 'MPC': 'NOT_RUN', 'alignment_outcome': 'NOT_EVALUATED', 'user_understanding': 'NOT_ASSESSED',
               'scope': 'verified seed-specific common MLP/Adam/RNG checkpoints; not final S1/SK training, fair closed-loop superiority, final test or personal understanding'}
    save(out / 'resources.json', resources); save(out / 'summary.json', summary)
    if not summary['pass']:
        raise ValueError('P3-04 mandatory check failed')
    return summary


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True)
    parser.add_argument('--learning-data', type=Path, default=DEFAULT_LEARNING)
    parser.add_argument('--training-config', type=Path, default=ROOT / 'configs/p3_training.v1.toml')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.run_id):
        raise ValueError('Unsafe run ID')
    out = ROOT / 'runs' / args.run_id; out.mkdir(parents=True, exist_ok=False)
    try:
        summary = run(args, out)
    except Exception as error:
        save(out / 'failure.json', {'task_id': 'P3-04', 'pass': False, 'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
        save(out / 'artifact_hashes.json', {'sha256': {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*')) if p.is_file()}})
        print(json.dumps({'run': str(out), 'pass': False, 'error': str(error)}), flush=True); return 1
    save(out / 'artifact_hashes.json', {'sha256': {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    print(json.dumps({'run': str(out), 'pass': summary['pass'], 'seeds': len(summary['seeds'])}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
