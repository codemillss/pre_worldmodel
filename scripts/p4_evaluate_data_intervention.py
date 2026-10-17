"""Registered five-arm development prediction evaluation, no new physics/train."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import traceback
from time import perf_counter

import numpy as np
import torch

from push_align.closed_loop import ENGINE_GUARDS, model_planning_scope
from push_align.config import ROOT
from push_align.data import digest
from push_align.data_intervention_evaluation import (ARMS, compare_arms, evaluate_development,
                                                     load_development, regression_geometry, require)
from push_align.learning_data import load_learning_data
from push_align.planning import LearnedPredictor
from push_align.prediction import evaluate_predictions
from p4_train_forks import tree_hash

GIB = 1024**3
DEFAULT_COLLECTION = 'runs/p4-data-main-20261013-01'
DEFAULT_LEARNING = 'runs/p3-02-20261009-01/learning_data'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def required_source_bindings():
    """Root registration may merge these paths with training/candidate sources."""
    paths = ['tests/test_p4_data_intervention_evaluation.py', 'scripts/p4_evaluate_data_intervention.py', 'src/push_align/data_intervention_evaluation.py',
             'src/push_align/data_intervention_training.py', 'src/push_align/dynamics.py',
             'src/push_align/normalization.py', 'src/push_align/representation.py',
             'src/push_align/prediction.py', 'src/push_align/planning.py',
             'src/push_align/data.py', 'src/push_align/coverage.py', 'src/push_align/closed_loop.py',
             'src/push_align/learning_data.py', 'src/push_align/preparation.py', 'scripts/p4_train_forks.py',
             'configs/p4_data_intervention.design.v1.json',
             DEFAULT_COLLECTION+'/registration.json', DEFAULT_COLLECTION+'/artifact_hashes.json',
             DEFAULT_COLLECTION+'/slots_bound.json', DEFAULT_COLLECTION+'/summary.json',
             'reports/p4_data_main_independent_review.json',
             DEFAULT_LEARNING+'/manifest.json', DEFAULT_LEARNING+'/prepared.npz',
             DEFAULT_LEARNING+'/normalizer/normalizer.json', DEFAULT_LEARNING+'/normalizer/statistics.npz']
    return {p: digest(ROOT/p) for p in paths}


def _verify_index(folder):
    index = json.loads((folder/'training_artifact_hashes.json').read_text())['sha256']
    for path, expected in index.items():
        require((folder/path).is_file() and digest(folder/path) == expected, 'Artifact changed: '+str(folder/path))
    return len(index)


def run(registry_path, registry_sha, training_run, output):
    from push_align.data_intervention_training import load_intervention_checkpoint
    start = perf_counter()
    require(digest(registry_path) == registry_sha, 'Evaluation registration differs')
    registry = json.loads(registry_path.read_text())
    bound = registry['bindings_sha256']
    required = required_source_bindings()
    require(all(bound.get(p) == sha for p, sha in required.items()), 'Required evaluation source absent or changed')
    require(all((ROOT/p).is_file() and digest(ROOT/p) == sha for p, sha in bound.items()), 'Execution binding changed')
    training_run, output = Path(training_run), Path(output)
    require(str(training_run.relative_to(ROOT)) == registry['training_run'], 'Unexpected training run')
    require(not output.exists(), 'Never overwrite a prediction evaluation')
    maximum_bytes = registry.get('prediction_evaluation_maximum_bytes', 64*1024**2)
    require(shutil.disk_usage(ROOT).free >= 2*GIB+maximum_bytes, 'Prediction artifact reserve unavailable')
    collection = ROOT/registry.get('collection_run', DEFAULT_COLLECTION)
    learning = ROOT/registry.get('learning_dir', DEFAULT_LEARNING)
    require(collection == ROOT/DEFAULT_COLLECTION and learning == ROOT/DEFAULT_LEARNING, 'Frozen datasets differ')
    training_index_count = _verify_index(training_run)
    main_index = json.loads((collection/'artifact_hashes.json').read_text())['sha256']
    audit = json.loads((ROOT/'reports/p4_data_main_independent_review.json').read_text())
    require(audit.get('status') == 'PASS', 'Independent main collection review not PASS')
    manifest_path = training_run/'manifest.json'
    manifest_sha = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    training_summary = json.loads((training_run/'summary.json').read_text())
    require(training_summary['status'] == 'PASS', 'Training/reproduction incomplete')
    output.mkdir(parents=True)
    save(output/'registration.json', registry)
    save(output/'status.json', {'status': 'RUNNING', 'scope': 'prediction only'})
    data = load_development(collection)
    nm = json.loads((learning/'normalizer/normalizer.json').read_text())['normalizer']
    prepared, normalizer = load_learning_data(learning, nm['schema'], nm['dataset_identity'], nm['normalizer_hash'])
    design = json.loads((ROOT/'configs/p4_data_intervention.design.v1.json').read_text())
    require(normalizer.metadata['normalizer_hash'] == design['model']['normalizer_hash'], 'Normalizer differs')
    config = manifest['physical_config']
    require(config == prepared.metadata['contract']['resolved_config'], 'Learning physical config differs')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    guarded = dict.fromkeys(ENGINE_GUARDS, 0)
    rng_before = torch.get_rng_state().clone()
    results, identities, forwards, regressions = {}, {}, {}, {}
    final_identities = {row['arm']: row for row in training_summary['arms']}
    require(set(final_identities) == set(ARMS), 'Training final arm inventory differs')
    for arm in ARMS:
        path = training_run/'arms'/arm/'checkpoints/update-1000'
        bundle = load_intervention_checkpoint(path, normalizer, config,
            expected_manifest_sha256=manifest_sha, expected_arm=arm,
            expected_dataset_hash=manifest['training_provenance']['arms'][arm]['data_hash'],
            expected_initial_checkpoint_hash=manifest['initial_checkpoint']['checkpoint_hash'])
        require(bundle.metadata['additional_updates'] == 1000 and bundle.metadata['total_updates'] == 2300,
                'Non-final checkpoint requested')
        require(bundle.metadata['arm'] == arm
                and bundle.metadata['dataset_hash'] == manifest['training_provenance']['arms'][arm]['data_hash']
                and bundle.metadata['initial_checkpoint_hash'] == design['model']['initial_checkpoint_hash']
                and bundle.metadata['checkpoint_hash'] == final_identities[arm]['final_checkpoint_hash'],
                'Checkpoint arm/dataset/origin/final identity differs')
        before = tree_hash(bundle.model.state_dict())
        predictor = LearnedPredictor(bundle.model, normalizer)
        with model_planning_scope(guarded):
            result, traces = evaluate_development(predictor, data, config)
            regression = evaluate_predictions(bundle.model, normalizer, prepared, split='validation', horizons=(1, 5, 10))
            geometry = regression_geometry(regression.traces, config)
        require(tree_hash(bundle.model.state_dict()) == before, 'Evaluation mutated weights')
        require(torch.equal(rng_before, torch.get_rng_state()), 'Evaluation mutated Torch RNG')
        arm_dir = output/arm
        arm_dir.mkdir()
        save(arm_dir/'development.json', result)
        np.savez_compressed(arm_dir/'development_traces.npz', **traces)
        save(arm_dir/'original_validation.json', regression.summary)
        save(arm_dir/'original_validation_geometry.json', geometry)
        np.savez_compressed(arm_dir/'original_validation_traces.npz', **regression.traces)
        results[arm] = result
        regressions[arm] = {'horizons': regression.summary['horizons'], 'geometry': geometry,
                            'historical_selection_status_descriptive_only': regression.summary['selection_status']}
        identities[arm] = {'checkpoint_path': str(path.relative_to(ROOT)),
                           'checkpoint_json_sha256': digest(path/'checkpoint.json'),
                           'checkpoint_hash': bundle.metadata['checkpoint_hash'], 'weights_tree_hash': before}
        validation_rows = sum(int(h)*entry['attempted_windows'] for h, entry in regression.summary['horizons'].items())
        forwards[arm] = {'development_calls': predictor.calls, 'development_rows': predictor.rows,
                         'original_validation_nominal_maximum_rows': validation_rows,
                         'validation_row_note': 'Maximum assuming all windows stay valid; unchanged legacy evaluator retains invalid rows but does not count retry forwards'}
        used = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        require(used <= maximum_bytes and shutil.disk_usage(ROOT).free >= 2*GIB, 'Prediction storage guard exceeded')
        save(output/'status.json', {'status': 'RUNNING', 'completed_arms': list(results), 'stored_bytes': used})
    comparison, draws = compare_arms(results)
    save(output/'comparison.json', comparison)
    np.savez_compressed(output/'bootstrap_family_draws.npz', draws=draws)
    primary_complete = all(all(v is not None for v in metrics.values()) for r in results.values()
                           for phase in r['primary'].values() for metrics in phase.values())
    checks = {'all_five_final_models': set(results) == set(ARMS), 'same_development_actual_samples': True,
              'all_twelve_families_retained': all(len(r['families']) == 12 for r in results.values()),
              'normalizer_fixed': normalizer.metadata['normalizer_hash'] == design['model']['normalizer_hash'],
              'model_weights_and_rng_unchanged': True, 'zero_future_engine_calls': all(v == 0 for v in guarded.values()),
              'execution_bindings_unchanged': all(digest(ROOT/p) == sha for p, sha in bound.items()),
              'training_manifest_unchanged': digest(manifest_path) == manifest_sha,
              'development_raw_bindings_unchanged': all(digest(collection/p) == sha for p, sha in data.source_sha256.items())}
    summary = {'schema': 'p4-data-intervention-evaluation-run-v1', 'status': 'PASS' if all(checks.values()) else 'FAIL',
               'created_at_utc': datetime.now(timezone.utc).isoformat(), 'registry_sha256': registry_sha,
               'training_manifest_sha256': manifest_sha, 'training_artifact_count_verified': training_index_count,
               'training_artifact_index_sha256': digest(training_run/'training_artifact_hashes.json'),
               'collection_artifact_index_sha256': digest(collection/'artifact_hashes.json'),
               'normalizer_hash': normalizer.metadata['normalizer_hash'],
               'normalizer_new_data_refit': False, 'original_train_statistics_integrity_recomputation': True,
               'model_identities': identities, 'checks': checks, 'forward_counts': forwards,
               'primary_complete': primary_complete, 'effect_decisions': comparison['directional_support'],
               'execution_PASS_is_not_effect_support': True, 'future_engine_calls': guarded,
               'new_physics_steps': 0, 'learning_updates': 0, 'final_test_used': False,
               'new_closed_loop_execution': False, 'P5_01': 'HOLD', 'personal_understanding': 'NOT_ASSESSED',
               'wall_time_s': perf_counter()-start, 'maximum_output_bytes': maximum_bytes,
               'scope': 'Development prediction and old validation regression; source99 candidate analysis is a separate artifact; no superiority/task success claim'}
    save(output/'summary.json', summary)
    save(output/'status.json', {'status': summary['status'], 'completed_arms': list(ARMS)})
    save(output/'artifact_hashes.json', {'sha256': {str(p.relative_to(output)): digest(p)
         for p in sorted(output.rglob('*')) if p.is_file()}})
    require(all(checks.values()), 'Evaluation integrity failed; artifacts retained')
    print(json.dumps({'status': summary['status'], 'primary_complete': primary_complete,
                      'effect_decisions': summary['effect_decisions'], 'wall_time_s': summary['wall_time_s']}))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', required=True, type=Path)
    parser.add_argument('--registry-sha', required=True)
    parser.add_argument('--training-run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    existed_before = args.output.exists()
    try:
        run(args.registry, args.registry_sha, args.training_run, args.output)
    except Exception as error:
        # Never write into an output directory that existed before this attempt.
        if not existed_before and args.output.is_dir():
            save(args.output/'failure.json', {'status': 'FAIL', 'error': str(error),
                'traceback': traceback.format_exc(), 'automatic_retry': False})
            save(args.output/'status.json', {'status': 'FAIL', 'error': str(error)})
            save(args.output/'artifact_hashes.json', {'sha256': {str(p.relative_to(args.output)): digest(p)
                for p in sorted(args.output.rglob('*')) if p.is_file() and p.name != 'artifact_hashes.json'}})
        raise
