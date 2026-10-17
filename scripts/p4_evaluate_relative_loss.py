"""Registered C0/R1 fixed-D11 final prediction comparison, no new physics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from time import perf_counter
import traceback

import numpy as np
import torch

from push_align.closed_loop import ENGINE_GUARDS, model_planning_scope
from push_align.config import ROOT
from push_align.data import digest
from push_align.data_intervention_evaluation import load_development, regression_geometry, require
from push_align.learning_data import load_learning_data
from push_align.planning import LearnedPredictor
from push_align.prediction import evaluate_predictions
from push_align.relative_loss_evaluation import (ARMS, BOOTSTRAP_SEED,
    compare_relative_arms, evaluate_relative_development)
from push_align.data_intervention_training import tree_hash

GIB = 1024**3
DEFAULT_COLLECTION = 'runs/p4-data-main-20261013-01'
DEFAULT_LEARNING = 'runs/p3-02-20261009-01/learning_data'
HISTORICAL_EVALUATION = 'runs/p4-data-retrain-20261013-02/prediction_evaluation'
HISTORICAL_TRAINING = 'runs/p4-data-retrain-20261013-02'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def required_source_bindings():
    paths = ['tests/test_p4_relative_loss_evaluation.py', 'scripts/p4_evaluate_relative_loss.py',
        'src/push_align/relative_loss_evaluation.py', 'src/push_align/relative_loss_training.py',
        'src/push_align/data_intervention_evaluation.py', 'src/push_align/data_intervention_training.py',
        'src/push_align/dynamics.py', 'src/push_align/normalization.py', 'src/push_align/representation.py',
        'src/push_align/prediction.py', 'src/push_align/planning.py', 'src/push_align/data.py',
        'src/push_align/coverage.py', 'src/push_align/closed_loop.py', 'src/push_align/learning_data.py',
        'src/push_align/preparation.py', 'configs/p4_data_intervention.design.v1.json',
        DEFAULT_COLLECTION+'/registration.json', DEFAULT_COLLECTION+'/artifact_hashes.json',
        DEFAULT_COLLECTION+'/slots_bound.json', DEFAULT_COLLECTION+'/summary.json',
        'reports/p4_data_main_independent_review.json', DEFAULT_LEARNING+'/manifest.json',
        DEFAULT_LEARNING+'/prepared.npz', DEFAULT_LEARNING+'/normalizer/normalizer.json',
        DEFAULT_LEARNING+'/normalizer/statistics.npz', HISTORICAL_EVALUATION+'/artifact_hashes.json',
        HISTORICAL_TRAINING+'/manifest.json', HISTORICAL_TRAINING+'/training_artifact_hashes.json']
    paths += [HISTORICAL_EVALUATION+'/D11/'+name for name in (
        'development.json', 'development_traces.npz', 'original_validation.json',
        'original_validation_geometry.json', 'original_validation_traces.npz')]
    return {p: digest(ROOT/p) for p in paths}


def verify_index(folder, name):
    index = json.loads((folder/name).read_text())['sha256']
    for relative, expected in index.items():
        require((folder/relative).is_file() and digest(folder/relative) == expected,
                'Artifact changed: '+str(folder/relative))
    return len(index)


def exact_arrays(actual, path):
    with np.load(path, allow_pickle=False) as saved:
        require(set(actual) == set(saved.files), 'Historical trace key inventory differs')
        checks = {}
        for name in sorted(actual):
            a,b = actual[name], saved[name]
            same = a.dtype == b.dtype and a.shape == b.shape
            if same:
                same = np.array_equal(a,b,equal_nan=True) if a.dtype.kind in 'fc' else np.array_equal(a,b)
            checks[name] = bool(same)
    return checks


def historical_control_check(legacy, traces, regression, geometry, *, folder):
    """Exact C0 recurrence is required, independent of apparent R1 gains."""
    checks = {'development_legacy_json_exact': legacy == json.loads((folder/'development.json').read_text()),
              'original_validation_json_exact': regression.summary == json.loads((folder/'original_validation.json').read_text()),
              'original_validation_geometry_exact': geometry == json.loads((folder/'original_validation_geometry.json').read_text())}
    arrays = {'development_traces': exact_arrays(traces,folder/'development_traces.npz'),
              'original_validation_traces': exact_arrays(regression.traces,folder/'original_validation_traces.npz')}
    result = {'status': 'PASS' if all(checks.values()) and all(all(v.values()) for v in arrays.values()) else 'FAIL',
              'json_checks': checks, 'array_checks': arrays, 'exact_values_dtype_shape_with_equal_nan': True,
              'scope': 'β0 new-run predictions exactly reproduce historical D11; NPZ container bytes are not the equality target'}
    return result


def run(registry_path, registry_sha, training_run, output):
    from push_align.relative_loss_training import load_relative_checkpoint
    start = perf_counter()
    require(digest(registry_path) == registry_sha, 'Evaluation registration differs')
    registry = json.loads(registry_path.read_text())
    bound = registry['bindings_sha256']
    require(all(bound.get(p) == sha for p,sha in required_source_bindings().items()),
            'Required evaluation source absent or changed')
    require(all((ROOT/p).is_file() and digest(ROOT/p) == sha for p,sha in bound.items()), 'Execution binding changed')
    training_run, output = Path(training_run).resolve(), Path(output).resolve()
    require(str(training_run.relative_to(ROOT)) == registry['training_run'], 'Unexpected training run')
    require(output == training_run/'prediction_evaluation' and not output.exists(), 'Never overwrite or relocate prediction evaluation')
    maximum = registry.get('prediction_evaluation_maximum_bytes',64*1024**2)
    require(shutil.disk_usage(ROOT).free >= 2*GIB+maximum, 'Prediction artifact reserve unavailable')
    require(registry.get('collection_run',DEFAULT_COLLECTION) == DEFAULT_COLLECTION
            and registry.get('learning_dir',DEFAULT_LEARNING) == DEFAULT_LEARNING
            and registry.get('historical_evaluation_run',HISTORICAL_EVALUATION) == HISTORICAL_EVALUATION,
            'Frozen data/history path differs')
    collection, learning = ROOT/DEFAULT_COLLECTION, ROOT/DEFAULT_LEARNING
    training_index_count = verify_index(training_run,'training_artifact_hashes.json')
    verify_index(ROOT/HISTORICAL_EVALUATION,'artifact_hashes.json')
    require(json.loads((ROOT/'reports/p4_data_main_independent_review.json').read_text())['status']=='PASS', 'Main collection independent audit not PASS')
    manifest_path = training_run/'manifest.json'; manifest_sha = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    train_summary = json.loads((training_run/'summary.json').read_text())
    require(train_summary['status']=='PASS', 'Training not complete')
    final = {row['branch']:row for row in train_summary['arms']}
    require(set(final)==set(ARMS), 'Both final branches mandatory')
    output.mkdir(); save(output/'registration.json',registry)
    save(output/'status.json',{'status':'RUNNING','scope':'prediction only'})
    data = load_development(collection)
    nm=json.loads((learning/'normalizer/normalizer.json').read_text())['normalizer']
    prepared,normalizer=load_learning_data(learning,nm['schema'],nm['dataset_identity'],nm['normalizer_hash'])
    design=json.loads((ROOT/'configs/p4_data_intervention.design.v1.json').read_text())
    require(normalizer.metadata['normalizer_hash']==design['model']['normalizer_hash'],'Normalizer differs')
    config=manifest['physical_config']
    require(config==prepared.metadata['contract']['resolved_config'],'Physical config differs')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    guarded=dict.fromkeys(ENGINE_GUARDS,0); rng_before=torch.get_rng_state().clone()
    results,identities,forwards,regressions={}, {}, {}, {}
    control_check=None
    for arm in ARMS:
        path=training_run/final[arm]['final_checkpoint']
        require(path==training_run/'arms'/arm/'checkpoints/update-1000', 'Only final1000 checkpoint allowed')
        dataset_hash=manifest['training_provenance']['arms'][arm]['data_hash']
        bundle=load_relative_checkpoint(path,normalizer,config,expected_manifest_sha256=manifest_sha,
            expected_branch=arm,expected_dataset_hash=dataset_hash,
            expected_initial_checkpoint_hash=manifest['initial_checkpoint']['checkpoint_hash'])
        require(bundle.metadata['additional_updates']==1000 and bundle.metadata['total_updates']==2300
                and bundle.metadata['relative_weight']==(0. if arm=='C0' else .2), 'Final loss/update contract differs')
        require(bundle.metadata['checkpoint_hash']==final[arm]['final_checkpoint_hash']
                and bundle.metadata['initial_checkpoint_hash']==design['model']['initial_checkpoint_hash'], 'Final checkpoint identity differs')
        before=tree_hash(bundle.model.state_dict()); predictor=LearnedPredictor(bundle.model,normalizer)
        with model_planning_scope(guarded):
            result,legacy,traces=evaluate_relative_development(predictor,data,config)
            regression=evaluate_predictions(bundle.model,normalizer,prepared,split='validation',horizons=(1,5,10))
            geometry=regression_geometry(regression.traces,config)
        require(tree_hash(bundle.model.state_dict())==before and torch.equal(rng_before,torch.get_rng_state()),
                'Evaluation changed weights or Torch RNG')
        arm_dir=output/arm; arm_dir.mkdir()
        save(arm_dir/'development.json',result); save(arm_dir/'development_legacy.json',legacy)
        np.savez_compressed(arm_dir/'development_traces.npz',**traces)
        save(arm_dir/'original_validation.json',regression.summary)
        save(arm_dir/'original_validation_geometry.json',geometry)
        np.savez_compressed(arm_dir/'original_validation_traces.npz',**regression.traces)
        if arm=='C0':
            control_check=historical_control_check(legacy,traces,regression,geometry,folder=ROOT/HISTORICAL_EVALUATION/'D11')
            save(output/'historical_control_reproduction.json',control_check)
            require(control_check['status']=='PASS','New C0 predictions did not exactly reproduce historical D11; no effect promotion')
        results[arm]=result
        identities[arm]={'checkpoint_path':str(path.relative_to(ROOT)), 'checkpoint_json_sha256':digest(path/'checkpoint.json'),
            'checkpoint_hash':bundle.metadata['checkpoint_hash'],'weights_tree_hash':before,'dataset_hash':dataset_hash,
            'relative_weight':bundle.metadata['relative_weight']}
        regressions[arm]={'horizons':regression.summary['horizons'],'geometry':geometry,
            'historical_selection_status_descriptive_only':regression.summary['selection_status']}
        nominal=sum(int(h)*entry['attempted_windows'] for h,entry in regression.summary['horizons'].items())
        forwards[arm]={'development_calls':predictor.calls,'development_rows':predictor.rows,
            'original_validation_nominal_maximum_rows':nominal,
            'validation_row_note':'Unchanged legacy evaluator retains invalid denominators; nominal rows are not exact retry-call counts'}
        used=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        require(used<=maximum and shutil.disk_usage(ROOT).free>=2*GIB,'Prediction storage guard exceeded')
        save(output/'status.json',{'status':'RUNNING','completed_arms':list(results),'stored_bytes':used})
    comparison,draws=compare_relative_arms(results,bootstrap_seed=BOOTSTRAP_SEED)
    save(output/'comparison.json',comparison); np.savez_compressed(output/'bootstrap_family_draws.npz',draws=draws)
    checks={'both_final_models':set(results)==set(ARMS), 'strict_control_match':control_check['status']=='PASS',
        'same_D11_data_hash':len({identities[a]['dataset_hash'] for a in ARMS})==1,
        'same_development_actual_samples':True,'all_twelve_families_retained':all(len(r['families'])==12 for r in results.values()),
        'weights_and_rng_unchanged':True,'zero_future_engine_calls':all(v==0 for v in guarded.values()),
        'normalizer_fixed':normalizer.metadata['normalizer_hash']==design['model']['normalizer_hash'],
        'execution_bindings_unchanged':all(digest(ROOT/p)==sha for p,sha in bound.items()),
        'training_manifest_unchanged':digest(manifest_path)==manifest_sha,
        'development_raw_bindings_unchanged':all(digest(collection/p)==sha for p,sha in data.source_sha256.items())}
    summary={'schema':'p4-relative-loss-evaluation-run-v1','status':'PASS' if all(checks.values()) else 'FAIL',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'registry_sha256':registry_sha,
        'training_manifest_sha256':manifest_sha,'training_artifact_count_verified':training_index_count,
        'training_artifact_index_sha256':digest(training_run/'training_artifact_hashes.json'),
        'normalizer_hash':normalizer.metadata['normalizer_hash'],'normalizer_new_data_refit':False,
        'model_identities':identities,'checks':checks,'forward_counts':forwards,
        'effect_decisions':comparison['effect_decisions'],'future_engine_calls':guarded,
        'new_physics_steps':0,'learning_updates':0,'final_test_used':False,'new_closed_loop_execution':False,
        'P5_01':'HOLD','personal_understanding':'NOT_ASSESSED','execution_PASS_is_not_effect_support':True,
        'wall_time_s':perf_counter()-start,'maximum_output_bytes':maximum,
        'scope':'Fixed known development relative/gap prediction and unchanged old validation; no task-success, generalization or uniquely identified mechanism claim'}
    save(output/'regression_comparison.json',regressions); save(output/'summary.json',summary)
    save(output/'status.json',{'status':summary['status'],'completed_arms':list(ARMS)})
    save(output/'artifact_hashes.json',{'sha256':{str(p.relative_to(output)):digest(p)
        for p in sorted(output.rglob('*')) if p.is_file()}})
    require(all(checks.values()),'Evaluation integrity failed; artifacts retained')
    print(json.dumps({'status':summary['status'],'effect_decisions':summary['effect_decisions'],'wall_time_s':summary['wall_time_s']}))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--registry',required=True,type=Path)
    parser.add_argument('--registry-sha',required=True); parser.add_argument('--training-run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path); args=parser.parse_args()
    existed_before=args.output.exists()
    try: run(args.registry,args.registry_sha,args.training_run,args.output)
    except Exception as error:
        if not existed_before and args.output.is_dir():
            save(args.output/'failure.json',{'status':'FAIL','error':str(error),'traceback':traceback.format_exc(),'automatic_retry':False})
            save(args.output/'status.json',{'status':'FAIL','error':str(error)})
            save(args.output/'artifact_hashes.json',{'sha256':{str(p.relative_to(args.output)):digest(p)
                for p in sorted(args.output.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'}})
        raise
