"""Exclusive E1 seals and read-only adoption of a complete frozen result.

Seals must be created by the coordinator at the original writer callbacks:
  predictions_frozen -> seal_prediction_bundle(run,event['choices_sha256'])
  repeat_complete -> seal_repeat_receipt(run,id,sha(repeat/summary.json))
  stage_complete -> seal_complete_e1(run,event['summary_sha256'])
A later read never invents a missing seal or retries a partial E1 stage.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

from push_align import final_recording_columns as codec
from push_align.contact_geometry import check_contact_geometry
from push_align.final_driver import exact_comparison, read_full_trace, typed_digest
from push_align.final_manifest import expected_E1
from push_align.final_recording import file_sha256
from push_align.planning import CostBatch, PlannerContext, PredictionBatch, choose, score
from push_align.reproduction import json_safe
from p5_final_e1 import COMPONENT_NAMES, H, MODELS, N, REPEATS, _save


class E1AdoptionError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise E1AdoptionError(message)


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise E1AdoptionError(f'Missing/malformed E1 evidence: {path}') from error


def _same(left, right):
    return json_safe(left) == json_safe(right)


def _hash(path, expected):
    _require(isinstance(expected, str) and len(expected) == 64 and Path(path).is_file()
             and file_sha256(path) == expected, 'Missing/changed sealed evidence: '+str(path))


def _manifest(run):
    run = Path(run).resolve()
    receipt = _read(run/'freeze_receipt.json')
    _hash(run/'manifest.json', receipt['sha256'])
    manifest = _read(run/'manifest.json')
    _require(manifest.get('schema') == 'p5-frozen-final-manifest-v1'
             and manifest.get('status') == 'FROZEN_BEFORE_FINAL_OUTCOMES'
             and manifest['E1'] == expected_E1(manifest['families']) and len(manifest['E1']) == 12,
             'An exact frozen12-snapshot E1 manifest is required')
    return run, manifest, receipt['sha256']


def _repeat(run, manifest, repeat_id):
    for spec in manifest['E1']:
        for candidate in range(N):
            for repeat in range(REPEATS):
                expected = spec['snapshot_id']+f'/candidate-{candidate:02d}/repeat-{repeat}'
                if repeat_id == expected:
                    folder = run/'E1/snapshots'/spec['snapshot_id']/'physical'/f'candidate-{candidate:02d}'/f'repeat-{repeat}'
                    return spec, candidate, repeat, folder
    raise E1AdoptionError('Unknown/out-of-range E1 repeat ID: '+str(repeat_id))


def _repeat_seal_path(run, repeat_id):
    return run/'E1/receipts/repeats'/(hashlib.sha256(repeat_id.encode()).hexdigest()+'.json')


def _validate_file_map(run, files, *, deep):
    _require(isinstance(files, dict) and bool(files), 'Nonempty E1 file hashes required')
    actual_bytes = 0
    for path, digest in files.items():
        saved = Path(path).resolve()
        _require(saved.is_relative_to(run/'E1') and saved.is_file(), 'Missing/escaping E1 evidence')
        if deep:
            _hash(saved, digest)
        actual_bytes += saved.stat().st_size
    return actual_bytes


def _prediction_paths(run, manifest):
    stage = run/'E1'
    expected = {str(stage/'choices_before_any_physical_future.json')}
    for spec in manifest['E1']:
        folder = stage/'snapshots'/spec['snapshot_id']
        for model in MODELS:
            expected.update(str(folder/'models'/model/name)
                            for name in ('trace.npz', 'choice_before_physics.json'))
    return expected


def _check_predictions(run, manifest):
    stage = run/'E1'
    index = _read(stage/'prediction_files_before_any_physical_future.json')
    _require(set(index) == _prediction_paths(run, manifest), 'Exact84 model trace/choice pairs required')
    _validate_file_map(run, index, deep=True)
    all_choices = _read(stage/'choices_before_any_physical_future.json')
    expected_pairs = {(spec['snapshot_id'], model) for spec in manifest['E1'] for model in MODELS}
    pairs = [(row['snapshot_id'], row['model']) for row in all_choices]
    _require(len(pairs) == 84 and len(set(pairs)) == 84 and set(pairs) == expected_pairs,
             'Choice denominator must be exactly12 snapshots*7 models')
    by = {(row['snapshot_id'], row['model']): row for row in all_choices}
    bound = dict(index)
    bound[str(stage/'prediction_files_before_any_physical_future.json')] = file_sha256(stage/'prediction_files_before_any_physical_future.json')
    bound[str(stage/'manifest.json')] = file_sha256(stage/'manifest.json')
    execution = _read(stage/'manifest.json')
    _require(execution['frozen_contract_hash'] == typed_digest(manifest)
             and execution['frozen_E1_entries'] == manifest['E1']
             and execution['candidate_seed_base'] == 6811 and execution['candidate_stream_id'] == 1,
             'E1 execution manifest differs from frozen scope')
    for spec in manifest['E1']:
        folder = stage/'snapshots'/spec['snapshot_id']
        info = _read(folder/'input.json')
        with np.load(folder/'candidates.npz', allow_pickle=False) as source:
            actions = source['actions'].copy()
            _require(actions.dtype == np.float64 and actions.shape == (N, H, 2), 'Malformed N16/H5 candidates')
        _require(info['spec'] == spec and info['warm_sequence'] is None
                 and info['candidate_hash'] == typed_digest(actions), 'Candidate/input intervention differs')
        for name in ('input.json', 'candidates.npz'):
            bound[str(folder/name)] = file_sha256(folder/name)
        for model in MODELS:
            destination = folder/'models'/model
            choice = _read(destination/'choice_before_physics.json')
            _require(choice == by[(spec['snapshot_id'], model)], 'Choice index differs from individual frozen choice')
            with np.load(destination/'trace.npz', allow_pickle=False) as source:
                arrays = {key: source[key].copy() for key in source.files}
            _require(arrays['predicted_states'].dtype == np.float64 and arrays['predicted_states'].shape == (N, H+1, 10)
                     and arrays['raw_deltas'].dtype == np.float64 and arrays['raw_deltas'].shape == (N, H, 10)
                     and arrays['valid'].dtype == bool and arrays['valid'].shape == (N,)
                     and arrays['reasons'].dtype.kind in 'US' and arrays['reasons'].shape == (N,)
                     and arrays['failure_steps'].dtype.kind in 'iu' and arrays['failure_steps'].shape == (N,)
                     and arrays['total_cost'].dtype == np.float64 and arrays['total_cost'].shape == (N,),
                     'Full invalid/cost/prediction candidate denominator malformed')
            predicted = PredictionBatch(arrays['predicted_states'], arrays['raw_deltas'], arrays['valid'],
                                        tuple(arrays['reasons']), arrays['failure_steps'])
            costs = CostBatch(arrays['total_cost'], {key: arrays['cost_'+key] for key in COMPONENT_NAMES})
            selected, action, reason = choose(costs, predicted, actions)
            _require(exact_comparison(arrays['candidates'], actions)['status'] == 'PASS_EXACT'
                     and choice['candidate_hash'] == typed_digest(actions)
                     and choice['input_hash'] == info['input_hash']
                     and choice['prediction_hash'] == typed_digest(predicted.states)
                     and choice['cost_hash'] == typed_digest(costs.total)
                     and choice['selected_index'] == selected and choice['reason'] == reason
                     and exact_comparison(np.asarray(choice['selected_action'], np.float64), action)['status'] == 'PASS_EXACT'
                     and choice['previous_applied_action_hash'] == typed_digest(np.asarray(info['previous_applied_action'], np.float64))
                     and choice['warm_sequence'] is None
                     and choice['candidate_rng_keys'] == [6811, spec['episode_index'], spec['tick'], 1],
                     'Frozen model-only selection/input/RNG does not regenerate')
    return bound


def seal_prediction_bundle(run, expected_choices_sha256):
    run, manifest, manifest_sha = _manifest(run)
    target = run/'E1/receipts/predictions.json'
    if target.exists():
        raise FileExistsError(target)
    _require(not any((run/'E1').glob('snapshots/**/physical/candidate-*')),
             'Cannot backdate a prediction seal after a physical attempt began')
    _hash(run/'E1/choices_before_any_physical_future.json', expected_choices_sha256)
    files = _check_predictions(run, manifest)
    receipt = {'schema': 'p5-E1-prediction-seal-v1', 'frozen_manifest_sha256': manifest_sha,
               'choices_sha256': expected_choices_sha256, 'files_sha256': files,
               'prediction_sets': 84, 'physical_future_started': False}
    path = run/'E1/receipts/predictions.json'
    _save(path, receipt)
    return {'path': str(path), 'sha256': file_sha256(path), **receipt}


def _prediction_seal(run, manifest_sha, *, deep):
    path = run/'E1/receipts/predictions.json'
    receipt = _read(path)
    _require(receipt['schema'] == 'p5-E1-prediction-seal-v1'
             and receipt['frozen_manifest_sha256'] == manifest_sha and receipt['prediction_sets'] == 84,
             'Missing/wrong before-future prediction seal')
    if deep:
        _validate_file_map(run, receipt['files_sha256'], deep=True)
    else:
        _require(isinstance(receipt.get('files_sha256'), dict) and bool(receipt['files_sha256']), 'Prediction seal files missing')
    return path, receipt


def _source_input(run, manifest, spec):
    folder = run/'E1/snapshots'/spec['snapshot_id']
    info = _read(folder/'input.json')
    expected_source = run/'E2'/spec['family_id']/'G/manifest.json'
    _require(Path(info['source_manifest']).resolve() == expected_source, 'E1 source is another G family')
    _hash(expected_source, info['source_manifest_sha256'])
    capture, arrays, metadata = read_full_trace(info['source_full_input'])
    _require(not arrays and metadata['plan'] is None and capture['tick'] == spec['tick']
             and capture['episode_index'] == spec['episode_index'] and capture['controller_id'] == 'G'
             and capture['case_id'] == spec['family_id'], 'Raw prefix source is another fixed G input')
    previous = capture['planner_context'].previous_applied_action.copy()
    context = PlannerContext(previous, None, spec['tick'])
    _require(info['warm_sequence'] is None and info['previous_applied_action'] == previous.tolist()
             and info['input_hash'] == typed_digest({'full_actual_G_capture': capture, 'shared_context': context}),
             'Shared input/warm/previous actual action differs')
    return capture, previous


def _raw_row(run, manifest, spec, candidate, row):
    receipt = row['raw_receipt']
    initial, results = codec.read_raw(receipt['prefix'] if 'prefix' in receipt else row['_prefix'],
                                       expected_sha256=receipt['files_sha256'])
    capture, previous = _source_input(run, manifest, spec)
    _require(exact_comparison(initial, capture['snapshot'])['status'] == 'PASS_EXACT', 'Raw initial full snapshot differs')
    folder = run/'E1/snapshots'/spec['snapshot_id']
    with np.load(folder/'candidates.npz', allow_pickle=False) as source:
        actions = source['actions'][candidate].copy()
    states = np.full((H+1, 10), np.nan, np.float64)
    applied = np.full((H, 2), np.nan, np.float64)
    states[0] = capture['observation'].state
    valid, completed = len(results) == H, 0
    physical, profile = manifest['physical_config'], manifest['profile']
    tolerance = physical['validity']['dt_match_tolerance_s']
    for index, step in enumerate(results):
        _require(np.array_equal(step.requested_action, actions[index]), 'Raw requested command differs from frozen candidate')
        states[index+1] = step.observation.state
        applied[index] = step.applied_action
        full_interval = math.isfinite(step.actual_dt) and abs(step.actual_dt-physical['physics']['action_dt_s']) <= tolerance
        completed += int(full_interval)
        geometry = check_contact_geometry(step.diagnostics['samples'], profile['physics_suite']['actual_contact_penetration_max_m'])
        valid = bool(valid and full_interval and step.transition_valid and step.observation.observation_valid
                     and np.isfinite(step.observation.state).all() and np.array_equal(step.applied_action, actions[index])
                     and geometry['pass'])
    observed = PredictionBatch(states[None], np.diff(states, axis=0)[None], np.array([valid]), ('',), np.array([-1], np.int64))
    planner = deepcopy(manifest['planner_config'])
    planner['planner']['candidate_count'] = N
    costs = score(observed, actions[None], capture['task_spec'], previous, planner)
    cost = float(costs.total[0])
    if valid and not math.isfinite(cost):
        valid = False
    _require(row['state_hash'] == typed_digest(states) and row['applied_action_hash'] == typed_digest(applied)
             and row['actual_returned_intervals'] == len(results) and row['completed_intervals'] == completed
             and row['physical_valid'] is valid and row['cost'] == (cost if valid else None),
             'Actual raw prefix/state/action/validity/cost does not regenerate')
    for key in COMPONENT_NAMES:
        left, right = float(row['cost_components'][key]), float(costs.components[key][0])
        _require(struct.pack('=d', left) == struct.pack('=d', right), 'Actual cost component differs: '+key)
    return {'state_hash': typed_digest(states), 'applied_action_hash': typed_digest(applied),
            'valid': valid, 'cost': cost, 'completed_intervals': completed,
            'intervals': len(results), 'cost_components': {key: float(costs.components[key][0]) for key in COMPONENT_NAMES}}


def _inspect_repeat(run, manifest, repeat_id, *, deep):
    spec, candidate, repeat, folder = _repeat(run, manifest, repeat_id)
    path = folder/'summary.json'
    row = _read(path)
    expected = {'repeat_id': repeat_id, 'snapshot_id': spec['snapshot_id'], 'family_id': spec['family_id'],
                'candidate_index': candidate, 'repeat_index': repeat, 'reserved_raw_records': 1, 'reserved_intervals': H,
                'status': 'COMPLETE_RECORDED', 'initial_snapshot_exact': True}
    _require(all(row.get(key) == value for key, value in expected.items()), 'E1 repeat summary is not this completed identity')
    _require(type(row.get('physical_valid')) is bool and type(row.get('actual_returned_intervals')) is int
             and 1 <= row['actual_returned_intervals'] <= H
             and type(row.get('completed_intervals')) is int and 0 <= row['completed_intervals'] <= row['actual_returned_intervals'],
             'Invalid physical prefix counts/flags')
    receipt = row['raw_receipt']
    prefix = folder/'raw_execution'
    expected_paths = {str(path) for path in codec.raw_paths(prefix)}
    _require(receipt['schema'] == codec.SCHEMA and receipt['interval_count'] == row['actual_returned_intervals']
             and set(receipt['files_sha256']) == expected_paths and set(receipt['bytes']) == expected_paths,
             'Full lossless raw pair/schema is missing or belongs to another repeat')
    raw_bytes = _validate_file_map(run, receipt['files_sha256'], deep=deep)
    _require(raw_bytes == sum(receipt['bytes'].values())
             and all(Path(path).stat().st_size == size for path, size in receipt['bytes'].items()), 'Raw recorded byte size changed')
    evidence = {'summary': row, 'summary_path': str(path), 'summary_sha256': file_sha256(path),
                'raw_bytes': raw_bytes, 'actual_bytes': raw_bytes+path.stat().st_size}
    if deep:
        reconstruction_row = {**row, '_prefix': str(prefix)}
        evidence['raw_regeneration'] = _raw_row(run, manifest, spec, candidate, reconstruction_row)
    return evidence


def seal_repeat_receipt(run, repeat_id, expected_summary_sha256):
    run, manifest, manifest_sha = _manifest(run)
    target = _repeat_seal_path(run, repeat_id)
    if target.exists():
        raise FileExistsError(target)
    prediction_path, prediction = _prediction_seal(run, manifest_sha, deep=False)
    _, _, _, folder = _repeat(run, manifest, repeat_id)
    _hash(folder/'summary.json', expected_summary_sha256)
    inspected = _inspect_repeat(run, manifest, repeat_id, deep=True)
    receipt = {'schema': 'p5-E1-repeat-seal-v1', 'repeat_id': repeat_id,
        'frozen_manifest_sha256': manifest_sha, 'prediction_seal_sha256': file_sha256(prediction_path),
        'summary_path': inspected['summary_path'], 'summary_sha256': expected_summary_sha256,
        'raw_files_sha256': inspected['summary']['raw_receipt']['files_sha256'],
        'raw_bytes': inspected['raw_bytes'], 'actual_bytes': inspected['actual_bytes'],
        'raw_regeneration': inspected['raw_regeneration']}
    path = _repeat_seal_path(run, repeat_id)
    _save(path, receipt)
    return {'path': str(path), 'sha256': file_sha256(path), **receipt}


def verified_repeat_receipt(run, repeat_id, deep=True):
    run, manifest, manifest_sha = _manifest(run)
    path = _repeat_seal_path(run, repeat_id)
    receipt = _read(path)
    _require(receipt['schema'] == 'p5-E1-repeat-seal-v1' and receipt['repeat_id'] == repeat_id
             and receipt['frozen_manifest_sha256'] == manifest_sha, 'Missing/wrong E1 repeat seal')
    prediction_path, prediction = _prediction_seal(run, manifest_sha, deep=False)
    _hash(prediction_path, receipt['prediction_seal_sha256'])
    _hash(receipt['summary_path'], receipt['summary_sha256'])
    inspected = _inspect_repeat(run, manifest, repeat_id, deep=deep)
    _require(receipt['summary_path'] == inspected['summary_path'] and receipt['summary_sha256'] == inspected['summary_sha256']
             and receipt['raw_files_sha256'] == inspected['summary']['raw_receipt']['files_sha256']
             and receipt['raw_bytes'] == inspected['raw_bytes'] and receipt['actual_bytes'] == inspected['actual_bytes'],
             'E1 seal differs from current completed evidence')
    if deep:
        _require(_same(receipt['raw_regeneration'], inspected['raw_regeneration']), 'Raw regeneration changed from its seal')
    return {**inspected, 'seal_path': str(path), 'seal_sha256': file_sha256(path),
            'verification': 'full raw hash and pure cost/state regeneration' if deep else 'sealed identity/summary hash and current raw sizes; raw bytes not rehashed'}


def _complete_files(run, manifest, summary):
    from push_align.candidate_diagnostics import compare_candidates
    stage = run/'E1'
    _require(summary.get('status') == 'COMPLETE_RECORDED' and summary.get('expected_snapshots') == manifest['E1']
             and summary.get('completed_snapshots') == 12 and summary.get('completed_prediction_sets') == 84
             and summary.get('diagnostic_count') == 84 and summary.get('completed_physical_repeats') == 576
             and summary.get('attempted_physical_repeats') == 576 and summary.get('actual_future_intervals', 2881) <= 2880
             and summary.get('all_sources_and_model_weights_unchanged') is True
             and not any(summary['future_engine_calls_during_model_prediction'].values()), 'E1 stage denominator or scientific checks incomplete')
    prediction_path, prediction = _prediction_seal(run, file_sha256(run/'manifest.json'), deep=True)
    files = {str(prediction_path): file_sha256(prediction_path), **prediction['files_sha256']}
    ledger = _read(stage/'comparison_ledger.json')
    by = {(row['snapshot_id'], row['model']): row for row in ledger}
    expected_pairs = {(spec['snapshot_id'], model) for spec in manifest['E1'] for model in MODELS}
    _require(len(ledger) == len(by) == 84 and set(by) == expected_pairs, 'All84 diagnostic rows are required')
    completed_ids, returned = [], 0
    for spec in manifest['E1']:
        folder = stage/'snapshots'/spec['snapshot_id']
        with np.load(folder/'actual_futures.npz', allow_pickle=False) as source:
            arrays = {key: source[key].copy() for key in source.files}
        _require(arrays['states'].shape == (N, REPEATS, H+1, 10) and arrays['states'].dtype == np.float64
                 and arrays['applied_actions'].shape == (N, REPEATS, H, 2) and arrays['applied_actions'].dtype == np.float64
                 and arrays['total_cost'].shape == (N, REPEATS) and arrays['total_cost'].dtype == np.float64
                 and arrays['valid'].shape == (N, REPEATS) and arrays['valid'].dtype == bool, 'Actual future tensor population malformed')
        for candidate in range(N):
            for repeat in range(REPEATS):
                repeat_id = spec['snapshot_id']+f'/candidate-{candidate:02d}/repeat-{repeat}'
                receipt = verified_repeat_receipt(run, repeat_id, deep=True)
                row, regenerated = receipt['summary'], receipt['raw_regeneration']
                _require(typed_digest(arrays['states'][candidate, repeat]) == regenerated['state_hash']
                    and typed_digest(arrays['applied_actions'][candidate, repeat]) == regenerated['applied_action_hash']
                    and bool(arrays['valid'][candidate, repeat]) == regenerated['valid']
                    and arrays['total_cost'][candidate, repeat] == regenerated['cost'], 'Actual future tensor differs from its raw prefix')
                for key in COMPONENT_NAMES:
                    _require(struct.pack('=d', arrays['cost_'+key][candidate, repeat]) == struct.pack('=d', regenerated['cost_components'][key]),
                             'Actual future component tensor differs from raw')
                files[receipt['seal_path']] = receipt['seal_sha256']
                files[receipt['summary_path']] = receipt['summary_sha256']
                files.update(row['raw_receipt']['files_sha256'])
                completed_ids.append(repeat_id)
                returned += row['actual_returned_intervals']
        for model in MODELS:
            diagnostic_path = folder/'models'/model/'diagnostics.json'
            diagnostic = _read(diagnostic_path)
            _require(diagnostic == by[(spec['snapshot_id'], model)], 'Diagnostic table differs from its row')
            with np.load(folder/'models'/model/'trace.npz', allow_pickle=False) as trace:
                predicted, valid = trace['total_cost'].copy(), trace['valid'].copy()
            criteria = manifest['profile']['counterfactual_check']
            regenerated = compare_candidates(predicted, valid & np.isfinite(predicted), arrays['total_cost'], arrays['valid'],
                atol=criteria['cost_padding_atol'], rtol=criteria['cost_padding_rtol'])
            _require(all(diagnostic.get(key) == value for key, value in regenerated.items()), 'Cost/rank/loss/invalid diagnostics do not regenerate')
            files[str(diagnostic_path)] = file_sha256(diagnostic_path)
        for name in ('actual_futures.npz', 'physical_replays.json', 'summary.json'):
            files[str(folder/name)] = file_sha256(folder/name)
    _require(returned == summary['actual_future_intervals'], 'Summary work count differs from all576 raw prefixes')
    files[str(stage/'comparison_ledger.json')] = file_sha256(stage/'comparison_ledger.json')
    files[str(stage/'summary.json')] = file_sha256(stage/'summary.json')
    return files, completed_ids


def seal_complete_e1(run, expected_summary_sha256):
    run, manifest, manifest_sha = _manifest(run)
    target = run/'E1/stage_receipt.json'
    if target.exists():
        raise FileExistsError(target)
    path = run/'E1/summary.json'
    _hash(path, expected_summary_sha256)
    _require(not (run/'E1/post_summary_failure.json').exists(), 'A post-summary failure prevents adoption')
    summary = _read(path)
    files, ids = _complete_files(run, manifest, summary)
    receipt = {'schema': 'p5-E1-stage-seal-v1', 'frozen_manifest_sha256': manifest_sha,
        'summary_sha256': expected_summary_sha256, 'completed_repeat_ids': ids,
        'files_sha256': files, 'all576_raw_prefixes_and84_base_diagnostics_regenerated': True}
    target = run/'E1/stage_receipt.json'
    _save(target, receipt)
    return {'path': str(target), 'sha256': file_sha256(target), **receipt}


def adopt_complete_e1(run, manifest):
    run, actual, manifest_sha = _manifest(run)
    _require(_same(actual, manifest), 'Adoption caller uses another frozen manifest')
    summary_path = run/'E1/summary.json'
    if not summary_path.exists():
        return None
    summary = _read(summary_path)
    if summary.get('status') != 'COMPLETE_RECORDED':
        return None  # Caller must stop on an existing partial E1 directory.
    _require(not (run/'E1/post_summary_failure.json').exists(), 'Post-summary resource/error stop remains unresolved')
    receipt_path = run/'E1/stage_receipt.json'
    receipt = _read(receipt_path)  # Missing seal is not freshly invented here.
    _require(receipt['schema'] == 'p5-E1-stage-seal-v1' and receipt['frozen_manifest_sha256'] == manifest_sha,
             'Stage seal belongs to another frozen experiment')
    _hash(summary_path, receipt['summary_sha256'])
    _validate_file_map(run, receipt['files_sha256'], deep=True)
    files, ids = _complete_files(run, manifest, summary)
    _require(files == receipt['files_sha256'] and ids == receipt['completed_repeat_ids'], 'Complete E1 seal has missing/extra/reordered evidence')
    return {'status': 'ADOPT_COMPLETE_VERIFIED_E1', 'summary': summary,
        'summary_sha256': receipt['summary_sha256'], 'stage_seal_path': str(receipt_path),
        'stage_seal_sha256': file_sha256(receipt_path), 'completed_repeat_ids': ids,
        'actual_bytes': sum(Path(path).stat().st_size for path in files),
        'verification': 'all576 external raw receipts plus pure state/action/cost and84 cost/rank/invalid diagnostics regeneration'}
