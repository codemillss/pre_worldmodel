"""Independent recorded-prefix review; never runs a model or physical engine.

A STOP_REVIEW label is never changed. A separate resolution accepts a completed
known task failure only after files, witnesses, raw samples, geometry and task
termination agree. Compact fresh records retain an explicit scope limitation:
their new substeps can only be checked directly while the caller still supplies
the actual EpisodeExecution; otherwise the frozen driver's exact-raw receipt is
verified against the retained original, alongside fresh boundary reproduction.
"""
from __future__ import annotations
from dataclasses import asdict, replace
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import resource
import sys
from types import SimpleNamespace

import numpy as np

from .contact_geometry import check_contact_geometry
from .coverage import _contact_events
from .environment import Observation, PlanarPushEnv
from .final_driver import read_full_trace, typed_digest
from .final_execution import EpisodeExecution, array_hash
from .final_recording import file_sha256, read_json_gzip, read_plan_index
from .planning import TaskSpec
from .planning_protocol import GeometryProposalAdapter
from .reproduction import _validate, compare_witness, json_safe, value_hash
from .task import start_task, consume_step, fail_task, handoff_metrics, pose_in_goal

KNOWN_TERMINALS = {'ready', 'timeout', 'planner_invalid', 'controller_infeasible',
                   'physics_error', 'geometry_invalid', 'workspace_exit', 'planarity_violation'}
RAW_MODULES = {'push_align.final_recording', 'push_align.final_recording_columns'}
MAX_RSS_BYTES = 6 * 1024**3
MAX_RAW_PAIR_BYTES = 64 * 1024**2


class ReviewError(ValueError):
    """Evidence is incomplete, inconsistent, changed or outside review bounds."""


def _need(condition, message):
    if not condition:
        raise ReviewError(message)


def _rss():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) * (1 if sys.platform == 'darwin' else 1024)


def _memory():
    _need(_rss() <= MAX_RSS_BYTES, 'review_RSS_budget_exceeded_preserve_prefix')


def _same(left, right):
    return value_hash(left) == value_hash(right)


def _files(bindings, label):
    _need(type(bindings) is dict and bool(bindings), label + ': missing file bindings')
    for path, digest in bindings.items():
        _need(type(path) is str and type(digest) is str and len(digest) == 64,
              label + ': malformed file binding')
        _need(Path(path).is_file() and file_sha256(path) == digest,
              label + ': missing or changed file:' + str(path))


def _manifest(path, expected=None):
    path = Path(path).resolve()
    _need(path.is_file(), 'record_manifest_missing')
    if expected is not None:
        _need(file_sha256(path) == expected, 'record_manifest_sha_mismatch')
    value = json.loads(path.read_text())
    _need(value.get('schema') == 'p5-recorded-episode-v1', 'unsupported_record_manifest')
    _need(type(value.get('files_sha256')) is dict and sum(Path(p).stat().st_size
        for p in value['files_sha256'] if Path(p).is_file()) <= MAX_RAW_PAIR_BYTES,
        'recorded_cell_exceeds_declared64MiB_review_budget_reestimate')
    _files(value.get('files_sha256'), 'recorded_output')
    _files(value.get('source_bindings'), 'recorded_source')
    _need(value.get('source_binding_check_after_execution') is True, 'source_change_during_execution')
    _need(value.get('recording_status') in {'COMPLETE_RECORDED', 'STOP_REVIEW'}, 'incomplete_recording_status')
    _need(value.get('witness_validation_issues') == [], 'saved_witness_validation_incomplete')
    _need(value.get('missing_required_input_ticks') == [], 'missing_required_full_input')
    witness_path = value['witness_path']
    _need(witness_path in value['files_sha256'], 'witness_not_in_recorded_file_bindings')
    witness = read_json_gzip(witness_path, expected_sha256=value['files_sha256'][witness_path])
    _need(witness.get('record_kind') == value.get('record_kind'), 'manifest_witness_record_kind_differs')
    _need(_validate(witness, 'independent_review', 1e-9) == [], 'witness_structure_counts_or_task_invalid')
    _need(_same(witness['identity'], value['identity']) and _same(witness['source_bindings'], value['source_bindings']),
          'manifest_witness_identity_or_sources_differ')
    if witness['controller_id'] == 'G':
        _need(value['identity'].get('model') is None and value['identity'].get('normalizer') is None,
              'G_must_not_claim_learned_model_or_normalizer')
    else:
        _need(value['identity'].get('model') is not None and value['identity'].get('normalizer') is not None,
              'learned_controller_missing_model_or_normalizer_identity')
    summary = value['execution']
    for field in ('execution_id', 'case_id', 'controller_id', 'run_scope', 'execution_status', 'termination',
                  'counts', 'future_engine_calls', 'initial_full_snapshot_equal', 'error'):
        _need(_same(summary[field], witness[field]), 'manifest_witness_summary_differs:' + field)
    _need(_same(summary['task'], witness['final_task']), 'manifest_final_task_differs')
    _need(witness['execution_status'] == 'completed' and witness['termination'] in KNOWN_TERMINALS,
          'unresolved_infrastructure_or_unknown_terminal')
    _need(witness['error'] is None and not any(witness['future_engine_calls'].values()), 'error_or_oracle_query')
    return path, value, witness


def _receipt_files(manifest, receipt, label):
    _need(type(receipt) is dict, label + ': missing receipt')
    _files(receipt.get('files_sha256'), label)
    _need(all(manifest['files_sha256'].get(p) == h for p, h in receipt['files_sha256'].items()),
          label + ': receipt not covered by output bindings')


def _raw(manifest):
    receipt = manifest['raw']
    _receipt_files(manifest, receipt, 'raw')
    module = manifest.get('raw_codec_module')
    _need(module in RAW_MODULES, 'unsupported_raw_codec')
    codec = importlib.import_module(module)
    _need(set(receipt['files_sha256']) == {str(path) for path in codec.raw_paths(receipt['prefix'])},
          'raw_pair_paths_differ')
    _need(sum(Path(p).stat().st_size for p in receipt['files_sha256']) <= MAX_RAW_PAIR_BYTES,
          'raw_pair_exceeds_declared64MiB_review_budget_reestimate')
    _memory()
    snapshot, results = codec.read_raw(receipt['prefix'], expected_sha256=receipt['files_sha256'])
    _memory()
    _need(receipt['interval_count'] == len(results), 'raw_receipt_interval_count_differs')
    return snapshot, results


def _inputs(manifest, witness, physical_config, profile):
    values = {}
    for receipt in manifest['full_traces']:
        _receipt_files(manifest, receipt, 'full_trace')
        value, arrays, metadata = read_full_trace(receipt)
        tick = receipt['tick']
        _need(type(tick) is int and tick not in values and value['tick'] == tick, 'duplicate_or_inconsistent_full_input_tick')
        _need(_same(value['physical_config'], physical_config), 'full_input_physical_config_differs')
        _need(value['penetration_budget_m'] == profile['physics_suite']['actual_contact_penetration_max_m'],
              'full_input_penetration_budget_differs')
        _need(_same(value['identity'], manifest['identity']) and _same(value['source_bindings'], manifest['source_bindings']),
              'full_input_identity_or_sources_differ')
        values[tick] = value
        decision = next((row for row in witness['decisions'] if row['index'] == tick),None)
        _need(decision is not None, 'full_input_has_no_controller_decision')
        info = decision['plan_identity']
        _need(receipt['full_model_arrays'] is (info is not None), 'full_trace_model_presence_differs')
        if info is not None:
            for field, key in (('candidate_hash','candidates'),('prediction_hash','predicted_states'),
                               ('delta_hash','raw_deltas'),('total_cost_hash','total_cost')):
                _need(array_hash(arrays[key]) == info[field], 'full_plan_array_hash_differs:' + field)
            _need(metadata['plan']['selected_index'] == info['selected_index']
                  and metadata['plan']['reason'] == info['reason']
                  and _same(metadata['plan']['action'],info['action']), 'full_plan_choice_differs')
            # Only small per-index vectors need survive this helper; full future
            # arrays have been checked and need not increase review lifetime RAM.
            value['_review_index_arrays'] = {k:v for k,v in arrays.items()
                if k in {'valid','reasons','failure_steps','total_cost'} or k.startswith('cost_')}
    ticks = {d['index'] for d in witness['decisions']}
    required = {0, 29, 299} & ticks
    if witness['unexecuted_plan'] is not None:
        required.add(witness['unexecuted_plan']['index'])
    _need(required <= set(values), 'required_full_input_ticks_missing')
    _need(0 in values, 'initial_full_input_missing')
    return values


def _index(manifest, witness):
    receipt = manifest['plan_index']
    _receipt_files(manifest, receipt, 'plan_index')
    rows = read_plan_index(receipt['path'], expected_sha256=receipt['files_sha256'][receipt['path']])
    expected = [d['index'] for d in witness['decisions'] if d['plan_identity'] is not None]
    _need([r['tick'] for r in rows] == expected, 'plan_index_decision_ticks_differ')
    return {r['tick']: r['arrays'] for r in rows}


def _check_index(index, witness, inputs):
    for decision in witness['decisions']:
        info = decision['plan_identity']
        if info is None:
            continue
        tick = decision['index']; arrays = index[tick]
        valid, total = arrays['valid'], arrays['total_cost']
        eligible = valid & np.isfinite(total)
        _need(info['valid_predictions'] == int(valid.sum()) and info['eligible_costs'] == int(eligible.sum())
              and info['invalid_reasons'] == dict(Counter(str(v) for v in arrays['reasons'] if str(v))),
              'plan_index_counts_or_reasons_differ')
        _need(info['total_cost_hash'] == array_hash(total), 'plan_index_total_cost_hash_differs')
        selected = info['selected_index']
        if selected is not None:
            _need(type(selected) is int and 0 <= selected < len(valid) and eligible[selected],
                  'selected_plan_index_is_not_eligible')
            _need(info['selected_cost'] == float(total[selected]), 'selected_cost_differs')
        if tick in inputs:
            saved = inputs[tick]['_review_index_arrays']
            _need(typed_digest(saved) == typed_digest(arrays), 'full_trace_and_per_index_values_differ')


def _check_physics(step, snapshot, physical):
    """Reapply stored-sample validity rules without constructing an environment."""
    samples = step.diagnostics['samples']
    proxy = SimpleNamespace(config=physical, _reference_height=snapshot.reference_height)
    violations = []
    last_validity = None
    tolerance = physical['validity']['dt_match_tolerance_s']
    timestep = physical['physics']['timestep_s']
    for i, sample in enumerate(samples):
        current = PlanarPushEnv._validity(proxy, sample)
        violations.extend(current); last_validity = current
        if i:
            force = np.asarray(sample['integration_actuator_force_N'], np.float64)
            _need(force.shape == (2,), 'integration_force_shape_missing')
            if np.max(np.abs(force)) > physical['pusher']['actuator_force_limit_abs_N'] + 1e-12:
                violations.append('integration_actuator_force_exceeded')
            if abs(sample['sim_time'] - samples[i-1]['sim_time'] - timestep) > tolerance:
                violations.append('physics_time_discontinuity')
        _need(not violations or i == len(samples)-1, 'raw_continues_after_physical_failure')
    if abs(step.actual_dt - physical['physics']['action_dt_s']) > tolerance:
        violations.append('partial_transition')
    _need(sorted(set(violations)) == sorted(step.diagnostics['validity_violations']), 'independent_physical_violations_differ')
    _need(step.transition_valid is (not bool(violations)), 'independent_transition_validity_differs')
    _need(step.observation.observation_valid is (not bool(last_validity)), 'independent_observation_validity_differs')


def recompute_raw_metrics(snapshot, results, witness, *, physical_config, profile,
                          full_inputs, plan_index):
    """Re-evaluate raw task/geometry and return independently measured metrics.

    This reuses frozen pure criteria, not saved task labels, and never steps
    physics. It verifies each saved before/after task status against recomputed
    status. Planner-invalid additionally needs a complete no-eligible index.
    """
    _memory()
    task = TaskSpec.from_config(physical_config);dt = float(physical_config['physics']['action_dt_s'])
    _need(_same(asdict(task), witness['task_spec']), 'witness_task_spec_differs_from_frozen_task')
    initial = full_inputs[0]['initial_observation']
    _need(_same(initial.state.tolist(), witness['initial']['state']) and initial.sim_time == witness['initial']['sim_time']
          and initial.observation_valid is witness['initial']['observation_valid']
          and initial.schema_version == witness['initial']['schema_version'], 'initial_observation_differs')
    _need(typed_digest(full_inputs[0]['snapshot']) == typed_digest(snapshot)
          and value_hash(snapshot) == witness['initial']['snapshot_hash'], 'initial_snapshot_differs')
    status = start_task(initial, task, dt)
    _need(_same(asdict(status), witness['initial']['task']), 'initial_task_not_independently_reproduced')
    _need(len(results) == len(witness['intervals']) == witness['returned_results'], 'raw_boundary_count_differs')
    last = initial if initial.observation_valid and np.isfinite(initial.state).all() else None
    previous = initial
    for step, boundary in zip(results, witness['intervals']):
        _need(status.phase in {'running', 'holding'}, 'raw_interval_after_recomputed_terminal')
        _need(_same(asdict(status), boundary['task_before']), 'raw_task_before_differs')
        _need(np.array_equal(np.asarray(step.diagnostics['samples'][0]['state']), previous.state, equal_nan=True),
              'raw_initial_sample_not_previous_observation')
        _check_physics(step, snapshot, physical_config)
        geometry = check_contact_geometry(step.diagnostics['samples'], profile['physics_suite']['actual_contact_penetration_max_m'])
        _need(_same(geometry, boundary['geometry']), 'independent_contact_geometry_differs')
        holding = pose_in_goal(previous.state, task)
        _need(holding is boundary['holding'], 'independent_holding_state_differs')
        if holding:
            _need(np.array_equal(step.requested_action, np.zeros(2)) and np.array_equal(step.applied_action,np.zeros(2)),
                  'holding_command_not_zero')
        status = consume_step(status, step, task, dt)
        if not geometry['pass'] and status.terminal_reason != 'physics_error':
            status = replace(fail_task(status, 'geometry_invalid'),
                violations=tuple(dict.fromkeys((*status.violations,*geometry['reasons']))))
        _need(_same(asdict(status), boundary['task_after']), 'raw_task_after_differs')
        for key, actual in (('state_after', step.observation.state), ('requested_action', step.requested_action),
                            ('applied_action', step.applied_action)):
            _need(_same(boundary[key], actual.tolist()), 'raw_boundary_value_differs:' + key)
        _need(boundary['sim_time_after'] == step.observation.sim_time and boundary['actual_dt_s'] == step.actual_dt
              and boundary['transition_valid'] is step.transition_valid
              and boundary['observation_valid'] is step.observation.observation_valid, 'raw_boundary_time_or_validity_differs')
        if step.transition_valid and step.observation.observation_valid and geometry['pass'] and np.isfinite(step.observation.state).all():
            last = step.observation
        previous = step.observation
    terminal = witness['unexecuted_plan']
    if terminal is not None:
        _need(status.phase in {'running','holding'} and terminal['executed'] is False, 'invalid_unexecuted_terminal_position')
        _need(_same(asdict(status), terminal['task_status']), 'unexecuted_terminal_task_context_differs')
        reason = witness['termination']
        if reason == 'planner_invalid':
            tick = terminal['index']; index = plan_index[tick]
            valid, total = index['valid'], index['total_cost']
            _need(not (valid & np.isfinite(total)).any() and not (valid & np.isneginf(total)).any(),
                  'planner_invalid_has_eligible_candidate')
            info = terminal['plan_identity']
            _need(info['reason'] == 'planner_invalid' and info['selected_index'] is None
                  and info['eligible_costs'] == 0, 'planner_invalid_choice_not_supported')
        elif reason == 'controller_infeasible':
            _need(terminal['plan_identity'] is None and witness['controller_id'] == 'G', 'controller_infeasible_not_G')
            value = full_inputs[terminal['index']]
            adapter = GeometryProposalAdapter(physical_config,value['geometry_parameters'],value['transition_parameters'])
            proposal = adapter.propose(previous.state.copy(),task,value['geometry_context'],status.elapsed_s,
                control_tick=terminal['index'])
            _need(proposal.geometry_action is None, 'controller_infeasible_not_reproduced')
        else:
            raise ReviewError('unknown_unexecuted_failure_reason')
        status = fail_task(status, reason)
    _need(status.phase in {'ready','failed'} and status.terminal_reason in KNOWN_TERMINALS,
          'raw_prefix_has_no_known_terminal')
    _need(status.terminal_reason == witness['termination'] and _same(asdict(status), witness['final_task']),
          'independent_final_task_differs')
    endpoint = results[-1].observation if results else initial
    available = last is endpoint and last is not None and last.observation_valid
    metrics = None if last is None else handoff_metrics(last.state, task)
    contacts = _contact_events([step.diagnostics for step in results], physical_config['evaluation']['contact_active_normal_force_min_N'],1e-9)
    _memory()
    return {'success':status.terminal_reason == 'ready','elapsed_s':float(status.elapsed_s),
        'final_position_error_m':None if metrics is None else metrics['e_pos_m'],
        'final_yaw_error_rad':None if metrics is None else metrics['e_yaw_rad'],
        'final_linear_speed_m_s':None if metrics is None else metrics['linear_speed_m_s'],
        'final_angular_speed_rad_s':None if metrics is None else metrics['angular_speed_rad_s'],
        'final_observation_available':bool(available),'metric_state_origin':'terminal_observation' if available else ('last_valid_observation' if last is not None else 'unavailable'),
        'metric_state_time_s':None if last is None else float(last.sim_time-status.initial_sim_time),
        'unavailable_reason':None if last is not None else 'no independently valid observation retained',
        'recontact_count':len(contacts['qualified_recontacts']),'contact_events':contacts,
        'counts':witness['counts'],'future_engine_calls':witness['future_engine_calls']}


def review_recording(manifest_path, *, physical_config, profile, expected_manifest_sha256=None,
                     expected_identity=None, expected_source_bindings=None,
                     expected_run_scope='final_evaluation', raw_execution=None,
                     raw_execution_origin='actual_controller_return'):
    """Return an explicit read-only resolution; failures retain their old label.

    Callers persist this returned receipt and bind its SHA to the worker result.
    A rejection is not a task failure and must never release work from a ledger.
    """
    receipt = {'schema':'p5-independent-recording-review-v1','status':'REJECT_INCOMPLETE_EVIDENCE',
        'accepted':False,'issues':[],'metrics':None,'manifest_sha256':None,
        'raw_provenance':None,'fresh_substeps_independently_rechecked':False,
        'provided_raw_execution_origin':None,
        'limits':['No model or physical engine re-execution; pure frozen sample/task/geometry rules are reapplied.',
            'Recorded engine_finite/warning/contact metadata are checked under frozen sample rules; unrecorded engine arrays such as qacc are not reconstructed.',
            'Compact fresh raw reuse alone verifies original substeps and fresh boundaries plus the source-bound driver exact-raw attestation; new substeps need raw_execution for direct reassessment.']}
    try:
        _memory()
        _need(raw_execution_origin in {'actual_controller_return','reconstructed_saved_raw'},
              'unknown_provided_raw_execution_origin')
        path, manifest, witness = _manifest(manifest_path,expected_manifest_sha256)
        receipt['manifest_sha256'] = file_sha256(path)
        if expected_identity is not None:
            _need(_same(manifest['identity'],expected_identity),'requested_identity_differs')
        if expected_source_bindings is not None:
            _need(_same(manifest['source_bindings'],expected_source_bindings),'requested_source_bindings_differ')
        _need(witness['run_scope'] == expected_run_scope,'run_scope_differs')
        original, original_witness = manifest,witness
        fresh = manifest['record_kind'] == 'fresh_controller_replay'
        _need(fresh or manifest['record_kind'] == 'reference_controller_execution','unknown_record_kind')
        if fresh:
            _, original, original_witness = _manifest(manifest['reference_manifest'],manifest['reference_manifest_sha256'])
            _need(original['record_kind'] == 'reference_controller_execution'
                  and _same(original['identity'],manifest['identity'])
                  and _same(original['source_bindings'],manifest['source_bindings']),'fresh_original_identity_or_sources_differ')
            comparison = compare_witness(original_witness,witness,TaskSpec.from_config(physical_config),profile)
            _need(comparison['status'] == 'PASS_REPRODUCTION','fresh_boundary_reproduction_not_verified')
            saved = manifest['comparison']
            _need(saved['raw_exact']['status'] == 'PASS_EXACT' and saved['plan_index_exact']['status'] == 'PASS_EXACT',
                  'fresh_exact_driver_attestation_not_pass')
            receipt['fresh_boundary_reproduction'] = comparison
        source = manifest if manifest['raw'] is not None else original
        receipt['fresh_substeps_independently_rechecked'] = fresh and manifest['raw'] is not None
        snapshot, results = _raw(source)
        inputs = _inputs(source, witness if source is manifest else original_witness,physical_config,profile)
        index = _index(source,witness if source is manifest else original_witness)
        _check_index(index,witness,inputs)
        raw_reference = witness['raw_reference'] if manifest['raw'] is not None else original_witness['raw_reference']
        _need(raw_reference['files'] == source['raw']['files_sha256']
              and raw_reference['executed_intervals'] == len(results)
              and raw_reference['scope'] == 'complete_raw_execution_prefix','witness_raw_reference_differs')
        _need(value_hash(snapshot) == witness['initial']['snapshot_hash'],'fresh_raw_initial_snapshot_differs')
        raw_digest = typed_digest((snapshot,results))
        if fresh:
            saved = manifest['comparison']['raw_exact']
            _need(saved['left_sha256'] == saved['right_sha256'] == raw_digest,'fresh_exact_raw_digest_differs')
            index_digest = typed_digest([{'tick':tick,'arrays':arrays} for tick,arrays in index.items()])
            saved_index = manifest['comparison']['plan_index_exact']
            _need(saved_index['left_sha256'] == saved_index['right_sha256'] == index_digest,
                  'fresh_exact_plan_index_digest_differs')
            _need(_same(witness['original_raw_reference'],original_witness['raw_reference']),
                  'fresh_original_raw_reference_differs')
            if source is manifest:
                original_snapshot, original_results = _raw(original)
                _need(typed_digest((original_snapshot,original_results)) == raw_digest,
                      'fresh_saved_raw_differs_from_actual_original_raw')
                del original_snapshot,original_results
                original_index = _index(original,original_witness)
                _need(typed_digest([{'tick':tick,'arrays':arrays} for tick,arrays in original_index.items()]) == index_digest,
                      'fresh_saved_plan_index_differs_from_actual_original_index')
                del original_index
            if manifest['raw'] is None:
                reuse=manifest['raw_reference_reuse'];reuse_index=manifest['plan_index_reference_reuse']
                _need(reuse['files_sha256'] == original['raw']['files_sha256']
                      and reuse['prefix'] == original['raw']['prefix']
                      and reuse_index['files_sha256'] == original['plan_index']['files_sha256']
                      and reuse_index['path'] == original['plan_index']['path'],'fresh_reuse_binding_differs')
        if raw_execution is not None:
            if raw_execution_origin == 'actual_controller_return':
                _need(isinstance(raw_execution,EpisodeExecution)
                      and raw_execution.execution_id == witness['execution_id']
                      and raw_execution.case_id == witness['case_id']
                      and raw_execution.controller_id == witness['controller_id']
                      and raw_execution.run_scope == witness['run_scope']
                      and raw_execution.execution_status == witness['execution_status'],
                      'provided_raw_execution_identity_or_scope_differs')
            _need(typed_digest((raw_execution.initial_snapshot,raw_execution.results)) == raw_digest,
                  'provided_actual_raw_execution_differs')
            snapshot,results=raw_execution.initial_snapshot,raw_execution.results
            receipt['provided_raw_execution_origin'] = raw_execution_origin
            receipt['fresh_substeps_independently_rechecked'] = (
                receipt['fresh_substeps_independently_rechecked']
                or fresh and raw_execution_origin == 'actual_controller_return')
        receipt['raw_provenance'] = 'fresh_saved_full_prefix' if fresh and manifest['raw'] is not None else ('original_referenced_prefix' if fresh else 'original_saved_full_prefix')
        receipt['raw_exact_sha256']=raw_digest
        receipt['metrics']=recompute_raw_metrics(snapshot,results,witness,physical_config=physical_config,
            profile=profile,full_inputs=inputs,plan_index=index)
        physical_failure = (witness['termination'] in {'physics_error','geometry_invalid'}
            or any(not s.transition_valid or not s.observation.observation_valid for s in results))
        if manifest['recording_status'] == 'STOP_REVIEW':
            _need(witness['termination'] != 'ready' and physical_failure,'STOP_REVIEW_not_explained_by_verified_task_failure')
            _need(manifest['raw'] is not None,'failure_requires_own_full_raw_prefix')
        if fresh and physical_failure:
            _need(manifest['raw'] is not None,'fresh_physical_failure_requires_own_full_raw_prefix')
        receipt.update(accepted=True,status='ACCEPT_COMPLETED_RECORDING' if witness['termination']=='ready' else 'ACCEPT_COMPLETED_TASK_FAILURE',
            recording_status_preserved=manifest['recording_status'],termination=witness['termination'],
            independent_final_task=json_safe(witness['final_task']),
            source_bindings_sha256=value_hash(manifest['source_bindings']),
            verified_recorded_files=dict(manifest['files_sha256']))
    except (Exception,) as error:
        receipt['issues'].append(type(error).__name__ + ': ' + str(error))
        receipt['metrics']=None
    return receipt
