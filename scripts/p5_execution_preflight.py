"""Bounded development-only check of the final executor and replay contract.

This never creates final families or uses the final RNG stream. All real cells,
fixed-prefix reconstruction, injected faults and missing-evidence challenges
are labelled separately. Existing scientific files are only read.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
import time
import tomllib
import traceback
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from push_align.config import ROOT
from push_align.contact_geometry import check_contact_geometry
from push_align.data import digest, _encode_nonfinite
from push_align.environment import PlanarPushEnv
from push_align.final_execution import execute_episode, check_result_completeness, check_snapshot_completeness
from push_align.learning_data import load_learning_data
from push_align.planning import LearnedPredictor, PlannerContext
from push_align.planning_protocol import GeometryProposalAdapter
from push_align.closed_loop import consume_actual_step, model_planning_scope, ENGINE_GUARDS
from push_align.reproduction import build_witness, compare_witness, raw_reference
from p2_collect_pilot import save_trace
from p4_compare_candidates import load_models, prediction_trace
from p4_development_inputs import load_development_snapshot
from p4_verify_sequence_loss import snapshot_sources

DESIGN = ROOT / 'configs/p5_execution_preflight.design.v1.json'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(_encode_nonfinite(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def tree_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file())


def asdict_capture(value):
    return {key: asdict(item) if hasattr(item, '__dataclass_fields__') else item for key,item in value.items()}


def bindings_match(bindings):
    return all((ROOT / p).is_file() and digest(ROOT / p) == h for p, h in bindings.items())


def save_plan_traces(folder, execution):
    """Retain complete arrays even for unexecuted all-invalid decisions."""
    for tick, planned in execution.plan_results.items():
        target = Path(folder) / 'plans' / f'step-{tick:03d}'
        prediction_trace(target, planned.predictions, planned.costs, planned.candidates)
        save(target / 'identity.json', next(d for d in execution.decisions if d['index'] == tick))


class CountedPredictor:
    def __init__(self, predictor, fault=None):
        self.predictor, self.fault, self.calls, self.rows = predictor, fault, 0, 0

    def __call__(self, state, action):
        self.calls += 1
        self.rows += len(state)
        if self.fault == 'all_nonfinite_prediction':
            return np.full_like(state, np.nan)
        if self.fault == 'forbidden_future_engine':
            return mujoco.mj_forward(None, None)
        if self.fault == 'predictor_exception':
            raise RuntimeError('Explicit development predictor exception injection')
        result = self.predictor(state, action)
        if self.fault == 'mixed_nonfinite_prediction':
            result = result.copy()
            result[0] = np.nan
        return result


def raw_exact(left, right):
    with np.load(left / 'raw_execution.npz', allow_pickle=False) as a, np.load(right / 'raw_execution.npz', allow_pickle=False) as b:
        arrays = a.files == b.files and all(np.array_equal(a[k], b[k], equal_nan=True) for k in a.files)
    with gzip.open(left / 'raw_execution.json.gz', 'rt') as f:
        a = json.load(f)
    with gzip.open(right / 'raw_execution.json.gz', 'rt') as f:
        b = json.load(f)
    return bool(arrays and a == b)


def run(out):
    started = time.perf_counter()
    design = read(DESIGN)
    limits = design['budget']
    frozen = read(ROOT / 'configs/p5_freeze_candidate.v1.json')
    previous = read(ROOT / 'configs/p4_first_moment.v1.json')['bindings_sha256']
    assert bindings_match(previous) and bindings_match(frozen['source_sha256'])
    assert design['planning']['candidate_seed_base'] == 4711 and not design['final_test_opened']
    physical = read(ROOT / 'runs/p4-04-20261013-01/manifest.json')['physical']
    planner = read(ROOT / 'runs/p4-04-20261013-01/manifest.json')['planner']
    shared = read(ROOT / 'configs/p3_planning.v1.json')
    assert all(planner[k] == shared[k] for k in ('planner', 'cost', 'task'))
    profile = tomllib.loads((ROOT / 'configs/acceptance.v1.toml').read_text())
    geometry = read(ROOT / 'configs/p3_geometry_selected.v2.json')
    transition = read(ROOT / 'configs/p3_geometry_transition.v2.json')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    normalizer_metadata = read(ROOT / 'runs/p3-02-20261009-01/learning_data/normalizer/normalizer.json')['normalizer']
    _, normalizer = load_learning_data(ROOT / 'runs/p3-02-20261009-01/learning_data', normalizer_metadata['schema'],
                                        normalizer_metadata['dataset_identity'], normalizer_metadata['normalizer_hash'])
    models, identities = load_models(normalizer, physical, read(ROOT / 'configs/p4_development_e1.v1.json'))
    assert ['G', *models] == design['controller_ids']
    source_hashes = snapshot_sources(out)
    source_hashes[str(DESIGN.relative_to(ROOT))] = digest(DESIGN)
    setup = {'new_action_intervals': 0, 'mujoco_step_calls_including_setup': 0,
             'restored_fixed_prefix_intervals': 0, 'model_calls': 0, 'model_rows': 0}
    prior_attempt_intervals = {}
    for prior in sorted((ROOT / 'runs').glob('p5-preflight-20261015-*')):
        if prior == out:
            continue
        total = 0
        for p in prior.rglob('raw_execution.npz'):
            if 'source' not in p.relative_to(prior).parts:
                with np.load(p, allow_pickle=False) as z:
                    total += len(z['state'])
        prior_attempt_intervals[str(prior.relative_to(ROOT))] = total
    observations = []

    def guard():
        free = shutil.disk_usage(ROOT).free
        stored = tree_bytes(out)
        if free < limits['minimum_free_reserve_bytes'] + 4 * 1024**2 or stored > 64 * 1024**2:
            raise RuntimeError('Preflight storage/reserve guard reached; preserve prefix')
        if time.perf_counter() - started > limits['maximum_run_wall_time_s']:
            raise RuntimeError('Preflight time guard reached; preserve prefix')
        observations.append({'elapsed_s': time.perf_counter() - started, 'free_bytes': free, 'run_bytes': stored,
                             'action_intervals': setup['new_action_intervals']})

    original_step, engine_step = PlanarPushEnv.step, mujoco.mj_step

    def counted_step(env, action):
        guard()
        if setup['new_action_intervals'] + sum(prior_attempt_intervals.values()) >= limits['maximum_new_action_intervals']:
            raise RuntimeError('Declared100-action-interval cap reached before new step')
        setup['new_action_intervals'] += 1
        try:
            return original_step(env, action)
        finally:
            with (out / 'execution_ledger.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({**setup, 'wall_time_s': time.perf_counter()-started}, allow_nan=False)+'\n')

    def counted_engine(*args, **kwargs):
        setup['mujoco_step_calls_including_setup'] += 1
        return engine_step(*args, **kwargs)

    source_cases = design['actual_cases']
    manifest = {'schema': 'p5-execution-preflight-run-v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'design': design, 'design_sha256': digest(DESIGN), 'source_hashes': source_hashes,
                'frozen_candidate_sha256': digest(ROOT / 'configs/p5_freeze_candidate.v1.json'),
                'model_identities': identities, 'physical': physical, 'planner': planner,
                'prior_attempt_intervals_from_retained_raw': prior_attempt_intervals,
                'scope': 'Existing development inputs only; no final-test families generated or opened.',
                'final_test_used': False, 'final_family_count': 0, 'learning_updates': 0,
                'personal_understanding': 'NOT_ASSESSED'}
    save(out / 'manifest.json', manifest)
    common_bindings = {**source_hashes, 'manifest.json': digest(out / 'manifest.json')}
    pairs, result_rows, expected = [], [], []
    fault_rows, evidence_challenges = [], {}

    with patch.object(PlanarPushEnv, 'step', counted_step), patch.object(mujoco, 'mj_step', counted_engine):
        inputs = {}
        for spec in source_cases:
            guard()
            inputs[spec['id']] = load_development_snapshot(ROOT / spec['source_folder'], spec['source_tick'], physical, profile)
        env = PlanarPushEnv(physical)
        resumed = {}
        for key, data in inputs.items():
            resumed[key] = {'snapshot': data.snapshot, 'status': data.status, 'context': data.context,
                            'gctx': data.gctx, 'episode_index': data.case['episode_index'],
                            'origin': {'case': data.case, 'source': source_cases[list(inputs).index(key)],
                                       'input_proof': data.proof, 'source_files_sha256': data.source_files_sha256}}
        # Derive the known candidate11 fifth boundary from unchanged recorded commands.
        data = inputs['holding']
        spec = next(s for s in source_cases if s['id'] == 'holding')
        base = ROOT / spec['archived_fixed_prefix']
        with np.load(base.with_suffix('.npz'), allow_pickle=False) as raw:
            commands, states = raw['action_applied'][:5].copy(), raw['state'][:5].copy()
        with gzip.open(base.with_suffix('.json.gz'), 'rt') as stream:
            original_diagnostics = json.load(stream)['diagnostics'][:5]
        env.restore(data.snapshot)
        status, context, gctx = deepcopy(data.status), deepcopy(data.context), deepcopy(data.gctx)
        adapter = GeometryProposalAdapter(physical, geometry, transition)
        fixed_results, prefix_equal = [], []
        for index, action in enumerate(commands):
            observed = env.observe()
            tick = context.control_step
            guards = dict.fromkeys(ENGINE_GUARDS, 0)
            with model_planning_scope(guards):
                proposal = adapter.propose(observed.state, data.task, gctx, status.elapsed_s, control_tick=tick)
            gctx = adapter.commit(proposal, actual_control_tick=tick)
            context = PlannerContext(context.previous_applied_action.copy(), None, tick + 1)
            step = env.step(action)
            fixed_results.append(step)
            geometry_check = check_contact_geometry(step.diagnostics['samples'], profile['physics_suite']['actual_contact_penetration_max_m'])
            status, context = consume_actual_step(status, step, data.task, env.action_dt, context,
                                                   holding=False, geometry=geometry_check)
            prefix_equal.append(bool(np.array_equal(step.observation.state, states[index]) and
                                      _encode_nonfinite(step.diagnostics) == original_diagnostics[index]))
        setup['restored_fixed_prefix_intervals'] = len(fixed_results)
        dest = out / 'holding_reconstruction'
        dest.mkdir()
        save_trace(dest, 'raw_execution', data.snapshot, fixed_results)
        save(dest / 'summary.json', {'source': str(base.relative_to(ROOT)), 'prefix_steps': 5,
                                    'each_state_and_raw_diagnostic_exact': prefix_equal,
                                    'task': asdict(status), 'context': _encode_nonfinite(asdict(context)),
                                    'geometry_context': asdict(gctx),
                                    'scope': 'Known fixed-action branch reconstruction, not learned controller alignment.'})
        assert all(prefix_equal) and status.phase == 'holding'
        resumed['holding'].update(snapshot=env.snapshot(), status=status, context=context, gctx=gctx)
        resumed['holding']['origin']['fixed_prefix_reconstruction'] = 'holding_reconstruction/summary.json'

        def execute(key, controller, attempt, *, fault=None, intervals=None):
            guard()
            state = resumed[key]
            spec = next(s for s in source_cases if s['id'] == key)
            predictor = None if controller == 'G' else CountedPredictor(LearnedPredictor(models[controller].model, normalizer), fault)
            label = f'{controller}/{key}' if fault is None else f'{controller}/fault-{fault}'
            dest = out / ('actual' if fault is None else 'faults') / key / controller / attempt
            dest.mkdir(parents=True, exist_ok=False)
            execution = execute_episode(env, state['snapshot'], case_id=key if fault is None else f'fault-{fault}',
                episode_index=state['episode_index'], controller_id=controller, execution_id=f'{out.name}/{label}/{attempt}',
                predictor=predictor, physical_config=physical, planner_config=planner,
                geometry_parameters=geometry, transition_parameters=transition,
                candidate_seed_base=4711, candidate_stream_id=0,
                max_intervals=intervals or spec['new_intervals_per_attempt'], run_scope='development_preflight',
                penetration_budget_m=profile['physics_suite']['actual_contact_penetration_max_m'],
                initial_status=state['status'], initial_context=state['context'], initial_gctx=state['gctx'],
                origin={**state['origin'], 'fault_injection': fault},
                capture_ticks=(0, 29) if key == 'start' and controller == 'G' else ())
            save_trace(dest, 'raw_execution', state['snapshot'], execution.results)
            raw = raw_reference([dest / 'raw_execution.npz', dest / 'raw_execution.json.gz'], len(execution.results))
            save(dest / 'summary.json', execution.summary())
            save(dest / 'decisions.json', execution.decisions)
            save(dest / 'unexecuted_plan.json', execution.unexecuted_plan)
            save(dest / 'captured_inputs.json', {str(k): asdict_capture(v) for k,v in execution.captured_inputs.items()})
            save_plan_traces(dest, execution)
            if predictor is not None:
                setup['model_calls'] += predictor.calls
                setup['model_rows'] += predictor.rows
            return execution, dest, raw

        for spec in source_cases:
            key = spec['id']
            controllers = design['controller_ids'] if spec['controllers'] == 'all8' else spec['controllers']
            for controller in controllers:
                reference, ref_dir, raw = execute(key, controller, 'reference')
                identity = {'case_id': key, 'controller_id': controller,
                            'model': identities.get(controller, {'kind': 'geometry_G4'}),
                            'snapshot_source': resumed[key]['origin'], 'candidate_seed_base': 4711}
                # Absolute manifest key avoids resolving a source relative to the wrong directory.
                bindings = {str(out / 'source' / p) if p != 'manifest.json' else str(out / p): h for p, h in common_bindings.items()}
                witness = build_witness(reference, identity=identity, source_bindings=bindings,
                                        raw_reference=raw, attempt_id='reference')
                save(ref_dir / 'witness.json', witness)
                fresh, fresh_dir, fresh_raw = execute(key, controller, 'fresh')
                compact = build_witness(fresh, identity=identity, source_bindings=bindings, raw_reference=None,
                                        attempt_id='fresh', original_raw_reference=raw)
                save(fresh_dir / 'witness_compact.json', compact)
                comparison = compare_witness(witness, compact, reference.task_spec, profile)
                full = build_witness(fresh, identity=identity, source_bindings=bindings, raw_reference=fresh_raw,
                                     attempt_id='fresh', original_raw_reference=raw)
                save(fresh_dir / 'witness_with_raw.json', full)
                comparison_full = compare_witness(witness, full, reference.task_spec, profile)
                save(fresh_dir / 'comparison.json', comparison)
                save(fresh_dir / 'comparison_with_raw.json', comparison_full)
                raw_equal = raw_exact(ref_dir, fresh_dir)
                row = {'case_id': key, 'controller_id': controller, 'reference': reference.summary(),
                       'fresh': fresh.summary(), 'comparison': comparison, 'comparison_with_raw': comparison_full,
                       'raw_reexecution_exact': raw_equal,
                       'compact_witness_bytes': (fresh_dir / 'witness_compact.json').stat().st_size,
                       'reference_raw_bytes': sum(p.stat().st_size for p in ref_dir.glob('raw_execution.*')),
                       'reference_path': str(ref_dir.relative_to(ROOT)), 'fresh_path': str(fresh_dir.relative_to(ROOT))}
                pairs.append(row)
                expected.append(reference.cell_id)
                result_rows.append(reference.summary())
                save(out / f'pair-{key}-{controller}.json', row)
                if key == 'start' and controller == 'R':
                    exemplar = (reference, witness, fresh, compact, full, raw, fresh_raw, identity, bindings)
                if key == 'start' and controller == 'G':
                    g_snapshot_slots = check_snapshot_completeness([0, 29], reference.captured_inputs)
                    assert check_snapshot_completeness([0], reference.captured_inputs)['status'] == 'PASS_COMPLETE'

        for fault in ('mixed_nonfinite_prediction', 'all_nonfinite_prediction', 'forbidden_future_engine', 'predictor_exception'):
            execution, folder, raw = execute('start', 'R', fault, fault=fault, intervals=1)
            plans = list(execution.plan_results.values())
            summary = execution.summary()
            if fault == 'mixed_nonfinite_prediction':
                plan = plans[0]
                okay = bool(len(plan.predictions.valid) == 128 and not plan.predictions.valid[0]
                             and np.isposinf(plan.costs.total[0]) and plan.selected_index != 0
                             and plan.predictions.valid[plan.selected_index])
            elif fault == 'all_nonfinite_prediction':
                okay = summary['termination'] == 'planner_invalid' and len(execution.results) == 0 \
                       and execution.unexecuted_plan is not None and np.isposinf(plans[0].costs.total).all()
            else:
                okay = summary['execution_status'] == 'infrastructure_error' and not execution.results
                if fault == 'forbidden_future_engine':
                    okay &= execution.future_engine_calls['mj_forward'] == 1
            fault_rows.append({'fault': fault, 'expected_branch_observed': bool(okay), 'summary': summary,
                               'scope': 'Explicit fault injection; not a naturally observed failure.'})

        reference, ref_w, fresh, compact, full, raw, fresh_raw, identity, bindings = exemplar
        altered = deepcopy(full)
        altered['initial']['state'][0] += .01
        for interval in altered['intervals']:
            interval['state_before'][0] += .01
            interval['state_after'][0] += .01
        for decision in altered['decisions']:
            decision['state'][0] += .01
        mismatch = compare_witness(ref_w, altered, reference.task_spec, profile)
        evidence_challenges['boundary_mismatch_with_raw'] = mismatch
        altered = deepcopy(compact)
        altered['intervals'].pop()
        evidence_challenges['missing_boundary'] = compare_witness(ref_w, altered, reference.task_spec, profile)
        altered = deepcopy(full)
        first_file = next(iter(altered['raw_reference']['files']))
        altered['raw_reference']['files'][first_file] = '0' * 64
        evidence_challenges['wrong_raw_hash'] = compare_witness(ref_w, altered, reference.task_spec, profile)
        evidence_challenges['missing_result'] = check_result_completeness(expected, result_rows[:-1])
        evidence_challenges['duplicate_result'] = check_result_completeness(expected, result_rows + result_rows[:1])
        evidence_challenges['unexpected_result'] = check_result_completeness(expected, result_rows + [
            {**result_rows[0], 'cell_id': 'unknown/case'}])
        evidence_challenges['required_tick29_missing'] = g_snapshot_slots
        save(out / 'evidence_challenges.json', evidence_challenges)

    complete = check_result_completeness(expected, result_rows, expected_run_scope='development_preflight')
    holding = next(p for p in pairs if p['case_id'] == 'holding')
    holding_dir = ROOT / holding['reference_path']
    with gzip.open(holding_dir / 'raw_execution.json.gz', 'rt') as stream:
        holding_raw = json.load(stream)
    hold_states = []
    # Use saved boundary witness for reported qualification; raw is separately retained.
    hold_witness = read(holding_dir / 'witness.json')
    for row in hold_witness['intervals']:
        hold_states.append({'held_time_s': row['task_after']['held_time_s'],
                            'phase': row['task_after']['phase'], 'applied': row['applied_action']})
    checks = {
        'all13_reference_cells_present_in_order': len(pairs) == 13 and complete['status'] == 'PASS_COMPLETE',
        'all_actual_boundary_reproductions_pass': all(p['comparison']['status'] == 'PASS_REPRODUCTION' for p in pairs),
        'all_actual_fullraw_reexecutions_exact': all(p['raw_reexecution_exact'] for p in pairs),
        'all_actual_future_engine_calls_zero': all(not any(p[a]['future_engine_calls'].values()) for p in pairs for a in ('reference', 'fresh')),
        'holding_below03_not_ready_then_ready': any(r['held_time_s'] < .3 and r['phase'] != 'ready' for r in hold_states)
            and holding['reference']['termination'] == 'ready' and all(r['applied'] == [0., 0.] for r in hold_states),
        'holding_no_post_terminal_interval': holding['reference']['counts']['physics_intervals_returned'] == len(hold_states)
            and len(hold_states) < 6 and hold_states[-1]['phase'] == 'ready' and all(r['phase'] != 'ready' for r in hold_states[:-1]),
        'actual_original_clock_timeout': next(p for p in pairs if p['case_id'] == 'timeout')['reference']['termination'] == 'timeout',
        'all_four_injected_fault_branches_detected': len(fault_rows) == 4 and all(r['expected_branch_observed'] for r in fault_rows),
        'all_missing_or_corrupt_evidence_rejected': all(r['status'] in {'FAIL', 'INCOMPLETE'} for r in evidence_challenges.values()),
        'original_scientific748_and_candidate97_unchanged': bindings_match(previous) and bindings_match(frozen['source_sha256']),
        'all_run_sources_unchanged': bindings_match(source_hashes),
        'bounded_physical_execution': setup['new_action_intervals'] + sum(prior_attempt_intervals.values()) <= 100,
        'final_test_unused': True,
    }
    guard()
    summary = {'schema': 'p5-execution-preflight-summary-v1', 'status': 'PASS_PREFLIGHT' if all(checks.values()) else 'HOLD_PREFLIGHT',
               'checks': checks, 'actual_controller_pairs': len(pairs), 'actual_attempts': len(pairs) * 2,
               'actual_cases': pairs, 'fault_injection_cases': fault_rows, 'evidence_challenges': evidence_challenges,
               'result_completeness': complete, 'holding_timeline': hold_states,
               'resources': {**setup, 'prior_attempt_intervals_from_retained_raw': prior_attempt_intervals,
                             'total_action_intervals_including_failed_attempts': setup['new_action_intervals'] + sum(prior_attempt_intervals.values()),
                             'stored_bytes_before_summary': tree_bytes(out), 'wall_time_s': time.perf_counter()-started,
                             'free_bytes_at_closing': shutil.disk_usage(ROOT).free,
                             'compact_witness_bytes_total': sum(p['compact_witness_bytes'] for p in pairs),
                             'reference_raw_bytes_total': sum(p['reference_raw_bytes'] for p in pairs),
                             'measurements': observations},
               'P5_01': 'HOLD', 'final_test_used': False, 'new_final_families': 0, 'learning_updates': 0,
               'alignment_improvement': 'NOT_EVALUATED', 'personal_understanding': 'NOT_ASSESSED',
               'scope': 'Development G1 preflight only; no final-batch resource signoff or frozen final-family manifest.'}
    save(out / 'summary.json', summary)
    if not all(checks.values()):
        raise RuntimeError('Preflight gates incomplete; preserve outputs and investigate')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if not args.run_id.startswith('p5-preflight-') or Path(args.run_id).name != args.run_id:
        raise ValueError('Use a dedicated p5-preflight run identity')
    out = ROOT / 'runs' / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    try:
        result = run(out)
        print(json.dumps({'status': result['status'], 'checks': result['checks'],
                          'resources': {k:v for k,v in result['resources'].items() if k != 'measurements'}}, ensure_ascii=False))
    except Exception as error:
        save(out / 'failure.json', {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
        raise
    finally:
        save(out / 'artifact_hashes.json', {'sha256': {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*'))
                                                     if p.is_file() and p.name != 'artifact_hashes.json'}})


if __name__ == '__main__':
    main()
