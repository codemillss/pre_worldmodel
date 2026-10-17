"""One predeclared real development R episode and one fresh controller rerun.

The parent is standard-library-only. Each attempt gets a new sequential child
process so original result objects cannot coexist with the fresh controller in
parent RAM. No final families, model training, parameter tuning, fallback case
selection or automatic retry is available in this entry point.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT/'configs/p5_driver_native_preflight.design.v1.json'
DEFAULT_OUT = ROOT/'runs/p5-native-driver-20261016-01'
REPORT = ROOT/'reports/p5_driver_native_preflight_review.json'
GIB = 1024**3


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
        stream.flush()
        import os
        os.fsync(stream.fileno())


def tree_bytes(folder):
    return sum(path.stat().st_size for path in Path(folder).rglob('*') if path.is_file())


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value*1024)


def verify(bindings):
    wrong = [str(path) for path, expected in bindings.items()
             if not Path(path).is_file() or sha(path) != expected]
    if wrong:
        raise ValueError(f'Source/original binding changed: {wrong[:5]}')
    return {'checked': len(bindings), 'all_unchanged': True}


def source_graph():
    """Snapshot the local static runtime import closure without importing it."""
    pending = [Path(__file__).resolve()]
    seen = set()
    while pending:
        path = pending.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if path.parent == ROOT/'src/push_align':
                        names.append('push_align.'+(node.module or ''))
                        if not node.module:
                            names.extend('push_align.'+alias.name for alias in node.names)
                elif node.module:
                    names.append(node.module)
                    names.extend(node.module+'.'+alias.name for alias in node.names)
            for name in names:
                if name.startswith('push_align.'):
                    candidate = ROOT/'src'/Path(*name.split('.')).with_suffix('.py')
                elif name.startswith('p') and '.' not in name:
                    candidate = ROOT/'scripts'/f'{name}.py'
                else:
                    continue
                if candidate.is_file():
                    pending.append(candidate)
    return sorted(seen)


def checked_design():
    design = read(DESIGN)
    g1 = read(ROOT/'configs/p5_execution_preflight.design.v1.json')
    old_start = next(row for row in g1['actual_cases'] if row['id'] == 'start')
    expected = {'source_folder': old_start['source_folder'], 'source_tick': old_start['source_tick']}
    if (design['source'] != expected or design['controller_id'] != 'R'
            or design['candidate_seed_base'] != 4711 or design['candidate_stream_id'] != 0
            or design['max_intervals_per_execution'] != 300
            or design['maximum_new_action_intervals'] != 600 or design['automatic_retries'] != 0
            or design['actual_final_test_generated'] or design['raw_codec'] != 'push_align.final_recording_columns'):
        raise ValueError('Predeclared development-only native design changed')
    return design


def run_child(out, attempt):
    # Heavy numerical imports are intentionally inside the fresh child only.
    from unittest.mock import patch
    import mujoco
    import torch
    from push_align import final_recording_columns as codec
    from push_align.config import ROOT as PROJECT_ROOT
    from push_align.environment import PlanarPushEnv
    from push_align.final_driver import (DriverStop, LiveResourceGuard, read_full_trace,
        exact_comparison, run_recorded_episode)
    from push_align.final_recording import read_plan_index, read_json_gzip, write_plan_index, write_witness_gzip
    from push_align.learning_data import load_learning_data
    from push_align.planning import LearnedPredictor
    from push_align.reproduction import json_safe
    from p4_compare_candidates import load_models
    from p4_development_inputs import load_development_snapshot

    started = time.perf_counter()
    design = checked_design()
    parent_manifest = read(out/'manifest.json')
    bindings = parent_manifest['all_bound_files_sha256']
    source_before = verify(bindings)
    if str(PROJECT_ROOT.resolve()) != str(ROOT.resolve()):
        raise ValueError('Runtime source root differs')
    child = out/attempt
    child.mkdir(exist_ok=False)
    ledger = {'action_interval_attempts': 0, 'action_intervals_returned': 0,
              'model_calls': 0, 'model_rows': 0,
              'engine_calls_including_setup': dict.fromkeys(
                  ('mj_step', 'mj_step1', 'mj_step2', 'mj_forward', 'mj_inverse'), 0)}
    import contextlib
    native_phase = ['setup']
    original_step = PlanarPushEnv.step
    originals = {name: getattr(mujoco, name) for name in ledger['engine_calls_including_setup']}
    ledger_path = child/'execution_ledger.jsonl'

    def checkpoint():
        if peak_rss() > design['memory']['pre_action_stop_RSS_bytes']:
            raise DriverStop({'status': 'STOP_PRESERVE_PREFIX', 'reasons': ['native_pre_action_RSS_limit']})
        used = tree_bytes(out)
        free = shutil.disk_usage(ROOT).free
        if used > design['maximum_new_output_bytes'] or free < design['minimum_free_reserve_bytes']:
            raise DriverStop({'status': 'STOP_PRESERVE_PREFIX', 'reasons': ['native_output_or_free_limit']})

    class NativeGuard(LiveResourceGuard):
        def before_action(self):
            checkpoint()
            return super().before_action()

    def counted_step(env, action):
        checkpoint()
        if ledger['action_interval_attempts'] >= design['max_intervals_per_execution']:
            raise DriverStop({'status': 'STOP_PRESERVE_PREFIX', 'reasons': ['native_300_interval_limit']})
        ledger['action_interval_attempts'] += 1
        try:
            result = original_step(env, action)
            ledger['action_intervals_returned'] += 1
            return result
        finally:
            with ledger_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'phase': native_phase[0], **ledger,
                    'elapsed_s': time.perf_counter()-started, 'process_peak_rss_bytes': peak_rss()},
                    allow_nan=False)+'\n')

    def counted_engine(name):
        def invoke(*args, **kwargs):
            ledger['engine_calls_including_setup'][name] += 1
            return originals[name](*args, **kwargs)
        return invoke

    guard = NativeGuard(ROOT,
        remaining_required_bytes=lambda: max(design['minimum_free_reserve_bytes'],
            design['whole_bounded_probe_required_bytes']-(tree_bytes(out)-tree_bytes(child))),
        cell_write_allowance_bytes=design['cell_write_allowance_bytes'],
        mismatch_limit_bytes=design['cell_write_allowance_bytes'],
        hard_reserve_bytes=design['minimum_free_reserve_bytes'])
    result = None
    try:
        checkpoint()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(PlanarPushEnv, 'step', counted_step))
            for name in originals:
                stack.enter_context(patch.object(mujoco, name, counted_engine(name)))
            physical = read(ROOT/'runs/p4-04-20261013-01/manifest.json')['physical']
            planner = read(ROOT/'runs/p4-04-20261013-01/manifest.json')['planner']
            common = read(ROOT/'configs/p3_planning.v1.json')
            if any(planner[key] != common[key] for key in ('planner', 'cost', 'task')):
                raise ValueError('Frozen N128/H5/cost/task changed')
            import tomllib
            profile = tomllib.loads((ROOT/'configs/acceptance.v1.toml').read_text())
            geometry = read(ROOT/'configs/p3_geometry_selected.v2.json')
            transition = read(ROOT/'configs/p3_geometry_transition.v2.json')
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            meta = read(ROOT/'runs/p3-02-20261009-01/learning_data/normalizer/normalizer.json')['normalizer']
            prepared, normalizer = load_learning_data(ROOT/'runs/p3-02-20261009-01/learning_data',
                meta['schema'], meta['dataset_identity'], meta['normalizer_hash'])
            del prepared
            gc.collect()
            models, identities = load_models(normalizer, physical, read(ROOT/'configs/p4_development_e1.v1.json'))
            selected = models['R']
            loaded_model_ids = list(models)
            del models
            gc.collect()
            data = load_development_snapshot(ROOT/design['source']['source_folder'],
                design['source']['source_tick'], physical, profile)
            for path, expected in data.source_files_sha256.items():
                if bindings.get(str(Path(path).resolve())) != expected:
                    raise ValueError('Development snapshot input lacks prebound source SHA')
            env = PlanarPushEnv(physical)
            learned = LearnedPredictor(selected.model, normalizer)

            def predictor(state, action):
                ledger['model_calls'] += 1
                ledger['model_rows'] += len(state)
                return learned(state, action)

            identity = {'model': identities['R'], 'normalizer': {
                'normalizer_hash': meta['normalizer_hash'], 'schema': meta['schema'],
                'dataset_identity': meta['dataset_identity']},
                'frozen_candidate_sha256': sha(ROOT/'configs/p5_freeze_candidate.v1.json')}
            origin = {'source': design['source'], 'development_case': data.case,
                      'input_proof': data.proof, 'source_files_sha256': data.source_files_sha256,
                      'native_design_sha256': sha(DESIGN),
                      'native_parent_manifest_sha256': sha(out/'manifest.json')}
            native_phase[0] = 'actual_controller_execution'
            guard.before_stage()
            result = run_recorded_episode(env, data.snapshot, folder=child/'episode',
                identity=identity, source_bindings=bindings, profile=profile, guard=guard,
                raw_codec=codec, reference_manifest=(out/'reference/episode/manifest.json'
                    if attempt == 'fresh' else None),
                case_id='native-existing-development-start', episode_index=data.case['episode_index'],
                controller_id='R', execution_id=f'{out.name}/{attempt}', predictor=predictor,
                physical_config=physical, planner_config=planner, geometry_parameters=geometry,
                transition_parameters=transition, candidate_seed_base=4711, candidate_stream_id=0,
                max_intervals=300, run_scope='development_preflight',
                penetration_budget_m=profile['physics_suite']['actual_contact_penetration_max_m'],
                initial_status=data.status, initial_context=data.context, initial_gctx=data.gctx, origin=origin)
            native_phase[0] = 'post_execution_storage_audit'
            audit = {'full_input_roundtrips': [], 'all_saved_file_hashes_verified': True}
            verify(result.manifest['files_sha256'])
            for receipt in result.manifest['full_traces']:
                full_input, full_arrays, full_meta = read_full_trace(receipt)
                planned = result.execution.plan_results.get(receipt['tick'])
                if planned is not None:
                    expected = {'candidates': planned.candidates.actions,
                        'predicted_states': planned.predictions.states, 'raw_deltas': planned.predictions.raw_deltas,
                        **final_index_arrays(planned)}
                    comparison = exact_comparison(full_arrays, expected)
                    if comparison['status'] != 'PASS_EXACT':
                        raise ValueError('Full trace array roundtrip failed')
                else:
                    comparison = {'status': 'PASS_NO_MODEL_PLAN_AT_INPUT_TICK'}
                if full_input['rng_keys'] != next(row['rng_keys'] for row in result.execution.decisions
                                                 if row['index'] == receipt['tick']):
                    raise ValueError('Saved full input RNG differs from actual decision')
                current_capture = result.execution.captured_inputs[receipt['tick']]
                actual_input = {key: full_input[key] for key in current_capture}
                input_comparison = exact_comparison(actual_input, current_capture)
                if input_comparison['status'] != 'PASS_EXACT':
                    raise ValueError('Full input differs from actual predecision capture')
                if (full_input['identity'] != identity or full_input['physical_config'] != physical
                        or full_input['planner_config'] != planner
                        or full_input['source_bindings'] != bindings
                        or full_input['episode_index'] != data.case['episode_index']):
                    raise ValueError('Full input omits actual model/config/source/episode binding')
                audit['full_input_roundtrips'].append({'tick': receipt['tick'],
                    'actual_capture_comparison': input_comparison,
                    'input_exact_sha256': receipt['input_exact_sha256'], 'arrays': comparison,
                    'bytes': sum(receipt['bytes'].values())})
                del full_input, full_arrays
            sizes = measure_saved_cell(child, result, codec)
            summary = {'status': result.status, 'execution': result.execution.summary(),
                'comparison': result.comparison, 'loaded_frozen_model_ids': loaded_model_ids,
                'manifest_path': None if result.manifest_path is None else str(result.manifest_path),
                'manifest_sha256': None if result.manifest_path is None else sha(result.manifest_path),
                'ledger': ledger, 'source_before': source_before, 'source_after': verify(bindings),
                'storage': sizes, 'audit': audit, 'combined_process_peak_rss_bytes': peak_rss(),
                'wall_time_s': time.perf_counter()-started,
                'new_learning_updates': 0, 'actual_final_family_count': 0,
                'scope': 'Actual development R controller+physics+lossless recording+fresh exact comparison'}
            # This explicit destruction matters for a same-process caller; the
            # native harness additionally ends the child before the next cell.
            del result, env, data, selected, learned, normalizer
            gc.collect()
            summary['RSS_highwater_after_cleanup_bytes'] = peak_rss()
            summary['cleanup_completed'] = True
            summary['combined_process_peak_rss_bytes'] = peak_rss()
            summary['cell_outputs_bytes'] = tree_bytes(child)
            save(child/'summary.json', summary)
            print(json.dumps({'attempt': attempt, 'status': summary['status'],
                'intervals': summary['ledger']['action_intervals_returned'],
                'peak_rss_bytes': summary['combined_process_peak_rss_bytes'],
                'wall_time_s': summary['wall_time_s']}, allow_nan=False), flush=True)
            if summary['status'] != 'COMPLETE_RECORDED':
                return 2
            return 0
    except Exception as error:
        failure = {'status': 'STOP_NATIVE_PREFLIGHT', 'attempt': attempt,
            'error': {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()},
            'ledger': ledger, 'combined_process_peak_rss_bytes': peak_rss(),
            'wall_time_s': time.perf_counter()-started, 'current_output_bytes': tree_bytes(out),
            'prefix_manifest': None if result is None or result.manifest_path is None else str(result.manifest_path),
            'automatic_retry': False}
        save(child/'failure.json', failure)
        print(json.dumps({'attempt': attempt, 'status': 'STOP_NATIVE_PREFLIGHT',
                          'error': failure['error']['message']}, allow_nan=False), flush=True)
        return 2


def final_index_arrays(planned):
    from push_align.final_driver import plan_index_records_from_plan
    return plan_index_records_from_plan(planned)


def measure_saved_cell(child, result, codec):
    """Storage-only measurements of actual just-saved cell values, retained."""
    from push_align.final_recording import write_plan_index, write_witness_gzip
    measurements = child/'storage_measurements'
    measurements.mkdir(exist_ok=False)
    result_measurement = {'cell_bytes_before_measurements': tree_bytes(child),
        'intervals': len(result.execution.results), 'decisions': len(result.execution.decisions),
        'learned_plans': len(result.execution.plan_results),
        'full_trace_and_input_total_bytes': sum(sum(row['bytes'].values()) for row in result.manifest['full_traces']),
        'full_trace_and_input_max_bytes': max((sum(row['bytes'].values()) for row in result.manifest['full_traces']), default=0)}
    if result.manifest['raw'] is not None:
        empty = codec.write_raw(measurements/'empty', result.execution.initial_snapshot, [], xz_preset=6)
        fixed = sum(empty['bytes'].values())
        total = sum(result.manifest['raw']['bytes'].values())
        result_measurement.update(raw_total_bytes=total, raw_fixed_bytes=fixed,
            raw_increment_bytes=total-fixed, raw_average_interval_bytes=(total-fixed)/len(result.execution.results)
                if result.execution.results else None)
    if result.manifest['plan_index'] is not None:
        empty_index = write_plan_index(measurements/'empty_index.zip', [])
        fixed = sum(empty_index['bytes'].values())
        total = sum(result.manifest['plan_index']['bytes'].values())
        result_measurement.update(plan_index_total_bytes=total, plan_index_fixed_bytes=fixed,
            plan_index_average_record_bytes=(total-fixed)/len(result.execution.plan_results)
                if result.execution.plan_results else None)
    receipt = write_witness_gzip(measurements/'witness_members.json.gz', result.witness)
    result_measurement['witness'] = receipt
    result_measurement['measurement_files_sha256'] = {
        str(path): sha(path) for path in measurements.rglob('*') if path.is_file()}
    return result_measurement


def run_parent(out):
    started = time.perf_counter()
    design = checked_design()
    if out.exists() or REPORT.exists():
        raise FileExistsError('Preserve existing native attempt/report; no automatic retry')
    if shutil.disk_usage(ROOT).free < design['whole_bounded_probe_required_bytes']:
        raise RuntimeError('Bounded native budget plus reserve does not fit')
    scientific = read(ROOT/'configs/p4_first_moment.v1.json')['bindings_sha256']
    frozen = read(ROOT/'configs/p5_freeze_candidate.v1.json')['source_sha256']
    scientific = {str((ROOT/path).resolve()): expected for path, expected in scientific.items()}
    frozen = {str((ROOT/path).resolve()): expected for path, expected in frozen.items()}
    verify(scientific)
    verify(frozen)
    graph = source_graph()
    explicit = [DESIGN, ROOT/'configs/p5_execution_preflight.design.v1.json',
        ROOT/'configs/p5_freeze_candidate.v1.json', ROOT/'configs/p4_first_moment.v1.json',
        ROOT/'configs/p4_development_e1.v1.json', ROOT/'runs/p4-04-20261013-01/manifest.json']
    source_folder = ROOT/design['source']['source_folder']
    source_inputs = [source_folder/name for name in ('manifest.json', 'steps.jsonl',
        'raw_execution.npz', 'raw_execution.json.gz', 'plans/step-000/identity.json')]
    new_bindings = {str(path.resolve()): sha(path) for path in graph+explicit+source_inputs}
    for collection in (frozen, new_bindings):
        for path, expected in collection.items():
            if path in scientific and scientific[path] != expected:
                raise ValueError('Conflicting scientific/source binding')
            if path in frozen and frozen[path] != expected:
                raise ValueError('Conflicting frozen/source binding')
    bindings = {**scientific, **frozen, **new_bindings}
    out.mkdir(parents=True, exist_ok=False)
    for path in graph+explicit:
        destination = out/'source'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    manifest = {'schema': 'p5-native-driver-run-v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'as_of_client_date': '2026-10-16', 'design': design, 'design_sha256': sha(DESIGN),
        'source_graph_files': [str(path.relative_to(ROOT)) for path in graph],
        'scientific_bindings_count': len(scientific), 'frozen_candidate_bindings_count': len(frozen),
        'all_bound_files_sha256': bindings,
        'free_bytes_before': shutil.disk_usage(ROOT).free, 'python': sys.version, 'platform': platform.platform(),
        'final_test_used': False, 'actual_final_family_count': 0, 'new_training_updates': 0,
        'child_process_policy': 'Sequential fresh process per attempt; no automatic retry'}
    save(out/'manifest.json', manifest)
    completed = []
    terminal = None
    for attempt in ('reference', 'fresh'):
        if tree_bytes(out) > design['maximum_new_output_bytes']:
            terminal = 'STOP_OUTPUT_LIMIT_BEFORE_NEXT_CHILD'
            break
        if shutil.disk_usage(ROOT).free < design['minimum_free_reserve_bytes']+design['cell_write_allowance_bytes']:
            terminal = 'STOP_FREE_LIMIT_BEFORE_NEXT_CHILD'
            break
        with (out/f'{attempt}.stdout.log').open('x') as stdout, (out/f'{attempt}.stderr.log').open('x') as stderr:
            proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--out', str(out),
                '--child', attempt], cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
        completed.append({'attempt': attempt, 'exit_code': proc.returncode})
        print(json.dumps(completed[-1]), flush=True)
        if proc.returncode != 0:
            terminal = 'STOP_CHILD_FAILURE_NO_RETRY'
            break
    summaries = {attempt: read(out/attempt/'summary.json') for attempt in ('reference', 'fresh')
                 if (out/attempt/'summary.json').is_file()}
    actual_ledgers = {}
    for attempt in ('reference', 'fresh'):
        path = out/attempt/'execution_ledger.jsonl'
        if path.is_file():
            rows = []
            for line in path.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    break  # An interrupted final line never erases prior attempts.
            if rows:
                actual_ledgers[attempt] = rows[-1]
    intervals = sum(row['action_interval_attempts'] for row in actual_ledgers.values())
    failures = [read(path) for path in out.glob('*/failure.json')]
    source_after = verify(bindings)
    fresh = summaries.get('fresh')
    reference = summaries.get('reference')
    checks = {'both_attempts_completed_recording': len(summaries) == 2 and all(
            row['status'] == 'COMPLETE_RECORDED' for row in summaries.values()),
        'actual_independent_controller_and_physics': len(summaries) == 2 and all(
            row['ledger']['action_intervals_returned'] > 0 and row['ledger']['model_calls'] > 0
            for row in summaries.values()),
        '600_action_cap': intervals <= design['maximum_new_action_intervals'],
        'output_limit': tree_bytes(out) <= design['maximum_new_output_bytes'],
        'free_reserve': shutil.disk_usage(ROOT).free >= design['minimum_free_reserve_bytes'],
        'source_bindings_unchanged': source_after['all_unchanged'],
        'future_physics_zero': len(summaries) == 2 and all(
            sum(row['execution']['future_engine_calls'].values()) == 0 for row in summaries.values()),
        'same_terminal': bool(reference and fresh and
            reference['execution']['termination'] == fresh['execution']['termination']),
        'boundary_PASS_REPRODUCTION': bool(fresh and fresh.get('comparison') and
            fresh['comparison']['boundary']['status'] == 'PASS_REPRODUCTION'),
        'raw_PASS_EXACT': bool(fresh and fresh.get('comparison') and
            fresh['comparison']['raw_exact']['status'] == 'PASS_EXACT'),
        'all_index_PASS_EXACT': bool(fresh and fresh.get('comparison') and
            fresh['comparison']['plan_index_exact']['status'] == 'PASS_EXACT'),
        'combined_peak_within_checkpoint': len(summaries) == 2 and all(
            row['combined_process_peak_rss_bytes'] <= design['memory']['pre_action_stop_RSS_bytes']
            for row in summaries.values()),
        'no_final_families_or_training': all(row['new_learning_updates'] == 0 and
            row['actual_final_family_count'] == 0 for row in summaries.values())}
    report = {'schema': 'p5-native-driver-preflight-review-v1',
        'status': 'PASS_BOUNDED_NATIVE_DRIVER_PREFLIGHT' if all(checks.values()) and terminal is None else 'HOLD',
        'checks': checks, 'stop': terminal, 'run': str(out), 'design_sha256': sha(DESIGN),
        'source_code_sha256': {str(path.relative_to(ROOT)): sha(path) for path in graph},
        'original_bindings_verified': source_after, 'attempts': completed,
        'summaries': summaries, 'failures': failures, 'total_new_action_interval_attempts': intervals,
        'actual_last_execution_ledgers': actual_ledgers,
        'total_new_action_intervals_returned': sum(row['action_intervals_returned']
            for row in actual_ledgers.values()),
        'whole_run_bytes_before_report': tree_bytes(out), 'free_bytes_after': shutil.disk_usage(ROOT).free,
        'combined_peak_rss_bytes': max((row['combined_process_peak_rss_bytes'] for row in summaries.values()), default=None),
        'wall_time_s': time.perf_counter()-started, 'actual_final_test_generated': False,
        'learning_updates': 0, 'P5_01': 'HOLD',
        'limits': ['One existing R development condition only; not all controller/machine memory bounds.',
                   'Memory highwater is measured; pre-action checkpoint is not an OS guarantee.',
                   'No per-action power-loss durability; all existing original records remain untouched.',
                   'A natural early terminal retains its real duration and limits300-step extrapolation.'],
        'output_files_sha256': {str(path): sha(path) for path in out.rglob('*') if path.is_file()}}
    save(REPORT, report)
    print(json.dumps({'status': report['status'], 'interval_attempts': intervals,
        'run_bytes': report['whole_run_bytes_before_report'], 'peak_rss_bytes': report['combined_peak_rss_bytes'],
        'report': str(REPORT)}, allow_nan=False), flush=True)
    return 0 if report['status'] == 'PASS_BOUNDED_NATIVE_DRIVER_PREFLIGHT' else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--child', choices=('reference', 'fresh'))
    args = parser.parse_args()
    out = args.out.resolve()
    if args.child:
        return run_child(out, args.child)
    return run_parent(out)


if __name__ == '__main__':
    raise SystemExit(main())
