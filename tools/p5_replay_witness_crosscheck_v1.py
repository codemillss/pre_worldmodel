#!/usr/bin/env python3
"""Independently crosscheck all33 saved replay witnesses using only stdlib.

Run once after all declared replays are complete and the run is quiescent.
This does not launch a controller, import project modules, decompress raw/index
archives, retry work, or change producer verdicts. Only witness JSON gzip is
decompressed, with explicit limits. All raw/source files are SHA-streamed only.
Scientific identity must agree; attempt/execution identities must differ.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re

CONTROLLERS = ('G', 'R', 'S1-11', 'S1-23', 'S1-47', 'SK-11', 'SK-23', 'SK-47')
STRATA = ('translation', 'rotation', 'offset')
ENGINE_GUARDS = ('mj_step', 'mj_step1', 'mj_step2', 'mj_forward', 'mj_inverse')
COUNT_FIELDS = ('controller_decisions', 'physics_interval_attempts',
                'physics_intervals_returned', 'completed_boundary_records')
RULE = ('per controller/learning seed: minimum episode ID per stratum UNION '
        'minimum ID per observed termination type; deduplicate')
TERMINATIONS = {'ready', 'timeout', 'planner_invalid', 'controller_infeasible',
                'physics_error', 'geometry_invalid', 'workspace_exit', 'planarity_violation'}
SCHEMA = 'p5-saved-replay-witness-crosscheck-v1'
MAX_JSON_BYTES = 16 * 1024**2
MAX_GZIP_BYTES = 8 * 1024**2
MAX_WITNESS_EXPANDED_BYTES = 16 * 1024**2
MAX_JSON_ITEMS = 1000000
MAX_JSON_DEPTH = 64
MAX_OUTPUT_BYTES = 4 * 1024**2
SHA = re.compile(r'[0-9a-f]{64}\Z')
IDENTITY_KEYS = ('family_id', 'episode_id', 'episode_index', 'stratum', 'controller_id', 'cell_id')
PAYLOAD_FIELDS = ('decisions', 'intervals', 'final_task', 'counts', 'termination',
                  'unexecuted_plan', 'future_engine_calls')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def same(left, right):
    return canonical(left) == canonical(right)


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'Duplicate JSON object key: ' + key)
        result[key] = value
    return result


def strict_json(data):
    def invalid(value):
        raise ValueError('Nonfinite JSON number: ' + value)
    def integer(value):
        need(len(value) <= 128, 'Oversized JSON integer')
        return int(value)
    value = json.loads(data, object_pairs_hook=no_duplicates,
                       parse_constant=invalid, parse_int=integer)
    count = 0
    def walk(item, depth):
        nonlocal count
        count += 1
        need(count <= MAX_JSON_ITEMS and depth <= MAX_JSON_DEPTH, 'JSON size/depth limit exceeded')
        if type(item) is float:
            need(math.isfinite(item), 'Nonfinite/overflowed JSON float')
        elif type(item) is dict:
            for child in item.values(): walk(child, depth + 1)
        elif type(item) is list:
            for child in item: walk(child, depth + 1)
    walk(value, 0)
    return value


class Evidence:
    def __init__(self, run):
        self.run = Path(run).resolve()
        self.root = self.run.parent.parent
        need(self.run.parent.name == 'runs' and self.run.name.startswith('p5-final-'),
             'Explicit managed p5-final run required')
        self.bindings = {}
        self.bytes_hashed = 0

    def path(self, value, scope=None):
        need(isinstance(value, (str, Path)), 'Explicit evidence path required')
        raw = Path(value)
        need('..' not in raw.parts, 'Parent traversal is forbidden: ' + str(value))
        raw = raw if raw.is_absolute() else self.root / raw
        path = raw.resolve()
        need(path.is_relative_to(self.root) and path.is_file(), 'Missing/out-of-project evidence: ' + str(value))
        if scope is not None:
            need(path.is_relative_to(Path(scope).resolve()), 'Evidence escapes expected scope: ' + str(value))
        probe = raw
        while probe != self.root and probe.is_relative_to(self.root):
            need(not probe.is_symlink(), 'Evidence symlink is forbidden: ' + str(probe))
            probe = probe.parent
        return path

    def digest(self, path):
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024**2), b''):
                value.update(block)
                self.bytes_hashed += len(block)
        return value.hexdigest()

    def bind(self, value, expected=None, scope=None):
        path = self.path(value, scope)
        key = str(path)
        if expected is not None:
            need(type(expected) is str and SHA.fullmatch(expected), 'Explicit SHA256 required: ' + key)
        actual = self.bindings.get(key)
        if actual is None:
            actual = self.digest(path)
            self.bindings[key] = actual
        need(expected is None or actual == expected, 'Evidence SHA256 mismatch: ' + key)
        return path

    def json(self, value, expected=None, scope=None, *, witness=False):
        path = self.bind(value, expected, scope)
        limit = MAX_GZIP_BYTES if witness else MAX_JSON_BYTES
        need(path.stat().st_size <= limit, 'JSON/compressed witness exceeds size limit: ' + str(path))
        if witness:
            with gzip.open(path, 'rb') as stream:
                data = stream.read(MAX_WITNESS_EXPANDED_BYTES + 1)
            need(len(data) <= MAX_WITNESS_EXPANDED_BYTES, 'Expanded witness exceeds size limit')
        else:
            data = path.read_bytes()
            need(len(data) <= MAX_JSON_BYTES, 'JSON size changed beyond limit')
        return strict_json(data)

    def graph(self, values, scope=None):
        need(type(values) is dict and bool(values), 'Nonempty evidence SHA map required')
        for path, digest in values.items(): self.bind(path, digest, scope)

    def recheck(self):
        for path, digest in self.bindings.items():
            need(self.digest(self.path(path)) == digest, 'Input changed during crosscheck: ' + path)


def guards(value):
    need(type(value) is dict and set(value) == set(ENGINE_GUARDS)
         and all(type(n) is int and n == 0 for n in value.values()), 'Future-engine guards are not explicit zeroes')


def vector(value, size):
    need(type(value) is list and len(value) == size
         and all(type(x) in (int, float) and math.isfinite(x) for x in value),
         'Finite JSON numeric vector of size ' + str(size) + ' required')


def worker(evidence, manifest, phase, cell):
    folder = evidence.run / ('E2' if phase == 'E2' else 'replays') / cell['family_id'] / cell['controller_id']
    seal = evidence.json(folder / 'worker_result_receipt.json', scope=folder)
    value = evidence.json(folder / 'worker_result.json', seal.get('sha256'), folder)
    row = value.get('row')
    need(value.get('phase') == phase and same(value.get('cell'), cell)
         and type(row) is dict and all(same(row.get(k), v) for k, v in cell.items()), 'Worker phase/cell/row identity differs')
    need(value.get('status') in {'ACCEPT_COMPLETED_RECORDING', 'ACCEPT_COMPLETED_TASK_FAILURE'}
         and row.get('execution_status') == 'completed' and row.get('termination') in TERMINATIONS
         and type(row.get('success')) is bool and row['success'] == (row['termination'] == 'ready'),
         'Worker has unresolved or invalid task outcome')
    guards(row.get('future_engine_calls'))
    return folder, value


def record_and_witness(evidence, manifest, manifest_sha, phase, cell, folder, value):
    path = evidence.bind(value.get('record_manifest'), value.get('record_manifest_sha256'), folder)
    need(path == folder / 'manifest.json', 'Worker recording path differs')
    record = evidence.json(path)
    role = 'reference_controller_execution' if phase == 'E2' else 'fresh_controller_replay'
    scientific = {'study': 'P5-final', 'family_id': cell['family_id'], 'controller_id': cell['controller_id'],
                  'model': None if cell['controller_id'] == 'G' else manifest['model_identities'][cell['controller_id']],
                  'normalizer': None if cell['controller_id'] == 'G' else manifest['normalizer_identity'],
                  'final_manifest_sha256': manifest_sha}
    sources = {str((evidence.root / p).resolve()): h for p, h in
               {**manifest['source_sha256'], **manifest['runtime_source_sha256']}.items()}
    sources[str(evidence.run / 'manifest.json')] = manifest_sha
    need(record.get('schema') == 'p5-recorded-episode-v1' and record.get('record_kind') == role
         and record.get('recording_status') == 'COMPLETE_RECORDED'
         and same(record.get('identity'), scientific) and same(record.get('source_bindings'), sources),
         'Recording role/status/scientific identity/source graph differs')
    execution = record.get('execution', {})
    for key in ('execution_status', 'termination', 'counts', 'future_engine_calls'):
        need(same(execution.get(key), value['row'].get(key)), 'Worker row differs from recording: ' + key)
    evidence.graph(record.get('files_sha256'), folder)
    review = evidence.json(folder / 'review.json', value.get('review_sha256'), folder)
    component = evidence.json(folder / 'component_budget_review.json', value.get('component_review_sha256'), folder)
    need(review.get('accepted') is True and review.get('status') == value['status']
         and review.get('manifest_sha256') == value['record_manifest_sha256']
         and type(review.get('metrics')) is dict
         and all(same(value['row'].get(k), v) for k, v in review['metrics'].items())
         and component.get('status') == 'PASS_MEASURED_COMPONENT_ALLOWANCES', 'Recording review/component receipt unresolved')
    witness_path = evidence.path(record.get('witness_path'), folder)
    need(str(witness_path) in record['files_sha256'], 'Witness is outside sealed recording file graph')
    witness = evidence.json(witness_path, record['files_sha256'][str(witness_path)], folder, witness=True)
    need(witness.get('schema') == 'p5-controller-boundary-witness-v1' and witness.get('record_kind') == role
         and same(witness.get('identity'), scientific) and same(witness.get('source_bindings'), sources)
         and witness.get('controller_id') == cell['controller_id'] and witness.get('case_id') == cell['family_id']
         and witness.get('run_scope') == 'final_evaluation' and witness.get('initial_full_snapshot_equal') is True
         and witness.get('execution_status') == 'completed' and witness.get('error') is None,
         'Saved witness identity/scope/completion differs')
    for key in ('termination', 'counts', 'future_engine_calls'):
        need(same(witness.get(key), value['row'].get(key)), 'Saved witness differs from recorded outcome: ' + key)
    validate_witness(witness)
    return record, witness


def validate_witness(witness):
    decisions, intervals, counts = witness.get('decisions'), witness.get('intervals'), witness.get('counts')
    need(type(decisions) is list and type(intervals) is list and len(decisions) <= 301 and len(intervals) <= 300,
         'Bounded explicit decision/interval arrays required')
    need(type(counts) is dict and set(counts) == set(COUNT_FIELDS)
         and all(type(n) is int and 0 <= n <= 301 for n in counts.values()), 'Explicit bounded execution counts required')
    need(counts['controller_decisions'] == len(decisions)
         and counts['physics_interval_attempts'] == counts['physics_intervals_returned'] ==
         counts['completed_boundary_records'] == len(intervals) == witness.get('returned_results')
         and len(decisions) == len(intervals) + int(witness.get('unexecuted_plan') is not None),
         'Witness decision/interval counts differ')
    vector(witness.get('initial', {}).get('state'), 10)
    guards(witness.get('future_engine_calls'))
    for index, row in enumerate(decisions):
        need(type(row) is dict and type(row.get('index')) is int and row['index'] == index,
             'Decision index population differs')
        need(row.get('source') in {'model', 'geometry', 'holding'} and 'plan_identity' in row,
             'Explicit decision source/plan identity required')
        need((type(row['plan_identity']) is dict and bool(row['plan_identity']))
             if row['source'] == 'model' else row['plan_identity'] is None,
             'Learned plan identity differs from decision source')
        vector(row.get('state'), 10)
    for index, row in enumerate(intervals):
        need(type(row) is dict and type(row.get('index')) is int and row['index'] == index,
             'Interval index population differs')
        for key, size in (('state_before', 10), ('state_after', 10), ('requested_action', 2), ('applied_action', 2)):
            vector(row.get(key), size)
    if witness.get('unexecuted_plan') is not None:
        need(bool(decisions) and same(witness['unexecuted_plan'], decisions[-1]), 'Unexecuted final decision differs')


def expected_population(manifest):
    families = manifest.get('families')
    need(type(families) is list and len(families) == 12
         and len({f['family_id'] for f in families}) == 12
         and Counter(f['stratum'] for f in families) == Counter({s: 4 for s in STRATA}), 'Frozen twelve-family strata differ')
    result = []
    for family in families:
        need(family['family_id'] == family['episode_id'] and '/' not in family['family_id'], 'Unsafe/ambiguous family ID')
        for controller in CONTROLLERS:
            result.append({**{key: family[key] for key in ('family_id', 'episode_id', 'episode_index', 'stratum')},
                           'controller_id': controller, 'cell_id': family['episode_id'] + '/' + controller})
    need(same(manifest.get('E2'), result), 'Frozen exact expected96 identities differ')
    return result


def select_replays(rows):
    selected = []
    for controller in CONTROLLERS:
        part = [row for row in rows if row['controller_id'] == controller]
        need(len(part) == 12, 'Replay selection lacks complete controller population')
        chosen = set()
        for stratum in STRATA:
            chosen.add(min((row for row in part if row['stratum'] == stratum), key=lambda r: r['episode_id'])['cell_id'])
        for termination in sorted({row['termination'] for row in part}):
            chosen.add(min((row for row in part if row['termination'] == termination), key=lambda r: r['episode_id'])['cell_id'])
        selected.extend(row for row in sorted(part, key=lambda r: r['episode_id']) if row['cell_id'] in chosen)
    return selected


def crosscheck(run, expected_manifest_sha256):
    tool_path = Path(__file__).resolve()
    tool_before = hashlib.sha256(tool_path.read_bytes()).hexdigest()
    result = {'schema': SCHEMA, 'status': 'FAIL_RETAIN_DIAGNOSTICS', 'accepted': False,
              'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'run': str(run),
              'expected_replays': 33, 'verified_replays': 0, 'cases': [], 'errors': []}
    evidence = None
    try:
        evidence = Evidence(run)
        need(type(expected_manifest_sha256) is str and SHA.fullmatch(expected_manifest_sha256), 'Explicit frozen manifest SHA required')
        freeze = evidence.json(evidence.run / 'freeze_receipt.json')
        manifest_path = evidence.bind(evidence.run / 'manifest.json', expected_manifest_sha256)
        need(freeze.get('sha256') == expected_manifest_sha256 and freeze.get('bytes') == manifest_path.stat().st_size,
             'Frozen manifest receipt differs')
        manifest = evidence.json(manifest_path)
        need(manifest.get('schema') == 'p5-frozen-final-manifest-v1'
             and manifest.get('status') == 'FROZEN_BEFORE_FINAL_OUTCOMES'
             and manifest.get('replay_selection_rule') == RULE, 'Frozen study/replay rule differs')
        expected = expected_population(manifest)
        progress_seal = evidence.json(evidence.run / 'progress_receipt.json')
        progress = evidence.json(evidence.run / 'progress.json', progress_seal.get('sha256'))
        selection = evidence.json(evidence.run / 'replay_selection.json')
        completed = progress.get('replay_done')
        need(type(completed) is list and len(completed) == len(set(completed)) == 33,
             'All exactly33 declared replays must complete before this crosscheck')
        need(type(progress.get('E2_done')) is list and len(progress['E2_done']) == 96
             and set(progress['E2_done']) == {c['cell_id'] for c in expected}, 'Original96 E2 completion population differs')
        evidence.graph(manifest.get('source_sha256'))
        need(type(manifest.get('runtime_source_sha256')) is dict
             and all(manifest['source_sha256'].get(p) == h for p, h in manifest['runtime_source_sha256'].items()),
             'Frozen runtime is not source-bound')
        evidence.graph(manifest['runtime_source_sha256'])
        originals = {c['cell_id']: worker(evidence, manifest, 'E2', c) for c in expected}
        chosen = select_replays([v['row'] for _, v in originals.values()])
        need(len(chosen) == 33 and same(selection, chosen) and same(progress.get('replay_selection'), chosen)
             and set(completed) == {row['cell_id'] for row in chosen}, 'Independent replay union/progress/selection differs')
        result['selection_rule'] = RULE
        result['frozen_manifest_sha256'] = expected_manifest_sha256
        result['selected_cell_ids'] = [row['cell_id'] for row in chosen]
        by_cell = {cell['cell_id']: cell for cell in expected}
        for selected in chosen:
            cell = by_cell[selected['cell_id']]
            item = {'cell_id': cell['cell_id'], 'controller_id': cell['controller_id'], 'accepted': False}
            try:
                folder, original_worker = originals[cell['cell_id']]
                old, left = record_and_witness(evidence, manifest, expected_manifest_sha256, 'E2', cell, folder, original_worker)
                fresh_folder, fresh_worker = worker(evidence, manifest, 'replay', cell)
                new, right = record_and_witness(evidence, manifest, expected_manifest_sha256, 'replay', cell, fresh_folder, fresh_worker)
                need(evidence.path(new.get('reference_manifest'), folder) == folder / 'manifest.json'
                     and new.get('reference_manifest_sha256') == original_worker['record_manifest_sha256'], 'Fresh original-record link differs')
                need(same(fresh_worker.get('comparison'), new.get('comparison')), 'Worker/record producer comparison differs')
                producer = new.get('comparison')
                need(type(producer) is dict, 'Explicit producer comparison required')
                item['producer_comparison'] = producer
                boundary = producer.get('boundary', {})
                need(boundary.get('status') == 'PASS_REPRODUCTION' and boundary.get('comparison_matches') is True
                     and not boundary.get('incomplete_reasons') and not boundary.get('mismatch_reasons')
                     and boundary.get('fresh_full_raw_prefix_required') is False
                     and type(boundary.get('maximum_differences')) is dict, 'Producer boundary comparison is unresolved')
                maxima = boundary['maximum_differences']
                need(bool(maxima) and 'position_m' in maxima and 'state_components' in maxima,
                     'Producer boundary maxima are incomplete')
                vector(maxima['state_components'], 10)
                for maximum in maxima.values():
                    values = maximum if type(maximum) is list else [maximum]
                    need(len(values) <= 10 and all(type(n) in (int, float) and math.isfinite(n) and n >= 0
                                                  for n in values), 'Malformed/nonfinite producer maximum')
                for name in ('raw_exact', 'plan_index_exact'):
                    check = producer.get(name, {})
                    need(check.get('status') == 'PASS_EXACT' and type(check.get('left_sha256')) is str
                         and SHA.fullmatch(check['left_sha256']) and check['left_sha256'] == check.get('right_sha256'),
                         'Producer exact check unresolved: ' + name)
                ticks = producer.get('differing_ticks')
                need(type(ticks) is list and len(ticks) <= 301 and all(type(t) is int and 0 <= t <= 300 for t in ticks)
                     and ticks == sorted(set(ticks)), 'Malformed diagnostic differing_ticks')
                for key in ('attempt_id', 'execution_id'):
                    need(type(left.get(key)) is str and left[key] and type(right.get(key)) is str
                         and right[key] and left[key] != right[key], 'Fresh independent execution identity required: ' + key)
                item['execution_identities'] = {key: {'original': left[key], 'fresh': right[key]} for key in ('attempt_id', 'execution_id')}
                for key in ('identity', 'source_bindings', 'origin', 'controller_id', 'case_id', 'run_scope',
                            'initial_full_snapshot_equal', 'initial', 'task_spec'):
                    need(same(left.get(key), right.get(key)), 'Scientific/initial witness identity differs: ' + key)
                need(type(left.get('raw_reference')) is dict and same(right.get('original_raw_reference'), left['raw_reference'])
                     and same(left['raw_reference'].get('files'), old.get('raw', {}).get('files_sha256')),
                     'Fresh original raw-reference binding differs')
                raw = left['raw_reference']
                need(raw.get('scope') == 'complete_raw_execution_prefix'
                     and raw.get('executed_intervals') == left['counts']['physics_intervals_returned'], 'Original raw-reference scope/count differs')
                evidence.graph(raw.get('files'), folder)
                item['payload_checks'] = {}
                for key in PAYLOAD_FIELDS:
                    need(key in left and key in right, 'Missing explicit witness payload: ' + key)
                    a, b = canonical(left[key]), canonical(right[key])
                    item['payload_checks'][key] = {'exact': a == b, 'original_sha256': hashlib.sha256(a).hexdigest(),
                                                   'fresh_sha256': hashlib.sha256(b).hexdigest()}
                item['learned_plan_decisions'] = sum(row.get('plan_identity') is not None for row in right['decisions'])
                item['G_empty_learned_plan_index'] = cell['controller_id'] == 'G' and item['learned_plan_decisions'] == 0
                need(all(check['exact'] for check in item['payload_checks'].values()), 'Saved normalized witness payload differs')
                item['accepted'] = True
                result['verified_replays'] += 1
            except Exception as error:
                item['error'] = {'type': type(error).__name__, 'message': str(error)}
                result['errors'].append({'cell_id': cell['cell_id'], **item['error']})
            result['cases'].append(item)
        evidence.recheck()
        result['accepted'] = result['verified_replays'] == 33 and not result['errors']
        result['status'] = 'PASS_SAVED_WITNESS_CROSSCHECK' if result['accepted'] else 'FAIL_RETAIN_DIAGNOSTICS'
    except Exception as error:
        result['errors'].append({'phase': 'entry_or_final_integrity', 'type': type(error).__name__, 'message': str(error)})
    if evidence is not None:
        result['input_bindings_sha256'] = evidence.bindings
        result['io'] = {'unique_files': len(evidence.bindings), 'bytes_SHA_streamed': evidence.bytes_hashed,
                        'raw_archives_decompressed': 0, 'only_witness_gzip_decompressed': True}
    tool_after = hashlib.sha256(tool_path.read_bytes()).hexdigest()
    result['supplementary_tool'] = {'path': str(tool_path), 'sha256_before': tool_before,
                                    'sha256_after': tool_after, 'unfrozen_extra_diagnostic': True}
    if tool_before != tool_after:
        result['accepted'] = False
        result['status'] = 'FAIL_RETAIN_DIAGNOSTICS'
        result['errors'].append({'phase': 'tool_integrity', 'message': 'Supplementary tool changed during crosscheck'})
    result['scope'] = 'Supplementary saved JSON-normalized witness equality, not new execution or independent recreation of fresh substep bytes. Producer primary checks are collected as source-bound attestations. differing_ticks is preserved, never counted as actual behavior differences.'
    result['producer_verdicts_modified'] = False
    result['new_physical_calls'] = result['new_model_calls'] = result['project_imports'] = result['automatic_retries'] = 0
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    run, output = args.run.resolve(), args.out.resolve()
    root = run.parent.parent
    need(output.is_relative_to(root) and output.suffix == '.json'
         and output.relative_to(root).parts[0] in {'reports', 'tmp'}, 'New report/tmp JSON output required')
    need(not args.out.is_symlink() and not output.exists(), 'Prior output is never overwritten or retried')
    result = crosscheck(run, args.manifest_sha256)
    data = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode('utf-8') + b'\n'
    if len(data) > MAX_OUTPUT_BYTES:
        result = {'schema': SCHEMA, 'status': 'FAIL_RETAIN_DIAGNOSTICS', 'accepted': False,
                  'run': str(run), 'expected_replays': 33,
                  'errors': [{'phase': 'output_limit', 'message': 'Detailed diagnostic exceeds4MiB',
                              'detailed_bytes': len(data), 'detailed_sha256': hashlib.sha256(data).hexdigest()}],
                  'scope': 'Input artifacts preserved; no producer verdict changed. No automatic retry.'}
        data = canonical(result) + b'\n'
        need(len(data) <= MAX_OUTPUT_BYTES, 'Diagnostic output limit is too small even for failure receipt')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'status': result['status'], 'accepted': result['accepted'], 'out': str(output)}, ensure_ascii=False))
    return 0 if result['accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
