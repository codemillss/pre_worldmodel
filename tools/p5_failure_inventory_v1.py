#!/usr/bin/env python3
"""Post hoc endpoint/failure inventory from sealed E2 rows and witnesses.

Read-only with respect to experimental inputs/results. No project imports,
model inference, physical future, success relabeling or parameter changes.
Boundary-held samples are not a reconstruction of all physical substeps.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def build(run):
    manifest = read(run / 'manifest.json')
    assert sha(run / 'manifest.json') == read(run / 'freeze_receipt.json')['sha256']
    expected = manifest['E2']
    assert len(expected) == 96 and set(read(run / 'progress.json')['E2_done']) == {c['cell_id'] for c in expected}
    task = manifest['physical_config']['task']
    thresholds = {
        'position': ('final_position_error_m', task['position_tolerance_m']),
        'yaw': ('final_yaw_error_rad', task['yaw_tolerance_rad']),
        'linear_speed': ('final_linear_speed_m_s', task['linear_speed_tolerance_m_s']),
        'angular_speed': ('final_angular_speed_rad_s', task['angular_speed_tolerance_rad_s']),
    }
    bindings = {str(run / 'manifest.json'): sha(run / 'manifest.json')}
    cases = []
    for cell in expected:
        folder = run / 'E2' / cell['cell_id']
        result_path = folder / 'worker_result.json'
        assert sha(result_path) == read(folder / 'worker_result_receipt.json')['sha256']
        result = read(result_path)
        assert result['cell'] == cell and result['row']['execution_status'] == 'completed'
        row = result['row']
        assert row['success'] == (row['termination'] == 'ready')
        record_path = folder / 'manifest.json'
        assert sha(record_path) == result['record_manifest_sha256']
        record = read(record_path)
        witness_path = Path(record['witness_path'])
        assert witness_path.is_relative_to(folder) and sha(witness_path) == record['files_sha256'][str(witness_path)]
        witness = json.loads(gzip.decompress(witness_path.read_bytes()))
        assert witness['termination'] == row['termination'] and witness['counts'] == row['counts']
        statuses = [witness['final_task']]
        for interval in witness['intervals']:
            statuses.extend((interval['task_before'], interval['task_after']))
        last = witness['decisions'][-1] if witness['decisions'] else {}
        plan = last.get('plan_identity') or {}
        geometry = last.get('geometry_proposal') or {}
        flags = {name: None if row[metric] is None else row[metric] <= threshold
                 for name, (metric, threshold) in thresholds.items()}
        entry = {
            **cell, 'termination': row['termination'], 'success': row['success'],
            'elapsed_s': row['elapsed_s'], 'metric_state_origin': row['metric_state_origin'],
            'metric_state_time_s': row['metric_state_time_s'],
            'final_observation_available': row['final_observation_available'],
            'endpoint_criteria_met': flags,
            'endpoint_criteria_not_met': [k for k, v in flags.items() if v is False],
            'endpoint_metrics': {metric: row[metric] for metric, _ in thresholds.values()},
            'final_held_time_s': witness['final_task']['held_time_s'],
            'boundary_sampled_held_time_max_s': max(s['held_time_s'] for s in statuses),
            'last_decision': {
                'index': last.get('index'), 'source': last.get('source'),
                'executed': last.get('executed'), 'holding': last.get('holding'),
                'plan_reason': plan.get('reason'), 'invalid_reasons': plan.get('invalid_reasons'),
                'valid_predictions': plan.get('valid_predictions'),
                'eligible_costs': plan.get('eligible_costs'),
                'geometry_feasible': geometry.get('feasible'),
                'geometry_reason': geometry.get('reason'),
            },
        }
        if row['success']:
            assert all(v is True for v in flags.values())
            assert entry['final_held_time_s'] + 1e-9 >= task['hold_time_s']
        cases.append(entry)
        for path in (result_path, record_path, witness_path):
            bindings[str(path)] = sha(path)
    assert len({c['cell_id'] for c in cases}) == 96
    def aggregate(part):
        failures = [c for c in part if not c['success']]
        return {
            'episodes': len(part), 'successes': sum(c['success'] for c in part),
            'termination_counts': dict(Counter(c['termination'] for c in part)),
            'failure_endpoint_criterion_not_met_counts': dict(Counter(k for c in failures for k in c['endpoint_criteria_not_met'])),
            'failures_with_all_four_endpoint_criteria_met': sum(all(v is True for v in c['endpoint_criteria_met'].values()) for c in failures),
            'failure_joint_endpoint_labels': dict(Counter('+'.join(c['endpoint_criteria_not_met']) or 'none_or_unavailable' for c in failures)),
        }
    result = {
        'schema': 'p5-posthoc-E2-failure-inventory-v1',
        'status': 'COMPLETE_96_SAVED_WITNESS_ENDPOINT_INVENTORY',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'run': str(run), 'posthoc_descriptive_analysis': True,
        'overall': aggregate(cases),
        'by_controller': {c: aggregate([r for r in cases if r['controller_id'] == c]) for c in manifest['visualization']['controllers']},
        'cases': cases, 'source_sha256': bindings,
        'limits': [
            'Endpoint failures describe the saved final/last-valid observation, not a causal explanation for the entire trajectory.',
            'Maximum boundary-sampled held time does not reconstruct qualifications between action boundaries; original sealed raw review supplies success acceptance.',
            'Predicted invalidity is not proof of actual candidate infeasibility; no extra candidate future was run here.',
            'Original success labels, thresholds, denominators and scientific settings are not changed.',
            'This extra analysis was written after E2 outcomes and is not a preregistered new performance endpoint.',
        ],
        'model_calls': 0, 'physics_calls': 0, 'training_updates': 0,
    }
    assert all(sha(Path(p)) == h for p, h in bindings.items())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.out.resolve()
    assert not output.exists() and output.is_relative_to(run.parent.parent / 'reports')
    result = build(run)
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'overall': result['overall'], 'out': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
