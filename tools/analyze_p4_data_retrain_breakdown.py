"""Post-hoc accounting of already frozen prediction failures, no new forecasts.

Never relaxes validity or computes a complete-case replacement primary effect.
Contact labels describe actual recorded intervals, not predicted contacts.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import hashlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/p4-data-retrain-20261013-02/prediction_evaluation'
OUT = ROOT/'reports/p4_data_retrain_failure_breakdown.json'
ARMS = ('R', 'D00', 'D10', 'D01', 'D11')
PHASES = {'head': 'Y_heads_teacher_conditioned', 'stop': 'S1_free_stop',
          'stop_local': 'S1_teacher_conditioned_stop', 'full_H10': 'core_H10_free'}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def gap(state, config):
    relative = state[6:8]-state[:2]; yaw = state[2]
    local = np.asarray([np.cos(yaw)*relative[0]+np.sin(yaw)*relative[1],
                        -np.sin(yaw)*relative[0]+np.cos(yaw)*relative[1]])
    q = np.abs(local)-np.asarray(config['object']['size_full_xyz_m'][:2])/2
    return float(np.linalg.norm(np.maximum(q, 0))+min(np.max(q), 0)-config['pusher']['radius_m'])


def number(value): return float(value) if np.isfinite(value) else None


def main():
    index = json.loads((RUN/'artifact_hashes.json').read_text())['sha256']
    bindings = {}
    def verify(name):
        path = RUN/name
        if sha(path) != index[name]: raise ValueError('Frozen analysis source changed: '+name)
        bindings[str(path.relative_to(ROOT))] = sha(path)
        return path
    comparison = json.loads(verify('comparison.json').read_text())
    config_path = RUN.parent/'manifest.json'
    config = json.loads(config_path.read_text())['physical_config']
    bindings[str(config_path.relative_to(ROOT))] = sha(config_path)
    development, regression, failures = {}, {}, []
    baseline = json.loads(verify('R/original_validation.json').read_text())
    for arm in ARMS:
        summary = json.loads(verify(arm+'/development.json').read_text())
        with np.load(verify(arm+'/development_traces.npz'), allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        phase_stats = {}
        for prefix, phase in PHASES.items():
            table, by_family, by_start_regime, by_window_contact, by_y, by_geometry, by_family_contact = [], {}, {}, {}, {}, {}, {}
            actual_states, actual_regimes = arrays['actual_states'], arrays['actual_regimes']
            valid = arrays[prefix+'_valid']; steps = arrays[prefix+'_failure_steps']
            reasons = arrays[prefix+'_failure_reasons']
            for row, identity in enumerate(summary[phase]['identities']):
                if prefix == 'head':
                    n = int(arrays['head_indices'][row//5]); start = row%5; length = 1
                elif prefix == 'stop_local':
                    n = int(arrays['stop_indices'][row//5]); start = 10+row%5; length = 1
                elif prefix == 'stop':
                    n = int(arrays['stop_indices'][row]); start = 10; length = 5
                else:
                    n = row; start = 0; length = 10
                first_regime = str(actual_regimes[n, start])
                any_contact = bool(np.any(actual_regimes[n, start:start+length] != 'noncontact'))
                record = {'arm': arm, 'phase': phase, 'row': row, **identity,
                          'actual_core_start_index': start, 'forecast_length': length,
                          'actual_first_transition_regime': first_regime,
                          'actual_window_contact': 'any_contact' if any_contact else 'all_noncontact',
                          'valid': bool(valid[row]), 'failure_step': int(steps[row]),
                          'failure_reason': str(reasons[row])}
                table.append(record)
                groups = ((by_family, identity['family_id']), (by_start_regime, first_regime),
                          (by_window_contact, record['actual_window_contact']), (by_y, identity['y_condition']),
                          (by_geometry, identity['geometry']),
                          (by_family_contact, identity['family_id']+'|'+identity['y_condition']+'|'+first_regime))
                for group, key in groups:
                    group.setdefault(key, {'attempted': 0, 'valid': 0, 'invalid': 0})
                    group[key]['attempted'] += 1
                    group[key]['valid' if valid[row] else 'invalid'] += 1
                if not valid[row]:
                    fail = int(steps[row]); target_index = start+fail
                    state = arrays[prefix+'_states'][row, fail]
                    truth = actual_states[n, target_index]
                    previous = arrays[prefix+'_states'][row, max(fail-1, 0)]
                    contact_at_failure = str(actual_regimes[n, target_index-1]) if fail > 0 else None
                    error = state-truth
                    error[2] = (error[2]+np.pi) % (2*np.pi)-np.pi
                    failures.append({**record, 'failure_origin': 'initial_actual_state' if fail == 0 else 'model_future_state',
                        'actual_failed_core_action_index': target_index-1 if fail > 0 else None,
                        'actual_failed_transition_regime': contact_at_failure,
                        'actual_initial_gap_m': number(gap(actual_states[n, start], config)),
                        'actual_target_gap_m': number(gap(truth, config)),
                        'predicted_previous_gap_m': number(gap(previous, config)),
                        'predicted_failed_gap_m': number(gap(state, config)),
                        'signed_state_error_at_failure': [number(v) for v in error],
                        'invalid_error_note': 'Diagnostic only, excluded from preregistered valid prediction primary values; never used to restore a mean'})
            phase_stats[phase] = {'attempted': len(valid), 'valid': int(valid.sum()), 'invalid': int((~valid).sum()),
                                 'failure_reasons': dict(Counter(str(r) for r in reasons if r)),
                                 'failure_steps': dict(Counter(str(int(s)) for s in steps if s >= 0)),
                                 'by_family': by_family, 'by_Y': by_y, 'by_geometry': by_geometry,
                                 'by_actual_first_transition_regime': by_start_regime,
                                 'by_actual_window_any_contact': by_window_contact,
                                 'by_family_Y_actual_first_regime': by_family_contact, 'rows': table}
        development[arm] = {'phases': phase_stats, 'primary_family_validity': {phase: {
            'attempted': 12, 'valid': sum(all(v is not None for v in row.values()) for row in values.values()),
            'invalid_family_ids': [family for family, row in values.items() if any(v is None for v in row.values())]}
            for phase, values in summary['primary'].items()}}
        legacy = json.loads(verify(arm+'/original_validation.json').read_text())
        geo = json.loads(verify(arm+'/original_validation_geometry.json').read_text())
        regression[arm] = {}
        for k, horizon in legacy['horizons'].items():
            metrics = horizon['metrics_valid_windows_only']; ref = baseline['horizons'][k]['metrics_valid_windows_only']
            compare = {}
            for name in ('object_xy_distance_m', 'pusher_xy_distance_m', 'object_yaw_absolute_rad',
                         'object_linear_velocity_distance_m_s', 'pusher_linear_velocity_distance_m_s',
                         'object_angular_velocity_absolute_rad_s'):
                compare[name] = {'mean': metrics[name]['mean'], 'R_mean': ref[name]['mean'],
                                 'difference_from_R': metrics[name]['mean']-ref[name]['mean']}
            regression[arm][k] = {'attempted_windows': horizon['attempted_windows'],
                'numeric_valid_windows': horizon['valid_windows'], 'numeric_invalid_windows': horizon['invalid_windows'],
                'geometry_invalid_windows_supplement': geo[k]['geometry_invalid_windows'],
                'comparison_with_R': compare,
                'regimes': {r: {'attempted_windows': row['attempted_windows'], 'numeric_invalid_windows': row['invalid_windows'],
                    'geometry_invalid_windows_supplement': geo[k]['regimes'][r]['geometry_invalid_windows'],
                    'object_xy_mean_m': row['metrics_valid_windows_only']['object_xy_distance_m']['mean'],
                    'pusher_xy_mean_m': row['metrics_valid_windows_only']['pusher_xy_distance_m']['mean']}
                    for r, row in horizon['regimes'].items()}}
    initial = sum(r['failure_origin'] == 'initial_actual_state' for r in failures)
    source = Path(__file__)
    output = {'status': 'DESCRIPTIVE_COMPLETE', 'scope': 'Frozen prediction failure accounting, no threshold changes or complete-case replacement effect',
              'source_sha256': bindings, 'analysis_source': str(source.relative_to(ROOT)),
              'analysis_source_sha256': sha(source), 'new_model_forwards': 0,
              'new_physics': 0, 'new_training': 0, 'predicted_penetration_tolerance_m': config['validity']['predicted_penetration_tolerance_m'],
              'all_phase_failures_not_independent_samples': len(failures),
              'failure_reason_counts': dict(Counter(r['failure_reason'] for r in failures)),
              'initial_actual_state_failures': initial, 'model_future_failures': len(failures)-initial,
              'failure_actual_transition_regimes': dict(Counter(r['actual_failed_transition_regime'] for r in failures)),
              'development': development, 'failures': failures, 'original_validation_regression': regression,
              'original_validation_validity_note': 'Frozen legacy metrics gate finite/yaw validity, while geometry is a separate supplement; do not equate this numeric-valid set with the new primary geometry-valid set.',
              'family_independence_note': 'Same family appears in multiple phases/Y/S/arms; counts are diagnostic rows, not independent trials.',
              'effect_support': {factor: row['status'] for factor, row in comparison['directional_support'].items()},
              'diagnostic_checks': {
                  'all_reported_penetration_failures_cross_existing_threshold': all(
                      r['predicted_failed_gap_m'] is not None
                      and r['predicted_failed_gap_m'] < -config['validity']['predicted_penetration_tolerance_m']
                      for r in failures if r['failure_reason'] == 'predicted_deep_penetration'),
                  'all_actual_targets_within_same_geometric_threshold': all(
                      r['actual_target_gap_m'] is not None
                      and r['actual_target_gap_m'] >= -config['validity']['predicted_penetration_tolerance_m']
                      for r in failures)}}
    for path, expected in bindings.items():
        if sha(ROOT/path) != expected: raise ValueError('Source changed during diagnostic analysis')
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(OUT), 'initial_failures': initial, 'future_failures':len(failures)-initial,
                      'failure_reasons': output['failure_reason_counts'], 'actual_contact': output['failure_actual_transition_regimes']}))


if __name__ == '__main__': main()
