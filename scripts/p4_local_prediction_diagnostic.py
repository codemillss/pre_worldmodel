"""Offline recorded-state conditioning diagnostic; never an online planner.

Uses only existing source99 candidate11/77 labels. No physics or learning.
Resets the input to a recorded actual state for each one-step forecast, then
separates that local residual from the difference caused by recursive inputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np
import torch

from push_align.config import ROOT
from push_align.closed_loop import ENGINE_GUARDS, model_planning_scope
from push_align.data import digest
from push_align.fork_checkpoint import load_fork_checkpoint
from push_align.normalization import load_normalizer
from push_align.planning import LearnedPredictor
from push_align.representation import STATE_FIELDS, encode, integrate, state_delta
from p3_train_common_mlp import load_common_checkpoint
from p4_train_forks import tree_hash

MODELS = ('MLP-11', 'S1-11', 'SK-11')
CANDIDATES = (11, 77)
PARENT = ROOT / 'runs/p4-cost-20261013-02'

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def local_error_decomposition(actual_states, free_states, free_delta, conditioned_delta):
    """Return signed physical errors, retaining their additive vector meaning.

    e_next = e_previous + local_error + recursive_input_effect. This is an
    algebraic identity on XY/velocities, not a causal percentage partition.
    Yaw is checked with the declared principal wrap. Norms are not additive.
    """
    for name, value, shape in (
        ('actual_states', actual_states, (11, 10)),
        ('free_states', free_states, (11, 10)),
        ('free_delta', free_delta, (10, 10)),
        ('conditioned_delta', conditioned_delta, (10, 10))):
        if not isinstance(value, np.ndarray) or value.dtype != np.float64 or value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid {name}')
    actual_delta = state_delta(actual_states[:-1], actual_states[1:])
    local = conditioned_delta - actual_delta
    input_effect = free_delta - conditioned_delta
    previous = state_delta(actual_states[:-1], free_states[:-1])
    next_error = state_delta(actual_states[1:], free_states[1:])
    reconstructed = previous + local + input_effect
    reconstructed[:, 2] = (reconstructed[:, 2] + np.pi) % (2*np.pi) - np.pi
    residual = reconstructed - next_error
    residual[:, 2] = (residual[:, 2] + np.pi) % (2*np.pi) - np.pi
    if np.max(np.abs(residual)) > 1e-12:
        raise ValueError('Error decomposition does not close')
    return dict(actual_delta=actual_delta, local_error=local,
                recursive_input_effect=input_effect, previous_error=previous,
                free_next_error=next_error, closure_residual=residual)

def norms(a):
    return {
        'object_xy_mm': (np.linalg.norm(a[..., :2], axis=-1)*1000).tolist(),
        'object_yaw_deg': np.rad2deg(np.abs(a[..., 2])).tolist(),
        'object_velocity_mm_s': (np.linalg.norm(a[..., 3:5], axis=-1)*1000).tolist(),
        'object_wz_rad_s': np.abs(a[..., 5]).tolist(),
        'pusher_xy_mm': (np.linalg.norm(a[..., 6:8], axis=-1)*1000).tolist(),
        'pusher_velocity_mm_s': (np.linalg.norm(a[..., 8:10], axis=-1)*1000).tolist()
    }

def phase_summary(a):
    return {phase: {k: {'mean': float(np.mean(v[lo:hi])),
                        'maximum': float(np.max(v[lo:hi]))}
                    for k, v in norms(a).items()}
            for phase, lo, hi in [('head', 0, 5), ('zero_suffix', 5, 10)]}

def run(registry_path, expected_sha, output):
    if digest(registry_path) != expected_sha:
        raise ValueError('Diagnostic registration differs')
    registry = json.loads(registry_path.read_text())
    for path, expected in registry['bindings_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('Source binding differs: '+path)
    if output.exists():
        raise ValueError('Never overwrite a diagnostic run')
    if shutil.disk_usage(ROOT).free < 2*1024**3 + registry['budget']['maximum_bytes']:
        raise ValueError('Free reserve does not fit')
    output.mkdir(parents=True)
    save(output/'registration.json', registry)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rng_before = torch.get_rng_state().clone()
    parent_manifest = json.loads((PARENT/'manifest.json').read_text())
    physical = parent_manifest['physical']
    nm = json.loads((ROOT/registry['normalizer_dir']/'normalizer.json').read_text())['normalizer']
    normalizer = load_normalizer(ROOT/registry['normalizer_dir'], nm['schema'],
                                 nm['dataset_identity'], nm['normalizer_hash'])
    common_manifest = json.loads((ROOT/'runs/p3-04-20261009-01/manifest.json').read_text())
    fork_manifest_path = ROOT/'runs/p4-02-20261012-01/manifest.json'
    fork_manifest = json.loads(fork_manifest_path.read_text())
    budget = json.loads((ROOT/'configs/p4_fork_budget.v2.json').read_text())
    bundles = {}
    for name in MODELS:
        identity = parent_manifest['model_identities'][name]
        if name == 'MLP-11':
            bundle = load_common_checkpoint(ROOT/identity['path'], normalizer, physical,
                common_manifest['provenance'], common_manifest['settings'], 11)
        else:
            branch = 'S1' if name == 'S1-11' else 'SK-lambda0.1'
            bundle = load_fork_checkpoint(ROOT/identity['path'], normalizer, physical,
                fork_manifest['provenance'], budget, seed=11, branch=branch,
                expected_checkpoint_hash=identity['model_hash'],
                expected_run_manifest_sha256=digest(fork_manifest_path))
        if bundle.metadata['checkpoint_hash'] != identity['model_hash']:
            raise ValueError('Parent checkpoint differs')
        bundles[name] = bundle
    before = {name: tree_hash(b.model.state_dict()) for name, b in bundles.items()}
    initial = json.loads((PARENT/'input.json').read_text())
    initial_state = np.asarray(initial['state'], np.float64)
    actual = {}
    actions = {}
    for candidate in CANDIDATES:
        with np.load(PARENT/f'actual/candidate-{candidate:03d}/repeat-0/raw_execution.npz', allow_pickle=False) as z:
            if not (len(z['state']) == 10 and np.all(np.abs(z['actual_dt']-.1) < 1e-9)):
                raise ValueError('Not ten complete fixed-dt recorded intervals')
            actual[candidate] = np.vstack((initial_state, z['state']))
            actions[candidate] = z['action_applied'].copy()
            if not np.array_equal(z['action_requested'], z['action_applied']):
                raise ValueError('Unexpected command intervention')
    guarded = dict.fromkeys(ENGINE_GUARDS, 0)
    arrays = {}
    summaries = []
    bridge = []
    counts = {}
    for name, bundle in bundles.items():
        predictor = LearnedPredictor(bundle.model, normalizer)
        with np.load(PARENT/f'initial_models/{name}/predictions.npz', allow_pickle=False) as z:
            cached = {k: z[k].copy() for k in ('states', 'raw_deltas', 'full_actions', 'valid')}
        if not cached['valid'].all():
            raise ValueError('This registration expects all128 predictions valid')
        for candidate in CANDIDATES:
            if not (np.array_equal(cached['states'][candidate, 0], initial_state)
                    and np.array_equal(cached['full_actions'][candidate], actions[candidate])):
                raise ValueError('Actual/forecast initial state or commands differ')
        # Same N128 leading shape as the archived planner: verify every step's
        # checkpoint/normalizer/float boundary without creating a new rollout.
        max_difference = 0.0
        exact = True
        with model_planning_scope(guarded):
            for step in range(10):
                recomputed = predictor(cached['states'][:, step], cached['full_actions'][:, step])
                max_difference = max(max_difference, float(np.max(np.abs(recomputed-cached['raw_deltas'][:, step]))))
                exact = exact and np.array_equal(recomputed, cached['raw_deltas'][:, step])
            recorded_inputs = np.concatenate([actual[c][:-1] for c in CANDIDATES])
            recorded_actions = np.concatenate([actions[c] for c in CANDIDATES])
            conditioned = predictor(recorded_inputs, recorded_actions)
        if max_difference > registry['bridge_maximum_absolute_difference']:
            raise ValueError('Archived inference bridge no longer matches')
        bridge.append({'model': name, 'rows': 1280, 'exact': exact,
                       'maximum_absolute_difference': max_difference})
        for offset, candidate in enumerate(CANDIDATES):
            states = actual[candidate]
            delta = conditioned[offset*10:(offset+1)*10]
            one_next = integrate(states[:-1], delta)
            d = local_error_decomposition(states, cached['states'][candidate],
                                            cached['raw_deltas'][candidate], delta)
            prefix = f'{name.replace("-", "_")}_candidate_{candidate:03d}_'
            items = {'actual_states': states, 'actions': actions[candidate],
                     'free_states': cached['states'][candidate],
                     'free_delta': cached['raw_deltas'][candidate],
                     'conditioned_delta': delta, 'conditioned_next': one_next,
                     **d}
            arrays.update({prefix+k: v for k, v in items.items()})
            summaries.append({
                'model': name, 'candidate': candidate,
                'first_input_actual_equals_free': bool(np.array_equal(states[0], cached['states'][candidate, 0])),
                'first_step_same_input_batch_roundoff_maximum': float(np.max(np.abs(d['recursive_input_effect'][0]))),
                'batch_roundoff_note': 'Cached forward uses128 rows; actual-conditioned forward uses20. A tiny first-row effect with identical inputs is numerical batch roundoff, not recursive-state influence.',
                'actual_delta_norms': norms(d['actual_delta']),
                'conditioned_delta_norms': norms(delta),
                'conditioned_one_step_error_norms': norms(d['local_error']),
                'free_endpoint_error_norms': norms(d['free_next_error']),
                'recursive_input_effect_norms': norms(d['recursive_input_effect']),
                'local_error_phases': phase_summary(d['local_error']),
                'recursive_input_effect_phases': phase_summary(d['recursive_input_effect']),
                'maximum_decomposition_closure_error': float(np.max(np.abs(d['closure_residual'])))
            })
        counts[name] = {'calls': predictor.calls, 'rows': predictor.rows,
                        'cached_bridge_rows': 1280, 'recorded_state_rows': 20}
    np.savez_compressed(output/'transitions.npz', **arrays)
    checks = {
        'source_and_command_alignment': True,
        'archived_forward_bridge': True,
        'all60_conditioned_transition_rows': sum(v['recorded_state_rows'] for v in counts.values()) == 60,
        'all3840_cached_bridge_rows': sum(v['cached_bridge_rows'] for v in counts.values()) == 3840,
        'zero_future_engine_calls': all(v == 0 for v in guarded.values()),
        'model_weights_unchanged': all(before[n] == tree_hash(b.model.state_dict()) for n, b in bundles.items()),
        'torch_rng_unchanged': torch.equal(rng_before, torch.get_rng_state()),
        'registered_bindings_unchanged': all(digest(ROOT/p) == h for p, h in registry['bindings_sha256'].items()),
        'maximum_forward_rows': sum(v['rows'] for v in counts.values()) <= registry['budget']['maximum_forward_rows'],
        'resource_budget': sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) <= registry['budget']['maximum_bytes']
    }
    summary = {
        'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
        'created_at': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        'registry_sha256': expected_sha, 'models': list(MODELS),
        'candidates': list(CANDIDATES), 'conditioned_rows': 60,
        'bridge': bridge, 'forward_counts': counts, 'future_engine_calls': guarded,
        'learning_updates': 0, 'physical_intervals': 0, 'final_test_used': False,
        'personal_understanding': 'NOT_ASSESSED', 'results': summaries,
        'scope': 'Post-hoc one reused source99 state and two existing candidate trajectories. Recorded future labels are deliberately used as one-step inputs offline; never used for online selection. Decomposition is algebra, not causal attribution or a deployable trajectory.'
    }
    save(output/'summary.json', summary)
    save(output/'artifact_hashes.json', {'sha256': {str(p.relative_to(output)): digest(p)
        for p in sorted(output.rglob('*')) if p.is_file()}})
    print(json.dumps({'status': summary['status'], 'rows': 3900,
                      'new_physics': 0, 'learning': 0, 'bridge': bridge}))
    if not all(checks.values()):
        raise ValueError('Offline diagnostic checks failed')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--registry-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.registry, args.registry_sha, args.output)
