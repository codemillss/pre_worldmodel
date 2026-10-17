"""Decompose stored contact prediction errors without inference or physics.

Hybrid states are inputs to a geometry function only. Exact Shapley arithmetic
is a descriptive allocation of gap error; it is not a causal intervention on
the learner, a feasible physical state, or an improved policy.
"""
from pathlib import Path
from itertools import combinations
import argparse
import hashlib
import json
import math

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'runs/p4-data-retrain-20261013-02'
ARMS = ('R', 'D00', 'D10', 'D01', 'D11')
BLOCKS = ('object_xy', 'pusher_xy', 'object_yaw')
INDICES = ((0, 1), (6, 7), (2,))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometry(state, config):
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (10,) or not np.isfinite(state).all():
        raise ValueError('Geometry requires a finite ten-component state')
    theta = state[2]
    co, si = np.cos(theta), np.sin(theta)
    rotation = np.asarray([[co, -si], [si, co]])
    relative = state[6:8] - state[:2]
    local = rotation.T @ relative
    half = np.asarray(config['object']['size_full_xyz_m'][:2]) / 2
    q = np.abs(local) - half
    positive = np.maximum(q, 0)
    outside = float(np.linalg.norm(positive))
    signed_box = outside + min(float(np.max(q)), 0.)
    gap = signed_box - float(config['pusher']['radius_m'])
    # The epsilon only identifies nondifferentiable/ambiguous geometry for a
    # supplementary derivative. It does not alter the original validity gate.
    ambiguous = bool(np.any(np.abs(q) <= 1e-12))
    nondifferentiable = False
    if outside > 0:
        region = 'outside_corner' if np.count_nonzero(positive) == 2 else 'outside_face'
        normal_local = np.sign(local) * positive / outside
        face = 'corner' if region == 'outside_corner' else ('x' if positive[0] > 0 else 'y')
    else:
        tied = abs(float(q[0] - q[1])) <= 1e-12
        axis = int(np.argmax(q))
        ambiguous |= tied or abs(float(local[axis])) <= 1e-12
        nondifferentiable = bool(q[0] == q[1] or local[axis] == 0)
        region, face = 'inside_box', ('x' if axis == 0 else 'y')
        normal_local = np.zeros(2)
        normal_local[axis] = np.sign(local[axis])
    if ambiguous:
        normal_world, yaw_derivative = None, None
    else:
        normal_world = (rotation @ normal_local).tolist()
        yaw_derivative = float(normal_local[0] * local[1] - normal_local[1] * local[0])
    return {'gap_m': gap, 'body_relative_xy_m': local.tolist(), 'world_relative_xy_m': relative.tolist(),
            'box_offset_q_m': q.tolist(), 'region': region, 'active_face': face,
            'derivative_status': ('NOT_DEFINED' if nondifferentiable else 'NEAR_SWITCH_NOT_ASSESSED') if ambiguous else 'DEFINED',
            'normal_world': normal_world, 'gap_yaw_derivative_m_per_rad': yaw_derivative}


def state_error(predicted, actual):
    error = np.asarray(predicted) - np.asarray(actual)
    error = error.copy()
    error[2] = (error[2] + np.pi) % (2*np.pi) - np.pi
    return error


def selected_contact_episode(identities):
    rows = [n for n, i in enumerate(identities)
            if i['family_id']=='p4-data-development-2026101302-f001'
            and i['family_rank']==1 and i['y_condition']=='Y1' and i['stop_condition']=='S0']
    if len(rows) != 1:
        raise ValueError('Exactly one canonical deduplicated contact head required')
    return rows[0]


def decompose(actual, predicted, config):
    actual = np.asarray(actual, np.float64)
    predicted = np.asarray(predicted, np.float64)
    if actual.shape != (10,) or predicted.shape != (10,) or not np.isfinite([actual, predicted]).all():
        raise ValueError('Finite matched states required')
    hybrids = {}
    for mask in range(8):
        state = actual.copy()
        for j, indices in enumerate(INDICES):
            if mask & (1 << j):
                state[list(indices)] = predicted[list(indices)]
        hybrids[mask] = geometry(state, config)
    baseline = hybrids[0]['gap_m']
    coefficients = {}
    for mask in range(1, 8):
        bits = [j for j in range(3) if mask & (1 << j)]
        values = []
        for n in range(len(bits)+1):
            for chosen in combinations(bits, n):
                submask = sum(1 << j for j in chosen)
                values.append((-1)**(len(bits)-n) * hybrids[submask]['gap_m'])
        coefficients[mask] = math.fsum(values)
    contributions = {BLOCKS[j]: math.fsum(coefficients[m] / m.bit_count()
                         for m in coefficients if m & (1 << j)) for j in range(3)}
    exact_error = hybrids[7]['gap_m'] - baseline
    closure = math.fsum(contributions.values()) - exact_error
    error = state_error(predicted, actual)
    derivative = hybrids[0]
    local = {'status': derivative['derivative_status']}
    if local['status'] == 'DEFINED':
        normal = np.asarray(derivative['normal_world'])
        terms = {'object_xy': -float(normal @ error[:2]), 'pusher_xy': float(normal @ error[6:8]),
                 'object_yaw': derivative['gap_yaw_derivative_m_per_rad'] * float(error[2])}
        approximation = math.fsum(terms.values())
        local.update({'terms_m': terms, 'total_m': approximation, 'nonlinear_remainder_m': exact_error - approximation,
                      'relative_normal_error_m': float(normal @ (error[6:8]-error[:2])),
                      'relative_tangent_error_m': float(np.asarray([-normal[1], normal[0]]) @ (error[6:8]-error[:2]))})
    return {'actual_gap_m': baseline, 'predicted_gap_m': hybrids[7]['gap_m'],
            'gap_error_m': exact_error, 'state_error': error.tolist(),
            'relative_xy_error_m': (error[6:8]-error[:2]).tolist(),
            'hybrids': {str(m): {'predicted_blocks': [BLOCKS[j] for j in range(3) if m & (1 << j)], **g}
                        for m, g in hybrids.items()},
            'inclusion_exclusion_m': {'+'.join(BLOCKS[j] for j in range(3) if m & (1 << j)): v
                                      for m, v in coefficients.items()},
            'shapley_gap_error_m': contributions, 'closure_residual_m': closure, 'local_linearization': local,
            'hybrid_scope': 'Geometry-only component substitution; not feasible states, learned counterfactuals, or causal attribution.'}


def run(output):
    evaluation = PARENT / 'prediction_evaluation'
    index = json.loads((evaluation/'artifact_hashes.json').read_text())['sha256']
    bindings = {}
    def source(relative):
        path = evaluation / relative
        if relative not in index or sha(path) != index[relative]:
            raise ValueError('Frozen evaluation source differs: ' + relative)
        bindings[str(path.relative_to(ROOT))] = sha(path)
        return path
    manifest_path = PARENT/'manifest.json'
    config = json.loads(manifest_path.read_text())['physical_config']
    bindings[str(manifest_path.relative_to(ROOT))] = sha(manifest_path)
    rows = []
    for arm in ARMS:
        summary = json.loads(source(arm+'/development.json').read_text())
        with np.load(source(arm+'/development_traces.npz'), allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        identities = summary['core_H10_free']['identities']
        episode = selected_contact_episode(identities)
        head = np.flatnonzero(arrays['head_indices']==episode)
        if len(head) != 1:
            raise ValueError('Exactly one matching head required')
        row = int(head[0])*5+1
        current = arrays['actual_states'][episode,1]
        actual = arrays['actual_states'][episode,2]
        predicted = arrays['head_states'][row,1]
        if not np.array_equal(arrays['head_states'][row,0],current):
            raise ValueError('Local forecast did not begin at actual state')
        decomposition = decompose(actual,predicted,config)
        if abs(decomposition['closure_residual_m']) > 1e-12:
            raise ValueError('Exact gap attribution failed to reconstruct gap error')
        deep = decomposition['predicted_gap_m'] < -config['validity']['predicted_penetration_tolerance_m']
        if not deep or arrays['head_valid'][row] or str(arrays['head_failure_reasons'][row])!='predicted_deep_penetration':
            raise ValueError('Previously observed failure identity differs')
        rows.append({'arm':arm,**identities[episode],'core_current_state':1,'core_next_state':2,
                     'teacher_head_row':row,'action':arrays['actual_actions'][episode,1].tolist(),
                     'actual_start_state':current.tolist(),'actual_end_state':actual.tolist(),
                     'predicted_end_state':predicted.tolist(),'stored_raw_delta':arrays['head_raw_deltas'][row,0].tolist(),
                     'actual_recorded_transition_regime':str(arrays['actual_regimes'][episode,1]),
                     'valid':False,'failure_step':int(arrays['head_failure_steps'][row]),
                     'failure_reason':str(arrays['head_failure_reasons'][row]),
                     'actual_start_geometry':geometry(current,config), **decomposition})
    result={'schema':'p4-stored-contact-geometry-mechanism-v1','status':'DESCRIPTIVE_COMPLETE',
            'analysis_type':'Post hoc selected development failure; no blind causal preregistration',
            'target_rows':5,'target_development_families':1,'S1_duplicate_heads_excluded':True,
            'source_sha256':bindings,'analysis_source_sha256':sha(Path(__file__)),
            'predicted_penetration_tolerance_m':config['validity']['predicted_penetration_tolerance_m'],
            'rows':rows,'new_model_forwards':0,'learning_updates':0,'physics_steps':0,
            'scope':'Endpoint geometry error attribution only. Velocity influences on learned next states remain untested; no primary metric repair.'}
    output=Path(output)
    if output.exists():
        raise FileExistsError('Preserve previous diagnostic output')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'rows':len(rows),'output':str(output.relative_to(ROOT))}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
