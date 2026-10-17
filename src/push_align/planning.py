"""Model-only candidate planning; no environment, simulator futures or repair.

Physical arrays are explicit float64 NumPy batches. Learned inference retains
the registered encoder/normalizer/float32 MLP boundary. Invalid candidates are
retained, never resampled, and cannot win. This proves the planning connection;
candidate quality and closed-loop alignment require separate evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter

import numpy as np

from .representation import RepresentationError, integrate


class PlanningError(ValueError):
    """A planning input or cost contract is invalid."""


def _require(condition, message):
    if not condition:
        raise PlanningError(message)


def _array(value, shape, name, *, finite=True):
    _require(isinstance(value, np.ndarray) and value.dtype == np.float64
             and value.shape == shape, f"{name} requires float64 shape {shape}")
    if finite:
        _require(bool(np.isfinite(value).all()), f"{name} requires finite values")


def _positive(value, name, *, zero=False):
    _require(not isinstance(value, (bool, np.bool_))
             and isinstance(value, (int, float, np.integer, np.floating))
             and math.isfinite(float(value)) and (value >= 0 if zero else value > 0),
             f"{name} requires a finite {'nonnegative' if zero else 'positive'} scalar")
    return float(value)


@dataclass(frozen=True)
class TaskSpec:
    goal: np.ndarray
    position_tolerance_m: float
    yaw_tolerance_rad: float
    linear_speed_tolerance_m_s: float
    angular_speed_tolerance_rad_s: float
    hold_time_s: float
    max_episode_time_s: float

    @classmethod
    def from_config(cls, config):
        task = config['task']
        goal = np.asarray([*task['goal_xy_m'], task['goal_yaw_rad']], dtype=np.float64)
        result = cls(goal, *(task[key] for key in (
            'position_tolerance_m', 'yaw_tolerance_rad', 'linear_speed_tolerance_m_s',
            'angular_speed_tolerance_rad_s', 'hold_time_s', 'max_episode_time_s')))
        result.validate()
        return result

    def validate(self):
        _array(self.goal, (3,), 'goal')
        for key in ('position_tolerance_m', 'yaw_tolerance_rad', 'linear_speed_tolerance_m_s',
                    'angular_speed_tolerance_rad_s', 'hold_time_s', 'max_episode_time_s'):
            _positive(getattr(self, key), key)


@dataclass(frozen=True)
class PredictionBatch:
    states: np.ndarray
    raw_deltas: np.ndarray
    valid: np.ndarray
    reasons: tuple[str, ...]
    failure_steps: np.ndarray


@dataclass(frozen=True)
class CostBatch:
    total: np.ndarray
    components: dict[str, np.ndarray]


@dataclass(frozen=True)
class PlannerContext:
    previous_applied_action: np.ndarray
    optimal_actions: np.ndarray | None = None
    control_step: int = 0

    @classmethod
    def initial(cls):
        return cls(np.zeros(2, dtype=np.float64))


@dataclass(frozen=True)
class PlanResult:
    action: np.ndarray
    selected_index: int | None
    reason: str
    context: PlannerContext
    candidates: object
    predictions: PredictionBatch
    costs: CostBatch
    diagnostics: dict


def _action_batch(actions, config):
    _require(isinstance(actions, np.ndarray) and actions.ndim == 3
             and actions.shape[0] >= 1 and actions.shape[1] >= 1 and actions.shape[2] == 2,
             'actions require [N,H,2], N/H >= 1')
    _array(actions, actions.shape, 'actions')
    physical = _positive(config['pusher']['command_speed_norm_max_m_s'], 'physical_action_limit')
    limit = _positive(config['planner'].get('command_speed_norm_max_m_s', physical), 'planner_action_limit')
    _require(limit <= physical, 'Planner action limit exceeds physical limit')
    _require(bool((np.hypot(actions[..., 0], actions[..., 1]) <= limit + 1e-15).all()),
             'Candidates exceed applied action norm limit')
    _require(config['planner']['use_simulator_for_candidate_prediction'] is False,
             'Simulator futures are forbidden')
    return actions.shape[:2]


def contact_distance(states, config):
    """Signed rectangle distance minus pusher radius, in metres."""
    relative = states[..., 6:8] - states[..., :2]
    yaw = states[..., 2]
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.stack((c*relative[..., 0] + s*relative[..., 1],
                      -s*relative[..., 0] + c*relative[..., 1]), axis=-1)
    half = np.asarray(config['object']['size_full_xyz_m'][:2], np.float64)/2
    q = np.abs(local) - half
    return np.hypot(np.maximum(q[..., 0], 0), np.maximum(q[..., 1], 0)) \
        + np.minimum(np.max(q, axis=-1), 0) - config['pusher']['radius_m']


def _state_reasons(states, config):
    reasons = np.full(len(states), '', dtype='U200')
    finite = np.isfinite(states).all(axis=1)
    reasons[~finite] = 'nonfinite_predicted_state'
    rows = np.flatnonzero(finite)
    if not len(rows):
        return reasons
    state = states[rows]
    half = np.asarray(config['object']['size_full_xyz_m'][:2], np.float64)/2
    corners = half*np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], np.float64)
    c, s = np.cos(state[:, 2]), np.sin(state[:, 2])
    x = state[:, 0, None] + c[:, None]*corners[:, 0] - s[:, None]*corners[:, 1]
    y = state[:, 1, None] + s[:, None]*corners[:, 0] + c[:, None]*corners[:, 1]
    x0, x1 = config['workspace']['x_bounds_m']
    y0, y1 = config['workspace']['y_bounds_m']
    object_bad = ((x < x0) | (x > x1) | (y < y0) | (y > y1)).any(axis=1)
    r = config['pusher']['radius_m']
    pusher_bad = (state[:, 6] < x0+r) | (state[:, 6] > x1-r) \
        | (state[:, 7] < y0+r) | (state[:, 7] > y1-r)
    deep = contact_distance(state, config) < -config['validity']['predicted_penetration_tolerance_m']
    reasons[rows[object_bad]] = 'object_corners_outside_workspace'
    reasons[rows[pusher_bad & ~object_bad]] = 'pusher_extent_outside_workspace'
    reasons[rows[deep & ~object_bad & ~pusher_bad]] = 'predicted_deep_penetration'
    return reasons


def _attempt(predictor, state, action):
    """Retain good rows if the model rejects a batch containing a bad peer."""
    delta = np.full(state.shape, np.nan, dtype=np.float64)
    reasons = np.full(len(state), '', dtype='U200')
    try:
        output = predictor(state.copy(), action.copy())
        _array(output, state.shape, 'physical_delta', finite=False)
        delta[:] = output
    except (ValueError, FloatingPointError):
        for index in range(len(state)):
            try:
                output = predictor(state[index:index+1].copy(), action[index:index+1].copy())
                _array(output, (1, 10), 'physical_delta', finite=False)
                delta[index] = output[0]
            except (ValueError, FloatingPointError) as error:
                reasons[index] = f'inference_rejected:{type(error).__name__}:{error}'[:200]
    nonfinite = ~np.isfinite(delta).all(axis=1)
    reasons[nonfinite & (reasons == '')] = 'nonfinite_physical_delta'
    bad_yaw = ~nonfinite & (np.abs(delta[:, 2]) >= math.pi)
    reasons[bad_yaw & (reasons == '')] = 'raw_delta_yaw_abs_ge_pi'
    return delta, reasons


def rollout(initial_states, actions, predictor, config):
    """Predict all candidate futures, masking failures without clipping states."""
    n, h = _action_batch(actions, config)
    _array(initial_states, (n, 10), 'initial_states')
    _require(callable(predictor), 'predictor must be callable')
    _require(config['validity']['predicted_raw_yaw_increment_abs_limit_rad'] == math.pi,
             'Raw yaw validity requires the registered pi limit')
    states = np.full((n, h+1, 10), np.nan, np.float64)
    deltas = np.full((n, h, 10), np.nan, np.float64)
    states[:, 0] = initial_states
    reasons = _state_reasons(initial_states, config)
    failure_steps = np.where(reasons == '', -1, 0).astype(np.int64)
    for step in range(h):
        rows = np.flatnonzero(reasons == '')
        if not len(rows):
            break
        delta, failed = _attempt(predictor, states[rows, step], actions[rows, step])
        deltas[rows, step] = delta
        okay = np.flatnonzero(failed == '')
        if len(okay):
            # Integrate separately so a finite overflow cannot reject its peers.
            try:
                with np.errstate(over='ignore', invalid='ignore'):
                    states[rows[okay], step+1] = integrate(states[rows[okay], step], delta[okay])
            except RepresentationError:
                for local in okay:
                    try:
                        with np.errstate(over='ignore', invalid='ignore'):
                            states[rows[local], step+1] = integrate(states[rows[local], step], delta[local])
                    except RepresentationError:
                        failed[local] = 'nonfinite_predicted_state'
            available = okay[failed[okay] == '']
            if len(available):
                failed[available] = _state_reasons(states[rows[available], step+1], config)
        bad = failed != ''
        reasons[rows[bad]] = failed[bad]
        failure_steps[rows[bad]] = step+1
    return PredictionBatch(states, deltas, reasons == '', tuple(map(str, reasons)), failure_steps)


def score(predictions, actions, task, previous_applied_action, config):
    """Use post-action states; dt-weight stage costs, then add terminal cost.

    Action costs retain the physical action-limit denominator even when the
    pilot searches a smaller subset. Gap is auxiliary, not a success criterion.
    """
    n, h = _action_batch(actions, config)
    _require(isinstance(task, TaskSpec), 'task requires TaskSpec')
    task.validate()
    _array(previous_applied_action, (2,), 'previous_applied_action')
    _require(isinstance(predictions, PredictionBatch), 'predictions require PredictionBatch')
    _array(predictions.states, (n, h+1, 10), 'predicted_states', finite=False)
    _require(isinstance(predictions.valid, np.ndarray) and predictions.valid.dtype == np.bool_
             and predictions.valid.shape == (n,), 'Invalid candidate mask')
    weights = config['cost']
    names = ('stage_position', 'stage_yaw', 'stage_action', 'stage_action_change', 'stage_gap',
             'terminal_position', 'terminal_yaw', 'terminal_speed', 'terminal_angular_speed')
    for key in names:
        _positive(weights[key], key, zero=True)
    _require(weights['time_sum'] == 'action_dt_weighted_stage_plus_terminal', 'Unknown cost time convention')
    dt = _positive(config['physics']['action_dt_s'], 'action_dt_s')
    limit = _positive(config['pusher']['command_speed_norm_max_m_s'], 'action_limit')
    _require(float(np.hypot(*previous_applied_action)) <= limit + 1e-15,
             'Previous applied action exceeds physical norm limit')
    gap_scale = _positive(weights['gap_scale_m'], 'gap_scale_m')
    components = {key: np.full(n, np.nan, np.float64) for key in names}
    total = np.full(n, np.inf, np.float64)
    rows = np.flatnonzero(predictions.valid)
    if not len(rows):
        return CostBatch(total, components)
    _require(bool(np.isfinite(predictions.states[rows]).all()), 'Valid predictions contain nonfinite states')
    state, action = predictions.states[rows, 1:], actions[rows]
    before = np.concatenate((np.broadcast_to(previous_applied_action, (len(rows), 1, 2)),
                             action[:, :-1]), axis=1)
    with np.errstate(over='ignore', invalid='ignore'):
        position = np.hypot(state[..., 0]-task.goal[0], state[..., 1]-task.goal[1])/task.position_tolerance_m
        angle = state[..., 2]-task.goal[2]
        yaw = (.5*np.arctan2(np.sin(2*angle), np.cos(2*angle)))/task.yaw_tolerance_rad
        stage = {
            'stage_position': position**2,
            'stage_yaw': yaw**2,
            'stage_action': np.sum((action/limit)**2, axis=-1),
            'stage_action_change': np.sum(((action-before)/limit)**2, axis=-1),
            'stage_gap': (np.maximum(contact_distance(state, config), 0)/gap_scale)**2,
        }
        terminal = {
            'terminal_position': position[:, -1]**2,
            'terminal_yaw': yaw[:, -1]**2,
            'terminal_speed': (np.hypot(state[:, -1, 3], state[:, -1, 4])/task.linear_speed_tolerance_m_s)**2,
            'terminal_angular_speed': (state[:, -1, 5]/task.angular_speed_tolerance_rad_s)**2,
        }
        for key, value in stage.items():
            components[key][rows] = weights[key]*dt*np.sum(value, axis=1)
        for key, value in terminal.items():
            components[key][rows] = weights[key]*value
        totals = np.sum(np.stack([components[key][rows] for key in names]), axis=0)
    total[rows[np.isfinite(totals)]] = totals[np.isfinite(totals)]
    return CostBatch(total, components)


def choose(costs, predictions, actions):
    """Exact cost ties choose the lowest candidate index; all-invalid fails."""
    _require(isinstance(costs, CostBatch) and isinstance(predictions, PredictionBatch), 'Invalid selection objects')
    _require(isinstance(actions, np.ndarray) and actions.ndim == 3
             and actions.shape[0] >= 1 and actions.shape[1] >= 1 and actions.shape[2] == 2,
             'Invalid selection actions')
    _array(actions, actions.shape, 'actions')
    n = len(actions)
    _array(costs.total, (n,), 'total_cost', finite=False)
    _require(not bool(np.isnan(costs.total).any()) and bool((costs.total >= 0).all()),
             'Costs must be nonnegative finite values or positive infinity')
    _require(predictions.valid.shape == (n,) and predictions.valid.dtype == np.bool_, 'Invalid selection mask')
    valid = predictions.valid & np.isfinite(costs.total)
    if not bool(valid.any()):
        return None, np.zeros(2, np.float64), 'planner_invalid'
    eligible = np.flatnonzero(valid)
    index = int(eligible[np.argmin(costs.total[eligible])])
    return index, actions[index, 0].copy(), 'selected'


class LearnedPredictor:
    """Instrumented inference adapter using the original learned transforms."""
    def __init__(self, model, normalizer):
        self.model, self.normalizer = model, normalizer
        self.calls = self.rows = 0

    def __call__(self, state, action):
        import torch
        from .dynamics import predict_delta
        self.calls += 1
        self.rows += len(state)
        mode = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                output = predict_delta(self.model, self.normalizer,
                                       torch.from_numpy(state.copy()), torch.from_numpy(action.copy()))
            return output.detach().cpu().numpy()
        finally:
            self.model.train(mode)


def plan(observation, task, context, rng, predictor, config, *, geometry_action=None, tangent_direction=None):
    """Return first command and warm context without executing any physics."""
    from .candidates import generate_candidates
    _array(observation, (10,), 'observation')
    _require(isinstance(context, PlannerContext) and type(context.control_step) is int
             and context.control_step >= 0, 'Invalid planner context')
    _array(context.previous_applied_action, (2,), 'previous_applied_action')
    started = perf_counter()
    candidates = generate_candidates(config, rng, previous_actions=context.optimal_actions,
                                     geometry_action=geometry_action, tangent_direction=tangent_direction)
    initial = np.broadcast_to(observation, (len(candidates.actions), 10)).copy()
    predictions = rollout(initial, candidates.actions, predictor, config)
    costs = score(predictions, candidates.actions, task, context.previous_applied_action, config)
    index, action, reason = choose(costs, predictions, candidates.actions)
    warm = None if index is None else candidates.actions[index].copy()
    # This context represents a proposed command. The executor must replace it
    # with the actually applied command (including holding zero) before replanning.
    new_context = PlannerContext(action.copy(), warm, context.control_step+1)
    diagnostics = {'candidate_count': len(candidates.actions), 'horizon': candidates.actions.shape[1],
                   'valid_predictions': int(predictions.valid.sum()),
                   'eligible_costs': int((predictions.valid & np.isfinite(costs.total)).sum()),
                   'selected_index': index, 'reason': reason,
                   'wall_time_s': perf_counter()-started, 'candidate_generation': candidates.diagnostics,
                   'stage_state_convention': 'post-action predicted states 1..H',
                   'action_cost_scale': 'physical action limit; planner search subset does not change costs'}
    return PlanResult(action, index, reason, new_context, candidates, predictions, costs, diagnostics)
