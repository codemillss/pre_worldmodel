"""Pure H10 endpoint-surrogate ranking for the bounded handoff diagnostic.

This evaluates learned state predictions, never physics futures. Endpoint
qualification and endpoint-adjacent duration do not prove continuous physical
holding. Invalid full-horizon predictions cannot win, including predictions
that appear to reach a surrogate event before a later failure.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .planning import CostBatch, PlanningError, PredictionBatch, TaskSpec

HORIZON = 10
ACTION_DT_S = .1
TIME_TOLERANCE_S = 1e-9


class HandoffCostError(ValueError):
    """The endpoint-surrogate input or selection contract is malformed."""


def _require(condition, message):
    if not condition:
        raise HandoffCostError(message)


def _array(value, shape, name, *, finite=True):
    _require(isinstance(value, np.ndarray) and value.dtype == np.float64
             and value.shape == shape, f'{name} requires float64 shape {shape}')
    if finite:
        _require(bool(np.isfinite(value).all()), f'{name} requires finite values')


def _scalar(value, name, *, positive=False):
    _require(not isinstance(value, (bool, np.bool_))
             and isinstance(value, (int, float, np.integer, np.floating))
             and math.isfinite(float(value))
             and (float(value) > 0 if positive else float(value) >= 0),
             f'{name} requires a finite {"positive" if positive else "nonnegative"} scalar')
    return float(value)


@dataclass(frozen=True)
class HandoffScores:
    """Auditable surrogate quantities; ``ready`` is never actual task success.

    The four rank columns are non-ready, terminal maximum positive excess,
    dt-weighted accumulated maximum positive excess, and original whole-H10 J.
    New excess terms end at the first surrogate-ready boundary, if present.
    The original J remains a whole-horizon secondary tie-breaker.
    The held-time array retains the full predicted endpoint timeline, including
    later endpoints after the first event; it is not an absorbing task status.
    """
    ready: np.ndarray
    first_ready_step: np.ndarray
    terminal_max_excess: np.ndarray
    integrated_max_excess: np.ndarray
    eligible: np.ndarray
    ratios: np.ndarray
    held_time_s: np.ndarray
    rank_keys: np.ndarray
    reasons: tuple[str, ...]


def score_handoff(predictions, task, legacy_costs, action_dt_s, *,
                  initial_held_time_s=0., initial_previous_qualifies=False):
    """Rank a frozen H5-plus-five-zero H10 prediction without tuned weights.

    Ratios order: object position, pi-symmetric yaw, linear speed, angular
    speed. Both adjacent endpoint samples must qualify to add their interval.
    Carry only the supplied measured source hold; its qualification flag must
    agree with every finite, measurable predicted initial state. The supplied
    source status must still be active (held time below the required duration).
    """
    _require(isinstance(task, TaskSpec), 'task requires TaskSpec')
    try:
        task.validate()
    except (PlanningError, TypeError, AttributeError, OverflowError) as error:
        raise HandoffCostError(str(error)) from error
    dt = _scalar(action_dt_s, 'action_dt_s', positive=True)
    _require(abs(dt-ACTION_DT_S) <= 1e-15, 'This registered H10 surrogate requires action_dt_s=0.1')
    held_initial = _scalar(initial_held_time_s, 'initial_held_time_s')
    _require(isinstance(initial_previous_qualifies, (bool, np.bool_)),
             'initial_previous_qualifies requires a boolean')
    previous_initial = bool(initial_previous_qualifies)
    _require(previous_initial or held_initial == 0., 'Broken initial qualification cannot retain held time')
    _require(held_initial+TIME_TOLERANCE_S < task.hold_time_s,
             'Initial source hold must describe an active, not already ready, task')
    _require(isinstance(predictions, PredictionBatch), 'predictions requires PredictionBatch')
    states = predictions.states
    _require(isinstance(states, np.ndarray) and states.ndim == 3 and states.shape[0] >= 1,
             'predicted_states requires a nonempty candidate batch')
    n = len(states)
    _array(states, (n, HORIZON+1, 10), 'predicted_states', finite=False)
    _array(predictions.raw_deltas, (n, HORIZON, 10), 'raw_deltas', finite=False)
    _require(isinstance(predictions.valid, np.ndarray) and predictions.valid.dtype == np.bool_
             and predictions.valid.shape == (n,), 'prediction mask requires bool[N]')
    _require(isinstance(predictions.reasons, tuple) and len(predictions.reasons) == n
             and all(isinstance(reason, str) for reason in predictions.reasons),
             'prediction reasons requires N strings')
    _require(isinstance(predictions.failure_steps, np.ndarray)
             and predictions.failure_steps.dtype == np.int64
             and predictions.failure_steps.shape == (n,), 'failure_steps requires int64[N]')
    _require(bool(np.all(np.where(predictions.valid,
             predictions.failure_steps == -1,
             (predictions.failure_steps >= 0) & (predictions.failure_steps <= HORIZON)))),
             'Prediction validity disagrees with failure step')
    _require(all((reason == '') == bool(valid)
                 for reason, valid in zip(predictions.reasons, predictions.valid)),
             'Prediction validity disagrees with reason')
    _require(isinstance(legacy_costs, CostBatch), 'legacy_costs requires CostBatch')
    _array(legacy_costs.total, (n,), 'legacy_total', finite=False)
    _require(bool((~np.isnan(legacy_costs.total) & (legacy_costs.total >= 0)).all()),
             'Legacy J requires nonnegative finite values or positive infinity')
    _require(isinstance(legacy_costs.components, dict), 'Legacy components requires a dictionary')
    for name, component in legacy_costs.components.items():
        _require(isinstance(name, str), 'Legacy component names require strings')
        _array(component, (n,), f'legacy component {name}', finite=False)

    ratios = np.full((n, HORIZON+1, 4), np.nan, np.float64)
    held = np.full((n, HORIZON+1), np.nan, np.float64)
    ready = np.zeros(n, np.bool_)
    first = np.full(n, -1, np.int64)
    terminal = np.full(n, np.inf, np.float64)
    integrated = np.full(n, np.inf, np.float64)
    keys = np.full((n, 4), np.inf, np.float64)
    eligible = np.zeros(n, np.bool_)
    reasons = []
    for index in range(n):
        state = states[index]
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            angle = state[:, 2]-task.goal[2]
            ratios[index, :, 0] = np.hypot(state[:, 0]-task.goal[0], state[:, 1]-task.goal[1])/task.position_tolerance_m
            ratios[index, :, 1] = np.abs(.5*np.arctan2(np.sin(2*angle), np.cos(2*angle)))/task.yaw_tolerance_rad
            ratios[index, :, 2] = np.hypot(state[:, 3], state[:, 4])/task.linear_speed_tolerance_m_s
            ratios[index, :, 3] = np.abs(state[:, 5])/task.angular_speed_tolerance_rad_s
        if bool(np.isfinite(ratios[index, 0]).all()):
            _require(bool((ratios[index, 0] <= 1).all()) == previous_initial,
                     'Initial endpoint qualification disagrees with observed source flag')
        if not bool(predictions.valid[index]):
            reasons.append('prediction_invalid:'+predictions.reasons[index])
            continue
        if not bool(np.isfinite(state).all() and np.isfinite(predictions.raw_deltas[index]).all()):
            reasons.append('prediction_nonfinite')
            continue
        if not bool(np.isfinite(legacy_costs.total[index])):
            reasons.append('legacy_cost_nonfinite')
            continue
        if not bool(np.isfinite(ratios[index]).all()):
            reasons.append('handoff_metrics_overflow')
            continue
        qualifies = (ratios[index] <= 1).all(axis=-1)
        held[index, 0] = held_initial
        previous = previous_initial
        first_ready = -1
        for step in range(1, HORIZON+1):
            held[index, step] = held[index, step-1]+dt if qualifies[step] and previous else 0.
            previous = bool(qualifies[step])
            if first_ready == -1 and held[index, step]+TIME_TOLERANCE_S >= task.hold_time_s:
                first_ready = step
        end = first_ready if first_ready != -1 else HORIZON
        with np.errstate(over='ignore', invalid='ignore'):
            excess = np.maximum(ratios[index]-1., 0.).max(axis=-1)
            last_excess = float(excess[end])
            accumulated = float(dt*np.sum(excess[1:end+1]))
        if not math.isfinite(accumulated) or not bool(np.isfinite(held[index]).all()):
            reasons.append('handoff_score_overflow')
            continue
        ready[index] = first_ready != -1
        first[index] = first_ready
        terminal[index] = last_excess
        integrated[index] = accumulated
        keys[index] = (float(not ready[index]), last_excess, accumulated, legacy_costs.total[index])
        eligible[index] = True
        reasons.append('')
    return HandoffScores(ready, first, terminal, integrated, eligible,
                         ratios, held, keys, tuple(reasons))


def choose_handoff(scores, actions):
    """Choose lexicographically; exact ties retain the lowest original index."""
    _require(isinstance(scores, HandoffScores), 'scores requires HandoffScores')
    _require(isinstance(actions, np.ndarray) and actions.ndim == 3 and actions.shape[0] >= 1,
             'actions requires a nonempty candidate batch')
    n = len(actions)
    _array(actions, (n, HORIZON, 2), 'actions')
    for name in ('ready', 'eligible'):
        value = getattr(scores, name)
        _require(isinstance(value, np.ndarray) and value.dtype == np.bool_
                 and value.shape == (n,), f'{name} requires bool[N]')
    _require(isinstance(scores.first_ready_step, np.ndarray)
             and scores.first_ready_step.dtype == np.int64
             and scores.first_ready_step.shape == (n,), 'first_ready_step requires int64[N]')
    _array(scores.terminal_max_excess, (n,), 'terminal_max_excess', finite=False)
    _array(scores.integrated_max_excess, (n,), 'integrated_max_excess', finite=False)
    _array(scores.ratios, (n, HORIZON+1, 4), 'ratios', finite=False)
    _array(scores.held_time_s, (n, HORIZON+1), 'held_time_s', finite=False)
    _array(scores.rank_keys, (n, 4), 'rank_keys', finite=False)
    _require(isinstance(scores.reasons, tuple) and len(scores.reasons) == n
             and all(isinstance(reason, str) for reason in scores.reasons),
             'Score reasons requires N strings')
    _require(bool((~np.isnan(scores.rank_keys) & (scores.rank_keys >= 0)).all()),
             'Rank keys require nonnegative values or positive infinity')
    valid = scores.eligible
    _require(bool(np.isfinite(scores.rank_keys[valid]).all()), 'Eligible rank keys must be finite')
    _require(bool(np.isfinite(scores.ratios[valid]).all()
             and (scores.ratios[valid] >= 0).all()
             and np.isfinite(scores.held_time_s[valid]).all()
             and (scores.held_time_s[valid] >= 0).all()),
             'Eligible ratio and held-time diagnostics must be finite and nonnegative')
    _require(bool(np.isinf(scores.rank_keys[~valid]).all()), 'Ineligible rank keys must remain positive infinity')
    _require(bool((~scores.ready[~valid]).all()
             and (scores.first_ready_step[~valid] == -1).all()),
             'Ineligible predictions cannot claim a surrogate-ready event')
    _require(bool((scores.rank_keys[valid, 0] == (~scores.ready[valid]).astype(np.float64)).all()),
             'Rank ready key disagrees with surrogate event')
    _require(bool(np.array_equal(scores.rank_keys[valid, 1], scores.terminal_max_excess[valid])
             and np.array_equal(scores.rank_keys[valid, 2], scores.integrated_max_excess[valid])),
             'Rank excess keys disagree with components')
    _require(bool(np.all(np.where(scores.ready[valid],
             (scores.first_ready_step[valid] >= 1) & (scores.first_ready_step[valid] <= HORIZON),
             scores.first_ready_step[valid] == -1))), 'Ready event step is inconsistent')
    _require(all((reason == '') == bool(okay) for reason, okay in zip(scores.reasons, valid)),
             'Eligibility disagrees with score reason')
    choices = np.flatnonzero(valid)
    if not len(choices):
        return None, np.zeros(2, np.float64), 'planner_invalid'
    index = int(min(choices, key=lambda item: (*scores.rank_keys[item], int(item))))
    return index, actions[index, 0].copy(), 'selected'
