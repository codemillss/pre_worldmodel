"""Execution boundary: guard model planning, then consume actual physics.

This module never modifies an engine state. The caller owns recorder.step;
planning sees the present state only. Physical/geometric errors outrank ready.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from .planning import PlannerContext
from .task import consume_step, fail_task

ENGINE_GUARDS = ('mj_step', 'mj_step1', 'mj_step2', 'mj_forward', 'mj_inverse')


@contextmanager
def model_planning_scope(counters):
    """Guard only inference/choice; real execution occurs outside this scope."""
    import mujoco
    if set(counters) != set(ENGINE_GUARDS):
        raise ValueError('All registered future-engine guards are required')
    def forbidden(name):
        def invoke(*args, **kwargs):
            counters[name] += 1
            raise AssertionError(f'Future engine call during model planning: {name}')
        return invoke
    with ExitStack() as guards:
        for name in ENGINE_GUARDS:
            guards.enter_context(patch.object(mujoco, name, side_effect=forbidden(name)))
        yield


def consume_actual_step(status, result, task, action_dt_s, context, *, holding, geometry):
    """Update task and context with actual commands, retaining failure details."""
    if not isinstance(context, PlannerContext) or type(holding) is not bool:
        raise ValueError('Valid planner context and explicit holding flag required')
    if not isinstance(geometry, dict) or type(geometry.get('pass')) is not bool:
        raise ValueError('Explicit contact-geometry check required')
    action = result.applied_action
    if not isinstance(action, np.ndarray) or action.dtype != np.float64 or action.shape != (2,) or not np.isfinite(action).all():
        raise ValueError('Applied action requires finite float64[2]')
    if holding and not np.array_equal(action, np.zeros(2)):
        raise ValueError('Holding must apply the common zero command')
    next_status = consume_step(status, result, task, action_dt_s)
    if not geometry['pass'] and next_status.terminal_reason != 'physics_error':
        next_status = replace(fail_task(next_status, 'geometry_invalid'),
                              violations=tuple(dict.fromkeys((*next_status.violations, *geometry['reasons']))))
    # Candidate generation performs the single shift on the next plan. Keeping
    # the unshifted selected sequence here prevents a double shift.
    next_context = PlannerContext(action.copy(), None if holding else context.optimal_actions,
                                  context.control_step + (1 if holding else 0))
    return next_status, next_context
