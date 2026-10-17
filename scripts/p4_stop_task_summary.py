"""Pure task-prefix supervision for a fixed-action stopping diagnostic.

The caller executes and retains the full open-loop suffix. This helper neither
changes commands nor advances physics. A task ends at its first valid terminal
boundary, while later raw endpoints remain visible as diagnostic observations.
"""
from dataclasses import asdict, replace

import numpy as np

from push_align.contact_geometry import check_contact_geometry
from push_align.task import (
    TaskError, _status, _task, consume_step, fail_task, handoff_metrics,
)


def summarize_task_prefix(results, status, task, dt, profile):
    """Supervise recorded intervals from the supplied source task history.

    ``first_ready_interval`` and timeline ``index`` are one-based within this
    diagnostic suffix. ``qualifies_at_end`` is endpoint pose/speed membership,
    not a task-success verdict or evidence of continuous hold.
    """
    _status(status)
    _task(task, dt)
    if (not isinstance(profile, dict) or
            profile.get('physics_suite', {}).get('check_contact_geometry') is not True):
        raise ValueError('Stopping supervision requires enabled contact geometry checks')
    budget = profile['physics_suite']['actual_contact_penetration_max_m']
    results = tuple(results)
    if not results:
        raise ValueError('Stopping supervision requires recorded physical intervals')
    current = status
    first_ready = None
    timeline = []
    end_qualifies = False
    for index, result in enumerate(results, start=1):
        state = result.observation.state
        if (not isinstance(state, np.ndarray) or state.dtype != np.float64 or
                state.shape != (10,)):
            raise ValueError('Recorded endpoint requires float64[10]')
        if not isinstance(result.transition_valid, (bool, np.bool_)):
            raise ValueError('Recorded physical validity requires a boolean')
        geometry = check_contact_geometry(result.diagnostics['samples'], budget)
        consumed = current.phase in ('running', 'holding')
        if consumed:
            current = consume_step(current, result, task, dt)
            # Keep exactly the existing execution-boundary failure priority:
            # a geometric failure cannot be hidden by readiness or timeout.
            if not geometry['pass'] and current.terminal_reason != 'physics_error':
                current = replace(fail_task(current, 'geometry_invalid'),
                    violations=tuple(dict.fromkeys(
                        (*current.violations, *geometry['reasons']))))
            if current.phase == 'ready' and first_ready is None:
                first_ready = index
        metrics = None
        if np.isfinite(state).all():
            try:
                metrics = handoff_metrics(state, task)
            except TaskError:
                # Overflowing task metrics remain invalid, not a fabricated
                # finite endpoint. Original raw records remain with the caller.
                pass
        end_qualifies = bool(metrics is not None and metrics['qualifies'])
        timeline.append({'index': index, 'phase': current.phase,
            'held_time_s': current.held_time_s, 'metrics': metrics,
            'physical_valid': bool(result.transition_valid),
            'geometry_pass': geometry['pass'], 'task_consumed': consumed})
    return {'first_ready_interval': first_ready, 'status': asdict(current),
            'timeline': timeline, 'qualifies_at_end': end_qualifies}
