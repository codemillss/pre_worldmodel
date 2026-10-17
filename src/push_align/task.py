"""Pure task supervision over observed physical samples.

The supervisor never advances a simulator or modifies a physical state. Pose
membership requests holding; readiness additionally requires low measured
velocity continuously over sampled timestamps. Success/timeout are assessed
only at a completed action boundary, after physical validity failures.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .planning import TaskSpec
from .representation import STATE_FIELDS

STATE_SCHEMA = "push-align-state-v1"


class TaskError(ValueError):
    """A task input has an invalid shape, schema or temporal structure."""


def _require(condition, message):
    if not condition:
        raise TaskError(message)


def _scalar(value, name, *, positive=False):
    _require(not isinstance(value, (bool, np.bool_))
             and isinstance(value, (int, float, np.integer, np.floating))
             and math.isfinite(float(value)), f"{name} requires a finite scalar")
    result = float(value)
    _require(result > 0 if positive else result >= 0,
             f"{name} requires a {'positive' if positive else 'nonnegative'} scalar")
    return result


def _boolean(value, name):
    _require(isinstance(value, (bool, np.bool_)), f"{name} requires a boolean")
    return bool(value)


def _array(value, shape, name, *, finite=True):
    _require(isinstance(value, np.ndarray) and value.dtype == np.float64
             and value.shape == shape, f"{name} requires float64 shape {shape}")
    if finite:
        _require(bool(np.isfinite(value).all()), f"{name} requires finite values")
    return value


def _task(task, action_dt_s=None, *, tolerance=1e-9):
    _require(isinstance(task, TaskSpec), "task requires TaskSpec")
    try:
        task.validate()
    except (ValueError, TypeError, AttributeError, OverflowError) as error:
        raise TaskError(str(error)) from error
    if action_dt_s is not None:
        dt = _scalar(action_dt_s, "action_dt_s", positive=True)
        ratio = task.max_episode_time_s / dt
        _require(math.isfinite(ratio), "Episode/action time ratio overflowed")
        count = round(ratio)
        _require(count >= 1 and abs(task.max_episode_time_s-count*dt) <= tolerance,
                 "max_episode_time_s must be a positive integer multiple of action_dt_s")
        return dt


@dataclass(frozen=True)
class TaskStatus:
    phase: str
    initial_sim_time: float
    last_sim_time: float
    held_time_s: float
    previous_qualifies: bool
    terminal_reason: str | None
    violations: tuple[str, ...]

    @property
    def elapsed_s(self):
        return self.last_sim_time - self.initial_sim_time


def _status(status):
    _require(isinstance(status, TaskStatus), "status requires TaskStatus")
    _require(isinstance(status.phase, str)
             and status.phase in {"running", "holding", "ready", "failed"}, "Unknown task phase")
    initial = _scalar(status.initial_sim_time, "initial_sim_time")
    last = _scalar(status.last_sim_time, "last_sim_time")
    held = _scalar(status.held_time_s, "held_time_s")
    _require(last >= initial and held <= last-initial+1e-9, "Invalid elapsed/held time")
    previous = _boolean(status.previous_qualifies, "previous_qualifies")
    _require(previous or held == 0, "A broken hold cannot retain held time")
    _require(isinstance(status.violations, tuple)
             and all(isinstance(item, str) and item for item in status.violations),
             "violations requires a tuple of nonempty strings")
    if status.phase in {"running", "holding"}:
        _require(status.terminal_reason is None, "Active task cannot have terminal_reason")
    else:
        _require(isinstance(status.terminal_reason, str) and status.terminal_reason,
                 "Terminal task requires terminal_reason")


def handoff_metrics(state, task):
    """Return observed task quantities in world SI units (pi-symmetric yaw)."""
    _task(task)
    _array(state, (10,), "state")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            position = float(np.hypot(*(state[:2]-task.goal[:2])))
            angle = float(state[2]-task.goal[2])
            yaw = abs(.5*math.atan2(math.sin(2*angle), math.cos(2*angle)))
            speed = float(np.hypot(state[3], state[4]))
            angular = abs(float(state[5]))
    except (ValueError, OverflowError) as error:
        raise TaskError("Task metrics overflowed") from error
    _require(all(math.isfinite(item) for item in (position, yaw, speed, angular)),
             "Task metrics overflowed")
    pose = position <= task.position_tolerance_m and yaw <= task.yaw_tolerance_rad
    qualifies = pose and speed <= task.linear_speed_tolerance_m_s \
        and angular <= task.angular_speed_tolerance_rad_s
    return {"e_pos_m": position, "e_yaw_rad": yaw,
            "linear_speed_m_s": speed, "angular_speed_rad_s": angular,
            "pose_within": bool(pose), "qualifies": bool(qualifies)}


def pose_in_goal(state, task):
    return handoff_metrics(state, task)["pose_within"]


def _observation(observation):
    try:
        state = _array(observation.state, (10,), "observation.state", finite=False)
        time = _scalar(observation.sim_time, "observation.sim_time")
        _require(observation.schema_version == STATE_SCHEMA, "Unsupported observation schema")
        valid = _boolean(observation.observation_valid, "observation_valid")
    except AttributeError as error:
        raise TaskError(f"Malformed observation: {error}") from error
    return state, time, valid


def start_task(observation, task, action_dt_s):
    """Start after reset settling, without counting any settling/initial hold."""
    _task(task, action_dt_s)
    state, time, valid = _observation(observation)
    if not np.isfinite(state).all() or not valid:
        reason = "nonfinite_state" if not np.isfinite(state).all() else "observation_invalid"
        return TaskStatus("failed", time, time, 0., False, "physics_error", (reason,))
    try:
        metrics = handoff_metrics(state, task)
    except TaskError:
        return TaskStatus("failed", time, time, 0., False, "physics_error", ("task_metrics_nonfinite",))
    return TaskStatus("holding" if metrics["pose_within"] else "running",
                      time, time, 0., metrics["qualifies"], None, ())


def _raw_state(value, name):
    try:
        array = np.asarray(value)
        _require(array.shape == (10,) and array.dtype.kind in "fiu",
                 f"{name} requires ten numeric columns")
        return array.astype(np.float64, copy=False)
    except (TypeError, ValueError) as error:
        if isinstance(error, TaskError):
            raise
        raise TaskError(f"Malformed {name}: {error}") from error


def _step(step, status, dt, tolerance):
    """Check observable structure independently of declared validity flags."""
    try:
        state, end, observed_valid = _observation(step.observation)
        _array(step.requested_action, (2,), "requested_action")
        _array(step.applied_action, (2,), "applied_action")
        actual = _scalar(step.actual_dt, "actual_dt")
        transition_valid = _boolean(step.transition_valid, "transition_valid")
        _require(isinstance(step.substep_states, np.ndarray) and step.substep_states.ndim == 2
                 and step.substep_states.shape[1] == 10, "substep_states requires [N,10]")
        n = len(step.substep_states)
        substate = _array(step.substep_states, (n, 10), "substep_states", finite=False)
        times = _array(step.substep_times, (n,), "substep_times")
        _require(isinstance(step.diagnostics, dict), "diagnostics requires dict")
        diagnostics = step.diagnostics
        _require(tuple(diagnostics.get("state_fields", ())) == STATE_FIELDS,
                 "Diagnostic state field order mismatch")
        samples = diagnostics.get("samples")
        _require(isinstance(samples, list) and len(samples) == n+1,
                 "Raw samples must include start plus every substep")
        violations = diagnostics.get("validity_violations")
        _require(isinstance(violations, (list, tuple))
                 and all(isinstance(item, str) and item for item in violations),
                 "validity_violations requires a string sequence")
        raw_states, raw_times = [], []
        for index, sample in enumerate(samples):
            _require(isinstance(sample, dict), "Raw sample requires dict")
            raw_states.append(_raw_state(sample["state"], f"raw_sample[{index}].state"))
            raw_times.append(_scalar(sample["sim_time"], f"raw_sample[{index}].sim_time"))
        raw_states = np.asarray(raw_states, dtype=np.float64)
        raw_times = np.asarray(raw_times, dtype=np.float64)
        _require(abs(raw_times[0]-status.last_sim_time) <= tolerance,
                 "Raw interval start differs from previous action boundary")
        _require(bool((np.diff(raw_times) > 0).all()), "Substep timestamps must strictly increase")
        _require(bool(np.allclose(raw_times[1:], times, rtol=0, atol=tolerance)),
                 "Raw and substep timestamps differ")
        _require(np.array_equal(raw_states[1:], substate, equal_nan=True),
                 "Raw and substep states differ")
        _require(abs(end-raw_times[-1]) <= tolerance
                 and np.array_equal(state, raw_states[-1], equal_nan=True),
                 "Final observation differs from final physical sample")
        _require(abs((end-status.last_sim_time)-actual) <= tolerance,
                 "actual_dt differs from observed elapsed time")
        _require(actual <= dt+tolerance, "Transition exceeds one action interval")
        return state, end, observed_valid, actual, transition_valid, raw_states, raw_times, tuple(violations)
    except (AttributeError, KeyError, TypeError) as error:
        raise TaskError(f"Malformed StepResult: {error}") from error


def _failure_reason(violations, numerical_failure):
    numerical = {"nonfinite_state", "physics_warning", "physics_time_discontinuity",
                 "observation_invalid", "task_metrics_nonfinite"}
    workspace = {"object_corners_outside_workspace", "pusher_extent_outside_workspace"}
    planarity = {"height_deviation", "tilt_exceeded", "vertical_speed_exceeded",
                 "horizontal_angular_speed_exceeded"}
    if numerical_failure or numerical.intersection(violations):
        return "physics_error"
    if workspace.intersection(violations):
        return "workspace_exit"
    if planarity.intersection(violations):
        return "planarity_violation"
    return "physics_error"


def consume_step(status, step, task, action_dt_s, *, time_tolerance_s=1e-9):
    """Consume all actual samples, without counting an interval boundary twice."""
    _status(status)
    _require(status.phase in {"running", "holding"}, "Cannot consume a terminal task")
    tolerance = _scalar(time_tolerance_s, "time_tolerance_s")
    dt = _task(task, action_dt_s, tolerance=tolerance)
    _require(tolerance < dt/2, "Time tolerance must be smaller than half an action interval")
    _require(abs(status.elapsed_s-round(status.elapsed_s/dt)*dt) <= tolerance,
             "Active status is not at an action boundary")
    state, end, observed_valid, actual, transition_valid, raw_states, raw_times, violations = \
        _step(step, status, dt, tolerance)
    _require(end-status.initial_sim_time <= task.max_episode_time_s+tolerance,
             "Step exceeds maximum permitted episode time")
    numerical_failure = not bool(np.isfinite(raw_states).all())
    reasons = list(dict.fromkeys(violations))
    if numerical_failure and "nonfinite_state" not in reasons:
        reasons.append("nonfinite_state")
    if not observed_valid and not reasons:
        reasons.append("observation_invalid")
    if abs(actual-dt) > tolerance or len(raw_states) < 2:
        if "partial_transition" not in reasons:
            reasons.append("partial_transition")
    if not transition_valid and not reasons:
        reasons.append("invalid_transition")

    held, previous, last_time = status.held_time_s, status.previous_qualifies, status.last_sim_time
    try:
        initial_qualifies = handoff_metrics(raw_states[0], task)["qualifies"] \
            if np.isfinite(raw_states[0]).all() else False
    except TaskError:
        numerical_failure = True
        initial_qualifies = False
        if "task_metrics_nonfinite" not in reasons:
            reasons.append("task_metrics_nonfinite")
    if not numerical_failure:
        _require(initial_qualifies == previous,
                 "Initial sample disagrees with previous hold qualification")
    for sample, sample_time in zip(raw_states[1:], raw_times[1:]):
        if not np.isfinite(sample).all():
            held, previous = 0., False
        else:
            try:
                qualified = handoff_metrics(sample, task)["qualifies"]
            except TaskError:
                numerical_failure, qualified = True, False
                if "task_metrics_nonfinite" not in reasons:
                    reasons.append("task_metrics_nonfinite")
            held = held+float(sample_time-last_time) if qualified and previous else 0.
            previous = qualified
        last_time = float(sample_time)
    if reasons:
        return TaskStatus("failed", status.initial_sim_time, end, held, previous,
                          _failure_reason(reasons, numerical_failure), tuple(reasons))
    if held+tolerance >= task.hold_time_s:
        return TaskStatus("ready", status.initial_sim_time, end, held, previous, "ready", ())
    if end-status.initial_sim_time+tolerance >= task.max_episode_time_s:
        return TaskStatus("failed", status.initial_sim_time, end, held, previous, "timeout", ())
    return TaskStatus("holding" if pose_in_goal(state, task) else "running",
                      status.initial_sim_time, end, held, previous, None, ())


def fail_task(status, reason):
    """Record an explicit supervisory failure; caller owns failure priority.

    Allows overriding readiness at the same boundary when an external geometry
    checker discovers a higher-priority failure. No further step may be consumed.
    """
    _status(status)
    _require(isinstance(reason, str) and reason, "Failure reason requires a nonempty string")
    return replace(status, phase="failed", terminal_reason=reason)
