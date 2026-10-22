"""Post-freeze fixed-candidate A/B physical probes, without model or training.

A uses the exact nominal H5 actions and the original H5 cost. Its readiness
monitor is passive: the first official ready status is frozen while physics
continues to H5. B applies the original *nonlatched* current-pose zero rule,
then three zero intervals, and terminates at ready/physical failure/deadline.
Neither panel implements future replanning or an online oracle.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from .candidate_replay import _same_tree, _validate_snapshot, snapshots_exact
from .contact_geometry import check_contact_geometry
from .environment import PlanarPushEnv, SnapshotBundle, StepResult
from .planning import PredictionBatch, TaskSpec, _action_batch, _array, score
from .task import (TaskError, TaskStatus, _status, _step, _task, consume_step,
                   fail_task, handoff_metrics, pose_in_goal)


@dataclass(frozen=True)
class FollowupProbe:
    panel: str
    states: np.ndarray
    boundary_times: np.ndarray
    nominal_actions: np.ndarray
    requested_actions: np.ndarray
    applied_actions: np.ndarray
    supervisor_overrides: np.ndarray
    physical_valid: bool
    horizon_completed: bool
    completed_intervals: int
    termination: str
    reasons: tuple[str, ...]
    cost: float | None
    cost_components: dict[str, float]
    first_ready_interval: int | None
    first_ready_sim_time: float | None
    first_ready_elapsed_s: float | None
    ready_prefix: bool
    ready: bool
    ready_eligible: bool
    task_statuses: tuple[TaskStatus, ...]
    diagnostic_hold_s: np.ndarray
    diagnostic_qualifies: np.ndarray
    hold_traces: tuple[np.ndarray, ...]
    qualification_traces: tuple[np.ndarray, ...]
    geometries: tuple[dict, ...]
    results: tuple[StepResult, ...]
    initial_snapshot_exact: bool
    final_snapshot: SnapshotBundle


def _passive_hold(status, step, task, dt, tolerance):
    """Independent nonterminal tracker; never consumes a terminal TaskStatus.

    The existing observable-contract validator is reused. The separate status
    is active only as a structural record; no ready or timeout verdict is made
    here. The original task supervisor alone supplies the first-hit verdict.
    """
    _, end, _, _, _, raw_states, raw_times, _ = _step(step, status, dt, tolerance)
    held, previous, last = status.held_time_s, status.previous_qualifies, status.last_sim_time
    initial_finite = bool(np.isfinite(raw_states[0]).all())
    initial = handoff_metrics(raw_states[0], task)["qualifies"] if initial_finite else False
    if initial_finite and initial != previous:
        raise TaskError("Initial sample disagrees with previous diagnostic hold qualification")
    holds, qualifications = [held], [previous]
    for state, time in zip(raw_states[1:], raw_times[1:]):
        try:
            qualified = bool(np.isfinite(state).all()) and handoff_metrics(state, task)["qualifies"]
        except TaskError:
            qualified = False
        held = held+float(time-last) if qualified and previous else 0.
        previous, last = qualified, float(time)
        holds.append(held)
        qualifications.append(previous)
    pose = bool(np.isfinite(raw_states[-1]).all()) and pose_in_goal(raw_states[-1], task)
    following = TaskStatus("holding" if pose else "running", status.initial_sim_time,
                           end, held, previous, None, ())
    return following, np.asarray(holds, np.float64), np.asarray(qualifications, np.bool_)


def _physical_reasons(result, command, env, tolerance, geometry):
    """Match frozen candidate_replay validity, preserving a failing prefix."""
    local = []
    if not result.transition_valid:
        local.extend(result.diagnostics.get("validity_violations", []) or ["invalid_physical_transition"])
    if not result.observation.observation_valid:
        local.append("invalid_observed_endpoint")
    if not np.isfinite(result.observation.state).all():
        local.append("nonfinite_observed_endpoint")
    if not np.array_equal(result.applied_action, command):
        local.append("candidate_applied_action_mismatch")
    if not np.array_equal(result.requested_action, command):
        local.append("candidate_requested_action_mismatch")
    if not math.isfinite(result.actual_dt) or abs(result.actual_dt-env.action_dt) > tolerance:
        local.append("partial_physical_interval")
    if not geometry["pass"]:
        local.extend(geometry["reasons"])
    return list(dict.fromkeys(local))


def run_followup_probe(physical_config, planner_config, snapshot, task,
                       original_status, actions, previous_applied_action, profile,
                       *, panel="A", env_factory=None, on_step=None):
    """Run one candidate/repeat from an exactly restored full snapshot.

    Infrastructure/schema/restore failures raise; callers must preserve any
    separately recorded execution prefix and may not silently retry/remove a
    candidate. Physical invalidity is returned with its raw prefix. ``ready``
    means an official first hit occurred; ``ready_eligible`` additionally
    requires physical validity across the prescribed A horizon (or B prefix
    ending at its absorbing ready). B has no H8-as-H5 cost: cost is None.
    All times are original simulator/episode times, never reset to this probe.
    Optional ``on_step(index, result, command)`` is invoked immediately after
    each returned physical result, before checks, so a recording caller can
    preserve an executed prefix even if a later schema/transport check raises.
    """
    if on_step is not None and not callable(on_step):
        raise ValueError("on_step must be callable or None")
    if panel not in {"A", "B"}:
        raise ValueError("panel must be A or B")
    _array(actions, (5, 2), "nominal_actions")
    _action_batch(actions[None], planner_config)
    if planner_config["planner"]["horizon"] != 5:
        raise ValueError("Followup design requires the unchanged H5 planner")
    if planner_config["planner"].get("block_lengths") != [2, 3]:
        raise ValueError("Followup design requires unchanged [2,3] action blocks")
    for key in ("physics", "object", "pusher", "workspace"):
        if not _same_tree(physical_config[key], planner_config[key]):
            raise ValueError(f"Physical and cost/planner {key} contracts differ")
    if not isinstance(task, TaskSpec):
        raise ValueError("Probe requires TaskSpec")
    _status(original_status)
    if original_status.phase not in {"running", "holding"}:
        raise ValueError("Probe must start from the active status before failure/ready")
    dt = _task(task, physical_config["physics"]["action_dt_s"])
    tolerance = physical_config["validity"]["dt_match_tolerance_s"]
    if not math.isfinite(tolerance) or not 0 <= tolerance < dt/2:
        raise ValueError("Invalid time tolerance")
    if abs(dt-.1) > tolerance:
        raise ValueError("Followup design requires action_dt=.1s")
    if abs(original_status.elapsed_s-round(original_status.elapsed_s/dt)*dt) > tolerance:
        raise ValueError("Original status must be on an action boundary")
    if original_status.elapsed_s >= task.max_episode_time_s-tolerance:
        raise ValueError("Original active status is already at the episode deadline")
    # A must remain an identical-length H5 comparison; never shorten it silently.
    if panel == "A" and original_status.elapsed_s+5*dt > task.max_episode_time_s+tolerance:
        raise ValueError("A H5 would exceed the original episode deadline")
    _array(previous_applied_action, (2,), "previous_applied_action")
    if not isinstance(profile, dict) or profile.get("physics_suite", {}).get("check_contact_geometry") is not True:
        raise ValueError("Probe requires enabled contact geometry checks")
    budget = profile["physics_suite"]["actual_contact_penetration_max_m"]
    check_contact_geometry([{"contacts": []}], budget)
    # Validate unchanged cost contract before any physical work, also for B.
    empty = PredictionBatch(np.full((1, 6, 10), np.nan, np.float64),
                            np.full((1, 5, 10), np.nan, np.float64),
                            np.array([False]), ("not_executed",), np.array([0], np.int64))
    score(empty, actions[None], task, previous_applied_action, planner_config)
    factory = PlanarPushEnv if env_factory is None else env_factory
    env = factory(deepcopy(physical_config))
    _validate_snapshot(snapshot, env)
    observed = env.restore(deepcopy(snapshot))
    if not snapshots_exact(snapshot, env.snapshot()):
        raise ValueError("Restored physical snapshot differs from its complete source")
    _array(observed.state, (10,), "restored_observed_state")
    if observed.observation_valid is not True:
        raise ValueError("Restored snapshot observation is physically invalid")
    if abs(observed.sim_time-original_status.last_sim_time) > tolerance:
        raise ValueError("Restored simulator clock differs from original TaskStatus")
    if handoff_metrics(observed.state, task)["qualifies"] != original_status.previous_qualifies:
        raise ValueError("Restored qualification differs from original TaskStatus")
    if (original_status.phase == "holding") != pose_in_goal(observed.state, task):
        raise ValueError("Restored pose phase differs from original TaskStatus")
    count = 5 if panel == "A" else 8
    nominal = np.zeros((count, 2), np.float64)
    nominal[:5] = actions
    states = np.full((count+1, 10), np.nan, np.float64)
    times = np.full(count+1, np.nan, np.float64)
    requested, applied = (np.full((count, 2), np.nan, np.float64) for _ in range(2))
    overrides = np.zeros(count, np.bool_)
    holds = np.full(count+1, np.nan, np.float64)
    qualified = np.zeros(count+1, np.bool_)
    states[0], times[0] = observed.state.copy(), observed.sim_time
    holds[0], qualified[0] = original_status.held_time_s, original_status.previous_qualifies
    official = deepcopy(original_status)
    tracker = deepcopy(original_status)
    statuses, results, geometries, hold_traces, qual_traces, reasons = [official], [], [], [], [], []
    first_ready_interval = first_ready_time = first_ready_elapsed = None
    completed, termination = 0, "horizon_completed"
    for index in range(count):
        if tracker.elapsed_s+dt > task.max_episode_time_s+tolerance:
            termination = "timeout"
            break
        command = nominal[index].copy()
        if panel == "B" and (index >= 5 or pose_in_goal(states[index], task)):
            command[:] = 0.
            overrides[index] = index < 5 and not np.array_equal(command, nominal[index])
        result = env.step(command.copy())
        results.append(result)
        if on_step is not None:
            on_step(index, result, command.copy())
        _array(result.observation.state, (10,), "observed_endpoint", finite=False)
        _array(result.applied_action, (2,), "observed_applied_action", finite=False)
        states[index+1], times[index+1] = result.observation.state, result.observation.sim_time
        _array(result.requested_action, (2,), "observed_requested_action", finite=False)
        requested[index], applied[index] = result.requested_action, result.applied_action
        geometry = check_contact_geometry(result.diagnostics["samples"], budget)
        geometries.append(geometry)
        local = _physical_reasons(result, command, env, tolerance, geometry)
        if math.isfinite(result.actual_dt) and abs(result.actual_dt-dt) <= tolerance:
            completed += 1
        tracker, hold_trace, qual_trace = _passive_hold(tracker, result, task, dt, tolerance)
        hold_traces.append(hold_trace)
        qual_traces.append(qual_trace)
        holds[index+1], qualified[index+1] = tracker.held_time_s, tracker.previous_qualifies
        if official.phase in {"running", "holding"}:
            official = consume_step(official, result, task, dt, time_tolerance_s=tolerance)
            if local:
                official = fail_task(official, "physical_invalid")
            if official.phase == "ready" and first_ready_interval is None:
                first_ready_interval = index+1
                first_ready_time = float(result.observation.sim_time)
                first_ready_elapsed = float(official.elapsed_s)
        statuses.append(official)
        if local:
            reasons.extend(f"interval_{index+1}:{reason}" for reason in local)
            termination = "physical_invalid"
            break
        if official.phase == "failed":
            termination = official.terminal_reason or "task_failed"
            break
        if panel == "B" and first_ready_interval is not None:
            termination = "ready"
            break
    physical_valid = not reasons
    full = len(results) == count and completed == count
    ready = first_ready_interval is not None
    cost, components = None, {}
    if panel == "A":
        valid_cost = physical_valid and full
        pred = PredictionBatch(states[None], np.diff(states, axis=0)[None],
                               np.array([valid_cost]), (";".join(reasons),),
                               np.array([-1 if valid_cost else len(results)], np.int64))
        costs = score(pred, actions[None], task, previous_applied_action, planner_config)
        cost = float(costs.total[0])
        components = {key: float(value[0]) for key, value in costs.components.items()}
        if valid_cost and not math.isfinite(cost):
            # Cost validity is separate from actual physical validity.
            reasons.append("nonfinite_physical_candidate_cost")
            termination = "nonfinite_cost"
    eligible = ready and physical_valid and (full if panel == "A" else True)
    return FollowupProbe(panel, states, times, nominal, requested, applied, overrides,
                         physical_valid, full, completed, termination, tuple(reasons),
                         cost, components, first_ready_interval, first_ready_time,
                         first_ready_elapsed, ready and first_ready_interval <= 5,
                         ready, eligible, tuple(statuses), holds, qualified,
                         tuple(hold_traces), tuple(qual_traces), tuple(geometries),
                         tuple(results), True, env.snapshot())
