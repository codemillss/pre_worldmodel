"""Offline physical counterfactual for a fixed recorded candidate sequence.

Each call restores a fresh simulator. No learned model, controller, task
supervisor or candidate repair selects/changes commands. The existing pure
planner cost evaluates observed post-action states, while actual simulator
and contact-geometry evidence alone determines physical validity. A finite
cost is not a downstream handoff-success verdict.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from .contact_geometry import check_contact_geometry
from .environment import PlanarPushEnv, SnapshotBundle, StepResult
from .planning import PredictionBatch, TaskSpec, _action_batch, _array, score


@dataclass(frozen=True)
class PhysicalCandidateReplay:
    states: np.ndarray
    applied_actions: np.ndarray
    valid: bool
    cost: float
    cost_components: dict
    completed_intervals: int
    results: list[StepResult]
    initial_snapshot_exact: bool
    final_snapshot: SnapshotBundle
    reasons: tuple[str, ...]


def _same_tree(left, right):
    if isinstance(left, np.ndarray):
        return (isinstance(right, np.ndarray) and left.dtype == right.dtype
                and left.shape == right.shape and np.array_equal(left, right))
    if isinstance(left, dict):
        return (isinstance(right, dict) and set(left) == set(right)
                and all(_same_tree(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_same_tree(a, b) for a, b in zip(left, right)))
    return type(left) is type(right) and left == right


def snapshots_exact(left, right):
    """Compare every declared physical snapshot field, including local RNG."""
    return (isinstance(left, SnapshotBundle) and isinstance(right, SnapshotBundle)
            and all(_same_tree(getattr(left, name), getattr(right, name))
                    for name in SnapshotBundle.__dataclass_fields__))


def _validate_snapshot(snapshot, env):
    if not isinstance(snapshot, SnapshotBundle):
        raise ValueError("Physical replay requires a complete SnapshotBundle")
    _array(snapshot.integration_state, env.snapshot().integration_state.shape,
           "snapshot.integration_state")
    if (type(snapshot.needs_reset) is not bool or snapshot.needs_reset
            or type(snapshot.reference_height) not in (int, float)
            or not math.isfinite(snapshot.reference_height)
            or not isinstance(snapshot.rng_state, dict)):
        raise ValueError("Snapshot must be finite and executable without reset")
    for name in ("warning_counts", "warning_lastinfo"):
        actual, expected = getattr(snapshot, name), getattr(env.snapshot(), name)
        if (not isinstance(actual, np.ndarray) or actual.shape != expected.shape
                or actual.dtype != expected.dtype or (actual < 0).any()):
            raise ValueError(f"Invalid snapshot {name}")


def replay_candidate(physical_config, planner_config, snapshot, actions, task,
                     previous_applied_action, profile):
    """Execute the entire fixed horizon, preserving an invalid physical prefix.

    ``completed_intervals`` counts attempted intervals whose actual duration
    reaches the registered action dt, including a later geometry rejection.
    ``results`` also retains a partial failing interval; its observed endpoint
    is preserved before NaN suffix padding. All-invalid costs are infinity.
    Readiness never ends/replaces the supplied candidate: this is an open-loop
    diagnostic, not a task-supervised execution or an online oracle planner.
    Restore/config/schema errors are infrastructure exceptions, not candidates
    that may be silently removed from the population.
    """
    if (not isinstance(actions, np.ndarray) or actions.ndim != 2
            or actions.shape[0] < 1 or actions.shape[1] != 2):
        raise ValueError("Physical candidate actions require float64[H,2]")
    horizon = len(actions)
    _array(actions, (horizon, 2), "physical_candidate_actions")
    _action_batch(actions[None], planner_config)
    if (type(planner_config["planner"]["horizon"]) is not int
            or planner_config["planner"]["horizon"] != horizon):
        raise ValueError("Physical candidate horizon differs from the registered planner")
    for key in ("physics", "object", "pusher", "workspace"):
        if not _same_tree(physical_config[key], planner_config[key]):
            raise ValueError(f"Physical and cost/planner {key} contracts differ")
    if not isinstance(task, TaskSpec):
        raise ValueError("Physical candidate requires TaskSpec")
    task.validate()
    _array(previous_applied_action, (2,), "previous_applied_action")
    if not isinstance(profile, dict) or profile.get("physics_suite", {}).get("check_contact_geometry") is not True:
        raise ValueError("Physical candidate requires enabled contact geometry checks")
    budget = profile["physics_suite"]["actual_contact_penetration_max_m"]
    # Validate geometry-budget/schema and cost contracts before physical work.
    check_contact_geometry([{"contacts": []}], budget)
    empty = PredictionBatch(np.full((1, horizon + 1, 10), np.nan, np.float64),
                            np.full((1, horizon, 10), np.nan, np.float64),
                            np.array([False]), ("not_executed",), np.array([0], np.int64))
    score(empty, actions[None], task, previous_applied_action, planner_config)
    requested = actions.copy()
    env = PlanarPushEnv(deepcopy(physical_config))
    _validate_snapshot(snapshot, env)
    initial = env.restore(deepcopy(snapshot))
    if not snapshots_exact(snapshot, env.snapshot()):
        raise ValueError("Restored physical snapshot does not exactly match its complete source")
    _array(initial.state, (10,), "restored_observed_state")
    if initial.observation_valid is not True:
        raise ValueError("Restored snapshot observation is physically invalid")
    states = np.full((horizon + 1, 10), np.nan, np.float64)
    applied = np.full((horizon, 2), np.nan, np.float64)
    states[0] = initial.state.copy()
    results, reasons, completed = [], [], 0
    tolerance = physical_config["validity"]["dt_match_tolerance_s"]
    for index, command in enumerate(requested):
        result = env.step(command.copy())
        results.append(result)
        _array(result.observation.state, (10,), "observed_endpoint", finite=False)
        _array(result.applied_action, (2,), "observed_applied_action", finite=False)
        states[index + 1] = result.observation.state
        applied[index] = result.applied_action
        if math.isfinite(result.actual_dt) and abs(result.actual_dt - env.action_dt) <= tolerance:
            completed += 1
        geometry = check_contact_geometry(result.diagnostics["samples"], budget)
        local = []
        if not result.transition_valid:
            local.extend(result.diagnostics.get("validity_violations", []) or ["invalid_physical_transition"])
        if not result.observation.observation_valid:
            local.append("invalid_observed_endpoint")
        if not np.isfinite(result.observation.state).all():
            local.append("nonfinite_observed_endpoint")
        if not np.array_equal(result.applied_action, command):
            local.append("candidate_applied_action_mismatch")
        if not math.isfinite(result.actual_dt) or abs(result.actual_dt - env.action_dt) > tolerance:
            local.append("partial_physical_interval")
        if not geometry["pass"]:
            local.extend(geometry["reasons"])
        if local:
            reasons.extend(f"interval_{index + 1}:{item}" for item in dict.fromkeys(local))
            break
    valid = not reasons and len(results) == horizon
    observed = PredictionBatch(states[None], np.diff(states, axis=0)[None],
                               np.array([valid]), (";".join(reasons),),
                               np.array([-1 if valid else len(results)], np.int64))
    costs = score(observed, requested[None], task, previous_applied_action, planner_config)
    cost = float(costs.total[0])
    if valid and not math.isfinite(cost):
        valid = False
        reasons.append("nonfinite_physical_candidate_cost")
    return PhysicalCandidateReplay(states, applied, valid, cost,
                                   {key: float(value[0]) for key, value in costs.components.items()},
                                   completed, results, True, env.snapshot(), tuple(reasons))
