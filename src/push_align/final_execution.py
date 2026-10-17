"""Shared episode execution with explicit unexecuted and incomplete outcomes.

The caller owns model loading, source bindings, durable raw records and the
predeclared case list. This module executes the existing planner and supervisor
without changing either. Development limits do not change the task timeout.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import traceback

import numpy as np

from .closed_loop import ENGINE_GUARDS, consume_actual_step, model_planning_scope
from .contact_geometry import check_contact_geometry
from .data import _encode_nonfinite
from .environment import Observation, SnapshotBundle
from .geometry_control import GeometryContext
from .planning import PlannerContext, TaskSpec, plan
from .planning_protocol import GeometryProposalAdapter, keyed_planner_rng
from .task import TaskStatus, _status, fail_task, handoff_metrics, pose_in_goal, start_task


class FinalExecutionError(ValueError):
    """An execution request violates the pre-execution contract."""


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise FinalExecutionError(f"{name} requires a nonnegative integer")
    return int(value)


def _identity(value, name):
    if not isinstance(value, str) or not value:
        raise FinalExecutionError(f"{name} requires a nonempty string")
    return value


def _json_hash(value):
    return hashlib.sha256(json.dumps(_encode_nonfinite(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_hash(value):
    """Same dtype/shape/bytes convention as the existing P3/P4 records."""
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.dtype.str.encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def _context_dict(context):
    return _encode_nonfinite(asdict(context))


def _plan_identity(planned):
    index = planned.selected_index
    choice = {"selected_index": index, "reason": planned.reason,
              "action": planned.action.tolist(), "total_cost_hash": array_hash(planned.costs.total)}
    return {**choice,
            "candidate_hash": array_hash(planned.candidates.actions),
            "prediction_hash": array_hash(planned.predictions.states),
            "delta_hash": array_hash(planned.predictions.raw_deltas),
            "choice_hash": _json_hash(choice),
            "valid_predictions": int(planned.predictions.valid.sum()),
            "eligible_costs": int((planned.predictions.valid & np.isfinite(planned.costs.total)).sum()),
            "invalid_reasons": dict(Counter(reason for reason in planned.predictions.reasons if reason)),
            "selected_cost": None if index is None else float(planned.costs.total[index]),
            "selected_actions": None if index is None else planned.candidates.actions[index].tolist(),
            "candidate_generation": deepcopy(planned.candidates.diagnostics)}


@dataclass
class EpisodeExecution:
    execution_id: str
    case_id: str
    controller_id: str
    origin: dict
    task_spec: TaskSpec
    initial_snapshot: object
    run_scope: str
    initial_observation: object | None = None
    initial_task: TaskStatus | None = None
    final_task: TaskStatus | None = None
    initial_context: PlannerContext | None = None
    final_context: PlannerContext | None = None
    initial_gctx: object | None = None
    final_gctx: object | None = None
    records: list = field(default_factory=list)
    results: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    unexecuted_plan: dict | None = None
    plan_results: dict = field(default_factory=dict, repr=False)
    captured_inputs: dict = field(default_factory=dict, repr=False)
    counts: dict = field(default_factory=lambda: {
        "controller_decisions": 0, "physics_interval_attempts": 0,
        "physics_intervals_returned": 0, "completed_boundary_records": 0})
    future_engine_calls: dict = field(default_factory=lambda: dict.fromkeys(ENGINE_GUARDS, 0))
    initial_full_snapshot_equal: bool = False
    execution_status: str = "incomplete"
    termination: str = "not_started"
    error: dict | None = None

    @property
    def cell_id(self):
        return f"{self.controller_id}/{self.case_id}"

    def summary(self):
        """Compact outcome; raw StepResults and full plans stay separate."""
        return {"cell_id": self.cell_id, "execution_id": self.execution_id,
                "case_id": self.case_id, "controller_id": self.controller_id,
                "run_scope": self.run_scope, "execution_status": self.execution_status,
                "termination": self.termination,
                "task": None if self.final_task is None else asdict(self.final_task),
                "counts": deepcopy(self.counts),
                "future_engine_calls": deepcopy(self.future_engine_calls),
                "initial_full_snapshot_equal": self.initial_full_snapshot_equal,
                "error": deepcopy(self.error)}


def execute_episode(env, initial_snapshot, *, case_id, episode_index, controller_id,
                    execution_id, predictor, physical_config, planner_config,
                    geometry_parameters, transition_parameters, candidate_seed_base,
                    max_intervals, run_scope, penetration_budget_m,
                    candidate_stream_id=0, initial_status=None, initial_context=None,
                    initial_gctx=None, origin=None, capture_ticks=()):
    """Run actual control from a bound snapshot and retain every attempted prefix.

    ``predictor=None`` selects G. Otherwise the existing legacy planner is used.
    A resumed run must provide all three supervisory contexts. Its keyed random
    control tick comes from the original task elapsed time, including holding.
    Exceptions after an attempt are explicit infrastructure outcomes; malformed
    invocation arguments raise before any restore/step. No retry is automatic.
    """
    for value, name in ((case_id, "case_id"), (controller_id, "controller_id"),
                        (execution_id, "execution_id")):
        _identity(value, name)
    episode_index = _integer(episode_index, "episode_index")
    seed = _integer(candidate_seed_base, "candidate_seed_base")
    stream = _integer(candidate_stream_id, "candidate_stream_id")
    max_intervals = _integer(max_intervals, "max_intervals")
    if max_intervals < 1 or run_scope not in {"development_preflight", "final_evaluation"}:
        raise FinalExecutionError("Positive max_intervals and an explicit supported run_scope are required")
    if predictor is not None and not callable(predictor):
        raise FinalExecutionError("predictor requires a callable or None for G")
    resumed = (initial_status is not None, initial_context is not None, initial_gctx is not None)
    if any(resumed) and not all(resumed):
        raise FinalExecutionError("Resume requires status, planner context and G context together")
    if all(resumed) and (not isinstance(initial_status, TaskStatus) or
                         not isinstance(initial_context, PlannerContext)):
        raise FinalExecutionError("Resume requires typed task/planner contexts")
    if origin is not None and not isinstance(origin, dict):
        raise FinalExecutionError("origin requires a mapping")
    if not isinstance(capture_ticks, (tuple, list)):
        raise FinalExecutionError("capture_ticks requires an explicit tick sequence")
    capture_ticks = tuple(_integer(value, "capture_tick") for value in capture_ticks)
    if len(set(capture_ticks)) != len(capture_ticks):
        raise FinalExecutionError("capture_ticks cannot contain duplicate slots")
    if isinstance(penetration_budget_m, (bool, np.bool_)) or not isinstance(penetration_budget_m, (int, float)) \
            or not np.isfinite(penetration_budget_m) or penetration_budget_m < 0:
        raise FinalExecutionError("penetration_budget_m requires a finite nonnegative scalar")
    task = TaskSpec.from_config(physical_config)
    dt = float(env.action_dt)
    if not np.isfinite(dt) or dt <= 0 or abs(dt-float(physical_config["physics"]["action_dt_s"])) > 1e-12:
        raise FinalExecutionError("Environment/action interval differs from physical configuration")
    output = EpisodeExecution(execution_id, case_id, controller_id, deepcopy(origin or {}),
                              task, deepcopy(initial_snapshot), run_scope)
    status = context = gctx = None
    try:
        observed = env.restore(initial_snapshot)
        restored = env.snapshot()
        output.initial_full_snapshot_equal = bool(
            np.array_equal(restored.integration_state, initial_snapshot.integration_state)
            and _json_hash(asdict(restored)) == _json_hash(asdict(initial_snapshot)))
        if not output.initial_full_snapshot_equal:
            raise FinalExecutionError("Initial full snapshot failed exact restoration")
        output.initial_observation = deepcopy(observed)
        adapter = GeometryProposalAdapter(physical_config, geometry_parameters, transition_parameters)
        status = deepcopy(initial_status) if all(resumed) else start_task(observed, task, dt)
        context = deepcopy(initial_context) if all(resumed) else PlannerContext.initial()
        gctx = deepcopy(initial_gctx) if all(resumed) else adapter.initial_context()
        _status(status)
        if abs(status.last_sim_time-observed.sim_time) > 1e-9:
            raise FinalExecutionError("Resumed task time differs from current snapshot observation")
        tick_float = status.elapsed_s/dt
        start_tick = round(tick_float)
        if start_tick < 0 or abs(status.elapsed_s-start_tick*dt) > 1e-9:
            raise FinalExecutionError("Initial task is not at an actual action boundary")
        if status.elapsed_s > task.max_episode_time_s+1e-9:
            raise FinalExecutionError("Initial task already exceeds the task episode time")
        if type(context.control_step) is not int or context.control_step != start_tick:
            raise FinalExecutionError("Planner context must match the original actual control tick")
        if np.isfinite(observed.state).all() and observed.observation_valid:
            metrics = handoff_metrics(observed.state, task)
            if status.previous_qualifies != metrics["qualifies"]:
                raise FinalExecutionError("Initial task qualification differs from current observed state")
            if status.phase == "ready" and (not metrics["qualifies"] or status.held_time_s+1e-9 < task.hold_time_s):
                raise FinalExecutionError("Initial ready status lacks the required actual hold")
            if status.phase in {"running", "holding"} and (status.phase == "holding") != metrics["pose_within"]:
                raise FinalExecutionError("Initial active phase differs from current observed pose")
        output.initial_task, output.initial_context, output.initial_gctx = \
            deepcopy(status), deepcopy(context), deepcopy(gctx)
        for local_tick in range(max_intervals):
            if status.phase in {"ready", "failed"}:
                break
            if status.elapsed_s+dt > task.max_episode_time_s+1e-9:
                raise FinalExecutionError("Active task has no complete permitted action interval remaining")
            tick = start_tick+local_tick
            if tick in capture_ticks:
                # Capture present physical inputs only. A terminal episode or
                # an unavailable tick never creates a replacement snapshot.
                output.captured_inputs[tick] = {
                    "tick": tick, "snapshot": deepcopy(env.snapshot()),
                    "observation": deepcopy(observed), "task_status": deepcopy(status),
                    "planner_context": deepcopy(context), "geometry_context": deepcopy(gctx)}
            before = observed.state.copy()
            before_time = float(observed.sim_time)
            before_status = asdict(status)
            before_context, before_gctx = _context_dict(context), _context_dict(gctx)
            holding = pose_in_goal(before, task)
            planned = None
            keys = [seed, episode_index, tick, stream]
            with model_planning_scope(output.future_engine_calls):
                proposal = adapter.propose(before.copy(), task, gctx, status.elapsed_s, control_tick=tick)
                if holding:
                    action, source = np.zeros(2, np.float64), "holding"
                elif predictor is None:
                    action, source = proposal.geometry_action, "geometry"
                else:
                    planned = plan(before.copy(), task, context, keyed_planner_rng(*keys), predictor,
                                   planner_config, geometry_action=proposal.geometry_action,
                                   tangent_direction=proposal.tangent_direction)
                    action, source = planned.action, "model"
            plan_info = None if planned is None else _plan_identity(planned)
            decision = {"index": tick, "local_index": local_tick, "source": source,
                        "state": before.tolist(), "sim_time": before_time,
                        "holding": holding, "task_status": before_status,
                        "planner_context_before": before_context,
                        "geometry_context_before": before_gctx,
                        "geometry_proposal": deepcopy(proposal.diagnostics),
                        "rng_keys": keys if planned is not None else None,
                        "plan_identity": deepcopy(plan_info), "executed": False}
            output.decisions.append(decision)
            output.counts["controller_decisions"] += 1
            if planned is not None:
                output.plan_results[tick] = planned
                if planned.reason == "planner_invalid":
                    output.unexecuted_plan = deepcopy(decision)
                    status = fail_task(status, "planner_invalid")
                    break
                context = planned.context
            elif action is None:
                output.unexecuted_plan = deepcopy(decision)
                status = fail_task(status, "controller_infeasible")
                break
            # Context commit records the current observation. It never asserts
            # that the G proposal was the command actually executed.
            gctx = adapter.commit(proposal, actual_control_tick=tick)
            output.counts["physics_interval_attempts"] += 1
            step = env.step(action)
            # Retain raw immediately, even when the checker or boundary record
            # below fails; an executed prefix must not vanish from the result.
            output.results.append(step)
            output.counts["physics_intervals_returned"] += 1
            decision["executed"] = True
            geometry = check_contact_geometry(step.diagnostics["samples"], penetration_budget_m)
            if predictor is None and not holding:
                # plan() already advances learned contexts and holding is
                # advanced by consume_actual_step. G needs its single advance.
                context = PlannerContext(context.previous_applied_action.copy(),
                                         context.optimal_actions, context.control_step+1)
            status, context = consume_actual_step(status, step, task, dt, context,
                                                  holding=holding, geometry=geometry)
            observed = step.observation
            record = {"index": tick, "local_index": local_tick, "source": source,
                      "state_before": before.tolist(), "state_after": observed.state.tolist(),
                      "sim_time_before": before_time, "sim_time_after": float(observed.sim_time),
                      "requested_action": step.requested_action.tolist(),
                      "applied_action": step.applied_action.tolist(),
                      "actual_dt_s": float(step.actual_dt), "holding": holding,
                      "task_before": before_status, "task_after": asdict(status),
                      "planner_context_before": before_context, "planner_context_after": _context_dict(context),
                      "geometry_context_before": before_gctx, "geometry_context_after": _context_dict(gctx),
                      "geometry": geometry, "observation_valid": bool(observed.observation_valid),
                      "transition_valid": bool(step.transition_valid), "plan_identity": plan_info,
                      "rng_keys": keys if planned is not None else None}
            output.records.append(record)
            output.counts["completed_boundary_records"] += 1
        if status.phase in {"ready", "failed"}:
            output.execution_status = "completed"
            output.termination = status.terminal_reason
        elif run_scope == "development_preflight":
            output.execution_status, output.termination = "completed", "development_budget"
        else:
            output.execution_status, output.termination = "incomplete", "missing_terminal_outcome"
    except Exception as error:
        output.execution_status, output.termination = "infrastructure_error", "infrastructure_error"
        output.error = {"type": type(error).__name__, "message": str(error),
                        "traceback": traceback.format_exc()}
    finally:
        output.final_task, output.final_context, output.final_gctx = status, context, gctx
    return output


def check_snapshot_completeness(expected_ticks, captured_inputs):
    """Check exact predeclared snapshot slots; never replace a missing tick."""
    if not isinstance(expected_ticks, (list, tuple)) or not expected_ticks:
        raise FinalExecutionError("Expected snapshot ticks require a nonempty sequence")
    expected = [_integer(value, "expected_tick") for value in expected_ticks]
    if len(set(expected)) != len(expected):
        raise FinalExecutionError("Expected snapshot ticks must be unique")
    if not isinstance(captured_inputs, dict):
        raise FinalExecutionError("Captured inputs require a tick-indexed mapping")
    for key in captured_inputs:
        _integer(key, "captured_tick")
    missing = [tick for tick in expected if tick not in captured_inputs]
    extra = [tick for tick in captured_inputs if tick not in expected]
    invalid = []
    for tick, value in captured_inputs.items():
        if not isinstance(value, dict) or value.get("tick") != tick or not (
                isinstance(value.get("snapshot"), SnapshotBundle)
                and isinstance(value.get("observation"), Observation)
                and isinstance(value.get("task_status"), TaskStatus)
                and isinstance(value.get("planner_context"), PlannerContext)
                and isinstance(value.get("geometry_context"), GeometryContext)):
            invalid.append(tick)
    return {"status": "PASS_COMPLETE" if not (missing or extra or invalid) else "INCOMPLETE",
            "expected_ticks": expected, "available_ticks": list(captured_inputs),
            "expected_count": len(expected), "available_count": len(captured_inputs),
            "missing": missing, "extra": extra, "invalid_ticks": invalid,
            "substitution_permitted": False}


def check_result_completeness(expected_cell_ids, results, *, expected_run_scope=None):
    """Audit the declared denominator; never select only surviving outcomes.

    Missing/duplicate/unexpected/reordered rows and incomplete attempts fail the
    audit. A completed task failure such as planner_invalid remains a present
    valid outcome. Infrastructure attempts remain counted but not complete.
    """
    if not isinstance(expected_cell_ids, (list, tuple)) or not expected_cell_ids \
            or any(not isinstance(item, str) or not item for item in expected_cell_ids) \
            or len(set(expected_cell_ids)) != len(expected_cell_ids):
        raise FinalExecutionError("Expected cell IDs must be nonempty and unique")
    if expected_run_scope not in {None, "development_preflight", "final_evaluation"}:
        raise FinalExecutionError("Unsupported expected execution scope")
    rows = [item.summary() if isinstance(item, EpisodeExecution) else item for item in results]
    if any(not isinstance(row, dict) for row in rows):
        raise FinalExecutionError("Results require EpisodeExecution or mapping rows")
    actual = [row.get("cell_id") for row in rows]
    counts = Counter(item for item in actual if isinstance(item, str))
    missing = [item for item in expected_cell_ids if counts[item] == 0]
    duplicate = sorted(item for item, count in counts.items() if count > 1)
    unexpected = [item for item in actual if item not in expected_cell_ids]
    task_outcomes = {"ready", "timeout", "planner_invalid", "controller_infeasible", "physics_error",
                     "geometry_invalid", "workspace_exit", "planarity_violation"}
    scope_mismatch = [row.get("cell_id") for row in rows if expected_run_scope is not None
                      and row.get("run_scope") != expected_run_scope]
    incomplete = [row.get("cell_id") for row in rows
                  if row.get("execution_status") != "completed"
                  or (row.get("termination") not in task_outcomes and not (
                      row.get("termination") == "development_budget"
                      and row.get("run_scope") == "development_preflight"))]
    ordered = actual == list(expected_cell_ids)
    return {"status": "PASS_COMPLETE" if ordered and not incomplete and not scope_mismatch else "INCOMPLETE",
            "expected_count": len(expected_cell_ids), "attempted_rows": len(rows),
            "missing": missing, "duplicate": duplicate, "unexpected": unexpected,
            "order_matches": ordered, "incomplete_rows": incomplete,
            "expected_run_scope": expected_run_scope, "scope_mismatch_rows": scope_mismatch,
            "denominator": list(expected_cell_ids),
            "task_failures_are_present_outcomes": True}
