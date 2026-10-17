"""Observed-state geometric feedback, without contact labels or dynamics.

The rear-face controller explicitly retracts before changing contact offset.
Its path check concerns a segment around the *currently observed*, radius-
expanded rectangle. It neither predicts moving-body collision avoidance nor
certifies the physical execution of an intended push. The common supervisor
retains responsibility for actual contact geometry, validity and readiness.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .planning import TaskSpec
from .representation import STATE_FIELDS


class GeometryControlError(ValueError):
    """A configuration, state, context or time violates the G contract."""


MODES = ("RETRACT", "SLIDE", "APPROACH", "PUSH")
PARAMETER_NAMES = (
    "position_gain_per_s", "speed_max_m_s", "waypoint_gain_per_s",
    "waypoint_velocity_damping", "approach_speed_max_m_s", "yaw_creep_speed_m_s",
    "yaw_offset_gain_m_rad", "angular_damping_gain_m_s_rad",
    "contact_offset_abs_max_m", "retreat_gap_m", "approach_gap_m",
    "near_contact_loss_gap_m", "waypoint_tolerance_m",
    "contact_offset_replan_threshold_m", "minimum_inward_direction_cosine",
    "contact_mode_max_elapsed_s",
)


def _require(condition, message):
    if not condition:
        raise GeometryControlError(message)


def _number(value, name, *, zero=False):
    _require(not isinstance(value, (bool, np.bool_))
             and isinstance(value, (int, float, np.integer, np.floating)),
             f"{name} requires a finite real scalar")
    result = float(value)
    _require(math.isfinite(result) and (result >= 0 if zero else result > 0),
             f"{name} requires a finite {'nonnegative' if zero else 'positive'} scalar")
    return result


def _finite(value, name):
    _require(not isinstance(value, (bool, np.bool_))
             and isinstance(value, (int, float, np.integer, np.floating))
             and math.isfinite(float(value)), f"{name} requires a finite real scalar")
    return float(value)


def _array(value, shape, name):
    _require(isinstance(value, np.ndarray) and value.dtype == np.float64
             and value.shape == shape and bool(np.isfinite(value).all()),
             f"{name} requires finite float64 shape {shape}")
    return value


def _norm_limit(action, limit):
    magnitude = math.hypot(float(action[0]), float(action[1]))
    _require(math.isfinite(magnitude), "Waypoint command overflowed")
    saturated = magnitude > limit
    if saturated:
        action = action * (limit / magnitude)
    return np.asarray(action, dtype=np.float64), saturated


def _segment_enters_open_rectangle(start, end, half):
    """Slab intersection with the open conservative expanded AABB.

    Expanding each half extent by the disk radius contains the exact rounded
    Minkowski rectangle. Boundary tangency is allowed; an interior segment
    is rejected. This conservative check is for free-motion waypoints only.
    """
    low, high = 0.0, 1.0
    delta = end - start
    for axis in (0, 1):
        if abs(float(delta[axis])) <= 1e-15:
            if abs(float(start[axis])) >= half[axis] - 1e-12:
                return False
            continue
        first = (-half[axis] - start[axis]) / delta[axis]
        second = (half[axis] - start[axis]) / delta[axis]
        low = max(low, min(first, second))
        high = min(high, max(first, second))
        if high <= low + 1e-12:
            return False
    return high > max(low, 0.0) + 1e-12 and low < 1.0 - 1e-12


@dataclass(frozen=True)
class GeometryContext:
    mode: str = "RETRACT"
    locked_offset_m: float | None = None
    retreat_offset_m: float | None = None
    phase_started_s: float = 0.0
    last_elapsed_s: float = 0.0
    reassessment_count: int = 0
    saturation_count: int = 0


@dataclass(frozen=True)
class GeometryDecision:
    action: np.ndarray
    context: GeometryContext
    diagnostics: dict


class GeometryController:
    """A rear-face G proposal from state10, TaskSpec and its own context.

    All configured values are copied at construction. Mutating the caller's
    dictionary cannot alter a controller during an episode. The constructor
    deliberately reads neither friction, contact parameters nor simulator
    internals. ``elapsed_s`` is the measured episode time after reset settle.
    """

    def __init__(self, config):
        try:
            values = config["geometry_baseline"]
            nonnegative = {"waypoint_velocity_damping", "angular_damping_gain_m_s_rad"}
            self.parameters = {
                name: _number(values[name], name, zero=name in nonnegative)
                for name in PARAMETER_NAMES
            }
            size = np.asarray(config["object"]["size_full_xyz_m"], dtype=np.float64)
            _require(size.shape == (3,) and bool(np.isfinite(size).all())
                     and bool((size > 0).all()), "object size requires three positive lengths")
            self.half = size[:2].copy() / 2
            self.radius = _number(config["pusher"]["radius_m"], "pusher radius")
            self.physical_speed = _number(config["pusher"]["command_speed_norm_max_m_s"],
                                          "physical command norm limit")
            self.action_dt_s = _number(config["physics"]["action_dt_s"], "action interval")
            self.workspace = np.asarray([
                config["workspace"]["x_bounds_m"], config["workspace"]["y_bounds_m"]
            ], dtype=np.float64)
            _require(self.workspace.shape == (2, 2)
                     and bool(np.isfinite(self.workspace).all())
                     and bool((self.workspace[:, 1] > self.workspace[:, 0]).all()),
                     "workspace requires ordered finite x/y bounds")
            _require(tuple(config["schema"]["state_fields"]) == tuple(STATE_FIELDS),
                     "state field order differs from state10")
            _require(config["schema_version"] == "push-align-state-v1",
                     "G requires the registered state10 schema version")
            _require(config["schema"]["task_yaw_error"] == "symmetric_pi",
                     "G requires the pi-symmetric task yaw convention")
            _require(config["schema"]["action_semantics"]
                     == "constant_target_velocity_per_action_interval",
                     "G requires velocity commands constant over each action interval")
        except (KeyError, TypeError, AttributeError, OverflowError) as error:
            raise GeometryControlError(f"Missing or malformed G configuration: {error}") from error
        p = self.parameters
        _require(p["speed_max_m_s"] <= self.physical_speed,
                 "G speed exceeds the physical command norm limit")
        _require(p["approach_speed_max_m_s"] <= p["speed_max_m_s"]
                 and p["yaw_creep_speed_m_s"] <= p["speed_max_m_s"],
                 "Approach and yaw-creep speeds must not exceed G speed")
        _require(p["minimum_inward_direction_cosine"] <= 1,
                 "minimum inward cosine must lie in (0,1]")
        _require(p["approach_gap_m"] < p["near_contact_loss_gap_m"]
                 < p["retreat_gap_m"], "G gaps require approach < loss < retreat")
        _require(p["waypoint_tolerance_m"] < p["retreat_gap_m"],
                 "Waypoint tolerance must be smaller than retreat clearance")
        _require(p["contact_offset_abs_max_m"] + self.radius < self.half[1],
                 "Rear contact offset and pusher radius must fit the rear face")
        _require(p["contact_offset_replan_threshold_m"] <= 2*p["contact_offset_abs_max_m"],
                 "Offset replan threshold exceeds the entire feasible offset range")
        _require(bool((self.workspace[:, 1] - self.workspace[:, 0] > 2*self.radius).all()),
                 "Pusher cannot fit the workspace")
        self.half.flags.writeable = False
        self.workspace.flags.writeable = False

    def initial_context(self):
        return GeometryContext()

    def _context(self, context, elapsed):
        _require(isinstance(context, GeometryContext) and context.mode in MODES,
                 "context requires a known GeometryContext mode")
        first = _number(context.phase_started_s, "phase_started_s", zero=True)
        last = _number(context.last_elapsed_s, "last_elapsed_s", zero=True)
        _require(first <= last + 1e-9 and elapsed >= last - 1e-9,
                 "G context time must be ordered and episode time must be monotonic")
        for name in ("locked_offset_m", "retreat_offset_m"):
            if getattr(context, name) is not None:
                _finite(getattr(context, name), name)
        if context.locked_offset_m is not None:
            _require(abs(context.locked_offset_m)
                     <= self.parameters["contact_offset_abs_max_m"] + 1e-12,
                     "Locked contact offset exceeds its configured maximum")
        for name in ("reassessment_count", "saturation_count"):
            value = getattr(context, name)
            _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                     and value >= 0, f"{name} requires a nonnegative integer")
        _require(context.mode == "RETRACT" or
                 (context.locked_offset_m is not None and context.retreat_offset_m is not None),
                 "Active contact modes require locked and retreat offsets")

    def propose(self, state, task, context, elapsed_s):
        _array(state, (10,), "state")
        _require(isinstance(task, TaskSpec), "task requires TaskSpec")
        try:
            task.validate()
        except (ValueError, TypeError, AttributeError, OverflowError) as error:
            raise GeometryControlError(str(error)) from error
        elapsed = _number(elapsed_s, "elapsed_s", zero=True)
        self._context(context, elapsed)
        p = self.parameters
        with np.errstate(over="ignore", invalid="ignore"):
            goal_delta = task.goal[:2] - state[:2]
            relative = state[6:8] - state[:2]
        _require(bool(np.isfinite(goal_delta).all()) and bool(np.isfinite(relative).all()),
                 "Relative coordinates overflowed")
        position_error = math.hypot(*goal_delta)
        angle = float(state[2] - task.goal[2])
        _require(math.isfinite(angle) and math.isfinite(2*float(state[2]))
                 and math.isfinite(2*angle), "Yaw calculation overflowed")
        theta_bar = .5*math.atan2(math.sin(2*float(state[2])), math.cos(2*float(state[2])))
        yaw_error = .5*math.atan2(math.sin(2*angle), math.cos(2*angle))
        d = np.array([math.cos(theta_bar), math.sin(theta_bar)], np.float64)
        t = np.array([-d[1], d[0]], np.float64)
        local = np.array([relative @ d, relative @ t], np.float64)
        _require(bool(np.isfinite(local).all()) and math.isfinite(position_error),
                 "Body-frame coordinates overflowed")
        updated = replace(context, last_elapsed_s=elapsed)
        diagnostics = {
            "controller": "geometry_rear_face_v1", "feasible": True, "reason": "command",
            "mode_before": context.mode, "mode_after": context.mode,
            "transition_reason": "none", "transitions": [],
            "position_error_m": position_error, "yaw_error_rad": yaw_error,
            "theta_bar_rad": theta_bar, "normal_direction": d.tolist(),
            "tangent_direction": t.tolist(), "pusher_local_xy_m": local.tolist(),
            "b_target_m": None, "b_unclipped_m": None,
            "direction_unit": None, "direction_cosine": None,
            "waypoint_world_xy_m": None, "waypoint_local_xy_m": None,
            "path_checked": False, "path_safe": None,
            "path_geometry": "current_radius_expanded_conservative_AABB",
            "intentional_contact_motion": False, "raw_contact_labels_used": False,
            "future_physics_queries": 0, "saturated": False,
        }

        def finish(action, new_context, *, feasible=True, reason="command"):
            action = np.asarray(action, dtype=np.float64).copy()
            _array(action, (2,), "G action")
            action.flags.writeable = False
            diagnostics.update(feasible=feasible, reason=reason, mode_after=new_context.mode,
                               reassessment_count=new_context.reassessment_count,
                               saturation_count=new_context.saturation_count)
            return GeometryDecision(action, new_context, diagnostics)

        pose_inside = position_error <= task.position_tolerance_m \
            and abs(yaw_error) <= task.yaw_tolerance_rad
        if pose_inside:
            # Pose requests holding. Only the common physical supervisor can
            # declare ready. Clear stale approach/push state if holding breaks.
            cleared = replace(updated, mode="RETRACT", locked_offset_m=None,
                              retreat_offset_m=None, phase_started_s=elapsed)
            diagnostics["transition_reason"] = "pose_inside_common_holding"
            return finish(np.zeros(2, np.float64), cleared, reason="holding")
        yaw_only = position_error <= task.position_tolerance_m
        direction = d.copy() if yaw_only else goal_delta/position_error
        requested_push_speed = (p["yaw_creep_speed_m_s"] if yaw_only else
                                p["position_gain_per_s"]*position_error)
        _require(math.isfinite(requested_push_speed), "Proportional push speed overflowed")
        speed = min(requested_push_speed, p["speed_max_m_s"])
        inward = float(d @ direction)
        diagnostics.update(direction_unit=direction.tolist(), direction_cosine=inward,
                           yaw_only=yaw_only, push_speed_m_s=speed,
                           requested_push_speed_m_s=requested_push_speed)
        if inward < p["minimum_inward_direction_cosine"]:
            # The controller supports the rear face, not pulling or changing
            # the task goal. Recompute from the observed goal every call and
            # expose the unsupported direction rather than divide by it.
            return finish(np.zeros(2, np.float64), updated, feasible=False,
                          reason="controller_infeasible:insufficient_inward_goal_direction")
        b_raw = (p["yaw_offset_gain_m_rad"]*yaw_error
                 + p["angular_damping_gain_m_s_rad"]*float(state[5])
                 - self.half[0]*float(t @ direction)/inward)
        _require(math.isfinite(b_raw), "Contact-offset formula overflowed")
        b_target = float(np.clip(b_raw, -p["contact_offset_abs_max_m"],
                                 p["contact_offset_abs_max_m"]))
        offset_saturated = abs(b_raw) > p["contact_offset_abs_max_m"]
        diagnostics.update(b_target_m=b_target, b_unclipped_m=b_raw,
                           offset_saturated=offset_saturated)
        if offset_saturated:
            updated = replace(updated, saturation_count=updated.saturation_count+1)
        rear_boundary = -self.half[0] - self.radius
        retreat_x = rear_boundary - p["retreat_gap_m"]
        approach_x = rear_boundary - p["approach_gap_m"]
        gap = float(-local[0] - self.half[0] - self.radius)
        diagnostics["rear_face_gap_m"] = gap
        if local[0] > rear_boundary + p["near_contact_loss_gap_m"]:
            return finish(np.zeros(2, np.float64), updated, feasible=False,
                          reason="controller_infeasible:pusher_beyond_rear_local_recovery")

        def transition(mode, reason, *, relock=False):
            nonlocal updated
            diagnostics["transitions"].append({"from": updated.mode, "to": mode, "reason": reason})
            diagnostics["transition_reason"] = reason
            changes = {"mode": mode, "phase_started_s": elapsed}
            if relock:
                changes.update(locked_offset_m=b_target, retreat_offset_m=float(local[1]),
                               reassessment_count=updated.reassessment_count+1)
            updated = replace(updated, **changes)

        if updated.locked_offset_m is None or updated.retreat_offset_m is None:
            transition("RETRACT", "initialize_contact_offset", relock=True)
        elif elapsed - updated.phase_started_s >= p["contact_mode_max_elapsed_s"] - 1e-12:
            # A phase timeout prompts an observed-state contact re-evaluation,
            # not an artificial task failure or shortening of the 30 s budget.
            transition("RETRACT", "phase_time_reassessment", relock=True)
        elif updated.mode == "PUSH":
            contact_lost = gap > p["near_contact_loss_gap_m"] \
                or abs(local[1] - updated.locked_offset_m) > (
                    p["contact_offset_replan_threshold_m"] + p["waypoint_tolerance_m"])
            if abs(b_target - updated.locked_offset_m) > p["contact_offset_replan_threshold_m"]:
                transition("RETRACT", "contact_offset_reassessment", relock=True)
            elif contact_lost:
                transition("RETRACT", "geometric_contact_loss_estimate", relock=True)

        # Several already-reached waypoints may be crossed without spending
        # an artificial action interval at each bookkeeping phase.
        for _ in range(4):
            mode = updated.mode
            if mode == "RETRACT" and local[0] <= retreat_x + p["waypoint_tolerance_m"]:
                transition("SLIDE", "retreat_clearance_reached")
            elif mode == "SLIDE" and abs(local[1] - updated.locked_offset_m) <= p["waypoint_tolerance_m"]:
                transition("APPROACH", "tangent_offset_reached")
            elif mode == "APPROACH" and gap <= p["approach_gap_m"] + p["waypoint_tolerance_m"] \
                    and abs(local[1] - updated.locked_offset_m) <= p["waypoint_tolerance_m"]:
                transition("PUSH", "geometric_near_contact")
            else:
                break
        if updated.mode == "PUSH":
            command, speed_saturated = _norm_limit(speed*direction, p["speed_max_m_s"])
            speed_saturated = speed_saturated or requested_push_speed > p["speed_max_m_s"]
            diagnostics["intentional_contact_motion"] = True
        else:
            waypoint = np.array([
                approach_x if updated.mode == "APPROACH" else retreat_x,
                updated.retreat_offset_m if updated.mode == "RETRACT" else updated.locked_offset_m,
            ], np.float64)
            world = state[:2] + waypoint[0]*d + waypoint[1]*t
            expanded_half = self.half + self.radius
            intersects = _segment_enters_open_rectangle(local, waypoint, expanded_half)
            # An observed shallow contact is allowed to retreat *outward*;
            # it must preserve tangent offset rather than cut through a corner.
            outward_release = (updated.mode == "RETRACT"
                               and waypoint[0] <= local[0]
                               and abs(waypoint[1] - local[1]) <= 1e-12
                               and local[0] <= rear_boundary + p["near_contact_loss_gap_m"])
            safe = not intersects or outward_release
            inside_workspace = bool(((world >= self.workspace[:, 0]+self.radius)
                                     & (world <= self.workspace[:, 1]-self.radius)).all())
            diagnostics.update(waypoint_world_xy_m=world.tolist(),
                               waypoint_local_xy_m=waypoint.tolist(), path_checked=True,
                               path_safe=safe and inside_workspace,
                               outward_contact_release=outward_release,
                               waypoint_inside_workspace=inside_workspace)
            if not safe or not inside_workspace:
                return finish(np.zeros(2, np.float64), updated, feasible=False,
                              reason="controller_infeasible:unsafe_same_face_waypoint")
            with np.errstate(over="ignore", invalid="ignore"):
                raw = p["waypoint_gain_per_s"]*(world-state[6:8]) \
                    - p["waypoint_velocity_damping"]*state[8:10]
            _require(bool(np.isfinite(raw).all()), "Waypoint feedback overflowed")
            limit = p["approach_speed_max_m_s"] if updated.mode == "APPROACH" else p["speed_max_m_s"]
            command, speed_saturated = _norm_limit(raw, limit)
        diagnostics["speed_saturated"] = speed_saturated
        diagnostics["saturated"] = offset_saturated or speed_saturated
        if speed_saturated:
            updated = replace(updated, saturation_count=updated.saturation_count+1)
        return finish(command, updated)
