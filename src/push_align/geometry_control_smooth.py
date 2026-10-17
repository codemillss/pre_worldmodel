"""Opt-in continuous commands around the legacy rear-face G transitions.

The legacy state machine, rear-face feasibility checks and sixteen G
parameters remain unchanged. This candidate tapers push/retraction commands
and gives SLIDE/APPROACH a common continuously varying free waypoint. The
extra segment checks concern a stationary rectangle at the *current* observed
pose and a velocity-command endpoint. They do not predict the executed servo
trajectory or moving-object collisions.
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from .geometry_control import (
    GeometryControlError, GeometryController, GeometryDecision, _array, _norm_limit, _number,
    _require, _segment_enters_open_rectangle,
)
from .planning import TaskSpec


TRANSITION_SCHEMA_VERSION = "geometry-transition-v1"
TRANSITION_SCHEMA_V2 = "geometry-transition-v2"
TRANSITION_PARAMETER_NAMES = (
    "offset_taper_start_m", "phase_end_taper_s", "phase_entry_ramp_s",
    "slide_blend_outer_tolerance_multiplier", "free_motion_speed_max_m_s",
)


def _smoothstep(value):
    clipped = min(1.0, max(0.0, float(value)))
    return clipped * clipped * (3.0 - 2.0 * clipped)


class SmoothGeometryController(GeometryController):
    """An explicitly selected transition candidate with the legacy G API.

    ``GeometryContext`` is unchanged. Its phase start is set by the legacy
    state machine, so every actual PUSH/RETRACT entry starts its ramp at zero.
    Construction copies all transition scalars; caller dictionary mutations
    cannot change an episode's controller.
    """

    def __init__(self, config, transition_parameters):
        super().__init__(config)
        _require(isinstance(transition_parameters, dict),
                 "Transition parameters require a dictionary")
        version = transition_parameters.get("schema_version")
        _require(version in (TRANSITION_SCHEMA_VERSION, TRANSITION_SCHEMA_V2),
                 "Unknown geometry transition schema")
        registered = {"schema_version", *TRANSITION_PARAMETER_NAMES}
        if version == TRANSITION_SCHEMA_V2:
            registered.add("clock_snap_tolerance_s")
        _require(set(transition_parameters) == registered,
                 "Transition parameters require the exact registered keys")
        self.transition_schema_version = version
        self.transition_parameters = {
            name: _number(transition_parameters[name], name)
            for name in TRANSITION_PARAMETER_NAMES
        }
        if version == TRANSITION_SCHEMA_V2:
            tolerance = _number(transition_parameters["clock_snap_tolerance_s"],
                                "clock_snap_tolerance_s")
            _require(tolerance <= self.action_dt_s * 1e-6,
                     "Scheduler clock tolerance exceeds one millionth of an action interval")
            self.transition_parameters["clock_snap_tolerance_s"] = tolerance
        p, q = self.parameters, self.transition_parameters
        _require(q["offset_taper_start_m"] < p["contact_offset_replan_threshold_m"],
                 "Offset taper must begin below the unchanged replan threshold")
        _require(q["phase_end_taper_s"] < p["contact_mode_max_elapsed_s"],
                 "Phase taper must be shorter than the unchanged phase budget")
        _require(q["phase_entry_ramp_s"] < p["contact_mode_max_elapsed_s"],
                 "Entry ramp must be shorter than the unchanged phase budget")
        _require(q["slide_blend_outer_tolerance_multiplier"] > 1,
                 "The outer slide blend tolerance must exceed the legacy inner tolerance")
        _require(q["free_motion_speed_max_m_s"] <= p["speed_max_m_s"],
                 "Free-motion speed must not exceed the unchanged G speed limit")

    def propose(self, state, task, context, elapsed_s):
        # Prepare a current outward route before the legacy path check. A
        # body translation/rotation may change local_y by tiny amounts while
        # RETRACT's entry ramp is zero. Keeping the old local_y would turn an
        # intended normal release into a segment with a tangent component.
        # Validate before deriving geometry or touching the supplied context;
        # this never rescues an infeasible result *after* its path check.
        _array(state, (10,), "state")
        _require(isinstance(task, TaskSpec), "task requires TaskSpec")
        try:
            task.validate()
        except (ValueError, TypeError, AttributeError, OverflowError) as error:
            raise GeometryControlError(str(error)) from error
        raw_elapsed = _number(elapsed_s, "elapsed_s", zero=True)
        self._context(context, raw_elapsed)
        # V2 corrects only floating-clock roundoff near the declared action
        # grid, after the original raw time/context checks. It does not
        # shorten physical intervals, shift observations, or grant holding
        # time. Non-grid times remain unchanged, including partial steps.
        elapsed = raw_elapsed
        prepared_context = context
        if self.transition_schema_version == TRANSITION_SCHEMA_V2:
            tolerance = self.transition_parameters["clock_snap_tolerance_s"]

            def scheduling_time(value):
                value = float(value)
                ratio = value / self.action_dt_s
                if not math.isfinite(ratio):
                    return value
                grid = round(ratio) * self.action_dt_s
                return grid if math.isfinite(grid) and abs(grid - value) <= tolerance else value

            elapsed = scheduling_time(raw_elapsed)
            prepared_context = replace(
                context,
                phase_started_s=scheduling_time(context.phase_started_s),
                last_elapsed_s=scheduling_time(context.last_elapsed_s),
            )
        clock_snapped = (elapsed != raw_elapsed
                         or prepared_context.phase_started_s != context.phase_started_s
                         or prepared_context.last_elapsed_s != context.last_elapsed_s)
        # Holding has priority over route preparation as well as over the
        # returned command. Use the same position and pi-symmetric yaw
        # predicate as the legacy controller, with its overflow checks.
        with np.errstate(over="ignore", invalid="ignore"):
            goal_delta = task.goal[:2] - state[:2]
        _require(bool(np.isfinite(goal_delta).all()), "Relative coordinates overflowed")
        position_error = math.hypot(*goal_delta)
        angle = float(state[2] - task.goal[2])
        _require(math.isfinite(position_error) and math.isfinite(angle)
                 and math.isfinite(2 * float(state[2])) and math.isfinite(2 * angle),
                 "Pose calculation overflowed")
        yaw_error = .5 * math.atan2(math.sin(2 * angle), math.cos(2 * angle))
        pose_inside = (position_error <= task.position_tolerance_m
                       and abs(yaw_error) <= task.yaw_tolerance_rad)
        retreat_before = context.retreat_offset_m
        reanchored = False
        reanchor_reason = "not_needed"
        if not pose_inside and context.mode == "RETRACT" and context.locked_offset_m is not None \
                and context.retreat_offset_m is not None:
            _require(math.isfinite(2 * float(state[2])), "Yaw calculation overflowed")
            theta = .5 * math.atan2(math.sin(2 * float(state[2])),
                                    math.cos(2 * float(state[2])))
            normal = np.array([math.cos(theta), math.sin(theta)], np.float64)
            tangent = np.array([-normal[1], normal[0]], np.float64)
            with np.errstate(over="ignore", invalid="ignore"):
                relative = state[6:8] - state[:2]
                observed_local = np.array([relative @ normal, relative @ tangent], np.float64)
            _require(bool(np.isfinite(observed_local).all()), "Body-frame coordinates overflowed")
            rear_boundary = -self.half[0] - self.radius
            if rear_boundary < observed_local[0] <= (
                    rear_boundary + self.parameters["near_contact_loss_gap_m"]):
                prepared_context = replace(prepared_context,
                                           retreat_offset_m=float(observed_local[1]))
                reanchored = context.retreat_offset_m != prepared_context.retreat_offset_m
                reanchor_reason = "current_observed_shallow_rear_normal_release"

        # State/task/time/context validation and all discrete transitions are
        # still repeated by the existing controller. Never turn its
        # infeasible result into a smoothed nonzero command.
        original = super().propose(state, task, prepared_context, elapsed)
        p, q = self.parameters, self.transition_parameters
        updated = original.context
        diagnostics = dict(original.diagnostics)
        phase_age = max(0.0, elapsed - updated.phase_started_s)
        diagnostics.update(
            controller="geometry_rear_face_continuous_transition_" + (
                "v2" if self.transition_schema_version == TRANSITION_SCHEMA_V2 else "v1"),
            legacy_controller=original.diagnostics["controller"],
            smoothing_schema_version=self.transition_schema_version,
            raw_elapsed_s=raw_elapsed,
            scheduling_elapsed_s=elapsed,
            clock_snapped=clock_snapped,
            clock_scheduling_only=True,
            clock_scope="G phase schedule only; observed state, physical time, validity and supervisor holding timer unchanged",
            context_phase_start_raw_s=context.phase_started_s,
            context_phase_start_scheduling_s=prepared_context.phase_started_s,
            context_last_elapsed_raw_s=context.last_elapsed_s,
            context_last_elapsed_scheduling_s=prepared_context.last_elapsed_s,
            command_before_smoothing=original.action.tolist(),
            retreat_route_reanchored=reanchored,
            retreat_route_reanchor_reason=reanchor_reason,
            retreat_offset_before_m=retreat_before,
            retreat_offset_after_m=prepared_context.retreat_offset_m,
            retreat_offset_returned_m=updated.retreat_offset_m,
            phase_age_s=phase_age,
            entry_ramp_gain=1.0,
            offset_taper_gain=1.0,
            phase_end_taper_gain=1.0,
            free_waypoint_blend_gain=None,
            base_push_gain=0.0,
            noise_gain=0.0,
            command_endpoint_checked=False,
            command_endpoint_path_safe=None,
            command_endpoint_inside_workspace=None,
            command_endpoint_world_xy_m=None,
            command_geometry="current_radius_expanded_conservative_AABB",
            smoothing_outward_tangent_projection=False,
            smoothing_uses_future_physics=False,
        )

        def finish(command, *, feasible=True, reason=None):
            command = np.asarray(command, dtype=np.float64).copy()
            _array(command, (2,), "Smoothed G action")
            command.flags.writeable = False
            diagnostics["command_after_smoothing"] = command.tolist()
            diagnostics["feasible"] = feasible
            if reason is not None:
                diagnostics["reason"] = reason
            return GeometryDecision(command, updated, diagnostics)

        if not diagnostics["feasible"] or diagnostics["reason"] == "holding":
            return finish(np.zeros(2, np.float64), feasible=diagnostics["feasible"])

        d = np.asarray(diagnostics["normal_direction"], np.float64)
        t = np.asarray(diagnostics["tangent_direction"], np.float64)
        local = np.asarray(diagnostics["pusher_local_xy_m"], np.float64)
        mode = updated.mode
        rear_boundary = -self.half[0] - self.radius
        retreat_x = rear_boundary - p["retreat_gap_m"]
        approach_x = rear_boundary - p["approach_gap_m"]
        expanded_half = self.half + self.radius
        waypoint = None
        outward_release = False

        if mode == "PUSH":
            offset_error = abs(float(diagnostics["b_target_m"]) - updated.locked_offset_m)
            offset_gain = 1.0 - _smoothstep(
                (offset_error - q["offset_taper_start_m"])
                / (p["contact_offset_replan_threshold_m"] - q["offset_taper_start_m"])
            )
            end_gain = 1.0 - _smoothstep(
                (phase_age - (p["contact_mode_max_elapsed_s"] - q["phase_end_taper_s"]))
                / q["phase_end_taper_s"]
            )
            entry_gain = _smoothstep(phase_age / q["phase_entry_ramp_s"])
            gain = entry_gain * offset_gain * end_gain
            diagnostics.update(entry_ramp_gain=entry_gain,
                               offset_taper_gain=offset_gain,
                               phase_end_taper_gain=end_gain,
                               contact_offset_error_m=offset_error,
                               base_push_gain=gain, noise_gain=gain)
            # This is still deliberate contact motion. A free-motion AABB
            # rejection would disallow the push itself, so actual execution
            # remains the physical supervisor's responsibility.
            return finish(original.action * gain)

        if mode == "RETRACT":
            entry_gain = _smoothstep(phase_age / q["phase_entry_ramp_s"])
            command = original.action * entry_gain
            waypoint = np.asarray(diagnostics["waypoint_local_xy_m"], np.float64).copy()
            if local[0] <= rear_boundary + p["near_contact_loss_gap_m"] \
                    and local[0] > rear_boundary:
                # During an observed shallow contact, release along the
                # current outward normal with no leftover tangent command.
                command = min(0.0, float(command @ d)) * d
                waypoint[1] = local[1]
                outward_release = waypoint[0] <= local[0]
                diagnostics["smoothing_outward_tangent_projection"] = True
            diagnostics["entry_ramp_gain"] = entry_gain
        else:
            # The legacy state machine permits SLIDE only after observed
            # RETRACT clearance. APPROACH keeps that cleared-route phase;
            # demanding retreat clearance again on every approach tick
            # would force a retreat/approach oscillation.
            _require(mode in {"SLIDE", "APPROACH"}, "Unknown active free-motion mode")
            tangent_error = abs(float(local[1]) - updated.locked_offset_m)
            tolerance = p["waypoint_tolerance_m"]
            outer = q["slide_blend_outer_tolerance_multiplier"] * tolerance
            blend = _smoothstep((tangent_error - tolerance) / (outer - tolerance))
            waypoint = np.array([
                approach_x + blend * (retreat_x - approach_x),
                updated.locked_offset_m,
            ], np.float64)
            world = state[:2] + waypoint[0] * d + waypoint[1] * t
            with np.errstate(over="ignore", invalid="ignore"):
                raw = p["waypoint_gain_per_s"] * (world - state[6:8]) \
                    - p["waypoint_velocity_damping"] * state[8:10]
            _require(bool(np.isfinite(raw).all()), "Continuous waypoint feedback overflowed")
            command, saturated = _norm_limit(raw, q["free_motion_speed_max_m_s"])
            diagnostics.update(free_waypoint_blend_gain=blend,
                               free_waypoint_tangent_error_m=tangent_error,
                               free_motion_speed_saturated=saturated)

        # Recheck the changed waypoint and its current-pose route. Keeping
        # the waypoint behind the rear plane is not enough if damping sends
        # the final velocity elsewhere, hence the second endpoint check.
        world_waypoint = state[:2] + waypoint[0] * d + waypoint[1] * t
        waypoint_inside = bool(((world_waypoint >= self.workspace[:, 0] + self.radius)
                                & (world_waypoint <= self.workspace[:, 1] - self.radius)).all())
        waypoint_safe = not _segment_enters_open_rectangle(local, waypoint, expanded_half) \
            or outward_release
        diagnostics.update(waypoint_world_xy_m=world_waypoint.tolist(),
                           waypoint_local_xy_m=waypoint.tolist(), path_checked=True,
                           path_safe=waypoint_safe and waypoint_inside,
                           waypoint_inside_workspace=waypoint_inside,
                           outward_contact_release=outward_release)
        if not waypoint_safe or not waypoint_inside:
            return finish(np.zeros(2, np.float64), feasible=False,
                          reason="controller_infeasible:unsafe_continuous_free_waypoint")

        with np.errstate(over="ignore", invalid="ignore"):
            endpoint = state[6:8] + self.action_dt_s * command
            endpoint_relative = endpoint - state[:2]
        _require(bool(np.isfinite(endpoint).all()) and bool(np.isfinite(endpoint_relative).all()),
                 "Continuous command endpoint overflowed")
        endpoint_local = np.array([endpoint_relative @ d, endpoint_relative @ t], np.float64)
        endpoint_inside = bool(((endpoint >= self.workspace[:, 0] + self.radius)
                                & (endpoint <= self.workspace[:, 1] - self.radius)).all())
        endpoint_outward = (mode == "RETRACT" and outward_release
                            and endpoint_local[0] <= local[0] + 1e-12
                            and abs(float(endpoint_local[1] - local[1])) <= 1e-12)
        endpoint_safe = not _segment_enters_open_rectangle(local, endpoint_local, expanded_half) \
            or endpoint_outward
        diagnostics.update(command_endpoint_checked=True,
                           command_endpoint_world_xy_m=endpoint.tolist(),
                           command_endpoint_local_xy_m=endpoint_local.tolist(),
                           command_endpoint_path_safe=endpoint_safe,
                           command_endpoint_inside_workspace=endpoint_inside,
                           command_endpoint_outward_release=endpoint_outward)
        if not endpoint_safe or not endpoint_inside:
            return finish(np.zeros(2, np.float64), feasible=False,
                          reason="controller_infeasible:unsafe_continuous_free_command")
        return finish(command)
