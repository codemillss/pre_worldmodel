"""Independent task-supervisor tests using recorded-state fixtures only.

These tests establish sampled holding and failure semantics. They do not show
robot alignment performance or continuous-time motion between physical samples.
"""
from dataclasses import FrozenInstanceError, replace
import math
from types import SimpleNamespace
import unittest

import numpy as np

from push_align.planning import TaskSpec
from push_align.task import (TaskError, consume_step, fail_task, handoff_metrics,
                             pose_in_goal, start_task)


FIELDS = ("obj_x", "obj_y", "obj_yaw", "obj_vx", "obj_vy", "obj_wz",
          "pusher_x", "pusher_y", "pusher_vx", "pusher_vy")
SCHEMA = "push-align-state-v1"
ACTION_DT = .1


def task(**changes):
    value = TaskSpec(np.array([.08, 0., 0.], np.float64), .005,
                     math.radians(3), .01, .05, .3, 30.)
    return replace(value, **changes)


def state(*, x=.08, y=0., yaw=0., vx=0., vy=0., wz=0.):
    return np.array([x, y, yaw, vx, vy, wz, .03, 0., 0., 0.], np.float64)


def observation(value=None, sim_time=10., *, valid=True, schema=SCHEMA):
    return SimpleNamespace(state=state() if value is None else value.copy(),
                           sim_time=sim_time, schema_version=schema,
                           observation_valid=valid)


def step(start_time, start_state=None, *, values=None, offsets=(.05, .1),
         actual_dt=None, valid=True, violations=()):
    first = state() if start_state is None else start_state.copy()
    if values is None:
        values = [first.copy() for _ in offsets]
    states = np.asarray(values, dtype=np.float64).reshape(-1, 10)
    times = np.asarray([start_time + dt for dt in offsets], np.float64)
    endpoint = states[-1] if len(states) else first
    end_time = float(times[-1]) if len(times) else float(start_time)
    samples = [{"state": first.tolist(), "sim_time": float(start_time)}]
    samples.extend({"state": value.tolist(), "sim_time": float(t)}
                   for value, t in zip(states, times))
    return SimpleNamespace(
        requested_action=np.zeros(2, np.float64),
        applied_action=np.zeros(2, np.float64),
        observation=observation(endpoint, end_time, valid=valid),
        actual_dt=(end_time-start_time if actual_dt is None else actual_dt),
        transition_valid=valid,
        diagnostics={"samples": samples, "validity_violations": list(violations),
                     "state_fields": FIELDS},
        substep_states=states, substep_times=times)


class P2TaskTest(unittest.TestCase):
    def assert_task_error(self, function, *args, **kwargs):
        with self.assertRaises(TaskError):
            function(*args, **kwargs)

    def test_status_is_frozen_and_initial_elapsed_excludes_settle(self):
        status = start_task(observation(sim_time=12.5), task(), ACTION_DT)
        self.assertEqual(status.initial_sim_time, 12.5)
        self.assertEqual(status.last_sim_time, 12.5)
        self.assertEqual(status.elapsed_s, 0.)
        self.assertEqual(status.held_time_s, 0.)
        self.assertEqual(status.phase, "holding")
        self.assertTrue(status.previous_qualifies)
        self.assertIsNone(status.terminal_reason)
        with self.assertRaises(FrozenInstanceError):
            status.phase = "ready"

    def test_initial_pose_outside_is_running(self):
        status = start_task(observation(state(x=0.)), task(), ACTION_DT)
        self.assertEqual(status.phase, "running")
        self.assertFalse(status.previous_qualifies)
        self.assertEqual(status.held_time_s, 0.)

    def test_initial_pose_inside_with_high_speed_holds_without_credit(self):
        for initial in (state(vx=.011), state(wz=-.051)):
            with self.subTest(initial=initial):
                status = start_task(observation(initial), task(), ACTION_DT)
                self.assertEqual(status.phase, "holding")
                self.assertFalse(status.previous_qualifies)
                self.assertEqual(status.held_time_s, 0.)

    def test_pi_symmetric_yaw_and_goal_relative_position(self):
        spec = task(goal=np.array([.04, -.02, .2], np.float64))
        value = state(x=.043, y=-.016, yaw=.2 + math.pi)
        metrics = handoff_metrics(value, spec)
        self.assertAlmostEqual(metrics["e_pos_m"], .005, places=15)
        self.assertAlmostEqual(metrics["e_yaw_rad"], 0., places=14)
        # Use the goal exactly to avoid a binary-rounding boundary in XY.
        self.assertTrue(pose_in_goal(state(x=.04, y=-.02,
                                           yaw=.2 + math.pi), spec))
        self.assertFalse(pose_in_goal(state(x=.04, y=-.02,
                                            yaw=.2 + math.pi/2), spec))

    def test_speed_is_xy_norm_and_absolute_angular_speed(self):
        result = handoff_metrics(state(vx=.006, vy=.008, wz=-.05), task())
        self.assertAlmostEqual(result["linear_speed_m_s"], .01)
        self.assertAlmostEqual(result["angular_speed_rad_s"], .05)
        self.assertTrue(result["qualifies"])
        high = handoff_metrics(state(vx=.008, vy=.008), task())
        self.assertTrue(high["pose_within"])
        self.assertFalse(high["qualifies"])

    def test_pose_tolerances_are_inclusive_and_speeds_separate(self):
        spec = task(goal=np.zeros(3, np.float64), yaw_tolerance_rad=.05)
        self.assertTrue(pose_in_goal(state(x=.003, y=.004, yaw=.05), spec))
        self.assertFalse(pose_in_goal(state(x=.003, y=.004001), spec))
        self.assertFalse(pose_in_goal(state(x=0., yaw=.050001), spec))
        self.assertTrue(pose_in_goal(state(x=0., vx=1.), spec))
        self.assertFalse(handoff_metrics(state(x=0., vx=1.), spec)["qualifies"])

    def test_three_full_intervals_become_ready_with_no_boundary_double_count(self):
        spec = task()
        status = start_task(observation(), spec, ACTION_DT)
        for expected in (.1, .2):
            status = consume_step(status, step(status.last_sim_time), spec, ACTION_DT)
            self.assertEqual(status.phase, "holding")
            self.assertAlmostEqual(status.held_time_s, expected, places=12)
        status = consume_step(status, step(status.last_sim_time), spec, ACTION_DT)
        self.assertEqual(status.phase, "ready")
        self.assertEqual(status.terminal_reason, "ready")
        self.assertAlmostEqual(status.held_time_s, .3, places=12)
        self.assertAlmostEqual(status.elapsed_s, .3, places=12)

    def test_first_entry_sample_does_not_credit_pre_entry_interval(self):
        spec = task()
        outside = state(x=.02)
        status = start_task(observation(outside), spec, ACTION_DT)
        result = step(10., outside, values=[state(), state()])
        updated = consume_step(status, result, spec, ACTION_DT)
        self.assertEqual(updated.phase, "holding")
        self.assertAlmostEqual(updated.held_time_s, .05, places=12)

    def test_interior_pose_exit_resets_hold_even_when_boundary_returns(self):
        spec = task()
        status = start_task(observation(), spec, ACTION_DT)
        status = consume_step(status, step(10.), spec, ACTION_DT)
        exited = step(status.last_sim_time,
                      values=[state(x=.02), state()])
        status = consume_step(status, exited, spec, ACTION_DT)
        self.assertEqual(status.phase, "holding")
        self.assertAlmostEqual(status.held_time_s, 0., places=12)
        status = consume_step(status, step(status.last_sim_time), spec, ACTION_DT)
        self.assertEqual(status.phase, "holding")
        self.assertAlmostEqual(status.held_time_s, .1, places=12)

    def test_interior_speed_exit_resets_hold_even_when_boundary_is_still(self):
        spec = task()
        status = start_task(observation(), spec, ACTION_DT)
        status = consume_step(status, step(10.), spec, ACTION_DT)
        result = step(status.last_sim_time, values=[state(vy=.011), state()])
        updated = consume_step(status, result, spec, ACTION_DT)
        self.assertEqual(updated.phase, "holding")
        self.assertAlmostEqual(updated.held_time_s, 0., places=12)

    def test_final_outside_boundary_returns_to_running(self):
        spec = task()
        status = start_task(observation(), spec, ACTION_DT)
        updated = consume_step(status,
                               step(10., values=[state(), state(x=.02)]),
                               spec, ACTION_DT)
        self.assertEqual(updated.phase, "running")
        self.assertEqual(updated.held_time_s, 0.)
        self.assertFalse(updated.previous_qualifies)

    def test_mid_interval_hold_threshold_does_not_hide_final_exit(self):
        spec = task()
        outside = state(x=.02)
        status = start_task(observation(outside), spec, ACTION_DT)
        status = consume_step(status, step(10., outside,
                                          values=[state(), state()]),
                              spec, ACTION_DT)
        for _ in range(2):
            status = consume_step(status, step(status.last_sim_time), spec, ACTION_DT)
        self.assertAlmostEqual(status.held_time_s, .25, places=12)
        # At +.05 it reaches .3; at +.1 it leaves the pose tolerance.
        updated = consume_step(status,
                               step(status.last_sim_time,
                                    values=[state(), state(x=.02)]),
                               spec, ACTION_DT)
        self.assertEqual(updated.phase, "running")
        self.assertIsNone(updated.terminal_reason)
        self.assertEqual(updated.held_time_s, 0.)

    def test_holding_uses_nonuniform_recorded_timestamps(self):
        spec = task()
        outside = state(x=.02)
        status = start_task(observation(outside), spec, ACTION_DT)
        updated = consume_step(status,
                               step(10., outside, values=[state(), state(), state()],
                                    offsets=(.02, .07, .1)), spec, ACTION_DT)
        self.assertAlmostEqual(updated.held_time_s, .08, places=12)

    def test_ready_at_deadline_outranks_timeout_when_final_interval_valid(self):
        spec = task(max_episode_time_s=.3)
        status = start_task(observation(), spec, ACTION_DT)
        for _ in range(3):
            status = consume_step(status, step(status.last_sim_time), spec, ACTION_DT)
        self.assertEqual((status.phase, status.terminal_reason), ("ready", "ready"))

    def test_nonintegral_action_deadline_rejected_before_episode(self):
        self.assert_task_error(start_task, observation(),
                               task(max_episode_time_s=.25), ACTION_DT)

    def test_reaching_goal_on_deadline_without_hold_is_timeout(self):
        spec = task(max_episode_time_s=.3)
        outside = state(x=.02)
        status = start_task(observation(outside), spec, ACTION_DT)
        for _ in range(2):
            status = consume_step(status, step(status.last_sim_time, outside),
                                  spec, ACTION_DT)
        status = consume_step(status, step(status.last_sim_time, outside,
                                          values=[state(), state()]),
                              spec, ACTION_DT)
        self.assertEqual((status.phase, status.terminal_reason), ("failed", "timeout"))
        self.assertAlmostEqual(status.held_time_s, .05, places=12)

    def test_timeout_counts_from_start_observation_not_absolute_sim_time(self):
        spec = task(max_episode_time_s=.2)
        outside = state(x=.02)
        status = start_task(observation(outside, 100.), spec, ACTION_DT)
        status = consume_step(status, step(100., outside), spec, ACTION_DT)
        self.assertEqual(status.phase, "running")
        status = consume_step(status, step(status.last_sim_time, outside), spec, ACTION_DT)
        self.assertEqual((status.phase, status.terminal_reason), ("failed", "timeout"))
        self.assertAlmostEqual(status.elapsed_s, .2, places=12)

    def test_physics_warning_outranks_ready_and_preserves_violations(self):
        spec = task(hold_time_s=.1)
        status = start_task(observation(), spec, ACTION_DT)
        invalid = step(10., valid=False, violations=("physics_warning",))
        updated = consume_step(status, invalid, spec, ACTION_DT)
        self.assertEqual((updated.phase, updated.terminal_reason), ("failed", "physics_error"))
        self.assertIn("physics_warning", updated.violations)

    def test_partial_interval_cannot_be_ready_even_if_claimed_valid(self):
        spec = task(hold_time_s=.05)
        status = start_task(observation(), spec, ACTION_DT)
        for valid in (True, False):
            with self.subTest(valid=valid):
                partial = step(10., offsets=(.025, .05), valid=valid)
                updated = consume_step(status, partial, spec, ACTION_DT)
                self.assertEqual(updated.phase, "failed")
                self.assertNotEqual(updated.terminal_reason, "ready")
                self.assertIn("partial_transition", updated.violations)

    def test_explicit_environment_violation_outranks_ready(self):
        spec = task(hold_time_s=.1)
        status = start_task(observation(), spec, ACTION_DT)
        invalid = step(10., valid=False,
                       violations=("object_corners_outside_workspace",))
        updated = consume_step(status, invalid, spec, ACTION_DT)
        self.assertEqual((updated.phase, updated.terminal_reason), ("failed", "workspace_exit"))
        self.assertIn("object_corners_outside_workspace", updated.violations)

    def test_domain_failure_mapping_and_partial_stoppage_reason(self):
        status = start_task(observation(), task(), ACTION_DT)
        for violation, expected in (
                ("pusher_extent_outside_workspace", "workspace_exit"),
                ("height_deviation", "planarity_violation"),
                ("tilt_exceeded", "planarity_violation"),
                ("partial_transition", "physics_error")):
            with self.subTest(violation=violation):
                result = step(10., offsets=(.025, .05), valid=False,
                              violations=(violation, "partial_transition"))
                updated = consume_step(status, result, task(), ACTION_DT)
                self.assertEqual((updated.phase, updated.terminal_reason), ("failed", expected))
                self.assertIn(violation, updated.violations)
                self.assertIn("partial_transition", updated.violations)

    def test_physics_error_outranks_simultaneous_domain_failure(self):
        status = start_task(observation(), task(), ACTION_DT)
        result = step(10., valid=False,
                      violations=("object_corners_outside_workspace", "physics_warning"))
        updated = consume_step(status, result, task(), ACTION_DT)
        self.assertEqual(updated.terminal_reason, "physics_error")
        self.assertEqual(set(updated.violations),
                         {"object_corners_outside_workspace", "physics_warning"})

    def test_invalid_observation_without_specific_violation_is_physics_error(self):
        spec = task()
        status = start_task(observation(), spec, ACTION_DT)
        result = step(10.)
        result.observation.observation_valid = False
        updated = consume_step(status, result, spec, ACTION_DT)
        self.assertEqual((updated.phase, updated.terminal_reason), ("failed", "physics_error"))

    def test_invalid_initial_observation_fails_before_any_hold_credit(self):
        for obs in (observation(valid=False), observation(state(x=float("nan")))):
            with self.subTest(obs=obs):
                status = start_task(obs, task(), ACTION_DT)
                self.assertEqual((status.phase, status.terminal_reason), ("failed", "physics_error"))
                self.assertEqual(status.held_time_s, 0.)

    def test_failure_transition_preserves_all_raw_violations(self):
        spec = task()
        status = start_task(observation(), spec, ACTION_DT)
        names = ("physics_warning", "partial_transition", "nonfinite_state")
        updated = consume_step(status, step(10., valid=False, violations=names),
                               spec, ACTION_DT)
        self.assertEqual(updated.phase, "failed")
        self.assertTrue(set(names).issubset(set(updated.violations)))

    def test_nonfinite_interior_sample_fails_even_if_endpoint_recovers(self):
        spec = task(hold_time_s=.1)
        status = start_task(observation(), spec, ACTION_DT)
        result = step(10., values=[state(vx=float("nan")), state()])
        updated = consume_step(status, result, spec, ACTION_DT)
        self.assertEqual((updated.phase, updated.terminal_reason), ("failed", "physics_error"))
        self.assertIn("nonfinite_state", updated.violations)
        self.assertEqual(updated.held_time_s, 0.)

    def test_zero_substep_invalid_interval_is_partial_failure(self):
        spec = task()
        status = start_task(observation(), spec, ACTION_DT)
        result = step(10., offsets=(), values=[], valid=False,
                      violations=("physics_warning",))
        updated = consume_step(status, result, spec, ACTION_DT)
        self.assertEqual((updated.phase, updated.terminal_reason), ("failed", "physics_error"))
        self.assertEqual(set(updated.violations), {"physics_warning", "partial_transition"})
        self.assertEqual(updated.last_sim_time, 10.)

    def test_validity_flag_does_not_hide_recorded_physics_failure(self):
        status = start_task(observation(), task(), ACTION_DT)
        result = step(10., valid=True, violations=("physics_warning",))
        updated = consume_step(status, result, task(), ACTION_DT)
        self.assertEqual(updated.terminal_reason, "physics_error")
        self.assertIn("physics_warning", updated.violations)

    def test_terminal_status_cannot_consume_another_interval(self):
        spec = task(hold_time_s=.1)
        ready = consume_step(start_task(observation(), spec, ACTION_DT),
                             step(10.), spec, ACTION_DT)
        failed = fail_task(start_task(observation(), spec, ACTION_DT), "planner_invalid")
        for status in (ready, failed):
            with self.subTest(status=status):
                self.assert_task_error(consume_step, status, step(status.last_sim_time),
                                       spec, ACTION_DT)

    def test_fail_task_records_reason_without_changing_clock_or_claiming_success(self):
        status = start_task(observation(), task(), ACTION_DT)
        failed = fail_task(status, "planner_invalid")
        self.assertEqual((failed.phase, failed.terminal_reason), ("failed", "planner_invalid"))
        self.assertEqual(failed.last_sim_time, status.last_sim_time)
        self.assertEqual(failed.initial_sim_time, status.initial_sim_time)
        self.assertEqual(failed.elapsed_s, 0.)
        self.assertEqual(status.phase, "holding")

    def test_post_step_geometry_failure_can_override_ready(self):
        spec = task(hold_time_s=.1)
        ready = consume_step(start_task(observation(), spec, ACTION_DT),
                             step(10.), spec, ACTION_DT)
        failed = fail_task(ready, "physics_error")
        self.assertEqual((failed.phase, failed.terminal_reason), ("failed", "physics_error"))
        self.assertEqual(failed.last_sim_time, ready.last_sim_time)
        self.assertEqual(ready.terminal_reason, "ready")

    def test_state_shape_and_dtype_are_explicit(self):
        invalid = (state().astype(np.float32), state().tolist(), state()[:9],
                   state().reshape(1, 10))
        for value in invalid:
            with self.subTest(value=value):
                self.assert_task_error(handoff_metrics, value, task())
                self.assert_task_error(pose_in_goal, value, task())
        for value in invalid:
            with self.subTest(initial_value=value):
                obs = observation()
                obs.state = value
                self.assert_task_error(start_task, obs, task(), ACTION_DT)

    def test_invalid_task_and_action_dt_rejected(self):
        for spec in (task(position_tolerance_m=0.), task(hold_time_s=float("nan")),
                     task(goal=np.zeros(2, np.float64))):
            with self.subTest(spec=spec):
                self.assert_task_error(start_task, observation(), spec, ACTION_DT)
        for action_dt in (0., -.1, float("nan"), float("inf"), True):
            with self.subTest(action_dt=action_dt):
                self.assert_task_error(start_task, observation(), task(), action_dt)

    def test_initial_timestamp_and_schema_rejected(self):
        for time in (-1., float("nan"), float("inf"), True):
            with self.subTest(sim_time=time):
                self.assert_task_error(start_task, observation(sim_time=time), task(), ACTION_DT)
        self.assert_task_error(start_task, observation(schema="unknown"), task(), ACTION_DT)

    def test_validity_flags_require_boolean_values(self):
        obs = observation()
        obs.observation_valid = 1
        self.assert_task_error(start_task, obs, task(), ACTION_DT)
        status = start_task(observation(), task(), ACTION_DT)
        for attribute in ("transition_valid", "observation_valid"):
            result = step(10.)
            if attribute == "transition_valid":
                result.transition_valid = 1
            else:
                result.observation.observation_valid = 1
            with self.subTest(attribute=attribute):
                self.assert_task_error(consume_step, status, result, task(), ACTION_DT)

    def test_previous_hold_qualification_mismatch_rejected(self):
        status = start_task(observation(), task(), ACTION_DT)
        result = step(10., state(vx=.02), values=[state(), state()])
        self.assert_task_error(consume_step, status, result, task(), ACTION_DT)

    def test_nonboundary_active_status_and_episode_overrun_rejected(self):
        status = start_task(observation(), task(), ACTION_DT)
        nonboundary = replace(status, last_sim_time=10.025)
        self.assert_task_error(consume_step, nonboundary, step(10.025), task(), ACTION_DT)
        spec = task(max_episode_time_s=.1)
        forged_at_end = replace(status, last_sim_time=10.1)
        self.assert_task_error(consume_step, forged_at_end, step(10.1), spec, ACTION_DT)

    def test_substep_dtype_shape_and_count_mismatch_rejected(self):
        status = start_task(observation(), task(), ACTION_DT)
        mutations = (
            lambda r: setattr(r, "substep_states", r.substep_states.astype(np.float32)),
            lambda r: setattr(r, "substep_times", r.substep_times.astype(np.float32)),
            lambda r: setattr(r, "substep_states", r.substep_states[:, :9]),
            lambda r: setattr(r, "substep_times", r.substep_times[:1]),
            lambda r: setattr(r, "substep_times", r.substep_times.reshape(-1, 1)),
        )
        for mutate in mutations:
            result = step(10.)
            mutate(result)
            with self.subTest(mutate=mutate):
                self.assert_task_error(consume_step, status, result, task(), ACTION_DT)

    def test_repeated_or_reversed_substep_timestamps_rejected(self):
        status = start_task(observation(), task(), ACTION_DT)
        for offsets in ((.1, .1), (.1, .05), (0., .1), (-.01, .1)):
            with self.subTest(offsets=offsets):
                self.assert_task_error(consume_step, status, step(10., offsets=offsets),
                                       task(), ACTION_DT)

    def test_interval_start_discontinuity_rejected(self):
        status = start_task(observation(), task(), ACTION_DT)
        self.assert_task_error(consume_step, status, step(10.001), task(), ACTION_DT)

    def test_actual_dt_and_observation_endpoint_mismatch_rejected(self):
        status = start_task(observation(), task(), ACTION_DT)
        wrong_dt = step(10., actual_dt=.2)
        self.assert_task_error(consume_step, status, wrong_dt, task(), ACTION_DT)
        for mutate in (lambda r: setattr(r.observation, "sim_time", 10.101),
                       lambda r: setattr(r.observation, "state", state(x=.081))):
            result = step(10.)
            mutate(result)
            with self.subTest(mutate=mutate):
                self.assert_task_error(consume_step, status, result, task(), ACTION_DT)

    def test_raw_sample_vs_substep_and_schema_mismatch_rejected(self):
        status = start_task(observation(), task(), ACTION_DT)
        mutations = (
            lambda r: r.diagnostics["samples"][1]["state"].__setitem__(0, .07),
            lambda r: r.diagnostics["samples"][1].__setitem__("sim_time", 10.051),
            lambda r: r.diagnostics.__setitem__("state_fields", FIELDS[::-1]),
            lambda r: r.diagnostics.__setitem__("samples", r.diagnostics["samples"][:-1]),
            lambda r: setattr(r.observation, "schema_version", "unknown"),
        )
        for mutate in mutations:
            result = step(10.)
            mutate(result)
            with self.subTest(mutate=mutate):
                self.assert_task_error(consume_step, status, result, task(), ACTION_DT)

    def test_tolerance_scalar_must_be_finite_nonnegative(self):
        status = start_task(observation(), task(), ACTION_DT)
        for tolerance in (-1., float("nan"), float("inf"), True):
            with self.subTest(tolerance=tolerance):
                self.assert_task_error(consume_step, status, step(10.), task(),
                                       ACTION_DT, time_tolerance_s=tolerance)


if __name__ == "__main__":
    unittest.main()
