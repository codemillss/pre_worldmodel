"""Scripted physical records only: no MuJoCo construction or source futures."""
from copy import deepcopy
import json
import tomllib
import unittest

import numpy as np

from push_align.config import ROOT, load_pilot_config
from push_align.environment import Observation, SnapshotBundle, StepResult
from push_align.followup_probe import run_followup_probe
from push_align.planning import TaskSpec
from push_align.representation import STATE_FIELDS
from push_align.task import TaskStatus


def state(x=.08, *, vx=0.):
    value = np.zeros(10, np.float64)
    value[0], value[3], value[6] = x, vx, x-.051
    return value


class ScriptedEnv:
    """Return supplied states with exact raw/substep consistency; no physics."""
    action_dt = .1

    def __init__(self, initial, time, script=(), geometry_invalid_at=None, physical_invalid_at=None):
        self.state, self.time = initial.copy(), time
        self.script = script
        self.geometry_invalid_at, self.physical_invalid_at = geometry_invalid_at, physical_invalid_at
        self.commands = []

    def observe(self):
        return Observation(self.state.copy(), self.time, "push-align-state-v1", True)

    def snapshot(self):
        return SnapshotBundle(np.r_[self.time, self.state], "synthetic-model", "synthetic-env",
                              .01, {"state": 17}, np.zeros(1, np.int64), np.zeros(1, np.int64), False)

    def restore(self, snapshot):
        self.time, self.state = float(snapshot.integration_state[0]), snapshot.integration_state[1:].copy()
        return self.observe()

    def step(self, action):
        index = len(self.commands)
        self.commands.append(action.copy())
        start, initial = self.time, self.state.copy()
        values = self.script[index] if index < len(self.script) else [initial.copy(), initial.copy()]
        states = np.asarray(values, np.float64)
        times = start+np.arange(1, len(states)+1, dtype=np.float64)*(.1/len(states))
        self.state, self.time = states[-1].copy(), float(times[-1])
        raw = [{"state": s.tolist(), "sim_time": float(t), "contacts": []}
               for s, t in zip(np.vstack([initial, states]), np.r_[start, times])]
        if index == self.geometry_invalid_at:
            raw[-1]["contacts"] = [{"pair": "pusher_object", "distance_m": -.1,
                                    "normal_force_N": 1., "normal_world": [1., 0., 0.],
                                    "normal_center_cosine": 1.}]
        valid = index != self.physical_invalid_at
        return StepResult(action.copy(), action.copy(), self.observe(), .1, valid,
                          {"samples": raw, "state_fields": STATE_FIELDS,
                           "validity_violations": [] if valid else ["height_deviation"]},
                          states, times)


class FollowupProbeTest(unittest.TestCase):
    def setUp(self):
        self.physical = load_pilot_config()
        self.planner = deepcopy(self.physical)
        override = json.loads((ROOT/"configs/p3_planning.v1.json").read_text())
        for key in ("planner", "cost", "task"):
            self.planner[key] = deepcopy(override[key])
        self.task = TaskSpec.from_config(self.planner)
        self.profile = tomllib.loads((ROOT/"configs/acceptance.v1.toml").read_text())
        self.actions = np.tile(np.array([.02, 0.], np.float64), (5, 1))
        self.previous = np.zeros(2, np.float64)

    def probe(self, *, panel="A", initial=None, time=10., elapsed=5., held=0., script=(),
              geometry_invalid_at=None, physical_invalid_at=None, status=None, **kwargs):
        initial = state() if initial is None else initial
        env = ScriptedEnv(initial, time, script, geometry_invalid_at, physical_invalid_at)
        qualifies = abs(initial[0]-.08) <= .005 and abs(initial[3]) <= .01
        pose = abs(initial[0]-.08) <= .005
        status = status or TaskStatus("holding" if pose else "running", time-elapsed,
                                      time, held, qualifies, None, ())
        out = run_followup_probe(self.physical, self.planner, env.snapshot(), self.task,
                                 status, self.actions, self.previous, self.profile,
                                 panel=panel, env_factory=lambda config: env, **kwargs)
        return out, env

    def test_A_freezes_first_ready_but_keeps_nominal_actions_and_tracks_broken_hold(self):
        # Existing .2 s hold reaches .3 at boundary1, then breaks at boundary2.
        out, env = self.probe(held=.2, script=[[state(), state()], [state(.06), state(.06)],
                                              [state(), state()], [state(), state()], [state(), state()]])
        self.assertEqual(len(env.commands), 5)
        self.assertTrue(all(np.array_equal(x, self.actions[i]) for i, x in enumerate(env.commands)))
        self.assertEqual(out.first_ready_interval, 1)
        self.assertAlmostEqual(out.task_statuses[-1].last_sim_time, 10.1)
        self.assertAlmostEqual(out.boundary_times[-1], 10.5)
        self.assertAlmostEqual(out.diagnostic_hold_s[2], 0.)
        self.assertAlmostEqual(out.diagnostic_hold_s[-1], .25)
        self.assertTrue(out.physical_valid and out.horizon_completed and out.ready_eligible)
        self.assertEqual(len(out.cost_components), 9)
        self.assertTrue(np.isfinite(out.cost))

    def test_B_zero_override_is_not_latched_and_tail_is_zero(self):
        # A return outside resumes the prescribed nominal action at boundary1.
        script = [[state(.06), state(.06)], [state(), state()], [state(.06), state(.06)]]
        out, env = self.probe(panel="B", script=script)
        np.testing.assert_array_equal(env.commands[0], np.zeros(2))
        np.testing.assert_array_equal(env.commands[1], self.actions[1])
        np.testing.assert_array_equal(env.commands[2], np.zeros(2))
        for command in env.commands[5:]:
            np.testing.assert_array_equal(command, np.zeros(2))
        self.assertEqual(out.supervisor_overrides[:3].tolist(), [True, False, True])
        self.assertIsNone(out.cost)
        self.assertEqual(out.cost_components, {})
        self.assertFalse(out.ready)

    def test_intermediate_velocity_break_resets_existing_hold(self):
        out, _ = self.probe(held=.2, script=[[state(vx=.011), state()]])
        self.assertEqual(out.first_ready_interval, 4)
        self.assertAlmostEqual(out.diagnostic_hold_s[1], 0.)
        self.assertEqual(out.qualification_traces[0].tolist(), [True, False, True])
        self.assertAlmostEqual(out.hold_traces[0][-1], 0.)

    def test_geometry_failure_has_priority_over_same_boundary_ready(self):
        for panel in ("A", "B"):
            with self.subTest(panel=panel):
                out, env = self.probe(panel=panel, held=.2, geometry_invalid_at=0)
                self.assertEqual(len(env.commands), 1)
                self.assertFalse(out.physical_valid)
                self.assertFalse(out.ready)
                self.assertIsNone(out.first_ready_interval)
                self.assertEqual(out.termination, "physical_invalid")
                self.assertIn("interval_1:contact_penetration_exceeded", out.reasons)
                self.assertEqual(len(out.results), 1)
                if panel == "A":
                    self.assertTrue(np.isinf(out.cost))

    def test_A_later_invalid_keeps_first_hit_but_ineligible_for_H5_comparison(self):
        out, env = self.probe(held=.2, physical_invalid_at=2)
        self.assertEqual(out.first_ready_interval, 1)
        self.assertTrue(out.ready)
        self.assertFalse(out.ready_eligible)
        self.assertFalse(out.physical_valid)
        self.assertEqual(len(out.results), 3)
        self.assertTrue(np.isnan(out.states[4:]).all())
        self.assertTrue(np.isinf(out.cost))

    def test_B_preserves_original_clock_and_stops_at_original_deadline(self):
        out, env = self.probe(panel="B", initial=state(.06), time=44.9, elapsed=29.9)
        self.assertEqual(len(env.commands), 1)
        self.assertEqual(out.termination, "timeout")
        self.assertFalse(out.ready)
        self.assertTrue(out.physical_valid)
        self.assertAlmostEqual(out.task_statuses[-1].elapsed_s, 30.)
        self.assertAlmostEqual(out.boundary_times[1], 45.)
        self.assertTrue(np.isnan(out.boundary_times[2:]).all())
        self.assertFalse(out.horizon_completed)

    def test_B_ready_in_tail_is_distinct_from_prefix(self):
        script = [[state(.06), state(.06)] for _ in range(4)]
        script += [[state(), state()], [state(), state()], [state(), state()], [state(), state()]]
        out, env = self.probe(panel="B", initial=state(.06), script=script)
        self.assertEqual(out.first_ready_interval, 8)
        self.assertTrue(out.ready and out.ready_eligible)
        self.assertFalse(out.ready_prefix)
        self.assertEqual(out.termination, "ready")
        self.assertEqual(len(env.commands), 8)

    def test_B_stops_at_ready_instead_of_executing_nominal_suffix(self):
        out, env = self.probe(panel="B", held=.2)
        self.assertEqual(len(env.commands), 1)
        self.assertTrue(out.ready_prefix)
        self.assertEqual(out.first_ready_interval, 1)
        self.assertTrue(np.isnan(out.applied_actions[1:]).all())
        self.assertFalse(out.horizon_completed)

    def test_clock_or_hold_mismatch_is_infrastructure_error_before_step(self):
        initial = state()
        env = ScriptedEnv(initial, 10.)
        for status in (TaskStatus("holding", 5., 9.9, 0., True, None, ()),
                       TaskStatus("holding", 5., 10., 0., False, None, ())):
            with self.subTest(status=status), self.assertRaises(ValueError):
                run_followup_probe(self.physical, self.planner, env.snapshot(), self.task,
                                   status, self.actions, self.previous, self.profile,
                                   env_factory=lambda config: env)
        self.assertEqual(len(env.commands), 0)

    def test_A_never_truncates_horizon_to_fit_deadline(self):
        with self.assertRaisesRegex(ValueError, "exceed"):
            self.probe(panel="A", time=39.8, elapsed=29.8)

    def test_malformed_raw_contact_is_infrastructure_error_not_valid_no_contact(self):
        env = ScriptedEnv(state(), 10.)
        original = env.step
        def malformed(action):
            result = original(action)
            result.diagnostics["samples"][-1].pop("contacts")
            return result
        env.step = malformed
        status = TaskStatus("holding", 5., 10., 0., True, None, ())
        retained = []
        with self.assertRaises(ValueError):
            run_followup_probe(self.physical, self.planner, env.snapshot(), self.task,
                               status, self.actions, self.previous, self.profile,
                               env_factory=lambda config: env,
                               on_step=lambda index, result, command: retained.append((index, result, command)))
        self.assertEqual(len(env.commands), 1)
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0][0], 0)
        self.assertNotIn("contacts", retained[0][1].diagnostics["samples"][-1])
        np.testing.assert_array_equal(retained[0][2], self.actions[0])


if __name__ == "__main__":
    unittest.main()
