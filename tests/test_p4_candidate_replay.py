"""Actual fixed-command counterfactual replay and explicit failure controls."""
from copy import deepcopy
from dataclasses import replace
import math
import tomllib
import unittest
from unittest.mock import patch

import numpy as np

from push_align import InitialSpec, PlanarPushEnv
from push_align.candidate_replay import replay_candidate, snapshots_exact
from push_align.config import ROOT
from push_align.planning import PredictionBatch, TaskSpec, score


class PhysicalCandidateReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.physical = tomllib.loads((ROOT / "configs/pilot.v4.toml").read_text())
        cls.planner = deepcopy(cls.physical)
        cls.planner["planner"]["horizon"] = 5
        cls.planner["planner"]["command_speed_norm_max_m_s"] = .03
        cls.profile = tomllib.loads((ROOT / "configs/acceptance.v1.toml").read_text())
        cls.task = TaskSpec.from_config(cls.physical)

    def setUp(self):
        self.env = PlanarPushEnv(self.physical)
        self.env.reset()
        self.snapshot = self.env.snapshot()
        self.previous = np.zeros(2, np.float64)
        self.zero = np.zeros((5, 2), np.float64)

    def replay(self, actions=None, **changes):
        options = {"physical_config": self.physical, "planner_config": self.planner,
                   "snapshot": self.snapshot, "actions": self.zero if actions is None else actions,
                   "task": self.task, "previous_applied_action": self.previous,
                   "profile": self.profile}
        options.update(changes)
        return replay_candidate(**options)

    def test_five_zero_intervals_keep_actual_dt_shapes_and_complete_raw_samples(self):
        result = self.replay()
        self.assertTrue(result.valid)
        self.assertTrue(result.initial_snapshot_exact)
        self.assertEqual(result.states.shape, (6, 10))
        self.assertEqual(result.applied_actions.shape, (5, 2))
        self.assertEqual(result.states.dtype, np.float64)
        self.assertEqual(result.completed_intervals, 5)
        self.assertEqual(len(result.results), 5)
        self.assertEqual(result.reasons, ())
        self.assertTrue(np.array_equal(result.applied_actions, self.zero))
        self.assertTrue(math.isfinite(result.cost))
        for index, actual in enumerate(result.results):
            self.assertAlmostEqual(actual.actual_dt, .1, places=12)
            self.assertEqual(len(actual.diagnostics["samples"]), 201)
            self.assertEqual(actual.substep_states.shape, (200, 10))
            self.assertTrue(np.array_equal(actual.observation.state, result.states[index + 1]))
        self.assertAlmostEqual(result.results[-1].observation.sim_time -
                               result.results[0].diagnostics["samples"][0]["sim_time"], .5, places=12)

    def test_noncontact_commands_move_pusher_away_without_creating_side_contact(self):
        command = np.tile(np.array([-.03, 0.], np.float64), (5, 1))
        result = self.replay(command)
        self.assertTrue(result.valid)
        self.assertLess(result.states[-1, 6], result.states[0, 6] - .01)
        self.assertTrue(np.array_equal(result.applied_actions, command))
        self.assertFalse(any(contact["pair"] == "pusher_object"
                             for step in result.results for sample in step.diagnostics["samples"]
                             for contact in sample["contacts"]))

    def test_central_and_eccentric_contacts_keep_requested_sequence_and_record_rotation(self):
        command = np.tile(np.array([.03, 0.], np.float64), (5, 1))
        results = []
        for offset in (0., .015, -.015):
            env = PlanarPushEnv(self.physical)
            env.reset(InitialSpec(pusher_x=-.051, pusher_y=offset))
            result = self.replay(command, snapshot=env.snapshot())
            self.assertTrue(result.valid, result.reasons)
            self.assertTrue(np.array_equal(result.applied_actions, command))
            self.assertTrue(any(contact["pair"] == "pusher_object"
                                for step in result.results for sample in step.diagnostics["samples"]
                                for contact in sample["contacts"]))
            results.append(result)
        yaw_changes = [abs(result.states[-1, 2] - result.states[0, 2]) for result in results]
        self.assertGreater(yaw_changes[1], yaw_changes[0] + .005)
        self.assertGreater(yaw_changes[2], yaw_changes[0] + .005)

    def test_three_fresh_replays_are_bitwise_identical_for_zero_noncontact_and_contact(self):
        for command in (self.zero, np.tile([-.03, 0.], (5, 1)), np.tile([.03, 0.], (5, 1))):
            with self.subTest(command=command[0]):
                copies = [self.replay(command.astype(np.float64)) for _ in range(3)]
                for result in copies[1:]:
                    self.assertTrue(np.array_equal(copies[0].states, result.states))
                    self.assertTrue(np.array_equal(copies[0].applied_actions, result.applied_actions))
                    self.assertEqual(copies[0].cost, result.cost)
                    self.assertEqual(copies[0].cost_components, result.cost_components)
                    self.assertTrue(snapshots_exact(copies[0].final_snapshot, result.final_snapshot))
                    for a, b in zip(copies[0].results, result.results):
                        self.assertTrue(np.array_equal(a.substep_states, b.substep_states))
                        self.assertTrue(np.array_equal(a.substep_times, b.substep_times))
                        self.assertEqual(a.diagnostics, b.diagnostics)

    def test_cost_reuses_pure_planner_postaction_formula_and_components(self):
        previous = np.array([-.01, .005], np.float64)
        actions = np.tile([.025, .003], (5, 1)).astype(np.float64)
        result = self.replay(actions, previous_applied_action=previous)
        self.assertTrue(result.valid)
        recorded = PredictionBatch(result.states[None], np.diff(result.states, axis=0)[None],
                                   np.array([True]), ("",), np.array([-1], np.int64))
        expected = score(recorded, actions[None], self.task, previous, self.planner)
        self.assertEqual(result.cost, float(expected.total[0]))
        self.assertEqual(result.cost_components, {key: float(value[0]) for key, value in expected.components.items()})
        self.assertAlmostEqual(sum(result.cost_components.values()), result.cost, places=12)
        zero_previous = score(recorded, actions[None], self.task, np.zeros(2), self.planner)
        self.assertNotEqual(expected.components["stage_action_change"][0],
                            zero_previous.components["stage_action_change"][0])

    def test_task_pi_symmetry_is_retained_while_states_stay_observed(self):
        equivalent_task = replace(self.task, goal=self.task.goal + np.array([0., 0., math.pi]))
        original, equivalent = self.replay(), self.replay(task=equivalent_task)
        self.assertTrue(np.array_equal(original.states, equivalent.states))
        self.assertAlmostEqual(original.cost, equivalent.cost, places=9)

    def test_ready_pose_does_not_stop_or_replace_later_nonzero_commands(self):
        self.env.reset(InitialSpec(object_x=.08))
        command = np.zeros((5, 2), np.float64)
        command[3:] = [-.03, 0.]
        # Three zero intervals would qualify for the held goal. The diagnostic
        # must nevertheless apply the last two commands without supervision.
        with patch("push_align.task.consume_step", side_effect=AssertionError("task supervisor must not run")):
            result = self.replay(command, snapshot=self.env.snapshot())
        self.assertTrue(result.valid)
        self.assertEqual(result.completed_intervals, 5)
        self.assertTrue(np.array_equal(result.applied_actions, command))
        self.assertLess(result.states[-1, 6], result.states[3, 6] - .004)

    def test_no_model_or_geometric_controller_prediction_is_called(self):
        with patch("push_align.dynamics.predict_delta", side_effect=AssertionError("model not allowed")), \
                patch("push_align.planning.rollout", side_effect=AssertionError("model rollout not allowed")):
            result = self.replay()
        self.assertTrue(result.valid)

    def test_input_config_snapshot_actions_previous_profile_and_global_rng_are_unchanged(self):
        import torch
        config, planner, profile = deepcopy(self.physical), deepcopy(self.planner), deepcopy(self.profile)
        original_snapshot = deepcopy(self.snapshot)
        command = self.zero.copy(); previous = self.previous.copy()
        before_np, before_torch = np.random.get_state(), torch.get_rng_state().clone()
        source_env = self.env.snapshot()
        self.replay(command, physical_config=config, planner_config=planner,
                    profile=profile, previous_applied_action=previous)
        self.assertEqual(config, self.physical); self.assertEqual(planner, self.planner)
        self.assertEqual(profile, self.profile)
        self.assertTrue(np.array_equal(command, self.zero))
        self.assertTrue(np.array_equal(previous, self.previous))
        self.assertTrue(snapshots_exact(original_snapshot, self.snapshot))
        self.assertTrue(snapshots_exact(source_env, self.env.snapshot()))
        after_np = np.random.get_state()
        self.assertEqual(before_np[0], after_np[0]); self.assertTrue(np.array_equal(before_np[1], after_np[1]))
        self.assertEqual(before_np[2:], after_np[2:])
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))

    def test_prediction_penetration_guard_is_not_a_physical_failure_rule(self):
        # Calling the predicted-state checker on an actual endpoint is forbidden;
        # recorded geometry/physics guards above determine this classification.
        with patch("push_align.planning._state_reasons", side_effect=AssertionError("prediction validity is not physical validity")):
            result = self.replay(np.tile([.03, 0.], (5, 1)).astype(np.float64))
        self.assertTrue(result.valid)

    def test_geometry_failed_interval_is_retained_and_suffix_is_nan_with_inf_cost(self):
        from push_align.contact_geometry import check_contact_geometry
        calls = 0
        def geometry(samples, budget):
            nonlocal calls
            calls += 1
            result = check_contact_geometry(samples, budget)
            # First call validates an empty-schema budget before execution.
            if calls == 3:
                result.update({"pass": False, "reasons": ["injected_geometry_failure"]})
            return result
        with patch("push_align.candidate_replay.check_contact_geometry", side_effect=geometry):
            result = self.replay()
        self.assertFalse(result.valid)
        self.assertEqual(result.completed_intervals, 2)
        self.assertEqual(len(result.results), 2)
        self.assertTrue(np.isfinite(result.states[:3]).all())
        self.assertTrue(np.isnan(result.states[3:]).all())
        self.assertTrue(np.isfinite(result.applied_actions[:2]).all())
        self.assertTrue(np.isnan(result.applied_actions[2:]).all())
        self.assertTrue(math.isinf(result.cost))
        self.assertTrue(all(math.isnan(value) for value in result.cost_components.values()))
        self.assertIn("interval_2:injected_geometry_failure", result.reasons)

    def test_partial_physics_failure_keeps_failed_raw_result_without_retry(self):
        original_step = PlanarPushEnv.step
        calls = 0
        def partial(env, action, **kwargs):
            nonlocal calls
            calls += 1
            actual = original_step(env, action, **kwargs)
            if calls == 2:
                diagnostics = deepcopy(actual.diagnostics)
                diagnostics["validity_violations"] = ["injected_warning"]
                return replace(actual, actual_dt=.0005, transition_valid=False, diagnostics=diagnostics)
            return actual
        with patch.object(PlanarPushEnv, "step", partial):
            result = self.replay()
        self.assertEqual(calls, 2)
        self.assertFalse(result.valid)
        self.assertEqual(result.completed_intervals, 1)
        self.assertEqual(len(result.results), 2)
        self.assertEqual(result.results[-1].actual_dt, .0005)
        self.assertTrue(np.array_equal(result.states[2], result.results[-1].observation.state))
        self.assertTrue(np.isnan(result.states[3:]).all())
        self.assertIn("interval_2:injected_warning", result.reasons)
        self.assertIn("interval_2:partial_physical_interval", result.reasons)
        self.assertTrue(math.isinf(result.cost))

    def test_applied_action_mismatch_is_invalid_and_actual_command_is_preserved(self):
        original_step = PlanarPushEnv.step
        def changed(env, action, **kwargs):
            return replace(original_step(env, action, **kwargs), applied_action=np.array([.001, 0.]))
        with patch.object(PlanarPushEnv, "step", changed):
            result = self.replay()
        self.assertFalse(result.valid)
        self.assertTrue(np.array_equal(result.applied_actions[0], np.array([.001, 0.])))
        self.assertIn("interval_1:candidate_applied_action_mismatch", result.reasons)

    def test_wrong_environment_or_malformed_snapshot_is_infrastructure_error(self):
        for snapshot in (replace(self.snapshot, model_hash="wrong"),
                         replace(self.snapshot, environment_hash="wrong"),
                         replace(self.snapshot, integration_state=self.snapshot.integration_state.astype(np.float32)),
                         replace(self.snapshot, integration_state=self.snapshot.integration_state[:1]),
                         replace(self.snapshot, reference_height=float("nan")),
                         replace(self.snapshot, warning_counts=np.zeros(1, np.int32)),
                         replace(self.snapshot, needs_reset=True)):
            with self.subTest(snapshot=snapshot), self.assertRaises(ValueError):
                self.replay(snapshot=snapshot)

    def test_snapshot_restore_must_be_exact_for_every_field(self):
        original_snapshot = PlanarPushEnv.snapshot
        def wrong_rng(env):
            result = original_snapshot(env)
            state = deepcopy(result.rng_state)
            state["state"]["state"] += 1
            return replace(result, rng_state=state)
        with patch.object(PlanarPushEnv, "snapshot", wrong_rng):
            with self.assertRaisesRegex(ValueError, "exactly match"):
                self.replay()

    def test_action_dtype_shape_nonfinite_norm_and_horizon_are_rejected_before_physics(self):
        for actions in (self.zero.astype(np.float32), np.zeros((5, 3)), np.zeros((0, 2)),
                        np.zeros((4, 2)), np.full((5, 2), np.nan), np.tile([.04, 0.], (5, 1))):
            with self.subTest(shape=actions.shape), patch.object(PlanarPushEnv, "step", side_effect=AssertionError("must reject before stepping")), \
                    self.assertRaises(ValueError):
                self.replay(actions)

    def test_physics_geometry_and_previous_command_contract_mismatches_are_rejected(self):
        for key in ("physics", "object", "pusher", "workspace"):
            planner = deepcopy(self.planner)
            planner[key]["injected"] = "mismatch"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "contracts differ"):
                self.replay(planner_config=planner)
        with self.assertRaises(ValueError):
            self.replay(previous_applied_action=np.array([.08, 0.]))
        profile = deepcopy(self.profile); profile["physics_suite"]["check_contact_geometry"] = False
        with self.assertRaises(ValueError):
            self.replay(profile=profile)


if __name__ == "__main__":
    unittest.main()
