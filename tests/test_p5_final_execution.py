"""Fault checks for final execution; no simulator or learned model is run."""
from copy import deepcopy
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from push_align.config import ROOT, load_pilot_config
from push_align.environment import Observation, SnapshotBundle, StepResult
from push_align.final_execution import (FinalExecutionError, check_result_completeness,
                                       check_snapshot_completeness, execute_episode)
from push_align.planning import PlannerContext, TaskSpec
from push_align.planning_protocol import GeometryProposalAdapter
from push_align.representation import STATE_FIELDS
from push_align.task import start_task


class SyntheticEnv:
    """Only scripted observable samples; does not construct or step MuJoCo."""
    action_dt = .1

    def __init__(self, state, *, time=2., script=None, raise_at=None, malformed=False):
        self.state, self.time = state.copy(), time
        self.script, self.raise_at, self.malformed = script or [], raise_at, malformed
        self.calls = 0
        self.commands = []

    def observation(self):
        return Observation(self.state.copy(), self.time, "push-align-state-v1", True)

    def snapshot(self):
        return SnapshotBundle(np.r_[self.time, self.state], "model", "environment", .01,
                              {"state": 17}, np.zeros(1, np.int64), np.zeros(1, np.int64), False)

    def restore(self, snapshot):
        self.time, self.state = float(snapshot.integration_state[0]), snapshot.integration_state[1:].copy()
        return self.observation()

    def step(self, action):
        index = self.calls
        self.calls += 1
        self.commands.append(action.copy())
        if index == self.raise_at:
            raise RuntimeError("injected physical execution transport failure")
        before, start = self.state.copy(), self.time
        samples = self.script[index] if index < len(self.script) else [before.copy(), before.copy()]
        substate = np.asarray(samples, np.float64)
        times = start+np.arange(1, len(samples)+1, dtype=np.float64)*(.1/len(samples))
        self.state, self.time = substate[-1].copy(), float(times[-1])
        raw = [{"state": state.tolist(), "sim_time": float(t), "contacts": []}
               for state, t in zip(np.vstack([before, substate]), np.r_[start, times])]
        if self.malformed:
            raw[-1].pop("contacts")
        return StepResult(action.copy(), action.copy(), self.observation(), .1, True,
                          {"validity_violations": [], "state_fields": STATE_FIELDS, "samples": raw},
                          substate, times)


def state_at_goal(goal=False):
    state = np.zeros(10, np.float64)
    state[0] = .08 if goal else 0.
    state[6] = state[0]-.051
    return state


def stationary_predictor(state, action):
    return np.zeros_like(state)


class FinalExecutionTest(unittest.TestCase):
    def setUp(self):
        self.config = load_pilot_config()
        self.planner = deepcopy(self.config)
        self.planner["planner"].update(candidate_count=16, horizon=5, block_lengths=[2, 3])
        self.geometry = json.loads((ROOT/"configs/p3_geometry_selected.v2.json").read_text())
        self.transition = json.loads((ROOT/"configs/p3_geometry_transition.v2.json").read_text())

    def execute(self, env, **overrides):
        args = dict(case_id="known-development", episode_index=0, controller_id="test",
                    execution_id="synthetic-attempt", predictor=stationary_predictor,
                    physical_config=self.config, planner_config=self.planner,
                    geometry_parameters=self.geometry, transition_parameters=self.transition,
                    candidate_seed_base=4711, candidate_stream_id=0, max_intervals=2,
                    run_scope="development_preflight", penetration_budget_m=.0005)
        args.update(overrides)
        return execute_episode(env, env.snapshot(), **args)

    def test_all_invalid_is_unexecuted_task_failure_not_zero_command(self):
        env = SyntheticEnv(state_at_goal())
        output = self.execute(env, predictor=lambda state, action: np.full_like(state, np.nan))
        self.assertEqual(output.execution_status, "completed")
        self.assertEqual(output.termination, "planner_invalid")
        self.assertEqual(env.calls, 0)
        self.assertEqual(output.counts["physics_interval_attempts"], 0)
        self.assertEqual(output.unexecuted_plan["plan_identity"]["valid_predictions"], 0)
        self.assertFalse(output.unexecuted_plan["executed"])
        self.assertEqual(len(output.decisions), 1)
        self.assertEqual(sum(output.future_engine_calls.values()), 0)

    def test_some_invalid_candidates_are_retained_without_resampling(self):
        def mixed(state, action):
            output = np.zeros_like(state)
            output[action[:, 0] > .0001] = np.nan
            return output
        output = self.execute(SyntheticEnv(state_at_goal()), predictor=mixed, max_intervals=1)
        info = output.records[0]["plan_identity"]
        self.assertGreater(info["valid_predictions"], 0)
        self.assertLess(info["valid_predictions"], 16)
        self.assertTrue(info["invalid_reasons"])
        planned = output.plan_results[0]
        self.assertEqual(planned.candidates.actions.shape, (16, 5, 2))
        self.assertEqual(planned.predictions.valid.shape, (16,))
        self.assertTrue(np.isinf(planned.costs.total[~planned.predictions.valid]).all())
        self.assertTrue(planned.predictions.valid[planned.selected_index])
        self.assertTrue(np.isfinite(planned.costs.total[planned.selected_index]))
        self.assertEqual(output.termination, "development_budget")
        self.assertEqual(output.counts["physics_intervals_returned"], 1)

    def test_holding_uses_zero_and_actual_continuous_point_three_seconds(self):
        env = SyntheticEnv(state_at_goal(True))
        def forbidden_model(*args):
            raise AssertionError("Holding must not call predictor")
        output = self.execute(env, predictor=forbidden_model, max_intervals=10)
        self.assertEqual(output.termination, "ready")
        self.assertEqual(env.calls, 3)
        self.assertAlmostEqual(output.final_task.held_time_s, .3)
        self.assertTrue(all(record["holding"] for record in output.records))
        self.assertTrue(all(np.array_equal(action, np.zeros(2)) for action in env.commands))
        self.assertEqual(len(output.plan_results), 0)

    def test_substep_hold_break_is_not_hidden_by_good_action_boundaries(self):
        goal = state_at_goal(True)
        bad = goal.copy(); bad[4] = .02
        env = SyntheticEnv(goal, script=[[goal, goal], [bad, goal], [goal, goal], [goal, goal]])
        output = self.execute(env, max_intervals=4)
        self.assertEqual(output.termination, "development_budget")
        self.assertAlmostEqual(output.final_task.held_time_s, .2)
        self.assertEqual(env.calls, 4)
        self.assertTrue(all(record["holding"] for record in output.records))

    def test_timeout_is_original_task_time_and_no_post_terminal_step(self):
        env = SyntheticEnv(state_at_goal(), time=31.9)
        status = start_task(env.observation(), TaskSpec.from_config(self.config), .1)
        status = replace(status, initial_sim_time=2., last_sim_time=31.9)
        gctx = GeometryProposalAdapter(self.config, self.geometry, self.transition).initial_context()
        output = self.execute(env, max_intervals=10, initial_status=status,
                              initial_context=PlannerContext(np.zeros(2, np.float64), control_step=299), initial_gctx=gctx)
        self.assertEqual(output.termination, "timeout")
        self.assertEqual(env.calls, 1)
        self.assertEqual(output.records[0]["index"], 299)
        self.assertEqual(output.records[0]["rng_keys"], [4711, 0, 299, 0])

    def test_initial_terminal_context_executes_nothing(self):
        env = SyntheticEnv(state_at_goal(True), time=2.3)
        status = replace(start_task(env.observation(), TaskSpec.from_config(self.config), .1),
                         phase="ready", terminal_reason="ready", initial_sim_time=2., held_time_s=.3)
        gctx = GeometryProposalAdapter(self.config, self.geometry, self.transition).initial_context()
        output = self.execute(env, initial_status=status,
                              initial_context=PlannerContext(np.zeros(2, np.float64), control_step=3), initial_gctx=gctx)
        self.assertEqual(output.termination, "ready")
        self.assertEqual(env.calls, 0)
        self.assertEqual(output.decisions, [])

    def test_future_engine_call_is_blocked_before_actual_execution(self):
        def bad_predictor(*args):
            mujoco.mj_step(None, None)
        env = SyntheticEnv(state_at_goal())
        output = self.execute(env, predictor=bad_predictor)
        self.assertEqual(output.execution_status, "infrastructure_error")
        self.assertEqual(output.future_engine_calls["mj_step"], 1)
        self.assertEqual(env.calls, 0)
        self.assertEqual(output.error["type"], "AssertionError")

    def test_exception_after_prefix_preserves_returned_actual_raw(self):
        env = SyntheticEnv(state_at_goal(), raise_at=1)
        output = self.execute(env)
        self.assertEqual(output.execution_status, "infrastructure_error")
        self.assertEqual(len(output.results), 1)
        self.assertEqual(len(output.records), 1)
        self.assertEqual(output.counts["physics_interval_attempts"], 2)
        self.assertEqual(output.counts["physics_intervals_returned"], 1)
        self.assertFalse(output.decisions[-1]["executed"])

    def test_checker_exception_still_preserves_step_before_boundary_record(self):
        env = SyntheticEnv(state_at_goal(), malformed=True)
        output = self.execute(env)
        self.assertEqual(output.execution_status, "infrastructure_error")
        self.assertEqual(len(output.results), 1)
        self.assertEqual(len(output.records), 0)
        self.assertEqual(output.counts["physics_intervals_returned"], 1)

    def test_final_budget_without_terminal_is_incomplete_not_timeout(self):
        output = self.execute(SyntheticEnv(state_at_goal()), run_scope="final_evaluation", max_intervals=1)
        self.assertEqual(output.execution_status, "incomplete")
        self.assertEqual(output.termination, "missing_terminal_outcome")
        self.assertIsNone(output.final_task.terminal_reason)

    def test_partial_resume_is_rejected_before_restore(self):
        env = SyntheticEnv(state_at_goal())
        with patch.object(env, "restore", wraps=env.restore) as restore:
            with self.assertRaises(FinalExecutionError):
                self.execute(env, initial_context=PlannerContext.initial())
            restore.assert_not_called()

    def test_G_context_tracks_each_actual_tick_without_double_increment(self):
        env = SyntheticEnv(state_at_goal())
        output = self.execute(env, predictor=None, max_intervals=2)
        self.assertEqual(output.final_context.control_step, 2)
        self.assertEqual([record["planner_context_after"]["control_step"] for record in output.records], [1, 2])

    def test_bad_resume_tick_and_qualification_fail_before_physical_step(self):
        for fault in ("tick", "qualification"):
            with self.subTest(fault=fault):
                env = SyntheticEnv(state_at_goal(), time=2.1)
                status = replace(start_task(env.observation(), TaskSpec.from_config(self.config), .1), initial_sim_time=2.)
                if fault == "qualification":
                    status = replace(status, previous_qualifies=True)
                context = PlannerContext(np.zeros(2, np.float64), control_step=0 if fault == "tick" else 1)
                gctx = GeometryProposalAdapter(self.config, self.geometry, self.transition).initial_context()
                output = self.execute(env, initial_status=status, initial_context=context, initial_gctx=gctx)
                self.assertEqual(output.execution_status, "infrastructure_error")
                self.assertEqual(env.calls, 0)

    def test_physics_invalid_outranks_ready_and_stops_next_interval(self):
        env = SyntheticEnv(state_at_goal(True))
        original = env.step
        def invalid(action):
            result = original(action)
            diagnostics = deepcopy(result.diagnostics)
            diagnostics["validity_violations"] = ["physics_warning"]
            return replace(result, transition_valid=False, diagnostics=diagnostics)
        with patch.object(env, "step", side_effect=invalid):
            output = self.execute(env, max_intervals=10)
        self.assertEqual(output.termination, "physics_error")
        self.assertEqual(env.calls, 1)

    def test_actual_start_capture_does_not_invent_unavailable_tick_29(self):
        env = SyntheticEnv(state_at_goal())
        output = self.execute(env, max_intervals=2, capture_ticks=(0, 29))
        self.assertEqual(list(output.captured_inputs), [0])
        captured = output.captured_inputs[0]
        np.testing.assert_array_equal(captured["snapshot"].integration_state, np.r_[2., state_at_goal()])
        self.assertEqual(captured["task_status"].elapsed_s, 0.)
        self.assertEqual(captured["planner_context"].control_step, 0)
        result = check_snapshot_completeness((0, 29), output.captured_inputs)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["missing"], [29])
        self.assertFalse(result["substitution_permitted"])

    def test_tick_29_positive_is_captured_at_actual_resumed_predecision(self):
        env = SyntheticEnv(state_at_goal(), time=4.9)
        status = replace(start_task(env.observation(), TaskSpec.from_config(self.config), .1), initial_sim_time=2.)
        context = PlannerContext(np.zeros(2, np.float64), control_step=29)
        gctx = GeometryProposalAdapter(self.config, self.geometry, self.transition).initial_context()
        output = self.execute(env, max_intervals=1, initial_status=status,
                              initial_context=context, initial_gctx=gctx, capture_ticks=(29,))
        self.assertEqual(output.termination, "development_budget")
        self.assertEqual(output.counts["physics_intervals_returned"], 1)
        captured = output.captured_inputs[29]
        self.assertEqual(captured["observation"].sim_time, 4.9)
        self.assertEqual(captured["planner_context"].control_step, 29)
        self.assertAlmostEqual(captured["task_status"].elapsed_s, 2.9)
        self.assertEqual(check_snapshot_completeness((29,), output.captured_inputs)["status"], "PASS_COMPLETE")

    def test_early_terminal_does_not_capture_future_requested_slots(self):
        output = self.execute(SyntheticEnv(state_at_goal(True)), max_intervals=10,
                              capture_ticks=(0, 1, 2, 3, 29))
        self.assertEqual(output.termination, "ready")
        self.assertEqual(list(output.captured_inputs), [0, 1, 2])
        result = check_snapshot_completeness((0, 1, 2, 3, 29), output.captured_inputs)
        self.assertEqual(result["missing"], [3, 29])

    def test_capture_checker_rejects_extra_or_generic_unbound_slots(self):
        output = self.execute(SyntheticEnv(state_at_goal()), max_intervals=2, capture_ticks=(0, 1))
        extra = check_snapshot_completeness((0,), output.captured_inputs)
        self.assertEqual(extra["status"], "INCOMPLETE")
        self.assertEqual(extra["extra"], [1])
        invalid = check_snapshot_completeness((29,), {29: {"tick": 29}})
        self.assertEqual(invalid["invalid_ticks"], [29])
        self.assertEqual(invalid["status"], "INCOMPLETE")


class CompletenessTest(unittest.TestCase):
    def row(self, name, **changes):
        return {"cell_id": name, "execution_status": "completed", "termination": "planner_invalid",
                "run_scope": "final_evaluation", **changes}

    def test_completed_task_failures_count_in_full_declared_denominator(self):
        result = check_result_completeness(["A", "B"], [self.row("A"), self.row("B", termination="timeout")])
        self.assertEqual(result["status"], "PASS_COMPLETE")
        self.assertEqual(result["denominator"], ["A", "B"])

    def test_missing_duplicate_extra_and_reordered_results_are_detected(self):
        cases = [([self.row("A")], "missing"),
                 ([self.row("A"), self.row("A")], "duplicate"),
                 ([self.row("A"), self.row("B"), self.row("C")], "unexpected"),
                 ([self.row("B"), self.row("A")], "order_matches")]
        for rows, key in cases:
            with self.subTest(key=key):
                result = check_result_completeness(["A", "B"], rows)
                self.assertEqual(result["status"], "INCOMPLETE")
                if key == "order_matches":
                    self.assertFalse(result[key])
                else:
                    self.assertTrue(result[key])

    def test_development_budget_cannot_complete_final_scope(self):
        final_budget = self.row("A", termination="development_budget")
        self.assertEqual(check_result_completeness(["A"], [final_budget])["status"], "INCOMPLETE")
        dev = self.row("A", termination="development_budget", run_scope="development_preflight")
        self.assertEqual(check_result_completeness(["A"], [dev],
                         expected_run_scope="development_preflight")["status"], "PASS_COMPLETE")
        wrong_scope = check_result_completeness(["A"], [dev], expected_run_scope="final_evaluation")
        self.assertEqual(wrong_scope["status"], "INCOMPLETE")
        self.assertEqual(wrong_scope["scope_mismatch_rows"], ["A"])

    def test_unknown_terminal_does_not_pass_a_complete_result_table(self):
        self.assertEqual(check_result_completeness(["A"], [self.row("A", termination="arbitrary")])["status"],
                         "INCOMPLETE")

    def test_infrastructure_or_missing_terminal_is_present_but_incomplete(self):
        for changes in ({"execution_status": "infrastructure_error"}, {"termination": None},
                        {"execution_status": "incomplete", "termination": "missing_terminal_outcome"}):
            with self.subTest(changes=changes):
                result = check_result_completeness(["A"], [self.row("A", **changes)])
                self.assertEqual(result["attempted_rows"], 1)
                self.assertEqual(result["missing"], [])
                self.assertEqual(result["incomplete_rows"], ["A"])
                self.assertEqual(result["status"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
