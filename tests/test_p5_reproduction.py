"""Synthetic witness validation: no model or physics execution in these tests."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from push_align.planning import TaskSpec
from push_align.reproduction import (ENGINE_GUARDS, build_witness, compare_witness,
                                     json_safe, raw_reference, value_hash)


class ReproductionWitnessTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name)
        self.task = TaskSpec(np.array([.08, 0., 0.]), .005, .05, .002, .05, .3, 30.)
        self.profile = {"controller_replay": {"tolerance_fraction_of_task_tolerance": .001,
                                             "require_matching_terminal_type": True}}
        self.state = np.zeros(10, np.float64)
        self.state[6] = -.06
        self.context = {"previous_applied_action": [0., 0.], "optimal_actions": None, "control_step": 0}
        self.gctx = {"mode": "RETRACT", "last_elapsed_s": 0.}
        self.initial_task = {"phase": "running", "initial_sim_time": .1, "last_sim_time": .1,
                             "held_time_s": 0., "previous_qualifies": False,
                             "terminal_reason": None, "violations": []}
        self.final_task = {**self.initial_task, "last_sim_time": .2}
        self.plan = {"candidate_hash": "1"*64, "prediction_hash": "2"*64, "choice_hash": "3"*64}
        self.record = {"index": 29, "local_index": 0, "source": "model",
            "state_before": self.state.tolist(), "state_after": self.state.tolist(),
            "sim_time_before": .1, "sim_time_after": .2, "requested_action": [0., 0.],
            "applied_action": [0., 0.], "actual_dt_s": .1, "holding": False,
            "task_before": deepcopy(self.initial_task), "task_after": deepcopy(self.final_task),
            "geometry": {"pass": True, "reasons": []}, "observation_valid": True,
            "transition_valid": True, "planner_context_before": deepcopy(self.context),
            "planner_context_after": deepcopy(self.context), "geometry_context_before": deepcopy(self.gctx),
            "geometry_context_after": deepcopy(self.gctx), "plan_identity": deepcopy(self.plan),
            "rng_keys": [1, 0, 29, 0]}
        self.reference_raw = self.raw("reference", 1)
        self.fresh_raw = self.raw("fresh", 1)

    def raw(self, name, count):
        # Pure checker fixtures, not claims that these are physical raw traces.
        files = [self.path/f"{name}.npz", self.path/f"{name}.json.gz"]
        for path in files:
            path.write_bytes(f"synthetic hash fixture {name}".encode())
        return raw_reference(files, count)

    def execution(self, name):
        decision = {"index": self.record["index"], "local_index": 0, "source": "model",
                    "state": self.state.tolist(), "sim_time": .1, "holding": False,
                    "task_status": deepcopy(self.initial_task), "planner_context_before": deepcopy(self.context),
                    "geometry_context_before": deepcopy(self.gctx), "plan_identity": deepcopy(self.plan),
                    "rng_keys": [1, 0, 29, 0], "executed": True}
        return SimpleNamespace(execution_id=name, controller_id="S1-seed11", case_id="existing-dev",
            run_scope="development_preflight", origin={"kind": "existing_development_snapshot", "tick": 29},
            task_spec=self.task, initial_snapshot={"integration": [0., 1.]},
            initial_full_snapshot_equal=True,
            initial_observation=SimpleNamespace(state=self.state.copy(), sim_time=.1,
                observation_valid=True, schema_version="push-align-state-v1"),
            initial_task=deepcopy(self.initial_task), final_task=deepcopy(self.final_task),
            initial_context=deepcopy(self.context), initial_gctx=deepcopy(self.gctx),
            records=[deepcopy(self.record)], results=[object()],
            decisions=[decision],
            unexecuted_plan=None, execution_status="completed", termination="development_budget", error=None,
            counts={"controller_decisions": 1, "physics_interval_attempts": 1,
                    "physics_intervals_returned": 1, "completed_boundary_records": 1},
            future_engine_calls=dict.fromkeys(ENGINE_GUARDS, 0))

    def witnesses(self, fresh_raw=None):
        options = {"identity": {"case": "existing-dev", "controller": "S1-seed11"},
                   "source_bindings": {"synthetic-source.py": "a"*64}}
        a = build_witness(self.execution("original-execution"), **options,
                          raw_reference=self.reference_raw, attempt_id="reference-attempt")
        b = build_witness(self.execution("fresh-execution"), **options,
                          raw_reference=fresh_raw, original_raw_reference=self.reference_raw,
                          attempt_id="fresh-attempt")
        return a, b

    def compare(self, a, b):
        return compare_witness(a, b, self.task, self.profile)

    def test_fresh_bounded_development_rerun_can_reference_original_raw(self):
        a, b = self.witnesses()
        result = self.compare(a, b)
        self.assertEqual(result["status"], "PASS_REPRODUCTION")
        self.assertFalse(result["fresh_full_raw_prefix_required"])
        self.assertFalse(result["fresh_full_raw_prefix_present_and_verified"])
        self.assertEqual(result["maximum_differences"]["state_components"], [0.]*10)
        self.assertEqual(result["numeric_budgets"]["position_m"], 5e-6)
        self.assertEqual(a["initial"]["planner_context"], self.context)
        self.assertEqual(a["intervals"][0]["index"], 29)

    def test_serialization_roundtrip_keeps_hash_and_nonfinite_tokens(self):
        a, b = self.witnesses()
        self.assertEqual(value_hash(a), value_hash(json.loads(json.dumps(a))))
        safe = json_safe(np.array([np.nan, np.inf, -np.inf]))
        self.assertEqual(safe, ["NaN", "Infinity", "-Infinity"])
        json.dumps(safe, allow_nan=False)
        before_a, before_b = deepcopy(a), deepcopy(b)
        self.compare(a, b)
        self.assertEqual(a, before_a); self.assertEqual(b, before_b)

    def test_copied_execution_or_attempt_identity_is_not_fresh_evidence(self):
        for key in ("execution_id", "attempt_id"):
            with self.subTest(key=key):
                a, b = self.witnesses(self.fresh_raw)
                b[key] = a[key]
                result = self.compare(a, b)
                self.assertEqual(result["status"], "INCOMPLETE")
                self.assertIn("fresh:independent_execution_identity_required", result["incomplete_reasons"])

    def test_numerical_state_budget_is_observed_not_hash_equality(self):
        a, b = self.witnesses(self.fresh_raw)
        b["intervals"][0]["state_after"][0] += 4e-6
        self.assertEqual(self.compare(a, b)["status"], "PASS_REPRODUCTION")
        b["intervals"][0]["state_after"][0] += 2e-6
        result = self.compare(a, b)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("numeric_budget_exceeded:position_m", result["mismatch_reasons"])

    def test_two_dimensional_position_error_uses_norm_not_only_components(self):
        a, b = self.witnesses(self.fresh_raw)
        b["intervals"][0]["state_after"][0] += 4e-6
        b["intervals"][0]["state_after"][1] += 4e-6
        result = self.compare(a, b)
        self.assertEqual(result["status"], "FAIL")
        self.assertGreater(result["maximum_differences"]["position_m"], 5e-6)

    def test_action_and_time_budget_are_checked(self):
        for key, difference in (("requested_action", 3e-6), ("applied_action", 3e-6)):
            a, b = self.witnesses(self.fresh_raw)
            b["intervals"][0][key][0] = difference
            result = self.compare(a, b)
            self.assertIn(f"numeric_budget_exceeded:{key}_components", result["mismatch_reasons"])
        for epsilon, expected in ((.5e-9, "PASS_REPRODUCTION"), (2e-9, "FAIL")):
            a, b = self.witnesses(self.fresh_raw)
            row = b["intervals"][0]
            row["sim_time_after"] += epsilon
            row["actual_dt_s"] += epsilon
            row["task_after"]["last_sim_time"] += epsilon
            b["final_task"]["last_sim_time"] += epsilon
            self.assertEqual(self.compare(a, b)["status"], expected)

    def test_holding_requires_actual_zero_command_and_no_plan(self):
        a, b = self.witnesses(self.fresh_raw)
        for w in (a, b):
            row = w["intervals"][0]
            row.update(source="holding", holding=True, plan_identity=None)
            w["decisions"][0].update(source="holding", holding=True, plan_identity=None)
        self.assertEqual(self.compare(a, b)["status"], "PASS_REPRODUCTION")
        b["intervals"][0]["applied_action"][0] = 1e-9
        result = self.compare(a, b)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertIn("fresh:nonzero_or_inconsistent_holding:0", result["incomplete_reasons"])

    def test_hash_mismatch_requires_full_new_prefix_and_preserves_failure_reason(self):
        a, b = self.witnesses()
        b["intervals"][0]["plan_identity"]["prediction_hash"] = "f"*64
        b["decisions"][0]["plan_identity"]["prediction_hash"] = "f"*64
        absent = self.compare(a, b)
        self.assertEqual(absent["status"], "INCOMPLETE")
        self.assertTrue(absent["mismatch_reasons"])
        self.assertIn("fresh_prefix:missing_raw_reference", absent["incomplete_reasons"])
        b["raw_reference"] = self.fresh_raw
        result = self.compare(a, b)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(result["fresh_full_raw_prefix_present_and_verified"])

    def test_missing_interval_returned_result_and_missing_boundary_are_incomplete(self):
        for mutation in (lambda b: b["intervals"].clear(),
                         lambda b: b.update(returned_results=0),
                         lambda b: b["intervals"][0].pop("task_after"),
                         lambda b: b["counts"].update(physics_interval_attempts=2)):
            a, b = self.witnesses(self.fresh_raw)
            mutation(b)
            self.assertEqual(self.compare(a, b)["status"], "INCOMPLETE")
        a, b = self.witnesses(self.fresh_raw)
        b["intervals"] = None
        self.assertEqual(self.compare(a, b)["status"], "INCOMPLETE")

    def test_original_raw_missing_or_changed_blocks_pass(self):
        a, b = self.witnesses(self.fresh_raw)
        Path(next(iter(a["raw_reference"]["files"]))).write_bytes(b"modified")
        result = self.compare(a, b)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertTrue(any("raw_file_missing_or_changed" in item for item in result["incomplete_reasons"]))

    def test_terminal_type_difference_is_failure_and_no_post_terminal_interval_allowed(self):
        a, b = self.witnesses(self.fresh_raw)
        for w in (a, b):
            w["run_scope"] = "final_evaluation"
            w["final_task"].update(phase="failed", terminal_reason="timeout")
            w["intervals"][0]["task_after"] = deepcopy(w["final_task"])
            w["termination"] = "timeout"
        b["final_task"]["terminal_reason"] = "planner_invalid"
        b["intervals"][0]["task_after"] = deepcopy(b["final_task"])
        b["termination"] = "planner_invalid"
        self.assertEqual(self.compare(a, b)["status"], "FAIL")
        b["intervals"][0]["task_before"]["phase"] = "failed"
        self.assertEqual(self.compare(a, b)["status"], "INCOMPLETE")

    def test_guard_violation_and_wrong_task_budget_do_not_pass(self):
        a, b = self.witnesses(self.fresh_raw)
        b["future_engine_calls"]["mj_step"] = 1
        self.assertEqual(self.compare(a, b)["status"], "INCOMPLETE")
        a, b = self.witnesses(self.fresh_raw)
        a["task_spec"]["position_tolerance_m"] *= 2
        b["task_spec"]["position_tolerance_m"] *= 2
        self.assertIn("comparison_task_differs_from_bound_task_spec", self.compare(a, b)["incomplete_reasons"])

    def test_nonfinite_state_cannot_be_disguised_as_valid_and_missing_task_is_detected(self):
        a, b = self.witnesses(self.fresh_raw)
        b["intervals"][0]["state_after"][0] = "NaN"
        self.assertEqual(self.compare(a, b)["status"], "INCOMPLETE")
        a, b = self.witnesses(self.fresh_raw)
        b["intervals"][0]["task_after"].pop("held_time_s")
        self.assertEqual(self.compare(a, b)["status"], "INCOMPLETE")

    def test_stale_decision_cannot_pass_within_boundary_numeric_budget(self):
        a, b = self.witnesses(self.fresh_raw)
        b["decisions"][0]["state"][0] += 1e-7
        result = self.compare(a, b)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertIn("fresh:decision_boundary_mismatch:0", result["incomplete_reasons"])

    def test_all_invalid_zero_interval_decision_is_valid_and_never_invents_a_step(self):
        options = {"identity": {"case": "existing-dev"}, "source_bindings": {"source": "a"*64}}
        witnesses = []
        raw = self.raw("all-invalid-reference", 0)
        for name in ("original", "fresh"):
            execution = self.execution(name)
            execution.records.clear(); execution.results.clear()
            execution.counts.update(physics_interval_attempts=0, physics_intervals_returned=0, completed_boundary_records=0)
            execution.decisions = [{"executed": False, "reason": "planner_invalid", "plan_identity": self.plan}]
            execution.unexecuted_plan = deepcopy(execution.decisions[0])
            execution.final_task = {**execution.initial_task, "phase": "failed", "terminal_reason": "planner_invalid"}
            execution.termination = "planner_invalid"
            execution.run_scope = "final_evaluation"
            witnesses.append(build_witness(execution, **options, attempt_id=f"{name}-attempt",
                raw_reference=raw if name == "original" else None,
                original_raw_reference=None if name == "original" else raw))
        result = self.compare(*witnesses)
        self.assertEqual(result["status"], "PASS_REPRODUCTION")
        self.assertEqual(witnesses[1]["counts"]["physics_interval_attempts"], 0)

    def test_physical_failure_demands_full_prefix_even_if_it_reproduces(self):
        a, b = self.witnesses()
        for w in (a, b):
            w["run_scope"] = "final_evaluation"
            w["termination"] = "physics_error"
            w["final_task"].update(phase="failed", terminal_reason="physics_error", violations=["partial_transition"])
            w["intervals"][0]["task_after"] = deepcopy(w["final_task"])
            w["intervals"][0]["transition_valid"] = False
        self.assertEqual(self.compare(a, b)["status"], "INCOMPLETE")
        b["raw_reference"] = self.fresh_raw
        result = self.compare(a, b)
        self.assertEqual(result["status"], "PASS_REPRODUCTION")
        self.assertTrue(result["physical_failure_observed"])
        self.assertTrue(result["fresh_full_raw_prefix_required"])
        self.assertEqual(result["task_success_claim"], "not_assessed_by_boundary_reproduction")


if __name__ == "__main__":
    unittest.main()
