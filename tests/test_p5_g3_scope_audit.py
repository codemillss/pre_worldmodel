"""Independent, pure-fixture probes for the P5 pre-freeze scope audit.

No final seeds, real model calls, environment construction or physics stepping.
Failing probes identify contract gaps; stubbed review tests check wiring only.
"""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from push_align.final_manifest import CONTROLLERS, expected_E1, expected_E2, family_records
from push_align.final_statistics import METRICS, analyze_E2

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p5_scope_audit_coordinator", PROJECT / "scripts/p5_final_run.py")
coordinator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coordinator)


def fixture_rows():
    physical = {"object": {"size_full_xyz_m": [.06, .03, .02]},
                "pusher": {"radius_m": .015}, "reset": {"pusher_surface_gap_m": .006}}
    families, _ = family_records(physical, seed=123, new_family_seed=456, initial_id_prefix="audit-fixture")
    expected = expected_E2(families)
    rows = []
    for cell in expected:
        row = dict(cell, execution_status="completed", termination="timeout",
                   **{key: 1. for key in METRICS})
        row.update(success=False, elapsed_s=30., recontact_count=1,
                   final_observation_available=True, metric_state_origin="terminal_observation",
                   metric_state_time_s=30.)
        rows.append(row)
    return families, expected, rows


class IndependentStatisticsProbes(unittest.TestCase):
    def test_family_seed_mean_and_shared_bootstrap_match_direct_arithmetic(self):
        _, expected, rows = fixture_rows()
        for row in rows:
            family_index = row["episode_index"]
            controller = row["controller_id"]
            seed_shift = {"S1-11": 1., "S1-23": 4., "S1-47": 10.,
                          "SK-11": 3., "SK-23": 8., "SK-47": 16.}.get(controller, 0.)
            row["planning_wall_s"] = family_index * (2. if controller.startswith("SK") else 1.) + seed_shift
        result, arrays = analyze_E2(rows, expected)
        ordered = result["bootstrap"]["families_in_order"]
        by = {(row["family_id"], row["controller_id"]): row for row in rows}
        differences = np.array([
            sum(by[f, f"SK-{seed}"]["planning_wall_s"] for seed in (11, 23, 47)) / 3
            - sum(by[f, f"S1-{seed}"]["planning_wall_s"] for seed in (11, 23, 47)) / 3
            for f in ordered])
        direct = differences[arrays["family_draw_indices"]].mean(axis=1)
        np.testing.assert_allclose(arrays["SK_minus_S1_planning_wall_s"], direct, atol=1e-12, rtol=1e-14)
        contrast = next(x for x in result["comparisons"] if (x["left"], x["right"]) == ("SK", "S1"))
        np.testing.assert_allclose(contrast["metrics"]["planning_wall_s"]["percentile95_interval"],
                                   np.percentile(direct, [2.5, 97.5]), atol=1e-12, rtol=1e-14)
        self.assertAlmostEqual(contrast["metrics"]["planning_wall_s"]["difference_mean"], differences.mean())

    def test_all_missing_state_keeps_all_failure_denominators_and_zero_eligible_CI(self):
        _, expected, rows = fixture_rows()
        for row in rows:
            if row["controller_id"] == "G":
                row.update(termination="physics_error", final_observation_available=False,
                           metric_state_origin="unavailable", metric_state_time_s=None,
                           unavailable_reason="synthetic missing-state fixture")
                for key in METRICS[2:6]: row[key] = None
        result, arrays = analyze_E2(rows, expected)
        self.assertEqual(result["controllers"]["G"]["episode_count"], 12)
        self.assertEqual(result["controllers"]["G"]["termination_counts"], {"physics_error": 12})
        self.assertEqual(result["controllers"]["G"]["metric_denominators"]["final_position_error_m"], 0)
        contrast = next(x for x in result["comparisons"] if (x["left"], x["right"]) == ("R", "G"))
        metric = contrast["metrics"]["final_position_error_m"]
        self.assertEqual(metric["eligible_family_denominator"], 0)
        self.assertIsNone(metric["percentile95_interval"])
        self.assertEqual(metric["bootstrap_valid_replicates"], 0)
        self.assertFalse(arrays["R_minus_G_final_position_error_m_valid"].any())
        self.assertEqual(contrast["metrics"]["success"]["eligible_family_denominator"], 12)

    def test_frozen_visualization_uses_the_visual_tool_field_name(self):
        tree = ast.parse((PROJECT / "scripts/p5_final_run.py").read_text())
        visual = [value for node in ast.walk(tree) if isinstance(node, ast.Dict)
                  for key, value in zip(node.keys, node.values)
                  if isinstance(key, ast.Constant) and key.value == "visualization" and isinstance(value, ast.Dict)]
        self.assertEqual(len(visual), 1)
        keys = {key.value for key in visual[0].keys if isinstance(key, ast.Constant)}
        self.assertIn("postterminal_tail_s", keys)


class IndependentAdoptionProbes(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="p5-scope-audit-fixture-", dir=PROJECT / "tmp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"; self.run.mkdir()
        families, expected, _ = fixture_rows()
        self.manifest = {"families": families, "E2": expected, "E1": expected_E1(families),
                         "model_identities": {key: {"fixture_model": key} for key in CONTROLLERS[1:]},
                         "normalizer_identity": {"fixture_normalizer": "current-study"},
                         "source_sha256": {}, "runtime_source_sha256": {},
                         "physical_config": {}, "profile": {}}
        self.write(self.run / "manifest.json", self.manifest)
        self.patch_root = patch.object(coordinator, "ROOT", self.root); self.patch_root.start()
        self.addCleanup(self.patch_root.stop)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_empty_hash_seal_is_not_a_complete_statistics_stage(self):
        (self.run / "statistics").mkdir()
        coordinator.seal_derived_stage(self.run, "statistics", {})
        with self.assertRaisesRegex(RuntimeError, "Incomplete|Missing|population|output|summary"):
            coordinator.adopt_derived_stage(self.run, "statistics")

    def test_resume_review_expectations_are_bound_to_current_study(self):
        # This isolates the coordinator's expectation arguments, not raw-review
        # correctness. An old study record is otherwise internally consistent.
        cell = self.manifest["E2"][0]
        folder = self.run / "E2" / cell["family_id"] / cell["controller_id"]
        folder.mkdir(parents=True)
        execution = {"execution_status": "completed", "termination": "timeout",
                     "counts": {}, "future_engine_calls": {"mj_step": 0}}
        identity = {"study": "P5-final", "family_id": cell["family_id"],
                    "controller_id": cell["controller_id"], "model": None, "normalizer": None,
                    "final_manifest_sha256": "another-study-manifest"}
        record = {"record_kind": "reference_controller_execution", "identity": identity,
                  "execution": execution, "files_sha256": {}, "source_bindings": {}}
        self.write(folder / "manifest.json", record)
        review = {"status": "ACCEPT_COMPLETED_TASK_FAILURE", "metrics": {"success": False}}
        self.write(folder / "review.json", review)
        result = {"status": review["status"], "phase": "E2", "cell": cell,
                  "row": dict(cell, **execution, success=False),
                  "record_manifest": str(folder / "manifest.json"),
                  "record_manifest_sha256": coordinator.sha(folder / "manifest.json"),
                  "review_sha256": coordinator.sha(folder / "review.json")}
        self.write(folder / "worker_result.json", result)
        self.write(folder / "worker_result_receipt.json", {"sha256": coordinator.sha(folder / "worker_result.json")})
        from push_align import final_result_review
        with patch.object(final_result_review, "review_recording", return_value=deepcopy(review)) as independent:
            with self.assertRaisesRegex(RuntimeError, "current frozen study/model/source contract"):
                coordinator.verified_result(self.run, "E2", cell["cell_id"], self.manifest)
            independent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
