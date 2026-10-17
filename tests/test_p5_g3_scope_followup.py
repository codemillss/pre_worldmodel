"""Independent small-file shared-pool checks; no model/physics/final poses."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p5_scope_followup", PROJECT / "scripts/p5_final_run.py")
coordinator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coordinator)


class SharedPoolFollowup(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="p5-scope-pool-fixture-", dir=PROJECT / "tmp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"; self.run.mkdir()
        self.plan = json.loads((PROJECT / "configs/p5_g2_resource_plan.v3.json").read_text())
        self.manifest = {"resource_plan_path": "configs/plan.json", "E2": [], "E1": [],
                         "application_family": {"family_id": "fixture-application"}}
        self.state = {"E2_done": [], "E1_done": [], "application_done": [],
                      "replay_done": [], "replay_selection": None, "operational_bytes": 400}
        self.write(self.root / "configs/plan.json", self.plan)
        self.write(self.run / "manifest.json", self.manifest)
        coordinator.save_progress(self.run, self.state)
        root_patch = patch.object(coordinator, "ROOT", self.root)
        root_patch.start(); self.addCleanup(root_patch.stop)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_later_application_metadata_shares_existing_statistics_allowance(self):
        # Producer/reviewer results are stubbed only to isolate the resource
        # commit. The tiny file sizes and ledger write are real.
        folder = self.run / "application/fixture-application/G"
        self.write(folder / "manifest.json", {})
        self.write(folder / "worker_result.json", {})
        review = {"status": "PASS_MEASURED_COMPONENT_ALLOWANCES",
                  "measured_component_paths": [], "mismatch_charged": False}
        self.write(folder / "component_budget_review.json", review)
        self.write(self.run / "statistics/retained.json", {"payload": "x" * 1000})
        stats_bytes = coordinator.tree_bytes(self.run / "statistics")
        cell_bytes = coordinator.tree_bytes(folder)
        allowance = self.state["operational_bytes"] + cell_bytes + stats_bytes - 1
        self.plan["fixed_allowances_bytes"]["P5_boundary_analysis"] = allowance
        self.write(self.root / "configs/plan.json", self.plan)
        self.assertLess(self.state["operational_bytes"] + cell_bytes, allowance)
        self.assertLess(self.state["operational_bytes"] + stats_bytes, allowance)
        result = {"record_manifest": str(folder / "manifest.json"),
                  "component_review_sha256": coordinator.sha(folder / "component_budget_review.json"),
                  "row": {"termination": "timeout"}}
        with patch.object(coordinator, "frozen", return_value=self.manifest), \
             patch.object(coordinator, "validated_progress", return_value=deepcopy(self.state)), \
             patch.object(coordinator, "capacity_check", return_value={}), \
             patch.object(coordinator, "verified_result", return_value=result), \
             patch("push_align.final_manifest.CONTROLLERS", ("G",)):
            with self.assertRaisesRegex(RuntimeError, "Operational metadata/analysis allowance"):
                coordinator.execute_cells(self.run, "application")
        self.assertEqual(coordinator.progress(self.run)["application_done"], [])

    def test_E1_metadata_over_shared_pool_stops_before_any_repeat(self):
        import p5_final_e1
        import p5_final_e1_adoption
        self.plan["fixed_allowances_bytes"]["E1_prediction_analysis"] = 1000
        self.write(self.root / "configs/plan.json", self.plan)
        callbacks = []

        def inert_writer(run, manifest, callback):
            self.write(run / "E1/retained_metadata.json", {"payload": "x" * 1001})
            callbacks.append("predictions_frozen")
            callback({"event": "predictions_frozen", "choices_sha256": "a" * 64})
            callbacks.append("would_start_physical_repeat")
            return {"status": "COMPLETE_RECORDED"}

        with patch.object(coordinator, "frozen", return_value=self.manifest), \
             patch.object(coordinator, "capacity_check", return_value={}), \
             patch.object(p5_final_e1_adoption, "adopt_complete_e1", return_value=None), \
             patch.object(p5_final_e1_adoption, "seal_prediction_bundle"), \
             patch.object(p5_final_e1, "run_e1", side_effect=inert_writer):
            with self.assertRaisesRegex(RuntimeError, "allowance|pool|Analysis"):
                coordinator.E1(self.run)
        self.assertEqual(callbacks, ["predictions_frozen"])

    def test_analysis_pool_combines_E1_nonraw_fixed_prediction_and_E1_statistics(self):
        self.write(self.run / "E1/snapshots/fixture/physical/summary.json", {
            "status": "COMPLETE_RECORDED", "raw_receipt": {"files_sha256": {
                str(self.run / "E1/raw.bin"): "fixture-hash"}}})
        (self.run / "E1/raw.bin").write_bytes(b"x" * 101)
        self.write(self.run / "fixed_prediction/summary.json", {"fixture": "fixed"})
        self.write(self.run / "e1_statistics/summary.json", {"fixture": "E1"})
        expected = (coordinator.tree_bytes(self.run / "E1") - 101
                    + coordinator.tree_bytes(self.run / "fixed_prediction")
                    + coordinator.tree_bytes(self.run / "e1_statistics"))
        for name in ("fixed_prediction", "e1_statistics", "E1"):
            self.assertEqual(coordinator.derived_pool_bytes(self.run, name), expected)


if __name__ == "__main__":
    unittest.main()
