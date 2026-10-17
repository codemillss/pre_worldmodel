"""Synthetic, stdlib-only tests; no scientific run, model or physics is used."""
from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools/p5_replay_witness_crosscheck_v1.py"
CONTROLLERS = ("G", "R", "S1-11", "S1-23", "S1-47", "SK-11", "SK-23", "SK-47")
STRATA = ("translation", "rotation", "offset")
RULE = "per controller/learning seed: minimum episode ID per stratum UNION minimum ID per observed termination type; deduplicate"
ENGINE_ZERO = dict.fromkeys(("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse"), 0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(path.read_text())


def write_gzip(path, data, *, allow_nan=False):
    path.write_bytes(gzip.compress(json.dumps(data, sort_keys=True, allow_nan=allow_nan).encode(), mtime=0))


def union_selection(rows):
    """Independent fixture recipe, without importing the production selector."""
    selected = []
    for controller in CONTROLLERS:
        part = [row for row in rows if row["controller_id"] == controller]
        ids = {min((row for row in part if row["stratum"] == stratum),
                   key=lambda row: row["episode_id"])["cell_id"] for stratum in STRATA}
        for terminal in {row["termination"] for row in part}:
            ids.add(min((row for row in part if row["termination"] == terminal),
                        key=lambda row: row["episode_id"])["cell_id"])
        selected.extend(deepcopy(row) for row in sorted(part, key=lambda row: row["episode_id"])
                        if row["cell_id"] in ids)
    return selected


class SyntheticRun:
    """Self-contained 96-cell identity graph with exactly 33 selected replays."""

    def __init__(self, root):
        root = Path(root).resolve()
        self.root = root
        self.run = root / "runs/p5-final-synthetic"
        self.run.mkdir(parents=True)
        source = root / "source/scientific.txt"
        runtime = root / "source/runtime.txt"
        source.parent.mkdir(parents=True)
        source.write_text("synthetic scientific identity only\n")
        runtime.write_text("synthetic runtime identity only\n")
        self.cells, self.rows = [], []
        for si, stratum in enumerate(STRATA):
            for number in range(4):
                family = f"p5-final-{stratum}-{number:02d}"
                for controller in CONTROLLERS:
                    cell = {"family_id": family, "episode_id": family,
                            "episode_index": si * 4 + number, "stratum": stratum,
                            "controller_id": controller, "cell_id": f"{family}/{controller}"}
                    terminal = "ready" if stratum == "translation" and number == 1 else "timeout"
                    if controller == "S1-11" and stratum == "offset" and number == 1:
                        terminal = "planner_invalid"
                    self.cells.append(cell)
                    self.rows.append({**cell, "execution_status": "completed", "termination": terminal,
                        "success": terminal == "ready", "elapsed_s": .3 if terminal == "ready" else 30.,
                        "counts": {key: 1 for key in ("controller_decisions", "physics_interval_attempts",
                            "physics_intervals_returned", "completed_boundary_records")},
                        "future_engine_calls": dict(ENGINE_ZERO), "final_position_error_m": .001,
                        "final_yaw_error_rad": 0., "final_linear_speed_m_s": 0.,
                        "final_angular_speed_rad_s": 0., "final_observation_available": True,
                        "metric_state_origin": "terminal_observation", "metric_state_time_s": 30.,
                        "unavailable_reason": None})
        self.selection = union_selection(self.rows)
        assert len(self.cells) == 96 and len(self.selection) == 33
        families = [{key: cell[key] for key in ("family_id", "episode_id", "episode_index", "stratum")}
                    for cell in self.cells[::len(CONTROLLERS)]]
        self.manifest = {"schema": "p5-frozen-final-manifest-v1", "status": "FROZEN_BEFORE_FINAL_OUTCOMES",
            "families": families, "E2": self.cells, "replay_selection_rule": RULE,
            "source_sha256": {str(source.relative_to(root)): sha(source),
                              str(runtime.relative_to(root)): sha(runtime)},
            "runtime_source_sha256": {str(runtime.relative_to(root)): sha(runtime)},
            "model_identities": {c: {"model_hash": hashlib.sha256(c.encode()).hexdigest()}
                                 for c in CONTROLLERS if c != "G"},
            "normalizer_identity": {"normalizer_hash": "b" * 64},
            "physical_config": {"task": {"position_tolerance_m": .005,
                "yaw_tolerance_rad": .05235987755982989, "linear_speed_tolerance_m_s": .01,
                "angular_speed_tolerance_rad_s": .05, "hold_time_s": .3,
                "max_episode_time_s": 30.}},
            "profile": {"controller_replay": {"tolerance_fraction_of_task_tolerance": .001}}}
        self.manifest_path = self.run / "manifest.json"
        write_json(self.manifest_path, self.manifest)
        self.manifest_sha = sha(self.manifest_path)
        write_json(self.run / "freeze_receipt.json", {"path": str(self.manifest_path),
            "sha256": self.manifest_sha, "bytes": self.manifest_path.stat().st_size})
        self.bindings = {str(root / p): h for key in ("source_sha256", "runtime_source_sha256")
                         for p, h in self.manifest[key].items()}
        self.bindings[str(self.manifest_path)] = self.manifest_sha
        for row in self.rows:
            self.make_cell("E2", row)
        for row in self.selection:
            self.make_cell("replay", row)
        write_json(self.run / "replay_selection.json", self.selection)
        self.progress = {"E2_done": [c["cell_id"] for c in self.cells],
            "replay_done": [r["cell_id"] for r in self.selection],
            "replay_selection": self.selection, "application_done": [], "E1_done": []}
        self.save_progress()

    def save_progress(self):
        write_json(self.run / "progress.json", self.progress)
        write_json(self.run / "progress_receipt.json", {"sha256": sha(self.run / "progress.json")})

    def folder(self, phase, cell_id=None):
        return self.run / ("E2" if phase == "E2" else "replays") / (cell_id or self.selection[0]["cell_id"])

    def witness(self, phase, row, identity, raw_reference):
        fresh = phase == "replay"
        original_folder = self.folder("E2", row["cell_id"])
        original = read_json(original_folder / "manifest.json") if fresh else None
        original_witness = json.loads(gzip.decompress((original_folder / "witness.json.gz").read_bytes())) if fresh else None
        task = {"phase": "ready" if row["success"] else "failed", "initial_sim_time": 0.,
            "last_sim_time": row["elapsed_s"], "held_time_s": .3 if row["success"] else 0.,
            "previous_qualifies": row["success"], "terminal_reason": row["termination"], "violations": []}
        initial = {"state": [0.] * 10, "sim_time": 0., "observation_valid": True,
            "schema_version": "state10-v1", "snapshot_hash": "c" * 64,
            "task": {"phase": "running"}, "planner_context": {}, "geometry_context": {}}
        decision = {"index": 0, "local_index": 0, "source": "geometry" if row["controller_id"] == "G" else "model",
            "state": [0.] * 10, "sim_time": 0., "holding": False,
            "geometry_proposal": {"b_unclipped_m": .012, "offset_saturated": False},
            "plan_identity": None if row["controller_id"] == "G" else {
                "candidate_hash": "1" * 64, "prediction_hash": "2" * 64, "choice_hash": "3" * 64},
            "executed": True}
        interval = {"index": 0, "state_before": [0.] * 10, "state_after": [0.] * 10,
            "sim_time_before": 0., "sim_time_after": .1, "actual_dt_s": .1,
            "requested_action": [0., 0.], "applied_action": [0., 0.], "holding": False}
        return {"schema": "p5-controller-boundary-witness-v1", "identity": identity,
            "source_bindings": self.bindings, "attempt_id": f"attempt-{phase}-{row['cell_id']}",
            "execution_id": f"execution-{phase}-{row['cell_id']}",
            "record_kind": "fresh_controller_replay" if fresh else "reference_controller_execution",
            "origin": {"kind": "synthetic_saved_snapshot"}, "controller_id": row["controller_id"],
            "case_id": row["family_id"], "run_scope": "final_evaluation",
            "initial_full_snapshot_equal": True, "initial": initial,
            "task_spec": self.manifest["physical_config"]["task"], "intervals": [interval],
            "decisions": [decision], "unexecuted_plan": None, "final_task": task,
            "execution_status": "completed", "termination": row["termination"], "error": None,
            "counts": row["counts"], "future_engine_calls": dict(ENGINE_ZERO), "returned_results": 1,
            "raw_reference": None if fresh else raw_reference,
            "original_raw_reference": original_witness["raw_reference"] if fresh else None,
            "scope": "synthetic JSON witness; no physical simulation"}

    def make_cell(self, phase, row):
        folder = self.folder(phase, row["cell_id"])
        folder.mkdir(parents=True)
        cell = next(c for c in self.cells if c["cell_id"] == row["cell_id"])
        identity = {"study": "P5-final", "family_id": row["family_id"],
            "controller_id": row["controller_id"], "final_manifest_sha256": self.manifest_sha,
            "model": None if row["controller_id"] == "G" else self.manifest["model_identities"][row["controller_id"]],
            "normalizer": None if row["controller_id"] == "G" else self.manifest["normalizer_identity"]}
        raw_file = folder / "raw.bin"
        raw_file.write_bytes(b"synthetic-not-physics")
        raw_ref = {"scope": "complete_raw_execution_prefix", "files": {str(raw_file): sha(raw_file)},
                   "executed_intervals": 1}
        witness_path = folder / "witness.json.gz"
        witness = self.witness(phase, row, identity, raw_ref)
        write_gzip(witness_path, witness)
        fresh = phase == "replay"
        exact = {"status": "PASS_EXACT", "left_sha256": "d" * 64, "right_sha256": "d" * 64}
        comparison = {"boundary": {"status": "PASS_REPRODUCTION", "comparison_matches": True,
            "maximum_differences": {"position_m": 0., "state_components": [0.] * 10},
            "incomplete_reasons": [], "mismatch_reasons": [], "physical_failure_observed": False,
            "fresh_full_raw_prefix_required": False}, "raw_exact": deepcopy(exact),
            "plan_index_exact": deepcopy(exact), "differing_ticks": [0]} if fresh else None
        original_path = self.folder("E2", row["cell_id"]) / "manifest.json"
        record = {"schema": "p5-recorded-episode-v1", "record_kind": witness["record_kind"],
            "recording_status": "COMPLETE_RECORDED", "identity": identity,
            "source_bindings": self.bindings, "execution": {"execution_status": "completed",
                "execution_id": witness["execution_id"], "case_id": row["family_id"],
                "controller_id": row["controller_id"], "run_scope": "final_evaluation",
                "termination": row["termination"], "counts": row["counts"],
                "future_engine_calls": dict(ENGINE_ZERO)},
            "files_sha256": {str(witness_path): sha(witness_path), str(raw_file): sha(raw_file)},
            "witness_path": str(witness_path), "comparison": comparison,
            "reference_manifest": str(original_path) if fresh else None,
            "reference_manifest_sha256": sha(original_path) if fresh else None,
            "raw": None if fresh else {"prefix": str(folder / "raw"), "files_sha256": raw_ref["files"]},
            "plan_index": None, "full_traces": []}
        write_json(folder / "manifest.json", record)
        status = "ACCEPT_COMPLETED_RECORDING" if row["success"] else "ACCEPT_COMPLETED_TASK_FAILURE"
        review = {"schema": "p5-independent-recording-review-v1", "accepted": True,
            "status": status, "issues": [], "manifest_sha256": sha(folder / "manifest.json"),
            "metrics": {"final_position_error_m": row["final_position_error_m"]}}
        write_json(folder / "review.json", review)
        write_json(folder / "component_budget_review.json", {"status": "PASS_MEASURED_COMPONENT_ALLOWANCES"})
        worker = {"status": status, "phase": phase, "cell": cell, "row": deepcopy(row),
            "record_manifest": str(folder / "manifest.json"),
            "record_manifest_sha256": sha(folder / "manifest.json"),
            "review_sha256": sha(folder / "review.json"),
            "component_review_sha256": sha(folder / "component_budget_review.json"),
            "recording_status": "COMPLETE_RECORDED", "comparison": comparison, "learning_updates": 0}
        write_json(folder / "worker_result.json", worker)
        self.seal_worker(folder)

    def seal_worker(self, folder):
        write_json(folder / "worker_result_receipt.json", {"sha256": sha(folder / "worker_result.json")})

    def reseal_record(self, folder, record):
        """Keep every downstream synthetic seal current: changes must reach comparison."""
        write_json(folder / "manifest.json", record)
        review = read_json(folder / "review.json")
        review["manifest_sha256"] = sha(folder / "manifest.json")
        write_json(folder / "review.json", review)
        worker = read_json(folder / "worker_result.json")
        worker.update(record_manifest_sha256=sha(folder / "manifest.json"),
                      review_sha256=sha(folder / "review.json"), comparison=record["comparison"])
        write_json(folder / "worker_result.json", worker)
        self.seal_worker(folder)

    def mutate_witness(self, change, *, phase="replay", allow_nan=False):
        folder = self.folder(phase)
        record = read_json(folder / "manifest.json")
        p = Path(record["witness_path"])
        data = json.loads(gzip.decompress(p.read_bytes()))
        change(data)
        write_gzip(p, data, allow_nan=allow_nan)
        record["files_sha256"][str(p)] = sha(p)
        self.reseal_record(folder, record)


class ReplayWitnessCrosscheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("p5_replay_witness_crosscheck_under_test", TOOL_PATH)
        cls.tool = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.tool
        spec.loader.exec_module(cls.tool)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="p5-json-witness-synthetic-")
        self.addCleanup(temporary.cleanup)
        self.fixture = SyntheticRun(Path(temporary.name))

    def crosscheck(self):
        return self.tool.crosscheck(self.fixture.run, self.fixture.manifest_sha)

    def assert_rejected(self):
        result = self.crosscheck()
        self.assertIsInstance(result, dict)
        brief = {key: result.get(key) for key in ("status", "accepted", "verified_replays")}
        brief["errors"] = result.get("errors", [])[:3]
        self.assertEqual(result["schema"], "p5-saved-replay-witness-crosscheck-v1")
        self.assertEqual(result["status"], "FAIL_RETAIN_DIAGNOSTICS", brief)
        self.assertIs(result["accepted"], False)
        self.assertTrue(result["errors"], brief)
        return result

    def test_complete_33_replays_pass_without_changing_inputs(self):
        before = {str(p): sha(p) for p in self.fixture.root.rglob("*") if p.is_file()}
        result = self.crosscheck()
        self.assertEqual(result["schema"], "p5-saved-replay-witness-crosscheck-v1")
        self.assertEqual(result["status"], "PASS_SAVED_WITNESS_CROSSCHECK", result.get("errors", [])[:3])
        self.assertIs(result["accepted"], True)
        self.assertEqual(result["expected_replays"], 33)
        self.assertEqual(result["verified_replays"], 33)
        self.assertEqual(len(result["cases"]), 33)
        self.assertEqual(result["errors"], [])
        for case in result["cases"]:
            is_g = case["controller_id"] == "G"
            self.assertEqual(case["G_empty_learned_plan_index"], is_g)
            self.assertEqual(case["learned_plan_decisions"], 0 if is_g else 1)
        self.assertEqual(len(self.fixture.selection), 33)
        after = {str(p): sha(p) for p in self.fixture.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_missing_33rd_replay_is_not_complete(self):
        self.fixture.progress["replay_done"].pop()
        self.fixture.save_progress()
        self.assert_rejected()

    def test_matching_unexecuted_terminal_decision_is_valid(self):
        cell_id = "p5-final-offset-01/S1-11"
        for phase in ("E2", "replay"):
            folder = self.fixture.folder(phase, cell_id)
            record = read_json(folder / "manifest.json")
            path = Path(record["witness_path"])
            data = json.loads(gzip.decompress(path.read_bytes()))
            decision = deepcopy(data["decisions"][-1])
            decision.update(index=1, local_index=1, executed=False, sim_time=.1)
            decision["plan_identity"] = {"reason": "planner_invalid"}
            data["decisions"].append(decision)
            data["unexecuted_plan"] = deepcopy(decision)
            data["counts"]["controller_decisions"] += 1
            write_gzip(path, data)
            record["files_sha256"][str(path)] = sha(path)
            record["execution"]["counts"] = deepcopy(data["counts"])
            if phase == "replay":
                original = self.fixture.folder("E2", cell_id) / "manifest.json"
                record["reference_manifest_sha256"] = sha(original)
            worker = read_json(folder / "worker_result.json")
            worker["row"]["counts"] = deepcopy(data["counts"])
            write_json(folder / "worker_result.json", worker)
            self.fixture.reseal_record(folder, record)
        row = next(row for row in self.fixture.rows if row["cell_id"] == cell_id)
        row["counts"]["controller_decisions"] += 1
        selection = union_selection(self.fixture.rows)
        self.assertEqual(len(selection), 33)
        write_json(self.fixture.run / "replay_selection.json", selection)
        self.fixture.progress["replay_selection"] = selection
        self.fixture.save_progress()
        result = self.crosscheck()
        self.assertEqual(result["status"], "PASS_SAVED_WITNESS_CROSSCHECK", result.get("errors", [])[:3])
        self.assertEqual(result["verified_replays"], 33)

    def test_witness_sha_break_is_rejected(self):
        with (self.fixture.folder("replay") / "witness.json.gz").open("ab") as f:
            f.write(b"corruption")
        self.assert_rejected()

    def test_decision_change_even_after_all_seals_rewritten_is_rejected(self):
        self.fixture.mutate_witness(lambda d: d["decisions"][0]["state"].__setitem__(0, .123))
        result = self.assert_rejected()
        self.assertIn("payload differs", json.dumps(result["errors"]))

    def test_interval_change_even_after_all_seals_rewritten_is_rejected(self):
        self.fixture.mutate_witness(lambda d: d["intervals"][0]["applied_action"].__setitem__(0, .001))
        self.assert_rejected()

    def test_valid_selection_member_replacement_is_rejected(self):
        replacement = next(row for row in self.fixture.rows if row["controller_id"] == "G"
                           and row["family_id"] == "p5-final-offset-03")
        altered = deepcopy(self.fixture.selection)
        altered[0] = replacement
        write_json(self.fixture.run / "replay_selection.json", altered)
        self.fixture.progress["replay_selection"] = altered
        self.fixture.progress["replay_done"] = [r["cell_id"] for r in altered]
        self.fixture.save_progress()
        self.assert_rejected()

    def test_witness_path_escape_is_rejected_after_resealing(self):
        folder = self.fixture.folder("replay")
        record = read_json(folder / "manifest.json")
        outside = self.fixture.root / "escaped-witness.json.gz"
        outside.write_bytes(Path(record["witness_path"]).read_bytes())
        del record["files_sha256"][record["witness_path"]]
        record["witness_path"] = str(outside)
        record["files_sha256"][str(outside)] = sha(outside)
        self.fixture.reseal_record(folder, record)
        self.assert_rejected()

    def test_nonfinite_witness_is_rejected(self):
        self.fixture.mutate_witness(lambda d: d["decisions"][0].update(sim_time=float("nan")), allow_nan=True)
        self.assert_rejected()

    def test_duplicate_json_key_is_rejected_with_valid_sha(self):
        folder = self.fixture.folder("replay")
        p = folder / "worker_result.json"
        text = p.read_text().strip()
        p.write_text('{"status":"ACCEPT_COMPLETED_TASK_FAILURE",' + text[1:])
        self.fixture.seal_worker(folder)
        result = self.assert_rejected()
        self.assertIn("Duplicate JSON", json.dumps(result["errors"]))

    def test_oversize_json_is_rejected(self):
        folder = self.fixture.folder("replay")
        p = folder / "worker_result.json"
        p.write_bytes(b" " * (16 * 1024 * 1024 + 1) + p.read_bytes())
        self.fixture.seal_worker(folder)
        self.assert_rejected()

    def test_gzip_expansion_limit_is_enforced(self):
        self.fixture.mutate_witness(lambda d: d.update(oversize_padding="x" * (16 * 1024 * 1024 + 1)))
        result = self.assert_rejected()
        self.assertIn("Expanded witness exceeds size limit", json.dumps(result["errors"]))

    def test_compressed_gzip_size_limit_is_enforced(self):
        folder = self.fixture.folder("replay")
        p = folder / "witness.json.gz"
        data = json.loads(gzip.decompress(p.read_bytes()))
        data["padding"] = "x" * (8 * 1024 * 1024 + 1)
        # Valid gzip/JSON, expanded below16MiB, compressed above8MiB.
        # Compression level0 isolates the compressed-size boundary.
        payload = json.dumps(data, sort_keys=True, allow_nan=False).encode()
        self.assertLess(len(payload), 16 * 1024 * 1024)
        p.write_bytes(gzip.compress(payload, compresslevel=0, mtime=0))
        self.assertGreater(p.stat().st_size, 8 * 1024 * 1024)
        record = read_json(folder / "manifest.json")
        record["files_sha256"][str(p)] = sha(p)
        self.fixture.reseal_record(folder, record)
        result = self.assert_rejected()
        self.assertIn("size limit", json.dumps(result["errors"]))

    def test_primary_boundary_gate_fail_is_rejected(self):
        folder = self.fixture.folder("replay")
        record = read_json(folder / "manifest.json")
        record["comparison"]["boundary"]["status"] = "FAIL"
        self.fixture.reseal_record(folder, record)
        self.assert_rejected()

    def test_primary_raw_gate_fail_is_rejected(self):
        folder = self.fixture.folder("replay")
        record = read_json(folder / "manifest.json")
        record["comparison"]["raw_exact"]["status"] = "FAIL_EXACT"
        self.fixture.reseal_record(folder, record)
        self.assert_rejected()

    def test_primary_numeric_diagnostic_cannot_be_a_string(self):
        folder = self.fixture.folder("replay")
        record = read_json(folder / "manifest.json")
        record["comparison"]["boundary"]["maximum_differences"]["position_m"] = "unknown"
        self.fixture.reseal_record(folder, record)
        self.assert_rejected()

    def test_primary_exact_digest_disagreement_is_rejected(self):
        folder = self.fixture.folder("replay")
        record = read_json(folder / "manifest.json")
        record["comparison"]["plan_index_exact"]["right_sha256"] = "e" * 64
        self.fixture.reseal_record(folder, record)
        self.assert_rejected()

    def test_reused_attempt_identity_is_rejected(self):
        original = json.loads(gzip.decompress((self.fixture.folder("E2") / "witness.json.gz").read_bytes()))
        self.fixture.mutate_witness(lambda d: d.update(attempt_id=original["attempt_id"]))
        self.assert_rejected()

    def test_reused_execution_identity_is_rejected(self):
        original = json.loads(gzip.decompress((self.fixture.folder("E2") / "witness.json.gz").read_bytes()))
        self.fixture.mutate_witness(lambda d: d.update(execution_id=original["execution_id"]))
        self.assert_rejected()

    def test_wrong_scientific_identity_is_rejected(self):
        self.fixture.mutate_witness(lambda d: d["identity"].update(controller_id="R"))
        self.assert_rejected()

    def test_wrong_frozen_manifest_hash_is_rejected(self):
        result = self.tool.crosscheck(self.fixture.run, "f" * 64)
        self.assertEqual(result["status"], "FAIL_RETAIN_DIAGNOSTICS")
        self.assertIs(result["accepted"], False)
        self.assertTrue(result["errors"])

    def test_frozen_source_change_is_rejected(self):
        (self.fixture.root / "source/scientific.txt").write_text("different source bytes\n")
        self.assert_rejected()

    def test_component_review_failure_is_rejected(self):
        folder = self.fixture.folder("replay")
        p = folder / "component_budget_review.json"
        write_json(p, {"status": "FAIL_COMPONENT_ALLOWANCE"})
        worker = read_json(folder / "worker_result.json")
        worker["component_review_sha256"] = sha(p)
        write_json(folder / "worker_result.json", worker)
        self.fixture.seal_worker(folder)
        self.assert_rejected()

    def test_review_not_accepted_is_rejected(self):
        folder = self.fixture.folder("replay")
        p = folder / "review.json"
        review = read_json(p)
        review.update(accepted=False, status="STOP_REVIEW", issues=["synthetic rejection"])
        write_json(p, review)
        worker = read_json(folder / "worker_result.json")
        worker["review_sha256"] = sha(p)
        write_json(folder / "worker_result.json", worker)
        self.fixture.seal_worker(folder)
        self.assert_rejected()

    def test_cli_writes_new_supplement_without_overwriting(self):
        out = self.fixture.root / "reports/crosscheck.json"
        out.parent.mkdir()
        arguments = ["--run", str(self.fixture.run), "--manifest-sha256", self.fixture.manifest_sha,
                     "--out", str(out)]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.tool.main(arguments), 0)
        self.assertTrue(str(read_json(out)["status"]).startswith("PASS"))
        original = out.read_bytes()
        try:
            with redirect_stdout(io.StringIO()):
                code = self.tool.main(arguments)
        except (ValueError, RuntimeError, FileExistsError):
            code = 1
        self.assertNotEqual(code, 0)
        self.assertEqual(original, out.read_bytes())


if __name__ == "__main__":
    unittest.main()
