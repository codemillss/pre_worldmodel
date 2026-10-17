"""Pure main-runner gate tests: mocked recorders never call a physics engine."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from push_align.config import load_pilot_config
from push_align.data_intervention_main import build_main_plan

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import p4_data_main as main_runner  # noqa: E402


class MainCollectionRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = json.loads((ROOT / "configs/p4_data_intervention.design.v1.json").read_text())
        cls.config = load_pilot_config()
        cls.plans = build_main_plan(cls.design, cls.config)
        cls.slots = json.loads((ROOT / "reports/p4_data_intervention_slots.design.json").read_text())

    def test_measured_y_uses_observation_with_inclusive_registered_bounds(self):
        high = {"y_condition": "Y1"}
        low = {"y_condition": "Y0"}
        self.assertTrue(main_runner.measured_y_pass(high, .018, self.design))
        self.assertFalse(main_runner.measured_y_pass(high, np.nextafter(.018, -np.inf), self.design))
        ceiling = .0149908461
        self.assertTrue(main_runner.measured_y_pass(low, ceiling, self.design))
        self.assertFalse(main_runner.measured_y_pass(low, np.nextafter(ceiling, np.inf), self.design))
        for plan in self.plans:
            self.assertTrue(main_runner.measured_y_pass(plan, plan["moving_action"][1], self.design))
        # A correct target command cannot substitute for an actual low measurement.
        self.assertFalse(main_runner.measured_y_pass(self.plans[2], .017, self.design))

    def test_nonfinite_measured_y_never_passes(self):
        for y in ("Y0", "Y1"):
            for value in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(y=y, value=value):
                    self.assertIs(main_runner.measured_y_pass({"y_condition": y}, value, self.design), False)

    def test_window_contract_requires_each_count_and_exact_all_zero_location(self):
        base = {"transition_count": 20, "valid_transitions": 20,
                "valid_K5_windows": 16, "zero_action_count": 5,
                "zero_occurrences_in_all_K5": 25, "all_zero_K5_starts": []}
        for stop in ("S0", "S1"):
            expected = dict(base, all_zero_K5_starts=[10] if stop == "S1" else [])
            plan = {"stop_condition": stop}
            self.assertTrue(main_runner.window_pass(plan, expected))
            for key in base:
                bad = deepcopy(expected)
                bad[key] = ([9] if stop == "S1" else [10]) if key == "all_zero_K5_starts" else expected[key] - 1
                with self.subTest(stop=stop, field=key):
                    self.assertFalse(main_runner.window_pass(plan, bad))
            if stop == "S1":
                for changed in ([], [10, 10], [9, 10], [11]):
                    self.assertFalse(main_runner.window_pass(plan, dict(expected, all_zero_K5_starts=changed)))

    @staticmethod
    def coverage_row(split, y, family, contact, noncontact, role="core"):
        return {"role": role, "semantic_split": split, "Y": y, "family_id": family,
                "measurements": {"active_contact_transitions": contact,
                                 "noncontact_transitions": noncontact}}

    def test_contact_coverage_uses_unique_families_and_allows_overlap(self):
        rows = []
        for split, count in (("train", 3), ("development", 2)):
            for y in ("Y0", "Y1"):
                for index in range(count):
                    row = self.coverage_row(split, y, f"{split}-f{index}", 2, 18)
                    rows.extend([row, deepcopy(row)])  # S0/S1 do not double the family count.
        coverage = main_runner.contact_coverage(rows)
        self.assertEqual(len(coverage), 4)
        self.assertTrue(all(r["pass"] for r in coverage))
        for row in coverage:
            expected_count = 3 if row["split"] == "train" else 2
            self.assertEqual(len(row["contact_families"]), expected_count)
            self.assertEqual(row["contact_families"], row["noncontact_families"])

    def test_contact_coverage_cannot_use_duplicate_s_or_warmups_to_fill_missing_families(self):
        rows = []
        for split in ("train", "development"):
            for y in ("Y0", "Y1"):
                same = self.coverage_row(split, y, f"{split}-only", 20, 0)
                rows.extend([deepcopy(same) for _ in range(8)])
                rows.extend(self.coverage_row(split, y, f"{split}-warm{i}", 5, 15, role="warmup")
                            for i in range(4))
        for row in main_runner.contact_coverage(rows):
            self.assertFalse(row["pass"])
            self.assertEqual(len(row["contact_families"]), 1)
            self.assertEqual(row["noncontact_families"], [])

    def test_slot_bindings_are_core_train_only_and_partition_original_slots_exactly(self):
        old_rows = np.arange(2960 * 5, dtype=np.int64).reshape(2960, 5)
        with patch.object(main_runner, "PlanarPushEnv", side_effect=AssertionError("No physics")):
            bound = main_runner.bind_slots(self.plans, self.slots, old_rows)
        self.assertEqual(bound["status"], "REGISTERED_NOT_TRAINED")
        entries = bound["entries"]
        chosen = {e["slot"] for e in entries}
        old = set(bound["unchanged_old_slots"])
        self.assertEqual((len(entries), len(chosen), len(old)), (384, 384, 2576))
        self.assertFalse(chosen & old)
        self.assertEqual(chosen | old, set(range(2960)))
        self.assertEqual([(e["family_rank"], e["window_start"]) for e in entries],
                         [(f, w) for f in range(24) for w in range(16)])
        lookup = {p["episode_id"]: p for p in self.plans}
        for entry, registered in zip(entries, self.slots["entries"]):
            self.assertEqual(entry["slot"], registered["slot"])
            self.assertEqual(entry["replaced_original_prepared_rows"], old_rows[entry["slot"]].tolist())
            self.assertEqual(set(entry["arm_episodes"]), {"D00", "D10", "D01", "D11"})
            for arm, episode_id in entry["arm_episodes"].items():
                p = lookup[episode_id]
                self.assertEqual(p["semantic_split"], "train")
                self.assertTrue(p["core_training_eligible"])
                self.assertFalse(p["warmup_training_eligible"])
                self.assertEqual(p["family_id"], entry["family_id"])
                self.assertEqual(p["family_rank"], entry["family_rank"])
                self.assertEqual(arm, "D" + p["y_condition"][1] + p["stop_condition"][1])

    def test_malformed_slot_bindings_fail_before_physics(self):
        old_rows = np.arange(2960 * 5, dtype=np.int64).reshape(2960, 5)
        cases = []
        for key, value in (("slot", -1), ("slot", 2960), ("occurrences", 22),
                           ("window_start", 16), ("family_rank", 24)):
            slots = deepcopy(self.slots)
            slots["entries"][0][key] = value
            cases.append((f"{key}={value}", slots, old_rows))
        duplicate = deepcopy(self.slots)
        duplicate["entries"][1]["slot"] = duplicate["entries"][0]["slot"]
        cases.extend([("duplicate-slot", duplicate, old_rows),
                      ("too-few-old-rows", self.slots, old_rows[:-1])])
        for name, slots, data in cases:
            with self.subTest(name=name):
                with self.assertRaises((AssertionError, ValueError, IndexError,
                                        KeyError, main_runner.PreflightFailure)):
                    main_runner.bind_slots(self.plans, slots, data)

    def exercise_record(self, plan, role, *, invalid_at=None, clipped_at=None, action_count=3):
        """Mock only storage and physics to inspect real runner branching/accounting."""
        runner = main_runner.MainRunner(Path("/unused/p4-main-test"), self.design)
        created = []
        loaded = SimpleNamespace(transition_count=0, metadata={})

        class Recorder:
            def __init__(self, env, **kwargs):
                self.kwargs = kwargs
                self.results = []
                self.invalid = False
                created.append(self)

            def step(self, action):
                n = len(self.results) + 1
                self.invalid = n == invalid_at
                result = SimpleNamespace(requested_action=np.asarray(action),
                    applied_action=np.asarray(action) / 2 if n == clipped_at else np.asarray(action),
                    substep_times=np.zeros(200), actual_dt=.1)
                self.results.append(result)
                return result

            def finish(self, reason):
                loaded.transition_count = len(self.results)
                loaded.metadata = {**self.kwargs, "initial_record": {}, "terminal_reason": reason}
                return loaded

        env = SimpleNamespace(action_dt=.1, config=self.config)
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, "resources", return_value={}))
            stack.enter_context(patch.object(main_runner, "EpisodeRecorder", Recorder))
            stack.enter_context(patch.object(main_runner, "write_dataset"))
            stack.enter_context(patch.object(main_runner, "iter_dataset", side_effect=lambda *_: iter([loaded])))
            stack.enter_context(patch.object(main_runner, "validate_family_splits"))
            stack.enter_context(patch.object(main_runner, "exact_paths", return_value={"pass": True}))
            stack.enter_context(patch.object(main_runner, "digest", return_value="test-hash"))
            stack.enter_context(patch.object(main_runner, "measurements", side_effect=lambda *_: {
                "valid_transitions": loaded.transition_count - int(invalid_at is not None),
                "active_contact_transitions": 0}))
            stack.enter_context(patch.object(main_runner, "save"))
            stack.enter_context(patch("builtins.print"))
            error = None
            try:
                runner._record(env, plan, "mock-episode", [plan["moving_action"]] * action_count, {}, role)
            except main_runner.PreflightFailure as exc:
                error = exc
        return runner, created[0], loaded, error

    def test_record_metadata_eligibility_is_core_only_and_split_specific(self):
        for semantic in ("train", "development"):
            p = next(p for p in self.plans if p["semantic_split"] == semantic)
            for role in ("warmup", "core"):
                with self.subTest(semantic=semantic, role=role):
                    runner, recorder, loaded, error = self.exercise_record(p, role)
                    self.assertIsNone(error)
                    policy = recorder.kwargs["collection_policy"]
                    self.assertIs(policy["training_eligible"], role == "core" and semantic == "train")
                    self.assertIs(policy["development_evaluation_eligible"], role == "core" and semantic == "development")
                    self.assertIs(policy["independent_final_test"], False)
                    self.assertEqual(recorder.kwargs["split"], p["split"])
                    self.assertEqual(runner.counts["by_role"], {role: 3})
                    self.assertEqual(loaded.metadata["terminal_reason"], "collection_complete")

    def test_invalid_or_clipped_transition_stops_without_discarding_prefix(self):
        for condition in ("invalid_at", "clipped_at"):
            with self.subTest(condition=condition):
                runner, recorder, loaded, error = self.exercise_record(
                    self.plans[0], "core", **{condition: 2}, action_count=20)
                self.assertIsInstance(error, main_runner.PreflightFailure)
                self.assertEqual(len(recorder.results), 2)
                self.assertEqual(runner.counts["action_calls"], 2)
                self.assertEqual(runner.counts["physics_substeps"], 400)
                self.assertEqual(loaded.metadata["terminal_reason"], "physical_geometry_or_clipping")
                self.assertEqual(len(runner.episodes), 1)
                self.assertEqual(runner.episodes[0]["terminal_reason"], "physical_geometry_or_clipping")

    def test_resource_failure_occurs_before_recorder_or_actions(self):
        runner = main_runner.MainRunner(Path("/unused/p4-main-test"), self.design)
        with patch.object(runner, "resources", side_effect=main_runner.PreflightFailure("capacity")), \
             patch.object(main_runner, "EpisodeRecorder") as recorder:
            with self.assertRaises(main_runner.PreflightFailure):
                runner._record(None, self.plans[0], "mock", [], {}, "core")
            recorder.assert_not_called()
        self.assertEqual(runner.counts["action_calls"], 0)


if __name__ == "__main__":
    unittest.main()
