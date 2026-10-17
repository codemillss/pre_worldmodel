"""Synthetic saved histories: weighting, immutable inputs and bound probes."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from push_align.adam_history import (AdamHistoryError, LOSS_COLUMNS, PROBE_COUNTS,
                                     SNAPSHOT_IDS, summarize_histories)


class AdamHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        windows = np.full(1000, 64, dtype=np.int64)
        windows[79:100] = 16
        self.positions = np.arange(62992, dtype=np.int64) % 2960
        self.traces = {}
        for arm in ("C0", "R1"):
            one = np.ones(1000, dtype=np.float64)
            one[79:100] = 10.
            rollout = np.full(1000, 2., dtype=np.float64)
            if arm == "R1":
                one += 3.
                rollout += 2.
            teacher = np.full(1000, 2. if arm == "R1" else 0.)
            free = np.full(1000, 4. if arm == "R1" else 0.)
            total = one + .1 * rollout + (.2 * (teacher + .1 * free) if arm == "R1" else 0.)
            norm = np.tile(np.arange(5, dtype=np.float64), 200) + (0.5 if arm == "R1" else 0.)
            self.traces[arm] = {"window_positions": self.positions.copy(),
                "batch_offsets": np.r_[np.int64(0), np.cumsum(windows)],
                "batch_data_hashes": np.asarray([sha256(str(i).encode()).hexdigest() for i in range(1000)]),
                "losses": np.column_stack((one, rollout, teacher, free, total, norm)),
                "loss_columns": np.asarray(LOSS_COLUMNS), "update_wall_time_s": np.zeros(1000)}
        self.registration = {"schema": "p4-gradient-probe-protocol-v1", "snapshots": [],
                             "batch_counts": {}, "batch_selections": {}}
        shots = []
        for index, name in enumerate(SNAPSHOT_IDS):
            branch = None if index == 0 else "C0" if index < 4 else "R1"
            updates = 0 if index == 0 else (100, 500, 1000)[(index - 1) % 3]
            shot = {"snapshot_id": name, "branch": branch, "additional_updates": updates,
                    "checkpoint_path": f"frozen/{name}", "checkpoint_hash": "1" * 64,
                    "trained_objective": f"saved objective {name}"}
            self.registration["snapshots"].append(deepcopy(shot))
            shots.append({**shot, "model_tree_hash": "2" * 64, "checkpoint_json_sha256": "3" * 64,
                          "files_sha256": {"weights.pt": "4" * 64, "training_state.pt": "5" * 64}})
        self.batches = {}
        for name, count in PROBE_COUNTS.items():
            slots = self.positions[:count].tolist()
            counts = {"windows": count, "families": count}
            self.batches[name] = {"name": name, "slots": slots, "counts": counts,
                                  "selection": "fixed synthetic selection", "overlap": {}}
            self.registration["batch_counts"][name] = deepcopy(counts)
            self.registration["batch_selections"][name] = slots.copy()
        self.manifest = {"schema": "p4-gradient-probe-manifest-v1", "model_snapshots": shots,
                         "normalizer_hash": "6" * 64, "dataset_hash": "7" * 64,
                         "probe_batches_sha256": "8" * 64}
        self.aggregate = {"schema": "p4-gradient-probe-aggregate-v1", "row_count": 28, "rows": []}
        for shot in shots:
            for name, count in PROBE_COUNTS.items():
                self.aggregate["rows"].append({"snapshot_id": shot["snapshot_id"], "probe_name": name,
                    "trained_objective": shot["trained_objective"], "batch_count": count,
                    "summary_path": f"probes/{shot['snapshot_id']}/{name}/summary.json",
                    "loss_values": {"one_step": 1., "rollout": 2., "relative_teacher": 3.,
                                    "relative_rollout": 4., "total": 1.88}})
        self.save()

    def write_json(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
        return sha256(path.read_bytes()).hexdigest()

    def save(self):
        for arm, values in self.traces.items():
            np.savez_compressed(self.root / f"{arm}.npz", **values)
        self.manifest["protocol_sha256"] = self.write_json("registration.json", self.registration)
        self.manifest["batch_manifest_sha256"] = self.write_json("batch_manifest.json", self.batches)
        self.write_json("manifest.json", self.manifest)
        self.write_json("aggregate.json", self.aggregate)

    def summarize(self):
        return summarize_histories(self.root / "C0.npz", self.root / "R1.npz", self.root / "aggregate.json")

    def test_partial_batch_weighting_does_not_treat_sixteen_windows_as_sixtyfour(self):
        result = self.summarize()
        first = result["arms"]["C0"]["blocks"][0]
        means = first["losses"]["one_step"]
        self.assertAlmostEqual(means["update_mean"], (21 * 10 + 79) / 100)
        self.assertAlmostEqual(means["window_weighted_mean"], (21 * 16 * 10 + 79 * 64) / (21 * 16 + 79 * 64))
        self.assertGreater(means["update_mean"], means["window_weighted_mean"])
        self.assertEqual(first["batch_counts"], {"full64": 79, "partial16": 21})
        self.assertEqual(first["denominators"]["transition_occurrences"], (21 * 16 + 79 * 64) * 5)

    def test_all_ten_blocks_and_whole_denominators_cover_every_update(self):
        result = self.summarize()
        for arm in ("C0", "R1"):
            blocks = result["arms"][arm]["blocks"]
            self.assertEqual([(b["update_start"], b["update_end"]) for b in blocks],
                             [(i + 1, i + 100) for i in range(0, 1000, 100)])
            whole = result["arms"][arm]["whole"]
            self.assertEqual(whole["denominators"]["window_occurrences"], 62992)
            self.assertEqual(whole["denominators"]["transition_occurrences"], 314960)
            self.assertEqual(whole["denominators"]["base_scalar_terms_per_term"], 3149600)
            self.assertEqual(sum(b["denominators"]["window_occurrences"] for b in blocks), 62992)

    def test_zero_c0_placeholders_become_null_not_false_zero_error(self):
        c0 = self.summarize()["arms"]["C0"]
        self.assertEqual(c0["relative_loss_status"], "NOT_COMPUTED_BETA_ZERO_PLACEHOLDERS")
        for section in [c0["whole"], *c0["blocks"]]:
            self.assertIsNone(section["denominators"]["relative_scalar_terms_per_term"])
            for key in ("relative_teacher", "relative_free", "weighted_relative_teacher",
                        "weighted_relative_free", "weighted_aux"):
                self.assertIsNone(section["losses"][key])

    def test_paired_difference_only_uses_shared_objective_components(self):
        result = self.summarize()
        paired = result["paired_shared_base"]
        self.assertFalse(paired["unlike_total_objectives_compared"])
        for section in [paired["whole"], *paired["blocks"]]:
            self.assertEqual(set(section["losses"]), {"one_step", "weighted_rollout", "base"})
            self.assertAlmostEqual(section["losses"]["base"]["update_mean"], 3.2)
            self.assertAlmostEqual(section["losses"]["base"]["window_weighted_mean"], 3.2)
        r1 = result["arms"]["R1"]["whole"]
        self.assertAlmostEqual(r1["losses"]["weighted_aux"]["update_mean"], .48)
        self.assertEqual(r1["denominators"]["relative_scalar_terms_per_term"], 629920)

    def test_clipping_frequency_is_update_based_strict_threshold_and_linear_p90(self):
        result = self.summarize()
        c0 = result["arms"]["C0"]["whole"]["clipping"]
        self.assertEqual((c0["count"], c0["fraction"]), (600, .6))
        self.assertEqual((c0["gradient_norm_median"], c0["gradient_norm_p90"]), (2., 4.))
        self.assertEqual(c0["quantile_method"], "linear")
        self.assertEqual(result["arms"]["R1"]["whole"]["clipping"]["count"], 800)

    def test_all28_probes_and_bound_frozen_identities_are_retained_without_population_average(self):
        probes = self.summarize()["fixed_probe_losses"]
        self.assertEqual(probes["row_count"], 28)
        self.assertEqual(probes["snapshot_identities"], self.manifest["model_snapshots"])
        self.assertIsNone(probes["population_average"])
        self.assertIsNone(probes["statistical_CI"])
        self.assertEqual([(r["snapshot_id"], r["probe_name"]) for r in probes["rows"]],
                         [(r["snapshot_id"], r["probe_name"]) for r in self.aggregate["rows"]])
        row = probes["rows"][0]
        self.assertAlmostEqual(row["base"], 1.2)
        self.assertAlmostEqual(row["weighted_aux"], .68)
        self.assertAlmostEqual(row["common_hypothetical_total"], 1.88)

    def test_only_saved_inputs_are_read_and_files_remain_byte_identical(self):
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        result = self.summarize()
        self.assertTrue(result["source_files_unchanged"])
        self.assertFalse(result["new_model_or_gradient_or_optimizer_or_physics_execution"])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})
        self.assertEqual(len(result["source_identities"]), 6)
        self.assertIsInstance(json.dumps(result, allow_nan=False), str)

    def test_rejects_trace_columns_dtype_nonfinite_and_uncomputed_nonzero(self):
        for modification, message in (("columns", "columns"), ("dtype", "float64"),
                                       ("nan", "finite"), ("placeholder", "placeholders")):
            with self.subTest(modification=modification):
                original = deepcopy(self.traces["C0"])
                if modification == "columns": self.traces["C0"]["loss_columns"] = np.asarray(LOSS_COLUMNS[::-1])
                if modification == "dtype": self.traces["C0"]["losses"] = self.traces["C0"]["losses"].astype(np.float32)
                if modification == "nan": self.traces["C0"]["losses"][0, 0] = np.nan
                if modification == "placeholder": self.traces["C0"]["losses"][0, 2] = 1.
                self.save()
                with self.assertRaisesRegex(AdamHistoryError, message): self.summarize()
                self.traces["C0"] = original

    def test_mismatched_batch_order_hashes_offsets_and_population_are_rejected(self):
        for modification, message in (("order", "window_positions"), ("hash", "batch_data_hashes"),
                                       ("offset", "batches"), ("population", "2960")):
            with self.subTest(modification=modification):
                original = deepcopy(self.traces["R1"])
                if modification == "order": self.traces["R1"]["window_positions"][[0, 1]] = self.positions[[1, 0]]
                if modification == "hash": self.traces["R1"]["batch_data_hashes"][0] = "f" * 64
                if modification == "offset": self.traces["R1"]["batch_offsets"][1] += 1
                if modification == "population": self.traces["R1"]["window_positions"][self.positions == 2959] = 2958
                self.save()
                with self.assertRaisesRegex(AdamHistoryError, message): self.summarize()
                self.traces["R1"] = original

    def test_mismatched_historical_objective_arithmetic_is_rejected(self):
        self.traces["R1"]["losses"][500, 4] += 1.
        self.save()
        with self.assertRaisesRegex(AdamHistoryError, "historical total"): self.summarize()

    def test_duplicate_missing_or_invalid_fixed_probe_rows_are_rejected(self):
        for modification, message in (("duplicate", "duplicated"), ("missing", "28"),
                                       ("negative", "nonnegative"), ("total", "objective")):
            with self.subTest(modification=modification):
                original = deepcopy(self.aggregate)
                if modification == "duplicate": self.aggregate["rows"][-1] = deepcopy(self.aggregate["rows"][0])
                if modification == "missing": self.aggregate["rows"].pop()
                if modification == "negative": self.aggregate["rows"][0]["loss_values"]["one_step"] = -1.
                if modification == "total": self.aggregate["rows"][0]["loss_values"]["total"] = 9.
                self.save()
                with self.assertRaisesRegex(AdamHistoryError, message): self.summarize()
                self.aggregate = original

    def test_manifest_hash_and_checkpoint_identity_tampering_is_rejected(self):
        self.manifest["model_snapshots"][1]["checkpoint_hash"] = "a" * 64
        self.save()
        with self.assertRaisesRegex(AdamHistoryError, "checkpoint identity"): self.summarize()
        self.manifest["model_snapshots"][1]["checkpoint_hash"] = "1" * 64
        self.save()
        registration_path = self.root / "registration.json"
        registration_path.write_bytes(registration_path.read_bytes() + b"\n")
        with self.assertRaisesRegex(AdamHistoryError, "source binding"): self.summarize()

    def test_historical_first_probe_must_match_saved_first_batch(self):
        self.batches["historical_update1"]["slots"][0] = 100
        self.registration["batch_selections"]["historical_update1"][0] = 100
        self.save()
        with self.assertRaisesRegex(AdamHistoryError, "first training batch"): self.summarize()


if __name__ == "__main__":
    unittest.main()
