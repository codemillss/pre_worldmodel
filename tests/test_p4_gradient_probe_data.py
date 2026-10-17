"""Train-only probe selection and source reconstruction; no learned models.

Pure tests exercise selection failures and denominators. The live read-only
fixture additionally checks the existing frozen D11 files without loading a
checkpoint, performing physics, fitting statistics, or applying an update.
"""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np

from push_align.gradient_probe_data import (PROBE_NAMES, RELATIVE_RUN,
                                            build_probe_batches, contact_regimes,
                                            select_probe_slots, summarize_probes)


ROOT = Path(__file__).resolve().parents[1]


def record(slot, family, start, regimes, *, zero=False, source="new_core"):
    return {
        "slot": slot, "family_id": family, "episode_id": family + "-episode",
        "window_start": start, "actual_contact_regimes": list(regimes),
        "source": source, "role": "core" if source == "new_core" else "original_train_transition_table",
        "split": "train", "training_eligible": True,
        "y_condition": "Y1" if source == "new_core" else None,
        "stop_condition": "S1" if source == "new_core" else None,
        "all_zero_actions": zero, "all_nonzero_actions": not zero,
        "geometry": "side_pass" if set(regimes) == {"noncontact"} else "central_approach",
        "transition_ids": [source + ":" + family + ":" + str(i) for i in range(start, start + 5)],
    }


def fixture():
    transition = ["noncontact", "transition"] + ["sustained_contact"]*3
    noncontact = ["noncontact"]*5
    stop = ["transition"] + ["noncontact"]*4
    return [record(0, "new-A", 0, transition), record(1, "new-B", 0, noncontact),
            record(2, "new-A", 10, stop, zero=True), record(3, "new-B", 10, noncontact, zero=True),
            record(4, "new-C", 0, transition), record(5, "new-C", 10, stop, zero=True),
            record(6, "old-A", 0, noncontact, source="old_train"),
            record(7, "new-A", 1, transition)]


class GradientProbeSelectionTest(unittest.TestCase):
    def test_four_probes_preserve_anchor_order_and_balance_family_once(self):
        selected = select_probe_slots(fixture(), [6, 3, 0])
        self.assertEqual(tuple(selected), PROBE_NAMES)
        self.assertEqual(selected["historical_update1"], [6, 3, 0])
        self.assertEqual(selected["D11_zero5"], [2, 3, 5])
        self.assertEqual(selected["D11_transition"], [0, 4])
        self.assertEqual(selected["D11_noncontact"], [1])

    def test_selection_is_invariant_to_input_order_and_model_error_annotations(self):
        first = fixture(); second = deepcopy(first[::-1])
        for index, row in enumerate(second):
            row["model_prediction_error"] = float(1000-index)
        self.assertEqual(select_probe_slots(first, [6, 0]), select_probe_slots(second, [6, 0]))

    def test_earliest_eligible_start_uses_slot_as_final_tie_break(self):
        records = fixture()
        records[7]["window_start"] = records[0]["window_start"]
        records[7]["transition_ids"] = records[0]["transition_ids"].copy()
        self.assertEqual(select_probe_slots(records, [6])["D11_transition"], [0, 4])

    def test_development_and_ineligible_population_fail_even_outside_selected_slots(self):
        for change in ({"split": "development"}, {"training_eligible": False}):
            records = fixture(); records[7].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "Nontraining"):
                select_probe_slots(records, [6])

    def test_new_core_role_and_d11_conditions_are_strict(self):
        for change in ({"role": "warmup"}, {"y_condition": "Y0"}, {"stop_condition": "S0"}):
            records = fixture(); records[0].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "D11 core"):
                select_probe_slots(records, [6])

    def test_duplicate_missing_or_boolean_slot_is_rejected(self):
        for slot in (0, 100, True):
            records = fixture(); records[7]["slot"] = slot
            with self.subTest(slot=slot), self.assertRaises(ValueError):
                select_probe_slots(records, [6])

    def test_unknown_or_incomplete_actual_labels_are_rejected(self):
        for regimes in (["noncontact"]*4, ["noncontact"]*4+["predicted_contact"]):
            records = fixture(); records[0]["actual_contact_regimes"] = regimes
            with self.subTest(regimes=regimes), self.assertRaisesRegex(ValueError, "actual regimes"):
                select_probe_slots(records, [6])

    def test_transition_identity_cannot_repeat_inside_window(self):
        records = fixture(); records[0]["transition_ids"][1] = records[0]["transition_ids"][0]
        with self.assertRaisesRegex(ValueError, "repeats an original"):
            select_probe_slots(records, [6])

    def test_zero5_start_and_family_uniqueness_are_checked(self):
        records = fixture(); records[2]["window_start"] = 9
        with self.assertRaisesRegex(ValueError, "state10"):
            select_probe_slots(records, [6])
        records = fixture(); records[5]["family_id"] = "new-A"
        with self.assertRaisesRegex(ValueError, "Multiple zero5"):
            select_probe_slots(records, [6])

    def test_anchor_cannot_have_duplicate_unknown_or_boolean_indices(self):
        for slots in ([0, 0], [99], [True], []):
            with self.subTest(slots=slots), self.assertRaisesRegex(ValueError, "anchor"):
                select_probe_slots(fixture(), slots)

    def test_missing_supported_group_is_not_filled_by_replacement_sampling(self):
        records = fixture()
        for row in records:
            if row["source"] == "new_core":
                row["actual_contact_regimes"] = ["sustained_contact"]*5
        with self.assertRaisesRegex(ValueError, "actual train support"):
            select_probe_slots(records, [6])

    def test_counts_distinguish_occurrences_unique_transitions_and_family_overlap(self):
        records = fixture(); slots = select_probe_slots(records, [0, 7, 3])
        result = summarize_probes(records, slots, source_sha256={"fake": "a"*64})
        anchor = result["historical_update1"]
        self.assertEqual(anchor["counts"]["windows"], 3)
        self.assertEqual(anchor["counts"]["families"], 2)
        self.assertEqual(anchor["counts"]["transition_occurrences"], 15)
        self.assertEqual(anchor["counts"]["unique_original_transitions"], 11)
        self.assertEqual(anchor["overlap"]["D11_transition"]["shared_slots"], [0])
        self.assertEqual(anchor["overlap"]["D11_transition"]["shared_families"], ["new-A"])
        self.assertEqual(anchor["overlap"]["D11_transition"]["shared_unique_original_transitions"], 5)
        self.assertEqual(result["D11_zero5"]["overlap"]["D11_transition"]["shared_slots"], [])
        self.assertEqual(result["D11_zero5"]["overlap"]["D11_transition"]["shared_families"], ["new-A", "new-C"])
        json.dumps(result, allow_nan=False)

    def test_metadata_is_owned_copy_and_does_not_mutate_source_selection(self):
        records = fixture(); original = deepcopy(records)
        selected = select_probe_slots(records, [6]); source = {"f": "a"*64}
        result = summarize_probes(records, selected, source_sha256=source)
        result["D11_zero5"]["window_provenance"][0]["family_id"] = "changed"
        result["D11_zero5"]["source_sha256"]["f"] = "b"*64
        self.assertEqual(records, original)
        self.assertEqual(source, {"f": "a"*64})

    def test_force_fraction_labels_use_actual_summary_and_strict_endpoints(self):
        summary = np.zeros((20, 4), np.float64)
        summary[:, 2] = 1.
        summary[0, 2] = 0.; summary[1, 2] = np.nextafter(0., 1.)
        summary[2, 2] = np.nextafter(1., 0.)
        regimes = contact_regimes(summary)
        self.assertEqual(regimes[:4].tolist(), ["noncontact", "transition", "transition", "sustained_contact"])

    def test_invalid_contact_summary_is_rejected(self):
        for kind in ("shape", "dtype", "nan", "negative", "aboveone"):
            summary = np.zeros((20, 4), np.float64)
            if kind == "shape": summary = summary[:-1]
            if kind == "dtype": summary = summary.astype(np.float32)
            if kind == "nan": summary[0, 2] = np.nan
            if kind == "negative": summary[0, 2] = -.01
            if kind == "aboveone": summary[0, 2] = 1.01
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                contact_regimes(summary)


class FrozenD11GradientProbeDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probes = build_probe_batches(ROOT)

    def test_actual_four_batch_counts_and_train_support(self):
        self.assertEqual({key: row["counts"]["windows"] for key, row in self.probes.items()},
                         dict(zip(PROBE_NAMES, (64, 24, 16, 8))))
        self.assertEqual([row["counts"]["families"] for row in self.probes.values()], [32, 24, 16, 8])
        self.assertEqual([row["counts"]["unique_original_transitions"] for row in self.probes.values()], [300, 120, 80, 40])
        for probe in self.probes.values():
            for row in probe["window_provenance"]:
                self.assertEqual(row["split"], "train")
                self.assertIs(row["training_eligible"], True)
                self.assertNotIn("development", row["episode_id"])

    def test_anchor_is_exact_historical_order_and_all_snapshots_share_slots(self):
        trace = ROOT / "runs/p4-02-20261012-01/seeds/seed-11/SK-lambda0.1/training_trace.npz"
        with np.load(trace, allow_pickle=False) as archive:
            expected = archive["window_positions"][:64].tolist()
        self.assertEqual(self.probes["historical_update1"]["slots"], expected)
        # No checkpoint/model identity enters the batch-factory interface.
        for probe in self.probes.values():
            self.assertEqual(probe["dataset"], RELATIVE_RUN + "/prepared_training.npz")
            self.assertNotIn("checkpoint", probe)

    def test_approach_strata_are_family_balanced_and_control_geometry_is_disclosed(self):
        transition = self.probes["D11_transition"]
        control = self.probes["D11_noncontact"]
        self.assertEqual(transition["counts"]["geometry_windows"], {"central_approach": 8, "eccentric_approach": 8})
        self.assertEqual(control["counts"]["geometry_windows"], {"side_pass": 8})
        self.assertIn("not a matched causal control", control["control_limit"])
        for row in transition["window_provenance"]+control["window_provenance"]:
            self.assertEqual(row["window_start"], 0)
            self.assertIs(row["all_nonzero_actions"], True)
            self.assertEqual((row["y_condition"], row["stop_condition"]), ("Y1", "S1"))

    def test_zero5_selects_commands_not_measured_velocity(self):
        probe = self.probes["D11_zero5"]
        with np.load(ROOT / probe["dataset"], allow_pickle=False) as archive:
            actions = archive["actions"][probe["slots"]]
            starts = archive["states"][probe["slots"], 0]
        self.assertTrue(np.all(actions == 0.))
        self.assertGreater(float(np.max(np.linalg.norm(starts[:, 8:10], axis=1))), .02)
        self.assertTrue(all(row["window_start"] == 10 for row in probe["window_provenance"]))
        self.assertEqual(probe["counts"]["actual_regime_occurrences"], {"noncontact": 104, "transition": 16})

    def test_bound_raw_sources_are_only_referenced_train_cores_and_serializable(self):
        source = self.probes["D11_zero5"]["source_sha256"]
        raw = [rel for rel in source if "/episodes/" in rel]
        self.assertEqual(len(raw), 72)  #24 D11 core manifests, sidecars and NPZs.
        self.assertTrue(all("p4-data-train-" in rel and "-Y1-S1" in rel for rel in raw))
        self.assertFalse(any("warmup" in rel or "development" in rel for rel in raw))
        json.dumps(self.probes, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
