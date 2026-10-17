"""Synthetic saved-array tests; no model or forward is constructed/executed."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests import test_p4_finite_step_audit as finite_tests
from tools import check_p4_cross_batch as oracle


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def observations():
    return [{"snapshot_id": str(index), "source_probe_name": source, "case_status": "PASS",
             "change_relations": ["NUMERIC_TIE", "INCREASE", "DECREASE", "NUMERIC_TIE"]}
            for index in range(7) for source in oracle.SOURCES]


class PrimaryGatingTests(unittest.TestCase):
    def test_all_group_denominators_and_per_alpha_relations(self):
        result = oracle.primary_counts(observations(), True)
        self.assertEqual((result["all"]["case_count"], result["all"]["planned_positive_endpoints"]), (28, 84))
        self.assertEqual(result["self_stop"]["planned_positive_endpoints"], 21)
        self.assertEqual(result["other_sources"]["planned_positive_endpoints"], 63)
        self.assertEqual(result["all"]["by_alpha"]["0.25"]["INCREASE"], 28)
        self.assertEqual(result["all"]["by_alpha"]["0.5"]["DECREASE"], 28)
        self.assertEqual(result["all"]["by_alpha"]["1.0"]["NUMERIC_TIE"], 28)
        for group in result["by_source_probe"].values():
            self.assertEqual(group["case_count"], 7)
            self.assertTrue(all(sum(row.values()) == 7 for row in group["by_alpha"].values()))

    def test_any_held_case_holds_all_three_positive_endpoints(self):
        for status in ("NUMERICAL_FAILURE", "REPRODUCTION_FAILURE"):
            rows = observations(); rows[0]["case_status"] = status
            counts = oracle.primary_counts(rows, True)
            self.assertEqual(counts["all"]["planned_positive_endpoints"], 84)
            for alpha in ("0.25", "0.5", "1.0"):
                self.assertEqual(counts["all"]["by_alpha"][alpha]["NOT_INTERPRETED"], 1)
                self.assertEqual(sum(counts["all"]["by_alpha"][alpha].values()), 28)

    def test_global_control_failure_makes_all_primary_counts_null(self):
        self.assertIsNone(oracle.primary_counts(observations(), False))

    def test_missing_duplicate_or_wrong_source_never_shrinks_the_denominator(self):
        variants = [observations()[:-1], observations()]
        variants[1][-1] = deepcopy(variants[1][0])
        for rows in variants:
            with self.assertRaises(ValueError): oracle.primary_counts(rows, True)

    def test_inconclusive_secondary_decision_does_not_hide_target_increases(self):
        rows = observations()
        for row in rows: row["secondary_decision"] = "INCONCLUSIVE"
        self.assertEqual(oracle.primary_counts(rows, True)["all"]["by_alpha"]["0.25"]["INCREASE"], 28)


class SavedControlTests(unittest.TestCase):
    def fixture(self):
        rows = [{"status": "PASS", "loss_values": {"one_step": 1., "rollout": 0., "relative_teacher": 0.,
                 "relative_rollout": 0., "total": 1., "base": 1., "weighted_aux": 0., "common_total": 1.}} for _ in range(4)]
        predictions = {"teacher_predictions": np.zeros((4, 10, 10), np.float32),
                       "free_predicted_states": np.zeros((4, 2, 6, 10), np.float64)}
        return {"evaluations": rows}, predictions

    def test_self_stop_all_four_outputs_reproduce_exactly(self):
        e, p = self.fixture(); result = oracle.stop_controls(e, p, deepcopy(e), deepcopy(p), True)
        self.assertTrue(result["alpha0_target_baseline_reproduced"] and result["self_stop_control_pass"])

    def test_alpha0_exact_control_is_stricter_than_raw_reference_tolerance(self):
        e, p = self.fixture(); previous = deepcopy(e); e["evaluations"][0]["loss_values"]["one_step"] += 1e-12
        self.assertTrue(oracle.finite.reproduction(oracle.canonical_losses(e["evaluations"][0]["loss_values"]), previous["evaluations"][0]["loss_values"])["pass"])
        self.assertFalse(oracle.stop_controls(e, p, previous, p, False)["alpha0_target_baseline_reproduced"])

    def test_alpha0_prediction_mutation_fails_even_if_losses_match(self):
        e, p = self.fixture(); previous = deepcopy(p); p["teacher_predictions"][0, 0, 0] = 1e-8
        self.assertFalse(oracle.stop_controls(e, p, e, previous, False)["alpha0_target_baseline_reproduced"])

    def test_nonself_positive_effects_are_not_self_control_failures(self):
        e, p = self.fixture(); previous = deepcopy(p); p["teacher_predictions"][3, 0, 0] = 1.
        result = oracle.stop_controls(e, p, e, previous, False)
        self.assertTrue(result["alpha0_target_baseline_reproduced"])
        self.assertIsNone(result["self_stop_control_pass"])

    def test_self_positive_loss_status_or_prediction_change_holds_global_control(self):
        for kind in ("loss", "status", "prediction"):
            e, p = self.fixture(); old_e, old_p = deepcopy(e), deepcopy(p)
            if kind == "loss": e["evaluations"][3]["loss_values"]["total"] += 1e-12
            elif kind == "status": e["evaluations"][3]["status"] = "NUMERICAL_FAILURE"
            else: p["free_predicted_states"][3, 0, 1, 0] = 1e-10
            self.assertFalse(oracle.stop_controls(e, p, old_e, old_p, True)["self_stop_control_pass"])

    def test_typed_array_identity_binds_precision_shape_and_coordinate_bytes(self):
        a = np.array([1., 2.], np.float32)
        expected = hashlib.sha256(b'{"dtype":"float32","shape":[2]}\0' + a.tobytes()).hexdigest()
        self.assertEqual(oracle.array_sha256(a), expected)
        self.assertNotEqual(oracle.array_sha256(a), oracle.array_sha256(a.astype(np.float64)))
        self.assertNotEqual(oracle.array_sha256(a), oracle.array_sha256(a.reshape(1, 2)))
        self.assertNotEqual(oracle.array_sha256(a), oracle.array_sha256(a[::-1]))


class SavedCrossCaseTests(unittest.TestCase):
    def fixture(self, directory, source="historical_update1"):
        root, _, old_case, spec, weights, _, old_folder, record = finite_tests.SavedSyntheticCaseAuditTests().fixture(directory)
        snapshot = spec["snapshot_id"]; order = record["parameter_order"]; model_hash = spec["model_tree_hash"]
        output = oracle.arrays(old_folder / "parameter_displacements.npz")
        old_predictions = oracle.arrays(old_folder / "predictions.npz")
        predictions = {"teacher_predictions": np.repeat(old_predictions["teacher_predictions"], 12, axis=1),
                       "free_predicted_states": np.repeat(old_predictions["free_predicted_states"], 12, axis=1)}
        old_gradient_folder = root / Path(old_case["original_gradient_summary"]).parent
        old_adam_folder = root / Path(old_case["source_direction_npz"]).parent
        target = root / f"runs/p4-gradient-probe-20261015-01/probes/{snapshot}/{oracle.TARGET}"
        adam = root / f"runs/p4-adam-probe-20261015-01/probes/{snapshot}/{source}"
        target.mkdir(parents=True); adam.mkdir(parents=True)
        gradient = oracle.arrays(old_gradient_folder / "gradients.npz")
        np.savez_compressed(target / "gradients.npz", **gradient)
        np.savez_compressed(adam / "directions.npz", **oracle.arrays(old_adam_folder / "directions.npz"))
        reference = json.loads((old_gradient_folder / "summary.json").read_text())["loss_values"]
        write(target / "summary.json", {"snapshot_id": snapshot, "probe_name": oracle.TARGET,
            "parameter_order": order, "loss_values": reference, "before_weights_hash": model_hash})
        write(adam / "summary.json", {"snapshot_id": snapshot, "probe_name": source, "source_weights_tree_hash": model_hash,
            "parameter_order": order, "directions": [{}, {}, {}, {"objective": "total", "mode": "KEEP"}]})
        e = record["evaluation"]; denominators = {"batch_windows": 24, "valid_transitions": 120, "base_scalar_terms": 1200, "relative_scalar_terms": 240}
        e["denominators"] = denominators; e["counters"]["root_model_forward_rows"] = 960
        trace = []
        for index, row in enumerate(e["evaluations"]):
            row["denominators"] = denominators; row["forward_rows"] = [120, 24, 24, 24, 24, 24]; row["forward_total_rows"] = 240
            h = oracle.parameter_tree_hash(output["evaluated_parameters"][index], order)
            trace += [{"rows": size, "parameter_tree_hash": h} for size in row["forward_rows"]]
        previous_source = root / f"runs/p4-finite-step-20261015-01/cases/{snapshot}/{source}"
        previous_stop = root / f"runs/p4-finite-step-20261015-01/cases/{snapshot}/{oracle.TARGET}"
        for folder, name in ((previous_source, source), (previous_stop, oracle.TARGET)):
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / "parameter_displacements.npz", **output)
            np.savez_compressed(folder / "predictions.npz", **predictions)
            previous = deepcopy(record); previous["probe_name"] = name
            previous["source"]["source_direction_npz"] = str((adam / "directions.npz").relative_to(root))
            previous["parameter_arrays_sha256"] = oracle.sha(folder / "parameter_displacements.npz")
            previous["prediction_arrays_sha256"] = oracle.sha(folder / "predictions.npz")
            write(folder / "summary.json", previous)
        paths = {"source_direction_npz": adam / "directions.npz", "source_direction_summary": adam / "summary.json",
            "target_gradient_summary": target / "summary.json", "target_gradient_npz": target / "gradients.npz",
            "previous_finite_source_summary": previous_source / "summary.json",
            "previous_finite_source_parameters": previous_source / "parameter_displacements.npz",
            "previous_finite_stop_summary": previous_stop / "summary.json",
            "previous_finite_stop_parameters": previous_stop / "parameter_displacements.npz",
            "previous_finite_stop_predictions": previous_stop / "predictions.npz"}
        case = {"snapshot_id": snapshot, "source_probe_name": source, "target_probe_name": oracle.TARGET, "source_direction_index": 3}
        for key, path in paths.items(): case[key] = str(path.relative_to(root)); case[key + "_sha256"] = oracle.sha(path)
        states = np.zeros((24, 6, 10), np.float64); actions = np.zeros((24, 5, 2), np.float64); mask = np.ones((24, 5), np.bool_)
        delta = oracle.arrays(adam / "directions.npz")["directions"][3]
        bindings = oracle.routing_bindings(snapshot, model_hash, order, source, delta, gradient["gradients"][4], reference, states, actions, mask)
        e["cross_batch_routing"] = oracle.routing_certificate(bindings, e["decision"])
        control = {"source_parameter_maps_previous_exact": True, "alpha0_matches_previous_stop": True,
            "alpha0_matches_snapshot_first_source": True, "self_stop_previous": None if source != oracle.TARGET else {"pass": True,
                "checks": {"all_four_loss_values_and_statuses_exact": True, "teacher_predictions_exact": True, "free_predictions_exact": True}}}
        run = root / "runs/p4-cross-batch-20261015-01"; folder = run / "cases" / snapshot / source; folder.mkdir(parents=True)
        np.savez_compressed(folder / "parameter_displacements.npz", **output)
        np.savez_compressed(folder / "predictions.npz", **predictions)
        record = {"snapshot_id": snapshot, "source_probe_name": source, "target_probe_name": oracle.TARGET,
            "source": case, "parameter_order": order, "evaluation": e, "controls": control, "source_forward_trace": trace,
            "source_weights_tree_hash": model_hash, "source_optimizer_tree_hash": "synthetic-no-optimizer",
            "parameter_arrays_sha256": oracle.sha(folder / "parameter_displacements.npz"), "prediction_arrays_sha256": oracle.sha(folder / "predictions.npz")}
        write(folder / "summary.json", record)
        probes = {oracle.TARGET + "_states": states, oracle.TARGET + "_actions": actions, oracle.TARGET + "_mask": mask}
        if source != oracle.TARGET:
            probes.update({source + "_states": np.full((2, 6, 10), 99., np.float64), source + "_actions": np.zeros((2, 5, 2)), source + "_mask": np.ones((2, 5), np.bool_)})
        return root, run, case, spec, weights, probes, folder, record

    def check_case(self, fixture):
        root, run, case, spec, weights, probes, _, _ = fixture
        with patch.object(oracle, "ROOT", root):
            checker = oracle.Audit(); checker.case(run, case, spec, weights, "synthetic-no-optimizer", probes,
                {"delta_mean": np.zeros(10), "delta_scale": np.ones(10)})
        return checker

    def test_nonself_case_uses_fixed24_target_not_poisoned_source_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            checker = self.check_case(self.fixture(directory))
            self.assertEqual(checker.successful_evaluations, 4); self.assertEqual(checker.forward_rows, 960)
            self.assertTrue(checker.controls[0]["alpha0_target_baseline_reproduced"])

    def test_self_stop_case_reproduces_prior_all_four_saved_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            checker = self.check_case(self.fixture(directory, oracle.TARGET))
            self.assertTrue(checker.controls[0]["self_stop_control_pass"])

    def test_target_path_cannot_route_to_a_source_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory); fixture[2]["target_gradient_npz"] = fixture[2]["source_direction_npz"]
            with self.assertRaisesRegex(ValueError, "exact_source_target_previous_case_paths"): self.check_case(fixture)

    def test_target_gradient_binding_cannot_claim_source_gradient_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory); record, folder = fixture[-1], fixture[-2]
            record["evaluation"]["cross_batch_routing"]["target_gradient"]["gradient_sha256"] = "0" * 64
            write(folder / "summary.json", record)
            with self.assertRaisesRegex(ValueError, "independent_source_target_gradient_reference_routing"): self.check_case(fixture)

    def test_wrong_target_loss_is_rejected_from_saved_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory); record, folder = fixture[-1], fixture[-2]
            record["evaluation"]["evaluations"][1]["loss_values"]["one_step"] += .01
            write(folder / "summary.json", record)
            with self.assertRaisesRegex(ValueError, "independent_saved_prediction_losses"): self.check_case(fixture)

    def test_Taylor_uses_target_gradient_even_when_rebound_source_gradient_is_finite(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory); root, _, case, _, _, _, folder, record = fixture
            path = root / case["target_gradient_npz"]; values = oracle.arrays(path); values["gradients"][4] *= 3
            np.savez_compressed(path, **values); case["target_gradient_npz_sha256"] = oracle.sha(path)
            routing = record["evaluation"]["cross_batch_routing"]; digest = oracle.array_sha256(values["gradients"][4])
            routing["target_gradient"]["gradient_sha256"] = routing["binding_records"]["gradient"]["gradient_sha256"] = digest
            record["source"] = case; write(folder / "summary.json", record)
            with self.assertRaisesRegex(ValueError, "three_distinct_directional_predictions"): self.check_case(fixture)

    def test_previous_source_parameter_map_mismatch_is_hard_contract_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory); root, _, case, _, _, _, folder, record = fixture
            path = root / case["previous_finite_source_parameters"]; values = oracle.arrays(path); values["evaluated_parameters"][1, 0] += .1
            np.savez_compressed(path, **values); case["previous_finite_source_parameters_sha256"] = oracle.sha(path)
            previous = oracle.read(root / case["previous_finite_source_summary"]); previous["parameter_arrays_sha256"] = oracle.sha(path)
            write(root / case["previous_finite_source_summary"], previous); case["previous_finite_source_summary_sha256"] = oracle.sha(root / case["previous_finite_source_summary"])
            record["source"] = case; write(folder / "summary.json", record)
            with self.assertRaisesRegex(ValueError, "parameter_maps_bitwise_equal_previous_finite_source"): self.check_case(fixture)

    def test_actual_root_trace_must_bind_the_evaluated_parameter_map(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory); record, folder = fixture[-1], fixture[-2]
            record["source_forward_trace"][6]["parameter_tree_hash"] = "0" * 64; write(folder / "summary.json", record)
            with self.assertRaisesRegex(ValueError, "actual_source_forwards_used_evaluated_parameter_maps"): self.check_case(fixture)

    def test_case_audit_calls_no_model_gradient_optimizer_or_physics(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            with patch.object(torch.nn.Module, "__init__", side_effect=AssertionError("no model")), \
                 patch.object(torch.nn.Module, "_call_impl", side_effect=AssertionError("no forward")), \
                 patch.object(torch.autograd, "grad", side_effect=AssertionError("no derivative")), \
                 patch.object(torch.optim.Adam, "__init__", side_effect=AssertionError("no optimizer")):
                self.assertEqual(self.check_case(fixture).forward_calls, 24)


if __name__ == "__main__":
    unittest.main()
