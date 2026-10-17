"""Independent finite-step oracle tests; no learned model or model forward."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.check_p4_finite_step import (ALPHAS, change_relation, classify_case,
    loss_oracle, parameter_arithmetic, reproduction, state_difference, taylor_metrics)
from tools import check_p4_finite_step as oracle


def losses(total):
    return {"one_step": float(total), "rollout": 0., "relative_teacher": 0.,
            "relative_free": 0., "total": float(total), "base": float(total), "aux": 0.}


REFERENCE = {"one_step": 1., "rollout": 0., "relative_teacher": 0.,
             "relative_rollout": 0., "total": 1.}


class ParameterAndTaylorOracleTests(unittest.TestCase):
    def test_each_alpha_starts_from_same_original_and_inputs_stay_exact(self):
        original, direction = np.array([.1, -3., 0.], np.float32), np.array([2., 5., -1.], np.float32)
        before = original.copy(), direction.copy()
        result = parameter_arithmetic(original, direction)
        self.assertEqual(result["evaluated_parameters"].shape, (4, 3))
        np.testing.assert_array_equal(result["evaluated_parameters"][0], original)
        np.testing.assert_array_equal(result["rounded_parameter_differences"][0], np.zeros(3, np.float32))
        for index, alpha in enumerate(ALPHAS):
            np.testing.assert_array_equal(result["evaluated_parameters"][index], original + np.float32(alpha) * direction)
        self.assertFalse(np.array_equal(result["evaluated_parameters"][2], result["evaluated_parameters"][1] + .5 * direction))
        np.testing.assert_array_equal(original, before[0]); np.testing.assert_array_equal(direction, before[1])

    def test_float32_weight_addition_can_erase_nominal_motion(self):
        result = parameter_arithmetic(np.array([1e8], np.float32), np.array([1.], np.float32))
        np.testing.assert_array_equal(result["rounded_parameter_differences"], np.zeros((4, 1), np.float32))
        self.assertEqual(float(result["evaluated_parameters"][-1, 0]), 1e8)

    def test_registered_rounded_primary_stays_distinct_from_exact_endpoint(self):
        original, direction, gradient = np.array([1e-8], np.float32), np.array([1.], np.float32), np.array([1000.], np.float32)
        result = parameter_arithmetic(original, direction)
        metric = taylor_metrics(original, result["evaluated_parameters"][-1], result["rounded_parameter_differences"][-1],
                                direction, gradient, 1., 2., 3.)
        self.assertEqual(metric["primary_first_order_change"], 1000.)
        self.assertLess(metric["endpoint_float64_first_order_change"], 1000.)
        self.assertGreater(metric["primary_vs_endpoint_float64_change_difference"], 0.)
        self.assertEqual(metric["primary_taylor_loss"], 1002.)
        self.assertEqual(metric["actual_minus_primary_taylor"], -999.)

    def test_nominal_and_actual_Taylor_use_their_own_displacements(self):
        original, direction, gradient = np.array([1e8], np.float32), np.array([1.], np.float32), np.array([-2.], np.float32)
        result = parameter_arithmetic(original, direction)
        metric = taylor_metrics(original, result["evaluated_parameters"][2], result["rounded_parameter_differences"][2],
                                direction, gradient, .5, 1., 1.)
        self.assertEqual(metric["nominal_first_order_change"], -1.)
        self.assertEqual(metric["primary_first_order_change"], 0.)
        self.assertEqual(metric["primary_taylor_loss"], 1.)
        self.assertEqual(metric["nominal_taylor_loss"], 0.)

    def test_copy_arithmetic_matches_tensor_only_float32_oracle(self):
        import torch
        generator = np.random.default_rng(882)
        original = generator.normal(0, .2, 313).astype(np.float32)
        direction = generator.normal(0, .01, 313).astype(np.float32)
        result = parameter_arithmetic(original, direction)
        for index, alpha in enumerate(ALPHAS):
            expected = torch.from_numpy(original.copy()) + torch.from_numpy(direction.copy()) * alpha
            np.testing.assert_array_equal(result["evaluated_parameters"][index], expected.numpy())
            np.testing.assert_array_equal(result["rounded_parameter_differences"][index], (expected - torch.from_numpy(original)).numpy())

    def test_parameter_input_validation(self):
        one = np.ones(2, np.float32)
        for original, direction in ((one.astype(np.float64), one), (one, one[:1]), (one * np.nan, one), (one[:, None], one[:, None])):
            with self.assertRaises(ValueError): parameter_arithmetic(original, direction)


class SavedPredictionLossOracleTests(unittest.TestCase):
    def fixture(self):
        states = np.zeros((2, 6, 10), np.float64)
        return states, np.ones((2, 5), np.bool_), np.zeros((10, 10), np.float32), states.copy(), np.zeros(10), np.ones(10)

    def test_zero_predictions_and_targets_have_zero_loss_with_full_denominators(self):
        result = loss_oracle(*self.fixture())
        self.assertTrue(all(value == 0 for value in result["losses"].values()))
        self.assertEqual(result["denominators"], {"valid_transitions": 10, "base_scalar_terms": 100, "relative_scalar_terms": 20})

    def test_teacher_free_and_relative_terms_have_different_exact_denominators(self):
        states, mask, teacher, free, mean, scale = self.fixture()
        teacher[:, 0], teacher[:, 6] = 1., 3.
        free[:, 1:, 0], free[:, 1:, 6] = .5, 1.5
        result = loss_oracle(states, mask, teacher, free, mean, scale)["losses"]
        for key, expected in {"one_step": 1., "rollout": .25, "relative_teacher": 1., "relative_free": .25,
                              "base": 1.025, "aux": .205, "total": 1.23}.items():
            self.assertAlmostEqual(result[key], expected, places=13)

    def test_teacher_output_is_denormalized_with_original_mean_and_scale(self):
        states, mask, teacher, free, mean, scale = self.fixture()
        mean[0], mean[6] = .001, .002
        for step in range(6):
            states[:, step, 0], states[:, step, 6] = step * .001, step * .002
        free[:] = states
        result = loss_oracle(states, mask, teacher, free, mean, scale)
        self.assertLess(result["losses"]["total"], 1e-30)

    def test_principal_yaw_residual_uses_2pi_not_pi_symmetry(self):
        initial, endpoint = np.zeros((1, 10)), np.zeros((1, 10))
        initial[0, 2], endpoint[0, 2] = np.pi - .1, -np.pi + .1
        self.assertAlmostEqual(state_difference(initial, endpoint)[0, 2], .2, places=14)
        endpoint[0, 2] = initial[0, 2] + np.pi
        self.assertAlmostEqual(state_difference(initial, endpoint)[0, 2], -np.pi, places=14)

    def test_rollout_k0_and_frozen_mask_cannot_change_denominator(self):
        fixture = list(self.fixture()); fixture[3][0, 0, 0] = .1
        with self.assertRaises(ValueError): loss_oracle(*fixture)
        fixture = list(self.fixture()); fixture[1][0, -1] = False
        with self.assertRaises(ValueError): loss_oracle(*fixture)

    def test_saved_prediction_precision_nonfinite_and_normalizer_fail_closed(self):
        for index, replacement in ((2, np.zeros((10, 10), np.float64)), (2, np.full((10, 10), np.nan, np.float32)),
                                   (5, np.zeros(10)), (4, np.full(10, np.inf))):
            fixture = list(self.fixture()); fixture[index] = replacement
            with self.assertRaises(ValueError): loss_oracle(*fixture)


class ReproductionAndDecisionOracleTests(unittest.TestCase):
    def test_alpha0_checks_every_raw_term_with_strict_tolerance(self):
        self.assertTrue(reproduction(losses(1.), REFERENCE)["pass"])
        changed = losses(1.); changed["relative_teacher"] = 1e-8
        self.assertFalse(reproduction(changed, REFERENCE)["pass"])
        self.assertFalse(reproduction(losses(1.000001), REFERENCE)["pass"])

    def test_tie_band_boundaries_are_inclusive(self):
        for value in (-1e-6, 0., 1e-6): self.assertEqual(change_relation(value, 1e-6), "NUMERIC_TIE")
        self.assertEqual(change_relation(1.01e-6, 1e-6), "INCREASE")
        self.assertEqual(change_relation(-1.01e-6, 1e-6), "DECREASE")

    def test_nominal_increase_and_either_smaller_decrease_support_fixed_case_only(self):
        for values in ((1., .9, 1.1, 1.2), (1., 1.1, .9, 1.2)):
            result = classify_case([losses(value) for value in values], REFERENCE)
            self.assertEqual(result["verdict"], "NOMINAL_STEP_OVERSHOOT_SUPPORTED")
            self.assertEqual(result["retained_alpha_count"], 4)

    def test_all_positive_alphas_decrease(self):
        result = classify_case([losses(value) for value in (1., .99, .98, .97)], REFERENCE)
        self.assertEqual(result["verdict"], "ALL_POSITIVE_ALPHAS_DECREASE")
        self.assertEqual(result["numeric_tie_margin"], 1e-6)

    def test_mixed_increase_or_tie_patterns_stay_inconclusive(self):
        for values in ((1., 1.1, 1.2, 1.3), (1., 1.1, 1.2, .9), (1., 1., 1., 1.)):
            self.assertEqual(classify_case([losses(value) for value in values], REFERENCE)["verdict"], "INCONCLUSIVE")

    def test_reproduction_failure_cannot_be_overridden_by_apparent_overshoot(self):
        result = classify_case([losses(value) for value in (1.0001, .9, .9, 1.2)], REFERENCE)
        self.assertEqual(result["verdict"], "REPRODUCTION_FAILURE")

    def test_numerical_failure_keeps_all_four_rows(self):
        rows = [losses(1.), losses(.9), None, losses(1.2)]
        result = classify_case(rows, REFERENCE)
        self.assertEqual(result["verdict"], "NUMERICAL_FAILURE")
        self.assertEqual(result["retained_alpha_count"], 4)
        self.assertIsNone(result["total_loss_changes"][2])
        with self.assertRaises(ValueError): classify_case(rows[:3], REFERENCE)


class SavedSyntheticCaseAuditTests(unittest.TestCase):
    def fixture(self, directory):
        """Write hand-computed saved predictions; never create/call a model."""
        import torch
        root, snapshot, name = Path(directory).resolve(), "common_initial", "historical_update1"
        order, weights, cursor = [], {}, 0
        for key, shape in (("network.0.weight", [128, 11]), ("network.0.bias", [128]),
            ("network.2.weight", [128, 128]), ("network.2.bias", [128]),
            ("network.4.weight", [10, 128]), ("network.4.bias", [10])):
            count = int(np.prod(shape)); weights[key] = torch.full(shape, .05, dtype=torch.float32)
            order.append({"name": key, "shape": shape, "numel": count, "start": cursor, "end": cursor + count}); cursor += count
        original = np.concatenate([value.numpy().reshape(-1) for value in weights.values()])
        direction, gradient = np.full(cursor, -.001, np.float32), np.ones(cursor, np.float32)
        evaluated = np.stack([(torch.from_numpy(original.copy()) + torch.from_numpy(direction.copy()) * alpha).numpy() for alpha in ALPHAS])
        rounded = np.stack([(torch.from_numpy(value.copy()) - torch.from_numpy(original.copy())).numpy() for value in evaluated])
        teacher = np.zeros((4, 10, 10), np.float32); free = np.zeros((4, 2, 6, 10), np.float64)
        loss_rows = []
        for index, factor in enumerate((1., .9, .8, .7)):
            teacher[index, :, 0], teacher[index, :, 6] = factor, 3 * factor
            free[index, :, 1:, 0], free[index, :, 1:, 6] = .5 * factor, 1.5 * factor
            a, b = map(float, teacher[index, 0, [0, 6]])
            c, d = .5 * factor, 1.5 * factor
            one, roll, rt, rf = (a*a+b*b)/10, (c*c+d*d)/10, (b-a)**2/4, (d-c)**2/4
            base, aux = one + .1 * roll, .2 * (rt + .1 * rf)
            loss_rows.append({"one_step": one, "rollout": roll, "relative_teacher": rt, "relative_rollout": rf,
                              "total": base + aux, "base": base, "weighted_aux": aux, "common_total": base + aux})
        reference = {key: loss_rows[0][key] for key in ("one_step", "rollout", "relative_teacher", "relative_rollout", "total")}
        old = root / "oldgrad/probes" / snapshot / name; old.mkdir(parents=True)
        (old / "summary.json").write_text(json.dumps({"parameter_order": order, "loss_values": reference}))
        raw = np.zeros((5, cursor), np.float32); raw[4] = gradient
        np.savez_compressed(old / "gradients.npz", gradients=raw)
        adam = root / "oldadam/probes" / snapshot / name; adam.mkdir(parents=True)
        saved_directions = np.zeros((6, cursor), np.float32); saved_directions[3] = direction
        np.savez_compressed(adam / "directions.npz", directions=saved_directions)
        (adam / "summary.json").write_text(json.dumps({"parameter_order": order, "directions": [{}, {}, {}, {"objective": "total", "mode": "KEEP"}]}))
        case = {"snapshot_id": snapshot, "probe_name": name, "source_direction_npz": str((adam / "directions.npz").relative_to(root)),
            "source_direction_index": 3, "source_summary_sha256": oracle.sha(adam / "summary.json"),
            "original_gradient_summary": str((old / "summary.json").relative_to(root)), "original_gradient_summary_sha256": oracle.sha(old / "summary.json")}
        run = root / "runs/p4-finite-step-synthetic"; folder = run / "cases" / snapshot / name; folder.mkdir(parents=True)
        np.savez_compressed(folder / "parameter_displacements.npz", alphas=np.asarray(ALPHAS, np.float64),
                            evaluated_parameters=evaluated, rounded_parameter_differences=rounded)
        np.savez_compressed(folder / "predictions.npz", teacher_predictions=teacher, free_predicted_states=free)
        denominators = {"batch_windows": 2, "valid_transitions": 10, "base_scalar_terms": 100, "relative_scalar_terms": 20}
        rows, trace = [], []
        for index, alpha in enumerate(ALPHAS):
            actual64, endpoint64, nominal = rounded[index].astype(np.float64), evaluated[index].astype(np.float64)-original.astype(np.float64), alpha*direction.astype(np.float64)
            primary, exact, nominal_dot = float(gradient.astype(np.float64)@actual64), float(gradient.astype(np.float64)@endpoint64), float(gradient.astype(np.float64)@nominal)
            baseline, total = loss_rows[0]["total"], loss_rows[index]["total"]
            primary_pred, exact_pred = baseline + primary, baseline + exact
            trace_hash = oracle.tensor_tree_hash({item["name"]: torch.from_numpy(evaluated[index, item["start"]:item["end"]].reshape(item["shape"]).copy()) for item in order})
            trace += [{"rows": count, "parameter_tree_hash": trace_hash} for count in [10, 2, 2, 2, 2, 2]]
            rows.append({"index": index, "alpha": alpha, "status": "PASS", "loss_values": loss_rows[index], "denominators": denominators,
                "forward_calls": 6, "forward_rows": [10, 2, 2, 2, 2, 2], "forward_total_rows": 20,
                "displacement": {"nominal_scaled_norm": float(np.linalg.norm(nominal)), "rounded_difference_norm": float(np.linalg.norm(actual64)),
                    "exact_endpoint_difference_norm": float(np.linalg.norm(endpoint64)), "rounded_minus_nominal_l2": float(np.linalg.norm(actual64-nominal)), "exact_minus_nominal_l2": float(np.linalg.norm(endpoint64-nominal))},
                "taylor": {"nominal_directional_change": nominal_dot, "rounded_directional_change": primary, "exact_endpoint_directional_change": exact,
                    "primary_prediction": primary_pred, "endpoint_sensitivity_prediction": exact_pred, "nominal_prediction": baseline+nominal_dot,
                    "residual_actual_minus_primary": total-primary_pred, "residual_actual_minus_endpoint": total-exact_pred},
                "loss_differences": {key: loss_rows[index][key]-loss_rows[0][key] for key in ("base", "weighted_aux", "common_total")},
                "total_change_relation": "NUMERIC_TIE" if index == 0 else "DECREASE", "error": None,
                "evaluated_parameters_hash": hashlib.sha256(evaluated[index].tobytes()).hexdigest(), "rounded_difference_hash": hashlib.sha256(rounded[index].tobytes()).hexdigest()})
        reproduction = {key: {"actual": value, "reference": value, "absolute_error": 0., "allowed_error": 1e-12+1e-10*abs(value), "pass": True} for key, value in reference.items()}
        checks = {key: True for key in ("source_state_unchanged", "source_parameter_identities_unchanged", "source_buffer_identities_unchanged",
            "source_gradient_buffers_unchanged", "source_modes_unchanged", "source_hooks_unchanged", "global_rng_unchanged",
            "batch_inputs_unchanged", "normalizer_unchanged", "direction_and_gradient_inputs_unchanged")}
        evaluation = {"schema": "p4-fixed-direction-finite-step-evaluation-v1", "status": "PASS", "alphas": list(ALPHAS), "parameter_order": order,
            "parameter_count": cursor, "raw_loss_order": list(reference), "denominators": denominators, "retained_alpha_count": 4, "case_denominator": 1,
            "alpha0_reproduction": {"status": "PASS", "rtol": 1e-10, "atol": 1e-12, "per_loss": reproduction},
            "neutral_band": 1e-6*max(1.,reference["total"]), "decision": "NO_ALPHA1_OVERSHOOT_OBSERVED_ON_GRID", "evaluations": rows, "checks": checks,
            "source_state_hash_before": oracle.mapping_hash(original, order), "source_state_hash_after": oracle.mapping_hash(original, order),
            "counters": {"evaluations": 4, "successful_evaluations": 4, "root_model_forward_calls": 24, "root_model_forward_rows": 80,
                "new_autograd_traversals": 0, "new_optimizer_steps": 0, "new_training_updates": 0, "new_physics_steps": 0}}
        spec = {"snapshot_id": snapshot, "model_tree_hash": oracle.tensor_tree_hash(weights)}
        record = {"snapshot_id": snapshot, "probe_name": name, "source": case, "parameter_order": order,
            "source_weights_tree_hash": spec["model_tree_hash"], "source_optimizer_tree_hash": "synthetic-no-optimizer",
            "evaluation": evaluation, "source_forward_trace": trace, "parameter_arrays_sha256": oracle.sha(folder / "parameter_displacements.npz"),
            "prediction_arrays_sha256": oracle.sha(folder / "predictions.npz")}
        (folder / "summary.json").write_text(json.dumps(record, allow_nan=False))
        probes = {name+"_states": np.zeros((2, 6, 10), np.float64), name+"_actions": np.zeros((2, 5, 2), np.float64), name+"_mask": np.ones((2, 5), np.bool_)}
        return root, run, case, spec, weights, probes, folder, record

    def test_saved_prediction_and_parameter_case_roundtrip_without_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root, run, case, spec, weights, probes, _, _ = self.fixture(directory)
            with patch.object(oracle, "ROOT", root):
                checker = oracle.Audit(); checker.case(run, case, spec, weights, "synthetic-no-optimizer", probes,
                                                     {"delta_mean": np.zeros(10), "delta_scale": np.ones(10)})
            self.assertEqual(checker.successful_evaluations, 4); self.assertEqual(checker.forward_calls, 24)
            self.assertTrue(all(check["pass"] for check in checker.checks))

    def test_wrong_reported_loss_is_rejected_from_saved_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root, run, case, spec, weights, probes, folder, record = self.fixture(directory)
            record["evaluation"]["evaluations"][1]["loss_values"]["one_step"] += .1
            (folder / "summary.json").write_text(json.dumps(record))
            with patch.object(oracle, "ROOT", root), self.assertRaisesRegex(ValueError, "independent_saved_prediction_losses:one_step"):
                oracle.Audit().case(run, case, spec, weights, "synthetic-no-optimizer", probes, {"delta_mean": np.zeros(10), "delta_scale": np.ones(10)})

    def test_source_forward_trace_cannot_claim_different_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            root, run, case, spec, weights, probes, folder, record = self.fixture(directory)
            record["source_forward_trace"][6]["parameter_tree_hash"] = "0"*64
            (folder / "summary.json").write_text(json.dumps(record))
            with patch.object(oracle, "ROOT", root), self.assertRaisesRegex(ValueError, "actual_source_forwards_used_evaluated_parameter_maps"):
                oracle.Audit().case(run, case, spec, weights, "synthetic-no-optimizer", probes, {"delta_mean": np.zeros(10), "delta_scale": np.ones(10)})


if __name__ == "__main__": unittest.main()
