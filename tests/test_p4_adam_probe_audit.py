"""Adversarial arithmetic tests for the independent saved-only Adam oracle."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.check_p4_adam_probe import (COSINE_THRESHOLD, MODES, clip_reference64,
                                      direction_metrics, displacement_error, ideal_adam,
                                      precision_certificate, history_metrics,
                                      history_statistics, paired_history_statistics)
from tools import check_p4_adam_probe as oracle


class AdamOracleTests(unittest.TestCase):
    def test_cold_is_componentwise_normalized_negative_gradient(self):
        gradient = np.array([.25, -2., 0.], np.float32)
        m = np.array([20., -20., 1.], np.float32)
        v = np.array([.1, .2, .3], np.float32)
        before = (gradient.copy(), m.copy(), v.copy())
        result = ideal_adam(gradient, m, v, 2000, "COLD")
        expected = -.001 * gradient.astype(np.float64) / (np.abs(gradient.astype(np.float64)) + 1e-8)
        np.testing.assert_allclose(result["displacement"], expected, rtol=1e-14, atol=1e-16)
        self.assertEqual(result["next_step"], 1)
        for actual, saved in zip((gradient, m, v), before):
            np.testing.assert_array_equal(actual, saved)

    def test_old_first_moment_can_reverse_self_direction(self):
        gradient = np.array([1.], np.float32)
        m, v = np.array([-2.], np.float32), np.array([1.], np.float32)
        keep = ideal_adam(gradient, m, v, 1300, "KEEP")
        zero_m = ideal_adam(gradient, m, v, 1300, "ZERO_M")
        self.assertEqual(direction_metrics(gradient, keep["displacement"])["relation"], "ASCENT")
        self.assertEqual(direction_metrics(gradient, zero_m["displacement"])["relation"], "DESCENT")
        np.testing.assert_array_equal(keep["next_second_moment"], zero_m["next_second_moment"])
        self.assertEqual(keep["next_step"], zero_m["next_step"])

    def test_cold_changes_second_moment_and_clock(self):
        g = np.array([.2, 2.], np.float32)
        m, v = np.ones(2, np.float32), np.array([100., .0001], np.float32)
        zero_m = ideal_adam(g, m, v, 2300, "ZERO_M")
        cold = ideal_adam(g, m, v, 2300, "COLD")
        self.assertEqual(zero_m["next_step"], 2301)
        self.assertEqual(cold["next_step"], 1)
        self.assertFalse(np.array_equal(zero_m["next_second_moment"], cold["next_second_moment"]))

    def test_cancellation_bound_does_not_collapse_with_displacement(self):
        g = np.array([9.], np.float32)
        m, v = np.array([-1.], np.float32), np.array([1.], np.float32)
        result = ideal_adam(g, m, v, 1300, "KEEP")
        self.assertLess(abs(result["displacement"][0]), 1e-17)
        self.assertGreater(result["coordinate_error_bound"][0], 1e-9)
        self.assertTrue(displacement_error(np.array([1e-10]), result)["pass"])
        self.assertFalse(displacement_error(np.array([1e-2]), result)["pass"])

    def test_displacement_sign_and_scale_corruption_are_rejected(self):
        g = np.array([.2, -.4], np.float32)
        reference = ideal_adam(g, np.zeros(2, np.float32), np.ones(2, np.float32), 1300, "ZERO_M")
        self.assertTrue(displacement_error(reference["displacement"], reference)["pass"])
        self.assertFalse(displacement_error(-reference["displacement"], reference)["pass"])
        self.assertFalse(displacement_error(2. * reference["displacement"], reference)["pass"])

    def test_mode_input_validation(self):
        one = np.ones(2, np.float32)
        for invalid in ("RESET_V", "keep", None):
            with self.assertRaises(ValueError): ideal_adam(one, one, one, 1, invalid)
        for invalid in (-1, 1.5, True):
            with self.assertRaises(ValueError): ideal_adam(one, one, one, invalid, "KEEP")
        with self.assertRaises(ValueError): ideal_adam(one, one, -one, 1, "KEEP")
        with self.assertRaises(ValueError): ideal_adam(one, one[:1], one, 1, "KEEP")
        for mode in MODES:
            with self.assertRaises(ValueError): ideal_adam(one * np.nan, one, one, 1, mode)


class DirectionAndClipOracleTests(unittest.TestCase):
    def test_precision_certificate_preserves_unresolved_sign(self):
        g = np.array([1., 0.])
        uncertain = precision_certificate(g, np.array([1e-9, 1.]), np.array([1e-8, 1e-8]))
        self.assertEqual(uncertain["rounding_stable_sign"], "UNRESOLVED")
        self.assertEqual(uncertain["dot_error_radius"], 1e-8)
        self.assertLess(uncertain["dot_interval"][0], 0)
        self.assertGreater(uncertain["dot_interval"][1], 0)
        self.assertEqual(precision_certificate(g, -g, np.array([1e-8, 1e-8]))["rounding_stable_sign"], "DESCENT")
        self.assertEqual(precision_certificate(g, g, np.array([1e-8, 1e-8]))["rounding_stable_sign"], "ASCENT")
        self.assertEqual(precision_certificate(np.zeros(2), g, np.zeros(2))["rounding_stable_sign"], "UNRESOLVED")

    def test_descent_cosine_uses_negative_dot(self):
        g = np.array([3., 4.])
        result = direction_metrics(g, -g)
        self.assertEqual(result["dot"], -25.)
        self.assertEqual(result["descent_cosine"], 1.)
        self.assertEqual(result["relation"], "DESCENT")
        self.assertEqual(direction_metrics(g, g)["relation"], "ASCENT")

    def test_near_zero_and_orthogonal_classification(self):
        self.assertEqual(direction_metrics(np.zeros(2), np.ones(2))["relation"], "UNDEFINED_NEAR_ZERO_NORM")
        self.assertIsNone(direction_metrics(np.ones(2), np.zeros(2))["descent_cosine"])
        g = np.array([1., 0.])
        self.assertEqual(direction_metrics(g, np.array([0., 1.]))["relation"], "NEAR_ORTHOGONAL")
        self.assertEqual(direction_metrics(g, np.array([-COSINE_THRESHOLD / 2, 1.]))["relation"], "NEAR_ORTHOGONAL")
        self.assertEqual(direction_metrics(g, np.array([-COSINE_THRESHOLD * 2, 1.]))["relation"], "DESCENT")
        self.assertEqual(direction_metrics(g, np.array([COSINE_THRESHOLD * 2, 1.]))["relation"], "ASCENT")

    def test_clip_uses_six_blocks_and_global_scalar(self):
        order = [{"name": str(i), "shape": [1], "numel": 1, "start": i, "end": i + 1} for i in range(6)]
        gradient = np.array([3., 4., 0., 0., 0., 0.], np.float32)
        result = clip_reference64(gradient, order)
        self.assertEqual(result["norm"], 5.)
        self.assertEqual(result["coefficient"], 1. / (5. + 1e-6))
        self.assertEqual(result["parameter_norms"], [3., 4., 0., 0., 0., 0.])
        broken = deepcopy(order); broken[-1]["start"] -= 1
        with self.assertRaises(ValueError): clip_reference64(gradient, broken)
        with self.assertRaises(ValueError): clip_reference64(gradient, order[:-1])


class HistoryOracleTests(unittest.TestCase):
    @staticmethod
    def fixture(branch):
        losses = np.array([[1., 10., 2., 20., 0., .5], [3., 30., 4., 40., 0., 2.]], np.float64)
        if branch == "C0": losses[:, 2:4] = 0
        losses[:, 4] = losses[:, 0] + .1 * losses[:, 1]
        if branch == "R1": losses[:, 4] += .2 * (losses[:, 2] + .1 * losses[:, 3])
        return losses

    def test_short_partial_batch_keeps_both_mean_denominators(self):
        losses = self.fixture("R1")
        result = history_statistics(losses, np.array([64, 16], np.int64), "R1", 0, 2)
        self.assertEqual(result["losses"]["one_step"]["update_mean"], 2.)
        self.assertEqual(result["losses"]["one_step"]["window_weighted_mean"], 1.4)
        self.assertEqual(result["denominators"], {"updates": 2, "window_occurrences": 80,
            "transition_occurrences": 400, "base_scalar_terms_per_term": 4000,
            "relative_scalar_terms_per_term": 800})
        self.assertEqual(result["clipping"]["count"], 1)
        self.assertEqual(result["clipping"]["fraction"], .5)

    def test_c0_relative_zeros_are_reported_uncomputed(self):
        losses = self.fixture("C0")
        result = history_statistics(losses, np.array([64, 16], np.int64), "C0", 0, 2)
        self.assertIsNone(result["denominators"]["relative_scalar_terms_per_term"])
        for key in ("relative_teacher", "relative_free", "weighted_relative_teacher", "weighted_relative_free", "weighted_aux"):
            self.assertIsNone(result["losses"][key])
        losses[0, 2] = .1
        with self.assertRaises(ValueError): history_metrics(losses, "C0")

    def test_strict_clipping_threshold_and_objective_closure(self):
        losses = self.fixture("R1"); losses[:, 5] = [1., 1. + 1e-10]
        result = history_statistics(losses, np.array([64, 16], np.int64), "R1", 0, 2)
        self.assertEqual(result["clipping"]["count"], 1)
        losses[0, 4] += .01
        with self.assertRaises(ValueError): history_metrics(losses, "R1")

    def test_paired_differences_use_shared_batch_schedule(self):
        c0, r1 = self.fixture("C0"), self.fixture("R1")
        r1[:, 0] += np.array([2., -1.]); r1[:, 4] += np.array([2., -1.])
        result = paired_history_statistics(c0, r1, np.array([64, 16], np.int64), 0, 2)
        self.assertEqual(result["losses"]["one_step"]["update_mean"], .5)
        self.assertEqual(result["losses"]["one_step"]["window_weighted_mean"], 1.4)

    def test_saved_history_has_1000_updates_and_fixed_ten_blocks(self):
        root = Path(__file__).resolve().parents[1]
        for branch, clipped in (("C0", 323), ("R1", 408)):
            path = root / "runs/p4-relative-loss-20261014-01/arms" / branch / "training_trace.npz"
            with np.load(path, allow_pickle=False) as data:
                losses, counts = data["losses"], np.diff(data["batch_offsets"])
            self.assertEqual(losses.shape, (1000, 6))
            whole = history_statistics(losses, counts, branch)
            blocks = [history_statistics(losses, counts, branch, start, start + 100) for start in range(0, 1000, 100)]
            self.assertEqual(whole["denominators"]["window_occurrences"], 62992)
            self.assertEqual(whole["batch_counts"], {"full64": 979, "partial16": 21})
            self.assertEqual(sum(block["denominators"]["window_occurrences"] for block in blocks), 62992)
            self.assertEqual(sum(block["clipping"]["count"] for block in blocks), clipped)
            self.assertEqual(whole["clipping"]["count"], clipped)

    def test_saved_history_summary_is_independently_recomputed(self):
        from push_align.adam_history import summarize_histories
        root = Path(__file__).resolve().parents[1]
        parent = root / "runs/p4-relative-loss-20261014-01/arms"
        summary = summarize_histories(parent / "C0/training_trace.npz", parent / "R1/training_trace.npz",
                                      root / "runs/p4-gradient-probe-20261015-01/aggregate.json")
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "history_summary.json").write_text(json.dumps(summary, allow_nan=False))
            checker = oracle.Audit()
            checker.history(run)
        self.assertEqual(len(checker.observed), 2)
        self.assertTrue(all(item["pass"] for item in checker.checks))


class SavedSyntheticProbeAuditTests(unittest.TestCase):
    def fixture(self, directory):
        """Exercise saved-output review with synthetic derivatives, never a model."""
        from push_align.adam_direction_probe import virtual_directions
        import torch
        order, cursor = [], 0
        shapes = [("network.0.weight", [128, 11]), ("network.0.bias", [128]),
                  ("network.2.weight", [128, 128]), ("network.2.bias", [128]),
                  ("network.4.weight", [10, 128]), ("network.4.bias", [10])]
        for name, shape in shapes:
            count = int(np.prod(shape))
            order.append({"name": name, "shape": shape, "numel": count, "start": cursor, "end": cursor + count})
            cursor += count
        generator = np.random.default_rng(317)
        raw = generator.normal(0, .03, (5, cursor)).astype(np.float32)
        base = (raw[0].astype(np.float64) + .1 * raw[1].astype(np.float64)).astype(np.float32)
        raw[4] = (base.astype(np.float64) + .2 * (raw[2].astype(np.float64) + .1 * raw[3].astype(np.float64))).astype(np.float32)
        optimizer = {"state": {}, "param_groups": [{"lr": .001, "betas": (.9, .999), "eps": 1e-8,
            "weight_decay": 0., "amsgrad": False, "maximize": False, "foreach": None,
            "capturable": False, "differentiable": False, "fused": None,
            "decoupled_weight_decay": False, "params": list(range(6))}]}
        for index, item in enumerate(order):
            optimizer["state"][index] = {"step": torch.tensor(1300., dtype=torch.float32),
                "exp_avg": torch.from_numpy(generator.normal(0, .01, item["shape"]).astype(np.float32)),
                "exp_avg_sq": torch.from_numpy(generator.uniform(.001, .02, item["shape"]).astype(np.float32))}
        root, snapshot, name = Path(directory).resolve(), "common_initial", "historical_update1"
        old = root / "runs/p4-gradient-probe-20261015-01/probes" / snapshot / name
        old.mkdir(parents=True)
        (old / "summary.json").write_text(json.dumps({"parameter_order": order}))
        np.savez_compressed(old / "gradients.npz", gradients=raw, base=base)
        checkpoint = root / "synthetic_training_state.pt"
        torch.save({"optimizer": optimizer}, checkpoint)
        moments = oracle.read_moments(checkpoint, order, 1300)
        result = virtual_directions(raw, base, order, optimizer)
        run = root / "runs/synthetic-adam-test"
        folder = run / "probes" / snapshot / name
        folder.mkdir(parents=True)
        np.savez_compressed(folder / "directions.npz", directions=result["directions"], clipped_gradients=result["clipped_gradients"])
        summary = {**result["summary"], "snapshot_id": snapshot, "probe_name": name,
            "source_summary_path": (old / "summary.json").relative_to(root).as_posix(),
            "source_summary_sha256": oracle.sha(old / "summary.json"),
            "source_gradients_sha256": oracle.sha(old / "gradients.npz"),
            "source_optimizer_tree_hash": moments["optimizer_tree_hash"], "source_weights_tree_hash": "synthetic-weights",
            "no_weights_or_moments_changed": True}
        (folder / "summary.json").write_text(json.dumps(summary, allow_nan=False))
        return root, run, {"snapshot_id": snapshot}, name, moments, folder, summary

    def test_saved_synthetic_float32_directions_match_independent_float64_oracle(self):
        with tempfile.TemporaryDirectory() as directory:
            root, run, spec, name, moments, _, _ = self.fixture(directory)
            with patch.object(oracle, "ROOT", root):
                checker = oracle.Audit()
                checker.probe(run, spec, name, moments, "synthetic-weights")
            self.assertEqual(len(checker.precision), 6)
            self.assertTrue(all(item["adam_error"]["pass"] for item in checker.precision))
            self.assertTrue(all(item["base_rounding_sign_agreement"] for item in checker.precision))
            self.assertTrue(all(item["pass"] for item in checker.checks))

    def test_saved_summary_wrong_directional_derivative_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, run, spec, name, moments, folder, summary = self.fixture(directory)
            summary["directions"][0]["loss_geometry"]["base"]["dot"] += 1.
            (folder / "summary.json").write_text(json.dumps(summary, allow_nan=False))
            with patch.object(oracle, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "loss_geometry:base:dot"):
                    oracle.Audit().probe(run, spec, name, moments, "synthetic-weights")


if __name__ == "__main__":
    unittest.main()
