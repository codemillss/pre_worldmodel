"""Controlled free-prediction checks independent of a learned score."""
import json
import math
import unittest
from unittest.mock import patch

import numpy as np

from push_align.data import DataContractError
from push_align.prediction import evaluate_predictions, evaluate_state_hold
from push_align.preparation import PreparedData


class FakeModel:
    def __init__(self, training=True):
        self.training = training

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode


class FakeNormalizer:
    arrays = {"delta_scale": np.ones(10, dtype=np.float64)}

    def validate(self):
        pass


def fixture(regimes=(0, 1, 2), *, initial=None, physical_delta=None, all_starts=False):
    """Each episode contains ten original continuous applied-action intervals."""
    initial = np.zeros((len(regimes), 10), np.float64) if initial is None else np.array(initial, np.float64)
    physical_delta = np.zeros(10, np.float64) if physical_delta is None else np.array(physical_delta, np.float64)
    current, future, actions, indices, episode_codes, regime_codes = [], [], [], [], [], []
    episodes, windows = [], {k: [] for k in (1, 5, 10)}
    for episode_index, regime in enumerate(regimes):
        states = initial[episode_index]+np.arange(11)[:, None]*physical_delta
        offset = len(current)
        for step in range(10):
            current.append(states[step]); future.append(states[step+1])
            actions.append([.001*step, -.0005*step]); indices.append(step)
            episode_codes.append(episode_index); regime_codes.append(regime)
        for k in windows:
            starts = range(11-k) if all_starts else (0,)
            windows[k].extend(np.arange(offset+s, offset+s+k) for s in starts)
        episodes.append({"episode_index": episode_index, "episode_id": f"episode-{episode_index}",
                         "family_id": f"family-{episode_index}", "split": "validation"})
    arrays = {
        "current_state": np.asarray(current, np.float64), "next_state": np.asarray(future, np.float64),
        "applied_action": np.asarray(actions, np.float64), "split_code": np.ones(len(current), np.int8),
        "regime_code": np.asarray(regime_codes, np.int8), "episode_index": np.asarray(episode_codes, np.int64),
        "transition_index": np.asarray(indices, np.int64),
    }
    metadata = {"episodes": episodes, "contract": {"resolved_config": {"task": {
        "position_tolerance_m": .005, "yaw_tolerance_rad": math.pi/60,
    }}}}
    return PreparedData(arrays, metadata, {k: np.asarray(w, np.int64).reshape(-1, k) for k, w in windows.items()})


def evaluate(prepared, predictor, **kwargs):
    with patch("push_align.prediction._physical_delta", side_effect=lambda model, normalizer, state, action: predictor(state, action)):
        return evaluate_predictions(FakeModel(), FakeNormalizer(), prepared, **kwargs)


class P2PredictionTest(unittest.TestCase):
    def test_known_endpoint_all_ten_columns_and_physical_group_errors(self):
        prepared = fixture()
        delta = np.array([.001, -.002, .01, .003, -.004, .02, .002, .003, -.004, .005], np.float64)
        result = evaluate(prepared, lambda state, action: np.broadcast_to(delta, state.shape).copy())
        expected = 5*delta
        np.testing.assert_allclose(result.traces["h5_endpoint_errors"], np.tile(expected, (3, 1)), atol=1e-15)
        metrics = result.summary["horizons"]["5"]["metrics_valid_windows_only"]
        self.assertAlmostEqual(metrics["object_xy_distance_m"]["mean"], np.hypot(*expected[:2]))
        self.assertAlmostEqual(metrics["pusher_xy_distance_m"]["rmse"], np.hypot(*expected[6:8]))
        self.assertAlmostEqual(metrics["object_linear_velocity_distance_m_s"]["p95_absolute"], .025)
        self.assertAlmostEqual(metrics["object_angular_velocity_absolute_rad_s"]["mean"], .1)
        self.assertAlmostEqual(metrics["normalized_state_rmse"], np.sqrt(np.mean(expected**2)))
        for index, field in enumerate(metrics["state_components"]):
            self.assertAlmostEqual(metrics["state_components"][field]["mean"], expected[index])
            self.assertAlmostEqual(metrics["state_components"][field]["mae"], abs(expected[index]))

    def test_recurrence_uses_predictions_and_original_applied_actions(self):
        initial = np.zeros((3, 10), np.float64); initial[:, 0] = 1.
        prepared = fixture(initial=initial)
        observed = []
        def doubling(state, action):
            observed.append((state.copy(), action.copy()))
            delta = np.zeros_like(state); delta[:, 0] = state[:, 0]
            return delta
        result = evaluate(prepared, doubling, horizons=(5,))
        np.testing.assert_array_equal(result.traces["h5_predicted_states"][:, :, 0], np.tile([1, 2, 4, 8, 16, 32], (3, 1)))
        for step, (state, action) in enumerate(observed):
            np.testing.assert_array_equal(state[:, 0], np.full(3, 2**step))
            np.testing.assert_array_equal(action, prepared.arrays["applied_action"][[step, step+10, step+20]])
        np.testing.assert_array_equal(result.traces["h5_truth_states"][:, :, 0], np.ones((3, 6)))

    def test_selection_uses_endpoint_rather_than_time_average(self):
        truth_delta = np.zeros(10, np.float64); truth_delta[0] = .002
        prepared = fixture(physical_delta=truth_delta)
        def one_mm(state, action):
            delta = np.zeros_like(state); delta[:, 0] = .001
            return delta
        result = evaluate(prepared, one_mm)
        self.assertAlmostEqual(result.summary["selection_score"], 1.)
        self.assertAlmostEqual(result.summary["horizons"]["1"]["metrics_valid_windows_only"]["object_xy_distance_m"]["mean"], .001)
        self.assertAlmostEqual(result.summary["horizons"]["5"]["metrics_valid_windows_only"]["object_xy_distance_m"]["mean"], .005)

    def test_equal_regime_weight_ignores_unequal_window_counts(self):
        initial = np.zeros((4, 10), np.float64); initial[:, 1] = [1., 1., 3., 6.]
        prepared = fixture((0, 0, 1, 2), initial=initial)
        def tagged(state, action):
            delta = np.zeros_like(state); delta[:, 0] = .001*state[:, 1]
            return delta
        result = evaluate(prepared, tagged)
        self.assertAlmostEqual(result.summary["selection_score"], (1+3+6)/3)
        self.assertNotAlmostEqual(result.summary["selection_score"], (1+1+3+6)/4)
        self.assertEqual(result.summary["horizons"]["5"]["regimes"]["noncontact"]["attempted_windows"], 2)

    def test_missing_regime_blocks_selection_but_preserves_valid_diagnostics(self):
        result = evaluate(fixture((0, 1)), lambda state, action: np.zeros_like(state))
        self.assertIsNone(result.summary["selection_score"])
        self.assertEqual(result.summary["selection_status"], "missing_selection_regime")
        self.assertEqual(result.summary["horizons"]["5"]["valid_windows"], 2)
        self.assertIsNone(result.summary["horizons"]["5"]["regimes"]["transition"]["metrics_valid_windows_only"]["object_xy_distance_m"]["mean"])

    def test_invalid_raw_yaw_and_nonfinite_row_keep_peers_and_block_selection(self):
        initial = np.zeros((3, 10), np.float64); initial[:, 1] = [0., 1., 2.]
        def invalid(state, action):
            delta = np.zeros_like(state)
            delta[state[:, 1] == 1., 2] = math.pi
            delta[state[:, 1] == 2., 0] = np.nan
            return delta
        result = evaluate(fixture(initial=initial), invalid)
        horizon = result.summary["horizons"]["5"]
        self.assertEqual((horizon["attempted_windows"], horizon["valid_windows"], horizon["invalid_windows"]), (3, 1, 2))
        self.assertIsNone(result.summary["selection_score"])
        self.assertEqual(result.summary["selection_status"], "invalid_selection_windows")
        self.assertEqual({detail["reason"] for detail in horizon["invalid_details"]}, {"raw_delta_yaw_abs_ge_pi", "nonfinite_physical_delta"})
        np.testing.assert_array_equal(result.traces["h5_failure_step"], [-1, 1, 1])
        self.assertTrue(np.isnan(result.traces["h5_predicted_states"][1:, 1:]).all())
        self.assertTrue(np.isfinite(result.traces["h5_predicted_states"][0]).all())
        json.dumps(result.summary, allow_nan=False)

    def test_rejected_batch_falls_back_to_rows_without_dropping_valid_windows(self):
        initial = np.zeros((3, 10), np.float64); initial[:, 1] = [0., 1., 2.]
        def rejects(state, action):
            if (state[:, 1] == 1.).any():
                raise ValueError("controlled nonfinite output")
            return np.zeros_like(state)
        result = evaluate(fixture(initial=initial), rejects)
        self.assertEqual(result.summary["horizons"]["5"]["valid_windows"], 2)
        self.assertEqual(result.summary["horizons"]["5"]["invalid_windows"], 1)
        self.assertIn("inference_rejected:ValueError", result.traces["h5_failure_reason"][1])
        self.assertTrue(np.isfinite(result.traces["h5_predicted_states"][[0, 2]]).all())

    def test_finite_outside_workspace_prediction_is_reported_without_clipping(self):
        def far(state, action):
            delta = np.zeros_like(state); delta[:, 0] = 1.e100
            return delta
        result = evaluate(fixture(), far)
        self.assertEqual(result.summary["horizons"]["10"]["invalid_windows"], 0)
        self.assertAlmostEqual(result.traces["h10_predicted_states"][0, -1, 0]/1.e101, 1.)
        self.assertAlmostEqual(result.summary["horizons"]["5"]["metrics_valid_windows_only"]["object_xy_distance_m"]["rmse"]/5e100, 1.)
        json.dumps(result.summary, allow_nan=False)

    def test_prediction_yaw_error_uses_two_pi_even_for_symmetric_part(self):
        prepared = fixture()
        prepared.arrays["current_state"][:, 2] = math.pi-.01
        prepared.arrays["next_state"][:, 2] = math.pi-.01
        # Change whole two later episodes without breaking their continuity.
        prepared.arrays["current_state"][10:20, 2] = 0.
        prepared.arrays["next_state"][10:20, 2] = 0.
        def signed_yaw(state, action):
            delta = np.zeros_like(state)
            delta[state[:, 2] > 3., 2] = .02
            delta[(state[:, 2] >= 0.) & (state[:, 2] < 3.), 2] = (math.pi-.01)/5
            return delta
        result = evaluate(prepared, signed_yaw, horizons=(5,))
        self.assertAlmostEqual(result.traces["h5_endpoint_errors"][0, 2], .02)
        self.assertAlmostEqual(abs(result.traces["h5_endpoint_errors"][1, 2]), math.pi-.01)
        # A held yaw across the signed boundary is a small 2*pi error.
        held = fixture(); held.arrays["next_state"][4::10, 2] = -math.pi+.01
        held.arrays["current_state"][[0, 10, 20], 2] = math.pi-.01
        # Use K1 to avoid an intentionally discontinuous toy's longer windows.
        held.windows.clear(); held.windows[1] = np.array([[4], [14], [24]], np.int64)
        held.arrays["current_state"][[4, 14, 24], 2] = math.pi-.01
        error = evaluate_state_hold(held, horizons=(1,)).traces["h1_endpoint_errors"][:, 2]
        np.testing.assert_allclose(error, -.02, atol=1e-15)

    def test_windows_reject_split_episode_gap_and_discontinuous_truth(self):
        for mutation in (
            lambda p: p.arrays["split_code"].__setitem__(1, 0),
            lambda p: p.arrays["episode_index"].__setitem__(1, 99),
            lambda p: p.arrays["transition_index"].__setitem__(1, 50),
            lambda p: p.arrays["current_state"].__setitem__((1, 0), .123),
        ):
            prepared = fixture(); mutation(prepared)
            with self.assertRaises(DataContractError):
                evaluate_state_hold(prepared, horizons=(5,))

    def test_modes_restored_and_hold_reference_has_limited_claim(self):
        prepared = fixture()
        for mode in (True, False):
            model = FakeModel(mode)
            with patch("push_align.prediction._physical_delta", side_effect=RuntimeError("infrastructure failure")):
                with self.assertRaises(RuntimeError):
                    evaluate_predictions(model, FakeNormalizer(), prepared)
            self.assertEqual(model.training, mode)
        delta = np.zeros(10, np.float64); delta[0] = .001; delta[8] = .002
        hold = evaluate_state_hold(fixture(physical_delta=delta))
        self.assertEqual(hold.summary["kind"], "state_hold_diagnostic")
        self.assertIn("not assessed", hold.summary["scope"])
        np.testing.assert_array_equal(hold.traces["h5_predicted_states"], np.zeros((3, 6, 10)))
        self.assertAlmostEqual(hold.traces["h5_endpoint_errors"][0, 8], -.01)
        self.assertTrue(all(value.dtype != object for value in hold.traces.values()))

    def test_requested_split_only_and_empty_split_is_explicit(self):
        prepared = fixture()
        prepared.arrays["split_code"][:10] = 2
        result = evaluate_state_hold(prepared, split="smoke-test")
        self.assertEqual(result.summary["horizons"]["5"]["attempted_windows"], 1)
        np.testing.assert_array_equal(result.traces["h5_episode_index"], [0])
        empty = evaluate_state_hold(prepared, split="train")
        self.assertEqual(empty.summary["horizons"]["5"]["attempted_windows"], 0)
        self.assertIsNone(empty.summary["selection_score"])
        self.assertEqual(empty.traces["h5_truth_states"].shape, (0, 6, 10))
        json.dumps(empty.summary, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
