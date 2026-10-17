"""Independent numerical examples and persistence boundaries for main Ridge.

These small, synthetic feature tables are unit fixtures, not robot data or
evidence of predictive accuracy.  Fit receives an already selected table;
the runner must prove that its table contains only train rows.  Here the
normalizer remains train-bound and validation prediction cannot refit it.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
import warnings
from zipfile import ZipFile

import numpy as np
import torch

from push_align import PlanarPushEnv
from push_align.data import _json_bytes, make_contract
from push_align.dynamics import DynamicModelError, free_rollout, predict_delta
from push_align.normalization import fit_normalizer, learning_schema
from push_align.prediction import evaluate_predictions
from push_align.preparation import PreparedData
from push_align.ridge import DeltaRidge, RidgeError, fit_ridge, load_ridge, save_ridge


def _write_json(path, metadata):
    Path(path).write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False), encoding="utf-8")


def _rebind_untrusted_container(path, metadata=None):
    """Reach inner semantic guards after adversarial edits, never repair data."""
    path = Path(path)
    if metadata is None:
        metadata = json.loads((path / "ridge.json").read_text(encoding="utf-8"))
    metadata["files"] = {
        "weights.npz": hashlib.sha256((path / "weights.npz").read_bytes()).hexdigest()
    }
    metadata["ridge_hash"] = hashlib.sha256(_json_bytes({
        key: value for key, value in metadata.items() if key != "ridge_hash"
    })).hexdigest()
    _write_json(path / "ridge.json", metadata)


def _unsafe_marker(path):
    Path(path).write_text("object pickle executed", encoding="utf-8")
    return 0


class _UnsafeObject:
    def __init__(self, path):
        self.path = str(path)

    def __reduce__(self):
        return _unsafe_marker, (self.path,)


class P3RidgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = make_contract(PlanarPushEnv())
        cls.config = deepcopy(cls.contract["resolved_config"])
        cls.schema = learning_schema(cls.contract)
        cls.identity = {
            "dataset_id": "independent-ridge-unit-fixture",
            "manifest_sha256": "a" * 64,
            "data_contract_hash": cls.schema["data_contract_hash"],
        }
        cls.normalizer = cls.make_normalizer()
        cls.provenance = {
            "learning_manifest_sha256": "b" * 64,
            "dataset_identity": deepcopy(cls.identity),
            "validation_split": "validation",
            "task_scope": "P3-03-main-ridge",
        }

    @classmethod
    def make_normalizer(cls, *, identity=None, reference_scale=.001):
        # Reference rows establish nonzero input scale and zero physical
        # target mean.  They are separate numerical-normalizer fixtures.
        x = np.zeros((3, 11), dtype=np.float64)
        x[:, 0] = [-reference_scale, 0., reference_scale]
        return fit_normalizer(
            x, np.zeros((3, 10), dtype=np.float64), np.zeros(3, dtype=np.int8),
            cls.schema, cls.identity if identity is None else identity,
            np.zeros(3, dtype=np.int64), np.arange(3, dtype=np.int64),
        )

    def physical_table(self, x, y, normalizer=None):
        """Construct physical f64 rows whose normalized values are specified."""
        norm = self.normalizer if normalizer is None else normalizer
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return (x * norm.arrays["input_scale"] + norm.arrays["input_mean"],
                y * norm.arrays["delta_scale"] + norm.arrays["delta_mean"])

    def scalar_table(self, *, slope=2., intercept=3., duplicate=False):
        x = np.zeros((3, 11), dtype=np.float64)
        x[:, 0] = [-1., 0., 1.]
        if duplicate:
            x[:, 1] = x[:, 0]
        y = np.zeros((3, 10), dtype=np.float64)
        y[:, 0] = slope * x[:, 0] + intercept
        return self.physical_table(x, y)

    def fitted(self, *, alpha=2.):
        features, targets = self.scalar_table()
        return fit_ridge(features, targets, self.normalizer, alpha)

    def saved(self, directory, *, alpha=2., selection_score=.25):
        model, diagnostics = self.fitted(alpha=alpha)
        path = Path(directory) / "ridge"
        metadata = save_ridge(
            path, model, self.normalizer, self.config, self.provenance,
            alpha=alpha, fit_diagnostics=diagnostics, selection_score=selection_score,
        )
        return path, metadata, model, diagnostics

    def loaded(self, path, **overrides):
        options = {
            "normalizer": self.normalizer, "config": self.config,
            "provenance": self.provenance,
        }
        options.update(overrides)
        return load_ridge(path, **options)

    def test_hand_scalar_sum_objective_unpenalized_intercept(self):
        # For x=-1,0,1 and y=2x+3, SUM RSS + 2w² has w=1,b=3.
        # Mean RSS would instead give w=.5, so this checks objective units.
        model, d = self.fitted(alpha=2.)
        expected = torch.zeros((10, 11), dtype=torch.float32)
        expected[0, 0] = 1.
        torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)
        torch.testing.assert_close(model.bias, torch.tensor([3.] + [0.] * 9), rtol=0, atol=0)
        self.assertEqual(d["fit_count"], 3)
        self.assertEqual(d["rank"], 1)
        self.assertEqual(d["alpha"], 2.)
        self.assertAlmostEqual(d["residual_sum_squares"], 2., delta=1e-12)
        self.assertAlmostEqual(d["coefficient_squared_norm"], 1., delta=1e-12)
        self.assertAlmostEqual(d["objective"], 4., delta=1e-12)
        self.assertAlmostEqual(d["normalized_residual_mse"], 2. / 30., delta=1e-12)
        self.assertEqual(d["solve_dtype"], "float64")
        self.assertEqual(d["model_dtype"], "float32")

    def test_zero_alpha_exact_solution_and_rank_cutoff(self):
        model, d = self.fitted(alpha=0.)
        self.assertEqual(float(model.weight[0, 0]), 2.)
        self.assertEqual(float(model.bias[0]), 3.)
        self.assertAlmostEqual(d["residual_sum_squares"], 0., delta=1e-25)
        self.assertAlmostEqual(d["rank_threshold"],
                               11 * np.finfo(np.float64).eps * math.sqrt(2.), delta=1e-27)
        self.assertFalse(d["condition_full_rank"])

    def test_repeating_rows_changes_sum_penalty_tradeoff(self):
        x, y = self.scalar_table()
        first, _ = fit_ridge(x, y, self.normalizer, 2.)
        repeated, _ = fit_ridge(np.repeat(x, 2, axis=0), np.repeat(y, 2, axis=0),
                                self.normalizer, 2.)
        self.assertEqual(float(first.weight[0, 0]), 1.)
        self.assertAlmostEqual(float(repeated.weight[0, 0]), 4. / 3., delta=1e-7)
        self.assertEqual(float(repeated.bias[0]), 3.)

    def test_shifted_inputs_do_not_penalize_physical_intercept(self):
        x = np.zeros((3, 11), dtype=np.float64)
        x[:, 0] = [4., 5., 6.]
        y = np.zeros((3, 10), dtype=np.float64)
        y[:, 0] = 2. * x[:, 0] + 3.
        model, _ = fit_ridge(*self.physical_table(x, y), self.normalizer, 2.)
        # Centered slope 1; mean(y)=13 and mean(x)=5 imply intercept 8.
        self.assertEqual(float(model.weight[0, 0]), 1.)
        self.assertEqual(float(model.bias[0]), 8.)

    def test_constant_and_single_row_tables_fit_intercept_without_shrinkage(self):
        for n in (1, 4):
            x = np.full((n, 11), 2., dtype=np.float64)
            y = np.tile(np.arange(10, dtype=np.float64) - 4., (n, 1))
            with self.subTest(n=n):
                model, d = fit_ridge(*self.physical_table(x, y), self.normalizer, 1e6)
                torch.testing.assert_close(model.weight, torch.zeros((10, 11)), rtol=0, atol=0)
                torch.testing.assert_close(model.bias, torch.arange(10, dtype=torch.float32) - 4.,
                                           rtol=0, atol=0)
                self.assertEqual(d["rank"], 0)
                self.assertAlmostEqual(d["objective"], 0., delta=1e-20)

    def test_duplicate_columns_return_minimum_norm_and_stable_shrinkage(self):
        x, y = self.scalar_table(duplicate=True)
        for alpha, coefficient in ((0., 1.), (2., 2. / 3.)):
            with self.subTest(alpha=alpha):
                model, d = fit_ridge(x, y, self.normalizer, alpha)
                self.assertAlmostEqual(float(model.weight[0, 0]), coefficient, delta=1e-7)
                self.assertAlmostEqual(float(model.weight[0, 1]), coefficient, delta=1e-7)
                self.assertEqual(float(model.bias[0]), 3.)
                self.assertEqual(d["rank"], 1)
                self.assertTrue(bool(torch.isfinite(model.weight).all()))
                torch.testing.assert_close(model.weight[:, 2:], torch.zeros((10, 9)), rtol=0, atol=1e-12)

    def test_multiple_outputs_share_fit_but_have_independent_coefficients(self):
        x = np.zeros((3, 11), dtype=np.float64)
        x[:, 0] = [-1., 0., 1.]
        y = x[:, :1] * np.arange(1., 11.) + np.arange(10.) - 4.
        model, _ = fit_ridge(*self.physical_table(x, y), self.normalizer, 2.)
        torch.testing.assert_close(model.weight[:, 0], torch.arange(1., 11.) / 2., rtol=0, atol=0)
        torch.testing.assert_close(model.bias, torch.arange(10.) - 4., rtol=0, atol=0)
        torch.testing.assert_close(model.weight[:, 1:], torch.zeros((10, 10)), rtol=0, atol=0)

    def test_near_null_direction_below_zero_alpha_threshold_is_discarded(self):
        x = np.zeros((4, 11), dtype=np.float64)
        x[:, 0] = [-1., -1., 1., 1.]
        x[:, 1] = np.array([-1., 1., -1., 1.]) * 1e-16
        y = np.zeros((4, 10), dtype=np.float64)
        y[:, 0] = [-1., 1., -1., 1.]
        model, d = fit_ridge(*self.physical_table(x, y), self.normalizer, 0.)
        self.assertEqual(d["rank"], 1)
        self.assertLess(abs(float(model.weight[0, 1])), 1e-10)
        self.assertTrue(bool(torch.isfinite(model.weight).all()))
        self.assertAlmostEqual(d["residual_sum_squares"], 4., delta=1e-12)

    def test_positive_alpha_retains_ridge_gain_below_pseudoinverse_rank_cutoff(self):
        x = np.zeros((4, 11), dtype=np.float64)
        x[:, 0] = [-1., -1., 1., 1.]
        x[:, 1] = np.array([-1., 1., -1., 1.]) * 1e-16
        y = np.zeros((4, 10), dtype=np.float64)
        y[:, 0] = np.array([-1., 1., -1., 1.]) * 1e16
        physical_x, physical_y = self.physical_table(x, y)
        pseudoinverse, d0 = fit_ridge(physical_x, physical_y, self.normalizer, 0.)
        regularized, d2 = fit_ridge(physical_x, physical_y, self.normalizer, 2.)
        self.assertEqual(d0["rank"], 1)
        self.assertEqual(d2["rank"], 1)
        self.assertLess(abs(float(pseudoinverse.weight[0, 1])), 1e-10)
        # Orthogonal scalar direction: sum(z*y)/(sum(z*z)+2) ~= 2.
        self.assertAlmostEqual(float(regularized.weight[0, 1]), 2., delta=1e-5)

    def test_rank_cutoff_uses_row_count_when_larger_than_feature_width(self):
        x = np.zeros((22, 11), dtype=np.float64)
        x[:, 0] = np.tile([-1., 1.], 11)
        y = np.zeros((22, 10), dtype=np.float64)
        y[:, 0] = x[:, 0]
        model, d = fit_ridge(*self.physical_table(x, y), self.normalizer, 0.)
        self.assertEqual(float(model.weight[0, 0]), 1.)
        self.assertEqual(d["rank"], 1)
        self.assertAlmostEqual(d["rank_threshold"],
                               22 * np.finfo(np.float64).eps * math.sqrt(22.), delta=1e-26)

    def test_fit_obeys_float32_boundary_before_float64_solve(self):
        # These physical f64 differences disappear at the registered f32
        # boundary. Solving the pre-cast f64 table would infer a huge slope.
        x = np.zeros((3, 11), dtype=np.float64)
        x[:, 0] = [1., 1. + 2. ** -30, 1. + 2. ** -29]
        y = np.zeros((3, 10), dtype=np.float64)
        y[:, 0] = [-1., 0., 1.]
        physical_x, physical_y = self.physical_table(x, y)
        np.testing.assert_array_equal(self.normalizer.normalize_features(physical_x)[:, 0],
                                      np.ones(3, dtype=np.float32))
        model, d = fit_ridge(physical_x, physical_y, self.normalizer, 0.)
        self.assertEqual(d["rank"], 0)
        torch.testing.assert_close(model.weight, torch.zeros((10, 11)), rtol=0, atol=0)
        torch.testing.assert_close(model.bias, torch.zeros(10), rtol=0, atol=0)

    def test_fit_constructor_and_archive_leave_rng_and_global_dtype_unchanged(self):
        original_dtype = torch.get_default_dtype()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        try:
            torch.set_default_dtype(torch.float64)
            direct = DeltaRidge()
            model, d = self.fitted()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "ridge"
                save_ridge(path, model, self.normalizer, self.config, self.provenance,
                           alpha=2., fit_diagnostics=d)
                loaded = self.loaded(path)
            for candidate in (direct, model, loaded.model):
                self.assertEqual(candidate.weight.dtype, torch.float32)
                self.assertEqual(candidate.bias.dtype, torch.float32)
                self.assertEqual(candidate.weight.device.type, "cpu")
                self.assertEqual(candidate.weight.shape, (10, 11))
                self.assertEqual(candidate.bias.shape, (10,))
            self.assertEqual(torch.get_default_dtype(), torch.float64)
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
            after_numpy = np.random.get_state()
            self.assertEqual(numpy_state[0], after_numpy[0])
            np.testing.assert_array_equal(numpy_state[1], after_numpy[1])
            self.assertEqual(numpy_state[2:], after_numpy[2:])
        finally:
            torch.set_default_dtype(original_dtype)

    def test_fit_preserves_input_arrays_and_train_normalizer(self):
        x, y = self.scalar_table()
        before_x, before_y = x.copy(), y.copy()
        before_metadata = deepcopy(self.normalizer.metadata)
        before_arrays = {key: value.copy() for key, value in self.normalizer.arrays.items()}
        fit_ridge(x, y, self.normalizer, 2.)
        np.testing.assert_array_equal(x, before_x)
        np.testing.assert_array_equal(y, before_y)
        self.assertEqual(self.normalizer.metadata, before_metadata)
        for key, value in before_arrays.items():
            np.testing.assert_array_equal(self.normalizer.arrays[key], value)

    def test_forward_leading_shapes_match_shared_float32_interface(self):
        model, _ = self.fitted()
        for shape in ((11,), (3, 11), (2, 3, 11)):
            with self.subTest(shape=shape):
                inputs = torch.zeros(shape, dtype=torch.float32)
                output = model(inputs)
                self.assertEqual(output.shape, (*shape[:-1], 10))
                self.assertEqual(output.dtype, torch.float32)
                self.assertTrue(bool(torch.isfinite(output).all()))
                torch.testing.assert_close(output[..., 0], torch.full(shape[:-1], 3.),
                                           rtol=0, atol=0)

    def test_forward_rejects_incompatible_inputs_and_corrupted_model_buffers(self):
        model, _ = self.fitted()
        cases = [np.zeros(11, dtype=np.float32), torch.zeros(10),
                 torch.zeros(11, dtype=torch.float64), torch.full((11,), float("nan")),
                 torch.full((11,), float("inf"))]
        for value in cases:
            with self.subTest(type=type(value).__name__), self.assertRaises(DynamicModelError):
                model(value)
        model.double()
        with self.assertRaises(DynamicModelError):
            model(torch.zeros(11, dtype=torch.float32))
        model, _ = self.fitted()
        with torch.no_grad():
            model.weight[0, 0] = float("nan")
        with self.assertRaises(DynamicModelError):
            model(torch.zeros(11, dtype=torch.float32))

    def test_autocast_cannot_change_registered_model_boundary(self):
        model, _ = self.fitted()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with self.assertRaises(DynamicModelError):
                model(torch.zeros(11, dtype=torch.float32))

    def recurrent_model(self):
        # Physical delta object_x = .5 * (pusher_x - object_x).
        coefficient = (.5 * self.normalizer.arrays["input_scale"][0]
                       / self.normalizer.arrays["delta_scale"][0])
        return fit_ridge(*self.scalar_table(slope=coefficient, intercept=0.),
                         self.normalizer, 0.)[0]

    def test_shared_prediction_denormalizes_and_rollout_uses_predicted_state(self):
        model = self.recurrent_model()
        state = torch.zeros(10, dtype=torch.float64)
        state[6] = .01
        actions = torch.tensor([[.02, -.01]] * 5, dtype=torch.float64)
        before_state, before_actions = state.clone(), actions.clone()
        delta = predict_delta(model, self.normalizer, state, actions[0])
        self.assertEqual(delta.dtype, torch.float64)
        self.assertAlmostEqual(float(delta[0]), .005, delta=2e-9)
        model.eval()
        states = free_rollout(model, self.normalizer, state, actions)
        expected = .01 * (1. - .5 ** np.arange(6))
        np.testing.assert_allclose(states[:, 0].numpy(), expected, rtol=0, atol=2e-9)
        self.assertGreater(float(states[-1, 0]), .009)
        self.assertFalse(model.training)
        torch.testing.assert_close(state, before_state, rtol=0, atol=0)
        torch.testing.assert_close(actions, before_actions, rtol=0, atol=0)

    def validation_fixture(self):
        # Stored truth remains fixed, so any recurrence from truth at each
        # step would predict .005 rather than .01*(1-.5**k).
        arrays = {
            "current_state": np.zeros((30, 10), dtype=np.float64),
            "next_state": np.zeros((30, 10), dtype=np.float64),
            "applied_action": np.tile(np.array([.02, -.01], dtype=np.float64), (30, 1)),
            "split_code": np.ones(30, dtype=np.int8),
            "regime_code": np.repeat(np.arange(3, dtype=np.int8), 10),
            "episode_index": np.repeat(np.arange(3, dtype=np.int64), 10),
            "transition_index": np.tile(np.arange(10, dtype=np.int64), 3),
        }
        arrays["current_state"][:, 6] = .01
        arrays["next_state"][:, 6] = .01
        windows = {k: np.stack([np.arange(10 * episode, 10 * episode + k)
                                for episode in range(3)]) for k in (1, 5, 10)}
        metadata = {
            "contract": deepcopy(self.contract),
            "episodes": [{"episode_index": i, "episode_id": f"validation-{i}",
                          "family_id": f"validation-family-{i}", "split": "validation"}
                         for i in range(3)],
        }
        return PreparedData(arrays, metadata, windows)

    def test_actual_shared_evaluator_accepts_ridge_and_does_not_fit_validation(self):
        model = self.recurrent_model()
        before_weights = model.weight.clone(), model.bias.clone()
        before_normalizer = deepcopy(self.normalizer.metadata)
        prepared = self.validation_fixture()
        before_arrays = {key: value.copy() for key, value in prepared.arrays.items()}
        model.train()
        evaluation = evaluate_predictions(model, self.normalizer, prepared)
        self.assertTrue(model.training)
        expected = np.tile(.01 * (1. - .5 ** np.arange(6)), (3, 1))
        np.testing.assert_allclose(evaluation.traces["h5_predicted_states"][:, :, 0],
                                   expected, rtol=0, atol=2e-9)
        np.testing.assert_array_equal(evaluation.traces["h5_truth_states"][:, :, 0],
                                      np.zeros((3, 6)))
        self.assertEqual(evaluation.summary["selection_status"], "eligible")
        self.assertAlmostEqual(evaluation.summary["selection_score"], .0096875 / .005,
                               delta=1e-6)
        for horizon in (1, 5, 10):
            self.assertEqual(evaluation.summary["horizons"][str(horizon)]["invalid_windows"], 0)
        torch.testing.assert_close(model.weight, before_weights[0], rtol=0, atol=0)
        torch.testing.assert_close(model.bias, before_weights[1], rtol=0, atol=0)
        self.assertEqual(self.normalizer.metadata, before_normalizer)
        for key, value in before_arrays.items():
            np.testing.assert_array_equal(prepared.arrays[key], value)

    def test_fit_rejects_invalid_alpha_without_bool_as_numeric(self):
        x, y = self.scalar_table()
        for alpha in (True, False, -1., float("nan"), float("inf"), -float("inf"),
                      "2", None, np.array([2.])):
            with self.subTest(alpha=repr(alpha)), self.assertRaises(RidgeError):
                fit_ridge(x, y, self.normalizer, alpha)

    def test_fit_rejects_nonphysical_dtype_invalid_shapes_and_nonfinite_rows(self):
        x, y = self.scalar_table()
        invalid_x = [x.astype(np.float32), x.tolist(), torch.from_numpy(x), x[:, :10],
                     x[0], x.reshape(1, 3, 11), x[:0], x.copy(), x.copy()]
        invalid_x[-2][0, 0] = np.nan
        invalid_x[-1][0, 0] = np.inf
        for bad in invalid_x:
            with self.subTest(side="features", shape=getattr(bad, "shape", None)), self.assertRaises(RidgeError):
                fit_ridge(bad, y, self.normalizer, 2.)
        invalid_y = [y.astype(np.float32), y.tolist(), y[:, :9], y[:2], y[0],
                     y.copy(), y.copy()]
        invalid_y[-2][0, 0] = np.nan
        invalid_y[-1][0, 0] = np.inf
        for bad in invalid_y:
            with self.subTest(side="targets", shape=getattr(bad, "shape", None)), self.assertRaises(RidgeError):
                fit_ridge(x, bad, self.normalizer, 2.)
        with self.assertRaises(RidgeError):
            fit_ridge(x, y, object(), 2.)

    def test_fit_rejects_float32_boundary_overflow_and_unrepresentable_coefficients(self):
        x, y = self.scalar_table()
        x[0, 0] = np.finfo(np.float64).max
        with self.assertRaises(RidgeError):
            fit_ridge(x, y, self.normalizer, 0.)
        normalized_x = np.zeros((3, 11), dtype=np.float64)
        normalized_x[:, 0] = [-1e-20, 0., 1e-20]
        normalized_y = np.zeros((3, 10), dtype=np.float64)
        normalized_y[:, 0] = [-1e20, 0., 1e20]
        with self.assertRaises(RidgeError):
            fit_ridge(*self.physical_table(normalized_x, normalized_y), self.normalizer, 0.)

    def test_archive_roundtrip_preserves_all_weights_and_physical_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            path, metadata, model, d = self.saved(directory)
            restored = self.loaded(path)
            self.assertEqual(set(p.name for p in path.iterdir()), {"ridge.json", "weights.npz"})
            self.assertFalse(restored.model.training)
            self.assertEqual(restored.metadata, metadata)
            self.assertEqual(metadata["training_provenance"], self.provenance)
            self.assertEqual(metadata["alpha"], 2.)
            torch.testing.assert_close(restored.model.weight, model.weight, rtol=0, atol=0)
            torch.testing.assert_close(restored.model.bias, model.bias, rtol=0, atol=0)
            with np.load(path / "weights.npz", allow_pickle=False) as arrays:
                self.assertEqual(set(arrays.files), {"weight", "bias"})
                self.assertEqual(arrays["weight"].dtype, np.float32)
                self.assertEqual(arrays["bias"].dtype, np.float32)
            state = torch.zeros(10, dtype=torch.float64)
            state[6] = .005
            action = torch.tensor([.02, -.01], dtype=torch.float64)
            torch.testing.assert_close(
                predict_delta(restored.model, self.normalizer, state, action),
                predict_delta(model, self.normalizer, state, action), rtol=0, atol=0,
            )

    def test_existing_archive_cannot_be_overwritten_even_by_same_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, model, d = self.saved(directory)
            before = {p.name: p.read_bytes() for p in path.iterdir()}
            with self.assertRaises((RidgeError, FileExistsError)):
                save_ridge(path, model, self.normalizer, self.config, self.provenance,
                           alpha=2., fit_diagnostics=d)
            self.assertEqual({p.name: p.read_bytes() for p in path.iterdir()}, before)

    def test_modified_fitted_weights_cannot_be_saved_with_original_diagnostics(self):
        model, d = self.fitted()
        with torch.no_grad():
            model.weight[0, 0] += .25
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ridge"
            with self.assertRaises(RidgeError):
                save_ridge(path, model, self.normalizer, self.config, self.provenance,
                           alpha=2., fit_diagnostics=d)
            self.assertFalse(path.exists())

    def test_rehashed_finite_weight_change_is_rejected_by_fit_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, _, _ = self.saved(directory)
            with np.load(path / "weights.npz", allow_pickle=False) as archive:
                weight, bias = archive["weight"].copy(), archive["bias"].copy()
            weight[0, 0] += .25
            np.savez_compressed(path / "weights.npz", weight=weight, bias=bias)
            _rebind_untrusted_container(path)
            with self.assertRaises(RidgeError):
                self.loaded(path)

    def test_archive_rejects_unbound_file_and_metadata_byte_changes(self):
        for filename in ("weights.npz", "ridge.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                path, _, _, _ = self.saved(directory)
                if filename.endswith("npz"):
                    with (path / filename).open("ab") as output:
                        output.write(b"unbound bytes")
                else:
                    metadata = json.loads((path / filename).read_text())
                    metadata["alpha"] = 3.
                    _write_json(path / filename, metadata)
                with self.assertRaises(RidgeError):
                    self.loaded(path)

    def test_archive_rejects_rehashed_weight_shape_dtype_nonfinite_and_extra_columns(self):
        valid = {"weight": np.zeros((10, 11), np.float32), "bias": np.zeros(10, np.float32)}
        variants = []
        for name in ("weight", "bias"):
            for issue in ("shape", "dtype", "nonfinite"):
                modified = {k: v.copy() for k, v in valid.items()}
                if issue == "shape":
                    modified[name] = modified[name][:-1]
                elif issue == "dtype":
                    modified[name] = modified[name].astype(np.float64)
                else:
                    modified[name].flat[0] = np.nan
                variants.append((f"{name}-{issue}", modified))
        variants.extend([("extra", {**valid, "extra": np.zeros(1, np.float32)}),
                         ("missing", {"weight": valid["weight"]})])
        for label, arrays in variants:
            with self.subTest(issue=label), tempfile.TemporaryDirectory() as directory:
                path, _, _, _ = self.saved(directory)
                np.savez_compressed(path / "weights.npz", **arrays)
                _rebind_untrusted_container(path)
                with self.assertRaises(RidgeError):
                    self.loaded(path)

    def test_archive_rejects_duplicate_npz_member_after_hash_rebinding(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, _, _ = self.saved(directory)
            with ZipFile(path / "weights.npz", "r") as archive:
                duplicate = archive.read("weight.npy")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with ZipFile(path / "weights.npz", "a") as archive:
                    archive.writestr("weight.npy", duplicate)
            _rebind_untrusted_container(path)
            with self.assertRaises(RidgeError):
                self.loaded(path)

    def test_object_archive_is_rejected_without_executing_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, _, _ = self.saved(directory)
            marker = Path(directory) / "object-executed"
            weight = np.zeros((10, 11), dtype=object)
            weight[0, 0] = _UnsafeObject(marker)
            np.savez_compressed(path / "weights.npz", weight=weight,
                                bias=np.zeros(10, dtype=np.float32))
            _rebind_untrusted_container(path)
            with self.assertRaises(RidgeError):
                self.loaded(path)
            self.assertFalse(marker.exists())

    def test_load_requires_exact_dataset_normalizer_schema_config_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, _, _ = self.saved(directory)
            new_identity = deepcopy(self.identity)
            new_identity["dataset_id"] = "another-dataset"
            dataset_norm = self.make_normalizer(identity=new_identity)
            dataset_provenance = deepcopy(self.provenance)
            dataset_provenance["dataset_identity"] = deepcopy(new_identity)
            changed_stats = self.make_normalizer(reference_scale=.003)
            changed_config = deepcopy(self.config)
            changed_config["physics"]["action_dt_s"] *= 2.
            changed_schema = deepcopy(self.normalizer)
            changed_schema.metadata["schema"]["feature_fields"][0:2] = reversed(
                changed_schema.metadata["schema"]["feature_fields"][0:2])
            changed_manifest = deepcopy(self.provenance)
            changed_manifest["learning_manifest_sha256"] = "c" * 64
            variants = [
                {"normalizer": dataset_norm, "provenance": dataset_provenance},
                {"normalizer": changed_stats}, {"normalizer": changed_schema},
                {"config": changed_config}, {"provenance": changed_manifest},
            ]
            for options in variants:
                with self.subTest(keys=list(options)), self.assertRaises(RidgeError):
                    self.loaded(path, **options)

    def test_save_rejects_wrong_scope_split_identity_hash_and_config_before_writing(self):
        model, d = self.fitted()
        provenance_cases = []
        for key, value in (("task_scope", "P2-04-pilot"), ("validation_split", "train"),
                           ("learning_manifest_sha256", "bad-hash"),
                           ("dataset_identity", {**self.identity, "dataset_id": "other"})):
            p = deepcopy(self.provenance)
            p[key] = value
            provenance_cases.append(p)
        for index, provenance in enumerate(provenance_cases):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "ridge"
                with self.assertRaises(RidgeError):
                    save_ridge(path, model, self.normalizer, self.config, provenance,
                               alpha=2., fit_diagnostics=d)
                self.assertFalse(path.exists())
        config = deepcopy(self.config)
        config["task"]["position_tolerance_m"] *= 2.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ridge"
            with self.assertRaises(RidgeError):
                save_ridge(path, model, self.normalizer, config, self.provenance,
                           alpha=2., fit_diagnostics=d)
            self.assertFalse(path.exists())

    def test_save_rejects_invalid_alpha_score_and_inconsistent_fit_diagnostics(self):
        model, d = self.fitted()
        cases = [{"alpha": True}, {"alpha": -1.}, {"alpha": float("nan")},
                 {"alpha": 3.}, {"selection_score": -1.},
                 {"selection_score": float("nan")}, {"selection_score": float("inf")}]
        invalid_d = deepcopy(d)
        invalid_d["fit_count"] = 0
        cases.append({"fit_diagnostics": invalid_d})
        for options in cases:
            with self.subTest(options=repr(options)), tempfile.TemporaryDirectory() as directory:
                kwargs = {"alpha": 2., "fit_diagnostics": d, "selection_score": .25}
                kwargs.update(options)
                path = Path(directory) / "ridge"
                with self.assertRaises(RidgeError):
                    save_ridge(path, model, self.normalizer, self.config, self.provenance,
                               **kwargs)
                self.assertFalse(path.exists())

    def test_archive_metadata_semantic_changes_cannot_be_hidden_by_outer_hash(self):
        modifications = [
            lambda m: m.update(format="unknown-ridge-version"),
            lambda m: m.update(alpha=True),
            lambda m: m["training_provenance"].update(validation_split="train"),
            lambda m: m.update(normalizer_hash="d" * 64),
            lambda m: m["learning_schema"]["feature_fields"].reverse(),
            lambda m: m["fit_diagnostics"].update(residual_sum_squares=20.),
            lambda m: m["fit_diagnostics"].update(normalized_residual_mse=.5),
        ]
        for index, change in enumerate(modifications):
            with self.subTest(case=index), tempfile.TemporaryDirectory() as directory:
                path, metadata, _, _ = self.saved(directory)
                change(metadata)
                _rebind_untrusted_container(path, metadata)
                with self.assertRaises(RidgeError):
                    self.loaded(path)

    def test_archive_missing_truncated_and_non_json_artifacts_fail_explicitly(self):
        for filename, replacement in (("weights.npz", None), ("weights.npz", b"not-a-zip"),
                                      ("ridge.json", None), ("ridge.json", b"{\"broken\":")):
            with self.subTest(filename=filename, replacement=replacement), tempfile.TemporaryDirectory() as directory:
                path, _, _, _ = self.saved(directory)
                if replacement is None:
                    (path / filename).unlink()
                else:
                    (path / filename).write_bytes(replacement)
                with self.assertRaises(RidgeError):
                    self.loaded(path)


if __name__ == "__main__":
    unittest.main()
