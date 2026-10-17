"""Independent train-only statistics, numeric boundary and bundle checks.

Small hand-defined values distinguish population statistics from sample
statistics and prove that held-out values cannot change fitted statistics.
These checks certify data preparation, not learned dynamics performance.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import warnings
import zipfile

import numpy as np

from push_align import PlanarPushEnv
from push_align.data import contract_hash, make_contract
from push_align.normalization import (
    NormalizationError, fit_normalizer, learning_schema, load_normalizer,
    save_normalizer, schema_hash,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def change_field(value, key, change):
    """Change one semantic field without depending on schema nesting."""
    if key in value:
        value[key] = change(value[key])
        return True
    for child in value.values():
        if isinstance(child, dict) and change_field(child, key, change):
            return True
    return False


class P2NormalizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = make_contract(PlanarPushEnv())
        cls.schema = learning_schema(cls.contract)
        cls.identity = {
            "dataset_id": "normalization-fixture",
            "manifest_sha256": "a" * 64,
            "data_contract_hash": contract_hash(cls.contract),
        }

    def setUp(self):
        self.features = np.array([
            [2., 4., 0., 1., .01, .03, .2, .02, .04, .01, .001],
            [4., 8., .5, .5, .03, .07, .4, .06, .08, .03, .001],
            [90., 80., -.5, .5, 9., 8., 7., 6., 5., 4., 3.],
            [-90., -80., .25, .75, -9., -8., -7., -6., -5., -4., -3.],
        ], dtype=np.float64)
        self.deltas = np.array([
            [.0002, .0004, .002, .002, .004, .02, .0002, .0006, .002, .004],
            [.0006, .0012, .006, .006, .012, .06, .0006, .0018, .006, .012],
            [9., 8., .1, 6., 5., 4., 3., 2., 1., .5],
            [-9., -8., -.1, -6., -5., -4., -3., -2., -1., -.5],
        ], dtype=np.float64)
        self.splits = np.array([0, 0, 1, 2], dtype=np.int8)
        self.episodes = np.array([0, 0, 1, 2], dtype=np.int64)
        self.transitions = np.array([0, 1, 0, 0], dtype=np.int64)

    def fit(self, *, features=None, deltas=None, splits=None, identity=None,
            episodes=None, transitions=None, schema=None):
        return fit_normalizer(
            self.features if features is None else features,
            self.deltas if deltas is None else deltas,
            self.splits if splits is None else splits,
            self.schema if schema is None else schema,
            self.identity if identity is None else identity,
            self.episodes if episodes is None else episodes,
            self.transitions if transitions is None else transitions,
        )

    def saved(self, root):
        path = Path(root) / "normalizer"
        normalizer = self.fit()
        save_normalizer(path, normalizer)
        return path, normalizer

    def test_schema_hash_is_order_independent_but_binds_semantics(self):
        reverse = dict(reversed(list(self.schema.items())))
        self.assertEqual(schema_hash(self.schema), schema_hash(reverse))
        changed = deepcopy(self.schema)
        self.assertTrue(change_field(changed, "feature_fields", lambda x: [x[1], x[0], *x[2:]]))
        self.assertNotEqual(schema_hash(self.schema), schema_hash(changed))

    def test_hand_defined_train_population_statistics_floors_and_fixed_angles(self):
        normalizer = self.fit()
        expected_mean = np.array([3., 6., 0., 0., .02, .05, .3, .04, .06, .02, .001])
        expected_std = np.array([1., 2., .25, .25, .01, .02, .1, .02, .02, .01, 0.])
        expected_scale = expected_std.copy()
        expected_scale[[2, 3]] = 1.
        expected_scale[10] = .001
        np.testing.assert_allclose(normalizer.arrays["input_mean"], expected_mean, atol=1e-16, rtol=1e-15)
        np.testing.assert_allclose(normalizer.arrays["input_std"], expected_std, atol=1e-16, rtol=1e-15)
        np.testing.assert_allclose(normalizer.arrays["input_scale"], expected_scale, atol=1e-16, rtol=1e-15)
        expected_delta_mean = np.array([.0004, .0008, .004, .004, .008, .04, .0004, .0012, .004, .008])
        expected_delta_std = np.array([.0002, .0004, .002, .002, .004, .02, .0002, .0006, .002, .004])
        np.testing.assert_allclose(normalizer.arrays["delta_mean"], expected_delta_mean, atol=1e-16, rtol=1e-15)
        np.testing.assert_allclose(normalizer.arrays["delta_std"], expected_delta_std, atol=1e-16, rtol=1e-15)
        np.testing.assert_allclose(normalizer.arrays["delta_scale"], expected_delta_std, atol=1e-16, rtol=1e-15)
        self.assertEqual(normalizer.metadata["fit_split"], "train")
        self.assertEqual(normalizer.metadata["fit_count"], 2)

    def test_held_out_mutation_preserves_statistics_and_train_hash_but_changes_provenance(self):
        original = self.fit()
        features, deltas = self.features.copy(), self.deltas.copy()
        features[2:] *= 123.
        deltas[2:] -= 50.
        identity = deepcopy(self.identity)
        identity["manifest_sha256"] = "b" * 64
        changed = self.fit(features=features, deltas=deltas, identity=identity)
        for key in original.arrays:
            np.testing.assert_array_equal(original.arrays[key], changed.arrays[key])
        for key in ("stats_hash", "train_rows_sha256"):
            self.assertEqual(original.metadata[key], changed.metadata[key])
        self.assertNotEqual(original.metadata["normalizer_hash"], changed.metadata["normalizer_hash"])

    def test_train_mutation_positive_control_changes_statistics_and_train_hash(self):
        original = self.fit()
        features = self.features.copy()
        features[0, 0] += 1.
        changed = self.fit(features=features)
        self.assertNotEqual(original.metadata["stats_hash"], changed.metadata["stats_hash"])
        self.assertNotEqual(original.metadata["train_rows_sha256"], changed.metadata["train_rows_sha256"])
        self.assertNotEqual(original.arrays["input_mean"][0], changed.arrays["input_mean"][0])

    def test_nontrain_rows_cannot_become_fit_data_and_unknown_split_is_rejected(self):
        for splits in (np.array([1, 1, 2, 2], dtype=np.int8),
                       np.array([0, 0, 1, 3], dtype=np.int8),
                       np.array([0, 0, 1, 4], dtype=np.int8)):
            with self.subTest(splits=splits.tolist()), self.assertRaises(NormalizationError):
                self.fit(splits=splits)

    def test_repeated_original_transition_identity_is_rejected(self):
        transitions = self.transitions.copy()
        transitions[1] = 0
        with self.assertRaises(NormalizationError):
            self.fit(transitions=transitions)

    def test_fit_rejects_bad_shape_dtype_row_count_and_nonfinite_values(self):
        bad_features = [self.features[:, :10], self.features.astype(np.float32),
                        self.features[:3], self.features.astype(np.int64)]
        nonfinite = self.features.copy()
        nonfinite[0, 0] = np.nan
        bad_features.append(nonfinite)
        for features in bad_features:
            with self.subTest(shape=features.shape, dtype=str(features.dtype)), self.assertRaises(NormalizationError):
                self.fit(features=features)
        for replacements in (
            {"deltas": self.deltas[:, :9]},
            {"splits": self.splits.astype(np.int64)},
            {"episodes": self.episodes.astype(np.int32)},
            {"transitions": self.transitions[:3]},
        ):
            with self.subTest(replacements=list(replacements)), self.assertRaises(NormalizationError):
                self.fit(**replacements)

    def test_fit_rejects_missing_schema_fields_and_invalid_scale_floors(self):
        missing = deepcopy(self.schema)
        del missing["normalization"]
        for schema in [missing]:
            with self.subTest(case="missing-normalization"), self.assertRaises(NormalizationError):
                self.fit(schema=schema)
        for value in (0., -.001, np.nan, np.inf):
            schema = deepcopy(self.schema)
            schema["normalization"]["input_scale_floor"][0] = value
            with self.subTest(floor=value), self.assertRaises(NormalizationError):
                self.fit(schema=schema)

    def test_singleton_and_constant_columns_use_positive_configured_floors(self):
        normalizer = self.fit(features=self.features[:1], deltas=self.deltas[:1],
                              splits=self.splits[:1], episodes=self.episodes[:1],
                              transitions=self.transitions[:1])
        floors = self.contract["resolved_config"]["normalization"]
        expected_input = np.array(floors["input_scale_floor"], dtype=np.float64)
        np.testing.assert_array_equal(normalizer.arrays["input_scale"], expected_input)
        np.testing.assert_array_equal(normalizer.arrays["delta_scale"], floors["delta_scale_floor"])
        np.testing.assert_array_equal(normalizer.arrays["input_std"], np.zeros(11))
        np.testing.assert_array_equal(normalizer.arrays["delta_std"], np.zeros(10))
        np.testing.assert_array_equal(normalizer.arrays["input_mean"][[2, 3]], [0., 0.])
        self.assertTrue(np.isfinite(normalizer.normalize_features(self.features[:1])).all())

    def test_numpy_roundtrip_preserves_leading_shape_and_explicit_dtypes(self):
        normalizer = self.fit()
        for values, normalize, denormalize in (
            (self.features[:2].reshape(1, 2, 11), normalizer.normalize_features, normalizer.denormalize_features),
            (self.deltas[:2].reshape(1, 2, 10), normalizer.normalize_delta, normalizer.denormalize_delta),
        ):
            normalized = normalize(values)
            restored = denormalize(normalized)
            self.assertEqual(normalized.dtype, np.dtype("float32"))
            self.assertEqual(restored.dtype, np.dtype("float64"))
            self.assertEqual(restored.shape, values.shape)
            np.testing.assert_allclose(normalize(restored), normalized, atol=1e-6, rtol=1e-5)
            np.testing.assert_allclose(restored, values, atol=1e-8, rtol=1e-6)

    def test_numpy_transforms_reject_wrong_width_integer_and_nonfinite_input(self):
        normalizer = self.fit()
        for operation, width in (
            (normalizer.normalize_features, 11), (normalizer.denormalize_features, 11),
            (normalizer.normalize_delta, 10), (normalizer.denormalize_delta, 10),
        ):
            for value in (np.zeros((2, width - 1), dtype=np.float64),
                          np.zeros((2, width), dtype=np.int64),
                          np.full((2, width), np.inf, dtype=np.float64)):
                with self.subTest(operation=operation.__name__, shape=value.shape), self.assertRaises(NormalizationError):
                    operation(value)

    def test_save_load_restores_exact_statistics_and_hashes(self):
        with tempfile.TemporaryDirectory() as root:
            path, original = self.saved(root)
            restored = load_normalizer(path, self.schema, self.identity,
                                       original.metadata["normalizer_hash"])
            self.assertEqual(original.metadata, restored.metadata)
            for key in original.arrays:
                self.assertEqual(restored.arrays[key].dtype, np.dtype("float64"))
                np.testing.assert_array_equal(original.arrays[key], restored.arrays[key])
            np.testing.assert_array_equal(original.normalize_features(self.features),
                                          restored.normalize_features(self.features))

    def test_shape_compatible_schema_order_units_time_and_environment_mismatches_are_rejected(self):
        changes = {
            "feature_fields": lambda x: [x[1], x[0], *x[2:]],
            "feature_units": lambda x: ["mm", *x[1:]],
            "action_dt_s": lambda x: x * 2,
            "environment_hash": lambda x: "b" * 64 if x != "b" * 64 else "c" * 64,
        }
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.saved(root)
            for field, change in changes.items():
                with self.subTest(field=field):
                    schema = deepcopy(self.schema)
                    self.assertTrue(change_field(schema, field, change), f"Learning schema must bind {field}")
                    with self.assertRaises(NormalizationError):
                        load_normalizer(path, schema, self.identity)

    def test_dataset_identity_and_content_sha_mismatches_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.saved(root)
            for field, value in (("dataset_id", "other-data"),
                                 ("manifest_sha256", "b" * 64),
                                 ("data_contract_hash", "b" * 64)):
                identity = deepcopy(self.identity)
                identity[field] = value
                with self.subTest(field=field), self.assertRaises(NormalizationError):
                    load_normalizer(path, self.schema, identity)

    def test_expected_normalizer_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.saved(root)
            with self.assertRaises(NormalizationError):
                load_normalizer(path, self.schema, self.identity, "f" * 64)

    def test_damaged_npz_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.saved(root)
            npz = path / "statistics.npz"
            with npz.open("ab") as file:
                file.write(b"changed archived bytes")
            with self.assertRaises(NormalizationError):
                load_normalizer(path, self.schema, self.identity)

    def test_metadata_content_and_hash_tampering_is_rejected(self):
        for field in ("fit_count", "stats_hash", "schema_hash", "normalizer_hash"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as root:
                path, _ = self.saved(root)
                metadata_path = path / "normalizer.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                inner = metadata["normalizer"]
                inner[field] = inner[field] + 1 if field == "fit_count" else "f" * 64
                metadata_path.write_text(json.dumps(metadata, allow_nan=False), encoding="utf-8")
                with self.assertRaises(NormalizationError):
                    load_normalizer(path, self.schema, self.identity)

    def test_shape_compatible_stats_change_is_rejected_even_after_file_hash_update(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.saved(root)
            npz, metadata_path = path / "statistics.npz", path / "normalizer.json"
            with np.load(npz, allow_pickle=False) as saved:
                arrays = {key: saved[key].copy() for key in saved.files}
            arrays["delta_mean"][0] += .1
            np.savez_compressed(npz, **arrays)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertTrue(change_field(metadata, "npz_sha256", lambda x: sha(npz)))
            metadata_path.write_text(json.dumps(metadata, allow_nan=False), encoding="utf-8")
            with self.assertRaises(NormalizationError):
                load_normalizer(path, self.schema, self.identity)

    def test_duplicate_npz_members_are_rejected_even_with_matching_file_hash(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.saved(root)
            npz, metadata_path = path / "statistics.npz", path / "normalizer.json"
            with zipfile.ZipFile(npz) as archive:
                first = archive.namelist()[0]
                repeated = archive.read(first)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(npz, "a") as archive:
                    archive.writestr(first, repeated)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["npz_sha256"] = sha(npz)
            metadata_path.write_text(json.dumps(metadata, allow_nan=False), encoding="utf-8")
            with self.assertRaises(NormalizationError):
                load_normalizer(path, self.schema, self.identity)

    def test_torch_transforms_preserve_dtype_device_and_gradients(self):
        import torch

        normalizer = self.fit()
        for dtype in (torch.float32, torch.float64):
            for values, normalize, denormalize, scale_key in (
                (self.features[:2], normalizer.normalize_features, normalizer.denormalize_features, "input_scale"),
                (self.deltas[:2], normalizer.normalize_delta, normalizer.denormalize_delta, "delta_scale"),
            ):
                with self.subTest(dtype=str(dtype), scale=scale_key):
                    source = torch.tensor(values, dtype=dtype, device="cpu", requires_grad=True)
                    normalized = normalize(source)
                    self.assertEqual(normalized.dtype, dtype)
                    self.assertEqual(normalized.device, source.device)
                    normalized.sum().backward()
                    expected = 1 / torch.tensor(normalizer.arrays[scale_key], dtype=dtype, device=source.device)
                    torch.testing.assert_close(source.grad, expected.expand_as(source), atol=1e-6, rtol=1e-5)
                    source.grad = None
                    restored = denormalize(normalize(source))
                    self.assertEqual(restored.dtype, dtype)
                    self.assertEqual(restored.device, source.device)
                    torch.testing.assert_close(restored, source, atol=1e-6, rtol=1e-5)
                    restored.sum().backward()
                    torch.testing.assert_close(source.grad, torch.ones_like(source), atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
