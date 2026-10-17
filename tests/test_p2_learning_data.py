"""Adversarial persistence checks for the preparation/normalizer pair.

The compact arrays are deliberately synthetic transform/storage fixtures.
They do not establish valid physics, alignment performance, or a source
dataset's authenticity. Source provenance hashes identify the fixture only.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
import warnings
import zipfile

import numpy as np

from push_align import PlanarPushEnv
from push_align.data import DataContractError, digest, make_contract
from push_align.learning_data import (
    load_learning_data, save_learning_data, validate_prepared,
)
from push_align.normalization import fit_normalizer, learning_schema
from push_align.preparation import HORIZONS, PreparedData, REGIME_CODES, SPLIT_CODES
from push_align.representation import encode, state_delta


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False), encoding="utf-8")


def clone(prepared):
    return PreparedData({key: value.copy() for key, value in prepared.arrays.items()},
                        deepcopy(prepared.metadata),
                        {key: value.copy() for key, value in prepared.windows.items()})


def fixture(contract):
    chunks = {name: [] for name in (
        "current_state", "next_state", "applied_action", "features", "deltas",
        "split_code", "episode_index", "transition_index", "regime_code",
    )}
    episodes, offset = [], 0
    window_lists = {k: [] for k in HORIZONS}
    split_counts = {}
    for episode_index, (split, code) in enumerate(SPLIT_CODES.items()):
        indices = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
                           if episode_index == 0 else list(range(12)), dtype=np.int64)
        n, recorded = len(indices), int(indices[-1])+1
        def states(at):
            s = np.zeros((len(at), 10), dtype=np.float64)
            s[:, 0] = episode_index*.01 + at*.001
            s[:, 1] = -.005 + at*.0002
            s[:, 2] = .01 + at*.05
            s[:, 3] = .01
            s[:, 5] = .05
            s[:, 6] = s[:, 0]-.055
            s[:, 7] = s[:, 1]+.002
            s[:, 8] = .012
            return s
        current, future = states(indices), states(indices+1)
        action = np.zeros((n, 2), dtype=np.float64)
        action[:, 0] = .012
        values = {
            "current_state": current, "next_state": future,
            "applied_action": action, "features": encode(current, action),
            "deltas": state_delta(current, future),
            "split_code": np.full(n, code, dtype=np.int8),
            "episode_index": np.full(n, episode_index, dtype=np.int64),
            "transition_index": indices,
            "regime_code": (indices % 3).astype(np.int8),
        }
        for key, value in values.items():
            chunks[key].append(value)
        rows = offset + np.arange(n, dtype=np.int64)
        for k in HORIZONS:
            for start in range(n-k+1):
                if np.all(np.diff(indices[start:start+k]) == 1):
                    window_lists[k].append(rows[start:start+k])
        episodes.append({"episode_id": f"fixture-{episode_index}",
                         "episode_index": episode_index,
                         "family_id": f"fixture-family-{episode_index}",
                         "split": split, "table_start": offset, "table_stop": offset+n,
                         "recorded_transition_count": recorded,
                         "valid_transition_count": n, "excluded_transition_count": recorded-n})
        split_counts[split] = {"episodes": 1, "recorded_transitions": recorded,
                              "valid_transitions": n, "excluded_transitions": recorded-n}
        offset += n
    arrays = {key: np.concatenate(value, axis=0) for key, value in chunks.items()}
    windows = {k: np.asarray(value, dtype=np.int64).reshape(-1, k)
               for k, value in window_lists.items()}
    schema = learning_schema(contract)
    metadata = {"schema": "push-align-prepared-v1", "dataset_id": "storage-fixture",
                "manifest_sha256": hashlib.sha256(b"synthetic storage fixture").hexdigest(),
                "data_contract_hash": schema["data_contract_hash"], "contract": deepcopy(contract),
                "episode_count": 3, "valid_transition_count": offset,
                "recorded_transition_count": sum(e["recorded_transition_count"] for e in episodes),
                "excluded_transition_count": 1, "episodes": episodes,
                "split_counts": split_counts, "split_codes": dict(SPLIT_CODES),
                "regime_codes": dict(REGIME_CODES),
                "window_counts": {str(k): len(w) for k, w in windows.items()},
                "scope": "synthetic representation/storage fixture; no physics claim"}
    identity = {k: metadata[k] for k in ("dataset_id", "manifest_sha256", "data_contract_hash")}
    normalizer = fit_normalizer(arrays["features"], arrays["deltas"], arrays["split_code"],
                               schema, identity, arrays["episode_index"], arrays["transition_index"])
    return PreparedData(arrays, metadata, windows), normalizer


class P2LearningDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared, cls.normalizer = fixture(make_contract(PlanarPushEnv()))

    def saved(self, root):
        path = Path(root)/"learning"
        save_learning_data(path, self.prepared, self.normalizer)
        return path

    def load(self, path, **overrides):
        args = {"expected_schema": self.normalizer.metadata["schema"],
                "expected_dataset": self.normalizer.metadata["dataset_identity"],
                "expected_normalizer_hash": self.normalizer.metadata["normalizer_hash"]}
        args.update(overrides)
        return load_learning_data(path, **args)

    def rewrite_arrays(self, path, change):
        file = path/"prepared.npz"
        with np.load(file, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        change(arrays)
        with file.open("wb") as output:
            np.savez_compressed(output, **arrays)
        manifest = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
        manifest["npz_sha256"] = digest(file)
        json_write(path/"manifest.json", manifest)

    def rewrite_metadata(self, path, change):
        manifest = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
        change(manifest["preparation"])
        json_write(path/"manifest.json", manifest)

    def test_roundtrip_preserves_exact_rows_windows_and_train_statistics(self):
        with tempfile.TemporaryDirectory() as root:
            loaded, normalizer = self.load(self.saved(root))
            self.assertEqual(loaded.metadata, self.prepared.metadata)
            self.assertEqual(normalizer.metadata, self.normalizer.metadata)
            for key, values in self.prepared.arrays.items():
                np.testing.assert_array_equal(loaded.arrays[key], values)
            for k, values in self.prepared.windows.items():
                np.testing.assert_array_equal(loaded.windows[k], values)
            self.assertEqual(validate_prepared(loaded, normalizer)["train_fit_count"], 12)

    def test_existing_destination_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.saved(root)
            before = {str(p.relative_to(path)): digest(p) for p in path.rglob("*") if p.is_file()}
            with self.assertRaises(FileExistsError):
                save_learning_data(path, self.prepared, self.normalizer)
            after = {str(p.relative_to(path)): digest(p) for p in path.rglob("*") if p.is_file()}
            self.assertEqual(before, after)

    def test_expected_schema_dataset_and_normalizer_identity_are_required(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.saved(root)
            schema = deepcopy(self.normalizer.metadata["schema"])
            schema["feature_units"][0] = "mm"
            identity = deepcopy(self.normalizer.metadata["dataset_identity"])
            identity["manifest_sha256"] = "f"*64
            for override in ({"expected_schema": schema}, {"expected_dataset": identity},
                             {"expected_normalizer_hash": "f"*64}):
                with self.subTest(override=list(override)):
                    with self.assertRaises(DataContractError):
                        self.load(path, **override)

    def test_raw_file_corruption_is_rejected_by_artifact_hash(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.saved(root)
            with (path/"prepared.npz").open("ab") as file:
                file.write(b"corruption")
            with self.assertRaisesRegex(DataContractError, "artifact hash"):
                self.load(path)

    def test_incomplete_zip_is_reported_as_artifact_error_even_after_file_hash_update(self):
        for file in ("prepared.npz", "normalizer/statistics.npz"):
            with self.subTest(file=file), tempfile.TemporaryDirectory() as root:
                path = self.saved(root)
                archive_path = path/file
                archive_path.write_bytes(b"PK\x03\x04broken")
                manifest = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
                if file == "prepared.npz":
                    manifest["npz_sha256"] = digest(archive_path)
                else:
                    nm = json.loads((path/"normalizer/normalizer.json").read_text(encoding="utf-8"))
                    nm["npz_sha256"] = digest(archive_path)
                    json_write(path/"normalizer/normalizer.json", nm)
                    manifest["normalizer_manifest_sha256"] = digest(path/"normalizer/normalizer.json")
                json_write(path/"manifest.json", manifest)
                with self.assertRaises(DataContractError):
                    self.load(path)

    def test_normalized_value_corruption_with_updated_file_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.saved(root)
            self.rewrite_arrays(path, lambda a: a["normalized_features"].__setitem__((0, 0), 123.))
            with self.assertRaisesRegex(DataContractError, "normalized data"):
                self.load(path)

    def test_recomputed_features_still_require_statistics_bound_to_saved_train_rows(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.saved(root)
            def change(a):
                a["applied_action"][0, 0] += .001
                a["features"] = encode(a["current_state"], a["applied_action"])
                a["normalized_features"] = self.normalizer.normalize_features(a["features"])
            self.rewrite_arrays(path, change)
            with self.assertRaisesRegex(DataContractError, "Statistics do not match"):
                self.load(path)

    def test_targets_must_match_physical_current_and_next_states(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.saved(root)
            self.rewrite_arrays(path, lambda a: a["deltas"].__setitem__((12, 0), .5))
            with self.assertRaisesRegex(DataContractError, "Features or targets"):
                self.load(path)

    def test_compact_rows_cannot_form_window_across_excluded_original_transition(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.saved(root)
            self.rewrite_arrays(path, lambda a: a["windows_5"].__setitem__(0, np.arange(4, 9)))
            with self.assertRaisesRegex(DataContractError, "Window"):
                self.load(path)

    def test_missing_or_duplicated_window_is_rejected_even_when_declared_count_is_updated(self):
        for missing in (True, False):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as root:
                path = self.saved(root)
                def change(a):
                    a["windows_5"] = a["windows_5"][1:] if missing else np.concatenate((a["windows_5"], a["windows_5"][:1]))
                self.rewrite_arrays(path, change)
                count = self.prepared.metadata["window_counts"]["5"] + (-1 if missing else 1)
                self.rewrite_metadata(path, lambda m: m["window_counts"].__setitem__("5", count))
                with self.assertRaisesRegex(DataContractError, "Window"):
                    self.load(path)

    def test_original_adjacent_states_must_meet_at_boundary(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.saved(root)
            def change(a):
                a["next_state"][12, 0] += .0001
                a["deltas"] = state_delta(a["current_state"], a["next_state"])
                a["normalized_deltas"] = self.normalizer.normalize_delta(a["deltas"])
            self.rewrite_arrays(path, change)
            with self.assertRaisesRegex(DataContractError, "Adjacent original states"):
                self.load(path)

    def test_dtype_and_unknown_columns_are_rejected(self):
        for mode in ("dtype", "extra"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root:
                path = self.saved(root)
                def change(a):
                    if mode == "dtype":
                        a["episode_index"] = a["episode_index"].astype(np.float64)
                    else:
                        a["unknown_column"] = np.zeros(1)
                self.rewrite_arrays(path, change)
                with self.assertRaisesRegex(DataContractError, "preparation"):
                    self.load(path)

    def test_split_and_total_metadata_cannot_silently_change(self):
        for change in (lambda m: m["split_counts"]["train"].__setitem__("valid_transitions", 13),
                       lambda m: m.__setitem__("recorded_transition_count", 999)):
            with tempfile.TemporaryDirectory() as root:
                path = self.saved(root)
                self.rewrite_metadata(path, change)
                with self.assertRaises(DataContractError):
                    self.load(path)

    def test_episode_ids_and_family_split_lineage_must_be_preserved(self):
        for field in ("episode_id", "family_id"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as root:
                path = self.saved(root)
                self.rewrite_metadata(path, lambda m: m["episodes"][1].__setitem__(field, m["episodes"][0][field]))
                with self.assertRaises(DataContractError):
                    self.load(path)

    def test_duplicate_npz_member_is_rejected_after_hashes_are_updated(self):
        for file in ("prepared.npz", "normalizer/statistics.npz"):
            with self.subTest(file=file), tempfile.TemporaryDirectory() as root:
                path = self.saved(root)
                archive_path = path/file
                with zipfile.ZipFile(archive_path) as archive:
                    member = archive.namelist()[0]
                    payload = archive.read(member)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    with zipfile.ZipFile(archive_path, "a") as archive:
                        archive.writestr(member, payload)
                if file == "prepared.npz":
                    manifest = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
                    manifest["npz_sha256"] = digest(archive_path)
                else:
                    nm = json.loads((path/"normalizer/normalizer.json").read_text(encoding="utf-8"))
                    nm["npz_sha256"] = digest(archive_path)
                    json_write(path/"normalizer/normalizer.json", nm)
                    manifest = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
                    manifest["normalizer_manifest_sha256"] = digest(path/"normalizer/normalizer.json")
                json_write(path/"manifest.json", manifest)
                with self.assertRaisesRegex(DataContractError, "Duplicate|duplicate"):
                    self.load(path)

    def test_pi_target_cannot_be_saved_as_an_integratable_transition(self):
        prepared = clone(self.prepared)
        a = prepared.arrays
        a["next_state"][0, 2] = a["current_state"][0, 2] + math.pi
        a["current_state"][1, 2] = a["next_state"][0, 2]
        a["features"] = encode(a["current_state"], a["applied_action"])
        a["deltas"] = state_delta(a["current_state"], a["next_state"])
        n = fit_normalizer(a["features"], a["deltas"], a["split_code"], self.normalizer.metadata["schema"],
                           self.normalizer.metadata["dataset_identity"], a["episode_index"], a["transition_index"])
        with self.assertRaises(DataContractError):
            validate_prepared(prepared, n)


if __name__ == "__main__":
    unittest.main()
