"""Train-only, schema-bound normalization shared by learning and rollout.

Statistics bind only original train rows. The complete artifact separately
binds the entire dataset manifest: changing validation changes provenance,
not train statistics. No optimizer or dynamics model lives in this module.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from zipfile import BadZipFile

import numpy as np

from .data import _json_bytes, _read_json, _validate_contract, _write_json, contract_hash, digest
from .representation import (STATE_FIELDS, STATE_UNITS, FEATURE_FIELDS, FEATURE_UNITS,
                             DELTA_FIELDS, DELTA_UNITS, _descriptor, _finite)


class NormalizationError(ValueError):
    """Normalization data, statistics or provenance is incompatible."""


def _hash(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def schema_hash(schema):
    return _hash(schema)


def learning_schema(contract):
    _validate_contract(contract)
    cfg = contract["resolved_config"]
    declared = cfg["schema"]
    for key, expected in (("state_fields", STATE_FIELDS), ("state_units", STATE_UNITS),
                          ("feature_fields", FEATURE_FIELDS), ("delta_fields", DELTA_FIELDS)):
        if declared[key] != list(expected):
            raise NormalizationError(f"Unsupported {key}")
    for key, expected in (("yaw_storage", "wrap_2pi_minus_pi_inclusive"), ("task_yaw_error", "symmetric_pi"),
                          ("learning_yaw_error", "wrap_2pi"), ("action_semantics", "constant_target_velocity_per_action_interval")):
        if declared.get(key) != expected:
            raise NormalizationError(f"Unsupported schema meaning {key}")
    policy = deepcopy(cfg["normalization"])
    if policy["fit_split"] != "train" or policy["input_fixed_unit_indices"] != [2, 3] or policy["rollout_error_scale"] != "train_delta_scale":
        raise NormalizationError("Unsupported normalization policy")
    for key, width in (("input_scale_floor", 11), ("delta_scale_floor", 10)):
        a = np.asarray(policy[key], dtype=np.float64)
        if a.shape != (width,) or not np.isfinite(a).all() or (a <= 0).any():
            raise NormalizationError(f"Invalid {key}")
    if cfg["runtime"]["model_dtype"] != "float32":
        raise NormalizationError("Only the registered float32 model boundary is supported")
    return {"schema_version": "push-align-learning-v1", "data_contract_hash": contract_hash(contract),
            "state_fields": list(STATE_FIELDS), "state_units": list(STATE_UNITS),
            "feature_fields": list(FEATURE_FIELDS), "feature_units": list(FEATURE_UNITS),
            "delta_fields": list(DELTA_FIELDS), "delta_units": list(DELTA_UNITS),
            "coordinates": "world_SI", "feature_yaw": "sin_2yaw_cos_2yaw_pi_symmetry",
            "delta_yaw": "wrap_2pi; interval absolute displacement strictly below pi",
            "action_semantics": "applied constant target world XY velocity, not measured response",
            "action_dt_s": contract["action_dt_s"], "environment_hash": contract["environment_hash"],
            "compiled_model_hash": contract["compiled_model_hash"], "config_sha256": contract["config_sha256"],
            "profile_sha256": contract["profile_sha256"], "physical_dtype": "float64",
            "model_dtype": "float32", "normalization": policy}


def _validate_schema(schema):
    if not isinstance(schema, dict):
        raise NormalizationError("Learning schema must be an object")
    meanings = {"coordinates": "world_SI", "feature_yaw": "sin_2yaw_cos_2yaw_pi_symmetry",
                "delta_yaw": "wrap_2pi; interval absolute displacement strictly below pi",
                "action_semantics": "applied constant target world XY velocity, not measured response"}
    for key, expected in meanings.items():
        if schema.get(key) != expected:
            raise NormalizationError(f"Incompatible learning meaning {key}")
    for key in ("data_contract_hash", "environment_hash", "compiled_model_hash", "config_sha256", "profile_sha256"):
        value = schema.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise NormalizationError(f"Invalid learning schema hash {key}")
    for key, expected in (("state_fields", STATE_FIELDS), ("state_units", STATE_UNITS),
                          ("feature_fields", FEATURE_FIELDS), ("feature_units", FEATURE_UNITS),
                          ("delta_fields", DELTA_FIELDS), ("delta_units", DELTA_UNITS)):
        if schema.get(key) != list(expected):
            raise NormalizationError(f"Incompatible learning schema {key}")
    if schema.get("schema_version") != "push-align-learning-v1" or schema.get("physical_dtype") != "float64" or schema.get("model_dtype") != "float32":
        raise NormalizationError("Incompatible learning schema version/dtype")
    policy = schema.get("normalization", {})
    if policy.get("fit_split") != "train" or policy.get("input_fixed_unit_indices") != [2, 3] or policy.get("rollout_error_scale") != "train_delta_scale":
        raise NormalizationError("Incompatible normalization policy")
    for key, width in (("input_scale_floor", 11), ("delta_scale_floor", 10)):
        a = np.asarray(policy.get(key), dtype=np.float64)
        if a.shape != (width,) or not np.isfinite(a).all() or (a <= 0).any():
            raise NormalizationError(f"Invalid {key}")
    if type(schema.get("action_dt_s")) not in (int, float) or not np.isfinite(schema["action_dt_s"]) or schema["action_dt_s"] <= 0:
        raise NormalizationError("Invalid action duration")


def _identity(identity, schema):
    if set(identity) != {"dataset_id", "manifest_sha256", "data_contract_hash"} or not isinstance(identity["dataset_id"], str) or not identity["dataset_id"]:
        raise NormalizationError("Invalid dataset identity")
    for key in ("manifest_sha256", "data_contract_hash"):
        value = identity[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise NormalizationError(f"Invalid dataset identity {key}")
    if identity["data_contract_hash"] != schema["data_contract_hash"]:
        raise NormalizationError("Dataset contract differs from learning schema")


def _stats_payload(arrays, metadata):
    return {"schema_hash": metadata["schema_hash"], "fit_split": metadata["fit_split"],
            "fit_count": metadata["fit_count"], "train_rows_sha256": metadata["train_rows_sha256"],
            "arrays": {k: v.tolist() for k, v in sorted(arrays.items())}}


def _content_hash(metadata):
    return _hash({k: v for k, v in metadata.items() if k != "normalizer_hash"})


@dataclass(frozen=True)
class Normalizer:
    arrays: dict
    metadata: dict

    def __post_init__(self):
        copies = {k: np.array(v, copy=True) for k, v in self.arrays.items()}
        for value in copies.values():
            value.flags.writeable = False
        object.__setattr__(self, "arrays", copies)
        object.__setattr__(self, "metadata", deepcopy(self.metadata))
        self.validate()

    def validate(self):
        m = self.metadata
        _validate_schema(m["schema"]); _identity(m["dataset_identity"], m["schema"])
        if m["schema_hash"] != schema_hash(m["schema"]) or m["fit_split"] != "train" or type(m["fit_count"]) is not int or m["fit_count"] < 1:
            raise NormalizationError("Invalid schema hash or train fit count")
        keys = {f"{p}_{s}" for p in ("input", "delta") for s in ("mean", "std", "scale")}
        if set(self.arrays) != keys:
            raise NormalizationError("Unexpected statistics columns")
        for p, width in (("input", 11), ("delta", 10)):
            for suffix in ("mean", "std", "scale"):
                a = self.arrays[f"{p}_{suffix}"]
                if a.dtype != np.float64 or a.shape != (width,) or not np.isfinite(a).all():
                    raise NormalizationError("Invalid statistics dtype/shape/value")
            std, scale = self.arrays[f"{p}_std"], self.arrays[f"{p}_scale"]
            if (std < 0).any():
                raise NormalizationError("Negative standard deviation")
            expected = np.maximum(std, m["schema"]["normalization"][f"{p}_scale_floor"])
            if p == "input":
                expected[2:4] = 1
                if not np.array_equal(self.arrays["input_mean"][2:4], [0., 0.]):
                    raise NormalizationError("Fixed angle features must have zero mean")
            if not np.array_equal(scale, expected):
                raise NormalizationError("Scale violates registered floor/fixed-unit rule")
        if m["stats_hash"] != _hash(_stats_payload(self.arrays, m)) or m["normalizer_hash"] != _content_hash(m):
            raise NormalizationError("Statistics or metadata hash mismatch")

    def _transform(self, value, prefix, inverse):
        self.validate()
        width = 11 if prefix == "input" else 10
        try:
            backend = _descriptor(value, width, "normalization input")
            _finite(value, backend, "normalization input")
        except ValueError as error:
            raise NormalizationError(str(error)) from error
        mean, scale = self.arrays[f"{prefix}_mean"], self.arrays[f"{prefix}_scale"]
        if backend == "numpy":
            physical = value.astype(np.float64, copy=False)
            with np.errstate(over="ignore", invalid="ignore"):
                output = physical*scale + mean if inverse else ((physical-mean)/scale).astype(np.float32)
        else:
            torch = sys.modules["torch"]
            mean = torch.tensor(mean, dtype=value.dtype, device=value.device)
            scale = torch.tensor(scale, dtype=value.dtype, device=value.device)
            output = value*scale + mean if inverse else (value-mean)/scale
        try:
            _finite(output, backend, "normalization output")
        except ValueError as error:
            raise NormalizationError(str(error)) from error
        return output

    def normalize_features(self, value):
        return self._transform(value, "input", False)

    def denormalize_features(self, value):
        return self._transform(value, "input", True)

    def normalize_delta(self, value):
        return self._transform(value, "delta", False)

    def denormalize_delta(self, value):
        return self._transform(value, "delta", True)


def fit_normalizer(features, deltas, split_codes, schema, dataset_identity, episode_indices, transition_indices):
    _validate_schema(schema); _identity(dataset_identity, schema)
    n = len(features)
    for a, width in ((features, 11), (deltas, 10)):
        if not isinstance(a, np.ndarray) or a.dtype != np.float64 or a.shape != (n, width) or not np.isfinite(a).all():
            raise NormalizationError("Fit requires finite physical float64 rows")
    if not isinstance(split_codes, np.ndarray) or split_codes.dtype != np.int8 or split_codes.shape != (n,) or not np.isin(split_codes, [0, 1, 2]).all():
        raise NormalizationError("Invalid pilot split codes")
    for a in (episode_indices, transition_indices):
        if not isinstance(a, np.ndarray) or a.dtype != np.int64 or a.shape != (n,) or (a < 0).any():
            raise NormalizationError("Invalid original transition identity")
    identities = np.column_stack((episode_indices, transition_indices))
    if len(np.unique(identities, axis=0)) != n:
        raise NormalizationError("Repeated original transition; windows cannot weight statistics")
    for episode in np.unique(episode_indices):
        if len(np.unique(split_codes[episode_indices == episode])) != 1:
            raise NormalizationError("An episode cannot cross splits")
    train = split_codes == 0
    if not train.any():
        raise NormalizationError("No valid train transitions")
    arrays = {}
    row_digest = hashlib.sha256()
    for a in (identities[train], features[train], deltas[train]):
        row_digest.update(np.ascontiguousarray(a).astype(a.dtype.newbyteorder("<"), copy=False).tobytes())
    for p, a in (("input", features[train]), ("delta", deltas[train])):
        mean, std = a.mean(axis=0), a.std(axis=0, ddof=0)
        scale = np.maximum(std, schema["normalization"][f"{p}_scale_floor"])
        if p == "input":
            mean[2:4] = 0; scale[2:4] = 1
        arrays.update({f"{p}_mean": mean, f"{p}_std": std, f"{p}_scale": scale})
    m = {"schema": deepcopy(schema), "schema_hash": schema_hash(schema),
         "dataset_identity": deepcopy(dataset_identity), "fit_split": "train",
         "fit_count": int(train.sum()), "train_rows_sha256": row_digest.hexdigest(),
         "fit_unit": "original valid transition once; no window duplication", "std_ddof": 0}
    m["stats_hash"] = _hash(_stats_payload(arrays, m))
    m["normalizer_hash"] = _content_hash(m)
    return Normalizer(arrays, m)


def save_normalizer(path, normalizer):
    normalizer.validate()
    path = Path(path); path.mkdir(parents=True, exist_ok=False)
    with (path/"statistics.npz").open("xb") as file:
        np.savez_compressed(file, **normalizer.arrays)
    metadata = {"format": "push-align-normalizer-v1", "npz": "statistics.npz",
                "npz_sha256": digest(path/"statistics.npz"), "normalizer": normalizer.metadata}
    _write_json(path/"normalizer.json", metadata)
    return metadata


def load_normalizer(path, expected_schema, expected_dataset, expected_normalizer_hash=None):
    path = Path(path)
    try:
        m = _read_json(path/"normalizer.json")
        if m["format"] != "push-align-normalizer-v1" or m["npz"] != "statistics.npz" or digest(path/"statistics.npz") != m["npz_sha256"]:
            raise NormalizationError("Normalizer file hash or format mismatch")
        if m["normalizer"]["schema"] != expected_schema or m["normalizer"]["dataset_identity"] != expected_dataset:
            raise NormalizationError("Normalizer schema or dataset provenance mismatch")
        if expected_normalizer_hash is not None and m["normalizer"]["normalizer_hash"] != expected_normalizer_hash:
            raise NormalizationError("Unexpected normalizer hash")
        with (path/"statistics.npz").open("rb") as source:
            with np.load(source, allow_pickle=False) as archive:
                if len(set(archive.files)) != len(archive.files):
                    raise NormalizationError("Duplicate statistics NPZ members")
                normalizer = Normalizer({k: archive[k] for k in archive.files}, m["normalizer"])
        return normalizer
    except (KeyError, TypeError, OSError, ValueError, BadZipFile, EOFError) as error:
        if isinstance(error, NormalizationError):
            raise
        raise NormalizationError(f"Invalid normalizer artifact: {error}") from error
