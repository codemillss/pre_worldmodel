"""Persist a checked preparation/normalizer pair for the next learning step."""
from __future__ import annotations

from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from .data import DataContractError, _read_json, _write_json, digest
from .normalization import fit_normalizer, learning_schema, load_normalizer, save_normalizer
from .preparation import PreparedData, HORIZONS, SPLIT_CODES, REGIME_CODES
from .representation import encode, state_delta, integrate, RepresentationError


def validate_prepared(prepared, normalizer):
    normalizer.validate()
    a, m = prepared.arrays, prepared.metadata
    identity = {k: m[k] for k in ("dataset_id", "manifest_sha256", "data_contract_hash")}
    if normalizer.metadata["schema"] != learning_schema(m["contract"]) or normalizer.metadata["dataset_identity"] != identity:
        raise DataContractError("Preparation/normalizer schema or provenance mismatch")
    if m["schema"] != "push-align-prepared-v1" or m["split_codes"] != SPLIT_CODES or m["regime_codes"] != REGIME_CODES:
        raise DataContractError("Unexpected preparation schema/codes")
    n = m["valid_transition_count"]
    expected = {"current_state": (np.float64, (n, 10)), "next_state": (np.float64, (n, 10)),
                "applied_action": (np.float64, (n, 2)), "features": (np.float64, (n, 11)),
                "deltas": (np.float64, (n, 10)), "split_code": (np.int8, (n,)),
                "regime_code": (np.int8, (n,)), "episode_index": (np.int64, (n,)),
                "transition_index": (np.int64, (n,))}
    if set(a) != set(expected):
        raise DataContractError("Unexpected preparation columns")
    for key, (dtype, shape) in expected.items():
        if a[key].dtype != dtype or a[key].shape != shape or not np.isfinite(a[key]).all():
            raise DataContractError(f"Invalid preparation {key}")
    if not np.array_equal(a["features"], encode(a["current_state"], a["applied_action"])) or not np.array_equal(a["deltas"], state_delta(a["current_state"], a["next_state"])):
        raise DataContractError("Features or targets do not match physical transitions")
    # This checks the decoder's principal-delta contract. It cannot certify
    # microstep winding from endpoints; that evidence belongs to preparation.
    try:
        reconstructed = integrate(a["current_state"], a["deltas"])
    except RepresentationError as error:
        raise DataContractError(f"Unintegratable physical target: {error}") from error
    if not np.all(np.abs(state_delta(reconstructed, a["next_state"])) <= 1e-10):
        raise DataContractError("Physical target integration does not reconstruct next state")
    if not np.isin(a["split_code"], [0, 1, 2]).all() or not np.isin(a["regime_code"], [0, 1, 2]).all():
        raise DataContractError("Invalid preparation split/regime codes")
    # Train-row refit binds statistics to the actual saved physical rows, not
    # just a same-shaped NPZ and an independently self-consistent metadata file.
    refit = fit_normalizer(a["features"], a["deltas"], a["split_code"], normalizer.metadata["schema"],
                           identity, a["episode_index"], a["transition_index"])
    if refit.metadata["normalizer_hash"] != normalizer.metadata["normalizer_hash"]:
        raise DataContractError("Statistics do not match saved train rows")
    offset = 0
    seen, ids, family_splits = set(), set(), {}
    for episode in m["episodes"]:
        start, stop = episode["table_start"], episode["table_stop"]
        e = episode["episode_index"]
        if episode["episode_id"] in ids or family_splits.get(episode["family_id"], episode["split"]) != episode["split"]:
            raise DataContractError("Prepared episode ID duplicated or family crosses splits")
        ids.add(episode["episode_id"]); family_splits[episode["family_id"]] = episode["split"]
        if start != offset or stop < start or stop > n or e in seen or stop-start != episode["valid_transition_count"]:
            raise DataContractError("Invalid preparation episode table boundaries")
        indices = a["transition_index"][start:stop]
        if not np.all(a["episode_index"][start:stop] == e) or not np.all(a["split_code"][start:stop] == SPLIT_CODES[episode["split"]]):
            raise DataContractError("Episode table identity/split mismatch")
        if (indices < 0).any() or (indices >= episode["recorded_transition_count"]).any() or not np.all(np.diff(indices) > 0):
            raise DataContractError("Invalid original transition indices")
        if episode["excluded_transition_count"] != episode["recorded_transition_count"]-(stop-start):
            raise DataContractError("Episode excluded count mismatch")
        if len(indices) > 1:
            adjacent = np.diff(indices) == 1
            if not np.array_equal(a["next_state"][start:stop-1][adjacent], a["current_state"][start+1:stop][adjacent]):
                raise DataContractError("Adjacent original states do not meet")
        offset = stop; seen.add(e)
    if offset != n or len(seen) != m["episode_count"] or sum(e["recorded_transition_count"] for e in m["episodes"]) != m["recorded_transition_count"] or m["recorded_transition_count"]-n != m["excluded_transition_count"]:
        raise DataContractError("Preparation total counts mismatch")
    expected_counts = {}
    for split, code in SPLIT_CODES.items():
        es = [e for e in m["episodes"] if e["split"] == split]
        expected_counts[split] = {"episodes": len(es), "recorded_transitions": sum(e["recorded_transition_count"] for e in es),
                                  "valid_transitions": int((a["split_code"] == code).sum()),
                                  "excluded_transitions": sum(e["excluded_transition_count"] for e in es)}
    if m["split_counts"] != expected_counts or set(prepared.windows) != set(HORIZONS):
        raise DataContractError("Preparation split counts or horizons mismatch")
    for k, windows in prepared.windows.items():
        expected_rows = []
        for e in m["episodes"]:
            rows = np.arange(e["table_start"], e["table_stop"], dtype=np.int64)
            original = a["transition_index"][rows]
            for start in range(len(rows)-k+1):
                if np.all(np.diff(original[start:start+k]) == 1):
                    expected_rows.append(rows[start:start+k])
        target = np.asarray(expected_rows, dtype=np.int64).reshape(-1, k)
        if windows.dtype != np.int64 or not np.array_equal(windows, target) or m["window_counts"][str(k)] != len(windows):
            raise DataContractError("Window missing, duplicated or crossing original time gaps")
    return {"valid_transitions": n, "train_fit_count": normalizer.metadata["fit_count"], "windows": m["window_counts"]}


def save_learning_data(path, prepared, normalizer):
    validate_prepared(prepared, normalizer)
    path = Path(path); path.mkdir(parents=True, exist_ok=False)
    save_normalizer(path/"normalizer", normalizer)
    arrays = {**prepared.arrays, **{f"windows_{k}": v for k, v in prepared.windows.items()},
              "normalized_features": normalizer.normalize_features(prepared.arrays["features"]),
              "normalized_deltas": normalizer.normalize_delta(prepared.arrays["deltas"])}
    with (path/"prepared.npz").open("xb") as file:
        np.savez_compressed(file, **arrays)
    manifest = {"format": "push-align-learning-data-v1", "npz": "prepared.npz", "npz_sha256": digest(path/"prepared.npz"),
                "normalizer_manifest_sha256": digest(path/"normalizer/normalizer.json"),
                "normalizer_hash": normalizer.metadata["normalizer_hash"], "preparation": prepared.metadata}
    _write_json(path/"manifest.json", manifest)
    return manifest


def load_learning_data(path, expected_schema, expected_dataset, expected_normalizer_hash=None):
    path = Path(path)
    try:
        m = _read_json(path/"manifest.json")
        if m["format"] != "push-align-learning-data-v1" or m["npz"] != "prepared.npz" or digest(path/"prepared.npz") != m["npz_sha256"] or digest(path/"normalizer/normalizer.json") != m["normalizer_manifest_sha256"]:
            raise DataContractError("Learning-data artifact hash/format mismatch")
        if expected_normalizer_hash is not None and m["normalizer_hash"] != expected_normalizer_hash:
            raise DataContractError("Unexpected learning-data normalizer hash")
        normalizer = load_normalizer(path/"normalizer", expected_schema, expected_dataset, m["normalizer_hash"])
        with (path/"prepared.npz").open("rb") as source:
            with np.load(source, allow_pickle=False) as archive:
                if len(set(archive.files)) != len(archive.files):
                    raise DataContractError("Duplicate prepared NPZ member")
                all_arrays = {k: archive[k].copy() for k in archive.files}
        windows = {k: all_arrays.pop(f"windows_{k}") for k in HORIZONS}
        x, y = all_arrays.pop("normalized_features"), all_arrays.pop("normalized_deltas")
        prepared = PreparedData(all_arrays, m["preparation"], windows)
        validate_prepared(prepared, normalizer)
        if x.dtype != np.float32 or y.dtype != np.float32 or not np.array_equal(x, normalizer.normalize_features(all_arrays["features"])) or not np.array_equal(y, normalizer.normalize_delta(all_arrays["deltas"])):
            raise DataContractError("Stored normalized data differs from shared transform")
        return prepared, normalizer
    except (KeyError, TypeError, OSError, ValueError, BadZipFile, EOFError) as error:
        if isinstance(error, DataContractError):
            raise
        raise DataContractError(f"Invalid learning-data artifact: {error}") from error
