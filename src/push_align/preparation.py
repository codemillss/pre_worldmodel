"""Streaming development-data preparation without joining excluded time gaps."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .coverage import transition_regimes
from .data import (
    DataContractError, _diagnostics, _read_json, _validate_contract,
    contract_hash, digest, iter_dataset,
)
from .representation import encode, state_delta

SPLIT_CODES = {"train": 0, "validation": 1, "smoke-test": 2}
REGIME_CODES = {"noncontact": 0, "sustained_contact": 1, "transition": 2}
HORIZONS = (1, 5, 10)
YAW_FORMULA_ATOL_RAD = 1e-10


@dataclass(frozen=True)
class PreparedData:
    arrays: dict[str, np.ndarray]
    metadata: dict
    windows: dict[int, np.ndarray]


def _require(condition, message):
    if not condition:
        raise DataContractError(message)


def _yaw_interval(episode, index, diagnostics):
    """Check winding using substeps, with a sampled angular-speed alias guard.

    Endpoint observations alone cannot reveal complete revolutions. Recorded
    microsteps resolve their sampled winding. The speed guard uses observed
    endpoint speeds, not a proven continuous-time speed bound; unobserved
    intra-microstep peaks remain a limitation, recorded in output metadata.
    """
    arrays = episode.arrays
    start, stop = arrays["substep_offsets"][index:index+2]
    substeps = arrays["substep_states"][start:stop]
    times = np.concatenate(([arrays["sim_time"][index]], arrays["substep_times"][start:stop]))
    yaws = np.concatenate(([arrays["state"][index, 2]], substeps[:, 2]))
    _require(len(substeps) > 0 and np.isfinite(yaws).all(), "Valid yaw interval requires finite microsteps")
    increments = (np.diff(yaws)+math.pi) % (2*math.pi)-math.pi
    _require(np.all(np.abs(increments) < math.pi), "Ambiguous pi microstep yaw increment")
    cumulative = np.cumsum(increments)
    _require(np.all(np.abs(cumulative) < math.pi), "Recorded yaw excursion reaches pi within action interval")
    endpoint_delta = float(state_delta(arrays["state"][index], arrays["state"][index+1])[2])
    _require(abs(float(cumulative[-1])-endpoint_delta) <= YAW_FORMULA_ATOL_RAD,
             "Microstep yaw winding disagrees with principal endpoint delta")
    samples = diagnostics[index]["samples"]
    _require(len(samples) == len(times), "Yaw diagnostic/sample count mismatch")
    velocities = np.asarray([s["object_velocity_world"][:3] for s in samples], dtype=np.float64)
    tilts = np.asarray([s["tilt_rad"] for s in samples], dtype=np.float64)
    _require(np.isfinite(velocities).all() and np.isfinite(tilts).all(), "Yaw speed guard requires finite physics samples")
    _require(np.all(np.diff(times) > 0) and np.all(np.abs(tilts) < math.pi/2), "Yaw speed guard requires positive dt and planar tilt")
    # Azimuth rate is bounded by |world omega| / cos(tilt) at each sample.
    observed_rates = np.linalg.norm(velocities, axis=1)/np.cos(tilts)
    estimated_step_rotations = np.maximum(observed_rates[:-1], observed_rates[1:])*np.diff(times)
    _require(np.all(estimated_step_rotations < math.pi), "Observed angular speed makes microstep yaw aliasing uncertain")


def prepare_dataset(path, expected_contract=None):
    """Exhaust the validated stream, then return compact valid-transition data.

    No partially prepared object is returned when final manifest-total or
    family-split checks fail. Raw float64 states/actions are copied unchanged.
    Windows contain compact table indices but originate from contiguous valid
    original transitions within one episode, including no excluded gaps.
    """
    path = Path(path)
    manifest_path = path/"manifest.json"
    manifest_digest = digest(manifest_path)
    manifest = _read_json(manifest_path)
    contract = manifest["contract"] if expected_contract is None else expected_contract
    _validate_contract(contract)
    _require(contract_hash(contract) == manifest["contract_hash"], "Preparation expected contract/hash mismatch")
    chunks = {name: [] for name in (
        "current_state", "next_state", "applied_action", "features", "deltas",
        "split_code", "episode_index", "transition_index", "regime_code",
    )}
    window_chunks = {horizon: [] for horizon in HORIZONS}
    identities, table_offset = [], 0
    split_counts = {split: {"episodes": 0, "recorded_transitions": 0, "valid_transitions": 0,
                            "excluded_transitions": 0} for split in SPLIT_CODES}
    for episode in iter_dataset(path, contract):
        arrays, identity = episode.arrays, episode.metadata
        split = identity["split"]
        _require(split in SPLIT_CODES, "Preparation accepts development splits only")
        valid = arrays["transition_valid"]
        indices = np.flatnonzero(valid).astype(np.int64)
        diagnostics = _diagnostics(episode)
        for index in indices:
            _yaw_interval(episode, int(index), diagnostics)
        labels = transition_regimes(episode)
        current = arrays["state"][indices].copy()
        future = arrays["state"][indices+1].copy()
        applied = arrays["action_applied"][indices].copy()
        count = len(indices)
        values = {
            "current_state": current, "next_state": future, "applied_action": applied,
            "features": encode(current, applied), "deltas": state_delta(current, future),
            "split_code": np.full(count, SPLIT_CODES[split], dtype=np.int8),
            "episode_index": np.full(count, identity["episode_index"], dtype=np.int64),
            "transition_index": indices,
            "regime_code": np.asarray([REGIME_CODES[labels[index]] for index in indices], dtype=np.int8),
        }
        for name, value in values.items():
            chunks[name].append(value)
        original_to_table = np.full(len(valid), -1, dtype=np.int64)
        original_to_table[indices] = table_offset+np.arange(count, dtype=np.int64)
        for horizon in HORIZONS:
            for start in range(len(valid)-horizon+1):
                if valid[start:start+horizon].all():
                    window_chunks[horizon].append(original_to_table[start:start+horizon].copy())
        identities.append({
            **{key: deepcopy(identity[key]) for key in ("episode_id", "episode_index", "family_id", "split")},
            "table_start": table_offset, "table_stop": table_offset+count,
            "recorded_transition_count": len(valid), "valid_transition_count": count,
            "excluded_transition_count": len(valid)-count,
        })
        for key, amount in (("episodes", 1), ("recorded_transitions", len(valid)),
                            ("valid_transitions", count), ("excluded_transitions", len(valid)-count)):
            split_counts[split][key] += amount
        table_offset += count
    # iter_dataset performs global checks only after its last yield.
    _require(bool(identities), "Preparation requires at least one episode")
    _require(digest(manifest_path) == manifest_digest, "Manifest changed during preparation")
    result_arrays = {name: np.concatenate(values, axis=0) for name, values in chunks.items()}
    windows = {horizon: np.asarray(rows, dtype=np.int64).reshape(-1, horizon)
               for horizon, rows in window_chunks.items()}
    metadata = {
        "schema": "push-align-prepared-v1", "dataset_id": manifest["dataset_id"],
        "manifest_sha256": manifest_digest, "data_contract_hash": contract_hash(contract),
        "contract": deepcopy(contract), "episode_count": len(identities),
        "valid_transition_count": table_offset,
        "recorded_transition_count": sum(item["recorded_transition_count"] for item in identities),
        "excluded_transition_count": sum(item["excluded_transition_count"] for item in identities),
        "episodes": identities, "split_counts": split_counts,
        "split_codes": dict(SPLIT_CODES), "regime_codes": dict(REGIME_CODES),
        "window_counts": {str(horizon): len(rows) for horizon, rows in windows.items()},
        "yaw_check": {
            "endpoint_formula_atol_rad": YAW_FORMULA_ATOL_RAD,
            "cumulative_microstep_excursion_limit_rad_exclusive": math.pi,
            "alias_guard": "maximum adjacent observed world-angular-speed norms / cos(tilt) times microstep dt < pi",
            "limitation": "sampled speed is not a certified continuous-time bound; unobserved intra-microstep peaks cannot be ruled out",
        },
        "scope": "validated development-data representation; no model training or alignment assessment",
    }
    return PreparedData(result_arrays, metadata, windows)
