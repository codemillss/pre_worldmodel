"""Lossless recording candidates for P5; not yet connected to a batch runner.

The existing raw writer and all original files remain unchanged.  This codec
adds explicit flags, observation times/schemas and ragged substep offsets to
the same raw arrays.  Diagnostics remain complete JSON, compressed with XZ.
Checksums bind the two files; callers retain the returned receipt externally.
No model or physical environment is instantiated by this module.
"""
from __future__ import annotations

from dataclasses import asdict
import gzip
import hashlib
import io
import json
import lzma
import os
from pathlib import Path
import zipfile

import numpy as np

from .data import _encode_nonfinite
from .environment import Observation, SnapshotBundle, StepResult


SCHEMA = "p5-lossless-raw-xz-v1"
PLAN_INDEX_SCHEMA = "p5-plan-per-index-archive-v1"


class RecordingError(ValueError):
    """A saved record is malformed, incomplete or inconsistent."""


def _require(condition, message):
    if not condition:
        raise RecordingError(message)


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def raw_paths(prefix):
    prefix = Path(prefix)
    return Path(str(prefix) + ".raw.npz"), Path(str(prefix) + ".raw.json.xz")


def _float_array(value, shape, name):
    _require(isinstance(value, np.ndarray) and value.dtype == np.float64
             and value.shape == shape, f"{name}: float64 array shape {shape} required")
    return value


def _flag(value, name):
    _require(type(value) in (bool, np.bool_), f"{name}: boolean required")
    return bool(value)


def _scalar(value, name):
    _require(type(value) in (float, np.float64), f"{name}: float64 scalar required")
    return float(value)


def _snapshot_arrays(snapshot):
    _require(isinstance(snapshot, SnapshotBundle), "SnapshotBundle required")
    _require(isinstance(snapshot.integration_state, np.ndarray)
             and snapshot.integration_state.dtype == np.float64
             and snapshot.integration_state.ndim == 1, "snapshot integration_state dtype/shape")
    for key in ("model_hash", "environment_hash"):
        _require(type(getattr(snapshot, key)) is str and bool(getattr(snapshot, key)),
                 f"snapshot {key}: nonempty string required")
    _require(type(snapshot.rng_state) is dict, "snapshot rng_state: mapping required")
    for key in ("warning_counts", "warning_lastinfo"):
        value = getattr(snapshot, key)
        _require(isinstance(value, np.ndarray) and value.dtype.kind in "iu"
                 and value.ndim == 1, f"snapshot {key}: integer vector required")
    _require(snapshot.warning_counts.shape == snapshot.warning_lastinfo.shape,
             "snapshot warning vector shapes differ")
    return {
        "initial_integration_state": snapshot.integration_state,
        "initial_warning_counts": snapshot.warning_counts,
        "initial_warning_lastinfo": snapshot.warning_lastinfo,
        "initial_reference_height": np.array(_scalar(snapshot.reference_height, "reference_height"), np.float64),
        "initial_needs_reset": np.array(_flag(snapshot.needs_reset, "needs_reset"), bool),
    }


def _raw_payload(snapshot, results):
    arrays = _snapshot_arrays(snapshot)
    _require(type(results) in (list, tuple), "results must be an explicit list or tuple")
    for index, step in enumerate(results):
        _require(isinstance(step, StepResult), f"result {index}: StepResult required")
        _require(isinstance(step.observation, Observation), f"result {index}: Observation required")
        for key in ("requested_action", "applied_action"):
            _float_array(getattr(step, key), (2,), f"result {index} {key}")
        _float_array(step.observation.state, (10,), f"result {index} state")
        _require(isinstance(step.substep_times, np.ndarray) and step.substep_times.ndim == 1,
                 f"result {index}: substep times vector required")
        count = len(step.substep_times)
        _float_array(step.substep_states, (count, 10), f"result {index} substep_states")
        _float_array(step.substep_times, (count,), f"result {index} substep_times")
        _scalar(step.actual_dt, f"result {index} actual_dt")
        _scalar(step.observation.sim_time, f"result {index} sim_time")
        _flag(step.transition_valid, f"result {index} transition_valid")
        _flag(step.observation.observation_valid, f"result {index} observation_valid")
        _require(type(step.observation.schema_version) is str and bool(step.observation.schema_version),
                 f"result {index}: nonempty observation schema required")
        _require(type(step.diagnostics) is dict, f"result {index}: diagnostics mapping required")
    paths, values = [], []
    diagnostics = _encode_nonfinite([step.diagnostics for step in results], paths=paths, values=values)
    arrays.update({
        "state": np.array([step.observation.state for step in results], np.float64).reshape(-1, 10),
        "actual_dt": np.array([step.actual_dt for step in results], np.float64),
        "action_requested": np.array([step.requested_action for step in results], np.float64).reshape(-1, 2),
        "action_applied": np.array([step.applied_action for step in results], np.float64).reshape(-1, 2),
        "substep_states": (np.concatenate([step.substep_states for step in results])
                           if results else np.empty((0, 10), np.float64)),
        "substep_times": (np.concatenate([step.substep_times for step in results])
                          if results else np.empty(0, np.float64)),
        "diagnostic_nonfinite_values": np.array(values, np.float64),
        "observation_sim_time": np.array([step.observation.sim_time for step in results], np.float64),
        "observation_valid": np.array([step.observation.observation_valid for step in results], bool),
        "transition_valid": np.array([step.transition_valid for step in results], bool),
        "substep_offsets": np.concatenate(([0], np.cumsum([len(step.substep_times) for step in results]))).astype(np.int64),
    })
    # Preserve the established raw JSON fields as well as explicit codec data.
    metadata = {
        "schema": SCHEMA, "interval_count": len(results),
        "initial_snapshot": _encode_nonfinite(asdict(snapshot)),
        "diagnostics": diagnostics, "diagnostic_nonfinite_paths": paths,
        "observation_schema_versions": [step.observation.schema_version for step in results],
        "scope": "lossless raw storage candidate; no final batch integration or task-success claim",
    }
    return arrays, metadata


def _receipt(paths, **extra):
    return {**extra, "files_sha256": {str(path): file_sha256(path) for path in paths},
            "bytes": {str(path): path.stat().st_size for path in paths}}


def write_raw(prefix, initial_snapshot, results, *, xz_preset=6):
    """Create two new files exclusively, retaining any partial file on failure.

    A compression/write exception is propagated.  No created file is removed or
    overwritten: the caller marks the attempt incomplete and keeps its prefix.
    """
    _require(type(xz_preset) is int and xz_preset in (3, 6), "XZ preset must be 3 or 6")
    paths = raw_paths(prefix)
    if any(path.exists() for path in paths):
        raise FileExistsError("Raw recording prefix already exists")
    arrays, metadata = _raw_payload(initial_snapshot, results)
    with paths[0].open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    metadata.update({"npz_filename": paths[0].name, "npz_sha256": file_sha256(paths[0]),
                     "xz_preset": xz_preset})
    with paths[1].open("xb") as target:
        with lzma.open(target, "wt", encoding="utf-8", preset=xz_preset) as stream:
            json.dump(metadata, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        target.flush()
        os.fsync(target.fileno())
    return _receipt(paths, schema=SCHEMA, interval_count=len(results), xz_preset=xz_preset)


def _verify_expected(paths, expected):
    if expected is None:
        return
    _require(type(expected) is dict and set(expected) == {str(path) for path in paths},
             "Expected SHA mapping must contain exactly the recording files")
    for path in paths:
        _require(file_sha256(path) == expected[str(path)], f"SHA mismatch: {path.name}")


def _restore_diagnostics(metadata, values):
    diagnostics, paths = metadata["diagnostics"], metadata["diagnostic_nonfinite_paths"]
    _require(type(diagnostics) is list and type(paths) is list
             and len(paths) == len(values) and not np.isfinite(values).any(),
             "Diagnostic nonfinite mapping mismatch")
    _require(len({json.dumps(path) for path in paths}) == len(paths), "Duplicate nonfinite path")
    for path, value in zip(paths, values):
        _require(type(path) is list and len(path) >= 2, "Invalid diagnostic nonfinite path")
        target = diagnostics
        try:
            for part in path[:-1]:
                target = target[part]
            _require(target[path[-1]] is None, "Nonfinite placeholder mismatch")
            target[path[-1]] = float(value)
        except (KeyError, IndexError, TypeError) as error:
            raise RecordingError("Invalid diagnostic nonfinite path") from error
    return diagnostics


def read_raw(prefix, *, expected_sha256=None):
    """Verify pair integrity and reconstruct snapshot and every StepResult field.

    The internally stored SHA always checks numeric NPZ integrity.  Supply the
    external receipt to also authenticate JSON bytes against a prior record.
    This is structural storage validation, not a physics or task-validity test.
    """
    paths = raw_paths(prefix)
    _verify_expected(paths, expected_sha256)
    try:
        with lzma.open(paths[1], "rt", encoding="utf-8") as stream:
            metadata = json.load(stream)
        _require(type(metadata) is dict and metadata.get("schema") == SCHEMA, "Unsupported raw schema")
        _require(metadata.get("npz_filename") == paths[0].name
                 and file_sha256(paths[0]) == metadata.get("npz_sha256"), "NPZ checksum/name mismatch")
        _require(type(metadata.get("interval_count")) is int and metadata["interval_count"] >= 0,
                 "Invalid interval count")
        with np.load(paths[0], allow_pickle=False) as source:
            arrays = {name: source[name].copy() for name in source.files}
        n = metadata["interval_count"]
        shapes = {"state": (n, 10), "actual_dt": (n,), "action_requested": (n, 2),
                  "action_applied": (n, 2), "observation_sim_time": (n,)}
        for name, shape in shapes.items():
            _float_array(arrays[name], shape, name)
        for name in ("observation_valid", "transition_valid"):
            _require(arrays[name].dtype == bool and arrays[name].shape == (n,), f"{name}: bool vector required")
        offsets = arrays["substep_offsets"]
        _require(offsets.dtype == np.int64 and offsets.shape == (n + 1,) and offsets[0] == 0
                 and (np.diff(offsets) >= 0).all(), "Invalid substep offsets")
        count = int(offsets[-1])
        _float_array(arrays["substep_states"], (count, 10), "substep_states")
        _float_array(arrays["substep_times"], (count,), "substep_times")
        values = arrays["diagnostic_nonfinite_values"]
        _require(values.dtype == np.float64 and values.ndim == 1, "Invalid nonfinite numeric values")
        diagnostics = _restore_diagnostics(metadata, values)
        schemas = metadata["observation_schema_versions"]
        _require(len(diagnostics) == n and all(type(row) is dict for row in diagnostics)
                 and type(schemas) is list and len(schemas) == n
                 and all(type(value) is str and bool(value) for value in schemas),
                 "Diagnostic/schema count or type mismatch")
        saved = metadata["initial_snapshot"]
        _require(type(saved) is dict and set(saved) == set(SnapshotBundle.__dataclass_fields__),
                 "Snapshot fields missing or extra")
        _require(arrays["initial_reference_height"].shape == ()
                 and arrays["initial_reference_height"].dtype == np.float64
                 and arrays["initial_needs_reset"].shape == ()
                 and arrays["initial_needs_reset"].dtype == bool, "Snapshot scalar dtype/shape mismatch")
        snapshot = SnapshotBundle(
            arrays["initial_integration_state"], saved["model_hash"], saved["environment_hash"],
            float(arrays["initial_reference_height"]), saved["rng_state"],
            arrays["initial_warning_counts"], arrays["initial_warning_lastinfo"],
            bool(arrays["initial_needs_reset"]))
        _snapshot_arrays(snapshot)
        # Duplicated JSON snapshot values bind to the numeric representation,
        # including null placeholders for nonfinite values and signed zero.
        _require(json.dumps(_encode_nonfinite(asdict(snapshot)), sort_keys=True, allow_nan=False)
                 == json.dumps(saved, sort_keys=True, allow_nan=False), "Snapshot JSON/NPZ mismatch")
        results = [StepResult(
            arrays["action_requested"][i].copy(), arrays["action_applied"][i].copy(),
            Observation(arrays["state"][i].copy(), float(arrays["observation_sim_time"][i]),
                        schemas[i], bool(arrays["observation_valid"][i])),
            float(arrays["actual_dt"][i]), bool(arrays["transition_valid"][i]), diagnostics[i],
            arrays["substep_states"][offsets[i]:offsets[i + 1]].copy(),
            arrays["substep_times"][offsets[i]:offsets[i + 1]].copy()) for i in range(n)]
        return snapshot, results
    except (KeyError, TypeError, IndexError, json.JSONDecodeError, lzma.LZMAError) as error:
        raise RecordingError(f"Incomplete or malformed raw pair: {error}") from error


def write_json_gzip(path, value):
    """Write JSON-compatible witness data exclusively with no dropped fields."""
    # Validate first, so bad types cannot leave a misleading complete record.
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    path = Path(path)
    with path.open("xb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as stream:
            stream.write(payload)
        target.flush()
        os.fsync(target.fileno())
    return _receipt((path,), schema="json-gzip-lossless-v1")


def write_witness_gzip(path, witness):
    """Write one logical witness JSON using independently priced gzip members.

    Every interval and decision occupies a separate gzip member, with its JSON
    comma included.  Concatenated decompression is still a single complete JSON
    object read by ``read_json_gzip``.  Returned member lengths support additive
    storage estimates without claiming a measured native300-step execution.
    """
    _require(type(witness) is dict and type(witness.get("intervals")) is list
             and type(witness.get("decisions")) is list,
             "Witness requires intervals and decisions lists")
    fixed = {key: value for key, value in witness.items() if key not in {"intervals", "decisions"}}
    header = json.dumps(fixed, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    header = header[:-1] + (b"," if fixed else b"") + b'"intervals":['
    path = Path(path)
    interval_bytes, decision_bytes = [], []
    with path.open("xb") as target:
        def member(payload):
            before = target.tell()
            with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as stream:
                stream.write(payload)
            return target.tell() - before

        fixed_bytes = member(header)
        for index, row in enumerate(witness["intervals"]):
            payload = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
            interval_bytes.append(member((b"," if index else b"") + payload))
        fixed_bytes += member(b'],"decisions":[')
        for index, row in enumerate(witness["decisions"]):
            payload = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
            decision_bytes.append(member((b"," if index else b"") + payload))
        fixed_bytes += member(b"]}")
        target.flush()
        os.fsync(target.fileno())
    _require(path.stat().st_size == fixed_bytes + sum(interval_bytes) + sum(decision_bytes),
             "Gzip member byte accounting mismatch")
    return _receipt((path,), schema="witness-json-gzip-members-v1", fixed_bytes=fixed_bytes,
                    interval_member_bytes=interval_bytes, decision_member_bytes=decision_bytes,
                    logical_witness_schema=witness.get("schema"),
                    scope="additive codec storage; not native long-run execution evidence")


def read_json_gzip(path, *, expected_sha256=None):
    """Read complete JSON, optionally checking its previously retained SHA."""
    path = Path(path)
    if expected_sha256 is not None:
        _require(file_sha256(path) == expected_sha256, f"SHA mismatch: {path.name}")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _plan_arrays(record):
    _require(type(record) is dict and set(record) == {"tick", "arrays"}, "Plan record fields differ")
    tick, arrays = record["tick"], record["arrays"]
    _require(type(tick) is int and tick >= 0, "Plan tick must be a nonnegative integer")
    _require(type(arrays) is dict, "Plan arrays mapping required")
    required = {"valid", "reasons", "failure_steps", "total_cost"}
    _require(required <= set(arrays) and all(key in required or (
        type(key) is str and key.startswith("cost_") and len(key) > 5) for key in arrays),
        "Plan index requires valid/reasons/failure_steps/total_cost and only cost_* additions")
    _require(all(isinstance(array, np.ndarray) and not array.dtype.hasobject
                 and array.ndim == 1 for array in arrays.values()), "Plan arrays must be nonobject vectors")
    n = len(arrays["valid"])
    _require(n > 0 and all(array.shape == (n,) for array in arrays.values()), "Plan candidate shapes differ")
    _require(arrays["valid"].dtype == bool, "Plan valid must be boolean")
    _require(arrays["reasons"].dtype.kind in "US", "Plan reasons must be fixed-width strings")
    _require(arrays["failure_steps"].dtype.kind in "iu", "Plan failure_steps must be integers")
    for key, array in arrays.items():
        if key == "total_cost" or key.startswith("cost_"):
            _require(array.dtype == np.float64, f"Plan {key} must be float64")
    return tick, arrays


def write_plan_index(path, records):
    """Preserve every per-candidate mask/reason/failure/cost in one archive.

    Each tick is a compressed numeric NPZ in a ZIP_STORED container.  Full
    predicted states, raw deltas and action candidates are separate, selected
    trace evidence.  No value is rounded or filtered here, including invalid
    rows and nonfinite costs.  Failed attempts keep their partial archive.
    """
    _require(type(records) is list, "Plan records must be a list")
    checked = [_plan_arrays(record) for record in records]
    ticks = [tick for tick, _ in checked]
    _require(len(set(ticks)) == len(ticks) and ticks == sorted(ticks), "Plan ticks must be unique and sorted")
    path = Path(path)
    entries = []
    with path.open("xb") as target:
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
            for tick, arrays in checked:
                buffer = io.BytesIO()
                np.savez_compressed(buffer, **arrays)
                payload = buffer.getvalue()
                member = f"tick-{tick:06d}.npz"
                archive.writestr(member, payload)
                entries.append({"tick": tick, "member": member,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "arrays": {key: {"dtype": array.dtype.str, "shape": list(array.shape)}
                               for key, array in arrays.items()}})
            manifest = {"schema": PLAN_INDEX_SCHEMA, "record_count": len(entries), "records": entries,
                        "scope": "every-plan per-index values; selected full traces remain separate"}
            archive.writestr("index.json", json.dumps(manifest, ensure_ascii=False,
                allow_nan=False, separators=(",", ":")).encode("utf-8"))
        target.flush()
        os.fsync(target.fileno())
    return _receipt((path,), schema=PLAN_INDEX_SCHEMA, record_count=len(entries), ticks=ticks)


def read_plan_index(path, *, expected_sha256=None):
    """Authenticate and restore every tick's arrays with exact dtype/bytes."""
    path = Path(path)
    if expected_sha256 is not None:
        _require(file_sha256(path) == expected_sha256, f"SHA mismatch: {path.name}")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            _require(len(names) == len(set(names)) and "index.json" in names,
                     "Duplicate archive members or missing manifest")
            manifest = json.loads(archive.read("index.json"))
            _require(type(manifest) is dict and manifest.get("schema") == PLAN_INDEX_SCHEMA,
                     "Unsupported plan index schema")
            entries = manifest["records"]
            _require(type(entries) is list and type(manifest.get("record_count")) is int
                     and manifest["record_count"] == len(entries), "Plan index record count mismatch")
            records = []
            expected_names = {"index.json"}
            for entry in entries:
                tick, member = entry["tick"], entry["member"]
                _require(type(tick) is int and tick >= 0 and member == f"tick-{tick:06d}.npz",
                         "Invalid plan member/tick")
                expected_names.add(member)
                payload = archive.read(member)
                _require(hashlib.sha256(payload).hexdigest() == entry["sha256"], "Plan member checksum mismatch")
                with np.load(io.BytesIO(payload), allow_pickle=False) as source:
                    arrays = {key: source[key].copy() for key in source.files}
                record = {"tick": tick, "arrays": arrays}
                _plan_arrays(record)
                specs = entry["arrays"]
                _require(type(specs) is dict and set(specs) == set(arrays), "Plan array manifest fields differ")
                for key, array in arrays.items():
                    _require(specs[key] == {"dtype": array.dtype.str, "shape": list(array.shape)},
                             "Plan array dtype/shape manifest mismatch")
                records.append(record)
            ticks = [record["tick"] for record in records]
            _require(ticks == sorted(ticks) and len(ticks) == len(set(ticks)), "Duplicate/unsorted plan ticks")
            _require(set(names) == expected_names, "Unexpected or missing archive members")
            return records
    except (KeyError, TypeError, IndexError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        if isinstance(error, RecordingError):
            raise
        raise RecordingError(f"Incomplete or malformed plan index: {error}") from error
