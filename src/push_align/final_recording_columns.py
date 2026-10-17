"""Lossless columnar P5 raw codec; original v1 codec remains immutable.

Every diagnostic and snapshot JSON field is retained. Numeric scalar vectors
are grouped by structural path and fixed-vector component; ragged lists retain
exact offsets and heterogeneous lists retain positions and types. XZ delta8 is
reversible byte filtering, never rounding or numerical prediction.
"""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import lzma
import os
from pathlib import Path
import struct
import numpy as np
from .data import _encode_nonfinite
from .environment import Observation, SnapshotBundle, StepResult
from .final_recording import (RecordingError, _require, _raw_payload, _receipt,
    _verify_expected, _float_array, _restore_diagnostics, _snapshot_arrays, file_sha256)
SCHEMA = "p5-lossless-raw-columnar-xz-v1"
CONTAINER_SCHEMA = "p5-columnar-json-v1"
MAGIC = b"P5COLRAW1\n"
MAX_HEADER_BYTES = 16 * 1024**2
MAX_ARRAY_BYTES = 2 * 1024**3


def raw_paths(prefix):
    prefix = Path(prefix)
    return Path(str(prefix) + ".raw.npz"), Path(str(prefix) + ".raw.columns.xz")
def tree_digest(value):
    """Typed ordered tree digest, including all float64 bits and signed zero."""
    h = hashlib.sha256()

    def visit(v):
        if v is None:
            h.update(b'n')
        elif type(v) is bool:
            h.update(b'b1' if v else b'b0')
        elif type(v) is int:
            b = str(v).encode()
            h.update(b'i' + struct.pack('<Q', len(b)) + b)
        elif type(v) is float:
            h.update(b'f' + struct.pack('<d', v))
        elif type(v) is str:
            b = v.encode('utf-8')
            h.update(b's' + struct.pack('<Q', len(b)) + b)
        elif type(v) is list:
            h.update(b'l' + struct.pack('<Q', len(v)))
            for item in v:
                visit(item)
        elif type(v) is dict:
            h.update(b'd' + struct.pack('<Q', len(v)))
            for key, item in v.items():
                visit(key)
                visit(item)
        else:
            raise TypeError(type(v))
    visit(value)
    return h.hexdigest()

def pack_column(values, arrays):
    n = len(values)
    if not n:
        return {'t': 'empty', 'n': 0}
    kinds = {type(v) for v in values}

    def array(value):
        a = np.ascontiguousarray(value)
        index = len(arrays)
        arrays.append(a)
        return index
    if len(kinds) != 1:
        groups = []
        for typ in (type(None), bool, int, float, str, list, dict):
            positions = [i for i, v in enumerate(values) if type(v) is typ]
            if positions:
                groups.append({'p': array(np.asarray(positions, np.int64)), 'v': pack_column([values[i] for i in positions], arrays)})
        _require(sum((len(arrays[g['p']]) for g in groups)) == n, 'Columnar tree/count inconsistency')
        return {'t': 'mixed', 'n': n, 'groups': groups}
    kind = next(iter(kinds))
    if kind is type(None):
        return {'t': 'none', 'n': n}
    if kind is float:
        return {'t': 'float', 'n': n, 'a': array(np.asarray(values, np.float64))}
    if kind is bool:
        return {'t': 'bool', 'n': n, 'a': array(np.asarray(values, bool))}
    if kind is int:
        if min(values) >= -2 ** 63 and max(values) < 2 ** 63:
            return {'t': 'int', 'n': n, 'a': array(np.asarray(values, np.int64))}
        return {'t': 'bigint', 'n': n, 'values': values}
    if kind is str:
        unique = list(dict.fromkeys(values))
        codes = {v: i for i, v in enumerate(unique)}
        return {'t': 'str', 'n': n, 'unique': unique, 'a': array(np.asarray([codes[v] for v in values], np.int64))}
    if kind is list:

        def numeric_tree(item):
            return type(item) in (int, float, bool) or (type(item) is list and len(item) <= 64 and all((numeric_tree(x) for x in item)))
        if values[0] and (len(values[0]) <= 64) and all((len(v) == len(values[0]) and numeric_tree(v) for v in values)):
            return {'t': 'fixed_list', 'n': n, 'length': len(values[0]), 'v': [pack_column([v[j] for v in values], arrays) for j in range(len(values[0]))]}
        offsets = np.concatenate(([0], np.cumsum([len(v) for v in values], dtype=np.int64)))
        flat = [x for v in values for x in v]
        return {'t': 'list', 'n': n, 'offset': array(offsets), 'v': pack_column(flat, arrays)}
    if kind is dict:
        _require(all(type(k) is str for v in values for k in v), "JSON dictionary keys must be strings")
        keysets = list(dict.fromkeys((tuple(v.keys()) for v in values)))
        if len(keysets) > 1:
            groups = []
            for keys in keysets:
                positions = [i for i, v in enumerate(values) if tuple(v.keys()) == keys]
                groups.append({'p': array(np.asarray(positions, np.int64)), 'v': pack_column([values[i] for i in positions], arrays)})
            return {'t': 'mixed', 'n': n, 'groups': groups}
        keys = keysets[0]
        return {'t': 'dict', 'n': n, 'keys': keys, 'v': [pack_column([v[key] for v in values], arrays) for key in keys]}
    raise TypeError(kind)

def unpack_column(recipe, arrays):
    _require(type(recipe) is dict, "Column recipe must be a dictionary")
    typ, n = (recipe['t'], recipe['n'])
    _require(type(typ) is str and type(n) is int and 0 <= n <= 10000000,
             "Invalid or oversized column recipe count")
    def column(index, dtype, count):
        _require(type(index) is int and 0 <= index < len(arrays), "Invalid column reference")
        value = arrays[index]
        _require(value.dtype == np.dtype(dtype) and value.shape == (count,),
                 "Column recipe dtype/count mismatch")
        return value
    if typ == 'empty':
        _require(n == 0, "Empty recipe must have zero values")
        return []
    if typ == 'none':
        return [None] * n
    if typ in ('float', 'int', 'bool'):
        cast = {'float': float, 'int': int, 'bool': bool}[typ]
        dtype = {'float': np.float64, 'int': np.int64, 'bool': bool}[typ]
        return [cast(v) for v in column(recipe['a'], dtype, n)]
    if typ == 'bigint':
        _require(type(recipe['values']) is list and len(recipe['values']) == n
                 and all(type(v) is int for v in recipe['values']), "Invalid bigint column")
        return recipe['values']
    if typ == 'str':
        unique = recipe['unique']
        _require(type(unique) is list and all(type(v) is str for v in unique)
                 and len(set(unique)) == len(unique), "Invalid string dictionary")
        codes = column(recipe['a'], np.int64, n)
        _require(((codes >= 0) & (codes < len(unique))).all(), "Invalid string code")
        return [unique[int(i)] for i in codes]
    if typ == 'list':
        offsets = column(recipe['offset'], np.int64, n + 1)
        flat = unpack_column(recipe['v'], arrays)
        _require(len(offsets) == n + 1 and offsets[0] == 0 and (offsets[-1] == len(flat)), 'Columnar tree/count inconsistency')
        _require(np.all(offsets[1:] >= offsets[:-1]), 'Columnar tree/count inconsistency')
        return [flat[int(a):int(b)] for a, b in zip(offsets[:-1], offsets[1:])]
    if typ == 'fixed_list':
        columns = [unpack_column(v, arrays) for v in recipe['v']]
        _require(len(columns) == recipe['length'] and all((len(v) == n for v in columns)), 'Columnar tree/count inconsistency')
        return [[column[i] for column in columns] for i in range(n)]
    if typ == 'dict':
        _require(type(recipe['keys']) is list and all(type(k) is str for k in recipe['keys'])
                 and len(set(recipe['keys'])) == len(recipe['keys'])
                 and len(recipe['v']) == len(recipe['keys']), "Invalid dictionary recipe keys")
        columns = [unpack_column(v, arrays) for v in recipe['v']]
        _require(all((len(v) == n for v in columns)), 'Columnar tree/count inconsistency')
        return [{key: column[i] for key, column in zip(recipe['keys'], columns)} for i in range(n)]
    if typ == 'mixed':
        result, seen = ([None] * n, set())
        for group in recipe['groups']:
            items = unpack_column(group['v'], arrays)
            positions = column(group['p'], np.int64, len(items))
            _require(len(positions) == len(items), 'Columnar tree/count inconsistency')
            for pos, item in zip(positions, items):
                pos = int(pos)
                _require(0 <= pos < n and pos not in seen, 'Columnar tree/count inconsistency')
                seen.add(pos)
                result[pos] = item
        _require(len(seen) == n, 'Columnar tree/count inconsistency')
        return result
    raise ValueError(typ)

def _write_columns(target, value, preset=6):
    arrays = []
    recipe = pack_column([value], arrays)
    entries = [{"dtype": a.dtype.str, "shape": list(a.shape), "bytes": a.nbytes,
        "sha256": hashlib.sha256(a.tobytes()).hexdigest()} for a in arrays]
    header = json.dumps({"schema": CONTAINER_SCHEMA, "recipe": recipe, "arrays": entries},
        ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    _require(len(header) <= MAX_HEADER_BYTES, "Columnar metadata header exceeds bounded reader")
    _require(sum(a.nbytes for a in arrays) <= MAX_ARRAY_BYTES, "Columnar arrays exceed bounded reader")
    filters = [{"id": lzma.FILTER_DELTA, "dist": 8}, {"id": lzma.FILTER_LZMA2, "preset": preset}]
    with lzma.LZMAFile(target, "wb", format=lzma.FORMAT_XZ, filters=filters) as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<Q", len(header)))
        stream.write(header)
        for a in arrays:
            stream.write(a.tobytes())
    return {"column_count": len(arrays), "header_bytes": len(header),
            "uncompressed_array_bytes": sum(a.nbytes for a in arrays)}


def _read_exact(stream, count):
    data = stream.read(count)
    _require(len(data) == count, "Truncated columnar stream")
    return data


def _read_columns(path):
    with lzma.open(path, "rb") as stream:
        _require(_read_exact(stream, len(MAGIC)) == MAGIC, "Unsupported columnar magic")
        header_bytes = struct.unpack("<Q", _read_exact(stream, 8))[0]
        _require(header_bytes <= MAX_HEADER_BYTES, "Oversized columnar header")
        header = json.loads(_read_exact(stream, header_bytes))
        _require(type(header) is dict and set(header) == {"schema", "recipe", "arrays"}
                 and header["schema"] == CONTAINER_SCHEMA, "Unsupported columnar schema")
        entries = header["arrays"]
        _require(type(entries) is list and len(entries) <= 100000, "Invalid column array manifest")
        arrays, total_bytes = [], 0
        for entry in entries:
            _require(type(entry) is dict and set(entry) == {"dtype", "shape", "bytes", "sha256"},
                     "Invalid column manifest entry")
            _require(type(entry["dtype"]) is str and entry["dtype"] in ("<f8", ">f8", "<i8", ">i8", "|b1"),
                     "Unsupported column dtype; object arrays forbidden")
            _require(type(entry["shape"]) is list and len(entry["shape"]) == 1
                     and type(entry["shape"][0]) is int and entry["shape"][0] >= 0,
                     "Invalid column vector shape")
            dtype = np.dtype(entry["dtype"])
            _require(type(entry["bytes"]) is int and entry["bytes"] == entry["shape"][0] * dtype.itemsize,
                     "Column byte/shape mismatch")
            total_bytes += entry["bytes"]
            _require(total_bytes <= MAX_ARRAY_BYTES, "Columnar arrays exceed bounded reader")
            raw = _read_exact(stream, entry["bytes"])
            _require(type(entry["sha256"]) is str and hashlib.sha256(raw).hexdigest() == entry["sha256"],
                     "Column array checksum mismatch")
            arrays.append(np.frombuffer(raw, dtype=dtype).copy())
        _require(not stream.read(1), "Unexpected trailing decoded columnar bytes")
    result = unpack_column(header["recipe"], arrays)
    _require(len(result) == 1, "Columnar root count must be one")
    return result[0]


def write_raw(prefix, initial_snapshot, results, *, xz_preset=6):
    """Exclusively write an NPZ/columnar pair; preserve partial output on error."""
    _require(type(xz_preset) is int and xz_preset in (3, 6), "XZ preset must be 3 or 6")
    paths = raw_paths(prefix)
    if any(path.exists() for path in paths):
        raise FileExistsError("Raw recording prefix already exists")
    arrays, metadata = _raw_payload(initial_snapshot, results)
    metadata["schema"] = SCHEMA
    metadata["scope"] = "lossless complete raw storage; no task-success or batch-readiness claim"
    with paths[0].open("xb") as target:
        np.savez_compressed(target, **arrays)
        target.flush()
        os.fsync(target.fileno())
    metadata.update(npz_filename=paths[0].name, npz_sha256=file_sha256(paths[0]), xz_preset=xz_preset)
    with paths[1].open("xb") as target:
        details = _write_columns(target, metadata, preset=xz_preset)
        target.flush()
        os.fsync(target.fileno())
    return _receipt(paths, schema=SCHEMA, interval_count=len(results), xz_preset=xz_preset,
                    columnar=details)


def read_raw(prefix, *, expected_sha256=None):
    """Authenticate pair and reconstruct every StepResult, including float bits."""
    paths = raw_paths(prefix)
    _verify_expected(paths, expected_sha256)
    try:
        metadata = _read_columns(paths[1])
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
    except (KeyError, TypeError, IndexError, ValueError, OverflowError, EOFError, UnicodeDecodeError, lzma.LZMAError) as error:
        raise RecordingError(f"Incomplete or malformed raw pair: {error}") from error
