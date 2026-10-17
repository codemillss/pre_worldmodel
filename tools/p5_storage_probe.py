"""Storage-only columnar diagnostic prototype; never runs a model or physics.

The production codec and every existing file are read-only inputs.  This probe
preserves the complete JSON tree, dict order, Python scalar types and float64
bits, rather than deciding which diagnostic fields appear useful.  Ragged lists
retain exact offsets.  It is not a production recording schema or batch gate.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import lzma
from pathlib import Path
import resource
import shutil
import struct
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/p5-storage-probe"
REPORT = ROOT / "reports/p5_storage_feasibility.json"
MIB = 1024**2
GIB = 1024**3
MAGIC = b"P5COLUMN1\n"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(MIB), b""):
            h.update(block)
    return h.hexdigest()


def bytes_used():
    return sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) if OUT.exists() else 0


def guard():
    assert bytes_used() <= 64 * MIB, "Keep outputs and stop: own64MiB ceiling"
    assert shutil.disk_usage(ROOT).free >= 2 * GIB, "Keep outputs and stop: reserve2GiB"


def peak_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (MIB if sys.platform == "darwin" else 1024)


def tree_digest(value):
    """Typed ordered tree digest, including all float64 bits and signed zero."""
    h = hashlib.sha256()
    def visit(v):
        if v is None:
            h.update(b"n")
        elif type(v) is bool:
            h.update(b"b1" if v else b"b0")
        elif type(v) is int:
            b = str(v).encode(); h.update(b"i" + struct.pack("<Q", len(b)) + b)
        elif type(v) is float:
            h.update(b"f" + struct.pack("<d", v))
        elif type(v) is str:
            b = v.encode("utf-8"); h.update(b"s" + struct.pack("<Q", len(b)) + b)
        elif type(v) is list:
            h.update(b"l" + struct.pack("<Q", len(v)))
            for item in v: visit(item)
        elif type(v) is dict:
            h.update(b"d" + struct.pack("<Q", len(v)))
            for key, item in v.items(): visit(key); visit(item)
        else:
            raise TypeError(type(v))
    visit(value)
    return h.hexdigest()


def pack_column(values, arrays):
    n = len(values)
    if not n:
        return {"t": "empty", "n": 0}
    kinds = {type(v) for v in values}
    def array(value):
        a = np.ascontiguousarray(value)
        index = len(arrays); arrays.append(a)
        return index
    if len(kinds) != 1:
        groups = []
        for typ in (type(None), bool, int, float, str, list, dict):
            positions = [i for i, v in enumerate(values) if type(v) is typ]
            if positions:
                groups.append({"p": array(np.asarray(positions, np.int64)),
                    "v": pack_column([values[i] for i in positions], arrays)})
        assert sum(len(arrays[g["p"]]) for g in groups) == n
        return {"t": "mixed", "n": n, "groups": groups}
    kind = next(iter(kinds))
    if kind is type(None): return {"t": "none", "n": n}
    if kind is float:
        return {"t": "float", "n": n, "a": array(np.asarray(values, np.float64))}
    if kind is bool:
        return {"t": "bool", "n": n, "a": array(np.asarray(values, bool))}
    if kind is int:
        if min(values) >= -(2**63) and max(values) < 2**63:
            return {"t": "int", "n": n, "a": array(np.asarray(values, np.int64))}
        return {"t": "bigint", "n": n, "values": values}
    if kind is str:
        unique = list(dict.fromkeys(values)); codes = {v: i for i, v in enumerate(unique)}
        return {"t": "str", "n": n, "unique": unique,
                "a": array(np.asarray([codes[v] for v in values], np.int64))}
    if kind is list:
        offsets = np.concatenate(([0], np.cumsum([len(v) for v in values], dtype=np.int64)))
        flat = [x for v in values for x in v]
        return {"t": "list", "n": n, "offset": array(offsets), "v": pack_column(flat, arrays)}
    if kind is dict:
        keysets = list(dict.fromkeys(tuple(v.keys()) for v in values))
        if len(keysets) > 1:
            groups = []
            for keys in keysets:
                positions = [i for i, v in enumerate(values) if tuple(v.keys()) == keys]
                groups.append({"p": array(np.asarray(positions, np.int64)),
                    "v": pack_column([values[i] for i in positions], arrays)})
            return {"t": "mixed", "n": n, "groups": groups}
        keys = keysets[0]
        return {"t": "dict", "n": n, "keys": keys,
            "v": [pack_column([v[key] for v in values], arrays) for key in keys]}
    raise TypeError(kind)


def unpack_column(recipe, arrays):
    typ, n = recipe["t"], recipe["n"]
    if typ == "empty": return []
    if typ == "none": return [None] * n
    if typ in ("float", "int", "bool"):
        cast = {"float": float, "int": int, "bool": bool}[typ]
        return [cast(v) for v in arrays[recipe["a"]]]
    if typ == "bigint": return recipe["values"]
    if typ == "str": return [recipe["unique"][int(i)] for i in arrays[recipe["a"]]]
    if typ == "list":
        offsets = arrays[recipe["offset"]]; flat = unpack_column(recipe["v"], arrays)
        assert len(offsets) == n + 1 and offsets[0] == 0 and offsets[-1] == len(flat)
        assert np.all(offsets[1:] >= offsets[:-1])
        return [flat[int(a):int(b)] for a, b in zip(offsets[:-1], offsets[1:])]
    if typ == "dict":
        columns = [unpack_column(v, arrays) for v in recipe["v"]]
        assert all(len(v) == n for v in columns)
        return [{key: column[i] for key, column in zip(recipe["keys"], columns)} for i in range(n)]
    if typ == "mixed":
        result, seen = [None] * n, set()
        for group in recipe["groups"]:
            positions, items = arrays[group["p"]], unpack_column(group["v"], arrays)
            assert len(positions) == len(items)
            for pos, item in zip(positions, items):
                pos = int(pos); assert 0 <= pos < n and pos not in seen
                seen.add(pos); result[pos] = item
        assert len(seen) == n
        return result
    raise ValueError(typ)


def write_columnar(path, value):
    arrays = []; recipe = pack_column([value], arrays)
    entries = [{"dtype": a.dtype.str, "shape": list(a.shape), "bytes": a.nbytes,
                "sha256": hashlib.sha256(a.tobytes()).hexdigest()} for a in arrays]
    header = json.dumps({"schema": "p5-columnar-json-probe-v1", "recipe": recipe,
                         "arrays": entries}, ensure_ascii=False, allow_nan=False,
                        separators=(",", ":")).encode()
    with Path(path).open("xb") as target:
        filters = [{"id": lzma.FILTER_DELTA, "dist": 8}, {"id": lzma.FILTER_LZMA2, "preset": 6}]
        with lzma.LZMAFile(target, "wb", format=lzma.FORMAT_XZ, filters=filters) as stream:
            stream.write(MAGIC); stream.write(struct.pack("<Q", len(header))); stream.write(header)
            for a in arrays: stream.write(a.tobytes())
    return {"array_columns": len(arrays), "uncompressed_array_bytes": sum(a.nbytes for a in arrays),
            "recipe_header_bytes": len(header), "sha256": sha(path), "bytes": Path(path).stat().st_size}


def read_columnar(path, expected_sha):
    assert sha(path) == expected_sha
    with lzma.open(path, "rb") as stream:
        assert stream.read(len(MAGIC)) == MAGIC
        header = json.loads(stream.read(struct.unpack("<Q", stream.read(8))[0]))
        assert header["schema"] == "p5-columnar-json-probe-v1"
        arrays = []
        for entry in header["arrays"]:
            raw = stream.read(entry["bytes"])
            assert len(raw) == entry["bytes"] and hashlib.sha256(raw).hexdigest() == entry["sha256"]
            a = np.frombuffer(raw, dtype=entry["dtype"]).reshape(entry["shape"]).copy()
            assert a.nbytes == entry["bytes"]; arrays.append(a)
        assert not stream.read(1)
    result = unpack_column(header["recipe"], arrays)
    assert len(result) == 1
    return result[0]


def measure(label, prefix):
    json_path, npz_path = Path(str(prefix) + ".raw.json.xz"), Path(str(prefix) + ".raw.npz")
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in (json_path, npz_path)}
    started = perf_counter()
    with lzma.open(json_path, "rt") as f: original = json.load(f)
    load_s = perf_counter() - started
    original_digest = tree_digest(original)
    with np.load(npz_path, allow_pickle=False) as f:
        npz_arrays = {k: {"dtype": f[k].dtype.str, "shape": list(f[k].shape),
                         "sha256": hashlib.sha256(f[k].tobytes()).hexdigest()} for k in f.files}
    path = OUT / (label + ".columnar.xz")
    guard(); started = perf_counter(); receipt = write_columnar(path, original); write_s = perf_counter() - started
    guard(); started = perf_counter(); restored = read_columnar(path, receipt["sha256"]); read_s = perf_counter() - started
    assert tree_digest(restored) == original_digest
    for key, expected in hashes.items(): assert sha(ROOT / key) == expected
    return {"label": label, "intervals": original["interval_count"], "source_sha256": hashes,
        "source_npz_unchanged_arrays": npz_arrays, "complete_tree_typed_ordered_floatbit_digest": original_digest,
        "all_json_fields_bit_semantics_exact": True, "npz_original_unchanged": True,
        "original_json_xz_bytes": json_path.stat().st_size, "numeric_npz_bytes": npz_path.stat().st_size,
        "old_total_bytes": json_path.stat().st_size + npz_path.stat().st_size,
        "new_total_bytes": receipt["bytes"] + npz_path.stat().st_size,
        "json_fraction": receipt["bytes"] / json_path.stat().st_size,
        "load_wall_s": load_s, "write_wall_s": write_s, "read_wall_s": read_s,
        "peak_process_MiB_so_far": peak_mib(), "receipt": receipt}


def main():
    assert not OUT.exists() and not REPORT.exists(), "Never overwrite existing evidence"
    assert shutil.disk_usage(ROOT).free > 2 * GIB + 64 * MIB
    OUT.mkdir(); before = shutil.disk_usage(ROOT).free
    rows = []
    # Worst observed short-case first, so a bad prototype does not justify a long run.
    rows.append(measure("contact-R2", ROOT / "tmp/p5-g2-codec/policy-measurement/raw-01"))
    rows.append(measure("empty-contact-R", ROOT / "tmp/p5-g2-codec/policy-measurement/raw-empty-01"))
    rows.append(measure("long-R300", ROOT / "tmp/p5-g2-native300-benchmark/saved-long-R"))
    synthetic = {"floatbits": [-0.0, 0.0, float("inf"), float("-inf"), float("nan")],
        "mixed": [True, 1, 1.0, None, "값", [], {}, {"a": 1}, {"b": 2, "a": 3}],
        "bigint": [2**128, -(2**128)], "ragged": [[], [1.0], [2.0, 3.0]],
        "unicode_keys": {"한국어": ["가", "나", "가"]}}
    path = OUT / "synthetic.columnar.xz"; receipt = write_columnar(path, synthetic)
    assert tree_digest(synthetic) == tree_digest(read_columnar(path, receipt["sha256"]))
    contact, empty, long = rows
    measured_increment = (contact["new_total_bytes"] - empty["new_total_bytes"]) / 2
    budget = json.loads((ROOT / "configs/p5_g2_resource_plan.v1.json").read_text())
    new_raw = int(np.ceil(1.5 * (measured_increment * budget["counts"]["raw_intervals"]
                               + empty["new_total_bytes"] * budget["counts"]["raw_records"])))
    new_total = budget["estimate"]["required_bytes_including_reserve"] - budget["estimate"]["parts_bytes"]["raw"] + new_raw
    guard()
    report = {"schema": "p5-storage-feasibility-probe-v1", "status": "PASS_PROTOTYPE_NOT_PRODUCTION_GATE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "rows": rows,
        "synthetic_mixed_ragged_signedzero_nonfinite_bigint_roundtrip": True,
        "own_bytes": bytes_used(), "own_max_bytes": 64 * MIB,
        "free_before": before, "free_after": shutil.disk_usage(ROOT).free,
        "new_model_calls": 0, "new_physics_calls": 0, "new_optimizer_updates": 0,
        "all_original_hashes_unchanged": True,
        "illustrative_budget_same_1_5_multiplier_and_scope": {
            "contact_increment_bytes_per_interval": measured_increment,
            "empty_fixed_bytes": empty["new_total_bytes"] , "raw_component_bytes": new_raw,
            "total_including_unchanged_reserve_bytes": new_total,
            "old_total_bytes": budget["estimate"]["required_bytes_including_reserve"],
            "estimated_saving_bytes": budget["estimate"]["required_bytes_including_reserve"] - new_total,
            "status": "ILLUSTRATIVE_TWO_REGIMES_ONLY_REQUIRES_BROADER_MEASUREMENT_AND_PRODUCTION_TESTS"},
        "limits": ["Existing files only; no final controller/model integration or new simulation.",
            "Probe stores complete metadata tree in columnar floats/offsets; production input validation, corruption tests and reader integration remain.",
            "NPZ is referenced unchanged, not duplicated. New output production must continue saving all numerical arrays and complete diagnostics.",
            "Two measured action regimes do not prove a worst-case bound; retain1.5 allowance and live guards.",
            "No original file, codec candidate, model, scope, criteria or final recording library changed."],
        "source_sha256": {"tools/p5_storage_probe.py": sha(__file__),
            "src/push_align/final_recording.py": sha(ROOT / "src/push_align/final_recording.py"),
            "configs/p5_g2_resource_plan.v1.json": sha(ROOT / "configs/p5_g2_resource_plan.v1.json")}}
    with REPORT.open("x") as f: json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")
    print(json.dumps({"report": str(REPORT.relative_to(ROOT)), "report_sha256": sha(REPORT),
                      "rows": [{k: r[k] for k in ("label", "old_total_bytes", "new_total_bytes", "write_wall_s", "read_wall_s", "peak_process_MiB_so_far")} for r in rows],
                      "budget": report["illustrative_budget_same_1_5_multiplier_and_scope"]}, ensure_ascii=False))


if __name__ == "__main__": main()
