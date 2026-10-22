#!/usr/bin/env python3
"""Profile existing D2 evidence without physics, models or original mutations.

The production codec and exact comparator are called unchanged. Timed wrappers
attribute nested calls by exclusive wall time, retaining every authentication,
fsync and bit-exact check. Inputs are authenticated once before timed passes;
thus these are warm-filesystem measurements, never cold-cache I/O claims.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import cProfile
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pstats
import shutil
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from push_align import final_driver_v2 as driver
from push_align import final_recording as recording
from push_align import final_recording_columns as codec

CAP_SECONDS = 1200
CAP_BYTES = 512 * 1024**2
RESERVE_BYTES = 2 * 1024**3
SCIENCE_KEYS = (
    "state", "candidate_id", "panel", "physical_valid", "horizon_completed",
    "intervals", "termination", "reasons", "first_ready_interval",
    "ready_prefix", "ready", "ready_eligible", "cost", "cost_finite",
    "cost_components", "initial_snapshot_exact",
)


def dump(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def relative(path, base):
    return Path(path).resolve().relative_to(base.resolve()).as_posix()


class ExclusiveTimers:
    """Measure mutually exclusive categories, including recursive calls."""

    def __init__(self):
        self.ns = Counter()
        self.calls = Counter()
        self.stack = []

    def wrap(self, category, function):
        def measured(*args, **kwargs):
            frame = [time.perf_counter_ns(), 0]
            self.stack.append(frame)
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = time.perf_counter_ns() - frame[0]
                self.stack.pop()
                self.ns[category] += elapsed - frame[1]
                self.calls[category] += 1
                if self.stack:
                    self.stack[-1][1] += elapsed
        return measured

    @contextmanager
    def instrumentation(self):
        targets = [
            (codec, "write_raw", "write_bookkeeping"),
            (codec, "read_raw", "read_npz_validate_reconstruct"),
            (codec, "_raw_payload", "payload_construct"),
            (codec, "pack_column", "column_pack"),
            (codec, "_write_columns", "column_header_xz_write"),
            (codec, "_read_columns", "column_read_decompress_validate"),
            (codec, "unpack_column", "column_unpack"),
            (codec, "file_sha256", "file_hash"),
            (recording, "file_sha256", "file_hash"),
            (np, "savez_compressed", "npz_compress_write"),
            (os, "fsync", "fsync"),
            (driver, "exact_comparison", "direct_exact_compare"),
            (driver, "typed_digest", "typed_digest"),
        ]
        # Binding originals before patching prevents wrapper-on-wrapper aliases.
        originals = [(module, name, category, getattr(module, name))
                     for module, name, category in targets]
        with ExitStack() as stack:
            for module, name, category, function in originals:
                stack.enter_context(patch.object(module, name, self.wrap(category, function)))
            yield

    def snapshot(self):
        return dict(self.ns), dict(self.calls)


def stage(name, function):
    timer = ExclusiveTimers()
    with timer.instrumentation():
        started = time.perf_counter_ns()
        value = function()
        total = time.perf_counter_ns() - started
    phases, calls = timer.snapshot()
    phases["unattributed_call_overhead"] = total - sum(phases.values())
    assert phases["unattributed_call_overhead"] >= 0
    assert total == sum(phases.values())
    return value, {"stage": name, "total_ns": total,
                   "exclusive_phase_ns": phases, "phase_calls": calls,
                   "sum_exclusive_ns": sum(phases.values())}


def row_path(row):
    return (Path("D2") / row["state"] / f"candidate-{row['candidate_id']:03d}"
            / row["panel"] / f"repeat-{row['repeat']}")


def choose_cases(rows, cases, run, bind):
    """Keep state/panel size extrema, all ready cases, then deterministic fill."""
    first = [row for row in rows if row["repeat"] == 0]
    first.sort(key=lambda row: (row["state"], row["panel"], row["candidate_id"]))
    for row in first:
        receipt_path = run / row_path(row) / "raw_receipt.json"
        bind(receipt_path)
        receipt = json.loads(receipt_path.read_text())
        row["profile_source_raw_bytes"] = sum(receipt["bytes"].values())
    selected = {}
    for state, panel in sorted({(row["state"], row["panel"]) for row in first}):
        group = [row for row in first if (row["state"], row["panel"]) == (state, panel)]
        ordered = sorted(group, key=lambda row: (row["profile_source_raw_bytes"], row["candidate_id"]))
        for row in (ordered[0], ordered[-1]):
            selected[(state, panel, row["candidate_id"])] = row
    for row in first:
        if row["ready"]:
            selected[(row["state"], row["panel"], row["candidate_id"])] = row
    if len(selected) > cases:
        raise ValueError(f"--cases must be >= {len(selected)} for extrema and ready coverage")
    # Round-robin state/panel groups avoids filling with only the first state.
    groups = sorted({(row["state"], row["panel"]) for row in first})
    for candidate in sorted({row["candidate_id"] for row in first}):
        for state, panel in groups:
            row = next(row for row in first if (row["state"], row["panel"], row["candidate_id"]) == (state, panel, candidate))
            if len(selected) < cases:
                selected[(state, panel, candidate)] = row
    return sorted(selected.values(), key=lambda row: (row["state"], row["panel"], row["candidate_id"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cases", type=int, default=9)
    ap.add_argument("--passes", type=int, default=3)
    args = ap.parse_args()
    if not 9 <= args.cases <= 64 or not 1 <= args.passes <= 5:
        ap.error("cases must be 9..64 and passes 1..5")
    run, out = args.run.resolve(), args.output.resolve()
    if out == run or run in out.parents:
        ap.error("output must be outside the immutable source run")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    bound, rows = {}, []
    summary = {"schema": "followup-existing-evidence-io-profile-v1", "status": "INCOMPLETE",
               "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "physics_calls": 0, "model_calls": 0, "training_updates": 0,
               "source_writes": 0, "automatic_retries": 0,
               "cache_policy": "Authenticated source load before timing; repeated filesystem/cache state, no cold-cache claim.",
               "phase_policy": "Nested wrappers measure exclusive perf_counter_ns; production codec, fsync, authentication and exact comparator unchanged.",
               "repeat_scope": "Exact complete saved initial snapshot/StepResults, every saved diagnostic NPZ array, and all scientific summary fields; this does not recreate unsaved FollowupProbe final integration snapshots or replace original D2 full-result repeat checks.",
               "limits": {"wall_s": CAP_SECONDS, "incremental_bytes": CAP_BYTES, "free_reserve_bytes": RESERVE_BYTES},
               "implementation_sha256": {p: recording.file_sha256(ROOT / p) for p in (
                   "scripts/profile_followup_io.py", "scripts/followup_pilot.py",
                   "src/push_align/final_recording_columns.py", "src/push_align/final_recording.py",
                   "src/push_align/final_driver_v2.py", "src/push_align/followup_probe.py")},
               "runtime": {"python": sys.version, "numpy": np.__version__, "platform": sys.platform},
               "measurements": rows}

    def bind(path):
        key = relative(path, run)
        digest = recording.file_sha256(path)
        if key in bound and bound[key] != digest:
            raise RuntimeError("Source changed during initial load: " + key)
        bound[key] = digest
        return digest

    def guard(extra=16 * 1024**2):
        used = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        if time.monotonic() - started > CAP_SECONDS or used + extra > CAP_BYTES:
            raise RuntimeError("STOP_RESOURCE_CAP before next profiling unit")
        if shutil.disk_usage(out).free < RESERVE_BYTES + extra:
            raise RuntimeError("STOP_FREE_SPACE_RESERVE before next profiling unit")

    def load_case(row, repeat):
        folder = run / row_path({**row, "repeat": repeat})
        receipt_path = folder / "raw_receipt.json"
        bind(receipt_path)
        receipt = json.loads(receipt_path.read_text())
        # Rebase receipt paths by basename, never trust an arbitrary source path.
        expected = {str(folder / Path(path).name): sha for path, sha in receipt["files_sha256"].items()}
        actual = codec.raw_paths(folder / "raw")
        assert set(expected) == {str(path) for path in actual}
        for path in actual:
            assert bind(path) == expected[str(path)]
        snapshot, steps = codec.read_raw(folder / "raw", expected_sha256=expected)
        diagnostic = folder / "diagnostic.npz"
        bind(diagnostic)
        with np.load(diagnostic, allow_pickle=False) as source:
            arrays = {name: source[name].copy() for name in source.files}
        saved = folder / "summary.json"
        bind(saved)
        metadata = json.loads(saved.read_text())
        assert metadata["raw_round_trip"] == "PASS_EXACT"
        assert metadata["repeat_comparison"] == ("FIRST_REPEAT" if repeat == 0 else "PASS_EXACT")
        return {"snapshot": snapshot, "steps": steps}, arrays, {key: metadata[key] for key in SCIENCE_KEYS}, receipt

    try:
        guard()
        source_summary = run / "D2" / "summary.json"
        bind(source_summary)
        pilot = json.loads(source_summary.read_text())
        assert pilot["status"] == "PASS_PILOT_RECORDING_AND_REPRODUCTION_GATES"
        selected = choose_cases(pilot["rows"], args.cases, run, bind)
        summary["selection"] = {"policy": "state/panel min and max raw bytes, every ready case, then balanced deterministic candidate fill; all selected source repeat=0", "cases": len(selected), "passes": args.passes,
                                "source_rows": [{key: row[key] for key in ("state", "panel", "candidate_id", "intervals", "profile_source_raw_bytes")} for row in selected]}
        dump(out / "entry.json", {k: v for k, v in summary.items() if k != "measurements"})
        loaded = []
        for row in selected:
            guard()
            versions = [load_case(row, repeat) for repeat in range(3)]
            loaded.append((row, versions))
        summary["initial_source_load_seconds"] = time.monotonic() - started
        summary["source_sha256"] = bound
        # Source load is fully authenticated but excluded from the timed rows.
        for pass_id in range(args.passes):
            for case_id, (row, versions) in enumerate(loaded):
                guard()
                raw, diagnostics, science, source_receipt = versions[0]
                folder = out / f"pass-{pass_id:02d}" / f"case-{case_id:03d}"
                folder.mkdir(parents=True, exist_ok=False)
                receipt, write_timing = stage("write_raw", lambda: codec.write_raw(folder / "raw", raw["snapshot"], raw["steps"], xz_preset=3))
                restored, read_timing = stage("read_raw", lambda: codec.read_raw(folder / "raw", expected_sha256=receipt["files_sha256"]))
                comparison, exact_timing = stage("raw_roundtrip_exact", lambda: driver.exact_comparison(raw, {"snapshot": restored[0], "steps": restored[1]}))
                assert comparison["status"] == "PASS_EXACT"

                def save_diagnostic():
                    with (folder / "diagnostic.npz").open("xb") as stream:
                        np.savez_compressed(stream, **diagnostics)
                _, diagnostic_timing = stage("diagnostic_npz", save_diagnostic)

                def compare_repeats():
                    verdicts = []
                    for other_raw, other_diagnostic, other_science, _ in versions[1:]:
                        verdict = driver.exact_comparison(
                            {"raw": raw, "diagnostic": diagnostics, "science": science},
                            {"raw": other_raw, "diagnostic": other_diagnostic, "science": other_science})
                        assert verdict["status"] == "PASS_EXACT"
                        verdicts.append(verdict)
                    return verdicts
                repeat_checks, repeat_timing = stage("two_saved_evidence_repeat_comparisons", compare_repeats)
                portable_receipt = {**receipt,
                    "files_sha256": {relative(Path(path), out): sha for path, sha in receipt["files_sha256"].items()},
                    "bytes": {relative(Path(path), out): size for path, size in receipt["bytes"].items()}}
                # Same raw basename/preset allows direct original byte proof.
                source_by_name = {Path(path).name: sha for path, sha in source_receipt["files_sha256"].items()}
                byte_identical = {Path(path).name: sha == source_by_name[Path(path).name]
                                  for path, sha in receipt["files_sha256"].items()}
                assert all(byte_identical.values()), "Fresh codec bytes differ from source"
                phases = [write_timing, read_timing, exact_timing, diagnostic_timing, repeat_timing]
                measurement = {"pass": pass_id, "case": case_id, "state": row["state"],
                    "panel": row["panel"], "candidate_id": row["candidate_id"],
                    "intervals": len(raw["steps"]), "phases": phases,
                    "sum_stage_ns": sum(item["total_ns"] for item in phases),
                    "raw_roundtrip": comparison, "source_repeat_checks": repeat_checks,
                    "raw_receipt": portable_receipt, "raw_bytes_identical_to_source": byte_identical}
                dump(folder / "measurement.json", measurement)
                rows.append(measurement)
                print(f"pass {pass_id + 1}/{args.passes}, case {case_id + 1}/{len(selected)} complete", flush=True)

        # Separate one-run CPU profile, never mix profiler overhead into timings.
        guard()
        worst_index = max(range(len(loaded)), key=lambda i: loaded[i][0]["profile_source_raw_bytes"])
        worst_row, versions = loaded[worst_index]
        raw = versions[0][0]
        profile = cProfile.Profile()
        profile_started = time.perf_counter_ns()
        profile.enable()
        assert driver.exact_comparison(raw, versions[1][0])["status"] == "PASS_EXACT"
        profile.disable()
        stats = pstats.Stats(profile)
        functions = []
        for (filename, line, name), (primitive, total, own_s, cumulative_s, _) in stats.stats.items():
            path = Path(filename)
            portable_name = relative(path, ROOT) if path.is_absolute() and (ROOT == path or ROOT in path.parents) else path.name
            functions.append({"file": portable_name, "line": line, "function": name,
                              "primitive_calls": primitive, "total_calls": total,
                              "own_s": own_s, "cumulative_s": cumulative_s})
        summary["separate_comparator_cprofile"] = {"state": worst_row["state"], "panel": worst_row["panel"], "candidate_id": worst_row["candidate_id"],
            "wall_ns": time.perf_counter_ns() - profile_started,
            "scope": "One complete source raw repeat comparison, after timed passes; no physics and no model.",
            "top_by_own_time": sorted(functions, key=lambda item: item["own_s"], reverse=True)[:30]}
        for rel, sha in bound.items():
            assert recording.file_sha256(run / rel) == sha, "Original source changed: " + rel
        for rel, sha in summary["implementation_sha256"].items():
            assert recording.file_sha256(ROOT / rel) == sha, "Execution code changed: " + rel
        summary["all_source_bindings_unchanged"] = len(bound)
        summary["status"] = "PASS_UNCHANGED_CODEC_AND_EXACT_SAVED_EVIDENCE_PROFILE"
        summary["total_wall_seconds"] = time.monotonic() - started
        summary["output_bytes_before_summary"] = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
        summary["aggregate_stage_ns"] = dict(sum((Counter({stage["stage"]: stage["total_ns"] for stage in row["phases"]}) for row in rows), Counter()))
        dump(out / "summary.json", summary)
        print(json.dumps({key: summary[key] for key in ("status", "total_wall_seconds", "all_source_bindings_unchanged")}), flush=True)
    except Exception as error:
        # Preserve partial profiling files and any completed measurements.
        summary["status"] = "STOP_INCOMPLETE_NO_RETRY"
        summary["error"] = {"type": type(error).__name__, "message": str(error).replace(str(run), "<source-run>").replace(str(out), "<profile-output>")}
        summary["source_sha256"] = bound
        summary["total_wall_seconds"] = time.monotonic() - started
        dump(out / "incomplete.json", summary)
        print(json.dumps({"status": summary["status"], "error": summary["error"]}), flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
