#!/usr/bin/env python3
"""Aggregate saved timing trees without adding overlapping inclusive times."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import time


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def category(path):
    if "/self_compare" in path or path.startswith(("bundle/tree_compare", "bundle/probe_compare")):
        return "four_exact_comparisons"
    if path.endswith("/typed_encode"):
        return "typed_tree_encoding"
    if path.endswith(("/typed_tree_restore", "/probe_restore")):
        return "typed_restoration"
    if path.endswith(("/external_sha", "/internal_npz_sha", "/metadata_sha", "/npz_sha", "/receipt_hash_stat")):
        return "file_authentication_hashes_and_stats"
    if path.endswith(("/gzip_decompress_json_parse", "/npz_decompress_copy")):
        return "reread_decompress_parse_copy"
    if path.endswith("/json_encode"):
        return "json_serialization"
    if path.endswith(("/npz_compress_write", "/npz_flush", "/npz_fsync", "/gzip_write", "/flush", "/fsync", "/receipt_write")):
        return "compression_stream_writes_flush_fsync"
    return "tree_build_control_and_timer_overhead"


def summarize(run):
    begin = time.perf_counter()
    run = Path(run)
    summary_path, process_path = run / "summary.json", run / "process_wall_receipt.json"
    summary, process = [json.loads(p.read_text()) for p in (summary_path, process_path)]
    require(summary["status"] == "COMPLETE_SAVED_STORAGE_PROFILE" and process["exit_code"] == 0, "Incomplete profile")
    require(all(v == 0 for v in summary["forbidden_compute_calls"].values()), "Compute guard nonzero")
    grouped = defaultdict(list)
    for row in summary["trials"]:
        stages = row["stages"]
        require(sum(r["exclusive_ns"] for r in stages.values()) == row["total_ns"], "Wall timer sum differs")
        require(sum(r["exclusive_cpu_ns"] for r in stages.values()) == stages["bundle"]["inclusive_cpu_ns"], "CPU timer sum differs")
        buckets = defaultdict(int)
        for path, timer in stages.items():
            buckets[category(path)] += timer["exclusive_ns"]
        require(sum(buckets.values()) == row["total_ns"], "Partition dropped/doubled time")
        grouped[row["case_id"]].append((row, dict(buckets)))
    cases = {}
    for name, values in grouped.items():
        rows = [v[0] for v in values]
        require(sorted(r["repeat"] for r in rows) == [0, 1, 2], "Missing/duplicate case repeats")
        require(all(r["shape"] == rows[0]["shape"] for r in rows), "Same-source complexity differs")
        totals = [r["total_ns"] / 1e9 for r in rows]
        top = {path: statistics.mean(r["stages"][path]["inclusive_ns"] / 1e9 for r in rows)
               for path in rows[0]["stages"] if path.count("/") == 1}
        keys = sorted({key for _, buckets in values for key in buckets})
        parts = {key: statistics.mean(buckets.get(key, 0) / 1e9 for _, buckets in values) for key in keys}
        require(abs(sum(parts.values()) - statistics.mean(totals)) < 1e-12, "Mean parts differ from mean bundle")
        cases[name] = {"bundle_seconds": totals, "mean_seconds": statistics.mean(totals),
            "median_seconds": statistics.median(totals), "min_seconds": min(totals), "max_seconds": max(totals),
            "mean_top_level_inclusive_seconds": top, "mean_exclusive_category_seconds": parts,
            "mean_exclusive_category_percent": {k: 100*v/statistics.mean(totals) for k, v in parts.items()},
            "mean_cpu_seconds": statistics.mean(r["stages"]["bundle"]["inclusive_cpu_ns"] / 1e9 for r in rows),
            "shape": rows[0]["shape"], "output_bytes_by_repeat": [r["output_bytes"] for r in rows]}
    calibration = summary["reference_calibration"]
    ratios = [r["calibration"]["timed_over_reference_ratio"] for r in summary["trials"] if "calibration" in r]
    all_bundles = summary["trials"] + calibration
    return {"schema": "nt-storage-profile-attribution-v1", "status": "PASS_SAVED_TIMING_PARTITION",
        "source_summary_sha256": sha(summary_path), "source_process_wall_sha256": sha(process_path),
        "summarizer_sha256": sha(__file__), "timed_bundles": len(summary["trials"]),
        "original_reference_bundles": len(calibration), "cases": cases,
        "reference_original_seconds": [r["total_ns"] / 1e9 for r in calibration],
        "timed_over_reference_ratios": ratios,
        "maximum_observed_complete_bundle_seconds_including_original_reference": max(r["total_ns"] for r in all_bundles) / 1e9,
        "maximum_complete_bundle_output_bytes": max(r["output_bytes"] for r in all_bundles),
        "outer_profile_process_seconds": process["process_wall_seconds"],
        "aggregation_seconds": time.perf_counter() - begin,
        "limits": ["Six fixed saved sources; warmed filesystem cache; three repetitions are timing checks, not independent task trials.",
            "Top-level inclusive phases partition the bundle but overlap their own children. Only the separately partitioned exclusive categories are summed.",
            "Four exact comparisons include both typed hashes and direct recursive equality. No hash-only or weakened validation was substituted.",
            "Compression/stream writes include compute and buffered IO, not pure physical disk latency.",
            "Paired calibration reports ordinary timing variability as well as instrumentation; it cannot isolate a precise instrumentation overhead.",
            "Larger nested source data and measured processing burden narrow the workload cause, not historical OS/cache/thermal conditions or all unexecuted candidates."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "cases": len(result["cases"]),
        "max_bundle_seconds": result["maximum_observed_complete_bundle_seconds_including_original_reference"]}))
