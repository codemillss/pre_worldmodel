#!/usr/bin/env python3
"""Saved-evidence-only comparison prototype; no production bindings change."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time
import types

START = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import followup_nt_executor as nt
import profile_nt_storage as profile
from push_align import final_driver_v2 as driver
from push_align import final_recording_columns as codec
from push_align import exact_pair_prototype as prototype


def need(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def clone(function, comparison):
    # Copy the original code object and globals without rebinding the module.
    namespace = dict(function.__globals__, exact_comparison=comparison)
    result = types.FunctionType(function.__code__, namespace, function.__name__,
                                function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def bundle(folder, probe, comparison):
    calls = []

    def counted(left, right):
        verdict = comparison(left, right)
        calls.append(verdict)
        return verdict

    write = clone(driver.write_full_trace, counted)
    read = clone(driver.read_full_trace, counted)
    started = time.perf_counter_ns()
    tree = nt.complete_probe_tree(probe)
    receipt = write(folder / "complete_probe", tick=78, captured_input=tree)
    restored, _, _ = read(receipt)
    need(counted(tree, restored)["status"] == "PASS_EXACT", "Restored tree differs")
    following = nt.reconstruct_probe(restored)
    need(counted(probe, following)["status"] == "PASS_EXACT", "Restored Probe differs")
    profile.durable(folder / "complete_probe_receipt.json", receipt)
    elapsed = time.perf_counter_ns() - started
    need(len(calls) == 4 and all(v["status"] == "PASS_EXACT" for v in calls),
         "All four original comparisons must remain")
    return {"elapsed_ns": elapsed, "exact_comparison_calls": len(calls),
            "comparison_verdicts": calls,
            "payload_sha256": {Path(p).name: h for p, h in receipt["files_sha256"].items()},
            "payload_bytes": sum(receipt["bytes"].values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = load(args.config)
    configuration_sha256 = profile.sha(args.config)
    original, output = args.original_root.resolve(), args.output.resolve()
    need(output.parent == original / "runs" and not output.exists(), "New exclusive run required")
    need(cfg["schema"] == "exact-pair-prototype-design-v1" and cfg["repeats"] == 3,
         "Unreviewed profile design")
    output.mkdir()
    baseline = load(ROOT / cfg["source_baseline_path"])
    need(profile.sha(ROOT / cfg["source_baseline_path"]) == cfg["source_baseline_sha256"],
         "Source baseline binding differs")
    for path, digest in cfg["implementation_sha256"].items():
        need(profile.sha(ROOT / path) == digest, "Registered implementation changed")
    guards = profile.no_compute_guards()
    bindings = {name: getattr(driver, name) for name in
                ("exact_comparison", "write_full_trace", "read_full_trace")}
    before_runs = {name: profile.inventory(original / "runs" / name)
                   for name in baseline["source_run_inventories"]}
    need(before_runs == baseline["source_run_inventories"], "Original source run changed")
    need(all(profile.sha(ROOT / p) == h for p, h in baseline["tracked_file_sha256"].items()),
         "Original tracked source changed before benchmark")
    started_utc = datetime.now(timezone.utc).isoformat()
    profile.durable(output / "entry.json", {"schema": "exact-pair-prototype-entry-v1",
        "config_sha256": configuration_sha256, "started_utc": started_utc,
        "code_sha256": cfg["implementation_sha256"],
        "source_baseline_sha256": cfg["source_baseline_sha256"],
        "scope": "Saved cases, warmed source cache, exclusive copies; no model/physics or production replacement."})
    rows, raw_rows, corruption_rows = [], [], []
    status, error = "COMPLETE_SAVED_EXACT_PAIR_PROTOTYPE", None

    def guard():
        need(time.monotonic() - START + cfg["sealing_reserve_seconds"] < cfg["process_cap_seconds"],
             "Prototype process cap")
        elapsed_task = (datetime.now(timezone.utc) - datetime.fromisoformat(cfg["task_clock_start_utc"])).total_seconds()
        need(cfg["prior_cumulative_seconds"] + cfg["pre_clock_allowance_seconds"] + elapsed_task
             + cfg["final_review_reserve_seconds"] < cfg["cumulative_cap_seconds"], "Cumulative cap")
        used = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        need(used < cfg["output_cap_bytes"], "Prototype output cap")
        need(shutil.disk_usage(output).free >= cfg["free_space_reserve_bytes"] + cfg["output_cap_bytes"],
             "Free disk reserve")

    try:
        for case in cfg["cases"]:
            guard()
            source = original / "runs" / case["run"] / case["trial_relative"]
            need(profile.sha(source / "summary.json") == case["summary_sha256"], "Source summary differs")
            receipt = load(source / "complete_probe_receipt.json")
            need(profile.sha(source / "complete_probe_receipt.json") == case["receipt_sha256"], "Source receipt differs")
            tree, _, _ = driver.read_full_trace(receipt)
            probe = nt.reconstruct_probe(tree)
            need(driver.typed_digest(probe) == case["whole_probe_typed_sha256"], "Probe source identity differs")
            raw_receipt = load(source / "raw_receipt.json")
            _, steps_left = codec.read_raw(source / "raw", expected_sha256=raw_receipt["files_sha256"])
            _, steps_right = codec.read_raw(source / "raw", expected_sha256=raw_receipt["files_sha256"])
            source_payload = {Path(p).name: h for p, h in receipt["files_sha256"].items()}
            for repetition in range(cfg["repeats"]):
                order = ("original", "paired") if repetition % 2 == 0 else ("paired", "original")
                paired_rows = []
                for label in order:
                    guard()
                    comparison = driver.exact_comparison if label == "original" else prototype.exact_comparison
                    folder = output / case["id"] / f"repeat-{repetition}" / label
                    folder.mkdir(parents=True, exist_ok=False)
                    row = bundle(folder, probe, comparison)
                    need(row["payload_sha256"] == source_payload, "Saved payload is not byte-identical to source")
                    row.update(case_id=case["id"], repeat=repetition, implementation=label, order=list(order))
                    rows.append(row)
                    paired_rows.append(row)
                    profile.durable(folder / "measurement.json", row)
                    tick = time.perf_counter_ns()
                    verdict = comparison(steps_left[-1], steps_right[-1])
                    raw_elapsed = time.perf_counter_ns() - tick
                    expected = driver.exact_comparison(steps_left[-1], steps_right[-1])
                    need(verdict == expected and verdict["status"] == "PASS_EXACT", "Raw interval contract differs")
                    raw_row = {"case_id": case["id"], "repeat": repetition, "implementation": label,
                               "elapsed_ns": raw_elapsed, "verdict": verdict,
                               "scope": "Last saved StepResult only, independent raw decodes; not full raw persistence time."}
                    raw_rows.append(raw_row)
                    profile.durable(folder / "raw_interval_comparison.json", raw_row)
                need(paired_rows[0]["comparison_verdicts"] == paired_rows[1]["comparison_verdicts"],
                     "Original/prototype comparison verdicts differ")
                print(json.dumps({"case": case["id"], "repeat": repetition,
                    "bundle_seconds": {r["implementation"]: r["elapsed_ns"] / 1e9 for r in paired_rows}}), flush=True)

            # Corrupt a *new copy*, retaining both original and altered evidence.
            # Each reader receives the copy's original signed SHA receipts.
            for label, comparison in (("original", driver.exact_comparison), ("paired", prototype.exact_comparison)):
                guard()
                copied = output / case["id"] / ("corruption-" + label)
                copied.mkdir()
                source_copy = output / case["id"] / "repeat-0" / label / "complete_probe"
                hashes = {}
                for filename, digest in source_payload.items():
                    target = copied / filename
                    shutil.copyfile(source_copy / filename, target)
                    hashes[str(target)] = digest
                target = copied / "trace.npz"
                with target.open("r+b") as stream:
                    stream.seek(-1, 2)
                    last = stream.read(1)
                    stream.seek(-1, 2)
                    stream.write(bytes([last[0] ^ 1]))
                    stream.flush()
                    import os
                    os.fsync(stream.fileno())
                caught = None
                try:
                    clone(driver.read_full_trace, comparison)({"files_sha256": hashes})
                except Exception as exc:
                    caught = {"type": type(exc).__name__, "message": str(exc)}
                need(caught is not None, "Corrupt copied payload accepted")
                corruption_rows.append({"case_id": case["id"], "implementation": label, "rejection": caught})
        for case in cfg["cases"]:
            rejections = [r["rejection"] for r in corruption_rows if r["case_id"] == case["id"]]
            need(len(rejections) == 2 and rejections[0] == rejections[1],
                 "Original/prototype corruption rejections differ")
        need(all(v == 0 for v in guards.values()), "Forbidden compute occurred")
    except (Exception, KeyboardInterrupt) as exc:
        status, error = "STOP_EXACT_PAIR_PROTOTYPE", {"type": type(exc).__name__, "message": str(exc)}

    tracked_unchanged = all(profile.sha(ROOT / p) == h for p, h in baseline["tracked_file_sha256"].items())
    runs_unchanged = {name: profile.inventory(original / "runs" / name) == before
                      for name, before in before_runs.items()}
    production_bindings_unchanged = all(getattr(driver, name) is value for name, value in bindings.items())
    implementation_unchanged = all(profile.sha(ROOT / p) == h for p, h in cfg["implementation_sha256"].items())
    configuration_unchanged = profile.sha(args.config) == configuration_sha256
    if not (tracked_unchanged and all(runs_unchanged.values()) and production_bindings_unchanged
            and implementation_unchanged and configuration_unchanged):
        status, error = "STOP_EXACT_PAIR_PROTOTYPE", {"type": "SourceChanged", "message": "Preserved source changed"}
    aggregate = {}
    for case in cfg["cases"]:
        group = {}
        for scope, measurements in (("complete_bundle", rows), ("raw_interval_comparison", raw_rows)):
            values = {label: [r["elapsed_ns"] / 1e9 for r in measurements
                             if r["case_id"] == case["id"] and r["implementation"] == label]
                      for label in ("original", "paired")}
            if all(len(v) == cfg["repeats"] for v in values.values()):
                medians = {label: statistics.median(times) for label, times in values.items()}
                group[scope] = {"samples_seconds": values, "median_seconds": medians,
                    "paired_over_original_median": medians["paired"] / medians["original"]}
        aggregate[case["id"]] = group
    summary = {"schema": "exact-pair-prototype-summary-v1", "status": status, "error": error,
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": configuration_sha256, "rows": rows, "raw_comparison_rows": raw_rows,
        "corruption_rejections": corruption_rows, "aggregate": aggregate,
        "original_tracked_files_unchanged": tracked_unchanged, "original_tracked_file_count": len(baseline["tracked_file_sha256"]),
        "source_runs_unchanged": runs_unchanged, "production_bindings_unchanged": production_bindings_unchanged,
        "implementation_unchanged": implementation_unchanged, "configuration_unchanged": configuration_unchanged,
        "forbidden_compute_calls": guards, "new_actual_trajectories": 0, "training_updates": 0,
        "process_seconds_before_summary": time.monotonic() - START,
        "limitations": ["Two selected warm-cache saved cases; not a worst-case runtime bound.",
            "Raw measurements cover exact comparison only; physical execution, raw codec and repeated-trial overhead are uncalibrated.",
            "All four complete-Probe exact comparisons retained; source codec unchanged except private prototype comparison binding.",
            "No production replacement, historical rate decrease, model/cost/candidate change or scientific-run resumption."]}
    profile.durable(output / "summary.json", summary)
    print(json.dumps({"status": status, "error": error, "aggregate": aggregate}), flush=True)
    return 0 if status == "COMPLETE_SAVED_EXACT_PAIR_PROTOTYPE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
