#!/usr/bin/env python3
"""Profile unchanged lossless saved-Probe operations, never model or physics.

Only timing context managers are inserted into private in-memory function copies.
Removing those contexts must recover the original AST exactly. Source files and
the stopped experiment are immutable; each profiling output is exclusive.
"""
from __future__ import annotations

import time
START = time.monotonic()
import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import followup_nt_executor as nt
from push_align import final_driver_v2 as driver
from push_align import final_recording as recording


def need(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def inventory(folder):
    return {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob("*")) if p.is_file()}


def durable(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())


class Meter:
    def __init__(self, wall=time.perf_counter_ns, cpu=time.process_time_ns):
        self.wall, self.cpu = wall, cpu
        self.stack, self.records = [], {}

    @contextmanager
    def span(self, name):
        path = "/".join([s["name"] for s in self.stack] + [name])
        item = {"name": name, "wall": self.wall(), "cpu": self.cpu(), "child_wall": 0, "child_cpu": 0}
        self.stack.append(item)
        try:
            yield
        finally:
            elapsed, cpu = self.wall() - item["wall"], self.cpu() - item["cpu"]
            self.stack.pop()
            record = self.records.setdefault(path, {"calls": 0, "inclusive_ns": 0, "exclusive_ns": 0,
                "inclusive_cpu_ns": 0, "exclusive_cpu_ns": 0})
            record["calls"] += 1
            record["inclusive_ns"] += elapsed
            record["exclusive_ns"] += elapsed - item["child_wall"]
            record["inclusive_cpu_ns"] += cpu
            record["exclusive_cpu_ns"] += cpu - item["child_cpu"]
            if self.stack:
                self.stack[-1]["child_wall"] += elapsed
                self.stack[-1]["child_cpu"] += cpu

    def verify(self):
        need(not self.stack and "bundle" in self.records, "Unclosed/missing timer")
        need(all(r["exclusive_ns"] >= 0 and r["exclusive_cpu_ns"] >= 0 for r in self.records.values()),
             "Negative exclusive timing")
        need(sum(r["exclusive_ns"] for r in self.records.values()) == self.records["bundle"]["inclusive_ns"],
             "Nested wall time double-count or omission")
        need(sum(r["exclusive_cpu_ns"] for r in self.records.values()) == self.records["bundle"]["inclusive_cpu_ns"],
             "Nested CPU time double-count or omission")


class StripTimers(ast.NodeTransformer):
    def visit_With(self, node):
        node = self.generic_visit(node)
        if len(node.items) == 1:
            call = node.items[0].context_expr
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and \
                    isinstance(call.func.value, ast.Name) and call.func.value.id == "_profile_meter":
                return node.body
        return node


def instrument(function, namespace, labels):
    """Copy the original function body; add only selected timing contexts."""
    original = ast.parse(textwrap.dedent(inspect.getsource(function)))
    transformed = ast.parse(textwrap.dedent(inspect.getsource(function)))
    used = set()

    class AddTimers(ast.NodeTransformer):
        def generic_visit(self, node):
            if isinstance(node, ast.stmt) and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                first = ast.unparse(node).splitlines()[0]
                label = next((label for prefix, label in labels if first.startswith(prefix)), None)
                node = super().generic_visit(node)
                if label:
                    used.add(label)
                    context = ast.Call(func=ast.Attribute(value=ast.Name(id="_profile_meter", ctx=ast.Load()),
                        attr="span", ctx=ast.Load()), args=[ast.Constant(label)], keywords=[])
                    return ast.copy_location(ast.With(items=[ast.withitem(context_expr=context)], body=[node]), node)
                return node
            return super().generic_visit(node)

    transformed = ast.fix_missing_locations(AddTimers().visit(transformed))
    stripped = StripTimers().visit(ast.parse(ast.unparse(transformed)))
    need(ast.dump(stripped, include_attributes=False) == ast.dump(original, include_attributes=False),
         "Instrumentation changed source semantics: " + function.__name__)
    need(used == {label for _, label in labels}, "Unmatched instrumentation labels: " + function.__name__)
    exec(compile(transformed, "<timed-original:" + function.__name__ + ">", "exec"), namespace)
    return namespace[function.__name__]


def timed_functions(meter):
    # Original recursion still resolves in original module globals, so timers
    # are not injected once per leaf and original function objects stay intact.
    namespace = dict(driver.__dict__, _profile_meter=meter)
    exact = instrument(driver.exact_comparison, namespace, [
        ("a, b =", "typed_hashes"), ("equal =", "direct_recursive_equality")])
    rec_namespace = dict(recording.__dict__, _profile_meter=meter)
    write_json = instrument(recording.write_json_gzip, rec_namespace, [
        ("payload =", "json_encode"), ("with gzip.GzipFile", "gzip_write"),
        ("target.flush()", "flush"), ("os.fsync", "fsync"), ("return _receipt", "receipt_hash_stat")])
    read_json = instrument(recording.read_json_gzip, rec_namespace, [
        ("if expected_sha256", "metadata_sha"), ("with gzip.open", "gzip_decompress_json_parse")])
    namespace.update(exact_comparison=exact, write_json_gzip=write_json, read_json_gzip=read_json)
    write = instrument(driver.write_full_trace, namespace, [
        ("encoded =", "typed_encode"), ("info =", "self_compare"),
        ("np.savez_compressed", "npz_compress_write"), ("target.flush()", "npz_flush"),
        ("os.fsync", "npz_fsync"), ("info['npz_sha256']", "npz_sha"),
        ("write_json_gzip", "metadata_save"), ("return {'tick'", "receipt_hash_stat")])
    read = instrument(driver.read_full_trace, namespace, [
        ("for path, expected", "external_sha"), ("metadata =", "metadata_read"),
        ("if file_sha256(npz_path)", "internal_npz_sha"), ("with np.load", "npz_decompress_copy"),
        ("value = _decode_input", "typed_tree_restore"),
        ("if exact_comparison", "self_compare")])
    return write, read, exact


def bundle(folder, probe, *, meter=None):
    write, read, exact = (driver.write_full_trace, driver.read_full_trace, driver.exact_comparison) \
        if meter is None else timed_functions(meter)

    @contextmanager
    def span(name):
        if meter is None:
            yield
        else:
            with meter.span(name):
                yield

    start = time.perf_counter_ns()
    with span("bundle"):
        with span("tree_build"):
            tree = nt.complete_probe_tree(probe)
        with span("write"):
            receipt = write(folder / "complete_probe", tick=78, captured_input=tree)
        with span("read"):
            restored, _, _ = read(receipt)
        with span("tree_compare"):
            tree_verdict = exact(tree, restored)
            need(tree_verdict["status"] == "PASS_EXACT", "Complete tree roundtrip failed")
        with span("probe_restore"):
            following = nt.reconstruct_probe(restored)
        with span("probe_compare"):
            whole_verdict = exact(probe, following)
            need(whole_verdict["status"] == "PASS_EXACT", "Whole reconstructed Probe differs")
        with span("receipt_write"):
            durable(folder / "complete_probe_receipt.json", receipt)
    elapsed = time.perf_counter_ns() - start
    if meter:
        meter.verify()
    return {"total_ns": meter.records["bundle"]["inclusive_ns"] if meter else elapsed,
        "outer_bundle_ns": elapsed, "stages": meter.records if meter else None,
        "tree_exact": tree_verdict, "whole_probe_exact": whole_verdict,
        "payload_sha256": {Path(p).name: h for p, h in receipt["files_sha256"].items()},
        "output_bytes": sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())}


def shape_stats(probe):
    counts = {"step_results": len(probe.results), "diagnostic_samples": 0,
              "diagnostic_contact_records": 0, "substep_samples": 0,
              "diagnostic_nodes": 0, "diagnostic_dicts": 0,
              "diagnostic_sequences": 0, "diagnostic_scalar_leaves": 0}
    def visit(value):
        counts["diagnostic_nodes"] += 1
        if isinstance(value, dict):
            counts["diagnostic_dicts"] += 1
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple, np.ndarray)):
            counts["diagnostic_sequences"] += 1
            for item in value:
                visit(item)
        else:
            counts["diagnostic_scalar_leaves"] += 1
    for step in probe.results:
        visit(step.diagnostics)
        counts["substep_samples"] += len(step.substep_states)
        samples = step.diagnostics.get("samples", [])
        counts["diagnostic_samples"] += len(samples)
        for sample in samples:
            if isinstance(sample, dict):
                contacts = sample.get("contacts", [])
                if isinstance(contacts, (list, tuple)):
                    counts["diagnostic_contact_records"] += len(contacts)
    return counts


def no_compute_guards():
    counts = {}
    def block(name):
        counts[name] = 0
        def fail(*args, **kwargs):
            counts[name] += 1
            raise RuntimeError("Forbidden compute in saved-storage profile: " + name)
        return fail
    import mujoco
    for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse"):
        setattr(mujoco, name, block(name))
    from push_align.environment import PlanarPushEnv
    for name in ("__init__", "reset", "restore", "step"):
        setattr(PlanarPushEnv, name, block("environment." + name))
    import torch
    torch.nn.Module.__call__ = block("model.__call__")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = load(args.config)
    original, output = args.original_root.resolve(), args.output.resolve()
    need(not output.exists() and output.parent == original / "runs", "Exclusive new run folder required")
    need(cfg["schema"] == "nt-storage-profile-design-v1" and cfg["repeats"] == 3, "Profile design mismatch")
    output.mkdir()
    guards = no_compute_guards()
    source_inventories, probes, trial_records, calibration_records = {}, {}, [], []
    source_code = {p: sha(ROOT / p) for p in cfg["code_paths"]}
    need(source_code == cfg["source_code_sha256"], "Profile or preserved codec source changed")
    frozen_run = original / "runs/followup-nt-full-20261022-01"
    for path, digest in ((frozen_run / "summary.json", cfg["bindings"]["source_run_summary_sha256"]),
                         (frozen_run / "entry.json", cfg["bindings"]["source_entry_sha256"]),
                         (ROOT / "configs/followup_nt_full.v1.json", cfg["bindings"]["executed_config_sha256"]),
                         (ROOT / "artifacts/validation/nt_full_delivery_review.v1.json", cfg["bindings"]["prior_delivery_review_sha256"])):
        need(sha(path) == digest, "Preserved source or prior ledger binding changed")
    need(cfg["prior_conservative_seconds"] + cfg["process_cap_seconds"] + 600 < 7200,
         "Profile and final review reserve do not fit cumulative cap")
    entry = {"schema": "nt-storage-profile-entry-v1", "config_sha256": sha(args.config),
        "source_code_sha256": source_code, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "bindings": cfg["bindings"],
        "source_runs": {}, "case_order_by_round": cfg["case_order_by_round"],
        "cache_policy": "Existing source data authenticated/read before timed copies; warm page cache, no cache flushing or cold-I/O claim."}
    status, failure = "COMPLETE_SAVED_STORAGE_PROFILE", None
    try:
        for run_name in cfg["source_runs"]:
            run = original / "runs" / run_name
            source_inventories[run_name] = inventory(run)
            entry["source_runs"][run_name] = {"file_count": len(source_inventories[run_name]),
                "inventory_sha256": hashlib.sha256(json.dumps(source_inventories[run_name], sort_keys=True).encode()).hexdigest()}
        durable(output / "entry.json", entry)
        for case in cfg["cases"]:
            source = original / "runs" / case["run"] / case["trial_relative"]
            need(sha(source / "summary.json") == case["summary_sha256"], "Source summary differs")
            receipt = load(source / "complete_probe_receipt.json")
            need(sha(source / "complete_probe_receipt.json") == case["receipt_sha256"], "Source receipt differs")
            tree, _, _ = driver.read_full_trace(receipt)
            probe = nt.reconstruct_probe(tree)
            need(driver.typed_digest(probe) == case["whole_probe_typed_sha256"], "Source Probe typed identity differs")
            probes[case["id"]] = probe
        # Alternating reference/timed order per round on the slow witness. Both
        # use the same saved Probe. Calibration outputs stay preserved.
        for repetition, order in enumerate(cfg["case_order_by_round"]):
            need(sorted(order) == sorted(probes), "Case order omitted/duplicated source")
            for index, case_id in enumerate(order):
                free = shutil.disk_usage(output).free
                need(free >= cfg["free_space_reserve_bytes"] + cfg["output_cap_bytes"], "Profile free-space reserve")
                need(time.monotonic() - START + cfg["sealing_reserve_seconds"] < cfg["process_cap_seconds"], "Profile time cap")
                need(sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) < cfg["output_cap_bytes"], "Profile output cap")
                folder = output / f"round-{repetition}" / case_id
                folder.mkdir(parents=True, exist_ok=False)
                calibration = case_id == cfg["calibration_case_id"]
                reference = None
                def reference_pass():
                    target = output / f"calibration-{repetition}"
                    target.mkdir(exist_ok=False)
                    result = bundle(target, probes[case_id])
                    result.update(case_id=case_id, repeat=repetition, order="before" if repetition % 2 == 0 else "after")
                    calibration_records.append(result)
                    durable(target / "measurement.json", result)
                    return result
                if calibration and repetition % 2 == 0:
                    reference = reference_pass()
                meter = Meter()
                record = bundle(folder, probes[case_id], meter=meter)
                record.update(case_id=case_id, repeat=repetition, order_index=index,
                    shape=shape_stats(probes[case_id]), process_wall_seconds=time.monotonic() - START)
                if calibration and repetition % 2 == 1:
                    reference = reference_pass()
                if calibration:
                    need(record["payload_sha256"] == reference["payload_sha256"], "Instrumented/reference payload bytes differ")
                    record["calibration"] = {"reference_total_ns": reference["total_ns"],
                        "timed_over_reference_ratio": record["total_ns"] / reference["total_ns"],
                        "payload_bytes_identical": True}
                trial_records.append(record)
                durable(folder / "measurement.json", record)
                print(json.dumps({"case_id": case_id, "repeat": repetition,
                    "completed_profile_trials": len(trial_records), "bundle_seconds": record["total_ns"] / 1e9,
                    "elapsed_seconds": time.monotonic() - START}), flush=True)
        need(len(trial_records) == len(cfg["cases"]) * 3 and all(v == 0 for v in guards.values()), "Profile dimension/compute guard mismatch")
    except (Exception, KeyboardInterrupt) as error:
        status, failure = "STOP_STORAGE_PROFILE_INCOMPLETE", {"type": type(error).__name__, "message": str(error)}
    preservation = {name: inventory(original / "runs" / name) == previous for name, previous in source_inventories.items()}
    code_preserved = {p: sha(ROOT / p) == digest for p, digest in source_code.items()}
    if not all(preservation.values()) or not all(code_preserved.values()) or sha(args.config) != entry["config_sha256"]:
        status, failure = "STOP_STORAGE_PROFILE_INCOMPLETE", {"type": "SourceChanged", "message": "Source/code/config inventory changed"}
    summary = {"schema": "nt-storage-profile-summary-v1", "status": status, "error": failure,
        "bindings": cfg["bindings"],
        "entry_sha256": sha(output / "entry.json"), "trials": trial_records, "reference_calibration": calibration_records,
        "source_runs_unchanged": preservation, "source_code_unchanged": code_preserved,
        "forbidden_compute_calls": guards, "training_updates": 0, "new_actual_trajectories": 0,
        "wall_seconds": time.monotonic() - START, "peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "output_bytes_before_summary": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
        "limits": ["Fixed saved case profiling with warmed source cache, not independent physics or generalization evidence.",
            "Inclusive parent/child durations overlap. Only exclusive categories may be added; exact sum checked for every profile.",
            "Compression and stream IO remain combined where named. No direct physical-disk latency or original historical OS cause is measured.",
            "Original global maxima cannot be lowered from faster new samples. No full-run continuation was launched."]}
    durable(output / "summary.json", summary)
    print(json.dumps({"status": status, "trials": len(trial_records), "wall_seconds": summary["wall_seconds"]}), flush=True)
    return 0 if status == "COMPLETE_SAVED_STORAGE_PROFILE" else 2


if __name__ == "__main__":
    sys.exit(main())
