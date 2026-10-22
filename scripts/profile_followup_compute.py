#!/usr/bin/env python3
"""Time unchanged probe computation separately; verify and retain every new raw.

This is an execution/recording benchmark, not extra D3 evidence. Selected old
groups are rerun without a model and checked against their authenticated raw.
The original codec, physical parameters and full-result comparator are used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from push_align.environment import PlanarPushEnv
from push_align.final_driver_v2 import exact_comparison
from push_align.final_recording import file_sha256
from push_align import final_recording_columns as codec
from push_align.followup_probe import run_followup_probe
from followup_pilot import get_input
from followup_restore import write_json


def select_groups(run):
    rows = json.loads((run / "D2/summary.json").read_text())["rows"]
    selected = {}
    for state in ("N-T", "F-O0"):
        for panel in ("A", "B"):
            group = [r for r in rows if r["state"] == state and r["panel"] == panel and r["repeat"] == 0]
            def size(row):
                prefix = run / "D2" / state / f"candidate-{row['candidate_id']:03d}" / panel / "repeat-0/raw"
                return sum(p.stat().st_size for p in codec.raw_paths(prefix))
            for label, row in (("min_raw_bytes", min(group, key=lambda r: (size(r), r["candidate_id"]))),
                               ("max_raw_bytes", max(group, key=lambda r: (size(r), -r["candidate_id"])))):
                key = (state, row["candidate_id"], panel)
                selected.setdefault(key, []).append(label)
    selected.setdefault(("N-T", 9, "B"), []).append("six_interval_ready_case")
    return [{"state": s, "candidate_id": i, "panel": p, "selection_reasons": why}
            for (s, i, p), why in sorted(selected.items())]


class TimedEnv:
    def __init__(self, config, phases):
        self._phases = phases
        start = time.perf_counter_ns()
        self._env = PlanarPushEnv(config)
        phases["env_create"] += time.perf_counter_ns() - start

    def __getattr__(self, name):
        value = getattr(self._env, name)
        if name not in {"restore", "observe", "snapshot", "step"}:
            return value
        def timed(*args, **kwargs):
            start = time.perf_counter_ns()
            try:
                return value(*args, **kwargs)
            finally:
                self._phases["env_" + name] += time.perf_counter_ns() - start
        return timed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--original-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--passes", type=int, default=3)
    args = ap.parse_args()
    if args.passes != 3:
        raise ValueError("Three complete-result repeat checks are required")
    run, original, out = args.run.resolve(), args.original_root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    groups, rows, bindings = select_groups(run), [], {}
    for p in run.rglob("*"):
        if p.is_file():
            bindings[p.relative_to(run).as_posix()] = file_sha256(p)
    codes = ["scripts/profile_followup_compute.py", "scripts/followup_pilot.py",
             "src/push_align/followup_probe.py", "src/push_align/final_driver_v2.py",
             "src/push_align/final_recording_columns.py", "src/push_align/environment.py"]
    entry = {"schema": "followup-compute-profile-v1", "started_at_utc": datetime.now(timezone.utc).isoformat(),
             "groups": groups, "passes": args.passes, "source_sha256": bindings,
             "code_sha256": {p: file_sha256(ROOT / p) for p in codes},
             "selection": "min/max complete raw-pair bytes in each state/panel, first repeats; union plus N-T candidate9 B",
             "limits": {"wall_s": 1200, "bytes": 512 * 1024**2, "reserve_bytes": 2 * 1024**3},
             "scope": "Same-candidate benchmark only; no new bank, selection, model call or training. env_step includes MuJoCo plus unchanged per-substep environment diagnostics."}
    write_json(out / "entry.json", entry)
    try:
        manifest_path = original / "runs/p5-final-20261017-02/manifest.json"
        design = json.loads((ROOT / "configs/followup_candidate_diagnostic.v1.json").read_text())
        assert file_sha256(manifest_path) == design["original_manifest_sha256"]
        frozen = json.loads(manifest_path.read_text())
        inputs = {s: get_input(run, s)[0:2] for s in ("N-T", "F-O0")}
        for group in groups:
            state, index, panel = (group[k] for k in ("state", "candidate_id", "panel"))
            data, bank = inputs[state]
            source = run / "D2" / state / f"candidate-{index:03d}" / panel / "repeat-0"
            source_receipt = json.loads((source / "raw_receipt.json").read_text())
            snap0, steps0 = codec.read_raw(source / "raw", expected_sha256=source_receipt["files_sha256"])
            original_summary = json.loads((source / "summary.json").read_text())
            reference = None
            for repeat in range(3):
                used = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
                if time.monotonic()-started > 1200 or used > 512*1024**2 or shutil.disk_usage(out).free < 2*1024**3 + 16*1024**2:
                    raise RuntimeError("NOT_RUN_RESOURCE before next benchmark trial")
                trial = out / state / f"candidate-{index:03d}" / panel / f"repeat-{repeat}"
                trial.mkdir(parents=True, exist_ok=False)
                phases, prefix = defaultdict(int), []
                total_start = time.perf_counter_ns()
                probe_start = time.perf_counter_ns()
                try:
                    result = run_followup_probe(data["physical_config"], data["planner_config"], data["snapshot"],
                        data["task_spec"], data["task_status"], bank[index], data["planner_context"].previous_applied_action,
                        frozen["profile"], panel=panel, env_factory=lambda c: TimedEnv(c, phases),
                        on_step=lambda i, step, c: prefix.append(step))
                except Exception:
                    receipt = codec.write_raw(trial / "raw", data["snapshot"], prefix, xz_preset=3)
                    write_json(trial / "raw_receipt.json", receipt)
                    raise
                phases["probe_total"] = time.perf_counter_ns()-probe_start
                phases["probe_other_validation_task_cost"] = phases["probe_total"]-sum(v for k, v in phases.items() if k.startswith("env_"))
                check_start = time.perf_counter_ns()
                original_match = exact_comparison({"snapshot": snap0, "steps": steps0},
                    {"snapshot": data["snapshot"], "steps": list(result.results)})
                phases["new_raw_vs_original_exact"] = time.perf_counter_ns()-check_start
                check_start = time.perf_counter_ns()
                repeat_match = "FIRST_REPEAT" if reference is None else exact_comparison(reference, result)["status"]
                phases["full_probe_repeat_exact"] = time.perf_counter_ns()-check_start
                if reference is None:
                    reference = result
                t = time.perf_counter_ns()
                receipt = codec.write_raw(trial / "raw", data["snapshot"], prefix, xz_preset=3)
                phases["write_raw_total"] = time.perf_counter_ns()-t
                write_json(trial / "raw_receipt.json", receipt)
                t = time.perf_counter_ns()
                loaded_snap, loaded_steps = codec.read_raw(trial / "raw", expected_sha256=receipt["files_sha256"])
                phases["read_raw_total"] = time.perf_counter_ns()-t
                t = time.perf_counter_ns()
                comparison = exact_comparison({"snapshot": data["snapshot"], "steps": prefix},
                    {"snapshot": loaded_snap, "steps": loaded_steps})
                phases["roundtrip_exact"] = time.perf_counter_ns()-t
                assert comparison["status"] == "PASS_EXACT"
                # Preserve and authenticate the complete new raw before any
                # original/repeat verdict can stop the profiling run.
                assert original_match["status"] == "PASS_EXACT", "Benchmark differs from authenticated original raw"
                assert len(result.results) == original_summary["intervals"]
                for key in ("physical_valid", "ready", "ready_prefix", "ready_eligible", "first_ready_interval", "termination"):
                    assert getattr(result, key) == original_summary[key], key
                assert repeat_match in {"FIRST_REPEAT", "PASS_EXACT"}, "Full probe repeat changed"
                t = time.perf_counter_ns()
                with (trial / "diagnostic.npz").open("xb") as target:
                    np.savez_compressed(target, states=result.states, boundary_times=result.boundary_times,
                        nominal_actions=result.nominal_actions, requested_actions=result.requested_actions,
                        applied_actions=result.applied_actions, supervisor_overrides=result.supervisor_overrides,
                        hold_s=result.diagnostic_hold_s, qualifies=result.diagnostic_qualifies)
                phases["diagnostic_npz"] = time.perf_counter_ns()-t
                phases["trial_total"] = time.perf_counter_ns()-total_start
                pipeline_keys = ("probe_total", "new_raw_vs_original_exact", "full_probe_repeat_exact", "write_raw_total", "read_raw_total", "roundtrip_exact", "diagnostic_npz")
                phases["trial_other_receipt_checks"] = phases["trial_total"]-sum(phases[k] for k in pipeline_keys)
                assert phases["probe_other_validation_task_cost"] >= 0 and phases["trial_other_receipt_checks"] >= 0
                row = {**group, "repeat": repeat, "intervals": len(prefix), "phases_ns": dict(phases),
                    "original_exact": original_match["status"], "roundtrip_exact": comparison["status"],
                    "full_probe_repeat_exact": repeat_match, "raw_receipt_sha256": file_sha256(trial / "raw_receipt.json"),
                    "raw_bytes": sum(p.stat().st_size for p in codec.raw_paths(trial / "raw"))}
                write_json(trial / "timing.json", row)
                rows.append(row)
            print(f"{state} candidate{index:03d} {panel}: compute/raw/full-result exact x3", flush=True)
        for rel, sha in bindings.items():
            assert file_sha256(run / rel) == sha, "Original profile input changed: " + rel
        for rel, sha in entry["code_sha256"].items():
            assert file_sha256(ROOT / rel) == sha
        summary = {"schema": "followup-compute-profile-summary-v1", "status": "PASS_TIMED_COMPUTE_AND_FULL_FIDELITY",
            "rows": rows, "groups": len(groups), "trajectories": len(rows), "actual_intervals": sum(r["intervals"] for r in rows),
            "wall_seconds": time.monotonic()-started, "original_files_unchanged": len(bindings),
            "model_calls": 0, "training_updates": 0,
            "peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "limitations": ["Same nine saved candidate groups; no claim of universal timing bound.",
                "env_step includes environment-generated contact diagnostics as well as engine stepping.",
                "Additional original comparison belongs to benchmark overhead, not original D2 driver cost.",
                "No cold-cache or power-loss guarantee. Original pipeline durability unchanged."]}
        write_json(out / "summary.json", summary)
        print(json.dumps({k: summary[k] for k in ("status", "groups", "trajectories", "actual_intervals", "wall_seconds")}), flush=True)
    except Exception as error:
        write_json(out / "incomplete.json", {"status": "STOP_INCOMPLETE_NO_AUTOMATIC_RETRY", "rows": rows,
            "error": type(error).__name__ + ": " + str(error), "traceback": traceback.format_exc(), "wall_seconds": time.monotonic()-started})
        raise


if __name__ == "__main__":
    main()
