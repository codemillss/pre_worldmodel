#!/usr/bin/env python3
"""Charge this bounded prototype once; do not invent a new executor unit rate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

import reassess_nt_remaining as prior_tools

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg_path = ROOT / "configs/exact_pair_prototype.v2.json"
    cfg = prior_tools.load(cfg_path)
    prior_path = ROOT / cfg["prior_ledger_path"]
    prior_tools.need(prior_tools.sha(prior_path) == cfg["prior_ledger_sha256"], "Prior ledger changed")
    prior = prior_tools.load(prior_path)
    run = ROOT.parents[1] / "runs/followup-exact-pair-20261028-01"
    summary = prior_tools.load(run / "summary.json")
    process = prior_tools.load(run / "process_receipt.json")
    prior_tools.need(summary["status"] == "COMPLETE_SAVED_EXACT_PAIR_PROTOTYPE" and process["returncode"] == 0,
                     "Incomplete saved prototype")
    prior_tools.need(summary["config_sha256"] == prior_tools.sha(cfg_path), "Profile config differs")
    _, _, bindings, work = prior_tools.authenticate_frozen()
    # Whole active task window includes benchmark/tests/audit/preparation once.
    # The external child process is a subset, never an additional elapsed scope.
    assessed_utc = datetime.now(timezone.utc)
    task_seconds = (assessed_utc - datetime.fromisoformat(cfg["task_clock_start_utc"])).total_seconds()
    outer_seconds = process["outer_seconds"]
    prior_tools.need(task_seconds >= outer_seconds, "Child clock exceeds task window")
    additional = task_seconds - outer_seconds + cfg["pre_clock_allowance_seconds"]
    new_files = [p for p in ROOT.rglob("*") if p.is_file()
                 and (p.name.startswith("exact_pair") or p.name == "profile_exact_pair.py"
                      or p.name == "reassess_exact_pair.py" or p.parent.name == "20261028-exact-pair")]
    preserved_bytes = prior["preserved_local_experiment_bytes_at_assessment"] + prior_tools.logical_bytes(run)
    preserved_bytes += sum(p.stat().st_size for p in new_files)
    # Keep the previous observed maxima. A two-case prototype does not calibrate
    # actual physics + raw codec + complete per-interval production processing.
    rates = prior["profile_only_raise_scenario"]["unit_rates"]
    result = prior_tools.assess(prior_seconds=cfg["prior_cumulative_seconds"], process_seconds=outer_seconds,
        elapsed_postwork_seconds=additional, work=work, rates=rates, preserved_bytes=preserved_bytes,
        free_bytes=shutil.disk_usage(run).free)
    result.update(schema="exact-pair-resource-review-v1", assessed_utc=assessed_utc.isoformat(),
        preserved_local_experiment_bytes_at_assessment=preserved_bytes,
        bindings={"profile_summary_sha256": prior_tools.sha(run / "summary.json"),
                  "profile_outer_process_sha256": prior_tools.sha(run / "process_receipt.json"),
                  "config_sha256": prior_tools.sha(cfg_path), "prior_ledger_sha256": prior_tools.sha(prior_path),
                  "frozen_run_summary_sha256": bindings["full_run_summary_sha256"],
                  "assessment_tool_sha256": prior_tools.sha(Path(__file__))},
        task_clock={"start_utc": cfg["task_clock_start_utc"], "elapsed_task_seconds": task_seconds,
                    "pre_clock_allowance_seconds": cfg["pre_clock_allowance_seconds"],
                    "allowance_is_conservative_reserve_not_measured_runtime": True,
                    "outer_process_is_included_once": True},
        prototype_executor_projection={"status": "NOT_CALIBRATED_NOT_RUN",
            "reason": "Complete bundle and one StepResult comparison were timed. Full raw persistence, actual physics and production end-to-end execution have not been calibrated with this prototype."},
        scientific_state={"completed": 70, "not_executed": 698, "new_actual_trajectories": 0,
                          "model_training_updates": 0, "model_cost_candidates_success_contract_changed": False},
        scope="Forecast below remains the frozen implementation at prior maximum rates, not a speedup forecast for a new executor.")
    frozen = result["frozen_rate_scenario"]
    result["zero_complete_bundle_hypothetical_minutes"] = (
        frozen["projected_cumulative_seconds"] - frozen["future_seconds"]["first_repeat_complete_probe"]) / 60
    prior_tools.need(result["time_accounting"]["cumulative_spent_seconds"] ==
                     cfg["prior_cumulative_seconds"] + task_seconds + cfg["pre_clock_allowance_seconds"],
                     "Task/benchmark elapsed double counted")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": frozen["status"], "spent_minutes": result["time_accounting"]["cumulative_spent_seconds"] / 60,
                      "frozen_executor_projected_minutes": frozen["projected_cumulative_minutes"],
                      "prototype_executor": result["prototype_executor_projection"]["status"]}))


if __name__ == "__main__":
    main()
