#!/usr/bin/env python3
"""Fresh N-T 128 x A/B x3 assay, with exact restoration and live whole-workflow caps.

Reuses the unchanged preflight codecs/checkers. All seven predictions are sealed
before any new physics; first-repeat A teacher rows are evaluated AFTER actuals.
The caller must retain the process wall receipt outside this process as well.
"""
from __future__ import annotations

import time
PROGRAM_START = time.monotonic()

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
import followup_nt_executor as nt
from push_align.followup_probe import _physical_reasons
from push_align.contact_geometry import check_contact_geometry
from push_align.followup_nt_live_budget import (LiveBudget, ObservedRates, PrefixDeadlineRecorder,
    ResourceLimits, ResourceStop, expected_trial_keys, remaining_work)

MODELS = nt.MODELS
CODE_PATHS = tuple(dict.fromkeys(("scripts/followup_nt_full.py", "src/push_align/followup_nt_live_budget.py") + nt.CODE_PATHS))
require, compare_pass = nt.require, nt.compare_pass


def append_json(path, value):
    """Durable append; a partial last line after OS failure is never a result row."""
    with path.open("a", encoding="utf-8") as target:
        target.write(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
        target.flush()
        os.fsync(target.fileno())


def validate_config(config):
    require(config["schema"] == "followup-nt-fullbank-design-v1", "Reviewed full-bank design required")
    require(config["state"] == "N-T" and config["candidate_count"] == 128 and
            config["candidate_ids"] == list(range(128)), "Original N-T bank0..127 only")
    require(config["models"] == list(MODELS) and config["panels"] == ["A", "B"] and config["repeats"] == 3,
            "Seven frozen models / A,B /3 repeats required")
    require(config["teacher_actual_rows_max"] == 640 and not config["scientific_row_reuse"] and
            not config["conditional_extension"], "No row reuse, extension or repeated teacher labels")
    default = ResourceLimits()
    limits = config["limits"]
    checks = {"wall_time_cap_s": default.wall_cap_seconds, "incremental_storage_cap_bytes": default.storage_cap_bytes,
        "free_space_reserve_bytes": default.free_reserve_bytes, "safety_factor": default.safety_factor,
        "final_report_reserve_seconds": 600., "sealing_reserve_seconds": default.sealing_reserve_seconds,
        "sealing_reserve_bytes": default.sealing_reserve_bytes,
        "future_fixed_seconds_reserve": default.fixed_future_seconds,
        "prior_worst_seconds_per_interval": ObservedRates().seconds_per_interval,
        "prior_worst_bytes_per_interval": ObservedRates().bytes_per_interval,
        "source_reference_seconds_per_trial": ObservedRates().source_comparison_seconds,
        "complete_probe_seconds_per_group": ObservedRates().complete_probe_seconds,
        "complete_probe_bytes_per_group": ObservedRates().complete_probe_bytes}
    require(all(limits[key] == expected for key, expected in checks.items()), "Reviewed caps/reserves/rate floors changed")
    require(abs(limits["loop_seconds_per_trial"] - ObservedRates().loop_seconds_per_trial) < 1e-18, "Loop floor changed")
    prior = config["base_reviewed_prior_wall_seconds"] + sum(item["wall_seconds"] for item in config["recovery_attempts"])
    prior += config["completed_preflight"]["charged_process_wall_seconds"]
    require(prior == config["cumulative_prior_wall_seconds"] == default.prior_seconds, "Prior cumulative wall differs")
    require(config["existing_run_bytes_before_full"] == default.prior_bytes, "Prior complete preserved bytes differ")


def fresh_teacher(folder, output, data, bank, profile, predictors, guards, rows):
    """One unique candidate/interval row per fresh A first repeat, never timing clones."""
    folder.mkdir()
    arrays = {key: [] for key in ("start", "actual", "applied", "candidate_interval", "actual_transition_valid", "actual_observation_valid", "actual_contact_geometry_valid", "actual_dt_match", "actual_endpoint_finite", "actual_physical_valid")}
    first_rows = [row for row in rows if row["panel"] == "A" and row["repeat"] == 0]
    require(len(first_rows) == 128, "Fresh teacher requires all128 first-repeat A prefixes")
    for row in first_rows:
        index = row["candidate_id"]
        trial = output / "actual/N-T" / f"candidate-{index:03d}/A/repeat-0"
        receipt = json.loads((trial / "raw_receipt.json").read_text())
        snapshot, steps = nt.codec.read_raw(trial / "raw", expected_sha256=receipt["files_sha256"])
        compare_pass(snapshot, data["snapshot"], "Fresh teacher initial snapshot differs")
        current = data["observation"].state
        for interval, step in enumerate(steps):
            require(np.isfinite(current).all(), "Fresh teacher input nonfinite; retain raw and stop rather than drop rows")
            arrays["start"].append(current.copy())
            arrays["actual"].append(step.observation.state.copy())
            arrays["applied"].append(step.applied_action.copy())
            arrays["candidate_interval"].append((index, interval))
            arrays["actual_transition_valid"].append(bool(step.transition_valid))
            arrays["actual_observation_valid"].append(bool(step.observation.observation_valid))
            geometry = check_contact_geometry(step.diagnostics["samples"],
                profile["physics_suite"]["actual_contact_penetration_max_m"])
            arrays["actual_contact_geometry_valid"].append(bool(geometry["pass"]))
            arrays["actual_dt_match"].append(math.isfinite(step.actual_dt) and
                abs(step.actual_dt - data["physical_config"]["physics"]["action_dt_s"]) <=
                    data["physical_config"]["validity"]["dt_match_tolerance_s"])
            arrays["actual_endpoint_finite"].append(bool(np.isfinite(step.observation.state).all()))
            reasons = _physical_reasons(step, bank[index, interval],
                SimpleNamespace(action_dt=data["physical_config"]["physics"]["action_dt_s"]),
                data["physical_config"]["validity"]["dt_match_tolerance_s"], geometry)
            arrays["actual_physical_valid"].append(not reasons)
            current = step.observation.state
    arrays = {key: np.asarray(value, np.bool_ if key.startswith("actual_") else np.int64 if key == "candidate_interval" else np.float64)
              for key, value in arrays.items()}
    count = len(arrays["start"])
    require(0 < count <= 640 and len(set(map(tuple, arrays["candidate_interval"]))) == count,
            "Fresh teacher rows must be unique candidate/interval keys, <=640")
    source = nt.npz_roundtrip(folder / "source-fresh-A-first-repeat.npz", arrays)
    metadata = {"schema": "followup-nt-fresh-teacher-v1", "rows": count, "source": source,
        "scope": "Fresh full-run A first-repeat recorded transitions; no repeated timing labels; masks retained for all prefixes. actual_physical_valid uses unchanged probe contact/transition/observation/finite/action/dt checks.",
        "all_actual_futures_finished_before_teacher": True, "models": {}}
    for name in MODELS:
        phase = time.perf_counter_ns()
        with nt.model_planning_scope(guards):
            prediction = nt.rollout(arrays["start"], arrays["applied"][:, None, :], predictors[name], data["planner_config"])
        infer_ns = time.perf_counter_ns() - phase
        phase = time.perf_counter_ns()
        receipt = nt.npz_roundtrip(folder / f"one-step-{name}.npz", {
            "predicted": prediction.states[:, 1], "raw_deltas": prediction.raw_deltas[:, 0],
            "valid": prediction.valid, "reasons": np.asarray(prediction.reasons), "failure_steps": prediction.failure_steps})
        metadata["models"][name] = {"rows": count, "rollout_ns": infer_ns,
            "durable_save_readback_exact_ns": time.perf_counter_ns() - phase, "receipt": receipt}
    require(all(value == 0 for value in guards.values()), "Future engine call during fresh teacher evaluation")
    nt.write_json(folder / "summary.json", metadata)
    return metadata


def _terminate(signum, frame):
    raise KeyboardInterrupt("Received termination signal; seal returned raw prefix and stop")


def main():
    # SIGINT already raises KeyboardInterrupt. Convert SIGTERM to the same
    # cooperative preservation path rather than terminating between raw writes.
    signal.signal(signal.SIGTERM, _terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/followup_nt_full.v1.json")
    args = parser.parse_args()
    original, source_run, output, config_path = (path.resolve() for path in
        (args.original_root, args.source_run, args.output, args.config))
    require(not output.exists() and output != source_run and source_run not in output.parents and
            output not in original.parents, "Exclusive output separate from preserved source required")
    output.mkdir(parents=True, exist_ok=False)
    rows, completed, stages = [], set(), {}
    guards = dict.fromkeys(nt.ENGINE_GUARDS, 0)
    source_bindings, original_bindings, model_bindings, receipt_bindings = {}, {}, {}, {}
    models, teacher_metadata, prediction_metadata = {}, None, None
    budget = LiveBudget()
    total_intervals = total_substeps = 0
    status, failure, active_key = "STOP_INCOMPLETE", None, None
    source_verified = False
    entry = {"schema": "followup-nt-full-entry-v1", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(), "full_execution_authorized": True, "state": "N-T", "candidate_ids": list(range(128)),
        "panels": ["A", "B"], "repeats": 3, "models": list(MODELS), "expected_trajectories": 768,
        "maximum_intervals": 4992, "training_updates": 0, "scientific_row_reuse": False, "conditional_extension": False,
        "durability": "Exclusive fsynced raw per completed/error prefix; cooperative action-boundary deadline with60s sealing+600s final audit/report reserves, not universal OS/hardware guarantee."}
    last_size = 0

    def resource_gate(stage, *, reject=True):
        nonlocal last_size
        last_size = nt.logical_bytes(output)
        decision = budget.evaluate(completed, process_seconds=time.monotonic() - PROGRAM_START,
            output_bytes=last_size, free_bytes=shutil.disk_usage(output).free, stage=stage)
        append_json(output / "resource_decisions.jsonl", decision)
        if reject and decision["decision"] != "PASS_RESOURCE":
            raise ResourceStop(stage + ": " + ",".join(decision["failures"]))
        return decision

    def step_check():
        budget.assert_step_window(process_seconds=time.monotonic() - PROGRAM_START,
            output_bytes=last_size, free_bytes=shutil.disk_usage(output).free)

    try:
        phase = time.perf_counter_ns()
        config = json.loads(config_path.read_text())
        validate_config(config)
        budget = LiveBudget(replace(ResourceLimits(), report_reserve_seconds=config["limits"]["final_report_reserve_seconds"]))
        entry["config_sha256"] = nt.file_sha256(config_path)
        entry["code_sha256"] = {path: nt.file_sha256(ROOT / path) for path in CODE_PATHS}
        limits = config["limits"]
        entry["resource_limits"] = asdict(budget.limits)
        entry["initial_budget"] = resource_gate("initial_before_identity_work")
        for path_key, sha_key in (("base_preflight_config_path", "base_preflight_config_sha256"),
                                  ("scoped_resource_budget_path", "scoped_resource_budget_sha256"),
                                  ("resource_ledger_path", "resource_ledger_sha256")):
            path = ROOT / config[path_key]
            require(nt.file_sha256(path) == config[sha_key], "Authenticated resource/config record changed: " + path_key)
            receipt_bindings[config[path_key]] = config[sha_key]
        ledger = json.loads((ROOT / config["resource_ledger_path"]).read_text())
        require(ledger["budget"]["already_spent_seconds"] == config["base_reviewed_prior_wall_seconds"], "Base ledger wall differs")
        preflight_config = json.loads((ROOT / config["base_preflight_config_path"]).read_text())
        nt.validate_config(preflight_config)
        require(preflight_config["expected_model_normalizer_sha256"] == config["expected_model_normalizer_sha256"], "Full/preflight model maps differ")
        preserved_receipts = {}
        for record in config["recovery_attempts"]:
            path = (original / record["failure_record_path"]).resolve()
            require(path.is_relative_to(original) and nt.file_sha256(path) == record["failure_record_sha256"], "Failed preflight receipt differs")
            payload = json.loads(path.read_text())
            require(payload["wall_seconds"] == record["wall_seconds"] and payload["actual_physics_intervals_returned"] == 0,
                    "Preserved recovery wall/scope differs")
            preserved_receipts[record["failure_record_path"]] = record["failure_record_sha256"]
        prior = config["completed_preflight"]
        for path_key, sha_key in (("process_wall_receipt_path", "process_wall_receipt_sha256"),
                                  ("summary_path", "summary_sha256"), ("source_entry_path", "source_entry_sha256")):
            path = (original / prior[path_key]).resolve()
            require(path.is_relative_to(original) and nt.file_sha256(path) == prior[sha_key], "Completed preflight receipt changed")
            preserved_receipts[prior[path_key]] = prior[sha_key]
        wall_receipt = json.loads((original / prior["process_wall_receipt_path"]).read_text())
        require(wall_receipt["exit_code"] == 0 and wall_receipt["process_wall_seconds"] == prior["charged_process_wall_seconds"], "Completed process wall differs")
        prior_summary = json.loads((original / prior["summary_path"]).read_text())
        require(prior_summary["status"] == "PASS_NT_EXECUTOR_PREFLIGHT" and prior_summary["actual_trajectories"] == 18,
                "Source preflight did not pass all18 checks")
        resource_record = json.loads((ROOT / config["scoped_resource_budget_path"]).read_text())
        require(resource_record["cumulative_spent_wall_seconds"] == config["cumulative_prior_wall_seconds"] and
                resource_record["storage_from_unchanged_v1"]["prior_run_bytes"] + prior["preserved_run_bytes"] == config["existing_run_bytes_before_full"],
                "Authenticated scoped ledger prior differs")
        source_bindings = nt.inventory(source_run)
        public_source_manifest = ROOT / "artifacts/followup/20261022-d1-d2/manifest.json"
        require(nt.file_sha256(public_source_manifest) == config["source_run_D1_D2_inventory_sha256"], "D1/D2 public inventory changed")
        for relative, key in (("D1/N-T/restoration.json", "source_D1_report_sha256"),
                              ("D2/entry.json", "source_D2_entry_sha256"), ("D2/summary.json", "source_D2_summary_sha256")):
            require(nt.file_sha256(source_run / relative) == config[key], "Source D1/D2 identity changed: " + relative)
        original_model_bindings = json.loads((source_run / "D2/entry.json").read_text())["model_and_normalizer_sha256"]
        nt.verify_model_bindings(config["expected_model_normalizer_sha256"], original_model_bindings)
        manifest_path = original / "runs/p5-final-20261017-02/manifest.json"
        require(nt.file_sha256(manifest_path) == config["original_manifest_sha256"], "Original manifest changed")
        frozen = json.loads(manifest_path.read_text())
        require(nt.file_sha256(ROOT / "configs/followup_candidate_diagnostic.v1.json") == config["original_design_sha256"], "Frozen diagnostic design changed")
        data, bank, labels, d1 = nt.get_input(source_run, "N-T")
        capture = next(item for item in d1["captures"] if item["tick"] == 78)
        require(d1["design_sha256"] == config["original_design_sha256"] and
                nt.typed_digest(data) == capture["input_typed_sha256"] == config["expected_input_typed_sha256"], "Complete input identity differs")
        require(nt.file_sha256(source_run / "D1/N-T" / capture["bank"]) == config["expected_bank_file_sha256"] and
                nt.array_hash(bank) == capture["candidate_hash"] == config["expected_candidate_hash"], "Original128 bank differs")
        original_bindings = dict(d1["input_sha256"])
        for relative, digest in original_bindings.items():
            require(nt.file_sha256(original / relative) == digest, "Original input changed: " + relative)
        for relative, digest in frozen["runtime_source_sha256"].items():
            require(nt.file_sha256(original / relative) == digest, "Original runtime changed: " + relative)
            if relative.startswith("src/"):
                require(nt.file_sha256(ROOT / relative) == digest, "Public frozen runtime changed: " + relative)
        adapter = nt.GeometryProposalAdapter(data["physical_config"], data["geometry_parameters"], data["transition_parameters"])
        proposal, generated, keys = nt.regenerate(data, data["observation"], data["task_status"], data["planner_context"],
                                                data["geometry_context"], adapter, data["tick"], guards)
        candidate_compare = compare_pass(bank, generated.actions, "Regenerated128 bank differs")
        compare_pass(labels, np.asarray(generated.sources), "Candidate source/order differs")
        require(keys == data["rng_keys"], "Candidate RNG keys differ")
        bank_receipt = nt.npz_roundtrip(output / "bank.npz", {"actions": bank, "sources": labels, "original_ids": np.arange(128, dtype=np.int64)})
        full_input_receipt = nt.write_full_trace(output / "input", tick=78, captured_input=data)
        reread_input, _, _ = nt.read_full_trace(full_input_receipt)
        input_compare = compare_pass(data, reread_input, "Input exact roundtrip differs")
        stages["bindings_input_candidate_regeneration_ns"] = time.perf_counter_ns() - phase
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        phase = time.perf_counter_ns()
        models, model_bindings = nt.load_models()
        nt.verify_model_bindings(model_bindings, original_model_bindings)
        require(sorted(models) == sorted(MODELS), "Seven model identities differ")
        provenance_path = ROOT / "artifacts/provenance/publication_files.json"
        provenance = {record["public_path"]: record for record in json.loads(provenance_path.read_text())["records"]}
        for relative, digest in model_bindings.items():
            record = provenance[relative]
            source = record["source_path"]
            require(record["transformation"] == "byte_identical" and digest == record["public_sha256"] == record["source_sha256"] ==
                    frozen["runtime_source_sha256"][source] == nt.file_sha256(original / source), "Model original provenance differs: " + relative)
        stages["strict_models_load_provenance_ns"] = time.perf_counter_ns() - phase
        entry.update(source_run_sha256=source_bindings, original_sha256=original_bindings,
            authenticated_prior_receipts=preserved_receipts, reviewed_resource_sha256=receipt_bindings,
            model_and_normalizer_sha256=model_bindings, public_provenance_sha256=nt.file_sha256(provenance_path),
            full_input_typed_sha256=nt.typed_digest(data), bank_hash=nt.array_hash(bank), rng_keys=keys,
            unique_candidate_sequences=len({nt.array_hash(row) for row in bank}), input_roundtrip=input_compare,
            regenerated_bank_exact=candidate_compare, bank_receipt=bank_receipt, full_input_receipt=full_input_receipt,
            cumulative_prior_wall_seconds=config["cumulative_prior_wall_seconds"], prior_preserved_bytes=config["existing_run_bytes_before_full"],
            runtime={"python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "mujoco": __import__("mujoco").__version__,
                     "device": "CPU", "torch_threads": 1, "deterministic_algorithms": True, "physical_state_dtype": "float64"})
        nt.write_json(output / "entry.json", entry)
        phase = time.perf_counter_ns()
        prediction_metadata = nt.forecasts(output / "forecasts", data, bank, models, guards)
        stages["full128_forecasts_cost_selection_durable_exact_ns"] = time.perf_counter_ns() - phase
        forecast_seal = {"schema": "followup-nt-full-prospective-forecast-seal-v1", "before_any_new_actual_future": True,
            "forecasts": {path.relative_to(output).as_posix(): nt.file_sha256(path) for path in sorted((output / "forecasts").iterdir()) if path.is_file()},
            "prediction_calls_before_actual": {name: {"calls": model.calls, "rows": model.rows} for name, model in models.items()},
            "future_engine_calls_during_prediction": dict(guards), "wall_seconds_before_any_actual": time.monotonic() - PROGRAM_START,
            "teacher_order": "Fresh A teacher inputs and seven one-step evaluations AFTER768 actual rows; no duplicate benchmark rows",
            "stages_ns": dict(stages), "timestamp_utc": datetime.now(timezone.utc).isoformat()}
        nt.write_json(output / "prospective_forecast_seal.json", forecast_seal)
        seal_sha = nt.file_sha256(output / "prospective_forecast_seal.json")
        decision = resource_gate("entry_complete_before_any_new_physics")
        nt.write_json(output / "entry_gate.json", {"status": "PASS_ENTRY_IDENTITIES_FORECAST_SEAL_FULL_RESOURCE", "resource": decision,
            "prospective_seal_sha256": seal_sha, "elapsed_seconds": time.monotonic() - PROGRAM_START})
        print(json.dumps({"status": "PASS_ENTRY", "pid": os.getpid(), "expected_fresh_trials": 768,
            "projected_cumulative_seconds": decision["projected_cumulative_seconds"], "input_hash": entry["full_input_typed_sha256"], "bank_hash": entry["bank_hash"]}), flush=True)
        trial_phase = time.perf_counter_ns()
        for index in range(128):
            for panel in ("A", "B"):
                reference = None
                for repeat in range(3):
                    loop_start = time.perf_counter_ns()
                    resource_gate(f"before_trial:{index}:{panel}:{repeat}")
                    active_key = (index, panel, repeat)
                    trial = output / "actual/N-T" / f"candidate-{index:03d}" / panel / f"repeat-{repeat}"
                    trial.mkdir(parents=True, exist_ok=False)
                    nt.write_json(trial / "entry.json", {"candidate_id": index, "panel": panel, "repeat": repeat,
                        "candidate_hash": nt.array_hash(bank[index]), "full_input_typed_sha256": entry["full_input_typed_sha256"], "prospective_seal_sha256": seal_sha})
                    loop_before_ns = time.perf_counter_ns() - loop_start
                    prefix, timings, result, execution_error = [], {}, None, None
                    trial_start = time.perf_counter_ns()
                    phase = time.perf_counter_ns()
                    try:
                        result = nt.run_followup_probe(data["physical_config"], data["planner_config"], data["snapshot"], data["task_spec"],
                            data["task_status"], bank[index], data["planner_context"].previous_applied_action, frozen["profile"], panel=panel,
                            on_step=PrefixDeadlineRecorder(prefix, step_check))
                    except (Exception, KeyboardInterrupt) as error:
                        execution_error = error
                    timings["physical_probe_including_unchanged_validation_ns"] = time.perf_counter_ns() - phase
                    total_intervals += len(prefix)
                    total_substeps += sum(len(step.substep_states) for step in prefix)
                    # Always durable before any source/repeat/full-field comparator.
                    receipt, roundtrip = nt.seal_and_validate_prefix(trial, data["snapshot"], prefix, timings, validate=True)
                    if execution_error is not None:
                        nt.write_json(trial / "incomplete.json", {"status": "STOP_RESOURCE" if isinstance(execution_error, ResourceStop) else "STOP_INCOMPLETE",
                            "type": type(execution_error).__name__, "message": str(execution_error), "returned_intervals": len(prefix), "raw_roundtrip": roundtrip,
                            "timings_ns": timings, "candidate_id": index, "panel": panel, "repeat": repeat})
                        raise execution_error
                    phase = time.perf_counter_ns()
                    repeat_compare = None if reference is None else compare_pass(reference, result, "UNSTABLE_REPLAY complete FollowupProbe differs")
                    timings["full_probe_repeat_exact_ns"] = time.perf_counter_ns() - phase
                    phase = time.perf_counter_ns()
                    original_compare = None
                    if index < 16:
                        old = source_run / "D2/N-T" / f"candidate-{index:03d}" / panel / "repeat-0"
                        old_receipt = json.loads((old / "raw_receipt.json").read_text())
                        old_snapshot, old_steps = nt.codec.read_raw(old / "raw", expected_sha256=old_receipt["files_sha256"])
                        original_compare = compare_pass({"snapshot": old_snapshot, "steps": old_steps},
                            {"snapshot": data["snapshot"], "steps": prefix}, "Fresh actual differs from source D2")
                    timings["source_D2_raw_read_and_exact_ns"] = time.perf_counter_ns() - phase
                    phase = time.perf_counter_ns()
                    full_probe_receipt = full_probe_compare = None
                    if repeat == 0:
                        tree = nt.complete_probe_tree(result)
                        full_probe_receipt = nt.write_full_trace(trial / "complete_probe", tick=78, captured_input=tree)
                        loaded_tree, _, _ = nt.read_full_trace(full_probe_receipt)
                        compare_pass(tree, loaded_tree, "Full probe tree roundtrip differs")
                        full_probe_compare = compare_pass(result, nt.reconstruct_probe(loaded_tree), "Reconstructed whole FollowupProbe differs")
                        nt.write_json(trial / "complete_probe_receipt.json", full_probe_receipt)
                        reference = result
                    timings["first_repeat_complete_probe_durable_roundtrip_ns"] = time.perf_counter_ns() - phase
                    phase = time.perf_counter_ns()
                    diagnostic_receipt = nt.npz_roundtrip(trial / "diagnostic.npz", {"states": result.states, "boundary_times": result.boundary_times,
                        "nominal_actions": result.nominal_actions, "requested_actions": result.requested_actions, "applied_actions": result.applied_actions,
                        "supervisor_overrides": result.supervisor_overrides, "hold_s": result.diagnostic_hold_s, "qualifies": result.diagnostic_qualifies})
                    timings["diagnostic_durable_roundtrip_ns"] = time.perf_counter_ns() - phase
                    phase = time.perf_counter_ns()
                    whole_digest = nt.typed_digest(result)
                    timings["whole_probe_typed_digest_ns"] = time.perf_counter_ns() - phase
                    phase = time.perf_counter_ns()
                    trial_bytes = nt.logical_bytes(trial)
                    timings["trial_output_traversal_ns"] = time.perf_counter_ns() - phase
                    full_bytes = (sum(full_probe_receipt["bytes"].values()) + (trial / "complete_probe_receipt.json").stat().st_size) if full_probe_receipt else 0
                    row = {"state": "N-T", "candidate_id": index, "candidate_hash": nt.array_hash(bank[index]), "panel": panel, "repeat": repeat,
                        "intervals": len(prefix), "physical_valid": result.physical_valid, "ready": result.ready, "ready_prefix": result.ready_prefix,
                        "ready_eligible": result.ready_eligible, "first_ready_interval": result.first_ready_interval,
                        "horizon_completed": result.horizon_completed, "termination": result.termination, "reasons": list(result.reasons),
                        "cost": result.cost if result.cost is not None and math.isfinite(result.cost) else None,
                        "cost_finite": result.cost is not None and math.isfinite(result.cost),
                        "cost_components": {key: value if math.isfinite(value) else None for key, value in result.cost_components.items()},
                        "raw_roundtrip": roundtrip, "full_probe_repeat_exact": repeat_compare, "full_probe_roundtrip": full_probe_compare,
                        "source_D2_raw_exact": original_compare, "source_D2_scope": "All original IDs0..15, both panels, everyrepeat; IDs16..127 have no original-D2 future",
                        "diagnostic_receipt": diagnostic_receipt, "timings_ns": timings, "raw_bytes": sum(path.stat().st_size for path in nt.codec.raw_paths(trial / "raw")),
                        "trial_bytes_before_summary": trial_bytes, "first_repeat_complete_probe_bytes": full_bytes,
                        "whole_probe_typed_sha256": whole_digest, "initial_snapshot_exact": result.initial_snapshot_exact}
                    phase = time.perf_counter_ns()
                    nt.write_json(trial / "summary.json", row)
                    timings["summary_write_fsync_ns"] = time.perf_counter_ns() - phase
                    timings["trial_total_through_summary_ns"] = time.perf_counter_ns() - trial_start
                    nt.write_json(trial / "timing_final.json", {"timings_ns": timings, "scope": "All measured work through summary; timing receipt follows measurement."})
                    rows.append(row)
                    completed.add(active_key)
                    source_s = timings["source_D2_raw_read_and_exact_ns"] / 1e9 if index < 16 else 0.
                    complete_s = timings["first_repeat_complete_probe_durable_roundtrip_ns"] / 1e9 if repeat == 0 else 0.
                    baseline_s = max(0., timings["trial_total_through_summary_ns"] / 1e9 - source_s - complete_s)
                    # Whole-remaining check is itself tracked as loop overhead and
                    # folded into the next guard; no measured rate can decrease.
                    after_start = time.perf_counter_ns()
                    trial_bytes_now = nt.logical_bytes(trial)
                    budget.rates.observe(intervals=len(prefix), baseline_seconds=baseline_s,
                        baseline_bytes=max(0, trial_bytes_now - full_bytes), source_seconds=source_s,
                        complete_seconds=complete_s, complete_bytes=full_bytes,
                        loop_seconds=loop_before_ns / 1e9)
                    decision = resource_gate(f"after_sealed_trial:{index}:{panel}:{repeat}", reject=False)
                    after_s = (time.perf_counter_ns() - after_start) / 1e9
                    budget.rates.loop_seconds_per_trial = max(budget.rates.loop_seconds_per_trial, loop_before_ns / 1e9 + after_s)
                    if decision["decision"] != "PASS_RESOURCE":
                        raise ResourceStop("Whole remaining workflow no longer fits after sealed trial")
                    active_key = None
                print(json.dumps({"candidate_id": index, "panel": panel, "sealed_rows": len(rows),
                    "actual_intervals": total_intervals, "cumulative_seconds": budget.limits.prior_seconds + time.monotonic() - PROGRAM_START,
                    "projected_cumulative_seconds": decision["projected_cumulative_seconds"]}), flush=True)
            append_json(output / "progress.jsonl", {"candidate_id": index, "sealed_rows": len(rows), "actual_intervals": total_intervals,
                "actual_substeps": total_substeps, "remaining": asdict(remaining_work(completed)), "wall_seconds": time.monotonic() - PROGRAM_START,
                "last_trial_summary_sha256": nt.file_sha256(output / "actual/N-T" / f"candidate-{index:03d}/B/repeat-2/summary.json")})
        stages["actual_full_trials_and_all_verification_ns"] = time.perf_counter_ns() - trial_phase
        resource_gate("before_fresh_teacher_and_final_audit")
        phase = time.perf_counter_ns()
        teacher_metadata = fresh_teacher(output / "teacher", output, data, bank, frozen["profile"], models, guards, rows)
        stages["fresh_teacher_source_authentication_and_seven_one_step_npz_ns"] = time.perf_counter_ns() - phase
        require(len(rows) == 768 and len(completed) == 768, "Full requested Cartesian rows missing")
        require(sum(row["source_D2_raw_exact"] is not None for row in rows) == 96, "All96 applicable original-D2 comparisons required")
        require(sum(row["full_probe_roundtrip"] is not None for row in rows) == 256, "All256 complete-field durable reconstructions required")
        require(sum(row["full_probe_repeat_exact"] is not None for row in rows) == 512, "All512 full-result repeated comparisons required")
        status = "COMPLETE_NT_FULLBANK_EXACT_768"
    except ResourceStop as error:
        status = "STOP_RESOURCE"
        failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    except (Exception, KeyboardInterrupt) as error:
        status = "STOP_INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "STOP_INCOMPLETE"
        failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    # The original and new evidence is audited even for a bounded resource stop.
    phase = time.perf_counter_ns()
    try:
        if source_bindings:
            require(nt.inventory(source_run) == source_bindings, "Preserved source run changed")
        if "forecast_seal" in locals():
            require(nt.file_sha256(output / "prospective_forecast_seal.json") == seal_sha, "Prospective seal changed")
            for relative, digest in forecast_seal["forecasts"].items():
                require(nt.file_sha256(output / relative) == digest, "Prospective forecast changed")
        for relative, digest in original_bindings.items():
            require(nt.file_sha256(original / relative) == digest, "Original source changed: " + relative)
        if "frozen" in locals():
            for relative, digest in frozen["runtime_source_sha256"].items():
                require(nt.file_sha256(original / relative) == digest, "Original runtime changed: " + relative)
        for relative, digest in model_bindings.items():
            require(nt.file_sha256(ROOT / relative) == digest, "Loaded model changed: " + relative)
        for relative, digest in entry.get("code_sha256", {}).items():
            require(nt.file_sha256(ROOT / relative) == digest, "Execution code changed: " + relative)
        if "config_sha256" in entry:
            require(nt.file_sha256(config_path) == entry["config_sha256"], "Full config changed")
        if "provenance_path" in locals():
            require(nt.file_sha256(provenance_path) == entry["public_provenance_sha256"], "Public provenance changed")
        if "public_source_manifest" in locals():
            require(nt.file_sha256(public_source_manifest) == config["source_run_D1_D2_inventory_sha256"], "Public D1/D2 manifest changed")
        for relative, digest in receipt_bindings.items():
            require(nt.file_sha256(ROOT / relative) == digest, "Reviewed resource evidence changed")
        for relative, digest in entry.get("authenticated_prior_receipts", {}).items():
            require(nt.file_sha256(original / relative) == digest, "Prior process receipt changed")
        require(all(value == 0 for value in guards.values()), "Future engine guard nonzero")
        source_verified = bool(source_bindings and original_bindings and model_bindings)
    except (Exception, KeyboardInterrupt) as error:
        status = "STOP_INCOMPLETE"
        failure = {"prior_failure": failure, "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    stages["post_execution_source_code_model_config_rehash_ns"] = time.perf_counter_ns() - phase
    decision = resource_gate("final_before_summary", reject=False)
    if decision["decision"] != "PASS_RESOURCE" and status == "COMPLETE_NT_FULLBANK_EXACT_768":
        status = "STOP_RESOURCE_AFTER_ACTUALS"
    elapsed = time.monotonic() - PROGRAM_START
    missing = [{"candidate_id": index, "panel": panel, "repeat": repeat} for index, panel, repeat in expected_trial_keys() if (index, panel, repeat) not in completed]
    summary = {"schema": "followup-nt-full-summary-v1", "status": status, "error": failure, "rows": rows,
        "expected_trajectories": 768, "completed_trajectories": len(rows), "missing_row_keys": missing,
        "active_unsealed_result_key": list(active_key) if active_key is not None else None,
        "actual_physics_intervals_returned": total_intervals, "actual_physics_substeps_returned": total_substeps,
        "model_ids": list(MODELS), "forecast_candidate_count": 128, "forecast_metadata": prediction_metadata,
        "teacher_metadata": teacher_metadata, "training_updates": 0, "conditional_extension": "NOT_RUN", "scientific_row_reuse": False,
        "future_engine_calls_during_prediction": guards, "model_calls": {name: {"calls": model.calls, "rows": model.rows} for name, model in models.items()},
        "source_D2_comparison_count": sum(row["source_D2_raw_exact"] is not None for row in rows),
        "complete_probe_rehydration_count": sum(row["full_probe_roundtrip"] is not None for row in rows),
        "whole_probe_repeat_comparison_count": sum(row["full_probe_repeat_exact"] is not None for row in rows),
        "original_source_models_code_config_unchanged": source_verified, "stages_ns": stages,
        "wall_seconds": elapsed, "cumulative_prior_wall_seconds": budget.limits.prior_seconds,
        "cumulative_spent_wall_seconds": budget.limits.prior_seconds + elapsed,
        "output_bytes_before_summary": nt.logical_bytes(output), "free_bytes": shutil.disk_usage(output).free,
        "final_resource_decision": decision, "observed_unit_maxima": asdict(budget.rates),
        "peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limits_of_evidence": ["One restored N-T state and one original finite128 bank; neither full MPC nor independent initial-condition success rate.",
            "A nominal H5 fixed cost differs from B supervisor+zero-tail/horizon; no B-as-H5 cost.",
            "Max observed component rates plus1.5 safety remain empirical forecasts, not universal OS upper bounds.",
            "Cooperative deadline checked after each returned physical interval; returned prefixes persist before comparator; unreturned native-call hangs/power/storage failure are outside this assurance.",
            "Whole-result first-repeat persistence256 plus repeat checks512, raw all768, original-D2 reference96 only where those futures exist.",
            "Process wall includes entry/import/forecast/all trials/teacher/final rehash; final summary write and process shutdown require external process-wall receipt."]}
    nt.write_json(output / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("status", "completed_trajectories", "actual_physics_intervals_returned", "wall_seconds", "cumulative_spent_wall_seconds")}), flush=True)
    if status != "COMPLETE_NT_FULLBANK_EXACT_768":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
