#!/usr/bin/env python3
"""Preflight the N-T 128-candidate executor without authorizing the full batch.

The seven frozen predictors forecast the complete bank before any new physical
future. Only predeclared IDs 0, 9 and 127 are physically probed (A/B x3). Every
returned prefix is sealed before comparison failures can stop the run. This
entry point intentionally has no full-execution mode.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import fields
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback

PROGRAM_START = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from push_align.closed_loop import ENGINE_GUARDS, model_planning_scope
from push_align.environment import StepResult
from push_align.final_driver_v2 import exact_comparison, read_full_trace, typed_digest, write_full_trace
from push_align.final_execution import array_hash
from push_align.final_recording import file_sha256
from push_align import final_recording_columns as codec
from push_align.followup_probe import FollowupProbe, run_followup_probe
from push_align.planning import rollout, score, choose, contact_distance
from push_align.planning_protocol import GeometryProposalAdapter
from followup_pilot import get_input, load_models
from followup_restore import regenerate, write_json

MODELS = ("R", "S1-11", "S1-23", "S1-47", "SK-11", "SK-23", "SK-47")
IDS = (0, 9, 127)
CODE_PATHS = (
    "scripts/followup_nt_executor.py", "scripts/followup_pilot.py",
    "scripts/followup_restore.py", "src/push_align/followup_probe.py",
    "src/push_align/final_driver_v2.py", "src/push_align/final_recording_columns.py",
    "src/push_align/planning.py", "src/push_align/environment.py",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def complete_probe_tree(probe):
    """Lossless explicit field tree supported by the unchanged typed codec."""
    values = {field.name: getattr(probe, field.name) for field in fields(probe)}
    values["results"] = tuple({field.name: getattr(step, field.name)
                               for field in fields(step)} for step in probe.results)
    return {"schema": "followup-complete-probe-tree-v1", "class": "FollowupProbe", "fields": values}


def reconstruct_probe(tree):
    require(tree.get("schema") == "followup-complete-probe-tree-v1" and tree.get("class") == "FollowupProbe",
            "Unrecognized complete probe tree")
    values = dict(tree["fields"])
    require(set(values) == set(FollowupProbe.__dataclass_fields__), "Complete probe fields differ")
    require(isinstance(values["results"], tuple), "Probe results must retain tuple type")
    values["results"] = tuple(StepResult(**step) for step in values["results"])
    return FollowupProbe(**values)


def npz_roundtrip(path, arrays):
    """Exclusive durable write plus exact authenticated array read-back."""
    with path.open("xb") as target:
        np.savez_compressed(target, **arrays)
        target.flush()
        os.fsync(target.fileno())
    digest = file_sha256(path)
    with np.load(path, allow_pickle=False) as source:
        loaded = {key: source[key].copy() for key in source.files}
    require(file_sha256(path) == digest, "NPZ changed during read-back")
    comparison = exact_comparison(arrays, loaded)
    require(comparison["status"] == "PASS_EXACT", "NPZ typed bit-exact read-back failed")
    return {"file": path.name, "sha256": digest, "bytes": path.stat().st_size,
            "roundtrip": comparison}


def inventory(folder):
    return {p.relative_to(folder).as_posix(): file_sha256(p)
            for p in sorted(folder.rglob("*")) if p.is_file()}


def logical_bytes(folder):
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


def compare_pass(left, right, message):
    verdict = exact_comparison(left, right)
    require(verdict["status"] == "PASS_EXACT", message)
    return verdict



def seal_and_validate_prefix(trial, snapshot, prefix, timings, *, validate=True):
    """Persist first, so an exception in any later verifier retains new raw."""
    phase = time.perf_counter_ns()
    receipt = codec.write_raw(trial / "raw", snapshot, prefix, xz_preset=3)
    timings["raw_write_fsync_ns"] = time.perf_counter_ns() - phase
    write_json(trial / "raw_receipt.json", receipt)
    if not validate:
        return receipt, None
    phase = time.perf_counter_ns()
    loaded_snapshot, loaded_steps = codec.read_raw(trial / "raw", expected_sha256=receipt["files_sha256"])
    timings["raw_authenticated_read_ns"] = time.perf_counter_ns() - phase
    phase = time.perf_counter_ns()
    comparison = compare_pass({"snapshot": snapshot, "steps": prefix},
        {"snapshot": loaded_snapshot, "steps": loaded_steps}, "Raw round-trip differs")
    timings["raw_roundtrip_full_exact_ns"] = time.perf_counter_ns() - phase
    return receipt, comparison


def teacher_inputs(source_run, data):
    """Use original D2 first-repeat A observations; no new physics or labels."""
    starts, truth, commands, identifiers = [], [], [], []
    for index in range(16):
        folder = source_run / "D2/N-T" / f"candidate-{index:03d}/A/repeat-0"
        receipt = json.loads((folder / "raw_receipt.json").read_text())
        snapshot, steps = codec.read_raw(folder / "raw", expected_sha256=receipt["files_sha256"])
        compare_pass(snapshot, data["snapshot"], "Teacher source snapshot mismatch")
        current = data["observation"].state
        for interval, step in enumerate(steps):
            require(np.isfinite(current).all(), "Nonfinite teacher source")
            starts.append(current.copy())
            truth.append(step.observation.state.copy())
            commands.append(step.applied_action.copy())
            identifiers.append((index, interval))
            current = step.observation.state
    require(0 < len(starts) <= 80, "Expected at most 80 original teacher rows")
    return {"start": np.asarray(starts, np.float64), "actual": np.asarray(truth, np.float64),
            "applied": np.asarray(commands, np.float64), "candidate_interval": np.asarray(identifiers, np.int64)}


def forecasts(folder, data, bank, predictors, guards):
    folder.mkdir()
    metadata = {}
    for name in MODELS:
        start = time.perf_counter_ns()
        with model_planning_scope(guards):
            predicted = rollout(np.broadcast_to(data["observation"].state, (128, 10)).copy(),
                                bank, predictors[name], data["planner_config"])
            costs = score(predicted, bank, data["task_spec"], data["planner_context"].previous_applied_action,
                          data["planner_config"])
            selected, _, reason = choose(costs, predicted, bank)
            gap = contact_distance(predicted.states, data["planner_config"])
        infer_ns = time.perf_counter_ns() - start
        arrays = {"states": predicted.states, "raw_deltas": predicted.raw_deltas,
                  "valid": predicted.valid, "reasons": np.asarray(predicted.reasons),
                  "failure_steps": predicted.failure_steps, "total_cost": costs.total,
                  "geometric_gap_proxy": gap, **{"cost_" + key: value for key, value in costs.components.items()}}
        start = time.perf_counter_ns()
        receipt = npz_roundtrip(folder / f"prediction-{name}.npz", arrays)
        metadata[name] = {"prediction_cost_selection_ns": infer_ns,
                          "durable_save_readback_exact_ns": time.perf_counter_ns() - start,
                          "valid_count": int(predicted.valid.sum()), "selected_original_candidate_id": selected,
                          "reason": reason, "receipt": receipt, "candidate_count": 128, "horizon": 5,
                          "selection_scope": "same original128 bank; no physical outcomes used"}
    require(all(value == 0 for value in guards.values()), "Future engine call during model forecasting")
    write_json(folder / "predictions.json", metadata)
    return metadata


def run_teacher(folder, arrays, data, predictors, guards, benchmark_rows):
    folder.mkdir()
    source = npz_roundtrip(folder / "source-original-D2.npz", arrays)
    indices = np.arange(benchmark_rows, dtype=np.int64) % len(arrays["start"])
    benchmark = {key: value[indices].copy() for key, value in arrays.items()}
    benchmark["source_row_index"] = indices
    benchmark_source = npz_roundtrip(folder / "benchmark-repeated-prefix.npz", benchmark)
    metadata = {"source": source, "benchmark_source": benchmark_source,
                "source_rows": len(arrays["start"]), "benchmark_rows": benchmark_rows,
                "scope": "Original D2 labels only. Repeated-prefix 640-row throughput benchmark is not independent teacher evidence.", "models": {}}
    for name in MODELS:
        model_rows = {}
        for label, sample in (("source", arrays), ("benchmark", benchmark)):
            start = time.perf_counter_ns()
            with model_planning_scope(guards):
                prediction = rollout(sample["start"], sample["applied"][:, None, :], predictors[name], data["planner_config"])
            elapsed = time.perf_counter_ns() - start
            start = time.perf_counter_ns()
            receipt = npz_roundtrip(folder / f"{label}-{name}.npz", {
                "predicted": prediction.states[:, 1], "raw_deltas": prediction.raw_deltas[:, 0],
                "valid": prediction.valid, "reasons": np.asarray(prediction.reasons),
                "failure_steps": prediction.failure_steps})
            model_rows[label] = {"rows": len(sample["start"]), "rollout_ns": elapsed,
                                 "durable_save_readback_exact_ns": time.perf_counter_ns() - start,
                                 "receipt": receipt}
        metadata["models"][name] = model_rows
    require(all(value == 0 for value in guards.values()), "Future engine call during teacher benchmark")
    write_json(folder / "summary.json", metadata)
    return metadata


def validate_config(config):
    require(config["state"] == "N-T" and config["candidate_count"] == 128, "N-T full128 only")
    require(config["preflight_candidate_ids"] == list(IDS), "Predeclared preflight IDs required")
    require(config["models"] == list(MODELS), "Seven fixed model identities/order required")
    require(config["panels"] == ["A", "B"] and config["repeats"] == 3, "A/B and three repeats required")
    require(config["teacher_benchmark_rows"] == 640, "640-row benchmark required")
    limits = config["limits"]
    require(limits["wall_time_cap_s"] == 7200 and limits["safety_factor"] == 1.5,
            "Cumulative time cap or safety factor changed")
    require(limits["incremental_storage_cap_bytes"] == 5 * 1024**3 and limits["free_space_reserve_bytes"] == 2 * 1024**3,
            "Storage cap/reserve changed")
    require(config["cumulative_prior_wall_seconds"] == 589.8162389602512, "Prior experiment-process clock differs from reviewed ledger")
    require(limits["preflight_max_wall_seconds"] == 1200 and limits["preflight_write_allowance_bytes"] == 256 * 1024**2, "Preflight subcaps changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/followup_nt_preflight.v1.json")
    parser.add_argument("--mode", choices=("preflight",), default="preflight")
    args = parser.parse_args()
    original, source_run, output, config_path = (p.resolve() for p in
        (args.original_root, args.source_run, args.output, args.config))
    require(not output.exists() and output != source_run and source_run not in output.parents,
            "Output must be an exclusive separate run, outside source run")
    output.mkdir(parents=True, exist_ok=False)
    rows, stages, guards, source_bindings, original_bindings, models, model_bindings = [], {}, dict.fromkeys(ENGINE_GUARDS, 0), {}, {}, {}, {}
    total_intervals = 0
    total_substeps = 0
    entry = {"schema": "followup-nt-executor-preflight-v1", "mode": args.mode,
             "started_at_utc": datetime.now(timezone.utc).isoformat(), "full_execution_authorized": False,
             "actual_candidate_ids": list(IDS), "actual_panels": ["A", "B"], "actual_repeats": 3,
             "forecast_candidate_count": 128, "models": list(MODELS), "training_updates": 0,
             "durability": "Exclusive fsynced complete raw at trial completion/error; no per-substep power-loss guarantee."}
    try:
        phase = time.perf_counter_ns()
        config = json.loads(config_path.read_text())
        validate_config(config)
        limits = config["limits"]
        entry["config_sha256"] = file_sha256(config_path)
        entry["code_sha256"] = {path: file_sha256(ROOT / path) for path in CODE_PATHS}
        require(output not in original.parents, "Output may not be a parent of originals")
        source_bindings = inventory(source_run)
        public_source_manifest = ROOT / "artifacts/followup/20261022-d1-d2/manifest.json"
        require(file_sha256(public_source_manifest) == config["source_run_D1_D2_inventory_sha256"], "Published D1/D2 evidence manifest changed")
        require(file_sha256(ROOT / config["resource_ledger_path"]) == config["resource_ledger_sha256"], "Reviewed resource ledger changed")
        for relative, key in (("D1/N-T/restoration.json", "source_D1_report_sha256"),
                              ("D2/entry.json", "source_D2_entry_sha256"),
                              ("D2/summary.json", "source_D2_summary_sha256")):
            require(file_sha256(source_run / relative) == config[key], "Source run entry/summary identity changed: " + relative)
        manifest_path = original / "runs/p5-final-20261017-02/manifest.json"
        require(file_sha256(manifest_path) == config["original_manifest_sha256"], "Original manifest SHA mismatch")
        frozen = json.loads(manifest_path.read_text())
        design_path = ROOT / "configs/followup_candidate_diagnostic.v1.json"
        require(file_sha256(design_path) == config["original_design_sha256"], "Original design changed")
        data, bank, labels, d1 = get_input(source_run, "N-T")
        require(d1["design_sha256"] == config["original_design_sha256"], "D1 design SHA mismatch")
        capture = next(item for item in d1["captures"] if item["tick"] == 78)
        require(typed_digest(data) == capture["input_typed_sha256"] == config["expected_input_typed_sha256"], "D1 complete typed input mismatch")
        require(file_sha256(source_run / "D1/N-T" / capture["bank"]) == config["expected_bank_file_sha256"], "Expected D1 bank file SHA mismatch")
        require(array_hash(bank) == capture["candidate_hash"] == config["expected_candidate_hash"], "D1 full-bank hash mismatch")
        original_bindings = dict(d1["input_sha256"])
        for rel, digest in original_bindings.items():
            require(file_sha256(original / rel) == digest, "Changed original " + rel)
        for rel, digest in frozen["runtime_source_sha256"].items():
            require(file_sha256(original / rel) == digest, "Original runtime changed " + rel)
            if rel.startswith("src/"):
                require(file_sha256(ROOT / rel) == digest, "Public frozen runtime changed " + rel)
        adapter = GeometryProposalAdapter(data["physical_config"], data["geometry_parameters"], data["transition_parameters"])
        proposal, generated, keys = regenerate(data, data["observation"], data["task_status"], data["planner_context"],
                                              data["geometry_context"], adapter, data["tick"], guards)
        candidate_compare = compare_pass(bank, generated.actions, "Regenerated candidate bank differs")
        compare_pass(labels, np.asarray(generated.sources), "Regenerated candidate source labels differ")
        require(keys == data["rng_keys"], "RNG keys differ")
        bank_receipt = npz_roundtrip(output / "bank.npz", {"actions": bank, "sources": labels,
            "original_ids": np.arange(128, dtype=np.int64)})
        full_input_receipt = write_full_trace(output / "input", tick=78, captured_input=data)
        reread_input, _, _ = read_full_trace(full_input_receipt)
        input_compare = compare_pass(data, reread_input, "Input save/read exact mismatch")
        entry.update(source_run_sha256=source_bindings, original_sha256=original_bindings,
            source_run_D1_D2_inventory_scope="Config inventory SHA binds published D1/D2 manifest; source_run_sha256 separately authenticates all local files before/after.",
            source_run_D1_D2_inventory_sha256=file_sha256(public_source_manifest),
            resource_ledger_sha256=file_sha256(ROOT / config["resource_ledger_path"]),
            full_input_typed_sha256=typed_digest(data), bank_hash=array_hash(bank),
            unique_candidate_sequences=len({array_hash(row) for row in bank}), rng_keys=keys,
            input_roundtrip=input_compare, regenerated_bank_exact=candidate_compare, bank_receipt=bank_receipt,
            full_input_receipt=full_input_receipt, resource_limits=limits,
            cumulative_prior_wall_seconds=config["cumulative_prior_wall_seconds"],
            runtime={"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                     "mujoco": __import__("mujoco").__version__, "device": "CPU", "torch_threads": 1,
                     "deterministic_algorithms": True, "physical_state_dtype": "float64"})
        stages["bindings_input_candidate_regeneration_ns"] = time.perf_counter_ns() - phase
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        phase = time.perf_counter_ns()
        models, model_bindings = load_models()
        require(sorted(models) == sorted(MODELS), "Model identities differ")
        provenance_path = ROOT / "artifacts/provenance/publication_files.json"
        provenance = {r["public_path"]: r for r in json.loads(provenance_path.read_text())["records"]}
        require(len(model_bindings) == 24, "Expected exactly 24 model/normalizer file bindings")
        for rel, digest in model_bindings.items():
            record = provenance[rel]
            source = record["source_path"]
            require(record["transformation"] == "byte_identical" and
                digest == record["public_sha256"] == record["source_sha256"] == frozen["runtime_source_sha256"][source] == file_sha256(original / source),
                "Loaded model/normalizer differs from frozen original: " + rel)
        stages["strict_models_load_and_provenance_ns"] = time.perf_counter_ns() - phase
        entry["model_and_normalizer_sha256"] = model_bindings
        entry["public_provenance_sha256"] = file_sha256(provenance_path)
        write_json(output / "entry.json", entry)
        phase = time.perf_counter_ns()
        prediction_metadata = forecasts(output / "forecasts", data, bank, models, guards)
        stages["full128_forecast_cost_durable_save_readback_ns"] = time.perf_counter_ns() - phase
        phase = time.perf_counter_ns()
        teacher_arrays = teacher_inputs(source_run, data)
        stages["original_D2_teacher_input_authentication_ns"] = time.perf_counter_ns() - phase
        phase = time.perf_counter_ns()
        teacher_metadata = run_teacher(output / "teacher", teacher_arrays, data, models, guards, config["teacher_benchmark_rows"])
        stages["teacher_source_and_640row_benchmark_ns"] = time.perf_counter_ns() - phase
        forecast_seal = {"schema": "followup-nt-prospective-forecast-seal-v1",
            "before_any_new_actual_future": True, "forecasts": {p.relative_to(output).as_posix(): file_sha256(p)
                for p in sorted((output / "forecasts").iterdir()) if p.is_file()},
            "teacher": {p.relative_to(output).as_posix(): file_sha256(p)
                for p in sorted((output / "teacher").iterdir()) if p.is_file()},
            "prediction_calls_before_actual": {k: {"calls": v.calls, "rows": v.rows} for k, v in models.items()},
            "future_engine_calls_during_prediction": dict(guards), "wall_seconds_before_any_actual": time.monotonic() - PROGRAM_START,
            "stages_ns": dict(stages), "timestamp_utc": datetime.now(timezone.utc).isoformat()}
        write_json(output / "prospective_forecast_seal.json", forecast_seal)
        fixed_before_actual = time.monotonic() - PROGRAM_START
        print("N-T full128: seven forecasts and 640-row teacher benchmark sealed before any new physics", flush=True)

        def guard_trial():
            remaining_s = config["cumulative_prior_wall_seconds"] + (time.monotonic() - PROGRAM_START)
            # Reserve a conservative complete longest trial's persistence cost;
            # the full768 batch remains a separate approval/gate decision.
            next_s = 1.5 * 8 * limits["prior_worst_seconds_per_interval"] + 60.0  # complete-field sealing reserve
            require(remaining_s + next_s + limits["final_report_reserve_seconds"] <= limits["wall_time_cap_s"], "NOT_RUN_RESOURCE cumulative time cap")
            require(time.monotonic() - PROGRAM_START + next_s <= limits["preflight_max_wall_seconds"], "NOT_RUN_RESOURCE preflight time subcap")
            used = logical_bytes(output)
            next_bytes = math.ceil(1.5 * 8 * limits["prior_worst_bytes_per_interval"]) + 16 * 1024**2
            require(config["existing_run_bytes_before_preflight"] + used + next_bytes <= limits["incremental_storage_cap_bytes"], "NOT_RUN_RESOURCE cumulative storage cap")
            require(used + next_bytes <= limits["preflight_write_allowance_bytes"], "NOT_RUN_RESOURCE preflight storage allowance")
            require(shutil.disk_usage(output).free >= limits["free_space_reserve_bytes"] + next_bytes,
                    "NOT_RUN_RESOURCE free-space reserve")

        trial_phase = time.perf_counter_ns()
        for index in IDS:
            for panel in ("A", "B"):
                reference = None
                for repeat in range(3):
                    guard_trial()
                    trial = output / "actual/N-T" / f"candidate-{index:03d}" / panel / f"repeat-{repeat}"
                    trial.mkdir(parents=True, exist_ok=False)
                    write_json(trial / "entry.json", {"candidate_id": index, "panel": panel, "repeat": repeat,
                        "candidate_hash": array_hash(bank[index]), "full_input_typed_sha256": typed_digest(data),
                        "prospective_seal_sha256": file_sha256(output / "prospective_forecast_seal.json")})
                    prefix, timings, result, execution_failure = [], {}, None, None
                    trial_start = time.perf_counter_ns()
                    phase = time.perf_counter_ns()
                    try:
                        result = run_followup_probe(data["physical_config"], data["planner_config"], data["snapshot"],
                            data["task_spec"], data["task_status"], bank[index], data["planner_context"].previous_applied_action,
                            frozen["profile"], panel=panel, on_step=lambda i, step, command: prefix.append(step))
                    except Exception as error:
                        execution_failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
                    timings["physical_probe_including_unchanged_validation_ns"] = time.perf_counter_ns() - phase
                    total_intervals += len(prefix)
                    total_substeps += sum(len(step.substep_states) for step in prefix)
                    # Always seal the returned prefix BEFORE comparator, status,
                    # full-result codec, or original-reference errors can stop.
                    receipt, roundtrip = seal_and_validate_prefix(trial, data["snapshot"], prefix, timings,
                        validate=execution_failure is None)
                    if execution_failure is not None:
                        write_json(trial / "incomplete.json", execution_failure)
                        raise RuntimeError("Probe infrastructure failure; completed raw prefix retained")
                    phase = time.perf_counter_ns()
                    repeat_compare = None if reference is None else compare_pass(reference, result, "UNSTABLE_REPLAY complete FollowupProbe differs")
                    timings["full_probe_repeat_exact_ns"] = time.perf_counter_ns() - phase
                    phase = time.perf_counter_ns()
                    original_compare = None
                    if index < 16:
                        old = source_run / "D2/N-T" / f"candidate-{index:03d}" / panel / "repeat-0"
                        old_receipt = json.loads((old / "raw_receipt.json").read_text())
                        old_snapshot, old_steps = codec.read_raw(old / "raw", expected_sha256=old_receipt["files_sha256"])
                        original_compare = compare_pass({"snapshot": old_snapshot, "steps": old_steps},
                            {"snapshot": data["snapshot"], "steps": prefix}, "Fresh actual differs from source D2")
                    timings["source_D2_raw_read_and_exact_ns"] = time.perf_counter_ns() - phase
                    phase = time.perf_counter_ns()
                    full_probe_receipt = None
                    full_probe_compare = None
                    if reference is None:
                        tree = complete_probe_tree(result)
                        full_probe_receipt = write_full_trace(trial / "complete_probe", tick=78, captured_input=tree)
                        loaded_tree, _, _ = read_full_trace(full_probe_receipt)
                        compare_pass(tree, loaded_tree, "Full probe tree save/read differs")
                        full_probe_compare = compare_pass(result, reconstruct_probe(loaded_tree), "Reconstructed full FollowupProbe differs")
                        write_json(trial / "complete_probe_receipt.json", full_probe_receipt)
                        reference = result
                    timings["first_repeat_complete_probe_durable_roundtrip_ns"] = time.perf_counter_ns() - phase
                    phase = time.perf_counter_ns()
                    diagnostic_receipt = npz_roundtrip(trial / "diagnostic.npz", {
                        "states": result.states, "boundary_times": result.boundary_times,
                        "nominal_actions": result.nominal_actions, "requested_actions": result.requested_actions,
                        "applied_actions": result.applied_actions, "supervisor_overrides": result.supervisor_overrides,
                        "hold_s": result.diagnostic_hold_s, "qualifies": result.diagnostic_qualifies})
                    timings["diagnostic_durable_roundtrip_ns"] = time.perf_counter_ns() - phase
                    phase = time.perf_counter_ns()
                    whole_probe_digest = typed_digest(result)
                    timings["whole_probe_typed_digest_ns"] = time.perf_counter_ns() - phase
                    phase = time.perf_counter_ns()
                    trial_bytes = logical_bytes(trial)
                    timings["trial_output_traversal_ns"] = time.perf_counter_ns() - phase
                    timings["trial_total_before_summary_ns"] = time.perf_counter_ns() - trial_start
                    row = {"state": "N-T", "candidate_id": index, "candidate_hash": array_hash(bank[index]),
                        "panel": panel, "repeat": repeat, "intervals": len(prefix),
                        "physical_valid": result.physical_valid, "ready": result.ready,
                        "ready_prefix": result.ready_prefix, "ready_eligible": result.ready_eligible,
                        "first_ready_interval": result.first_ready_interval, "horizon_completed": result.horizon_completed,
                        "termination": result.termination, "reasons": list(result.reasons),
                        "cost": result.cost if result.cost is not None and math.isfinite(result.cost) else None,
                        "cost_finite": result.cost is not None and math.isfinite(result.cost),
                        "cost_components": {k: v if math.isfinite(v) else None for k, v in result.cost_components.items()},
                        "raw_roundtrip": roundtrip, "full_probe_repeat_exact": repeat_compare,
                        "full_probe_roundtrip": full_probe_compare, "source_D2_raw_exact": original_compare,
                        "source_D2_scope": "Available IDs0/9 only; 127 has no prior D2 actual",
                        "diagnostic_receipt": diagnostic_receipt, "timings_ns": timings,
                        "raw_bytes": sum(p.stat().st_size for p in codec.raw_paths(trial / "raw")),
                        "trial_bytes_before_summary": trial_bytes,
                        "first_repeat_complete_probe_bytes": (sum(full_probe_receipt["bytes"].values()) +
                            (trial / "complete_probe_receipt.json").stat().st_size) if full_probe_receipt else 0,
                        "whole_probe_typed_sha256": whole_probe_digest,
                        "initial_snapshot_exact": result.initial_snapshot_exact}
                    phase = time.perf_counter_ns()
                    write_json(trial / "summary.json", row)
                    timings["summary_write_fsync_ns"] = time.perf_counter_ns() - phase
                    timings["trial_total_through_summary_ns"] = time.perf_counter_ns() - trial_start
                    write_json(trial / "timing_final.json", {"timings_ns": timings,
                        "scope": "Includes whole-probe digest, summary serialization/fsync and all preceding work; this small timing receipt write follows measurement."})
                    rows.append(row)
                print(f"N-T candidate{index:03d} {panel}: complete raw roundtrip and full-probe repeat exact x3", flush=True)
        stages["actual_preflight_trials_and_all_verification_ns"] = time.perf_counter_ns() - trial_phase
        phase = time.perf_counter_ns()
        require(inventory(source_run) == source_bindings, "Source D1/D2 run changed during preflight")
        for relative, digest in {**forecast_seal["forecasts"], **forecast_seal["teacher"]}.items():
            require(file_sha256(output / relative) == digest, "Prospectively sealed forecast/benchmark changed: " + relative)
        for rel, digest in original_bindings.items():
            require(file_sha256(original / rel) == digest, "Original file changed during preflight: " + rel)
        for rel, digest in frozen["runtime_source_sha256"].items():
            require(file_sha256(original / rel) == digest, "Original runtime changed during preflight: " + rel)
        for rel, digest in model_bindings.items():
            require(file_sha256(ROOT / rel) == digest, "Loaded model changed during preflight: " + rel)
        for rel, digest in entry["code_sha256"].items():
            require(file_sha256(ROOT / rel) == digest, "Executor code changed during preflight: " + rel)
        require(file_sha256(config_path) == entry["config_sha256"], "Preflight config changed during execution")
        require(file_sha256(provenance_path) == entry["public_provenance_sha256"], "Public provenance changed")
        require(file_sha256(public_source_manifest) == config["source_run_D1_D2_inventory_sha256"], "Published D1/D2 manifest changed")
        require(file_sha256(ROOT / config["resource_ledger_path"]) == config["resource_ledger_sha256"], "Reviewed resource ledger changed")
        require(all(value == 0 for value in guards.values()), "Future engine guard nonzero")
        stages["post_execution_source_code_model_config_rehash_ns"] = time.perf_counter_ns() - phase
        phase = time.perf_counter_ns()
        used = logical_bytes(output)
        free = shutil.disk_usage(output).free
        stages["output_traversal_and_free_space_ns"] = time.perf_counter_ns() - phase
        fixed_after_ns = stages["post_execution_source_code_model_config_rehash_ns"] + stages["output_traversal_and_free_space_ns"]
        elapsed = time.monotonic() - PROGRAM_START
        require(len(rows) == 18, "Expected all18 physical preflight trajectories")
        require(config["cumulative_prior_wall_seconds"] + elapsed + limits["final_report_reserve_seconds"] <= limits["wall_time_cap_s"], "NOT_RUN_RESOURCE cumulative final gate")
        require(elapsed <= limits["preflight_max_wall_seconds"], "NOT_RUN_RESOURCE final preflight wall gate")
        require(used <= limits["preflight_write_allowance_bytes"] and config["existing_run_bytes_before_preflight"] + used <= limits["incremental_storage_cap_bytes"], "NOT_RUN_RESOURCE final storage gate")
        require(free >= limits["free_space_reserve_bytes"], "NOT_RUN_RESOURCE final reserve gate")
        summary = {"schema": "followup-nt-executor-preflight-summary-v1", "status": "PASS_NT_EXECUTOR_PREFLIGHT",
            "mode": "preflight", "full128_actual_execution": "NOT_RUN", "conditional_extension": "NOT_RUN",
            "rows": rows, "model_ids": list(MODELS), "forecast_candidate_count": 128,
            "forecast_metadata": prediction_metadata, "teacher_metadata": teacher_metadata,
            "stages_ns": stages, "fixed_before_any_actual_seconds": fixed_before_actual,
            "fixed_after_actual_seconds": fixed_after_ns / 1e9,
            "wall_seconds": elapsed, "cumulative_prior_wall_seconds": config["cumulative_prior_wall_seconds"],
            "cumulative_spent_wall_seconds": config["cumulative_prior_wall_seconds"] + elapsed,
            "actual_trajectories": len(rows), "actual_physics_intervals": total_intervals,
            "actual_physics_substeps": total_substeps, "training_updates": 0,
            "future_engine_calls_during_prediction": guards,
            "model_calls": {k: {"calls": v.calls, "rows": v.rows} for k, v in models.items()},
            "unchanged_source_run_files": len(source_bindings), "unchanged_original_files": len(original_bindings),
            "unchanged_model_normalizer_files": len(model_bindings), "source_run_bytes": logical_bytes(source_run),
            "current_preflight_bytes_before_summary": used, "free_bytes": free,
            "peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "next_full_batch_dimensions": {"states": 1, "candidates": 128, "panels": ["A", "B"], "repeats": 3,
                "physical_trajectories": 768, "max_intervals": 4992, "teacher_max_unique_transition_rows": 640,
                "D2_reuse": False, "preflight_reuse": False},
            "limits": limits, "resource_projection": "Independent reviewer must reprice measured fixed stages and unchanged per-interval trial costs before full execution.",
            "limits_of_evidence": ["Three predeclared candidate IDs only have new actual futures.",
                "Full128 predictions do not establish full128 physical behavior or cost/readiness conclusions.",
                "640-row teacher benchmark repeats original at-most80 rows; it creates no new training/evaluation evidence.",
                "No future replanning, new bank, training or conditional extension executed.",
                "Complete raw preserved before every later comparator/codec may raise; storage hardware failure remains infrastructure failure.",
                "Measured wall includes imports since module timing start, source rehash and output traversal; summary-file write itself follows this measurement."]}
        write_json(output / "summary.json", summary)
        print(json.dumps({k: summary[k] for k in ("status", "actual_trajectories", "actual_physics_intervals",
            "fixed_before_any_actual_seconds", "fixed_after_actual_seconds", "wall_seconds")}), flush=True)
    except Exception as error:
        failure = {"schema": "followup-nt-executor-preflight-incomplete-v1", "status": "STOP_INCOMPLETE_NO_AUTOMATIC_RETRY",
            "rows": rows, "actual_physics_intervals_returned": total_intervals, "actual_physics_substeps_returned": total_substeps, "stages_ns": stages,
            "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
            "wall_seconds": time.monotonic() - PROGRAM_START, "full_execution_started": False, "training_updates": 0}
        write_json(output / "incomplete.json", failure)
        print(json.dumps({k: failure[k] for k in ("status", "error", "actual_physics_intervals_returned")}), flush=True)
        raise


if __name__ == "__main__":
    main()
