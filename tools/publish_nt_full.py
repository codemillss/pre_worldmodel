#!/usr/bin/env python3
"""Publish portable saved N-T full-bank evidence, without models or physics.

Publication starts only after a final process-wall receipt exists. All source
files are read-only. Full raw, complete probes and the complete typed input are
inventoried by SHA and size but omitted from the public subset. JSON views are
path-portable derivatives, with original SHA identities retained in a manifest.
Both complete and explicitly stopped runs are supported without row reuse.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import shutil
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("R", "S1-11", "S1-23", "S1-47", "SK-11", "SK-23", "SK-47")
COMPLETE = "COMPLETE_NT_FULLBANK_EXACT_768"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as target:
        json.dump(payload, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


def portable_path(path):
    return unicodedata.normalize("NFC", Path(path).as_posix())


def path_mapper(roots):
    mappings = sorted(((unicodedata.normalize("NFC", str(path.resolve())), alias)
                       for path, alias in roots), key=lambda pair: len(pair[0]), reverse=True)
    substitutions = set()

    def text(value):
        original = value
        value = unicodedata.normalize("NFC", value)
        for prefix, alias in mappings:
            value = value.replace(prefix, alias)
        if value != original:
            substitutions.add(original)
        return value

    def transform(value):
        if isinstance(value, dict):
            return {text(key): transform(item) for key, item in value.items()}
        if isinstance(value, list):
            return [transform(item) for item in value]
        return text(value) if isinstance(value, str) else value

    return transform, substitutions


def assert_portable(value):
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False)
    require(not any(pattern in payload for pattern in
                    ("/Users/", "/private/var/", "/var/folders/", "file://")),
            "Private absolute path remains in public JSON")


def jsonl_view(path):
    """Retain all parsed lines; an interrupted final line is explicitly omitted."""
    raw = path.read_bytes()
    lines, omitted = [], None
    pieces = raw.splitlines(keepends=True)
    for index, piece in enumerate(pieces):
        try:
            lines.append(json.loads(piece))
        except (json.JSONDecodeError, UnicodeDecodeError):
            require(index == len(pieces) - 1 and not piece.endswith(b"\n"),
                    "Malformed non-final JSONL record")
            omitted = {"line_1based": index + 1, "bytes": len(piece),
                       "sha256": hashlib.sha256(piece).hexdigest(),
                       "reason": "Interrupted final JSONL line; not a scientific result"}
    return lines, omitted


def checked_files(receipt, run, *, kind):
    records = []
    for filename, expected in receipt["files_sha256"].items():
        path = Path(filename).resolve()
        require(path.is_relative_to(run) and path.is_file(), kind + " payload outside this run/missing")
        require(sha(path) == expected, kind + " payload SHA changed")
        records.append({"path": portable_path(path.relative_to(run)), "sha256": expected,
                        "bytes": path.stat().st_size, "included_in_public_repository": False})
    return records


def publish(run, output, config_path, analysis_path, process_path):
    started = time.perf_counter()
    run, output, config_path, analysis_path, process_path = (path.resolve() for path in
        (run, output, config_path, analysis_path, process_path))
    require(run.is_dir() and not output.exists() and not output.is_relative_to(run)
            and not run.is_relative_to(output), "Exclusive separate destination required")
    original = run.parent.parent
    source_run = original / "runs/followup-near-goal-20261022-01"
    entry_path, summary_path = run / "entry.json", run / "summary.json"
    config, entry, summary = load(config_path), load(entry_path), load(summary_path)
    analysis, process = load(analysis_path), load(process_path)
    status = summary["status"]
    require(status == COMPLETE or status.startswith("STOP_"), "Final COMPLETE/STOP receipt required")
    require(process.get("exit_code") is not None and math.isfinite(process["process_wall_seconds"])
            and process["process_wall_seconds"] >= summary["wall_seconds"], "Final external process-wall receipt required")
    require((status == COMPLETE) == (process["exit_code"] == 0), "Process exit/final status disagree")
    require(analysis["schema"] == "followup-nt-fullbank-saved-result-summary-v1"
            and analysis["source_run_status"] == status, "Wrong analysis run/status")
    require(analysis["bindings"]["run_receipt_sha256"] == sha(summary_path)
            and analysis["bindings"]["entry_sha256"] == sha(entry_path), "Analysis source changed")
    require(sha(ROOT / "tools/summarize_nt_full.py") == analysis["bindings"]["summarizer_sha256"],
            "Analysis implementation changed")
    require(sha(config_path) == entry["config_sha256"], "Executed config changed")
    require(entry["full_input_typed_sha256"] == config["expected_input_typed_sha256"]
            and entry["bank_hash"] == config["expected_candidate_hash"], "Input/bank identity mismatch")
    require(sha(run / "bank.npz") == entry["bank_receipt"]["sha256"], "Bank artifact changed")
    require(summary["model_ids"] == list(MODELS) and summary["forecast_candidate_count"] == 128,
            "Seven-model/full128 forecast dimensions changed")
    require(entry["model_and_normalizer_sha256"] == config["expected_model_normalizer_sha256"],
            "Model/normalizer map changed")
    source_audits = {}
    for scope, base, mapping in (("execution_code", ROOT, entry["code_sha256"]),
            ("loaded_model_normalizer", ROOT, entry["model_and_normalizer_sha256"]),
            ("original", original, entry["original_sha256"]),
            ("D1_D2_source", source_run, entry["source_run_sha256"]),
            ("reviewed_resource", ROOT, entry.get("reviewed_resource_sha256", {})),
            ("preserved_prior_receipts", original, entry.get("authenticated_prior_receipts", {}))):
        for relative, digest in mapping.items():
            path = (base / relative).resolve()
            require(path.is_relative_to(base.resolve()) and sha(path) == digest,
                    "Immutable " + scope + " changed: " + relative)
        source_audits[scope] = {"files": len(mapping), "fresh_sha_check": "PASS"}
    seal_path = run / "prospective_forecast_seal.json"
    seal = load(seal_path)
    require(seal["before_any_new_actual_future"] is True
            and all(value == 0 for value in seal["future_engine_calls_during_prediction"].values())
            and all(value == 0 for value in summary["future_engine_calls_during_prediction"].values()),
            "Forecast seal or future-engine guard invalid")
    for relative, digest in seal["forecasts"].items():
        require(sha(run / relative) == digest, "Prospective prediction changed")
    rows = summary["rows"]
    row_map = {(row["candidate_id"], row["panel"], row["repeat"]): row for row in rows}
    expected = set(product(range(128), ("A", "B"), range(3)))
    require(len(row_map) == len(rows) and set(row_map).issubset(expected), "Invalid/duplicate trial keys")
    require(len(rows) == summary["completed_trajectories"] == analysis["dimensions"]["recorded_complete_trajectories"],
            "Complete-row count mismatch")
    if status == COMPLETE:
        require(set(row_map) == expected and len(rows) == 768
                and analysis["status"] == "COMPLETE_768_STABLE_SAVED_EVIDENCE"
                and summary["original_source_models_code_config_unchanged"] is True,
                "Complete run missing rows/analysis/immutability")
    transform, substitutions = path_mapper(((run, "local_run"), (source_run, "source_D1_D2_run"),
                                             (ROOT, "repository_root"), (original, "project_root")))
    output.mkdir(parents=True, exist_ok=False)
    copy_records = []

    def emit(source, target, *, scope="run", kind="json"):
        source, destination = Path(source), output / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        original_sha = sha(source)
        if kind == "json":
            value = transform(load(source))
            assert_portable(value)
            dump(destination, value)
            action = "derived_json_paths_mapped_and_unicode_NFC"
        elif kind == "jsonl":
            lines, omitted = jsonl_view(source)
            with destination.open("x", encoding="utf-8") as stream:
                for line in lines:
                    value = transform(line)
                    assert_portable(value)
                    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
            action = "derived_jsonl_paths_mapped_and_unicode_NFC"
            if omitted is not None:
                dump(destination.with_suffix(destination.suffix + ".omitted_tail.json"), omitted)
                action += "_explicit_interrupted_tail_omitted"
        else:
            require(kind == "binary", "Unsupported publication kind")
            with source.open("rb") as reader, destination.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
            require(sha(destination) == original_sha, "Binary copy changed")
            action = "byte_identical"
        if scope == "run":
            source_label = source.relative_to(run)
        elif scope == "repository":
            source_label = source.relative_to(ROOT)
        else:
            source_label = source.name
        copy_records.append({"source_scope": scope, "source_path": portable_path(source_label),
            "source_sha256": original_sha, "public_path": portable_path(destination.relative_to(output)),
            "public_sha256": sha(destination), "bytes": destination.stat().st_size,
            "transformation": action})

    emit(config_path, "executed_config.json", scope="repository")
    for relative in ("entry.json", "entry_gate.json", "summary.json", "prospective_forecast_seal.json"):
        if (run / relative).is_file():
            emit(run / relative, relative)
    for relative in ("resource_decisions.jsonl", "progress.jsonl"):
        if (run / relative).is_file():
            emit(run / relative, relative, kind="jsonl")
    emit(run / "bank.npz", "bank.npz", kind="binary")
    emit(analysis_path, "analysis.json", scope="analysis")
    emit(process_path, "process_wall_receipt.json", scope="process_receipt")
    analysis_wall_path = run / "analysis_process_wall_receipt.json"
    analysis_seconds = analysis["analysis_seconds"]
    if analysis_wall_path.exists():
        analysis_wall = load(analysis_wall_path)
        require(analysis_wall["exit_code"] == 0 and math.isfinite(analysis_wall["process_wall_seconds"])
                and analysis_wall["process_wall_seconds"] >= analysis_seconds, "Analysis process wall receipt invalid")
        analysis_seconds = analysis_wall["process_wall_seconds"]
        emit(analysis_wall_path, "analysis_process_wall_receipt.json")
    for folder in ("forecasts", "teacher"):
        if (run / folder).is_dir():
            for source in sorted((run / folder).iterdir()):
                require(source.suffix in (".npz", ".json") and source.is_file(), "Unexpected forecast/teacher payload")
                emit(source, source.relative_to(run), kind="json" if source.suffix == ".json" else "binary")
    inventories, trial_index = [], []
    raw_count = complete_count = diagnostic_count = 0
    for index, panel, repeat in sorted(expected):
        trial = run / "actual/N-T" / f"candidate-{index:03d}" / panel / f"repeat-{repeat}"
        if not trial.exists():
            continue
        key = (index, panel, repeat)
        row = row_map.get(key)
        relative = trial.relative_to(run)
        emitted = []
        for name in ("entry.json", "summary.json", "timing_final.json", "raw_receipt.json", "complete_probe_receipt.json", "incomplete.json"):
            if (trial / name).exists():
                emit(trial / name, relative / name)
                emitted.append(name)
        raw_files = checked_files(load(trial / "raw_receipt.json"), run, kind="Raw") if (trial / "raw_receipt.json").exists() else []
        complete_files = checked_files(load(trial / "complete_probe_receipt.json"), run, kind="Complete probe") if (trial / "complete_probe_receipt.json").exists() else []
        if raw_files:
            require(len(raw_files) == 2, "Expected exactly one lossless raw pair")
            raw_count += 1
        if complete_files:
            require(repeat == 0 and len(complete_files) == 2, "Complete-probe evidence must be first-repeat pair")
            complete_count += 1
        if repeat == 0 and (trial / "diagnostic.npz").exists():
            if row is not None:
                require(sha(trial / "diagnostic.npz") == row["diagnostic_receipt"]["sha256"], "Diagnostic identity changed")
            emit(trial / "diagnostic.npz", relative / "diagnostic.npz", kind="binary")
            diagnostic_count += 1
        if row is not None:
            require(raw_files and row["initial_snapshot_exact"] and row["raw_roundtrip"]["status"] == "PASS_EXACT", "Sealed scientific row lacks exact raw evidence")
            comparison = row["full_probe_roundtrip"] if repeat == 0 else row["full_probe_repeat_exact"]
            require(comparison and comparison["status"] == "PASS_EXACT", "Sealed row full-probe check failed")
        inventories.append({"candidate_id": index, "panel": panel, "repeat": repeat,
            "sealed_scientific_row": row is not None,
            "whole_probe_typed_sha256": row.get("whole_probe_typed_sha256") if row else None,
            "raw_files": raw_files, "complete_probe_files": complete_files})
        trial_index.append({"candidate_id": index, "panel": panel, "repeat": repeat,
            "sealed_scientific_row": row is not None, "public_receipts": emitted,
            "first_repeat_diagnostic_included": repeat == 0 and (trial / "diagnostic.npz").exists(),
            "raw_pair_included": False, "complete_probe_included": False})
    if status == COMPLETE:
        require(raw_count == 768 and complete_count == diagnostic_count == 256, "Complete public evidence inventory dimensions differ")
    input_files = checked_files(entry["full_input_receipt"], run, kind="Complete input")
    dump(output / "local_evidence_inventory.json", {"schema": "followup-nt-full-local-evidence-inventory-v1",
        "scope": "Inventory only: all raw pairs, complete first-repeat probes and full typed input remain local. Paths/hashes identify omitted payloads, not downloadable public copies.",
        "source_run_status": status, "raw_pair_count": raw_count,
        "complete_probe_group_count": complete_count, "first_repeat_boundary_diagnostic_count": diagnostic_count,
        "records": inventories, "full_input_files": input_files})
    dump(output / "trial_index.json", {"schema": "followup-nt-full-public-trial-index-v1",
        "planned_trajectories": 768, "sealed_scientific_rows": len(rows),
        "missing_keys": [dict(zip(("candidate_id", "panel", "repeat"), key)) for key in sorted(expected - set(row_map))],
        "records": trial_index})
    measured_wall = process["process_wall_seconds"]
    known_seconds = config["cumulative_prior_wall_seconds"] + measured_wall + analysis_seconds
    elapsed = time.perf_counter() - started
    resource = {"schema": "followup-nt-full-observed-resource-receipt-v1",
        "cumulative_clock_definition": config["cumulative_clock_definition"],
        "prior_charged_process_seconds": config["cumulative_prior_wall_seconds"],
        "full_process_seconds_including_shutdown": measured_wall,
        "full_process_wall_source_sha256": sha(process_path),
        "analysis_reported_seconds": analysis["analysis_seconds"],
        "analysis_charged_process_seconds": analysis_seconds,
        "publisher_seconds_before_final_metadata": elapsed,
        "known_cumulative_seconds_before_final_metadata": known_seconds + elapsed,
        "wall_cap_seconds": config["limits"]["wall_time_cap_s"],
        "remaining_seconds_before_other_postwork": config["limits"]["wall_time_cap_s"] - known_seconds - elapsed,
        "full_process_within_cap": config["cumulative_prior_wall_seconds"] + measured_wall <= config["limits"]["wall_time_cap_s"],
        "observed_full_run_directory_bytes_before_publication": sum(path.stat().st_size for path in run.rglob("*") if path.is_file()),
        "publisher_future_calls": {"physics": 0, "models": 0},
        "scope": "Known measured processes only. Independent review and later finalization must be added by the final delivery ledger; this is not a claim that unmeasured postwork is free.",
        "executor_final_resource_decision": summary["final_resource_decision"],
        "fresh_source_sha_checks": source_audits}
    dump(output / "observed_resources.json", resource)
    dump(output / "path_mapping.json", {"schema": "followup-nt-full-portable-path-mapping-v1",
        "scope": "Public JSON/JSONL are path-portable derived views, binary arrays are byte-identical; source SHA identities are retained in manifest copy records.",
        "aliases": {"local_run": "current saved full-bank run root", "source_D1_D2_run": "preserved D1/D2 run root",
            "repository_root": "public repository", "project_root": "preserved original project"},
        "substitution_count": len(substitutions), "private_original_path_strings_published": False})
    for path in output.rglob("*.json"):
        assert_portable(load(path))
    files = [{"path": portable_path(path.relative_to(output)), "sha256": sha(path), "bytes": path.stat().st_size}
             for path in sorted(output.rglob("*")) if path.is_file()]
    manifest = {"schema": "followup-nt-full-selected-public-evidence-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_relative_path": "runs/" + run.name, "source_run_status": status,
        "scope": "One N-T fixed128-bank seven-model forecast and fresh A/B three-repeat diagnostic. First-repeat boundary arrays and fresh one-step teacher inputs/predictions only; full raw/probe/input omitted. No model/physics/training execution by publication.",
        "source_entry_sha256": sha(entry_path), "source_summary_sha256": sha(summary_path),
        "source_analysis_sha256": sha(analysis_path), "executed_config_sha256": sha(config_path),
        "publisher_sha256": sha(__file__), "source_copy_records": copy_records,
        "files": files, "files_excluding_self": len(files), "bytes_excluding_self": sum(row["bytes"] for row in files),
        "scientific_rows": len(rows), "local_raw_pair_count": raw_count, "local_complete_probe_group_count": complete_count,
        "public_first_repeat_diagnostic_count": diagnostic_count,
        "public_limits": "Public files authenticate reported exact checks and permit boundary/forecast analyses; omitted local complete payloads cannot be independently reread from public files alone.",
        "publisher_seconds_before_manifest": time.perf_counter() - started}
    dump(output / "manifest.json", manifest)
    return {"source_run_status": status, "public_files": len(files), "public_bytes": manifest["bytes_excluding_self"],
            "scientific_rows": len(rows), "local_raw_pair_count": raw_count,
            "local_complete_probe_group_count": complete_count,
            "publisher_seconds": time.perf_counter() - started}


def self_test():
    transform, _ = path_mapper(((Path("/Users/example/project/runs/new"), "local_run"),
                               (Path("/Users/example/project"), "project_root")))
    value = transform({"/Users/example/project/runs/new/x": ["/Users/example/project/src/y", "normal"]})
    require(value == {"local_run/x": ["project_root/src/y", "normal"]}, "Longest-prefix path mapping failed")
    assert_portable(value)
    caught = False
    try:
        assert_portable({"unmapped": "/Users/other/private"})
    except ValueError:
        caught = True
    require(caught, "Unmapped private path not rejected")
    print("PASS 2 portability checks; no run files/models/physics read")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/followup/20261022-nt-full")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/followup_nt_full.v1.json")
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--process-wall-receipt", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        require(args.run is None and args.analysis is None and args.process_wall_receipt is None,
                "Self-test cannot publish a run")
        self_test()
    else:
        require(args.run is not None and args.analysis is not None and args.process_wall_receipt is not None,
                "--run, --analysis and --process-wall-receipt are required")
        print(json.dumps(publish(args.run, args.output, args.config, args.analysis, args.process_wall_receipt), indent=2))
