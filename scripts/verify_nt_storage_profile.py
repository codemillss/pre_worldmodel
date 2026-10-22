#!/usr/bin/env python3
"""Verify selected public storage-profile evidence using only the stdlib.

No local raw payloads, codec, predictor or physics engine are loaded. Recorded
exact verdicts and omitted-payload hashes remain evidence with that stated scope.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"artifacts/followup/20261022-nt-storage-profile"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def equal(left, right, context):
    """Structural equality with rounding tolerance only on finite floats."""
    if isinstance(left, dict) and isinstance(right, dict):
        need(left.keys() == right.keys(), context+": keys differ")
        for key in left:
            equal(left[key], right[key], context+"/"+key)
    elif isinstance(left, list) and isinstance(right, list):
        need(len(left) == len(right), context+": lengths differ")
        for index, (a, b) in enumerate(zip(left, right)):
            equal(a, b, context+"/"+str(index))
    elif type(left) is float or type(right) is float:
        need(type(left) in (int, float) and type(right) in (int, float) and
             math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-8), context+": number differs")
    else:
        need(type(left) is type(right) and left == right, context+": value differs")


def verify(base=BASE):
    started = time.perf_counter()
    base = Path(base).resolve()
    manifest = load(base/"manifest.json")
    need(manifest["schema"] == "nt-storage-profile-public-subset-v1" and
         manifest["original_status"] == "STOP_RESOURCE" and manifest["remaining_actual_trajectories"] == 698 and
         manifest["new_physics_calls"] == manifest["new_model_calls"] == manifest["new_actual_trajectories"] == 0, "Public scope/status assertions differ")
    files = manifest["files"]
    seen = set()
    for record in files:
        relative = record["path"]
        path = (base/relative).resolve()
        need(relative not in seen and path.is_relative_to(base) and path.name != "manifest.json", "Duplicate/unsafe/self manifest path")
        seen.add(relative)
        need(path.is_file() and path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], "Public hash/size mismatch: "+relative)
        need(path.suffix == ".json", "Unexpected binary/unlisted payload publication")
        text = path.read_text(encoding="utf-8")
        need(not any(pattern in text for pattern in ("/Users/", "/private/var/", "/var/folders/", "file://")), "Private absolute path in public evidence: "+relative)
    actual = {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file() and p.name != "manifest.json"}
    need(actual == seen, "Manifest omitted or invented package file")
    if "files_excluding_self" in manifest:
        need(manifest["files_excluding_self"] == len(files), "Manifest file count differs")
    if "file_count_excluding_self" in manifest:
        need(manifest["file_count_excluding_self"] == len(files), "Manifest file count differs")
    if "bytes_excluding_self" in manifest:
        need(manifest["bytes_excluding_self"] == sum(r["bytes"] for r in files), "Manifest byte count differs")
    raw_copies = manifest.get("copies", manifest.get("source_copy_records", []))
    need(isinstance(raw_copies, list) and raw_copies, "Source copy provenance missing")
    copies = {}
    for record in raw_copies:
        name = record.get("public_path", record.get("path"))
        need(name in seen and name not in copies, "Duplicate/unknown copied public file")
        copies[name] = record
        if "public_sha256" in record:
            need(record["public_sha256"] == sha(base/name), "Copy public SHA differs")
        need(isinstance(record.get("source_sha256"), str) and len(record["source_sha256"]) == 64, "Copy source SHA missing")
        need(record["source_sha256"] == sha(base/name), "Reviewed byte-identical subset was transformed")

    def source_sha(name):
        need(name in copies, "Required original-source binding absent: "+name)
        return copies[name]["source_sha256"]

    entry, summary, process, config, attribution, resource = [load(base/name) for name in
        ("entry.json", "summary.json", "process_wall_receipt.json", "executed_config.json", "attribution.json", "resource_reassessment.json")]
    need(entry["schema"] == "nt-storage-profile-entry-v1" and summary["schema"] == "nt-storage-profile-summary-v1" and
         summary["status"] == "COMPLETE_SAVED_STORAGE_PROFILE" and process["exit_code"] == 0, "Incomplete/wrong profile schema")
    need(summary["entry_sha256"] == source_sha("entry.json") and entry["config_sha256"] == source_sha("executed_config.json"), "Profile entry/config provenance differs")
    need(summary["bindings"] == entry["bindings"] == config["bindings"], "Inherited profile bindings differ")
    need(config["remaining_scientific_run_authorized_in_this_profile"] is False, "Profile authorized actual continuation")
    need(manifest["source_summary_sha256"] == source_sha("summary.json") and
         manifest["source_process_wall_receipt_sha256"] == source_sha("process_wall_receipt.json"), "Manifest source binding differs")
    need(entry["source_code_sha256"] == config["source_code_sha256"], "Executed profile code identity map differs")
    for relative, digest in entry["source_code_sha256"].items():
        need(sha(ROOT/relative) == digest, "Executed profile source changed: "+relative)
    guards = summary["forbidden_compute_calls"]
    expected_guards = {"mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse", "environment.__init__",
        "environment.reset", "environment.restore", "environment.step", "model.__call__"}
    need(set(guards) == expected_guards and all(type(v) is int and v == 0 for v in guards.values()), "Forbidden compute counter missing/nonzero")
    need(summary["training_updates"] == summary["new_actual_trajectories"] == 0, "Learning/new actual trajectory executed")
    need(set(summary["source_runs_unchanged"]) == set(config["source_runs"]) and all(v is True for v in summary["source_runs_unchanged"].values()), "Source preservation receipt failed")
    need(set(summary["source_code_unchanged"]) == set(config["source_code_sha256"]) and all(v is True for v in summary["source_code_unchanged"].values()), "Code preservation receipt failed")
    need(process["process_wall_seconds"] >= summary["wall_seconds"] >= 0, "Outer process clock does not cover inner clock")
    trials, calibration = summary["trials"], summary["reference_calibration"]
    expected = {(c["id"], r) for c in config["cases"] for r in range(config["repeats"])}
    observed = [(r["case_id"], r["repeat"]) for r in trials]
    need(len(trials) == len(expected) == 18 and len(set(observed)) == 18 and set(observed) == expected, "Profile case/repeat Cartesian mismatch")
    need(len(calibration) == 3 and {(r["case_id"], r["repeat"]) for r in calibration} == {(config["calibration_case_id"], i) for i in range(3)}, "Original calibration Cartesian mismatch")
    for kind, rows in (("instrumented", trials), ("calibration", calibration)):
        for row in rows:
            folder = f"round-{row['repeat']}/{row['case_id']}" if kind == "instrumented" else f"calibration-{row['repeat']}"
            equal(load(base/folder/"measurement.json"), row, "measurement/"+folder)
            for key in ("tree_exact", "whole_probe_exact"):
                verdict = row[key]
                need(verdict["status"] == "PASS_EXACT" and verdict["left_sha256"] == verdict["right_sha256"], "Recorded exact comparison failed")
            need(set(row["payload_sha256"]) == {"trace.npz", "input.json.gz"} and all(len(h) == 64 for h in row["payload_sha256"].values()), "Complete payload hash pair missing")
            if kind == "instrumented":
                stages = row["stages"]
                need(stages["bundle"]["inclusive_ns"] == row["total_ns"] and sum(r["exclusive_ns"] for r in stages.values()) == row["total_ns"] and
                     all(r["exclusive_ns"] >= 0 for r in stages.values()), "Nested wall timing does not partition bundle")
                need(sum(r["exclusive_cpu_ns"] for r in stages.values()) == stages["bundle"]["inclusive_cpu_ns"] and
                     all(r["exclusive_cpu_ns"] >= 0 for r in stages.values()), "Nested CPU timing does not partition bundle")
    for reference in calibration:
        timed = next(r for r in trials if (r["case_id"], r["repeat"]) == (reference["case_id"], reference["repeat"]))
        need(timed["payload_sha256"] == reference["payload_sha256"] and timed["calibration"]["payload_bytes_identical"] is True, "Original/timed payload hash pair differs")
        equal(timed["calibration"]["timed_over_reference_ratio"], timed["total_ns"]/reference["total_ns"], "calibration ratio")
    summarize = module(ROOT/"tools/summarize_nt_storage_profile.py", "nt_storage_public_summarizer")
    need(attribution["summarizer_sha256"] == sha(ROOT/"tools/summarize_nt_storage_profile.py"), "Attribution summarizer source changed")
    expected_attribution = summarize.summarize(base)
    normalized = dict(attribution)
    # A derived public JSON retains original input SHAs. The copy provenance
    # authenticates their relationship to path-redacted public inputs.
    for field, filename in (("source_summary_sha256", "summary.json"), ("source_process_wall_sha256", "process_wall_receipt.json")):
        need(normalized[field] in {source_sha(filename), sha(base/filename)}, "Attribution input source binding differs")
        normalized[field] = expected_attribution[field]
    normalized.pop("aggregation_seconds", None)
    expected_attribution.pop("aggregation_seconds", None)
    equal(normalized, expected_attribution, "attribution")
    reassess = module(ROOT/"tools/reassess_nt_remaining.py", "nt_storage_public_resource_math")
    need(resource["assessment_tool_sha256"] == sha(ROOT/"tools/reassess_nt_remaining.py"), "Resource assessment tool source changed")
    work = {"trials": 698, "intervals": 4540, "source_comparisons": 26, "complete_probe_groups": 232}
    need(resource["remaining"] == work, "Remaining-work count differs")
    clock, limits = resource["time_accounting"], resource["limits"]
    need(clock["prior_delivery_conservative_seconds"] == reassess.PRIOR_SECONDS and clock["current_profile_outer_process_seconds"] == process["process_wall_seconds"], "Prior/profile clock differs")
    need(limits == {"wall_cap_seconds":7200., "safety_factor":1.5, "final_report_reserve_seconds":600., "sealing_reserve_seconds":60.,
        "storage_cap_bytes":5*1024**3, "free_reserve_bytes":2*1024**3, "sealing_reserve_bytes":16*1024**2}, "Resource caps/reserves changed")
    base_rates = resource["frozen_rate_scenario"]["unit_rates"]
    frozen_summary = load(ROOT/"artifacts/followup/20261022-nt-full/summary.json")
    need(base_rates == frozen_summary["observed_unit_maxima"], "Historical rate floor changed")
    maximum_s = max(r["total_ns"] for r in trials+calibration)/1e9
    maximum_b = max(r["output_bytes"] for r in trials+calibration)
    computed = reassess.assess(prior_seconds=clock["prior_delivery_conservative_seconds"], process_seconds=clock["current_profile_outer_process_seconds"],
        elapsed_postwork_seconds=clock["nonoverlapping_current_task_additional_work_seconds"], assessment_seconds=clock["this_assessment_elapsed_seconds"],
        work=work, rates=base_rates, preserved_bytes=resource["preserved_local_experiment_bytes_at_assessment"],
        free_bytes=resource.get("free_bytes_at_assessment", 100*1024**3), profile_bundle_seconds=maximum_s, profile_bundle_bytes=maximum_b)
    for field in ("time_accounting", "remaining", "limits", "step_guard_envelope", "authorization"):
        equal(resource[field], computed[field], "resource/"+field)
    for name in ("frozen_rate_scenario", "profile_only_raise_scenario"):
        # The historical assessment does not expose a live free-space reading.
        # All time, storage growth, floor and crossover arithmetic is public.
        equal({k:v for k,v in resource[name].items() if k not in {"failures", "status"}},
              {k:v for k,v in computed[name].items() if k not in {"failures", "status"}}, "resource/"+name)
        need(resource[name]["status"] == "HOLD_RESOURCE" and "cumulative_wall_cap" in resource[name]["failures"] and
             resource[name]["projected_cumulative_seconds"] > 7200., "Resource HOLD/time-cap conclusion differs")
    need(resource["profile_inclusive_bundle_max_seconds"] == maximum_s and resource["profile_inclusive_bundle_max_bytes"] == maximum_b and resource["profile_bundle_count"] == 21, "Max bundle/bytes/count differs")
    inventory = load(base/"local_evidence_inventory.json")
    additional = load(base/"additional_work_receipt.json")
    need(inventory["schema"] == "nt-storage-profile-local-inventory-v1" and inventory["run_name"] == manifest["source_run"] and
         inventory["local_complete_bundle_count"] == 21, "Local omitted-evidence inventory scope differs")
    inventory_rows = {}
    for row in inventory["files"]:
        relative = row["path"]
        need(relative not in inventory_rows and not Path(relative).is_absolute() and ".." not in Path(relative).parts, "Duplicate/unsafe local inventory path")
        inventory_rows[relative] = row
        included = row["included_in_public_repository"]
        need(type(included) is bool and included == (relative in seen), "Local/public inclusion flag differs")
        if included:
            need(sha(base/relative) == row["sha256"] and (base/relative).stat().st_size == row["bytes"], "Included local file hash/size differs")
        else:
            need(not (base/relative).exists(), "Omitted raw evidence unexpectedly published")
    for kind, rows in (("instrumented", trials), ("calibration", calibration)):
        for row in rows:
            folder = f"round-{row['repeat']}/{row['case_id']}" if kind == "instrumented" else f"calibration-{row['repeat']}"
            bundle_bytes = 0
            for name in ("trace.npz", "input.json.gz"):
                saved = inventory_rows[folder+"/complete_probe/"+name]
                need(saved["included_in_public_repository"] is False and saved["sha256"] == row["payload_sha256"][name], "Omitted complete-payload hash differs")
                bundle_bytes += saved["bytes"]
            receipt = inventory_rows[folder+"/complete_probe_receipt.json"]
            need(receipt["included_in_public_repository"] is False, "Local absolute-path receipt unexpectedly published")
            bundle_bytes += receipt["bytes"]
            need(bundle_bytes == row["output_bytes"], "Omitted complete bundle byte total differs")
    need(additional["schema"] == "nt-profile-additional-nonoverlap-work-v1" and additional["current_setup_tests_inside_preparation"] is True, "Additional-work receipt schema/scope differs")
    extra_parts = [additional[k] for k in ("preparation_from_first_profile_source_creation_seconds", "post_profile_elapsed_through_budget_call_seconds", "final_publication_verification_reserve_seconds")]
    need(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in extra_parts), "Negative/invalid additional clock component")
    equal(sum(extra_parts), additional["charged_additional_work_seconds"], "additional-work component sum")
    equal(additional["charged_additional_work_seconds"], clock["nonoverlapping_current_task_additional_work_seconds"], "resource additional-work charge")
    boundaries = additional["boundaries"]
    equal(boundaries["profile_outer_process_seconds"], process["process_wall_seconds"], "additional-work profile clock")
    equal(boundaries["profile_process_receipt_mtime_epoch"]-boundaries["profile_outer_process_seconds"]-boundaries["first_profile_source_birthtime_epoch"],
          additional["preparation_from_first_profile_source_creation_seconds"], "additional-work preparation boundaries")
    return {"schema":"nt-storage-profile-public-checks-v1", "status":"PASS_PUBLIC_HASHES_TIMING_PARTITION_AND_RESOURCE_HOLD",
        "created_at_utc":datetime.now(timezone.utc).isoformat(), "package_manifest_sha256":sha(base/"manifest.json"),
        "verifier_sha256":sha(__file__), "public_files_checked":len(files), "instrumented_bundles":18, "original_calibration_bundles":3,
        "recorded_exact_pairs_checked":42, "new_actual_trajectories":0, "training_updates":0,
        "forbidden_compute_counts":guards, "resource_status":"HOLD_RESOURCE", "remaining":work,
        "local_evidence_inventory_sha256":sha(base/"local_evidence_inventory.json"), "additional_work_receipt_sha256":sha(base/"additional_work_receipt.json"),
        "local_inventory_entries_checked":len(inventory_rows), "omitted_complete_payload_files_checked":42,
        "omitted_complete_receipts_checked":21, "additional_work_component_sum":"PASS",
        "new_physics_or_predictor_calls_by_verifier":0, "elapsed_seconds":time.perf_counter()-started,
        "limits":["Public package omits complete NPZ/gzip evidence. Hash inventory and recorded exact verdicts do not independently decode those local payloads.",
            "No cold-cache or historical OS/thermal/disk cause can be inferred from warmed saved-case timing.",
            "Free-space live reading is absent from the original resource record; public verification recomputes time and storage-growth formulas, not current live disk availability.",
            "Additional-work receipt is authenticated public provenance; this verifier cannot independently reconstruct the human/task elapsed clock.",
            "No remaining actual trajectory is executed or authorized by this verification."]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=BASE)
    parser.add_argument("--output", type=Path)
    args=parser.parse_args()
    if args.output:
        need(not args.output.exists() and args.output.resolve().parent.is_dir(), "Exclusive output with existing parent required")
        need(not args.output.resolve().is_relative_to(args.package.resolve()), "Verifier cannot modify public package")
    result=verify(args.package)
    if args.output:
        with args.output.open("x", encoding="utf-8") as target:
            json.dump(result,target,ensure_ascii=False,indent=2,allow_nan=False)
            target.write("\n");target.flush();os.fsync(target.fileno())
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
