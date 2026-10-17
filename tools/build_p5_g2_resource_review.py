#!/usr/bin/env python3
"""Recompute the remaining P5 estimate from measured saved-record codecs only."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from push_align.resource_budget import estimate_storage, capacity_decision


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    measurement = json.loads((ROOT / "reports/p5_g2_recording_policy_measurement.json").read_text())
    summary = measurement["summary"]
    scope = json.loads((ROOT / "configs/p3_evaluation_scope.v3.json").read_text())
    assert scope["E2"]["episodes"] == 96 and scope["E1"]["actual_future_max_intervals"] == 2880
    assert scope["P5_replay_application_resource_allowance"]["existing_replay_episode_upper_bound"] == 96
    counts = dict(
        raw_records=96 + 12 * 16 * 3 + 8,
        raw_intervals=28800 + 2880 + 2400,
        witness_records=96 + 96 + 8,
        witness_intervals=28800 + 28800 + 2400,
        witness_decisions=(96 + 96 + 8) * 301,
        plan_archives=84 + 7,
        plan_records=(84 + 7) * 301,
        full_traces=(84 + 7) * 4,
        G_captures=(12 + 12 + 1) * 3,
    )
    units = dict(
        raw_fixed=summary["raw_empty_prefix_bytes_max"],
        raw_interval=int(summary["raw_incremental_bytes_per_interval_max"]),
        witness_fixed=summary["witness_fixed_bytes_max"],
        witness_interval=summary["witness_interval_member_bytes_max"],
        witness_decision=summary["witness_decision_member_bytes_max"],
        plan_index_fixed=summary["plan_index_fixed_bytes_max"],
        plan_index_record=summary["plan_index_incremental_record_bytes_max"],
        full_trace_npz=115779,
        full_trace_identity=7143,
        snapshot_envelope=16384,
    )
    allowances = dict(
        E1_prediction_analysis=107374183,
        P5_boundary_analysis=107374183,
        source_and_model_bundle=32 * 1024 ** 2,
        future_report_archive_QA=64 * 1024 ** 2,
        working_disk_scratch=32 * 1024 ** 2,
        mismatch_prefix_and_diagnostics=256 * 1024 ** 2,
        hard_free_reserve=2 * 1024 ** 3,
    )
    estimate = estimate_storage(counts, units, allowances)
    evidence_paths = [
        "configs/p3_evaluation_scope.v3.json", "configs/p5_recording_policy.v1.json",
        "configs/p5_freeze_candidate.v1.json", "src/push_align/resource_budget.py",
        "src/push_align/final_recording.py", "tests/test_p5_resource_budget.py",
        "tests/test_p5_final_recording.py", "tools/build_p5_g2_resource_review.py",
        "reports/p5_g2_long_record_measurement.json", "reports/p5_g2_xz_measurement.json",
        "reports/p5_g2_xz_regime_measurement.v2.json",
        "reports/p5_g2_recording_codec_review.v2.json",
        "reports/p5_g2_recording_policy_measurement.json",
        "reports/p5_g2_recording_actual_value_audit.json",
    ]
    native = ROOT / "reports/p5_g2_native300_codec_measurement.json"
    if native.exists():
        evidence_paths += [str(native.relative_to(ROOT)), "tools/p5_g2_native_benchmark.py"]
    native_value = json.loads(native.read_text()) if native.exists() else None
    model_sources = json.loads((ROOT / "configs/p5_freeze_candidate.v1.json").read_text())["source_sha256"]
    free = shutil.disk_usage(ROOT).free
    result = {
        "schema": "p5-g2-resource-plan-v1", "as_of_client_date": "2026-10-16",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "MEASURED_POLICY_ESTIMATE_NOT_BATCH_AUTHORIZATION",
        "counts": counts, "units_bytes": units, "fixed_allowances_bytes": allowances,
        "estimate": estimate,
        "capacity": capacity_decision(free, estimate["required_bytes_including_reserve"]),
        "old_remaining_estimate_bytes": 11828384856,
        "scope_counts_explanation": {
            "E2": "12 families*8 controllers*300 actions=28,800; learned84 cells",
            "E1": "12 snapshots*16 candidates*3 physical repeats=576 records*5=2,880 actions; shared by7 models, not7 copies",
            "P5": "up to96 actual fresh controller reruns plus1 same-range new family*8 controllers; no already-completed learning/P4 allowance charged again",
            "fresh": "normal verified fresh witness references original raw/index; mismatch required extra raw/index/trace charged to256MiB allowance then STOP",
            "decisions": "301 slots/cell includes one conservative unexecuted terminal decision beyond300 actions; no claim that all are physically executable",
            "full_trace": "91 learned originals/new cells*at most4 predeclared-or-allinvalid full arrays+input snapshots; union removes duplicates",
        },
        "unit_evidence_limits": {
            "raw": "13 G1 references+empty records explicitly serialized in new format, max incremental rate122762B; wider19 old-record XZ cases and one native300 codec benchmark supplement, not a proven worst-case bound",
            "witness": "26 G1 dictionaries roundtrip; actual gzip member fixed/interval/decision maxima, additive sizes for300 slots; not a new300 controller run",
            "plan_index": "40 original actual/fault plan arrays in22 archives; same-index values/dtypes/shapes preserved; future per-index fresh reuse still needs driver wiring",
            "full_trace": "40 preserved G1 traces maximum NPZ115779B+identity7143B; input snapshot16KiB envelope vs measured G start capture3003B, monitor and stop on growth",
            "multiplier": "1.5 planning allowance on measured and snapshot-envelope components; not probability bound or guarantee",
            "working_disk": "writer creates final files directly, counted as final output; no extra whole-episode disk staging;32MiB working scratch monitored separately. RAM staging is separate and still needs integration measurement",
        },
        "source_model_bundle_current_bytes": sum((ROOT / path).stat().st_size for path in model_sources),
        "time": {
            "historical_remaining_action_work": 62880,
            "historical_factor_1_5_action_extrapolation_s": 15592.99640793804,
            "old_raw_write_time_included_in_historical": True,
            "new_XZ_write_supplement": "Use measured native300 write seconds/300*34,080*1.5 as conservative extra on top of historical4.33h, which already includes old writing; not a guaranteed finish time. Codec reads, setup, E1 labels/bootstrap/reports/debug/retries and other apps remain uncertain.",
            "concurrency": "one worker/cell at a time; no parallel final physics or training",
            "native300_codec_write_wall_s": native_value["native_codec"]["write_wall_s"] if native_value else None,
            "native300_codec_read_wall_s": native_value["native_codec"]["read_wall_s"] if native_value else None,
            "extra_codec_write_read_factor1_5_extrapolation_s":
                ((native_value["native_codec"]["write_wall_s"] + native_value["native_codec"]["read_wall_s"])
                 / 300 * counts["raw_intervals"] * 1.5) if native_value else None,
        },
        "RAM": {
            "physical_GiB": 16,
            "historical_controller_process_peak_range_GiB": [1.24, 3.09],
            "new_raw_short_codec_peak_MiB": summary["process_peak_rss_MiB"],
            "native300_codec_only_peak_MiB": native_value["native_codec"]["rss"]["after_roundtrip_checks_MiB"] if native_value else None,
            "combined_final_driver_and_native300_codec_peak": "NOT_MEASURED",
            "gate": "bounded one-cell integration before batch, one-worker lifetime and cleanup/RSS monitoring; historical and codec-only peaks do not prove combined fit",
        },
        "money": {"new_service_equipment_spending_KRW": 0, "ceiling_KRW_exclusive": 100000},
        "remaining_gates": [
            "Connect final driver to new codecs, exact fresh-index comparison or explicitly limited reuse, source receipts and live per-stage/per-cell disk guards",
            "Measure bounded combined native final-driver recording RAM/writer lifetime; verify prefix preservation on errors and cleanup between cells",
            "G3: predeclare actual disjoint final families, seeds, manifest hashes and complete source bundle; no final family generated here",
            "Repeat live capacity check immediately before each stage; capacity decline or larger units=>STOP and reestimate without dropping originals/cases/criteria",
        ],
        "new_model_calls": 0, "new_training_updates": 0, "new_physics_calls": 0,
        "actual_final_test_manifest_generated": False,
        "P5_01": "HOLD", "overall_done": "23/28",
        "source_sha256": {path: sha(ROOT / path) for path in evidence_paths},
    }
    out = ROOT / "configs/p5_g2_resource_plan.v1.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path": str(out), "required_bytes": estimate["required_bytes_including_reserve"],
                      "capacity": result["capacity"], "sha256": sha(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
