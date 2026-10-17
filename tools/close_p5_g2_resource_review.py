#!/usr/bin/env python3
"""Close saved G2 evidence after PDF verification; never execute experiments."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from push_align.resource_budget import capacity_decision


def read(path): return json.loads((ROOT / path).read_text())
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    plan = read("configs/p5_g2_resource_plan.v1.json")
    pdfbase = "output/pdf/프로젝트_통합보고서"
    verification = read(pdfbase + ".verification.json")
    assert verification["status"] == "PASS"
    assert verification["pdf_sha256"] == sha(ROOT / (pdfbase + ".pdf"))
    assert verification["manifest_sha256"] == sha(ROOT / (pdfbase + ".manifest.json"))
    mutable_history = {"README.md", "docs/00_프로젝트_목표와_전체진행안내.md",
                       "reports/00_전체보고서_색인.md", "reports/task_status.md",
                       "reports/p4_progress_summary.md", "tools/export_report_pdf.py",
                       *(pdfbase + suffix for suffix in (".pdf", ".manifest.json", ".verification.json"))}
    groups = {
        "scientific748": read("configs/p4_first_moment.v1.json")["bindings_sha256"],
        "frozen_candidate97": read("configs/p5_freeze_candidate.v1.json")["source_sha256"],
        "G1_index_except_declared_living_and_pdf": {path: digest for path, digest in read("reports/p5_preflight_evidence_index.json")["sha256"].items() if path not in mutable_history},
        "resource_plan_sources": plan["source_sha256"],
        "independent_audit_sources": read("reports/p5_g2_independent_audit.json")["source_sha256"],
    }
    preservation = {}
    for name, bindings in groups.items():
        mismatches = [path for path, digest in bindings.items() if not (ROOT / path).is_file() or sha(ROOT / path) != digest]
        preservation[name] = {"count": len(bindings), "all_exact": not mismatches, "mismatches": mismatches}
    assert all(value["all_exact"] for value in preservation.values()), preservation
    new_paths = set()
    for pattern in ("tmp/*g2*", "reports/p5_g2*", "configs/p5*g2*", "configs/p5_recording*",
                    "tools/*p5*g2*", "tmp/pdfs/p5-g2-resource-qa", "output/pdf/history/20261016-before-p5-g2-resource"):
        for path in ROOT.glob(pattern):
            new_paths.update(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else new_paths.add(path)
    new_paths.update(ROOT / path for path in mutable_history if (ROOT / path).is_file())
    new_paths.update(ROOT / path for path in ("src/push_align/final_recording.py", "src/push_align/resource_budget.py",
                     "tests/test_p5_final_recording.py", "tests/test_p5_resource_budget.py",
                     "docs/09_최종기록정책과_자원게이트.md"))
    new_paths.difference_update(ROOT / path for path in
        ("reports/p5_g2_evidence_index.json", "reports/p5_g2_final_snapshot.json"))
    # Count complete new copies and even full rewritten living/PDF files, a
    # conservative own-operation footprint rather than a claim about APFS blocks.
    bytes_before_closure = sum(path.stat().st_size for path in new_paths)
    assert bytes_before_closure < 256 * 1024 ** 2
    free = shutil.disk_usage(ROOT).free
    capacity = capacity_decision(free, plan["estimate"]["required_bytes_including_reserve"])
    hold_path = ROOT / "reports/p5_g2_capacity_hold.json"
    hold = json.loads(hold_path.read_text()) if hold_path.exists() else None
    evidence = {str(path.relative_to(ROOT)): sha(path) for path in sorted(new_paths)}
    evidence_path = ROOT / "reports/p5_g2_evidence_index.json"
    evidence_path.write_text(json.dumps({"schema": "p5-g2-evidence-index-v1",
        "status": "SAVED_EVIDENCE_BOUND_NO_NEW_EXPERIMENT",
        "scope": "Current G2 artifacts and exact preserved sources. Current final snapshot and this index exclude self hashes.",
        "preservation": preservation, "source_sha256": evidence}, ensure_ascii=False, indent=2) + "\n")
    measured_own_bytes = bytes_before_closure + evidence_path.stat().st_size
    snapshot = {"schema": "p5-g2-final-snapshot-v1", "as_of_client_date": "2026-10-16",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "HOLD_CAPACITY" if hold is not None else capacity["status"],
        "capacity": capacity, "capacity_is_timestamped_not_reserved": True,
        "observed_capacity_hold": hold,
        "hold_reopening_requires_stable_recheck": hold is not None,
        "resource_readiness": ("CAPACITY_HOLD_AND_INTEGRATION_PENDING" if hold is not None
                               else "INTEGRATION_AND_RAM_MEASUREMENT_PENDING"),
        "source_preservation": preservation,
        "evidence_index_sha256": sha(evidence_path),
        "PDF": {"sha256": verification["pdf_sha256"], "manifest_sha256": verification["manifest_sha256"],
                "verification_sha256": sha(ROOT / (pdfbase + ".verification.json")),
                "pages": verification["page_count"], "chapters": verification["chapter_count"],
                "direct_visual_pages": verification["visual_review"]["actual_page_count"]},
        "G2_own_resources": {"new_files_and_full_rewrites_bytes_before_snapshot": measured_own_bytes,
                             "ceiling_bytes": 256 * 1024 ** 2, "hard_free_reserve_bytes": 2 * 1024 ** 3},
        "tests": {"codec": 33, "resource_guard": 13, "independent_audit": 65},
        "new_model_forwards": 0, "new_physics_calls": 0, "new_training_updates": 0,
        "actual_final_test_manifest_generated": False, "P5_01": "HOLD", "overall_done": "23/28",
        "remaining_gates": plan["remaining_gates"]}
    target = ROOT / "reports/p5_g2_final_snapshot.json"
    target.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    assert measured_own_bytes + target.stat().st_size < 256 * 1024 ** 2
    print(json.dumps({"snapshot": str(target.relative_to(ROOT)), "sha256": sha(target),
                      "capacity": capacity, "own_bytes": measured_own_bytes + target.stat().st_size,
                      "preservation": preservation}, ensure_ascii=False))


if __name__ == "__main__":
    main()
