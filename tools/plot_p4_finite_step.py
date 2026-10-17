"""Render all saved finite-step JSON endpoints without executing a model.

Ratios and logarithmic plotting coordinates are display transformations only.
All112 absolute losses, ratios, statuses and fixed decision labels are retained
in the figure metadata outside the immutable scientific run directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from reportlab.graphics import renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, Rect, String
from reportlab.lib.colors import HexColor
from reportlab.pdfbase.pdfmetrics import stringWidth


ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT, FONT = 1200, 1000, 23
INK, MUTED, GRID = "#193144", "#516A7B", "#DEE6EC"
ALPHAS = (0., .25, .5, 1.)
SHOTS = ("common_initial", "C0_u0100", "C0_u0500", "C0_u1000", "R1_u0100", "R1_u0500", "R1_u1000")
LABELS = ("Common+0", "C0+100", "C0+500", "C0+1000", "R1+100", "R1+500", "R1+1000")
COLORS = ("#34495E", "#4C78A8", "#55A6A6", "#275B8E", "#CC5A49", "#A46BAF", "#18806A")
BATCHES = ("historical_update1", "D11_zero5", "D11_transition", "D11_noncontact")
PANEL_LABELS = ("A  Schedule anchor (64 windows)", "B  D11 zero5 (24 windows)",
                "C  D11 transition (16 windows)", "D  D11 noncontact (8 windows)")
AXES = ((.35, 1.1, (.4, .6, .8, 1.)), (.35, 1.1, (.4, .6, .8, 1.)),
        (.35, 1.1, (.4, .6, .8, 1.)), (.3, 8., (.4, 1., 2., 4., 8.)))
CAPTION = (
    "고정 배치별 동일 TOTAL/KEEP 방향의 유한 크기 손실비 L(alpha)/L(0). "
    "각 패널은 7개 스냅샷과 사전 지정 alpha0·0.25·0.5·1의 모든 점을 포함한다. "
    "각 축은 로그축이며 A-C는 0.35-1.1, D는 0.3-8 범위다. 점선1은 각 조건의 원래 손실이며 "
    "연결선은 네 측정점을 읽기 위한 표시다. 112개 절대 손실과 원래 L(0)는 figure.json에 모두 보존했다. "
    "8/28 조건에서 nominal 증가와 작은 스텝 감소가 함께 관찰됐으며 이는 해당 스냅샷·배치·방향의 "
    "결과다. 전체 학습 원인·일반화·작업 성과나 최적 alpha 선택의 근거는 아니다. "
    "C0와 공통 출발점의 total은 과거 학습 목적함수가 아닌 같은 가상 공통 목적함수다."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def text(drawing, x, y, value, size=FONT, color=INK, anchor="start"):
    require(size >= FONT, "Minimum figure text must stay at least23px")
    width = stringWidth(value, "Helvetica", size)
    left = x - width if anchor == "end" else x - width / 2. if anchor == "middle" else x
    require(left >= 0 and left + width <= WIDTH, "Figure text outside horizontal bounds: " + value)
    drawing.add(String(x, y, value, fontName="Helvetica", fontSize=size,
                       fillColor=HexColor(color), textAnchor=anchor))


def line(drawing, x1, y1, x2, y2, color=GRID, width=1., dash=None):
    drawing.add(Line(x1, y1, x2, y2, strokeColor=HexColor(color), strokeWidth=width, strokeDashArray=dash))


def marker(drawing, x, y, index):
    color, radius = HexColor(COLORS[index]), 4.8
    fill = color if index < 4 else HexColor("#FFFFFF")
    if index in (0, 4):
        drawing.add(Circle(x, y, radius, fillColor=fill, strokeColor=color, strokeWidth=2.))
    elif index in (1, 5):
        drawing.add(Rect(x - radius, y - radius, 2 * radius, 2 * radius,
                         fillColor=fill, strokeColor=color, strokeWidth=2.))
    elif index == 2:
        drawing.add(Polygon([x, y + radius + 1, x + radius + 1, y, x, y - radius - 1, x - radius - 1, y],
                            fillColor=fill, strokeColor=color, strokeWidth=2.))
    else:
        drawing.add(Polygon([x, y + radius + 1, x + radius + 1, y - radius, x - radius - 1, y - radius],
                            fillColor=fill, strokeColor=color, strokeWidth=2.))


def make_figure(records):
    drawing = Drawing(WIDTH, HEIGHT)
    drawing.add(Rect(0, 0, WIDTH, HEIGHT, fillColor=HexColor("#FFFFFF"), strokeColor=None))
    text(drawing, 45, 958, "Finite loss along each saved TOTAL / KEEP direction", 30)
    text(drawing, 45, 918, "All112 endpoints: 7 snapshots x 4 fixed train batches x 4 alphas; no failed or missing endpoints.")
    for index, (label, x, y) in enumerate(zip(LABELS,
                                             (45., 325., 605., 885., 325., 605., 885.),
                                             (875., 875., 875., 875., 838., 838., 838.))):
        dash = [8, 5] if index >= 4 else [3, 4] if index == 0 else None
        line(drawing, x, y + 7, x + 39, y + 7, COLORS[index], 2.5, dash)
        marker(drawing, x + 20, y + 7, index)
        text(drawing, x + 50, y, label)
    for panel, batch in enumerate(BATCHES):
        column, row = panel % 2, panel // 2
        left = 112. + 600. * column
        bottom, height, width = (492. if row == 0 else 119.), 235., 420.
        title_y = 790. if row == 0 else 417.
        low, high, ticks = AXES[panel]
        text(drawing, 45. + 600. * column, title_y, PANEL_LABELS[panel], 24)
        text(drawing, 45. + 600. * column, title_y - 34., f"L(alpha)/L(0), log y [{low:g}, {high:g}]", 23, MUTED)
        def y(value):
            require(math.isfinite(value) and low <= value <= high, "A stored ratio lies outside the declared panel range")
            return bottom + height * (math.log(value) - math.log(low)) / (math.log(high) - math.log(low))
        for tick in ticks:
            current_y = y(tick)
            line(drawing, left, current_y, left + width, current_y,
                 "#7C91A0" if tick == 1. else GRID, 1.8 if tick == 1. else 1., [6, 5] if tick == 1. else None)
            text(drawing, left - 16., current_y - 8., f"{tick:g}", 23, MUTED, "end")
        for alpha in ALPHAS:
            x = left + width * alpha
            line(drawing, x, bottom, x, bottom - 7., "#8396A4")
            text(drawing, x, bottom - 38., f"{alpha:g}", 23, MUTED, "middle")
        for index, snapshot in enumerate(SHOTS):
            record = records[snapshot, batch]
            points = [(left + width * alpha, y(value)) for alpha, value in zip(ALPHAS, record["common_total_ratios"])]
            dash = [8, 5] if index >= 4 else [3, 4] if index == 0 else None
            for (x1, y1), (x2, y2) in zip(points, points[1:]):
                line(drawing, x1, y1, x2, y2, COLORS[index], 2.5, dash)
            for x, current_y in points:
                marker(drawing, x, current_y, index)
    text(drawing, 600, 48, "Horizontal axis: alpha (same saved direction within each condition)", 23, MUTED, "middle")
    text(drawing, 45, 12, "8/28 nominal increases with smaller-step decrease; local train-case contrast, not a whole-training cause.", 23, MUTED)
    return drawing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=ROOT / "reports/p4_finite_step_independent_review.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/p4_finite_step.v1.json")
    parser.add_argument("--output-directory", type=Path, default=ROOT / "reports/figures")
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output_directory.resolve()
    index_path = run / "artifact_hashes.json"
    index = read_json(index_path)["sha256"]
    sources = {str(index_path.relative_to(ROOT)): sha(index_path)}
    def saved(name):
        path = run / name
        require(index.get(name) == sha(path), "Frozen source binding differs: " + name)
        sources[str(path.relative_to(ROOT))] = sha(path)
        return read_json(path)
    aggregate, summary, manifest, registration = (saved(name) for name in
                                                  ("aggregate.json", "summary.json", "manifest.json", "registration.json"))
    require(summary["execution_status"] == "PASS" and summary["analysis_status"] == "DESCRIPTIVE_COMPLETE",
            "Official finite-step execution incomplete")
    require(summary["counters"] == {"snapshots": 7, "cases": 28, "attempted_evaluations": 112,
                                    "completed_evaluations": 112, "forward_calls": 672, "forward_rows": 31360},
            "The figure must retain all112 successful frozen endpoints")
    protocol_path, audit_path = args.protocol.resolve(), args.audit.resolve()
    require(sha(protocol_path) == summary["protocol_sha256"] == manifest["protocol_sha256"]
            and read_json(protocol_path) == registration and sha(run / "manifest.json") == summary["manifest_sha256"],
            "Protocol, manifest or registration binding changed")
    sources[str(protocol_path.relative_to(ROOT))] = sha(protocol_path)
    audit = read_json(audit_path)
    require(audit["status"] == "PASS" and audit["run"] == str(run.relative_to(ROOT)), "Independent audit is not PASS for this run")
    sources[str(audit_path.relative_to(ROOT))] = sha(audit_path)
    require(aggregate["row_count"] == 28, "Expected all28 saved cases")
    rows = {(row["snapshot_id"], row["probe_name"]): row for row in aggregate["rows"]}
    require(set(rows) == {(snapshot, batch) for snapshot in SHOTS for batch in BATCHES}, "Frozen full case inventory changed")
    audit_rows = {(row["snapshot_id"], row["probe_name"]): row for row in audit["observed"]["cases"]}
    require(set(audit_rows) == set(rows), "Independent audit case inventory differs")
    records, counts = {}, Counter()
    for snapshot in SHOTS:
        for batch in BATCHES:
            row = rows[snapshot, batch]
            original = saved(row["summary_path"])
            evaluation = row["evaluation"]
            require(original["evaluation"] == evaluation, "Original and aggregate evaluation differ")
            require(evaluation["status"] == evaluation["alpha0_reproduction"]["status"] == "PASS"
                    and evaluation["alphas"] == list(ALPHAS) and len(evaluation["evaluations"]) == 4,
                    "Failed, unreproduced or missing endpoint cannot be silently omitted")
            endpoints = evaluation["evaluations"]
            require(all(endpoint["status"] == "PASS" and endpoint["alpha"] == alpha
                        for endpoint, alpha in zip(endpoints, ALPHAS)), "Endpoint status or alpha order changed")
            losses = [endpoint["loss_values"] for endpoint in endpoints]
            values = [loss["common_total"] for loss in losses]
            require(all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0. for value in values),
                    "Positive finite absolute losses required for the declared logarithmic ratio figure")
            independent = audit_rows[snapshot, batch]
            require(independent["case_status"] == independent["independent_reproduction_status"] == "PASS"
                    and independent["independent_decision"] == evaluation["decision"], "Audited case status or decision differs")
            require(all(math.isclose(value, audit_loss["total"], rel_tol=1e-10, abs_tol=1e-12)
                        for value, audit_loss in zip(values, independent["losses_from_saved_predictions"])),
                    "Independent saved-prediction loss and plotted absolute loss differ")
            records[snapshot, batch] = {"snapshot_id": snapshot, "probe_name": batch, "alphas": list(ALPHAS),
                                        "endpoint_statuses": [endpoint["status"] for endpoint in endpoints],
                                        "absolute_loss_values": losses, "alpha0_absolute_common_total": values[0],
                                        "common_total_ratios": [value / values[0] for value in values],
                                        "neutral_band_absolute": evaluation["neutral_band"],
                                        "decision": evaluation["decision"], "source_summary_path": row["summary_path"]}
            counts[evaluation["decision"]] += 1
    require(dict(counts) == audit["observed"]["decision_counts"]
            == {"NO_ALPHA1_OVERSHOOT_OBSERVED_ON_GRID": 20, "NOMINAL_INCREASE_SMALLER_DECREASE": 8},
            "Frozen complete decision inventory differs")
    metadata = {"schema": "p4-finite-step-saved-json-figure-v1", "source_sha256": sources,
                "plot_source_sha256": sha(Path(__file__)), "run_artifact_index_sha256": sha(index_path),
                "audit_status": audit["status"], "audit_check_count": audit["check_count"],
                "dimensions_px": [WIDTH, HEIGHT], "minimum_font_size_px": FONT,
                "minimum_font_size_at_499pt_width": FONT * 499. / WIDTH,
                "retained_case_count": 28, "retained_endpoint_count": 112, "failed_endpoint_count": 0,
                "missing_endpoint_count": 0, "snapshot_order": list(SHOTS), "probe_order": list(BATCHES),
                "alpha_order": list(ALPHAS), "plotted_quantity": "L(alpha)/L(alpha0) per fixed snapshot/batch",
                "absolute_loss_units": "Normalized dimensionless common total; exact absolute loss values retained for all112 endpoints below",
                "panel_axes": [{"probe": batch, "scale": "log", "low": axis[0], "high": axis[1], "ticks": list(axis[2])}
                               for batch, axis in zip(BATCHES, AXES)],
                "line_meaning": "Connecting segments are a visual guide between four measured points, not an evaluated continuous curve",
                "decision_counts": dict(counts), "all_endpoint_values": [records[snapshot, batch] for snapshot in SHOTS for batch in BATCHES],
                "new_model_forwards": 0, "new_derivative_traversals": 0, "new_optimizer_updates": 0,
                "new_training_updates": 0, "new_physics_steps": 0,
                "caption": CAPTION, "uncertainty": "No error bars; overlapping fixed train probes and repeated snapshots are not independent trials",
                "scope": "Supplemental saved-JSON display outside immutable run artifacts; no checkpoint/alpha selection, historical-cause or task/generalization claim"}
    drawing = make_figure(records)
    output.mkdir(parents=True, exist_ok=True)
    svg, png, sidecar = [output / ("p4_finite_step." + suffix) for suffix in ("svg", "png", "figure.json")]
    require(all(not path.exists() for path in (svg, png, sidecar)), "Never overwrite a supplemental finite-step figure")
    renderSVG.drawToFile(drawing, str(svg))
    content = svg.read_text()
    at = content.index(">", content.index("<svg")) + 1
    node = ET.Element("metadata")
    node.text = json.dumps(metadata, sort_keys=True, ensure_ascii=True, allow_nan=False)
    svg.write_text(content[:at] + ET.tostring(node, encoding="unicode") + content[at:])
    subprocess.run(["/opt/homebrew/bin/rsvg-convert", "-w", str(WIDTH), "-h", str(HEIGHT),
                    str(svg), "-o", str(png)], check=True)
    require(all(sha(ROOT / name) == digest for name, digest in sources.items()), "Frozen source changed while rendering")
    metadata["output_sha256"] = {str(path.relative_to(ROOT)): sha(path) for path in (svg, png)}
    sidecar.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "RENDERED_VISUAL_PENDING", "svg": str(svg.relative_to(ROOT)),
                      "png": str(png.relative_to(ROOT)), "metadata": str(sidecar.relative_to(ROOT)),
                      "dimensions_px": [WIDTH, HEIGHT], "source_count": len(sources),
                      "retained_endpoints": 112}, ensure_ascii=False))


if __name__ == "__main__":
    main()
