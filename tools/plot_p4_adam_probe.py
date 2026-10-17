"""Supplemental saved-JSON figures for the frozen virtual Adam diagnostic.

No tensors, learned models, optimizer, gradient calculation or physics are
loaded. Rendered source identities live outside the immutable run directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from reportlab.graphics import renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Line, Rect, String
from reportlab.lib.colors import HexColor


ROOT = Path(__file__).resolve().parents[1]
WIDTH = 1200
MINIMUM_FONT = 23
INK, MUTED, GRID = "#193144", "#516A7B", "#DCE5EB"
C0, R1, AUX = "#286DA8", "#C47A26", "#15846B"
SHOTS = ("common_initial", "C0_u0100", "C0_u0500", "C0_u1000",
         "R1_u0100", "R1_u0500", "R1_u1000")
SHOT_LABELS = ("Common+0", "C0+100", "C0+500", "C0+1000", "R1+100", "R1+500", "R1+1000")
BATCHES = ("historical_update1", "D11_zero5", "D11_transition", "D11_noncontact")
BATCH_LABELS = ("Schedule anchor", "D11 zero5", "D11 transition", "D11 noncontact")
GEOMETRY_CAPTION = (
    "TOTAL/KEEP 가상 변위의 정규화 국소 하강 코사인 cos(-g, Δθ). "
    "7개 저장 스냅샷과 같은 4개 train probe를 모두 표시한다. 양수는 해당 손실의 "
    "일차 근사에서 하강 방향임을 뜻한다. 28개 TOTAL/KEEP 방향 모두 base·weighted aux·total에 "
    "국소 하강이며, 유한 스텝 손실 감소·물리 성과·독립 시행의 근거는 아니다."
)
HISTORY_CAPTION = (
    "고정된 100-update 구간 10개 전체의 window-occurrence 가중평균. "
    "위는 C0/R1의 공통 base 목적함수, 아래는 R1의 가중 보조손실이며 y축은 로그축이다. "
    "손실은 정규화된 무차원 값이고, 변화하는 모델과 표본 배치의 학습 경로 기록이다. "
    "최종 모델의 전체 train 평균이 아니며 C0 보조손실은 계산되지 않았다."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def text(drawing, x, y, value, size=MINIMUM_FONT, color=INK, anchor="start"):
    require(size >= MINIMUM_FONT, "All figure text must remain at least 23 px")
    drawing.add(String(x, y, value, fontName="Helvetica", fontSize=size,
                       fillColor=HexColor(color), textAnchor=anchor))


def line(drawing, x1, y1, x2, y2, color=GRID, width=1., dash=None):
    drawing.add(Line(x1, y1, x2, y2, strokeColor=HexColor(color),
                     strokeWidth=width, strokeDashArray=dash))


def dot(drawing, x, y, color, radius=5.):
    drawing.add(Circle(x, y, radius, fillColor=HexColor(color), strokeColor=None))


def blank(height):
    drawing = Drawing(WIDTH, height)
    drawing.add(Rect(0, 0, WIDTH, height, fillColor=HexColor("#FFFFFF"), strokeColor=None))
    return drawing


def plot_line(drawing, values, *, left, bottom, width, height, low, high, color, dashed=False):
    require(len(values) == 10 and all(math.isfinite(value) and low <= value <= high for value in values),
            "All ten fixed blocks must fit the declared logarithmic axis")
    coordinates = [(left + width * index / 9.,
                    bottom + height * (math.log10(value) - math.log10(low)) / (math.log10(high) - math.log10(low)))
                   for index, value in enumerate(values)]
    for (x1, y1), (x2, y2) in zip(coordinates, coordinates[1:]):
        line(drawing, x1, y1, x2, y2, color, 3., [9, 6] if dashed else None)
    for x, y in coordinates:
        dot(drawing, x, y, color, 5.)


def log_axis(drawing, *, bottom, height, low, high, ticks):
    left, width = 145., 940.
    for value in ticks:
        y = bottom + height * (math.log10(value) - math.log10(low)) / (math.log10(high) - math.log10(low))
        line(drawing, left, y, left + width, y)
        text(drawing, left - 20., y - 7., f"{value:g}", color=MUTED, anchor="end")
    for index in range(10):
        x = left + width * index / 9.
        line(drawing, x, bottom, x, bottom - 8., "#7F94A3")
        text(drawing, x, bottom - 39., str((index + 1) * 100), color=MUTED, anchor="middle")
    return left, width


def history_figure(history):
    drawing = blank(1060)
    text(drawing, 45, 1010, "Recorded training path: all ten fixed 100-update blocks", 30)
    text(drawing, 45, 969, "Window-occurrence-weighted means; normalized dimensionless loss; logarithmic y axes.", 23, MUTED)
    text(drawing, 45, 914, "A  Shared base = one-step + 0.1 free-rollout", 26)
    line(drawing, 800, 925, 842, 925, C0, 3.)
    text(drawing, 853, 917, "C0")
    line(drawing, 960, 925, 1002, 925, R1, 3., [9, 6])
    text(drawing, 1013, 917, "R1")
    left, width = log_axis(drawing, bottom=598., height=268., low=.04, high=.5, ticks=(.05, .1, .2, .5))
    values = {}
    for arm, color in (("C0", C0), ("R1", R1)):
        values[arm] = [block["losses"]["base"]["window_weighted_mean"] for block in history["arms"][arm]["blocks"]]
        plot_line(drawing, values[arm], left=left, bottom=598., width=width, height=268.,
                  low=.04, high=.5, color=color, dashed=arm == "R1")
    text(drawing, 45, 512, "B  R1 weighted auxiliary = 0.2 (relative teacher + 0.1 relative free)", 26)
    left, width = log_axis(drawing, bottom=205., height=258., low=.003, high=.05, ticks=(.004, .01, .02, .04))
    auxiliary = [block["losses"]["weighted_aux"]["window_weighted_mean"] for block in history["arms"]["R1"]["blocks"]]
    plot_line(drawing, auxiliary, left=left, bottom=205., width=width, height=258.,
              low=.003, high=.05, color=AUX)
    text(drawing, 615, 126, "Additional update (end of each fixed 100-update block)", 23, MUTED, "middle")
    text(drawing, 45, 78, "Changing models and sampled batches along training; not a final-model whole-train average.", 23, MUTED)
    text(drawing, 45, 40, "C0 auxiliary was not computed. No best checkpoint, uncertainty bars or task-effect claim.", 23, MUTED)
    return drawing, {"update_ends": list(range(100, 1001, 100)), "shared_base": values,
                     "R1_weighted_auxiliary": auxiliary, "aggregation": "window_weighted_mean",
                     "y_scales": {"shared_base": "log10", "R1_weighted_auxiliary": "log10"}}


def signed_fill(value):
    require(math.isfinite(value) and -1. <= value <= 1., "Cosine outside signed unit scale")
    target = C0 if value >= 0 else R1
    weight = abs(value)
    channels = [round(255. * (1. - weight) + int(target[index:index + 2], 16) * weight) for index in (1, 3, 5)]
    return "#" + "".join(f"{channel:02x}" for channel in channels)


def geometry_figure(rows):
    drawing = blank(1280)
    text(drawing, 45, 1230, "Virtual TOTAL / KEEP: all 7 snapshots x 4 fixed train probes", 29)
    text(drawing, 45, 1187, "KEEP retains saved m, v and age. Columns show snapshots (+ additional updates).", 23, MUTED)
    text(drawing, 45, 1149, "cos(-g, delta): positive = local descent; negative = local ascent. Signed color scale [-1, +1].", 23, MUTED)
    data = {}
    for reference, title, top, bottom in (("base", "A  Reference: shared base gradient", 1096., 805.),
                                          ("aux", "B  Reference: weighted auxiliary gradient", 746., 455.),
                                          ("total", "C  Reference: full total gradient", 396., 105.)):
        text(drawing, 45, top, title, 26)
        left, width, height = 250., 870., 232.
        col_width, row_height = width / 7., height / 4.
        for column, label in enumerate(SHOT_LABELS):
            text(drawing, left + (column + .5) * col_width, top - 43., label, 23, INK, "middle")
        values = []
        for row_index, (batch, label) in enumerate(zip(BATCHES, BATCH_LABELS)):
            center_y = bottom + (3.5 - row_index) * row_height
            text(drawing, left - 20., center_y - 8., label, 23, INK, "end")
            row_values = []
            for column, snapshot in enumerate(SHOTS):
                entry = rows[snapshot, batch]["summary"]["directions"][3]
                require(entry["objective"] == "total" and entry["mode"] == "KEEP", "Direction index differs")
                geometry = entry["loss_geometry"][reference]
                require(geometry["relation"] == "DESCENT", "TOTAL/KEEP descent inventory changed")
                value = float(geometry["descent_cosine"])
                row_values.append(value)
                drawing.add(Rect(left + column * col_width, bottom + (3 - row_index) * row_height,
                                 col_width, row_height, fillColor=HexColor(signed_fill(value)),
                                 strokeColor=HexColor("#FFFFFF"), strokeWidth=3))
                text(drawing, left + (column + .5) * col_width, center_y - 8., f"{value:.3f}", 24, INK, "middle")
            values.append(row_values)
        for boundary in (1, 4):
            line(drawing, left + boundary * col_width, bottom, left + boundary * col_width,
                 bottom + height, "#A6BAC7", 2.)
        data[reference] = values
    text(drawing, 45, 63, "All 28 TOTAL/KEEP directions descend locally on base, auxiliary and total; no finite-step loss was evaluated.", 23, MUTED)
    text(drawing, 45, 25, "Saved JSON only. Repeated snapshots and overlapping train probes are not independent trials.", 23, MUTED)
    return drawing, {"row_order": list(BATCHES), "column_order": list(SHOTS),
                     "objective": "total", "mode": "KEEP", "reference_order": ["base", "aux", "total"],
                     "descent_cosine": data, "color_scale": [-1., 1.],
                     "definition": "cos(-reference_gradient, virtual_displacement); positive means local first-order descent"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--audit", type=Path, default=ROOT / "reports/p4_adam_probe_independent_review.json")
    parser.add_argument("--output-directory", type=Path, default=ROOT / "reports/figures")
    args = parser.parse_args()
    run, audit_path, output = args.run.resolve(), args.audit.resolve(), args.output_directory.resolve()
    index_path = run / "artifact_hashes.json"
    index = read_json(index_path)["sha256"]
    sources = {str(index_path.relative_to(ROOT)): sha(index_path)}

    def saved(name):
        path = run / name
        require(index.get(name) == sha(path), "Frozen run source binding differs: " + name)
        sources[str(path.relative_to(ROOT))] = sha(path)
        return read_json(path)

    aggregate, history, summary, manifest, registration = (saved(name) for name in
        ("aggregate.json", "history_summary.json", "summary.json", "manifest.json", "registration.json"))
    require(summary["execution_status"] == "PASS" and history["status"] == "DESCRIPTIVE_COMPLETE",
            "Frozen run or saved history incomplete")
    require(aggregate["row_count"] == 28, "Expected all 28 saved probe appearances")
    rows = {(row["snapshot_id"], row["probe_name"]): row for row in aggregate["rows"]}
    require(set(rows) == {(snapshot, batch) for snapshot in SHOTS for batch in BATCHES},
            "Full snapshot/probe inventory differs")
    for identity, row in rows.items():
        require(saved(row["summary_path"]) == row["summary"], "Aggregate and per-probe summary differ: " + str(identity))
    for arm in ("C0", "R1"):
        blocks = history["arms"][arm]["blocks"]
        require(len(blocks) == 10 and [(block["update_start"], block["update_end"]) for block in blocks]
                == [(start, start + 99) for start in range(1, 1001, 100)], "Ten fixed history blocks required")
    audit = read_json(audit_path)
    sources[str(audit_path.relative_to(ROOT))] = sha(audit_path)
    require(audit["status"] == "PASS" and audit["run"] == str(run.relative_to(ROOT)),
            "Independent audit status or run identity differs")
    audited = audit["observed"]["per_direction"]
    require(len(audited) == 168 and all(row["base_conclusion_eligible"]
            and all(row["loss_certificates"][reference]["rounding_stable_sign"] in ("DESCENT", "ASCENT")
                    for reference in ("base", "aux", "total")) for row in audited),
            "All 168 independent precision conclusions must remain eligible and sign-stable")
    for row in audited:
        if row["objective"] == "total" and row["mode"] == "KEEP":
            production = rows[row["snapshot_id"], row["probe_name"]]["summary"]["directions"][3]
            for reference in ("base", "aux", "total"):
                require(abs(row["loss_geometry"][reference]["descent_cosine"]
                            - production["loss_geometry"][reference]["descent_cosine"]) < 1e-12,
                        "Audited and plotted TOTAL/KEEP geometry differ")
    metadata = {
        "schema": "p4-adam-probe-saved-json-figure-v1", "source_sha256": sources,
        "plot_source_sha256": sha(Path(__file__)), "run_artifact_index_sha256": sha(index_path),
        "audit_status": audit["status"], "audit_check_count": audit["check_count"],
        "snapshot_order": list(SHOTS), "batch_order": list(BATCHES),
        "new_model_forwards": 0, "new_derivative_traversals": 0,
        "new_optimizer_updates": 0, "new_physics_steps": 0,
        "minimum_font_size_px": MINIMUM_FONT, "minimum_font_size_at_499pt_width": MINIMUM_FONT * 499. / WIDTH,
        "uncertainty": "No error bars; changing-weight history and repeated fixed train probes are not independent trials",
        "scope": "Supplemental saved-JSON rendering outside immutable run artifacts; no model selection or finite-effect claim",
    }
    output.mkdir(parents=True, exist_ok=True)
    for suffix, maker, inputs, caption in (("history", history_figure, history, HISTORY_CAPTION),
                                           ("keep_geometry", geometry_figure, rows, GEOMETRY_CAPTION)):
        drawing, data = maker(inputs)
        svg, png, figure_json = [output / f"p4_adam_probe_{suffix}.{extension}" for extension in ("svg", "png", "figure.json")]
        require(all(not path.exists() for path in (svg, png, figure_json)), "Never overwrite supplemental Adam figures")
        item = {**metadata, "layout": suffix, "dimensions_px": [int(drawing.width), int(drawing.height)],
                "caption": caption, "plotted_data": data}
        renderSVG.drawToFile(drawing, str(svg))
        content = svg.read_text()
        at = content.index(">", content.index("<svg")) + 1
        node = ET.Element("metadata")
        node.text = json.dumps(item, sort_keys=True, ensure_ascii=True, allow_nan=False)
        svg.write_text(content[:at] + ET.tostring(node, encoding="unicode") + content[at:])
        subprocess.run(["/opt/homebrew/bin/rsvg-convert", "-w", str(int(drawing.width)),
                        "-h", str(int(drawing.height)), str(svg), "-o", str(png)], check=True)
        require(all(sha(ROOT / path) == digest for path, digest in sources.items()),
                "Frozen source changed during rendering")
        item["output_sha256"] = {str(path.relative_to(ROOT)): sha(path) for path in (svg, png)}
        figure_json.write_text(json.dumps(item, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps({"status": "RENDERED_VISUAL_PENDING", "svg": str(svg.relative_to(ROOT)),
                          "png": str(png.relative_to(ROOT)), "metadata": str(figure_json.relative_to(ROOT)),
                          "dimensions_px": item["dimensions_px"], "source_count": len(sources)}))


if __name__ == "__main__":
    main()
