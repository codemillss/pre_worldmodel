"""Render only recorded nominal preflight substeps; no models or physics.

Uses existing ReportLab and rsvg-convert because this workspace has no
matplotlib installation. The SVG embeds source hashes as metadata.
"""
from pathlib import Path
import hashlib
import json
import math
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from reportlab.graphics import renderSVG
from reportlab.graphics.shapes import Drawing, Line, Path as Curve, Rect, String
from reportlab.lib.colors import HexColor


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/p4-data-preflight-20261013-01"
OUT = ROOT / "reports/figures"
STEM = "p4_data_preflight_velocity"
NAMES = {
    "side_pass": "Side pass",
    "central_approach": "Central approach",
    "eccentric_approach": "Eccentric approach",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_recorded_curves():
    if not (RUN / "summary.json").is_file():
        raise RuntimeError("Final summary is not present; wait for the bounded runner to finish")
    summary = json.loads((RUN / "summary.json").read_text())
    registration = json.loads((RUN / "registration.json").read_text())
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in (RUN / "summary.json", RUN / "registration.json")}
    plans = {p["episode_id"]: p for p in registration["plans"]}
    curves = {}
    for row in summary["episodes"]:
        if row["role"] != "core":
            continue
        plan = plans[row["label"]]
        assert row["Y"] == plan["y_condition"] and row["S"] == plan["stop_condition"]
        folder = RUN / row["dataset"]
        manifest_path = folder / "manifest.json"
        assert sha(manifest_path) == row["manifest_sha256"]
        manifest = json.loads(manifest_path.read_text())
        assert len(manifest["episodes"]) == 1
        entry = manifest["episodes"][0]
        assert entry["episode_id"] == plan["episode_id"] and entry["family_id"] == plan["family_id"]
        npz = folder / entry["npz"]
        assert sha(npz) == entry["npz_sha256"]
        bindings[str(manifest_path.relative_to(ROOT))] = sha(manifest_path)
        bindings[str(npz.relative_to(ROOT))] = sha(npz)
        with np.load(npz, allow_pickle=False) as a:
            assert a["state"].shape[1] == a["substep_states"].shape[1] == 10
            elapsed = np.r_[0., a["substep_times"] - a["sim_time"][0]]
            states = np.vstack([a["state"][0], a["substep_states"]])
            assert len(elapsed) == len(states) and np.all(np.diff(elapsed) > 0)
            assert elapsed[-1] <= 2. + 1e-9
            # Include the measured initial state and every stored substep. No
            # endpoint interpolation, decimation, or invalid-transition filtering.
            curves[(plan["family_rank"], row["Y"], row["S"])] = {
                "time_s": elapsed, "pusher_vy_mm_s": states[:, 9] * 1000.,
                "object_xy_speed_mm_s": np.linalg.norm(states[:, 3:5], axis=1) * 1000.,
                "actions": len(a["actual_dt"]), "invalid_actions": int((~a["transition_valid"]).sum()),
                "substeps": len(a["substep_times"]), "terminal_reason": row["terminal_reason"],
            }
    assert len(curves) <= 12
    return summary, registration, bindings, curves


def main():
    summary, registration, bindings, curves = load_recorded_curves()
    width, height = 2400, 2070
    drawing = Drawing(width, height)
    c = HexColor
    ink, muted, grid = "#182F40", "#536D7C", "#D8E1E7"
    pusher_color, object_color = "#12678A", "#C55F21"
    drawing.add(Rect(0, 0, width, height, fillColor=c("#FFFFFF"), strokeColor=None))
    drawing.add(String(90, 1998, "Preflight: recorded physical velocities (no model predictions)",
                       fontSize=43, fillColor=c(ink)))
    drawing.add(String(90, 1941, "3 initial-condition families x 4 variants | Nominal physics substeps | Core only; warmup excluded",
                       fontSize=31, fillColor=c(muted)))
    legend_y = 1879
    legends = [(90, pusher_color, None, "Pusher Y velocity"),
               (655, object_color, None, "Object XY speed"),
               (1205, ink, None, "S0 dispersed zeros"),
               (1785, ink, [17, 10], "S1 continuous zeros")]
    for x, color, dash, label in legends:
        legend_line = Curve(strokeWidth=4, strokeColor=c(color), strokeDashArray=dash, fillColor=None)
        legend_line.moveTo(x, legend_y)
        legend_line.lineTo(x + 70, legend_y)
        drawing.add(legend_line)
        drawing.add(String(x + 85, legend_y - 10, label, fontSize=30, fillColor=c(ink)))
    drawing.add(Rect(90, 1806, 43, 28, fillColor=c("#E7EDF1"), strokeColor=None))
    drawing.add(String(150, 1808, "Gray band: S1 zero-command interval, 1.0-1.5 s. Zero command does not overwrite velocity.",
                       fontSize=30, fillColor=c(muted)))

    values = [v[key] for v in curves.values() for key in ("pusher_vy_mm_s", "object_xy_speed_mm_s")]
    finite = np.concatenate([v[np.isfinite(v)] for v in values]) if values else np.array([0.])
    low = min(0., math.floor(float(finite.min()) / 5) * 5) if len(finite) else 0.
    high = max(5., math.ceil(float(finite.max()) / 5) * 5) if len(finite) else 5.
    tick_step = 5. if high - low <= 40 else 10.
    family_plans = [p for p in registration["plans"] if p["y_condition"] == "Y0" and p["stop_condition"] == "S0"]
    for row, plan in enumerate(family_plans):
        for col, y_label in enumerate(("Y0", "Y1")):
            x, y = 140 + 1190 * col, 1277 - 511 * row
            w, h = 965, 356
            px = lambda t: x + t * w / 2.
            py = lambda v: y + (v - low) / (high - low) * h
            y_command = 14 if y_label == "Y0" else 29
            heading = f"Family {plan['family_rank']} | {NAMES[plan['geometry']]} | {y_label}, command {y_command} mm/s"
            drawing.add(String(x, y + h + 65, heading, fontSize=30, fillColor=c(ink)))
            drawing.add(String(x, y + h + 25, "Velocity / speed (mm/s)", fontSize=28, fillColor=c(muted)))
            drawing.add(Rect(px(1.), y, px(1.5) - px(1.), h, fillColor=c("#E7EDF1"), strokeColor=None))
            for tick in np.arange(low, high + tick_step / 2, tick_step):
                drawing.add(Line(x, py(tick), x + w, py(tick), strokeColor=c(grid), strokeWidth=1))
                drawing.add(String(x - 17, py(tick) - 9, f"{tick:g}", fontSize=27,
                                   textAnchor="end", fillColor=c(muted)))
            for tick in (0., .5, 1., 1.5, 2.):
                drawing.add(Line(px(tick), y, px(tick), y + h, strokeColor=c(grid), strokeWidth=1))
                drawing.add(String(px(tick), y - 34, f"{tick:g}", fontSize=27,
                                   textAnchor="middle", fillColor=c(muted)))
            drawing.add(Line(x, y, x + w, y, strokeColor=c(ink), strokeWidth=1.4))
            drawing.add(Line(x, y, x, y + h, strokeColor=c(ink), strokeWidth=1.4))
            notes = []
            for stop_label, dash in (("S0", None), ("S1", [17, 10])):
                data = curves.get((plan["family_rank"], y_label, stop_label))
                if data is None:
                    notes.append(f"{stop_label}: NOT RECORDED")
                    continue
                for key, color in (("pusher_vy_mm_s", pusher_color), ("object_xy_speed_mm_s", object_color)):
                    curve = Curve(strokeColor=c(color), strokeWidth=3.7, strokeDashArray=dash, fillColor=None)
                    started = False
                    for t, value in zip(data["time_s"], data[key]):
                        if not np.isfinite(value):
                            started = False
                            continue
                        if started:
                            curve.lineTo(px(t), py(value))
                        else:
                            curve.moveTo(px(t), py(value))
                            started = True
                    drawing.add(curve)
                nonfinite = sum(int((~np.isfinite(data[key])).sum()) for key in ("pusher_vy_mm_s", "object_xy_speed_mm_s"))
                if data["actions"] != 20 or data["invalid_actions"] or nonfinite:
                    notes.append(f"{stop_label}: {data['actions']}/20 actions; invalid {data['invalid_actions']}; nonfinite {nonfinite}")
            if row == len(family_plans) - 1:
                drawing.add(String(x + w / 2, y - 74, "Time from core start (s)", fontSize=28,
                                   textAnchor="middle", fillColor=c(muted)))
            if notes:
                drawing.add(String(x + 16, y + h - 33, " | ".join(notes), fontSize=26, fillColor=c("#A53232")))

    drawing.add(String(90, 125, f"Observed core trajectories: {len(curves)}/12 | Run status: {summary['status']} | No training or alignment-success evaluation",
                       fontSize=29, fillColor=c(ink)))
    drawing.add(String(90, 75, "S0 zero commands: 0.5-0.6, 0.7-0.8, 0.9-1.0, 1.1-1.2, 1.3-1.4 s. Curves show all stored substeps.",
                       fontSize=28, fillColor=c(muted)))
    drawing.add(String(90, 29, "Source: runs/p4-data-preflight-20261013-01 | Shared vertical scale | Data and source hashes embedded in SVG metadata",
                       fontSize=27, fillColor=c(muted)))
    OUT.mkdir(parents=True, exist_ok=True)
    svg, png = OUT / (STEM + ".svg"), OUT / (STEM + ".png")
    renderSVG.drawToFile(drawing, str(svg))
    bindings[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    metadata = {"source_sha256": bindings, "recorded_core_trajectories": len(curves),
                "recorded_substeps": sum(v["substeps"] for v in curves.values()),
                "axes_mm_s": [low, high], "dimensions_px": [width, height],
                "minimum_font_size_px": 26, "new_physics": 0, "new_learning": 0,
                "plot_scope": "Recorded nominal preflight core initial state plus every substep; no learned predictions"}
    text = svg.read_text()
    marker = text.index(">", text.index("<svg")) + 1
    # Metadata is machine-readable and does not change the rendered figure.
    node = ET.Element("metadata")
    node.text = json.dumps(metadata, ensure_ascii=True, sort_keys=True)
    svg.write_text(text[:marker] + ET.tostring(node, encoding="unicode") + text[marker:])
    subprocess.run(["/opt/homebrew/bin/rsvg-convert", "-w", str(width), "-h", str(height), str(svg), "-o", str(png)], check=True)
    assert all(sha(ROOT / path) == expected for path, expected in bindings.items())
    print(json.dumps({"png": str(png), "svg": str(svg), "recorded_core_trajectories": len(curves),
                      "recorded_substeps": metadata["recorded_substeps"], "status": summary["status"],
                      "png_sha256": sha(png), "svg_sha256": sha(svg)}))


if __name__ == "__main__":
    main()
