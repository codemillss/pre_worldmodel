"""Create the editable GitHub overview diagram; no measured paths or model calls."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/figures"
INK = "#172B3A"
GRAY = "#4F6470"
BLUE = "#174F84"
TEAL = "#276779"
RED = "#B23A36"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "svg.fonttype": "none",
    "svg.hashsalt": "pre-worldmodel-overview-v1",
    "savefig.facecolor": "white", "font.size": 10,
})
fig = plt.figure(figsize=(10, 3))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
fig.patch.set_facecolor("white")

ax.text(.035, .947, "PRE_WORLDMODEL  /  LEARNED DYNAMICS FOR ROBOTIC PUSHING",
        fontsize=9, color=TEAL, fontweight="bold", va="top")
ax.text(.035, .84, "Does better prediction improve the task?",
        fontsize=20.3, color=INK, fontweight="bold", va="top")


def card(x, width, accent, heading):
    ax.add_patch(FancyBboxPatch((x, .105), width, .55,
        boxstyle="round,pad=0.010,rounding_size=0.014",
        facecolor="#F7F9FB", edgecolor="#DFE6EA", linewidth=.9))
    ax.plot([x + .012, x + .052], [.607, .607], color=accent,
            lw=3.2, solid_capstyle="round")
    ax.text(x + .012, .567, heading, fontsize=8.2,
            color=accent, fontweight="bold", va="top")


card(.035, .285, TEAL, "WHY  /  HANDOFF READINESS")
card(.355, .295, BLUE, "HOW  /  PREDICT, PLAN, VERIFY")
card(.685, .28, RED, "SO WHAT  /  FROZEN EVIDENCE")

# An explicitly schematic part and target: no experimental coordinates or trajectory.
task = fig.add_axes([.048, .20, .132, .30])
task.set_xlim(-.05, 1.05); task.set_ylim(-.10, 1.05)
task.set_aspect("equal"); task.axis("off")
rect = np.array([[-.21, -.145], [.21, -.145], [.21, .145], [-.21, .145]])
angle = np.deg2rad(22)
rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
task.add_patch(Polygon(rect + [.68, .71], closed=True,
    facecolor="#FFF7D9", edgecolor="#967400", lw=1.2, linestyle=(0, (3, 2))))
task.add_patch(Polygon(rect @ rotation.T + [.40, .34], closed=True,
    facecolor="#DCEAF6", edgecolor=BLUE, lw=1.3))
task.add_patch(Circle((.075, .23), .065, facecolor="#CFD8DC", edgecolor=GRAY, lw=.9))
task.add_patch(FancyArrowPatch((-.01, .05), (.22, .15), arrowstyle="-|>",
    mutation_scale=9, color=RED, linewidth=1.0))
task.plot(.68, .71, "+", color="#967400", ms=5)

ax.text(.180, .451, "Target pose.\nStop. Hold.", fontsize=12.0,
        color=INK, fontweight="bold", va="top", linespacing=1.42)
ax.text(.047, .160, "Readiness, not a pass-by.", fontsize=8.8, color=GRAY)

cx = .503
ax.text(cx, .470, "Learned dynamics", fontsize=11.8, color=INK,
        fontweight="bold", ha="center", va="top")
ax.text(cx, .406, "one-step vs multi-step training", fontsize=8.5,
        color=GRAY, ha="center", va="top")
ax.add_patch(FancyArrowPatch((cx, .363), (cx, .322), arrowstyle="-|>",
    mutation_scale=9, color=BLUE, linewidth=1.0))
ax.text(cx, .301, "Candidate planning", fontsize=11.4,
        color=INK, ha="center", va="top")
ax.add_patch(FancyArrowPatch((cx, .249), (cx, .209), arrowstyle="-|>",
    mutation_scale=9, color=BLUE, linewidth=1.0))
ax.text(cx, .190, "Actual outcome checks", fontsize=11.2,
        color=INK, ha="center", va="top")

ax.text(.697, .464, "Some position prediction\nimproved.", fontsize=11.5,
        color=BLUE, fontweight="bold", va="top", linespacing=1.40)
ax.text(.697, .312, "Handoff improvement\nwas not confirmed.", fontsize=11.4,
        color=RED, fontweight="bold", va="top", linespacing=1.40)
ax.text(.697, .125, "Fixed simulated conditions", fontsize=8.0, color=GRAY)
ax.text(.035, .045, "Schematic / MuJoCo simulation", fontsize=8.2,
        color=GRAY, va="center")
ax.text(.965, .045, "Prediction  ≠  selection  ≠  task outcome", fontsize=8.2,
        color=GRAY, ha="right", va="center")

OUT.mkdir(parents=True, exist_ok=True)
png = OUT / "project_overview.png"
svg = OUT / "project_overview.svg"
fig.savefig(png, dpi=150, metadata={"Software": "Matplotlib; explanatory schematic"})
fig.savefig(svg, metadata={"Date": None, "Creator": "Matplotlib; explanatory schematic"})
plt.close(fig)

# Native SVG text is retained for editing and accessible standalone viewing.
namespace = "http://www.w3.org/2000/svg"
ET.register_namespace("", namespace)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
tree = ET.parse(svg)
element = tree.getroot()
element.set("role", "img")
element.set("aria-labelledby", "overview-title overview-description")
title = ET.Element(f"{{{namespace}}}title", {"id": "overview-title"})
title.text = "Learned dynamics for part-pushing alignment: why, how, and observed meaning"
description = ET.Element(f"{{{namespace}}}desc", {"id": "overview-description"})
description.text = (
    "Explanatory MuJoCo simulation schematic, not an experimental trajectory. "
    "A part must reach its target pose and stop and hold for handoff readiness. "
    "Learned dynamics, trained with one-step versus multi-step losses, are used for candidate planning; "
    "actual outcomes are checked separately. Some position prediction improved, "
    "but an improvement in handoff was not confirmed under the fixed simulated conditions."
)
element.insert(0, title)
element.insert(1, description)
tree.write(svg, encoding="utf-8", xml_declaration=True)
assert len(list(element.iter(f"{{{namespace}}}text"))) >= 15
assert not list(element.iter(f"{{{namespace}}}image"))
print(json.dumps({
    "scientific_data_changed": False, "additional_physics_intervals": 0,
    "files": [{"path": p.relative_to(ROOT).as_posix(),
               "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
               "bytes": p.stat().st_size} for p in (svg, png)],
}, indent=2))
