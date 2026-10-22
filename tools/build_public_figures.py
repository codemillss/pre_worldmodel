"""Render a standalone E2 success figure from saved rows; no model or physics calls."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "artifacts/results/E2/rows.json"
rows = json.loads(source.read_text())
groups = ["G", "R", "S1", "SK"]
counts = []
for group in groups:
    selected = [r for r in rows if r["controller_id"] == group
                or r["controller_id"].startswith(group + "-")]
    counts.append((sum(r["success"] for r in selected), len(selected)))
assert counts == [(8, 12), (0, 12), (13, 36), (12, 36)]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(8, 7.05))
fig.subplots_adjust(left=.10, right=.98, top=.855, bottom=.44)
fig.patch.set_facecolor("white")
colors = ["#276779", "#98A9B0", "#174f84", "#b23a36"]
ax.bar(range(4), [100 * a / b for a, b in counts], color=colors, width=.57)
for index, (success, total) in enumerate(counts):
    ax.text(index, 100 * success / total + 3, f"{success}/{total}",
            ha="center", fontsize=14, fontweight="bold", color="#172B3A")
ax.set_ylim(0, 100)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_ylabel("Handoff-ready executions (%)", fontsize=11)
ax.set_xticks(range(4), [
    "G\nGeometric\nfeedback", "R\nFeature-linear\nmodel + planner",
    "S1\nOne-step MLP\n+ planner", "SK\nOne-step + rollout\nMLP + planner",
], fontsize=10.2, linespacing=1.35)
ax.tick_params(axis="x", length=0, pad=10)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, alpha=.18)
ax.set_axisbelow(True)

ink, gray = "#172B3A", "#4F6470"
fig.text(.10, .956, "Better position prediction did not establish better handoff",
         fontsize=13.5, fontweight="bold", color=ink, va="top")
fig.text(.10, .915, "Frozen comparison · one rectangular part · fixed simulated conditions",
         fontsize=10.1, color=gray, va="top")
fig.text(.10, .293, "METHODS AND SHARED CONTROLS", fontsize=9.5,
         fontweight="bold", color=ink)
notes = [
    "G: direct geometric feedback; no learned dynamics or candidate search.",
    "R: feature-linear dynamics; selected regularization α = 0 (linear regression).",
    "S1: one-step MLP (multilayer perceptron). SK adds 0.1 × 5-step rollout loss.",
    "R / S1 / SK: 128 candidates, 5-step plan; execute 0.1 s, then reobserve.",
]
for y, note in zip((.263, .238, .213, .188), notes):
    fig.text(.10, y, note, fontsize=9.05, color=gray)
fig.add_artist(plt.Line2D([.10, .98], [.167, .167],
                         transform=fig.transFigure, color="#DCE4E8", lw=.8))
fig.text(.10, .142, "WHAT THE COUNTS MEAN", fontsize=9.5,
         fontweight="bold", color=ink)
notes = [
    "12 independent initial conditions; S1 / SK: 3 training seeds per condition.",
    "Success: ≤5 mm, ≤3°, ≤0.01 m/s, ≤0.05 rad/s for a continuous 0.3 s.",
    "30 s limit; all failures retained. Descriptive proportions; no confidence intervals.",
]
for y, note in zip((.113, .088, .063), notes):
    fig.text(.10, y, note, fontsize=9.05, color=gray)

destination = ROOT / "assets/figures/handoff_success.png"
destination.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(destination, dpi=200, facecolor="white")
plt.close(fig)
manifest = {
    "schema": "public-success-figure-v2",
    "presentation_revision": "2026-10-22-standalone-method-definitions",
    "source": source.relative_to(ROOT).as_posix(),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "generator": Path(__file__).relative_to(ROOT).as_posix(),
    "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "figure": destination.relative_to(ROOT).as_posix(),
    "figure_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    "counts": {g: {"successes": a, "executions": b}
               for g, (a, b) in zip(groups, counts)},
    "independent_initial_conditions": 12,
    "neural_training_seeds": [11, 23, 47],
    "metric": "Observed handoff-ready execution proportion; failures retained",
    "error_bars": "Not shown; descriptive proportions are not independent 36-family estimates",
    "scientific_data_changed": False,
    "additional_physics_intervals": 0,
    "visual_review": "PENDING_DIRECT_VIEW_CURRENT_REVISION",
}
manifest_path = destination.with_suffix(".manifest.json")
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
print(destination.relative_to(ROOT))
