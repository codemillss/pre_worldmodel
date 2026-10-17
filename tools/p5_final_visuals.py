"""Bound, offline figures and raw-pose replay; never a physics renderer.

Run extraction with the robot interpreter. Pillow drawing can run in a separate
bundled interpreter, so no robot packages are installed or changed. Every drawn
pose is an actual recorded action-boundary observation; there is no state
interpolation, extra dynamics, successful-case selection, or trajectory repair.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
CONTROLLERS = ("G", "R", "S1-11", "S1-23", "S1-47", "SK-11", "SK-23", "SK-47")
STRATA = ("translation", "rotation", "offset")
COLORS = ("#2463a2", "#c46d17", "#24844b", "#ac427b", "#7355a5", "#0d8585", "#b64039", "#726032")
DEFAULT_RENDERER = Path(sys.executable)
MAX_OUTPUT_BYTES = 64 * 1024**2
REPLAY_RULE = "smallest_family_id_per_stratum_all_eight_controllers"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bind(path, bindings, expected=None):
    path = Path(path).resolve()
    actual = sha(path)
    require(expected is None or actual == expected, "Changed visual source: " + str(path))
    require(str(path) not in bindings or bindings[str(path)] == actual, "Source changed during extraction")
    bindings[str(path)] = actual
    return path


def _state(value):
    require(len(value) == 10, "Recorded state10 required")
    return [float(x) if math.isfinite(float(x)) else None for x in value]


def select_replay_families(families):
    require(len(families) == 12 and len({row["family_id"] for row in families}) == 12,
            "All twelve distinct final families required")
    selected = {}
    for stratum in STRATA:
        group = sorted((row for row in families if row["stratum"] == stratum), key=lambda row: row["family_id"])
        require(len(group) == 4, "Exactly four families per stratum required")
        selected[stratum] = group[0]["family_id"]
    return selected


def _extract_cell(folder, cell, bindings, *, expected_study_sha256=None):
    """Decode one raw cell, reduce it, and return no StepResult references."""
    import numpy as np
    from push_align import final_recording, final_recording_columns
    worker_receipt_path = bind(folder / "worker_result_receipt.json", bindings)
    worker_path = bind(folder / "worker_result.json", bindings, read(worker_receipt_path)["sha256"])
    worker = read(worker_path)
    require(worker.get("status") in {"ACCEPT_COMPLETED_RECORDING", "ACCEPT_COMPLETED_TASK_FAILURE"},
            "Unresolved cell cannot enter final visuals")
    manifest_path = bind(folder / "manifest.json", bindings, worker["record_manifest_sha256"])
    manifest = read(manifest_path)
    review_path = bind(folder / "review.json", bindings, worker["review_sha256"])
    review = read(review_path)
    require(review.get("accepted") is True and review.get("status") == worker["status"]
            and review.get("manifest_sha256") == worker["record_manifest_sha256"],
            "Cell has no accepted source-bound independent raw/task review")
    row = worker["row"]
    require(isinstance(review.get("metrics"), dict) and all(row.get(k) == v for k,v in review["metrics"].items()),
            "Visual metrics differ from reviewed raw outcome")
    for key in ("family_id", "controller_id", "episode_id", "stratum", "cell_id"):
        require(row[key] == cell[key], "Visual cell differs from frozen expected identity: " + key)
    require(row["execution_status"] == "completed", "Incomplete execution cannot be drawn as final")
    require(manifest["execution"]["controller_id"] == cell["controller_id"]
            and manifest["execution"]["case_id"] == cell["family_id"], "Raw execution identity differs")
    if expected_study_sha256 is not None:
        require(manifest.get("identity", {}).get("final_manifest_sha256") == expected_study_sha256,
                "Cell belongs to a different frozen study")
    raw = manifest["raw"]
    require(isinstance(raw, dict) and raw.get("files_sha256"), "Original E2 complete raw required")
    codec_name = manifest.get("raw_codec_module")
    require(codec_name in {"push_align.final_recording", "push_align.final_recording_columns"},
            "Unrecognized raw codec")
    codec = final_recording_columns if codec_name.endswith("_columns") else final_recording
    for path, expected in raw["files_sha256"].items(): bind(path, bindings, expected)
    witness_path = Path(manifest["witness_path"])
    expected = manifest["files_sha256"].get(str(witness_path))
    require(expected is not None, "Witness lacks a saved SHA binding")
    bind(witness_path, bindings, expected)
    witness = final_recording.read_json_gzip(witness_path, expected_sha256=expected)
    snapshot, results = codec.read_raw(raw["prefix"], expected_sha256=raw["files_sha256"])
    require(len(results) == len(witness["intervals"]), "Raw/witness boundary count differs")
    require(witness["termination"] == row["termination"]
            and witness["execution_status"] == "completed", "Witness outcome differs from worker")
    initial_time = float(witness["initial"]["sim_time"])
    initial = np.asarray(witness["initial"]["state"], np.float64)
    if results:
        require(np.array_equal(initial, np.asarray(results[0].diagnostics["samples"][0]["state"], np.float64), equal_nan=True),
                "Raw initial state differs from recorded initial observation")
    samples = [{"relative_time_s": 0., "sim_time_s": initial_time, "state10": _state(initial),
                "observation_valid": bool(witness["initial"]["observation_valid"]),
                "phase": witness["initial"]["task"]["phase"],
                "held_time_s": float(witness["initial"]["task"]["held_time_s"])}]
    for index, (step, boundary) in enumerate(zip(results, witness["intervals"])):
        require(boundary["index"] == index, "Noncontiguous raw boundary index")
        require(np.array_equal(step.observation.state, np.asarray(boundary["state_after"], np.float64), equal_nan=True)
                and abs(step.observation.sim_time - boundary["sim_time_after"]) <= 1e-9,
                "Raw observed pose/time differs from saved witness")
        relative = float(step.observation.sim_time - initial_time)
        require(relative > samples[-1]["relative_time_s"], "Raw boundary times must increase")
        samples.append({"relative_time_s": relative, "sim_time_s": float(step.observation.sim_time),
            "state10": _state(step.observation.state), "observation_valid": bool(step.observation.observation_valid),
            "transition_valid": bool(step.transition_valid), "geometry_valid": boundary["geometry"]["pass"],
            "phase": boundary["task_after"]["phase"], "held_time_s": float(boundary["task_after"]["held_time_s"])})
    require(abs(float(row["elapsed_s"]) - samples[-1]["relative_time_s"]) <= 1e-7,
            "Actual raw terminal time differs from worker elapsed time")
    compact = {**cell, "termination": row["termination"], "success": row["success"],
               "elapsed_s": row["elapsed_s"], "samples": samples,
               "metrics": {key: row.get(key) for key in (
                   "final_position_error_m", "final_yaw_error_rad", "final_linear_speed_m_s",
                   "final_angular_speed_rad_s", "metric_state_origin", "metric_state_time_s",
                   "final_observation_available")},
               "source_worker_sha256": bindings[str(worker_path)],
               "source_record_manifest_sha256": bindings[str(manifest_path)]}
    # The caller iterates 96 raw files sequentially, never retaining diagnostics.
    del results, snapshot, witness
    return compact


def extract_run(run):
    run = Path(run).resolve(); bindings = {}
    manifest_path = bind(run / "manifest.json", bindings)
    freeze = read(bind(run / "freeze_receipt.json", bindings))
    require(freeze["sha256"] == bindings[str(manifest_path)], "Frozen study manifest changed")
    manifest = read(manifest_path)
    require(manifest.get("status") == "FROZEN_BEFORE_FINAL_OUTCOMES", "Frozen study required")
    selected = select_replay_families(manifest["families"])
    specification = manifest.get("visualization")
    require(isinstance(specification, dict) and specification.get("selection_rule") == REPLAY_RULE
            and specification.get("selected_families") == selected
            and specification.get("controllers") == list(CONTROLLERS)
            and specification.get("fps") == 5 and specification.get("postterminal_tail_s") == 1.,
            "Replay clips must be selected in the study manifest before outcomes")
    expected = manifest["E2"]
    require(len(expected) == 96 and len({row["cell_id"] for row in expected}) == 96,
            "Full expected96 cell inventory required")
    require({(row["family_id"], row["controller_id"]) for row in expected}
            == {(family["family_id"], controller) for family in manifest["families"] for controller in CONTROLLERS},
            "Expected cells omit or duplicate a family/controller")
    cells = []
    for cell in expected:
        cells.append(_extract_cell(run / "E2" / cell["family_id"] / cell["controller_id"], cell, bindings,
                                   expected_study_sha256=bindings[str(manifest_path)]))
        gc.collect()
    statistics_path = bind(run / "statistics" / "summary.json", bindings)
    statistics = read(statistics_path)
    require(statistics.get("status") == "COMPLETE_EXPECTED_E2_DENOMINATORS"
            and statistics.get("episode_count") == 96 and statistics.get("family_count") == 12
            and statistics.get("controller_count") == 8
            and statistics.get("bootstrap", {}).get("replicates") == 2000
            and statistics.get("bootstrap", {}).get("seed") == 7711, "Complete frozen final statistics required")
    for path, expected_hash in bindings.items(): require(sha(path) == expected_hash, "Source changed during extraction")
    return {"schema": "p5-visual-compact-actual-boundary-v1", "source_bindings_sha256": bindings,
            "source_frozen_manifest_sha256": bindings[str(manifest_path)],
            "families": manifest["families"], "selected_families": selected, "controllers": list(CONTROLLERS),
            "visualization": specification,
            "physical_config": manifest["physical_config"], "statistics": statistics, "cells": cells,
            "fixture_only": bool(manifest.get("fixture_only", False)),
            "render_scope": "Recorded actual action-boundary poses, zero-order hold; no interpolation, no new physics or photorealistic simulator render"}


def held_sample(cell, clock):
    """Latest actual measurement at/before frame time; no interpolation."""
    require(math.isfinite(clock) and clock >= 0, "Nonnegative replay clock required")
    samples = cell["samples"]
    times = [sample["relative_time_s"] for sample in samples]
    require(bool(times) and times[0] == 0. and all(b > a for a, b in zip(times, times[1:])),
            "Explicit monotonically increasing measurements required")
    index = max(0, bisect_right(times, clock + 1e-10) - 1)
    selected = samples[index]
    display_index = index
    while display_index >= 0 and any(value is None for value in samples[display_index]["state10"]):
        display_index -= 1
    return {"selected_index": index, "display_index": display_index,
            "selected": selected, "display": None if display_index < 0 else samples[display_index],
            "terminal": clock + 1e-9 >= cell["elapsed_s"],
            "nonfinite_selected_observation": display_index != index}


class BudgetFile:
    def __init__(self, path, budget):
        self.stream = Path(path).open("xb"); self.budget = budget; self.size = 0
    def write(self, value):
        end = self.stream.tell() + len(value); growth = max(0, end - self.size)
        self.budget.charge(growth)
        result = self.stream.write(value); self.size = max(self.size, end)
        return result
    def __getattr__(self, name): return getattr(self.stream, name)
    def fileno(self):
        # Force Pillow's Python write path; a C encoder writing to the raw file
        # descriptor would otherwise bypass the shared byte budget.
        raise io.UnsupportedOperation("Bounded output does not expose a raw descriptor")
    def __enter__(self): return self
    def __exit__(self, *args): self.stream.close()


class Budget:
    def __init__(self, folder, limit=MAX_OUTPUT_BYTES):
        self.folder = Path(folder); self.limit = limit
        self.lock = threading.Lock()
        require(type(limit) is int and 0 < limit <= MAX_OUTPUT_BYTES, "At most64MiB visual output allowed")
        self.used = sum(path.stat().st_size for path in self.folder.rglob("*") if path.is_file())
        self.charge(0)
    def charge(self, count):
        with self.lock:
            require(self.used + count <= self.limit, "Visual output budget reached; preserve existing files and stop")
            self.used += count
    def file(self, name): return BudgetFile(self.folder / name, self)
    def json(self, name, value):
        payload = (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
        with self.file(name) as stream: stream.write(payload)


def graphics():
    global Image, ImageDraw, ImageFont
    from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=32)
def font(size):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).is_file(): return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def text(draw, xy, value, size=15, color="#233044"):
    draw.text(xy, str(value), font=font(size), fill=color)


def fmt(value, factor=1, precision=2):
    return "n/a" if value is None else f"{value * factor:.{precision}f}"


def world_bounds(data):
    config = data["physical_config"]
    # Fit one shared, outcome-inclusive view to all96 raw traces plus the goal.
    # Include the full rotating part's corner radius, rather than hiding failed
    # workspace exits or wasting the plot on an unvisited workspace rectangle.
    padding = math.hypot(*[value / 2 for value in config["object"]["size_full_xyz_m"][:2]])
    gx, gy = config["task"]["goal_xy_m"]
    x0, x1, y0, y1 = gx - padding, gx + padding, gy - padding, gy + padding
    for cell in data["cells"]:
        for sample in cell["samples"]:
            state = sample["state10"]
            for xi, yi in ((0, 1), (6, 7)):
                if state[xi] is not None and state[yi] is not None:
                    x0, x1 = min(x0, state[xi] - padding), max(x1, state[xi] + padding)
                    y0, y1 = min(y0, state[yi] - padding), max(y1, state[yi] + padding)
    margin = .04 * max(x1 - x0, y1 - y0)
    return x0 - margin, x1 + margin, y0 - margin, y1 + margin


def mapping(box, bounds):
    left, top, right, bottom = box; x0, x1, y0, y1 = bounds
    scale = min((right - left) / (x1 - x0), (bottom - top) / (y1 - y0))
    cx, cy = (left + right) / 2, (top + bottom) / 2
    return lambda x, y: (cx + (x - (x0 + x1) / 2) * scale,
                          cy - (y - (y0 + y1) / 2) * scale), scale


def rectangle(state, config, transform):
    x, y, yaw = state[:3]; halfx, halfy = [value / 2 for value in config["object"]["size_full_xyz_m"][:2]]
    c, s = math.cos(yaw), math.sin(yaw)
    return [transform(x + dx * c - dy * s, y + dx * s + dy * c)
            for dx, dy in ((-halfx, -halfy), (halfx, -halfy), (halfx, halfy), (-halfx, halfy))]


def path_line(draw, cell, limit, indices, transform, color, width=2):
    line = []
    for sample in cell["samples"][:limit + 1]:
        state = sample["state10"]
        if state[indices[0]] is None or state[indices[1]] is None:
            if len(line) >= 2: draw.line(line, fill=color, width=width)
            line = []
        else: line.append(transform(state[indices[0]], state[indices[1]]))
    if len(line) >= 2: draw.line(line, fill=color, width=width)


def goal(draw, config, transform, scale):
    task = config["task"]; x, y = task["goal_xy_m"]; yaw = task["goal_yaw_rad"]
    points = rectangle([x, y, yaw], config, transform)
    draw.line(points + [points[0]], fill="#24844b", width=2)
    gx, gy = transform(x, y); radius = task["position_tolerance_m"] * scale
    draw.ellipse((gx - radius, gy - radius, gx + radius, gy + radius), outline="#24844b", width=1)
    draw.line((gx - 5, gy, gx + 5, gy), fill="#24844b", width=1)
    draw.line((gx, gy - 5, gx, gy + 5), fill="#24844b", width=1)


def endpoints_figure(data):
    image = Image.new("RGB", (1550, 1160), "#f7f9fc"); draw = ImageDraw.Draw(image)
    text(draw, (28, 18), "All96 outcomes:12 families x8 controllers", 25)
    text(draw, (28, 51), "Source metrics retain every task failure. P/yaw are final valid observation errors; n/a stays missing.", 15)
    lookup = {(cell["family_id"], cell["controller_id"]): cell for cell in data["cells"]}
    for column, controller in enumerate(CONTROLLERS): text(draw, (235 + column * 162, 83), controller, 19)
    for row, family in enumerate(data["families"]):
        top = 116 + row * 84
        text(draw, (20, top + 12), family["family_id"][-25:], 13)
        text(draw, (20, top + 34), family["stratum"], 14)
        for column, controller in enumerate(CONTROLLERS):
            cell = lookup[family["family_id"], controller]; left = 224 + column * 162
            draw.rounded_rectangle((left, top, left + 155, top + 76), radius=5,
                                   fill="#e3f3e8" if cell["success"] else "#f7e8df")
            text(draw, (left + 7, top + 4), cell["termination"][:21], 12)
            metrics = cell["metrics"]
            text(draw, (left + 7, top + 22), f"P {fmt(metrics['final_position_error_m'],1000)}mm / yaw {fmt(metrics['final_yaw_error_rad'],180/math.pi,1)}deg", 11)
            text(draw, (left + 7, top + 40), f"end {cell['elapsed_s']:.2f}s / state {fmt(metrics['metric_state_time_s'])}s", 11)
            text(draw, (left + 7, top + 57), str(metrics["metric_state_origin"])[:24], 10)
    return image


def trajectories_figure(data):
    image = Image.new("RGB", (1380, 1425), "#f7f9fc"); draw = ImageDraw.Draw(image)
    text(draw, (26, 15), "All actual object trajectories and endpoints", 25)
    text(draw, (26, 49), "Lines connect recorded action boundaries; circle=ready, x=task failure. Same world scale; goal outline green.", 14)
    for i, (controller, color) in enumerate(zip(CONTROLLERS, COLORS)):
        text(draw, (26 + i * 169, 77), controller, 17, color)
    bounds = world_bounds(data); lookup = {(c["family_id"], c["controller_id"]): c for c in data["cells"]}
    for index, family in enumerate(data["families"]):
        left, top = 20 + (index % 3) * 455, 115 + (index // 3) * 320
        draw.rounded_rectangle((left, top, left + 440, top + 305), radius=8, fill="white", outline="#d6dfeb")
        text(draw, (left + 12, top + 8), family["family_id"][-38:], 15)
        text(draw, (left + 12, top + 29), family["stratum"] + " | xy[m]", 12)
        transform, scale = mapping((left + 25, top + 55, left + 420, top + 282), bounds)
        goal(draw, data["physical_config"], transform, scale)
        for controller, color in zip(CONTROLLERS, COLORS):
            cell = lookup[family["family_id"], controller]; path_line(draw, cell, len(cell["samples"]) - 1, (0, 1), transform, color)
            finite = [s for s in cell["samples"] if s["state10"][0] is not None and s["state10"][1] is not None]
            if finite:
                state = finite[-1]["state10"]; x, y = transform(state[0], state[1])
                if cell["success"]: draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=2)
                else:
                    draw.line((x - 4, y - 4, x + 4, y + 4), fill=color, width=2)
                    draw.line((x - 4, y + 4, x + 4, y - 4), fill=color, width=2)
        text(draw, (left + 12, top + 285), f"x[{bounds[0]:.2f},{bounds[1]:.2f}] y[{bounds[2]:.2f},{bounds[3]:.2f}]", 10)
    return image


def intervals_figure(data):
    image = Image.new("RGB", (1440, 920), "#f7f9fc"); draw = ImageDraw.Draw(image)
    text(draw, (24, 16), "Paired differences:95% stratified family bootstrap intervals", 24)
    text(draw, (24, 48), "2000 common draws; family-first seed means. Orange=mean; blue=95% interval. Exploratory, not equivalence or industrial proof.", 14)
    metrics = (("success", 100., "Success [percentage points]"),
               ("final_position_error_m", 1000., "Position error [mm]"),
               ("final_yaw_error_rad", 180 / math.pi, "Yaw error [deg]"),
               ("elapsed_s", 1., "All-episode end time [s]"),
               ("planning_wall_s", 1., "Total planning wall time [s]"),
               ("execution_wall_s", 1., "Total execution wall time [s]"))
    comparisons = data["statistics"]["comparisons"]
    for panel, (metric, factor, title) in enumerate(metrics):
        left, top = 20 + panel % 3 * 473, 88 + panel // 3 * 410
        draw.rounded_rectangle((left, top, left + 458, top + 394), radius=7, fill="white", outline="#d6dfeb")
        text(draw, (left + 12, top + 12), title, 18)
        values = [v * factor for c in comparisons for v in (c["metrics"][metric]["percentile95_interval"] or [])]
        low, high = min([0., *values]), max([0., *values]); span = high - low or 1.
        low -= span * .08; high += span * .08
        xmap = lambda value: left + 164 + (value - low) / (high - low) * 264
        draw.line((xmap(0.), top + 47, xmap(0.), top + 352), fill="#b8c1cf", width=1)
        for row, comparison in enumerate(comparisons):
            y = top + 65 + row * 31; value = comparison["metrics"][metric]
            text(draw, (left + 10, y - 8), comparison["left"] + " - " + comparison["right"], 12)
            interval = value["percentile95_interval"]
            if interval is None: text(draw, (left + 170, y - 8), "n/a", 12)
            else:
                a, b = [xmap(x * factor) for x in interval]
                draw.line((a, y, b, y), fill="#2463a2", width=3)
                draw.line((a, y - 4, a, y + 4), fill="#2463a2", width=1)
                draw.line((b, y - 4, b, y + 4), fill="#2463a2", width=1)
                mean = xmap(value["difference_mean"] * factor)
                draw.ellipse((mean - 3, y - 3, mean + 3, y + 3), fill="#c46d17")
            text(draw, (left + 12, y + 8), "n=" + str(value.get("eligible_family_denominator", 12)), 9, "#68788e")
        text(draw, (xmap(low), top + 366), f"{low:.3g}", 10)
        text(draw, (xmap(high) - 34, top + 366), f"{high:.3g}", 10)
        if xmap(0.) - xmap(low) > 42 and xmap(high) - xmap(0.) > 42:
            text(draw, (xmap(0.) - 3, top + 366), "0", 10)
    return image


def replay_frame(data, cells, stratum, clock, bounds):
    image = Image.new("RGB", (1280, 850), "#f1f5fa"); draw = ImageDraw.Draw(image)
    text(draw, (22, 12), ("SYNTHETIC FIXTURE | " if data.get("fixture_only") else "")
         + "RAW POSES | no interpolation, no physics render", 21)
    text(draw, (22, 44), f"{stratum} | {cells[0]['family_id']} | display clock {clock:.2f}s", 17)
    metadata = []
    config = data["physical_config"]
    for index, (cell, color) in enumerate(zip(cells, COLORS)):
        left, top = 14 + (index % 4) * 317, 83 + (index // 4) * 367
        draw.rounded_rectangle((left, top, left + 307, top + 355), radius=7, fill="white", outline="#d3ddea")
        held = held_sample(cell, clock); sample = held["display"]
        text(draw, (left + 10, top + 7), cell["controller_id"], 20, color)
        terminal = cell["termination"] if held["terminal"] else held["selected"]["phase"]
        text(draw, (left + 10, top + 34), terminal + (" (held)" if held["terminal"] else ""), 14,
             "#24844b" if terminal == "ready" else "#ad4235" if held["terminal"] else "#516079")
        transform, scale = mapping((left + 14, top + 62, left + 293, top + 248), bounds)
        goal(draw, config, transform, scale)
        path_line(draw, cell, held["selected_index"], (0, 1), transform, color, 2)
        path_line(draw, cell, held["selected_index"], (6, 7), transform, "#a3adbb", 1)
        if sample is not None:
            state = sample["state10"]; draw.polygon(rectangle(state, config, transform), fill=color, outline="#14283d")
            radius = config["pusher"]["radius_m"] * scale; px, py = transform(state[6], state[7])
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill="#ced6df", outline="#3b4d66", width=2)
            x, y, yaw = state[:3]; gx, gy = config["task"]["goal_xy_m"]
            error = math.hypot(x - gx, y - gy) * 1000
            angle = abs(.5 * math.atan2(math.sin(2 * (yaw - config["task"]["goal_yaw_rad"])), math.cos(2 * (yaw - config["task"]["goal_yaw_rad"])))) * 180 / math.pi
            text(draw, (left + 10, top + 254), f"sample {sample['relative_time_s']:.3f}s | sim {sample['sim_time_s']:.3f}s", 12)
            text(draw, (left + 10, top + 275), f"P {error:.2f}mm | yaw {angle:.2f}deg", 14)
            text(draw, (left + 10, top + 299), f"actual hold {held['selected']['held_time_s']:.3f}s", 14)
        else: text(draw, (left + 10, top + 271), "No finite recorded pose available", 13, "#ad4235")
        invalid = (held["nonfinite_selected_observation"] or not held["selected"]["observation_valid"]
                   or not held["selected"].get("transition_valid", True)
                   or not held["selected"].get("geometry_valid", True))
        if invalid: text(draw, (left + 10, top + 326), "INVALID OBSERVATION / last finite pose", 11, "#ad4235")
        metadata.append({"cell_id": cell["cell_id"], "selected_raw_index": held["selected_index"],
                         "drawn_raw_index": held["display_index"], "sample_relative_time_s": None if sample is None else sample["relative_time_s"],
                         "terminal_pose_held": held["terminal"], "invalid_selected_observation": invalid})
    text(draw, (20, 822), "Goal outline green; actual object/pusher sizes. Between measurements and after termination: last measured pose is held.", 12)
    return image, metadata


def save_png(image, name, budget):
    with budget.file(name) as stream: image.save(stream, format="PNG")


def render_clip(data, stratum, budget, *, fps=5, prefer_mp4=True):
    family = data["selected_families"][stratum]
    lookup = {(cell["family_id"], cell["controller_id"]): cell for cell in data["cells"]}
    cells = [lookup[family, controller] for controller in CONTROLLERS]
    duration = max(cell["elapsed_s"] for cell in cells) + 1.
    count = math.ceil(duration * fps) + 1; sample_indices = {0, count // 2, count - 1}
    bounds = world_bounds(data); frame_index = []
    def frames():
        for index in range(count):
            clock = index / fps
            image, selections = replay_frame(data, cells, stratum, clock, bounds)
            frame_index.append({"frame": index, "display_clock_s": clock, "raw_measurements": selections})
            if index in sample_indices: save_png(image, f"clip_{stratum}_frame_{index:04d}.png", budget)
            yield image
    ffmpeg = shutil.which("ffmpeg") if prefer_mp4 else None
    name = f"clip_{stratum}.mp4" if ffmpeg else f"clip_{stratum}.gif"
    if ffmpeg:
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", "1280x850", "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                   "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags",
                   "frag_keyframe+empty_moov", "-f", "mp4", "pipe:1"]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        errors = []
        with budget.file(name) as target:
            def copy_output():
                try:
                    for block in iter(lambda: process.stdout.read(65536), b""): target.write(block)
                except Exception as error:
                    errors.append(error); process.terminate()
            thread = threading.Thread(target=copy_output); thread.start()
            try:
                for image in frames(): process.stdin.write(image.tobytes())
                process.stdin.close(); thread.join()
                diagnostic = process.stderr.read().decode(errors="replace")
                code = process.wait(timeout=30)
                if errors: raise errors[0]
                require(code == 0, "ffmpeg encoding failed: " + diagnostic[:500])
            finally:
                if process.poll() is None: process.terminate(); process.wait(timeout=10)
                thread.join(timeout=10)
    else:
        iterator = frames(); first = next(iterator)
        with budget.file(name) as stream:
            first.save(stream, format="GIF", save_all=True,
                append_images=(image.quantize(colors=96, dither=Image.Dither.NONE) for image in iterator),
                duration=round(1000 / fps), loop=0, optimize=False, disposal=2)
    budget.json(f"clip_{stratum}_frame_index.json", {"schema": "p5-raw-pose-frame-index-v1",
        "video": name, "fps": fps, "frame_count": count, "frame_source": "recorded action-boundary observations",
        "interpolation": False, "postterminal_tail_s": 1., "frames": frame_index})
    return {"path": name, "format": "mp4" if ffmpeg else "gif", "fps": fps, "frame_count": count,
            "family_id": family, "controllers": list(CONTROLLERS), "sampling": "latest actual boundary at/before each frame time"}


def render_compact(compact_path, output, *, fps=5, prefer_mp4=True, budget_bytes=MAX_OUTPUT_BYTES):
    graphics(); output = Path(output); compact_hash = sha(compact_path)
    renderer_hash = sha(__file__); data = read(compact_path)
    require(type(fps) is int and 1 <= fps <= 10, "Replay fps must be1..10")
    require(fps == data["visualization"]["fps"] == 5
            and data["visualization"]["postterminal_tail_s"] == 1., "Frozen replay timing must be retained")
    require(len(data["cells"]) == 96
            and {(cell["family_id"], cell["controller_id"]) for cell in data["cells"]}
            == {(family["family_id"], controller) for family in data["families"] for controller in CONTROLLERS},
            "All96 distinct compact family/controller cells required")
    budget = Budget(output, budget_bytes)
    for name, function in (("all_endpoints.png", endpoints_figure),
                           ("all_trajectories.png", trajectories_figure),
                           ("bootstrap_intervals.png", intervals_figure)):
        image = function(data)
        if data.get("fixture_only"):
            draw = ImageDraw.Draw(image)
            text(draw, (20, image.height - 27), "SYNTHETIC FIXTURE ONLY - not an executed robot result", 15, "#b64039")
        save_png(image, name, budget)
    clips = [render_clip(data, stratum, budget, fps=fps, prefer_mp4=prefer_mp4) for stratum in STRATA]
    require(sha(compact_path) == compact_hash, "Compact source changed while drawing")
    require(sha(__file__) == renderer_hash, "Drawing tool changed during rendering")
    index = {"schema": "p5-final-visual-index-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GENERATED_REQUIRES_DIRECT_VISUAL_REVIEW", "compact_source_sha256": compact_hash,
        "renderer_tool_sha256": renderer_hash,
        "original_source_bindings_sha256": data["source_bindings_sha256"], "selection_rule": REPLAY_RULE,
        "shared_view_world_bounds_m": world_bounds(data),
        "selected_families": data["selected_families"], "clips": clips,
        "files_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()},
        "bytes_before_index": budget.used, "maximum_output_bytes": budget.limit,
        "scope": data["render_scope"], "direct_visual_review": "NOT_YET_RECORDED",
        "no_claims": ["new physics", "simulator camera rendering", "real hardware", "industrial success", "learner understanding"]}
    budget.json("visual_index.json", index)
    return index


def build_visuals(run, output, *, renderer_python=None, fps=5, prefer_mp4=True, budget_bytes=MAX_OUTPUT_BYTES):
    data = extract_run(run)
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    budget = Budget(output, budget_bytes); budget.json("compact_visual_source.json", data)
    python = Path(renderer_python) if renderer_python else DEFAULT_RENDERER
    require(python.is_file(), "Pillow renderer interpreter unavailable: " + str(python))
    command = [str(python), str(Path(__file__).resolve()), "--render-compact", str(output / "compact_visual_source.json"),
               "--out", str(output), "--fps", str(fps), "--budget-bytes", str(budget_bytes)]
    if not prefer_mp4: command.append("--gif")
    result = subprocess.run(command, capture_output=True, text=True)
    require(result.returncode == 0, "Visual rendering stopped; existing derived files preserved: " + result.stderr[-1000:])
    for path, expected in data["source_bindings_sha256"].items(): require(sha(path) == expected, "Source changed while rendering")
    return read(output / "visual_index.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--render-compact", type=Path)
    parser.add_argument("--renderer-python", type=Path)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--budget-bytes", type=int, default=MAX_OUTPUT_BYTES)
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()
    if args.render_compact:
        result = render_compact(args.render_compact, args.out, fps=args.fps,
            prefer_mp4=not args.gif, budget_bytes=args.budget_bytes)
    else:
        require(args.run is not None, "--run required")
        result = build_visuals(args.run, args.out, renderer_python=args.renderer_python,
            fps=args.fps, prefer_mp4=not args.gif, budget_bytes=args.budget_bytes)
    print(json.dumps({"status": result["status"], "clips": result["clips"]}, ensure_ascii=False))


if __name__ == "__main__": main()
