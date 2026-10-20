"""Recompute the paper's reported numbers from selected published evidence.

This publication audit reads saved arrays and summaries; it does not train a
model, perform inference, or execute physics. It checks the current manuscript's
rounded numeric rows and figure values as well as E1/E2 arithmetic. It cannot
attest omitted high-frequency raw data or replace physical reproduction.

Requires NumPy. Run from any directory; relative --output paths use the repo root.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from push_align.candidate_diagnostics import compare_candidates
from push_align.final_e1_statistics import analyze_E1
from push_align.final_statistics_v2 import analyze_E2

GROUPS = {
    "G": ("G",),
    "R": ("R",),
    "S1": ("S1-11", "S1-23", "S1-47"),
    "SK": ("SK-11", "SK-23", "SK-47"),
}
MODELS = ("R", "S1-11", "SK-11", "S1-23", "SK-23", "S1-47", "SK-47")


def read_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_text(paper: str, text: str) -> None:
    require(text in paper, "Manuscript numeric statement differs: " + text)


def check_bootstrap(relative: str, regenerated: dict) -> int:
    with np.load(ROOT / relative, allow_pickle=False) as saved:
        require(set(regenerated) == set(saved.files), "Bootstrap columns differ: " + relative)
        for key, value in regenerated.items():
            require(np.array_equal(value, saved[key], equal_nan=True),
                    "Bootstrap array differs: " + relative + ":" + key)
    return len(regenerated)


def check_e2(paper: str, manifest: dict) -> dict:
    rows = read_json("artifacts/results/E2/rows.json")
    summary, arrays = analyze_E2(rows, manifest["E2"])
    require(summary == read_json("artifacts/results/E2/summary.json"),
            "E2 regenerated summary differs")
    array_count = check_bootstrap("artifacts/results/E2/bootstrap.npz", arrays)
    table = {}
    for name, members in GROUPS.items():
        part = [row for row in rows if row["controller_id"] in members]
        successes = sum(row["success"] for row in part)
        position = float(np.mean([row["final_position_error_m"] for row in part])) * 1000
        yaw = math.degrees(float(np.mean([row["final_yaw_error_rad"] for row in part])))
        elapsed = float(np.mean([row["elapsed_s"] for row in part]))
        require_text(paper, f"{name} & {successes}/{len(part)} & {position:.3f} & {yaw:.3f} & {elapsed:.3f}")
        strata = {}
        for stratum in ("translation", "rotation", "offset"):
            subset = [row for row in part if row["stratum"] == stratum]
            strata[stratum] = [sum(row["success"] for row in subset), len(subset)]
        require_text(paper, name + " & " + " & ".join(f"{a}/{b}" for a, b in strata.values()))
        table[name] = {"episodes": len(part), "successes": successes,
                       "position_mm": position, "yaw_deg": yaw,
                       "time_s": elapsed, "strata": strata}

    paired = []
    for comparison in summary["comparisons"][:6]:
        metric = comparison["metrics"]["success"]
        left, right = comparison["left"], comparison["right"]
        difference = metric["difference_mean"] * 100
        interval = [value * 100 for value in metric["percentile95_interval"]]
        printed = f"{difference:+.2f}" if difference > 0 else f"{difference:.2f}"
        require_text(paper, f"{left}--{right} & ${printed}$ & $[{interval[0]:.2f},{interval[1]:.2f}]$")
        require(metric["eligible_family_denominator"] == 12
                and metric["bootstrap_valid_replicates"] == 2000,
                "E2 comparison denominator changed")
        paired.append({"left": left, "right": right, "difference_pp": difference,
                       "ci95_pp": interval, "eligible_families": 12})

    failures = [row for row in rows if not row["success"]]
    misses = {
        "position": sum(row["final_position_error_m"] > .005 for row in failures),
        "yaw": sum(row["final_yaw_error_rad"] > math.radians(3) for row in failures),
        "linear_speed": sum(row["final_linear_speed_m_s"] > .01 for row in failures),
        "angular_speed": sum(row["final_angular_speed_rad_s"] > .05 for row in failures),
    }
    require(misses == {"position": 62, "yaw": 38, "linear_speed": 0, "angular_speed": 2},
            "Reported failure endpoint counts changed")
    return {"rows": len(rows), "families": summary["family_count"],
            "summary_exact": True, "bootstrap_arrays_exact": array_count,
            "table_and_strata_rows_match": table, "paired_rows_match": paired,
            "terminations": dict(Counter(row["termination"] for row in rows)),
            "failures": len(failures), "failure_endpoint_misses": misses}


def check_e1(paper: str, manifest: dict) -> dict:
    rows = []
    cost_ranges = []
    invalid = Counter()
    repeat_count = valid_repeat_count = 0
    maxima = {"states": 0., "applied_actions": 0., "total_cost": 0.}
    snapshots = sorted((ROOT / "artifacts/results/E1/snapshots").glob("*/*/tick-*"))
    require(len(snapshots) == 12, "Expected exactly 12 E1 snapshots")
    for folder in snapshots:
        with np.load(folder / "actual_futures.npz", allow_pickle=False) as actual:
            costs, validity = actual["total_cost"].copy(), actual["valid"].copy()
            require(costs.shape == validity.shape == (16, 3), "E1 actual tensor shape differs")
            cost_ranges.extend(np.ptp(costs, axis=1).tolist())
            repeat_count += validity.size
            valid_repeat_count += int(validity.sum())
            # Verify saved arrays directly; do not rely solely on exactness flags.
            for repeat in (1, 2):
                for key in ("states", "applied_actions", "total_cost", "valid"):
                    require(np.array_equal(actual[key][:, 0], actual[key][:, repeat]),
                            "E1 repeated array differs: " + str(folder.relative_to(ROOT)) + ":" + key)
                    if key != "valid":
                        difference = float(np.max(np.abs(actual[key][:, 0] - actual[key][:, repeat])))
                        maxima[key] = max(maxima[key], difference)
        for model_folder in sorted((folder / "models").iterdir()):
            diagnostic = json.loads((model_folder / "diagnostics.json").read_text(encoding="utf-8"))
            rows.append(diagnostic)
            invalid[diagnostic["model"]] += diagnostic["counts"]["predicted_invalid"]
            with np.load(model_folder / "trace.npz", allow_pickle=False) as prediction:
                regenerated = compare_candidates(
                    prediction["total_cost"],
                    prediction["valid"] & np.isfinite(prediction["total_cost"]),
                    costs, validity)
            require(all(diagnostic[key] == value for key, value in regenerated.items()),
                    "E1 saved diagnostic differs: " + str(model_folder.relative_to(ROOT)))
            require(diagnostic["all3_physical_repeats_exact"] is True,
                    "Stored E1 exactness flag contradicts checked arrays")

    summary, arrays = analyze_E1(rows, manifest["E1"])
    require(summary == read_json("artifacts/results/E1/summary.json"),
            "E1 regenerated summary differs")
    array_count = check_bootstrap("artifacts/results/E1/bootstrap.npz", arrays)
    means = {}
    for name in ("R", "S1", "SK"):
        population = summary["seed_groups"] if name in summary["seed_groups"] else summary["models"]
        value = population[name]["selection_loss"]["complete_family_mean"]
        means[name] = value
        require_text(paper, f"{name} {value:.6f}")

    prediction_count = sum(row["counts"]["candidate_count"] for row in rows)
    require(prediction_count == 1344 and sum(invalid.values()) == 13,
            "E1 prediction invalid denominator changed")
    require(invalid == Counter({"R": 10, "S1-11": 2, "SK-11": 1}),
            "E1 invalid candidates differ by model")
    require_text(paper, f"총 {prediction_count:,}개 후보 예측 중 {sum(invalid.values())}개(R 10, S1-11 2, SK-11 1)")
    require(valid_repeat_count == repeat_count == 576, "E1 physical repeat validity changed")
    require_text(paper, "실제 후보 반복 576회는 모두 유효했다")

    choices = {(row["snapshot_id"], row["model"]): row["selected_index"] for row in rows}
    agreement = sum(choices[(spec["snapshot_id"], f"S1-{seed}")]
                    == choices[(spec["snapshot_id"], f"SK-{seed}")]
                    for spec in manifest["E1"] for seed in (11, 23, 47))
    require(agreement == 35, "E1 S1/SK choice agreement changed")
    require_text(paper, "36쌍 중 35쌍")
    return {"diagnostics_regenerated": len(rows), "snapshots": len(snapshots),
            "families": summary["independent_family_count"], "summary_exact": True,
            "bootstrap_arrays_exact": array_count, "means_match_paper": means,
            "physical_repeat_count": repeat_count, "actual_valid_repeats": valid_repeat_count,
            "actual_futures_repeated_state_action_cost_and_valid_arrays_exact": True,
            "maximum_repeated_state_difference": maxima["states"],
            "maximum_repeated_action_difference": maxima["applied_actions"],
            "maximum_repeated_cost_difference": maxima["total_cost"],
            "maximum_repeated_cost_range": max(cost_ranges),
            "prediction_count": prediction_count,
            "predicted_invalid_candidate_count": sum(invalid.values()),
            "predicted_invalid_candidate_count_by_model": dict(invalid),
            "S1_SK_same_selected_index": f"{agreement}/36",
            "undefined_loss_sets": sum(row["selection_loss"] is None for row in rows)}


def family_metric(record: dict, family_ids: list, metric: str) -> float:
    values = [record["by_family"][family]["metrics_conditional_on_valid_prediction_and_truth"][metric]["mean"]
              for family in family_ids]
    computed = float(np.mean(values))
    saved = record["family_first_means_conditional_on_available_families"][metric]
    require(saved["family_denominator"] == 12 and abs(computed - saved["mean"]) < 1e-18,
            "Fixed-prediction family-first mean differs: " + metric)
    return computed


def check_fixed_prediction(paper: str) -> dict:
    source = "artifacts/results/fixed_prediction/summary.json"
    data = read_json(source)
    figure = read_json("paper/figures/fixed_prediction.values.json")
    require(figure["source_sha256"] == sha256(source), "Figure source SHA differs")
    require(data["family_count"] == figure["family_denominator"] == 12,
            "Fixed prediction family denominator changed")
    table = {}
    for model in MODELS:
        values, masks = [], []
        for horizon in (1, 5, 10):
            hdata = data["horizons"][str(horizon)]
            record = hdata["models"][model]
            value = family_metric(record, data["family_ids"], "object_xy_error_m")
            values.append(value * 1000)
            masks.append({"horizon": horizon, "truth_eligible": hdata["truth_eligible_windows"],
                          "valid": record["predicted_valid_windows"],
                          "invalid": record["predicted_invalid_windows"]})
            if model in ("S1-23", "SK-23", "S1-47", "SK-47"):
                method, seed = model.split("-")
                point = next(item for item in figure["seeds"][seed][method] if item["horizon"] == horizon)
                require(abs(point["family_first_mean_mm"] - value * 1000) < 1e-15,
                        "Plotted figure point differs: " + model)
                family_values = {
                    family: record["by_family"][family]["metrics_conditional_on_valid_prediction_and_truth"]["object_xy_error_m"]["mean"]
                    for family in data["family_ids"]}
                require(point["family_means_m"] == family_values, "Figure family values differ")
                require(record["predicted_valid_windows"] == hdata["truth_eligible_windows"]
                        and record["predicted_invalid_windows"] == 0,
                        "Plotted methods no longer share all truth-eligible windows")
        label = model + (r"$^\dagger$" if model in ("R", "S1-11") else "")
        require_text(paper, label + " & " + " & ".join(f"{value:.4f}" for value in values))
        table[model] = {"position_mm_by_K1_K5_K10": values, "masks": masks}

    # Verify the added state-component counterexample on the same all-valid K5 windows.
    components = {}
    for model in ("S1-47", "SK-47"):
        record = data["horizons"]["5"]["models"][model]
        components[model] = {
            "position_mm": family_metric(record, data["family_ids"], "object_xy_error_m") * 1000,
            "yaw_deg_2pi": math.degrees(family_metric(record, data["family_ids"], "object_yaw_error_rad_2pi")),
            "angular_velocity_rad_s": family_metric(record, data["family_ids"], "object_angular_velocity_error_rad_s"),
        }
    one, multi = components["S1-47"], components["SK-47"]
    require(multi["position_mm"] < one["position_mm"]
            and multi["yaw_deg_2pi"] > one["yaw_deg_2pi"]
            and multi["angular_velocity_rad_s"] > one["angular_velocity_rad_s"],
            "Reported mixed state-component change no longer holds")
    require_text(paper, f"Seed47의 K5 위치오차는 {one['position_mm']:.6f}에서 {multi['position_mm']:.6f}")
    require_text(paper, f"{one['yaw_deg_2pi']:.6f}^\\circ")
    require_text(paper, f"{multi['yaw_deg_2pi']:.6f}^\\circ")
    require_text(paper, f"각속도 오차는 {one['angular_velocity_rad_s']:.6f}에서 ${multi['angular_velocity_rad_s']:.6f}")
    return {"table_rows_all_horizons_match": table,
            "all_12_figure_points_match_per_family_and_summary": True,
            "figure_source_sha256_matches": True,
            "seed47_K5_state_component_counterexample": components,
            "scope": "Published aggregate family means verified; omitted per-window prediction arrays and source raw were not re-inferred."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("tmp/paper-content-review/results_audit.v2.json"),
                        help="Receipt path; relative paths are resolved from the repository root")
    args = parser.parse_args()
    paper = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
    manifest = read_json("artifacts/provenance/frozen_manifest.json")
    receipt = {
        "schema": "paper-content-independent-results-audit-v2",
        "status": "PASS_SELECTED_SAVED_RESULTS_AND_PAPER_NUMBERS",
        "verifier": "scripts/verify_paper_results.py",
        "verifier_sha256": sha256("scripts/verify_paper_results.py"),
        "paper_source_sha256": sha256("paper/main.tex"),
        "new_physics_calls": 0, "new_model_calls": 0, "new_training_updates": 0,
        "E2": check_e2(paper, manifest), "E1": check_e1(paper, manifest),
        "fixed_prediction": check_fixed_prediction(paper),
        "scope": "Publication audit of selected saved arrays, summary arithmetic and numeric manuscript statements. This is not new scientific execution, an independent physical replication, or an audit of omitted raw data.",
        "statistical_scope": "Family bootstrap conditions on the observed training seeds and empirical design strata; it does not cover the full training-seed population or industrial occurrence distribution.",
        "E1_comparator": {"code": "src/push_align/candidate_diagnostics.py",
                          "epsilon": "1e-6 + 1e-5 * max(abs(actual_repeat_costs))",
                          "meaning": "Expanded observed ranges, not confidence intervals; exact repeats test deterministic reproducibility."},
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"],
                      "E2_bootstrap_arrays_exact": receipt["E2"]["bootstrap_arrays_exact"],
                      "E1_bootstrap_arrays_exact": receipt["E1"]["bootstrap_arrays_exact"],
                      "E1_diagnostics_regenerated": receipt["E1"]["diagnostics_regenerated"],
                      "scope": receipt["scope"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
