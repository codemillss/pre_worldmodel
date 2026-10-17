"""Check design arithmetic against saved integer training slots, without models.

No torch, MuJoCo, model forward, training, or episode data is loaded. Outputs
are design artifacts, not a completed data collection or experiment manifest.
"""
from pathlib import Path
import hashlib
import json
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path = ROOT / "configs/p4_data_intervention.design.v1.json"
    p = json.loads(path.read_text())
    c, t = p["collection"], p["training"]
    checks = {}
    checks["design_not_execution"] = p["status"] == "DESIGNED_NOT_RUN" and not p["executable_runner_config"]
    checks["twenty_step_patterns_exact"] = (
        c["S0_pattern"] == "M"*5 + "ZM"*5 + "M"*5 and
        c["S1_pattern"] == "M"*10 + "Z"*5 + "M"*5)
    with np.load(ROOT / t["historical_schedule"], allow_pickle=False) as a:
        # Access only schedule arrays. Never read losses or model predictions.
        positions = a["window_positions"].copy()
        offsets = a["batch_offsets"].copy()
    sizes = np.diff(offsets)
    counts = np.bincount(positions, minlength=2960)
    checks["registered_historical_schedule"] = (
        positions.shape == (62992,) and offsets.shape == (1001,) and
        offsets[0] == 0 and offsets[-1] == len(positions) and
        int((sizes == 64).sum()) == 979 and int((sizes == 16).sum()) == 21 and
        set(counts.tolist()) == {21, 22} and int((counts == 22).sum()) == 832)
    checks["each_complete_epoch_is_a_permutation"] = all(
        np.array_equal(np.sort(positions[j*2960:(j+1)*2960]), np.arange(2960))
        for j in range(21))
    partial_positions = np.concatenate([positions[offsets[i]:offsets[i+1]]
                                        for i,n in enumerate(sizes) if n == 16])
    partial_counts = np.bincount(partial_positions, minlength=2960)
    eligible = np.flatnonzero((counts == 21) & (partial_counts == 0)).tolist()
    selected = sorted(sorted(eligible, key=lambda i: hashlib.sha256(
        f"p4-data-v1-slot:{i}".encode()).hexdigest())[:384])
    mapping = [{"slot": slot, "family_rank": j//16, "window_start": j%16,
                "occurrences": int(counts[slot])} for j, slot in enumerate(selected)]
    checks["all384_new_slots_have21_exposures"] = len(mapping) == 384 and all(x["occurrences"] == 21 for x in mapping)
    checks["new_slots_never_in_small_weighted_batch"] = len(eligible) == 1897 and all(partial_counts[x] == 0 for x in selected)
    pattern_stats = {}
    for label in ("S0", "S1"):
        s = c[label+"_pattern"]
        windows = [s[i:i+5] for i in range(len(s)-4)]
        weighted_zero = sum(w.count("Z") for w in windows)
        allzero = sum(w == "ZZZZZ" for w in windows)
        run_lengths = [len(z) for z in re.findall("Z+", s)]
        pattern_stats[label] = {
            "raw_M": s.count("M"), "raw_Z": s.count("Z"),
            "windows_per_trajectory": len(windows),
            "zero_transition_occurrences_per_window_sweep": weighted_zero,
            "moving_transition_occurrences_per_window_sweep": len(windows)*5-weighted_zero,
            "allzero_windows_per_trajectory": allzero,
            "max_zero_run": max(run_lengths),
            "new_zero_training_occurrences": weighted_zero*24*21,
            "new_zero_nominal_batch_mean_mass": weighted_zero*24*21/(64*5),
            "allzero_training_window_occurrences": allzero*24*21}
    checks["raw_and_overlap_weighted_zero_counts_equal"] = all(
        x["raw_M"] == 15 and x["raw_Z"] == 5 and
        x["windows_per_trajectory"] == 16 and
        x["zero_transition_occurrences_per_window_sweep"] == 25 and
        x["moving_transition_occurrences_per_window_sweep"] == 55 and
        x["new_zero_training_occurrences"] == 12600 for x in pattern_stats.values())
    checks["continuous_K5_available_only_in_S1"] = (
        pattern_stats["S0"]["max_zero_run"] == 1 and
        pattern_stats["S1"]["max_zero_run"] == 5 and
        pattern_stats["S0"]["allzero_training_window_occurrences"] == 0 and
        pattern_stats["S1"]["allzero_training_window_occurrences"] == 504)
    checks["nominal_loss_denominator_exposure_equal"] = all(
        x["new_zero_nominal_batch_mean_mass"] == t["new_zero_nominal_batch_mean_mass"] == 39.375
        for x in pattern_stats.values())
    checks["training_denominators_exact"] = (
        2576+384 == t["window_population"] == 2960 and
        384*21 == t["new_window_occurrences_per_arm"] == 8064 and
        8064*5 == t["new_transition_occurrences_per_arm"] == 40320 and
        62992*5 == t["transition_occurrences_per_arm"] == 314960 and
        1000*len(p["arms"]) == t["total_formal_updates"] == 5000 and
        1300+1000 == t["final_Adam_step"] == 2300)
    checks["paired_data_arm_sizes"] = all(
        arm["new_windows"]+arm["old_windows"] == 2960 and
        (arm["id"] == "R" or (arm["new_windows"], arm["old_windows"]) == (384, 2576))
        for arm in p["arms"])
    max_norm = float(np.hypot(.003, .029))
    checks["commands_under_frozen_planning_limit"] = max_norm <= .03
    checks["planned_physical_intervals_accounted"] = (
        (24+12)*4*20 == c["main_core_intervals"] == 2880 and
        (24+12)*4*2 == c["main_warmup_intervals_upper_bound"] == 288 and
        3*4*22 == c["preflight_base_intervals"] == 264 and
        3*2*22 == c["preflight_dt_half_intervals"] == c["preflight_exact_replay_intervals"] == 132 and
        264+132+132 == c["preflight_intervals_upper_bound"] == 528 and
        2880+288+528 == c["total_action_intervals_upper_bound"] == 3696)
    historical_budget = json.loads((ROOT/"configs/p4_fork_budget.v2.json").read_text())
    checks["common_checkpoint_matches_original_budget"] = (
        p["model"]["initial_checkpoint"] == historical_budget["common_checkpoints"][0]["path"] and
        p["model"]["initial_checkpoint_hash"] == historical_budget["common_checkpoints"][0]["checkpoint_hash"] and
        historical_budget["additional_updates_per_branch"] == t["updates_per_arm"] == 1000)
    immutable = [
        "configs/p4_fork_budget.v2.json", "configs/p3_training.v1.toml",
        "configs/p3_planning.v1.json", "configs/pilot.v4.toml", "configs/acceptance.v1.toml",
        "reports/p4_local_prediction_diagnostic.md", "reports/p4_local_input_support_review.json",
        "reports/p4_local_contact_review.json", "reports/p4_local_independent_review.json",
        "runs/p3-02-20261009-01/learning_data/normalizer/normalizer.json",
        "runs/p3-02-20261009-01/learning_data/normalizer/statistics.npz",
        "runs/p3-04-20261009-01/seeds/seed-11/checkpoints/epoch-100/checkpoint.json",
        "runs/p4-02-20261012-01/seeds/seed-11/SK-lambda0.1/branch_summary.json",
        t["historical_schedule"], "src/push_align/fork_sampler.py", "src/push_align/sequence_training.py"]
    slots_path = ROOT/"reports/p4_data_intervention_slots.design.json"
    slots_path.write_text(json.dumps({"status": "DESIGN_ONLY_NOT_TRAINING_DATA",
        "source_schedule_sha256": sha(ROOT/t["historical_schedule"]), "entries": mapping}, indent=2)+"\n")
    report = {"status": "STATIC_DESIGN_PASS" if all(checks.values()) else "FAIL",
        "checks": checks, "design_sha256": sha(path),
        "report_sha256": sha(ROOT/"reports/p4_data_intervention_design.md"),
        "checker_sha256": sha(Path(__file__)), "slot_design_sha256": sha(slots_path),
        "pattern_arithmetic": pattern_stats, "maximum_command_norm_m_s": max_norm,
        "new_pool_fraction": 384/2960, "new_training_window_fraction": 8064/62992,
        "historical_schedule_counts": {str(n): int((counts == n).sum()) for n in (21,22)},
        "immutable_source_bindings": {x: sha(ROOT/x) for x in immutable},
        "physics_intervals": 0, "model_forwards": 0, "learning_updates": 0,
        "interpretation": "Static design arithmetic only; physical collection, training and evaluation remain NOT_RUN"}
    output = ROOT/"reports/p4_data_intervention_design_review.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"status": report["status"], "checks": len(checks),
                      "failed": [k for k,v in checks.items() if not v],
                      "zero_exposures": {k:v["new_zero_training_occurrences"] for k,v in pattern_stats.items()}}, ensure_ascii=False))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
