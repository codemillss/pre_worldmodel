"""Family-paired final summaries and stratified exploratory uncertainty.

No valid-only deletion, no success-only overall time, and no episode/seed
pseudoreplication. Bootstrap family draws are common across all methods.
"""
from __future__ import annotations
from collections import Counter
import math
import numpy as np

from .final_manifest import CONTROLLERS, STRATA

METRICS = ("success", "elapsed_s", "final_position_error_m", "final_yaw_error_rad",
           "final_linear_speed_m_s", "final_angular_speed_rad_s", "recontact_count",
           "planning_wall_s", "execution_wall_s")
GROUPS = {"G": ("G",), "R": ("R",),
          "S1": ("S1-11", "S1-23", "S1-47"),
          "SK": ("SK-11", "SK-23", "SK-47")}
STATE_METRICS = METRICS[2:6]
TERMINATIONS = {'ready','timeout','planner_invalid','controller_infeasible',
                'physics_error','geometry_invalid','workspace_exit','planarity_violation'}


def _axis_mean(values, axis):
    values = np.asarray(values, np.float64)
    valid = np.isfinite(values)
    count = valid.sum(axis=axis)
    scale = np.max(np.where(valid, np.abs(values), 0.), axis=axis, keepdims=True)
    scale = np.where(scale == 0, 1., scale)
    normalized = np.divide(values, scale, out=np.zeros_like(values), where=valid)
    result = (np.sum(normalized, axis=axis) / np.where(count, count, 1)) * np.squeeze(scale, axis=axis)
    return np.where(count, result, np.nan)


def _mean(values):
    values = [value for value in values if value is not None]
    return float(_axis_mean(values, 0)) if len(values) else None


def _percentile(values, percentiles):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if not len(values): return None
    scale = float(np.max(np.abs(values)))
    return [0. for _ in percentiles] if scale == 0 else (np.percentile(values / scale, percentiles) * scale).tolist()


def _summary(rows):
    successful = [row for row in rows if row["success"]]
    return {"episode_count": len(rows), "family_count": len({row["family_id"] for row in rows}),
            "success_count": len(successful), "success_rate": len(successful) / len(rows),
            "timeout_count": sum(row["termination"] == "timeout" for row in rows),
            "timeout_rate": sum(row["termination"] == "timeout" for row in rows) / len(rows),
            "termination_counts": dict(sorted(Counter(row["termination"] for row in rows).items())),
            "success_conditional_time_s": {"denominator": len(successful),
                "mean": _mean([row["elapsed_s"] for row in successful]),
                "median": _percentile([row['elapsed_s'] for row in successful],[50])[0] if successful else None,
                "estimand": "pooled successful episodes; seed-group pool is separate from family-first paired comparisons"},
            "state_observation_populations": dict(Counter(row['metric_state_origin'] for row in rows)),
            "terminal_observation_unavailable_count": sum(not row['final_observation_available'] for row in rows),
            "all_episode_means": {metric: _mean([row[metric] for row in rows]) for metric in METRICS},
            "metric_denominators": {metric: sum(row[metric] is not None for row in rows) for metric in METRICS}}


def analyze_E2(rows, expected, *, bootstrap_replicates=2000, bootstrap_seed=7711):
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 2:
        raise ValueError("At least two predeclared bootstrap replicates required")
    expected_ids = {row["cell_id"] for row in expected}
    if len(expected) != 96 or len(expected_ids) != 96 or len(rows) != 96 or {row["cell_id"] for row in rows} != expected_ids:
        raise ValueError("Exact complete expected96 cells required; no missing/extra/duplicate cell")
    expected_by = {row["cell_id"]: row for row in expected}
    for row in rows:
        if row["execution_status"] != "completed" or row["termination"] not in TERMINATIONS:
            raise ValueError("Unresolved infrastructure is not a completed final denominator")
        if any(row[key] != expected_by[row["cell_id"]][key] for key in ("family_id", "stratum", "controller_id", "episode_id")):
            raise ValueError("Result identity differs from frozen expected cell")
        if type(row["success"]) is not bool or row["success"] != (row["termination"] == "ready"):
            raise ValueError("Success must follow actual ready termination")
        if (type(row['final_observation_available']) is not bool
                or row['metric_state_origin'] not in {'terminal_observation','last_valid_observation','unavailable'}):
            raise ValueError('Explicit final-observation availability and metric-state origin required')
        if (row['metric_state_origin'] == 'terminal_observation') != row['final_observation_available']:
            raise ValueError('Terminal observation origin differs from availability')
        missing = row['metric_state_origin'] == 'unavailable'
        state_time = row['metric_state_time_s']
        if missing:
            if state_time is not None or not row.get('unavailable_reason') or any(row[key] is not None for key in STATE_METRICS):
                raise ValueError('Unavailable state metrics require null values/time and a reason')
        elif (type(state_time) not in (int,float) or not math.isfinite(state_time)
              or state_time < 0 or state_time > row['elapsed_s'] + 1e-7):
            raise ValueError('Observed metric timestamp must be within the executed task prefix')
        if row['final_observation_available'] and abs(state_time - row['elapsed_s']) > 1e-7:
            raise ValueError('Terminal metric time must equal episode-relative elapsed time')
        if row['success'] and not row['final_observation_available']:
            raise ValueError('Actual ready termination requires a valid final observation')
        for metric in METRICS:
            if metric == 'success': continue
            if row[metric] is None and metric in STATE_METRICS and missing: continue
            if type(row[metric]) not in (int, float) or not math.isfinite(row[metric]) or row[metric] < 0:
                raise ValueError("Finite nonnegative final metric required: " + metric)
        if type(row['recontact_count']) is not int or row['elapsed_s'] > 30. + 1e-7:
            raise ValueError('Integer contact count and original30-second task duration required')
    families = sorted({row["family_id"] for row in rows})
    strata = {s: sorted({row["family_id"] for row in rows if row["stratum"] == s}) for s in STRATA}
    if len(families) != 12 or any(len(value) != 4 for value in strata.values()):
        raise ValueError("Original four families per stratum required")
    population = {}
    for controller in CONTROLLERS:
        part = [row for row in rows if row["controller_id"] == controller]
        if len(part) != 12 or {row["family_id"] for row in part} != set(families):
            raise ValueError("Each controller requires all12 same families")
        population[controller] = _summary(part)
        population[controller]["by_stratum"] = {s: _summary([row for row in part if row["stratum"] == s]) for s in STRATA}
    grouped = {name: _summary([row for row in rows if row["controller_id"] in members]) for name, members in GROUPS.items()}
    # Training seeds are averaged first within a family. They are not three
    # independent initial-condition samples or a seed-population confidence CI.
    by = {(row["family_id"], row["controller_id"]): row for row in rows}
    methods = {**GROUPS, **{name: (name,) for name in CONTROLLERS if name not in GROUPS}}
    arrays = {name: {metric: np.array([(_mean([by[(family, controller)][metric] for controller in members])
                        if all(by[(family, controller)][metric] is not None for controller in members) else np.nan)
                        for family in families], np.float64) for metric in METRICS}
              for name, members in methods.items()}
    rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
    draws = np.concatenate([rng.choice([families.index(f) for f in strata[s]], size=(bootstrap_replicates, 4), replace=True) for s in STRATA], axis=1)
    contrasts = [("R", "G"), ("S1", "G"), ("SK", "G"), ("S1", "R"), ("SK", "R"), ("SK", "S1")]
    contrasts += [(f"SK-{seed}", f"S1-{seed}") for seed in (11, 23, 47)]
    comparisons = []
    bootstrap_arrays = {"family_draw_indices": draws.astype(np.int64)}
    for left, right in contrasts:
        differences = {metric: arrays[left][metric] - arrays[right][metric] for metric in METRICS}
        stats = {}
        for metric, values in differences.items():
            boot = _axis_mean(values[draws], axis=1)
            interval = _percentile(boot, [2.5,97.5])
            key = f"{left}_minus_{right}_{metric}"
            bootstrap_arrays[key] = boot
            bootstrap_arrays[key + '_valid'] = np.isfinite(boot)
            stats[metric] = {"difference_mean": _mean([float(x) for x in values if np.isfinite(x)]),
                             "family_differences": [float(x) if np.isfinite(x) else None for x in values],
                             "eligible_family_denominator": int(np.isfinite(values).sum()),
                             "bootstrap_valid_replicates": int(np.isfinite(boot).sum()),
                             "missing_policy": "all members of both methods need a measured state; task outcomes retain every episode",
                             "percentile95_interval": interval,
                             "zero_in_interval": bool(interval[0] <= 0 <= interval[1]) if interval else None,
                             "unit": "proportion" if metric == "success" else metric,
                             "claim": "exploratory family-paired difference; no equivalence or industrial/general seed-population claim"}
        comparisons.append({"left":left, "right":right, "family_count":12, "metrics":stats})
    return {"status": "COMPLETE_EXPECTED_E2_DENOMINATORS",
            "episode_count":96, "family_count":12, "controller_count":8,
            "controllers":population, "seed_group_summary":grouped,
            "comparisons": comparisons, "bootstrap": {"seed":bootstrap_seed, "replicates":bootstrap_replicates,
                "unit":"family within stratum; all methods share each draw; seed average within family first",
                "families_in_order":families, "strata":strata,
                "limitations":"Four E2 families per stratum; exploratory uncertainty, no power/equivalence/industrial-frequency guarantee; training-seed variation not fully covered"},
            "performance_floor_or_minimum_meaningful_difference":"NOT_SPECIFIED; no post-result threshold",
            "metric_state_time_basis":"episode-relative seconds: absolute raw sim_time minus initial task sim_time",
            "personal_understanding":"NOT_ASSESSED"}, bootstrap_arrays
