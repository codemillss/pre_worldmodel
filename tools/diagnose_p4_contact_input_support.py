"""Describe training-input support around one saved contact-transition failure.

Rank unique recorded train transitions by Euclidean distance in the exact
frozen float32 11-feature model input. This is a descriptive neighborhood,
not a density test, proof of sufficient learning, or overfitting diagnosis.
No model is constructed/called, no optimizer is used, and no physics runs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from push_align.normalization import load_normalizer
from push_align.representation import encode, state_delta, FEATURE_FIELDS, FEATURE_UNITS, DELTA_FIELDS, DELTA_UNITS

ARMS = ("R", "D00", "D10", "D01", "D11")
TARGET_FAMILY = "p4-data-development-2026101302-f001"
MAIN = ROOT / "runs/p4-data-main-20261013-01"
LEARNING = ROOT / "runs/p3-02-20261009-01/learning_data"
TRAINING = ROOT / "runs/p4-data-retrain-20261013-02"
SOURCE_MODULES = ("src/push_align/representation.py", "src/push_align/normalization.py", "src/push_align/dynamics.py",
                  "src/push_align/data.py", "src/push_align/coverage.py", "src/push_align/sequence_training.py",
                  "src/push_align/training.py")
REGIMES = {0: "noncontact", 1: "sustained_contact", 2: "transition"}
GROUPS = {"relative_position": [0, 1], "orientation": [2, 3],
          "object_velocity": [4, 5, 6], "pusher_velocity": [7, 8], "command": [9, 10]}


def require(ok, reason):
    if not ok: raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def load_npz(path, fields=None):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in (fields or archive.files)}


def required_source_bindings():
    """Source inventory only: no target features/ranks or predictions computed."""
    paths = [Path(__file__).resolve(), *(ROOT / p for p in SOURCE_MODULES)]
    paths += [TRAINING / name for name in ("manifest.json", "training_artifact_hashes.json", "prepared_training.npz",
                                         "dataset_bindings.json", "summary.json")]
    paths += [MAIN / "registration.json", MAIN / "slots_bound.json", LEARNING / "manifest.json", LEARNING / "prepared.npz",
              LEARNING / "normalizer/normalizer.json", LEARNING / "normalizer/statistics.npz"]
    paths += [TRAINING / "prediction_evaluation/artifact_hashes.json"]
    for arm in ARMS:
        paths += [TRAINING / "prediction_evaluation" / arm / name for name in ("development.json", "development_traces.npz")]
        paths += [TRAINING / "arms" / arm / "training_trace.npz"]
    bindings = read(TRAINING / "dataset_bindings.json")
    # Raw JSON diagnostic traces are deliberately excluded; only the audited NPZ
    # force summary and small manifest are read by this descriptive analysis.
    paths += [MAIN / rel for rel in bindings["raw_core_artifact_sha256"]
              if rel.endswith("/manifest.json") or rel.endswith(".npz")]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(paths))}


def model_features(states, actions, normalizer):
    """Identical encode/normalize/cast boundary to dynamics.predict_delta, no net."""
    states, actions = np.asarray(states, np.float64), np.asarray(actions, np.float64)
    with torch.no_grad():
        features = encode(torch.from_numpy(states.copy()), torch.from_numpy(actions.copy()))
        normalized64 = normalizer.normalize_features(features)
        normalized32 = normalized64.to(torch.float32)
    # NumPy is an arithmetic sensitivity check only; it never defines main ranks.
    numpy64 = (encode(states, actions) - normalizer.arrays["input_mean"]) / normalizer.arrays["input_scale"]
    return features.numpy().copy(), normalized32.numpy().copy(), normalized64.numpy().copy(), numpy64


def ranked_distance(features, query, ids):
    """Float32 model values promoted before distance arithmetic; stable ID ties."""
    features, query = np.asarray(features), np.asarray(query)
    require(features.ndim == 2 and query.shape == (features.shape[1],) and len(ids) == len(features),
            "Ranked distance shapes differ")
    require(np.isfinite(features).all() and np.isfinite(query).all() and len(set(ids)) == len(ids),
            "Ranked distance requires finite features and unique IDs")
    squared = (features.astype(np.float64) - query.astype(np.float64)) ** 2
    distance = np.sqrt(squared.sum(axis=1))
    order = np.lexsort((np.asarray(ids), distance))
    ranks = np.empty(len(ids), np.int64); ranks[order] = np.arange(1, len(ids)+1)
    return order, ranks, distance, squared


def contact_regimes(summary):
    """Frozen active force fraction uses strict >0.001N, already raw-audited."""
    summary = np.asarray(summary)
    require(summary.shape == (20, 4) and np.isfinite(summary).all(), "Contact summary shape/finiteness differs")
    fraction = summary[:, 2]
    require(np.all((fraction >= 0) & (fraction <= 1)), "Contact fraction outside [0,1]")
    return np.where(fraction == 0, "noncontact", np.where(fraction == 1, "sustained_contact", "transition"))


def geometry(state, config):
    state = np.asarray(state, np.float64)
    delta = state[6:8] - state[:2]; co, si = np.cos(state[2]), np.sin(state[2])
    local = np.array([co*delta[0] + si*delta[1], -si*delta[0] + co*delta[1]])
    q = np.abs(local) - np.asarray(config["object"]["size_full_xyz_m"][:2])/2
    gap = np.linalg.norm(np.maximum(q, 0)) + min(max(q), 0) - config["pusher"]["radius_m"]
    return {"relative_world_xy_m": delta.tolist(), "relative_body_xy_m": local.tolist(),
            "boundary_gap_m": float(gap), "yaw_rad": float(state[2]),
            "object_velocity_m_s_rad_s": state[3:6].tolist(), "pusher_velocity_m_s": state[8:10].tolist(),
            "absolute_object_xy_m": state[:2].tolist(), "absolute_pusher_xy_m": state[6:8].tolist(),
            "gap_scope": "Planar boundary geometry, not contact force or interval contact evidence"}


def exposure_by_window(trace, count=2960):
    positions, offsets = trace["window_positions"], trace["batch_offsets"]
    require(positions.shape == (62992,) and offsets.shape == (1001,) and offsets[0] == 0
            and offsets[-1] == len(positions), "Frozen training stream dimensions differ")
    require(positions.min() >= 0 and positions.max() < count, "Invalid training slot")
    counts = np.bincount(positions, minlength=count)
    teacher_mass = np.zeros(count, np.float64)
    for begin, end in zip(offsets[:-1], offsets[1:]):
        size = int(end-begin)
        require(size in (16, 64), "Unregistered batch size")
        np.add.at(teacher_mass, positions[begin:end], 1. / (size * 5))
    return counts, teacher_mass


def add_occurrence(population, identity, source, state, action, next_state, slot, depth, exposure, mass):
    """One physical transition ID, with all overlapping-window exposure separate."""
    require(0 <= depth < 5 and slot >= 0 and exposure >= 0 and mass >= 0, "Invalid transition membership/exposure")
    if identity not in population:
        population[identity] = {**source, "transition_id": identity,
            "state": np.asarray(state, np.float64).copy(), "action": np.asarray(action, np.float64).copy(),
            "next_state": np.asarray(next_state, np.float64).copy(), "window_memberships": [],
            "teacher_transition_occurrences": 0, "rollout_initial_state_occurrences": 0,
            "rollout_target_occurrences_by_depth": [0]*5, "nominal_teacher_batch_mean_mass": 0.}
    row = population[identity]
    require(all(row[k] == v for k, v in source.items()), "Repeated transition ID has inconsistent provenance")
    require(np.array_equal(row["state"], state) and np.array_equal(row["action"], action)
            and np.array_equal(row["next_state"], next_state), "Repeated transition ID has inconsistent tensors")
    require([slot, depth] not in row["window_memberships"], "Duplicate window-depth identity")
    row["window_memberships"].append([slot, depth])
    row["teacher_transition_occurrences"] += int(exposure)
    row["rollout_target_occurrences_by_depth"][depth] += int(exposure)
    row["rollout_initial_state_occurrences"] += int(exposure) if depth == 0 else 0
    row["nominal_teacher_batch_mean_mass"] += float(mass)


def summarize_neighbor(row, rank, distance, squared, query, query_next, config):
    true_delta = state_delta(row["state"], row["next_state"])
    query_delta = state_delta(query, query_next)
    return {**{k: row[k] for k in ("transition_id", "source", "episode_id", "family_id", "transition_index",
            "regime", "teacher_transition_occurrences", "rollout_initial_state_occurrences",
            "rollout_target_occurrences_by_depth", "nominal_teacher_batch_mean_mass", "window_memberships")},
        "rank": int(rank), "distance_normalized_L2": float(distance),
        "squared_distance_by_feature": dict(zip(FEATURE_FIELDS, squared.tolist())),
        "squared_distance_by_group": {name: float(squared[index].sum()) for name, index in GROUPS.items()},
        "window_membership_count": len(row["window_memberships"]), "training_exposed": row["teacher_transition_occurrences"] > 0,
        "physical_state": row["state"].tolist(), "physical_action_m_s": row["action"].tolist(),
        "true_next_state": row["next_state"].tolist(), "true_target_delta": true_delta.tolist(),
        "true_target_delta_minus_query_delta": (true_delta-query_delta).tolist(),
        "current_geometry": geometry(row["state"], config), "next_geometry": geometry(row["next_state"], config)}


def analyze(expected_sources=None):
    training_run = TRAINING
    before = {}; checks = []
    def bound(path, expected=None):
        path = Path(path).resolve(); value = sha(path)
        require(path.is_relative_to(ROOT), "Source escaped project")
        relative = path.relative_to(ROOT).as_posix()
        if expected_sources is not None:
            require(relative in expected_sources and value == expected_sources[relative], "Preregistered source changed or missing: " + relative)
        if expected is not None: require(value == expected, "Bound source changed: " + str(path))
        before[relative] = value
        return path
    def check(name, ok, **detail):
        require(ok, name); checks.append({"name": name, "pass": True, **detail})
    bound(__file__)
    for p in SOURCE_MODULES:
        bound(ROOT / p)
    manifest = read(bound(training_run / "manifest.json"))
    training_index = read(bound(training_run / "training_artifact_hashes.json"))["sha256"]
    bound(training_run / "manifest.json", training_index["manifest.json"])
    prepared = load_npz(bound(training_run / "prepared_training.npz", manifest["prepared_training_sha256"]))
    bindings = read(bound(training_run / "dataset_bindings.json", manifest["dataset_bindings_sha256"]))
    train_summary = read(bound(training_run / "summary.json", training_index["summary.json"]))
    check("source_training_run_pass_and_frozen_5000", train_summary["status"] == "PASS" and train_summary["counters"]["formal_updates"] == 5000)
    collection = read(bound(MAIN / "registration.json", bindings["collection_registration_sha256"]))
    slot_document = read(bound(MAIN / "slots_bound.json", bindings["slots_bound_sha256"]))
    slots = {entry["slot"]: entry for entry in slot_document["entries"]}
    check("exact_384_replacement_slots", len(slots) == 384 and np.array_equal(np.array(sorted(slots)), prepared["replacement_slots"]))
    source_original = read(bound(LEARNING / "manifest.json"))
    original = load_npz(bound(LEARNING / "prepared.npz", source_original["npz_sha256"]))
    normalizer_dir = LEARNING / "normalizer"
    normalizer_manifest = read(bound(normalizer_dir / "normalizer.json", source_original["normalizer_manifest_sha256"]))
    bound(normalizer_dir / "statistics.npz", normalizer_manifest["npz_sha256"])
    nmeta = normalizer_manifest["normalizer"]
    normalizer = load_normalizer(normalizer_dir, nmeta["schema"], nmeta["dataset_identity"], manifest["normalizer_hash"])
    original_episode = {row["episode_index"]: row for row in source_original["preparation"]["episodes"]}
    core_plans = {p["episode_id"]: p for p in collection["plans"]}
    config = manifest["physical_config"]
    evaluation = training_run / "prediction_evaluation"
    eval_index = read(bound(evaluation / "artifact_hashes.json"))["sha256"]
    query, query_next, action = None, None, None
    saved_predictions, target_identity = {}, None
    for arm in ARMS:
        result = read(bound(evaluation / arm / "development.json", eval_index[arm + "/development.json"]))
        arrays = load_npz(bound(evaluation / arm / "development_traces.npz", eval_index[arm + "/development_traces.npz"]))
        ids = result["core_H10_free"]["identities"]
        targets = [index for index, item in enumerate(ids) if item["family_id"] == TARGET_FAMILY
                   and item["family_rank"] == 1
                   and item["y_condition"] == "Y1" and item["stop_condition"] == "S0"]
        check(arm + "_unique_target_head", len(targets) == 1)
        episode = targets[0]; row = int(np.flatnonzero(arrays["head_indices"] == episode)[0])*5+1
        s, t, a = arrays["actual_states"][episode, 1], arrays["actual_states"][episode, 2], arrays["actual_actions"][episode, 1]
        check(arm + "_head_is_actual_boundary1_to2", np.array_equal(arrays["head_states"][row, 0], s)
              and arrays["head_failure_steps"][row] == 1 and not arrays["head_valid"][row]
              and str(arrays["actual_regimes"][episode, 1]) == "transition")
        if query is None: query, query_next, action, target_identity = s.copy(), t.copy(), a.copy(), ids[episode]
        check(arm + "_same_actual_query", np.array_equal(query, s) and np.array_equal(query_next, t) and np.array_equal(action, a))
        predicted = arrays["head_states"][row, 1]
        saved_predictions[arm] = {"saved_head_row": row, "valid": False,
            "failure_reason": str(arrays["head_failure_reasons"][row]), "predicted_next_state": predicted.tolist(),
            "saved_predicted_delta": arrays["head_raw_deltas"][row, 0].tolist(),
            "predicted_geometry": geometry(predicted, config)}
    query_features, query32, query64, query_numpy64 = model_features(query, action, normalizer)
    new_core_cache = {}
    def new_core(episode):
        if episode in new_core_cache: return new_core_cache[episode]
        plan = core_plans[episode]
        require(plan["semantic_split"] == "train" and plan["training_eligible"] and not plan["warmup_training_eligible"], "Nontrain/warmup new input")
        folder = MAIN / "episodes" / episode
        mpath = folder / "manifest.json"; relative = mpath.relative_to(MAIN).as_posix()
        m = read(bound(mpath, bindings["raw_core_artifact_sha256"][relative])); raw = m["episodes"][0]
        require(raw["episode_id"] == episode and raw["split"] == "train", "Raw train identity differs")
        path = folder / raw["npz"]
        bound(path, bindings["raw_core_artifact_sha256"][path.relative_to(MAIN).as_posix()])
        arrays = load_npz(path, ("state", "action_applied", "transition_valid", "contact_summary"))
        require(arrays["transition_valid"].all(), "Invalid train transitions cannot enter neighborhood")
        arrays["regimes"] = contact_regimes(arrays["contact_summary"])
        new_core_cache[episode] = arrays
        return arrays
    arms, complete_rankings, top_union = {}, {}, set()
    for arm in ARMS:
        path = training_run / "arms" / arm / "training_trace.npz"
        trace = load_npz(bound(path, training_index[path.relative_to(training_run).as_posix()]))
        exposure, mass = exposure_by_window(trace)
        population = {}
        for slot in range(2960):
            for depth in range(5):
                state, next_state = prepared[arm + "_states"][slot, depth:depth+2]
                applied = prepared[arm + "_actions"][slot, depth]
                require(prepared[arm + "_mask"][slot, depth], "Unexpected masked training input")
                if arm != "R" and slot in slots:
                    entry = slots[slot]; episode = entry["arm_episodes"][arm]
                    transition = entry["window_start"] + depth; raw = new_core(episode)
                    require(np.array_equal(state, raw["state"][transition]) and np.array_equal(next_state, raw["state"][transition+1])
                            and np.array_equal(applied, raw["action_applied"][transition]), "New tensor/raw mismatch")
                    source = {"source": "new_core", "episode_id": episode, "family_id": entry["family_id"],
                              "transition_index": transition, "regime": str(raw["regimes"][transition])}
                    identity = "new:" + episode + ":" + f"{transition:03d}"
                else:
                    index = int(prepared["old_window_rows"][slot, depth]); ep = original_episode[int(original["episode_index"][index])]
                    require(original["split_code"][index] == 0 and ep["split"] == "train"
                            and np.array_equal(state, original["current_state"][index])
                            and np.array_equal(next_state, original["next_state"][index])
                            and np.array_equal(applied, original["applied_action"][index]), "Old train tensor/raw mismatch")
                    source = {"source": "old_train", "episode_id": ep["episode_id"], "family_id": ep["family_id"],
                              "transition_index": int(original["transition_index"][index]), "regime": REGIMES[int(original["regime_code"][index])]}
                    identity = f"old:prepared-row:{index:05d}"
                add_occurrence(population, identity, source, state, applied, next_state, slot, depth, exposure[slot], mass[slot])
        rows = [population[key] for key in sorted(population)]; ids = [r["transition_id"] for r in rows]
        states, actions = np.stack([r["state"] for r in rows]), np.stack([r["action"] for r in rows])
        features, values32, values64, numpy64 = model_features(states, actions, normalizer)
        order, ranks, distance, squared = ranked_distance(values32, query32, ids)
        order64, _, distance64, _ = ranked_distance(values64, query64, ids)
        order_numpy64, _, _, _ = ranked_distance(numpy64, query_numpy64, ids)
        check(arm + "_all_exposures_reconciled", sum(r["teacher_transition_occurrences"] for r in rows) == 314960
              and sum(r["rollout_initial_state_occurrences"] for r in rows) == 62992
              and sum(len(r["window_memberships"]) for r in rows) == 14800)
        top = [summarize_neighbor(rows[i], ranks[i], distance[i], squared[i], query, query_next, config) for i in order[:10]]
        top_union.update(n["transition_id"] for n in top)
        feature_ranges = {name: {"query_physical": float(query_features[k]), "minimum": float(features[:, k].min()),
            "maximum": float(features[:, k].max()), "inside_univariate_observed_range": bool(features[:, k].min() <= query_features[k] <= features[:, k].max())}
            for k, name in enumerate(FEATURE_FIELDS)}
        source_nearest = {}
        for source in ("old_train", "new_core"):
            matching = [i for i in order if rows[i]["source"] == source]
            source_nearest[source] = None if not matching else {"rank": int(ranks[matching[0]]), "transition_id": ids[matching[0]],
                                                               "distance_normalized_L2": float(distance[matching[0]])}
        arms[arm] = {"unique_train_transitions": len(rows), "unique_train_families": len({r["family_id"] for r in rows}),
            "counts_by_source": dict(Counter(r["source"] for r in rows)), "counts_by_actual_regime": dict(Counter(r["regime"] for r in rows)),
            "top10_unique_train_families": len({r["family_id"] for r in top}), "top10_families": sorted({r["family_id"] for r in top}),
            "top10_counts_by_source": dict(Counter(r["source"] for r in top)), "top10_counts_by_actual_regime": dict(Counter(r["regime"] for r in top)),
            "top10": top, "nearest_by_source": source_nearest,
            "feature_univariate_ranges": feature_ranges,
            "boundary_dtype_sensitivity": {"float32_cast_changes_any_rank": not np.array_equal(order, order64),
                "float32_vs_precast_float64_top10_identical_order": np.array_equal(order[:10], order64[:10]),
                "numpy_vs_torch_float64_top10_identical_order": np.array_equal(order_numpy64[:10], order64[:10]),
                "maximum_abs_distance_change_due_to_float32": float(np.max(np.abs(distance-distance64))),
                "maximum_abs_numpy_torch_normalized_feature_difference": float(np.max(np.abs(values64-numpy64)))},
            "exposure_totals": {"teacher_transition_occurrences": 314960, "rollout_initial_state_occurrences": 62992,
                                "nominal_teacher_batch_mean_mass": float(sum(r["nominal_teacher_batch_mean_mass"] for r in rows))}}
        complete_rankings[arm] = {ids[i]: {"rank": int(ranks[i]), "distance_normalized_L2": float(distance[i]),
            "teacher_transition_occurrences": rows[i]["teacher_transition_occurrences"],
            "rollout_initial_state_occurrences": rows[i]["rollout_initial_state_occurrences"],
            "window_membership_count": len(rows[i]["window_memberships"])} for i in range(len(rows))}
    cross_arm = [{"transition_id": identity, "arms": {arm: complete_rankings[arm].get(identity) for arm in ARMS}}
                 for identity in sorted(top_union)]
    check("every_read_source_unchanged", all(sha(ROOT / rel) == value for rel, value in before.items()))
    return {"schema": "p4-contact-input-support-v1", "status": "DESCRIPTIVE_COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "registered_sources_checked": expected_sources is not None,
        "source_sha256": before, "checks": checks, "target": {**target_identity, "transition_zero_based": 1,
            "actual_regime": "transition", "state": query.tolist(), "action_m_s": action.tolist(), "next_state": query_next.tolist(),
            "true_target_delta": state_delta(query, query_next).tolist(), "features_physical": query_features.tolist(),
            "features_normalized_float32": query32.tolist(), "current_geometry": geometry(query, config),
            "next_geometry": geometry(query_next, config), "saved_predictions": saved_predictions},
        "metric": {"ranking": "Euclidean L2 over frozen normalized 11-dimensional float32 model inputs; distance arithmetic float64",
            "feature_fields": list(FEATURE_FIELDS), "feature_physical_units": list(FEATURE_UNITS),
            "delta_fields": list(DELTA_FIELDS), "delta_units": list(DELTA_UNITS),
            "weights": "equal coordinates after original train scale; not learned similarity",
            "tie_break": "lexicographic stable transition ID", "top_k": 10,
            "normalizer_hash": normalizer.metadata["normalizer_hash"], "normalizer_refit": False,
            "absolute_world_xy_in_model_features": False, "body_frame_or_gap_used_for_ranking": False},
        "arms": arms, "top10_union_cross_arm_ranks_and_exposure": cross_arm,
        "scope": ["Unique recorded transitions remove K5 overlap only; nearby transitions and families remain correlated.",
                  "Actual contact regime and target delta are annotations, never used in neighbor selection.",
                  "L_one teacher-input and rollout-target exposure differs from actual-state rollout initialization; later rollout inputs are model predictions.",
                  "Nominal teacher batch mean mass sums 1/(B*K) per transition appearance, with its ten normalized squared errors averaged internally. It is not realized gradient influence or model credit.",
                  "Univariate feature coverage is not joint coverage; neighbor distance is not a calibrated uncertainty or sufficiency test.",
                  "Normalization defines an arbitrary metric relative to physical contact geometry; inspect coordinate contributions and body-frame gap separately.",
                  "World absolute XY is absent from the model input; physical positions are shown only for context and must not dominate a substitute raw-state distance.",
                  "Closer or more frequent training neighbors do not establish adequate learning, absence/presence of overfitting, causality, or task improvement.",
                  "New contact regimes use frozen raw force-summary fractions validated in the prior collection audit; this script does not redo full force validation."],
        "execution": {"model_constructions": 0, "model_forwards": 0, "optimizer_updates": 0, "physics_steps": 0,
                      "normalizer_refits": 0, "torch_operations": "representation/normalization tensors only"}}


def run(output, expected_sources=None):
    output = Path(output)
    require(not output.exists(), "Never overwrite an existing analysis")
    result = analyze(expected_sources)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False,
                  default=lambda item: item.item() if isinstance(item, np.generic) else (_ for _ in ()).throw(TypeError(type(item).__name__)))
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--registration-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(sha(args.registration) == args.registration_sha256, "New diagnostic registration hash differs")
    result = run(args.output, read(args.registration)["bindings_sha256"])
    print(json.dumps({"status": result["status"], "checks": len(result["checks"]),
                      "top10_families": {arm: result["arms"][arm]["top10_unique_train_families"] for arm in ARMS}}))


if __name__ == "__main__": main()
