"""Independent saved-gradient oracle; no model or autograd execution.

All vector arithmetic is recomputed in NumPy float64 from saved float32
component gradients. A cosine describes Euclidean local geometry only. It
does not establish a finite Adam update or explain an entire training path.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GRADIENT_NAMES = ("one_step", "rollout", "relative_teacher", "relative_rollout", "total")
EPS32 = float(np.finfo(np.float32).eps)
CLOSURE_EPS_MULTIPLIER = 64.
CLOSURE_ABSOLUTE_FLOOR = 1e-12
NORM_UNDEFINED_FLOOR = 1e-12
NEAR_ORTHOGONAL_COSINE = 1e-6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def tensor_tree_hash(value):
    """Inspect serialized tensors without constructing their owning models."""
    import torch
    hasher = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            hasher.update(b"tensor" + str(item.dtype).encode() + json.dumps(list(item.shape)).encode())
            hasher.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            hasher.update(b"dict")
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                visit(key); visit(item[key])
        elif isinstance(item, (list, tuple)):
            hasher.update(type(item).__name__.encode())
            for child in item: visit(child)
        else:
            hasher.update(type(item).__name__.encode() + json.dumps(item, sort_keys=True, allow_nan=False).encode())
    visit(value)
    return hasher.hexdigest()


def vector(value):
    value = np.asarray(value)
    require(value.ndim == 1 and value.size > 0 and value.dtype in (np.float32, np.float64)
            and np.isfinite(value).all(), "Nonempty finite floating gradient vector required")
    return value.astype(np.float64)


def norm(value):
    value = vector(value)
    return float(np.sqrt(np.sum(value * value, dtype=np.float64)))


def dot(first, second):
    first, second = vector(first), vector(second)
    require(first.shape == second.shape, "Gradient vectors must have identical dimensions")
    return float(np.sum(first * second, dtype=np.float64))


def cosine(first, second):
    first_norm, second_norm = norm(first), norm(second)
    return (None if min(first_norm, second_norm) <= NORM_UNDEFINED_FLOOR
            else float(dot(first, second) / (first_norm * second_norm)))


def cosine_relation(value):
    if value is None: return "UNDEFINED_NEAR_ZERO_NORM"
    require(type(value) in (float, int) and np.isfinite(value), "Finite cosine or null required")
    if abs(value) <= NEAR_ORTHOGONAL_COSINE: return "NEAR_ORTHOGONAL"
    return "OPPOSED" if value < 0 else "ALIGNED"


def pairwise(vectors):
    vectors = np.asarray(vectors)
    require(vectors.ndim == 2 and vectors.shape[0] > 0 and vectors.shape[1] > 0,
            "Nonempty gradient matrix required")
    count = len(vectors)
    return {"dot": [[dot(vectors[a], vectors[b]) for b in range(count)] for a in range(count)],
            "cosine": [[cosine(vectors[a], vectors[b]) for b in range(count)] for a in range(count)],
            "norm": [norm(value) for value in vectors]}


def component_geometry(raw_gradients):
    """Check the fixed objective using raw components, not fitted coefficients."""
    raw_gradients = np.asarray(raw_gradients)
    require(raw_gradients.ndim == 2 and raw_gradients.shape[0] == 5
            and raw_gradients.shape[1] > 0 and raw_gradients.dtype == np.float32
            and np.isfinite(raw_gradients).all(), "Five finite raw float32 gradient vectors required")
    raw = raw_gradients.astype(np.float64)
    one, rollout, relative_teacher, relative_free, actual_total = raw
    base = one + .1 * rollout
    auxiliary = .2 * (relative_teacher + .1 * relative_free)
    expected_total = base + auxiliary
    weighted = np.stack((one, .1 * rollout, .2 * relative_teacher, .02 * relative_free))
    component_norm_sum = sum(norm(row) for row in weighted)
    bound = CLOSURE_EPS_MULTIPLIER * EPS32 * component_norm_sum + CLOSURE_ABSOLUTE_FLOOR
    error = norm(actual_total - expected_total)
    denominator = norm(base) + norm(auxiliary)
    return {
        "base": base, "auxiliary": auxiliary, "expected_total": expected_total,
        "raw_pairwise": pairwise(raw), "weighted_norms": [norm(row) for row in weighted],
        "base_norm": norm(base), "auxiliary_norm": norm(auxiliary), "total_norm": norm(actual_total),
        "base_auxiliary_dot": dot(base, auxiliary), "base_auxiliary_cosine": cosine(base, auxiliary),
        "base_total_dot": dot(base, actual_total), "auxiliary_total_dot": dot(auxiliary, actual_total),
        "auxiliary_to_base_norm_ratio": None if norm(base) <= NORM_UNDEFINED_FLOOR else norm(auxiliary) / norm(base),
        "triangle_norm_ratio": None if denominator <= NORM_UNDEFINED_FLOOR else norm(expected_total) / denominator,
        "closure": {"l2_error": error, "l2_bound": bound, "pass": error <= bound,
                    "weighted_component_norm_sum": component_norm_sum,
                    "eps_multiplier": CLOSURE_EPS_MULTIPLIER,
                    "absolute_floor": CLOSURE_ABSOLUTE_FLOOR},
    }


def algebraic_clip(gradient, maximum=1., epsilon=1e-6):
    """Float64 flat-vector diagnostic, not PyTorch's parameterwise float32 norm."""
    require(type(maximum) in (float, int) and np.isfinite(maximum) and maximum > 0
            and type(epsilon) in (float, int) and np.isfinite(epsilon) and epsilon > 0,
            "Positive finite clipping norm and epsilon required")
    gradient = vector(gradient)
    original_norm = norm(gradient)
    coefficient = min(1., float(maximum / (original_norm + epsilon)))
    result = coefficient * gradient
    return {"gradient": result, "coefficient": coefficient,
            "original_norm": original_norm, "clipped_norm": norm(result),
            "direction_cosine": cosine(gradient, result)}


def head_partition_indices(order):
    by_name = {item["name"]: item for item in order}
    require("network.4.weight" in by_name and "network.4.bias" in by_name,
            "Final ten-output Linear weight and bias required")
    weight, bias = by_name["network.4.weight"], by_name["network.4.bias"]
    require(len(weight["shape"]) == 2 and weight["shape"][0] == 10 and weight["shape"][1] > 0
            and bias["shape"] == [10], "Final Linear must have ten output rows")
    matrix = np.arange(weight["start"], weight["end"], dtype=np.int64).reshape(weight["shape"])
    biases = np.arange(bias["start"], bias["end"], dtype=np.int64)
    require(len(biases) == 10, "Ten bias coordinates required")
    rows = {"object_xy": [0, 1], "yaw": [2], "object_velocity": [3, 4, 5],
            "pusher_xy": [6, 7], "pusher_velocity": [8, 9]}
    return {name: {"rows": selected,
                  "indices": np.concatenate((matrix[selected].reshape(-1), biases[selected]))}
            for name, selected in rows.items()}


class Audit:
    def __init__(self):
        self.checks = []
        self.sources = {}
        self.observed = []

    def check(self, name, passed, **details):
        self.checks.append({"name": name, "pass": bool(passed), **details})
        require(passed, name)

    def number(self, name, actual, expected, *, rtol=1e-10, atol=1e-12):
        self.check(name, actual is None if expected is None else actual is not None
                   and np.isclose(actual, expected, rtol=rtol, atol=atol))

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        require(path.is_relative_to(ROOT), "Input path outside project")
        digest = sha(path)
        relative = path.relative_to(ROOT).as_posix()
        self.check("immutable:" + relative, expected is None or digest == expected)
        if relative in self.sources:
            self.check("repeat_immutable:" + relative, digest == self.sources[relative])
        self.sources[relative] = digest
        return path

    def batches(self, run):
        """Rebuild train identities and force-derived regimes without project loaders."""
        sources = {}
        def source(relative, expected=None):
            path = self.bind(ROOT / relative, expected)
            sources[relative] = sha(path)
            return path
        def document(relative, expected=None): return read(source(relative, expected))
        def data(relative, expected=None): return load_arrays(source(relative, expected))
        old = "runs/p4-data-retrain-20261013-02"
        relative = "runs/p4-relative-loss-20261014-01"
        learning = "runs/p3-02-20261009-01/learning_data"
        collection = "runs/p4-data-main-20261013-01"
        prior = document(relative + "/manifest.json")
        document(relative + "/registration.json", prior["registration_sha256"])
        bindings = document(relative + "/dataset_bindings.json", prior["dataset_bindings_sha256"])
        frozen = data(relative + "/prepared_training.npz", prior["prepared_training_sha256"])
        old_data = data(bindings["source"], bindings["source_sha256"])
        self.check("original_D11_dataset_identity", bindings["source"] == old + "/prepared_training.npz"
                   and bindings["data_hash"] == prior["training_provenance"]["D11"]["data_hash"]
                   and bindings["normalizer_refit"] is False
                   and bindings["development_episodes_loaded_for_training"] == bindings["warmup_episodes_loaded_for_training"] == 0)
        for key, shape, dtype in (("states", (2960, 6, 10), np.float64),
                                   ("actions", (2960, 5, 2), np.float64), ("mask", (2960, 5), np.bool_)):
            self.check("frozen_D11_exact:" + key, frozen[key].shape == shape and frozen[key].dtype == dtype
                       and np.array_equal(frozen[key], old_data["D11_" + key]) and np.isfinite(frozen[key]).all())
        self.check("recorded_full_valid_denominator", frozen["mask"].all())
        original_meta = document(learning + "/manifest.json")
        original = data(learning + "/prepared.npz", original_meta["npz_sha256"])
        regimes = ("noncontact", "sustained_contact", "transition")
        self.check("original_label_semantics", original_meta["preparation"]["regime_codes"] == dict(zip(regimes, range(3)))
                   and original_meta["preparation"]["split_codes"]["train"] == 0)
        episodes = {item["episode_index"]: item for item in original_meta["preparation"]["episodes"]}
        upstream = bindings["original_training_provenance"]
        registration = document(collection + "/registration.json", upstream["collection_registration_sha256"])
        slot_doc = document(collection + "/slots_bound.json", upstream["slots_bound_sha256"])
        inventory = document(collection + "/artifact_hashes.json")["sha256"]
        summary = document(collection + "/summary.json", inventory["summary.json"])
        self.check("collection_complete", summary["status"] == "PASS" and summary["failure"] is None)
        entries = {entry["slot"]: entry for entry in slot_doc["entries"]}
        self.check("fixed384_replacement_slots", len(entries) == 384
                   and np.array_equal(np.asarray(list(entries), np.int64), old_data["replacement_slots"]))
        plans = {item["episode_id"]: item for item in registration["plans"]}
        outputs = {item["label"]: item for item in summary["episodes"]}
        raw_cache, records = {}, []
        for slot in range(2960):
            if slot in entries:
                entry = entries[slot]
                episode = entry["arm_episodes"]["D11"]
                plan, output = plans[episode], outputs[episode]
                self.check(f"slot{slot}:eligible_D11_train_core", plan["semantic_split"] == plan["split"] == "train"
                    and plan["training_eligible"] is True and plan["core_training_eligible"] is True
                    and plan["warmup_training_eligible"] is False and plan["development_evaluation_eligible"] is False
                    and output["role"] == "core" and plan["y_condition"] == "Y1" and plan["stop_condition"] == "S1"
                    and plan["family_id"] == entry["family_id"] == output["family_id"])
                if episode not in raw_cache:
                    prefix = collection + "/" + output["dataset"]
                    manifest = document(prefix + "/manifest.json", output["manifest_sha256"])
                    item, = manifest["episodes"]
                    self.check(episode + ":raw_identity", item["episode_id"] == episode and item["split"] == "train"
                               and item["family_id"] == plan["family_id"])
                    raw = data(prefix + "/" + item["npz"], item["npz_sha256"])
                    sidecar = document(prefix + "/" + item["json"], item["json_sha256"])
                    self.check(episode + ":only_actual_train_core", sidecar["episode_id"] == episode
                        and sidecar["collection_policy"]["role"] == "core"
                        and sidecar["collection_policy"]["training_eligible"] is True
                        and sidecar["collection_policy"]["development_evaluation_eligible"] is False
                        and sidecar["collection_policy"]["predeclared_plan"] == plan
                        and sidecar["diagnostic_nonfinite_paths"] == [])
                    actual_regimes, fraction = [], []
                    threshold = sidecar["contract"]["active_contact_force_min_N"]
                    for diagnostic in sidecar["diagnostics"]:
                        active = [any(contact["pair"] == "pusher_object" and contact["normal_force_N"] > threshold
                                      for contact in sample["contacts"]) for sample in diagnostic["samples"]]
                        self.check(episode + f":nonempty_force_samples{len(actual_regimes)}", bool(active))
                        actual_regimes.append("sustained_contact" if all(active) else "transition" if any(active) else "noncontact")
                        fraction.append(sum(active) / len(active))
                    self.check(episode + ":force_fraction_summary_exact", len(actual_regimes) == 20
                               and np.array_equal(raw["contact_summary"][:, 2], np.asarray(fraction)))
                    self.check(episode + ":raw_state_action_integrity", raw["state"].shape == (21, 10)
                        and raw["action_applied"].shape == (20, 2) and raw["transition_valid"].all()
                        and np.array_equal(raw["action_applied"], raw["action_requested"])
                        and np.array_equal(raw["action_applied"], np.asarray(plan["core_actions"]))
                        and np.all(np.abs(raw["actual_dt"] - .1) < 1e-10))
                    raw_cache[episode] = {"states": raw["state"], "actions": raw["action_applied"], "regimes": actual_regimes}
                raw, start = raw_cache[episode], entry["window_start"]
                states, actions = raw["states"][start:start + 6], raw["actions"][start:start + 5]
                row = {"source": "new_core", "role": "core", "episode_id": episode,
                    "family_id": entry["family_id"], "family_rank": entry["family_rank"], "geometry": plan["geometry"],
                    "window_start": start, "transition_indices": list(range(start, start + 5)),
                    "actual_contact_regimes": raw["regimes"][start:start + 5],
                    "prepared_row_indices": None, "y_condition": "Y1", "stop_condition": "S1"}
            else:
                old_rows = old_data["old_window_rows"][slot]
                episode_index = int(original["episode_index"][old_rows[0]])
                episode = episodes[episode_index]
                self.check(f"slot{slot}:original_train_eligibility", episode["split"] == "train"
                    and np.all(original["split_code"][old_rows] == 0)
                    and np.all(original["episode_index"][old_rows] == episode_index)
                    and np.all(np.diff(original["transition_index"][old_rows]) == 1))
                states = np.concatenate((original["current_state"][old_rows[:1]], original["next_state"][old_rows]))
                actions = original["applied_action"][old_rows]
                indices = original["transition_index"][old_rows].astype(int).tolist()
                row = {"source": "old_train", "role": "original_train_transition_table", "episode_id": episode["episode_id"],
                    "family_id": episode["family_id"], "family_rank": None, "geometry": "original_unspecified",
                    "window_start": indices[0], "transition_indices": indices,
                    "actual_contact_regimes": [regimes[int(code)] for code in original["regime_code"][old_rows]],
                    "prepared_row_indices": old_rows.tolist(), "y_condition": None, "stop_condition": None}
            self.check(f"slot{slot}:actual_frozen_tensor", np.array_equal(states, frozen["states"][slot])
                       and np.array_equal(actions, frozen["actions"][slot]))
            row.update({"slot": slot, "split": "train", "training_eligible": True,
                "all_zero_actions": bool(np.all(actions == 0.)),
                "all_nonzero_actions": bool(np.all(np.any(actions != 0., axis=1))),
                "transition_ids": [row["source"] + ":" + row["episode_id"] + ":" + str(index)
                                   for index in row["transition_indices"]]})
            records.append(row)
        self.check("raw_D11_cores24_only", len(raw_cache) == 24)
        historical = data("runs/p4-02-20261012-01/seeds/seed-11/SK-lambda0.1/training_trace.npz")
        new = [row for row in records if row["source"] == "new_core"]
        def one_per_family(rows):
            families = sorted({row["family_id"] for row in rows})
            return [min((row for row in rows if row["family_id"] == family),
                        key=lambda row: (row["window_start"], row["slot"]))["slot"] for family in families]
        selected = {"historical_update1": historical["window_positions"][:64].astype(int).tolist(),
            "D11_zero5": one_per_family([row for row in new if row["all_zero_actions"]]),
            "D11_transition": one_per_family([row for row in new if "transition" in row["actual_contact_regimes"]]),
            "D11_noncontact": one_per_family([row for row in new if set(row["actual_contact_regimes"]) == {"noncontact"}])}
        self.check("four_fixed_probe_population_counts", [len(value) for value in selected.values()] == [64, 24, 16, 8])
        report, tensors = read(run / "batch_manifest.json"), load_arrays(run / "probe_batches.npz")
        self.check("four_probe_name_inventory", set(report) == set(selected))
        for name, slots in selected.items():
            result = report[name]
            chosen = [records[slot] for slot in slots]
            self.check(name + ":selected_slots_and_all_provenance_exact", result["slots"] == slots
                       and result["window_provenance"] == chosen)
            self.check(name + ":same_bound_input_sources", result["source_sha256"] == sources)
            for key in ("states", "actions", "mask"):
                self.check(name + ":exact_batch_tensor:" + key, np.array_equal(tensors[name + "_" + key], frozen[key][slots]))
            self.check(name + ":exact_saved_slot_order", np.array_equal(tensors[name + "_slots"], np.asarray(slots, np.int64)))
            families = sorted({row["family_id"] for row in chosen})
            transition_ids = {value for row in chosen for value in row["transition_ids"]}
            expected_counts = {"windows": len(slots), "families": len(families), "transition_occurrences": 5 * len(slots),
                "unique_original_transitions": len(transition_ids),
                "source_windows": dict(Counter(row["source"] for row in chosen)),
                "actual_regime_occurrences": dict(Counter(value for row in chosen for value in row["actual_contact_regimes"])),
                "geometry_windows": dict(Counter(row["geometry"] for row in chosen)),
                "all_zero_windows": sum(row["all_zero_actions"] for row in chosen)}
            self.check(name + ":family_transition_denominators_exact", result["families"] == families and result["counts"] == expected_counts)
            if name == "D11_zero5": self.check("zero5_starts_state10", all(row["window_start"] == 10 for row in chosen))
            if name in ("D11_transition", "D11_noncontact"):
                self.check(name + ":earliest_start0_nonzero", all(row["window_start"] == 0 and row["all_nonzero_actions"] for row in chosen))
            for alternate, alternate_slots in selected.items():
                if alternate == name: continue
                alternate_rows = [records[slot] for slot in alternate_slots]
                other_transitions = {value for row in alternate_rows for value in row["transition_ids"]}
                expected = {"shared_slots": sorted(set(slots) & set(alternate_slots)),
                    "shared_families": sorted(set(families) & {row["family_id"] for row in alternate_rows}),
                    "shared_unique_original_transitions": len(transition_ids & other_transitions)}
                self.check(name + ":overlap:" + alternate, result["overlap"][alternate] == expected)
        self.batch_manifest, self.selected_slots, self.dataset_hash = report, selected, bindings["data_hash"]

    def pair(self, name, record, first, second):
        first_norm, second_norm, product = norm(first), norm(second), dot(first, second)
        angle = cosine(first, second)
        expected = {"first_norm": first_norm, "second_norm": second_norm, "dot": product,
            "cosine": angle, "second_to_first_norm_ratio": second_norm / first_norm
            if first_norm > NORM_UNDEFINED_FLOOR else None,
            "first_projection_gain_on_second": product / first_norm ** 2
            if first_norm > NORM_UNDEFINED_FLOOR else None,
            "second_projection_gain_on_first": product / second_norm ** 2
            if second_norm > NORM_UNDEFINED_FLOOR else None,
            "first_loss_derivative_along_negative_second": -product}
        for key, value in expected.items():
            self.number(name + ":" + key, record[key], value)
        self.check(name + ":relation", record["relation"] == cosine_relation(angle))

    def clip(self, name, record, gradient, order):
        self.check(name + ":fixed_clipping_contract", record["maximum_norm"] == 1.
                   and record["epsilon"] == 1e-6
                   and record["applied_to_parameters_or_grad_buffers"] is False)
        block_norms = [norm(gradient[item["start"]:item["end"]]) for item in order]
        self.check(name + ":parameter_norm_count", len(record["parameter_norms_float32"]) == len(order))
        for index, expected in enumerate(block_norms):
            self.number(name + f":parameter32norm{index}", record["parameter_norms_float32"][index],
                        expected, rtol=64 * EPS32, atol=1e-12)
        self.number(name + ":parameterwise32norm", record["parameterwise_float32_norm"], norm(gradient),
                    rtol=64 * EPS32, atol=1e-12)
        self.number(name + ":flat64norm", record["flat_float64_norm"], norm(gradient))
        value = np.float32(record["parameterwise_float32_norm"])
        coefficient = float(np.minimum(np.float32(1.) / (value + np.float32(1e-6)), np.float32(1.)))
        self.number(name + ":float32_coefficient_epsilon", record["coefficient"], coefficient,
                    rtol=2 * EPS32, atol=0.)
        self.check(name + ":scale_and_direction_flags", record["would_scale"] == (coefficient < 1.)
                   and record["positive_global_scalar_preserves_Euclidean_direction"]
                   == ((coefficient > 0.) if norm(gradient) > 1e-12 else None))

    def gradient(self, name, record, raw, derived):
        self.check(name + ":component_order", record["component_order"] == list(GRADIENT_NAMES)
                   and record["component_weights"] == [1., .1, .2, .02])
        result = component_geometry(raw)
        self.check(name + ":finite_float32_parameter_vectors", raw.shape == (5, 19338))
        order = record["parameter_order"]
        shapes = [("network.0.weight", [128, 11]), ("network.0.bias", [128]),
                  ("network.2.weight", [128, 128]), ("network.2.bias", [128]),
                  ("network.4.weight", [10, 128]), ("network.4.bias", [10])]
        expected_order, offset = [], 0
        for label, shape in shapes:
            size = int(np.prod(shape))
            expected_order.append({"name": label, "shape": shape, "numel": size, "start": offset, "end": offset + size})
            offset += size
        self.check(name + ":registered_parameter_order", order == expected_order and offset == 19338
                   and record["parameter_count"] == offset)
        base, auxiliary, total = result["base"], result["auxiliary"], raw[4].astype(np.float64)
        expected_derived = {"base": base, "weighted_aux": auxiliary, "total_from_components": result["expected_total"]}
        for key, value in expected_derived.items():
            self.check(name + ":derived_float32_rounding:" + key, derived[key].dtype == np.float32
                       and np.array_equal(derived[key], value.astype(np.float32)))
        summary = record["vector_summary"]
        self.check(name + ":fixed_direction_thresholds", summary["norm_floor"] == 1e-12
                   and summary["near_orthogonal_absolute_cosine"] == 1e-6)
        for index, label in enumerate(GRADIENT_NAMES):
            self.number(name + ":raw_norm:" + label, summary["raw_component_norms"][label], norm(raw[index]))
        for key, value in {"base": base, "weighted_aux": auxiliary,
                           "total_from_components": result["expected_total"], "independent_total": total}.items():
            self.number(name + ":derived_norm:" + key, summary["derived_norms"][key], norm(value))
        self.pair(name + ":base_aux", summary["base_vs_weighted_aux"], base, auxiliary)
        self.pair(name + ":base_total", summary["base_vs_independent_total"], base, total)
        self.pair(name + ":aux_total", summary["weighted_aux_vs_independent_total"], auxiliary, total)
        self.pair(name + ":teacher_free", summary["raw_teacher_free"]["one_step_vs_rollout"], raw[0], raw[1])
        self.pair(name + ":relative_teacher_free", summary["raw_teacher_free"]["relative_teacher_vs_relative_rollout"], raw[2], raw[3])
        self.check(name + ":six_parameter_block_inventory", len(summary["parameter_blocks"]) == 6)
        for item, block in zip(order, summary["parameter_blocks"]):
            self.check(name + ":parameter_block_metadata:" + item["name"], all(block[key] == value for key, value in item.items()))
            selected = slice(item["start"], item["end"])
            self.pair(name + ":parameter_block_aux:" + item["name"], block["base_vs_aux"], base[selected], auxiliary[selected])
            self.pair(name + ":parameter_block_total:" + item["name"], block["base_vs_total"], base[selected], total[selected])
        partitions = head_partition_indices(order)
        head = summary["output_head_groups"]
        self.check(name + ":ten_output_head_inventory", head["status"] == "APPLICABLE" and set(head["groups"]) == set(partitions))
        all_indices = []
        for group, item in partitions.items():
            block, indices = head["groups"][group], item["indices"]
            self.check(name + ":output_head_metadata:" + group,
                block["output_rows"] == item["rows"] and block["parameter_count"] == len(indices)
                and block["weight_parameter"] == "network.4.weight" and block["bias_parameter"] == "network.4.bias")
            self.pair(name + ":output_head_aux:" + group, block["base_vs_aux"], base[indices], auxiliary[indices])
            self.pair(name + ":output_head_total:" + group, block["base_vs_total"], base[indices], total[indices])
            all_indices.extend(indices.tolist())
        self.check(name + ":output_head_partition_exact_no_hidden", len(all_indices) == len(set(all_indices)) == 1290
                   and sorted(all_indices) == list(range(order[4]["start"], order[5]["end"])))
        for key, prefix in (("base_projection_gain", "base_vs_independent_total"),
                            ("aux_projection_gain", "weighted_aux_vs_independent_total")):
            self.number(name + ":projection_alias:" + key, summary[key], summary[prefix]["first_projection_gain_on_second"])
        closure = summary["closure"]
        self.number(name + ":closure_l2", closure["l2_residual"], result["closure"]["l2_error"])
        self.number(name + ":closure_max", closure["max_absolute_residual"],
                    float(np.max(np.abs(total - result["expected_total"]))))
        self.number(name + ":closure_relative_l2", closure["relative_l2_residual_descriptive"],
                    result["closure"]["l2_error"] / norm(total) if norm(total) > 1e-12 else None)
        self.number(name + ":closure_component_scale", closure["weighted_component_norm_sum"],
                    result["closure"]["weighted_component_norm_sum"])
        self.number(name + ":closure_bound", closure["l2_bound"], result["closure"]["l2_bound"])
        self.check(name + ":closure_pass", closure["pass"] == result["closure"]["pass"] and result["closure"]["pass"])
        self.clip(name + ":base_clip", summary["clip_diagnostic"]["base_from_component_sum_float32"], derived["base"], order)
        self.clip(name + ":total_clip", summary["clip_diagnostic"]["independent_total_float32"], raw[4], order)
        losses = record["loss_values"]
        self.number(name + ":loss_arithmetic", losses["total"], losses["one_step"] + .1 * losses["rollout"]
                    + .2 * (losses["relative_teacher"] + .1 * losses["relative_rollout"]))
        self.check(name + ":readonly_execution_counts", record["optimizer_steps"] == 0 and record["physics_steps"] == 0
                   and record["derivative_traversals"] == 5 and record["forward_calls"] == 6)
        count = record["denominators"]["valid_transitions"]
        self.check(name + ":six_forward_rows_shared_graph", record["forward_rows"] == [count] + [count // 5] * 5
                   and record["forward_total_rows"] == 2 * count
                   and record["denominators"]["base_scalar_terms"] == 10 * count
                   and record["denominators"]["relative_scalar_terms"] == 2 * count)
        self.observed.append({"probe": name, "base_auxiliary_cosine": result["base_auxiliary_cosine"],
            "auxiliary_to_base_norm_ratio": result["auxiliary_to_base_norm_ratio"],
            "triangle_norm_ratio_descriptive": result["triangle_norm_ratio"],
            "closure_l2_error": result["closure"]["l2_error"], "closure_bound": result["closure"]["l2_bound"]})

    def run(self, run):
        import torch
        protocol, manifest, summary = (read(run / name) for name in ("registration.json", "manifest.json", "summary.json"))
        self.check("completed_readonly_gradient_run", summary["execution_status"] == "PASS"
                   and summary["analysis_status"] == "DESCRIPTIVE_COMPLETE")
        for path, expected in read(run / "artifact_hashes.json")["sha256"].items():
            self.bind(run / path, expected)
        self.check("protocol_copy_manifest_result_binding", sha(run / "registration.json") == manifest["protocol_sha256"]
                   == summary["protocol_sha256"] and sha(run / "manifest.json") == summary["manifest_sha256"])
        for path, expected in protocol["bindings_sha256"].items(): self.bind(ROOT / path, expected)
        self.check("registered_readonly_design", protocol["snapshot_count"] == 7 and protocol["probe_count"] == 4
                   and protocol["allow_optimizer_updates"] is False and protocol["loss_weights"] == [1., .1, .2, .02])
        self.bind(run / "batch_manifest.json", manifest["batch_manifest_sha256"])
        self.bind(run / "probe_batches.npz", manifest["probe_batches_sha256"])
        self.batches(run)
        self.check("manifest_D11_hash", manifest["dataset_hash"] == self.dataset_hash and manifest["parameter_count"] == 19338)
        self.check("registered_batch_selections", protocol["batch_selections"] == self.selected_slots
                   and protocol["batch_counts"] == {name: value["counts"] for name, value in self.batch_manifest.items()})
        normalizer_folder = ROOT / "runs/p3-02-20261009-01/learning_data/normalizer"
        normalizer = read(normalizer_folder / "normalizer.json")
        self.bind(normalizer_folder / "statistics.npz", normalizer["npz_sha256"])
        scales = load_arrays(normalizer_folder / "statistics.npz")["delta_scale"]
        scale = np.sqrt(scales[:2] ** 2 + scales[6:8] ** 2)
        self.check("original_normalizer_no_refit", normalizer["normalizer"]["normalizer_hash"] == manifest["normalizer_hash"]
                   and summary["normalizer_refit"] is False)
        specs = manifest["model_snapshots"]
        expected_ids = ["common_initial"] + [f"{branch}_u{update:04d}" for branch in ("C0", "R1") for update in (100, 500, 1000)]
        self.check("fixed_seven_snapshot_inventory", [item["snapshot_id"] for item in specs] == expected_ids)
        self.check("registered_snapshot_specs", all(all(item[key] == value for key, value in registered.items())
                   for item, registered in zip(specs, protocol["snapshots"])) and len(protocol["snapshots"]) == 7)
        records, count_forward, count_rows, count_traversals = [], 0, 0, 0
        for spec in specs:
            snapshot = spec["snapshot_id"]
            checkpoint = ROOT / spec["checkpoint_path"]
            meta = read(self.bind(checkpoint / "checkpoint.json", spec["checkpoint_json_sha256"]))
            self.check(snapshot + ":checkpoint_metadata_integrity", meta["checkpoint_hash"] == spec["checkpoint_hash"]
                and meta["checkpoint_hash"] == hashlib.sha256(canonical({key: value for key, value in meta.items()
                                                                          if key != "checkpoint_hash"})).hexdigest())
            for filename in ("weights.pt", "training_state.pt"):
                self.bind(checkpoint / filename, meta["files"][filename])
                self.check(snapshot + ":manifest_file_identity:" + filename,
                           spec["files_sha256"][filename] == meta["files"][filename])
            weights = torch.load(checkpoint / "weights.pt", map_location="cpu", weights_only=True)
            weight_hash = tensor_tree_hash(weights)
            self.check(snapshot + ":checkpoint_normalizer_config", meta["normalizer_hash"] == manifest["normalizer_hash"]
                       and meta["config"] == manifest["physical_config"])
            if snapshot == "common_initial":
                self.check("common_initial_epoch100", meta["epoch"] == 100 and meta["updates"] == 1300
                           and "runs/p3-04-20261009-01/seeds/seed-11/checkpoints/epoch-100" == spec["checkpoint_path"])
            else:
                branch, update = snapshot.split("_u")
                self.check(snapshot + ":historical_snapshot_step_and_objective", meta["branch"] == branch
                           and meta["additional_updates"] == int(update) and meta["total_updates"] == 1300 + int(update)
                           and meta["relative_weight"] == (0. if branch == "C0" else .2)
                           and meta["dataset_hash"] == self.dataset_hash)
            for batch, slots in self.selected_slots.items():
                folder = run / "probes" / snapshot / batch
                record, gradient_arrays, residual = read(folder / "summary.json"), load_arrays(folder / "gradients.npz"), load_arrays(folder / "loss_residuals.npz")
                label = snapshot + ":" + batch
                self.check(label + ":same_snapshot_and_probe", record["snapshot_id"] == snapshot
                           and record["probe_name"] == batch and record["trained_objective"] == spec["trained_objective"]
                           and record["diagnostic_objective"] == protocol["diagnostic_objective"]
                           and record["batch_count"] == len(slots))
                self.check(label + ":weights_modes_buffers_preserved", record["before_weights_hash"] == weight_hash
                           == record["after_weights_hash"] and all(record["checks"].values())
                           and record["model_mode_before"] is False and record["model_mode_after"] is False)
                self.gradient(label, record, gradient_arrays["gradients"],
                    {key: gradient_arrays[key] for key in ("base", "weighted_aux", "total_from_components")})
                count = 5 * len(slots)
                self.check(label + ":registered_loss_denominator", record["denominators"]["valid_transitions"] == count)
                self.check(label + ":fixed_original_relative_scale", residual["relative_scale_m"].dtype == np.float64
                           and np.array_equal(residual["relative_scale_m"], scale)
                           and record["relative_scale_m"] == scale.tolist())
                for residual_key, loss_key in (("teacher_xy_errors", "relative_teacher"), ("free_xy_errors", "relative_rollout")):
                    xy = residual[residual_key]
                    self.check(label + ":saved_residual_shape:" + residual_key,
                               xy.shape == (count, 4) and xy.dtype == np.float64 and np.isfinite(xy).all())
                    value = float(np.square((xy[:, 2:] - xy[:, :2]) / scale).sum() / (2 * count))
                    self.number(label + ":independent_2N_loss:" + loss_key, record["loss_values"][loss_key], value)
                records.append({"snapshot_id": snapshot, "probe_name": batch, "trained_objective": spec["trained_objective"],
                    "batch_count": len(slots), "summary_path": (folder / "summary.json").relative_to(run).as_posix(),
                    "loss_values": record["loss_values"], "vector_summary": record["vector_summary"]})
                count_forward += record["forward_calls"]; count_rows += record["forward_total_rows"]
                count_traversals += record["derivative_traversals"]
        aggregate = read(run / "aggregate.json")
        self.check("aggregate_all28_exact_aliases", aggregate["row_count"] == len(records) == 28
                   and aggregate["rows"] == records)
        expected_counters = {"forward_calls": 168, "forward_rows": 7840, "derivative_traversals": 140, "completed_probes": 28}
        self.check("recorded_compute_counts", summary["counters"] == protocol["expected_counters"] == expected_counters
                   and count_forward == 168 and count_rows == 7840 and count_traversals == 140)
        self.check("no_new_updates_physics_evaluation_or_checkpoint", summary["new_optimizer_updates"] == 0
            and summary["new_physics_steps"] == 0 and summary["new_checkpoint_writes"] == 0
            and not any(summary["forbidden_execution_counts"].values())
            and summary["development_or_validation_probe_used"] is False and summary["final_test_used"] is False
            and summary["new_closed_loop_execution"] is False and summary["model_selection"] == "NONE"
            and summary["alignment_improvement"] == "NOT_EVALUATED" and summary["P5_01"] == "HOLD"
            and summary["global_rng_unchanged"] is True and summary["model_weights_unchanged"] is True
            and summary["gradient_buffers_unchanged"] is True and summary["model_mode_unchanged"] is True)


def audit(run):
    import torch
    checker = Audit()
    blocked_counts = {"learned_module_calls": 0, "autograd_grad_calls": 0, "backward_calls": 0}
    originals = (torch.nn.Module._call_impl, torch.autograd.grad, torch.Tensor.backward)
    rng_before = torch.get_rng_state().clone()
    def blocked(name):
        def call(*args, **kwargs):
            blocked_counts[name] += 1
            raise RuntimeError("Saved-gradient audit forbids " + name)
        return call
    torch.nn.Module._call_impl = blocked("learned_module_calls")
    torch.autograd.grad = blocked("autograd_grad_calls")
    torch.Tensor.backward = blocked("backward_calls")
    result = {"schema": "p4-gradient-probe-independent-review-v1", "status": "FAIL",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "run": run.relative_to(ROOT).as_posix(),
        "review_source_sha256": sha(__file__), "checks": checker.checks,
        "execution_during_review": {"model_instances": 0, "model_forwards": 0, "autograd_traversals": 0,
            "optimizer_updates": 0, "physics_steps": 0, "serialized_checkpoint_tensor_reads_allowed": True},
        "numeric_policy": {"scalar_float64_rtol": 1e-10, "scalar_float64_atol": 1e-12,
            "component_closure_epsilon_multiplier": 64., "component_closure_absolute_floor": 1e-12,
            "float32_parameter_norm_rtol": 64 * EPS32, "cosine_norm_undefined_floor": 1e-12,
            "near_orthogonal_absolute_cosine": 1e-6}}
    try:
        checker.run(run)
        checker.check("auditor_zero_forbidden_execution", not any(blocked_counts.values()))
        checker.check("auditor_global_rng_unchanged", torch.equal(rng_before, torch.get_rng_state()))
        result.update(status="PASS", observed=checker.observed, source_bindings_sha256=checker.sources,
            forbidden_execution_counts=blocked_counts,
            limitations=["Saved gradients are recombined independently; their autograd generation is not re-executed.",
                "Four train probes overlap in families and transitions and are not independent evaluation samples.",
                "Snapshot derivatives concern local Euclidean objective geometry; no Adam step, training-path cause, optimal weight or generalization is inferred.",
                "Float32 production norm arithmetic is checked within a fixed precision bound; exact arithmetic equality is required only for stored identity and float32 convenience casts."])
    except Exception as error:
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        torch.nn.Module._call_impl, torch.autograd.grad, torch.Tensor.backward = originals
    result["check_count"] = len(checker.checks)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    require(not arguments.output.exists(), "Never overwrite an audit or failed prefix")
    result = audit(arguments.run.resolve())
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "checks": result["check_count"], "failure": result.get("failure")}))
    if result["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__": main()
