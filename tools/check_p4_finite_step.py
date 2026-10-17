"""Independent saved-output oracle for the fixed-direction finite-step probe.

The reviewer never constructs or calls a learned model. It reconstructs
disposable parameter arithmetic and loss scalars from saved predictions using
NumPy, and reads frozen checkpoint tensors solely to bind their coordinates.
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
ALPHAS = (0., .25, .5, 1.)
RAW_LOSSES = ("one_step", "rollout", "relative_teacher", "relative_free", "total")
LOSS_NAMES = RAW_LOSSES + ("base", "aux")
REPRODUCTION_RTOL = 1e-10
REPRODUCTION_ATOL = 1e-12
METRIC_RTOL = 1e-10
METRIC_ATOL = 1e-12
TIE_MULTIPLIER = 1e-6


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


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def vector(value):
    value = np.asarray(value)
    require(value.ndim == 1 and value.size > 0 and value.dtype in (np.float32, np.float64)
            and np.isfinite(value).all(), "Finite nonempty floating vector required")
    return value.astype(np.float64)


def norm(value):
    value = vector(value)
    return float(np.sqrt(np.sum(value * value, dtype=np.float64)))


def dot(first, second):
    first, second = vector(first), vector(second)
    require(first.shape == second.shape, "Matched dot-product coordinates required")
    return float(np.sum(first * second, dtype=np.float64))


def parameter_arithmetic(original, direction):
    """Independent float32 construction, always from the same original theta."""
    original, direction = np.asarray(original), np.asarray(direction)
    require(original.dtype == direction.dtype == np.float32 and original.ndim == direction.ndim == 1
            and original.shape == direction.shape and original.size > 0
            and np.isfinite(original).all() and np.isfinite(direction).all(),
            "Matched finite float32 weights and fixed direction required")
    evaluated = np.stack([np.add(original, np.multiply(direction, np.float32(alpha), dtype=np.float32),
                                 dtype=np.float32) for alpha in ALPHAS])
    rounded = np.subtract(evaluated, original[None, :], dtype=np.float32)
    exact = evaluated.astype(np.float64) - original.astype(np.float64)[None, :]
    return {"evaluated_parameters": evaluated, "rounded_parameter_differences": rounded,
            "endpoint_differences_float64": exact}


def state_difference(current, endpoint):
    """Independent physical ten-state error with principal 2*pi yaw residual."""
    current, endpoint = np.asarray(current), np.asarray(endpoint)
    require(current.dtype == endpoint.dtype == np.float64 and current.shape == endpoint.shape
            and current.ndim >= 2 and current.shape[-1] == 10
            and np.isfinite(current).all() and np.isfinite(endpoint).all(),
            "Matched finite float64 ten-state endpoints required")
    residual = endpoint - current
    residual[..., 2] = np.remainder(residual[..., 2] + np.pi, 2 * np.pi) - np.pi
    return residual


def loss_oracle(states, mask, teacher_predictions, free_predicted_states, delta_mean, delta_scale):
    """Recompute all losses from serialized outputs, never from a model call."""
    states, mask = np.asarray(states), np.asarray(mask)
    teacher_predictions, free_predicted_states = np.asarray(teacher_predictions), np.asarray(free_predicted_states)
    delta_mean, delta_scale = np.asarray(delta_mean), np.asarray(delta_scale)
    require(states.dtype == np.float64 and states.ndim == 3 and states.shape[1:] == (6, 10)
            and len(states) > 0 and np.isfinite(states).all(), "Finite [B,6,10] float64 source states required")
    require(mask.dtype == np.bool_ and mask.shape == (len(states), 5) and mask.all(),
            "Frozen K5 probes require their full all-valid mask")
    count = int(np.count_nonzero(mask))
    require(teacher_predictions.dtype == np.float32 and teacher_predictions.shape == (count, 10)
            and np.isfinite(teacher_predictions).all(), "Finite saved float32 teacher outputs required")
    require(free_predicted_states.dtype == np.float64 and free_predicted_states.shape == states.shape
            and np.isfinite(free_predicted_states).all()
            and np.array_equal(free_predicted_states[:, 0], states[:, 0]),
            "Saved float64 rollout must preserve the exact recorded k0")
    require(delta_mean.dtype == delta_scale.dtype == np.float64 and delta_mean.shape == delta_scale.shape == (10,)
            and np.isfinite(delta_mean).all() and np.isfinite(delta_scale).all() and np.all(delta_scale > 0),
            "Original finite delta statistics and positive scales required")
    current, target = states[:, :-1][mask], states[:, 1:][mask]
    physical_target = state_difference(current, target)
    teacher_physical = teacher_predictions.astype(np.float64) * delta_scale + delta_mean
    teacher_errors = state_difference(physical_target, teacher_physical)
    free_errors = state_difference(target, free_predicted_states[:, 1:][mask])
    one = float(np.sum(np.square(teacher_errors / delta_scale), dtype=np.float64) / (10 * count))
    rollout = float(np.sum(np.square(free_errors / delta_scale), dtype=np.float64) / (10 * count))
    relative_scale = np.sqrt(np.square(delta_scale[:2]) + np.square(delta_scale[6:8]))
    relative_teacher = float(np.sum(np.square((teacher_errors[:, 6:8] - teacher_errors[:, :2]) / relative_scale), dtype=np.float64) / (2 * count))
    relative_free = float(np.sum(np.square((free_errors[:, 6:8] - free_errors[:, :2]) / relative_scale), dtype=np.float64) / (2 * count))
    base = one + .1 * rollout
    auxiliary = .2 * (relative_teacher + .1 * relative_free)
    losses = {"one_step": one, "rollout": rollout, "relative_teacher": relative_teacher,
              "relative_free": relative_free, "base": base, "aux": auxiliary, "total": base + auxiliary}
    require(all(np.isfinite(value) and value >= 0 for value in losses.values()), "Nonfinite independent saved-output loss")
    return {"losses": losses, "denominators": {"valid_transitions": count,
            "base_scalar_terms": 10 * count, "relative_scalar_terms": 2 * count},
            "relative_scale": relative_scale}


def reference_losses(value):
    """Canonicalize the old raw-component name without changing its meaning."""
    require(isinstance(value, dict), "Old fixed-probe reference losses required")
    if "relative_rollout" in value:
        return {key: value["relative_rollout" if key == "relative_free" else key] for key in RAW_LOSSES}
    return {key: value[key] for key in RAW_LOSSES}


def reproduction(actual, reference):
    reference = reference_losses(reference)
    if actual is None:
        return {"pass": False, "components": {key: False for key in RAW_LOSSES}}
    checks = {key: type(actual.get(key)) in (int, float) and np.isfinite(actual[key])
              and bool(np.isclose(actual[key], reference[key], rtol=REPRODUCTION_RTOL, atol=REPRODUCTION_ATOL))
              for key in RAW_LOSSES}
    return {"pass": bool(all(checks.values())), "components": checks}


def change_relation(change, margin):
    require(type(change) in (int, float) and np.isfinite(change)
            and type(margin) in (int, float) and np.isfinite(margin) and margin > 0,
            "Finite change and positive numeric tie margin required")
    return "INCREASE" if change > margin else "DECREASE" if change < -margin else "NUMERIC_TIE"


def classify_case(loss_rows, reference):
    """Independent preregistered branches; failed rows never disappear."""
    require(isinstance(loss_rows, (list, tuple)) and len(loss_rows) == 4, "Exactly four alpha loss records required")
    finite = [isinstance(row, dict) and all(type(row.get(key)) in (int, float)
              and np.isfinite(row[key]) and row[key] >= 0 for key in LOSS_NAMES) for row in loss_rows]
    baseline = reproduction(loss_rows[0] if finite[0] else None, reference)
    margin = TIE_MULTIPLIER * max(1., abs(loss_rows[0]["total"])) if finite[0] else None
    changes = [float(row["total"] - loss_rows[0]["total"]) if valid and finite[0] else None
               for row, valid in zip(loss_rows, finite)]
    relations = [change_relation(change, margin) if change is not None else "NUMERICAL_FAILURE" for change in changes]
    verdict = ("NUMERICAL_FAILURE" if not all(finite) else
               "REPRODUCTION_FAILURE" if not baseline["pass"] else
               "NOMINAL_STEP_OVERSHOOT_SUPPORTED" if relations[3] == "INCREASE" and "DECREASE" in relations[1:3] else
               "ALL_POSITIVE_ALPHAS_DECREASE" if all(value == "DECREASE" for value in relations[1:]) else "INCONCLUSIVE")
    return {"verdict": verdict, "reproduction": baseline, "numeric_tie_margin": margin,
            "total_loss_changes": changes, "change_relations": relations, "retained_alpha_count": 4}


def taylor_metrics(original, evaluated, rounded_difference, fixed_direction, raw_total_gradient, alpha, baseline_loss, actual_loss):
    """Keep the registered float32-difference primary distinct from sensitivity."""
    original, evaluated, rounded_difference, fixed_direction, raw_total_gradient = map(
        vector, (original, evaluated, rounded_difference, fixed_direction, raw_total_gradient))
    require(original.shape == evaluated.shape == rounded_difference.shape == fixed_direction.shape == raw_total_gradient.shape,
            "Matched Taylor/parameter coordinates required")
    require(alpha in ALPHAS and type(baseline_loss) in (float, int) and type(actual_loss) in (float, int)
            and np.isfinite(baseline_loss) and np.isfinite(actual_loss), "Registered alpha and finite measured losses required")
    endpoint_difference = evaluated - original
    nominal_change = float(alpha * dot(raw_total_gradient, fixed_direction))
    primary_change = dot(raw_total_gradient, rounded_difference)
    sensitivity_change = dot(raw_total_gradient, endpoint_difference)
    nominal_vector = alpha * fixed_direction
    return {"primary_first_order_change": primary_change, "nominal_first_order_change": nominal_change,
        "endpoint_float64_first_order_change": sensitivity_change,
        "primary_taylor_loss": float(baseline_loss + primary_change),
        "nominal_taylor_loss": float(baseline_loss + nominal_change),
        "endpoint_float64_taylor_loss": float(baseline_loss + sensitivity_change),
        "actual_total_change": float(actual_loss - baseline_loss),
        "actual_minus_primary_taylor": float(actual_loss - baseline_loss - primary_change),
        "actual_minus_nominal_taylor": float(actual_loss - baseline_loss - nominal_change),
        "primary_vs_endpoint_float64_change_difference": primary_change - sensitivity_change,
        "rounded_difference_norm": norm(rounded_difference), "endpoint_float64_difference_norm": norm(endpoint_difference),
        "nominal_difference_norm": norm(nominal_vector),
        "rounded_vs_nominal_difference_l2": norm(rounded_difference - nominal_vector),
        "rounded_vs_endpoint_float64_difference_l2": norm(rounded_difference - endpoint_difference)}


def parameter_tree_hash(flat, order):
    """Historical typed tensor-tree identity, computed directly from NumPy bytes."""
    flat = np.asarray(flat)
    require(flat.dtype == np.float32 and flat.ndim == 1, "Float32 parameter coordinate map required")
    digest = hashlib.sha256(b"dict")
    by_name = {item["name"]: item for item in order}
    require(len(by_name) == len(order), "Unique parameter names required")
    for name in sorted(by_name):
        item = by_name[name]
        digest.update(b"str" + json.dumps(name, sort_keys=True, allow_nan=False).encode())
        digest.update(b"tensor" + b"torch.float32" + json.dumps(item["shape"]).encode())
        digest.update(flat[item["start"]:item["end"]].reshape(item["shape"]).copy().tobytes())
    return digest.hexdigest()


def mapping_hash(flat, order):
    identities = []
    for item in order:
        value = np.asarray(flat)[item["start"]:item["end"]].reshape(item["shape"])
        descriptor = json.dumps(["torch.float32", item["shape"]], separators=(",", ":")).encode()
        identities.append([item["name"], hashlib.sha256(descriptor + value.copy().tobytes()).hexdigest()])
    return hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest()


def tensor_tree_hash(value):
    import torch
    hasher = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            hasher.update(b"tensor" + str(item.dtype).encode() + json.dumps(list(item.shape)).encode())
            hasher.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            hasher.update(b"dict")
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))): visit(key); visit(item[key])
        elif isinstance(item, (list, tuple)):
            hasher.update(type(item).__name__.encode())
            for child in item: visit(child)
        else:
            hasher.update(type(item).__name__.encode() + json.dumps(item, sort_keys=True, allow_nan=False).encode())
    visit(value)
    return hasher.hexdigest()


def validate_order(order, size=19338):
    expected_names = [f"network.{layer}.{kind}" for layer in (0, 2, 4) for kind in ("weight", "bias")]
    require(isinstance(order, list) and [item["name"] for item in order] == expected_names,
            "Frozen six-parameter coordinate order required")
    cursor = 0
    for item in order:
        require(set(item) == {"name", "shape", "numel", "start", "end"} and item["start"] == cursor
                and item["numel"] == int(np.prod(item["shape"])) == item["end"] - cursor > 0,
                "Parameter coordinates must cover every flattened tensor exactly")
        cursor = item["end"]
    require(cursor == size, "Parameter coordinates do not cover the fixed model")


def production_losses(losses):
    return {"one_step": losses["one_step"], "rollout": losses["rollout"],
        "relative_teacher": losses["relative_teacher"], "relative_rollout": losses["relative_free"],
        "total": losses["total"], "base": losses["base"], "weighted_aux": losses["aux"], "common_total": losses["total"]}


def canonical_losses(losses):
    return {"one_step": losses["one_step"], "rollout": losses["rollout"],
        "relative_teacher": losses["relative_teacher"], "relative_free": losses["relative_rollout"],
        "total": losses["common_total"], "base": losses["base"], "aux": losses["weighted_aux"]}


class Audit:
    def __init__(self):
        self.checks, self.sources, self.observed = [], {}, []
        self.forward_calls, self.forward_rows, self.successful_evaluations = 0, 0, 0

    def check(self, name, passed, **details):
        self.checks.append({"name": name, "pass": bool(passed), **details})
        require(passed, name)

    def number(self, name, actual, expected):
        self.check(name, actual is None if expected is None else type(actual) in (int, float)
                   and np.isfinite(actual) and bool(np.isclose(actual, expected, rtol=METRIC_RTOL, atol=METRIC_ATOL)))

    def tree(self, name, actual, expected):
        if isinstance(expected, dict):
            self.check(name + ":fields", isinstance(actual, dict) and set(expected).issubset(actual))
            for key, value in expected.items(): self.tree(name + ":" + key, actual[key], value)
        elif isinstance(expected, list):
            self.check(name + ":length", isinstance(actual, list) and len(actual) == len(expected))
            for index, value in enumerate(expected): self.tree(name + ":" + str(index), actual[index], value)
        elif expected is None or type(expected) in (int, float): self.number(name, actual, expected)
        else: self.check(name, type(actual) is type(expected) and actual == expected)

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        require(path.is_relative_to(ROOT), "Input path outside project")
        relative, digest = path.relative_to(ROOT).as_posix(), sha(path)
        self.check("immutable:" + relative, expected is None or digest == expected)
        if relative in self.sources: self.check("repeat_immutable:" + relative, self.sources[relative] == digest)
        self.sources[relative] = digest
        return path

    def case(self, run, case, spec, weights, optimizer_hash, probes, statistics):
        label = case["snapshot_id"] + ":" + case["probe_name"]
        folder = run / "cases" / case["snapshot_id"] / case["probe_name"]
        record = read(folder / "summary.json")
        output = arrays(folder / "parameter_displacements.npz")
        predictions = arrays(folder / "predictions.npz")
        old_path = self.bind(ROOT / case["original_gradient_summary"], case["original_gradient_summary_sha256"])
        old = read(old_path)
        order = old["parameter_order"]
        validate_order(order)
        original = np.concatenate([weights[item["name"]].detach().numpy().reshape(-1) for item in order])
        self.check(label + ":original_weight_tree", parameter_tree_hash(original, order) == spec["model_tree_hash"])
        source_direction = self.bind(ROOT / case["source_direction_npz"])
        direction_record = read(self.bind(source_direction.with_name("summary.json"), case["source_summary_sha256"]))
        self.check(label + ":same_TOTAL_KEEP_direction_coordinates", case["source_direction_index"] == 3
                   and direction_record["parameter_order"] == order and direction_record["directions"][3]["objective"] == "total"
                   and direction_record["directions"][3]["mode"] == "KEEP")
        fixed_direction = arrays(source_direction)["directions"][3]
        raw = arrays(self.bind(old_path.with_name("gradients.npz")))["gradients"][4]
        self.check(label + ":frozen_gradient_and_direction", fixed_direction.shape == raw.shape == (19338,)
                   and fixed_direction.dtype == raw.dtype == np.float32 and np.isfinite(fixed_direction).all() and np.isfinite(raw).all())
        expected_parameters = parameter_arithmetic(original, fixed_direction)
        self.check(label + ":parameter_array_keys", set(output) == {"alphas", "evaluated_parameters", "rounded_parameter_differences"})
        self.check(label + ":prediction_array_keys", set(predictions) == {"teacher_predictions", "free_predicted_states"})
        self.check(label + ":exact_alpha_coordinates", output["alphas"].dtype == np.float64
                   and np.array_equal(output["alphas"], np.asarray(ALPHAS, np.float64)))
        for key in ("evaluated_parameters", "rounded_parameter_differences"):
            self.check(label + ":exact_independent_parameter_arithmetic:" + key, output[key].dtype == np.float32
                       and output[key].shape == (4, 19338) and np.array_equal(output[key], expected_parameters[key], equal_nan=True))
        name = case["probe_name"]
        states, actions, mask = (probes[name + "_" + key] for key in ("states", "actions", "mask"))
        batch = len(states)
        self.check(label + ":fixed_batch_contract", states.dtype == actions.dtype == np.float64
                   and states.shape == (batch, 6, 10) and actions.shape == (batch, 5, 2)
                   and mask.dtype == np.bool_ and mask.shape == (batch, 5) and mask.all())
        self.check(label + ":prediction_array_shapes", predictions["teacher_predictions"].dtype == np.float32
                   and predictions["teacher_predictions"].shape == (4, 5 * batch, 10)
                   and predictions["free_predicted_states"].dtype == np.float64
                   and predictions["free_predicted_states"].shape == (4, batch, 6, 10))
        self.tree(label + ":case_identity", record, {"snapshot_id": case["snapshot_id"], "probe_name": name,
            "source": case, "parameter_order": order, "source_weights_tree_hash": spec["model_tree_hash"],
            "source_optimizer_tree_hash": optimizer_hash, "parameter_arrays_sha256": sha(folder / "parameter_displacements.npz"),
            "prediction_arrays_sha256": sha(folder / "predictions.npz")})
        evaluation = record["evaluation"]
        denominators = {"batch_windows": batch, "valid_transitions": 5 * batch,
                        "base_scalar_terms": 50 * batch, "relative_scalar_terms": 10 * batch}
        self.tree(label + ":evaluation_contract", evaluation, {"schema": "p4-fixed-direction-finite-step-evaluation-v1",
            "alphas": list(ALPHAS), "parameter_order": order, "parameter_count": 19338,
            "raw_loss_order": ["one_step", "rollout", "relative_teacher", "relative_rollout", "total"],
            "denominators": denominators, "retained_alpha_count": 4, "case_denominator": 1,
            "source_state_hash_before": mapping_hash(original, order), "source_state_hash_after": mapping_hash(original, order)})
        preservation_names = {"source_state_unchanged", "source_parameter_identities_unchanged", "source_buffer_identities_unchanged",
            "source_gradient_buffers_unchanged", "source_modes_unchanged", "source_hooks_unchanged", "global_rng_unchanged",
            "batch_inputs_unchanged", "normalizer_unchanged", "direction_and_gradient_inputs_unchanged"}
        self.check(label + ":source_preservation_checks", set(evaluation["checks"]) == preservation_names
                   and all(value is True for value in evaluation["checks"].values()))
        rows = evaluation["evaluations"]
        self.check(label + ":all_four_rows_retained", len(rows) == 4)
        independent_losses, expected_trace, case_calls, case_rows, successful = [], [], 0, 0, 0
        for index, alpha in enumerate(ALPHAS):
            row = rows[index]
            row_label = label + ":alpha:" + str(alpha)
            self.tree(row_label + ":identity_and_denominators", row, {"index": index, "alpha": alpha, "denominators": denominators,
                "evaluated_parameters_hash": hashlib.sha256(output["evaluated_parameters"][index].tobytes()).hexdigest(),
                "rounded_difference_hash": hashlib.sha256(output["rounded_parameter_differences"][index].tobytes()).hexdigest()})
            self.check(row_label + ":recorded_status", row["status"] in ("PASS", "NUMERICAL_FAILURE"))
            calls = row["forward_calls"]
            self.check(row_label + ":forward_prefix_budget", type(calls) is int and 0 <= calls <= 6
                       and row["forward_rows"] == ([5 * batch] + [batch] * 5)[:calls]
                       and row["forward_total_rows"] == sum(row["forward_rows"]))
            if row["status"] == "PASS":
                self.check(row_label + ":successful_root_call_contract", calls == 6 and row["error"] is None)
                independent = loss_oracle(states, mask, predictions["teacher_predictions"][index],
                    predictions["free_predicted_states"][index], statistics["delta_mean"], statistics["delta_scale"])
                expected = production_losses(independent["losses"])
                self.check(row_label + ":raw_and_derived_loss_fields", set(row["loss_values"]) == set(expected))
                self.tree(row_label + ":independent_saved_prediction_losses", row["loss_values"], expected)
                independent_losses.append(independent["losses"])
                successful += 1
            else:
                self.check(row_label + ":failure_never_drops_or_replaces_loss", set(row["loss_values"]) ==
                    {"one_step", "rollout", "relative_teacher", "relative_rollout", "total", "base", "weighted_aux", "common_total"}
                    and all(value is None for value in row["loss_values"].values()) and isinstance(row["error"], dict)
                    and set(row["error"]) == {"type", "message"} and all(isinstance(value, str) and value for value in row["error"].values()))
                independent_losses.append(None)
            evaluated, rounded = output["evaluated_parameters"][index], output["rounded_parameter_differences"][index]
            finite_parameters = bool(np.isfinite(evaluated).all() and np.isfinite(rounded).all())
            endpoint = evaluated.astype(np.float64) - original.astype(np.float64)
            nominal = alpha * fixed_direction.astype(np.float64)
            displacement = {"nominal_scaled_norm": norm(nominal),
                "rounded_difference_norm": norm(rounded) if finite_parameters else None,
                "exact_endpoint_difference_norm": norm(endpoint) if finite_parameters else None,
                "rounded_minus_nominal_l2": norm(rounded.astype(np.float64) - nominal) if finite_parameters else None,
                "exact_minus_nominal_l2": norm(endpoint - nominal) if finite_parameters else None}
            taylor = {"nominal_directional_change": dot(raw, nominal),
                "rounded_directional_change": dot(raw, rounded) if finite_parameters else None,
                "exact_endpoint_directional_change": dot(raw, endpoint) if finite_parameters else None}
            self.tree(row_label + ":displacement_norms", row["displacement"], displacement)
            self.tree(row_label + ":three_distinct_directional_predictions", row["taylor"], taylor)
            evaluated_hash = parameter_tree_hash(evaluated, order)
            expected_trace += [{"rows": count, "parameter_tree_hash": evaluated_hash} for count in row["forward_rows"]]
            case_calls += calls; case_rows += row["forward_total_rows"]
        self.check(label + ":actual_source_forwards_used_evaluated_parameter_maps", record["source_forward_trace"] == expected_trace)
        verified_recorded_losses = [canonical_losses(row["loss_values"]) if row["status"] == "PASS" else None for row in rows]
        classification = classify_case(verified_recorded_losses, old["loss_values"])
        baseline_status = ("NUMERICAL_FAILURE" if independent_losses[0] is None else
                           "PASS" if classification["reproduction"]["pass"] else "REPRODUCTION_FAILURE")
        per_loss = {}
        for key in ("one_step", "rollout", "relative_teacher", "relative_rollout", "total"):
            expected = old["loss_values"][key]
            observed = rows[0]["loss_values"][key]
            error = None if observed is None else abs(observed - expected)
            allowed = REPRODUCTION_ATOL + REPRODUCTION_RTOL * abs(expected)
            per_loss[key] = {"actual": observed, "reference": expected, "absolute_error": error,
                             "allowed_error": allowed, "pass": bool(error is not None and error <= allowed)}
        self.tree(label + ":strict_alpha0_reproduction", evaluation["alpha0_reproduction"],
            {"status": baseline_status, "rtol": REPRODUCTION_RTOL, "atol": REPRODUCTION_ATOL, "per_loss": per_loss})
        self.number(label + ":frozen_numeric_tie_band", evaluation["neutral_band"], classification["numeric_tie_margin"])
        for index, row in enumerate(rows):
            available = independent_losses[index] is not None and independent_losses[0] is not None
            differences = {key: row["loss_values"][key] - rows[0]["loss_values"][key] if available else None
                           for key in ("base", "weighted_aux", "common_total")}
            self.tree(label + ":alpha:" + str(ALPHAS[index]) + ":actual_loss_changes", row["loss_differences"], differences)
            taylor = {key: None for key in ("primary_prediction", "endpoint_sensitivity_prediction", "nominal_prediction",
                                           "residual_actual_minus_primary", "residual_actual_minus_endpoint")}
            if available:
                metric = taylor_metrics(original, output["evaluated_parameters"][index], output["rounded_parameter_differences"][index],
                    fixed_direction, raw, ALPHAS[index], verified_recorded_losses[0]["total"], verified_recorded_losses[index]["total"])
                taylor = {"primary_prediction": metric["primary_taylor_loss"],
                    "endpoint_sensitivity_prediction": metric["endpoint_float64_taylor_loss"], "nominal_prediction": metric["nominal_taylor_loss"],
                    "residual_actual_minus_primary": metric["actual_minus_primary_taylor"],
                    "residual_actual_minus_endpoint": verified_recorded_losses[index]["total"] - metric["endpoint_float64_taylor_loss"]}
            self.tree(label + ":alpha:" + str(ALPHAS[index]) + ":Taylor_residuals", row["taylor"], taylor)
            relation = classification["change_relations"][index] if available and baseline_status == "PASS" else "NOT_INTERPRETED"
            self.check(label + ":alpha:" + str(ALPHAS[index]) + ":tie_aware_relation", row["total_change_relation"] == relation)
        decisions = {"NOMINAL_STEP_OVERSHOOT_SUPPORTED": "NOMINAL_INCREASE_SMALLER_DECREASE",
                     "ALL_POSITIVE_ALPHAS_DECREASE": "NO_ALPHA1_OVERSHOOT_OBSERVED_ON_GRID"}
        decision = decisions.get(classification["verdict"], classification["verdict"])
        status = "NUMERICAL_FAILURE" if successful < 4 else baseline_status
        self.check(label + ":independent_failure_priority_and_decision", evaluation["decision"] == decision and evaluation["status"] == status)
        self.tree(label + ":evaluation_counters", evaluation["counters"], {"evaluations": 4, "successful_evaluations": successful,
            "root_model_forward_calls": case_calls, "root_model_forward_rows": case_rows, "new_autograd_traversals": 0,
            "new_optimizer_steps": 0, "new_training_updates": 0, "new_physics_steps": 0})
        self.forward_calls += case_calls; self.forward_rows += case_rows; self.successful_evaluations += successful
        self.observed.append({"snapshot_id": case["snapshot_id"], "probe_name": name,
            "independent_decision": decision, "case_status": status, "independent_reproduction_status": baseline_status,
            "retained_evaluation_count": 4, "successful_evaluation_count": successful,
            "numeric_tie_margin": classification["numeric_tie_margin"], "total_changes": classification["total_loss_changes"],
            "change_relations": classification["change_relations"], "losses_from_saved_predictions": independent_losses,
            "branch_operands": "Recorded scalar losses independently verified from saved teacher/free predictions under the fixed float64 metric tolerance",
            "limitations": "Primary Taylor uses registered rounded float32 subtraction; float64 endpoint difference remains a separate sensitivity. No model forward is replayed by the auditor."})
        return {"snapshot_id": case["snapshot_id"], "probe_name": name,
                "summary_path": (folder / "summary.json").relative_to(run).as_posix(), "evaluation": evaluation}

    def run(self, run):
        import platform
        import torch
        protocol, manifest, summary = (read(run / filename) for filename in ("registration.json", "manifest.json", "summary.json"))
        self.check("completed_registered_finite_step_run", run == ROOT / "runs/p4-finite-step-20261015-01"
                   and protocol["run_id"] == "p4-finite-step-20261015-01" and protocol["run_directory"] == run.relative_to(ROOT).as_posix()
                   and manifest["schema"] == "p4-finite-step-manifest-v1" and summary["schema"] == "p4-finite-step-summary-v1"
                   and summary["execution_status"] == "PASS" and summary["analysis_status"] == "DESCRIPTIVE_COMPLETE")
        self.check("execution_registration_manifest_summary_binding", sha(run / "registration.json") ==
                   sha(ROOT / "configs/p4_finite_step.v1.json") == manifest["protocol_sha256"] == summary["protocol_sha256"]
                   and sha(run / "manifest.json") == summary["manifest_sha256"])
        self.bind(ROOT / "configs/p4_finite_step.v1.json", manifest["protocol_sha256"])
        for relative, expected in protocol["bindings_sha256"].items(): self.bind(ROOT / relative, expected)
        self.check("auditor_and_synthetic_tests_bound_before_execution", all(relative in protocol["bindings_sha256"]
                   for relative in ("tools/check_p4_finite_step.py", "tests/test_p4_finite_step_audit.py",
                       "src/push_align/finite_step_probe.py", "scripts/p4_finite_step_probe.py")))
        self.check("original_design_path", protocol["design_path"] == "configs/p4_finite_step.design.v1.json")
        design = read(self.bind(ROOT / protocol["design_path"], protocol["design_sha256"]))
        self.check("same_original_28_cases_and_four_alphas", protocol["cases"] == design["cases"] and len(protocol["cases"]) == 28
                   and protocol["alphas"] == manifest["alphas"] == design["alphas"] == list(ALPHAS)
                   and design["case_count"] == 28 and design["evaluation_count"] == 112)
        self.check("fixed_original_reproduction_and_tie_policy", design["alpha0_reproduction"]["rtol"] == REPRODUCTION_RTOL
                   and design["alpha0_reproduction"]["atol"] == REPRODUCTION_ATOL
                   and design["loss_change_neutral_band"]["formula"] == "1e-6*max(1,abs(alpha0_total_loss))")
        adam_run = ROOT / "runs/p4-adam-probe-20261015-01"
        original_bindings = read(adam_run / "registration.json")["bindings_sha256"]
        original_index = read(adam_run / "artifact_hashes.json")["sha256"]
        self.check("original_469_bindings_and_61_outputs_inventory", len(original_bindings) == 469 and len(original_index) == 61)
        for relative, expected in original_bindings.items(): self.bind(ROOT / relative, expected)
        for relative, expected in original_index.items(): self.bind(adam_run / relative, expected)
        self.check("original_adam_direction_index_bound", sha(adam_run / "artifact_hashes.json") == design["source_run_index_sha256"]
                   == manifest["adam_run_index_sha256"] and sha(ROOT / "reports/p4_adam_probe_independent_review.json") == design["source_audit_sha256"])
        gradient_run = ROOT / "runs/p4-gradient-probe-20261015-01"
        gradient_manifest = read(self.bind(gradient_run / "manifest.json"))
        specs = protocol["snapshots"]
        expected_ids = ["common_initial"] + [f"{branch}_u{update:04d}" for branch in ("C0", "R1") for update in (100, 500, 1000)]
        self.check("same_seven_checkpoints", specs == manifest["model_snapshots"] == gradient_manifest["model_snapshots"]
                   and [spec["snapshot_id"] for spec in specs] == expected_ids)
        probe_names = ["historical_update1", "D11_zero5", "D11_transition", "D11_noncontact"]
        pairs = [(case["snapshot_id"], case["probe_name"]) for case in protocol["cases"]]
        self.check("unfiltered_fixed_case_order", pairs == [(snapshot, name) for snapshot in expected_ids for name in probe_names])
        expected_outputs = {"registration.json", "manifest.json", "aggregate.json", "summary.json"}
        expected_outputs |= {f"cases/{snapshot}/{name}/{filename}" for snapshot, name in pairs
                            for filename in ("summary.json", "parameter_displacements.npz", "predictions.npz")}
        index = read(run / "artifact_hashes.json")["sha256"]
        self.check("exact_88_indexed_outputs", set(index) == expected_outputs and len(index) == 88)
        for relative, expected in index.items():
            path = (run / relative).resolve()
            self.check("output_scope:" + relative, path.is_relative_to(run))
            self.bind(path, expected)
        runtime = protocol["runtime"]
        self.check("registered_runtime_versions", manifest["runtime"] == runtime and str(torch.__version__) == runtime["torch_version"]
                   and np.__version__ == runtime["numpy_version"] and platform.python_version() == runtime["python_version"])
        for path, expected in runtime["source_sha256"].items(): self.check("runtime_source:" + path, sha(Path(path)) == expected)
        self.check("same_frozen_data_indices", manifest["gradient_run_index_sha256"] == sha(gradient_run / "artifact_hashes.json")
                   and manifest["probe_batches_sha256"] == sha(gradient_run / "probe_batches.npz"))
        probes = arrays(self.bind(gradient_run / "probe_batches.npz", manifest["probe_batches_sha256"]))
        normalizer = ROOT / "runs/p3-02-20261009-01/learning_data/normalizer"
        normalizer_meta = read(self.bind(normalizer / "normalizer.json"))
        statistics = arrays(self.bind(normalizer / "statistics.npz", normalizer_meta["npz_sha256"]))
        self.check("original_normalizer_no_refit", normalizer_meta["normalizer"]["normalizer_hash"] == manifest["normalizer_hash"]
                   == gradient_manifest["normalizer_hash"])
        parent = read(self.bind(ROOT / "runs/p4-relative-loss-20261014-01/manifest.json"))
        self.check("same_physical_config", manifest["physical_config"] == parent["physical_config"] == gradient_manifest["physical_config"])
        records, tensor_checks = [], []
        for spec in specs:
            checkpoint = ROOT / spec["checkpoint_path"]
            metadata = read(self.bind(checkpoint / "checkpoint.json", spec["checkpoint_json_sha256"]))
            self.check(spec["snapshot_id"] + ":checkpoint_metadata", metadata["checkpoint_hash"] == spec["checkpoint_hash"]
                and metadata["checkpoint_hash"] == hashlib.sha256(canonical({key: value for key, value in metadata.items() if key != "checkpoint_hash"})).hexdigest())
            for filename in ("weights.pt", "training_state.pt"):
                self.bind(checkpoint / filename, spec["files_sha256"][filename])
                self.check(spec["snapshot_id"] + ":checkpoint_file:" + filename, metadata["files"][filename] == spec["files_sha256"][filename])
            weights = torch.load(checkpoint / "weights.pt", map_location="cpu", weights_only=True)
            optimizer = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)["optimizer"]
            self.check(spec["snapshot_id"] + ":saved_weight_precision", all(torch.is_tensor(value) and value.dtype == torch.float32
                       and value.device.type == "cpu" and not value.requires_grad and bool(torch.isfinite(value).all()) for value in weights.values()))
            model_hash, optimizer_hash = tensor_tree_hash(weights), tensor_tree_hash(optimizer)
            self.check(spec["snapshot_id"] + ":exact_source_model_tree", model_hash == spec["model_tree_hash"])
            self.check(spec["snapshot_id"] + ":saved_Adam_step_preserved", all(float(value["step"]) == 1300 + spec["additional_updates"]
                       for value in optimizer["state"].values()))
            for case in [case for case in protocol["cases"] if case["snapshot_id"] == spec["snapshot_id"]]:
                records.append(self.case(run, case, spec, weights, optimizer_hash, probes, statistics))
            tensor_checks.append({"snapshot_id": spec["snapshot_id"], "weights_unchanged": True, "optimizer_unchanged": True,
                                  "weight_tree_hash": model_hash, "optimizer_tree_hash": optimizer_hash})
        aggregate = read(run / "aggregate.json")
        self.check("all28_aggregate_exact_aliases", aggregate["schema"] == "p4-finite-step-aggregate-v1"
                   and aggregate["row_count"] == len(records) == 28 and aggregate["rows"] == records)
        self.tree("all7_source_tensor_preservation_records", summary["source_tensor_checks"], tensor_checks)
        self.tree("complete_28case_112attempt_denominators", summary["counters"], {"snapshots": 7, "cases": 28,
            "attempted_evaluations": 112, "completed_evaluations": self.successful_evaluations,
            "forward_calls": self.forward_calls, "forward_rows": self.forward_rows})
        self.check("bounded_forward_execution", self.forward_calls <= 672 and self.forward_rows <= 31360)
        if self.successful_evaluations == 112:
            self.check("all112_successful_exact_registered_forward_execution", self.forward_calls == 672 and self.forward_rows == 31360)
        forbidden_names = {"physics." + name for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse")}
        forbidden_names |= {"Adam_instances", "Optimizer_instances", "Adam.step", "Optimizer.step", "autograd.grad", "Tensor.backward", "torch.save"}
        self.check("no_forbidden_training_gradient_optimizer_checkpoint_or_physics", set(summary["forbidden_execution_counts"]) == forbidden_names
                   and all(type(value) is int and value == 0 for value in summary["forbidden_execution_counts"].values()))
        self.tree("finite_loss_evaluation_scope", summary, {"global_rng_unchanged": True, "new_derivative_traversals": 0,
            "new_optimizer_updates": 0, "new_physics_steps": 0, "new_checkpoint_writes": 0, "new_data": False,
            "normalizer_refit": False, "final_test_used": False, "validation_or_development_probes_used": False,
            "historical_trajectory_replayed": False, "model_selection": "NONE", "alignment_improvement": "NOT_EVALUATED",
            "personal_understanding": "NOT_ASSESSED", "P5_01": "HOLD"})
        for relative, expected in list(self.sources.items()): self.check("source_unchanged_after_audit:" + relative, sha(ROOT / relative) == expected)


def audit(run):
    import mujoco
    import random
    import torch
    run = Path(run).resolve()
    checker = Audit()
    entries = [(mujoco, name, "physics." + name) for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse")]
    entries += [(torch.nn.Module, "__init__", "model_instances"), (torch.nn.Module, "_call_impl", "model_forwards"),
        (torch.optim.Adam, "__init__", "Adam_instances"), (torch.optim.Optimizer, "__init__", "Optimizer_instances"),
        (torch.optim.Adam, "step", "Adam.step"), (torch.optim.Optimizer, "step", "Optimizer.step"),
        (torch.autograd, "grad", "autograd.grad"), (torch.Tensor, "backward", "Tensor.backward"), (torch, "save", "torch.save")]
    counts, originals = {label: 0 for _, _, label in entries}, []
    torch_rng, numpy_rng, python_rng = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    def blocked(label):
        def call(*args, **kwargs):
            counts[label] += 1
            raise RuntimeError("Saved-output finite-step audit forbids " + label)
        return call
    for owner, name, label in entries:
        originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, blocked(label))
    result = {"schema": "p4-finite-step-independent-review-v1", "status": "FAIL", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(run.relative_to(ROOT)), "review_source_sha256": sha(__file__), "checks": checker.checks,
        "numeric_policy": {"metric_rtol": METRIC_RTOL, "metric_atol": METRIC_ATOL,
            "alpha0_reproduction_rtol": REPRODUCTION_RTOL, "alpha0_reproduction_atol": REPRODUCTION_ATOL,
            "numeric_tie_margin": "1e-6*max(1,abs(alpha0_total_loss))", "parameter_arithmetic": "Exact float32 multiply, add and subtract on independent original copies",
            "primary_taylor": "Saved raw total gradient float64 dot promoted float32(theta_alpha-theta)",
            "sensitivity_taylor": "Separate float64 difference of the two represented float32 endpoints",
            "loss_oracle": "NumPy float64 from saved normalized teacher predictions, physical free states and immutable original delta statistics"}}
    try:
        checker.run(run)
        checker.check("auditor_zero_model_gradient_update_or_physics", not any(counts.values()))
        current_numpy = np.random.get_state()
        checker.check("auditor_rng_unchanged", torch.equal(torch_rng, torch.get_rng_state())
            and numpy_rng[0] == current_numpy[0] and np.array_equal(numpy_rng[1], current_numpy[1])
            and numpy_rng[2:] == current_numpy[2:] and python_rng == random.getstate())
        result.update(status="PASS", observed={"cases": checker.observed,
            "decision_counts": dict(Counter(row["independent_decision"] for row in checker.observed)),
            "retained_case_denominator": 28, "retained_evaluation_denominator": 112,
            "successful_evaluation_count": checker.successful_evaluations}, source_bindings_sha256=checker.sources,
            limitations=["The auditor reads saved predictions and checkpoint tensors and never executes a model forward, derivative, optimizer, training update or physics call.",
                "Saved teacher/free outputs permit an independent loss-arithmetic check; the registered forward trace binds every source call to its evaluated parameter map without replaying the learned function.",
                "The registered float32 subtraction defines the primary Taylor displacement. The float64 endpoint difference remains sensitivity only and does not silently replace the primary.",
                "Numeric ties are a fixed computational comparison margin, not statistical confidence or a learning-performance threshold.",
                "All 28 overlapping train cases and 112 planned endpoints remain, including numerical and reproduction failures. Findings do not identify the whole historical training cause, select a model or establish physical/generalization performance."])
    except Exception as error:
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        for owner, name, original in reversed(originals): setattr(owner, name, original)
    result["forbidden_execution_counts"] = counts
    result["check_count"] = len(checker.checks)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    require(not arguments.output.exists(), "Never overwrite a review or failed prefix")
    result = audit(arguments.run)
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "checks": result["check_count"], "failure": result.get("failure")}))
    if result["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__": main()

