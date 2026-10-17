"""Independent disposable parameter maps along one frozen TOTAL/KEEP addend.

Only the existing relative_sequence_loss evaluates the four registered alpha
values. No gradients or optimizer are computed. The source model's parameters,
buffers, modes, hooks, gradient buffers and CPU RNG remain unchanged.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
import random

import numpy as np
import torch
from torch import nn

from .normalization import Normalizer
from .relative_loss_training import relative_sequence_loss


ALPHAS = (0., .25, .5, 1.)
RAW_LOSS_ORDER = ("one_step", "rollout", "relative_teacher", "relative_rollout", "total")
LOSS_ORDER = RAW_LOSS_ORDER + ("base", "weighted_aux", "common_total")
REPRODUCTION_RTOL = 1e-10
REPRODUCTION_ATOL = 1e-12
TIE_MULTIPLIER = 1e-6


class FiniteStepProbeError(ValueError):
    """The fixed-case contract or preservation of an original input failed."""


def _require(condition, message):
    if not condition:
        raise FiniteStepProbeError(message)


def _tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    descriptor = json.dumps([str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
    return hashlib.sha256(descriptor + value.numpy().tobytes()).hexdigest()


def _mapping_hash(mapping):
    return hashlib.sha256(json.dumps([[name, _tensor_hash(tensor)] for name, tensor in mapping.items()],
                                    separators=(",", ":")).encode()).hexdigest()


def _normalizer_hash(normalizer):
    values = {"metadata": normalizer.metadata,
              "arrays": {name: [str(value.dtype), list(value.shape), value.tobytes().hex()]
                         for name, value in sorted(normalizer.arrays.items())}}
    return hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _gradient_identity(tensors):
    return [(id(value.grad), None if value.grad is None else _tensor_hash(value.grad)) for value in tensors]


def _numpy_rng_identity():
    state = np.random.get_state()
    return (state[0], state[1].tobytes(), *state[2:])


def _source_snapshot(model, normalizer, states, actions, mask, delta, gradient):
    parameters, buffers = tuple(model.named_parameters()), tuple(model.named_buffers())
    modules = tuple(model.modules())
    state = model.state_dict()
    _require(all(isinstance(value, torch.Tensor) for value in state.values()),
             "Only tensor source state is supported")
    return {"state_hash": _mapping_hash(state),
            "parameter_identity": [(name, id(value)) for name, value in parameters],
            "buffer_identity": [(name, id(value)) for name, value in buffers],
            "gradients": _gradient_identity([value for _, value in parameters]),
            "modes": [module.training for module in modules],
            "hooks": [(id(module), tuple((key, id(value)) for key, value in module._forward_pre_hooks.items()),
                       tuple((key, id(value)) for key, value in module._forward_hooks.items())) for module in modules],
            "torch_rng": _tensor_hash(torch.get_rng_state()),
            "numpy_rng": _numpy_rng_identity(), "python_rng": random.getstate(),
            "batch_hashes": [_tensor_hash(value) for value in (states, actions, mask)],
            "batch_gradients": _gradient_identity((states, actions, mask)),
            "normalizer_hash": _normalizer_hash(normalizer),
            "delta_bytes": delta.tobytes(), "gradient_bytes": gradient.tobytes()}


def _preservation_checks(before, after):
    return {"source_state_unchanged": before["state_hash"] == after["state_hash"],
            "source_parameter_identities_unchanged": before["parameter_identity"] == after["parameter_identity"],
            "source_buffer_identities_unchanged": before["buffer_identity"] == after["buffer_identity"],
            "source_gradient_buffers_unchanged": before["gradients"] == after["gradients"],
            "source_modes_unchanged": before["modes"] == after["modes"],
            "source_hooks_unchanged": before["hooks"] == after["hooks"],
            "global_rng_unchanged": all(before[key] == after[key] for key in ("torch_rng", "numpy_rng", "python_rng")),
            "batch_inputs_unchanged": before["batch_hashes"] == after["batch_hashes"]
                                      and before["batch_gradients"] == after["batch_gradients"],
            "normalizer_unchanged": before["normalizer_hash"] == after["normalizer_hash"],
            "direction_and_gradient_inputs_unchanged": before["delta_bytes"] == after["delta_bytes"]
                                                       and before["gradient_bytes"] == after["gradient_bytes"]}


class _FunctionalProxy(nn.Module):
    """The proxy owns root hooks, but never registers the source as a child.

    Existing caller hooks on the source still run once inside functional_call.
    The relative loss's teacher-capture hook belongs to this proxy and sees
    exactly one teacher output, followed by the five free-rollout outputs.
    """
    def __init__(self, source, parameters, buffers):
        super().__init__()
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_parameters_map", parameters)
        object.__setattr__(self, "_buffers_map", buffers)
        self.training = False

    def forward(self, features):
        _require(not torch.is_grad_enabled(), "Finite-step model calls require no_grad")
        return torch.func.functional_call(self._source, (self._parameters_map, self._buffers_map),
                                          (features,), strict=True)


def _validate(model, normalizer, states, actions, mask, delta, gradient, references, order):
    _require(isinstance(model, nn.Module) and all(not module.training for module in model.modules()),
             "The source model and every submodule must already be in eval mode")
    _require(not torch.is_autocast_enabled("cpu"), "CPU autocast is forbidden")
    _require(isinstance(normalizer, Normalizer), "A schema-bound original Normalizer is required")
    normalizer.validate()
    named = tuple(model.named_parameters())
    _require(bool(named), "The source model must have parameters")
    _require(len(tuple(model.named_parameters(remove_duplicate=False))) == len(named),
             "Tied parameter aliases are not supported by the fixed parameter order")
    expected, cursor = [], 0
    for name, value in named:
        _require(value.device.type == "cpu" and value.dtype == torch.float32
                 and value.layout == torch.strided and bool(torch.isfinite(value).all()),
                 "Finite dense CPU float32 source parameters required")
        expected.append({"name": name, "shape": list(value.shape), "numel": value.numel(),
                         "start": cursor, "end": cursor + value.numel()})
        cursor += value.numel()
    for value in model.buffers():
        _require(value.device.type == "cpu" and value.layout == torch.strided
                 and (not value.is_floating_point() or value.dtype == torch.float32)
                 and bool(torch.isfinite(value).all()), "Finite CPU source buffers required")
    _require(order is None or order == expected, "Saved parameter order differs from the source model")
    for value, name in ((delta, "TOTAL/KEEP direction"), (gradient, "raw total gradient")):
        _require(isinstance(value, np.ndarray) and value.dtype == np.float32
                 and value.shape == (cursor,) and np.isfinite(value).all(),
                 "Finite matching float32 " + name + " required")
    _require(isinstance(states, torch.Tensor) and states.dtype == torch.float64 and states.device.type == "cpu"
             and states.layout == torch.strided and states.ndim == 3 and states.shape[1:] == (6, 10)
             and len(states) > 0 and bool(torch.isfinite(states).all()), "Finite CPU float64 K5 states [B,6,10] required")
    batch = len(states)
    _require(isinstance(actions, torch.Tensor) and actions.dtype == torch.float64 and actions.device.type == "cpu"
             and actions.layout == torch.strided and actions.shape == (batch, 5, 2)
             and bool(torch.isfinite(actions).all()), "Finite matching CPU float64 K5 actions required")
    _require(isinstance(mask, torch.Tensor) and mask.dtype == torch.bool and mask.device.type == "cpu"
             and mask.layout == torch.strided and mask.shape == (batch, 5) and bool(mask.all()),
             "The registered case requires all-valid K5 masks without denominator changes")
    _require(isinstance(references, Mapping) and set(references) == set(RAW_LOSS_ORDER)
             and all(isinstance(value, (int, float, np.floating)) and math.isfinite(value) and value >= 0.
                     for value in references.values()), "Five finite nonnegative reference loss values required")
    return named, expected, batch


def _loss_values(terms):
    values = {name: float(getattr(terms, name)) for name in RAW_LOSS_ORDER}
    _require(all(math.isfinite(value) and value >= 0. for value in values.values()),
             "Existing loss returned a nonfinite or negative scalar")
    values["base"] = values["one_step"] + .1 * values["rollout"]
    values["weighted_aux"] = .2 * (values["relative_teacher"] + .1 * values["relative_rollout"])
    values["common_total"] = values["total"]
    return values


def classify_total_change(change, margin):
    """A deterministic numeric tie band, not a confidence or success threshold."""
    _require(math.isfinite(change) and math.isfinite(margin) and margin > 0., "Finite change and positive tie margin required")
    return "INCREASE" if change > margin else "DECREASE" if change < -margin else "NUMERIC_TIE"


def finite_step_probe(model, normalizer, states, actions, mask, total_keep_delta, *,
                      raw_total_gradient, reference_losses, parameter_order=None):
    """Evaluate alpha0/.25/.5/1 independently, retaining every failed endpoint.

    The primary Taylor displacement is float32(theta_alpha - theta). A separate
    float64 difference of the two stored float32 endpoints is sensitivity only.
    Numerical endpoint failures use NaN placeholders in numeric arrays and None
    scalar fields in JSON; they retain their fixed alpha/case denominators.
    """
    named, order, batch = _validate(model, normalizer, states, actions, mask,
                                    total_keep_delta, raw_total_gradient, reference_losses, parameter_order)
    before = _source_snapshot(model, normalizer, states, actions, mask, total_keep_delta, raw_total_gradient)
    originals = {name: value.detach().clone() for name, value in named}
    original_buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    theta = torch.cat([originals[item["name"]].reshape(-1) for item in order]).numpy().copy()
    size = len(theta)
    evaluated = np.full((4, size), np.nan, dtype=np.float32)
    rounded = np.full_like(evaluated, np.nan)
    teachers = np.full((4, 5 * batch, 10), np.nan, dtype=np.float32)
    free_states = np.full((4, batch, 6, 10), np.nan, dtype=np.float64)
    denominators = {"batch_windows": batch, "valid_transitions": 5 * batch,
                    "base_scalar_terms": 50 * batch, "relative_scalar_terms": 10 * batch}
    gradient64, delta64 = raw_total_gradient.astype(np.float64), total_keep_delta.astype(np.float64)
    records = []
    checks = None
    try:
        for index, alpha in enumerate(ALPHAS):
            parameters = {}
            with torch.no_grad():
                for item in order:
                    displacement = torch.from_numpy(total_keep_delta[item["start"]:item["end"]].copy()).reshape(item["shape"])
                    displacement.mul_(alpha)
                    parameters[item["name"]] = originals[item["name"]].clone().add_(displacement)
                evaluated[index] = torch.cat([parameters[item["name"]].reshape(-1) for item in order]).numpy()
                rounded[index] = torch.from_numpy(evaluated[index].copy()).sub_(torch.from_numpy(theta)).numpy()
            endpoint64 = evaluated[index].astype(np.float64) - theta.astype(np.float64)
            nominal = alpha * delta64
            finite_parameters = bool(np.isfinite(evaluated[index]).all() and np.isfinite(rounded[index]).all())
            actual64 = rounded[index].astype(np.float64)
            distance = {"nominal_scaled_norm": float(np.linalg.norm(nominal)),
                        "rounded_difference_norm": float(np.linalg.norm(actual64)) if finite_parameters else None,
                        "exact_endpoint_difference_norm": float(np.linalg.norm(endpoint64)) if finite_parameters else None,
                        "rounded_minus_nominal_l2": float(np.linalg.norm(actual64 - nominal)) if finite_parameters else None,
                        "exact_minus_nominal_l2": float(np.linalg.norm(endpoint64 - nominal)) if finite_parameters else None}
            taylor = {"nominal_directional_change": float(gradient64 @ nominal),
                      "rounded_directional_change": float(gradient64 @ actual64) if finite_parameters else None,
                      "exact_endpoint_directional_change": float(gradient64 @ endpoint64) if finite_parameters else None,
                      "primary_prediction": None, "endpoint_sensitivity_prediction": None,
                      "nominal_prediction": None, "residual_actual_minus_primary": None,
                      "residual_actual_minus_endpoint": None}
            record = {"index": index, "alpha": alpha, "status": "NUMERICAL_FAILURE",
                      "loss_values": {name: None for name in LOSS_ORDER}, "denominators": dict(denominators),
                      "forward_calls": 0, "forward_rows": [], "forward_total_rows": 0,
                      "displacement": distance, "taylor": taylor,
                      "loss_differences": {name: None for name in ("base", "weighted_aux", "common_total")},
                      "total_change_relation": "NOT_INTERPRETED", "error": None}
            buffers = {name: value.clone() for name, value in original_buffers.items()}
            proxy = _FunctionalProxy(model, parameters, buffers)
            rows = []
            def count(_module, inputs):
                rows.append(int(inputs[0].shape[0]))
            def capture(_module, _inputs, output):
                if len(rows) == 1 and isinstance(output, torch.Tensor) and output.shape == (5 * batch, 10):
                    teachers[index] = output.detach().numpy().copy()
            pre_handle = proxy.register_forward_pre_hook(count)
            output_handle = proxy.register_forward_hook(capture)
            parameter_hash = _mapping_hash(parameters)
            buffer_hash = _mapping_hash(buffers)
            try:
                if finite_parameters:
                    # Each alpha begins with the same original CPU RNG and leaves
                    # caller RNG untouched, including when evaluation fails.
                    with torch.random.fork_rng(devices=[]), torch.no_grad():
                        terms = relative_sequence_loss(proxy, normalizer, states, actions, mask,
                                                       relative_weight=.2, rollout_weight=.1)
                    _require(rows == [5 * batch] + [batch] * 5,
                             "Successful evaluation must use one teacher plus five free root calls")
                    _require((terms.valid_transitions, terms.scalar_terms, terms.relative_scalar_terms)
                             == (5 * batch, 50 * batch, 10 * batch), "Existing loss denominator changed")
                    _require(terms.predicted_states is not None and terms.predicted_states.shape == (batch, 6, 10),
                             "Full K5 free predictions are required")
                    free_states[index] = terms.predicted_states.detach().numpy().copy()
                    record["loss_values"] = _loss_values(terms)
                    record["status"] = "PASS"
                else:
                    record["error"] = {"type": "NonfiniteParameterMap", "message": "Float32 parameter addition or subtraction is nonfinite"}
            except (ValueError, RuntimeError, OverflowError) as error:
                if isinstance(error, FiniteStepProbeError):
                    raise
                record["error"] = {"type": type(error).__name__, "message": str(error)}
            finally:
                pre_handle.remove()
                output_handle.remove()
                record["forward_rows"] = rows
                record["forward_calls"] = len(rows)
                record["forward_total_rows"] = sum(rows)
                _require(_mapping_hash(parameters) == parameter_hash and _mapping_hash(buffers) == buffer_hash,
                         "Forward evaluation mutated a disposable parameter or buffer map")
            current = _source_snapshot(model, normalizer, states, actions, mask, total_keep_delta, raw_total_gradient)
            _require(all(_preservation_checks(before, current).values()), "An original input changed during finite-step evaluation")
            record["evaluated_parameters_hash"] = hashlib.sha256(evaluated[index].tobytes()).hexdigest()
            record["rounded_difference_hash"] = hashlib.sha256(rounded[index].tobytes()).hexdigest()
            records.append(record)
    finally:
        after = _source_snapshot(model, normalizer, states, actions, mask, total_keep_delta, raw_total_gradient)
        checks = _preservation_checks(before, after)
        _require(all(checks.values()), "Finite-step evaluation changed source model, modes, gradients, RNG or inputs")
    baseline = records[0]["loss_values"]
    reproduction_rows = {}
    for name in RAW_LOSS_ORDER:
        actual, reference = baseline[name], float(reference_losses[name])
        allowed = REPRODUCTION_ATOL + REPRODUCTION_RTOL * abs(reference)
        error = None if actual is None else abs(actual - reference)
        reproduction_rows[name] = {"actual": actual, "reference": reference,
                                    "absolute_error": error, "allowed_error": allowed,
                                    "pass": bool(error is not None and error <= allowed)}
    reproduction_status = ("NUMERICAL_FAILURE" if records[0]["status"] != "PASS" else
                           "PASS" if all(row["pass"] for row in reproduction_rows.values()) else "REPRODUCTION_FAILURE")
    reproduction = {"status": reproduction_status, "rtol": REPRODUCTION_RTOL, "atol": REPRODUCTION_ATOL,
                    "per_loss": reproduction_rows}
    neutral_band = None if baseline["common_total"] is None else TIE_MULTIPLIER * max(1., abs(baseline["common_total"]))
    for record in records:
        if record["status"] == "PASS" and baseline["common_total"] is not None:
            for name in ("base", "weighted_aux", "common_total"):
                record["loss_differences"][name] = record["loss_values"][name] - baseline[name]
            for source, target in (("rounded_directional_change", "primary_prediction"),
                                   ("exact_endpoint_directional_change", "endpoint_sensitivity_prediction"),
                                   ("nominal_directional_change", "nominal_prediction")):
                record["taylor"][target] = baseline["common_total"] + record["taylor"][source]
            record["taylor"]["residual_actual_minus_primary"] = record["loss_values"]["common_total"] - record["taylor"]["primary_prediction"]
            record["taylor"]["residual_actual_minus_endpoint"] = record["loss_values"]["common_total"] - record["taylor"]["endpoint_sensitivity_prediction"]
            if reproduction_status == "PASS":
                record["total_change_relation"] = classify_total_change(record["loss_differences"]["common_total"], neutral_band)
    numeric_failure = any(record["status"] != "PASS" for record in records)
    status = "NUMERICAL_FAILURE" if numeric_failure else reproduction_status
    if status != "PASS":
        decision = status
    elif records[3]["total_change_relation"] == "INCREASE" and any(record["total_change_relation"] == "DECREASE" for record in records[1:3]):
        decision = "NOMINAL_INCREASE_SMALLER_DECREASE"
    elif all(record["total_change_relation"] == "DECREASE" for record in records[1:]):
        decision = "NO_ALPHA1_OVERSHOOT_OBSERVED_ON_GRID"
    else:
        decision = "INCONCLUSIVE"
    summary = {"schema": "p4-fixed-direction-finite-step-evaluation-v1", "status": status,
               "alphas": list(ALPHAS), "parameter_order": order, "parameter_count": size,
               "raw_loss_order": list(RAW_LOSS_ORDER), "denominators": denominators,
               "retained_alpha_count": 4, "case_denominator": 1,
               "alpha0_reproduction": reproduction, "neutral_band": neutral_band,
               "decision": decision, "evaluations": records, "checks": checks,
               "source_state_hash_before": before["state_hash"], "source_state_hash_after": after["state_hash"],
               "counters": {"evaluations": 4, "successful_evaluations": sum(record["status"] == "PASS" for record in records),
                            "root_model_forward_calls": sum(record["forward_calls"] for record in records),
                            "root_model_forward_rows": sum(record["forward_total_rows"] for record in records),
                            "new_autograd_traversals": 0, "new_optimizer_steps": 0,
                            "new_training_updates": 0, "new_physics_steps": 0},
               "arithmetic": "Independent float32 scaled=delta*alpha; theta_alpha=original+scaled; primary displacement=float32(theta_alpha-original)",
               "primary_taylor": "float64(raw total gradient) dot promoted float32 rounded subtraction; baseline is evaluated alpha0",
               "endpoint_sensitivity": "Separate float64(theta_alpha_f32)-float64(original_f32) dot float64(raw total gradient)",
               "failure_policy": "All four alpha endpoints and the fixed case remain; failed scalar fields are null and unavailable prediction entries NaN",
               "scope": "Selected frozen train snapshot/batch/direction finite objective contrast only; no historical-cause, model-selection, generalization or physical-effect claim"}
    return {"alphas": np.asarray(ALPHAS, np.float64), "evaluated_parameters": evaluated,
            "rounded_parameter_differences": rounded, "teacher_predictions": teachers,
            "free_predicted_states": free_states, "summary": summary}
