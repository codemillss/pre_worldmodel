"""Virtual Adam addends from saved gradients and independently cloned moments.

This module neither constructs a model or optimizer nor applies an update to a
parameter. Every addend starts from one frozen optimizer state. Saved base
gradients are rounded component recombinations, not historical base backward
results, and the supplied probe gradients do not replay a historical batch.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

import numpy as np
import torch


OBJECTIVE_ORDER = ("base", "total")
MODE_ORDER = ("KEEP", "ZERO_M", "COLD")
NORM_FLOOR = 1e-12
ORTHOGONAL_COSINE = 1e-6
LEARNING_RATE = .001
BETAS = (.9, .999)
ADAM_EPSILON = 1e-8
CLIP_MAXIMUM = 1.
CLIP_EPSILON = 1e-6


class AdamDirectionProbeError(ValueError):
    """Saved inputs do not meet the supported virtual arithmetic contract."""


def _require(condition, message):
    if not condition:
        raise AdamDirectionProbeError(message)


def _integer(value):
    return type(value) is int


def _parameter_order(value, size):
    _require(isinstance(value, (list, tuple)) and len(value) > 0,
             "Nonempty parameter order required")
    result, cursor, names = [], 0, set()
    for item in value:
        _require(isinstance(item, Mapping)
                 and set(item) == {"name", "shape", "numel", "start", "end"},
                 "Parameter order requires name, shape, numel, start and end")
        name, shape = item["name"], item["shape"]
        _require(isinstance(name, str) and name and name not in names,
                 "Unique nonempty parameter names required")
        _require(isinstance(shape, (list, tuple))
                 and all(_integer(n) and n > 0 for n in shape),
                 "Parameter shape must contain positive integer dimensions")
        count = math.prod(shape)
        start, end = item["start"], item["end"]
        _require(all(_integer(item[k]) for k in ("numel", "start", "end"))
                 and item["numel"] == count and start == cursor
                 and end == start + count and end <= size,
                 "Parameter slices must exactly and contiguously cover gradients")
        result.append({"name": name, "shape": list(shape), "numel": count,
                       "start": start, "end": end})
        names.add(name)
        cursor = end
    _require(cursor == size, "Parameter slices must exactly cover gradients")
    return result


def _optimizer_state(value, order):
    _require(isinstance(value, Mapping) and set(value) == {"state", "param_groups"},
             "Optimizer state and parameter groups required")
    groups = value["param_groups"]
    _require(isinstance(groups, list) and len(groups) == 1,
             "Exactly one registered Adam parameter group required")
    group = groups[0]
    fields = {"lr", "betas", "eps", "weight_decay", "amsgrad", "maximize",
              "foreach", "capturable", "differentiable", "fused", "params",
              "decoupled_weight_decay"}
    _require(isinstance(group, Mapping) and set(group) == fields,
             "Unsupported or missing Adam parameter-group options")
    _require(all(type(group[k]) in (int, float) for k in ("lr", "eps", "weight_decay"))
             and group["lr"] == LEARNING_RATE and group["eps"] == ADAM_EPSILON
             and isinstance(group["betas"], (tuple, list))
             and all(type(beta) in (int, float) for beta in group["betas"])
             and tuple(group["betas"]) == BETAS and group["weight_decay"] == 0.,
             "Unsupported Adam learning rate, betas, epsilon or weight decay")
    _require(all(group[k] is False for k in
                 ("amsgrad", "maximize", "capturable", "differentiable",
                  "decoupled_weight_decay"))
             and (group["foreach"] is None or group["foreach"] is False)
             and (group["fused"] is None or group["fused"] is False),
             "Unsupported Adam execution options")
    ids = list(range(len(order)))
    _require(isinstance(group["params"], list)
             and all(_integer(i) for i in group["params"])
             and group["params"] == ids, "Adam parameter ids must follow parameter order")
    states = value["state"]
    _require(isinstance(states, Mapping) and set(states) == set(ids)
             and all(_integer(i) for i in states),
             "Adam state must cover every parameter id exactly")
    saved_step = None
    for index, item in enumerate(order):
        state = states[index]
        _require(isinstance(state, Mapping)
                 and set(state) == {"step", "exp_avg", "exp_avg_sq"},
                 "Unsupported or missing Adam moment fields")
        step = state["step"]
        _require(isinstance(step, torch.Tensor) and step.device.type == "cpu"
                 and step.dtype == torch.float32 and step.numel() == 1
                 and step.layout == torch.strided and not step.requires_grad
                 and bool(torch.isfinite(step).all()), "Finite CPU float32 Adam step required")
        number = float(step)
        _require(number >= 0 and number.is_integer() and number < 2**24,
                 "Adam step must be a nonnegative integer with representable next step")
        if saved_step is None:
            saved_step = int(number)
        _require(number == saved_step, "Adam saved step differs across parameters")
        for field in ("exp_avg", "exp_avg_sq"):
            tensor = state[field]
            _require(isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32
                     and tensor.device.type == "cpu" and tensor.layout == torch.strided
                     and list(tensor.shape) == item["shape"] and not tensor.requires_grad
                     and bool(torch.isfinite(tensor).all()),
                     "Finite CPU float32 Adam moments must match parameter shapes")
        _require(bool((state["exp_avg_sq"] >= 0).all()),
                 "Adam squared moments must be nonnegative")
    return states, saved_step, deepcopy(group)


def loss_geometry(gradient, displacement):
    """First-order directional derivative; positive cosine means descent."""
    g, delta = np.asarray(gradient, np.float64), np.asarray(displacement, np.float64)
    _require(g.ndim == 1 and g.shape == delta.shape and g.size > 0
             and np.isfinite(g).all() and np.isfinite(delta).all(),
             "Finite matched gradient and displacement vectors required")
    gradient_norm = float(np.linalg.norm(g))
    displacement_norm = float(np.linalg.norm(delta))
    dot = float(g @ delta)
    defined = min(gradient_norm, displacement_norm) > NORM_FLOOR
    cosine = float(np.clip(-dot / (gradient_norm * displacement_norm), -1., 1.)) if defined else None
    relation = ("UNDEFINED_NEAR_ZERO_NORM" if cosine is None else
                "DESCENT" if cosine > ORTHOGONAL_COSINE else
                "ASCENT" if cosine < -ORTHOGONAL_COSINE else "NEAR_ORTHOGONAL")
    return {"gradient_norm": gradient_norm, "displacement_norm": displacement_norm,
            "dot": dot, "descent_cosine": cosine, "relation": relation}


def direction_contrast(first, second):
    """Descriptive geometry of two independently computed virtual addends."""
    first, second = np.asarray(first, np.float64), np.asarray(second, np.float64)
    _require(first.ndim == 1 and first.shape == second.shape and first.size > 0
             and np.isfinite(first).all() and np.isfinite(second).all(),
             "Finite matched virtual directions required")
    na, nb = float(np.linalg.norm(first)), float(np.linalg.norm(second))
    dot = float(first @ second)
    cosine = float(np.clip(dot / (na * nb), -1., 1.)) if min(na, nb) > NORM_FLOOR else None
    relation = ("UNDEFINED_NEAR_ZERO_NORM" if cosine is None else
                "NEAR_ORTHOGONAL" if abs(cosine) <= ORTHOGONAL_COSINE else
                "ALIGNED" if cosine > 0 else "OPPOSED")
    return {"first_norm": na, "second_norm": nb, "dot": dot,
            "cosine": cosine, "relation": relation,
            "second_to_first_norm_ratio": nb / na if na > NORM_FLOOR else None,
            "difference_l2": float(np.linalg.norm(second - first))}


def _clip(flat, order):
    blocks = [torch.from_numpy(flat[item["start"]:item["end"]].copy()).reshape(item["shape"])
              for item in order]
    norms = torch._foreach_norm(blocks, 2.)
    norm = torch.linalg.vector_norm(torch.stack(norms), ord=2)
    coefficient = torch.clamp(CLIP_MAXIMUM / (norm + CLIP_EPSILON), max=1.)
    _require(bool(torch.isfinite(norm)) and bool(torch.isfinite(coefficient)),
             "Nonfinite float32 clipping norm or coefficient")
    torch._foreach_mul_(blocks, coefficient)
    result = torch.cat([block.reshape(-1) for block in blocks]).numpy().copy()
    return result, {"parameterwise_float32_norm": float(norm),
                    "flat_float64_norm": float(np.linalg.norm(flat.astype(np.float64))),
                    "parameter_norms_float32": [float(n) for n in norms],
                    "coefficient": float(coefficient), "maximum_norm": CLIP_MAXIMUM,
                    "epsilon": CLIP_EPSILON, "would_scale": bool(coefficient < 1),
                    "applied_to_parameters_or_grad_buffers": False,
                    "scope": "Global clipping arithmetic on disposable gradient copies only"}


def _virtual_addend(clipped, order, states, saved_step, mode):
    next_step = 1 if mode == "COLD" else saved_step + 1
    beta1, beta2 = BETAS
    step_size = LEARNING_RATE / (1. - beta1**next_step)
    correction2_sqrt = (1. - beta2**next_step)**.5
    blocks = []
    for index, item in enumerate(order):
        gradient = torch.from_numpy(clipped[item["start"]:item["end"]].copy()).reshape(item["shape"])
        state = states[index]
        first = state["exp_avg"].detach().clone() if mode == "KEEP" else torch.zeros_like(state["exp_avg"])
        second = torch.zeros_like(state["exp_avg_sq"]) if mode == "COLD" else state["exp_avg_sq"].detach().clone()
        first.lerp_(gradient, 1. - beta1)
        second.mul_(beta2).addcmul_(gradient, gradient, value=1. - beta2)
        denominator = (second.sqrt() / correction2_sqrt).add_(ADAM_EPSILON)
        delta = torch.zeros_like(first).addcdiv_(first, denominator, value=-step_size)
        _require(bool(torch.isfinite(first).all()) and bool(torch.isfinite(second).all())
                 and bool(torch.isfinite(denominator).all()) and bool((denominator > 0).all())
                 and bool(torch.isfinite(delta).all()), "Nonfinite virtual Adam arithmetic")
        blocks.append(delta.reshape(-1))
    return torch.cat(blocks).numpy().copy(), next_step


def _geometry_map(references, delta, indices=None):
    return {name: loss_geometry(g if indices is None else g[indices],
                                delta if indices is None else delta[indices])
            for name, g in references.items()}


def _head_groups(references, delta, order):
    by_name = {item["name"]: item for item in order}
    weight, bias = by_name.get("network.4.weight"), by_name.get("network.4.bias")
    scope = ("Final-layer parameter-coordinate subsets; no input attribution, "
             "observed state response or physical causation")
    if (weight is None or bias is None or len(weight["shape"]) != 2
            or weight["shape"][0] != 10 or bias["shape"] != [10]):
        return {"status": "NOT_APPLICABLE", "groups": {}, "scope": scope,
                "reason": "Registered network.4 ten-output Linear weight and bias are absent"}
    width, groups = weight["shape"][1], {}
    for name, rows in (("object_xy", [0, 1]), ("yaw", [2]),
                       ("object_velocity", [3, 4, 5]), ("pusher_xy", [6, 7]),
                       ("pusher_velocity", [8, 9])):
        indices = np.asarray([weight["start"] + row * width + column
                              for row in rows for column in range(width)]
                             + [bias["start"] + row for row in rows], dtype=np.int64)
        groups[name] = {"output_rows": rows, "parameter_count": int(indices.size),
                        "weight_parameter": weight["name"], "bias_parameter": bias["name"],
                        "delta_norm": float(np.linalg.norm(delta[indices].astype(np.float64))),
                        "loss_geometry": _geometry_map(references, delta, indices)}
    return {"status": "APPLICABLE", "groups": groups, "scope": scope,
            "partition": "All ten final output weight rows and biases exactly once; hidden parameters excluded"}


def virtual_directions(raw_gradients, base_f32, parameter_order, optimizer_state):
    """Return six proposed addends; original tensors and arrays stay untouched.

    Ordering is base/total, each with KEEP, ZERO_M, COLD. ZERO_M removes only
    the first moment while retaining second moments and saved step. COLD is a
    full Adam restart, including bias-correction age. No cases are chained.
    """
    raw, base = np.asarray(raw_gradients), np.asarray(base_f32)
    _require(raw.dtype == np.float32 and raw.ndim == 2 and raw.shape[0] == 5
             and raw.shape[1] > 0 and np.isfinite(raw).all(),
             "Five finite nonempty raw float32 gradient vectors required")
    _require(base.dtype == np.float32 and base.shape == (raw.shape[1],)
             and np.isfinite(base).all(), "Finite matching base float32 vector required")
    order = _parameter_order(parameter_order, raw.shape[1])
    states, saved_step, group = _optimizer_state(optimizer_state, order)
    raw64 = raw.astype(np.float64)
    references = {"base": base.astype(np.float64),
                  "aux": .2 * (raw64[2] + .1 * raw64[3]), "total": raw64[4]}
    clipped, clips, directions, records = [], {}, [], []
    with torch.no_grad():
        for objective, gradient in (("base", base), ("total", raw[4])):
            flat, clips[objective] = _clip(gradient, order)
            clipped.append(flat)
            for mode in MODE_ORDER:
                delta, next_step = _virtual_addend(flat, order, states, saved_step, mode)
                directions.append(delta)
                blocks = [{**item,
                           "delta_norm": float(np.linalg.norm(delta[item["start"]:item["end"]].astype(np.float64))),
                           "loss_geometry": _geometry_map(references, delta, slice(item["start"], item["end"]))}
                          for item in order]
                records.append({"index": len(directions) - 1, "objective": objective, "mode": mode,
                                "saved_step": saved_step, "next_step": next_step,
                                "delta_norm": float(np.linalg.norm(delta.astype(np.float64))),
                                "loss_geometry": _geometry_map(references, delta),
                                "parameter_blocks": blocks,
                                "output_head_groups": _head_groups(references, delta, order)})
    objective_contrasts = []
    for index, mode in enumerate(MODE_ORDER):
        objective_contrasts.append({"mode": mode, "first_objective": "base", "second_objective": "total",
                                   "first_index": index, "second_index": index + len(MODE_ORDER),
                                   **direction_contrast(directions[index], directions[index + len(MODE_ORDER)])})
    state_contrasts = []
    for index, objective in enumerate(OBJECTIVE_ORDER):
        offset = index * len(MODE_ORDER)
        for mode_index, mode in enumerate(MODE_ORDER[1:], start=1):
            state_contrasts.append({"objective": objective, "first_mode": "KEEP", "second_mode": mode,
                                   "first_index": offset, "second_index": offset + mode_index,
                                   **direction_contrast(directions[offset], directions[offset + mode_index])})
    summary = {"objective_order": list(OBJECTIVE_ORDER), "mode_order": list(MODE_ORDER),
               "parameter_order": order, "parameter_count": raw.shape[1],
               "adam": {"saved_step": saved_step, "learning_rate": group["lr"],
                        "betas": list(group["betas"]), "epsilon": group["eps"],
                        "weight_decay": group["weight_decay"], "amsgrad": False,
                        "arithmetic": "CPU float32 single-tensor Adam arithmetic on disposable copies; zero-origin addcdiv addend",
                        "modes": {"KEEP": "saved m, v and step; next step saved+1",
                                  "ZERO_M": "zero m; saved v and step; next step saved+1",
                                  "COLD": "zero m, v and step; next step 1"}},
               "norm_floor": NORM_FLOOR, "near_orthogonal_absolute_cosine": ORTHOGONAL_COSINE,
               "clips": clips, "directions": records,
               "objective_contrasts": objective_contrasts, "state_contrasts": state_contrasts,
               "reference_gradients": {"base": "Saved rounded float32 component recombination; not historical exact base backward",
                                       "aux": "Float64 0.2*(raw relative_teacher + 0.1*raw relative_rollout)",
                                       "total": "Saved independently differentiated raw float32 total"},
               "scope": ("Independent hypothetical next-batch Adam addends and first-order geometry only; "
                         "no parameter addition, optimizer call, historical replay, finite loss change, "
                         "physical effect, generalization or causal training attribution. Final parameter-addition "
                         "rounding is excluded. total-base addend differences are nonlinear counterfactual differences.")}
    return {"directions": np.stack(directions), "clipped_gradients": np.stack(clipped), "summary": summary}
