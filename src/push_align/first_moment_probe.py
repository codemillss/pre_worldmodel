"""Route frozen R1-final noncontact KEEP/ZERO_M directions to stop targets.

Only routing validation and provenance are added. The unchanged finite-step
helper owns all evaluation arithmetic, independent alpha copies and failures.
The caller binds the saved direction files and their common gradient/v/step.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json

import numpy as np
import torch
from torch import nn

from .cross_batch_probe import array_sha256
from .data_intervention_training import tree_hash
from .finite_step_probe import RAW_LOSS_ORDER, finite_step_probe


SNAPSHOT = "R1_u1000"
SOURCE_PROBE = "D11_noncontact"
TARGET_PROBE = "D11_zero5"
DIRECTION_INDICES = {"KEEP": 3, "ZERO_M": 4}


class FirstMomentProbeError(ValueError):
    """Bindings do not describe the registered first-moment contrast."""


def _require(condition, message):
    if not condition:
        raise FirstMomentProbeError(message)


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise FirstMomentProbeError("Routing bindings must be finite JSON data") from error


def _validate(model, states, actions, mask, delta, target_gradient, target_reference,
              source, target, gradient, loss, parameter_order):
    _require(isinstance(model, nn.Module), "A frozen source model is required")
    bindings = (source, target, gradient, loss)
    required = {"snapshot_id", "probe_name", "model_tree_hash", "parameter_order"}
    _require(all(isinstance(binding, Mapping) and required <= set(binding) for binding in bindings),
             "Every routing binding requires snapshot, probe, model tree hash and parameter order")
    _require(all(binding["snapshot_id"] == SNAPSHOT for binding in bindings),
             "Every binding must describe the same R1_u1000 snapshot")
    _require(source["probe_name"] == SOURCE_PROBE
             and all(binding["probe_name"] == TARGET_PROBE for binding in bindings[1:]),
             "Only D11_noncontact source to D11_zero5 target/gradient/loss is allowed")
    _require(isinstance(parameter_order, list) and bool(parameter_order)
             and all(binding["parameter_order"] == parameter_order for binding in bindings),
             "Source/target parameter orders must match the bound order exactly")
    model_hash = tree_hash(model.state_dict())
    _require(all(binding["model_tree_hash"] == model_hash for binding in bindings),
             "Routing model tree identities differ from the actual frozen model")
    mode = source.get("mode")
    _require(isinstance(mode, str) and mode in DIRECTION_INDICES
             and source.get("objective") == "total"
             and type(source.get("direction_index")) is int
             and source["direction_index"] == DIRECTION_INDICES[mode],
             "Only saved TOTAL/KEEP index3 or TOTAL/ZERO_M index4 is allowed")
    _require(isinstance(delta, np.ndarray) and delta.dtype == np.float32 and delta.ndim == 1
             and len(delta) > 0 and np.isfinite(delta).all(), "Finite native float32 source direction required")
    delta_hash = array_sha256(delta)
    _require(source.get("delta_sha256") == delta_hash,
             "Source direction typed hash differs; normalization or substitution is forbidden")
    _require(gradient.get("component") == "total", "Target Taylor gradient must be the raw total component")
    _require(isinstance(target_gradient, np.ndarray) and target_gradient.dtype == np.float32
             and target_gradient.shape == delta.shape and np.isfinite(target_gradient).all(),
             "Finite matching raw float32 target-stop gradient required")
    gradient_hash = array_sha256(target_gradient)
    _require(gradient.get("gradient_sha256") == gradient_hash,
             "Target gradient typed hash differs; the source gradient cannot replace it")
    _require(isinstance(target_reference, Mapping) and set(target_reference) == set(RAW_LOSS_ORDER)
             and loss.get("loss_values") == target_reference,
             "Reference losses must equal the bound target-stop raw5 values")
    _require(isinstance(states, torch.Tensor) and states.ndim == 3 and len(states) > 0
             and type(target.get("batch_windows")) is int and target["batch_windows"] == len(states),
             "Target stop tensor batch-window count differs")
    batch_hashes = {name: array_sha256(value) for name, value in
                    (("states", states), ("actions", actions), ("mask", mask))}
    _require(all(target.get(name + "_sha256") == digest for name, digest in batch_hashes.items()),
             "Target stop states/actions/mask typed identities differ")
    record = {"source": source, "target": target, "gradient": gradient, "loss": loss,
              "reference_losses": target_reference, "parameter_order": parameter_order}
    return mode, model_hash, delta_hash, gradient_hash, batch_hashes, record, _canonical(record)


def first_moment_probe(model, normalizer, target_states, target_actions, target_mask, source_delta, *,
                       source_binding, target_binding, target_gradient_binding, target_loss_binding,
                       target_stop_gradient, target_stop_reference, parameter_order):
    """Evaluate one saved native mode on bound same-snapshot stop targets.

    Returns the original five arrays and numerical summary unchanged, adding
    only ``summary.first_moment_routing``. No direction or gradient is recomputed.
    """
    mode, model_hash, delta_hash, gradient_hash, batch_hashes, record, metadata_before = _validate(
        model, target_states, target_actions, target_mask, source_delta,
        target_stop_gradient, target_stop_reference, source_binding, target_binding,
        target_gradient_binding, target_loss_binding, parameter_order)
    try:
        result = finite_step_probe(model, normalizer, target_states, target_actions, target_mask,
                                   source_delta, raw_total_gradient=target_stop_gradient,
                                   reference_losses=target_stop_reference, parameter_order=parameter_order)
    finally:
        _require(_canonical(record) == metadata_before, "First-moment routing metadata changed during evaluation")
        _require(array_sha256(source_delta) == delta_hash, "The native source direction changed during evaluation")
        _require(array_sha256(target_stop_gradient) == gradient_hash, "The target raw total gradient changed during evaluation")
        _require(all(array_sha256(value) == batch_hashes[name] for name, value in
                     (("states", target_states), ("actions", target_actions), ("mask", target_mask))),
                 "The bound target-stop input tensors changed during evaluation")
        _require(tree_hash(model.state_dict()) == model_hash, "The original frozen model changed during evaluation")
    summary = result["summary"]
    _require("first_moment_routing" not in summary, "Unexpected preexisting first-moment routing metadata")
    summary["first_moment_routing"] = {
        "schema": "p4-first-moment-routing-v1", "snapshot_id": SNAPSHOT,
        "source_probe_name": SOURCE_PROBE, "target_probe_name": TARGET_PROBE, "source_mode": mode,
        "model_tree_hash": model_hash, "parameter_order": deepcopy(parameter_order),
        "source_direction": {"objective": "total", "mode": mode, "direction_index": DIRECTION_INDICES[mode],
                             "delta_sha256": delta_hash, "norm_normalized": False},
        "target_gradient": {"snapshot_id": SNAPSHOT, "probe_name": TARGET_PROBE,
                            "component": "total", "gradient_sha256": gradient_hash},
        "target_reference_losses": deepcopy(dict(target_stop_reference)),
        "target_batch": {"batch_windows": len(target_states), "typed_sha256": batch_hashes},
        "binding_records": deepcopy(record),
        "primary_target_effect": "Per-alpha evaluations[].loss_differences.common_total and total_change_relation, relative to target alpha0 under unchanged helper failure/tie rules",
        "secondary_finite_pattern_decision": summary["decision"],
        "secondary_decision_limit": "The composite finite-pattern decision is secondary; INCONCLUSIVE does not mean absence of target increase",
        "routing_checks": {"same_snapshot": True, "same_actual_model": True, "same_parameter_order": True,
                           "allowed_saved_mode_and_index": True, "native_source_direction_unchanged": True,
                           "target_gradient_and_reference_bound": True, "target_batch_unchanged": True,
                           "binding_metadata_unchanged": True},
        "scope": "Native saved KEEP/ZERO_M addends retain their norms. The caller binds their common raw gradient, saved v and step; this wrapper verifies routing and typed identity only. Local stop-loss contrasts do not establish whole-training cause, generalization or physical outcomes",
    }
    return result
