"""Route one saved native source direction to the bound same-snapshot stop batch.

The frozen finite_step_probe performs every numerical evaluation unchanged.
This wrapper validates routing identities and adds provenance only. It never
normalizes a direction, computes a derivative, or changes a source/target loss.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json

import numpy as np
import torch
from torch import nn

from .data_intervention_training import tree_hash
from .finite_step_probe import RAW_LOSS_ORDER, finite_step_probe


SOURCE_PROBES = ("historical_update1", "D11_zero5", "D11_transition", "D11_noncontact")
TARGET_PROBE = "D11_zero5"


class CrossBatchProbeError(ValueError):
    """Source, target or bound provenance does not describe the registered route."""


def _require(condition, message):
    if not condition:
        raise CrossBatchProbeError(message)


def array_sha256(value):
    """Typed SHA256: canonical dtype/shape JSON, NUL, then C-order array bytes."""
    if isinstance(value, torch.Tensor):
        _require(value.device.type == "cpu" and value.layout == torch.strided,
                 "Typed array identity requires a dense CPU tensor")
        array = value.detach().numpy()
    else:
        _require(isinstance(value, np.ndarray), "Typed array identity requires an array or tensor")
        array = value
    _require(array.dtype.kind in "buif", "Typed array identity forbids object or complex data")
    descriptor = json.dumps({"dtype": str(array.dtype), "shape": list(array.shape)},
                            sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = np.ascontiguousarray(array).tobytes(order="C")
    return hashlib.sha256(descriptor + b"\0" + payload).hexdigest()


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise CrossBatchProbeError("Routing bindings must be finite JSON data") from error


def _identity(binding, label):
    required = {"snapshot_id", "probe_name", "model_tree_hash", "parameter_order"}
    _require(isinstance(binding, Mapping) and required <= set(binding),
             label + " binding requires snapshot, probe, model tree hash and parameter order")
    _require(isinstance(binding["snapshot_id"], str) and bool(binding["snapshot_id"]),
             label + " binding requires a nonempty snapshot id")
    _require(isinstance(binding["model_tree_hash"], str)
             and len(binding["model_tree_hash"]) == 64
             and set(binding["model_tree_hash"]) <= set("0123456789abcdef"),
             label + " binding requires a canonical model tree hash")
    return binding


def _validate_route(model, states, actions, mask, delta, target_gradient, target_reference,
                    source, target, gradient_binding, loss_binding, parameter_order):
    _require(isinstance(model, nn.Module), "A frozen source model is required")
    source, target, gradient_binding, loss_binding = (
        _identity(binding, label) for binding, label in
        ((source, "Source direction"), (target, "Target batch"),
         (gradient_binding, "Target gradient"), (loss_binding, "Target loss")))
    bindings = (source, target, gradient_binding, loss_binding)
    _require(all(binding["snapshot_id"] == source["snapshot_id"] for binding in bindings),
             "Every source and target binding must refer to the same snapshot")
    _require(source["probe_name"] in SOURCE_PROBES, "Unknown source probe")
    _require(all(binding["probe_name"] == TARGET_PROBE for binding in bindings[1:]),
             "Target batch, gradient and reference losses must all be D11_zero5")
    _require(isinstance(parameter_order, list) and bool(parameter_order)
             and all(binding["parameter_order"] == parameter_order for binding in bindings),
             "Source/target parameter orders must match the bound order exactly")
    model_hash = tree_hash(model.state_dict())
    _require(all(binding["model_tree_hash"] == model_hash for binding in bindings),
             "Source/target model tree identities do not match the current same-snapshot model")
    _require(source.get("objective") == "total" and source.get("mode") == "KEEP"
             and type(source.get("direction_index")) is int and source["direction_index"] == 3,
             "Only the saved source TOTAL/KEEP direction at index3 is allowed")
    _require(isinstance(delta, np.ndarray) and delta.dtype == np.float32 and delta.ndim == 1
             and len(delta) > 0 and np.isfinite(delta).all(), "Finite native float32 source direction required")
    delta_hash = array_sha256(delta)
    _require(source.get("delta_sha256") == delta_hash,
             "Source direction typed hash differs; normalization or substitution is forbidden")
    _require(gradient_binding.get("component") == "total", "Target Taylor gradient must be the raw total component")
    _require(isinstance(target_gradient, np.ndarray) and target_gradient.dtype == np.float32
             and target_gradient.shape == delta.shape and np.isfinite(target_gradient).all(),
             "Finite matching raw float32 target-stop gradient required")
    gradient_hash = array_sha256(target_gradient)
    _require(gradient_binding.get("gradient_sha256") == gradient_hash,
             "Target gradient typed hash differs; a source gradient cannot replace the target gradient")
    _require(isinstance(target_reference, Mapping) and set(target_reference) == set(RAW_LOSS_ORDER)
             and loss_binding.get("loss_values") == target_reference,
             "Reference loss values must match the bound target-stop raw5 values")
    _require(isinstance(states, torch.Tensor) and states.ndim == 3,
             "Target stop states must be the bound tensor batch")
    batch = len(states)
    _require(type(target.get("batch_windows")) is int and target["batch_windows"] == batch and batch > 0,
             "Target stop batch-window count differs")
    batch_hashes = {name: array_sha256(value) for name, value in
                    (("states", states), ("actions", actions), ("mask", mask))}
    _require(all(target.get(name + "_sha256") == digest for name, digest in batch_hashes.items()),
             "Target stop states/actions/mask typed identities differ")
    _canonical({"source": source, "target": target, "gradient": gradient_binding,
                "loss": loss_binding, "reference_losses": target_reference, "parameter_order": parameter_order})
    return model_hash, delta_hash, gradient_hash, batch_hashes


def cross_batch_probe(model, normalizer, target_states, target_actions, target_mask, source_delta, *,
                      source_binding, target_binding, target_gradient_binding, target_loss_binding,
                      target_stop_gradient, target_stop_reference, parameter_order):
    """Evaluate the native source addend on same-snapshot bound stop targets.

    The caller binds the original source files to the registration. This helper
    verifies the supplied typed arrays against those binding records before any
    model call. Existing endpoint statuses, tie rules, arrays, failed rows and
    denominators are returned unchanged; only cross_batch_routing is added.
    """
    model_hash, delta_hash, gradient_hash, batch_hashes = _validate_route(
        model, target_states, target_actions, target_mask, source_delta,
        target_stop_gradient, target_stop_reference, source_binding, target_binding,
        target_gradient_binding, target_loss_binding, parameter_order)
    binding_record = {"source": source_binding, "target": target_binding,
                      "gradient": target_gradient_binding, "loss": target_loss_binding,
                      "reference_losses": target_stop_reference, "parameter_order": parameter_order}
    metadata_before = _canonical(binding_record)
    try:
        # Source and target arrays are deliberately passed verbatim. The frozen
        # helper constructs independent alpha copies and uses only target losses
        # and the target raw total gradient for its Taylor comparisons.
        result = finite_step_probe(model, normalizer, target_states, target_actions, target_mask,
                                   source_delta, raw_total_gradient=target_stop_gradient,
                                   reference_losses=target_stop_reference, parameter_order=parameter_order)
    finally:
        _require(_canonical(binding_record) == metadata_before, "Cross-batch routing metadata changed during evaluation")
        _require(array_sha256(source_delta) == delta_hash, "The native source direction changed during evaluation")
        _require(array_sha256(target_stop_gradient) == gradient_hash, "The target raw total gradient changed during evaluation")
        _require(all(array_sha256(value) == batch_hashes[name] for name, value in
                     (("states", target_states), ("actions", target_actions), ("mask", target_mask))),
                 "The bound target-stop input tensors changed during evaluation")
        _require(tree_hash(model.state_dict()) == model_hash, "The original same-snapshot model changed during evaluation")
    summary = result["summary"]
    _require("cross_batch_routing" not in summary, "Unexpected preexisting cross-batch routing metadata")
    summary["cross_batch_routing"] = {
        "schema": "p4-cross-batch-routing-v1",
        "snapshot_id": source_binding["snapshot_id"],
        "source_probe_name": source_binding["probe_name"],
        "target_probe_name": TARGET_PROBE,
        "is_self_stop_control": source_binding["probe_name"] == TARGET_PROBE,
        "model_tree_hash": model_hash,
        "parameter_order": deepcopy(parameter_order),
        "source_direction": {"objective": "total", "mode": "KEEP", "direction_index": 3,
                             "delta_sha256": delta_hash, "norm_normalized": False},
        "target_gradient": {"snapshot_id": target_gradient_binding["snapshot_id"],
                            "probe_name": TARGET_PROBE, "component": "total", "gradient_sha256": gradient_hash},
        "target_reference_losses": deepcopy(dict(target_stop_reference)),
        "target_batch": {"batch_windows": len(target_states), "typed_sha256": batch_hashes},
        "binding_records": deepcopy(binding_record),
        "primary_target_effect": "Per-alpha evaluations[].loss_differences.common_total and total_change_relation, relative to target alpha0 under unchanged helper failure/tie rules",
        "secondary_finite_pattern_decision": summary["decision"],
        "secondary_decision_limit": "The composite finite-pattern decision is secondary; INCONCLUSIVE does not mean absence of cross-batch target increase",
        "routing_checks": {"same_snapshot": True, "same_actual_model": True, "same_parameter_order": True,
                           "native_source_direction_unchanged": True, "target_gradient_and_reference_bound": True,
                           "target_batch_unchanged": True, "binding_metadata_unchanged": True},
        "scope": "Native source directions retain their stored norms. Selected same-snapshot cross-batch stop-loss contrasts are not normalized-direction causal effects, whole-training causes, generalization or physical outcomes",
    }
    return result
