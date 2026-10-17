"""Float32 delta model with a shared float64 physical rollout boundary.

The network learns normalized state changes, not control commands. Rollout
uses only its previous predicted state and the supplied applied target
velocities. This module neither accesses a simulator nor clips predictions.
"""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from .normalization import Normalizer, NormalizationError
from .representation import RepresentationError, encode, integrate


class DynamicModelError(ValueError):
    """Model configuration, tensors, or predictions violate the contract."""


def model_spec(config):
    """Return the exact registered pilot architecture and numeric boundary."""
    if not isinstance(config, dict):
        raise DynamicModelError("Model configuration must be an object")
    runtime, model = config.get("runtime", {}), config.get("model", {})
    expected = {"kind": "mlp_delta", "hidden_widths": [128, 128],
                "activation": "silu", "output_activation": "linear",
                "learned_inputs": 11, "learned_outputs": 10}
    if not isinstance(runtime, dict) or not isinstance(model, dict):
        raise DynamicModelError("Missing runtime/model configuration")
    for key, value in expected.items():
        actual = model.get(key)
        if actual != value or (key in ("learned_inputs", "learned_outputs") and type(actual) is not int):
            raise DynamicModelError(f"Unsupported pilot model {key}")
    if any(type(width) is not int for width in model["hidden_widths"]):
        raise DynamicModelError("Hidden widths require integers")
    if runtime.get("device") != "cpu" or runtime.get("model_dtype") != "float32":
        raise DynamicModelError("Pilot model requires CPU float32")
    return {**deepcopy(expected), "architecture": [11, 128, 128, 10],
            "parameter_count": 19338, "device": "cpu", "model_dtype": "float32",
            "physical_dtype": "float64", "prediction": "normalized physical state delta",
            "forward_interface": "[...,11] float32 -> [...,10] float32",
            "rollout_interface": "physical float64 state/action -> physical float64 state"}


def _no_autocast():
    if torch.is_autocast_enabled("cpu"):
        raise DynamicModelError("CPU autocast is forbidden at the registered float32 boundary")


def _tensor(value, width, name, dtype):
    if not isinstance(value, torch.Tensor):
        raise DynamicModelError(f"{name} requires a Torch tensor")
    if value.dtype != dtype or value.device.type != "cpu" or value.layout != torch.strided:
        raise DynamicModelError(f"{name} requires dense CPU {dtype}")
    if value.ndim < 1 or value.shape[-1] != width:
        raise DynamicModelError(f"{name} requires shape [..., {width}]")
    if not bool(torch.isfinite(value).all()):
        raise DynamicModelError(f"{name} requires finite values")


def _model_parameters(model):
    if not isinstance(model, nn.Module):
        raise DynamicModelError("Dynamics model requires a torch.nn.Module")
    for value in (*model.parameters(), *model.buffers()):
        if value.device.type != "cpu":
            raise DynamicModelError("Dynamics model must reside on CPU")
        if value.is_floating_point() and value.dtype != torch.float32:
            raise DynamicModelError("Dynamics model floating values must be float32")
        if not bool(torch.isfinite(value).all()):
            raise DynamicModelError("Dynamics model contains nonfinite values")


class DeltaMLP(nn.Module):
    """Registered 11 -> 128 -> 128 -> 10 SiLU MLP.

    Callers seed Torch before construction. Construction explicitly chooses
    CPU/float32 without changing the global seed, default dtype, or mode.
    """

    def __init__(self, config):
        super().__init__()
        self.spec = model_spec(config)
        self.network = nn.Sequential(
            nn.Linear(11, 128, dtype=torch.float32, device="cpu"), nn.SiLU(),
            nn.Linear(128, 128, dtype=torch.float32, device="cpu"), nn.SiLU(),
            nn.Linear(128, 10, dtype=torch.float32, device="cpu"),
        )

    def forward(self, normalized_features):
        _no_autocast()
        _tensor(normalized_features, 11, "normalized_features", torch.float32)
        _model_parameters(self)
        result = self.network(normalized_features)
        _tensor(result, 10, "normalized_delta", torch.float32)
        return result


def predict_delta(model, normalizer, state, applied_action):
    """Predict physical float64 deltas through the shared feature transforms.

    The one deliberate cast occurs after float64 encoding/normalization and
    before network evaluation. The output is cast back before denormalizing;
    these differentiable casts preserve the recurrent autograd graph.
    This function checks finite values but leaves the raw yaw limit to
    integration, allowing evaluators to record and classify invalid deltas.
    """
    _no_autocast()
    _tensor(state, 10, "state", torch.float64)
    _tensor(applied_action, 2, "applied_action", torch.float64)
    if state.shape[:-1] != applied_action.shape[:-1]:
        raise DynamicModelError("State/action leading shapes must match without broadcasting")
    _model_parameters(model)
    if not isinstance(normalizer, Normalizer):
        raise DynamicModelError("Dynamics requires the schema-bound Normalizer")
    try:
        features = encode(state, applied_action)
        normalized = normalizer.normalize_features(features).to(torch.float32)
        _tensor(normalized, 11, "normalized_features", torch.float32)
        output = model(normalized)
        _tensor(output, 10, "normalized_delta", torch.float32)
        if output.shape[:-1] != state.shape[:-1]:
            raise DynamicModelError("Model output changed the leading batch shape")
        physical_delta = normalizer.denormalize_delta(output.to(torch.float64))
        _tensor(physical_delta, 10, "physical_delta", torch.float64)
        return physical_delta
    except (RepresentationError, NormalizationError) as error:
        raise DynamicModelError(str(error)) from error


def predict_next(model, normalizer, state, applied_action):
    """Integrate one model prediction without clipping or physics correction."""
    delta = predict_delta(model, normalizer, state, applied_action)
    try:
        return integrate(state, delta)
    except RepresentationError as error:
        raise DynamicModelError(str(error)) from error


def free_rollout(model, normalizer, initial_state, actions):
    """Return [..., K+1,10] from [...,10] and [...,K,2], K >= 1.

    No future state input is accepted. Every step recomputes features from
    the preceding prediction; gradients span all future steps. Module
    training/evaluation mode is left exactly as the caller selected it.
    """
    _no_autocast()
    _tensor(initial_state, 10, "initial_state", torch.float64)
    _tensor(actions, 2, "actions", torch.float64)
    if actions.ndim < 2 or actions.shape[-2] < 1 or initial_state.shape[:-1] != actions.shape[:-2]:
        raise DynamicModelError("Actions require matching [..., K, 2] with K >= 1")
    states = [initial_state]
    for index in range(actions.shape[-2]):
        states.append(predict_next(model, normalizer, states[-1], actions[..., index, :]))
    return torch.stack(states, dim=-2)
