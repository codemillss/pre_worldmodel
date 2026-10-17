"""Pure state/feature/delta transforms for the learned dynamics contract.

NumPy float32/float64 inputs are computed and returned as float64. Torch
float32/float64 tensors retain their dtype, device, leading dimensions and
autograd graph. Torch is optional: this module never imports it, and obtains
the already-loaded module only when callers supply actual Torch tensors.

The feature yaw has pi symmetry; transition yaw uses a 2*pi wrap. These
operations do not prove planar physics validity or infer a rotation's winding
number from its endpoints. No transform calls or reads a simulator.
"""
from __future__ import annotations

import math
import sys

import numpy as np

STATE_FIELDS = (
    "obj_x", "obj_y", "obj_yaw", "obj_vx", "obj_vy", "obj_wz",
    "pusher_x", "pusher_y", "pusher_vx", "pusher_vy",
)
STATE_UNITS = ("m", "m", "rad", "m/s", "m/s", "rad/s", "m", "m", "m/s", "m/s")
FEATURE_FIELDS = (
    "relative_pusher_x", "relative_pusher_y", "sin_2yaw", "cos_2yaw",
    "obj_vx", "obj_vy", "obj_wz", "pusher_vx", "pusher_vy",
    "applied_vx", "applied_vy",
)
FEATURE_UNITS = ("m", "m", "1", "1", "m/s", "m/s", "rad/s", "m/s", "m/s", "m/s", "m/s")
DELTA_FIELDS = tuple(f"delta_{field}" for field in STATE_FIELDS)
DELTA_UNITS = STATE_UNITS


class RepresentationError(ValueError):
    """A tensor, array or predicted physical delta violates the contract."""


def _descriptor(value, width, name):
    if isinstance(value, np.ndarray):
        backend = "numpy"
        if value.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise RepresentationError(f"{name} requires float32 or float64 dtype")
    else:
        torch = sys.modules.get("torch")
        if torch is None or not isinstance(value, torch.Tensor):
            raise RepresentationError(f"{name} requires a NumPy array or Torch tensor")
        backend = "torch"
        if value.dtype not in (torch.float32, torch.float64):
            raise RepresentationError(f"{name} requires float32 or float64 dtype")
        if value.layout != torch.strided:
            raise RepresentationError(f"{name} requires a dense strided tensor")
    if value.ndim < 1 or value.shape[-1] != width:
        raise RepresentationError(f"{name} requires shape [..., {width}]")
    return backend


def _finite(value, backend, name):
    if backend == "numpy":
        valid = bool(np.isfinite(value).all())
    else:
        if value.device.type == "meta":
            raise RepresentationError(f"{name} requires materialized tensor values")
        valid = bool(sys.modules["torch"].isfinite(value).all())
    if not valid:
        raise RepresentationError(f"{name} requires finite values")


def _pair(first, second, first_width, second_width, first_name, second_name):
    backend = _descriptor(first, first_width, first_name)
    other_backend = _descriptor(second, second_width, second_name)
    if backend != other_backend:
        raise RepresentationError("Inputs must use the same backend")
    if first.shape[:-1] != second.shape[:-1]:
        raise RepresentationError("Inputs must have identical leading shapes; broadcasting is forbidden")
    if first.dtype != second.dtype:
        raise RepresentationError("Inputs must have identical dtype")
    if backend == "torch" and first.device != second.device:
        raise RepresentationError("Inputs must use the same device")
    _finite(first, backend, first_name)
    _finite(second, backend, second_name)
    if backend == "numpy":
        first = first.astype(np.float64, copy=False)
        second = second.astype(np.float64, copy=False)
    return first, second, backend


def _wrap_2pi(angle, backend):
    if backend == "numpy":
        return np.remainder(angle + math.pi, 2 * math.pi) - math.pi
    return sys.modules["torch"].remainder(angle + math.pi, 2 * math.pi) - math.pi


def _with_yaw(value, yaw, backend):
    pieces = (value[..., :2], yaw[..., None], value[..., 3:])
    if backend == "numpy":
        return np.concatenate(pieces, axis=-1)
    return sys.modules["torch"].cat(pieces, dim=-1)


def encode(state, applied_action):
    """Return ``[..., 11]`` features from matching ``[..., 10]``/``[..., 2]``.

    Column order is FEATURE_FIELDS. The last two features are the applied
    target velocity, distinct from measured pusher velocity at indices 7:9.
    Shapes are preserved without silently adding/removing a batch axis.
    """
    state, action, backend = _pair(state, applied_action, 10, 2, "state", "applied_action")
    if backend == "numpy":
        with np.errstate(over="ignore", invalid="ignore"):
            yaw = 2 * state[..., 2]
            result = np.concatenate((
                state[..., 6:8] - state[..., :2],
                np.sin(yaw)[..., None], np.cos(yaw)[..., None],
                state[..., 3:6], state[..., 8:10], action,
            ), axis=-1)
    else:
        torch = sys.modules["torch"]
        yaw = 2 * state[..., 2]
        result = torch.cat((
            state[..., 6:8] - state[..., :2],
            torch.sin(yaw)[..., None], torch.cos(yaw)[..., None],
            state[..., 3:6], state[..., 8:10], action,
        ), dim=-1)
    _finite(result, backend, "encoded output")
    return result


def state_delta(current, next_state):
    """Return physical ``[..., 10]`` endpoint changes with 2*pi yaw wrapping.

    This reports the principal yaw displacement. Endpoints cannot distinguish
    a small displacement from a >=pi rotation or extra revolutions. Interval
    validity requires the recorded substep trajectory or an independently
    justified physical bound; successful encoding is not that certification.
    An exactly pi endpoint difference maps to -pi and is not an acceptable
    learned rollout increment under integrate's strict physical-yaw limit.
    """
    current, next_state, backend = _pair(current, next_state, 10, 10, "current", "next_state")
    if backend == "numpy":
        with np.errstate(over="ignore", invalid="ignore"):
            raw_delta = next_state - current
            result = _with_yaw(raw_delta, _wrap_2pi(raw_delta[..., 2], backend), backend)
    else:
        raw_delta = next_state - current
        result = _with_yaw(raw_delta, _wrap_2pi(raw_delta[..., 2], backend), backend)
    _finite(result, backend, "delta output")
    return result


def integrate(state, physical_delta):
    """Add physical ``[..., 10]`` deltas, returning yaw wrapped to [-pi, pi).

    Reject raw predicted abs(delta_yaw) >=pi *before* any wrapping. Callers
    must denormalize model output to physical units first. Other finite deltas
    are not silently clipped, and acceptance here does not establish contact,
    workspace, speed, or full physical validity of a predicted trajectory.
    """
    state, delta, backend = _pair(state, physical_delta, 10, 10, "state", "physical_delta")
    if backend == "numpy":
        excessive_yaw = bool(np.any(np.abs(delta[..., 2]) >= math.pi))
    else:
        excessive_yaw = bool((delta[..., 2].abs() >= math.pi).any())
    if excessive_yaw:
        raise RepresentationError("Raw physical delta_yaw must satisfy abs(delta_yaw) < pi")
    if backend == "numpy":
        with np.errstate(over="ignore", invalid="ignore"):
            raw_next = state + delta
            result = _with_yaw(raw_next, _wrap_2pi(raw_next[..., 2], backend), backend)
    else:
        raw_next = state + delta
        result = _with_yaw(raw_next, _wrap_2pi(raw_next[..., 2], backend), backend)
    _finite(result, backend, "integrated output")
    return result
