"""Reproducible two-block action proposals, without simulator access.

The generator consumes only its supplied PCG64 stream. It neither removes
duplicates nor draws replacement candidates. A planner support limit may be
lower than the physical actuator command limit; this is recorded explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


RESERVED_CANDIDATES = (
    "zero", "shifted_previous", "geometry_constant", "geometry_then_zero",
    "zero_then_geometry", "half_geometry", "geometry_plus_tangent",
    "geometry_minus_tangent",
)


class CandidateError(ValueError):
    """A proposal input or configuration violates the declared contract."""


@dataclass(frozen=True)
class CandidateBatch:
    actions: np.ndarray
    sources: tuple[str, ...]
    diagnostics: dict


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise CandidateError(f"{name} must be an integer")
    if value < minimum:
        raise CandidateError(f"{name} must be >= {minimum}")
    return int(value)


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise CandidateError(f"{name} must be a finite real number")
    if np.iscomplexobj(value):
        raise CandidateError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise CandidateError(f"{name} must be a finite real number") from error
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise CandidateError(f"{name} must be finite and {qualifier}")
    return result


def _array(value, shape, name):
    if not isinstance(value, np.ndarray) or value.dtype not in (
            np.dtype("float32"), np.dtype("float64")):
        raise CandidateError(f"{name} requires a float32 or float64 NumPy array")
    if value.shape != shape or not np.isfinite(value).all():
        raise CandidateError(f"{name} requires finite shape {shape}")
    return value.astype(np.float64, copy=True)


def _norm_limit(values, limit):
    """Radial projection, including finite inputs whose raw norm overflows."""
    result = values.copy()
    magnitude = np.max(np.abs(result), axis=-1)
    nonzero = magnitude > 0
    direction = np.zeros_like(result)
    direction[nonzero] = result[nonzero] / magnitude[nonzero, None]
    direction_norm = np.linalg.norm(direction, axis=-1)
    outside = nonzero & (magnitude > limit / np.maximum(direction_norm, 1.0))
    result[outside] = direction[outside] * (limit / direction_norm[outside, None])
    return result


def _disk(rng, count, speed):
    radius = speed * np.sqrt(rng.random(count))
    angle = rng.uniform(0.0, 2 * math.pi, count)
    return radius[:, None] * np.stack((np.cos(angle), np.sin(angle)), axis=-1)


def generate_candidates(config, rng: np.random.Generator, *, previous_actions=None,
                        geometry_action=None, tangent_direction=None) -> CandidateBatch:
    """Return float64 ``[N,H,2]`` two-block proposals and their source labels.

    ``previous_actions`` is shifted by one command, ends in zero, and is then
    averaged within each configured block. An enabled, available geometry
    proposal requires its current world-frame unit tangent; no future state
    is queried. Disabled/unavailable geometry leaves all six fixed geometry
    slots zero, while its random local group still explores around zero.
    """
    try:
        planner = config["planner"]
        physical_limit = _number(config["pusher"]["command_speed_norm_max_m_s"],
                                 "physical command limit", positive=True)
        speed = _number(planner.get("command_speed_norm_max_m_s", physical_limit),
                        "planner command limit", positive=True)
        count = _integer(planner["candidate_count"], "candidate_count", 8)
        horizon = _integer(planner["horizon"], "horizon", 2)
        lengths = planner["block_lengths"]
        if not isinstance(lengths, (list, tuple)) or len(lengths) != 2:
            raise CandidateError("block_lengths requires exactly two blocks")
        lengths = tuple(_integer(v, "block length", 1) for v in lengths)
        if sum(lengths) != horizon:
            raise CandidateError("block lengths must sum to horizon")
        if planner["kind"] != "random_shooting":
            raise CandidateError("only random_shooting is supported")
        if tuple(planner["reserved_candidates"]) != RESERVED_CANDIDATES:
            raise CandidateError("reserved_candidates must preserve the declared eight slots")
        enabled = planner["geometry_proposal_enabled"]
        if not isinstance(enabled, bool):
            raise CandidateError("geometry_proposal_enabled must be boolean")
        perturbation = _number(planner["geometry_tangent_perturbation_m_s"],
                               "geometry tangent perturbation")
        noise_std = _number(planner["random_local_noise_std_m_s"], "local noise std")
        rho = _number(planner["second_block_correlation"], "second block correlation")
        if rho > 1:
            raise CandidateError("second_block_correlation must be in [0,1]")
        if _number(planner["random_global_fraction"], "random global fraction") != 0.5:
            raise CandidateError("the declared global allocation is exactly one half")
        if config["reproducibility"]["rng"] != "numpy_PCG64":
            raise CandidateError("the declared RNG must be numpy_PCG64")
    except (KeyError, TypeError, AttributeError) as error:
        raise CandidateError(f"missing or malformed candidate configuration: {error}") from error
    if speed > physical_limit:
        raise CandidateError("planner command limit must not exceed the physical command limit")
    if not isinstance(rng, np.random.Generator) or type(rng.bit_generator) is not np.random.PCG64:
        raise CandidateError("rng requires a NumPy Generator using PCG64")

    # Validate every supplied input before consuming any random draws.
    previous = (None if previous_actions is None else
                _array(previous_actions, (horizon, 2), "previous_actions"))
    geometry = (None if geometry_action is None else
                _array(geometry_action, (2,), "geometry_action"))
    tangent = (None if tangent_direction is None else
               _array(tangent_direction, (2,), "tangent_direction"))
    if tangent is not None and not math.isclose(math.hypot(*tangent), 1.0,
                                                rel_tol=0.0, abs_tol=1e-7):
        raise CandidateError("tangent_direction must be a unit vector")
    available = enabled and geometry is not None
    if available and tangent is None:
        raise CandidateError("an enabled geometry proposal requires tangent_direction")

    zero = np.zeros((2, 2), dtype=np.float64)
    warm = zero.copy()
    if previous is not None:
        shifted = np.concatenate((previous[1:], np.zeros((1, 2))), axis=0)
        # Division before summation prevents overflow in finite large inputs.
        split = lengths[0]
        warm[0] = np.sum(shifted[:split] / lengths[0], axis=0)
        warm[1] = np.sum(shifted[split:] / lengths[1], axis=0)
    # Construct each declared proposal first, then project every proposal by
    # the same rule. Clipping g first would change e.g. the half-g candidate.
    g = geometry if available else np.zeros(2, dtype=np.float64)
    blocks = np.zeros((count, 2, 2), dtype=np.float64)
    blocks[1] = warm
    if available:
        blocks[2] = g
        blocks[3, 0] = g
        blocks[4, 1] = g
        blocks[5] = 0.5 * g
        with np.errstate(over="ignore", invalid="ignore"):
            blocks[6] = np.stack((g + perturbation * tangent, g))
            blocks[7] = np.stack((g - perturbation * tangent, g))

    remaining = count - 8
    global_count = remaining // 2
    local_geometry_count = remaining // 4
    local_warm_count = remaining - global_count - local_geometry_count
    cursor = 8
    v1 = _disk(rng, global_count, speed)
    w = _disk(rng, global_count, speed)
    blocks[cursor:cursor + global_count, 0] = v1
    blocks[cursor:cursor + global_count, 1] = rho * v1 + (1 - rho) * w
    cursor += global_count
    for center, size in ((np.stack((g, g)), local_geometry_count),
                         (warm, local_warm_count)):
        epsilon1 = rng.standard_normal((size, 2))
        epsilon2 = rng.standard_normal((size, 2))
        with np.errstate(over="ignore", invalid="ignore"):
            blocks[cursor:cursor + size, 0] = center[0] + noise_std * epsilon1
            blocks[cursor:cursor + size, 1] = center[1] + noise_std * (
                rho * epsilon1 + math.sqrt(1 - rho * rho) * epsilon2)
        cursor += size
    if not np.isfinite(blocks).all():
        raise CandidateError("candidate arithmetic produced nonfinite values")
    blocks = _norm_limit(blocks, speed)
    actions = np.concatenate((
        np.repeat(blocks[:, :1], lengths[0], axis=1),
        np.repeat(blocks[:, 1:], lengths[1], axis=1),
    ), axis=1)
    actions.setflags(write=False)
    sources = (RESERVED_CANDIDATES + ("random_global",) * global_count
               + ("random_local_geometry",) * local_geometry_count
               + ("random_local_warm",) * local_warm_count)
    diagnostics = {
        "candidate_count": count,
        "horizon": horizon,
        "block_lengths": list(lengths),
        "command_speed_norm_max_m_s": speed,
        "physical_command_speed_norm_max_m_s": physical_limit,
        "planner_support_limit_overridden": "command_speed_norm_max_m_s" in planner,
        "geometry_proposal_enabled": enabled,
        "geometry_proposal_status": ("available" if available else
                                     "unavailable_zero" if enabled else "disabled_zero"),
        "geometry_fixed_slots_zeroed": not available,
        "geometry_tangent_perturbation_applied": available,
        "warm_start_available": previous is not None,
        "random_group_counts": {"global": global_count,
                                "local_geometry": local_geometry_count,
                                "local_warm": local_warm_count},
        "rng": "numpy_PCG64",
        "replacement_draws": 0,
        "duplicates_removed": 0,
    }
    return CandidateBatch(actions, sources, diagnostics)
