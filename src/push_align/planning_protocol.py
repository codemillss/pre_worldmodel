"""Shared current-observation geometry and keyed planning randomness.

The resolved config is for the G controller only. It must not replace the
physical environment or learned-model config bound to the saved normalizer.
Proposals contain no executed action: the executor owns physical execution,
actual applied-action bookkeeping, and one context commit per actual tick.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np

from .geometry_control import GeometryContext, PARAMETER_NAMES
from .geometry_control_smooth import SmoothGeometryController, TRANSITION_SCHEMA_V2


SELECTED_PARAMETER_NAMES = frozenset((
    "position_gain_per_s", "yaw_creep_speed_m_s", "yaw_offset_gain_m_rad",
    "speed_max_m_s",
))


class PlanningProtocolError(ValueError):
    """A shared proposal, context, config or RNG key violates the contract."""


def _tick(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise PlanningProtocolError(f"{name} requires a nonnegative integer")
    return int(value)


def _readonly_vector(value, name):
    if not isinstance(value, np.ndarray) or value.dtype != np.float64 \
            or value.shape != (2,) or not np.isfinite(value).all():
        raise PlanningProtocolError(f"{name} requires a finite float64[2] array")
    result = value.copy()
    result.setflags(write=False)
    return result


def resolve_geometry_config(config, selected_parameters, transition_parameters):
    """Return a validated controller-only copy with exactly sixteen G values.

    The four selected-v2 overrides and explicit transition-v2 parameters are
    validated through the existing G implementation. No file is read and no
    source mapping is mutated. Original physical/model config stays separate.
    """
    if not isinstance(config, dict) or not isinstance(selected_parameters, Mapping) \
            or set(selected_parameters) != SELECTED_PARAMETER_NAMES:
        raise PlanningProtocolError("Geometry selection requires the four selected-v2 keys")
    if not isinstance(transition_parameters, Mapping) \
            or transition_parameters.get("schema_version") != TRANSITION_SCHEMA_V2:
        raise PlanningProtocolError("Shared geometry requires explicit transition-v2 parameters")
    try:
        resolved = deepcopy(config)
        if not isinstance(resolved.get("geometry_baseline"), dict) \
                or set(resolved["geometry_baseline"]) != set(PARAMETER_NAMES):
            raise PlanningProtocolError("Geometry baseline requires exactly sixteen registered parameters")
        resolved["geometry_baseline"].update(deepcopy(dict(selected_parameters)))
        controller = SmoothGeometryController(resolved, deepcopy(dict(transition_parameters)))
        resolved["geometry_baseline"] = deepcopy(controller.parameters)
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
        if isinstance(error, PlanningProtocolError):
            raise
        raise PlanningProtocolError(f"Invalid shared geometry configuration: {error}") from error
    return resolved


def keyed_planner_rng(base_seed, episode_index, control_tick, stream_id=0):
    """Create PCG64 from explicit keys, independent of call history or model.

    ``control_tick`` is the actual execution tick, including holding ticks.
    Skipping a draw during holding cannot shift the next tick's stream. Model
    identity and learning seed are deliberately absent from this interface.
    """
    keys = [_tick(value, name) for name, value in (
        ("base_seed", base_seed), ("episode_index", episode_index),
        ("control_tick", control_tick), ("stream_id", stream_id),
    )]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(keys)))


@dataclass(frozen=True)
class GeometryProposal:
    geometry_action: np.ndarray | None
    tangent_direction: np.ndarray
    context: GeometryContext
    control_tick: int
    diagnostics: dict


class GeometryProposalAdapter:
    """Stateless proposal adapter; callers retain episode-local G context.

    Call ``propose`` from the current observed state on every actual control
    tick, including holding. Commit its returned context once regardless of
    which MPC candidate is selected; this records G's observed-state phase
    update, not execution of its suggested command. Episode reset creates a
    fresh context. Holding uses G's existing pose-driven context reset.
    """

    def __init__(self, config, selected_parameters, transition_parameters):
        self._resolved_config = resolve_geometry_config(config, selected_parameters, transition_parameters)
        self._transition_parameters = deepcopy(dict(transition_parameters))
        self._controller = SmoothGeometryController(self._resolved_config, self._transition_parameters)

    @property
    def parameters(self):
        return deepcopy(self._controller.parameters)

    @property
    def transition_parameters(self):
        return deepcopy(self._transition_parameters)

    @property
    def resolved_config(self):
        return deepcopy(self._resolved_config)

    def initial_context(self):
        return self._controller.initial_context()

    def propose(self, state, task, context, elapsed_s, *, control_tick):
        tick = _tick(control_tick, "control_tick")
        try:
            decision = self._controller.propose(state, task, context, elapsed_s)
            feasible = decision.diagnostics["feasible"]
            if type(feasible) is not bool:
                raise PlanningProtocolError("G must provide an explicit feasibility boolean")
            # Use the existing pi-symmetric rear-face basis. Recomputing from
            # raw yaw would reverse slots 6/7 for yaw+pi equivalent poses.
            tangent = _readonly_vector(np.asarray(decision.diagnostics["tangent_direction"], dtype=np.float64),
                                       "current world tangent")
            if not np.isclose(np.linalg.norm(tangent), 1.0, rtol=0.0, atol=1e-7):
                raise PlanningProtocolError("Current world tangent must be a unit vector")
            action = _readonly_vector(decision.action, "G proposal action") if feasible else None
            diagnostics = deepcopy(decision.diagnostics)
        except (ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
            if isinstance(error, PlanningProtocolError):
                raise
            raise PlanningProtocolError(f"Invalid current-observation G proposal: {error}") from error
        diagnostics.update(
            proposal_control_tick=tick,
            geometry_action_status="available" if feasible else "unavailable_zero",
            proposal_is_executed_action=False,
            context_commit_policy="once_per_actual_tick_regardless_of_selected_command",
            tangent_basis="current_observed_pi_symmetric_G_rear_face_world_basis",
        )
        return GeometryProposal(action, tangent, decision.context, tick, diagnostics)

    def commit(self, proposal, *, actual_control_tick):
        """Return the proposed context for this tick without any action claim.

        This function does not advance physics or update the planner's actual
        applied action. The executor remains responsible for calling it once
        per actual tick and for preserving unexecuted decisions on failure.
        """
        tick = _tick(actual_control_tick, "actual_control_tick")
        if not isinstance(proposal, GeometryProposal):
            raise PlanningProtocolError("G context commit requires a GeometryProposal")
        if _tick(proposal.control_tick, "proposal_control_tick") != tick \
                or not isinstance(proposal.context, GeometryContext):
            raise PlanningProtocolError("G context commit requires the matching actual control tick")
        return replace(proposal.context)
