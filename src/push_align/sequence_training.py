"""Matched real-transition loss and masked, differentiable free-state rollout.

The mask describes recorded valid transitions, never model success. Invalid
predictions fail the update instead of being removed from its denominator.
S1 uses every real transition in a window; SK adds predicted-state errors at
k=1..K. Both use the immutable train delta scales and 2*pi yaw residuals.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .dynamics import predict_next
from .representation import encode, state_delta
from .training import TrainingError, one_step_loss


@dataclass(frozen=True)
class SequenceLoss:
    one_step: torch.Tensor
    rollout: torch.Tensor | None
    total: torch.Tensor
    valid_transitions: int
    scalar_terms: int
    predicted_states: torch.Tensor | None


def _physical(value, width, ndim, name):
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float64
            or value.device.type != "cpu" or value.layout != torch.strided
            or value.ndim != ndim or value.shape[-1] != width):
        raise TrainingError(f"{name} requires dense CPU float64 with {ndim} axes and last width {width}")


def _mask(initial_state, actions, mask):
    _physical(initial_state, 10, 2, "initial_state")
    _physical(actions, 2, 3, "actions")
    batch, horizon = actions.shape[:2]
    if (batch == 0 or horizon == 0 or initial_state.shape != (batch, 10)
            or not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
            or mask.device.type != "cpu" or mask.layout != torch.strided
            or mask.shape != (batch, horizon)):
        raise TrainingError("Nonempty aligned [B,K] boolean transition mask required")
    if bool((mask[:, 1:] & ~mask[:, :-1]).any()):
        raise TrainingError("Mask must be a valid prefix; restarting after termination is forbidden")
    count = int(mask.sum())
    if count == 0:
        raise TrainingError("At least one recorded valid transition is required")
    if not bool(torch.isfinite(initial_state).all()) or not bool(torch.isfinite(actions[mask]).all()):
        raise TrainingError("Initial states and valid actions must be finite")
    if torch.is_autocast_enabled("cpu"):
        raise TrainingError("Registered sequence loss forbids CPU autocast")
    return count


def masked_free_rollout(model, normalizer, initial_state, actions, mask):
    """Predict from k0 and applied commands only, without future truth input.

    Terminated rows repeat their last prediction for trace shape only. Those
    padding slots never enter a model call or the loss. index_copy and dtype
    conversions retain the graph through every active predicted full state.
    """
    _mask(initial_state, actions, mask)
    predicted = [initial_state]
    try:
        for step in range(actions.shape[1]):
            active = torch.nonzero(mask[:, step], as_tuple=False).flatten()
            previous = predicted[-1]
            if len(active):
                next_active = predict_next(model, normalizer, previous[active], actions[active, step])
                previous = previous.index_copy(0, active, next_active)
            predicted.append(previous)
    except ValueError as error:
        raise TrainingError(str(error)) from error
    return torch.stack(predicted, dim=1)


def sequence_loss(model, normalizer, states, actions, mask, rollout_weight=0.):
    """One identical real-transition term, plus optional free-rollout term.

    states: [B,K+1,10], actions: [B,K,2], mask: [B,K]. Only the k0
    state initializes free prediction. Later real states supply labels and
    the explicitly teacher-forced one-step term. The denominator is ten
    times the number of valid transitions, excluding k0 and suffix padding.
    With weight zero no rollout graph or extra model forwards are built.
    """
    if type(rollout_weight) not in (int, float) or not math.isfinite(rollout_weight) or rollout_weight < 0:
        raise TrainingError("Rollout weight must be finite and nonnegative")
    _physical(states, 10, 3, "states")
    _physical(actions, 2, 3, "actions")
    if states.shape[:2] != (actions.shape[0], actions.shape[1] + 1):
        raise TrainingError("States must contain k0 and one endpoint per action")
    count = _mask(states[:, 0], actions, mask)
    current, target = states[:, :-1][mask], states[:, 1:][mask]
    if not bool(torch.isfinite(current).all()) or not bool(torch.isfinite(target).all()):
        raise TrainingError("Recorded valid state endpoints must be finite")
    try:
        one = one_step_loss(model, normalizer, encode(current, actions[mask]), state_delta(current, target))
        predicted = roll = None
        total = one
        if rollout_weight > 0:
            predicted = masked_free_rollout(model, normalizer, states[:, 0], actions, mask)
            residual = state_delta(target, predicted[:, 1:][mask])
            scale = torch.tensor(normalizer.arrays["delta_scale"], dtype=torch.float64)
            roll = (residual / scale).square().sum() / (count * 10)
            total = one + rollout_weight * roll
        if not bool(torch.isfinite(total)):
            raise TrainingError("Nonfinite sequence loss; failed predictions cannot be masked out")
    except ValueError as error:
        if isinstance(error, TrainingError):
            raise
        raise TrainingError(str(error)) from error
    return SequenceLoss(one, roll, total, count, count * 10, predicted)


def sequence_update(model, optimizer, normalizer, states, actions, mask, rollout_weight, clip_norm):
    """One optimizer step; diagnostics do not claim improvement/generalization."""
    if type(clip_norm) not in (int, float) or not math.isfinite(clip_norm) or clip_norm <= 0:
        raise TrainingError("Gradient clipping norm must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    terms = sequence_loss(model, normalizer, states, actions, mask, rollout_weight)
    terms.total.backward()
    parameters = list(model.parameters())
    if not parameters or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise TrainingError("Missing or nonfinite sequence gradient")
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_norm, error_if_nonfinite=True)
    if not bool(torch.isfinite(norm)):
        raise TrainingError("Nonfinite sequence gradient norm")
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in parameters):
        raise TrainingError("Nonfinite parameter after sequence optimizer update")
    return {"loss_before_update": float(terms.total.detach()),
            "one_step_loss": float(terms.one_step.detach()),
            "rollout_loss": None if terms.rollout is None else float(terms.rollout.detach()),
            "gradient_norm_before_clip": float(norm), "window_count": len(states),
            "valid_transitions": terms.valid_transitions, "scalar_terms": terms.scalar_terms}
