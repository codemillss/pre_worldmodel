"""One-step angular-aware optimization helpers; no validation gradients."""
from __future__ import annotations

import math

import numpy as np
import torch

from .representation import state_delta


class TrainingError(ValueError):
    """Invalid optimization input or a nonfinite update."""


def one_step_loss(model, normalizer, features, physical_targets):
    if not isinstance(features, torch.Tensor) or not isinstance(physical_targets, torch.Tensor) or features.dtype != torch.float64 or physical_targets.dtype != torch.float64 or features.ndim != 2 or features.shape[1] != 11 or physical_targets.shape != (len(features), 10) or len(features) == 0 or features.device.type != "cpu" or physical_targets.device.type != "cpu":
        raise TrainingError("Loss requires nonempty CPU float64 feature/physical-target batches")
    if torch.is_autocast_enabled("cpu"):
        raise TrainingError("Registered loss forbids CPU autocast")
    try:
        normalized = normalizer.normalize_features(features).to(torch.float32)
        output = model(normalized)
        if not isinstance(output, torch.Tensor) or output.dtype != torch.float32 or output.device.type != "cpu" or output.shape != physical_targets.shape or not torch.isfinite(output).all():
            raise TrainingError("Normalized model output violates the float32 shape/finite boundary")
        prediction = normalizer.denormalize_delta(output.to(torch.float64))
        errors = state_delta(physical_targets, prediction)
    except ValueError as error:
        if isinstance(error, TrainingError):
            raise
        raise TrainingError(str(error)) from error
    scale = torch.tensor(normalizer.arrays["delta_scale"], dtype=torch.float64)
    loss = ((errors/scale)**2).mean()
    if not torch.isfinite(loss):
        raise TrainingError("Nonfinite one-step loss")
    return loss


def training_update(model, optimizer, normalizer, features, physical_targets, clip_norm):
    if type(clip_norm) not in (int, float) or not math.isfinite(clip_norm) or clip_norm <= 0:
        raise TrainingError("Gradient clipping norm must be positive")
    model.train(); optimizer.zero_grad(set_to_none=True)
    loss = one_step_loss(model, normalizer, features, physical_targets)
    loss.backward()
    gradients = [p.grad for p in model.parameters()]
    if any(g is None or not torch.isfinite(g).all() for g in gradients):
        raise TrainingError("Missing or nonfinite gradient")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise TrainingError("Nonfinite parameter after optimizer update")
    return {"loss_before_update": float(loss.detach()), "gradient_norm_before_clip": float(norm), "batch_size": len(features)}


def select_tiny_subset(prepared, count=16):
    """Eight largest train-contact translations then eight remaining rotations.

    This is a deliberate optimization probe, not a representative sample.
    Ties use original episode/transition identity, never held-out performance.
    """
    if type(count) is not int or count != 16:
        raise TrainingError("Registered tiny probe uses exactly 16 transitions")
    a = prepared.arrays
    n = len(a["deltas"])
    if a["deltas"].shape != (n, 10) or a["deltas"].dtype != np.float64 or not np.isfinite(a["deltas"]).all() or any(a[k].shape != (n,) for k in ("split_code", "regime_code", "episode_index", "transition_index")):
        raise TrainingError("Invalid tiny-probe data columns")
    identities = np.column_stack((a["episode_index"], a["transition_index"]))
    if len(np.unique(identities, axis=0)) != n:
        raise TrainingError("Repeated original transition identity in tiny-probe input")
    candidates = np.flatnonzero((a["split_code"] == 0) & (a["regime_code"] != 0))
    movement = np.linalg.norm(a["deltas"][:, :2], axis=1)
    rotation = np.abs(a["deltas"][:, 2])
    def ranked(rows, signal):
        return sorted(rows.tolist(), key=lambda i: (-signal[i], int(a["episode_index"][i]), int(a["transition_index"][i])))
    translation = ranked(candidates[movement[candidates] > 0], movement)[:8]
    remaining = np.array([i for i in candidates if i not in translation and rotation[i] > 0], dtype=np.int64)
    rotations = ranked(remaining, rotation)[:8]
    if len(translation) != 8 or len(rotations) != 8:
        raise TrainingError("Insufficient nonstationary train-contact probe rows")
    return np.array(translation+rotations, dtype=np.int64)
