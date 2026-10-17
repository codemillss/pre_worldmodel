"""A single relative-XY auxiliary objective on unchanged SK trajectories.

No contact masks, geometry penalty, normalization fit, duplicate forwards,
or simulator calls. Zero weight delegates to the historical update exactly.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch

from .checkpoint import (_clone, _config_schema, _rng, _validate_optimizer,
                         _validate_weights, make_optimizer)
from .data import digest
from .data_intervention_training import checkpoint_hash, read_json, require
from .dynamics import DeltaMLP
from .fork_checkpoint import ForkCheckpointBundle
from .fork_sampler import WindowSampler
from .representation import state_delta
from .sequence_training import sequence_loss, sequence_update
from .training import TrainingError

BRANCHES = {"C0": 0., "R1": .2}
FORMULA = "Lone+0.1*Lroll+beta*(Lrelative_teacher+0.1*Lrelative_free)"


@dataclass(frozen=True)
class RelativeSequenceLoss:
    one_step: torch.Tensor
    rollout: torch.Tensor | None
    relative_teacher: torch.Tensor | None
    relative_rollout: torch.Tensor | None
    total: torch.Tensor
    predicted_states: torch.Tensor | None
    teacher_xy_errors: torch.Tensor | None
    free_xy_errors: torch.Tensor | None
    valid_transitions: int
    scalar_terms: int
    relative_scalar_terms: int


def relative_scale(normalizer):
    """Fixed RSS of original delta scales; not an estimated new variance."""
    scales = np.asarray(normalizer.arrays["delta_scale"], dtype=np.float64)
    if scales.shape != (10,) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise TrainingError("Original positive finite ten-coordinate delta scales required")
    return torch.tensor(np.sqrt(scales[:2]**2 + scales[6:8]**2), dtype=torch.float64)


def _relative(errors, scale):
    xy = torch.cat((errors[:, :2], errors[:, 6:8]), dim=1)
    residual = xy[:, 2:] - xy[:, :2]
    return (residual / scale).square().sum() / (2 * len(errors)), xy


def relative_sequence_loss(model, normalizer, states, actions, mask,
                           relative_weight=.2, rollout_weight=.1):
    if type(relative_weight) not in (int, float) or not math.isfinite(relative_weight) or relative_weight < 0:
        raise TrainingError("Relative weight must be finite and nonnegative")
    if relative_weight == 0:
        base = sequence_loss(model, normalizer, states, actions, mask, rollout_weight)
        return RelativeSequenceLoss(base.one_step, base.rollout, None, None, base.total,
                                    base.predicted_states, None, None,
                                    base.valid_transitions, base.scalar_terms, 0)
    # The first root-module call is the historical teacher forward. Capturing
    # its tensor preserves the existing loss arithmetic and shared graph.
    captured = []
    def capture(_module, _inputs, output):
        if not captured:
            captured.append(output)
    handle = model.register_forward_hook(capture)
    try:
        base = sequence_loss(model, normalizer, states, actions, mask, rollout_weight)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise TrainingError("Teacher model output was not captured exactly once")
    current, target = states[:, :-1][mask], states[:, 1:][mask]
    predicted_delta = normalizer.denormalize_delta(captured[0].to(torch.float64))
    teacher_errors = state_delta(state_delta(current, target), predicted_delta)
    scale = relative_scale(normalizer)
    teacher, teacher_xy = _relative(teacher_errors, scale)
    free = free_xy = None
    if base.predicted_states is not None:
        free_errors = state_delta(target, base.predicted_states[:, 1:][mask])
        free, free_xy = _relative(free_errors, scale)
    total = base.total + relative_weight * (teacher + (rollout_weight * free if free is not None else 0.))
    if not bool(torch.isfinite(total)):
        raise TrainingError("Nonfinite relative loss; model failures cannot change denominator")
    return RelativeSequenceLoss(base.one_step, base.rollout, teacher, free, total,
                                base.predicted_states, teacher_xy, free_xy,
                                base.valid_transitions, base.scalar_terms,
                                2 * base.valid_transitions)


def relative_sequence_update(model, optimizer, normalizer, states, actions, mask,
                             relative_weight=.2, rollout_weight=.1, clip_norm=1.):
    if relative_weight == 0 and type(relative_weight) in (int, float):
        result = sequence_update(model, optimizer, normalizer, states, actions, mask,
                                 rollout_weight, clip_norm)
        return {**result, "relative_teacher_loss": None, "relative_rollout_loss": None,
                "relative_scalar_terms": 0, "teacher_xy_errors": None, "free_xy_errors": None}
    if type(clip_norm) not in (int, float) or not math.isfinite(clip_norm) or clip_norm <= 0:
        raise TrainingError("Gradient clipping norm must be positive")
    model.train(); optimizer.zero_grad(set_to_none=True)
    terms = relative_sequence_loss(model, normalizer, states, actions, mask,
                                   relative_weight, rollout_weight)
    terms.total.backward()
    parameters = list(model.parameters())
    if not parameters or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise TrainingError("Missing or nonfinite relative-loss gradient")
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_norm, error_if_nonfinite=True)
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in parameters):
        raise TrainingError("Nonfinite parameter after relative update")
    return {"loss_before_update": float(terms.total.detach()),
            "one_step_loss": float(terms.one_step.detach()),
            "rollout_loss": None if terms.rollout is None else float(terms.rollout.detach()),
            "relative_teacher_loss": float(terms.relative_teacher.detach()),
            "relative_rollout_loss": None if terms.relative_rollout is None else float(terms.relative_rollout.detach()),
            "gradient_norm_before_clip": float(norm), "window_count": len(states),
            "valid_transitions": terms.valid_transitions, "scalar_terms": terms.scalar_terms,
            "relative_scalar_terms": terms.relative_scalar_terms,
            "teacher_xy_errors": terms.teacher_xy_errors.detach().numpy().copy(),
            "free_xy_errors": None if terms.free_xy_errors is None else terms.free_xy_errors.detach().numpy().copy()}


def save_relative_checkpoint(path, model, optimizer, normalizer, config, *, branch,
                             sampler, manifest_sha256, dataset_hash, initial_checkpoint_hash):
    require(branch in BRANCHES and sampler.updates in range(100,1001,100), "Unregistered relative checkpoint")
    require(sampler.window_count == 2960 and sampler.batch_windows == 64
            and sampler.max_updates == 1000 and sampler.horizon == 5, "Sampler budget differs")
    _config_schema(config, normalizer)
    weights, state = _clone(model.state_dict()), {"optimizer": _clone(optimizer.state_dict()),
        "sampler": sampler.state_dict(), "torch_rng_state": torch.get_rng_state().clone()}
    _validate_weights(weights, model); _validate_optimizer(state["optimizer"], model, config,1300+sampler.updates)
    _rng(state["torch_rng_state"])
    path=Path(path); path.mkdir(parents=True, exist_ok=False)
    for name,value in (("weights.pt",weights),("training_state.pt",state)):
        with (path/name).open("xb") as stream: torch.save(value,stream)
    meta={"format":"push-align-relative-loss-checkpoint-v1", "branch":branch,
          "relative_weight":BRANCHES[branch], "rollout_weight":.1, "formula":FORMULA,
          "config":deepcopy(config), "normalizer_hash":normalizer.metadata["normalizer_hash"],
          "relative_scale_m":relative_scale(normalizer).tolist(), "manifest_sha256":manifest_sha256,
          "dataset_hash":dataset_hash, "initial_checkpoint_hash":initial_checkpoint_hash,"seed":11,
          "additional_updates":sampler.updates,"total_updates":1300+sampler.updates,
          "files":{name:digest(path/name) for name in ("weights.pt","training_state.pt")}}
    meta["checkpoint_hash"]=checkpoint_hash(meta)
    import json
    (path/"checkpoint.json").write_text(json.dumps(meta,indent=2,allow_nan=False)+"\n")
    return meta


def load_relative_checkpoint(path, normalizer, config, *, expected_manifest_sha256=None,
                             expected_branch=None, expected_dataset_hash=None,
                             expected_initial_checkpoint_hash=None, resume=False):
    path=Path(path);meta=read_json(path/"checkpoint.json")
    require(meta["format"]=="push-align-relative-loss-checkpoint-v1"
            and meta["checkpoint_hash"]==checkpoint_hash(meta), "Relative checkpoint integrity differs")
    require(meta["branch"] in BRANCHES and meta["relative_weight"]==BRANCHES[meta["branch"]]
            and meta["rollout_weight"]==.1 and meta["formula"]==FORMULA and meta["seed"]==11
            and meta["additional_updates"] in range(100,1001,100)
            and meta["total_updates"]==1300+meta["additional_updates"], "Relative objective/budget differs")
    require(meta["config"]==config and meta["normalizer_hash"]==normalizer.metadata["normalizer_hash"]
            and meta["relative_scale_m"]==relative_scale(normalizer).tolist(), "Config/fixed scales differ")
    for key,value in (("manifest_sha256",expected_manifest_sha256),("branch",expected_branch),
                      ("dataset_hash",expected_dataset_hash),("initial_checkpoint_hash",expected_initial_checkpoint_hash)):
        if value is not None: require(meta[key]==value,"Checkpoint binding differs: "+key)
    for name in ("weights.pt","training_state.pt"):
        require(digest(path/name)==meta["files"][name],"Checkpoint tensor artifact differs")
    _config_schema(config,normalizer)
    with torch.random.fork_rng(devices=[]): model=DeltaMLP(config)
    weights=torch.load(path/"weights.pt",map_location="cpu",weights_only=True)
    state=torch.load(path/"training_state.pt",map_location="cpu",weights_only=True)
    require(set(state)=={"optimizer","sampler","torch_rng_state"}, "Unexpected continuation state")
    _validate_weights(weights,model);model.load_state_dict(weights);model.eval()
    _validate_optimizer(state["optimizer"],model,config,meta["total_updates"]);_rng(state["torch_rng_state"])
    sampler=WindowSampler.from_state_dict(state["sampler"])
    require(sampler.updates==meta["additional_updates"] and sampler.window_count==2960
            and sampler.batch_windows==64 and sampler.max_updates==1000 and sampler.horizon==5,"Sampler state differs")
    optimizer=None
    if resume:
        optimizer=make_optimizer(model,config);optimizer.load_state_dict(_clone(state["optimizer"]))
    return ForkCheckpointBundle(model,meta,optimizer,sampler if resume else None,
                                state["torch_rng_state"].clone() if resume else None)
