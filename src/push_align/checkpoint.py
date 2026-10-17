"""Local inference weights and separately bound, complete Adam resume state."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
from zipfile import BadZipFile

import torch

from .data import _json_bytes, _read_json, _write_json, digest
from .dynamics import DeltaMLP, model_spec
from .config import scene_xml


class CheckpointError(ValueError):
    """A checkpoint's tensors, optimizer or provenance is incompatible."""


@dataclass(frozen=True)
class CheckpointBundle:
    model: DeltaMLP
    metadata: dict
    optimizer: object = None
    shuffle_generator: object = None
    torch_rng_state: object = None


def _hash(metadata):
    return hashlib.sha256(_json_bytes({k: v for k, v in metadata.items() if k != "checkpoint_hash"})).hexdigest()


def make_optimizer(model, config):
    c = config["training"]
    if c["optimizer"] != "adam" or not math.isfinite(c["learning_rate"]) or c["learning_rate"] <= 0 or not math.isfinite(c["weight_decay"]) or c["weight_decay"] < 0:
        raise CheckpointError("Invalid registered Adam configuration")
    return torch.optim.Adam(model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"])


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _clone(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone(v) for v in value)
    return deepcopy(value)


def _rng(value):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1 or value.device.type != "cpu":
        raise CheckpointError("Invalid CPU RNG tensor")
    generator = torch.Generator(device="cpu")
    generator.set_state(value)
    return generator


def _validate_optimizer(state, model, config, updates):
    baseline = make_optimizer(model, config).state_dict()
    if set(state) != {"state", "param_groups"} or state["param_groups"] != baseline["param_groups"]:
        raise CheckpointError("Adam parameter groups differ from registered configuration")
    parameters = list(model.parameters())
    expected = set(range(len(parameters))) if updates else set()
    if set(state["state"]) != expected:
        raise CheckpointError("Adam state does not cover every trained parameter")
    for index, entry in state["state"].items():
        if set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
            raise CheckpointError("Unexpected Adam state fields")
        step = entry["step"]
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or not torch.isfinite(step).all() or float(step) != updates:
            raise CheckpointError("Adam step differs from checkpoint update count")
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = entry[name]
            if not isinstance(tensor, torch.Tensor) or tensor.shape != parameters[index].shape or tensor.dtype != torch.float32 or tensor.device.type != "cpu" or not torch.isfinite(tensor).all():
                raise CheckpointError("Invalid Adam moment tensor")
        if (entry["exp_avg_sq"] < 0).any():
            raise CheckpointError("Negative Adam squared moment")


def _provenance(value, normalizer):
    expected = {"learning_manifest_sha256", "dataset_identity", "validation_split", "task_scope"}
    if not isinstance(value, Mapping) or set(value) != expected or value["dataset_identity"] != normalizer.metadata["dataset_identity"] or value["validation_split"] != "validation" or value["task_scope"] not in ("P2-04-pilot", "P3-04-main-common"):
        raise CheckpointError("Invalid learning/validation provenance")
    sha = value["learning_manifest_sha256"]
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise CheckpointError("Invalid prepared dataset manifest hash")


def _bound_training_settings(training_provenance, value):
    """Bind P3 settings without changing legacy P2 checkpoint metadata.

    The runner owns the registered keys and training budget. This boundary
    requires a nonempty, finite JSON mapping and preserves JSON numeric types
    in its canonical hash; dictionary equality alone treats 1 and 1.0 alike.
    """
    if training_provenance["task_scope"] == "P2-04-pilot":
        if value is not None:
            raise CheckpointError("P2 pilot checkpoints cannot contain P3 training settings")
        return None, None
    if not isinstance(value, Mapping) or not value:
        raise CheckpointError("P3 common checkpoints require a nonempty training settings mapping")

    def json_value(item):
        if isinstance(item, Mapping):
            if any(type(key) is not str for key in item):
                raise CheckpointError("Training settings require string JSON keys")
            return {key: json_value(child) for key, child in item.items()}
        if type(item) is list:
            return [json_value(child) for child in item]
        if item is None or type(item) in (str, bool, int):
            return item
        if type(item) is float and math.isfinite(item):
            return item
        raise CheckpointError("Training settings require finite JSON values")

    try:
        settings = json_value(value)
        sha = hashlib.sha256(_json_bytes(settings)).hexdigest()
    except (ValueError, TypeError, OverflowError, RecursionError) as error:
        if isinstance(error, CheckpointError):
            raise
        raise CheckpointError("Invalid JSON training settings") from error
    return settings, sha


def _config_schema(config, normalizer):
    schema = normalizer.metadata["schema"]
    for key in ("state_fields", "state_units", "feature_fields", "delta_fields"):
        if config["schema"][key] != schema[key]:
            raise CheckpointError("Checkpoint config columns differ from learning schema")
    if config["physics"]["action_dt_s"] != schema["action_dt_s"]:
        raise CheckpointError("Checkpoint action duration differs from learning schema")
    model_hash = hashlib.sha256(scene_xml(config).encode()).hexdigest()
    environment_hash = hashlib.sha256((model_hash+json.dumps(config, sort_keys=True)).encode()).hexdigest()
    if environment_hash != schema["environment_hash"]:
        raise CheckpointError("Full config/scene differs from the normalization environment")


def save_checkpoint(path, model, optimizer, normalizer, config, training_provenance, *, epoch, updates, seed, shuffle_generator, selection_score=None, training_settings=None):
    normalizer.validate(); _provenance(training_provenance, normalizer); _config_schema(config, normalizer)
    settings, settings_sha = _bound_training_settings(training_provenance, training_settings)
    if type(epoch) is not int or epoch < 0 or type(updates) is not int or updates < 0 or type(seed) is not int or seed < 0:
        raise CheckpointError("Invalid checkpoint counters/seed")
    if selection_score is not None and (not math.isfinite(selection_score) or selection_score < 0):
        raise CheckpointError("Invalid selection score")
    expected_spec = model_spec(config)
    actual_parameters = [p for group in optimizer.param_groups for p in group["params"]]
    expected_parameters = list(model.parameters())
    if len(actual_parameters) != len(expected_parameters) or any(a is not b for a, b in zip(actual_parameters, expected_parameters)):
        raise CheckpointError("Optimizer parameters belong to a different model/order")
    weights = _clone(model.state_dict())
    with torch.random.fork_rng(devices=[]):
        expected_model = DeltaMLP(config)
    _validate_weights(weights, expected_model)
    optimizer_state = _clone(optimizer.state_dict())
    _validate_optimizer(optimizer_state, model, config, updates)
    training_state = {"optimizer": optimizer_state, "torch_rng_state": torch.get_rng_state().clone(),
                      "shuffle_rng_state": shuffle_generator.get_state().clone()}
    _rng(training_state["torch_rng_state"]); _rng(training_state["shuffle_rng_state"])
    path = Path(path); path.mkdir(parents=True, exist_ok=False)
    for name, value in (("weights.pt", weights), ("training_state.pt", training_state)):
        with (path/name).open("xb") as file:
            torch.save(value, file)
    metadata = {"format": "push-align-checkpoint-v1", "config": deepcopy(config), "model_spec": expected_spec,
                "learning_schema": deepcopy(normalizer.metadata["schema"]), "normalizer_hash": normalizer.metadata["normalizer_hash"],
                "training_provenance": deepcopy(training_provenance), "epoch": epoch, "updates": updates, "seed": seed,
                "selection_score": selection_score, "torch_version": str(torch.__version__),
                "files": {name: digest(path/name) for name in ("weights.pt", "training_state.pt")}}
    if settings is not None:
        metadata["training_settings"] = settings
        metadata["training_settings_sha256"] = settings_sha
    metadata["checkpoint_hash"] = _hash(metadata)
    _write_json(path/"checkpoint.json", metadata)
    return metadata


def _validate_weights(weights, model):
    expected = model.state_dict()
    if set(weights) != set(expected):
        raise CheckpointError("Unexpected model state fields")
    for key, tensor in weights.items():
        if not isinstance(tensor, torch.Tensor) or tensor.shape != expected[key].shape or tensor.dtype != torch.float32 or tensor.device.type != "cpu" or not torch.isfinite(tensor).all():
            raise CheckpointError("Model tensor dtype/shape/finite mismatch")


def load_checkpoint(path, normalizer, config, training_provenance, *, resume=False, training_settings=None):
    path = Path(path)
    try:
        normalizer.validate(); _provenance(training_provenance, normalizer); _config_schema(config, normalizer)
        settings, settings_sha = _bound_training_settings(training_provenance, training_settings)
        m = _read_json(path/"checkpoint.json")
        if m["format"] != "push-align-checkpoint-v1" or m["checkpoint_hash"] != _hash(m) or m["torch_version"] != str(torch.__version__):
            raise CheckpointError("Checkpoint format/hash/runtime mismatch")
        if m["config"] != config or m["model_spec"] != model_spec(config) or m["learning_schema"] != normalizer.metadata["schema"] or m["normalizer_hash"] != normalizer.metadata["normalizer_hash"] or m["training_provenance"] != training_provenance:
            raise CheckpointError("Checkpoint schema/config/normalizer/provenance mismatch")
        if settings is None:
            if "training_settings" in m or "training_settings_sha256" in m:
                raise CheckpointError("P2 pilot metadata cannot contain P3 training settings")
        else:
            stored_settings, stored_sha = _bound_training_settings(training_provenance, m["training_settings"])
            if stored_settings != settings or stored_sha != settings_sha or m["training_settings_sha256"] != settings_sha:
                raise CheckpointError("Checkpoint training settings/hash mismatch")
        if type(m["epoch"]) is not int or m["epoch"] < 0 or type(m["updates"]) is not int or m["updates"] < 0 or type(m["seed"]) is not int or m["seed"] < 0 or (m["selection_score"] is not None and (not math.isfinite(m["selection_score"]) or m["selection_score"] < 0)):
            raise CheckpointError("Invalid checkpoint counters/score")
        if set(m["files"]) != {"weights.pt", "training_state.pt"} or any(digest(path/name) != sha for name, sha in m["files"].items()):
            raise CheckpointError("Checkpoint file hash mismatch")
        with (path/"weights.pt").open("rb") as file:
            weights = torch.load(file, map_location="cpu", weights_only=True)
        with torch.random.fork_rng(devices=[]):
            model = DeltaMLP(config)
        _validate_weights(weights, model); model.load_state_dict(weights, strict=True); model.eval()
        if not resume:
            return CheckpointBundle(model, m)
        with (path/"training_state.pt").open("rb") as file:
            state = torch.load(file, map_location="cpu", weights_only=True)
        if set(state) != {"optimizer", "torch_rng_state", "shuffle_rng_state"}:
            raise CheckpointError("Unexpected training state fields")
        _validate_optimizer(state["optimizer"], model, config, m["updates"])
        _rng(state["torch_rng_state"]); generator = _rng(state["shuffle_rng_state"])
        optimizer = make_optimizer(model, config); optimizer.load_state_dict(state["optimizer"])
        return CheckpointBundle(model, m, optimizer, generator, state["torch_rng_state"].clone())
    except (KeyError, TypeError, OSError, ValueError, RuntimeError, pickle.UnpicklingError, BadZipFile, EOFError) as error:
        if isinstance(error, CheckpointError):
            raise
        raise CheckpointError(f"Invalid checkpoint artifact: {error}") from error
