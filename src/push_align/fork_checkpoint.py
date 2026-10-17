"""P4 matched-window checkpoint boundary, separate from P3 provenance.

Inference weights and complete continuation state are distinct safe-load
files. A checkpoint binds the frozen budget, seed-specific common origin,
actual Adam step and actual mid-epoch window stream; loading does not mutate
the caller's RNG. Hashes detect accidental changes, not an authenticated
signature against someone who can replace all expected provenance.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path, PurePosixPath
import pickle
from zipfile import BadZipFile

import torch

from .checkpoint import (CheckpointError, _clone, _config_schema, _rng,
                         _validate_optimizer, _validate_weights, make_optimizer)
from .data import _json_bytes, _read_json, _write_json, digest
from .dynamics import DeltaMLP, model_spec


@dataclass(frozen=True)
class ForkCheckpointBundle:
    model: DeltaMLP
    metadata: dict
    optimizer: object = None
    sampler: object = None
    torch_rng_state: object = None


def _finite_json(value):
    """Canonical JSON preserves numeric types and rejects coercible objects."""
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise CheckpointError("P4 bindings require string JSON keys")
        return {key: _finite_json(item) for key, item in value.items()}
    if type(value) is list:
        return [_finite_json(item) for item in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise CheckpointError("P4 bindings require finite JSON values")


def _canonical(value):
    return _json_bytes(_finite_json(value))


def _sha(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _hash(metadata):
    return _sha({key: value for key, value in metadata.items()
                 if key != "checkpoint_hash"})


def _hash_string(value, name):
    if (type(value) is not str or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise CheckpointError(f"Invalid {name} SHA256")


def _counter(value, name, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise CheckpointError(f"Invalid {name} counter")


def _bindings(normalizer, config, provenance, budget, seed, branch):
    normalizer.validate()
    _config_schema(config, normalizer)
    expected = {"learning_manifest_sha256", "dataset_identity",
                "validation_split", "task_scope"}
    if (not isinstance(provenance, Mapping) or set(provenance) != expected
            or provenance["task_scope"] != "P4-02-matched-fork"
            or provenance["validation_split"] != "validation"
            or _canonical(provenance["dataset_identity"]) !=
            _canonical(normalizer.metadata["dataset_identity"])):
        raise CheckpointError("Invalid P4 learning/validation provenance")
    _hash_string(provenance["learning_manifest_sha256"], "learning manifest")
    if not isinstance(budget, Mapping) or not budget:
        raise CheckpointError("P4 frozen fork budget is required")
    _canonical(budget)
    _counter(seed, "seed")
    seeds = budget["seeds"]
    if (type(seeds) is not list or not seeds
            or any(type(value) is not int or value < 0 for value in seeds)
            or len(set(seeds)) != len(seeds) or seed not in seeds):
        raise CheckpointError("Seed is outside the frozen fork budget")
    for key in ("additional_updates_per_branch", "batch_windows", "K",
                "window_count_train", "window_count_validation",
                "checkpoint_every_updates"):
        _counter(budget[key], key, positive=True)
    if budget["drop_last"] is not False:
        raise CheckpointError("P4 window stream must retain the last partial batch")
    common = budget["common_checkpoints"]
    if (type(common) is not list or len(common) != len(seeds)
            or any(not isinstance(item, dict)
                   or set(item) != {"seed", "path", "checkpoint_hash", "initial_updates"}
                   for item in common)):
        raise CheckpointError("Invalid seed-specific common origins")
    for item in common:
        _counter(item["seed"], "common seed")
        _counter(item["initial_updates"], "initial updates")
        _hash_string(item["checkpoint_hash"], "common checkpoint")
        path = item["path"]
        if (type(path) is not str or not path or "\\" in path
                or PurePosixPath(path).is_absolute()
                or any(part in (".", "..") for part in path.split("/"))):
            raise CheckpointError("Common checkpoint path must be a relative stable identity")
    if sorted(item["seed"] for item in common) != sorted(seeds):
        raise CheckpointError("Common origins do not cover each seed exactly once")
    origin = next(item for item in common if item["seed"] == seed)
    lambdas = budget["lambda_candidates"]
    if (type(lambdas) is not list or not lambdas
            or any(type(value) is not float or not math.isfinite(value)
                   or value <= 0 for value in lambdas)
            or len(set(lambdas)) != len(lambdas)):
        raise CheckpointError("Invalid positive rollout weights")
    names = ["S1", *(f"SK-lambda{value}" for value in lambdas)]
    if (_canonical(budget["branches_per_seed"]) != _canonical(names)
            or type(branch) is not str or branch not in names):
        raise CheckpointError("Branch is outside the frozen fork budget")
    weight = 0.0 if branch == "S1" else lambdas[names.index(branch) - 1]
    return origin, weight


def _sampler_state(sampler, budget, additional_updates):
    from .fork_sampler import WindowSampler
    if not isinstance(sampler, WindowSampler):
        raise CheckpointError("P4 checkpoint requires its actual WindowSampler")
    state = sampler.state_dict()
    # The sampler validates permutation, cursor, counters and RNG itself.
    checked = WindowSampler.from_state_dict(state)
    expected = {"window_count": budget["window_count_train"],
                "batch_windows": budget["batch_windows"],
                "max_updates": budget["additional_updates_per_branch"],
                "horizon": budget["K"], "drop_last": False,
                "updates": additional_updates}
    for key, value in expected.items():
        actual = getattr(checked, key)
        if type(actual) is not type(value) or actual != value:
            raise CheckpointError(f"Sampler {key} differs from P4 budget/counter")
    return state


def _optimizer_state(state, model, config, updates):
    _validate_optimizer(state, model, config, updates)
    # Adam constructs scalar float32 CPU steps, regardless of model casts.
    for entry in state["state"].values():
        step = entry["step"]
        if step.dtype != torch.float32 or step.device.type != "cpu" or step.shape != ():
            raise CheckpointError("Adam step must be a scalar CPU float32 counter")


def save_fork_checkpoint(path, model, optimizer, normalizer, config,
                         training_provenance, fork_budget, *, seed, branch,
                         additional_updates, sampler, run_manifest_sha256):
    """Save an immutable new checkpoint after a registered branch update."""
    try:
        origin, weight = _bindings(normalizer, config, training_provenance,
                                   fork_budget, seed, branch)
        _hash_string(run_manifest_sha256, "run manifest")
        _counter(additional_updates, "additional updates")
        if additional_updates > fork_budget["additional_updates_per_branch"]:
            raise CheckpointError("Additional updates exceed the frozen budget")
        sampler_state = _sampler_state(sampler, fork_budget, additional_updates)
        actual = [parameter for group in optimizer.param_groups
                  for parameter in group["params"]]
        expected = list(model.parameters())
        if len(actual) != len(expected) or any(a is not b for a, b in zip(actual, expected)):
            raise CheckpointError("Optimizer parameters belong to another model/order")
        weights = _clone(model.state_dict())
        with torch.random.fork_rng(devices=[]):
            expected_model = DeltaMLP(config)
        _validate_weights(weights, expected_model)
        optimizer_state = _clone(optimizer.state_dict())
        total_updates = origin["initial_updates"] + additional_updates
        _optimizer_state(optimizer_state, model, config, total_updates)
        state = {"optimizer": optimizer_state,
                 "torch_rng_state": torch.get_rng_state().clone(),
                 "sampler": _clone(sampler_state)}
        _rng(state["torch_rng_state"])
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        for name, value in (("weights.pt", weights), ("training_state.pt", state)):
            with (path / name).open("xb") as file:
                torch.save(value, file)
        metadata = {"format": "push-align-fork-checkpoint-v1",
                    "config": deepcopy(config), "model_spec": model_spec(config),
                    "learning_schema": deepcopy(normalizer.metadata["schema"]),
                    "normalizer_hash": normalizer.metadata["normalizer_hash"],
                    "run_manifest_sha256": run_manifest_sha256,
                    "training_provenance": deepcopy(training_provenance),
                    "fork_budget": deepcopy(fork_budget),
                    "fork_budget_sha256": _sha(fork_budget),
                    "common_origin": deepcopy(origin), "seed": seed,
                    "branch": branch, "rollout_weight": weight,
                    "additional_updates": additional_updates,
                    "total_updates": total_updates,
                    "torch_version": str(torch.__version__),
                    "files": {name: digest(path / name)
                              for name in ("weights.pt", "training_state.pt")}}
        metadata["checkpoint_hash"] = _hash(metadata)
        _write_json(path / "checkpoint.json", metadata)
        return metadata
    except (KeyError, TypeError, OSError, ValueError, RuntimeError, AttributeError) as error:
        if isinstance(error, CheckpointError):
            raise
        raise CheckpointError(f"Invalid P4 checkpoint artifact: {error}") from error


def load_fork_checkpoint(path, normalizer, config, training_provenance,
                         fork_budget, *, seed, branch, resume=False,
                         expected_checkpoint_hash=None,
                         expected_run_manifest_sha256=None):
    """Validate even inference continuation files; resume returns owned copies."""
    try:
        origin, weight = _bindings(normalizer, config, training_provenance,
                                   fork_budget, seed, branch)
        if type(resume) is not bool:
            raise CheckpointError("resume must be boolean")
        if expected_checkpoint_hash is not None:
            _hash_string(expected_checkpoint_hash, "expected checkpoint")
        if expected_run_manifest_sha256 is not None:
            _hash_string(expected_run_manifest_sha256, "expected run manifest")
        path = Path(path)
        metadata = _read_json(path / "checkpoint.json")
        fields = {"format", "config", "model_spec", "learning_schema",
                  "normalizer_hash", "training_provenance", "fork_budget",
                  "fork_budget_sha256", "common_origin", "seed", "branch",
                  "rollout_weight", "additional_updates", "total_updates",
                  "torch_version", "files", "checkpoint_hash", "run_manifest_sha256"}
        if not isinstance(metadata, dict) or set(metadata) != fields:
            raise CheckpointError("Unexpected P4 checkpoint metadata fields")
        _hash_string(metadata["checkpoint_hash"], "checkpoint")
        _hash_string(metadata["run_manifest_sha256"], "run manifest")
        if (metadata["format"] != "push-align-fork-checkpoint-v1"
                or metadata["checkpoint_hash"] != _hash(metadata)
                or metadata["torch_version"] != str(torch.__version__)
                or (expected_checkpoint_hash is not None
                    and metadata["checkpoint_hash"] != expected_checkpoint_hash)
                or (expected_run_manifest_sha256 is not None
                    and metadata["run_manifest_sha256"] != expected_run_manifest_sha256)):
            raise CheckpointError("P4 checkpoint format/hash/runtime mismatch")
        expected = {"config": config, "model_spec": model_spec(config),
                    "learning_schema": normalizer.metadata["schema"],
                    "normalizer_hash": normalizer.metadata["normalizer_hash"],
                    "training_provenance": training_provenance,
                    "fork_budget": fork_budget, "fork_budget_sha256": _sha(fork_budget),
                    "common_origin": origin, "seed": seed, "branch": branch,
                    "rollout_weight": weight}
        if any(_canonical(metadata[key]) != _canonical(value)
               for key, value in expected.items()):
            raise CheckpointError("P4 config/normalizer/provenance/origin/budget/branch mismatch")
        _counter(metadata["additional_updates"], "additional updates")
        _counter(metadata["total_updates"], "total updates")
        if (metadata["additional_updates"] > fork_budget["additional_updates_per_branch"]
                or metadata["total_updates"] !=
                origin["initial_updates"] + metadata["additional_updates"]):
            raise CheckpointError("P4 actual update counters do not match the common origin")
        if (not isinstance(metadata["files"], dict)
                or set(metadata["files"]) != {"weights.pt", "training_state.pt"}):
            raise CheckpointError("P4 checkpoint file inventory mismatch")
        for name, sha in metadata["files"].items():
            _hash_string(sha, name)
            if digest(path / name) != sha:
                raise CheckpointError("P4 checkpoint file hash mismatch")
        with (path / "weights.pt").open("rb") as file:
            weights = torch.load(file, map_location="cpu", weights_only=True)
        with torch.random.fork_rng(devices=[]):
            model = DeltaMLP(config)
        _validate_weights(weights, model)
        model.load_state_dict(weights, strict=True)
        model.eval()
        with (path / "training_state.pt").open("rb") as file:
            state = torch.load(file, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or set(state) != {"optimizer", "torch_rng_state", "sampler"}:
            raise CheckpointError("Unexpected P4 training state fields")
        _optimizer_state(state["optimizer"], model, config, metadata["total_updates"])
        _rng(state["torch_rng_state"])
        from .fork_sampler import WindowSampler
        sampler = WindowSampler.from_state_dict(state["sampler"])
        _sampler_state(sampler, fork_budget, metadata["additional_updates"])
        if not resume:
            return ForkCheckpointBundle(model, metadata)
        optimizer = make_optimizer(model, config)
        optimizer.load_state_dict(state["optimizer"])
        return ForkCheckpointBundle(model, metadata, optimizer, sampler,
                                    state["torch_rng_state"].clone())
    except (KeyError, TypeError, OSError, ValueError, RuntimeError, AttributeError,
            pickle.UnpicklingError, BadZipFile, EOFError) as error:
        if isinstance(error, CheckpointError):
            raise
        raise CheckpointError(f"Invalid P4 checkpoint artifact: {error}") from error
