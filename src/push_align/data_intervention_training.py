"""Frozen core-only slot replacement and data-bound continuation checkpoints.

The old population remains 2,960 K5 windows. Intervention arms replace only
the 384 pre-registered slots; development families and warmups never enter it.
No physics or fitted normalization is performed in this module.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint import (_clone, _config_schema, _rng, _validate_optimizer,
                         _validate_weights, make_optimizer)
from .data import _json_bytes, digest
from .dynamics import DeltaMLP
from .fork_checkpoint import ForkCheckpointBundle
from .fork_sampler import WindowSampler

ARMS = ("R", "D00", "D10", "D01", "D11")
ARM_CONDITIONS = {"D00": ("Y0", "S0"), "D10": ("Y1", "S0"),
                  "D01": ("Y0", "S1"), "D11": ("Y1", "S1")}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read_json(path):
    return json.loads(Path(path).read_text(),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def json_hash(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def tree_hash(value):
    """Same typed tensor identity as the historical P4 runner."""
    h = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            h.update(b"tensor"); h.update(str(item.dtype).encode())
            h.update(json.dumps(list(item.shape)).encode())
            h.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            h.update(b"dict")
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                visit(key); visit(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(type(item).__name__.encode())
            for child in item: visit(child)
        else:
            h.update(type(item).__name__.encode())
            h.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
    visit(value)
    return h.hexdigest()


def data_hash(states, actions, mask):
    return tree_hash({"states": states, "applied_actions": actions, "mask": mask})


def checked_slots(bindings, old_rows, positions, offsets):
    entries = bindings["entries"]
    require(len(entries) == 384 and old_rows.shape == (2960, 5), "Frozen population changed")
    slots = np.array([e["slot"] for e in entries], dtype=np.int64)
    require(np.array_equal(slots, np.unique(slots)) and slots.min() >= 0 and slots.max() < 2960,
            "Replacement slots must be sorted unique valid positions")
    require(positions.dtype == np.int64 and positions.shape == (62992,)
            and offsets.dtype == np.int64 and offsets.shape == (1001,), "Historical schedule dimensions changed")
    require(offsets[0] == 0 and offsets[-1] == 62992
            and int((np.diff(offsets) == 64).sum()) == 979
            and int((np.diff(offsets) == 16).sum()) == 21, "Historical partial-batch exposure changed")
    counts = np.bincount(positions, minlength=2960)
    partial = np.concatenate([positions[a:b] for a, b in zip(offsets[:-1], offsets[1:]) if b-a == 16])
    require(np.all(counts[slots] == 21) and not np.isin(slots, partial).any(), "New slots changed exposure")
    for index, entry in enumerate(entries):
        require(entry["family_rank"] == index // 16 and entry["window_start"] == index % 16
                and entry["occurrences"] == 21, "Family/window mapping differs from frozen order")
        require(np.array_equal(old_rows[entry["slot"]], entry["replaced_original_prepared_rows"]),
                "Replaced original row identity differs")
    unchanged = np.setdiff1d(np.arange(2960), slots)
    require(np.array_equal(unchanged, bindings["unchanged_old_slots"]), "Unchanged slot list differs")
    return slots, unchanged


def validate_core(plan, row, arrays):
    require(plan["split"] == "train" and plan["semantic_split"] == "train"
            and plan["training_eligible"] is True and plan["core_training_eligible"] is True
            and plan["warmup_training_eligible"] is False
            and plan["development_evaluation_eligible"] is False, "Only registered train core is eligible")
    require(row["role"] == "core" and row["semantic_split"] == "train"
            and row["split"] == "train" and row["label"] == plan["episode_id"]
            and row["family_id"] == plan["family_id"], "Warmup/development/identity contamination")
    states, actions = arrays["state"], arrays["action_applied"]
    require(states.dtype == np.float64 and states.shape == (21, 10)
            and actions.dtype == np.float64 and actions.shape == (20, 2)
            and np.isfinite(states).all() and np.isfinite(actions).all(), "Core shape/dtype/finiteness invalid")
    require(np.array_equal(actions, np.asarray(plan["core_actions"], dtype=np.float64))
            and np.array_equal(actions, arrays["action_requested"]), "Applied core command differs")
    require(arrays["transition_valid"].shape == (20,) and bool(arrays["transition_valid"].all()),
            "Invalid core transition cannot be silently filtered")
    require(np.max(np.abs(arrays["actual_dt"] - .1)) < 1e-10, "Core time step differs")


def materialize_arms(main_run, old_rows, old_states, old_actions, old_mask, positions, offsets):
    """Load only referenced train core episodes, validating immutable raw hashes."""
    main_run = Path(main_run).resolve()
    registration = read_json(main_run / "registration.json")
    bindings = read_json(main_run / "slots_bound.json")
    inventory = read_json(main_run / "artifact_hashes.json")["sha256"]
    summary = read_json(main_run / "summary.json")
    require(summary["status"] == "PASS" and summary["failure"] is None, "Collection did not pass")
    for name in ("registration.json", "slots_bound.json", "summary.json"):
        require(digest(main_run / name) == inventory[name], f"Collection artifact changed: {name}")
    require(digest(main_run / "slots_bound.json") == registration["slots_bound_sha256"], "Slot registration changed")
    slots, unchanged = checked_slots(bindings, old_rows, positions, offsets)
    plans = {p["episode_id"]: p for p in registration["plans"]}
    rows = {r["label"]: r for r in summary["episodes"]}
    cores, raw_hashes = {}, {}
    referenced = sorted({episode for e in bindings["entries"] for episode in e["arm_episodes"].values()})
    require(len(referenced) == 96, "Expected 24 train families times four variants")
    for episode in referenced:
        plan, row = plans[episode], rows[episode]
        folder = (main_run / row["dataset"]).resolve()
        require(folder.is_relative_to(main_run), "Episode path escaped collection")
        manifest_path = folder / "manifest.json"
        rel = manifest_path.relative_to(main_run).as_posix()
        require(digest(manifest_path) == row["manifest_sha256"] == inventory[rel], "Raw manifest hash differs")
        manifest = read_json(manifest_path)
        require(len(manifest["episodes"]) == 1, "One core per dataset required")
        entry = manifest["episodes"][0]
        require(entry["episode_id"] == episode and entry["family_id"] == plan["family_id"]
                and entry["split"] == "train", "Raw manifest identity/split differs")
        raw_hashes[rel] = digest(manifest_path)
        for key in ("npz", "json"):
            source = (folder / entry[key]).resolve()
            require(source.parent == folder / "episodes", "Unsafe raw artifact path")
            rel = source.relative_to(main_run).as_posix()
            require(digest(source) == entry[key + "_sha256"] == inventory[rel], "Raw artifact hash differs")
            raw_hashes[rel] = digest(source)
        with np.load(folder / entry["npz"], allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in
                      ("state", "action_applied", "action_requested", "transition_valid", "actual_dt")}
        validate_core(plan, row, arrays)
        cores[episode] = arrays
    require(old_states.shape == (2960, 6, 10) and old_actions.shape == (2960, 5, 2)
            and old_mask.shape == (2960, 5) and bool(old_mask.all()), "Original tensors differ")
    datasets = {"R": (old_states.clone(), old_actions.clone(), old_mask.clone())}
    identities = {}
    for arm in ARM_CONDITIONS:
        states, actions, mask = (x.clone() for x in datasets["R"])
        for entry in bindings["entries"]:
            episode = entry["arm_episodes"][arm]
            plan = plans[episode]
            require((plan["y_condition"], plan["stop_condition"]) == ARM_CONDITIONS[arm]
                    and plan["family_id"] == entry["family_id"]
                    and plan["family_rank"] == entry["family_rank"], "Arm/family binding differs")
            start, slot = entry["window_start"], entry["slot"]
            states[slot] = torch.from_numpy(cores[episode]["state"][start:start+6].copy())
            actions[slot] = torch.from_numpy(cores[episode]["action_applied"][start:start+5].copy())
        require(torch.equal(states[unchanged], old_states[unchanged])
                and torch.equal(actions[unchanged], old_actions[unchanged])
                and torch.equal(mask, old_mask), "Old retained tensors or mask changed")
        datasets[arm] = states, actions, mask
    for arm, tensors in datasets.items():
        identities[arm] = {"data_hash": data_hash(*tensors), "old_windows": 2960 if arm == "R" else 2576,
                           "new_windows": 0 if arm == "R" else 384, "window_population": 2960}
    report = {"schema": "p4-data-training-bindings-v1", "arm_order": list(ARMS),
              "arms": identities, "collection_registration_sha256": digest(main_run / "registration.json"),
              "slots_bound_sha256": digest(main_run / "slots_bound.json"),
              "raw_core_artifact_sha256": raw_hashes, "train_core_episodes_loaded": len(cores),
              "development_episodes_loaded": 0, "warmup_episodes_loaded": 0,
              "normalizer_refit": False, "physics_steps": 0}
    return datasets, slots, report


def checkpoint_hash(metadata):
    return json_hash({k: v for k, v in metadata.items() if k != "checkpoint_hash"})


def save_intervention_checkpoint(path, model, optimizer, normalizer, config, *, arm,
                                 sampler, manifest_sha256, dataset_hash, initial_checkpoint_hash):
    require(arm in ARMS and sampler.updates in range(100, 1001, 100), "Unregistered checkpoint point")
    require(sampler.window_count == 2960 and sampler.batch_windows == 64
            and sampler.max_updates == 1000 and sampler.horizon == 5, "Sampler budget differs")
    _config_schema(config, normalizer)
    weights = _clone(model.state_dict())
    _validate_weights(weights, model)
    optimizer_state = _clone(optimizer.state_dict())
    _validate_optimizer(optimizer_state, model, config, 1300 + sampler.updates)
    state = {"optimizer": optimizer_state, "sampler": sampler.state_dict(),
             "torch_rng_state": torch.get_rng_state().clone()}
    _rng(state["torch_rng_state"])
    path = Path(path); path.mkdir(parents=True, exist_ok=False)
    for name, value in (("weights.pt", weights), ("training_state.pt", state)):
        with (path / name).open("xb") as stream: torch.save(value, stream)
    metadata = {"format": "push-align-data-intervention-checkpoint-v1", "arm": arm,
                "config": deepcopy(config), "normalizer_hash": normalizer.metadata["normalizer_hash"],
                "manifest_sha256": manifest_sha256, "dataset_hash": dataset_hash,
                "initial_checkpoint_hash": initial_checkpoint_hash, "seed": 11,
                "rollout_weight": .1, "additional_updates": sampler.updates,
                "total_updates": 1300 + sampler.updates,
                "files": {name: digest(path / name) for name in ("weights.pt", "training_state.pt")}}
    metadata["checkpoint_hash"] = checkpoint_hash(metadata)
    (path / "checkpoint.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata


def load_intervention_checkpoint(path, normalizer, config, *, expected_manifest_sha256=None,
                                 expected_arm=None, expected_dataset_hash=None,
                                 expected_initial_checkpoint_hash=None, resume=False):
    """Safe-load owned CPU model; validates continuation state even for inference."""
    path = Path(path); meta = read_json(path / "checkpoint.json")
    require(meta["format"] == "push-align-data-intervention-checkpoint-v1"
            and meta["checkpoint_hash"] == checkpoint_hash(meta), "Checkpoint metadata integrity mismatch")
    require(meta["arm"] in ARMS and meta["seed"] == 11 and meta["rollout_weight"] == .1
            and meta["additional_updates"] in range(100, 1001, 100)
            and meta["total_updates"] == 1300 + meta["additional_updates"], "Checkpoint arm/update contract differs")
    require(json_hash(config) == json_hash(meta["config"])
            and meta["normalizer_hash"] == normalizer.metadata["normalizer_hash"], "Checkpoint config/normalizer differs")
    if expected_manifest_sha256 is not None:
        require(meta["manifest_sha256"] == expected_manifest_sha256, "Checkpoint run manifest differs")
    for key, expected in (("arm", expected_arm), ("dataset_hash", expected_dataset_hash),
                          ("initial_checkpoint_hash", expected_initial_checkpoint_hash)):
        if expected is not None:
            require(meta[key] == expected, "Checkpoint binding differs: " + key)
    for name in ("weights.pt", "training_state.pt"):
        require(digest(path / name) == meta["files"][name], "Checkpoint tensor file hash mismatch")
    _config_schema(config, normalizer)
    with torch.random.fork_rng(devices=[]): model = DeltaMLP(config)
    weights = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
    require(set(state) == {"optimizer", "sampler", "torch_rng_state"}, "Unexpected continuation fields")
    _validate_weights(weights, model); model.load_state_dict(weights); model.eval()
    _validate_optimizer(state["optimizer"], model, config, meta["total_updates"])
    _rng(state["torch_rng_state"])
    sampler = WindowSampler.from_state_dict(state["sampler"])
    require(sampler.updates == meta["additional_updates"] and sampler.window_count == 2960
            and sampler.batch_windows == 64 and sampler.max_updates == 1000 and sampler.horizon == 5,
            "Checkpoint sampler budget differs")
    optimizer = None
    if resume:
        optimizer = make_optimizer(model, config); optimizer.load_state_dict(_clone(state["optimizer"]))
    return ForkCheckpointBundle(model, meta, optimizer,
                                sampler if resume else None,
                                state["torch_rng_state"].clone() if resume else None)
