"""Independent saved-record arithmetic and provenance audit.

The oracle uses NumPy arrays only for relative-position arithmetic. It never
imports the learning implementation, constructs a model, executes learned
inference, updates an optimizer, or reads a simulator. Checkpoint tensor
identities may be inspected without constructing their owning models.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "runs/p4-data-retrain-20261013-02"
ARMS = ("C0", "R1")
ATOL, RTOL = 1e-12, 1e-10


def read(path):
    return json.loads(Path(path).read_text(),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def equal(first, second):
    return (first.shape == second.shape and first.dtype == second.dtype
            and np.array_equal(first, second, equal_nan=first.dtype.kind in "fc"))


def tree_hash(value):
    """Independent implementation of typed saved tensor/state identity."""
    import torch
    hasher = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            hasher.update(b"tensor" + str(item.dtype).encode() + json.dumps(list(item.shape)).encode())
            hasher.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            hasher.update(b"dict")
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                visit(key); visit(item[key])
        elif isinstance(item, (list, tuple)):
            hasher.update(type(item).__name__.encode())
            for child in item: visit(child)
        else:
            hasher.update(type(item).__name__.encode()
                          + json.dumps(item, sort_keys=True, allow_nan=False).encode())
    visit(value)
    return hasher.hexdigest()


def tensors(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def sampler_hash(snapshot):
    """Hash only canonical sampler scalar metadata and independent state bytes."""
    fields = ("schema", "drop_last", "window_count", "batch_windows", "max_updates", "horizon",
              "epoch", "cursor", "updates", "window_occurrences", "transition_occurrences")
    hasher = hashlib.sha256(canonical({key: snapshot[key] for key in fields}))
    for key in ("permutation", "shuffle_rng_state"):
        value = snapshot[key]
        hasher.update(key.encode() + str(value.dtype).encode() + json.dumps(list(value.shape)).encode())
        hasher.update(value.contiguous().numpy().tobytes())
    return hasher.hexdigest()


def require(passed, message):
    if not passed:
        raise ValueError(message)


def relative_scale(delta_scale):
    """Use the frozen marginal delta scales without estimating statistics."""
    value = np.asarray(delta_scale)
    require(value.shape == (10,) and value.dtype == np.float64
            and np.isfinite(value).all() and np.all(value > 0),
            "Ten finite positive float64 delta scales required")
    return np.hypot(value[[0, 1]], value[[6, 7]])


def relative_residual(predicted, target):
    """Pusher-minus-object relative XY endpoint error; no yaw/gap claim."""
    predicted, target = np.asarray(predicted), np.asarray(target)
    require(predicted.shape == target.shape and predicted.ndim >= 2
            and predicted.shape[-1] == 10 and predicted.dtype == target.dtype == np.float64,
            "Matching float64 physical states with final width10 required")
    return ((predicted[..., [6, 7]] - predicted[..., [0, 1]])
            - (target[..., [6, 7]] - target[..., [0, 1]]))


def masked_relative_loss(predicted, target, mask, scale):
    """Independently average exactly two scalar terms per recorded transition."""
    residual = relative_residual(predicted, target)
    mask, scale = np.asarray(mask), np.asarray(scale)
    require(mask.dtype == np.bool_ and mask.shape == residual.shape[:-1],
            "Aligned boolean recorded-transition mask required")
    require(mask.any() and scale.shape == (2,) and scale.dtype == np.float64
            and np.isfinite(scale).all() and np.all(scale > 0),
            "Nonempty denominator and finite positive float64 XY scale required")
    selected = residual[mask]
    require(np.isfinite(selected).all(),
            "Nonfinite valid prediction cannot be removed from denominator")
    return float(np.square(selected / scale).sum() / (2 * np.count_nonzero(mask)))


def relative_moments(object_error, pusher_error, scale):
    """Moments preserve the cross-term that independent XY penalties omit."""
    object_error, pusher_error = np.asarray(object_error), np.asarray(pusher_error)
    scale = np.asarray(scale)
    require(object_error.shape == pusher_error.shape and object_error.ndim == 2
            and object_error.shape[1] == 2 and len(object_error) > 0
            and object_error.dtype == pusher_error.dtype == np.float64,
            "Matching nonempty float64 [N,2] XY errors required")
    require(scale.shape == (2,) and scale.dtype == np.float64
            and np.isfinite(scale).all() and np.all(scale > 0)
            and np.isfinite(object_error).all() and np.isfinite(pusher_error).all(),
            "Finite errors and positive XY scales required")
    obj, push = object_error / scale, pusher_error / scale
    return np.asarray([np.square(obj).sum(), np.square(push).sum(),
                       np.multiply(obj, push).sum()], dtype=np.float64)


def moments_loss(moments, valid_transitions):
    moments = np.asarray(moments)
    require(moments.shape == (3,) and moments.dtype == np.float64
            and np.isfinite(moments).all() and type(valid_transitions) is int
            and valid_transitions > 0, "Finite three-moment record and exact positive count required")
    return float((moments[0] + moments[1] - 2 * moments[2]) / (2 * valid_transitions))


def combined_loss(one, rollout, relative_one, relative_rollout,
                  rollout_weight, relative_weight):
    values = (one, rollout, relative_one, relative_rollout, rollout_weight, relative_weight)
    require(all(type(value) in (int, float) and np.isfinite(value) and value >= 0
                for value in values), "Loss components and weights must be finite nonnegative numbers")
    return float(one + rollout_weight * rollout
                 + relative_weight * (relative_one + rollout_weight * relative_rollout))


def segment_gap(state, config):
    """Signed distance to four world-space edges, avoiding the production box formula."""
    state = np.asarray(state)
    require(state.shape == (10,) and state.dtype == np.float64
            and np.isfinite(state).all(), "Finite physical float64 state required")
    hx, hy = np.asarray(config["object"]["size_full_xyz_m"][:2]) / 2
    angle = float(state[2])
    rotation = np.asarray([[math.cos(angle), -math.sin(angle)],
                           [math.sin(angle), math.cos(angle)]])
    vertices = np.asarray([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]]) @ rotation.T + state[:2]
    point = state[6:8]
    edges = np.roll(vertices, -1, axis=0) - vertices
    offset = point - vertices
    t = np.sum(offset * edges, axis=1) / np.sum(edges * edges, axis=1)
    closest = vertices + np.clip(t, 0., 1.)[:, None] * edges
    distance = float(np.min(np.linalg.norm(point - closest, axis=1)))
    inside = bool(np.all(edges[:, 0] * offset[:, 1] - edges[:, 1] * offset[:, 0] >= 0.))
    return (-distance if inside else distance) - float(config["pusher"]["radius_m"])


def metric_values(predicted, actual, config):
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    require(predicted.shape == actual.shape and predicted.ndim == 2
            and predicted.shape[1] == 10, "Matched state matrices required")
    residual = predicted - actual
    yaw = np.remainder(residual[:, 2] + np.pi, 2 * np.pi) - np.pi
    rel = relative_residual(predicted, actual)
    finite = np.isfinite(predicted).all(1) & np.isfinite(actual).all(1)
    predicted_gap = np.full(len(predicted), np.nan)
    actual_gap = np.full(len(predicted), np.nan)
    for row in np.flatnonzero(finite):
        predicted_gap[row] = segment_gap(predicted[row], config)
        actual_gap[row] = segment_gap(actual[row], config)
    return {
        "object_xy_m": np.linalg.norm(residual[:, :2], axis=1),
        "pusher_xy_m": np.linalg.norm(residual[:, 6:8], axis=1),
        "object_yaw_rad": np.abs(yaw),
        "object_velocity_m_s": np.linalg.norm(residual[:, 3:5], axis=1),
        "object_wz_rad_s": np.abs(residual[:, 5]),
        "pusher_velocity_m_s": np.linalg.norm(residual[:, 8:10], axis=1),
        "pusher_vy_absolute_m_s": np.abs(residual[:, 9]),
        "pusher_vy_signed_m_s": residual[:, 9],
        "relative_xy_m": np.linalg.norm(rel, axis=1),
        "relative_x_signed_m": rel[:, 0],
        "relative_y_signed_m": rel[:, 1],
        "predicted_gap_m": predicted_gap,
        "actual_gap_m": actual_gap,
        "gap_signed_error_m": predicted_gap - actual_gap,
        "gap_absolute_m": np.abs(predicted_gap - actual_gap),
        "predicted_object_speed_m_s": np.linalg.norm(predicted[:, 3:5], axis=1),
        "actual_object_speed_m_s": np.linalg.norm(actual[:, 3:5], axis=1),
        "predicted_pusher_speed_m_s": np.linalg.norm(predicted[:, 8:10], axis=1),
        "actual_pusher_speed_m_s": np.linalg.norm(actual[:, 8:10], axis=1),
        "predicted_object_wz_rad_s": predicted[:, 5],
        "actual_object_wz_rad_s": actual[:, 5],
    }


class Audit:
    def __init__(self):
        self.checks = []

    def check(self, name, passed, **details):
        self.checks.append({"name": name, "pass": bool(passed), **details})
        require(passed, name)

    def numeric(self, name, actual, expected):
        self.check(name, actual is None if expected is None else actual is not None
                   and np.isclose(float(actual), float(expected), rtol=RTOL, atol=ATOL))

    def training(self, run):
        """Inspect saved data, states and losses without reconstructing a model."""
        import torch
        manifest, summary, registration = (read(run / name) for name in
                                          ("manifest.json", "summary.json", "registration.json"))
        index = read(run / "training_artifact_hashes.json")["sha256"]
        self.check("training_completed", summary["status"] == "PASS"
                   and {arm["branch"] for arm in summary["arms"]} == set(ARMS))
        for relative, expected in index.items():
            self.check("training_artifact:" + relative, sha(run / relative) == expected)
        self.check("registered_protocol_copy_hash", manifest["registration_sha256"] == sha(run / "registration.json"))
        for relative, expected in registration["bindings_sha256"].items():
            self.check("source_upstream_binding:" + relative, sha(ROOT / relative) == expected)
        self.check("registered_one_change", registration["arms"] == {"C0": 0., "R1": .2}
                   and registration["training"]["updates_per_arm"] == 1000
                   and canonical(registration["loss"]) == canonical(manifest["loss"]))
        old_manifest = read(HISTORICAL / "manifest.json")
        old_index = read(HISTORICAL / "training_artifact_hashes.json")["sha256"]
        for relative in ("manifest.json", "prepared_training.npz", "dataset_bindings.json"):
            self.check("historical_original_artifact:" + relative, sha(HISTORICAL / relative) == old_index[relative])
        self.check("initial_checkpoint_is_same_epoch100", manifest["initial_checkpoint"] == old_manifest["initial_checkpoint"]
                   and manifest["initial_checkpoint"]["initial_updates"] == 1300
                   and manifest["physical_config"] == old_manifest["physical_config"]
                   and manifest["normalizer_hash"] == old_manifest["normalizer_hash"])
        dataset = arrays(run / "prepared_training.npz")
        historical_dataset = arrays(HISTORICAL / "prepared_training.npz")
        self.check("training_only_D11_fields", set(dataset) == {"states", "actions", "mask"})
        for field, shape, dtype in (("states", (2960, 6, 10), np.float64),
                                    ("actions", (2960, 5, 2), np.float64), ("mask", (2960, 5), np.bool_)):
            self.check("same_historical_D11_" + field, equal(dataset[field], historical_dataset["D11_" + field])
                       and dataset[field].shape == shape and dataset[field].dtype == dtype
                       and np.isfinite(dataset[field]).all())
        self.check("recorded_valid_mask_not_adaptive", dataset["mask"].all())
        data_identity = tree_hash({"states": torch.from_numpy(dataset["states"]),
                                  "applied_actions": torch.from_numpy(dataset["actions"]),
                                  "mask": torch.from_numpy(dataset["mask"])})
        expected_hash = old_manifest["training_provenance"]["arms"]["D11"]["data_hash"]
        self.check("D11_data_identity_exact", data_identity == expected_hash
                   and all(manifest["training_provenance"]["arms"][branch]["data_hash"] == expected_hash for branch in ARMS))
        bindings = read(run / "dataset_bindings.json")
        self.check("dataset_source_and_no_refit", bindings["data_hash"] == data_identity
                   and sha(ROOT / bindings["source"]) == bindings["source_sha256"]
                   and bindings["original_training_provenance"] == old_manifest["training_provenance"]
                   and bindings["development_episodes_loaded_for_training"] == 0
                   and bindings["warmup_episodes_loaded_for_training"] == 0 and bindings["normalizer_refit"] is False
                   and sha(run / "dataset_bindings.json") == manifest["dataset_bindings_sha256"]
                   and sha(run / "prepared_training.npz") == manifest["prepared_training_sha256"])
        learning = ROOT / "runs/p3-02-20261009-01/learning_data/normalizer"
        normalizer = read(learning / "normalizer.json")
        self.check("original_normalizer_statistics_unchanged", normalizer["normalizer"]["normalizer_hash"] == manifest["normalizer_hash"]
                   and sha(learning / "statistics.npz") == normalizer["npz_sha256"])
        scale = relative_scale(arrays(learning / "statistics.npz")["delta_scale"])
        self.check("registered_relative_scale_original_only", np.allclose(scale, manifest["loss"]["relative_scale_m"], rtol=RTOL, atol=ATOL))
        initial = ROOT / manifest["initial_checkpoint"]["path"]
        initial_weights, initial_state = tensors(initial / "weights.pt"), tensors(initial / "training_state.pt")
        initial_meta = read(initial / "checkpoint.json")
        self.check("common_epoch100_checkpoint_hash_and_files", initial_meta["checkpoint_hash"] == manifest["initial_checkpoint"]["checkpoint_hash"]
                   and initial_meta["epoch"] == 100 and initial_meta["updates"] == 1300
                   and all(sha(initial / field) == value for field, value in initial_meta["files"].items()))
        generator = torch.Generator(device="cpu")
        generator.set_state(initial_state["shuffle_rng_state"])
        permutation, cursor, epoch = torch.randperm(2960, generator=generator), 0, 1
        initial_sampler = {"schema": "p4-window-sampler-v1", "drop_last": False,
            "window_count": 2960, "batch_windows": 64, "max_updates": 1000, "horizon": 5,
            "epoch": 1, "cursor": 0, "updates": 0, "window_occurrences": 0, "transition_occurrences": 0,
            "permutation": permutation.clone(), "shuffle_rng_state": generator.get_state().clone()}
        start_identity = {"model": tree_hash(initial_weights), "Adam": tree_hash(initial_state["optimizer"]),
            "Torch_RNG": tree_hash(initial_state["torch_rng_state"]),
            "common_shuffle_RNG_before_window_permutation": tree_hash(initial_state["shuffle_rng_state"]),
            "initial_sampler": sampler_hash(initial_sampler)}
        expected_batches, offsets, boundaries = [], [0], {}
        for update in range(1, 1001):
            if cursor == 2960:
                permutation = torch.randperm(2960, generator=generator); cursor = 0; epoch += 1
            stop = min(cursor + 64, 2960)
            batch = permutation[cursor:stop].numpy().copy()
            expected_batches.append(batch); offsets.append(offsets[-1] + len(batch)); cursor = stop
            if update % 100 == 0:
                boundaries[update] = {"epoch": epoch, "cursor": cursor, "permutation": permutation.clone(),
                                      "shuffle_rng_state": generator.get_state().clone(), "windows": offsets[-1]}
        flat, offsets = np.concatenate(expected_batches), np.asarray(offsets, np.int64)
        historical_trace = arrays(HISTORICAL / "arms/D11/training_trace.npz")
        self.check("independently_reconstructed_historical_schedule", equal(flat, historical_trace["window_positions"])
                   and equal(offsets, historical_trace["batch_offsets"]))
        batch_hashes = np.asarray([tree_hash({"states": torch.from_numpy(dataset["states"][batch]),
            "applied_actions": torch.from_numpy(dataset["actions"][batch]), "mask": torch.from_numpy(dataset["mask"][batch])})
            for batch in expected_batches])
        self.check("all_actual_D11_batch_hashes", equal(batch_hashes, historical_trace["batch_data_hashes"]))
        self.check("same_979full21partial_total_exposures", len(flat) == 62992
                   and int((np.diff(offsets) == 64).sum()) == 979 and int((np.diff(offsets) == 16).sum()) == 21
                   and len(flat) * 5 == 314960)
        slots = historical_dataset["replacement_slots"]
        exposures = np.bincount(flat, minlength=2960)
        partial = np.concatenate([batch for batch in expected_batches if len(batch) == 16])
        self.check("384_new_slots_exact21_without_partial16", len(slots) == 384
                   and np.all(exposures[slots] == 21) and not np.isin(slots, partial).any()
                   and int(exposures[slots].sum()) == 8064)
        losses = {}
        for branch in ARMS:
            folder = run / "arms" / branch
            recorded_start, branch_summary = read(folder / "start_identity.json"), read(folder / "summary.json")
            self.check(branch + ":start_model_Adam_RNG_shuffle_exact", recorded_start == start_identity
                       == read(HISTORICAL / "arms/D11/start_identity.json") == branch_summary["start_identity"])
            trace = arrays(folder / "training_trace.npz")
            self.check(branch + ":full_sampler_data_order", equal(trace["window_positions"], flat)
                       and equal(trace["batch_offsets"], offsets) and equal(trace["batch_data_hashes"], batch_hashes))
            self.check(branch + ":loss_columns_and_shape", trace["losses"].shape == (1000, 6)
                       and trace["losses"].dtype == np.float64 and np.isfinite(trace["losses"]).all()
                       and trace["loss_columns"].tolist() == ["one_step", "rollout", "relative_teacher", "relative_free", "total", "gradient_norm"])
            losses[branch] = trace["losses"]
            if branch == "C0":
                self.check("C0_all1000_historical_losses_gradients_exact", equal(trace["losses"][:, [0, 1, 4, 5]], historical_trace["losses"])
                           and np.array_equal(trace["losses"][:, [2, 3]], np.zeros((1000, 2))))
            else:
                residual = arrays(folder / "relative_residual_trace.npz")
                self.check("R1_all_teacher_free_xy_residuals_finite", all(residual[key].shape == (314960, 4)
                           and residual[key].dtype == np.float64 and np.isfinite(residual[key]).all()
                           for key in ("teacher_xy_errors", "free_xy_errors")))
                self.check("R1_recorded_denominator_and_fixedscale", equal(residual["transition_offsets"], offsets * 5)
                           and np.allclose(residual["relative_scale_m"], scale, rtol=RTOL, atol=ATOL))
                for update in range(1000):
                    first, last = residual["transition_offsets"][update:update + 2]
                    count = int(last - first)
                    self.check(f"R1:{update + 1}:denominator_2N", count == len(expected_batches[update]) * 5)
                    for key, column in (("teacher_xy_errors", 2), ("free_xy_errors", 3)):
                        xy = residual[key][first:last]
                        direct = float(np.square((xy[:, 2:] - xy[:, :2]) / scale).sum() / (2 * count))
                        moment = moments_loss(relative_moments(xy[:, :2], xy[:, 2:], scale), count)
                        self.numeric(f"R1:{update + 1}:{key}:direct_loss", trace["losses"][update, column], direct)
                        self.numeric(f"R1:{update + 1}:{key}:cross_term_closure", moment, direct)
            for update in range(1000):
                row = trace["losses"][update]
                expected_total = combined_loss(*row[:4].tolist(), .1, 0. if branch == "C0" else .2)
                self.numeric(f"{branch}:{update + 1}:objective_arithmetic", row[4], expected_total)
            self.check(branch + ":clipping_count", branch_summary["clipped_updates"] == int((trace["losses"][:, 5] > 1.).sum()))
            self.check(branch + ":budget_and_final_only", branch_summary["status"] == "PASS"
                       and branch_summary["additional_updates"] == 1000 and branch_summary["total_Adam_steps"] == 2300
                       and branch_summary["checkpoint_selection"] == "final1000_only" and len(branch_summary["checkpoints"]) == 10
                       and branch_summary["window_occurrences"] == 62992 and branch_summary["transition_occurrences"] == 314960)
            for update in range(100, 1001, 100):
                path = folder / "checkpoints" / f"update-{update:04d}"
                self.checkpoint(branch, path, boundaries[update], update, run, manifest,
                                data_identity, initial_weights, initial_state, scale)
            proof = read(folder / "resume-0100.json")
            self.check(branch + ":recorded_discarded_resume_proof", proof["status"] == "PASS"
                       and proof["discarded_updates"] == 4 and proof["copies_after_two_updates_exact"] is True
                       and proof["official_rng_unchanged"] is True)
        self.check("first_update_base_terms_are_identical", equal(losses["C0"][0, :2], losses["R1"][0, :2]))
        self.check("exact2000formal8disposable", summary["counters"] == {"formal_attempts": 2000, "formal_updates": 2000,
                   "disposable_attempts": 8, "disposable_updates": 8})
        self.check("no_physics_finaltest_closedloop_task_claim", not any(summary["future_engine_calls"].values())
                   and summary["final_test_used"] is False and summary["new_closed_loop_execution"] is False
                   and summary["task_success"] == "NOT_EVALUATED" and summary["P5_01"] == "HOLD")
        self.check("training_manifest_summary_binding", sha(run / "manifest.json") == summary["manifest_sha256"])
        self.observed_training = {"formal_updates": 2000, "recorded_discarded_updates": 8,
            "windows_per_arm": 62992, "teacher_transitions_per_arm": 314960,
            "data_hash": data_identity, "original_delta_scale_derived_m": scale.tolist(),
            "clipped_updates": {branch: int((value[:, 5] > 1.).sum()) for branch, value in losses.items()}}

    def checkpoint(self, branch, path, expected, update, run, manifest, data_identity,
                   initial_weights, initial_state, scale):
        import torch
        metadata = read(path / "checkpoint.json")
        self.check(f"{branch}:{update}:checkpoint_metadata_hash", metadata["checkpoint_hash"]
                   == hashlib.sha256(canonical({key: value for key, value in metadata.items() if key != "checkpoint_hash"})).hexdigest()
                   and all(sha(path / field) == value for field, value in metadata["files"].items()))
        self.check(f"{branch}:{update}:checkpoint_provenance", metadata["format"] == "push-align-relative-loss-checkpoint-v1"
                   and metadata["branch"] == branch and metadata["relative_weight"] == (0. if branch == "C0" else .2)
                   and metadata["rollout_weight"] == .1
                   and metadata["formula"] == "Lone+0.1*Lroll+beta*(Lrelative_teacher+0.1*Lrelative_free)"
                   and metadata["config"] == manifest["physical_config"]
                   and metadata["normalizer_hash"] == manifest["normalizer_hash"]
                   and np.allclose(metadata["relative_scale_m"], scale, rtol=RTOL, atol=ATOL)
                   and metadata["seed"] == 11 and metadata["additional_updates"] == update
                   and metadata["total_updates"] == 1300 + update and metadata["dataset_hash"] == data_identity
                   and metadata["initial_checkpoint_hash"] == manifest["initial_checkpoint"]["checkpoint_hash"]
                   and metadata["manifest_sha256"] == sha(run / "manifest.json"))
        weights, state = tensors(path / "weights.pt"), tensors(path / "training_state.pt")
        self.check(f"{branch}:{update}:same_architecture_dtype", weights.keys() == initial_weights.keys()
                   and all(weights[key].shape == initial_weights[key].shape and weights[key].dtype == torch.float32
                           and bool(torch.isfinite(weights[key]).all()) for key in weights)
                   and sum(value.numel() for value in weights.values()) == 19338)
        optimizer = state["optimizer"]
        self.check(f"{branch}:{update}:Adam_exact_contract", optimizer["param_groups"] == initial_state["optimizer"]["param_groups"]
                   and optimizer["state"].keys() == initial_state["optimizer"]["state"].keys()
                   and all(value["step"].dtype == torch.float32 and float(value["step"]) == 1300 + update
                           and all(value[field].dtype == torch.float32 and bool(torch.isfinite(value[field]).all())
                                   for field in ("exp_avg", "exp_avg_sq"))
                           and bool((value["exp_avg_sq"] >= 0).all()) for value in optimizer["state"].values()))
        sampler = state["sampler"]
        self.check(f"{branch}:{update}:saved_sampler_regenerated", sampler["updates"] == update
                   and sampler["window_count"] == 2960 and sampler["batch_windows"] == 64
                   and sampler["max_updates"] == 1000 and sampler["horizon"] == 5 and sampler["drop_last"] is False
                   and sampler["epoch"] == expected["epoch"] and sampler["cursor"] == expected["cursor"]
                   and sampler["window_occurrences"] == expected["windows"]
                   and sampler["transition_occurrences"] == expected["windows"] * 5
                   and torch.equal(sampler["permutation"], expected["permutation"])
                   and torch.equal(sampler["shuffle_rng_state"], expected["shuffle_rng_state"])
                   and sampler["state_hash"] == sampler_hash(sampler))
        self.check(f"{branch}:{update}:unchanged_Torch_rng", torch.equal(state["torch_rng_state"], initial_state["torch_rng_state"]))
        if branch == "C0":
            historical = HISTORICAL / "arms/D11/checkpoints" / f"update-{update:04d}"
            self.check(f"C0:{update}:historical_weights_Adam_sampler_rng_exact",
                       tree_hash(weights) == tree_hash(tensors(historical / "weights.pt"))
                       and tree_hash(state) == tree_hash(tensors(historical / "training_state.pt")))

    def geometry_rows(self, name, record, predicted, actual, valid, identities, config):
        metrics = metric_values(predicted, actual, config)
        finite = np.isfinite(predicted).all(1) & np.isfinite(actual).all(1)
        self.check(name + ":denominators", record["attempted"] == len(valid)
                   and record["valid"] == int(valid.sum())
                   and record["invalid"] == int((~valid).sum())
                   and record["finite_diagnostic_rows"] == int(finite.sum())
                   and record["finite_invalid_diagnostic_rows"] == int((finite & ~valid).sum())
                   and record["missing_rows"] == int((~finite).sum()))
        self.check(name + ":all_rows_retained", len(record["rows"]) == len(valid))
        for index, (row, identity) in enumerate(zip(record["rows"], identities)):
            self.check(f"{name}:{index}:identity_and_flags",
                all(row[key] == value for key, value in identity.items())
                and row["prediction_valid"] is bool(valid[index])
                and row["finite_state_available"] is bool(finite[index])
                and row["finite_invalid_diagnostic_only"] is bool(finite[index] and not valid[index]))
            for metric, values in metrics.items():
                self.numeric(f"{name}:{index}:{metric}", row["metrics"][metric],
                             float(values[index]) if finite[index] else None)
        for metric, values in metrics.items():
            self.numeric(name + ":valid_mean:" + metric,
                         record["metrics_valid_only_descriptive"][metric],
                         float(values[valid].mean()) if valid.any() else None)
            self.numeric(name + ":finite_invalid_diagnostic_mean:" + metric,
                         record["metrics_finite_including_invalid_diagnostic_only"][metric],
                         float(values[finite].mean()) if finite.any() else None)
        return metrics

    def prediction(self, run, config):
        folder = run / "prediction_evaluation"
        summary = read(folder / "summary.json")
        index = read(folder / "artifact_hashes.json")["sha256"]
        for relative, expected in index.items():
            self.check("eval_artifact:" + relative, sha(folder / relative) == expected)
        self.check("evaluation_pass_no_physics_finaltest", summary["status"] == "PASS"
                   and summary["new_physics_steps"] == 0 and summary["learning_updates"] == 0
                   and summary["final_test_used"] is False and summary["P5_01"] == "HOLD")
        result_by_arm, primary_by_arm = {}, {}
        shared = None
        for arm in ARMS:
            record = read(folder / arm / "development.json")
            legacy = read(folder / arm / "development_legacy.json")
            trace = arrays(folder / arm / "development_traces.npz")
            actual, actions, regimes = (trace[key] for key in ("actual_states", "actual_actions", "actual_regimes"))
            self.check(arm + ":48complete_actuals", actual.shape == (48, 21, 10)
                       and actual.dtype == np.float64 and actions.shape == (48, 20, 2))
            if shared is None: shared = (actual, actions, regimes)
            else: self.check(arm + ":identical_actual_sample_values",
                             all(equal(first, second) for first, second in zip(shared, (actual, actions, regimes))))
            self.check(arm + ":twelve_families", record["family_count"] == 12
                       and record["model_seed_count"] == 1 and len(record["families"]) == len(set(record["families"])) == 12
                       and record["families"] == legacy["families"])
            hidx, sidx = trace["head_indices"], trace["stop_indices"]
            hids, sids, tids = (legacy[key]["identities"] for key in
                              ("Y_heads_teacher_conditioned", "S1_free_stop", "S1_teacher_conditioned_stop"))
            head = self.geometry_rows(arm + ":head", record["Y_heads_teacher_conditioned"],
                trace["head_states"][:, 1], actual[hidx, 1:6].reshape(-1, 10), trace["head_valid"], hids, config)
            stop = self.geometry_rows(arm + ":stop", record["S1_free_stop"],
                trace["stop_states"][:, 5], actual[sidx, 15], trace["stop_valid"], sids, config)
            self.geometry_rows(arm + ":teacher_stop", record["S1_teacher_conditioned_stop"],
                trace["stop_local_states"][:, 1], actual[sidx, 11:16].reshape(-1, 10), trace["stop_local_valid"], tids, config)
            for step in range(1, 6):
                valid = (trace["stop_failure_steps"] == -1) | (trace["stop_failure_steps"] > step)
                self.geometry_rows(f"{arm}:free_stop_k{step}", record["S1_free_stop_K1_to_K5"][str(step)],
                    trace["stop_states"][:, step], actual[sidx, 10 + step], valid, sids, config)
            for prefix in ("head", "stop", "stop_local", "full_H10"):
                self.check(arm + ":failure_mask:" + prefix,
                           equal(trace[prefix + "_valid"], trace[prefix + "_failure_steps"] == -1))
                self.rollout_closure(arm + ":" + prefix, trace, config)
            self.check(arm + ":deduplicated_head_S0", len(hidx) == 24 and len(sidx) == 24
                       and all(identity["stop_condition"] == "S0" for identity in hids)
                       and all(identity["stop_condition"] == "S1" for identity in sids))
            self.check(arm + ":five_zero_actual_stop_commands", np.array_equal(actions[sidx, 10:15], np.zeros((24, 5, 2))))
            primary_by_arm[arm] = {}
            for phase, metric_rows, valid, identities in (
                ("contact_Y1_teacher_first5", head, trace["head_valid"], hids),
                ("stop_S1_free_K5", stop, trace["stop_valid"], sids)):
                primary_by_arm[arm][phase] = {}
                for family in record["families"]:
                    selected = [i for i, identity in enumerate(identities) if identity["family_id"] == family
                                and (phase != "contact_Y1_teacher_first5" or identity["y_condition"] == "Y1")]
                    all_valid = bool(valid[selected].all())
                    membership = record["primary_membership"][phase][family]
                    self.check(arm + family + phase + ":membership", len(selected) == (5 if phase.startswith("contact") else 2)
                               and membership["attempted_rows"] == len(selected)
                               and membership["valid_rows"] == int(valid[selected].sum())
                               and membership["all_rows_valid"] is all_valid
                               and membership["invalid_row_identities"] == [identities[i] for i in selected if not valid[i]])
                    primary_by_arm[arm][phase][family] = {}
                    for metric in ("relative_xy_m", "gap_absolute_m", "object_xy_m", "pusher_xy_m"):
                        value = float(metric_rows[metric][selected].mean()) if all_valid else None
                        self.numeric(arm + family + phase + metric, record["primary"][phase][family][metric], value)
                        primary_by_arm[arm][phase][family][metric] = value
            result_by_arm[arm] = record
            if arm == "C0":
                old = HISTORICAL / "prediction_evaluation" / "D11"
                self.check("control_development_legacy_exact", canonical(legacy) == canonical(read(old / "development.json")))
                old_trace = arrays(old / "development_traces.npz")
                self.check("control_development_all_traces_exact", trace.keys() == old_trace.keys()
                           and all(equal(trace[key], old_trace[key]) for key in trace))
                for file in ("original_validation.json", "original_validation_geometry.json"):
                    self.check("control_exact:" + file, canonical(read(folder / arm / file)) == canonical(read(old / file)))
                observed, expected = arrays(folder / arm / "original_validation_traces.npz"), arrays(old / "original_validation_traces.npz")
                self.check("control_original_validation_all_traces_exact", observed.keys() == expected.keys()
                           and all(equal(observed[key], expected[key]) for key in observed))
        self.paired_comparison(folder, result_by_arm, primary_by_arm)

    def rollout_closure(self, name, trace, config):
        prefix = name.split(":")[-1]
        states, deltas = trace[prefix + "_states"], trace[prefix + "_raw_deltas"]
        failures, reasons = trace[prefix + "_failure_steps"], trace[prefix + "_failure_reasons"]
        for row in range(len(states)):
            failure = int(failures[row])
            for step in range(deltas.shape[1]):
                delta, previous, predicted = deltas[row, step], states[row, step], states[row, step + 1]
                if np.isfinite(delta).all() and np.isfinite(predicted).all():
                    expected = previous + delta
                    expected[2] = np.remainder(expected[2] + np.pi, 2 * np.pi) - np.pi
                    self.check(f"{name}:{row}:{step}:raw_delta_endpoint_closure",
                               np.allclose(expected, predicted, rtol=RTOL, atol=ATOL))
                    gap = segment_gap(predicted, config)
                    if failure == step + 1 and reasons[row] == "predicted_deep_penetration":
                        self.check(f"{name}:{row}:{step}:deep_gap_reproduced",
                                   gap < -config["validity"]["predicted_penetration_tolerance_m"])
                    if failure == -1 or failure > step + 1:
                        self.check(f"{name}:{row}:{step}:valid_gap_not_beyond_tolerance",
                                   gap >= -config["validity"]["predicted_penetration_tolerance_m"] - ATOL)
                if failure >= 0 and step + 1 > failure:
                    self.check(f"{name}:{row}:{step}:invalid_suffix_not_continued",
                               not np.isfinite(predicted).any() and not np.isfinite(delta).any())

    def paired_comparison(self, folder, result_by_arm, primary_by_arm):
        record = read(folder / "comparison.json")
        families = record["family_ids"]
        draws = np.random.default_rng(2026101401).integers(0, 12, (10000, 12), dtype=np.int64)
        self.check("registered_family_bootstrap_draws", equal(arrays(folder / "bootstrap_family_draws.npz")["draws"], draws))
        for phase in ("contact_Y1_teacher_first5", "stop_S1_free_K5"):
            independent = {}
            for metric in ("relative_xy_m", "gap_absolute_m", "object_xy_m", "pusher_xy_m"):
                values = []
                for family in families:
                    first, second = (primary_by_arm[arm][phase][family][metric] for arm in ARMS)
                    values.append(None if first is None or second is None else float(second - first))
                row = record["contrasts"][phase][metric]
                missing = [family for family, value in zip(families, values) if value is None]
                self.check(phase + metric + ":retained_all_twelve", row["attempted_families"] == 12
                           and row["family_ids"] == families and row["valid_families"] == 12 - len(missing)
                           and row["invalid_family_ids"] == missing)
                for i, value in enumerate(values):
                    self.numeric(phase + metric + f":difference{i}", row["family_differences"][i], value)
                if missing:
                    self.check(phase + metric + ":invalid_held", row["status"] == "HOLD_INVALID_PRIMARY"
                               and row["mean_difference"] is None and row["bootstrap_percentile_95"] == [None, None]
                               and row["improved"] is row["worsened"] is row["exact_float64_ties"] is None)
                    independent[metric] = None
                else:
                    values = np.asarray(values)
                    interval = np.quantile(values[draws].mean(1), [.025, .975])
                    self.numeric(phase + metric + ":mean", row["mean_difference"], values.mean())
                    self.check(phase + metric + ":interval", row["status"] == "COMPLETE"
                               and np.allclose(interval, row["bootstrap_percentile_95"], rtol=RTOL, atol=ATOL)
                               and (row["improved"], row["worsened"], row["exact_float64_ties"])
                               == (int((values < 0).sum()), int((values > 0).sum()), int((values == 0).sum())))
                    independent[metric] = (float(values.mean()), interval)
            primary = [independent[key] for key in ("relative_xy_m", "gap_absolute_m")]
            if any(value is None for value in primary): decision = "HOLD_INVALID_PRIMARY"
            elif all(value[1][1] < 0 for value in primary): decision = "DIRECTIONAL_SUPPORT"
            elif all(value[1][0] > 0 for value in primary): decision = "DIRECTIONAL_WORSENING"
            elif primary[0][0] * primary[1][0] < 0: decision = "MIXED"
            else: decision = "INCONCLUSIVE"
            self.check(phase + ":effect_decision", record["effect_decisions"][phase]["status"] == decision)
            grouped = {key: [] for key in ("both_valid", "C0_invalid_R1_valid", "C0_valid_R1_invalid", "both_invalid")}
            for family in families:
                first, second = (result_by_arm[arm]["primary_membership"][phase][family]["all_rows_valid"] for arm in ARMS)
                label = "both_valid" if first and second else "C0_valid_R1_invalid" if first else "C0_invalid_R1_valid" if second else "both_invalid"
                grouped[label].append(family)
            for label, members in grouped.items():
                self.check(phase + ":validity:" + label, record["validity_changes"][phase][label]
                           == {"count": len(members), "family_ids": members})


def audit(run, *, training_only=False):
    """No learned execution: deserialize states and recompute saved arithmetic."""
    import torch
    engine = {"learned_module_calls": 0, "backward_calls": 0}
    original_call, original_backward = torch.nn.Module._call_impl, torch.Tensor.backward
    original_rng = torch.get_rng_state().clone()
    def forbid_call(*args, **kwargs):
        engine["learned_module_calls"] += 1
        raise RuntimeError("Independent audit forbids learned model execution")
    def forbid_backward(*args, **kwargs):
        engine["backward_calls"] += 1
        raise RuntimeError("Independent audit forbids backward execution")
    torch.nn.Module._call_impl, torch.Tensor.backward = forbid_call, forbid_backward
    verifier = Audit()
    report = {"schema": "p4-relative-loss-independent-review-v1", "status": "FAIL",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(run.relative_to(ROOT)), "source_sha256": sha(__file__),
        "mode": "training_only" if training_only else "training_and_prediction",
        "execution_during_review": {"model_instances": 0, "model_forwards": 0,
            "optimizer_updates": 0, "physics_steps": 0, "serialization_tensor_reads_allowed": True},
        "checks": verifier.checks,
        "numerical_recomputation_tolerances": {"absolute": ATOL, "relative": RTOL,
            "scope": "Arithmetic equivalence only; never modifies validity or outcome thresholds"}}
    try:
        verifier.training(run)
        if not training_only:
            verifier.prediction(run, read(run / "manifest.json")["physical_config"])
        verifier.check("review_no_learned_or_backward_execution", not any(engine.values()))
        verifier.check("review_global_Torch_rng_unchanged", torch.equal(original_rng, torch.get_rng_state()))
        report.update(status="PASS", observed_training=verifier.observed_training,
            manifest_sha256=sha(run / "manifest.json"), summary_sha256=sha(run / "summary.json"),
            training_index_sha256=sha(run / "training_artifact_hashes.json"),
            forbidden_execution_counts=engine,
            limitations=["Saved forward residuals are independently reduced, not independently generated by the auditor.",
                "Checkpoint reload update proofs are inspected as records, not re-executed.",
                "Endpoint/gap oracle and family uncertainty check stored predictions; they do not rerun inference or physical integration.",
                "No new task-success, personal-understanding, generalization or causal-mechanism conclusion follows from audit PASS."])
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        torch.nn.Module._call_impl, torch.Tensor.backward = original_call, original_backward
    report["check_count"] = len(verifier.checks)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-only", action="store_true")
    arguments = parser.parse_args()
    require(not arguments.output.exists(), "Independent review output must be new; never overwrite a failed audit")
    result = audit(arguments.run.resolve(), training_only=arguments.training_only)
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "checks": result["check_count"], "failure": result.get("failure")}))
    if result["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__":
    main()
