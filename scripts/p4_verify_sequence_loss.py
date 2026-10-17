"""P4-01: frozen-source mathematical and real-window sequence-loss witnesses.

This runner makes disposable optimizer updates only. It neither starts formal
S1/SK training nor evaluates planning, alignment, validation selection or test.
Existing P3 data, normalizer, selected checkpoints and run inventories must be
identical before and after execution. Every output uses a new run directory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import subprocess
import sys
from time import perf_counter
import tomllib
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import torch

from push_align.config import ROOT
from push_align.data import digest
from push_align.dynamics import predict_next
from push_align.learning_data import load_learning_data
from push_align.normalization import fit_normalizer
from push_align.preparation import REGIME_CODES, SPLIT_CODES
from push_align.representation import encode, state_delta
from push_align.sequence_training import masked_free_rollout, sequence_loss, sequence_update
from push_align.training import training_update
from p2_episode_contract_check import source_bundle
from p2_train_pilot import parameter_vector, save, tensor_tree_equal
from p3_geometry_pilot import verify_artifacts
from p3_train_common_mlp import load_common_checkpoint, validate_settings
from p3_train_ridge import array_hash

LEARNING = ROOT / "runs/p3-02-20261009-01/learning_data"
COMMON = ROOT / "runs/p3-04-20261009-01"
FORK_CONFIG = ROOT / "configs/p4_fork_budget.v2.json"
FORK_SHA256 = "9054a25951e1e771c62fbb101cc900c3b5f2cf06fc67eed4e21d1eb9e0e95a70"
UPSTREAM = tuple(ROOT / f"runs/{name}" for name in (
    "p3-02-20261009-01", "p3-03-20261009-01",
    "p3-04-20261009-01", "p3-05-20261010-01"))
STRATA_COUNTS = {"noncontact": 22, "sustained_contact": 21, "transition": 21}
GIB = 1024 ** 3


def inventory(folder):
    """Include additions, even files not listed in the original hash index."""
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())


def upstream_state():
    return {str(p.relative_to(ROOT)): {
        "registered_artifacts": verify_artifacts(p), "file_inventory": inventory(p)
    } for p in UPSTREAM}


def snapshot_sources(out):
    hashes = source_bundle(out)
    extra = list((ROOT / "configs").glob("*.json"))
    extra += [ROOT / "docs/04_세부실행계획.md", ROOT / "docs/07_단계별_보고서_작성기준.md"]
    for path in extra:
        rel = str(path.relative_to(ROOT))
        if rel not in hashes:
            target = out / "source" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            hashes[rel] = digest(path)
    return hashes


def selected_batch(prepared):
    """First table-ordered windows in each predeclared first-transition stratum.

    This deterministic verification subset is neither a stochastic P4-02 epoch
    sampler nor a performance estimate. Overlapping windows remain overlapping.
    """
    a = prepared.arrays
    train_windows = prepared.windows[5][a["split_code"][prepared.windows[5][:, 0]] == SPLIT_CODES["train"]]
    if len(train_windows) != 2960 or not np.all(a["split_code"][train_windows] == SPLIT_CODES["train"]):
        raise ValueError("Frozen train K5 windows changed")
    parts = []
    strata = {}
    for name, count in STRATA_COUNTS.items():
        eligible = np.flatnonzero(a["regime_code"][train_windows[:, 0]] == REGIME_CODES[name])
        if len(eligible) < count:
            raise ValueError(f"Insufficient real verification windows: {name}")
        chosen = eligible[:count]
        parts.append(train_windows[chosen])
        strata[name] = {"eligible_windows": len(eligible), "selected_windows": count,
                        "original_train_window_indices": chosen.tolist()}
    rows = np.concatenate(parts)
    if rows.shape != (64, 5) or len(np.unique(rows[:, 0])) != 64:
        raise ValueError("Duplicate or malformed selected windows")
    if not np.array_equal(a["next_state"][rows[:, :-1]], a["current_state"][rows[:, 1:]]):
        raise ValueError("Real verification sequence is discontinuous")
    states = np.concatenate((a["current_state"][rows[:, :1]], a["next_state"][rows]), axis=1)
    actions = a["applied_action"][rows].copy()
    mask = np.ones((64, 5), dtype=np.bool_)
    selection = {"rule": "first table-ordered train windows by first-transition regime; 22/21/21",
                 "strata": strata, "window_rows": rows.tolist(), "window_rows_sha256": array_hash(rows),
                 "episode_index": a["episode_index"][rows].tolist(),
                 "transition_index": a["transition_index"][rows].tolist(),
                 "distinct_transition_rows": len(np.unique(rows)),
                 "distinct_episode_count": len(np.unique(a["episode_index"][rows[:, 0]])),
                 "scope": "representation/gradient witness subset; overlapping windows not independent samples"}
    return rows, torch.from_numpy(states), torch.from_numpy(actions), torch.from_numpy(mask), selection


def traced_loss(model, normalizer, states, actions, mask, weight):
    inputs = []
    outputs = []
    hook = model.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().clone()))
    output_hook = model.register_forward_hook(lambda module, args, result: outputs.append(result))
    try:
        terms = sequence_loss(model, normalizer, states, actions, mask, weight)
    finally:
        hook.remove()
        output_hook.remove()
    # L_one has one flattened real-transition call, followed by rollout calls.
    if weight > 0 and len(inputs) != 6:
        raise ValueError("Expected one L_one forward and five free-rollout forwards")
    return terms, inputs, outputs


class RelativeToy(torch.nn.Module):
    """Pure mathematical recurrence x_next=x+a*(p-x), p=1, a=1/4.

    Artificial identity-normalized numbers prove calculus and indexing only;
    they are deliberately not simulated contact dynamics or robotic evidence.
    """
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(.25, dtype=torch.float32))

    def forward(self, features):
        first = self.a * features[..., :1]
        return torch.cat((first, torch.zeros((*features.shape[:-1], 9), dtype=torch.float32)), dim=-1)


def toy_witness(schema, out):
    # A legitimate schema-bound synthetic normalizer with exact mean 0/scale 1.
    signs = np.array([-1., 1.], dtype=np.float64)
    normalizer = fit_normalizer(np.repeat(signs[:, None], 11, axis=1),
        np.repeat(signs[:, None], 10, axis=1), np.zeros(2, np.int8), schema,
        {"dataset_id": "p4-01-analytic-toy-only", "manifest_sha256": "a" * 64,
         "data_contract_hash": schema["data_contract_hash"]},
        np.zeros(2, np.int64), np.arange(2, dtype=np.int64))
    states = torch.zeros((2, 6, 10), dtype=torch.float64)
    states[:, :, 6] = 1.
    actions = torch.zeros((2, 5, 2), dtype=torch.float64)
    mask = torch.tensor([[True] * 5, [True] * 3 + [False] * 2])
    states[1, 4:] = float("nan")
    actions[1, 3:] = float("nan")
    model = RelativeToy()
    terms = sequence_loss(model, normalizer, states, actions, mask, 1.)
    predicted = terms.predicted_states.detach().numpy()
    a, q = .25, .75
    ks = [1, 2, 3, 4, 5, 1, 2, 3]
    expected_one = a * a / 10
    expected_roll = sum((1 - q ** k) ** 2 for k in ks) / 80
    expected_roll_grad = sum(2 * (1 - q ** k) * k * q ** (k - 1) for k in ks) / 80
    roll_grad = float(torch.autograd.grad(terms.rollout, model.a)[0])
    padded_states = torch.cat((states, torch.full((2, 2, 10), float("nan"), dtype=torch.float64)), dim=1)
    padded_actions = torch.cat((actions, torch.full((2, 2, 2), float("nan"), dtype=torch.float64)), dim=1)
    padded_mask = torch.cat((mask, torch.zeros((2, 2), dtype=torch.bool)), dim=1)
    padded = sequence_loss(model, normalizer, padded_states, padded_actions, padded_mask, 1.)
    original_initial = states[:1, 0].clone()
    valid_actions = actions[:1]
    path = [original_initial]
    for k in range(5):
        path.append(predict_next(model, normalizer, path[-1], valid_actions[:, k]))
    final_cost = path[-1][0, 0].square()
    early_vjp = torch.autograd.grad(final_cost, path[1], retain_graph=True)[0]
    final_grad = float(torch.autograd.grad(final_cost, model.a)[0])
    detached_path = [original_initial]
    for k in range(5):
        detached_path.append(predict_next(model, normalizer, detached_path[-1].detach(), valid_actions[:, k]))
    detached_grad = float(torch.autograd.grad(detached_path[-1][0, 0].square(), model.a)[0])
    teacher_forced = torch.stack([predict_next(model, normalizer, states[:, k][mask[:, k]], actions[:, k][mask[:, k]])[:, 0]
                                 for k in range(3)])
    # More directly, all eight teacher-forced endpoint x values are exactly a.
    forced_roll = a * a / 10
    expected_final_grad = 2 * (1 - q ** 5) * 5 * q ** 4
    expected_detached_grad = 2 * (1 - q ** 5) * q ** 4
    # Central difference has float32 resolution; h is explicit and bounded.
    h = 2 ** -12
    finite_costs = []
    for value in (a + h, a - h):
        probe = RelativeToy()
        with torch.no_grad():
            probe.a.fill_(value)
        cost = masked_free_rollout(probe, normalizer, original_initial, valid_actions,
                                   torch.ones((1, 5), dtype=torch.bool))[0, -1, 0].square()
        finite_costs.append(float(cost.detach()))
    finite_gradient = (finite_costs[0] - finite_costs[1]) / (2 * h)
    checks = {
        "closed_form_k1_to_k5_exact": np.array_equal(predicted[0, :, 0], np.array([1 - q ** k for k in range(6)])),
        "valid_prefix_lengths5_and3_count8_scalar80": terms.valid_transitions == 8 and terms.scalar_terms == 80,
        "k0_and_padding_excluded_from_mean": float(terms.rollout.detach()) == expected_roll and float(terms.one_step.detach()) == expected_one,
        "nan_padding_never_enters_valid_loss": torch.equal(terms.total, padded.total) and padded.scalar_terms == 80,
        "rollout_gradient_matches_closed_form": abs(roll_grad - expected_roll_grad) < 2e-7,
        "terminal_gradient_matches_full_recurrence": abs(final_grad - expected_final_grad) < 2e-7,
        "later_loss_reaches_first_predicted_state": abs(float(early_vjp[0, 0]) - expected_detached_grad) < 2e-7,
        "central_difference_matches_terminal_gradient": abs(finite_gradient - expected_final_grad) < 5e-4,
        "detach_negative_control_detected": abs(detached_grad - expected_detached_grad) < 2e-7 and abs(detached_grad - final_grad) > .1,
        "teacher_forcing_negative_control_detected": bool(torch.equal(teacher_forced, torch.full_like(teacher_forced, a))) and abs(forced_roll - expected_roll) > .01,
    }
    witness = {"checks": {k: bool(v) for k, v in checks.items()}, "pass": all(checks.values()),
        "recurrence": "x(k+1)=x(k)+a*(1-x(k)); x0=0; a=1/4; xk=1-(3/4)^k",
        "mask_lengths": [5, 3], "valid_terms": 8, "denominator": 80,
        "one_step": float(terms.one_step.detach()), "rollout": float(terms.rollout.detach()),
        "expected_rollout": expected_roll, "rollout_gradient": roll_grad, "expected_rollout_gradient": expected_roll_grad,
        "terminal_gradient": final_grad, "expected_terminal_gradient": expected_final_grad,
        "terminal_to_first_predicted_x_gradient": float(early_vjp[0, 0]),
        "detached_negative_gradient": detached_grad, "expected_detached_gradient": expected_detached_grad,
        "teacher_forcing_negative_rollout": forced_roll, "central_difference": finite_gradient, "central_difference_h": h,
        "scope": "analytic synthetic recurrence; proof of calculus/mask/indexing, no robot behavior or accuracy claim"}
    np.savez_compressed(out / "toy_trace.npz", predicted=predicted, mask=mask.numpy(),
                        early_state_vjp=early_vjp.detach().numpy())
    save(out / "toy_witness.json", witness)
    if not witness["pass"]:
        raise ValueError("Analytic sequence witness failed")
    return witness


def seed_witness(entry, prepared, normalizer, common_manifest, settings, states, actions, mask, out):
    seed = entry["seed"]
    path = ROOT / entry["path"]
    config, provenance = common_manifest["physical_config"], common_manifest["provenance"]
    selection = json.loads((COMMON / f"seeds/seed-{seed:02d}/selection.json").read_text())
    if (entry["checkpoint_hash"] != selection["selected_checkpoint_hash"]
            or entry["initial_updates"] != selection["selected_updates"]):
        raise ValueError("Fork checkpoint does not match immutable P3 selection")
    def restore():
        bundle = load_common_checkpoint(path, normalizer, config, provenance, settings, seed,
                                        epoch=selection["selected_epoch"], resume=True)
        if bundle.metadata["checkpoint_hash"] != entry["checkpoint_hash"]:
            raise ValueError("Unexpected common checkpoint hash")
        return bundle
    base = restore()
    terms, calls, production_outputs = traced_loss(base.model, normalizer, states, actions, mask, 1.)
    with torch.no_grad():
        one = sequence_loss(base.model, normalizer, states, actions, mask, 0.)
    rollout_calls = calls[1:]
    expected_calls = [normalizer.normalize_features(encode(terms.predicted_states[:, k], actions[:, k])).to(torch.float32).detach()
                      for k in range(5)]
    mutated = states.clone()
    mutated[:, 1:, 0] += .01
    mutated[:, 1:, 2] += .01
    altered, altered_calls, _ = traced_loss(base.model, normalizer, mutated, actions, mask, 1.)
    scale = torch.tensor(normalizer.arrays["delta_scale"], dtype=torch.float64)
    production_terminal = (state_delta(states[:, -1], terms.predicted_states[:, -1]) / scale).square().mean()
    # Retain the actual first rollout forward output, which is an ancestor of
    # k5. A slice of the subsequently stacked states is not that graph node.
    production_early_vjp = torch.autograd.grad(production_terminal, production_outputs[1], retain_graph=True)[0]
    # A separate explicit graph retains the real first predicted state node.
    path_states = [states[:, 0]]
    for k in range(5):
        path_states.append(predict_next(base.model, normalizer, path_states[-1], actions[:, k]))
    residual = state_delta(states[:, -1], path_states[-1])
    terminal_loss = (residual / scale).square().mean()
    early_vjp = torch.autograd.grad(terminal_loss, path_states[1], retain_graph=True)[0]
    rollout_grads = torch.autograd.grad(terms.rollout, tuple(base.model.parameters()))
    grad_vector = torch.cat([g.detach().reshape(-1) for g in rollout_grads])
    originals = [restore() for _ in range(4)]
    initial_weights = deepcopy(originals[0].model.state_dict())
    initial_adam = deepcopy(originals[0].optimizer.state_dict())
    identical_start = all(tensor_tree_equal(initial_weights, b.model.state_dict())
                          and tensor_tree_equal(initial_adam, b.optimizer.state_dict())
                          and torch.equal(originals[0].torch_rng_state, b.torch_rng_state)
                          and torch.equal(originals[0].shuffle_generator.get_state(), b.shuffle_generator.get_state())
                          for b in originals[1:])
    diagnostics = []
    with torch.random.fork_rng(devices=[]):
        for i, b in enumerate(originals):
            torch.set_rng_state(b.torch_rng_state)
            before = parameter_vector(b.model)
            if i == 0:
                result = training_update(b.model, b.optimizer, normalizer,
                    encode(states[:, :-1][mask], actions[mask]),
                    state_delta(states[:, :-1][mask], states[:, 1:][mask]), config["training"]["gradient_clip_norm"])
                label = "original_one_step_flat_reference"
            else:
                weight = (0., .1, 1.)[i - 1]
                result = sequence_update(b.model, b.optimizer, normalizer, states, actions, mask,
                                         weight, config["training"]["gradient_clip_norm"])
                label = f"sequence_lambda{weight:g}"
            diagnostics.append({"diagnostic": label, **result,
                "parameters_changed": not torch.equal(before, parameter_vector(b.model)),
                "max_parameter_change": float((before - parameter_vector(b.model)).abs().max()),
                "adam_steps": sorted({int(s["step"]) for s in b.optimizer.state_dict()["state"].values()}),
                "formal_additional_updates": 0})
    checks = {
        "common_checkpoint_selected_epoch_and_hash_match": True,
        "four_disposable_copies_weights_adam_rng_identical_at_start": identical_start,
        "same_L_one_exact_at_same_weights_S1_and_SK": torch.equal(one.one_step, terms.one_step),
        "all_5_real_transitions_per_window_scalar3200": terms.valid_transitions == 320 and terms.scalar_terms == 3200,
        "k0_comes_from_real_initial_state_exact": torch.equal(terms.predicted_states[:, 0], states[:, 0]),
        "each_rollout_call_uses_previous_full_predicted_state": all(torch.equal(a, b) for a, b in zip(rollout_calls, expected_calls)),
        "future_labels_changed_L_one_input_and_loss": not torch.equal(calls[0], altered_calls[0]) and not torch.equal(terms.one_step, altered.one_step),
        "future_labels_never_change_rollout_model_inputs": all(torch.equal(a, b) for a, b in zip(rollout_calls, altered_calls[1:])),
        "future_labels_never_change_predicted_states": torch.equal(terms.predicted_states, altered.predicted_states),
        "future_labels_are_active_loss_targets": not torch.equal(terms.rollout, altered.rollout),
        "rollout_only_parameter_gradient_finite_nonzero": bool(torch.isfinite(grad_vector).all()) and bool((grad_vector != 0).any()),
        "production_k5_loss_reaches_actual_k1_model_output": bool(torch.isfinite(production_early_vjp).all()) and bool((production_early_vjp != 0).any()),
        "terminal_loss_reaches_first_predicted_full_state": bool(torch.isfinite(early_vjp).all()) and bool((early_vjp != 0).any()),
        "lambda0_next_parameters_and_adam_exact_original_one_step": tensor_tree_equal(originals[0].model.state_dict(), originals[1].model.state_dict()) and tensor_tree_equal(originals[0].optimizer.state_dict(), originals[1].optimizer.state_dict()),
        "lambda0_loss_and_preclip_gradient_exact": diagnostics[0]["loss_before_update"] == diagnostics[1]["loss_before_update"] and diagnostics[0]["gradient_norm_before_clip"] == diagnostics[1]["gradient_norm_before_clip"],
        "all_diagnostic_updates_finite_and_parameters_change": all(d["parameters_changed"] and np.isfinite(d["loss_before_update"]) and np.isfinite(d["gradient_norm_before_clip"]) for d in diagnostics),
        "all_resume_adam_steps_increment_one": all(d["adam_steps"] == [entry["initial_updates"] + 1] for d in diagnostics),
    }
    np.savez_compressed(out / f"seed-{seed:02d}-trace.npz",
        predicted_states=terms.predicted_states.detach().numpy(), real_states=states.numpy(),
        applied_actions=actions.numpy(), mask=mask.numpy(),
        one_step_inputs=calls[0].numpy(), rollout_inputs=np.stack([c.numpy() for c in rollout_calls]),
        altered_future_labels=mutated.numpy(), altered_one_step_inputs=altered_calls[0].numpy(),
        altered_rollout_inputs=np.stack([c.numpy() for c in altered_calls[1:]]),
        terminal_to_first_predicted_state_vjp=early_vjp.detach().numpy(),
        production_terminal_to_first_model_output_vjp=production_early_vjp.detach().numpy(),
        rollout_parameter_gradient=grad_vector.numpy())
    witness = {"seed": seed, "checks": {k: bool(v) for k, v in checks.items()}, "pass": all(checks.values()),
        "common_checkpoint": {**entry, "selected_epoch": selection["selected_epoch"],
            "checkpoint_manifest_sha256": digest(path / "checkpoint.json"), "source_files": base.metadata["files"],
            "optimizer_restored": True, "torch_rng_restored": True, "shuffle_rng_restored": True},
        "normalizer_hash": normalizer.metadata["normalizer_hash"],
        "loss_before_updates": {"L_one": float(terms.one_step.detach()), "L_roll": float(terms.rollout.detach())},
        "rollout_only_gradient_norm": float(torch.linalg.vector_norm(grad_vector)),
        "terminal_to_first_state_vjp_norm": float(torch.linalg.vector_norm(early_vjp).detach()),
        "production_terminal_to_first_model_output_vjp_norm": float(torch.linalg.vector_norm(production_early_vjp).detach()),
        "model_forward_counts": {"L_one": 1, "L_roll": 5},
        "diagnostic_updates": diagnostics, "disposable_updates_count": 4,
        "formal_additional_updates": 0,
        "scope": "real train-sequence calculus and one disposable update per control; no learning-effect or task-performance conclusion"}
    save(out / f"seed-{seed:02d}-witness.json", witness)
    if not witness["pass"]:
        raise ValueError(f"Real sequence witness failed for seed {seed}")
    return witness


def run(args, out):
    clock = perf_counter()
    before_free = shutil.disk_usage(ROOT).free
    if before_free < 2 * GIB:
        raise ValueError("Less than 2 GiB free reserve before P4-01 verification")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if digest(FORK_CONFIG) != FORK_SHA256:
        raise ValueError("Registered P4 fork budget changed; declare another verification version")
    fork = json.loads(FORK_CONFIG.read_text())
    common_manifest = json.loads((COMMON / "manifest.json").read_text())
    settings = validate_settings(tomllib.loads((ROOT / "configs/p3_training.v1.toml").read_text()))
    before = upstream_state()
    save(out / "upstream_before.json", before)
    metadata = json.loads((LEARNING / "normalizer/normalizer.json").read_text())["normalizer"]
    prepared, normalizer = load_learning_data(LEARNING, metadata["schema"], metadata["dataset_identity"], settings["normalizer_hash"])
    if (normalizer.metadata["normalizer_hash"] != common_manifest["normalizer_hash"]
            or normalizer.metadata["stats_hash"] != common_manifest["stats_hash"]
            or common_manifest["provenance"]["learning_manifest_sha256"] != digest(LEARNING / "manifest.json")):
        raise ValueError("Common checkpoint/data/normalizer provenance mismatch")
    rows, states, actions, mask, selection = selected_batch(prepared)
    sources = snapshot_sources(out)
    verification_settings = {"id": "p4-sequence-loss-verification-v1", "K": 5,
        "batch_windows": 64, "first_transition_strata": STRATA_COUNTS,
        "toy_lengths": [5, 3], "positive_update_lambdas": [.1, 1.],
        "disposable_updates_per_seed": 4, "formal_additional_updates": 0,
        "same_L_one": "all valid real transitions / immutable train delta scale",
        "L_roll": "full predicted-state recursion; k1..K; wrap2pi yaw; valid terms x10 denominator",
        "minimum_free_reserve_bytes": 2 * GIB, "max_verification_output_bytes": GIB}
    manifest = {"task_id": "P4-01", "created_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "argv": sys.argv, "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "numpy": str(np.__version__), "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "source_hashes": sources, "verification_settings": verification_settings,
        "frozen_fork_config": fork, "frozen_fork_file_sha256": FORK_SHA256,
        "training_settings": settings, "normalizer_hash": normalizer.metadata["normalizer_hash"],
        "stats_hash": normalizer.metadata["stats_hash"], "learning_manifest_sha256": digest(LEARNING / "manifest.json"),
        "selected_train_windows": selection, "free_disk_before_bytes": before_free,
        "final_test_used": False, "formal_S1_SK": "NOT_RUN", "MPC": "NOT_RUN",
        "alignment_outcome": "NOT_EVALUATED", "personal_understanding": "NOT_ASSESSED"}
    save(out / "manifest.json", manifest)
    manifest_sha = digest(out / "manifest.json")
    np.savez_compressed(out / "selected_train_windows.npz", rows=rows, real_states=states.numpy(),
                        applied_actions=actions.numpy(), mask=mask.numpy())
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                           cwd=ROOT, capture_output=True, text=True,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    log = tests.stdout + tests.stderr
    (out / "tests.txt").write_text(log, encoding="utf-8")
    match = re.search(r"Ran (\d+) tests", log)
    count = int(match.group(1)) if match else 0
    p4_names = [line.split(" ... ")[0] for line in log.splitlines()
                if "test_p4_" in line and line.endswith(" ... ok")]
    regressions = {"returncode": tests.returncode, "test_count": count,
                   "P4_test_names_passed": p4_names, "P4_test_count": len(p4_names),
                   "no_skips": "skipped=" not in log and " ... skipped " not in log}
    save(out / "regression_result.json", regressions)
    if tests.returncode or count < 661 or len(p4_names) != 19 or not regressions["no_skips"]:
        raise ValueError("Required full regressions and independent P4 tests must pass without skips")
    print(json.dumps({"progress": "regression_complete", "tests": count, "P4_tests": len(p4_names)}), flush=True)
    toy = toy_witness(metadata["schema"], out)
    seeds = []
    for entry in fork["common_checkpoints"]:
        witness = seed_witness(entry, prepared, normalizer, common_manifest, settings, states, actions, mask, out)
        seeds.append(witness)
        print(json.dumps({"progress": "real_sequence_witness", "seed": entry["seed"], "pass": witness["pass"]}), flush=True)
    after = upstream_state()
    save(out / "upstream_after.json", after)
    bytes_used = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    free_after = shutil.disk_usage(ROOT).free
    checks = {"all_regressions_and_independent_P4_tests_pass": True,
        "analytic_recurrence_mask_denominator_and_gradient_pass": toy["pass"],
        "detach_and_teacher_forcing_negative_controls_detected": toy["checks"]["detach_negative_control_detected"] and toy["checks"]["teacher_forcing_negative_control_detected"],
        "all_three_common_seed_real_input_and_gradient_witnesses_pass": len(seeds) == 3 and all(s["pass"] for s in seeds),
        "S1_original_loss_and_lambda0_next_update_exact_all_seeds": all(s["checks"]["lambda0_next_parameters_and_adam_exact_original_one_step"] for s in seeds),
        "positive_lambdas_finite_disposable_updates_all_seeds": all(s["checks"]["all_diagnostic_updates_finite_and_parameters_change"] for s in seeds),
        "upstream_registered_artifacts_and_entire_file_inventories_unchanged": before == after,
        "original_data_stats_models_configs_and_sources_unchanged": all(digest(ROOT / rel) == sha for rel, sha in sources.items()),
        "run_manifest_and_snapshot_sources_frozen": digest(out / "manifest.json") == manifest_sha and all(digest(out / "source" / rel) == sha for rel, sha in sources.items()),
        "verification_storage_under_1GiB_and_free_reserve_2GiB": bytes_used < GIB and free_after >= 2 * GIB,
        "no_formal_branch_updates_MPC_or_final_test": True}
    resources = {"total_wall_time_s": perf_counter() - clock,
        "main_peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024),
        "regression_child_peak_rss_MiB": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024),
        "stored_bytes_before_final_reports_and_hash_index": bytes_used,
        "free_disk_before_bytes": before_free, "free_disk_after_bytes": free_after,
        "disposable_optimizer_updates": sum(s["disposable_updates_count"] for s in seeds),
        "formal_additional_updates": 0,
        "scope": "sequential local CPU verification; peak counters are cumulative per process/child, not simultaneous summed RAM",
        "later_stage_budget": "13.016 GiB prior estimate covers later training/P4/P5 plus reserve; remeasure/reestimate before later execution; not guaranteed here"}
    summary = {"task_id": "P4-01", "pass": all(checks.values()),
        "stage_verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "test_count": count, "P4_test_count": len(p4_names), "toy": toy,
        "seeds": seeds, "resources": resources, "next_task": "P4-02",
        "final_test_used": False, "formal_S1_SK": "NOT_RUN", "MPC": "NOT_RUN",
        "alignment_outcome": "NOT_EVALUATED", "personal_understanding": "NOT_ASSESSED",
        "scope": "loss/gradient/mask implementation verification; no training benefit, held-out generalization, alignment advantage or personal understanding claim"}
    save(out / "resources.json", resources)
    save(out / "summary.json", summary)
    if not summary["pass"]:
        raise ValueError("P4-01 mandatory verification checks incomplete")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("Unsafe run ID")
    out = ROOT / "runs" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    try:
        summary = run(args, out)
    except Exception as error:
        save(out / "failure.json", {"task_id": "P4-01", "pass": False,
            "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        print(json.dumps({"run": str(out), "pass": False, "error": str(error)}), flush=True)
        return 1
    finally:
        save(out / "artifact_hashes.json", {"sha256": {
            str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "artifact_hashes.json"}})
    print(json.dumps({"run": str(out), "pass": summary["pass"], "seeds": len(summary["seeds"])}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
