"""Shared train-window materialization and final-update global lambda choice.

Training consumes only original train windows. Lambda choice is a validation
decision across every registered seed, never a best-seed or best-epoch choice.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import torch

from .prediction import _windows


def train_sequences(prepared):
    """Copy checked original train K5 windows into physical sequence tensors.

    The shared window validator rejects excluded gaps, split/episode crossings,
    duplicate windows, and discontinuous truth. No normalization is fitted here.
    The six truth states are initial state plus five recorded next states; no
    model prediction or simulator future is used to construct training labels.
    """
    windows = _windows(prepared, "train", 5)
    if not len(windows):
        raise ValueError("P4 training requires at least one original train K5 window")
    arrays = prepared.arrays
    states = np.concatenate((arrays["current_state"][windows[:, 0], None],
                             arrays["next_state"][windows]), axis=1)
    actions = arrays["applied_action"][windows].copy()
    mask = torch.ones(windows.shape, dtype=torch.bool, device="cpu")
    return (windows.copy(), torch.from_numpy(states.copy()),
            torch.from_numpy(actions), mask)


def _budget_population(fork_budget):
    if not isinstance(fork_budget, dict):
        raise ValueError("Lambda selection requires the frozen fork-budget dictionary")
    seeds = fork_budget.get("seeds")
    weights = fork_budget.get("lambda_candidates")
    updates = fork_budget.get("additional_updates_per_branch")
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Frozen lambda-selection seeds must be unique exact integers")
    if not isinstance(weights, list) or not weights or any(type(weight) is not float or not math.isfinite(weight) or weight <= 0 for weight in weights) or len(set(weights)) != len(weights):
        raise ValueError("Frozen lambda candidates must be unique positive finite floats")
    if type(updates) is not int or updates <= 0:
        raise ValueError("Frozen additional-update budget must be a positive exact integer")
    return seeds, weights, updates


def _score(value):
    if type(value) not in (int, float):
        return None
    try:
        value = float(value)
    except (OverflowError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def choose_lambda(final_records, fork_budget):
    """Choose one lambda from all registered final-update seed scores.

    A malformed population is rejected. An otherwise identified final record
    with a partial update, invalid window, ineligible status, or absent/invalid
    score makes its entire lambda ineligible, while preserving the diagnostic
    reasons. All-ineligible returns FAIL with no selected/fallback lambda.
    """
    seeds, weights, expected_updates = _budget_population(fork_budget)
    if not isinstance(final_records, (list, tuple)):
        raise ValueError("Final lambda records must be a sequence")
    expected = {(seed, weight) for seed in seeds for weight in weights}
    indexed = {}
    for record in final_records:
        if not isinstance(record, dict) or type(record.get("seed")) is not int or type(record.get("lambda")) is not float:
            raise ValueError("Final record identities require an exact integer seed and float lambda")
        identity = (record["seed"], record["lambda"])
        if identity not in expected or identity in indexed:
            raise ValueError("Final lambda records contain an unknown or duplicated seed/lambda identity")
        if type(record.get("additional_updates")) is not int:
            raise ValueError("Final additional-update count must be an exact integer")
        invalid = record.get("K5_invalid_windows")
        if type(invalid) is not int or invalid < 0:
            raise ValueError("Final K5 invalid-window count must be a nonnegative exact integer")
        indexed[identity] = deepcopy(record)
    if set(indexed) != expected:
        raise ValueError("Final lambda records must include every registered seed/lambda exactly once")

    candidates = []
    for weight in sorted(weights):
        reasons, seed_records, scores = [], [], []
        for seed in seeds:
            record = indexed[(seed, weight)]
            score = _score(record.get("selection_score"))
            seed_reasons = []
            if record["additional_updates"] != expected_updates:
                seed_reasons.append("not_registered_final_additional_update")
            if record.get("selection_status") != "eligible":
                seed_reasons.append("ineligible_validation_selection_status")
            if record["K5_invalid_windows"] != 0:
                seed_reasons.append("invalid_K5_windows")
            if score is None:
                seed_reasons.append("absent_nonfinite_negative_or_untyped_selection_score")
            reasons.extend({"seed": seed, "reason": reason} for reason in seed_reasons)
            seed_records.append({
                "seed": seed, "lambda": weight,
                "additional_updates": record["additional_updates"],
                "selection_status": record.get("selection_status") if isinstance(record.get("selection_status"), str) else None,
                "selection_score": score, "K5_invalid_windows": record["K5_invalid_windows"],
                "eligible": not seed_reasons,
            })
            if not seed_reasons:
                scores.append(score)
        eligible = not reasons
        # Scale before summing: even sum(max_float / 3, ...) can overflow from
        # rounding. Every normalized term is in [0, 1], so this mean stays finite.
        peak = max(scores) if eligible else None
        mean = ((math.fsum(score / peak for score in scores) / len(seeds)) * peak
                if eligible and peak else (0. if eligible else None))
        candidates.append({"lambda": weight, "eligible": eligible,
                           "mean_final_validation_K5_score": mean,
                           "seed_records": seed_records, "reasons": reasons})
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected = min(eligible, key=lambda candidate: (candidate["mean_final_validation_K5_score"], candidate["lambda"])) if eligible else None
    return {"pass": selected is not None,
            "selected_lambda": selected["lambda"] if selected is not None else None,
            "candidates": candidates}
