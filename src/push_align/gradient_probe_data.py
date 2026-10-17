"""Read-only, train-only provenance and deterministic D11 gradient probes.

Selections use original exposure positions, actual recorded contact labels and
applied commands only. No model result, development trajectory, simulator,
normalizer fitting, Torch tensor, or optimizer is involved here. The probes
overlap in families and are descriptive subsets, not independent experiments.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

RELATIVE_RUN = "runs/p4-relative-loss-20261014-01"
DATA_RUN = "runs/p4-data-retrain-20261013-02"
LEARNING = "runs/p3-02-20261009-01/learning_data"
COLLECTION = "runs/p4-data-main-20261013-01"
HISTORICAL_TRACE = "runs/p4-02-20261012-01/seeds/seed-11/SK-lambda0.1/training_trace.npz"
REGIMES = ("noncontact", "sustained_contact", "transition")
PROBE_NAMES = ("historical_update1", "D11_zero5", "D11_transition", "D11_noncontact")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def contact_regimes(summary):
    """Use the existing audited force-summary fraction; do not infer from gap."""
    summary = np.asarray(summary)
    require(summary.dtype == np.float64 and summary.shape == (20, 4)
            and np.isfinite(summary).all(), "Finite float64 core contact summary required")
    fraction = summary[:, 2]
    require(np.all((fraction >= 0.) & (fraction <= 1.)), "Contact fraction outside [0,1]")
    return np.where(fraction == 0., REGIMES[0],
                    np.where(fraction == 1., REGIMES[1], REGIMES[2]))


def _family_first(records):
    chosen = {}
    for record in sorted(records, key=lambda row: (row["family_id"], row["window_start"], row["slot"])):
        chosen.setdefault(record["family_id"], record["slot"])
    return list(chosen.values())


def select_probe_slots(provenance, historical_slots):
    """Deterministic label-only selection; pure helper for independent tests."""
    require(isinstance(provenance, (list, tuple)) and provenance, "Nonempty train provenance required")
    indexed = {}
    for row in provenance:
        slot = row["slot"]
        require(type(slot) is int and slot >= 0 and slot not in indexed, "Unique exact integer slots required")
        require(row["split"] == "train" and row["training_eligible"] is True,
                "Nontraining window in probe population")
        require(len(row["actual_contact_regimes"]) == len(row["transition_ids"]) == 5
                and set(row["actual_contact_regimes"]) <= set(REGIMES), "Five known actual regimes required")
        require(len(set(row["transition_ids"])) == 5, "A window repeats an original transition")
        indexed[slot] = row
    require(set(indexed) == set(range(len(provenance))), "Window slots must be a complete zero-based population")
    require(isinstance(historical_slots, (list, tuple)) and historical_slots
            and all(type(slot) is int and slot in indexed for slot in historical_slots)
            and len(set(historical_slots)) == len(historical_slots), "Historical anchor slots invalid or repeated")
    new = [row for row in provenance if row["source"] == "new_core"]
    for row in new:
        require(row["role"] == "core" and row["y_condition"] == "Y1"
                and row["stop_condition"] == "S1", "New probe provenance is not D11 core")
    zero = [row for row in new if row["all_zero_actions"]]
    require(all(row["window_start"] == 10 for row in zero), "D11 zero5 windows must start at actual core state10")
    require(len(zero) == len({row["family_id"] for row in zero}), "Multiple zero5 windows in one family")
    transition = [row for row in new if REGIMES[2] in row["actual_contact_regimes"]]
    noncontact = [row for row in new if set(row["actual_contact_regimes"]) == {REGIMES[0]}]
    selected = {
        "historical_update1": list(historical_slots),
        "D11_zero5": [row["slot"] for row in sorted(zero, key=lambda row: row["family_id"])],
        "D11_transition": _family_first(transition),
        "D11_noncontact": _family_first(noncontact),
    }
    require(all(selected.values()), "Every registered probe must have actual train support")
    return selected


def _description(name):
    return {
        "historical_update1": "Exact historical update1 slot order; exposure anchor, not population-representative inference",
        "D11_zero5": "Every new train family once, all five applied commands zero, actual core state10 start",
        "D11_transition": "Each new train family having an actual transition once; earliest eligible K5 start, then slot tie-break",
        "D11_noncontact": "Each new train family having a fully noncontact K5 once; earliest eligible start, then slot tie-break",
    }[name]


def summarize_probes(provenance, slots_by_name, *, source_sha256=None):
    """JSON-safe descriptions, original-transition denominators and overlap."""
    rows = {row["slot"]: row for row in provenance}
    output = {}
    for name, slots in slots_by_name.items():
        require(name in PROBE_NAMES and slots and len(set(slots)) == len(slots), "Invalid probe selection")
        selected = [deepcopy(rows[slot]) for slot in slots]
        families = sorted({row["family_id"] for row in selected})
        transitions = {item for row in selected for item in row["transition_ids"]}
        output[name] = {
            "name": name, "slots": list(slots), "families": families,
            "window_provenance": selected, "selection": _description(name),
            "counts": {
                "windows": len(slots), "families": len(families),
                "transition_occurrences": 5 * len(slots),
                "unique_original_transitions": len(transitions),
                "source_windows": dict(sorted(Counter(row["source"] for row in selected).items())),
                "actual_regime_occurrences": dict(sorted(Counter(regime for row in selected for regime in row["actual_contact_regimes"]).items())),
                "geometry_windows": dict(sorted(Counter(row["geometry"] for row in selected).items())),
                "all_zero_windows": sum(row["all_zero_actions"] for row in selected),
            },
            "source_sha256": deepcopy(source_sha256 or {}),
            "dataset": RELATIVE_RUN + "/prepared_training.npz",
            "scope": "Train-only fixed probes shared across snapshots; no outcome-based selection or independent-sample claim",
            "control_limit": "New transition/noncontact strata have different geometries; not a matched causal control",
        }
    for name, item in output.items():
        own_transitions = {tid for row in item["window_provenance"] for tid in row["transition_ids"]}
        item["overlap"] = {}
        for other, alternate in output.items():
            if other == name:
                continue
            other_transitions = {tid for row in alternate["window_provenance"] for tid in row["transition_ids"]}
            item["overlap"][other] = {
                "shared_slots": sorted(set(item["slots"]) & set(alternate["slots"])),
                "shared_families": sorted(set(item["families"]) & set(alternate["families"])),
                "shared_unique_original_transitions": len(own_transitions & other_transitions),
            }
    return output


def build_probe_batches(root):
    """Validate the frozen D11 tensors, reconstruct every train slot, select probes.

    Returns JSON-safe ``{name: {slots, families, window_provenance, ...}}``.
    Tensor materialization belongs to the guarded runner. Only the 24 referenced
    D11 train cores are opened; no development/warmup raw episode is loaded.
    Original prepared files contain other split rows, but only checked train
    row indices enter reconstruction, labels or probes.
    """
    root = Path(root).resolve()
    sources = {}

    def bound(rel, expected=None):
        path = (root / rel).resolve()
        require(path.is_relative_to(root) and path.is_file(), "Missing or escaped bound source: " + rel)
        value = _sha(path)
        if expected is not None:
            require(value == expected, "Bound source hash differs: " + rel)
        sources[path.relative_to(root).as_posix()] = value
        return path

    def read(rel, expected=None):
        return json.loads(bound(rel, expected).read_text(),
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

    def load(rel, expected=None, keys=None):
        with np.load(bound(rel, expected), allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in (archive.files if keys is None else keys)}

    manifest = read(RELATIVE_RUN + "/manifest.json")
    read(RELATIVE_RUN + "/registration.json", manifest["registration_sha256"])
    binding = read(RELATIVE_RUN + "/dataset_bindings.json", manifest["dataset_bindings_sha256"])
    require(binding["source"] == DATA_RUN + "/prepared_training.npz"
            and binding["development_episodes_loaded_for_training"] == 0
            and binding["warmup_episodes_loaded_for_training"] == 0
            and binding["normalizer_refit"] is False, "Unexpected D11 source or contamination")
    old_provenance = binding["original_training_provenance"]
    require(old_provenance["arms"]["D11"] == manifest["training_provenance"]["D11"]
            and binding["data_hash"] == manifest["training_provenance"]["D11"]["data_hash"],
            "D11 dataset identity differs")
    frozen = load(RELATIVE_RUN + "/prepared_training.npz", manifest["prepared_training_sha256"])
    prepared = load(binding["source"], binding["source_sha256"],
                    ("old_window_rows", "replacement_slots", "D11_states", "D11_actions", "D11_mask"))
    for key, shape, dtype in (("states", (2960, 6, 10), np.float64),
                              ("actions", (2960, 5, 2), np.float64),
                              ("mask", (2960, 5), np.bool_)):
        value = frozen[key]
        require(value.shape == shape and value.dtype == dtype and np.isfinite(value).all()
                and np.array_equal(value, prepared["D11_" + key]), "Frozen D11 tensor differs: " + key)
    require(frozen["mask"].all(), "D11 probe population contains invalid recorded transitions")
    original_meta = read(LEARNING + "/manifest.json")
    original = load(LEARNING + "/prepared.npz", original_meta["npz_sha256"],
                    ("current_state", "next_state", "applied_action", "split_code",
                     "episode_index", "transition_index", "regime_code"))
    require(original_meta["preparation"]["regime_codes"] == dict(zip(REGIMES, range(3)))
            and original_meta["preparation"]["split_codes"]["train"] == 0, "Original split/regime meanings differ")
    episodes = {row["episode_index"]: row for row in original_meta["preparation"]["episodes"]}
    old_rows = prepared["old_window_rows"]
    require(old_rows.shape == (2960, 5) and old_rows.dtype == np.int64
            and old_rows.min() >= 0 and old_rows.max() < len(original["split_code"]), "Original K5 row matrix invalid")
    require(np.all(original["split_code"][old_rows] == 0)
            and np.all(np.diff(original["transition_index"][old_rows], axis=1) == 1)
            and np.all(original["episode_index"][old_rows] == original["episode_index"][old_rows[:, :1]]),
            "Original windows cross split/episode or skip transitions")
    require(np.array_equal(original["current_state"][old_rows[:, 1:]], original["next_state"][old_rows[:, :-1]]),
            "Original truth continuity differs")
    collection = read(COLLECTION + "/registration.json", old_provenance["collection_registration_sha256"])
    slots = read(COLLECTION + "/slots_bound.json", old_provenance["slots_bound_sha256"])
    inventory = read(COLLECTION + "/artifact_hashes.json")["sha256"]
    summary = read(COLLECTION + "/summary.json", inventory["summary.json"])
    require(summary["status"] == "PASS" and summary["failure"] is None, "Collection not complete")
    entries = slots["entries"]
    replacement = {entry["slot"]: entry for entry in entries}
    require(len(entries) == len(replacement) == 384
            and np.array_equal(np.asarray(list(replacement), np.int64), prepared["replacement_slots"]),
            "Replacement slot identity differs")
    plans = {row["episode_id"]: row for row in collection["plans"]}
    summaries = {row["label"]: row for row in summary["episodes"]}
    cache = {}
    provenance = []
    for slot in range(2960):
        entry = replacement.get(slot)
        if entry is None:
            rows = old_rows[slot]
            episode_index = int(original["episode_index"][rows[0]])
            episode = episodes[episode_index]
            require(episode["split"] == "train", "Original metadata split is not train")
            states = np.concatenate((original["current_state"][rows[:1]], original["next_state"][rows]))
            actions = original["applied_action"][rows]
            regime_codes = original["regime_code"][rows]
            require(np.isin(regime_codes, [0, 1, 2]).all(), "Unknown original actual regime")
            indices = original["transition_index"][rows].astype(int).tolist()
            row = {"source": "old_train", "role": "original_train_transition_table",
                   "episode_id": episode["episode_id"], "family_id": episode["family_id"],
                   "family_rank": None, "geometry": "original_unspecified",
                   "window_start": indices[0], "transition_indices": indices,
                   "actual_contact_regimes": [REGIMES[int(code)] for code in regime_codes],
                   "prepared_row_indices": rows.tolist(), "y_condition": None, "stop_condition": None}
        else:
            episode_id = entry["arm_episodes"]["D11"]
            plan, recorded = plans[episode_id], summaries[episode_id]
            require(plan["split"] == plan["semantic_split"] == recorded["split"] == recorded["semantic_split"] == "train"
                    and plan["training_eligible"] is True and plan["core_training_eligible"] is True
                    and plan["warmup_training_eligible"] is False and plan["development_evaluation_eligible"] is False
                    and recorded["role"] == "core" and plan["y_condition"] == "Y1" and plan["stop_condition"] == "S1"
                    and plan["family_id"] == recorded["family_id"] == entry["family_id"]
                    and plan["family_rank"] == entry["family_rank"], "Train core identity/eligibility differs")
            if episode_id not in cache:
                prefix = COLLECTION + "/" + recorded["dataset"]
                raw_manifest_rel = recorded["dataset"] + "/manifest.json"
                raw_manifest = read(prefix + "/manifest.json", recorded["manifest_sha256"])
                require(recorded["manifest_sha256"] == inventory[raw_manifest_rel]
                        and len(raw_manifest["episodes"]) == 1, "Train raw manifest differs")
                raw_entry = raw_manifest["episodes"][0]
                require(raw_entry["episode_id"] == episode_id and raw_entry["family_id"] == plan["family_id"]
                        and raw_entry["split"] == "train", "Raw train entry differs")
                for key in ("npz", "json"):
                    rel = recorded["dataset"] + "/" + raw_entry[key]
                    expected = old_provenance["raw_core_artifact_sha256"][rel]
                    require(expected == inventory[rel] == raw_entry[key + "_sha256"], "Original raw core binding differs")
                    bound(COLLECTION + "/" + rel, expected)
                raw = load(prefix + "/" + raw_entry["npz"], raw_entry["npz_sha256"],
                           ("state", "action_applied", "action_requested", "transition_valid", "actual_dt", "contact_summary"))
                sidecar = read(prefix + "/" + raw_entry["json"], raw_entry["json_sha256"])
                policy = sidecar["collection_policy"]
                require(sidecar["episode_id"] == episode_id and sidecar["family_id"] == plan["family_id"]
                        and policy["role"] == "core" and policy["semantic_split"] == "train"
                        and policy["training_eligible"] is True and policy["development_evaluation_eligible"] is False
                        and policy["predeclared_plan"] == plan, "Raw sidecar training role differs")
                require(raw["state"].shape == (21, 10) and raw["state"].dtype == np.float64
                        and raw["action_applied"].shape == (20, 2) and raw["action_applied"].dtype == np.float64
                        and np.isfinite(raw["state"]).all() and np.isfinite(raw["action_applied"]).all()
                        and raw["transition_valid"].dtype == np.bool_ and raw["transition_valid"].shape == (20,)
                        and raw["transition_valid"].all()
                        and np.array_equal(raw["action_applied"], raw["action_requested"])
                        and np.array_equal(raw["action_applied"], np.asarray(plan["core_actions"], np.float64))
                        and np.max(np.abs(raw["actual_dt"] - .1)) < 1e-10, "Recorded core tensors/actions differ")
                raw["regimes"] = contact_regimes(raw["contact_summary"])
                cache[episode_id] = raw
            raw = cache[episode_id]
            start = entry["window_start"]
            require(type(start) is int and 0 <= start <= 15
                    and entry["occurrences"] == 21
                    and np.array_equal(old_rows[slot], entry["replaced_original_prepared_rows"]), "New window mapping differs")
            states, actions = raw["state"][start:start + 6], raw["action_applied"][start:start + 5]
            row = {"source": "new_core", "role": "core", "episode_id": episode_id,
                   "family_id": entry["family_id"], "family_rank": entry["family_rank"],
                   "geometry": plan["geometry"], "window_start": start,
                   "transition_indices": list(range(start, start + 5)),
                   "actual_contact_regimes": raw["regimes"][start:start + 5].tolist(),
                   "prepared_row_indices": None, "y_condition": "Y1", "stop_condition": "S1"}
        require(np.array_equal(frozen["states"][slot], states)
                and np.array_equal(frozen["actions"][slot], actions), "Train slot tensor/provenance mismatch")
        row.update({"slot": slot, "split": "train", "training_eligible": True,
                    "all_zero_actions": bool(np.all(actions == 0.)),
                    "all_nonzero_actions": bool(np.all(np.any(actions != 0., axis=1))),
                    "transition_ids": [row["source"] + ":" + row["episode_id"] + ":" + str(i)
                                       for i in row["transition_indices"]]})
        provenance.append(row)
    require(len(cache) == len({row["family_id"] for row in provenance if row["source"] == "new_core"}) == 24,
            "Exactly24 D11 train cores/families required")
    historical = load(HISTORICAL_TRACE, keys=("window_positions", "batch_offsets"))
    offsets, positions = historical["batch_offsets"], historical["window_positions"]
    require(offsets.dtype == positions.dtype == np.int64 and offsets.shape == (1001,)
            and positions.shape == (62992,) and offsets[0] == 0 and offsets[1] == 64,
            "Historical update1 exposure boundary differs")
    anchor = positions[:64].astype(int).tolist()
    selected = select_probe_slots(provenance, anchor)
    require({name: len(slots) for name, slots in selected.items()} == dict(zip(PROBE_NAMES, (64, 24, 16, 8))),
            "Actual train support differs from registered four-probe count")
    indexed = {row["slot"]: row for row in provenance}
    require(all(indexed[slot]["window_start"] == 0 and indexed[slot]["all_nonzero_actions"]
                for name in ("D11_transition", "D11_noncontact") for slot in selected[name]),
            "Transition/control earliest approach windows differ")
    return summarize_probes(provenance, selected, source_sha256=sources)
