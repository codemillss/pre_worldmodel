"""Read-only, retrospective summaries of the two saved P4 training histories.

Only NumPy arrays and JSON are read. No model, autodiff or optimizer is loaded.
Mixed sampled-batch histories and fixed probes remain separate descriptions;
neither is an estimate of a final model's whole-training-population objective.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np


LOSS_COLUMNS = ("one_step", "rollout", "relative_teacher", "relative_free", "total", "gradient_norm")
TRACE_KEYS = frozenset(("window_positions", "batch_offsets", "batch_data_hashes", "losses",
                        "loss_columns", "update_wall_time_s"))
SNAPSHOT_IDS = ("common_initial", "C0_u0100", "C0_u0500", "C0_u1000",
                "R1_u0100", "R1_u0500", "R1_u1000")
PROBE_COUNTS = {"historical_update1": 64, "D11_zero5": 24,
                "D11_transition": 16, "D11_noncontact": 8}
RELATIVE_METRICS = ("relative_teacher", "relative_free", "weighted_relative_teacher",
                    "weighted_relative_free", "weighted_aux")


class AdamHistoryError(ValueError):
    """Saved history or fixed-probe identity does not meet the frozen contract."""


def _require(condition, message):
    if not condition:
        raise AdamHistoryError(message)


def _hex_hash(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _read(path, sources, *, arrays=False):
    path = Path(path)
    try:
        raw = path.read_bytes()
        if arrays:
            with np.load(BytesIO(raw), allow_pickle=False) as archive:
                value = {key: archive[key].copy() for key in archive.files}
        else:
            value = json.loads(raw)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise AdamHistoryError(f"Cannot read saved source {path}: {error}") from error
    sources.append({"path": str(path), "sha256": sha256(raw).hexdigest()})
    return value


def _validate_trace(trace, arm):
    _require(set(trace) == TRACE_KEYS, f"{arm}: exact training trace fields required")
    losses = trace["losses"]
    _require(losses.dtype == np.float64 and losses.shape == (1000, 6)
             and np.isfinite(losses).all() and bool((losses >= 0).all()),
             f"{arm}: 1000 finite nonnegative float64 loss rows required")
    columns = trace["loss_columns"]
    _require(columns.ndim == 1 and columns.dtype.kind == "U" and columns.tolist() == list(LOSS_COLUMNS),
             f"{arm}: loss columns differ")
    positions, offsets = trace["window_positions"], trace["batch_offsets"]
    _require(positions.dtype == np.int64 and positions.shape == (62992,)
             and bool(((positions >= 0) & (positions < 2960)).all())
             and np.unique(positions).size == 2960,
             f"{arm}: 62992 window occurrences over all 2960 train slots required")
    _require(offsets.dtype == np.int64 and offsets.shape == (1001,)
             and offsets[0] == 0 and offsets[-1] == 62992,
             f"{arm}: batch offsets or window denominator differ")
    windows = np.diff(offsets)
    _require(int((windows == 64).sum()) == 979 and int((windows == 16).sum()) == 21
             and int(windows.sum()) * 5 == 314960,
             f"{arm}: 979 full64 and 21 partial16 K5 batches required")
    hashes = trace["batch_data_hashes"]
    _require(hashes.dtype == np.dtype("U64") and hashes.shape == (1000,)
             and all(_hex_hash(value) for value in hashes.tolist()),
             f"{arm}: 1000 valid batch data hashes required")
    times = trace["update_wall_time_s"]
    _require(times.dtype == np.float64 and times.shape == (1000,)
             and np.isfinite(times).all() and bool((times >= 0).all()),
             f"{arm}: finite nonnegative update times required")
    if arm == "C0":
        _require(np.array_equal(losses[:, 2:4], np.zeros((1000, 2))),
                 "C0: relative placeholders must be zero and remain uncomputed")
    expected = losses[:, 0] + .1 * losses[:, 1]
    if arm == "R1":
        expected = expected + .2 * (losses[:, 2] + .1 * losses[:, 3])
    _require(np.allclose(losses[:, 4], expected, rtol=1e-12, atol=1e-12),
             f"{arm}: recorded historical total objective arithmetic differs")
    return windows


def _means(values, windows):
    return {"update_mean": float(np.mean(values, dtype=np.float64)),
            "window_weighted_mean": float(np.average(values, weights=windows))}


def _loss_vectors(trace, arm):
    values = trace["losses"]
    result = {"one_step": values[:, 0], "weighted_rollout": .1 * values[:, 1],
              "base": values[:, 0] + .1 * values[:, 1], "actual_total": values[:, 4]}
    result.update({name: None for name in RELATIVE_METRICS})
    if arm == "R1":
        result.update(relative_teacher=values[:, 2], relative_free=values[:, 3],
                      weighted_relative_teacher=.2 * values[:, 2],
                      weighted_relative_free=.02 * values[:, 3],
                      weighted_aux=.2 * (values[:, 2] + .1 * values[:, 3]))
    return result


def _block(trace, windows, arm, start, end):
    section = slice(start, end)
    weights, norms = windows[section], trace["losses"][section, 5]
    total_windows = int(weights.sum())
    clipped = int((norms > 1.).sum())
    return {"update_start": start + 1, "update_end": end, "update_count": end - start,
            "batch_counts": {"full64": int((weights == 64).sum()),
                             "partial16": int((weights == 16).sum())},
            "denominators": {"updates": end - start, "window_occurrences": total_windows,
                             "transition_occurrences": total_windows * 5,
                             "base_scalar_terms_per_term": total_windows * 50,
                             "relative_scalar_terms_per_term": total_windows * 10 if arm == "R1" else None},
            "losses": {key: None if value is None else _means(value[section], weights)
                       for key, value in _loss_vectors(trace, arm).items()},
            "clipping": {"criterion": "recorded preclip gradient_norm > 1; update-based frequency",
                         "threshold": 1., "count": clipped, "fraction": clipped / (end - start),
                         "gradient_norm_median": float(np.median(norms)),
                         "gradient_norm_p90": float(np.quantile(norms, .9, method="linear")),
                         "quantile_method": "linear"}}


def _paired(first, second, windows, start, end):
    section = slice(start, end)
    a, b = _loss_vectors(first, "C0"), _loss_vectors(second, "R1")
    return {"update_start": start + 1, "update_end": end, "update_count": end - start,
            "losses": {key: _means((b[key] - a[key])[section], windows[section])
                       for key in ("one_step", "weighted_rollout", "base")}}


def _fixed_probes(path, trace, sources):
    path = Path(path)
    aggregate = _read(path, sources)
    manifest = _read(path.parent / "manifest.json", sources)
    registration = _read(path.parent / "registration.json", sources)
    batches = _read(path.parent / "batch_manifest.json", sources)
    hashes = {Path(item["path"]).name: item["sha256"] for item in sources[-4:]}
    _require(isinstance(aggregate, dict) and aggregate.get("schema") == "p4-gradient-probe-aggregate-v1"
             and aggregate.get("row_count") == 28 and isinstance(aggregate.get("rows"), list)
             and len(aggregate["rows"]) == 28, "All 28 fixed-probe aggregate rows required")
    _require(isinstance(manifest, dict) and manifest.get("schema") == "p4-gradient-probe-manifest-v1"
             and isinstance(registration, dict)
             and registration.get("schema") == "p4-gradient-probe-protocol-v1",
             "Fixed-probe manifest and registration schemas differ")
    _require(manifest.get("protocol_sha256") == hashes["registration.json"]
             and manifest.get("batch_manifest_sha256") == hashes["batch_manifest.json"],
             "Fixed-probe manifest source binding differs")
    shots = manifest.get("model_snapshots")
    registered = registration.get("snapshots")
    _require(isinstance(shots, list) and isinstance(registered, list) and len(shots) == len(registered) == 7
             and all(isinstance(item, dict) for item in shots + registered),
             "Seven bound frozen snapshots required")
    _require([item.get("snapshot_id") for item in shots] == list(SNAPSHOT_IDS)
             and [item.get("snapshot_id") for item in registered] == list(SNAPSHOT_IDS),
             "Frozen snapshot identities or order differ")
    for index, (shot, original) in enumerate(zip(shots, registered)):
        expected_branch = None if index == 0 else "C0" if index < 4 else "R1"
        expected_update = 0 if index == 0 else (100, 500, 1000)[(index - 1) % 3]
        _require(all(shot.get(key) == value for key, value in original.items())
                 and all(key in original for key in ("branch", "additional_updates", "checkpoint_path",
                                                     "checkpoint_hash", "trained_objective"))
                 and shot.get("branch") == expected_branch and shot.get("additional_updates") == expected_update
                 and isinstance(shot.get("checkpoint_path"), str) and bool(shot["checkpoint_path"])
                 and isinstance(shot.get("trained_objective"), str) and bool(shot["trained_objective"])
                 and _hex_hash(shot.get("checkpoint_hash"))
                 and _hex_hash(shot.get("model_tree_hash"))
                 and _hex_hash(shot.get("checkpoint_json_sha256"))
                 and isinstance(shot.get("files_sha256"), dict)
                 and set(shot["files_sha256"]) == {"weights.pt", "training_state.pt"}
                 and all(_hex_hash(value) for value in shot["files_sha256"].values()),
                 "Frozen checkpoint identity binding differs")
    _require(all(_hex_hash(manifest.get(key)) for key in ("normalizer_hash", "dataset_hash", "probe_batches_sha256")),
             "Fixed-probe dataset/normalizer/batch identities required")
    _require(isinstance(batches, dict) and set(batches) == set(PROBE_COUNTS)
             and isinstance(registration.get("batch_counts"), dict)
             and set(registration["batch_counts"]) == set(PROBE_COUNTS)
             and isinstance(registration.get("batch_selections"), dict)
             and set(registration["batch_selections"]) == set(PROBE_COUNTS),
             "Four bound fixed training probes required")
    for name, count in PROBE_COUNTS.items():
        batch = batches[name]
        _require(isinstance(batch, dict) and batch.get("name") == name
                 and isinstance(batch.get("counts"), dict) and isinstance(batch.get("slots"), list)
                 and batch.get("counts") == registration["batch_counts"][name]
                 and batch["counts"].get("windows") == count
                 and batch.get("slots") == registration["batch_selections"][name]
                 and all(type(slot) is int and 0 <= slot < 2960 for slot in batch["slots"])
                 and len(batch["slots"]) == len(set(batch["slots"])) == count,
                 f"{name}: fixed probe slots/counts differ")
    _require(trace["batch_offsets"][1] == 64
             and batches["historical_update1"]["slots"] == trace["window_positions"][:64].tolist(),
             "Historical update1 probe differs from the saved first training batch")
    identities = {item["snapshot_id"]: item for item in shots}
    rows, seen = [], set()
    required_losses = {"one_step", "rollout", "relative_teacher", "relative_rollout", "total"}
    for row in aggregate["rows"]:
        _require(isinstance(row, dict), "Fixed-probe row must be an object")
        shot, name = row.get("snapshot_id"), row.get("probe_name")
        _require(shot in identities and name in PROBE_COUNTS and (shot, name) not in seen,
                 "Fixed-probe pair duplicated or outside frozen identities")
        seen.add((shot, name))
        values = row.get("loss_values")
        _require(isinstance(values, dict) and set(values) == required_losses
                 and all(type(value) in (int, float) and np.isfinite(value) and value >= 0
                         for value in values.values()), "Finite nonnegative fixed-probe losses required")
        _require(row.get("batch_count") == PROBE_COUNTS[name]
                 and row.get("trained_objective") == identities[shot]["trained_objective"]
                 and row.get("summary_path") == f"probes/{shot}/{name}/summary.json",
                 "Fixed-probe row identity/count/summary path differs")
        base = values["one_step"] + .1 * values["rollout"]
        auxiliary = .2 * (values["relative_teacher"] + .1 * values["relative_rollout"])
        _require(np.isclose(values["total"], base + auxiliary, rtol=1e-12, atol=1e-12),
                 "Fixed-probe common diagnostic objective arithmetic differs")
        rows.append({"snapshot_id": shot, "probe_name": name, "batch_count": row["batch_count"],
                     "trained_objective": row["trained_objective"], "summary_path": row["summary_path"],
                     "loss_values": deepcopy(values), "base": float(base),
                     "weighted_aux": float(auxiliary), "common_hypothetical_total": float(base + auxiliary)})
    return {"row_count": 28, "rows": rows, "snapshot_identities": deepcopy(shots),
            "normalizer_hash": manifest["normalizer_hash"], "dataset_hash": manifest["dataset_hash"],
            "probe_batches_sha256": manifest["probe_batches_sha256"],
            "probe_identities": {name: {key: deepcopy(batch[key])
                                for key in ("name", "slots", "counts", "selection", "overlap") if key in batch}
                                 for name, batch in batches.items()},
            "population_average": None, "statistical_CI": None,
            "scope": "All28 frozen snapshot/probe appearances retained in saved order; overlapping train probes are not a population mean or independent experiments; common total is prospective at C0/common"}


def summarize_histories(C0_npz_path, R1_npz_path, gradient_aggregate_path):
    """Return a JSON-safe summary; read only the two traces and probe JSONs.

    The aggregate's sibling manifest, registration and batch manifest bind its
    frozen identities. Source hashes are checked again before returning. This
    is a new retrospective description, not a preregistered original-training
    analysis and not a replay or assessment of any optimizer update.
    """
    sources = []
    traces = {"C0": _read(C0_npz_path, sources, arrays=True),
              "R1": _read(R1_npz_path, sources, arrays=True)}
    windows = {arm: _validate_trace(trace, arm) for arm, trace in traces.items()}
    for field in ("window_positions", "batch_offsets", "batch_data_hashes"):
        _require(np.array_equal(traces["C0"][field], traces["R1"][field]),
                 f"Matched C0/R1 {field} must be exactly equal")
    intervals = [(start, start + 100) for start in range(0, 1000, 100)]
    arms = {arm: {"historical_objective": "Lone+0.1*Lroll" + ("+0.2*(Lrelative_teacher+0.1*Lrelative_free)" if arm == "R1" else ""),
                  "relative_loss_status": "COMPUTED" if arm == "R1" else "NOT_COMPUTED_BETA_ZERO_PLACEHOLDERS",
                  "whole": _block(trace, windows[arm], arm, 0, 1000),
                  "blocks": [_block(trace, windows[arm], arm, start, end) for start, end in intervals]}
            for arm, trace in traces.items()}
    result = {"schema": "p4-adam-history-summary-v1", "status": "DESCRIPTIVE_COMPLETE",
              "analysis_registration": "Retrospective saved-history description; registration for this new diagnostic is not preregistration before the original training",
              "block_rule": "All ten fixed nonoverlapping 100-update blocks, existing checkpoint cadence; no selected best checkpoint",
              "weighting": {"update_mean": "Each of the recorded updates has equal weight",
                            "window_weighted_mean": "Batch loss weighted by recorded window occurrences; K5 masks were fixed all-valid, so transition weighting is equivalent",
                            "population_limit": "Along-path sampled-batch losses at changing weights; not a fixed final-model whole-train objective"},
              "trace_identity_checks": {"window_positions_exact": True, "batch_offsets_exact": True,
                                        "batch_data_hashes_exact": True, "population": 2960,
                                        "window_occurrences_per_arm": 62992, "transition_occurrences_per_arm": 314960,
                                        "full64_per_arm": 979, "partial16_per_arm": 21},
              "arms": arms,
              "paired_shared_base": {"difference": "R1 minus C0", "unlike_total_objectives_compared": False,
                    "whole": _paired(traces["C0"], traces["R1"], windows["C0"], 0, 1000),
                    "blocks": [_paired(traces["C0"], traces["R1"], windows["C0"], start, end) for start, end in intervals]},
              "fixed_probe_losses": _fixed_probes(gradient_aggregate_path, traces["C0"], sources),
              "source_identities": sources, "source_files_unchanged": True,
              "new_model_or_gradient_or_optimizer_or_physics_execution": False,
              "model_selection": "NONE", "P5_01": "HOLD"}
    for item in sources:
        _require(sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"],
                 "Source changed during retrospective history reading")
    return result
