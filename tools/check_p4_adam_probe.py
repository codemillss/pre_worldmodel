"""Independent saved-only NumPy oracle for the P4 virtual Adam diagnostic.

The oracle reads serialized arrays and optimizer moments. It never constructs a
learned model, computes a derivative, or applies a parameter update. Its Adam
reference is ideal float64 arithmetic using the saved float32 clipped gradient
and saved float32 moments; production rounding is bounded separately.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EPS32 = float(np.finfo(np.float32).eps)
NORM_FLOOR = 1e-12
COSINE_THRESHOLD = 1e-6
ROUNDING_MULTIPLIER = 64.
ROUNDING_ABSOLUTE_FLOOR = 1e-12
MODES = ("KEEP", "ZERO_M", "COLD")
OBJECTIVES = ("base", "total")
METRICS = ("base", "aux", "total")
HISTORY_COLUMNS = ("one_step", "rollout", "relative_teacher", "relative_free", "total", "gradient_norm")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def vector(value):
    value = np.asarray(value)
    require(value.ndim == 1 and value.size > 0 and value.dtype in (np.float32, np.float64)
            and np.isfinite(value).all(), "Finite nonempty floating vector required")
    return value.astype(np.float64)


def norm(value):
    value = vector(value)
    return float(np.sqrt(np.sum(value * value, dtype=np.float64)))


def direction_metrics(gradient, displacement):
    """A normalized negative derivative: positive means first-order descent."""
    gradient, displacement = vector(gradient), vector(displacement)
    require(gradient.shape == displacement.shape, "Gradient and displacement dimensions differ")
    gradient_norm, displacement_norm = norm(gradient), norm(displacement)
    dot = float(np.sum(gradient * displacement, dtype=np.float64))
    cosine = (None if min(gradient_norm, displacement_norm) <= NORM_FLOOR
              else float(-dot / (gradient_norm * displacement_norm)))
    relation = ("UNDEFINED_NEAR_ZERO_NORM" if cosine is None else
                "DESCENT" if cosine > COSINE_THRESHOLD else
                "ASCENT" if cosine < -COSINE_THRESHOLD else "NEAR_ORTHOGONAL")
    return {"gradient_norm": gradient_norm, "displacement_norm": displacement_norm,
            "dot": dot, "descent_cosine": cosine, "relation": relation}


def ideal_adam(gradient, first_moment, second_moment, step, mode):
    """Independent ideal Adam plus a preregistered cancellation-aware envelope.

    No float32 production implementation is imported or re-executed here. The
    envelope scales with the uncancelled first-moment numerator so a legitimate
    cancellation does not require an outcome-dependent relative tolerance.
    """
    gradient, first_moment, second_moment = map(vector, (gradient, first_moment, second_moment))
    require(gradient.shape == first_moment.shape == second_moment.shape, "Adam vector dimensions differ")
    require(np.all(second_moment >= 0), "Adam second moments must be nonnegative")
    require(type(step) is int and step >= 0, "Adam step must be a nonnegative integer")
    require(mode in MODES, "Unknown moment mode")
    m = np.zeros_like(first_moment) if mode != "KEEP" else first_moment.copy()
    v = np.zeros_like(second_moment) if mode == "COLD" else second_moment.copy()
    next_step = (0 if mode == "COLD" else step) + 1
    m_next = .9 * m + .1 * gradient
    v_next = .999 * v + .001 * np.square(gradient)
    step_size = .001 / (1. - .9 ** next_step)
    denominator = np.sqrt(v_next) / np.sqrt(1. - .999 ** next_step) + 1e-8
    displacement = -step_size * m_next / denominator
    scale = step_size * (.9 * np.abs(m) + .1 * np.abs(gradient)) / denominator
    envelope = ROUNDING_MULTIPLIER * EPS32 * scale + ROUNDING_ABSOLUTE_FLOOR
    require(np.isfinite(displacement).all() and np.isfinite(envelope).all(), "Nonfinite Adam reference")
    return {"displacement": displacement, "coordinate_error_bound": envelope,
            "l2_error_bound": norm(envelope), "next_step": next_step,
            "next_first_moment": m_next, "next_second_moment": v_next}


def displacement_error(actual, reference):
    actual = vector(actual)
    expected, bound = reference["displacement"], reference["coordinate_error_bound"]
    require(actual.shape == expected.shape, "Saved displacement has wrong dimensions")
    error = np.abs(actual - expected)
    return {"pass": bool(np.all(error <= bound) and norm(error) <= reference["l2_error_bound"]),
            "max_absolute_error": float(np.max(error)), "l2_error": norm(error),
            "l2_error_bound": reference["l2_error_bound"],
            "max_coordinate_bound_ratio": float(np.max(error / bound))}


def precision_certificate(gradient, displacement, coordinate_error_bound):
    """Certify the raw derivative sign under the fixed Adam rounding envelope."""
    gradient, displacement, bound = map(vector, (gradient, displacement, coordinate_error_bound))
    require(gradient.shape == displacement.shape == bound.shape and np.all(bound >= 0),
            "Matched vectors and nonnegative coordinate bounds required")
    dot = float(np.sum(gradient * displacement, dtype=np.float64))
    radius = float(np.sum(np.abs(gradient) * bound, dtype=np.float64))
    lower, upper = dot - radius, dot + radius
    sign = ("UNRESOLVED" if min(norm(gradient), norm(displacement)) <= NORM_FLOOR else
            "DESCENT" if upper < 0 else "ASCENT" if lower > 0 else "UNRESOLVED")
    return {"dot_error_radius": radius, "dot_interval": [lower, upper],
            "rounding_stable_sign": sign}


def clip_reference64(gradient, parameter_order):
    gradient = vector(gradient)
    require(len(parameter_order) == 6, "Exactly six parameter blocks required")
    cursor, block_norms = 0, []
    for item in parameter_order:
        require(item["start"] == cursor and item["end"] > cursor
                and item["numel"] == item["end"] - cursor == int(np.prod(item["shape"])),
                "Parameter slices must cover the flat vector exactly")
        block_norms.append(norm(gradient[cursor:item["end"]]))
        cursor = item["end"]
    require(cursor == len(gradient), "Parameter slices do not cover the gradient")
    total_norm = norm(np.asarray(block_norms, np.float64))
    return {"parameter_norms": block_norms, "norm": total_norm,
            "coefficient": float(min(1., 1. / (total_norm + 1e-6)))}


def history_metrics(losses, branch):
    """Independent objective reconstruction, keeping C0 placeholders unobserved."""
    losses = np.asarray(losses)
    require(branch in ("C0", "R1") and losses.dtype == np.float64
            and losses.ndim == 2 and losses.shape[1] == 6
            and np.isfinite(losses).all() and np.all(losses >= 0), "Finite nonnegative six-column history required")
    one, rollout, teacher, free, total, _ = losses.T
    if branch == "C0":
        require(np.all(teacher == 0) and np.all(free == 0), "C0 relative columns must remain uncomputed placeholders")
    base = one + .1 * rollout
    aux = .2 * (teacher + .1 * free) if branch == "R1" else None
    expected = base + aux if aux is not None else base
    require(np.allclose(total, expected, rtol=1e-12, atol=1e-12), "History total objective closure failed")
    return {"one_step": one, "weighted_rollout": .1 * rollout, "base": base,
            "relative_teacher": teacher if aux is not None else None,
            "relative_free": free if aux is not None else None,
            "weighted_relative_teacher": .2 * teacher if aux is not None else None,
            "weighted_relative_free": .02 * free if aux is not None else None,
            "weighted_aux": aux, "actual_total": total}


def history_statistics(losses, batch_counts, branch, start=0, stop=1000):
    """Summarize the fixed interval using both explicit update/window weights."""
    metrics = history_metrics(losses, branch)
    counts = np.asarray(batch_counts)
    require(counts.ndim == 1 and counts.dtype == np.int64 and len(counts) == len(losses)
            and np.isin(counts, [16, 64]).all() and 0 <= start < stop <= len(losses),
            "Matched 16/64 batch counts and nonempty fixed update interval required")
    selected = counts[start:stop]
    windows, updates = int(selected.sum()), stop - start
    observed = {}
    for key, values in metrics.items():
        observed[key] = None if values is None else {
            "update_mean": float(np.sum(values[start:stop], dtype=np.float64) / updates),
            "window_weighted_mean": float(np.sum(values[start:stop] * selected, dtype=np.float64) / windows)}
    gradient_norms = losses[start:stop, 5]
    clipped = int(np.count_nonzero(gradient_norms > 1.))
    return {"update_start": start + 1, "update_end": stop, "update_count": updates,
            "batch_counts": {"full64": int(np.count_nonzero(selected == 64)),
                             "partial16": int(np.count_nonzero(selected == 16))},
            "denominators": {"updates": updates, "window_occurrences": windows,
                "transition_occurrences": 5 * windows, "base_scalar_terms_per_term": 50 * windows,
                "relative_scalar_terms_per_term": 10 * windows if branch == "R1" else None},
            "losses": observed,
            "clipping": {"count": clipped, "fraction": clipped / updates,
                "gradient_norm_median": float(np.quantile(gradient_norms, .5, method="linear")),
                "gradient_norm_p90": float(np.quantile(gradient_norms, .9, method="linear"))}}


def paired_history_statistics(c0_losses, r1_losses, batch_counts, start=0, stop=1000):
    c0, r1 = history_metrics(c0_losses, "C0"), history_metrics(r1_losses, "R1")
    counts = np.asarray(batch_counts)
    require(counts.dtype == np.int64 and counts.shape == (len(c0_losses),)
            and len(c0_losses) == len(r1_losses) and 0 <= start < stop <= len(counts)
            and np.isin(counts, [16, 64]).all(), "Matched paired history counts required")
    selected = counts[start:stop]
    result = {"update_start": start + 1, "update_end": stop, "update_count": stop - start, "losses": {}}
    for key in ("one_step", "weighted_rollout", "base"):
        difference = (r1[key] - c0[key])[start:stop]
        result["losses"][key] = {"update_mean": float(np.sum(difference) / len(difference)),
            "window_weighted_mean": float(np.sum(difference * selected) / np.sum(selected))}
    return result


def contrast_metrics(first, second):
    first, second = vector(first), vector(second)
    require(first.shape == second.shape, "Contrast dimensions differ")
    na, nb = norm(first), norm(second)
    dot = float(np.sum(first * second, dtype=np.float64))
    cosine = None if min(na, nb) <= NORM_FLOOR else float(dot / (na * nb))
    relation = ("UNDEFINED_NEAR_ZERO_NORM" if cosine is None else
                "NEAR_ORTHOGONAL" if abs(cosine) <= COSINE_THRESHOLD else
                "ALIGNED" if cosine > 0 else "OPPOSED")
    return {"first_norm": na, "second_norm": nb, "dot": dot, "cosine": cosine,
            "relation": relation, "second_to_first_norm_ratio": None if na <= NORM_FLOOR else nb / na,
            "difference_l2": norm(second - first)}


def head_indices(parameter_order):
    by_name = {item["name"]: item for item in parameter_order}
    weight, bias = by_name["network.4.weight"], by_name["network.4.bias"]
    require(weight["shape"] == [10, 128] and bias["shape"] == [10], "Registered ten-output head required")
    matrix = np.arange(weight["start"], weight["end"], dtype=np.int64).reshape(weight["shape"])
    offsets = np.arange(bias["start"], bias["end"], dtype=np.int64)
    rows = {"object_xy": [0, 1], "yaw": [2], "object_velocity": [3, 4, 5],
            "pusher_xy": [6, 7], "pusher_velocity": [8, 9]}
    return {key: {"rows": selected, "indices": np.concatenate((matrix[selected].reshape(-1), offsets[selected]))}
            for key, selected in rows.items()}


def tensor_hash(value):
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
            hasher.update(type(item).__name__.encode() + json.dumps(item, sort_keys=True, allow_nan=False).encode())
    visit(value)
    return hasher.hexdigest()


def read_moments(path, parameter_order, expected_step):
    """Read only a frozen tensor serialization; never instantiate its model."""
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(state, dict) and "optimizer" in state, "Serialized optimizer state required")
    optimizer = state["optimizer"]
    require(set(optimizer) == {"state", "param_groups"} and len(optimizer["param_groups"]) == 1,
            "Exactly one Adam state/group required")
    group = optimizer["param_groups"][0]
    require(set(group) == {"lr", "betas", "eps", "weight_decay", "amsgrad", "maximize", "foreach",
            "capturable", "differentiable", "fused", "params", "decoupled_weight_decay"}, "Unexpected Adam group fields")
    require(group["lr"] == .001 and tuple(group["betas"]) == (.9, .999)
            and group["eps"] == 1e-8 and group["weight_decay"] == 0., "Adam hyperparameters changed")
    require(all(group[key] is False for key in ("amsgrad", "maximize", "capturable", "differentiable", "decoupled_weight_decay"))
            and group["foreach"] in (None, False) and group["fused"] in (None, False), "Unsupported Adam option")
    ids = list(range(6))
    require(group["params"] == ids and set(optimizer["state"]) == set(ids)
            and all(type(key) is int for key in optimizer["state"]), "Adam parameter ids changed")
    first, second = [], []
    for index, item in enumerate(parameter_order):
        moments = optimizer["state"][index]
        require(set(moments) == {"step", "exp_avg", "exp_avg_sq"}, "Unexpected Adam moment fields")
        step = moments["step"]
        require(torch.is_tensor(step) and step.device.type == "cpu" and step.dtype == torch.float32
                and step.numel() == 1 and not step.requires_grad and float(step) == expected_step,
                "Adam state age mismatch")
        for field, output in (("exp_avg", first), ("exp_avg_sq", second)):
            tensor = moments[field]
            require(torch.is_tensor(tensor) and tensor.device.type == "cpu" and tensor.dtype == torch.float32
                    and tensor.layout == torch.strided and not tensor.requires_grad
                    and list(tensor.shape) == item["shape"] and bool(torch.isfinite(tensor).all()),
                    "Saved Adam moment shape, precision or finiteness mismatch")
            output.append(tensor.detach().numpy().reshape(-1).copy())
    first, second = np.concatenate(first), np.concatenate(second)
    require(np.all(second >= 0), "Saved Adam second moment is negative")
    return {"first_moment": first, "second_moment": second, "step": expected_step,
            "optimizer_tree_hash": tensor_hash(optimizer)}


class Audit:
    def __init__(self):
        self.checks, self.sources, self.observed, self.precision = [], {}, [], []

    def check(self, name, passed, **details):
        self.checks.append({"name": name, "pass": bool(passed), **details})
        require(passed, name)

    def number(self, name, actual, expected, *, rtol=1e-10, atol=1e-12):
        valid = (actual is None if expected is None else type(actual) in (int, float)
                 and not isinstance(actual, bool) and np.isfinite(actual)
                 and bool(np.isclose(actual, expected, rtol=rtol, atol=atol)))
        self.check(name, valid)

    def tree(self, name, actual, expected):
        """Check every expected material field; explanatory prose may be extra."""
        if isinstance(expected, dict):
            self.check(name + ":fields", isinstance(actual, dict) and set(expected).issubset(actual))
            for key, value in expected.items(): self.tree(name + ":" + key, actual[key], value)
        elif isinstance(expected, list):
            self.check(name + ":length", isinstance(actual, list) and len(actual) == len(expected))
            for index, value in enumerate(expected): self.tree(name + ":" + str(index), actual[index], value)
        elif expected is None or type(expected) in (float, int):
            self.number(name, actual, expected)
        else:
            self.check(name, type(actual) is type(expected) and actual == expected)

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        require(path.is_relative_to(ROOT), "Source outside project")
        relative, digest = path.relative_to(ROOT).as_posix(), sha(path)
        self.check("immutable:" + relative, expected is None or digest == expected)
        if relative in self.sources: self.check("repeat_immutable:" + relative, self.sources[relative] == digest)
        self.sources[relative] = digest
        return path

    def history(self, run):
        document = read(run / "history_summary.json")
        self.check("history_schema", document["schema"] == "p4-adam-history-summary-v1")
        traces = {}
        for branch in ("C0", "R1"):
            source = self.bind(ROOT / "runs/p4-relative-loss-20261014-01/arms" / branch / "training_trace.npz")
            trace = arrays(source)
            self.check(branch + ":history_exact_fields", set(trace) == {"window_positions", "batch_offsets",
                "batch_data_hashes", "losses", "loss_columns", "update_wall_time_s"})
            self.check(branch + ":history_rows_precision", trace["losses"].shape == (1000, 6)
                       and trace["losses"].dtype == np.float64 and trace["loss_columns"].tolist() == list(HISTORY_COLUMNS))
            offsets, positions = trace["batch_offsets"], trace["window_positions"]
            self.check(branch + ":history_occurrence_denominators", offsets.dtype == positions.dtype == np.int64
                       and offsets.shape == (1001,) and offsets[0] == 0 and offsets[-1] == 62992
                       and positions.shape == (62992,) and set(positions.tolist()) == set(range(2960)))
            counts = np.diff(offsets)
            self.check(branch + ":history_batch_denominators", np.count_nonzero(counts == 64) == 979
                       and np.count_nonzero(counts == 16) == 21 and np.isin(counts, [16, 64]).all())
            self.check(branch + ":history_hashes_and_time", trace["batch_data_hashes"].shape == (1000,)
                       and all(len(value) == 64 and set(value) <= set("0123456789abcdef") for value in trace["batch_data_hashes"])
                       and trace["update_wall_time_s"].shape == (1000,)
                       and np.isfinite(trace["update_wall_time_s"]).all() and np.all(trace["update_wall_time_s"] >= 0))
            whole = history_statistics(trace["losses"], counts, branch)
            blocks = [history_statistics(trace["losses"], counts, branch, start, start + 100)
                      for start in range(0, 1000, 100)]
            arm = document["arms"][branch]
            self.check(branch + ":relative_history_observation_status", arm["relative_loss_status"] ==
                       ("COMPUTED" if branch == "R1" else "NOT_COMPUTED_BETA_ZERO_PLACEHOLDERS"))
            self.check(branch + ":historical_objective", arm["historical_objective"] ==
                       "Lone+0.1*Lroll" + ("+0.2*(Lrelative_teacher+0.1*Lrelative_free)" if branch == "R1" else ""))
            self.tree(branch + ":whole_history", arm["whole"], whole)
            self.tree(branch + ":fixed_ten_history_blocks", arm["blocks"], blocks)
            self.check(branch + ":clipping_quantile_policy", all(value["clipping"]["threshold"] == 1.
                       and value["clipping"]["quantile_method"] == "linear" for value in [arm["whole"], *arm["blocks"]]))
            self.observed.append({"history_branch": branch, "updates": 1000, "clipping": whole["clipping"],
                                  "denominators": whole["denominators"]})
            traces[branch] = trace
        first, second = traces["C0"], traces["R1"]
        self.check("paired_history_same_schedule", all(np.array_equal(first[key], second[key])
                   for key in ("window_positions", "batch_offsets", "batch_data_hashes")))
        self.tree("history_trace_identity_record", document["trace_identity_checks"],
            {"window_positions_exact": True, "batch_offsets_exact": True, "batch_data_hashes_exact": True,
             "population": 2960, "window_occurrences_per_arm": 62992, "transition_occurrences_per_arm": 314960,
             "full64_per_arm": 979, "partial16_per_arm": 21})
        counts = np.diff(first["batch_offsets"])
        paired = document["paired_shared_base"]
        self.check("paired_history_subtraction_order", paired["difference"] == "R1 minus C0")
        self.tree("paired_whole_history", paired["whole"], paired_history_statistics(first["losses"], second["losses"], counts))
        self.tree("paired_fixed_ten_history_blocks", paired["blocks"],
            [paired_history_statistics(first["losses"], second["losses"], counts, start, start + 100)
             for start in range(0, 1000, 100)])
        self.check("unlike_objective_totals_not_paired", paired["unlike_total_objectives_compared"] is False)
        prior = ROOT / "runs/p4-gradient-probe-20261015-01"
        aggregate, manifest, batch_manifest = (read(prior / filename) for filename in
                                               ("aggregate.json", "manifest.json", "batch_manifest.json"))
        rows = []
        for row in aggregate["rows"]:
            values = row["loss_values"]
            base = values["one_step"] + .1 * values["rollout"]
            auxiliary = .2 * (values["relative_teacher"] + .1 * values["relative_rollout"])
            rows.append({key: row[key] for key in ("snapshot_id", "probe_name", "batch_count", "trained_objective", "summary_path")}
                        | {"loss_values": values, "base": base, "weighted_aux": auxiliary,
                           "common_hypothetical_total": base + auxiliary})
        fixed = document["fixed_probe_losses"]
        self.tree("all28_fixed_probe_losses", fixed, {"row_count": 28, "rows": rows,
            "snapshot_identities": manifest["model_snapshots"], "normalizer_hash": manifest["normalizer_hash"],
            "dataset_hash": manifest["dataset_hash"], "probe_batches_sha256": manifest["probe_batches_sha256"],
            "probe_identities": {name: {key: batch[key] for key in ("name", "slots", "counts", "selection", "overlap")
                                        if key in batch} for name, batch in batch_manifest.items()},
            "population_average": None, "statistical_CI": None})
        self.check("historical_update1_fixed_batch_identity", batch_manifest["historical_update1"]["slots"] ==
                   first["window_positions"][:64].tolist())
        expected_history_sources = [ROOT / "runs/p4-relative-loss-20261014-01/arms" / branch / "training_trace.npz"
                                    for branch in ("C0", "R1")]
        expected_history_sources += [prior / filename for filename in
                                     ("aggregate.json", "manifest.json", "registration.json", "batch_manifest.json")]
        self.tree("exact_six_history_source_bindings", document["source_identities"],
                  [{"path": str(path), "sha256": sha(path)} for path in expected_history_sources])
        for item in document["source_identities"]: self.bind(Path(item["path"]), item["sha256"])
        self.check("history_readonly_scope", document["source_files_unchanged"] is True
                   and document["new_model_or_gradient_or_optimizer_or_physics_execution"] is False
                   and document["model_selection"] == "NONE" and document["P5_01"] == "HOLD")

    def probe(self, run, spec, name, moments, weight_hash):
        source = ROOT / "runs/p4-gradient-probe-20261015-01/probes" / spec["snapshot_id"] / name
        previous = read(self.bind(source / "summary.json"))
        saved = arrays(self.bind(source / "gradients.npz"))
        raw, base = saved["gradients"], saved["base"]
        self.check(spec["snapshot_id"] + ":" + name + ":raw_gradient_identity", raw.dtype == base.dtype == np.float32
                   and raw.shape == (5, 19338) and base.shape == (19338,) and np.isfinite(raw).all()
                   and np.isfinite(base).all() and np.array_equal(base, (raw[0].astype(np.float64) + .1 * raw[1].astype(np.float64)).astype(np.float32)))
        order = previous["parameter_order"]
        references = {"base": base.astype(np.float64), "aux": .2 * (raw[2].astype(np.float64) + .1 * raw[3].astype(np.float64)),
                      "total": raw[4].astype(np.float64)}
        raw_base = raw[0].astype(np.float64) + .1 * raw[1].astype(np.float64)
        folder = run / "probes" / spec["snapshot_id"] / name
        record, output = read(folder / "summary.json"), arrays(folder / "directions.npz")
        label = spec["snapshot_id"] + ":" + name
        self.tree(label + ":identity", record, {"snapshot_id": spec["snapshot_id"], "probe_name": name,
            "objective_order": list(OBJECTIVES), "mode_order": list(MODES), "parameter_order": order,
            "parameter_count": 19338, "norm_floor": NORM_FLOOR, "near_orthogonal_absolute_cosine": COSINE_THRESHOLD,
            "source_summary_path": (source / "summary.json").relative_to(ROOT).as_posix(),
            "source_summary_sha256": sha(source / "summary.json"), "source_gradients_sha256": sha(source / "gradients.npz"),
            "source_optimizer_tree_hash": moments["optimizer_tree_hash"], "source_weights_tree_hash": weight_hash,
            "no_weights_or_moments_changed": True})
        self.tree(label + ":adam", record["adam"], {"saved_step": moments["step"], "learning_rate": .001,
            "betas": [.9, .999], "epsilon": 1e-8, "weight_decay": 0., "amsgrad": False})
        self.check(label + ":output_arrays", set(output) == {"directions", "clipped_gradients"}
                   and output["directions"].shape == (6, 19338) and output["clipped_gradients"].shape == (2, 19338)
                   and all(value.dtype == np.float32 and np.isfinite(value).all() for value in output.values()))
        deltas, clips = output["directions"], output["clipped_gradients"]
        self.check(label + ":exact_six_direction_records", len(record["directions"]) == 6)
        for objective_index, objective in enumerate(OBJECTIVES):
            original = base if objective == "base" else raw[4]
            clip_record, expected_clip = record["clips"][objective], clip_reference64(original, order)
            self.number(label + ":" + objective + ":clip_norm", clip_record["parameterwise_float32_norm"], expected_clip["norm"], rtol=64 * EPS32)
            self.number(label + ":" + objective + ":flat_float64_norm", clip_record["flat_float64_norm"], norm(original))
            self.number(label + ":" + objective + ":clip_coefficient", clip_record["coefficient"], expected_clip["coefficient"], rtol=64 * EPS32)
            self.check(label + ":" + objective + ":six_norm_blocks", len(clip_record["parameter_norms_float32"]) == 6)
            for block_index, expected_norm in enumerate(expected_clip["parameter_norms"]):
                self.number(label + ":" + objective + ":block_norm:" + str(block_index),
                            clip_record["parameter_norms_float32"][block_index], expected_norm, rtol=64 * EPS32)
            coefficient = clip_record["coefficient"]
            self.check(label + ":" + objective + ":exact_f32_global_clip", 0 < coefficient <= 1
                       and float(np.float32(coefficient)) == coefficient
                       and np.array_equal(clips[objective_index], original * np.float32(coefficient)))
            self.tree(label + ":" + objective + ":clip_contract", clip_record,
                      {"maximum_norm": 1., "epsilon": 1e-6, "would_scale": coefficient < 1.,
                       "applied_to_parameters_or_grad_buffers": False})
            for mode_index, mode in enumerate(MODES):
                index, delta = 3 * objective_index + mode_index, deltas[3 * objective_index + mode_index]
                row = record["directions"][index]
                ideal = ideal_adam(clips[objective_index], moments["first_moment"], moments["second_moment"], moments["step"], mode)
                error = displacement_error(delta, ideal)
                self.check(label + ":" + objective + ":" + mode + ":independent_ideal_adam_envelope", error["pass"], **{key: value for key, value in error.items() if key != "pass"})
                geometry = {key: direction_metrics(gradient, delta) for key, gradient in references.items()}
                self.tree(label + ":direction:" + str(index), row, {"index": index, "objective": objective,
                    "mode": mode, "saved_step": moments["step"], "next_step": ideal["next_step"],
                    "delta_norm": norm(delta), "loss_geometry": geometry})
                self.check(label + ":direction:" + str(index) + ":six_parameter_blocks", len(row["parameter_blocks"]) == 6)
                for block_index, item in enumerate(order):
                    section = slice(item["start"], item["end"])
                    self.tree(label + ":direction:" + str(index) + ":block:" + str(block_index), row["parameter_blocks"][block_index],
                        {**item, "delta_norm": norm(delta[section]),
                         "loss_geometry": {key: direction_metrics(gradient[section], delta[section]) for key, gradient in references.items()}})
                head = row["output_head_groups"]
                self.check(label + ":direction:" + str(index) + ":applicable_head_groups", head["status"] == "APPLICABLE"
                           and set(head["groups"]) == set(head_indices(order)))
                for group_name, group in head_indices(order).items():
                    selection = group["indices"]
                    self.tree(label + ":direction:" + str(index) + ":head:" + group_name, head["groups"][group_name],
                        {"output_rows": group["rows"], "parameter_count": len(selection),
                         "weight_parameter": "network.4.weight", "bias_parameter": "network.4.bias",
                         "delta_norm": norm(delta[selection]),
                         "loss_geometry": {key: direction_metrics(gradient[selection], delta[selection]) for key, gradient in references.items()}})
                certificates = {key: precision_certificate(gradient, delta, ideal["coordinate_error_bound"])
                                for key, gradient in references.items()}
                raw_base_geometry = direction_metrics(raw_base, delta)
                raw_base_certificate = precision_certificate(raw_base, delta, ideal["coordinate_error_bound"])
                agreement = geometry["base"]["relation"] == raw_base_geometry["relation"]
                self.precision.append({"snapshot_id": spec["snapshot_id"], "probe_name": name,
                    "objective": objective, "mode": mode, "adam_error": error,
                    "loss_certificates": certificates, "raw_component_base_geometry": raw_base_geometry,
                    "raw_component_base_certificate": raw_base_certificate,
                    "base_rounding_sign_agreement": bool(np.sign(geometry["base"]["dot"]) == np.sign(raw_base_geometry["dot"])),
                    "base_rounding_relation_agreement": agreement,
                    "base_conclusion_eligible": bool(agreement and geometry["base"]["relation"] in ("DESCENT", "ASCENT")
                        and certificates["base"]["rounding_stable_sign"] == geometry["base"]["relation"]
                        and raw_base_certificate["rounding_stable_sign"] == raw_base_geometry["relation"])})
                self.observed.append({"snapshot_id": spec["snapshot_id"], "probe_name": name,
                    "objective": objective, "mode": mode, "loss_geometry": geometry})
        objective_contrasts = [{"mode": mode, "first_objective": "base", "second_objective": "total",
            "first_index": index, "second_index": index + 3, **contrast_metrics(deltas[index], deltas[index + 3])}
            for index, mode in enumerate(MODES)]
        state_contrasts = [{"objective": objective, "first_mode": "KEEP", "second_mode": mode,
            "first_index": 3 * objective_index, "second_index": 3 * objective_index + mode_index,
            **contrast_metrics(deltas[3 * objective_index], deltas[3 * objective_index + mode_index])}
            for objective_index, objective in enumerate(OBJECTIVES) for mode_index, mode in enumerate(MODES[1:], 1)]
        self.tree(label + ":all_objective_contrasts", record["objective_contrasts"], objective_contrasts)
        self.tree(label + ":all_state_contrasts", record["state_contrasts"], state_contrasts)
        return {"snapshot_id": spec["snapshot_id"], "probe_name": name,
                "summary_path": (folder / "summary.json").relative_to(run).as_posix(), "summary": record}

    def run(self, run):
        import importlib
        import inspect
        import torch
        protocol, manifest, summary = (read(run / filename) for filename in
                                       ("registration.json", "manifest.json", "summary.json"))
        self.check("new_registered_diagnostic", run == ROOT / "runs/p4-adam-probe-20261015-01"
                   and protocol["run_id"] == "p4-adam-probe-20261015-01"
                   and protocol["run_directory"] == run.relative_to(ROOT).as_posix()
                   and manifest["schema"] == "p4-adam-probe-manifest-v1"
                   and summary["schema"] == "p4-adam-probe-summary-v1"
                   and summary["execution_status"] == "PASS" and summary["analysis_status"] == "DESCRIPTIVE_COMPLETE")
        self.check("registration_manifest_result_binding", sha(run / "registration.json") == manifest["protocol_sha256"]
                   == summary["protocol_sha256"] == sha(ROOT / "configs/p4_adam_probe.v1.json")
                   and sha(run / "manifest.json") == summary["manifest_sha256"])
        self.bind(ROOT / "configs/p4_adam_probe.v1.json", manifest["protocol_sha256"])
        for relative, expected in protocol["bindings_sha256"].items(): self.bind(ROOT / relative, expected)
        self.check("registered_cartesian_product", protocol["virtual_direction_count"] == 168
                   and protocol["objective_order"] == manifest["objective_order"] == list(OBJECTIVES)
                   and protocol["mode_order"] == manifest["mode_order"] == list(MODES))
        runtime = protocol["runtime"]
        adam_module = importlib.import_module("torch.optim.adam")
        self.check("registered_local_adam_runtime", str(torch.__version__) == runtime["torch_version"]
                   and sha(inspect.getsourcefile(adam_module)) == runtime["torch_adam_source_sha256"]
                   and manifest["runtime"] == runtime)
        prior = ROOT / "runs/p4-gradient-probe-20261015-01"
        previous = read(self.bind(prior / "manifest.json"))
        self.check("original_gradient_artifact_index_binding", manifest["gradient_run"] == prior.relative_to(ROOT).as_posix()
                   and manifest["gradient_run_index_sha256"] == sha(prior / "artifact_hashes.json"))
        specs = protocol["snapshots"]
        expected_ids = ["common_initial"] + [f"{branch}_u{update:04d}" for branch in ("C0", "R1") for update in (100, 500, 1000)]
        self.check("fixed_seven_snapshot_inventory", manifest["model_snapshots"] == specs == previous["model_snapshots"]
                   and [item["snapshot_id"] for item in specs] == expected_ids)
        expected_probes = ["historical_update1", "D11_zero5", "D11_transition", "D11_noncontact"]
        self.check("fixed_four_train_probe_inventory", protocol["probe_names"] == expected_probes)
        expected_files = {filename for filename in ("registration.json", "manifest.json", "history_summary.json", "aggregate.json", "summary.json")}
        expected_files |= {f"probes/{snapshot}/{probe}/{filename}" for snapshot in expected_ids
                           for probe in expected_probes for filename in ("directions.npz", "summary.json")}
        index = read(run / "artifact_hashes.json")["sha256"]
        self.check("exact_output_index_61_files", set(index) == expected_files and len(index) == 61)
        for relative, expected in index.items():
            path = (run / relative).resolve()
            self.check("output_path_scope:" + relative, path.is_relative_to(run))
            self.bind(path, expected)
        self.check("history_summary_manifest_binding", sha(run / "history_summary.json") == manifest["history_summary_sha256"])
        self.history(run)
        records, tensor_checks = [], []
        for spec in specs:
            checkpoint = ROOT / spec["checkpoint_path"]
            metadata = read(self.bind(checkpoint / "checkpoint.json", spec["checkpoint_json_sha256"]))
            self.check(spec["snapshot_id"] + ":checkpoint_metadata_identity", metadata["checkpoint_hash"] == spec["checkpoint_hash"]
                       and metadata["checkpoint_hash"] == hashlib.sha256(canonical({key: value for key, value in metadata.items()
                                                                                if key != "checkpoint_hash"})).hexdigest())
            for filename in ("weights.pt", "training_state.pt"):
                self.bind(checkpoint / filename, spec["files_sha256"][filename])
                self.check(spec["snapshot_id"] + ":checkpoint_file_binding:" + filename,
                           spec["files_sha256"][filename] == metadata["files"][filename])
            first_probe = read(prior / "probes" / spec["snapshot_id"] / expected_probes[0] / "summary.json")
            order = first_probe["parameter_order"]
            moments = read_moments(checkpoint / "training_state.pt", order, 1300 + spec["additional_updates"])
            weights = torch.load(checkpoint / "weights.pt", map_location="cpu", weights_only=True)
            self.check(spec["snapshot_id"] + ":weight_coordinate_mapping", list(weights) == [item["name"] for item in order]
                       and all(list(weights[item["name"]].shape) == item["shape"] and weights[item["name"]].dtype == torch.float32
                               and not weights[item["name"]].requires_grad for item in order))
            weight_hash = tensor_hash(weights)
            self.check(spec["snapshot_id"] + ":source_weight_tree", weight_hash == spec["model_tree_hash"])
            for name in expected_probes: records.append(self.probe(run, spec, name, moments, weight_hash))
            tensor_checks.append({"snapshot_id": spec["snapshot_id"], "weights_unchanged": True, "optimizer_unchanged": True,
                                  "weight_tree_hash": weight_hash, "optimizer_tree_hash": moments["optimizer_tree_hash"]})
        aggregate = read(run / "aggregate.json")
        self.check("all28_aggregate_exact_aliases", aggregate["schema"] == "p4-adam-probe-aggregate-v1"
                   and aggregate["row_count"] == 28 and aggregate["rows"] == records)
        self.tree("source_tensor_preservation_checks", summary["source_tensor_checks"], tensor_checks)
        self.tree("registered_execution_counts", summary["counters"], {"snapshots": 7, "probes": 28, "virtual_directions": 168})
        self.check("all168_independent_direction_appearances_retained", len(self.precision) == 168)
        forbidden_names = {"physics." + name for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse")}
        forbidden_names |= {"model_instances", "model_forwards", "Adam_instances", "Adam.step", "Optimizer.step", "autograd.grad", "Tensor.backward"}
        self.check("no_forbidden_execution", set(summary["forbidden_execution_counts"]) == forbidden_names
                   and all(type(value) is int and value == 0 for value in summary["forbidden_execution_counts"].values()))
        self.tree("readonly_result_scope", summary, {"global_rng_unchanged": True, "new_model_forwards": 0,
            "new_derivative_traversals": 0, "new_optimizer_updates": 0, "new_physics_steps": 0,
            "new_checkpoint_writes": 0, "new_data": False, "normalizer_refit": False, "final_test_used": False,
            "finite_step_loss_evaluated": False, "historical_trajectory_replayed": False, "model_selection": "NONE",
            "alignment_improvement": "NOT_EVALUATED", "personal_understanding": "NOT_ASSESSED", "P5_01": "HOLD"})
        for relative, expected in list(self.sources.items()):
            self.check("source_unchanged_after_review:" + relative, sha(ROOT / relative) == expected)


def audit(run):
    """Review saved outputs while runtime guards reject all forbidden actions."""
    import mujoco
    import torch
    run = Path(run).resolve()
    checker = Audit()
    guarded = [(mujoco, name, "physics." + name) for name in
               ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse")]
    guarded += [(torch.nn.Module, "__init__", "model_instances"), (torch.nn.Module, "_call_impl", "model_forwards"),
                (torch.optim.Adam, "__init__", "Adam_instances"), (torch.optim.Adam, "step", "Adam.step"),
                (torch.optim.Optimizer, "step", "Optimizer.step"), (torch.autograd, "grad", "autograd.grad"),
                (torch.Tensor, "backward", "Tensor.backward")]
    counts = {label: 0 for _, _, label in guarded}
    originals, rng_before = [], torch.get_rng_state().clone()
    def blocked(label):
        def call(*args, **kwargs):
            counts[label] += 1
            raise RuntimeError("Saved-only Adam audit forbids " + label)
        return call
    for owner, name, label in guarded:
        originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, blocked(label))
    result = {"schema": "p4-adam-probe-independent-review-v1", "status": "FAIL",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "run": str(run.relative_to(ROOT)),
        "review_source_sha256": sha(__file__), "checks": checker.checks,
        "numeric_policy": {"scalar_float64_rtol": 1e-10, "scalar_float64_atol": 1e-12,
            "float32_clip_norm_rtol": 64 * EPS32, "rounding_epsilon_multiplier": ROUNDING_MULTIPLIER,
            "rounding_absolute_floor": ROUNDING_ABSOLUTE_FLOOR,
            "coordinate_bound": "64*eps32*(lr/bc1)*(0.9*abs(m)+0.1*abs(clipped_gradient))/denom64+1e-12",
            "l2_bound": "L2 norm of the coordinate envelope", "norm_undefined_floor": NORM_FLOOR,
            "near_orthogonal_absolute_descent_cosine": COSINE_THRESHOLD,
            "base_primary_reference": "saved rounded float32 base promoted to float64",
            "base_sensitivity_reference": "raw64 one_step + 0.1*raw64 rollout; base conclusion requires sign/relation agreement and stable signs"}}
    try:
        checker.run(run)
        checker.check("auditor_no_forbidden_execution", not any(counts.values()))
        checker.check("auditor_rng_unchanged", torch.equal(rng_before, torch.get_rng_state()))
        histories = [item for item in checker.observed if "history_branch" in item]
        directions = [item | {key: value for key, value in precision.items() if key not in ("snapshot_id", "probe_name", "objective", "mode")}
                      for item, precision in zip([item for item in checker.observed if "objective" in item], checker.precision)]
        result.update(status="PASS", observed={"history": histories, "per_direction": directions},
            source_bindings_sha256=checker.sources, forbidden_execution_counts=counts,
            limitations=["Saved derivatives are reused; no model forward, derivative traversal, optimizer step, parameter addition or physics is performed.",
                "The ideal NumPy float64 Adam formula is checked against disposable float32 addends using a fixed rounding envelope; final addition-to-weight rounding is excluded.",
                "Directional derivatives and certified signs concern first-order local behavior only; finite loss change and full historical trajectory causes remain untested.",
                "ZERO_M isolates the old first moment with saved v and age fixed; COLD also changes second moments and bias-correction age and is not a pure second-moment intervention.",
                "The 168 directions are appearances of overlapping fixed train probes at seven snapshots, not independent experiments or generalization observations.",
                "History is an along-path sampled-batch description at changing weights; C0 relative zeros are uncomputed placeholders and no best checkpoint is selected."])
    except Exception as error:
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        for owner, name, original in reversed(originals): setattr(owner, name, original)
    result["check_count"] = len(checker.checks)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    require(not arguments.output.exists(), "Never overwrite an audit or failed prefix")
    result = audit(arguments.run)
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "checks": result["check_count"], "failure": result.get("failure")}))
    if result["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__": main()

