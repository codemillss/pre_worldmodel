"""Lossless recording boundary around the unchanged final episode executor.

This module does not select/freeze final cases, load models, or authorize a
batch. The caller supplies the declared cell list and a remaining-work budget.
It runs one actual controller episode at a time. Fresh reproduction always
calls the controller/executor again; recorded commands are never replay inputs.
Completed prefixes are retained on failures. Persistence is at cell completion,
not a claim of per-action crash durability.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import traceback

import numpy as np

from .environment import Observation, SnapshotBundle, StepResult
from .final_execution import EpisodeExecution, array_hash, execute_episode
from . import final_recording
from .final_recording import (file_sha256, read_json_gzip, read_plan_index,
                              write_json_gzip, write_plan_index, write_witness_gzip)
from .geometry_control import GeometryContext
from .planning import PlannerContext, TaskSpec
from .reproduction import _validate, build_witness, compare_witness, json_safe, raw_reference
from .resource_budget import stage_guard
from .task import TaskStatus

SCHEMA = "p5-recorded-episode-v1"
FULL_TRACE_TICKS = (0, 29, 299)
_CAPTURE_TYPES = {cls.__name__: cls for cls in
                  (SnapshotBundle, Observation, TaskStatus, TaskSpec, PlannerContext, GeometryContext)}


class DriverStop(RuntimeError):
    """Resource/source stop; an already returned episode stays on the exception."""

    def __init__(self, decision, *, execution=None):
        super().__init__("; ".join(decision.get("reasons", [decision.get("status", "driver stop")])))
        self.decision = decision
        self.execution = execution


def _nonnegative(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} requires a nonnegative integer")
    return value


def bindings_match(bindings):
    """Read source/original bindings at the cell boundary, not on every action."""
    if not isinstance(bindings, dict) or not bindings:
        return False
    return all(isinstance(path, str) and isinstance(sha, str) and len(sha) == 64
               and Path(path).is_file() and file_sha256(path) == sha
               for path, sha in bindings.items())


class LiveResourceGuard:
    """Live stage/cell/action/write checks with an explicit empirical allowance.

    ``remaining_required_bytes`` includes the 2GiB reserve and active cell until
    the caller commits its ledger after return. During a cell, bytes already
    saved are subtracted exactly once. The caller must update remaining counts
    between cells; stale higher counts stop conservatively. Observed growth
    beyond a cell allowance stops and requires re-estimation, never omission.
    All byte allowances are planning bounds, not a proof of worst-case size.
    """

    def __init__(self, root, *, remaining_required_bytes, cell_write_allowance_bytes,
                 hard_reserve_bytes=2 * 1024**3, mismatch_limit_bytes=256 * 1024**2,
                 mismatch_used_bytes=0, free_bytes=None):
        self.root = Path(root)
        if not callable(remaining_required_bytes):
            raise TypeError("remaining_required_bytes must reprice remaining work")
        self.remaining_required_bytes = remaining_required_bytes
        self.cell_write_allowance_bytes = _nonnegative(cell_write_allowance_bytes, "cell allowance")
        self.hard_reserve_bytes = _nonnegative(hard_reserve_bytes, "reserve")
        self.mismatch_limit_bytes = _nonnegative(mismatch_limit_bytes, "mismatch limit")
        self.mismatch_used_bytes = _nonnegative(mismatch_used_bytes, "mismatch used")
        self.free_bytes = free_bytes or (lambda: shutil.disk_usage(self.root).free)
        if not callable(self.free_bytes):
            raise TypeError("free_bytes must be callable")
        self.active = False
        self.written_bytes = 0
        self.verified_bindings = {}
        self.last_decision = None
        self.history = []
        self._expected_mismatch_bytes = 0
        self.stopped = False

    def _check(self, phase, *, next_write_bytes=0, mismatch_bytes=0, source_ok=True,
               preserve_already_authorized=False):
        free = _nonnegative(self.free_bytes(), "live free bytes")
        remaining = _nonnegative(self.remaining_required_bytes(), "remaining required bytes")
        remaining = max(self.hard_reserve_bytes, remaining - self.written_bytes)
        # Once actual control has run, its prefix already owns the reservation.
        # Capacity loss forbids further control, but cannot make us discard it.
        # Emergency preservation still must fit both reserve and preauthorized
        # cell/mismatch allowances; it never starts another action.
        if preserve_already_authorized:
            remaining = min(remaining, self.hard_reserve_bytes + next_write_bytes)
        decision = stage_guard(free_bytes=free, remaining_required_bytes=remaining,
            next_write_upper_bytes=next_write_bytes, hard_reserve_bytes=self.hard_reserve_bytes,
            observed_unit_bytes=self.written_bytes,
            allowed_unit_bytes=self.cell_write_allowance_bytes,
            mismatch_used_bytes=self.mismatch_used_bytes,
            next_mismatch_write_upper_bytes=mismatch_bytes,
            mismatch_limit_bytes=self.mismatch_limit_bytes, bindings_match=source_ok)
        self.last_decision = {**decision, "phase": phase, "free_bytes": free,
            "remaining_required_bytes": remaining, "written_bytes": self.written_bytes,
            "next_write_allowance_bytes": next_write_bytes,
            "mismatch_used_bytes": self.mismatch_used_bytes}
        self.history.append(deepcopy(self.last_decision))
        if decision["status"] != "CONTINUE_RESOURCE_CHECK_ONLY":
            raise DriverStop(self.last_decision)
        return self.last_decision

    def before_stage(self):
        """Read live whole-work capacity without starting a cell."""
        return self._check("before_stage")

    def begin_cell(self, bindings, *, fresh=False):
        if self.stopped:
            raise DriverStop({"status": "STOP_PRESERVE_PREFIX",
                              "reasons": ["prior_cell_requires_review_and_reestimate"]})
        if self.active:
            raise RuntimeError("A previous cell has not been closed")
        self.written_bytes = 0
        self.history = []  # Each receipt covers this cell, never the whole batch.
        self.verified_bindings = deepcopy(bindings)
        self._expected_mismatch_bytes = self.cell_write_allowance_bytes if fresh else 0
        self._check("before_cell", next_write_bytes=self.cell_write_allowance_bytes,
                    mismatch_bytes=self._expected_mismatch_bytes,
                    source_ok=bindings_match(bindings))
        self.active = True

    def before_action(self):
        if not self.active:
            raise RuntimeError("No active cell")
        return self._check("before_action", next_write_bytes=max(
            0, self.cell_write_allowance_bytes-self.written_bytes),
            mismatch_bytes=self._expected_mismatch_bytes)

    def before_write(self, *, preserving=False):
        if not self.active:
            raise RuntimeError("No active cell")
        return self._check("before_preserve" if preserving else "before_write",
            next_write_bytes=max(0, self.cell_write_allowance_bytes-self.written_bytes),
            mismatch_bytes=self._expected_mismatch_bytes,
            preserve_already_authorized=preserving)

    def observe(self, folder, *, mismatch=False):
        self.written_bytes = sum(p.stat().st_size for p in Path(folder).rglob("*") if p.is_file())
        # Charge an entire failed fresh cell, including witnesses and metadata,
        # once, at finish. Prewrite checks reserve its full configured allowance.
        if self.written_bytes > self.cell_write_allowance_bytes:
            self._check("observed_output_growth", mismatch_bytes=(
                self.written_bytes if mismatch else self._expected_mismatch_bytes))
        return self.written_bytes

    def finish_cell(self, *, mismatch=False, stop=False):
        if mismatch:
            self.mismatch_used_bytes += self.written_bytes
        self.active = False
        self._expected_mismatch_bytes = 0
        self.stopped = self.stopped or bool(stop)


class _GuardedEnvironment:
    def __init__(self, env, guard):
        self._env, self._guard = env, guard

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        self._guard.before_action()
        return self._env.step(action)


def _bytes_view(array):
    array = np.ascontiguousarray(array)
    return memoryview(array).cast("B") if array.size else memoryview(b"")


def _frame(hasher, marker, payload=b""):
    hasher.update(marker)
    hasher.update(struct.pack("<Q", len(payload)))
    hasher.update(payload)


def _diagnostic_scalar(value):
    # The original diagnostics writer maps NumPy scalars to JSON scalars.
    # Conversion is local to one leaf; never copy the entire diagnostics tree.
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _hash_typed(value, hasher, *, diagnostic=False):
    """Stream one typed tree with bounded auxiliary memory.

    Diagnostic lists/tuples/NumPy arrays follow the original JSON schema;
    numeric state/action arrays outside diagnostics retain exact type/bytes.
    No mirrored Python tree, giant JSON string or whole-array byte copy is made.
    """
    if diagnostic:
        value = _diagnostic_scalar(value)
    if is_dataclass(value):
        _frame(hasher, b"D", type(value).__name__.encode())
        for field in fields(value):
            _frame(hasher, b"K", field.name.encode())
            _hash_typed(getattr(value, field.name), hasher,
                diagnostic=diagnostic or isinstance(value, StepResult) and field.name == "diagnostics")
        _frame(hasher, b"E")
    elif isinstance(value, np.ndarray) and not diagnostic:
        if value.dtype.hasobject:
            raise TypeError("Object arrays are not lossless numeric evidence")
        _frame(hasher, b"A", value.dtype.str.encode())
        _frame(hasher, b"S", str(value.shape).encode())
        view = _bytes_view(value)
        _frame(hasher, b"L", str(len(view)).encode())
        for offset in range(0, len(view), 1024**2):
            hasher.update(view[offset:offset+1024**2])
    elif isinstance(value, np.generic):
        _frame(hasher, b"N")
        _hash_typed(np.asarray(value), hasher)
    elif isinstance(value, dict):
        if any(type(key) not in (str, int) for key in value):
            raise TypeError("Only string/integer dictionary keys supported")
        _frame(hasher, b"M", str(len(value)).encode())
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            _hash_typed(str(key) if diagnostic else key, hasher, diagnostic=diagnostic)
            _hash_typed(item, hasher, diagnostic=diagnostic)
        _frame(hasher, b"E")
    elif isinstance(value, (tuple, list)) or diagnostic and isinstance(value, np.ndarray):
        marker = b"L" if diagnostic or isinstance(value, list) else b"T"
        _frame(hasher, marker, str(len(value)).encode())
        for item in value:
            _hash_typed(item, hasher, diagnostic=diagnostic)
        _frame(hasher, b"E")
    elif value is None:
        _frame(hasher, b"0")
    elif type(value) is bool:
        _frame(hasher, b"B", b"1" if value else b"0")
    elif type(value) is int:
        _frame(hasher, b"I", str(value).encode())
    elif type(value) is float:
        _frame(hasher, b"F", struct.pack("=d", value))
    elif type(value) is str:
        _frame(hasher, b"U", value.encode())
    else:
        raise TypeError(f"Unsupported exact evidence type: {type(value).__name__}")


def typed_digest(value):
    hasher = hashlib.sha256()
    _hash_typed(value, hasher)
    return hasher.hexdigest()


def _typed_equal(left, right, *, diagnostic=False):
    if diagnostic:
        left, right = _diagnostic_scalar(left), _diagnostic_scalar(right)
    if is_dataclass(left) or is_dataclass(right):
        return (type(left) is type(right) and all(_typed_equal(
            getattr(left, field.name), getattr(right, field.name),
            diagnostic=diagnostic or isinstance(left, StepResult) and field.name == "diagnostics")
            for field in fields(left)))
    if isinstance(left, np.ndarray) and not diagnostic or isinstance(right, np.ndarray) and not diagnostic:
        if not (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                and left.dtype.str == right.dtype.str and left.shape == right.shape
                and not left.dtype.hasobject and not right.dtype.hasobject):
            return False
        a, b = _bytes_view(left), _bytes_view(right)
        return all(a[offset:offset+1024**2] == b[offset:offset+1024**2]
                   for offset in range(0, len(a), 1024**2))
    if isinstance(left, np.generic) or isinstance(right, np.generic):
        return (isinstance(left, np.generic) and isinstance(right, np.generic)
                and _typed_equal(np.asarray(left), np.asarray(right)))
    if isinstance(left, dict) or isinstance(right, dict):
        if not (isinstance(left, dict) and isinstance(right, dict) and len(left) == len(right)):
            return False
        a = sorted(left.items(), key=lambda pair: str(pair[0]))
        b = sorted(right.items(), key=lambda pair: str(pair[0]))
        return all(_typed_equal(str(ka) if diagnostic else ka, str(kb) if diagnostic else kb,
                                diagnostic=diagnostic)
                   and _typed_equal(va, vb, diagnostic=diagnostic)
                   for (ka, va), (kb, vb) in zip(a, b))
    sequence = (list, tuple, np.ndarray) if diagnostic else (list, tuple)
    if isinstance(left, sequence) or isinstance(right, sequence):
        return (isinstance(left, sequence) and isinstance(right, sequence)
                and (diagnostic or type(left) is type(right)) and len(left) == len(right)
                and all(_typed_equal(a, b, diagnostic=diagnostic) for a, b in zip(left, right)))
    if type(left) is not type(right):
        return False
    if type(left) is float:
        return struct.pack("=d", left) == struct.pack("=d", right)
    return left == right


def exact_comparison(left, right):
    a, b = typed_digest(left), typed_digest(right)
    equal = a == b and _typed_equal(left, right)
    return {"status": "PASS_EXACT" if equal else "FAIL_EXACT",
            "comparison": "streamed typed hash plus direct recursive equality; dtype/shape/bytes, scalar bits, flags, schema and complete JSON diagnostics",
            "left_sha256": a, "right_sha256": b}


def plan_index_records(execution):
    """Retain each original candidate row, including terminal and invalid rows."""
    result = []
    decision_ticks = [row["index"] for row in execution.decisions if row.get("plan_identity") is not None]
    if sorted(execution.plan_results) != sorted(decision_ticks) or len(set(decision_ticks)) != len(decision_ticks):
        raise ValueError("Every learned decision requires one complete plan")
    for tick, planned in sorted(execution.plan_results.items()):
        result.append({"tick": tick, "arrays": {
            "valid": planned.predictions.valid.copy(),
            "reasons": np.asarray(planned.predictions.reasons),
            "failure_steps": planned.predictions.failure_steps.copy(),
            "total_cost": planned.costs.total.copy(),
            **{"cost_" + key: value.copy() for key, value in planned.costs.components.items()}}})
    return result


def _encode_input(value, arrays):
    if is_dataclass(value):
        if type(value).__name__ not in _CAPTURE_TYPES:
            raise TypeError("Unsupported full-input dataclass")
        return {"kind": "dataclass", "class": type(value).__name__, "fields": {
            field.name: _encode_input(getattr(value, field.name), arrays) for field in fields(value)}}
    if isinstance(value, (np.ndarray, np.generic)):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError("Object-array full input")
        key = f"input_{len(arrays):04d}"
        arrays[key] = array.copy()
        return {"kind": "numpy_scalar" if isinstance(value, np.generic) else "array", "key": key,
                "dtype": array.dtype.str, "shape": list(array.shape)}
    if type(value) is float:
        return {"kind": "float64", "bits": struct.pack("=d", value).hex()}
    if isinstance(value, dict):
        return {"kind": "dict", "items": [[_encode_input(key, arrays), _encode_input(item, arrays)]
                                          for key, item in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"kind": "tuple" if isinstance(value, tuple) else "list",
                "items": [_encode_input(item, arrays) for item in value]}
    if value is None or type(value) in (str, int, bool):
        return {"kind": "scalar", "value": value}
    raise TypeError(f"Unsupported full input type: {type(value).__name__}")


def _decode_input(value, arrays):
    kind = value["kind"]
    if kind in {"array", "numpy_scalar"}:
        array = arrays[value["key"]]
        if array.dtype.str != value["dtype"] or list(array.shape) != value["shape"]:
            raise ValueError("Full-input array type mismatch")
        return array.copy() if kind == "array" else array[()]
    if kind == "float64":
        return struct.unpack("=d", bytes.fromhex(value["bits"]))[0]
    if kind == "scalar":
        return value["value"]
    if kind == "dict":
        return {_decode_input(key, arrays): _decode_input(item, arrays) for key, item in value["items"]}
    if kind in {"tuple", "list"}:
        items = [_decode_input(item, arrays) for item in value["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "dataclass":
        return _CAPTURE_TYPES[value["class"]](**{
            key: _decode_input(item, arrays) for key, item in value["fields"].items()})
    raise ValueError("Unknown full-input type")


def write_full_trace(folder, *, tick, captured_input, planned=None):
    """Save full typed regeneration input and, when present, all model arrays."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    arrays = {}
    encoded = _encode_input(captured_input, arrays)
    info = {"schema": "p5-full-plan-input-v1", "tick": tick, "input": encoded,
            "input_exact_sha256": exact_comparison(captured_input, captured_input)["left_sha256"],
            "plan": None}
    if planned is not None:
        arrays.update({"candidates": planned.candidates.actions, "predicted_states": planned.predictions.states,
                       "raw_deltas": planned.predictions.raw_deltas,
                       **plan_index_records_from_plan(planned)})
        info["plan"] = json_safe({"selected_index": planned.selected_index, "reason": planned.reason,
            "action": planned.action, "context": planned.context,
            "candidate_sources": planned.candidates.sources,
            "candidate_generation": planned.candidates.diagnostics,
            "planner_diagnostics": planned.diagnostics})
    path = folder / "trace.npz"
    with path.open("xb") as target:
        np.savez_compressed(target, **arrays)
        target.flush()
        os.fsync(target.fileno())
    info["npz_sha256"] = file_sha256(path)
    meta = folder / "input.json.gz"
    write_json_gzip(meta, info)
    return {"tick": tick, "files_sha256": {str(p): file_sha256(p) for p in (path, meta)},
            "bytes": {str(p): p.stat().st_size for p in (path, meta)},
            "input_exact_sha256": info["input_exact_sha256"], "full_model_arrays": planned is not None}


def read_full_trace(receipt):
    for path, expected in receipt["files_sha256"].items():
        if not Path(path).is_file() or file_sha256(path) != expected:
            raise ValueError("Missing or changed full trace")
    metadata_path = next(path for path in receipt["files_sha256"] if path.endswith("input.json.gz"))
    metadata = read_json_gzip(metadata_path,
        expected_sha256=receipt["files_sha256"][metadata_path])
    npz_path = Path(metadata_path).parent / "trace.npz"
    if file_sha256(npz_path) != metadata["npz_sha256"]:
        raise ValueError("Full trace internal binding mismatch")
    with np.load(npz_path, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    value = _decode_input(metadata["input"], arrays)
    if exact_comparison(value, value)["left_sha256"] != metadata["input_exact_sha256"]:
        raise ValueError("Full-input exact-tree mismatch")
    return value, {key: value for key, value in arrays.items() if not key.startswith("input_")}, metadata


def plan_index_records_from_plan(planned):
    return {"valid": planned.predictions.valid, "reasons": np.asarray(planned.predictions.reasons),
            "failure_steps": planned.predictions.failure_steps, "total_cost": planned.costs.total,
            **{"cost_" + key: value for key, value in planned.costs.components.items()}}


def _capture_terminal_input(env, execution):
    decision = execution.unexecuted_plan
    if decision is None:
        return
    tick = decision["index"]
    if tick in execution.captured_inputs:
        return
    observed = execution.results[-1].observation if execution.results else execution.initial_observation
    if observed is None or not np.array_equal(observed.state, np.asarray(decision["state"]), equal_nan=True) \
            or observed.sim_time != decision["sim_time"]:
        raise ValueError("Terminal input no longer at the unexecuted decision boundary")
    execution.captured_inputs[tick] = {"tick": tick, "snapshot": deepcopy(env.snapshot()),
        "observation": deepcopy(observed), "task_status": TaskStatus(**decision["task_status"]),
        "planner_context": deepcopy(execution.final_context),
        "geometry_context": deepcopy(execution.final_gctx)}


def _full_input(capture, execution, kwargs, identity, source_bindings):
    decision = next((row for row in execution.decisions if row["index"] == capture["tick"]), None)
    return {**deepcopy(capture), "initial_observation": deepcopy(execution.initial_observation),
        "task_spec": deepcopy(execution.task_spec), "identity": deepcopy(identity),
        "source_bindings": deepcopy(source_bindings), "origin": deepcopy(execution.origin),
        "episode_index": kwargs["episode_index"], "case_id": execution.case_id,
        "controller_id": execution.controller_id, "candidate_seed_base": kwargs["candidate_seed_base"],
        "candidate_stream_id": kwargs.get("candidate_stream_id", 0),
        "rng_keys": None if decision is None else deepcopy(decision["rng_keys"]),
        "physical_config": deepcopy(kwargs["physical_config"]),
        "planner_config": deepcopy(kwargs["planner_config"]),
        "geometry_parameters": deepcopy(kwargs["geometry_parameters"]),
        "transition_parameters": deepcopy(kwargs["transition_parameters"]),
        "penetration_budget_m": kwargs["penetration_budget_m"],
        "run_scope": kwargs["run_scope"], "max_intervals": kwargs["max_intervals"]}


def _read_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported episode manifest")
    for saved, expected in manifest["files_sha256"].items():
        if not Path(saved).is_file() or file_sha256(saved) != expected:
            raise ValueError(f"Missing or changed original evidence: {saved}")
    return manifest


@dataclass
class RecordedEpisode:
    execution: EpisodeExecution
    manifest_path: Path | None
    manifest: dict
    witness: dict | None
    comparison: dict | None
    status: str


def run_recorded_episode(env, initial_snapshot, *, folder, identity, source_bindings,
                         profile, guard, reference_manifest=None, raw_codec=None, **kwargs):
    """Run one cell with frozen semantics, lossless saving and fresh exact audit.

    The root batch driver owns G3 final-manifest approval and cell denominator.
    Source files are verified once immediately before actual execution. Identity
    must include model and normalizer bindings for learned controllers (G uses
    explicit None). Their files must occur in ``source_bindings``. No retry or
    completed-case selection is performed here. Exceptions during serialization
    produce a STOP result with any existing partial files retained.
    """
    raw_codec = raw_codec or final_recording
    if not all(callable(getattr(raw_codec, name, None)) for name in ("write_raw", "read_raw")):
        raise TypeError("Raw codec must expose write_raw/read_raw with SHA receipts")
    if not isinstance(guard, LiveResourceGuard):
        raise TypeError("LiveResourceGuard required")
    if not isinstance(identity, dict) or not {"model", "normalizer"} <= set(identity):
        raise ValueError("Explicit model/normalizer identity required, including None for G")
    if kwargs.get("predictor") is not None and (identity["model"] is None or identity["normalizer"] is None):
        raise ValueError("Learned controller needs model and normalizer identities")
    if "capture_ticks" in kwargs and tuple(kwargs["capture_ticks"]) != FULL_TRACE_TICKS:
        raise ValueError("Final recording uses predeclared capture ticks 0/29/299")
    folder = Path(folder).resolve()
    if folder.exists():
        raise FileExistsError(folder)
    # Missing original evidence is rejected before new physical work starts.
    original = None if reference_manifest is None else _read_manifest(reference_manifest)
    if original is not None and original.get("record_kind") != "reference_controller_execution":
        raise ValueError("Fresh replay requires a retained original controller execution")
    bound = deepcopy(source_bindings)
    if original is not None:
        if original["source_bindings"] != bound or original["identity"] != json_safe(identity):
            raise ValueError("Fresh source/model identity differs from original")
        bound.update(original["files_sha256"])
        bound[str(Path(reference_manifest).resolve())] = file_sha256(reference_manifest)
    guard.begin_cell(bound, fresh=original is not None)
    folder.mkdir(parents=True, exist_ok=False)
    kwargs = deepcopy({key: value for key, value in kwargs.items() if key != "predictor"}) | {
        "predictor": kwargs.get("predictor"), "capture_ticks": FULL_TRACE_TICKS}
    # Capture current, never future, inputs transiently. Normal disk retention
    # remains exactly the selected slots. A mismatch at any other tick needs its
    # actual full input, not a fabricated state-only replacement.
    core_kwargs = {**kwargs, "capture_ticks": tuple(range(
        round(TaskSpec.from_config(kwargs["physical_config"]).max_episode_time_s / env.action_dt) + 1))}
    try:
        execution = execute_episode(_GuardedEnvironment(env, guard), initial_snapshot, **core_kwargs)
    except Exception:
        guard.finish_cell(stop=True)
        raise
    mismatch, comparison, witness = False, None, None
    saved = {}
    manifest = {"schema": SCHEMA, "record_kind": "reference_controller_execution" if original is None
                else "fresh_controller_replay", "identity": json_safe(identity),
                "source_bindings": deepcopy(source_bindings), "execution": execution.summary(),
                "raw": None, "plan_index": None, "full_traces": [], "files_sha256": saved,
                "reference_manifest": None if original is None else str(Path(reference_manifest).resolve()),
                "reference_manifest_sha256": None if original is None else file_sha256(reference_manifest),
                "recording_status": "INCOMPLETE", "comparison": None,
                "missing_predeclared_ticks": [tick for tick in FULL_TRACE_TICKS if tick not in execution.captured_inputs]}
    raw_ref, original_ref, ref_witness = None, None, None
    differing_ticks = set()
    try:
        _capture_terminal_input(env, execution)
        indexes = plan_index_records(execution)
        if original is not None:
            try:
                ref_witness_path = original["witness_path"]
                ref_witness = read_json_gzip(ref_witness_path,
                    expected_sha256=original["files_sha256"][ref_witness_path])
                original_ref = ref_witness["raw_reference"]
                compact = build_witness(execution, identity=identity, source_bindings=source_bindings,
                    raw_reference=None, original_raw_reference=original_ref, attempt_id=kwargs["execution_id"])
                comparison = {"boundary": compare_witness(ref_witness, compact, execution.task_spec, profile)}
                # The saved original chooses its codec. A cross-format comparison
                # must pass the corresponding module explicitly to the caller.
                if original.get("raw_codec_module") != raw_codec.__name__:
                    raise ValueError("Original raw codec differs from requested reader")
                ref_snapshot, ref_results = raw_codec.read_raw(original["raw"]["prefix"],
                    expected_sha256=original["raw"]["files_sha256"])
                comparison["raw_exact"] = exact_comparison((ref_snapshot, ref_results),
                                                            (execution.initial_snapshot, execution.results))
                for index, (left, right) in enumerate(zip(ref_results, execution.results)):
                    if exact_comparison(left, right)["status"] != "PASS_EXACT" and index < len(execution.records):
                        differing_ticks.add(execution.records[index]["index"])
                del ref_snapshot, ref_results
                ref_index = read_plan_index(original["plan_index"]["path"],
                    expected_sha256=original["plan_index"]["files_sha256"][original["plan_index"]["path"]])
                comparison["plan_index_exact"] = exact_comparison(ref_index, indexes)
                originals_by_tick = {row["tick"]: row for row in ref_index}
                for row in indexes:
                    if row["tick"] not in originals_by_tick or exact_comparison(
                            originals_by_tick[row["tick"]], row)["status"] != "PASS_EXACT":
                        differing_ticks.add(row["tick"])
                del ref_index
                original_decisions = {row["index"]: row for row in ref_witness["decisions"]}
                for row in execution.decisions:
                    if row["index"] not in original_decisions or exact_comparison(
                            original_decisions[row["index"]], row)["status"] != "PASS_EXACT":
                        differing_ticks.add(row["index"])
                mismatch = (comparison["boundary"]["status"] != "PASS_REPRODUCTION"
                            or comparison["raw_exact"]["status"] != "PASS_EXACT"
                            or comparison["plan_index_exact"]["status"] != "PASS_EXACT"
                            or comparison["boundary"]["fresh_full_raw_prefix_required"])
                witness = compact
            except Exception as error:
                # An original disappearing after execution began cannot make
                # the fresh physical prefix vanish. Save it and stop review.
                comparison = {**(comparison or {}), "evidence_error": {
                    "type": type(error).__name__, "message": str(error)}, "status": "INCOMPLETE"}
                mismatch = True
                differing_ticks.update(execution.plan_results)
            comparison["differing_ticks"] = sorted(differing_ticks)
            manifest["comparison"] = comparison
        required_input_ticks = set(FULL_TRACE_TICKS) & {row["index"] for row in execution.decisions}
        if execution.unexecuted_plan is not None:
            required_input_ticks.add(execution.unexecuted_plan["index"])
        missing_input_ticks = sorted(required_input_ticks - set(execution.captured_inputs))
        manifest["missing_required_input_ticks"] = missing_input_ticks
        sources_unchanged = bindings_match(bound)
        manifest["source_binding_check_after_execution"] = sources_unchanged
        # Errors/physical failures are never represented by a successful compact
        # reference. A normal task timeout/planner-invalid is still an outcome.
        physical_failure = execution.termination in {"physics_error", "geometry_invalid"} or any(
            not step.transition_valid or not step.observation.observation_valid for step in execution.results)
        must_write_full = original is None or mismatch or physical_failure \
            or execution.execution_status != "completed" or not sources_unchanged or bool(missing_input_ticks)
        preserving = mismatch or physical_failure or execution.execution_status != "completed" \
            or not sources_unchanged or bool(missing_input_ticks)
        mismatch = bool(original is not None and preserving)
        guard.before_write(preserving=preserving)
        if must_write_full:
            prefix = folder / "raw_execution"
            raw_receipt = raw_codec.write_raw(prefix, execution.initial_snapshot, execution.results, xz_preset=6)
            manifest["raw"] = {"prefix": str(prefix), **raw_receipt}
            saved.update(raw_receipt["files_sha256"])
            raw_ref = raw_reference(raw_receipt["files_sha256"], len(execution.results))
            manifest["raw_codec_module"] = raw_codec.__name__
            guard.observe(folder, mismatch=mismatch)
            guard.before_write(preserving=preserving)
            index_path = folder / "plan_index.zip"
            index_receipt = write_plan_index(index_path, indexes)
            manifest["plan_index"] = {"path": str(index_path), **index_receipt}
            saved.update(index_receipt["files_sha256"])
            guard.observe(folder, mismatch=mismatch)
        else:
            manifest["raw_reference_reuse"] = {"files_sha256": original["raw"]["files_sha256"],
                "prefix": original["raw"]["prefix"], "validated_by": comparison["raw_exact"]}
            manifest["plan_index_reference_reuse"] = {"files_sha256": original["plan_index"]["files_sha256"],
                "path": original["plan_index"]["path"], "validated_by": comparison["plan_index_exact"]}
        # Store G fresh input captures too, as priced in the 25 G episodes.
        # Normal learned fresh cells reuse the source-bound original traces;
        # mismatches retain any differing tick in addition to normal slots.
        if must_write_full or kwargs["predictor"] is None:
            trace_ticks = set(FULL_TRACE_TICKS) & set(execution.captured_inputs)
            if execution.unexecuted_plan is not None:
                trace_ticks.add(execution.unexecuted_plan["index"])
            if mismatch:
                trace_ticks.update(differing_ticks)
            for tick in sorted(trace_ticks):
                if tick not in execution.captured_inputs:
                    manifest.setdefault("missing_diagnostic_input_ticks", []).append(tick)
                    continue
                guard.before_write(preserving=preserving)
                receipt = write_full_trace(folder / "plans" / f"step-{tick:03d}", tick=tick,
                    captured_input=_full_input(execution.captured_inputs[tick], execution,
                        kwargs, identity, source_bindings), planned=execution.plan_results.get(tick))
                saved.update(receipt["files_sha256"])
                manifest["full_traces"].append(receipt)
                guard.observe(folder, mismatch=mismatch)
        if witness is None or must_write_full:
            witness = build_witness(execution, identity=identity, source_bindings=source_bindings,
                raw_reference=raw_ref, attempt_id=kwargs["execution_id"],
                original_raw_reference=None if original is None else original_ref)
        if original is not None and must_write_full and ref_witness is not None:
            comparison["boundary_with_preserved_prefix"] = compare_witness(
                ref_witness, witness, execution.task_spec, profile)
        witness_issues = _validate(witness, "recorded", 1e-9)
        manifest["witness_validation_issues"] = witness_issues
        if witness_issues:
            preserving = True
        guard.before_write(preserving=preserving)
        witness_path = folder / "witness.json.gz"
        receipt = write_witness_gzip(witness_path, witness)
        saved.update(receipt["files_sha256"])
        manifest["witness_path"] = str(witness_path)
        manifest["recording_status"] = "STOP_REVIEW" if preserving else "COMPLETE_RECORDED"
        manifest["comparison"] = comparison
        guard.observe(folder, mismatch=mismatch)
        guard.before_write(preserving=preserving)
        manifest["resources"] = {"bytes_before_manifest": guard.written_bytes,
            "guard_checks": deepcopy(guard.history), "mismatch_charged": mismatch,
            "durability": "completed prefix at cell end; no per-interval power-loss recovery"}
        target = folder / "manifest.json"
        with target.open("x") as stream:
            json.dump(json_safe(manifest), stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        guard.observe(folder, mismatch=mismatch)
        guard.finish_cell(mismatch=mismatch, stop=preserving)
        return RecordedEpisode(execution, target, manifest, witness, comparison, manifest["recording_status"])
    except Exception as error:
        # Keep partial raw/index/trace files. Never overwrite or delete them and
        # never pretend an in-memory prefix is durable when writing failed.
        manifest["recording_status"] = "STOP_INCOMPLETE_RECORDING"
        manifest["recording_error"] = {"type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc()}
        manifest["partial_files_sha256"] = {str(path): file_sha256(path)
            for path in folder.rglob("*") if path.is_file()}
        failure_path = folder / "recording_failure.json"
        try:
            guard.before_write(preserving=True)
            with failure_path.open("x") as stream:
                json.dump(json_safe(manifest), stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            failure_path = None
        try:
            guard.observe(folder, mismatch=mismatch)
        except DriverStop:
            pass  # Existing oversize partial files remain; next cell is forbidden.
        guard.finish_cell(mismatch=mismatch, stop=True)
        return RecordedEpisode(execution, failure_path, manifest, witness, comparison,
                               "STOP_INCOMPLETE_RECORDING")
