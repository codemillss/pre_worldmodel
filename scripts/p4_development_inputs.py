"""Restore predeclared P4 development inputs from preserved P3 executions.

Source regimes describe the original recorded next interval, not a state-only
property or the regime of a new candidate. A derived transition snapshot has
no original mid-interval engine payload to compare: its evidence is a checked
recorded-command prefix and exact persistence of the resulting complete state.
No model, candidate future, data fit, or final test is used here.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import tomllib

import numpy as np

from push_align.candidate_replay import snapshots_exact
from push_align.contact_geometry import check_contact_geometry
from push_align.data import _encode_nonfinite, make_contract
from push_align.environment import Observation, PlanarPushEnv, SnapshotBundle
from push_align.geometry_control import GeometryContext, MODES
from push_align.planning import PlannerContext, TaskSpec
from push_align.task import TaskStatus, _status, start_task


class DevelopmentInputError(ValueError):
    """An archived source, complete snapshot, or recorded context disagrees."""


@dataclass(frozen=True)
class DevelopmentSnapshot:
    snapshot: SnapshotBundle
    state: np.ndarray
    task: TaskSpec
    status: TaskStatus
    context: PlannerContext
    gctx: GeometryContext
    case: dict
    original_plan: dict
    proof: dict
    source_files_sha256: dict


def _require(condition, message):
    if not bool(condition):
        raise DevelopmentInputError(message)


def _canonical(value):
    return json.dumps(_encode_nonfinite(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def _nonfinite(value):
    raise DevelopmentInputError(f"Nonstandard nonfinite JSON constant: {value}")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_nonfinite)


def _read_steps(path):
    return [json.loads(line, parse_constant=_nonfinite)
            for line in Path(path).read_text(encoding="utf-8").splitlines()]


def _snapshot(value, env):
    _require(isinstance(value, dict) and set(value) == set(SnapshotBundle.__dataclass_fields__),
             "Complete archived snapshot fields are required")
    current = env.snapshot()
    integration = np.asarray(value["integration_state"], dtype=np.float64)
    _require(integration.shape == current.integration_state.shape and np.isfinite(integration).all(),
             "Snapshot integration vector is incompatible or nonfinite")
    _require(value["model_hash"] == env.model_hash and value["environment_hash"] == env.environment_hash,
             "Snapshot model/environment differs from the physical contract")
    warnings = []
    for name in ("warning_counts", "warning_lastinfo"):
        raw = np.asarray(value[name])
        _require(raw.shape == getattr(current, name).shape and raw.dtype.kind in "iu"
                 and (raw >= 0).all() and (raw <= np.iinfo(getattr(current, name).dtype).max).all(),
                 f"Invalid snapshot {name}")
        warnings.append(raw.astype(getattr(current, name).dtype, copy=True))
    height = value["reference_height"]
    _require(type(height) in (int, float) and math.isfinite(height) and height > 0
             and value["needs_reset"] is False and isinstance(value["rng_state"], dict),
             "Snapshot must be finite, ready, and contain local RNG state")
    return SnapshotBundle(integration.copy(), value["model_hash"], value["environment_hash"],
        height, deepcopy(value["rng_state"]), *warnings, False)


def _physical_array(value, shape, name):
    a = np.asarray(value, dtype=np.float64)
    _require(a.shape == shape and np.isfinite(a).all(), f"Invalid {name} physical array")
    return a.copy()


def _task_status(value):
    _require(isinstance(value, dict) and set(value) == set(TaskStatus.__dataclass_fields__),
             "Original task status fields are incomplete")
    fields = deepcopy(value); fields["violations"] = tuple(fields["violations"])
    status = TaskStatus(**fields); _status(status)
    _require(status.phase in ("running", "holding"), "Source must be a pre-decision active task")
    return status


def _g_context(value, elapsed_s):
    _require(isinstance(value, dict) and set(value) == set(GeometryContext.__dataclass_fields__),
             "Original G context fields are incomplete")
    context = GeometryContext(**deepcopy(value))
    _require(context.mode in MODES and 0 <= context.phase_started_s <= context.last_elapsed_s + 1e-9
             and context.last_elapsed_s <= elapsed_s + 1e-9, "G phase/time context is inconsistent")
    for name in ("locked_offset_m", "retreat_offset_m"):
        v = getattr(context, name)
        _require(v is None or (type(v) in (int, float) and math.isfinite(v)), "Nonfinite G offset")
    for name in ("reassessment_count", "saturation_count"):
        v = getattr(context, name); _require(type(v) is int and v >= 0, "Invalid G counter")
    _require(context.mode == "RETRACT" or
             (context.locked_offset_m is not None and context.retreat_offset_m is not None),
             "An active G contact mode requires its retained offsets")
    return context


def _state_diff(actual, expected):
    difference = actual - expected
    difference[..., 2] = (difference[..., 2] + math.pi) % (2 * math.pi) - math.pi
    return difference


def _within_replay(actual, expected, budgets):
    delta = _state_diff(actual, expected)
    for axes, key in (((0, 1), "position_m"), ((6, 7), "position_m"),
                      ((3, 4), "linear_speed_m_s"), ((8, 9), "linear_speed_m_s")):
        _require(float(np.linalg.norm(delta[..., axes], axis=-1).max()) <= budgets[key],
                 f"Recorded prefix {key} exceeds its existing replay budget")
    _require(float(np.abs(delta[..., 2]).max()) <= budgets["yaw_rad"] and
             float(np.abs(delta[..., 5]).max()) <= budgets["angular_speed_rad_s"],
             "Recorded prefix yaw/angular-speed replay budget exceeded")
    return np.max(np.abs(delta), axis=0).tolist() if delta.ndim == 2 else np.abs(delta).tolist()


def load_development_snapshot(source_folder: Path, tick: int, physical_config: dict,
                              profile: dict) -> DevelopmentSnapshot:
    """Load one archived pre-decision context; only absent engine state is replayed.

    Stored snapshots restore exactly in every declared field. Missing mid-epoch
    physical payloads are reconstructed from the initial complete snapshot and
    first ``tick`` actual applied commands. Every recorded boundary and raw
    substep is checked against the existing profile, with exactness also reported.
    The source's warm sequence is returned; choosing a shared warm=None policy
    belongs to the new candidate experiment, not this source loader.
    """
    _require(type(tick) is int and tick >= 0, "tick must be an exact nonnegative integer")
    folder = Path(source_folder)
    paths = {name: folder / name for name in
             ("manifest.json", "steps.jsonl", "raw_execution.npz", "raw_execution.json.gz")}
    stored_path = folder / "plans" / f"step-{tick:03d}" / "identity.json"
    if stored_path.is_file(): paths["stored_identity"] = stored_path
    before = {str(p): _digest(p) for p in paths.values()}
    manifest = _read_json(paths["manifest.json"]); steps = _read_steps(paths["steps.jsonl"])
    _require(tick < len(steps) and all(row["index"] == i for i, row in enumerate(steps)),
             "Source tick is absent or source step indices are discontinuous")
    original = deepcopy(steps[tick]["plan"])
    _require(isinstance(original, dict) and original["tick"] == tick,
             "Archived source lacks the selected compact model-plan record")
    _require(original["source_manifest_sha256"] == before[str(paths["manifest.json"])],
             "Compact plan does not bind its source manifest")
    contract = manifest["physical_contract"]
    _require(_canonical(contract["resolved_config"]) == _canonical(physical_config),
             "Requested physical config differs from the archived source")
    _require(_canonical(tomllib.loads(contract["profile_toml"])) == _canonical(profile),
             "Requested acceptance profile differs from the archived source")
    env = PlanarPushEnv(deepcopy(physical_config))
    _require(_canonical(make_contract(env)) == _canonical(contract),
             "Original complete physical/data contract does not match this runtime")
    with gzip.open(paths["raw_execution.json.gz"], "rt", encoding="utf-8") as source:
        raw = json.load(source, parse_constant=_nonfinite)
    _require(raw["diagnostic_nonfinite_paths"] == [], "Selected source has nonfinite raw diagnostics")
    with np.load(paths["raw_execution.npz"], allow_pickle=False) as z:
        arrays = {name: z[name].copy() for name in z.files}
    n = len(steps); substeps = int(physical_config["physics"]["frame_skip"])
    expected = {"state": (n, 10), "actual_dt": (n,), "action_requested": (n, 2),
                "action_applied": (n, 2), "substep_states": (n * substeps, 10),
                "substep_times": (n * substeps,),
                "initial_integration_state": env.snapshot().integration_state.shape}
    _require(len(raw["diagnostics"]) == n, "Raw source interval count is inconsistent")
    for name, shape in expected.items():
        _require(arrays[name].dtype == np.float64 and arrays[name].shape == shape
                 and np.isfinite(arrays[name]).all(), f"Invalid archived {name} array")
    initial = _snapshot(raw["initial_snapshot"], env)
    _require(np.array_equal(initial.integration_state, arrays["initial_integration_state"]),
             "Initial raw engine vector differs from its JSON snapshot")
    initial_observation = env.restore(initial)
    _require(snapshots_exact(initial, env.snapshot()), "Initial complete snapshot restore is not exact")
    task = TaskSpec.from_config(physical_config)
    _require(_canonical(original["task_spec"]) == _canonical(asdict(task)),
             "Source task spec differs from the physical contract")
    initial_status = start_task(initial_observation, task, env.action_dt)
    status = _task_status(original["task_status"])
    expected_status = asdict(initial_status) if tick == 0 else steps[tick - 1]["task"]
    _require(_canonical(asdict(status)) == _canonical(expected_status),
             "Pre-decision task status differs from initial/previous actual task")
    recorded_state = _physical_array(original["state"], (10,), "source state")
    source_first = raw["diagnostics"][tick]["samples"][0]
    _require(np.array_equal(recorded_state, np.asarray(source_first["state"], np.float64))
             and np.array_equal(recorded_state, np.asarray(steps[tick]["state_before"], np.float64)),
             "Selected original state differs from the recorded first raw sample")
    _require(abs(status.last_sim_time - source_first["sim_time"]) <= contract["dt_tolerance_s"],
             "Selected task clock differs from the recorded raw state")
    previous = np.zeros(2, np.float64) if tick == 0 else arrays["action_applied"][tick - 1].copy()
    pc = original["planner_context_before"]
    _require(set(pc) == {"previous_applied_action", "optimal_actions", "control_step"}
             and type(pc["control_step"]) is int and pc["control_step"] == tick,
             "Source planner tick/context fields are inconsistent")
    _require(np.array_equal(previous, _physical_array(pc["previous_applied_action"], (2,), "source previous action")),
             "Planner previous action was not the actual previously applied command")
    warm = None if pc["optimal_actions"] is None else _physical_array(
        pc["optimal_actions"], (manifest["planner_config"]["planner"]["horizon"], 2), "source warm sequence")
    context = PlannerContext(previous.copy(), warm, tick)
    gctx = _g_context(original["geometry_context_before"], status.elapsed_s)
    fraction = profile["numeric"]["replay_tolerance_fraction_of_task_tolerance"]
    _require(type(fraction) in (int, float) and math.isfinite(fraction) and fraction > 0,
             "Existing positive replay tolerance fraction is required")
    budgets = {"position_m": fraction * task.position_tolerance_m,
               "yaw_rad": fraction * task.yaw_tolerance_rad,
               "linear_speed_m_s": fraction * task.linear_speed_tolerance_m_s,
               "angular_speed_rad_s": fraction * task.angular_speed_tolerance_rad_s}
    prefix = []; stored = "physical_snapshot" in original
    if stored:
        _require("stored_identity" in paths and _canonical(_read_json(stored_path)) == _canonical(original),
                 "Stored full identity differs from its compact execution record")
        snapshot = _snapshot(original["physical_snapshot"], env)
        observation = env.restore(snapshot)
        _require(snapshots_exact(snapshot, env.snapshot()), "Stored complete snapshot restore is not exact")
        _require(np.array_equal(observation.state, recorded_state), "Stored snapshot observed state differs from source")
    else:
        for i in range(tick):
            before_state = env.observe().state
            _require(np.array_equal(np.asarray(steps[i]["applied_action"], np.float64), arrays["action_applied"][i])
                     and np.array_equal(np.asarray(steps[i]["requested_action"], np.float64), arrays["action_requested"][i]),
                     "Source command arrays differ from recorded executed commands")
            before_max = _within_replay(before_state, np.asarray(steps[i]["state_before"], np.float64), budgets)
            result = env.step(arrays["action_applied"][i].copy())
            _require(result.transition_valid is True and steps[i]["physical_valid"] is True
                     and result.observation.observation_valid is True,
                     "Source replay prefix contains an invalid physical transition")
            _require(np.array_equal(result.applied_action, arrays["action_applied"][i])
                     and abs(result.actual_dt - env.action_dt) <= contract["dt_tolerance_s"]
                     and abs(result.actual_dt - arrays["actual_dt"][i]) <= contract["dt_tolerance_s"],
                     "Prefix applied command/duration differs from source")
            span = slice(i * substeps, (i + 1) * substeps)
            maxima = _within_replay(result.substep_states, arrays["substep_states"][span], budgets)
            boundary_max = _within_replay(result.observation.state, arrays["state"][i], budgets)
            raw_source = raw["diagnostics"][i]["samples"]
            _require(len(raw_source) == substeps + 1
                     and np.array_equal(np.asarray([s["state"] for s in raw_source[1:]], np.float64), arrays["substep_states"][span])
                     and np.array_equal(np.asarray([s["sim_time"] for s in raw_source[1:]], np.float64), arrays["substep_times"][span]),
                     "Source JSON samples differ from stored raw substep arrays")
            max_time = float(np.abs(result.substep_times - arrays["substep_times"][span]).max())
            _require(max_time <= contract["dt_tolerance_s"]
                     and abs(result.observation.sim_time - steps[i]["task"]["last_sim_time"]) <= contract["dt_tolerance_s"],
                     "Prefix raw clock/boundary differs from source")
            geometry = check_contact_geometry(result.diagnostics["samples"], profile["physics_suite"]["actual_contact_penetration_max_m"])
            _require(geometry["pass"] and steps[i]["geometry"]["pass"], "Prefix contact geometry failed")
            prefix.append({"index": i, "pass": True, "raw_substeps": len(result.substep_times),
                "before_state_max_abs_component_difference": before_max,
                "boundary_max_abs_component_difference": boundary_max,
                "raw_max_abs_component_difference": maxima, "raw_time_max_abs_difference_s": max_time,
                "raw_state_exact": bool(np.array_equal(result.substep_states, arrays["substep_states"][span])),
                "raw_time_exact": bool(np.array_equal(result.substep_times, arrays["substep_times"][span])),
                "applied_action_exact": True, "duration_matches": True, "geometry": geometry})
        observation = env.observe(); snapshot = env.snapshot()
        _within_replay(observation.state, recorded_state, budgets)
        replay_env = PlanarPushEnv(deepcopy(physical_config)); again = replay_env.restore(deepcopy(snapshot))
        _require(snapshots_exact(snapshot, replay_env.snapshot()) and np.array_equal(again.state, observation.state),
                 "Derived complete snapshot cannot restore its resulting state exactly")
    _require(abs(observation.sim_time - status.last_sim_time) <= contract["dt_tolerance_s"],
             "Restored snapshot time differs from pre-decision task time")
    force_threshold = contract["active_contact_force_min_N"]
    active = []
    for sample in raw["diagnostics"][tick]["samples"]:
        forces = [c["normal_force_N"] for c in sample["contacts"] if c["pair"] == "pusher_object"]
        _require(np.isfinite(forces).all(), "Cannot classify nonfinite source contact forces")
        active.append(any(f > force_threshold for f in forces))
    _require(len(active) == substeps + 1, "Source regime requires every boundary and substep sample")
    regime = "sustained_contact" if all(active) else "transition" if any(active) else "noncontact"
    after = {str(p): _digest(p) for p in paths.values()}
    _require(before == after, "Archived source changed while constructing the input")
    proof = {"pass": True, "source_tick": tick, "source_case": manifest["case"]["episode_id"],
        "source_regime": regime, "source_active_samples": sum(active), "source_sample_count": len(active),
        "source_force_threshold_N": force_threshold, "source_regime_scope": "original recorded next interval only; new candidates need not preserve it",
        "snapshot_kind": "stored_complete" if stored else "derived_checked_applied_prefix",
        "complete_snapshot_restore_exact": True, "source_observed_state_exact": bool(np.array_equal(observation.state, recorded_state)),
        "source_time_difference_s": float(abs(observation.sim_time - status.last_sim_time)),
        "prefix_restore_intervals": len(prefix), "prefix": prefix, "replay_budgets": budgets,
        "actual_previous_action_exact": True, "task_matches_previous_actual_boundary": True,
        "source_files_unchanged": True, "candidate_future_intervals": 0, "formal_updates": 0,
        "scope": "Derived source has no original middle engine payload for exact comparison; checked raw prefix plus exact derived-state persistence. No independent family or new task-effect proof."}
    return DevelopmentSnapshot(deepcopy(snapshot), observation.state.copy(), task, status, context,
        gctx, deepcopy(manifest["case"]), original, proof, before)
