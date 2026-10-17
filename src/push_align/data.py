"""Episode storage and continuous fixed-dt windows for learned dynamics.

Task failures are retained. Only physically valid, geometry-valid, complete
transitions enter learning windows; removing an invalid row never joins time.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import tomllib

import mujoco
import numpy as np

from .config import CONFIG_PATH, ROOT
from .contact_geometry import check_contact_geometry
from .environment import STATE_FIELDS, SnapshotBundle, wrap_angle

DATA_SCHEMA = "push-align-episode-v1"
STATE_UNITS = ("m", "m", "rad", "m/s", "m/s", "rad/s", "m", "m", "m/s", "m/s")
SPLITS = {"train", "validation", "smoke-test", "test"}
PROFILE_PATH = ROOT / "configs/acceptance.v1.toml"


class DataContractError(ValueError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def contract_hash(contract):
    return hashlib.sha256(_json_bytes(contract)).hexdigest()


def _require(condition, message):
    if not condition:
        raise DataContractError(message)


def _identifier(value):
    _require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value),
             "IDs must be nonempty path-safe identifiers")
    return value


def make_contract(env, config_path=CONFIG_PATH, acceptance_path=PROFILE_PATH):
    config_text = Path(config_path).read_text(encoding="utf-8")
    profile_text = Path(acceptance_path).read_text(encoding="utf-8")
    profile = tomllib.loads(profile_text)
    _require(tomllib.loads(config_text) == env.config, "Config file does not describe this environment")
    _require(profile["physics_suite"]["check_contact_geometry"] is True, "Geometry checking must be enabled")
    snap = env.snapshot()
    contract = {
        "data_schema": DATA_SCHEMA, "state_schema": env.config["schema_version"],
        "state_fields": list(STATE_FIELDS), "state_units": list(STATE_UNITS),
        "coordinates": "world_SI", "raw_dtype": "float64",
        "action_fields": ["target_vx_world", "target_vy_world"], "action_units": ["m/s", "m/s"],
        "action_dt_s": env.action_dt, "physics_dt_s": float(env.model.opt.timestep),
        "frame_skip": env.config["physics"]["frame_skip"],
        "dt_tolerance_s": env.config["validity"]["dt_match_tolerance_s"],
        "action_norm_max_m_s": env.config["pusher"]["command_speed_norm_max_m_s"],
        "penetration_budget_m": profile["physics_suite"]["actual_contact_penetration_max_m"],
        "active_contact_force_min_N": env.config["evaluation"]["contact_active_normal_force_min_N"],
        "config_id": env.config["config_id"], "config_toml": config_text,
        "config_sha256": digest(config_path), "resolved_config": deepcopy(env.config),
        "profile_id": profile["profile_id"], "profile_toml": profile_text,
        "profile_sha256": digest(acceptance_path), "resolved_xml": env.resolved_xml,
        "model_hash": env.model_hash, "environment_hash": env.environment_hash,
        "compiled_model_hash": env._compiled_fingerprint(),
        "snapshot_state_spec": int(mujoco.mjtState.mjSTATE_INTEGRATION),
        "snapshot_size": len(snap.integration_state), "warning_size": len(snap.warning_counts),
        "versions": {"numpy": np.__version__, "mujoco": mujoco.__version__},
    }
    _validate_contract(contract)
    return contract


def _validate_contract(c):
    try:
        _require(c["data_schema"] == DATA_SCHEMA and c["state_schema"] == "push-align-state-v1", "Unsupported schema")
        _require(c["state_fields"] == list(STATE_FIELDS) and c["state_units"] == list(STATE_UNITS), "State columns/units mismatch")
        _require(c["coordinates"] == "world_SI" and c["raw_dtype"] == "float64", "Coordinate/dtype mismatch")
        _require(c["action_fields"] == ["target_vx_world", "target_vy_world"] and c["action_units"] == ["m/s", "m/s"], "Action schema mismatch")
        cfg, profile = tomllib.loads(c["config_toml"]), tomllib.loads(c["profile_toml"])
        _require(cfg == c["resolved_config"], "Resolved config mismatch")
        for text_key, hash_key in [("config_toml", "config_sha256"), ("profile_toml", "profile_sha256"), ("resolved_xml", "model_hash")]:
            _require(hashlib.sha256(c[text_key].encode()).hexdigest() == c[hash_key], f"{hash_key} mismatch")
        env_hash = hashlib.sha256((c["model_hash"] + json.dumps(cfg, sort_keys=True)).encode()).hexdigest()
        _require(env_hash == c["environment_hash"], "Environment hash mismatch")
        for key in ("model_hash", "environment_hash", "compiled_model_hash"):
            _require(re.fullmatch(r"[0-9a-f]{64}", c[key]), f"Invalid {key}")
        _require(cfg["config_id"] == c["config_id"] and profile["profile_id"] == c["profile_id"], "Config/profile ID mismatch")
        _require(profile["physics_suite"]["check_contact_geometry"] is True, "Missing geometry policy")
        bindings = {
            "state_schema": cfg["schema_version"], "frame_skip": cfg["physics"]["frame_skip"],
            "action_dt_s": cfg["physics"]["action_dt_s"], "physics_dt_s": cfg["physics"]["timestep_s"],
            "dt_tolerance_s": cfg["validity"]["dt_match_tolerance_s"],
            "action_norm_max_m_s": cfg["pusher"]["command_speed_norm_max_m_s"],
            "active_contact_force_min_N": cfg["evaluation"]["contact_active_normal_force_min_N"],
            "penetration_budget_m": profile["physics_suite"]["actual_contact_penetration_max_m"],
        }
        _require(all(c[k] == v for k, v in bindings.items()), "Config/profile derived values mismatch")
        _require(c["snapshot_state_spec"] == int(mujoco.mjtState.mjSTATE_INTEGRATION), "Snapshot state specification mismatch")
        _require(type(c["snapshot_size"]) is int and c["snapshot_size"] > 0 and type(c["warning_size"]) is int and c["warning_size"] > 0, "Invalid snapshot dimensions")
        _require(type(c["frame_skip"]) is int and c["frame_skip"] > 0, "Invalid frame_skip")
        for k in ("action_dt_s", "physics_dt_s", "dt_tolerance_s", "action_norm_max_m_s", "penetration_budget_m"):
            _require(np.isfinite(c[k]) and c[k] > 0, f"Invalid {k}")
        _require(abs(c["physics_dt_s"]*c["frame_skip"]-c["action_dt_s"]) <= c["dt_tolerance_s"], "Action time mismatch")
    except (KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        raise DataContractError(f"Malformed contract: {error}") from error


def _encode_nonfinite(value, path=(), paths=None, values=None):
    """JSON uses null; corresponding numeric NPZ values preserve NaN/±Inf."""
    paths = [] if paths is None else paths
    values = [] if values is None else values
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {k: _encode_nonfinite(v, path+(k,), paths, values) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_nonfinite(v, path+(i,), paths, values) for i, v in enumerate(value)]
    if isinstance(value, float) and not np.isfinite(value):
        paths.append(list(path)); values.append(value)
        return None
    return value


def _diagnostics(episode):
    result = deepcopy(episode.metadata["diagnostics"])
    paths = episode.metadata["diagnostic_nonfinite_paths"]
    values = episode.arrays["diagnostic_nonfinite_values"]
    _require(len(paths) == len(values) and not np.isfinite(values).any(), "Nonfinite diagnostics mapping mismatch")
    _require(len({json.dumps(p) for p in paths}) == len(paths), "Duplicate diagnostic nonfinite path")
    for path, value in zip(paths, values):
        _require(isinstance(path, list) and len(path) >= 2, "Invalid diagnostic path")
        target = result
        try:
            for key in path[:-1]:
                target = target[key]
            _require(target[path[-1]] is None, "Nonfinite placeholder mismatch")
            target[path[-1]] = float(value)
        except (KeyError, IndexError, TypeError) as error:
            raise DataContractError("Invalid diagnostic nonfinite path") from error
    return result


def _sample_failures(s, c, reference_height, include_integration=True):
    cfg, errors = c["resolved_config"], []
    state, velocity = np.asarray(s["state"], dtype=float), np.asarray(s["object_velocity_world"], dtype=float)
    corners, force = np.asarray(s["object_corners_xy"], dtype=float), np.asarray(s["actuator_force_N"], dtype=float)
    _require(state.shape == (10,) and velocity.shape == (6,) and corners.shape == (8, 2) and force.shape == (2,), "Malformed physical sample")
    _require(np.asarray(s["pusher_velocity_world"]).shape == (6,) and np.asarray(s["warning_counts"]).shape == (c["warning_size"],), "Malformed velocity/warning sample")
    _require(type(s["engine_finite"]) is bool, "Invalid engine_finite flag")
    numeric = [s["object_height_m"], s["tilt_rad"], *velocity, *state, *force, *corners.ravel()]
    if not s["engine_finite"] or not np.isfinite(state).all():
        return ["nonfinite_state"]
    if not np.isfinite(numeric).all():
        errors.append("nonfinite_state")
    if any(s["warning_counts"]): errors.append("physics_warning")
    x0, x1 = cfg["workspace"]["x_bounds_m"]; y0, y1 = cfg["workspace"]["y_bounds_m"]
    if np.any(corners[:, 0] < x0) or np.any(corners[:, 0] > x1) or np.any(corners[:, 1] < y0) or np.any(corners[:, 1] > y1): errors.append("object_corners_outside_workspace")
    r = cfg["pusher"]["radius_m"]
    if not (x0+r <= state[6] <= x1-r and y0+r <= state[7] <= y1-r): errors.append("pusher_extent_outside_workspace")
    v = cfg["validity"]
    if abs(s["object_height_m"]-reference_height) > v["height_deviation_from_settled_m"]: errors.append("height_deviation")
    if s["tilt_rad"] > v["tilt_max_rad"]: errors.append("tilt_exceeded")
    if abs(velocity[5]) > v["vertical_speed_abs_max_m_s"]: errors.append("vertical_speed_exceeded")
    if np.linalg.norm(velocity[:2]) > v["horizontal_angular_speed_norm_max_rad_s"]: errors.append("horizontal_angular_speed_exceeded")
    limit = cfg["pusher"]["actuator_force_limit_abs_N"] + 1e-12
    if np.max(np.abs(force)) > limit: errors.append("actuator_force_exceeded")
    if include_integration and "integration_actuator_force_N" in s:
        f = np.asarray(s["integration_actuator_force_N"], dtype=float)
        _require(f.shape == (2,), "Invalid integration force shape")
        if not np.isfinite(f).all(): errors.append("nonfinite_state")
        if np.max(np.abs(f)) > limit: errors.append("integration_actuator_force_exceeded")
    return sorted(set(errors))


def _summaries(samples, c, reference):
    active = np.array([any(x["pair"] == "pusher_object" and x["normal_force_N"] > c["active_contact_force_min_N"] for x in s["contacts"]) for s in samples])
    forces = [x["normal_force_N"] for s in samples for x in s["contacts"] if x["pair"] == "pusher_object"]
    contact = [float(active[0]), float(active[-1]), float(active.mean()), max(forces or [0.])]
    validity = [max(s["tilt_rad"] for s in samples), max(abs(s["object_height_m"]-reference) for s in samples),
                max(abs(s["object_velocity_world"][5]) for s in samples),
                max(np.linalg.norm(s["object_velocity_world"][:2]) for s in samples),
                max(sum(s["warning_counts"]) for s in samples)]
    return np.array(contact, dtype=np.float64), np.array(validity, dtype=np.float64)


@dataclass(frozen=True)
class Episode:
    arrays: dict[str, np.ndarray]
    metadata: dict

    @property
    def transition_count(self):
        return len(self.arrays["actual_dt"])


class EpisodeRecorder:
    """Own every step until finish; external reset/restore/step is rejected."""
    def __init__(self, env, *, episode_id, episode_index, family_id, family_definition,
                 split, collection_policy, seed, contract=None):
        self.env = env
        self.contract = deepcopy(contract if contract is not None else make_contract(env))
        _validate_contract(self.contract)
        _require(self.contract["environment_hash"] == env.environment_hash and self.contract["compiled_model_hash"] == env._compiled_fingerprint(), "Recorder environment mismatch")
        _identifier(episode_id); _identifier(family_id)
        _require(type(episode_index) is int and episode_index >= 0 and type(seed) is int and seed >= 0, "Index/seed must be nonnegative integers")
        _require(split in SPLITS and isinstance(collection_policy, dict) and bool(collection_policy), "Split/policy required")
        _require(isinstance(family_definition, dict) and isinstance(family_definition.get("base_initial_spec"), dict) and bool(family_definition["base_initial_spec"]) and isinstance(family_definition.get("variant_rule"), str) and bool(family_definition["variant_rule"]), "Explicit family lineage required")
        _json_bytes(family_definition); _json_bytes(collection_policy)
        observation = env.observe()
        _require(observation.observation_valid, "Episode must start from a valid observation")
        self.generation, self.closed, self.invalid = env.state_generation, False, False
        self.states, self.times, self.observation_valid = [observation.state.copy()], [observation.sim_time], [True]
        self.snapshots, self.results, self.geometry = [env.snapshot()], [], []
        self.metadata = {"episode_id": episode_id, "episode_index": episode_index, "family_id": family_id,
                         "family_definition": deepcopy(family_definition), "split": split,
                         "collection_policy": deepcopy(collection_policy), "seed": seed,
                         "contract": self.contract, "contract_hash": contract_hash(self.contract),
                         "state_generation": self.generation, "initial_record": deepcopy(env.initial_record)}

    def _assert_live(self):
        _require(not self.closed and not self.invalid, "Episode is closed or invalid; finish it and start another")
        self._assert_owned_state()

    def _assert_owned_state(self):
        _require(self.env.state_generation == self.generation, "Reset/restore replaced the episode state")
        snap, previous = self.env.snapshot(), self.snapshots[-1]
        _require(self.env.config == self.contract["resolved_config"] and self.env._compiled_fingerprint() == self.contract["compiled_model_hash"], "Environment changed during episode")
        _require(np.array_equal(snap.integration_state, previous.integration_state, equal_nan=True) and np.array_equal(snap.warning_counts, previous.warning_counts) and np.array_equal(snap.warning_lastinfo, previous.warning_lastinfo) and snap.rng_state == previous.rng_state and snap.reference_height == previous.reference_height and snap.needs_reset == previous.needs_reset, "Unrecorded state change/step")

    def step(self, action):
        self._assert_live()
        result = self.env.step(action)
        geometry = check_contact_geometry(result.diagnostics["samples"], self.contract["penetration_budget_m"])
        self.results.append(deepcopy(result)); self.geometry.append(geometry)
        self.states.append(result.observation.state.copy()); self.times.append(result.observation.sim_time)
        self.observation_valid.append(bool(result.observation.observation_valid)); self.snapshots.append(self.env.snapshot())
        self.invalid = not (result.transition_valid and geometry["pass"])
        return result

    def finish(self, terminal_reason="collection_complete"):
        _require(not self.closed and bool(self.results), "Nonempty unfinished episode required")
        self._assert_owned_state()
        _require(isinstance(terminal_reason, str) and bool(terminal_reason), "Terminal reason required")
        rs, ss, c = self.results, self.snapshots, self.contract
        masks = [r.transition_valid and g["pass"] and r.observation.observation_valid and abs(r.actual_dt-c["action_dt_s"]) <= c["dt_tolerance_s"] for r, g in zip(rs, self.geometry)]
        if self.invalid:
            terminal_reason = "physical_invalid" if not rs[-1].transition_valid else "geometry_invalid"
        samples = [r.diagnostics["samples"] for r in rs]
        summaries = [_summaries(s, c, ss[i].reference_height) for i, s in enumerate(samples)]
        arrays = {
            "state": np.asarray(self.states, dtype=np.float64), "sim_time": np.asarray(self.times, dtype=np.float64),
            "action_requested": np.array([r.requested_action for r in rs], dtype=np.float64),
            "action_applied": np.array([r.applied_action for r in rs], dtype=np.float64),
            "actual_dt": np.array([r.actual_dt for r in rs], dtype=np.float64),
            "step_valid": np.array([r.transition_valid for r in rs], dtype=bool),
            "geometry_valid": np.array([g["pass"] for g in self.geometry], dtype=bool),
            "observation_valid": np.array(self.observation_valid, dtype=bool),
            "transition_valid": np.array(masks, dtype=bool), "terminal": np.arange(len(rs)) == len(rs)-1,
            "raw_integration_state": np.array([s.integration_state for s in ss], dtype=np.float64),
            "raw_reference_height": np.array([s.reference_height for s in ss], dtype=np.float64),
            "raw_warning_counts": np.array([s.warning_counts for s in ss], dtype=np.int64),
            "raw_warning_lastinfo": np.array([s.warning_lastinfo for s in ss], dtype=np.int64),
            "raw_needs_reset": np.array([s.needs_reset for s in ss], dtype=bool),
            "substep_offsets": np.concatenate(([0], np.cumsum([len(r.substep_times) for r in rs]))).astype(np.int64),
            "substep_states": np.concatenate([r.substep_states for r in rs]).astype(np.float64),
            "substep_times": np.concatenate([r.substep_times for r in rs]).astype(np.float64),
            "contact_summary": np.array([s[0] for s in summaries], dtype=np.float64),
            "validity_summary": np.array([s[1] for s in summaries], dtype=np.float64),
        }
        paths, values = [], []
        diagnostics = _encode_nonfinite([r.diagnostics for r in rs], paths=paths, values=values)
        arrays["diagnostic_nonfinite_values"] = np.array(values, dtype=np.float64)
        metadata = deepcopy(self.metadata)
        metadata.update({"diagnostics": diagnostics, "diagnostic_nonfinite_paths": paths,
                         "geometry_checks": deepcopy(self.geometry), "snapshot_rng_states": [deepcopy(s.rng_state) for s in ss],
                         "terminal_reason": terminal_reason, "task_outcome": "not_evaluated",
                         "contact_summary_fields": ["active_start", "active_end", "active_fraction", "max_normal_force_N"],
                         "validity_summary_fields": ["max_tilt_rad", "max_height_deviation_m", "max_vertical_speed_m_s", "max_horizontal_angular_speed_rad_s", "max_warning_count"]})
        metadata = _encode_nonfinite(metadata)
        episode = Episode(arrays, metadata)
        validate_episode(episode, self.contract)
        self.closed = True
        return episode


def snapshot_from_episode(episode, index):
    a, m = episode.arrays, episode.metadata
    _require(type(index) is int and 0 <= index <= episode.transition_count, "Snapshot index out of range")
    return SnapshotBundle(a["raw_integration_state"][index].copy(), m["contract"]["model_hash"],
                          m["contract"]["environment_hash"], float(a["raw_reference_height"][index]),
                          deepcopy(m["snapshot_rng_states"][index]), a["raw_warning_counts"][index].copy(),
                          a["raw_warning_lastinfo"][index].copy(), bool(a["raw_needs_reset"][index]))


@lru_cache(maxsize=4)
def _snapshot_model(xml, compiled_hash):
    model = mujoco.MjModel.from_xml_string(xml)
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    _require(hashlib.sha256(buffer.tobytes()).hexdigest() == compiled_hash, "Compiled snapshot model mismatch")
    return model


def _validate_snapshots(episode, diagnostics):
    a, m = episode.arrays, episode.metadata; c = m["contract"]
    model = _snapshot_model(c["resolved_xml"], c["compiled_model_hash"])
    _require(mujoco.mj_stateSize(model, c["snapshot_state_spec"]) == c["snapshot_size"], "Engine snapshot size mismatch")
    data = mujoco.MjData(model)
    obj, pusher = model.body("object").id, model.body("pusher").id
    for i in range(episode.transition_count+1):
        sample = diagnostics[0]["samples"][0] if i == 0 else diagnostics[i-1]["samples"][-1]
        _require(np.array_equal(a["raw_warning_counts"][i], sample["warning_counts"]) and np.all(a["raw_warning_counts"][i] >= 0) and np.all(a["raw_warning_lastinfo"][i] >= 0), "Snapshot/sample warnings mismatch")
        try:
            rng = np.random.Generator(np.random.PCG64())
            rng.bit_generator.state = deepcopy(m["snapshot_rng_states"][i])
        except (TypeError, ValueError, KeyError, OverflowError) as error:
            raise DataContractError("Invalid snapshot RNG state") from error
        raw = a["raw_integration_state"][i]
        mujoco.mj_setState(model, data, raw, c["snapshot_state_spec"])
        _require(np.isfinite(data.time) and abs(data.time-a["sim_time"][i]) <= c["dt_tolerance_s"], "Snapshot/boundary time mismatch")
        if not np.isfinite(raw).all():
            _require(not a["observation_valid"][i], "Nonfinite valid snapshot")
            continue  # Retain invalid raw values; do not forward a nonfinite state.
        mujoco.mj_forward(model, data)
        ov, pv = np.empty(6), np.empty(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, obj, ov, 0)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, pusher, pv, 0)
        rot = data.xmat[obj].reshape(3, 3)
        state = np.array([*data.xpos[obj, :2], float(wrap_angle(np.arctan2(rot[1, 0], rot[0, 0]))),
                          *ov[3:5], ov[2], *data.xpos[pusher, :2], *pv[3:5]])
        _require(np.allclose(state, a["state"][i], rtol=0, atol=1e-12, equal_nan=True), "Snapshot/boundary observation mismatch")
        _require(abs(data.xpos[obj, 2]-sample["object_height_m"]) <= 1e-12 and np.allclose(ov, sample["object_velocity_world"], rtol=0, atol=1e-12), "Snapshot/raw physics mismatch")


def validate_episode(episode, expected_contract):
    """Recompute masks from raw diagnostics instead of trusting stored flags."""
    a, m = episode.arrays, episode.metadata
    try:
        c = m["contract"]; _validate_contract(c)
        _require(contract_hash(c) == m["contract_hash"] == contract_hash(expected_contract), "Expected contract/hash mismatch")
        _identifier(m["episode_id"]); _identifier(m["family_id"])
        _require(type(m["episode_index"]) is int and m["episode_index"] >= 0 and m["split"] in SPLITS, "Invalid episode identity/split")
        t = len(a["actual_dt"]); w = c["warning_size"]
        _require(t > 0, "Empty episode")
        float_shapes = {"state": (t+1, 10), "sim_time": (t+1,), "action_requested": (t, 2),
                        "action_applied": (t, 2), "actual_dt": (t,), "raw_integration_state": (t+1, c["snapshot_size"]),
                        "raw_reference_height": (t+1,), "contact_summary": (t, 4), "validity_summary": (t, 5)}
        for name, shape in float_shapes.items():
            _require(a[name].shape == shape and a[name].dtype == np.float64, f"{name} shape/dtype mismatch")
        for name in ("step_valid", "geometry_valid", "transition_valid", "terminal"):
            _require(a[name].shape == (t,) and a[name].dtype == bool, f"{name} shape/dtype mismatch")
        for name in ("observation_valid", "raw_needs_reset"):
            _require(a[name].shape == (t+1,) and a[name].dtype == bool, f"{name} shape/dtype mismatch")
        for name in ("raw_warning_counts", "raw_warning_lastinfo"):
            _require(a[name].shape == (t+1, w) and a[name].dtype == np.int64, f"{name} shape/dtype mismatch")
        _require(all(isinstance(x, np.ndarray) and not x.dtype.hasobject for x in a.values()), "Object/non-array storage forbidden")
        _require(a["diagnostic_nonfinite_values"].dtype == np.float64 and a["diagnostic_nonfinite_values"].ndim == 1, "Nonfinite raw array mismatch")
        _require(np.isfinite(a["sim_time"]).all() and np.isfinite(a["actual_dt"]).all() and np.all(a["actual_dt"] >= 0), "Invalid times")
        _require(np.all(np.abs(np.diff(a["sim_time"])-a["actual_dt"]) <= c["dt_tolerance_s"]), "Boundary time discontinuity")
        _require(np.isfinite(a["action_requested"]).all() and np.isfinite(a["action_applied"]).all(), "Nonfinite commands")
        for requested, applied in zip(a["action_requested"], a["action_applied"]):
            largest = np.max(np.abs(requested)); scaled = requested/largest if largest else requested
            expected = scaled*(c["action_norm_max_m_s"]/np.linalg.norm(scaled)) if largest and largest > c["action_norm_max_m_s"]/np.linalg.norm(scaled) else requested
            _require(np.allclose(expected, applied, rtol=0, atol=1e-15), "Requested/applied action mismatch")
        _require(np.array_equal(a["terminal"], np.arange(t) == t-1), "Terminal must occur only at final transition")
        _require(len(m["snapshot_rng_states"]) == t+1 and len(m["geometry_checks"]) == t, "Snapshot/geometry count mismatch")
        offsets = a["substep_offsets"]
        _require(offsets.shape == (t+1,) and offsets.dtype == np.int64 and offsets[0] == 0 and np.all(np.diff(offsets) >= 0), "Substep offsets mismatch")
        _require(a["substep_states"].shape == (offsets[-1], 10) and a["substep_states"].dtype == np.float64 and a["substep_times"].shape == (offsets[-1],) and a["substep_times"].dtype == np.float64, "Substep arrays mismatch")
        diagnostics = _diagnostics(episode)
        _require(len(diagnostics) == t, "Diagnostic count mismatch")
        derived_masks = []
        for i, d in enumerate(diagnostics):
            samples = d["samples"]
            _require(d["state_fields"] == list(STATE_FIELDS), "Diagnostic schema mismatch")
            lo, hi = offsets[i:i+2]; n = int(hi-lo)
            _require(len(samples) == n+1 and n <= c["frame_skip"], "Sample count mismatch")
            states = np.asarray([s["state"] for s in samples], dtype=float)
            times = np.asarray([s["sim_time"] for s in samples], dtype=float)
            _require(np.array_equal(states[0], a["state"][i], equal_nan=True) and np.array_equal(states[-1], a["state"][i+1], equal_nan=True), "Reset/endpoint state misjoin")
            _require(np.array_equal(states[1:], a["substep_states"][lo:hi], equal_nan=True) and np.array_equal(times[1:], a["substep_times"][lo:hi]), "Substep record mismatch")
            _require(np.isfinite(times).all() and abs(times[0]-a["sim_time"][i]) <= c["dt_tolerance_s"] and abs(times[-1]-a["sim_time"][i+1]) <= c["dt_tolerance_s"], "Sample/boundary time mismatch")
            _require(np.all(np.abs(np.diff(times)-c["physics_dt_s"]) <= c["dt_tolerance_s"]), "Substep time discontinuity")
            _require(abs(a["actual_dt"][i]-n*c["physics_dt_s"]) <= c["dt_tolerance_s"], "Actual dt/sample count mismatch")
            reference = a["raw_reference_height"][i]
            _require(np.isfinite(reference) and reference == a["raw_reference_height"][i+1], "Reference height changed within episode")
            failures = sorted(set(e for s in samples for e in _sample_failures(s, c, reference)))
            complete = n == c["frame_skip"] and abs(a["actual_dt"][i]-c["action_dt_s"]) <= c["dt_tolerance_s"]
            if not complete: failures = sorted(set(failures + ["partial_transition"]))
            _require(sorted(d["validity_violations"]) == failures, "Physical failure record mismatch")
            step_valid = not failures
            _require(bool(a["step_valid"][i]) == step_valid and d["terminal_type"] == ("completed_interval" if step_valid else "invalid"), "Stored physical mask/termination mismatch")
            endpoint_valid = not _sample_failures(samples[-1], c, reference, include_integration=False)
            _require(bool(a["observation_valid"][i+1]) == endpoint_valid and bool(a["raw_needs_reset"][i+1]) == (not step_valid), "Endpoint/reset flag mismatch")
            geometry = check_contact_geometry(samples, c["penetration_budget_m"])
            _require(_json_bytes(geometry) == _json_bytes(m["geometry_checks"][i]) and bool(a["geometry_valid"][i]) == geometry["pass"], "Geometry record/mask mismatch")
            contact, validity = _summaries(samples, c, reference)
            _require(np.array_equal(contact, a["contact_summary"][i], equal_nan=True) and np.array_equal(validity, a["validity_summary"][i], equal_nan=True), "Contact/validity summary mismatch")
            mask = step_valid and geometry["pass"] and endpoint_valid and complete and bool(a["observation_valid"][i])
            derived_masks.append(mask)
            _require(mask or i == t-1, "An invalid transition must end the episode")
        _require(bool(a["observation_valid"][0]) and not a["raw_needs_reset"][0] and not _sample_failures(diagnostics[0]["samples"][0], c, a["raw_reference_height"][0], include_integration=False), "Invalid initial boundary")
        _validate_snapshots(episode, diagnostics)
        _require(np.array_equal(a["transition_valid"], derived_masks), "Learning mask differs from raw evidence")
        _require(np.isfinite(a["state"][a["observation_valid"]]).all() and np.isfinite(a["raw_integration_state"][a["observation_valid"]]).all(), "Nonfinite valid boundary")
        _require(isinstance(m["terminal_reason"], str) and bool(m["terminal_reason"]), "Missing terminal reason")
        if not derived_masks[-1]:
            _require(m["terminal_reason"] == ("physical_invalid" if not a["step_valid"][-1] else "geometry_invalid"), "Invalid episode reason mismatch")
        _json_bytes(m)
        return {"transitions": t, "learning_transitions": int(sum(derived_masks)), "excluded_transitions": int(t-sum(derived_masks)), "terminal_reason": m["terminal_reason"]}
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise DataContractError(f"Malformed episode: {error}") from error


@dataclass(frozen=True)
class SequenceWindow:
    episode_id: str
    family_id: str
    split: str
    start: int
    states: np.ndarray
    actions: np.ndarray
    terminal: np.ndarray


def sequence_windows(episode, length, expected_contract):
    validate_episode(episode, expected_contract)
    _require(type(length) is int and length > 0, "Positive integer window length required")
    a, m = episode.arrays, episode.metadata
    return [SequenceWindow(m["episode_id"], m["family_id"], m["split"], i,
                           a["state"][i:i+length+1].copy(), a["action_applied"][i:i+length].copy(),
                           a["terminal"][i:i+length].copy())
            for i in range(episode.transition_count-length+1) if a["transition_valid"][i:i+length].all()]


def split_families(family_ids, split_counts, seed):
    families = sorted(family_ids)
    _require(len(families) == len(set(families)) and bool(families), "Unique family list required")
    for family in families: _identifier(family)
    _require(type(seed) is int and seed >= 0 and set(split_counts) <= SPLITS, "Invalid split seed/labels")
    _require(all(type(n) is int and n >= 0 for n in split_counts.values()) and sum(split_counts.values()) == len(families), "Split counts must cover all families")
    order = np.random.Generator(np.random.PCG64(seed)).permutation(families)
    result, offset = {}, 0
    for split in ("train", "validation", "smoke-test", "test"):
        count = split_counts.get(split, 0)
        result.update({str(f): split for f in order[offset:offset+count]}); offset += count
    return result


def validate_family_splits(episodes):
    ids, indices, assignments, definitions, bases, actual_initials = set(), set(), {}, {}, {}, {}
    for e in episodes:
        m = e.metadata; family, split = m["family_id"], m["split"]
        _require(m["episode_id"] not in ids and m["episode_index"] not in indices, "Duplicate episode ID/index")
        ids.add(m["episode_id"]); indices.add(m["episode_index"])
        _require(family not in assignments or assignments[family] == split, "Family leaks across splits")
        definition = m["family_definition"]
        _require(isinstance(definition.get("base_initial_spec"), dict) and bool(definition["base_initial_spec"]) and isinstance(definition.get("variant_rule"), str) and bool(definition["variant_rule"]), "Missing family lineage")
        encoded = _json_bytes(definition); base = _json_bytes(definition["base_initial_spec"])
        _require(family not in definitions or definitions[family] == encoded, "Family definition mismatch")
        _require(base not in bases or bases[base] == family, "Identical declared base initial condition uses different families")
        actual = _json_bytes(m["initial_record"]["initial_spec"])
        _require(actual not in actual_initials or actual_initials[actual] == family, "Repeated actual initial condition uses different families")
        actual_initials[actual] = family
        assignments[family] = split; definitions[family] = encoded; bases[base] = family
    return assignments


def _write_json(path, value):
    with Path(path).open("xb") as file:
        file.write(_json_bytes(value) + b"\n")


def _split_record(assignments, seed):
    counts = {split: sum(v == split for v in assignments.values()) for split in sorted(set(assignments.values()))}
    if seed is not None:
        _require(split_families(list(assignments), counts, seed) == assignments, "Declared split seed does not reproduce assignments")
    return {"split_algorithm": "explicit_family_assignments" if seed is None else "sorted_family_ids_numpy_PCG64_permutation",
            "split_seed": seed, "split_counts": counts}


def write_dataset(path, episodes, *, purpose="contract-check", split_seed=None):
    episodes = list(episodes); _require(bool(episodes), "Nonempty dataset required")
    _require(purpose in {"contract-check", "pilot", "main", "final-evaluation"}, "Unknown dataset purpose")
    c = episodes[0].metadata["contract"]
    reviews = [validate_episode(e, c) for e in episodes]
    assignments = validate_family_splits(episodes)
    _require(set(assignments.values()) == {"test"} if purpose == "final-evaluation" else "test" not in assignments.values(), "Final test must be separate from development data")
    split_record = _split_record(assignments, split_seed)
    path = Path(path); _identifier(path.name)
    path.mkdir(parents=True, exist_ok=False); folder = path/"episodes"; folder.mkdir()
    entries = []
    for e, review in zip(episodes, reviews):
        episode_id = e.metadata["episode_id"]
        npz, metadata_path = folder/f"{episode_id}.npz", folder/f"{episode_id}.json"
        with npz.open("xb") as file: np.savez_compressed(file, **e.arrays)
        meta = deepcopy(e.metadata); meta["npz_sha256"] = digest(npz)
        _write_json(metadata_path, meta)
        entries.append({"episode_id": episode_id, "episode_index": meta["episode_index"], "family_id": meta["family_id"],
                        "split": meta["split"], "npz": str(npz.relative_to(path)), "json": str(metadata_path.relative_to(path)),
                        "npz_sha256": digest(npz), "json_sha256": digest(metadata_path), "review": review})
    manifest = {"dataset_id": path.name, "data_schema": DATA_SCHEMA, "purpose": purpose,
                "contract": c, "contract_hash": contract_hash(c), "family_assignments": assignments,
                **split_record,
                "episodes": entries, "episode_count": len(entries),
                "transition_count": sum(r["transitions"] for r in reviews),
                "learning_transition_count": sum(r["learning_transitions"] for r in reviews),
                "excluded_transition_count": sum(r["excluded_transitions"] for r in reviews)}
    _write_json(path/"manifest.json", manifest)
    return manifest


def _read_json(path):
    def reject_constant(token): raise DataContractError(f"Nonstandard JSON number: {token}")
    def unique_pairs(pairs):
        result = {}
        for k, v in pairs:
            _require(k not in result, "Duplicate JSON key"); result[k] = v
        return result
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file, parse_constant=reject_constant, object_pairs_hook=unique_pairs)


def load_episode(npz_path, metadata_path, expected_contract):
    meta = _read_json(metadata_path)
    _require(digest(npz_path) == meta["npz_sha256"], "Episode NPZ hash mismatch")
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            _require(len(archive.files) == len(set(archive.files)), "Duplicate NPZ members")
            arrays = {k: archive[k].copy() for k in archive.files}
    except ValueError as error:
        raise DataContractError(f"Unsafe or invalid NPZ: {error}") from error
    episode = Episode(arrays, meta); validate_episode(episode, expected_contract)
    return episode


def iter_dataset(path, expected_contract):
    """Read one episode at a time; exhaust to validate all manifest totals."""
    path = Path(path); manifest = _read_json(path/"manifest.json")
    _require(manifest["data_schema"] == DATA_SCHEMA and manifest["dataset_id"] == path.name and manifest["contract_hash"] == contract_hash(expected_contract) == contract_hash(manifest["contract"]), "Dataset identity/contract mismatch")
    _require(manifest["purpose"] in {"contract-check", "pilot", "main", "final-evaluation"}, "Unknown dataset purpose")
    episodes, reviews, files = [], [], set()
    for entry in manifest["episodes"]:
        _identifier(entry["episode_id"])
        npz, meta = path/entry["npz"], path/entry["json"]
        _require(entry["npz"] == f"episodes/{entry['episode_id']}.npz" and entry["json"] == f"episodes/{entry['episode_id']}.json", "Manifest paths must match episode IDs")
        _require(npz.resolve().parent == (path/"episodes").resolve() and meta.resolve().parent == (path/"episodes").resolve(), "Episode path escapes dataset")
        _require(digest(npz) == entry["npz_sha256"] and digest(meta) == entry["json_sha256"], "Manifest artifact hash mismatch")
        e = load_episode(npz, meta, expected_contract)
        _require(all(entry[k] == e.metadata[k] for k in ("episode_id", "episode_index", "family_id", "split")), "Manifest/episode identity mismatch")
        review = validate_episode(e, expected_contract)
        _require(review == entry["review"], "Manifest review mismatch")
        identity = {k: deepcopy(e.metadata[k]) for k in ("episode_id", "episode_index", "family_id", "family_definition", "initial_record", "split")}
        episodes.append(Episode({}, identity)); reviews.append(review); files.update([npz.name, meta.name])
        yield e
    _require(files == {p.name for p in (path/"episodes").iterdir()}, "Unlisted or missing episode artifacts")
    assignments = validate_family_splits(episodes)
    _require(assignments == manifest["family_assignments"], "Manifest family assignment mismatch")
    _require(all(manifest[k] == v for k, v in _split_record(assignments, manifest["split_seed"]).items()), "Split provenance mismatch")
    _require(set(assignments.values()) == {"test"} if manifest["purpose"] == "final-evaluation" else "test" not in assignments.values(), "Final test mixed with development")
    totals = {"episode_count": len(episodes), "transition_count": sum(r["transitions"] for r in reviews),
              "learning_transition_count": sum(r["learning_transitions"] for r in reviews),
              "excluded_transition_count": sum(r["excluded_transitions"] for r in reviews)}
    _require(all(manifest[k] == v for k, v in totals.items()) and bool(episodes), "Manifest counts mismatch")


def load_dataset(path, expected_contract):
    return list(iter_dataset(path, expected_contract)), _read_json(Path(path)/"manifest.json")


class DatasetWriter:
    """Persist validated episodes without retaining their large raw traces.

    A failed/incomplete writer retains evidence but has no completed manifest.
    """
    def __init__(self, path, contract, *, purpose="pilot", split_seed=None):
        _validate_contract(contract)
        _require(purpose in {"contract-check", "pilot", "main", "final-evaluation"}, "Unknown dataset purpose")
        self.path, self.contract = Path(path), deepcopy(contract)
        _identifier(self.path.name)
        self.path.mkdir(parents=True, exist_ok=False)
        self.folder = self.path/"episodes"; self.folder.mkdir()
        self.purpose, self.seed, self.closed = purpose, split_seed, False
        self.identities, self.entries, self.reviews = [], [], []

    def append(self, episode):
        _require(not self.closed, "Dataset writer is closed")
        review = validate_episode(episode, self.contract)
        identity = {k: deepcopy(episode.metadata[k]) for k in ("episode_id", "episode_index", "family_id", "family_definition", "initial_record", "split")}
        identities = self.identities + [Episode({}, identity)]
        assignments = validate_family_splits(identities)
        _require(set(assignments.values()) == {"test"} if self.purpose == "final-evaluation" else "test" not in assignments.values(), "Final test mixed with development data")
        episode_id = identity["episode_id"]
        npz, metadata_path = self.folder/f"{episode_id}.npz", self.folder/f"{episode_id}.json"
        with npz.open("xb") as file: np.savez_compressed(file, **episode.arrays)
        meta = deepcopy(episode.metadata); meta["npz_sha256"] = digest(npz)
        _write_json(metadata_path, meta)
        self.entries.append({"episode_id": episode_id, "episode_index": identity["episode_index"],
                             "family_id": identity["family_id"], "split": identity["split"],
                             "npz": str(npz.relative_to(self.path)), "json": str(metadata_path.relative_to(self.path)),
                             "npz_sha256": digest(npz), "json_sha256": digest(metadata_path), "review": review})
        self.identities, self.reviews = identities, self.reviews + [review]

    def finish(self):
        _require(not self.closed and bool(self.entries), "Nonempty unfinished dataset required")
        assignments = validate_family_splits(self.identities)
        manifest = {"dataset_id": self.path.name, "data_schema": DATA_SCHEMA, "purpose": self.purpose,
                    "contract": self.contract, "contract_hash": contract_hash(self.contract),
                    "family_assignments": assignments, **_split_record(assignments, self.seed),
                    "episodes": self.entries, "episode_count": len(self.entries),
                    "transition_count": sum(r["transitions"] for r in self.reviews),
                    "learning_transition_count": sum(r["learning_transitions"] for r in self.reviews),
                    "excluded_transition_count": sum(r["excluded_transitions"] for r in self.reviews)}
        _write_json(self.path/"manifest.json", manifest)
        self.closed = True
        return manifest
