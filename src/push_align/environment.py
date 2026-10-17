"""Headless contact dynamics; world-frame observations and explicit validity."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import itertools
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
import mujoco
import numpy as np
from .config import SCENE_PATH, load_pilot_config, scene_xml

STATE_FIELDS = ("obj_x", "obj_y", "obj_yaw", "obj_vx", "obj_vy", "obj_wz", "pusher_x", "pusher_y", "pusher_vx", "pusher_vy")

def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi

def task_angle_error(angle):
    return .5 * np.arctan2(np.sin(2 * np.asarray(angle)), np.cos(2 * np.asarray(angle)))

@dataclass(frozen=True)
class InitialSpec:
    object_x: float = 0.
    object_y: float = 0.
    object_yaw: float = 0.
    pusher_x: float | None = None
    pusher_y: float | None = None

@dataclass(frozen=True)
class Observation:
    state: np.ndarray
    sim_time: float
    schema_version: str
    observation_valid: bool

@dataclass(frozen=True)
class StepResult:
    requested_action: np.ndarray
    applied_action: np.ndarray
    observation: Observation
    actual_dt: float
    transition_valid: bool
    diagnostics: dict[str, Any]
    substep_states: np.ndarray
    substep_times: np.ndarray

@dataclass(frozen=True)
class SnapshotBundle:
    integration_state: np.ndarray
    model_hash: str
    environment_hash: str
    reference_height: float
    rng_state: dict
    warning_counts: np.ndarray
    warning_lastinfo: np.ndarray
    needs_reset: bool

class PlanarPushEnv:
    """State assignment is limited to reset and diagnostic restore.

    Task/planner state does not exist in P1. Future supervisors must wrap
    this physical snapshot with their own state; this is not that bundle.
    """
    def __init__(self, config: dict | None = None, scene_path: Path | None = None):
        self.config = deepcopy(config if config is not None else load_pilot_config())
        if tuple(self.config['schema']['state_fields']) != STATE_FIELDS:
            raise ValueError('Unsupported observation column order')
        self.resolved_xml = scene_xml(self.config, scene_path or SCENE_PATH)
        self.model_hash = hashlib.sha256(self.resolved_xml.encode()).hexdigest()
        comparison_spec = ET.fromstring(self.resolved_xml)
        comparison_spec.find('option').set('timestep', 'DT_COMPARISON_ONLY')
        self._dt_independent_spec = ET.tostring(comparison_spec)
        self.environment_hash = hashlib.sha256((self.model_hash + json.dumps(self.config, sort_keys=True)).encode()).hexdigest()
        self.model = mujoco.MjModel.from_xml_string(self.resolved_xml)
        self._compiled_hash = self._compiled_fingerprint()
        self.data = mujoco.MjData(self.model)
        self._object_body = self.model.body('object').id
        self._pusher_body = self.model.body('pusher').id
        self._object_geom = self.model.geom('object_geom').id
        self._pusher_geom = self.model.geom('pusher_geom').id
        self._floor_geom = self.model.geom('floor').id
        self._object_qposadr = int(self.model.jnt_qposadr[self.model.joint('object_free').id])
        self._pusher_qposadr = [int(self.model.jnt_qposadr[self.model.joint(n).id]) for n in ('pusher_x', 'pusher_y')]
        self._state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self.rng = np.random.default_rng(self.config['reproducibility']['base_seed'])
        self._reference_height = self.config['object']['size_full_xyz_m'][2]/2
        self._needs_reset = True
        self._state_generation = 0
        self.reset()

    @property
    def action_dt(self):
        return float(self.config['physics']['action_dt_s'])

    @property
    def state_generation(self):
        """Recorder lifecycle token; resets/restores replace an episode state.

        This counter is not physical state and is never restored from a
        snapshot, so even an identical reset cannot silently join episodes.
        """
        return self._state_generation

    def reset(self, initial: InitialSpec | None = None):
        initial = initial or InitialSpec()
        if not np.isfinite([initial.object_x, initial.object_y, initial.object_yaw]).all() or ((initial.pusher_x is None) != (initial.pusher_y is None)):
            raise ValueError('Finite initial pose and both or neither pusher coordinates required')
        px, py = self._default_pusher_position(initial)
        if not np.isfinite([px, py]).all():
            raise ValueError('Nonfinite pusher position')
        self._state_generation += 1
        self._needs_reset = True
        mujoco.mj_resetData(self.model, self.data)
        q = self._object_qposadr
        self.data.qpos[q:q+3] = [initial.object_x, initial.object_y, self.config['reset']['initial_object_center_z_m']]
        self.data.qpos[q+3:q+7] = [np.cos(initial.object_yaw/2), 0, 0, np.sin(initial.object_yaw/2)]
        self.data.qpos[self._pusher_qposadr] = [px, py]
        # mj_resetData initializes velocity/ctrl; never zero velocity after settling.
        self._reference_height = self.config['object']['size_full_xyz_m'][2]/2
        mujoco.mj_forward(self.model, self.data)
        pre = self._sample()
        failures = self._validity(pre, settling=True)
        if any(c['distance_m'] < 0 for c in pre['contacts']):
            failures.append('initial_penetration')
        if failures:
            raise ValueError(f'Invalid initial configuration: {failures}')
        self._settle()
        self._reference_height = float(self.data.xpos[self._object_body, 2])
        post = self._sample()
        if self._validity(post):
            raise RuntimeError(f'Invalid settled state: {self._validity(post)}')
        self._needs_reset = False
        self.initial_record = {'initial_spec': vars(initial), 'settle_time_s': float(self.data.time),
            'stable_time_s': self._stable_time, 'reference_height_m': self._reference_height,
            'settled_sample': post, 'model_hash': self.model_hash, 'environment_hash': self.environment_hash}
        return self._observation_from_sample(post)

    def _default_pusher_position(self, initial):
        if initial.pusher_x is not None:
            return float(initial.pusher_x), float(initial.pusher_y)
        direction = np.array([np.cos(initial.object_yaw), np.sin(initial.object_yaw)])
        tangent = np.array([-direction[1], direction[0]])
        distance = self.config['object']['size_full_xyz_m'][0]/2 + self.config['pusher']['radius_m'] + self.config['reset']['pusher_surface_gap_m']
        xy = np.array([initial.object_x, initial.object_y]) - distance*direction + self.config['reset']['initial_pusher_tangent_offset_m']*tangent
        return float(xy[0]), float(xy[1])

    def _velocity(self, body):
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
        return velocity  # world orientation; angular first, then linear

    def _sample(self):
        pos = self.data.xpos[self._object_body]
        rot = self.data.xmat[self._object_body].reshape(3, 3)
        pp = self.data.xpos[self._pusher_body]
        ov, pv = self._velocity(self._object_body), self._velocity(self._pusher_body)
        yaw = float(wrap_angle(np.arctan2(rot[1, 0], rot[0, 0])))
        state = [float(x) for x in (pos[0], pos[1], yaw, ov[3], ov[4], ov[2], pp[0], pp[1], pv[3], pv[4])]
        half = np.asarray(self.config['object']['size_full_xyz_m'])/2
        corners = np.array(list(itertools.product([-1, 1], repeat=3))) * half
        corners = corners @ rot.T + pos
        contacts = []
        force = np.empty(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {int(c.geom1), int(c.geom2)}
            label = 'pusher_object' if pair == {self._pusher_geom, self._object_geom} else 'floor_object' if pair == {self._floor_geom, self._object_geom} else 'other'
            mujoco.mj_contactForce(self.model, self.data, i, force)
            center_vector = self.data.geom_xpos[c.geom2] - self.data.geom_xpos[c.geom1]
            center_norm = float(np.linalg.norm(center_vector))
            contacts.append({'pair': label, 'normal_force_N': float(force[0]), 'distance_m': float(c.dist),
                             'friction': c.friction.tolist(), 'dimension': int(c.dim),
                             'normal_world': c.frame[:3].tolist(), 'geom_order': [int(c.geom1),int(c.geom2)],
                             'normal_center_cosine': float(c.frame[:3] @ center_vector / center_norm) if center_norm > 0 else None})
        return {'sim_time': float(self.data.time), 'state': state, 'object_height_m': float(pos[2]),
            'tilt_rad': float(np.arccos(np.clip(rot[2, 2], -1, 1))), 'object_velocity_world': ov.tolist(),
            'pusher_velocity_world': pv.tolist(), 'object_corners_xy': corners[:, :2].tolist(),
            'actuator_force_N': self.data.actuator_force.tolist(), 'contacts': contacts,
            'warning_counts': self.data.warning.number.tolist(),
            'engine_finite': bool(all(np.isfinite(a).all() for a in (self.data.qpos, self.data.qvel, self.data.qacc)))}

    def _validity(self, sample, settling=False):
        s, v = np.asarray(sample['state']), self.config['validity']
        if not sample['engine_finite'] or not np.isfinite(s).all():
            return ['nonfinite_state']
        errors = []
        if any(sample['warning_counts']):
            errors.append('physics_warning')
        corners = np.asarray(sample['object_corners_xy'])
        x0, x1 = self.config['workspace']['x_bounds_m']
        y0, y1 = self.config['workspace']['y_bounds_m']
        if np.any(corners[:, 0] < x0) or np.any(corners[:, 0] > x1) or np.any(corners[:, 1] < y0) or np.any(corners[:, 1] > y1):
            errors.append('object_corners_outside_workspace')
        r = self.config['pusher']['radius_m']
        if not (x0+r <= s[6] <= x1-r and y0+r <= s[7] <= y1-r):
            errors.append('pusher_extent_outside_workspace')
        if abs(sample['object_height_m'] - self._reference_height) > v['height_deviation_from_settled_m']:
            errors.append('height_deviation')
        if sample['tilt_rad'] > v['tilt_max_rad']:
            errors.append('tilt_exceeded')
        velocity = np.asarray(sample['object_velocity_world'])
        # Settling includes a physical fall; supported 3D rest is required before reset returns.
        if not settling and abs(velocity[5]) > v['vertical_speed_abs_max_m_s']:
            errors.append('vertical_speed_exceeded')
        if np.linalg.norm(velocity[:2]) > v['horizontal_angular_speed_norm_max_rad_s']:
            errors.append('horizontal_angular_speed_exceeded')
        if np.max(np.abs(sample['actuator_force_N'])) > self.config['pusher']['actuator_force_limit_abs_N'] + 1e-12:
            errors.append('actuator_force_exceeded')
        return errors

    def _observation_from_sample(self, sample):
        return Observation(np.array(sample['state']), sample['sim_time'], self.config['schema_version'], not self._validity(sample))

    def observe(self):
        return self._observation_from_sample(self._sample())

    _observation = observe  # compatibility with original P1 scripts
    _wrap = staticmethod(wrap_angle)

    def _settle(self):
        cfg = self.config['reset']
        stable_since = None
        self._stable_time = 0.
        while self.data.time < cfg['max_settle_time_s']:
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            sample = self._sample()
            failures = self._validity(sample, settling=True)
            if failures:
                raise RuntimeError(f'Settling failed: {failures}')
            ov, pv = np.asarray(sample['object_velocity_world']), np.asarray(sample['pusher_velocity_world'])
            supported = any(c['pair'] == 'floor_object' and c['normal_force_N'] > self.config['evaluation']['contact_active_normal_force_min_N'] for c in sample['contacts'])
            stable = supported and max(np.linalg.norm(ov[3:]), np.linalg.norm(pv[3:])) <= cfg['linear_speed_tolerance_m_s'] and np.linalg.norm(ov[:3]) <= cfg['angular_speed_tolerance_rad_s']
            stable_since = (sample['sim_time'] if stable_since is None else stable_since) if stable else None
            self._stable_time = 0. if stable_since is None else sample['sim_time'] - stable_since
            if self._stable_time + 1e-12 >= cfg['required_stable_time_s']:
                return
        raise RuntimeError('Reset did not reach supported 3D rest within max_settle_time_s')

    def _integration_solver_sample(self, start_time):
        """Read the solve used by mj_step, before mj_forward replaces its caches.

        Contact geometry belongs to the start of this integration interval;
        data.time has already advanced. No solve or state write occurs here.
        """
        records = []
        capacity = len(self.data.solver.gradient) // len(self.data.solver_niter)
        for island, count in enumerate(self.data.solver_niter):
            count = int(count)
            if not count:
                continue
            stored = min(count, capacity)
            stats = []
            for j in range(stored):
                index = island * capacity + j
                stats.append({name: float(getattr(self.data.solver, name)[index])
                              for name in ('gradient', 'improvement', 'lineslope', 'neval', 'nactive', 'nchange', 'nupdate')})
            records.append({'island': island, 'iterations': count, 'stored_iterations': stored,
                            'at_iteration_limit': count >= self.model.opt.iterations,
                            'statistics': stats})
        contacts = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            contacts.append({'geoms': [int(c.geom1), int(c.geom2)], 'distance_m': float(c.dist),
                             'force_contact_frame': force.tolist(), 'normal_force_N': float(force[0]),
                             'frame': c.frame.tolist(), 'position': c.pos.tolist(),
                             'active': bool(c.exclude == 0 and c.efc_address >= 0),
                             'dimension': int(c.dim), 'friction': c.friction.tolist()})
        return {'interval_start_s': float(start_time), 'interval_end_s': float(self.data.time),
                'geometry_time_s': float(start_time), 'solver_niter': self.data.solver_niter.tolist(),
                'statistics_capacity_per_island': capacity, 'islands': records, 'contacts': contacts,
                'constraint_count': int(self.data.nefc)}

    def step(self, requested_action, *, record_solver=False):
        if self._needs_reset:
            raise RuntimeError('Reset or restore required after invalid transition')
        requested = np.array(requested_action, dtype=np.float64, copy=True)
        if requested.shape != (2,) or not np.isfinite(requested).all():
            raise ValueError('Action must be finite world XY velocity with shape (2,)')
        applied = self._clip_action_norm(requested)
        start = float(self.data.time)
        samples = [self._sample()]
        violations = self._validity(samples[0])
        self.data.ctrl[:] = applied
        for _ in range(int(self.config['physics']['frame_skip'])):
            if violations:
                break
            previous = float(self.data.time)
            mujoco.mj_step(self.model, self.data)
            integration_force = self.data.actuator_force.copy()
            solver_sample = self._integration_solver_sample(previous) if record_solver else None
            mujoco.mj_forward(self.model, self.data)
            sample = self._sample()
            sample['integration_actuator_force_N'] = integration_force.tolist()
            if solver_sample is not None:
                sample['integration_solver'] = solver_sample
            samples.append(sample)
            violations.extend(self._validity(sample))
            if np.max(np.abs(integration_force)) > self.config['pusher']['actuator_force_limit_abs_N'] + 1e-12:
                violations.append('integration_actuator_force_exceeded')
            if abs(self.data.time - previous - self.model.opt.timestep) > self.config['validity']['dt_match_tolerance_s']:
                violations.append('physics_time_discontinuity')
        actual_dt = float(self.data.time - start)
        if abs(actual_dt - self.action_dt) > self.config['validity']['dt_match_tolerance_s']:
            violations.append('partial_transition')
        valid = not violations
        self._needs_reset = not valid
        contact_samples = [{'pusher_object_contact': any(c['pair']=='pusher_object' for c in s['contacts']),
                            'normal_force_N': max([c['normal_force_N'] for c in s['contacts'] if c['pair']=='pusher_object'] or [0.])} for s in samples]
        return StepResult(requested, applied, self._observation_from_sample(samples[-1]), actual_dt, valid,
            {'samples': samples, 'contact_samples': contact_samples, 'validity_violations': sorted(set(violations)),
             'state_fields': STATE_FIELDS, 'max_abs_actuator_force': max(max(abs(x) for x in s['actuator_force_N'] + s.get('integration_actuator_force_N', [])) for s in samples),
             'terminal_type': 'completed_interval' if valid else 'invalid'},
            np.asarray([s['state'] for s in samples[1:]], dtype=float).reshape(-1, 10),
            np.asarray([s['sim_time'] for s in samples[1:]]))

    def _clip_action_norm(self, action):
        largest = float(np.max(np.abs(action)))
        if largest == 0:
            return action.copy()
        scaled = action/largest
        scale_norm = float(np.linalg.norm(scaled))
        limit = self.config['pusher']['command_speed_norm_max_m_s']
        return action.copy() if largest <= limit/scale_norm else scaled*(limit/scale_norm)

    def snapshot(self):
        state = np.empty(mujoco.mj_stateSize(self.model, self._state_spec))
        mujoco.mj_getState(self.model, self.data, state, self._state_spec)
        return SnapshotBundle(state, self.model_hash, self.environment_hash, self._reference_height,
            deepcopy(self.rng.bit_generator.state), self.data.warning.number.copy(), self.data.warning.lastinfo.copy(), self._needs_reset)

    def restore(self, snapshot):
        if self._compiled_fingerprint() != self._compiled_hash:
            raise ValueError('Compiled model was mutated after construction')
        if snapshot.model_hash != self.model_hash or snapshot.environment_hash != self.environment_hash:
            raise ValueError('Snapshot environment mismatch')
        if snapshot.integration_state.shape != (mujoco.mj_stateSize(self.model, self._state_spec),):
            raise ValueError('Snapshot state shape mismatch')
        self._state_generation += 1
        mujoco.mj_setState(self.model, self.data, snapshot.integration_state, self._state_spec)
        self._reference_height = snapshot.reference_height
        self._needs_reset = snapshot.needs_reset
        self.rng.bit_generator.state = deepcopy(snapshot.rng_state)
        self.data.warning.number[:] = snapshot.warning_counts
        self.data.warning.lastinfo[:] = snapshot.warning_lastinfo
        warmstart = self.data.qacc_warmstart.copy()
        mujoco.mj_forward(self.model, self.data)
        self.data.qacc_warmstart[:] = warmstart
        return self.observe()

    def import_dt_comparison_state(self, source):
        """Diagnostic-only transfer: timestep/frame_skip may differ, nothing else.

        Both comparison branches call this before action sequences. Solver
        warmstart is zeroed in both branches, not copied between integrators.
        """
        if any(e._compiled_fingerprint() != e._compiled_hash for e in (self, source)):
            raise ValueError('Compiled model was mutated after construction')
        if self._dt_independent_spec != source._dt_independent_spec:
            raise ValueError('dt comparison scene differs beyond timestep')
        a, b = deepcopy(self.config), deepcopy(source.config)
        for c in (a, b):
            c['physics'].pop('timestep_s')
            c['physics'].pop('frame_skip')
        if a != b:
            raise ValueError('dt comparison can change only timestep and frame_skip')
        snap = source.snapshot()
        self._state_generation += 1
        mujoco.mj_setState(self.model, self.data, snap.integration_state, self._state_spec)
        self._reference_height = snap.reference_height
        self._needs_reset = snap.needs_reset
        self.data.warning.number[:] = snap.warning_counts
        self.data.warning.lastinfo[:] = snap.warning_lastinfo
        self.rng.bit_generator.state = deepcopy(snap.rng_state)
        self.data.qacc_warmstart[:] = 0.
        mujoco.mj_forward(self.model, self.data)
        self.data.qacc_warmstart[:] = 0.
        return self.observe()

    def _compiled_fingerprint(self):
        buffer = np.empty(mujoco.mj_sizeModel(self.model), dtype=np.uint8)
        mujoco.mj_saveModel(self.model, buffer=buffer)
        return hashlib.sha256(buffer.tobytes()).hexdigest()
