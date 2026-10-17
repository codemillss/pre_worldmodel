"""Bounded static contact-geometry diagnostic from an archived P1 snapshot.

This reproduces one pose with ordinary integration, then changes only CCD
flags on separately compiled models. It is not a trajectory acceptance test
or evidence of general superiority of any collision backend.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from push_align.config import ROOT
from push_align.environment import PlanarPushEnv, SnapshotBundle


REFERENCE = ROOT / 'runs/p1-06-20260928-05'
MICROSTEPS = 502


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save(path, value):
    with path.open('x') as file:
        json.dump(value, file, default=encode, ensure_ascii=False, allow_nan=False, indent=2)
        file.write('\n')


def canonical_xml(xml):
    root = ET.fromstring(xml)
    flag = root.find('option/flag')
    if flag is not None:
        flag.attrib.pop('nativeccd', None)
        flag.attrib.pop('multiccd', None)
        if not flag.attrib:
            root.find('option').remove(flag)
    return ET.tostring(root, encoding='unicode')


def variant_xml(xml, native, multi):
    root = ET.fromstring(xml)
    option = root.find('option')
    flag = option.find('flag')
    if flag is None:
        flag = ET.SubElement(option, 'flag')
    flag.set('nativeccd', 'enable' if native else 'disable')
    flag.set('multiccd', 'enable' if multi else 'disable')
    return ET.tostring(root, encoding='unicode')


def model_digest_except_ccd_flags(model):
    # This only normalizes fresh diagnostic models, before state transfer.
    # A full compiled-model hash catches changes beyond the selected flags.
    original = int(model.opt.disableflags)
    mask = int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD) | int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
    try:
        model.opt.disableflags = original & ~mask
        buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
        mujoco.mj_saveModel(model, buffer=buffer)
        return hashlib.sha256(buffer.tobytes()).hexdigest()
    finally:
        model.opt.disableflags = original


def contacts(model, data):
    result = []
    force = np.empty(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        mujoco.mj_contactForce(model, data, index, force)
        first, second = map(int, contact.geom)
        relative = data.geom_xpos[second] - data.geom_xpos[first]
        result.append({
            'index': index, 'geom_ids': [first, second],
            'geom_names': [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in (first, second)],
            'distance_m': float(contact.dist), 'position_world_m': contact.pos.copy(),
            'normal_geom1_to_geom2_world': contact.frame[:3].copy(),
            'frame_rows': contact.frame.reshape(3, 3).copy(),
            'normal_dot_center_displacement_m': float(contact.frame[:3] @ relative),
            'force_contact_frame_N_Nm': force.copy(),
            'force_world_N': contact.frame.reshape(3, 3).T @ force[:3],
            'dimension': int(contact.dim), 'friction': contact.friction.copy(),
            'efc_address': int(contact.efc_address),
        })
    return result


def state(data):
    return {'time_s': float(data.time), 'qpos': data.qpos.copy(), 'qvel': data.qvel.copy(),
            'ctrl': data.ctrl.copy(), 'qacc_warmstart': data.qacc_warmstart.copy()}


def geometry_check(model, data):
    og, pg = model.geom('object_geom').id, model.geom('pusher_geom').id
    rotation = data.geom_xmat[og].reshape(3, 3)
    object_position, pusher_position = data.geom_xpos[og], data.geom_xpos[pg]
    local = rotation.T @ (pusher_position - object_position)
    normal = rotation[:, 0] if local[0] < 0 else -rotation[:, 0]
    # Cylinder axis is world Z in this scene (XY slide joints only).
    cylinder_rotation = data.geom_xmat[pg].reshape(3, 3)
    axis = cylinder_rotation[:, 2]
    axial_projection = float(normal @ axis)
    radial_projection = np.sqrt(max(0., float(normal @ normal) - axial_projection**2))
    radius, half_height = model.geom_size[pg, :2]
    support = radius * radial_projection + half_height * abs(axial_projection)
    signed_clearance = abs(float(normal @ (pusher_position - object_position))) - model.geom_size[og, 0] - support
    return {
        'object_position_world_m': object_position.copy(), 'object_rotation_rows': rotation.copy(),
        'pusher_position_world_m': pusher_position.copy(), 'pusher_rotation_rows': cylinder_rotation.copy(),
        'center_distance_m': float(np.linalg.norm(pusher_position - object_position)),
        'xy_center_distance_m': float(np.linalg.norm((pusher_position-object_position)[:2])),
        'pusher_center_object_local_m': local,
        'facing_face_inward_normal_pusher_to_object': normal,
        'cylinder_support_along_face_normal_m': float(support),
        'facing_face_signed_clearance_m': float(signed_clearance),
        'separation_translation_upper_bound_along_face_normal_m': float(max(0., -signed_clearance)),
        'interpretation': 'This is a separating-axis translation upper bound, not an exact 3D closest distance. It applies to this box/cylinder pose and scene.',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    reference_paths = [REFERENCE / filename for filename in (
        'manifest.json', 'resolved_base.xml', 'yaw+0_central_snapshot.json',
        'yaw+0_central_replay0.json.gz')]
    source_paths = [Path(__file__).resolve(), ROOT/'src/push_align/config.py',
                    ROOT/'src/push_align/environment.py', ROOT/'assets/push_scene.xml']
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    save(out/'manifest.json', {
        'created_at': datetime.now().astimezone().isoformat(), 'argv': sys.argv,
        'executable': sys.executable, 'platform': platform.platform(),
        'versions': {'mujoco': mujoco.__version__, 'numpy': np.__version__},
        'reference_hashes': {str(p.relative_to(ROOT)): digest(p) for p in reference_paths},
        'source_hashes': source_hashes,
        'scope': 'One archived central-push pose, ordinary 502-step reproduction, then static CCD flag comparison. No environment acceptance or engine-wide claim.',
    })
    reference_manifest = json.loads(reference_paths[0].read_text())
    config = reference_manifest['config']
    archived_xml = reference_paths[1].read_text()
    saved = json.loads(reference_paths[2].read_text())
    env = PlanarPushEnv(config)
    if env.resolved_xml != archived_xml:
        raise ValueError('Current default compiled XML differs from the archived reference; diagnostic must not silently substitute a scene')
    array_fields = {'integration_state', 'warning_counts', 'warning_lastinfo'}
    snapshot = SnapshotBundle(**{f.name: np.asarray(saved[f.name]) if f.name in array_fields else saved[f.name]
                               for f in fields(SnapshotBundle)})
    env.restore(snapshot)
    initial = state(env.data)
    env.data.ctrl[:] = [.03, 0.]
    history = []
    for index in range(MICROSTEPS):
        # The only state assignment occurred at diagnostic restore above.
        mujoco.mj_step(env.model, env.data)
        integration_contacts = contacts(env.model, env.data)
        mujoco.mj_forward(env.model, env.data)
        history.append({'microstep': index+1, 'relative_time_s': float(env.data.time-initial['time_s']),
                        'integration_contacts': integration_contacts,
                        'post_forward_contacts': contacts(env.model, env.data)})
    reproduced = state(env.data)
    save(out/'reproduced_state.json', {'initial': initial, 'state_at_diagnostic_pose': reproduced,
                                      'geometry_check': geometry_check(env.model, env.data)})
    with gzip.open(out/'reproduction_contacts.json.gz', 'xt') as file:
        json.dump(history, file, default=encode, allow_nan=False, separators=(',', ':'))
    with gzip.open(reference_paths[3], 'rt') as file:
        archived_trace = json.load(file)
    target = archived_trace['samples'][MICROSTEPS]
    observation = env.observe().state
    archived_state_difference = float(np.max(np.abs(observation-np.asarray(target['state']))))
    archived_time_difference = abs(env.data.time-target['sim_time'])
    archived_contacts = [c for c in target['contacts'] if c['pair']=='pusher_object']
    diagnostic_contacts = [c for c in contacts(env.model, env.data) if set(c['geom_names'])=={'pusher_geom', 'object_geom'}]
    distance_match = np.array_equal(np.asarray([c['distance_m'] for c in archived_contacts]),
                                   np.asarray([c['distance_m'] for c in diagnostic_contacts]))
    base_signature = model_digest_except_ccd_flags(env.model)
    integration = np.empty(mujoco.mj_stateSize(env.model, env._state_spec))
    mujoco.mj_getState(env.model, env.data, integration, env._state_spec)
    variants = []
    for label, native, multi in [('native_multi', True, True), ('native_single', True, False), ('legacy_multi', False, True)]:
        xml = variant_xml(archived_xml, native, multi)
        model = mujoco.MjModel.from_xml_string(xml)
        xml_match = canonical_xml(xml) == canonical_xml(archived_xml)
        structure_match = model_digest_except_ccd_flags(model) == base_signature
        if not xml_match or not structure_match:
            raise ValueError(f'Variant {label} differs beyond selected CCD flags')
        data = mujoco.MjData(model)
        if mujoco.mj_stateSize(model, env._state_spec) != integration.size:
            raise ValueError('Diagnostic state layout changed')
        mujoco.mj_setState(model, data, integration, env._state_spec)
        before = state(data)
        identical_input = all(np.array_equal(before[k], reproduced[k]) for k in reproduced)
        mujoco.mj_forward(model, data)
        after = state(data)
        difference = {k: float(np.max(np.abs(np.asarray(after[k])-np.asarray(before[k])))) for k in before}
        save(out/f'{label}_contacts.json', {
            'label': label, 'requested_flags': {'nativeccd':native, 'multiccd':multi},
            'compiled_flags': {'disableflags': int(model.opt.disableflags), 'enableflags': int(model.opt.enableflags)},
            'xml_equal_except_ccd_flags': xml_match, 'compiled_model_equal_except_ccd_flags': structure_match,
            'identical_input_state': identical_input, 'input_state': before, 'after_forward_state': after,
            'forward_state_difference': difference, 'contacts': contacts(model, data),
            'geometry_check': geometry_check(model, data),
        })
        pusher_contacts = [c for c in contacts(model, data) if set(c['geom_names'])=={'pusher_geom', 'object_geom'}]
        expected = geometry_check(model, data)['facing_face_inward_normal_pusher_to_object']
        variants.append({'label': label, 'identical_input_state': identical_input,
                         'compiled_model_equal_except_ccd_flags': structure_match,
                         'pusher_contact_count': len(pusher_contacts),
                         'pusher_contact_distances_m': [c['distance_m'] for c in pusher_contacts],
                         'normal_dot_expected': [float(c['normal_geom1_to_geom2_world']@expected) for c in pusher_contacts],
                         'forward_state_difference': difference})
        with (out/f'{label}.xml').open('x') as file:
            file.write(xml)
    sources_unchanged = all(digest(ROOT/p)==h for p,h in source_hashes.items())
    reproduced_exactly = archived_state_difference==0 and archived_time_difference==0 and distance_match
    summary = {
        'run_kind': 'Static contact geometry diagnostic', 'microsteps': MICROSTEPS,
        'reproduced_archived_state_exactly': reproduced_exactly,
        'archived_observation_max_difference': archived_state_difference,
        'archived_time_difference_s': archived_time_difference,
        'archived_pusher_contact_distances_match': distance_match,
        'relative_time_s': float(env.data.time-initial['time_s']),
        'geometry_check': geometry_check(env.model, env.data), 'variants': variants,
        'sources_unchanged_during_run': sources_unchanged,
        'diagnostic_execution_verified': reproduced_exactly and sources_unchanged and all(v['identical_input_state'] and v['compiled_model_equal_except_ccd_flags'] for v in variants),
        'interpretation': 'Native CCD produces inconsistent contact geometry at this reproduced pose; legacy produces shallow contacts aligned with the facing surface. This does not identify every timestep failure, establish general backend superiority, or prove a confirmed upstream engine bug.',
        'P1_06_acceptance': 'NOT_ASSESSED', 'main_data_collection_authorized': False,
    }
    save(out/'summary.json', summary)
    save(out/'artifact_hashes.json', {p.name:digest(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({'output_dir':str(out), 'verified':summary['diagnostic_execution_verified'],
                      'reproduced_archived_state_exactly': reproduced_exactly, 'variants':variants}, default=encode))
    return 0 if summary['diagnostic_execution_verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
