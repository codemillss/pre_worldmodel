from pathlib import Path
import tomllib
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "pilot.v4.toml"
SCENE_PATH = ROOT / "assets" / "push_scene.xml"

def load_pilot_config() -> dict:
    with CONFIG_PATH.open("rb") as file:
        return tomllib.load(file)

def scene_xml(c: dict, template: Path = SCENE_PATH) -> str:
    """Resolve configurable physical values before compiling MuJoCo."""
    root = ET.fromstring(template.read_text())
    def set_values(xpath, **values):
        element = root.find(xpath)
        if element is None:
            raise ValueError(f"Scene template is missing {xpath}")
        for key, value in values.items():
            element.set(key, " ".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value))
    p, o, u, f = c['physics'], c['object'], c['pusher'], c['contact']
    if p['timestep_s'] <= 0 or not isinstance(p['frame_skip'], int) or p['frame_skip'] < 1:
        raise ValueError('Positive timestep and integer frame_skip are required')
    if abs(p['timestep_s'] * p['frame_skip'] - p['action_dt_s']) > c['validity']['dt_match_tolerance_s']:
        raise ValueError('timestep * frame_skip must equal action_dt')
    set_values('option', timestep=p['timestep_s'], gravity=p['gravity_m_s2'], integrator=p['integrator'],
               solver=p['solver'], iterations=p['solver_iterations'], tolerance=p['solver_tolerance'])
    if 'friction_cone' in p:
        if p['friction_cone'] not in ('pyramidal', 'elliptic'):
            raise ValueError('Unsupported friction cone')
        set_values('option', cone=p['friction_cone'])
    if 'native_ccd' in p:
        if not isinstance(p['native_ccd'], bool):
            raise ValueError('native_ccd must be a boolean')
        option = root.find('option')
        flag = option.find('flag')
        if flag is None:
            flag = ET.SubElement(option, 'flag')
        flag.set('nativeccd', 'enable' if p['native_ccd'] else 'disable')
    set_values('default/geom', condim=f['condim'], margin=f['margin_m'], gap=f['gap_m'], solref=f['solref'], solimp=f['solimp'])
    set_values("worldbody/body[@name='object']/geom", size=[x/2 for x in o['size_full_xyz_m']], mass=o['mass_kg'])
    set_values("worldbody/body[@name='pusher']", pos=[0, 0, u['center_z_m']])
    set_values("worldbody/body[@name='pusher']/geom", size=[u['radius_m'], u['height_full_m']/2], mass=u['mass_kg'])
    for axis in ('x', 'y'):
        set_values(f"worldbody/body[@name='pusher']/joint[@name='pusher_{axis}']", range=u['joint_range_xy_m'],
                   damping=u['joint_damping_N_s_m'], frictionloss=u['joint_frictionloss_N'])
        set_values(f"actuator/velocity[@name='pusher_v{axis}']", kv=u['velocity_gain_N_s_m'], gear=u['gear'],
                   forcerange=[-u['actuator_force_limit_abs_N'], u['actuator_force_limit_abs_N']])
    for geom, friction in [('floor', f['floor_object_friction']), ('pusher_geom', f['pusher_object_friction'])]:
        set_values(f"contact/pair[@geom1='{geom}']", friction=friction, condim=f['condim'],
                   solref=f['solref'], solimp=f['solimp'], margin=f['margin_m'], gap=f['gap_m'])
    return ET.tostring(root, encoding='unicode')
