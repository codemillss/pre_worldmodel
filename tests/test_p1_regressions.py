"""Tests for defects found in the P1-06 prerequisite audit."""
from copy import deepcopy
import unittest
import mujoco
import numpy as np
from push_align import InitialSpec, PlanarPushEnv
from push_align.config import load_pilot_config
from push_align.environment import task_angle_error, wrap_angle


class P1RegressionTest(unittest.TestCase):
    def test_config_changes_compiled_physics_and_keeps_action_dt(self):
        c = load_pilot_config()
        c['physics'].update(timestep_s=.001, frame_skip=100)
        c['object']['mass_kg'] = .12
        c['contact']['floor_object_friction'][:2] = [.4, .4]
        env = PlanarPushEnv(c)
        self.assertEqual(env.model.opt.timestep, .001)
        self.assertAlmostEqual(env.model.body_mass[env.model.body('object').id], .12)
        np.testing.assert_allclose(env.model.pair_friction[0, :2], [.4, .4])
        step = env.step([.01, 0])
        self.assertEqual(len(step.substep_times), 100)
        self.assertAlmostEqual(step.actual_dt, .1)

    def test_world_angular_velocity_at_nonzero_roll(self):
        env = PlanarPushEnv()
        # Diagnostic state injection, outside the production action path.
        roll = np.deg2rad(4)
        env.data.qpos[3:7] = [np.cos(roll/2), np.sin(roll/2), 0, 0]
        env.data.qvel[3:6] = [0, .3, 0]
        mujoco.mj_forward(env.model, env.data)
        self.assertAlmostEqual(env.observe().state[5], .3*np.sin(roll), places=10)

    def test_default_pusher_rotates_with_object(self):
        env = PlanarPushEnv()
        yaw = np.deg2rad(15)
        s = env.reset(InitialSpec(object_yaw=yaw)).state
        direction = np.array([np.cos(yaw), np.sin(yaw)])
        np.testing.assert_allclose(s[6:8], -.051*direction, atol=1e-12)

    def test_validity_rejects_height_tilt_vertical_and_horizontal_rotation(self):
        env = PlanarPushEnv()
        cases = [('height_deviation', 'object_height_m', .04),
                 ('tilt_exceeded', 'tilt_rad', np.deg2rad(10)),
                 ('vertical_speed_exceeded', 'object_velocity_world', [0, 0, 0, 0, 0, .1]),
                 ('horizontal_angular_speed_exceeded', 'object_velocity_world', [.6, 0, 0, 0, 0, 0])]
        for reason, field, value in cases:
            with self.subTest(reason=reason):
                s = env._sample()
                s[field] = value
                self.assertIn(reason, env._validity(s))
        s = env._sample()
        s['object_velocity_world'] = [0, 0, .6, 0, 0, 0]
        self.assertNotIn('horizontal_angular_speed_exceeded', env._validity(s))

    def test_workspace_uses_extents(self):
        env = PlanarPushEnv()
        with self.assertRaisesRegex(ValueError, 'object_corners_outside_workspace'):
            env.reset(InitialSpec(object_x=.19))
        with self.assertRaisesRegex(ValueError, 'pusher_extent_outside_workspace'):
            env.reset(InitialSpec(pusher_x=.198, pusher_y=.1))

    def test_partial_transition_keeps_failure_and_blocks_continuing(self):
        env = PlanarPushEnv()
        env.reset(InitialSpec(pusher_x=.191, pusher_y=.1))
        step = env.step([.06, 0])
        self.assertFalse(step.transition_valid)
        self.assertLess(step.actual_dt, .1)
        self.assertIn('pusher_extent_outside_workspace', step.diagnostics['validity_violations'])
        self.assertEqual(len(step.substep_times), len(step.substep_states))
        with self.assertRaises(RuntimeError):
            env.step([0, 0])

    def test_probe_does_not_convert_invalid_to_finite_success(self):
        import sys
        from push_align.config import ROOT
        sys.path.insert(0, str(ROOT / 'scripts'))
        from p1_contact_probe import run_condition
        env = PlanarPushEnv()
        # A narrow boundary intentionally tests recorder rejection, not physical task difficulty.
        env.config['workspace']['x_bounds_m'][1] = .0401
        result = run_condition(env, 0., 'central', {'probe_offset_abs_m':.015, 'probe_command_speed_m_s':.03, 'observation_duration_s':1.})
        self.assertFalse(result['transition_valid'])
        self.assertTrue(np.isfinite(result['final_state']).all())

    def test_snapshot_rejects_other_environment_and_restores_rng(self):
        env = PlanarPushEnv()
        snapshot = env.snapshot()
        random_a = env.rng.random()
        env.restore(snapshot)
        self.assertEqual(random_a, env.rng.random())
        c = load_pilot_config()
        c['physics'].update(timestep_s=.001, frame_skip=100)
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            PlanarPushEnv(c).restore(snapshot)

    def test_actual_speed_comes_from_engine_in_first_microstep(self):
        env = PlanarPushEnv()
        env.reset(InitialSpec(pusher_y=.1, pusher_x=-.1))
        result = env.step([.03, 0])
        self.assertLess(result.substep_states[0, 8], result.applied_action[0])
        self.assertGreater(result.substep_states[0, 8], 0.)
        np.testing.assert_allclose(result.observation.state[8:10], env.data.qvel[-2:])

    def test_boundary_and_pi_symmetry_are_different_contracts(self):
        self.assertAlmostEqual(float(task_angle_error(np.pi)), 0., places=10)
        self.assertAlmostEqual(abs(float(wrap_angle(np.pi))), np.pi, places=10)
        self.assertAlmostEqual(float(wrap_angle((-np.pi+1e-9)-(np.pi-1e-9))), 2e-9, places=12)

    def test_warning_and_nonfinite_are_not_valid(self):
        env = PlanarPushEnv()
        s = env._sample()
        s['warning_counts'][0] += 1
        self.assertIn('physics_warning', env._validity(s))
        s = env._sample()
        s['engine_finite'] = False
        self.assertIn('nonfinite_state', env._validity(s))

    def test_huge_and_bad_actions(self):
        env = PlanarPushEnv()
        applied = env._clip_action_norm(np.array([1e308, 1e308]))
        self.assertTrue(np.isfinite(applied).all())
        self.assertAlmostEqual(np.linalg.norm(applied), .06)
        for action in ([float('nan'), 0], [1], [0, 0, 0]):
            with self.assertRaises(ValueError):
                env.step(action)

    def test_dt_transfer_rejects_hidden_model_changes(self):
        env = PlanarPushEnv()
        other = PlanarPushEnv()
        other.model.opt.gravity[2] = -1.
        with self.assertRaisesRegex(ValueError, 'mutated'):
            other.import_dt_comparison_state(env)

    def test_force_peak_includes_integration_force(self):
        env = PlanarPushEnv()
        env.reset(InitialSpec(pusher_x=-.1, pusher_y=.1))
        result = env.step([.03, 0])
        self.assertAlmostEqual(result.diagnostics['max_abs_actuator_force'], .6)


if __name__ == '__main__':
    unittest.main()
