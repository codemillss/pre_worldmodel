import unittest
import numpy as np
from push_align import InitialSpec, PlanarPushEnv

class P1ContractsTest(unittest.TestCase):
    def setUp(self): self.env = PlanarPushEnv()
    def test_scene_contract(self):
        self.assertAlmostEqual(self.env.model.opt.timestep, self.env.config['physics']['timestep_s'])
        self.assertEqual(self.env.model.nu, 2)
        self.assertEqual(tuple(self.env.model.actuator_forcerange[0]), (-2., 2.))
        self.assertEqual(tuple(self.env.model.actuator_forcerange[1]), (-2., 2.))
    def test_configured_initial_yaws_settle_without_invalid_observation(self):
        for yaw in np.deg2rad([-10., 0., 10.]):
            observation = self.env.reset(InitialSpec(object_yaw=float(yaw)))
            self.assertTrue(observation.observation_valid)
            self.assertTrue(np.isfinite(observation.state).all())
    def test_requested_and_applied_actions_are_distinct(self):
        result = self.env.step(np.array([.06, .06]))
        self.assertGreater(np.linalg.norm(result.requested_action), .06)
        self.assertAlmostEqual(np.linalg.norm(result.applied_action), .06, places=12)
        self.assertAlmostEqual(result.actual_dt, .1, places=12)
        self.assertTrue(result.transition_valid)
        self.assertEqual(result.substep_states.shape, (self.env.config['physics']['frame_skip'], 10))
    def test_actual_velocity_has_a_separate_state_field(self):
        result = self.env.step(np.array([.03, 0.]))
        self.assertEqual(result.applied_action.shape, (2,))
        self.assertEqual(result.observation.state[8:10].shape, (2,))
        self.assertIn("pusher_vx", result.diagnostics["state_fields"])
        self.assertIn("pusher_vy", result.diagnostics["state_fields"])
    def test_snapshot_replays_same_transition(self):
        self.env.reset(InitialSpec(object_yaw=.1))
        snapshot = self.env.snapshot()
        first = self.env.step(np.array([.02, 0.])).observation.state
        self.env.restore(snapshot)
        second = self.env.step(np.array([.02, 0.])).observation.state
        np.testing.assert_allclose(first, second, rtol=0, atol=1e-12)
    def test_yaw_wrap_keeps_boundary_increment_small(self):
        wrap = self.env._wrap
        self.assertAlmostEqual(wrap(-np.pi + 1e-10 - (np.pi - 1e-10)), 2e-10, places=12)

if __name__ == "__main__": unittest.main()
