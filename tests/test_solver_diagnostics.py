"""Instrumentation must not alter the trajectory it is used to diagnose."""
from copy import deepcopy
import unittest
import numpy as np
from push_align import PlanarPushEnv, InitialSpec
from push_align.config import load_pilot_config


class SolverDiagnosticTest(unittest.TestCase):
    def test_collision_backend_flag_is_compiled_and_not_a_runtime_mutation(self):
        import mujoco
        config = deepcopy(load_pilot_config())
        config['physics']['native_ccd'] = True
        native = PlanarPushEnv(config)
        config['physics']['native_ccd'] = False
        legacy = PlanarPushEnv(config)
        bit = int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
        self.assertFalse(native.model.opt.disableflags & bit)
        self.assertTrue(legacy.model.opt.disableflags & bit)
        self.assertEqual(native.model.opt.timestep,legacy.model.opt.timestep)
        with self.assertRaisesRegex(ValueError,'scene differs'):
            legacy.import_dt_comparison_state(native)

    def test_elliptic_cone_changes_representation_and_keeps_friction_coefficients(self):
        import mujoco
        config=deepcopy(load_pilot_config())
        config['physics']['friction_cone']='pyramidal'
        base=PlanarPushEnv(config)
        config['physics']['native_ccd']=False
        config['physics']['friction_cone']='elliptic'
        ellipse=PlanarPushEnv(config)
        self.assertEqual(ellipse.model.opt.cone,mujoco.mjtCone.mjCONE_ELLIPTIC)
        np.testing.assert_array_equal(base.model.pair_friction,ellipse.model.pair_friction)

    def test_geometry_guard_rejects_nan_and_reversed_normals(self):
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from p1_physics_pilot import contact_geometry_check
        good={'pair':'pusher_object','distance_m':-1e-6,'normal_force_N':.1,'normal_world':[1.,0.,0.],'normal_center_cosine':1.}
        for field,value in [('distance_m',float('nan')),('normal_center_cosine',float('nan')),('normal_center_cosine',-1.)]:
            bad={**good,field:value}
            self.assertFalse(contact_geometry_check('fixture','dt',{'samples':[{'contacts':[bad]}]},.001)['pass'])
        self.assertTrue(contact_geometry_check('fixture','dt',{'samples':[{'contacts':[good]}]},.001)['pass'])

    def test_observer_preserves_microsteps_and_captures_solve_before_forward(self):
        config = deepcopy(load_pilot_config())
        config['physics']['timestep_s'] = .0005
        config['physics']['frame_skip'] = 200
        normal, recorded = PlanarPushEnv(config), PlanarPushEnv(config)
        normal.reset(InitialSpec(object_yaw=np.deg2rad(10)))
        snapshot = normal.snapshot()
        recorded.restore(snapshot)
        direction = .03*np.array([np.cos(np.deg2rad(10)),np.sin(np.deg2rad(10))])
        for _ in range(4):
            a,b = normal.step(direction), recorded.step(direction, record_solver=True)
            np.testing.assert_array_equal(a.substep_states,b.substep_states)
            np.testing.assert_array_equal(a.substep_times,b.substep_times)
            np.testing.assert_array_equal(normal.snapshot().integration_state,recorded.snapshot().integration_state)
            self.assertEqual(a.transition_valid,b.transition_valid)
            for sample in b.diagnostics['samples'][1:]:
                diag = sample['integration_solver']
                self.assertAlmostEqual(diag['interval_end_s']-diag['interval_start_s'],config['physics']['timestep_s'])
                self.assertEqual(diag['geometry_time_s'],diag['interval_start_s'])
                self.assertEqual(diag['interval_end_s'],sample['sim_time'])
                for island in diag['islands']:
                    self.assertEqual(len(island['statistics']),min(island['iterations'],diag['statistics_capacity_per_island']))


if __name__=='__main__': unittest.main()
