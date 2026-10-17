"""Synthetic checks for signed residual separation, not model performance."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from p4_local_prediction_diagnostic import local_error_decomposition

class TestLocalDecomposition(unittest.TestCase):
    def test_separates_known_local_bias_and_recursive_input_effect(self):
        actual = np.zeros((11, 10), np.float64)
        actual[:, 0] = np.arange(11)*.01
        free_delta = np.zeros((10, 10), np.float64)
        free_delta[:, 0] = .013
        free = np.zeros_like(actual)
        free[1:, 0] = np.cumsum(free_delta[:, 0])
        teacher = np.zeros((10, 10), np.float64)
        teacher[:, 0] = .011
        result = local_error_decomposition(actual, free, free_delta, teacher)
        np.testing.assert_allclose(result['local_error'][:, 0], .001, atol=1e-16)
        np.testing.assert_allclose(result['recursive_input_effect'][:, 0], .002, atol=1e-16)
        np.testing.assert_allclose(result['free_next_error'][:, 0], np.arange(1,11)*.003, atol=1e-16)

    def test_signed_cancellation_does_not_become_additive_norms(self):
        actual = np.zeros((11, 10), np.float64)
        free = actual.copy()
        teacher = np.zeros((10, 10), np.float64); teacher[:, :2] = [.001, -.002]
        result = local_error_decomposition(actual, free, np.zeros_like(teacher), teacher)
        np.testing.assert_array_equal(result['local_error']+result['recursive_input_effect'], np.zeros_like(teacher))
        self.assertTrue((np.linalg.norm(result['local_error'][:, :2], axis=1)>0).all())

    def test_principal_yaw_crossing_is_not_two_pi_error(self):
        actual = np.zeros((11,10), np.float64);actual[0,2]=np.pi-.01;actual[1:,2]=-np.pi+.01
        free=actual.copy()
        delta=np.zeros((10,10),np.float64);delta[0,2]=.02
        result=local_error_decomposition(actual,free,delta,delta)
        self.assertLess(np.max(np.abs(result['closure_residual'])),1e-14)

    def test_rejects_inconsistent_archived_delta(self):
        states=np.zeros((11,10),np.float64)
        delta=np.zeros((10,10),np.float64);delta[3,0]=.01
        with self.assertRaisesRegex(ValueError,'does not close'):
            local_error_decomposition(states,states,delta,np.zeros_like(delta))

    def test_missing_nonfinite_or_wrong_shape_does_not_pass(self):
        states=np.zeros((11,10),np.float64);delta=np.zeros((10,10),np.float64)
        for bad in (states[:-1], states.astype(np.float32), np.full_like(states,np.nan)):
            with self.subTest(shape=bad.shape):
                with self.assertRaisesRegex(ValueError,'Invalid actual_states'):
                    local_error_decomposition(bad,states,delta,delta)

if __name__=='__main__': unittest.main()
