"""Independent angular loss, optimization failure, and train probe checks."""
import math
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from push_align import PlanarPushEnv
from push_align.data import make_contract
from push_align.training import (TrainingError, one_step_loss, select_tiny_subset,
                                 training_update)
from test_p2_learning_data import fixture


class BiasModel(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(10, dtype=dtype))

    def forward(self, features):
        return self.bias.expand(*features.shape[:-1], 10)


def probe_fixture():
    n = 36
    arrays = {
        "deltas": np.zeros((n, 10), dtype=np.float64),
        "split_code": np.array([0]*24+[1]*6+[2]*6, dtype=np.int8),
        "regime_code": np.array([1]*20+[0]*4+[1]*12, dtype=np.int8),
        "episode_index": np.arange(n, dtype=np.int64)//8,
        "transition_index": np.arange(n, dtype=np.int64)%8,
    }
    arrays["deltas"][:, 0] = np.arange(1, n+1)*.0001
    arrays["deltas"][:, 2] = (n-np.arange(n))*.001
    arrays["deltas"][18:20, 0] = .003  # a tie resolved by original identity
    arrays["deltas"][20:, :3] = 100.    # ignored noncontact/held-out extremes
    return SimpleNamespace(arrays=arrays)


class P2TrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.prepared, cls.normalizer = fixture(make_contract(PlanarPushEnv()))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.features = torch.tensor(self.prepared.arrays["features"][:2], dtype=torch.float64)
        self.mean = torch.tensor(self.normalizer.arrays["delta_mean"], dtype=torch.float64)
        self.scale = torch.tensor(self.normalizer.arrays["delta_scale"], dtype=torch.float64)

    def test_hand_computed_loss_and_normalized_parameter_gradient(self):
        model = BiasModel()
        target_normalized = torch.tensor([1., -2., .5, 0., 1., 0., -1., 2., 0., 1.], dtype=torch.float64)
        targets = (self.mean+self.scale*target_normalized).repeat(2, 1)
        loss = one_step_loss(model, self.normalizer, self.features, targets)
        self.assertAlmostEqual(float(loss.detach()), float(target_normalized.square().mean()), delta=1e-10)
        loss.backward()
        torch.testing.assert_close(model.bias.grad, (-2*target_normalized/10).float(), atol=1e-7, rtol=1e-6)

    def test_yaw_residual_wraps_before_scaling_instead_of_naive_normalized_mse(self):
        model = BiasModel()
        with torch.no_grad():
            model.bias[2] = (-math.pi+.01-self.mean[2])/self.scale[2]
        targets = self.mean.repeat(2, 1)
        targets[:, 2] = math.pi-.01
        physical_prediction = self.mean+self.scale*model.bias.detach().double()
        raw_difference = float(physical_prediction[2]-targets[0, 2])
        expected_wrapped = raw_difference+2*math.pi
        expected_loss = (expected_wrapped/float(self.scale[2]))**2/10
        naive_loss = (raw_difference/float(self.scale[2]))**2/10
        loss = one_step_loss(model, self.normalizer, self.features, targets)
        self.assertAlmostEqual(float(loss.detach()), expected_loss, delta=1e-8)
        self.assertGreater(naive_loss, 10000*expected_loss)
        loss.backward()
        expected_gradient = 2*expected_wrapped/float(self.scale[2])/10
        self.assertAlmostEqual(float(model.bias.grad[2]), expected_gradient, delta=1e-5)

    def test_repeated_updates_reduce_known_loss_and_report_preclip_norm(self):
        model = BiasModel()
        targets = (self.mean+self.scale).repeat(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.)
        initial = float(one_step_loss(model, self.normalizer, self.features, targets).detach())
        before = model.bias.detach().clone()
        first = training_update(model, optimizer, self.normalizer, self.features, targets, 10.)
        self.assertAlmostEqual(first["loss_before_update"], initial, delta=1e-10)
        self.assertAlmostEqual(first["gradient_norm_before_clip"], math.sqrt(.4), delta=1e-7)
        self.assertEqual(first["batch_size"], 2)
        for _ in range(39):
            training_update(model, optimizer, self.normalizer, self.features, targets, 10.)
        final = float(one_step_loss(model, self.normalizer, self.features, targets).detach())
        self.assertLess(final, initial*1e-6)
        self.assertFalse(torch.equal(model.bias, before))

    def test_loss_rejects_bad_batch_dtype_shape_empty_and_nonfinite(self):
        model = BiasModel()
        targets = self.mean.repeat(2, 1)
        cases = [(self.features.float(), targets), (self.features, targets.float()),
                 (self.features[:, :10], targets), (self.features, targets[:1]),
                 (self.features[:0], targets[:0]), (self.features.numpy(), targets),
                 (torch.full_like(self.features, float("nan")), targets),
                 (self.features, torch.full_like(targets, float("inf")))]
        for features, target in cases:
            with self.subTest(features_type=type(features).__name__), self.assertRaises(TrainingError):
                one_step_loss(model, self.normalizer, features, target)

    def test_update_rejects_nonfinite_gradient_parameter_and_invalid_clip(self):
        model = BiasModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.)
        targets = (self.mean+self.scale).repeat(2, 1)
        hook = model.bias.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
        with self.assertRaises(TrainingError):
            training_update(model, optimizer, self.normalizer, self.features, targets, 1.)
        hook.remove()
        class CorruptOptimizer(torch.optim.SGD):
            def step(self, closure=None):
                with torch.no_grad():
                    self.param_groups[0]["params"][0].fill_(float("inf"))
        bad = CorruptOptimizer(model.parameters(), lr=1.)
        with self.assertRaises(TrainingError):
            training_update(model, bad, self.normalizer, self.features, targets, 1.)
        for clip in [0., -1., float("nan"), float("inf"), True, "1"]:
            with self.subTest(clip=clip), self.assertRaises(TrainingError):
                fresh = BiasModel()
                training_update(fresh, torch.optim.SGD(fresh.parameters(), lr=1.),
                                self.normalizer, self.features, targets, clip)

    def test_subset_follows_train_contact_top_rules_and_identity_ties_after_shuffle(self):
        prepared = probe_fixture()
        selected = select_tiny_subset(prepared)
        expected = np.array([18, 19, 17, 16, 15, 14, 13, 12, 0, 1, 2, 3, 4, 5, 6, 7])
        np.testing.assert_array_equal(selected, expected)
        self.assertTrue(np.all(prepared.arrays["split_code"][selected] == 0))
        self.assertTrue(np.all(prepared.arrays["regime_code"][selected] != 0))
        self.assertEqual(len(np.unique(selected)), 16)
        order = np.random.default_rng(11).permutation(36)
        shuffled = SimpleNamespace(arrays={key: value[order] for key, value in prepared.arrays.items()})
        shuffled_selected = select_tiny_subset(shuffled)
        np.testing.assert_array_equal(order[shuffled_selected], expected)

    def test_subset_refuses_insufficient_rows_duplicate_identity_and_wrong_count(self):
        prepared = probe_fixture()
        small = SimpleNamespace(arrays={key: value[:15] for key, value in prepared.arrays.items()})
        with self.assertRaises(TrainingError):
            select_tiny_subset(small)
        duplicate = probe_fixture()
        duplicate.arrays["episode_index"][1] = duplicate.arrays["episode_index"][0]
        duplicate.arrays["transition_index"][1] = duplicate.arrays["transition_index"][0]
        with self.assertRaises(TrainingError):
            select_tiny_subset(duplicate)
        for count in [8, 15, 17, 16.]:
            with self.subTest(count=count), self.assertRaises(TrainingError):
                select_tiny_subset(prepared, count=count)

    def test_loss_rejects_float64_model_output_and_cpu_autocast(self):
        targets = self.mean.repeat(2, 1)
        with self.assertRaises(TrainingError):
            one_step_loss(BiasModel(torch.float64), self.normalizer, self.features, targets)
        with torch.autocast("cpu", dtype=torch.bfloat16), self.assertRaises(TrainingError):
            one_step_loss(BiasModel(), self.normalizer, self.features, targets)


if __name__ == "__main__":
    unittest.main()
