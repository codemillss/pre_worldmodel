"""Common-MLP epoch-boundary persistence and independent-next-update checks.

Synthetic physical rows exercise training/persistence, not robot accuracy.
Identical one-step losses are a continuation control, not an assertion that
the later one-step and positive-weight rollout branches must stay identical.
The runner owns the registered settings keys, seeds and update budget; core
checkpoint tests verify exact binding rather than inventing runner rules.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from push_align import PlanarPushEnv
from push_align.checkpoint import (CheckpointError, load_checkpoint,
                                   make_optimizer, save_checkpoint)
from push_align.data import _json_bytes, make_contract
from push_align.dynamics import DeltaMLP, predict_delta
from push_align.normalization import fit_normalizer, learning_schema
from push_align.representation import encode, state_delta
from push_align.training import training_update


def tree_equal(case, left, right):
    if isinstance(left, torch.Tensor):
        case.assertIsInstance(right, torch.Tensor)
        case.assertEqual(left.dtype, right.dtype)
        case.assertEqual(left.device, right.device)
        case.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        case.assertEqual(set(left), set(right))
        for key in left:
            tree_equal(case, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        case.assertEqual(type(left), type(right))
        case.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            tree_equal(case, a, b)
    else:
        case.assertEqual(left, right)


def file_bytes(path):
    return {p.name: p.read_bytes() for p in Path(path).iterdir() if p.is_file()}


def rebind_untrusted_hashes(path, metadata=None):
    """Adversarial edits reach inner guards; this is not artifact repair."""
    path = Path(path)
    if metadata is None:
        metadata = json.loads((path / "checkpoint.json").read_text())
    metadata["files"] = {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                         for name in ("weights.pt", "training_state.pt")}
    metadata["checkpoint_hash"] = hashlib.sha256(_json_bytes({
        k: v for k, v in metadata.items() if k != "checkpoint_hash"
    })).hexdigest()
    (path / "checkpoint.json").write_text(json.dumps(metadata, sort_keys=True,
                                                     allow_nan=False), encoding="utf-8")


class P3CommonCheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        contract = make_contract(PlanarPushEnv())
        cls.config = deepcopy(contract["resolved_config"])
        cls.schema = learning_schema(contract)
        cls.state = np.zeros((12, 10), dtype=np.float64)
        cls.state[:, 0] = np.linspace(-.004, .004, 12)
        cls.state[:, 2] = np.linspace(-.15, .15, 12)
        cls.state[:, 3] = np.linspace(.001, .004, 12)
        cls.state[:, 6] = cls.state[:, 0] - .052
        cls.state[:, 7] = np.linspace(-.008, .008, 12)
        cls.action = np.zeros((12, 2), dtype=np.float64)
        cls.action[:, 0] = np.linspace(.01, .02, 12)
        cls.action[:, 1] = np.linspace(-.004, .003, 12)
        future = cls.state.copy()
        future[:, 0] += np.linspace(.0005, .0012, 12)
        future[:, 2] += np.linspace(-.002, .002, 12)
        future[:, 6] += .001
        cls.features = encode(cls.state, cls.action)
        cls.targets = state_delta(cls.state, future)
        cls.identity = {"dataset_id": "common-checkpoint-unit-fixture", "manifest_sha256": "a" * 64,
                        "data_contract_hash": cls.schema["data_contract_hash"]}
        cls.split = np.array([0] * 8 + [1] * 4, dtype=np.int8)
        cls.normalizer = cls.make_normalizer(cls.features)
        cls.provenance = {"learning_manifest_sha256": "b" * 64,
                          "dataset_identity": deepcopy(cls.identity),
                          "validation_split": "validation", "task_scope": "P3-04-main-common"}
        # Unit settings deliberately describe the small numerical fixture.
        # Official 3-seed/100-epoch/1300-update rules are runner tests.
        cls.settings = {"id": "synthetic-common-checkpoint-v1", "seed": 11,
                        "shuffle_seed": 1011, "epochs": 5, "batch_size": 4,
                        "fit_split": "train", "fit_count": 8,
                        "selection": {"horizon": 5, "earliest_exact_tie": True},
                        "loss": "angular_aware_one_step", "scope": "unit_fixture"}

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @classmethod
    def make_normalizer(cls, features):
        return fit_normalizer(
            features, cls.targets, cls.split, cls.schema, cls.identity,
            np.array([0] * 8 + [1] * 4, dtype=np.int64),
            np.array(list(range(8)) + list(range(4)), dtype=np.int64),
        )

    def setUp(self):
        self.prior_rng = torch.get_rng_state().clone()
        self.prior_dtype = torch.get_default_dtype()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.settings["seed"])
            self.model = DeltaMLP(self.config)
        self.optimizer = make_optimizer(self.model, self.config)
        self.generator = torch.Generator(device="cpu").manual_seed(self.settings["shuffle_seed"])
        self.x = torch.from_numpy(self.features.copy())
        self.y = torch.from_numpy(self.targets.copy())
        self.train = np.flatnonzero(self.split == 0)
        self.clip = self.config["training"]["gradient_clip_norm"]
        self.updates = 0
        for _ in range(2):
            order = self.train[torch.randperm(len(self.train), generator=self.generator).numpy()]
            for offset in range(0, len(order), 4):
                rows = order[offset:offset + 4]
                training_update(self.model, self.optimizer, self.normalizer,
                                self.x[rows], self.y[rows], self.clip)
                self.updates += 1

    def tearDown(self):
        torch.set_rng_state(self.prior_rng)
        torch.set_default_dtype(self.prior_dtype)

    def saved(self, directory, *, name="common", **overrides):
        path = Path(directory) / name
        arguments = {"epoch": 2, "updates": self.updates, "seed": 11,
                     "shuffle_generator": self.generator, "selection_score": .25,
                     "training_settings": self.settings}
        arguments.update(overrides)
        metadata = save_checkpoint(path, self.model, self.optimizer, self.normalizer,
                                   self.config, self.provenance, **arguments)
        return path, metadata

    def loaded(self, path, **overrides):
        arguments = {"normalizer": self.normalizer, "config": self.config,
                     "training_provenance": self.provenance,
                     "training_settings": self.settings, "resume": True}
        arguments.update(overrides)
        return load_checkpoint(path, **arguments)

    def test_two_independent_resumes_match_witness_next_original_batch_and_adam_update(self):
        with tempfile.TemporaryDirectory() as directory:
            path, metadata = self.saved(directory)
            before_files = file_bytes(path)
            before_normalizer = deepcopy(self.normalizer.metadata)
            before_x, before_y = self.x.clone(), self.y.clone()
            a, b = self.loaded(path), self.loaded(path)
            self.assertEqual(metadata["epoch"], 2)
            self.assertEqual(metadata["updates"], 4)
            for bundle in (a, b):
                tree_equal(self, self.model.state_dict(), bundle.model.state_dict())
                tree_equal(self, self.optimizer.state_dict(), bundle.optimizer.state_dict())
                self.assertTrue(any(bool(entry["exp_avg"].abs().sum() > 0)
                                    for entry in bundle.optimizer.state.values()))
            self.assertIsNot(a.shuffle_generator, b.shuffle_generator)
            self.assertTrue(all(x.data_ptr() != y.data_ptr() for x, y in zip(a.model.parameters(), b.model.parameters())))
            for key in a.optimizer.state_dict()["state"]:
                for moment in ("exp_avg", "exp_avg_sq"):
                    self.assertNotEqual(a.optimizer.state_dict()["state"][key][moment].data_ptr(),
                                        b.optimizer.state_dict()["state"][key][moment].data_ptr())
            orders = [self.train[torch.randperm(len(self.train), generator=g).numpy()]
                      for g in (self.generator, a.shuffle_generator, b.shuffle_generator)]
            for order in orders[1:]:
                np.testing.assert_array_equal(order, orders[0])
            torch.testing.assert_close(self.generator.get_state(), a.shuffle_generator.get_state(), rtol=0, atol=0)
            torch.testing.assert_close(a.shuffle_generator.get_state(), b.shuffle_generator.get_state(), rtol=0, atol=0)
            before_b = deepcopy(b.model.state_dict()), deepcopy(b.optimizer.state_dict())
            before_b_generator = b.shuffle_generator.get_state().clone()
            diagnostics = []
            for candidate, optimizer, order in ((self.model, self.optimizer, orders[0]),
                                                 (a.model, a.optimizer, orders[1]),
                                                 (b.model, b.optimizer, orders[2])):
                diagnostics.append(training_update(candidate, optimizer, self.normalizer,
                                                    self.x[order[:4]], self.y[order[:4]], self.clip))
                if candidate is a.model:
                    tree_equal(self, before_b[0], b.model.state_dict())
                    tree_equal(self, before_b[1], b.optimizer.state_dict())
                    torch.testing.assert_close(before_b_generator, b.shuffle_generator.get_state(), rtol=0, atol=0)
            self.assertEqual(diagnostics[0], diagnostics[1])
            self.assertEqual(diagnostics[1], diagnostics[2])
            for bundle in (a, b):
                tree_equal(self, self.model.state_dict(), bundle.model.state_dict())
                tree_equal(self, self.optimizer.state_dict(), bundle.optimizer.state_dict())
                self.assertTrue(all(float(s["step"]) == 5. for s in bundle.optimizer.state.values()))
            torch.testing.assert_close(self.x, before_x, rtol=0, atol=0)
            torch.testing.assert_close(self.y, before_y, rtol=0, atol=0)
            self.assertEqual(self.normalizer.metadata, before_normalizer)
            self.assertEqual(file_bytes(path), before_files)

    def test_fresh_adam_negative_control_changes_next_update_despite_same_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.saved(directory)
            continued = self.loaded(path)
            reset = self.loaded(path, resume=False)
            tree_equal(self, continued.model.state_dict(), reset.model.state_dict())
            fresh = make_optimizer(reset.model, self.config)
            self.assertEqual(fresh.state_dict()["state"], {})
            order = self.train[torch.randperm(len(self.train), generator=continued.shuffle_generator).numpy()][:4]
            for model, optimizer in ((continued.model, continued.optimizer), (reset.model, fresh)):
                training_update(model, optimizer, self.normalizer, self.x[order], self.y[order], self.clip)
            difference = max(float((x - y).abs().max().detach())
                             for x, y in zip(continued.model.parameters(), reset.model.parameters()))
            self.assertGreater(difference, 1e-8)

    def test_load_preserves_global_rng_but_caller_restores_saved_stream_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            saved_rng = torch.get_rng_state().clone()
            path, _ = self.saved(directory)
            for _ in range(2):
                torch.rand(13)
                before = torch.get_rng_state().clone()
                bundle = self.loaded(path)
                self.assertTrue(torch.equal(torch.get_rng_state(), before))
                self.assertTrue(torch.equal(bundle.torch_rng_state, saved_rng))
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(saved_rng)
                expected = torch.rand(9)
                torch.set_rng_state(bundle.torch_rng_state)
                self.assertTrue(torch.equal(expected, torch.rand(9)))
            # Current SiLU MLP has no stochastic forward. Random draw equality
            # is checked separately; weights alone could miss wrong RNG state.

    def test_advancing_one_branch_shuffle_cannot_advance_other_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.saved(directory)
            a, b = self.loaded(path), self.loaded(path)
            expected = torch.randperm(len(self.train), generator=self.generator)
            before_b = b.shuffle_generator.get_state().clone()
            first_a = torch.randperm(len(self.train), generator=a.shuffle_generator)
            second_a = torch.randperm(len(self.train), generator=a.shuffle_generator)
            torch.testing.assert_close(before_b, b.shuffle_generator.get_state(), rtol=0, atol=0)
            actual_b = torch.randperm(len(self.train), generator=b.shuffle_generator)
            torch.testing.assert_close(first_a, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual_b, expected, rtol=0, atol=0)
            self.assertFalse(torch.equal(second_a, actual_b))

    def test_inference_roundtrip_has_no_optimizer_and_preserves_physical_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            path, metadata = self.saved(directory)
            rng = torch.get_rng_state().clone()
            torch.set_default_dtype(torch.float64)
            restored = self.loaded(path, resume=False)
            self.assertIsNone(restored.optimizer)
            self.assertIsNone(restored.shuffle_generator)
            self.assertIsNone(restored.torch_rng_state)
            self.assertFalse(restored.model.training)
            self.assertEqual(torch.get_default_dtype(), torch.float64)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertTrue(all(p.dtype == torch.float32 and p.device.type == "cpu" for p in restored.model.parameters()))
            self.assertEqual(restored.metadata, metadata)
            state, action = torch.from_numpy(self.state[:3]), torch.from_numpy(self.action[:3])
            torch.testing.assert_close(predict_delta(self.model, self.normalizer, state, action),
                                       predict_delta(restored.model, self.normalizer, state, action), rtol=0, atol=0)

    def test_settings_canonical_hash_binds_nested_values_independent_of_key_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path, metadata = self.saved(directory)
            expected = hashlib.sha256(_json_bytes(self.settings)).hexdigest()
            self.assertEqual(metadata["training_settings_sha256"], expected)
            reverse = dict(reversed(list(self.settings.items())))
            reverse["selection"] = dict(reversed(list(reverse["selection"].items())))
            self.loaded(path, training_settings=reverse)
            changed = deepcopy(self.settings)
            changed["epochs"] = 5.0  # Python == is true, canonical bytes differ.
            self.assertEqual(changed, self.settings)
            with self.assertRaises(CheckpointError):
                self.loaded(path, training_settings=changed)
            changed = deepcopy(self.settings)
            changed["selection"]["earliest_exact_tie"] = 1
            self.assertEqual(changed, self.settings)  # bool/int equality is insufficient.
            with self.assertRaises(CheckpointError):
                self.loaded(path, training_settings=changed)

    def test_other_seed_budget_batch_or_loss_settings_rejected_for_resume_and_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.saved(directory)
            for key, value in (("seed", 23), ("shuffle_seed", 1023), ("epochs", 6),
                               ("batch_size", 8), ("loss", "positive_rollout_loss")):
                changed = deepcopy(self.settings)
                changed[key] = value
                for resume in (False, True):
                    with self.subTest(key=key, resume=resume), self.assertRaises(CheckpointError):
                        self.loaded(path, resume=resume, training_settings=changed)

    def test_p3_requires_settings_mapping_and_rejects_non_json_values_before_save(self):
        cyclic = {}; cyclic["self"] = cyclic
        values = [None, {}, [], {1: "bad-key"}, {"x": np.int64(1)}, {"x": np.array([1])},
                  {"x": float("nan")}, {"x": float("inf")}, {"x": (1, 2)}, cyclic]
        for value in values:
            with self.subTest(type=type(value).__name__), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(CheckpointError):
                    self.saved(directory, training_settings=value)
                self.assertFalse((Path(directory) / "common").exists())
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.saved(directory)
            for value in (None, {}, []):
                for resume in (False, True):
                    with self.subTest(value=value, resume=resume), self.assertRaises(CheckpointError):
                        self.loaded(path, training_settings=value, resume=resume)

    def test_mutating_input_settings_after_save_cannot_change_saved_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            original = deepcopy(self.settings)
            path, metadata = self.saved(directory, training_settings=original)
            before = file_bytes(path)
            original["selection"]["horizon"] = 99
            original["epochs"] = 99
            self.assertEqual(metadata["training_settings"], self.settings)
            self.loaded(path)
            self.assertEqual(file_bytes(path), before)

    def test_p2_legacy_format_unchanged_and_p2_p3_scope_mixing_rejected(self):
        legacy = deepcopy(self.provenance)
        legacy["task_scope"] = "P2-04-pilot"
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / "legacy"
            metadata = save_checkpoint(old, self.model, self.optimizer, self.normalizer, self.config,
                                       legacy, epoch=2, updates=4, seed=11,
                                       shuffle_generator=self.generator)
            self.assertEqual(metadata["format"], "push-align-checkpoint-v1")
            self.assertNotIn("training_settings", metadata)
            self.assertNotIn("training_settings_sha256", metadata)
            before = file_bytes(old)
            load_checkpoint(old, self.normalizer, self.config, legacy, resume=True)
            current, _ = self.saved(directory)
            for path, provenance, settings in ((old, self.provenance, self.settings),
                                                (current, legacy, None), (old, legacy, self.settings)):
                with self.subTest(scope=provenance["task_scope"]), self.assertRaises(CheckpointError):
                    load_checkpoint(path, self.normalizer, self.config, provenance,
                                    resume=True, training_settings=settings)
            with self.assertRaises(CheckpointError):
                save_checkpoint(Path(directory) / "wrong", self.model, self.optimizer,
                                self.normalizer, self.config, legacy, epoch=2, updates=4, seed=11,
                                shuffle_generator=self.generator, training_settings=self.settings)
            self.assertEqual(file_bytes(old), before)

    def test_rehashed_metadata_cannot_hide_settings_hash_scope_or_original_contract_mismatch(self):
        changes = [lambda m: m["training_settings"].update(epochs=6),
                   lambda m: m.update(training_settings_sha256="c" * 64),
                   lambda m: m.pop("training_settings"),
                   lambda m: m["training_provenance"].update(task_scope="P2-04-pilot"),
                   lambda m: m.update(normalizer_hash="d" * 64),
                   lambda m: m["config"]["physics"].update(action_dt_s=.2)]
        for index, change in enumerate(changes):
            with self.subTest(change=index), tempfile.TemporaryDirectory() as directory:
                path, metadata = self.saved(directory)
                change(metadata)
                rebind_untrusted_hashes(path, metadata)
                for resume in (False, True):
                    with self.subTest(resume=resume), self.assertRaises(CheckpointError):
                        self.loaded(path, resume=resume)

    def test_rehashed_settings_and_outer_hash_still_require_original_expected_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path, metadata = self.saved(directory)
            metadata["training_settings"]["epochs"] = 6
            metadata["training_settings_sha256"] = hashlib.sha256(
                _json_bytes(metadata["training_settings"])).hexdigest()
            rebind_untrusted_hashes(path, metadata)
            with self.assertRaises(CheckpointError):
                self.loaded(path)

    def test_valid_different_statistics_and_semantic_physical_config_are_rejected(self):
        changed_features = self.features.copy()
        changed_features[:8, 0] += .001
        changed_norm = self.make_normalizer(changed_features)
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.saved(directory)
            with self.assertRaises(CheckpointError):
                self.loaded(path, normalizer=changed_norm)
            for section, key, value in (("physics", "action_dt_s", .2),
                                       ("object", "mass_kg", .11),
                                       ("training", "learning_rate", .002)):
                changed = deepcopy(self.config)
                changed[section][key] = value
                with self.subTest(key=key), self.assertRaises(CheckpointError):
                    self.loaded(path, config=changed)

    def test_rehashed_bad_adam_moments_step_or_rng_rejected_during_resume(self):
        defects = ("nan_moment", "negative_squared_moment", "wrong_step", "missing_parameter",
                   "moment_dtype", "torch_rng_dtype", "shuffle_rng_length")
        for defect in defects:
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                path, _ = self.saved(directory)
                state = torch.load(path / "training_state.pt", weights_only=True, map_location="cpu")
                key = next(iter(state["optimizer"]["state"]))
                entry = state["optimizer"]["state"][key]
                if defect == "nan_moment": entry["exp_avg"].reshape(-1)[0] = float("nan")
                elif defect == "negative_squared_moment": entry["exp_avg_sq"].reshape(-1)[0] = -1.
                elif defect == "wrong_step": entry["step"] += 1.
                elif defect == "missing_parameter": del state["optimizer"]["state"][key]
                elif defect == "moment_dtype": entry["exp_avg"] = entry["exp_avg"].double()
                elif defect == "torch_rng_dtype": state["torch_rng_state"] = torch.zeros(5, dtype=torch.float32)
                else: state["shuffle_rng_state"] = torch.zeros(5, dtype=torch.uint8)
                torch.save(state, path / "training_state.pt")
                rebind_untrusted_hashes(path)
                with self.assertRaises(CheckpointError):
                    self.loaded(path)

    def test_rehashed_wrong_weight_shape_dtype_and_nonfinite_rejected(self):
        for defect in ("shape", "dtype", "nonfinite"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                path, _ = self.saved(directory)
                weights = torch.load(path / "weights.pt", weights_only=True, map_location="cpu")
                key = next(iter(weights))
                if defect == "shape": weights[key] = weights[key][:-1]
                elif defect == "dtype": weights[key] = weights[key].double()
                else: weights[key].reshape(-1)[0] = float("nan")
                torch.save(weights, path / "weights.pt")
                rebind_untrusted_hashes(path)
                for resume in (False, True):
                    with self.subTest(resume=resume), self.assertRaises(CheckpointError):
                        self.loaded(path, resume=resume)

    def test_save_requires_owned_optimizer_and_counter_matching_nonzero_adam_state(self):
        with tempfile.TemporaryDirectory() as directory:
            other = DeltaMLP(self.config)
            with self.assertRaises(CheckpointError):
                save_checkpoint(Path(directory) / "wrong", other, self.optimizer,
                                self.normalizer, self.config, self.provenance, epoch=2, updates=4,
                                seed=11, shuffle_generator=self.generator, training_settings=self.settings)
            for arguments in ({"updates": 5}, {"updates": True}, {"seed": True}, {"epoch": -1}):
                with self.subTest(arguments=arguments), self.assertRaises(CheckpointError):
                    self.saved(directory, **arguments)

    def test_existing_checkpoint_and_saved_snapshot_immutable_after_future_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.saved(directory)
            before_files = file_bytes(path)
            before = self.loaded(path)
            expected = deepcopy(before.model.state_dict()), deepcopy(before.optimizer.state_dict())
            training_update(self.model, self.optimizer, self.normalizer, self.x[:4], self.y[:4], self.clip)
            self.generator.manual_seed(999)
            after = self.loaded(path)
            tree_equal(self, expected[0], after.model.state_dict())
            tree_equal(self, expected[1], after.optimizer.state_dict())
            with self.assertRaises(FileExistsError):
                self.saved(directory, updates=5)
            self.assertEqual(file_bytes(path), before_files)


if __name__ == "__main__":
    unittest.main()
