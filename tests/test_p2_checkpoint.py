"""Checkpoint restoration, continuation and adversarial contract checks.

The tiny synthetic fit fixture exercises persistence and optimization state;
it is not a physics trajectory or an accuracy/alignment evaluation.
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
from push_align.checkpoint import CheckpointError, load_checkpoint, save_checkpoint
from push_align.data import _json_bytes, make_contract
from push_align.dynamics import DeltaMLP, predict_next
from push_align.normalization import fit_normalizer, learning_schema
from push_align.representation import encode, state_delta


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False), encoding="utf-8")


def rebind_outer_hashes(path, metadata=None):
    """Rebind tampered containers to reach semantic/tensor/safe-pickle guards.

    This is an adversarial fixture, not a repair or trusted-identity claim.
    """
    path = Path(path)
    if metadata is None:
        metadata = json.loads((path/"checkpoint.json").read_text(encoding="utf-8"))
    metadata["files"] = {name: hashlib.sha256((path/name).read_bytes()).hexdigest()
                         for name in ("weights.pt", "training_state.pt")}
    metadata["checkpoint_hash"] = hashlib.sha256(
        _json_bytes({key: value for key, value in metadata.items() if key != "checkpoint_hash"})
    ).hexdigest()
    json_write(path/"checkpoint.json", metadata)


def unsafe_marker(path):
    """A deliberately unallowlisted pickle callable with a local-only marker."""
    Path(path).write_text("unsafe pickle executed", encoding="utf-8")
    return {}


class UnsafePickle:
    def __init__(self, path):
        self.path = str(path)

    def __reduce__(self):
        return unsafe_marker, (self.path,)


def tree_equal(case, left, right):
    """Compare Adam's nested tensors, including moments and step counters."""
    if isinstance(left, torch.Tensor):
        case.assertIsInstance(right, torch.Tensor)
        case.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        case.assertEqual(set(left), set(right))
        for key in left:
            tree_equal(case, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        case.assertEqual(type(left), type(right))
        case.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            tree_equal(case, a, b)
    else:
        case.assertEqual(left, right)


class P2CheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        contract = make_contract(PlanarPushEnv())
        cls.config = deepcopy(contract["resolved_config"])
        schema = learning_schema(contract)
        cls.state = np.zeros((12, 10), dtype=np.float64)
        cls.state[:, 0] = np.arange(12)*.0003
        cls.state[:, 2] = np.arange(12)*.03
        cls.state[:, 3] = np.linspace(.001, .004, 12)
        cls.state[:, 6] = cls.state[:, 0]-.052
        cls.state[:, 7] = np.linspace(-.008, .008, 12)
        cls.action = np.zeros((12, 2), dtype=np.float64)
        cls.action[:, 0] = np.linspace(.01, .02, 12)
        future = cls.state.copy()
        future[:, 0] += np.linspace(.0005, .0012, 12)
        future[:, 2] += np.linspace(-.002, .002, 12)
        future[:, 6] += .001
        features, deltas = encode(cls.state, cls.action), state_delta(cls.state, future)
        identity = {"dataset_id": "checkpoint-fixture", "manifest_sha256": "a"*64,
                    "data_contract_hash": schema["data_contract_hash"]}
        cls.normalizer = fit_normalizer(
            features, deltas, np.array([0]*8+[1]*4, dtype=np.int8), schema, identity,
            np.array([0]*8+[1]*4, dtype=np.int64), np.array(list(range(8))+list(range(4)), dtype=np.int64),
        )
        cls.provenance = {"learning_manifest_sha256": "b"*64,
                          "dataset_identity": deepcopy(identity),
                          "validation_split": "validation", "task_scope": "P2-04-pilot"}
        cls.x = torch.from_numpy(cls.normalizer.normalize_features(features[:8]))
        cls.y = torch.from_numpy(cls.normalizer.normalize_delta(deltas[:8]))

    def setUp(self):
        torch.manual_seed(11)
        self.model = DeltaMLP(self.config)
        t = self.config["training"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=t["learning_rate"],
                                          weight_decay=t["weight_decay"])
        self.generator = torch.Generator(device="cpu").manual_seed(17)
        self.update(self.model, self.optimizer)
        self.update(self.model, self.optimizer)

    def update(self, model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        loss = (model(self.x)-self.y).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config["training"]["gradient_clip_norm"])
        optimizer.step()
        return float(loss.detach())

    def saved(self, root, **overrides):
        path = Path(root)/"checkpoint"
        options = dict(epoch=1, updates=2, seed=11, shuffle_generator=self.generator,
                       selection_score=.8)
        options.update(overrides)
        metadata = save_checkpoint(path, self.model, self.optimizer, self.normalizer,
                                   self.config, self.provenance, **options)
        return path, metadata

    def loaded(self, path, **overrides):
        options = dict(normalizer=self.normalizer, config=self.config,
                       training_provenance=self.provenance)
        options.update(overrides)
        return load_checkpoint(path, **options)

    def test_inference_roundtrip_preserves_normalized_and_physical_prediction(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.saved(root)
            loaded = self.loaded(path)
            self.assertIsNone(loaded.optimizer)
            self.assertIsNone(loaded.shuffle_generator)
            self.assertIsNone(loaded.torch_rng_state)
            with torch.no_grad():
                torch.testing.assert_close(loaded.model(self.x), self.model(self.x), atol=1e-6, rtol=1e-5)
                state, action = torch.from_numpy(self.state[:3]), torch.from_numpy(self.action[:3])
                a = predict_next(self.model, self.normalizer, state, action)
                b = predict_next(loaded.model, self.normalizer, state, action)
                self.assertEqual(a.dtype, torch.float64)
                self.assertTrue(torch.equal(a, b))

    def test_resume_next_adam_update_and_moments_equal_uninterrupted(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.saved(root)
            loaded = self.loaded(path, resume=True)
            self.update(self.model, self.optimizer)
            self.update(loaded.model, loaded.optimizer)
            tree_equal(self, self.model.state_dict(), loaded.model.state_dict())
            tree_equal(self, self.optimizer.state_dict(), loaded.optimizer.state_dict())

    def test_resume_rng_states_continue_but_load_does_not_restore_global_rng(self):
        with tempfile.TemporaryDirectory() as root:
            saved_global = torch.get_rng_state().clone()
            saved_shuffle = self.generator.get_state().clone()
            path, _ = self.saved(root)
            expected_shuffle = torch.randperm(29, generator=self.generator)
            torch.rand(7)
            advanced_global = torch.get_rng_state().clone()
            loaded = self.loaded(path, resume=True)
            self.assertTrue(torch.equal(torch.get_rng_state(), advanced_global))
            self.assertTrue(torch.equal(loaded.torch_rng_state, saved_global))
            self.assertTrue(torch.equal(loaded.shuffle_generator.get_state(), saved_shuffle))
            self.assertTrue(torch.equal(torch.randperm(29, generator=loaded.shuffle_generator), expected_shuffle))
            torch.set_rng_state(saved_global)
            expected_global = torch.rand(5)
            torch.set_rng_state(loaded.torch_rng_state)
            self.assertTrue(torch.equal(torch.rand(5), expected_global))

    def test_saved_snapshot_does_not_follow_later_model_or_adam_mutations(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.saved(root)
            before = self.loaded(path, resume=True)
            expected_weights, expected_optimizer = deepcopy(before.model.state_dict()), deepcopy(before.optimizer.state_dict())
            self.update(self.model, self.optimizer)
            with torch.no_grad():
                next(self.model.parameters()).add_(3.)
            after = self.loaded(path, resume=True)
            tree_equal(self, expected_weights, after.model.state_dict())
            tree_equal(self, expected_optimizer, after.optimizer.state_dict())
            self.assertFalse(torch.equal(next(self.model.parameters()), next(after.model.parameters())))

    def test_semantic_same_shape_config_and_provenance_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.saved(root)
            variants = []
            for section, field, value in (("training", "learning_rate", .002),
                                          ("training", "batch_size", 128),
                                          ("runtime", "model_dtype", "float64"),
                                          ("model", "hidden_widths", [64, 64]),
                                          ("physics", "action_dt_s", .2),
                                          ("physics", "solver_iterations", 51),
                                          ("object", "mass_kg", .11),
                                          ("contact", "floor_object_friction", [.35, *self.config["contact"]["floor_object_friction"][1:]]),
                                          ("normalization", "input_scale_floor", [.002, *self.config["normalization"]["input_scale_floor"][1:]]),
                                          ("schema", "state_units", ["mm", *self.config["schema"]["state_units"][1:]]),
                                          ("schema", "feature_fields", list(reversed(self.config["schema"]["feature_fields"])))):
                changed = deepcopy(self.config); changed[section][field] = value
                variants.append({"config": changed})
            for field, value in (("learning_manifest_sha256", "c"*64),
                                 ("validation_split", "smoke-test"),
                                 ("task_scope", "different-task")):
                changed = deepcopy(self.provenance); changed[field] = value
                variants.append({"training_provenance": changed})
            for options in variants:
                with self.subTest(options=options), self.assertRaises(CheckpointError):
                    self.loaded(path, **options)

    def test_wrong_normalizer_same_shapes_is_rejected(self):
        changed = deepcopy(self.normalizer)
        changed.metadata["normalizer_hash"] = "c"*64
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.saved(root)
            with self.assertRaises(CheckpointError):
                self.loaded(path, normalizer=changed)

    def test_changed_weights_or_optimizer_file_bytes_are_rejected(self):
        for filename in ("weights.pt", "training_state.pt"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as root:
                path, _ = self.saved(root)
                with (path/filename).open("ab") as file:
                    file.write(b"tampered bytes")
                with self.assertRaises(CheckpointError):
                    self.loaded(path, resume=True)

    def test_invalid_or_nonfinite_training_counters_cannot_be_saved(self):
        for options in ({"epoch": -1}, {"updates": -1}, {"updates": True}, {"updates": 3},
                        {"seed": True}, {"selection_score": float("nan")},
                        {"selection_score": float("inf")}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as root:
                with self.assertRaises(CheckpointError):
                    self.saved(root, **options)

    def test_nonfinite_optimizer_moments_cannot_be_saved(self):
        state = next(iter(self.optimizer.state.values()))
        state["exp_avg"].reshape(-1)[0] = float("nan")
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(CheckpointError):
                self.saved(root)


    def test_optimizer_from_another_same_shaped_model_cannot_be_saved(self):
        other_model = DeltaMLP(self.config)
        # Correct shape, complete Adam moments and matching update count are
        # insufficient if those moments belong to another parameter set.
        other_optimizer = torch.optim.Adam(other_model.parameters(), lr=.001)
        self.update(other_model, other_optimizer)
        self.update(other_model, other_optimizer)
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(CheckpointError):
                save_checkpoint(Path(root)/"checkpoint", self.model, other_optimizer,
                                self.normalizer, self.config, self.provenance,
                                epoch=1, updates=2, seed=11,
                                shuffle_generator=self.generator, selection_score=.8)

    def test_fresh_save_cannot_bind_an_incompatible_config_to_existing_normalizer(self):
        for section, field, value in (
            ("physics", "action_dt_s", .2),
            ("physics", "solver_iterations", 51),
            ("object", "mass_kg", .11),
            ("contact", "floor_object_friction", [.35, *self.config["contact"]["floor_object_friction"][1:]]),
            ("normalization", "input_scale_floor", [.002, *self.config["normalization"]["input_scale_floor"][1:]]),
            ("schema", "state_units", ["mm", *self.config["schema"]["state_units"][1:]]),
            ("schema", "feature_fields", list(reversed(self.config["schema"]["feature_fields"]))),
        ):
            changed = deepcopy(self.config)
            changed[section][field] = value
            with self.subTest(section=section, field=field), tempfile.TemporaryDirectory() as root:
                with self.assertRaises(CheckpointError):
                    save_checkpoint(Path(root)/"checkpoint", self.model, self.optimizer,
                                    self.normalizer, changed, self.provenance,
                                    epoch=1, updates=2, seed=11,
                                    shuffle_generator=self.generator, selection_score=.8)


    def test_hash_valid_metadata_still_cannot_change_schema_counters_or_source(self):
        changes = (
            lambda m: m["learning_schema"].update(action_dt_s=.2),
            lambda m: m["learning_schema"].update(feature_fields=list(reversed(m["learning_schema"]["feature_fields"]))),
            lambda m: m.update(normalizer_hash="c"*64),
            lambda m: m["training_provenance"].update(learning_manifest_sha256="d"*64),
            lambda m: m.update(epoch=-1),
            lambda m: m.update(updates=True),
            lambda m: m.update(seed=-1),
        )
        for index, change in enumerate(changes):
            with self.subTest(change=index), tempfile.TemporaryDirectory() as root:
                path, metadata = self.saved(root)
                change(metadata)
                rebind_outer_hashes(path, metadata)
                with self.assertRaises(CheckpointError):
                    self.loaded(path, resume=True)

    def test_hash_valid_optimizer_requires_finite_complete_consistent_adam_and_rng(self):
        for defect in ("nan_moment", "negative_squared_moment", "step_mismatch", "missing_parameter", "bad_rng"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as root:
                path, _ = self.saved(root)
                state = torch.load(path/"training_state.pt", weights_only=True, map_location="cpu")
                key = min(state["optimizer"]["state"])
                entry = state["optimizer"]["state"][key]
                if defect == "nan_moment":
                    entry["exp_avg"].reshape(-1)[0] = float("nan")
                elif defect == "negative_squared_moment":
                    entry["exp_avg_sq"].reshape(-1)[0] = -1.
                elif defect == "step_mismatch":
                    entry["step"].fill_(3)
                elif defect == "missing_parameter":
                    del state["optimizer"]["state"][key]
                else:
                    state["shuffle_rng_state"] = torch.zeros(5, dtype=torch.float32)
                torch.save(state, path/"training_state.pt")
                rebind_outer_hashes(path)
                with self.assertRaises(CheckpointError):
                    self.loaded(path, resume=True)

    def test_hash_valid_unsafe_pickle_is_refused_without_executing_marker(self):
        for filename in ("weights.pt", "training_state.pt"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as root:
                path, _ = self.saved(root)
                marker = Path(root)/"unsafe-executed.txt"
                torch.save(UnsafePickle(marker), path/filename)
                rebind_outer_hashes(path)
                with self.assertRaises(CheckpointError):
                    self.loaded(path, resume=True)
                self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
