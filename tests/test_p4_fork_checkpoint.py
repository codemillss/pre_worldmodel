"""P4 mid-epoch continuation and adverse provenance/tensor checks.

Small synthetic rows test persistence; they are not physical/accuracy evidence.
Actual Adam moments cover every parameter of the original registered MLP.
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
from push_align.checkpoint import CheckpointError, make_optimizer
from push_align.data import _json_bytes, make_contract
from push_align.dynamics import DeltaMLP, predict_delta
from push_align.fork_checkpoint import load_fork_checkpoint, save_fork_checkpoint
from push_align.fork_sampler import WindowSampler, sampler_state_hash
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
        case.assertEqual(type(left), type(right))
        case.assertEqual(left, right)


def rebind(path, metadata=None):
    """Untrusted container hashes are changed only to reach inner guards."""
    if metadata is None:
        metadata = json.loads((path / "checkpoint.json").read_text())
    metadata["files"] = {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                         for name in ("weights.pt", "training_state.pt")}
    metadata["checkpoint_hash"] = hashlib.sha256(_json_bytes({
        key: value for key, value in metadata.items() if key != "checkpoint_hash"
    })).hexdigest()
    (path / "checkpoint.json").write_text(json.dumps(metadata, sort_keys=True,
                                                     allow_nan=False), encoding="utf-8")


def unsafe_marker(path):
    Path(path).write_text("untrusted pickle executed", encoding="utf-8")
    return {}


class UnsafePickle:
    def __init__(self, path):
        self.path = str(path)

    def __reduce__(self):
        return unsafe_marker, (self.path,)


class ForkCheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
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
        future = cls.state.copy()
        future[:, 0] += np.linspace(.0005, .0012, 12)
        future[:, 2] += np.linspace(-.002, .002, 12)
        future[:, 6] += .001
        cls.features, cls.targets = encode(cls.state, cls.action), state_delta(cls.state, future)
        identity = {"dataset_id": "fork-checkpoint-unit-fixture", "manifest_sha256": "a" * 64,
                    "data_contract_hash": cls.schema["data_contract_hash"]}
        cls.normalizer = fit_normalizer(
            cls.features, cls.targets, np.array([0] * 9 + [1] * 3, dtype=np.int8),
            cls.schema, identity, np.array([0] * 9 + [1] * 3, dtype=np.int64),
            np.array(list(range(9)) + list(range(3)), dtype=np.int64))
        cls.provenance = {"learning_manifest_sha256": "b" * 64,
                          "dataset_identity": identity, "validation_split": "validation",
                          "task_scope": "P4-02-matched-fork"}
        cls.budget = {"training_id": "synthetic-fork-persistence-unit-v1", "seeds": [11],
                      "common_checkpoints": [{"seed": 11, "path": "synthetic/common/seed-11",
                                              "checkpoint_hash": "c" * 64, "initial_updates": 2}],
                      "additional_updates_per_branch": 6, "batch_windows": 4, "K": 5,
                      "drop_last": False, "window_count_train": 9,
                      "window_count_validation": 3, "checkpoint_every_updates": 2,
                      "lambda_candidates": [0.1, 1.0],
                      "branches_per_seed": ["S1", "SK-lambda0.1", "SK-lambda1.0"]}

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.rng = torch.get_rng_state().clone()
        self.dtype = torch.get_default_dtype()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(11)
            self.model = DeltaMLP(self.config)
        self.optimizer = make_optimizer(self.model, self.config)
        self.x = torch.from_numpy(self.features.copy())
        self.y = torch.from_numpy(self.targets.copy())
        self.sampler = WindowSampler(9, 4, torch.Generator().manual_seed(1011),
                                     max_updates=6, horizon=5)
        # Two inherited updates, followed by two actual branch updates.
        for _ in range(2):
            self.update(self.model, self.optimizer, torch.arange(4))
        for _ in range(2):
            self.update(self.model, self.optimizer, self.sampler.next_batch())

    def tearDown(self):
        torch.set_rng_state(self.rng)
        torch.set_default_dtype(self.dtype)

    def update(self, model, optimizer, rows):
        return training_update(model, optimizer, self.normalizer, self.x[rows], self.y[rows],
                               self.config["training"]["gradient_clip_norm"])

    def saved(self, directory, name="fork", **changes):
        path = Path(directory) / name
        options = {"seed": 11, "branch": "S1", "additional_updates": self.sampler.updates,
                   "sampler": self.sampler, "run_manifest_sha256": "d" * 64}
        options.update(changes)
        metadata = save_fork_checkpoint(path, self.model, self.optimizer, self.normalizer,
                                        self.config, self.provenance, self.budget, **options)
        return path, metadata

    def loaded(self, path, **changes):
        options = {"normalizer": self.normalizer, "config": self.config,
                   "training_provenance": self.provenance, "fork_budget": self.budget,
                   "seed": 11, "branch": "S1", "resume": True}
        options.update(changes)
        return load_fork_checkpoint(path, **options)

    def test_mid_epoch_resume_next_partial_batch_and_adam_update_are_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            path, metadata = self.saved(directory)
            original_files = {p.name: p.read_bytes() for p in path.iterdir()}
            a, b = self.loaded(path), self.loaded(path)
            self.assertEqual(metadata["additional_updates"], 2)
            self.assertEqual(metadata["total_updates"], 4)
            self.assertEqual(a.sampler.cursor, 8)
            tree_equal(self, self.sampler.state_dict(), a.sampler.state_dict())
            tree_equal(self, self.model.state_dict(), a.model.state_dict())
            tree_equal(self, self.optimizer.state_dict(), a.optimizer.state_dict())
            self.assertEqual(len(a.optimizer.state_dict()["state"]), 6)
            rows = [sampler.next_batch() for sampler in (self.sampler, a.sampler, b.sampler)]
            self.assertEqual(len(rows[0]), 1)
            for item in rows[1:]:
                tree_equal(self, rows[0], item)
            diagnostics = [self.update(model, optimizer, batch)
                           for model, optimizer, batch in
                           ((self.model, self.optimizer, rows[0]), (a.model, a.optimizer, rows[1]),
                            (b.model, b.optimizer, rows[2]))]
            self.assertEqual(diagnostics[0], diagnostics[1])
            self.assertEqual(diagnostics[0], diagnostics[2])
            for bundle in (a, b):
                tree_equal(self, self.model.state_dict(), bundle.model.state_dict())
                tree_equal(self, self.optimizer.state_dict(), bundle.optimizer.state_dict())
                self.assertTrue(all(float(entry["step"]) == 5
                                    for entry in bundle.optimizer.state.values()))
            self.assertEqual(original_files, {p.name: p.read_bytes() for p in path.iterdir()})

    def test_final_checkpoint_keeps_budget_and_probe_copy_has_exact_next_update(self):
        while self.sampler.updates < 6:
            self.update(self.model, self.optimizer, self.sampler.next_batch())
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.saved(directory)
            a = self.loaded(path)
            before = a.sampler.state_dict()
            for sampler in (self.sampler, a.sampler):
                with self.assertRaises(StopIteration):
                    sampler.next_batch()
            rows = [sampler.clone_for_resume_probe().next_batch()
                    for sampler in (self.sampler, a.sampler)]
            tree_equal(self, rows[0], rows[1])
            self.assertEqual(self.update(self.model, self.optimizer, rows[0]),
                             self.update(a.model, a.optimizer, rows[1]))
            tree_equal(self, self.model.state_dict(), a.model.state_dict())
            tree_equal(self, self.optimizer.state_dict(), a.optimizer.state_dict())
            tree_equal(self, before, a.sampler.state_dict())

    def test_independent_reloads_do_not_alias_model_optimizer_or_sampler(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.saved(directory)
            a, b = self.loaded(path), self.loaded(path)
            before_b = (deepcopy(b.model.state_dict()), deepcopy(b.optimizer.state_dict()),
                        b.sampler.state_dict())
            self.update(a.model, a.optimizer, a.sampler.next_batch())
            for old, new in zip(before_b, (b.model.state_dict(), b.optimizer.state_dict(), b.sampler.state_dict())):
                tree_equal(self, old, new)
            for left, right in zip(a.model.parameters(), b.model.parameters()):
                self.assertNotEqual(left.data_ptr(), right.data_ptr())

    def test_inference_roundtrip_keeps_physical_predictions_rng_and_default_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            path, metadata = self.saved(directory)
            torch.rand(9)
            before = torch.get_rng_state().clone()
            torch.set_default_dtype(torch.float64)
            bundle = self.loaded(path, resume=False, expected_checkpoint_hash=metadata["checkpoint_hash"],
                                 expected_run_manifest_sha256="d" * 64)
            self.assertIsNone(bundle.optimizer)
            self.assertIsNone(bundle.sampler)
            self.assertIsNone(bundle.torch_rng_state)
            self.assertFalse(bundle.model.training)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(torch.get_default_dtype(), torch.float64)
            state, action = torch.from_numpy(self.state[:3]), torch.from_numpy(self.action[:3])
            tree_equal(self, predict_delta(self.model, self.normalizer, state, action),
                       predict_delta(bundle.model, self.normalizer, state, action))

    def test_resume_returns_saved_rng_without_changing_callers_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            saved_rng = torch.get_rng_state().clone()
            path, _ = self.saved(directory)
            torch.rand(7)
            current = torch.get_rng_state().clone()
            a = self.loaded(path)
            tree_equal(self, current, torch.get_rng_state())
            tree_equal(self, saved_rng, a.torch_rng_state)
            with torch.random.fork_rng(devices=[]):
                torch.set_rng_state(saved_rng)
                expected = torch.rand(11)
                torch.set_rng_state(a.torch_rng_state)
                tree_equal(self, expected, torch.rand(11))

    def test_expected_manifest_checkpoint_branch_seed_and_budget_mismatches_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.saved(directory)
            changed_budget = deepcopy(self.budget)
            changed_budget["common_checkpoints"][0]["checkpoint_hash"] = "e" * 64
            changed_provenance = deepcopy(self.provenance)
            changed_provenance["learning_manifest_sha256"] = "f" * 64
            for options in ({"expected_checkpoint_hash": "e" * 64},
                            {"expected_run_manifest_sha256": "e" * 64},
                            {"branch": "SK-lambda0.1"}, {"seed": 23},
                            {"fork_budget": changed_budget},
                            {"training_provenance": changed_provenance}):
                with self.subTest(options=options), self.assertRaises(CheckpointError):
                    self.loaded(path, **options)

    def test_p3_provenance_cannot_masquerade_as_p4(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.saved(directory)
            provenance = deepcopy(self.provenance)
            provenance["task_scope"] = "P3-04-main-common"
            with self.assertRaisesRegex(CheckpointError, "P4 learning"):
                self.loaded(path, training_provenance=provenance)

    def test_typed_metadata_counters_and_branch_weight_rejected_after_outer_rehash(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, (key, value) in enumerate((
                    ("additional_updates", 2.0), ("total_updates", True),
                    ("seed", 11.0), ("rollout_weight", 0),
                    ("additional_updates", 7), ("total_updates", 5),
                    ("run_manifest_sha256", "not-a-hash"))):
                path, metadata = self.saved(directory, name=f"changed-{index}")
                metadata[key] = value
                rebind(path, metadata)
                with self.subTest(key=key, value=value), self.assertRaises(CheckpointError):
                    self.loaded(path)

    def test_numeric_budget_types_are_not_equal_by_python_coercion(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.saved(directory)
            for key, value in (("batch_windows", 4.0), ("K", True),
                               ("drop_last", 0), ("lambda_candidates", [1, .1])):
                changed = deepcopy(self.budget)
                changed[key] = value
                with self.subTest(key=key), self.assertRaises(CheckpointError):
                    self.loaded(path, fork_budget=changed)

    def test_save_rejects_optimizer_step_or_sampler_counter_before_creating_files(self):
        with tempfile.TemporaryDirectory() as directory:
            for options in ({"additional_updates": 1}, {"additional_updates": True},
                            {"seed": 11.0}, {"run_manifest_sha256": "bad"}):
                with self.subTest(options=options), self.assertRaises(CheckpointError):
                    self.saved(directory, **options)
                self.assertFalse((Path(directory) / "fork").exists())
            self.optimizer.state[next(iter(self.model.parameters()))]["step"] += 1
            with self.assertRaisesRegex(CheckpointError, "Adam step"):
                self.saved(directory)
            self.assertFalse((Path(directory) / "fork").exists())

    def test_save_rejects_other_models_optimizer_even_with_identical_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            other = DeltaMLP(self.config)
            with self.assertRaisesRegex(CheckpointError, "another model"):
                save_fork_checkpoint(Path(directory) / "other", other, self.optimizer,
                                     self.normalizer, self.config, self.provenance, self.budget,
                                     seed=11, branch="S1", additional_updates=2,
                                     sampler=self.sampler, run_manifest_sha256="d" * 64)

    def test_model_tensor_shape_dtype_and_nonfinite_rejected_after_file_rehash(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, change in enumerate(("shape", "dtype", "nonfinite", "extra")):
                path, _ = self.saved(directory, name=f"weights-{index}")
                weights = torch.load(path / "weights.pt", weights_only=True)
                key = next(iter(weights))
                if change == "shape":
                    weights[key] = weights[key][:1]
                elif change == "dtype":
                    weights[key] = weights[key].double()
                elif change == "nonfinite":
                    weights[key].flatten()[0] = float("nan")
                else:
                    weights["unregistered"] = torch.ones(1)
                torch.save(weights, path / "weights.pt")
                rebind(path)
                with self.subTest(change=change), self.assertRaises(CheckpointError):
                    self.loaded(path)

    def test_adam_moments_step_and_missing_parameter_rejected_after_file_rehash(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, change in enumerate(("step", "step_type", "moment", "negative", "missing", "lr")):
                path, _ = self.saved(directory, name=f"adam-{index}")
                state = torch.load(path / "training_state.pt", weights_only=True)
                adam = state["optimizer"]
                entry = adam["state"][0]
                if change == "step":
                    entry["step"] += 1
                elif change == "step_type":
                    entry["step"] = entry["step"].double()
                elif change == "moment":
                    entry["exp_avg"] = entry["exp_avg"].double()
                elif change == "negative":
                    entry["exp_avg_sq"].flatten()[0] = -1.
                elif change == "missing":
                    del adam["state"][5]
                else:
                    adam["param_groups"][0]["lr"] = .002
                torch.save(state, path / "training_state.pt")
                rebind(path)
                with self.subTest(change=change), self.assertRaises(CheckpointError):
                    self.loaded(path)

    def test_sampler_and_global_rng_tamper_rejected_including_inference_load(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, change in enumerate(("permutation", "cursor", "updates", "rng", "budget", "global_rng")):
                path, _ = self.saved(directory, name=f"sampler-{index}")
                state = torch.load(path / "training_state.pt", weights_only=True)
                sampler = state["sampler"]
                if change == "permutation":
                    sampler["permutation"][0] = sampler["permutation"][1]
                elif change == "cursor":
                    sampler["cursor"] = 7
                elif change == "updates":
                    sampler["updates"] = 2.0
                elif change == "rng":
                    sampler["shuffle_rng_state"] = torch.zeros(3, dtype=torch.uint8)
                elif change == "budget":
                    sampler["max_updates"] = 7
                    sampler["state_hash"] = sampler_state_hash(sampler)
                else:
                    state["torch_rng_state"] = torch.zeros(3, dtype=torch.uint8)
                torch.save(state, path / "training_state.pt")
                rebind(path)
                with self.subTest(change=change), self.assertRaises(CheckpointError):
                    self.loaded(path, resume=False)

    def test_file_corruption_and_extra_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.saved(directory)
            with (path / "weights.pt").open("ab") as file:
                file.write(b"corrupt")
            with self.assertRaisesRegex(CheckpointError, "file hash"):
                self.loaded(path)
            path, metadata = self.saved(directory, name="extra-metadata")
            metadata["selection_score"] = 0.0
            rebind(path, metadata)
            with self.assertRaisesRegex(CheckpointError, "metadata fields"):
                self.loaded(path)

    def test_unallowlisted_pickle_never_executes_in_either_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, filename in enumerate(("weights.pt", "training_state.pt")):
                path, _ = self.saved(directory, name=f"unsafe-{index}")
                marker = Path(directory) / f"executed-{index}.txt"
                torch.save(UnsafePickle(marker), path / filename)
                rebind(path)
                with self.subTest(filename=filename), self.assertRaises(CheckpointError):
                    self.loaded(path, resume=False)
                self.assertFalse(marker.exists())

    def test_config_schema_normalizer_and_origin_metadata_mismatches_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, key in enumerate(("config", "learning_schema", "normalizer_hash", "common_origin")):
                path, metadata = self.saved(directory, name=f"provenance-{index}")
                if key == "config":
                    metadata[key]["training"]["learning_rate"] = .002
                elif key == "learning_schema":
                    metadata[key]["action_dt_s"] = .2
                elif key == "normalizer_hash":
                    metadata[key] = "e" * 64
                else:
                    metadata[key]["initial_updates"] = 3
                rebind(path, metadata)
                with self.subTest(key=key), self.assertRaises(CheckpointError):
                    self.loaded(path)

    def test_saved_path_is_immutable_and_existing_checkpoint_not_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.saved(directory)
            files = {p.name: p.read_bytes() for p in path.iterdir()}
            with self.assertRaises(CheckpointError):
                self.saved(directory)
            self.assertEqual(files, {p.name: p.read_bytes() for p in path.iterdir()})


if __name__ == "__main__":
    unittest.main()
