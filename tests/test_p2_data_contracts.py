"""P2-01 data contracts exercised with real short MuJoCo recordings.

The workspace-edge fixture deliberately interrupts an action interval to
exercise storage/exclusion; it does not establish a field failure rate.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from push_align import InitialSpec, PlanarPushEnv
from push_align.data import (
    DataContractError, Episode, EpisodeRecorder, load_dataset, make_contract,
    sequence_windows, snapshot_from_episode, split_families, validate_episode,
    write_dataset,
)


def clone(episode):
    return Episode({k: v.copy() for k, v in episode.arrays.items()},
                   deepcopy(episode.metadata))


def recorder(env, *, episode_id="normal", episode_index=0, family_id="family0",
             split="train", base=None, contract=None):
    return EpisodeRecorder(
        env, episode_id=episode_id, episode_index=episode_index,
        family_id=family_id,
        family_definition={"base_initial_spec": base or {"object_x": 0., "object_y": 0., "object_yaw": 0.},
                           "variant_rule": "exact base, prescribed action sequence"},
        split=split, collection_policy={"name": "contract-fixture", "version": 1},
        seed=20261007, contract=contract,
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


class P2DataContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = PlanarPushEnv()
        cls.contract = make_contract(env)
        r = recorder(env, contract=cls.contract)
        for action in ([.03, 0], [.03, 0], [.03, 0], [0, 0]):
            result = r.step(action)
            if not result.transition_valid:
                raise AssertionError("Normal physics fixture unexpectedly invalid")
        cls.normal = r.finish("timeout")

        env = PlanarPushEnv()
        env.reset(InitialSpec(pusher_x=.191, pusher_y=.1))
        r = recorder(env, episode_id="partial", episode_index=1, family_id="family1",
                     base={"pusher_x": .191, "pusher_y": .1}, contract=cls.contract)
        r.step([0, 0]); r.step([0, 0])
        cls.partial_result = r.step([.06, 0])
        cls.partial = r.finish("collection_complete")

    def dataset(self, root, episodes=None, **kwargs):
        path = Path(root)/"fixture"
        manifest = write_dataset(path, episodes or [self.normal, self.partial], **kwargs)
        return path, manifest

    def mutate_manifest(self, path, change):
        file = path/"manifest.json"
        value = json.loads(file.read_text(encoding="utf-8"))
        change(value)
        write_json(file, value)

    def replace_npz_with_matching_hashes(self, path, arrays):
        """Pass the byte-integrity checks so semantic/unsafe-NPZ guards run."""
        manifest_path = path/"manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = manifest["episodes"][0]
        npz, meta_path = path/entry["npz"], path/entry["json"]
        np.savez_compressed(npz, **arrays)
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata["npz_sha256"] = sha(npz)
        write_json(meta_path, metadata)
        entry["npz_sha256"], entry["json_sha256"] = sha(npz), sha(meta_path)
        write_json(manifest_path, manifest)

    def test_roundtrip_retains_every_raw_array_and_snapshot(self):
        with tempfile.TemporaryDirectory() as root:
            path, original_manifest = self.dataset(root)
            loaded, manifest = load_dataset(path, self.contract)
            self.assertEqual(manifest, original_manifest)
            for original, restored in zip((self.normal, self.partial), loaded):
                self.assertEqual(set(original.arrays), set(restored.arrays))
                for key in original.arrays:
                    with self.subTest(episode=original.metadata["episode_id"], array=key):
                        self.assertEqual(original.arrays[key].dtype, restored.arrays[key].dtype)
                        np.testing.assert_array_equal(original.arrays[key], restored.arrays[key])
                for i in range(original.transition_count+1):
                    s = snapshot_from_episode(restored, i)
                    np.testing.assert_array_equal(s.integration_state, original.arrays["raw_integration_state"][i])
                    self.assertEqual(s.rng_state, original.metadata["snapshot_rng_states"][i])
                    self.assertEqual(s.needs_reset, bool(original.arrays["raw_needs_reset"][i]))
            self.assertEqual(manifest["excluded_transition_count"], 1)
            self.assertEqual(manifest["learning_transition_count"], 6)

    def test_restored_snapshot_replays_a_stored_complete_interval(self):
        env = PlanarPushEnv()
        episode = self.normal
        index = 1
        snapshot = snapshot_from_episode(episode, index)
        env.restore(snapshot)
        np.testing.assert_array_equal(env.observe().state, episode.arrays["state"][index])
        expected_random = env.rng.random()
        env.restore(snapshot)
        self.assertEqual(env.rng.random(), expected_random)
        env.restore(snapshot)
        result = env.step(episode.arrays["action_requested"][index])
        np.testing.assert_array_equal(result.observation.state, episode.arrays["state"][index+1])
        np.testing.assert_array_equal(env.snapshot().integration_state,
                                      episode.arrays["raw_integration_state"][index+1])

    def test_finite_raw_snapshot_time_or_qpos_cannot_disagree_with_boundary(self):
        for field, message in (("time", "time mismatch"), ("qpos", "observation mismatch")):
            with self.subTest(field=field):
                e = clone(self.normal)
                env = PlanarPushEnv(); env.restore(snapshot_from_episode(e, 1))
                if field == "time": env.data.time += .001
                else: env.data.qpos[env._object_qposadr] += .001
                mujoco.mj_getState(env.model, env.data, e.arrays["raw_integration_state"][1],
                                   self.contract["snapshot_state_spec"])
                self.assertTrue(np.isfinite(e.arrays["raw_integration_state"][1]).all())
                with self.assertRaisesRegex(DataContractError, message):
                    validate_episode(e, self.contract)

    def test_snapshot_warning_evidence_and_pcg64_structure_are_validated(self):
        for field, message in (("warnings", "warnings mismatch"), ("rng", "RNG state")):
            with self.subTest(field=field):
                e = clone(self.normal)
                if field == "warnings": e.arrays["raw_warning_counts"][1, 0] += 1
                else: e.metadata["snapshot_rng_states"][1]["bit_generator"] = "MT19937"
                with self.assertRaisesRegex(DataContractError, message):
                    validate_episode(e, self.contract)

    def test_timeout_is_a_valid_terminal_not_a_discarded_sample(self):
        review = validate_episode(self.normal, self.contract)
        self.assertEqual(review["terminal_reason"], "timeout")
        self.assertEqual(review["learning_transitions"], 4)
        windows = sequence_windows(self.normal, 2, self.contract)
        self.assertEqual([w.start for w in windows], [0, 1, 2])
        np.testing.assert_array_equal(windows[-1].terminal, [False, True])
        self.assertEqual(self.normal.metadata["task_outcome"], "not_evaluated")

    def test_partial_raw_interval_is_retained_but_cannot_enter_windows(self):
        e = self.partial
        self.assertFalse(self.partial_result.transition_valid)
        self.assertLess(self.partial_result.actual_dt, self.contract["action_dt_s"])
        self.assertGreater(self.partial_result.actual_dt, 0)
        self.assertEqual(e.metadata["terminal_reason"], "physical_invalid")
        np.testing.assert_array_equal(e.arrays["transition_valid"], [True, True, False])
        self.assertEqual(validate_episode(e, self.contract)["excluded_transitions"], 1)
        self.assertEqual([w.start for w in sequence_windows(e, 1, self.contract)], [0, 1])
        self.assertEqual([w.start for w in sequence_windows(e, 2, self.contract)], [0])
        self.assertEqual(sequence_windows(e, 3, self.contract), [])
        env = PlanarPushEnv(); env.restore(snapshot_from_episode(e, e.transition_count))
        with self.assertRaises(RuntimeError):
            env.step([0, 0])

    def test_masks_are_recomputed_from_evidence(self):
        for original, field, index, value in [
            (self.partial, "transition_valid", -1, True),
            (self.partial, "step_valid", -1, True),
            (self.normal, "transition_valid", 0, False),
            (self.normal, "geometry_valid", 0, False),
            (self.partial, "raw_needs_reset", -1, False),
        ]:
            with self.subTest(field=field, index=index):
                e = clone(original); e.arrays[field][index] = value
                with self.assertRaises(DataContractError):
                    validate_episode(e, self.contract)

    def test_invalid_interval_cannot_be_joined_to_a_following_episode(self):
        """Retain original evidence rather than merely flipping an end flag."""
        first, second = self.partial, self.normal
        a, b = first.arrays, second.arrays
        boundary_arrays = {"state", "sim_time", "observation_valid", "raw_integration_state",
                           "raw_reference_height", "raw_warning_counts", "raw_warning_lastinfo",
                           "raw_needs_reset"}
        joined = {}
        time_shift = a["sim_time"][-1]-b["sim_time"][0]
        for key in a:
            if key == "substep_offsets":
                joined[key] = np.concatenate([a[key], b[key][1:]+a[key][-1]])
            elif key in boundary_arrays:
                tail = b[key][1:]+time_shift if key == "sim_time" else b[key][1:]
                joined[key] = np.concatenate([a[key], tail])
            else:
                tail = b[key]+time_shift if key == "substep_times" else b[key]
                joined[key] = np.concatenate([a[key], tail])
        joined["terminal"][:] = False; joined["terminal"][-1] = True
        meta = deepcopy(first.metadata)
        diagnostics = deepcopy(second.metadata["diagnostics"])
        for d in diagnostics:
            for sample in d["samples"]: sample["sim_time"] += time_shift
        meta["diagnostics"] += diagnostics
        meta["geometry_checks"] += deepcopy(second.metadata["geometry_checks"])
        meta["snapshot_rng_states"] += deepcopy(second.metadata["snapshot_rng_states"][1:])
        with self.assertRaisesRegex(DataContractError, "invalid transition must end"):
            sequence_windows(Episode(joined, meta), 2, self.contract)

    def test_actual_norm_clipping_and_measured_velocity_remain_distinct(self):
        env = PlanarPushEnv(); env.reset(InitialSpec(pusher_x=-.1, pusher_y=.1))
        r = recorder(env, contract=self.contract)
        result = r.step([.6, .8])
        e = r.finish()
        np.testing.assert_array_equal(e.arrays["action_requested"][0], [.6, .8])
        np.testing.assert_allclose(e.arrays["action_applied"][0], [.036, .048], rtol=0, atol=1e-15)
        self.assertAlmostEqual(np.linalg.norm(e.arrays["action_applied"][0]), .06)
        self.assertLess(np.linalg.norm(result.substep_states[0, 8:10]), .06)
        window = sequence_windows(e, 1, self.contract)[0]
        np.testing.assert_array_equal(window.actions[0], e.arrays["action_applied"][0])
        self.assertEqual(validate_episode(e, self.contract)["learning_transitions"], 1)

    def test_recorder_rejects_identical_reset_and_restore(self):
        for operation in ("reset", "restore"):
            with self.subTest(operation=operation):
                env = PlanarPushEnv(); r = recorder(env, contract=self.contract)
                r.step([0, 0]); state = env.snapshot()
                env.reset() if operation == "reset" else env.restore(state)
                with self.assertRaisesRegex(DataContractError, "replaced"):
                    r.step([0, 0])
                with self.assertRaisesRegex(DataContractError, "replaced"):
                    r.finish()

    def test_recorder_rejects_external_step_and_rng_change_before_step_or_finish(self):
        for operation in ("step", "rng"):
            for next_call in ("step", "finish"):
                with self.subTest(operation=operation, next_call=next_call):
                    env = PlanarPushEnv(); r = recorder(env, contract=self.contract)
                    r.step([0, 0])
                    env.step([.01, 0]) if operation == "step" else env.rng.random()
                    with self.assertRaisesRegex(DataContractError, "Unrecorded"):
                        r.step([0, 0]) if next_call == "step" else r.finish()

    def test_invalid_and_closed_recorders_cannot_keep_appending(self):
        env = PlanarPushEnv(); env.reset(InitialSpec(pusher_x=.191, pusher_y=.1))
        r = recorder(env, contract=self.contract)
        r.step([.06, 0])
        with self.assertRaises(DataContractError): r.step([0, 0])
        r.finish()
        with self.assertRaises(DataContractError): r.step([0, 0])
        with self.assertRaises(DataContractError): r.finish()
        with self.assertRaises(DataContractError): recorder(PlanarPushEnv()).finish()

    def test_same_family_cannot_leak_to_another_split(self):
        e = clone(self.normal)
        e.metadata.update(episode_id="variant", episode_index=2, split="validation")
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(DataContractError, "leaks"):
                self.dataset(root, [self.normal, e])
            self.assertFalse((Path(root)/"fixture").exists())

    def test_same_declared_initial_base_cannot_be_relabeled_as_new_family(self):
        e = clone(self.normal)
        e.metadata.update(episode_id="relabeled", episode_index=2,
                          family_id="other_family", split="validation")
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(DataContractError, "different families"):
                self.dataset(root, [self.normal, e])

    def test_actual_initial_condition_cannot_be_hidden_by_a_new_declared_base(self):
        e = clone(self.normal)
        e.metadata.update(episode_id="hidden_copy", episode_index=2,
                          family_id="hidden_family", split="validation")
        e.metadata["family_definition"]["base_initial_spec"] = {"object_x": .012, "object_y": 0.}
        # Different declared lineage does not change this identical measured reset.
        self.assertEqual(e.metadata["initial_record"], self.normal.metadata["initial_record"])
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(DataContractError, "Repeated actual initial"):
                self.dataset(root, [self.normal, e])

    def test_family_definition_and_episode_identity_are_consistent(self):
        for change, message in [
            ({"episode_id": "normal", "episode_index": 8}, "Duplicate"),
            ({"episode_id": "other", "episode_index": 0}, "Duplicate"),
            ({"episode_id": "other", "episode_index": 8,
              "family_definition": {"base_initial_spec": {"object_x": .001}, "variant_rule": "changed"}}, "definition mismatch"),
        ]:
            with self.subTest(change=change):
                e = clone(self.normal); e.metadata.update(change)
                with tempfile.TemporaryDirectory() as root:
                    with self.assertRaisesRegex(DataContractError, message):
                        self.dataset(root, [self.normal, e])

    def test_seeded_family_assignment_is_order_independent_and_total(self):
        ids = [f"family{i}" for i in range(12)]
        counts = {"train": 6, "validation": 3, "smoke-test": 2, "test": 1}
        a = split_families(ids, counts, 17)
        self.assertEqual(a, split_families(reversed(ids), counts, 17))
        self.assertEqual(set(a), set(ids))
        self.assertEqual({s: list(a.values()).count(s) for s in counts}, counts)
        for ids_bad, counts_bad in [(ids+ids[:1], counts), (ids, {"train": 11}), (ids, {"bogus": 12})]:
            with self.assertRaises(DataContractError): split_families(ids_bad, counts_bad, 17)

    def test_declared_split_seed_and_algorithm_must_reproduce_actual_assignments(self):
        second = clone(self.partial); second.metadata["split"] = "validation"
        assignments = {"family0": "train", "family1": "validation"}
        counts = {"train": 1, "validation": 1}
        matching = next(seed for seed in range(100)
                        if split_families(assignments, counts, seed) == assignments)
        wrong = next(seed for seed in range(100)
                     if split_families(assignments, counts, seed) != assignments)
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(DataContractError, "seed does not reproduce"):
                self.dataset(root, [self.normal, second], split_seed=wrong)
            self.assertFalse((Path(root)/"fixture").exists())
        for key, value in (("split_seed", wrong), ("split_algorithm", "explicit_family_assignments")):
            with self.subTest(field=key), tempfile.TemporaryDirectory() as root:
                path, manifest = self.dataset(root, [self.normal, second], split_seed=matching)
                self.assertEqual(load_dataset(path, self.contract)[1], manifest)
                self.mutate_manifest(path, lambda m: m.__setitem__(key, value))
                with self.assertRaisesRegex(DataContractError, "seed does not reproduce|provenance mismatch"):
                    load_dataset(path, self.contract)

    def test_smoke_is_development_and_final_test_is_separate(self):
        smoke, test = clone(self.normal), clone(self.partial)
        smoke.metadata["split"] = "smoke-test"; test.metadata["split"] = "test"
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.dataset(root, [smoke])
            loaded, _ = load_dataset(path, self.contract)
            self.assertEqual(loaded[0].metadata["split"], "smoke-test")
        for purpose in ("contract-check", "pilot", "main"):
            with tempfile.TemporaryDirectory() as root:
                with self.assertRaises(DataContractError): self.dataset(root, [test], purpose=purpose)
        with tempfile.TemporaryDirectory() as root:
            path, manifest = self.dataset(root, [test], purpose="final-evaluation")
            self.assertEqual(load_dataset(path, self.contract)[1], manifest)
        for development in ([smoke], [smoke, test]):
            with tempfile.TemporaryDirectory() as root:
                with self.assertRaises(DataContractError):
                    self.dataset(root, development, purpose="final-evaluation")

    def test_shape_dtype_schema_and_time_tampering_are_rejected(self):
        edits = [
            lambda e: e.arrays.update(state=e.arrays["state"][:, :9]),
            lambda e: e.arrays.update(state=e.arrays["state"].astype(np.float32)),
            lambda e: e.arrays["sim_time"].__setitem__(1, e.arrays["sim_time"][1]+.0005),
            lambda e: e.arrays["substep_times"].__setitem__(5, e.arrays["substep_times"][5]+.0005),
            lambda e: e.arrays["state"].__setitem__((1, 0), .1),
            lambda e: e.arrays["terminal"].__setitem__(0, True),
            lambda e: e.metadata["contract"]["state_fields"].reverse(),
            lambda e: e.metadata["contract"].__setitem__("config_sha256", "0"*64),
            lambda e: e.arrays["action_applied"].__setitem__((0, 0), .04),
        ]
        for i, edit in enumerate(edits):
            with self.subTest(edit=i):
                e = clone(self.normal); edit(e)
                with self.assertRaises(DataContractError): validate_episode(e, self.contract)

    def test_window_data_are_copies_and_length_cannot_cross_an_episode(self):
        e = clone(self.normal)
        windows = sequence_windows(e, 4, self.contract)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].states.shape, (5, 10))
        self.assertEqual(windows[0].actions.shape, (4, 2))
        windows[0].states[0, 0] = 123
        windows[0].actions[0, 0] = 123
        self.assertNotEqual(e.arrays["state"][0, 0], 123)
        self.assertNotEqual(e.arrays["action_applied"][0, 0], 123)
        self.assertEqual(sequence_windows(e, 5, self.contract), [])
        for length in (0, -1, 1.5, True):
            with self.assertRaises(DataContractError): sequence_windows(e, length, self.contract)

    def test_overwrite_and_unlisted_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.dataset(root)
            before = sha(path/"manifest.json")
            with self.assertRaises(FileExistsError): write_dataset(path, [self.normal])
            self.assertEqual(sha(path/"manifest.json"), before)
            (path/"episodes"/"unlisted.txt").write_text("unlisted", encoding="utf-8")
            with self.assertRaisesRegex(DataContractError, "Unlisted"):
                load_dataset(path, self.contract)

    def test_corrupted_artifact_hash_is_detected(self):
        for filename in ("normal.npz", "normal.json"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as root:
                path, _ = self.dataset(root)
                with (path/"episodes"/filename).open("ab") as file: file.write(b"corruption")
                with self.assertRaisesRegex(DataContractError, "hash mismatch"):
                    load_dataset(path, self.contract)

    def test_object_npz_is_rejected_even_when_file_hashes_match(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.dataset(root, [self.normal])
            arrays = {k: v.copy() for k, v in self.normal.arrays.items()}
            arrays["unsafe"] = np.array([{"arbitrary": "object"}], dtype=object)
            self.replace_npz_with_matching_hashes(path, arrays)
            with self.assertRaisesRegex(DataContractError, "Unsafe|Object"):
                load_dataset(path, self.contract)

    def test_nonstandard_json_numbers_and_duplicate_keys_are_rejected(self):
        for suffix in (', "nonfinite": NaN}', ', "purpose": "contract-check"}'):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as root:
                path, _ = self.dataset(root)
                file = path/"manifest.json"
                original = file.read_text(encoding="utf-8").rstrip()
                file.write_text(original[:-1]+suffix, encoding="utf-8")
                with self.assertRaises(DataContractError): load_dataset(path, self.contract)

    def test_manifest_duplicates_count_errors_and_path_escape_are_rejected(self):
        def duplicate(m):
            m["episodes"].append(deepcopy(m["episodes"][0]))
            m["episode_count"] += 1
        changes = [duplicate, lambda m: m.__setitem__("transition_count", 999),
                   lambda m: m["episodes"][0].__setitem__("npz", "../outside.npz"),
                   lambda m: m["family_assignments"].__setitem__("family0", "validation")]
        for i, change in enumerate(changes):
            with self.subTest(change=i), tempfile.TemporaryDirectory() as root:
                path, _ = self.dataset(root); self.mutate_manifest(path, change)
                with self.assertRaises(DataContractError): load_dataset(path, self.contract)

    def test_nonfinite_diagnostic_annotations_use_strict_json_and_numeric_npz(self):
        env = PlanarPushEnv(); r = recorder(env, contract=self.contract)
        r.step([0, 0])
        # An explicit storage sentinel, not an observed nonfinite physical state.
        r.results[0].diagnostics["storage_sentinel"] = [float("nan"), float("inf"), float("-inf")]
        e = r.finish()
        self.assertEqual(e.metadata["diagnostics"][0]["storage_sentinel"], [None, None, None])
        np.testing.assert_array_equal(e.arrays["diagnostic_nonfinite_values"], [np.nan, np.inf, -np.inf])
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.dataset(root, [e])
            text = (path/"episodes"/"normal.json").read_text(encoding="utf-8")
            json.loads(text, parse_constant=lambda value: self.fail(f"Nonstandard JSON {value}"))
            loaded, _ = load_dataset(path, self.contract)
            np.testing.assert_array_equal(loaded[0].arrays["diagnostic_nonfinite_values"], e.arrays["diagnostic_nonfinite_values"])
        e = clone(e); e.metadata["diagnostic_nonfinite_paths"][1] = e.metadata["diagnostic_nonfinite_paths"][0]
        with self.assertRaisesRegex(DataContractError, "Duplicate"):
            validate_episode(e, self.contract)


if __name__ == "__main__":
    unittest.main()
