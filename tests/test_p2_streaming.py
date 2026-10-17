"""Streaming pilot persistence preserves evidence without retaining raw traces.

The workspace-edge recording is a deliberate storage fixture. It is not a
claim about alignment-task failure rates or the prevalence of field failures.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from push_align import InitialSpec, PlanarPushEnv
from push_align.data import (
    DataContractError, DatasetWriter, Episode, EpisodeRecorder, iter_dataset,
    load_dataset, make_contract, sequence_windows, split_families,
)


def clone(episode):
    return Episode({key: value.copy() for key, value in episode.arrays.items()},
                   deepcopy(episode.metadata))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False),
                          encoding="utf-8")


def record(contract, *, episode_id, episode_index, family_id, split="train",
           partial=False):
    """Build short real physics records, including actual reset lineage."""
    env = PlanarPushEnv()
    initial = (InitialSpec(pusher_x=.191, pusher_y=.1) if partial else InitialSpec())
    env.reset(initial)
    recorder = EpisodeRecorder(
        env, episode_id=episode_id, episode_index=episode_index,
        family_id=family_id,
        family_definition={"base_initial_spec": deepcopy(env.initial_record["initial_spec"]),
                           "variant_rule": "exact base, prescribed streaming-fixture actions"},
        split=split, collection_policy={"name": "streaming-contract-fixture", "version": 1},
        seed=20261007, contract=contract,
    )
    for action in ([0., 0.], [0., 0.], [.06, 0.]) if partial else ([.03, 0.], [0., 0.]):
        recorder.step(action)
    return recorder.finish("collection_complete")


class P2StreamingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = make_contract(PlanarPushEnv())
        cls.normal = record(cls.contract, episode_id="normal", episode_index=0,
                            family_id="family0")
        cls.partial = record(cls.contract, episode_id="partial", episode_index=1,
                             family_id="family1", partial=True)
        if not cls.normal.arrays["transition_valid"].all():
            raise AssertionError("Normal recording unexpectedly invalid")
        if cls.partial.arrays["transition_valid"][-1]:
            raise AssertionError("Workspace-edge fixture did not interrupt its last action")

    def stream(self, root, *, episodes=None, split_seed=None, purpose="pilot"):
        path = Path(root)/"stream"
        writer = DatasetWriter(path, self.contract, purpose=purpose, split_seed=split_seed)
        for episode in episodes or (self.normal, self.partial):
            writer.append(episode)
        return path, writer.finish()

    def mutate_manifest(self, path, change):
        file = path/"manifest.json"
        manifest = json.loads(file.read_text(encoding="utf-8"))
        change(manifest)
        write_json(file, manifest)

    def test_two_episode_roundtrip_matches_bulk_loader_and_retains_raw(self):
        with tempfile.TemporaryDirectory() as root:
            path, manifest = self.stream(root)
            streamed = list(iter_dataset(path, self.contract))
            bulk, loaded_manifest = load_dataset(path, self.contract)
            self.assertEqual(manifest, loaded_manifest)
            self.assertEqual(len(streamed), 2)
            for original, restored, loaded in zip((self.normal, self.partial), streamed, bulk):
                self.assertEqual(restored.metadata, loaded.metadata)
                expected_metadata = deepcopy(original.metadata)
                expected_metadata["npz_sha256"] = sha(path/"episodes"/f"{original.metadata['episode_id']}.npz")
                self.assertEqual(restored.metadata, expected_metadata)
                self.assertEqual(set(original.arrays), set(restored.arrays))
                for name, values in original.arrays.items():
                    with self.subTest(episode=original.metadata["episode_id"], array=name):
                        self.assertEqual(values.dtype, restored.arrays[name].dtype)
                        np.testing.assert_array_equal(values, restored.arrays[name])
                        np.testing.assert_array_equal(restored.arrays[name], loaded.arrays[name])
            for entry in manifest["episodes"]:
                self.assertEqual(sha(path/entry["npz"]), entry["npz_sha256"])
                self.assertEqual(sha(path/entry["json"]), entry["json_sha256"])
                with np.load(path/entry["npz"], allow_pickle=False) as arrays:
                    self.assertTrue(all(arrays[name].dtype != object for name in arrays.files))

    def test_partial_interval_is_stored_but_excluded_from_learning(self):
        with tempfile.TemporaryDirectory() as root:
            path, manifest = self.stream(root)
            partial = list(iter_dataset(path, self.contract))[-1]
            np.testing.assert_array_equal(partial.arrays["transition_valid"], [True, True, False])
            np.testing.assert_array_equal(partial.arrays["actual_dt"], self.partial.arrays["actual_dt"])
            self.assertGreater(partial.arrays["actual_dt"][-1], 0.)
            self.assertLess(partial.arrays["actual_dt"][-1], self.contract["action_dt_s"])
            self.assertEqual(partial.metadata["terminal_reason"], "physical_invalid")
            self.assertEqual(manifest["transition_count"], 5)
            self.assertEqual(manifest["learning_transition_count"], 4)
            self.assertEqual(manifest["excluded_transition_count"], 1)
            self.assertEqual([window.start for window in sequence_windows(partial, 1, self.contract)], [0, 1])
            self.assertEqual(sequence_windows(partial, 3, self.contract), [])
            self.assertGreater(len(partial.arrays["raw_integration_state"]), 0)

    def test_writer_retains_identity_not_episode_arrays_or_diagnostics(self):
        with tempfile.TemporaryDirectory() as root:
            writer = DatasetWriter(Path(root)/"stream", self.contract)
            writer.append(self.normal)
            self.assertEqual(writer.identities[0].arrays, {})
            self.assertEqual(set(writer.identities[0].metadata),
                             {"episode_id", "episode_index", "family_id", "family_definition",
                              "initial_record", "split"})
            self.assertNotIn("diagnostics", writer.identities[0].metadata)
            self.assertNotIn("snapshot_rng_states", writer.identities[0].metadata)
            self.assertFalse((writer.path/"manifest.json").exists())
            self.assertTrue((writer.folder/"normal.npz").exists())

    def test_incomplete_and_empty_writers_have_no_completed_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            writer = DatasetWriter(Path(root)/"stream", self.contract)
            with self.assertRaisesRegex(DataContractError, "Nonempty"):
                writer.finish()
            self.assertFalse((writer.path/"manifest.json").exists())
            writer.append(self.normal)
            self.assertFalse((writer.path/"manifest.json").exists())
            self.assertEqual(sorted(item.name for item in writer.folder.iterdir()),
                             ["normal.json", "normal.npz"])
            # Reading unfinished evidence must not treat it as a completed dataset.
            with self.assertRaises(FileNotFoundError):
                list(iter_dataset(writer.path, self.contract))

    def test_closed_writer_rejects_append_and_second_finish(self):
        with tempfile.TemporaryDirectory() as root:
            writer = DatasetWriter(Path(root)/"stream", self.contract)
            writer.append(self.normal); writer.finish()
            before = sha(writer.path/"manifest.json")
            with self.assertRaisesRegex(DataContractError, "closed"):
                writer.append(self.partial)
            with self.assertRaises(DataContractError):
                writer.finish()
            self.assertEqual(sha(writer.path/"manifest.json"), before)
            self.assertEqual(len(list(writer.folder.iterdir())), 2)

    def test_writer_cannot_overwrite_completed_or_incomplete_evidence(self):
        for finish in (False, True):
            with self.subTest(completed=finish), tempfile.TemporaryDirectory() as root:
                path = Path(root)/"stream"
                writer = DatasetWriter(path, self.contract); writer.append(self.normal)
                if finish:
                    writer.finish()
                before = {item.name: sha(item) for item in (path/"episodes").iterdir()}
                with self.assertRaises(FileExistsError):
                    DatasetWriter(path, self.contract)
                self.assertEqual({item.name: sha(item) for item in (path/"episodes").iterdir()}, before)
                self.assertEqual((path/"manifest.json").exists(), finish)

    def test_duplicate_identity_and_family_leak_are_rejected_before_writing(self):
        changes = [
            ({"episode_index": 2}, "Duplicate"),
            ({"episode_id": "variant"}, "Duplicate"),
            ({"episode_id": "variant", "episode_index": 2, "split": "validation"}, "leaks"),
        ]
        for change, message in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as root:
                writer = DatasetWriter(Path(root)/"stream", self.contract); writer.append(self.normal)
                duplicate = clone(self.normal); duplicate.metadata.update(change)
                with self.assertRaisesRegex(DataContractError, message):
                    writer.append(duplicate)
                self.assertEqual(len(writer.entries), 1)
                self.assertEqual(sorted(item.name for item in writer.folder.iterdir()),
                                 ["normal.json", "normal.npz"])
                self.assertFalse((writer.path/"manifest.json").exists())

    def test_family_relabeling_and_definition_changes_are_rejected(self):
        for relabel in (False, True):
            with self.subTest(relabel=relabel), tempfile.TemporaryDirectory() as root:
                writer = DatasetWriter(Path(root)/"stream", self.contract); writer.append(self.normal)
                variant = clone(self.normal)
                variant.metadata.update(episode_id="variant", episode_index=2)
                variant.metadata["family_definition"]["base_initial_spec"]["object_x"] = .012
                if relabel:
                    variant.metadata["family_id"] = "relabeled"
                with self.assertRaisesRegex(DataContractError, "definition mismatch|Repeated actual initial"):
                    writer.append(variant)
                self.assertFalse((writer.folder/"variant.npz").exists())

    def test_invalid_episode_cannot_create_a_written_entry(self):
        with tempfile.TemporaryDirectory() as root:
            writer = DatasetWriter(Path(root)/"stream", self.contract)
            invalid = clone(self.partial); invalid.arrays["transition_valid"][-1] = True
            with self.assertRaises(DataContractError):
                writer.append(invalid)
            self.assertEqual(writer.entries, [])
            self.assertEqual(list(writer.folder.iterdir()), [])
            self.assertFalse((writer.path/"manifest.json").exists())

    def test_development_and_final_test_are_kept_separate(self):
        test = clone(self.normal); test.metadata["split"] = "test"
        for purpose, episode in (("pilot", test), ("main", test),
                                 ("final-evaluation", self.normal)):
            with self.subTest(purpose=purpose), tempfile.TemporaryDirectory() as root:
                writer = DatasetWriter(Path(root)/"stream", self.contract, purpose=purpose)
                with self.assertRaisesRegex(DataContractError, "Final test"):
                    writer.append(episode)
                self.assertEqual(list(writer.folder.iterdir()), [])
                self.assertFalse((writer.path/"manifest.json").exists())
        with tempfile.TemporaryDirectory() as root:
            path, manifest = self.stream(root, episodes=[test], purpose="final-evaluation")
            self.assertEqual(len(list(iter_dataset(path, self.contract))), 1)
            self.assertEqual(manifest["purpose"], "final-evaluation")

    def test_wrong_seed_prevents_finish_and_preserves_unfinished_evidence(self):
        second = clone(self.partial); second.metadata["split"] = "validation"
        assignments = {"family0": "train", "family1": "validation"}
        wrong = next(seed for seed in range(100)
                     if split_families(assignments, {"train": 1, "validation": 1}, seed) != assignments)
        with tempfile.TemporaryDirectory() as root:
            writer = DatasetWriter(Path(root)/"stream", self.contract, split_seed=wrong)
            writer.append(self.normal); writer.append(second)
            with self.assertRaisesRegex(DataContractError, "seed does not reproduce"):
                writer.finish()
            self.assertFalse((writer.path/"manifest.json").exists())
            self.assertEqual(len(list(writer.folder.iterdir())), 4)

    def test_seed_and_algorithm_provenance_are_checked_on_full_consumption(self):
        second = clone(self.partial); second.metadata["split"] = "validation"
        assignments = {"family0": "train", "family1": "validation"}
        seeds = {matches: next(seed for seed in range(100)
                             if (split_families(assignments, {"train": 1, "validation": 1}, seed)
                                 == assignments) == matches) for matches in (True, False)}
        for key, value in (("split_seed", seeds[False]),
                           ("split_algorithm", "explicit_family_assignments")):
            with self.subTest(field=key), tempfile.TemporaryDirectory() as root:
                path, _ = self.stream(root, episodes=[self.normal, second], split_seed=seeds[True])
                self.mutate_manifest(path, lambda manifest: manifest.__setitem__(key, value))
                iterator = iter_dataset(path, self.contract)
                self.assertEqual(next(iterator).metadata["episode_id"], "normal")
                self.assertEqual(next(iterator).metadata["episode_id"], "partial")
                with self.assertRaisesRegex(DataContractError, "seed does not reproduce|provenance mismatch"):
                    next(iterator)

    def test_manifest_counts_are_checked_when_iterator_is_exhausted(self):
        for key in ("episode_count", "transition_count", "learning_transition_count", "excluded_transition_count"):
            with self.subTest(count=key), tempfile.TemporaryDirectory() as root:
                path, _ = self.stream(root)
                self.mutate_manifest(path, lambda manifest: manifest.__setitem__(key, manifest[key]+1))
                iterator = iter_dataset(path, self.contract)
                next(iterator); next(iterator)
                with self.assertRaisesRegex(DataContractError, "counts mismatch"):
                    next(iterator)

    def test_unlisted_artifacts_are_checked_when_iterator_is_exhausted(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.stream(root)
            (path/"episodes"/"unlisted.txt").write_text("unlisted evidence", encoding="utf-8")
            iterator = iter_dataset(path, self.contract)
            next(iterator); next(iterator)
            with self.assertRaisesRegex(DataContractError, "Unlisted"):
                next(iterator)

    def test_corrupted_artifacts_are_rejected_before_yield(self):
        for extension in ("npz", "json"):
            with self.subTest(extension=extension), tempfile.TemporaryDirectory() as root:
                path, _ = self.stream(root)
                with (path/"episodes"/f"normal.{extension}").open("ab") as file:
                    file.write(b"corruption")
                with self.assertRaisesRegex(DataContractError, "hash mismatch"):
                    next(iter_dataset(path, self.contract))

    def test_object_npz_is_rejected_even_if_manifest_and_metadata_hashes_match(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.stream(root)
            manifest_path = path/"manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry = manifest["episodes"][0]
            npz, meta_path = path/entry["npz"], path/entry["json"]
            arrays = {name: values.copy() for name, values in self.normal.arrays.items()}
            arrays["unsafe"] = np.array([{"arbitrary": "object"}], dtype=object)
            np.savez_compressed(npz, **arrays)
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            metadata["npz_sha256"] = sha(npz); write_json(meta_path, metadata)
            entry["npz_sha256"], entry["json_sha256"] = sha(npz), sha(meta_path)
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(DataContractError, "Unsafe|Object"):
                next(iter_dataset(path, self.contract))


if __name__ == "__main__":
    unittest.main()
