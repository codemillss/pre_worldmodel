"""Storage-only synthetic checks: no model construction or physics calls."""
from copy import deepcopy
from dataclasses import replace
import json
import lzma
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from push_align.environment import Observation, SnapshotBundle, StepResult
from push_align.final_recording import (RecordingError, file_sha256, raw_paths,
                                       read_json_gzip, read_plan_index, read_raw,
                                       write_json_gzip, write_plan_index, write_raw,
                                       write_witness_gzip)


def bits(value):
    return struct.pack("=d", float(value))


class FinalRecordingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.prefix = self.folder / "synthetic"
        self.snapshot = SnapshotBundle(
            np.array([0., -0., .1, np.inf], np.float64), "model-id", "environment-id", -.0,
            {"bit_generator": "PCG64", "state": {"state": 2**127 + 11, "inc": 19},
             "has_uint32": 0, "uinteger": 0},
            np.array([0, 4], np.int32), np.array([0, 17], np.int32), True)
        self.nan_payload = np.array([0x7ff8000000000123], np.uint64).view(np.float64)[0]

    def step(self, count=3, *, valid=True):
        state = np.array([0., -0., self.nan_payload, np.inf, -np.inf, .3, .4, .5, .6, .7], np.float64)
        return StepResult(np.array([-.0, .02], np.float64), np.array([0., .01], np.float64),
            Observation(state, .123, "synthetic-state-v1", valid), .0015, valid,
            {"samples": [{"time": .1, "special": float(self.nan_payload), "negative_zero": -.0,
                          "flags": [False, True], "arbitrary_nested": {"text": "접촉", "inf": np.inf}}],
             "all_original_fields": [None, 2**80, .1], "type": "complete" if valid else "partial"},
            np.vstack([state for _ in range(count)]) if count else np.empty((0, 10), np.float64),
            np.arange(count, dtype=np.float64) * .0005)

    def assert_array_exact(self, left, right):
        self.assertEqual(left.dtype, right.dtype)
        self.assertEqual(left.shape, right.shape)
        self.assertEqual(left.tobytes(), right.tobytes())

    def assert_step_exact(self, a, b):
        for key in ("requested_action", "applied_action", "substep_states", "substep_times"):
            self.assert_array_exact(getattr(a, key), getattr(b, key))
        self.assert_array_exact(a.observation.state, b.observation.state)
        self.assertEqual(bits(a.observation.sim_time), bits(b.observation.sim_time))
        self.assertEqual(a.observation.schema_version, b.observation.schema_version)
        self.assertEqual(a.observation.observation_valid, b.observation.observation_valid)
        self.assertEqual(bits(a.actual_dt), bits(b.actual_dt))
        self.assertEqual(a.transition_valid, b.transition_valid)
        self.assertEqual(bits(a.diagnostics["samples"][0]["special"]),
                         bits(b.diagnostics["samples"][0]["special"]))
        self.assertEqual(bits(a.diagnostics["samples"][0]["negative_zero"]),
                         bits(b.diagnostics["samples"][0]["negative_zero"]))
        aa, bb = deepcopy(a.diagnostics), deepcopy(b.diagnostics)
        aa["samples"][0]["special"] = bb["samples"][0]["special"] = "nonfinite compared bitwise"
        self.assertEqual(aa, bb)

    def test_all_step_and_snapshot_fields_bitwise_roundtrip(self):
        original = [self.step(), self.step(1, valid=False)]
        for preset in (3, 6):
            with self.subTest(preset=preset):
                prefix = self.folder / f"preset-{preset}"
                receipt = write_raw(prefix, self.snapshot, original, xz_preset=preset)
                saved, results = read_raw(prefix, expected_sha256=receipt["files_sha256"])
                self.assertEqual(len(results), 2)
                for key in ("integration_state", "warning_counts", "warning_lastinfo"):
                    self.assert_array_exact(getattr(self.snapshot, key), getattr(saved, key))
                for key in ("model_hash", "environment_hash", "rng_state", "needs_reset"):
                    self.assertEqual(getattr(self.snapshot, key), getattr(saved, key))
                self.assertEqual(bits(saved.reference_height), bits(self.snapshot.reference_height))
                for a, b in zip(original, results):
                    self.assert_step_exact(a, b)

    def test_empty_all_invalid_prefix_retains_complete_snapshot(self):
        receipt = write_raw(self.prefix, self.snapshot, [])
        snapshot, results = read_raw(self.prefix, expected_sha256=receipt["files_sha256"])
        self.assertEqual(results, [])
        self.assert_array_exact(snapshot.integration_state, self.snapshot.integration_state)
        with np.load(raw_paths(self.prefix)[0], allow_pickle=False) as source:
            self.assertEqual(source["state"].shape, (0, 10))
            self.assertEqual(source["action_requested"].shape, (0, 2))
            self.assertEqual(source["substep_states"].shape, (0, 10))
            self.assertEqual(source["substep_offsets"].tolist(), [0])

    def test_partial_intervals_have_explicit_offsets_and_validity(self):
        original = [self.step(3), self.step(0, valid=False), self.step(1, valid=False)]
        write_raw(self.prefix, self.snapshot, original)
        _, restored = read_raw(self.prefix)
        self.assertEqual([len(step.substep_times) for step in restored], [3, 0, 1])
        self.assertEqual([step.transition_valid for step in restored], [True, False, False])
        self.assertEqual([step.observation.observation_valid for step in restored], [True, False, False])

    def test_float32_and_bad_state_shape_rejected_before_files(self):
        step = self.step()
        for broken in (replace(step, requested_action=step.requested_action.astype(np.float32)),
                       replace(step, observation=replace(step.observation, state=np.zeros(9, np.float64))),
                       replace(step, substep_times=np.zeros(2, np.float64))):
            with self.subTest(broken=broken.observation.state.shape):
                with self.assertRaises(RecordingError):
                    write_raw(self.prefix, self.snapshot, [broken])
                self.assertFalse(any(path.exists() for path in raw_paths(self.prefix)))

    def test_non_boolean_flags_and_non_step_input_rejected(self):
        for broken in (replace(self.step(), transition_valid=1),
                       replace(self.step(), observation=replace(self.step().observation, observation_valid="true")),
                       object()):
            with self.assertRaises(RecordingError):
                write_raw(self.prefix, self.snapshot, [broken])

    def test_invalid_snapshot_field_types_rejected(self):
        for broken in (replace(self.snapshot, warning_counts=np.array([0., 1.])),
                       replace(self.snapshot, needs_reset=0),
                       replace(self.snapshot, reference_height=0),
                       replace(self.snapshot, rng_state=[])):
            with self.assertRaises(RecordingError):
                write_raw(self.prefix, broken, [])

    def test_npz_changed_bytes_caught_even_without_external_receipt(self):
        write_raw(self.prefix, self.snapshot, [self.step()])
        path, _ = raw_paths(self.prefix)
        with path.open("ab") as target:
            target.write(b"tampered")
        with self.assertRaisesRegex(RecordingError, "checksum"):
            read_raw(self.prefix)

    def test_json_changed_bytes_caught_with_external_receipt(self):
        receipt = write_raw(self.prefix, self.snapshot, [self.step()])
        _, path = raw_paths(self.prefix)
        with lzma.open(path, "rt") as source:
            metadata = json.load(source)
        metadata["diagnostics"][0]["type"] = "changed"
        with lzma.open(path, "wt") as target:
            json.dump(metadata, target)
        with self.assertRaisesRegex(RecordingError, "SHA mismatch"):
            read_raw(self.prefix, expected_sha256=receipt["files_sha256"])

    def rewrite_metadata(self, update):
        _, path = raw_paths(self.prefix)
        with lzma.open(path, "rt") as source:
            metadata = json.load(source)
        update(metadata)
        with lzma.open(path, "wt") as target:
            json.dump(metadata, target)

    def test_missing_flag_or_schema_not_filled_with_defaults(self):
        write_raw(self.prefix, self.snapshot, [self.step()])
        self.rewrite_metadata(lambda metadata: metadata.pop("observation_schema_versions"))
        with self.assertRaises(RecordingError):
            read_raw(self.prefix)

    def test_missing_numeric_flag_is_not_reconstructed_from_diagnostics(self):
        write_raw(self.prefix, self.snapshot, [self.step(valid=False)])
        path, _ = raw_paths(self.prefix)
        with np.load(path, allow_pickle=False) as source:
            arrays = {key: source[key].copy() for key in source.files if key != "transition_valid"}
        with path.open("wb") as target:
            np.savez_compressed(target, **arrays)
        self.rewrite_metadata(lambda metadata: metadata.update(npz_sha256=file_sha256(path)))
        with self.assertRaises(RecordingError):
            read_raw(self.prefix)

    def test_altered_shape_rejected_even_if_file_checksum_is_updated(self):
        write_raw(self.prefix, self.snapshot, [self.step()])
        path, _ = raw_paths(self.prefix)
        with np.load(path, allow_pickle=False) as source:
            arrays = {key: source[key].copy() for key in source.files}
        arrays["state"] = arrays["state"][:, :9]
        with path.open("wb") as target:
            np.savez_compressed(target, **arrays)
        self.rewrite_metadata(lambda metadata: metadata.update(npz_sha256=file_sha256(path)))
        with self.assertRaisesRegex(RecordingError, "state: float64 array shape"):
            read_raw(self.prefix)

    def test_snapshot_json_and_numeric_duplicate_conflict_is_rejected(self):
        write_raw(self.prefix, self.snapshot, [])
        self.rewrite_metadata(lambda metadata: metadata["initial_snapshot"].update(needs_reset=False))
        with self.assertRaisesRegex(RecordingError, "Snapshot JSON/NPZ mismatch"):
            read_raw(self.prefix)

    def test_nonfinite_path_corruption_rejected(self):
        write_raw(self.prefix, self.snapshot, [self.step()])
        self.rewrite_metadata(lambda metadata: metadata["diagnostic_nonfinite_paths"].append([0, "missing"]))
        with self.assertRaisesRegex(RecordingError, "nonfinite mapping"):
            read_raw(self.prefix)

    def test_duplicate_or_overwrite_prefix_refused_and_original_unchanged(self):
        receipt = write_raw(self.prefix, self.snapshot, [self.step()])
        with self.assertRaises(FileExistsError):
            write_raw(self.prefix, self.snapshot, [])
        self.assertEqual(receipt["files_sha256"], {str(path): file_sha256(path) for path in raw_paths(self.prefix)})

    def test_existing_one_file_refuses_new_pair(self):
        path, other = raw_paths(self.prefix)
        path.write_bytes(b"prior incomplete attempt")
        with self.assertRaises(FileExistsError):
            write_raw(self.prefix, self.snapshot, [])
        self.assertEqual(path.read_bytes(), b"prior incomplete attempt")
        self.assertFalse(other.exists())

    def test_compression_failure_keeps_completed_npz_and_partial_json(self):
        with patch("push_align.final_recording.lzma.open", side_effect=OSError("synthetic compressor failure")):
            with self.assertRaises(OSError):
                write_raw(self.prefix, self.snapshot, [self.step()])
        path, partial = raw_paths(self.prefix)
        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 0)
        self.assertTrue(partial.is_file())
        with self.assertRaises(FileExistsError):
            write_raw(self.prefix, self.snapshot, [])

    def test_numeric_write_failure_keeps_partial_prefix(self):
        def fail(stream, **arrays):
            stream.write(b"synthetic partial npz")
            raise OSError("synthetic NPZ failure")
        with patch("push_align.final_recording.np.savez_compressed", side_effect=fail):
            with self.assertRaises(OSError):
                write_raw(self.prefix, self.snapshot, [self.step()])
        self.assertEqual(raw_paths(self.prefix)[0].read_bytes(), b"synthetic partial npz")
        self.assertFalse(raw_paths(self.prefix)[1].exists())

    def test_arbitrary_json_witness_full_roundtrip_and_checksum(self):
        path = self.folder / "witness.json.gz"
        witness = {"intervals": [{"index": 29, "holding": True, "value": -.0}],
                   "unexecuted_plan": {"reason": "planner_invalid", "rows": [None, "NaN", "Infinity"]},
                   "evidence": {"hashes": ["a" * 64], "unicode": "재현", "large_id": 2**100}}
        receipt = write_json_gzip(path, witness)
        actual = read_json_gzip(path, expected_sha256=receipt["files_sha256"][str(path)])
        self.assertEqual(witness, actual)
        self.assertEqual(bits(witness["intervals"][0]["value"]), bits(actual["intervals"][0]["value"]))
        with self.assertRaises(FileExistsError):
            write_json_gzip(path, {})
        with path.open("ab") as target:
            target.write(b"bad")
        with self.assertRaisesRegex(RecordingError, "SHA mismatch"):
            read_json_gzip(path, expected_sha256=receipt["files_sha256"][str(path)])

    def test_non_json_values_and_untagged_nonfinites_rejected(self):
        for bad in ({"nan": np.nan}, {"not_json": object()}):
            with self.assertRaises((ValueError, TypeError)):
                write_json_gzip(self.folder / "invalid.json.gz", bad)
            self.assertFalse((self.folder / "invalid.json.gz").exists())

    def test_external_receipt_requires_both_exact_file_bindings(self):
        write_raw(self.prefix, self.snapshot, [])
        with self.assertRaises(RecordingError):
            read_raw(self.prefix, expected_sha256={})

    def test_preset_must_be_explicit_supported_integer(self):
        for preset in (True, 0, 9, "3"):
            with self.assertRaises(RecordingError):
                write_raw(self.prefix, self.snapshot, [], xz_preset=preset)

    def plan_arrays(self, invalid=False):
        return {"valid": np.array([not invalid, False, False], bool),
                "reasons": np.array(["", "nonfinite_delta", "부호 오류"]),
                "failure_steps": np.array([-1, 0, 4], np.int64),
                "total_cost": np.array([-.0, np.inf, self.nan_payload], np.float64),
                "cost_position": np.array([.1, -np.inf, self.nan_payload], np.float64),
                "cost_terminal": np.array([.2, .3, .4], np.float64)}

    def test_plan_index_every_array_exact_and_outer_zip_stored(self):
        path = self.folder / "plans.zip"
        records = [{"tick": 0, "arrays": self.plan_arrays()},
                   {"tick": 29, "arrays": self.plan_arrays(invalid=True)}]
        receipt = write_plan_index(path, records)
        restored = read_plan_index(path, expected_sha256=receipt["files_sha256"][str(path)])
        self.assertEqual([row["tick"] for row in restored], [0, 29])
        for actual, expected in zip(restored, records):
            self.assertEqual(set(actual["arrays"]), set(expected["arrays"]))
            for key in expected["arrays"]:
                self.assert_array_exact(actual["arrays"][key], expected["arrays"][key])
        with zipfile.ZipFile(path) as archive:
            self.assertTrue(all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist()))

    def test_empty_plan_archive_is_explicit_no_model_plan_case(self):
        path = self.folder / "empty-plans.zip"
        receipt = write_plan_index(path, [])
        self.assertEqual(read_plan_index(path, expected_sha256=receipt["files_sha256"][str(path)]), [])

    def test_plan_full_arrays_object_or_bad_shapes_not_silently_saved(self):
        arrays = self.plan_arrays()
        invalid = [{**arrays, "predicted_states": np.zeros((3, 6, 10), np.float64)},
                   {**arrays, "reasons": np.array([None, "bad", ""], object)},
                   {**arrays, "total_cost": np.zeros(2, np.float64)},
                   {key: value for key, value in arrays.items() if key != "valid"}]
        for index, broken in enumerate(invalid):
            path = self.folder / f"bad-{index}.zip"
            with self.assertRaises(RecordingError):
                write_plan_index(path, [{"tick": 0, "arrays": broken}])
            self.assertFalse(path.exists())

    def test_plan_ticks_duplicate_unsorted_or_bool_rejected(self):
        for ticks in ([0, 0], [29, 0], [True], [-1]):
            with self.assertRaises(RecordingError):
                write_plan_index(self.folder / "invalid-ticks.zip",
                                 [{"tick": tick, "arrays": self.plan_arrays()} for tick in ticks])

    def test_plan_checksum_and_overwrite_refusal(self):
        path = self.folder / "plans.zip"
        receipt = write_plan_index(path, [{"tick": 0, "arrays": self.plan_arrays()}])
        with self.assertRaises(FileExistsError):
            write_plan_index(path, [])
        with path.open("ab") as target:
            target.write(b"tampered")
        with self.assertRaisesRegex(RecordingError, "SHA mismatch"):
            read_plan_index(path, expected_sha256=receipt["files_sha256"][str(path)])

    def test_plan_member_checksum_detects_replaced_values(self):
        path, altered = self.folder / "plans.zip", self.folder / "altered-plans.zip"
        write_plan_index(path, [{"tick": 0, "arrays": self.plan_arrays()}])
        with zipfile.ZipFile(path) as source, zipfile.ZipFile(altered, "w") as target:
            for name in source.namelist():
                payload = source.read(name)
                target.writestr(name, payload + b"tampered" if name.endswith(".npz") else payload)
        with self.assertRaisesRegex(RecordingError, "member checksum"):
            read_plan_index(altered)

    def test_plan_compression_error_retains_partial_and_forbids_reuse(self):
        path = self.folder / "partial-plans.zip"
        with patch("push_align.final_recording.np.savez_compressed", side_effect=OSError("synthetic failure")):
            with self.assertRaises(OSError):
                write_plan_index(path, [{"tick": 0, "arrays": self.plan_arrays()}])
        self.assertTrue(path.is_file())
        with self.assertRaises(RecordingError):
            read_plan_index(path)
        with self.assertRaises(FileExistsError):
            write_plan_index(path, [])

    def test_member_witness_empty_lists_remain_single_logical_json(self):
        path = self.folder / "empty-witness.json.gz"
        witness = {"schema": "unchanged-v1", "intervals": [], "decisions": [], "terminal": "planner_invalid"}
        receipt = write_witness_gzip(path, witness)
        actual = read_json_gzip(path, expected_sha256=receipt["files_sha256"][str(path)])
        self.assertEqual(actual, witness)
        self.assertEqual(receipt["fixed_bytes"], path.stat().st_size)
        self.assertEqual(receipt["interval_member_bytes"], [])
        self.assertEqual(receipt["decision_member_bytes"], [])

    def test_member_witness_three_frames_and_byte_sum_exact(self):
        path = self.folder / "witness-3.json.gz"
        witness = {"schema": "p5-controller-boundary-witness-v1", "nested": {"한글": "보존"},
                   "intervals": [{"tick": i, "holding": i == 2, "state": [0., -.0, i / 3]} for i in range(3)],
                   "decisions": [{"index": i, "plan": {"cost": "Infinity"}, "rng": [11, 5, i]} for i in range(3)]}
        receipt = write_witness_gzip(path, witness)
        actual = read_json_gzip(path, expected_sha256=receipt["files_sha256"][str(path)])
        self.assertEqual(actual, witness)
        self.assertEqual(bits(actual["intervals"][0]["state"][1]), bits(-.0))
        self.assertEqual(len(receipt["interval_member_bytes"]), 3)
        self.assertEqual(len(receipt["decision_member_bytes"]), 3)
        self.assertEqual(path.stat().st_size, receipt["fixed_bytes"]
            + sum(receipt["interval_member_bytes"]) + sum(receipt["decision_member_bytes"]))
        with self.assertRaises(FileExistsError):
            write_witness_gzip(path, witness)

    def test_member_witness_no_other_fields_is_valid(self):
        path = self.folder / "bare-witness.json.gz"
        value = {"intervals": [{"flag": False}], "decisions": [{"flag": None}]}
        write_witness_gzip(path, value)
        self.assertEqual(read_json_gzip(path), value)

    def test_member_witness_missing_lists_not_filled_in(self):
        for value in ({}, {"intervals": [], "decisions": None}):
            with self.assertRaises(RecordingError):
                write_witness_gzip(self.folder / "missing-witness.json.gz", value)

    def test_member_witness_midstream_bad_json_keeps_partial(self):
        path = self.folder / "bad-witness.json.gz"
        with self.assertRaises(TypeError):
            write_witness_gzip(path, {"intervals": [{"ok": True}, {"bad": object()}], "decisions": []})
        self.assertTrue(path.is_file())
        with self.assertRaises(json.JSONDecodeError):
            read_json_gzip(path)
        with self.assertRaises(FileExistsError):
            write_witness_gzip(path, {"intervals": [], "decisions": []})


if __name__ == "__main__":
    unittest.main()
