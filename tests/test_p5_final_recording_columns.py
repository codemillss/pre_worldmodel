"""Storage-only synthetic checks: no model construction or physics calls."""
from copy import deepcopy
from dataclasses import replace
import json
import lzma
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from push_align.environment import Observation, SnapshotBundle, StepResult
from push_align.final_recording_columns import (RecordingError, file_sha256, raw_paths,
    read_raw, write_raw, _read_columns, _write_columns, tree_digest, MAGIC, MAX_HEADER_BYTES)



def bits(value):
    return struct.pack("=d", float(value))


class FinalRecordingColumnsTest(unittest.TestCase):
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
        metadata = _read_columns(path)
        metadata["diagnostics"][0]["type"] = "changed"
        with path.open("wb") as target:
            _write_columns(target, metadata)
        with self.assertRaisesRegex(RecordingError, "SHA mismatch"):
            read_raw(self.prefix, expected_sha256=receipt["files_sha256"])

    def rewrite_metadata(self, update):
        _, path = raw_paths(self.prefix)
        metadata = _read_columns(path)
        update(metadata)
        with path.open("wb") as target:
            _write_columns(target, metadata)

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
        with patch("push_align.final_recording_columns._write_columns", side_effect=OSError("synthetic compressor failure")):
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
        with patch("push_align.final_recording_columns.np.savez_compressed", side_effect=fail):
            with self.assertRaises(OSError):
                write_raw(self.prefix, self.snapshot, [self.step()])
        self.assertEqual(raw_paths(self.prefix)[0].read_bytes(), b"synthetic partial npz")
        self.assertFalse(raw_paths(self.prefix)[1].exists())

    def test_external_receipt_requires_both_exact_file_bindings(self):
        write_raw(self.prefix, self.snapshot, [])
        with self.assertRaises(RecordingError):
            read_raw(self.prefix, expected_sha256={})

    def test_preset_must_be_explicit_supported_integer(self):
        for preset in (True, 0, 9, "3"):
            with self.assertRaises(RecordingError):
                write_raw(self.prefix, self.snapshot, [], xz_preset=preset)

    def test_complete_tree_preserves_ragged_mixed_dict_order_and_float_bits(self):
        value = {"samples": [[1.0, -0.0], [], [2.0]],
            "mixed": [True, 1, 1.0, None, "문자열", {}, [], {"b": 2, "a": 1}],
            "contacts": [{"pair": "a", "force": .2}, {"force": .3, "pair": "b"},
                         {"pair": "c", "force": .4, "extra": False}],
            "bits": [float(self.nan_payload), float("inf"), float("-inf"), -.0],
            "huge": [2**128, -(2**128)]}
        path = self.folder / "complete.columns.xz"
        with path.open("xb") as target:
            _write_columns(target, value)
        restored = _read_columns(path)
        self.assertEqual(tree_digest(value), tree_digest(restored))
        self.assertEqual(list(restored["contacts"][1]), ["force", "pair"])

    def corrupt_container(self, update):
        path = raw_paths(self.prefix)[1]
        payload = lzma.decompress(path.read_bytes())
        offset = len(MAGIC) + 8
        count = struct.unpack("<Q", payload[len(MAGIC):offset])[0]
        header = json.loads(payload[offset:offset + count])
        update(header)
        encoded = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode()
        path.write_bytes(lzma.compress(MAGIC + struct.pack("<Q", len(encoded)) + encoded + payload[offset + count:]))

    def test_column_payload_checksum_and_truncation_detected(self):
        write_raw(self.prefix, self.snapshot, [self.step()])
        path = raw_paths(self.prefix)[1]
        valid = path.read_bytes()
        payload = bytearray(lzma.decompress(valid)); payload[-1] ^= 1
        path.write_bytes(lzma.compress(payload))
        with self.assertRaisesRegex(RecordingError, "Column array checksum"):
            read_raw(self.prefix)
        path.write_bytes(valid[:-8])
        with self.assertRaises(RecordingError):
            read_raw(self.prefix)

    def test_object_dtype_and_byte_shape_mismatch_rejected(self):
        write_raw(self.prefix, self.snapshot, [])
        path = raw_paths(self.prefix)[1]; valid = path.read_bytes()
        for field, broken in (("dtype", "|O"), ("shape", [999999]), ("bytes", -1)):
            path.write_bytes(valid)
            self.corrupt_container(lambda h: h["arrays"][0].update({field: broken}))
            with self.assertRaises(RecordingError):
                read_raw(self.prefix)

    def test_bad_recipe_count_reference_and_dictionary_keys_rejected(self):
        write_raw(self.prefix, self.snapshot, [])
        path = raw_paths(self.prefix)[1]; valid = path.read_bytes()
        for mutation in (lambda h: h["recipe"].update(n=2**128),
                         lambda h: h["recipe"].update(keys=["duplicate", "duplicate"]),
                         lambda h: h["recipe"].update(t="float", a=-1)):
            path.write_bytes(valid); self.corrupt_container(mutation)
            with self.assertRaises(RecordingError):
                read_raw(self.prefix)

    def test_trailing_decoded_bytes_and_oversized_header_rejected(self):
        write_raw(self.prefix, self.snapshot, [])
        path = raw_paths(self.prefix)[1]; valid = path.read_bytes()
        path.write_bytes(lzma.compress(lzma.decompress(valid) + b"extra"))
        with self.assertRaisesRegex(RecordingError, "trailing"):
            read_raw(self.prefix)
        path.write_bytes(lzma.compress(MAGIC + struct.pack("<Q", MAX_HEADER_BYTES + 1)))
        with self.assertRaisesRegex(RecordingError, "Oversized"):
            read_raw(self.prefix)

    def test_successful_pair_fsyncs_both_files(self):
        with patch("push_align.final_recording_columns.os.fsync", wraps=os.fsync) as checked:
            write_raw(self.prefix, self.snapshot, [self.step()])
        self.assertEqual(checked.call_count, 2)

    def test_unsupported_metadata_type_preserves_incomplete_pair(self):
        broken = replace(self.step(), diagnostics={"unsupported": object()})
        with self.assertRaises(TypeError):
            write_raw(self.prefix, self.snapshot, [broken])
        self.assertTrue(all(p.is_file() for p in raw_paths(self.prefix)))
        with self.assertRaises(RecordingError):
            read_raw(self.prefix)



if __name__ == "__main__": unittest.main()
