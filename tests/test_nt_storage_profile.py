"""Measurement and evidence-contract checks, no model or physical execution."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import profile_nt_storage as profile


class CounterClock:
    def __init__(self):
        self.value = 0
    def __call__(self):
        self.value += 1
        return self.value


class ProfileContract(unittest.TestCase):
    def test_nested_and_repeated_spans_have_no_double_charge(self):
        meter = profile.Meter(CounterClock(), CounterClock())
        with meter.span("bundle"):
            with meter.span("write"):
                with meter.span("compare"):
                    pass
            with meter.span("write"):
                pass
        meter.verify()
        self.assertEqual(meter.records["bundle/write"]["calls"], 2)
        self.assertGreater(sum(r["inclusive_ns"] for r in meter.records.values()),
                           meter.records["bundle"]["inclusive_ns"])
        meter.records["bundle/write"]["exclusive_ns"] += 1
        with self.assertRaisesRegex(ValueError, "double-count"):
            meter.verify()

    def test_original_ast_copy_and_bits_survive_complete_codec(self):
        original_write = profile.driver.write_full_trace
        meter = profile.Meter()
        writer, reader, comparison = profile.timed_functions(meter)
        values = {"tuple": (np.array([0., -0., np.nan], np.float64), np.int32(4)),
                  "list": [True, None, 7.], "nested": {3: "key", "x": [1, 2]}}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with meter.span("bundle"):
                with meter.span("write"):
                    receipt = writer(root / "timed", tick=78, captured_input=values)
                with meter.span("read"):
                    restored, _, _ = reader(receipt)
                with meter.span("compare"):
                    self.assertEqual(comparison(values, restored)["status"], "PASS_EXACT")
            meter.verify()
            reference = original_write(root / "reference", tick=78, captured_input=values)
            self.assertEqual({Path(p).name: h for p, h in receipt["files_sha256"].items()},
                             {Path(p).name: h for p, h in reference["files_sha256"].items()})
            restored["tuple"][0][1] = 0.
            self.assertEqual(profile.driver.exact_comparison(values, restored)["status"], "FAIL_EXACT")
        self.assertIs(profile.driver.write_full_trace, original_write)

    def test_corrupt_metadata_is_rejected_before_accepted_read(self):
        with tempfile.TemporaryDirectory() as name:
            receipt = profile.driver.write_full_trace(Path(name) / "saved", tick=78,
                                                     captured_input={"x": np.arange(3)})
            metadata = next(Path(p) for p in receipt["files_sha256"] if p.endswith("input.json.gz"))
            metadata.write_bytes(metadata.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "changed full trace"):
                profile.driver.read_full_trace(receipt)


if __name__ == "__main__":
    unittest.main()
