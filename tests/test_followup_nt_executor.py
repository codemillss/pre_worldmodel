"""Small persistence/contract tests using scripted records, with no physics."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import followup_nt_executor as executor
from push_align.final_driver_v2 import exact_comparison, read_full_trace, write_full_trace
import test_followup_probe as probe_tests


class ExecutorContractTest(unittest.TestCase):
    def scripted_result(self):
        helper = probe_tests.FollowupProbeTest()
        helper.setUp()
        return helper.probe(panel="A")[0]

    def test_complete_probe_persistence_preserves_all_fields_and_dataclass_types(self):
        result = self.scripted_result()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            tree = executor.complete_probe_tree(result)
            receipt = write_full_trace(path / "probe", tick=78, captured_input=tree)
            loaded, _, _ = read_full_trace(receipt)
            restored = executor.reconstruct_probe(loaded)
            self.assertEqual(exact_comparison(result, restored)["status"], "PASS_EXACT")
            self.assertEqual(type(restored.results[0]), type(result.results[0]))
            self.assertEqual(len(restored.results[0].diagnostics["samples"]),
                             len(result.results[0].diagnostics["samples"]))

    def test_verifier_error_retains_durable_new_raw_and_receipt(self):
        result = self.scripted_result()
        snapshot = result.final_snapshot  # Codec test: no claim this is the physical initial state.
        with tempfile.TemporaryDirectory() as temporary:
            trial = Path(temporary)
            with patch.object(executor.codec, "read_raw", side_effect=ValueError("injected reader failure")):
                with self.assertRaisesRegex(ValueError, "injected reader failure"):
                    executor.seal_and_validate_prefix(trial, snapshot, list(result.results), {})
            receipt = json.loads((trial / "raw_receipt.json").read_text())
            self.assertTrue(all(Path(path).is_file() for path in receipt["files_sha256"]))
            loaded, steps = executor.codec.read_raw(trial / "raw", expected_sha256=receipt["files_sha256"])
            self.assertEqual(exact_comparison({"snapshot": snapshot, "steps": list(result.results)},
                                             {"snapshot": loaded, "steps": steps})["status"], "PASS_EXACT")

    def test_exact_npz_retains_signed_zero_and_nonfinite_bits(self):
        arrays = {"float": np.array([0., -0., np.nan, np.inf], np.float64),
                  "bool": np.array([True, False]), "labels": np.array(["A", "B"])}
        with tempfile.TemporaryDirectory() as temporary:
            receipt = executor.npz_roundtrip(Path(temporary) / "arrays.npz", arrays)
            self.assertEqual(receipt["roundtrip"]["status"], "PASS_EXACT")

    def test_config_rejects_silent_model_or_repeat_reduction(self):
        config = json.loads((ROOT / "configs/followup_nt_preflight.v2.json").read_text())
        executor.validate_config(config)
        for key, value in (("models", config["models"][:-1]), ("repeats", 1),
                           ("preflight_candidate_ids", [0, 9]), ("cumulative_prior_wall_seconds", 0.)):
            changed = deepcopy(config)
            changed[key] = value
            with self.assertRaises(RuntimeError, msg=key):
                executor.validate_config(changed)

    def test_model_file_binding_uses_exact_original_map_instead_of_presumed_count(self):
        config = json.loads((ROOT / "configs/followup_nt_preflight.v2.json").read_text())
        original = json.loads((ROOT / "artifacts/followup/20261022-d1-d2/entry.json").read_text())["model_and_normalizer_sha256"]
        executor.verify_model_bindings(config["expected_model_normalizer_sha256"], original)
        for change in (dict(list(original.items())[:-1]), {**original, "unexpected": "0"*64}):
            with self.assertRaises(RuntimeError):
                executor.verify_model_bindings(change, original)
        changed = dict(original)
        changed[next(iter(changed))] = "0"*64
        with self.assertRaises(RuntimeError):
            executor.verify_model_bindings(changed, original)


if __name__ == "__main__":
    unittest.main()
