"""Regression for a real late-stage summary serialization failure."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from push_align.config import ROOT


class P2TrainingRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = str(ROOT/"scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        spec = importlib.util.spec_from_file_location("p2_training_run_test_module", ROOT/"scripts/p2_train_pilot.py")
        cls.runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.runner)

    def test_failed_json_serialization_never_creates_partial_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"summary.json"
            for bad in (np.bool_(True), float("nan")):
                with self.subTest(value_type=type(bad).__name__), self.assertRaises((TypeError, ValueError)):
                    self.runner.save(path, {"stage_verdict": "PASS", "checks": {"comparison": bad}})
                self.assertFalse(path.exists())
            self.runner.save(path, {"stage_verdict": "PASS", "checks": {"comparison": bool(np.isclose(1., 1.))}})
            self.assertEqual(json.loads(path.read_text())["checks"]["comparison"], True)

    def test_completed_summary_is_not_overwritten_by_new_result(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"summary.json"
            self.runner.save(path, {"stage_verdict": "FAIL"})
            with self.assertRaises(FileExistsError):
                self.runner.save(path, {"stage_verdict": "PASS"})
            self.assertEqual(json.loads(path.read_text())["stage_verdict"], "FAIL")
