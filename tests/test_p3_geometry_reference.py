"""Independent admission witnesses for a formally accepted G reference.

References here are synthetic metadata fixtures, never physical acceptance
evidence. The counterexamples test whether actual main collection is bound to
the formal mode, exact physical contract and controller runtime dependencies;
the gate-only witness mocks physics and cannot establish numerical accuracy.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from push_align import PlanarPushEnv
from push_align.config import ROOT
from push_align.data import contract_hash, digest, make_contract


REQUIRED_FORMAL_CHECKS = (
    "four_fixed_cases_accounted", "four_cases_ready", "four_case_requirements",
    "regression_tests_pass", "four_raw_episodes_audited", "future_engine_queries_zero",
    "no_unresolved_physics", "source_frozen", "prior_artifacts_unchanged",
)
RUNTIME_DEPENDENCIES = (
    "geometry_control.py", "environment.py", "config.py", "planning.py", "task.py",
    "representation.py", "contact_geometry.py", "closed_loop.py", "data.py",
    "geometry_control_smooth.py",
)
TRANSITION = {
    "schema_version": "geometry-transition-v2",
    "offset_taper_start_m": .002,
    "phase_end_taper_s": .4,
    "phase_entry_ramp_s": .4,
    "slide_blend_outer_tolerance_multiplier": 2.0,
    "free_motion_speed_max_m_s": .01,
    "clock_snap_tolerance_s": 1e-9,
}


class GeometryReferenceAdmissionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = str(ROOT / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        spec = importlib.util.spec_from_file_location(
            "p3_geometry_reference_test_runner", ROOT / "scripts/p3_collect_main.py")
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)
        cls.contract = make_contract(PlanarPushEnv())

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="p3-reference-contract-")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name) / "reference"
        self.folder.mkdir()
        self.parameter_path = Path(self.temporary.name) / "selected.json"
        self.parameter_path.write_bytes((ROOT / "configs/p3_geometry_selected.v1.json").read_bytes())
        self.settings = {"geometry_transition": {
            **TRANSITION,
            "noise_weighting": "same predeclared PUSH-only vector times continuous phase entry and pre-reassessment gain; zero in holding/free modes",
        }}
        self.reference = {
            "task_id": "P3-01", "mode": "formal_four_case_acceptance",
            "contract_hash": contract_hash(self.contract),
            "profile_sha256": self.contract["profile_sha256"],
            "physical_config": deepcopy(self.contract["resolved_config"]),
            "transition_parameters": deepcopy(TRANSITION),
            "requested_parameter_file_sha256": digest(self.parameter_path),
            "source_hashes": {
                "src/push_align/" + name: digest(ROOT / "src/push_align" / name)
                for name in RUNTIME_DEPENDENCIES
            },
        }
        self.result = {
            "task_id": "P3-01", "mode": "formal_four_case_acceptance", "pass": True,
            "checks": {name: True for name in REQUIRED_FORMAL_CHECKS},
        }

    def write_reference(self):
        (self.folder / "manifest.json").write_text(json.dumps(self.reference), encoding="utf-8")
        (self.folder / "summary.json").write_text(json.dumps(self.result), encoding="utf-8")

    def checks(self):
        self.write_reference()
        return self.runner.geometry_reference_checks(
            self.folder, self.contract, self.parameter_path, self.settings)

    def assert_only_rejected(self, name):
        checks = self.checks()
        self.assertFalse(checks[name])
        self.assertFalse(all(checks.values()))
        self.assertTrue(all(value for key, value in checks.items() if key != name))

    def test_same_formal_contract_and_all_runtime_dependencies_are_admitted(self):
        self.assertTrue(all(self.checks().values()))

    def test_legacy_reference_requires_no_smooth_module_or_transition(self):
        self.settings = {}
        self.reference["transition_parameters"] = None
        del self.reference["source_hashes"]["src/push_align/geometry_control_smooth.py"]
        self.assertTrue(all(self.checks().values()))

    def test_development_manifest_cannot_be_used_as_formal_even_when_passed(self):
        self.reference["mode"] = "development_tuning"
        self.assert_only_rejected("G_reference_is_formal_acceptance")

    def test_development_summary_cannot_be_used_as_formal_even_when_passed(self):
        self.result["mode"] = "development_tuning"
        self.assert_only_rejected("G_reference_is_formal_acceptance")

    def test_wrong_manifest_task_is_rejected(self):
        self.reference["task_id"] = "P3-02"
        self.assert_only_rejected("G_reference_is_formal_acceptance")

    def test_wrong_summary_task_is_rejected(self):
        self.result["task_id"] = "P3-02"
        self.assert_only_rejected("G_reference_is_formal_acceptance")

    def test_pass_requires_actual_boolean_true(self):
        for value in (False, None, 1, "PASS", "true"):
            with self.subTest(value=value):
                self.result["pass"] = value
                self.assert_only_rejected("G_reference_is_formal_acceptance")

    def test_each_missing_required_formal_check_is_rejected(self):
        for name in REQUIRED_FORMAL_CHECKS:
            with self.subTest(check=name):
                del self.result["checks"][name]
                self.assert_only_rejected("G_reference_is_formal_acceptance")
                self.result["checks"][name] = True

    def test_each_false_or_nonboolean_formal_check_is_rejected(self):
        for name in REQUIRED_FORMAL_CHECKS:
            for value in (False, 1, "PASS"):
                with self.subTest(check=name, value=value):
                    self.result["checks"][name] = value
                    self.assert_only_rejected("G_reference_is_formal_acceptance")
            self.result["checks"][name] = True

    def test_missing_formal_checks_dictionary_is_rejected(self):
        del self.result["checks"]
        self.assert_only_rejected("G_reference_is_formal_acceptance")

    def test_different_physical_contract_hash_is_rejected(self):
        self.reference["contract_hash"] = "0" * 64
        self.assert_only_rejected("G_reference_contract_matches")

    def test_different_profile_is_rejected_even_if_other_contract_fields_match(self):
        self.reference["profile_sha256"] = "0" * 64
        self.assert_only_rejected("G_reference_contract_matches")

    def test_different_physical_config_is_rejected_even_if_claimed_hash_matches(self):
        self.reference["physical_config"]["physics"]["timestep_s"] *= .5
        self.assert_only_rejected("G_reference_contract_matches")

    def test_missing_physical_contract_field_is_rejected(self):
        for key in ("contract_hash", "profile_sha256", "physical_config"):
            with self.subTest(field=key):
                old = self.reference.pop(key)
                self.assert_only_rejected("G_reference_contract_matches")
                self.reference[key] = old

    def test_transition_parameter_value_change_is_rejected(self):
        self.reference["transition_parameters"]["phase_entry_ramp_s"] = .2
        self.assert_only_rejected("G_reference_transition_matches")

    def test_transition_schema_change_is_rejected(self):
        self.reference["transition_parameters"]["schema_version"] = "geometry-transition-v1"
        self.assert_only_rejected("G_reference_transition_matches")

    def test_missing_transition_parameter_is_rejected(self):
        del self.reference["transition_parameters"]["clock_snap_tolerance_s"]
        self.assert_only_rejected("G_reference_transition_matches")

    def test_extra_transition_parameter_cannot_be_hidden_by_subset_matching(self):
        self.reference["transition_parameters"]["additional_unverified_knob"] = .001
        self.assert_only_rejected("G_reference_transition_matches")

    def test_new_transition_cannot_use_legacy_none_reference(self):
        self.reference["transition_parameters"] = None
        self.assert_only_rejected("G_reference_transition_matches")

    def test_legacy_collection_cannot_silently_drop_accepted_transition(self):
        self.settings = {}
        self.assert_only_rejected("G_reference_transition_matches")

    def test_selected_json_byte_change_is_rejected_even_with_equivalent_numeric_json(self):
        value = json.loads(self.parameter_path.read_text())
        self.parameter_path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        self.assert_only_rejected("G_reference_selected_parameters_match")

    def test_missing_selected_parameter_hash_is_rejected(self):
        del self.reference["requested_parameter_file_sha256"]
        self.assert_only_rejected("G_reference_selected_parameters_match")

    def test_each_runtime_dependency_hash_change_is_rejected(self):
        for name in RUNTIME_DEPENDENCIES:
            with self.subTest(source=name):
                key = "src/push_align/" + name
                old = self.reference["source_hashes"][key]
                self.reference["source_hashes"][key] = "0" * 64
                self.assert_only_rejected("G_reference_runtime_dependencies_match")
                self.reference["source_hashes"][key] = old

    def test_each_missing_runtime_dependency_hash_is_rejected(self):
        for name in RUNTIME_DEPENDENCIES:
            with self.subTest(source=name):
                key = "src/push_align/" + name
                old = self.reference["source_hashes"].pop(key)
                self.assert_only_rejected("G_reference_runtime_dependencies_match")
                self.reference["source_hashes"][key] = old

    def test_missing_source_hash_dictionary_is_rejected(self):
        del self.reference["source_hashes"]
        self.assert_only_rejected("G_reference_runtime_dependencies_match")

    def runner_fixture(self, *, gate_only=False, stdout="Ran 1 tests\nOK\n"):
        """Mocks execution only; the actual admission helper remains intact."""
        self.write_reference()
        out = Path(self.temporary.name) / "run"
        out.mkdir()
        args = SimpleNamespace(
            collection_config=ROOT / "configs/p3_collection.v1.toml",
            geometry_parameters=ROOT / "configs/p3_geometry_selected.v1.json",
            geometry_reference_run=self.folder,
            gate_only=gate_only,
        )
        # For run() these proposed values come from its selected TOML. Supply
        # the identical bounded fixture via its plan-settings validation.
        real_toml = self.runner.tomllib.loads
        def parsed(text):
            value = real_toml(text)
            if "collection_id" in value:
                value["geometry_transition"] = deepcopy(self.settings["geometry_transition"])
            return value
        return out, args, (
            patch.object(self.runner, "PlanarPushEnv", return_value=SimpleNamespace(config=self.contract["resolved_config"])),
            patch.object(self.runner, "make_contract", return_value=self.contract),
            patch.object(self.runner, "build_main_collection_plan", return_value=([], {})),
            patch.object(self.runner, "validate_settings"),
            patch.object(self.runner, "source_bundle", return_value={}),
            patch.object(self.runner, "verify_artifacts", return_value={"count": 0, "manifest_sha256": "fixture"}),
            patch.object(self.runner.shutil, "disk_usage", return_value=SimpleNamespace(free=10**12)),
            patch.object(self.runner.platform, "platform", return_value="synthetic-admission-fixture"),
            patch.object(self.runner.tomllib, "loads", side_effect=parsed),
            patch.object(self.runner.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=stdout, stderr="")),
            patch.object(self.runner, "physics_gate", return_value={"pass": True}),
            patch.object(self.runner, "collect", side_effect=AssertionError("Collection must not start in this witness")),
        )

    def test_actual_main_rejects_development_reference_before_tests_or_physics(self):
        from contextlib import ExitStack
        self.reference["mode"] = self.result["mode"] = "development_tuning"
        out, args, patches = self.runner_fixture()
        with ExitStack() as stack:
            mocks = [stack.enter_context(item) for item in patches]
            with self.assertRaisesRegex(ValueError, "Prerequisites"):
                self.runner.run(args, out)
            mocks[-3].assert_not_called()
            mocks[-2].assert_not_called()
            mocks[-1].assert_not_called()

    def test_gate_only_candidate_remains_separate_and_never_collects(self):
        from contextlib import ExitStack
        self.reference["mode"] = self.result["mode"] = "development_tuning"
        out, args, patches = self.runner_fixture(gate_only=True)
        with ExitStack() as stack:
            mocks = [stack.enter_context(item) for item in patches]
            self.assertTrue(self.runner.run(args, out))
            mocks[-2].assert_called_once()
            mocks[-1].assert_not_called()
        result = json.loads((out / "summary.json").read_text())
        self.assertEqual(result["stage_verdict"], "GATE_PASS")
        self.assertEqual(result["main_collection"], "NOT_RUN")

    def test_zero_regression_tests_cannot_open_main_admission(self):
        from contextlib import ExitStack
        out, args, patches = self.runner_fixture(stdout="Ran 0 tests\nOK\n")
        with ExitStack() as stack:
            mocks = [stack.enter_context(item) for item in patches]
            with self.assertRaisesRegex(ValueError, "Positive complete regression"):
                self.runner.run(args, out)
            mocks[-2].assert_not_called()
            mocks[-1].assert_not_called()

    def test_missing_regression_count_cannot_open_main_admission(self):
        from contextlib import ExitStack
        out, args, patches = self.runner_fixture(stdout="OK\n")
        with ExitStack() as stack:
            mocks = [stack.enter_context(item) for item in patches]
            with self.assertRaisesRegex(ValueError, "Positive complete regression"):
                self.runner.run(args, out)
            mocks[-2].assert_not_called()

    def test_skipped_regression_tests_cannot_open_main_admission(self):
        from contextlib import ExitStack
        out, args, patches = self.runner_fixture(stdout="Ran 10 tests\nOK (skipped=1)\n")
        with ExitStack() as stack:
            mocks = [stack.enter_context(item) for item in patches]
            with self.assertRaisesRegex(ValueError, "Positive complete regression"):
                self.runner.run(args, out)
            mocks[-2].assert_not_called()


if __name__ == "__main__":
    unittest.main()
