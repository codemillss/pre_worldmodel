"""Boundary tests for the opt-in exact-comparison prototype.

Only tiny synthetic values are used.  These tests do not construct or step a
physical environment, load learned models, or modify saved experiment records.
The existing comparator is the compatibility oracle, including its diagnostic
JSON normalization and its rejection of unsupported evidence types.
"""
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass, make_dataclass, replace
import struct
import unittest

import numpy as np

from push_align.environment import Observation, StepResult
from push_align import final_driver_v2 as original
from push_align import exact_pair_prototype as prototype


def float_from_bits(bits):
    return struct.unpack("=d", struct.pack("=Q", bits))[0]


def step_result(diagnostics):
    return StepResult(
        requested_action=np.array([0.01, -0.02], dtype=np.float64),
        applied_action=np.array([0.01, -0.02], dtype=np.float64),
        observation=Observation(np.zeros(10, dtype=np.float64), 0.1, "synthetic-v1", True),
        actual_dt=0.1,
        transition_valid=True,
        diagnostics=diagnostics,
        substep_states=np.zeros((2, 10), dtype=np.float64),
        substep_times=np.array([0.0, 0.1], dtype=np.float64),
    )


@dataclass
class Evidence:
    state: object
    flags: object


class ExactPairPrototypeTest(unittest.TestCase):
    def assert_same_verdict(self, left, right, *, expected=None):
        before_left = original.typed_digest(left)
        before_right = original.typed_digest(right)
        reference = original.exact_comparison(left, right)
        result = prototype.exact_comparison(left, right)
        self.assertEqual(result, reference)
        if expected is not None:
            self.assertEqual(result["status"], expected)
        self.assertEqual(original.typed_digest(left), before_left)
        self.assertEqual(original.typed_digest(right), before_right)
        return result

    def test_scalar_bits_and_types_are_preserved(self):
        values = [None, False, True, 0, -7, 2**80, "", "부품\u0000정지", 0.0, -0.0,
                  float("inf"), -float("inf"), float_from_bits(0x7ff8000000000011)]
        for value in values:
            with self.subTest(value_type=type(value).__name__, representation=repr(value)):
                self.assert_same_verdict(value, deepcopy(value), expected="PASS_EXACT")
        pairs = [(False, 0), (True, 1), (1, 1.0), (0.0, -0.0),
                 (float_from_bits(0x7ff8000000000011), float_from_bits(0x7ff8000000000012)),
                 ("1", 1), (None, "")]
        for left, right in pairs:
            with self.subTest(left=repr(left), right=repr(right)):
                self.assert_same_verdict(left, right, expected="FAIL_EXACT")

    def test_arrays_keep_dtype_shape_bytes_and_logical_layout(self):
        base = np.arange(12, dtype=np.float64).reshape(3, 4)
        same_values = [base[:, ::2], base.T, np.asfortranarray(base),
                       np.zeros((0, 3), dtype=np.float32), np.zeros((3, 0), dtype=np.int16),
                       np.asarray(-0.0, dtype=np.float64), np.array([1, 2], dtype=">i4")]
        for value in same_values:
            with self.subTest(shape=value.shape, dtype=value.dtype.str):
                self.assert_same_verdict(value, value.copy(order="C"), expected="PASS_EXACT")
        self.assert_same_verdict(base, base.astype(np.float32), expected="FAIL_EXACT")
        self.assert_same_verdict(base, base.reshape(4, 3), expected="FAIL_EXACT")
        changed = base.copy()
        changed[1, 2] = np.nextafter(changed[1, 2], np.inf)
        self.assert_same_verdict(base, changed, expected="FAIL_EXACT")
        self.assert_same_verdict(np.array([1, 2], dtype="<i4"),
                                 np.array([1, 2], dtype=">i4"), expected="FAIL_EXACT")
        self.assert_same_verdict(np.zeros((0, 2)), np.zeros((2, 0)), expected="FAIL_EXACT")

    def test_numpy_scalars_are_distinct_outside_diagnostics(self):
        for value in [np.bool_(True), np.int32(-3), np.int64(-3), np.float32(0.25), np.float64(-0.0)]:
            with self.subTest(dtype=value.dtype.str):
                self.assert_same_verdict(value, value.copy(), expected="PASS_EXACT")
                self.assert_same_verdict(value, value.item(), expected="FAIL_EXACT")
                self.assert_same_verdict(value, np.asarray(value), expected="FAIL_EXACT")
        self.assert_same_verdict(np.int32(4), np.int64(4), expected="FAIL_EXACT")

    def test_dictionary_order_and_tied_string_sort_keys_follow_original(self):
        self.assert_same_verdict({"z": 1, 2: "two", "a": (3, 4)},
                                 {"a": (3, 4), 2: "two", "z": 1}, expected="PASS_EXACT")
        self.assert_same_verdict({1: "integer"}, {"1": "integer"}, expected="FAIL_EXACT")
        # The existing stable sort uses str(key); equal sort keys keep insertion
        # order.  Compatibility must not silently invent a new key convention.
        self.assert_same_verdict({1: "integer", "1": "string"},
                                 {"1": "string", 1: "integer"}, expected="FAIL_EXACT")

    def test_equal_hash_different_dataclass_type_still_requires_direct_check(self):
        left_type = make_dataclass("SameEvidenceName", [("value", int)])
        right_type = make_dataclass("SameEvidenceName", [("value", int)])
        left, right = left_type(7), right_type(7)
        self.assertIsNot(type(left), type(right))
        # No mocked hasher: the frozen protocol hashes the class name, whereas
        # direct equality additionally requires the actual class identity.
        self.assertEqual(original.typed_digest(left), original.typed_digest(right))
        self.assert_same_verdict(left, right, expected="FAIL_EXACT")

    def test_equal_hash_tuple_subclass_still_requires_direct_check(self):
        pair_type = namedtuple("Pair", ["x", "y"])
        left, right = pair_type(1, 2), (1, 2)
        self.assertEqual(original.typed_digest(left), original.typed_digest(right))
        self.assert_same_verdict(left, right, expected="FAIL_EXACT")

    def test_nested_mutations_and_sequence_types_are_not_hidden(self):
        left = Evidence({"stages": [{"state": np.array([1.0, -0.0]), "valid": True}]},
                        (False, {"complete": True}))
        self.assert_same_verdict(left, deepcopy(left), expected="PASS_EXACT")
        mutations = []
        changed = deepcopy(left)
        changed.state["stages"][0]["state"][1] = 0.0
        mutations.append(changed)
        changed = deepcopy(left)
        changed.state["stages"][0]["valid"] = 1
        mutations.append(changed)
        changed = deepcopy(left)
        changed.state["stages"].append({"state": np.zeros(2), "valid": True})
        mutations.append(changed)
        mutations.append(replace(deepcopy(left), flags=[False, {"complete": True}]))
        for index, changed in enumerate(mutations):
            with self.subTest(mutation=index):
                self.assert_same_verdict(left, changed, expected="FAIL_EXACT")

    def test_diagnostic_json_normalization_is_preserved(self):
        left = step_result({1: (np.int32(3), np.float32(0.25), np.asarray(-0.0)),
                            "contacts": np.array([[1, 2], [3, 4]], dtype=np.int16),
                            "valid": np.bool_(True)})
        right = step_result({"1": [3, 0.25, -0.0], "contacts": [[1, 2], [3, 4]], "valid": True})
        self.assert_same_verdict(left, right, expected="PASS_EXACT")
        self.assert_same_verdict([left], [right], expected="PASS_EXACT")

    def test_diagnostic_normalization_does_not_leak_to_state_fields(self):
        left = step_result({"samples": [1, 2]})
        changes = [replace(left, requested_action=left.requested_action.astype(np.float32)),
                   replace(left, requested_action=left.requested_action.tolist()),
                   replace(left, transition_valid=1),
                   replace(left, observation=replace(left.observation, observation_valid=np.bool_(True)))]
        for index, right in enumerate(changes):
            with self.subTest(field_change=index):
                self.assert_same_verdict(left, right, expected="FAIL_EXACT")

    def test_diagnostic_leaf_mutations_are_detected_after_normalization(self):
        left = step_result({"contact": {"normal": 1.0, "valid": True}, "samples": [1, 2]})
        changes = [replace(left, diagnostics={"contact": {"normal": 1.0000000001, "valid": True},
                                              "samples": [1, 2]}),
                   replace(left, diagnostics={"contact": {"normal": 1.0, "valid": 1},
                                              "samples": [1, 2]}),
                   replace(left, diagnostics={"contact": {"normal": 1.0, "valid": True},
                                              "samples": [1, 2, 3]})]
        for index, right in enumerate(changes):
            with self.subTest(mutation=index):
                self.assert_same_verdict(left, right, expected="FAIL_EXACT")

    def test_nested_dataclass_in_diagnostics_matches_existing_normalization(self):
        left = step_result({"record": Evidence(np.array([1, 2], dtype=np.int32), (np.bool_(True),))})
        right = step_result({"record": Evidence([1, 2], [True])})
        self.assert_same_verdict(left, right, expected="PASS_EXACT")

    def test_readonly_and_aliased_values_are_not_modified(self):
        values = np.arange(5, dtype=np.float64)
        values.flags.writeable = False
        left = {"first": values, "second": values}
        right = {"first": values.copy(), "second": values.copy()}
        self.assert_same_verdict(left, right, expected="PASS_EXACT")
        self.assertFalse(values.flags.writeable)
        self.assertIs(left["first"], left["second"])

    def test_unsupported_evidence_is_rejected_even_if_shapes_already_differ(self):
        bad_values = [np.array([{"x": 1}], dtype=object), {True: 1}, {np.int64(1): 1},
                      {("key",): 1}, {"set": {1, 2}}, {"opaque": object()},
                      Evidence(1, {"bad": object()})]
        for index, bad in enumerate(bad_values):
            for left, right in [(bad, bad), ([], bad), (bad, [])]:
                with self.subTest(case=index, left_type=type(left).__name__, right_type=type(right).__name__):
                    with self.assertRaises(Exception) as reference:
                        original.exact_comparison(left, right)
                    with self.assertRaises(type(reference.exception)):
                        prototype.exact_comparison(left, right)

    def test_tiny_seeded_mixed_trees_follow_existing_contract(self):
        rng = np.random.default_rng(20261028)
        for index in range(24):
            tree = Evidence(
                {"path": [int(rng.integers(-5, 6)), bool(index % 2),
                          np.asarray(rng.normal(size=(2, 2)), dtype=np.float32), None],
                 index: ("stop", float(rng.normal()))},
                {"valid": True, "counter": np.int64(index)},
            )
            with self.subTest(tree=index):
                self.assert_same_verdict(tree, deepcopy(tree), expected="PASS_EXACT")
                changed = deepcopy(tree)
                changed.state["path"][2][0, 1] = np.nextafter(changed.state["path"][2][0, 1], np.float32(np.inf))
                self.assert_same_verdict(tree, changed, expected="FAIL_EXACT")


if __name__ == "__main__":
    unittest.main()
