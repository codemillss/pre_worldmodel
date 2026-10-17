"""The collector must distinguish absent checks from valid empty contacts."""
from copy import deepcopy
import json
import unittest

import numpy as np

from push_align.contact_geometry import check_contact_geometry


def contact(pair="pusher_object", distance=-0.0002, cosine=1.0):
    return {"pair": pair, "distance_m": distance, "normal_force_N": 0.01,
            "normal_world": [1.0, 0.0, 0.0], "normal_center_cosine": cosine}


class ContactGeometryContractTest(unittest.TestCase):
    def test_present_empty_contacts_are_valid_noncontact(self):
        result = check_contact_geometry([{"contacts": []}], 0.001)
        self.assertTrue(result["pass"])
        self.assertEqual(result["contact_count"], 0)
        self.assertEqual(result["max_penetration_m"], 0.0)
        self.assertEqual(result["reasons"], [])

    def test_shallow_side_contacts_and_budget_boundary_are_valid(self):
        result = check_contact_geometry([
            {"contacts": [contact(distance=-0.001)]},
            {"contacts": [contact("floor_object", -0.0001, -1.0)]},
        ], 0.001)
        self.assertTrue(result["pass"])
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["side_contact_count"], 1)
        self.assertEqual(result["max_penetration_m"], 0.001)

    def test_all_contacts_including_floor_are_checked_for_penetration(self):
        result = check_contact_geometry([{"contacts": [contact("floor_object", -0.00101)]}], 0.001)
        self.assertFalse(result["pass"])
        self.assertEqual(result["over_budget_penetration_contacts"], 1)
        self.assertIn("contact_penetration_exceeded", result["reasons"])

    def test_bad_side_directions_are_not_silently_valid(self):
        for cosine in (None, np.nan, np.inf, -1.0, 0.0):
            with self.subTest(cosine=cosine):
                result = check_contact_geometry([{"contacts": [contact(cosine=cosine)]}], 0.001)
                self.assertFalse(result["pass"])
                self.assertEqual(result["reversed_side_contact_normals"], 1)

    def test_every_sample_is_checked_including_interval_start(self):
        samples = [{"contacts": [contact(cosine=-1)]}, {"contacts": []}]
        self.assertFalse(check_contact_geometry(samples, 0.001)["pass"])

    def test_nonfinite_distance_force_or_normal_is_reported_without_repair(self):
        for field, value in (("distance_m", np.nan), ("distance_m", -np.inf),
                             ("normal_force_N", np.inf), ("normal_world", [1.0, np.nan, 0.0])):
            with self.subTest(field=field, value=value):
                item = contact()
                item[field] = value
                result = check_contact_geometry([{"contacts": [item]}], 0.001)
                self.assertFalse(result["pass"])
                self.assertEqual(result["nonfinite_contacts"], 1)
                json.dumps(result, allow_nan=False)
                if field == "distance_m":
                    self.assertIsNone(result["max_penetration_m"])
                    self.assertEqual(result["max_finite_penetration_m"], 0.0)
                elif field == "normal_world":
                    self.assertTrue(np.isnan(item[field][1]))
                else:
                    self.assertTrue(np.isinf(item[field]))

    def test_missing_or_malformed_sample_is_rejected(self):
        for samples in (None, [], {}, "samples", [None], [{}], [{"contacts": None}],
                        [{"contacts": {}}], [{"contacts": [None]}]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                check_contact_geometry(samples, 0.001)

    def test_missing_required_contact_fields_are_rejected(self):
        for field in ("pair", "distance_m", "normal_force_N", "normal_world", "normal_center_cosine"):
            item = contact()
            del item[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_contact_geometry([{"contacts": [item]}], 0.001)

    def test_none_and_wrong_scalar_or_vector_shapes_are_rejected(self):
        cases = (("pair", None), ("distance_m", None), ("normal_force_N", None),
                 ("distance_m", [0.0]), ("distance_m", True), ("distance_m", "0.0"),
                 ("normal_world", None), ("normal_world", [1.0, 0.0]),
                 ("normal_world", [[1.0, 0.0, 0.0]]), ("normal_world", [1.0, None, 0.0]),
                 ("normal_center_cosine", [1.0]))
        for field, value in cases:
            item = contact()
            item[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                check_contact_geometry([{"contacts": [item]}], 0.001)

    def test_invalid_budget_is_rejected(self):
        for budget in (None, True, -0.001, np.nan, np.inf, [0.001]):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                check_contact_geometry([{"contacts": []}], budget)

    def test_inputs_are_not_mutated(self):
        samples = [{"contacts": [contact()]}]
        original = deepcopy(samples)
        check_contact_geometry(samples, 0.001)
        self.assertEqual(samples, original)


if __name__ == "__main__":
    unittest.main()
