import unittest
from push_align.resource_budget import capacity_decision, estimate_storage, stage_guard


class ResourceBudgetTests(unittest.TestCase):
    def inputs(self):
        return dict(raw_records=2, raw_intervals=5, witness_records=3,
                    witness_intervals=7, witness_decisions=8, plan_archives=2,
                    plan_records=5, full_traces=2, G_captures=1), dict(
                        raw_fixed=10, raw_interval=20, witness_fixed=1,
                        witness_interval=2, witness_decision=3, plan_index_fixed=4,
                        plan_index_record=5, full_trace_npz=6,
                        full_trace_identity=7, snapshot_envelope=8)

    def guard(self, **overrides):
        values = dict(free_bytes=100, remaining_required_bytes=90,
                      next_write_upper_bytes=20, hard_reserve_bytes=30,
                      observed_unit_bytes=8, allowed_unit_bytes=10,
                      mismatch_used_bytes=5, next_mismatch_write_upper_bytes=0,
                      mismatch_limit_bytes=20,
                      bindings_match=True)
        values.update(overrides)
        return stage_guard(**values)

    def test_fixed_and_incremental_terms_and_upward_rounding(self):
        counts, units = self.inputs()
        out = estimate_storage(counts, units, {"reserve": 100})
        self.assertEqual(out["parts_bytes"], dict(raw=180, witness=62,
                         plan_index=50, full_traces_and_inputs=63,
                         G_captured_inputs=12, reserve=100))
        self.assertEqual(out["required_bytes_including_reserve"], 467)

    def test_shared_E1_is_not_implicitly_multiplied_by_models(self):
        counts, units = self.inputs()
        counts.update(raw_records=576, raw_intervals=2880)
        out = estimate_storage(counts, units, {})
        self.assertEqual(out["parts_bytes"]["raw"], 95040)

    def test_decision_without_action_is_not_omitted(self):
        counts, units = self.inputs()
        counts["witness_decisions"] += 1
        self.assertEqual(estimate_storage(counts, units, {})["parts_bytes"]["witness"], 66)

    def test_normal_fresh_does_not_silently_duplicate_original_raw(self):
        counts, units = self.inputs()
        out = estimate_storage(counts, units, {})
        counts["witness_records"] += 1
        self.assertEqual(estimate_storage(counts, units, {})["parts_bytes"]["raw"], out["parts_bytes"]["raw"])

    def test_equality_is_fit_but_never_batch_authorization(self):
        self.assertEqual(capacity_decision(10, 10)["status"], "FIT_ESTIMATE_ONLY")
        self.assertFalse(capacity_decision(10, 10)["batch_authorized"])
        self.assertEqual(capacity_decision(9, 10)["headroom_bytes"], -1)

    def test_remaining_whole_budget_must_fit(self):
        self.assertIn("remaining_budget_exceeds_free", self.guard(free_bytes=89)["reasons"])

    def test_next_write_must_leave_reserve(self):
        self.assertIn("next_write_crosses_hard_reserve", self.guard(next_write_upper_bytes=71)["reasons"])

    def test_observed_growth_stops_instead_of_relaxing(self):
        self.assertIn("measured_unit_exceeds_allowance_reestimate", self.guard(observed_unit_bytes=11)["reasons"])

    def test_mismatch_limit_is_not_permission_to_drop_case(self):
        self.assertEqual(self.guard(mismatch_used_bytes=21)["status"], "STOP_PRESERVE_PREFIX")

    def test_pending_mismatch_output_stops_before_exceeding_cap(self):
        self.assertEqual(self.guard(mismatch_used_bytes=20,
                         next_mismatch_write_upper_bytes=1)["status"], "STOP_PRESERVE_PREFIX")
        self.assertEqual(self.guard(mismatch_used_bytes=5,
                         next_mismatch_write_upper_bytes=15)["status"], "CONTINUE_RESOURCE_CHECK_ONLY")

    def test_original_bindings_must_match(self):
        self.assertIn("source_or_original_binding_mismatch", self.guard(bindings_match=False)["reasons"])

    def test_valid_resource_check_does_not_authorize_batch(self):
        self.assertFalse(self.guard()["batch_authorized"])
        self.assertEqual(self.guard()["status"], "CONTINUE_RESOURCE_CHECK_ONLY")

    def test_invalid_inputs_rejected(self):
        counts, units = self.inputs()
        for bad in (-1, True, 1.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                capacity_decision(bad, 10)
        with self.assertRaises(ValueError):
            estimate_storage(counts, units, {}, multiplier="0.9")
        with self.assertRaises(ValueError):
            estimate_storage(counts, units, {"raw": 2})


if __name__ == "__main__":
    unittest.main()
