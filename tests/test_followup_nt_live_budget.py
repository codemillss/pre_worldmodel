"""Meaningful resource-contract tests; no predictor or physics execution."""
from dataclasses import replace
import unittest

from push_align.followup_nt_live_budget import (LiveBudget, ObservedRates, PrefixDeadlineRecorder,
    ResourceLimits, ResourceStop, expected_trial_keys, remaining_work)


class LiveBudgetTests(unittest.TestCase):
    def test_full_cartesian_counts(self):
        self.assertEqual(len(expected_trial_keys()), 768)
        work = remaining_work(())
        self.assertEqual((work.trials, work.intervals, work.source_comparisons, work.complete_probe_groups),
                         (768, 4992, 96, 256))

    def test_partial_group_keeps_other_repeats_and_source_checks(self):
        work = remaining_work({(0, "A", 0)})
        self.assertEqual((work.trials, work.intervals, work.source_comparisons, work.complete_probe_groups),
                         (767, 4987, 95, 255))
        work = remaining_work({(0, "A", 0), (0, "A", 1)})
        self.assertEqual((work.trials, work.intervals, work.source_comparisons, work.complete_probe_groups),
                         (766, 4982, 94, 255))

    def test_source_checks_only_existing_ids(self):
        completed = {key for key in expected_trial_keys() if key[0] < 16}
        self.assertEqual(remaining_work(completed).source_comparisons, 0)
        self.assertEqual(remaining_work(completed).complete_probe_groups, 224)

    def test_completed_work_retains_fixed_and_report_reserve(self):
        budget = LiveBudget()
        decision = budget.evaluate(expected_trial_keys(), process_seconds=100., output_bytes=1000,
                                   free_bytes=10 * 1024**3, stage="end")
        self.assertEqual(decision["decision"], "PASS_RESOURCE")
        self.assertEqual(decision["future_seconds"]["interval_baseline"], 0)
        self.assertEqual(decision["future_seconds"]["report_reserve"], 120.)
        self.assertGreater(decision["projected_cumulative_seconds"], 882.)

    def test_initial_scope_fits_reviewed_cap(self):
        decision = LiveBudget().evaluate((), process_seconds=0., output_bytes=0,
            free_bytes=10 * 1024**3, stage="entry")
        self.assertEqual(decision["decision"], "PASS_RESOURCE")
        self.assertAlmostEqual(decision["projected_cumulative_seconds"], 6150.482789542896)

    def test_stronger_full_run_audit_reserve_still_fits(self):
        budget = LiveBudget(replace(ResourceLimits(), report_reserve_seconds=600.))
        decision = budget.evaluate((), process_seconds=0., output_bytes=0,
            free_bytes=10 * 1024**3, stage="full_entry")
        self.assertEqual(decision["decision"], "PASS_RESOURCE")
        self.assertAlmostEqual(decision["projected_cumulative_seconds"], 6630.482789542896)
        self.assertEqual(decision["future_seconds"]["report_reserve"], 600.)

    def test_whole_future_rejected_even_if_next_trial_would_fit(self):
        decision = LiveBudget().evaluate((), process_seconds=1200., output_bytes=0,
            free_bytes=10 * 1024**3, stage="before_trial")
        self.assertEqual(decision["decision"], "STOP_RESOURCE")
        self.assertIn("cumulative_wall_cap", decision["failures"])

    def test_storage_and_reserve_are_independent(self):
        full = LiveBudget()
        decision = full.evaluate((), process_seconds=0., output_bytes=5 * 1024**3,
            free_bytes=10 * 1024**3, stage="entry")
        self.assertIn("cumulative_storage_cap", decision["failures"])
        decision = full.evaluate((), process_seconds=0., output_bytes=0,
            free_bytes=2 * 1024**3, stage="entry")
        self.assertIn("free_space_reserve", decision["failures"])

    def test_observed_rates_only_increase(self):
        rates = ObservedRates()
        floor = vars(rates).copy()
        rates.observe(intervals=5, baseline_seconds=.1, baseline_bytes=1,
                      source_seconds=.1, complete_seconds=.1, complete_bytes=1, loop_seconds=.0001)
        self.assertEqual(vars(rates), floor)
        rates.observe(intervals=5, baseline_seconds=10, baseline_bytes=1000000,
                      source_seconds=3, complete_seconds=8, complete_bytes=900000, loop_seconds=.2)
        self.assertEqual(rates.seconds_per_interval, 2.)
        self.assertEqual(rates.bytes_per_interval, 200000.)
        for key, value in vars(rates).items():
            self.assertGreaterEqual(value, floor[key])
        with self.assertRaises(ValueError):
            rates.observe(intervals=1, baseline_seconds=float("nan"), baseline_bytes=1,
                          source_seconds=0, complete_seconds=0, complete_bytes=0, loop_seconds=0)

    def test_prefix_survives_cooperative_deadline(self):
        budget, prefix = LiveBudget(), []
        marker = object()
        record = PrefixDeadlineRecorder(prefix, lambda: budget.assert_step_window(
            process_seconds=6400., output_bytes=0, free_bytes=10 * 1024**3))
        with self.assertRaises(ResourceStop):
            record(0, marker, None)
        self.assertEqual(prefix, [marker])

    def test_step_storage_reserve_stops_before_next_step(self):
        budget = LiveBudget()
        with self.assertRaises(ResourceStop):
            budget.assert_step_window(process_seconds=0, output_bytes=0,
                                      free_bytes=2 * 1024**3)

    def test_unrecognized_or_impossible_keys_rejected(self):
        with self.assertRaises(ValueError):
            remaining_work({(128, "A", 0)})
        self.assertEqual(remaining_work(expected_trial_keys()).trials, 0)


if __name__ == "__main__":
    unittest.main()
