"""Pure resource accounting for the bounded, fresh N-T full-bank assay.

No physics, predictors, or experiment files are imported here. Maxima only
increase; finishing one repeat never erases the remaining repeats in its group.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


class ResourceStop(RuntimeError):
    """Stop new work while retaining already returned physical prefixes."""


def expected_trial_keys():
    return tuple((candidate, panel, repeat) for candidate in range(128)
                 for panel in ("A", "B") for repeat in range(3))


@dataclass(frozen=True)
class RemainingWork:
    trials: int
    intervals: int
    source_comparisons: int
    complete_probe_groups: int


def remaining_work(completed):
    completed = set(completed)
    expected = set(expected_trial_keys())
    if not completed <= expected:
        raise ValueError("Completed keys outside the prespecified Cartesian bank")
    pending = expected - completed
    return RemainingWork(len(pending), sum(5 if panel == "A" else 8 for _, panel, _ in pending),
                         sum(candidate < 16 for candidate, _, _ in pending),
                         sum(repeat == 0 for _, _, repeat in pending))


@dataclass
class ObservedRates:
    seconds_per_interval: float = 0.48116094996221365
    bytes_per_interval: float = 89074.4
    source_comparison_seconds: float = 1.155413333
    complete_probe_seconds: float = 4.1144065
    complete_probe_bytes: float = 706522.
    loop_seconds_per_trial: float = 0.005735933055555556

    def observe(self, *, intervals, baseline_seconds, baseline_bytes,
                source_seconds, complete_seconds, complete_bytes, loop_seconds):
        if intervals < 0:
            raise ValueError("Negative returned interval count")
        values = (baseline_seconds, baseline_bytes, source_seconds, complete_seconds,
                  complete_bytes, loop_seconds)
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("Invalid measured resource quantity")
        if intervals:
            self.seconds_per_interval = max(self.seconds_per_interval, baseline_seconds / intervals)
            self.bytes_per_interval = max(self.bytes_per_interval, baseline_bytes / intervals)
        self.source_comparison_seconds = max(self.source_comparison_seconds, source_seconds)
        self.complete_probe_seconds = max(self.complete_probe_seconds, complete_seconds)
        self.complete_probe_bytes = max(self.complete_probe_bytes, complete_bytes)
        self.loop_seconds_per_trial = max(self.loop_seconds_per_trial, loop_seconds)


@dataclass(frozen=True)
class ResourceLimits:
    prior_seconds: float = 662.5052578353789
    prior_bytes: int = 94034554
    wall_cap_seconds: float = 7200.
    storage_cap_bytes: int = 5 * 1024**3
    free_reserve_bytes: int = 2 * 1024**3
    safety_factor: float = 1.5
    fixed_future_seconds: float = 12.124927558461845
    fixed_future_bytes: int = 1988750
    report_reserve_seconds: float = 120.
    sealing_reserve_seconds: float = 60.
    sealing_reserve_bytes: int = 16 * 1024**2


class LiveBudget:
    def __init__(self, limits=None, rates=None):
        self.limits = ResourceLimits() if limits is None else limits
        self.rates = ObservedRates() if rates is None else rates

    def evaluate(self, completed, *, process_seconds, output_bytes, free_bytes, stage):
        """Whole remaining workflow, including partly completed groups."""
        if process_seconds < 0 or output_bytes < 0 or free_bytes < 0:
            raise ValueError("Negative live resource reading")
        work, rate, lim = remaining_work(completed), self.rates, self.limits
        seconds = {
            "interval_baseline": lim.safety_factor * work.intervals * rate.seconds_per_interval,
            "source_D2_comparisons": lim.safety_factor * work.source_comparisons * rate.source_comparison_seconds,
            "first_repeat_complete_probe": lim.safety_factor * work.complete_probe_groups * rate.complete_probe_seconds,
            "trial_loop": lim.safety_factor * work.trials * rate.loop_seconds_per_trial,
            "fixed_future": lim.fixed_future_seconds,
            "report_reserve": lim.report_reserve_seconds,
        }
        byte_components = {
            "interval_baseline": math.ceil(lim.safety_factor * work.intervals * rate.bytes_per_interval),
            "first_repeat_complete_probe": math.ceil(lim.safety_factor * work.complete_probe_groups * rate.complete_probe_bytes),
            "fixed_future": lim.fixed_future_bytes,
        }
        future_seconds, future_bytes = sum(seconds.values()), sum(byte_components.values())
        projected_seconds = lim.prior_seconds + process_seconds + future_seconds
        projected_bytes = lim.prior_bytes + output_bytes + future_bytes
        failures = []
        if projected_seconds > lim.wall_cap_seconds:
            failures.append("cumulative_wall_cap")
        if projected_bytes > lim.storage_cap_bytes:
            failures.append("cumulative_storage_cap")
        if free_bytes < lim.free_reserve_bytes + future_bytes:
            failures.append("free_space_reserve")
        return {"stage": stage, "decision": "STOP_RESOURCE" if failures else "PASS_RESOURCE", "failures": failures,
                "remaining": asdict(work), "observed_rates": asdict(rate), "process_seconds": process_seconds,
                "cumulative_spent_seconds": lim.prior_seconds + process_seconds,
                "output_bytes": output_bytes, "free_bytes": free_bytes,
                "future_seconds": seconds, "future_bytes": byte_components,
                "projected_cumulative_seconds": projected_seconds, "projected_cumulative_bytes": projected_bytes,
                "limits": asdict(lim)}

    def assert_step_window(self, *, process_seconds, output_bytes, free_bytes):
        """Called AFTER the caller records a returned step, BEFORE another step.

        This is a cooperative one-action-boundary guard, not an OS watchdog.
        Sealing/report reserves leave time to persist a just-returned prefix.
        """
        lim = self.limits
        needed_seconds = (lim.safety_factor * self.rates.seconds_per_interval
                          + lim.sealing_reserve_seconds + lim.report_reserve_seconds)
        if lim.prior_seconds + process_seconds + needed_seconds > lim.wall_cap_seconds:
            raise ResourceStop("Cumulative step deadline: seal prefix and stop new physical intervals")
        if lim.prior_bytes + output_bytes + lim.sealing_reserve_bytes > lim.storage_cap_bytes:
            raise ResourceStop("Cumulative step storage: seal prefix and stop new physical intervals")
        if free_bytes < lim.free_reserve_bytes + lim.sealing_reserve_bytes:
            raise ResourceStop("Step free-space reserve: seal prefix and stop new physical intervals")


class PrefixDeadlineRecorder:
    """Append first, then check; a guard exception cannot drop a returned step."""
    def __init__(self, prefix, check):
        self.prefix = prefix
        self.check = check

    def __call__(self, index, result, command):
        self.prefix.append(result)
        self.check()
