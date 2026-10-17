"""Measured pilot coverage, with separate transition/window/event denominators.

Collection modes describe the prescribed policy. Regimes describe measured
force patterns over one action interval; mixed patterns can include chatter.
Overlapping windows and raw samples are not independent statistical units.
"""
from __future__ import annotations

import numpy as np

from .data import DataContractError, _diagnostics, validate_episode
from .environment import wrap_angle

REGIMES = ("noncontact", "sustained_contact", "transition")
RECONTACT_INACTIVE_MIN_S = 0.04
RECONTACT_ACTIVE_MIN_S = 0.01


def _active(sample, threshold):
    # Match the episode contract's strict threshold and designated geom pair.
    return any(contact["pair"] == "pusher_object" and
               contact["normal_force_N"] > threshold
               for contact in sample["contacts"])


def _regimes(diagnostics, threshold):
    result = []
    for diagnostic in diagnostics:
        samples = diagnostic["samples"]
        if not samples:
            raise DataContractError("Regime requires recorded raw samples")
        pattern = [_active(sample, threshold) for sample in samples]
        result.append("sustained_contact" if all(pattern) else
                      "transition" if any(pattern) else "noncontact")
    return result


def transition_regimes(episode):
    """Return T measured labels, restoring raw NaN/Inf before thresholding.

This diagnostic labels recorded intervals, including excluded intervals. The
caller uses episode validation and the learning mask before counting usable
data. Nonfinite forces cannot enter learning because geometry validation fails.
"""
    return _regimes(_diagnostics(episode),
                    episode.metadata["contract"]["active_contact_force_min_N"])


def _contact_events(diagnostics, threshold, tolerance, *,
                    inactive_min_s=RECONTACT_INACTIVE_MIN_S,
                    active_min_s=RECONTACT_ACTIVE_MIN_S):
    """Count threshold crossings once across shared action-boundary samples.

Qualification uses the first-to-last *observed* sample span of each contiguous
inactive/active run, without extending a run beyond its last observation.
Unknown/nonfinite force samples break the chain instead of creating crossings.
All recorded transitions participate, including an excluded terminal interval.
"""
    for value in (threshold, inactive_min_s, active_min_s):
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
            raise ValueError("Contact thresholds must be finite and positive")
    if isinstance(tolerance, (bool, np.bool_)) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Contact time tolerance must be finite and nonnegative")
    timeline = []
    unknown = 0
    for diagnostic in diagnostics:
        for sample in diagnostic["samples"]:
            time = float(sample["sim_time"])
            forces = [contact["normal_force_N"] for contact in sample["contacts"]
                      if contact["pair"] == "pusher_object"]
            active = _active(sample, threshold) if np.isfinite(forces).all() else None
            if timeline and abs(time-timeline[-1][0]) <= tolerance:
                if active != timeline[-1][1]:
                    raise DataContractError("Conflicting contact at shared boundary")
                continue
            if not np.isfinite(time) or (timeline and time <= timeline[-1][0]):
                raise DataContractError("Contact timeline must have finite increasing times")
            timeline.append((time, active))
            unknown += active is None

    entries, exits, runs = [], [], []
    current = None
    for time, active in timeline:
        if active is None:
            if current is not None:
                runs.append(current)
                current = None
            runs.append({"active": None, "first": time, "last": time})
        elif current is None:
            current = {"active": active, "first": time, "last": time}
        elif active == current["active"]:
            current["last"] = time
        else:
            (entries if active else exits).append(time)
            runs.append(current)
            current = {"active": active, "first": time, "last": time}
    if current is not None:
        runs.append(current)

    recontacts = []
    for previous, inactive, resumed in zip(runs, runs[1:], runs[2:]):
        if (previous["active"] is True and inactive["active"] is False
                and resumed["active"] is True):
            inactive_span = inactive["last"]-inactive["first"]
            active_span = resumed["last"]-resumed["first"]
            if (inactive_span+tolerance >= inactive_min_s and
                    active_span+tolerance >= active_min_s):
                recontacts.append({
                    "contact_loss_time_s": inactive["first"],
                    "contact_resume_time_s": resumed["first"],
                    "observed_inactive_span_s": float(inactive_span),
                    "observed_resumed_active_span_s": float(active_span),
                })
    return {
        "entry_times_s": entries, "exit_times_s": exits,
        "raw_crossing_count": len(entries)+len(exits),
        "qualified_recontacts": recontacts,
        "deduplicated_raw_sample_count": len(timeline),
        "unknown_force_sample_count": int(unknown),
        "inactive_min_s": float(inactive_min_s),
        "resumed_active_min_s": float(active_min_s),
        "duration_rule": "contiguous run last observed time minus first observed time",
        "scope": "all raw intervals, including excluded terminal intervals; threshold events, not controller decisions",
    }


def episode_coverage(episode, expected_contract, *, position_signal_m,
                     yaw_signal_rad, horizons=(1, 5, 10)):
    """Validate once, then summarize usable windows and measured motion.

Motion uses the initial boundary and boundaries incident to valid transitions.
An otherwise valid terminal observation of an excluded interval is not used to
claim a valid motion signal. Signals require at least one active valid interval.
Window labels use only their first transition, not the entire future window.
"""
    for value in (position_signal_m, yaw_signal_rad):
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
            raise ValueError("Motion signal thresholds must be finite and positive")
    horizons = tuple(horizons)
    if (not horizons or any(type(horizon) is not int or horizon <= 0 for horizon in horizons)
            or len(set(horizons)) != len(horizons)):
        raise ValueError("Horizons must be distinct positive integers")

    review = validate_episode(episode, expected_contract)
    arrays, metadata = episode.arrays, episode.metadata
    diagnostics = _diagnostics(episode)
    threshold = metadata["contract"]["active_contact_force_min_N"]
    regimes = _regimes(diagnostics, threshold)
    valid = arrays["transition_valid"]
    counts_all = {regime: regimes.count(regime) for regime in REGIMES}
    counts_valid = {regime: sum(bool(mask) and label == regime
                              for mask, label in zip(valid, regimes))
                    for regime in REGIMES}

    windows = {}
    for horizon in horizons:
        starts = [start for start in range(len(valid)-horizon+1)
                  if valid[start:start+horizon].all()]
        by_regime = {regime: [start for start in starts if regimes[start] == regime]
                     for regime in REGIMES}
        windows[str(horizon)] = {
            "total": len(starts), "by_first_regime": {key: len(value) for key, value in by_regime.items()},
            "start_indices": starts, "start_indices_by_first_regime": by_regime,
        }

    boundaries = sorted({0} | {index for start in np.flatnonzero(valid)
                              for index in (int(start), int(start)+1)})
    states = arrays["state"][boundaries]
    displacement = np.linalg.norm(states[:, :2]-arrays["state"][0, :2], axis=1)
    yaw_change = np.abs(wrap_angle(states[:, 2]-arrays["state"][0, 2]))
    max_position, max_yaw = float(displacement.max()), float(yaw_change.max())
    has_contact = any(bool(mask) and label != "noncontact"
                      for mask, label in zip(valid, regimes))
    evaluation = metadata["contract"]["resolved_config"]["evaluation"]
    events = _contact_events(
        diagnostics, threshold, metadata["contract"]["dt_tolerance_s"],
        inactive_min_s=evaluation["recontact_absence_min_s"],
        active_min_s=evaluation["recontact_presence_min_s"],
    )
    return {
        "episode_id": metadata["episode_id"], "family_id": metadata["family_id"],
        "split": metadata["split"], "mode": metadata["collection_policy"].get("kind"),
        "terminal_reason": metadata["terminal_reason"],
        "recorded_transitions": review["transitions"],
        "valid_transitions": review["learning_transitions"],
        "excluded_transitions": review["excluded_transitions"],
        "regime_counts_all": counts_all, "regime_counts_valid": counts_valid,
        "windows": windows, "valid_boundary_indices": boundaries,
        "max_position_excursion_m": max_position, "max_yaw_excursion_rad": max_yaw,
        "valid_contact_present": bool(has_contact),
        "translation_signal": bool(has_contact and max_position >= position_signal_m),
        "rotation_signal": bool(has_contact and max_yaw >= yaw_signal_rad),
        "position_signal_threshold_m": float(position_signal_m),
        "yaw_signal_threshold_rad": float(yaw_signal_rad),
        "contact_force_threshold_N": float(threshold),
        "contact_events": events,
        "qualified_recontact_count": len(events["qualified_recontacts"]),
        "scope": "pilot coverage; overlapping windows/raw samples are not independent units; no alignment or model-performance assessment",
    }
