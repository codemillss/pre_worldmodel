"""Recorded-contact checks for the declared planar pushing geometry.

The positive center-normal test is limited to the project's shallow exterior
cylinder--box side contacts. It is not a general normal-orientation test for
arbitrary concave bodies, embedded contacts, or other contact arrangements.
No state, force, or geometry is repaired by this diagnostic.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Real

import numpy as np


def _sequence(value, field):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a sequence")
    return value


def _scalar(value, field):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a real scalar")
    return float(value)


def check_contact_geometry(samples, penetration_budget_m) -> dict:
    """Inspect every recorded contact, including the action's initial sample.

    Missing samples/fields and malformed shapes raise ``ValueError``: an absent
    check must not look like a valid no-contact interval. Finite-schema contacts
    containing NaN/Inf instead yield ``pass=False`` so their raw records can be
    retained. A missing *value* (None) for a side contact's center cosine also
    fails its direction check, matching the original P1 diagnostic.

    A present empty ``contacts`` sequence is a valid noncontact sample. If any
    recorded distance is nonfinite, ``max_penetration_m`` is None; the separately
    reported finite maximum must not be mistaken for a maximum over all contacts.
    """
    budget = _scalar(penetration_budget_m, "penetration_budget_m")
    if not isfinite(budget) or budget < 0:
        raise ValueError("penetration_budget_m must be finite and nonnegative")
    samples = _sequence(samples, "samples")
    if not samples:
        raise ValueError("samples must contain at least one recorded sample")

    nonfinite = reversed_side = deep = contact_count = side_count = 0
    finite_penetration_max = 0.0
    all_distances_finite = True
    for sample_index, sample in enumerate(samples):
        location = f"samples[{sample_index}]"
        if not isinstance(sample, Mapping) or "contacts" not in sample:
            raise ValueError(f"{location} must contain contacts")
        contacts = _sequence(sample["contacts"], f"{location}.contacts")
        for contact_index, contact in enumerate(contacts):
            contact_location = f"{location}.contacts[{contact_index}]"
            if not isinstance(contact, Mapping):
                raise ValueError(f"{contact_location} must be a mapping")
            required = ("pair", "distance_m", "normal_force_N", "normal_world")
            missing = [field for field in required if field not in contact]
            if missing:
                raise ValueError(f"{contact_location} missing fields: {missing}")
            pair = contact["pair"]
            if not isinstance(pair, str) or not pair:
                raise ValueError(f"{contact_location}.pair must be a nonempty string")
            distance = _scalar(contact["distance_m"], f"{contact_location}.distance_m")
            force = _scalar(contact["normal_force_N"], f"{contact_location}.normal_force_N")
            try:
                normal = np.asarray(contact["normal_world"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{contact_location}.normal_world must have shape (3,)") from error
            if normal.shape != (3,) or normal.dtype.kind not in "iuf":
                raise ValueError(f"{contact_location}.normal_world must be numeric with shape (3,)")
            normal = normal.astype(np.float64)
            contact_count += 1
            if not (isfinite(distance) and isfinite(force) and np.isfinite(normal).all()):
                nonfinite += 1
            if isfinite(distance):
                finite_penetration_max = max(finite_penetration_max, max(0.0, -distance))
            else:
                all_distances_finite = False
            if -distance > budget:
                deep += 1
            if pair == "pusher_object":
                side_count += 1
                if "normal_center_cosine" not in contact:
                    raise ValueError(f"{contact_location} missing normal_center_cosine")
                cosine = contact["normal_center_cosine"]
                if cosine is None:
                    reversed_side += 1
                else:
                    cosine = _scalar(cosine, f"{contact_location}.normal_center_cosine")
                    if not isfinite(cosine) or cosine <= 0.0:
                        reversed_side += 1

    reasons = []
    if nonfinite:
        reasons.append("nonfinite_contact_geometry")
    if reversed_side:
        reasons.append("reversed_side_contact_normal")
    if deep:
        reasons.append("contact_penetration_exceeded")
    return {
        "pass": not reasons,
        "sample_count": len(samples),
        "contact_count": contact_count,
        "side_contact_count": side_count,
        "nonfinite_contacts": nonfinite,
        "reversed_side_contact_normals": reversed_side,
        "over_budget_penetration_contacts": deep,
        "max_penetration_m": finite_penetration_max if all_distances_finite else None,
        "max_finite_penetration_m": finite_penetration_max,
        "penetration_budget_m": budget,
        "reasons": reasons,
        "scope": "shallow exterior cylinder-box side contacts; all recorded contacts' penetration",
    }
