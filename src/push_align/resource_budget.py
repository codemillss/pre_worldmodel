"""Pure arithmetic and stop decisions for the P5 storage proposal.

No environment/model is instantiated and no final batch is authorized here.
The final driver must call these checks with live disk measurements and its
remaining work, while retaining incomplete prefixes on any stop.
"""
from __future__ import annotations

from fractions import Fraction
import math


def _count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name}: nonnegative integer required")
    return value


def estimate_storage(counts, units, allowances, *, multiplier="1.5"):
    factor = Fraction(str(multiplier))
    if factor < 1:
        raise ValueError("Safety multiplier must be >= 1")
    c = {key: _count(value, key) for key, value in counts.items()}
    u = {key: _count(value, key) for key, value in units.items()}
    a = {key: _count(value, key) for key, value in allowances.items()}
    measured = {
        "raw": c["raw_records"] * u["raw_fixed"] + c["raw_intervals"] * u["raw_interval"],
        "witness": (c["witness_records"] * u["witness_fixed"]
                    + c["witness_intervals"] * u["witness_interval"]
                    + c["witness_decisions"] * u["witness_decision"]),
        "plan_index": c["plan_archives"] * u["plan_index_fixed"]
                      + c["plan_records"] * u["plan_index_record"],
        "full_traces_and_inputs": c["full_traces"] * (u["full_trace_npz"]
                                  + u["full_trace_identity"] + u["snapshot_envelope"]),
        "G_captured_inputs": c["G_captures"] * u["snapshot_envelope"],
    }
    parts = {key: math.ceil(value * factor) for key, value in measured.items()}
    if set(parts) & set(a):
        raise ValueError("Allowance names collide with measured components")
    parts.update(a)
    return {"measured_components_before_multiplier": measured,
            "multiplier": str(factor), "parts_bytes": parts,
            "required_bytes_including_reserve": sum(parts.values())}


def capacity_decision(free_bytes, required_bytes):
    free = _count(free_bytes, "free_bytes")
    required = _count(required_bytes, "required_bytes")
    return {"status": "FIT_ESTIMATE_ONLY" if free >= required else "HOLD_CAPACITY",
            "free_bytes": free, "required_bytes": required,
            "headroom_bytes": free - required,
            "batch_authorized": False}


def stage_guard(*, free_bytes, remaining_required_bytes, next_write_upper_bytes,
                hard_reserve_bytes, observed_unit_bytes, allowed_unit_bytes,
                mismatch_used_bytes, next_mismatch_write_upper_bytes,
                mismatch_limit_bytes, bindings_match):
    values = locals().copy()
    for key in values:
        if key != "bindings_match":
            _count(values[key], key)
    if type(bindings_match) is not bool:
        raise ValueError("bindings_match must be bool")
    reasons = []
    if free_bytes < remaining_required_bytes:
        reasons.append("remaining_budget_exceeds_free")
    if free_bytes < hard_reserve_bytes + next_write_upper_bytes:
        reasons.append("next_write_crosses_hard_reserve")
    if observed_unit_bytes > allowed_unit_bytes:
        reasons.append("measured_unit_exceeds_allowance_reestimate")
    if mismatch_used_bytes + next_mismatch_write_upper_bytes > mismatch_limit_bytes:
        reasons.append("mismatch_allowance_exceeded")
    if not bindings_match:
        reasons.append("source_or_original_binding_mismatch")
    return {"status": "STOP_PRESERVE_PREFIX" if reasons else "CONTINUE_RESOURCE_CHECK_ONLY",
            "reasons": reasons, "batch_authorized": False}
