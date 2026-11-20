"""Experimental paired exact comparison; not a production recording backend.

The frozen recorder is deliberately untouched.  This prototype emits the same
two typed SHA streams and checks values directly while traversing a matched
pair.  It never treats equal hashes as a replacement for direct equality, never
reuses cached digests, and never changes the number of caller comparisons.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import struct

import numpy as np

from . import final_driver_v2 as reference


_REFERENCE_EXACT = reference.exact_comparison
_COMPARISON = ("streamed typed hash plus direct recursive equality; dtype/shape/bytes, "
               "scalar bits, flags, schema and complete JSON diagnostics")
_CHUNK = 1024**2


def _frame_pair(left_hash, right_hash, marker, left_payload=b"", right_payload=b""):
    """Append identical framing protocol to two independent hash streams."""
    left_hash.update(marker + struct.pack("<Q", len(left_payload)))
    left_hash.update(left_payload)
    right_hash.update(marker + struct.pack("<Q", len(right_payload)))
    right_hash.update(right_payload)


def _fallback(left, right, left_hash, right_hash, diagnostic):
    # Structurally different nodes are uncommon in valid roundtrips.  Retain
    # complete hashing and the reference's independent direct comparison here.
    reference._hash_typed(left, left_hash, diagnostic=diagnostic)
    reference._hash_typed(right, right_hash, diagnostic=diagnostic)
    return reference._typed_equal(left, right, diagnostic=diagnostic)


def _pair(left, right, left_hash, right_hash, *, diagnostic=False):
    if diagnostic:
        left = reference._diagnostic_scalar(left)
        right = reference._diagnostic_scalar(right)
    left_type, right_type = type(left), type(right)

    # Most diagnostic leaves are exact builtin scalars.  Handle these before
    # repeatedly querying the dataclass/NumPy/container protocols at every leaf.
    if left_type is right_type:
        if left_type is float:
            a, b = struct.pack("=d", left), struct.pack("=d", right)
            _frame_pair(left_hash, right_hash, b"F", a, b)
            return a == b  # preserves NaN payloads and the sign of zero
        if left_type is int:
            _frame_pair(left_hash, right_hash, b"I", str(left).encode(), str(right).encode())
            return left == right
        if left_type is bool:
            _frame_pair(left_hash, right_hash, b"B", b"1" if left else b"0", b"1" if right else b"0")
            return left == right
        if left_type is str:
            _frame_pair(left_hash, right_hash, b"U", left.encode(), right.encode())
            return left == right
        if left is None:
            _frame_pair(left_hash, right_hash, b"0")
            return True

    if is_dataclass(left) or is_dataclass(right):
        if left_type is not right_type:
            return _fallback(left, right, left_hash, right_hash, diagnostic)
        name = left_type.__name__.encode()
        _frame_pair(left_hash, right_hash, b"D", name, name)
        equal = True
        for field in fields(left):
            key = field.name.encode()
            _frame_pair(left_hash, right_hash, b"K", key, key)
            child_equal = _pair(getattr(left, field.name), getattr(right, field.name),
                left_hash, right_hash, diagnostic=diagnostic or
                isinstance(left, reference.StepResult) and field.name == "diagnostics")
            equal = child_equal and equal
        _frame_pair(left_hash, right_hash, b"E")
        return equal

    if (isinstance(left, np.ndarray) or isinstance(right, np.ndarray)) and not diagnostic:
        if not (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                and left.dtype.str == right.dtype.str and left.shape == right.shape
                and not left.dtype.hasobject and not right.dtype.hasobject):
            return _fallback(left, right, left_hash, right_hash, diagnostic)
        dtype, shape = left.dtype.str.encode(), str(left.shape).encode()
        _frame_pair(left_hash, right_hash, b"A", dtype, dtype)
        _frame_pair(left_hash, right_hash, b"S", shape, shape)
        a, b = reference._bytes_view(left), reference._bytes_view(right)
        length = str(len(a)).encode()
        _frame_pair(left_hash, right_hash, b"L", length, length)
        equal = True
        for offset in range(0, len(a), _CHUNK):
            aa, bb = a[offset:offset+_CHUNK], b[offset:offset+_CHUNK]
            left_hash.update(aa)
            right_hash.update(bb)
            equal = (aa == bb) and equal
        return equal

    if isinstance(left, np.generic) or isinstance(right, np.generic):
        if not (isinstance(left, np.generic) and isinstance(right, np.generic)):
            return _fallback(left, right, left_hash, right_hash, diagnostic)
        _frame_pair(left_hash, right_hash, b"N")
        return _pair(np.asarray(left), np.asarray(right), left_hash, right_hash)

    if isinstance(left, dict) or isinstance(right, dict):
        if not (isinstance(left, dict) and isinstance(right, dict) and len(left) == len(right)):
            return _fallback(left, right, left_hash, right_hash, diagnostic)
        if any(type(key) not in (str, int) for key in left):
            raise TypeError("Only string/integer dictionary keys supported")
        if any(type(key) not in (str, int) for key in right):
            raise TypeError("Only string/integer dictionary keys supported")
        length = str(len(left)).encode()
        _frame_pair(left_hash, right_hash, b"M", length, length)
        a = sorted(left.items(), key=lambda item: str(item[0]))
        b = sorted(right.items(), key=lambda item: str(item[0]))
        equal = True
        for (ka, va), (kb, vb) in zip(a, b):
            key_equal = _pair(str(ka) if diagnostic else ka, str(kb) if diagnostic else kb,
                left_hash, right_hash, diagnostic=diagnostic)
            value_equal = _pair(va, vb, left_hash, right_hash, diagnostic=diagnostic)
            equal = key_equal and value_equal and equal
        _frame_pair(left_hash, right_hash, b"E")
        return equal

    sequence = (list, tuple, np.ndarray) if diagnostic else (list, tuple)
    if isinstance(left, sequence) or isinstance(right, sequence):
        if not (isinstance(left, sequence) and isinstance(right, sequence)
                and (diagnostic or left_type is right_type) and len(left) == len(right)):
            return _fallback(left, right, left_hash, right_hash, diagnostic)
        marker = b"L" if diagnostic or isinstance(left, list) else b"T"
        length = str(len(left)).encode()
        _frame_pair(left_hash, right_hash, marker, length, length)
        equal = True
        for a, b in zip(left, right):
            child_equal = _pair(a, b, left_hash, right_hash, diagnostic=diagnostic)
            equal = child_equal and equal
        _frame_pair(left_hash, right_hash, b"E")
        return equal

    return _fallback(left, right, left_hash, right_hash, diagnostic)


def exact_comparison(left, right):
    """Return the frozen comparator's receipt without modifying either input."""
    left_hash, right_hash = hashlib.sha256(), hashlib.sha256()
    try:
        direct_equal = _pair(left, right, left_hash, right_hash)
    except (TypeError, ValueError, AttributeError):
        # On unsupported/corrupt objects preserve reference rejection order and
        # error text (the original fully hashes the left before the right).
        return _REFERENCE_EXACT(left, right)
    a, b = left_hash.hexdigest(), right_hash.hexdigest()
    return {"status": "PASS_EXACT" if a == b and direct_equal else "FAIL_EXACT",
            "comparison": _COMPARISON, "left_sha256": a, "right_sha256": b}
