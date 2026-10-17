"""Resumable, branch-independent shuffle of the original P4 train windows.

The final short batch is an update, not discarded or filled by repeating rows.
Snapshots describe the current permutation as well as the future shuffle RNG;
an RNG state alone cannot resume an update in the middle of an epoch.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import torch


class SamplerError(ValueError):
    """Invalid or inconsistent original-window sampling state."""


_SCHEMA = "p4-window-sampler-v1"
_INT_FIELDS = (
    "window_count", "batch_windows", "max_updates", "horizon", "epoch",
    "cursor", "updates", "window_occurrences", "transition_occurrences",
)
_CORE_FIELDS = ("schema", "drop_last") + _INT_FIELDS
_TENSOR_FIELDS = ("permutation", "shuffle_rng_state")
_HASH_FIELDS = ("permutation_hash", "shuffle_rng_hash", "state_hash")


def _tensor_hash(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def sampler_state_hash(state):
    """Hash canonical scalar metadata and the actual permutation/RNG bytes.

    Existing hash fields are deliberately excluded; they are checked separately
    on load. This helper does not validate the semantic consistency of a state.
    """
    hasher = hashlib.sha256()
    scalars = {name: state[name] for name in _CORE_FIELDS}
    hasher.update(json.dumps(scalars, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    for name in _TENSOR_FIELDS:
        tensor = state[name]
        hasher.update(name.encode())
        hasher.update(str(tensor.dtype).encode())
        hasher.update(json.dumps(list(tensor.shape)).encode())
        hasher.update(tensor.contiguous().numpy().tobytes())
    return hasher.hexdigest()


def _validate_state(state):
    if not isinstance(state, dict) or set(state) != set(_CORE_FIELDS + _TENSOR_FIELDS + _HASH_FIELDS):
        raise SamplerError("Sampler snapshot fields do not match the registered schema")
    if state["schema"] != _SCHEMA or state["drop_last"] is not False:
        raise SamplerError("P4 sampler requires its registered schema and drop_last=False")
    if any(type(state[name]) is not int for name in _INT_FIELDS):
        raise SamplerError("Sampler counters and dimensions must be exact integers")
    count, size, budget, horizon = (state[name] for name in _INT_FIELDS[:4])
    if min(count, size, budget, horizon) <= 0:
        raise SamplerError("Sampler dimensions and update budget must be positive")
    permutation, rng = (state[name] for name in _TENSOR_FIELDS)
    if not isinstance(permutation, torch.Tensor) or permutation.device.type != "cpu" or permutation.dtype != torch.int64 or permutation.shape != (count,):
        raise SamplerError("Sampler permutation must be a CPU int64 vector of all original windows")
    if not torch.equal(torch.sort(permutation).values, torch.arange(count, dtype=torch.int64)):
        raise SamplerError("Sampler permutation must contain each original window exactly once")
    expected_rng_shape = torch.Generator(device="cpu").get_state().shape
    if not isinstance(rng, torch.Tensor) or rng.device.type != "cpu" or rng.dtype != torch.uint8 or rng.shape != expected_rng_shape:
        raise SamplerError("Sampler RNG state must match the CPU generator uint8 state shape")
    try:
        torch.Generator(device="cpu").set_state(rng.clone())
    except RuntimeError as error:
        raise SamplerError("Sampler RNG state cannot restore a CPU generator") from error
    epoch, cursor, updates, windows, transitions = (state[name] for name in _INT_FIELDS[4:])
    batches = (count + size - 1) // size
    valid_cursor = 0 <= cursor <= count and (cursor == count or cursor % size == 0)
    if epoch < 1 or not valid_cursor or (epoch > 1 and cursor == 0):
        raise SamplerError("Sampler epoch/cursor is not a returned-batch boundary")
    expected_updates = (epoch - 1) * batches + (cursor + size - 1) // size
    expected_windows = (epoch - 1) * count + cursor
    if updates != expected_updates or windows != expected_windows or transitions != horizon * windows:
        raise SamplerError("Sampler counters disagree with its epoch/cursor and original-window budget")
    if not 0 <= updates <= budget:
        raise SamplerError("Sampler update count exceeds its frozen additional-update budget")
    if state["permutation_hash"] != _tensor_hash(permutation) or state["shuffle_rng_hash"] != _tensor_hash(rng) or state["state_hash"] != sampler_state_hash(state):
        raise SamplerError("Sampler snapshot integrity hash mismatch")


class WindowSampler:
    """Shuffle every complete epoch, with exact mid-epoch checkpoint resumes.

    The caller's restored generator is cloned before the initial randperm. Each
    training branch therefore owns its future shuffle state even if the caller
    reuses the same generator object to construct the three branches.
    """

    def __init__(self, window_count=2960, batch_windows=64, shuffle_generator=None,
                 *, max_updates=1000, horizon=5, drop_last=False):
        dimensions = (window_count, batch_windows, max_updates, horizon)
        if any(type(value) is not int or value <= 0 for value in dimensions) or drop_last is not False:
            raise SamplerError("Sampler requires positive integer dimensions and drop_last=False")
        if not isinstance(shuffle_generator, torch.Generator) or shuffle_generator.device.type != "cpu":
            raise SamplerError("Sampler requires the restored common CPU shuffle generator")
        self.window_count = window_count
        self.batch_windows = batch_windows
        self.max_updates = max_updates
        self.horizon = horizon
        self.drop_last = False
        self._generator = torch.Generator(device="cpu")
        self._generator.set_state(shuffle_generator.get_state().clone())
        self._permutation = torch.randperm(window_count, generator=self._generator)
        self.epoch = 1
        self.cursor = self.updates = self.window_occurrences = self.transition_occurrences = 0

    @property
    def permutation(self):
        """An independent copy; callers cannot alter future training batches."""
        return self._permutation.clone()

    def next_batch(self):
        if self.updates >= self.max_updates:
            raise StopIteration("Frozen P4 additional-update budget is exhausted")
        if self.cursor == self.window_count:
            self._permutation = torch.randperm(self.window_count, generator=self._generator)
            self.epoch += 1
            self.cursor = 0
        stop = min(self.cursor + self.batch_windows, self.window_count)
        indices = self._permutation[self.cursor:stop].clone()
        self.cursor = stop
        self.updates += 1
        self.window_occurrences += len(indices)
        self.transition_occurrences += self.horizon * len(indices)
        return indices

    def state_dict(self):
        state = {"schema": _SCHEMA, "drop_last": self.drop_last}
        state.update({name: getattr(self, name) for name in _INT_FIELDS})
        state["permutation"] = self._permutation.clone()
        state["shuffle_rng_state"] = self._generator.get_state().clone()
        state["permutation_hash"] = _tensor_hash(state["permutation"])
        state["shuffle_rng_hash"] = _tensor_hash(state["shuffle_rng_state"])
        state["state_hash"] = sampler_state_hash(state)
        _validate_state(state)
        return state

    def load_state_dict(self, state):
        """Validate completely before replacing any current sampler state."""
        _validate_state(state)
        replacement = deepcopy(state)
        generator = torch.Generator(device="cpu")
        generator.set_state(replacement["shuffle_rng_state"])
        for name in _INT_FIELDS:
            setattr(self, name, replacement[name])
        self.drop_last = False
        self._generator = generator
        self._permutation = replacement["permutation"]

    @classmethod
    def from_state_dict(cls, state):
        sampler = cls.__new__(cls)
        sampler.load_state_dict(state)
        return sampler

    def clone_for_resume_probe(self):
        """Make a separate one-next-update probe, including at update 1000.

        This copy's budget ends exactly one update after its starting point.
        Its deliberately changed budget is rehashed; the official sampler and
        snapshot remain unchanged. It must never replace a formal branch.
        """
        state = self.state_dict()
        state["max_updates"] = self.updates + 1
        state["state_hash"] = sampler_state_hash(state)
        return type(self).from_state_dict(state)
