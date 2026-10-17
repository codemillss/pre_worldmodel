"""Original-window budgets, branch isolation, and mid-epoch resume evidence."""
from copy import deepcopy
import hashlib
import io
import unittest

import torch

from push_align.fork_sampler import SamplerError, WindowSampler, sampler_state_hash


def sampler(seed=11, **kwargs):
    return WindowSampler(shuffle_generator=torch.Generator(device="cpu").manual_seed(seed), **kwargs)


def rehash(state):
    """Allow semantic-tampering tests to get past byte-integrity validation."""
    state["permutation_hash"] = hashlib.sha256(state["permutation"].numpy().tobytes()).hexdigest()
    state["shuffle_rng_hash"] = hashlib.sha256(state["shuffle_rng_state"].numpy().tobytes()).hexdigest()
    state["state_hash"] = sampler_state_hash(state)
    return state


class P4ForkSamplerTest(unittest.TestCase):
    def test_complete_epoch_has_original_windows_once_and_short_tail(self):
        item = sampler()
        batches = [item.next_batch() for _ in range(47)]
        self.assertEqual([len(batch) for batch in batches], [64] * 46 + [16])
        self.assertTrue(torch.equal(torch.sort(torch.cat(batches)).values, torch.arange(2960)))
        self.assertEqual((item.epoch, item.cursor, item.updates), (1, 2960, 47))
        item.next_batch()
        self.assertEqual((item.epoch, item.cursor, item.updates), (2, 64, 48))

    def test_formal_1000_update_budget_and_terminal_cursor(self):
        item = sampler()
        observed_windows = observed_transitions = 0
        for _ in range(1000):
            batch = item.next_batch()
            observed_windows += len(batch)
            observed_transitions += 5 * len(batch)
        self.assertEqual(observed_windows, 62992)
        self.assertEqual(observed_transitions, 314960)
        self.assertEqual((item.epoch, item.cursor, item.updates), (22, 832, 1000))
        self.assertEqual((item.window_occurrences, item.transition_occurrences), (observed_windows, observed_transitions))
        before = item.state_dict()["state_hash"]
        with self.assertRaises(StopIteration):
            item.next_batch()
        self.assertEqual(item.state_dict()["state_hash"], before)

    def test_common_generator_is_cloned_and_three_branches_match_every_batch(self):
        generator = torch.Generator(device="cpu").manual_seed(23)
        before = generator.get_state().clone()
        branches = [WindowSampler(shuffle_generator=generator) for _ in range(3)]
        self.assertTrue(torch.equal(before, generator.get_state()))
        for _ in range(1000):
            batches = [branch.next_batch() for branch in branches]
            self.assertTrue(torch.equal(batches[0], batches[1]))
            self.assertTrue(torch.equal(batches[0], batches[2]))
        self.assertEqual(len({branch.state_dict()["state_hash"] for branch in branches}), 1)
        self.assertTrue(torch.equal(before, generator.get_state()))

    def test_different_seeds_change_order_without_changing_windows(self):
        first, second = sampler(11), sampler(47)
        self.assertFalse(torch.equal(first.permutation, second.permutation))
        self.assertTrue(torch.equal(torch.sort(first.permutation).values, torch.sort(second.permutation).values))

    def test_sampling_does_not_consume_global_torch_rng(self):
        before = torch.get_rng_state().clone()
        item = sampler()
        for _ in range(48):
            item.next_batch()
        self.assertTrue(torch.equal(torch.get_rng_state(), before))

    def test_serialized_update_100_resume_keeps_next_batch_and_future_epochs(self):
        item = sampler()
        for _ in range(100):
            item.next_batch()
        self.assertEqual((item.epoch, item.cursor), (3, 384))
        buffer = io.BytesIO()
        torch.save(item.state_dict(), buffer)
        buffer.seek(0)
        restored = WindowSampler.from_state_dict(torch.load(buffer, weights_only=True))
        for _ in range(50):
            self.assertTrue(torch.equal(item.next_batch(), restored.next_batch()))
        self.assertEqual(item.state_dict()["state_hash"], restored.state_dict()["state_hash"])

    def test_last_batch_boundary_resume_creates_same_new_epoch(self):
        item = sampler()
        for _ in range(47):
            item.next_batch()
        restored = WindowSampler.from_state_dict(item.state_dict())
        self.assertTrue(torch.equal(item.next_batch(), restored.next_batch()))
        self.assertEqual(item.state_dict()["state_hash"], restored.state_dict()["state_hash"])

    def test_update_1000_one_update_probe_isolated_from_official_sampler(self):
        item = sampler()
        for _ in range(1000):
            item.next_batch()
        official = item.state_dict()
        restored = WindowSampler.from_state_dict(official)
        first, second = item.clone_for_resume_probe(), restored.clone_for_resume_probe()
        expected = official["permutation"][832:896]
        self.assertTrue(torch.equal(first.next_batch(), expected))
        self.assertTrue(torch.equal(second.next_batch(), expected))
        self.assertEqual(first.state_dict()["state_hash"], second.state_dict()["state_hash"])
        with self.assertRaises(StopIteration):
            first.next_batch()
        self.assertEqual(item.state_dict()["state_hash"], official["state_hash"])
        self.assertEqual(restored.state_dict()["state_hash"], official["state_hash"])
        self.assertEqual((item.max_updates, first.max_updates), (1000, 1001))

    def test_probe_at_update_100_permits_only_one_diagnostic_update(self):
        item = sampler()
        for _ in range(100):
            item.next_batch()
        probe = item.clone_for_resume_probe()
        self.assertTrue(torch.equal(probe.next_batch(), item.permutation[384:448]))
        with self.assertRaises(StopIteration):
            probe.next_batch()
        self.assertEqual((item.updates, item.max_updates), (100, 1000))

    def test_snapshot_batch_and_restore_inputs_are_independent_copies(self):
        item = sampler()
        exported = item.state_dict()
        restored = WindowSampler.from_state_dict(exported)
        expected = item.permutation[:64]
        exported["permutation"].fill_(0)
        exported["shuffle_rng_state"].fill_(0)
        public_permutation = item.permutation
        public_permutation.fill_(0)
        batch = item.next_batch()
        self.assertTrue(torch.equal(batch, expected))
        batch.fill_(0)
        self.assertTrue(torch.equal(restored.next_batch(), expected))
        self.assertEqual(item.state_dict()["state_hash"], restored.state_dict()["state_hash"])

    def test_constructor_rejects_invalid_contract_without_consuming_rng(self):
        generator = torch.Generator(device="cpu").manual_seed(11)
        before = generator.get_state().clone()
        for kwargs in ({"window_count": True}, {"window_count": 0}, {"batch_windows": 0}, {"max_updates": 0}, {"horizon": 0}, {"drop_last": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(SamplerError):
                WindowSampler(shuffle_generator=generator, **kwargs)
        with self.assertRaises(SamplerError):
            WindowSampler()
        self.assertTrue(torch.equal(before, generator.get_state()))

    def test_recomputed_hash_cannot_hide_duplicate_or_out_of_range_permutation(self):
        state = sampler().state_dict()
        for value in (int(state["permutation"][1]), 2960, -1):
            tampered = deepcopy(state)
            tampered["permutation"][0] = value
            with self.subTest(value=value), self.assertRaises(SamplerError):
                WindowSampler.from_state_dict(rehash(tampered))

    def test_counter_cursor_epoch_and_budget_tampering_rejected_after_rehash(self):
        item = sampler()
        for _ in range(100):
            item.next_batch()
        state = item.state_dict()
        alterations = (
            {"cursor": 385}, {"cursor": -1}, {"cursor": 2961}, {"epoch": 0},
            {"updates": 99}, {"window_occurrences": state["window_occurrences"] + 1},
            {"transition_occurrences": state["transition_occurrences"] + 1},
            {"max_updates": 99}, {"epoch": 4, "cursor": 0, "updates": 141,
             "window_occurrences": 8880, "transition_occurrences": 44400},
            {"updates": True}, {"drop_last": True},
        )
        for replacement in alterations:
            tampered = deepcopy(state)
            tampered.update(replacement)
            with self.subTest(replacement=replacement), self.assertRaises(SamplerError):
                WindowSampler.from_state_dict(rehash(tampered))

    def test_rng_dtype_length_and_invalid_internal_state_rejected(self):
        state = sampler().state_dict()
        for rng in (state["shuffle_rng_state"].to(torch.int64), state["shuffle_rng_state"][:-1], torch.zeros_like(state["shuffle_rng_state"])):
            tampered = deepcopy(state)
            tampered["shuffle_rng_state"] = rng
            with self.subTest(dtype=rng.dtype, length=len(rng)), self.assertRaises(SamplerError):
                WindowSampler.from_state_dict(rehash(tampered))

    def test_integrity_hash_failure_and_failed_load_leave_sampler_unchanged(self):
        item = sampler()
        original = item.state_dict()
        for field in ("permutation_hash", "shuffle_rng_hash", "state_hash"):
            tampered = deepcopy(original)
            tampered[field] = "0" * 64
            with self.subTest(field=field), self.assertRaises(SamplerError):
                item.load_state_dict(tampered)
            self.assertEqual(item.state_dict()["state_hash"], original["state_hash"])
        tampered = deepcopy(original)
        tampered["permutation"][[0, 1]] = tampered["permutation"][[1, 0]]
        with self.assertRaises(SamplerError):
            item.load_state_dict(tampered)
        self.assertEqual(item.state_dict()["state_hash"], original["state_hash"])

    def test_unknown_or_missing_snapshot_fields_rejected(self):
        original = sampler().state_dict()
        extra = deepcopy(original)
        extra["unknown"] = 1
        missing = deepcopy(original)
        del missing["cursor"]
        for state in (extra, missing):
            with self.assertRaises(SamplerError):
                WindowSampler.from_state_dict(state)

    def test_window_count_smaller_than_batch_size_is_single_short_epoch(self):
        item = sampler(window_count=7, batch_windows=64, max_updates=2)
        for epoch in (1, 2):
            batch = item.next_batch()
            self.assertEqual(len(batch), 7)
            self.assertTrue(torch.equal(torch.sort(batch).values, torch.arange(7)))
            self.assertEqual((item.epoch, item.cursor), (epoch, 7))
        self.assertEqual((item.updates, item.window_occurrences, item.transition_occurrences), (2, 14, 70))


if __name__ == "__main__":
    unittest.main()
