"""Small real episodes and isolated streaming/gap/yaw adversarial fixtures."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from push_align import InitialSpec, PlanarPushEnv
from push_align.data import (
    DataContractError, Episode, EpisodeRecorder, contract_hash, make_contract,
    write_dataset,
)
from push_align.preparation import prepare_dataset
from push_align.representation import encode, state_delta


def toy_episode(count=12, *, episode_index=0, split="train", invalid=()):
    """Schema-light mock; fake streaming replaces full physical validation."""
    state = np.zeros((count+1, 10), dtype=np.float64)
    state[:, 0] = np.arange(count+1)*.001
    state[:, 6] = state[:, 0]-.05
    valid = np.ones(count, dtype=bool); valid[list(invalid)] = False
    times = np.arange(count+1, dtype=np.float64)*.1
    diagnostics = []
    for i in range(count):
        samples = []
        for j in (i, i+1):
            contacts = [] if i%3 == 0 or (i%3 == 2 and j == i) else [
                {"pair": "pusher_object", "normal_force_N": .01}]
            samples.append({"object_velocity_world": [0.]*6, "tilt_rad": 0., "contacts": contacts})
        diagnostics.append({"samples": samples})
    return Episode({
        "state": state, "action_applied": np.full((count, 2), .01),
        "transition_valid": valid, "substep_offsets": np.arange(count+1, dtype=np.int64),
        "substep_states": state[1:].copy(), "sim_time": times,
        "substep_times": times[1:].copy(), "diagnostic_nonfinite_values": np.array([], dtype=np.float64),
    }, {
        "episode_id": f"toy{episode_index}", "episode_index": episode_index,
        "family_id": f"family{episode_index}", "split": split,
        "contract": {"active_contact_force_min_N": .001},
        "diagnostics": diagnostics, "diagnostic_nonfinite_paths": [],
    })


class ToyPreparationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/"fixture"
        self.path.mkdir()
        self.contract = {"active_contact_force_min_N": .001}
        self.manifest = {"dataset_id": "fixture", "contract": self.contract,
                         "contract_hash": contract_hash(self.contract)}
        self.write_manifest()

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self):
        (self.path/"manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def prepare(self, episodes, *, iterator=None, expected=None):
        with patch("push_align.preparation._validate_contract") as validate, \
             patch("push_align.preparation.iter_dataset", return_value=iterator or iter(episodes)):
            result = prepare_dataset(self.path, expected)
        validate.assert_called_once_with(self.contract if expected is None else expected)
        return result

    def test_compact_rows_keep_raw_values_original_identity_and_regime_codes(self):
        episodes = [toy_episode(12, episode_index=7),
                    toy_episode(12, episode_index=20, split="validation"),
                    toy_episode(12, episode_index=99, split="smoke-test")]
        before = [{key: value.copy() for key, value in e.arrays.items()} for e in episodes]
        result = self.prepare(episodes)
        self.assertEqual(result.metadata["dataset_id"], "fixture")
        self.assertEqual(result.metadata["valid_transition_count"], 36)
        self.assertEqual(result.metadata["data_contract_hash"], contract_hash(self.contract))
        self.assertEqual([e["episode_index"] for e in result.metadata["episodes"]], [7, 20, 99])
        np.testing.assert_array_equal(result.arrays["episode_index"], np.repeat([7, 20, 99], 12))
        np.testing.assert_array_equal(result.arrays["split_code"], np.repeat([0, 1, 2], 12))
        np.testing.assert_array_equal(result.arrays["transition_index"], np.tile(np.arange(12), 3))
        np.testing.assert_array_equal(result.arrays["regime_code"], np.tile([0, 1, 2], 12))
        for name in ("current_state", "next_state", "applied_action", "features", "deltas"):
            self.assertEqual(result.arrays[name].dtype, np.float64)
        for name in ("split_code", "regime_code"):
            self.assertEqual(result.arrays[name].dtype, np.int8)
        for name in ("episode_index", "transition_index"):
            self.assertEqual(result.arrays[name].dtype, np.int64)
        np.testing.assert_array_equal(result.arrays["current_state"][:12], episodes[0].arrays["state"][:-1])
        np.testing.assert_array_equal(result.arrays["next_state"][:12], episodes[0].arrays["state"][1:])
        np.testing.assert_array_equal(result.arrays["features"], encode(result.arrays["current_state"], result.arrays["applied_action"]))
        np.testing.assert_array_equal(result.arrays["deltas"], state_delta(result.arrays["current_state"], result.arrays["next_state"]))
        for original, copies in zip(episodes, before):
            for key, value in copies.items():
                np.testing.assert_array_equal(original.arrays[key], value)

    def test_windows_respect_original_gap_and_cannot_cross_episodes_or_splits(self):
        # Interior gaps are deliberately mocked; normal recorders stop on invalidity.
        episodes = [toy_episode(12, invalid=(5,)), toy_episode(12, episode_index=8, split="validation")]
        result = self.prepare(episodes)
        np.testing.assert_array_equal(result.arrays["transition_index"][:11], [0,1,2,3,4,6,7,8,9,10,11])
        self.assertEqual(result.windows[1].shape, (23, 1))
        self.assertEqual(result.windows[5].shape, (11, 5))
        self.assertEqual(result.windows[10].shape, (3, 10))
        for horizon, windows in result.windows.items():
            self.assertEqual(windows.dtype, np.int64)
            for window in windows:
                original = result.arrays["transition_index"][window]
                np.testing.assert_array_equal(np.diff(original), np.ones(horizon-1, dtype=np.int64))
                self.assertEqual(len(np.unique(result.arrays["episode_index"][window])), 1)
                self.assertEqual(len(np.unique(result.arrays["split_code"][window])), 1)

    def test_final_generator_failure_does_not_return_partial_preparation(self):
        exhausted = []
        def invalid_stream():
            yield toy_episode(1)
            exhausted.append(True)
            raise DataContractError("late global totals failure")
        with self.assertRaisesRegex(DataContractError, "late global totals"):
            self.prepare([], iterator=invalid_stream())
        self.assertEqual(exhausted, [True])

    def test_manifest_change_during_stream_is_rejected(self):
        def changing_stream():
            yield toy_episode(1)
            self.manifest["late_mutation"] = True
            self.write_manifest()
        with self.assertRaisesRegex(DataContractError, "Manifest changed"):
            self.prepare([], iterator=changing_stream())

    def test_empty_and_final_test_streams_are_rejected(self):
        with self.assertRaisesRegex(DataContractError, "at least one episode"):
            self.prepare([])
        with self.assertRaisesRegex(DataContractError, "development splits"):
            self.prepare([toy_episode(1, split="test")])

    def test_expected_contract_hash_mismatch_is_rejected_before_stream(self):
        with self.assertRaisesRegex(DataContractError, "expected contract/hash"):
            self.prepare([toy_episode(1)], expected={"different": True})

    def test_default_contract_is_actually_validated(self):
        # Deliberately malformed real validation, rather than the mock fixtures.
        with self.assertRaisesRegex(DataContractError, "Malformed contract"):
            prepare_dataset(self.path)

    def test_zero_valid_rows_keep_shape_and_excluded_count(self):
        result = self.prepare([toy_episode(2, invalid=(0, 1))])
        self.assertEqual(result.arrays["current_state"].shape, (0, 10))
        self.assertEqual(result.arrays["features"].shape, (0, 11))
        self.assertEqual(result.metadata["excluded_transition_count"], 2)
        for horizon in (1, 5, 10):
            self.assertEqual(result.windows[horizon].shape, (0, horizon))

    def test_microstep_yaw_winding_and_sampled_alias_uncertainty_are_rejected(self):
        winding = toy_episode(1)
        unwrapped = np.arange(1, 9)*math.pi/4
        yaws = (unwrapped+math.pi)%(2*math.pi)-math.pi
        winding.arrays["substep_offsets"] = np.array([0, 8], dtype=np.int64)
        winding.arrays["substep_states"] = np.zeros((8, 10)); winding.arrays["substep_states"][:, 2] = yaws
        winding.arrays["substep_times"] = np.arange(1, 9)*.1/8
        with self.assertRaisesRegex(DataContractError, "yaw excursion"):
            self.prepare([winding])
        alias = toy_episode(1)
        # Identical yaw endpoints hide 2*pi, while observed omega exposes risk.
        for sample in alias.metadata["diagnostics"][0]["samples"]:
            sample["object_velocity_world"][2] = 2*math.pi/.1
        with self.assertRaisesRegex(DataContractError, "aliasing uncertain"):
            self.prepare([alias])

    def test_yaw_endpoint_mismatch_and_exact_pi_are_rejected(self):
        mismatch = toy_episode(1)
        mismatch.arrays["substep_states"][0, 2] = .02
        with self.assertRaisesRegex(DataContractError, "endpoint delta"):
            self.prepare([mismatch])
        ambiguous = toy_episode(1)
        ambiguous.arrays["state"][1, 2] = -math.pi
        ambiguous.arrays["substep_states"][0, 2] = -math.pi
        with self.assertRaisesRegex(DataContractError, "Ambiguous pi"):
            self.prepare([ambiguous])

    def test_small_signed_microstep_yaw_crossing_is_accepted(self):
        crossing = toy_episode(1)
        crossing.arrays["state"][:, 2] = [math.pi-.02, -math.pi+.03]
        crossing.arrays["substep_states"][0, 2] = -math.pi+.03
        result = self.prepare([crossing])
        self.assertAlmostEqual(result.arrays["deltas"][0, 2], .05, places=14)
        self.assertIn("not a certified continuous-time bound", result.metadata["yaw_check"]["limitation"])


class RealPreparationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = PlanarPushEnv(); cls.contract = make_contract(env)
        def recorder(env, name, index):
            return EpisodeRecorder(env, episode_id=name, episode_index=index,
                family_id=f"family{index}", split="train",
                family_definition={"base_initial_spec": {"object_x": 0., "pusher_x": float(env.observe().state[6])}, "variant_rule": "fixed actions"},
                collection_policy={"name": "preparation-test"}, seed=20261007,
                contract=cls.contract)
        normal = recorder(env, "normal", 0)
        for action in ([.03, 0], [.03, 0], [0, 0]): normal.step(action)
        cls.normal = normal.finish()
        env.reset(InitialSpec(pusher_x=.191, pusher_y=.1))
        partial = recorder(env, "partial", 1)
        partial.step([0,0]); partial.step([0,0]); partial.step([.06,0])
        cls.partial = partial.finish()

    def test_real_small_dataset_default_contract_and_partial_exclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"small-real"
            write_dataset(path, [self.normal, self.partial])
            result = prepare_dataset(path)
            self.assertEqual(result.metadata["recorded_transition_count"], 6)
            self.assertEqual(result.metadata["valid_transition_count"], 5)
            self.assertEqual(result.metadata["excluded_transition_count"], 1)
            self.assertEqual(result.windows[1].shape, (5, 1))
            np.testing.assert_array_equal(result.arrays["transition_index"], [0,1,2,0,1])
            np.testing.assert_array_equal(result.arrays["episode_index"], [0,0,0,1,1])

    def test_real_late_manifest_count_failure_cannot_be_hidden(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"small-real"
            write_dataset(path, [self.normal])
            manifest_path = path/"manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["learning_transition_count"] += 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(DataContractError, "Manifest counts mismatch"):
                prepare_dataset(path, self.contract)


if __name__ == "__main__":
    unittest.main()
