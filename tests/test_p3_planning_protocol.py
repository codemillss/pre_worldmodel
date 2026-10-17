"""Independent P3-05 keyed-randomness and genuinely shared G inputs.

Pure synthetic states exercise protocol wiring and meaningful counterexamples.
No robot simulation, learning update, original artifact write or performance
claim is made. Closed-loop states/warm starts are allowed to differ after
different actions; a frozen-state comparison must instead share all inputs.
"""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, replace
import math
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from push_align.candidates import generate_candidates
from push_align.config import load_pilot_config
from push_align.geometry_control import GeometryContext, PARAMETER_NAMES
from push_align.geometry_control_smooth import SmoothGeometryController
from push_align.planning import PlannerContext, TaskSpec, plan
from push_align.planning_protocol import (
    GeometryProposalAdapter, PlanningProtocolError, keyed_planner_rng,
    resolve_geometry_config,
)


SELECTED = {
    "position_gain_per_s": 4., "yaw_creep_speed_m_s": .02,
    "yaw_offset_gain_m_rad": .09, "speed_max_m_s": .03,
}
TRANSITION = {
    "schema_version": "geometry-transition-v2", "offset_taper_start_m": .002,
    "phase_end_taper_s": .4, "phase_entry_ramp_s": .4,
    "slide_blend_outer_tolerance_multiplier": 2.,
    "free_motion_speed_max_m_s": .01, "clock_snap_tolerance_s": 1e-9,
}


def observed_state(yaw=0., local_x=-.049, local_y=0.):
    theta = .5 * math.atan2(math.sin(2 * yaw), math.cos(2 * yaw))
    normal = np.array([math.cos(theta), math.sin(theta)], np.float64)
    tangent = np.array([-normal[1], normal[0]], np.float64)
    result = np.zeros(10, np.float64)
    result[2] = yaw
    result[6:8] = local_x * normal + local_y * tangent
    return result


def push_context(**changes):
    return replace(GeometryContext(mode="PUSH", locked_offset_m=0.,
                                   retreat_offset_m=0., phase_started_s=0.,
                                   last_elapsed_s=0.), **changes)


def numpy_global_equal(a, b):
    return (a[0] == b[0] and np.array_equal(a[1], b[1])
            and a[2:] == b[2:])


class KeyedPlannerRngTest(unittest.TestCase):
    def test_four_integer_key_matches_declared_PCG64_stream(self):
        actual = keyed_planner_rng(3711, 7, 2, 3)
        expected = np.random.Generator(np.random.PCG64(
            np.random.SeedSequence([3711, 7, 2, 3])))
        self.assertIs(type(actual.bit_generator), np.random.PCG64)
        np.testing.assert_array_equal(actual.bit_generator.random_raw(32),
                                      expected.bit_generator.random_raw(32))

    def test_same_physical_tick_is_independent_of_holding_or_prior_draw_count(self):
        reference = keyed_planner_rng(3711, 2, 19, 0).random(40)
        # One branch planned at earlier ticks; another held and skipped them.
        for tick in range(19):
            keyed_planner_rng(3711, 2, tick, 0).random(3 + tick * 11)
        keyed_planner_rng(3711, 2, 19, 0).random(999)
        np.testing.assert_array_equal(reference,
                                      keyed_planner_rng(3711, 2, 19, 0).random(40))

    def test_each_registered_key_has_a_distinct_stream(self):
        base = (3711, 2, 19, 0)
        reference = keyed_planner_rng(*base).random(40)
        for axis in range(4):
            changed = list(base)
            changed[axis] += 1
            with self.subTest(axis=axis):
                self.assertFalse(np.array_equal(reference,
                    keyed_planner_rng(*changed).random(40)))

    def test_model_or_training_seed_is_not_a_hidden_stream_argument(self):
        for name, value in (("model_id", "ridge"), ("training_seed", 47),
                            ("model_seed", 11)):
            with self.subTest(name=name), self.assertRaises(TypeError):
                keyed_planner_rng(3711, 2, 19, **{name: value})
        first, second = keyed_planner_rng(3711, 2, 19), keyed_planner_rng(3711, 2, 19, 0)
        np.testing.assert_array_equal(first.random(20), second.random(20))

    def test_malformed_key_is_rejected_rather_than_coerced(self):
        for axis in range(4):
            for value in (True, False, np.bool_(True), -1, .5, float("nan"),
                          float("inf"), "1", None, 1 + 0j):
                key = [3711, 2, 19, 0]
                key[axis] = value
                with self.subTest(axis=axis, value=value), self.assertRaises(PlanningProtocolError):
                    keyed_planner_rng(*key)

    def test_generators_are_independent_and_do_not_change_ambient_numpy_rng(self):
        before = deepcopy(np.random.get_state())
        a, b = keyed_planner_rng(3711, 0, 0), keyed_planner_rng(3711, 0, 0)
        a.random(600)
        np.testing.assert_array_equal(b.random(20),
                                      keyed_planner_rng(3711, 0, 0).random(20))
        self.assertTrue(numpy_global_equal(before, np.random.get_state()))


class SharedGeometryProtocolTest(unittest.TestCase):
    def setUp(self):
        self.config = load_pilot_config()
        self.selected, self.transition = deepcopy(SELECTED), deepcopy(TRANSITION)
        self.task = TaskSpec.from_config(self.config)
        self.adapter = GeometryProposalAdapter(self.config, self.selected, self.transition)

    def proposal(self, state=None, context=None, elapsed=.6, tick=6):
        return self.adapter.propose(observed_state() if state is None else state,
                                    self.task, push_context() if context is None else context,
                                    elapsed, control_tick=tick)

    def candidates(self, proposal, *, previous=None, tick=6):
        config = deepcopy(self.config)
        config["planner"].update(geometry_proposal_enabled=True,
            command_speed_norm_max_m_s=.03, candidate_count=16,
            horizon=2, block_lengths=[1, 1])
        return generate_candidates(config, keyed_planner_rng(3711, 2, tick),
            previous_actions=previous, geometry_action=proposal.geometry_action,
            tangent_direction=proposal.tangent_direction)

    def test_resolver_retains_all_sixteen_effective_parameters_and_physics(self):
        original = deepcopy(self.config)
        resolved = resolve_geometry_config(self.config, self.selected, self.transition)
        self.assertEqual(set(resolved["geometry_baseline"]), set(PARAMETER_NAMES))
        expected = {**original["geometry_baseline"], **SELECTED}
        self.assertEqual(resolved["geometry_baseline"], expected)
        self.assertEqual(len(expected), 16)
        for name in original:
            if name != "geometry_baseline":
                self.assertEqual(resolved[name], original[name])
        self.assertEqual(self.config, original)
        resolved["geometry_baseline"]["speed_max_m_s"] = .02
        resolved["physics"]["timestep_s"] = 123.
        self.assertEqual(self.config, original)

    def test_selected_patch_requires_exact_four_fields(self):
        for changed in ({k: v for k, v in SELECTED.items() if k != "speed_max_m_s"},
                        {**SELECTED, "position_gain": 4.}, {}):
            with self.subTest(changed=changed), self.assertRaises(PlanningProtocolError):
                resolve_geometry_config(self.config, changed, self.transition)

    def test_v1_or_unknown_transition_cannot_silently_replace_v2(self):
        v1 = {k: v for k, v in TRANSITION.items() if k != "clock_snap_tolerance_s"}
        v1["schema_version"] = "geometry-transition-v1"
        for value in (v1, {**TRANSITION, "unknown": 1.},
                      {**TRANSITION, "schema_version": "unknown"}):
            with self.subTest(value=value), self.assertRaises(PlanningProtocolError):
                GeometryProposalAdapter(self.config, self.selected, value)

    def test_bad_geometry_values_and_clock_tolerance_are_not_sanitized(self):
        for field, value in (("speed_max_m_s", True), ("yaw_creep_speed_m_s", float("nan")),
                             ("position_gain_per_s", 0.), ("speed_max_m_s", .07)):
            with self.subTest(field=field, value=value), self.assertRaises(PlanningProtocolError):
                GeometryProposalAdapter(self.config, {**SELECTED, field: value}, self.transition)
        for value in (True, 0., -1., float("inf"), 1e-6):
            with self.subTest(clock=value), self.assertRaises(PlanningProtocolError):
                GeometryProposalAdapter(self.config, self.selected,
                                        {**TRANSITION, "clock_snap_tolerance_s": value})

    def test_adapter_copies_config_and_exposes_full_selected_identity(self):
        before = self.proposal()
        self.assertEqual(self.adapter.parameters, {**self.config["geometry_baseline"], **SELECTED})
        self.assertEqual(self.adapter.transition_parameters, TRANSITION)
        self.config["geometry_baseline"]["speed_max_m_s"] = .06
        self.config["object"]["size_full_xyz_m"][0] = 9.
        self.selected["position_gain_per_s"] = 999.
        self.transition["phase_entry_ramp_s"] = .01
        after = self.proposal()
        np.testing.assert_array_equal(before.geometry_action, after.geometry_action)
        self.assertEqual(before.context, after.context)
        self.assertEqual(before.diagnostics, after.diagnostics)

    def test_full_gain_known_push_is_selected_point_zero_three(self):
        proposal = self.proposal()
        np.testing.assert_allclose(proposal.geometry_action, [.03, 0.], rtol=0, atol=1e-15)
        self.assertEqual(proposal.context.mode, "PUSH")
        self.assertEqual(proposal.control_tick, 6)
        self.assertEqual(proposal.geometry_action.dtype, np.dtype("float64"))
        self.assertFalse(proposal.geometry_action.flags.writeable)
        self.assertFalse(proposal.tangent_direction.flags.writeable)

    def test_public_config_accessors_cannot_mutate_future_proposals(self):
        original = self.proposal()
        parameters = self.adapter.parameters
        parameters["speed_max_m_s"] = .06
        transition = self.adapter.transition_parameters
        transition["clock_snap_tolerance_s"] = 123.
        resolved = self.adapter.resolved_config
        resolved["object"]["size_full_xyz_m"][0] = 9.
        resolved["geometry_baseline"]["position_gain_per_s"] = 123.
        repeated = self.proposal()
        np.testing.assert_array_equal(original.geometry_action, repeated.geometry_action)
        self.assertEqual(original.context, repeated.context)
        self.assertEqual(original.diagnostics, repeated.diagnostics)
        self.assertEqual(self.adapter.parameters["speed_max_m_s"], .03)
        self.assertEqual(self.adapter.transition_parameters["clock_snap_tolerance_s"], 1e-9)

    def test_adapter_is_identical_to_real_G_at_same_frozen_inputs(self):
        controller = SmoothGeometryController(
            {**deepcopy(self.config), "geometry_baseline":
             {**self.config["geometry_baseline"], **SELECTED}}, deepcopy(TRANSITION))
        value, context = observed_state(local_y=.0004), push_context()
        direct = controller.propose(value.copy(), self.task, context, .6)
        shared = self.proposal(value, context)
        np.testing.assert_array_equal(shared.geometry_action, direct.action)
        np.testing.assert_array_equal(shared.tangent_direction,
                                      np.asarray(direct.diagnostics["tangent_direction"], np.float64))
        self.assertEqual(shared.context, direct.context)

    def test_feasible_entry_zero_keeps_available_tangent_candidates(self):
        zero = self.proposal(context=self.adapter.initial_context(), elapsed=0., tick=0)
        np.testing.assert_array_equal(zero.geometry_action, np.zeros(2))
        self.assertIs(zero.diagnostics["feasible"], True)
        self.assertEqual(zero.diagnostics["geometry_action_status"], "available")
        batch = self.candidates(zero, tick=0)
        self.assertEqual(batch.diagnostics["geometry_proposal_status"], "available")
        np.testing.assert_array_equal(batch.actions[2:6], np.zeros((4, 2, 2)))
        np.testing.assert_allclose(batch.actions[6, 0], [.0, .01], rtol=0, atol=1e-15)
        np.testing.assert_allclose(batch.actions[7, 0], [.0, -.01], rtol=0, atol=1e-15)

    def test_infeasible_G_is_None_and_all_six_reserved_slots_are_zero(self):
        proposal = self.proposal(observed_state(local_x=.07))
        self.assertIsNone(proposal.geometry_action)
        self.assertIs(proposal.diagnostics["feasible"], False)
        self.assertEqual(proposal.diagnostics["geometry_action_status"], "unavailable_zero")
        batch = self.candidates(proposal)
        np.testing.assert_array_equal(batch.actions[2:8], np.zeros((6, 2, 2)))
        self.assertEqual(batch.diagnostics["geometry_proposal_status"], "unavailable_zero")
        self.assertFalse(batch.diagnostics["geometry_tangent_perturbation_applied"])
        # Counterexample: zero is a possible command, not an unavailable marker.
        config = deepcopy(self.config)
        config["planner"].update(geometry_proposal_enabled=True)
        wrong = generate_candidates(config, keyed_planner_rng(3711, 2, 6),
            geometry_action=np.zeros(2), tangent_direction=proposal.tangent_direction)
        self.assertTrue(np.any(wrong.actions[6:8] != 0))

    def test_tangent_uses_real_G_pi_symmetric_basis_not_raw_yaw(self):
        theta = .2
        value = observed_state(yaw=math.pi + theta)
        task = replace(self.task, goal=np.array([.08*math.cos(theta),
                                                .08*math.sin(theta), theta], np.float64))
        proposal = self.adapter.propose(value, task, push_context(), .6, control_tick=6)
        np.testing.assert_allclose(proposal.tangent_direction,
            [-math.sin(theta), math.cos(theta)], rtol=0, atol=1e-15)
        np.testing.assert_allclose(proposal.geometry_action,
            .03*np.array([math.cos(theta), math.sin(theta)]), rtol=0, atol=1e-15)
        raw_tangent = np.array([-math.sin(value[2]), math.cos(value[2])])
        self.assertLess(float(raw_tangent @ proposal.tangent_direction), -.999999)

    def test_time_and_true_G_history_matter_even_with_same_state(self):
        full = self.proposal()
        ramp = self.proposal(context=push_context(phase_started_s=.4, last_elapsed_s=.6))
        np.testing.assert_allclose(ramp.geometry_action, .5*full.geometry_action, rtol=0, atol=1e-15)
        reassess = self.proposal(context=push_context(locked_offset_m=.005))
        self.assertEqual(reassess.context.mode, "RETRACT")
        np.testing.assert_array_equal(reassess.geometry_action, np.zeros(2))

    def test_v2_clock_roundoff_does_not_create_different_G_phase(self):
        ctx = push_context(last_elapsed_s=1.9)
        left = self.proposal(context=ctx, elapsed=2.-5e-12, tick=20)
        right = self.proposal(context=ctx, elapsed=2.+5e-12, tick=20)
        self.assertEqual(left.context, right.context)
        self.assertEqual(left.context.mode, "RETRACT")
        np.testing.assert_array_equal(left.geometry_action, right.geometry_action)
        np.testing.assert_array_equal(left.geometry_action, np.zeros(2))

    def test_proposal_is_pure_and_frozen_context_can_be_reused(self):
        value, context = observed_state(), push_context()
        before_state, before_goal, before_context = value.copy(), self.task.goal.copy(), asdict(context)
        value.flags.writeable = False
        self.task.goal.flags.writeable = False
        first = self.proposal(value, context)
        second = self.proposal(value, context)
        np.testing.assert_array_equal(first.geometry_action, second.geometry_action)
        self.assertEqual(first.context, second.context)
        self.assertEqual(first.diagnostics, second.diagnostics)
        np.testing.assert_array_equal(value, before_state)
        np.testing.assert_array_equal(self.task.goal, before_goal)
        self.assertEqual(asdict(context), before_context)
        self.assertFalse(np.shares_memory(first.geometry_action, second.geometry_action))
        self.assertFalse(np.shares_memory(first.tangent_direction, second.tangent_direction))
        with self.assertRaises(FrozenInstanceError):
            first.control_tick = 77
        with self.assertRaises(ValueError):
            first.geometry_action[0] = 1.
        first.diagnostics["test_only_mutation"] = True
        self.assertNotIn("test_only_mutation", second.diagnostics)

    def test_commit_checks_actual_tick_without_implying_G_action_was_selected(self):
        proposal = self.proposal()
        self.assertEqual(self.adapter.commit(proposal, actual_control_tick=6), proposal.context)
        for tick in (5, 7, True, -1, .5):
            with self.subTest(tick=tick), self.assertRaises(PlanningProtocolError):
                self.adapter.commit(proposal, actual_control_tick=tick)
        np.testing.assert_allclose(self.proposal().geometry_action, [.03, 0.], rtol=0, atol=1e-15)

    def test_commit_rejects_unrelated_objects_and_malformed_proposal_identity(self):
        proposal = self.proposal()
        for value in (None, {}, proposal.context,
                      replace(proposal, context={}), replace(proposal, control_tick=True)):
            with self.subTest(value=value), self.assertRaises(PlanningProtocolError):
                self.adapter.commit(value, actual_control_tick=6)
        committed = self.adapter.commit(proposal, actual_control_tick=6)
        self.assertEqual(committed, proposal.context)
        self.assertIsNot(committed, proposal.context)

    def test_holding_zero_clears_G_history_without_claiming_readiness(self):
        value = observed_state()
        value[:3] = self.task.goal
        value[6:8] += self.task.goal[:2]
        held = self.proposal(value, push_context(locked_offset_m=.01), elapsed=.7, tick=7)
        np.testing.assert_array_equal(held.geometry_action, np.zeros(2))
        self.assertEqual(held.context.mode, "RETRACT")
        self.assertIsNone(held.context.locked_offset_m)
        self.assertIsNone(held.context.retreat_offset_m)
        self.assertEqual(self.adapter.commit(held, actual_control_tick=7), held.context)

    def test_malformed_state_time_context_or_tick_cannot_be_hidden_as_unavailable(self):
        bad_state = observed_state(); bad_state[0] = float("nan")
        for value, ctx, elapsed, tick in (
            (bad_state, push_context(), .6, 6),
            (observed_state().astype(np.float32), push_context(), .6, 6),
            (observed_state(), push_context(mode="UNKNOWN"), .6, 6),
            (observed_state(), push_context(last_elapsed_s=.7), .6, 6),
            (observed_state(), push_context(), float("nan"), 6),
            (observed_state(), push_context(), .6, True),
        ):
            with self.subTest(elapsed=elapsed, tick=tick), self.assertRaises(PlanningProtocolError):
                self.adapter.propose(value, self.task, ctx, elapsed, control_tick=tick)

    def test_G_proposal_has_no_future_engine_or_execution_rng_access(self):
        before = deepcopy(np.random.get_state())
        with ExitStack() as scope:
            for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse"):
                scope.enter_context(patch.object(mujoco, name,
                    side_effect=AssertionError("G may inspect current state only")))
            for name in ("default_rng", "random", "normal", "uniform"):
                scope.enter_context(patch.object(np.random, name,
                    side_effect=AssertionError("G may not create or advance an execution stream")))
            result = self.proposal()
        np.testing.assert_allclose(result.geometry_action, [.03, 0.], rtol=0, atol=1e-15)
        self.assertTrue(numpy_global_equal(before, np.random.get_state()))

    def test_frozen_context_and_key_give_identical_candidates_when_predictor_changes(self):
        config = deepcopy(self.config)
        config["planner"].update(geometry_proposal_enabled=True,
            command_speed_norm_max_m_s=.03, candidate_count=16,
            horizon=2, block_lengths=[1, 1])
        value = observed_state()
        context = PlannerContext(np.array([.005, -.002], np.float64),
                                 np.full((2, 2), .002, np.float64), 6)
        before = deepcopy(context)
        proposal = self.proposal(value)
        outputs = []
        for direction in (1., -1.):
            def predictor(state, action, sign=direction):
                delta = np.zeros_like(state)
                delta[:, 0] = sign * .1 * action[:, 0]
                delta[:, 6] = delta[:, 0]
                return delta
            outputs.append(plan(value, self.task, context,
                keyed_planner_rng(3711, 2, 6), predictor, config,
                geometry_action=proposal.geometry_action,
                tangent_direction=proposal.tangent_direction))
        np.testing.assert_array_equal(outputs[0].candidates.actions, outputs[1].candidates.actions)
        self.assertEqual(outputs[0].candidates.sources, outputs[1].candidates.sources)
        self.assertFalse(np.array_equal(outputs[0].predictions.raw_deltas, outputs[1].predictions.raw_deltas))
        self.assertNotEqual(outputs[0].selected_index, outputs[1].selected_index)
        np.testing.assert_array_equal(context.previous_applied_action, before.previous_applied_action)
        np.testing.assert_array_equal(context.optimal_actions, before.optimal_actions)
        self.assertEqual(context.control_step, before.control_step)

    def test_closed_loop_warm_changes_are_allowed_with_same_key_and_G_rule(self):
        proposal = self.proposal()
        positive = self.candidates(proposal, previous=np.full((2, 2), .002))
        negative = self.candidates(proposal, previous=np.full((2, 2), -.002))
        # The keyed base draws agree; different branch histories change warm proposals.
        np.testing.assert_array_equal(positive.actions[8:14], negative.actions[8:14])
        self.assertFalse(np.array_equal(positive.actions[1], negative.actions[1]))
        self.assertFalse(np.array_equal(positive.actions, negative.actions))


if __name__ == "__main__":
    unittest.main()
