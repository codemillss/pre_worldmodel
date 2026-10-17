"""Predeclared development collection with bounded observed-state feedback.

Families are split before modes or physical outcomes are observed. The four
short open-loop templates retain the pilot's meaning; two additional modes
run the already selected geometric controller. A stored perturbation vector
is added only to an eligible PUSH proposal, never to recovery or holding.
This module does not advance a simulator, consult contact labels, or decide
whether a physical transition is valid for learning.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math

import numpy as np

from .data import DataContractError, split_families
from .environment import InitialSpec
from .geometry_control import GeometryContext, GeometryController, PARAMETER_NAMES
from .planning import TaskSpec
from .task import pose_in_goal


MODES = (
    "noncontact", "central", "eccentric", "contact_transition",
    "geometry_feedback", "geometry_perturbed",
)
FEEDBACK_MODES = frozenset(MODES[-2:])


def _require(condition, message):
    if not condition:
        raise DataContractError(message)


def _number(value, name, *, zero=False):
    _require(not isinstance(value, (bool, np.bool_))
             and isinstance(value, (int, float, np.integer, np.floating))
             and math.isfinite(float(value))
             and (value >= 0 if zero else value > 0),
             f"{name} requires a finite {'nonnegative' if zero else 'positive'} scalar")
    return float(value)


def _array(value, shape, name):
    _require(isinstance(value, np.ndarray) and value.dtype == np.float64
             and value.shape == shape and bool(np.isfinite(value).all()),
             f"{name} requires finite float64 shape {shape}")
    return value


def _readonly(value):
    array = np.asarray(value, dtype=np.float64).copy()
    array.flags.writeable = False
    return array


def _settings(settings, environment_config):
    """Validate the declared v1 design, not any observed collection result."""
    _require(settings["purpose"] == "main-development"
             and settings["modes"] == list(MODES), "Unsupported main development collection")
    _require(type(settings["actions_per_episode"]) is int
             and settings["actions_per_episode"] == 16, "Open-loop templates require 16 actions")
    _require(type(settings["feedback_max_ticks"]) is int
             and settings["feedback_max_ticks"] == 128, "Feedback v1 requires a 128-tick cap")
    counts = settings["split_counts"]
    _require(set(counts) == {"train", "validation"}
             and all(type(value) is int for value in counts.values())
             and counts == {"train": 60, "validation": 30},
             "Main v1 requires 60 train and 30 validation independent families; no final test")
    fixed = {
        "initial_sampling": "independent_uniform_within_nominal_xy_yaw_ranges",
        "family_rule": "one independently generated reset per family; any future repeats/variants retain that family",
        "mode_assignment": "after_global_family_split_before_physics; sorted_ids_cycle_six_modes_within_each_split",
    }
    _require(all(settings[key] == value for key, value in fixed.items()),
             "Unsupported main family/mode generation policy")
    for key in ("initial_seed", "split_seed", "action_seed"):
        _require(type(settings[key]) is int and settings[key] >= 0,
                 "Nonnegative integer RNG seeds required")
    commands = settings["commands"]
    command_fixed = {
        "blocks": "four_steps_per_speed_draw; constant_world_commands; no feedback or learned controller",
        "eccentric_sign_rule": "alternate_within_each_split",
        "noncontact_phases": ["zero:2", "retreat:4", "lateral:4", "partial_return:4", "zero:2"],
        "contact_transition_phases": ["forward:5", "retreat:3", "reapproach:8"],
    }
    _require(all(commands[key] == value for key, value in command_fixed.items()),
             "Open-loop policy differs from declared templates")
    for key in ("forward_speed_min_m_s", "forward_speed_max_m_s", "retreat_speed_m_s",
                "noncontact_lateral_speed_m_s", "noncontact_return_speed_m_s",
                "central_tangent_abs_max_m_s", "eccentric_tangent_abs_max_m_s",
                "command_support_norm_max_m_s", "eccentric_offset_abs_min_m",
                "eccentric_offset_abs_max_m"):
        _number(commands[key], key)
    _require(0 < commands["forward_speed_min_m_s"] <= commands["forward_speed_max_m_s"]
             <= commands["command_support_norm_max_m_s"]
             <= environment_config["pusher"]["command_speed_norm_max_m_s"],
             "Main command support exceeds environment limits")
    _require(commands["command_support_norm_max_m_s"] == .04,
             "Main v1 collection command norm cap is 0.04 m/s")
    _require(commands["eccentric_offset_abs_min_m"] <= commands["eccentric_offset_abs_max_m"]
             < environment_config["object"]["size_full_xyz_m"][1]/2,
             "Eccentric reset offset is outside the rear face")
    perturbation = settings["perturbation"]
    _require(type(perturbation["block_steps"]) is int and perturbation["block_steps"] == 4,
             "Perturbations require four-step predeclared blocks")
    _require(_number(perturbation["world_xy_std_m_s"], "world_xy_std_m_s") == .003
             and perturbation["apply_mode"] == "PUSH_only"
             and perturbation["holding_noise_enabled"] is False
             and perturbation["frame"] == "world_xy"
             and perturbation["generation"] == (
                 "PCG64([action_seed, episode_index]); Gaussian blocks pregenerated before physics"),
             "Perturbation v1 requires world-XY std 0.003 m/s, PUSH-only, no holding noise")
    limits = settings["limits"]
    _require(limits["feedback_collection_enabled"] is True
             and limits["includes_p2_01_stress_fixture"] is False
             and limits["final_evaluation_allowed"] is False,
             "Main collection must be bounded development feedback with no stress/final test")
    initial = environment_config["initial_conditions"]
    for key in ("object_x_range_m", "object_y_range_m", "object_yaw_range_rad"):
        bounds = np.asarray(initial[key], np.float64)
        _require(bounds.shape == (2,) and bool(np.isfinite(bounds).all())
                 and bounds[1] > bounds[0], f"{key} requires ordered finite bounds")
    return commands


def build_main_collection_plan(settings, environment_config):
    """Return immutable-in-purpose JSON plans and a pre-physics family split.

    One independently sampled reset constitutes one family. Prefixing IDs
    with ``main`` prevents accidental lineage collisions with the P2 pilot.
    Stored vectors exist for all perturbed-mode ticks, including ticks where
    they will be ineligible; execution never advances an RNG conditionally.
    """
    try:
        return _build_main_collection_plan(settings, environment_config)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as error:
        raise DataContractError(f"Malformed main collection configuration: {error}") from error


def _build_main_collection_plan(settings, environment_config):
    commands = _settings(settings, environment_config)
    counts = settings["split_counts"]
    families = [f"main-family-{index:03d}" for index in range(sum(counts.values()))]
    assignments = split_families(families, counts, settings["split_seed"])
    rng = np.random.Generator(np.random.PCG64(settings["initial_seed"]))
    ranges = environment_config["initial_conditions"]
    positions = [{
        "object_x": float(rng.uniform(*ranges["object_x_range_m"])),
        "object_y": float(rng.uniform(*ranges["object_y_range_m"])),
        "object_yaw": float(rng.uniform(*ranges["object_yaw_range_rad"])),
    } for _ in families]
    mode_rank = {}
    for split in ("train", "validation"):
        members = sorted(family for family in families if assignments[family] == split)
        for rank, family in enumerate(members):
            mode_rank[family] = (MODES[rank % len(MODES)], rank // len(MODES))
    plans = []
    for index, family in enumerate(families):
        mode, rank = mode_rank[family]
        rng_key = [settings["action_seed"], index]
        action_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(rng_key)))
        position = positions[index]
        yaw = position["object_yaw"]
        direction = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
        tangent = np.array([-direction[1], direction[0]], dtype=np.float64)
        offset = ((1 if rank % 2 == 0 else -1)
                  * float(action_rng.uniform(commands["eccentric_offset_abs_min_m"],
                                             commands["eccentric_offset_abs_max_m"]))) \
            if mode == "eccentric" else 0.
        distance = (environment_config["object"]["size_full_xyz_m"][0]/2
                    + environment_config["pusher"]["radius_m"]
                    + environment_config["reset"]["pusher_surface_gap_m"])
        pusher = np.array([position["object_x"], position["object_y"]]) \
            - distance*direction + offset*tangent
        initial = InitialSpec(**position, pusher_x=float(pusher[0]), pusher_y=float(pusher[1]))
        actions, phases = [], []

        def append(normal, lateral, ticks, label):
            action = normal*direction + lateral*tangent
            magnitude = float(np.linalg.norm(action))
            if magnitude > commands["command_support_norm_max_m_s"]:
                action *= commands["command_support_norm_max_m_s"]/magnitude
            actions.extend([action.tolist() for _ in range(ticks)])
            phases.extend([label]*ticks)

        if mode == "noncontact":
            append(0, 0, 2, "zero")
            append(-commands["retreat_speed_m_s"], 0, 4, "retreat")
            append(0, (1 if rank % 2 == 0 else -1)
                   * commands["noncontact_lateral_speed_m_s"], 4, "lateral")
            append(commands["noncontact_return_speed_m_s"], 0, 4, "partial_return")
            append(0, 0, 2, "zero")
        elif mode == "contact_transition":
            speed = float(action_rng.uniform(commands["forward_speed_min_m_s"],
                                             commands["forward_speed_max_m_s"]))
            append(speed, 0, 5, "forward")
            append(-commands["retreat_speed_m_s"], 0, 3, "retreat")
            append(speed, 0, 8, "reapproach")
        elif mode in {"central", "eccentric"}:
            lateral_limit = commands["central_tangent_abs_max_m_s"] if mode == "central" \
                else commands["eccentric_tangent_abs_max_m_s"]
            for block in range(4):
                append(float(action_rng.uniform(commands["forward_speed_min_m_s"],
                                                commands["forward_speed_max_m_s"])),
                       float(action_rng.uniform(-lateral_limit, lateral_limit)),
                       4, f"forward_block_{block}")
        noise = []
        if mode == "geometry_perturbed":
            blocks = action_rng.normal(0., settings["perturbation"]["world_xy_std_m_s"],
                                       size=(settings["feedback_max_ticks"]//4, 2))
            noise = np.repeat(blocks, 4, axis=0).tolist()
        maximum = settings["feedback_max_ticks"] if mode in FEEDBACK_MODES \
            else settings["actions_per_episode"]
        _require(mode in FEEDBACK_MODES or len(actions) == maximum,
                 "Incorrect open-loop action template length")
        plans.append({
            "episode_id": f"main-episode-{index:03d}", "episode_index": index,
            "family_id": family, "split": assignments[family], "mode": mode,
            "family_definition": {"base_initial_spec": asdict(initial),
                                  "variant_rule": settings["family_rule"]},
            "initial_spec": asdict(initial), "pusher_tangent_offset_m": offset,
            "actions": actions, "phases": phases, "max_ticks": maximum,
            "action_rng_key": rng_key, "perturbation_vectors_m_s": noise,
        })
    return plans, assignments


@dataclass(frozen=True)
class MainCollectionContext:
    geometry_context: GeometryContext | None
    previous_applied_action: np.ndarray


@dataclass(frozen=True)
class MainCollectionDecision:
    action: np.ndarray
    context: MainCollectionContext
    diagnostics: dict


class MainCollectionPolicy:
    """Propose from fixed commands or current state; actual validity is external.

    The runner must replace ``previous_applied_action`` with the environment's
    returned command after executing a proposal. It is recorded for provenance
    and is not an additional G input. No noise is injected into an infeasible
    decision, non-PUSH recovery, or an observed pose inside the handoff region.
    """

    def __init__(self, config, settings, plan, selectedGoverride):
        try:
            _settings(settings, config)
            self.plan = deepcopy(plan)
            self.mode = self.plan["mode"]
            _require(self.mode in MODES, "Unknown collection mode")
            self.maximum = self.plan["max_ticks"]
            expected = settings["feedback_max_ticks"] if self.mode in FEEDBACK_MODES \
                else settings["actions_per_episode"]
            _require(type(self.maximum) is int and self.maximum == expected,
                     "Plan cap differs from declared collection settings")
            self.limit = float(settings["commands"]["command_support_norm_max_m_s"])
            self.physical_limit = _number(config["pusher"]["command_speed_norm_max_m_s"],
                                          "physical command norm limit")
            self.task = TaskSpec.from_config(config)
            self.controller = None
            self.actions = np.asarray(self.plan["actions"], dtype=np.float64)
            self.noise = np.asarray(self.plan["perturbation_vectors_m_s"], dtype=np.float64)
            self.phases = tuple(self.plan["phases"])
            if self.mode in FEEDBACK_MODES:
                _require(self.actions.size == 0 and not self.phases,
                         "Feedback actions must be generated from observed state")
                _require(isinstance(selectedGoverride, dict)
                         and not set(selectedGoverride)-set(PARAMETER_NAMES),
                         "Only registered geometric-controller parameters may be selected")
                copied_config = deepcopy(config)
                copied_config["geometry_baseline"].update(deepcopy(selectedGoverride))
                if "geometry_transition" in settings:
                    from .geometry_control_smooth import SmoothGeometryController
                    if settings["geometry_transition"].get("noise_weighting") != "same predeclared PUSH-only vector times continuous phase entry and pre-reassessment gain; zero in holding/free modes":
                        raise DataContractError("Unknown transition noise weighting rule")
                    transition_parameters = {k: v for k, v in settings["geometry_transition"].items() if k != "noise_weighting"}
                    self.controller = SmoothGeometryController(copied_config, transition_parameters)
                else:
                    self.controller = GeometryController(copied_config)
                _require(self.controller.parameters["speed_max_m_s"] <= self.limit,
                         "Selected G command limit exceeds collection support")
            else:
                _require(self.actions.shape == (self.maximum, 2)
                         and bool(np.isfinite(self.actions).all())
                         and bool((np.linalg.norm(self.actions, axis=1) <= self.limit+1e-15).all())
                         and len(self.phases) == self.maximum,
                         "Plan requires bounded finite actions and one phase per tick")
            if self.mode == "geometry_perturbed":
                _require(self.noise.shape == (self.maximum, 2)
                         and bool(np.isfinite(self.noise).all())
                         and bool((self.noise.reshape(-1, 4, 2)
                                   == self.noise[::4, None, :]).all()),
                         "Plan requires predeclared finite four-tick perturbation blocks")
            else:
                _require(self.noise.size == 0, "Only perturbed feedback may contain noise")
            self.actions.flags.writeable = False
            self.noise.flags.writeable = False
        except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as error:
            raise DataContractError(f"Malformed main collection policy: {error}") from error

    def initial_context(self):
        return MainCollectionContext(self.controller.initial_context() if self.controller else None,
                                     _readonly(np.zeros(2, dtype=np.float64)))

    def propose(self, state, context, elapsed_s, tick):
        _array(state, (10,), "state")
        _require(isinstance(context, MainCollectionContext), "MainCollectionContext required")
        previous = _array(context.previous_applied_action, (2,), "previous_applied_action")
        _require(float(np.linalg.norm(previous)) <= self.physical_limit+1e-15,
                 "Previous actual command exceeds physical limits")
        elapsed = _number(elapsed_s, "elapsed_s", zero=True)
        _require(type(tick) is int and 0 <= tick < self.maximum, "Tick is outside episode cap")
        diagnostics = {
            "collection_mode": self.mode, "tick": tick, "elapsed_s": elapsed,
            "feasible": True, "reason": "command", "holding": False,
            "previous_applied_action": previous.tolist(), "raw_contact_labels_used": False,
            "future_physics_queries": 0, "noise_applied": False,
            "predeclared_noise_m_s": [0., 0.], "saturated": False,
            "post_noise_norm_saturated": False, "applied_noise_delta_m_s": [0., 0.],
            "geometry_context_reset": False,
        }
        if self.mode == "geometry_perturbed":
            diagnostics["predeclared_noise_m_s"] = self.noise[tick].tolist()
        geometry = context.geometry_context
        if not self.controller:
            _require(geometry is None, "Open-loop context cannot contain G state")
            action = self.actions[tick].copy()
            diagnostics.update(controller="predeclared_open_loop", phase=self.phases[tick],
                               mode_before=None, mode_after=None, base_action_m_s=action.tolist())
        else:
            _require(isinstance(geometry, GeometryContext), "Feedback requires GeometryContext")
            # The existing controller validates the entire G context and time
            # even for holding, so a goal pose cannot hide a malformed history.
            proposal = self.controller.propose(state, self.task, geometry, elapsed)
            diagnostics.update(proposal.diagnostics)
            if pose_in_goal(state, self.task):
                # Holding is collection behaviour, not a new readiness decision.
                # The runner samples its duration and continues to the fixed cap.
                geometry = replace(self.controller.initial_context(), phase_started_s=elapsed,
                                   last_elapsed_s=elapsed)
                action = np.zeros(2, dtype=np.float64)
                diagnostics.update(controller="geometry_rear_face_v1", holding=True,
                                   reason="holding", mode_before=context.geometry_context.mode,
                                   mode_after="RETRACT", base_action_m_s=[0., 0.],
                                   geometry_context_reset=True, reassessment_count=0,
                                   saturation_count=0)
            else:
                geometry = proposal.context
                action = proposal.action.copy()
                diagnostics["base_action_m_s"] = action.tolist()
                if self.mode == "geometry_perturbed":
                    noise = self.noise[tick]
                    if diagnostics["feasible"] and geometry.mode == "PUSH":
                        noise_gain = float(proposal.diagnostics.get("noise_gain", 1.))
                        if not 0. <= noise_gain <= 1.:
                            raise DataContractError("Declared transition noise gain must be in [0,1]")
                        diagnostics["noise_gain"] = noise_gain
                        diagnostics["noise_eligible"] = True
                        diagnostics["nonzero_noise_applied"] = bool(noise_gain > 0 and np.any(noise != 0))
                        action += noise_gain * noise
                        magnitude = float(np.linalg.norm(action))
                        if magnitude > self.limit:
                            action *= self.limit/magnitude
                            diagnostics["saturated"] = True
                            diagnostics["post_noise_norm_saturated"] = True
                        diagnostics["noise_applied"] = True
                        diagnostics["applied_noise_delta_m_s"] = (
                            action-proposal.action).tolist()
        _array(action, (2,), "collection action")
        _require(float(np.linalg.norm(action)) <= self.limit+1e-15,
                 "Collection proposal exceeds declared support")
        new_context = MainCollectionContext(geometry, _readonly(previous))
        diagnostics["requested_action_m_s"] = action.tolist()
        return MainCollectionDecision(_readonly(action), new_context, diagnostics)
