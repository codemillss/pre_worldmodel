"""Independent saved-output cross-batch stop-loss reviewer.

Only immutable tensors and serialized predictions are read. No learned model,
forward, derivative, optimizer, training or physics is executed by this tool.
The frozen finite-step oracle supplies the unchanged numerical primitives.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import check_p4_finite_step as finite

ROOT = Path(__file__).resolve().parents[1]
TARGET = "D11_zero5"
SOURCES = ("historical_update1", TARGET, "D11_transition", "D11_noncontact")
ALPHAS = finite.ALPHAS
require, read, sha, arrays = finite.require, finite.read, finite.sha, finite.arrays
canonical, norm, dot = finite.canonical, finite.norm, finite.dot
parameter_arithmetic, loss_oracle = finite.parameter_arithmetic, finite.loss_oracle
validate_order, parameter_tree_hash = finite.validate_order, finite.parameter_tree_hash
mapping_hash, tensor_tree_hash = finite.mapping_hash, finite.tensor_tree_hash
canonical_losses, production_losses = finite.canonical_losses, finite.production_losses
classify_case, taylor_metrics = finite.classify_case, finite.taylor_metrics
REPRODUCTION_RTOL, REPRODUCTION_ATOL = finite.REPRODUCTION_RTOL, finite.REPRODUCTION_ATOL
METRIC_RTOL, METRIC_ATOL = finite.METRIC_RTOL, finite.METRIC_ATOL


def array_sha256(value):
    value = np.asarray(value)
    descriptor = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)},
                            sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(descriptor + b"\0" + np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def routing_bindings(snapshot, model_hash, order, source, direction, gradient, reference, states, actions, mask):
    """Reconstruct every identity from audited inputs, never from a wrapper."""
    require(source in SOURCES and isinstance(snapshot, str) and snapshot, "Registered source probe and snapshot required")
    common = {"snapshot_id": snapshot, "model_tree_hash": model_hash, "parameter_order": order}
    return {
        "source_binding": {**common, "probe_name": source, "objective": "total", "mode": "KEEP",
                           "direction_index": 3, "delta_sha256": array_sha256(direction)},
        "target_binding": {**common, "probe_name": TARGET, "batch_windows": len(states),
                           "states_sha256": array_sha256(states), "actions_sha256": array_sha256(actions),
                           "mask_sha256": array_sha256(mask)},
        "target_gradient_binding": {**common, "probe_name": TARGET, "component": "total",
                                    "gradient_sha256": array_sha256(gradient)},
        "target_loss_binding": {**common, "probe_name": TARGET, "loss_values": reference},
    }


def same_arrays(first, second):
    return set(first) == set(second) and all(first[key].dtype == second[key].dtype
        and first[key].shape == second[key].shape and np.array_equal(first[key], second[key], equal_nan=True)
        for key in first)


def routing_certificate(bindings, secondary_decision):
    source, target, gradient, loss = (bindings[name] for name in
        ("source_binding", "target_binding", "target_gradient_binding", "target_loss_binding"))
    return {"schema": "p4-cross-batch-routing-v1", "snapshot_id": source["snapshot_id"],
        "source_probe_name": source["probe_name"], "target_probe_name": TARGET,
        "is_self_stop_control": source["probe_name"] == TARGET, "model_tree_hash": source["model_tree_hash"],
        "parameter_order": source["parameter_order"],
        "source_direction": {"objective": "total", "mode": "KEEP", "direction_index": 3,
                             "delta_sha256": source["delta_sha256"], "norm_normalized": False},
        "target_gradient": {"snapshot_id": source["snapshot_id"], "probe_name": TARGET, "component": "total",
                            "gradient_sha256": gradient["gradient_sha256"]},
        "target_reference_losses": loss["loss_values"],
        "target_batch": {"batch_windows": target["batch_windows"],
                         "typed_sha256": {key: target[key + "_sha256"] for key in ("states", "actions", "mask")}},
        "binding_records": {"source": source, "target": target, "gradient": gradient, "loss": loss,
                            "reference_losses": loss["loss_values"], "parameter_order": source["parameter_order"]},
        "secondary_finite_pattern_decision": secondary_decision,
        "routing_checks": {key: True for key in ("same_snapshot", "same_actual_model", "same_parameter_order",
            "native_source_direction_unchanged", "target_gradient_and_reference_bound", "target_batch_unchanged",
            "binding_metadata_unchanged")}}


def stop_controls(evaluation, predictions, previous_evaluation, previous_predictions, is_self):
    """A failed scientific control holds interpretation without deleting data."""
    rows, previous = evaluation["evaluations"], previous_evaluation["evaluations"]
    require(len(rows) == len(previous) == 4, "All four control endpoints must remain")
    baseline = rows[0]
    baseline_ok = (baseline["status"] == previous[0]["status"] == "PASS"
        and baseline["loss_values"] == previous[0]["loss_values"]
        and same_arrays({key: value[0] for key, value in predictions.items()},
                        {key: value[0] for key, value in previous_predictions.items()}))
    same_losses = all(row["status"] == old["status"] and row["loss_values"] == old["loss_values"]
                      for row, old in zip(rows, previous)) if is_self else None
    same_predictions = same_arrays(predictions, previous_predictions) if is_self else None
    teacher_exact = (predictions["teacher_predictions"].dtype == previous_predictions["teacher_predictions"].dtype
        and np.array_equal(predictions["teacher_predictions"], previous_predictions["teacher_predictions"])) if is_self else None
    free_exact = (predictions["free_predicted_states"].dtype == previous_predictions["free_predicted_states"].dtype
        and np.array_equal(predictions["free_predicted_states"], previous_predictions["free_predicted_states"])) if is_self else None
    if is_self:
        same_predictions = teacher_exact and free_exact
    self_ok = (same_losses and same_predictions) if is_self else None
    return {"alpha0_target_baseline_reproduced": bool(baseline_ok), "is_self_stop": bool(is_self),
            "self_stop_all_four_loss_records_exact": same_losses,
            "self_stop_all_saved_predictions_exact": same_predictions,
            "self_stop_teacher_predictions_exact": teacher_exact, "self_stop_free_predictions_exact": free_exact,
            "self_stop_control_pass": bool(self_ok) if is_self else None}


def primary_counts(observations, controls_pass):
    """Retain every planned positive endpoint and apply case/global gating."""
    require(len(observations) == 28 and len({(row["snapshot_id"], row["source_probe_name"]) for row in observations}) == 28,
            "Exactly 28 distinct retained source/target pairs required")
    require(Counter(row["source_probe_name"] for row in observations) == Counter({name: 7 for name in SOURCES}),
            "Seven retained snapshots per source required")
    if not controls_pass:
        return None
    def counts(rows):
        result = {"case_count": len(rows), "planned_positive_endpoints": 3 * len(rows), "by_alpha": {}}
        for index, alpha in enumerate(ALPHAS[1:], 1):
            counter = Counter(row["change_relations"][index] if row["case_status"] == "PASS" else "NOT_INTERPRETED"
                              for row in rows)
            require(set(counter) <= {"INCREASE", "DECREASE", "NUMERIC_TIE", "NOT_INTERPRETED"}, "Registered target relation required")
            result["by_alpha"][str(alpha)] = {name: counter[name] for name in ("INCREASE", "DECREASE", "NUMERIC_TIE", "NOT_INTERPRETED")}
        return result
    return {"all": counts(observations), "self_stop": counts([row for row in observations if row["source_probe_name"] == TARGET]),
            "other_sources": counts([row for row in observations if row["source_probe_name"] != TARGET]),
            "by_source_probe": {name: counts([row for row in observations if row["source_probe_name"] == name]) for name in SOURCES}}


class Audit(finite.Audit):
    def __init__(self):
        super().__init__()
        self.controls = []
        self.snapshot_baselines = {}

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        require(path.is_relative_to(ROOT), "Input path outside project")
        relative, digest = path.relative_to(ROOT).as_posix(), sha(path)
        self.check("immutable:" + relative, expected is None or digest == expected)
        if relative in self.sources:
            self.check("repeat_immutable:" + relative, self.sources[relative] == digest)
        self.sources[relative] = digest
        return path

    def case_paths(self, case):
        snapshot, source = case["snapshot_id"], case["source_probe_name"]
        require(isinstance(snapshot, str) and "/" not in snapshot and source in SOURCES
                and case["target_probe_name"] == TARGET and case["source_direction_index"] == 3,
                "Fixed source/stop-target and TOTAL/KEEP index required")
        adam = f"runs/p4-adam-probe-20261015-01/probes/{snapshot}/{source}"
        target = f"runs/p4-gradient-probe-20261015-01/probes/{snapshot}/{TARGET}"
        previous_source = f"runs/p4-finite-step-20261015-01/cases/{snapshot}/{source}"
        previous_stop = f"runs/p4-finite-step-20261015-01/cases/{snapshot}/{TARGET}"
        expected = {"source_direction_npz": adam + "/directions.npz", "source_direction_summary": adam + "/summary.json",
            "target_gradient_summary": target + "/summary.json", "target_gradient_npz": target + "/gradients.npz",
            "previous_finite_source_summary": previous_source + "/summary.json",
            "previous_finite_source_parameters": previous_source + "/parameter_displacements.npz",
            "previous_finite_stop_summary": previous_stop + "/summary.json",
            "previous_finite_stop_parameters": previous_stop + "/parameter_displacements.npz",
            "previous_finite_stop_predictions": previous_stop + "/predictions.npz"}
        self.check(snapshot + ":" + source + ":exact_source_target_previous_case_paths",
                   all(case.get(key) == path for key, path in expected.items()))

    def previous_records(self, case, output, predictions, evaluation):
        label = case["snapshot_id"] + ":" + case["source_probe_name"]
        paths = {key: self.bind(ROOT / case[key], case[key + "_sha256"]) for key in (
            "previous_finite_source_summary", "previous_finite_source_parameters", "previous_finite_stop_summary",
            "previous_finite_stop_parameters", "previous_finite_stop_predictions")}
        source, stop = (read(paths[key]) for key in ("previous_finite_source_summary", "previous_finite_stop_summary"))
        self.check(label + ":previous_source_identity_and_parameter_binding", source["snapshot_id"] == case["snapshot_id"]
            and source["probe_name"] == case["source_probe_name"] and source["source"]["source_direction_npz"] == case["source_direction_npz"]
            and source["source"]["source_direction_index"] == 3 and source["parameter_order"] == evaluation["parameter_order"]
            and source["parameter_arrays_sha256"] == sha(paths["previous_finite_source_parameters"]))
        self.check(label + ":parameter_maps_bitwise_equal_previous_finite_source", same_arrays(output, arrays(paths["previous_finite_source_parameters"])))
        self.check(label + ":previous_target_stop_identity_and_output_binding", stop["snapshot_id"] == case["snapshot_id"]
            and stop["probe_name"] == TARGET and stop["parameter_order"] == evaluation["parameter_order"]
            and stop["parameter_arrays_sha256"] == sha(paths["previous_finite_stop_parameters"])
            and stop["prediction_arrays_sha256"] == sha(paths["previous_finite_stop_predictions"]))
        if case["source_probe_name"] == TARGET:
            self.check(label + ":self_stop_parameter_maps_exact", same_arrays(output, arrays(paths["previous_finite_stop_parameters"])))
        return {"evaluation": stop["evaluation"], "predictions": arrays(paths["previous_finite_stop_predictions"])}

    def case(self, run, case, spec, weights, optimizer_hash, probes, statistics):
        self.case_paths(case)
        label = case["snapshot_id"] + ":" + case["source_probe_name"]
        folder = run / "cases" / case["snapshot_id"] / case["source_probe_name"]
        record = read(folder / "summary.json")
        output = arrays(folder / "parameter_displacements.npz")
        predictions = arrays(folder / "predictions.npz")
        old_path = self.bind(ROOT / case["target_gradient_summary"], case["target_gradient_summary_sha256"])
        old = read(old_path)
        order = old["parameter_order"]
        validate_order(order)
        original = np.concatenate([weights[item["name"]].detach().numpy().reshape(-1) for item in order])
        self.check(label + ":original_weight_tree", parameter_tree_hash(original, order) == spec["model_tree_hash"])
        source_direction = self.bind(ROOT / case["source_direction_npz"], case["source_direction_npz_sha256"])
        direction_record = read(self.bind(ROOT / case["source_direction_summary"], case["source_direction_summary_sha256"]))
        self.check(label + ":source_target_checkpoint_coordinate_metadata", direction_record["snapshot_id"] == old["snapshot_id"] == case["snapshot_id"]
            and direction_record["probe_name"] == case["source_probe_name"] and old["probe_name"] == TARGET
            and direction_record["source_weights_tree_hash"] == old["before_weights_hash"] == spec["model_tree_hash"])
        self.check(label + ":same_TOTAL_KEEP_direction_coordinates", case["source_direction_index"] == 3
                   and direction_record["parameter_order"] == order and direction_record["directions"][3]["objective"] == "total"
                   and direction_record["directions"][3]["mode"] == "KEEP")
        fixed_direction = arrays(source_direction)["directions"][3]
        raw = arrays(self.bind(ROOT / case["target_gradient_npz"], case["target_gradient_npz_sha256"]))["gradients"][4]
        self.check(label + ":frozen_gradient_and_direction", fixed_direction.shape == raw.shape == (19338,)
                   and fixed_direction.dtype == raw.dtype == np.float32 and np.isfinite(fixed_direction).all() and np.isfinite(raw).all())
        expected_parameters = parameter_arithmetic(original, fixed_direction)
        self.check(label + ":parameter_array_keys", set(output) == {"alphas", "evaluated_parameters", "rounded_parameter_differences"})
        self.check(label + ":prediction_array_keys", set(predictions) == {"teacher_predictions", "free_predicted_states"})
        self.check(label + ":exact_alpha_coordinates", output["alphas"].dtype == np.float64
                   and np.array_equal(output["alphas"], np.asarray(ALPHAS, np.float64)))
        for key in ("evaluated_parameters", "rounded_parameter_differences"):
            self.check(label + ":exact_independent_parameter_arithmetic:" + key, output[key].dtype == np.float32
                       and output[key].shape == (4, 19338) and np.array_equal(output[key], expected_parameters[key], equal_nan=True))
        name = case["source_probe_name"]
        states, actions, mask = (probes[TARGET + "_" + key] for key in ("states", "actions", "mask"))
        batch = len(states)
        self.check(label + ":fixed_batch_contract", batch == 24 and states.dtype == actions.dtype == np.float64
                   and states.shape == (batch, 6, 10) and actions.shape == (batch, 5, 2)
                   and mask.dtype == np.bool_ and mask.shape == (batch, 5) and mask.all())
        self.check(label + ":prediction_array_shapes", predictions["teacher_predictions"].dtype == np.float32
                   and predictions["teacher_predictions"].shape == (4, 5 * batch, 10)
                   and predictions["free_predicted_states"].dtype == np.float64
                   and predictions["free_predicted_states"].shape == (4, batch, 6, 10))
        self.tree(label + ":case_identity", record, {"snapshot_id": case["snapshot_id"], "source_probe_name": name, "target_probe_name": TARGET,
            "source": case, "parameter_order": order, "source_weights_tree_hash": spec["model_tree_hash"],
            "source_optimizer_tree_hash": optimizer_hash, "parameter_arrays_sha256": sha(folder / "parameter_displacements.npz"),
            "prediction_arrays_sha256": sha(folder / "predictions.npz")})
        evaluation = record["evaluation"]
        expected_bindings = routing_bindings(case["snapshot_id"], spec["model_tree_hash"], order, name,
            fixed_direction, raw, old["loss_values"], states, actions, mask)
        routing = evaluation["cross_batch_routing"]
        self.tree(label + ":independent_source_target_gradient_reference_routing", routing,
                  routing_certificate(expected_bindings, evaluation["decision"]))
        self.check(label + ":exact_typed_input_binding_records", routing["binding_records"] ==
                   routing_certificate(expected_bindings, evaluation["decision"])["binding_records"])
        previous = self.previous_records(case, output, predictions, evaluation)
        denominators = {"batch_windows": batch, "valid_transitions": 5 * batch,
                        "base_scalar_terms": 50 * batch, "relative_scalar_terms": 10 * batch}
        self.tree(label + ":evaluation_contract", evaluation, {"schema": "p4-fixed-direction-finite-step-evaluation-v1",
            "alphas": list(ALPHAS), "parameter_order": order, "parameter_count": 19338,
            "raw_loss_order": ["one_step", "rollout", "relative_teacher", "relative_rollout", "total"],
            "denominators": denominators, "retained_alpha_count": 4, "case_denominator": 1,
            "source_state_hash_before": mapping_hash(original, order), "source_state_hash_after": mapping_hash(original, order)})
        preservation_names = {"source_state_unchanged", "source_parameter_identities_unchanged", "source_buffer_identities_unchanged",
            "source_gradient_buffers_unchanged", "source_modes_unchanged", "source_hooks_unchanged", "global_rng_unchanged",
            "batch_inputs_unchanged", "normalizer_unchanged", "direction_and_gradient_inputs_unchanged"}
        self.check(label + ":source_preservation_checks", set(evaluation["checks"]) == preservation_names
                   and all(value is True for value in evaluation["checks"].values()))
        rows = evaluation["evaluations"]
        self.check(label + ":all_four_rows_retained", len(rows) == 4)
        independent_losses, expected_trace, case_calls, case_rows, successful = [], [], 0, 0, 0
        for index, alpha in enumerate(ALPHAS):
            row = rows[index]
            row_label = label + ":alpha:" + str(alpha)
            self.tree(row_label + ":identity_and_denominators", row, {"index": index, "alpha": alpha, "denominators": denominators,
                "evaluated_parameters_hash": hashlib.sha256(output["evaluated_parameters"][index].tobytes()).hexdigest(),
                "rounded_difference_hash": hashlib.sha256(output["rounded_parameter_differences"][index].tobytes()).hexdigest()})
            self.check(row_label + ":recorded_status", row["status"] in ("PASS", "NUMERICAL_FAILURE"))
            calls = row["forward_calls"]
            self.check(row_label + ":forward_prefix_budget", type(calls) is int and 0 <= calls <= 6
                       and row["forward_rows"] == ([5 * batch] + [batch] * 5)[:calls]
                       and row["forward_total_rows"] == sum(row["forward_rows"]))
            if row["status"] == "PASS":
                self.check(row_label + ":successful_root_call_contract", calls == 6 and row["error"] is None)
                independent = loss_oracle(states, mask, predictions["teacher_predictions"][index],
                    predictions["free_predicted_states"][index], statistics["delta_mean"], statistics["delta_scale"])
                expected = production_losses(independent["losses"])
                self.check(row_label + ":raw_and_derived_loss_fields", set(row["loss_values"]) == set(expected))
                self.tree(row_label + ":independent_saved_prediction_losses", row["loss_values"], expected)
                independent_losses.append(independent["losses"])
                successful += 1
            else:
                self.check(row_label + ":failure_never_drops_or_replaces_loss", set(row["loss_values"]) ==
                    {"one_step", "rollout", "relative_teacher", "relative_rollout", "total", "base", "weighted_aux", "common_total"}
                    and all(value is None for value in row["loss_values"].values()) and isinstance(row["error"], dict)
                    and set(row["error"]) == {"type", "message"} and all(isinstance(value, str) and value for value in row["error"].values()))
                independent_losses.append(None)
            evaluated, rounded = output["evaluated_parameters"][index], output["rounded_parameter_differences"][index]
            finite_parameters = bool(np.isfinite(evaluated).all() and np.isfinite(rounded).all())
            endpoint = evaluated.astype(np.float64) - original.astype(np.float64)
            nominal = alpha * fixed_direction.astype(np.float64)
            displacement = {"nominal_scaled_norm": norm(nominal),
                "rounded_difference_norm": norm(rounded) if finite_parameters else None,
                "exact_endpoint_difference_norm": norm(endpoint) if finite_parameters else None,
                "rounded_minus_nominal_l2": norm(rounded.astype(np.float64) - nominal) if finite_parameters else None,
                "exact_minus_nominal_l2": norm(endpoint - nominal) if finite_parameters else None}
            taylor = {"nominal_directional_change": dot(raw, nominal),
                "rounded_directional_change": dot(raw, rounded) if finite_parameters else None,
                "exact_endpoint_directional_change": dot(raw, endpoint) if finite_parameters else None}
            self.tree(row_label + ":displacement_norms", row["displacement"], displacement)
            self.tree(row_label + ":three_distinct_directional_predictions", row["taylor"], taylor)
            evaluated_hash = parameter_tree_hash(evaluated, order)
            expected_trace += [{"rows": count, "parameter_tree_hash": evaluated_hash} for count in row["forward_rows"]]
            case_calls += calls; case_rows += row["forward_total_rows"]
        self.check(label + ":actual_source_forwards_used_evaluated_parameter_maps", record["source_forward_trace"] == expected_trace)
        verified_recorded_losses = [canonical_losses(row["loss_values"]) if row["status"] == "PASS" else None for row in rows]
        classification = classify_case(verified_recorded_losses, old["loss_values"])
        baseline_status = ("NUMERICAL_FAILURE" if independent_losses[0] is None else
                           "PASS" if classification["reproduction"]["pass"] else "REPRODUCTION_FAILURE")
        per_loss = {}
        for key in ("one_step", "rollout", "relative_teacher", "relative_rollout", "total"):
            expected = old["loss_values"][key]
            observed = rows[0]["loss_values"][key]
            error = None if observed is None else abs(observed - expected)
            allowed = REPRODUCTION_ATOL + REPRODUCTION_RTOL * abs(expected)
            per_loss[key] = {"actual": observed, "reference": expected, "absolute_error": error,
                             "allowed_error": allowed, "pass": bool(error is not None and error <= allowed)}
        self.tree(label + ":strict_alpha0_reproduction", evaluation["alpha0_reproduction"],
            {"status": baseline_status, "rtol": REPRODUCTION_RTOL, "atol": REPRODUCTION_ATOL, "per_loss": per_loss})
        self.number(label + ":frozen_numeric_tie_band", evaluation["neutral_band"], classification["numeric_tie_margin"])
        for index, row in enumerate(rows):
            available = independent_losses[index] is not None and independent_losses[0] is not None
            differences = {key: row["loss_values"][key] - rows[0]["loss_values"][key] if available else None
                           for key in ("base", "weighted_aux", "common_total")}
            self.tree(label + ":alpha:" + str(ALPHAS[index]) + ":actual_loss_changes", row["loss_differences"], differences)
            taylor = {key: None for key in ("primary_prediction", "endpoint_sensitivity_prediction", "nominal_prediction",
                                           "residual_actual_minus_primary", "residual_actual_minus_endpoint")}
            if available:
                metric = taylor_metrics(original, output["evaluated_parameters"][index], output["rounded_parameter_differences"][index],
                    fixed_direction, raw, ALPHAS[index], verified_recorded_losses[0]["total"], verified_recorded_losses[index]["total"])
                taylor = {"primary_prediction": metric["primary_taylor_loss"],
                    "endpoint_sensitivity_prediction": metric["endpoint_float64_taylor_loss"], "nominal_prediction": metric["nominal_taylor_loss"],
                    "residual_actual_minus_primary": metric["actual_minus_primary_taylor"],
                    "residual_actual_minus_endpoint": verified_recorded_losses[index]["total"] - metric["endpoint_float64_taylor_loss"]}
            self.tree(label + ":alpha:" + str(ALPHAS[index]) + ":Taylor_residuals", row["taylor"], taylor)
            relation = classification["change_relations"][index] if available and baseline_status == "PASS" else "NOT_INTERPRETED"
            self.check(label + ":alpha:" + str(ALPHAS[index]) + ":tie_aware_relation", row["total_change_relation"] == relation)
        decisions = {"NOMINAL_STEP_OVERSHOOT_SUPPORTED": "NOMINAL_INCREASE_SMALLER_DECREASE",
                     "ALL_POSITIVE_ALPHAS_DECREASE": "NO_ALPHA1_OVERSHOOT_OBSERVED_ON_GRID"}
        decision = decisions.get(classification["verdict"], classification["verdict"])
        status = "NUMERICAL_FAILURE" if successful < 4 else baseline_status
        self.check(label + ":independent_failure_priority_and_decision", evaluation["decision"] == decision and evaluation["status"] == status)
        self.tree(label + ":evaluation_counters", evaluation["counters"], {"evaluations": 4, "successful_evaluations": successful,
            "root_model_forward_calls": case_calls, "root_model_forward_rows": case_rows, "new_autograd_traversals": 0,
            "new_optimizer_steps": 0, "new_training_updates": 0, "new_physics_steps": 0})
        control = stop_controls(evaluation, predictions, previous["evaluation"], previous["predictions"], name == TARGET)
        current0 = {"losses": rows[0]["loss_values"],
                    **{key: value[0].copy() for key, value in predictions.items()}}
        first0 = self.snapshot_baselines.setdefault(case["snapshot_id"], current0)
        same0 = current0["losses"] == first0["losses"] and all(
            np.array_equal(current0[key], first0[key]) for key in predictions)
        control["alpha0_matches_snapshot_first_source"] = bool(same0)
        previous_self = None if name != TARGET else {"pass": control["self_stop_control_pass"], "checks": {
            "all_four_loss_values_and_statuses_exact": control["self_stop_all_four_loss_records_exact"],
            "teacher_predictions_exact": control["self_stop_teacher_predictions_exact"],
            "free_predictions_exact": control["self_stop_free_predictions_exact"]}}
        expected_controls = {"source_parameter_maps_previous_exact": True,
            "alpha0_matches_previous_stop": control["alpha0_target_baseline_reproduced"],
            "alpha0_matches_snapshot_first_source": bool(same0), "self_stop_previous": previous_self}
        self.check(label + ":independent_recorded_saved_controls", record["controls"] == expected_controls)
        self.controls.append({"snapshot_id": case["snapshot_id"], "source_probe_name": name, **control})
        self.forward_calls += case_calls; self.forward_rows += case_rows; self.successful_evaluations += successful
        self.observed.append({"snapshot_id": case["snapshot_id"], "source_probe_name": name, "target_probe_name": TARGET,
            "independent_decision": decision, "secondary_decision": decision, "case_status": status, "independent_reproduction_status": baseline_status,
            "retained_evaluation_count": 4, "successful_evaluation_count": successful,
            "numeric_tie_margin": classification["numeric_tie_margin"], "total_changes": classification["total_loss_changes"],
            "change_relations": classification["change_relations"], "losses_from_saved_predictions": independent_losses,
            "branch_operands": "Recorded scalar losses independently verified from saved teacher/free predictions under the fixed float64 metric tolerance",
            "controls": control,
            "limitations": "Target stop loss and target raw gradient are fixed; source direction retains its native norm. Primary Taylor uses registered rounded float32 subtraction; float64 endpoint difference remains a separate sensitivity. No model forward is replayed by the auditor."})
        return {"snapshot_id": case["snapshot_id"], "source_probe_name": name, "target_probe_name": TARGET,
                "summary_path": (folder / "summary.json").relative_to(run).as_posix(), "evaluation": evaluation, "controls": expected_controls}

    def run(self, run):
        import platform
        import torch
        protocol, manifest, summary = (read(run / filename) for filename in ("registration.json", "manifest.json", "summary.json"))
        self.check("completed_registered_finite_step_run", run == ROOT / "runs/p4-cross-batch-20261015-01"
                   and protocol["run_id"] == "p4-cross-batch-20261015-01" and protocol["run_directory"] == run.relative_to(ROOT).as_posix()
                   and manifest["schema"] == "p4-cross-batch-manifest-v1" and summary["schema"] == "p4-cross-batch-summary-v1"
                   and summary["execution_status"] == "PASS" and summary["analysis_status"] in ("DESCRIPTIVE_COMPLETE", "HOLD_CONTROL_FAILURE"))
        self.check("execution_registration_manifest_summary_binding", sha(run / "registration.json") ==
                   sha(ROOT / "configs/p4_cross_batch.v1.json") == manifest["protocol_sha256"] == summary["protocol_sha256"]
                   and sha(run / "manifest.json") == summary["manifest_sha256"])
        self.bind(ROOT / "configs/p4_cross_batch.v1.json", manifest["protocol_sha256"])
        for relative, expected in protocol["bindings_sha256"].items(): self.bind(ROOT / relative, expected)
        self.check("auditor_and_synthetic_tests_bound_before_execution", all(relative in protocol["bindings_sha256"]
                   for relative in ("tools/check_p4_cross_batch.py", "tests/test_p4_cross_batch_audit.py",
                       "src/push_align/cross_batch_probe.py", "scripts/p4_cross_batch_probe.py")))
        self.check("original_design_path", protocol["design_path"] == "configs/p4_cross_batch.design.v1.json")
        design = read(self.bind(ROOT / protocol["design_path"], protocol["design_sha256"]))
        self.check("same_original_28_cases_and_four_alphas", protocol["cases"] == design["cases"] and len(protocol["cases"]) == 28
                   and protocol["alphas"] == manifest["alphas"] == design["alphas"] == list(ALPHAS)
                   and design["case_count"] == 28 and design["evaluation_count"] == 112)
        self.check("fixed_original_reproduction_and_tie_policy", design["alpha0_reproduction"]["rtol"] == REPRODUCTION_RTOL
                   and design["alpha0_reproduction"]["atol"] == REPRODUCTION_ATOL
                   and design["neutral_margin"].startswith("1e-6*max(1,abs(target_total_at_alpha0))"))
        self.check("fixed_stop_target_and_native_source_design", design["source_probe_names"] == list(SOURCES)
                   and design["target_probe_name"] == TARGET and design["target_batch_windows"] == 24
                   and design["self_stop_cases"] == 7 and design["other_source_cases"] == 21
                   and design["primary_positive_endpoint_count"] == 84)
        previous_run = ROOT / "runs/p4-finite-step-20261015-01"
        previous_registration = read(previous_run / "registration.json")
        previous_index = read(previous_run / "artifact_hashes.json")["sha256"]
        self.check("original_finite_544_bindings_and_88_output_inventory",
                   len(previous_registration["bindings_sha256"]) == 544 and len(previous_index) == 88)
        for relative, expected in previous_registration["bindings_sha256"].items(): self.bind(ROOT / relative, expected)
        for relative, expected in previous_index.items(): self.bind(previous_run / relative, expected)
        self.bind(previous_run / "artifact_hashes.json")
        adam_run = ROOT / "runs/p4-adam-probe-20261015-01"
        original_bindings = read(adam_run / "registration.json")["bindings_sha256"]
        original_index = read(adam_run / "artifact_hashes.json")["sha256"]
        self.check("original_469_bindings_and_61_outputs_inventory", len(original_bindings) == 469 and len(original_index) == 61)
        for relative, expected in original_bindings.items(): self.bind(ROOT / relative, expected)
        for relative, expected in original_index.items(): self.bind(adam_run / relative, expected)
        self.bind(adam_run / "artifact_hashes.json")
        gradient_run = ROOT / "runs/p4-gradient-probe-20261015-01"
        gradient_manifest = read(self.bind(gradient_run / "manifest.json"))
        specs = protocol["snapshots"]
        expected_ids = ["common_initial"] + [f"{branch}_u{update:04d}" for branch in ("C0", "R1") for update in (100, 500, 1000)]
        self.check("same_seven_checkpoints", specs == manifest["model_snapshots"] == gradient_manifest["model_snapshots"]
                   and [spec["snapshot_id"] for spec in specs] == expected_ids)
        probe_names = ["historical_update1", "D11_zero5", "D11_transition", "D11_noncontact"]
        pairs = [(case["snapshot_id"], case["source_probe_name"]) for case in protocol["cases"]]
        self.check("unfiltered_fixed_case_order", pairs == [(snapshot, name) for snapshot in expected_ids for name in probe_names])
        expected_outputs = {"registration.json", "manifest.json", "aggregate.json", "summary.json"}
        expected_outputs |= {f"cases/{snapshot}/{name}/{filename}" for snapshot, name in pairs
                            for filename in ("summary.json", "parameter_displacements.npz", "predictions.npz")}
        index = read(run / "artifact_hashes.json")["sha256"]
        self.check("exact_88_indexed_outputs", set(index) == expected_outputs and len(index) == 88)
        for relative, expected in index.items():
            path = (run / relative).resolve()
            self.check("output_scope:" + relative, path.is_relative_to(run))
            self.bind(path, expected)
        runtime = protocol["runtime"]
        self.check("registered_runtime_versions", manifest["runtime"] == runtime and str(torch.__version__) == runtime["torch_version"]
                   and np.__version__ == runtime["numpy_version"] and platform.python_version() == runtime["python_version"])
        for path, expected in runtime["source_sha256"].items(): self.check("runtime_source:" + path, sha(Path(path)) == expected)
        self.bind(gradient_run / "artifact_hashes.json")
        self.check("same_frozen_data_indices", manifest["probe_batches_sha256"] == sha(gradient_run / "probe_batches.npz"))
        probes = arrays(self.bind(gradient_run / "probe_batches.npz", manifest["probe_batches_sha256"]))
        self.check("fixed_target_batch_and_typed_identities", manifest["source_probe_names"] == list(SOURCES)
            and manifest["target_probe_name"] == TARGET and manifest["target_batch_windows"] == 24
            and manifest["target_array_sha256"] == {key: array_sha256(probes[TARGET + "_" + key]) for key in ("states", "actions", "mask")})
        normalizer = ROOT / "runs/p3-02-20261009-01/learning_data/normalizer"
        normalizer_meta = read(self.bind(normalizer / "normalizer.json"))
        statistics = arrays(self.bind(normalizer / "statistics.npz", normalizer_meta["npz_sha256"]))
        self.check("original_normalizer_no_refit", normalizer_meta["normalizer"]["normalizer_hash"] == manifest["normalizer_hash"]
                   == gradient_manifest["normalizer_hash"])
        parent = read(self.bind(ROOT / "runs/p4-relative-loss-20261014-01/manifest.json"))
        self.check("same_physical_config", manifest["physical_config"] == parent["physical_config"] == gradient_manifest["physical_config"])
        records, tensor_checks = [], []
        for spec in specs:
            checkpoint = ROOT / spec["checkpoint_path"]
            metadata = read(self.bind(checkpoint / "checkpoint.json", spec["checkpoint_json_sha256"]))
            self.check(spec["snapshot_id"] + ":checkpoint_metadata", metadata["checkpoint_hash"] == spec["checkpoint_hash"]
                and metadata["checkpoint_hash"] == hashlib.sha256(canonical({key: value for key, value in metadata.items() if key != "checkpoint_hash"})).hexdigest())
            for filename in ("weights.pt", "training_state.pt"):
                self.bind(checkpoint / filename, spec["files_sha256"][filename])
                self.check(spec["snapshot_id"] + ":checkpoint_file:" + filename, metadata["files"][filename] == spec["files_sha256"][filename])
            weights = torch.load(checkpoint / "weights.pt", map_location="cpu", weights_only=True)
            optimizer = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)["optimizer"]
            self.check(spec["snapshot_id"] + ":saved_weight_precision", all(torch.is_tensor(value) and value.dtype == torch.float32
                       and value.device.type == "cpu" and not value.requires_grad and bool(torch.isfinite(value).all()) for value in weights.values()))
            model_hash, optimizer_hash = tensor_tree_hash(weights), tensor_tree_hash(optimizer)
            self.check(spec["snapshot_id"] + ":exact_source_model_tree", model_hash == spec["model_tree_hash"])
            self.check(spec["snapshot_id"] + ":saved_Adam_step_preserved", all(float(value["step"]) == 1300 + spec["additional_updates"]
                       for value in optimizer["state"].values()))
            for case in [case for case in protocol["cases"] if case["snapshot_id"] == spec["snapshot_id"]]:
                records.append(self.case(run, case, spec, weights, optimizer_hash, probes, statistics))
            tensor_checks.append({"snapshot_id": spec["snapshot_id"], "weights_unchanged": True, "optimizer_unchanged": True,
                                  "weight_tree_hash": model_hash, "optimizer_tree_hash": optimizer_hash})
        aggregate = read(run / "aggregate.json")
        self.check("all28_aggregate_exact_aliases", aggregate["schema"] == "p4-cross-batch-aggregate-v1"
                   and aggregate["row_count"] == len(records) == 28 and aggregate["rows"] == records)
        self.tree("all7_source_tensor_preservation_records", summary["source_tensor_checks"], tensor_checks)
        self.tree("complete_28case_112attempt_denominators", summary["counters"], {"snapshots": 7, "cases": 28,
            "attempted_evaluations": 112, "completed_evaluations": self.successful_evaluations,
            "forward_calls": self.forward_calls, "forward_rows": self.forward_rows})
        self.check("bounded_forward_execution", self.forward_calls <= 672 and self.forward_rows <= 26880)
        if self.successful_evaluations == 112:
            self.check("all112_successful_exact_registered_forward_execution", self.forward_calls == 672 and self.forward_rows == 26880)
        forbidden_names = {"physics." + name for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse")}
        forbidden_names |= {"Adam_instances", "Optimizer_instances", "Adam.step", "Optimizer.step", "autograd.grad", "Tensor.backward", "torch.save"}
        self.check("no_forbidden_training_gradient_optimizer_checkpoint_or_physics", set(summary["forbidden_execution_counts"]) == forbidden_names
                   and all(type(value) is int and value == 0 for value in summary["forbidden_execution_counts"].values()))
        self.tree("finite_loss_evaluation_scope", summary, {"global_rng_unchanged": True, "new_derivative_traversals": 0,
            "new_optimizer_updates": 0, "new_physics_steps": 0, "new_checkpoint_writes": 0, "new_data": False,
            "normalizer_refit": False, "final_test_used": False, "validation_or_development_probes_used": False,
            "historical_trajectory_replayed": False, "model_selection": "NONE", "learning_rate_adoption": "NONE", "alignment_improvement": "NOT_EVALUATED",
            "personal_understanding": "NOT_ASSESSED", "P5_01": "HOLD"})
        controls_pass = all(row["alpha0_target_baseline_reproduced"] and row["alpha0_matches_snapshot_first_source"]
                            for row in self.controls) and all(row["self_stop_control_pass"] for row in self.controls if row["is_self_stop"])
        self.check("all28_baseline_and_7_self_controls_retained", len(self.controls) == 28
                   and sum(row["is_self_stop"] for row in self.controls) == 7)
        self.check("global_saved_control_gate", summary["analysis_status"] ==
                   ("DESCRIPTIVE_COMPLETE" if controls_pass else "HOLD_CONTROL_FAILURE"))
        independent_primary = primary_counts(self.observed, controls_pass)
        global_checks = {"seven_snapshots_four_unique_sources": len({row["snapshot_id"] for row in self.controls}) == 7
                and all({row["source_probe_name"] for row in self.controls if row["snapshot_id"] == snapshot} == set(SOURCES)
                        for snapshot in expected_ids),
            "seven_self_stop_cases": sum(row["is_self_stop"] for row in self.controls) == 7,
            "all_self_stop_previous_reproduced": all(row["self_stop_control_pass"] for row in self.controls if row["is_self_stop"]),
            "all28_alpha0_previous_stop_reproduced": len(self.controls) == 28 and all(row["alpha0_target_baseline_reproduced"] for row in self.controls),
            "all_sources_alpha0_same_within_snapshot": all(row["alpha0_matches_snapshot_first_source"] for row in self.controls)}
        self.tree("independent_global_saved_control_records", summary["global_controls"], {"pass": controls_pass,
            "checks": global_checks, "self_control_case_count": 7, "alpha0_control_case_count": 28})
        self.check("aggregate_control_alias", aggregate["global_controls"] == summary["global_controls"])
        self.check("independent_case_gating_and_all_primary_counts", summary["primary_counts"] == aggregate["primary_counts"] == independent_primary)
        self.primary = independent_primary
        self.controls_pass = controls_pass
        for relative, expected in list(self.sources.items()): self.check("source_unchanged_after_audit:" + relative, sha(ROOT / relative) == expected)

def audit(run):
    import mujoco
    import random
    import torch
    run = Path(run).resolve()
    checker = Audit()
    entries = [(mujoco, name, "physics." + name) for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse")]
    entries += [(torch.nn.Module, "__init__", "model_instances"), (torch.nn.Module, "_call_impl", "model_forwards"),
        (torch.optim.Adam, "__init__", "Adam_instances"), (torch.optim.Optimizer, "__init__", "Optimizer_instances"),
        (torch.optim.Adam, "step", "Adam.step"), (torch.optim.Optimizer, "step", "Optimizer.step"),
        (torch.autograd, "grad", "autograd.grad"), (torch.Tensor, "backward", "Tensor.backward"), (torch, "save", "torch.save")]
    counts, originals = {label: 0 for _, _, label in entries}, []
    torch_rng, numpy_rng, python_rng = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    def blocked(label):
        def call(*args, **kwargs):
            counts[label] += 1
            raise RuntimeError("Saved-output cross-batch audit forbids " + label)
        return call
    for owner, name, label in entries:
        originals.append((owner, name, getattr(owner, name)))
        setattr(owner, name, blocked(label))
    result = {"schema": "p4-cross-batch-independent-review-v1", "status": "FAIL", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(run.relative_to(ROOT)), "review_source_sha256": sha(__file__), "checks": checker.checks,
        "numeric_policy": {"metric_rtol": METRIC_RTOL, "metric_atol": METRIC_ATOL,
            "alpha0_reproduction_rtol": REPRODUCTION_RTOL, "alpha0_reproduction_atol": REPRODUCTION_ATOL,
            "numeric_tie_margin": "1e-6*max(1,abs(alpha0_total_loss))", "parameter_arithmetic": "Exact float32 multiply, add and subtract on independent original copies",
            "primary_taylor": "Saved raw total gradient float64 dot promoted float32(theta_alpha-theta)",
            "sensitivity_taylor": "Separate float64 difference of the two represented float32 endpoints",
            "loss_oracle": "NumPy float64 from saved normalized teacher predictions, physical free states and immutable original delta statistics"}}
    try:
        checker.run(run)
        checker.check("auditor_zero_model_gradient_update_or_physics", not any(counts.values()))
        current_numpy = np.random.get_state()
        checker.check("auditor_rng_unchanged", torch.equal(torch_rng, torch.get_rng_state())
            and numpy_rng[0] == current_numpy[0] and np.array_equal(numpy_rng[1], current_numpy[1])
            and numpy_rng[2:] == current_numpy[2:] and python_rng == random.getstate())
        result.update(status="PASS", observed={"cases": checker.observed,
            "secondary_decision_counts": dict(Counter(row["secondary_decision"] for row in checker.observed)),
            "retained_case_denominator": 28, "retained_evaluation_denominator": 112,
            "planned_positive_endpoint_denominator": 84, "self_stop_case_count": 7, "other_source_case_count": 21,
            "successful_evaluation_count": checker.successful_evaluations, "primary_counts": checker.primary,
            "global_controls_pass": checker.controls_pass, "controls": checker.controls}, source_bindings_sha256=checker.sources,
            limitations=["No model construction/forward, derivative, optimizer, training, checkpoint write or physics is executed by the auditor.",
                "Losses are independently recomputed from saved teacher/free predictions; actual evaluated parameter maps and root forward trace are bound without replaying a learned function.",
                "Native source directions retain different norms. This is an effect of the specified source addends on a frozen same-snapshot stop batch, not an isolated regime or normalized-direction causal effect.",
                "Target raw total gradients define Taylor comparisons. Registered float32 subtraction stays primary; represented-endpoint float64 difference remains separate sensitivity.",
                "Every28 case,112 evaluation and84 positive endpoint is retained. Numeric/reference failures hold whole cases; any exact saved-control mismatch holds all primary counts.",
                "Alpha0 loss/prediction controls and seven self-stop saved-output reproductions are computational negative controls, not independent experimental replication.",
                "Target loss increases/decreases are per-alpha primary observations. Composite finite-pattern decisions remain secondary; INCONCLUSIVE is not evidence of no target increase.",
                "No checkpoint, learning rate or direction is selected; whole historical causes, independent generalization, alignment and personal understanding remain unassessed."])
    except Exception as error:
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        for owner, name, original in reversed(originals): setattr(owner, name, original)
    result["forbidden_execution_counts"] = counts
    result["check_count"] = len(checker.checks)
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    require(not arguments.output.exists(), "Never overwrite a review or failed prefix")
    result = audit(arguments.run)
    arguments.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "checks": result["check_count"], "failure": result.get("failure")}))
    if result["status"] != "PASS": raise SystemExit(1)

if __name__ == "__main__":
    main()
