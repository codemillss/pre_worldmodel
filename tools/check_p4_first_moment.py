"""Saved-output first-moment audit; never construct/call a learned model."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import check_p4_finite_step as F

MODES = ("KEEP", "ZERO_M")
INDICES = {"KEEP": 3, "ZERO_M": 4}
SNAPSHOT, SOURCE, TARGET = "R1_u1000", "D11_noncontact", "D11_zero5"
ALPHAS = F.ALPHAS
read, sha, arrays, require = F.read, F.sha, F.arrays, F.require


def typed_hash(value):
    a = np.asarray(value)
    descriptor = json.dumps({"dtype": str(a.dtype), "shape": list(a.shape)}, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(descriptor + b"\0" + np.ascontiguousarray(a).tobytes()).hexdigest()


def array_equal(first, second):
    return first.dtype == second.dtype and first.shape == second.shape and np.array_equal(first, second)


def saved_controls(mode, evaluation, predictions, parameters, previous, previous_predictions, previous_parameters, keep0):
    """Exact saved negative controls, separate from tolerant raw5 reproduction."""
    rows, old = evaluation["evaluations"], previous["evaluations"]
    require(mode in MODES and len(rows) == len(old) == 4, "Both registered modes retain all four endpoints")
    current0 = {"loss": rows[0]["loss_values"], **{key: value[0] for key, value in predictions.items()}}
    baseline = current0["loss"] == old[0]["loss_values"] and all(array_equal(current0[key], previous_predictions[key][0]) for key in predictions)
    same0 = current0["loss"] == keep0["loss"] and all(array_equal(current0[key], keep0[key]) for key in predictions)
    exact = None
    if mode == "KEEP":
        checks = {"parameter_maps_exact": set(parameters) == set(previous_parameters) and all(array_equal(parameters[key], previous_parameters[key]) for key in parameters),
            "all_four_loss_values_statuses_exact": all(row["loss_values"] == oldrow["loss_values"] and row["status"] == oldrow["status"] for row, oldrow in zip(rows, old)),
            "teacher_predictions_exact": array_equal(predictions["teacher_predictions"], previous_predictions["teacher_predictions"]),
            "free_predictions_exact": array_equal(predictions["free_predicted_states"], previous_predictions["free_predicted_states"])}
        exact = {"pass": all(checks.values()), "checks": checks}
    return {"alpha0_matches_previous_stop": bool(baseline), "alpha0_matches_keep": bool(same0), "keep_previous_cross": exact}


def primary_counts(cases, controls_pass):
    require(len(cases) == 2 and [row["source_mode"] for row in cases] == list(MODES), "KEEP and ZERO_M retained in order")
    if not controls_pass:
        return None
    relations = ("INCREASE", "DECREASE", "NUMERIC_TIE", "NOT_INTERPRETED")
    def group(rows):
        by_alpha = {}
        for index, alpha in enumerate(ALPHAS[1:], 1):
            count = Counter(row["change_relations"][index] if row["case_status"] == "PASS" else "NOT_INTERPRETED" for row in rows)
            require(set(count) <= set(relations), "Registered target change relation required")
            by_alpha[str(alpha)] = {name: count[name] for name in relations}
        return {"case_count": len(rows), "planned_positive_endpoints": 3 * len(rows), "by_alpha": by_alpha}
    zero = cases[1]
    return {"by_mode": {mode: group([cases[index]]) for index, mode in enumerate(MODES)}, "all": group(cases),
        "zero_m_increase_persists": {str(alpha): zero["change_relations"][index] == "INCREASE" if zero["case_status"] == "PASS" else None
                                    for index, alpha in enumerate(ALPHAS[1:], 1)},
        "interpretation_gate": "PASS" if all(row["case_status"] == "PASS" for row in cases) else "HOLD_CASE_FAILURE"}


class Audit(F.Audit):
    def __init__(self):
        super().__init__()
        self.keep0 = None

    def bind(self, path, expected=None):
        path = Path(path).resolve()
        require(path.is_relative_to(ROOT), "Input path outside project")
        name, digest = path.relative_to(ROOT).as_posix(), sha(path)
        self.check("input:" + name, expected is None or digest == expected)
        self.check("repeat:" + name, name not in self.sources or self.sources[name] == digest)
        self.sources[name] = digest
        return path

    def routing(self, route, case, spec, order, direction, gradient, reference, states, actions, mask, decision):
        common = {"snapshot_id": SNAPSHOT, "model_tree_hash": spec["model_tree_hash"], "parameter_order": order}
        source = {**common, "probe_name": SOURCE, "objective": "total", "mode": case["source_mode"],
                  "direction_index": INDICES[case["source_mode"]], "delta_sha256": typed_hash(direction)}
        batch_hashes = {key: typed_hash(value) for key, value in (("states", states), ("actions", actions), ("mask", mask))}
        target = {**common, "probe_name": TARGET, "batch_windows": 24, **{key + "_sha256": value for key, value in batch_hashes.items()}}
        gradient_binding = {**common, "probe_name": TARGET, "component": "total", "gradient_sha256": typed_hash(gradient)}
        loss = {**common, "probe_name": TARGET, "loss_values": reference}
        self.tree(case["source_mode"] + ":routing", route, {"schema": "p4-first-moment-routing-v1", "snapshot_id": SNAPSHOT,
            "source_probe_name": SOURCE, "target_probe_name": TARGET, "source_mode": case["source_mode"],
            "model_tree_hash": spec["model_tree_hash"], "parameter_order": order,
            "source_direction": {"objective": "total", "mode": case["source_mode"], "direction_index": INDICES[case["source_mode"]], "delta_sha256": typed_hash(direction), "norm_normalized": False},
            "target_gradient": {"snapshot_id": SNAPSHOT, "probe_name": TARGET, "component": "total", "gradient_sha256": typed_hash(gradient)},
            "target_reference_losses": reference, "target_batch": {"batch_windows": 24, "typed_sha256": batch_hashes},
            "secondary_finite_pattern_decision": decision,
            "routing_checks": {key: True for key in ("same_snapshot", "same_actual_model", "same_parameter_order", "allowed_saved_mode_and_index",
                "native_source_direction_unchanged", "target_gradient_and_reference_bound", "target_batch_unchanged", "binding_metadata_unchanged")}})
        self.check(case["source_mode"] + ":exact_typed_binding_records", route["binding_records"] == {"source": source, "target": target,
            "gradient": gradient_binding, "loss": loss, "reference_losses": reference, "parameter_order": order})

    def case(self, run, case, spec, weights, optimizer_hash, probes, statistics):
        mode = case["source_mode"]
        self.check("fixed_mode_identity", mode in MODES and case["snapshot_id"] == SNAPSHOT and case["source_probe_name"] == SOURCE
            and case["target_probe_name"] == TARGET and case["source_direction_index"] == INDICES[mode])
        label = mode
        paths = {"source_direction_npz": f"runs/p4-adam-probe-20261015-01/probes/{SNAPSHOT}/{SOURCE}/directions.npz",
            "source_direction_summary": f"runs/p4-adam-probe-20261015-01/probes/{SNAPSHOT}/{SOURCE}/summary.json",
            "target_gradient_summary": f"runs/p4-gradient-probe-20261015-01/probes/{SNAPSHOT}/{TARGET}/summary.json",
            "target_gradient_npz": f"runs/p4-gradient-probe-20261015-01/probes/{SNAPSHOT}/{TARGET}/gradients.npz",
            **{"prevcross_" + key: f"runs/p4-cross-batch-20261015-01/cases/{SNAPSHOT}/{SOURCE}/{filename}"
               for key, filename in (("summary", "summary.json"), ("parameters", "parameter_displacements.npz"), ("predictions", "predictions.npz"))}}
        self.check(label + ":paths", all(case[key] == value for key, value in paths.items()))
        bound = {key: self.bind(ROOT / value, case[key + "_sha256"]) for key, value in paths.items()}
        target, source, previous_record = (read(bound[key]) for key in ("target_gradient_summary", "source_direction_summary", "prevcross_summary"))
        order = target["parameter_order"]; F.validate_order(order)
        self.check(label + ":bound_same_model_coordinates", source["parameter_order"] == order == previous_record["parameter_order"]
            and source["snapshot_id"] == target["snapshot_id"] == previous_record["snapshot_id"] == SNAPSHOT
            and source["probe_name"] == previous_record["source_probe_name"] == SOURCE and target["probe_name"] == previous_record["target_probe_name"] == TARGET
            and source["source_weights_tree_hash"] == target["before_weights_hash"] == previous_record["source_weights_tree_hash"] == spec["model_tree_hash"])
        self.check(label + ":stored_first_moment_mode", source["objective_order"] == ["base", "total"] and source["mode_order"] == ["KEEP", "ZERO_M", "COLD"]
            and source["directions"][INDICES[mode]]["objective"] == "total" and source["directions"][INDICES[mode]]["mode"] == mode
            and source["directions"][3]["saved_step"] == source["directions"][4]["saved_step"] == 2300
            and source["directions"][3]["next_step"] == source["directions"][4]["next_step"] == 2301)
        self.check(label + ":saved_optimizer_and_constants", source["source_optimizer_tree_hash"] == optimizer_hash
            and previous_record["source_optimizer_tree_hash"] == optimizer_hash and previous_record["source"]["source_direction_index"] == 3
            and source["adam"]["learning_rate"] == .001 and source["adam"]["betas"] == [.9, .999]
            and source["adam"]["epsilon"] == 1e-8 and source["adam"]["weight_decay"] == 0 and source["adam"]["amsgrad"] is False)
        direction = arrays(bound["source_direction_npz"])["directions"][INDICES[mode]]
        gradient = arrays(bound["target_gradient_npz"])["gradients"][4]
        original = np.concatenate([weights[item["name"]].detach().numpy().reshape(-1) for item in order])
        expected_parameters = F.parameter_arithmetic(original, direction)
        self.check(label + ":direction_target_gradient", direction.dtype == gradient.dtype == np.float32 and direction.shape == gradient.shape == (19338,) and np.isfinite(gradient).all())
        folder = run / "cases" / SNAPSHOT / mode
        record = read(folder / "summary.json"); parameters = arrays(folder / "parameter_displacements.npz"); predictions = arrays(folder / "predictions.npz")
        self.check(label + ":array_fields", set(parameters) == {"alphas", "evaluated_parameters", "rounded_parameter_differences"}
            and set(predictions) == {"teacher_predictions", "free_predicted_states"})
        self.check(label + ":alphas", array_equal(parameters["alphas"], np.asarray(ALPHAS, np.float64)))
        for key in ("evaluated_parameters", "rounded_parameter_differences"):
            self.check(label + ":independent_float32_copy:" + key, array_equal(parameters[key], expected_parameters[key]))
        self.tree(label + ":record", record, {"snapshot_id": SNAPSHOT, "source_probe_name": SOURCE, "target_probe_name": TARGET, "source_mode": mode,
            "source": case, "parameter_order": order, "source_weights_tree_hash": spec["model_tree_hash"], "source_optimizer_tree_hash": optimizer_hash,
            "parameter_arrays_sha256": sha(folder / "parameter_displacements.npz"), "prediction_arrays_sha256": sha(folder / "predictions.npz")})
        states, actions, mask = (probes[TARGET + "_" + key] for key in ("states", "actions", "mask"))
        self.check(label + ":fixed_stop", states.dtype == actions.dtype == np.float64 and states.shape == (24, 6, 10)
            and actions.shape == (24, 5, 2) and mask.dtype == np.bool_ and mask.shape == (24, 5) and mask.all())
        self.check(label + ":prediction_shapes", predictions["teacher_predictions"].dtype == np.float32 and predictions["teacher_predictions"].shape == (4, 120, 10)
            and predictions["free_predicted_states"].dtype == np.float64 and predictions["free_predicted_states"].shape == (4, 24, 6, 10))
        evaluation = record["evaluation"]; rows = evaluation["evaluations"]
        denominators = {"batch_windows": 24, "valid_transitions": 120, "base_scalar_terms": 1200, "relative_scalar_terms": 240}
        self.tree(label + ":evaluation", evaluation, {"schema": "p4-fixed-direction-finite-step-evaluation-v1", "alphas": list(ALPHAS), "parameter_order": order,
            "parameter_count": 19338, "raw_loss_order": ["one_step", "rollout", "relative_teacher", "relative_rollout", "total"], "retained_alpha_count": 4, "case_denominator": 1,
            "denominators": denominators, "source_state_hash_before": F.mapping_hash(original, order), "source_state_hash_after": F.mapping_hash(original, order)})
        self.check(label + ":all_rows_and_preservation", len(rows) == 4 and set(evaluation["checks"]) == {"source_state_unchanged", "source_parameter_identities_unchanged",
            "source_buffer_identities_unchanged", "source_gradient_buffers_unchanged", "source_modes_unchanged", "source_hooks_unchanged", "global_rng_unchanged",
            "batch_inputs_unchanged", "normalizer_unchanged", "direction_and_gradient_inputs_unchanged"} and all(value is True for value in evaluation["checks"].values()))
        losses, trace, calls, processed = [], [], 0, 0
        for index, alpha in enumerate(ALPHAS):
            row = rows[index]; key = label + ":" + str(alpha)
            self.tree(key + ":identity", row, {"index": index, "alpha": alpha, "denominators": denominators,
                "evaluated_parameters_hash": hashlib.sha256(parameters["evaluated_parameters"][index].tobytes()).hexdigest(),
                "rounded_difference_hash": hashlib.sha256(parameters["rounded_parameter_differences"][index].tobytes()).hexdigest()})
            self.check(key + ":calls", row["status"] in ("PASS", "NUMERICAL_FAILURE") and type(row["forward_calls"]) is int and 0 <= row["forward_calls"] <= 6
                and row["forward_rows"] == [120, 24, 24, 24, 24, 24][:row["forward_calls"]] and row["forward_total_rows"] == sum(row["forward_rows"]))
            if row["status"] == "PASS":
                self.check(key + ":success", row["forward_calls"] == 6 and row["error"] is None)
                independent = F.loss_oracle(states, mask, predictions["teacher_predictions"][index], predictions["free_predicted_states"][index], statistics["delta_mean"], statistics["delta_scale"])["losses"]
                expected_losses = F.production_losses(independent)
                self.check(key + ":loss_fields", set(row["loss_values"]) == set(expected_losses))
                self.tree(key + ":independent_loss", row["loss_values"], expected_losses)
                losses.append(F.canonical_losses(row["loss_values"]))
            else:
                self.check(key + ":retained_failure", set(row["loss_values"]) == {"one_step", "rollout", "relative_teacher", "relative_rollout", "total", "base", "weighted_aux", "common_total"}
                    and all(value is None for value in row["loss_values"].values()) and isinstance(row["error"], dict) and set(row["error"]) == {"type", "message"})
                losses.append(None)
            evaluated, rounded = parameters["evaluated_parameters"][index], parameters["rounded_parameter_differences"][index]
            exact, nominal = evaluated.astype(np.float64) - original.astype(np.float64), alpha * direction.astype(np.float64)
            self.tree(key + ":displacement", row["displacement"], {"nominal_scaled_norm": F.norm(nominal), "rounded_difference_norm": F.norm(rounded),
                "exact_endpoint_difference_norm": F.norm(exact), "rounded_minus_nominal_l2": F.norm(rounded.astype(np.float64) - nominal), "exact_minus_nominal_l2": F.norm(exact - nominal)})
            self.tree(key + ":target_Taylor_changes", row["taylor"], {"nominal_directional_change": F.dot(gradient, nominal), "rounded_directional_change": F.dot(gradient, rounded), "exact_endpoint_directional_change": F.dot(gradient, exact)})
            trace.extend({"rows": count, "parameter_tree_hash": F.parameter_tree_hash(evaluated, order)} for count in row["forward_rows"])
            calls += row["forward_calls"]; processed += row["forward_total_rows"]
        self.check(label + ":actual_parameter_forward_trace", record["source_forward_trace"] == trace)
        classification = F.classify_case(losses, target["loss_values"])
        baseline_status = "NUMERICAL_FAILURE" if losses[0] is None else "PASS" if classification["reproduction"]["pass"] else "REPRODUCTION_FAILURE"
        per_loss = {}
        for name, expected in target["loss_values"].items():
            actual = rows[0]["loss_values"][name]; error = None if actual is None else abs(actual - expected); allowed = F.REPRODUCTION_ATOL + F.REPRODUCTION_RTOL * abs(expected)
            per_loss[name] = {"actual": actual, "reference": expected, "absolute_error": error, "allowed_error": allowed, "pass": error is not None and error <= allowed}
        self.tree(label + ":raw5_reference", evaluation["alpha0_reproduction"], {"status": baseline_status, "rtol": F.REPRODUCTION_RTOL, "atol": F.REPRODUCTION_ATOL, "per_loss": per_loss})
        self.number(label + ":tie_margin", evaluation["neutral_band"], classification["numeric_tie_margin"])
        for index, row in enumerate(rows):
            available = losses[index] is not None and losses[0] is not None; key = label + ":" + str(ALPHAS[index])
            self.tree(key + ":actual_changes", row["loss_differences"], {name: row["loss_values"][name] - rows[0]["loss_values"][name] if available else None for name in ("base", "weighted_aux", "common_total")})
            taylor = dict.fromkeys(("primary_prediction", "endpoint_sensitivity_prediction", "nominal_prediction", "residual_actual_minus_primary", "residual_actual_minus_endpoint"))
            if available:
                m = F.taylor_metrics(original, parameters["evaluated_parameters"][index], parameters["rounded_parameter_differences"][index], direction, gradient, ALPHAS[index], losses[0]["total"], losses[index]["total"])
                taylor = {"primary_prediction": m["primary_taylor_loss"], "endpoint_sensitivity_prediction": m["endpoint_float64_taylor_loss"], "nominal_prediction": m["nominal_taylor_loss"],
                    "residual_actual_minus_primary": m["actual_minus_primary_taylor"], "residual_actual_minus_endpoint": losses[index]["total"] - m["endpoint_float64_taylor_loss"]}
            self.tree(key + ":Taylor_predictions", row["taylor"], taylor)
            self.check(key + ":relation", row["total_change_relation"] == (classification["change_relations"][index] if available and baseline_status == "PASS" else "NOT_INTERPRETED"))
        decision = {"NOMINAL_STEP_OVERSHOOT_SUPPORTED": "NOMINAL_INCREASE_SMALLER_DECREASE", "ALL_POSITIVE_ALPHAS_DECREASE": "NO_ALPHA1_OVERSHOOT_OBSERVED_ON_GRID"}.get(classification["verdict"], classification["verdict"])
        successful = sum(value is not None for value in losses); status = "NUMERICAL_FAILURE" if successful < 4 else baseline_status
        self.check(label + ":failure_priority", evaluation["status"] == status and evaluation["decision"] == decision)
        self.tree(label + ":counters", evaluation["counters"], {"evaluations": 4, "successful_evaluations": successful, "root_model_forward_calls": calls, "root_model_forward_rows": processed,
            "new_autograd_traversals": 0, "new_optimizer_steps": 0, "new_training_updates": 0, "new_physics_steps": 0})
        current0 = {"loss": rows[0]["loss_values"], **{name: value[0].copy() for name, value in predictions.items()}}
        if self.keep0 is None:
            require(mode == "KEEP", "KEEP baseline must be observed first"); self.keep0 = current0
        controls = saved_controls(mode, evaluation, predictions, parameters, previous_record["evaluation"], arrays(bound["prevcross_predictions"]), arrays(bound["prevcross_parameters"]), self.keep0)
        self.check(label + ":saved_control_records", record["controls"] == controls)
        self.routing(evaluation["first_moment_routing"], case, spec, order, direction, gradient, target["loss_values"], states, actions, mask, decision)
        self.observed.append({"source_mode": mode, "case_status": status, "controls": controls, "change_relations": classification["change_relations"], "target_changes": classification["total_loss_changes"], "secondary_decision": decision})
        self.forward_calls += calls; self.forward_rows += processed; self.successful_evaluations += successful
        return {"snapshot_id": SNAPSHOT, "source_probe_name": SOURCE, "target_probe_name": TARGET, "source_mode": mode,
            "summary_path": str((folder / "summary.json").relative_to(run)), "evaluation": evaluation, "controls": controls}

    def run(self, run):
        import platform
        import torch
        protocol, manifest, summary, aggregate = (read(run / name) for name in ("registration.json", "manifest.json", "summary.json", "aggregate.json"))
        self.check("registered_run_identity", run == ROOT / "runs/p4-first-moment-20261015-01" and protocol["run_id"] == run.name
            and protocol["run_directory"] == str(run.relative_to(ROOT)) and manifest["schema"] == "p4-first-moment-manifest-v1"
            and summary["schema"] == "p4-first-moment-summary-v1" and summary["execution_status"] == "PASS")
        self.check("registration_and_manifest_binding", sha(run / "registration.json") == sha(ROOT / "configs/p4_first_moment.v1.json") == manifest["protocol_sha256"] == summary["protocol_sha256"]
            and sha(run / "manifest.json") == summary["manifest_sha256"])
        for name, digest in protocol["bindings_sha256"].items(): self.bind(ROOT / name, digest)
        self.bind(ROOT / "configs/p4_first_moment.v1.json", summary["protocol_sha256"])
        required = ("tools/check_p4_first_moment.py", "tests/test_p4_first_moment_audit.py", "src/push_align/first_moment_probe.py", "scripts/p4_first_moment_probe.py", "tools/check_p4_finite_step.py")
        self.check("auditor_helpers_and_tests_registered_before_execution", all(name in protocol["bindings_sha256"] for name in required))
        design = read(self.bind(ROOT / protocol["design_path"], protocol["design_sha256"]))
        self.check("two_fixed_modes_and_four_alphas", protocol["cases"] == design["cases"] and len(protocol["cases"]) == 2
            and [(case["snapshot_id"], case["source_probe_name"], case["target_probe_name"], case["source_mode"], case["source_direction_index"]) for case in protocol["cases"]]
                == [(SNAPSHOT, SOURCE, TARGET, mode, INDICES[mode]) for mode in MODES]
            and protocol["alphas"] == manifest["alphas"] == design["alphas"] == list(ALPHAS))
        self.check("one_frozen_snapshot", protocol["snapshots"] == design["snapshots"] == manifest["model_snapshots"] and len(protocol["snapshots"]) == 1 and protocol["snapshots"][0]["snapshot_id"] == SNAPSHOT)
        runtime = protocol["runtime"]
        self.check("runtime", runtime == manifest["runtime"] and runtime["torch_version"] == str(torch.__version__) and runtime["numpy_version"] == np.__version__ and runtime["python_version"] == platform.python_version())
        for name, digest in runtime["source_sha256"].items(): self.check("runtime_source:" + name, sha(Path(name)) == digest)
        expected_outputs = {"registration.json", "manifest.json", "summary.json", "aggregate.json"} | {f"cases/{SNAPSHOT}/{mode}/{name}" for mode in MODES for name in ("summary.json", "parameter_displacements.npz", "predictions.npz")}
        index = read(run / "artifact_hashes.json")["sha256"]
        self.check("exact10_output_inventory", set(index) == expected_outputs)
        for name, digest in index.items(): self.bind(run / name, digest)
        spec = protocol["snapshots"][0]; checkpoint = ROOT / spec["checkpoint_path"]
        metadata = read(self.bind(checkpoint / "checkpoint.json", spec["checkpoint_json_sha256"]))
        self.check("checkpoint_metadata", metadata["checkpoint_hash"] == spec["checkpoint_hash"] == hashlib.sha256(F.canonical({key: value for key, value in metadata.items() if key != "checkpoint_hash"})).hexdigest())
        for name in ("weights.pt", "training_state.pt"): self.bind(checkpoint / name, spec["files_sha256"][name])
        weights = torch.load(checkpoint / "weights.pt", map_location="cpu", weights_only=True)
        optimizer = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)["optimizer"]
        model_hash, optimizer_hash = F.tensor_tree_hash(weights), F.tensor_tree_hash(optimizer)
        self.check("saved_model_Adam_identity_and_age", model_hash == spec["model_tree_hash"] and all(float(state["step"]) == 2300 for state in optimizer["state"].values())
            and all(value.dtype == torch.float32 and not value.requires_grad and torch.isfinite(value).all() for value in weights.values()))
        adam_review = read(self.bind(ROOT / "reports/p4_adam_probe_independent_review.json"))
        self.check("bound_prior_independent_m_only_Adam_arithmetic", adam_review["status"] == "PASS"
            and adam_review["review_source_sha256"] == sha(ROOT / "tools/check_p4_adam_probe.py")
            and all(any(check["name"] == f"{SNAPSHOT}:{SOURCE}:total:{mode}:independent_ideal_adam_envelope" and check["pass"] for check in adam_review["checks"]) for mode in MODES))
        for path in (protocol["cases"][0]["source_direction_npz"], f"{spec['checkpoint_path']}/training_state.pt",
                     f"runs/p4-gradient-probe-20261015-01/probes/{SNAPSHOT}/{SOURCE}/gradients.npz"):
            self.bind(ROOT / path, adam_review["source_bindings_sha256"][path])
        gradient_run = ROOT / "runs/p4-gradient-probe-20261015-01"
        probes = arrays(self.bind(gradient_run / "probe_batches.npz", manifest["probe_batches_sha256"]))
        self.check("fixed_target_manifest", manifest["source_probe_name"] == SOURCE and manifest["target_probe_name"] == TARGET and manifest["source_modes"] == list(MODES)
            and manifest["target_batch_windows"] == 24 and manifest["target_array_sha256"] == {key: typed_hash(probes[TARGET + "_" + key]) for key in ("states", "actions", "mask")})
        normalizer = ROOT / "runs/p3-02-20261009-01/learning_data/normalizer"; normalizer_meta = read(self.bind(normalizer / "normalizer.json"))
        statistics = arrays(self.bind(normalizer / "statistics.npz", normalizer_meta["npz_sha256"]))
        self.check("original_normalizer_and_physical_config", manifest["normalizer_hash"] == normalizer_meta["normalizer"]["normalizer_hash"]
            and manifest["physical_config"] == read(ROOT / "runs/p4-relative-loss-20261014-01/manifest.json")["physical_config"])
        rows = [self.case(run, case, spec, weights, optimizer_hash, probes, statistics) for case in protocol["cases"]]
        self.check("aggregate_aliases", aggregate["schema"] == "p4-first-moment-aggregate-v1" and aggregate["row_count"] == 2 and aggregate["rows"] == rows)
        checks = {"exactly_keep_zero_m": [row["source_mode"] for row in rows] == list(MODES),
            "two_alpha0_previous_stop_exact": all(row["controls"]["alpha0_matches_previous_stop"] for row in rows),
            "two_alpha0_equal": all(row["controls"]["alpha0_matches_keep"] for row in rows), "keep_previous_cross_exact": rows[0]["controls"]["keep_previous_cross"]["pass"]}
        controls_pass = all(checks.values()); self.primary = primary_counts(self.observed, controls_pass)
        self.tree("global_controls", summary["global_controls"], {"pass": controls_pass, "checks": checks})
        self.check("global_gating_and_primary_counts", aggregate["global_controls"] == summary["global_controls"] and aggregate["primary_counts"] == summary["primary_counts"] == self.primary
            and summary["analysis_status"] == ("HOLD_CONTROL_FAILURE" if not controls_pass else "HOLD_CASE_FAILURE"
                if any(case["case_status"] != "PASS" for case in self.observed) else "DESCRIPTIVE_COMPLETE"))
        self.tree("retained8_evaluations_and_forward_counts", summary["counters"], {"snapshots": 1, "cases": 2, "attempted_evaluations": 8, "completed_evaluations": self.successful_evaluations,
            "forward_calls": self.forward_calls, "forward_rows": self.forward_rows})
        self.check("forward_budget", self.forward_calls <= 48 and self.forward_rows <= 1920 and (self.successful_evaluations < 8 or (self.forward_calls, self.forward_rows) == (48, 1920)))
        self.tree("source_tensor_preservation", summary["source_tensor_checks"], [{"snapshot_id": SNAPSHOT, "weights_unchanged": True, "optimizer_unchanged": True, "weight_tree_hash": model_hash, "optimizer_tree_hash": optimizer_hash}])
        forbidden = {"physics." + name for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse")} | {"Adam_instances", "Optimizer_instances", "Adam.step", "Optimizer.step", "autograd.grad", "Tensor.backward", "torch.save"}
        self.check("no_forbidden_execution", set(summary["forbidden_execution_counts"]) == forbidden and not any(summary["forbidden_execution_counts"].values()))
        self.tree("scope", summary, {"global_rng_unchanged": True, "new_derivative_traversals": 0, "new_optimizer_updates": 0, "new_physics_steps": 0, "new_checkpoint_writes": 0,
            "new_data": False, "normalizer_refit": False, "final_test_used": False, "validation_or_development_probes_used": False, "historical_trajectory_replayed": False,
            "model_selection": "NONE", "learning_rate_adoption": "NONE", "optimizer_change_adoption": "NONE", "alignment_improvement": "NOT_EVALUATED", "personal_understanding": "NOT_ASSESSED", "P5_01": "HOLD"})
        self.controls = summary["global_controls"]
        for name, digest in self.sources.items(): self.check("unchanged_after_audit:" + name, sha(ROOT / name) == digest)


def audit(run):
    import mujoco
    import random
    import torch
    checker = Audit(); run = Path(run).resolve()
    hooks = [(mujoco, name, "physics." + name) for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse")]
    hooks += [(torch.nn.Module, "__init__", "model_instances"), (torch.nn.Module, "_call_impl", "model_forwards"),
        (torch.optim.Adam, "__init__", "Adam_instances"), (torch.optim.Optimizer, "__init__", "Optimizer_instances"),
        (torch.optim.Adam, "step", "Adam.step"), (torch.optim.Optimizer, "step", "Optimizer.step"),
        (torch.autograd, "grad", "autograd.grad"), (torch.Tensor, "backward", "Tensor.backward"), (torch, "save", "torch.save")]
    counts, originals = {name: 0 for _, _, name in hooks}, []
    rng = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    def block(label):
        def forbidden(*args, **kwargs):
            counts[label] += 1
            raise RuntimeError("Saved-output first-moment audit forbids " + label)
        return forbidden
    for owner, name, label in hooks:
        originals.append((owner, name, getattr(owner, name))); setattr(owner, name, block(label))
    result = {"schema": "p4-first-moment-independent-review-v1", "status": "FAIL", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(run.relative_to(ROOT)), "review_source_sha256": sha(__file__), "checks": checker.checks,
        "numeric_policy": {"metric_rtol": F.METRIC_RTOL, "metric_atol": F.METRIC_ATOL, "alpha0_raw5_rtol": F.REPRODUCTION_RTOL,
            "alpha0_raw5_atol": F.REPRODUCTION_ATOL, "tie_margin": "1e-6*max(1,abs(target_alpha0_total))", "saved_controls": "Exact arrays/losses/statuses",
            "primary_taylor": "Promoted float32 endpoint subtraction dot saved target raw total gradient; float64 endpoint sensitivity kept separate",
            "m_only_source": "Frozen original Adam independent ideal64 checks and matching gclip/source-gradient,optimizer-state,direction artifact hashes; no Adam recomputation"}}
    try:
        checker.run(run)
        checker.check("auditor_forbidden_counts_zero", not any(counts.values()))
        now = np.random.get_state()
        checker.check("auditor_rng_unchanged", torch.equal(rng[0], torch.get_rng_state()) and rng[1][0] == now[0]
            and np.array_equal(rng[1][1], now[1]) and rng[1][2:] == now[2:] and rng[2] == random.getstate())
        result.update(status="PASS", source_bindings_sha256=checker.sources, observed={"cases": checker.observed,
            "primary_counts": checker.primary, "global_controls": checker.controls, "retained_case_denominator": 2,
            "retained_evaluation_denominator": 8, "planned_positive_endpoint_denominator": 6, "successful_evaluation_count": checker.successful_evaluations},
            limitations=["Audit reads saved predictions and tensor bytes only; no learned function is reexecuted.",
                "KEEP exact replay and two alpha0 controls are computational controls,not independent replication.",
                "ZERO_M retains current-gradient contribution and saved v/t; its native angle and magnitude both change. No angle-only effect is claimed.",
                "All2 modes,8 evaluations and6 positive endpoints remain. Control failure holds all primary counts; numerical/reference failures hold each whole case.",
                "One final-R1 train-stop contrast does not establish historical whole-training cause,generalization,physical alignment or optimizer-change adoption."])
    except Exception as error:
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        for owner, name, original in reversed(originals): setattr(owner, name, original)
    result["forbidden_execution_counts"] = counts; result["check_count"] = len(checker.checks)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); require(not args.output.exists(), "Never overwrite a review or failed prefix")
    result = audit(args.run); args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "checks": result["check_count"], "failure": result.get("failure")}))
    if result["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__":
    main()
