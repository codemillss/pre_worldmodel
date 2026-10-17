"""A centered SVD Ridge delta model at the shared float32 rollout boundary.

Fit receives physical rows; the existing Normalizer supplies the same
float32 normalized inputs/targets used by the other models. Only the linear
solve is float64. Row split selection belongs to the caller: this API accepts
no split labels and cannot establish that its rows came from train.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import torch
from torch import nn

from .checkpoint import _config_schema
from .data import _json_bytes, _read_json, _write_json, digest
from .dynamics import _model_parameters, _no_autocast, _tensor
from .normalization import Normalizer


class RidgeError(ValueError):
    """Ridge fit or its saved contract is invalid or incompatible."""


OBJECTIVE_DEFINITION = "sum_squared_normalized_residuals + alpha * coefficient_squared_Frobenius_norm; intercept_unpenalized"
ARTIFACT_FORMAT = "push-align-ridge-v1"


def _require(condition, message):
    if not condition:
        raise RidgeError(message)


def _scalar(value, name):
    _require(not isinstance(value, (bool, np.bool_))
             and isinstance(value, (int, float, np.integer, np.floating)),
             f"{name} requires a finite nonnegative real scalar")
    result = float(value)
    _require(math.isfinite(result) and result >= 0,
             f"{name} requires a finite nonnegative real scalar")
    return result


def ridge_spec(config=None):
    if config is not None:
        _require(isinstance(config, dict), "Ridge config requires an object")
        runtime, model = config.get("runtime", {}), config.get("model", {})
        _require(isinstance(runtime, dict) and runtime.get("device") == "cpu"
                 and runtime.get("model_dtype") == "float32", "Ridge requires CPU float32")
        _require(isinstance(model, dict)
                 and type(model.get("learned_inputs")) is int and model["learned_inputs"] == 11
                 and type(model.get("learned_outputs")) is int and model["learned_outputs"] == 10,
                 "Ridge requires the existing 11-input/10-delta contract")
    return {"kind": "ridge_delta", "learned_inputs": 11, "learned_outputs": 10,
            "architecture": [11, 10], "parameter_count": 120,
            "device": "cpu", "model_dtype": "float32", "physical_dtype": "float64",
            "prediction": "normalized physical state delta",
            "forward_interface": "[...,11] float32 -> [...,10] float32",
            "rollout_interface": "physical float64 state/action -> physical float64 state",
            "fit_method": "centered_float64_SVD",
            "objective_definition": OBJECTIVE_DEFINITION}


class DeltaRidge(nn.Module):
    """Fixed learned coefficients with no random initialization or optimizer.

    ``weight[10,11]`` and ``bias[10]`` are float32 CPU buffers. Their 120
    coefficients are learned by the closed-form solve rather than Torch
    gradient updates. Input gradients still pass through the linear map.
    Forward errors deliberately use the shared DynamicModelError contract.
    """

    def __init__(self, config=None):
        super().__init__()
        self.spec = ridge_spec(config)
        self.register_buffer("weight", torch.zeros((10, 11), dtype=torch.float32, device="cpu"))
        self.register_buffer("bias", torch.zeros(10, dtype=torch.float32, device="cpu"))

    def forward(self, normalized_features):
        _no_autocast()
        _tensor(normalized_features, 11, "normalized_features", torch.float32)
        _model_parameters(self)
        result = torch.nn.functional.linear(normalized_features, self.weight, self.bias)
        _tensor(result, 10, "normalized_delta", torch.float32)
        return result


def _array(value, width, name):
    _require(isinstance(value, np.ndarray) and value.dtype == np.float64
             and value.ndim == 2 and value.shape[1] == width and len(value) > 0
             and bool(np.isfinite(value).all()),
             f"{name} requires nonempty finite physical float64 [N,{width}]")


def _normalizer(normalizer):
    _require(isinstance(normalizer, Normalizer), "Ridge requires the schema-bound Normalizer")
    normalizer.validate()


def _weights_hash(weight, bias):
    hashed = hashlib.sha256()
    for value in (weight, bias):
        hashed.update(np.ascontiguousarray(value, dtype="<f4").tobytes())
    return hashed.hexdigest()


def fit_ridge(features, targets, normalizer, alpha):
    """Fit SUM-loss Ridge with an unpenalized intercept, without an inverse.

    Diagnostics named objective/RSS/MSE describe the float64 solution.
    ``stored_model_*`` additionally evaluates its rounded float32 coefficients
    in float64 on the same normalized rows; it is not Torch inference timing
    or an assertion that float32 and float64 arithmetic are identical.
    """
    try:
        _array(features, 11, "features"); _array(targets, 10, "targets")
        _require(len(features) == len(targets), "Feature/target row counts differ")
        _normalizer(normalizer)
        alpha = _scalar(alpha, "alpha")
        x = normalizer.normalize_features(features).astype(np.float64)
        y = normalizer.normalize_delta(targets).astype(np.float64)
        x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
        xc, yc = x - x_mean, y - y_mean
        u, singular, vh = np.linalg.svd(xc, full_matrices=False)
        maximum = float(singular[0]) if len(singular) else 0.0
        threshold = max(xc.shape) * np.finfo(np.float64).eps * maximum
        retained = singular > threshold
        factors = np.zeros_like(singular)
        if alpha == 0:
            np.divide(1.0, singular, out=factors, where=retained)
        else:
            factors = singular / (singular * singular + alpha)
        coefficient = vh.T @ (factors[:, None] * (u.T @ yc))
        intercept = y_mean - x_mean @ coefficient
        _require(bool(np.isfinite(coefficient).all()) and bool(np.isfinite(intercept).all()),
                 "SVD solution is nonfinite")
        with np.errstate(over="ignore", invalid="ignore"):
            weight = coefficient.T.astype(np.float32)
            bias = intercept.astype(np.float32)
        _require(bool(np.isfinite(weight).all()) and bool(np.isfinite(bias).all()),
                 "SVD solution cannot be represented by the float32 model")
        model = DeltaRidge()
        with torch.no_grad():
            model.weight.copy_(torch.from_numpy(weight))
            model.bias.copy_(torch.from_numpy(bias))
        model.eval()
        residual = x @ coefficient + intercept - y
        rss = float(np.sum(residual * residual))
        coefficient_norm = float(np.sum(coefficient * coefficient))
        stored_coefficient = weight.T.astype(np.float64)
        stored_residual = x @ stored_coefficient + bias.astype(np.float64) - y
        stored_rss = float(np.sum(stored_residual * stored_residual))
        stored_norm = float(np.sum(stored_coefficient * stored_coefficient))
        rank = int(retained.sum())
        diagnostics = {
            "fit_count": len(x), "rank": rank, "rank_threshold": float(threshold),
            "condition_estimate": float(singular[0] / singular[rank-1]) if rank else None,
            "condition_full_rank": rank == 11,
            "condition_scope": "centered design, largest/smallest retained singular value; null when rank zero",
            "singular_values": singular.tolist(),
            "alpha": alpha, "objective_definition": OBJECTIVE_DEFINITION,
            "residual_sum_squares": rss, "normalized_residual_mse": rss / y.size,
            "coefficient_squared_norm": coefficient_norm,
            "objective": rss + alpha * coefficient_norm,
            "stored_model_residual_sum_squares": stored_rss,
            "stored_model_normalized_residual_mse": stored_rss / y.size,
            "stored_model_coefficient_squared_norm": stored_norm,
            "stored_model_objective": stored_rss + alpha * stored_norm,
            "objective_numeric_scope": "float64 solve on float32 normalized rows; stored-model metrics use rounded coefficients with float64 arithmetic",
            "model_weights_sha256": _weights_hash(weight, bias),
            "normalized_features_sha256": hashlib.sha256(x.astype("<f8", copy=False).tobytes()).hexdigest(),
            "normalized_targets_sha256": hashlib.sha256(y.astype("<f8", copy=False).tobytes()).hexdigest(),
            "solve_dtype": "float64", "normalized_boundary_dtype": "float32",
            "model_dtype": "float32", "device": "cpu",
            "split_scope": "caller supplies rows; fit API cannot verify train/validation membership",
        }
        _validate_fit_diagnostics(diagnostics, alpha, weight, bias)
        model.fit_diagnostics = deepcopy(diagnostics)
        return model, diagnostics
    except (ValueError, TypeError, KeyError, OverflowError, RuntimeError, np.linalg.LinAlgError) as error:
        if isinstance(error, RidgeError):
            raise
        raise RidgeError(f"Invalid Ridge fit: {error}") from error


def _sha(value, name):
    _require(isinstance(value, str) and len(value) == 64
             and all(c in "0123456789abcdef" for c in value), f"Invalid {name} hash")


def _provenance(value, normalizer):
    expected = {"learning_manifest_sha256", "dataset_identity", "validation_split", "task_scope"}
    _require(isinstance(value, dict) and set(value) == expected,
             "Ridge provenance requires the exact registered fields")
    _require(value["dataset_identity"] == normalizer.metadata["dataset_identity"]
             and value["validation_split"] == "validation"
             and value["task_scope"] == "P3-03-main-ridge", "Invalid Ridge learning/validation provenance")
    _sha(value["learning_manifest_sha256"], "learning manifest")


def _validate_weights(weight, bias):
    for value, shape in ((weight, (10, 11)), (bias, (10,))):
        _require(isinstance(value, np.ndarray) and value.dtype == np.float32
                 and value.shape == shape and bool(np.isfinite(value).all()),
                 "Ridge weights require finite float32 weight[10,11]/bias[10]")


def _validate_fit_diagnostics(value, alpha, weight, bias):
    _require(isinstance(value, dict) and type(value.get("fit_count")) is int
             and value["fit_count"] > 0, "Invalid Ridge fit count")
    _require(type(value.get("rank")) is int and 0 <= value["rank"] <= min(11, value["fit_count"] - 1),
             "Invalid centered design rank")
    _require(value.get("condition_full_rank") is (value["rank"] == 11),
             "Rank/condition diagnostic mismatch")
    _require(_scalar(value.get("alpha"), "fit alpha") == alpha
             and value.get("objective_definition") == OBJECTIVE_DEFINITION
             and value.get("solve_dtype") == "float64"
             and value.get("normalized_boundary_dtype") == "float32"
             and value.get("model_dtype") == "float32" and value.get("device") == "cpu",
             "Ridge fit objective/alpha/numeric boundary mismatch")
    for key in ("rank_threshold", "residual_sum_squares", "normalized_residual_mse",
                "coefficient_squared_norm", "objective", "stored_model_residual_sum_squares",
                "stored_model_normalized_residual_mse", "stored_model_coefficient_squared_norm",
                "stored_model_objective"):
        _scalar(value.get(key), key)
    condition = value.get("condition_estimate")
    _require((condition is None) == (value["rank"] == 0), "Rank/condition estimate mismatch")
    if condition is not None:
        _require(_scalar(condition, "condition_estimate") >= 1, "Invalid design condition estimate")
    for prefix in ("", "stored_model_"):
        rss, norm = value[prefix + "residual_sum_squares"], value[prefix + "coefficient_squared_norm"]
        _require(math.isclose(value[prefix + "objective"], rss + alpha * norm, rel_tol=1e-12, abs_tol=1e-12)
                 and math.isclose(value[prefix + "normalized_residual_mse"], rss / (value["fit_count"] * 10),
                                  rel_tol=1e-12, abs_tol=1e-12), "Ridge diagnostic objective/denominator mismatch")
    _require(value.get("model_weights_sha256") == _weights_hash(weight, bias),
             "Ridge diagnostics belong to different stored coefficients")
    for key in ("normalized_features_sha256", "normalized_targets_sha256"):
        _sha(value.get(key), key)
    # JSON cannot hide NaN, non-native arrays or unsupported scalar objects.
    _json_bytes(value)


def _artifact_hash(value):
    return hashlib.sha256(_json_bytes({k: v for k, v in value.items() if k != "ridge_hash"})).hexdigest()


@dataclass(frozen=True)
class RidgeBundle:
    model: DeltaRidge
    metadata: dict


def save_ridge(path, model, normalizer, config, provenance, *, alpha, fit_diagnostics,
               selection_score=None):
    """Create a fresh, exact-contract, pickle-free local inference bundle."""
    try:
        _normalizer(normalizer); _provenance(provenance, normalizer)
        _config_schema(config, normalizer)
        alpha = _scalar(alpha, "alpha")
        if selection_score is not None:
            selection_score = _scalar(selection_score, "selection_score")
        _require(isinstance(model, DeltaRidge) and model.spec == ridge_spec(config),
                 "Only the registered Ridge delta model can be saved")
        _model_parameters(model)
        _require(set(model.state_dict()) == {"weight", "bias"}, "Unexpected Ridge model state")
        weight, bias = (model.state_dict()[name].detach().cpu().numpy().copy()
                        for name in ("weight", "bias"))
        _validate_weights(weight, bias)
        _validate_fit_diagnostics(fit_diagnostics, alpha, weight, bias)
        _require(getattr(model, "fit_diagnostics", None) == fit_diagnostics,
                 "Supplied diagnostics differ from the fitted model")
        path = Path(path); path.mkdir(parents=True, exist_ok=False)
        with (path / "weights.npz").open("xb") as file:
            np.savez_compressed(file, weight=weight, bias=bias)
        metadata = {
            "format": ARTIFACT_FORMAT, "config": deepcopy(config), "model_spec": ridge_spec(config),
            "learning_schema": deepcopy(normalizer.metadata["schema"]),
            "normalizer_hash": normalizer.metadata["normalizer_hash"],
            "normalizer_stats_hash": normalizer.metadata["stats_hash"],
            "normalizer_train_rows_sha256": normalizer.metadata["train_rows_sha256"],
            "normalizer_train_fit_count": normalizer.metadata["fit_count"],
            "training_provenance": deepcopy(provenance),
            "alpha": alpha, "objective_definition": OBJECTIVE_DEFINITION,
            "fit_diagnostics": deepcopy(fit_diagnostics), "selection_score": selection_score,
            "numpy_version": str(np.__version__), "torch_version": str(torch.__version__),
            "files": {"weights.npz": digest(path / "weights.npz")},
        }
        metadata["ridge_hash"] = _artifact_hash(metadata)
        _write_json(path / "ridge.json", metadata)
        return metadata
    except (KeyError, TypeError, OSError, ValueError, OverflowError, RuntimeError) as error:
        if isinstance(error, RidgeError):
            raise
        raise RidgeError(f"Invalid Ridge save: {error}") from error


def load_ridge(path, normalizer, config, provenance):
    """Refuse bytes, schema, model, normalization, data or config mismatch."""
    try:
        _normalizer(normalizer); _provenance(provenance, normalizer)
        _config_schema(config, normalizer)
        path = Path(path); metadata = _read_json(path / "ridge.json")
        expected_keys = {"format", "config", "model_spec", "learning_schema", "normalizer_hash",
                         "normalizer_stats_hash", "normalizer_train_rows_sha256", "normalizer_train_fit_count",
                         "training_provenance", "alpha", "objective_definition", "fit_diagnostics",
                         "selection_score", "numpy_version", "torch_version", "files", "ridge_hash"}
        _require(set(metadata) == expected_keys and metadata["format"] == ARTIFACT_FORMAT
                 and metadata["ridge_hash"] == _artifact_hash(metadata), "Ridge metadata format/hash mismatch")
        expected_values = {
            "config": config, "model_spec": ridge_spec(config),
            "learning_schema": normalizer.metadata["schema"],
            "normalizer_hash": normalizer.metadata["normalizer_hash"],
            "normalizer_stats_hash": normalizer.metadata["stats_hash"],
            "normalizer_train_rows_sha256": normalizer.metadata["train_rows_sha256"],
            "normalizer_train_fit_count": normalizer.metadata["fit_count"],
            "training_provenance": provenance, "objective_definition": OBJECTIVE_DEFINITION,
            "numpy_version": str(np.__version__), "torch_version": str(torch.__version__),
        }
        _require(all(metadata[key] == value for key, value in expected_values.items()),
                 "Ridge schema/config/normalizer/provenance/runtime mismatch")
        alpha = _scalar(metadata["alpha"], "alpha")
        if metadata["selection_score"] is not None:
            _scalar(metadata["selection_score"], "selection_score")
        _require(isinstance(metadata["files"], dict) and set(metadata["files"]) == {"weights.npz"}
                 and metadata["files"]["weights.npz"] == digest(path / "weights.npz"),
                 "Ridge weight file hash mismatch")
        with (path / "weights.npz").open("rb") as source:
            with np.load(source, allow_pickle=False) as archive:
                _require(len(archive.files) == len(set(archive.files))
                         and set(archive.files) == {"weight", "bias"}, "Unexpected Ridge NPZ members")
                weight, bias = archive["weight"].copy(), archive["bias"].copy()
        _validate_weights(weight, bias)
        _validate_fit_diagnostics(metadata["fit_diagnostics"], alpha, weight, bias)
        model = DeltaRidge(config)
        with torch.no_grad():
            model.weight.copy_(torch.from_numpy(weight))
            model.bias.copy_(torch.from_numpy(bias))
        model.fit_diagnostics = deepcopy(metadata["fit_diagnostics"])
        model.eval()
        return RidgeBundle(model, metadata)
    except (KeyError, TypeError, OSError, ValueError, OverflowError, RuntimeError, BadZipFile, EOFError) as error:
        if isinstance(error, RidgeError):
            raise
        raise RidgeError(f"Invalid Ridge artifact: {error}") from error
