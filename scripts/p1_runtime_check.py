"""P1-01 evidence: headless scene creation and one deterministic CPU update.

Run from the project root with PYTHONPATH=src. This checks the runtime path;
it does not establish pushing performance or train a project checkpoint.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tomllib
from typing import Any


DEFAULT_SEED = 20260928


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_number(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _physical_memory() -> dict[str, Any]:
    """One ordinary read only; permission failure stays unknown."""
    result: dict[str, Any] = {
        "total_bytes": None,
        "status": "unknown",
        "source": "sysctl hw.memsize",
    }
    if platform.system() != "Darwin":
        result["reason"] = "This check only reads macOS hw.memsize."
        return result
    try:
        probe = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if probe.returncode != 0:
            result["reason"] = probe.stderr.strip() or f"exit {probe.returncode}"
            return result
        total = int(probe.stdout.strip())
        if total <= 0:
            raise ValueError("hw.memsize did not return a positive byte count")
        result.update(total_bytes=total, status="known")
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    return result


def _run_checks(report: dict[str, Any], seed: int, config_path: Path | None = None) -> None:
    # Deferred imports let missing/broken dependencies produce a FAIL report.
    import mujoco
    import numpy as np
    import torch
    from push_align import PlanarPushEnv
    from push_align.config import CONFIG_PATH

    checks = report["checks"]
    checks["imports_succeeded"] = True
    report["library_versions"] = {
        "numpy": np.__version__, "mujoco": mujoco.__version__, "torch": torch.__version__,
    }
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    report["determinism"] = {
        "seed": seed,
        "seeded_libraries": ["random", "numpy", "torch"],
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_num_threads": torch.get_num_threads(),
        "scope": "same runtime and device; cross-version equality is not promised",
    }

    resolved_config_path = (config_path or CONFIG_PATH).expanduser().resolve()
    report["config_path"] = str(resolved_config_path)
    config_bytes = resolved_config_path.read_bytes()
    report["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    config = tomllib.loads(config_bytes.decode("utf-8"))
    env = PlanarPushEnv(config=config)
    observation = env.reset()
    report["physics"] = {
        "headless": True,
        "model_nq": int(env.model.nq), "model_nv": int(env.model.nv), "model_nu": int(env.model.nu),
        "config_id": env.config["config_id"],
        "schema_version": observation.schema_version,
        "physics_dt_s": float(env.model.opt.timestep),
        "frame_skip": int(env.config["physics"]["frame_skip"]),
        "action_dt_s": float(env.action_dt),
        "initial_sim_time_s": _finite_number(observation.sim_time),
    }
    checks["physics_model_created"] = bool(env.model.nq > 0 and env.model.nv > 0)
    checks["initial_observation_finite"] = bool(np.isfinite(observation.state).all())
    checks["initial_observation_valid"] = bool(observation.observation_valid)

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 8), torch.nn.SiLU(), torch.nn.Linear(8, 2)
    ).to(device="cpu", dtype=torch.float32)
    inputs = torch.tensor([[1.0, -1.0], [0.5, 0.25]], device="cpu", dtype=torch.float32)
    targets = torch.tensor([[0.25, -0.5], [-0.25, 0.75]], device="cpu", dtype=torch.float32)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    checks["loss_finite"] = bool(torch.isfinite(loss).item())
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    checks["gradients_present"] = bool(all(g is not None for g in grads))
    checks["gradients_finite"] = bool(
        checks["gradients_present"] and all(torch.isfinite(g).all().item() for g in grads)
    )
    gradient_norm = (
        float(torch.linalg.vector_norm(torch.cat([g.reshape(-1) for g in grads])).item())
        if checks["gradients_present"] else float("nan")
    )
    checks["gradient_norm_positive"] = bool(math.isfinite(gradient_norm) and gradient_norm > 0)
    if checks["loss_finite"] and checks["gradients_finite"] and checks["gradient_norm_positive"]:
        optimizer.step()
        checks["optimizer_step_executed"] = True
    after = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    parameter_delta = float(torch.linalg.vector_norm(after - before).item())
    checks["parameters_finite"] = bool(torch.isfinite(after).all().item())
    checks["parameter_delta_positive"] = bool(math.isfinite(parameter_delta) and parameter_delta > 0)
    checks["cpu_float32"] = bool(
        all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters())
    )
    report["learning"] = {
        "architecture": [2, 8, 2], "activation": "SiLU",
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
        "parameter_count": int(before.numel()),
        "optimizer": "SGD", "learning_rate": 0.1,
        "optimizer_steps": int(checks["optimizer_step_executed"]),
        "loss_before_update": _finite_number(loss.detach().item()),
        "gradient_l2_norm": _finite_number(gradient_norm),
        "parameter_delta_l2_norm": _finite_number(parameter_delta),
        "checkpoint_written": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--config", type=Path, help="TOML configuration; defaults to the project's pilot config.")
    parser.add_argument("--output", type=Path, help="New JSON file; existing files are never overwritten.")
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be an integer in [0, 2**32).")
    output_file = None
    if args.output is not None:
        try:
            output_file = args.output.open("x", encoding="utf-8")
        except OSError as exc:
            parser.error(f"Cannot create a new output file: {exc}")

    report: dict[str, Any] = {
        "record_kind": "executed_runtime_check", "report_schema": "p1-runtime-check-v1",
        "task_id": "P1-01", "status": "FAIL", "started_at": _utc_now(),
        "command": [sys.executable, *sys.argv],
        "command_environment": {"PYTHONPATH": os.environ.get("PYTHONPATH")},
        "working_directory": str(Path.cwd()),
        "python": sys.version, "executable": sys.executable,
        "executable_realpath": str(Path(sys.executable).resolve()),
        "platform": platform.platform(), "machine_architecture": platform.machine(),
        "physical_memory": _physical_memory(), "requested_device": "cpu", "seed": args.seed,
        "checks": dict.fromkeys([
            "imports_succeeded", "physics_model_created", "initial_observation_finite",
            "initial_observation_valid", "loss_finite", "gradients_present", "gradients_finite",
            "gradient_norm_positive", "optimizer_step_executed", "parameters_finite",
            "parameter_delta_positive", "cpu_float32",
        ], False),
    }
    try:
        _run_checks(report, args.seed, args.config)
        report["status"] = "PASS" if all(report["checks"].values()) else "FAIL"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    report["finished_at"] = _utc_now()
    report["failed_checks"] = [key for key, value in report["checks"].items() if not value]
    try:
        payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        if output_file is not None:
            output_file.write(payload + "\n")
            output_file.flush()
        print(payload)
    except (OSError, ValueError) as exc:
        print(f"FAIL: could not write the runtime evidence: {exc}", file=sys.stderr)
        return 1
    finally:
        if output_file is not None:
            output_file.close()
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
