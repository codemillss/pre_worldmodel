"""Fresh-process storage benchmark for one preserved300-interval raw prefix.

This reconstructs values from files and never instantiates a physical model or
runs a controller.  Historical observation flags/time/schema were not direct
old raw fields; their provenance is recorded rather than called original bytes.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mujoco
import numpy as np

from push_align.data import _encode_nonfinite
from push_align.environment import Observation, PlanarPushEnv, SnapshotBundle, StepResult
from push_align.final_recording import file_sha256, read_raw, write_raw

SOURCE = ROOT / "runs/p3-05-20261010-01/full_resource/R"
OUT = ROOT / "tmp/p5-g2-native300-benchmark"
REPORT = ROOT / "reports/p5_g2_native300_codec_measurement.json"
MIB = 1024**2
GIB = 1024**3
MAX_OWN_BYTES = 64 * MIB
MAX_G2_BYTES = 256 * MIB


def peak_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (MIB if sys.platform == "darwin" else 1024)


def size(path):
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def g2_files():
    return (list((ROOT / "tmp").glob("*g2*"))
            + list((ROOT / "reports").glob("p5_g2*"))
            + [ROOT / "src/push_align/final_recording.py", ROOT / "tests/test_p5_final_recording.py",
               Path(__file__).resolve()])


def budget_snapshot():
    total = sum(size(path) for path in g2_files())
    return {"own_bytes": size(OUT), "g2_named_artifact_bytes": total,
            "free_bytes": shutil.disk_usage(ROOT).free,
            "accounting_scope": "Existing tmp/*g2*, reports/p5_g2*, codec/test and benchmark; final PDF/document closure accounted separately by root."}


def guard(*, before=False):
    measured = budget_snapshot()
    assert measured["own_bytes"] <= MAX_OWN_BYTES, "Preserve prefix: own64MiB budget reached"
    assert measured["g2_named_artifact_bytes"] + (MAX_OWN_BYTES if before else 0) <= MAX_G2_BYTES, \
        "Preserve prefix: named G2 artifacts256MiB budget reached"
    assert measured["free_bytes"] >= 2 * GIB + (MAX_OWN_BYTES if before else 0), \
        "Preserve prefix: need2GiB free reserve plus initial64MiB allowance"
    return measured


def exact(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def run():
    baseline = guard(before=True)
    assert not OUT.exists() and not REPORT.exists(), "Use a new ID; existing evidence is never overwritten"
    OUT.mkdir()
    paths = [SOURCE / name for name in ("raw_execution.npz", "raw_execution.json.gz", "steps.jsonl", "manifest.json")]
    paths += [ROOT / "src/push_align/final_recording.py", ROOT / "src/push_align/environment.py",
              ROOT / "tests/test_p5_final_recording.py", Path(__file__).resolve()]
    hashes = {str(path.relative_to(ROOT)): file_sha256(path) for path in paths}
    rss = {"fresh_process_after_import_MiB": peak_mib()}
    started = perf_counter()
    with gzip.open(SOURCE / "raw_execution.json.gz", "rt", encoding="utf-8") as stream:
        original = json.load(stream)
    with np.load(SOURCE / "raw_execution.npz", allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    with (SOURCE / "steps.jsonl").open() as stream:
        steps = [json.loads(line) for line in stream if line.strip()]
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    load_wall_s = perf_counter() - started
    rss["after_source_load_MiB"] = peak_mib()
    assert len(steps) == len(original["diagnostics"]) == len(arrays["state"]) == 300
    assert original["diagnostic_nonfinite_paths"] == []
    assert arrays["diagnostic_nonfinite_values"].shape == (0,)
    raw_snapshot = original["initial_snapshot"]
    snapshot = SnapshotBundle(np.asarray(raw_snapshot["integration_state"], np.float64),
        raw_snapshot["model_hash"], raw_snapshot["environment_hash"], raw_snapshot["reference_height"],
        raw_snapshot["rng_state"], np.asarray(raw_snapshot["warning_counts"], np.int64),
        np.asarray(raw_snapshot["warning_lastinfo"], np.int64), raw_snapshot["needs_reset"])
    config = manifest["physical_contract"]["resolved_config"]
    schema = config["schema_version"]
    assert schema == manifest["physical_contract"]["state_schema"]
    skip = config["physics"]["frame_skip"]
    assert arrays["substep_states"].shape == (300 * skip, 10)
    assert arrays["substep_times"].shape == (300 * skip,)
    proxy = SimpleNamespace(config=config, _reference_height=snapshot.reference_height)
    # _validity reads saved arrays/config only.  This proxy has no MuJoCo model
    # or data and cannot restore or execute a physical state.
    results, validity_samples = [], 0
    started = perf_counter()
    for i, (step, diagnostics) in enumerate(zip(steps, original["diagnostics"])):
        assert step["index"] == i and step["physical_valid"] is True
        assert diagnostics["validity_violations"] == [] and diagnostics["terminal_type"] == "completed_interval"
        assert len(diagnostics["samples"]) == skip + 1
        for sample in diagnostics["samples"]:
            assert not PlanarPushEnv._validity(proxy, sample)
            validity_samples += 1
            if "integration_actuator_force_N" in sample:
                assert max(abs(value) for value in sample["integration_actuator_force_N"]) <= \
                    config["pusher"]["actuator_force_limit_abs_N"] + 1e-12
        span = slice(i * skip, (i + 1) * skip)
        observed_time = float(arrays["substep_times"][span][-1])
        assert np.float64(observed_time).tobytes() == np.float64(diagnostics["samples"][-1]["sim_time"]).tobytes()
        assert exact(arrays["state"][i], np.asarray(step["state_after"], np.float64))
        assert exact(arrays["action_requested"][i], np.asarray(step["requested_action"], np.float64))
        assert exact(arrays["action_applied"][i], np.asarray(step["applied_action"], np.float64))
        assert np.float64(arrays["actual_dt"][i]).tobytes() == np.float64(step["actual_dt_s"]).tobytes()
        observation_valid = not PlanarPushEnv._validity(proxy, diagnostics["samples"][-1])
        results.append(StepResult(arrays["action_requested"][i], arrays["action_applied"][i],
            Observation(arrays["state"][i], observed_time, schema, observation_valid),
            float(arrays["actual_dt"][i]), step["physical_valid"], diagnostics,
            arrays["substep_states"][span], arrays["substep_times"][span]))
    reconstructed_wall_s = perf_counter() - started
    rss["after_reconstruction_MiB"] = peak_mib()
    prefix = OUT / "saved-long-R"
    guard()
    started = perf_counter()
    receipt = write_raw(prefix, snapshot, results, xz_preset=6)
    write_wall_s = perf_counter() - started
    rss["after_native_write_MiB"] = peak_mib()
    guard()
    started = perf_counter()
    decoded_snapshot, decoded = read_raw(prefix, expected_sha256=receipt["files_sha256"])
    read_wall_s = perf_counter() - started
    rss["after_native_read_MiB"] = peak_mib()
    assert len(decoded) == 300
    checks = {}
    with np.load(Path(str(prefix) + ".raw.npz"), allow_pickle=False) as saved:
        for key, original_array in arrays.items():
            checks[key] = exact(original_array, saved[key])
            assert checks[key], key
    assert canonical(_encode_nonfinite(asdict(decoded_snapshot))) == canonical(raw_snapshot)
    for i, (old, new) in enumerate(zip(results, decoded)):
        assert canonical(new.diagnostics) == canonical(original["diagnostics"][i])
        for key in ("requested_action", "applied_action", "substep_states", "substep_times"):
            assert exact(getattr(old, key), getattr(new, key))
        assert exact(old.observation.state, new.observation.state)
        assert old.observation.schema_version == new.observation.schema_version == schema
        assert old.observation.observation_valid is new.observation.observation_valid is True
        assert old.transition_valid is new.transition_valid is True
        assert np.float64(old.observation.sim_time).tobytes() == np.float64(new.observation.sim_time).tobytes()
        assert np.float64(old.actual_dt).tobytes() == np.float64(new.actual_dt).tobytes()
    assert all(file_sha256(ROOT / path) == sha for path, sha in hashes.items())
    rss["after_roundtrip_checks_MiB"] = peak_mib()
    final_budget = guard()
    report = {"schema": "p5-g2-native300-codec-measurement-v1",
        "status": "PASS_NATIVE_CODEC_STORAGE_ONLY_NOT_CONTROLLER_BATCH_INTEGRATION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source": str(SOURCE.relative_to(ROOT)),
        "process": {"pid": os.getpid(), "parent_pid": os.getppid(), "python": sys.executable,
                    "platform": platform.platform(), "scope": "fresh standalone interpreter, one codec run only"},
        "native_codec": {"intervals": 300, "substep_samples": 300 * skip,
            "validity_sample_checks": validity_samples, "physical_valid300": True,
            "diagnostic_violations_empty300": True, "xz_preset": 6, "receipt": receipt,
            "new_raw_bytes": sum(receipt["bytes"].values()),
            "new_raw_bytes_per_interval": sum(receipt["bytes"].values()) / 300,
            "original_raw_bytes": sum((SOURCE / name).stat().st_size for name in
                                       ("raw_execution.npz", "raw_execution.json.gz")),
            "source_load_wall_s": load_wall_s, "reconstruction_and_validity_wall_s": reconstructed_wall_s,
            "write_wall_s": write_wall_s, "read_wall_s": read_wall_s, "rss": rss,
            "original_numeric_arrays_bitwise": checks,
            "complete_diagnostics_canonical_json_exact": True,
            "complete_initial_snapshot_json_semantics_exact": True,
            "reconstructed_additional_fields_roundtrip_exact": True},
        "field_provenance": {
            "transition_valid": "Copied from preserved steps.jsonl physical_valid; asserted true for all300 and empty raw validity violations.",
            "observation_sim_time": "Reconstructed from each preserved NPZ final substep time; also matches saved diagnostic final time.",
            "observation_valid": "Derived using unchanged PlanarPushEnv._validity on saved samples/config only; no environment constructed. All60,300 saved samples pass.",
            "observation_schema_version": "Reconstructed from bound original physical-contract/config state schema.",
            "snapshot_warning_dtype": "Old JSON had values only; reconstructed int64, not a claim about original engine array dtype.",
            "claim_boundary": "Old8 NPZ arrays and full diagnostic/snapshot JSON values are preserved; absent old explicit fields are not claimed to have byte-identical original storage."},
        "source_sha256": hashes, "original_inputs_and_codec_test_unchanged": True,
        "budget_before": baseline, "budget_after": final_budget,
        "budget_limits": {"own_output_max_bytes": MAX_OWN_BYTES, "named_G2_artifacts_max_bytes": MAX_G2_BYTES,
                          "free_reserve_bytes": 2 * GIB},
        "new_model_forwards": 0, "new_physics_intervals": 0, "new_optimizer_updates": 0,
        "new_final_families": 0, "P5_01": "HOLD",
        "limits": ["Stored300-interval input serialized by the actual codec; not a new300-interval controller execution.",
            "RSS includes saved diagnostics, reconstructed StepResults, encode/decode and checks; no final model/controller memory or concurrency bound.",
            "One normal R prefix does not establish a full-regime/worst-case storage bound or device portability.",
            "Final batch recording integration and guards remain separate work."]}
    with REPORT.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps({"report": str(REPORT.relative_to(ROOT)), "sha256": file_sha256(REPORT),
        "new_bytes": report["native_codec"]["new_raw_bytes"], "write_s": write_wall_s,
        "read_s": read_wall_s, "peak_rss_MiB": peak_mib(), "new_execution": 0}, ensure_ascii=False))


if __name__ == "__main__":
    def forbidden(*args, **kwargs):
        raise AssertionError("Physical model construction/execution is forbidden in this storage benchmark")
    with patch.object(PlanarPushEnv, "__init__", forbidden), \
         patch.multiple(mujoco, mj_step=forbidden, mj_step1=forbidden, mj_step2=forbidden,
                        mj_forward=forbidden, mj_inverse=forbidden):
        run()
