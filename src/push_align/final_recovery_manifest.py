"""Explicit, outcome-aware infrastructure restart of an existing P5 freeze.

The first frozen run is retained independently.  This module permits only the
same scientific population and settings to be frozen again after an approved
recording-only repair; it does not manufacture a fresh held-out population.
No model, simulator, initial reset or result selection is executed here.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from .final_manifest import (
    CONTROLLERS, FAMILY_RECIPE_VERSION, _same, _sha, _source,
    canonical_initial, expected_E1, expected_E2, family_records,
    frozen_source_contract, prior_initial_inventory,
)

PROTOCOL_VERSION = "p5-recording-reservation-v2"
USER_AUTHORIZATION = "저장기 보완 후 같은 조건으로 새 실행 (권장)"
SELECTION_RULE = "all96 restart; retain parent separately; no pooling or best-case replacement"
_RECOVERY_KEYS = {
    "parent_run", "parent_manifest_sha256", "parent_freeze_receipt_sha256",
    "user_authorization", "protocol_version", "selection_rule", "parent_outcomes_observed",
}
# These describe publication, provenance and measured admission, never science.
_PUBLICATION_FIELDS = {
    "source_sha256", "runtime_source_sha256", "source_bundle_map", "source_archives",
    "prior_initial_inventory", "resource_receipt", "integration_receipt", "frozen_at_utc",
    "schema", "status", "final_outcomes_opened", "infrastructure_recovery", "recovery_metadata",
}
_REQUIRED = {
    "families", "application_family", "E2", "E1", "source_sha256",
    "model_identities", "normalizer_identity", "physical_config", "planner_config",
    "profile", "statistics", "prediction_evaluation", "failure_policy",
    "replay_selection_rule", "prior_initial_inventory", "resource_receipt",
    "integration_receipt", "family_generation", "minimum_meaningful_difference",
    "runtime_source_sha256", "source_bundle_map", "candidate_generation",
    "visualization_selection", "visualization", "infrastructure_recovery",
}


def _read_mapping(path, label):
    try:
        value = json.loads(path.read_text())
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError(label + " must be readable JSON") from error
    if type(value) is not dict:
        raise ValueError(label + " must be a mapping")
    return value


def _verify_bindings(root, bindings, label):
    if type(bindings) is not dict or not bindings:
        raise ValueError(label + " requires nonempty explicit SHA256 bindings")
    for relative, digest in bindings.items():
        source = _source(root, relative)
        if (type(digest) is not str or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest) or _sha(source) != digest):
            raise ValueError(label + " hash mismatch: " + relative)


def _parent_contract(root, recovery, output):
    if type(recovery) is not dict or set(recovery) != _RECOVERY_KEYS:
        raise ValueError("Exact explicit infrastructure-recovery authorization required")
    if (recovery["protocol_version"] != PROTOCOL_VERSION
            or recovery["user_authorization"] != USER_AUTHORIZATION
            or recovery["selection_rule"] != SELECTION_RULE
            or recovery["parent_outcomes_observed"] is not True):
        raise ValueError("Outcome-aware approved all96 restart declaration required")
    parent_relative = recovery["parent_run"]
    if type(parent_relative) is not str:
        raise ValueError("Explicit relative parent final run required")
    parts = Path(parent_relative).parts
    if (Path(parent_relative).is_absolute() or len(parts) != 2 or parts[0] != "runs"
            or not parts[1].startswith("p5-final-") or ".." in parts):
        raise ValueError("One top-level parent p5-final run required")
    parent_path = _source(root, parent_relative + "/manifest.json")
    if parent_path.parent == output.parent:
        raise ValueError("Recovery must publish a separate run, never replace the parent")
    receipt_path = _source(root, parent_relative + "/freeze_receipt.json")
    if (_sha(parent_path) != recovery["parent_manifest_sha256"]
            or _sha(receipt_path) != recovery["parent_freeze_receipt_sha256"]):
        raise ValueError("Parent manifest or external freeze receipt SHA256 mismatch")
    receipt = _read_mapping(receipt_path, "Parent freeze receipt")
    receipt_target = Path(receipt.get("path", ""))
    if (not receipt_target.is_absolute() or receipt_target.resolve() != parent_path
            or receipt.get("sha256") != recovery["parent_manifest_sha256"]
            or type(receipt.get("bytes")) is not int or receipt["bytes"] != parent_path.stat().st_size):
        raise ValueError("Parent freeze receipt does not identify exact parent manifest bytes")
    parent = _read_mapping(parent_path, "Parent frozen manifest")
    if (parent.get("schema") != "p5-frozen-final-manifest-v1"
            or parent.get("status") != "FROZEN_BEFORE_FINAL_OUTCOMES"
            or parent.get("final_outcomes_opened") is not False
            or "infrastructure_recovery" in parent):
        raise ValueError("Original independently frozen parent manifest required")
    _verify_bindings(root, parent.get("source_sha256"), "Parent source graph")
    _verify_bindings(root, parent.get("runtime_source_sha256"), "Parent runtime graph")
    for relative, digest in parent["runtime_source_sha256"].items():
        if parent["source_sha256"].get(relative) != digest:
            raise ValueError("Parent runtime graph is not bound to its source graph")
    parent_archives = parent.get("source_archives", {})
    if type(parent_archives) is not dict:
        raise ValueError("Parent document archives must be explicit")
    for original, archived in parent_archives.items():
        if archived != str((parent_path.parent / "source" / original).relative_to(root)):
            raise ValueError("Parent archive path differs from its own frozen source bundle")
    original_contract = frozen_source_contract(root, source_archives=parent_archives)
    for key in ("model_identities", "normalizer_identity", "physical_config", "planner_config", "profile"):
        if not _same(parent.get(key), original_contract[key]):
            raise ValueError("Parent historical contract differs: " + key)
    if any(parent["source_sha256"].get(p) != h for p, h in original_contract["source_sha256"].items()):
        raise ValueError("Parent omits actual historical source/model/config bindings")
    return parent, parent_path, receipt_path


def _validate_inventory(root, output, manifest, parent, parent_path):
    prior = manifest["prior_initial_inventory"]
    if type(prior) is not dict:
        raise ValueError("Complete current prior initial inventory mapping required")
    current = prior_initial_inventory(root, exclude_final_run=prior.get("excluded_final_run"))
    expected_exclusion = str(output.parent.relative_to(root))
    if (current["excluded_final_run"] != expected_exclusion
            or not _same(prior, current) or not prior["source_manifest_count"] or prior["unresolved"]):
        raise ValueError("Current inventory must include parent and exclude only this new run")
    original = parent.get("prior_initial_inventory")
    if (type(original) is not dict or not original.get("source_manifest_count") or original.get("unresolved")
            or original.get("excluded_final_run") != str(parent_path.parent.relative_to(root))):
        raise ValueError("Parent original prior inventory is incomplete or unresolved")
    parent_rel = str(parent_path.relative_to(root))
    parent_manifests = {
        str(p.relative_to(root)): _sha(p)
        for p in sorted(parent_path.parent.rglob("manifest.json"))
        if "source" not in p.relative_to(parent_path.parent).parts
    }
    expected_sources = {**original["source_manifests"], **parent_manifests}
    if (not _same(prior["source_manifests"], expected_sources)
            or prior["source_manifest_count"] != len(expected_sources)):
        raise ValueError("Only the explicitly retained parent manifest subtree may extend original prior sources")
    for relative, digest in original["source_manifests"].items():
        if _sha(_source(root, relative)) != digest:
            raise ValueError("Original prior source manifest changed: " + relative)
    # Verify the old recognized rows by their original source. Nested copies in
    # parent.manifest add repetitions, but may never hide a train/dev overlap.
    old_rows = [row for row in prior["recognized_initials"] if row["source"] not in parent_manifests]
    if not _same(old_rows, original["recognized_initials"]):
        raise ValueError("Original recognized reset inventory changed")
    original_ids = set(original["canonical_identities"])
    if original_ids != {row["canonical_identity"] for row in original["recognized_initials"]}:
        raise ValueError("Parent original canonical inventory differs from recognized rows")
    parent_rows = parent["families"] + [parent["application_family"]]
    parent_ids = {row["canonical_identity"] for row in parent_rows}
    if len(parent_ids) != 13 or parent_ids & original_ids:
        raise ValueError("Parent thirteen families overlap original train/development resets")
    if (set(prior["canonical_identities"]) != original_ids | parent_ids
            or set(row["canonical_identity"] for row in prior["recognized_initials"] if row["source"] in parent_manifests)
                != original_ids | parent_ids):
        raise ValueError("Current canonical inventory must add exactly parent thirteen identities")
    archives = prior["excluded_source_manifest_paths"]
    for relative, digest in original["excluded_source_manifest_paths"].items():
        if archives.get(relative) != digest or _sha(_source(root, relative)) != digest:
            raise ValueError("Previously archived manifest source changed")
    for relative in set(archives) - set(original["excluded_source_manifest_paths"]):
        if not (Path(relative).is_relative_to(parent_path.parent.relative_to(root) / "source")
                or Path(relative).is_relative_to(output.parent.relative_to(root) / "source")):
            raise ValueError("Unrelated archived source manifest appeared after original freeze")
    identities = {row["canonical_identity"] for row in manifest["families"] + [manifest["application_family"]]}
    if identities & set(prior["canonical_identities"]) != parent_ids:
        raise ValueError("The only permitted current-prior overlap is the exact retained parent population")
    return original, parent_ids


def freeze_recovery_manifest(path, manifest, *, integration_pass, resource_capacity, source_root):
    """Exclusively freeze the same scientific comparison after approved repair.

    All original freeze checks remain. The sole independence change is explicit:
    parent outcomes have been observed; the same thirteen resets are deliberately
    reused, and only outcomes of this new run remain unopened at this freeze.
    """
    if (integration_pass is not True or type(resource_capacity) is not dict
            or resource_capacity.get("status") != "FIT_ESTIMATE_ONLY"):
        raise ValueError("Measured integration and current full-budget capacity required before freeze")
    if type(manifest) is not dict or not _REQUIRED <= set(manifest):
        raise ValueError("Incomplete explicit recovery final manifest fields")
    root, output = Path(source_root).resolve(), Path(path).resolve()
    if (not output.is_relative_to(root) or output.name != "manifest.json"
            or output.parent.parent != root / "runs" or not output.parent.name.startswith("p5-final-")):
        raise ValueError("Recovery must belong to its own top-level managed final run")
    parent, parent_path, parent_receipt_path = _parent_contract(root, manifest["infrastructure_recovery"], output)
    for key in set(parent) - _PUBLICATION_FIELDS:
        if key not in manifest or not _same(manifest[key], parent[key]):
            raise ValueError("Recovery scientific declaration differs from parent: " + key)
    extras = set(manifest) - set(parent) - {"infrastructure_recovery"}
    if extras:
        raise ValueError("Undeclared recovery manifest fields: " + str(sorted(extras)))
    archives = manifest.get("source_archives", {})
    if type(archives) is not dict:
        raise ValueError("Document source archives must be a mapping")
    for original, relative in archives.items():
        if relative != str((output.parent / "source" / original).relative_to(root)):
            raise ValueError("Document archive must be the exact new final-run/source/original path")
    contract = frozen_source_contract(root, source_archives=archives)
    for key in ("model_identities", "normalizer_identity", "physical_config", "planner_config", "profile"):
        if not _same(manifest[key], contract[key]):
            raise ValueError("Manifest differs from actual frozen historical contract: " + key)
    original_prior, parent_ids = _validate_inventory(root, output, manifest, parent, parent_path)
    recipe = manifest["family_generation"]
    if (type(recipe) is not dict or set(recipe) != {"recipe_version", "seed", "application_seed", "initial_id_prefix"}
            or recipe["recipe_version"] != FAMILY_RECIPE_VERSION
            or type(recipe["initial_id_prefix"]) is not str or not recipe["initial_id_prefix"]):
        raise ValueError("Exact declared family generation recipe required")
    families, application = family_records(contract["physical_config"], seed=recipe["seed"],
        new_family_seed=recipe["application_seed"], initial_id_prefix=recipe["initial_id_prefix"],
        forbidden_identities=original_prior["canonical_identities"])
    if not _same(manifest["families"], families) or not _same(manifest["application_family"], application):
        raise ValueError("Recovery families differ from the unchanged original seed recipe")
    if manifest["E2"] != expected_E2(families) or manifest["E1"] != expected_E1(families):
        raise ValueError("Frozen E1/E2 entries differ from exact predeclared identities")
    rows = manifest["families"] + [manifest["application_family"]]
    if (len({row["canonical_identity"] for row in rows}) != 13
            or any(row["canonical_identity"] != canonical_initial(row["initial_spec"]) for row in rows)):
        raise ValueError("Duplicate or inconsistent canonical families")
    if set(manifest["model_identities"]) != set(CONTROLLERS[1:]) or not manifest["normalizer_identity"]:
        raise ValueError("All seven models and common normalizer must be bound")
    statistics = manifest["statistics"]
    if (statistics.get("replicates") != 2000 or statistics.get("seed") != 7711
            or statistics.get("unit") != "family_within_stratum_seed_mean_first"):
        raise ValueError("Original bootstrap counts/seed/unit required")
    if (manifest["prediction_evaluation"].get("horizons") != [1, 5, 10]
            or manifest["prediction_evaluation"].get("source") != "all_E2_G_raw_fixed_applied_actions"):
        raise ValueError("Fixed common heldout trajectory K1/5/10 evaluation required")
    if manifest["minimum_meaningful_difference"] != contract["profile"]["final_review"]["minimum_meaningful_improvement"]:
        raise ValueError("Minimum meaningful difference must retain the prior acceptance declaration")
    bindings = manifest["source_sha256"]
    _verify_bindings(root, bindings, "Recovery source/model/config graph")
    if any(bindings.get(relative) != digest for relative, digest in contract["source_sha256"].items()):
        raise ValueError("Required actual historical source/model/config bindings missing or altered")
    for source in (parent_path, parent_receipt_path):
        if bindings.get(str(source.relative_to(root))) != _sha(source):
            raise ValueError("Retained parent manifest and external receipt must be source-bound")
    runtime = manifest["runtime_source_sha256"]
    _verify_bindings(root, runtime, "Recovery runtime graph")
    for relative, digest in parent["runtime_source_sha256"].items():
        if runtime.get(relative) != digest:
            raise ValueError("Recovery removed or changed an original frozen runtime source")
    for relative in set(runtime) - set(parent["runtime_source_sha256"]):
        if Path(relative).suffix != ".py" or Path(relative).parts[0] not in ("src", "scripts", "tests", "tools"):
            raise ValueError("Recovery runtime additions must be explicit new Python modules")
    if any(bindings.get(relative) != digest for relative, digest in runtime.items()):
        raise ValueError("Recovery runtime source graph must be source-bound")
    bundle = manifest["source_bundle_map"]
    if type(bundle) is not dict or not bundle or not set(runtime) <= set(bundle):
        raise ValueError("Explicit recovery source bundle must snapshot every runtime binding")
    for original, archived in bundle.items():
        if archived != str((output.parent / "source" / original).relative_to(root)):
            raise ValueError("Recovery source snapshot must preserve its original relative path")
        snapshot = _source(root, archived)
        if original in archives:
            if bindings.get(archived) != contract["source_sha256"].get(archived):
                raise ValueError("Recovery historical document snapshot changed")
        else:
            original_path = _source(root, original)
            if (bindings.get(original) != _sha(original_path) or bindings.get(archived) != _sha(snapshot)
                    or _sha(original_path) != _sha(snapshot)):
                raise ValueError("Recovery source bundle mapping is not hash-bound to original bytes")
    if any(bundle.get(original) != archived for original, archived in archives.items()):
        raise ValueError("Recovery historical document archives absent from source bundle")
    for key in ("resource_receipt", "integration_receipt"):
        receipt = manifest[key]
        if type(receipt) is not dict or set(receipt) != {"path", "sha256"} or bindings.get(receipt["path"]) != receipt["sha256"]:
            raise ValueError("Receipt not included in frozen source bindings")
        if Path(receipt["path"]).parent != output.parent.relative_to(root):
            raise ValueError("New-run admission receipts must belong to the new run")
        saved = _read_mapping(_source(root, receipt["path"]), key)
        if key == "integration_receipt":
            if saved.get("status") != "PASS" or not saved.get("checks") or not all(value is True for value in saved["checks"].values()):
                raise ValueError("Integration receipt lacks complete verified checks")
        elif not _same(saved.get("capacity"), resource_capacity):
            raise ValueError("Capacity receipt differs from live admitted measurement")
    if (type(resource_capacity.get("free_bytes")) is not int or type(resource_capacity.get("required_bytes")) is not int
            or resource_capacity["required_bytes"] < 2 * 1024**3
            or resource_capacity["free_bytes"] < resource_capacity["required_bytes"]
            or resource_capacity.get("headroom_bytes") != resource_capacity["free_bytes"] - resource_capacity["required_bytes"]):
        raise ValueError("A numeric complete capacity receipt is required")
    bound = deepcopy(manifest)
    bound.update(schema="p5-frozen-final-manifest-v1", status="FROZEN_BEFORE_FINAL_OUTCOMES",
        frozen_at_utc=datetime.now(timezone.utc).isoformat(), final_outcomes_opened=False,
        recovery_metadata={"freeze_scope": "before_new_run_outcomes_only", "parent_outcomes_observed": True,
            "parent_run": manifest["infrastructure_recovery"]["parent_run"],
            "original_prior_canonical_identities": sorted(original_prior["canonical_identities"]),
            "retained_parent_canonical_identities": sorted(parent_ids),
            "new_run_prior_canonical_count": len(manifest["prior_initial_inventory"]["canonical_identities"]),
            "result_policy": SELECTION_RULE})
    payload = (json.dumps(bound, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(Path(path)), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
