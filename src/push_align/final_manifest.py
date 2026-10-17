"""Outcomes-blind construction and validation for the existing P5 scope.

These pure functions do not reset a robot or open a final result.  The caller
must pass measured resource/integration evidence before publishing a manifest.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tomllib

import numpy as np

CONTROLLERS = ("G", "R", "S1-11", "S1-23", "S1-47", "SK-11", "SK-23", "SK-47")
STRATA = ("translation", "rotation", "offset")
CANDIDATE_PATH = "configs/p5_freeze_candidate.v1.json"
CANDIDATE_SHA256 = "74b4a1040bbc5913d6569a248a65d6b86268ac0ffad68e0816a2f55a53deaa83"
P4_MANIFEST_PATH = "runs/p4-04-20261013-01/manifest.json"
P4_MANIFEST_SHA256 = "4d874339a340518324f7ec7b8c3e0f0e08c92ef83e86d3e5982941715c8bc59a"
FAMILY_RECIPE_VERSION = "p5-family-strata-v1"


def _same(left, right):
    """JSON-exact values, distinguishing bool/int and integer/float spellings."""
    return json.dumps(left, sort_keys=True, allow_nan=False, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _source(root, relative):
    if type(relative) is not str or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Explicit relative source path required")
    source = (root / relative).resolve()
    if not source.is_relative_to(root) or not source.is_file():
        raise ValueError("Missing or escaping source: " + relative)
    return source


def _physical_config(value):
    return (type(value) is dict and all(type(value.get(key)) is dict for key in ("object", "pusher", "reset"))
            and "size_full_xyz_m" in value["object"] and "radius_m" in value["pusher"]
            and "pusher_surface_gap_m" in value["reset"])


def frozen_source_contract(source_root, *, source_archives=None):
    """Read the exact historical candidate and return its actual seven bindings.

    Candidate and P4 physical/planner manifest hashes are fixed trust anchors.
    All97 candidate-bound files are checked. Only document bytes may be read
    from an explicitly supplied archive; executable/config/model files stay live.
    No model is loaded and no final family is generated here.
    """
    root = Path(source_root).resolve()
    archives = {} if source_archives is None else source_archives
    if type(archives) is not dict:
        raise ValueError("source_archives must be a mapping")
    for original, archived in archives.items():
        if not (original.startswith(("docs/", "reports/")) or Path(original).name.startswith("README")):
            raise ValueError("Only document sources may be archived: " + original)
        _source(root, archived)
    candidate_path = _source(root, CANDIDATE_PATH)
    if _sha(candidate_path) != CANDIDATE_SHA256:
        raise ValueError("Historical freeze candidate changed")
    candidate = json.loads(candidate_path.read_text())
    expected_sources = {**candidate["source_sha256"], CANDIDATE_PATH: CANDIDATE_SHA256,
                        P4_MANIFEST_PATH: P4_MANIFEST_SHA256}
    if len(candidate["source_sha256"]) != 97:
        raise ValueError("Historical97-source candidate required")
    for relative, digest in expected_sources.items():
        path = _source(root, archives.get(relative, relative))
        if _sha(path) != digest:
            raise ValueError("Historical source/model/config changed: " + relative)
    if not set(archives) <= set(expected_sources):
        raise ValueError("Document archive maps an unknown historical source")
    p4 = json.loads(_source(root, P4_MANIFEST_PATH).read_text())
    normalizer_path = candidate["normalizer"]
    normalizer_metadata = normalizer_path + "/normalizer.json"
    normalizer_stats = normalizer_path + "/statistics.npz"
    normalizer = json.loads(_source(root, normalizer_metadata).read_text())
    normalizer_hash = normalizer["normalizer"]["normalizer_hash"]
    if (normalizer_hash != p4["normalizer_hash"] or normalizer["npz"] != "statistics.npz"
            or normalizer["npz_sha256"] != expected_sources[normalizer_stats]):
        raise ValueError("Common normalizer metadata/statistics differ")
    models = {}
    for selected in candidate["learned_models"]:
        controller, folder = selected["final_controller_id"], selected["path"]
        if controller in models:
            raise ValueError("Duplicate selected model controller")
        metadata_name = "ridge.json" if controller == "R" else "checkpoint.json"
        metadata_path = folder + "/" + metadata_name
        metadata = json.loads(_source(root, metadata_path).read_text())
        model_hash = selected["ridge_hash" if controller == "R" else "checkpoint_hash"]
        if (metadata.get("ridge_hash" if controller == "R" else "checkpoint_hash") != model_hash
                or metadata.get("normalizer_hash") != normalizer_hash
                or not _same(metadata["files"], selected["files_sha256"])):
            raise ValueError("Selected model identity/normalizer/files differ: " + controller)
        files = {folder + "/" + name: digest for name, digest in selected["files_sha256"].items()}
        if any(expected_sources.get(relative) != digest for relative, digest in files.items()):
            raise ValueError("Model files not bound to candidate: " + controller)
        models[controller] = {"path": folder, "model_hash": model_hash,
            "metadata_path": metadata_path, "metadata_sha256": expected_sources[metadata_path],
            "files_sha256": files}
    if set(models) != set(CONTROLLERS[1:]):
        raise ValueError("Exact selected seven-controller set required")
    profile_path = "configs/acceptance.v1.toml"
    with _source(root, profile_path).open("rb") as stream:
        profile = tomllib.load(stream)
    if (not _same(candidate["task"], p4["physical"]["task"])
            or not _same(candidate["task"], p4["planner"]["task"])
            or p4["planner"]["planner"]["candidate_count"] != candidate["common_planning"]["N"]
            or p4["planner"]["planner"]["horizon"] != candidate["common_planning"]["H"]):
        raise ValueError("Historical physical/planner/task scope mismatch")
    return {"model_identities": models,
        "normalizer_identity": {"path": normalizer_path, "normalizer_hash": normalizer_hash,
            "metadata_path": normalizer_metadata, "metadata_sha256": expected_sources[normalizer_metadata],
            "statistics_path": normalizer_stats, "statistics_sha256": expected_sources[normalizer_stats]},
        "physical_config": deepcopy(p4["physical"]), "planner_config": deepcopy(p4["planner"]),
        "profile": profile,
        "source_sha256": {archives.get(relative, relative): digest for relative, digest in expected_sources.items()}}


def canonical_initial(initial):
    """Deduplicate rectangle poses at 1e-12 SI precision, including pi symmetry.

    This deliberately conservative identity is for leakage/duplicate exclusion,
    not a claim that separate uniform draws form an industrial population.
    """
    keys = ("object_x", "object_y", "object_yaw", "pusher_x", "pusher_y")
    values = []
    for key in keys:
        value = initial[key]
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError("Finite explicit initial coordinates required: " + key)
        if key == "object_yaw":
            value = .5 * math.atan2(math.sin(2 * value), math.cos(2 * value))
            if round(value, 12) == round(math.pi / 2, 12):
                value = -math.pi / 2
        values.append(round(float(value), 12) or 0.)
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def extract_prior_initials(value, *, source, path="$", found=None, unresolved=None,
                           physical=None):
    """Read declared explicit resets and bound initial observations in manifests."""
    found = [] if found is None else found
    if isinstance(value, dict):
        # A local archived physical config takes precedence; never silently use
        # today's physical parameters to complete an old omitted pusher pose.
        local = value.get("physical_contract", {})
        candidates = ([local.get("resolved_config")] if type(local) is dict else [])
        candidates += [value.get(key) for key in ("resolved_config", "physical_config", "physical", "config")]
        for candidate in candidates:
            if _physical_config(candidate):
                physical = candidate
                break
        keys = ("object_x", "object_y", "object_yaw", "pusher_x", "pusher_y")
        if "initial_spec" in value and not (type(value["initial_spec"]) is dict
                and all(key in value["initial_spec"] for key in keys[:3])):
            issue = {"source": source, "json_path": path + ".initial_spec",
                     "reason": "unsupported_initial_spec_representation", "value": value["initial_spec"]}
            if unresolved is None:
                raise ValueError("Initial specification representation unresolved: " + source + path)
            unresolved.append(issue)
        if all(key in value for key in keys[:3]):
            initial = {key: value.get(key) for key in keys}
            resolution = "explicit_five_coordinates"
            finite_object = all(type(initial[key]) in (int, float) and math.isfinite(initial[key]) for key in keys[:3])
            if initial['pusher_x'] is None or initial['pusher_y'] is None:
                if finite_object and physical is not None and initial['pusher_x'] is None and initial['pusher_y'] is None:
                    yaw = initial['object_yaw']
                    distance = (physical['object']['size_full_xyz_m'][0] / 2
                        + physical['pusher']['radius_m'] + physical['reset']['pusher_surface_gap_m'])
                    offset = physical['reset']['initial_pusher_tangent_offset_m']
                    initial['pusher_x'] = initial['object_x'] - distance * math.cos(yaw) - offset * math.sin(yaw)
                    initial['pusher_y'] = initial['object_y'] - distance * math.sin(yaw) + offset * math.cos(yaw)
                    resolution = "derived_from_bound_local_reset_config"
                else:
                    issue = {"source": source, "json_path": path, "reason": "unresolved_pusher_coordinates",
                             "initial_spec": initial}
                    if unresolved is None:
                        raise ValueError("Initial reset coordinates unresolved: " + source + path)
                    unresolved.append(issue)
                    initial = None
            if initial is not None:
                if not all(type(initial[key]) in (int, float) and math.isfinite(initial[key]) for key in keys):
                    issue = {"source": source, "json_path": path, "reason": "unresolved_nonfinite_or_null_coordinates",
                             "initial_spec": initial}
                    if unresolved is None:
                        raise ValueError("Initial coordinates unresolved: " + source + path)
                    unresolved.append(issue)
                    initial = None
            if initial is not None:
                found.append({"source": source, "json_path": path, "initial_spec": initial,
                              "canonical_identity": canonical_initial(initial), "resolution": resolution})
        state = value.get("initial_state")
        if isinstance(state, list) and len(state) == 10:
            initial = dict(zip(keys, (state[0], state[1], state[2], state[6], state[7])))
            if all(type(initial[key]) in (int, float) and math.isfinite(initial[key]) for key in keys):
                found.append({"source": source, "json_path": path + ".initial_state",
                              "initial_spec": initial, "canonical_identity": canonical_initial(initial),
                              "resolution": "explicit_ten_field_initial_observation"})
            elif unresolved is not None:
                unresolved.append({"source": source, "json_path": path + ".initial_state",
                                   "reason": "unresolved_initial_observation", "initial_spec": initial})
            else:
                raise ValueError("Initial observation unresolved: " + source + path)
        elif "initial_state" in value:
            issue = {"source": source, "json_path": path + ".initial_state",
                     "reason": "unsupported_initial_state_representation", "value": state}
            if unresolved is None:
                raise ValueError("Initial observation representation unresolved: " + source + path)
            unresolved.append(issue)
        for key, child in value.items():
            extract_prior_initials(child, source=source, path=path + "." + key, found=found,
                                   unresolved=unresolved, physical=physical)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            extract_prior_initials(child, source=source, path=f"{path}[{index}]", found=found,
                                   unresolved=unresolved, physical=physical)
    return found


def prior_initial_inventory(source_root, *, exclude_final_run=None):
    """Recompute every current runs/**/manifest.json and every recognized pose.

    Source archive copies are listed separately. The only other allowed
    exclusion is one top-level p5-final-* run, bound to the output at freeze.
    Unknown/null recognized pose forms are retained as unresolved and block
    freezing. This is an inventory of declared manifest poses, not a claim that
    an unrecorded reset in arbitrary historical source code has been recovered.
    """
    root = Path(source_root).resolve()
    excluded = None
    if exclude_final_run is not None:
        candidate = Path(exclude_final_run)
        if candidate.is_absolute():
            try:
                candidate = candidate.resolve().relative_to(root)
            except ValueError as error:
                raise ValueError("Final run exclusion escapes source root") from error
        parts = candidate.parts
        if len(parts) != 2 or parts[0] != "runs" or not parts[1].startswith("p5-final-"):
            raise ValueError("Only one top-level p5-final-* run may be excluded")
        excluded = str(candidate)
    sources, archived, recognized, unresolved = {}, {}, [], []
    paths = sorted((root / "runs").rglob("manifest.json"))
    for path in paths:
        relative = str(path.relative_to(root))
        if not path.resolve().is_relative_to(root):
            raise ValueError("Historical manifest escapes source root: " + relative)
        if "source" in path.relative_to(root).parts[2:]:
            archived[relative] = _sha(path)
            continue
        if excluded is not None and path.is_relative_to(root / excluded):
            continue
        sources[relative] = _sha(path)
        try:
            value = json.loads(path.read_text())
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError("Unreadable historical manifest; cannot silently omit: " + relative) from error
        extract_prior_initials(value, source=relative, found=recognized, unresolved=unresolved)
    return {"schema": "p5-prior-initial-inventory-v1", "scan_glob": "runs/**/manifest.json",
        "excluded_source_manifest_paths": archived, "excluded_final_run": excluded,
        "source_manifest_count": len(sources), "source_manifests": sources,
        "recognized_initials": recognized,
        "canonical_identities": sorted({row["canonical_identity"] for row in recognized}),
        "unresolved": unresolved,
        "recognition_scope": "Explicit object_x/object_y/object_yaw poses (missing pusher XY only from local archived physical reset), and ten-field initial_state observations; unrecorded arbitrary historical resets are outside this inventory."}


def family_records(physical, *, seed, initial_id_prefix, new_family_seed,
                   forbidden_identities=()):
    """Build fixed strata and one same-range application family, never resample."""
    if type(seed) is not int or type(new_family_seed) is not int or seed == new_family_seed:
        raise ValueError("Distinct integer family seeds required")
    if not initial_id_prefix:
        raise ValueError("Family ID prefix required")
    seen = set(forbidden_identities)
    distance = (physical["object"]["size_full_xyz_m"][0] / 2
                + physical["pusher"]["radius_m"] + physical["reset"]["pusher_surface_gap_m"])
    rows = []
    for stratum_index, stratum in enumerate(STRATA):
        for rank in range(4):
            key = [seed, stratum_index, rank]
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(key)))
            x, y = float(rng.uniform(-.02, .02)), float(rng.uniform(-.01, .01))
            sign = 1 if rank < 2 else -1
            yaw = (0. if stratum == "translation" else
                   float(np.deg2rad(sign * rng.uniform(5, 15))) if stratum == "rotation" else
                   float(np.deg2rad(rng.uniform(-15, 15))))
            offset = sign * float(rng.uniform(.01, .015)) if stratum == "offset" else 0.
            normal, tangent = np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])
            pusher = np.array([x, y]) - distance * normal + offset * tangent
            initial = dict(object_x=x, object_y=y, object_yaw=yaw,
                           pusher_x=float(pusher[0]), pusher_y=float(pusher[1]))
            identity = canonical_initial(initial)
            if identity in seen:
                raise ValueError("Initial identity collides; stop without outcome-based replacement")
            seen.add(identity)
            rows.append({"family_id": f"{initial_id_prefix}-{stratum}-{rank:02d}",
                         "episode_id": f"{initial_id_prefix}-{stratum}-{rank:02d}",
                         "episode_index": 4 * stratum_index + rank, "stratum": stratum,
                         "rank_within_stratum": rank, "rng_key": key, "initial_spec": initial,
                         "tangent_offset_m": offset, "canonical_identity": identity})
    # The application family is drawn by a separate stream, before any final
    # outcome. Its expected effect is uncertainty, never promised improvement.
    rng = np.random.Generator(np.random.PCG64(new_family_seed))
    x, y = float(rng.uniform(-.02, .02)), float(rng.uniform(-.01, .01))
    yaw, offset = float(np.deg2rad(rng.uniform(-15, 15))), float(rng.uniform(.01, .015))
    pusher = np.array([x, y]) - distance * np.array([math.cos(yaw), math.sin(yaw)]) + offset * np.array([-math.sin(yaw), math.cos(yaw)])
    initial = dict(object_x=x, object_y=y, object_yaw=yaw, pusher_x=float(pusher[0]), pusher_y=float(pusher[1]))
    identity = canonical_initial(initial)
    if identity in seen:
        raise ValueError("Application family collides; no replacement")
    application = {"family_id": initial_id_prefix + "-application-00", "episode_id": initial_id_prefix + "-application-00",
                   "episode_index": 12, "stratum": "offset", "rng_key": [new_family_seed],
                   "initial_spec": initial, "tangent_offset_m": offset, "canonical_identity": identity,
                   "predeclared_expectation": "Same physical/task/range and unchanged learned models, planner, costs, action limits, recording and task criteria should execute on the new settled initial state without parameter adjustment. Contact timing, selected actions and success/termination can change and are measured after the freeze; no outcome is promised. This one same-range family tests application of the frozen implementation, not population generalization, superiority or an additional eight replay runs.",
                   "changes": "Only initial pose/pusher coordinates and independent RNG family key; no parameter adjustment"}
    return rows, application


def expected_E2(families):
    if len(families) != 12 or {row["stratum"] for row in families} != set(STRATA):
        raise ValueError("Original twelve-family/three-stratum scope required")
    if (len({row['family_id'] for row in families}) != 12
            or len({row['episode_id'] for row in families}) != 12
            or sorted(row['episode_index'] for row in families) != list(range(12))
            or any(sum(row['stratum'] == s for row in families) != 4 for s in STRATA)):
        raise ValueError("Unique identities and exactly four families per stratum required")
    return [{"family_id": row["family_id"], "episode_id": row["episode_id"],
             "episode_index": row["episode_index"], "stratum": row["stratum"],
             "controller_id": controller, "cell_id": row["episode_id"] + "/" + controller}
            for row in families for controller in CONTROLLERS]


def expected_E1(families):
    selected = []
    for stratum in STRATA:
        rows = sorted((row for row in families if row["stratum"] == stratum), key=lambda row: row["family_id"])
        if len(rows) != 4:
            raise ValueError("Four families per stratum required")
        for row in rows[:2]:
            for tick in (0, 29):
                selected.append({"snapshot_id": row["family_id"] + f"/G/tick-{tick:03d}",
                                 "family_id": row["family_id"], "episode_index": row["episode_index"],
                                 "stratum": stratum, "controller_id": "G", "tick": tick,
                                 "candidates": 16, "horizon": 5, "physical_repeats": 3,
                                 "missing_policy": "INCOMPLETE without replacement or postterminal generation"})
    return selected


def replay_union(result_rows):
    """Predeclared ID order, union across stratum and observed terminal type."""
    selected = []
    for controller in CONTROLLERS:
        rows = [row for row in result_rows if row["controller_id"] == controller]
        if len(rows) != 12 or len({row["family_id"] for row in rows}) != 12:
            raise ValueError("Complete twelve-family population required before replay selection")
        if any(row.get("execution_status") != "completed" or row.get("termination") in (None, "infrastructure_error", "development_budget") for row in rows):
            raise ValueError("Unresolved infrastructure/development outcome cannot be a complete final population")
        groups = ([row for row in rows if row["stratum"] == stratum] for stratum in STRATA)
        chosen = {min(group, key=lambda row: row["episode_id"])["cell_id"] for group in groups}
        for terminal in sorted({row["termination"] for row in rows}):
            group = [row for row in rows if row["termination"] == terminal]
            chosen.add(min(group, key=lambda row: row["episode_id"])["cell_id"])
        selected.extend(row for row in sorted(rows, key=lambda row: row["episode_id"]) if row["cell_id"] in chosen)
    return deepcopy(selected)


def freeze_manifest(path, manifest, *, integration_pass, resource_capacity, source_root):
    """Exclusive publication, with an external receipt; never edits a prior freeze."""
    if (integration_pass is not True or type(resource_capacity) is not dict
            or resource_capacity.get("status") != "FIT_ESTIMATE_ONLY"):
        raise ValueError("Measured integration and current full-budget capacity required before freeze")
    if type(manifest) is not dict:
        raise ValueError("Explicit final manifest mapping required")
    needed = {'families', 'application_family', 'E2', 'E1', 'source_sha256',
              'model_identities', 'normalizer_identity', 'physical_config', 'planner_config',
              'profile', 'statistics', 'prediction_evaluation', 'failure_policy',
              'replay_selection_rule', 'prior_initial_inventory', 'resource_receipt',
              'integration_receipt', 'family_generation', 'minimum_meaningful_difference'}
    if not needed <= set(manifest):
        raise ValueError("Incomplete final manifest fields: " + str(sorted(needed - set(manifest))))
    root, output = Path(source_root).resolve(), Path(path).resolve()
    if not output.is_relative_to(root):
        raise ValueError("Final manifest must belong to its explicit source root")
    archives = manifest.get("source_archives", {})
    if type(archives) is not dict:
        raise ValueError("Document source archives must be a mapping")
    for original, relative in archives.items():
        expected = str((output.parent / "source" / original).relative_to(root))
        if relative != expected:
            raise ValueError("Document archive must be the exact final-run/source/original path")
    contract = frozen_source_contract(root, source_archives=archives)
    for key in ("model_identities", "normalizer_identity", "physical_config", "planner_config", "profile"):
        if not _same(manifest[key], contract[key]):
            raise ValueError("Manifest differs from actual frozen historical contract: " + key)
    prior = manifest["prior_initial_inventory"]
    if type(prior) is not dict:
        raise ValueError("Complete prior initial inventory mapping required")
    current_prior = prior_initial_inventory(root, exclude_final_run=prior.get("excluded_final_run"))
    if current_prior["excluded_final_run"] is not None and output.parent != (root / current_prior["excluded_final_run"]).resolve():
        raise ValueError("Excluded final run must be the actual new manifest output directory")
    if not _same(prior, current_prior) or not prior["source_manifest_count"] or prior["unresolved"]:
        raise ValueError("Prior manifest inventory is omitted, stale, tampered or unresolved")
    recipe = manifest["family_generation"]
    if (type(recipe) is not dict or set(recipe) != {"recipe_version", "seed", "application_seed", "initial_id_prefix"}
            or recipe["recipe_version"] != FAMILY_RECIPE_VERSION
            or type(recipe["initial_id_prefix"]) is not str or not recipe["initial_id_prefix"]):
        raise ValueError("Exact declared family generation recipe required")
    expected_families, expected_application = family_records(contract["physical_config"],
        seed=recipe["seed"], new_family_seed=recipe["application_seed"],
        initial_id_prefix=recipe["initial_id_prefix"], forbidden_identities=prior["canonical_identities"])
    if (not _same(manifest["families"], expected_families)
            or not _same(manifest["application_family"], expected_application)):
        raise ValueError("Family poses/geometry/strata/ranges/identities differ from exact seed recipe")
    if manifest['E2'] != expected_E2(manifest['families']) or manifest['E1'] != expected_E1(manifest['families']):
        raise ValueError("Frozen E1/E2 entries differ from exact predeclared identities")
    identities = [row['canonical_identity'] for row in manifest['families'] + [manifest['application_family']]]
    if len(set(identities)) != 13 or any(row['canonical_identity'] != canonical_initial(row['initial_spec']) for row in manifest['families'] + [manifest['application_family']]):
        raise ValueError("Duplicate or inconsistent canonical families")
    if (not prior.get('source_manifest_count') or prior.get('unresolved')
            or set(identities) & set(prior['canonical_identities'])):
        raise ValueError("Prior initial inventory incomplete or canonical overlap")
    if set(manifest['model_identities']) != set(CONTROLLERS[1:]) or not manifest['normalizer_identity']:
        raise ValueError("All seven models and common normalizer must be bound")
    statistics = manifest['statistics']
    if statistics.get('replicates') != 2000 or statistics.get('seed') != 7711 or statistics.get('unit') != 'family_within_stratum_seed_mean_first':
        raise ValueError("Original bootstrap counts/seed/unit required")
    if manifest['prediction_evaluation'].get('horizons') != [1,5,10] or manifest['prediction_evaluation'].get('source') != 'all_E2_G_raw_fixed_applied_actions':
        raise ValueError("Fixed common heldout trajectory K1/5/10 evaluation required")
    if manifest['minimum_meaningful_difference'] != contract['profile']['final_review']['minimum_meaningful_improvement']:
        raise ValueError("Minimum meaningful difference must retain the prior acceptance declaration")
    bindings = manifest['source_sha256']
    if type(bindings) is not dict or not bindings:
        raise ValueError("Nonempty source/model/config bindings required")
    if any(bindings.get(relative) != digest for relative, digest in contract["source_sha256"].items()):
        raise ValueError("Required actual historical source/model/config bindings missing or altered")
    for relative, digest in bindings.items():
        source = _source(root, relative)
        if (type(digest) is not str or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest) or _sha(source) != digest):
            raise ValueError("Source/model/config hash mismatch: " + relative)
    for key in ('resource_receipt', 'integration_receipt'):
        receipt = manifest[key]
        if bindings.get(receipt['path']) != receipt['sha256']:
            raise ValueError("Receipt not included in frozen source bindings")
        saved = json.loads((root / receipt['path']).read_text())
        if key == 'integration_receipt':
            if saved.get('status') != 'PASS' or not saved.get('checks') or not all(value is True for value in saved['checks'].values()):
                raise ValueError("Integration receipt lacks complete verified checks")
        else:
            if saved.get('capacity') != resource_capacity:
                raise ValueError("Capacity receipt differs from live admitted measurement")
    if (type(resource_capacity.get('free_bytes')) is not int or type(resource_capacity.get('required_bytes')) is not int
            or resource_capacity['required_bytes'] < 2 * 1024**3
            or resource_capacity['free_bytes'] < resource_capacity['required_bytes']
            or resource_capacity.get('headroom_bytes') != resource_capacity['free_bytes'] - resource_capacity['required_bytes']):
        raise ValueError("A numeric complete capacity receipt is required")
    bound = deepcopy(manifest)
    bound.update(schema="p5-frozen-final-manifest-v1", status="FROZEN_BEFORE_FINAL_OUTCOMES",
                 frozen_at_utc=datetime.now(timezone.utc).isoformat(), final_outcomes_opened=False)
    payload = (json.dumps(bound, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path = Path(path)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
