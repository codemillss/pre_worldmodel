"""Read-only source-bound exact regeneration of the complete E1 statistics.

The original E1 adoption validates all576 raw prefixes and84 diagnostics first.
This reviewer independently reloads the ledger and regenerates the prescribed
aggregation using the same frozen analysis algorithm. It is not an independent
statistical estimator or a new model/physics execution.
"""
from __future__ import annotations

import json
from pathlib import Path
import resource
import sys
import zipfile

import numpy as np

from .final_driver import typed_digest
from .final_e1_statistics import analyze_E1
from .final_recording import file_sha256

REQUIRED_FILES = ('summary.json', 'bootstrap.npz', 'reproduction.json')
SOURCE_FILES = ('manifest.json', 'E1/summary.json', 'E1/stage_receipt.json', 'E1/comparison_ledger.json')
MAX_EXPANDED_NPZ_BYTES = 64*1024**2
MAX_RSS_BYTES = 6*1024**3


class E1StatisticsReviewError(ValueError):
    pass


def _need(condition, message):
    if not condition:
        raise E1StatisticsReviewError('E1 statistics review: '+message)


def _read(path):
    try:
        with Path(path).open() as stream:
            value = json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
        return value
    except (OSError, ValueError) as error:
        raise E1StatisticsReviewError('Missing/malformed E1 statistics evidence: '+str(path)) from error


def _hash(path, expected):
    _need(type(expected) is str and len(expected) == 64 and Path(path).is_file()
          and file_sha256(path) == expected, 'Missing/changed evidence: '+str(path))


def _memory():
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform == 'darwin' else 1024)
    _need(peak <= MAX_RSS_BYTES, 'RSS limit exceeded; preserve evidence and reestimate')


def _json_same(left, right):
    return typed_digest(left) == typed_digest(right)


def _validate(run, m):
    _memory()
    manifest_path = run/'manifest.json'
    _hash(manifest_path, _read(run/'freeze_receipt.json')['sha256'])
    actual_m = _read(manifest_path)
    _need(_json_same(actual_m, m), 'Caller uses another frozen manifest')
    _need(m.get('schema') == 'p5-frozen-final-manifest-v1'
          and m.get('status') == 'FROZEN_BEFORE_FINAL_OUTCOMES', 'A frozen final manifest is required')
    settings = m['statistics']
    _need(type(settings.get('replicates')) is int and settings['replicates'] == 2000
          and type(settings.get('seed')) is int and settings['seed'] == 7711,
          'Retain fixed2000 family resamples and seed7711')
    folder = run/'e1_statistics'
    receipt_path = folder/'stage_receipt.json'
    external_path = folder/'stage_receipt.sha256.json'
    receipt_digest = _read(external_path)['sha256']
    external_digest = file_sha256(external_path)
    _hash(receipt_path, receipt_digest)
    receipt = _read(receipt_path)
    _need(receipt.get('schema') == 'p5-derived-stage-seal-v1' and receipt.get('stage') == 'e1_statistics'
          and receipt.get('frozen_manifest_sha256') == file_sha256(manifest_path), 'Derived stage identity differs')
    expected_files = {str(folder/name) for name in REQUIRED_FILES}
    _need(type(receipt.get('files_sha256')) is dict and set(receipt['files_sha256']) == expected_files,
          'Exact summary/bootstrap/reproduction file set required')
    for path, digest in receipt['files_sha256'].items():
        _hash(path, digest)
    expected_sources = {str(run/name): file_sha256(run/name) for name in SOURCE_FILES}
    _need(receipt.get('source_sha256') == expected_sources, 'Exact four original E1/manifest source paths and hashes required')

    # Import lazily: ordinary module import performs no original-E1 adoption.
    from p5_final_e1_adoption import adopt_complete_e1
    adopted = adopt_complete_e1(run, m)
    _need(type(adopted) is dict and adopted.get('status') == 'ADOPT_COMPLETE_VERIFIED_E1',
          'The original E1 stage is not a complete verified576-prefix source')
    expected_ids = {spec['snapshot_id']+f'/candidate-{candidate:02d}/repeat-{repeat}'
                    for spec in m['E1'] for candidate in range(16) for repeat in range(3)}
    ids = adopted.get('completed_repeat_ids')
    _need(type(ids) is list and len(ids) == len(set(ids)) == 576 and set(ids) == expected_ids,
          'Original E1 adoption has another repeat denominator')
    _need(adopted.get('summary_sha256') == expected_sources[str(run/'E1/summary.json')]
          and adopted.get('stage_seal_path') == str(run/'E1/stage_receipt.json')
          and adopted.get('stage_seal_sha256') == expected_sources[str(run/'E1/stage_receipt.json')],
          'Adoption is bound to other original E1 files')
    _memory()
    ledger = _read(run/'E1/comparison_ledger.json')
    regenerated, arrays = analyze_E1(ledger, m['E1'], replicates=2000, seed=7711)
    stored = _read(folder/'summary.json')
    _need(_json_same(stored, regenerated), 'Summary does not exactly regenerate from all84 original diagnostic rows')
    path = folder/'bootstrap.npz'
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        _need(len(names) == len(set(names)) and all(name.endswith('.npy') and '/' not in name for name in names),
              'Duplicate/malformed bootstrap NPZ members')
        _need(sum(member.file_size for member in members) <= MAX_EXPANDED_NPZ_BYTES,
              'Expanded bootstrap NPZ exceeds the bounded review allowance')
    with np.load(path, allow_pickle=False) as saved:
        _need(set(saved.files) == set(arrays), 'Bootstrap NPZ key population differs')
        for key, expected in arrays.items():
            actual = saved[key]
            _need(actual.dtype == expected.dtype and actual.shape == expected.shape,
                  'Bootstrap dtype/shape differs: '+key)
            _need(actual.tobytes() == expected.tobytes(), 'Bootstrap exact values differ: '+key)
    reproduction = _read(folder/'reproduction.json')
    _need(reproduction.get('status') == 'PASS_EXACT_REGENERATION'
          and type(reproduction.get('source_rows')) is int and reproduction['source_rows'] == 84
          and type(reproduction.get('snapshots')) is int and reproduction['snapshots'] == 12
          and type(reproduction.get('families')) is int and reproduction['families'] == 6
          and reproduction.get('summary_sha256') == file_sha256(folder/'summary.json')
          and reproduction.get('bootstrap_sha256') == file_sha256(path),
          'Reproduction receipt has incomplete populations or different output hashes')
    # Reject a concurrent source mutation after adoption/regeneration as well.
    for source, digest in expected_sources.items():
        _hash(source, digest)
    for output, digest in receipt['files_sha256'].items():
        _hash(output, digest)
    _hash(receipt_path, receipt_digest)
    _hash(external_path, external_digest)
    _memory()
    return {'schema': 'p5-E1-statistics-independent-regeneration-review-v1',
        'status': 'PASS_E1_STATISTICS_EXACT_REGENERATION', 'accepted': True,
        'frozen_manifest_sha256': file_sha256(manifest_path),
        'source_sha256': expected_sources, 'files_sha256': dict(receipt['files_sha256']),
        'original_E1_adoption_status': adopted['status'], 'verified_physical_repeat_ids': 576,
        'verified_model_sets': 84, 'snapshots': 12, 'independent_families': 6,
        'bootstrap_replicates': 2000, 'bootstrap_seed': 7711, 'exact_npz_arrays': len(arrays),
        'additional_model_forward': 0, 'additional_physics': 0, 'training_updates': 0,
        'limits': ['Reproduces the frozen statistical algorithm after independent source reload; not a different statistical estimator.',
            'Original E1 adoption checks576 stored raw prefixes and84 diagnostic rows without re-stepping physics.',
            'No new final outcome generation, model superiority, industrial benefit or personal understanding is inferred.']}


def validate_e1_statistics_stage(run, m):
    """Verify an already sealed e1_statistics stage; return review or raise.

    Required source graph: run/manifest.json and original E1 summary,
    stage_receipt.json and comparison_ledger.json. This function writes nothing
    and never invents an absent receipt or retries an incomplete stage.
    """
    try:
        return _validate(Path(run).resolve(), m)
    except E1StatisticsReviewError:
        raise
    except (KeyError, TypeError, ValueError, OSError, zipfile.BadZipFile) as error:
        raise E1StatisticsReviewError('Incomplete or malformed E1 statistics stage') from error
