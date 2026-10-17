"""Additive post-freeze E2 reporting repair: canonical reduction order only.

The frozen final_statistics implementation and all recorded outcomes remain
unchanged. Its descriptive floating-point reductions depend on row order, so
strict forward/reverse reproduction can fail in the last few floating bits.
This wrapper orders a new list by the frozen expected cell_id sequence before
delegating every validation,
denominator, family/seed estimand and bootstrap operation to that exact frozen
implementation. It is reporting infrastructure, not a changed scientific
estimator, a new independent analysis or permission to overwrite sealed stages.
"""
from __future__ import annotations

from .final_statistics import analyze_E2 as _frozen_analyze_E2

ANALYSIS_INFRASTRUCTURE_VERSION = 'p5-E2-canonical-row-reporting-v2'
CANONICAL_ORDER = 'position of cell_id in the fixed expected E2 manifest sequence before frozen analysis'


def analyze_E2(rows, expected, *, bootstrap_replicates=2000, bootstrap_seed=7711):
    """Return the original result schema with deterministic reduction ordering.

    The frozen expected sequence is part of the explicit input contract and
    preserves the original forward reporting order. Inputs are not mutated. No case is added, removed, filled, reclassified or
    selected based on its outcome; the original complete96 validator still
    rejects duplicates, missing cases, invalid metrics and infrastructure rows.
    """
    if not isinstance(rows, (list, tuple)):
        raise ValueError('Explicit complete recorded row sequence required')
    if any(type(row) is not dict or type(row.get('cell_id')) is not str for row in rows):
        raise ValueError('Explicit string cell_id required for canonical row ordering')
    if not isinstance(expected, (list, tuple)) or any(
            type(row) is not dict or type(row.get('cell_id')) is not str for row in expected):
        raise ValueError('Explicit frozen expected cell_id sequence required')
    ranks = {row['cell_id']: rank for rank, row in enumerate(expected)}
    # Unknown and duplicate identities remain present. The unchanged original
    # validator rejects them; no result is silently omitted or substituted.
    canonical = sorted(rows, key=lambda row: ranks.get(row['cell_id'], len(ranks)))
    return _frozen_analyze_E2(canonical, expected,
        bootstrap_replicates=bootstrap_replicates, bootstrap_seed=bootstrap_seed)
