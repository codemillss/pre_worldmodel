"""Verify the selected publication, without retraining or physical execution.

Default: hashes and recorded outcome counts using only the standard library.
--recompute: also regenerate E2 statistics and all bootstrap arrays from the
published 96 rows, plus validate E1 saved cost-array/choice consistency.
This cannot attest omitted substep raw or repeat the original whole-run audit.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))

def require(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recompute', action='store_true')
    args = parser.parse_args()
    inventory = read('artifacts/provenance/publication_files.json')
    seen = set()
    for entry in inventory['records']:
        relative = entry['public_path']
        require(relative not in seen, 'Duplicate public inventory path')
        seen.add(relative)
        target = (ROOT / relative).resolve()
        require(target.is_relative_to(ROOT), 'Public inventory path escapes checkout')
        data = target.read_bytes()
        require(len(data) == entry['bytes'], 'Size mismatch: '+relative)
        require(hashlib.sha256(data).hexdigest() == entry['public_sha256'], 'Hash mismatch: '+relative)
        if entry['transformation'] == 'byte_identical':
            require(entry['source_sha256'] == entry['public_sha256'], 'False byte-identity claim')
    rows = read('artifacts/results/E2/rows.json')
    expected = read('artifacts/provenance/frozen_manifest.json')['E2']
    require(len(rows) == 96 and len({row['cell_id'] for row in rows}) == 96, 'Expected exactly 96 unique E2 rows')
    require({row['cell_id'] for row in rows} == {cell['cell_id'] for cell in expected}, 'Frozen cell identities differ')
    require(len({row['family_id'] for row in rows}) == 12, 'Expected 12 independent E2 families')
    require(all(row['success'] == (row['termination'] == 'ready') for row in rows), 'Success/termination mismatch')
    controllers = {}
    summary = read('artifacts/results/E2/summary.json')
    for name in sorted({row['controller_id'] for row in rows}):
        subset = [row for row in rows if row['controller_id'] == name]
        successes = sum(row['success'] for row in subset)
        require(len(subset) == 12, 'Controller denominator must be 12')
        require(summary['controllers'][name]['success_count'] == successes, 'Published success aggregate differs')
        controllers[name] = {'success': successes, 'episodes': len(subset), 'terminations': dict(Counter(row['termination'] for row in subset))}
    app = read('artifacts/audits/p5_application_completion.v1.json')
    require(len(app['rows']) == 8 and sum(row['success'] for row in app['rows']) == 1, 'Application eight-row outcome differs')
    result = {'status': 'PASS_SELECTED_PUBLIC_EVIDENCE', 'verified_copied_file_count': len(seen),
              'E2_episodes': 96, 'E2_independent_families': 12, 'controllers': controllers,
              'application': {'episodes': 8, 'successes': 1},
              'new_physics': 0, 'new_model_calls': 0, 'training_updates': 0,
              'scope': 'Publication hashes and saved outcomes; omitted substep raw and physical replay are not revalidated'}
    if args.recompute:
        import numpy as np
        from push_align.final_statistics_v2 import analyze_E2
        actual, arrays = analyze_E2(rows, expected)
        require(actual == summary, 'E2 regenerated summary differs')
        with np.load(ROOT/'artifacts/results/E2/bootstrap.npz', allow_pickle=False) as saved:
            require(set(saved.files) == set(arrays), 'Bootstrap columns differ')
            for key, value in arrays.items():
                require(np.array_equal(saved[key], value, equal_nan=True), 'Bootstrap array differs: '+key)
        snapshots = list((ROOT/'artifacts/results/E1/snapshots').glob('*/*/tick-*'))
        require(len(snapshots) == 12, 'Expected 12 E1 snapshots')
        choices = {}
        model_sets = 0
        for folder in snapshots:
            with np.load(folder/'actual_futures.npz', allow_pickle=False) as truth:
                require(truth['total_cost'].shape == (16, 3), 'E1 actual cost shape differs')
                require(truth['valid'].shape == (16, 3), 'E1 validity shape differs')
                require(np.isfinite(truth['total_cost']).all(), 'Nonfinite E1 actual costs')
            for model_folder in (folder/'models').iterdir():
                diagnostics = json.loads((model_folder/'diagnostics.json').read_text())
                with np.load(model_folder/'trace.npz', allow_pickle=False) as prediction:
                    require(prediction['predicted_states'].shape == (16, 6, 10), 'E1 prediction shape differs')
                    eligible = np.flatnonzero(prediction['valid'] & np.isfinite(prediction['total_cost']))
                    selected = int(eligible[np.argmin(prediction['total_cost'][eligible])]) if len(eligible) else None
                    require(selected == diagnostics['selected_index'], 'E1 selected saved cost does not agree')
                choices[(str(folder.relative_to(ROOT)),model_folder.name)] = selected
                model_sets += 1
        agreement = 0
        for folder in snapshots:
            for seed in (11,23,47):
                base = str(folder.relative_to(ROOT))
                agreement += choices[(base,f'S1-{seed}')] == choices[(base,f'SK-{seed}')]
        require(model_sets == 84 and agreement == 35, 'E1 84 sets / 35 of 36 agreement differs')
        result.update(E2_recomputed_summary_exact=True, E2_bootstrap_arrays_exact=len(arrays),
                      E1_saved_model_sets_checked=model_sets, E1_S1_SK_same_selected_index=f'{agreement}/36',
                      recomputation_scope='E2 exact arithmetic; E1 saved cost selection only; not original raw generation or fixed-prediction window re-inference')
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
