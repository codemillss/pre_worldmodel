"""Small CPU smoke: one physical interval and seven bound model rollouts.

No optimizer step, retraining, final experiment, or project outcome overwrite.
Run only from a source checkout; strict original loaders require recorded
NumPy/Torch versions. A mismatch is reported, never silently bypassed.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def main():
    import numpy as np
    import torch
    from push_align import PlanarPushEnv
    from push_align.normalization import load_normalizer
    from push_align.fork_checkpoint import load_fork_checkpoint
    from push_align.ridge import load_ridge
    from push_align.dynamics import free_rollout
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    env = PlanarPushEnv()
    initial = env.reset()
    step = env.step(np.zeros(2, np.float64))
    if not initial.observation_valid or not step.transition_valid:
        raise RuntimeError('Minimal physical contract failed')
    normalizer_path = ROOT/'artifacts/learning/normalizer'
    nm = read(normalizer_path/'normalizer.json')['normalizer']
    normalizer = load_normalizer(normalizer_path,nm['schema'],nm['dataset_identity'],nm['normalizer_hash'])
    with np.load(ROOT/'artifacts/learning/prepared.npz',allow_pickle=False) as prepared:
        window = prepared['windows_5'][0]
        state = torch.from_numpy(prepared['current_state'][window[0]].copy())
        actions = torch.from_numpy(prepared['applied_action'][window].copy())
    results = {}
    for folder in sorted((ROOT/'artifacts/models').iterdir()):
        if folder.name == 'R':
            m = read(folder/'ridge.json')
            bundle = load_ridge(folder,normalizer,m['config'],m['training_provenance'])
        else:
            m = read(folder/'checkpoint.json')
            bundle = load_fork_checkpoint(folder,normalizer,m['config'],m['training_provenance'],m['fork_budget'],
                seed=m['seed'],branch=m['branch'],resume=False,expected_checkpoint_hash=m['checkpoint_hash'],
                expected_run_manifest_sha256=m['run_manifest_sha256'])
        with torch.no_grad():
            states = free_rollout(bundle.model,normalizer,state,actions)
        if states.shape != (6,10) or not bool(torch.isfinite(states).all()):
            raise RuntimeError('Nonfinite/wrong-shape five-step rollout: '+folder.name)
        results[folder.name] = {'bound_loader':'PASS','rollout_shape':list(states.shape),'finite':True}
    print(json.dumps({'status':'PASS_PUBLIC_CPU_SMOKE','physics_intervals':1,'action_dt_s':step.actual_dt,
        'models':results,'training_updates':0,'scope':'Runtime and bound inference only; no alignment or cross-platform performance claim'},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
