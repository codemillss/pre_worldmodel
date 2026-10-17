"""P4-02: nine matched, final-update learned-dynamics training branches.

Validation selects one global lambda. No test, simulator future, MPC or new
alignment is used. Checkpoint replay updates happen only on disposable copies.
Existing data, common models, failures and frozen budgets remain immutable.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import subprocess
import sys
from time import perf_counter
import tomllib
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import torch

from push_align.checkpoint import make_optimizer
from push_align.config import ROOT
from push_align.data import digest
from push_align.fork_checkpoint import save_fork_checkpoint, load_fork_checkpoint
from push_align.fork_sampler import WindowSampler
from push_align.fork_training import train_sequences, choose_lambda
from push_align.learning_data import load_learning_data
from push_align.prediction import _windows, evaluate_predictions
from push_align.sequence_training import sequence_update
from p2_train_pilot import save, save_evaluation, parameter_vector, tensor_tree_equal
from p3_train_common_mlp import load_common_checkpoint, validate_settings, evaluation_equal
from p3_train_ridge import array_hash
from p4_verify_sequence_loss import snapshot_sources, upstream_state, inventory

LEARNING = ROOT / 'runs/p3-02-20261009-01/learning_data'
COMMON = ROOT / 'runs/p3-04-20261009-01'
PRIOR_P4 = ROOT / 'runs/p4-01-20261012-01'
BUDGET = ROOT / 'configs/p4_fork_budget.v2.json'
BUDGET_SHA = '9054a25951e1e771c62fbb101cc900c3b5f2cf06fc67eed4e21d1eb9e0e95a70'
P4_01_SHA = '10a4edeae55ff64c3e20a372433097bfc4bcd21edf9635e9c33e22a4427a7aba'
GIB = 1024**3


def tree_hash(value):
    """Typed structural identity of model/Adam/RNG, independent of pickle bytes."""
    h = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            h.update(b'tensor'); h.update(str(item.dtype).encode())
            h.update(json.dumps(list(item.shape)).encode())
            h.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, dict):
            h.update(b'dict')
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                visit(key); visit(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(type(item).__name__.encode())
            for child in item: visit(child)
        else:
            h.update(type(item).__name__.encode())
            h.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
    visit(value)
    return h.hexdigest()


def byte_count(folder):
    return sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())


def storage_guard(out):
    used, free = byte_count(out), shutil.disk_usage(ROOT).free
    if used > GIB or free < 2*GIB:
        raise RuntimeError('P4-02 storage/reserve guard exceeded; preserve prefix and reestimate')
    return {'stored_bytes': used, 'free_bytes': free}


def all_upstream():
    from p3_geometry_pilot import verify_artifacts
    state = upstream_state()
    state[str(PRIOR_P4.relative_to(ROOT))] = {
        'registered_artifacts': verify_artifacts(PRIOR_P4), 'file_inventory': inventory(PRIOR_P4)}
    return state


def data_hash(states, actions, mask):
    return tree_hash({'states': states, 'applied_actions': actions, 'mask': mask})


def restore_common(entry, normalizer, config, common_manifest, settings):
    bundle = load_common_checkpoint(ROOT/entry['path'], normalizer, config,
        common_manifest['provenance'], settings, entry['seed'], resume=True)
    if (bundle.metadata['checkpoint_hash'] != entry['checkpoint_hash']
            or bundle.metadata['updates'] != entry['initial_updates']):
        raise ValueError('Restored common model differs from the frozen fork origin')
    return bundle


def start_identity(bundle, sampler):
    return {'model': tree_hash(bundle.model.state_dict()),
            'Adam': tree_hash(bundle.optimizer.state_dict()),
            'Torch_RNG': tree_hash(bundle.torch_rng_state),
            'common_shuffle_RNG_before_window_permutation': tree_hash(bundle.shuffle_generator.get_state()),
            'initial_sampler': sampler.state_dict()['state_hash']}


def resume_proof(path, metadata, model, optimizer, sampler, normalizer, config,
                 provenance, budget, seed, branch, run_sha, states, actions, mask, evaluation, prepared,
                 probe_counter):
    """Live reference and two independent disk copies, each one diagnostic step."""
    saved_before = {p.name: digest(p) for p in path.iterdir() if p.is_file()}
    live_weights, live_adam = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())
    live_sampler, live_rng = sampler.state_dict(), torch.get_rng_state().clone()
    copies = [load_fork_checkpoint(path, normalizer, config, provenance, budget,
        seed=seed, branch=branch, resume=True, expected_checkpoint_hash=metadata['checkpoint_hash'],
        expected_run_manifest_sha256=run_sha) for _ in range(2)]
    initial = all(tensor_tree_equal(c.model.state_dict(), live_weights)
        and tensor_tree_equal(c.optimizer.state_dict(), live_adam)
        and tensor_tree_equal(c.sampler.state_dict(), live_sampler)
        and torch.equal(c.torch_rng_state, live_rng) for c in copies)
    reloaded_eval = evaluate_predictions(copies[0].model, normalizer, prepared=prepared,
                                        horizons=(5,))
    # Probe copies have a one-step extension, never a new official budget.
    with torch.random.fork_rng(devices=[]):
        reference = deepcopy(model)
        reference_optimizer = make_optimizer(reference, config)
        reference_optimizer.load_state_dict(deepcopy(live_adam))
        samplers = [sampler.clone_for_resume_probe(), *(c.sampler.clone_for_resume_probe() for c in copies)]
        positions = [s.next_batch() for s in samplers]
        if not all(torch.equal(positions[0], p) for p in positions[1:]):
            raise ValueError('Restored checkpoint next train window differs')
        before_second = deepcopy(copies[1].model.state_dict())
        before_second_adam = deepcopy(copies[1].optimizer.state_dict())
        diagnostics, after_rng = [], []
        weight = 0. if branch == 'S1' else float(branch.removeprefix('SK-lambda'))
        for i, (current, opt, rng) in enumerate([(reference, reference_optimizer, live_rng),
                (copies[0].model, copies[0].optimizer, copies[0].torch_rng_state),
                (copies[1].model, copies[1].optimizer, copies[1].torch_rng_state)]):
            torch.set_rng_state(rng)
            p = positions[i]
            probe_counter['attempted'] += 1
            diagnostics.append(sequence_update(current, opt, normalizer, states[p], actions[p], mask[p],
                                                 weight, config['training']['gradient_clip_norm']))
            probe_counter['completed'] += 1
            after_rng.append(torch.get_rng_state().clone())
            if i == 1:
                independent = tensor_tree_equal(copies[1].model.state_dict(), before_second) and tensor_tree_equal(copies[1].optimizer.state_dict(), before_second_adam)
        equal_parameters = all(tensor_tree_equal(reference.state_dict(), c.model.state_dict()) for c in copies)
        equal_adam = all(tensor_tree_equal(reference_optimizer.state_dict(), c.optimizer.state_dict()) for c in copies)
    checks = {'initial_model_adam_sampler_torch_rng_exact': initial,
        'reloaded_K5_summary_and_every_trace_exact': evaluation_equal(evaluation, reloaded_eval),
        'next_batch_exact': all(torch.equal(positions[0], p) for p in positions[1:]),
        'restored_models_and_adam_do_not_share_memory': independent,
        'next_model_and_adam_exact_live_reference': equal_parameters and equal_adam,
        'next_loss_gradient_diagnostics_exact': diagnostics[0] == diagnostics[1] == diagnostics[2],
        'next_sampler_permutation_cursor_rng_exact': all(tensor_tree_equal(samplers[0].state_dict(), s.state_dict()) for s in samplers[1:]),
        'next_Torch_rng_exact': all(torch.equal(after_rng[0], r) for r in after_rng[1:]),
        'live_model_adam_sampler_and_global_rng_unchanged': tensor_tree_equal(model.state_dict(), live_weights) and tensor_tree_equal(optimizer.state_dict(), live_adam) and tensor_tree_equal(sampler.state_dict(), live_sampler) and torch.equal(torch.get_rng_state(), live_rng),
        'saved_checkpoint_files_unchanged': saved_before == {p.name: digest(p) for p in path.iterdir() if p.is_file()}}
    if not all(checks.values()):
        raise ValueError('Mid-epoch checkpoint resume proof failed')
    return {'pass': True, 'checks': checks, 'additional_updates': sampler.updates,
        'sampler_epoch': sampler.epoch, 'sampler_cursor': sampler.cursor,
        'next_original_window_positions': positions[0].tolist(),
        'diagnostic_updates': 3, 'formal_budget_changed': False,
        'next_loss_diagnostics': diagnostics[0], 'scope': 'three disposable copies; saved/live formal weights never replaced'}


def train_only_probe(entry, prepared, normalizer, config, common_manifest, settings, budget):
    """Three fixed updates; validation-only change versus train-positive control."""
    copies = [deepcopy(prepared) for _ in range(3)]
    val = copies[1].arrays['split_code'] == 1
    for field in ('current_state', 'next_state'):
        copies[1].arrays[field][val, 0] += .01
    train = copies[2].arrays['split_code'] == 0
    for field in ('current_state', 'next_state'):
        copies[2].arrays[field][train, 6] += .002
    results = []
    with torch.random.fork_rng(devices=[]):
        for data in copies:
            rows, states, actions, mask = train_sequences(data)
            bundle = restore_common(entry, normalizer, config, common_manifest, settings)
            sampler = WindowSampler(len(rows), budget['batch_windows'], bundle.shuffle_generator,
                                    max_updates=budget['additional_updates_per_branch'], horizon=budget['K'])
            torch.set_rng_state(bundle.torch_rng_state)
            order = []
            for _ in range(3):
                positions = sampler.next_batch(); order.append(positions.clone())
                sequence_update(bundle.model, bundle.optimizer, normalizer, states[positions],
                    actions[positions], mask[positions], .1, config['training']['gradient_clip_norm'])
            results.append({'weights': deepcopy(bundle.model.state_dict()), 'Adam': deepcopy(bundle.optimizer.state_dict()),
                            'order': order, 'data_hash': data_hash(states, actions, mask)})
    checks = {'validation_only_change_train_data_exact': results[0]['data_hash'] == results[1]['data_hash'],
        'validation_only_change_weights_adam_and_order_exact': tensor_tree_equal(results[0]['weights'], results[1]['weights']) and tensor_tree_equal(results[0]['Adam'], results[1]['Adam']) and tensor_tree_equal(results[0]['order'], results[1]['order']),
        'train_positive_control_changes_data_and_weights': results[0]['data_hash'] != results[2]['data_hash'] and not tensor_tree_equal(results[0]['weights'], results[2]['weights'])}
    if not all(checks.values()): raise ValueError('Train-only mask probe failed')
    return {'pass': True, 'checks': checks, 'diagnostic_updates': 9, 'formal_updates': 0,
        'scope': 'fixed-update train invariance; later validation selection may change; raw/stat files unchanged'}


def train_branch(entry, branch, prepared, normalizer, config, common_manifest, settings,
                 provenance, budget, run_sha, rows, states, actions, mask, out):
    seed, began = entry['seed'], perf_counter()
    folder = out/'seeds'/f'seed-{seed:02d}'/branch; folder.mkdir(parents=True, exist_ok=False)
    bundle = restore_common(entry, normalizer, config, common_manifest, settings)
    model, optimizer = bundle.model, bundle.optimizer
    torch.set_rng_state(bundle.torch_rng_state)
    sampler = WindowSampler(len(rows), budget['batch_windows'], bundle.shuffle_generator,
                            max_updates=budget['additional_updates_per_branch'], horizon=budget['K'])
    start = start_identity(bundle, sampler)
    save(folder/'start_identity.json', start)
    initial_weights = parameter_vector(model)
    weight = 0. if branch == 'S1' else float(branch.removeprefix('SK-lambda'))
    losses, timings, positions_all, offsets, hashes = [], [], [], [0], []
    checkpoints = []; training_s = evaluation_s = saving_s = replay_s = 0.
    probe_counter = {'attempted': 0, 'completed': 0}
    positions = None
    final_evaluation = final_metadata = None
    try:
        for update in range(1, budget['additional_updates_per_branch']+1):
            positions = sampler.next_batch()
            current_states, current_actions, current_mask = states[positions], actions[positions], mask[positions]
            tick = perf_counter()
            diagnostic = sequence_update(model, optimizer, normalizer, current_states,
                current_actions, current_mask, weight, config['training']['gradient_clip_norm'])
            elapsed = perf_counter()-tick; training_s += elapsed
            losses.append([diagnostic['one_step_loss'], np.nan if diagnostic['rollout_loss'] is None else diagnostic['rollout_loss'], diagnostic['loss_before_update'], diagnostic['gradient_norm_before_clip']])
            timings.append(elapsed); positions_all.append(positions.numpy().copy())
            offsets.append(offsets[-1]+len(positions)); hashes.append(data_hash(current_states, current_actions, current_mask))
            if diagnostic['valid_transitions'] != len(positions)*5 or diagnostic['scalar_terms'] != len(positions)*50:
                raise ValueError('L_one/mask denominator violates common actual window population')
            if update % budget['checkpoint_every_updates'] == 0:
                checkpoint = folder/'checkpoints'/f'update-{update:04d}'
                tick = perf_counter()
                final_metadata = save_fork_checkpoint(checkpoint, model, optimizer, normalizer,
                    config, provenance, budget, seed=seed, branch=branch, additional_updates=update,
                    sampler=sampler, run_manifest_sha256=run_sha)
                saving_s += perf_counter()-tick
                tick = perf_counter()
                k5 = evaluate_predictions(model, normalizer, prepared, horizons=(5,))
                if update == budget['additional_updates_per_branch']:
                    final_evaluation = evaluate_predictions(model, normalizer, prepared, horizons=(1,5,10))
                evaluation_s += perf_counter()-tick
                save_evaluation(folder/'evaluations'/f'update-{update:04d}', final_evaluation if final_evaluation else k5)
                tick = perf_counter()
                proof = resume_proof(checkpoint, final_metadata, model, optimizer, sampler, normalizer,
                    config, provenance, budget, seed, branch, run_sha, states, actions, mask, k5, prepared,
                    probe_counter)
                replay_s += perf_counter()-tick
                save(folder/f'resume-{update:04d}.json', proof)
                checkpoints.append({'additional_updates':update,'checkpoint':str(checkpoint.relative_to(out)),
                    'checkpoint_hash':final_metadata['checkpoint_hash'], 'sampler_epoch':sampler.epoch,
                    'sampler_cursor':sampler.cursor,'sampler_state_hash':sampler.state_dict()['state_hash'],
                    'K5':k5.summary['horizons']['5'],
                    'validation_selection_score':k5.summary['selection_score'],
                    'validation_selection_status':k5.summary['selection_status'], 'resume_pass':proof['pass']})
                storage_guard(out)
                print(json.dumps({'progress':'checkpoint','seed':seed,'branch':branch,'update':update,
                    'K5_score':k5.summary['selection_score']}),flush=True)
        flat = np.concatenate(positions_all)
        np.savez_compressed(folder/'training_trace.npz', window_positions=flat,
            batch_offsets=np.array(offsets,dtype=np.int64), batch_data_hashes=np.array(hashes,dtype='U64'),
            losses=np.array(losses,dtype=np.float64), update_wall_time_s=np.array(timings,dtype=np.float64))
        final_path = folder/'checkpoints'/'update-1000'
        reloaded = load_fork_checkpoint(final_path, normalizer, config, provenance, budget,
            seed=seed, branch=branch, expected_checkpoint_hash=final_metadata['checkpoint_hash'],
            expected_run_manifest_sha256=run_sha)
        reload_final = evaluate_predictions(reloaded.model, normalizer, prepared, horizons=(1,5,10))
        val_hashes = {str(k):array_hash(final_evaluation.traces[f'h{k}_window_rows']) for k in (1,5,10)}
        checks = {'all_1000_additional_updates':sampler.updates == 1000 and len(losses)==1000,
            'all_10_checkpoints_and_resumes':len(checkpoints)==10 and all(c['resume_pass'] for c in checkpoints),
            'window_62992_transition_314960_epoch22_cursor832':sampler.window_occurrences==62992 and sampler.transition_occurrences==314960 and sampler.epoch==22 and sampler.cursor==832,
            'all_recorded_batches_finite_S1_rollout_absent_SK_rollout_finite':bool(np.isfinite(np.asarray(losses)[:,[0,2,3]]).all()) and (bool(np.isnan(np.asarray(losses)[:,1]).all()) if weight==0 else bool(np.isfinite(np.asarray(losses)[:,1]).all())),
            'start_weights_changed':not torch.equal(initial_weights,parameter_vector(model)),
            'final_live_and_loaded_weights_exact':tensor_tree_equal(model.state_dict(),reloaded.model.state_dict()),
            'final_K1_K5_K10_reloaded_trace_exact':evaluation_equal(final_evaluation,reload_final),
            'actual_979_full64_and21_partial16_batches':int((np.diff(offsets)==64).sum())==979 and int((np.diff(offsets)==16).sum())==21,
            'actual_adam_total_updates_bound':final_metadata['total_updates']==entry['initial_updates']+1000}
        result={'seed':seed,'branch':branch,'lambda':weight,'pass':all(checks.values()),'completed':True,
            'checks':checks,'start_identity':start,'additional_updates':sampler.updates,
            'window_occurrences':sampler.window_occurrences,'transition_occurrences':sampler.transition_occurrences,
            'scalar_terms':sampler.transition_occurrences*10,'sampler_epoch':sampler.epoch,'sampler_cursor':sampler.cursor,
            'checkpoint_count':len(checkpoints),'final_checkpoint':str(final_path.relative_to(out)),
            'final_checkpoint_hash':final_metadata['checkpoint_hash'],'checkpoints':checkpoints,
            'validation':final_evaluation.summary,'validation_window_hashes':val_hashes,
            'schedule_hash':array_hash(flat),'batch_offsets_hash':array_hash(np.array(offsets,dtype=np.int64)),
            'batch_data_hash_sequence_hash':array_hash(np.array(hashes,dtype='U64')),
            'resources':{'training_updates_s':training_s,'validation_s':evaluation_s,
                'checkpoint_s':saving_s,'replay_diagnostic_s':replay_s,'total_branch_s':perf_counter()-began},
            'diagnostic_resume_updates':probe_counter['completed'],
            'diagnostic_resume_update_attempts':probe_counter['attempted'],
            'alignment_outcome':'NOT_EVALUATED','final_test_used':False}
        if not result['pass']:raise ValueError('Branch counters/checkpoint/final reload incomplete')
    except Exception as error:
        # A failed prefix is diagnostic only; never a selected final checkpoint.
        with (folder/'failed_live_diagnostic.pt').open('xb') as f:
            torch.save({'weights':deepcopy(model.state_dict()),'optimizer':deepcopy(optimizer.state_dict()),
                'sampler':sampler.state_dict(),'torch_rng_state':torch.get_rng_state().clone()},f)
        result={'seed':seed,'branch':branch,'lambda':weight,'pass':False,'completed':False,
            'additional_updates_completed':len(losses),'sampler_batches_requested':sampler.updates,
            'diagnostic_resume_updates':probe_counter['completed'],
            'diagnostic_resume_update_attempts':probe_counter['attempted'],
            'last_requested_window_positions':None if positions is None else positions.tolist(),
            'error':str(error),'traceback':traceback.format_exc(),'checkpoints':checkpoints,
            'diagnostic_scope':'failed prefix only; no final selection or fallback'}
        np.savez_compressed(folder/'failed_training_trace.npz', window_positions=np.concatenate(positions_all) if positions_all else np.zeros(0,dtype=np.int64),batch_offsets=np.array(offsets,dtype=np.int64),losses=np.asarray(losses,dtype=np.float64),batch_data_hashes=np.array(hashes,dtype='U64'),update_wall_time_s=np.asarray(timings,dtype=np.float64))
    save(folder/'branch_summary.json',result)
    return result


def run(args,out):
    began=perf_counter(); free_before=shutil.disk_usage(ROOT).free
    if free_before < 3*GIB:raise RuntimeError('Need 1GiB stage allowance and 2GiB reserve')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    if digest(BUDGET)!=BUDGET_SHA or digest(PRIOR_P4/'summary.json')!=P4_01_SHA:
        raise ValueError('Frozen fork budget or verified P4-01 result changed')
    prior=json.loads((PRIOR_P4/'summary.json').read_text())
    if not prior['pass'] or not all(prior['checks'].values()):raise ValueError('P4-01 prerequisite failed')
    budget=json.loads(BUDGET.read_text()); common_manifest=json.loads((COMMON/'manifest.json').read_text())
    settings=validate_settings(tomllib.loads((ROOT/'configs/p3_training.v1.toml').read_text()))
    before=all_upstream(); save(out/'upstream_before.json',before)
    metadata=json.loads((LEARNING/'normalizer/normalizer.json').read_text())['normalizer']
    prepared,normalizer=load_learning_data(LEARNING,metadata['schema'],metadata['dataset_identity'],settings['normalizer_hash'])
    config=prepared.metadata['contract']['resolved_config']
    if normalizer.metadata['stats_hash'] != common_manifest['stats_hash'] or digest(LEARNING/'manifest.json')!=common_manifest['provenance']['learning_manifest_sha256']:
        raise ValueError('Common checkpoint/data/stat provenance changed')
    rows,states,actions,mask=train_sequences(prepared)
    if rows.shape!=(2960,5) or int((prepared.arrays['split_code']==0).sum())!=3200 or int((prepared.arrays['split_code']==1).sum())!=1600:
        raise ValueError('Frozen real train/validation populations changed')
    validation_hashes={str(k):array_hash(_windows(prepared,'validation',k)) for k in (1,5,10)}
    sources=snapshot_sources(out)
    provenance={'learning_manifest_sha256':digest(LEARNING/'manifest.json'),
        'dataset_identity':metadata['dataset_identity'],'validation_split':'validation','task_scope':'P4-02-matched-fork'}
    manifest={'task_id':'P4-02','created_at':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        'argv':sys.argv,'python':platform.python_version(),'torch':str(torch.__version__),'numpy':str(np.__version__),
        'platform':platform.platform(),'torch_threads':1,'deterministic_algorithms':True,
        'budget':budget,'budget_file_sha256':BUDGET_SHA,'source_hashes':sources,'provenance':provenance,
        'physical_config':config,'normalizer_hash':metadata['normalizer_hash'],'stats_hash':metadata['stats_hash'],
        'common_settings':settings,'train_windows_hash':array_hash(rows),'train_sequence_hash':data_hash(states,actions,mask),
        'train_window_count':len(rows),'train_original_row_count':3200,'validation_original_row_count':1600,
        'validation_window_hashes':validation_hashes,'free_disk_before_bytes':free_before,
        'stage_storage_allowance_bytes':GIB,'free_reserve_bytes':2*GIB,'final_test_used':False,
        'MPC':'NOT_RUN','alignment_outcome':'NOT_EVALUATED','personal_understanding':'NOT_ASSESSED'}
    save(out/'manifest.json',manifest); run_sha=digest(out/'manifest.json')
    np.savez_compressed(out/'train_windows.npz',window_rows=rows)
    process=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,
        capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    log=process.stdout+process.stderr; (out/'tests.txt').write_text(log)
    found=re.search(r'Ran (\d+) tests',log); count=int(found.group(1)) if found else 0
    new_tests=[line.split(' ... ')[0] for line in log.splitlines() if 'test_p4_fork_' in line and line.endswith(' ... ok')]
    if process.returncode or count<714 or len(new_tests)!=53 or 'skipped=' in log or ' ... skipped ' in log:
        raise ValueError('Required complete regression/new P4 fork tests failed or skipped')
    save(out/'regression_result.json',{'pass':True,'test_count':count,'P4_fork_test_count':len(new_tests),'P4_fork_test_names':new_tests,'no_skips':True})
    print(json.dumps({'progress':'regressions_complete','tests':count,'P4_fork_tests':len(new_tests)}),flush=True)
    probe=train_only_probe(budget['common_checkpoints'][0],prepared,normalizer,config,common_manifest,settings,budget)
    save(out/'train_only_probe.json',probe)
    results=[]
    for entry in budget['common_checkpoints']:
        for branch in budget['branches_per_seed']:
            result=train_branch(entry,branch,prepared,normalizer,config,common_manifest,settings,
                provenance,budget,run_sha,rows,states,actions,mask,out)
            results.append(result)
            print(json.dumps({'progress':'branch_complete','seed':entry['seed'],'branch':branch,'pass':result['pass']}),flush=True)
            storage_guard(out)
    final_records=[]
    for r in results:
        if r['branch']=='S1':continue
        validation=r['validation'] if r['completed'] else None
        k5=validation['horizons']['5'] if validation else None
        final_records.append({'seed':r['seed'],'lambda':float(r['lambda']),
            'additional_updates':r['additional_updates'] if r['completed'] else r['additional_updates_completed'],
            'selection_status':validation['selection_status'] if validation else 'ineligible',
            'selection_score':validation['selection_score'] if validation else None,
            'K5_invalid_windows':k5['invalid_windows'] if k5 else None})
    save(out/'final_lambda_records.json',final_records)
    selection=(choose_lambda(final_records,budget) if all(r['completed'] for r in results)
        else {'pass':False,'selected_lambda':None,'candidates':[],
              'reason':'training_incomplete; missing final validation is not an observed invalid-window count'})
    save(out/'lambda_selection.json',selection)
    selected_models={str(seed):next((r['final_checkpoint'] for r in results if r['completed'] and r['seed']==seed and r['branch']==f"SK-lambda{selection['selected_lambda']}"),None) for seed in budget['seeds']} if selection['pass'] else {}
    control=[]
    for seed in budget['seeds']:
        group=[r for r in results if r['seed']==seed]
        complete=all(r['completed'] for r in group)
        control.append({'seed':seed,'pass':complete and all(r['start_identity']==group[0]['start_identity'] and r['schedule_hash']==group[0]['schedule_hash'] and r['batch_offsets_hash']==group[0]['batch_offsets_hash'] and r['batch_data_hash_sequence_hash']==group[0]['batch_data_hash_sequence_hash'] for r in group),
            'scope':'weights/Adam/Torch/common shuffle RNG and actual batch state/action/mask identities; numerical L_one may differ after branching'})
    save(out/'matched_controls.json',{'pass':all(c['pass'] for c in control),'seeds':control})
    after=all_upstream(); save(out/'upstream_after.json',after)
    store=storage_guard(out)
    checks={'all_714_or_more_regressions_and53_fork_tests':True,'train_only_validation_invariance_and_positive_control':probe['pass'],
        'all9_branches_complete_exact1000_updates':len(results)==9 and all(r['pass'] for r in results),
        'three_seed_matched_start_and_entire_batch_data_order':all(c['pass'] for c in control),
        'all90_checkpoints_and_independent_mid_epoch_next_update_proofs':sum(r.get('checkpoint_count',0) for r in results)==90 and all(r.get('checks',{}).get('all_10_checkpoints_and_resumes',False) for r in results),
        'same_final_validation_window_population_K1_K5_K10':all(r['completed'] and r['validation_window_hashes']==validation_hashes for r in results),
        'one_global_lambda_selected_from_all3_final_seed_scores':selection['pass'] and all(selected_models.values()),
        'all_upstream_registered_artifacts_and_file_inventories_unchanged':before==after,
        'source_snapshots_and_original_sources_frozen_during_training':all(digest(ROOT/rel)==sha and digest(out/'source'/rel)==sha for rel,sha in sources.items()),
        'manifest_frozen_during_training':digest(out/'manifest.json')==run_sha,
        'within1GiB_stage_and2GiB_free_reserve':store['stored_bytes']<=GIB and store['free_bytes']>=2*GIB,
        'final_test_and_MPC_unused':True}
    resources={'total_wall_time_s':perf_counter()-began,
        'main_peak_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024 if sys.platform=='darwin' else 1024),
        'regression_child_peak_rss_MiB':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/(1024*1024 if sys.platform=='darwin' else 1024),
        'training_updates_s':sum(r.get('resources',{}).get('training_updates_s',0) for r in results),
        'validation_s':sum(r.get('resources',{}).get('validation_s',0) for r in results),
        'checkpoint_s':sum(r.get('resources',{}).get('checkpoint_s',0) for r in results),
        'replay_diagnostic_s':sum(r.get('resources',{}).get('replay_diagnostic_s',0) for r in results),
        'stored_bytes_before_final_summary_index':store['stored_bytes'],'free_disk_before_bytes':free_before,
        'free_disk_after_bytes':store['free_bytes'],'formal_updates':sum(r.get('additional_updates',r.get('additional_updates_completed',0)) for r in results),
        'disposable_probe_updates':probe['diagnostic_updates']+sum(r.get('diagnostic_resume_updates',0) for r in results),
        'scope':'local CPU/Torch1thread; cumulative peaks are not simultaneous sum; no task/hardware real-time claim',
        'remaining_full_evaluation_budget':'reestimate remaining P4/P5 against actual free; stage PASS is not a guarantee of later13.016GiB budget'}
    summary={'task_id':'P4-02','pass':all(checks.values()),'stage_verdict':'PASS' if all(checks.values()) else 'FAIL',
        'checks':checks,'test_count':count,'P4_fork_test_count':len(new_tests),'branches':results,
        'lambda_selection':selection,'selected_SK_checkpoints':selected_models,'matched_controls':control,
        'resources':resources,'next_task':'P4-03','final_test_used':False,'formal_S1_SK':'RUN_COMPLETE' if all(r['completed'] for r in results) else 'INCOMPLETE',
        'MPC':'NOT_RUN','alignment_outcome':'NOT_EVALUATED','personal_understanding':'NOT_ASSESSED',
        'scope':'matched final-update training and reused-validation selection; no held-out test, action-choice improvement, alignment superiority or personal-understanding proof'}
    save(out/'resources.json',resources); save(out/'summary.json',summary)
    if not summary['pass']:raise ValueError('P4-02 incomplete/ineligible; preserve branches and never select fallback SK')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run-id',required=True); args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id):raise ValueError('Unsafe run id')
    out=ROOT/'runs'/args.run_id; out.mkdir(parents=True,exist_ok=False)
    try:summary=run(args,out)
    except Exception as error:
        save(out/'failure.json',{'task_id':'P4-02','pass':False,'error':str(error),'traceback':traceback.format_exc()})
        print(json.dumps({'run':str(out),'pass':False,'error':str(error)}),flush=True); return 1
    finally:
        save(out/'artifact_hashes.json',{'sha256':{str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'}})
    print(json.dumps({'run':str(out),'pass':summary['pass'],'selected_lambda':summary['lambda_selection']['selected_lambda']}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
