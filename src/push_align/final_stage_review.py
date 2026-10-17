"""Read-only scientific contracts for sealed P5 derived stages.

The caller verifies the external stage-receipt seal and frozen runtime sources
before entry. This module binds the exact current E2 source population, then
recomputes statistics or checks stored prediction arithmetic without inference.
It does not authorize execution or claim that saved deltas were newly inferred.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import gc
import importlib
import json
from pathlib import Path
import resource
import sys
import zipfile

import numpy as np

from .closed_loop import ENGINE_GUARDS
from .contact_geometry import check_contact_geometry
from .environment import STATE_FIELDS, wrap_angle
from .final_driver import exact_comparison, read_full_trace
from .final_manifest import CONTROLLERS, STRATA, expected_E2
from .final_prediction import (ERROR_METRICS, MODELS, REGIMES, TRUTH_MASK_PROVENANCE,
    _group_summary, _mean, _population, _windows)
from .final_recording import file_sha256, read_json_gzip
from .final_statistics import analyze_E2
from .planning import _state_reasons
from .representation import RepresentationError, integrate

MAX_RSS_BYTES = 6 * 1024**3
MAX_RAW_BYTES = 64 * 1024**2
MAX_NPZ_EXPANDED_BYTES = 512 * 1024**2


def _need(condition, message):
    if not condition:
        raise RuntimeError('Derived stage review: '+message)


def _memory():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    _need(int(value)*(1 if sys.platform == 'darwin' else 1024) <= MAX_RSS_BYTES,
          'RSS checkpoint exceeded; preserve artifacts and reestimate')


def _read(path):
    return json.loads(Path(path).read_text())


def _json_equal(left, right):
    # JSON roundtrip makes integer dictionary keys match their saved spelling,
    # while preserving booleans/numbers and signed floating-point zero.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _array_equal(actual, expected, label):
    _need(isinstance(actual, np.ndarray) and actual.dtype == expected.dtype
          and actual.shape == expected.shape, label+': dtype/shape changed')
    _need(actual.tobytes() == expected.tobytes(), label+': exact array values changed')


def _typed(array, dtype, shape, label):
    _need(array.dtype == np.dtype(dtype) and array.shape == shape, label+': dtype/shape invalid')


def _npz(path):
    with zipfile.ZipFile(path) as archive:
        names = [row.filename for row in archive.infolist()]
        _need(len(names) == len(set(names)) and all(name.endswith('.npy') and '/' not in name for name in names),
              'duplicate or malformed NPZ members')
        _need(sum(row.file_size for row in archive.infolist()) <= MAX_NPZ_EXPANDED_BYTES,
              'expanded NPZ allowance exceeded; preserve and reestimate')
    return np.load(path, allow_pickle=False)


def _population_rows(m, load_e2_rows):
    _need(m['E2'] == expected_E2(m['families']), 'frozen E2 population is not the declared96 cells')
    rows = list(load_e2_rows())
    expected = {row['cell_id']:row for row in m['E2']}
    _need(len(rows) == 96 and len({row['cell_id'] for row in rows}) == 96
          and {row['cell_id'] for row in rows} == set(expected), 'verified source population is not exactly96 unique rows')
    for row in rows:
        _need(all(row.get(key) == value for key,value in expected[row['cell_id']].items())
              and row.get('execution_status') == 'completed', 'verified E2 identity/status differs')
    return rows


def _stage_sources(run, name, m):
    if name == 'statistics':
        paths = [run/'E2'/cell['cell_id']/'worker_result.json' for cell in m['E2']]
        return {str(path):file_sha256(path) for path in paths}, None
    sources, records = {}, []
    for family in m['families']:
        folder = run/'E2'/family['family_id']/'G'
        path = folder/'manifest.json'
        record = _read(path)
        identity, execution = record['identity'], record['execution']
        _need(record.get('schema') == 'p5-recorded-episode-v1'
              and record.get('record_kind') == 'reference_controller_execution'
              and record.get('recording_status') in {'COMPLETE_RECORDED','STOP_REVIEW'}
              and identity.get('family_id') == family['family_id'] and identity.get('controller_id') == 'G'
              and identity.get('model') is None and identity.get('normalizer') is None
              and identity.get('final_manifest_sha256') == file_sha256(run/'manifest.json')
              and execution.get('run_scope') == 'final_evaluation'
              and execution.get('execution_status') == 'completed', 'G source is not this completed frozen family')
        files = record['files_sha256']
        _need(type(files) is dict and files, 'G source file graph missing')
        _need(sum(Path(p).stat().st_size for p in files) <= MAX_RAW_BYTES, 'G cell exceeds64MiB review allowance')
        required = {record['witness_path'], *record['raw']['files_sha256']}
        for trace in record['full_traces']:
            required.update(trace['files_sha256'])
        _need(required <= set(files), 'G raw/witness/fullinput is outside the source graph')
        sources[str(path)] = file_sha256(path)
        for p,h in files.items():
            _need(Path(p).resolve().is_relative_to(folder) and file_sha256(p) == h, 'G source file changed or escapes its cell')
            sources[p] = h
        records.append((family, record))
    return sources, records


def _stage_contract(run, name, m):
    required = {'statistics':{'summary.json','bootstrap.npz','reproduction.json'},
                'fixed_prediction':{'summary.json','traces.npz'}}[name]
    folder = run/name
    receipt = _read(folder/'stage_receipt.json')
    _need(receipt.get('schema') == 'p5-derived-stage-seal-v1' and receipt.get('stage') == name
          and receipt.get('frozen_manifest_sha256') == file_sha256(run/'manifest.json'), 'stage receipt identity differs')
    expected_files = {str(folder/file) for file in required}
    _need(type(receipt.get('files_sha256')) is dict and set(receipt['files_sha256']) == expected_files,
          'stage file graph is missing or adds undeclared files')
    for p,h in receipt['files_sha256'].items():
        _need(Path(p).is_file() and file_sha256(p) == h, 'derived file missing or changed: '+p)
    sources, records = _stage_sources(run, name, m)
    _need(receipt.get('source_sha256') == sources, 'exact E2 source graph or SHA differs')
    _memory()
    return receipt, records


def _statistics(run, m, rows):
    settings = m['statistics']
    _need(settings.get('replicates') == 2000 and type(settings['replicates']) is int
          and settings.get('seed') == 7711 and type(settings['seed']) is int,
          'bootstrap must retain2000 shared draws and seed7711')
    summary, arrays = analyze_E2(rows, m['E2'], bootstrap_replicates=2000, bootstrap_seed=7711)
    folder = run/'statistics'
    _need(_json_equal(_read(folder/'summary.json'), summary), 'summary does not regenerate from verified96 rows')
    with _npz(folder/'bootstrap.npz') as saved:
        _need(set(saved.files) == set(arrays), 'bootstrap NPZ full key population differs')
        for key, expected in arrays.items():
            _array_equal(saved[key], expected, 'bootstrap '+key)
    reproduction = _read(folder/'reproduction.json')
    _need(reproduction.get('status') == 'PASS_EXACT_REGENERATION'
          and type(reproduction.get('source_rows')) is int and reproduction['source_rows'] == 96
          and type(reproduction.get('families')) is int and reproduction['families'] == 12
          and reproduction.get('summary_sha256') == file_sha256(folder/'summary.json')
          and reproduction.get('bootstrap_sha256') == file_sha256(folder/'bootstrap.npz'),
          'table reproduction receipt incomplete or bound to another output')
    _memory()
    return {'verified_rows':96,'families':12,'controllers':8,'bootstrap_replicates':2000,
            'bootstrap_seed':7711,'exact_npz_arrays':len(arrays),'all_statistics_regenerated':True}


def _g_trajectories(records, m, rows):
    by = {row['family_id']:row for row in rows if row['controller_id'] == 'G'}
    for family,record in records:
        _memory()
        traces = [trace for trace in record['full_traces'] if trace['tick'] == 0]
        _need(len(traces) == 1, 'G needs one complete tick0 source input')
        initial, arrays, metadata = read_full_trace(traces[0])
        _need(not arrays and metadata['plan'] is None and initial.get('tick') == 0
              and initial.get('case_id') == family['family_id'] and initial.get('controller_id') == 'G'
              and initial.get('episode_index') == family['episode_index'], 'G input belongs to another family/tick')
        module = record.get('raw_codec_module')
        _need(module in {'push_align.final_recording','push_align.final_recording_columns'}, 'unapproved raw codec')
        codec = importlib.import_module(module)
        snapshot, steps = codec.read_raw(record['raw']['prefix'], expected_sha256=record['raw']['files_sha256'])
        _need(exact_comparison(snapshot,initial['snapshot'])['status'] == 'PASS_EXACT', 'G raw initial full snapshot differs')
        n = len(steps)
        _need(n <= 300 and n == record['raw']['interval_count']
              and n == record['execution']['counts']['physics_intervals_returned']
              and n == by[family['family_id']]['counts']['physics_intervals_returned'], 'G source interval counts differ')
        witness = read_json_gzip(record['witness_path'], expected_sha256=record['files_sha256'][record['witness_path']])
        _need(len(witness['intervals']) == n, 'G witness interval denominator differs')
        mask = []
        for step,boundary in zip(steps,witness['intervals']):
            geometry = check_contact_geometry(step.diagnostics['samples'],m['profile']['physics_suite']['actual_contact_penetration_max_m'])
            _need(_json_equal(geometry,boundary['geometry']), 'G source geometry mask does not regenerate')
            mask.append(bool(step.transition_valid and step.observation.observation_valid and geometry['pass']))
        yield {'family_id':family['family_id'],'stratum':family['stratum'],'controller_id':'G',
            'initial_state10':initial['observation'].state.copy(),'initial_sim_time':initial['observation'].sim_time,
            'results':steps,'interval_valid':np.asarray(mask,np.bool_),'truth_mask_provenance':TRUTH_MASK_PROVENANCE}
        del snapshot, steps, initial, arrays, metadata, witness, mask
        if n:
            del step,boundary
        gc.collect()


def _model_from_saved(saved, prefix, window, horizon, config):
    """Reapply delta integration/validity; never call a predictor."""
    count = len(window['truth_valid'])
    delta = saved[prefix+'raw_deltas']
    stored_reasons = saved[prefix+'reasons']
    _typed(delta,np.float64,(count,horizon,10),prefix+'raw_deltas')
    _typed(stored_reasons,'U200',(count,),prefix+'reasons')
    states = np.full((count,horizon+1,10),np.nan,np.float64)
    reasons = np.full(count,'not_evaluated_truth_ineligible',dtype='U200')
    failure = np.full(count,-2,np.int64)
    eligible = np.flatnonzero(window['truth_valid'])
    states[eligible,0] = window['truth_states'][eligible,0]
    reasons[eligible] = _state_reasons(states[eligible,0],config)
    failure[eligible] = np.where(reasons[eligible] == '',-1,0)
    min_calls = min_rows = max_calls = 0
    for tick in range(horizon):
        active = np.flatnonzero(reasons == '')
        inactive = np.flatnonzero(reasons != '')
        _need(np.isnan(delta[inactive,tick]).all(), prefix+'raw deltas retained after failed/ineligible prefix')
        if not len(active):
            continue
        min_calls += 1; min_rows += len(active); max_calls += 1+len(active)
        step_delta = delta[active,tick]
        bad = ~np.isfinite(step_delta).all(axis=1)
        failed = np.full(len(active),'',dtype='U200')
        failed[bad] = 'nonfinite_physical_delta'
        # An exception message is not derivable from raw NaNs. Keep its saved
        # explanation only at the exact nonfinite failed transition.
        for local in np.flatnonzero(bad):
            text = str(stored_reasons[active[local]])
            if text.startswith(('inference_rejected:ValueError:','inference_rejected:FloatingPointError:')):
                failed[local] = text
        failed[~bad & (np.abs(step_delta[:,2]) >= np.pi)] = 'raw_delta_yaw_abs_ge_pi'
        okay = np.flatnonzero(failed == '')
        if len(okay):
            try:
                states[active[okay],tick+1] = integrate(states[active[okay],tick],step_delta[okay])
            except RepresentationError:
                for local in okay:
                    try:
                        states[active[local],tick+1] = integrate(states[active[local],tick],step_delta[local])
                    except RepresentationError:
                        failed[local] = 'nonfinite_predicted_state'
            available = okay[failed[okay] == '']
            if len(available):
                failed[available] = _state_reasons(states[active[available],tick+1],config)
        failed_rows = active[failed != '']
        reasons[failed_rows] = failed[failed != '']; failure[failed_rows] = tick+1
    valid = reasons == ''
    errors = np.full((count,10),np.nan,np.float64)
    active = np.flatnonzero(valid)
    with np.errstate(over='ignore',invalid='ignore'):
        errors[active] = states[active,-1]-window['truth_states'][active,-1]
        errors[active,2] = wrap_angle(errors[active,2])
    invalid = active[~np.isfinite(errors[active]).all(axis=1)]
    valid[invalid] = False; reasons[invalid] = 'nonfinite_endpoint_error'
    failure[invalid] = horizon; errors[invalid] = np.nan
    with np.errstate(over='ignore',invalid='ignore'):
        metrics = dict(zip(ERROR_METRICS,(np.hypot(errors[:,0],errors[:,1]),np.abs(errors[:,2]),
            np.hypot(errors[:,6],errors[:,7]),np.hypot(errors[:,3],errors[:,4]),
            np.abs(errors[:,5]),np.hypot(errors[:,8],errors[:,9]))))
    finite = np.logical_and.reduce([np.isfinite(value) for value in metrics.values()])
    invalid = valid & ~finite
    valid[invalid] = False; reasons[invalid] = 'nonfinite_endpoint_metric'
    failure[invalid] = horizon; errors[invalid] = np.nan
    for value in metrics.values():
        value[invalid] = np.nan
    expected = {'predicted_states':states,'valid':valid,'reasons':reasons,'failure_steps':failure,
                'endpoint_state_error':errors,**metrics}
    for key,value in expected.items():
        _array_equal(saved[prefix+key],value,prefix+key)
    _memory()
    return valid,reasons,metrics,(min_calls,max_calls,min_rows,2*min_rows)


def _prediction(run, m, rows, records):
    summary = _read(run/'fixed_prediction/summary.json')
    required = {'schema','status','family_count','family_ids','source','horizons','regime_codes','state_fields',
        'yaw_error','recorded_intervals_by_family','source_interval_populations','truth_mask_provenance',
        'failure_step_codes','inference','future_engine_calls','online_settings_mutated','additional_physics_intervals','limitations'}
    _need(set(summary) == required and summary['schema'] == 'p5-common-G-fixed-trace-prediction-v1'
          and summary['status'] == 'COMPLETE_DECLARED_RAW_EVALUATION'
          and type(summary['family_count']) is int and summary['family_count'] == 12
          and summary['source'] == 'all_E2_G_raw_fixed_applied_actions'
          and set(summary['horizons']) == {'1','5','10'}
          and summary['truth_mask_provenance'] == TRUTH_MASK_PROVENANCE
          and summary['state_fields'] == list(STATE_FIELDS)
          and summary['regime_codes'] == {str(i):name for i,name in enumerate(REGIMES)}
          and summary['failure_step_codes'] == {'truth_not_evaluated':-2,'valid_prediction':-1,
              'invalid_initial_predicted_state':0,'prediction_transition':'1 through K'}
          and summary['online_settings_mutated'] is False and type(summary['additional_physics_intervals']) is int
          and summary['additional_physics_intervals'] == 0
          and isinstance(summary['limitations'],str) and bool(summary['limitations'])
          and summary['yaw_error'] == 'signed endpoint residual wrapped2pi; absolute scalar metric, not pi-symmetric task angle',
          'prediction summary schema/status/frozen scope differs')
    guards = summary['future_engine_calls']
    _need(type(guards) is dict and set(guards) == set(ENGINE_GUARDS)
          and all(type(value) is int and value == 0 for value in guards.values()), 'prediction guard counts are missing/nonzero')
    _need(type(summary['inference']) is dict and set(summary['inference']) == set(MODELS), 'seven-model inference population missing')
    prepared = _population(_g_trajectories(records,m,rows),m['physical_config'],m['planner_config'])
    ids = [row['family_id'] for row in prepared]
    _need(summary['family_ids'] == ids, 'prediction family IDs differ from frozen G sources')
    counts = {row['family_id']:len(row['actions']) for row in prepared}
    populations = {row['family_id']:{'stratum':row['stratum'],'recorded_intervals':len(row['actions']),
        'valid_intervals':sum(not reason for reason in row['interval_reasons']),
        'invalid_intervals':sum(bool(reason) for reason in row['interval_reasons']),
        'invalid_reasons':dict(Counter(reason for reason in row['interval_reasons'] if reason))} for row in prepared}
    _need(summary['recorded_intervals_by_family'] == counts and summary['source_interval_populations'] == populations,
          'raw interval masks/populations differ')
    inference = {name:[0,0,0,0] for name in MODELS}
    expected_keys = set()
    with _npz(run/'fixed_prediction/traces.npz') as saved:
        for horizon in (1,5,10):
            window = _windows(prepared,horizon)
            count = len(window['window_id']); eligible = np.flatnonzero(window['truth_valid'])
            for key,value in window.items():
                name = f'K{horizon}__{key}'; expected_keys.add(name)
                _array_equal(saved[name],value,name)
            models = {}
            config = deepcopy(m['planner_config']); config['planner']['horizon'] = horizon
            for name in MODELS:
                prefix = f'K{horizon}__{name}__'
                expected_keys.update(prefix+key for key in ('predicted_states','raw_deltas','valid','reasons',
                    'failure_steps','endpoint_state_error',*ERROR_METRICS))
                valid,reasons,metrics,bounds = _model_from_saved(saved,prefix,window,horizon,config)
                inference[name] = [left+right for left,right in zip(inference[name],bounds)]
                families = {row['family_id']:{'stratum':row['stratum'],**_group_summary(window,valid,reasons,metrics,
                    window['family_id'] == row['family_id'])} for row in prepared}
                means = {metric:[value['metrics_conditional_on_valid_prediction_and_truth'][metric]['mean']
                    for value in families.values()] for metric in ERROR_METRICS}
                models[name] = {**_group_summary(window,valid,reasons,metrics,np.ones(count,np.bool_)),
                    'by_first_interval_regime':{regime:_group_summary(window,valid,reasons,metrics,window['regime_code'] == code)
                        for code,regime in enumerate(REGIMES)},'by_family':families,
                    'family_first_means_conditional_on_available_families':{metric:{'family_denominator':sum(value is not None for value in values),
                        'mean':_mean([value for value in values if value is not None])} for metric,values in means.items()}}
            boundary = window['truth_reason'] == 'insufficient_remaining_intervals'
            expected = {'recorded_start_opportunities':count,'boundary_excluded_windows':int(boundary.sum()),
                'full_horizon_windows':int((~boundary).sum()),'raw_invalid_full_windows':int((~boundary & ~window['truth_valid']).sum()),
                'truth_eligible_windows':len(eligible),'prediction_comparison_status':'EVALUATED_ON_ELIGIBLE_WINDOWS' if len(eligible) else 'NO_ELIGIBLE_GROUND_TRUTH_WINDOWS',
                'truth_exclusion_reasons':dict(Counter(str(value) for value in window['truth_reason'][~window['truth_valid']])), 'models':models}
            _need(_json_equal(summary['horizons'][str(horizon)],expected), 'prediction K'+str(horizon)+' summary/denominators do not regenerate')
            _memory()
        _need(set(saved.files) == expected_keys, 'prediction NPZ full key population differs')
    for name,(min_calls,max_calls,min_rows,max_rows) in inference.items():
        observed = summary['inference'][name]
        _need(set(observed) == {'calls','rows'} and type(observed['calls']) is int and type(observed['rows']) is int
              and min_calls <= observed['calls'] <= max_calls and min_rows <= observed['rows'] <= max_rows,
              'inference counts contradict retained prediction opportunities')
    return {'verified_rows':96,'families':12,'models':7,'horizons':[1,5,10],
        'recorded_start_opportunities_per_horizon':sum(counts.values()),'exact_npz_arrays':len(expected_keys),
        'raw_truth_masks_and_windows_regenerated':True,'stored_delta_integration_and_all_errors_checked':True}


def validate_derived_stage(run, name, m, *, load_e2_rows):
    """Return a PASS receipt or raise RuntimeError; never write or infer.

    ``load_e2_rows()`` must independently return the verified frozen96 task rows.
    The external stage-receipt seal is caller-owned; this function rechecks its
    scientific file/source contract and the supplied frozen manifest equality.
    """
    try:
        _memory()
        _need(name in {'statistics','fixed_prediction'}, 'unsupported stage')
        run = Path(run).resolve()
        _need(_json_equal(_read(run/'manifest.json'),m), 'caller uses another frozen manifest')
        _need(callable(load_e2_rows), 'verified E2 row loader required')
        rows = _population_rows(m,load_e2_rows)
        receipt, records = _stage_contract(run,name,m)
        checks = _statistics(run,m,rows) if name == 'statistics' else _prediction(run,m,rows,records)
        _memory()
        return {'schema':'p5-derived-scientific-review-v1','status':'PASS','stage':name,
            'frozen_manifest_sha256':file_sha256(run/'manifest.json'),
            'files_sha256':receipt['files_sha256'],'source_sha256':receipt['source_sha256'],'checks':checks,
            'new_model_forwards':0,'new_physical_calls':0,'new_training_updates':0,
            'limits':['External stage seal and frozen runtime-source integrity are checked by the caller before this API.',
                'Statistics are fully regenerated from caller-verified rows; this module does not replace their raw task reviews.',
                'Prediction raw truth, saved delta integration/validity, arrays and summary arithmetic are checked without new inference. Equality of saved deltas to frozen model outputs remains source-bound producer evidence.',
                'Inference fallback counts are checked against possible active-row bounds; original exception messages and exact fallback call counts cannot be regenerated from saved deltas.']}
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError('Derived stage review: '+type(error).__name__+': '+str(error)) from error
