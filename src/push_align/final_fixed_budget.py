"""File-backed fixed-allowance credits for one frozen P5 run.

The coordinator chooses semantically eligible evidence. This engine checks its
frozen category scopes, exact files and immutable revision chain. It never
credits estimated counters, unit-priced data, the reserve, or its own files.
New snapshots append only new entries, avoiding quadratic full-graph copies.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat


POOLS = ('E1_prediction_analysis','P5_boundary_analysis','source_and_model_bundle',
         'future_report_archive_QA','working_disk_scratch',
         'mismatch_prefix_and_diagnostics','hard_free_reserve')
RESERVE = 'hard_free_reserve'
SCHEMA = 'p5-fixed-spend-delta-v1'
RECEIPT_SCHEMA = 'p5-fixed-spend-external-sha-v1'
_NAME = re.compile(r'^fixed_spend(?:\.([0-9]{6}))?(\.receipt)?\.json$')
_SHA = re.compile(r'^[0-9a-f]{64}$')


def _need(condition, message):
    if not condition:
        raise RuntimeError('Fixed budget: '+message)


def _digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):
            result.update(block)
    return result.hexdigest()


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _load(path):
    def unique(pairs):
        result = {}
        for key,value in pairs:
            _need(key not in result,'duplicate JSON key in immutable evidence')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(),object_pairs_hook=unique)


def _relative(value, *, prefix=False):
    _need(type(value) is str and bool(value) and '\\' not in value,'canonical run-relative scope/path required')
    text = value[:-1] if prefix and value.endswith('/') else value
    parts = PurePosixPath(text).parts
    _need(not text.startswith('/') and bool(parts) and all(part not in ('.','..') for part in parts)
          and str(PurePosixPath(text)) == text,'scope/path is absolute or noncanonical')
    return text+'/' if prefix else text


def _contract(run, manifest_sha256, allowances, scopes):
    run = Path(run).resolve()
    _need(run.is_dir(),'existing run directory required')
    _need(type(manifest_sha256) is str and _SHA.fullmatch(manifest_sha256) is not None,
          'valid expected frozen manifest SHA required')
    manifest = run/'manifest.json'
    _need(manifest.is_file() and not manifest.is_symlink() and manifest.resolve() == manifest
          and _digest(manifest) == manifest_sha256,
          'current run manifest missing or differs from expected SHA')
    _need(type(allowances) is dict and set(allowances) == set(POOLS)
          and all(type(value) is int and value >= 0 for value in allowances.values()),
          'exact seven nonnegative integer fixed allowances required')
    normalized = {}
    _need(type(scopes) is dict,'explicit category scopes required')
    for category,scope in scopes.items():
        _need(type(category) is str and bool(re.fullmatch(r'[A-Za-z0-9_-]+',category))
              and type(scope) is dict and set(scope) == {'pool','prefixes','exact_paths'},'malformed category scope')
        pool = scope['pool']
        _need(pool in POOLS and pool != RESERVE,'reserve is never a creditable category')
        selected = {}
        for key in ('prefixes','exact_paths'):
            values = scope[key]
            _need(type(values) in (list,tuple,set) and all(type(value) is str for value in values),
                  'scope paths must be an explicit string collection')
            paths = [_relative(value,prefix=key == 'prefixes') for value in values]
            _need(len(paths) == len(set(paths)),'duplicate category scope path')
            selected[key] = sorted(paths)
        _need(selected['prefixes'] or selected['exact_paths'],'empty category scope cannot authorize files')
        normalized[category] = {'pool':pool,**selected}
    contract = {'run':str(run),'manifest_sha256':manifest_sha256,
                'allowances':dict(allowances),'scopes':normalized}
    return run,contract,_hash(contract)


def _paths(run, revision):
    stem = 'fixed_spend' if revision == 0 else f'fixed_spend.{revision:06d}'
    return run/(stem+'.json'),run/(stem+'.receipt.json')


def _is_managed(relative):
    return '/' not in relative and relative.startswith('fixed_spend.')


def _file_path(run, value):
    _need(type(value) is str,'file path must be a string')
    path = Path(value)
    if path.is_absolute():
        _need(path == path.resolve() and path.is_relative_to(run),'file belongs to another run or has an alias')
        relative = _relative(path.relative_to(run).as_posix())
    else:
        relative = _relative(value)
        path = run/relative
    _need(not _is_managed(relative),'fixed-spend ledger/receipt cannot credit itself')
    _need(path.resolve().is_relative_to(run) and path.resolve() == path,'symlink/escaping file cannot earn a credit')
    info = path.stat()
    _need(stat.S_ISREG(info.st_mode) and not path.is_symlink(),'only existing regular files can earn a credit')
    _need(info.st_nlink == 1,'duplicate hardlink/inode alias cannot earn a credit')
    return path,relative,info


def _in_scope(relative, category, scopes):
    _need(category in scopes,'unknown category')
    scope = scopes[category]
    _need(relative in scope['exact_paths'] or any(relative.startswith(prefix) for prefix in scope['prefixes']),
          'file is outside the frozen category scope: '+relative)
    return scope['pool']


def _check_entry(run, entry, contract, *, deep):
    _need(type(entry) is dict and set(entry) == {'path','sha256','size_bytes','category','pool'},
          'malformed file receipt; arbitrary counters are forbidden')
    relative = _relative(entry['path'])
    path,relative,info = _file_path(run,relative)
    pool = _in_scope(relative,entry['category'],contract['scopes'])
    _need(entry['pool'] == pool and pool != RESERVE,'file category/pool differs from frozen scope')
    _need(type(entry['sha256']) is str and _SHA.fullmatch(entry['sha256']) is not None
          and type(entry['size_bytes']) is int and entry['size_bytes'] >= 0
          and info.st_size == entry['size_bytes'],'file missing or size differs from its receipt')
    if deep:
        _need(_digest(path) == entry['sha256'],'file SHA changed: '+relative)
        after = path.stat()
        _need((info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns) ==
              (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'file changed during SHA verification')
    return str(path),(info.st_dev,info.st_ino)


def _chain(run, contract, contract_sha256):
    inventory = {}
    for path in run.glob('fixed_spend.*'):
        match = _NAME.fullmatch(path.name)
        _need(match is not None,'unrecognized or partial fixed-spend revision: '+path.name)
        revision = int(match.group(1)) if match.group(1) is not None else 0
        _need(revision != 0 or match.group(1) is None,'noncanonical zero revision')
        kind = 'receipt' if match.group(2) else 'payload'
        _need((revision,kind) not in inventory,'duplicate fixed-spend revision')
        _need(path.is_file() and not path.is_symlink(),'ledger revision is not a regular file')
        _need(path.stat().st_nlink == 1,'ledger hardlink/inode alias violates immutable ownership')
        inventory[revision,kind] = path
    if not inventory:
        return [],None,None,{}
    revisions = sorted({revision for revision,_ in inventory})
    _need(revisions == list(range(revisions[-1]+1)),'fixed-spend revision gap; review retained prefix')
    previous = None; entries = []; ledger_files = {}
    for revision in revisions:
        _need((revision,'payload') in inventory and (revision,'receipt') in inventory,
              'incomplete fixed-spend snapshot/receipt; review retained prefix')
        path,receipt_path = _paths(run,revision)
        digest = _digest(path); receipt = _load(receipt_path)
        _need(type(receipt) is dict and set(receipt) == {'schema','revision','run','manifest_sha256','sha256'}
              and receipt['schema'] == RECEIPT_SCHEMA and type(receipt['revision']) is int
              and receipt['revision'] == revision and receipt['run'] == str(run)
              and receipt['manifest_sha256'] == contract['manifest_sha256'] and receipt['sha256'] == digest,
              'external ledger SHA/run/manifest receipt differs')
        value = _load(path)
        required = {'schema','revision','run','manifest_sha256','contract_sha256','previous_snapshot_sha256','entries'}
        if revision == 0:
            required.add('contract')
        _need(type(value) is dict and set(value) == required and value['schema'] == SCHEMA
              and type(value['revision']) is int and value['revision'] == revision
              and value['run'] == str(run) and value['manifest_sha256'] == contract['manifest_sha256']
              and value['contract_sha256'] == contract_sha256 and value['previous_snapshot_sha256'] == previous,
              'ledger chain/contract/run/manifest binding differs')
        if revision == 0:
            _need(value['contract'] == contract,'original allowances/category scopes changed')
        added = value['entries']
        _need(type(added) is list and bool(added),'empty or malformed credit snapshot')
        entries.extend(added)
        previous = digest
        ledger_files[str(path)] = digest; ledger_files[str(receipt_path)] = _digest(receipt_path)
    return entries,revisions[-1],previous,ledger_files


def _result(run, contract, entries, revision, digest, ledger_files, *, deep):
    pools = {pool:{'allowance_bytes':contract['allowances'][pool],'spent_bytes':0,
                   'remaining_bytes':contract['allowances'][pool],'files':[]} for pool in POOLS}
    paths,identities = {},set()
    for entry in entries:
        path,identity = _check_entry(run,entry,contract,deep=deep)
        _need(path not in paths and identity not in identities,'duplicate file/hardlink credit across categories or pools')
        identities.add(identity); paths[path] = dict(entry)
        pool = entry['pool']; pools[pool]['spent_bytes'] += entry['size_bytes']; pools[pool]['files'].append(path)
    for pool,value in pools.items():
        _need(value['spent_bytes'] <= value['allowance_bytes'],'fixed pool overspent: '+pool)
        value['remaining_bytes'] -= value['spent_bytes']
    _need(pools[RESERVE]['spent_bytes'] == 0 and pools[RESERVE]['remaining_bytes'] == contract['allowances'][RESERVE],
          'hard reserve must remain entirely uncredited')
    return {'schema':'p5-fixed-spend-review-v1','status':'PASS','run':str(run),
        'manifest_sha256':contract['manifest_sha256'],'revision':revision,
        'ledger_path':str(_paths(run,revision)[0]) if revision is not None else None,
        'ledger_sha256':digest,'ledger_files_sha256':ledger_files,'pools':pools,'files':paths,
        'allowances_bytes':dict(contract['allowances']),
        'spent_bytes_by_pool':{pool:value['spent_bytes'] for pool,value in pools.items()},
        'remaining_allowances_bytes':{pool:value['remaining_bytes'] for pool,value in pools.items()},
        'total_fixed_allowance_bytes':sum(contract['allowances'].values()),
        'total_fixed_spent_bytes':sum(value['spent_bytes'] for value in pools.values()),
        'total_fixed_remaining_bytes':sum(value['remaining_bytes'] for value in pools.values()),
        'verification':'full_file_sha256' if deep else 'current_file_existence_size_and_immutable_ledger_sha',
        'limits':['Coordinator-owned frozen semantic selection decides which evidence is eligible; this engine validates the supplied immutable scopes and file graph.',
                  'Light reads do not rehash credited file contents; caller must independently deep-verify at stage/resume boundaries.',
                  'Ledger/receipt bytes never credit themselves; the coordinator retains an uncredited allowance for their overhead.']}


def read_fixed_spend(run, *, manifest_sha256, allowances, scopes, deep=True):
    """Read the complete immutable graph; no files are created or repaired.

    An absent graph returns zero credit. Any partial/gapped/changed graph stops,
    including a durable payload whose external receipt was not written yet.
    """
    try:
        _need(type(deep) is bool,'explicit boolean verification mode required')
        run,contract,contract_sha = _contract(run,manifest_sha256,allowances,scopes)
        entries,revision,digest,ledger_files = _chain(run,contract,contract_sha)
        return _result(run,contract,entries,revision,digest,ledger_files,deep=deep)
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError('Fixed budget: '+type(error).__name__+': '+str(error)) from error


def _exclusive_json(path, value, *, before_write_bytes=None):
    payload = (json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')
    if before_write_bytes is not None:
        decision = before_write_bytes(len(payload))
        accepted = decision is None or decision is True or (
            type(decision) is dict and decision.get('status') in {'PASS','CONTINUE_RESOURCE_CHECK_ONLY'})
        _need(accepted,'ledger write declined by resource admission')
    with Path(path).open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def _sync_directory(run):
    descriptor = os.open(run,os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def add_fixed_spend(run, *, manifest_sha256, allowances, scopes, category, files_sha256, deep=True,
                    before_write_bytes=None):
    """Append verified new files exclusively; duplicate calls intentionally fail.

    Existing graph validation honors ``deep``; every newly added file is always
    SHA-verified and stat-sized. All checks precede mutation. A failed durable
    write is retained and a later read fails closed, never inventing a receipt.
    Production callers pass ``before_write_bytes(n)`` for every exact UTF-8
    payload/receipt write. It approves by returning None/True/PASS/CONTINUE or
    stops by raising/returning a nonapproval. It must not recursively read the
    graph between the two exclusive writes, when the new snapshot is partial.
    """
    try:
        _need(type(deep) is bool,'explicit boolean verification mode required')
        _need(before_write_bytes is None or callable(before_write_bytes),'callable prewrite byte admission required')
        run,contract,contract_sha = _contract(run,manifest_sha256,allowances,scopes)
        entries,revision,previous,ledger_files = _chain(run,contract,contract_sha)
        _result(run,contract,entries,revision,previous,ledger_files,deep=deep)
        _need(category in contract['scopes'],'unknown category')
        _need(type(files_sha256) is dict and bool(files_sha256),'nonempty exact path/SHA file map required; no counters')
        added = []
        for value,digest in files_sha256.items():
            path,relative,info = _file_path(run,value)
            pool = _in_scope(relative,category,contract['scopes'])
            added.append({'path':relative,'sha256':digest,'size_bytes':info.st_size,'category':category,'pool':pool})
        added.sort(key=lambda value:value['path'])
        # Check all new files and aggregate limits without changing durable state.
        for entry in added:
            _check_entry(run,entry,contract,deep=True)
        next_revision = 0 if revision is None else revision+1
        _need(next_revision <= 999999,'fixed-spend revision capacity reached; preserve and review')
        _result(run,contract,[*entries,*added],next_revision,previous,ledger_files,deep=False)
        payload = {'schema':SCHEMA,'revision':next_revision,'run':str(run),
            'manifest_sha256':manifest_sha256,'contract_sha256':contract_sha,
            'previous_snapshot_sha256':previous,'entries':added}
        if next_revision == 0:
            payload['contract'] = contract
        path,receipt_path = _paths(run,next_revision)
        _exclusive_json(path,payload,before_write_bytes=before_write_bytes)
        _exclusive_json(receipt_path,{'schema':RECEIPT_SCHEMA,'revision':next_revision,'run':str(run),
            'manifest_sha256':manifest_sha256,'sha256':_digest(path)},before_write_bytes=before_write_bytes)
        _sync_directory(run)
        # Both snapshots are now durable. Re-read the chain rather than trusting
        # a counter assembled before the exclusive write boundary.
        return read_fixed_spend(run,manifest_sha256=manifest_sha256,allowances=allowances,scopes=scopes,deep=deep)
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError('Fixed budget: '+type(error).__name__+': '+str(error)) from error
