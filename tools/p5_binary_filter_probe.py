"""Read-only bit-preserving compression probe; no final trial or source edit."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,io,json,lzma,struct,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def sh(data):return hashlib.sha256(data).hexdigest()
def main():
    report=json.loads((ROOT/'reports/p5_storage_feasibility.native_codec.json').read_text())
    row=max(report['rows'],key=lambda r:r['average_increment_bytes_per_interval'])
    p=next(Path(p) for p in row['full_receipt']['files_sha256'] if p.endswith('.xz'))
    expected=row['full_receipt']['files_sha256'][str(p)]
    original=p.read_bytes();assert sh(original)==expected
    decoded=lzma.decompress(original);magic=b'P5COLRAW1\n';assert decoded.startswith(magic)
    hlen=struct.unpack('<Q',decoded[len(magic):len(magic)+8])[0]
    offset=len(magic)+8+hlen;header=decoded[:offset]
    meta=json.loads(decoded[len(magic)+8:offset]);raws=[]
    for e in meta['arrays']:
        r=decoded[offset:offset+e['bytes']];offset+=e['bytes'];assert sh(r)==e['sha256'];raws.append(r)
    assert offset==len(decoded)
    rows=[]
    for transform in ['identity','shuffle','xor_shuffle','delta_u64_shuffle']:
        parts=[]
        for e,raw in zip(meta['arrays'],raws):
            if np.dtype(e['dtype']).itemsize==8:
                arr=np.frombuffer(raw,np.uint64).copy()
                if transform=='xor_shuffle' and len(arr)>1:arr[1:]=np.bitwise_xor(arr[1:].copy(),arr[:-1].copy())
                if transform=='delta_u64_shuffle' and len(arr)>1:arr[1:]=arr[1:].copy()-arr[:-1].copy()
                result=arr.view(np.uint8).reshape(-1,8).T.copy().tobytes() if transform!='identity' else raw
                restored=np.frombuffer(result,np.uint8).reshape(8,-1).T.copy().reshape(-1).view(np.uint64) if transform!='identity' else arr
                if transform=='xor_shuffle':restored=np.bitwise_xor.accumulate(restored)
                if transform=='delta_u64_shuffle':restored=np.cumsum(restored,dtype=np.uint64)
                assert restored.tobytes()==raw
                parts.append(result)
            else:parts.append(raw)
        payload=header+b''.join(parts)
        for delta in [None,1,8]:
            filters=([{'id':lzma.FILTER_DELTA,'dist':delta}] if delta else [])+[{'id':lzma.FILTER_LZMA2,'preset':6}]
            t=time.perf_counter();compressed=lzma.compress(payload,filters=filters)
            assert lzma.decompress(compressed)==payload
            rows.append({'transform':transform,'delta_filter_distance':delta,'bytes':len(compressed),'wall_s':time.perf_counter()-t,'reversible_exact':True})
    r={'schema':'p5-binary-filter-probe-v1','checked_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'One already-preserved highest measured short record; no production codec change or worst-case guarantee',
      'source':str(p.relative_to(ROOT)),'source_sha256':expected,'intervals':row['intervals'],'baseline_xz_bytes':len(original),'npz_bytes_unchanged':sum(v for k,v in row['full_receipt']['bytes'].items() if k.endswith('.npz')),
      'decoded_bytes':len(decoded),'variants':rows,'best':min(rows,key=lambda r:r['bytes']),'original_unchanged':sh(p.read_bytes())==expected,'model_forward':0,'physics':0,'training':0}
    with (ROOT/'reports/p5_binary_filter_probe.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps({'baseline_xz_bytes':r['baseline_xz_bytes'],'npz_bytes_unchanged':r['npz_bytes_unchanged'],'best':r['best'],'all_reversible':all(x['reversible_exact'] for x in rows)}))
if __name__=='__main__':main()
