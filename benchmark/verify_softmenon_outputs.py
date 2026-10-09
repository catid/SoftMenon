#!/usr/bin/env python3
"""Verify final CPU/CUDA binaries against recorded SoftMenon/paper output hashes.

This does not rescore images. It verifies every selected image, crop and CFA
against benchmark/results/quality-cpu.json.gz.
"""
import argparse,gzip,hashlib,importlib.util,json,os,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('benchmark_helpers',ROOT/'benchmark/run.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
METHODS={'softmenon':'softmenon','menon2007':'menon2007'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def normalize(v):
    if isinstance(v,dict):return {normalize(k):normalize(x) for k,x in v.items()}
    if isinstance(v,list):return [normalize(x) for x in v]
    if isinstance(v,str):return v.replace(str(ROOT)+'/', '')
    return v
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,required=True)
    p.add_argument('--rows',type=Path,default=ROOT/'benchmark/results/quality-cpu.json.gz')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backends',nargs='+',choices=['cpu','cuda'],default=['cpu','cuda'])
    p.add_argument('--methods',nargs='+',choices=list(METHODS),default=list(METHODS))
    p.add_argument('--workers',type=int,default=8);p.add_argument('--limit',type=int);a=p.parse_args()
    if a.workers<1 or (a.limit is not None and a.limit<1):p.error('Positive workers and optional limit required')
    build=json.loads(a.build.read_text());checked={str(a.build.resolve()):sha(a.build),str(Path(__file__).resolve()):sha(__file__)}
    for name in a.backends:
        entry=build[name]
        if sha(entry['library'])!=entry['library_sha256']:raise RuntimeError('Library changed')
        checked[entry['library']]=entry['library_sha256']
        for path,digest in entry['source_sha256'].items():
            target=(ROOT/path).resolve()
            if sha(target)!=digest:raise RuntimeError(f'Source changed: {path}')
            checked[str(target)]=digest
    summary_path=a.rows.parent/'summary.json';summary=json.loads(summary_path.read_text())
    if sha(a.rows)!=summary['artifact_sha256'][a.rows.name]:raise RuntimeError('Quality artifact changed')
    with gzip.open(a.rows,'rt') as stream:metadata=json.load(stream)
    rows=metadata['rows']
    expected={(r['path'],r['inset'],r['pattern'],r['method']):r['output_sha256_rgb'] for r in rows if r['method'] in a.methods}
    manifest_path=ROOT/'benchmark/datasets.json';manifest=json.loads(manifest_path.read_text())
    if sha(manifest_path)!=metadata['manifest_sha256']:raise RuntimeError('Dataset manifest changed')
    checked[str(a.rows.resolve())]=sha(a.rows);checked[str(summary_path.resolve())]=sha(summary_path)
    checked[str(manifest_path.resolve())]=sha(manifest_path);checked[str(ROOT/'benchmark/run.py')]=sha(ROOT/'benchmark/run.py')
    cases=manifest['images'][:a.limit] if a.limit else manifest['images']
    backends={name:helper.Backend(build[name],a.workers if name=='cpu' else 0) for name in a.backends}
    count={name:0 for name in a.backends};start=time.time()
    try:
      for i,case in enumerate(cases):
        original=helper.load_image(case)
        for inset in (0,16):
          rgb=original[inset:-inset,inset:-inset].copy() if inset else original
          for pattern in helper.PATTERNS:
            raw=helper.mosaic(rgb,pattern)
            for method in a.methods:
              digest=expected[case['path'],inset,pattern,method]
              for name,backend in backends.items():
                out=backend.process(raw,METHODS[method],pattern)
                if hashlib.sha256(out.tobytes()).hexdigest()!=digest:raise RuntimeError(f'Output changed: {name}/{method}/{case["path"]}/{inset}/{pattern}')
                count[name]+=1
        if (i+1)%50==0 or i+1==len(cases):print(f'{i+1}/{len(cases)} images {count}',flush=True)
      for path,digest in checked.items():
        if sha(path)!=digest:raise RuntimeError(f'Frozen input changed during verification: {path}')
      required=len(cases)*4*len(a.methods)
      if any(n!=required for n in count.values()):raise RuntimeError('Incomplete verification')
      report={'complete':True,'images':len(cases),'methods':a.methods,'insets':[0,16],'patterns':list(helper.PATTERNS),
        'exact_output_hash_matches':count,'builds':{name:build[name] for name in a.backends},'checked_files':checked,
        'rows_sha256':sha(a.rows),'summary_sha256':sha(summary_path),'script_sha256':sha(__file__),
        'elapsed_seconds':time.time()-start,'cpu_affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        'workers':a.workers,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'scope':'Full corpus output identity only. This is not a new quality estimate or inference benchmark.'}
      a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(normalize(report),indent=2)+'\n')
    finally:
      for b in backends.values():b.close()
if __name__=='__main__':main()
