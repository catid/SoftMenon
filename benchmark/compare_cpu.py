#!/usr/bin/env python3
"""Compare frozen CPU builds with interleaved calls and exact output checks.

Example (run from the current checkout after building both manifests):
  taskset -c <eight physical cores> python benchmark/compare_cpu.py \
    --before build/cpu-before/build.json --before-root build/paper-before \
    --after build/cpu-after/build.json --after-root . --output build/cpu-compare.json
The old checkout can be created with: git worktree add --detach build/paper-before ad2eb36
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
from PIL import Image, __version__ as pillow_version

ROOT=Path(__file__).resolve().parents[1]
METHODS={'malvar':2,'menon2007':3,'softmenon':4}
PATTERNS={'RGGB':((0,1),(1,2)),'BGGR':((2,1),(1,0))}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def load_build(path,source_root):
    entry=json.loads(path.read_text())['cpu']
    library=Path(entry['library']).resolve()
    if sha(library)!=entry['library_sha256']:raise RuntimeError(f'Binary hash mismatch: {library}')
    checked={str(path.resolve()):sha(path),str(library):sha(library)}
    if source_root:
        for name,digest in entry['source_sha256'].items():
            source=(source_root/name).resolve()
            if sha(source)!=digest:raise RuntimeError(f'Source hash mismatch: {source}')
            checked[str(source)]=digest
    return entry,checked


class Backend:
    def __init__(self,entry,workers):
        self.lib=C.CDLL(str(Path(entry['library']).resolve()))
        self.lib.bench_create.argtypes=[C.c_int];self.lib.bench_create.restype=C.c_void_p
        self.lib.bench_destroy.argtypes=[C.c_void_p];self.lib.bench_destroy.restype=None
        self.lib.bench_process.argtypes=[C.c_void_p]*3+[C.c_int]*6;self.lib.bench_process.restype=C.c_int
        self.context=self.lib.bench_create(workers)
        if not self.context:raise RuntimeError('CPU context creation failed')
    def call(self,raw,out,method,phase):
        status=self.lib.bench_process(self.context,raw.ctypes.data,out.ctypes.data,
            raw.shape[1],raw.shape[0],raw.strides[0],out.strides[0],METHODS[method],1 if phase=='RGGB' else 2)
        if status:raise RuntimeError(f'{method}/{phase} returned {status}')
    def close(self):self.lib.bench_destroy(self.context)


def small_checks(backends):
    rng=np.random.default_rng(73891);checks=0
    for h,w in ((2,2),(3,5),(19,17),(129,131)):
        storage=np.full((h+2,w+11),0xA5,np.uint8);raw=storage[1:-1,3:3+w]
        raw[:]=rng.integers(0,256,raw.shape,dtype=np.uint8);snapshot=storage.copy()
        for phase in PATTERNS:
            for method in METHODS:
                outputs=[]
                for backend in backends.values():
                    guard=np.full((h+2,w*3+17),0xC7,np.uint8)
                    out=guard[1:-1,5:5+w*3].reshape(h,w,3)
                    backend.call(raw,out,method,phase);outputs.append(out.copy())
                    for y in range(2):
                        for x in range(2):
                            c=2-PATTERNS[phase][y][x] # Native output is BGR.
                            if not np.array_equal(out[y::2,x::2,c],raw[y::2,x::2]):raise RuntimeError('Measured sample changed')
                    if not np.array_equal(storage,snapshot):raise RuntimeError('Input changed')
                    guard[1:-1,5:5+w*3]=0xC7
                    if not np.all(guard==0xC7):raise RuntimeError('Output sentinel changed')
                if not np.array_equal(*outputs):raise RuntimeError(f'Odd/strided output mismatch: {method}/{phase}/{h}/{w}')
                checks+=1
    return checks


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True)
    p.add_argument('--before-root',type=Path,help='Optional checkout root for source hash verification')
    p.add_argument('--after-root',type=Path,help='Optional checkout root for source hash verification')
    p.add_argument('--manifest',type=Path,default=ROOT/'benchmark/datasets.json')
    p.add_argument('--data-root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',nargs='+',type=int,default=[1,8])
    p.add_argument('--warmup',type=int,default=10);p.add_argument('--rounds',type=int,default=100)
    p.add_argument('--batch',type=int,default=5);p.add_argument('--single-rounds',type=int,default=25)
    p.add_argument('--single-batch',type=int,default=1);p.add_argument('--seed',type=int,default=20261009)
    p.add_argument('--check-small',action='store_true',help='Untimed tiny/odd/strided and guard preflight')
    args=p.parse_args()
    if min(args.workers+[args.warmup,args.rounds,args.batch,args.single_rounds,args.single_batch])<1:p.error('Counts must be positive')
    entries={};checked={}
    for name in ('before','after'):
        entries[name],hashes=load_build(getattr(args,name),getattr(args,name+'_root'));checked.update(hashes)
    data=json.loads(args.manifest.read_text())
    case=next(c for c in data['images'] if c['dataset']=='div2k' and Path(c['path']).name=='0801.png')
    source=args.data_root/case['path']
    if sha(source)!=case['sha256']:raise RuntimeError('Dataset file hash mismatch')
    with Image.open(source) as im:rgb=np.array(im.convert('RGB'),dtype=np.uint8)
    if list(rgb.shape[:2])!=case['shape_hw']:raise RuntimeError('Dataset dimensions mismatch')
    decoded=hashlib.sha256(f'{rgb.shape[1]}x{rgb.shape[0]}:RGB:'.encode()+rgb.tobytes()).hexdigest()
    if decoded!=case['decoded_rgb_sha256']:raise RuntimeError('Decoded pixels mismatch')
    rgb=rgb[:1080,:1920].copy()
    if rgb.shape!=(1080,1920,3):raise RuntimeError('1080p crop unavailable')
    raws={}
    for phase,pattern in PATTERNS.items():
        raw=np.empty(rgb.shape[:2],np.uint8)
        for y in range(2):
            for x in range(2):raw[y::2,x::2]=rgb[y::2,x::2,pattern[y][x]]
        raws[phase]=raw
    allowed=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None
    if allowed and max(args.workers)>len(allowed):p.error('Worker count exceeds permitted CPUs; expand the affinity mask')
    frozen={'builds':entries,'checked_files':checked,'reference':case,'crop_yxhw':[0,0,1080,1920],
        'manifest_sha256':sha(args.manifest),'script_sha256':sha(__file__),'seed':args.seed,
        'workers':args.workers,'initial_affinity':allowed,'numpy':np.__version__,'pillow':pillow_version,
        'warmup':args.warmup,'rounds':args.rounds,'batch':args.batch,
        'single_rounds':args.single_rounds,'single_batch':args.single_batch}
    save(args.output.with_suffix('.freeze.json'),frozen)
    args.output.with_suffix('.runner.py').write_bytes(Path(__file__).read_bytes())
    rows=[];checks=0
    for workers in args.workers:
        if allowed:os.sched_setaffinity(0,allowed[:workers])
        backends={name:Backend(entry,workers) for name,entry in entries.items()}
        lanes=[(name,method,phase) for name in entries for method in METHODS for phase in PATTERNS]
        outputs=[np.empty_like(rgb) for _ in lanes];hashes=[];samples=[[] for _ in lanes]
        rounds=args.single_rounds if workers==1 else args.rounds;batch=args.single_batch if workers==1 else args.batch
        def call(i):
            name,method,phase=lanes[i];backends[name].call(raws[phase],outputs[i],method,phase)
        try:
            if args.check_small:checks+=small_checks(backends)
            reference={}
            for i,(name,method,phase) in enumerate(lanes):
                call(i);digest=hashlib.sha256(outputs[i].tobytes()).hexdigest();hashes.append(digest)
                if name=='before':reference[method,phase]=digest
                elif digest!=reference[method,phase]:raise RuntimeError(f'Candidate output mismatch: {method}/{phase}')
            for _ in range(args.warmup):
                for i in range(len(lanes)):call(i)
            rng=np.random.default_rng(args.seed)
            for _ in range(rounds):
                for i in rng.permutation(len(lanes)):
                    begin=time.perf_counter_ns()
                    for _ in range(batch):call(i)
                    samples[i].append((time.perf_counter_ns()-begin)/(1e6*batch))
            for i,(name,method,phase) in enumerate(lanes):
                if hashlib.sha256(outputs[i].tobytes()).hexdigest()!=hashes[i]:raise RuntimeError('Timed output changed')
                rows.append({'implementation':name,'method':method,'pattern':phase,'workers':workers,
                    'median_ms':float(np.median(samples[i])),'p10_ms':float(np.percentile(samples[i],10)),
                    'p90_ms':float(np.percentile(samples[i],90)),'milliseconds_per_call':samples[i],
                    'rounds':rounds,'calls_per_sample':batch,'warmup_rounds':args.warmup,
                    'affinity':sorted(os.sched_getaffinity(0)) if allowed else None,'output_sha256_bgr':hashes[i]})
            print(f'Finished {workers} workers',flush=True)
        finally:
            for backend in backends.values():backend.close()
            if allowed:os.sched_setaffinity(0,allowed)
    for path,digest in checked.items():
        if sha(path)!=digest:raise RuntimeError(f'Input changed during timing: {path}')
    summary=[]
    for workers in args.workers:
        for method in METHODS:
            values={name:float(np.mean([r['median_ms'] for r in rows if (r['workers'],r['method'],r['implementation'])==(workers,method,name)])) for name in entries}
            summary.append({'workers':workers,'method':method,'mean_of_phase_medians_ms':values,'speedup':values['before']/values['after']})
    save(args.output,{'complete':True,'frozen':frozen,'rows':rows,'summary':summary,
        'exact_before_after_outputs':True,'untimed_small_comparisons':checks,'host':platform.node(),
        'cpu':subprocess.run(['lscpu'],capture_output=True,text=True).stdout if shutil.which('lscpu') else platform.processor(),
        'scope':'Warm synchronous host-to-host public CPU API, retained scratch, randomized build/method/CFA calls, Python/ctypes included. Single worker is pinned to the first permitted core on Linux.'})
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
