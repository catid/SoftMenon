#!/usr/bin/env python3
"""Paired old/new timing allowing intentional SoftMenon quality changes only."""
import argparse,ctypes as C,hashlib,importlib.util,json,os,platform,shutil,subprocess,time
from pathlib import Path
import numpy as np
from PIL import __version__ as pillow_version
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('soft_timing_helpers',ROOT/'benchmark/run.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
METHODS={'malvar':2,'menon2007':3,'softmenon':4}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2)+'\n')
def affinity():return sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None
def command_metadata(command):
    return subprocess.run(command,capture_output=True,text=True).stdout if shutil.which(command[0]) else None
class Backend:
    def __init__(self,path,workers):
        self.lib=C.CDLL(str(Path(path).resolve()));self.lib.bench_create.argtypes=[C.c_int]
        self.lib.bench_create.restype=C.c_void_p;self.lib.bench_destroy.argtypes=[C.c_void_p]
        self.lib.bench_destroy.restype=None
        self.lib.bench_process.argtypes=[C.c_void_p]*3+[C.c_int]*6;self.lib.bench_process.restype=C.c_int
        self.context=self.lib.bench_create(workers)
        if not self.context:raise RuntimeError('Context creation failed')
    def call(self,raw,out,method,phase):
        status=self.lib.bench_process(self.context,raw.ctypes.data,out.ctypes.data,raw.shape[1],raw.shape[0],
            raw.strides[0],out.strides[0],METHODS[method],1 if phase=='RGGB' else 2)
        if status:raise RuntimeError((method,phase,status))
    def close(self):self.lib.bench_destroy(self.context)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-build',type=Path,required=True)
    p.add_argument('--candidate-build',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--old-root',type=Path,help='Optional source checkout root for baseline source hash verification')
    p.add_argument('--candidate-root',type=Path,help='Optional source checkout root for candidate source hash verification')
    p.add_argument('--backend',choices=['cpu','cuda'],default='cpu');p.add_argument('--workers',nargs='+',type=int,default=[1,8])
    p.add_argument('--warmup',type=int,default=10);p.add_argument('--rounds',type=int,default=100)
    p.add_argument('--batch',type=int,default=5);p.add_argument('--seed',type=int,default=20261009)
    a=p.parse_args();allowed=affinity()
    if min(a.workers+[a.warmup,a.rounds,a.batch])<1:p.error('Workers and repeat counts must be positive')
    if max(a.workers)>256:p.error('Worker counts cannot exceed the benchmark ABI limit of 256')
    if a.backend=='cpu' and allowed and max(a.workers)>len(allowed):p.error('Worker count exceeds permitted CPU affinity')
    provenance={k:json.loads(path.read_text())[a.backend] for k,path in [('old',a.old_build),('candidate',a.candidate_build)]}
    checked={str(path.resolve()):sha(path) for path in (a.old_build,a.candidate_build,Path(__file__),ROOT/'benchmark/run.py')}
    for name,entry in provenance.items():
        if sha(entry['library'])!=entry['library_sha256']:raise RuntimeError('Library hash mismatch: '+entry['library'])
        checked[entry['library']]=entry['library_sha256']
        source_root=a.old_root if name=='old' else a.candidate_root
        if source_root:
            for filename,digest in entry['source_sha256'].items():
                source=(source_root/filename).resolve()
                if sha(source)!=digest:raise RuntimeError('Source hash mismatch: '+str(source))
                checked[str(source)]=digest
    cases=json.loads((ROOT/'benchmark/datasets.json').read_text())['images']
    case=next(c for c in cases if c['shape_hw'][0]>=1080 and c['shape_hw'][1]>=1920)
    rgb=helpers.load_image(case)[:1080,:1920].copy()
    if rgb.shape!=(1080,1920,3):raise RuntimeError('Expected exact 1080p crop unavailable')
    raws={phase:helpers.mosaic(rgb,phase) for phase in helpers.PATTERNS}
    report={'backend':a.backend,'libraries':provenance,'reference':case,'crop_yxhw':[0,0,1080,1920],
        'manifest_sha256':sha(ROOT/'benchmark/datasets.json'),'script_sha256':sha(__file__),
        'helpers_sha256':helpers.SCRIPT_HASH,'requested_workers':a.workers,'initial_cpu_affinity':allowed,
        'warmup_rounds':a.warmup,'rounds':a.rounds,'calls_per_round':a.batch,'seed':a.seed,'host':platform.node(),
        'numpy':np.__version__,'pillow':pillow_version,'checked_files':checked,
        'cpu':command_metadata(['lscpu']),'scope':'Warm synchronous host-to-host; reusable allocations; CPU padding/scheduling and CUDA pageable transfers/padding/kernels/synchronization included; Python/ctypes overhead included.'}
    if a.backend=='cuda':report['gpu']=command_metadata(['nvidia-smi','--query-gpu=index,name,driver_version','--format=csv,noheader'])
    save(a.output.with_suffix('.freeze.json'),report);rows=[]
    for workers in (a.workers if a.backend=='cpu' else [0]):
        if allowed:os.sched_setaffinity(0,allowed[:workers] if workers else allowed)
        backends={name:Backend(entry['library'],workers) for name,entry in provenance.items()}
        lanes=[(name,method,phase) for name in backends for method in METHODS for phase in helpers.PATTERNS]
        outputs=[np.empty_like(rgb) for _ in lanes];hashes=[];samples=[[] for _ in lanes];control={}
        def call(i):
            name,method,phase=lanes[i];backends[name].call(raws[phase],outputs[i],method,phase)
        try:
            for i,(name,method,phase) in enumerate(lanes):
                call(i);digest=hashlib.sha256(outputs[i].tobytes()).hexdigest();hashes.append(digest)
                if name=='old':control[method,phase]=digest
                elif method!='softmenon':assert digest==control[method,phase],f'Changed control {method}/{phase}'
                for y in range(2):
                    for x in range(2):
                        channel=2-helpers.PATTERNS[phase][y][x]
                        assert np.array_equal(outputs[i][y::2,x::2,channel],raws[phase][y::2,x::2])
            for _ in range(a.warmup):
                for i in range(len(lanes)):call(i)
            rng=np.random.default_rng(a.seed)
            for _ in range(a.rounds):
                for i in rng.permutation(len(lanes)):
                    start=time.perf_counter_ns()
                    for _ in range(a.batch):call(i)
                    samples[i].append((time.perf_counter_ns()-start)/(1e6*a.batch))
            for i,(name,method,phase) in enumerate(lanes):
                assert hashlib.sha256(outputs[i].tobytes()).hexdigest()==hashes[i]
                rows.append({'implementation':name,'method':method,'pattern':phase,'workers':workers,
                    'median_ms':float(np.median(samples[i])),'p10_ms':float(np.percentile(samples[i],10)),
                    'p90_ms':float(np.percentile(samples[i],90)),'milliseconds_per_call':samples[i],
                    'output_sha256_bgr':hashes[i],'cpu_affinity':affinity()})
                print(name,workers,method,phase,rows[-1]['median_ms'],flush=True)
        finally:
            for backend in backends.values():backend.close()
    comparisons=[]
    for workers in sorted(set(r['workers'] for r in rows)):
        for method in METHODS:
            means={name:float(np.mean([r['median_ms'] for r in rows if r['workers']==workers and r['method']==method and r['implementation']==name])) for name in provenance}
            comparisons.append({'workers':workers,'method':method,'mean_of_phase_medians_ms':means,'speedup':means['old']/means['candidate']})
    if allowed:os.sched_setaffinity(0,allowed)
    for filename,digest in checked.items():
        if sha(filename)!=digest:raise RuntimeError('Input changed during timing: '+filename)
    report.update(rows=rows,comparisons=comparisons,all_outputs_repeatable=True,controls_unchanged=True,measured_samples_preserved=True)
    save(a.output,report);print(json.dumps(comparisons,indent=2))
if __name__=='__main__':main()
