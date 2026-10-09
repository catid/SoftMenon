#!/usr/bin/env python3
"""Compare initial/refined SoftMenon and paper Menon on the verified corpus.

Build the old checkout and current CPU/CUDA libraries with benchmark/run.py;
pass their build.json files explicitly. Baseline sources are verified against
the old checkout. Optional historical results/oracle files add output checks.
"""
import argparse,gzip,hashlib,importlib.util,json,os,platform,shutil,subprocess,time
from pathlib import Path
import numpy as np
from PIL import __version__ as pillow_version,features

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('helpers',ROOT/'benchmark/run.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pixels_sha(a):return hashlib.sha256(a.tobytes()).hexdigest()
def save(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def region_metrics(rgb,out):
    delta=out.astype(np.int16)-rgb;error=np.square(delta,dtype=np.int32)
    full=error.sum(axis=(0,1),dtype=np.int64);inside=error[8:-8,8:-8].sum(axis=(0,1),dtype=np.int64)
    n=rgb.shape[0]*rgb.shape[1];ni=(rgb.shape[0]-16)*(rgb.shape[1]-16)
    return {name:{'sse_rgb':list(map(int,sse)),'pixels':count,'psnr_db':helper.psnr(int(sse.sum()),count*3),
        'channel_psnr_db':[helper.psnr(int(v),count) for v in sse]}
        for name,sse,count in [('full',full,n),('interior8',inside,ni),('border8',full-inside,n-ni)]}
def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--baseline-source-root',type=Path,required=True)
    p.add_argument('--history',type=Path,help='Optional benchmark/run.py historical quality output')
    p.add_argument('--oracle',type=Path,help='Optional independently postprocessed original-corpus result')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--limit',type=int);a=p.parse_args()
    if a.workers<1 or (a.limit is not None and a.limit<1):p.error('workers and optional limit must be positive')
    candidate=json.loads(a.candidate.read_text());baseline=json.loads(a.baseline.read_text());checked={}
    for manifest in (a.candidate,a.baseline):checked[str(manifest)]=sha(manifest)
    for name,entry in candidate.items():
        for path,digest in entry['source_sha256'].items():
            target=(ROOT/path).resolve()
            if sha(target)!=digest:raise RuntimeError(f'Candidate source changed: {path}')
            checked[str(target)]=digest
        if sha(entry['library'])!=entry['library_sha256']:raise RuntimeError('Candidate binary changed')
        checked[entry['library']]=entry['library_sha256']
    if sha(baseline['cpu']['library'])!=baseline['cpu']['library_sha256']:raise RuntimeError('Baseline binary changed')
    checked[baseline['cpu']['library']]=baseline['cpu']['library_sha256']
    for path,digest in baseline['cpu']['source_sha256'].items():
        target=(a.baseline_source_root/path).resolve()
        if sha(target)!=digest:raise RuntimeError(f'Baseline source changed: {target}')
        checked[str(target)]=digest
    manifest_path=ROOT/'benchmark/datasets.json';manifest=json.loads(manifest_path.read_text());cases=manifest['images']
    for path in (Path(__file__),ROOT/'benchmark/run.py',manifest_path,a.history,a.oracle):
        if path:checked[str(path.resolve())]=sha(path)
    historical=None
    if a.history:
        history=json.loads(a.history.read_text())
        if history['manifest_sha256']!=sha(manifest_path):raise RuntimeError('Historical dataset manifest changed')
        historical={(r['path'],r['pattern'],r['method']):r['output_sha256_rgb'] for r in history['rows'] if r['inset']==0 and r['method'] in ('softmenon','menon2007')}
    oracle={(r['path'],r['phase']):r['output_sha256_rgb'] for r in json.loads(a.oracle.read_text())['rows'] if r['method']=='green_median_full'} if a.oracle else None
    if a.limit:cases=cases[:a.limit]
    frozen={'schema_version':1,'candidate_build':candidate,'baseline_build':baseline['cpu'],'checked_files':checked,
        'manifest_sha256':sha(manifest_path),'history_sha256':sha(a.history) if a.history else None,'script_sha256':sha(__file__),
        'independent_oracle_sha256':sha(a.oracle) if a.oracle else None,
        'helpers_sha256':sha(ROOT/'benchmark/run.py'),'images':len(cases),'selected_paths':[c['path'] for c in cases],
        'methods':['initial_softmenon','paper','refined_softmenon'],'insets':[0,16],'patterns':list(helper.PATTERNS),
        'regions':['full','interior8','border8'],'cpu_workers':a.workers,'cpu_affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        'host':platform.node(),'cpu':subprocess.check_output(['lscpu'],text=True) if shutil.which('lscpu') else platform.processor(),
        'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version','--format=csv,noheader'],text=True) if shutil.which('nvidia-smi') else None,
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'pillow':pillow_version,'libjpeg':features.version('jpg')},
        'data_domain':'Stored rendered uint8RGB, remosaiced at native resolution; no inverse gamma or resize. Inset16 crops each reference before remosaicing.',
        'scope':'Quality only. Every refined CPU output compared byte-for-byte with CUDA; no quality runtime is reported as inference latency.'}
    save(a.output.with_suffix('.freeze.json'),frozen)
    old=helper.Backend(baseline['cpu'],a.workers);cpu=helper.Backend(candidate['cpu'],a.workers);gpu=helper.Backend(candidate['cuda'],0)
    rows=0;exact=0;history_checks=0;oracle_checks=0;start=time.time();a.output.parent.mkdir(parents=True,exist_ok=True)
    meta={'complete':False,'frozen':frozen};save(a.output.with_suffix('.meta.json'),meta)
    try:
      with gzip.open(a.output,'wt') as stream:
        for i,case in enumerate(cases):
          original=helper.load_image(case)
          for inset in (0,16):
            rgb=original[inset:-inset,inset:-inset].copy() if inset else original
            for phase in helper.PATTERNS:
              raw=helper.mosaic(rgb,phase);raw_hash=pixels_sha(raw)
              outputs={'initial_softmenon':old.process(raw,'softmenon',phase),'paper':old.process(raw,'menon2007',phase),
                'refined_softmenon':cpu.process(raw,'softmenon',phase)}
              cuda=gpu.process(raw,'softmenon',phase)
              if not np.array_equal(cuda,outputs['refined_softmenon']):raise RuntimeError(f'CPU/GPU mismatch {case["path"]}/{inset}/{phase}')
              exact+=1
              for name,out in outputs.items():
                for y in range(2):
                  for x in range(2):
                    if not np.array_equal(out[y::2,x::2,helper.PATTERNS[phase][y][x]],raw[y::2,x::2]):raise RuntimeError(f'CFA changed {name}')
                digest=pixels_sha(out)
                if oracle and inset==0 and name=='refined_softmenon':
                    if digest!=oracle[case['path'],phase]:raise RuntimeError('Independent full-corpus oracle mismatch')
                    oracle_checks+=1
                if historical and inset==0 and name!='refined_softmenon':
                    historical_name='softmenon' if name=='initial_softmenon' else 'menon2007'
                    if digest!=historical[case['path'],phase,historical_name]:raise RuntimeError(f'Historical baseline mismatch {name}')
                    history_checks+=1
                row={'path':case['path'],'dataset':case['dataset'],'inset':inset,'pattern':phase,'method':name,
                    'shape_hw':list(raw.shape),'regions':region_metrics(rgb,out),'output_sha256_rgb':digest,
                    'measured_samples_changed':0}
                if name=='refined_softmenon':row['cuda_output_sha256_rgb']=pixels_sha(cuda)
                stream.write(json.dumps(row,allow_nan=False)+'\n');rows+=1
              if pixels_sha(raw)!=raw_hash:raise RuntimeError('Input modified')
          stream.flush()
          if (i+1)%20==0 or i+1==len(cases):print(f'{i+1}/{len(cases)} images, {rows} rows, {time.time()-start:.1f}s',flush=True)
      for path,digest in checked.items():
        if sha(path)!=digest:raise RuntimeError(f'Frozen input changed: {path}')
      if rows!=len(cases)*12 or exact!=len(cases)*4:raise RuntimeError('Incomplete corpus coverage')
      if historical and history_checks!=len(cases)*4:raise RuntimeError('Incomplete historical checks')
      if oracle and oracle_checks!=len(cases)*2:raise RuntimeError('Incomplete oracle checks')
      meta.update(complete=True,rows=rows,cpu_cuda_exact_comparisons=exact,historical_baseline_hash_matches=history_checks,
        independent_oracle_original_hash_matches=oracle_checks,output_sha256=sha(a.output),elapsed_seconds=time.time()-start)
      save(a.output.with_suffix('.meta.json'),meta)
    finally:old.close();cpu.close();gpu.close()
if __name__=='__main__':main()
