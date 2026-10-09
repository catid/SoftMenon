#!/usr/bin/env python3
"""Paired scene-level SoftMenon quality statistics with clustered bootstrap."""
import argparse,gzip,hashlib,json,math,statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def score(region):
    error=sum(region['sse_rgb'])
    return math.inf if not error else 10*math.log10(255**2*region['pixels']*3/error)
def delta_score(base,candidate):
    before,after=score(base),score(candidate)
    return 0.0 if before==after else after-before
def aggregate_region(rows,region):
    pixels=sum(row['regions'][region]['pixels'] for row in rows)
    sse=[sum(row['regions'][region]['sse_rgb'][c] for row in rows) for c in range(3)]
    return {'pixels':pixels,'sse_rgb':sse,'pooled_psnr_db':score({'pixels':pixels,'sse_rgb':sse}),
        'macro_psnr_db':statistics.mean(score(row['regions'][region]) for row in rows)}
def paired_summary(base,candidates,region,bootstrap,seed):
    by_image=defaultdict(list);datasets={}
    for b,c in zip(base,candidates):
        if (b['path'],b['cohort'],b['pattern'])!=(c['path'],c['cohort'],c['pattern']):raise ValueError('Unpaired candidate rows')
        by_image[b['path']].append(delta_score(b['regions'][region],c['regions'][region]));datasets[b['path']]=b['dataset']
    image_delta={key:statistics.mean(value) for key,value in by_image.items()};values=list(image_delta.values());ci=None
    if bootstrap and all(math.isfinite(value) for value in values):
        strata=defaultdict(list)
        for key,value in image_delta.items():strata[datasets[key]].append(value)
        rng=np.random.default_rng(seed);means=np.zeros(bootstrap,dtype=np.float64)
        for stratum in strata.values():
            a=np.array(stratum,dtype=np.float64)
            for begin in range(0,bootstrap,128):
                count=min(128,bootstrap-begin)
                means[begin:begin+count]+=a[rng.integers(0,len(a),(count,len(a)))].sum(axis=1)
        ci=[float(x) for x in np.percentile(means/len(values),[2.5,97.5])]
    ranked=sorted(image_delta.items(),key=lambda item:item[1]);before=aggregate_region(base,region);after=aggregate_region(candidates,region)
    return {'images':len(values),'CFA_pairs':len(candidates),'baseline':before,'candidate':after,
        'mean_paired_delta_db':statistics.mean(values),'median_image_delta_db':statistics.median(values),
        'pooled_psnr_delta_db':after['pooled_psnr_db']-before['pooled_psnr_db'],
        'bootstrap95_mean_delta_db':ci,'wins':sum(v>1e-12 for v in values),
        'ties':sum(abs(v)<=1e-12 for v in values),'losses':sum(v< -1e-12 for v in values),
        'identical_outputs':sum(b['output_sha256_rgb']==c['output_sha256_rgb'] for b,c in zip(base,candidates)),
        'worst5':[{'path':path,'delta_db':value} for path,value in ranked[:5]],
        'best5':[{'path':path,'delta_db':value} for path,value in ranked[-5:][::-1]]}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('rows',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=10000);p.add_argument('--seed',type=int,default=20261010);a=p.parse_args()
    if a.bootstrap<0:p.error('bootstrap must be nonnegative')
    meta_path=a.rows.with_suffix('.meta.json');meta=json.loads(meta_path.read_text())
    if not meta['complete'] or sha(a.rows)!=meta['output_sha256']:raise RuntimeError('Incomplete or altered result')
    rows=[json.loads(line) for line in gzip.open(a.rows,'rt')];frozen=meta['frozen']
    if len(rows)!=meta['rows'] or len(rows)!=12*frozen['images']:raise RuntimeError('Wrong row count')
    keys={(r['path'],r['inset'],r['pattern'],r['method']) for r in rows}
    if len(keys)!=len(rows):raise RuntimeError('Duplicate result row')
    expected={(path,inset,phase,method) for path in frozen['selected_paths'] for inset in frozen['insets']
        for phase in frozen['patterns'] for method in frozen['methods']}
    if keys!=expected:raise RuntimeError('Incomplete or unexpected result coverage')
    for r in rows:
        r['cohort']='original' if r['inset']==0 else 'inset16'
        if r['measured_samples_changed']!=0:raise RuntimeError('Measured CFA samples changed')
        if r['method']=='refined_softmenon' and r['output_sha256_rgb']!=r['cuda_output_sha256_rgb']:raise RuntimeError('CPU/CUDA hashes differ')
    datasets=sorted({r['dataset'] for r in rows});summaries=[];comparisons=[];scene_deltas=[]
    for inset in (0,16):
      for dataset in ['all',*datasets]:
        subset=[r for r in rows if r['inset']==inset and (dataset=='all' or r['dataset']==dataset)]
        for region in ('full','interior8','border8'):
          for method in ('initial_softmenon','paper','refined_softmenon'):
            selected=[r for r in subset if r['method']==method]
            summary=aggregate_region(selected,region)
            summary['channel_pooled_psnr_db']=[10*math.log10(255**2*summary['pixels']/sse) if sse else 'Infinity' for sse in summary['sse_rgb']]
            summaries.append({'inset':inset,'dataset':dataset,'region':region,'method':method,**summary})
          cand=sorted([r for r in subset if r['method']=='refined_softmenon'],key=lambda r:(r['path'],r['pattern']))
          for baseline in ('initial_softmenon','paper'):
            before=sorted([r for r in subset if r['method']==baseline],key=lambda r:(r['path'],r['pattern']))
            comparison=paired_summary(before,cand,region,a.bootstrap,a.seed)
            comparisons.append({'inset':inset,'dataset':dataset,'region':region,'baseline_method':baseline,
              'candidate_method':'refined_softmenon',**comparison})
            if dataset=='all' and region=='full':
                grouped={}
                for b,c in zip(before,cand):
                    grouped.setdefault(b['path'],[]).append(delta_score(b['regions'][region],c['regions'][region]))
                scene_deltas.extend({'path':path,'inset':inset,'baseline_method':baseline,'delta_db':float(np.mean(deltas))} for path,deltas in grouped.items())
    result={'complete':True,'images':frozen['images'],'rows':len(rows),'cpu_cuda_exact_comparisons':meta['cpu_cuda_exact_comparisons'],
        'historical_baseline_hash_matches':meta['historical_baseline_hash_matches'],
        'independent_oracle_original_hash_matches':meta['independent_oracle_original_hash_matches'],
        'bootstrap':{'replicates':a.bootstrap,'seed':a.seed,'unit':'Image; both CFA phases clustered, fixed dataset sizes stratified; cohorts separate'},
        'scope':frozen['data_domain'],'summary':summaries,'comparisons':comparisons,'scene_deltas':scene_deltas,
        'provenance':{'rows_path':str(a.rows),'rows_sha256':sha(a.rows),'metadata_path':str(meta_path),'metadata_sha256':sha(meta_path),
            'script_sha256':sha(__file__)}}
    save(a.output,result)
    for r in comparisons:
        if r['dataset']=='all' and r['region']=='full':print(json.dumps({k:r[k] for k in ('inset','baseline_method','mean_paired_delta_db','bootstrap95_mean_delta_db','wins','ties','losses','worst5')}))
if __name__=='__main__':main()
