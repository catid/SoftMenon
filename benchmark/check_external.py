#!/usr/bin/env python3
"""Validate optional OpenCV/NPP bridge geometry, CFA registration and channels."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import numpy as np

PATTERNS={1:((2,1),(1,0)),2:((0,1),(1,2))} # BGR channel indices

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--opencv',type=Path)
    p.add_argument('--npp',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result={'passed':True,'checks':{},'failures':[],'libraries':{}}
    rng=np.random.default_rng(32179)
    for backend,path,algorithms in [('opencv',args.opencv,[1,2,3]),('npp',args.npp,[1])]:
        if path is None:continue
        path=path.resolve();lib=C.CDLL(str(path))
        lib.bench_create.argtypes=[C.c_int];lib.bench_create.restype=C.c_void_p
        lib.bench_destroy.argtypes=[C.c_void_p]
        lib.bench_process.argtypes=[C.c_void_p]*3+[C.c_int]*6;lib.bench_process.restype=C.c_int
        result['libraries'][backend]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        context=lib.bench_create(4)
        if not context:raise RuntimeError(f'{backend}: context creation failed')
        if backend=='opencv':
            lib.bench_vng_row_offset.argtypes=[C.c_void_p]
            result['libraries'][backend]['vng_row_offset_compensation']=lib.bench_vng_row_offset(context)
        count=0
        try:
            for h,w in [(2,2),(2,3),(3,2),(3,3),(7,9),(18,17),(31,35),(129,131)]:
                y,x=np.indices((h,w))
                colors=[np.full((h,w,3),value,dtype=np.uint8) for value in [(11,87,211),(255,0,0),(0,255,0),(0,0,255)]]
                colors.append(np.stack((20+x,70+y,180-x),axis=-1).astype(np.uint8))
                colors.append(rng.integers(0,256,(h,w,3),dtype=np.uint8))
                for pattern in (1,2):
                    for case,rgb in enumerate(colors):
                        guarded=np.full((h+2,w+11),177,dtype=np.uint8)
                        raw=guarded[1:-1,3:3+w]
                        for dy in range(2):
                            for dx in range(2):raw[dy::2,dx::2]=rgb[dy::2,dx::2,PATTERNS[pattern][dy][dx]]
                        before=guarded.copy()
                        for algorithm in algorithms:
                            storage=np.full((h+2,w*3+17),199,dtype=np.uint8)
                            out=storage[1:-1,5:5+w*3].reshape(h,w,3)
                            code=lib.bench_process(context,raw.ctypes.data,out.ctypes.data,w,h,raw.strides[0],out.strides[0],algorithm,pattern)
                            detail=[backend,algorithm,pattern,h,w,case]
                            failures=[]
                            if code:failures.append(['status',code])
                            else:
                                if case<4 and not np.array_equal(out,rgb):failures.append(['constant_color',int(np.max(np.abs(out.astype(int)-rgb)))])
                                if case==4 and min(h,w)>12 and not np.array_equal(out[6:-6,6:-6],rgb[6:-6,6:-6]):failures.append(['affine_color'])
                                for dy in range(2):
                                    for dx in range(2):
                                        if not np.array_equal(out[dy::2,dx::2,PATTERNS[pattern][dy][dx]],raw[dy::2,dx::2]):failures.append(['measured_sample',dy,dx])
                            if not np.array_equal(guarded,before):failures.append(['input_modified'])
                            guards=storage.copy();guards[1:-1,5:5+w*3]=199
                            if not np.all(guards==199):failures.append(['output_guard'])
                            if failures:result['failures'].append({'case':detail,'failures':failures})
                            count+=1
        finally:lib.bench_destroy(context)
        result['checks'][backend]=count
    result['passed']=not result['failures']
    result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='libraries'}))
    return int(not result['passed'])
if __name__=='__main__':raise SystemExit(main())
