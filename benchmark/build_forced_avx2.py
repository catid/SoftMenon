#!/usr/bin/env python3
"""Build an isolated CPU library with AVX512 runtime dispatch disabled.

Use this to compare AVX2 implementations on an AVX512-capable host. It does
not simulate an older CPU's microarchitecture. Production files are untouched;
only __builtin_cpu_supports("avx512...") predicates become false in copies.
"""
import argparse,hashlib,json,re,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,default=ROOT)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compiler',default='g++')
    args=p.parse_args();source=args.source_root.resolve();out=args.output.resolve()
    if out==source or source in out.parents and out.parts[-1] in ('cpu','common','benchmark'):
        p.error('Output must be a separate build directory')
    if (out/'build.json').exists():p.error('Output already contains a frozen build; choose a fresh directory')
    for sub in ('cpu','common','benchmark'):(out/sub).mkdir(parents=True,exist_ok=True)
    files=list((source/'cpu').glob('*.hpp'))+list((source/'cpu').glob('*.cpp'))+[
        source/'common/menon2007.hpp',source/'benchmark/bridge_cpu.cpp']
    original={};counts={}
    for path in files:
        relative=path.relative_to(source);original[str(relative)]=sha(path)
        text,count=re.subn(r'__builtin_cpu_supports\("avx512[^"\n]+"\)','false',path.read_text())
        (out/relative).write_text(text)
        if count:counts[str(relative)]=count
    if not counts:raise RuntimeError('No AVX512 dispatch predicates found')
    command=[args.compiler,'-std=c++17','-O3','-shared','-fPIC','-fvisibility=hidden','-pthread',
        *[str(out/file) for file in ('benchmark/bridge_cpu.cpp','cpu/cpu_debayer.cpp','cpu/cpu_kernel.cpp','cpu/threadpool.cpp')],
        '-o',str(out/'cpu.so')]
    subprocess.run(command,check=True)
    entry={'library':str(out/'cpu.so'),'library_sha256':sha(out/'cpu.so'),'command':command,
        'compiler':subprocess.check_output([args.compiler,'--version'],text=True),
        'source_sha256':{str(path.relative_to(out)):sha(path) for path in out.rglob('*') if path.suffix in ('.hpp','.cpp')},
        'original_source_root':str(source),'original_source_sha256':original,
        'forced_avx512_predicate_counts':counts,'generator_sha256':sha(Path(__file__)),
        'scope':'AVX2 dispatch on this physical host; not a performance model of older AVX2-only processors'}
    (out/'build.json').write_text(json.dumps({'cpu':entry},indent=2)+'\n')
    print(out/'build.json')
if __name__=='__main__':main()
