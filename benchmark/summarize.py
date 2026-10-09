#!/usr/bin/env python3
"""Package final benchmark records and summarize quality and throughput.

Run benchmark/run.py with all four backends, insets 0 16, quality, timing and
examples first. This preserves source, binary, dataset and per-output hashes.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ('cpu', 'cuda', 'opencv', 'npp')
METHODS = {'cpu': {'bilinear', 'malvar', 'menon2007', 'softmenon'},
           'cuda': {'bilinear', 'malvar', 'menon2007', 'softmenon'},
           'opencv': {'opencv_bilinear', 'opencv_ea', 'opencv_vng'},
           'npp': {'npp_cfa'}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def normalize(value):
    if isinstance(value, dict):
        return {normalize(k): normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    if isinstance(value, str):
        return value.replace(str(ROOT) + '/', '')
    return value


def paired(rows, inset, dataset):
    lanes = defaultdict(dict)
    for row in rows:
        if row['inset'] == inset and (dataset == 'all' or row['dataset'] == dataset):
            if row['method'] in ('softmenon', 'menon2007'):
                lanes[row['path'], row['pattern']][row['method']] = row
    scenes = defaultdict(list)
    scene_datasets = {}
    for (path, _), methods in lanes.items():
        scenes[path].append(methods['softmenon']['psnr'] - methods['menon2007']['psnr'])
        scene_datasets[path] = methods['softmenon']['dataset']
    deltas = {path: statistics.mean(values) for path, values in scenes.items()}
    strata = defaultdict(list)
    for path, delta in deltas.items():
        strata[scene_datasets[path]].append(delta)
    rng = np.random.default_rng(20261010)
    boot = np.zeros(10000)
    for values in strata.values():
        a = np.array(values)
        for start in range(0, len(boot), 100):
            boot[start:start + 100] += a[rng.integers(0, len(a), (100, len(a)))].sum(axis=1)
    ordered = sorted(deltas.items(), key=lambda pair: pair[1])
    regions = {}
    for region, prefix in [('full', ''), ('interior8', 'interior8_'), ('border8', 'border8_')]:
        regions[region] = statistics.mean(
            m['softmenon'][prefix + 'psnr'] - m['menon2007'][prefix + 'psnr']
            for m in lanes.values())
    return {'inset': inset, 'dataset': dataset, 'images': len(deltas),
            'baseline': 'menon2007', 'method': 'softmenon',
            'mean_gain_db': statistics.mean(deltas.values()),
            'region_mean_gain_db': regions,
            'wins': sum(v > 1e-12 for v in deltas.values()),
            'ties': sum(abs(v) <= 1e-12 for v in deltas.values()),
            'losses': sum(v < -1e-12 for v in deltas.values()),
            'bootstrap95_mean_gain_db': np.percentile(boot / len(deltas), [2.5, 97.5]).tolist(),
            'worst5': [{'path': p, 'gain_db': v} for p, v in ordered[:5]],
            'best5': [{'path': p, 'gain_db': v} for p, v in ordered[-5:][::-1]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'build/benchmark')
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmark/results')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / 'benchmark/datasets.json'
    cases = json.loads(manifest.read_text())['images']
    qualities = {}
    timings = []
    artifacts = {}
    for name in BACKENDS:
        quality = json.loads((args.input / f'quality-{name}.json').read_text())
        if quality['manifest_sha256'] != sha(manifest) or quality['image_count'] != len(cases):
            raise RuntimeError('Wrong dataset manifest or incomplete quality run')
        keys = [(r['path'], r['inset'], r['pattern'], r['method']) for r in quality['rows']]
        methods = {r['method'] for r in quality['rows']}
        if methods != METHODS[name]:
            raise RuntimeError('Unexpected or missing benchmark methods')
        expected = {(c['path'], i, p, m) for c in cases for i in (0, 16)
                    for p in ('RGGB', 'BGGR') for m in methods}
        if len(keys) != len(expected) or set(keys) != expected:
            raise RuntimeError('Incomplete or duplicate quality rows')
        qualities[name] = quality
        packed = args.output / f'quality-{name}.json.gz'
        packed.write_bytes(gzip.compress(json.dumps(normalize(quality), separators=(',', ':'),
                                                    allow_nan=False).encode(), mtime=0))
        artifacts[packed.name] = sha(packed)
        timing = json.loads((args.input / f'timing-{name}.json').read_text())
        if timing['build'] != quality['build']:
            raise RuntimeError('Quality and timing used different builds')
        keys = [(r['method'], r['pattern']) for r in timing['rows']]
        if len(keys) != len(methods) * 2 or set(keys) != {
                (m, p) for m in methods for p in ('RGGB', 'BGGR')}:
            raise RuntimeError('Incomplete or duplicate timing rows')
        packed = args.output / f'timing-{name}.json'
        save(packed, normalize(timing))
        artifacts[packed.name] = sha(packed)
        for method in sorted({r['method'] for r in timing['rows']}):
            rows = [r for r in timing['rows'] if r['method'] == method]
            latency = statistics.mean(r['median_ms'] for r in rows)
            timings.append({'backend': name, 'method': method,
                            'mean_phase_median_ms': latency, 'fps': 1000 / latency})
    cpu = {(r['path'], r['inset'], r['pattern'], r['method']): r['output_sha256_rgb']
           for r in qualities['cpu']['rows']}
    cuda = {(r['path'], r['inset'], r['pattern'], r['method']): r['output_sha256_rgb']
            for r in qualities['cuda']['rows']}
    mismatches = {method: 0 for method in ('bilinear', 'malvar', 'menon2007', 'softmenon')}
    if cpu.keys() != cuda.keys():
        raise RuntimeError('CPU/CUDA coverage differs')
    for key, digest in cpu.items():
        mismatches[key[-1]] += digest != cuda[key]
    if mismatches['softmenon'] or mismatches['menon2007']:
        raise RuntimeError('SoftMenon/paper CPU/CUDA output mismatch')
    datasets = ['all', *sorted({c['dataset'] for c in cases})]
    summary = {'images': len(cases), 'manifest_sha256': sha(manifest),
               'quality': {name: q['summary'] for name, q in qualities.items()},
               'softmenon_vs_paper': [paired(qualities['cpu']['rows'], inset, dataset)
                                     for inset in (0, 16) for dataset in datasets],
               'bootstrap': {'resamples': 10000, 'seed': 20261010,
                             'unit': 'source image; phases kept together; stratified by dataset',
                             'scope': 'Pointwise uncertainty on this evaluation corpus; not a held-out generalization estimate'},
               'timing': timings, 'cpu_cuda_mismatches': mismatches,
               'artifact_sha256': artifacts, 'generator_sha256': sha(__file__)}
    examples = json.loads((args.input / 'examples.json').read_text())
    save(args.output / 'examples.json', normalize(examples))
    artifacts['examples.json'] = sha(args.output / 'examples.json')
    save(args.output / 'summary.json', summary)
    print(json.dumps({'cpu_cuda_mismatches': mismatches,
                      'softmenon_vs_paper': summary['softmenon_vs_paper'][0], 'timing': timings}, indent=2))


if __name__ == '__main__':
    main()
