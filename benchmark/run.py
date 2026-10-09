#!/usr/bin/env python3
"""Build production backends and reproduce quality, latency, and example images.

Requires NumPy/Pillow and the verified dataset manifest. Detailed artifacts go
under build/ by default. See ../benchmarks.md for interpretation and protocols.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import time

import numpy as np
from PIL import Image, features

ROOT = Path(__file__).resolve().parents[1]
METHODS = {"bilinear": 1, "malvar": 2, "menon2007": 3, "softmenon": 4}
BACKEND_METHODS = {"cpu": METHODS, "cuda": METHODS,
    "opencv": {"opencv_bilinear": 1, "opencv_ea": 2, "opencv_vng": 3},
    "npp": {"npp_cfa": 1}}
PATTERNS = {"RGGB": ((0, 1), (1, 2)), "BGGR": ((2, 1), (1, 0))}
SCRIPT_BYTES = Path(__file__).read_bytes()
SCRIPT_HASH = hashlib.sha256(SCRIPT_BYTES).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def build(args):
    dest = args.output / "libraries"
    dest.mkdir(parents=True, exist_ok=True)
    include = dest / "include/libdebayer"
    include.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "c/include/debayer.h", include / "debayer.h")
    metadata = json.loads((args.output / "build.json").read_text()) if (args.output / "build.json").exists() else {}
    for backend in args.backends:
        link_flags = []
        if backend == "cpu":
            sources = [ROOT / "benchmark/bridge_cpu.cpp", *[ROOT / "cpu" / n for n in
                       ("cpu_debayer.cpp", "cpu_kernel.cpp", "threadpool.cpp")]]
            dependencies = [*sorted((ROOT / "cpu").glob("*.hpp")), ROOT / "common/menon2007.hpp"]
            command = ["g++", "-std=c++17", "-O3", "-shared", "-fPIC", "-fvisibility=hidden", "-pthread"]
        elif backend == "cuda":
            sources = [ROOT / "benchmark/bridge_cuda.cpp", ROOT / "c/src/debayer.cu",
                       ROOT / "c/src/debayer_kernel.cu", ROOT / "cpp/src/debayer_cpp.cu"]
            dependencies = [ROOT / "c/include/debayer.h", ROOT / "c/src/debayer_kernel.h",
                            ROOT / "c/src/chroma_median.cuh", ROOT / "c/src/softmenon_green.cuh",
                            ROOT / "cpp/include/debayer_cpp.h",
                            ROOT / "common/menon2007.hpp"]
            command = [args.nvcc, "-std=c++17", "-O3", "-shared", "-Xcompiler=-fPIC,-fvisibility=hidden",
                       "-lineinfo", f"-arch={args.cuda_arch}", "-I" + str(dest / "include"),
                       "-I" + str(ROOT / "c/include"), "-I" + str(ROOT / "cpp/include")]
        elif backend == "opencv":
            sources = [ROOT / "benchmark/bridge_opencv.cpp"]
            dependencies = [ROOT / "benchmark/external_padding.hpp"]
            command = ["g++", "-std=c++17", "-O3", "-shared", "-fPIC", "-fvisibility=hidden", "-pthread"]
            command += shlex.split(subprocess.check_output(["pkg-config", "--cflags", "opencv4"], text=True))
            link_flags = shlex.split(subprocess.check_output(["pkg-config", "--libs", "opencv4"], text=True))
        else:
            sources = [ROOT / "benchmark/bridge_npp.cu"]
            dependencies = [ROOT / "benchmark/external_padding.hpp"]
            cuda_lib = str(Path(args.nvcc).resolve().parents[1] / "lib64")
            command = [args.nvcc, "-std=c++17", "-O3", "-shared", "-Xcompiler=-fPIC,-fvisibility=hidden",
                       "-lineinfo", f"-arch={args.cuda_arch}"]
            link_flags = ["-lnppicc", "-lnppc", "-L" + cuda_lib, "-Xlinker", "-rpath," + cuda_lib]
        library = dest / (backend + ".so")
        command += [*map(str, sources), "-o", str(library), *link_flags]
        hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources + dependencies}
        result = subprocess.run(command, capture_output=True, text=True)
        (dest / (backend + ".log")).write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"{backend} build failed; see {dest / (backend + '.log')}")
        if any(sha(ROOT / p) != h for p, h in hashes.items()):
            raise RuntimeError("Sources changed during compilation")
        metadata[backend] = {"library": str(library.resolve()), "library_sha256": sha(library),
            "backend": backend,
            "source_sha256": hashes, "command": command,
            "compiler": subprocess.check_output([command[0], "--version"], text=True)}
        if backend == "opencv":
            metadata[backend]["opencv_version"] = subprocess.check_output(["pkg-config", "--modversion", "opencv4"], text=True).strip()
        print(f"Built {backend}", flush=True)
    save(args.output / "build.json", metadata)


class Backend:
    def __init__(self, entry, workers):
        if sha(entry["library"]) != entry["library_sha256"]:
            raise RuntimeError("Backend binary changed")
        self.library = ctypes.CDLL(entry["library"])
        self.name = entry.get("backend", Path(entry["library"]).stem)
        self.methods = BACKEND_METHODS[self.name]
        self.notes = {}
        pointer = ctypes.c_void_p
        self.library.bench_create.argtypes = [ctypes.c_int]
        self.library.bench_create.restype = pointer
        self.library.bench_destroy.argtypes = [pointer]
        self.library.bench_destroy.restype = None
        for name in ("bench_process",):
            fn = getattr(self.library, name)
            fn.argtypes = [pointer, pointer, pointer] + [ctypes.c_int] * 6
            fn.restype = ctypes.c_int
        self.context = self.library.bench_create(workers)
        if not self.context:
            raise RuntimeError("Backend context creation failed")
        if self.name == "opencv":
            self.library.bench_vng_row_offset.argtypes = [pointer]
            self.library.bench_vng_row_offset.restype = ctypes.c_int
            offset = self.library.bench_vng_row_offset(self.context)
            self.notes["vng_row_offset_compensation"] = offset
            if offset < 0:
                self.close()
                raise RuntimeError("OpenCV VNG registration probe failed")

    def call(self, raw, output, method, pattern):
        number = self.methods[method]
        function = self.library.bench_process
        status = function(self.context, raw.ctypes.data, output.ctypes.data, raw.shape[1], raw.shape[0],
            raw.strides[0], output.strides[0], number,
            1 if pattern == "RGGB" else 2)
        if status:
            raise RuntimeError(f"{method}/{pattern} failed: {status}")

    def process(self, raw, method, pattern):
        output = np.empty((*raw.shape, 3), dtype=np.uint8)
        self.call(raw, output, method, pattern)
        return output[..., ::-1].copy()

    def close(self):
        if self.context:
            self.library.bench_destroy(self.context)
            self.context = None


def mosaic(rgb, pattern):
    result = np.empty(rgb.shape[:2], dtype=np.uint8)
    for y in range(2):
        for x in range(2):
            result[y::2, x::2] = rgb[y::2, x::2, PATTERNS[pattern][y][x]]
    return result


def psnr(sse, count):
    return "Infinity" if sse == 0 else 10 * math.log10(255**2 * count / int(sse))


def metrics(reference, output):
    difference = reference.astype(np.int32) - output
    error = difference * difference
    full = int(error.sum(dtype=np.int64))
    inside = error[8:-8, 8:-8]
    interior = int(inside.sum(dtype=np.int64))
    border_count = error.size - inside.size
    return {"sse": full, "count": error.size, "psnr": psnr(full, error.size),
            "interior8_sse": interior, "interior8_count": inside.size,
            "interior8_psnr": psnr(interior, inside.size),
            "border8_sse": full - interior, "border8_count": border_count,
            "border8_psnr": psnr(full - interior, border_count),
            "output_sha256_rgb": hashlib.sha256(output.tobytes()).hexdigest()}


def load_image(case):
    path = ROOT / case["path"]
    if sha(path) != case["sha256"]:
        raise RuntimeError(f"Changed dataset image: {path}")
    with Image.open(path) as im:
        rgb = np.array(im.convert("RGB"), dtype=np.uint8)
    if list(rgb.shape[:2]) != case["shape_hw"]:
        raise RuntimeError(f"Changed dimensions: {path}")
    prefix = f"{rgb.shape[1]}x{rgb.shape[0]}:RGB:".encode()
    if hashlib.sha256(prefix + rgb.tobytes()).hexdigest() != case["decoded_rgb_sha256"]:
        raise RuntimeError(f"Decoded pixels differ; check Pillow/JPEG versions: {path}")
    return rgb


def quality(args, cases, builds, manifest_hash):
    outputs = {}
    for backend_name in args.backends:
        backend = Backend(builds[backend_name], args.workers)
        rows = []
        try:
            for i, case in enumerate(cases):
                original = load_image(case)
                for inset in args.insets:
                    rgb = original if inset == 0 else original[inset:-inset, inset:-inset]
                    for pattern in PATTERNS:
                        raw = mosaic(rgb, pattern)
                        for method in backend.methods:
                            image = backend.process(raw, method, pattern)
                            for y in range(2):
                                for x in range(2):
                                    if not np.array_equal(image[y::2, x::2, PATTERNS[pattern][y][x]], raw[y::2, x::2]):
                                        raise RuntimeError("Measured CFA samples changed")
                            rows.append({"backend": backend_name, "method": method, "pattern": pattern,
                                "dataset": case["dataset"], "path": case["path"], "inset": inset,
                                "shape_hw": list(rgb.shape[:2]), **metrics(rgb, image)})
                if (i + 1) % 25 == 0 or i + 1 == len(cases):
                    print(f"Quality {backend_name}: {i + 1}/{len(cases)}", flush=True)
        finally:
            backend.close()
        summary = []
        for inset in args.insets:
            for dataset in ["all", *sorted({c["dataset"] for c in cases})]:
                for method in backend.methods:
                    selection = [r for r in rows if r["method"] == method and r["inset"] == inset and
                                 (dataset == "all" or r["dataset"] == dataset)]
                    values = [r["psnr"] for r in selection]
                    summary.append({"dataset": dataset, "method": method, "inset": inset,
                        "pairs": len(selection), "mean_psnr": "Infinity" if "Infinity" in values else float(np.mean(values)),
                        "pooled_psnr": psnr(sum(r["sse"] for r in selection), sum(r["count"] for r in selection))})
        outputs[backend_name] = {"rows": rows, "summary": summary, "image_count": len(cases),
            "manifest_sha256": manifest_hash, "build": builds[backend_name], "script_sha256": SCRIPT_HASH,
            "backend_notes": backend.notes,
            "numpy": np.__version__, "pillow": Image.__version__, "libjpeg": features.version("jpg"),
            "domain": "stored uint8 RGB, no scaling, inverse gamma, denoising, or color conversion beyond RGB decode"}
        save(args.output / ("quality-" + backend_name + ".json"), outputs[backend_name])
    return outputs


def timing(args, cases, builds):
    case = next(c for c in cases if c["shape_hw"][0] >= 1080 and c["shape_hw"][1] >= 1920)
    rgb = load_image(case)[:1080, :1920].copy()
    raws = {pattern: mosaic(rgb, pattern) for pattern in PATTERNS}
    reports = {}
    for name in args.backends:
        backend = Backend(builds[name], args.workers)
        lanes = [(method, pattern) for method in backend.methods for pattern in PATTERNS]
        outputs = [np.empty_like(rgb) for _ in lanes]
        expected = []
        samples = [[] for _ in lanes]
        try:
            for (method, pattern), output in zip(lanes, outputs):
                backend.call(raws[pattern], output, method, pattern)
                expected.append(hashlib.sha256(output.tobytes()).hexdigest())
            def call(i):
                method, pattern = lanes[i]
                backend.call(raws[pattern], outputs[i], method, pattern)
            for _ in range(args.warmup):
                for i in range(len(lanes)):
                    call(i)
            rng = np.random.default_rng(args.seed)
            for _ in range(args.rounds):
                for i in rng.permutation(len(lanes)):
                    start = time.perf_counter_ns()
                    for _ in range(args.batch):
                        call(i)
                    samples[i].append((time.perf_counter_ns() - start) / (1e6 * args.batch))
            rows = []
            for (method, pattern), output, hashed, values in zip(lanes, outputs, expected, samples):
                if hashlib.sha256(output.tobytes()).hexdigest() != hashed:
                    raise RuntimeError("Timed output changed")
                rows.append({"method": method, "pattern": pattern, "median_ms": float(np.median(values)),
                    "p10_ms": float(np.percentile(values, 10)), "p90_ms": float(np.percentile(values, 90)),
                    "milliseconds_per_call": values, "output_sha256_bgr": hashed})
        finally:
            backend.close()
        report = {"backend": name, "rows": rows, "shape_hw": [1080, 1920], "reference": case,
            "crop_yxhw": [0, 0, 1080, 1920], "warmup_rounds": args.warmup,
            "rounds": args.rounds, "calls_per_round": args.batch, "seed": args.seed,
            "cpu_workers": args.workers, "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "host": platform.node(), "cpu": subprocess.check_output(["lscpu"], text=True),
            "gpu": subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout if shutil.which("nvidia-smi") else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "build": builds[name],
            "script_sha256": SCRIPT_HASH, "all_outputs_verified": True, "backend_notes": backend.notes,
            "scope": "warm synchronous host-to-host; reusable buffers; CPU padding/dispatch included; GPU pageable transfers, padding, kernels and synchronization included; Python/ctypes overhead included"}
        save(args.output / ("timing-" + name + ".json"), report)
        reports[name] = report
        print(f"Timing {name} complete", flush=True)
    return reports


def examples(args, cases, builds):
    name = "cuda" if "cuda" in args.backends else "cpu"
    backend = Backend(builds[name], args.workers)
    records = []
    try:
        for filename, suffix in (("kodim11.png", "out"), ("kodim19.png", "lighthouse")):
            case = next(c for c in cases if c["dataset"] == "kodak24" and c["path"].endswith(filename))
            rgb = load_image(case)
            for method in ("bilinear", "malvar", "menon2007", "softmenon"):
                image = backend.process(mosaic(rgb, "BGGR"), method, "BGGR")
                path = args.output / "images" / f"{method}.{suffix}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(path)
                records.append({"method": method, "reference": case["path"], "backend": name,
                                "pattern": "BGGR", "path": str(path), **metrics(rgb, image)})
    finally:
        backend.close()
    save(args.output / "examples.json", records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/benchmark")
    parser.add_argument("--manifest", type=Path, default=ROOT / "benchmark/datasets.json")
    parser.add_argument("--backends", nargs="+", choices=tuple(BACKEND_METHODS), default=["cpu", "cuda"])
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--examples", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--insets", nargs="+", type=int, default=[0])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--cuda-arch", default="sm_120")
    parser.add_argument("--nvcc", default=os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc"))
    args = parser.parse_args()
    if args.workers < 1 or min(args.warmup, args.rounds, args.batch) < 1 or any(i < 0 or i % 2 for i in args.insets):
        parser.error("Use positive workers/rounds/batch and nonnegative even insets")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"run-{SCRIPT_HASH[:12]}.py").write_bytes(SCRIPT_BYTES)
    if args.build:
        build(args)
    if not (args.quality or args.timing or args.examples):
        return
    builds = json.loads((args.output / "build.json").read_text())
    for name in args.backends:
        for path, expected in builds[name]["source_sha256"].items():
            if sha(ROOT / path) != expected:
                raise RuntimeError(f"Changed source; rebuild before benchmarking: {path}")
    cases = json.loads(args.manifest.read_text())["images"]
    if args.quality:
        quality(args, cases[:args.limit] if args.limit else cases, builds, sha(args.manifest))
    if args.timing:
        timing(args, cases, builds)
    if args.examples:
        examples(args, cases, builds)


if __name__ == "__main__":
    main()
