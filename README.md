# SoftMenon

Fast 8-bit Bayer demosaicing on CPU and CUDA, with C/C++ and Rust interfaces.
Supports **RGGB and BGGR → BGR**, including odd image sizes and strided rows.
Forked from [libdebayer](https://github.com/catid/libdebayer); existing library
and API names are retained.

## Algorithms

- **Bilinear** — simple local interpolation.
- **Malvar 2004** — fixed cross-channel correction filters.
- **Menon 2007** — full paper DDFAPD, including refinement.
- **SoftMenon (initial version)** — the earlier custom reconstruction with soft
  green decisions and a sample-preserving 3×3 median of R−G and B−G.

SoftMenon improves its custom parent by **0.603 dB**. Full paper Menon has
**0.317 dB higher mean PSNR** than this initial version. A new ten-factor sweep
from paper Menon found **+0.458 dB** from soft initial green alone; conservative
close-score averaging gained **+0.296 dB** and improved 441/442 scenes. These
paper-based candidates remain experimental; [benchmarks.md](benchmarks.md)
reports every result, including regressions and reproduction details.

## Measured comparison

442 images, both Bayer phases, full-image all-channel mean PSNR. Library PSNR
uses CUDA outputs; CPU scores agree at the displayed precision. Latency is warm
**1920×1080 host-to-host**, mean of the two phase medians. CPU: eight physical
cores of a Threadripper PRO 9985WX. GPU: RTX PRO 6000 Blackwell Max-Q; transfers
included. These are workstation results, not Jetson measurements.

| Method | PSNR dB | CPU ms | CUDA ms |
|---|---:|---:|---:|
| Bilinear | 28.914 | 0.281 | 0.351 |
| Malvar 2004 | 33.963 | 0.641 | 0.350 |
| Menon 2007, full paper | **36.928** | **1.200** | 0.539 |
| SoftMenon, initial version | 36.612 | 0.524 | 0.366 |
| Legacy custom Menon control | 36.009 | 1.020 | 0.356 |
| Legacy + soft green only | 36.379 | 0.393 | 0.357 |
| OpenCV bilinear | 28.914 | 0.310 | — |
| OpenCV edge-aware | 28.927 | 0.328 | — |
| OpenCV VNG, registration corrected | 33.509 | 6.748 | — |
| NPP CFA reconstruction | 29.103 | — | 0.406 |

Paper Menon's CPU optimization preserves every output byte. A separate
interleaved before/after test measured **16.019 → 1.268 ms (12.63× faster)**
with eight workers and **120.957 → 8.334 ms (14.51×)** with one worker.
The table uses a fresh all-method CPU run; CUDA and external adapter timings
are retained from the preceding baseline run on the same workstation.

The initial SoftMenon CPU path is **18% faster than our Malvar implementation**
in the table's run. Dataset splits, hashes, decoder rules, baseline validation, timing
scope, external adapter details, all ablation results, and reproduction commands
are in [benchmarks.md](benchmarks.md). Historical blue-only PSNR figures have
been withdrawn; these scores include all three channels.

## Examples

Kodak `kodim11`, BGGR; individual full-image PSNR:

| Bilinear: 28.761 dB | Malvar: 34.366 dB |
|---|---|
| ![Bilinear](bilinear.out.png) | ![Malvar](malvar2004.out.png) |
| **Paper Menon: 39.102 dB** | **Initial SoftMenon: 38.190 dB** |
| ![Menon](menon2007.out.png) | ![SoftMenon](softmenon.out.png) |

Kodak `kodim19`, BGGR: [paper Menon, 39.918 dB](menon2007.lighthouse.png)
and [initial SoftMenon, 38.965 dB](softmenon.lighthouse.png).
Images courtesy of Kodak / [Rich Franzen's collection](https://r0k.us/graphics/kodak/).

## Build and use

CPU build and regression tests:

```bash
cmake -S cpu -B build/cpu -DCMAKE_BUILD_TYPE=Release -DDEBAYER_CPU_BUILD_DEMO=OFF
cmake --build build/cpu -j
ctest --test-dir build/cpu --output-on-failure
```

Use `SARONIC_DEBAYER_SOFTMENON` in the CPU or CUDA C++ wrapper. The CPU wrapper
supports `Debayer(8)` to select eight workers, including the calling thread.
See [CPU usage](cpu/README.md).

The [CUDA C API](c/include/debayer.h) accepts device buffers. Its workspace
entry points avoid per-frame allocation; the [C++ wrapper](cpp/include/debayer_cpp.h)
manages transfers and reusable scratch. The [Rust API](rust/src/lib.rs) exposes
`SoftMenonRggb2Bgr` and `SoftMenonBggr2Bgr`. Existing Menon entry points now implement
the complete paper baseline; they intentionally differ from the old approximation.

`nix develop` provides the original development environment. For dataset download
and standalone CPU/CUDA benchmark builds, follow [benchmarks.md](benchmarks.md).
