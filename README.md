# SoftMenon

Fast 8-bit Bayer demosaicing on CPU and CUDA, with C/C++ and Rust interfaces.
Supports **RGGB and BGGR → BGR**, including odd image sizes and strided rows.
Forked from [libdebayer](https://github.com/catid/libdebayer); existing library
and API names are retained.

## Algorithms

- **Bilinear** — simple local interpolation.
- **Malvar 2004** — fixed cross-channel correction filters.
- **Menon 2007** — full paper DDFAPD, including refinement.
- **SoftMenon** — soft green decisions followed by a 3×3 chroma-median
  refinement of missing green and red/blue, preserving every measured sample.

The revised SoftMenon scores **0.564 dB above full paper Menon** and
**0.881 dB above initial SoftMenon** on our 442-image benchmark. It improves
mean quality on all five datasets versus paper Menon, with wins on 377/442
scenes; some scenes still favor Menon. [benchmarks.md](benchmarks.md) reports
the selection process, regressions, raw results, and reproduction instructions.

## Measured comparison

442 images, both Bayer phases, full-image all-channel mean PSNR. Library PSNR
uses CUDA outputs; CPU scores agree at the displayed precision. Latency is warm
**1920×1080 host-to-host**, mean of the two phase medians. CPU: eight physical
cores of a Threadripper PRO 9985WX. GPU: RTX PRO 6000 Blackwell Max-Q; transfers
included. These are workstation results, not Jetson measurements.

| Method | PSNR dB | CPU ms | CUDA ms |
|---|---:|---:|---:|
| Bilinear | 28.914 | 0.278 | 0.353 |
| Malvar 2004 | 33.963 | 0.638 | 0.352 |
| Menon 2007, full paper | 36.928 | 1.204 | 0.541 |
| SoftMenon | **37.493** | **0.469** | 0.368 |
| Legacy custom Menon control | 36.009 | 1.014 | 0.358 |
| Legacy + soft green only | 36.379 | 0.322 | 0.359 |
| OpenCV bilinear | 28.914 | 0.310 | — |
| OpenCV edge-aware | 28.927 | 0.328 | — |
| OpenCV VNG, registration corrected | 33.509 | 6.748 | — |
| NPP CFA reconstruction | 29.103 | — | 0.406 |

In the matched initial-versus-current SoftMenon test, CPU latency improves
**0.539 → 0.497 ms** with eight workers and **3.251 → 2.774 ms** with one.
CUDA is unchanged at **0.367 ms**; forced AVX2 dispatch is also effectively flat.
The quality improvement reuses existing medians, while AVX2/AVX512 green
optimizations remove calculations at already measured green sites.

The table uses a separate fresh all-method CPU/CUDA run; external adapter
timings are retained from the earlier run on the same workstation. Paper
Menon's earlier exact CPU optimization reduced its latency by 12.63×.
Dataset splits, hashes, decoder rules, baseline validation, timing
scope, external adapter details, all ablation results, and reproduction commands
are in [benchmarks.md](benchmarks.md). Historical blue-only PSNR figures have
been withdrawn; these scores include all three channels.

## Examples

Look closely at the boat's ropes and rigging: the lower-quality reconstructions
show false-color fringes along these fine lines. Open the images at full size
to compare the artifacts.

Kodak `kodim11`, BGGR; individual full-image PSNR:

| Bilinear: 28.761 dB | Malvar: 34.366 dB |
|---|---|
| ![Bilinear](bilinear.out.png) | ![Malvar](malvar2004.out.png) |
| **Paper Menon: 39.102 dB** | **SoftMenon: 39.511 dB** |
| ![Menon](menon2007.out.png) | ![SoftMenon](softmenon.out.png) |

Kodak `kodim19`, BGGR: [paper Menon, 39.918 dB](menon2007.lighthouse.png)
and [SoftMenon, 40.254 dB](softmenon.lighthouse.png).
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
