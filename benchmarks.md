# Benchmark methods and reproduction

This document describes how to reconstruct the 442-image dataset and reproduce
the quality, timing, and correctness results for SoftMenon and its baselines.
The library provides Bilinear, Malvar 2004, full paper Menon 2007, and SoftMenon.

## Reconstruct the exact dataset

Run from the repository root. Python 3.12 was used. The downloader uses only the
Python standard library; scoring additionally requires NumPy and Pillow.

```bash
python3 -m venv build/benchmark-venv
build/benchmark-venv/bin/pip install numpy==1.26.4 Pillow==10.2.0
build/benchmark-venv/bin/python benchmark/datasets.py
build/benchmark-venv/bin/python benchmark/datasets.py --verify \
  --decoded-report build/dataset-decoding.json
```

[benchmark/datasets.json](benchmark/datasets.json) is the authoritative manifest:
442 filenames, source URLs, archive members, sizes, SHA-256 hashes, dimensions,
and decoded RGB hashes. It is sufficient to reconstruct the selection without any
local research files. The downloader fetches approximately **582 MB**
(581,942,644 bytes of archives/direct images), retains downloaded archives, and
extracts only the selected images. Allow about 1.2 GB for files and archives,
plus space for benchmark outputs. Datasets are ignored by Git.

| Dataset | Exact selection | Source and reconstruction |
|---|---|---|
| Kodak24 | All 24 `kodim01.png` through `kodim24.png` | [Rich Franzen's Kodak page](https://r0k.us/graphics/kodak/); original PNG files under `kodak/` |
| McMaster | All 18 `1.tif` through `18.tif` | [Author page](https://www4.comp.polyu.edu.hk/~cslzhang/CDM_Dataset.htm); `McM.zip`, members `McM/1.tif` etc.; published password `McM_CDM` |
| Urban100 | 100 `img_NNN_SRF_2_HR.png` files | [Author repository](https://github.com/jbhuang0604/SelfExSR/tree/8f6dd8c1d20cb7e8792a7177b4f6fd677633f598/data/Urban100/image_SRF_2), commit `8f6dd8c1d20cb7e8792a7177b4f6fd677633f598`; use the author's **SRF2 HR crops**, not SRF3/SRF4 HR or LR files |
| DIV2K | **Validation HR**, all 100 images `0801.png` through `0900.png` | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/); `DIV2K_valid_HR.zip`, members `DIV2K_valid_HR/0801.png` etc. |
| BSDS500 | **Test split only**, 200 original JPEGs | [Berkeley](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html); `BSR_bsds500.tgz`, members `BSR/BSDS500/data/images/test/*.jpg` |

Images are placed at `datasets/reference/images/<dataset>/`; BSDS retains a
`test/` subdirectory and DIV2K a `validation/` subdirectory. No file is resized
or re-encoded. In particular, converting
BSDS JPEGs to some other downloaded PNG distribution can change the experiment.
The image-set proportions are 24:18:100:100:200; the overall metric is not an
equal-weight average of five dataset means.

Archive hashes:

```text
McM.zip           6c3eaba44ab5801f155cc470d12125d5fe18f6f2fbaad363353e5e92667023c1
DIV2K_valid_HR.zip 20dd31fd84d777bc1cf5d6b7654a3f569c0aec74458ae094122ad1d0489900fc
BSR_bsds500.tgz    97e49d31764f3912f0c4122707d53062ac9e783ba0f095e447a4d53c1a41af8e
```

The downloader reuses complete files after verifying them; an interrupted payload
restarts from the beginning. It refuses to replace a file whose bytes differ,
verifies downloads before installing them, and never executes downloaded code.
`--verify` is offline. To reconstruct elsewhere, supply
`--root /absolute/destination`; the same relative `datasets/reference/` layout is
created there. The scorer expects the default repository layout.

Both file hashes and pixel hashes must match. Canonical decoded hash:

```python
sha256(f"{width}x{height}:RGB:".encode() + rgb_uint8_row_major_bytes)
```

`datasets.py --decoded-report` records Pillow/JPEG versions and fails if pixels
differ. `run.py` also enforces these hashes before scoring. Our production scoring
used NumPy 1.26.4, Pillow 10.2.0, and libjpeg API version 8.0. Decoded hashes
also match Pillow 12.3.0 with libjpeg API version 6.2.
Versions are provenance; matching decoded bytes is the actual acceptance check.
If a URL changes, obtain the exact bytes from a mirror and verify against the
manifest instead of silently substituting another split or crop.

Dataset images retain their owners' terms and attribution. The manifest links
source pages and records use notes. Downloading data does not put it under this
repository's code license. The evaluation uses rendered images; native linear
RAW and temporal quality require separate measurements.

## Input and metric contract

1. Decode each stored image to uint8 RGB with Pillow `convert("RGB")`. Do not
   apply inverse gamma, white balance, a color matrix, noise, resizing, or an ISP.
2. Construct both RGGB and BGGR mosaics by selecting the measured RGB channel
   at each pixel. Top-left RGGB is red; top-left BGGR is blue. Green remains at
   `(even, odd)` and `(odd, even)`. There are 884 image/phase observations per
   method in the original cohort, drawn from 442 source images.
3. Call the uint8 reconstruction, convert its BGR channel order to RGB, and
   verify every measured CFA sample exactly. Output hashes are kept; source-file
   and decoded-RGB hashes are verified against the dataset manifest.
4. For each image/phase, compute integer squared error over **all three channels**:

   `SSE = sum((int32(reference) - int32(output))**2, dtype=int64)`

   `PSNR = 10 * log10(255**2 * (3 * H * W) / SSE)`

   Exact equality has infinite PSNR, serialized as `"Infinity"`, without a cap.
5. Primary quality is the arithmetic mean of per-image/per-phase PSNR. Pooled
   PSNR instead sums SSE and sample counts before taking the logarithm. They
   answer different questions and can rank methods differently.

**Original** uses the complete input image, including its existing boundary.
**Inset16** first crops 16 pixels from each input edge, then remosaics and
reconstructs that smaller image independently. This even crop preserves CFA
phase and exposes a different scene boundary. The algorithm cannot read the
discarded surrounding pixels. **Interior8** is a scoring region: exclude eight
output pixels on every edge of whichever cohort was reconstructed. It does not
change the reconstruction input. Inset16 and original reuse the same scenes and
are not independent datasets.

A border band contains each pixel once; its sample count is three times the
number of pixels in the band. For width 8, full SSE equals interior8
SSE plus border8 SSE. Regional perfect matches retain infinite PSNR; finite-only
PSNR deltas are labeled as such, while MSE and win/loss counts retain every case.

## Baselines and algorithm identity

- **Bilinear:** local independent-channel interpolation.
- **Malvar 2004:** the Malvar–He–Cutler cross-channel correction filters.
- **Menon 2007:** full DDFAPD, including its three refinement stages, following
  [Menon, Andriani and Calvagno, TIP 2007](https://doi.org/10.1109/TIP.2006.884928)
  and the [pinned Colour implementation](https://github.com/colour-science/colour-demosaicing/blob/f4f67d46c8a803164e9bc4b36d828931e6377e7c/colour_demosaicing/bayer/demosaicing/menon2007.py).
- **SoftMenon:** Hamilton–Adams directional green estimates with three-quarter
  correction strength, neighborhood consistency scores from colocated color
  differences, and squared soft weights. Each estimate rounds its adjacent-green
  average and correction separately, including estimates used for neighboring
  color differences. Initial red/blue reconstruction interpolates differences
  from green. Diagonal pairs use equal weights for a score gap up to 16 DN,
  three-to-one weights up to 64 DN, and the more consistent pair above 64 DN;
  pair estimates are rounded before blending. A 3×3
  chroma-median refinement then reconstructs missing green from the measured
  red/blue sample and reconstructs the other missing color. Every measured CFA
  sample is preserved. [The complete equations and diagrams](README.md#how-softmenon-works)
  define the three stages.

The full paper reference is defined in
[common/menon2007.hpp](common/menon2007.hpp), which the CUDA backend executes
directly. The optimized CPU implementation in
[cpu/menon2007_cpu.hpp](cpu/menon2007_cpu.hpp) is checked against that unchanged
reference. Scale-144 signed integer arithmetic
represents its rational calculations exactly; output is clipped and rounded to
nearest uint8, with half ties upward, only at the end. Interpolation/refinement
use reflect101, while the posterior classifier convolution uses a **zero gradient
halo**, matching Colour. Describing every stage as reflected would be incorrect.
The CUDA reference uses 33 bytes/pixel of paper-specific scratch, excluding
input/output. CPU scratch is retained per worker and bounded by its tile size.

The baseline matched an independent rational NumPy oracle on 182 fixtures and
364 comparisons with pinned upstream floating calculations (maximum unrounded
difference 1.14e-13 DN). Forty-four final uint8 discrepancies with upstream float
rounding occurred only at mathematically exact half ties. CPU/CUDA exact pairing
was checked for Bilinear, Malvar, and paper Menon. SoftMenon matched across
CPU/CUDA on all 1,768 image/phase/cohort observations.

Optional OpenCV/NPP comparisons use
[dedicated adapters](benchmark/bridge_opencv.cpp). Both get a four-pixel reflect101
RAW halo, one extra reflected bottom/right pixel when needed for even dimensions,
and a crop back to the original extent. OpenCV uses explicit four-letter CFA
enums. The VNG adapter probes spatial registration at startup: OpenCV 4.6 needs
a two-row crop correction; reconstructed samples are not modified. The correction
is recorded in JSON. Unknown registration fails the run. NPP uses
`nppiCFAToRGB_8u_C1C3R_Ctx` with required `NPPI_INTER_UNDEFINED`, then RGB-to-BGR;
it is not a cubic-interpolation mode. These adapter costs are included in timing.

## Reproduce production quality, timing, and images

A C++17 compiler with pthreads is sufficient to build the native CPU backend;
the Python runner also needs the NumPy/Pillow environment above. CUDA requires `nvcc` and a
compatible GPU; optional OpenCV requires `pkg-config opencv4`, and NPP requires
the CUDA NPP libraries. Commands below use CUDA architecture `sm_120`, matching
our workstation; choose your device's architecture when reproducing elsewhere.

```bash
build/benchmark-venv/bin/python benchmark/run.py --build \
  --backends cpu cuda opencv npp --output build/benchmark --cuda-arch sm_120

CUDA_VISIBLE_DEVICES=0 build/benchmark-venv/bin/python benchmark/check_external.py \
  --opencv build/benchmark/libraries/opencv.so \
  --npp build/benchmark/libraries/npp.so --output build/benchmark/external-checks.json

CUDA_VISIBLE_DEVICES=0 taskset -c 16-23 \
  build/benchmark-venv/bin/python benchmark/run.py \
  --backends cpu cuda opencv npp --output build/benchmark \
  --quality --insets 0 16 --examples --workers 8

CUDA_VISIBLE_DEVICES=0 taskset -c 16-23 \
  build/benchmark-venv/bin/python benchmark/run.py \
  --backends cpu cuda opencv npp --output build/benchmark \
  --timing --workers 8 --warmup 10 --rounds 100 --batch 5 --seed 20261009

build/benchmark-venv/bin/python benchmark/summarize.py \
  --input build/benchmark --output benchmark/results
```

For a CPU-only machine use `--backends cpu`, without CUDA/OpenCV dependencies.
`--limit 2` is a quality smoke test, not a reportable result. For one worker,
repeat timing with `--workers 1` and one physical CPU in the affinity mask;
save or rename the timing JSON before rerunning because the filename is reused.
Choose available physical cores on your host; `16-23` is our specific machine's
eight-core mask. The worker count includes the caller.

The runner produces `build.json`, per-image `quality-<backend>.json`, raw latency
samples in `timing-<backend>.json`, and `examples.json` with images. It snapshots
its own source and records dataset manifest, build source, binary and output
hashes. It fails if build sources or binary hashes have changed. Per-image scores
can be reaggregated without running kernels again. Native backends evaluate the
four supported algorithms; OpenCV and NPP supply the external comparisons.

Quality images retain their original dimensions. Timing uses the unscaled
top-left **1920-by-1080 crop of DIV2K 0801.png**, remosaiced in both phases.
The measured call is warm, synchronous **host-to-host latency**: input/output
host arrays and native scratch are reused; CPU padding and dispatch are included;
GPU pageable transfers, padding, kernels, synchronization, and Python/ctypes call
overhead are included. Decode, remosaicing, quality calculation, and allocation
warmup are excluded. It is not a kernel-only or pipelined video throughput number.
Ten warmup rounds precede 100 randomized-order measurement rounds; each sample
averages five synchronous calls. Report phase medians separately or explicitly
label their arithmetic mean. Raw samples and p10/p90 accompany medians. Timed
outputs are hashed and checked for consistency.

Workstation: AMD Ryzen Threadripper PRO 9985WX, RTX PRO 6000 Blackwell Max-Q,
driver 615.71.09, GCC 13.3.0, CUDA 13.4.59, OpenCV 4.6.0, NPP 13.2.0.35,
Linux x86-64. CPU kernels runtime
dispatch to AVX512BW/VBMI, AVX2, or scalar paths. These are workstation
measurements; Jetson and ARM CPU speed remain unmeasured. Timing JSON records
hardware, compiler/build commands, affinity, visible GPU, seed, and round counts.
Quality JSON additionally records NumPy, Pillow, and JPEG versions; the complete
runner commands are given above.
Paper Menon uses a tiled CPU implementation with cached directional gradients
and runtime SIMD dispatch. Its implementation and correctness checks
are described below. SoftMenon has separate SIMD kernels.

Examples use Kodak `kodim11.png` (boat and harbor) and `kodim19.png`
(lighthouse), BGGR. Captions must use those individual full-image PSNR values,
not the Kodak mean or interior-only scores.

## SoftMenon refinement

Let `I` be the immutable, complete image after initial green and red/blue
interpolation. The final pass uses these equations:

```text
mR = median_3x3(I.R - I.G)
mB = median_3x3(I.B - I.G)

At a measured R: Gnew = clip(Rmeasured - mR)
At a measured B: Gnew = clip(Bmeasured - mB)
At a measured G: Gnew = Gmeasured

Rnew = Rmeasured if R was measured, otherwise clip(Gnew + mR)
Bnew = Bmeasured if B was measured, otherwise clip(Gnew + mB)
```

Both medians read the unchanged image with reflect101 boundaries. Clip green
into `[0,255]` **before** reconstructing the other color. Interpolated green
can change; every measured CFA value remains exact. Green and red/blue
refinement share the same two medians and output pass. See the
[algorithm diagrams](README.md#how-softmenon-works) for the complete pipeline.

## Quality results

Full-image mean PSNR, averaging both CFA phases per source image:

| Dataset / input cohort | Full paper Menon | SoftMenon | SoftMenon − paper |
|---|---:|---:|---:|
| Kodak24 | 39.2055 | **39.9900** | +0.7844 |
| McMaster | 34.2268 | **34.7665** | +0.5398 |
| Urban100 | 33.6647 | **34.5204** | +0.8558 |
| DIV2K validation | 38.3940 | **39.1488** | +0.7548 |
| BSDS500 test | 37.7975 | **39.0978** | +1.3004 |
| All 442, original | 36.9285 | **37.9458** | +1.0174 |
| All 442, Inset16 | 36.7859 | **37.8218** | +1.0359 |

SoftMenon wins on **429 of 442 scenes** and loses on 13 against full paper
Menon, averaging the two phases per scene. Its largest loss is Urban100
`img_055_SRF_2_HR.png`, **−1.4951 dB**. Pooled PSNR improves **0.7268 dB**;
Interior8 and Border8 mean PSNR improve **1.0317 dB** and **0.8720 dB**.

The pointwise 95% bootstrap interval for the mean gain is **[0.9715, 1.0632] dB**,
using 10,000 dataset-stratified source-image resamples with both phases kept
together (seed 20261010). This interval describes this evaluation corpus and
does not account for parameter selection.

These are averages, not a guarantee for every scene. Mean per-image PSNR and
pooled PSNR use different weighting, as defined above; inspect per-image records
when evaluating fine textures or color boundaries. This corpus was used to
choose algorithm parameters, so its results are not an independent estimate of
generalization. Results cover rendered uint8 RGB remosaicing; linear sensor RAW
and temporal behavior require separate evaluation.

The [README comparison](README.md#measured-comparison) includes the remaining
library algorithms and external baselines under the same metric contract.

## Timing results

The README table reports the mean of the RGGB and BGGR median latencies from
[CPU](benchmark/results/timing-cpu.json),
[CUDA](benchmark/results/timing-cuda.json),
[OpenCV](benchmark/results/timing-opencv.json), and
[NPP](benchmark/results/timing-npp.json). All calls include the work described
in the timing protocol above.

| SoftMenon path | Latency ms | FPS |
|---|---:|---:|
| CPU, eight workers, native AVX512 dispatch | 0.493 | 2,027 |
| CUDA, synchronous host-to-host | 0.374 | 2,674 |

FPS is `1000 / mean of phase-median milliseconds`; it describes repeated
synchronous calls under this protocol.

CPU results must identify the worker count and instruction-set dispatch;
CUDA host-to-host results include transfers and synchronization. ARM CPU and
Jetson runtime performance remain unmeasured. Report repeated measurements when
comparing small timing differences.

## Implementation and correctness

SoftMenon's AVX512 green kernel computes 32 chromatic sites per 64-pixel strip.
It caches signed 16-bit horizontal and vertical color differences, then reuses
them when forming the neighborhood scores. AVX2 and scalar paths use the same
estimates, scores, squared weights, rounding, and clipping. The CPU processes
64-row output strips through green interpolation, red/blue reconstruction, and
median refinement. Overlapping halos provide neighboring samples without
sharing writable intermediate buffers between workers. Thread-local scratch
is reused; allocation failures either take an exact fallback or return an error
after draining queued work. Eight-pixel RAW padding covers the green stencil and
its reconstruction halo.

The SIMD CPU blend uses double-precision arithmetic for the weighted integer sum.
Its bounded uint8 inputs make those products and sums exact, and the quotient
has enough precision to preserve the integer rounding rule. CUDA computes the
same blend with integer arithmetic. The cleanup uses signed chroma medians;
CUDA uses a packed comparison network and one cleanup launch. Every measured
Bayer sample remains exact.

Full paper Menon uses 192×96 CPU output tiles with a ten-pixel RAW halo, covering
its complete dependency radius of nine. Its seven stages and cached directional
gradients stay in per-worker scratch. Quarter-DN signed 16-bit candidates and
scale-144 reconstruction preserve the reference's exact rational arithmetic.
Each worker retains 673,920 bytes of paper scratch, about 5.14 MiB at eight
workers, in addition to input/output and wrapper padding. AVX512, AVX2, and
portable paths preserve the paper's refinement, tie, boundary, and rounding rules.

Validation covers:

- Byte-for-byte CPU/CUDA agreement across all 1,768 image/phase/cohort outputs,
  with every measured CFA sample preserved.
- Independent integer green and sorted-median oracles, scalar/SIMD agreement,
  odd and tiny dimensions, strided buffers, and row guards.
- Worker-count invariance, scratch reuse, allocation-failure behavior, and exact
  recovery on subsequent frames.
- CUDA memory and initialization checks with Compute Sanitizer.
- The full paper Menon implementation against its independent rational reference.

CPU tests can be rerun without downloading the dataset:

```bash
cmake -S cpu -B build/cpu -DCMAKE_BUILD_TYPE=Release -DDEBAYER_CPU_BUILD_DEMO=OFF
cmake --build build/cpu -j
ctest --test-dir build/cpu --output-on-failure
```

See [CPU test coverage](cpu/README.md#regression-coverage) and
[CUDA checks](c/tests/README.md) for detailed procedures. The full-corpus hash
check uses freshly built libraries and the downloaded references:

```bash
CUDA_VISIBLE_DEVICES=0 taskset -c 16-23 \
  build/benchmark-venv/bin/python benchmark/verify_softmenon_outputs.py \
  --build build/benchmark/build.json --output build/output-verification.json
```

Pass `--backends cpu` for a CPU-only hash check. Rebuild after source changes;
the checker validates source, binary, dataset, and reference-result hashes.

## Result artifacts

Result metadata pins the measured source files, compiled libraries, dataset
manifest, and output hashes.

- Per-image quality records for [CPU](benchmark/results/quality-cpu.json.gz),
  [CUDA](benchmark/results/quality-cuda.json.gz),
  [OpenCV](benchmark/results/quality-opencv.json.gz), and
  [NPP](benchmark/results/quality-npp.json.gz), including decoder and build provenance.
- [Quality aggregates and comparisons](benchmark/results/summary.json).
- Raw timing samples for [CPU](benchmark/results/timing-cpu.json),
  [CUDA](benchmark/results/timing-cuda.json),
  [OpenCV](benchmark/results/timing-opencv.json), and
  [NPP](benchmark/results/timing-npp.json).
- [Example image records](benchmark/results/examples.json).
- [Full-corpus output verification](benchmark/results/output-verification.json).

The production runner emits one `softmenon` method alongside `bilinear`,
`malvar`, and `menon2007`. The dataset manifest/downloader and production
runner above reconstruct the inputs and reproduce the supported methods'
scores, timings, and example images.
