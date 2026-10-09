# Benchmark methods and reproduction

This document describes how to reconstruct the 442-image dataset and reproduce
the quality, timing, and correctness results for SoftMenon and its baselines.
The library provides Bilinear, Malvar 2004, full paper Menon 2007, and SoftMenon.

## Reconstruct the exact dataset

Run from the repository root. Python 3.12 was used. The downloader uses only the
Python standard library; scoring additionally requires NumPy and Pillow.

```bash
python3 -m venv build/benchmark-venv
build/benchmark-venv/bin/pip install numpy==2.5.3 Pillow==12.3.0
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
used NumPy 2.5.3, Pillow 12.3.0, and libjpeg API version 6.2. Independent dataset
verification also matched with Pillow 10.2.0/libjpeg 8.0/libjpeg-turbo 2.1.5.
Versions are provenance; matching decoded bytes is the actual acceptance check.
If a URL changes, obtain the exact bytes from a mirror and verify against the
manifest instead of silently substituting another split or crop.

Dataset images retain their owners' terms and attribution. The manifest links
source pages and records use notes. Downloading data does not put it under this
repository's code license. The larger local RAW/video collection is **not needed**
for the PSNR results here: no native linear RAW or temporal quality claim follows
from this rendered-image benchmark.

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

The historical scorer summed only blue-channel error but divided by three-channel
sample count. Those old PSNR numbers are invalid and withdrawn. All numbers in
this document use the corrected all-channel scorer.

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
- **SoftMenon:** soft directional green interpolation followed by a 3×3 chroma
  median refinement. The median color differences reconstruct missing green
  from the measured red/blue sample, then reconstruct the other missing color.
  Every measured CFA sample is preserved.

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
CPU/CUDA on all 1,768 image/phase/cohort observations; this was measured rather
than assumed from its name.

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

| Dataset | Full paper Menon | SoftMenon | SoftMenon − paper |
|---|---:|---:|---:|
| Kodak24 | 39.2055 | **39.5499** | +0.3443 |
| McMaster | 34.2268 | **34.4812** | +0.2544 |
| Urban100 | 33.6647 | **33.9078** | +0.2431 |
| DIV2K validation | 38.3940 | **38.7985** | +0.4045 |
| BSDS500 test | 37.7975 | **38.6566** | +0.8591 |
| All 442, original | 36.9285 | **37.4928** | **+0.5643** |
| All 442, Inset16 | 36.7859 | **37.3762** | **+0.5903** |

Original pooled PSNR improves 0.4294 dB versus paper Menon. Interior8 mean PSNR
improves 0.5803 dB and border8 mean improves 0.4933 dB. The advantage is not
confined to a border or one aggregation rule.

SoftMenon wins on 377 scenes and loses on 65 versus paper Menon, after averaging
phases per scene. Its worst original loss is Urban100 `img_011_SRF_2_HR.png`,
**−7.6304 dB** (37.8043 versus 45.4347 dB). The Inset16 loss is −2.1966 dB.
These are average improvements, not a guarantee for every scene.

The pointwise 95% bootstrap interval for the original mean gain is
**[+0.4984, +0.6225] dB**. It uses 10,000 dataset-stratified source-image resamples,
keeping both phases together, seed 20261010. These images were also used during
algorithm development. The interval is not adjusted for that selection and is
not a held-out estimate of generalization. Results cover rendered uint8 RGB
remosaicing; linear sensor RAW and temporal behavior require separate evaluation.

The [README comparison](README.md#measured-comparison) includes the remaining
library algorithms and external baselines under the same metric contract.

## Timing results

The README's CPU/CUDA table uses the all-method run in
[softmenon-performance.json](benchmark/results/softmenon-performance.json),
`standard_mean_of_phase_medians_ms`. External adapter timings come from a
separate run on the same workstation. All calls include the work described in
the timing protocol above.

Additional SoftMenon measurements from the same workstation, mean of the
RGGB/BGGR medians:

| Path | SoftMenon ms |
|---|---:|
| CPU, 8 workers, native AVX512 dispatch | 0.496665 |
| CPU, 1 worker, native AVX512 dispatch | 2.774459 |
| CPU, 8 workers, forced AVX2 dispatch | 0.791920 |
| CPU, 1 worker, forced AVX2 dispatch | 5.260995 |
| CUDA, synchronous host-to-host | 0.366876 |

These are a separate timing session from the README table, not different
algorithms. The AVX2 experiment forces dispatch on the same Threadripper;
it does not measure a different AVX2-only processor. ARM CPU and Jetson runtime
performance remain unmeasured. Do not interpret sub-percent timing differences
as speedups without stable controls and repeated measurements.

## Implementation and correctness

SoftMenon's AVX512 green kernel handles 16 chromatic sites per 32-pixel strip;
AVX2 extracts the same sites from packed byte pairs. Measured-green sites pass
through unchanged. Scalar/vector tails retain the same candidate, score,
blend, rounding, and clipping results. The cleanup uses signed chroma medians;
CUDA uses a packed comparison network and one cleanup launch.

Full paper Menon uses 192×96 CPU output tiles with a ten-pixel RAW halo, covering
its complete dependency radius of nine. Its seven stages and cached directional
gradients stay in per-worker scratch. Quarter-DN signed 16-bit candidates and
scale-144 reconstruction preserve the reference's exact rational arithmetic.
Each worker retains 673,920 bytes of paper scratch, about 5.14 MiB at eight
workers, in addition to input/output and wrapper padding. AVX512, AVX2, and
portable paths preserve the paper's refinement, tie, boundary, and rounding rules.

Recorded validation includes:

- All 1,768 SoftMenon CPU/CUDA image/phase/cohort outputs match byte for byte,
  and all measured CFA samples remain unchanged.
- Final binaries match recorded SoftMenon and paper hashes in 7,072 checks:
  442 images × two cohorts × two phases × two methods × two backends.
- GCC, Clang, and ASan/UBSan pass all three CPU CTests. The independent sorted-median
  oracle covers 600 public API outputs and 1,600 direct cleanup outputs.
- Forced AVX2 initial reconstruction matches its reference on 288 comparisons
  spanning 67.4 million pixels.
- CUDA passes 192 regression cases and 160 cases each under compute-sanitizer
  memcheck and initcheck, with zero errors. SM87 and SM120 compile; runtime
  checks use SM120.
- Paper Menon matches its reference on 1,768 corpus reconstructions. Independent
  testing covers 292 synthetic cases, 1,195 public API comparisons, and 1,168
  direct scalar/runtime/AVX2/AVX512 comparisons.

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

The measured algorithm sources are pinned by the source hashes in the result
metadata; the SoftMenon implementation is at commit
`04836b961d808c72cf3e703cd9617831721778a6`. Documentation and reporting edits do
not change those algorithm outputs.

- [Per-image quality records](benchmark/results/softmenon-quality.jsonl.gz),
  [decoder/build metadata](benchmark/results/softmenon-quality.jsonl.meta.json),
  and [aggregates and worst scenes](benchmark/results/softmenon-quality-summary.json).
- [Timing samples and build provenance](benchmark/results/softmenon-performance.json).
- [SoftMenon exactness checks](benchmark/results/softmenon-quality-exactness.json)
  and [final output verification](benchmark/results/softmenon-final-output-verification.json).
- [Paper reference checks](benchmark/results/paper-cpu-exactness.json).

The original measurement files retain their immutable identifiers and control
records for provenance. In those files, `refined_softmenon` identifies the
SoftMenon algorithm documented here, and `paper` identifies full paper Menon.
For the additional timing session use the `candidate` records; the README
uses the `standard_cpu` and `standard_cuda` records. Archived controls are not
additional supported SoftMenon algorithms. The normal production runner emits
one `softmenon` method.

The dataset manifest/downloader and production runner above are sufficient to
reconstruct the inputs and reproduce the supported methods' scores, timings,
and example images. Exploratory notes and experiment plans stay local.
