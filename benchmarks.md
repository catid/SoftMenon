# Benchmark methods and reproduction

This document describes the data, scoring, implementation controls, and selection
process behind SoftMenon. The original custom Menon implementation and the full
Menon 2007 paper implementation are different baselines. The first experiments
improved the custom implementation; a subsequent comparison found that the full
paper implementation had higher mean PSNR. We therefore started a fresh set of
one-factor experiments from the full paper implementation. Historical gains must
not be attributed to that stronger baseline.

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

Historical border scoring additionally used bands of width 1/2/4/8 and the four
4-by-4 corners. A border band contains each pixel once; its sample count is three
times the number of pixels in the union. For width 8, full SSE equals interior8
SSE plus border8 SSE. Regional perfect matches retain infinite PSNR; finite-only
PSNR deltas are labeled as such, while MSE and win/loss counts retain every case.

## Baselines and algorithm identity

- **Bilinear:** local independent-channel interpolation.
- **Malvar 2004:** the Malvar–He–Cutler cross-channel correction filters.
- **Menon 2007:** full DDFAPD, including its three refinement stages, following
  [Menon, Andriani and Calvagno, TIP 2007](https://doi.org/10.1109/TIP.2006.884928)
  and the [pinned Colour implementation](https://github.com/colour-science/colour-demosaicing/blob/f4f67d46c8a803164e9bc4b36d828931e6377e7c/colour_demosaicing/bayer/demosaicing/menon2007.py).
- **Legacy Menon control:** the repository's earlier custom integer interpolation,
  which used different classifiers, close-score averaging, intermediate rounding
  and clipping, and omitted paper refinement. It is retained only as a benchmark
  diagnostic, not labeled as a reproduction of the paper.
- **Initial SoftMenon:** legacy interpolation plus soft initial green blending
  and one sample-preserving 3-by-3 chroma median. This was the winner of the
  legacy-based experiment. Paper-based ablations are a separate experiment.

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
was checked for Bilinear, Malvar, and paper Menon. The legacy backends retain
small historical arithmetic differences and are scored separately. Initial
SoftMenon matched across CPU/CUDA on all 884 original image/phase observations;
this was measured rather than assumed from its name.

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
can be reaggregated without running kernels again. Baseline diagnostics
`legacy_menon` and `soft_green` are included to make old-versus-new claims explicit.

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
and runtime SIMD dispatch. Its optimization and matched before/after experiment
are described below. Initial SoftMenon has separate SIMD kernels.

Examples use Kodak `kodim11.png` (the existing building scene) and `kodim19.png`
(lighthouse), BGGR. Captions must use those individual full-image PSNR values,
not the Kodak mean or interior-only scores.

## Exact CPU optimization of full paper Menon

The original full-paper CPU implementation at commit
`ad2eb3642e0212473cdefd591078380c6a1c4508` executed the shared reference one pixel
at a time, with seven full-image stage barriers. Its posterior classifier
recomputed directional gradients for every filter tap. An instrumented profile
at 1080p attributed about 8.48 ms of the roughly 16 ms eight-worker call to that
classifier alone. Stage instrumentation is diagnostic; reported speedups use
uninstrumented public API calls.

The optimized implementation caches the directional gradients, keeps the seven
paper stages in per-worker tiles, and updates only each stage's missing CFA
class. Neighboring CFA classes needed by that stage remain unchanged. A single
dispatch across workers completes an entire frame. Compiler-generated AVX2 and
AVX512F/BW/VL loops handle tile arithmetic, selected at runtime with a generic
fallback. Every refinement, classifier tie rule, boundary rule and final
rounding rule is preserved; none of the quality ablations below is included.

Output tiles are 192 by 96 pixels with a ten-pixel RAW context halo. The complete
dependency radius is nine; artificial tile boundaries cannot affect the output
core. Real image edges retain stage-specific reflection and zero-gradient rules.
Initial candidates and gradients use exact quarter-DN signed 16-bit storage,
then expand to scale 144 before color reconstruction. There is no reduced
precision or saturation. Each active worker retains 673,920 bytes of Menon
scratch: about 5.14 MiB at eight workers, versus 65.26 MiB of paper-specific
scratch for the old 1080p implementation. Input/output and wrapper padding
buffers are additional. Refinement divisions by three use an exact modular
inverse for their provably divisible integer numerators.

Matched before/after results on the workstation above, using the same 1080p
DIV2K crop and averaging the two phase medians:

| Method | Workers | Before ms | After ms | Speedup |
|---|---:|---:|---:|---:|
| Full paper Menon | 1 | 120.9568 | 8.3342 | **14.51×** |
| Full paper Menon | 8 | 16.0186 | 1.2682 | **12.63×** |
| Malvar control | 1 | 4.6125 | 4.6262 | 1.00× |
| Malvar control | 8 | 0.6783 | 0.6738 | 1.01× |
| Initial SoftMenon control | 1 | 3.6042 | 3.6473 | 0.99× |
| Initial SoftMenon control | 8 | 0.6009 | 0.5943 | 1.01× |

The optimized paper implementation is about 1.88 times Malvar's latency in this
matched eight-worker run, with 2.965 dB higher mean PSNR. Control differences
of roughly one percent are timing variation, not algorithm changes. This is
a separate matched experiment from the original README measurement of
15.947 ms. Raw samples, phase medians, p10/p90, source and binary hashes,
input provenance and hardware details are retained in
[paper-cpu-optimization.json](benchmark/results/paper-cpu-optimization.json).
This workstation selects the AVX512 paper path. The speedup is not a measured
claim for AVX2-only CPUs or ARM/Jetson CPUs.

A separate single-core diagnostic calls each instruction path directly:
generic fallback 24.475 ms, AVX2 19.902 ms, AVX512 8.043 ms on the same crop.
All match the frozen old public output. These timings retain tile processing
but exclude public wrapper padding and dispatch; they are not interchangeable
with the public API table. The generic path may use baseline compiler
auto-vectorization. Its samples and embedded harness are in
[paper-cpu-isa.json](benchmark/results/paper-cpu-isa.json).

Final validation found **zero changed output bytes in 1,768 reconstructions**:
442 images, both Bayer phases, original and independently remosaiced Inset16
cohorts, totaling 1,350,584,288 compared pixels. Mean original PSNR therefore
remains **36.928451 dB**. All measured CFA samples and input bytes were preserved.
The independent reference test passed 292 synthetic cases, 1,195 public API
comparisons and 1,168 direct scalar/runtime/AVX2/AVX512 sliced comparisons with
GCC 13.3, Clang 18.1.3, and AddressSanitizer/UndefinedBehaviorSanitizer. The final
test inputs are deterministic across compilers. Counts, commands and hashes
are recorded in [paper-cpu-exactness.json](benchmark/results/paper-cpu-exactness.json).

To recreate the before/after experiment, download the dataset above, then build
the pinned old checkout and the current checkout separately:

```bash
git worktree add --detach build/paper-before ad2eb3642e0212473cdefd591078380c6a1c4508
build/benchmark-venv/bin/python build/paper-before/benchmark/run.py \
  --build --backends cpu --output build/cpu-before
build/benchmark-venv/bin/python benchmark/run.py \
  --build --backends cpu --output build/cpu-after
taskset -c 16-23 build/benchmark-venv/bin/python benchmark/compare_cpu.py \
  --before build/cpu-before/build.json --before-root build/paper-before \
  --after build/cpu-after/build.json --after-root . \
  --output build/cpu-compare.json --check-small
```

The comparison runner verifies both binary and source hashes, checks the exact
DIV2K input and decoded pixels, and randomly interleaves old/new calls for paper
Menon, Malvar and initial SoftMenon. It checks matching output bytes before
timing and stable output hashes afterward. Eight-worker settings use 100 rounds
of five calls, one-worker settings use 25 rounds of one call, both after ten
warmup rounds. The one-worker run uses the first selected physical core.
The two unchanged algorithms are controls for run-to-run timing variation.

For full-corpus reproduction, the normal quality runner records every output
hash. Run it from both checkouts, sharing the downloaded images:

```bash
ln -s "$(pwd)/datasets" build/paper-before/datasets
build/benchmark-venv/bin/python build/paper-before/benchmark/run.py \
  --backends cpu --output build/cpu-before --quality --insets 0 16 --workers 8
build/benchmark-venv/bin/python benchmark/run.py \
  --backends cpu --output build/cpu-after --quality --insets 0 16 --workers 8
```

Match quality rows by `(method, path, inset, pattern)` and compare their
`output_sha256_rgb` fields and integer SSE totals. The independent C++ reference
test described in [cpu/README.md](cpu/README.md) additionally exercises synthetic
edge cases and every supported CPU instruction path without downloading data.

## How the first candidates were selected

These were exploratory screens on the same 442 sources. They identify useful
mechanisms, not held-out estimates of generalization. Candidate definitions were
fixed before each sweep; subsequent combinations used the observed winners.
Paired bootstrap intervals used 2,000 resamples of **source images**, retaining
phases together, seed 20261009. They are pointwise, not selection-adjusted.

### Border screen: retain reflect101

We tested 16 fixed same-phase halo patterns and four external controls, alongside
the native path. Every external halo had width 8, preserving CFA origin. Seven
legacy lanes were tested: CUDA bilinear RGGB, CUDA Malvar RGGB/BGGR, and custom
Menon RGGB/BGGR on CPU and CUDA. Original and inset cohorts plus an inset
real-context diagnostic produced **133,042 reconstructions**. The real-context
diagnostic crops a reconstruction of the original image; it sees information
unavailable at a true boundary and is not a deployable policy or guaranteed upper
bound. Native and externally reflected outputs were identical in 6,188 pairs.

For an exterior coordinate `q`, let `a` be the nearest in-bounds coordinate of
the same parity, `d=abs(q-a)/2`, `t=-d`. Taps `j=0..k-1` move inward from `a`
in steps of two RAW pixels, keeping both green parities separate.

| Patterns | Fixed coefficients |
|---|---|
| Slope gains 0.25, 0.5, 0.75, 1, 1.25 (5) | Two taps: `[1+gain*d, -gain*d]` |
| Linear least squares, supports 3/4/5 (3) | `z_j=j-(k-1)/2`, `u=t-(k-1)/2`; `w_j=1/k+u*z_j/sum(z*z)` |
| Quadratic, supports 3 and 5 (2) | Add `(u*u-m2)*(z_j*z_j-m2)/sum((z*z-m2)**2)` to the linear weights, `m2=(k*k-1)/12` |
| Mean2, mean3, binomial3 (3) | `[1,1]/2`, `[1,1,1]/3`, `[1,2,1]/4` |
| Reflection/nearest mixes (3) | Reflection weights 0.25/0.5/0.75, remaining weight on same-phase nearest |

Use separable float64 evaluation and tensor-product corner weights, with no
intermediate clipping or rounding; finally apply nearest-even rounding and clip
to uint8. Keep original pixels unchanged. If an axis has too few same-phase
samples, reduce support; quadratic falls back to linear at two taps and constant
at one. Means renormalize; binomial uses the shorter binomial row. Controls were
reflect101, same-phase nearest, ordinary scalar RAW replication, and zero fill.
Reflection folds with period `2*(length-1)` and does not repeat the edge.

No candidate consistently improved source-average finite border2 PSNR.
For example, inset CUDA custom Menon binomial3 improved **pooled** border2
PSNR from 33.7377 to 33.7868 dB, yet worsened mean per-image/phase border2 PSNR
by 0.1331 dB. On original boundaries it reduced full-image mean PSNR by 0.0066 dB.
RAW replication and zero fill also fail constant-color reconstruction. We kept
reflect101 rather than selecting from pooled scores alone.

### Ten single changes to the legacy algorithms

The screen held reflection fixed and compared each candidate only with its own
parent, producing **20,332 reconstructions**. These are research-inspired local
adaptations, not ten complete paper implementations. All candidates are shown:

| Change | Original mean PSNR delta | Inset16 delta |
|---|---:|---:|
| Custom Menon: soft initial green | +0.3698 | +0.3600 |
| Custom Menon: chroma median | +0.2622 | +0.2672 |
| Bilinear: soft green | +0.0888 | +0.0899 |
| Bilinear: hard green direction | +0.0171 | +0.0200 |
| Custom Menon: always average diagonals | -0.0115 | -0.0121 |
| Custom Menon: local green clamp | -0.1400 | -0.1384 |
| Custom Menon: Hamilton–Adams classifier | -0.3194 | -0.3258 |
| Custom Menon: remove close-score averages | -0.7721 | -0.7943 |
| Malvar: local output clamp | -1.0489 | -1.0116 |
| Malvar: half cross-channel correction | -1.7704 | -1.7771 |

Soft green kept the legacy directional candidates `Gh,Gv` and classifiers `Sh,Sv`:

```text
G = round(((Sv+1)*Gh + (Sh+1)*Gv) / (Sh+Sv+2))
```

The one-DN stabilizer was fixed, not fitted. Chroma cleanup first completed the
unchanged reconstruction `I`, then formed signed `R-G` and `B-G`. Each missing
red/blue value became `clip(G + median_3x3(C-G))`, reading an immutable source with
reflect101. Every green value and measured red/blue sample stayed unchanged.

Other mechanisms: bilinear hard green chose the smaller `abs(left-right)` or
`abs(up-down)` difference, averaging on a tie; bilinear soft green used crossed
gradient-plus-one weights. Hamilton–Adams scores were
`abs(GL-GR)+abs(2*C0-CL2-CR2)` and the vertical counterpart. Clamps used measured
target-channel extrema in the bilinear footprint. Half-correction Malvar used
`B+(M-B)/2` on unrounded estimates before final quantization. Diagonal averaging
retained the legacy two-step rounded color-difference means. Removing close
averages retained hard horizontal preference on ties in green and diagonal stages.

### Combining the two legacy winners

Four variants, two phases, and two cohorts produced **7,072 reconstructions**:

| Legacy-family variant | Original mean PSNR | Inset16 mean PSNR |
|---|---:|---:|
| Legacy control | 36.0089 | 35.8807 |
| Soft green only | 36.3787 | 36.2407 |
| Chroma median only | 36.2711 | 36.1479 |
| Both (initial SoftMenon) | 36.6116 | 36.4781 |

The combined original gain was +0.6027 dB versus legacy and +0.2329 dB versus
soft green alone. On McMaster, it **lost 0.1354 dB** versus legacy and 0.2445 dB
versus soft green. The gains therefore do not apply uniformly to every dataset.
The two gains were slightly subadditive; their combination was measured directly.

An exact median implementation optimization preserved all 1,768 output/green
hashes. A fixed 19-comparator network, packed signed 16-bit GPU arithmetic,
persistent scratch, and removal of per-frame allocation/copy reduced isolated
1080p cleanup from 303.85 to 13.93 microseconds on this GPU. Historical warm
combined latency changed from 0.736755 to 0.366107 ms, versus 0.357447 ms for
soft-only in that same optimization run. These historical experimental timings
are not the final production timing table. CPU optimization separately uses
runtime SIMD dispatch and immutable-source reconstruction; exactness is checked
against frozen scalar outputs.

## Why the new sweep starts from paper Menon

The corrected, full-refinement CUDA baseline scores **36.9285 dB**, versus
**36.6116 dB** for initial SoftMenon, on the same original 442 images/two phases.
Paper Menon wins mean PSNR by 0.3168 dB. Pooled PSNR reverses this ordering
(35.8299 versus 35.9263 dB), which is why both are recorded and the primary
metric is stated before selection.

| Dataset | Full paper Menon | Initial SoftMenon |
|---|---:|---:|
| Kodak24 | 39.2055 | 38.5976 |
| McMaster | 34.2268 | 34.8191 |
| Urban100 | 33.6647 | 33.2525 |
| DIV2K validation | 38.3940 | 38.3451 |
| BSDS500 test | 37.7975 | 37.3475 |

The new experiment changes one mechanism at a time from **full paper Menon**.
It preserves all unchanged refinement and boundary rules and checks an isolated
control against production before interpreting candidate scores. No legacy gain
is carried over, and no combination is selected before measuring its ingredients.
The following ten candidates were frozen before looking at the new scores.
Each has paper Menon as its direct parent; none contains another candidate.

| Candidate | The one change |
|---|---|
| `initial_green_soft` | Blend initial `Gh,Gv` with crossed posterior-score-plus-one weights; leave the original hard direction map intact in every later stage |
| `initial_green_close_average` | Average initial `Gh,Gv` only when `10*abs(Sh-Sv) <= Sh+Sv`; leave the original hard direction map intact |
| `classifier_uniform` | Change the eight classifier weights from `[1,1,1,3,3,1,1,1]` to all ones; keep their spatial support |
| `initial_green_clamp` | Clamp the selected initial green to the extrema of four axial measured greens; compute the original scores/map before clamping |
| `omit_refine_green` | Copy through the first refinement stage, retaining both subsequent refinements |
| `omit_refine_colors_at_green` | Copy through the second refinement stage, retaining the first and third |
| `omit_refine_opposite` | Skip only the third refinement correction, then quantize normally |
| `half_refine_green` | Use `oldG + (paperNewG-oldG)/2` in the first refinement |
| `median_refine_green` | Replace only the first refinement's directional three-sample mean of signed `C-G` differences by their median |
| `output_chroma_median` | Append one immutable-source 3×3 median of final quantized `R-G` and `B-G`, preserving every green value and measured CFA sample |

The close-average threshold is fixed at a 10% normalized score difference,
including equality; zero/zero scores average. It is not a fit to these results.
The median green refinement uses the center and its two immediately adjacent
neighbors along the original hard-selected axis; its output is
`measured_C - median3(CminusG_left, CminusG_center, CminusG_right)`.
Refinement omissions are contribution/cost controls. Uniform classifier weights
test the paper's central-axis emphasis. Soft selection and range constraints are
adaptations of weighted edge-aware and PPG ideas; the two medians use the
[Freeman color-difference principle](https://patents.google.com/patent/US4774565A/en).

Experimental precision is **10368 = 144 × 72** for all eleven lanes, including
the control. This is an exact representation change for paper Menon. For soft
green, set `epsilon=10368` (one DN of the **summed** classifier, not its average):

```text
n = (Sv+epsilon)*Gh + (Sh+epsilon)*Gv
d = Sh + Sv + 2*epsilon
Gstored = 72 * floor(n/(72*d) + 1/2)
```

Use signed int64 products and round half ties toward positive infinity. This
rounds this one new rational blend to 1/144 DN; the remaining denominators
`2*2*3*2*3=72` then divide exactly, avoiding an unintended truncation change in
later stages. Half refinement and the other candidates are exact at this scale.
All unchanged stages retain the paper formulas and boundary rules; clipping and
uint8 quantization occur only at output. The output-median candidate alone adds
a filter after that quantization. Every variant has its own compiled library,
derived from a frozen copy of the paper header, with source and binary hashes.

Before scoring, all 11 variants passed 182 fixtures each: 2,002 outputs, 8,008
measured-CFA checks, pitched equivalence, input immutability, and buffer guards.
The control matched all 182 independent rational reference outputs exactly.
The final chroma median matched an independent NumPy composition on all 182
fixtures and preserved the complete green plane. A separate NumPy interpretation
of all ten mechanisms also matched all 2,002 outputs and checked 6,289,268
selected intermediate divisions for exact divisibility, including 382 negative
soft-blend numerators and ten exact half ties.

The isolated research kernels are not optimized production implementations:
omitted refinements retain a plane-copy kernel, and the output median uses
insertion sorting. Their timings compare these experimental implementations;
they do not establish the best achievable speed of each mechanism.

The complete sweep contains
**19,448 reconstructions**: 442 sources × two cohorts × two phases × eleven lanes.
Its original control must additionally match all 884 production output hashes.

### Paper-based single-change results

All values below are measured against **full paper Menon**, with every unchanged
stage retained. Primary quality is mean per-image/per-phase PSNR. Positive delta
means improvement. GPU times use the isolated experimental bridge described above;
they are not the optimized initial SoftMenon timings in the README.

| Single change | Original delta dB | Inset16 delta dB | Original interior8 delta dB | Original PSNR dB | 1080p GPU ms |
|---|---:|---:|---:|---:|---:|
| Full paper control | 0 | 0 | 0 | 36.9285 | 0.5401 |
| `initial_green_soft` | +0.4575 | +0.4701 | +0.4625 | 37.3860 | 0.5405 |
| `initial_green_close_average` | +0.2961 | +0.2970 | +0.2972 | 37.2246 | 0.5368 |
| `initial_green_clamp` | +0.1788 | +0.1845 | +0.1786 | 37.1072 | 0.5395 |
| `output_chroma_median` | +0.0750 | +0.0781 | +0.0760 | 37.0035 | 0.7852 |
| `omit_refine_opposite` | -0.0500 | -0.0501 | -0.0496 | 36.8784 | 0.5337 |
| `median_refine_green` | -0.0853 | -0.0859 | -0.0843 | 36.8431 | 0.5393 |
| `classifier_uniform` | -0.1161 | -0.1167 | -0.1203 | 36.8123 | 0.5396 |
| `half_refine_green` | -0.2353 | -0.2369 | -0.2368 | 36.6931 | 0.5396 |
| `omit_refine_colors_at_green` | -0.2700 | -0.2716 | -0.2710 | 36.6585 | 0.5283 |
| `omit_refine_green` | -0.5545 | -0.5577 | -0.5579 | 36.3739 | 0.5322 |

Dataset-specific original full-image mean PSNR changes:

| Single change | Kodak24 | McMaster | Urban100 | DIV2K | BSDS500 |
|---|---:|---:|---:|---:|---:|
| `initial_green_soft` | +0.3970 | +0.1757 | +0.1427 | +0.3828 | +0.6849 |
| `initial_green_close_average` | +0.3003 | +0.1226 | +0.2061 | +0.2498 | +0.3795 |
| `initial_green_clamp` | +0.0238 | +0.4292 | +0.2562 | +0.1684 | +0.1413 |
| `output_chroma_median` | +0.0447 | -0.0841 | +0.1064 | -0.0174 | +0.1235 |
| `omit_refine_opposite` | -0.0697 | +0.0757 | -0.0195 | +0.0040 | -0.1013 |
| `median_refine_green` | -0.0992 | +0.1812 | -0.0521 | -0.0079 | -0.1630 |
| `classifier_uniform` | -0.1145 | +0.0114 | -0.0675 | -0.0562 | -0.1821 |
| `half_refine_green` | -0.2689 | +0.1887 | -0.2196 | -0.0945 | -0.3477 |
| `omit_refine_colors_at_green` | -0.3119 | +0.2579 | -0.2498 | -0.0879 | -0.4136 |
| `omit_refine_green` | -0.6301 | +0.3235 | -0.5074 | -0.2808 | -0.7849 |

**Soft initial green has the highest mean quality**, +0.4575 dB, with a
pointwise 95% bootstrap interval of [+0.4130, +0.4970] dB. It improves the
average on every dataset and gains +0.4625 dB in the original interior8 region.
Pooled PSNR also improves, by +0.4063 dB. It is essentially tied with the paper
control in this exploratory GPU timing run (0.5405 versus 0.5401 ms). This is
one change to the initial green estimate; downstream hard direction choices and
all three paper refinements remain intact.

**Close-score averaging is the more consistent candidate on these scenes.**
It gains +0.2961 dB [pointwise 95% interval +0.2863, +0.3054], improving 441 of
442 source images when phases are averaged per scene. Its only loss is Urban100
`img_081_SRF_2_HR.png`, -0.0643 dB. Soft green improves 385 scenes and loses on
57; its worst is Urban100 `img_011_SRF_2_HR.png`, -5.0461 dB on the original
image and -2.1288 dB with inset16. The higher mean therefore does not imply a
better worst case. Keep both candidates for subsequent evaluation; these are
two alternative green decisions, not ingredients already tested together.

Final chroma cleanup gains only +0.0750 dB from the stronger paper baseline,
with losses on DIV2K and McMaster. Removing refinements loses mean quality.
These results support retaining the full paper pipeline while improving its
initial decision. No combination of these paper-based candidates has been
measured or selected in this sweep.

The new confidence intervals use 2,000 **dataset-stratified source-image**
bootstrap samples, seed 20261009, keeping phases of each scene together and
preserving dataset counts. They are pointwise and not corrected for selection
among ten candidates. The reused datasets and inset crops are exploratory
evidence, not a held-out confirmation set.

Independent audit verified all 19,448 unique expected rows, all 884 original
paper hashes and error totals against production, and all 720 summary groups
by recomputing macro/pooled PSNR, paired scene deltas, medians and win/loss counts
from stored channel errors. Every measured CFA sample was preserved. Timing
used 10 warmups and 100 randomized rounds of five calls per lane, with all
timed outputs checked. Differences below about one percent are not treated as
established speedups. The median attachment here is the unoptimized insertion
sort reference, and stage omissions retain copy launches. These CPU variants
have not been implemented or timed.

## Reproduction scope

The checked-in dataset manifest/downloader and production benchmark runner are
the supported route to reconstruct data and reproduce current production scores
and examples. The large historical research folders, per-image experimental
artifacts, and isolated variant generators remain local and ignored. Historical
tables above preserve their cohort, mechanism, and baseline identity; reproducing
those exploratory kernels requires reconstructing the stated variants rather
than treating today's paper implementation as the old custom baseline.
