# Debayer on CPU

## Setup

On Linux:

```bash
sudo apt update
sudo apt install -y build-essential cmake git libopencv-dev
```

On Mac:

```bash
brew install opencv llvm libomp
```

## Build and test

From the repository root:

```bash
cmake -S cpu -B build/cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu -j
ctest --test-dir build/cpu --output-on-failure
```

The `debayer_cpu_lib` target provides the library. The `debayer_cpu` image-directory
demo requires OpenCV; use `-DDEBAYER_CPU_BUILD_DEMO=OFF` to build only the library
and core regression tests without OpenCV. `BUILD_TESTING=OFF` disables the tests.

## Supported inputs

This CPU implementation accepts 8-bit RGGB and BGGR images with width and height
at least two, including odd dimensions. Choose `SARONIC_DEBAYER_BILINEAR`,
`SARONIC_DEBAYER_MALVAR2004`, `SARONIC_DEBAYER_MENON2007`, or
`SARONIC_DEBAYER_SOFTMENON`. Menon implements DDFAPD with full paper refinement;
SoftMenon retains the custom interpolation with soft green decisions and a
sample-preserving median of color differences. Output channels are BGR. Row
pitches are byte counts; zero means tightly packed, otherwise each pitch must
cover its complete image row.
The caller owns the input and output buffers and must allocate enough storage
for the declared dimensions and pitches.

Interpolation borders use reflection without repeating the edge sample, which
preserves the Bayer phase. The paper Menon classifier zero-extends its gradient
image, matching the reference algorithm. Calls to `Process` on the same instance
are serialized. Different instances own independent worker pools. `Debayer debayer;` defaults to at most
eight workers; `Debayer debayer(4);` explicitly selects four, and `WorkerCount()`
reports the configured count, including the calling thread. A setting of one
uses no helper threads.
Scratch buffers are retained across frames and resized when dimensions change.
Bilinear and Malvar write directly to the caller's output; SoftMenon fuses its
final cleanup and output channel order. Full paper Menon caches its gradients
and runs all refinement stages in private per-worker tiles, preserving the
reference output exactly. AVX512 and AVX2 kernels are selected at runtime on
supported x86 CPUs, with a portable scalar fallback. Paper Menon requires
AVX512F/BW/VL for its widest path; the median kernel also uses VBMI.
Do not call `Allocate`,
`Free`, or the destructor concurrently with `Process` on that instance. `Debayer` cannot be copied.

`Process` returns zero on success, `-1` for null pointers, `-2` for invalid
dimensions or pitches, `-3` for allocation or task-submission failure, and `-4`
for unsupported algorithm or CFA selectors.

## Regression coverage

`cpu_regression` checks constant colors, affine color planes, measured CFA sample
preservation for all four algorithms, repeated outputs, odd and small dimensions,
strides and buffer guards, worker-count equivalence, algorithm/size switches,
invalid descriptors, concurrent calls, and thread-pool
completion after a failed task submission. `cpu_psnr_regression` additionally
checks the demo's PSNR calculation across all three channels when OpenCV is enabled.

`cpu_menon_reference` compares full paper Menon byte for byte against the
independent implementation in `common/menon2007.hpp`. It checks tiny, odd, HD,
random and extreme images, tile boundaries, strides and guards, worker counts,
algorithm switches, and scalar/AVX2/AVX512 paths available on the host. Direct
kernel checks also process arbitrary row slices in reverse order and verify
that rows outside each slice remain untouched.

Run the core tests with AddressSanitizer and UndefinedBehaviorSanitizer:

```bash
cmake -S cpu -B build/cpu-sanitizer \
  -DDEBAYER_CPU_BUILD_DEMO=OFF -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined'
cmake --build build/cpu-sanitizer -j
ASAN_OPTIONS=detect_leaks=1 ctest --test-dir build/cpu-sanitizer --output-on-failure
```
