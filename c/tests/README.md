# CUDA correctness tests

Configure with CMake 3.18+ and the CUDA architecture for the target GPU. For
example, this development host uses `120`; Jetson Orin uses `87`.

```sh
cmake -S c -B build/c -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/c -j
ctest --test-dir build/c --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 77 build/c/debayer_regression --quick
compute-sanitizer --tool initcheck --error-exitcode 77 build/c/debayer_regression --uninitialized-output
```

The CUDA regression covers all five exported reconstruction functions at tiny,
odd, aligned, unaligned, 1920×1080, and 1921×1081 dimensions. It checks complete
flat-color reconstruction, exact native-sample preservation, independent CPU
bilinear/MHC filters, phase-preserving reflected input padding, allocation
canaries, untouched output halos and stride bytes, invalid arguments, and
independence from previous output contents. The initcheck mode leaves output
interiors uninitialized to detect reads before the Menon stages write them.

The separate C translation unit verifies that the public API remains usable
and linkable from C. Installed consumers may use `find_package(libdebayer CONFIG
REQUIRED)` with target `libdebayer::debayer`, or `pkg-config libdebayer`. The old
explicit `<prefix>/lib/cmake` package location remains supported.
