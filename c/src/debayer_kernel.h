#pragma once

#include <stdint.h>
#include <stddef.h>

#include <cuda_runtime.h>

//------------------------------------------------------------------------------
// CUDA Kernels

extern __global__ void mirrorEdgesTopBottom(
    uint8_t* data,
    int width,
    int height,
    ptrdiff_t pitch,
    int pad);

extern __global__ void mirrorEdgesLeftRight(
    uint8_t* data,
    int width,
    int height,
    ptrdiff_t pitch,
    int pad);

extern __global__ void rggb_malvar2004(
    const uint8_t* raw,
    ptrdiff_t raw_pitch,
    uint8_t* bgr,
    ptrdiff_t bgr_pitch,
    int width,
    int height);

extern __global__ void bggr_malvar2004(
    const uint8_t* raw,
    ptrdiff_t raw_pitch,
    uint8_t* bgr,
    ptrdiff_t bgr_pitch,
    int width,
    int height);

extern __global__ void rggb_bilinear(
    const uint8_t* raw,
    ptrdiff_t raw_pitch,
    uint8_t* bgr,
    ptrdiff_t bgr_pitch,
    int width,
    int height);

extern __global__ void rggb_softmenon_g(
    const uint8_t* raw,
    ptrdiff_t raw_pitch,
    uint8_t* bgr,
    ptrdiff_t bgr_pitch,
    int width,
    int height);

extern __global__ void rggb_softmenon_rb(
    uint8_t* bgr,
    ptrdiff_t bgr_pitch,
    int width,
    int height);

extern __global__ void bggr_softmenon_g(
    const uint8_t* raw,
    ptrdiff_t raw_pitch,
    uint8_t* bgr,
    ptrdiff_t bgr_pitch,
    int width,
    int height);

extern __global__ void bggr_softmenon_rb(
    uint8_t* bgr,
    ptrdiff_t bgr_pitch,
    int width,
    int height);

extern __global__ void bggr_bilinear(
    const uint8_t* raw,
    ptrdiff_t raw_pitch,
    uint8_t* bgr,
    ptrdiff_t bgr_pitch,
    int width,
    int height);
