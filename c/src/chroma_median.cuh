#pragma once

// Immutable 3x3 B-G / R-G median with measured-color green reconstruction.
// Preserve measured CFA samples; clamp reconstructed green before adding
// the other color's median chroma.
// BGR byte layout; source/destination pitches are independent and in bytes.
// The caller owns distinct, nonoverlapping source and destination images.
// This header allocates/copies/synchronizes nothing: one asynchronous launch.
// Packed compare-exchanges use native signed16x2 on SM90+ and a scalar fallback.

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace median_optimized_detail {

__device__ __forceinline__ void compare_exchange(int& a, int& b) {
    const int low = min(a, b);
    b = max(a, b);
    a = low;
}

__device__ __forceinline__ int low_signed(uint32_t value) {
    const int part = static_cast<int>(value & 0xffffu);
    return part < 32768 ? part : part - 65536;
}

__device__ __forceinline__ int high_signed(uint32_t value) {
    const int part = static_cast<int>(value >> 16);
    return part < 32768 ? part : part - 65536;
}

__device__ __forceinline__ uint32_t pack_signed(int low, int high) {
    return (static_cast<uint32_t>(low) & 0xffffu) |
           (static_cast<uint32_t>(high) << 16);
}

__device__ __forceinline__ void compare_exchange(uint32_t& a, uint32_t& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    uint32_t low, high;
    // Signed independent halfwords: B-G is low, R-G is high.
    asm("min.s16x2 %0, %1, %2;" : "=r"(low) : "r"(a), "r"(b));
    asm("max.s16x2 %0, %1, %2;" : "=r"(high) : "r"(a), "r"(b));
    a = low;
    b = high;
#else
    const int al = low_signed(a), ah = high_signed(a);
    const int bl = low_signed(b), bh = high_signed(b);
    a = pack_signed(min(al, bl), min(ah, bh));
    b = pack_signed(max(al, bl), max(ah, bh));
#endif
}

template <typename T>
__device__ __forceinline__ T median9(T p0, T p1, T p2, T p3, T p4,
                                    T p5, T p6, T p7, T p8) {
    // 19 compare-exchanges; only p4 is the sorted median, other positions
    // are deliberately unspecified. Also works lane-wise for packed pairs.
    compare_exchange(p1, p2); compare_exchange(p4, p5); compare_exchange(p7, p8);
    compare_exchange(p0, p1); compare_exchange(p3, p4); compare_exchange(p6, p7);
    compare_exchange(p1, p2); compare_exchange(p4, p5); compare_exchange(p7, p8);
    compare_exchange(p0, p3); compare_exchange(p5, p8); compare_exchange(p4, p7);
    compare_exchange(p3, p6); compare_exchange(p1, p4); compare_exchange(p2, p5);
    compare_exchange(p4, p7); compare_exchange(p4, p2); compare_exchange(p6, p4);
    compare_exchange(p4, p2);
    return p4;
}

__device__ __forceinline__ uint32_t differences(const uint8_t* pixel) {
    const int green = pixel[1];
    return pack_signed(static_cast<int>(pixel[0]) - green,
                       static_cast<int>(pixel[2]) - green);
}

template <bool Packed>
__device__ __forceinline__ uint32_t median_pair(const uint32_t (&v)[9]) {
    if constexpr (Packed) {
        return median9(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
    } else {
        const int blue = median9(low_signed(v[0]), low_signed(v[1]), low_signed(v[2]),
                                low_signed(v[3]), low_signed(v[4]), low_signed(v[5]),
                                low_signed(v[6]), low_signed(v[7]), low_signed(v[8]));
        const int red = median9(high_signed(v[0]), high_signed(v[1]), high_signed(v[2]),
                               high_signed(v[3]), high_signed(v[4]), high_signed(v[5]),
                               high_signed(v[6]), high_signed(v[7]), high_signed(v[8]));
        return pack_signed(blue, red);
    }
}

__device__ __forceinline__ uint8_t saturated(int value) {
    return static_cast<uint8_t>(max(0, min(value, 255)));
}

__device__ __forceinline__ void write_pixel(uint8_t* destination, size_t pitch,
        int x, int y, bool rggb, int green, uint32_t center, uint32_t filtered) {
    uint8_t* output = destination + static_cast<size_t>(y) * pitch + static_cast<size_t>(x) * 3;
    const bool chromatic_site = (x & 1) == (y & 1);
    const bool measured_red = chromatic_site && (((x & 1) == 0) == rggb);
    const bool measured_blue = chromatic_site && !measured_red;
    const int original_blue = green + low_signed(center);
    const int original_red = green + high_signed(center);
    if (measured_blue) green = saturated(original_blue - low_signed(filtered));
    if (measured_red) green = saturated(original_red - high_signed(filtered));
    output[0] = measured_blue ? static_cast<uint8_t>(original_blue)
                             : saturated(green + low_signed(filtered));
    output[1] = static_cast<uint8_t>(green);
    output[2] = measured_red ? static_cast<uint8_t>(original_red)
                            : saturated(green + high_signed(filtered));
}

template <bool Packed>
__global__ void direct_kernel(const uint8_t* __restrict__ source, size_t source_pitch,
        uint8_t* __restrict__ destination, size_t destination_pitch,
        int width, int height, bool rggb) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    uint32_t values[9];
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
#pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
            const uint8_t* row = source + static_cast<size_t>(y + dy) * source_pitch;
#pragma unroll
            for (int dx = -1; dx <= 1; ++dx)
                values[(dy + 1) * 3 + dx + 1] = differences(row + static_cast<size_t>(x + dx) * 3);
        }
    } else {
        // Radius one needs at most one reflection, including width/height=1.
        const int xs[3] = {x > 0 ? x - 1 : min(1, width - 1), x,
                           x < width - 1 ? x + 1 : max(width - 2, 0)};
        const int ys[3] = {y > 0 ? y - 1 : min(1, height - 1), y,
                           y < height - 1 ? y + 1 : max(height - 2, 0)};
#pragma unroll
        for (int dy = 0; dy < 3; ++dy) {
            const uint8_t* row = source + static_cast<size_t>(ys[dy]) * source_pitch;
#pragma unroll
            for (int dx = 0; dx < 3; ++dx)
                values[dy * 3 + dx] = differences(row + static_cast<size_t>(xs[dx]) * 3);
        }
    }
    const int green = source[static_cast<size_t>(y) * source_pitch + static_cast<size_t>(x) * 3 + 1];
    write_pixel(destination, destination_pitch, x, y, rggb, green, values[4], median_pair<Packed>(values));
}

} // namespace median_optimized_detail

inline cudaError_t softmenon_chroma_median(const uint8_t* source, size_t source_pitch,
        uint8_t* destination, size_t destination_pitch, int width, int height,
        bool rggb, cudaStream_t stream) {
    const dim3 block(32, 8), grid((width - 1) / 32 + 1, (height - 1) / 8 + 1);
    median_optimized_detail::direct_kernel<true><<<grid, block, 0, stream>>>(
        source, source_pitch, destination, destination_pitch, width, height, rggb);
    return cudaGetLastError();
}
