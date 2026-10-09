#pragma once

// Exact tiled CPU implementation of common/menon2007.hpp, following Menon,
// Andriani and Calvagno (2007), DOI 10.1109/TIP.2006.884928. The reference is
// adapted from Colour Developers' BSD-3-Clause code; copyright 2015 Colour
// Developers. See ../common/COLOUR_DEMOSAICING_LICENSE.
//
// Each worker owns its workspace; only disjoint output rows are shared. The
// classifier's asymmetric support and forward gradient together reach radius 4
// in RAW. Five subsequent neighbor stages extend the dependency radius to 9.
// The ten-pixel tile halo therefore isolates every output core from artificial
// tile boundaries. At true image boundaries each completed stage independently
// uses reflect101, except for the classifier's zero-extended gradient planes.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define MENON_CPU_X86 1
#define MENON_CPU_INLINE __attribute__((always_inline)) inline
#else
#define MENON_CPU_X86 0
#define MENON_CPU_INLINE inline
#endif

namespace menon2007_cpu {
constexpr int scale = 144;
constexpr int core_width = 192, core_height = 96, halo = 10;
constexpr int pitch = core_width + 2 * halo + 4;
constexpr int plane_rows = core_height + 2 * halo + 4;
constexpr int plane_size = pitch * plane_rows;

struct Workspace {
    // Initial candidates are exact quarter-DN values in [-510,1530]. Chroma
    // differences lie in [-1020,1020], gradients in [0,2040], and the weighted
    // classifier in [0,24480]. No initial int16 value is saturated or rounded.
    alignas(64) std::array<int16_t, 5 * plane_size> initial;
    // Selected green and measured colors expand by 36 to exact scale 144 before
    // the remaining stages. These wider planes permit overshoot until output.
    alignas(64) std::array<int32_t, 4 * plane_size> colors;
};

namespace detail {
MENON_CPU_INLINE int fold(int x, int size) {
    if (x >= 0 && x < size)
        return x;
    const int period = 2 * (size - 1);
    x %= period;
    if (x < 0)
        x += period;
    return x < size ? x : period - x;
}

template <typename T> MENON_CPU_INLINE void pad(T *p, int width, int height, bool zero = false) {
    for (int y = 0; y < height; ++y) {
        T *row = p + y * pitch;
        for (int x : {-2, -1, width, width + 1})
            row[x] = zero ? 0 : row[fold(x, width)];
    }
    for (int y : {-2, -1, height, height + 1}) {
        T *row = p + y * pitch;
        const T *source = p + fold(y, height) * pitch;
        for (int x = -2; x < width + 2; ++x)
            row[x] = zero ? 0 : source[x];
    }
}

MENON_CPU_INLINE int absolute(int x) {
    return x < 0 ? -x : x;
}
// Stored paper values are exact at scale 144, so both refinement numerators
// are multiples of 3. Unsigned modular multiplication recovers their signed
// quotient exactly; memcpy preserves its bit representation without a signed
// narrowing conversion. Even loose interval bounds keep these numerators
// below 2 million in magnitude, well within int32.
MENON_CPU_INLINE int32_t divide3(int32_t numerator) {
    const uint32_t bits = static_cast<uint32_t>(numerator) * uint32_t{0xAAAAAAAB};
    int32_t result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
MENON_CPU_INLINE uint8_t quantize(int x) {
    return static_cast<uint8_t>(std::min(255, std::max(0, x + scale / 2) / scale));
}

MENON_CPU_INLINE void tile(Workspace &scratch, const uint8_t *raw, size_t raw_pitch, uint8_t *output,
                           size_t output_pitch, int image_width, int image_height, bool rggb, int output_x,
                           int output_y, int output_width, int output_height) {
    const int left = std::max(0, output_x - halo), top = std::max(0, output_y - halo);
    const int width = std::min(image_width, output_x + output_width + halo) - left;
    const int height = std::min(image_height, output_y + output_height + halo) - top;
    int16_t *initial = scratch.initial.data() + 2 * pitch + 2;
    int32_t *colors = scratch.colors.data() + 2 * pitch + 2;
    int16_t *cfa = initial;
    int16_t *gh = initial + plane_size;
    int16_t *gv = initial + 2 * plane_size;
    int16_t *dh = initial + 3 * plane_size;
    int16_t *dv = initial + 4 * plane_size;
    int32_t *blue = colors;
    int32_t *green = colors + plane_size;
    int32_t *red = colors + 2 * plane_size;
    int32_t *direction = colors + 3 * plane_size;
    for (int y = 0; y < height; ++y) {
        const uint8_t *input = raw + static_cast<size_t>(top + y) * raw_pitch + left;
        int16_t *dst = cfa + y * pitch;
        for (int x = 0; x < width; ++x)
            dst[x] = 4 * input[x];
    }
    pad(cfa, width, height);
    // Directional green candidates; measured green remains exact.
    for (int y = 0; y < height; ++y) {
        const int16_t *c = cfa + y * pitch;
        int16_t *h = gh + y * pitch;
        int16_t *v = gv + y * pitch;
        const int chromatic = (top + y - left) & 1;
        for (int x = 0; x < width; ++x) {
            h[x] = (x & 1) == chromatic ? (c[x - 1] + c[x + 1]) / 2 + (2 * c[x] - c[x - 2] - c[x + 2]) / 4
                                        : c[x];
            v[x] = (x & 1) == chromatic ? (c[x - pitch] + c[x + pitch]) / 2 +
                                              (2 * c[x] - c[x - 2 * pitch] - c[x + 2 * pitch]) / 4
                                        : c[x];
        }
    }
    pad(gh, width, height);
    pad(gv, width, height);
    // Cache each directional forward chroma gradient exactly once.
    for (int y = 0; y < height; ++y) {
        const int16_t *c = cfa + y * pitch;
        const int16_t *h = gh + y * pitch;
        const int16_t *v = gv + y * pitch;
        int16_t *a = dh + y * pitch;
        int16_t *b = dv + y * pitch;
        const int chromatic = (top + y - left) & 1;
        for (int x = 0; x < width; ++x) {
            a[x] = (x & 1) == chromatic ? absolute(c[x] - h[x] - c[x + 2] + h[x + 2]) : 0;
            b[x] = (x & 1) == chromatic ? absolute(c[x] - v[x] - c[x + 2 * pitch] + v[x + 2 * pitch]) : 0;
        }
    }
    pad(dh, width, height, true);
    pad(dv, width, height, true);
    // The asymmetric eight-tap classifier is a convolution, as in the paper
    // reference. Unit-stride classification also computes unused directions at
    // green sites, enabling full-width SIMD without gathers or masked stores.
    for (int y = 0; y < height; ++y) {
        const int16_t *h = dh + y * pitch;
        const int16_t *v = dv + y * pitch;
        const int16_t *c = cfa + y * pitch;
        int32_t *b = blue + y * pitch;
        int32_t *g = green + y * pitch;
        int32_t *r = red + y * pitch;
        int32_t *d = direction + y * pitch;
        const int16_t *candidate_h = gh + y * pitch;
        const int16_t *candidate_v = gv + y * pitch;
        const int chromatic = (top + y - left) & 1;
        const bool blue_row = ((top + y) & 1) == static_cast<int>(rggb);
        for (int x = 0; x < width; ++x) {
            const int32_t mask = -static_cast<int32_t>((x & 1) == chromatic);
            b[x] = (36 * c[x]) & mask & -static_cast<int32_t>(blue_row);
            r[x] = (36 * c[x]) & mask & -static_cast<int32_t>(!blue_row);
            g[x] = 36 * c[x];
            d[x] = 0;
        }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC ivdep
#endif
        for (int x = 0; x < width; ++x) {
            const int sh = h[x + 2 * pitch] + h[x + 2 * pitch - 2] + h[x + pitch - 1] + 3 * h[x] +
                           3 * h[x - 2] + h[x - pitch - 1] + h[x - 2 * pitch] + h[x - 2 * pitch - 2];
            const int sv = v[x + 2] + v[x - 2 * pitch + 2] + v[x - pitch + 1] + 3 * v[x] +
                           3 * v[x - 2 * pitch] + v[x - pitch - 1] + v[x - 2] + v[x - 2 * pitch - 2];
            d[x] = sv >= sh;
            g[x] = 36 * (sv >= sh ? candidate_h[x] : candidate_v[x]);
        }
    }
    pad(blue, width, height);
    pad(green, width, height);
    pad(red, width, height);
    // Each stage writes only its missing CFA class. All neighboring values
    // consumed by that stage are from other classes and remain unchanged.
    for (int y = 0; y < height; ++y) {
        int32_t *b = blue + y * pitch;
        int32_t *r = red + y * pitch;
        const int32_t *g = green + y * pitch;
        const int chromatic = (top + y - left) & 1;
        const int baxis = (((top + y) & 1) == static_cast<int>(rggb)) ? 1 : pitch;
        const int raxis = baxis == 1 ? pitch : 1;
        for (int x = 1 - chromatic; x < width; x += 2) {
            b[x] = g[x] + (b[x - baxis] + b[x + baxis] - g[x - baxis] - g[x + baxis]) / 2;
            r[x] = g[x] + (r[x - raxis] + r[x + raxis] - g[x - raxis] - g[x + raxis]) / 2;
        }
    }
    pad(blue, width, height);
    pad(red, width, height);
    for (int y = 0; y < height; ++y) {
        const bool blue_row = ((top + y) & 1) == static_cast<int>(rggb);
        int32_t *target = (blue_row ? red : blue) + y * pitch;
        const int32_t *measured = (blue_row ? blue : red) + y * pitch;
        const int32_t *d = direction + y * pitch;
        const int chromatic = (top + y - left) & 1;
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC ivdep
#endif
        for (int x = chromatic; x < width; x += 2) {
            const int horizontal = target[x - 1] + target[x + 1] - measured[x - 1] - measured[x + 1];
            const int vertical =
                target[x - pitch] + target[x + pitch] - measured[x - pitch] - measured[x + pitch];
            target[x] = measured[x] + (d[x] ? horizontal : vertical) / 2;
        }
    }
    pad(blue, width, height);
    pad(red, width, height);
    for (int y = 0; y < height; ++y) {
        int32_t *g = green + y * pitch;
        const int32_t *c = ((((top + y) & 1) == static_cast<int>(rggb)) ? blue : red) + y * pitch;
        const int32_t *d = direction + y * pitch;
        const int chromatic = (top + y - left) & 1;
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC ivdep
#endif
        for (int x = chromatic; x < width; x += 2) {
            const int horizontal = c[x - 1] + c[x + 1] - g[x - 1] - g[x + 1];
            const int vertical = c[x - pitch] + c[x + pitch] - g[x - pitch] - g[x + pitch];
            g[x] = c[x] - divide3((d[x] ? horizontal : vertical) + c[x] - g[x]);
        }
    }
    pad(green, width, height);
    for (int y = 0; y < height; ++y) {
        int32_t *b = blue + y * pitch;
        int32_t *r = red + y * pitch;
        const int32_t *g = green + y * pitch;
        const int chromatic = (top + y - left) & 1;
        const int baxis = (((top + y) & 1) == static_cast<int>(rggb)) ? 1 : pitch;
        const int raxis = baxis == 1 ? pitch : 1;
        for (int x = 1 - chromatic; x < width; x += 2) {
            b[x] = g[x] + (b[x - baxis] + b[x + baxis] - g[x - baxis] - g[x + baxis]) / 2;
            r[x] = g[x] + (r[x - raxis] + r[x + raxis] - g[x - raxis] - g[x + raxis]) / 2;
        }
    }
    pad(blue, width, height);
    pad(red, width, height);
    for (int y = output_y - top; y < output_y - top + output_height; ++y) {
        const int32_t *b = blue + y * pitch;
        const int32_t *r = red + y * pitch;
        const int32_t *g = green + y * pitch;
        const bool blue_row = ((top + y) & 1) == static_cast<int>(rggb);
        const int32_t *target = blue_row ? r : b;
        const int32_t *measured = blue_row ? b : r;
        const int32_t *d = direction + y * pitch;
        const int chromatic = (top + y - left) & 1;
        uint8_t *dst = output + static_cast<size_t>(top + y) * output_pitch + 3 * output_x;
        for (int x = output_x - left; x < output_x - left + output_width; ++x) {
            int bv = b[x], rv = r[x];
            if ((x & 1) == chromatic) {
                const int horizontal = target[x - 1] + target[x + 1] - measured[x - 1] - measured[x + 1];
                const int vertical =
                    target[x - pitch] + target[x + pitch] - measured[x - pitch] - measured[x + pitch];
                const int opposite =
                    measured[x] + divide3((d[x] ? horizontal : vertical) + target[x] - measured[x]);
                if (blue_row)
                    rv = opposite;
                else
                    bv = opposite;
            }
            dst[0] = quantize(bv);
            dst[1] = quantize(g[x]);
            dst[2] = quantize(rv);
            dst += 3;
        }
    }
}

MENON_CPU_INLINE void rows(Workspace &scratch, const uint8_t *raw, size_t raw_pitch, uint8_t *output,
                           size_t output_pitch, int width, int height, bool rggb, int begin_y, int end_y) {
    begin_y = std::max(0, begin_y);
    end_y = std::min(height, end_y);
    for (int y = begin_y; y < end_y; y += core_height)
        for (int x = 0; x < width; x += core_width)
            tile(scratch, raw, raw_pitch, output, output_pitch, width, height, rggb, x, y,
                 std::min(core_width, width - x), std::min(core_height, end_y - y));
}
} // namespace detail

inline void scalar_rows(Workspace &scratch, const uint8_t *raw, size_t raw_pitch, uint8_t *output,
                        size_t output_pitch, int width, int height, bool rggb, int begin_y, int end_y) {
    detail::rows(scratch, raw, raw_pitch, output, output_pitch, width, height, rggb, begin_y, end_y);
}

#if MENON_CPU_X86
__attribute__((target("avx2"))) inline void avx2_rows(Workspace &scratch, const uint8_t *raw,
                                                      size_t raw_pitch, uint8_t *output, size_t output_pitch,
                                                      int width, int height, bool rggb, int begin_y,
                                                      int end_y) {
    detail::rows(scratch, raw, raw_pitch, output, output_pitch, width, height, rggb, begin_y, end_y);
}
__attribute__((target("avx512f,avx512bw,avx512vl"))) inline void
avx512_rows(Workspace &scratch, const uint8_t *raw, size_t raw_pitch, uint8_t *output, size_t output_pitch,
            int width, int height, bool rggb, int begin_y, int end_y) {
    detail::rows(scratch, raw, raw_pitch, output, output_pitch, width, height, rggb, begin_y, end_y);
}
#endif

inline void process_rows(Workspace &scratch, const uint8_t *raw, size_t raw_pitch, uint8_t *output,
                         size_t output_pitch, int width, int height, bool rggb, int begin_y, int end_y) {
#if MENON_CPU_X86
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512vl")) {
        avx512_rows(scratch, raw, raw_pitch, output, output_pitch, width, height, rggb, begin_y, end_y);
        return;
    }
    if (__builtin_cpu_supports("avx2")) {
        avx2_rows(scratch, raw, raw_pitch, output, output_pitch, width, height, rggb, begin_y, end_y);
        return;
    }
#endif
    scalar_rows(scratch, raw, raw_pitch, output, output_pitch, width, height, rggb, begin_y, end_y);
}
} // namespace menon2007_cpu

#undef MENON_CPU_INLINE
