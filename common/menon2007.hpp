#pragma once

// Menon, Andriani and Calvagno, IEEE TIP 16(1), 132-141 (2007), including
// the full three-part refinement. DOI: 10.1109/TIP.2006.884928.
// Algorithm adapted from Colour Developers' BSD-3-Clause implementation:
// colour-science/colour-demosaicing, commit f4f67d46c8a803164e9bc4b36d828931e6377e7c.
// Copyright 2015 Colour Developers. See COLOUR_DEMOSAICING_LICENSE beside
// this header. This implementation uses exact scale-144 integer arithmetic.
// Interpolation reads use reflect101. Classifier convolution has a zero halo,
// matching the pinned reference. Only final uint8 output is rounded/clipped.

#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define LIBDEBAYER_MENON_HD __host__ __device__ __forceinline__
#else
#define LIBDEBAYER_MENON_HD inline
#endif

namespace libdebayer_menon2007 {

constexpr int scale = 144;

// All pointers start at the valid image origin. RAW/output pitches are bytes.
// gh/gv: N int32 each; first/second: 3*N int32 each in BGR pixel order;
// direction: N bytes, 1 = horizontal, 0 = vertical. Planes never overlap.
// Call all pixels of each stage and complete a global barrier before the next:
// 1 RAW -> gh/gv; 2 gh/gv + RAW -> first/direction; 3 first -> second;
// 4 second -> first; 5 first -> second; 6 second -> first; 7 first -> output.
// Stage 2 also reads gh/gv neighbors. Stages 3-7 read immutable source planes.
struct Buffers {
    const uint8_t* raw;
    size_t raw_pitch;
    uint8_t* output;
    size_t output_pitch;
    int width;
    int height;
    bool rggb;
    int32_t* gh;
    int32_t* gv;
    int32_t* first;
    int32_t* second;
    uint8_t* direction;
};

LIBDEBAYER_MENON_HD bool valid(const Buffers& b, int x, int y) {
    return x >= 0 && y >= 0 && x < b.width && y < b.height;
}

LIBDEBAYER_MENON_HD int reflect(int coordinate, int length) {
    if (coordinate >= 0 && coordinate < length) return coordinate;
    if (length <= 1) return 0;
    const int64_t period = 2 * (static_cast<int64_t>(length) - 1);
    int64_t value = coordinate % period;
    if (value < 0) value += period;
    return static_cast<int>(value < length ? value : period - value);
}

LIBDEBAYER_MENON_HD size_t index(const Buffers& b, int x, int y) {
    return static_cast<size_t>(y) * b.width + x;
}

LIBDEBAYER_MENON_HD int color(const Buffers& b, int x, int y) {
    if ((x & 1) != (y & 1)) return 1;
    return (((x & 1) == 0) == b.rggb) ? 2 : 0;
}

LIBDEBAYER_MENON_HD int32_t raw(const Buffers& b, int x, int y) {
    x = reflect(x, b.width); y = reflect(y, b.height);
    return static_cast<int32_t>(b.raw[static_cast<size_t>(y) * b.raw_pitch + x]) * scale;
}

LIBDEBAYER_MENON_HD int32_t pixel(const Buffers& b, const int32_t* image, int x, int y, int channel) {
    x = reflect(x, b.width); y = reflect(y, b.height);
    return image[index(b, x, y) * 3 + channel];
}

LIBDEBAYER_MENON_HD int32_t chroma(const Buffers& b, const int32_t* green, int x, int y) {
    x = reflect(x, b.width); y = reflect(y, b.height);
    return color(b, x, y) == 1 ? 0 : raw(b, x, y) - green[index(b, x, y)];
}

LIBDEBAYER_MENON_HD int32_t absolute(int32_t value) {
    return value < 0 ? -value : value;
}

LIBDEBAYER_MENON_HD int32_t gradient(const Buffers& b, int x, int y, bool horizontal) {
    // Zero extension is applied to the gradient image itself. Its valid-image
    // forward chroma difference still reflects the +2 endpoint, as upstream.
    if (!valid(b, x, y)) return 0;
    const int32_t* green = horizontal ? b.gh : b.gv;
    const int nx = horizontal ? x + 2 : x;
    const int ny = horizontal ? y : y + 2;
    return absolute(chroma(b, green, x, y) - chroma(b, green, nx, ny));
}

LIBDEBAYER_MENON_HD int32_t classifier(const Buffers& b, int x, int y, bool horizontal) {
    // The asymmetric 5x5 kernel is convolved (not correlated). These offsets
    // encode its eight nonzero coefficients; the vertical kernel is transposed.
    const int dx[8] = {0, -2, -1, 0, -2, -1, 0, -2};
    const int dy[8] = {2, 2, 1, 0, 0, -1, -2, -2};
    const int weight[8] = {1, 1, 1, 3, 3, 1, 1, 1};
    int32_t result = 0;
    for (int i = 0; i < 8; ++i) {
        const int nx = x + (horizontal ? dx[i] : dy[i]);
        const int ny = y + (horizontal ? dy[i] : dx[i]);
        result += weight[i] * gradient(b, nx, ny, horizontal);
    }
    return result;
}

LIBDEBAYER_MENON_HD void stage_green(const Buffers& b, int x, int y) {
    if (!valid(b, x, y)) return;
    const size_t i = index(b, x, y);
    const int32_t center = raw(b, x, y);
    if (color(b, x, y) == 1) {
        b.gh[i] = b.gv[i] = center;
    } else {
        b.gh[i] = (raw(b, x - 1, y) + raw(b, x + 1, y)) / 2 +
                   (2 * center - raw(b, x - 2, y) - raw(b, x + 2, y)) / 4;
        b.gv[i] = (raw(b, x, y - 1) + raw(b, x, y + 1)) / 2 +
                   (2 * center - raw(b, x, y - 2) - raw(b, x, y + 2)) / 4;
    }
}

LIBDEBAYER_MENON_HD void stage_decision(const Buffers& b, int x, int y) {
    if (!valid(b, x, y)) return;
    const size_t i = index(b, x, y);
    const int measured = color(b, x, y);
    const bool horizontal = classifier(b, x, y, false) >= classifier(b, x, y, true);
    b.direction[i] = static_cast<uint8_t>(horizontal);
    b.first[i * 3] = measured == 0 ? raw(b, x, y) : 0;
    b.first[i * 3 + 1] = horizontal ? b.gh[i] : b.gv[i];
    b.first[i * 3 + 2] = measured == 2 ? raw(b, x, y) : 0;
}

LIBDEBAYER_MENON_HD int32_t neighbor_difference_sum(const Buffers& b, const int32_t* image,
        int x, int y, int target, int reference, bool horizontal) {
    const int dx = horizontal ? 1 : 0;
    const int dy = horizontal ? 0 : 1;
    return pixel(b, image, x - dx, y - dy, target) + pixel(b, image, x + dx, y + dy, target) -
           pixel(b, image, x - dx, y - dy, reference) - pixel(b, image, x + dx, y + dy, reference);
}

LIBDEBAYER_MENON_HD void stage_colors_at_green(const Buffers& b, int x, int y) {
    if (!valid(b, x, y)) return;
    const size_t i = index(b, x, y) * 3;
    for (int c = 0; c < 3; ++c) b.second[i + c] = b.first[i + c];
    if (color(b, x, y) != 1) return;
    for (int c = 0; c <= 2; c += 2) {
        const bool horizontal = color(b, x - 1, y) == c;
        b.second[i + c] = b.first[i + 1] + neighbor_difference_sum(b, b.first, x, y, c, 1, horizontal) / 2;
    }
}

LIBDEBAYER_MENON_HD void stage_opposite(const Buffers& b, int x, int y) {
    if (!valid(b, x, y)) return;
    const size_t p = index(b, x, y), i = p * 3;
    for (int c = 0; c < 3; ++c) b.first[i + c] = b.second[i + c];
    const int measured = color(b, x, y);
    if (measured == 1) return;
    const int target = 2 - measured;
    b.first[i + target] = b.second[i + measured] +
        neighbor_difference_sum(b, b.second, x, y, target, measured, b.direction[p] != 0) / 2;
}

LIBDEBAYER_MENON_HD int32_t three_difference_sum(const Buffers& b, const int32_t* image,
        int x, int y, int target, int reference, bool horizontal) {
    return neighbor_difference_sum(b, image, x, y, target, reference, horizontal) +
           pixel(b, image, x, y, target) - pixel(b, image, x, y, reference);
}

LIBDEBAYER_MENON_HD void stage_refine_green(const Buffers& b, int x, int y) {
    if (!valid(b, x, y)) return;
    const size_t p = index(b, x, y), i = p * 3;
    for (int c = 0; c < 3; ++c) b.second[i + c] = b.first[i + c];
    const int measured = color(b, x, y);
    if (measured == 1) return;
    b.second[i + 1] = b.first[i + measured] -
        three_difference_sum(b, b.first, x, y, measured, 1, b.direction[p] != 0) / 3;
}

LIBDEBAYER_MENON_HD void stage_refine_colors_at_green(const Buffers& b, int x, int y) {
    if (!valid(b, x, y)) return;
    const size_t i = index(b, x, y) * 3;
    for (int c = 0; c < 3; ++c) b.first[i + c] = b.second[i + c];
    if (color(b, x, y) != 1) return;
    for (int c = 0; c <= 2; c += 2) {
        const bool horizontal = color(b, x - 1, y) == c;
        b.first[i + c] = b.second[i + 1] + neighbor_difference_sum(b, b.second, x, y, c, 1, horizontal) / 2;
    }
}

LIBDEBAYER_MENON_HD uint8_t quantize(int32_t value) {
    if (value <= 0) return 0;
    if (value >= 255 * scale) return 255;
    return static_cast<uint8_t>((value + scale / 2) / scale);
}

LIBDEBAYER_MENON_HD void stage_refine_opposite_output(const Buffers& b, int x, int y) {
    if (!valid(b, x, y)) return;
    const size_t p = index(b, x, y), i = p * 3;
    const int measured = color(b, x, y);
    uint8_t* output = b.output + static_cast<size_t>(y) * b.output_pitch + static_cast<size_t>(x) * 3;
    for (int c = 0; c < 3; ++c) {
        int32_t value = b.first[i + c];
        if (measured != 1 && c == 2 - measured)
            value = b.first[i + measured] + three_difference_sum(b, b.first, x, y, c, measured, b.direction[p] != 0) / 3;
        output[c] = quantize(value);
    }
}

} // namespace libdebayer_menon2007

#undef LIBDEBAYER_MENON_HD
