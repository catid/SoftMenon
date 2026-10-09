#include "debayer_kernel.h"

/*
    References:

    [1] "HIGH-QUALITY LINEAR INTERPOLATION FOR DEMOSAICING OF BAYER-PATTERNED COLOR IMAGES" (Malvar 2004)
        https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Demosaicing_ICASSP04.pdf

    [2] "Demosaicing With Directional Filtering and a posteriori Decision" (Menon 2007)
        https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=8c8e4a3cf6d0b8dfdcd36652718ad54afd2fe5fe
*/

#include <stdio.h>


//------------------------------------------------------------------------------
// Tools

#define BLUE  0
#define GREEN 1
#define RED   2

#define B_SET_CENTER(x, y) \
    const ptrdiff_t V = raw_pitch; \
    const uint8_t* P = block + y * V + x;

#define B_AT(x, y) (int16_t)P[y * V + x]

__device__ inline uint8_t saturate_cast_int16_to_uint8(int16_t val) {
    return static_cast<uint8_t>(max(0, min(255, val)));
}

__device__ inline void WriteBGRBlockPixel(
    uint8_t* bgr_block,
    ptrdiff_t bgr_pitch,
    int quad_x,
    int quad_y,
    int16_t b,
    int16_t g,
    int16_t r)
{
    uint8_t* bgr_pixel = bgr_block + quad_y * bgr_pitch + quad_x * 3;
    bgr_pixel[0] = saturate_cast_int16_to_uint8(b);
    bgr_pixel[1] = saturate_cast_int16_to_uint8(g);
    bgr_pixel[2] = saturate_cast_int16_to_uint8(r);
}


//------------------------------------------------------------------------------
// Mirror Edges Kernels

// Reflect without repeating the edge sample (reflect-101). The even period
// preserves Bayer phase, including when an image dimension is odd.
__device__ inline int reflect_coordinate(ptrdiff_t coordinate, int length) {
    if (coordinate >= 0 && coordinate < length) return static_cast<int>(coordinate);
    const ptrdiff_t period = 2 * (static_cast<ptrdiff_t>(length) - 1);
    coordinate %= period;
    if (coordinate < 0) coordinate += period;
    return static_cast<int>(coordinate < length ? coordinate : period - coordinate);
}

__global__ void mirrorEdgesTopBottom(uint8_t* data, int width, int height, ptrdiff_t pitch, int pad)
{
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (x >= static_cast<size_t>(width) + 2 * pad || y >= static_cast<size_t>(pad)) return;
    const uint8_t* image = data + pitch * pad + pad;
    const int source_x = reflect_coordinate(static_cast<ptrdiff_t>(x) - pad, width);
    const int top_y = reflect_coordinate(static_cast<ptrdiff_t>(y) - pad, height);
    const int bottom_y = reflect_coordinate(height + static_cast<ptrdiff_t>(y), height);
    data[y * pitch + x] = image[top_y * pitch + source_x];
    data[(static_cast<ptrdiff_t>(pad) + height + y) * pitch + x] = image[bottom_y * pitch + source_x];
}

__global__ void mirrorEdgesLeftRight(uint8_t* data, int width, int height, ptrdiff_t pitch, int pad)
{
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (x >= static_cast<size_t>(pad) || y >= static_cast<size_t>(height)) return;
    const uint8_t* image = data + pitch * pad + pad;
    data[(y + pad) * pitch + x] = image[y * pitch + reflect_coordinate(static_cast<ptrdiff_t>(x) - pad, width)];
    data[(y + pad) * pitch + pad + width + x] = image[y * pitch + reflect_coordinate(width + static_cast<ptrdiff_t>(x), width)];
}

__device__ inline int16_t read_bgr_reflected(const uint8_t* bgr, ptrdiff_t pitch,
    int width, int height, int x, int y, int channel)
{
    return bgr[reflect_coordinate(y, height) * pitch +
               static_cast<ptrdiff_t>(reflect_coordinate(x, width)) * 3 + channel];
}


//------------------------------------------------------------------------------
// Malvar 2004 Algorithm

__global__ void rggb_malvar2004(
    const uint8_t* raw,
    ptrdiff_t raw_pitch,
    uint8_t* bgr,
    ptrdiff_t bgr_pitch,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= (width + 1) / 2 || y >= (height + 1) / 2) return;

    // We are looking at a 2x2 block
    const uint8_t* block = reinterpret_cast<const uint8_t*>(raw + (y * raw_pitch + x) * 2);
    uint8_t* bgr_block = reinterpret_cast<uint8_t*>(bgr + (y * bgr_pitch + static_cast<ptrdiff_t>(x) * 3) * 2);

    /*
        RGGB layout:

            R G
            G B

        G at R/B:

            0  0 -1  0  0
            0  0  2  0  0
           -1  2  4  2 -1 + 4) / 8
            0  0  2  0  0
            0  0 -1  0  0

        R at G, R column:

            0  0 -2  0  0
            0 -2  8 -2  0
            1  0 10  0  1 + 8) / 16
            0 -2  8 -2  0
            0  0 -2  0  0

        R at G, B column:

            0  0  1  0  0
            0 -2  0 -2  0
           -2  8 10  8 -2 + 8) / 16
            0 -2  0 -2  0
            0  0  1  0  0

        B at G, B column:

            0  0 -2  0  0
            0 -2  8 -2  0
            1  0 10  0  1 + 8) / 16
            0 -2  8 -2  0
            0  0 -2  0  0

        B at G, R column:

            0  0  1  0  0
            0 -2  0 -2  0
           -2  8 10  8 -2 + 8) / 16
            0 -2  0 -2  0
            0  0  1  0  0

        R at B, B column:

            0  0 -3  0  0
            0  4  0  4  0
           -3  0 12  0 -3 + 8) / 16
            0  4  0  4  0
            0  0 -3  0  0

        B at R, R column:

            0  0 -3  0  0
            0  4  0  4  0
           -3  0 12  0 -3 + 8) / 16
            0  4  0  4  0
            0  0 -3  0  0
    */

    // Upper left:
    if (x * 2 + 0 < width && y * 2 + 0 < height) {
        B_SET_CENTER(0,0);
        int16_t b = (12 * B_AT(0,0)
                   + 4 * (B_AT(-1,-1) + B_AT(-1,1) + B_AT(1,-1) + B_AT(1,1))
                   - 3 * (B_AT(0,-2) + B_AT(0,2) + B_AT(-2,0) + B_AT(2,0)) + 8) / 16;
        int16_t g = (4 * B_AT(0,0)
                   + 2 * (B_AT(0,-1) + B_AT(0,1) + B_AT(-1,0) + B_AT(1,0))
                   - (B_AT(0,-2) + B_AT(0,2) + B_AT(-2,0) + B_AT(2,0)) + 4) / 8;
        int16_t r = B_AT(0,0);
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 0, 0, b, g, r);
    }

    // Upper right:
    if (x * 2 + 1 < width && y * 2 + 0 < height) {
        B_SET_CENTER(1,0);
        int16_t b = (10 * B_AT(0,0)
                   + 8 * (B_AT(0,-1) + B_AT(0,1))
                   - 2 * (B_AT(-1,-1) + B_AT(-1,1) + B_AT(1,-1) + B_AT(1,1) + B_AT(0,-2) + B_AT(0,2))
                   + B_AT(-2,0) + B_AT(2,0) + 8) / 16;
        int16_t g = B_AT(0,0);
        int16_t r = (10 * B_AT(0,0)
                   + 8 * (B_AT(-1,0) + B_AT(1,0))
                   - 2 * (B_AT(-1,-1) + B_AT(1,1) + B_AT(-1,1) + B_AT(1,-1) + B_AT(-2,0) + B_AT(2,0))
                   + B_AT(0,-2) + B_AT(0,2) + 8) / 16;
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 1, 0, b, g, r);
    }

    // Lower left:
    if (x * 2 + 0 < width && y * 2 + 1 < height) {
        B_SET_CENTER(0,1);
        int16_t b = (10 * B_AT(0,0)
                   + 8 * (B_AT(-1,0) + B_AT(1,0))
                   - 2 * (B_AT(-1,-1) + B_AT(1,1) + B_AT(-1,1) + B_AT(1,-1) + B_AT(-2,0) + B_AT(2,0))
                   + B_AT(0,-2) + B_AT(0,2) + 8) / 16;
        int16_t g = B_AT(0,0);
        int16_t r = (10 * B_AT(0,0)
                   + 8 * (B_AT(0,-1) + B_AT(0,1))
                   - 2 * (B_AT(-1,-1) + B_AT(-1,1) + B_AT(1,-1) + B_AT(1,1) + B_AT(0,-2) + B_AT(0,2))
                   + B_AT(-2,0) + B_AT(2,0) + 8) / 16;
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 0, 1, b, g, r);
    }

    // Lower right:
    if (x * 2 + 1 < width && y * 2 + 1 < height) {
        B_SET_CENTER(1,1);
        int16_t b = B_AT(0,0);
        int16_t g = (4 * B_AT(0,0)
                   + 2 * (B_AT(0,-1) + B_AT(0,1) + B_AT(-1,0) + B_AT(1,0))
                   - (B_AT(0,-2) + B_AT(0,2) + B_AT(-2,0) + B_AT(2,0)) + 4) / 8;
        int16_t r = (12 * B_AT(0,0)
                   + 4 * (B_AT(-1,-1) + B_AT(-1,1) + B_AT(1,-1) + B_AT(1,1))
                   - 3 * (B_AT(0,-2) + B_AT(0,2) + B_AT(-2,0) + B_AT(2,0)) + 8) / 16;
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 1, 1, b, g, r);
    }
}

__global__ void bggr_malvar2004(
    const uint8_t* raw,
    ptrdiff_t raw_pitch,
    uint8_t* bgr,
    ptrdiff_t bgr_pitch,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= (width + 1) / 2 || y >= (height + 1) / 2) return;

    // We are looking at a 2x2 block
    const uint8_t* block = reinterpret_cast<const uint8_t*>(raw + (y * raw_pitch + x) * 2);
    uint8_t* bgr_block = reinterpret_cast<uint8_t*>(bgr + (y * bgr_pitch + static_cast<ptrdiff_t>(x) * 3) * 2);

    /*
        BGGR layout:

            B G
            G R

        Just swap the R and B channels from RGGB code.
    */

    // Upper left:
    if (x * 2 + 0 < width && y * 2 + 0 < height) {
        B_SET_CENTER(0,0);
        int16_t b = B_AT(0,0);
        int16_t g = (4 * B_AT(0,0)
                   + 2 * (B_AT(0,-1) + B_AT(0,1) + B_AT(-1,0) + B_AT(1,0))
                   - (B_AT(0,-2) + B_AT(0,2) + B_AT(-2,0) + B_AT(2,0)) + 4) / 8;
        int16_t r = (12 * B_AT(0,0)
                   + 4 * (B_AT(-1,-1) + B_AT(-1,1) + B_AT(1,-1) + B_AT(1,1))
                   - 3 * (B_AT(0,-2) + B_AT(0,2) + B_AT(-2,0) + B_AT(2,0)) + 8) / 16;
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 0, 0, b, g, r);
    }

    // Upper right:
    if (x * 2 + 1 < width && y * 2 + 0 < height) {
        B_SET_CENTER(1,0);
        int16_t b = (10 * B_AT(0,0)
                   + 8 * (B_AT(-1,0) + B_AT(1,0))
                   - 2 * (B_AT(-1,-1) + B_AT(1,1) + B_AT(-1,1) + B_AT(1,-1) + B_AT(-2,0) + B_AT(2,0))
                   + B_AT(0,-2) + B_AT(0,2) + 8) / 16;
        int16_t g = B_AT(0,0);
        int16_t r = (10 * B_AT(0,0)
                   + 8 * (B_AT(0,-1) + B_AT(0,1))
                   - 2 * (B_AT(-1,-1) + B_AT(-1,1) + B_AT(1,-1) + B_AT(1,1) + B_AT(0,-2) + B_AT(0,2))
                   + B_AT(-2,0) + B_AT(2,0) + 8) / 16;
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 1, 0, b, g, r);
    }

    // Lower left:
    if (x * 2 + 0 < width && y * 2 + 1 < height) {
        B_SET_CENTER(0,1);
        int16_t b = (10 * B_AT(0,0)
                   + 8 * (B_AT(0,-1) + B_AT(0,1))
                   - 2 * (B_AT(-1,-1) + B_AT(-1,1) + B_AT(1,-1) + B_AT(1,1) + B_AT(0,-2) + B_AT(0,2))
                   + B_AT(-2,0) + B_AT(2,0) + 8) / 16;
        int16_t g = B_AT(0,0);
        int16_t r = (10 * B_AT(0,0)
                   + 8 * (B_AT(-1,0) + B_AT(1,0))
                   - 2 * (B_AT(-1,-1) + B_AT(1,1) + B_AT(-1,1) + B_AT(1,-1) + B_AT(-2,0) + B_AT(2,0))
                   + B_AT(0,-2) + B_AT(0,2) + 8) / 16;
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 0, 1, b, g, r);
    }

    // Lower right:
    if (x * 2 + 1 < width && y * 2 + 1 < height) {
        B_SET_CENTER(1,1);
        int16_t b = (12 * B_AT(0,0)
                   + 4 * (B_AT(-1,-1) + B_AT(-1,1) + B_AT(1,-1) + B_AT(1,1))
                   - 3 * (B_AT(0,-2) + B_AT(0,2) + B_AT(-2,0) + B_AT(2,0)) + 8) / 16;
        int16_t g = (4 * B_AT(0,0)
                   + 2 * (B_AT(0,-1) + B_AT(0,1) + B_AT(-1,0) + B_AT(1,0))
                   - (B_AT(0,-2) + B_AT(0,2) + B_AT(-2,0) + B_AT(2,0)) + 4) / 8;
        int16_t r = B_AT(0,0);
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 1, 1, b, g, r);
    }
}


//------------------------------------------------------------------------------
// Bilinear Algorithm

__global__ void rggb_bilinear(
    const uint8_t* raw,
    ptrdiff_t raw_pitch,
    uint8_t* bgr,
    ptrdiff_t bgr_pitch,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= (width + 1) / 2 || y >= (height + 1) / 2) return;

    // We are looking at a 2x2 block
    const uint8_t* block = reinterpret_cast<const uint8_t*>(raw + (y * raw_pitch + x) * 2);
    uint8_t* bgr_block = reinterpret_cast<uint8_t*>(bgr + (y * bgr_pitch + static_cast<ptrdiff_t>(x) * 3) * 2);

    /*
        RGGB layout:

            R G
            G B

        G at R/B:

            0 G 0
            G x G
            0 G 0

        R at G, R column:

            G R G
            B x B
            G R G

        R at G, B column:

            G B G
            R x R
            G B G

        B at G, B column:

            G B G
            R x R
            G B G

        B at G, R column:

            G R G
            B x B
            G R G

        R at B, B column:

            R G R
            G x G
            R G R

        B at R, R column:

            B G B
            G x G
            B G B
    */

    // Upper left:
    if (x * 2 + 0 < width && y * 2 + 0 < height) {
        B_SET_CENTER(0,0);
        int16_t b = (B_AT(-1,-1) + B_AT(1,1) + B_AT(-1,1) + B_AT(1,-1) + 2) / 4;
        int16_t g = (B_AT(1,0) + B_AT(-1,0) + B_AT(0,1) + B_AT(0,-1) + 2) / 4;
        int16_t r = B_AT(0,0);
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 0, 0, b, g, r);
    }

    // Upper right:
    if (x * 2 + 1 < width && y * 2 + 0 < height) {
        B_SET_CENTER(1,0);
        int16_t b = (B_AT(0,1) + B_AT(0,-1) + 1) / 2;
        int16_t g = B_AT(0,0);
        int16_t r = (B_AT(1,0) + B_AT(-1,0) + 1) / 2;
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 1, 0, b, g, r);
    }

    // Lower left:
    if (x * 2 + 0 < width && y * 2 + 1 < height) {
        B_SET_CENTER(0,1);
        int16_t b = (B_AT(1,0) + B_AT(-1,0) + 1) / 2;
        int16_t g = B_AT(0,0);
        int16_t r = (B_AT(0,1) + B_AT(0,-1) + 1) / 2;
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 0, 1, b, g, r);
    }

    // Lower right:
    if (x * 2 + 1 < width && y * 2 + 1 < height) {
        B_SET_CENTER(1,1);
        int16_t b = B_AT(0,0);
        int16_t g = (B_AT(1,0) + B_AT(-1,0) + B_AT(0,1) + B_AT(0,-1) + 2) / 4;
        int16_t r = (B_AT(-1,-1) + B_AT(1,1) + B_AT(-1,1) + B_AT(1,-1) + 2) / 4;
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 1, 1, b, g, r);
    }
}


__global__ void bggr_bilinear(
    const uint8_t* raw,
    ptrdiff_t raw_pitch,
    uint8_t* bgr,
    ptrdiff_t bgr_pitch,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= (width + 1) / 2 || y >= (height + 1) / 2) return;

    // We are looking at a 2x2 block
    const uint8_t* block = reinterpret_cast<const uint8_t*>(raw + (y * raw_pitch + x) * 2);
    uint8_t* bgr_block = reinterpret_cast<uint8_t*>(bgr + (y * bgr_pitch + static_cast<ptrdiff_t>(x) * 3) * 2);

    /*
        RGGB layout:

            R G
            G B

        G at R/B:

            0 G 0
            G x G
            0 G 0

        R at G, R column:

            G R G
            B x B
            G R G

        R at G, B column:

            G B G
            R x R
            G B G

        B at G, B column:

            G B G
            R x R
            G B G

        B at G, R column:

            G R G
            B x B
            G R G

        R at B, B column:

            R G R
            G x G
            R G R

        B at R, R column:

            B G B
            G x G
            B G B
    */

    // Upper left:
    if (x * 2 + 0 < width && y * 2 + 0 < height) {
        B_SET_CENTER(0,0);
        int16_t b = (B_AT(-1,-1) + B_AT(1,1) + B_AT(-1,1) + B_AT(1,-1) + 2) / 4;
        int16_t g = (B_AT(1,0) + B_AT(-1,0) + B_AT(0,1) + B_AT(0,-1) + 2) / 4;
        int16_t r = B_AT(0,0);
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 0, 0, r, g, b);
    }

    // Upper right:
    if (x * 2 + 1 < width && y * 2 + 0 < height) {
        B_SET_CENTER(1,0);
        int16_t b = (B_AT(0,1) + B_AT(0,-1) + 1) / 2;
        int16_t g = B_AT(0,0);
        int16_t r = (B_AT(1,0) + B_AT(-1,0) + 1) / 2;
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 1, 0, r, g, b);
    }

    // Lower left:
    if (x * 2 + 0 < width && y * 2 + 1 < height) {
        B_SET_CENTER(0,1);
        int16_t b = (B_AT(1,0) + B_AT(-1,0) + 1) / 2;
        int16_t g = B_AT(0,0);
        int16_t r = (B_AT(0,1) + B_AT(0,-1) + 1) / 2;
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 0, 1, r, g, b);
    }

    // Lower right:
    if (x * 2 + 1 < width && y * 2 + 1 < height) {
        B_SET_CENTER(1,1);
        int16_t b = B_AT(0,0);
        int16_t g = (B_AT(1,0) + B_AT(-1,0) + B_AT(0,1) + B_AT(0,-1) + 2) / 4;
        int16_t r = (B_AT(-1,-1) + B_AT(1,1) + B_AT(-1,1) + B_AT(1,-1) + 2) / 4;
        WriteBGRBlockPixel(bgr_block, bgr_pitch, 1, 1, r, g, b);
    }
}

//------------------------------------------------------------------------------
// SoftMenon: posterior-weighted green followed by color-difference reconstruction.
#include "softmenon_green.cuh"

__global__ void rggb_softmenon_g(
    const uint8_t* raw,
    ptrdiff_t raw_pitch,
    uint8_t* bgr,
    ptrdiff_t bgr_pitch,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= (width + 1) / 2 || y >= (height + 1) / 2) return;

    // We are looking at a 2x2 block
    const uint8_t* block = reinterpret_cast<const uint8_t*>(raw + (y * raw_pitch + x) * 2);
    uint8_t* bgr_block = reinterpret_cast<uint8_t*>(bgr + (y * bgr_pitch + static_cast<ptrdiff_t>(x) * 3) * 2);

    /*
        R G
        G B
    */

    // Upper Left:
    if (x * 2 + 0 < width && y * 2 + 0 < height) {
        const uint8_t* P = block;
        uint8_t* bgr = bgr_block;

        const int G_est = softmenon_green::estimate(P, raw_pitch);

        bgr[1] = saturate_cast_int16_to_uint8(G_est);
        bgr[2] = P[0];
    }

    // Lower Right:
    if (x * 2 + 1 < width && y * 2 + 1 < height) {
        const uint8_t* P = block + raw_pitch + 1;
        uint8_t* bgr = bgr_block + bgr_pitch + 3;

        const int G_est = softmenon_green::estimate(P, raw_pitch);

        // Clamp the value
        bgr[0] = P[0];
        bgr[1] = saturate_cast_int16_to_uint8(G_est);
    }

    // Upper Right:
    if (x * 2 + 1 < width && y * 2 + 0 < height) {
        const uint8_t* P = block + 1;
        uint8_t* bgr = bgr_block + 3;

        bgr[1] = P[0];
    }

    // Lower Left:
    if (x * 2 + 0 < width && y * 2 + 1 < height) {
        const uint8_t* P = block + raw_pitch;
        uint8_t* bgr = bgr_block + bgr_pitch;

        bgr[1] = P[0];
    }
}

__global__ void rggb_softmenon_rb(
    uint8_t* bgr,
    ptrdiff_t bgr_pitch,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= (width + 1) / 2 || y >= (height + 1) / 2) return;

    // We are looking at a 2x2 block
    uint8_t* bgr_block = reinterpret_cast<uint8_t*>(bgr + (y * bgr_pitch + static_cast<ptrdiff_t>(x) * 3) * 2);

    /*
        R G
        G B
    */

    // Upper Left:
    if (x * 2 + 0 < width && y * 2 + 0 < height) {
        const int pixel_x = x * 2 + 0;
        const int pixel_y = y * 2 + 0;
        uint8_t* bgr_pixel = bgr_block; // Pointer to the pixel at P0

        // Edge-Directed Interpolation of (B - G) at red pixel
        // Using diagonal neighbors for interpolation

        // Get neighboring blue and green values
        int16_t B_UR = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (1), pixel_y + (-1), 0);       // Blue at (i - 1, j + 1)
        int16_t B_LL = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (-1), pixel_y + (1), 0);       // Blue at (i + 1, j - 1)
        int16_t G_UR = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (1), pixel_y + (-1), 1);       // Green at (i - 1, j + 1)
        int16_t G_LL = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (-1), pixel_y + (1), 1);       // Green at (i + 1, j - 1)

        int16_t B_UL = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (-1), pixel_y + (-1), 0);       // Blue at (i - 1, j - 1)
        int16_t B_LR = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (1), pixel_y + (1), 0);       // Blue at (i + 1, j + 1)
        int16_t G_UL = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (-1), pixel_y + (-1), 1);       // Green at (i - 1, j - 1)
        int16_t G_LR = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (1), pixel_y + (1), 1);       // Green at (i + 1, j + 1)

        // Compute color differences
        int16_t CD_UR = B_UR - G_UR;
        int16_t CD_LL = B_LL - G_LL;
        int16_t CD_UL = B_UL - G_UL;
        int16_t CD_LR = B_LR - G_LR;

        // Horizontal and Vertical estimates of (B - G)
        int16_t CD_h = (CD_UL + CD_LR + 1) >> 1;
        int16_t CD_v = (CD_UR + CD_LL + 1) >> 1;

        // Compute gradients for edge detection
        int16_t Grad_h = abs((CD_UL - CD_LR));
        int16_t Grad_v = abs((CD_UR - CD_LL));

        // Decision based on gradients
        const int16_t CD_best = (Grad_h <= Grad_v) ? CD_h : CD_v;
        const int CD_gap = abs(Grad_h - Grad_v);
        int16_t CD_est = CD_best;
        if (CD_gap <= 16) {
            CD_est = (CD_h + CD_v + 1) >> 1;
        } else if (CD_gap <= 64) {
            CD_est = (2 * CD_best + CD_h + CD_v + 2) >> 2;
        }


        // Estimate Blue value at red pixel
        int16_t G_center = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (0), pixel_y + (0), 1);   // Green at current red pixel
        int16_t B_est = G_center + CD_est;

        // Clamp the value
        bgr_pixel[0] = saturate_cast_int16_to_uint8(B_est);
    }

    // Lower Right
    if (x * 2 + 1 < width && y * 2 + 1 < height) {
        const int pixel_x = x * 2 + 1;
        const int pixel_y = y * 2 + 1;
        uint8_t* bgr_pixel = bgr_block + bgr_pitch + 3; // Pointer to the pixel at P3

        // Get neighboring red and green values
        int16_t R_UL = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (-1), pixel_y + (-1), 2);    // Red at (i - 1, j - 1)
        int16_t R_LR = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (1), pixel_y + (1), 2);    // Red at (i + 1, j + 1)
        int16_t G_UL = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (-1), pixel_y + (-1), 1);    // Green at (i - 1, j - 1)
        int16_t G_LR = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (1), pixel_y + (1), 1);    // Green at (i + 1, j + 1)

        int16_t R_UR = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (1), pixel_y + (-1), 2);    // Red at (i - 1, j + 1)
        int16_t R_LL = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (-1), pixel_y + (1), 2);    // Red at (i + 1, j - 1)
        int16_t G_UR = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (1), pixel_y + (-1), 1);    // Green at (i - 1, j + 1)
        int16_t G_LL = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (-1), pixel_y + (1), 1);    // Green at (i + 1, j - 1)

        // Compute color differences
        int16_t CD_UL = R_UL - G_UL;
        int16_t CD_LR = R_LR - G_LR;
        int16_t CD_UR = R_UR - G_UR;
        int16_t CD_LL = R_LL - G_LL;

        // Horizontal and Vertical estimates of (R - G)
        int16_t CD_h = (CD_UL + CD_LR + 1) >> 1;
        int16_t CD_v = (CD_UR + CD_LL + 1) >> 1;

        // Compute gradients for edge detection
        int16_t Grad_h = abs((CD_UL - CD_LR));
        int16_t Grad_v = abs((CD_UR - CD_LL));

        // Decision based on gradients
        const int16_t CD_best = (Grad_h <= Grad_v) ? CD_h : CD_v;
        const int CD_gap = abs(Grad_h - Grad_v);
        int16_t CD_est = CD_best;
        if (CD_gap <= 16) {
            CD_est = (CD_h + CD_v + 1) >> 1;
        } else if (CD_gap <= 64) {
            CD_est = (2 * CD_best + CD_h + CD_v + 2) >> 2;
        }


        // Estimate Red value at blue pixel
        int16_t G_center = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (0), pixel_y + (0), 1);   // Green at current blue pixel
        int16_t R_est = G_center + CD_est;

        // Clamp the value
        bgr_pixel[2] = saturate_cast_int16_to_uint8(R_est);
    }

    // Lower Left
    if (x * 2 + 0 < width && y * 2 + 1 < height) {
        const int pixel_x = x * 2 + 0;
        const int pixel_y = y * 2 + 1;
        uint8_t* bgr_pixel = bgr_block + bgr_pitch; // Pointer to the pixel at P2

        // Estimate Red at green pixel using bilinear interpolation of (R - G)
        int16_t R_up = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (0), pixel_y + (-1), 2);       // Red at (i - 1, j)
        int16_t R_down = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (0), pixel_y + (1), 2);     // Red at (i + 1, j)
        int16_t G_up = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (0), pixel_y + (-1), 1);
        int16_t G_down = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (0), pixel_y + (1), 1);

        int16_t CD_RU = R_up - G_up;
        int16_t CD_RD = R_down - G_down;

        int16_t CD_R = (CD_RU + CD_RD + 1) >> 1;
        int16_t G_center = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (0), pixel_y + (0), 1);
        int16_t R_est = G_center + CD_R;

        // Estimate Blue at green pixel using bilinear interpolation of (B - G)
        int16_t B_left = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (-1), pixel_y + (0), 0);         // Blue at (i, j - 1)
        int16_t B_right = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (1), pixel_y + (0), 0);        // Blue at (i, j + 1)
        int16_t G_left = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (-1), pixel_y + (0), 1);
        int16_t G_right = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (1), pixel_y + (0), 1);

        int16_t CD_BL = B_left - G_left;
        int16_t CD_BR = B_right - G_right;

        int16_t CD_B = (CD_BL + CD_BR + 1) >> 1;
        int16_t B_est = G_center + CD_B;

        // Clamp the values
        bgr_pixel[2] = saturate_cast_int16_to_uint8(R_est);
        bgr_pixel[0] = saturate_cast_int16_to_uint8(B_est);
    }

    // Upper Right
    if (x * 2 + 1 < width && y * 2 + 0 < height) {
        const int pixel_x = x * 2 + 1;
        const int pixel_y = y * 2 + 0;
        uint8_t* bgr_pixel = bgr_block + 3; // Pointer to the pixel at P1

        // Estimate Red at green pixel using bilinear interpolation of (R - G)
        int16_t R_left = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (-1), pixel_y + (0), 2);       // Red at (i, j - 1)
        int16_t R_right = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (1), pixel_y + (0), 2);      // Red at (i, j + 1)
        int16_t G_left = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (-1), pixel_y + (0), 1);       // Green at (i, j - 1)
        int16_t G_right = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (1), pixel_y + (0), 1);      // Green at (i, j + 1)

        int16_t CD_RL = R_left - G_left;
        int16_t CD_RR = R_right - G_right;

        int16_t CD_R = (CD_RL + CD_RR + 1) >> 1;
        int16_t G_center = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (0), pixel_y + (0), 1);
        int16_t R_est = G_center + CD_R;

        // Estimate Blue at green pixel using bilinear interpolation of (B - G)
        int16_t B_up = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (0), pixel_y + (-1), 0); // Blue at (i - 1, j)
        int16_t B_down = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (0), pixel_y + (1), 0); // Blue at (i + 1, j)
        int16_t G_up = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (0), pixel_y + (-1), 1);
        int16_t G_down = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (0), pixel_y + (1), 1);

        int16_t CD_BU = B_up - G_up;
        int16_t CD_BD = B_down - G_down;

        int16_t CD_B = (CD_BU + CD_BD + 1) >> 1;
        int16_t B_est = G_center + CD_B;

        // Clamp the values
        bgr_pixel[2] = saturate_cast_int16_to_uint8(R_est);
        bgr_pixel[0] = saturate_cast_int16_to_uint8(B_est);
    }
}

__global__ void bggr_softmenon_g(
    const uint8_t* raw,
    ptrdiff_t raw_pitch,
    uint8_t* bgr,
    ptrdiff_t bgr_pitch,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Column index in 2x2 blocks
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Row index in 2x2 blocks

    if (x >= (width + 1) / 2 || y >= (height + 1) / 2) return;

    // Calculate the starting index of the 2x2 block
    const uint8_t* block = raw + (y * raw_pitch + x) * 2;
    uint8_t* bgr_block = bgr + (y * bgr_pitch + static_cast<ptrdiff_t>(x) * 3) * 2;

    /*
        BGGR pattern in a 2x2 block:

            B G
            G R
    */

    // Upper Left (P0): B pixel
    if (x * 2 + 0 < width && y * 2 + 0 < height) {
        const uint8_t* P = block;
        uint8_t* bgr = bgr_block;

        const int G_est = softmenon_green::estimate(P, raw_pitch);

        // Clamp the value
        bgr[0] = P[0]; // B value
        bgr[1] = saturate_cast_int16_to_uint8(G_est);
    }

    // Upper Right (P1): G pixel
    if (x * 2 + 1 < width && y * 2 + 0 < height) {
        const uint8_t* P = block + 1;
        uint8_t* bgr = bgr_block + 3;

        // G pixel, green value is known
        bgr[1] = P[0];
    }

    // Lower Left (P2): G pixel
    if (x * 2 + 0 < width && y * 2 + 1 < height) {
        const uint8_t* P = block + raw_pitch;
        uint8_t* bgr = bgr_block + bgr_pitch;

        // G pixel, green value is known
        bgr[1] = P[0];
    }

    // Lower Right (P3): R pixel
    if (x * 2 + 1 < width && y * 2 + 1 < height) {
        const uint8_t* P = block + raw_pitch + 1;
        uint8_t* bgr = bgr_block + bgr_pitch + 3;

        const int G_est = softmenon_green::estimate(P, raw_pitch);

        // Clamp the value
        bgr[1] = saturate_cast_int16_to_uint8(G_est);
        bgr[2] = P[0]; // R value
    }
}

__global__ void bggr_softmenon_rb(
    uint8_t* bgr,
    ptrdiff_t bgr_pitch,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Column index in 2x2 blocks
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Row index in 2x2 blocks

    if (x >= (width + 1) / 2 || y >= (height + 1) / 2) return;

    // Calculate the starting index of the 2x2 block
    uint8_t* bgr_block = bgr + (y * bgr_pitch + static_cast<ptrdiff_t>(x) * 3) * 2;

    /*
        BGGR pattern in a 2x2 block:

        Positions:
        P0: Upper Left (B pixel)
        P1: Upper Right (G pixel)
        P2: Lower Left (G pixel)
        P3: Lower Right (R pixel)
    */

    // Upper Left (P0): B pixel
    if (x * 2 + 0 < width && y * 2 + 0 < height) {
        const int pixel_x = x * 2 + 0;
        const int pixel_y = y * 2 + 0;
        uint8_t* bgr_pixel = bgr_block; // Pointer to the pixel at P0

        // Estimate Red at Blue pixel (similar to estimating Blue at Red pixel in RGGB)
        // Get neighboring red and green values
        int16_t R_UR = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (1), pixel_y + (-1), 2);    // Red at (i - 1, j + 1)
        int16_t R_LL = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (-1), pixel_y + (1), 2);    // Red at (i + 1, j - 1)
        int16_t G_UR = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (1), pixel_y + (-1), 1);    // Green at (i - 1, j + 1)
        int16_t G_LL = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (-1), pixel_y + (1), 1);    // Green at (i + 1, j - 1)

        int16_t R_UL = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (-1), pixel_y + (-1), 2);    // Red at (i - 1, j - 1)
        int16_t R_LR = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (1), pixel_y + (1), 2);    // Red at (i + 1, j + 1)
        int16_t G_UL = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (-1), pixel_y + (-1), 1);    // Green at (i - 1, j - 1)
        int16_t G_LR = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (1), pixel_y + (1), 1);    // Green at (i + 1, j + 1)

        // Compute color differences
        int16_t CD_UR = R_UR - G_UR;
        int16_t CD_LL = R_LL - G_LL;
        int16_t CD_UL = R_UL - G_UL;
        int16_t CD_LR = R_LR - G_LR;

        // Horizontal and Vertical estimates of (R - G)
        int16_t CD_h = (CD_UL + CD_LR + 1) >> 1;
        int16_t CD_v = (CD_UR + CD_LL + 1) >> 1;

        // Compute gradients for edge detection
        int16_t Grad_h = abs(CD_UL - CD_LR);
        int16_t Grad_v = abs(CD_UR - CD_LL);

        // Decision based on gradients
        const int16_t CD_best = (Grad_h <= Grad_v) ? CD_h : CD_v;
        const int CD_gap = abs(Grad_h - Grad_v);
        int16_t CD_est = CD_best;
        if (CD_gap <= 16) {
            CD_est = (CD_h + CD_v + 1) >> 1;
        } else if (CD_gap <= 64) {
            CD_est = (2 * CD_best + CD_h + CD_v + 2) >> 2;
        }


        // Estimate Red value at blue pixel
        int16_t G_center = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (0), pixel_y + (0), 1);   // Green at current blue pixel
        int16_t R_est = G_center + CD_est;

        // Clamp the value
        bgr_pixel[2] = saturate_cast_int16_to_uint8(R_est);
    }

    // Lower Right (P3): R pixel
    if (x * 2 + 1 < width && y * 2 + 1 < height) {
        const int pixel_x = x * 2 + 1;
        const int pixel_y = y * 2 + 1;
        uint8_t* bgr_pixel = bgr_block + bgr_pitch + 3; // Pointer to the pixel at P3

        // Estimate Blue at Red pixel (similar to estimating Red at Blue pixel in RGGB)
        // Get neighboring blue and green values
        int16_t B_UR = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (1), pixel_y + (-1), 0);    // Blue at (i - 1, j + 1)
        int16_t B_LL = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (-1), pixel_y + (1), 0);    // Blue at (i + 1, j - 1)
        int16_t G_UR = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (1), pixel_y + (-1), 1);    // Green at (i - 1, j + 1)
        int16_t G_LL = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (-1), pixel_y + (1), 1);    // Green at (i + 1, j - 1)

        int16_t B_UL = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (-1), pixel_y + (-1), 0);    // Blue at (i - 1, j - 1)
        int16_t B_LR = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (1), pixel_y + (1), 0);    // Blue at (i + 1, j + 1)
        int16_t G_UL = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (-1), pixel_y + (-1), 1);    // Green at (i - 1, j - 1)
        int16_t G_LR = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (1), pixel_y + (1), 1);    // Green at (i + 1, j + 1)

        // Compute color differences
        int16_t CD_UR = B_UR - G_UR;
        int16_t CD_LL = B_LL - G_LL;
        int16_t CD_UL = B_UL - G_UL;
        int16_t CD_LR = B_LR - G_LR;

        // Horizontal and Vertical estimates of (B - G)
        int16_t CD_h = (CD_UL + CD_LR + 1) >> 1;
        int16_t CD_v = (CD_UR + CD_LL + 1) >> 1;

        // Compute gradients for edge detection
        int16_t Grad_h = abs(CD_UL - CD_LR);
        int16_t Grad_v = abs(CD_UR - CD_LL);

        // Decision based on gradients
        const int16_t CD_best = (Grad_h <= Grad_v) ? CD_h : CD_v;
        const int CD_gap = abs(Grad_h - Grad_v);
        int16_t CD_est = CD_best;
        if (CD_gap <= 16) {
            CD_est = (CD_h + CD_v + 1) >> 1;
        } else if (CD_gap <= 64) {
            CD_est = (2 * CD_best + CD_h + CD_v + 2) >> 2;
        }


        // Estimate Blue value at red pixel
        int16_t G_center = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (0), pixel_y + (0), 1);   // Green at current red pixel
        int16_t B_est = G_center + CD_est;

        // Clamp the value
        bgr_pixel[0] = saturate_cast_int16_to_uint8(B_est);
    }

    // Upper Right (P1): G pixel
    if (x * 2 + 1 < width && y * 2 + 0 < height) {
        const int pixel_x = x * 2 + 1;
        const int pixel_y = y * 2 + 0;
        uint8_t* bgr_pixel = bgr_block + 3; // Pointer to the pixel at P1

        // Estimate Red at green pixel using bilinear interpolation of (R - G)
        int16_t R_up = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (0), pixel_y + (-1), 2);       // Red at (i - 1, j)
        int16_t R_down = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (0), pixel_y + (1), 2);     // Red at (i + 1, j)
        int16_t G_up = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (0), pixel_y + (-1), 1);
        int16_t G_down = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (0), pixel_y + (1), 1);

        int16_t CD_RU = R_up - G_up;
        int16_t CD_RD = R_down - G_down;

        int16_t CD_R = (CD_RU + CD_RD + 1) >> 1;
        int16_t G_center = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (0), pixel_y + (0), 1);
        int16_t R_est = G_center + CD_R;

        // Estimate Blue at green pixel using bilinear interpolation of (B - G)
        int16_t B_left = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (-1), pixel_y + (0), 0);         // Blue at (i, j - 1)
        int16_t B_right = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (1), pixel_y + (0), 0);        // Blue at (i, j + 1)
        int16_t G_left = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (-1), pixel_y + (0), 1);
        int16_t G_right = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (1), pixel_y + (0), 1);

        int16_t CD_BL = B_left - G_left;
        int16_t CD_BR = B_right - G_right;

        int16_t CD_B = (CD_BL + CD_BR + 1) >> 1;
        int16_t B_est = G_center + CD_B;

        // Clamp the values
        bgr_pixel[2] = saturate_cast_int16_to_uint8(R_est);
        bgr_pixel[0] = saturate_cast_int16_to_uint8(B_est);
    }

    // Lower Left (P2): G pixel
    if (x * 2 + 0 < width && y * 2 + 1 < height) {
        const int pixel_x = x * 2 + 0;
        const int pixel_y = y * 2 + 1;
        uint8_t* bgr_pixel = bgr_block + bgr_pitch; // Pointer to the pixel at P2

        // Estimate Red at green pixel using bilinear interpolation of (R - G)
        int16_t R_left = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (-1), pixel_y + (0), 2);       // Red at (i, j - 1)
        int16_t R_right = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (1), pixel_y + (0), 2);      // Red at (i, j + 1)
        int16_t G_left = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (-1), pixel_y + (0), 1);       // Green at (i, j - 1)
        int16_t G_right = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (1), pixel_y + (0), 1);      // Green at (i, j + 1)

        int16_t CD_RL = R_left - G_left;
        int16_t CD_RR = R_right - G_right;

        int16_t CD_R = (CD_RL + CD_RR + 1) >> 1;
        int16_t G_center = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (0), pixel_y + (0), 1);
        int16_t R_est = G_center + CD_R;

        // Estimate Blue at green pixel using bilinear interpolation of (B - G)
        int16_t B_up = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (0), pixel_y + (-1), 0); // Blue at (i - 1, j)
        int16_t B_down = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (0), pixel_y + (1), 0); // Blue at (i + 1, j)
        int16_t G_up = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (0), pixel_y + (-1), 1);
        int16_t G_down = read_bgr_reflected(bgr, bgr_pitch, width, height, pixel_x + (0), pixel_y + (1), 1);

        int16_t CD_BU = B_up - G_up;
        int16_t CD_BD = B_down - G_down;

        int16_t CD_B = (CD_BU + CD_BD + 1) >> 1;
        int16_t B_est = G_center + CD_B;

        // Clamp the values
        bgr_pixel[2] = saturate_cast_int16_to_uint8(R_est);
        bgr_pixel[0] = saturate_cast_int16_to_uint8(B_est);
    }
}
