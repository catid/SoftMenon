/**
 * @file cpu_kernel.cpp
 * @brief CPU-based kernel functions for image padding and Bayer pattern demosaicing.
 */

#include "cpu_kernel.hpp"
#include "chroma_median.hpp"

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <cstddef>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define CPU_SOFT_AVX2 1
#define SOFT_INLINE __attribute__((target("avx2"),always_inline)) inline
#define SOFT512_INLINE __attribute__((target("avx512f,avx512bw,avx512vbmi"),always_inline)) inline
#else
#define CPU_SOFT_AVX2 0
#endif

#define ENABLE_CLOSE_AVERAGING

namespace {
// Integer division truncation for negative numerators cannot affect a clipped
// uint8 result: both truncation and floor clip to zero. Clamp before shifting
// to preserve the CUDA formulas exactly without signed-division corrections.
inline uint8_t scaled_u8(int numerator,int shift) {
    return static_cast<uint8_t>(std::min(255,std::max(0,numerator)>>shift));
}
}

void bilinear_cpu_rows(const uint8_t* raw,size_t raw_pitch,uint8_t* bgr,
    size_t bgr_pitch,int width,int height,bool rggb,int begin_y,int end_y) {
    const ptrdiff_t pitch=static_cast<ptrdiff_t>(raw_pitch);
    begin_y=std::max(0,begin_y); end_y=std::min(height,end_y);
    for (int y=begin_y;y<end_y;++y) {
        const uint8_t* input=raw+static_cast<size_t>(y)*raw_pitch;
        uint8_t* output=bgr+static_cast<size_t>(y)*bgr_pitch;
        const bool red_row=((y&1)==0)==rggb;
        for (int x=0;x<width;++x) {
            const uint8_t* p=input+x;
            int b,g,r;
            if ((x&1)==(y&1)) {
                const int opposite=(p[-pitch-1]+p[-pitch+1]+p[pitch-1]+p[pitch+1]+2)>>2;
                g=(p[-1]+p[1]+p[-pitch]+p[pitch]+2)>>2;
                r=red_row ? p[0] : opposite;
                b=red_row ? opposite : p[0];
            } else {
                const int horizontal=(p[-1]+p[1]+1)>>1;
                const int vertical=(p[-pitch]+p[pitch]+1)>>1;
                g=p[0]; r=red_row ? horizontal : vertical;
                b=red_row ? vertical : horizontal;
            }
            output[3*x]=static_cast<uint8_t>(b);
            output[3*x+1]=static_cast<uint8_t>(g);
            output[3*x+2]=static_cast<uint8_t>(r);
        }
    }
}

void malvar2004_cpu_rows(const uint8_t* raw,size_t raw_pitch,uint8_t* bgr,
    size_t bgr_pitch,int width,int height,bool rggb,int begin_y,int end_y) {
    const ptrdiff_t pitch=static_cast<ptrdiff_t>(raw_pitch);
    begin_y=std::max(0,begin_y); end_y=std::min(height,end_y);
    for (int y=begin_y;y<end_y;++y) {
        const uint8_t* input=raw+static_cast<size_t>(y)*raw_pitch;
        uint8_t* output=bgr+static_cast<size_t>(y)*bgr_pitch;
        const bool red_row=((y&1)==0)==rggb;
        for (int x=0;x<width;++x) {
            const uint8_t* p=input+x;
            const int center=p[0];
            const int horizontal=p[-1]+p[1],vertical=p[-pitch]+p[pitch];
            const int far_horizontal=p[-2]+p[2],far_vertical=p[-2*pitch]+p[2*pitch];
            const int diagonals=p[-pitch-1]+p[-pitch+1]+p[pitch-1]+p[pitch+1];
            uint8_t b,g,r;
            if ((x&1)==(y&1)) {
                g=scaled_u8(4*center+2*(horizontal+vertical)-far_horizontal-far_vertical+4,3);
                const uint8_t opposite=scaled_u8(12*center+4*diagonals-3*(far_horizontal+far_vertical)+8,4);
                r=red_row ? static_cast<uint8_t>(center) : opposite;
                b=red_row ? opposite : static_cast<uint8_t>(center);
            } else {
                const uint8_t along_horizontal=scaled_u8(10*center+8*horizontal-2*(diagonals+far_horizontal)+far_vertical+8,4);
                const uint8_t along_vertical=scaled_u8(10*center+8*vertical-2*(diagonals+far_vertical)+far_horizontal+8,4);
                g=static_cast<uint8_t>(center);
                r=red_row ? along_horizontal : along_vertical;
                b=red_row ? along_vertical : along_horizontal;
            }
            output[3*x]=b; output[3*x+1]=g; output[3*x+2]=r;
        }
    }
}

/**
 * @brief Pads the top and bottom edges of an image by replicating the first and last rows.
 *
 * This function copies the first `width` pixels of the original image to the top padding
 * and the last `width` pixels to the bottom padding. The padding size is specified by `pad`.
 *
 * @param data Pointer to the image data buffer.
 * @param width Width of the original image in pixels.
 * @param height Height of the original image in pixels.
 * @param pitch Number of bytes per row in the image buffer (must be at least `width + 2 * pad`).
 * @param pad Number of padding rows to add to both the top and bottom.
 *
 * @pre `data` must not be `nullptr`.
 * @pre `width` and `height` must be positive.
 * @pre `pitch` must be at least `width + 2 * pad`.
 * @pre `pad` must be positive.
 *
 * @note The function assumes that the `data` buffer has been allocated with enough space
 *       to accommodate the additional padding rows.
 */
void padTopBottomEdges(uint8_t* data, int width, int height, int pitch, int pad) {
    // Validate input parameters
    assert(data != nullptr && "Data pointer must not be null.");
    assert(width > 0 && "Image width must be positive.");
    assert(height > 0 && "Image height must be positive.");
    assert(pitch >= (width + 2 * pad) && "Pitch must be at least width + 2 * pad.");
    assert(pad > 0 && "Padding size must be positive.");

    // Calculate pointers to the first and last rows of the original image
    uint8_t* original_first_row = data + pad * pitch + pad;
    uint8_t* original_last_row  = data + (pad + height - 1) * pitch + pad;

    // Pad the top edges by replicating the first row of the original image
    for (int y = 0; y < pad; ++y) {
        uint8_t* dest_top = data + y * pitch + pad;
        memcpy(dest_top, original_first_row, width * sizeof(uint8_t));
    }

    // Pad the bottom edges by replicating the last row of the original image
    for (int y = 0; y < pad; ++y) {
        uint8_t* dest_bottom = data + (pad + height + y) * pitch + pad;
        memcpy(dest_bottom, original_last_row, width * sizeof(uint8_t));
    }
}

/**
 * @brief Pads the left and right edges of an image by replicating the first and last columns.
 *
 * This function fills the left padding with the first column of the original image and the
 * right padding with the last column. The padding size is specified by `pad`.
 *
 * @param data Pointer to the image data buffer.
 * @param width Width of the original image in pixels.
 * @param height Height of the original image in pixels.
 * @param pitch Number of bytes per row in the image buffer (must be at least `width + 2 * pad`).
 * @param pad Number of padding columns to add to both the left and right.
 *
 * @pre `data` must not be `nullptr`.
 * @pre `width` and `height` must be positive.
 * @pre `pitch` must be at least `width + 2 * pad`.
 * @pre `pad` must be positive.
 *
 * @note The function assumes that the `data` buffer has been allocated with enough space
 *       to accommodate the additional padding columns.
 */
void padLeftRightEdges(uint8_t* data, int width, int height, int pitch, int pad) {
    // Validate input parameters
    assert(data != nullptr && "Data pointer must not be null.");
    assert(width > 0 && "Image width must be positive.");
    assert(height > 0 && "Image height must be positive.");
    assert(pitch >= (width + 2 * pad) && "Pitch must be at least width + 2 * pad.");
    assert(pad > 0 && "Padding size must be positive.");

    for (int y = 0; y < height; ++y) {
        uint8_t* original_row = data + (pad + y) * pitch + pad;
        uint8_t* dest_left  = data + (pad + y) * pitch;
        uint8_t* dest_right = data + (pad + y) * pitch + pad + width;

        uint8_t first_pixel = original_row[0];
        uint8_t last_pixel  = original_row[width - 1];

        // Use memset-like operations for padding
        std::memset(dest_left, first_pixel, pad);
        std::memset(dest_right, last_pixel, pad);
    }
}

/**
 * @brief Clamps a 16-bit signed integer to an 8-bit unsigned integer range [0, 255].
 *
 * This inline helper function ensures that the input value does not exceed the
 * bounds of an 8-bit unsigned integer.
 *
 * @param x The 16-bit signed integer to clamp.
 * @return The clamped 8-bit unsigned integer.
 */
inline uint8_t saturate_cast_int16_to_uint8(int16_t x) {
    if (x < 0) return 0;
    if (x > 255) return 255;
    return static_cast<uint8_t>(x);
}

/**
 * @brief Estimates and fills the Green channel in a BGGR Bayer pattern image.
 *
 * This retained legacy core computes local horizontal and vertical classifiers.
 * SoftDecision selects the frozen inverse-score blend instead of the original
 * thresholded direction decision. Neither mode is the paper's full DDFAPD.
 *
 * @param raw Pointer to the raw Bayer data buffer.
 * @param raw_pitch Number of bytes per row in the raw Bayer data.
 * @param bgr Pointer to BGR output (green and measured colors are filled).
 * @param bgr_pitch Number of bytes per row in the BGR data buffer.
 * @param width Width of the image in pixels (must be even).
 * @param height Height of the image in pixels (must be even).
 *
 * @pre `raw` and `bgr` must not be `nullptr`.
 * @pre `width` and `height` must be positive and even.
 * @pre `raw_pitch` and `bgr_pitch` must be sufficient to hold the respective image data.
 *
 * @note Missing red and blue values are filled in a separate pass.
 */
template <bool SoftDecision>
static void bggr_legacy_g_impl(
    const uint8_t* raw,
    int raw_pitch,
    uint8_t* bgr,
    int bgr_pitch,
    int width,
    int height)
{
    // Validate input parameters
    assert(raw  != nullptr && "Raw data pointer must not be null.");
    assert(bgr   != nullptr && "BGR data pointer must not be null.");
    assert(width > 0 && "Image width must be positive.");
    assert(height > 0 && "Image height must be positive.");
    assert(width % 2 == 0 && "Image width must be even.");
    assert(height % 2 == 0 && "Image height must be even.");
    assert(raw_pitch >= width && "Raw pitch must be at least equal to image width.");
    assert(bgr_pitch >= width * 3 && "BGR pitch must be at least equal to image width * 3.");

    // Iterate over the image in 2x2 blocks
    for(int y = 0; y < height; y += 2) {
        for(int x = 0; x < width; x += 2) {
            // Calculate the starting index of the 2x2 block in raw and BGR buffers
            const uint8_t* block = raw + y * raw_pitch + x;
            uint8_t* bgr_block   = bgr + y * bgr_pitch + x * 3;

            // --- Upper Left Pixel (P0): Blue Pixel ---
            {
                const uint8_t* P = block;
                uint8_t* bgr_p0    = bgr_block;

                // Initialize estimations for Green channel
                int16_t G_h = 0; // Horizontal estimation
                int16_t G_v = 0; // Vertical estimation

                // Horizontal estimation
                // Ensure that we do not access out-of-bounds memory
                // This assumes that the image has been appropriately padded
                int16_t G_left  = static_cast<int16_t>(P[-1]); // G at (y, x-1)
                int16_t G_right = static_cast<int16_t>(P[1]);  // G at (y, x+1)
                int16_t B_center = static_cast<int16_t>(P[0]); // B at (y, x)
                int16_t B_left  = static_cast<int16_t>(P[-2]); // B at (y, x-2)
                int16_t B_right = static_cast<int16_t>(P[2]);  // B at (y, x+2)

                // Compute horizontal Green estimation
                G_h = ((G_left + G_right + 1) >> 1) + ((2 * B_center - B_left - B_right + 2) >> 2);

                // Vertical estimation
                int16_t G_up    = static_cast<int16_t>(P[-raw_pitch]);       // G at (y-1, x)
                int16_t G_down  = static_cast<int16_t>(P[raw_pitch]);        // G at (y+1, x)
                int16_t B_up    = static_cast<int16_t>(P[-2 * raw_pitch]);   // B at (y-2, x)
                int16_t B_down  = static_cast<int16_t>(P[2 * raw_pitch]);    // B at (y+2, x)

                // Compute vertical Green estimation
                G_v = ((G_up + G_down + 1) >> 1) + ((2 * B_center - B_up - B_down + 2) >> 2);

                // Compute classifiers S_h and S_v for decision making
                int16_t C_center_h = B_center - G_h;
                int16_t C_left      = B_left - G_left;
                int16_t C_right     = B_right - G_right;
                int16_t S_h         = std::abs(C_center_h - C_left) + std::abs(C_center_h - C_right);

                int16_t C_center_v = B_center - G_v;
                int16_t C_up        = B_up - G_up;
                int16_t C_down      = B_down - G_down;
                int16_t S_v         = std::abs(C_center_v - C_up) + std::abs(C_center_v - C_down);

                // Decision based on classifiers
                int16_t G_est;
                if constexpr (SoftDecision) {
                    const int denominator=S_h+S_v+2;
                    const int numerator=(S_v+1)*G_h+(S_h+1)*G_v;
                    int quotient=numerator/denominator;
                    int remainder=numerator%denominator;
                    if (remainder<0) { --quotient; remainder+=denominator; }
                    G_est=static_cast<int16_t>(quotient+(2*remainder>=denominator));
                } else {
                    G_est=(S_h<=S_v) ? G_h : G_v;
#ifdef ENABLE_CLOSE_AVERAGING
                    if (std::abs(S_h-S_v)<=29)
                        G_est=(G_h+G_v+1)>>1;
#endif
                }

                // Assign Blue and estimated Green values to the BGR buffer
                bgr_p0[0] = P[0]; // Blue channel
                bgr_p0[1] = saturate_cast_int16_to_uint8(G_est); // Green channel
                // Red channel will be filled in a separate function
            }

            // --- Upper Right Pixel (P1): Green Pixel ---
            {
                const uint8_t* P1 = block + 1;
                uint8_t* bgr_p1    = bgr_block + 3;

                // Green channel is directly available from the raw data
                bgr_p1[1] = P1[0];
                // Blue and Red channels will be filled in separate functions
            }

            // --- Lower Left Pixel (P2): Green Pixel ---
            {
                const uint8_t* P2 = block + raw_pitch;
                uint8_t* bgr_p2    = bgr_block + bgr_pitch;

                // Green channel is directly available from the raw data
                bgr_p2[1] = P2[0];
                // Blue and Red channels will be filled in separate functions
            }

            // --- Lower Right Pixel (P3): Red Pixel ---
            {
                const uint8_t* P3 = block + raw_pitch + 1;
                uint8_t* bgr_p3    = bgr_block + bgr_pitch + 3;

                // Initialize estimations for Green channel
                int16_t G_h = 0; // Horizontal estimation
                int16_t G_v = 0; // Vertical estimation

                // Horizontal estimation
                int16_t G_left  = static_cast<int16_t>(P3[-1]); // G at (y+1, x)
                int16_t G_right = static_cast<int16_t>(P3[1]);  // G at (y+1, x+2)
                int16_t R_center = static_cast<int16_t>(P3[0]); // R at (y+1, x+1)
                int16_t R_left  = static_cast<int16_t>(P3[-2]); // R at (y+1, x-1)
                int16_t R_right = static_cast<int16_t>(P3[2]);  // R at (y+1, x+3)

                // Compute horizontal Green estimation
                G_h = ((G_left + G_right + 1) >> 1) + ((2 * R_center - R_left - R_right + 2) >> 2);

                // Vertical estimation
                int16_t G_up    = static_cast<int16_t>(P3[-raw_pitch]);       // G at (y, x+1)
                int16_t G_down  = static_cast<int16_t>(P3[raw_pitch]);        // G at (y+2, x+1)
                int16_t R_up    = static_cast<int16_t>(P3[-2 * raw_pitch]);   // R at (y-1, x+1)
                int16_t R_down  = static_cast<int16_t>(P3[2 * raw_pitch]);    // R at (y+3, x+1)

                // Compute vertical Green estimation
                G_v = ((G_up + G_down + 1) >> 1) + ((2 * R_center - R_up - R_down + 2) >> 2);

                // Compute classifiers S_h and S_v for decision making
                int16_t C_center_h = R_center - G_h;
                int16_t C_left      = R_left - G_left;
                int16_t C_right     = R_right - G_right;
                int16_t S_h         = std::abs(C_center_h - C_left) + std::abs(C_center_h - C_right);

                int16_t C_center_v = R_center - G_v;
                int16_t C_up        = R_up - G_up;
                int16_t C_down      = R_down - G_down;
                int16_t S_v         = std::abs(C_center_v - C_up) + std::abs(C_center_v - C_down);

                // Decision based on classifiers
                int16_t G_est_p3;
                if constexpr (SoftDecision) {
                    const int denominator=S_h+S_v+2;
                    const int numerator=(S_v+1)*G_h+(S_h+1)*G_v;
                    int quotient=numerator/denominator;
                    int remainder=numerator%denominator;
                    if (remainder<0) { --quotient; remainder+=denominator; }
                    G_est_p3=static_cast<int16_t>(quotient+(2*remainder>=denominator));
                } else {
                    G_est_p3=(S_h<=S_v) ? G_h : G_v;
#ifdef ENABLE_CLOSE_AVERAGING
                    if (std::abs(S_h-S_v)<=27)
                        G_est_p3=(G_h+G_v+1)>>1;
#endif
                }

                // Assign Red and estimated Green values to the BGR buffer
                bgr_p3[2] = P3[0]; // Red channel
                bgr_p3[1] = saturate_cast_int16_to_uint8(G_est_p3); // Green channel
                // Blue channel will be filled in separate functions
            }
        }
    }
}

// SIMD implementation of the frozen SoftMenon green blend.
#if CPU_SOFT_AVX2
namespace {
SOFT_INLINE __m256i soft_load16(const uint8_t* p) {
    return _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

SOFT_INLINE __m256i soft_weighted8(__m128i gh16,__m128i gv16,__m128i sh16,__m128i sv16) {
    const __m256i one=_mm256_set1_epi32(1);
    const __m256i gh=_mm256_cvtepi16_epi32(gh16),gv=_mm256_cvtepi16_epi32(gv16);
    const __m256i sh=_mm256_add_epi32(_mm256_cvtepi16_epi32(sh16),one);
    const __m256i sv=_mm256_add_epi32(_mm256_cvtepi16_epi32(sv16),one);
    const __m256i denominator=_mm256_add_epi32(sh,sv);
    const __m256i numerator=_mm256_add_epi32(_mm256_mullo_epi32(sv,gh),_mm256_mullo_epi32(sh,gv));
    const __m256i rounded=_mm256_max_epi32(_mm256_add_epi32(numerator,_mm256_srli_epi32(denominator,1)),
                                          _mm256_setzero_si256());
    // Exact integer quotient in this bounded domain: denominator <=2554,
    // 0<=rounded/denominator<384, and rounded<2^20 is exactly representable.
    // Distance to an integer boundary is >=1/2554, over 12 float32 ULPs;
    // exact integer quotients are representable. Negative results clip to 0.
    // Use actual division, never an approximate reciprocal.
    const __m256 quotient=_mm256_div_ps(_mm256_cvtepi32_ps(rounded),_mm256_cvtepi32_ps(denominator));
    return _mm256_cvttps_epi32(quotient);
}

SOFT_INLINE __m128i soft_green16(const uint8_t* p,ptrdiff_t pitch) {
    const __m256i one=_mm256_set1_epi16(1),two=_mm256_set1_epi16(2);
    const __m256i center=soft_load16(p),left=soft_load16(p-1),right=soft_load16(p+1);
    const __m256i up=soft_load16(p-pitch),down=soft_load16(p+pitch);
    const __m256i left2=soft_load16(p-2),right2=soft_load16(p+2);
    const __m256i up2=soft_load16(p-2*pitch),down2=soft_load16(p+2*pitch);
    const __m256i twice=_mm256_slli_epi16(center,1);
    const __m256i gh=_mm256_add_epi16(_mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(left,right),one),1),
        _mm256_srai_epi16(_mm256_add_epi16(_mm256_sub_epi16(twice,_mm256_add_epi16(left2,right2)),two),2));
    const __m256i gv=_mm256_add_epi16(_mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(up,down),one),1),
        _mm256_srai_epi16(_mm256_add_epi16(_mm256_sub_epi16(twice,_mm256_add_epi16(up2,down2)),two),2));
    const __m256i ch=_mm256_sub_epi16(center,gh),cv=_mm256_sub_epi16(center,gv);
    const __m256i sh=_mm256_add_epi16(_mm256_abs_epi16(_mm256_sub_epi16(ch,_mm256_sub_epi16(left2,left))),
                                    _mm256_abs_epi16(_mm256_sub_epi16(ch,_mm256_sub_epi16(right2,right))));
    const __m256i sv=_mm256_add_epi16(_mm256_abs_epi16(_mm256_sub_epi16(cv,_mm256_sub_epi16(up2,up))),
                                    _mm256_abs_epi16(_mm256_sub_epi16(cv,_mm256_sub_epi16(down2,down))));
    const __m256i lo=soft_weighted8(_mm256_castsi256_si128(gh),_mm256_castsi256_si128(gv),
                                   _mm256_castsi256_si128(sh),_mm256_castsi256_si128(sv));
    const __m256i hi=soft_weighted8(_mm256_extracti128_si256(gh,1),_mm256_extracti128_si256(gv,1),
                                   _mm256_extracti128_si256(sh,1),_mm256_extracti128_si256(sv,1));
    const __m256i words=_mm256_permute4x64_epi64(_mm256_packs_epi32(lo,hi),0xd8);
    return _mm_packus_epi16(_mm256_castsi256_si128(words),_mm256_extracti128_si256(words,1));
}

SOFT_INLINE void soft_store16(uint8_t* output,__m128i blue,__m128i green,__m128i red) {
    const __m128i b0=_mm_shuffle_epi8(blue,_mm_setr_epi8(0,-128,-128,1,-128,-128,2,-128,-128,3,-128,-128,4,-128,-128,5));
    const __m128i g0=_mm_shuffle_epi8(green,_mm_setr_epi8(-128,0,-128,-128,1,-128,-128,2,-128,-128,3,-128,-128,4,-128,-128));
    const __m128i r0=_mm_shuffle_epi8(red,_mm_setr_epi8(-128,-128,0,-128,-128,1,-128,-128,2,-128,-128,3,-128,-128,4,-128));
    const __m128i b1=_mm_shuffle_epi8(blue,_mm_setr_epi8(-128,-128,6,-128,-128,7,-128,-128,8,-128,-128,9,-128,-128,10,-128));
    const __m128i g1=_mm_shuffle_epi8(green,_mm_setr_epi8(5,-128,-128,6,-128,-128,7,-128,-128,8,-128,-128,9,-128,-128,10));
    const __m128i r1=_mm_shuffle_epi8(red,_mm_setr_epi8(-128,5,-128,-128,6,-128,-128,7,-128,-128,8,-128,-128,9,-128,-128));
    const __m128i b2=_mm_shuffle_epi8(blue,_mm_setr_epi8(-128,11,-128,-128,12,-128,-128,13,-128,-128,14,-128,-128,15,-128,-128));
    const __m128i g2=_mm_shuffle_epi8(green,_mm_setr_epi8(-128,-128,11,-128,-128,12,-128,-128,13,-128,-128,14,-128,-128,15,-128));
    const __m128i r2=_mm_shuffle_epi8(red,_mm_setr_epi8(10,-128,-128,11,-128,-128,12,-128,-128,13,-128,-128,14,-128,-128,15));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output),_mm_or_si128(b0,_mm_or_si128(g0,r0)));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output+16),_mm_or_si128(b1,_mm_or_si128(g1,r1)));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output+32),_mm_or_si128(b2,_mm_or_si128(g2,r2)));
}

__attribute__((target("avx2"))) void soft_green_avx2(const uint8_t* raw,int raw_pitch,
        uint8_t* bgr,int bgr_pitch,int width,int height) {
    const __m128i even=_mm_setr_epi8(-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0);
    const __m128i odd=_mm_xor_si128(even,_mm_set1_epi8(-1)),zero=_mm_setzero_si128();
    const int vector_width=width&~15;
    for (int y=0;y<height;y+=2) {
        for (int dy=0;dy<2;++dy) {
            const uint8_t* input=raw+static_cast<size_t>(y+dy)*raw_pitch;
            uint8_t* output=bgr+static_cast<size_t>(y+dy)*bgr_pitch;
            for (int x=0;x<vector_width;x+=16) {
                const __m128i original=_mm_loadu_si128(reinterpret_cast<const __m128i*>(input+x));
                const __m128i green=_mm_blendv_epi8(soft_green16(input+x,raw_pitch),original,dy ? even : odd);
                soft_store16(output+3*x,dy ? zero : _mm_and_si128(original,even),green,
                             dy ? _mm_and_si128(original,odd) : zero);
            }
        }
        if (vector_width<width)
            bggr_legacy_g_impl<true>(raw+static_cast<size_t>(y)*raw_pitch+vector_width,raw_pitch,
                bgr+static_cast<size_t>(y)*bgr_pitch+vector_width*3,bgr_pitch,width-vector_width,2);
    }
}
}
#endif

void bggr_menon2007_g_cpu(const uint8_t* raw,int raw_pitch,uint8_t* bgr,
    int bgr_pitch,int width,int height) {
    bggr_legacy_g_impl<false>(raw,raw_pitch,bgr,bgr_pitch,width,height);
}

#include "softmenon_green.hpp"

void bggr_softmenon_g_cpu(const uint8_t* raw,int raw_pitch,uint8_t* bgr,
    int bgr_pitch,int width,int height) {
#if CPU_SOFT_AVX2
    if (width>=32 && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vbmi")) {
        soft_green_avx512(raw,raw_pitch,bgr,bgr_pitch,width,height);
        return;
    }
    if (width>=16 && __builtin_cpu_supports("avx2")) {
        if (width>=32) soft_green_chromatic_avx2(raw,raw_pitch,bgr,bgr_pitch,width,height);
        else soft_green_avx2(raw,raw_pitch,bgr,bgr_pitch,width,height);
        return;
    }
#endif
    bggr_legacy_g_impl<true>(raw,raw_pitch,bgr,bgr_pitch,width,height);
}

namespace {
inline void soft_rb_pixel(const uint8_t* source,ptrdiff_t pitch,uint8_t* output,int x,int y) {
    const uint8_t* p=source+static_cast<ptrdiff_t>(y)*pitch+static_cast<ptrdiff_t>(x)*3;
    const int green=p[1];
    const bool chromatic=(x&1)==(y&1);
    output[1]=static_cast<uint8_t>(green);
    for (int channel : {0,2}) {
        const bool measured=chromatic && (((y&1)==0)==(channel==0));
        if (measured) { output[channel]=p[channel]; continue; }
        int difference;
        if (chromatic) {
            const int ul=static_cast<int>(p[-pitch-3+channel])-p[-pitch-2];
            const int ur=static_cast<int>(p[-pitch+3+channel])-p[-pitch+4];
            const int dl=static_cast<int>(p[pitch-3+channel])-p[pitch-2];
            const int dr=static_cast<int>(p[pitch+3+channel])-p[pitch+4];
            const int horizontal=(ul+dr+1)>>1,vertical=(ur+dl+1)>>1;
            const int gh=std::abs(ul-dr),gv=std::abs(ur-dl);
            difference=gh<=gv ? horizontal : vertical;
            if (std::abs(gh-gv)<=26) difference=(horizontal+vertical+1)>>1;
        } else {
            const bool horizontal=((y&1)==0)==(channel==0);
            const ptrdiff_t axis=horizontal ? 3 : pitch;
            difference=(static_cast<int>(p[-axis+channel])-p[-axis+1]
                       +static_cast<int>(p[axis+channel])-p[axis+1]+1)>>1;
        }
        output[channel]=saturate_cast_int16_to_uint8(static_cast<int16_t>(green+difference));
    }
}

#if CPU_SOFT_AVX2
template <bool OddRow>
SOFT_INLINE void soft_rb8(const uint8_t* source,ptrdiff_t pitch,uint8_t* output,int x,int y) {
    __m256i values[9],green=_mm256_setzero_si256();
    for (int dy=-1;dy<=1;++dy) {
        const uint8_t* row=source+static_cast<ptrdiff_t>(y+dy)*pitch+static_cast<ptrdiff_t>(x-1)*3;
        const __m128i a=_mm_loadu_si128(reinterpret_cast<const __m128i*>(row));
        const __m128i b=_mm_loadu_si128(reinterpret_cast<const __m128i*>(row+14));
        const auto left=median_cpu_detail::row_chroma<0>(a,b);
        const auto center=median_cpu_detail::row_chroma<3>(a,b);
        const auto right=median_cpu_detail::row_chroma<6>(a,b);
        values[(dy+1)*3]=left.differences;
        values[(dy+1)*3+1]=center.differences;
        values[(dy+1)*3+2]=right.differences;
        if (dy==0) green=center.green;
    }
    const __m256i one=_mm256_set1_epi16(1);
    const __m256i diagonal_h=_mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(values[0],values[8]),one),1);
    const __m256i diagonal_v=_mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(values[2],values[6]),one),1);
    const __m256i gradient_h=_mm256_abs_epi16(_mm256_sub_epi16(values[0],values[8]));
    const __m256i gradient_v=_mm256_abs_epi16(_mm256_sub_epi16(values[2],values[6]));
    __m256i diagonal=_mm256_blendv_epi8(diagonal_h,diagonal_v,_mm256_cmpgt_epi16(gradient_h,gradient_v));
    const __m256i close=_mm256_cmpgt_epi16(_mm256_set1_epi16(27),
                                          _mm256_abs_epi16(_mm256_sub_epi16(gradient_h,gradient_v)));
    diagonal=_mm256_blendv_epi8(diagonal,_mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(diagonal_h,diagonal_v),one),1),close);
    const __m256i horizontal=_mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(values[3],values[5]),one),1);
    const __m256i vertical=_mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(values[1],values[7]),one),1);
    const __m256i axial=_mm256_blend_epi16(horizontal,vertical,OddRow ? 0x55 : 0xaa);
    const __m256i difference=_mm256_blend_epi16(diagonal,axial,OddRow ? 0x33 : 0xcc);
    const __m256i estimates=_mm256_add_epi16(green,difference);
    const __m256i measured=_mm256_add_epi16(green,values[4]);
    const __m128i colors=median_cpu_detail::clipped_bytes(_mm256_blend_epi16(estimates,measured,OddRow ? 0x88 : 0x11));
    const uint8_t* center=source+static_cast<ptrdiff_t>(y)*pitch+static_cast<ptrdiff_t>(x)*3;
    const __m128i a=_mm_or_si128(_mm_shuffle_epi8(colors,_mm_setr_epi8(0,-128,1,2,-128,3,4,-128,5,6,-128,7,8,-128,9,10)),
        _mm_and_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(center)),
                      _mm_setr_epi8(0,-1,0,0,-1,0,0,-1,0,0,-1,0,0,-1,0,0)));
    const __m128i b=_mm_or_si128(_mm_shuffle_epi8(colors,_mm_setr_epi8(5,6,-128,7,8,-128,9,10,-128,11,12,-128,13,14,-128,15)),
        _mm_and_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(center+8)),
                      _mm_setr_epi8(0,0,-1,0,0,-1,0,0,-1,0,0,-1,0,0,-1,0)));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output),a);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output+8),b);
}

__attribute__((target("avx2"))) void soft_rb_avx2(const uint8_t* source,size_t source_pitch,
        uint8_t* destination,size_t destination_pitch,int width,int begin_y,int end_y) {
    const ptrdiff_t pitch=static_cast<ptrdiff_t>(source_pitch);
    for (int y=begin_y;y<end_y;++y) {
        uint8_t* output=destination+static_cast<size_t>(y)*destination_pitch;
        int x=0;
        if (y&1) {
            for (;x<=width-8;x+=8) soft_rb8<true>(source,pitch,output+3*x,x,y);
        } else {
            for (;x<=width-8;x+=8) soft_rb8<false>(source,pitch,output+3*x,x,y);
        }
        for (;x<width;++x) soft_rb_pixel(source,pitch,output+3*x,x,y);
    }
}

template <bool OddRow>
SOFT512_INLINE void soft_rb16(const uint8_t* source,ptrdiff_t pitch,uint8_t* output,int x,int y) {
    __m512i values[9],green=_mm512_setzero_si512(),center_row=_mm512_setzero_si512();
    for (int dy=-1;dy<=1;++dy) {
        const uint8_t* input=source+static_cast<ptrdiff_t>(y+dy)*pitch+static_cast<ptrdiff_t>(x-1)*3;
        const __m512i row=_mm512_maskz_loadu_epi8((uint64_t{1}<<54)-1,input);
        const auto left=median_cpu_detail::row_chroma512<0>(row);
        const auto center=median_cpu_detail::row_chroma512<3>(row);
        const auto right=median_cpu_detail::row_chroma512<6>(row);
        values[(dy+1)*3]=left.differences;
        values[(dy+1)*3+1]=center.differences;
        values[(dy+1)*3+2]=right.differences;
        if (dy==0) { green=center.green; center_row=row; }
    }
    const __m512i one=_mm512_set1_epi16(1);
    const __m512i diagonal_h=_mm512_srai_epi16(_mm512_add_epi16(_mm512_add_epi16(values[0],values[8]),one),1);
    const __m512i diagonal_v=_mm512_srai_epi16(_mm512_add_epi16(_mm512_add_epi16(values[2],values[6]),one),1);
    const __m512i gradient_h=_mm512_abs_epi16(_mm512_sub_epi16(values[0],values[8]));
    const __m512i gradient_v=_mm512_abs_epi16(_mm512_sub_epi16(values[2],values[6]));
    __m512i diagonal=_mm512_mask_mov_epi16(diagonal_h,_mm512_cmpgt_epi16_mask(gradient_h,gradient_v),diagonal_v);
    const __mmask32 close=_mm512_cmpgt_epi16_mask(_mm512_set1_epi16(27),
        _mm512_abs_epi16(_mm512_sub_epi16(gradient_h,gradient_v)));
    diagonal=_mm512_mask_mov_epi16(diagonal,close,
        _mm512_srai_epi16(_mm512_add_epi16(_mm512_add_epi16(diagonal_h,diagonal_v),one),1));
    const __m512i horizontal=_mm512_srai_epi16(_mm512_add_epi16(_mm512_add_epi16(values[3],values[5]),one),1);
    const __m512i vertical=_mm512_srai_epi16(_mm512_add_epi16(_mm512_add_epi16(values[1],values[7]),one),1);
    const __m512i axial=_mm512_mask_mov_epi16(horizontal,OddRow ? 0x55555555u : 0xaaaaaaaau,vertical);
    const __m512i difference=_mm512_mask_mov_epi16(diagonal,OddRow ? 0x33333333u : 0xccccccccu,axial);
    const __m512i estimates=_mm512_add_epi16(green,difference);
    const __m512i measured=_mm512_add_epi16(green,values[4]);
    const __m512i colors=_mm512_mask_mov_epi16(estimates,OddRow ? 0x88888888u : 0x11111111u,measured);
    const __m512i clipped=_mm512_min_epi16(_mm512_max_epi16(colors,_mm512_setzero_si512()),_mm512_set1_epi16(255));
    const __m512i bytes=_mm512_castsi256_si512(_mm512_cvtepi16_epi8(clipped));
    alignas(64) static constexpr auto scatter_index=[] {
        std::array<uint8_t,64> index{};
        for (int i=0;i<48;++i) index[i]=2*(i/3)+(i%3==2);
        return index;
    }();
    alignas(64) static constexpr auto green_index=[] {
        std::array<uint8_t,64> index{};
        for (int i=0;i<48;++i) index[i]=i+3;
        return index;
    }();
    constexpr __mmask64 green_mask=[] {
        uint64_t mask=0;
        for (int i=1;i<48;i+=3) mask|=uint64_t{1}<<i;
        return mask;
    }();
    const __m512i result=_mm512_mask_mov_epi8(
        _mm512_permutexvar_epi8(_mm512_load_si512(scatter_index.data()),bytes),green_mask,
        _mm512_permutexvar_epi8(_mm512_load_si512(green_index.data()),center_row));
    _mm512_mask_storeu_epi8(output,(uint64_t{1}<<48)-1,result);
}

__attribute__((target("avx512f,avx512bw,avx512vbmi"))) void soft_rb_avx512(
        const uint8_t* source,size_t source_pitch,uint8_t* destination,size_t destination_pitch,
        int width,int begin_y,int end_y) {
    const ptrdiff_t pitch=static_cast<ptrdiff_t>(source_pitch);
    for (int y=begin_y;y<end_y;++y) {
        uint8_t* output=destination+static_cast<size_t>(y)*destination_pitch;
        int x=0;
        if (y&1) {
            for (;x<=width-16;x+=16) soft_rb16<true>(source,pitch,output+3*x,x,y);
        } else {
            for (;x<=width-16;x+=16) soft_rb16<false>(source,pitch,output+3*x,x,y);
        }
        for (;x<width;++x) soft_rb_pixel(source,pitch,output+3*x,x,y);
    }
}
#endif
}

void bggr_softmenon_rb_rows(const uint8_t* source,size_t source_pitch,uint8_t* destination,
        size_t destination_pitch,int width,int height,int begin_y,int end_y) {
    begin_y=std::max(0,begin_y); end_y=std::min(height,end_y);
#if CPU_SOFT_AVX2
    if (width>=16 && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vbmi")) {
        soft_rb_avx512(source,source_pitch,destination,destination_pitch,width,begin_y,end_y);
        return;
    }
    if (width>=8 && __builtin_cpu_supports("avx2")) {
        soft_rb_avx2(source,source_pitch,destination,destination_pitch,width,begin_y,end_y);
        return;
    }
#endif
    for (int y=begin_y;y<end_y;++y) {
        uint8_t* output=destination+static_cast<size_t>(y)*destination_pitch;
        for (int x=0;x<width;++x)
            soft_rb_pixel(source,static_cast<ptrdiff_t>(source_pitch),output+3*x,x,y);
    }
}

void bggr_menon2007_rb_cpu(
    uint8_t* bgr,
    int bgr_pitch,
    int width,
    int height)
{
    // Validate input parameters
    assert(bgr != nullptr && "BGR data pointer must not be null.");
    assert(width > 0 && "Image width must be positive.");
    assert(height > 0 && "Image height must be positive.");
    assert(width % 2 == 0 && "Image width must be even.");
    assert(height % 2 == 0 && "Image height must be even.");
    assert(bgr_pitch >= width * 3 && "BGR pitch must be at least equal to image width * 3.");

    // Iterate over the image in 2x2 blocks
    for(int y = 0; y < height; y += 2) {
        for(int x = 0; x < width; x += 2) {
            // Calculate the starting index of the 2x2 block in the BGR buffer
            uint8_t* bgr_block = bgr + y * bgr_pitch + x * 3;

            // Pointers to the four pixels in the 2x2 block
            uint8_t* P0 = bgr_block;                     // Upper Left (B)
            uint8_t* P1 = bgr_block + 3;                 // Upper Right (G)
            uint8_t* P2 = bgr_block + bgr_pitch;         // Lower Left (G)
            uint8_t* P3 = bgr_block + bgr_pitch + 3;     // Lower Right (R)

            // --- Estimate Red at P0 (Blue Pixel) ---
            {
                // Neighboring Red and Green pixels
                // Ensure that we do not access out-of-bounds memory
                // This assumes that the image has been appropriately padded
                int16_t R_UR = *(P0 + 2 + 3 - bgr_pitch);   // R at (x+1, y-1)
                int16_t R_LL = *(P0 + 2 - 3 + bgr_pitch);   // R at (x-1, y+1)
                int16_t G_UR = *(P0 + 1 + 3 - bgr_pitch);   // G at (x+1, y-1)
                int16_t G_LL = *(P0 + 1 - 3 + bgr_pitch);   // G at (x-1, y+1)

                int16_t R_UL = *(P0 + 2 - 3 - bgr_pitch);   // R at (x-1, y-1)
                int16_t R_LR = *(P0 + 2 + 3 + bgr_pitch);   // R at (x+1, y+1)
                int16_t G_UL = *(P0 + 1 - 3 - bgr_pitch);   // G at (x-1, y-1)
                int16_t G_LR = *(P0 + 1 + 3 + bgr_pitch);   // G at (x+1, y+1)

                // Compute color differences for Red channel estimation
                int16_t CD_UR = R_UR - G_UR;
                int16_t CD_LL = R_LL - G_LL;
                int16_t CD_UL = R_UL - G_UL;
                int16_t CD_LR = R_LR - G_LR;

                // Horizontal and Vertical estimates
                int16_t CD_h = (CD_UL + CD_LR + 1) >> 1;
                int16_t CD_v = (CD_UR + CD_LL + 1) >> 1;

                // Gradients to determine the direction of interpolation
                int16_t Grad_h = std::abs(CD_UL - CD_LR);
                int16_t Grad_v = std::abs(CD_UR - CD_LL);

                // Decision based on gradients
                int16_t CD_est = (Grad_h <= Grad_v) ? CD_h : CD_v;

                // Optional Close Averaging for smoother transitions
                #ifdef ENABLE_CLOSE_AVERAGING
                if (std::abs(Grad_h - Grad_v) <= 26) {
                    CD_est = (CD_h + CD_v + 1) >> 1;
                }
                #endif

                // Estimate Red value
                int16_t G_center = static_cast<int16_t>(P0[1]); // Green at P0
                int16_t R_est    = G_center + CD_est;

                // Clamp and assign Red value
                P0[2] = saturate_cast_int16_to_uint8(R_est);
            }

            // --- Estimate Blue at P3 (Red Pixel) ---
            {
                // Neighboring Blue and Green pixels
                uint8_t B_UR = *(P3 + 0 + 3 - bgr_pitch);   // B at (x+1, y-1)
                uint8_t B_LL = *(P3 + 0 - 3 + bgr_pitch);   // B at (x-1, y+1)
                uint8_t G_UR = *(P3 + 1 + 3 - bgr_pitch);   // G at (x+1, y-1)
                uint8_t G_LL = *(P3 + 1 - 3 + bgr_pitch);   // G at (x-1, y+1)

                uint8_t B_UL = *(P3 + 0 - 3 - bgr_pitch);   // B at (x-1, y-1)
                uint8_t B_LR = *(P3 + 0 + 3 + bgr_pitch);   // B at (x+1, y+1)
                uint8_t G_UL = *(P3 + 1 - 3 - bgr_pitch);   // G at (x-1, y-1)
                uint8_t G_LR = *(P3 + 1 + 3 + bgr_pitch);   // G at (x+1, y+1)

                // Compute color differences for Blue channel estimation
                int16_t CD_UR = static_cast<int16_t>(B_UR) - static_cast<int16_t>(G_UR);
                int16_t CD_LL = static_cast<int16_t>(B_LL) - static_cast<int16_t>(G_LL);
                int16_t CD_UL = static_cast<int16_t>(B_UL) - static_cast<int16_t>(G_UL);
                int16_t CD_LR = static_cast<int16_t>(B_LR) - static_cast<int16_t>(G_LR);

                // Horizontal and Vertical estimates
                int16_t CD_h = (CD_UL + CD_LR + 1) >> 1;
                int16_t CD_v = (CD_UR + CD_LL + 1) >> 1;

                // Gradients to determine the direction of interpolation
                int16_t Grad_h = std::abs(CD_UL - CD_LR);
                int16_t Grad_v = std::abs(CD_UR - CD_LL);

                // Decision based on gradients
                int16_t CD_est = (Grad_h <= Grad_v) ? CD_h : CD_v;

                // Optional Close Averaging for smoother transitions
                #ifdef ENABLE_CLOSE_AVERAGING
                if (std::abs(Grad_h - Grad_v) <= 26) {
                    CD_est = (CD_h + CD_v + 1) >> 1;
                }
                #endif

                // Estimate Blue value
                int16_t G_center = static_cast<int16_t>(P3[1]); // Green at P3
                int16_t B_est    = G_center + CD_est;

                // Clamp and assign Blue value
                P3[0] = saturate_cast_int16_to_uint8(B_est);
            }

            // --- Estimate Red and Blue at P1 (Upper Right Green Pixel) ---
            {
                // --- Estimate Red at P1 ---
                // Bilinear interpolation of (R - G)
                uint8_t R_up    = *(P1 + 2 - bgr_pitch);   // R at (x+1, y-1)
                uint8_t R_down  = *(P1 + 2 + bgr_pitch);   // R at (x+1, y+1)
                uint8_t G_up    = *(P1 + 1 - bgr_pitch);   // G at (x+1, y-1)
                uint8_t G_down  = *(P1 + 1 + bgr_pitch);   // G at (x+1, y+1)

                int16_t CD_RU = static_cast<int16_t>(R_up) - static_cast<int16_t>(G_up);
                int16_t CD_RD = static_cast<int16_t>(R_down) - static_cast<int16_t>(G_down);
                int16_t CD_R  = (CD_RU + CD_RD + 1) >> 1;

                int16_t G_center = static_cast<int16_t>(P1[1]); // Green at P1
                int16_t R_est    = G_center + CD_R;

                // --- Estimate Blue at P1 ---
                // Bilinear interpolation of (B - G)
                uint8_t B_left  = *(P1 - 3 + 0); // B at (x, y)
                uint8_t B_right = *(P1 + 3 + 0); // B at (x+2, y)
                uint8_t G_left  = *(P1 - 3 + 1); // G at (x, y)
                uint8_t G_right = *(P1 + 3 + 1); // G at (x+2, y)

                int16_t CD_BL = static_cast<int16_t>(B_left) - static_cast<int16_t>(G_left);
                int16_t CD_BR = static_cast<int16_t>(B_right) - static_cast<int16_t>(G_right);
                int16_t CD_B  = (CD_BL + CD_BR + 1) >> 1;

                int16_t B_est = G_center + CD_B;

                // Clamp and assign Red and Blue values
                P1[2] = saturate_cast_int16_to_uint8(R_est); // Red channel
                P1[0] = saturate_cast_int16_to_uint8(B_est); // Blue channel
            }

            // --- Estimate Red and Blue at P2 (Lower Left Green Pixel) ---
            {
                // --- Estimate Red at P2 ---
                // Bilinear interpolation of (R - G)
                uint8_t R_left  = *(P2 - 3 + 2); // R at (x-1, y+1)
                uint8_t R_right = *(P2 + 3 + 2); // R at (x+1, y+1)
                uint8_t G_left  = *(P2 - 3 + 1); // G at (x-1, y+1)
                uint8_t G_right = *(P2 + 3 + 1); // G at (x+1, y+1)

                int16_t CD_RL = static_cast<int16_t>(R_left) - static_cast<int16_t>(G_left);
                int16_t CD_RR = static_cast<int16_t>(R_right) - static_cast<int16_t>(G_right);
                int16_t CD_R  = (CD_RL + CD_RR + 1) >> 1;

                int16_t G_center = static_cast<int16_t>(P2[1]); // Green at P2
                int16_t R_est    = G_center + CD_R;

                // --- Estimate Blue at P2 ---
                // Bilinear interpolation of (B - G)
                uint8_t B_up    = *(P2 - bgr_pitch + 0); // B at (x, y)
                uint8_t B_down  = *(P2 + bgr_pitch + 0); // B at (x, y+2)
                uint8_t G_up    = *(P2 - bgr_pitch + 1); // G at (x, y)
                uint8_t G_down  = *(P2 + bgr_pitch + 1); // G at (x, y+2)

                int16_t CD_BU = static_cast<int16_t>(B_up) - static_cast<int16_t>(G_up);
                int16_t CD_BD = static_cast<int16_t>(B_down) - static_cast<int16_t>(G_down);
                int16_t CD_B  = (CD_BU + CD_BD + 1) >> 1;

                int16_t B_est = G_center + CD_B;

                // Clamp and assign Red and Blue values
                P2[2] = saturate_cast_int16_to_uint8(R_est); // Red channel
                P2[0] = saturate_cast_int16_to_uint8(B_est); // Blue channel
            }
        }
    }
}
