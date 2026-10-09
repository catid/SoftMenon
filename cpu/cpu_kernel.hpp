/**
 * @file cpu_kernel.hpp
 * @brief Header file for CPU-based kernel functions used in image processing, including image padding and Bayer pattern demosaicing.
 */

#ifndef CPU_KERNEL_HPP
#define CPU_KERNEL_HPP

#include <cstdint>
#include <cstddef>

// Raw points at logical (0,0), with a reflected halo supplied by the caller.
// Bilinear requires radius 1; Malvar requires radius 2. Pitches are bytes,
// row ranges are [begin_y,end_y), and output is interleaved BGR. Odd dimensions
// and both CFA phases are supported without touching row padding.
void bilinear_cpu_rows(const uint8_t* raw, size_t raw_pitch, uint8_t* bgr,
    size_t bgr_pitch, int width, int height, bool rggb, int begin_y, int end_y);
void malvar2004_cpu_rows(const uint8_t* raw, size_t raw_pitch, uint8_t* bgr,
    size_t bgr_pitch, int width, int height, bool rggb, int begin_y, int end_y);

// The legacy two-stage core is retained for diagnostics and SoftMenon. This
// entry replaces only its green decision with the frozen inverse-score blend.
// Like bggr_menon2007_g_cpu, the processed rectangle starts at a BGGR blue
// site, has even dimensions, and requires a two-pixel raw halo.
void bggr_softmenon_g_cpu(const uint8_t* raw, int raw_pitch, uint8_t* bgr,
    int bgr_pitch, int width, int height);

// Complete legacy chroma from an immutable BGGR green/measured-color image.
// Source names logical (0,0), including an initialized one-pixel halo;
// destination is a separate BGR image. Rows are absolute [begin_y,end_y).
void bggr_softmenon_rb_rows(const uint8_t* source, size_t source_pitch,
    uint8_t* destination, size_t destination_pitch, int width, int height,
    int begin_y, int end_y);

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
void padTopBottomEdges(uint8_t* data, int width, int height, int pitch, int pad);

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
void padLeftRightEdges(uint8_t* data, int width, int height, int pitch, int pad);

/**
 * @brief Clamps a 16-bit signed integer to an 8-bit unsigned integer range [0, 255].
 *
 * This inline helper function ensures that the input value does not exceed the
 * bounds of an 8-bit unsigned integer.
 *
 * @param x The 16-bit signed integer to clamp.
 * @return The clamped 8-bit unsigned integer.
 */
inline uint8_t saturate_cast_int16_to_uint8(int16_t x);

/**
 * @brief Fills green using the retained legacy Menon-inspired core.
 *
 * This function processes the raw Bayer data to estimate the Green channel values
 * using the original local classifiers and close-averaging thresholds. This
 * diagnostic core is separate from the faithful public Menon implementation.
 *
 * @param raw Pointer to the raw Bayer data buffer.
 * @param raw_pitch Number of bytes per row in the raw Bayer data.
 * @param bgr Pointer to the output BGR buffer (green and measured colors are filled).
 * @param bgr_pitch Number of bytes per row in the BGR data buffer.
 * @param width Width of the image in pixels (must be even).
 * @param height Height of the image in pixels (must be even).
 *
 * @pre `raw` and `bgr` must not be `nullptr`.
 * @pre `width` and `height` must be positive and even.
 * @pre `raw_pitch` and `bgr_pitch` must be sufficient to hold the respective image data.
 *
 * @note The pointers name the first processed BGGR pixel, not an allocation
 *       origin. raw needs a two-pixel halo on every side. This pass also stores
 *       the measured red/blue samples; missing colors are filled separately.
 */
void bggr_menon2007_g_cpu(
    const uint8_t* raw,
    int raw_pitch,
    uint8_t* bgr,
    int bgr_pitch,
    int width,
    int height);

/**
 * @brief Fills red and blue using the retained legacy chroma stage.
 *
 * This function processes the partially filled BGR buffer to estimate the missing Red and
 * Blue channel values using diagonal color differences and close averaging.
 * This is the legacy diagnostic stage, not the paper's DDFAPD refinement.
 *
 * @param bgr Pointer to the BGR data buffer (Green channel should be already filled).
 * @param bgr_pitch Number of bytes per row in the BGR data buffer.
 * @param width Width of the image in pixels (must be even).
 * @param height Height of the image in pixels (must be even).
 *
 * @pre `bgr` must not be `nullptr`.
 * @pre `width` and `height` must be positive and even.
 * @pre `bgr_pitch` must be sufficient to hold the image data.
 *
 * @note bgr must point into a buffer with a one-pixel halo on every side.
 *       Green and measured red/blue samples, including that halo, must already
 *       be filled. The even-sized region starts on a BGGR blue pixel.
 */
void bggr_menon2007_rb_cpu(
    uint8_t* bgr,
    int bgr_pitch,
    int width,
    int height);

#endif // CPU_KERNEL_HPP
