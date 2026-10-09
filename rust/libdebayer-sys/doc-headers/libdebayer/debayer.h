#ifndef DEBAYERC_H
#define DEBAYERC_H

#include "cuda_runtime.h"

#include <stdio.h>
#include <stdint.h>

/* Two-pixel halo on each side of both device buffers. Width and height name
 * the interior and must be >= 2; odd dimensions are supported. Pitches are in
 * bytes and must cover width + 4 input bytes or 3 * (width + 4) output bytes.
 * Each allocation must contain height + 4 complete pitched rows. Input and
 * output allocations must not overlap. Call debayer_mirror_image after filling
 * the input interior at (2, 2), before calling a demosaicing entry point.
 *
 * Padding uses reflect-101, preserving Bayer phase. Only the output interior
 * is defined; output halo bytes are never needed or changed. Calls enqueue work
 * on stream and return validation/launch status. Asynchronous execution errors
 * must still be checked when the caller synchronizes the stream.
 */
#define SARONIC_DEBAYER_PAD 2

// Kernel block size
#define KERNEL_BLOCK_SIZE 8

#ifdef __cplusplus
extern "C" {
#endif
cudaError_t debayer_mirror_image(cudaStream_t stream, int32_t width, int32_t height, size_t input_pitch, uint8_t* input_data);

cudaError_t debayer_rggb2bgr_malvar2004(cudaStream_t stream, int32_t width, int32_t height, size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data);
cudaError_t debayer_bggr2bgr_malvar2004(cudaStream_t stream, int32_t width, int32_t height, size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data);

cudaError_t debayer_rggb2bgr_bilinear(cudaStream_t stream, int32_t width, int32_t height, size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data);
cudaError_t debayer_bggr2bgr_bilinear(cudaStream_t stream, int32_t width, int32_t height, size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data);

cudaError_t debayer_rggb2bgr_menon2007(cudaStream_t stream, int32_t width, int32_t height, size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data);
cudaError_t debayer_bggr2bgr_menon2007(cudaStream_t stream, int32_t width, int32_t height, size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data);

/* Menon is the complete DDFAPD baseline, including refinement. SoftMenon retains
 * the project's integer reconstruction with soft green and chroma cleanup.
 * Convenience entry points allocate stream-ordered scratch. For repeated
 * frames, use the workspace forms below with reusable, nonoverlapping storage.
 */
cudaError_t debayer_rggb2bgr_softmenon(cudaStream_t stream, int32_t width, int32_t height, size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data);
cudaError_t debayer_bggr2bgr_softmenon(cudaStream_t stream, int32_t width, int32_t height, size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data);
size_t debayer_menon2007_workspace_size(int32_t width, int32_t height);
size_t debayer_softmenon_workspace_size(int32_t width, int32_t height);
/* rggb must be 1 for RGGB or 0 for BGGR; all other values are rejected.
 * Workspace is device memory, aligned to at least 4 bytes, whose lifetime
 * extends until stream completion. Queries return 0 for invalid dimensions.
 */
cudaError_t debayer_menon2007_with_workspace(cudaStream_t stream, int32_t width, int32_t height, size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data, int32_t rggb, void* workspace, size_t workspace_bytes);
cudaError_t debayer_softmenon_with_workspace(cudaStream_t stream, int32_t width, int32_t height, size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data, int32_t rggb, void* workspace, size_t workspace_bytes);

#ifdef __cplusplus
}
#endif
#endif // DEBAYERC_H
