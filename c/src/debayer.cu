#include "debayer.h"
#include "debayer_kernel.h"
#include "chroma_median.cuh"
#include "../../common/menon2007.hpp"

#include <limits>

namespace {

bool valid_buffer(int32_t width, int32_t height, size_t pitch, size_t channels,
                  const uint8_t* data) {
    if (!data || width < 2 || height < 2 ||
        width > std::numeric_limits<int32_t>::max() - 2 * SARONIC_DEBAYER_PAD ||
        height > std::numeric_limits<int32_t>::max() - 2 * SARONIC_DEBAYER_PAD) {
        return false;
    }
    const size_t rows = static_cast<size_t>(height) + 2 * SARONIC_DEBAYER_PAD;
    const size_t columns = static_cast<size_t>(width) + 2 * SARONIC_DEBAYER_PAD;
    const size_t max_offset = static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max());
    return columns <= max_offset / channels && pitch >= columns * channels &&
           pitch <= max_offset / rows;
}

bool overlapping(const uint8_t* input, size_t input_bytes,
                 const uint8_t* output, size_t output_bytes) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(input);
    const uintptr_t b = reinterpret_cast<uintptr_t>(output);
    return a <= b ? b - a < input_bytes : a - b < output_bytes;
}

enum class Method { RggbMalvar, BggrMalvar, RggbBilinear, BggrBilinear };

cudaError_t demosaic(Method method, cudaStream_t stream, int32_t width, int32_t height,
                    size_t input_pitch, size_t output_pitch,
                    uint8_t* input_data, uint8_t* output_data) {
    if (!valid_buffer(width, height, input_pitch, 1, input_data) ||
        !valid_buffer(width, height, output_pitch, 3, output_data)) {
        return cudaErrorInvalidValue;
    }
    const size_t rows = static_cast<size_t>(height) + 2 * SARONIC_DEBAYER_PAD;
    if (overlapping(input_data, input_pitch * rows, output_data, output_pitch * rows)) {
        return cudaErrorInvalidValue;
    }
    const ptrdiff_t raw_pitch = static_cast<ptrdiff_t>(input_pitch);
    const ptrdiff_t bgr_pitch = static_cast<ptrdiff_t>(output_pitch);
    const uint8_t* raw = input_data + SARONIC_DEBAYER_PAD * raw_pitch + SARONIC_DEBAYER_PAD;
    uint8_t* bgr = output_data + SARONIC_DEBAYER_PAD * bgr_pitch + SARONIC_DEBAYER_PAD * 3;
    const dim3 block(KERNEL_BLOCK_SIZE, KERNEL_BLOCK_SIZE);
    const dim3 grid(((static_cast<unsigned>(width) + 1) / 2 + block.x - 1) / block.x,
                    ((static_cast<unsigned>(height) + 1) / 2 + block.y - 1) / block.y);
    switch (method) {
    case Method::RggbMalvar:
        rggb_malvar2004<<<grid, block, 0, stream>>>(raw, raw_pitch, bgr, bgr_pitch, width, height);
        break;
    case Method::BggrMalvar:
        bggr_malvar2004<<<grid, block, 0, stream>>>(raw, raw_pitch, bgr, bgr_pitch, width, height);
        break;
    case Method::RggbBilinear:
        rggb_bilinear<<<grid, block, 0, stream>>>(raw, raw_pitch, bgr, bgr_pitch, width, height);
        break;
    case Method::BggrBilinear:
        bggr_bilinear<<<grid, block, 0, stream>>>(raw, raw_pitch, bgr, bgr_pitch, width, height);
        break;
    }
    return cudaGetLastError();
}

} // namespace

namespace {
cudaError_t advanced_allocating(bool soft, bool rggb, cudaStream_t stream, int32_t width, int32_t height,
    size_t input_pitch, size_t output_pitch, uint8_t* input, uint8_t* output);
}

cudaError_t debayer_mirror_image(cudaStream_t stream, int32_t width, int32_t height,
                                size_t input_pitch, uint8_t* input_data) {
    if (!valid_buffer(width, height, input_pitch, 1, input_data)) return cudaErrorInvalidValue;
    const ptrdiff_t pitch = static_cast<ptrdiff_t>(input_pitch);
    const dim3 top_block(128, SARONIC_DEBAYER_PAD);
    const dim3 top_grid((static_cast<unsigned>(width) + 2 * SARONIC_DEBAYER_PAD + top_block.x - 1) / top_block.x);
    mirrorEdgesTopBottom<<<top_grid, top_block, 0, stream>>>(
        input_data, width, height, pitch, SARONIC_DEBAYER_PAD);
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    const dim3 side_block(SARONIC_DEBAYER_PAD, 128);
    const dim3 side_grid(1, (static_cast<unsigned>(height) + side_block.y - 1) / side_block.y);
    mirrorEdgesLeftRight<<<side_grid, side_block, 0, stream>>>(
        input_data, width, height, pitch, SARONIC_DEBAYER_PAD);
    return cudaGetLastError();
}

cudaError_t debayer_rggb2bgr_malvar2004(cudaStream_t stream, int32_t width, int32_t height,
    size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data) {
    return demosaic(Method::RggbMalvar, stream, width, height, input_pitch, output_pitch, input_data, output_data);
}

cudaError_t debayer_bggr2bgr_malvar2004(cudaStream_t stream, int32_t width, int32_t height,
    size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data) {
    return demosaic(Method::BggrMalvar, stream, width, height, input_pitch, output_pitch, input_data, output_data);
}

cudaError_t debayer_rggb2bgr_bilinear(cudaStream_t stream, int32_t width, int32_t height,
    size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data) {
    return demosaic(Method::RggbBilinear, stream, width, height, input_pitch, output_pitch, input_data, output_data);
}

cudaError_t debayer_rggb2bgr_menon2007(cudaStream_t stream, int32_t width, int32_t height,
    size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data) {
    return advanced_allocating(false, true, stream, width, height, input_pitch, output_pitch, input_data, output_data);
}

cudaError_t debayer_bggr2bgr_menon2007(cudaStream_t stream, int32_t width, int32_t height,
    size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data) {
    return advanced_allocating(false, false, stream, width, height, input_pitch, output_pitch, input_data, output_data);
}

namespace {
bool advanced_buffers(int width, int height, size_t raw_pitch, size_t bgr_pitch,
                      uint8_t* raw, uint8_t* bgr) {
    if (!valid_buffer(width, height, raw_pitch, 1, raw) ||
        !valid_buffer(width, height, bgr_pitch, 3, bgr)) return false;
    const size_t rows = static_cast<size_t>(height) + 2 * SARONIC_DEBAYER_PAD;
    return !overlapping(raw, raw_pitch * rows, bgr, bgr_pitch * rows);
}

size_t soft_pitch(int width) {
    return ((static_cast<size_t>(width) + 4) * 3 + 127) / 128 * 128;
}

bool valid_workspace(int width, int height, size_t raw_pitch, size_t bgr_pitch,
    uint8_t* raw, uint8_t* bgr, int rggb, void* workspace, size_t bytes, size_t required) {
    if (!required || !workspace || bytes < required || (reinterpret_cast<uintptr_t>(workspace) & 3) ||
        (rggb != 0 && rggb != 1) || !advanced_buffers(width, height, raw_pitch, bgr_pitch, raw, bgr)) return false;
    const size_t rows = static_cast<size_t>(height) + 2 * SARONIC_DEBAYER_PAD;
    const auto* scratch = static_cast<uint8_t*>(workspace);
    return !overlapping(raw, raw_pitch * rows, scratch, required) &&
           !overlapping(bgr, bgr_pitch * rows, scratch, required);
}

template <int Stage>
__global__ void paper_menon_stage(libdebayer_menon2007::Buffers buffers) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= buffers.width || y >= buffers.height) return;
    using namespace libdebayer_menon2007;
    if constexpr (Stage == 0) stage_green(buffers, x, y);
    if constexpr (Stage == 1) stage_decision(buffers, x, y);
    if constexpr (Stage == 2) stage_colors_at_green(buffers, x, y);
    if constexpr (Stage == 3) stage_opposite(buffers, x, y);
    if constexpr (Stage == 4) stage_refine_green(buffers, x, y);
    if constexpr (Stage == 5) stage_refine_colors_at_green(buffers, x, y);
    if constexpr (Stage == 6) stage_refine_opposite_output(buffers, x, y);
}

cudaError_t advanced_allocating(bool soft, bool rggb, cudaStream_t stream, int32_t width, int32_t height,
    size_t input_pitch, size_t output_pitch, uint8_t* input, uint8_t* output) {
    if (!advanced_buffers(width, height, input_pitch, output_pitch, input, output)) return cudaErrorInvalidValue;
    const size_t bytes = soft ? debayer_softmenon_workspace_size(width, height) : debayer_menon2007_workspace_size(width, height);
    if (!bytes) return cudaErrorInvalidValue;
    void* workspace = nullptr;
    cudaError_t status = cudaMallocAsync(&workspace, bytes, stream);
    if (status != cudaSuccess) return status;
    status = soft ? debayer_softmenon_with_workspace(stream, width, height, input_pitch, output_pitch, input, output, rggb, workspace, bytes) :
                    debayer_menon2007_with_workspace(stream, width, height, input_pitch, output_pitch, input, output, rggb, workspace, bytes);
    const cudaError_t cleanup = cudaFreeAsync(workspace, stream);
    return status == cudaSuccess ? cleanup : status;
}
} // namespace

size_t debayer_menon2007_workspace_size(int32_t width, int32_t height) {
    if (width < 2 || height < 2 || width > INT32_MAX - 2 * SARONIC_DEBAYER_PAD ||
        height > INT32_MAX - 2 * SARONIC_DEBAYER_PAD) return 0;
    const size_t max_size = static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max());
    if (static_cast<size_t>(width) > max_size / 33 / height) return 0;
    return static_cast<size_t>(width) * height * 33;
}

size_t debayer_softmenon_workspace_size(int32_t width, int32_t height) {
    if (width < 2 || height < 2 || width > INT32_MAX - 2 * SARONIC_DEBAYER_PAD ||
        height > INT32_MAX - 2 * SARONIC_DEBAYER_PAD) return 0;
    const size_t pitch = soft_pitch(width), rows = static_cast<size_t>(height) + 4;
    if (pitch > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) / rows) return 0;
    return pitch * rows;
}

cudaError_t debayer_menon2007_with_workspace(cudaStream_t stream, int32_t width, int32_t height,
    size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data,
    int32_t rggb, void* workspace, size_t workspace_bytes) {
    const size_t required = debayer_menon2007_workspace_size(width, height);
    if (!valid_workspace(width, height, input_pitch, output_pitch, input_data, output_data,
        rggb, workspace, workspace_bytes, required)) return cudaErrorInvalidValue;
    const size_t pixels = static_cast<size_t>(width) * height;
    auto* integers = static_cast<int32_t*>(workspace);
    libdebayer_menon2007::Buffers buffers{
        input_data + SARONIC_DEBAYER_PAD * input_pitch + SARONIC_DEBAYER_PAD, input_pitch,
        output_data + SARONIC_DEBAYER_PAD * output_pitch + 3 * SARONIC_DEBAYER_PAD, output_pitch, width, height, rggb != 0,
        integers, integers + pixels, integers + 2 * pixels, integers + 5 * pixels,
        reinterpret_cast<uint8_t*>(integers + 8 * pixels)};
    const dim3 block(32, 8),
        grid((static_cast<unsigned>(width) + 31u) / 32u,
             (static_cast<unsigned>(height) + 7u) / 8u);
#define MENON_STAGE(stage) \
    paper_menon_stage<stage><<<grid, block, 0, stream>>>(buffers); \
    if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) return status
    MENON_STAGE(0); MENON_STAGE(1); MENON_STAGE(2); MENON_STAGE(3);
    MENON_STAGE(4); MENON_STAGE(5); MENON_STAGE(6);
#undef MENON_STAGE
    return cudaSuccess;
}

cudaError_t debayer_softmenon_with_workspace(cudaStream_t stream, int32_t width, int32_t height,
    size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data,
    int32_t rggb, void* workspace, size_t workspace_bytes) {
    const size_t required = debayer_softmenon_workspace_size(width, height);
    if (!valid_workspace(width, height, input_pitch, output_pitch, input_data, output_data,
        rggb, workspace, workspace_bytes, required)) return cudaErrorInvalidValue;
    const size_t pitch = soft_pitch(width);
    const uint8_t* raw = input_data + SARONIC_DEBAYER_PAD * input_pitch + SARONIC_DEBAYER_PAD;
    uint8_t* intermediate = static_cast<uint8_t*>(workspace) + 2 * pitch + 6;
    const dim3 block(KERNEL_BLOCK_SIZE, KERNEL_BLOCK_SIZE);
    const dim3 grid(((width + 1) / 2 + block.x - 1) / block.x,
                    ((height + 1) / 2 + block.y - 1) / block.y);
    if (rggb) rggb_softmenon_g<<<grid, block, 0, stream>>>(raw, input_pitch, intermediate, pitch, width, height);
    else bggr_softmenon_g<<<grid, block, 0, stream>>>(raw, input_pitch, intermediate, pitch, width, height);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    if (rggb) rggb_softmenon_rb<<<grid, block, 0, stream>>>(intermediate, pitch, width, height);
    else bggr_softmenon_rb<<<grid, block, 0, stream>>>(intermediate, pitch, width, height);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    return softmenon_chroma_median(intermediate, pitch,
        output_data + SARONIC_DEBAYER_PAD * output_pitch + 3 * SARONIC_DEBAYER_PAD, output_pitch, width, height, rggb != 0, stream);
}

cudaError_t debayer_rggb2bgr_softmenon(cudaStream_t stream, int32_t width, int32_t height,
    size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data) {
    return advanced_allocating(true, true, stream, width, height, input_pitch, output_pitch, input_data, output_data);
}
cudaError_t debayer_bggr2bgr_softmenon(cudaStream_t stream, int32_t width, int32_t height,
    size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data) {
    return advanced_allocating(true, false, stream, width, height, input_pitch, output_pitch, input_data, output_data);
}
cudaError_t debayer_bggr2bgr_bilinear(cudaStream_t stream, int32_t width, int32_t height,
    size_t input_pitch, size_t output_pitch, uint8_t* input_data, uint8_t* output_data) {
    return demosaic(Method::BggrBilinear, stream, width, height, input_pitch, output_pitch, input_data, output_data);
}
