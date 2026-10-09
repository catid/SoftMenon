#include "debayer_cpp.h"
#include "libdebayer/debayer.h"

#include <limits>

void Debayer::Free()
{
    // Complete any queued transfers before releasing their buffers.
    if (stream) cudaStreamSynchronize(stream);
    if (raw_cuda_data) cudaFree(raw_cuda_data);
    if (bgr_cuda_data) cudaFree(bgr_cuda_data);
    if (workspace_cuda_data) cudaFree(workspace_cuda_data);
    if (stream) cudaStreamDestroy(stream);
    raw_cuda_data = nullptr;
    bgr_cuda_data = nullptr;
    workspace_cuda_data = nullptr;
    workspace_cuda_bytes = 0;
    raw_cuda_pitch = bgr_cuda_pitch = 0;
    stream = nullptr;
    cuda_width = cuda_height = -1;
}

static size_t PaddedSize(int size)
{
    const size_t value = static_cast<size_t>(size) + SARONIC_DEBAYER_PAD;
    return SARONIC_DEBAYER_PAD +
        ((value + KERNEL_BLOCK_SIZE - 1) / KERNEL_BLOCK_SIZE) * KERNEL_BLOCK_SIZE;
}

bool Debayer::Allocate(int width, int height)
{
    if (cuda_width == width && cuda_height == height) return true;
    Free();

    const size_t padded_width = PaddedSize(width);
    const size_t padded_height = PaddedSize(height);
    if (cudaStreamCreate(&stream) != cudaSuccess ||
        cudaMallocPitch(&raw_cuda_data, &raw_cuda_pitch, padded_width, padded_height) != cudaSuccess ||
        cudaMemset2D(raw_cuda_data, raw_cuda_pitch, 0, padded_width, padded_height) != cudaSuccess ||
        cudaMallocPitch(&bgr_cuda_data, &bgr_cuda_pitch, padded_width * 3, padded_height) != cudaSuccess ||
        cudaMemset2D(bgr_cuda_data, bgr_cuda_pitch, 0, padded_width * 3, padded_height) != cudaSuccess) {
        Free();
        cudaGetLastError(); // Allocation failure was handled; do not poison a later launch.
        return false;
    }
    // Only a fully initialized allocation may be reused.
    cuda_width = width;
    cuda_height = height;
    return true;
}

int32_t Debayer::Process(const raw_image_t* input, const bgr_image_t* output)
{
    if (!input || !output || !input->raw_data || !output->bgr_data) return -8;
    if (input->width != output->width || input->height != output->height) return -1;
    if (input->width < 2 || input->height < 2 ||
        input->width > std::numeric_limits<int32_t>::max() / 3 ||
        input->pitch < 0 || output->pitch < 0) return -8;

    const size_t input_pitch = input->pitch ? static_cast<size_t>(input->pitch) : input->width;
    const size_t output_width_bytes = static_cast<size_t>(output->width) * 3;
    const size_t output_pitch = output->pitch ? static_cast<size_t>(output->pitch) : output_width_bytes;
    if (input_pitch < static_cast<size_t>(input->width) || output_pitch < output_width_bytes) return -8;
    if (input->algorithm != SARONIC_DEBAYER_BILINEAR &&
        input->algorithm != SARONIC_DEBAYER_MALVAR2004 &&
        input->algorithm != SARONIC_DEBAYER_MENON2007 &&
        input->algorithm != SARONIC_DEBAYER_SOFTMENON) return -6;
    if (input->format != SARONIC_DEBAYER_RGGB && input->format != SARONIC_DEBAYER_BGGR) return -7;
    if (!Allocate(input->width, input->height)) return -2;

    const size_t required = input->algorithm == SARONIC_DEBAYER_MENON2007 ?
        debayer_menon2007_workspace_size(input->width, input->height) :
        input->algorithm == SARONIC_DEBAYER_SOFTMENON ?
        debayer_softmenon_workspace_size(input->width, input->height) : 0;
    if (required > workspace_cuda_bytes) {
        void* replacement = nullptr;
        if (cudaMalloc(&replacement, required) != cudaSuccess) {
            cudaGetLastError();
            return -2;
        }
        if (workspace_cuda_data) cudaFree(workspace_cuda_data);
        workspace_cuda_data = replacement;
        workspace_cuda_bytes = required;
    }

    // Every error after upload drains the stream before returning host buffers
    // to the caller, who may immediately destroy or reuse them.
    const auto fail = [this](int32_t code) {
        cudaStreamSynchronize(stream);
        cudaGetLastError();
        return code;
    };
    cudaError_t result = cudaMemcpy2DAsync(
        raw_cuda_data + SARONIC_DEBAYER_PAD * raw_cuda_pitch + SARONIC_DEBAYER_PAD,
        raw_cuda_pitch, input->raw_data, input_pitch, input->width, input->height,
        cudaMemcpyHostToDevice, stream);
    if (result != cudaSuccess) return fail(-3);

    if (input->algorithm != SARONIC_DEBAYER_MENON2007) {
        result = debayer_mirror_image(stream, input->width, input->height, raw_cuda_pitch, raw_cuda_data);
        if (result != cudaSuccess) return fail(-9);
    }
    if (input->algorithm == SARONIC_DEBAYER_BILINEAR) {
        if (input->format == SARONIC_DEBAYER_RGGB)
            result = debayer_rggb2bgr_bilinear(stream, input->width, input->height, raw_cuda_pitch, bgr_cuda_pitch, raw_cuda_data, bgr_cuda_data);
        else
            result = debayer_bggr2bgr_bilinear(stream, input->width, input->height, raw_cuda_pitch, bgr_cuda_pitch, raw_cuda_data, bgr_cuda_data);
    } else if (input->algorithm == SARONIC_DEBAYER_MALVAR2004) {
        if (input->format == SARONIC_DEBAYER_RGGB)
            result = debayer_rggb2bgr_malvar2004(stream, input->width, input->height, raw_cuda_pitch, bgr_cuda_pitch, raw_cuda_data, bgr_cuda_data);
        else
            result = debayer_bggr2bgr_malvar2004(stream, input->width, input->height, raw_cuda_pitch, bgr_cuda_pitch, raw_cuda_data, bgr_cuda_data);
    } else if (input->algorithm == SARONIC_DEBAYER_MENON2007) {
        result = debayer_menon2007_with_workspace(stream, input->width, input->height,
            raw_cuda_pitch, bgr_cuda_pitch, raw_cuda_data, bgr_cuda_data,
            input->format == SARONIC_DEBAYER_RGGB, workspace_cuda_data, workspace_cuda_bytes);
    } else {
        result = debayer_softmenon_with_workspace(stream, input->width, input->height,
            raw_cuda_pitch, bgr_cuda_pitch, raw_cuda_data, bgr_cuda_data,
            input->format == SARONIC_DEBAYER_RGGB, workspace_cuda_data, workspace_cuda_bytes);
    }
    if (result != cudaSuccess) return fail(-9);

    result = cudaMemcpy2DAsync(output->bgr_data, output_pitch,
        bgr_cuda_data + SARONIC_DEBAYER_PAD * bgr_cuda_pitch + SARONIC_DEBAYER_PAD * 3,
        bgr_cuda_pitch, output_width_bytes, output->height, cudaMemcpyDeviceToHost, stream);
    if (result != cudaSuccess) return fail(-4);
    if (cudaStreamSynchronize(stream) != cudaSuccess) return -5;
    return 0;
}
