#include "cpu_debayer.hpp"
#include "chroma_median.hpp"
#include "../common/menon2007.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>

namespace {
int ReflectBayerCoordinate(int coordinate, int length)
{
    const int64_t period = 2 * (static_cast<int64_t>(length) - 1);
    int64_t reflected = coordinate % period;
    if (reflected < 0) reflected += period;
    return static_cast<int>(reflected < length ? reflected : period - reflected);
}

size_t DefaultWorkers()
{
    const size_t available = std::thread::hardware_concurrency();
    return available ? std::min<size_t>(available, 8) : 4;
}
}

Debayer::Debayer(size_t workers) : worker_count(workers ? workers : DefaultWorkers()) {}

Debayer::~Debayer()
{
    Free();
}

void Debayer::InitializeThreadPool()
{
    if (!thread_pool && worker_count > 1)
        thread_pool = std::make_unique<ThreadPool>(worker_count - 1);
}

void Debayer::RunRows(int rows, int alignment, const std::function<void(int, int)>& task)
{
    // A single worker runs on the calling thread without queue/barrier overhead.
    // Small images also use one task. Native 2x2 kernels require even slices.
    const size_t tasks = std::min(worker_count, static_cast<size_t>((rows + 31) / 32));
    if (tasks <= 1) {
        task(0, rows);
        return;
    }
    InitializeThreadPool();
    const int units = rows / alignment;
    const int units_per_task = static_cast<int>((static_cast<size_t>(units) + tasks - 1) / tasks);
    const int rows_per_task = units_per_task * alignment;
    for (int begin = 0; begin < rows; begin += rows_per_task) {
        const int end = std::min(rows, begin + rows_per_task);
        // Copy the callable: if a later submission fails, queued tasks must not
        // retain a reference to a destroyed temporary std::function.
        if (end == rows) {
            // The caller is one of the configured workers. Keeping one slice
            // here avoids an extra runnable coordinator on a fully used CPU.
            task(begin, end);
        } else {
            thread_pool->Submit([task, begin, end] { task(begin, end); });
        }
    }
    thread_pool->WaitAll();
}

bool Debayer::Allocate(int new_width, int new_height)
{
    if (new_width < 2 || new_height < 2) return false;
    const int64_t padded_width = 2 * SARONIC_DEBAYER_PAD +
        ((static_cast<int64_t>(new_width) + KERNEL_BLOCK_SIZE - 1) /
         KERNEL_BLOCK_SIZE) * KERNEL_BLOCK_SIZE;
    const int64_t padded_height = 2 * SARONIC_DEBAYER_PAD +
        ((static_cast<int64_t>(new_height) + KERNEL_BLOCK_SIZE - 1) /
         KERNEL_BLOCK_SIZE) * KERNEL_BLOCK_SIZE;
    // Native kernels use signed int offsets, including interleaved BGR offsets.
    if (padded_width > std::numeric_limits<int>::max() / 3 ||
        padded_height > std::numeric_limits<int>::max() ||
        padded_width * padded_height > std::numeric_limits<int>::max() / 3)
        return false;
    if (width == new_width && height == new_height && raw_padded_data && bgr_padded_data)
        return true;

    const size_t raw_bytes = static_cast<size_t>(padded_width * padded_height);
    std::unique_ptr<uint8_t[]> raw(new (std::nothrow) uint8_t[raw_bytes]());
    std::unique_ptr<uint8_t[]> bgr(new (std::nothrow) uint8_t[raw_bytes * 3]());
    if (!raw || !bgr) return false;

    // Commit only after all allocations succeed, preserving prior valid state.
    Free();
    width = new_width;
    height = new_height;
    raw_padded_width = bgr_padded_width = static_cast<int>(padded_width);
    raw_padded_height = bgr_padded_height = static_cast<int>(padded_height);
    raw_padded_pitch = raw_padded_width;
    bgr_padded_pitch = bgr_padded_width * 3;
    raw_padded_data = raw.release();
    bgr_padded_data = bgr.release();
    return true;
}

void Debayer::Free()
{
    delete[] raw_padded_data;
    delete[] bgr_padded_data;
    raw_padded_data = bgr_padded_data = nullptr;
    raw_padded_pitch = raw_padded_width = raw_padded_height = 0;
    bgr_padded_pitch = bgr_padded_width = bgr_padded_height = 0;
    width = height = 0;
    soft_colors.reset();
    menon_planes.reset();
    menon_direction.reset();
}

bool Debayer::AllocateSoftScratch()
{
    if (soft_colors) return true;
    const size_t bytes = static_cast<size_t>(width) * height * 3;
    soft_colors.reset(new (std::nothrow) uint8_t[bytes]);
    return soft_colors != nullptr;
}

bool Debayer::AllocateMenonScratch()
{
    if (menon_planes && menon_direction) return true;
    const size_t pixels = static_cast<size_t>(width) * height;
    std::unique_ptr<int32_t[]> planes(new (std::nothrow) int32_t[pixels * 8]);
    std::unique_ptr<uint8_t[]> direction(new (std::nothrow) uint8_t[pixels]);
    if (!planes || !direction) return false;
    menon_planes = std::move(planes);
    menon_direction = std::move(direction);
    return true;
}

void Debayer::PadRaw(const uint8_t* source, int source_pitch)
{
    // Only the small halo needs reflection arithmetic; the interior is a row
    // copy. Precompute side indexes once, then copy completed rows vertically.
    int left[SARONIC_DEBAYER_PAD];
    int right[SARONIC_DEBAYER_PAD + KERNEL_BLOCK_SIZE];
    const int right_count = raw_padded_width - SARONIC_DEBAYER_PAD - width;
    for (int x = 0; x < SARONIC_DEBAYER_PAD; ++x)
        left[x] = ReflectBayerCoordinate(x - SARONIC_DEBAYER_PAD, width);
    for (int x = 0; x < right_count; ++x)
        right[x] = ReflectBayerCoordinate(width + x, width);
    for (int y = 0; y < height; ++y) {
        const uint8_t* input_row = source + static_cast<size_t>(y) * source_pitch;
        uint8_t* row = raw_padded_data + static_cast<size_t>(y + SARONIC_DEBAYER_PAD) * raw_padded_pitch;
        std::memcpy(row + SARONIC_DEBAYER_PAD, input_row, static_cast<size_t>(width));
        for (int x = 0; x < SARONIC_DEBAYER_PAD; ++x) row[x] = input_row[left[x]];
        for (int x = 0; x < right_count; ++x) row[SARONIC_DEBAYER_PAD + width + x] = input_row[right[x]];
    }
    for (int y = 0; y < raw_padded_height; ++y) {
        if (y >= SARONIC_DEBAYER_PAD && y < SARONIC_DEBAYER_PAD + height) continue;
        const int source_y = SARONIC_DEBAYER_PAD + ReflectBayerCoordinate(y - SARONIC_DEBAYER_PAD, height);
        std::memcpy(raw_padded_data + static_cast<size_t>(y) * raw_padded_pitch,
            raw_padded_data + static_cast<size_t>(source_y) * raw_padded_pitch,
            static_cast<size_t>(raw_padded_pitch));
    }
}

int Debayer::Process(const raw_image_t* input, bgr_image_t* output)
{
    return ProcessImpl(input, output, 0);
}

int Debayer::ProcessImpl(const raw_image_t* input, bgr_image_t* output, int diagnostic)
{
    std::lock_guard<std::mutex> processing_lock(process_mutex);
    if (!input || !output || !input->raw_data || !output->bgr_data) return -1;
    if (input->width < 2 || input->height < 2 ||
        input->width != output->width || input->height != output->height ||
        input->width > std::numeric_limits<int>::max() / 3) return -2;
    const int input_pitch = input->pitch ? input->pitch : input->width;
    const int output_pitch = output->pitch ? output->pitch : output->width * 3;
    if (input_pitch < input->width || output_pitch < output->width * 3) return -2;
    if (input->algorithm < SARONIC_DEBAYER_BILINEAR || input->algorithm > SARONIC_DEBAYER_SOFTMENON ||
        (input->format != SARONIC_DEBAYER_BGGR && input->format != SARONIC_DEBAYER_RGGB))
        return -4;
    if (!Allocate(input->width, input->height)) return -3;

    const bool rggb = input->format == SARONIC_DEBAYER_RGGB;
    const int algorithm = input->algorithm;
    PadRaw(input->raw_data, input_pitch);
    const uint8_t* raw_origin = raw_padded_data + SARONIC_DEBAYER_PAD * raw_padded_pitch + SARONIC_DEBAYER_PAD;
    uint8_t* bgr_origin = bgr_padded_data + SARONIC_DEBAYER_PAD * bgr_padded_pitch + SARONIC_DEBAYER_PAD * 3;
    try {
        if (!diagnostic && algorithm == SARONIC_DEBAYER_BILINEAR) {
            RunRows(height, 1, [=](int begin, int end) {
                bilinear_cpu_rows(raw_origin, raw_padded_pitch, output->bgr_data, output_pitch,
                    width, height, rggb, begin, end);
            });
        } else if (!diagnostic && algorithm == SARONIC_DEBAYER_MALVAR2004) {
            RunRows(height, 1, [=](int begin, int end) {
                malvar2004_cpu_rows(raw_origin, raw_padded_pitch, output->bgr_data, output_pitch,
                    width, height, rggb, begin, end);
            });
        } else if (!diagnostic && algorithm == SARONIC_DEBAYER_MENON2007) {
            if (!AllocateMenonScratch()) return -3;
            const size_t pixels = static_cast<size_t>(width) * height;
            const libdebayer_menon2007::Buffers buffers{
                raw_origin, static_cast<size_t>(raw_padded_pitch),
                output->bgr_data, static_cast<size_t>(output_pitch), width, height, rggb,
                menon_planes.get(), menon_planes.get() + pixels,
                menon_planes.get() + pixels * 2, menon_planes.get() + pixels * 5,
                menon_direction.get()};
            // Each full-image barrier publishes the previous stage's planes.
            // Name each stage directly so the scalar core can inline its stencil.
#define MENON_STAGE(name) \
            RunRows(height, 1, [=](int begin, int end) { \
                for (int y = begin; y < end; ++y) \
                    for (int x = 0; x < width; ++x) \
                        libdebayer_menon2007::name(buffers, x, y); \
            })
            MENON_STAGE(stage_green);
            MENON_STAGE(stage_decision);
            MENON_STAGE(stage_colors_at_green);
            MENON_STAGE(stage_opposite);
            MENON_STAGE(stage_refine_green);
            MENON_STAGE(stage_refine_colors_at_green);
            MENON_STAGE(stage_refine_opposite_output);
#undef MENON_STAGE
        } else {
            const bool soft = diagnostic != 1;
            if (soft && !AllocateSoftScratch()) return -3;
            const int work_width = (width + 1) & ~1;
            const int work_height = (height + 1) & ~1;
            const int green_origin = SARONIC_DEBAYER_PAD - 2;
            RunRows(work_height + 4, 2, [=](int begin, int end) {
                const uint8_t* raw = raw_padded_data + (green_origin + begin) * raw_padded_pitch + green_origin;
                uint8_t* bgr = bgr_padded_data + (green_origin + begin) * bgr_padded_pitch + green_origin * 3;
                if (soft)
                    bggr_softmenon_g_cpu(raw, raw_padded_pitch, bgr, bgr_padded_pitch, work_width + 4, end - begin);
                else
                    bggr_menon2007_g_cpu(raw, raw_padded_pitch, bgr, bgr_padded_pitch, work_width + 4, end - begin);
            });
            uint8_t* completed = soft ? soft_colors.get() : bgr_origin;
            const int completed_pitch = soft ? width * 3 : bgr_padded_pitch;
            if (soft) {
                // Immutable source lets SIMD load whole interleaved pixels
                // without reading bytes another task is concurrently writing.
                RunRows(height, 1, [=](int begin, int end) {
                    bggr_softmenon_rb_rows(bgr_origin, bgr_padded_pitch,
                        completed, completed_pitch, width, height, begin, end);
                });
            } else {
                RunRows(work_height, 2, [=](int begin, int end) {
                    bggr_menon2007_rb_cpu(bgr_origin + begin * bgr_padded_pitch,
                        bgr_padded_pitch, work_width, end - begin);
                });
            }
            if (!diagnostic) {
                const int variant = median_cpu_variant_supported(2) ? 2 : 1;
                RunRows(height, 1, [=](int begin, int end) {
                    median_cpu_rows(completed, completed_pitch, output->bgr_data, output_pitch,
                        width, height, false, begin, end, variant, rggb);
                });
            } else {
                RunRows(height, 1, [=](int begin, int end) {
                    for (int y = begin; y < end; ++y) {
                        const uint8_t* source = completed + static_cast<size_t>(y) * completed_pitch;
                        uint8_t* destination = output->bgr_data + static_cast<size_t>(y) * output_pitch;
                        if (!rggb) std::memcpy(destination, source, static_cast<size_t>(width) * 3);
                        else for (int x = 0; x < width; ++x) {
                            destination[3 * x] = source[3 * x + 2];
                            destination[3 * x + 1] = source[3 * x + 1];
                            destination[3 * x + 2] = source[3 * x];
                        }
                    }
                });
            }
        }
    } catch (...) {
        // Queued work may still reference this instance and its allocations.
        if (thread_pool) thread_pool->WaitAll();
        return -3;
    }
    return 0;
}
