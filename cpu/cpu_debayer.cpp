#include "cpu_debayer.hpp"
#include "chroma_median.hpp"
#include "menon2007_cpu.hpp"

#include <algorithm>
#include <atomic>
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
    if (width == new_width && height == new_height && raw_padded_data)
        return true;

    const size_t raw_bytes = static_cast<size_t>(padded_width * padded_height);
    std::unique_ptr<uint8_t[]> raw(new (std::nothrow) uint8_t[raw_bytes]());
    if (!raw) return false;

    // Commit only after all allocations succeed, preserving prior valid state.
    Free();
    width = new_width;
    height = new_height;
    raw_padded_width = static_cast<int>(padded_width);
    raw_padded_height = static_cast<int>(padded_height);
    raw_padded_pitch = raw_padded_width;
    raw_padded_data = raw.release();
    return true;
}

void Debayer::Free()
{
    delete[] raw_padded_data;
    raw_padded_data = nullptr;
    raw_padded_pitch = raw_padded_width = raw_padded_height = 0;
    width = height = 0;
    menon_workspaces.clear();
}

bool Debayer::AllocateMenonScratch()
{
    const size_t tasks = std::min(worker_count, static_cast<size_t>((height + 31) / 32));
    if (menon_workspaces.size() == tasks) return true;
    std::vector<std::unique_ptr<menon2007_cpu::Workspace>> workspaces(tasks);
    for (auto& workspace : workspaces) {
        workspace.reset(new (std::nothrow) menon2007_cpu::Workspace);
        if (!workspace) return false;
    }
    menon_workspaces = std::move(workspaces);
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
    try {
        if (algorithm == SARONIC_DEBAYER_BILINEAR) {
            RunRows(height, 1, [=](int begin, int end) {
                bilinear_cpu_rows(raw_origin, raw_padded_pitch, output->bgr_data, output_pitch,
                    width, height, rggb, begin, end);
            });
        } else if (algorithm == SARONIC_DEBAYER_MALVAR2004) {
            RunRows(height, 1, [=](int begin, int end) {
                malvar2004_cpu_rows(raw_origin, raw_padded_pitch, output->bgr_data, output_pitch,
                    width, height, rggb, begin, end);
            });
        } else if (algorithm == SARONIC_DEBAYER_MENON2007) {
            if (!AllocateMenonScratch()) return -3;
            // This partition is identical to RunRows(height, 1). Each task
            // reuses one private tile workspace and publishes disjoint rows.
            const int task_rows = static_cast<int>(
                (height + menon_workspaces.size() - 1) / menon_workspaces.size());
            RunRows(height, 1, [=](int begin, int end) {
                menon2007_cpu::process_rows(*menon_workspaces[begin / task_rows], raw_origin,
                    static_cast<size_t>(raw_padded_pitch), output->bgr_data, static_cast<size_t>(output_pitch),
                    width, height, rggb, begin, end);
            });
        } else {
            const int work_width = (width + 1) & ~1;
            const int work_height = (height + 1) & ~1;
            // Complete a small strip while both intermediates fit private cache.
            // Strips start on an even CFA row. Duplicate halo computation avoids
            // intermediate synchronization and preserves the exact stage order.
            const int variant = median_cpu_variant_supported(2) ? 2 : 1;
            std::atomic<bool> tile_failed{false};
            // The referenced failure flag must outlive queued callbacks,
            // including when submission or the caller task throws.
            try {
                RunRows(work_height, 2, [=, &tile_failed](int task_begin, int task_end) {
                    try {
                        const int gp = ((work_width + 4) * 3 + 63) & ~63;
                        const int cp = width * 3;
                        thread_local std::vector<uint8_t> green_tile, color_tile;
                        green_tile.resize(static_cast<size_t>(64 + 4) * gp);
                        color_tile.resize(static_cast<size_t>(64 + 4) * cp);
                        for (int begin = task_begin; begin < task_end; begin += 64) {
                            const int end = std::min(task_end, begin + 64);
                            const int green_first = begin - 2;
                            bggr_softmenon_g_cpu(
                                raw_origin + static_cast<ptrdiff_t>(green_first) * raw_padded_pitch - 2,
                                raw_padded_pitch, green_tile.data(), gp, work_width + 4, end - begin + 4);
                            const int rb_begin = std::max(0, begin - 1) - green_first;
                            const int rb_end = std::min(height, end + 1) - green_first;
                            bggr_softmenon_rb_rows(green_tile.data() + 6, gp, color_tile.data(), cp,
                                width, rb_end, rb_begin, rb_end);
                            const int source_first = begin == 0 ? 0 : green_first;
                            const int skip = source_first - green_first;
                            median_cpu_rows(color_tile.data() + static_cast<size_t>(skip) * cp, cp,
                                output->bgr_data + static_cast<size_t>(source_first) * output_pitch,
                                output_pitch, width, std::min(height, end + 1) - source_first, false,
                                begin - source_first, std::min(height, end) - source_first, variant, rggb);
                        }
                    } catch (...) {
                        tile_failed.store(true, std::memory_order_release);
                    }
                });
            } catch (...) {
                if (thread_pool) thread_pool->WaitAll();
                throw;
            }
            return tile_failed.load(std::memory_order_acquire) ? -3 : 0;
        }

    } catch (...) {
        // Queued work may still reference this instance and its allocations.
        if (thread_pool) thread_pool->WaitAll();
        return -3;
    }
    return 0;
}
