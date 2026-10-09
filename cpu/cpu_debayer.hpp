/** CPU demosaicing of 8-bit RGGB/BGGR images into interleaved BGR. */
#ifndef DEBAYER_CPP_H
#define DEBAYER_CPP_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>

#include "cpu_kernel.hpp"
#include "threadpool.hpp"

constexpr int SARONIC_DEBAYER_PAD = 4;
constexpr int KERNEL_BLOCK_SIZE = 16;

enum DebayerAlgorithm {
    SARONIC_DEBAYER_BILINEAR = 0,
    SARONIC_DEBAYER_MALVAR2004 = 1,
    SARONIC_DEBAYER_MENON2007 = 2,
    SARONIC_DEBAYER_SOFTMENON = 3
};

enum BayerFormat {
    SARONIC_DEBAYER_RGGB = 0,
    SARONIC_DEBAYER_BGGR = 1
};

struct raw_image_t {
    int width = -1;
    int height = -1;
    uint8_t* raw_data = nullptr;
    int pitch = 0;       // Bytes per row; zero means width.
    int algorithm = 0;
    int format = 0;
};

struct bgr_image_t {
    int width = -1;
    int height = -1;
    uint8_t* bgr_data = nullptr;
    int pitch = 0;       // Bytes per row; zero means 3 * width.
};

class Debayer {
public:
    // Zero selects min(hardware_concurrency(), 8), or 4 if unavailable.
    // Count includes the calling thread; each instance owns its helper pool.
    // Allocations are reused across frames.
    explicit Debayer(size_t workers = 0);
    ~Debayer();
    Debayer(const Debayer&) = delete;
    Debayer& operator=(const Debayer&) = delete;
    size_t WorkerCount() const noexcept { return worker_count; }

    // Explicit allocation is optional. A failed allocation preserves existing
    // valid storage. Allocate/Free must not run concurrently with Process.
    bool Allocate(int width, int height);
    void Free();

    // Dimensions >= 2, odd dimensions and independent positive pitches are
    // supported. Interpolation borders use phase-preserving REFLECT_101; the
    // paper Menon classifier uses a zero gradient halo. Calls on the
    // same instance serialize; independent instances have independent workers.
    // Returns 0 on success, -1 for null buffers, -2 for invalid sizes/pitches,
    // -3 for allocation/task submission failure, -4 for unsupported options.
    int Process(const raw_image_t* input, bgr_image_t* output);

private:
    friend struct DebayerBenchmarkAccess; // Internal validation controls.
    bool AllocateMenonScratch();
    bool AllocateSoftScratch();
    int ProcessImpl(const raw_image_t* input, bgr_image_t* output, int diagnostic);
    void InitializeThreadPool();
    void RunRows(int rows, int alignment, const std::function<void(int, int)>& task);
    void PadRaw(const uint8_t* source, int source_pitch);

    std::mutex process_mutex;
    const size_t worker_count;
    std::unique_ptr<ThreadPool> thread_pool;

    uint8_t* raw_padded_data = nullptr;
    int raw_padded_pitch = -1;
    int raw_padded_width = -1;
    int raw_padded_height = -1;
    uint8_t* bgr_padded_data = nullptr;
    int bgr_padded_pitch = -1;
    int bgr_padded_width = -1;
    int bgr_padded_height = -1;
    int width = -1;
    int height = -1;
    std::unique_ptr<uint8_t[]> soft_colors;
    std::unique_ptr<int32_t[]> menon_planes;
    std::unique_ptr<uint8_t[]> menon_direction;
};

#endif
