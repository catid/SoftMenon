// Minimal C ABI for production CPU timing/validation. Algorithm IDs are
// 1=Bilinear, 2=Malvar2004, 3=paper Menon2007, 4=SoftMenon; patterns 1=RGGB, 2=BGGR.
// Build against cpu/cpu_debayer.cpp, cpu/cpu_kernel.cpp and cpu/threadpool.cpp.
#include "../cpu/cpu_debayer.hpp"

#if defined(_WIN32)
#define BENCH_EXPORT extern "C" __declspec(dllexport)
#else
#define BENCH_EXPORT extern "C" __attribute__((visibility("default")))
#endif

struct DebayerBenchmarkAccess {
    static int Process(Debayer& debayer, const raw_image_t* input,
                       bgr_image_t* output, int diagnostic) {
        return debayer.ProcessImpl(input, output, diagnostic);
    }
};

BENCH_EXPORT void* bench_create(int workers)
{
    if (workers < 0 || workers > 256) return nullptr;
    try { return new Debayer(static_cast<size_t>(workers)); }
    catch (...) { return nullptr; }
}

BENCH_EXPORT void bench_destroy(void* context)
{
    delete static_cast<Debayer*>(context);
}

BENCH_EXPORT int bench_workers(void* context)
{
    return context ? static_cast<int>(static_cast<Debayer*>(context)->WorkerCount()) : 0;
}

BENCH_EXPORT int bench_process(void* context, const uint8_t* raw, uint8_t* bgr,
    int width, int height, int raw_pitch, int bgr_pitch, int algorithm, int pattern)
{
    if (!context) return -1;
    if (algorithm < 1 || algorithm > 4 || pattern < 1 || pattern > 2) return -4;
    raw_image_t input{width, height, const_cast<uint8_t*>(raw), raw_pitch, algorithm - 1, pattern - 1};
    bgr_image_t output{width, height, bgr, bgr_pitch};
    return static_cast<Debayer*>(context)->Process(&input, &output);
}

// Diagnostic modes retain the native custom implementation (1) and the same
// implementation with soft green but without chroma cleanup (2). Both use the
// production allocator/padding/scheduler and never alter public algorithm IDs.
BENCH_EXPORT int bench_diagnostic(void* context, const uint8_t* raw, uint8_t* bgr,
    int width, int height, int raw_pitch, int bgr_pitch, int mode, int pattern)
{
    if (!context) return -1;
    if (mode < 1 || mode > 2 || pattern < 1 || pattern > 2) return -4;
    raw_image_t input{width, height, const_cast<uint8_t*>(raw), raw_pitch,
        SARONIC_DEBAYER_SOFTMENON, pattern - 1};
    bgr_image_t output{width, height, bgr, bgr_pitch};
    return DebayerBenchmarkAccess::Process(*static_cast<Debayer*>(context), &input, &output, mode);
}
