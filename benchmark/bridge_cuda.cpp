// Same small benchmark ABI as bridge_cpu.cpp; classes stay local to each DSO.
#include "../cpp/include/debayer_cpp.h"
#define EXPORT extern "C" __attribute__((visibility("default")))
struct DebayerBenchmarkAccess {
    static int call(Debayer& d, const raw_image_t* in, const bgr_image_t* out, int diagnostic) {
        return d.ProcessImpl(in, out, diagnostic);
    }
};
EXPORT void* bench_create(int) { try { return new Debayer; } catch (...) { return nullptr; } }
EXPORT void bench_destroy(void* context) { delete static_cast<Debayer*>(context); }
EXPORT int bench_workers(void*) { return 0; }
static int process(void* context, const uint8_t* raw, uint8_t* bgr, int w, int h,
    int rp, int bp, int algorithm, int pattern, int diagnostic) {
    if (!context) return -1;
    if (algorithm < 1 || algorithm > 4 || pattern < 1 || pattern > 2) return -4;
    raw_image_t input{raw, rp, w, h, pattern, algorithm};
    bgr_image_t output{bgr, bp, w, h};
    return DebayerBenchmarkAccess::call(*static_cast<Debayer*>(context), &input, &output, diagnostic);
}
EXPORT int bench_process(void* context, const uint8_t* raw, uint8_t* bgr, int w, int h,
    int rp, int bp, int algorithm, int pattern) {
    return process(context, raw, bgr, w, h, rp, bp, algorithm, pattern, 0);
}
EXPORT int bench_diagnostic(void* context, const uint8_t* raw, uint8_t* bgr, int w, int h,
    int rp, int bp, int mode, int pattern) {
    if (mode < 1 || mode > 2) return -4;
    return process(context, raw, bgr, w, h, rp, bp, SARONIC_DEBAYER_SOFTMENON, pattern, mode);
}
