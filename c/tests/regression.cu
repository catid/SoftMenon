#include "debayer.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Function = cudaError_t (*)(cudaStream_t, int32_t, int32_t, size_t, size_t, uint8_t*, uint8_t*);
struct Method { const char* name; Function run; bool bggr; int oracle; };
const Method methods[] = {
    {"rggb-bilinear", debayer_rggb2bgr_bilinear, false, 1},
    {"bggr-bilinear", debayer_bggr2bgr_bilinear, true, 1},
    {"rggb-malvar", debayer_rggb2bgr_malvar2004, false, 2},
    {"bggr-malvar", debayer_bggr2bgr_malvar2004, true, 2},
    {"rggb-menon", debayer_rggb2bgr_menon2007, false, 0},
    {"bggr-menon", debayer_bggr2bgr_menon2007, true, 0},
    {"rggb-softmenon", debayer_rggb2bgr_softmenon, false, 0},
    {"bggr-softmenon", debayer_bggr2bgr_softmenon, true, 0},
};
constexpr size_t guard = 64;
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
int reflected(int coordinate, int length) {
    while (coordinate < 0 || coordinate >= length)
        coordinate = coordinate < 0 ? -coordinate : 2 * length - 2 - coordinate;
    return coordinate;
}
int color(int x, int y, bool bggr) {
    if ((x + y) & 1) return 1;
    return ((x & 1) != bggr) ? 0 : 2;
}
int sample(const std::vector<uint8_t>& raw, int width, int height, int x, int y) {
    return raw[static_cast<size_t>(reflected(y, height)) * width + reflected(x, width)];
}
int expected(const std::vector<uint8_t>& raw, int width, int height, int x, int y,
             int channel, const Method& method) {
    const int native = color(x, y, method.bggr);
    if (channel == native) return sample(raw, width, height, x, y);
    if (method.oracle == 1) {
        int sum = 0, count = 0;
        for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
            if (color(x + dx, y + dy, method.bggr) == channel) {
                sum += sample(raw, width, height, x + dx, y + dy); ++count;
            }
        }
        return (sum + count / 2) / count;
    }
    // Independent published MHC filter matrices, in units of 1/16.
    static const int green[5][5] = {
        {0,0,-2,0,0}, {0,0,4,0,0}, {-2,4,8,4,-2}, {0,0,4,0,0}, {0,0,-2,0,0}};
    static const int opposite[5][5] = {
        {0,0,-3,0,0}, {0,4,0,4,0}, {-3,0,12,0,-3}, {0,4,0,4,0}, {0,0,-3,0,0}};
    static const int horizontal[5][5] = {
        {0,0,1,0,0}, {0,-2,0,-2,0}, {-2,8,10,8,-2}, {0,-2,0,-2,0}, {0,0,1,0,0}};
    int sum = 0;
    const bool horizontal_color = color(x - 1, y, method.bggr) == channel;
    for (int dy = -2; dy <= 2; ++dy) for (int dx = -2; dx <= 2; ++dx) {
        const int weight = channel == 1 ? green[dy + 2][dx + 2] :
            native != 1 ? opposite[dy + 2][dx + 2] :
            horizontal_color ? horizontal[dy + 2][dx + 2] : horizontal[dx + 2][dy + 2];
        sum += weight * sample(raw, width, height, x + dx, y + dy);
    }
    return std::max(0, std::min(255, (sum + 8) / 16));
}

void run_case(int width, int height, const Method& method, bool constant, bool uninitialized) {
    const size_t raw_pitch = width + 4 + 7;
    const size_t bgr_pitch = 3 * (width + 4) + 13;
    const size_t rows = height + 4;
    const size_t raw_bytes = raw_pitch * rows, bgr_bytes = bgr_pitch * rows;
    const std::array<uint8_t, 3> flat = {31, 117, 203};
    std::vector<uint8_t> raw(static_cast<size_t>(width) * height);
    uint32_t random = 123456789;
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        random ^= random << 13; random ^= random >> 17; random ^= random << 5;
        raw[static_cast<size_t>(y) * width + x] = constant ? flat[color(x,y,method.bggr)] : static_cast<uint8_t>(random);
    }
    std::vector<uint8_t> padded(raw_bytes + 2 * guard, 0xD3);
    for (int y = 0; y < height; ++y)
        std::copy_n(raw.data() + static_cast<size_t>(y) * width, width, padded.data() + guard + (y + 2) * raw_pitch + 2);
    uint8_t *device_raw = nullptr, *device_bgr = nullptr;
    cudaStream_t stream;
    check(cudaStreamCreate(&stream));
    check(cudaMalloc(&device_raw, padded.size()));
    check(cudaMalloc(&device_bgr, bgr_bytes + 2 * guard));
    check(cudaMemcpyAsync(device_raw, padded.data(), padded.size(), cudaMemcpyHostToDevice, stream));
    check(debayer_mirror_image(stream, width, height, raw_pitch, device_raw + guard));
    std::vector<uint8_t> reflected_raw(padded.size());
    check(cudaMemcpyAsync(reflected_raw.data(), device_raw, reflected_raw.size(), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    for (int y = 0; y < height + 4; ++y) for (int x = 0; x < width + 4; ++x)
        padded[guard + y * raw_pitch + x] = sample(raw, width, height, x - 2, y - 2);
    require(reflected_raw == padded, std::string(method.name) + ": mirror corrupted guard/pitch bytes or CFA phase");
    std::vector<uint8_t> first;
    for (int repeat = 0; repeat < (uninitialized ? 1 : 2); ++repeat) {
        const uint8_t poison = repeat ? 0x5B : 0xA7;
        if (uninitialized) {
            // Leave the interior uninitialized so compute-sanitizer initcheck
            // verifies that Menon never reads an unwritten channel or halo.
            check(cudaMemsetAsync(device_bgr, poison, guard, stream));
            check(cudaMemsetAsync(device_bgr + guard + bgr_bytes, poison, guard, stream));
            for (size_t y = 0; y < rows; ++y) {
                uint8_t* row = device_bgr + guard + y * bgr_pitch;
                if (y < 2 || y >= static_cast<size_t>(height) + 2) check(cudaMemsetAsync(row, poison, bgr_pitch, stream));
                else {
                    check(cudaMemsetAsync(row, poison, 6, stream));
                    check(cudaMemsetAsync(row + 6 + width * 3, poison, bgr_pitch - 6 - width * 3, stream));
                }
            }
        } else check(cudaMemsetAsync(device_bgr, poison, bgr_bytes + 2 * guard, stream));
        check(method.run(stream, width, height, raw_pitch, bgr_pitch, device_raw + guard, device_bgr + guard));
        std::vector<uint8_t> output(bgr_bytes + 2 * guard);
        check(cudaMemcpyAsync(output.data(), device_bgr, output.size(), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        for (size_t i = 0; i < output.size(); ++i) {
            bool interior = false;
            if (i >= guard && i < guard + bgr_bytes) {
                const size_t row = (i - guard) / bgr_pitch, column = (i - guard) % bgr_pitch;
                interior = row >= 2 && row < static_cast<size_t>(height) + 2 && column >= 6 && column < static_cast<size_t>(width) * 3 + 6;
            }
            if (!interior) require(output[i] == poison, std::string(method.name) + ": output guard, halo, or stride padding overwritten");
        }
        std::vector<uint8_t> result(static_cast<size_t>(width) * height * 3);
        for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) for (int c = 0; c < 3; ++c) {
            const int actual = output[guard + (y + 2) * bgr_pitch + (x + 2) * 3 + c];
            result[(static_cast<size_t>(y) * width + x) * 3 + c] = actual;
            if (c == color(x, y, method.bggr))
                require(actual == raw[static_cast<size_t>(y) * width + x], std::string(method.name) + ": measured sample changed");
            if (constant) require(actual == flat[c], std::string(method.name) + ": flat-color reconstruction failed at " + std::to_string(x) + "," + std::to_string(y));
            else if (method.oracle)
                require(actual == expected(raw,width,height,x,y,c,method), std::string(method.name) + ": CPU filter mismatch at " + std::to_string(x) + "," + std::to_string(y));
        }
        if (repeat) require(result == first, std::string(method.name) + ": result depends on prior output/halo contents");
        else first = std::move(result);
    }
    check(cudaFree(device_bgr)); check(cudaFree(device_raw)); check(cudaStreamDestroy(stream));
}

void invalid_arguments() {
    uint8_t* a = reinterpret_cast<uint8_t*>(uintptr_t(0x100000));
    uint8_t* b = reinterpret_cast<uint8_t*>(uintptr_t(0x200000));
    require(debayer_mirror_image(nullptr, 1, 2, 6, a) == cudaErrorInvalidValue, "minimum width validation");
    require(debayer_mirror_image(nullptr, 2, -1, 6, a) == cudaErrorInvalidValue, "negative height validation");
    require(debayer_mirror_image(nullptr, 2, 2, 5, a) == cudaErrorInvalidValue, "short input pitch validation");
    require(debayer_mirror_image(nullptr, 2, 2, SIZE_MAX, a) == cudaErrorInvalidValue, "overflow input pitch validation");
    require(debayer_mirror_image(nullptr, INT_MAX, 2, 6, a) == cudaErrorInvalidValue, "dimension overflow validation");
    require(debayer_mirror_image(nullptr, 2, 2, 6, nullptr) == cudaErrorInvalidValue, "null input validation");
    require(debayer_menon2007_workspace_size(1, 2) == 0, "paper workspace dimensions");
    require(debayer_softmenon_workspace_size(2, 1) == 0, "soft workspace dimensions");
    const size_t paper_bytes = debayer_menon2007_workspace_size(2, 2);
    const size_t soft_bytes = debayer_softmenon_workspace_size(2, 2);
    uint8_t* scratch = reinterpret_cast<uint8_t*>(uintptr_t(0x300000));
    for (bool soft : {false, true}) {
        const auto function = soft ? debayer_softmenon_with_workspace : debayer_menon2007_with_workspace;
        const size_t bytes = soft ? soft_bytes : paper_bytes;
        require(function(nullptr,2,2,6,18,a,b,1,scratch,bytes-1) == cudaErrorInvalidValue, "short workspace");
        require(function(nullptr,2,2,6,18,a,b,1,a,bytes) == cudaErrorInvalidValue, "input workspace overlap");
        require(function(nullptr,2,2,6,18,a,b,1,b,bytes) == cudaErrorInvalidValue, "output workspace overlap");
        require(function(nullptr,2,2,6,18,a,b,1,scratch+1,bytes) == cudaErrorInvalidValue, "workspace alignment");
        require(function(nullptr,2,2,6,18,a,b,2,scratch,bytes) == cudaErrorInvalidValue, "workspace CFA");
    }
    for (const auto& method : methods) {
        require(method.run(nullptr, 2, 2, 6, 17, a, b) == cudaErrorInvalidValue, "short output pitch validation");
        require(method.run(nullptr, 2, 2, 6, SIZE_MAX, a, b) == cudaErrorInvalidValue, "overflow output pitch validation");
        require(method.run(nullptr, 2, 2, 6, 18, a, nullptr) == cudaErrorInvalidValue, "null output validation");
        require(method.run(nullptr, 2, 2, 6, 18, a, a + 3) == cudaErrorInvalidValue, "overlap validation");
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const bool quick = argc > 1 && std::string(argv[1]) == "--quick";
        const bool uninitialized = argc > 1 && std::string(argv[1]) == "--uninitialized-output";
        invalid_arguments();
        const std::array<std::array<int,2>,12> sizes = {{{2,2},{2,3},{3,2},{3,3},{7,9},{16,16},{17,19},{18,18},{30,34},{127,129},{1920,1080},{1921,1081}}};
        int cases = 0;
        for (const auto& size : sizes) {
            if ((quick || uninitialized) && size[0] > 127) continue;
            for (const auto& method : methods) for (bool constant : {false,true}) {
                run_case(size[0],size[1],method,constant,uninitialized); ++cases;
            }
        }
        std::printf("Passed %d CUDA cases: bounds, phase reflection, CPU filters, samples, and halo independence.\n",cases);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Regression failure: %s\n", error.what());
        return 1;
    }
}
