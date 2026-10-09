#include "debayer_cpp.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

static_assert(!std::is_copy_constructible<Debayer>::value, "CUDA ownership must not be copied");
static_assert(!std::is_copy_assignable<Debayer>::value, "CUDA ownership must not be copied");

static void require(bool condition, const char* message) {
    if (!condition) { std::cerr << message << '\n'; std::exit(1); }
}

class ReleasableDebayer : public Debayer {
public:
    using Debayer::Free;
};

int main() {
    ReleasableDebayer context;
    context.Free();
    context.Free();
    std::vector<uint8_t> input_data(64, 77), output_data(192, 0);
    raw_image_t input;
    input.raw_data = input_data.data(); input.width = input.height = 8;
    bgr_image_t output;
    output.bgr_data = output_data.data(); output.width = output.height = 8;
    require(context.Process(nullptr, &output) == -8, "null input accepted");
    require(context.Process(&input, nullptr) == -8, "null output accepted");
    input.pitch = 7;
    require(context.Process(&input, &output) == -8, "short input stride accepted");
    input.pitch = 0; output.pitch = -1;
    require(context.Process(&input, &output) == -8, "negative output stride accepted");
    output.pitch = 0; input.algorithm = 999;
    require(context.Process(&input, &output) == -6, "unknown algorithm accepted");
    input.algorithm = SARONIC_DEBAYER_BILINEAR; input.format = SARONIC_DEBAYER_GBRG;
    require(context.Process(&input, &output) == -7, "unsupported bilinear CFA accepted");
    input.algorithm = SARONIC_DEBAYER_MALVAR2004; input.format = SARONIC_DEBAYER_GRBG;
    require(context.Process(&input, &output) == -7, "unsupported Malvar CFA accepted");

    input.format = SARONIC_DEBAYER_RGGB;
    input.width = output.width = std::numeric_limits<int32_t>::max() / 3;
    input.height = output.height = std::numeric_limits<int32_t>::max();
    require(context.Process(&input, &output) == -2, "impossible allocation did not fail cleanly");
    require(context.Process(&input, &output) == -2, "failed dimensions were incorrectly cached");

    // Reuse, resize, explicit release, and retry across all supported lanes.
    for (int size : {8, 17, 2, 19, 8}) {
        const int in_pitch = size + 7, out_pitch = size * 3 + 11;
        input_data.assign(in_pitch * size, 77);
        output_data.assign(out_pitch * size, 0xA5);
        input.raw_data = input_data.data(); input.width = input.height = size; input.pitch = in_pitch;
        output.bgr_data = output_data.data(); output.width = output.height = size; output.pitch = out_pitch;
        for (int algorithm : {SARONIC_DEBAYER_BILINEAR, SARONIC_DEBAYER_MALVAR2004, SARONIC_DEBAYER_MENON2007, SARONIC_DEBAYER_SOFTMENON}) {
            input.algorithm = algorithm;
            for (int format : {SARONIC_DEBAYER_RGGB, SARONIC_DEBAYER_BGGR}) {
                input.format = format;
                require(context.Process(&input, &output) == 0, "valid processing failed");
                for (int y = 0; y < size; ++y) {
                    for (int x = 0; x < out_pitch; ++x) {
                        require(output_data[y * out_pitch + x] == (x < size * 3 ? 77 : 0xA5),
                                "pixel mismatch or output row guard overwritten");
                    }
                }
            }
        }
        context.Free();
        context.Free();
    }
    std::cout << "Wrapper validation, ownership, resize, CFA rejection, and pitched output passed\n";
}
