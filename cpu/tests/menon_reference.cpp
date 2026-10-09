// Keep the expected-image path independent of CPU-specific optimizations.
// common/menon2007.hpp is the separately validated, immutable paper reference.
#include "cpu_debayer.hpp"
#include "menon2007_cpu.hpp"
#include "../../common/menon2007.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
namespace reference = libdebayer_menon2007;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct Image {
    int width, height, phase, raw_pitch, output_pitch;
    static constexpr size_t prefix = 37, suffix = 43;
    std::vector<uint8_t> raw, output;

    Image(int w, int h, int p, int padding)
        : width(w), height(h), phase(p), raw_pitch(w + padding), output_pitch(w * 3 + padding + 4),
          raw(prefix + static_cast<size_t>(raw_pitch) * h + suffix, 0xAC),
          output(prefix + static_cast<size_t>(output_pitch) * h + suffix, 0xDB) {}

    uint8_t* raw_data() { return raw.data() + prefix; }
    uint8_t* output_data() { return output.data() + prefix; }

    int measured(int x, int y) const {
        if ((x ^ y) & 1) return 1;
        return ((x & 1) == 0) == (phase == SARONIC_DEBAYER_RGGB) ? 2 : 0;
    }

    void fill(int kind, std::mt19937& rng) {
        for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
            uint8_t value;
            switch (kind) {
                case 0: value = 0; break;
                case 1: value = 255; break;
                case 2: value = ((x / 3 + y / 2) & 1) ? 255 : 0; break;
                case 3: value = static_cast<uint8_t>((17 * x + 31 * y) & 255); break;
                case 4: value = std::array<uint8_t, 3>{9, 103, 247}[measured(x, y)]; break;
                case 5: value = static_cast<uint8_t>((rng() & 1) ? 255 : 0); break;
                default: value = static_cast<uint8_t>(rng() >> 24); break;
            }
            raw_data()[static_cast<size_t>(y) * raw_pitch + x] = value;
        }
    }

    void reset_output() { std::fill(output.begin(), output.end(), 0xDB); }

    void verify(const std::vector<uint8_t>& expected, const std::vector<uint8_t>& before,
                const std::string& label) {
        require(raw == before, label + ": modified input or input guards");
        for (size_t i = 0; i < prefix; ++i) require(output[i] == 0xDB, label + ": output prefix guard");
        const size_t end = prefix + static_cast<size_t>(output_pitch) * height;
        for (size_t i = end; i < output.size(); ++i) require(output[i] == 0xDB, label + ": output suffix guard");
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const size_t actual_index = static_cast<size_t>(y) * output_pitch + 3 * x;
                const size_t reference_index = (static_cast<size_t>(y) * width + x) * 3;
                for (int c = 0; c < 3; ++c) {
                    const uint8_t actual = output_data()[actual_index + c];
                    if (actual != expected[reference_index + c]) {
                        std::ostringstream error;
                        error << label << ": reference mismatch at " << x << ',' << y << ',' << c
                              << " actual=" << int(actual) << " expected=" << int(expected[reference_index + c]);
                        throw std::runtime_error(error.str());
                    }
                }
                require(output_data()[actual_index + measured(x, y)] == raw_data()[static_cast<size_t>(y) * raw_pitch + x],
                        label + ": measured CFA sample changed");
            }
            for (int x = 3 * width; x < output_pitch; ++x)
                require(output_data()[static_cast<size_t>(y) * output_pitch + x] == 0xDB, label + ": output row guard");
        }
    }

    int process(Debayer& backend, int algorithm = SARONIC_DEBAYER_MENON2007) {
        raw_image_t input{width, height, raw_data(), raw_pitch, algorithm, phase};
        bgr_image_t destination{width, height, output_data(), output_pitch};
        return backend.Process(&input, &destination);
    }
};

std::vector<uint8_t> expected_image(Image& image) {
    const size_t pixels = static_cast<size_t>(image.width) * image.height;
    std::vector<int32_t> planes(pixels * 8, -193);
    std::vector<uint8_t> direction(pixels, 0xCC), output(pixels * 3, 0xA7);
    const reference::Buffers b{image.raw_data(), static_cast<size_t>(image.raw_pitch), output.data(),
        static_cast<size_t>(image.width) * 3, image.width, image.height,
        image.phase == SARONIC_DEBAYER_RGGB, planes.data(), planes.data() + pixels,
        planes.data() + 2 * pixels, planes.data() + 5 * pixels, direction.data()};
    const auto stage = [&](auto function) {
        for (int y = 0; y < image.height; ++y)
            for (int x = 0; x < image.width; ++x) function(b, x, y);
    };
    stage(reference::stage_green);
    stage(reference::stage_decision);
    stage(reference::stage_colors_at_green);
    stage(reference::stage_opposite);
    stage(reference::stage_refine_green);
    stage(reference::stage_refine_colors_at_green);
    stage(reference::stage_refine_opposite_output);
    return output;
}
}

int main() {
    try {
        std::mt19937 rng(20261009);
        const std::array<size_t, 4> counts{1, 2, 4, 7};
        std::array<std::unique_ptr<Debayer>, 4> backends;
        for (size_t i = 0; i < backends.size(); ++i) backends[i] = std::make_unique<Debayer>(counts[i]);
        auto workspace = std::make_unique<menon2007_cpu::Workspace>();
        using RowFunction = void (*)(menon2007_cpu::Workspace&, const uint8_t*, size_t, uint8_t*, size_t,
                                     int, int, bool, int, int);
        std::vector<std::pair<std::string, RowFunction>> row_functions{
            {"scalar", menon2007_cpu::scalar_rows}, {"runtime dispatch", menon2007_cpu::process_rows}};
#if MENON_CPU_X86
        if (__builtin_cpu_supports("avx2")) row_functions.emplace_back("avx2", menon2007_cpu::avx2_rows);
        if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl"))
            row_functions.emplace_back("avx512", menon2007_cpu::avx512_rows);
#endif
        size_t cases = 0, comparisons = 0, direct_comparisons = 0, pixels_checked = 0;
        const auto check = [&](int width, int height, int kind, int phase) {
            Image image(width, height, phase, 1 + 2 * static_cast<int>(cases % 9));
            image.fill(kind, rng);
            const auto before = image.raw;
            const auto expected = expected_image(image);
            const std::string label = std::to_string(width) + "x" + std::to_string(height) +
                " phase=" + std::to_string(phase) + " input=" + std::to_string(kind);
            for (size_t i = 0; i < backends.size(); ++i) {
                image.reset_output();
                require(image.process(*backends[i]) == 0, label + ": Process failed");
                image.verify(expected, before, label + " workers=" + std::to_string(counts[i]));
                ++comparisons;
                pixels_checked += static_cast<size_t>(width) * height;
            }
            // Alternate public algorithms on reused allocation, then return to
            // paper Menon and demand exact output rather than only repeatability.
            if (cases % 11 == 0) {
                require(image.process(*backends.back(), SARONIC_DEBAYER_SOFTMENON) == 0, label + ": algorithm switch failed");
                image.reset_output();
                require(image.process(*backends.back()) == 0, label + ": paper switch-back failed");
                image.verify(expected, before, label + " after algorithm switch");
                ++comparisons;
            }
            // Each direct implementation also receives reverse-order, odd row
            // slices. A slice must leave every other destination row untouched;
            // this exposes global-parity and artificial tile-boundary mistakes.
            std::vector<int> boundaries{0, 1, height / 3, height / 2 + 1, height - 1, height};
            for (auto& row : boundaries) row = std::clamp(row, 0, height);
            std::sort(boundaries.begin(), boundaries.end());
            boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
            for (const auto& implementation : row_functions) {
                image.reset_output();
                for (size_t split = boundaries.size() - 1; split > 0; --split) {
                    const int begin = boundaries[split - 1], end = boundaries[split];
                    const auto prior_output = image.output;
                    implementation.second(*workspace, image.raw_data(), image.raw_pitch, image.output_data(),
                        image.output_pitch, width, height, phase == SARONIC_DEBAYER_RGGB, begin, end);
                    for (int y = 0; y < height; ++y) if (y < begin || y >= end) {
                        const size_t offset = Image::prefix + static_cast<size_t>(y) * image.output_pitch;
                        require(std::equal(image.output.begin() + offset, image.output.begin() + offset + image.output_pitch,
                                           prior_output.begin() + offset), label + ": direct slice wrote outside requested rows");
                    }
                }
                image.verify(expected, before, label + " direct " + implementation.first);
                ++direct_comparisons;
            }
            ++cases;
        };
        const int tiny[][2] = {{2,2},{2,3},{3,2},{3,3},{4,5},{5,4},{7,9},{9,7},{15,17},{17,15}};
        for (int phase : {SARONIC_DEBAYER_RGGB, SARONIC_DEBAYER_BGGR}) {
            for (const auto& shape : tiny)
                for (int kind = 0; kind <= 6; ++kind) check(shape[0], shape[1], kind, phase);
            // Straddle 8/16/32-pixel SIMD, historical tile boundaries, and
            // the wrapper's uneven row partitions with independent strides.
            const int boundaries[][2] = {{31,33},{32,65},{33,64},{63,129},{64,127},{65,128},
                {127,63},{128,64},{129,65},{255,127},{256,128},{257,129}};
            for (const auto& shape : boundaries)
                for (int kind : {2,5,6}) check(shape[0], shape[1], kind, phase);
            for (int dx : {-1, 0, 1}) for (int dy : {-1, 0, 1})
                check(menon2007_cpu::core_width + dx, menon2007_cpu::core_height + dy, 6, phase);
            check(2 * menon2007_cpu::core_width + 1, 2 * menon2007_cpu::core_height + 1, 5, phase);
            for (int i = 0; i < 24; ++i) {
                // Sequence RNG calls explicitly: argument evaluation order
                // must not make GCC and Clang generate different test images.
                const int random_width = 2 + rng() % 171;
                const int random_height = 2 + rng() % 153;
                check(random_width, random_height, 6, phase);
            }
            for (const auto& shape : {std::array<int,2>{511,257}, {513,259}, {1025,131}, {1921,129}})
                check(shape[0], shape[1], 6, phase);
            check(1920, 1080, 6, phase);
            // Shrink after HD to expose stale workspace or allocation assumptions.
            check(3, 5, 5, phase);
        }
        std::cout << "Paper CPU reference: " << cases << " distinct cases, " << comparisons
                  << " exact public comparisons, " << direct_comparisons << " direct sliced comparisons, "
                  << pixels_checked << " public compared pixels\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Paper CPU reference failure: " << error.what() << '\n';
        return 1;
    }
}
