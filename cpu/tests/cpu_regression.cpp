#include "cpu_debayer.hpp"

#include <atomic>
#include <chrono>
#include <climits>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
void Require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

int MeasuredChannel(int format, int x, int y)
{
    if ((x + y) & 1) return 1;
    const bool blue = ((y & 1) == 0) == (format == SARONIC_DEBAYER_BGGR);
    return blue ? 0 : 2;
}

struct ImageCase {
    int width, height, input_pitch, output_pitch, format;
    std::vector<uint8_t> raw, output;
    static constexpr size_t guard = 32;
    ImageCase(int w, int h, int phase)
        : width(w), height(h), input_pitch(w + 7), output_pitch(3 * w + 11), format(phase),
          raw(guard + static_cast<size_t>(input_pitch) * h + guard, 0xcd),
          output(guard + static_cast<size_t>(output_pitch) * h + guard, 0xa5) {}
    uint8_t& Sample(int x, int y) { return raw[guard + y * input_pitch + x]; }
    uint8_t Pixel(int x, int y, int channel) const { return output[guard + y * output_pitch + 3 * x + channel]; }
    int Run(Debayer& debayer, int algorithm = SARONIC_DEBAYER_SOFTMENON) {
        raw_image_t input{width, height, raw.data() + guard, input_pitch, algorithm, format};
        bgr_image_t destination{width, height, output.data() + guard, output_pitch};
        return debayer.Process(&input, &destination);
    }
    void CheckGuards() const {
        for (size_t i = 0; i < guard; ++i) {
            Require(output[i] == 0xa5, "output prefix guard overwritten");
            Require(output[output.size() - i - 1] == 0xa5, "output suffix guard overwritten");
        }
        for (int y = 0; y < height; ++y)
            for (int x = width * 3; x < output_pitch; ++x)
                Require(output[guard + y * output_pitch + x] == 0xa5, "output row padding overwritten");
    }
};

void ConstantCase(int width, int height, int phase, Debayer& debayer, int algorithm = SARONIC_DEBAYER_SOFTMENON)
{
    ImageCase image(width, height, phase);
    const uint8_t color[] = {11, 87, 211};
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            image.Sample(x, y) = color[MeasuredChannel(phase, x, y)];
    const auto original = image.raw;
    Require(image.Run(debayer, algorithm) == 0, "constant-image Process failed");
    Require(image.raw == original, "input was modified");
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            for (int channel = 0; channel < 3; ++channel)
                Require(image.Pixel(x, y, channel) == color[channel], "constant color changed at image edge or tile boundary");
    image.CheckGuards();
}

void ConcurrentInitialization()
{
    constexpr int count = 8;
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> errors(count);
    for (int i = 0; i < count; ++i) {
        threads.emplace_back([&, i] {
            ++ready;
            while (!start.load()) std::this_thread::yield();
            try {
                Debayer debayer;
                ConstantCase(65 + 2 * i, 67 + 2 * i, i % 2, debayer);
            } catch (...) { errors[i] = std::current_exception(); }
        });
    }
    while (ready.load() != count) std::this_thread::yield();
    start = true;
    for (auto& thread : threads) thread.join();
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
}

void ConstantAndStridedImages()
{
    const int sizes[][2] = {{2,2}, {2,3}, {3,2}, {3,3}, {7,9}, {63,65}, {64,64}, {65,67}, {130,129}};
    Debayer debayer;
    for (int phase : {SARONIC_DEBAYER_RGGB, SARONIC_DEBAYER_BGGR})
        for (int algorithm = SARONIC_DEBAYER_BILINEAR; algorithm <= SARONIC_DEBAYER_SOFTMENON; ++algorithm)
            for (const auto& size : sizes) ConstantCase(size[0], size[1], phase, debayer, algorithm);
}

void MeasuredSamplesAndLinearColors()
{
    Debayer debayer;
    for (int algorithm = SARONIC_DEBAYER_BILINEAR; algorithm <= SARONIC_DEBAYER_SOFTMENON; ++algorithm)
    for (int phase : {SARONIC_DEBAYER_RGGB, SARONIC_DEBAYER_BGGR}) {
        ImageCase random(131, 129, phase);
        uint32_t state = 0x914af32u;
        for (int y = 0; y < random.height; ++y)
            for (int x = 0; x < random.width; ++x) {
                state = state * 1664525u + 1013904223u;
                random.Sample(x, y) = static_cast<uint8_t>(state >> 24);
            }
        const auto original = random.raw;
        Require(random.Run(debayer, algorithm) == 0, "random image failed");
        for (int y = 0; y < random.height; ++y)
            for (int x = 0; x < random.width; ++x)
                Require(random.Pixel(x, y, MeasuredChannel(phase, x, y)) == random.Sample(x, y), "measured CFA sample changed");
        const auto first_output = random.output;
        Require(random.Run(debayer, algorithm) == 0 && random.output == first_output, "repeated output is nondeterministic");
        Require(random.raw == original, "random input modified");
        random.CheckGuards();

        // Each color is an affine plane. The unchanged interpolation arithmetic
        // must reconstruct it exactly wherever its full stencil is in the image.
        ImageCase linear(130, 129, phase);
        for (int y = 0; y < linear.height; ++y)
            for (int x = 0; x < linear.width; ++x) {
                const uint8_t color[] = {static_cast<uint8_t>(10 + x), static_cast<uint8_t>(30 + y), static_cast<uint8_t>(200 - x)};
                linear.Sample(x, y) = color[MeasuredChannel(phase, x, y)];
            }
        Require(linear.Run(debayer, algorithm) == 0, "affine-color image failed");
        for (int y = 12; y < linear.height - 12; ++y)
            for (int x = 12; x < linear.width - 12; ++x) {
                const int expected[] = {10 + x, 30 + y, 200 - x};
                for (int channel = 0; channel < 3; ++channel)
                    Require(linear.Pixel(x, y, channel) == expected[channel], "affine color changed across interpolation/tile boundary");
            }
    }
}

void InvalidDescriptors()
{
    Debayer debayer;
    std::vector<uint8_t> raw(16, 80), output(48, 0xa5);
    raw_image_t input{4,4,raw.data(),4,SARONIC_DEBAYER_MENON2007,SARONIC_DEBAYER_BGGR};
    bgr_image_t destination{4,4,output.data(),12};
    Require(debayer.Process(nullptr, &destination) != 0, "null image accepted");
    Require(debayer.Process(&input, nullptr) != 0, "null output image accepted");
    auto bad = input; bad.raw_data = nullptr;
    Require(debayer.Process(&bad, &destination) != 0, "null raw buffer accepted");
    auto bad_output = destination; bad_output.bgr_data = nullptr;
    Require(debayer.Process(&input, &bad_output) != 0, "null BGR buffer accepted");
    for (int pitch : {-1, 1, 3}) {
        bad = input; bad.pitch = pitch;
        Require(debayer.Process(&bad, &destination) != 0, "short/negative raw stride accepted");
    }
    for (int pitch : {-1, 3, 11}) {
        bad_output = destination; bad_output.pitch = pitch;
        Require(debayer.Process(&input, &bad_output) != 0, "short/negative BGR stride accepted");
    }
    const int invalid_algorithms[] = {-1, SARONIC_DEBAYER_SOFTMENON + 1, 100};
    for (int algorithm : invalid_algorithms) {
        bad = input; bad.algorithm = algorithm;
        Require(debayer.Process(&bad, &destination) == -4, "unsupported algorithm silently substituted");
    }
    bad = input; bad.format = 100;
    Require(debayer.Process(&bad, &destination) == -4, "invalid CFA accepted");
    for (int size : {-1, 0, 1, INT_MAX}) {
        bad = input; bad.width = size;
        bad_output = destination; bad_output.width = size;
        Require(debayer.Process(&bad, &bad_output) != 0, "invalid width accepted");
        bad = input; bad.height = size;
        bad_output = destination; bad_output.height = size;
        Require(debayer.Process(&bad, &bad_output) != 0, "invalid height accepted");
        Require(!debayer.Allocate(size, size), "invalid allocation accepted");
    }
    Require(debayer.Allocate(4,4), "valid allocation failed after invalid requests");
    Require(!debayer.Allocate(INT_MAX,INT_MAX), "overflowing allocation accepted");
    Require(debayer.Allocate(4,4), "failed allocation corrupted valid state");
    for (uint8_t value : output) Require(value == 0xa5, "invalid descriptor wrote output");
    input.pitch = 0; destination.pitch = 0;
    Require(debayer.Process(&input, &destination) == 0, "tightly packed zero strides rejected");
}

void ConcurrentSameInstance()
{
    Debayer shared;
    std::vector<std::thread> workers;
    std::vector<std::exception_ptr> errors(4);
    for (int i = 0; i < 4; ++i) workers.emplace_back([&, i] {
        try {
            for (int repetition = 0; repetition < 8; ++repetition)
                ConstantCase(65 + 2 * i, 61 + 2 * i, i % 2, shared);
        } catch (...) { errors[i] = std::current_exception(); }
    });
    for (auto& worker : workers) worker.join();
    for (auto& error : errors) if (error) std::rethrow_exception(error);
}

void WorkerCountsAndAlgorithmSwitches()
{
    Debayer defaults;
    Require(defaults.WorkerCount() >= 1 && defaults.WorkerCount() <= 8,
        "default worker count exceeds its documented cap");
    Debayer serial(1), parallel(4);
    Require(serial.WorkerCount() == 1 && parallel.WorkerCount() == 4,
        "explicit worker count ignored");
    // Alternate dimensions and algorithms on the same objects, then repeat at
    // the same size with different input to detect stale persistent scratch.
    for (int phase : {SARONIC_DEBAYER_RGGB, SARONIC_DEBAYER_BGGR}) {
        for (int iteration = 0; iteration < 8; ++iteration) {
            const int w = iteration < 4 ? 67 : 130;
            const int h = iteration < 4 ? 131 : 129;
            ImageCase first(w, h, phase), second(w, h, phase);
            uint32_t state = 1777 + iteration;
            for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
                state = state * 1664525u + 1013904223u;
                first.Sample(x, y) = second.Sample(x, y) = static_cast<uint8_t>(state >> 24);
            }
            for (int algorithm = SARONIC_DEBAYER_SOFTMENON; algorithm >= SARONIC_DEBAYER_BILINEAR; --algorithm) {
                Require(first.Run(serial, algorithm) == 0 && second.Run(parallel, algorithm) == 0,
                    "algorithm/size switch failed");
                Require(first.output == second.output, "worker count changed output bytes");
                first.CheckGuards(); second.CheckGuards();
            }
        }
    }
}

struct ThrowingCopy {
    bool* should_throw;
    explicit ThrowingCopy(bool* flag) : should_throw(flag) {}
    ThrowingCopy(const ThrowingCopy& other) : should_throw(other.should_throw) {
        if (*should_throw) throw std::runtime_error("intentional task copy failure");
    }
    void operator()() const {}
};

void ThreadPoolCompletion()
{
    ThreadPool pool(3);
    bool throw_copy = false;
    std::function<void()> task = ThrowingCopy(&throw_copy);
    throw_copy = true;
    bool caught = false;
    try { pool.Submit(task); } catch (const std::runtime_error&) { caught = true; }
    Require(caught, "throwing task copy did not propagate");
    pool.WaitAll(); // A failed enqueue must not leave a phantom pending task.

    std::vector<int> published(1000, 0);
    for (size_t i = 0; i < published.size(); ++i)
        pool.Submit([&, i] { published[i] = static_cast<int>(i + 1); });
    pool.WaitAll();
    for (size_t i = 0; i < published.size(); ++i)
        Require(published[i] == static_cast<int>(i + 1), "WaitAll returned before task publication");
    for (int i = 0; i < 2000; ++i) {
        int value = 0;
        pool.Submit([&] { value = 1; });
        pool.WaitAll();
        Require(value == 1, "lost wakeup or early WaitAll");
    }
    std::atomic<int> drained{0};
    {
        ThreadPool draining(2);
        for (int i = 0; i < 100; ++i) draining.Submit([&] { ++drained; });
    }
    Require(drained == 100, "destructor dropped queued tasks");
}
}

int main()
{
    static_assert(!std::is_copy_constructible<Debayer>::value, "Debayer must not shallow-copy owned buffers");
    try {
        ConcurrentInitialization(); // Must precede any other Process call.
        ConstantAndStridedImages();
        MeasuredSamplesAndLinearColors();
        InvalidDescriptors();
        ConcurrentSameInstance();
        WorkerCountsAndAlgorithmSwitches();
        ThreadPoolCompletion();
        std::cout << "CPU regression checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
