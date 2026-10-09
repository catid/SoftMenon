#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "libdebayercpp/debayer_cpp.h"
#include "metrics.h"

namespace fs = std::filesystem;

static cv::Mat convertToBGGR(const cv::Mat& image) {
    CV_Assert(image.type() == CV_8UC3);
    cv::Mat bayer(image.rows, image.cols, CV_8UC1);
    for (int y = 0; y < image.rows; ++y)
        for (int x = 0; x < image.cols; ++x)
            bayer.at<uint8_t>(y, x) = image.at<cv::Vec3b>(y, x)[(y & 1) == (x & 1) ? ((y & 1) ? 2 : 0) : 1];
    return bayer;
}

int main() {
    try {
        const char* setting = std::getenv("KODAK_FOLDER_PATH");
        const fs::path folder = setting ? setting : "../../kodak";
        std::vector<fs::path> files;
        for (const auto& entry : fs::directory_iterator(folder)) {
            const std::string name = entry.path().filename().string();
            if (entry.is_regular_file() && name.rfind("kodim", 0) == 0 && entry.path().extension() == ".png" &&
                (name.size() < 8 || name.compare(name.size() - 8, 8, ".out.png") != 0)) files.push_back(entry.path());
        }
        std::sort(files.begin(), files.end());
        if (files.empty()) { std::cerr << "No Kodak PNG images found in " << folder << '\n'; return 1; }
        std::cout << "Menon2007 BGGR; full-image all-channel BGR PSNR; missing-green PSNR uses a 4-pixel crop\n";
        Debayer context;
        double psnr_sum = 0.0, green_sum = 0.0;
        size_t processed = 0;
        for (const auto& file : files) {
            cv::Mat original = cv::imread(file.string(), cv::IMREAD_COLOR);
            if (original.empty()) { std::cerr << "Could not read " << file << '\n'; return 1; }
            cv::Mat bayer = convertToBGGR(original), result(original.size(), CV_8UC3);
            CV_Assert(bayer.step[0] <= std::numeric_limits<int32_t>::max() && result.step[0] <= std::numeric_limits<int32_t>::max());
            raw_image_t input;
            input.raw_data = bayer.data; input.pitch = static_cast<int32_t>(bayer.step[0]);
            input.width = bayer.cols; input.height = bayer.rows;
            input.format = SARONIC_DEBAYER_BGGR; input.algorithm = SARONIC_DEBAYER_MENON2007;
            bgr_image_t output;
            output.bgr_data = result.data; output.pitch = static_cast<int32_t>(result.step[0]);
            output.width = result.cols; output.height = result.rows;
            // Warm this shape before reporting the synchronous host-to-host time.
            int status = context.Process(&input, &output);
            if (status != 0) { std::cerr << "Processing failed: " << status << '\n'; return 1; }
            const auto begin = std::chrono::steady_clock::now();
            status = context.Process(&input, &output);
            const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count();
            if (status != 0) { std::cerr << "Processing failed: " << status << '\n'; return 1; }
            const double psnr = calculatePSNR(original, result);
            const double green = calculateGreenPSNRAtRedBlue(original, result);
            psnr_sum += psnr; green_sum += green; ++processed;
            fs::path destination = file; destination.replace_extension(".out.png");
            if (!cv::imwrite(destination.string(), result)) { std::cerr << "Could not write " << destination << '\n'; return 1; }
            std::cout << file.filename() << ": PSNR " << psnr << " dB; missing-green PSNR " << green
                << " dB; warm host-to-host time " << micros << " us (copies + kernels + synchronization; allocation excluded)\n";
        }
        std::cout << "Arithmetic mean of " << processed << " image PSNRs: " << psnr_sum / processed
            << " dB; missing-green PSNR: " << green_sum / processed << " dB\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
    return 0;
}
