#ifndef LIBDEBAYER_BENCHMARK_METRICS_H
#define LIBDEBAYER_BENCHMARK_METRICS_H
#include <opencv2/core.hpp>
#include <cmath>
#include <limits>

inline void validateImages(const cv::Mat& original, const cv::Mat& processed) {
    CV_Assert(!original.empty() && original.type() == CV_8UC3 && processed.type() == CV_8UC3);
    CV_Assert(original.size() == processed.size());
}

inline double psnrFromMse(double mse) {
    return mse == 0.0 ? std::numeric_limits<double>::infinity() : 10.0 * std::log10(255.0 * 255.0 / mse);
}

inline double calculatePSNR(const cv::Mat& original, const cv::Mat& processed) {
    validateImages(original, processed);
    return psnrFromMse(cv::norm(original, processed, cv::NORM_L2SQR) / (original.total() * 3.0));
}

// Missing green samples at R/B sites of RGGB or BGGR, with a four-pixel crop.
inline double calculateGreenPSNRAtRedBlue(const cv::Mat& original, const cv::Mat& processed) {
    validateImages(original, processed);
    constexpr int border = 4;
    CV_Assert(original.rows > 2 * border && original.cols > 2 * border);
    double sum = 0.0;
    size_t count = 0;
    for (int y = border; y < original.rows - border; ++y) {
        for (int x = border; x < original.cols - border; ++x) {
            if ((x & 1) != (y & 1)) continue;
            const double delta = static_cast<double>(original.at<cv::Vec3b>(y, x)[1]) - processed.at<cv::Vec3b>(y, x)[1];
            sum += delta * delta;
            ++count;
        }
    }
    CV_Assert(count > 0);
    return psnrFromMse(sum / count);
}
#endif
