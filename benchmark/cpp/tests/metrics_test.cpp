#include "metrics.h"
#include <cstdlib>
#include <iostream>
static void check(bool value) { if (!value) { std::cerr << "Metric regression failed\n"; std::exit(1); } }
int main() {
    cv::Mat reference(10, 10, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int channel = 0; channel < 3; ++channel) {
        cv::Scalar offset(0, 0, 0); offset[channel] = 30;
        cv::Mat changed(10, 10, CV_8UC3, offset);
        check(std::abs(calculatePSNR(reference, changed) - psnrFromMse(300.0)) < 1e-10);
    }
    check(std::isinf(calculatePSNR(reference, reference)));
    cv::Mat changed = reference.clone();
    changed.at<cv::Vec3b>(4, 4)[1] = 12;
    check(std::abs(calculateGreenPSNRAtRedBlue(reference, changed) - psnrFromMse(72.0)) < 1e-10);
    changed.at<cv::Vec3b>(4, 5)[1] = 200;
    check(std::abs(calculateGreenPSNRAtRedBlue(reference, changed) - psnrFromMse(72.0)) < 1e-10);
    std::cout << "All-channel PSNR and missing-green mask regressions passed\n";
}
