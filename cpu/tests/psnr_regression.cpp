// Exercise the metric used by the actual demo, rather than a test-only copy.
#define main debayer_cpu_demo_main
#include "../main.cpp"
#undef main

int main()
{
    const cv::Mat original(2, 3, CV_8UC3, cv::Scalar(0, 0, 0));
    if (!std::isinf(computePSNR(original, original))) return 1;
    const double expected = 10.0 * std::log10(255.0 * 255.0 / (100.0 / 3.0));
    for (int channel = 0; channel < 3; ++channel) {
        cv::Mat changed = original.clone();
        for (int y = 0; y < changed.rows; ++y)
            for (int x = 0; x < changed.cols; ++x)
                changed.at<cv::Vec3b>(y, x)[channel] = 10;
        if (std::abs(computePSNR(original, changed) - expected) > 1e-9) return 2;
    }
    if (computePSNR(cv::Mat(), cv::Mat()) >= 0.0) return 3;
    std::cout << "CPU PSNR checks passed\n";
    return 0;
}
