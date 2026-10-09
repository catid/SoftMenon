// Optional reference: g++ -O3 -shared -fPIC bridge_opencv.cpp \
//   $(pkg-config --cflags --libs opencv4) -o opencv.so
// IDs: 1=Bilinear, 2=edge-aware, 3=VNG; CFA 1=RGGB, 2=BGGR; output=BGR.
// Timed calls include host REFLECT_101 halo4, OpenCV, and crop copy. Odd sizes
// gain one extra reflected trailing row/column. Persistent buffers are reused.
// VNG startup ramp probe detects the OpenCV4.6 two-row registration defect;
// crop compensates that offset without changing reconstructed samples.
// OpenCV thread count is process-global: benchmark one context/setting at a time.
#include "external_padding.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <mutex>
#include <thread>

namespace {
std::mutex opencv_mutex;
struct Context { int workers; std::vector<uint8_t> padded; cv::Mat result; int vng_offset=-1; };
int vng_offset() {
    cv::Mat raw(48,48,CV_8UC1),result;
    for(int y=0;y<48;++y)for(int x=0;x<48;++x)raw.at<uint8_t>(y,x)=static_cast<uint8_t>(2*y+x);
    cv::demosaicing(raw,result,cv::COLOR_BayerRGGB2BGR_VNG);
    for(int offset : {0,2}) {
        bool matched=true;
        for(int y=8;y<32;++y)for(int x=8;x<32;++x)for(int channel=0;channel<3;++channel)
            matched &= result.at<cv::Vec3b>(y,x)[channel]==raw.at<uint8_t>(y+offset,x);
        if(matched)return offset;
    }
    return -1; // Unknown registration: decline VNG instead of guessing.
}
}
EXTERNAL_EXPORT void* bench_create(int workers) {
    if(workers<0 || workers>256)return nullptr;
    if(!workers)workers=std::max(1,std::min(8,static_cast<int>(std::thread::hardware_concurrency())));
    try {
        std::lock_guard<std::mutex> lock(opencv_mutex);
        Context* c=new Context{workers,{},cv::Mat(),-1};
        try {c->vng_offset=vng_offset();}catch(...){}
        return c;
    }catch(...){return nullptr;}
}
EXTERNAL_EXPORT void bench_destroy(void* p){delete static_cast<Context*>(p);}
EXTERNAL_EXPORT int bench_workers(void* p){return p ? static_cast<Context*>(p)->workers : 0;}
EXTERNAL_EXPORT int bench_vng_row_offset(void* p){return p ? static_cast<Context*>(p)->vng_offset : -1;}
EXTERNAL_EXPORT int bench_process(void* p,const uint8_t* raw,uint8_t* bgr,
        int w,int h,int rp,int bp,int algorithm,int pattern) {
    if(!p || !raw || !bgr)return -1;
    if(algorithm<1 || algorithm>3 || pattern<1 || pattern>2)return -4;
    external_benchmark::Geometry g;
    if(!external_benchmark::geometry(w,h,rp,bp,g))return -2;
    std::lock_guard<std::mutex> lock(opencv_mutex);
    auto& c=*static_cast<Context*>(p);
    if(algorithm==3 && c.vng_offset<0)return -4;
    try {
        if(cv::getNumThreads()!=c.workers)cv::setNumThreads(c.workers);
        external_benchmark::pad(raw,g,c.padded);
        const int codes[3][2]={{cv::COLOR_BayerRGGB2BGR,cv::COLOR_BayerBGGR2BGR},
            {cv::COLOR_BayerRGGB2BGR_EA,cv::COLOR_BayerBGGR2BGR_EA},
            {cv::COLOR_BayerRGGB2BGR_VNG,cv::COLOR_BayerBGGR2BGR_VNG}};
        const cv::Mat input(g.padded_height,g.padded_width,CV_8UC1,c.padded.data());
        cv::demosaicing(input,c.result,codes[algorithm-1][pattern-1]);
        const int crop_y=external_benchmark::halo-(algorithm==3 ? c.vng_offset : 0);
        for(int y=0;y<h;++y)std::memcpy(bgr+static_cast<size_t>(y)*g.bgr_pitch,
            c.result.ptr(y+crop_y)+external_benchmark::halo*3,static_cast<size_t>(w)*3);
        return 0;
    } catch(...) {return -3;}
}
EXTERNAL_EXPORT int bench_diagnostic(void*,const uint8_t*,uint8_t*,int,int,int,int,int,int){return -4;}
