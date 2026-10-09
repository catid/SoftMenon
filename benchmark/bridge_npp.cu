// Optional reference: nvcc -O3 -shared -Xcompiler=-fPIC bridge_npp.cu \
//   -lnppicc -lnppc -o npp.so
// ID1=NPP CFAToRGB, CFA1=RGGB/2=BGGR; output=BGR. Installed NPP documentation
// requires NPPI_INTER_UNDEFINED (0), not CUBIC. The fixed method uses bilinear
// R/B interpolation and adaptive chroma-correlated green (nppi_color_conversion.h).
// Timed calls include host REFLECT_101 halo4, H2D, NPP, device crop/RGB->BGR,
// D2H, and stream completion. Even source dimensions include one extra reflected
// trailing row/column for odd inputs. All allocations are reused at stable size.
#include "external_padding.hpp"
#include <cuda_runtime.h>
#include <nppi_color_conversion.h>
#include <mutex>

namespace {
struct Context {
    std::mutex mutex;
    NppStreamContext npp{};
    std::vector<uint8_t> padded;
    uint8_t *raw=nullptr,*rgb=nullptr,*bgr=nullptr;
    int width=0,height=0;
    ~Context(){cudaSetDevice(npp.nCudaDeviceId);cudaFree(raw);cudaFree(rgb);cudaFree(bgr);if(npp.hStream)cudaStreamDestroy(npp.hStream);}
    bool allocate(const external_benchmark::Geometry& g) {
        if(width==g.width && height==g.height && raw && rgb && bgr)return true;
        uint8_t *a=nullptr,*b=nullptr,*c=nullptr;
        const size_t pixels=static_cast<size_t>(g.padded_width)*g.padded_height;
        if(cudaMalloc(&a,pixels)!=cudaSuccess || cudaMalloc(&b,pixels*3)!=cudaSuccess ||
           cudaMalloc(&c,static_cast<size_t>(g.width)*g.height*3)!=cudaSuccess) {
            cudaFree(a);cudaFree(b);cudaFree(c);return false;
        }
        cudaFree(raw);cudaFree(rgb);cudaFree(bgr);raw=a;rgb=b;bgr=c;width=g.width;height=g.height;return true;
    }
};
__global__ void crop_bgr(const uint8_t* source,uint8_t* destination,int width,int height,int source_pitch) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=width*height)return;
    const int x=i%width,y=i/width;
    const uint8_t* p=source+(y+external_benchmark::halo)*source_pitch+(x+external_benchmark::halo)*3;
    destination[i*3]=p[2];destination[i*3+1]=p[1];destination[i*3+2]=p[0];
}
}
EXTERNAL_EXPORT void* bench_create(int workers) {
    if(workers<0 || workers>256)return nullptr;
    Context* c=nullptr;
    try {
        c=new Context;
        cudaDeviceProp properties{};
        if(cudaGetDevice(&c->npp.nCudaDeviceId)!=cudaSuccess ||
           cudaGetDeviceProperties(&properties,c->npp.nCudaDeviceId)!=cudaSuccess ||
           cudaStreamCreateWithFlags(&c->npp.hStream,cudaStreamNonBlocking)!=cudaSuccess){delete c;return nullptr;}
        c->npp.nMultiProcessorCount=properties.multiProcessorCount;
        c->npp.nMaxThreadsPerMultiProcessor=properties.maxThreadsPerMultiProcessor;
        c->npp.nMaxThreadsPerBlock=properties.maxThreadsPerBlock;
        c->npp.nSharedMemPerBlock=properties.sharedMemPerBlock;
        c->npp.nCudaDevAttrComputeCapabilityMajor=properties.major;
        c->npp.nCudaDevAttrComputeCapabilityMinor=properties.minor;
        if(cudaStreamGetFlags(c->npp.hStream,&c->npp.nStreamFlags)!=cudaSuccess){delete c;return nullptr;}
        return c;
    }catch(...){delete c;return nullptr;}
}
EXTERNAL_EXPORT void bench_destroy(void* p){delete static_cast<Context*>(p);}
EXTERNAL_EXPORT int bench_workers(void*){return 0;}
EXTERNAL_EXPORT int bench_process(void* p,const uint8_t* raw,uint8_t* bgr,
        int w,int h,int rp,int bp,int algorithm,int pattern) {
    if(!p || !raw || !bgr)return -1;
    if(algorithm!=1 || pattern<1 || pattern>2)return -4;
    external_benchmark::Geometry g;
    if(!external_benchmark::geometry(w,h,rp,bp,g))return -2;
    auto& c=*static_cast<Context*>(p);std::lock_guard<std::mutex> lock(c.mutex);
    try {
        if(cudaSetDevice(c.npp.nCudaDeviceId)!=cudaSuccess || !c.allocate(g))return -3;
        external_benchmark::pad(raw,g,c.padded);
        const size_t bytes=static_cast<size_t>(g.padded_width)*g.padded_height;
        if(cudaMemcpyAsync(c.raw,c.padded.data(),bytes,cudaMemcpyHostToDevice,c.npp.hStream)!=cudaSuccess)return -3;
        const NppiSize size{g.padded_width,g.padded_height};
        const NppiRect roi{0,0,size.width,size.height};
        const auto grid=pattern==1 ? NPPI_BAYER_RGGB : NPPI_BAYER_BGGR;
        const auto status=nppiCFAToRGB_8u_C1C3R_Ctx(c.raw,g.padded_width,size,roi,c.rgb,
            g.padded_width*3,grid,NPPI_INTER_UNDEFINED,c.npp);
        if(status!=NPP_SUCCESS){cudaStreamSynchronize(c.npp.hStream);return -1000+static_cast<int>(status);}
        crop_bgr<<<(w*h+255)/256,256,0,c.npp.hStream>>>(c.rgb,c.bgr,w,h,g.padded_width*3);
        if(cudaGetLastError()!=cudaSuccess || cudaMemcpy2DAsync(bgr,g.bgr_pitch,c.bgr,w*3,w*3,h,
            cudaMemcpyDeviceToHost,c.npp.hStream)!=cudaSuccess){cudaStreamSynchronize(c.npp.hStream);return -3;}
        return cudaStreamSynchronize(c.npp.hStream)==cudaSuccess ? 0 : -3;
    }catch(...){cudaStreamSynchronize(c.npp.hStream);return -3;}
}
