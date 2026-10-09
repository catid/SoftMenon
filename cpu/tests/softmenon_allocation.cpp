// Exercise real allocation failures without adding fault controls to the library.
#include "cpu_debayer.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <new>
#include <stdexcept>
#include <vector>

namespace {
enum class Fault { none, helper_tile, helper_cache, caller_countdown };
std::atomic<Fault> fault{Fault::none};
std::atomic<int> seen{0}, countdown{0};
std::atomic<size_t> cache_bytes{0};
thread_local bool calling_thread=false;

bool fail_allocation(size_t bytes) {
    const Fault mode=fault.load(std::memory_order_relaxed);
    bool match=false;
    if(mode==Fault::helper_tile)match=!calling_thread && bytes>=4096;
    if(mode==Fault::helper_cache)match=!calling_thread && bytes==cache_bytes.load();
    if(mode==Fault::caller_countdown && calling_thread)match=countdown.fetch_sub(1)==0;
    if(!match)return false;
    Fault expected=mode;
    if(!fault.compare_exchange_strong(expected,Fault::none))return false;
    ++seen;
    return true;
}
void* allocate(size_t bytes) {
    if(fail_allocation(bytes))throw std::bad_alloc();
    if(void* p=std::malloc(bytes?bytes:1))return p;
    throw std::bad_alloc();
}
void require(bool condition,const char* message) {
    if(!condition)throw std::runtime_error(message);
}
}

void* operator new(size_t bytes){return allocate(bytes);}
void* operator new[](size_t bytes){return allocate(bytes);}
void operator delete(void* p) noexcept{std::free(p);}
void operator delete[](void* p) noexcept{std::free(p);}
void operator delete(void* p,size_t) noexcept{std::free(p);}
void operator delete[](void* p,size_t) noexcept{std::free(p);}
void* operator new(size_t bytes,const std::nothrow_t&) noexcept {
    try{return allocate(bytes);}catch(...){return nullptr;}
}
void* operator new[](size_t bytes,const std::nothrow_t&) noexcept {
    try{return allocate(bytes);}catch(...){return nullptr;}
}
void operator delete(void* p,const std::nothrow_t&) noexcept{std::free(p);}
void operator delete[](void* p,const std::nothrow_t&) noexcept{std::free(p);}

namespace {
struct Image {
    static constexpr int width=257,height=193,raw_pitch=width+13,bgr_pitch=3*width+19;
    static constexpr size_t guard=37;
    std::vector<uint8_t> raw=std::vector<uint8_t>(guard+raw_pitch*height+guard,0xa7);
    std::vector<uint8_t> bgr=std::vector<uint8_t>(guard+bgr_pitch*height+guard,0xa7);
    Image() {
        uint32_t state=0x3952a1c7;
        for(int y=0;y<height;++y)for(int x=0;x<width;++x) {
            state=1664525u*state+1013904223u;
            raw[guard+y*raw_pitch+x]=static_cast<uint8_t>(state>>24);
        }
    }
    int run(Debayer& backend) {
        raw_image_t input{width,height,raw.data()+guard,raw_pitch,
            SARONIC_DEBAYER_SOFTMENON,SARONIC_DEBAYER_RGGB};
        bgr_image_t output{width,height,bgr.data()+guard,bgr_pitch};
        return backend.Process(&input,&output);
    }
    void check_guards()const {
        for(size_t i=0;i<guard;++i)
            require(bgr[i]==0xa7 && bgr[bgr.size()-i-1]==0xa7,"output guard changed");
        for(int y=0;y<height;++y)for(int x=3*width;x<bgr_pitch;++x)
            require(bgr[guard+y*bgr_pitch+x]==0xa7,"output row padding changed");
    }
};

void arm(Fault mode,int skip=0) {
    seen=0;
    countdown=skip;
    fault=mode;
}
int run_fault(Image& image,Debayer& backend,Fault mode,int skip=0) {
    std::fill(image.bgr.begin(),image.bgr.end(),0xa7);
    arm(mode,skip);
    const int result=image.run(backend);
    fault=Fault::none;
    image.check_guards();
    return result;
}
}

int main() {
    calling_thread=true;
    try {
        Image image;
        Debayer baseline(1);
        require(image.run(baseline)==0,"baseline failed");
        const auto expected=image.bgr,raw_before=image.raw;
        const auto recover=[&](Debayer& backend) {
            std::fill(image.bgr.begin(),image.bgr.end(),0xa7);
            return image.run(backend)==0 && image.bgr==expected;
        };

        {
            // A fresh pool has no per-thread tile buffers. Fail a helper's
            // first large allocation while its peers continue processing.
            Debayer backend(3);
            require(run_fault(image,backend,Fault::helper_tile)==-3,"helper tile failure was lost");
            require(seen==1,"helper tile failure was not injected");
            require(recover(backend),"helper failure recovery changed pixels");
        }

        bool cache_checked=false;
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
        if(__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
            __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vbmi")) {
            // This allocation size is the directional-difference cache of a
            // 64-row tile plus halos; outer RGB tiles have different sizes.
            const int green_width=((Image::width+1)&~1)+4;
            const int cache_pitch=(((green_width+4)/2)+31)&~31;
            cache_bytes=static_cast<size_t>(64+8)*cache_pitch*sizeof(int16_t);
            Debayer backend(3);
            require(run_fault(image,backend,Fault::helper_cache)==0,"cache failure did not fall back");
            require(seen==1,"cache allocation failure was not injected");
            require(image.bgr==expected,"cache fallback changed pixels");
            require(recover(backend),"cache recovery changed pixels");
            cache_checked=true;
        }
#endif
        int submission_failures=0;
        {
            Debayer backend(3);
            require(image.run(backend)==0,"pool warmup failed");
            // Each successive failure reaches one further allocation on the
            // calling thread, covering callable copies and queue submission.
            // In particular a later failed submission must drain prior tasks
            // while their referenced failure flag is still alive.
            for(int skip=0;skip<32;++skip) {
                const int status=run_fault(image,backend,Fault::caller_countdown,skip);
                if(seen==0){require(status==0 && image.bgr==expected,"uninjected call failed");break;}
                require(status==-3,"caller allocation failure was lost");
                ++submission_failures;
                require(recover(backend),"submission recovery changed pixels");
            }
        }
        require(submission_failures>=3,"insufficient caller allocation coverage");
        require(image.raw==raw_before,"input changed after a failure");
        std::cout<<"SoftMenon allocation recovery: helper tile, "<<submission_failures
            <<" caller allocation points, cache fallback "<<(cache_checked?"passed":"not available")<<'\n';
        return 0;
    }catch(const std::exception& error) {
        fault=Fault::none;
        std::cerr<<"SoftMenon allocation failure test: "<<error.what()<<'\n';
        return 1;
    }
}
