#pragma once
// Common host-side REFLECT_101 framing for optional external comparisons.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#define EXTERNAL_EXPORT extern "C" __attribute__((visibility("default")))
namespace external_benchmark {
constexpr int halo = 4;
struct Geometry { int width, height, raw_pitch, bgr_pitch, padded_width, padded_height; };
inline bool geometry(int w,int h,int rp,int bp,Geometry& g) {
    if (w<2 || h<2 || w>std::numeric_limits<int>::max()/3-16 || h>std::numeric_limits<int>::max()-16) return false;
    rp=rp ? rp : w; bp=bp ? bp : w*3;
    if (rp<w || bp<w*3) return false;
    const int pw=((w+1)&~1)+2*halo,ph=((h+1)&~1)+2*halo;
    if (static_cast<int64_t>(pw)*ph>std::numeric_limits<int>::max()/3) return false;
    g={w,h,rp,bp,pw,ph};return true;
}
inline int reflect(int p,int n) {
    const int period=2*(n-1);
    p%=period;if(p<0)p+=period;
    return p<n ? p : period-p;
}
inline void pad(const uint8_t* raw,const Geometry& g,std::vector<uint8_t>& padded) {
    padded.resize(static_cast<size_t>(g.padded_width)*g.padded_height);
    int left[halo],right[halo+1];
    const int nr=g.padded_width-halo-g.width;
    for(int x=0;x<halo;++x)left[x]=reflect(x-halo,g.width);
    for(int x=0;x<nr;++x)right[x]=reflect(g.width+x,g.width);
    for(int y=0;y<g.height;++y) {
        const uint8_t* src=raw+static_cast<size_t>(y)*g.raw_pitch;
        uint8_t* dst=padded.data()+static_cast<size_t>(y+halo)*g.padded_width;
        std::memcpy(dst+halo,src,g.width);
        for(int x=0;x<halo;++x)dst[x]=src[left[x]];
        for(int x=0;x<nr;++x)dst[halo+g.width+x]=src[right[x]];
    }
    for(int y=0;y<g.padded_height;++y) {
        if(y>=halo && y<halo+g.height)continue;
        const int source_y=halo+reflect(y-halo,g.height);
        std::memcpy(padded.data()+static_cast<size_t>(y)*g.padded_width,
            padded.data()+static_cast<size_t>(source_y)*g.padded_width,g.padded_width);
    }
}
}
