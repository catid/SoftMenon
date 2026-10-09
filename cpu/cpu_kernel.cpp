/**
 * @file cpu_kernel.cpp
 * @brief CPU-based kernel functions for image padding and Bayer pattern demosaicing.
 */

#include "cpu_kernel.hpp"
#include "chroma_median.hpp"

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <cstddef>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define CPU_SOFT_AVX2 1
#define SOFT_INLINE __attribute__((target("avx2"),always_inline)) inline
#define SOFT512_INLINE __attribute__((target("avx512f,avx512bw,avx512vbmi"),always_inline)) inline
#else
#define CPU_SOFT_AVX2 0
#endif


namespace {
// Integer division truncation for negative numerators cannot affect a clipped
// uint8 result: both truncation and floor clip to zero. Clamp before shifting
// to preserve the CUDA formulas exactly without signed-division corrections.
inline uint8_t scaled_u8(int numerator,int shift) {
    return static_cast<uint8_t>(std::min(255,std::max(0,numerator)>>shift));
}
}

void bilinear_cpu_rows(const uint8_t* raw,size_t raw_pitch,uint8_t* bgr,
    size_t bgr_pitch,int width,int height,bool rggb,int begin_y,int end_y) {
    const ptrdiff_t pitch=static_cast<ptrdiff_t>(raw_pitch);
    begin_y=std::max(0,begin_y); end_y=std::min(height,end_y);
    for (int y=begin_y;y<end_y;++y) {
        const uint8_t* input=raw+static_cast<size_t>(y)*raw_pitch;
        uint8_t* output=bgr+static_cast<size_t>(y)*bgr_pitch;
        const bool red_row=((y&1)==0)==rggb;
        for (int x=0;x<width;++x) {
            const uint8_t* p=input+x;
            int b,g,r;
            if ((x&1)==(y&1)) {
                const int opposite=(p[-pitch-1]+p[-pitch+1]+p[pitch-1]+p[pitch+1]+2)>>2;
                g=(p[-1]+p[1]+p[-pitch]+p[pitch]+2)>>2;
                r=red_row ? p[0] : opposite;
                b=red_row ? opposite : p[0];
            } else {
                const int horizontal=(p[-1]+p[1]+1)>>1;
                const int vertical=(p[-pitch]+p[pitch]+1)>>1;
                g=p[0]; r=red_row ? horizontal : vertical;
                b=red_row ? vertical : horizontal;
            }
            output[3*x]=static_cast<uint8_t>(b);
            output[3*x+1]=static_cast<uint8_t>(g);
            output[3*x+2]=static_cast<uint8_t>(r);
        }
    }
}

void malvar2004_cpu_rows(const uint8_t* raw,size_t raw_pitch,uint8_t* bgr,
    size_t bgr_pitch,int width,int height,bool rggb,int begin_y,int end_y) {
    const ptrdiff_t pitch=static_cast<ptrdiff_t>(raw_pitch);
    begin_y=std::max(0,begin_y); end_y=std::min(height,end_y);
    for (int y=begin_y;y<end_y;++y) {
        const uint8_t* input=raw+static_cast<size_t>(y)*raw_pitch;
        uint8_t* output=bgr+static_cast<size_t>(y)*bgr_pitch;
        const bool red_row=((y&1)==0)==rggb;
        for (int x=0;x<width;++x) {
            const uint8_t* p=input+x;
            const int center=p[0];
            const int horizontal=p[-1]+p[1],vertical=p[-pitch]+p[pitch];
            const int far_horizontal=p[-2]+p[2],far_vertical=p[-2*pitch]+p[2*pitch];
            const int diagonals=p[-pitch-1]+p[-pitch+1]+p[pitch-1]+p[pitch+1];
            uint8_t b,g,r;
            if ((x&1)==(y&1)) {
                g=scaled_u8(4*center+2*(horizontal+vertical)-far_horizontal-far_vertical+4,3);
                const uint8_t opposite=scaled_u8(12*center+4*diagonals-3*(far_horizontal+far_vertical)+8,4);
                r=red_row ? static_cast<uint8_t>(center) : opposite;
                b=red_row ? opposite : static_cast<uint8_t>(center);
            } else {
                const uint8_t along_horizontal=scaled_u8(10*center+8*horizontal-2*(diagonals+far_horizontal)+far_vertical+8,4);
                const uint8_t along_vertical=scaled_u8(10*center+8*vertical-2*(diagonals+far_vertical)+far_horizontal+8,4);
                g=static_cast<uint8_t>(center);
                r=red_row ? along_horizontal : along_vertical;
                b=red_row ? along_vertical : along_horizontal;
            }
            output[3*x]=b; output[3*x+1]=g; output[3*x+2]=r;
        }
    }
}

inline uint8_t saturate_cast_int16_to_uint8(int16_t x) {
    return static_cast<uint8_t>(std::clamp<int>(x,0,255));
}

#include "softmenon_green.hpp"

void bggr_softmenon_g_cpu(const uint8_t* raw,int raw_pitch,uint8_t* bgr,
    int bgr_pitch,int width,int height) {
    softmenon_cached_dispatch(raw,raw_pitch,bgr,bgr_pitch,width,height);
}

namespace {
inline void soft_rb_pixel(const uint8_t* source,ptrdiff_t pitch,uint8_t* output,int x,int y) {
    const uint8_t* p=source+static_cast<ptrdiff_t>(y)*pitch+static_cast<ptrdiff_t>(x)*3;
    const int green=p[1];
    const bool chromatic=(x&1)==(y&1);
    output[1]=static_cast<uint8_t>(green);
    for (int channel : {0,2}) {
        const bool measured=chromatic && (((y&1)==0)==(channel==0));
        if (measured) { output[channel]=p[channel]; continue; }
        int difference;
        if (chromatic) {
            const int ul=static_cast<int>(p[-pitch-3+channel])-p[-pitch-2];
            const int ur=static_cast<int>(p[-pitch+3+channel])-p[-pitch+4];
            const int dl=static_cast<int>(p[pitch-3+channel])-p[pitch-2];
            const int dr=static_cast<int>(p[pitch+3+channel])-p[pitch+4];
            const int horizontal=(ul+dr+1)>>1,vertical=(ur+dl+1)>>1;
            const int gh=std::abs(ul-dr),gv=std::abs(ur-dl);
            difference=gh<=gv ? horizontal : vertical;
            // Near-equal diagonals share equally; moderate gaps favor the
            // lower-variation direction 3:1, and large gaps use it alone.
            if (std::abs(gh-gv)<=16) difference=(horizontal+vertical+1)>>1;
            else if (std::abs(gh-gv)<=64) difference=(2*difference+horizontal+vertical+2)>>2;
        } else {
            const bool horizontal=((y&1)==0)==(channel==0);
            const ptrdiff_t axis=horizontal ? 3 : pitch;
            difference=(static_cast<int>(p[-axis+channel])-p[-axis+1]
                       +static_cast<int>(p[axis+channel])-p[axis+1]+1)>>1;
        }
        output[channel]=saturate_cast_int16_to_uint8(static_cast<int16_t>(green+difference));
    }
}

#if CPU_SOFT_AVX2
template <bool OddRow>
SOFT_INLINE void soft_rb8(const uint8_t* source,ptrdiff_t pitch,uint8_t* output,int x,int y) {
    __m256i values[9],green=_mm256_setzero_si256();
    for (int dy=-1;dy<=1;++dy) {
        const uint8_t* row=source+static_cast<ptrdiff_t>(y+dy)*pitch+static_cast<ptrdiff_t>(x-1)*3;
        const __m128i a=_mm_loadu_si128(reinterpret_cast<const __m128i*>(row));
        const __m128i b=_mm_loadu_si128(reinterpret_cast<const __m128i*>(row+14));
        const auto left=median_cpu_detail::row_chroma<0>(a,b);
        const auto center=median_cpu_detail::row_chroma<3>(a,b);
        const auto right=median_cpu_detail::row_chroma<6>(a,b);
        values[(dy+1)*3]=left.differences;
        values[(dy+1)*3+1]=center.differences;
        values[(dy+1)*3+2]=right.differences;
        if (dy==0) green=center.green;
    }
    const __m256i one=_mm256_set1_epi16(1);
    const __m256i diagonal_h=_mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(values[0],values[8]),one),1);
    const __m256i diagonal_v=_mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(values[2],values[6]),one),1);
    const __m256i gradient_h=_mm256_abs_epi16(_mm256_sub_epi16(values[0],values[8]));
    const __m256i gradient_v=_mm256_abs_epi16(_mm256_sub_epi16(values[2],values[6]));
    __m256i diagonal=_mm256_blendv_epi8(diagonal_h,diagonal_v,_mm256_cmpgt_epi16(gradient_h,gradient_v));
    const __m256i gap=_mm256_abs_epi16(_mm256_sub_epi16(gradient_h,gradient_v));
    const __m256i weak=_mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_slli_epi16(diagonal,1),_mm256_add_epi16(diagonal_h,diagonal_v)),_mm256_set1_epi16(2)),2);
    const __m256i mean=_mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(diagonal_h,diagonal_v),one),1);
    diagonal=_mm256_blendv_epi8(diagonal,weak,_mm256_cmpgt_epi16(_mm256_set1_epi16(65),gap));
    diagonal=_mm256_blendv_epi8(diagonal,mean,_mm256_cmpgt_epi16(_mm256_set1_epi16(17),gap));
    const __m256i horizontal=_mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(values[3],values[5]),one),1);
    const __m256i vertical=_mm256_srai_epi16(_mm256_add_epi16(_mm256_add_epi16(values[1],values[7]),one),1);
    const __m256i axial=_mm256_blend_epi16(horizontal,vertical,OddRow ? 0x55 : 0xaa);
    const __m256i difference=_mm256_blend_epi16(diagonal,axial,OddRow ? 0x33 : 0xcc);
    const __m256i estimates=_mm256_add_epi16(green,difference);
    const __m256i measured=_mm256_add_epi16(green,values[4]);
    const __m128i colors=median_cpu_detail::clipped_bytes(_mm256_blend_epi16(estimates,measured,OddRow ? 0x88 : 0x11));
    const uint8_t* center=source+static_cast<ptrdiff_t>(y)*pitch+static_cast<ptrdiff_t>(x)*3;
    const __m128i a=_mm_or_si128(_mm_shuffle_epi8(colors,_mm_setr_epi8(0,-128,1,2,-128,3,4,-128,5,6,-128,7,8,-128,9,10)),
        _mm_and_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(center)),
                      _mm_setr_epi8(0,-1,0,0,-1,0,0,-1,0,0,-1,0,0,-1,0,0)));
    const __m128i b=_mm_or_si128(_mm_shuffle_epi8(colors,_mm_setr_epi8(5,6,-128,7,8,-128,9,10,-128,11,12,-128,13,14,-128,15)),
        _mm_and_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(center+8)),
                      _mm_setr_epi8(0,0,-1,0,0,-1,0,0,-1,0,0,-1,0,0,-1,0)));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output),a);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output+8),b);
}

__attribute__((target("avx2"))) void soft_rb_avx2(const uint8_t* source,size_t source_pitch,
        uint8_t* destination,size_t destination_pitch,int width,int begin_y,int end_y) {
    const ptrdiff_t pitch=static_cast<ptrdiff_t>(source_pitch);
    for (int y=begin_y;y<end_y;++y) {
        uint8_t* output=destination+static_cast<size_t>(y)*destination_pitch;
        int x=0;
        if (y&1) {
            for (;x<=width-8;x+=8) soft_rb8<true>(source,pitch,output+3*x,x,y);
        } else {
            for (;x<=width-8;x+=8) soft_rb8<false>(source,pitch,output+3*x,x,y);
        }
        for (;x<width;++x) soft_rb_pixel(source,pitch,output+3*x,x,y);
    }
}

template <bool OddRow>
SOFT512_INLINE void soft_rb16(const uint8_t* source,ptrdiff_t pitch,uint8_t* output,int x,int y) {
    __m512i values[9],green=_mm512_setzero_si512(),center_row=_mm512_setzero_si512();
    for (int dy=-1;dy<=1;++dy) {
        const uint8_t* input=source+static_cast<ptrdiff_t>(y+dy)*pitch+static_cast<ptrdiff_t>(x-1)*3;
        const __m512i row=_mm512_maskz_loadu_epi8((uint64_t{1}<<54)-1,input);
        const auto left=median_cpu_detail::row_chroma512<0>(row);
        const auto center=median_cpu_detail::row_chroma512<3>(row);
        const auto right=median_cpu_detail::row_chroma512<6>(row);
        values[(dy+1)*3]=left.differences;
        values[(dy+1)*3+1]=center.differences;
        values[(dy+1)*3+2]=right.differences;
        if (dy==0) { green=center.green; center_row=row; }
    }
    const __m512i one=_mm512_set1_epi16(1);
    const __m512i diagonal_h=_mm512_srai_epi16(_mm512_add_epi16(_mm512_add_epi16(values[0],values[8]),one),1);
    const __m512i diagonal_v=_mm512_srai_epi16(_mm512_add_epi16(_mm512_add_epi16(values[2],values[6]),one),1);
    const __m512i gradient_h=_mm512_abs_epi16(_mm512_sub_epi16(values[0],values[8]));
    const __m512i gradient_v=_mm512_abs_epi16(_mm512_sub_epi16(values[2],values[6]));
    __m512i diagonal=_mm512_mask_mov_epi16(diagonal_h,_mm512_cmpgt_epi16_mask(gradient_h,gradient_v),diagonal_v);
    const __m512i gap=_mm512_abs_epi16(_mm512_sub_epi16(gradient_h,gradient_v));
    const __m512i weak=_mm512_srai_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_slli_epi16(diagonal,1),_mm512_add_epi16(diagonal_h,diagonal_v)),_mm512_set1_epi16(2)),2);
    const __m512i mean=_mm512_srai_epi16(_mm512_add_epi16(_mm512_add_epi16(diagonal_h,diagonal_v),one),1);
    diagonal=_mm512_mask_mov_epi16(diagonal,_mm512_cmpgt_epi16_mask(_mm512_set1_epi16(65),gap),weak);
    diagonal=_mm512_mask_mov_epi16(diagonal,_mm512_cmpgt_epi16_mask(_mm512_set1_epi16(17),gap),mean);
    const __m512i horizontal=_mm512_srai_epi16(_mm512_add_epi16(_mm512_add_epi16(values[3],values[5]),one),1);
    const __m512i vertical=_mm512_srai_epi16(_mm512_add_epi16(_mm512_add_epi16(values[1],values[7]),one),1);
    const __m512i axial=_mm512_mask_mov_epi16(horizontal,OddRow ? 0x55555555u : 0xaaaaaaaau,vertical);
    const __m512i difference=_mm512_mask_mov_epi16(diagonal,OddRow ? 0x33333333u : 0xccccccccu,axial);
    const __m512i estimates=_mm512_add_epi16(green,difference);
    const __m512i measured=_mm512_add_epi16(green,values[4]);
    const __m512i colors=_mm512_mask_mov_epi16(estimates,OddRow ? 0x88888888u : 0x11111111u,measured);
    const __m512i clipped=_mm512_min_epi16(_mm512_max_epi16(colors,_mm512_setzero_si512()),_mm512_set1_epi16(255));
    const __m512i bytes=_mm512_castsi256_si512(_mm512_cvtepi16_epi8(clipped));
    alignas(64) static constexpr auto scatter_index=[] {
        std::array<uint8_t,64> index{};
        for (int i=0;i<48;++i) index[i]=2*(i/3)+(i%3==2);
        return index;
    }();
    alignas(64) static constexpr auto green_index=[] {
        std::array<uint8_t,64> index{};
        for (int i=0;i<48;++i) index[i]=i+3;
        return index;
    }();
    constexpr __mmask64 green_mask=[] {
        uint64_t mask=0;
        for (int i=1;i<48;i+=3) mask|=uint64_t{1}<<i;
        return mask;
    }();
    const __m512i result=_mm512_mask_mov_epi8(
        _mm512_permutexvar_epi8(_mm512_load_si512(scatter_index.data()),bytes),green_mask,
        _mm512_permutexvar_epi8(_mm512_load_si512(green_index.data()),center_row));
    _mm512_mask_storeu_epi8(output,(uint64_t{1}<<48)-1,result);
}

__attribute__((target("avx512f,avx512bw,avx512vbmi"))) void soft_rb_avx512(
        const uint8_t* source,size_t source_pitch,uint8_t* destination,size_t destination_pitch,
        int width,int begin_y,int end_y) {
    const ptrdiff_t pitch=static_cast<ptrdiff_t>(source_pitch);
    for (int y=begin_y;y<end_y;++y) {
        uint8_t* output=destination+static_cast<size_t>(y)*destination_pitch;
        int x=0;
        if (y&1) {
            for (;x<=width-16;x+=16) soft_rb16<true>(source,pitch,output+3*x,x,y);
        } else {
            for (;x<=width-16;x+=16) soft_rb16<false>(source,pitch,output+3*x,x,y);
        }
        for (;x<width;++x) soft_rb_pixel(source,pitch,output+3*x,x,y);
    }
}
#endif
}

void bggr_softmenon_rb_rows(const uint8_t* source,size_t source_pitch,uint8_t* destination,
        size_t destination_pitch,int width,int height,int begin_y,int end_y) {
    begin_y=std::max(0,begin_y); end_y=std::min(height,end_y);
#if CPU_SOFT_AVX2
    if (width>=16 && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vbmi")) {
        soft_rb_avx512(source,source_pitch,destination,destination_pitch,width,begin_y,end_y);
        return;
    }
    if (width>=8 && __builtin_cpu_supports("avx2")) {
        soft_rb_avx2(source,source_pitch,destination,destination_pitch,width,begin_y,end_y);
        return;
    }
#endif
    for (int y=begin_y;y<end_y;++y) {
        uint8_t* output=destination+static_cast<size_t>(y)*destination_pitch;
        for (int x=0;x<width;++x)
            soft_rb_pixel(source,static_cast<ptrdiff_t>(source_pitch),output+3*x,x,y);
    }
}
