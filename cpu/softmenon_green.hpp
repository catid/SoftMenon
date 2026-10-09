#pragma once
// SoftMenon green interpolation. Separately rounded horizontal/vertical HA
// candidates are scored by posterior directional color-difference variation.
// Squared inverse scores are blended exactly, with half ties rounded upward.
// The RAW rectangle starts at a BGGR blue site, has even dimensions, and needs
// a four-pixel halo. The caller adds its two-pixel completion halo separately.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include <new>
#include <vector>

#if CPU_SOFT_AVX2
namespace {
SOFT_INLINE void soft_store16(uint8_t* output,__m128i blue,__m128i green,__m128i red) {
    const __m128i b0=_mm_shuffle_epi8(blue,_mm_setr_epi8(0,-128,-128,1,-128,-128,2,-128,-128,3,-128,-128,4,-128,-128,5));
    const __m128i g0=_mm_shuffle_epi8(green,_mm_setr_epi8(-128,0,-128,-128,1,-128,-128,2,-128,-128,3,-128,-128,4,-128,-128));
    const __m128i r0=_mm_shuffle_epi8(red,_mm_setr_epi8(-128,-128,0,-128,-128,1,-128,-128,2,-128,-128,3,-128,-128,4,-128));
    const __m128i b1=_mm_shuffle_epi8(blue,_mm_setr_epi8(-128,-128,6,-128,-128,7,-128,-128,8,-128,-128,9,-128,-128,10,-128));
    const __m128i g1=_mm_shuffle_epi8(green,_mm_setr_epi8(5,-128,-128,6,-128,-128,7,-128,-128,8,-128,-128,9,-128,-128,10));
    const __m128i r1=_mm_shuffle_epi8(red,_mm_setr_epi8(-128,5,-128,-128,6,-128,-128,7,-128,-128,8,-128,-128,9,-128,-128));
    const __m128i b2=_mm_shuffle_epi8(blue,_mm_setr_epi8(-128,11,-128,-128,12,-128,-128,13,-128,-128,14,-128,-128,15,-128,-128));
    const __m128i g2=_mm_shuffle_epi8(green,_mm_setr_epi8(-128,-128,11,-128,-128,12,-128,-128,13,-128,-128,14,-128,-128,15,-128));
    const __m128i r2=_mm_shuffle_epi8(red,_mm_setr_epi8(10,-128,-128,11,-128,-128,12,-128,-128,13,-128,-128,14,-128,-128,15));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output),_mm_or_si128(b0,_mm_or_si128(g0,r0)));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output+16),_mm_or_si128(b1,_mm_or_si128(g1,r1)));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output+32),_mm_or_si128(b2,_mm_or_si128(g2,r2)));
}
template <int Parity>
SOFT512_INLINE __m512i green_load_chromatic(const uint8_t* p) {
    alignas(64) static constexpr auto index=[] {
        std::array<uint8_t,64> a{};
        for(int i=0;i<16;++i) a[i]=2*i+Parity;
        return a;
    }();
    const __m512i bytes=_mm512_maskz_loadu_epi8(0xffffffffULL,p);
    return _mm512_cvtepu8_epi32(_mm512_castsi512_si128(
        _mm512_permutexvar_epi8(_mm512_load_si512(index.data()),bytes)));
}
template <int Parity,int Offset>
SOFT512_INLINE __m512i green_bgr_block(__m512i values) {
    alignas(64) static constexpr auto index=[] {
        std::array<uint8_t,64> a{};
        for(int i=0;i<64;++i) {
            const int j=i+Offset,x=j/3,c=j%3;
            a[i]=(c==1 && (x&1)==Parity) ? 32+x/2 : x;
        }
        return a;
    }();
    static constexpr uint64_t keep=[] {
        uint64_t m=0;
        for(int i=0;i<64;++i) {
            const int j=i+Offset,x=j/3,c=j%3;
            if(j<96 && (c==1 || ((x&1)==Parity && c==2*Parity)))m|=uint64_t(1)<<i;
        }
        return m;
    }();
    return _mm512_maskz_permutexvar_epi8(keep,_mm512_load_si512(index.data()),values);
}
template <int Parity>
SOFT_INLINE __m256i soft_load_chromatic16(const uint8_t* p) {
    const __m256i raw=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    // A RAW byte pair is already a uint16 lane: select the chromatic byte
    // directly, avoiding byte gathers and any work at observed green sites.
    if constexpr (Parity) return _mm256_srli_epi16(raw,8);
    return _mm256_and_si256(raw,_mm256_set1_epi16(255));
}
}
#endif
// Radius four plus the production green halo of two needs padding >=6.
namespace {
inline int softmenon_candidate(const uint8_t* p,ptrdiff_t a) {
    return ((int(p[-a])+p[a]+1)>>1)+((2*int(p[0])-p[-2*a]-p[2*a]+2)>>2);
}
inline int softmenon_difference(const uint8_t* p,ptrdiff_t a) {
    return int(p[0])-softmenon_candidate(p,a);
}
inline int softmenon_colocated(const uint8_t* p,ptrdiff_t a) {
    const int c=softmenon_difference(p,a);
    return std::abs(c-softmenon_difference(p-2*a,a))+std::abs(c-softmenon_difference(p+2*a,a));
}
inline int softmenon_score(const uint8_t* p,ptrdiff_t a,ptrdiff_t b) {
    const int c=softmenon_colocated(p,a);
    return 3*c+softmenon_colocated(p-2*b,a)+softmenon_colocated(p+2*b,a)
        +std::abs(softmenon_difference(p-a-b,a)-softmenon_difference(p+a-b,a))
        +std::abs(softmenon_difference(p-a+b,a)-softmenon_difference(p+a+b,a));
}
inline void softmenon_scalar(const uint8_t* raw,int rp,uint8_t* out,int op,int w,int h) {
    for(int y=0;y<h;++y)for(int x=0;x<w;++x) {
        const uint8_t* p=raw+ptrdiff_t(y)*rp+x;uint8_t* q=out+ptrdiff_t(y)*op+3*x;
        q[0]=q[2]=0;
        if((x&1)!=(y&1)){q[1]=p[0];continue;}
        const int gh=softmenon_candidate(p,1),gv=softmenon_candidate(p,rp);
        int64_t wh=softmenon_score(p,rp,1)+1,wv=softmenon_score(p,1,rp)+1;
        wh*=wh;wv*=wv;
        const int64_t den=wh+wv,num=wh*gh+wv*gv;
        q[1]=uint8_t(std::min<int64_t>(255,std::max<int64_t>(0,num+den/2)/den));
        q[(y&1)?2:0]=p[0];
    }
}
}
#if CPU_SOFT_AVX2
#define SM512 __attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi"),always_inline)) inline
namespace {

template<int Parity>
SM512 __m512i softmenon_candidate512(const uint8_t* p,ptrdiff_t a) {
    const __m512i c=green_load_chromatic<Parity>(p),l=green_load_chromatic<Parity>(p-a),r=green_load_chromatic<Parity>(p+a);
    const __m512i l2=green_load_chromatic<Parity>(p-2*a),r2=green_load_chromatic<Parity>(p+2*a);
    return _mm512_add_epi32(_mm512_srli_epi32(_mm512_add_epi32(_mm512_add_epi32(l,r),_mm512_set1_epi32(1)),1),
        _mm512_srai_epi32(_mm512_add_epi32(_mm512_sub_epi32(_mm512_slli_epi32(c,1),_mm512_add_epi32(l2,r2)),_mm512_set1_epi32(2)),2));
}
template<int Parity>
SM512 __m512i softmenon_difference512(const uint8_t* p,ptrdiff_t a) {
    return _mm512_sub_epi32(green_load_chromatic<Parity>(p),softmenon_candidate512<Parity>(p,a));
}
template<int Parity>
SM512 __m512i softmenon_colocated512(const uint8_t* p,ptrdiff_t a) {
    const __m512i c=softmenon_difference512<Parity>(p,a);
    return _mm512_add_epi32(_mm512_abs_epi32(_mm512_sub_epi32(c,softmenon_difference512<Parity>(p-2*a,a))),
                  _mm512_abs_epi32(_mm512_sub_epi32(c,softmenon_difference512<Parity>(p+2*a,a))));
}
template<int Parity>
SM512 __m512i softmenon_score512(const uint8_t* p,ptrdiff_t a,ptrdiff_t b) {
    const __m512i c=softmenon_colocated512<Parity>(p,a);
    const __m512i parallel=_mm512_add_epi32(softmenon_colocated512<Parity>(p-2*b,a),softmenon_colocated512<Parity>(p+2*b,a));
    const __m512i corners=_mm512_add_epi32(
        _mm512_abs_epi32(_mm512_sub_epi32(softmenon_difference512<Parity>(p-a-b,a),softmenon_difference512<Parity>(p+a-b,a))),
        _mm512_abs_epi32(_mm512_sub_epi32(softmenon_difference512<Parity>(p-a+b,a),softmenon_difference512<Parity>(p+a+b,a))));
    return _mm512_add_epi32(_mm512_add_epi32(c,_mm512_slli_epi32(c,1)),_mm512_add_epi32(parallel,corners));
}

template<int Parity>
SOFT_INLINE __m256i softmenon_candidate256(const uint8_t* p,ptrdiff_t a) {
    const __m256i c=soft_load_chromatic16<Parity>(p),l=soft_load_chromatic16<Parity>(p-a),r=soft_load_chromatic16<Parity>(p+a);
    const __m256i l2=soft_load_chromatic16<Parity>(p-2*a),r2=soft_load_chromatic16<Parity>(p+2*a);
    return _mm256_add_epi16(_mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(l,r),_mm256_set1_epi16(1)),1),
        _mm256_srai_epi16(_mm256_add_epi16(_mm256_sub_epi16(_mm256_slli_epi16(c,1),_mm256_add_epi16(l2,r2)),_mm256_set1_epi16(2)),2));
}
template<int Parity>
SOFT_INLINE __m256i softmenon_difference256(const uint8_t* p,ptrdiff_t a) {
    return _mm256_sub_epi16(soft_load_chromatic16<Parity>(p),softmenon_candidate256<Parity>(p,a));
}
template<int Parity>
SOFT_INLINE __m256i softmenon_colocated256(const uint8_t* p,ptrdiff_t a) {
    const __m256i c=softmenon_difference256<Parity>(p,a);
    return _mm256_add_epi16(_mm256_abs_epi16(_mm256_sub_epi16(c,softmenon_difference256<Parity>(p-2*a,a))),
                  _mm256_abs_epi16(_mm256_sub_epi16(c,softmenon_difference256<Parity>(p+2*a,a))));
}
template<int Parity>
SOFT_INLINE __m256i softmenon_score256(const uint8_t* p,ptrdiff_t a,ptrdiff_t b) {
    const __m256i c=softmenon_colocated256<Parity>(p,a);
    const __m256i parallel=_mm256_add_epi16(softmenon_colocated256<Parity>(p-2*b,a),softmenon_colocated256<Parity>(p+2*b,a));
    const __m256i corners=_mm256_add_epi16(
        _mm256_abs_epi16(_mm256_sub_epi16(softmenon_difference256<Parity>(p-a-b,a),softmenon_difference256<Parity>(p+a-b,a))),
        _mm256_abs_epi16(_mm256_sub_epi16(softmenon_difference256<Parity>(p-a+b,a),softmenon_difference256<Parity>(p+a+b,a))));
    return _mm256_add_epi16(_mm256_add_epi16(c,_mm256_slli_epi16(c,1)),_mm256_add_epi16(parallel,corners));
}

// Posterior weights require 64-bit products. Converting exact int32 weights
// and candidates to doubles gives exact products/sums (<2^36), then division.
// The nearest noninteger quotient is >=1/den (>1e-8) from an integer while
// double error here is <3e-14. Exact integer quotients are representable.
SM512 __m256i softmenon_blend_pd512(__m256i gh,__m256i gv,__m256i wh,__m256i wv,__m256i den) {
    const __m512d n=_mm512_add_pd(_mm512_mul_pd(_mm512_cvtepi32_pd(wh),_mm512_cvtepi32_pd(gh)),
                                  _mm512_mul_pd(_mm512_cvtepi32_pd(wv),_mm512_cvtepi32_pd(gv)));
    const __m512d rounded=_mm512_max_pd(_mm512_add_pd(n,_mm512_cvtepi32_pd(_mm256_srli_epi32(den,1))),_mm512_setzero_pd());
    return _mm512_cvttpd_epi32(_mm512_div_pd(rounded,_mm512_cvtepi32_pd(den)));
}
SM512 __m512i softmenon_blend512(__m512i gh,__m512i gv,__m512i sh,__m512i sv) {
    const __m512i one=_mm512_set1_epi32(1);
    __m512i wh=_mm512_add_epi32(sv,one),wv=_mm512_add_epi32(sh,one);
    wh=_mm512_mullo_epi32(wh,wh);wv=_mm512_mullo_epi32(wv,wv);
    const __m512i den=_mm512_add_epi32(wh,wv);
    const __m256i lo=softmenon_blend_pd512(_mm512_castsi512_si256(gh),_mm512_castsi512_si256(gv),_mm512_castsi512_si256(wh),_mm512_castsi512_si256(wv),_mm512_castsi512_si256(den));
    const __m256i hi=softmenon_blend_pd512(_mm512_extracti64x4_epi64(gh,1),_mm512_extracti64x4_epi64(gv,1),_mm512_extracti64x4_epi64(wh,1),_mm512_extracti64x4_epi64(wv,1),_mm512_extracti64x4_epi64(den,1));
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo),hi,1);
}
SOFT_INLINE __m128i softmenon_blend_pd256(__m128i gh,__m128i gv,__m128i wh,__m128i wv,__m128i den) {
    const __m256d n=_mm256_add_pd(_mm256_mul_pd(_mm256_cvtepi32_pd(wh),_mm256_cvtepi32_pd(gh)),
                                  _mm256_mul_pd(_mm256_cvtepi32_pd(wv),_mm256_cvtepi32_pd(gv)));
    const __m256d rounded=_mm256_max_pd(_mm256_add_pd(n,_mm256_cvtepi32_pd(_mm_srli_epi32(den,1))),_mm256_setzero_pd());
    return _mm256_cvttpd_epi32(_mm256_div_pd(rounded,_mm256_cvtepi32_pd(den)));
}
SOFT_INLINE __m256i softmenon_blend256(__m128i gh16,__m128i gv16,__m128i sh16,__m128i sv16) {
    const __m256i one=_mm256_set1_epi32(1);
    const __m256i gh=_mm256_cvtepi16_epi32(gh16),gv=_mm256_cvtepi16_epi32(gv16);
    __m256i wh=_mm256_add_epi32(_mm256_cvtepi16_epi32(sv16),one),wv=_mm256_add_epi32(_mm256_cvtepi16_epi32(sh16),one);
    wh=_mm256_mullo_epi32(wh,wh);wv=_mm256_mullo_epi32(wv,wv);
    const __m256i den=_mm256_add_epi32(wh,wv);
    const __m128i lo=softmenon_blend_pd256(_mm256_castsi256_si128(gh),_mm256_castsi256_si128(gv),_mm256_castsi256_si128(wh),_mm256_castsi256_si128(wv),_mm256_castsi256_si128(den));
    const __m128i hi=softmenon_blend_pd256(_mm256_extracti128_si256(gh,1),_mm256_extracti128_si256(gv,1),_mm256_extracti128_si256(wh,1),_mm256_extracti128_si256(wv,1),_mm256_extracti128_si256(den,1));
    return _mm256_inserti128_si256(_mm256_castsi128_si256(lo),hi,1);
}
template<int Parity>
SM512 __m128i softmenon_estimate512(const uint8_t* p,ptrdiff_t pitch) {
    const __m512i gh=softmenon_candidate512<Parity>(p,1),gv=softmenon_candidate512<Parity>(p,pitch);
    const __m512i sh=softmenon_score512<Parity>(p,1,pitch),sv=softmenon_score512<Parity>(p,pitch,1);
    return _mm512_cvtusepi32_epi8(softmenon_blend512(gh,gv,sh,sv));
}
template<int Parity>
SOFT_INLINE __m128i softmenon_estimate256(const uint8_t* p,ptrdiff_t pitch) {
    const __m256i gh=softmenon_candidate256<Parity>(p,1),gv=softmenon_candidate256<Parity>(p,pitch);
    const __m256i sh=softmenon_score256<Parity>(p,1,pitch),sv=softmenon_score256<Parity>(p,pitch,1);
    const __m256i lo=softmenon_blend256(_mm256_castsi256_si128(gh),_mm256_castsi256_si128(gv),_mm256_castsi256_si128(sh),_mm256_castsi256_si128(sv));
    const __m256i hi=softmenon_blend256(_mm256_extracti128_si256(gh,1),_mm256_extracti128_si256(gv,1),_mm256_extracti128_si256(sh,1),_mm256_extracti128_si256(sv,1));
    const __m256i words=_mm256_permute4x64_epi64(_mm256_packs_epi32(lo,hi),0xd8);
    return _mm_packus_epi16(_mm256_castsi256_si128(words),_mm256_extracti128_si256(words,1));
}
template<int Parity>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi")))
void softmenon_row512(const uint8_t* raw,int rp,uint8_t* bgr,int width) {
    for(int x=0;x<width;x+=32) {
        const __m128i estimates=softmenon_estimate512<Parity>(raw+x,rp);
        const __m512i values=_mm512_inserti32x4(_mm512_maskz_loadu_epi8(0xffffffffULL,raw+x),estimates,2);
        _mm512_storeu_si512(bgr+3*x,green_bgr_block<Parity,0>(values));
        _mm512_mask_storeu_epi8(bgr+3*x+64,0xffffffffULL,green_bgr_block<Parity,64>(values));
    }
}
template<int Parity>
__attribute__((target("avx2")))
void softmenon_row256(const uint8_t* raw,int rp,uint8_t* bgr,int width) {
    const __m128i even=_mm_setr_epi8(-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0);
    const __m128i odd=_mm_xor_si128(even,_mm_set1_epi8(-1)),zero=_mm_setzero_si128();
    for(int x=0;x<width;x+=32) {
        const __m128i estimated=softmenon_estimate256<Parity>(raw+x,rp);
        const __m128i expanded[2]={_mm_unpacklo_epi8(estimated,estimated),_mm_unpackhi_epi8(estimated,estimated)};
        for(int half=0;half<2;++half) {
            const __m128i original=_mm_loadu_si128(reinterpret_cast<const __m128i*>(raw+x+16*half));
            const __m128i green=_mm_blendv_epi8(expanded[half],original,Parity?even:odd);
            soft_store16(bgr+3*(x+16*half),Parity?zero:_mm_and_si128(original,even),green,
                Parity?_mm_and_si128(original,odd):zero);
        }
    }
}
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi")))
void softmenon_avx512(const uint8_t* raw,int rp,uint8_t* out,int op,int w,int h) {
    const int vw=w&~31;
    for(int y=0;y<h;y+=2) {
        softmenon_row512<0>(raw+ptrdiff_t(y)*rp,rp,out+ptrdiff_t(y)*op,vw);
        softmenon_row512<1>(raw+ptrdiff_t(y+1)*rp,rp,out+ptrdiff_t(y+1)*op,vw);
        if(vw<w)softmenon_scalar(raw+ptrdiff_t(y)*rp+vw,rp,out+ptrdiff_t(y)*op+3*vw,op,w-vw,2);
    }
}
__attribute__((target("avx2")))
void softmenon_avx2(const uint8_t* raw,int rp,uint8_t* out,int op,int w,int h) {
    const int vw=w&~31;
    for(int y=0;y<h;y+=2) {
        softmenon_row256<0>(raw+ptrdiff_t(y)*rp,rp,out+ptrdiff_t(y)*op,vw);
        softmenon_row256<1>(raw+ptrdiff_t(y+1)*rp,rp,out+ptrdiff_t(y+1)*op,vw);
        if(vw<w)softmenon_scalar(raw+ptrdiff_t(y)*rp+vw,rp,out+ptrdiff_t(y)*op+3*vw,op,w-vw,2);
    }
}
}
#undef SM512
#endif
namespace {
inline void softmenon_green_dispatch(const uint8_t* raw,int rp,uint8_t* out,int op,int w,int h) {
#if CPU_SOFT_AVX2
    if(w>=32 && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vbmi")) {
        softmenon_avx512(raw,rp,out,op,w,h);return;
    }
    if(w>=32 && __builtin_cpu_supports("avx2")){softmenon_avx2(raw,rp,out,op,w,h);return;}
#endif
    softmenon_scalar(raw,rp,out,op,w,h);
}
}
#if CPU_SOFT_AVX2
#define SMW512 __attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi"),always_inline)) inline
namespace {
template<int Parity>
SMW512 __m512i softmenon_load_wide(const uint8_t* p) {
    const __m512i x=_mm512_loadu_si512(p);
    if constexpr(Parity)return _mm512_srli_epi16(x,8);
    return _mm512_and_si512(x,_mm512_set1_epi16(255));
}
template<int Parity>
SMW512 __m512i softmenon_candidate_wide(const uint8_t* p,ptrdiff_t a) {
    const __m512i c=softmenon_load_wide<Parity>(p),l=softmenon_load_wide<Parity>(p-a),r=softmenon_load_wide<Parity>(p+a);
    const __m512i l2=softmenon_load_wide<Parity>(p-2*a),r2=softmenon_load_wide<Parity>(p+2*a);
    return _mm512_add_epi16(_mm512_srli_epi16(_mm512_add_epi16(_mm512_add_epi16(l,r),_mm512_set1_epi16(1)),1),
        _mm512_srai_epi16(_mm512_add_epi16(_mm512_sub_epi16(_mm512_slli_epi16(c,1),_mm512_add_epi16(l2,r2)),_mm512_set1_epi16(2)),2));
}
template<int Parity>
SMW512 __m512i softmenon_difference_wide(const uint8_t* p,ptrdiff_t a) {
    return _mm512_sub_epi16(softmenon_load_wide<Parity>(p),softmenon_candidate_wide<Parity>(p,a));
}
template<int Parity>
SMW512 __m512i softmenon_colocated_wide(const uint8_t* p,ptrdiff_t a) {
    const __m512i c=softmenon_difference_wide<Parity>(p,a);
    return _mm512_add_epi16(_mm512_abs_epi16(_mm512_sub_epi16(c,softmenon_difference_wide<Parity>(p-2*a,a))),
                  _mm512_abs_epi16(_mm512_sub_epi16(c,softmenon_difference_wide<Parity>(p+2*a,a))));
}
template<int Parity>
SMW512 __m512i softmenon_score_wide(const uint8_t* p,ptrdiff_t a,ptrdiff_t b) {
    const __m512i c=softmenon_colocated_wide<Parity>(p,a);
    const __m512i parallel=_mm512_add_epi16(softmenon_colocated_wide<Parity>(p-2*b,a),softmenon_colocated_wide<Parity>(p+2*b,a));
    const __m512i corners=_mm512_add_epi16(
        _mm512_abs_epi16(_mm512_sub_epi16(softmenon_difference_wide<Parity>(p-a-b,a),softmenon_difference_wide<Parity>(p+a-b,a))),
        _mm512_abs_epi16(_mm512_sub_epi16(softmenon_difference_wide<Parity>(p-a+b,a),softmenon_difference_wide<Parity>(p+a+b,a))));
    return _mm512_add_epi16(_mm512_add_epi16(c,_mm512_slli_epi16(c,1)),_mm512_add_epi16(parallel,corners));
}
template<int Parity>
SMW512 __m256i softmenon_estimate_wide(const uint8_t* p,ptrdiff_t pitch) {
    const __m512i gh=softmenon_candidate_wide<Parity>(p,1),gv=softmenon_candidate_wide<Parity>(p,pitch);
    const __m512i sh=softmenon_score_wide<Parity>(p,1,pitch),sv=softmenon_score_wide<Parity>(p,pitch,1);
    const __m512i lo=softmenon_blend512(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(gh)),_mm512_cvtepi16_epi32(_mm512_castsi512_si256(gv)),
        _mm512_cvtepi16_epi32(_mm512_castsi512_si256(sh)),_mm512_cvtepi16_epi32(_mm512_castsi512_si256(sv)));
    const __m512i hi=softmenon_blend512(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(gh,1)),_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(gv,1)),
        _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(sh,1)),_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(sv,1)));
    return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm512_cvtusepi32_epi8(lo)),_mm512_cvtusepi32_epi8(hi),1);
}
template<int Parity,int Offset>
SMW512 __m512i softmenon_wide_bgr(__m512i raw,__m512i green) {
    alignas(64) static constexpr auto index=[] {
        std::array<uint8_t,64> a{};
        for(int i=0;i<64;++i){const int j=i+Offset,x=j/3,c=j%3;a[i]=(c==1 && (x&1)==Parity)?64+x/2:x;}
        return a;
    }();
    static constexpr uint64_t keep=[] {
        uint64_t m=0;
        for(int i=0;i<64;++i){const int j=i+Offset,x=j/3,c=j%3;if(c==1 || ((x&1)==Parity && c==2*Parity))m|=uint64_t(1)<<i;}
        return m;
    }();
    return _mm512_maskz_permutex2var_epi8(keep,raw,_mm512_load_si512(index.data()),green);
}
template<int Parity>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi")))
void softmenon_row_wide(const uint8_t* raw,int rp,uint8_t* out,int width) {
    for(int x=0;x<width;x+=64) {
        const __m512i green=_mm512_castsi256_si512(softmenon_estimate_wide<Parity>(raw+x,rp));
        const __m512i original=_mm512_loadu_si512(raw+x);
        _mm512_storeu_si512(out+3*x,softmenon_wide_bgr<Parity,0>(original,green));
        _mm512_storeu_si512(out+3*x+64,softmenon_wide_bgr<Parity,64>(original,green));
        _mm512_storeu_si512(out+3*x+128,softmenon_wide_bgr<Parity,128>(original,green));
    }
}
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi")))
void softmenon_wide_avx512(const uint8_t* raw,int rp,uint8_t* out,int op,int w,int h) {
    const int vw=w&~63;
    for(int y=0;y<h;y+=2) {
        softmenon_row_wide<0>(raw+ptrdiff_t(y)*rp,rp,out+ptrdiff_t(y)*op,vw);
        softmenon_row_wide<1>(raw+ptrdiff_t(y+1)*rp,rp,out+ptrdiff_t(y+1)*op,vw);
        if(vw<w)softmenon_avx512(raw+ptrdiff_t(y)*rp+vw,rp,out+ptrdiff_t(y)*op+3*vw,op,w-vw,2);
    }
}
}
#undef SMW512
#endif
namespace {
inline void softmenon_wide_dispatch(const uint8_t* raw,int rp,uint8_t* out,int op,int w,int h) {
#if CPU_SOFT_AVX2
    if(w>=64 && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vbmi")) {
        softmenon_wide_avx512(raw,rp,out,op,w,h);return;
    }
#endif
    softmenon_green_dispatch(raw,rp,out,op,w,h);
}
}
#if CPU_SOFT_AVX2
#define SMC512 __attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi"),always_inline)) inline
namespace {
SMC512 __m512i cache_load(const int16_t* p){return _mm512_loadu_si512(p);}
SMC512 __m512i cache_diff(__m512i a,__m512i b){return _mm512_abs_epi16(_mm512_sub_epi16(a,b));}
SMC512 __m512i cache_pair(const int16_t* p,ptrdiff_t a){const __m512i c=cache_load(p);return _mm512_add_epi16(cache_diff(c,cache_load(p-a)),cache_diff(c,cache_load(p+a)));}
template<int Parity>
SMC512 __m256i cache_estimate(const uint8_t* raw,const int16_t* h,const int16_t* v,ptrdiff_t sp) {
    const __m512i center=softmenon_load_wide<Parity>(raw);
    const __m512i gh=_mm512_sub_epi16(center,cache_load(h)),gv=_mm512_sub_epi16(center,cache_load(v));
    // A +/-1 raw-x step between opposite parity rows changes the packed
    // column by {-1,0} on even rows and {0,+1} on odd rows.
    constexpr int left=Parity-1,right=Parity;
    const __m512i hc=cache_pair(h,1),vc=cache_pair(v,2*sp);
    const __m512i hp=_mm512_add_epi16(cache_pair(h-2*sp,1),cache_pair(h+2*sp,1));
    const __m512i vp=_mm512_add_epi16(cache_pair(v-1,2*sp),cache_pair(v+1,2*sp));
    const __m512i hd=_mm512_add_epi16(cache_diff(cache_load(h-sp+left),cache_load(h-sp+right)),cache_diff(cache_load(h+sp+left),cache_load(h+sp+right)));
    const __m512i vd=_mm512_add_epi16(cache_diff(cache_load(v-sp+left),cache_load(v+sp+left)),cache_diff(cache_load(v-sp+right),cache_load(v+sp+right)));
    const __m512i sh=_mm512_add_epi16(_mm512_add_epi16(hc,_mm512_slli_epi16(hc,1)),_mm512_add_epi16(hp,hd));
    const __m512i sv=_mm512_add_epi16(_mm512_add_epi16(vc,_mm512_slli_epi16(vc,1)),_mm512_add_epi16(vp,vd));
    const __m512i lo=softmenon_blend512(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(gh)),_mm512_cvtepi16_epi32(_mm512_castsi512_si256(gv)),
        _mm512_cvtepi16_epi32(_mm512_castsi512_si256(sh)),_mm512_cvtepi16_epi32(_mm512_castsi512_si256(sv)));
    const __m512i hi=softmenon_blend512(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(gh,1)),_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(gv,1)),
        _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(sh,1)),_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(sv,1)));
    return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm512_cvtusepi32_epi8(lo)),_mm512_cvtusepi32_epi8(hi),1);
}
template<int Parity>
SMC512 void cache_fill(const uint8_t* p,int rp,int16_t* h,int16_t* v,int count) {
    // count>=32 on this path. Overlap the final complete vector to avoid
    // both a scalar tail and any read beyond the radius-two D halo.
    for(int i=0;;i+=32){if(i+32>count)i=count-32;
        const __m512i c=softmenon_load_wide<Parity>(p+2*i);
        _mm512_storeu_si512(h+i,_mm512_sub_epi16(c,softmenon_candidate_wide<Parity>(p+2*i,1)));
        _mm512_storeu_si512(v+i,_mm512_sub_epi16(c,softmenon_candidate_wide<Parity>(p+2*i,rp)));
        if(i+32==count)break;
    }
}
template<int Parity>
SMC512 void cache_output(const uint8_t* raw,uint8_t* out,const int16_t* h,const int16_t* v,int sp,int width) {
    for(int x=0;;x+=64){
        if(x+64>width)x=width-64;
        const __m512i green=_mm512_castsi256_si512(cache_estimate<Parity>(raw+x,h+x/2,v+x/2,sp));
        const __m512i original=_mm512_loadu_si512(raw+x);
        _mm512_storeu_si512(out+3*x,softmenon_wide_bgr<Parity,0>(original,green));
        _mm512_storeu_si512(out+3*x+64,softmenon_wide_bgr<Parity,64>(original,green));
        _mm512_storeu_si512(out+3*x+128,softmenon_wide_bgr<Parity,128>(original,green));
        if(x+64==width)break;
    }
}
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi")))
void softmenon_cached512(const uint8_t* raw,int rp,uint8_t* out,int op,int w,int height) {
    if(w<64){softmenon_wide_avx512(raw,rp,out,op,w,height);return;}
    const int cols=(w+4)/2,sp=(cols+31)&~31;
    thread_local std::vector<int16_t> dh,dv;
    // Complete every possibly-throwing operation before writing any output.
    // Allocation failure takes the allocation-free exact fallback before
    // any pixels are written, preserving the same output arithmetic.
    try {
        const size_t size=size_t(std::min(height,128)+4)*sp;
        dh.resize(size);dv.resize(size);
    }catch(...){softmenon_wide_avx512(raw,rp,out,op,w,height);return;}
    for(int begin=0;begin<height;begin+=128){
        const int rows=std::min(128,height-begin);
        for(int y=-2;y<rows+2;++y){
            const uint8_t* p=raw+ptrdiff_t(begin+y)*rp-2;
            int16_t* h=dh.data()+ptrdiff_t(y+2)*sp;int16_t* v=dv.data()+ptrdiff_t(y+2)*sp;
            if(y&1)cache_fill<1>(p,rp,h,v,cols);else cache_fill<0>(p,rp,h,v,cols);
        }
        for(int y=0;y<rows;y+=2){
            const uint8_t* p=raw+ptrdiff_t(begin+y)*rp;uint8_t* q=out+ptrdiff_t(begin+y)*op;
            const int16_t* h=dh.data()+ptrdiff_t(y+2)*sp+1;const int16_t* v=dv.data()+ptrdiff_t(y+2)*sp+1;
            cache_output<0>(p,q,h,v,sp,w);cache_output<1>(p+rp,q+op,h+sp,v+sp,sp,w);
        }
    }
}
}
#undef SMC512
#endif
namespace {
inline void softmenon_cached_dispatch(const uint8_t* raw,int rp,uint8_t* out,int op,int w,int h) {
#if CPU_SOFT_AVX2
    if(w>=64 && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vbmi")) {
        softmenon_cached512(raw,rp,out,op,w,h);return;
    }
#endif
    softmenon_wide_dispatch(raw,rp,out,op,w,h);
}
}
