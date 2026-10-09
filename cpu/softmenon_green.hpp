#pragma once

// Exact SoftMenon green: evaluate only the 16 chromatic sites
// in a 32-pixel strip. Inputs have the same two-pixel halo as the AVX2 core.
// Estimates, scores, division and clipping match the scalar and AVX2 cores.
#if CPU_SOFT_AVX2
#define SG512 __attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi"),always_inline)) inline
namespace {
template <int Parity>
SG512 __m512i green_load_chromatic(const uint8_t* p) {
    alignas(64) static constexpr auto index=[] {
        std::array<uint8_t,64> a{};
        for(int i=0;i<16;++i) a[i]=2*i+Parity;
        return a;
    }();
    const __m512i bytes=_mm512_maskz_loadu_epi8(0xffffffffULL,p);
    return _mm512_cvtepu8_epi32(_mm512_castsi512_si128(
        _mm512_permutexvar_epi8(_mm512_load_si512(index.data()),bytes)));
}

template <int Parity>
SG512 __m128i green_estimate_chromatic(const uint8_t* p,ptrdiff_t pitch) {
    const __m512i one=_mm512_set1_epi32(1),two=_mm512_set1_epi32(2);
    const __m512i c=green_load_chromatic<Parity>(p);
    const __m512i l=green_load_chromatic<Parity>(p-1),r=green_load_chromatic<Parity>(p+1);
    const __m512i u=green_load_chromatic<Parity>(p-pitch),d=green_load_chromatic<Parity>(p+pitch);
    const __m512i l2=green_load_chromatic<Parity>(p-2),r2=green_load_chromatic<Parity>(p+2);
    const __m512i u2=green_load_chromatic<Parity>(p-2*pitch),d2=green_load_chromatic<Parity>(p+2*pitch);
    const __m512i twice=_mm512_slli_epi32(c,1);
    const __m512i gh=_mm512_add_epi32(_mm512_srli_epi32(_mm512_add_epi32(_mm512_add_epi32(l,r),one),1),
        _mm512_srai_epi32(_mm512_add_epi32(_mm512_sub_epi32(twice,_mm512_add_epi32(l2,r2)),two),2));
    const __m512i gv=_mm512_add_epi32(_mm512_srli_epi32(_mm512_add_epi32(_mm512_add_epi32(u,d),one),1),
        _mm512_srai_epi32(_mm512_add_epi32(_mm512_sub_epi32(twice,_mm512_add_epi32(u2,d2)),two),2));
    const __m512i ch=_mm512_sub_epi32(c,gh),cv=_mm512_sub_epi32(c,gv);
    const __m512i sh=_mm512_add_epi32(one,_mm512_add_epi32(_mm512_abs_epi32(_mm512_sub_epi32(ch,_mm512_sub_epi32(l2,l))),
        _mm512_abs_epi32(_mm512_sub_epi32(ch,_mm512_sub_epi32(r2,r)))));
    const __m512i sv=_mm512_add_epi32(one,_mm512_add_epi32(_mm512_abs_epi32(_mm512_sub_epi32(cv,_mm512_sub_epi32(u2,u))),
        _mm512_abs_epi32(_mm512_sub_epi32(cv,_mm512_sub_epi32(d2,d)))));
    const __m512i den=_mm512_add_epi32(sh,sv);
    const __m512i num=_mm512_add_epi32(_mm512_mullo_epi32(sv,gh),_mm512_mullo_epi32(sh,gv));
    const __m512i rounded=_mm512_max_epi32(_mm512_add_epi32(num,_mm512_srli_epi32(den,1)),_mm512_setzero_si512());
    // Exact integer quotient in this bounded domain: denominator <=2554,
    // 0<=rounded/denominator<384, and rounded<2^20 is exactly representable.
    // Distance to an integer boundary is >=1/2554, over 12 float32 ULPs;
    // exact integer quotients are representable. Negative results clip to 0.
    // Use actual division, never an approximate reciprocal.
    const __m512i quotient=_mm512_cvttps_epi32(_mm512_div_ps(_mm512_cvtepi32_ps(rounded),_mm512_cvtepi32_ps(den)));
    return _mm512_cvtusepi32_epi8(quotient);
}

template <int Parity,int Offset>
SG512 __m512i green_bgr_block(__m512i values) {
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
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi")))
void green_avx512_one_row(const uint8_t* raw,int rp,uint8_t* bgr,int width) {
    for(int x=0;x<width;x+=32) {
        const __m128i estimates=green_estimate_chromatic<Parity>(raw+x,rp);
        // bits 0..255 hold raw32, bits 256..383 hold the 16 green estimates.
        const __m512i values=_mm512_inserti32x4(_mm512_maskz_loadu_epi8(0xffffffffULL,raw+x),estimates,2);
        _mm512_storeu_si512(bgr+3*x,green_bgr_block<Parity,0>(values));
        _mm512_mask_storeu_epi8(bgr+3*x+64,0xffffffffULL,green_bgr_block<Parity,64>(values));
    }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512vbmi")))
void soft_green_avx512(const uint8_t* raw,int rp,uint8_t* bgr,int bp,int width,int height) {
    const int vector_width=width&~31;
    for(int y=0;y<height;y+=2) {
        green_avx512_one_row<0>(raw+size_t(y)*rp,rp,bgr+size_t(y)*bp,vector_width);
        green_avx512_one_row<1>(raw+size_t(y+1)*rp,rp,bgr+size_t(y+1)*bp,vector_width);
        if(vector_width<width)soft_green_avx2(raw+size_t(y)*rp+vector_width,rp,
            bgr+size_t(y)*bp+vector_width*3,bp,width-vector_width,2);
    }
}
}
#undef SG512
#endif

// AVX2 evaluates the same chromatic sites using packed byte-pair extraction.
#if CPU_SOFT_AVX2
namespace {
template <int Parity>
SOFT_INLINE __m256i soft_load_chromatic16(const uint8_t* p) {
    const __m256i raw=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    // A RAW byte pair is already a uint16 lane: select the chromatic byte
    // directly, avoiding byte gathers and any work at observed green sites.
    if constexpr (Parity) return _mm256_srli_epi16(raw,8);
    return _mm256_and_si256(raw,_mm256_set1_epi16(255));
}

template <int Parity>
SOFT_INLINE __m128i soft_green_chromatic16(const uint8_t* p,ptrdiff_t pitch) {
    const __m256i one=_mm256_set1_epi16(1),two=_mm256_set1_epi16(2);
    const __m256i center=soft_load_chromatic16<Parity>(p);
    const __m256i left=soft_load_chromatic16<Parity>(p-1),right=soft_load_chromatic16<Parity>(p+1);
    const __m256i up=soft_load_chromatic16<Parity>(p-pitch),down=soft_load_chromatic16<Parity>(p+pitch);
    const __m256i left2=soft_load_chromatic16<Parity>(p-2),right2=soft_load_chromatic16<Parity>(p+2);
    const __m256i up2=soft_load_chromatic16<Parity>(p-2*pitch),down2=soft_load_chromatic16<Parity>(p+2*pitch);
    const __m256i twice=_mm256_slli_epi16(center,1);
    const __m256i gh=_mm256_add_epi16(_mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(left,right),one),1),
        _mm256_srai_epi16(_mm256_add_epi16(_mm256_sub_epi16(twice,_mm256_add_epi16(left2,right2)),two),2));
    const __m256i gv=_mm256_add_epi16(_mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(up,down),one),1),
        _mm256_srai_epi16(_mm256_add_epi16(_mm256_sub_epi16(twice,_mm256_add_epi16(up2,down2)),two),2));
    const __m256i ch=_mm256_sub_epi16(center,gh),cv=_mm256_sub_epi16(center,gv);
    const __m256i sh=_mm256_add_epi16(_mm256_abs_epi16(_mm256_sub_epi16(ch,_mm256_sub_epi16(left2,left))),
                                    _mm256_abs_epi16(_mm256_sub_epi16(ch,_mm256_sub_epi16(right2,right))));
    const __m256i sv=_mm256_add_epi16(_mm256_abs_epi16(_mm256_sub_epi16(cv,_mm256_sub_epi16(up2,up))),
                                    _mm256_abs_epi16(_mm256_sub_epi16(cv,_mm256_sub_epi16(down2,down))));
    const __m256i lo=soft_weighted8(_mm256_castsi256_si128(gh),_mm256_castsi256_si128(gv),
                                   _mm256_castsi256_si128(sh),_mm256_castsi256_si128(sv));
    const __m256i hi=soft_weighted8(_mm256_extracti128_si256(gh,1),_mm256_extracti128_si256(gv,1),
                                   _mm256_extracti128_si256(sh,1),_mm256_extracti128_si256(sv,1));
    const __m256i words=_mm256_permute4x64_epi64(_mm256_packs_epi32(lo,hi),0xd8);
    return _mm_packus_epi16(_mm256_castsi256_si128(words),_mm256_extracti128_si256(words,1));
}

template <int Parity>
__attribute__((target("avx2")))
void soft_green_chromatic_row_avx2(const uint8_t* raw,int rp,uint8_t* bgr,int width) {
    const __m128i even=_mm_setr_epi8(-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0);
    const __m128i odd=_mm_xor_si128(even,_mm_set1_epi8(-1)),zero=_mm_setzero_si128();
    for(int x=0;x<width;x+=32) {
        const __m128i estimated=soft_green_chromatic16<Parity>(raw+x,rp);
        const __m128i expanded[2]={_mm_unpacklo_epi8(estimated,estimated),_mm_unpackhi_epi8(estimated,estimated)};
        for(int half=0;half<2;++half) {
            const __m128i original=_mm_loadu_si128(reinterpret_cast<const __m128i*>(raw+x+16*half));
            const __m128i green=_mm_blendv_epi8(expanded[half],original,Parity ? even : odd);
            soft_store16(bgr+3*(x+16*half),Parity ? zero : _mm_and_si128(original,even),green,
                         Parity ? _mm_and_si128(original,odd) : zero);
        }
    }
}

__attribute__((target("avx2")))
void soft_green_chromatic_avx2(const uint8_t* raw,int rp,uint8_t* bgr,int bp,int width,int height) {
    const int vector_width=width&~31;
    for(int y=0;y<height;y+=2) {
        soft_green_chromatic_row_avx2<0>(raw+size_t(y)*rp,rp,bgr+size_t(y)*bp,vector_width);
        soft_green_chromatic_row_avx2<1>(raw+size_t(y+1)*rp,rp,bgr+size_t(y+1)*bp,vector_width);
        if(vector_width<width)soft_green_avx2(raw+size_t(y)*rp+vector_width,rp,
            bgr+size_t(y)*bp+vector_width*3,bp,width-vector_width,2);
    }
}
}
#endif
