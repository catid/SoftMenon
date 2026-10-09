#pragma once

// Exact immutable-source 3x3 median of B-G and R-G. Source and destination
// images must not overlap; pitches are independent byte strides. Row ranges
// make calls independent, so the caller can use its existing worker pool.
// 0: insertion reference. 1: fixed scalar network. 2: runtime-selected SIMD
// network (AVX512BW+VBMI, AVX2, then scalar fallback), with scalar borders/tails.
// swap_output exchanges the final B/R channels; CFA masking always describes
// the SOURCE image. Every measured CFA sample is preserved. At red/blue
// sites, green is reconstructed from the measured color minus its median
// chroma; the other color uses this clipped green plus its own median chroma.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define MEDIAN_CPU_HAS_AVX2 1
#define MEDIAN_CPU_AVX2_INLINE __attribute__((target("avx2"), always_inline)) inline
#define MEDIAN_CPU_AVX2_TARGET __attribute__((target("avx2")))
#define MEDIAN_CPU_AVX512_INLINE __attribute__((target("avx512f,avx512bw,avx512vbmi"), always_inline)) inline
#define MEDIAN_CPU_AVX512_TARGET __attribute__((target("avx512f,avx512bw,avx512vbmi")))
#else
#define MEDIAN_CPU_HAS_AVX2 0
#endif

namespace median_cpu_detail {

inline void compare_exchange(int& a, int& b) {
    const int low = std::min(a, b);
    b = std::max(a, b);
    a = low;
}

inline int median9(int p0, int p1, int p2, int p3, int p4,
                   int p5, int p6, int p7, int p8) {
    compare_exchange(p1,p2); compare_exchange(p4,p5); compare_exchange(p7,p8);
    compare_exchange(p0,p1); compare_exchange(p3,p4); compare_exchange(p6,p7);
    compare_exchange(p1,p2); compare_exchange(p4,p5); compare_exchange(p7,p8);
    compare_exchange(p0,p3); compare_exchange(p5,p8); compare_exchange(p4,p7);
    compare_exchange(p3,p6); compare_exchange(p1,p4); compare_exchange(p2,p5);
    compare_exchange(p4,p7); compare_exchange(p4,p2); compare_exchange(p6,p4);
    compare_exchange(p4,p2);
    return p4;
}

inline int insertion_median(int (&values)[9]) {
    for (int i=1; i<9; ++i) {
        const int value=values[i];
        int j=i;
        while (j>0 && values[j-1]>value) {
            values[j]=values[j-1];
            --j;
        }
        values[j]=value;
    }
    return values[4];
}

inline uint8_t saturated(int value) {
    return static_cast<uint8_t>(std::max(0,std::min(value,255)));
}

template <bool Insertion>
inline void scalar_pixel(const uint8_t* source, size_t source_pitch,
        uint8_t* destination, size_t destination_pitch, int width, int height,
        int x, int y, bool rggb, bool swap_output) {
    // Only radius-one coordinates are needed, so singleton dimensions and
    // both endpoints can be reflected without a modulo operation.
    const int xs[3]={x>0 ? x-1 : std::min(1,width-1), x,
                     x<width-1 ? x+1 : std::max(width-2,0)};
    const int ys[3]={y>0 ? y-1 : std::min(1,height-1), y,
                     y<height-1 ? y+1 : std::max(height-2,0)};
    int blue[9],red[9];
    for (int dy=0;dy<3;++dy) {
        const uint8_t* row=source+static_cast<size_t>(ys[dy])*source_pitch;
        for (int dx=0;dx<3;++dx) {
            const uint8_t* pixel=row+static_cast<size_t>(xs[dx])*3;
            blue[dy*3+dx]=static_cast<int>(pixel[0])-pixel[1];
            red[dy*3+dx]=static_cast<int>(pixel[2])-pixel[1];
        }
    }
    int b,r;
    if constexpr (Insertion) {
        b=insertion_median(blue); r=insertion_median(red);
    } else {
        b=median9(blue[0],blue[1],blue[2],blue[3],blue[4],blue[5],blue[6],blue[7],blue[8]);
        r=median9(red[0],red[1],red[2],red[3],red[4],red[5],red[6],red[7],red[8]);
    }
    const uint8_t* center=source+static_cast<size_t>(y)*source_pitch+static_cast<size_t>(x)*3;
    uint8_t* output=destination+static_cast<size_t>(y)*destination_pitch+static_cast<size_t>(x)*3;
    const bool chromatic=(x&1)==(y&1);
    const bool measured_red=chromatic && (((x&1)==0)==rggb);
    const bool measured_blue=chromatic && !measured_red;
    const int green=measured_blue ? saturated(center[0]-b) :
        (measured_red ? saturated(center[2]-r) : center[1]);
    const uint8_t out_b=measured_blue ? center[0] : saturated(green+b);
    const uint8_t out_r=measured_red ? center[2] : saturated(green+r);
    output[0]=swap_output ? out_r : out_b;
    output[1]=static_cast<uint8_t>(green);
    output[2]=swap_output ? out_b : out_r;
}

template <bool Insertion>
inline void scalar_rows(const uint8_t* source, size_t source_pitch,
        uint8_t* destination, size_t destination_pitch, int width, int height,
        bool rggb, int begin_y, int end_y, bool swap_output) {
    for (int y=begin_y;y<end_y;++y)
        for (int x=0;x<width;++x)
            scalar_pixel<Insertion>(source,source_pitch,destination,destination_pitch,
                                    width,height,x,y,rggb,swap_output);
}

#if MEDIAN_CPU_HAS_AVX2
MEDIAN_CPU_AVX2_INLINE void compare_exchange(__m256i& a,__m256i& b) {
    const __m256i low=_mm256_min_epi16(a,b);
    b=_mm256_max_epi16(a,b);
    a=low;
}

MEDIAN_CPU_AVX2_INLINE __m256i median9(__m256i p0,__m256i p1,__m256i p2,
        __m256i p3,__m256i p4,__m256i p5,__m256i p6,__m256i p7,__m256i p8) {
    compare_exchange(p1,p2); compare_exchange(p4,p5); compare_exchange(p7,p8);
    compare_exchange(p0,p1); compare_exchange(p3,p4); compare_exchange(p6,p7);
    compare_exchange(p1,p2); compare_exchange(p4,p5); compare_exchange(p7,p8);
    compare_exchange(p0,p3); compare_exchange(p5,p8); compare_exchange(p4,p7);
    compare_exchange(p3,p6); compare_exchange(p1,p4); compare_exchange(p2,p5);
    compare_exchange(p4,p7); compare_exchange(p4,p2); compare_exchange(p6,p4);
    compare_exchange(p4,p2);
    return p4;
}

constexpr int first_index(int index) { return index<16 ? index : -128; }
constexpr int second_index(int index) { return index>=16 ? index-14 : -128; }

struct Chroma {
    __m256i differences;
    __m256i green;
};

// Two bounded 16-byte loads cover exactly the ten-pixel/30-byte row window:
// first = bytes 0..15, second = bytes 14..29. Offset is 0, 3, or 6 bytes.
template <int Offset>
MEDIAN_CPU_AVX2_INLINE Chroma row_chroma(__m128i first,__m128i second) {
#define MC_COLOR_INDEX(F) _mm_setr_epi8(F(Offset),F(Offset+2),F(Offset+3),F(Offset+5), \
    F(Offset+6),F(Offset+8),F(Offset+9),F(Offset+11),F(Offset+12),F(Offset+14), \
    F(Offset+15),F(Offset+17),F(Offset+18),F(Offset+20),F(Offset+21),F(Offset+23))
#define MC_GREEN_INDEX(F) _mm_setr_epi8(F(Offset+1),F(Offset+1),F(Offset+4),F(Offset+4), \
    F(Offset+7),F(Offset+7),F(Offset+10),F(Offset+10),F(Offset+13),F(Offset+13), \
    F(Offset+16),F(Offset+16),F(Offset+19),F(Offset+19),F(Offset+22),F(Offset+22))
    const __m128i colors=_mm_or_si128(_mm_shuffle_epi8(first,MC_COLOR_INDEX(first_index)),
                                     _mm_shuffle_epi8(second,MC_COLOR_INDEX(second_index)));
    const __m128i greens=_mm_or_si128(_mm_shuffle_epi8(first,MC_GREEN_INDEX(first_index)),
                                     _mm_shuffle_epi8(second,MC_GREEN_INDEX(second_index)));
#undef MC_COLOR_INDEX
#undef MC_GREEN_INDEX
    const __m256i green=_mm256_cvtepu8_epi16(greens);
    return {_mm256_sub_epi16(_mm256_cvtepu8_epi16(colors),green),green};
}

MEDIAN_CPU_AVX2_INLINE __m128i clipped_bytes(__m256i values) {
    const __m256i packed=_mm256_packus_epi16(values,_mm256_setzero_si256());
    return _mm_unpacklo_epi64(_mm256_castsi256_si128(packed),_mm256_extracti128_si256(packed,1));
}

MEDIAN_CPU_AVX2_TARGET inline void avx2_rows(const uint8_t* source,size_t source_pitch,
        uint8_t* destination,size_t destination_pitch,int width,int height,
        bool rggb,int begin_y,int end_y,bool swap_output) {
    const __m128i scatter_first=_mm_setr_epi8(0,-128,1,2,-128,3,4,-128,5,6,-128,7,8,-128,9,10);
    const __m128i scatter_second=_mm_setr_epi8(5,6,-128,7,8,-128,9,10,-128,11,12,-128,13,14,-128,15);
    const __m128i swap_pairs=_mm_setr_epi8(1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14);
    for (int y=begin_y;y<end_y;++y) {
        if (y==0 || y==height-1 || width<10) {
            for (int x=0;x<width;++x)
                scalar_pixel<false>(source,source_pitch,destination,destination_pitch,width,height,x,y,rggb,swap_output);
            continue;
        }
        scalar_pixel<false>(source,source_pitch,destination,destination_pitch,width,height,0,y,rggb,swap_output);
        // Every eight-pixel vector starts at odd x (1,9,...). A -1 lane keeps
        // an observed B/R sample; all green values are restored separately.
        const short blue0=((y&1)!=0 && rggb) ? -1 : 0;
        const short red0=((y&1)!=0 && !rggb) ? -1 : 0;
        const short blue1=((y&1)==0 && !rggb) ? -1 : 0;
        const short red1=((y&1)==0 && rggb) ? -1 : 0;
        const __m256i measured=_mm256_setr_epi16(blue0,red0,blue1,red1,blue0,red0,blue1,red1,
                                                blue0,red0,blue1,red1,blue0,red0,blue1,red1);
        int x=1;
        for (;x<=width-9;x+=8) {
            __m256i values[9],center_green=_mm256_setzero_si256();
            for (int dy=-1;dy<=1;++dy) {
                const uint8_t* input=source+static_cast<size_t>(y+dy)*source_pitch+static_cast<size_t>(x-1)*3;
                const __m128i first=_mm_loadu_si128(reinterpret_cast<const __m128i*>(input));
                const __m128i second=_mm_loadu_si128(reinterpret_cast<const __m128i*>(input+14));
                const Chroma left=row_chroma<0>(first,second);
                const Chroma center=row_chroma<3>(first,second);
                const Chroma right=row_chroma<6>(first,second);
                values[(dy+1)*3]=left.differences;
                values[(dy+1)*3+1]=center.differences;
                values[(dy+1)*3+2]=right.differences;
                if (dy==0) center_green=center.green;
            }
            const __m256i med=median9(values[0],values[1],values[2],
                values[3],values[4],values[5],values[6],values[7],values[8]);
            const __m256i original=_mm256_add_epi16(center_green,values[4]);
            const __m256i delta=_mm256_and_si256(measured,_mm256_sub_epi16(original,med));
            // Shuffle both halves in sequence; disjoint measured lanes avoid
            // addition/cross-channel interference when duplicating each pair.
            const __m256i duplicated=_mm256_or_si256(delta,
                _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(delta,0xb1),0xb1));
            const __m256i masks=_mm256_or_si256(measured,
                _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(measured,0xb1),0xb1));
            const __m256i green=_mm256_min_epi16(_mm256_max_epi16(
                _mm256_blendv_epi8(center_green,duplicated,masks),_mm256_setzero_si256()),_mm256_set1_epi16(255));
            const __m256i filtered=_mm256_add_epi16(green,med);
            __m128i colors=clipped_bytes(_mm256_blendv_epi8(filtered,original,measured));
            if (swap_output) colors=_mm_shuffle_epi8(colors,swap_pairs);
            uint8_t* output=destination+static_cast<size_t>(y)*destination_pitch+static_cast<size_t>(x)*3;
            const __m128i greens=clipped_bytes(green);
            const __m128i first=_mm_or_si128(_mm_shuffle_epi8(colors,scatter_first),
                _mm_shuffle_epi8(greens,_mm_setr_epi8(-128,0,-128,-128,2,-128,-128,4,-128,-128,6,-128,-128,8,-128,-128)));
            const __m128i second=_mm_or_si128(_mm_shuffle_epi8(colors,scatter_second),
                _mm_shuffle_epi8(greens,_mm_setr_epi8(-128,-128,6,-128,-128,8,-128,-128,10,-128,-128,12,-128,-128,14,-128)));
            // The stores overlap only within this vector's 24 output bytes;
            // the overlapping bytes agree. Adjacent vectors/rows never overlap.
            _mm_storeu_si128(reinterpret_cast<__m128i*>(output),first);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(output+8),second);
        }
        for (;x<width;++x)
            scalar_pixel<false>(source,source_pitch,destination,destination_pitch,width,height,x,y,rggb,swap_output);
    }
}

MEDIAN_CPU_AVX512_INLINE void compare_exchange(__m512i& a,__m512i& b) {
    const __m512i low=_mm512_min_epi16(a,b);
    b=_mm512_max_epi16(a,b);
    a=low;
}

MEDIAN_CPU_AVX512_INLINE __m512i median9(__m512i p0,__m512i p1,__m512i p2,
        __m512i p3,__m512i p4,__m512i p5,__m512i p6,__m512i p7,__m512i p8) {
    compare_exchange(p1,p2); compare_exchange(p4,p5); compare_exchange(p7,p8);
    compare_exchange(p0,p1); compare_exchange(p3,p4); compare_exchange(p6,p7);
    compare_exchange(p1,p2); compare_exchange(p4,p5); compare_exchange(p7,p8);
    compare_exchange(p0,p3); compare_exchange(p5,p8); compare_exchange(p4,p7);
    compare_exchange(p3,p6); compare_exchange(p1,p4); compare_exchange(p2,p5);
    compare_exchange(p4,p7); compare_exchange(p4,p2); compare_exchange(p6,p4);
    compare_exchange(p4,p2);
    return p4;
}

struct Chroma512 { __m512i differences,green; };

// The source holds 18 pixels in 54 bytes. VBMI gathers sixteen B/R pairs
// and their duplicated green values before widening to signed 16-bit lanes.
template <int Offset>
MEDIAN_CPU_AVX512_INLINE Chroma512 row_chroma512(__m512i source) {
    alignas(64) static constexpr auto color_index=[] {
        std::array<uint8_t,64> index{};
        for (int i=0;i<32;++i) index[i]=Offset+3*(i/2)+2*(i&1);
        return index;
    }();
    alignas(64) static constexpr auto green_index=[] {
        std::array<uint8_t,64> index{};
        for (int i=0;i<32;++i) index[i]=Offset+3*(i/2)+1;
        return index;
    }();
    const __m512i colors=_mm512_cvtepu8_epi16(_mm512_castsi512_si256(
        _mm512_permutexvar_epi8(_mm512_load_si512(color_index.data()),source)));
    const __m512i green=_mm512_cvtepu8_epi16(_mm512_castsi512_si256(
        _mm512_permutexvar_epi8(_mm512_load_si512(green_index.data()),source)));
    return {_mm512_sub_epi16(colors,green),green};
}

MEDIAN_CPU_AVX512_TARGET inline void avx512_rows(const uint8_t* source,size_t source_pitch,
        uint8_t* destination,size_t destination_pitch,int width,int height,
        bool rggb,int begin_y,int end_y,bool swap_output) {
    alignas(64) static constexpr auto scatter_index=[] {
        std::array<uint8_t,64> index{};
        for (int i=0;i<48;++i) index[i]=2*(i/3)+(i%3==2);
        return index;
    }();
    constexpr __mmask64 green_mask=[] {
        uint64_t mask=0;
        for (int i=1;i<48;i+=3) mask|=uint64_t{1}<<i;
        return mask;
    }();
    __m512i scatter=_mm512_load_si512(scatter_index.data());
    if (swap_output) scatter=_mm512_xor_si512(scatter,_mm512_set1_epi8(1));
    for (int y=begin_y;y<end_y;++y) {
        if (y==0 || y==height-1 || width<18) {
            for (int x=0;x<width;++x)
                scalar_pixel<false>(source,source_pitch,destination,destination_pitch,width,height,x,y,rggb,swap_output);
            continue;
        }
        scalar_pixel<false>(source,source_pitch,destination,destination_pitch,width,height,0,y,rggb,swap_output);
        const __mmask32 measured=0x11111111u*((y&1) ? (rggb ? 1u : 2u) : (rggb ? 8u : 4u));
        int x=1;
        for (;x<=width-17;x+=16) {
            __m512i values[9],center_green=_mm512_setzero_si512();
            for (int dy=-1;dy<=1;++dy) {
                const uint8_t* input=source+static_cast<size_t>(y+dy)*source_pitch+static_cast<size_t>(x-1)*3;
                const __m512i row=_mm512_maskz_loadu_epi8((uint64_t{1}<<54)-1,input);
                const Chroma512 left=row_chroma512<0>(row);
                const Chroma512 center=row_chroma512<3>(row);
                const Chroma512 right=row_chroma512<6>(row);
                values[(dy+1)*3]=left.differences;
                values[(dy+1)*3+1]=center.differences;
                values[(dy+1)*3+2]=right.differences;
                if (dy==0) center_green=center.green;
            }
            const __m512i med=median9(values[0],values[1],values[2],
                values[3],values[4],values[5],values[6],values[7],values[8]);
            const __m512i original=_mm512_add_epi16(center_green,values[4]);
            const __m512i delta=_mm512_maskz_sub_epi16(measured,original,med);
            const __m512i paired=_mm512_or_si512(delta,
                _mm512_shufflehi_epi16(_mm512_shufflelo_epi16(delta,0xb1),0xb1));
            const __mmask32 chromatic=measured|((measured&0x55555555u)<<1)|((measured&0xaaaaaaaau)>>1);
            const __m512i green=_mm512_min_epi16(_mm512_max_epi16(
                _mm512_mask_mov_epi16(center_green,chromatic,paired),_mm512_setzero_si512()),_mm512_set1_epi16(255));
            const __m512i filtered=_mm512_add_epi16(green,med);
            const __m512i colors=_mm512_mask_mov_epi16(filtered,measured,original);
            const __m512i clipped=_mm512_min_epi16(_mm512_max_epi16(colors,_mm512_setzero_si512()),_mm512_set1_epi16(255));
            const __m512i bytes=_mm512_castsi256_si512(_mm512_cvtepi16_epi8(clipped));
            const __m512i output=_mm512_mask_mov_epi8(_mm512_permutexvar_epi8(scatter,bytes),green_mask,
                _mm512_permutexvar_epi8(scatter,_mm512_castsi256_si512(_mm512_cvtepi16_epi8(green))));
            // Masked stores touch exactly this vector's 16 BGR pixels.
            _mm512_mask_storeu_epi8(destination+static_cast<size_t>(y)*destination_pitch+static_cast<size_t>(x)*3,
                (uint64_t{1}<<48)-1,output);
        }
        for (;x<width;++x)
            scalar_pixel<false>(source,source_pitch,destination,destination_pitch,width,height,x,y,rggb,swap_output);
    }
}
#endif

} // namespace median_cpu_detail

inline bool median_cpu_variant_supported(int variant) {
    if (variant==0 || variant==1) return true;
#if MEDIAN_CPU_HAS_AVX2
    if (variant==2) return __builtin_cpu_supports("avx2");
#endif
    return false;
}

inline void median_cpu_rows(const uint8_t* source,size_t source_pitch,
        uint8_t* destination,size_t destination_pitch,int width,int height,
        bool rggb,int begin_y,int end_y,int variant,bool swap_output=false) {
    if (!source || !destination || source==destination || width<=0 || height<=0 || variant<0 || variant>2)
        return;
    const size_t row_bytes=static_cast<size_t>(width)*3;
    if (source_pitch<row_bytes || destination_pitch<row_bytes) return;
    begin_y=std::max(0,begin_y);
    end_y=std::min(height,end_y);
    if (begin_y>=end_y) return;
    if (variant==0) {
        median_cpu_detail::scalar_rows<true>(source,source_pitch,destination,destination_pitch,
                                            width,height,rggb,begin_y,end_y,swap_output);
        return;
    }
#if MEDIAN_CPU_HAS_AVX2
    if (variant==2 && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vbmi")) {
        median_cpu_detail::avx512_rows(source,source_pitch,destination,destination_pitch,
                                      width,height,rggb,begin_y,end_y,swap_output);
        return;
    }
    if (variant==2 && median_cpu_variant_supported(2)) {
        median_cpu_detail::avx2_rows(source,source_pitch,destination,destination_pitch,
                                    width,height,rggb,begin_y,end_y,swap_output);
        return;
    }
#endif
    median_cpu_detail::scalar_rows<false>(source,source_pitch,destination,destination_pitch,
                                         width,height,rggb,begin_y,end_y,swap_output);
}

#if MEDIAN_CPU_HAS_AVX2
#undef MEDIAN_CPU_AVX2_INLINE
#undef MEDIAN_CPU_AVX2_TARGET
#undef MEDIAN_CPU_AVX512_INLINE
#undef MEDIAN_CPU_AVX512_TARGET
#endif
#undef MEDIAN_CPU_HAS_AVX2
