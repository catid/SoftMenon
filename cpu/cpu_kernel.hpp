#pragma once
#include <cstdint>
#include <cstddef>

// RAW points at logical(0,0) with a reflected halo. Pitches are bytes,
// row intervals are[begin_y,end_y), and output is BGR. Odd dimensions and
// either CFA phase are supported without touching row padding.
void bilinear_cpu_rows(const uint8_t* raw,size_t raw_pitch,uint8_t* bgr,
    size_t bgr_pitch,int width,int height,bool rggb,int begin_y,int end_y);
void malvar2004_cpu_rows(const uint8_t* raw,size_t raw_pitch,uint8_t* bgr,
    size_t bgr_pitch,int width,int height,bool rggb,int begin_y,int end_y);

// SoftMenon green and measured colors on an even BGGR-aligned rectangle.
// The caller provides a four-pixel RAW halo and independent byte pitches.
void bggr_softmenon_g_cpu(const uint8_t* raw,int raw_pitch,uint8_t* bgr,
    int bgr_pitch,int width,int height);

// Complete missing colors from immutable green/measured-color input.
// Source includes an initialized one-pixel halo; destination is separate.
void bggr_softmenon_rb_rows(const uint8_t* source,size_t source_pitch,
    uint8_t* destination,size_t destination_pitch,int width,int height,
    int begin_y,int end_y);
