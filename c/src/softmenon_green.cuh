#pragma once

namespace softmenon_green {
// Green candidates use a 3/4-strength Hamilton-Adams correction.
// Both terms round separately, including every candidate used by the scores.
// Candidate reconstruction reaches two RAW samples along the selected axis.
__device__ __forceinline__ int candidate(const uint8_t* p, ptrdiff_t axis) {
    return ((int(p[-axis]) + p[axis] + 1) >> 1) +
        ((6 * (2 * int(p[0]) - p[-2 * axis] - p[2 * axis]) + 16) >> 5);
}

__device__ __forceinline__ int difference(const uint8_t* p, ptrdiff_t axis) {
    return int(p[0]) - candidate(p, axis);
}

__device__ __forceinline__ int colocated_score(const uint8_t* p, ptrdiff_t axis) {
    const int center = difference(p, axis);
    return abs(center - difference(p - 2 * axis, axis)) +
           abs(center - difference(p + 2 * axis, axis));
}

// Compare co-located color differences along parallel and diagonal supports.
// The complete score reaches four RAW samples along either image axis.
__device__ __forceinline__ int directional_score(const uint8_t* p, ptrdiff_t axis,
                                                ptrdiff_t perpendicular) {
    return 3 * colocated_score(p, axis) +
        colocated_score(p - 2 * perpendicular, axis) +
        colocated_score(p + 2 * perpendicular, axis) +
        abs(difference(p - axis - perpendicular, axis) -
            difference(p + axis - perpendicular, axis)) +
        abs(difference(p - axis + perpendicular, axis) -
            difference(p + axis + perpendicular, axis));
}

__device__ __forceinline__ int estimate(const uint8_t* p, ptrdiff_t pitch) {
    const int gh = candidate(p, 1), gv = candidate(p, pitch);
    // Scores are <=6132; squared weights and signed weighted sums need int64.
    long long wh = directional_score(p, pitch, 1) + 1;
    long long wv = directional_score(p, 1, pitch) + 1;
    wh *= wh;
    wv *= wv;
    const long long denominator = wh + wv;
    const long long rounded = wh * gh + wv * gv + denominator / 2;
    // Nearest integer, ties toward positive infinity, followed by byte clipping.
    if (rounded <= 0) return 0;
    const int quotient = int(rounded / denominator);
    return quotient > 255 ? 255 : quotient;
}
} // namespace softmenon_green
