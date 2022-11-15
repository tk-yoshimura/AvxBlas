#pragma once
#pragma unmanaged

#include "../utils.h"
#include "inline_loadstore_xn_s.hpp"

__forceinline __m256 _mm256_where_ps(__m256i cond, __m256 x, __m256 y) {
    const __m256 ret = _mm256_blendv_ps(y, x, _mm256_castsi256_ps(cond));

    return ret;
}

__forceinline __m256 _mm256_condmaskload_ps(infloats ptr, const __m256i mask, const __m256 overs) {
    return _mm256_where_ps(mask, _mm256_maskload_ps(ptr, mask), overs);
}

__forceinline void _mm256_condmaskload_x1_ps(infloats ptr, __m256& x0, const __m256i mask, const __m256 overs) {
    _mm256_maskload_x1_ps(ptr, x0, mask);
    x0 = _mm256_where_ps(mask, x0, overs);
}

__forceinline void _mm256_condmaskload_x2_ps(infloats ptr, __m256& x0, __m256& x1, const __m256i mask, const __m256 overs) {
    _mm256_maskload_x2_ps(ptr, x0, x1, mask);
    x1 = _mm256_where_ps(mask, x1, overs);
}

__forceinline void _mm256_condmaskload_x3_ps(infloats ptr, __m256& x0, __m256& x1, __m256& x2, const __m256i mask, const __m256 overs) {
    _mm256_maskload_x3_ps(ptr, x0, x1, x2, mask);
    x2 = _mm256_where_ps(mask, x2, overs);
}

__forceinline void _mm256_condmaskload_x4_ps(infloats ptr, __m256& x0, __m256& x1, __m256& x2, __m256& x3, const __m256i mask, const __m256 overs) {
    _mm256_maskload_x4_ps(ptr, x0, x1, x2, x3, mask);
    x3 = _mm256_where_ps(mask, x3, overs);
}

__forceinline void _mm256_condmaskload_x5_ps(infloats ptr, __m256& x0, __m256& x1, __m256& x2, __m256& x3, __m256& x4, const __m256i mask, const __m256 overs) {
    _mm256_maskload_x5_ps(ptr, x0, x1, x2, x3, x4, mask);
    x4 = _mm256_where_ps(mask, x4, overs);
}

__forceinline void _mm256_condmaskload_x6_ps(infloats ptr, __m256& x0, __m256& x1, __m256& x2, __m256& x3, __m256& x4, __m256& x5, const __m256i mask, const __m256 overs) {
    _mm256_maskload_x6_ps(ptr, x0, x1, x2, x3, x4, x5, mask);
    x5 = _mm256_where_ps(mask, x5, overs);
}

__forceinline void _mm256_condmaskload_x7_ps(infloats ptr, __m256& x0, __m256& x1, __m256& x2, __m256& x3, __m256& x4, __m256& x5, __m256& x6, const __m256i mask, const __m256 overs) {
    _mm256_maskload_x7_ps(ptr, x0, x1, x2, x3, x4, x5, x6, mask);
    x6 = _mm256_where_ps(mask, x6, overs);
}

__forceinline void _mm256_condmaskload_x8_ps(infloats ptr, __m256& x0, __m256& x1, __m256& x2, __m256& x3, __m256& x4, __m256& x5, __m256& x6, __m256& x7, const __m256i mask, const __m256 overs) {
    _mm256_maskload_x8_ps(ptr, x0, x1, x2, x3, x4, x5, x6, x7, mask);
    x7 = _mm256_where_ps(mask, x7, overs);
}