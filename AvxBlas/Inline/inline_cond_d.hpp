#pragma once
#pragma unmanaged

#include "../utils.h"
#include "inline_loadstore_xn_d.hpp"

__forceinline __m256d _mm256_where_pd(__m256i cond, __m256d x, __m256d y) {
    const __m256d ret = _mm256_blendv_pd(y, x, _mm256_castsi256_pd(cond));

    return ret;
}

__forceinline __m256d _mm256_condmaskload_pd(indoubles ptr, const __m256i mask, const __m256d overs) {
    return _mm256_where_pd(mask, _mm256_maskload_pd(ptr, mask), overs);
}

__forceinline void _mm256_condmaskload_x1_pd(indoubles ptr, __m256d& x0, const __m256i mask, const __m256d overs) {
    _mm256_maskload_x1_pd(ptr, x0, mask);
    x0 = _mm256_where_pd(mask, x0, overs);
}

__forceinline void _mm256_condmaskload_x2_pd(indoubles ptr, __m256d& x0, __m256d& x1, const __m256i mask, const __m256d overs) {
    _mm256_maskload_x2_pd(ptr, x0, x1, mask);
    x1 = _mm256_where_pd(mask, x1, overs);
}

__forceinline void _mm256_condmaskload_x3_pd(indoubles ptr, __m256d& x0, __m256d& x1, __m256d& x2, const __m256i mask, const __m256d overs) {
    _mm256_maskload_x3_pd(ptr, x0, x1, x2, mask);
    x2 = _mm256_where_pd(mask, x2, overs);
}

__forceinline void _mm256_condmaskload_x4_pd(indoubles ptr, __m256d& x0, __m256d& x1, __m256d& x2, __m256d& x3, const __m256i mask, const __m256d overs) {
    _mm256_maskload_x4_pd(ptr, x0, x1, x2, x3, mask);
    x3 = _mm256_where_pd(mask, x3, overs);
}

__forceinline void _mm256_condmaskload_x5_pd(indoubles ptr, __m256d& x0, __m256d& x1, __m256d& x2, __m256d& x3, __m256d& x4, const __m256i mask, const __m256d overs) {
    _mm256_maskload_x5_pd(ptr, x0, x1, x2, x3, x4, mask);
    x4 = _mm256_where_pd(mask, x4, overs);
}

__forceinline void _mm256_condmaskload_x6_pd(indoubles ptr, __m256d& x0, __m256d& x1, __m256d& x2, __m256d& x3, __m256d& x4, __m256d& x5, const __m256i mask, const __m256d overs) {
    _mm256_maskload_x6_pd(ptr, x0, x1, x2, x3, x4, x5, mask);
    x5 = _mm256_where_pd(mask, x5, overs);
}

__forceinline void _mm256_condmaskload_x7_pd(indoubles ptr, __m256d& x0, __m256d& x1, __m256d& x2, __m256d& x3, __m256d& x4, __m256d& x5, __m256d& x6, const __m256i mask, const __m256d overs) {
    _mm256_maskload_x7_pd(ptr, x0, x1, x2, x3, x4, x5, x6, mask);
    x6 = _mm256_where_pd(mask, x6, overs);
}

__forceinline void _mm256_condmaskload_x8_pd(indoubles ptr, __m256d& x0, __m256d& x1, __m256d& x2, __m256d& x3, __m256d& x4, __m256d& x5, __m256d& x6, __m256d& x7, const __m256i mask, const __m256d overs) {
    _mm256_maskload_x8_pd(ptr, x0, x1, x2, x3, x4, x5, x6, x7, mask);
    x7 = _mm256_where_pd(mask, x7, overs);
}