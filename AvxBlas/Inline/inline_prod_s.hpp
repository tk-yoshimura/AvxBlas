#pragma once
#pragma unmanaged

#include <immintrin.h>
#include "inline_cond_s.hpp"

// e0,e1,e2,e3 -> e0*e1*e2*e3
__forceinline float _mm_prod4to1_ps(__m128 x) {
    __m128 y = _mm_mul_ps(x, _mm_movehl_ps(x, x));
    float ret = _mm_cvtss_f32(_mm_mul_ss(y, _mm_shuffle_ps(y, y, 1)));

    return ret;
}

// e0,e1,e2,e3,e4,e5,e6,e7 -> e0*e1*e2*e3*e4*e5*e6*e7
__forceinline float _mm256_prod8to1_ps(__m256 x) {
    __m128 y = _mm_mul_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    __m128 z = _mm_mul_ps(y, _mm_movehl_ps(y, y));
    float ret = _mm_cvtss_f32(_mm_mul_ss(z, _mm_shuffle_ps(z, z, 1)));

    return ret;
}

// e0,e1,e2,e3,e4,e5,e6,e7 -> e0*e2*e4*e6,e1*e3*e5*e7
__forceinline __m128 _mm256_prod8to2_ps(__m256 x) {
    __m128 y = _mm_mul_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    __m128 ret = _mm_mul_ps(y, _mm_movehl_ps(y, y));

    return ret;
}

// e0,e1,e2,e3,e4,e5,-,- -> e0*e3,e1*e4,e2*e5,-
__forceinline __m128 _mm256_prod6to3_ps(__m256 x) {
    const __m256i _perm43 = _mm256_setr_epi32(0, 1, 2, 6, 3, 4, 5, 7);

    __m256 y = _mm256_permutevar8x32_ps(x, _perm43);
    __m128 ret = _mm_mul_ps(_mm256_castps256_ps128(y), _mm256_extractf128_ps(y, 1));

    return ret;
}

// e0,e1,e2,e3,e4,e5,e6,e7 -> e0*e4,e1*e5,e2*e6,e3*e7
__forceinline __m128 _mm256_prod8to4_ps(__m256 x) {
    __m128 ret = _mm_mul_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));

    return ret;
}

// e0,...,e15 -> e0*...*e15
__forceinline float _mm256_prod16to1_ps(__m256 x, __m256 y) {
    float ret = _mm256_prod8to1_ps(_mm256_mul_ps(x, y));

    return ret;
}

// e0,...,e23 -> e0*...*e23
__forceinline float _mm256_prod24to1_ps(__m256 x, __m256 y, __m256 z) {
    float ret = _mm256_prod8to1_ps(_mm256_mul_ps(_mm256_mul_ps(x, y), z));

    return ret;
}

// e0,...,e31 -> e0*...*e31
__forceinline float _mm256_prod32to1_ps(__m256 x, __m256 y, __m256 z, __m256 w) {
    float ret = _mm256_prod8to1_ps(_mm256_mul_ps(_mm256_mul_ps(x, y), _mm256_mul_ps(z, w)));

    return ret;
}

// e0,...,e23 -> e0*e3*...*e21,e1*e4*...*e22,e2*e5*...*e23,zero
__forceinline __m128 _mm256_prod24to3_ps(__m256 x0, __m256 x1, __m256 x2) {
    const __m256 _ones = _mm256_set1_ps(1);

    const __m256i _perm_y0 = _mm256_setr_epi32(0, 1, 2, 6, 3, 4, 5, 7);
    const __m256i _perm_y1 = _mm256_setr_epi32(1, 2, 3, 0, 4, 5, 6, 7);
    const __m256i _perm_y2 = _mm256_setr_epi32(2, 3, 4, 1, 5, 6, 7, 0);

    const __m256i _perm_z0 = _mm256_setr_epi32(3, 7, 0, 1, 2, 4, 5, 6);
    const __m256i _perm_z1 = _mm256_setr_epi32(7, 0, 3, 1, 2, 4, 5, 6);
    const __m256i _perm_z2 = _mm256_setr_epi32(0, 7, 3, 1, 2, 4, 5, 6);

    const __m256i _mask_1 = _mm256_setr_epi32(~0u, ~0u, ~0u, 0, ~0u, ~0u, ~0u, 0);
    const __m256i _mask_2 = _mm256_setr_epi32(0, 0, 0, ~0u, 0, 0, 0, ~0u);

    __m256 y0 = _mm256_permutevar8x32_ps(x0, _perm_y0);
    __m256 y1 = _mm256_permutevar8x32_ps(x1, _perm_y1);
    __m256 y2 = _mm256_permutevar8x32_ps(x2, _perm_y2);

    __m256 z0 = _mm256_permutevar8x32_ps(_mm256_where_ps(_mask_2, y0, _ones), _perm_z0);
    __m256 z1 = _mm256_permutevar8x32_ps(_mm256_where_ps(_mask_2, y1, _ones), _perm_z1);
    __m256 z2 = _mm256_permutevar8x32_ps(_mm256_where_ps(_mask_2, y2, _ones), _perm_z2);

    __m256 w0 = _mm256_where_ps(_mask_1, _mm256_mul_ps(y0, _mm256_mul_ps(y1, y2)), _ones);
    __m256 w1 = _mm256_mul_ps(z0, _mm256_mul_ps(z1, z2));

    __m256 s = _mm256_mul_ps(w0, w1);

    __m128 ret = _mm_mul_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));

    return ret;
}

// e0,...,e39 -> e0*e5*...*e35,e1*e6*...*e36,e2*e7*...*e37,e3*e8*...*e38,e4*e9*...*e39,zero,zero,zero
__forceinline __m256 _mm256_prod40to5_ps(__m256 x0, __m256 x1, __m256 x2, __m256 x3, __m256 x4) {
    const __m256 _ones = _mm256_set1_ps(1);

    const __m256i _perm_y1 = _mm256_setr_epi32(2, 3, 4, 5, 6, 7, 0, 1);
    const __m256i _perm_y2 = _mm256_setr_epi32(4, 5, 6, 7, 3, 0, 1, 2);
    const __m256i _perm_y3 = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);
    const __m256i _perm_y4 = _mm256_setr_epi32(3, 4, 5, 6, 7, 0, 1, 2);

    const __m256i _perm_z0 = _mm256_setr_epi32(5, 6, 7, 0, 1, 2, 3, 4);
    const __m256i _perm_z1 = _mm256_setr_epi32(5, 0, 1, 6, 7, 2, 3, 4);
    const __m256i _perm_z2 = _mm256_setr_epi32(0, 5, 6, 7, 1, 2, 3, 4);
    const __m256i _perm_z3 = _mm256_setr_epi32(5, 6, 0, 1, 7, 2, 3, 4);
    const __m256i _perm_z4 = _mm256_setr_epi32(0, 1, 5, 6, 7, 2, 3, 4);

    const __m256i _mask_1 = _mm256_setr_epi32(~0u, ~0u, ~0u, ~0u, ~0u, 0, 0, 0);
    const __m256i _mask_2 = _mm256_setr_epi32(0, 0, 0, 0, 0, ~0u, ~0u, ~0u);

    __m256 y0 = x0;
    __m256 y1 = _mm256_permutevar8x32_ps(x1, _perm_y1);
    __m256 y2 = _mm256_permutevar8x32_ps(x2, _perm_y2);
    __m256 y3 = _mm256_permutevar8x32_ps(x3, _perm_y3);
    __m256 y4 = _mm256_permutevar8x32_ps(x4, _perm_y4);

    __m256 z0 = _mm256_permutevar8x32_ps(_mm256_where_ps(_mask_2, y0, _ones), _perm_z0);
    __m256 z1 = _mm256_permutevar8x32_ps(_mm256_where_ps(_mask_2, y1, _ones), _perm_z1);
    __m256 z2 = _mm256_permutevar8x32_ps(_mm256_where_ps(_mask_2, y2, _ones), _perm_z2);
    __m256 z3 = _mm256_permutevar8x32_ps(_mm256_where_ps(_mask_2, y3, _ones), _perm_z3);
    __m256 z4 = _mm256_permutevar8x32_ps(_mm256_where_ps(_mask_2, y4, _ones), _perm_z4);

    __m256 w0 = _mm256_where_ps(_mask_1, _mm256_mul_ps(y0, _mm256_mul_ps(_mm256_mul_ps(y1, y2), _mm256_mul_ps(y3, y4))), _ones);
    __m256 w1 = _mm256_mul_ps(z0, _mm256_mul_ps(_mm256_mul_ps(z1, z2), _mm256_mul_ps(z3, z4)));

    __m256 s = _mm256_mul_ps(w0, w1);

    return s;
}

// e0,...,e23 -> e0*e6*...*e18,e1*e7*...*e19,e2*e8*...*e20,e3*e9*...*e21,e4*e10*...*e22,e5*e11*...*e23,zero,zero
__forceinline __m256 _mm256_prod24to6_ps(__m256 x0, __m256 x1, __m256 x2) {
    const __m256 _ones = _mm256_set1_ps(1);

    const __m256i _perm_y1 = _mm256_setr_epi32(4, 5, 6, 7, 2, 3, 0, 1);
    const __m256i _perm_y2 = _mm256_setr_epi32(2, 3, 4, 5, 6, 7, 0, 1);

    const __m256i _perm_z0 = _mm256_setr_epi32(6, 7, 0, 1, 2, 3, 4, 5);
    const __m256i _perm_z1 = _mm256_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3);
    const __m256i _perm_z2 = _mm256_setr_epi32(2, 3, 4, 5, 6, 7, 0, 1);

    const __m256i _mask_1 = _mm256_setr_epi32(~0u, ~0u, ~0u, ~0u, ~0u, ~0u, 0, 0);
    const __m256i _mask_2 = _mm256_setr_epi32(0, 0, 0, 0, 0, 0, ~0u, ~0u);

    __m256 y0 = x0;
    __m256 y1 = _mm256_permutevar8x32_ps(x1, _perm_y1);
    __m256 y2 = _mm256_permutevar8x32_ps(x2, _perm_y2);

    __m256 z0 = _mm256_permutevar8x32_ps(_mm256_where_ps(_mask_2, y0, _ones), _perm_z0);
    __m256 z1 = _mm256_permutevar8x32_ps(_mm256_where_ps(_mask_2, y1, _ones), _perm_z1);
    __m256 z2 = _mm256_permutevar8x32_ps(_mm256_where_ps(_mask_2, y2, _ones), _perm_z2);

    __m256 w0 = _mm256_where_ps(_mask_1, _mm256_mul_ps(y0, _mm256_mul_ps(y1, y2)), _ones);
    __m256 w1 = _mm256_mul_ps(z0, _mm256_mul_ps(z1, z2));

    __m256 s = _mm256_mul_ps(w0, w1);

    return s;
}