#pragma once
#pragma unmanaged

#include <immintrin.h>

// e0,e1,e2,e3 -> e0+e1+e2+e3
__forceinline float _mm_sum4to1_ps(const __m128 x) {
    const __m128 y = _mm_add_ps(x, _mm_movehl_ps(x, x));
    const float ret = _mm_cvtss_f32(_mm_add_ss(y, _mm_shuffle_ps(y, y, 1)));

    return ret;
}

// e0,e1,e2,e3,e4,e5,e6,e7 -> e0+e1+e2+e3+e4+e5+e6+e7
__forceinline float _mm256_sum8to1_ps(const __m256 x) {
    const __m128 y = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    const __m128 z = _mm_add_ps(y, _mm_movehl_ps(y, y));
    const float ret = _mm_cvtss_f32(_mm_add_ss(z, _mm_shuffle_ps(z, z, 1)));

    return ret;
}

// e0,e1,e2,e3,e4,e5,e6,e7 -> e0+e2+e4+e6,e1+e3+e5+e7
__forceinline __m128 _mm256_sum8to2_ps(const __m256 x) {
    const __m128 y = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    const __m128 ret = _mm_add_ps(y, _mm_movehl_ps(y, y));

    return ret;
}

// e0,e1,e2,e3,e4,e5,-,- -> e0+e3,e1+e4,e2+e5,-,-
__forceinline __m128 _mm256_sum6to3_ps(const __m256 x) {
    const __m256i _perm43 = _mm256_setr_epi32(0, 1, 2, 6, 3, 4, 5, 7);

    const __m256 y = _mm256_permutevar8x32_ps(x, _perm43);
    const __m128 ret = _mm_add_ps(_mm256_castps256_ps128(y), _mm256_extractf128_ps(y, 1));

    return ret;
}

// e0,e1,e2,e3,e4,e5,e6,e7 -> e0+e4,e1+e5,e2+e6,e3+e7
__forceinline __m128 _mm256_sum8to4_ps(const __m256 x) {
    const __m128 ret = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));

    return ret;
}

// e0,...,e15 -> e0+...+e15
__forceinline float _mm256_sum16to1_ps(const __m256 x, const __m256 y) {
    float ret = _mm256_sum8to1_ps(_mm256_add_ps(x, y));

    return ret;
}

// e0,...,e23 -> e0+...+e23
__forceinline float _mm256_sum24to1_ps(const __m256 x, const __m256 y, const __m256 z) {
    float ret = _mm256_sum8to1_ps(_mm256_add_ps(_mm256_add_ps(x, y), z));

    return ret;
}

// e0,...,e31 -> e0+...+e31
__forceinline float _mm256_sum32to1_ps(const __m256 x, const __m256 y, const __m256 z, const __m256 w) {
    float ret = _mm256_sum8to1_ps(_mm256_add_ps(_mm256_add_ps(x, y), _mm256_add_ps(z, w)));

    return ret;
}

// e0,...,e23 -> e0+e3+...+e21,e1+e4+...+e22,e2+e5+...+e23,zero
__forceinline __m128 _mm256_sum24to3_ps(const __m256 x0, const __m256 x1, const __m256 x2) {
    const __m256i _perm_y0 = _mm256_setr_epi32(0, 1, 2, 6, 3, 4, 5, 7);
    const __m256i _perm_y1 = _mm256_setr_epi32(1, 2, 3, 0, 4, 5, 6, 7);
    const __m256i _perm_y2 = _mm256_setr_epi32(2, 3, 4, 1, 5, 6, 7, 0);

    const __m256i _perm_z0 = _mm256_setr_epi32(3, 7, 0, 1, 2, 4, 5, 6);
    const __m256i _perm_z1 = _mm256_setr_epi32(7, 0, 3, 1, 2, 4, 5, 6);
    const __m256i _perm_z2 = _mm256_setr_epi32(0, 7, 3, 1, 2, 4, 5, 6);

    const __m256 _mask_1 = _mm256_castsi256_ps(_mm256_setr_epi32(~0u, ~0u, ~0u, 0, ~0u, ~0u, ~0u, 0));
    const __m256 _mask_2 = _mm256_castsi256_ps(_mm256_setr_epi32(0, 0, 0, ~0u, 0, 0, 0, ~0u));

    const __m256 y0 = _mm256_permutevar8x32_ps(x0, _perm_y0);
    const __m256 y1 = _mm256_permutevar8x32_ps(x1, _perm_y1);
    const __m256 y2 = _mm256_permutevar8x32_ps(x2, _perm_y2);

    const __m256 z0 = _mm256_permutevar8x32_ps(_mm256_and_ps(y0, _mask_2), _perm_z0);
    const __m256 z1 = _mm256_permutevar8x32_ps(_mm256_and_ps(y1, _mask_2), _perm_z1);
    const __m256 z2 = _mm256_permutevar8x32_ps(_mm256_and_ps(y2, _mask_2), _perm_z2);

    const __m256 w0 = _mm256_and_ps(_mm256_add_ps(y0, _mm256_add_ps(y1, y2)), _mask_1);
    const __m256 w1 = _mm256_add_ps(z0, _mm256_add_ps(z1, z2));

    const __m256 s = _mm256_add_ps(w0, w1);

    const __m128 ret = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));

    return ret;
}

// e0,e1,e2,e3,e4,e5,e6,e7 -> e0+e1,e2+e3,e4+e5,e6+e7
__forceinline __m128 _mm256_hadd2_ps(const __m256 x) {
    const __m128 lo = _mm256_castps256_ps128(x);
    const __m128 hi = _mm256_extractf128_ps(x, 1);

    const __m128 ret = _mm_hadd_ps(lo, hi);

    return ret;
}