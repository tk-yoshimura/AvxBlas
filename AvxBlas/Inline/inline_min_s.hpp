#pragma once
#pragma unmanaged

#include <immintrin.h>

// e0,e1,e2,e3 -> min(e0,e1,e2,e3)
__forceinline float _mm_min4to1_ps(__m128 x) {
    __m128 y = _mm_min_ps(x, _mm_movehl_ps(x, x));
    float ret = _mm_cvtss_f32(_mm_min_ss(y, _mm_shuffle_ps(y, y, 1)));

    return ret;
}

// e0,e1,e2,e3,e4,e5,e6,e7 -> min(e0,e1,e2,e3,e4,e5,e6,e7)
__forceinline float _mm256_min8to1_ps(__m256 x) {
    __m128 y = _mm_min_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    __m128 z = _mm_min_ps(y, _mm_movehl_ps(y, y));
    float ret = _mm_cvtss_f32(_mm_min_ss(z, _mm_shuffle_ps(z, z, 1)));

    return ret;
}

// e0,e1,e2,e3,e4,e5,e6,e7 -> min(e0,e2),min(e4,e6),min(e1,e3),min(e5,e7)
__forceinline __m128 _mm256_min8to2_ps(__m256 x) {
    __m128 y = _mm_min_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    __m128 ret = _mm_min_ps(y, _mm_movehl_ps(y, y));

    return ret;
}

// e0,e1,e2,e3,e4,e5,-,- -> min(e0,e3),min(e1,e4),min(e2,e5),_
__forceinline __m128 _mm256_min6to3_ps(__m256 x) {
    const __m256i _perm43 = _mm256_setr_epi32(0, 1, 2, 6, 3, 4, 5, 7);

    __m256 y = _mm256_permutevar8x32_ps(x, _perm43);
    __m128 ret = _mm_min_ps(_mm256_castps256_ps128(y), _mm256_extractf128_ps(y, 1));

    return ret;
}

// e0,e1,e2,e3,e4,e5,e6,e7 -> min(e0,e4),min(e1,e5),min(e2,e6),min(e3,e7)
__forceinline __m128 _mm256_min8to4_ps(__m256 x) {
    __m128 ret = _mm_min_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));

    return ret;
}

// e0,...,e15 -> min(e0,...,e15)
__forceinline float _mm256_min16to1_ps(__m256 x, __m256 y) {
    float ret = _mm256_min8to1_ps(_mm256_min_ps(x, y));

    return ret;
}

// e0,...,e23 -> min(e0,...,e23)
__forceinline float _mm256_min24to1_ps(__m256 x, __m256 y, __m256 z) {
    float ret = _mm256_min8to1_ps(_mm256_min_ps(_mm256_min_ps(x, y), z));

    return ret;
}

// e0,...,e31 -> min(e0,...,e31)
__forceinline float _mm256_min32to1_ps(__m256 x, __m256 y, __m256 z, __m256 w) {
    float ret = _mm256_min8to1_ps(_mm256_min_ps(_mm256_min_ps(x, y), _mm256_min_ps(z, w)));

    return ret;
}

// e0,...,e23 -> min(e0,e3,...,e21),min(e1,e4,...,e22),min(e2,e5,...,e23),_
__forceinline __m128 _mm256_min24to3_ps(__m256 x0, __m256 x1, __m256 x2) {
    const __m256i _perm_y0 = _mm256_setr_epi32(0, 1, 2, 6, 3, 4, 5, 7);
    const __m256i _perm_y1 = _mm256_setr_epi32(1, 2, 3, 0, 4, 5, 6, 7);
    const __m256i _perm_y2 = _mm256_setr_epi32(2, 3, 4, 1, 5, 6, 7, 0);

    const __m256i _perm_z0 = _mm256_setr_epi32(6, 7, 2, 3, 0, 1, 5, 4);
    const __m256i _perm_z1 = _mm256_setr_epi32(7, 2, 0, 4, 1, 5, 3, 6);
    const __m256i _perm_z2 = _mm256_setr_epi32(2, 0, 1, 6, 5, 3, 4, 7);

    __m256 y0 = _mm256_permutevar8x32_ps(x0, _perm_y0);
    __m256 y1 = _mm256_permutevar8x32_ps(x1, _perm_y1);
    __m256 y2 = _mm256_permutevar8x32_ps(x2, _perm_y2);

    __m256 z0 = _mm256_permutevar8x32_ps(x0, _perm_z0);
    __m256 z1 = _mm256_permutevar8x32_ps(x1, _perm_z1);
    __m256 z2 = _mm256_permutevar8x32_ps(x2, _perm_z2);

    __m256 w0 = _mm256_min_ps(y0, _mm256_min_ps(y1, y2));
    __m256 w1 = _mm256_min_ps(z0, _mm256_min_ps(z1, z2));

    __m256 s = _mm256_min_ps(w0, w1);

    __m128 ret = _mm_min_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));

    return ret;
}

// e0,...,e39 -> min(e0,e5,...,e35),min(e1,e6,...,e36),min(e2,e7,...,e37),min(e3,e8,...,e38),min(e4,e9,...,e39),_,_,_
__forceinline __m256 _mm256_min40to5_ps(__m256 x0, __m256 x1, __m256 x2, __m256 x3, __m256 x4) {
    const __m256i _perm_y1 = _mm256_setr_epi32(2, 3, 4, 5, 6, 7, 0, 1);
    const __m256i _perm_y2 = _mm256_setr_epi32(4, 5, 6, 7, 3, 0, 1, 2);
    const __m256i _perm_y3 = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0);
    const __m256i _perm_y4 = _mm256_setr_epi32(3, 4, 5, 6, 7, 0, 1, 2);

    const __m256i _perm_z0 = _mm256_setr_epi32(5, 6, 7, 3, 4, 0, 1, 2);
    const __m256i _perm_z1 = _mm256_setr_epi32(7, 3, 4, 0, 1, 2, 5, 6);
    const __m256i _perm_z2 = _mm256_setr_epi32(4, 0, 1, 2, 3, 5, 6, 7);
    const __m256i _perm_z3 = _mm256_setr_epi32(6, 7, 3, 4, 0, 1, 2, 5);
    const __m256i _perm_z4 = _mm256_setr_epi32(3, 4, 0, 1, 2, 5, 6, 7);

    __m256 y0 = x0;
    __m256 y1 = _mm256_permutevar8x32_ps(x1, _perm_y1);
    __m256 y2 = _mm256_permutevar8x32_ps(x2, _perm_y2);
    __m256 y3 = _mm256_permutevar8x32_ps(x3, _perm_y3);
    __m256 y4 = _mm256_permutevar8x32_ps(x4, _perm_y4);

    __m256 z0 = _mm256_permutevar8x32_ps(x0, _perm_z0);
    __m256 z1 = _mm256_permutevar8x32_ps(x1, _perm_z1);
    __m256 z2 = _mm256_permutevar8x32_ps(x2, _perm_z2);
    __m256 z3 = _mm256_permutevar8x32_ps(x3, _perm_z3);
    __m256 z4 = _mm256_permutevar8x32_ps(x4, _perm_z4);

    __m256 w0 = _mm256_min_ps(y0, _mm256_min_ps(_mm256_min_ps(y1, y2), _mm256_min_ps(y3, y4)));
    __m256 w1 = _mm256_min_ps(z0, _mm256_min_ps(_mm256_min_ps(z1, z2), _mm256_min_ps(z3, z4)));

    __m256 s = _mm256_min_ps(w0, w1);

    return s;
}

// e0,...,e23 -> min(e0,e6,...,e18),min(e1,e7,...,e19),min(e2,e8,...,e20),min(e3,e9,...,e21),min(e4,e10,...,e22),min(e5,e11,...,e23),_,_
__forceinline __m256 _mm256_min24to6_ps(__m256 x0, __m256 x1, __m256 x2) {
    const __m256i _perm_y1 = _mm256_setr_epi32(4, 5, 6, 7, 2, 3, 0, 1);
    const __m256i _perm_y2 = _mm256_setr_epi32(2, 3, 4, 5, 6, 7, 0, 1);

    const __m256i _perm_z0 = _mm256_setr_epi32(6, 7, 2, 3, 4, 5, 0, 1);
    const __m256i _perm_z1 = _mm256_setr_epi32(4, 5, 0, 1, 2, 3, 6, 7);
    const __m256i _perm_z2 = _mm256_setr_epi32(2, 3, 4, 5, 0, 1, 6, 7);

    __m256 y0 = x0;
    __m256 y1 = _mm256_permutevar8x32_ps(x1, _perm_y1);
    __m256 y2 = _mm256_permutevar8x32_ps(x2, _perm_y2);

    __m256 z0 = _mm256_permutevar8x32_ps(x0, _perm_z0);
    __m256 z1 = _mm256_permutevar8x32_ps(x1, _perm_z1);
    __m256 z2 = _mm256_permutevar8x32_ps(x2, _perm_z2);

    __m256 w0 = _mm256_min_ps(y0, _mm256_min_ps(y1, y2));
    __m256 w1 = _mm256_min_ps(z0, _mm256_min_ps(z1, z2));

    __m256 s = _mm256_min_ps(w0, w1);

    return s;
}

__forceinline __m256 _mm256_minwise2_ps(__m256 x) {
    __m256 y = _mm256_min_ps(x, _mm256_permute_ps(x, _MM_PERM_CDAB));

    return y;
}

__forceinline __m256 _mm256_minwise3_ps(__m256 x) {
    const __m256i _perm0 = _mm256_setr_epi32(2, 0, 1, 5, 3, 4, 6, 7);
    const __m256i _perm1 = _mm256_setr_epi32(1, 2, 0, 4, 5, 3, 6, 7);

    __m256 y = _mm256_min_ps(_mm256_min_ps(x, _mm256_permutevar8x32_ps(x, _perm0)), _mm256_permutevar8x32_ps(x, _perm1));

    return y;
}

__forceinline __m256 _mm256_minwise4_ps(__m256 x) {
    __m256 y = _mm256_min_ps(x, _mm256_permute_ps(x, _MM_PERM_CDAB));
    __m256 z = _mm256_min_ps(y, _mm256_permute_ps(y, _MM_PERM_BADC));

    return z;
}

__forceinline __m256 _mm256_minwise8_ps(__m256 x) {
    __m256 y = _mm256_min_ps(x, _mm256_permute_ps(x, _MM_PERM_CDAB));
    __m256 z = _mm256_min_ps(y, _mm256_permute_ps(y, _MM_PERM_BADC));
    __m256 w = _mm256_min_ps(z, _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(z), _MM_PERM_BADC)));

    return w;
}