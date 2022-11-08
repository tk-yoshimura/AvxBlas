#pragma once
#pragma unmanaged

#include <immintrin.h>

// e0,e1 -> e0+e1
__forceinline double _mm_sum2to1_pd(__m128d x) {
    __m128d y = _mm_hadd_pd(x, x);
    double ret = _mm_cvtsd_f64(y);

    return ret;
}

// e0,e1,e2,e3 -> e0+e1+e2+e3
__forceinline double _mm256_sum4to1_pd(__m256d x) {
    __m128d y = _mm_add_pd(_mm256_castpd256_pd128(x), _mm256_extractf128_pd(x, 1));
    __m128d z = _mm_hadd_pd(y, y);
    double ret = _mm_cvtsd_f64(z);

    return ret;
}

// e0,e1,e2,e3 -> e0+e2,e1+e3
__forceinline __m128d _mm256_sum4to2_pd(__m256d x) {
    __m128d ret = _mm_add_pd(_mm256_castpd256_pd128(x), _mm256_extractf128_pd(x, 1));

    return ret;
}

// e0,e1,e2,e3,e4,e5,e6,e7 -> e0+e1+e2+e3+e4+e5+e6+e7
__forceinline double _mm256_sum8to1_pd(__m256d x, __m256d y) {
    double ret = _mm256_sum4to1_pd(_mm256_add_pd(x, y));

    return ret;
}

// e0,...,e11 -> e0+...+e11
__forceinline double _mm256_sum12to1_pd(__m256d x, __m256d y, __m256d z) {
    double ret = _mm256_sum4to1_pd(_mm256_add_pd(_mm256_add_pd(x, y), z));

    return ret;
}

// e0,...,e15 -> e0+...+e15
__forceinline double _mm256_sum16to1_pd(__m256d x, __m256d y, __m256d z, __m256d w) {
    double ret = _mm256_sum4to1_pd(_mm256_add_pd(_mm256_add_pd(x, y), _mm256_add_pd(z, w)));

    return ret;
}

// e0,...,e11 -> e0+e3+...+e9,e1+e4+...+e10,e2+e5+...+e11,zero
__forceinline __m256d _mm256_sum12to3_pd(__m256d x0, __m256d x1, __m256d x2) {
    const __m256d _mask_1 = _mm256_castsi256_pd(_mm256_setr_epi32(~0u, ~0u, ~0u, ~0u, ~0u, ~0u, 0, 0));
    const __m256d _mask_2 = _mm256_castsi256_pd(_mm256_setr_epi32(0, 0, 0, 0, 0, 0, ~0u, ~0u));

    __m256d y0 = x0;
    __m256d y1 = _mm256_permute4x64_pd(x1, _MM_PERM_ABDC);
    __m256d y2 = _mm256_permute4x64_pd(x2, _MM_PERM_ADCB);

    __m256d z0 = _mm256_permute4x64_pd(_mm256_and_pd(y0, _mask_2), _MM_PERM_AAAD);
    __m256d z1 = _mm256_permute4x64_pd(_mm256_and_pd(y1, _mask_2), _MM_PERM_AADA);
    __m256d z2 = _mm256_permute4x64_pd(_mm256_and_pd(y2, _mask_2), _MM_PERM_ADAA);

    __m256d w0 = _mm256_and_pd(_mm256_add_pd(y0, _mm256_add_pd(y1, y2)), _mask_1);
    __m256d w1 = _mm256_add_pd(z0, _mm256_add_pd(z1, z2));

    __m256d ret = _mm256_add_pd(w0, w1);

    return ret;
}

// e0,...,e19 -> e0+e5+...+e15,e1+e6+...+e16,e2+e7+...+e17,e3+e8+...+e18,e4+e9+...+e19,_,_,_
__forceinline __m256dx2 _mm256_sum20to5_pd(__m256d x0, __m256d x1, __m256d x2, __m256d x3, __m256d x4) {
    __m256d y1 = _mm256_permute4x64_pd(x1, _MM_PERM_ADCB);
    __m256d y2 = _mm256_permute4x64_pd(x2, _MM_PERM_ABDC);
    __m256d y3 = _mm256_permute4x64_pd(x3, _MM_PERM_BACD);
    __m256d y4 = _mm256_permute4x64_pd(x4, _MM_PERM_CBAD);

    __m256d z0 = x0;
    __m256d z1 = _mm256_blend_pd(y1, y4, 0b1000);
    __m256d z2 = _mm256_blend_pd(y2, y4, 0b0100);
    __m256d z3 = _mm256_blend_pd(y3, y4, 0b0010);

    __m256d w0 = _mm256_blend_pd(y1, y2, 0b0100);
    __m256d w1 = _mm256_blend_pd(y3, y4, 0b0001);
    __m256d wc = _mm256_blend_pd(w0, w1, 0b0011);
    __m256d wa = _mm256_permute4x64_pd(_mm256_hadd_pd(wc, wc), _MM_PERM_DBCA);
    __m256d wb = _mm256_hadd_pd(wa, wa);

    __m256d imm0 = _mm256_add_pd(_mm256_add_pd(z0, z1), _mm256_add_pd(z2, z3));
    __m256d imm1 = wb;

    return __m256dx2(imm0, imm1);
}

// e0,...,e11 -> e0+e6,e1+e7,e2+e8,e3+e9,e4+e10,e5+e11,_,_
__forceinline __m256dx2 _mm256_sum12to6_pd(__m256d x0, __m256d x1, __m256d x2) {
    __m256d y1 = _mm256_permute4x64_pd(x1, _MM_PERM_BADC);
    __m256d y2 = _mm256_permute4x64_pd(x2, _MM_PERM_BADC);

    __m256d z0 = x0;
    __m256d z1 = _mm256_blend_pd(y1, y2, 0b1100);
    __m256d z2 = _mm256_blend_pd(y1, y2, 0b0011);

    __m256d imm0 = _mm256_add_pd(z0, z1);
    __m256d imm1 = _mm256_add_pd(x1, y2);

    return __m256dx2(imm0, imm1);
}

// e0,e1,e2,e3 -> e0+e1,e2+e3
__forceinline __m128d _mm256_hadd2_pd(__m256d x) {
    __m128d lo = _mm256_castpd256_pd128(x);
    __m128d hi = _mm256_extractf128_pd(x, 1);

    __m128d ret = _mm_hadd_pd(lo, hi);

    return ret;
}

__forceinline __m256d _mm256_sumwise2_pd(__m256d x) {
    __m256d y = _mm256_add_pd(x, _mm256_permute4x64_pd(x, _MM_PERM_CDAB));

    return y;
}

__forceinline __m256d _mm256_sumwise3_pd(__m256d x) {
    __m256d y = _mm256_add_pd(_mm256_add_pd(x, _mm256_permute4x64_pd(x, _MM_PERM_DBAC)), _mm256_permute4x64_pd(x, _MM_PERM_DACB));

    return y;
}

__forceinline __m256d _mm256_sumwise4_pd(__m256d x) {
    __m256d y = _mm256_add_pd(x, _mm256_permute4x64_pd(x, _MM_PERM_CDAB));
    __m256d z = _mm256_add_pd(y, _mm256_permute4x64_pd(y, _MM_PERM_BADC));

    return z;
}