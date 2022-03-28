#pragma once
#pragma unmanaged

#include <immintrin.h>

// e0,e1 -> e0+e1
__forceinline double _mm_sum2to1_pd(const __m128d x) {
    const __m128d y = _mm_hadd_pd(x, x);
    const double ret = _mm_cvtsd_f64(y);

    return ret;
}

// e0,e1,e2,e3 -> e0+e1+e2+e3
__forceinline double _mm256_sum4to1_pd(const __m256d x) {
    const __m128d y = _mm_add_pd(_mm256_castpd256_pd128(x), _mm256_extractf128_pd(x, 1));
    const __m128d z = _mm_hadd_pd(y, y);
    const double ret = _mm_cvtsd_f64(z);

    return ret;
}

// e0,e1,e2,e3,e4,e5,e6,e7 -> e0+e2+e4+e6,e1+e3+e5+e7
__forceinline __m128d _mm256_sum4to2_pd(const __m256d x) {
    const __m128d ret = _mm_add_pd(_mm256_castpd256_pd128(x), _mm256_extractf128_pd(x, 1));

    return ret;
}

// e0,e1,e2,e3,e4,e5,e6,e7 -> e0+e2+e4+e6,e1+e3+e5+e7
__forceinline double _mm256_sum8to1_pd(const __m256d x, const __m256d y) {
    double ret = _mm256_sum4to1_pd(_mm256_add_pd(x, y));

    return ret;
}

// e0,...,e11 -> e0+...+e11
__forceinline double _mm256_sum12to1_pd(const __m256d x, const __m256d y, const __m256d z) {
    double ret = _mm256_sum4to1_pd(_mm256_add_pd(_mm256_add_pd(x, y), z));

    return ret;
}

// e0,...,e15 -> e0+...+e15
__forceinline double _mm256_sum16to1_pd(const __m256d x, const __m256d y, const __m256d z, const __m256d w) {
    double ret = _mm256_sum4to1_pd(_mm256_add_pd(_mm256_add_pd(x, y), _mm256_add_pd(z, w)));

    return ret;
}

// e0,...,e11 -> e0+e3+...+e9,e1+e4+...+e10,e2+e5+...+e11,zero
__forceinline __m256d _mm256_sum12to3_pd(const __m256d x0, const __m256d x1, const __m256d x2) {
    const __m256d _mask_1 = _mm256_castsi256_pd(_mm256_setr_epi32(~0u, ~0u, ~0u, ~0u, ~0u, ~0u, 0, 0));
    const __m256d _mask_2 = _mm256_castsi256_pd(_mm256_setr_epi32(0, 0, 0, 0, 0, 0, ~0u, ~0u));

    const __m256d y0 = x0;
    const __m256d y1 = _mm256_permute4x64_pd(x1, _MM_PERM_ABDC);
    const __m256d y2 = _mm256_permute4x64_pd(x2, _MM_PERM_ADCB);

    const __m256d z0 = _mm256_permute4x64_pd(_mm256_and_pd(y0, _mask_2), _MM_PERM_AAAD);
    const __m256d z1 = _mm256_permute4x64_pd(_mm256_and_pd(y1, _mask_2), _MM_PERM_AADA);
    const __m256d z2 = _mm256_permute4x64_pd(_mm256_and_pd(y2, _mask_2), _MM_PERM_ADAA);

    const __m256d w0 = _mm256_and_pd(_mm256_add_pd(y0, _mm256_add_pd(y1, y2)), _mask_1);
    const __m256d w1 = _mm256_add_pd(z0, _mm256_add_pd(z1, z2));

    const __m256d ret = _mm256_add_pd(w0, w1);

    return ret;
}

// e0,...,e19 -> e0+e5+...+e15,e1+e6+...+e16,e2+e7+...+e17,e3+e8+...+e18,e4+e9+...+e19,_,_,_
__forceinline __m256dx2 _mm256_sum20to5_pd(const __m256d x0, const __m256d x1, const __m256d x2, const __m256d x3, const __m256d x4) {
    const __m256d y1 = _mm256_permute4x64_pd(x1, _MM_PERM_ADCB);
    const __m256d y2 = _mm256_permute4x64_pd(x2, _MM_PERM_ABDC);
    const __m256d y3 = _mm256_permute4x64_pd(x3, _MM_PERM_BACD);
    const __m256d y4 = _mm256_permute4x64_pd(x4, _MM_PERM_CBAD);

    const __m256d z0 = x0;
    const __m256d z1 = _mm256_blend_pd(y1, y4, 0b1000);
    const __m256d z2 = _mm256_blend_pd(y2, y4, 0b0100);
    const __m256d z3 = _mm256_blend_pd(y3, y4, 0b0010);

    const __m256d w0 = _mm256_blend_pd(y1, y2, 0b0100);
    const __m256d w1 = _mm256_blend_pd(y3, y4, 0b0001);
    const __m256d wc = _mm256_blend_pd(w0, w1, 0b0011);
    const __m256d wa = _mm256_permute4x64_pd(_mm256_hadd_pd(wc, wc), _MM_PERM_DBCA);
    const __m256d wb = _mm256_hadd_pd(wa, wa);

    const __m256d imm0 = _mm256_add_pd(_mm256_add_pd(z0, z1), _mm256_add_pd(z2, z3));
    const __m256d imm1 = wb;

    return __m256dx2(imm0, imm1);
}

// e0,...,e11 -> e0+e6,e1+e7,e2+e8,e3+e9,e4+e10,e5+e11,_,_
__forceinline __m256dx2 _mm256_sum12to6_pd(const __m256d x0, const __m256d x1, const __m256d x2) {
    const __m256d y1 = _mm256_permute4x64_pd(x1, _MM_PERM_BADC);
    const __m256d y2 = _mm256_permute4x64_pd(x2, _MM_PERM_BADC);

    const __m256d z0 = x0;
    const __m256d z1 = _mm256_blend_pd(y1, y2, 0b1100);
    const __m256d z2 = _mm256_blend_pd(y1, y2, 0b0011);

    const __m256d imm0 = _mm256_add_pd(z0, z1);
    const __m256d imm1 = _mm256_add_pd(x1, y2);

    return __m256dx2(imm0, imm1);
}

// e0,e1,e2,e3 -> e0+e1,e2+e3
__forceinline __m128d _mm256_hadd2_pd(const __m256d x) {
    const __m128d lo = _mm256_castpd256_pd128(x);
    const __m128d hi = _mm256_extractf128_pd(x, 1);

    const __m128d ret = _mm_hadd_pd(lo, hi);

    return ret;
}