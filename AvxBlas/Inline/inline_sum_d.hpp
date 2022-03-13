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

// e0,e1,e2,e3 -> e0+e1,e2+e3
__forceinline __m128d _mm256_hadd2_pd(const __m256d x) {
    const __m128d lo = _mm256_castpd256_pd128(x);
    const __m128d hi = _mm256_extractf128_pd(x, 1);

    const __m128d ret = _mm_hadd_pd(lo, hi);

    return ret;
}