#pragma once
#pragma unmanaged

#include "../utils.h"

// double 2 elems 
// (s, c) += a * b
__forceinline void _mm_kahanfma_pd(const __m128d a, const __m128d b, __m128d& s, __m128d& c) {
    __m128d tmp = s;
    s = _mm_fmadd_pd(a, b, _mm_add_pd(c, s));
    c = _mm_add_pd(c, _mm_fmadd_pd(a, b, _mm_sub_pd(tmp, s)));
}

// double 4 elems
// (s, c) += a * b
__forceinline void _mm256_kahanfma_pd(const __m256d a, const __m256d b, __m256d& s, __m256d& c) {
    __m256d tmp = s;
    s = _mm256_fmadd_pd(a, b, _mm256_add_pd(c, s));
    c = _mm256_add_pd(c, _mm256_fmadd_pd(a, b, _mm256_sub_pd(tmp, s)));
}