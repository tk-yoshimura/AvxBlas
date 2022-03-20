#pragma once
#pragma unmanaged

#include "../utils.h"

// float 4 elems 
// (s, c) += a * b
__forceinline void _mm_kahanfma_ps(const __m128 a, const __m128 b, __m128& s, __m128& c) {
    __m128 tmp = s;
    s = _mm_fmadd_ps(a, b, _mm_add_ps(c, s));
    c = _mm_add_ps(c, _mm_fmadd_ps(a, b, _mm_sub_ps(tmp, s)));
}

// float 8 elems
// (s, c) += a * b
__forceinline void _mm256_kahanfma_ps(const __m256 a, const __m256 b, __m256& s, __m256& c) {
    __m256 tmp = s;
    s = _mm256_fmadd_ps(a, b, _mm256_add_ps(c, s));
    c = _mm256_add_ps(c, _mm256_fmadd_ps(a, b, _mm256_sub_ps(tmp, s)));
}