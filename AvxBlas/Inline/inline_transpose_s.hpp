#pragma once
#pragma unmanaged

#include "../utils.h"

// e0,e1,...,e14,e15 -> e0,e2,...,e14,e1,e3,...,e15 
__forceinline __m256x2 _mm256_transpose8x2_ps(__m256 x0, __m256 x1) {
    __m256 lo = _mm256_unpacklo_ps(x0, x1);
    __m256 hi = _mm256_unpackhi_ps(x0, x1);

    __m256 imm0 = _mm256_permute2f128_ps(lo, hi, 0b00100000);
    __m256 imm1 = _mm256_permute2f128_ps(lo, hi, 0b00110001);

    return __m256x2(imm0, imm1);
}