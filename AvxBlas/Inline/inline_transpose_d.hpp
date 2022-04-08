#pragma once
#pragma unmanaged

#include "../utils.h"

// e0,e1,...,e6,e7 -> e0,e2,...,e6,e1,e3,...e7 
__forceinline __m256dx2 _mm256_transpose4x2_pd(__m256d x0, __m256d x1) {
    __m256d lo = _mm256_unpacklo_pd(x0, x1);
    __m256d hi = _mm256_unpackhi_pd(x0, x1);

    __m256d imm0 = _mm256_permute2f128_pd(lo, hi, 0b00100000);
    __m256d imm1 = _mm256_permute2f128_pd(lo, hi, 0b00110001);

    return __m256dx2(imm0, imm1);
}