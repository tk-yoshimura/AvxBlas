#pragma once
#pragma unmanaged

#include "../utils.h"

// e0,e1,e2,e3,_,_,_,_ -> e0,e0,e1,e1,e2,e2,e3,e3
__forceinline __m256 _mm256_dilate2_ps(__m256 x) {
    const __m256i _perm = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);

    const __m256 ret = _mm256_permutevar8x32_ps(x, _perm);

    return ret;
}

// e0,e1,_,_,_,_,e6,e7 -> e0,e0,e0,e1,e1,e1,e6,e7
__forceinline __m256 _mm256_dilate3_ps(__m256 x) {
    const __m256i _perm = _mm256_setr_epi32(0, 0, 0, 1, 1, 1, 6, 7);

    const __m256 ret = _mm256_permutevar8x32_ps(x, _perm);

    return ret;
}

// e0,e1,_,_,_,_,_,_ -> e0,e0,e0,e0,e1,e1,e1,e1
__forceinline __m256 _mm256_dilate4_ps(__m256 x) {
    const __m256i _perm = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);

    const __m256 ret = _mm256_permutevar8x32_ps(x, _perm);

    return ret;
}