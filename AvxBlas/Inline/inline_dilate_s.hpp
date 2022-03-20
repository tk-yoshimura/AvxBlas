#pragma once
#pragma unmanaged

#include "../utils.h"

// e0,e1,e2,e3,_,_,_,_ -> e0,e0,e1,e1,e2,e2,e3,e3
__forceinline __m256 _mm256_dilate2_imm0_ps(__m256 x) {
    const __m256i _perm = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);

    const __m256 ret = _mm256_permutevar8x32_ps(x, _perm);

    return ret;
}

// _,_,_,_,e4,e5,e6,e7 -> e4,e4,e5,e5,e6,e6,e7,e7
__forceinline __m256 _mm256_dilate2_imm1_ps(__m256 x) {
    const __m256i _perm = _mm256_setr_epi32(4, 4, 5, 5, 6, 6, 7, 7);

    const __m256 ret = _mm256_permutevar8x32_ps(x, _perm);

    return ret;
}

// e0,e1,_,_,_,_,e6,e7 -> e0,e0,e0,e1,e1,e1,e6,e7
__forceinline __m256 _mm256_dilate3_imm0_ps(__m256 x) {
    const __m256i _perm = _mm256_setr_epi32(0, 0, 0, 1, 1, 1, 6, 7);

    const __m256 ret = _mm256_permutevar8x32_ps(x, _perm);

    return ret;
}

// _,_,e2,e3,_,_,e6,e7 -> e2,e2,e2,e3,e3,e3,e6,e7
__forceinline __m256 _mm256_dilate3_imm1_ps(__m256 x) {
    const __m256i _perm = _mm256_setr_epi32(2, 2, 2, 3, 3, 3, 6, 7);

    const __m256 ret = _mm256_permutevar8x32_ps(x, _perm);

    return ret;
}

// _,_,_,_,e4,e5,e6,e7 -> e4,e4,e4,e5,e5,e5,e6,e7
__forceinline __m256 _mm256_dilate3_imm2_ps(__m256 x) {
    const __m256i _perm = _mm256_setr_epi32(4, 4, 4, 5, 5, 5, 6, 7);

    const __m256 ret = _mm256_permutevar8x32_ps(x, _perm);

    return ret;
}

// _,_,_,_,_,_,e6,e7 -> e6,e6,e6,e7,e7,e7,e6,e7
__forceinline __m256 _mm256_dilate3_imm3_ps(__m256 x) {
    const __m256i _perm = _mm256_setr_epi32(6, 6, 6, 7, 7, 7, 6, 7);

    const __m256 ret = _mm256_permutevar8x32_ps(x, _perm);

    return ret;
}

// e0,e1,_,_,_,_,_,_ -> e0,e0,e0,e0,e1,e1,e1,e1
__forceinline __m256 _mm256_dilate4_imm0_ps(__m256 x) {
    const __m256i _perm = _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1);

    const __m256 ret = _mm256_permutevar8x32_ps(x, _perm);

    return ret;
}

// _,_,e2,e3,_,_,_,_ -> e2,e2,e2,e2,e3,e3,e3,e3
__forceinline __m256 _mm256_dilate4_imm1_ps(__m256 x) {
    const __m256i _perm = _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3);

    const __m256 ret = _mm256_permutevar8x32_ps(x, _perm);

    return ret;
}

// _,_,_,_,e4,e5,_,_ -> e4,e4,e4,e4,e5,e5,e5,e5
__forceinline __m256 _mm256_dilate4_imm2_ps(__m256 x) {
    const __m256i _perm = _mm256_setr_epi32(4, 4, 4, 4, 5, 5, 5, 5);

    const __m256 ret = _mm256_permutevar8x32_ps(x, _perm);

    return ret;
}

// _,_,_,_,_,_,e6,e7 -> e6,e6,e6,e6,e7,e7,e7,e7
__forceinline __m256 _mm256_dilate4_imm3_ps(__m256 x) {
    const __m256i _perm = _mm256_setr_epi32(6, 6, 6, 6, 7, 7, 7, 7);

    const __m256 ret = _mm256_permutevar8x32_ps(x, _perm);

    return ret;
}