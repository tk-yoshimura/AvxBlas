#pragma once
#pragma unmanaged

#include "../utils.h"
#include "../types.h"

__forceinline __m256 _mm256_set2_ps(float x0, float x1) {
    return _mm256_setr_ps(x0, x1, x0, x1, x0, x1, x0, x1);
}

__forceinline __m256x3 _mm256_set3_ps(float x0, float x1, float x2) {
    const __m256i __perm_x1 = _mm256_setr_epi32(2, 0, 1, 2, 3, 4, 5, 0);
    const __m256i __perm_x2 = _mm256_setr_epi32(1, 2, 0, 1, 2, 3, 4, 5);

    __m256 imm0 = _mm256_setr_ps(x0, x1, x2, x0, x1, x2, x0, x1);
    __m256 imm1 = _mm256_permutevar8x32_ps(imm0, __perm_x1);
    __m256 imm2 = _mm256_permutevar8x32_ps(imm0, __perm_x2);

    return __m256x3(imm0, imm1, imm2);
}

__forceinline __m256 _mm256_set4_ps(float x0, float x1, float x2, float x3) {
    return _mm256_setr_ps(x0, x1, x2, x3, x0, x1, x2, x3);
}

__forceinline __m256x5 _mm256_set5_ps(float x0, float x1, float x2, float x3, float x4) {
    const __m256i __perm_x1 = _mm256_setr_epi32(3, 4, 0, 1, 2, 3, 4, 0);
    const __m256i __perm_x2 = _mm256_setr_epi32(1, 2, 3, 4, 0, 1, 2, 3);
    const __m256i __perm_x3 = _mm256_setr_epi32(4, 0, 1, 2, 3, 4, 0, 1);
    const __m256i __perm_x4 = _mm256_setr_epi32(2, 3, 4, 0, 1, 2, 3, 4);

    __m256 imm0 = _mm256_setr_ps(x0, x1, x2, x3, x4, x0, x1, x2);
    __m256 imm1 = _mm256_permutevar8x32_ps(imm0, __perm_x1);
    __m256 imm2 = _mm256_permutevar8x32_ps(imm0, __perm_x2);
    __m256 imm3 = _mm256_permutevar8x32_ps(imm0, __perm_x3);
    __m256 imm4 = _mm256_permutevar8x32_ps(imm0, __perm_x4);

    return __m256x5(imm0, imm1, imm2, imm3, imm4);
}

__forceinline __m256x3 _mm256_set6_ps(float x0, float x1, float x2, float x3, float x4, float x5) {
    const __m256i __perm_x1 = _mm256_setr_epi32(2, 3, 4, 5, 0, 1, 2, 3);
    const __m256i __perm_x2 = _mm256_setr_epi32(4, 5, 0, 1, 2, 3, 4, 5);

    __m256 imm0 = _mm256_setr_ps(x0, x1, x2, x3, x4, x5, x0, x1);
    __m256 imm1 = _mm256_permutevar8x32_ps(imm0, __perm_x1);
    __m256 imm2 = _mm256_permutevar8x32_ps(imm0, __perm_x2);

    return __m256x3(imm0, imm1, imm2);
}