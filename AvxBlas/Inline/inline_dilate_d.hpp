#pragma once
#pragma unmanaged

#include "../utils.h"

// e0,e1,_,_ -> e0,e0,e1,e1
__forceinline __m256d _mm256_dilate2_imm0_pd(__m256d x) {
    __m256d ret = _mm256_permute4x64_pd(x, _MM_PERM_BBAA);

    return ret;
}

// _,_,e2,e3 -> e2,e2,e3,e3
__forceinline __m256d _mm256_dilate2_imm1_pd(__m256d x) {
    __m256d ret = _mm256_permute4x64_pd(x, _MM_PERM_DDCC);

    return ret;
}

// e0,_,_,e3 -> e0,e0,e0,e3
__forceinline __m256d _mm256_dilate3_imm0_pd(__m256d x) {
    __m256d ret = _mm256_permute4x64_pd(x, _MM_PERM_DAAA);

    return ret;
}

// _,e1,_,e3 -> e1,e1,e1,e3
__forceinline __m256d _mm256_dilate3_imm1_pd(__m256d x) {
    __m256d ret = _mm256_permute4x64_pd(x, _MM_PERM_DBBB);

    return ret;
}

// _,_,e2,_-> e2,e2,e2,e3
__forceinline __m256d _mm256_dilate3_imm2_pd(__m256d x) {
    __m256d ret = _mm256_permute4x64_pd(x, _MM_PERM_DCCC);

    return ret;
}

// e0,_,_,_ -> e0,e0,e0,e0
__forceinline __m256d _mm256_dilate4_imm0_pd(__m256d x) {
    __m256d ret = _mm256_permute4x64_pd(x, _MM_PERM_AAAA);

    return ret;
}

// _,e1,_,_ -> e1,e1,e1,e1
__forceinline __m256d _mm256_dilate4_imm1_pd(__m256d x) {
    __m256d ret = _mm256_permute4x64_pd(x, _MM_PERM_BBBB);

    return ret;
}

// _,_,e2,_ -> e2,e2,e2,e2
__forceinline __m256d _mm256_dilate4_imm2_pd(__m256d x) {
    __m256d ret = _mm256_permute4x64_pd(x, _MM_PERM_CCCC);

    return ret;
}

// _,_,_,e3 -> e3,e3,e3,e3
__forceinline __m256d _mm256_dilate4_imm3_pd(__m256d x) {
    __m256d ret = _mm256_permute4x64_pd(x, _MM_PERM_DDDD);

    return ret;
}