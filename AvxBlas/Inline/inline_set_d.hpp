#pragma once
#pragma unmanaged

#include "../utils.h"

__forceinline __m256d _mm256_set2_pd(double x0, double x1) {
    return _mm256_setr_pd(x0, x1, x0, x1);
}

__forceinline __m256dx3 _mm256_set3_pd(double x0, double x1, double x2) {
    __m256d imm0 = _mm256_setr_pd(x0, x1, x2, x0);
    __m256d imm1 = _mm256_permute4x64_pd(imm0, _MM_PERM_BACB);
    __m256d imm2 = _mm256_permute4x64_pd(imm0, _MM_PERM_CBAC);

    return __m256dx3(imm0, imm1, imm2);
}