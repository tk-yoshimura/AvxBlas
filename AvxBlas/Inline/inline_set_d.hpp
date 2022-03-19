#pragma once
#pragma unmanaged

#include "../utils.h"

__forceinline __m256d _mm256_set2_pd(double x1, double x2) {
    return _mm256_setr_pd(x1, x2, x1, x2);
}

__forceinline __m256d _mm256_set3_pd(double x1, double x2, double x3) {
    return _mm256_setr_pd(x1, x2, x3, 0.0);
}