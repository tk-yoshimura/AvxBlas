#pragma once
#pragma unmanaged

#include "../utils.h"

__forceinline __m256 _mm256_set2_ps(float x1, float x2) {
    return _mm256_setr_ps(x1, x2, x1, x2, x1, x2, x1, x2);
}

__forceinline __m256 _mm256_set3_ps(float x1, float x2, float x3) {
    return _mm256_setr_ps(x1, x2, x3, x1, x2, x3, 0.0f, 0.0f);
}

__forceinline __m256 _mm256_set4_ps(float x1, float x2, float x3, float x4) {
    return _mm256_setr_ps(x1, x2, x3, x4, x1, x2, x3, x4);
}