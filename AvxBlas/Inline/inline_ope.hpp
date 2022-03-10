#pragma once
#pragma unmanaged

#include "../utils.h"

__forceinline __m256 _mm256_abs_ps(const __m256 x) {
    const register __m256 bitmask = _mm256_set1_ps(_m32(0x7FFFFFFFu).f);
    
    const __m256 ret = _mm256_and_ps(x, bitmask);

    return ret;
}

__forceinline __m256d _mm256_abs_pd(const __m256d x) {
    const register __m256d bitmask = _mm256_set1_pd(_m64(0x7FFFFFFFFFFFFFFFul).f);
    
    const __m256d ret = _mm256_and_pd(x, bitmask);

    return ret;
}