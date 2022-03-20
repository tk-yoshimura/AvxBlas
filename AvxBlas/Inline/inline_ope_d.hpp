#pragma once
#pragma unmanaged

#include "../utils.h"

__forceinline __m256d _mm256_abs_pd(const __m256d x) {
    const __m256d bitmask = _mm256_set1_pd(_m64(0x7FFFFFFFFFFFFFFFul).f);

    const __m256d ret = _mm256_and_pd(x, bitmask);

    return ret;
}