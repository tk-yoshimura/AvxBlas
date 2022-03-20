#pragma once
#pragma unmanaged

#include "../utils.h"

__forceinline __m256 _mm256_abs_ps(const __m256 x) {
    const __m256 bitmask = _mm256_set1_ps(_m32(0x7FFFFFFFu).f);

    const __m256 ret = _mm256_and_ps(x, bitmask);

    return ret;
}