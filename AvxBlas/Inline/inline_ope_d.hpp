#pragma once
#pragma unmanaged

#include "../utils.h"

__forceinline __m256d _mm256_abs_pd(const __m256d x) {
    const __m256d bitmask = _mm256_set1_pd(_m64(0x7FFFFFFFFFFFFFFFul).f);

    const __m256d ret = _mm256_and_pd(x, bitmask);

    return ret;
}

__forceinline __m256d _mm256_neg_pd(const __m256d x) {
    const __m256d bitmask = _mm256_set1_pd(_m64(0x8000000000000000ul).f);

    const __m256d ret = _mm256_xor_pd(x, bitmask);

    return ret;
}

__forceinline __m256d _mm256_sign_pd(__m256d x) {
    const __m256d zeros = _mm256_setzero_pd();
    const __m256d ones = _mm256_set1_pd(1);

    __m256d y = _mm256_sub_pd(
        _mm256_and_pd(ones, _mm256_cmp_pd(x, zeros, _CMP_GT_OS)),
        _mm256_and_pd(ones, _mm256_cmp_pd(x, zeros, _CMP_LT_OS))
    );

    return y;
}

__forceinline __m256d _mm256_isnan_pd(__m256d x) {
    __m256d y = _mm256_cmp_pd(x, x, _CMP_NEQ_UQ);

    return y;
}

__forceinline __m256d _mm256_not_pd(__m256d x) {
    const __m256d setbits = _mm256_castsi256_pd(_mm256_set1_epi32(~0u));

    __m256d y = _mm256_xor_pd(x, setbits);

    return y;
}

__forceinline __m256d _mm256_square_pd(__m256d x) {
    const __m256d ret = _mm256_mul_pd(x, x);

    return ret;
}

__forceinline __m256d _mm256_cube_pd(__m256d x) {
    const __m256d ret = _mm256_mul_pd(_mm256_mul_pd(x, x), x);

    return ret;
}

__forceinline __m256d _mm256_signedpow_pd(__m256d x1, __m256d x2) {
    const __m256d bitmask_abs = _mm256_set1_pd(_m64(0x7FFFFFFFFFFFFFFFul).f);
    const __m256d bitmask_sign = _mm256_set1_pd(_m64(0x8000000000000000ul).f);

    __m256d y = _mm256_or_pd(
        _mm256_and_pd(bitmask_sign, x1),
        _mm256_pow_pd(_mm256_and_pd(bitmask_abs, x1), x2)
    );

    return y;
}

__forceinline __m256d _mm256_signedsqrt_pd(__m256d x) {
    const __m256d bitmask_abs = _mm256_set1_pd(_m64(0x7FFFFFFFFFFFFFFFul).f);
    const __m256d bitmask_sign = _mm256_set1_pd(_m64(0x8000000000000000ul).f);

    __m256d y = _mm256_or_pd(
        _mm256_and_pd(bitmask_sign, x),
        _mm256_sqrt_pd(_mm256_and_pd(bitmask_abs, x))
    );

    return y;
}

__forceinline __m256d _mm256_nanaszero_pd(__m256d x) {
    __m256d y = _mm256_and_pd(x, _mm256_cmp_pd(x, x, _CMP_EQ_OS));

    return y;
}