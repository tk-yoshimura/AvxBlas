#pragma once
#pragma unmanaged

#include "../utils.h"

__forceinline __m256 _mm256_abs_ps(const __m256 x) {
    const __m256 bitmask = _mm256_set1_ps(_m32(0x7FFFFFFFu).f);

    const __m256 ret = _mm256_and_ps(x, bitmask);

    return ret;
}

__forceinline __m256 _mm256_neg_ps(const __m256 x) {
    const __m256 bitmask = _mm256_set1_ps(_m32(0x80000000u).f);

    const __m256 ret = _mm256_xor_ps(x, bitmask);

    return ret;
}

__forceinline __m256 _mm256_sign_ps(__m256 x) {
    const __m256 zeros = _mm256_setzero_ps();
    const __m256 ones = _mm256_set1_ps(1);

    __m256 y = _mm256_sub_ps(
        _mm256_and_ps(ones, _mm256_cmp_ps(x, zeros, _CMP_GT_OS)),
        _mm256_and_ps(ones, _mm256_cmp_ps(x, zeros, _CMP_LT_OS))
    );

    return y;
}

__forceinline __m256 _mm256_isnan_ps(__m256 x) {
    __m256 y = _mm256_cmp_ps(x, x, _CMP_NEQ_UQ);

    return y;
}

__forceinline __m256 _mm256_not_ps(__m256 x) {
    const __m256 setbits = _mm256_castsi256_ps(_mm256_set1_epi32(~0u));

    __m256 y = _mm256_xor_ps(x, setbits);

    return y;
}

__forceinline __m256 _mm256_square_ps(__m256 x) {
    const __m256 ret = _mm256_mul_ps(x, x);

    return ret;
}

__forceinline __m256 _mm256_cube_ps(__m256 x) {
    const __m256 ret = _mm256_mul_ps(_mm256_mul_ps(x, x), x);

    return ret;
}

__forceinline __m256 _mm256_signedpow_ps(__m256 x1, __m256 x2) {
    const __m256 bitmask_abs = _mm256_set1_ps(_m32(0x7FFFFFFFu).f);
    const __m256 bitmask_sign = _mm256_set1_ps(_m32(0x80000000u).f);

    __m256 y = _mm256_or_ps(
        _mm256_and_ps(bitmask_sign, x1),
        _mm256_pow_ps(_mm256_and_ps(bitmask_abs, x1), x2)
    );

    return y;
}

__forceinline __m256 _mm256_signedsqrt_ps(__m256 x) {
    const __m256 bitmask_abs = _mm256_set1_ps(_m32(0x7FFFFFFFu).f);
    const __m256 bitmask_sign = _mm256_set1_ps(_m32(0x80000000u).f);

    __m256 y = _mm256_or_ps(
        _mm256_and_ps(bitmask_sign, x),
        _mm256_sqrt_ps(_mm256_and_ps(bitmask_abs, x))
    );

    return y;
}

__forceinline __m256 _mm256_nanaszero_ps(__m256 x) {
    __m256 y = _mm256_and_ps(x, _mm256_cmp_ps(x, x, _CMP_EQ_OS));

    return y;
}