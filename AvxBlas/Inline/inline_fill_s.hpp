#pragma once
#pragma unmanaged

#include "../utils.h"
#include "../constants.h"
#include "inline_loadstore_xn_s.hpp"

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

__forceinline void fill_n32x_s(const uint n, float c, outfloats y_ptr) {
#ifdef _DEBUG
    if ((n % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256 fillc = _mm256_set1_ps(c);

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_store_x4_ps(y_ptr, fillc, fillc, fillc, fillc);

        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
}

__forceinline void fill_aligned_s(const uint n, float c, outfloats y_ptr) {
#ifdef _DEBUG
    if ((n & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256 fillc = _mm256_set1_ps(c);

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_store_x4_ps(y_ptr, fillc, fillc, fillc, fillc);

        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_store_x2_ps(y_ptr, fillc, fillc);

        y_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_store_ps(y_ptr, fillc);
    }
}

__forceinline void fill_unaligned_s(const uint n, float c, outfloats y_ptr, const __m256i mask) {
#ifdef _DEBUG
    if ((n & AVX2_FLOAT_REMAIN_MASK) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256 fillc = _mm256_set1_ps(c);

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_storeu_x4_ps(y_ptr, fillc, fillc, fillc, fillc);

        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_storeu_x2_ps(y_ptr, fillc, fillc);

        y_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_storeu_x1_ps(y_ptr, fillc);

        y_ptr += AVX2_FLOAT_STRIDE;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r > 0) {
        _mm256_maskstore_x1_ps(y_ptr, fillc, mask);
    }
}