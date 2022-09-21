#pragma once
#pragma unmanaged

#include "../utils.h"
#include "../constants.h"
#include "inline_loadstore_xn_d.hpp"

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

__forceinline void fill_n16x_d(const uint n, double c, outdoubles y_ptr) {
#ifdef _DEBUG
    if ((n % (AVX2_DOUBLE_STRIDE * 4)) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256d fillc = _mm256_set1_pd(c);

    uint r = n;

    while (r >= AVX2_DOUBLE_STRIDE * 4) {
        _mm256_store_x4_pd(y_ptr, fillc, fillc, fillc, fillc);

        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 4;
    }
}

__forceinline void fill_aligned_d(const uint n, double c, outdoubles y_ptr) {
#ifdef _DEBUG
    if ((n & AVX2_DOUBLE_REMAIN_MASK) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256d fillc = _mm256_set1_pd(c);

    uint r = n;

    while (r >= AVX2_DOUBLE_STRIDE * 4) {
        _mm256_store_x4_pd(y_ptr, fillc, fillc, fillc, fillc);

        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 4;
    }
    if (r >= AVX2_DOUBLE_STRIDE * 2) {
        _mm256_store_x2_pd(y_ptr, fillc, fillc);

        y_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE * 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE) {
        _mm256_store_pd(y_ptr, fillc);
    }
}

__forceinline void fill_unaligned_d(const uint n, double c, outdoubles y_ptr, const __m256i mask) {
#ifdef _DEBUG
    if ((n & AVX2_DOUBLE_REMAIN_MASK) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256d fillc = _mm256_set1_pd(c);

    uint r = n;

    while (r >= AVX2_DOUBLE_STRIDE * 4) {
        _mm256_storeu_x4_pd(y_ptr, fillc, fillc, fillc, fillc);

        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 4;
    }
    if (r >= AVX2_DOUBLE_STRIDE * 2) {
        _mm256_storeu_x2_pd(y_ptr, fillc, fillc);

        y_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE * 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE) {
        _mm256_storeu_x1_pd(y_ptr, fillc);

        y_ptr += AVX2_DOUBLE_STRIDE;
        r -= AVX2_DOUBLE_STRIDE;
    }
    if (r > 0) {
        _mm256_maskstore_x1_pd(y_ptr, fillc, mask);
    }
}