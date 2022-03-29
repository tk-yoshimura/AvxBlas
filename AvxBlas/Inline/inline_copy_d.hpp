#pragma once
#pragma unmanaged

#include "../utils.h"
#include "../constants.h"
#include "inline_loadstore_xn_d.hpp"

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

__forceinline void copy_n16x_d(const uint n, indoubles x_ptr, outdoubles y_ptr) {
#ifdef _DEBUG
    if ((n % (AVX2_DOUBLE_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256d x0, x1, x2, x3;

    uint r = n;

    while (r >= AVX2_DOUBLE_STRIDE * 4) {
        _mm256_load_x4_pd(x_ptr, x0, x1, x2, x3);
        _mm256_stream_x4_pd(y_ptr, x0, x1, x2, x3);

        x_ptr += AVX2_DOUBLE_STRIDE * 4;
        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 4;
    }
}

__forceinline void copy_aligned_d(const uint n, indoubles x_ptr, outdoubles y_ptr) {
#ifdef _DEBUG
    if ((n & AVX2_DOUBLE_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256d x0, x1, x2, x3;

    uint r = n;

    while (r >= AVX2_DOUBLE_STRIDE * 4) {
        _mm256_load_x4_pd(x_ptr, x0, x1, x2, x3);
        _mm256_stream_x4_pd(y_ptr, x0, x1, x2, x3);

        x_ptr += AVX2_DOUBLE_STRIDE * 4;
        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 4;
    }
    if (r >= AVX2_DOUBLE_STRIDE * 2) {
        _mm256_load_x2_pd(x_ptr, x0, x1);
        _mm256_stream_x2_pd(y_ptr, x0, x1);

        x_ptr += AVX2_DOUBLE_STRIDE * 2;
        y_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE * 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE) {
        _mm256_load_x1_pd(x_ptr, x0);
        _mm256_stream_x1_pd(y_ptr, x0);
    }
}

__forceinline void copy_unaligned_d(const uint n, indoubles x_ptr, outdoubles y_ptr, const __m256i mask) {
#ifdef _DEBUG
    if ((n & AVX2_DOUBLE_REMAIN_MASK) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256d x0, x1, x2, x3;

    uint r = n;

    while (r >= AVX2_DOUBLE_STRIDE * 4) {
        _mm256_loadu_x4_pd(x_ptr, x0, x1, x2, x3);
        _mm256_storeu_x4_pd(y_ptr, x0, x1, x2, x3);

        x_ptr += AVX2_DOUBLE_STRIDE * 4;
        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 4;
    }
    if (r >= AVX2_DOUBLE_STRIDE * 2) {
        _mm256_loadu_x2_pd(x_ptr, x0, x1);
        _mm256_storeu_x2_pd(y_ptr, x0, x1);

        x_ptr += AVX2_DOUBLE_STRIDE * 2;
        y_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE * 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE) {
        _mm256_loadu_x1_pd(x_ptr, x0);
        _mm256_storeu_x1_pd(y_ptr, x0);

        x_ptr += AVX2_DOUBLE_STRIDE;
        y_ptr += AVX2_DOUBLE_STRIDE;
        r -= AVX2_DOUBLE_STRIDE;
    }
    if (r > 0) {
        _mm256_maskload_x1_pd(x_ptr, x0, mask);
        _mm256_maskstore_x1_pd(y_ptr, x0, mask);
    }
}

__forceinline void copy_srcaligned_d(const uint n, indoubles x_ptr, outdoubles y_ptr, const __m256i mask) {
#ifdef _DEBUG
    if ((n & AVX2_DOUBLE_REMAIN_MASK) == 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256d x0, x1, x2, x3;

    uint r = n;

    while (r >= AVX2_DOUBLE_STRIDE * 4) {
        _mm256_load_x4_pd(x_ptr, x0, x1, x2, x3);
        _mm256_storeu_x4_pd(y_ptr, x0, x1, x2, x3);

        x_ptr += AVX2_DOUBLE_STRIDE * 4;
        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 4;
    }
    if (r >= AVX2_DOUBLE_STRIDE * 2) {
        _mm256_load_x2_pd(x_ptr, x0, x1);
        _mm256_storeu_x2_pd(y_ptr, x0, x1);

        x_ptr += AVX2_DOUBLE_STRIDE * 2;
        y_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE * 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE) {
        _mm256_load_x1_pd(x_ptr, x0);
        _mm256_storeu_x1_pd(y_ptr, x0);

        x_ptr += AVX2_DOUBLE_STRIDE;
        y_ptr += AVX2_DOUBLE_STRIDE;
        r -= AVX2_DOUBLE_STRIDE;
    }
    if (r > 0) {
        _mm256_maskload_x1_pd(x_ptr, x0, mask);
        _mm256_maskstore_x1_pd(y_ptr, x0, mask);
    }
}


__forceinline void copy_dstaligned_d(const uint n, indoubles x_ptr, outdoubles y_ptr) {
#ifdef _DEBUG
    if ((n & AVX2_DOUBLE_REMAIN_MASK) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256d x0, x1, x2, x3;

    uint r = n;

    while (r >= AVX2_DOUBLE_STRIDE * 4) {
        _mm256_loadu_x4_pd(x_ptr, x0, x1, x2, x3);
        _mm256_stream_x4_pd(y_ptr, x0, x1, x2, x3);

        x_ptr += AVX2_DOUBLE_STRIDE * 4;
        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 4;
    }
    if (r >= AVX2_DOUBLE_STRIDE * 2) {
        _mm256_loadu_x2_pd(x_ptr, x0, x1);
        _mm256_stream_x2_pd(y_ptr, x0, x1);

        x_ptr += AVX2_DOUBLE_STRIDE * 2;
        y_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE * 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE) {
        _mm256_loadu_x1_pd(x_ptr, x0);
        _mm256_stream_x1_pd(y_ptr, x0);
    }
}