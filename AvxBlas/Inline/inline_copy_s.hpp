#pragma once
#pragma unmanaged

#include "../utils.h"
#include "../constants.h"

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

__forceinline void copy_n32x_s(const unsigned int n, const float* im_ptr, float* col_ptr) {
#ifdef _DEBUG
    if ((n % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)im_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)col_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    unsigned int r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        __m256 x0 = _mm256_load_ps(im_ptr);
        __m256 x1 = _mm256_load_ps(im_ptr + AVX2_FLOAT_STRIDE);
        __m256 x2 = _mm256_load_ps(im_ptr + AVX2_FLOAT_STRIDE * 2);
        __m256 x3 = _mm256_load_ps(im_ptr + AVX2_FLOAT_STRIDE * 3);

        _mm256_store_ps(col_ptr, x0);
        _mm256_store_ps(col_ptr + AVX2_FLOAT_STRIDE, x1);
        _mm256_store_ps(col_ptr + AVX2_FLOAT_STRIDE * 2, x2);
        _mm256_store_ps(col_ptr + AVX2_FLOAT_STRIDE * 3, x3);

        im_ptr += AVX2_FLOAT_STRIDE * 4;
        col_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
}

__forceinline void copy_aligned_s(const unsigned int n, const float* im_ptr, float* col_ptr) {
#ifdef _DEBUG
    if ((n & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)im_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)col_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    unsigned int r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        __m256 x0 = _mm256_load_ps(im_ptr);
        __m256 x1 = _mm256_load_ps(im_ptr + AVX2_FLOAT_STRIDE);
        __m256 x2 = _mm256_load_ps(im_ptr + AVX2_FLOAT_STRIDE * 2);
        __m256 x3 = _mm256_load_ps(im_ptr + AVX2_FLOAT_STRIDE * 3);

        _mm256_store_ps(col_ptr, x0);
        _mm256_store_ps(col_ptr + AVX2_FLOAT_STRIDE, x1);
        _mm256_store_ps(col_ptr + AVX2_FLOAT_STRIDE * 2, x2);
        _mm256_store_ps(col_ptr + AVX2_FLOAT_STRIDE * 3, x3);

        im_ptr += AVX2_FLOAT_STRIDE * 4;
        col_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 3) {
        __m256 x0 = _mm256_load_ps(im_ptr);
        __m256 x1 = _mm256_load_ps(im_ptr + AVX2_FLOAT_STRIDE);
        __m256 x2 = _mm256_load_ps(im_ptr + AVX2_FLOAT_STRIDE * 2);

        _mm256_store_ps(col_ptr, x0);
        _mm256_store_ps(col_ptr + AVX2_FLOAT_STRIDE, x1);
        _mm256_store_ps(col_ptr + AVX2_FLOAT_STRIDE * 2, x2);
    }
    else if (r >= AVX2_FLOAT_STRIDE * 2) {
        __m256 x0 = _mm256_load_ps(im_ptr);
        __m256 x1 = _mm256_load_ps(im_ptr + AVX2_FLOAT_STRIDE);

        _mm256_store_ps(col_ptr, x0);
        _mm256_store_ps(col_ptr + AVX2_FLOAT_STRIDE, x1);
    }
    else if (r >= AVX2_FLOAT_STRIDE) {
        __m256 x0 = _mm256_load_ps(im_ptr);

        _mm256_store_ps(col_ptr, x0);
    }
}

__forceinline void copy_unaligned_s(const unsigned int n, const float* im_ptr, float* col_ptr, const __m256i mask) {
#ifdef _DEBUG
    if ((n & AVX2_FLOAT_REMAIN_MASK) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    unsigned int r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        __m256 x0 = _mm256_loadu_ps(im_ptr);
        __m256 x1 = _mm256_loadu_ps(im_ptr + AVX2_FLOAT_STRIDE);
        __m256 x2 = _mm256_loadu_ps(im_ptr + AVX2_FLOAT_STRIDE * 2);
        __m256 x3 = _mm256_loadu_ps(im_ptr + AVX2_FLOAT_STRIDE * 3);

        _mm256_storeu_ps(col_ptr, x0);
        _mm256_storeu_ps(col_ptr + AVX2_FLOAT_STRIDE, x1);
        _mm256_storeu_ps(col_ptr + AVX2_FLOAT_STRIDE * 2, x2);
        _mm256_storeu_ps(col_ptr + AVX2_FLOAT_STRIDE * 3, x3);

        im_ptr += AVX2_FLOAT_STRIDE * 4;
        col_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 3) {
        __m256 x0 = _mm256_loadu_ps(im_ptr);
        __m256 x1 = _mm256_loadu_ps(im_ptr + AVX2_FLOAT_STRIDE);
        __m256 x2 = _mm256_loadu_ps(im_ptr + AVX2_FLOAT_STRIDE * 2);
        __m256 x3 = _mm256_loadu_ps(im_ptr + AVX2_FLOAT_STRIDE * 3);

        _mm256_storeu_ps(col_ptr, x0);
        _mm256_storeu_ps(col_ptr + AVX2_FLOAT_STRIDE, x1);
        _mm256_storeu_ps(col_ptr + AVX2_FLOAT_STRIDE * 2, x2);
        _mm256_maskstore_ps(col_ptr + AVX2_FLOAT_STRIDE * 3, mask, x3);
    }
    else if (r >= AVX2_FLOAT_STRIDE * 2) {
        __m256 x0 = _mm256_loadu_ps(im_ptr);
        __m256 x1 = _mm256_loadu_ps(im_ptr + AVX2_FLOAT_STRIDE);
        __m256 x2 = _mm256_loadu_ps(im_ptr + AVX2_FLOAT_STRIDE * 2);

        _mm256_storeu_ps(col_ptr, x0);
        _mm256_storeu_ps(col_ptr + AVX2_FLOAT_STRIDE, x1);
        _mm256_maskstore_ps(col_ptr + AVX2_FLOAT_STRIDE * 2, mask, x2);
    }
    else if (r >= AVX2_FLOAT_STRIDE) {
        __m256 x0 = _mm256_loadu_ps(im_ptr);
        __m256 x1 = _mm256_loadu_ps(im_ptr + AVX2_FLOAT_STRIDE);

        _mm256_storeu_ps(col_ptr, x0);
        _mm256_maskstore_ps(col_ptr + AVX2_FLOAT_STRIDE, mask, x1);
    }
    else {
        __m256 x0 = _mm256_loadu_ps(im_ptr);

        _mm256_maskstore_ps(col_ptr, mask, x0);
    }
}