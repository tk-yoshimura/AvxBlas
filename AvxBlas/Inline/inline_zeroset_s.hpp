#pragma once
#pragma unmanaged

#include "../utils.h"
#include "../constants.h"

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

__forceinline void zeroset_n32x_s(const unsigned int n, outfloats col_ptr) {
#ifdef _DEBUG
    if ((n % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)col_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    unsigned int r = n;

    __m256 fillz = _mm256_setzero_ps();

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_store_ps(col_ptr, fillz);
        _mm256_store_ps(col_ptr + AVX2_FLOAT_STRIDE, fillz);
        _mm256_store_ps(col_ptr + AVX2_FLOAT_STRIDE * 2, fillz);
        _mm256_store_ps(col_ptr + AVX2_FLOAT_STRIDE * 3, fillz);

        col_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
}

__forceinline void zeroset_aligned_s(const unsigned int n, outfloats col_ptr) {
#ifdef _DEBUG
    if ((n & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)col_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    unsigned int r = n;

    __m256 fillz = _mm256_setzero_ps();

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_store_ps(col_ptr, fillz);
        _mm256_store_ps(col_ptr + AVX2_FLOAT_STRIDE, fillz);
        _mm256_store_ps(col_ptr + AVX2_FLOAT_STRIDE * 2, fillz);
        _mm256_store_ps(col_ptr + AVX2_FLOAT_STRIDE * 3, fillz);

        col_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_store_ps(col_ptr, fillz);
        _mm256_store_ps(col_ptr + AVX2_FLOAT_STRIDE, fillz);

        col_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_store_ps(col_ptr, fillz);
    }
}

__forceinline void zeroset_unaligned_s(const unsigned int n, outfloats col_ptr, const __m256i mask) {
#ifdef _DEBUG
    if ((n & AVX2_FLOAT_REMAIN_MASK) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    unsigned int r = n;

    __m256 fillz = _mm256_setzero_ps();

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_storeu_ps(col_ptr, fillz);
        _mm256_storeu_ps(col_ptr + AVX2_FLOAT_STRIDE, fillz);
        _mm256_storeu_ps(col_ptr + AVX2_FLOAT_STRIDE * 2, fillz);
        _mm256_storeu_ps(col_ptr + AVX2_FLOAT_STRIDE * 3, fillz);

        col_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_storeu_ps(col_ptr, fillz);
        _mm256_storeu_ps(col_ptr + AVX2_FLOAT_STRIDE, fillz);

        col_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_storeu_ps(col_ptr, fillz);

        col_ptr += AVX2_FLOAT_STRIDE;
        r -= AVX2_FLOAT_STRIDE;
    }
    if(r > 0) {
        _mm256_maskstore_ps(col_ptr, mask, fillz);
    }
}