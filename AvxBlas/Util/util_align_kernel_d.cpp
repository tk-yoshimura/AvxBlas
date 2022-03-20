#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
using namespace System;

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

#pragma unmanaged

void align_kernel_d(
    const unsigned int n, const unsigned int unaligned_w_size, const unsigned int aligned_w_size,
    indoubles unaligned_w_ptr, outdoubles aligned_w_ptr) {

#ifdef _DEBUG
    if (((unaligned_w_size + AVX2_DOUBLE_REMAIN_MASK) & AVX2_DOUBLE_BATCH_MASK) != aligned_w_size || unaligned_w_size == aligned_w_size ||
        ((size_t)unaligned_w_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)aligned_w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_pd(unaligned_w_size & AVX2_DOUBLE_REMAIN_MASK);

    for (unsigned int i = 0; i < n; i++) {
        unsigned int r = unaligned_w_size;

        while (r >= AVX2_DOUBLE_STRIDE) {
            __m256d x = _mm256_loadu_pd(unaligned_w_ptr);
            _mm256_store_pd(aligned_w_ptr, x);

            unaligned_w_ptr += AVX2_DOUBLE_STRIDE;
            aligned_w_ptr += AVX2_DOUBLE_STRIDE;
            r -= AVX2_DOUBLE_STRIDE;
        }
        if (r > 0) {
            __m256d x = _mm256_maskload_pd(unaligned_w_ptr, mask);
            _mm256_store_pd(aligned_w_ptr, x);

            unaligned_w_ptr += r;
            aligned_w_ptr += AVX2_DOUBLE_STRIDE;
        }
    }
}

void unalign_kernel_d(
    const unsigned int n, const unsigned int aligned_w_size, const unsigned int unaligned_w_size,
    indoubles aligned_w_ptr, outdoubles unaligned_w_ptr) {

#ifdef _DEBUG
    if (((unaligned_w_size + AVX2_DOUBLE_REMAIN_MASK) & AVX2_DOUBLE_BATCH_MASK) != aligned_w_size || unaligned_w_size == aligned_w_size ||
        ((size_t)unaligned_w_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)aligned_w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(unaligned_w_size & AVX2_DOUBLE_REMAIN_MASK);

    for (unsigned int i = 0; i < n; i++) {
        unsigned int r = unaligned_w_size;

        while (r >= AVX2_DOUBLE_STRIDE) {
            __m256d x = _mm256_load_pd(aligned_w_ptr);
            _mm256_storeu_pd(unaligned_w_ptr, x);

            aligned_w_ptr += AVX2_DOUBLE_STRIDE;
            unaligned_w_ptr += AVX2_DOUBLE_STRIDE;
            r -= AVX2_DOUBLE_STRIDE;
        }
        if (r > 0) {
            __m256d x = _mm256_load_pd(aligned_w_ptr);
            _mm256_maskstore_pd(unaligned_w_ptr, mask, x);

            aligned_w_ptr += AVX2_DOUBLE_STRIDE;
            unaligned_w_ptr += r;
        }
    }
}