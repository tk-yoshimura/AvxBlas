#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
using namespace System;

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

#pragma unmanaged

void align_kernel_s(
    const uint n, const uint unaligned_w_size, const uint aligned_w_size,
    infloats unaligned_w_ptr, outfloats aligned_w_ptr) {

#ifdef _DEBUG
    if (((unaligned_w_size + AVX2_FLOAT_REMAIN_MASK) & AVX2_FLOAT_BATCH_MASK) != aligned_w_size || unaligned_w_size == aligned_w_size ||
        ((size_t)unaligned_w_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)aligned_w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(unaligned_w_size & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        uint r = unaligned_w_size;

        while (r >= AVX2_FLOAT_STRIDE) {
            __m256 x = _mm256_loadu_ps(unaligned_w_ptr);
            _mm256_store_ps(aligned_w_ptr, x);

            unaligned_w_ptr += AVX2_FLOAT_STRIDE;
            aligned_w_ptr += AVX2_FLOAT_STRIDE;
            r -= AVX2_FLOAT_STRIDE;
        }
        if (r > 0) {
            __m256 x = _mm256_maskload_ps(unaligned_w_ptr, mask);
            _mm256_store_ps(aligned_w_ptr, x);

            unaligned_w_ptr += r;
            aligned_w_ptr += AVX2_FLOAT_STRIDE;
        }
    }
}

void unalign_kernel_s(
    const uint n, const uint aligned_w_size, const uint unaligned_w_size,
    infloats aligned_w_ptr, outfloats unaligned_w_ptr) {

#ifdef _DEBUG
    if (((unaligned_w_size + AVX2_FLOAT_REMAIN_MASK) & AVX2_FLOAT_BATCH_MASK) != aligned_w_size || unaligned_w_size == aligned_w_size ||
        ((size_t)unaligned_w_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)aligned_w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(unaligned_w_size & AVX2_FLOAT_REMAIN_MASK);

    for (uint i = 0; i < n; i++) {
        uint r = unaligned_w_size;

        while (r >= AVX2_FLOAT_STRIDE) {
            __m256 x = _mm256_load_ps(aligned_w_ptr);
            _mm256_storeu_ps(unaligned_w_ptr, x);

            aligned_w_ptr += AVX2_FLOAT_STRIDE;
            unaligned_w_ptr += AVX2_FLOAT_STRIDE;
            r -= AVX2_FLOAT_STRIDE;
        }
        if (r > 0) {
            __m256 x = _mm256_load_ps(aligned_w_ptr);
            _mm256_maskstore_ps(unaligned_w_ptr, mask, x);

            aligned_w_ptr += AVX2_FLOAT_STRIDE;
            unaligned_w_ptr += r;
        }
    }
}