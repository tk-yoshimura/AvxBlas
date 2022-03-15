#pragma once
#pragma unmanaged

#include "../constants.h"
#include "../utils.h"
#include "inline_dilate.hpp"
#include "inline_set.hpp"

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

__forceinline void kernelfma_n1_aligned_s(
    const unsigned int ic, const unsigned int oc,
    const float* __restrict x_ptr, const float* __restrict y_ptr, float* __restrict w_ptr) {

#ifdef _DEBUG
    if (ic != 1 || (oc & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 x = _mm256_set1_ps(x_ptr[0]);

    unsigned r = oc;
    const float* src_ptr = y_ptr;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_load_ps(src_ptr),
            _mm256_load_ps(w_ptr)
        );
        __m256 w1 = _mm256_fmadd_ps(
            x,
            _mm256_load_ps(src_ptr + AVX2_FLOAT_STRIDE),
            _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE)
        );
        __m256 w2 = _mm256_fmadd_ps(
            x,
            _mm256_load_ps(src_ptr + AVX2_FLOAT_STRIDE * 2),
            _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE * 2)
        );
        __m256 w3 = _mm256_fmadd_ps(
            x,
            _mm256_load_ps(src_ptr + AVX2_FLOAT_STRIDE * 3),
            _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE * 3)
        );

        _mm256_store_ps(w_ptr, w0);
        _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE, w1);
        _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE * 2, w2);
        _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE * 3, w3);

        src_ptr += AVX2_FLOAT_STRIDE * 4;
        w_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 3) {
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_load_ps(src_ptr),
            _mm256_load_ps(w_ptr)
        );
        __m256 w1 = _mm256_fmadd_ps(
            x,
            _mm256_load_ps(src_ptr + AVX2_FLOAT_STRIDE),
            _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE)
        );
        __m256 w2 = _mm256_fmadd_ps(
            x,
            _mm256_load_ps(src_ptr + AVX2_FLOAT_STRIDE * 2),
            _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE * 2)
        );

        _mm256_store_ps(w_ptr, w0);
        _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE, w1);
        _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE * 2, w2);
    }
    else if (r >= AVX2_FLOAT_STRIDE * 2) {
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_load_ps(src_ptr),
            _mm256_load_ps(w_ptr)
        );
        __m256 w1 = _mm256_fmadd_ps(
            x,
            _mm256_load_ps(src_ptr + AVX2_FLOAT_STRIDE),
            _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE)
        );

        _mm256_store_ps(w_ptr, w0);
        _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE, w1);
    }
    else if (r >= AVX2_FLOAT_STRIDE) {
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_load_ps(src_ptr),
            _mm256_load_ps(w_ptr)
        );

        _mm256_store_ps(w_ptr, w0);
    }
}

__forceinline void kernelfma_n1_unaligned_s(
    const unsigned int ic, const unsigned int oc,
    const float* __restrict x_ptr, const float* __restrict y_ptr, float* __restrict w_ptr, const __m256i mask) {

#ifdef _DEBUG
    if (ic != 1 || (oc & AVX2_FLOAT_REMAIN_MASK) == 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 x = _mm256_set1_ps(x_ptr[0]);

    unsigned r = oc;
    const float* src_ptr = y_ptr;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_loadu_ps(src_ptr),
            _mm256_loadu_ps(w_ptr)
        );
        __m256 w1 = _mm256_fmadd_ps(
            x,
            _mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE),
            _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE)
        );
        __m256 w2 = _mm256_fmadd_ps(
            x,
            _mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE * 2),
            _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE * 2)
        );
        __m256 w3 = _mm256_fmadd_ps(
            x,
            _mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE * 3),
            _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE * 3)
        );

        _mm256_storeu_ps(w_ptr, w0);
        _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE, w1);
        _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE * 2, w2);
        _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE * 3, w3);

        src_ptr += AVX2_FLOAT_STRIDE * 4;
        w_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 2) {
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_loadu_ps(src_ptr),
            _mm256_loadu_ps(w_ptr)
        );
        __m256 w1 = _mm256_fmadd_ps(
            x,
            _mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE),
            _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE)
        );

        _mm256_storeu_ps(w_ptr, w0);
        _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE, w1);

        src_ptr += AVX2_FLOAT_STRIDE * 2;
        w_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE) {
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_loadu_ps(src_ptr),
            _mm256_loadu_ps(w_ptr)
        );

        _mm256_storeu_ps(w_ptr, w0);

        src_ptr += AVX2_FLOAT_STRIDE;
        w_ptr += AVX2_FLOAT_STRIDE;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r > 0) {
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_loadu_ps(src_ptr),
            _mm256_loadu_ps(w_ptr)
        );

        _mm256_maskstore_ps(w_ptr, mask, w0);
    }
}

__forceinline void kernelfma_n2_aligned_s(
    const unsigned int ic, const unsigned int oc,
    const float* __restrict x_ptr, const float* __restrict y_ptr, float* __restrict w_ptr) {

#ifdef _DEBUG
    if (ic != 2 || (oc % (AVX2_FLOAT_STRIDE / 2)) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 x = _mm256_set2_ps(x_ptr[0], x_ptr[1]);

    unsigned r = oc;
    const float* src_ptr = y_ptr;

    while (r >= AVX2_FLOAT_STRIDE * 2) {
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_dilate2_ps(_mm256_loadu_ps(src_ptr)),
            _mm256_load_ps(w_ptr)
        );
        __m256 w1 = _mm256_fmadd_ps(
            x,
            _mm256_dilate2_ps(_mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE / 2)),
            _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE)
        );
        __m256 w2 = _mm256_fmadd_ps(
            x,
            _mm256_dilate2_ps(_mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE)),
            _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE * 2)
        );
        __m256 w3 = _mm256_fmadd_ps(
            x,
            _mm256_dilate2_ps(_mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE * 3 / 2)),
            _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE * 3)
        );

        _mm256_store_ps(w_ptr, w0);
        _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE, w1);
        _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE * 2, w2);
        _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE * 3, w3);

        src_ptr += AVX2_FLOAT_STRIDE * 2;
        w_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE * 3 / 2) {
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_dilate2_ps(_mm256_loadu_ps(src_ptr)),
            _mm256_load_ps(w_ptr)
        );
        __m256 w1 = _mm256_fmadd_ps(
            x,
            _mm256_dilate2_ps(_mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE / 2)),
            _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE)
        );
        __m256 w2 = _mm256_fmadd_ps(
            x,
            _mm256_dilate2_ps(_mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE)),
            _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE * 2)
        );

        _mm256_store_ps(w_ptr, w0);
        _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE, w1);
        _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE * 2, w2);
    }
    else if (r >= AVX2_FLOAT_STRIDE) {
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_dilate2_ps(_mm256_loadu_ps(src_ptr)),
            _mm256_load_ps(w_ptr)
        );
        __m256 w1 = _mm256_fmadd_ps(
            x,
            _mm256_dilate2_ps(_mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE / 2)),
            _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE)
        );

        _mm256_store_ps(w_ptr, w0);
        _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE, w1);
    }
    else if (r >= AVX2_FLOAT_STRIDE / 2) {
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_dilate2_ps(_mm256_loadu_ps(src_ptr)),
            _mm256_load_ps(w_ptr)
        );

        _mm256_store_ps(w_ptr, w0);
    }
}

__forceinline void kernelfma_n2_unaligned_s(
    const unsigned int ic, const unsigned int oc,
    const float* __restrict x_ptr, const float* __restrict y_ptr, float* __restrict w_ptr, const __m256i mask) {

#ifdef _DEBUG
    if (ic != 2 || (oc % (AVX2_FLOAT_STRIDE / 2)) == 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 x = _mm256_set2_ps(x_ptr[0], x_ptr[1]);

    unsigned r = oc;
    const float* src_ptr = y_ptr;

    while (r >= AVX2_FLOAT_STRIDE * 2) {
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_dilate2_ps(_mm256_loadu_ps(src_ptr)),
            _mm256_loadu_ps(w_ptr)
        );
        __m256 w1 = _mm256_fmadd_ps(
            x,
            _mm256_dilate2_ps(_mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE / 2)),
            _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE)
        );
        __m256 w2 = _mm256_fmadd_ps(
            x,
            _mm256_dilate2_ps(_mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE)),
            _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE * 2)
        );
        __m256 w3 = _mm256_fmadd_ps(
            x,
            _mm256_dilate2_ps(_mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE * 3 / 2)),
            _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE * 3)
        );

        _mm256_storeu_ps(w_ptr, w0);
        _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE, w1);
        _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE * 2, w2);
        _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE * 3, w3);

        src_ptr += AVX2_FLOAT_STRIDE * 2;
        w_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE) {
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_dilate2_ps(_mm256_loadu_ps(src_ptr)),
            _mm256_loadu_ps(w_ptr)
        );
        __m256 w1 = _mm256_fmadd_ps(
            x,
            _mm256_dilate2_ps(_mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE / 2)),
            _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE)
        );

        _mm256_storeu_ps(w_ptr, w0);
        _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE, w1);

        src_ptr += AVX2_FLOAT_STRIDE;
        w_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r >= AVX2_FLOAT_STRIDE / 2) {
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_dilate2_ps(_mm256_loadu_ps(src_ptr)),
            _mm256_loadu_ps(w_ptr)
        );

        _mm256_storeu_ps(w_ptr, w0);

        src_ptr += AVX2_FLOAT_STRIDE / 2;
        w_ptr += AVX2_FLOAT_STRIDE;
        r -= AVX2_FLOAT_STRIDE / 2;
    }
    if (r > 0) {
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_dilate2_ps(_mm256_loadu_ps(src_ptr)),
            _mm256_loadu_ps(w_ptr)
        );

        _mm256_maskstore_ps(w_ptr, mask, w0);
    }
}

__forceinline void kernelfma_n3_aligned_s(
    const unsigned int ic, const unsigned int oc,
    const float* __restrict x_ptr, const float* __restrict y_ptr, float* __restrict w_ptr) {

#ifdef _DEBUG
    if (ic != 3 || (oc % 8) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 x = _mm256_set3_ps(x_ptr[0], x_ptr[1], x_ptr[2]);

    unsigned r = oc;
    const float* src_ptr = y_ptr;

    const __m256i __perm1 = _mm256_setr_epi32(2, 3, 4, 5, 6, 7, 0, 1);
    const __m256i __perm2 = _mm256_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3);
    const __m256i __perm3 = _mm256_setr_epi32(6, 7, 0, 1, 2, 3, 4, 5);

    while (r >= 8) {
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_dilate3_ps(_mm256_loadu_ps(src_ptr)),
            _mm256_loadu_ps(w_ptr)
        );
        __m256 w1 = _mm256_permutevar8x32_ps(_mm256_fmadd_ps(
            x,
            _mm256_dilate3_ps(_mm256_loadu_ps(src_ptr + 2)),
            _mm256_loadu_ps(w_ptr + 6)
        ), __perm1);
        __m256 w2 = _mm256_permutevar8x32_ps(_mm256_fmadd_ps(
            x,
            _mm256_dilate3_ps(_mm256_loadu_ps(src_ptr + 4)),
            _mm256_loadu_ps(w_ptr + 12)
        ), __perm2);
        __m256 w3 = _mm256_permutevar8x32_ps(_mm256_fmadd_ps(
            x,
            _mm256_dilate3_ps(_mm256_loadu_ps(src_ptr + 6)),
            _mm256_loadu_ps(w_ptr + 18)
        ), __perm3);

        __m256 w01 = _mm256_blend_ps(w0, w1, 0b11000000);
        __m256 w12 = _mm256_blend_ps(w1, w2, 0b11110000);
        __m256 w23 = _mm256_blend_ps(w2, w3, 0b11111100);

        _mm256_store_ps(w_ptr, w01);
        _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE, w12);
        _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE * 2, w23);

        src_ptr += 8;
        w_ptr += AVX2_FLOAT_STRIDE * 3;
        r -= 8;
    }
}


__forceinline void kernelfma_n3_unaligned_s(
    const unsigned int ic, const unsigned int oc,
    const float* __restrict x_ptr, const float* __restrict y_ptr, float* __restrict w_ptr, const __m256i mask) {

#ifdef _DEBUG
    if (ic != 3 || (oc % 8) == 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 x = _mm256_set3_ps(x_ptr[0], x_ptr[1], x_ptr[2]);

    unsigned r = oc;
    const float* src_ptr = y_ptr;

    const __m256i __perm1 = _mm256_setr_epi32(2, 3, 4, 5, 6, 7, 0, 1);
    const __m256i __perm2 = _mm256_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3);
    const __m256i __perm3 = _mm256_setr_epi32(6, 7, 0, 1, 2, 3, 4, 5);
    const __m256 __mask6 = _mm256_castsi256_ps(_mm256_setr_epi32(~0u, ~0u, ~0u, ~0u, ~0u, ~0u, 0, 0));

    while (r >= 8) {
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_dilate3_ps(_mm256_loadu_ps(src_ptr)),
            _mm256_loadu_ps(w_ptr)
        );
        __m256 w1 = _mm256_permutevar8x32_ps(_mm256_fmadd_ps(
            x,
            _mm256_dilate3_ps(_mm256_loadu_ps(src_ptr + 2)),
            _mm256_loadu_ps(w_ptr + 6)
        ), __perm1);
        __m256 w2 = _mm256_permutevar8x32_ps(_mm256_fmadd_ps(
            x,
            _mm256_dilate3_ps(_mm256_loadu_ps(src_ptr + 4)),
            _mm256_loadu_ps(w_ptr + 12)
        ), __perm2);
        __m256 w3 = _mm256_permutevar8x32_ps(_mm256_fmadd_ps(
            x,
            _mm256_dilate3_ps(_mm256_loadu_ps(src_ptr + 6)),
            _mm256_loadu_ps(w_ptr + 18)
        ), __perm3);

        __m256 w01 = _mm256_blend_ps(w0, w1, 0b11000000);
        __m256 w12 = _mm256_blend_ps(w1, w2, 0b11110000);
        __m256 w23 = _mm256_blend_ps(w2, w3, 0b11111100);

        _mm256_storeu_ps(w_ptr, w01);
        _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE, w12);
        _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE * 2, w23);

        src_ptr += 8;
        w_ptr += AVX2_FLOAT_STRIDE * 3;
        r -= 8;
    }
    if (r >= 5) { // 3 * r >= 6 + AVX2_FLOAT_STRIDE
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_dilate3_ps(_mm256_loadu_ps(src_ptr)),
            _mm256_loadu_ps(w_ptr)
        );
        __m256 w1 = _mm256_fmadd_ps(
            x,
            _mm256_and_ps(_mm256_dilate3_ps(_mm256_loadu_ps(src_ptr + 2)), __mask6),
            _mm256_loadu_ps(w_ptr + 6)
        );

        _mm256_storeu_ps(w_ptr, w0);
        _mm256_storeu_ps(w_ptr + 6, w1);

        src_ptr += 4;
        w_ptr += 12;
        r -= 4;
    }
    if (r >= 3) { // 3 * r >= AVX_FLOAT_STRIDE
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_and_ps(_mm256_dilate3_ps(_mm256_loadu_ps(src_ptr)), __mask6),
            _mm256_loadu_ps(w_ptr)
        );

        _mm256_storeu_ps(w_ptr, w0);

        src_ptr += 2;
        w_ptr += 6;
        r -= 2;
    }
    if (r > 0) {
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_dilate3_ps(_mm256_loadu_ps(src_ptr)),
            _mm256_loadu_ps(w_ptr)
        );

        _mm256_maskstore_ps(w_ptr, mask, w0);
    }
}

__forceinline void kernelfma_n4_aligned_s(
    const unsigned int ic, const unsigned int oc,
    const float* __restrict x_ptr, const float* __restrict y_ptr, float* __restrict w_ptr) {

#ifdef _DEBUG
    if (ic != 4 || (oc % 2) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 x = _mm256_set4_ps(x_ptr[0], x_ptr[1], x_ptr[2], x_ptr[3]);

    unsigned r = oc;
    const float* src_ptr = y_ptr;

    while (r >= 8) {
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_dilate4_ps(_mm256_loadu_ps(src_ptr)),
            _mm256_load_ps(w_ptr)
        );
        __m256 w1 = _mm256_fmadd_ps(
            x,
            _mm256_dilate4_ps(_mm256_loadu_ps(src_ptr + 2)),
            _mm256_load_ps(w_ptr + 8)
        );
        __m256 w2 = _mm256_fmadd_ps(
            x,
            _mm256_dilate4_ps(_mm256_loadu_ps(src_ptr + 4)),
            _mm256_load_ps(w_ptr + 16)
        );
        __m256 w3 = _mm256_fmadd_ps(
            x,
            _mm256_dilate4_ps(_mm256_loadu_ps(src_ptr + 6)),
            _mm256_load_ps(w_ptr + 24)
        );

        _mm256_store_ps(w_ptr, w0);
        _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE, w1);
        _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE * 2, w2);
        _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE * 3, w3);

        src_ptr += 8;
        w_ptr += 32;
        r -= 8;
    }
    if (r >= 6) {
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_dilate4_ps(_mm256_loadu_ps(src_ptr)),
            _mm256_load_ps(w_ptr)
        );
        __m256 w1 = _mm256_fmadd_ps(
            x,
            _mm256_dilate4_ps(_mm256_loadu_ps(src_ptr + 2)),
            _mm256_load_ps(w_ptr + 8)
        );
        __m256 w2 = _mm256_fmadd_ps(
            x,
            _mm256_dilate4_ps(_mm256_loadu_ps(src_ptr + 4)),
            _mm256_load_ps(w_ptr + 16)
        );

        _mm256_store_ps(w_ptr, w0);
        _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE, w1);
        _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE * 2, w2);
    }
    else if (r >= 4) {
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_dilate4_ps(_mm256_loadu_ps(src_ptr)),
            _mm256_load_ps(w_ptr)
        );
        __m256 w1 = _mm256_fmadd_ps(
            x,
            _mm256_dilate4_ps(_mm256_loadu_ps(src_ptr + 2)),
            _mm256_load_ps(w_ptr + 8)
        );

        _mm256_store_ps(w_ptr, w0);
        _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE, w1);
    }
    else if (r >= 2) {
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_dilate4_ps(_mm256_loadu_ps(src_ptr)),
            _mm256_load_ps(w_ptr)
        );

        _mm256_store_ps(w_ptr, w0);
    }
}

__forceinline void kernelfma_n4_unaligned_s(
    const unsigned int ic, const unsigned int oc,
    const float* __restrict x_ptr, const float* __restrict y_ptr, float* __restrict w_ptr, const __m256i mask) {

#ifdef _DEBUG
    if (ic != 4 || (oc % 2) == 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256 x = _mm256_set4_ps(x_ptr[0], x_ptr[1], x_ptr[2], x_ptr[3]);

    unsigned r = oc;
    const float* src_ptr = y_ptr;

    while (r >= 8) {
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_dilate4_ps(_mm256_loadu_ps(src_ptr)),
            _mm256_loadu_ps(w_ptr)
        );
        __m256 w1 = _mm256_fmadd_ps(
            x,
            _mm256_dilate4_ps(_mm256_loadu_ps(src_ptr + 2)),
            _mm256_loadu_ps(w_ptr + 8)
        );
        __m256 w2 = _mm256_fmadd_ps(
            x,
            _mm256_dilate4_ps(_mm256_loadu_ps(src_ptr + 4)),
            _mm256_loadu_ps(w_ptr + 16)
        );
        __m256 w3 = _mm256_fmadd_ps(
            x,
            _mm256_dilate4_ps(_mm256_loadu_ps(src_ptr + 6)),
            _mm256_loadu_ps(w_ptr + 24)
        );

        _mm256_storeu_ps(w_ptr, w0);
        _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE, w1);
        _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE * 2, w2);
        _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE * 3, w3);

        src_ptr += 8;
        w_ptr += 32;
        r -= 8;
    }
    if (r >= 4) {
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_dilate4_ps(_mm256_loadu_ps(src_ptr)),
            _mm256_loadu_ps(w_ptr)
        );
        __m256 w1 = _mm256_fmadd_ps(
            x,
            _mm256_dilate4_ps(_mm256_loadu_ps(src_ptr + 2)),
            _mm256_loadu_ps(w_ptr + 8)
        );

        _mm256_storeu_ps(w_ptr, w0);
        _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE, w1);

        src_ptr += 4;
        w_ptr += 16;
        r -= 4;
    }
    if (r >= 2) {
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_dilate4_ps(_mm256_loadu_ps(src_ptr)),
            _mm256_loadu_ps(w_ptr)
        );

        _mm256_storeu_ps(w_ptr, w0);

        src_ptr += 2;
        w_ptr += 8;
        r -= 2;
    }
    if (r > 0) {
        __m256 w0 = _mm256_fmadd_ps(
            x,
            _mm256_dilate4_ps(_mm256_loadu_ps(src_ptr)),
            _mm256_loadu_ps(w_ptr)
        );

        _mm256_maskstore_ps(w_ptr, mask, w0);
    }
}

__forceinline void kernelfma_n32x_s(
    const unsigned int ic, const unsigned int oc,
    const float* __restrict x_ptr, const float* __restrict y_ptr, float* __restrict w_ptr) {

#ifdef _DEBUG
    if ((ic % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    for (unsigned int i = 0; i < oc; i++) {
        __m256 y = _mm256_set1_ps(y_ptr[i]);

        unsigned r = ic;
        const float* src_ptr = x_ptr;

        while (r >= AVX2_FLOAT_STRIDE * 4) {
            __m256 w0 = _mm256_fmadd_ps(
                _mm256_load_ps(src_ptr),
                y,
                _mm256_load_ps(w_ptr)
            );
            __m256 w1 = _mm256_fmadd_ps(
                _mm256_load_ps(src_ptr + AVX2_FLOAT_STRIDE),
                y,
                _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE)
            );
            __m256 w2 = _mm256_fmadd_ps(
                _mm256_load_ps(src_ptr + AVX2_FLOAT_STRIDE * 2),
                y,
                _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE * 2)
            );
            __m256 w3 = _mm256_fmadd_ps(
                _mm256_load_ps(src_ptr + AVX2_FLOAT_STRIDE * 3),
                y,
                _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE * 3)
            );

            _mm256_store_ps(w_ptr, w0);
            _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE, w1);
            _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE * 2, w2);
            _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE * 3, w3);

            src_ptr += AVX2_FLOAT_STRIDE * 4;
            w_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
    }
}

__forceinline void kernelfma_aligned_s(
    const unsigned int ic, const unsigned int oc,
    const float* __restrict x_ptr, const float* __restrict y_ptr, float* __restrict w_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    for (unsigned int i = 0; i < oc; i++) {
        __m256 y = _mm256_set1_ps(y_ptr[i]);

        unsigned r = ic;
        const float* src_ptr = x_ptr;

        while (r >= AVX2_FLOAT_STRIDE * 4) {
            __m256 w0 = _mm256_fmadd_ps(
                _mm256_load_ps(src_ptr),
                y,
                _mm256_load_ps(w_ptr)
            );
            __m256 w1 = _mm256_fmadd_ps(
                _mm256_load_ps(src_ptr + AVX2_FLOAT_STRIDE),
                y,
                _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE)
            );
            __m256 w2 = _mm256_fmadd_ps(
                _mm256_load_ps(src_ptr + AVX2_FLOAT_STRIDE * 2),
                y,
                _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE * 2)
            );
            __m256 w3 = _mm256_fmadd_ps(
                _mm256_load_ps(src_ptr + AVX2_FLOAT_STRIDE * 3),
                y,
                _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE * 3)
            );

            _mm256_store_ps(w_ptr, w0);
            _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE, w1);
            _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE * 2, w2);
            _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE * 3, w3);

            src_ptr += AVX2_FLOAT_STRIDE * 4;
            w_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
        if (r == AVX2_FLOAT_STRIDE * 3) {
            __m256 w0 = _mm256_fmadd_ps(
                _mm256_load_ps(src_ptr),
                y,
                _mm256_load_ps(w_ptr)
            );
            __m256 w1 = _mm256_fmadd_ps(
                _mm256_load_ps(src_ptr + AVX2_FLOAT_STRIDE),
                y,
                _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE)
            );
            __m256 w2 = _mm256_fmadd_ps(
                _mm256_load_ps(src_ptr + AVX2_FLOAT_STRIDE * 2),
                y,
                _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE * 2)
            );

            _mm256_store_ps(w_ptr, w0);
            _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE, w1);
            _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE * 2, w2);
        }
        else if (r == AVX2_FLOAT_STRIDE * 2) {
            __m256 w0 = _mm256_fmadd_ps(
                _mm256_load_ps(src_ptr),
                y,
                _mm256_load_ps(w_ptr)
            );
            __m256 w1 = _mm256_fmadd_ps(
                _mm256_load_ps(src_ptr + AVX2_FLOAT_STRIDE),
                y,
                _mm256_load_ps(w_ptr + AVX2_FLOAT_STRIDE)
            );

            _mm256_store_ps(w_ptr, w0);
            _mm256_store_ps(w_ptr + AVX2_FLOAT_STRIDE, w1);
        }
        else if (r == AVX2_FLOAT_STRIDE) {
            __m256 w0 = _mm256_fmadd_ps(
                _mm256_load_ps(src_ptr),
                y,
                _mm256_load_ps(w_ptr)
            );

            _mm256_store_ps(w_ptr, w0);
        }

        w_ptr += r;
    }
}

__forceinline void kernelfma_unaligned_s(
    const unsigned int ic, const unsigned int oc,
    const float* __restrict x_ptr, const float* __restrict y_ptr, float* __restrict w_ptr, const __m256i mask) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    for (unsigned int i = 0; i < oc; i++) {
        __m256 y = _mm256_set1_ps(y_ptr[i]);

        unsigned r = ic;
        const float* src_ptr = x_ptr;

        while (r >= AVX2_FLOAT_STRIDE * 4) {
            __m256 w0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(src_ptr),
                y,
                _mm256_loadu_ps(w_ptr)
            );
            __m256 w1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE),
                y,
                _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE)
            );
            __m256 w2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE * 2),
                y,
                _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE * 2)
            );
            __m256 w3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE * 3),
                y,
                _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE * 3)
            );

            _mm256_storeu_ps(w_ptr, w0);
            _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE, w1);
            _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE * 2, w2);
            _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE * 3, w3);

            src_ptr += AVX2_FLOAT_STRIDE * 4;
            w_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
        if (r >= AVX2_FLOAT_STRIDE * 3) {
            __m256 w0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(src_ptr),
                y,
                _mm256_loadu_ps(w_ptr)
            );
            __m256 w1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE),
                y,
                _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE)
            );
            __m256 w2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE * 2),
                y,
                _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE * 2)
            );
            __m256 w3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE * 3),
                y,
                _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE * 3)
            );

            _mm256_storeu_ps(w_ptr, w0);
            _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE, w1);
            _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE * 2, w2);
            _mm256_maskstore_ps(w_ptr + AVX2_FLOAT_STRIDE * 3, mask, w3);
        }
        else if (r >= AVX2_FLOAT_STRIDE * 2) {
            __m256 w0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(src_ptr),
                y,
                _mm256_loadu_ps(w_ptr)
            );
            __m256 w1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE),
                y,
                _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE)
            );
            __m256 w2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE * 2),
                y,
                _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE * 2)
            );

            _mm256_storeu_ps(w_ptr, w0);
            _mm256_storeu_ps(w_ptr + AVX2_FLOAT_STRIDE, w1);
            _mm256_maskstore_ps(w_ptr + AVX2_FLOAT_STRIDE * 2, mask, w2);
        }
        else if (r >= AVX2_FLOAT_STRIDE) {
            __m256 w0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(src_ptr),
                y,
                _mm256_loadu_ps(w_ptr)
            );
            __m256 w1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(src_ptr + AVX2_FLOAT_STRIDE),
                y,
                _mm256_loadu_ps(w_ptr + AVX2_FLOAT_STRIDE)
            );

            _mm256_storeu_ps(w_ptr, w0);
            _mm256_maskstore_ps(w_ptr + AVX2_FLOAT_STRIDE, mask, w1);
        }
        else {
            __m256 w0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(src_ptr),
                y,
                _mm256_loadu_ps(w_ptr)
            );

            _mm256_maskstore_ps(w_ptr, mask, w0);
        }

        w_ptr += r;
    }
}