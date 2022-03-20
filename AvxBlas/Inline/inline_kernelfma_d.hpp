#pragma once
#pragma unmanaged

#include "../constants.h"
#include "../utils.h"
#include "inline_dilate_d.hpp"
#include "inline_set_d.hpp"

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

__forceinline void kernelfma_n1_aligned_d(
    const unsigned int ic, const unsigned int oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles w_ptr) {

#ifdef _DEBUG
    if (ic != 1 || (oc & AVX2_DOUBLE_REMAIN_MASK) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256d x = _mm256_set1_pd(x_ptr[0]);

    unsigned r = oc;
    const double* src_ptr = y_ptr;

    while (r >= AVX2_DOUBLE_STRIDE * 4) {
        __m256d w0 = _mm256_fmadd_pd(
            x,
            _mm256_load_pd(src_ptr),
            _mm256_load_pd(w_ptr)
        );
        __m256d w1 = _mm256_fmadd_pd(
            x,
            _mm256_load_pd(src_ptr + AVX2_DOUBLE_STRIDE),
            _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE)
        );
        __m256d w2 = _mm256_fmadd_pd(
            x,
            _mm256_load_pd(src_ptr + AVX2_DOUBLE_STRIDE * 2),
            _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2)
        );
        __m256d w3 = _mm256_fmadd_pd(
            x,
            _mm256_load_pd(src_ptr + AVX2_DOUBLE_STRIDE * 3),
            _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE * 3)
        );

        _mm256_store_pd(w_ptr, w0);
        _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE, w1);
        _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2, w2);
        _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE * 3, w3);

        src_ptr += AVX2_DOUBLE_STRIDE * 4;
        w_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 4;
    }
    if (r >= AVX2_DOUBLE_STRIDE * 3) {
        __m256d w0 = _mm256_fmadd_pd(
            x,
            _mm256_load_pd(src_ptr),
            _mm256_load_pd(w_ptr)
        );
        __m256d w1 = _mm256_fmadd_pd(
            x,
            _mm256_load_pd(src_ptr + AVX2_DOUBLE_STRIDE),
            _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE)
        );
        __m256d w2 = _mm256_fmadd_pd(
            x,
            _mm256_load_pd(src_ptr + AVX2_DOUBLE_STRIDE * 2),
            _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2)
        );

        _mm256_store_pd(w_ptr, w0);
        _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE, w1);
        _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2, w2);
    }
    else if (r >= AVX2_DOUBLE_STRIDE * 2) {
        __m256d w0 = _mm256_fmadd_pd(
            x,
            _mm256_load_pd(src_ptr),
            _mm256_load_pd(w_ptr)
        );
        __m256d w1 = _mm256_fmadd_pd(
            x,
            _mm256_load_pd(src_ptr + AVX2_DOUBLE_STRIDE),
            _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE)
        );

        _mm256_store_pd(w_ptr, w0);
        _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE, w1);
    }
    else if (r >= AVX2_DOUBLE_STRIDE) {
        __m256d w0 = _mm256_fmadd_pd(
            x,
            _mm256_load_pd(src_ptr),
            _mm256_load_pd(w_ptr)
        );

        _mm256_store_pd(w_ptr, w0);
    }
}

__forceinline void kernelfma_n1_unaligned_d(
    const unsigned int ic, const unsigned int oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles w_ptr, const __m256i mask) {

#ifdef _DEBUG
    if (ic != 1 || (oc & AVX2_DOUBLE_REMAIN_MASK) == 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256d x = _mm256_set1_pd(x_ptr[0]);

    unsigned r = oc;
    const double* src_ptr = y_ptr;

    while (r >= AVX2_DOUBLE_STRIDE * 4) {
        __m256d w0 = _mm256_fmadd_pd(
            x,
            _mm256_loadu_pd(src_ptr),
            _mm256_loadu_pd(w_ptr)
        );
        __m256d w1 = _mm256_fmadd_pd(
            x,
            _mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE),
            _mm256_loadu_pd(w_ptr + AVX2_DOUBLE_STRIDE)
        );
        __m256d w2 = _mm256_fmadd_pd(
            x,
            _mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE * 2),
            _mm256_loadu_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2)
        );
        __m256d w3 = _mm256_fmadd_pd(
            x,
            _mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE * 3),
            _mm256_loadu_pd(w_ptr + AVX2_DOUBLE_STRIDE * 3)
        );

        _mm256_storeu_pd(w_ptr, w0);
        _mm256_storeu_pd(w_ptr + AVX2_DOUBLE_STRIDE, w1);
        _mm256_storeu_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2, w2);
        _mm256_storeu_pd(w_ptr + AVX2_DOUBLE_STRIDE * 3, w3);

        src_ptr += AVX2_DOUBLE_STRIDE * 4;
        w_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 4;
    }
    if (r >= AVX2_DOUBLE_STRIDE * 2) {
        __m256d w0 = _mm256_fmadd_pd(
            x,
            _mm256_loadu_pd(src_ptr),
            _mm256_loadu_pd(w_ptr)
        );
        __m256d w1 = _mm256_fmadd_pd(
            x,
            _mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE),
            _mm256_loadu_pd(w_ptr + AVX2_DOUBLE_STRIDE)
        );

        _mm256_storeu_pd(w_ptr, w0);
        _mm256_storeu_pd(w_ptr + AVX2_DOUBLE_STRIDE, w1);

        src_ptr += AVX2_DOUBLE_STRIDE * 2;
        w_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE * 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE) {
        __m256d w0 = _mm256_fmadd_pd(
            x,
            _mm256_loadu_pd(src_ptr),
            _mm256_loadu_pd(w_ptr)
        );

        _mm256_storeu_pd(w_ptr, w0);

        src_ptr += AVX2_DOUBLE_STRIDE;
        w_ptr += AVX2_DOUBLE_STRIDE;
        r -= AVX2_DOUBLE_STRIDE;
    }
    if (r > 0) {
        __m256d w0 = _mm256_fmadd_pd(
            x,
            _mm256_loadu_pd(src_ptr),
            _mm256_loadu_pd(w_ptr)
        );

        _mm256_maskstore_pd(w_ptr, mask, w0);
    }
}

__forceinline void kernelfma_n2_aligned_d(
    const unsigned int ic, const unsigned int oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles w_ptr) {

#ifdef _DEBUG
    if (ic != 2 || (oc % (AVX2_DOUBLE_STRIDE / 2)) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256d x = _mm256_set2_pd(x_ptr[0], x_ptr[1]);

    unsigned r = oc;
    const double* src_ptr = y_ptr;

    while (r >= AVX2_DOUBLE_STRIDE * 2) {
        __m256d w0 = _mm256_fmadd_pd(
            x,
            _mm256_dilate2_imm0_pd(_mm256_loadu_pd(src_ptr)),
            _mm256_load_pd(w_ptr)
        );
        __m256d w1 = _mm256_fmadd_pd(
            x,
            _mm256_dilate2_imm0_pd(_mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE / 2)),
            _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE)
        );
        __m256d w2 = _mm256_fmadd_pd(
            x,
            _mm256_dilate2_imm0_pd(_mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE)),
            _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2)
        );
        __m256d w3 = _mm256_fmadd_pd(
            x,
            _mm256_dilate2_imm0_pd(_mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE * 3 / 2)),
            _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE * 3)
        );

        _mm256_store_pd(w_ptr, w0);
        _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE, w1);
        _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2, w2);
        _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE * 3, w3);

        src_ptr += AVX2_DOUBLE_STRIDE * 2;
        w_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE * 3 / 2) {
        __m256d w0 = _mm256_fmadd_pd(
            x,
            _mm256_dilate2_imm0_pd(_mm256_loadu_pd(src_ptr)),
            _mm256_load_pd(w_ptr)
        );
        __m256d w1 = _mm256_fmadd_pd(
            x,
            _mm256_dilate2_imm0_pd(_mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE / 2)),
            _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE)
        );
        __m256d w2 = _mm256_fmadd_pd(
            x,
            _mm256_dilate2_imm0_pd(_mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE)),
            _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2)
        );

        _mm256_store_pd(w_ptr, w0);
        _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE, w1);
        _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2, w2);
    }
    else if (r >= AVX2_DOUBLE_STRIDE) {
        __m256d w0 = _mm256_fmadd_pd(
            x,
            _mm256_dilate2_imm0_pd(_mm256_loadu_pd(src_ptr)),
            _mm256_load_pd(w_ptr)
        );
        __m256d w1 = _mm256_fmadd_pd(
            x,
            _mm256_dilate2_imm0_pd(_mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE / 2)),
            _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE)
        );

        _mm256_store_pd(w_ptr, w0);
        _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE, w1);
    }
    else if (r >= AVX2_DOUBLE_STRIDE / 2) {
        __m256d w0 = _mm256_fmadd_pd(
            x,
            _mm256_dilate2_imm0_pd(_mm256_loadu_pd(src_ptr)),
            _mm256_load_pd(w_ptr)
        );

        _mm256_store_pd(w_ptr, w0);
    }
}

__forceinline void kernelfma_n2_unaligned_d(
    const unsigned int ic, const unsigned int oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles w_ptr, const __m256i mask) {

#ifdef _DEBUG
    if (ic != 2 || (oc % (AVX2_DOUBLE_STRIDE / 2)) == 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256d x = _mm256_set2_pd(x_ptr[0], x_ptr[1]);

    unsigned r = oc;
    const double* src_ptr = y_ptr;

    while (r >= AVX2_DOUBLE_STRIDE * 2) {
        __m256d w0 = _mm256_fmadd_pd(
            x,
            _mm256_dilate2_imm0_pd(_mm256_loadu_pd(src_ptr)),
            _mm256_loadu_pd(w_ptr)
        );
        __m256d w1 = _mm256_fmadd_pd(
            x,
            _mm256_dilate2_imm0_pd(_mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE / 2)),
            _mm256_loadu_pd(w_ptr + AVX2_DOUBLE_STRIDE)
        );
        __m256d w2 = _mm256_fmadd_pd(
            x,
            _mm256_dilate2_imm0_pd(_mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE)),
            _mm256_loadu_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2)
        );
        __m256d w3 = _mm256_fmadd_pd(
            x,
            _mm256_dilate2_imm0_pd(_mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE * 3 / 2)),
            _mm256_loadu_pd(w_ptr + AVX2_DOUBLE_STRIDE * 3)
        );

        _mm256_storeu_pd(w_ptr, w0);
        _mm256_storeu_pd(w_ptr + AVX2_DOUBLE_STRIDE, w1);
        _mm256_storeu_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2, w2);
        _mm256_storeu_pd(w_ptr + AVX2_DOUBLE_STRIDE * 3, w3);

        src_ptr += AVX2_DOUBLE_STRIDE * 2;
        w_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE) {
        __m256d w0 = _mm256_fmadd_pd(
            x,
            _mm256_dilate2_imm0_pd(_mm256_loadu_pd(src_ptr)),
            _mm256_loadu_pd(w_ptr)
        );
        __m256d w1 = _mm256_fmadd_pd(
            x,
            _mm256_dilate2_imm0_pd(_mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE / 2)),
            _mm256_loadu_pd(w_ptr + AVX2_DOUBLE_STRIDE)
        );

        _mm256_storeu_pd(w_ptr, w0);
        _mm256_storeu_pd(w_ptr + AVX2_DOUBLE_STRIDE, w1);

        src_ptr += AVX2_DOUBLE_STRIDE;
        w_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE;
    }
    if (r >= AVX2_DOUBLE_STRIDE / 2) {
        __m256d w0 = _mm256_fmadd_pd(
            x,
            _mm256_dilate2_imm0_pd(_mm256_loadu_pd(src_ptr)),
            _mm256_loadu_pd(w_ptr)
        );

        _mm256_storeu_pd(w_ptr, w0);

        src_ptr += AVX2_DOUBLE_STRIDE / 2;
        w_ptr += AVX2_DOUBLE_STRIDE;
        r -= AVX2_DOUBLE_STRIDE / 2;
    }
    if (r > 0) {
        __m256d w0 = _mm256_fmadd_pd(
            x,
            _mm256_dilate2_imm0_pd(_mm256_loadu_pd(src_ptr)),
            _mm256_loadu_pd(w_ptr)
        );

        _mm256_maskstore_pd(w_ptr, mask, w0);
    }
}

__forceinline void kernelfma_n3_aligned_d(
    const unsigned int ic, const unsigned int oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles w_ptr) {

#ifdef _DEBUG
    if (ic != 3 || (oc % 4) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256d x = _mm256_set3_pd(x_ptr[0], x_ptr[1], x_ptr[2]);

    unsigned r = oc;
    const double* src_ptr = y_ptr;

    while (r >= 4) {
        __m256d w0 = _mm256_fmadd_pd(
            x,
            _mm256_set1_pd(src_ptr[0]),
            _mm256_loadu_pd(w_ptr)
        );
        __m256d w1 = _mm256_permute4x64_pd(_mm256_fmadd_pd(
            x,
            _mm256_set1_pd(src_ptr[1]),
            _mm256_loadu_pd(w_ptr + 3)
        ), _MM_PERM_ADCB);
        __m256d w2 = _mm256_permute4x64_pd(_mm256_fmadd_pd(
            x,
            _mm256_set1_pd(src_ptr[2]),
            _mm256_loadu_pd(w_ptr + 6)
        ), _MM_PERM_BADC);
        __m256d w3 = _mm256_permute4x64_pd(_mm256_fmadd_pd(
            x,
            _mm256_set1_pd(src_ptr[3]),
            _mm256_loadu_pd(w_ptr + 9)
        ), _MM_PERM_CBAD);

        __m256d w01 = _mm256_blend_pd(w0, w1, 0b1000);
        __m256d w12 = _mm256_blend_pd(w1, w2, 0b1100);
        __m256d w23 = _mm256_blend_pd(w2, w3, 0b1110);

        _mm256_store_pd(w_ptr, w01);
        _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE, w12);
        _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2, w23);

        src_ptr += 4;
        w_ptr += AVX2_DOUBLE_STRIDE * 3;
        r -= 4;
    }
}

__forceinline void kernelfma_n3_unaligned_d(
    const unsigned int ic, const unsigned int oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles w_ptr) {

#ifdef _DEBUG
    if (ic != 3 || (oc % 4) == 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256d x = _mm256_set3_pd(x_ptr[0], x_ptr[1], x_ptr[2]);

    unsigned r = oc;
    const double* src_ptr = y_ptr;

    const __m256i __mask3 = _mm256_setr_epi32(~0u, ~0u, ~0u, ~0u, ~0u, ~0u, 0, 0);

    while (r >= 4) {
        __m256d w0 = _mm256_fmadd_pd(
            x,
            _mm256_set1_pd(src_ptr[0]),
            _mm256_loadu_pd(w_ptr)
        );
        __m256d w1 = _mm256_permute4x64_pd(_mm256_fmadd_pd(
            x,
            _mm256_set1_pd(src_ptr[1]),
            _mm256_loadu_pd(w_ptr + 3)
        ), _MM_PERM_ADCB);
        __m256d w2 = _mm256_permute4x64_pd(_mm256_fmadd_pd(
            x,
            _mm256_set1_pd(src_ptr[2]),
            _mm256_loadu_pd(w_ptr + 6)
        ), _MM_PERM_BADC);
        __m256d w3 = _mm256_permute4x64_pd(_mm256_fmadd_pd(
            x,
            _mm256_set1_pd(src_ptr[3]),
            _mm256_loadu_pd(w_ptr + 9)
        ), _MM_PERM_CBAD);

        __m256d w01 = _mm256_blend_pd(w0, w1, 0b1000);
        __m256d w12 = _mm256_blend_pd(w1, w2, 0b1100);
        __m256d w23 = _mm256_blend_pd(w2, w3, 0b1110);

        _mm256_storeu_pd(w_ptr, w01);
        _mm256_storeu_pd(w_ptr + AVX2_DOUBLE_STRIDE, w12);
        _mm256_storeu_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2, w23);

        src_ptr += 4;
        w_ptr += AVX2_DOUBLE_STRIDE * 3;
        r -= 4;
    }
    if (r >= 2) {
        __m256d w0 = _mm256_fmadd_pd(
            x,
            _mm256_set1_pd(src_ptr[0]),
            _mm256_loadu_pd(w_ptr)
        );
        __m256d w1 = _mm256_fmadd_pd(
            x,
            _mm256_set1_pd(src_ptr[1]),
            _mm256_loadu_pd(w_ptr + 3)
        );

        _mm256_maskstore_pd(w_ptr, __mask3, w0);
        _mm256_maskstore_pd(w_ptr + 3, __mask3, w1);

        src_ptr += 2;
        w_ptr += 6;
        r -= 2;
    }
    if (r > 0) {
        __m256d w0 = _mm256_fmadd_pd(
            x,
            _mm256_set1_pd(src_ptr[0]),
            _mm256_loadu_pd(w_ptr)
        );

        _mm256_maskstore_pd(w_ptr, __mask3, w0);
    }
}

__forceinline void kernelfma_n4_d(
    const unsigned int ic, const unsigned int oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles w_ptr) {

#ifdef _DEBUG
    if (ic != 4 || (oc % 2) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    __m256d x = _mm256_load_pd(x_ptr);

    unsigned r = oc;
    const double* src_ptr = y_ptr;

    while (r >= 4) {
        __m256d w0 = _mm256_fmadd_pd(
            x,
            _mm256_set1_pd(src_ptr[0]),
            _mm256_load_pd(w_ptr)
        );
        __m256d w1 = _mm256_fmadd_pd(
            x,
            _mm256_set1_pd(src_ptr[1]),
            _mm256_load_pd(w_ptr + 4)
        );
        __m256d w2 = _mm256_fmadd_pd(
            x,
            _mm256_set1_pd(src_ptr[2]),
            _mm256_load_pd(w_ptr + 8)
        );
        __m256d w3 = _mm256_fmadd_pd(
            x,
            _mm256_set1_pd(src_ptr[3]),
            _mm256_load_pd(w_ptr + 12)
        );

        _mm256_store_pd(w_ptr, w0);
        _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE, w1);
        _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2, w2);
        _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE * 3, w3);

        src_ptr += 8;
        w_ptr += 32;
        r -= 8;
    }
    if (r >= 3) {
        __m256d w0 = _mm256_fmadd_pd(
            x,
            _mm256_set1_pd(src_ptr[0]),
            _mm256_load_pd(w_ptr)
        );
        __m256d w1 = _mm256_fmadd_pd(
            x,
            _mm256_set1_pd(src_ptr[1]),
            _mm256_load_pd(w_ptr + 4)
        );
        __m256d w2 = _mm256_fmadd_pd(
            x,
            _mm256_set1_pd(src_ptr[2]),
            _mm256_load_pd(w_ptr + 8)
        );

        _mm256_store_pd(w_ptr, w0);
        _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE, w1);
        _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2, w2);
    }
    else if (r >= 2) {
        __m256d w0 = _mm256_fmadd_pd(
            x,
            _mm256_set1_pd(src_ptr[0]),
            _mm256_load_pd(w_ptr)
        );
        __m256d w1 = _mm256_fmadd_pd(
            x,
            _mm256_set1_pd(src_ptr[1]),
            _mm256_load_pd(w_ptr + 4)
        );

        _mm256_store_pd(w_ptr, w0);
        _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE, w1);
    }
    else if (r >= 1) {
        __m256d w0 = _mm256_fmadd_pd(
            x,
            _mm256_set1_pd(src_ptr[0]),
            _mm256_load_pd(w_ptr)
        );

        _mm256_store_pd(w_ptr, w0);
    }
}

__forceinline void kernelfma_n16x_d(
    const unsigned int ic, const unsigned int oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles w_ptr) {

#ifdef _DEBUG
    if ((ic % (AVX2_DOUBLE_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    for (unsigned int i = 0; i < oc; i++) {
        __m256d y = _mm256_set1_pd(y_ptr[i]);

        unsigned r = ic;
        const double* src_ptr = x_ptr;

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            __m256d w0 = _mm256_fmadd_pd(
                _mm256_load_pd(src_ptr),
                y,
                _mm256_load_pd(w_ptr)
            );
            __m256d w1 = _mm256_fmadd_pd(
                _mm256_load_pd(src_ptr + AVX2_DOUBLE_STRIDE),
                y,
                _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE)
            );
            __m256d w2 = _mm256_fmadd_pd(
                _mm256_load_pd(src_ptr + AVX2_DOUBLE_STRIDE * 2),
                y,
                _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2)
            );
            __m256d w3 = _mm256_fmadd_pd(
                _mm256_load_pd(src_ptr + AVX2_DOUBLE_STRIDE * 3),
                y,
                _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE * 3)
            );

            _mm256_store_pd(w_ptr, w0);
            _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE, w1);
            _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2, w2);
            _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE * 3, w3);

            src_ptr += AVX2_DOUBLE_STRIDE * 4;
            w_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
    }
}

__forceinline void kernelfma_aligned_d(
    const unsigned int ic, const unsigned int oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles w_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_DOUBLE_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)w_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    for (unsigned int i = 0; i < oc; i++) {
        __m256d y = _mm256_set1_pd(y_ptr[i]);

        unsigned r = ic;
        const double* src_ptr = x_ptr;

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            __m256d w0 = _mm256_fmadd_pd(
                _mm256_load_pd(src_ptr),
                y,
                _mm256_load_pd(w_ptr)
            );
            __m256d w1 = _mm256_fmadd_pd(
                _mm256_load_pd(src_ptr + AVX2_DOUBLE_STRIDE),
                y,
                _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE)
            );
            __m256d w2 = _mm256_fmadd_pd(
                _mm256_load_pd(src_ptr + AVX2_DOUBLE_STRIDE * 2),
                y,
                _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2)
            );
            __m256d w3 = _mm256_fmadd_pd(
                _mm256_load_pd(src_ptr + AVX2_DOUBLE_STRIDE * 3),
                y,
                _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE * 3)
            );

            _mm256_store_pd(w_ptr, w0);
            _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE, w1);
            _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2, w2);
            _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE * 3, w3);

            src_ptr += AVX2_DOUBLE_STRIDE * 4;
            w_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        if (r == AVX2_DOUBLE_STRIDE * 3) {
            __m256d w0 = _mm256_fmadd_pd(
                _mm256_load_pd(src_ptr),
                y,
                _mm256_load_pd(w_ptr)
            );
            __m256d w1 = _mm256_fmadd_pd(
                _mm256_load_pd(src_ptr + AVX2_DOUBLE_STRIDE),
                y,
                _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE)
            );
            __m256d w2 = _mm256_fmadd_pd(
                _mm256_load_pd(src_ptr + AVX2_DOUBLE_STRIDE * 2),
                y,
                _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2)
            );

            _mm256_store_pd(w_ptr, w0);
            _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE, w1);
            _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2, w2);
        }
        else if (r == AVX2_DOUBLE_STRIDE * 2) {
            __m256d w0 = _mm256_fmadd_pd(
                _mm256_load_pd(src_ptr),
                y,
                _mm256_load_pd(w_ptr)
            );
            __m256d w1 = _mm256_fmadd_pd(
                _mm256_load_pd(src_ptr + AVX2_DOUBLE_STRIDE),
                y,
                _mm256_load_pd(w_ptr + AVX2_DOUBLE_STRIDE)
            );

            _mm256_store_pd(w_ptr, w0);
            _mm256_store_pd(w_ptr + AVX2_DOUBLE_STRIDE, w1);
        }
        else if (r == AVX2_DOUBLE_STRIDE) {
            __m256d w0 = _mm256_fmadd_pd(
                _mm256_load_pd(src_ptr),
                y,
                _mm256_load_pd(w_ptr)
            );

            _mm256_store_pd(w_ptr, w0);
        }

        w_ptr += r;
    }
}

__forceinline void kernelfma_unaligned_d(
    const unsigned int ic, const unsigned int oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles w_ptr, const __m256i mask) {

#ifdef _DEBUG
    if ((ic & AVX2_DOUBLE_REMAIN_MASK) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    for (unsigned int i = 0; i < oc; i++) {
        __m256d y = _mm256_set1_pd(y_ptr[i]);

        unsigned r = ic;
        const double* src_ptr = x_ptr;

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            __m256d w0 = _mm256_fmadd_pd(
                _mm256_loadu_pd(src_ptr),
                y,
                _mm256_loadu_pd(w_ptr)
            );
            __m256d w1 = _mm256_fmadd_pd(
                _mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE),
                y,
                _mm256_loadu_pd(w_ptr + AVX2_DOUBLE_STRIDE)
            );
            __m256d w2 = _mm256_fmadd_pd(
                _mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE * 2),
                y,
                _mm256_loadu_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2)
            );
            __m256d w3 = _mm256_fmadd_pd(
                _mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE * 3),
                y,
                _mm256_loadu_pd(w_ptr + AVX2_DOUBLE_STRIDE * 3)
            );

            _mm256_storeu_pd(w_ptr, w0);
            _mm256_storeu_pd(w_ptr + AVX2_DOUBLE_STRIDE, w1);
            _mm256_storeu_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2, w2);
            _mm256_storeu_pd(w_ptr + AVX2_DOUBLE_STRIDE * 3, w3);

            src_ptr += AVX2_DOUBLE_STRIDE * 4;
            w_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        if (r >= AVX2_DOUBLE_STRIDE * 3) {
            __m256d w0 = _mm256_fmadd_pd(
                _mm256_loadu_pd(src_ptr),
                y,
                _mm256_loadu_pd(w_ptr)
            );
            __m256d w1 = _mm256_fmadd_pd(
                _mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE),
                y,
                _mm256_loadu_pd(w_ptr + AVX2_DOUBLE_STRIDE)
            );
            __m256d w2 = _mm256_fmadd_pd(
                _mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE * 2),
                y,
                _mm256_loadu_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2)
            );
            __m256d w3 = _mm256_fmadd_pd(
                _mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE * 3),
                y,
                _mm256_loadu_pd(w_ptr + AVX2_DOUBLE_STRIDE * 3)
            );

            _mm256_storeu_pd(w_ptr, w0);
            _mm256_storeu_pd(w_ptr + AVX2_DOUBLE_STRIDE, w1);
            _mm256_storeu_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2, w2);
            _mm256_maskstore_pd(w_ptr + AVX2_DOUBLE_STRIDE * 3, mask, w3);
        }
        else if (r >= AVX2_DOUBLE_STRIDE * 2) {
            __m256d w0 = _mm256_fmadd_pd(
                _mm256_loadu_pd(src_ptr),
                y,
                _mm256_loadu_pd(w_ptr)
            );
            __m256d w1 = _mm256_fmadd_pd(
                _mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE),
                y,
                _mm256_loadu_pd(w_ptr + AVX2_DOUBLE_STRIDE)
            );
            __m256d w2 = _mm256_fmadd_pd(
                _mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE * 2),
                y,
                _mm256_loadu_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2)
            );

            _mm256_storeu_pd(w_ptr, w0);
            _mm256_storeu_pd(w_ptr + AVX2_DOUBLE_STRIDE, w1);
            _mm256_maskstore_pd(w_ptr + AVX2_DOUBLE_STRIDE * 2, mask, w2);
        }
        else if (r >= AVX2_DOUBLE_STRIDE) {
            __m256d w0 = _mm256_fmadd_pd(
                _mm256_loadu_pd(src_ptr),
                y,
                _mm256_loadu_pd(w_ptr)
            );
            __m256d w1 = _mm256_fmadd_pd(
                _mm256_loadu_pd(src_ptr + AVX2_DOUBLE_STRIDE),
                y,
                _mm256_loadu_pd(w_ptr + AVX2_DOUBLE_STRIDE)
            );

            _mm256_storeu_pd(w_ptr, w0);
            _mm256_maskstore_pd(w_ptr + AVX2_DOUBLE_STRIDE, mask, w1);
        }
        else {
            __m256d w0 = _mm256_fmadd_pd(
                _mm256_loadu_pd(src_ptr),
                y,
                _mm256_loadu_pd(w_ptr)
            );

            _mm256_maskstore_pd(w_ptr, mask, w0);
        }

        w_ptr += r;
    }
}