#pragma once
#pragma unmanaged

#include "../constants.h"
#include "../utils.h"
#include "inline_dilate_d.hpp"
#include "inline_set_d.hpp"
#include "inline_kahan_d.hpp"
#include "inline_loadstore_xn_d.hpp"

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

__forceinline void kernelfma_n1_aligned_dd(
    const uint ic, const uint oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles ws_ptr, outdoubles wc_ptr) {

#ifdef _DEBUG
    if (ic != 1 || (oc & AVX2_DOUBLE_REMAIN_MASK) != 0 || ((size_t)ws_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)wc_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256d x = _mm256_set1_pd(x_ptr[0]);
    __m256d s0, s1, s2, s3, c0, c1, c2, c3, y0, y1, y2, y3;

    unsigned r = oc;
    const double* src_ptr = y_ptr;

    while (r >= AVX2_DOUBLE_STRIDE * 4) {
        _mm256_load_x4_pd(ws_ptr, s0, s1, s2, s3);
        _mm256_load_x4_pd(wc_ptr, c0, c1, c2, c3);
        _mm256_load_x4_pd(src_ptr, y0, y1, y2, y3);

        _mm256_kahanfma_pd(x, y0, s0, c0);
        _mm256_kahanfma_pd(x, y1, s1, c1);
        _mm256_kahanfma_pd(x, y2, s2, c2);
        _mm256_kahanfma_pd(x, y3, s3, c3);

        _mm256_store_x4_pd(ws_ptr, s0, s1, s2, s3);
        _mm256_store_x4_pd(wc_ptr, c0, c1, c2, c3);

        src_ptr += AVX2_DOUBLE_STRIDE * 4;
        ws_ptr += AVX2_DOUBLE_STRIDE * 4;
        wc_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 4;
    }
    if (r >= AVX2_DOUBLE_STRIDE * 2) {
        _mm256_load_x2_pd(ws_ptr, s0, s1);
        _mm256_load_x2_pd(wc_ptr, c0, c1);
        _mm256_load_x2_pd(src_ptr, y0, y1);

        _mm256_kahanfma_pd(x, y0, s0, c0);
        _mm256_kahanfma_pd(x, y1, s1, c1);

        _mm256_store_x2_pd(ws_ptr, s0, s1);
        _mm256_store_x2_pd(wc_ptr, c0, c1);

        src_ptr += AVX2_DOUBLE_STRIDE * 2;
        ws_ptr += AVX2_DOUBLE_STRIDE * 2;
        wc_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE * 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE) {
        _mm256_load_x1_pd(ws_ptr, s0);
        _mm256_load_x1_pd(wc_ptr, c0);
        _mm256_load_x1_pd(src_ptr, y0);

        _mm256_kahanfma_pd(x, y0, s0, c0);

        _mm256_store_x1_pd(ws_ptr, s0);
        _mm256_store_x1_pd(wc_ptr, c0);
    }
}

__forceinline void kernelfma_n1_unaligned_dd(
    const uint ic, const uint oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles ws_ptr, outdoubles wc_ptr, const __m256i mask) {

#ifdef _DEBUG
    if (ic != 1 || (oc & AVX2_DOUBLE_REMAIN_MASK) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256d x = _mm256_set1_pd(x_ptr[0]);
    __m256d s0, s1, s2, s3, c0, c1, c2, c3, y0, y1, y2, y3;

    unsigned r = oc;
    const double* src_ptr = y_ptr;

    while (r >= AVX2_DOUBLE_STRIDE * 4) {
        _mm256_loadu_x4_pd(ws_ptr, s0, s1, s2, s3);
        _mm256_loadu_x4_pd(wc_ptr, c0, c1, c2, c3);
        _mm256_loadu_x4_pd(src_ptr, y0, y1, y2, y3);

        _mm256_kahanfma_pd(x, y0, s0, c0);
        _mm256_kahanfma_pd(x, y1, s1, c1);
        _mm256_kahanfma_pd(x, y2, s2, c2);
        _mm256_kahanfma_pd(x, y3, s3, c3);

        _mm256_storeu_x4_pd(ws_ptr, s0, s1, s2, s3);
        _mm256_storeu_x4_pd(wc_ptr, c0, c1, c2, c3);

        src_ptr += AVX2_DOUBLE_STRIDE * 4;
        ws_ptr += AVX2_DOUBLE_STRIDE * 4;
        wc_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 4;
    }
    if (r >= AVX2_DOUBLE_STRIDE * 2) {
        _mm256_loadu_x2_pd(ws_ptr, s0, s1);
        _mm256_loadu_x2_pd(wc_ptr, c0, c1);
        _mm256_loadu_x2_pd(src_ptr, y0, y1);

        _mm256_kahanfma_pd(x, y0, s0, c0);
        _mm256_kahanfma_pd(x, y1, s1, c1);

        _mm256_storeu_x2_pd(ws_ptr, s0, s1);
        _mm256_storeu_x2_pd(wc_ptr, c0, c1);

        src_ptr += AVX2_DOUBLE_STRIDE * 2;
        ws_ptr += AVX2_DOUBLE_STRIDE * 2;
        wc_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE * 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE) {
        _mm256_loadu_x1_pd(ws_ptr, s0);
        _mm256_loadu_x1_pd(wc_ptr, c0);
        _mm256_loadu_x1_pd(src_ptr, y0);

        _mm256_kahanfma_pd(x, y0, s0, c0);

        _mm256_storeu_x1_pd(ws_ptr, s0);
        _mm256_storeu_x1_pd(wc_ptr, c0);

        src_ptr += AVX2_DOUBLE_STRIDE;
        ws_ptr += AVX2_DOUBLE_STRIDE;
        wc_ptr += AVX2_DOUBLE_STRIDE;
        r -= AVX2_DOUBLE_STRIDE;
    }
    if (r > 0) {
        _mm256_loadu_x1_pd(ws_ptr, s0);
        _mm256_loadu_x1_pd(wc_ptr, c0);
        _mm256_loadu_x1_pd(src_ptr, y0);

        _mm256_kahanfma_pd(x, y0, s0, c0);

        _mm256_maskstore_x1_pd(ws_ptr, s0, mask);
        _mm256_maskstore_x1_pd(wc_ptr, c0, mask);
    }
}

__forceinline void kernelfma_n2_aligned_dd(
    const uint ic, const uint oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles ws_ptr, outdoubles wc_ptr) {

#ifdef _DEBUG
    if (ic != 2 || (oc % (AVX2_DOUBLE_STRIDE / 2)) != 0 || ((size_t)ws_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)wc_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256d x = _mm256_set2_pd(x_ptr[0], x_ptr[1]);
    __m256d s0, s1, s2, s3, c0, c1, c2, c3, y01, y23;

    unsigned r = oc;
    const double* src_ptr = y_ptr;

    while (r >= AVX2_DOUBLE_STRIDE * 2) {
        _mm256_load_x4_pd(ws_ptr, s0, s1, s2, s3);
        _mm256_load_x4_pd(wc_ptr, c0, c1, c2, c3);
        _mm256_loadu_x2_pd(src_ptr, y01, y23);

        _mm256_kahanfma_pd(x, _mm256_dilate2_imm0_pd(y01), s0, c0);
        _mm256_kahanfma_pd(x, _mm256_dilate2_imm1_pd(y01), s1, c1);
        _mm256_kahanfma_pd(x, _mm256_dilate2_imm0_pd(y23), s2, c2);
        _mm256_kahanfma_pd(x, _mm256_dilate2_imm1_pd(y23), s3, c3);

        _mm256_store_x4_pd(ws_ptr, s0, s1, s2, s3);
        _mm256_store_x4_pd(wc_ptr, c0, c1, c2, c3);

        src_ptr += AVX2_DOUBLE_STRIDE * 2;
        ws_ptr += AVX2_DOUBLE_STRIDE * 4;
        wc_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE) {
        _mm256_load_x2_pd(ws_ptr, s0, s1);
        _mm256_load_x2_pd(wc_ptr, c0, c1);
        _mm256_loadu_x1_pd(src_ptr, y01);

        _mm256_kahanfma_pd(x, _mm256_dilate2_imm0_pd(y01), s0, c0);
        _mm256_kahanfma_pd(x, _mm256_dilate2_imm1_pd(y01), s1, c1);

        _mm256_store_x2_pd(ws_ptr, s0, s1);
        _mm256_store_x2_pd(wc_ptr, c0, c1);

        src_ptr += AVX2_DOUBLE_STRIDE;
        ws_ptr += AVX2_DOUBLE_STRIDE * 2;
        wc_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE;
    }
    if (r >= AVX2_DOUBLE_STRIDE / 2) {
        _mm256_load_x1_pd(ws_ptr, s0);
        _mm256_load_x1_pd(wc_ptr, c0);
        _mm256_loadu_x1_pd(src_ptr, y01);

        _mm256_kahanfma_pd(x, _mm256_dilate2_imm0_pd(y01), s0, c0);

        _mm256_store_x1_pd(ws_ptr, s0);
        _mm256_store_x1_pd(wc_ptr, c0);
    }
}

__forceinline void kernelfma_n2_unaligned_dd(
    const uint ic, const uint oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles ws_ptr, outdoubles wc_ptr, const __m256i mask) {

#ifdef _DEBUG
    if (ic != 2 || (oc % (AVX2_DOUBLE_STRIDE / 2)) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256d x = _mm256_set2_pd(x_ptr[0], x_ptr[1]);
    __m256d s0, s1, s2, s3, c0, c1, c2, c3, y01, y23;

    unsigned r = oc;
    const double* src_ptr = y_ptr;

    while (r >= AVX2_DOUBLE_STRIDE * 2) {
        _mm256_loadu_x4_pd(ws_ptr, s0, s1, s2, s3);
        _mm256_loadu_x4_pd(wc_ptr, c0, c1, c2, c3);
        _mm256_loadu_x2_pd(src_ptr, y01, y23);

        _mm256_kahanfma_pd(x, _mm256_dilate2_imm0_pd(y01), s0, c0);
        _mm256_kahanfma_pd(x, _mm256_dilate2_imm1_pd(y01), s1, c1);
        _mm256_kahanfma_pd(x, _mm256_dilate2_imm0_pd(y23), s2, c2);
        _mm256_kahanfma_pd(x, _mm256_dilate2_imm1_pd(y23), s3, c3);

        _mm256_storeu_x4_pd(ws_ptr, s0, s1, s2, s3);
        _mm256_storeu_x4_pd(wc_ptr, c0, c1, c2, c3);

        src_ptr += AVX2_DOUBLE_STRIDE * 2;
        ws_ptr += AVX2_DOUBLE_STRIDE * 4;
        wc_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE) {
        _mm256_loadu_x2_pd(ws_ptr, s0, s1);
        _mm256_loadu_x2_pd(wc_ptr, c0, c1);
        _mm256_loadu_x1_pd(src_ptr, y01);

        _mm256_kahanfma_pd(x, _mm256_dilate2_imm0_pd(y01), s0, c0);
        _mm256_kahanfma_pd(x, _mm256_dilate2_imm1_pd(y01), s1, c1);

        _mm256_storeu_x2_pd(ws_ptr, s0, s1);
        _mm256_storeu_x2_pd(wc_ptr, c0, c1);

        src_ptr += AVX2_DOUBLE_STRIDE;
        ws_ptr += AVX2_DOUBLE_STRIDE * 2;
        wc_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE;
    }
    if (r >= AVX2_DOUBLE_STRIDE / 2) {
        _mm256_loadu_x1_pd(ws_ptr, s0);
        _mm256_loadu_x1_pd(wc_ptr, c0);
        _mm256_loadu_x1_pd(src_ptr, y01);

        _mm256_kahanfma_pd(x, _mm256_dilate2_imm0_pd(y01), s0, c0);

        _mm256_storeu_x1_pd(ws_ptr, s0);
        _mm256_storeu_x1_pd(wc_ptr, c0);

        src_ptr += AVX2_DOUBLE_STRIDE / 2;
        ws_ptr += AVX2_DOUBLE_STRIDE;
        wc_ptr += AVX2_DOUBLE_STRIDE;
        r -= AVX2_DOUBLE_STRIDE / 2;
    }
    if (r > 0) {
        _mm256_loadu_x1_pd(ws_ptr, s0);
        _mm256_loadu_x1_pd(wc_ptr, c0);
        _mm256_loadu_x1_pd(src_ptr, y01);

        _mm256_kahanfma_pd(x, _mm256_dilate2_imm0_pd(y01), s0, c0);

        _mm256_maskstore_x1_pd(ws_ptr, s0, mask);
        _mm256_maskstore_x1_pd(wc_ptr, c0, mask);
    }
}

__forceinline void kernelfma_n3_aligned_dd(
    const uint ic, const uint oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles ws_ptr, outdoubles wc_ptr) {

#ifdef _DEBUG
    if (ic != 3 || (oc % AVX2_DOUBLE_STRIDE) != 0 || ((size_t)ws_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)wc_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256dx3 x = _mm256_set3_pd(x_ptr[0], x_ptr[1], x_ptr[2]);
    
    __m256d s0, s1, s2, c0, c1, c2, y;

    unsigned r = oc;
    const double* src_ptr = y_ptr;

    while (r >= AVX2_DOUBLE_STRIDE) {
        _mm256_load_x3_pd(ws_ptr, s0, s1, s2);
        _mm256_load_x3_pd(wc_ptr, c0, c1, c2);
        _mm256_load_x1_pd(src_ptr, y);

        _mm256_kahanfma_pd(x.imm0, _mm256_permute4x64_pd(y, _MM_PERM_BAAA), s0, c0);
        _mm256_kahanfma_pd(x.imm1, _mm256_permute4x64_pd(y, _MM_PERM_CCBB), s1, c1);
        _mm256_kahanfma_pd(x.imm2, _mm256_permute4x64_pd(y, _MM_PERM_DDDC), s2, c2);

        _mm256_store_x3_pd(ws_ptr, s0, s1, s2);
        _mm256_store_x3_pd(wc_ptr, c0, c1, c2);

        src_ptr += AVX2_DOUBLE_STRIDE;
        ws_ptr += AVX2_DOUBLE_STRIDE * 3;
        wc_ptr += AVX2_DOUBLE_STRIDE * 3;
        r -= AVX2_DOUBLE_STRIDE;
    }
}

__forceinline void kernelfma_n3_unaligned_dd(
    const uint ic, const uint oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles ws_ptr, outdoubles wc_ptr, const __m256i mask) {

#ifdef _DEBUG
    if (ic != 3 || (oc % AVX2_DOUBLE_STRIDE) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256i __mask3 = _mm256_setr_epi32(~0u, ~0u, ~0u, ~0u, ~0u, ~0u, 0, 0);

    const __m256dx3 x = _mm256_set3_pd(x_ptr[0], x_ptr[1], x_ptr[2]);

    __m256d s0, s1, s2, c0, c1, c2, y;

    unsigned r = oc;
    const double* src_ptr = y_ptr;

    while (r >= AVX2_DOUBLE_STRIDE) {
        _mm256_loadu_x3_pd(ws_ptr, s0, s1, s2);
        _mm256_loadu_x3_pd(wc_ptr, c0, c1, c2);
        _mm256_loadu_x1_pd(src_ptr, y);

        _mm256_kahanfma_pd(x.imm0, _mm256_permute4x64_pd(y, _MM_PERM_BAAA), s0, c0);
        _mm256_kahanfma_pd(x.imm1, _mm256_permute4x64_pd(y, _MM_PERM_CCBB), s1, c1);
        _mm256_kahanfma_pd(x.imm2, _mm256_permute4x64_pd(y, _MM_PERM_DDDC), s2, c2);

        _mm256_storeu_x3_pd(ws_ptr, s0, s1, s2);
        _mm256_storeu_x3_pd(wc_ptr, c0, c1, c2);

        src_ptr += AVX2_DOUBLE_STRIDE;
        ws_ptr += AVX2_DOUBLE_STRIDE * 3;
        wc_ptr += AVX2_DOUBLE_STRIDE * 3;
        r -= AVX2_DOUBLE_STRIDE;
    }
    if (r >= 3) { // 3 * r >= AVX2_DOUBLE_STRIDE * 2
        _mm256_loadu_x3_pd(ws_ptr, s0, s1, s2);
        _mm256_loadu_x3_pd(wc_ptr, c0, c1, c2);
        _mm256_loadu_x1_pd(src_ptr, y);

        _mm256_kahanfma_pd(x.imm0, _mm256_permute4x64_pd(y, _MM_PERM_BAAA), s0, c0);
        _mm256_kahanfma_pd(x.imm1, _mm256_permute4x64_pd(y, _MM_PERM_CCBB), s1, c1);
        _mm256_kahanfma_pd(x.imm2, _mm256_permute4x64_pd(y, _MM_PERM_DDDC), s2, c2);

        _mm256_maskstore_x3_pd(ws_ptr, s0, s1, s2, mask);
        _mm256_maskstore_x3_pd(wc_ptr, c0, c1, c2, mask);
    }
    else if (r >= 2) { // 3 * r >= AVX2_DOUBLE_STRIDE
        _mm256_loadu_x2_pd(ws_ptr, s0, s1);
        _mm256_loadu_x2_pd(wc_ptr, c0, c1);
        _mm256_loadu_x1_pd(src_ptr, y);

        _mm256_kahanfma_pd(x.imm0, _mm256_permute4x64_pd(y, _MM_PERM_BAAA), s0, c0);
        _mm256_kahanfma_pd(x.imm1, _mm256_permute4x64_pd(y, _MM_PERM_CCBB), s1, c1);

        _mm256_maskstore_x2_pd(ws_ptr, s0, s1, mask);
        _mm256_maskstore_x2_pd(wc_ptr, c0, c1, mask);
    }
    else if (r >= 1) {
        _mm256_loadu_x1_pd(ws_ptr, s0);
        _mm256_loadu_x1_pd(wc_ptr, c0);
        _mm256_loadu_x1_pd(src_ptr, y);

        _mm256_kahanfma_pd(x.imm0, _mm256_permute4x64_pd(y, _MM_PERM_BAAA), s0, c0);

        _mm256_maskstore_x1_pd(ws_ptr, s0, mask);
        _mm256_maskstore_x1_pd(wc_ptr, c0, mask);
    }
}

__forceinline void kernelfma_n4_dd(
    const uint ic, const uint oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles ws_ptr, outdoubles wc_ptr) {

#ifdef _DEBUG
    if (ic != 4 || ((size_t)ws_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)wc_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256d x = _mm256_setr_pd(x_ptr[0], x_ptr[1], x_ptr[2], x_ptr[3]);
    __m256d s0, s1, s2, s3, c0, c1, c2, c3, y;

    unsigned r = oc;
    const double* src_ptr = y_ptr;

    while (r >= AVX2_DOUBLE_STRIDE) {
        _mm256_load_x4_pd(ws_ptr, s0, s1, s2, s3);
        _mm256_load_x4_pd(wc_ptr, c0, c1, c2, c3);
        _mm256_loadu_x1_pd(src_ptr, y);

        _mm256_kahanfma_pd(x, _mm256_dilate4_imm0_pd(y), s0, c0);
        _mm256_kahanfma_pd(x, _mm256_dilate4_imm1_pd(y), s1, c1);
        _mm256_kahanfma_pd(x, _mm256_dilate4_imm2_pd(y), s2, c2);
        _mm256_kahanfma_pd(x, _mm256_dilate4_imm3_pd(y), s3, c3);

        _mm256_store_x4_pd(ws_ptr, s0, s1, s2, s3);
        _mm256_store_x4_pd(wc_ptr, c0, c1, c2, c3);

        src_ptr += AVX2_DOUBLE_STRIDE;
        ws_ptr += AVX2_DOUBLE_STRIDE * 4;
        wc_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE;
    }
    if (r >= AVX2_DOUBLE_STRIDE / 2) {
        _mm256_load_x2_pd(ws_ptr, s0, s1);
        _mm256_load_x2_pd(wc_ptr, c0, c1);
        _mm256_loadu_x1_pd(src_ptr, y);

        _mm256_kahanfma_pd(x, _mm256_dilate4_imm0_pd(y), s0, c0);
        _mm256_kahanfma_pd(x, _mm256_dilate4_imm1_pd(y), s1, c1);

        _mm256_store_x2_pd(ws_ptr, s0, s1);
        _mm256_store_x2_pd(wc_ptr, c0, c1);

        src_ptr += AVX2_DOUBLE_STRIDE / 2;
        ws_ptr += AVX2_DOUBLE_STRIDE * 2;
        wc_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE / 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE / 4) {
        _mm256_load_x1_pd(ws_ptr, s0);
        _mm256_load_x1_pd(wc_ptr, c0);
        _mm256_loadu_x1_pd(src_ptr, y);

        _mm256_kahanfma_pd(x, _mm256_dilate4_imm0_pd(y), s0, c0);

        _mm256_store_x1_pd(ws_ptr, s0);
        _mm256_store_x1_pd(wc_ptr, c0);
    }
}

__forceinline void kernelfma_n16x_dd(
    const uint ic, const uint oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles ws_ptr, outdoubles wc_ptr) {

#ifdef _DEBUG
    if ((ic % (AVX2_DOUBLE_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0
        || ((size_t)ws_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)wc_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    for (uint i = 0; i < oc; i++) {
        const __m256d y = _mm256_set1_pd(y_ptr[i]);
        __m256d s0, s1, s2, s3, c0, c1, c2, c3, x0, x1, x2, x3;

        unsigned r = ic;
        const double* src_ptr = x_ptr;

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_load_x4_pd(ws_ptr, s0, s1, s2, s3);
            _mm256_load_x4_pd(wc_ptr, c0, c1, c2, c3);
            _mm256_load_x4_pd(src_ptr, x0, x1, x2, x3);

            _mm256_kahanfma_pd(x0, y, s0, c0);
            _mm256_kahanfma_pd(x1, y, s1, c1);
            _mm256_kahanfma_pd(x2, y, s2, c2);
            _mm256_kahanfma_pd(x3, y, s3, c3);

            _mm256_store_x4_pd(ws_ptr, s0, s1, s2, s3);
            _mm256_store_x4_pd(wc_ptr, c0, c1, c2, c3);

            src_ptr += AVX2_DOUBLE_STRIDE * 4;
            ws_ptr += AVX2_DOUBLE_STRIDE * 4;
            wc_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
    }
}

__forceinline void kernelfma_aligned_dd(
    const uint ic, const uint oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles ws_ptr, outdoubles wc_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_DOUBLE_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0
        || ((size_t)ws_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)wc_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    for (uint i = 0; i < oc; i++) {
        const __m256d y = _mm256_set1_pd(y_ptr[i]);
        __m256d s0, s1, s2, s3, c0, c1, c2, c3, x0, x1, x2, x3;

        unsigned r = ic;
        const double* src_ptr = x_ptr;

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_load_x4_pd(ws_ptr, s0, s1, s2, s3);
            _mm256_load_x4_pd(wc_ptr, c0, c1, c2, c3);
            _mm256_load_x4_pd(src_ptr, x0, x1, x2, x3);

            _mm256_kahanfma_pd(x0, y, s0, c0);
            _mm256_kahanfma_pd(x1, y, s1, c1);
            _mm256_kahanfma_pd(x2, y, s2, c2);
            _mm256_kahanfma_pd(x3, y, s3, c3);

            _mm256_store_x4_pd(ws_ptr, s0, s1, s2, s3);
            _mm256_store_x4_pd(wc_ptr, c0, c1, c2, c3);

            src_ptr += AVX2_DOUBLE_STRIDE * 4;
            ws_ptr += AVX2_DOUBLE_STRIDE * 4;
            wc_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        if (r >= AVX2_DOUBLE_STRIDE * 3) {
            _mm256_load_x3_pd(ws_ptr, s0, s1, s2);
            _mm256_load_x3_pd(wc_ptr, c0, c1, c2);
            _mm256_load_x3_pd(src_ptr, x0, x1, x2);

            _mm256_kahanfma_pd(x0, y, s0, c0);
            _mm256_kahanfma_pd(x1, y, s1, c1);
            _mm256_kahanfma_pd(x2, y, s2, c2);

            _mm256_store_x3_pd(ws_ptr, s0, s1, s2);
            _mm256_store_x3_pd(wc_ptr, c0, c1, c2);
        }
        else if (r >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_load_x2_pd(ws_ptr, s0, s1);
            _mm256_load_x2_pd(wc_ptr, c0, c1);
            _mm256_load_x2_pd(src_ptr, x0, x1);

            _mm256_kahanfma_pd(x0, y, s0, c0);
            _mm256_kahanfma_pd(x1, y, s1, c1);

            _mm256_store_x2_pd(ws_ptr, s0, s1);
            _mm256_store_x2_pd(wc_ptr, c0, c1);
        }
        else if (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_load_x1_pd(ws_ptr, s0);
            _mm256_load_x1_pd(wc_ptr, c0);
            _mm256_load_x1_pd(src_ptr, x0);

            _mm256_kahanfma_pd(x0, y, s0, c0);

            _mm256_store_x1_pd(ws_ptr, s0);
            _mm256_store_x1_pd(wc_ptr, c0);
        }

        ws_ptr += r;
        wc_ptr += r;
    }
}

__forceinline void kernelfma_unaligned_dd(
    const uint ic, const uint oc,
    indoubles x_ptr, indoubles y_ptr, outdoubles ws_ptr, outdoubles wc_ptr, const __m256i mask) {

#ifdef _DEBUG
    if ((ic & AVX2_DOUBLE_REMAIN_MASK) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    for (uint i = 0; i < oc; i++) {
        const __m256d y = _mm256_set1_pd(y_ptr[i]);
        __m256d s0, s1, s2, s3, c0, c1, c2, c3, x0, x1, x2, x3;

        unsigned r = ic;
        const double* src_ptr = x_ptr;

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_loadu_x4_pd(ws_ptr, s0, s1, s2, s3);
            _mm256_loadu_x4_pd(wc_ptr, c0, c1, c2, c3);
            _mm256_loadu_x4_pd(src_ptr, x0, x1, x2, x3);

            _mm256_kahanfma_pd(x0, y, s0, c0);
            _mm256_kahanfma_pd(x1, y, s1, c1);
            _mm256_kahanfma_pd(x2, y, s2, c2);
            _mm256_kahanfma_pd(x3, y, s3, c3);

            _mm256_storeu_x4_pd(ws_ptr, s0, s1, s2, s3);
            _mm256_storeu_x4_pd(wc_ptr, c0, c1, c2, c3);

            src_ptr += AVX2_DOUBLE_STRIDE * 4;
            ws_ptr += AVX2_DOUBLE_STRIDE * 4;
            wc_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        if (r >= AVX2_DOUBLE_STRIDE * 3) {
            _mm256_loadu_x4_pd(ws_ptr, s0, s1, s2, s3);
            _mm256_loadu_x4_pd(wc_ptr, c0, c1, c2, c3);
            _mm256_loadu_x4_pd(src_ptr, x0, x1, x2, x3);

            _mm256_kahanfma_pd(x0, y, s0, c0);
            _mm256_kahanfma_pd(x1, y, s1, c1);
            _mm256_kahanfma_pd(x2, y, s2, c2);
            _mm256_kahanfma_pd(x3, y, s3, c3);

            _mm256_maskstore_x4_pd(ws_ptr, s0, s1, s2, s3, mask);
            _mm256_maskstore_x4_pd(wc_ptr, c0, c1, c2, c3, mask);
        }
        else if (r >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_loadu_x3_pd(ws_ptr, s0, s1, s2);
            _mm256_loadu_x3_pd(wc_ptr, c0, c1, c2);
            _mm256_loadu_x3_pd(src_ptr, x0, x1, x2);

            _mm256_kahanfma_pd(x0, y, s0, c0);
            _mm256_kahanfma_pd(x1, y, s1, c1);
            _mm256_kahanfma_pd(x2, y, s2, c2);

            _mm256_maskstore_x3_pd(ws_ptr, s0, s1, s2, mask);
            _mm256_maskstore_x3_pd(wc_ptr, c0, c1, c2, mask);
        }
        else if (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_loadu_x2_pd(ws_ptr, s0, s1);
            _mm256_loadu_x2_pd(wc_ptr, c0, c1);
            _mm256_loadu_x2_pd(src_ptr, x0, x1);

            _mm256_kahanfma_pd(x0, y, s0, c0);
            _mm256_kahanfma_pd(x1, y, s1, c1);

            _mm256_maskstore_x2_pd(ws_ptr, s0, s1, mask);
            _mm256_maskstore_x2_pd(wc_ptr, c0, c1, mask);
        }
        else {
            _mm256_loadu_x1_pd(ws_ptr, s0);
            _mm256_loadu_x1_pd(wc_ptr, c0);
            _mm256_loadu_x1_pd(src_ptr, x0);

            _mm256_kahanfma_pd(x0, y, s0, c0);

            _mm256_maskstore_x1_pd(ws_ptr, s0, mask);
            _mm256_maskstore_x1_pd(wc_ptr, c0, mask);
        }

        ws_ptr += r;
        wc_ptr += r;
    }
}