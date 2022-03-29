#pragma once
#pragma unmanaged

#include "../constants.h"
#include "../utils.h"
#include "inline_dilate_s.hpp"
#include "inline_set_s.hpp"
#include "inline_kahan_s.hpp"
#include "inline_loadstore_xn_s.hpp"

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

__forceinline void kernelfma_n1_aligned_ss(
    const uint ic, const uint oc,
    infloats x_ptr, infloats y_ptr, outfloats ws_ptr, outfloats wc_ptr) {

#ifdef _DEBUG
    if (ic != 1 || (oc & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)ws_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)wc_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256 x = _mm256_set1_ps(x_ptr[0]);
    __m256 s0, s1, s2, s3, c0, c1, c2, c3, y0, y1, y2, y3;

    uint r = oc;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_load_x4_ps(ws_ptr, s0, s1, s2, s3);
        _mm256_load_x4_ps(wc_ptr, c0, c1, c2, c3);
        _mm256_load_x4_ps(y_ptr, y0, y1, y2, y3);

        _mm256_kahanfma_ps(x, y0, s0, c0);
        _mm256_kahanfma_ps(x, y1, s1, c1);
        _mm256_kahanfma_ps(x, y2, s2, c2);
        _mm256_kahanfma_ps(x, y3, s3, c3);

        _mm256_store_x4_ps(ws_ptr, s0, s1, s2, s3);
        _mm256_store_x4_ps(wc_ptr, c0, c1, c2, c3);

        y_ptr += AVX2_FLOAT_STRIDE * 4;
        ws_ptr += AVX2_FLOAT_STRIDE * 4;
        wc_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_load_x2_ps(ws_ptr, s0, s1);
        _mm256_load_x2_ps(wc_ptr, c0, c1);
        _mm256_load_x2_ps(y_ptr, y0, y1);

        _mm256_kahanfma_ps(x, y0, s0, c0);
        _mm256_kahanfma_ps(x, y1, s1, c1);

        _mm256_store_x2_ps(ws_ptr, s0, s1);
        _mm256_store_x2_ps(wc_ptr, c0, c1);

        y_ptr += AVX2_FLOAT_STRIDE * 2;
        ws_ptr += AVX2_FLOAT_STRIDE * 2;
        wc_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_load_x1_ps(ws_ptr, s0);
        _mm256_load_x1_ps(wc_ptr, c0);
        _mm256_load_x1_ps(y_ptr, y0);

        _mm256_kahanfma_ps(x, y0, s0, c0);

        _mm256_store_x1_ps(ws_ptr, s0);
        _mm256_store_x1_ps(wc_ptr, c0);
    }
}

__forceinline void kernelfma_n1_unaligned_ss(
    const uint ic, const uint oc,
    infloats x_ptr, infloats y_ptr, outfloats ws_ptr, outfloats wc_ptr, const __m256i mask) {

#ifdef _DEBUG
    if (ic != 1 || (oc & AVX2_FLOAT_REMAIN_MASK) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256 x = _mm256_set1_ps(x_ptr[0]);
    __m256 s0, s1, s2, s3, c0, c1, c2, c3, y0, y1, y2, y3;

    uint r = oc;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_loadu_x4_ps(ws_ptr, s0, s1, s2, s3);
        _mm256_loadu_x4_ps(wc_ptr, c0, c1, c2, c3);
        _mm256_loadu_x4_ps(y_ptr, y0, y1, y2, y3);

        _mm256_kahanfma_ps(x, y0, s0, c0);
        _mm256_kahanfma_ps(x, y1, s1, c1);
        _mm256_kahanfma_ps(x, y2, s2, c2);
        _mm256_kahanfma_ps(x, y3, s3, c3);

        _mm256_storeu_x4_ps(ws_ptr, s0, s1, s2, s3);
        _mm256_storeu_x4_ps(wc_ptr, c0, c1, c2, c3);

        y_ptr += AVX2_FLOAT_STRIDE * 4;
        ws_ptr += AVX2_FLOAT_STRIDE * 4;
        wc_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_loadu_x2_ps(ws_ptr, s0, s1);
        _mm256_loadu_x2_ps(wc_ptr, c0, c1);
        _mm256_loadu_x2_ps(y_ptr, y0, y1);

        _mm256_kahanfma_ps(x, y0, s0, c0);
        _mm256_kahanfma_ps(x, y1, s1, c1);

        _mm256_storeu_x2_ps(ws_ptr, s0, s1);
        _mm256_storeu_x2_ps(wc_ptr, c0, c1);

        y_ptr += AVX2_FLOAT_STRIDE * 2;
        ws_ptr += AVX2_FLOAT_STRIDE * 2;
        wc_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_loadu_x1_ps(ws_ptr, s0);
        _mm256_loadu_x1_ps(wc_ptr, c0);
        _mm256_loadu_x1_ps(y_ptr, y0);

        _mm256_kahanfma_ps(x, y0, s0, c0);

        _mm256_storeu_x1_ps(ws_ptr, s0);
        _mm256_storeu_x1_ps(wc_ptr, c0);

        y_ptr += AVX2_FLOAT_STRIDE;
        ws_ptr += AVX2_FLOAT_STRIDE;
        wc_ptr += AVX2_FLOAT_STRIDE;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r > 0) {
        _mm256_loadu_x1_ps(ws_ptr, s0);
        _mm256_loadu_x1_ps(wc_ptr, c0);
        _mm256_loadu_x1_ps(y_ptr, y0);

        _mm256_kahanfma_ps(x, y0, s0, c0);

        _mm256_maskstore_x1_ps(ws_ptr, s0, mask);
        _mm256_maskstore_x1_ps(wc_ptr, c0, mask);
    }
}

__forceinline void kernelfma_n2_aligned_ss(
    const uint ic, const uint oc,
    infloats x_ptr, infloats y_ptr, outfloats ws_ptr, outfloats wc_ptr) {

#ifdef _DEBUG
    if (ic != 2 || (oc % (AVX2_FLOAT_STRIDE / 2)) != 0 || ((size_t)ws_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)wc_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256 x = _mm256_set2_ps(x_ptr[0], x_ptr[1]);
    __m256 s0, s1, s2, s3, c0, c1, c2, c3, y01, y23;

    uint r = oc;

    while (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_load_x4_ps(ws_ptr, s0, s1, s2, s3);
        _mm256_load_x4_ps(wc_ptr, c0, c1, c2, c3);
        _mm256_loadu_x2_ps(y_ptr, y01, y23);

        _mm256_kahanfma_ps(x, _mm256_dilate2_imm0_ps(y01), s0, c0);
        _mm256_kahanfma_ps(x, _mm256_dilate2_imm1_ps(y01), s1, c1);
        _mm256_kahanfma_ps(x, _mm256_dilate2_imm0_ps(y23), s2, c2);
        _mm256_kahanfma_ps(x, _mm256_dilate2_imm1_ps(y23), s3, c3);

        _mm256_store_x4_ps(ws_ptr, s0, s1, s2, s3);
        _mm256_store_x4_ps(wc_ptr, c0, c1, c2, c3);

        y_ptr += AVX2_FLOAT_STRIDE * 2;
        ws_ptr += AVX2_FLOAT_STRIDE * 4;
        wc_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_load_x2_ps(ws_ptr, s0, s1);
        _mm256_load_x2_ps(wc_ptr, c0, c1);
        _mm256_loadu_x1_ps(y_ptr, y01);

        _mm256_kahanfma_ps(x, _mm256_dilate2_imm0_ps(y01), s0, c0);
        _mm256_kahanfma_ps(x, _mm256_dilate2_imm1_ps(y01), s1, c1);

        _mm256_store_x2_ps(ws_ptr, s0, s1);
        _mm256_store_x2_ps(wc_ptr, c0, c1);

        y_ptr += AVX2_FLOAT_STRIDE;
        ws_ptr += AVX2_FLOAT_STRIDE * 2;
        wc_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r >= AVX2_FLOAT_STRIDE / 2) {
        _mm256_load_x1_ps(ws_ptr, s0);
        _mm256_load_x1_ps(wc_ptr, c0);
        _mm256_loadu_x1_ps(y_ptr, y01);

        _mm256_kahanfma_ps(x, _mm256_dilate2_imm0_ps(y01), s0, c0);

        _mm256_store_x1_ps(ws_ptr, s0);
        _mm256_store_x1_ps(wc_ptr, c0);
    }
}

__forceinline void kernelfma_n2_unaligned_ss(
    const uint ic, const uint oc,
    infloats x_ptr, infloats y_ptr, outfloats ws_ptr, outfloats wc_ptr, const __m256i mask) {

#ifdef _DEBUG
    if (ic != 2 || (oc % (AVX2_FLOAT_STRIDE / 2)) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256 x = _mm256_set2_ps(x_ptr[0], x_ptr[1]);
    __m256 s0, s1, s2, s3, c0, c1, c2, c3, y01, y23;

    uint r = oc;

    while (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_loadu_x4_ps(ws_ptr, s0, s1, s2, s3);
        _mm256_loadu_x4_ps(wc_ptr, c0, c1, c2, c3);
        _mm256_loadu_x2_ps(y_ptr, y01, y23);

        _mm256_kahanfma_ps(x, _mm256_dilate2_imm0_ps(y01), s0, c0);
        _mm256_kahanfma_ps(x, _mm256_dilate2_imm1_ps(y01), s1, c1);
        _mm256_kahanfma_ps(x, _mm256_dilate2_imm0_ps(y23), s2, c2);
        _mm256_kahanfma_ps(x, _mm256_dilate2_imm1_ps(y23), s3, c3);

        _mm256_storeu_x4_ps(ws_ptr, s0, s1, s2, s3);
        _mm256_storeu_x4_ps(wc_ptr, c0, c1, c2, c3);

        y_ptr += AVX2_FLOAT_STRIDE * 2;
        ws_ptr += AVX2_FLOAT_STRIDE * 4;
        wc_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_loadu_x2_ps(ws_ptr, s0, s1);
        _mm256_loadu_x2_ps(wc_ptr, c0, c1);
        _mm256_loadu_x1_ps(y_ptr, y01);

        _mm256_kahanfma_ps(x, _mm256_dilate2_imm0_ps(y01), s0, c0);
        _mm256_kahanfma_ps(x, _mm256_dilate2_imm1_ps(y01), s1, c1);

        _mm256_storeu_x2_ps(ws_ptr, s0, s1);
        _mm256_storeu_x2_ps(wc_ptr, c0, c1);

        y_ptr += AVX2_FLOAT_STRIDE;
        ws_ptr += AVX2_FLOAT_STRIDE * 2;
        wc_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r >= AVX2_FLOAT_STRIDE / 2) {
        _mm256_loadu_x1_ps(ws_ptr, s0);
        _mm256_loadu_x1_ps(wc_ptr, c0);
        _mm256_loadu_x1_ps(y_ptr, y01);

        _mm256_kahanfma_ps(x, _mm256_dilate2_imm0_ps(y01), s0, c0);

        _mm256_storeu_x1_ps(ws_ptr, s0);
        _mm256_storeu_x1_ps(wc_ptr, c0);

        y_ptr += AVX2_FLOAT_STRIDE / 2;
        ws_ptr += AVX2_FLOAT_STRIDE;
        wc_ptr += AVX2_FLOAT_STRIDE;
        r -= AVX2_FLOAT_STRIDE / 2;
    }
    if (r > 0) {
        _mm256_loadu_x1_ps(ws_ptr, s0);
        _mm256_loadu_x1_ps(wc_ptr, c0);
        _mm256_loadu_x1_ps(y_ptr, y01);

        _mm256_kahanfma_ps(x, _mm256_dilate2_imm0_ps(y01), s0, c0);

        _mm256_maskstore_x1_ps(ws_ptr, s0, mask);
        _mm256_maskstore_x1_ps(wc_ptr, c0, mask);
    }
}

__forceinline void kernelfma_n3_aligned_ss(
    const uint ic, const uint oc,
    infloats x_ptr, infloats y_ptr, outfloats ws_ptr, outfloats wc_ptr) {

#ifdef _DEBUG
    if (ic != 3 || (oc % AVX2_FLOAT_STRIDE) != 0 || ((size_t)ws_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)wc_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256i __perm_y0 = _mm256_setr_epi32(0, 0, 0, 1, 1, 1, 2, 2);
    const __m256i __perm_y1 = _mm256_setr_epi32(2, 3, 3, 3, 4, 4, 4, 5);
    const __m256i __perm_y2 = _mm256_setr_epi32(5, 5, 6, 6, 6, 7, 7, 7);

    const __m256x3 x = _mm256_set3_ps(x_ptr[0], x_ptr[1], x_ptr[2]);

    __m256 s0, s1, s2, c0, c1, c2, y;

    uint r = oc;

    while (r >= AVX2_FLOAT_STRIDE) {
        _mm256_load_x3_ps(ws_ptr, s0, s1, s2);
        _mm256_load_x3_ps(wc_ptr, c0, c1, c2);
        _mm256_load_x1_ps(y_ptr, y);

        _mm256_kahanfma_ps(x.imm0, _mm256_permutevar8x32_ps(y, __perm_y0), s0, c0);
        _mm256_kahanfma_ps(x.imm1, _mm256_permutevar8x32_ps(y, __perm_y1), s1, c1);
        _mm256_kahanfma_ps(x.imm2, _mm256_permutevar8x32_ps(y, __perm_y2), s2, c2);

        _mm256_store_x3_ps(ws_ptr, s0, s1, s2);
        _mm256_store_x3_ps(wc_ptr, c0, c1, c2);

        y_ptr += AVX2_FLOAT_STRIDE;
        ws_ptr += AVX2_FLOAT_STRIDE * 3;
        wc_ptr += AVX2_FLOAT_STRIDE * 3;
        r -= AVX2_FLOAT_STRIDE;
    }
}

__forceinline void kernelfma_n3_unaligned_ss(
    const uint ic, const uint oc,
    infloats x_ptr, infloats y_ptr, outfloats ws_ptr, outfloats wc_ptr, const __m256i mask) {

#ifdef _DEBUG
    if (ic != 3 || (oc % AVX2_FLOAT_STRIDE) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256i __perm_y0 = _mm256_setr_epi32(0, 0, 0, 1, 1, 1, 2, 2);
    const __m256i __perm_y1 = _mm256_setr_epi32(2, 3, 3, 3, 4, 4, 4, 5);
    const __m256i __perm_y2 = _mm256_setr_epi32(5, 5, 6, 6, 6, 7, 7, 7);

    const __m256x3 x = _mm256_set3_ps(x_ptr[0], x_ptr[1], x_ptr[2]);

    __m256 s0, s1, s2, c0, c1, c2, y;

    uint r = oc;

    while (r >= AVX2_FLOAT_STRIDE) {
        _mm256_loadu_x3_ps(ws_ptr, s0, s1, s2);
        _mm256_loadu_x3_ps(wc_ptr, c0, c1, c2);
        _mm256_loadu_x1_ps(y_ptr, y);

        _mm256_kahanfma_ps(x.imm0, _mm256_permutevar8x32_ps(y, __perm_y0), s0, c0);
        _mm256_kahanfma_ps(x.imm1, _mm256_permutevar8x32_ps(y, __perm_y1), s1, c1);
        _mm256_kahanfma_ps(x.imm2, _mm256_permutevar8x32_ps(y, __perm_y2), s2, c2);

        _mm256_storeu_x3_ps(ws_ptr, s0, s1, s2);
        _mm256_storeu_x3_ps(wc_ptr, c0, c1, c2);

        y_ptr += AVX2_FLOAT_STRIDE;
        ws_ptr += AVX2_FLOAT_STRIDE * 3;
        wc_ptr += AVX2_FLOAT_STRIDE * 3;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r > AVX2_FLOAT_STRIDE * 2 / 3) {
        _mm256_loadu_x3_ps(ws_ptr, s0, s1, s2);
        _mm256_loadu_x3_ps(wc_ptr, c0, c1, c2);
        _mm256_loadu_x1_ps(y_ptr, y);

        _mm256_kahanfma_ps(x.imm0, _mm256_permutevar8x32_ps(y, __perm_y0), s0, c0);
        _mm256_kahanfma_ps(x.imm1, _mm256_permutevar8x32_ps(y, __perm_y1), s1, c1);
        _mm256_kahanfma_ps(x.imm2, _mm256_permutevar8x32_ps(y, __perm_y2), s2, c2);

        _mm256_maskstore_x3_ps(ws_ptr, s0, s1, s2, mask);
        _mm256_maskstore_x3_ps(wc_ptr, c0, c1, c2, mask);
    }
    else if (r > AVX2_FLOAT_STRIDE / 3) {
        _mm256_loadu_x2_ps(ws_ptr, s0, s1);
        _mm256_loadu_x2_ps(wc_ptr, c0, c1);
        _mm256_loadu_x1_ps(y_ptr, y);

        _mm256_kahanfma_ps(x.imm0, _mm256_permutevar8x32_ps(y, __perm_y0), s0, c0);
        _mm256_kahanfma_ps(x.imm1, _mm256_permutevar8x32_ps(y, __perm_y1), s1, c1);

        _mm256_maskstore_x2_ps(ws_ptr, s0, s1, mask);
        _mm256_maskstore_x2_ps(wc_ptr, c0, c1, mask);
    }
    else if (r > 0) {
        _mm256_loadu_x1_ps(ws_ptr, s0);
        _mm256_loadu_x1_ps(wc_ptr, c0);
        _mm256_loadu_x1_ps(y_ptr, y);

        _mm256_kahanfma_ps(x.imm0, _mm256_permutevar8x32_ps(y, __perm_y0), s0, c0);

        _mm256_maskstore_x1_ps(ws_ptr, s0, mask);
        _mm256_maskstore_x1_ps(wc_ptr, c0, mask);
    }
}

__forceinline void kernelfma_n4_aligned_ss(
    const uint ic, const uint oc,
    infloats x_ptr, infloats y_ptr, outfloats ws_ptr, outfloats wc_ptr) {

#ifdef _DEBUG
    if (ic != 4 || (oc % 2) != 0 || ((size_t)ws_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)wc_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256 x = _mm256_set4_ps(x_ptr[0], x_ptr[1], x_ptr[2], x_ptr[3]);
    __m256 s0, s1, s2, s3, c0, c1, c2, c3, y;

    uint r = oc;

    while (r >= AVX2_FLOAT_STRIDE) {
        _mm256_load_x4_ps(ws_ptr, s0, s1, s2, s3);
        _mm256_load_x4_ps(wc_ptr, c0, c1, c2, c3);
        _mm256_loadu_x1_ps(y_ptr, y);

        _mm256_kahanfma_ps(x, _mm256_dilate4_imm0_ps(y), s0, c0);
        _mm256_kahanfma_ps(x, _mm256_dilate4_imm1_ps(y), s1, c1);
        _mm256_kahanfma_ps(x, _mm256_dilate4_imm2_ps(y), s2, c2);
        _mm256_kahanfma_ps(x, _mm256_dilate4_imm3_ps(y), s3, c3);

        _mm256_store_x4_ps(ws_ptr, s0, s1, s2, s3);
        _mm256_store_x4_ps(wc_ptr, c0, c1, c2, c3);

        y_ptr += AVX2_FLOAT_STRIDE;
        ws_ptr += AVX2_FLOAT_STRIDE * 4;
        wc_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r >= AVX2_FLOAT_STRIDE / 2) {
        _mm256_load_x2_ps(ws_ptr, s0, s1);
        _mm256_load_x2_ps(wc_ptr, c0, c1);
        _mm256_loadu_x1_ps(y_ptr, y);

        _mm256_kahanfma_ps(x, _mm256_dilate4_imm0_ps(y), s0, c0);
        _mm256_kahanfma_ps(x, _mm256_dilate4_imm1_ps(y), s1, c1);

        _mm256_store_x2_ps(ws_ptr, s0, s1);
        _mm256_store_x2_ps(wc_ptr, c0, c1);

        y_ptr += AVX2_FLOAT_STRIDE / 2;
        ws_ptr += AVX2_FLOAT_STRIDE * 2;
        wc_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE / 2;
    }
    if (r >= AVX2_FLOAT_STRIDE / 4) {
        _mm256_load_x1_ps(ws_ptr, s0);
        _mm256_load_x1_ps(wc_ptr, c0);
        _mm256_loadu_x1_ps(y_ptr, y);

        _mm256_kahanfma_ps(x, _mm256_dilate4_imm0_ps(y), s0, c0);

        _mm256_store_x1_ps(ws_ptr, s0);
        _mm256_store_x1_ps(wc_ptr, c0);
    }
}

__forceinline void kernelfma_n4_unaligned_ss(
    const uint ic, const uint oc,
    infloats x_ptr, infloats y_ptr, outfloats ws_ptr, outfloats wc_ptr, const __m256i mask) {

#ifdef _DEBUG
    if (ic != 4 || (oc % 2) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const __m256 x = _mm256_set4_ps(x_ptr[0], x_ptr[1], x_ptr[2], x_ptr[3]);
    __m256 s0, s1, s2, s3, c0, c1, c2, c3, y;

    uint r = oc;

    while (r >= AVX2_FLOAT_STRIDE) {
        _mm256_loadu_x4_ps(ws_ptr, s0, s1, s2, s3);
        _mm256_loadu_x4_ps(wc_ptr, c0, c1, c2, c3);
        _mm256_loadu_x1_ps(y_ptr, y);

        _mm256_kahanfma_ps(x, _mm256_dilate4_imm0_ps(y), s0, c0);
        _mm256_kahanfma_ps(x, _mm256_dilate4_imm1_ps(y), s1, c1);
        _mm256_kahanfma_ps(x, _mm256_dilate4_imm2_ps(y), s2, c2);
        _mm256_kahanfma_ps(x, _mm256_dilate4_imm3_ps(y), s3, c3);

        _mm256_storeu_x4_ps(ws_ptr, s0, s1, s2, s3);
        _mm256_storeu_x4_ps(wc_ptr, c0, c1, c2, c3);

        y_ptr += AVX2_FLOAT_STRIDE;
        ws_ptr += AVX2_FLOAT_STRIDE * 4;
        wc_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r >= AVX2_FLOAT_STRIDE / 2) {
        _mm256_loadu_x2_ps(ws_ptr, s0, s1);
        _mm256_loadu_x2_ps(wc_ptr, c0, c1);
        _mm256_loadu_x1_ps(y_ptr, y);

        _mm256_kahanfma_ps(x, _mm256_dilate4_imm0_ps(y), s0, c0);
        _mm256_kahanfma_ps(x, _mm256_dilate4_imm1_ps(y), s1, c1);

        _mm256_storeu_x2_ps(ws_ptr, s0, s1);
        _mm256_storeu_x2_ps(wc_ptr, c0, c1);

        y_ptr += AVX2_FLOAT_STRIDE / 2;
        ws_ptr += AVX2_FLOAT_STRIDE * 2;
        wc_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE / 2;
    }
    if (r >= AVX2_FLOAT_STRIDE / 4) {
        _mm256_loadu_x1_ps(ws_ptr, s0);
        _mm256_loadu_x1_ps(wc_ptr, c0);
        _mm256_loadu_x1_ps(y_ptr, y);

        _mm256_kahanfma_ps(x, _mm256_dilate4_imm0_ps(y), s0, c0);

        _mm256_storeu_x1_ps(ws_ptr, s0);
        _mm256_storeu_x1_ps(wc_ptr, c0);

        y_ptr += AVX2_FLOAT_STRIDE / 4;
        ws_ptr += AVX2_FLOAT_STRIDE;
        wc_ptr += AVX2_FLOAT_STRIDE;
        r -= AVX2_FLOAT_STRIDE / 4;
    }
    if (r > 0) {
        _mm256_loadu_x1_ps(ws_ptr, s0);
        _mm256_loadu_x1_ps(wc_ptr, c0);
        _mm256_loadu_x1_ps(y_ptr, y);

        _mm256_kahanfma_ps(x, _mm256_dilate4_imm0_ps(y), s0, c0);

        _mm256_maskstore_x1_ps(ws_ptr, s0, mask);
        _mm256_maskstore_x1_ps(wc_ptr, c0, mask);
    }
}

__forceinline void kernelfma_n32x_ss(
    const uint ic, const uint oc,
    infloats x_ptr, infloats y_ptr, outfloats ws_ptr, outfloats wc_ptr) {

#ifdef _DEBUG
    if ((ic % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0
        || ((size_t)ws_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)wc_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    for (uint i = 0; i < oc; i++) {
        const __m256 y = _mm256_set1_ps(y_ptr[i]);
        __m256 s0, s1, s2, s3, c0, c1, c2, c3, x0, x1, x2, x3;

        uint r = ic;
        const float* xc_ptr = x_ptr;

        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_load_x4_ps(ws_ptr, s0, s1, s2, s3);
            _mm256_load_x4_ps(wc_ptr, c0, c1, c2, c3);
            _mm256_load_x4_ps(xc_ptr, x0, x1, x2, x3);

            _mm256_kahanfma_ps(x0, y, s0, c0);
            _mm256_kahanfma_ps(x1, y, s1, c1);
            _mm256_kahanfma_ps(x2, y, s2, c2);
            _mm256_kahanfma_ps(x3, y, s3, c3);

            _mm256_store_x4_ps(ws_ptr, s0, s1, s2, s3);
            _mm256_store_x4_ps(wc_ptr, c0, c1, c2, c3);

            xc_ptr += AVX2_FLOAT_STRIDE * 4;
            ws_ptr += AVX2_FLOAT_STRIDE * 4;
            wc_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
    }
}

__forceinline void kernelfma_aligned_ss(
    const uint ic, const uint oc,
    infloats x_ptr, infloats y_ptr, outfloats ws_ptr, outfloats wc_ptr) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0
        || ((size_t)ws_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)wc_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    for (uint i = 0; i < oc; i++) {
        const __m256 y = _mm256_set1_ps(y_ptr[i]);
        __m256 s0, s1, s2, s3, c0, c1, c2, c3, x0, x1, x2, x3;

        uint r = ic;
        const float* xc_ptr = x_ptr;

        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_load_x4_ps(ws_ptr, s0, s1, s2, s3);
            _mm256_load_x4_ps(wc_ptr, c0, c1, c2, c3);
            _mm256_load_x4_ps(xc_ptr, x0, x1, x2, x3);

            _mm256_kahanfma_ps(x0, y, s0, c0);
            _mm256_kahanfma_ps(x1, y, s1, c1);
            _mm256_kahanfma_ps(x2, y, s2, c2);
            _mm256_kahanfma_ps(x3, y, s3, c3);

            _mm256_store_x4_ps(ws_ptr, s0, s1, s2, s3);
            _mm256_store_x4_ps(wc_ptr, c0, c1, c2, c3);

            xc_ptr += AVX2_FLOAT_STRIDE * 4;
            ws_ptr += AVX2_FLOAT_STRIDE * 4;
            wc_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
        if (r >= AVX2_FLOAT_STRIDE * 3) {
            _mm256_load_x3_ps(ws_ptr, s0, s1, s2);
            _mm256_load_x3_ps(wc_ptr, c0, c1, c2);
            _mm256_load_x3_ps(xc_ptr, x0, x1, x2);

            _mm256_kahanfma_ps(x0, y, s0, c0);
            _mm256_kahanfma_ps(x1, y, s1, c1);
            _mm256_kahanfma_ps(x2, y, s2, c2);

            _mm256_store_x3_ps(ws_ptr, s0, s1, s2);
            _mm256_store_x3_ps(wc_ptr, c0, c1, c2);
        }
        else if (r >= AVX2_FLOAT_STRIDE * 2) {
            _mm256_load_x2_ps(ws_ptr, s0, s1);
            _mm256_load_x2_ps(wc_ptr, c0, c1);
            _mm256_load_x2_ps(xc_ptr, x0, x1);

            _mm256_kahanfma_ps(x0, y, s0, c0);
            _mm256_kahanfma_ps(x1, y, s1, c1);

            _mm256_store_x2_ps(ws_ptr, s0, s1);
            _mm256_store_x2_ps(wc_ptr, c0, c1);
        }
        else if (r >= AVX2_FLOAT_STRIDE) {
            _mm256_load_x1_ps(ws_ptr, s0);
            _mm256_load_x1_ps(wc_ptr, c0);
            _mm256_load_x1_ps(xc_ptr, x0);

            _mm256_kahanfma_ps(x0, y, s0, c0);

            _mm256_store_x1_ps(ws_ptr, s0);
            _mm256_store_x1_ps(wc_ptr, c0);
        }

        ws_ptr += r;
        wc_ptr += r;
    }
}

__forceinline void kernelfma_unaligned_ss(
    const uint ic, const uint oc,
    infloats x_ptr, infloats y_ptr, outfloats ws_ptr, outfloats wc_ptr, const __m256i mask) {

#ifdef _DEBUG
    if ((ic & AVX2_FLOAT_REMAIN_MASK) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    for (uint i = 0; i < oc; i++) {
        const __m256 y = _mm256_set1_ps(y_ptr[i]);
        __m256 s0, s1, s2, s3, c0, c1, c2, c3, x0, x1, x2, x3;

        uint r = ic;
        const float* xc_ptr = x_ptr;

        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_loadu_x4_ps(ws_ptr, s0, s1, s2, s3);
            _mm256_loadu_x4_ps(wc_ptr, c0, c1, c2, c3);
            _mm256_loadu_x4_ps(xc_ptr, x0, x1, x2, x3);

            _mm256_kahanfma_ps(x0, y, s0, c0);
            _mm256_kahanfma_ps(x1, y, s1, c1);
            _mm256_kahanfma_ps(x2, y, s2, c2);
            _mm256_kahanfma_ps(x3, y, s3, c3);

            _mm256_storeu_x4_ps(ws_ptr, s0, s1, s2, s3);
            _mm256_storeu_x4_ps(wc_ptr, c0, c1, c2, c3);

            xc_ptr += AVX2_FLOAT_STRIDE * 4;
            ws_ptr += AVX2_FLOAT_STRIDE * 4;
            wc_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
        if (r >= AVX2_FLOAT_STRIDE * 3) {
            _mm256_loadu_x4_ps(ws_ptr, s0, s1, s2, s3);
            _mm256_loadu_x4_ps(wc_ptr, c0, c1, c2, c3);
            _mm256_loadu_x4_ps(xc_ptr, x0, x1, x2, x3);

            _mm256_kahanfma_ps(x0, y, s0, c0);
            _mm256_kahanfma_ps(x1, y, s1, c1);
            _mm256_kahanfma_ps(x2, y, s2, c2);
            _mm256_kahanfma_ps(x3, y, s3, c3);

            _mm256_maskstore_x4_ps(ws_ptr, s0, s1, s2, s3, mask);
            _mm256_maskstore_x4_ps(wc_ptr, c0, c1, c2, c3, mask);
        }
        else if (r >= AVX2_FLOAT_STRIDE * 2) {
            _mm256_loadu_x3_ps(ws_ptr, s0, s1, s2);
            _mm256_loadu_x3_ps(wc_ptr, c0, c1, c2);
            _mm256_loadu_x3_ps(xc_ptr, x0, x1, x2);

            _mm256_kahanfma_ps(x0, y, s0, c0);
            _mm256_kahanfma_ps(x1, y, s1, c1);
            _mm256_kahanfma_ps(x2, y, s2, c2);

            _mm256_maskstore_x3_ps(ws_ptr, s0, s1, s2, mask);
            _mm256_maskstore_x3_ps(wc_ptr, c0, c1, c2, mask);
        }
        else if (r >= AVX2_FLOAT_STRIDE) {
            _mm256_loadu_x2_ps(ws_ptr, s0, s1);
            _mm256_loadu_x2_ps(wc_ptr, c0, c1);
            _mm256_loadu_x2_ps(xc_ptr, x0, x1);

            _mm256_kahanfma_ps(x0, y, s0, c0);
            _mm256_kahanfma_ps(x1, y, s1, c1);

            _mm256_maskstore_x2_ps(ws_ptr, s0, s1, mask);
            _mm256_maskstore_x2_ps(wc_ptr, c0, c1, mask);
        }
        else {
            _mm256_loadu_x1_ps(ws_ptr, s0);
            _mm256_loadu_x1_ps(wc_ptr, c0);
            _mm256_loadu_x1_ps(xc_ptr, x0);

            _mm256_kahanfma_ps(x0, y, s0, c0);

            _mm256_maskstore_x1_ps(ws_ptr, s0, mask);
            _mm256_maskstore_x1_ps(wc_ptr, c0, mask);
        }

        ws_ptr += r;
        wc_ptr += r;
    }
}