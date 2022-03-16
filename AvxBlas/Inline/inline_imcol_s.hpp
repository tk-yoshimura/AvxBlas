#pragma once
#pragma unmanaged

#include "../constants.h"
#include "../utils.h"

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

__forceinline void imcol1d_padnone_aligned_s(
    const unsigned int c, 
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const float* im_ptr, float* col_ptr) {

#ifdef _DEBUG
    if (((c * kw) & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)im_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)col_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    im_ptr += c * ix;

    unsigned int r = c * kw;
        
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

__forceinline float imcol1d_padnone_unaligned_s(
    const unsigned int c, 
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const float* im_ptr, float* col_ptr, const __m256i mask) {

#ifdef _DEBUG
    if (((c * kw) & AVX2_FLOAT_REMAIN_MASK) == 0 || ((size_t)im_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)col_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    im_ptr += c * ix;

    unsigned int r = c * kw;

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

__forceinline void imcol2d_padnone_aligned_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const unsigned int kh, const unsigned int ih, const unsigned int iy,
    const float* im_ptr, float* col_ptr) {

    im_ptr += c * iw * iy;

    for (int ky = 0; ky < kh; ky++) {
        imcol1d_padnone_aligned_s(c, kw, iw, ix, im_ptr, col_ptr);

        im_ptr += c * iw;
        col_ptr += c * kw;
    }
}

__forceinline void imcol2d_padnone_unaligned_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const unsigned int kh, const unsigned int ih, const unsigned int iy,
    const float* im_ptr, float* col_ptr, const __m256i mask) {

    im_ptr += c * iw * iy;

    for (int ky = 0; ky < kh; ky++) {
        imcol1d_padnone_unaligned_s(c, kw, iw, ix, im_ptr, col_ptr, mask);

        im_ptr += c * iw;
        col_ptr += c * kw;
    }
}

__forceinline void imcol3d_padnone_aligned_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const unsigned int kh, const unsigned int ih, const unsigned int iy,
    const unsigned int kd, const unsigned int id, const unsigned int iz,
    const float* im_ptr, float* col_ptr) {

    im_ptr += c * iw * ih * iz;

    for (int kz = 0; kz < kd; kz++) {
        imcol2d_padnone_aligned_s(c, kw, iw, ix, kh, ih, iy, im_ptr, col_ptr);

        im_ptr += c * iw * ih;
        col_ptr += c * kw * kh;
    }
}

__forceinline void imcol3d_padnone_unaligned_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const unsigned int kh, const unsigned int ih, const unsigned int iy,
    const unsigned int kd, const unsigned int id, const unsigned int iz,
    const float* im_ptr, float* col_ptr, const __m256i mask) {

    im_ptr += c * iw * ih * iz;

    for (int kz = 0; kz < kd; kz++) {
        imcol2d_padnone_unaligned_s(c, kw, iw, ix, kh, ih, iy, im_ptr, col_ptr, mask);

        im_ptr += c * iw * ih;
        col_ptr += c * kw * kh;
    }
}