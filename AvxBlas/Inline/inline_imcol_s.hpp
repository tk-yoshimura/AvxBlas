#pragma once
#pragma unmanaged

#include "../constants.h"
#include "../utils.h"

#include "inline_numeric.hpp"
#include "inline_copy_s.hpp"
#include "inline_zeroset_s.hpp"

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

#pragma region padnone

__forceinline void imcol1d_padnone_n32x_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const float* __restrict im_ptr, float* __restrict col_ptr) {

#ifdef _DEBUG
    if (((c * kw) % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)im_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)col_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    copy_n32x_s(c * kw, im_ptr + c * ix, col_ptr);
}

__forceinline void imcol1d_padnone_aligned_s(
    const unsigned int c, 
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const float* __restrict im_ptr, float* __restrict col_ptr) {

#ifdef _DEBUG
    if (((c * kw) & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)im_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)col_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG
            
    copy_aligned_s(c * kw, im_ptr + c * ix, col_ptr);
}

__forceinline void imcol1d_padnone_unaligned_s(
    const unsigned int c, 
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const float* __restrict im_ptr, float* __restrict col_ptr, const __m256i mask) {

#ifdef _DEBUG
    if (((c * kw) & AVX2_FLOAT_REMAIN_MASK) == 0 || ((size_t)im_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)col_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    copy_unaligned_s(c * kw, im_ptr + c * ix, col_ptr, mask);
}

__forceinline void imcol2d_padnone_n32x_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const unsigned int kh, const unsigned int ih, const unsigned int iy,
    const float* __restrict im_ptr, float* __restrict col_ptr) {

    im_ptr += c * iw * iy;

    for (unsigned int ky = 0; ky < kh; ky++) {
        imcol1d_padnone_n32x_s(c, kw, iw, ix, im_ptr, col_ptr);

        im_ptr += c * iw;
        col_ptr += c * kw;
    }
}

__forceinline void imcol2d_padnone_aligned_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const unsigned int kh, const unsigned int ih, const unsigned int iy,
    const float* __restrict im_ptr, float* __restrict col_ptr) {

    im_ptr += c * iw * iy;

    for (unsigned int ky = 0; ky < kh; ky++) {
        imcol1d_padnone_aligned_s(c, kw, iw, ix, im_ptr, col_ptr);

        im_ptr += c * iw;
        col_ptr += c * kw;
    }
}

__forceinline void imcol2d_padnone_unaligned_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const unsigned int kh, const unsigned int ih, const unsigned int iy,
    const float* __restrict im_ptr, float* __restrict col_ptr, const __m256i mask) {

    im_ptr += c * iw * iy;

    for (unsigned int ky = 0; ky < kh; ky++) {
        imcol1d_padnone_unaligned_s(c, kw, iw, ix, im_ptr, col_ptr, mask);

        im_ptr += c * iw;
        col_ptr += c * kw;
    }
}

__forceinline void imcol3d_padnone_n32x_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const unsigned int kh, const unsigned int ih, const unsigned int iy,
    const unsigned int kd, const unsigned int id, const unsigned int iz,
    const float* __restrict im_ptr, float* __restrict col_ptr) {

    im_ptr += c * iw * ih * iz;

    for (unsigned int kz = 0; kz < kd; kz++) {
        imcol2d_padnone_n32x_s(c, kw, iw, ix, kh, ih, iy, im_ptr, col_ptr);

        im_ptr += c * iw * ih;
        col_ptr += c * kw * kh;
    }
}

__forceinline void imcol3d_padnone_aligned_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const unsigned int kh, const unsigned int ih, const unsigned int iy,
    const unsigned int kd, const unsigned int id, const unsigned int iz,
    const float* __restrict im_ptr, float* __restrict col_ptr) {

    im_ptr += c * iw * ih * iz;

    for (unsigned int kz = 0; kz < kd; kz++) {
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
    const float* __restrict im_ptr, float* __restrict col_ptr, const __m256i mask) {

    im_ptr += c * iw * ih * iz;

    for (unsigned int kz = 0; kz < kd; kz++) {
        imcol2d_padnone_unaligned_s(c, kw, iw, ix, kh, ih, iy, im_ptr, col_ptr, mask);

        im_ptr += c * iw * ih;
        col_ptr += c * kw * kh;
    }
}

#pragma endregion padnone

#pragma region padzero

__forceinline void imcol1d_padzero_n32x_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const float* __restrict im_ptr, float* __restrict col_ptr) {

#ifdef _DEBUG
    if ((c % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)im_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)col_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const unsigned int pw = kw / 2;

    if (ix >= pw && ix + pw < iw) {
        copy_n32x_s(c * kw, im_ptr + c * (ix - pw), col_ptr);
    }
    else {
        for (unsigned int kx = 0, x = ix - pw; kx < kw; kx++, x++) {
            if (x < iw) {
                copy_n32x_s(c, im_ptr + c * x, col_ptr);
            }
            else {
                zeroset_n32x_s(c, col_ptr);
            }
            col_ptr += c;
        }
    }
}

__forceinline void imcol1d_padzero_aligned_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const float* __restrict im_ptr, float* __restrict col_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)im_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)col_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const unsigned int pw = kw / 2;

    if (ix >= pw && ix + pw < iw) {
        copy_aligned_s(c * kw, im_ptr + c * (ix - pw), col_ptr);
    }
    else {
        for (unsigned int kx = 0, x = ix - pw; kx < kw; kx++, x++) {
            if (x < iw) {
                copy_aligned_s(c, im_ptr + c * x, col_ptr);
            }
            else {
                zeroset_aligned_s(c, col_ptr);
            }
            col_ptr += c;
        }
    }
}

__forceinline float imcol1d_padzero_unaligned_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const float* __restrict im_ptr, float* __restrict col_ptr, const __m256i mask) {

#ifdef _DEBUG
    if (((c * kw) & AVX2_FLOAT_REMAIN_MASK) == 0 || ((size_t)im_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)col_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const unsigned int pw = kw / 2;

    if (ix >= pw && ix + pw < iw) {
        copy_unaligned_s(c * kw, im_ptr + c * (ix - pw), col_ptr, mask);
    }
    else {
        const __m256i mask_c = _mm256_setmask_ps(c & AVX2_FLOAT_REMAIN_MASK);

        for (unsigned int kx = 0, x = ix - pw; kx < kw; kx++, x++) {
            if (x < iw) {
                copy_unaligned_s(c, im_ptr + c * x, col_ptr, mask_c);
            }
            else {
                zeroset_unaligned_s(c, col_ptr, mask_c);
            }
            col_ptr += c;
        }
    }
}

__forceinline void imcol2d_padzero_n32x_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const unsigned int kh, const unsigned int ih, const unsigned int iy,
    const float* __restrict im_ptr, float* __restrict col_ptr) {

    const unsigned int ph = kh / 2;

    for (unsigned int ky = 0, y = iy - ph; ky < kh; ky++, y++) {
        if (y < ih) {
            imcol1d_padzero_n32x_s(c, kw, iw, ix, im_ptr + c * iw * y, col_ptr);
        }
        else {
            zeroset_n32x_s(c * kw, col_ptr);
        }

        col_ptr += c * kw;
    }
}

__forceinline void imcol2d_padzero_aligned_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const unsigned int kh, const unsigned int ih, const unsigned int iy,
    const float* __restrict im_ptr, float* __restrict col_ptr) {

    const unsigned int ph = kh / 2;

    for (unsigned int ky = 0, y = iy - ph; ky < kh; ky++, y++) {
        if (y < ih) {
            imcol1d_padzero_aligned_s(c, kw, iw, ix, im_ptr + c * iw * y, col_ptr);
        }
        else {
            zeroset_aligned_s(c * kw, col_ptr);
        }

        col_ptr += c * kw;
    }
}

__forceinline void imcol2d_padzero_unaligned_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const unsigned int kh, const unsigned int ih, const unsigned int iy,
    const float* __restrict im_ptr, float* __restrict col_ptr, const __m256i mask) {

    const unsigned int ph = kh / 2;

    for (unsigned int ky = 0, y = iy - ph; ky < kh; ky++, y++) {
        if (y < ih) {
            imcol1d_padzero_unaligned_s(c, kw, iw, ix, im_ptr + c * iw * y, col_ptr, mask);
        }
        else {
            zeroset_unaligned_s(c * kw, col_ptr, _mm256_setmask_ps((c * kw) & AVX2_FLOAT_REMAIN_MASK));
        }

        col_ptr += c * kw;
    }
}

__forceinline void imcol3d_padzero_n32x_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const unsigned int kh, const unsigned int ih, const unsigned int iy,
    const unsigned int kd, const unsigned int id, const unsigned int iz,
    const float* __restrict im_ptr, float* __restrict col_ptr) {

    const unsigned int pd = kd / 2;

    for (unsigned int kz = 0, z = iz - pd; kz < kd; kz++, z++) {
        if (z < id) {
            imcol2d_padzero_n32x_s(c, kw, iw, ix, kw, ih, iy, im_ptr + c * iw * ih * z, col_ptr);
        }
        else {
            zeroset_n32x_s(c * kw * kh, col_ptr);
        }

        col_ptr += c * kw * kh;
    }
}

__forceinline void imcol3d_padzero_aligned_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const unsigned int kh, const unsigned int ih, const unsigned int iy,
    const unsigned int kd, const unsigned int id, const unsigned int iz,
    const float* __restrict im_ptr, float* __restrict col_ptr) {

    const unsigned int pd = kd / 2;

    for (unsigned int kz = 0, z = iz - pd; kz < kd; kz++, z++) {
        if (z < id) {
            imcol2d_padzero_aligned_s(c, kw, iw, ix, kw, ih, iy, im_ptr + c * iw * ih * z, col_ptr);
        }
        else {
            zeroset_aligned_s(c * kw * kh, col_ptr);
        }

        col_ptr += c * kw * kh;
    }
}

__forceinline void imcol3d_padzero_unaligned_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const unsigned int kh, const unsigned int ih, const unsigned int iy,
    const unsigned int kd, const unsigned int id, const unsigned int iz,
    const float* __restrict im_ptr, float* __restrict col_ptr, const __m256i mask) {

    const unsigned int pd = kd / 2;

    for (unsigned int kz = 0, z = iz - pd; kz < kd; kz++, z++) {
        if (z < id) {
            imcol2d_padzero_unaligned_s(c, kw, iw, ix, kw, ih, iy, im_ptr + c * iw * ih * z, col_ptr, mask);
        }
        else {
            zeroset_unaligned_s(c * kw * kh, col_ptr, _mm256_setmask_ps((c * kw * kh) & AVX2_FLOAT_REMAIN_MASK));
        }

        col_ptr += c * kw * kh;
    }
}

#pragma endregion padzero


#pragma region padedge

__forceinline void imcol1d_padedge_n32x_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const float* __restrict im_ptr, float* __restrict col_ptr) {

#ifdef _DEBUG
    if ((c % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)im_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)col_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const unsigned int pw = kw / 2;

    if (ix >= pw && ix + pw < iw) {
        copy_n32x_s(c * kw, im_ptr + c * (ix - pw), col_ptr);
    }
    else {
        for (unsigned int kx = 0; kx < kw; kx++) {
            const unsigned int x = padclip(ix + kx, iw, pw);

            copy_n32x_s(c, im_ptr + c * x, col_ptr);

            col_ptr += c;
        }
    }
}

__forceinline void imcol1d_padedge_aligned_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const float* __restrict im_ptr, float* __restrict col_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)im_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)col_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const unsigned int pw = kw / 2;

    if (ix >= pw && ix + pw < iw) {
        copy_aligned_s(c * kw, im_ptr + c * (ix - pw), col_ptr);
    }
    else {
        for (unsigned int kx = 0; kx < kw; kx++) {
            const unsigned int x = padclip(ix + kx, iw, pw);

            copy_aligned_s(c, im_ptr + c * x, col_ptr);

            col_ptr += c;
        }
    }
}

__forceinline float imcol1d_padedge_unaligned_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const float* __restrict im_ptr, float* __restrict col_ptr, const __m256i mask) {

#ifdef _DEBUG
    if (((c * kw) & AVX2_FLOAT_REMAIN_MASK) == 0 || ((size_t)im_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)col_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    const unsigned int pw = kw / 2;

    if (ix >= pw && ix + pw < iw) {
        copy_unaligned_s(c * kw, im_ptr + c * (ix - pw), col_ptr, mask);
    }
    else {
        const __m256i mask_c = _mm256_setmask_ps(c & AVX2_FLOAT_REMAIN_MASK);

        for (unsigned int kx = 0; kx < kw; kx++) {
            const unsigned int x = padclip(ix + kx, iw, pw);

            copy_unaligned_s(c, im_ptr + c * x, col_ptr, mask_c);

            col_ptr += c;
        }
    }
}

__forceinline void imcol2d_padedge_n32x_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const unsigned int kh, const unsigned int ih, const unsigned int iy,
    const float* __restrict im_ptr, float* __restrict col_ptr) {

    const unsigned int ph = kh / 2;

    for (unsigned int ky = 0; ky < kh; ky++) {
        const unsigned int y = padclip(iy + ky, ih, ph);

        imcol1d_padedge_n32x_s(c, kw, iw, ix, im_ptr + c * iw * y, col_ptr);

        col_ptr += c * kw;
    }
}

__forceinline void imcol2d_padedge_aligned_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const unsigned int kh, const unsigned int ih, const unsigned int iy,
    const float* __restrict im_ptr, float* __restrict col_ptr) {

    const unsigned int ph = kh / 2;

    for (unsigned int ky = 0; ky < kh; ky++) {
        const unsigned int y = padclip(iy + ky, ih, ph);

        imcol1d_padedge_aligned_s(c, kw, iw, ix, im_ptr + c * iw * y, col_ptr);

        col_ptr += c * kw;
    }
}

__forceinline void imcol2d_padedge_unaligned_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const unsigned int kh, const unsigned int ih, const unsigned int iy,
    const float* __restrict im_ptr, float* __restrict col_ptr, const __m256i mask) {

    const unsigned int ph = kh / 2;

    for (unsigned int ky = 0; ky < kh; ky++) {
        const unsigned int y = padclip(iy + ky, ih, ph);

        imcol1d_padedge_unaligned_s(c, kw, iw, ix, im_ptr + c * iw * y, col_ptr, mask);

        col_ptr += c * kw;
    }
}

__forceinline void imcol3d_padedge_n32x_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const unsigned int kh, const unsigned int ih, const unsigned int iy,
    const unsigned int kd, const unsigned int id, const unsigned int iz,
    const float* __restrict im_ptr, float* __restrict col_ptr) {

    const unsigned int pd = kd / 2;

    for (unsigned int kz = 0; kz < kd; kz++) {
        const unsigned int z = padclip(iz + kz, id, pd);

        imcol2d_padedge_n32x_s(c, kw, iw, ix, kh, ih, iy, im_ptr + c * iw * ih * z, col_ptr);

        col_ptr += c * kw * kh;
    }
}

__forceinline void imcol3d_padedge_aligned_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const unsigned int kh, const unsigned int ih, const unsigned int iy,
    const unsigned int kd, const unsigned int id, const unsigned int iz,
    const float* __restrict im_ptr, float* __restrict col_ptr) {

    const unsigned int pd = kd / 2;

    for (unsigned int kz = 0; kz < kd; kz++) {
        const unsigned int z = padclip(iz + kz, id, pd);

        imcol2d_padedge_aligned_s(c, kw, iw, ix, kh, ih, iy, im_ptr + c * iw * ih * z, col_ptr);

        col_ptr += c * kw * kh;
    }
}

__forceinline void imcol3d_padedge_unaligned_s(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const unsigned int kh, const unsigned int ih, const unsigned int iy,
    const unsigned int kd, const unsigned int id, const unsigned int iz,
    const float* __restrict im_ptr, float* __restrict col_ptr, const __m256i mask) {

    const unsigned int pd = kd / 2;

    for (unsigned int kz = 0; kz < kd; kz++) {
        const unsigned int z = padclip(iz + kz, id, pd);

        imcol2d_padedge_unaligned_s(c, kw, iw, ix, kh, ih, iy, im_ptr + c * iw * ih * z, col_ptr, mask);

        col_ptr += c * kw * kh;
    }
}

#pragma endregion padedge