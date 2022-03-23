#pragma once
#pragma unmanaged

#include "../constants.h"
#include "../utils.h"

#include "inline_numeric.hpp"
#include "inline_copy_d.hpp"
#include "inline_zeroset_d.hpp"

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

#pragma region padnone

__forceinline void imcol1d_padnone_n16x_d(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    indoubles im_ptr, outdoubles col_ptr) {

#ifdef _DEBUG
    if ((c % (AVX2_DOUBLE_STRIDE * 4)) != 0 || ((size_t)im_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)col_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    copy_n16x_d(c * kw, im_ptr + c * ix, col_ptr);
}

__forceinline void imcol1d_padnone_aligned_d(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    indoubles im_ptr, outdoubles col_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_DOUBLE_REMAIN_MASK) != 0 || ((size_t)im_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)col_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    copy_aligned_d(c * kw, im_ptr + c * ix, col_ptr);
}

__forceinline void imcol1d_padnone_unaligned_d(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    indoubles im_ptr, outdoubles col_ptr, const __m256i mask) {

#ifdef _DEBUG
    if ((c & AVX2_DOUBLE_REMAIN_MASK) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    copy_unaligned_d(c * kw, im_ptr + c * ix, col_ptr, mask);
}

__forceinline void imcol2d_padnone_n16x_d(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const unsigned int kh, const unsigned int ih, const unsigned int iy,
    indoubles im_ptr, outdoubles col_ptr) {

    im_ptr += c * iw * iy;

    for (unsigned int ky = 0; ky < kh; ky++) {
        imcol1d_padnone_n16x_d(c, kw, iw, ix, im_ptr, col_ptr);

        im_ptr += c * iw;
        col_ptr += c * kw;
    }
}

__forceinline void imcol2d_padnone_aligned_d(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const unsigned int kh, const unsigned int ih, const unsigned int iy,
    indoubles im_ptr, outdoubles col_ptr) {

    im_ptr += c * iw * iy;

    for (unsigned int ky = 0; ky < kh; ky++) {
        imcol1d_padnone_aligned_d(c, kw, iw, ix, im_ptr, col_ptr);

        im_ptr += c * iw;
        col_ptr += c * kw;
    }
}

__forceinline void imcol2d_padnone_unaligned_d(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const unsigned int kh, const unsigned int ih, const unsigned int iy,
    indoubles im_ptr, outdoubles col_ptr, const __m256i mask) {

    im_ptr += c * iw * iy;

    for (unsigned int ky = 0; ky < kh; ky++) {
        imcol1d_padnone_unaligned_d(c, kw, iw, ix, im_ptr, col_ptr, mask);

        im_ptr += c * iw;
        col_ptr += c * kw;
    }
}

__forceinline void imcol3d_padnone_n16x_d(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const unsigned int kh, const unsigned int ih, const unsigned int iy,
    const unsigned int kd, const unsigned int id, const unsigned int iz,
    indoubles im_ptr, outdoubles col_ptr) {

    im_ptr += c * iw * ih * iz;

    for (unsigned int kz = 0; kz < kd; kz++) {
        imcol2d_padnone_n16x_d(c, kw, iw, ix, kh, ih, iy, im_ptr, col_ptr);

        im_ptr += c * iw * ih;
        col_ptr += c * kw * kh;
    }
}

__forceinline void imcol3d_padnone_aligned_d(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const unsigned int kh, const unsigned int ih, const unsigned int iy,
    const unsigned int kd, const unsigned int id, const unsigned int iz,
    indoubles im_ptr, outdoubles col_ptr) {

    im_ptr += c * iw * ih * iz;

    for (unsigned int kz = 0; kz < kd; kz++) {
        imcol2d_padnone_aligned_d(c, kw, iw, ix, kh, ih, iy, im_ptr, col_ptr);

        im_ptr += c * iw * ih;
        col_ptr += c * kw * kh;
    }
}

__forceinline void imcol3d_padnone_unaligned_d(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix,
    const unsigned int kh, const unsigned int ih, const unsigned int iy,
    const unsigned int kd, const unsigned int id, const unsigned int iz,
    indoubles im_ptr, outdoubles col_ptr, const __m256i mask) {

    im_ptr += c * iw * ih * iz;

    for (unsigned int kz = 0; kz < kd; kz++) {
        imcol2d_padnone_unaligned_d(c, kw, iw, ix, kh, ih, iy, im_ptr, col_ptr, mask);

        im_ptr += c * iw * ih;
        col_ptr += c * kw * kh;
    }
}

#pragma endregion padnone

#pragma region padzero

__forceinline void imcol1d_padzero_n16x_d(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix, const unsigned int pw,
    indoubles im_ptr, outdoubles col_ptr) {

#ifdef _DEBUG
    if ((c % (AVX2_DOUBLE_STRIDE * 4)) != 0 || ((size_t)im_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)col_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    if (ix >= pw && ix + pw < iw) {
        copy_n16x_d(c * kw, im_ptr + c * (ix - pw), col_ptr);
    }
    else {
        for (unsigned int kx = 0, x = ix - pw; kx < kw; kx++, x++) {
            if (x < iw) {
                copy_n16x_d(c, im_ptr + c * x, col_ptr);
            }
            else {
                zeroset_n16x_d(c, col_ptr);
            }
            col_ptr += c;
        }
    }
}

__forceinline void imcol1d_padzero_aligned_d(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix, const unsigned int pw,
    indoubles im_ptr, outdoubles col_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_DOUBLE_REMAIN_MASK) != 0 || ((size_t)im_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)col_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    if (ix >= pw && ix + pw < iw) {
        copy_aligned_d(c * kw, im_ptr + c * (ix - pw), col_ptr);
    }
    else {
        for (unsigned int kx = 0, x = ix - pw; kx < kw; kx++, x++) {
            if (x < iw) {
                copy_aligned_d(c, im_ptr + c * x, col_ptr);
            }
            else {
                zeroset_aligned_d(c, col_ptr);
            }
            col_ptr += c;
        }
    }
}

__forceinline double imcol1d_padzero_unaligned_d(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix, const unsigned int pw,
    indoubles im_ptr, outdoubles col_ptr, const __m256i mask) {

#ifdef _DEBUG
    if ((c & AVX2_DOUBLE_REMAIN_MASK) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    if (ix >= pw && ix + pw < iw) {
        copy_unaligned_d(c * kw, im_ptr + c * (ix - pw), col_ptr, mask);
    }
    else {
        const __m256i mask_c = _mm256_setmask_pd(c & AVX2_DOUBLE_REMAIN_MASK);

        for (unsigned int kx = 0, x = ix - pw; kx < kw; kx++, x++) {
            if (x < iw) {
                copy_unaligned_d(c, im_ptr + c * x, col_ptr, mask_c);
            }
            else {
                zeroset_unaligned_d(c, col_ptr, mask_c);
            }
            col_ptr += c;
        }
    }
}

__forceinline void imcol2d_padzero_n16x_d(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix, const unsigned int pw,
    const unsigned int kh, const unsigned int ih, const unsigned int iy, const unsigned int ph,
    indoubles im_ptr, outdoubles col_ptr) {

    for (unsigned int ky = 0, y = iy - ph; ky < kh; ky++, y++) {
        if (y < ih) {
            imcol1d_padzero_n16x_d(c, kw, iw, ix, pw, im_ptr + c * iw * y, col_ptr);
        }
        else {
            zeroset_n16x_d(c * kw, col_ptr);
        }

        col_ptr += c * kw;
    }
}

__forceinline void imcol2d_padzero_aligned_d(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix, const unsigned int pw,
    const unsigned int kh, const unsigned int ih, const unsigned int iy, const unsigned int ph,
    indoubles im_ptr, outdoubles col_ptr) {

    for (unsigned int ky = 0, y = iy - ph; ky < kh; ky++, y++) {
        if (y < ih) {
            imcol1d_padzero_aligned_d(c, kw, iw, ix, pw, im_ptr + c * iw * y, col_ptr);
        }
        else {
            zeroset_aligned_d(c * kw, col_ptr);
        }

        col_ptr += c * kw;
    }
}

__forceinline void imcol2d_padzero_unaligned_d(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix, const unsigned int pw,
    const unsigned int kh, const unsigned int ih, const unsigned int iy, const unsigned int ph,
    indoubles im_ptr, outdoubles col_ptr, const __m256i mask) {

    for (unsigned int ky = 0, y = iy - ph; ky < kh; ky++, y++) {
        if (y < ih) {
            imcol1d_padzero_unaligned_d(c, kw, iw, ix, pw, im_ptr + c * iw * y, col_ptr, mask);
        }
        else {
            zeroset_unaligned_d(c * kw, col_ptr, _mm256_setmask_pd((c * kw) & AVX2_DOUBLE_REMAIN_MASK));
        }

        col_ptr += c * kw;
    }
}

__forceinline void imcol3d_padzero_n16x_d(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix, const unsigned int pw,
    const unsigned int kh, const unsigned int ih, const unsigned int iy, const unsigned int ph,
    const unsigned int kd, const unsigned int id, const unsigned int iz, const unsigned int pd,
    indoubles im_ptr, outdoubles col_ptr) {

    for (unsigned int kz = 0, z = iz - pd; kz < kd; kz++, z++) {
        if (z < id) {
            imcol2d_padzero_n16x_d(c, kw, iw, ix, pw, kh, ih, iy, ph, im_ptr + c * iw * ih * z, col_ptr);
        }
        else {
            zeroset_n16x_d(c * kw * kh, col_ptr);
        }

        col_ptr += c * kw * kh;
    }
}

__forceinline void imcol3d_padzero_aligned_d(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix, const unsigned int pw,
    const unsigned int kh, const unsigned int ih, const unsigned int iy, const unsigned int ph,
    const unsigned int kd, const unsigned int id, const unsigned int iz, const unsigned int pd,
    indoubles im_ptr, outdoubles col_ptr) {

    for (unsigned int kz = 0, z = iz - pd; kz < kd; kz++, z++) {
        if (z < id) {
            imcol2d_padzero_aligned_d(c, kw, iw, ix, pw, kh, ih, iy, ph, im_ptr + c * iw * ih * z, col_ptr);
        }
        else {
            zeroset_aligned_d(c * kw * kh, col_ptr);
        }

        col_ptr += c * kw * kh;
    }
}

__forceinline void imcol3d_padzero_unaligned_d(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix, const unsigned int pw,
    const unsigned int kh, const unsigned int ih, const unsigned int iy, const unsigned int ph,
    const unsigned int kd, const unsigned int id, const unsigned int iz, const unsigned int pd,
    indoubles im_ptr, outdoubles col_ptr, const __m256i mask) {

    for (unsigned int kz = 0, z = iz - pd; kz < kd; kz++, z++) {
        if (z < id) {
            imcol2d_padzero_unaligned_d(c, kw, iw, ix, pw, kh, ih, iy, ph, im_ptr + c * iw * ih * z, col_ptr, mask);
        }
        else {
            zeroset_unaligned_d(c * kw * kh, col_ptr, _mm256_setmask_pd((c * kw * kh) & AVX2_DOUBLE_REMAIN_MASK));
        }

        col_ptr += c * kw * kh;
    }
}

#pragma endregion padzero


#pragma region padedge

__forceinline void imcol1d_padedge_n16x_d(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix, const unsigned int pw,
    indoubles im_ptr, outdoubles col_ptr) {

#ifdef _DEBUG
    if ((c % (AVX2_DOUBLE_STRIDE * 4)) != 0 || ((size_t)im_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)col_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    if (ix >= pw && ix + pw < iw) {
        copy_n16x_d(c * kw, im_ptr + c * (ix - pw), col_ptr);
    }
    else {
        for (unsigned int kx = 0; kx < kw; kx++) {
            const unsigned int x = padclip(ix + kx, iw, pw);

            copy_n16x_d(c, im_ptr + c * x, col_ptr);

            col_ptr += c;
        }
    }
}

__forceinline void imcol1d_padedge_aligned_d(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix, const unsigned int pw,
    indoubles im_ptr, outdoubles col_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_DOUBLE_REMAIN_MASK) != 0 || ((size_t)im_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)col_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    if (ix >= pw && ix + pw < iw) {
        copy_aligned_d(c * kw, im_ptr + c * (ix - pw), col_ptr);
    }
    else {
        for (unsigned int kx = 0; kx < kw; kx++) {
            const unsigned int x = padclip(ix + kx, iw, pw);

            copy_aligned_d(c, im_ptr + c * x, col_ptr);

            col_ptr += c;
        }
    }
}

__forceinline double imcol1d_padedge_unaligned_d(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix, const unsigned int pw,
    indoubles im_ptr, outdoubles col_ptr, const __m256i mask) {

#ifdef _DEBUG
    if ((c & AVX2_DOUBLE_REMAIN_MASK) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    if (ix >= pw && ix + pw < iw) {
        copy_unaligned_d(c * kw, im_ptr + c * (ix - pw), col_ptr, mask);
    }
    else {
        const __m256i mask_c = _mm256_setmask_pd(c & AVX2_DOUBLE_REMAIN_MASK);

        for (unsigned int kx = 0; kx < kw; kx++) {
            const unsigned int x = padclip(ix + kx, iw, pw);

            copy_unaligned_d(c, im_ptr + c * x, col_ptr, mask_c);

            col_ptr += c;
        }
    }
}

__forceinline void imcol2d_padedge_n16x_d(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix, const unsigned int pw,
    const unsigned int kh, const unsigned int ih, const unsigned int iy, const unsigned int ph,
    indoubles im_ptr, outdoubles col_ptr) {

    for (unsigned int ky = 0; ky < kh; ky++) {
        const unsigned int y = padclip(iy + ky, ih, ph);

        imcol1d_padedge_n16x_d(c, kw, iw, ix, pw, im_ptr + c * iw * y, col_ptr);

        col_ptr += c * kw;
    }
}

__forceinline void imcol2d_padedge_aligned_d(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix, const unsigned int pw,
    const unsigned int kh, const unsigned int ih, const unsigned int iy, const unsigned int ph,
    indoubles im_ptr, outdoubles col_ptr) {

    for (unsigned int ky = 0; ky < kh; ky++) {
        const unsigned int y = padclip(iy + ky, ih, ph);

        imcol1d_padedge_aligned_d(c, kw, iw, ix, pw, im_ptr + c * iw * y, col_ptr);

        col_ptr += c * kw;
    }
}

__forceinline void imcol2d_padedge_unaligned_d(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix, const unsigned int pw,
    const unsigned int kh, const unsigned int ih, const unsigned int iy, const unsigned int ph,
    indoubles im_ptr, outdoubles col_ptr, const __m256i mask) {

    for (unsigned int ky = 0; ky < kh; ky++) {
        const unsigned int y = padclip(iy + ky, ih, ph);

        imcol1d_padedge_unaligned_d(c, kw, iw, ix, pw, im_ptr + c * iw * y, col_ptr, mask);

        col_ptr += c * kw;
    }
}

__forceinline void imcol3d_padedge_n16x_d(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix, const unsigned int pw,
    const unsigned int kh, const unsigned int ih, const unsigned int iy, const unsigned int ph,
    const unsigned int kd, const unsigned int id, const unsigned int iz, const unsigned int pd,
    indoubles im_ptr, outdoubles col_ptr) {

    for (unsigned int kz = 0; kz < kd; kz++) {
        const unsigned int z = padclip(iz + kz, id, pd);

        imcol2d_padedge_n16x_d(c, kw, iw, ix, pw, kh, ih, iy, ph, im_ptr + c * iw * ih * z, col_ptr);

        col_ptr += c * kw * kh;
    }
}

__forceinline void imcol3d_padedge_aligned_d(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix, const unsigned int pw,
    const unsigned int kh, const unsigned int ih, const unsigned int iy, const unsigned int ph,
    const unsigned int kd, const unsigned int id, const unsigned int iz, const unsigned int pd,
    indoubles im_ptr, outdoubles col_ptr) {

    for (unsigned int kz = 0; kz < kd; kz++) {
        const unsigned int z = padclip(iz + kz, id, pd);

        imcol2d_padedge_aligned_d(c, kw, iw, ix, pw, kh, ih, iy, ph, im_ptr + c * iw * ih * z, col_ptr);

        col_ptr += c * kw * kh;
    }
}

__forceinline void imcol3d_padedge_unaligned_d(
    const unsigned int c,
    const unsigned int kw, const unsigned int iw, const unsigned int ix, const unsigned int pw,
    const unsigned int kh, const unsigned int ih, const unsigned int iy, const unsigned int ph,
    const unsigned int kd, const unsigned int id, const unsigned int iz, const unsigned int pd,
    indoubles im_ptr, outdoubles col_ptr, const __m256i mask) {

    for (unsigned int kz = 0; kz < kd; kz++) {
        const unsigned int z = padclip(iz + kz, id, pd);

        imcol2d_padedge_unaligned_d(c, kw, iw, ix, pw, kh, ih, iy, ph, im_ptr + c * iw * ih * z, col_ptr, mask);

        col_ptr += c * kw * kh;
    }
}

#pragma endregion padedge