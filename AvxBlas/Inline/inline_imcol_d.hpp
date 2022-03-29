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
    const uint c,
    const uint kw, const uint iw, const uint ix,
    indoubles im_ptr, outdoubles col_ptr) {

#ifdef _DEBUG
    if ((c % (AVX2_DOUBLE_STRIDE * 4)) != 0 || ((size_t)im_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)col_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    copy_n16x_d(c * kw, im_ptr + c * ix, col_ptr);
}

__forceinline void imcol1d_padnone_aligned_d(
    const uint c,
    const uint kw, const uint iw, const uint ix,
    indoubles im_ptr, outdoubles col_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_DOUBLE_REMAIN_MASK) != 0 || ((size_t)im_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)col_ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    copy_aligned_d(c * kw, im_ptr + c * ix, col_ptr);
}

__forceinline void imcol1d_padnone_unaligned_d(
    const uint c,
    const uint kw, const uint iw, const uint ix,
    indoubles im_ptr, outdoubles col_ptr, const __m256i mask) {

#ifdef _DEBUG
    if ((c & AVX2_DOUBLE_REMAIN_MASK) == 0) {
        throw std::exception();
    }
#endif // _DEBUG

    copy_unaligned_d(c * kw, im_ptr + c * ix, col_ptr, mask);
}

__forceinline void imcol2d_padnone_n16x_d(
    const uint c,
    const uint kw, const uint iw, const uint ix,
    const uint kh, const uint ih, const uint iy,
    indoubles im_ptr, outdoubles col_ptr) {

    im_ptr += c * iw * iy;

    for (uint ky = 0; ky < kh; ky++) {
        imcol1d_padnone_n16x_d(c, kw, iw, ix, im_ptr, col_ptr);

        im_ptr += c * iw;
        col_ptr += c * kw;
    }
}

__forceinline void imcol2d_padnone_aligned_d(
    const uint c,
    const uint kw, const uint iw, const uint ix,
    const uint kh, const uint ih, const uint iy,
    indoubles im_ptr, outdoubles col_ptr) {

    im_ptr += c * iw * iy;

    for (uint ky = 0; ky < kh; ky++) {
        imcol1d_padnone_aligned_d(c, kw, iw, ix, im_ptr, col_ptr);

        im_ptr += c * iw;
        col_ptr += c * kw;
    }
}

__forceinline void imcol2d_padnone_unaligned_d(
    const uint c,
    const uint kw, const uint iw, const uint ix,
    const uint kh, const uint ih, const uint iy,
    indoubles im_ptr, outdoubles col_ptr, const __m256i mask) {

    im_ptr += c * iw * iy;

    for (uint ky = 0; ky < kh; ky++) {
        imcol1d_padnone_unaligned_d(c, kw, iw, ix, im_ptr, col_ptr, mask);

        im_ptr += c * iw;
        col_ptr += c * kw;
    }
}

__forceinline void imcol3d_padnone_n16x_d(
    const uint c,
    const uint kw, const uint iw, const uint ix,
    const uint kh, const uint ih, const uint iy,
    const uint kd, const uint id, const uint iz,
    indoubles im_ptr, outdoubles col_ptr) {

    im_ptr += c * iw * ih * iz;

    for (uint kz = 0; kz < kd; kz++) {
        imcol2d_padnone_n16x_d(c, kw, iw, ix, kh, ih, iy, im_ptr, col_ptr);

        im_ptr += c * iw * ih;
        col_ptr += c * kw * kh;
    }
}

__forceinline void imcol3d_padnone_aligned_d(
    const uint c,
    const uint kw, const uint iw, const uint ix,
    const uint kh, const uint ih, const uint iy,
    const uint kd, const uint id, const uint iz,
    indoubles im_ptr, outdoubles col_ptr) {

    im_ptr += c * iw * ih * iz;

    for (uint kz = 0; kz < kd; kz++) {
        imcol2d_padnone_aligned_d(c, kw, iw, ix, kh, ih, iy, im_ptr, col_ptr);

        im_ptr += c * iw * ih;
        col_ptr += c * kw * kh;
    }
}

__forceinline void imcol3d_padnone_unaligned_d(
    const uint c,
    const uint kw, const uint iw, const uint ix,
    const uint kh, const uint ih, const uint iy,
    const uint kd, const uint id, const uint iz,
    indoubles im_ptr, outdoubles col_ptr, const __m256i mask) {

    im_ptr += c * iw * ih * iz;

    for (uint kz = 0; kz < kd; kz++) {
        imcol2d_padnone_unaligned_d(c, kw, iw, ix, kh, ih, iy, im_ptr, col_ptr, mask);

        im_ptr += c * iw * ih;
        col_ptr += c * kw * kh;
    }
}

#pragma endregion padnone

#pragma region padzero

__forceinline void imcol1d_padzero_n16x_d(
    const uint c,
    const uint kw, const uint iw, const uint ix, const uint pw,
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
        for (uint kx = 0, x = ix - pw; kx < kw; kx++, x++) {
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
    const uint c,
    const uint kw, const uint iw, const uint ix, const uint pw,
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
        for (uint kx = 0, x = ix - pw; kx < kw; kx++, x++) {
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

__forceinline void imcol1d_padzero_unaligned_d(
    const uint c,
    const uint kw, const uint iw, const uint ix, const uint pw,
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

        for (uint kx = 0, x = ix - pw; kx < kw; kx++, x++) {
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
    const uint c,
    const uint kw, const uint iw, const uint ix, const uint pw,
    const uint kh, const uint ih, const uint iy, const uint ph,
    indoubles im_ptr, outdoubles col_ptr) {

    for (uint ky = 0, y = iy - ph; ky < kh; ky++, y++) {
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
    const uint c,
    const uint kw, const uint iw, const uint ix, const uint pw,
    const uint kh, const uint ih, const uint iy, const uint ph,
    indoubles im_ptr, outdoubles col_ptr) {

    for (uint ky = 0, y = iy - ph; ky < kh; ky++, y++) {
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
    const uint c,
    const uint kw, const uint iw, const uint ix, const uint pw,
    const uint kh, const uint ih, const uint iy, const uint ph,
    indoubles im_ptr, outdoubles col_ptr, const __m256i mask) {

    for (uint ky = 0, y = iy - ph; ky < kh; ky++, y++) {
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
    const uint c,
    const uint kw, const uint iw, const uint ix, const uint pw,
    const uint kh, const uint ih, const uint iy, const uint ph,
    const uint kd, const uint id, const uint iz, const uint pd,
    indoubles im_ptr, outdoubles col_ptr) {

    for (uint kz = 0, z = iz - pd; kz < kd; kz++, z++) {
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
    const uint c,
    const uint kw, const uint iw, const uint ix, const uint pw,
    const uint kh, const uint ih, const uint iy, const uint ph,
    const uint kd, const uint id, const uint iz, const uint pd,
    indoubles im_ptr, outdoubles col_ptr) {

    for (uint kz = 0, z = iz - pd; kz < kd; kz++, z++) {
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
    const uint c,
    const uint kw, const uint iw, const uint ix, const uint pw,
    const uint kh, const uint ih, const uint iy, const uint ph,
    const uint kd, const uint id, const uint iz, const uint pd,
    indoubles im_ptr, outdoubles col_ptr, const __m256i mask) {

    for (uint kz = 0, z = iz - pd; kz < kd; kz++, z++) {
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
    const uint c,
    const uint kw, const uint iw, const uint ix, const uint pw,
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
        for (uint kx = 0; kx < kw; kx++) {
            const uint x = padclip(ix + kx, iw, pw);

            copy_n16x_d(c, im_ptr + c * x, col_ptr);

            col_ptr += c;
        }
    }
}

__forceinline void imcol1d_padedge_aligned_d(
    const uint c,
    const uint kw, const uint iw, const uint ix, const uint pw,
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
        for (uint kx = 0; kx < kw; kx++) {
            const uint x = padclip(ix + kx, iw, pw);

            copy_aligned_d(c, im_ptr + c * x, col_ptr);

            col_ptr += c;
        }
    }
}

__forceinline void imcol1d_padedge_unaligned_d(
    const uint c,
    const uint kw, const uint iw, const uint ix, const uint pw,
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

        for (uint kx = 0; kx < kw; kx++) {
            const uint x = padclip(ix + kx, iw, pw);

            copy_unaligned_d(c, im_ptr + c * x, col_ptr, mask_c);

            col_ptr += c;
        }
    }
}

__forceinline void imcol2d_padedge_n16x_d(
    const uint c,
    const uint kw, const uint iw, const uint ix, const uint pw,
    const uint kh, const uint ih, const uint iy, const uint ph,
    indoubles im_ptr, outdoubles col_ptr) {

    for (uint ky = 0; ky < kh; ky++) {
        const uint y = padclip(iy + ky, ih, ph);

        imcol1d_padedge_n16x_d(c, kw, iw, ix, pw, im_ptr + c * iw * y, col_ptr);

        col_ptr += c * kw;
    }
}

__forceinline void imcol2d_padedge_aligned_d(
    const uint c,
    const uint kw, const uint iw, const uint ix, const uint pw,
    const uint kh, const uint ih, const uint iy, const uint ph,
    indoubles im_ptr, outdoubles col_ptr) {

    for (uint ky = 0; ky < kh; ky++) {
        const uint y = padclip(iy + ky, ih, ph);

        imcol1d_padedge_aligned_d(c, kw, iw, ix, pw, im_ptr + c * iw * y, col_ptr);

        col_ptr += c * kw;
    }
}

__forceinline void imcol2d_padedge_unaligned_d(
    const uint c,
    const uint kw, const uint iw, const uint ix, const uint pw,
    const uint kh, const uint ih, const uint iy, const uint ph,
    indoubles im_ptr, outdoubles col_ptr, const __m256i mask) {

    for (uint ky = 0; ky < kh; ky++) {
        const uint y = padclip(iy + ky, ih, ph);

        imcol1d_padedge_unaligned_d(c, kw, iw, ix, pw, im_ptr + c * iw * y, col_ptr, mask);

        col_ptr += c * kw;
    }
}

__forceinline void imcol3d_padedge_n16x_d(
    const uint c,
    const uint kw, const uint iw, const uint ix, const uint pw,
    const uint kh, const uint ih, const uint iy, const uint ph,
    const uint kd, const uint id, const uint iz, const uint pd,
    indoubles im_ptr, outdoubles col_ptr) {

    for (uint kz = 0; kz < kd; kz++) {
        const uint z = padclip(iz + kz, id, pd);

        imcol2d_padedge_n16x_d(c, kw, iw, ix, pw, kh, ih, iy, ph, im_ptr + c * iw * ih * z, col_ptr);

        col_ptr += c * kw * kh;
    }
}

__forceinline void imcol3d_padedge_aligned_d(
    const uint c,
    const uint kw, const uint iw, const uint ix, const uint pw,
    const uint kh, const uint ih, const uint iy, const uint ph,
    const uint kd, const uint id, const uint iz, const uint pd,
    indoubles im_ptr, outdoubles col_ptr) {

    for (uint kz = 0; kz < kd; kz++) {
        const uint z = padclip(iz + kz, id, pd);

        imcol2d_padedge_aligned_d(c, kw, iw, ix, pw, kh, ih, iy, ph, im_ptr + c * iw * ih * z, col_ptr);

        col_ptr += c * kw * kh;
    }
}

__forceinline void imcol3d_padedge_unaligned_d(
    const uint c,
    const uint kw, const uint iw, const uint ix, const uint pw,
    const uint kh, const uint ih, const uint iy, const uint ph,
    const uint kd, const uint id, const uint iz, const uint pd,
    indoubles im_ptr, outdoubles col_ptr, const __m256i mask) {

    for (uint kz = 0; kz < kd; kz++) {
        const uint z = padclip(iz + kz, id, pd);

        imcol2d_padedge_unaligned_d(c, kw, iw, ix, pw, kh, ih, iy, ph, im_ptr + c * iw * ih * z, col_ptr, mask);

        col_ptr += c * kw * kh;
    }
}

#pragma endregion padedge