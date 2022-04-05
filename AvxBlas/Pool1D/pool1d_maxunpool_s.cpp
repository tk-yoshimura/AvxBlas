#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_numeric.hpp"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include "../Inline/inline_copy_s.hpp"

using namespace System;

#pragma unmanaged

int pool1d_maxunpool_n32x_seqk_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    infloats x_ptr, infloats y_ptr, infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((c % (AVX2_FLOAT_STRIDE * 4)) != 0 || (sx != kw)
        || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0
        || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0) {

        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const uint pw = (kw - 1) / 2;
    const uint ew = ow * sx - pw;

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;
    __m256 dy0, dy1, dy2, dy3, dx0, dx1, dx2, dx3;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ox < ow; ix += sx, ox++) {
            for (uint kx = pw <= ix ? 0 : pw - ix, x = ix + kx - pw; kx < kw && x < iw; kx++, x++) {

                const float* xc_ptr = x_ptr + c * x;
                const float* yc_ptr = y_ptr + c * ox;
                float* dxc_ptr = dx_ptr + c * x;
                const float* dyc_ptr = dy_ptr + c * ox;

                uint r = c;

                while (r >= AVX2_FLOAT_STRIDE * 4) {
                    _mm256_load_x4_ps(xc_ptr, x0, x1, x2, x3);
                    _mm256_load_x4_ps(yc_ptr, y0, y1, y2, y3);
                    _mm256_load_x4_ps(dyc_ptr, dy0, dy1, dy2, dy3);

                    dx0 = _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS));
                    dx1 = _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS));
                    dx2 = _mm256_and_ps(dy2, _mm256_cmp_ps(y2, x2, _CMP_LE_OS));
                    dx3 = _mm256_and_ps(dy3, _mm256_cmp_ps(y3, x3, _CMP_LE_OS));

                    _mm256_stream_x4_ps(dxc_ptr, dx0, dx1, dx2, dx3);

                    xc_ptr += AVX2_FLOAT_STRIDE * 4;
                    yc_ptr += AVX2_FLOAT_STRIDE * 4;
                    dxc_ptr += AVX2_FLOAT_STRIDE * 4;
                    dyc_ptr += AVX2_FLOAT_STRIDE * 4;
                    r -= AVX2_FLOAT_STRIDE * 4;
                }
            }
        }
        if (ew < iw) {
            const uint dx = iw - ew;

            zeroset_s(c * dx, dx_ptr + c * ew);
        }

        x_ptr += c * iw;
        y_ptr += c * ow;
        dx_ptr += c * iw;
        dy_ptr += c * ow;
    }

    return SUCCESS;
}

int pool1d_maxunpool_n32x_sltk_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    infloats x_ptr, infloats y_ptr, infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((c % (AVX2_FLOAT_STRIDE * 4)) != 0 || (sx >= kw)
        || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0
        || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0) {

        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const uint pw = (kw - 1) / 2;

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;
    __m256 dy0, dy1, dy2, dy3, dx0, dx1, dx2, dx3;

    zeroset_s(n * c * iw, dx_ptr);

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ox < ow; ix += sx, ox++) {
            for (uint kx = pw <= ix ? 0 : pw - ix, x = ix + kx - pw; kx < kw && x < iw; kx++, x++) {

                const float* xc_ptr = x_ptr + c * x;
                const float* yc_ptr = y_ptr + c * ox;
                float* dxc_ptr = dx_ptr + c * x;
                const float* dyc_ptr = dy_ptr + c * ox;

                uint r = c;

                while (r >= AVX2_FLOAT_STRIDE * 4) {
                    _mm256_load_x4_ps(xc_ptr, x0, x1, x2, x3);
                    _mm256_load_x4_ps(yc_ptr, y0, y1, y2, y3);
                    _mm256_load_x4_ps(dyc_ptr, dy0, dy1, dy2, dy3);
                    _mm256_load_x4_ps(dxc_ptr, dx0, dx1, dx2, dx3);

                    dx0 = _mm256_add_ps(dx0, _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS)));
                    dx1 = _mm256_add_ps(dx1, _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS)));
                    dx2 = _mm256_add_ps(dx2, _mm256_and_ps(dy2, _mm256_cmp_ps(y2, x2, _CMP_LE_OS)));
                    dx3 = _mm256_add_ps(dx3, _mm256_and_ps(dy3, _mm256_cmp_ps(y3, x3, _CMP_LE_OS)));

                    _mm256_store_x4_ps(dxc_ptr, dx0, dx1, dx2, dx3);

                    xc_ptr += AVX2_FLOAT_STRIDE * 4;
                    yc_ptr += AVX2_FLOAT_STRIDE * 4;
                    dxc_ptr += AVX2_FLOAT_STRIDE * 4;
                    dyc_ptr += AVX2_FLOAT_STRIDE * 4;
                    r -= AVX2_FLOAT_STRIDE * 4;
                }
            }
        }

        x_ptr += c * iw;
        y_ptr += c * ow;
        dx_ptr += c * iw;
        dy_ptr += c * ow;
    }

    return SUCCESS;
}

int pool1d_maxunpool_n32x_sgtk_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    infloats x_ptr, infloats y_ptr, infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((c % (AVX2_FLOAT_STRIDE * 4)) != 0 || (sx <= kw)
        || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0
        || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0) {

        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const uint pw = (kw - 1) / 2;

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;
    __m256 dy0, dy1, dy2, dy3, dx0, dx1, dx2, dx3;

    zeroset_s(n * c * iw, dx_ptr);

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ox < ow; ix += sx, ox++) {
            for (uint kx = pw <= ix ? 0 : pw - ix, x = ix + kx - pw; kx < kw && x < iw; kx++, x++) {

                const float* xc_ptr = x_ptr + c * x;
                const float* yc_ptr = y_ptr + c * ox;
                float* dxc_ptr = dx_ptr + c * x;
                const float* dyc_ptr = dy_ptr + c * ox;

                uint r = c;

                while (r >= AVX2_FLOAT_STRIDE * 4) {
                    _mm256_load_x4_ps(xc_ptr, x0, x1, x2, x3);
                    _mm256_load_x4_ps(yc_ptr, y0, y1, y2, y3);
                    _mm256_load_x4_ps(dyc_ptr, dy0, dy1, dy2, dy3);

                    dx0 = _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS));
                    dx1 = _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS));
                    dx2 = _mm256_and_ps(dy2, _mm256_cmp_ps(y2, x2, _CMP_LE_OS));
                    dx3 = _mm256_and_ps(dy3, _mm256_cmp_ps(y3, x3, _CMP_LE_OS));

                    _mm256_stream_x4_ps(dxc_ptr, dx0, dx1, dx2, dx3);

                    xc_ptr += AVX2_FLOAT_STRIDE * 4;
                    yc_ptr += AVX2_FLOAT_STRIDE * 4;
                    dxc_ptr += AVX2_FLOAT_STRIDE * 4;
                    dyc_ptr += AVX2_FLOAT_STRIDE * 4;
                    r -= AVX2_FLOAT_STRIDE * 4;
                }
            }
        }

        x_ptr += c * iw;
        y_ptr += c * ow;
        dx_ptr += c * iw;
        dy_ptr += c * ow;
    }

    return SUCCESS;
}

int pool1d_maxunpool_aligned_seqk_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    infloats x_ptr, infloats y_ptr, infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) != 0 || (sx != kw)
        || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0
        || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0) {

        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const uint pw = (kw - 1) / 2;
    const uint ew = ow * sx - pw;

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;
    __m256 dy0, dy1, dy2, dy3, dx0, dx1, dx2, dx3;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ox < ow; ix += sx, ox++) {
            for (uint kx = pw <= ix ? 0 : pw - ix, x = ix + kx - pw; kx < kw && x < iw; kx++, x++) {

                const float* xc_ptr = x_ptr + c * x;
                const float* yc_ptr = y_ptr + c * ox;
                float* dxc_ptr = dx_ptr + c * x;
                const float* dyc_ptr = dy_ptr + c * ox;

                uint r = c;

                while (r >= AVX2_FLOAT_STRIDE * 4) {
                    _mm256_load_x4_ps(xc_ptr, x0, x1, x2, x3);
                    _mm256_load_x4_ps(yc_ptr, y0, y1, y2, y3);
                    _mm256_load_x4_ps(dyc_ptr, dy0, dy1, dy2, dy3);

                    dx0 = _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS));
                    dx1 = _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS));
                    dx2 = _mm256_and_ps(dy2, _mm256_cmp_ps(y2, x2, _CMP_LE_OS));
                    dx3 = _mm256_and_ps(dy3, _mm256_cmp_ps(y3, x3, _CMP_LE_OS));

                    _mm256_stream_x4_ps(dxc_ptr, dx0, dx1, dx2, dx3);

                    xc_ptr += AVX2_FLOAT_STRIDE * 4;
                    yc_ptr += AVX2_FLOAT_STRIDE * 4;
                    dxc_ptr += AVX2_FLOAT_STRIDE * 4;
                    dyc_ptr += AVX2_FLOAT_STRIDE * 4;
                    r -= AVX2_FLOAT_STRIDE * 4;
                }
                if (r >= AVX2_FLOAT_STRIDE * 2) {
                    _mm256_load_x2_ps(xc_ptr, x0, x1);
                    _mm256_load_x2_ps(yc_ptr, y0, y1);
                    _mm256_load_x2_ps(dyc_ptr, dy0, dy1);

                    dx0 = _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS));
                    dx1 = _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS));

                    _mm256_stream_x2_ps(dxc_ptr, dx0, dx1);

                    xc_ptr += AVX2_FLOAT_STRIDE * 2;
                    yc_ptr += AVX2_FLOAT_STRIDE * 2;
                    dxc_ptr += AVX2_FLOAT_STRIDE * 2;
                    dyc_ptr += AVX2_FLOAT_STRIDE * 2;
                    r -= AVX2_FLOAT_STRIDE * 2;
                }
                if (r >= AVX2_FLOAT_STRIDE) {
                    _mm256_load_x1_ps(xc_ptr, x0);
                    _mm256_load_x1_ps(yc_ptr, y0);
                    _mm256_load_x1_ps(dyc_ptr, dy0);

                    dx0 = _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS));

                    _mm256_stream_x1_ps(dxc_ptr, dx0);
                }
            }
        }
        if (ew < iw) {
            const uint dx = iw - ew;

            zeroset_s(c * dx, dx_ptr + c * ew);
        }

        x_ptr += c * iw;
        y_ptr += c * ow;
        dx_ptr += c * iw;
        dy_ptr += c * ow;
    }

    return SUCCESS;
}

int pool1d_maxunpool_aligned_sltk_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    infloats x_ptr, infloats y_ptr, infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) != 0 || (sx >= kw)
        || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0
        || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0) {

        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const uint pw = (kw - 1) / 2;

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;
    __m256 dy0, dy1, dy2, dy3, dx0, dx1, dx2, dx3;

    zeroset_s(n * c * iw, dx_ptr);

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ox < ow; ix += sx, ox++) {
            for (uint kx = pw <= ix ? 0 : pw - ix, x = ix + kx - pw; kx < kw && x < iw; kx++, x++) {

                const float* xc_ptr = x_ptr + c * x;
                const float* yc_ptr = y_ptr + c * ox;
                float* dxc_ptr = dx_ptr + c * x;
                const float* dyc_ptr = dy_ptr + c * ox;

                uint r = c;

                while (r >= AVX2_FLOAT_STRIDE * 4) {
                    _mm256_load_x4_ps(xc_ptr, x0, x1, x2, x3);
                    _mm256_load_x4_ps(yc_ptr, y0, y1, y2, y3);
                    _mm256_load_x4_ps(dyc_ptr, dy0, dy1, dy2, dy3);
                    _mm256_load_x4_ps(dxc_ptr, dx0, dx1, dx2, dx3);

                    dx0 = _mm256_add_ps(dx0, _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS)));
                    dx1 = _mm256_add_ps(dx1, _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS)));
                    dx2 = _mm256_add_ps(dx2, _mm256_and_ps(dy2, _mm256_cmp_ps(y2, x2, _CMP_LE_OS)));
                    dx3 = _mm256_add_ps(dx3, _mm256_and_ps(dy3, _mm256_cmp_ps(y3, x3, _CMP_LE_OS)));

                    _mm256_store_x4_ps(dxc_ptr, dx0, dx1, dx2, dx3);

                    xc_ptr += AVX2_FLOAT_STRIDE * 4;
                    yc_ptr += AVX2_FLOAT_STRIDE * 4;
                    dxc_ptr += AVX2_FLOAT_STRIDE * 4;
                    dyc_ptr += AVX2_FLOAT_STRIDE * 4;
                    r -= AVX2_FLOAT_STRIDE * 4;
                }
                if (r >= AVX2_FLOAT_STRIDE * 2) {
                    _mm256_load_x2_ps(xc_ptr, x0, x1);
                    _mm256_load_x2_ps(yc_ptr, y0, y1);
                    _mm256_load_x2_ps(dyc_ptr, dy0, dy1);
                    _mm256_load_x2_ps(dxc_ptr, dx0, dx1);

                    dx0 = _mm256_add_ps(dx0, _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS)));
                    dx1 = _mm256_add_ps(dx1, _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS)));

                    _mm256_store_x2_ps(dxc_ptr, dx0, dx1);

                    xc_ptr += AVX2_FLOAT_STRIDE * 2;
                    yc_ptr += AVX2_FLOAT_STRIDE * 2;
                    dxc_ptr += AVX2_FLOAT_STRIDE * 2;
                    dyc_ptr += AVX2_FLOAT_STRIDE * 2;
                    r -= AVX2_FLOAT_STRIDE * 2;
                }
                if (r >= AVX2_FLOAT_STRIDE) {
                    _mm256_load_x1_ps(xc_ptr, x0);
                    _mm256_load_x1_ps(yc_ptr, y0);
                    _mm256_load_x1_ps(dyc_ptr, dy0);
                    _mm256_load_x1_ps(dxc_ptr, dx0);

                    dx0 = _mm256_add_ps(dx0, _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS)));

                    _mm256_store_x1_ps(dxc_ptr, dx0);

                    xc_ptr += AVX2_FLOAT_STRIDE;
                    yc_ptr += AVX2_FLOAT_STRIDE;
                    dxc_ptr += AVX2_FLOAT_STRIDE;
                    dyc_ptr += AVX2_FLOAT_STRIDE;
                    r -= AVX2_FLOAT_STRIDE;
                }
            }
        }

        x_ptr += c * iw;
        y_ptr += c * ow;
        dx_ptr += c * iw;
        dy_ptr += c * ow;
    }

    return SUCCESS;
}

int pool1d_maxunpool_aligned_sgtk_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    infloats x_ptr, infloats y_ptr, infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) != 0 || (sx <= kw)
        || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0
        || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0) {

        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const uint pw = (kw - 1) / 2;

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;
    __m256 dy0, dy1, dy2, dy3, dx0, dx1, dx2, dx3;

    zeroset_s(n * c * iw, dx_ptr);

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ox < ow; ix += sx, ox++) {
            for (uint kx = pw <= ix ? 0 : pw - ix, x = ix + kx - pw; kx < kw && x < iw; kx++, x++) {

                const float* xc_ptr = x_ptr + c * x;
                const float* yc_ptr = y_ptr + c * ox;
                float* dxc_ptr = dx_ptr + c * x;
                const float* dyc_ptr = dy_ptr + c * ox;

                uint r = c;

                while (r >= AVX2_FLOAT_STRIDE * 4) {
                    _mm256_load_x4_ps(xc_ptr, x0, x1, x2, x3);
                    _mm256_load_x4_ps(yc_ptr, y0, y1, y2, y3);
                    _mm256_load_x4_ps(dyc_ptr, dy0, dy1, dy2, dy3);

                    dx0 = _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS));
                    dx1 = _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS));
                    dx2 = _mm256_and_ps(dy2, _mm256_cmp_ps(y2, x2, _CMP_LE_OS));
                    dx3 = _mm256_and_ps(dy3, _mm256_cmp_ps(y3, x3, _CMP_LE_OS));

                    _mm256_stream_x4_ps(dxc_ptr, dx0, dx1, dx2, dx3);

                    xc_ptr += AVX2_FLOAT_STRIDE * 4;
                    yc_ptr += AVX2_FLOAT_STRIDE * 4;
                    dxc_ptr += AVX2_FLOAT_STRIDE * 4;
                    dyc_ptr += AVX2_FLOAT_STRIDE * 4;
                    r -= AVX2_FLOAT_STRIDE * 4;
                }
                if (r >= AVX2_FLOAT_STRIDE * 2) {
                    _mm256_load_x2_ps(xc_ptr, x0, x1);
                    _mm256_load_x2_ps(yc_ptr, y0, y1);
                    _mm256_load_x2_ps(dyc_ptr, dy0, dy1);

                    dx0 = _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS));
                    dx1 = _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS));

                    _mm256_stream_x2_ps(dxc_ptr, dx0, dx1);

                    xc_ptr += AVX2_FLOAT_STRIDE * 2;
                    yc_ptr += AVX2_FLOAT_STRIDE * 2;
                    dxc_ptr += AVX2_FLOAT_STRIDE * 2;
                    dyc_ptr += AVX2_FLOAT_STRIDE * 2;
                    r -= AVX2_FLOAT_STRIDE * 2;
                }
                if (r >= AVX2_FLOAT_STRIDE) {
                    _mm256_load_x1_ps(xc_ptr, x0);
                    _mm256_load_x1_ps(yc_ptr, y0);
                    _mm256_load_x1_ps(dyc_ptr, dy0);

                    dx0 = _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS));

                    _mm256_stream_x1_ps(dxc_ptr, dx0);
                }
            }
        }

        x_ptr += c * iw;
        y_ptr += c * ow;
        dx_ptr += c * iw;
        dy_ptr += c * ow;
    }

    return SUCCESS;
}


int pool1d_maxunpool_unaligned_seqk_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    infloats x_ptr, infloats y_ptr, infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) == 0 || (sx != kw)
        || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0
        || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0) {

        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const uint pw = (kw - 1) / 2;
    const uint ew = ow * sx - pw;

    const __m256i mask = _mm256_setmask_ps(c & AVX2_FLOAT_REMAIN_MASK);

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;
    __m256 dy0, dy1, dy2, dy3, dx0, dx1, dx2, dx3;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ox < ow; ix += sx, ox++) {
            for (uint kx = pw <= ix ? 0 : pw - ix, x = ix + kx - pw; kx < kw && x < iw; kx++, x++) {

                const float* xc_ptr = x_ptr + c * x;
                const float* yc_ptr = y_ptr + c * ox;
                float* dxc_ptr = dx_ptr + c * x;
                const float* dyc_ptr = dy_ptr + c * ox;

                uint r = c;

                while (r >= AVX2_FLOAT_STRIDE * 4) {
                    _mm256_loadu_x4_ps(xc_ptr, x0, x1, x2, x3);
                    _mm256_loadu_x4_ps(yc_ptr, y0, y1, y2, y3);
                    _mm256_loadu_x4_ps(dyc_ptr, dy0, dy1, dy2, dy3);

                    dx0 = _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS));
                    dx1 = _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS));
                    dx2 = _mm256_and_ps(dy2, _mm256_cmp_ps(y2, x2, _CMP_LE_OS));
                    dx3 = _mm256_and_ps(dy3, _mm256_cmp_ps(y3, x3, _CMP_LE_OS));

                    _mm256_storeu_x4_ps(dxc_ptr, dx0, dx1, dx2, dx3);

                    xc_ptr += AVX2_FLOAT_STRIDE * 4;
                    yc_ptr += AVX2_FLOAT_STRIDE * 4;
                    dxc_ptr += AVX2_FLOAT_STRIDE * 4;
                    dyc_ptr += AVX2_FLOAT_STRIDE * 4;
                    r -= AVX2_FLOAT_STRIDE * 4;
                }
                if (r >= AVX2_FLOAT_STRIDE * 2) {
                    _mm256_loadu_x2_ps(xc_ptr, x0, x1);
                    _mm256_loadu_x2_ps(yc_ptr, y0, y1);
                    _mm256_loadu_x2_ps(dyc_ptr, dy0, dy1);

                    dx0 = _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS));
                    dx1 = _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS));

                    _mm256_storeu_x2_ps(dxc_ptr, dx0, dx1);

                    xc_ptr += AVX2_FLOAT_STRIDE * 2;
                    yc_ptr += AVX2_FLOAT_STRIDE * 2;
                    dxc_ptr += AVX2_FLOAT_STRIDE * 2;
                    dyc_ptr += AVX2_FLOAT_STRIDE * 2;
                    r -= AVX2_FLOAT_STRIDE * 2;
                }
                if (r >= AVX2_FLOAT_STRIDE) {
                    _mm256_loadu_x1_ps(xc_ptr, x0);
                    _mm256_loadu_x1_ps(yc_ptr, y0);
                    _mm256_loadu_x1_ps(dyc_ptr, dy0);

                    dx0 = _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS));

                    _mm256_storeu_x1_ps(dxc_ptr, dx0);

                    xc_ptr += AVX2_FLOAT_STRIDE;
                    yc_ptr += AVX2_FLOAT_STRIDE;
                    dxc_ptr += AVX2_FLOAT_STRIDE;
                    dyc_ptr += AVX2_FLOAT_STRIDE;
                    r -= AVX2_FLOAT_STRIDE;
                }
                if (r > 0) {
                    _mm256_loadu_x1_ps(xc_ptr, x0);
                    _mm256_loadu_x1_ps(yc_ptr, y0);
                    _mm256_loadu_x1_ps(dyc_ptr, dy0);

                    dx0 = _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS));

                    _mm256_maskstore_x1_ps(dxc_ptr, dx0, mask);
                }
            }
        }
        if (ew < iw) {
            const uint dx = iw - ew;

            zeroset_s(c * dx, dx_ptr + c * ew);
        }

        x_ptr += c * iw;
        y_ptr += c * ow;
        dx_ptr += c * iw;
        dy_ptr += c * ow;
    }

    return SUCCESS;
}

int pool1d_maxunpool_unaligned_sltk_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    infloats x_ptr, infloats y_ptr, infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) == 0 || (sx >= kw)
        || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0
        || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0) {

        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const uint pw = (kw - 1) / 2;
    const __m256i mask = _mm256_setmask_ps(c & AVX2_FLOAT_REMAIN_MASK);

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;
    __m256 dy0, dy1, dy2, dy3, dx0, dx1, dx2, dx3;

    zeroset_s(n * c * iw, dx_ptr);

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ox < ow; ix += sx, ox++) {
            for (uint kx = pw <= ix ? 0 : pw - ix, x = ix + kx - pw; kx < kw && x < iw; kx++, x++) {

                const float* xc_ptr = x_ptr + c * x;
                const float* yc_ptr = y_ptr + c * ox;
                float* dxc_ptr = dx_ptr + c * x;
                const float* dyc_ptr = dy_ptr + c * ox;

                uint r = c;

                while (r >= AVX2_FLOAT_STRIDE * 4) {
                    _mm256_loadu_x4_ps(xc_ptr, x0, x1, x2, x3);
                    _mm256_loadu_x4_ps(yc_ptr, y0, y1, y2, y3);
                    _mm256_loadu_x4_ps(dyc_ptr, dy0, dy1, dy2, dy3);
                    _mm256_loadu_x4_ps(dxc_ptr, dx0, dx1, dx2, dx3);

                    dx0 = _mm256_add_ps(dx0, _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS)));
                    dx1 = _mm256_add_ps(dx1, _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS)));
                    dx2 = _mm256_add_ps(dx2, _mm256_and_ps(dy2, _mm256_cmp_ps(y2, x2, _CMP_LE_OS)));
                    dx3 = _mm256_add_ps(dx3, _mm256_and_ps(dy3, _mm256_cmp_ps(y3, x3, _CMP_LE_OS)));

                    _mm256_storeu_x4_ps(dxc_ptr, dx0, dx1, dx2, dx3);

                    xc_ptr += AVX2_FLOAT_STRIDE * 4;
                    yc_ptr += AVX2_FLOAT_STRIDE * 4;
                    dxc_ptr += AVX2_FLOAT_STRIDE * 4;
                    dyc_ptr += AVX2_FLOAT_STRIDE * 4;
                    r -= AVX2_FLOAT_STRIDE * 4;
                }
                if (r >= AVX2_FLOAT_STRIDE * 2) {
                    _mm256_loadu_x2_ps(xc_ptr, x0, x1);
                    _mm256_loadu_x2_ps(yc_ptr, y0, y1);
                    _mm256_loadu_x2_ps(dyc_ptr, dy0, dy1);
                    _mm256_loadu_x2_ps(dxc_ptr, dx0, dx1);

                    dx0 = _mm256_add_ps(dx0, _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS)));
                    dx1 = _mm256_add_ps(dx1, _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS)));

                    _mm256_storeu_x2_ps(dxc_ptr, dx0, dx1);

                    xc_ptr += AVX2_FLOAT_STRIDE * 2;
                    yc_ptr += AVX2_FLOAT_STRIDE * 2;
                    dxc_ptr += AVX2_FLOAT_STRIDE * 2;
                    dyc_ptr += AVX2_FLOAT_STRIDE * 2;
                    r -= AVX2_FLOAT_STRIDE * 2;
                }
                if (r >= AVX2_FLOAT_STRIDE) {
                    _mm256_loadu_x1_ps(xc_ptr, x0);
                    _mm256_loadu_x1_ps(yc_ptr, y0);
                    _mm256_loadu_x1_ps(dyc_ptr, dy0);
                    _mm256_loadu_x1_ps(dxc_ptr, dx0);

                    dx0 = _mm256_add_ps(dx0, _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS)));

                    _mm256_storeu_x1_ps(dxc_ptr, dx0);

                    xc_ptr += AVX2_FLOAT_STRIDE;
                    yc_ptr += AVX2_FLOAT_STRIDE;
                    dxc_ptr += AVX2_FLOAT_STRIDE;
                    dyc_ptr += AVX2_FLOAT_STRIDE;
                    r -= AVX2_FLOAT_STRIDE;
                }
                if (r > 0) {
                    _mm256_loadu_x1_ps(xc_ptr, x0);
                    _mm256_loadu_x1_ps(yc_ptr, y0);
                    _mm256_loadu_x1_ps(dyc_ptr, dy0);
                    _mm256_loadu_x1_ps(dxc_ptr, dx0);

                    dx0 = _mm256_add_ps(dx0, _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS)));

                    _mm256_maskstore_x1_ps(dxc_ptr, dx0, mask);
                }
            }
        }

        x_ptr += c * iw;
        y_ptr += c * ow;
        dx_ptr += c * iw;
        dy_ptr += c * ow;
    }

    return SUCCESS;
}

int pool1d_maxunpool_unaligned_sgtk_s(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint sx, const uint kw,
    infloats x_ptr, infloats y_ptr, infloats dy_ptr, outfloats dx_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) == 0 || (sx <= kw)
        || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0
        || ((size_t)dy_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)dx_ptr % AVX2_ALIGNMENT) != 0) {

        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const uint pw = (kw - 1) / 2;
    const __m256i mask = _mm256_setmask_ps(c & AVX2_FLOAT_REMAIN_MASK);

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;
    __m256 dy0, dy1, dy2, dy3, dx0, dx1, dx2, dx3;

    zeroset_s(n * c * iw, dx_ptr);

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ox < ow; ix += sx, ox++) {
            for (uint kx = pw <= ix ? 0 : pw - ix, x = ix + kx - pw; kx < kw && x < iw; kx++, x++) {

                const float* xc_ptr = x_ptr + c * x;
                const float* yc_ptr = y_ptr + c * ox;
                float* dxc_ptr = dx_ptr + c * x;
                const float* dyc_ptr = dy_ptr + c * ox;

                uint r = c;

                while (r >= AVX2_FLOAT_STRIDE * 4) {
                    _mm256_loadu_x4_ps(xc_ptr, x0, x1, x2, x3);
                    _mm256_loadu_x4_ps(yc_ptr, y0, y1, y2, y3);
                    _mm256_loadu_x4_ps(dyc_ptr, dy0, dy1, dy2, dy3);

                    dx0 = _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS));
                    dx1 = _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS));
                    dx2 = _mm256_and_ps(dy2, _mm256_cmp_ps(y2, x2, _CMP_LE_OS));
                    dx3 = _mm256_and_ps(dy3, _mm256_cmp_ps(y3, x3, _CMP_LE_OS));

                    _mm256_storeu_x4_ps(dxc_ptr, dx0, dx1, dx2, dx3);

                    xc_ptr += AVX2_FLOAT_STRIDE * 4;
                    yc_ptr += AVX2_FLOAT_STRIDE * 4;
                    dxc_ptr += AVX2_FLOAT_STRIDE * 4;
                    dyc_ptr += AVX2_FLOAT_STRIDE * 4;
                    r -= AVX2_FLOAT_STRIDE * 4;
                }
                if (r >= AVX2_FLOAT_STRIDE * 2) {
                    _mm256_loadu_x2_ps(xc_ptr, x0, x1);
                    _mm256_loadu_x2_ps(yc_ptr, y0, y1);
                    _mm256_loadu_x2_ps(dyc_ptr, dy0, dy1);

                    dx0 = _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS));
                    dx1 = _mm256_and_ps(dy1, _mm256_cmp_ps(y1, x1, _CMP_LE_OS));

                    _mm256_storeu_x2_ps(dxc_ptr, dx0, dx1);

                    xc_ptr += AVX2_FLOAT_STRIDE * 2;
                    yc_ptr += AVX2_FLOAT_STRIDE * 2;
                    dxc_ptr += AVX2_FLOAT_STRIDE * 2;
                    dyc_ptr += AVX2_FLOAT_STRIDE * 2;
                    r -= AVX2_FLOAT_STRIDE * 2;
                }
                if (r >= AVX2_FLOAT_STRIDE) {
                    _mm256_loadu_x1_ps(xc_ptr, x0);
                    _mm256_loadu_x1_ps(yc_ptr, y0);
                    _mm256_loadu_x1_ps(dyc_ptr, dy0);

                    dx0 = _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS));

                    _mm256_storeu_x1_ps(dxc_ptr, dx0);

                    xc_ptr += AVX2_FLOAT_STRIDE;
                    yc_ptr += AVX2_FLOAT_STRIDE;
                    dxc_ptr += AVX2_FLOAT_STRIDE;
                    dyc_ptr += AVX2_FLOAT_STRIDE;
                    r -= AVX2_FLOAT_STRIDE;
                }
                if (r > 0) {
                    _mm256_loadu_x1_ps(xc_ptr, x0);
                    _mm256_loadu_x1_ps(yc_ptr, y0);
                    _mm256_loadu_x1_ps(dyc_ptr, dy0);

                    dx0 = _mm256_and_ps(dy0, _mm256_cmp_ps(y0, x0, _CMP_LE_OS));

                    _mm256_maskstore_x1_ps(dxc_ptr, dx0, mask);
                }
            }
        }

        x_ptr += c * iw;
        y_ptr += c * ow;
        dx_ptr += c * iw;
        dy_ptr += c * ow;
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Pool1D::MaxUnpooling(
    UInt32 n, UInt32 c, UInt32 iw,
    UInt32 sx, UInt32 kw,
    Array<float>^ x, Array<float>^ y, Array<float>^ dy, Array<float>^ dx) {

    if (n > MAX_BATCHES) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidBatches);
    }
    if (c > MAX_CHANNELS) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidChannels);
    }
    if (sx <= 0 || sx > MAX_POOL_STRIDE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidPoolStride);
    }
    if (kw <= 1 || kw > MAX_KERNEL_SIZE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidKernelSize);
    }
    if (iw > MAX_MAP_SIZE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidDataSize);
    }

    if (n <= 0 || c <= 0 || iw <= 0) {
        return;
    }

    UInt32 ow = (iw - 1) / sx + 1;

    Util::CheckProdOverflow(n, c, iw);
    Util::CheckProdOverflow(n, c, ow);

    Util::CheckLength(n * c * iw, x);
    Util::CheckLength(n * c * ow, y);
    Util::CheckLength(n * c * iw, dx);
    Util::CheckLength(n * c * ow, dy);

    Util::CheckDuplicateArray(x, dy, dx);

    const float* x_ptr = (const float*)(x->Ptr.ToPointer());
    const float* y_ptr = (const float*)(y->Ptr.ToPointer());
    const float* dy_ptr = (const float*)(dy->Ptr.ToPointer());
    float* dx_ptr = (float*)(dx->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if ((c % (AVX2_FLOAT_STRIDE * 4)) == 0) {
        if (sx == kw) {
#ifdef _DEBUG
            Console::WriteLine("type n32x sx = kx");
#endif // _DEBUG

            ret = pool1d_maxunpool_n32x_seqk_s(n, c, iw, ow, sx, kw, x_ptr, y_ptr, dy_ptr, dx_ptr);
        }
        else if (sx < kw) {
#ifdef _DEBUG
            Console::WriteLine("type n32x sx < kx");
#endif // _DEBUG

            ret = pool1d_maxunpool_n32x_sltk_s(n, c, iw, ow, sx, kw, x_ptr, y_ptr, dy_ptr, dx_ptr);
        }
        else if (sx > kw) {
#ifdef _DEBUG
            Console::WriteLine("type n32x sx > kx");
#endif // _DEBUG

            ret = pool1d_maxunpool_n32x_sgtk_s(n, c, iw, ow, sx, kw, x_ptr, y_ptr, dy_ptr, dx_ptr);
        }
    }
    else if ((c & AVX2_FLOAT_REMAIN_MASK) == 0) {
        if (sx == kw) {
#ifdef _DEBUG
            Console::WriteLine("type aligned sx = kx");
#endif // _DEBUG

            ret = pool1d_maxunpool_aligned_seqk_s(n, c, iw, ow, sx, kw, x_ptr, y_ptr, dy_ptr, dx_ptr);
        }
        else if (sx < kw) {
#ifdef _DEBUG
            Console::WriteLine("type aligned sx < kx");
#endif // _DEBUG

            ret = pool1d_maxunpool_aligned_sltk_s(n, c, iw, ow, sx, kw, x_ptr, y_ptr, dy_ptr, dx_ptr);
        }
        else if (sx > kw) {
#ifdef _DEBUG
            Console::WriteLine("type aligned sx > kx");
#endif // _DEBUG

            ret = pool1d_maxunpool_aligned_sgtk_s(n, c, iw, ow, sx, kw, x_ptr, y_ptr, dy_ptr, dx_ptr);
        }
    }
    else {
        if (sx == kw) {
#ifdef _DEBUG
            Console::WriteLine("type unaligned sx = kx");
#endif // _DEBUG

            ret = pool1d_maxunpool_unaligned_seqk_s(n, c, iw, ow, sx, kw, x_ptr, y_ptr, dy_ptr, dx_ptr);
        }
        else if (sx < kw) {
#ifdef _DEBUG
            Console::WriteLine("type unaligned sx < kx");
#endif // _DEBUG

            ret = pool1d_maxunpool_unaligned_sltk_s(n, c, iw, ow, sx, kw, x_ptr, y_ptr, dy_ptr, dx_ptr);
        }
        else if (sx > kw) {
#ifdef _DEBUG
            Console::WriteLine("type unaligned sx > kx");
#endif // _DEBUG

            ret = pool1d_maxunpool_unaligned_sgtk_s(n, c, iw, ow, sx, kw, x_ptr, y_ptr, dy_ptr, dx_ptr);
        }
    }

    Util::AssertReturnCode(ret);
}