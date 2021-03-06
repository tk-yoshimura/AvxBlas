#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_numeric.hpp"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include "../Inline/inline_transpose_s.hpp"

#pragma unmanaged

__forceinline floatx4 float_linear2d(
    float xlu, float xu, float xru,
    float xl, float xc, float xr,
    float xld, float xd, float xrd) {

    float wlc = xl + xc, wl2c2 = wlc + wlc;
    float wrc = xr + xc, wr2c2 = wrc + wrc;
    float wuc = xu + xc, wu2c2 = wuc + wuc;
    float wdc = xd + xc, wd2c2 = wdc + wdc;

    float ylu = xlu + (wl2c2 + wu2c2);
    float yru = xru + (wr2c2 + wu2c2);
    float yld = xld + (wl2c2 + wd2c2);
    float yrd = xrd + (wr2c2 + wd2c2);

    return floatx4(ylu, yru, yld, yrd);
}

__forceinline __m128x4 _mm_linear2d_ps(
    __m128 xlu, __m128 xu, __m128 xru,
    __m128 xl, __m128 xc, __m128 xr,
    __m128 xld, __m128 xd, __m128 xrd) {

    __m128 wlc = _mm_add_ps(xl, xc), wl2c2 = _mm_add_ps(wlc, wlc);
    __m128 wrc = _mm_add_ps(xr, xc), wr2c2 = _mm_add_ps(wrc, wrc);
    __m128 wuc = _mm_add_ps(xu, xc), wu2c2 = _mm_add_ps(wuc, wuc);
    __m128 wdc = _mm_add_ps(xd, xc), wd2c2 = _mm_add_ps(wdc, wdc);

    __m128 ylu = _mm_add_ps(xlu, _mm_add_ps(wl2c2, wu2c2));
    __m128 yru = _mm_add_ps(xru, _mm_add_ps(wr2c2, wu2c2));
    __m128 yld = _mm_add_ps(xld, _mm_add_ps(wl2c2, wd2c2));
    __m128 yrd = _mm_add_ps(xrd, _mm_add_ps(wr2c2, wd2c2));

    return __m128x4(ylu, yru, yld, yrd);
}

__forceinline __m256x4 _mm256_linear2d_ps(
    __m256 xlu, __m256 xu, __m256 xru,
    __m256 xl, __m256 xc, __m256 xr,
    __m256 xld, __m256 xd, __m256 xrd) {

    __m256 wlc = _mm256_add_ps(xl, xc), wl2c2 = _mm256_add_ps(wlc, wlc);
    __m256 wrc = _mm256_add_ps(xr, xc), wr2c2 = _mm256_add_ps(wrc, wrc);
    __m256 wuc = _mm256_add_ps(xu, xc), wu2c2 = _mm256_add_ps(wuc, wuc);
    __m256 wdc = _mm256_add_ps(xd, xc), wd2c2 = _mm256_add_ps(wdc, wdc);

    __m256 ylu = _mm256_add_ps(xlu, _mm256_add_ps(wl2c2, wu2c2));
    __m256 yru = _mm256_add_ps(xru, _mm256_add_ps(wr2c2, wu2c2));
    __m256 yld = _mm256_add_ps(xld, _mm256_add_ps(wl2c2, wd2c2));
    __m256 yrd = _mm256_add_ps(xrd, _mm256_add_ps(wr2c2, wd2c2));

    return __m256x4(ylu, yru, yld, yrd);
}

int upsample2d_linear_aligned(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 xlu, xu, xru, xl, xc, xr, xld, xd, xrd;

    for (uint i = 0; i < n; i++) {
        for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
            for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);
                const uint iyu = max(iy, 1u) - 1u, iyd = min(iy + 1u, ih - 1u);

                const float* xlu_ptr = x_ptr + c * (ixl + iw * iyu);
                const float* xu_ptr = x_ptr + c * (ix + iw * iyu);
                const float* xru_ptr = x_ptr + c * (ixr + iw * iyu);
                const float* xl_ptr = x_ptr + c * (ixl + iw * iy);
                const float* xc_ptr = x_ptr + c * (ix + iw * iy);
                const float* xr_ptr = x_ptr + c * (ixr + iw * iy);
                const float* xld_ptr = x_ptr + c * (ixl + iw * iyd);
                const float* xd_ptr = x_ptr + c * (ix + iw * iyd);
                const float* xrd_ptr = x_ptr + c * (ixr + iw * iyd);

                float* ylu_ptr = y_ptr + c * (ox + ow * oy);
                float* yru_ptr = ylu_ptr + c;
                float* yld_ptr = ylu_ptr + c * ow;
                float* yrd_ptr = ylu_ptr + c * (ow + 1);

                uint r = c;

                while (r >= AVX2_FLOAT_STRIDE) {
                    _mm256_load_x1_ps(xlu_ptr, xlu);
                    _mm256_load_x1_ps(xu_ptr, xu);
                    _mm256_load_x1_ps(xru_ptr, xru);
                    _mm256_load_x1_ps(xl_ptr, xl);
                    _mm256_load_x1_ps(xc_ptr, xc);
                    _mm256_load_x1_ps(xr_ptr, xr);
                    _mm256_load_x1_ps(xld_ptr, xld);
                    _mm256_load_x1_ps(xd_ptr, xd);
                    _mm256_load_x1_ps(xrd_ptr, xrd);

                    __m256x4 y = _mm256_linear2d_ps(xlu, xu, xru, xl, xc, xr, xld, xd, xrd);

                    _mm256_stream_x1_ps(ylu_ptr, y.imm0);
                    _mm256_stream_x1_ps(yru_ptr, y.imm1);
                    _mm256_stream_x1_ps(yld_ptr, y.imm2);
                    _mm256_stream_x1_ps(yrd_ptr, y.imm3);

                    xlu_ptr += AVX2_FLOAT_STRIDE;
                    xu_ptr += AVX2_FLOAT_STRIDE;
                    xru_ptr += AVX2_FLOAT_STRIDE;
                    xl_ptr += AVX2_FLOAT_STRIDE;
                    xc_ptr += AVX2_FLOAT_STRIDE;
                    xr_ptr += AVX2_FLOAT_STRIDE;
                    xld_ptr += AVX2_FLOAT_STRIDE;
                    xd_ptr += AVX2_FLOAT_STRIDE;
                    xrd_ptr += AVX2_FLOAT_STRIDE;

                    ylu_ptr += AVX2_FLOAT_STRIDE;
                    yru_ptr += AVX2_FLOAT_STRIDE;
                    yld_ptr += AVX2_FLOAT_STRIDE;
                    yrd_ptr += AVX2_FLOAT_STRIDE;
                    r -= AVX2_FLOAT_STRIDE;
                }
            }
        }

        x_ptr += c * iw * ih;
        y_ptr += c * ow * oh;
    }

    return SUCCESS;
}

int upsample2d_linear_unaligned(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) == 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(c & AVX2_FLOAT_REMAIN_MASK);

    __m256 xlu, xu, xru, xl, xc, xr, xld, xd, xrd;

    for (uint i = 0; i < n; i++) {
        for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
            for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);
                const uint iyu = max(iy, 1u) - 1u, iyd = min(iy + 1u, ih - 1u);

                const float* xlu_ptr = x_ptr + c * (ixl + iw * iyu);
                const float* xu_ptr = x_ptr + c * (ix + iw * iyu);
                const float* xru_ptr = x_ptr + c * (ixr + iw * iyu);
                const float* xl_ptr = x_ptr + c * (ixl + iw * iy);
                const float* xc_ptr = x_ptr + c * (ix + iw * iy);
                const float* xr_ptr = x_ptr + c * (ixr + iw * iy);
                const float* xld_ptr = x_ptr + c * (ixl + iw * iyd);
                const float* xd_ptr = x_ptr + c * (ix + iw * iyd);
                const float* xrd_ptr = x_ptr + c * (ixr + iw * iyd);

                float* ylu_ptr = y_ptr + c * (ox + ow * oy);
                float* yru_ptr = ylu_ptr + c;
                float* yld_ptr = ylu_ptr + c * ow;
                float* yrd_ptr = ylu_ptr + c * (ow + 1);

                uint r = c;

                while (r >= AVX2_FLOAT_STRIDE) {
                    _mm256_loadu_x1_ps(xlu_ptr, xlu);
                    _mm256_loadu_x1_ps(xu_ptr, xu);
                    _mm256_loadu_x1_ps(xru_ptr, xru);
                    _mm256_loadu_x1_ps(xl_ptr, xl);
                    _mm256_loadu_x1_ps(xc_ptr, xc);
                    _mm256_loadu_x1_ps(xr_ptr, xr);
                    _mm256_loadu_x1_ps(xld_ptr, xld);
                    _mm256_loadu_x1_ps(xd_ptr, xd);
                    _mm256_loadu_x1_ps(xrd_ptr, xrd);

                    __m256x4 y = _mm256_linear2d_ps(xlu, xu, xru, xl, xc, xr, xld, xd, xrd);

                    _mm256_storeu_x1_ps(ylu_ptr, y.imm0);
                    _mm256_storeu_x1_ps(yru_ptr, y.imm1);
                    _mm256_storeu_x1_ps(yld_ptr, y.imm2);
                    _mm256_storeu_x1_ps(yrd_ptr, y.imm3);

                    xlu_ptr += AVX2_FLOAT_STRIDE;
                    xu_ptr += AVX2_FLOAT_STRIDE;
                    xru_ptr += AVX2_FLOAT_STRIDE;
                    xl_ptr += AVX2_FLOAT_STRIDE;
                    xc_ptr += AVX2_FLOAT_STRIDE;
                    xr_ptr += AVX2_FLOAT_STRIDE;
                    xld_ptr += AVX2_FLOAT_STRIDE;
                    xd_ptr += AVX2_FLOAT_STRIDE;
                    xrd_ptr += AVX2_FLOAT_STRIDE;

                    ylu_ptr += AVX2_FLOAT_STRIDE;
                    yru_ptr += AVX2_FLOAT_STRIDE;
                    yld_ptr += AVX2_FLOAT_STRIDE;
                    yrd_ptr += AVX2_FLOAT_STRIDE;
                    r -= AVX2_FLOAT_STRIDE;
                }
                if (r > 0) {
                    _mm256_loadu_x1_ps(xlu_ptr, xlu);
                    _mm256_loadu_x1_ps(xu_ptr, xu);
                    _mm256_loadu_x1_ps(xru_ptr, xru);
                    _mm256_loadu_x1_ps(xl_ptr, xl);
                    _mm256_loadu_x1_ps(xc_ptr, xc);
                    _mm256_loadu_x1_ps(xr_ptr, xr);
                    _mm256_loadu_x1_ps(xld_ptr, xld);
                    _mm256_loadu_x1_ps(xd_ptr, xd);
                    _mm256_loadu_x1_ps(xrd_ptr, xrd);

                    __m256x4 y = _mm256_linear2d_ps(xlu, xu, xru, xl, xc, xr, xld, xd, xrd);

                    _mm256_maskstore_x1_ps(ylu_ptr, y.imm0, mask);
                    _mm256_maskstore_x1_ps(yru_ptr, y.imm1, mask);
                    _mm256_maskstore_x1_ps(yld_ptr, y.imm2, mask);
                    _mm256_maskstore_x1_ps(yrd_ptr, y.imm3, mask);
                }
            }
        }

        x_ptr += c * iw * ih;
        y_ptr += c * ow * oh;
    }

    return SUCCESS;
}

int upsample2d_linear_c1(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c != 1) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    if (iw > 1u) {
        const __m256i mask = _mm256_setmask_ps(((iw - 2u) * 2u) & AVX2_FLOAT_REMAIN_MASK);

        for (uint i = 0; i < n; i++) {
            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
                const uint iyu = max(iy, 1u) - 1u, iyd = min(iy + 1u, ih - 1u);

                {
                    const uint ix = 0, ixr = 1, ox = ix * 2;

                    const float* xu_ptr = x_ptr + (ix + iw * iyu);
                    const float* xru_ptr = x_ptr + (ixr + iw * iyu);
                    const float* xc_ptr = x_ptr + (ix + iw * iy);
                    const float* xr_ptr = x_ptr + (ixr + iw * iy);
                    const float* xd_ptr = x_ptr + (ix + iw * iyd);
                    const float* xrd_ptr = x_ptr + (ixr + iw * iyd);

                    float* ylu_ptr = y_ptr + (ox + ow * oy);
                    float* yru_ptr = ylu_ptr + 1;
                    float* yld_ptr = ylu_ptr + ow;
                    float* yrd_ptr = ylu_ptr + (ow + 1);

                    float xu = *xu_ptr;
                    float xru = *xru_ptr;
                    float xc = *xc_ptr;
                    float xr = *xr_ptr;
                    float xd = *xd_ptr;
                    float xrd = *xrd_ptr;

                    floatx4 y = float_linear2d(xu, xu, xru, xc, xc, xr, xd, xd, xrd);

                    *ylu_ptr = y.imm0;
                    *yru_ptr = y.imm1;
                    *yld_ptr = y.imm2;
                    *yrd_ptr = y.imm3;
                }
                for (uint ix = 1, ox = 2; ix < iw - 1u; ix += AVX2_FLOAT_STRIDE, ox += AVX2_FLOAT_STRIDE * 2) {
                    const uint ixl = ix - 1u, ixr = ix + 1u;

                    const float* xlu_ptr = x_ptr + (ixl + iw * iyu);
                    const float* xu_ptr = x_ptr + (ix + iw * iyu);
                    const float* xru_ptr = x_ptr + (ixr + iw * iyu);
                    const float* xl_ptr = x_ptr + (ixl + iw * iy);
                    const float* xc_ptr = x_ptr + (ix + iw * iy);
                    const float* xr_ptr = x_ptr + (ixr + iw * iy);
                    const float* xld_ptr = x_ptr + (ixl + iw * iyd);
                    const float* xd_ptr = x_ptr + (ix + iw * iyd);
                    const float* xrd_ptr = x_ptr + (ixr + iw * iyd);

                    float* ylu_ptr = y_ptr + (ox + ow * oy);
                    float* yld_ptr = ylu_ptr + ow;

                    __m256 xlu = _mm256_loadu_ps(xlu_ptr);
                    __m256 xu = _mm256_loadu_ps(xu_ptr);
                    __m256 xru = _mm256_loadu_ps(xru_ptr);
                    __m256 xl = _mm256_loadu_ps(xl_ptr);
                    __m256 xc = _mm256_loadu_ps(xc_ptr);
                    __m256 xr = _mm256_loadu_ps(xr_ptr);
                    __m256 xld = _mm256_loadu_ps(xld_ptr);
                    __m256 xd = _mm256_loadu_ps(xd_ptr);
                    __m256 xrd = _mm256_loadu_ps(xrd_ptr);

                    __m256x4 y = _mm256_linear2d_ps(xlu, xu, xru, xl, xc, xr, xld, xd, xrd);
                    __m256x2 yt0 = _mm256_transpose8x2_ps(y.imm0, y.imm1);
                    __m256x2 yt1 = _mm256_transpose8x2_ps(y.imm2, y.imm3);

                    if (ix + AVX2_FLOAT_STRIDE <= iw - 1u) {
                        _mm256_storeu_x2_ps(ylu_ptr, yt0.imm0, yt0.imm1);
                        _mm256_storeu_x2_ps(yld_ptr, yt1.imm0, yt1.imm1);
                    }
                    else if (ix + AVX2_FLOAT_STRIDE / 2 < iw - 1u) {
                        _mm256_maskstore_x2_ps(ylu_ptr, yt0.imm0, yt0.imm1, mask);
                        _mm256_maskstore_x2_ps(yld_ptr, yt1.imm0, yt1.imm1, mask);
                    }
                    else if (ix + AVX2_FLOAT_STRIDE / 2 == iw - 1u) {
                        _mm256_storeu_x1_ps(ylu_ptr, yt0.imm0);
                        _mm256_storeu_x1_ps(yld_ptr, yt1.imm0);
                    }
                    else {
                        _mm256_maskstore_x1_ps(ylu_ptr, yt0.imm0, mask);
                        _mm256_maskstore_x1_ps(yld_ptr, yt1.imm0, mask);
                    }
                }
                {
                    const uint ixl = (iw - 2u), ix = (iw - 1u), ox = ix * 2;

                    const float* xlu_ptr = x_ptr + (ixl + iw * iyu);
                    const float* xu_ptr = x_ptr + (ix + iw * iyu);
                    const float* xl_ptr = x_ptr + (ixl + iw * iy);
                    const float* xc_ptr = x_ptr + (ix + iw * iy);
                    const float* xld_ptr = x_ptr + (ixl + iw * iyd);
                    const float* xd_ptr = x_ptr + (ix + iw * iyd);

                    float* ylu_ptr = y_ptr + (ox + ow * oy);
                    float* yru_ptr = ylu_ptr + 1;
                    float* yld_ptr = ylu_ptr + ow;
                    float* yrd_ptr = ylu_ptr + (ow + 1);

                    float xlu = *xlu_ptr;
                    float xu = *xu_ptr;
                    float xl = *xl_ptr;
                    float xc = *xc_ptr;
                    float xld = *xld_ptr;
                    float xd = *xd_ptr;

                    floatx4 y = float_linear2d(xlu, xu, xu, xl, xc, xc, xld, xd, xd);

                    *ylu_ptr = y.imm0;
                    *yru_ptr = y.imm1;
                    *yld_ptr = y.imm2;
                    *yrd_ptr = y.imm3;
                }
            }

            x_ptr += iw * ih;
            y_ptr += ow * oh;
        }
    }
    else {
        for (uint i = 0; i < n; i++) {
            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
                const uint iyu = max(iy, 1u) - 1u, iyd = min(iy + 1u, ih - 1u);

                const float* xu_ptr = x_ptr + (iw * iyu);
                const float* xc_ptr = x_ptr + (iw * iy);
                const float* xd_ptr = x_ptr + (iw * iyd);

                float* ylu_ptr = y_ptr + (2u * oy);
                float* yld_ptr = ylu_ptr + 2;

                float xu = *xu_ptr;
                float xc = *xc_ptr;
                float xd = *xd_ptr;

                float xc2 = xc + xc;
                float yu = xu + xc2;
                float yd = xd + xc2;

                yu = yu + yu + yu;
                yd = yd + yd + yd;

                ylu_ptr[0] = ylu_ptr[1] = yu;
                yld_ptr[0] = yld_ptr[1] = yd;
            }

            x_ptr += ih;
            y_ptr += 2u * oh;
        }
    }

    return SUCCESS;
}

int upsample2d_linear_c2to3(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c <= 1 || c >= AVX1_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m128i mask = _mm_setmask_ps(c);

    __m128 xlu, xu, xru, xl, xc, xr, xld, xd, xrd;

    for (uint i = 0; i < n; i++) {
        for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
            for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);
                const uint iyu = max(iy, 1u) - 1u, iyd = min(iy + 1u, ih - 1u);

                const float* xlu_ptr = x_ptr + c * (ixl + iw * iyu);
                const float* xu_ptr = x_ptr + c * (ix + iw * iyu);
                const float* xru_ptr = x_ptr + c * (ixr + iw * iyu);
                const float* xl_ptr = x_ptr + c * (ixl + iw * iy);
                const float* xc_ptr = x_ptr + c * (ix + iw * iy);
                const float* xr_ptr = x_ptr + c * (ixr + iw * iy);
                const float* xld_ptr = x_ptr + c * (ixl + iw * iyd);
                const float* xd_ptr = x_ptr + c * (ix + iw * iyd);
                const float* xrd_ptr = x_ptr + c * (ixr + iw * iyd);

                float* ylu_ptr = y_ptr + c * (ox + ow * oy);
                float* yru_ptr = ylu_ptr + c;
                float* yld_ptr = ylu_ptr + c * ow;
                float* yrd_ptr = ylu_ptr + c * (ow + 1);

                xlu = _mm_loadu_ps(xlu_ptr);
                xu = _mm_loadu_ps(xu_ptr);
                xru = _mm_loadu_ps(xru_ptr);
                xl = _mm_loadu_ps(xl_ptr);
                xc = _mm_loadu_ps(xc_ptr);
                xr = _mm_loadu_ps(xr_ptr);
                xld = _mm_loadu_ps(xld_ptr);
                xd = _mm_loadu_ps(xd_ptr);
                xrd = _mm_loadu_ps(xrd_ptr);

                __m128x4 y = _mm_linear2d_ps(xlu, xu, xru, xl, xc, xr, xld, xd, xrd);

                _mm_maskstore_ps(ylu_ptr, mask, y.imm0);
                _mm_maskstore_ps(yru_ptr, mask, y.imm1);
                _mm_maskstore_ps(yld_ptr, mask, y.imm2);
                _mm_maskstore_ps(yrd_ptr, mask, y.imm3);
            }
        }

        x_ptr += c * iw * ih;
        y_ptr += c * ow * oh;
    }

    return SUCCESS;
}

int upsample2d_linear_c4(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c != AVX1_FLOAT_STRIDE || ((size_t)x_ptr % AVX1_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX1_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m128 xlu, xu, xru, xl, xc, xr, xld, xd, xrd;

    for (uint i = 0; i < n; i++) {
        for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
            for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);
                const uint iyu = max(iy, 1u) - 1u, iyd = min(iy + 1u, ih - 1u);

                const float* xlu_ptr = x_ptr + c * (ixl + iw * iyu);
                const float* xu_ptr = x_ptr + c * (ix + iw * iyu);
                const float* xru_ptr = x_ptr + c * (ixr + iw * iyu);
                const float* xl_ptr = x_ptr + c * (ixl + iw * iy);
                const float* xc_ptr = x_ptr + c * (ix + iw * iy);
                const float* xr_ptr = x_ptr + c * (ixr + iw * iy);
                const float* xld_ptr = x_ptr + c * (ixl + iw * iyd);
                const float* xd_ptr = x_ptr + c * (ix + iw * iyd);
                const float* xrd_ptr = x_ptr + c * (ixr + iw * iyd);

                float* ylu_ptr = y_ptr + c * (ox + ow * oy);
                float* yru_ptr = ylu_ptr + c;
                float* yld_ptr = ylu_ptr + c * ow;
                float* yrd_ptr = ylu_ptr + c * (ow + 1);

                xlu = _mm_load_ps(xlu_ptr);
                xu = _mm_load_ps(xu_ptr);
                xru = _mm_load_ps(xru_ptr);
                xl = _mm_load_ps(xl_ptr);
                xc = _mm_load_ps(xc_ptr);
                xr = _mm_load_ps(xr_ptr);
                xld = _mm_load_ps(xld_ptr);
                xd = _mm_load_ps(xd_ptr);
                xrd = _mm_load_ps(xrd_ptr);

                __m128x4 y = _mm_linear2d_ps(xlu, xu, xru, xl, xc, xr, xld, xd, xrd);

                _mm_stream_ps(ylu_ptr, y.imm0);
                _mm_stream_ps(yru_ptr, y.imm1);
                _mm_stream_ps(yld_ptr, y.imm2);
                _mm_stream_ps(yrd_ptr, y.imm3);
            }
        }

        x_ptr += c * iw * ih;
        y_ptr += c * ow * oh;
    }

    return SUCCESS;
}

int upsample2d_linear_c5to7(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c <= AVX1_FLOAT_STRIDE || c >= AVX2_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(c);

    __m256 xlu, xu, xru, xl, xc, xr, xld, xd, xrd;

    for (uint i = 0; i < n; i++) {
        for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
            for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);
                const uint iyu = max(iy, 1u) - 1u, iyd = min(iy + 1u, ih - 1u);

                const float* xlu_ptr = x_ptr + c * (ixl + iw * iyu);
                const float* xu_ptr = x_ptr + c * (ix + iw * iyu);
                const float* xru_ptr = x_ptr + c * (ixr + iw * iyu);
                const float* xl_ptr = x_ptr + c * (ixl + iw * iy);
                const float* xc_ptr = x_ptr + c * (ix + iw * iy);
                const float* xr_ptr = x_ptr + c * (ixr + iw * iy);
                const float* xld_ptr = x_ptr + c * (ixl + iw * iyd);
                const float* xd_ptr = x_ptr + c * (ix + iw * iyd);
                const float* xrd_ptr = x_ptr + c * (ixr + iw * iyd);

                float* ylu_ptr = y_ptr + c * (ox + ow * oy);
                float* yru_ptr = ylu_ptr + c;
                float* yld_ptr = ylu_ptr + c * ow;
                float* yrd_ptr = ylu_ptr + c * (ow + 1);

                _mm256_loadu_x1_ps(xlu_ptr, xlu);
                _mm256_loadu_x1_ps(xu_ptr, xu);
                _mm256_loadu_x1_ps(xru_ptr, xru);
                _mm256_loadu_x1_ps(xl_ptr, xl);
                _mm256_loadu_x1_ps(xc_ptr, xc);
                _mm256_loadu_x1_ps(xr_ptr, xr);
                _mm256_loadu_x1_ps(xld_ptr, xld);
                _mm256_loadu_x1_ps(xd_ptr, xd);
                _mm256_loadu_x1_ps(xrd_ptr, xrd);

                __m256x4 y = _mm256_linear2d_ps(xlu, xu, xru, xl, xc, xr, xld, xd, xrd);

                _mm256_maskstore_x1_ps(ylu_ptr, y.imm0, mask);
                _mm256_maskstore_x1_ps(yru_ptr, y.imm1, mask);
                _mm256_maskstore_x1_ps(yld_ptr, y.imm2, mask);
                _mm256_maskstore_x1_ps(yrd_ptr, y.imm3, mask);
            }
        }

        x_ptr += c * iw * ih;
        y_ptr += c * ow * oh;
    }

    return SUCCESS;
}

int upsample2d_linear_c8(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c != AVX2_FLOAT_STRIDE || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 xlu, xu, xru, xl, xc, xr, xld, xd, xrd;

    for (uint i = 0; i < n; i++) {
        for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
            for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);
                const uint iyu = max(iy, 1u) - 1u, iyd = min(iy + 1u, ih - 1u);

                const float* xlu_ptr = x_ptr + c * (ixl + iw * iyu);
                const float* xu_ptr = x_ptr + c * (ix + iw * iyu);
                const float* xru_ptr = x_ptr + c * (ixr + iw * iyu);
                const float* xl_ptr = x_ptr + c * (ixl + iw * iy);
                const float* xc_ptr = x_ptr + c * (ix + iw * iy);
                const float* xr_ptr = x_ptr + c * (ixr + iw * iy);
                const float* xld_ptr = x_ptr + c * (ixl + iw * iyd);
                const float* xd_ptr = x_ptr + c * (ix + iw * iyd);
                const float* xrd_ptr = x_ptr + c * (ixr + iw * iyd);

                float* ylu_ptr = y_ptr + c * (ox + ow * oy);
                float* yru_ptr = ylu_ptr + c;
                float* yld_ptr = ylu_ptr + c * ow;
                float* yrd_ptr = ylu_ptr + c * (ow + 1);

                _mm256_load_x1_ps(xlu_ptr, xlu);
                _mm256_load_x1_ps(xu_ptr, xu);
                _mm256_load_x1_ps(xru_ptr, xru);
                _mm256_load_x1_ps(xl_ptr, xl);
                _mm256_load_x1_ps(xc_ptr, xc);
                _mm256_load_x1_ps(xr_ptr, xr);
                _mm256_load_x1_ps(xld_ptr, xld);
                _mm256_load_x1_ps(xd_ptr, xd);
                _mm256_load_x1_ps(xrd_ptr, xrd);

                __m256x4 y = _mm256_linear2d_ps(xlu, xu, xru, xl, xc, xr, xld, xd, xrd);

                _mm256_stream_x1_ps(ylu_ptr, y.imm0);
                _mm256_stream_x1_ps(yru_ptr, y.imm1);
                _mm256_stream_x1_ps(yld_ptr, y.imm2);
                _mm256_stream_x1_ps(yrd_ptr, y.imm3);
            }
        }

        x_ptr += c * iw * ih;
        y_ptr += c * ow * oh;
    }

    return SUCCESS;
}

int upsample2d_linear_cleq8(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh,
    infloats x_ptr, outfloats y_ptr) {

    if (c == 1) {
        return upsample2d_linear_c1(n, c, iw, ow, ih, oh, x_ptr, y_ptr);
    }
    if (c < AVX1_FLOAT_STRIDE) {
        return upsample2d_linear_c2to3(n, c, iw, ow, ih, oh, x_ptr, y_ptr);
    }
    if (c == AVX1_FLOAT_STRIDE) {
        return upsample2d_linear_c4(n, c, iw, ow, ih, oh, x_ptr, y_ptr);
    }
    if (c < AVX2_FLOAT_STRIDE) {
        return upsample2d_linear_c5to7(n, c, iw, ow, ih, oh, x_ptr, y_ptr);
    }
    if (c == AVX2_FLOAT_STRIDE) {
        return upsample2d_linear_c8(n, c, iw, ow, ih, oh, x_ptr, y_ptr);
    }

    return FAILURE_BADPARAM;
}

#pragma managed

void AvxBlas::Upsample2D::LinearX2(
    UInt32 n, UInt32 c, UInt32 iw, UInt32 ih,
    Array<float>^ x, Array<float>^ y) {

    if (n > MAX_BATCHES) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidBatches);
    }
    if (c > MAX_CHANNELS) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidChannels);
    }
    if (n <= 0 || c <= 0 || iw <= 0 || ih <= 0) {
        return;
    }

    Util::CheckProdOverflow(iw, 2u);
    Util::CheckProdOverflow(ih, 2u);
    UInt32 ow = iw * 2, oh = ih * 2;

    if (ow > MAX_MAP_SIZE || oh > MAX_MAP_SIZE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidDataSize);
    }

    Util::CheckProdOverflow(n, c, iw, ih);
    Util::CheckProdOverflow(n, c, ow, oh);

    Util::CheckLength(n * c * iw * ih, x);
    Util::CheckLength(n * c * ow * oh, y);

    Util::CheckDuplicateArray(x, y);

    const float* x_ptr = (const float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if (c <= AVX2_FLOAT_STRIDE) {
#ifdef _DEBUG
        Console::WriteLine("type leq8");
#endif // _DEBUG

        ret = upsample2d_linear_cleq8(n, c, iw, ow, ih, oh, x_ptr, y_ptr);
    }
    else if ((c & AVX2_FLOAT_REMAIN_MASK) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = upsample2d_linear_aligned(n, c, iw, ow, ih, oh, x_ptr, y_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = upsample2d_linear_unaligned(n, c, iw, ow, ih, oh, x_ptr, y_ptr);
    }
    if (ret != SUCCESS) {
        Util::AssertReturnCode(ret);
        return;
    }

    Constant::Mul(n * c * ow * oh, y, 1.0f / 9.0f, y);
}
