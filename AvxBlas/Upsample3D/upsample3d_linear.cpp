#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_numeric.hpp"
#include "../Inline/inline_loadstore_xn_s.hpp"

#pragma unmanaged

__forceinline floatx4 float_linear3d(
    float xluf, float xuf, float xruf,
    float xlf, float xf, float xrf,
    float xldf, float xdf, float xrdf,
    float xlu, float xu, float xru,
    float xl, float xc, float xr,
    float xld, float xd, float xrd, 
    float xlub, float xub, float xrub,
    float xlb, float xb, float xrb,
    float xldb, float xdb, float xrdb) {

    float xc2 = xc + xc, xc4 = xc2 + xc2, xc8 = xc4 + xc4;

    float xl2 = xl + xl, xl4 = xl2 + xl2;
    float xr2 = xr + xr, xr4 = xr2 + xr2;
    float xu2 = xu + xu, xu4 = xu2 + xu2;
    float xd2 = xd + xd, xd4 = xd2 + xd2;
    float xf2 = xf + xf, xf4 = xf2 + xf2;
    float xb2 = xb + xb, xb4 = xb2 + xb2;

    float xlu2 = xlu + xlu;
    float xru2 = xru + xru;
    float xlb2 = xlb + xlb;
    float xrb2 = xrb + xrb;
    float xuf2 = xuf + xuf;
    float xdf2 = xdf + xdf;
    float xub2 = xub + xub;
    float xdb2 = xdb + xdb;
    float xlf2 = xlf + xlf;
    float xrf2 = xrf + xrf;

    float xlc = xl4 + xc8;
    float xrc = xr4 + xc8;
    float xuc = xu4 + xc8;
    float xdc = xd4 + xc8;
    float xfc = xf4 + xc8;
    float xbc = xb4 + xc8;

    float ylu = xlu + xu2 + xlc;
    float yru = xru + xu2 + xrc;
    float yld = xld + xd2 + xlc;
    float yrd = xrd + xd2 + xrc;

    return floatx4(ylu, yru, yld, yrd);
}

__forceinline __m128x4 _mm_linear3d_ps(
    __m128 xlu, __m128 xu, __m128 xru,
    __m128 xl, __m128 xc, __m128 xr,
    __m128 xld, __m128 xd, __m128 xrd) {

    __m128 xc2 = _mm_add_ps(xc, xc), xc4 = _mm_add_ps(xc2, xc2);
    __m128 xl2 = _mm_add_ps(xl, xl);
    __m128 xr2 = _mm_add_ps(xr, xr);
    __m128 xu2 = _mm_add_ps(xu, xu);
    __m128 xd2 = _mm_add_ps(xd, xd);

    __m128 xlc = _mm_add_ps(xl2, xc4);
    __m128 xrc = _mm_add_ps(xr2, xc4);

    __m128 ylu = _mm_add_ps(_mm_add_ps(xlu, xu2), xlc);
    __m128 yru = _mm_add_ps(_mm_add_ps(xru, xu2), xrc);
    __m128 yld = _mm_add_ps(_mm_add_ps(xld, xd2), xlc);
    __m128 yrd = _mm_add_ps(_mm_add_ps(xrd, xd2), xrc);

    return __m128x4(ylu, yru, yld, yrd);
}

__forceinline __m256x4 _mm256_linear3d_ps(
    __m256 xlu, __m256 xu, __m256 xru,
    __m256 xl, __m256 xc, __m256 xr,
    __m256 xld, __m256 xd, __m256 xrd) {

    __m256 xc2 = _mm256_add_ps(xc, xc), xc4 = _mm256_add_ps(xc2, xc2);
    __m256 xl2 = _mm256_add_ps(xl, xl);
    __m256 xr2 = _mm256_add_ps(xr, xr);
    __m256 xu2 = _mm256_add_ps(xu, xu);
    __m256 xd2 = _mm256_add_ps(xd, xd);

    __m256 xlc = _mm256_add_ps(xl2, xc4);
    __m256 xrc = _mm256_add_ps(xr2, xc4);

    __m256 ylu = _mm256_add_ps(_mm256_add_ps(xlu, xu2), xlc);
    __m256 yru = _mm256_add_ps(_mm256_add_ps(xru, xu2), xrc);
    __m256 yld = _mm256_add_ps(_mm256_add_ps(xld, xd2), xlc);
    __m256 yrd = _mm256_add_ps(_mm256_add_ps(xrd, xd2), xrc);

    return __m256x4(ylu, yru, yld, yrd);
}

int upsample3d_linear_aligned(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh, const uint id, const uint od,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 xlu, xu, xru, xl, xc, xr, xld, xd, xrd;

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0, oz = 0; iz < id; iz++, oz += 2) {
            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
                for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                    const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);
                    const uint iyu = max(iy, 1u) - 1u, iyd = min(iy + 1u, ih - 1u);
                    const uint izf = max(iz, 1u) - 1u, izb = min(iz + 1u, id - 1u);

                    uint r = c;

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
                    float* yru_ptr = y_ptr + c * ((ox + 1) + ow * oy);
                    float* yld_ptr = y_ptr + c * (ox + ow * (oy + 1));
                    float* yrd_ptr = y_ptr + c * ((ox + 1) + ow * (oy + 1));

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

                        __m256x4 y = _mm256_linear3d_ps(xlu, xu, xru, xl, xc, xr, xld, xd, xrd);

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
        }

        x_ptr += c * iw * ih * id;
        y_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int upsample3d_linear_unaligned(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh, const uint id, const uint od,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) == 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(c & AVX2_FLOAT_REMAIN_MASK);

    __m256 xlu, xu, xru, xl, xc, xr, xld, xd, xrd;

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0, oz = 0; iz < id; iz++, oz += 2) {
            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
                for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                    const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);
                    const uint iyu = max(iy, 1u) - 1u, iyd = min(iy + 1u, ih - 1u);
                    const uint izf = max(iz, 1u) - 1u, izb = min(iz + 1u, id - 1u);

                    uint r = c;

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
                    float* yru_ptr = y_ptr + c * ((ox + 1) + ow * oy);
                    float* yld_ptr = y_ptr + c * (ox + ow * (oy + 1));
                    float* yrd_ptr = y_ptr + c * ((ox + 1) + ow * (oy + 1));

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

                        __m256x4 y = _mm256_linear3d_ps(xlu, xu, xru, xl, xc, xr, xld, xd, xrd);

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

                        __m256x4 y = _mm256_linear3d_ps(xlu, xu, xru, xl, xc, xr, xld, xd, xrd);

                        _mm256_maskstore_x1_ps(ylu_ptr, y.imm0, mask);
                        _mm256_maskstore_x1_ps(yru_ptr, y.imm1, mask);
                        _mm256_maskstore_x1_ps(yld_ptr, y.imm2, mask);
                        _mm256_maskstore_x1_ps(yrd_ptr, y.imm3, mask);
                    }
                }
            }
        }

        x_ptr += c * iw * ih * id;
        y_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int upsample3d_linear_c1(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh, const uint id, const uint od,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c != 1) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float xlu, xu, xru, xl, xc, xr, xld, xd, xrd;

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0, oz = 0; iz < id; iz++, oz += 2) {
            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
                for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                    const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);
                    const uint iyu = max(iy, 1u) - 1u, iyd = min(iy + 1u, ih - 1u);
                    const uint izf = max(iz, 1u) - 1u, izb = min(iz + 1u, id - 1u);

                    const float* xlu_ptr = x_ptr + c * (ixl + iw * iyu);
                    const float* xu_ptr = x_ptr + c * (ix + iw * iyu);
                    const float* xru_ptr = x_ptr + c * (ixr + iw * iyu);
                    const float* xl_ptr = x_ptr + c * (ixl + iw * iy);
                    const float* xc_ptr = x_ptr + c * (ix + iw * iy);
                    const float* xr_ptr = x_ptr + c * (ixr + iw * iy);
                    const float* xld_ptr = x_ptr + c * (ixl + iw * iyd);
                    const float* xd_ptr = x_ptr + c * (ix + iw * iyd);
                    const float* xrd_ptr = x_ptr + c * (ixr + iw * iyd);

                    float* ylu_ptr = y_ptr + (ox + ow * oy);
                    float* yru_ptr = y_ptr + ((ox + 1) + ow * oy);
                    float* yld_ptr = y_ptr + (ox + ow * (oy + 1));
                    float* yrd_ptr = y_ptr + ((ox + 1) + ow * (oy + 1));

                    xlu = *xlu_ptr;
                    xu = *xu_ptr;
                    xru = *xru_ptr;
                    xl = *xl_ptr;
                    xc = *xc_ptr;
                    xr = *xr_ptr;
                    xld = *xld_ptr;
                    xd = *xd_ptr;
                    xrd = *xrd_ptr;

                    floatx4 y = float_linear3d(xlu, xu, xru, xl, xc, xr, xld, xd, xrd, xlu, xu, xru, xl, xc, xr, xld, xd, xrd, xlu, xu, xru, xl, xc, xr, xld, xd, xrd);

                    *ylu_ptr = y.imm0;
                    *yru_ptr = y.imm1;
                    *yld_ptr = y.imm2;
                    *yrd_ptr = y.imm3;
                }
            }
        }

        x_ptr += c * iw * ih * id;
        y_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int upsample3d_linear_c2to3(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh, const uint id, const uint od,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c <= 1 && c >= AVX1_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m128i mask = _mm_setmask_ps(c);

    __m128 xlu, xu, xru, xl, xc, xr, xld, xd, xrd;

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0, oz = 0; iz < id; iz++, oz += 2) {
            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
                for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                    const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);
                    const uint iyu = max(iy, 1u) - 1u, iyd = min(iy + 1u, ih - 1u);
                    const uint izf = max(iz, 1u) - 1u, izb = min(iz + 1u, id - 1u);

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
                    float* yru_ptr = y_ptr + c * ((ox + 1) + ow * oy);
                    float* yld_ptr = y_ptr + c * (ox + ow * (oy + 1));
                    float* yrd_ptr = y_ptr + c * ((ox + 1) + ow * (oy + 1));

                    xlu = _mm_loadu_ps(xlu_ptr);
                    xu = _mm_loadu_ps(xu_ptr);
                    xru = _mm_loadu_ps(xru_ptr);
                    xl = _mm_loadu_ps(xl_ptr);
                    xc = _mm_loadu_ps(xc_ptr);
                    xr = _mm_loadu_ps(xr_ptr);
                    xld = _mm_loadu_ps(xld_ptr);
                    xd = _mm_loadu_ps(xd_ptr);
                    xrd = _mm_loadu_ps(xrd_ptr);

                    __m128x4 y = _mm_linear3d_ps(xlu, xu, xru, xl, xc, xr, xld, xd, xrd);

                    _mm_maskstore_ps(ylu_ptr, mask, y.imm0);
                    _mm_maskstore_ps(yru_ptr, mask, y.imm1);
                    _mm_maskstore_ps(yld_ptr, mask, y.imm2);
                    _mm_maskstore_ps(yrd_ptr, mask, y.imm3);
                }
            }
        }

        x_ptr += c * iw * ih * id;
        y_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int upsample3d_linear_c4(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh, const uint id, const uint od,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c != AVX1_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m128 xlu, xu, xru, xl, xc, xr, xld, xd, xrd;

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0, oz = 0; iz < id; iz++, oz += 2) {
            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
                for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                    const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);
                    const uint iyu = max(iy, 1u) - 1u, iyd = min(iy + 1u, ih - 1u);
                    const uint izf = max(iz, 1u) - 1u, izb = min(iz + 1u, id - 1u);

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
                    float* yru_ptr = y_ptr + c * ((ox + 1) + ow * oy);
                    float* yld_ptr = y_ptr + c * (ox + ow * (oy + 1));
                    float* yrd_ptr = y_ptr + c * ((ox + 1) + ow * (oy + 1));

                    xlu = _mm_load_ps(xlu_ptr);
                    xu = _mm_load_ps(xu_ptr);
                    xru = _mm_load_ps(xru_ptr);
                    xl = _mm_load_ps(xl_ptr);
                    xc = _mm_load_ps(xc_ptr);
                    xr = _mm_load_ps(xr_ptr);
                    xld = _mm_load_ps(xld_ptr);
                    xd = _mm_load_ps(xd_ptr);
                    xrd = _mm_load_ps(xrd_ptr);

                    __m128x4 y = _mm_linear3d_ps(xlu, xu, xru, xl, xc, xr, xld, xd, xrd);

                    _mm_stream_ps(ylu_ptr, y.imm0);
                    _mm_stream_ps(yru_ptr, y.imm1);
                    _mm_stream_ps(yld_ptr, y.imm2);
                    _mm_stream_ps(yrd_ptr, y.imm3);
                }
            }
        }

        x_ptr += c * iw * ih * id;
        y_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int upsample3d_linear_c5to7(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh, const uint id, const uint od,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c <= AVX1_FLOAT_STRIDE || c >= AVX2_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(c);

    __m256 xlu, xu, xru, xl, xc, xr, xld, xd, xrd;

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0, oz = 0; iz < id; iz++, oz += 2) {
            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
                for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                    const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);
                    const uint iyu = max(iy, 1u) - 1u, iyd = min(iy + 1u, ih - 1u);
                    const uint izf = max(iz, 1u) - 1u, izb = min(iz + 1u, id - 1u);

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
                    float* yru_ptr = y_ptr + c * ((ox + 1) + ow * oy);
                    float* yld_ptr = y_ptr + c * (ox + ow * (oy + 1));
                    float* yrd_ptr = y_ptr + c * ((ox + 1) + ow * (oy + 1));

                    _mm256_loadu_x1_ps(xlu_ptr, xlu);
                    _mm256_loadu_x1_ps(xu_ptr, xu);
                    _mm256_loadu_x1_ps(xru_ptr, xru);
                    _mm256_loadu_x1_ps(xl_ptr, xl);
                    _mm256_loadu_x1_ps(xc_ptr, xc);
                    _mm256_loadu_x1_ps(xr_ptr, xr);
                    _mm256_loadu_x1_ps(xld_ptr, xld);
                    _mm256_loadu_x1_ps(xd_ptr, xd);
                    _mm256_loadu_x1_ps(xrd_ptr, xrd);

                    __m256x4 y = _mm256_linear3d_ps(xlu, xu, xru, xl, xc, xr, xld, xd, xrd);

                    _mm256_maskstore_x1_ps(ylu_ptr, y.imm0, mask);
                    _mm256_maskstore_x1_ps(yru_ptr, y.imm1, mask);
                    _mm256_maskstore_x1_ps(yld_ptr, y.imm2, mask);
                    _mm256_maskstore_x1_ps(yrd_ptr, y.imm3, mask);
                }
            }
        }

        x_ptr += c * iw * ih * id;
        y_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int upsample3d_linear_c8(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh, const uint id, const uint od,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c != AVX2_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 xlu, xu, xru, xl, xc, xr, xld, xd, xrd;

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0, oz = 0; iz < id; iz++, oz += 2) {
            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
                for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                    const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);
                    const uint iyu = max(iy, 1u) - 1u, iyd = min(iy + 1u, ih - 1u);
                    const uint izf = max(iz, 1u) - 1u, izb = min(iz + 1u, id - 1u);

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
                    float* yru_ptr = y_ptr + c * ((ox + 1) + ow * oy);
                    float* yld_ptr = y_ptr + c * (ox + ow * (oy + 1));
                    float* yrd_ptr = y_ptr + c * ((ox + 1) + ow * (oy + 1));

                    _mm256_load_x1_ps(xlu_ptr, xlu);
                    _mm256_load_x1_ps(xu_ptr, xu);
                    _mm256_load_x1_ps(xru_ptr, xru);
                    _mm256_load_x1_ps(xl_ptr, xl);
                    _mm256_load_x1_ps(xc_ptr, xc);
                    _mm256_load_x1_ps(xr_ptr, xr);
                    _mm256_load_x1_ps(xld_ptr, xld);
                    _mm256_load_x1_ps(xd_ptr, xd);
                    _mm256_load_x1_ps(xrd_ptr, xrd);

                    __m256x4 y = _mm256_linear3d_ps(xlu, xu, xru, xl, xc, xr, xld, xd, xrd);

                    _mm256_stream_x1_ps(ylu_ptr, y.imm0);
                    _mm256_stream_x1_ps(yru_ptr, y.imm1);
                    _mm256_stream_x1_ps(yld_ptr, y.imm2);
                    _mm256_stream_x1_ps(yrd_ptr, y.imm3);
                }
            }
        }

        x_ptr += c * iw * ih * id;
        y_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

int upsample3d_linear_cleq8(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh, const uint id, const uint od,
    infloats x_ptr, outfloats y_ptr) {

    if (c == 1) {
        return upsample3d_linear_c1(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
    }
    if (c < AVX1_FLOAT_STRIDE) {
        return upsample3d_linear_c2to3(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
    }
    if (c == AVX1_FLOAT_STRIDE) {
        return upsample3d_linear_c4(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
    }
    if (c < AVX2_FLOAT_STRIDE) {
        return upsample3d_linear_c5to7(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
    }
    if (c == AVX2_FLOAT_STRIDE) {
        return upsample3d_linear_c8(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
    }

    return FAILURE_BADPARAM;
}

#pragma managed

void AvxBlas::Upsample3D::LinearX2(
    UInt32 n, UInt32 c, UInt32 iw, UInt32 ih, UInt32 id,
    Array<float>^ x, Array<float>^ y) {

    if (n > MAX_BATCHES) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidBatches);
    }
    if (c > MAX_CHANNELS) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidChannels);
    }
    if (n <= 0 || c <= 0 || iw <= 0 || ih <= 0 || id <= 0) {
        return;
    }

    Util::CheckProdOverflow(iw, 2u);
    Util::CheckProdOverflow(ih, 2u);
    Util::CheckProdOverflow(id, 2u);
    UInt32 ow = iw * 2, oh = ih * 2, od = id * 2;

    if (ow > MAX_MAP_SIZE || oh > MAX_MAP_SIZE || od > MAX_MAP_SIZE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidDataSize);
    }

    Util::CheckProdOverflow(n, c, iw, ih, id);
    Util::CheckProdOverflow(n, c, ow, oh, od);

    Util::CheckLength(n * c * iw * ih * id, x);
    Util::CheckLength(n * c * ow * oh * od, y);

    Util::CheckDuplicateArray(x, y);

    const float* x_ptr = (const float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if (c <= AVX2_FLOAT_STRIDE) {
#ifdef _DEBUG
        Console::WriteLine("type leq8");
#endif // _DEBUG

        ret = upsample3d_linear_cleq8(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
    }
    else if ((c & AVX2_FLOAT_REMAIN_MASK) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = upsample3d_linear_aligned(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = upsample3d_linear_unaligned(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
    }
    if (ret != SUCCESS) {
        Util::AssertReturnCode(ret);
        return;
    }

    Constant::Mul(n * c * ow * oh * od, y, 1.0f / 27.0f, y);
}
