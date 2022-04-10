#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_numeric.hpp"
#include "../Inline/inline_loadstore_xn_s.hpp"

#pragma unmanaged

__forceinline floatx8 float_linear3d(
    float xluf, float xuf, float xruf,
    float xlf, float xf, float xrf,
    float xldf, float xdf, float xrdf,
    float xlu, float xu, float xru,
    float xl, float xc, float xr,
    float xld, float xd, float xrd,
    float xlub, float xub, float xrub,
    float xlb, float xb, float xrb,
    float xldb, float xdb, float xrdb) {

    float wc2 = xc + xc;

    float wl2c = (xl + xl) + xc;
    float wr2c = (xr + xr) + xc;
    float wu2c = (xu + xu) + xc;
    float wd2c = (xd + xd) + xc;
    float wf2c = (xf + xf) + xc;
    float wb2c = (xb + xb) + xc;

    float wlu2l2u2c2 = (xlu + xlu) + (wl2c + wu2c);
    float wld2l2d2c2 = (xld + xld) + (wl2c + wd2c);
    float wlf2l2f2c2 = (xlf + xlf) + (wl2c + wf2c);
    float wlb2l2b2c2 = (xlb + xlb) + (wl2c + wb2c);
    float wru2r2u2c2 = (xru + xru) + (wr2c + wu2c);
    float wrd2r2d2c2 = (xrd + xrd) + (wr2c + wd2c);
    float wrf2r2f2c2 = (xrf + xrf) + (wr2c + wf2c);
    float wrb2r2b2c2 = (xrb + xrb) + (wr2c + wb2c);
    float wuf2u2f2c2 = (xuf + xuf) + (wu2c + wf2c);
    float wub2u2b2c2 = (xub + xub) + (wu2c + wb2c);
    float wdf2d2f2c2 = (xdf + xdf) + (wd2c + wf2c);
    float wdb2d2b2c2 = (xdb + xdb) + (wd2c + wb2c);

    float yluf = (xluf + wc2) + (wlu2l2u2c2 + (wlf2l2f2c2 + wuf2u2f2c2));
    float yruf = (xruf + wc2) + (wru2r2u2c2 + (wrf2r2f2c2 + wuf2u2f2c2));
    float yldf = (xldf + wc2) + (wld2l2d2c2 + (wlf2l2f2c2 + wdf2d2f2c2));
    float yrdf = (xrdf + wc2) + (wrd2r2d2c2 + (wrf2r2f2c2 + wdf2d2f2c2));
    float ylub = (xlub + wc2) + (wlu2l2u2c2 + (wlb2l2b2c2 + wub2u2b2c2));
    float yrub = (xrub + wc2) + (wru2r2u2c2 + (wrb2r2b2c2 + wub2u2b2c2));
    float yldb = (xldb + wc2) + (wld2l2d2c2 + (wlb2l2b2c2 + wdb2d2b2c2));
    float yrdb = (xrdb + wc2) + (wrd2r2d2c2 + (wrb2r2b2c2 + wdb2d2b2c2));

    return floatx8(yluf, yruf, yldf, yrdf, ylub, yrub, yldb, yrdb);
}

__forceinline __m128x8 _mm_linear3d_ps(
    __m128 xluf, __m128 xuf, __m128 xruf,
    __m128 xlf, __m128 xf, __m128 xrf,
    __m128 xldf, __m128 xdf, __m128 xrdf,
    __m128 xlu, __m128 xu, __m128 xru,
    __m128 xl, __m128 xc, __m128 xr,
    __m128 xld, __m128 xd, __m128 xrd,
    __m128 xlub, __m128 xub, __m128 xrub,
    __m128 xlb, __m128 xb, __m128 xrb,
    __m128 xldb, __m128 xdb, __m128 xrdb) {

    __m128 wc2 = _mm_add_ps(xc, xc);

    __m128 wl2c = _mm_add_ps(_mm_add_ps(xl, xl), xc);
    __m128 wr2c = _mm_add_ps(_mm_add_ps(xr, xr), xc);
    __m128 wu2c = _mm_add_ps(_mm_add_ps(xu, xu), xc);
    __m128 wd2c = _mm_add_ps(_mm_add_ps(xd, xd), xc);
    __m128 wf2c = _mm_add_ps(_mm_add_ps(xf, xf), xc);
    __m128 wb2c = _mm_add_ps(_mm_add_ps(xb, xb), xc);

    __m128 wlu2l2u2c2 = _mm_add_ps(_mm_add_ps(xlu, xlu), _mm_add_ps(wl2c, wu2c));
    __m128 wld2l2d2c2 = _mm_add_ps(_mm_add_ps(xld, xld), _mm_add_ps(wl2c, wd2c));
    __m128 wlf2l2f2c2 = _mm_add_ps(_mm_add_ps(xlf, xlf), _mm_add_ps(wl2c, wf2c));
    __m128 wlb2l2b2c2 = _mm_add_ps(_mm_add_ps(xlb, xlb), _mm_add_ps(wl2c, wb2c));
    __m128 wru2r2u2c2 = _mm_add_ps(_mm_add_ps(xru, xru), _mm_add_ps(wr2c, wu2c));
    __m128 wrd2r2d2c2 = _mm_add_ps(_mm_add_ps(xrd, xrd), _mm_add_ps(wr2c, wd2c));
    __m128 wrf2r2f2c2 = _mm_add_ps(_mm_add_ps(xrf, xrf), _mm_add_ps(wr2c, wf2c));
    __m128 wrb2r2b2c2 = _mm_add_ps(_mm_add_ps(xrb, xrb), _mm_add_ps(wr2c, wb2c));
    __m128 wuf2u2f2c2 = _mm_add_ps(_mm_add_ps(xuf, xuf), _mm_add_ps(wu2c, wf2c));
    __m128 wub2u2b2c2 = _mm_add_ps(_mm_add_ps(xub, xub), _mm_add_ps(wu2c, wb2c));
    __m128 wdf2d2f2c2 = _mm_add_ps(_mm_add_ps(xdf, xdf), _mm_add_ps(wd2c, wf2c));
    __m128 wdb2d2b2c2 = _mm_add_ps(_mm_add_ps(xdb, xdb), _mm_add_ps(wd2c, wb2c));

    __m128 yluf = _mm_add_ps(_mm_add_ps(xluf, wc2), _mm_add_ps(wlu2l2u2c2, _mm_add_ps(wlf2l2f2c2, wuf2u2f2c2)));
    __m128 yruf = _mm_add_ps(_mm_add_ps(xruf, wc2), _mm_add_ps(wru2r2u2c2, _mm_add_ps(wrf2r2f2c2, wuf2u2f2c2)));
    __m128 yldf = _mm_add_ps(_mm_add_ps(xldf, wc2), _mm_add_ps(wld2l2d2c2, _mm_add_ps(wlf2l2f2c2, wdf2d2f2c2)));
    __m128 yrdf = _mm_add_ps(_mm_add_ps(xrdf, wc2), _mm_add_ps(wrd2r2d2c2, _mm_add_ps(wrf2r2f2c2, wdf2d2f2c2)));
    __m128 ylub = _mm_add_ps(_mm_add_ps(xlub, wc2), _mm_add_ps(wlu2l2u2c2, _mm_add_ps(wlb2l2b2c2, wub2u2b2c2)));
    __m128 yrub = _mm_add_ps(_mm_add_ps(xrub, wc2), _mm_add_ps(wru2r2u2c2, _mm_add_ps(wrb2r2b2c2, wub2u2b2c2)));
    __m128 yldb = _mm_add_ps(_mm_add_ps(xldb, wc2), _mm_add_ps(wld2l2d2c2, _mm_add_ps(wlb2l2b2c2, wdb2d2b2c2)));
    __m128 yrdb = _mm_add_ps(_mm_add_ps(xrdb, wc2), _mm_add_ps(wrd2r2d2c2, _mm_add_ps(wrb2r2b2c2, wdb2d2b2c2)));

    return __m128x8(yluf, yruf, yldf, yrdf, ylub, yrub, yldb, yrdb);
}

__forceinline __m256x8 _mm256_linear3d_ps(
    __m256 xluf, __m256 xuf, __m256 xruf,
    __m256 xlf, __m256 xf, __m256 xrf,
    __m256 xldf, __m256 xdf, __m256 xrdf,
    __m256 xlu, __m256 xu, __m256 xru,
    __m256 xl, __m256 xc, __m256 xr,
    __m256 xld, __m256 xd, __m256 xrd,
    __m256 xlub, __m256 xub, __m256 xrub,
    __m256 xlb, __m256 xb, __m256 xrb,
    __m256 xldb, __m256 xdb, __m256 xrdb) {

    __m256 wc2 = _mm256_add_ps(xc, xc);

    __m256 wl2c = _mm256_add_ps(_mm256_add_ps(xl, xl), xc);
    __m256 wr2c = _mm256_add_ps(_mm256_add_ps(xr, xr), xc);
    __m256 wu2c = _mm256_add_ps(_mm256_add_ps(xu, xu), xc);
    __m256 wd2c = _mm256_add_ps(_mm256_add_ps(xd, xd), xc);
    __m256 wf2c = _mm256_add_ps(_mm256_add_ps(xf, xf), xc);
    __m256 wb2c = _mm256_add_ps(_mm256_add_ps(xb, xb), xc);

    __m256 wlu2l2u2c2 = _mm256_add_ps(_mm256_add_ps(xlu, xlu), _mm256_add_ps(wl2c, wu2c));
    __m256 wld2l2d2c2 = _mm256_add_ps(_mm256_add_ps(xld, xld), _mm256_add_ps(wl2c, wd2c));
    __m256 wlf2l2f2c2 = _mm256_add_ps(_mm256_add_ps(xlf, xlf), _mm256_add_ps(wl2c, wf2c));
    __m256 wlb2l2b2c2 = _mm256_add_ps(_mm256_add_ps(xlb, xlb), _mm256_add_ps(wl2c, wb2c));
    __m256 wru2r2u2c2 = _mm256_add_ps(_mm256_add_ps(xru, xru), _mm256_add_ps(wr2c, wu2c));
    __m256 wrd2r2d2c2 = _mm256_add_ps(_mm256_add_ps(xrd, xrd), _mm256_add_ps(wr2c, wd2c));
    __m256 wrf2r2f2c2 = _mm256_add_ps(_mm256_add_ps(xrf, xrf), _mm256_add_ps(wr2c, wf2c));
    __m256 wrb2r2b2c2 = _mm256_add_ps(_mm256_add_ps(xrb, xrb), _mm256_add_ps(wr2c, wb2c));
    __m256 wuf2u2f2c2 = _mm256_add_ps(_mm256_add_ps(xuf, xuf), _mm256_add_ps(wu2c, wf2c));
    __m256 wub2u2b2c2 = _mm256_add_ps(_mm256_add_ps(xub, xub), _mm256_add_ps(wu2c, wb2c));
    __m256 wdf2d2f2c2 = _mm256_add_ps(_mm256_add_ps(xdf, xdf), _mm256_add_ps(wd2c, wf2c));
    __m256 wdb2d2b2c2 = _mm256_add_ps(_mm256_add_ps(xdb, xdb), _mm256_add_ps(wd2c, wb2c));

    __m256 yluf = _mm256_add_ps(_mm256_add_ps(xluf, wc2), _mm256_add_ps(wlu2l2u2c2, _mm256_add_ps(wlf2l2f2c2, wuf2u2f2c2)));
    __m256 yruf = _mm256_add_ps(_mm256_add_ps(xruf, wc2), _mm256_add_ps(wru2r2u2c2, _mm256_add_ps(wrf2r2f2c2, wuf2u2f2c2)));
    __m256 yldf = _mm256_add_ps(_mm256_add_ps(xldf, wc2), _mm256_add_ps(wld2l2d2c2, _mm256_add_ps(wlf2l2f2c2, wdf2d2f2c2)));
    __m256 yrdf = _mm256_add_ps(_mm256_add_ps(xrdf, wc2), _mm256_add_ps(wrd2r2d2c2, _mm256_add_ps(wrf2r2f2c2, wdf2d2f2c2)));
    __m256 ylub = _mm256_add_ps(_mm256_add_ps(xlub, wc2), _mm256_add_ps(wlu2l2u2c2, _mm256_add_ps(wlb2l2b2c2, wub2u2b2c2)));
    __m256 yrub = _mm256_add_ps(_mm256_add_ps(xrub, wc2), _mm256_add_ps(wru2r2u2c2, _mm256_add_ps(wrb2r2b2c2, wub2u2b2c2)));
    __m256 yldb = _mm256_add_ps(_mm256_add_ps(xldb, wc2), _mm256_add_ps(wld2l2d2c2, _mm256_add_ps(wlb2l2b2c2, wdb2d2b2c2)));
    __m256 yrdb = _mm256_add_ps(_mm256_add_ps(xrdb, wc2), _mm256_add_ps(wrd2r2d2c2, _mm256_add_ps(wrb2r2b2c2, wdb2d2b2c2)));

    return __m256x8(yluf, yruf, yldf, yrdf, ylub, yrub, yldb, yrdb);
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

    __m256 xluf, xuf, xruf, xlf, xf, xrf, xldf, xdf, xrdf;
    __m256 xlu, xu, xru, xl, xc, xr, xld, xd, xrd;
    __m256 xlub, xub, xrub, xlb, xb, xrb, xldb, xdb, xrdb;

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0, oz = 0; iz < id; iz++, oz += 2) {
            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
                for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                    const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);
                    const uint iyu = max(iy, 1u) - 1u, iyd = min(iy + 1u, ih - 1u);
                    const uint izf = max(iz, 1u) - 1u, izb = min(iz + 1u, id - 1u);

                    uint r = c;

                    const float* xluf_ptr = x_ptr + c * (ixl + iw * (iyu + ih * izf));
                    const float* xuf_ptr = x_ptr + c * (ix + iw * (iyu + ih * izf));
                    const float* xruf_ptr = x_ptr + c * (ixr + iw * (iyu + ih * izf));
                    const float* xlf_ptr = x_ptr + c * (ixl + iw * (iy + ih * izf));
                    const float* xf_ptr = x_ptr + c * (ix + iw * (iy + ih * izf));
                    const float* xrf_ptr = x_ptr + c * (ixr + iw * (iy + ih * izf));
                    const float* xldf_ptr = x_ptr + c * (ixl + iw * (iyd + ih * izf));
                    const float* xdf_ptr = x_ptr + c * (ix + iw * (iyd + ih * izf));
                    const float* xrdf_ptr = x_ptr + c * (ixr + iw * (iyd + ih * izf));

                    const float* xlu_ptr = x_ptr + c * (ixl + iw * (iyu + ih * iz));
                    const float* xu_ptr = x_ptr + c * (ix + iw * (iyu + ih * iz));
                    const float* xru_ptr = x_ptr + c * (ixr + iw * (iyu + ih * iz));
                    const float* xl_ptr = x_ptr + c * (ixl + iw * (iy + ih * iz));
                    const float* xc_ptr = x_ptr + c * (ix + iw * (iy + ih * iz));
                    const float* xr_ptr = x_ptr + c * (ixr + iw * (iy + ih * iz));
                    const float* xld_ptr = x_ptr + c * (ixl + iw * (iyd + ih * iz));
                    const float* xd_ptr = x_ptr + c * (ix + iw * (iyd + ih * iz));
                    const float* xrd_ptr = x_ptr + c * (ixr + iw * (iyd + ih * iz));

                    const float* xlub_ptr = x_ptr + c * (ixl + iw * (iyu + ih * izb));
                    const float* xub_ptr = x_ptr + c * (ix + iw * (iyu + ih * izb));
                    const float* xrub_ptr = x_ptr + c * (ixr + iw * (iyu + ih * izb));
                    const float* xlb_ptr = x_ptr + c * (ixl + iw * (iy + ih * izb));
                    const float* xb_ptr = x_ptr + c * (ix + iw * (iy + ih * izb));
                    const float* xrb_ptr = x_ptr + c * (ixr + iw * (iy + ih * izb));
                    const float* xldb_ptr = x_ptr + c * (ixl + iw * (iyd + ih * izb));
                    const float* xdb_ptr = x_ptr + c * (ix + iw * (iyd + ih * izb));
                    const float* xrdb_ptr = x_ptr + c * (ixr + iw * (iyd + ih * izb));

                    float* yluf_ptr = y_ptr + c * (ox + ow * (oy + oh * oz));
                    float* yruf_ptr = yluf_ptr + c;
                    float* yldf_ptr = yluf_ptr + c * ow;
                    float* yrdf_ptr = yluf_ptr + c * (ow + 1);
                    float* ylub_ptr = yluf_ptr + c * (ow * oh);
                    float* yrub_ptr = yruf_ptr + c * (ow * oh);
                    float* yldb_ptr = yldf_ptr + c * (ow * oh);
                    float* yrdb_ptr = yrdf_ptr + c * (ow * oh);

                    while (r >= AVX2_FLOAT_STRIDE) {
                        _mm256_load_x1_ps(xluf_ptr, xluf);
                        _mm256_load_x1_ps(xuf_ptr, xuf);
                        _mm256_load_x1_ps(xruf_ptr, xruf);
                        _mm256_load_x1_ps(xlf_ptr, xlf);
                        _mm256_load_x1_ps(xf_ptr, xf);
                        _mm256_load_x1_ps(xrf_ptr, xrf);
                        _mm256_load_x1_ps(xldf_ptr, xldf);
                        _mm256_load_x1_ps(xdf_ptr, xdf);
                        _mm256_load_x1_ps(xrdf_ptr, xrdf);

                        _mm256_load_x1_ps(xlu_ptr, xlu);
                        _mm256_load_x1_ps(xu_ptr, xu);
                        _mm256_load_x1_ps(xru_ptr, xru);
                        _mm256_load_x1_ps(xl_ptr, xl);
                        _mm256_load_x1_ps(xc_ptr, xc);
                        _mm256_load_x1_ps(xr_ptr, xr);
                        _mm256_load_x1_ps(xld_ptr, xld);
                        _mm256_load_x1_ps(xd_ptr, xd);
                        _mm256_load_x1_ps(xrd_ptr, xrd);

                        _mm256_load_x1_ps(xlub_ptr, xlub);
                        _mm256_load_x1_ps(xub_ptr, xub);
                        _mm256_load_x1_ps(xrub_ptr, xrub);
                        _mm256_load_x1_ps(xlb_ptr, xlb);
                        _mm256_load_x1_ps(xb_ptr, xb);
                        _mm256_load_x1_ps(xrb_ptr, xrb);
                        _mm256_load_x1_ps(xldb_ptr, xldb);
                        _mm256_load_x1_ps(xdb_ptr, xdb);
                        _mm256_load_x1_ps(xrdb_ptr, xrdb);

                        __m256x8 y = _mm256_linear3d_ps(
                            xluf, xuf, xruf, xlf, xf, xrf, xldf, xdf, xrdf,
                            xlu, xu, xru, xl, xc, xr, xld, xd, xrd,
                            xlub, xub, xrub, xlb, xb, xrb, xldb, xdb, xrdb
                        );

                        _mm256_stream_x1_ps(yluf_ptr, y.imm0);
                        _mm256_stream_x1_ps(yruf_ptr, y.imm1);
                        _mm256_stream_x1_ps(yldf_ptr, y.imm2);
                        _mm256_stream_x1_ps(yrdf_ptr, y.imm3);
                        _mm256_stream_x1_ps(ylub_ptr, y.imm4);
                        _mm256_stream_x1_ps(yrub_ptr, y.imm5);
                        _mm256_stream_x1_ps(yldb_ptr, y.imm6);
                        _mm256_stream_x1_ps(yrdb_ptr, y.imm7);

                        xluf_ptr += AVX2_FLOAT_STRIDE;
                        xuf_ptr += AVX2_FLOAT_STRIDE;
                        xruf_ptr += AVX2_FLOAT_STRIDE;
                        xlf_ptr += AVX2_FLOAT_STRIDE;
                        xf_ptr += AVX2_FLOAT_STRIDE;
                        xrf_ptr += AVX2_FLOAT_STRIDE;
                        xldf_ptr += AVX2_FLOAT_STRIDE;
                        xdf_ptr += AVX2_FLOAT_STRIDE;
                        xrdf_ptr += AVX2_FLOAT_STRIDE;

                        xlu_ptr += AVX2_FLOAT_STRIDE;
                        xu_ptr += AVX2_FLOAT_STRIDE;
                        xru_ptr += AVX2_FLOAT_STRIDE;
                        xl_ptr += AVX2_FLOAT_STRIDE;
                        xc_ptr += AVX2_FLOAT_STRIDE;
                        xr_ptr += AVX2_FLOAT_STRIDE;
                        xld_ptr += AVX2_FLOAT_STRIDE;
                        xd_ptr += AVX2_FLOAT_STRIDE;
                        xrd_ptr += AVX2_FLOAT_STRIDE;

                        xlub_ptr += AVX2_FLOAT_STRIDE;
                        xub_ptr += AVX2_FLOAT_STRIDE;
                        xrub_ptr += AVX2_FLOAT_STRIDE;
                        xlb_ptr += AVX2_FLOAT_STRIDE;
                        xb_ptr += AVX2_FLOAT_STRIDE;
                        xrb_ptr += AVX2_FLOAT_STRIDE;
                        xldb_ptr += AVX2_FLOAT_STRIDE;
                        xdb_ptr += AVX2_FLOAT_STRIDE;
                        xrdb_ptr += AVX2_FLOAT_STRIDE;

                        yluf_ptr += AVX2_FLOAT_STRIDE;
                        yruf_ptr += AVX2_FLOAT_STRIDE;
                        yldf_ptr += AVX2_FLOAT_STRIDE;
                        yrdf_ptr += AVX2_FLOAT_STRIDE;
                        ylub_ptr += AVX2_FLOAT_STRIDE;
                        yrub_ptr += AVX2_FLOAT_STRIDE;
                        yldb_ptr += AVX2_FLOAT_STRIDE;
                        yrdb_ptr += AVX2_FLOAT_STRIDE;

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

    __m256 xluf, xuf, xruf, xlf, xf, xrf, xldf, xdf, xrdf;
    __m256 xlu, xu, xru, xl, xc, xr, xld, xd, xrd;
    __m256 xlub, xub, xrub, xlb, xb, xrb, xldb, xdb, xrdb;

    for (uint i = 0; i < n; i++) {
        for (uint iz = 0, oz = 0; iz < id; iz++, oz += 2) {
            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
                for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                    const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);
                    const uint iyu = max(iy, 1u) - 1u, iyd = min(iy + 1u, ih - 1u);
                    const uint izf = max(iz, 1u) - 1u, izb = min(iz + 1u, id - 1u);

                    uint r = c;

                    const float* xluf_ptr = x_ptr + c * (ixl + iw * (iyu + ih * izf));
                    const float* xuf_ptr = x_ptr + c * (ix + iw * (iyu + ih * izf));
                    const float* xruf_ptr = x_ptr + c * (ixr + iw * (iyu + ih * izf));
                    const float* xlf_ptr = x_ptr + c * (ixl + iw * (iy + ih * izf));
                    const float* xf_ptr = x_ptr + c * (ix + iw * (iy + ih * izf));
                    const float* xrf_ptr = x_ptr + c * (ixr + iw * (iy + ih * izf));
                    const float* xldf_ptr = x_ptr + c * (ixl + iw * (iyd + ih * izf));
                    const float* xdf_ptr = x_ptr + c * (ix + iw * (iyd + ih * izf));
                    const float* xrdf_ptr = x_ptr + c * (ixr + iw * (iyd + ih * izf));

                    const float* xlu_ptr = x_ptr + c * (ixl + iw * (iyu + ih * iz));
                    const float* xu_ptr = x_ptr + c * (ix + iw * (iyu + ih * iz));
                    const float* xru_ptr = x_ptr + c * (ixr + iw * (iyu + ih * iz));
                    const float* xl_ptr = x_ptr + c * (ixl + iw * (iy + ih * iz));
                    const float* xc_ptr = x_ptr + c * (ix + iw * (iy + ih * iz));
                    const float* xr_ptr = x_ptr + c * (ixr + iw * (iy + ih * iz));
                    const float* xld_ptr = x_ptr + c * (ixl + iw * (iyd + ih * iz));
                    const float* xd_ptr = x_ptr + c * (ix + iw * (iyd + ih * iz));
                    const float* xrd_ptr = x_ptr + c * (ixr + iw * (iyd + ih * iz));

                    const float* xlub_ptr = x_ptr + c * (ixl + iw * (iyu + ih * izb));
                    const float* xub_ptr = x_ptr + c * (ix + iw * (iyu + ih * izb));
                    const float* xrub_ptr = x_ptr + c * (ixr + iw * (iyu + ih * izb));
                    const float* xlb_ptr = x_ptr + c * (ixl + iw * (iy + ih * izb));
                    const float* xb_ptr = x_ptr + c * (ix + iw * (iy + ih * izb));
                    const float* xrb_ptr = x_ptr + c * (ixr + iw * (iy + ih * izb));
                    const float* xldb_ptr = x_ptr + c * (ixl + iw * (iyd + ih * izb));
                    const float* xdb_ptr = x_ptr + c * (ix + iw * (iyd + ih * izb));
                    const float* xrdb_ptr = x_ptr + c * (ixr + iw * (iyd + ih * izb));

                    float* yluf_ptr = y_ptr + c * (ox + ow * (oy + oh * oz));
                    float* yruf_ptr = yluf_ptr + c;
                    float* yldf_ptr = yluf_ptr + c * ow;
                    float* yrdf_ptr = yluf_ptr + c * (ow + 1);
                    float* ylub_ptr = yluf_ptr + c * (ow * oh);
                    float* yrub_ptr = yruf_ptr + c * (ow * oh);
                    float* yldb_ptr = yldf_ptr + c * (ow * oh);
                    float* yrdb_ptr = yrdf_ptr + c * (ow * oh);

                    while (r >= AVX2_FLOAT_STRIDE) {
                        _mm256_loadu_x1_ps(xluf_ptr, xluf);
                        _mm256_loadu_x1_ps(xuf_ptr, xuf);
                        _mm256_loadu_x1_ps(xruf_ptr, xruf);
                        _mm256_loadu_x1_ps(xlf_ptr, xlf);
                        _mm256_loadu_x1_ps(xf_ptr, xf);
                        _mm256_loadu_x1_ps(xrf_ptr, xrf);
                        _mm256_loadu_x1_ps(xldf_ptr, xldf);
                        _mm256_loadu_x1_ps(xdf_ptr, xdf);
                        _mm256_loadu_x1_ps(xrdf_ptr, xrdf);

                        _mm256_loadu_x1_ps(xlu_ptr, xlu);
                        _mm256_loadu_x1_ps(xu_ptr, xu);
                        _mm256_loadu_x1_ps(xru_ptr, xru);
                        _mm256_loadu_x1_ps(xl_ptr, xl);
                        _mm256_loadu_x1_ps(xc_ptr, xc);
                        _mm256_loadu_x1_ps(xr_ptr, xr);
                        _mm256_loadu_x1_ps(xld_ptr, xld);
                        _mm256_loadu_x1_ps(xd_ptr, xd);
                        _mm256_loadu_x1_ps(xrd_ptr, xrd);

                        _mm256_loadu_x1_ps(xlub_ptr, xlub);
                        _mm256_loadu_x1_ps(xub_ptr, xub);
                        _mm256_loadu_x1_ps(xrub_ptr, xrub);
                        _mm256_loadu_x1_ps(xlb_ptr, xlb);
                        _mm256_loadu_x1_ps(xb_ptr, xb);
                        _mm256_loadu_x1_ps(xrb_ptr, xrb);
                        _mm256_loadu_x1_ps(xldb_ptr, xldb);
                        _mm256_loadu_x1_ps(xdb_ptr, xdb);
                        _mm256_loadu_x1_ps(xrdb_ptr, xrdb);

                        __m256x8 y = _mm256_linear3d_ps(
                            xluf, xuf, xruf, xlf, xf, xrf, xldf, xdf, xrdf,
                            xlu, xu, xru, xl, xc, xr, xld, xd, xrd,
                            xlub, xub, xrub, xlb, xb, xrb, xldb, xdb, xrdb
                        );

                        _mm256_storeu_x1_ps(yluf_ptr, y.imm0);
                        _mm256_storeu_x1_ps(yruf_ptr, y.imm1);
                        _mm256_storeu_x1_ps(yldf_ptr, y.imm2);
                        _mm256_storeu_x1_ps(yrdf_ptr, y.imm3);
                        _mm256_storeu_x1_ps(ylub_ptr, y.imm4);
                        _mm256_storeu_x1_ps(yrub_ptr, y.imm5);
                        _mm256_storeu_x1_ps(yldb_ptr, y.imm6);
                        _mm256_storeu_x1_ps(yrdb_ptr, y.imm7);

                        xluf_ptr += AVX2_FLOAT_STRIDE;
                        xuf_ptr += AVX2_FLOAT_STRIDE;
                        xruf_ptr += AVX2_FLOAT_STRIDE;
                        xlf_ptr += AVX2_FLOAT_STRIDE;
                        xf_ptr += AVX2_FLOAT_STRIDE;
                        xrf_ptr += AVX2_FLOAT_STRIDE;
                        xldf_ptr += AVX2_FLOAT_STRIDE;
                        xdf_ptr += AVX2_FLOAT_STRIDE;
                        xrdf_ptr += AVX2_FLOAT_STRIDE;

                        xlu_ptr += AVX2_FLOAT_STRIDE;
                        xu_ptr += AVX2_FLOAT_STRIDE;
                        xru_ptr += AVX2_FLOAT_STRIDE;
                        xl_ptr += AVX2_FLOAT_STRIDE;
                        xc_ptr += AVX2_FLOAT_STRIDE;
                        xr_ptr += AVX2_FLOAT_STRIDE;
                        xld_ptr += AVX2_FLOAT_STRIDE;
                        xd_ptr += AVX2_FLOAT_STRIDE;
                        xrd_ptr += AVX2_FLOAT_STRIDE;

                        xlub_ptr += AVX2_FLOAT_STRIDE;
                        xub_ptr += AVX2_FLOAT_STRIDE;
                        xrub_ptr += AVX2_FLOAT_STRIDE;
                        xlb_ptr += AVX2_FLOAT_STRIDE;
                        xb_ptr += AVX2_FLOAT_STRIDE;
                        xrb_ptr += AVX2_FLOAT_STRIDE;
                        xldb_ptr += AVX2_FLOAT_STRIDE;
                        xdb_ptr += AVX2_FLOAT_STRIDE;
                        xrdb_ptr += AVX2_FLOAT_STRIDE;

                        yluf_ptr += AVX2_FLOAT_STRIDE;
                        yruf_ptr += AVX2_FLOAT_STRIDE;
                        yldf_ptr += AVX2_FLOAT_STRIDE;
                        yrdf_ptr += AVX2_FLOAT_STRIDE;
                        ylub_ptr += AVX2_FLOAT_STRIDE;
                        yrub_ptr += AVX2_FLOAT_STRIDE;
                        yldb_ptr += AVX2_FLOAT_STRIDE;
                        yrdb_ptr += AVX2_FLOAT_STRIDE;

                        r -= AVX2_FLOAT_STRIDE;
                    }
                    if (r > 0) {
                        _mm256_loadu_x1_ps(xluf_ptr, xluf);
                        _mm256_loadu_x1_ps(xuf_ptr, xuf);
                        _mm256_loadu_x1_ps(xruf_ptr, xruf);
                        _mm256_loadu_x1_ps(xlf_ptr, xlf);
                        _mm256_loadu_x1_ps(xf_ptr, xf);
                        _mm256_loadu_x1_ps(xrf_ptr, xrf);
                        _mm256_loadu_x1_ps(xldf_ptr, xldf);
                        _mm256_loadu_x1_ps(xdf_ptr, xdf);
                        _mm256_loadu_x1_ps(xrdf_ptr, xrdf);

                        _mm256_loadu_x1_ps(xlu_ptr, xlu);
                        _mm256_loadu_x1_ps(xu_ptr, xu);
                        _mm256_loadu_x1_ps(xru_ptr, xru);
                        _mm256_loadu_x1_ps(xl_ptr, xl);
                        _mm256_loadu_x1_ps(xc_ptr, xc);
                        _mm256_loadu_x1_ps(xr_ptr, xr);
                        _mm256_loadu_x1_ps(xld_ptr, xld);
                        _mm256_loadu_x1_ps(xd_ptr, xd);
                        _mm256_loadu_x1_ps(xrd_ptr, xrd);

                        _mm256_loadu_x1_ps(xlub_ptr, xlub);
                        _mm256_loadu_x1_ps(xub_ptr, xub);
                        _mm256_loadu_x1_ps(xrub_ptr, xrub);
                        _mm256_loadu_x1_ps(xlb_ptr, xlb);
                        _mm256_loadu_x1_ps(xb_ptr, xb);
                        _mm256_loadu_x1_ps(xrb_ptr, xrb);
                        _mm256_loadu_x1_ps(xldb_ptr, xldb);
                        _mm256_loadu_x1_ps(xdb_ptr, xdb);
                        _mm256_loadu_x1_ps(xrdb_ptr, xrdb);

                        __m256x8 y = _mm256_linear3d_ps(
                            xluf, xuf, xruf, xlf, xf, xrf, xldf, xdf, xrdf,
                            xlu, xu, xru, xl, xc, xr, xld, xd, xrd,
                            xlub, xub, xrub, xlb, xb, xrb, xldb, xdb, xrdb
                        );

                        _mm256_maskstore_x1_ps(yluf_ptr, y.imm0, mask);
                        _mm256_maskstore_x1_ps(yruf_ptr, y.imm1, mask);
                        _mm256_maskstore_x1_ps(yldf_ptr, y.imm2, mask);
                        _mm256_maskstore_x1_ps(yrdf_ptr, y.imm3, mask);
                        _mm256_maskstore_x1_ps(ylub_ptr, y.imm4, mask);
                        _mm256_maskstore_x1_ps(yrub_ptr, y.imm5, mask);
                        _mm256_maskstore_x1_ps(yldb_ptr, y.imm6, mask);
                        _mm256_maskstore_x1_ps(yrdb_ptr, y.imm7, mask);
                    }
                }
            }
        }

        x_ptr += c * iw * ih * id;
        y_ptr += c * ow * oh * od;
    }

    return SUCCESS;
}

//int upsample3d_linear_c1(
//    const uint n, const uint c,
//    const uint iw, const uint ow, const uint ih, const uint oh, const uint id, const uint od,
//    infloats x_ptr, outfloats y_ptr) {
//
//#ifdef _DEBUG
//    if (c != 1) {
//        return FAILURE_BADPARAM;
//    }
//#endif // _DEBUG
//
//    float xlu, xu, xru, xl, xc, xr, xld, xd, xrd;
//
//    for (uint i = 0; i < n; i++) {
//        for (uint iz = 0, oz = 0; iz < id; iz++, oz += 2) {
//            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
//                for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
//                    const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);
//                    const uint iyu = max(iy, 1u) - 1u, iyd = min(iy + 1u, ih - 1u);
//                    const uint izf = max(iz, 1u) - 1u, izb = min(iz + 1u, id - 1u);
//
//                    const float* xlu_ptr = x_ptr + c * (ixl + iw * iyu);
//                    const float* xu_ptr = x_ptr + c * (ix + iw * iyu);
//                    const float* xru_ptr = x_ptr + c * (ixr + iw * iyu);
//                    const float* xl_ptr = x_ptr + c * (ixl + iw * iy);
//                    const float* xc_ptr = x_ptr + c * (ix + iw * iy);
//                    const float* xr_ptr = x_ptr + c * (ixr + iw * iy);
//                    const float* xld_ptr = x_ptr + c * (ixl + iw * iyd);
//                    const float* xd_ptr = x_ptr + c * (ix + iw * iyd);
//                    const float* xrd_ptr = x_ptr + c * (ixr + iw * iyd);
//
//                    float* ylu_ptr = y_ptr + (ox + ow * oy);
//                    float* yru_ptr = y_ptr + ((ox + 1) + ow * oy);
//                    float* yld_ptr = y_ptr + (ox + ow * (oy + 1));
//                    float* yrd_ptr = y_ptr + ((ox + 1) + ow * (oy + 1));
//
//                    xlu = *xlu_ptr;
//                    xu = *xu_ptr;
//                    xru = *xru_ptr;
//                    xl = *xl_ptr;
//                    xc = *xc_ptr;
//                    xr = *xr_ptr;
//                    xld = *xld_ptr;
//                    xd = *xd_ptr;
//                    xrd = *xrd_ptr;
//
//                    floatx8 y = float_linear3d(xlu, xu, xru, xl, xc, xr, xld, xd, xrd, xlu, xu, xru, xl, xc, xr, xld, xd, xrd, xlu, xu, xru, xl, xc, xr, xld, xd, xrd);
//
//                    *ylu_ptr = y.imm0;
//                    *yru_ptr = y.imm1;
//                    *yld_ptr = y.imm2;
//                    *yrd_ptr = y.imm3;
//                }
//            }
//        }
//
//        x_ptr += c * iw * ih * id;
//        y_ptr += c * ow * oh * od;
//    }
//
//    return SUCCESS;
//}
//
//int upsample3d_linear_c2to3(
//    const uint n, const uint c,
//    const uint iw, const uint ow, const uint ih, const uint oh, const uint id, const uint od,
//    infloats x_ptr, outfloats y_ptr) {
//
//#ifdef _DEBUG
//    if (c <= 1 && c >= AVX1_FLOAT_STRIDE) {
//        return FAILURE_BADPARAM;
//    }
//#endif // _DEBUG
//
//    const __m128i mask = _mm_setmask_ps(c);
//
//    __m128 xlu, xu, xru, xl, xc, xr, xld, xd, xrd;
//
//    for (uint i = 0; i < n; i++) {
//        for (uint iz = 0, oz = 0; iz < id; iz++, oz += 2) {
//            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
//                for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
//                    const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);
//                    const uint iyu = max(iy, 1u) - 1u, iyd = min(iy + 1u, ih - 1u);
//                    const uint izf = max(iz, 1u) - 1u, izb = min(iz + 1u, id - 1u);
//
//                    const float* xlu_ptr = x_ptr + c * (ixl + iw * iyu);
//                    const float* xu_ptr = x_ptr + c * (ix + iw * iyu);
//                    const float* xru_ptr = x_ptr + c * (ixr + iw * iyu);
//                    const float* xl_ptr = x_ptr + c * (ixl + iw * iy);
//                    const float* xc_ptr = x_ptr + c * (ix + iw * iy);
//                    const float* xr_ptr = x_ptr + c * (ixr + iw * iy);
//                    const float* xld_ptr = x_ptr + c * (ixl + iw * iyd);
//                    const float* xd_ptr = x_ptr + c * (ix + iw * iyd);
//                    const float* xrd_ptr = x_ptr + c * (ixr + iw * iyd);
//
//                    float* ylu_ptr = y_ptr + c * (ox + ow * oy);
//                    float* yru_ptr = y_ptr + c * ((ox + 1) + ow * oy);
//                    float* yld_ptr = y_ptr + c * (ox + ow * (oy + 1));
//                    float* yrd_ptr = y_ptr + c * ((ox + 1) + ow * (oy + 1));
//
//                    xlu = _mm_loadu_ps(xlu_ptr);
//                    xu = _mm_loadu_ps(xu_ptr);
//                    xru = _mm_loadu_ps(xru_ptr);
//                    xl = _mm_loadu_ps(xl_ptr);
//                    xc = _mm_loadu_ps(xc_ptr);
//                    xr = _mm_loadu_ps(xr_ptr);
//                    xld = _mm_loadu_ps(xld_ptr);
//                    xd = _mm_loadu_ps(xd_ptr);
//                    xrd = _mm_loadu_ps(xrd_ptr);
//
//                    __m128x4 y = _mm_linear3d_ps(xlu, xu, xru, xl, xc, xr, xld, xd, xrd);
//
//                    _mm_maskstore_ps(ylu_ptr, mask, y.imm0);
//                    _mm_maskstore_ps(yru_ptr, mask, y.imm1);
//                    _mm_maskstore_ps(yld_ptr, mask, y.imm2);
//                    _mm_maskstore_ps(yrd_ptr, mask, y.imm3);
//                }
//            }
//        }
//
//        x_ptr += c * iw * ih * id;
//        y_ptr += c * ow * oh * od;
//    }
//
//    return SUCCESS;
//}
//
//int upsample3d_linear_c4(
//    const uint n, const uint c,
//    const uint iw, const uint ow, const uint ih, const uint oh, const uint id, const uint od,
//    infloats x_ptr, outfloats y_ptr) {
//
//#ifdef _DEBUG
//    if (c != AVX1_FLOAT_STRIDE) {
//        return FAILURE_BADPARAM;
//    }
//#endif // _DEBUG
//
//    __m128 xlu, xu, xru, xl, xc, xr, xld, xd, xrd;
//
//    for (uint i = 0; i < n; i++) {
//        for (uint iz = 0, oz = 0; iz < id; iz++, oz += 2) {
//            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
//                for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
//                    const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);
//                    const uint iyu = max(iy, 1u) - 1u, iyd = min(iy + 1u, ih - 1u);
//                    const uint izf = max(iz, 1u) - 1u, izb = min(iz + 1u, id - 1u);
//
//                    const float* xlu_ptr = x_ptr + c * (ixl + iw * iyu);
//                    const float* xu_ptr = x_ptr + c * (ix + iw * iyu);
//                    const float* xru_ptr = x_ptr + c * (ixr + iw * iyu);
//                    const float* xl_ptr = x_ptr + c * (ixl + iw * iy);
//                    const float* xc_ptr = x_ptr + c * (ix + iw * iy);
//                    const float* xr_ptr = x_ptr + c * (ixr + iw * iy);
//                    const float* xld_ptr = x_ptr + c * (ixl + iw * iyd);
//                    const float* xd_ptr = x_ptr + c * (ix + iw * iyd);
//                    const float* xrd_ptr = x_ptr + c * (ixr + iw * iyd);
//
//                    float* ylu_ptr = y_ptr + c * (ox + ow * oy);
//                    float* yru_ptr = y_ptr + c * ((ox + 1) + ow * oy);
//                    float* yld_ptr = y_ptr + c * (ox + ow * (oy + 1));
//                    float* yrd_ptr = y_ptr + c * ((ox + 1) + ow * (oy + 1));
//
//                    xlu = _mm_load_ps(xlu_ptr);
//                    xu = _mm_load_ps(xu_ptr);
//                    xru = _mm_load_ps(xru_ptr);
//                    xl = _mm_load_ps(xl_ptr);
//                    xc = _mm_load_ps(xc_ptr);
//                    xr = _mm_load_ps(xr_ptr);
//                    xld = _mm_load_ps(xld_ptr);
//                    xd = _mm_load_ps(xd_ptr);
//                    xrd = _mm_load_ps(xrd_ptr);
//
//                    __m128x4 y = _mm_linear3d_ps(xlu, xu, xru, xl, xc, xr, xld, xd, xrd);
//
//                    _mm_stream_ps(ylu_ptr, y.imm0);
//                    _mm_stream_ps(yru_ptr, y.imm1);
//                    _mm_stream_ps(yld_ptr, y.imm2);
//                    _mm_stream_ps(yrd_ptr, y.imm3);
//                }
//            }
//        }
//
//        x_ptr += c * iw * ih * id;
//        y_ptr += c * ow * oh * od;
//    }
//
//    return SUCCESS;
//}
//
//int upsample3d_linear_c5to7(
//    const uint n, const uint c,
//    const uint iw, const uint ow, const uint ih, const uint oh, const uint id, const uint od,
//    infloats x_ptr, outfloats y_ptr) {
//
//#ifdef _DEBUG
//    if (c <= AVX1_FLOAT_STRIDE || c >= AVX2_FLOAT_STRIDE) {
//        return FAILURE_BADPARAM;
//    }
//#endif // _DEBUG
//
//    const __m256i mask = _mm256_setmask_ps(c);
//
//    __m256 xlu, xu, xru, xl, xc, xr, xld, xd, xrd;
//
//    for (uint i = 0; i < n; i++) {
//        for (uint iz = 0, oz = 0; iz < id; iz++, oz += 2) {
//            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
//                for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
//                    const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);
//                    const uint iyu = max(iy, 1u) - 1u, iyd = min(iy + 1u, ih - 1u);
//                    const uint izf = max(iz, 1u) - 1u, izb = min(iz + 1u, id - 1u);
//
//                    const float* xlu_ptr = x_ptr + c * (ixl + iw * iyu);
//                    const float* xu_ptr = x_ptr + c * (ix + iw * iyu);
//                    const float* xru_ptr = x_ptr + c * (ixr + iw * iyu);
//                    const float* xl_ptr = x_ptr + c * (ixl + iw * iy);
//                    const float* xc_ptr = x_ptr + c * (ix + iw * iy);
//                    const float* xr_ptr = x_ptr + c * (ixr + iw * iy);
//                    const float* xld_ptr = x_ptr + c * (ixl + iw * iyd);
//                    const float* xd_ptr = x_ptr + c * (ix + iw * iyd);
//                    const float* xrd_ptr = x_ptr + c * (ixr + iw * iyd);
//
//                    float* ylu_ptr = y_ptr + c * (ox + ow * oy);
//                    float* yru_ptr = y_ptr + c * ((ox + 1) + ow * oy);
//                    float* yld_ptr = y_ptr + c * (ox + ow * (oy + 1));
//                    float* yrd_ptr = y_ptr + c * ((ox + 1) + ow * (oy + 1));
//
//                    _mm256_loadu_x1_ps(xlu_ptr, xlu);
//                    _mm256_loadu_x1_ps(xu_ptr, xu);
//                    _mm256_loadu_x1_ps(xru_ptr, xru);
//                    _mm256_loadu_x1_ps(xl_ptr, xl);
//                    _mm256_loadu_x1_ps(xc_ptr, xc);
//                    _mm256_loadu_x1_ps(xr_ptr, xr);
//                    _mm256_loadu_x1_ps(xld_ptr, xld);
//                    _mm256_loadu_x1_ps(xd_ptr, xd);
//                    _mm256_loadu_x1_ps(xrd_ptr, xrd);
//
//                    __m256x4 y = _mm256_linear3d_ps(xlu, xu, xru, xl, xc, xr, xld, xd, xrd);
//
//                    _mm256_maskstore_x1_ps(ylu_ptr, y.imm0, mask);
//                    _mm256_maskstore_x1_ps(yru_ptr, y.imm1, mask);
//                    _mm256_maskstore_x1_ps(yld_ptr, y.imm2, mask);
//                    _mm256_maskstore_x1_ps(yrd_ptr, y.imm3, mask);
//                }
//            }
//        }
//
//        x_ptr += c * iw * ih * id;
//        y_ptr += c * ow * oh * od;
//    }
//
//    return SUCCESS;
//}
//
//int upsample3d_linear_c8(
//    const uint n, const uint c,
//    const uint iw, const uint ow, const uint ih, const uint oh, const uint id, const uint od,
//    infloats x_ptr, outfloats y_ptr) {
//
//#ifdef _DEBUG
//    if (c != AVX2_FLOAT_STRIDE) {
//        return FAILURE_BADPARAM;
//    }
//#endif // _DEBUG
//
//    __m256 xlu, xu, xru, xl, xc, xr, xld, xd, xrd;
//
//    for (uint i = 0; i < n; i++) {
//        for (uint iz = 0, oz = 0; iz < id; iz++, oz += 2) {
//            for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
//                for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
//                    const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);
//                    const uint iyu = max(iy, 1u) - 1u, iyd = min(iy + 1u, ih - 1u);
//                    const uint izf = max(iz, 1u) - 1u, izb = min(iz + 1u, id - 1u);
//
//                    const float* xlu_ptr = x_ptr + c * (ixl + iw * iyu);
//                    const float* xu_ptr = x_ptr + c * (ix + iw * iyu);
//                    const float* xru_ptr = x_ptr + c * (ixr + iw * iyu);
//                    const float* xl_ptr = x_ptr + c * (ixl + iw * iy);
//                    const float* xc_ptr = x_ptr + c * (ix + iw * iy);
//                    const float* xr_ptr = x_ptr + c * (ixr + iw * iy);
//                    const float* xld_ptr = x_ptr + c * (ixl + iw * iyd);
//                    const float* xd_ptr = x_ptr + c * (ix + iw * iyd);
//                    const float* xrd_ptr = x_ptr + c * (ixr + iw * iyd);
//
//                    float* ylu_ptr = y_ptr + c * (ox + ow * oy);
//                    float* yru_ptr = y_ptr + c * ((ox + 1) + ow * oy);
//                    float* yld_ptr = y_ptr + c * (ox + ow * (oy + 1));
//                    float* yrd_ptr = y_ptr + c * ((ox + 1) + ow * (oy + 1));
//
//                    _mm256_load_x1_ps(xlu_ptr, xlu);
//                    _mm256_load_x1_ps(xu_ptr, xu);
//                    _mm256_load_x1_ps(xru_ptr, xru);
//                    _mm256_load_x1_ps(xl_ptr, xl);
//                    _mm256_load_x1_ps(xc_ptr, xc);
//                    _mm256_load_x1_ps(xr_ptr, xr);
//                    _mm256_load_x1_ps(xld_ptr, xld);
//                    _mm256_load_x1_ps(xd_ptr, xd);
//                    _mm256_load_x1_ps(xrd_ptr, xrd);
//
//                    __m256x4 y = _mm256_linear3d_ps(xlu, xu, xru, xl, xc, xr, xld, xd, xrd);
//
//                    _mm256_stream_x1_ps(ylu_ptr, y.imm0);
//                    _mm256_stream_x1_ps(yru_ptr, y.imm1);
//                    _mm256_stream_x1_ps(yld_ptr, y.imm2);
//                    _mm256_stream_x1_ps(yrd_ptr, y.imm3);
//                }
//            }
//        }
//
//        x_ptr += c * iw * ih * id;
//        y_ptr += c * ow * oh * od;
//    }
//
//    return SUCCESS;
//}
//
//int upsample3d_linear_cleq8(
//    const uint n, const uint c,
//    const uint iw, const uint ow, const uint ih, const uint oh, const uint id, const uint od,
//    infloats x_ptr, outfloats y_ptr) {
//
//    if (c == 1) {
//        return upsample3d_linear_c1(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
//    }
//    if (c < AVX1_FLOAT_STRIDE) {
//        return upsample3d_linear_c2to3(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
//    }
//    if (c == AVX1_FLOAT_STRIDE) {
//        return upsample3d_linear_c4(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
//    }
//    if (c < AVX2_FLOAT_STRIDE) {
//        return upsample3d_linear_c5to7(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
//    }
//    if (c == AVX2_FLOAT_STRIDE) {
//        return upsample3d_linear_c8(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
//    }
//
//    return FAILURE_BADPARAM;
//}

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

    //    if (c <= AVX2_FLOAT_STRIDE) {
    //#ifdef _DEBUG
    //        Console::WriteLine("type leq8");
    //#endif // _DEBUG
    //
    //        ret = upsample3d_linear_cleq8(n, c, iw, ow, ih, oh, id, od, x_ptr, y_ptr);
    //    }
    if ((c & AVX2_FLOAT_REMAIN_MASK) == 0) {
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
