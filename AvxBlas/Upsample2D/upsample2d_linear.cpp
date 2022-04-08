#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_numeric.hpp"
#include "../Inline/inline_loadstore_xn_s.hpp"

#pragma unmanaged

__forceinline __m256x4 _mm256_linear2d_ps(
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

int upsample2d_linear_c32x(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 xlu0, xlu1, xlu2, xlu3, xu0, xu1, xu2, xu3, xru0, xru1, xru2, xru3;
    __m256 xl0, xl1, xl2, xl3, xc0, xc1, xc2, xc3, xr0, xr1, xr2, xr3;
    __m256 xld0, xld1, xld2, xld3, xd0, xd1, xd2, xd3, xrd0, xrd1, xrd2, xrd3;

    for (uint i = 0; i < n; i++) {
        for (uint iy = 0, oy = 0; iy < ih; iy++, oy += 2) {
            for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);
                const uint iyu = max(iy, 1u) - 1u, iyd = min(iy + 1u, ih - 1u);

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

                float* ylu_ptr = y_ptr + c * ox;
                float* yru_ptr = y_ptr + c * (ox + 1);
                float* yld_ptr = y_ptr + c * (ox + ow);
                float* yrd_ptr = y_ptr + c * (ox + 1 + ow);

                while (r >= AVX2_FLOAT_STRIDE * 4) {
                    _mm256_load_x4_ps(xlu_ptr, xlu0, xlu1, xlu2, xlu3);
                    _mm256_load_x4_ps(xu_ptr, xu0, xu1, xu2, xu3);
                    _mm256_load_x4_ps(xru_ptr, xru0, xru1, xru2, xru3);
                    _mm256_load_x4_ps(xl_ptr, xl0, xl1, xl2, xl3);
                    _mm256_load_x4_ps(xc_ptr, xc0, xc1, xc2, xc3);
                    _mm256_load_x4_ps(xr_ptr, xr0, xr1, xr2, xr3);
                    _mm256_load_x4_ps(xld_ptr, xld0, xld1, xld2, xld3);
                    _mm256_load_x4_ps(xd_ptr, xd0, xd1, xd2, xd3);
                    _mm256_load_x4_ps(xrd_ptr, xrd0, xrd1, xrd2, xrd3);

                    __m256x4 y0 = _mm256_linear2d_ps(xlu0, xu0, xru0, xl0, xc0, xr0, xld0, xd0, xrd0);
                    __m256x4 y1 = _mm256_linear2d_ps(xlu1, xu1, xru1, xl1, xc1, xr1, xld1, xd1, xrd1);
                    __m256x4 y2 = _mm256_linear2d_ps(xlu2, xu2, xru2, xl2, xc2, xr2, xld2, xd2, xrd2);
                    __m256x4 y3 = _mm256_linear2d_ps(xlu3, xu3, xru3, xl3, xc3, xr3, xld3, xd3, xrd3);

                    _mm256_stream_x4_ps(ylu_ptr, y0.imm0, y1.imm0, y2.imm0, y3.imm0);
                    _mm256_stream_x4_ps(yru_ptr, y0.imm1, y1.imm1, y2.imm1, y3.imm1);
                    _mm256_stream_x4_ps(ylu_ptr, y0.imm0, y1.imm0, y2.imm0, y3.imm0);
                    _mm256_stream_x4_ps(yru_ptr, y0.imm1, y1.imm1, y2.imm1, y3.imm1);

                    xl_ptr += AVX2_FLOAT_STRIDE * 4;
                    xc_ptr += AVX2_FLOAT_STRIDE * 4;
                    xr_ptr += AVX2_FLOAT_STRIDE * 4;
                    yl_ptr += AVX2_FLOAT_STRIDE * 4;
                    yr_ptr += AVX2_FLOAT_STRIDE * 4;
                    r -= AVX2_FLOAT_STRIDE * 4;
                }
            }
        }

        x_ptr += c * iw;
        y_ptr += c * ow;
    }

    return SUCCESS;
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

    __m256 xl0, xl1, xl2, xl3, xc0, xc1, xc2, xc3, xr0, xr1, xr2, xr3;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
            const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);

            uint r = c;

            const float* xl_ptr = x_ptr + c * ixl;
            const float* xc_ptr = x_ptr + c * ix;
            const float* xr_ptr = x_ptr + c * ixr;

            float* yl_ptr = y_ptr + c * ox;
            float* yr_ptr = y_ptr + c * (ox + 1);

            while (r >= AVX2_FLOAT_STRIDE * 4) {
                _mm256_load_x4_ps(xl_ptr, xl0, xl1, xl2, xl3);
                _mm256_load_x4_ps(xc_ptr, xc0, xc1, xc2, xc3);
                _mm256_load_x4_ps(xr_ptr, xr0, xr1, xr2, xr3);

                __m256x2 y0 = _mm256_linear2d_ps(xl0, xc0, xr0);
                __m256x2 y1 = _mm256_linear2d_ps(xl1, xc1, xr1);
                __m256x2 y2 = _mm256_linear2d_ps(xl2, xc2, xr2);
                __m256x2 y3 = _mm256_linear2d_ps(xl3, xc3, xr3);

                _mm256_stream_x4_ps(yl_ptr, y0.imm0, y1.imm0, y2.imm0, y3.imm0);
                _mm256_stream_x4_ps(yr_ptr, y0.imm1, y1.imm1, y2.imm1, y3.imm1);

                xl_ptr += AVX2_FLOAT_STRIDE * 4;
                xc_ptr += AVX2_FLOAT_STRIDE * 4;
                xr_ptr += AVX2_FLOAT_STRIDE * 4;
                yl_ptr += AVX2_FLOAT_STRIDE * 4;
                yr_ptr += AVX2_FLOAT_STRIDE * 4;
                r -= AVX2_FLOAT_STRIDE * 4;
            }
            if (r >= AVX2_FLOAT_STRIDE * 3) {
                _mm256_load_x3_ps(xl_ptr, xl0, xl1, xl2);
                _mm256_load_x3_ps(xc_ptr, xc0, xc1, xc2);
                _mm256_load_x3_ps(xr_ptr, xr0, xr1, xr2);

                __m256x2 y0 = _mm256_linear2d_ps(xl0, xc0, xr0);
                __m256x2 y1 = _mm256_linear2d_ps(xl1, xc1, xr1);
                __m256x2 y2 = _mm256_linear2d_ps(xl2, xc2, xr2);

                _mm256_stream_x3_ps(yl_ptr, y0.imm0, y1.imm0, y2.imm0);
                _mm256_stream_x3_ps(yr_ptr, y0.imm1, y1.imm1, y2.imm1);
            }
            else if (r >= AVX2_FLOAT_STRIDE * 2) {
                _mm256_load_x2_ps(xl_ptr, xl0, xl1);
                _mm256_load_x2_ps(xc_ptr, xc0, xc1);
                _mm256_load_x2_ps(xr_ptr, xr0, xr1);

                __m256x2 y0 = _mm256_linear2d_ps(xl0, xc0, xr0);
                __m256x2 y1 = _mm256_linear2d_ps(xl1, xc1, xr1);

                _mm256_stream_x2_ps(yl_ptr, y0.imm0, y1.imm0);
                _mm256_stream_x2_ps(yr_ptr, y0.imm1, y1.imm1);
            }
            else if (r >= AVX2_FLOAT_STRIDE) {
                _mm256_load_x1_ps(xl_ptr, xl0);
                _mm256_load_x1_ps(xc_ptr, xc0);
                _mm256_load_x1_ps(xr_ptr, xr0);

                __m256x2 y0 = _mm256_linear2d_ps(xl0, xc0, xr0);

                _mm256_stream_x1_ps(yl_ptr, y0.imm0);
                _mm256_stream_x1_ps(yr_ptr, y0.imm1);
            }

        }

        x_ptr += c * iw;
        y_ptr += c * ow;
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

    __m256 xl0, xl1, xl2, xl3, xc0, xc1, xc2, xc3, xr0, xr1, xr2, xr3;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
            const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);

            uint r = c;

            const float* xl_ptr = x_ptr + c * ixl;
            const float* xc_ptr = x_ptr + c * ix;
            const float* xr_ptr = x_ptr + c * ixr;

            float* yl_ptr = y_ptr + c * ox;
            float* yr_ptr = y_ptr + c * (ox + 1);

            while (r >= AVX2_FLOAT_STRIDE * 4) {
                _mm256_loadu_x4_ps(xl_ptr, xl0, xl1, xl2, xl3);
                _mm256_loadu_x4_ps(xc_ptr, xc0, xc1, xc2, xc3);
                _mm256_loadu_x4_ps(xr_ptr, xr0, xr1, xr2, xr3);

                __m256x2 y0 = _mm256_linear2d_ps(xl0, xc0, xr0);
                __m256x2 y1 = _mm256_linear2d_ps(xl1, xc1, xr1);
                __m256x2 y2 = _mm256_linear2d_ps(xl2, xc2, xr2);
                __m256x2 y3 = _mm256_linear2d_ps(xl3, xc3, xr3);

                _mm256_storeu_x4_ps(yl_ptr, y0.imm0, y1.imm0, y2.imm0, y3.imm0);
                _mm256_storeu_x4_ps(yr_ptr, y0.imm1, y1.imm1, y2.imm1, y3.imm1);

                xl_ptr += AVX2_FLOAT_STRIDE * 4;
                xc_ptr += AVX2_FLOAT_STRIDE * 4;
                xr_ptr += AVX2_FLOAT_STRIDE * 4;
                yl_ptr += AVX2_FLOAT_STRIDE * 4;
                yr_ptr += AVX2_FLOAT_STRIDE * 4;
                r -= AVX2_FLOAT_STRIDE * 4;
            }
            if (r >= AVX2_FLOAT_STRIDE * 3) {
                _mm256_loadu_x4_ps(xl_ptr, xl0, xl1, xl2, xl3);
                _mm256_loadu_x4_ps(xc_ptr, xc0, xc1, xc2, xc3);
                _mm256_loadu_x4_ps(xr_ptr, xr0, xr1, xr2, xr3);

                __m256x2 y0 = _mm256_linear2d_ps(xl0, xc0, xr0);
                __m256x2 y1 = _mm256_linear2d_ps(xl1, xc1, xr1);
                __m256x2 y2 = _mm256_linear2d_ps(xl2, xc2, xr2);
                __m256x2 y3 = _mm256_linear2d_ps(xl3, xc3, xr3);

                _mm256_maskstore_x4_ps(yl_ptr, y0.imm0, y1.imm0, y2.imm0, y3.imm0, mask);
                _mm256_maskstore_x4_ps(yr_ptr, y0.imm1, y1.imm1, y2.imm1, y3.imm1, mask);
            }
            else if (r >= AVX2_FLOAT_STRIDE * 2) {
                _mm256_loadu_x3_ps(xl_ptr, xl0, xl1, xl2);
                _mm256_loadu_x3_ps(xc_ptr, xc0, xc1, xc2);
                _mm256_loadu_x3_ps(xr_ptr, xr0, xr1, xr2);

                __m256x2 y0 = _mm256_linear2d_ps(xl0, xc0, xr0);
                __m256x2 y1 = _mm256_linear2d_ps(xl1, xc1, xr1);
                __m256x2 y2 = _mm256_linear2d_ps(xl2, xc2, xr2);

                _mm256_maskstore_x3_ps(yl_ptr, y0.imm0, y1.imm0, y2.imm0, mask);
                _mm256_maskstore_x3_ps(yr_ptr, y0.imm1, y1.imm1, y2.imm1, mask);
            }
            else if (r >= AVX2_FLOAT_STRIDE) {
                _mm256_loadu_x2_ps(xl_ptr, xl0, xl1);
                _mm256_loadu_x2_ps(xc_ptr, xc0, xc1);
                _mm256_loadu_x2_ps(xr_ptr, xr0, xr1);

                __m256x2 y0 = _mm256_linear2d_ps(xl0, xc0, xr0);
                __m256x2 y1 = _mm256_linear2d_ps(xl1, xc1, xr1);

                _mm256_maskstore_x2_ps(yl_ptr, y0.imm0, y1.imm0, mask);
                _mm256_maskstore_x2_ps(yr_ptr, y0.imm1, y1.imm1, mask);
            }
            else if (r > 0) {
                _mm256_loadu_x1_ps(xl_ptr, xl0);
                _mm256_loadu_x1_ps(xc_ptr, xc0);
                _mm256_loadu_x1_ps(xr_ptr, xr0);

                __m256x2 y0 = _mm256_linear2d_ps(xl0, xc0, xr0);

                _mm256_maskstore_x1_ps(yl_ptr, y0.imm0, mask);
                _mm256_maskstore_x1_ps(yr_ptr, y0.imm1, mask);
            }
        }

        x_ptr += c * iw;
        y_ptr += c * ow;
    }

    return SUCCESS;
}

int upsample2d_linear_cleq8(
    const uint n, const uint c,
    const uint iw, const uint ow, const uint ih, const uint oh,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c > AVX2_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    if (c == AVX2_FLOAT_STRIDE) {
        __m256 xl0, xc0, xr0;

        for (uint i = 0; i < n; i++) {
            for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);

                const float* xl_ptr = x_ptr + c * ixl;
                const float* xc_ptr = x_ptr + c * ix;
                const float* xr_ptr = x_ptr + c * ixr;

                float* yl_ptr = y_ptr + c * ox;
                float* yr_ptr = y_ptr + c * (ox + 1);

                _mm256_load_x1_ps(xl_ptr, xl0);
                _mm256_load_x1_ps(xc_ptr, xc0);
                _mm256_load_x1_ps(xr_ptr, xr0);

                __m256x2 y0 = _mm256_linear2d_ps(xl0, xc0, xr0);

                _mm256_stream_x1_ps(yl_ptr, y0.imm0);
                _mm256_stream_x1_ps(yr_ptr, y0.imm1);
            }

            x_ptr += c * iw;
            y_ptr += c * ow;
        }

        return SUCCESS;
    }

    if (c > AVX1_FLOAT_STRIDE && c < AVX2_FLOAT_STRIDE) {
        const __m256i mask = _mm256_setmask_ps(c);

        __m256 xl0, xc0, xr0;

        for (uint i = 0; i < n; i++) {
            for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);

                const float* xl_ptr = x_ptr + c * ixl;
                const float* xc_ptr = x_ptr + c * ix;
                const float* xr_ptr = x_ptr + c * ixr;

                float* yl_ptr = y_ptr + c * ox;
                float* yr_ptr = y_ptr + c * (ox + 1);

                _mm256_loadu_x1_ps(xl_ptr, xl0);
                _mm256_loadu_x1_ps(xc_ptr, xc0);
                _mm256_loadu_x1_ps(xr_ptr, xr0);

                __m256x2 y0 = _mm256_linear2d_ps(xl0, xc0, xr0);

                _mm256_maskstore_x1_ps(yl_ptr, y0.imm0, mask);
                _mm256_maskstore_x1_ps(yr_ptr, y0.imm1, mask);
            }

            x_ptr += c * iw;
            y_ptr += c * ow;
        }

        return SUCCESS;
    }

    if (c == AVX1_FLOAT_STRIDE) {
        __m128 xl0, xc0, xr0;

        for (uint i = 0; i < n; i++) {
            for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);

                const float* xl_ptr = x_ptr + c * ixl;
                const float* xc_ptr = x_ptr + c * ix;
                const float* xr_ptr = x_ptr + c * ixr;

                float* yl_ptr = y_ptr + c * ox;
                float* yr_ptr = y_ptr + c * (ox + 1);

                xl0 = _mm_load_ps(xl_ptr);
                xc0 = _mm_load_ps(xc_ptr);
                xr0 = _mm_load_ps(xr_ptr);

                __m128x2 y0 = _mm128_linear2d_ps(xl0, xc0, xr0);

                _mm_stream_ps(yl_ptr, y0.imm0);
                _mm_stream_ps(yr_ptr, y0.imm1);
            }

            x_ptr += c * iw;
            y_ptr += c * ow;
        }

        return SUCCESS;
    }

    if (c > 1 && c < AVX1_FLOAT_STRIDE) {
        const __m128i mask = _mm_setmask_ps(c);

        __m128 xl0, xc0, xr0;

        for (uint i = 0; i < n; i++) {
            for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);

                const float* xl_ptr = x_ptr + c * ixl;
                const float* xc_ptr = x_ptr + c * ix;
                const float* xr_ptr = x_ptr + c * ixr;

                float* yl_ptr = y_ptr + c * ox;
                float* yr_ptr = y_ptr + c * (ox + 1);

                xl0 = _mm_loadu_ps(xl_ptr);
                xc0 = _mm_loadu_ps(xc_ptr);
                xr0 = _mm_loadu_ps(xr_ptr);

                __m128x2 y0 = _mm128_linear2d_ps(xl0, xc0, xr0);

                _mm_maskstore_ps(yl_ptr, mask, y0.imm0);
                _mm_maskstore_ps(yr_ptr, mask, y0.imm1);
            }

            x_ptr += c * iw;
            y_ptr += c * ow;
        }

        return SUCCESS;
    }

    if (c == 1) {
        float xl0, xc0, xr0;

        for (uint i = 0; i < n; i++) {
            for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
                const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);

                const float* xl_ptr = x_ptr + ixl;
                const float* xc_ptr = x_ptr + ix;
                const float* xr_ptr = x_ptr + ixr;

                float* yl_ptr = y_ptr + ox;
                float* yr_ptr = y_ptr + (ox + 1);

                xl0 = *xl_ptr;
                xc0 = *xc_ptr;
                xr0 = *xr_ptr;

                floatx2 y0 = float_linear2d(xl0, xc0, xr0);

                *yl_ptr = y0.imm0;
                *yr_ptr = y0.imm1;
            }

            x_ptr += c * iw;
            y_ptr += c * ow;
        }

        return SUCCESS;
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

    if (iw > MAX_MAP_SIZE || ow > MAX_MAP_SIZE || ih > MAX_MAP_SIZE || oh > MAX_MAP_SIZE) {
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

    if ((c % (AVX2_FLOAT_STRIDE * 4)) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type c32x");
#endif // _DEBUG

        ret = upsample2d_linear_c32x(n, c, iw, ow, ih, oh, x_ptr, y_ptr);
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
