#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_numeric.hpp"
#include "../Inline/inline_loadstore_xn_s.hpp"

#pragma unmanaged

__forceinline floatx2 float_linear1d(float xl, float xc, float xr) {
    float xc2 = xc + xc;
    float yl = xl + xc2;
    float yr = xr + xc2;

    return floatx2(yl, yr);
}

__forceinline __m128x2 _mm128_linear1d_ps(__m128 xl, __m128 xc, __m128 xr) {
    __m128 xc2 = _mm_add_ps(xc, xc);
    __m128 yl = _mm_add_ps(xl, xc2);
    __m128 yr = _mm_add_ps(xr, xc2);

    return __m128x2(yl, yr);
}

__forceinline __m256x2 _mm256_linear1d_ps(__m256 xl, __m256 xc, __m256 xr) {
    __m256 xc2 = _mm256_add_ps(xc, xc);
    __m256 yl = _mm256_add_ps(xl, xc2);
    __m256 yr = _mm256_add_ps(xr, xc2);

    return __m256x2(yl, yr);
}

int upsample1d_linear_c32x(
    const uint n, const uint c,
    const uint iw, const uint ow,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c % (AVX2_FLOAT_STRIDE * 4)) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
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

                __m256x2 y0 = _mm256_linear1d_ps(xl0, xc0, xr0);
                __m256x2 y1 = _mm256_linear1d_ps(xl1, xc1, xr1);
                __m256x2 y2 = _mm256_linear1d_ps(xl2, xc2, xr2);
                __m256x2 y3 = _mm256_linear1d_ps(xl3, xc3, xr3);

                _mm256_stream_x4_ps(yl_ptr, y0.imm0, y1.imm0, y2.imm0, y3.imm0);
                _mm256_stream_x4_ps(yr_ptr, y0.imm1, y1.imm1, y2.imm1, y3.imm1);

                xl_ptr += AVX2_FLOAT_STRIDE * 4;
                xc_ptr += AVX2_FLOAT_STRIDE * 4;
                xr_ptr += AVX2_FLOAT_STRIDE * 4;
                yl_ptr += AVX2_FLOAT_STRIDE * 4;
                yr_ptr += AVX2_FLOAT_STRIDE * 4;
                r -= AVX2_FLOAT_STRIDE * 4;
            }
        }

        x_ptr += c * iw;
        y_ptr += c * ow;
    }

    return SUCCESS;
}

int upsample1d_linear_aligned(
    const uint n, const uint c,
    const uint iw, const uint ow,
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

                __m256x2 y0 = _mm256_linear1d_ps(xl0, xc0, xr0);
                __m256x2 y1 = _mm256_linear1d_ps(xl1, xc1, xr1);
                __m256x2 y2 = _mm256_linear1d_ps(xl2, xc2, xr2);
                __m256x2 y3 = _mm256_linear1d_ps(xl3, xc3, xr3);

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

                __m256x2 y0 = _mm256_linear1d_ps(xl0, xc0, xr0);
                __m256x2 y1 = _mm256_linear1d_ps(xl1, xc1, xr1);
                __m256x2 y2 = _mm256_linear1d_ps(xl2, xc2, xr2);

                _mm256_stream_x3_ps(yl_ptr, y0.imm0, y1.imm0, y2.imm0);
                _mm256_stream_x3_ps(yr_ptr, y0.imm1, y1.imm1, y2.imm1);
            }
            else if (r >= AVX2_FLOAT_STRIDE * 2) {
                _mm256_load_x2_ps(xl_ptr, xl0, xl1);
                _mm256_load_x2_ps(xc_ptr, xc0, xc1);
                _mm256_load_x2_ps(xr_ptr, xr0, xr1);

                __m256x2 y0 = _mm256_linear1d_ps(xl0, xc0, xr0);
                __m256x2 y1 = _mm256_linear1d_ps(xl1, xc1, xr1);

                _mm256_stream_x2_ps(yl_ptr, y0.imm0, y1.imm0);
                _mm256_stream_x2_ps(yr_ptr, y0.imm1, y1.imm1);
            }
            else if (r >= AVX2_FLOAT_STRIDE) {
                _mm256_load_x1_ps(xl_ptr, xl0);
                _mm256_load_x1_ps(xc_ptr, xc0);
                _mm256_load_x1_ps(xr_ptr, xr0);

                __m256x2 y0 = _mm256_linear1d_ps(xl0, xc0, xr0);

                _mm256_stream_x1_ps(yl_ptr, y0.imm0);
                _mm256_stream_x1_ps(yr_ptr, y0.imm1);
            }

        }

        x_ptr += c * iw;
        y_ptr += c * ow;
    }

    return SUCCESS;
}

int upsample1d_linear_unaligned(
    const uint n, const uint c,
    const uint iw, const uint ow,
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

                __m256x2 y0 = _mm256_linear1d_ps(xl0, xc0, xr0);
                __m256x2 y1 = _mm256_linear1d_ps(xl1, xc1, xr1);
                __m256x2 y2 = _mm256_linear1d_ps(xl2, xc2, xr2);
                __m256x2 y3 = _mm256_linear1d_ps(xl3, xc3, xr3);

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

                __m256x2 y0 = _mm256_linear1d_ps(xl0, xc0, xr0);
                __m256x2 y1 = _mm256_linear1d_ps(xl1, xc1, xr1);
                __m256x2 y2 = _mm256_linear1d_ps(xl2, xc2, xr2);
                __m256x2 y3 = _mm256_linear1d_ps(xl3, xc3, xr3);

                _mm256_maskstore_x4_ps(yl_ptr, y0.imm0, y1.imm0, y2.imm0, y3.imm0, mask);
                _mm256_maskstore_x4_ps(yr_ptr, y0.imm1, y1.imm1, y2.imm1, y3.imm1, mask);
            }
            else if (r >= AVX2_FLOAT_STRIDE * 2) {
                _mm256_loadu_x3_ps(xl_ptr, xl0, xl1, xl2);
                _mm256_loadu_x3_ps(xc_ptr, xc0, xc1, xc2);
                _mm256_loadu_x3_ps(xr_ptr, xr0, xr1, xr2);

                __m256x2 y0 = _mm256_linear1d_ps(xl0, xc0, xr0);
                __m256x2 y1 = _mm256_linear1d_ps(xl1, xc1, xr1);
                __m256x2 y2 = _mm256_linear1d_ps(xl2, xc2, xr2);

                _mm256_maskstore_x3_ps(yl_ptr, y0.imm0, y1.imm0, y2.imm0, mask);
                _mm256_maskstore_x3_ps(yr_ptr, y0.imm1, y1.imm1, y2.imm1, mask);
            }
            else if (r >= AVX2_FLOAT_STRIDE) {
                _mm256_loadu_x2_ps(xl_ptr, xl0, xl1);
                _mm256_loadu_x2_ps(xc_ptr, xc0, xc1);
                _mm256_loadu_x2_ps(xr_ptr, xr0, xr1);

                __m256x2 y0 = _mm256_linear1d_ps(xl0, xc0, xr0);
                __m256x2 y1 = _mm256_linear1d_ps(xl1, xc1, xr1);

                _mm256_maskstore_x2_ps(yl_ptr, y0.imm0, y1.imm0, mask);
                _mm256_maskstore_x2_ps(yr_ptr, y0.imm1, y1.imm1, mask);
            }
            else if (r > 0) {
                _mm256_loadu_x1_ps(xl_ptr, xl0);
                _mm256_loadu_x1_ps(xc_ptr, xc0);
                _mm256_loadu_x1_ps(xr_ptr, xr0);

                __m256x2 y0 = _mm256_linear1d_ps(xl0, xc0, xr0);

                _mm256_maskstore_x1_ps(yl_ptr, y0.imm0, mask);
                _mm256_maskstore_x1_ps(yr_ptr, y0.imm1, mask);
            }
        }

        x_ptr += c * iw;
        y_ptr += c * ow;
    }

    return SUCCESS;
}

int upsample1d_linear_cleq8(
    const uint n, const uint c,
    const uint iw, const uint ow,
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

                __m256x2 y0 = _mm256_linear1d_ps(xl0, xc0, xr0);

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

                __m256x2 y0 = _mm256_linear1d_ps(xl0, xc0, xr0);

                _mm256_maskstore_x1_ps(yl_ptr, y0.imm0, mask);
                _mm256_maskstore_x1_ps(yr_ptr, y0.imm1, mask);
            }

            x_ptr += c * iw;
            y_ptr += c * ow;
        }

        return SUCCESS;
    }

    if(c == AVX1_FLOAT_STRIDE) {
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

                __m128x2 y0 = _mm128_linear1d_ps(xl0, xc0, xr0);

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

                __m128x2 y0 = _mm128_linear1d_ps(xl0, xc0, xr0);

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

                floatx2 y0 = float_linear1d(xl0, xc0, xr0);

                *yl_ptr = y0.imm0;
                *yr_ptr = y0.imm1;
            }

            x_ptr += iw;
            y_ptr += ow;
        }

        return SUCCESS;
    }

    return FAILURE_BADPARAM;
}

#pragma managed

void AvxBlas::Upsample1D::LinearX2(
    UInt32 n, UInt32 c, UInt32 iw,
    Array<float>^ x, Array<float>^ y) {

    if (n > MAX_BATCHES) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidBatches);
    }
    if (c > MAX_CHANNELS) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidChannels);
    }
    if (n <= 0 || c <= 0 || iw <= 0) {
        return;
    }

    Util::CheckProdOverflow(iw, 2u);
    UInt32 ow = iw * 2;

    if (iw > MAX_MAP_SIZE || ow > MAX_MAP_SIZE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidDataSize);
    }

    Util::CheckProdOverflow(n, c, iw);
    Util::CheckProdOverflow(n, c, ow);

    Util::CheckLength(n * c * iw, x);
    Util::CheckLength(n * c * ow, y);

    Util::CheckDuplicateArray(x, y);

    const float* x_ptr = (const float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    int ret = UNEXECUTED;

    if (c <= AVX2_FLOAT_STRIDE) {
#ifdef _DEBUG
        Console::WriteLine("type leq8");
#endif // _DEBUG

        ret = upsample1d_linear_cleq8(n, c, iw, ow, x_ptr, y_ptr);
    }
    else if ((c % (AVX2_FLOAT_STRIDE * 4)) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type c32x");
#endif // _DEBUG

        ret = upsample1d_linear_c32x(n, c, iw, ow, x_ptr, y_ptr);
    }
    else if ((c & AVX2_FLOAT_REMAIN_MASK) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = upsample1d_linear_aligned(n, c, iw, ow, x_ptr, y_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = upsample1d_linear_unaligned(n, c, iw, ow, x_ptr, y_ptr);
    }
    if (ret != SUCCESS) {
        Util::AssertReturnCode(ret);
        return;
    }

    Constant::Mul(n * c * ow, y, 1.0f / 3.0f, y);
}
