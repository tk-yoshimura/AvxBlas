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

__forceinline __m128x2 _mm_linear1d_ps(__m128 xl, __m128 xc, __m128 xr) {
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

int upsample1d_linear_aligned(
    const uint n, const uint c,
    const uint iw, const uint ow,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 xl, xc, xr;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
            const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);

            uint r = c;

            const float* xl_ptr = x_ptr + c * ixl;
            const float* xc_ptr = x_ptr + c * ix;
            const float* xr_ptr = x_ptr + c * ixr;

            float* yl_ptr = y_ptr + c * ox;
            float* yr_ptr = yl_ptr + c;

            while (r >= AVX2_FLOAT_STRIDE) {
                _mm256_load_x1_ps(xl_ptr, xl);
                _mm256_load_x1_ps(xc_ptr, xc);
                _mm256_load_x1_ps(xr_ptr, xr);

                __m256x2 y = _mm256_linear1d_ps(xl, xc, xr);

                _mm256_stream_x1_ps(yl_ptr, y.imm0);
                _mm256_stream_x1_ps(yr_ptr, y.imm1);

                xl_ptr += AVX2_FLOAT_STRIDE;
                xc_ptr += AVX2_FLOAT_STRIDE;
                xr_ptr += AVX2_FLOAT_STRIDE;
                yl_ptr += AVX2_FLOAT_STRIDE;
                yr_ptr += AVX2_FLOAT_STRIDE;
                r -= AVX2_FLOAT_STRIDE;
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

    __m256 xl, xc, xr;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
            const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);

            uint r = c;

            const float* xl_ptr = x_ptr + c * ixl;
            const float* xc_ptr = x_ptr + c * ix;
            const float* xr_ptr = x_ptr + c * ixr;

            float* yl_ptr = y_ptr + c * ox;
            float* yr_ptr = yl_ptr + c;

            while (r >= AVX2_FLOAT_STRIDE) {
                _mm256_loadu_x1_ps(xl_ptr, xl);
                _mm256_loadu_x1_ps(xc_ptr, xc);
                _mm256_loadu_x1_ps(xr_ptr, xr);

                __m256x2 y = _mm256_linear1d_ps(xl, xc, xr);

                _mm256_storeu_x1_ps(yl_ptr, y.imm0);
                _mm256_storeu_x1_ps(yr_ptr, y.imm1);

                xl_ptr += AVX2_FLOAT_STRIDE;
                xc_ptr += AVX2_FLOAT_STRIDE;
                xr_ptr += AVX2_FLOAT_STRIDE;
                yl_ptr += AVX2_FLOAT_STRIDE;
                yr_ptr += AVX2_FLOAT_STRIDE;
                r -= AVX2_FLOAT_STRIDE;
            }
            if (r > 0) {
                _mm256_loadu_x1_ps(xl_ptr, xl);
                _mm256_loadu_x1_ps(xc_ptr, xc);
                _mm256_loadu_x1_ps(xr_ptr, xr);

                __m256x2 y = _mm256_linear1d_ps(xl, xc, xr);

                _mm256_maskstore_x1_ps(yl_ptr, y.imm0, mask);
                _mm256_maskstore_x1_ps(yr_ptr, y.imm1, mask);
            }
        }

        x_ptr += c * iw;
        y_ptr += c * ow;
    }

    return SUCCESS;
}

int upsample1d_linear_c1(
    const uint n, const uint c,
    const uint iw, const uint ow,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c != 1) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    float xl, xc, xr;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
            const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);

            const float* xl_ptr = x_ptr + ixl;
            const float* xc_ptr = x_ptr + ix;
            const float* xr_ptr = x_ptr + ixr;

            float* yl_ptr = y_ptr + ox;
            float* yr_ptr = yl_ptr + 1;

            xl = *xl_ptr;
            xc = *xc_ptr;
            xr = *xr_ptr;

            floatx2 y = float_linear1d(xl, xc, xr);

            *yl_ptr = y.imm0;
            *yr_ptr = y.imm1;
        }

        x_ptr += iw;
        y_ptr += ow;
    }

    return SUCCESS;
}

int upsample1d_linear_c2to3(
    const uint n, const uint c,
    const uint iw, const uint ow,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c <= 1 && c >= AVX1_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m128i mask = _mm_setmask_ps(c);

    __m128 xl, xc, xr;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
            const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);

            const float* xl_ptr = x_ptr + c * ixl;
            const float* xc_ptr = x_ptr + c * ix;
            const float* xr_ptr = x_ptr + c * ixr;

            float* yl_ptr = y_ptr + c * ox;
            float* yr_ptr = yl_ptr + c;

            xl = _mm_loadu_ps(xl_ptr);
            xc = _mm_loadu_ps(xc_ptr);
            xr = _mm_loadu_ps(xr_ptr);

            __m128x2 y = _mm_linear1d_ps(xl, xc, xr);

            _mm_maskstore_ps(yl_ptr, mask, y.imm0);
            _mm_maskstore_ps(yr_ptr, mask, y.imm1);
        }

        x_ptr += c * iw;
        y_ptr += c * ow;
    }

    return SUCCESS;
}

int upsample1d_linear_c4(
    const uint n, const uint c,
    const uint iw, const uint ow,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c != AVX1_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m128 xl, xc, xr;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
            const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);

            const float* xl_ptr = x_ptr + c * ixl;
            const float* xc_ptr = x_ptr + c * ix;
            const float* xr_ptr = x_ptr + c * ixr;

            float* yl_ptr = y_ptr + c * ox;
            float* yr_ptr = yl_ptr + c;

            xl = _mm_load_ps(xl_ptr);
            xc = _mm_load_ps(xc_ptr);
            xr = _mm_load_ps(xr_ptr);

            __m128x2 y = _mm_linear1d_ps(xl, xc, xr);

            _mm_stream_ps(yl_ptr, y.imm0);
            _mm_stream_ps(yr_ptr, y.imm1);
        }

        x_ptr += c * iw;
        y_ptr += c * ow;
    }

    return SUCCESS;
}

int upsample1d_linear_c5to7(
    const uint n, const uint c,
    const uint iw, const uint ow,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c <= AVX1_FLOAT_STRIDE || c >= AVX2_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(c);

    __m256 xl, xc, xr;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
            const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);

            const float* xl_ptr = x_ptr + c * ixl;
            const float* xc_ptr = x_ptr + c * ix;
            const float* xr_ptr = x_ptr + c * ixr;

            float* yl_ptr = y_ptr + c * ox;
            float* yr_ptr = yl_ptr + c;

            _mm256_loadu_x1_ps(xl_ptr, xl);
            _mm256_loadu_x1_ps(xc_ptr, xc);
            _mm256_loadu_x1_ps(xr_ptr, xr);

            __m256x2 y = _mm256_linear1d_ps(xl, xc, xr);

            _mm256_maskstore_x1_ps(yl_ptr, y.imm0, mask);
            _mm256_maskstore_x1_ps(yr_ptr, y.imm1, mask);
        }

        x_ptr += c * iw;
        y_ptr += c * ow;
    }

    return SUCCESS;
}

int upsample1d_linear_c8(
    const uint n, const uint c,
    const uint iw, const uint ow,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c != AVX2_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 xl, xc, xr;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ix < iw; ix++, ox += 2) {
            const uint ixl = max(ix, 1u) - 1u, ixr = min(ix + 1u, iw - 1u);

            const float* xl_ptr = x_ptr + c * ixl;
            const float* xc_ptr = x_ptr + c * ix;
            const float* xr_ptr = x_ptr + c * ixr;

            float* yl_ptr = y_ptr + c * ox;
            float* yr_ptr = yl_ptr + c;

            _mm256_load_x1_ps(xl_ptr, xl);
            _mm256_load_x1_ps(xc_ptr, xc);
            _mm256_load_x1_ps(xr_ptr, xr);

            __m256x2 y = _mm256_linear1d_ps(xl, xc, xr);

            _mm256_stream_x1_ps(yl_ptr, y.imm0);
            _mm256_stream_x1_ps(yr_ptr, y.imm1);
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

    if (c == 1) {
        return upsample1d_linear_c1(n, c, iw, ow, x_ptr, y_ptr);
    }
    if (c < AVX1_FLOAT_STRIDE) {
        return upsample1d_linear_c2to3(n, c, iw, ow, x_ptr, y_ptr);
    }
    if (c == AVX1_FLOAT_STRIDE) {
        return upsample1d_linear_c4(n, c, iw, ow, x_ptr, y_ptr);
    }
    if (c < AVX2_FLOAT_STRIDE) {
        return upsample1d_linear_c5to7(n, c, iw, ow, x_ptr, y_ptr);
    }
    if (c == AVX2_FLOAT_STRIDE) {
        return upsample1d_linear_c8(n, c, iw, ow, x_ptr, y_ptr);
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

    if (ow > MAX_MAP_SIZE) {
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
