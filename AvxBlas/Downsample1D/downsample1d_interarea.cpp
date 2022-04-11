#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_numeric.hpp"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include "../Inline/inline_sum_s.hpp"

#pragma unmanaged

__forceinline __m128 _mm_interarea1d_ps(__m128 xl, __m128 xr) {
    __m128 y = _mm_add_ps(xl, xr);

    return y;
}

__forceinline __m256 _mm256_interarea1d_ps(__m256 xl, __m256 xr) {
    __m256 y = _mm256_add_ps(xl, xr);

    return y;
}

int downsample1d_interarea_aligned(
    const uint n, const uint c,
    const uint iw, const uint ow,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) != 0 || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 xl, xr;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ox < ow; ix += 2, ox++) {

            const float* xl_ptr = x_ptr + c * ix;
            const float* xr_ptr = xl_ptr + c;

            float* yc_ptr = y_ptr + c * ox;

            uint r = c;

            while (r >= AVX2_FLOAT_STRIDE) {
                _mm256_load_x1_ps(xl_ptr, xl);
                _mm256_load_x1_ps(xr_ptr, xr);

                __m256 y = _mm256_interarea1d_ps(xl, xr);

                _mm256_stream_x1_ps(yc_ptr, y);

                xl_ptr += AVX2_FLOAT_STRIDE;
                xr_ptr += AVX2_FLOAT_STRIDE;
                yc_ptr += AVX2_FLOAT_STRIDE;
                r -= AVX2_FLOAT_STRIDE;
            }
        }

        x_ptr += c * iw;
        y_ptr += c * ow;
    }

    return SUCCESS;
}

int downsample1d_interarea_unaligned(
    const uint n, const uint c,
    const uint iw, const uint ow,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if ((c & AVX2_FLOAT_REMAIN_MASK) == 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(c & AVX2_FLOAT_REMAIN_MASK);

    __m256 xl, xr;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ox < ow; ix += 2, ox++) {

            const float* xl_ptr = x_ptr + c * ix;
            const float* xr_ptr = xl_ptr + c;

            float* yc_ptr = y_ptr + c * ox;

            uint r = c;

            while (r >= AVX2_FLOAT_STRIDE) {
                _mm256_loadu_x1_ps(xl_ptr, xl);
                _mm256_loadu_x1_ps(xr_ptr, xr);

                __m256 y = _mm256_interarea1d_ps(xl, xr);

                _mm256_storeu_x1_ps(yc_ptr, y);

                xl_ptr += AVX2_FLOAT_STRIDE;
                xr_ptr += AVX2_FLOAT_STRIDE;
                yc_ptr += AVX2_FLOAT_STRIDE;
                r -= AVX2_FLOAT_STRIDE;
            }
            if (r > 0) {
                _mm256_loadu_x1_ps(xl_ptr, xl);
                _mm256_loadu_x1_ps(xr_ptr, xr);

                __m256 y = _mm256_interarea1d_ps(xl, xr);

                _mm256_maskstore_x1_ps(yc_ptr, y, mask);
            }
        }

        x_ptr += c * iw;
        y_ptr += c * ow;
    }

    return SUCCESS;
}

int downsample1d_interarea_c1(
    const uint n, const uint c,
    const uint iw, const uint ow,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c != 1) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(ow & AVX2_FLOAT_REMAIN_MASK);

    __m256 y;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ox < ow; ix += AVX2_FLOAT_STRIDE * 2, ox += AVX2_FLOAT_STRIDE) {

            const float* xl_ptr = x_ptr + ix;

            float* yc_ptr = y_ptr + ox;

            if (ix + AVX2_FLOAT_STRIDE < iw) {
                __m256 x0 = _mm256_loadu_ps(xl_ptr);
                __m256 x1 = _mm256_loadu_ps(xl_ptr + AVX2_FLOAT_STRIDE);

                y = _mm256_hadd2_x2_ps(x0, x1);
            }
            else {
                __m256 x0 = _mm256_loadu_ps(xl_ptr);

                y = _mm256_hadd2_x2_ps(x0, x0);
            }

            if (ox + AVX2_FLOAT_STRIDE <= ow) {
                _mm256_storeu_x1_ps(yc_ptr, y);
            }
            else {
                _mm256_maskstore_x1_ps(yc_ptr, y, mask);
            }
        }

        x_ptr += iw;
        y_ptr += ow;
    }

    return SUCCESS;
}

int downsample1d_interarea_c2to3(
    const uint n, const uint c,
    const uint iw, const uint ow,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c <= 1 || c >= AVX1_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m128i mask = _mm_setmask_ps(c & AVX2_FLOAT_REMAIN_MASK);

    __m128 xl, xr;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ox < ow; ix += 2, ox++) {

            const float* xl_ptr = x_ptr + c * ix;
            const float* xr_ptr = xl_ptr + c;

            float* yc_ptr = y_ptr + c * ox;

            xl = _mm_loadu_ps(xl_ptr);
            xr = _mm_loadu_ps(xr_ptr);

            __m128 y = _mm_interarea1d_ps(xl, xr);

            _mm_maskstore_ps(yc_ptr, mask, y);
        }

        x_ptr += c * iw;
        y_ptr += c * ow;
    }

    return SUCCESS;
}

int downsample1d_interarea_c4(
    const uint n, const uint c,
    const uint iw, const uint ow,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c != AVX1_FLOAT_STRIDE || ((size_t)x_ptr % AVX1_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX1_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m128 xl, xr;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ox < ow; ix += 2, ox++) {

            const float* xl_ptr = x_ptr + c * ix;
            const float* xr_ptr = xl_ptr + c;

            float* yc_ptr = y_ptr + c * ox;

            xl = _mm_load_ps(xl_ptr);
            xr = _mm_load_ps(xr_ptr);

            __m128 y = _mm_interarea1d_ps(xl, xr);

            _mm_stream_ps(yc_ptr, y);
        }

        x_ptr += c * iw;
        y_ptr += c * ow;
    }

    return SUCCESS;
}

int downsample1d_interarea_c5to7(
    const uint n, const uint c,
    const uint iw, const uint ow,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c <= AVX1_FLOAT_STRIDE || c >= AVX2_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256i mask = _mm256_setmask_ps(c);

    __m256 xl, xr;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ox < ow; ix += 2, ox++) {

            const float* xl_ptr = x_ptr + c * ix;
            const float* xr_ptr = xl_ptr + c;

            float* yc_ptr = y_ptr + c * ox;

            _mm256_loadu_x1_ps(xl_ptr, xl);
            _mm256_loadu_x1_ps(xr_ptr, xr);

            __m256 y = _mm256_interarea1d_ps(xl, xr);

            _mm256_maskstore_x1_ps(yc_ptr, y, mask);
        }

        x_ptr += c * iw;
        y_ptr += c * ow;
    }

    return SUCCESS;
}

int downsample1d_interarea_c8(
    const uint n, const uint c,
    const uint iw, const uint ow,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (c != AVX2_FLOAT_STRIDE || ((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 xl, xr;

    for (uint i = 0; i < n; i++) {
        for (uint ix = 0, ox = 0; ox < ow; ix += 2, ox++) {

            const float* xl_ptr = x_ptr + c * ix;
            const float* xr_ptr = xl_ptr + c;

            float* yc_ptr = y_ptr + c * ox;

            _mm256_load_x1_ps(xl_ptr, xl);
            _mm256_load_x1_ps(xr_ptr, xr);

            __m256 y = _mm256_interarea1d_ps(xl, xr);

            _mm256_stream_ps(yc_ptr, y);
        }

        x_ptr += c * iw;
        y_ptr += c * ow;
    }

    return SUCCESS;
}

int downsample1d_interarea_cleq8(
    const uint n, const uint c,
    const uint iw, const uint ow,
    infloats x_ptr, outfloats y_ptr) {

    if (c == 1) {
        return downsample1d_interarea_c1(n, c, iw, ow, x_ptr, y_ptr);
    }
    if (c < AVX1_FLOAT_STRIDE) {
        return downsample1d_interarea_c2to3(n, c, iw, ow, x_ptr, y_ptr);
    }
    if (c == AVX1_FLOAT_STRIDE) {
        return downsample1d_interarea_c4(n, c, iw, ow, x_ptr, y_ptr);
    }
    if (c < AVX2_FLOAT_STRIDE) {
        return downsample1d_interarea_c5to7(n, c, iw, ow, x_ptr, y_ptr);
    }
    if (c == AVX2_FLOAT_STRIDE) {
        return downsample1d_interarea_c8(n, c, iw, ow, x_ptr, y_ptr);
    }

    return FAILURE_BADPARAM;
}

#pragma managed

void AvxBlas::Downsample1D::InterareaX2(
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
    if ((iw & 1u) != 0 || iw > MAX_MAP_SIZE) {
        throw gcnew System::ArgumentOutOfRangeException(ErrorMessage::InvalidDataSize);
    }

    UInt32 ow = iw / 2;

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

        ret = downsample1d_interarea_cleq8(n, c, iw, ow, x_ptr, y_ptr);
    }
    else if ((c & AVX2_FLOAT_REMAIN_MASK) == 0) {
#ifdef _DEBUG
        Console::WriteLine("type aligned");
#endif // _DEBUG

        ret = downsample1d_interarea_aligned(n, c, iw, ow, x_ptr, y_ptr);
    }
    else {
#ifdef _DEBUG
        Console::WriteLine("type unaligned");
#endif // _DEBUG

        ret = downsample1d_interarea_unaligned(n, c, iw, ow, x_ptr, y_ptr);
    }
    if (ret != SUCCESS) {
        Util::AssertReturnCode(ret);
        return;
    }

    Constant::Mul(n * c * ow, y, 1.0f / 2.0f, y);
}
