#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include "../Inline/inline_loadstore_xn_d.hpp"

using namespace System;

#pragma unmanaged

int cast_double_to_float(
    const uint n,
    indoubles x_ptr, outfloats y_ptr) {

    if ((size_t)x_ptr == (size_t)y_ptr) {
        return FAILURE_BADPARAM;
    }

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256d x0, x1, x2, x3;
    __m128 y0, y1, y2, y3;

    uint r = n;

    while (r >= AVX2_DOUBLE_STRIDE * 4) {
        _mm256_load_x4_pd(x_ptr, x0, x1, x2, x3);

        y0 = _mm256_cvtpd_ps(x0);
        y1 = _mm256_cvtpd_ps(x1);
        y2 = _mm256_cvtpd_ps(x2);
        y3 = _mm256_cvtpd_ps(x3);

        _mm_stream_ps(y_ptr, y0);
        _mm_stream_ps(y_ptr + AVX2_FLOAT_STRIDE / 2, y1);
        _mm_stream_ps(y_ptr + AVX2_FLOAT_STRIDE, y2);
        _mm_stream_ps(y_ptr + AVX2_FLOAT_STRIDE * 3 / 2, y3);

        x_ptr += AVX2_DOUBLE_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE * 4;
    }
    if (r >= AVX2_DOUBLE_STRIDE * 2) {
        _mm256_load_x2_pd(x_ptr, x0, x1);

        y0 = _mm256_cvtpd_ps(x0);
        y1 = _mm256_cvtpd_ps(x1);

        _mm_stream_ps(y_ptr, y0);
        _mm_stream_ps(y_ptr + AVX2_FLOAT_STRIDE / 2, y1);

        x_ptr += AVX2_DOUBLE_STRIDE * 2;
        y_ptr += AVX2_FLOAT_STRIDE;
        r -= AVX2_DOUBLE_STRIDE * 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE) {
        _mm256_load_x1_pd(x_ptr, x0);

        y0 = _mm256_cvtpd_ps(x0);

        _mm_stream_ps(y_ptr, y0);

        x_ptr += AVX2_DOUBLE_STRIDE;
        y_ptr += AVX2_FLOAT_STRIDE / 2;
        r -= AVX2_DOUBLE_STRIDE;
    }
    if (r > 0) {
        const __m128i mask_ps = _mm_setmask_ps(r);
        const __m256i mask_pd = _mm256_setmask_pd(r);

        _mm256_maskload_x1_pd(x_ptr, x0, mask_pd);

        y0 = _mm256_cvtpd_ps(x0);

        _mm_maskstore_ps(y_ptr, mask_ps, y0);
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Elementwise::Cast(UInt32 n, Array<double>^ x, Array<float>^ y) {
    if (n <= 0) {
        return;
    }

    Util::CheckLength(n, x);
    Util::CheckLength(n, y);

    const double* x_ptr = (const double*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    int ret = cast_double_to_float(n, x_ptr, y_ptr);

    Util::AssertReturnCode(ret);
}