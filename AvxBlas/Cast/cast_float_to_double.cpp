#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include "../Inline/inline_loadstore_xn_d.hpp"

using namespace System;

#pragma unmanaged

int cast_float_to_double(
    const uint n,
    infloats x_ptr, outdoubles y_ptr) {

    if ((size_t)x_ptr == (size_t)y_ptr) {
        return FAILURE_BADPARAM;
    }

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m128 x0, x1, x2, x3;
    __m256d y0, y1, y2, y3;

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE * 2) {
        x0 = _mm_load_ps(x_ptr);
        x1 = _mm_load_ps(x_ptr + AVX2_FLOAT_STRIDE / 2);
        x2 = _mm_load_ps(x_ptr + AVX2_FLOAT_STRIDE);
        x3 = _mm_load_ps(x_ptr + AVX2_FLOAT_STRIDE * 3 / 2);

        y0 = _mm256_cvtps_pd(x0);
        y1 = _mm256_cvtps_pd(x1);
        y2 = _mm256_cvtps_pd(x2);
        y3 = _mm256_cvtps_pd(x3);

        _mm256_stream_x4_pd(y_ptr, y0, y1, y2, y3);

        x_ptr += AVX2_FLOAT_STRIDE * 2;
        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE) {
        x0 = _mm_load_ps(x_ptr);
        x1 = _mm_load_ps(x_ptr + AVX2_FLOAT_STRIDE / 2);

        y0 = _mm256_cvtps_pd(x0);
        y1 = _mm256_cvtps_pd(x1);

        _mm256_stream_x2_pd(y_ptr, y0, y1);

        x_ptr += AVX2_FLOAT_STRIDE;
        y_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r >= AVX2_FLOAT_STRIDE / 2) {
        x0 = _mm_load_ps(x_ptr);

        y0 = _mm256_cvtps_pd(x0);

        _mm256_stream_x1_pd(y_ptr, y0);

        x_ptr += AVX2_FLOAT_STRIDE / 2;
        y_ptr += AVX2_DOUBLE_STRIDE;
        r -= AVX2_FLOAT_STRIDE / 2;
    }
    if (r > 0) {
        const __m128i mask_ps = _mm_setmask_ps(r);
        const __m256i mask_pd = _mm256_setmask_pd(r);

        x0 = _mm_maskload_ps(x_ptr, mask_ps);

        y0 = _mm256_cvtps_pd(x0);

        _mm256_maskstore_x1_pd(y_ptr, y0, mask_pd);
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Cast::AsType(UInt32 n, Array<float>^ x, Array<double>^ y) {
    if (n <= 0) {
        return;
    }

    Util::CheckLength(n, x);
    Util::CheckLength(n, y);

    const float* x_ptr = (const float*)(x->Ptr.ToPointer());
    double* y_ptr = (double*)(y->Ptr.ToPointer());

    int ret = cast_float_to_double(n, x_ptr, y_ptr);

    Util::AssertReturnCode(ret);
}