#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_ope_s.hpp"

using namespace System;

#pragma unmanaged

int ew_abs_s(
    const uint n,
    INPTR(float) x_ptr, OUTPTR(float) y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        __m256 x0 = _mm256_load_ps(x_ptr);
        __m256 x1 = _mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE);
        __m256 x2 = _mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE * 2);
        __m256 x3 = _mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE * 3);

        __m256 y0 = _mm256_abs_ps(x0);
        __m256 y1 = _mm256_abs_ps(x1);
        __m256 y2 = _mm256_abs_ps(x2);
        __m256 y3 = _mm256_abs_ps(x3);

        _mm256_stream_ps(y_ptr, y0);
        _mm256_stream_ps(y_ptr + AVX2_FLOAT_STRIDE, y1);
        _mm256_stream_ps(y_ptr + AVX2_FLOAT_STRIDE * 2, y2);
        _mm256_stream_ps(y_ptr + AVX2_FLOAT_STRIDE * 3, y3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 2) {
        __m256 x0 = _mm256_load_ps(x_ptr);
        __m256 x1 = _mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE);

        __m256 y0 = _mm256_abs_ps(x0);
        __m256 y1 = _mm256_abs_ps(x1);

        _mm256_stream_ps(y_ptr, y0);
        _mm256_stream_ps(y_ptr + 8, y1);

        x_ptr += AVX2_FLOAT_STRIDE * 2;
        y_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE) {
        __m256 x0 = _mm256_load_ps(x_ptr);

        __m256 y0 = _mm256_abs_ps(x0);

        _mm256_stream_ps(y_ptr, y0);

        x_ptr += AVX2_FLOAT_STRIDE;
        y_ptr += AVX2_FLOAT_STRIDE;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps(r);

        __m256 x = _mm256_maskload_ps(x_ptr, mask);

        __m256 y = _mm256_abs_ps(x);

        _mm256_maskstore_ps(y_ptr, mask, y);
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Elementwise::Abs(UInt32 n, Array<float>^ x, Array<float>^ y) {
    if (n <= 0) {
        return;
    }

    Util::CheckLength(n, x, y);

    const float* x_ptr = (const float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    int ret = ew_abs_s(n, x_ptr, y_ptr);

    Util::AssertReturnCode(ret);
}