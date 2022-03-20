#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"

using namespace System;

#pragma unmanaged

int ew_copy_s(
    const uint n,
    INPTR(float) x_ptr, OUTPTR(float) y_ptr) {

    if (x_ptr == y_ptr) {
        return SUCCESS;
    }

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

        _mm256_stream_ps(y_ptr, x0);
        _mm256_stream_ps(y_ptr + AVX2_FLOAT_STRIDE, x1);
        _mm256_stream_ps(y_ptr + AVX2_FLOAT_STRIDE * 2, x2);
        _mm256_stream_ps(y_ptr + AVX2_FLOAT_STRIDE * 3, x3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 2) {
        __m256 x0 = _mm256_load_ps(x_ptr);
        __m256 x1 = _mm256_load_ps(x_ptr + AVX2_FLOAT_STRIDE);

        _mm256_stream_ps(y_ptr, x0);
        _mm256_stream_ps(y_ptr + AVX2_FLOAT_STRIDE, x1);

        x_ptr += AVX2_FLOAT_STRIDE * 2;
        y_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE) {
        __m256 x0 = _mm256_load_ps(x_ptr);

        _mm256_stream_ps(y_ptr, x0);

        x_ptr += AVX2_FLOAT_STRIDE;
        y_ptr += AVX2_FLOAT_STRIDE;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps(r);

        __m256 x = _mm256_maskload_ps(x_ptr, mask);

        _mm256_maskstore_ps(y_ptr, mask, x);
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Elementwise::Copy(UInt32 n, Array<float>^ x, Array<float>^ y) {
    if (n <= 0) {
        return;
    }

    Util::CheckLength(n, x, y);

    const float* x_ptr = (const float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    int ret = ew_copy_s(n, x_ptr, y_ptr);

    Util::AssertReturnCode(ret);
}