#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"

using namespace System;

#pragma unmanaged

int ew_add_s(
    const unsigned int n,
    const float* __restrict x1_ptr, const float* __restrict x2_ptr, float* __restrict y_ptr) {

#ifdef _DEBUG
    if (((size_t)x1_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)x2_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    unsigned int r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        __m256 x01 = _mm256_load_ps(x1_ptr);
        __m256 x02 = _mm256_load_ps(x2_ptr);
        __m256 x11 = _mm256_load_ps(x1_ptr + AVX2_FLOAT_STRIDE);
        __m256 x12 = _mm256_load_ps(x2_ptr + AVX2_FLOAT_STRIDE);
        __m256 x21 = _mm256_load_ps(x1_ptr + AVX2_FLOAT_STRIDE * 2);
        __m256 x22 = _mm256_load_ps(x2_ptr + AVX2_FLOAT_STRIDE * 2);
        __m256 x31 = _mm256_load_ps(x1_ptr + AVX2_FLOAT_STRIDE * 3);
        __m256 x32 = _mm256_load_ps(x2_ptr + AVX2_FLOAT_STRIDE * 3);

        __m256 y0 = _mm256_add_ps(x01, x02);
        __m256 y1 = _mm256_add_ps(x11, x12);
        __m256 y2 = _mm256_add_ps(x21, x22);
        __m256 y3 = _mm256_add_ps(x31, x32);

        _mm256_stream_ps(y_ptr, y0);
        _mm256_stream_ps(y_ptr + AVX2_FLOAT_STRIDE, y1);
        _mm256_stream_ps(y_ptr + AVX2_FLOAT_STRIDE * 2, y2);
        _mm256_stream_ps(y_ptr + AVX2_FLOAT_STRIDE * 3, y3);

        x1_ptr += AVX2_FLOAT_STRIDE * 4;
        x2_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 2) {
        __m256 x01 = _mm256_load_ps(x1_ptr);
        __m256 x02 = _mm256_load_ps(x2_ptr);
        __m256 x11 = _mm256_load_ps(x1_ptr + AVX2_FLOAT_STRIDE);
        __m256 x12 = _mm256_load_ps(x2_ptr + AVX2_FLOAT_STRIDE);

        __m256 y0 = _mm256_add_ps(x01, x02);
        __m256 y1 = _mm256_add_ps(x11, x12);

        _mm256_stream_ps(y_ptr, y0);
        _mm256_stream_ps(y_ptr + AVX2_FLOAT_STRIDE, y1);

        x1_ptr += AVX2_FLOAT_STRIDE * 2;
        x2_ptr += AVX2_FLOAT_STRIDE * 2;
        y_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE) {
        __m256 x01 = _mm256_load_ps(x1_ptr);
        __m256 x02 = _mm256_load_ps(x2_ptr);

        __m256 y0 = _mm256_add_ps(x01, x02);

        _mm256_stream_ps(y_ptr, y0);

        x1_ptr += AVX2_FLOAT_STRIDE;
        x2_ptr += AVX2_FLOAT_STRIDE;
        y_ptr += AVX2_FLOAT_STRIDE;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps(r);

        __m256 x1 = _mm256_maskload_ps(x1_ptr, mask);
        __m256 x2 = _mm256_maskload_ps(x2_ptr, mask);

        __m256 y = _mm256_add_ps(x1, x2);

        _mm256_maskstore_ps(y_ptr, mask, y);
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Elementwise::Add(UInt32 n, Array<float>^ x1, Array<float>^ x2, Array<float>^ y) {
    if (n <= 0) {
        return;
    }

    Util::CheckLength(n, x1, x2, y);

    const float* x1_ptr = (const float*)(x1->Ptr.ToPointer());
    const float* x2_ptr = (const float*)(x2->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    ew_add_s(n, x1_ptr, x2_ptr, y_ptr);
}
