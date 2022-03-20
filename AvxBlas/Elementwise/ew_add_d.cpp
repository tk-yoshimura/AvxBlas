#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"

using namespace System;

#pragma unmanaged

int ew_add_d(
    const uint n,
    indoubles x1_ptr, indoubles x2_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (((size_t)x1_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)x2_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    uint r = n;

    while (r >= AVX2_DOUBLE_STRIDE * 4) {
        __m256d x01 = _mm256_load_pd(x1_ptr);
        __m256d x02 = _mm256_load_pd(x2_ptr);
        __m256d x11 = _mm256_load_pd(x1_ptr + AVX2_DOUBLE_STRIDE);
        __m256d x12 = _mm256_load_pd(x2_ptr + AVX2_DOUBLE_STRIDE);
        __m256d x21 = _mm256_load_pd(x1_ptr + AVX2_DOUBLE_STRIDE * 2);
        __m256d x22 = _mm256_load_pd(x2_ptr + AVX2_DOUBLE_STRIDE * 2);
        __m256d x31 = _mm256_load_pd(x1_ptr + AVX2_DOUBLE_STRIDE * 3);
        __m256d x32 = _mm256_load_pd(x2_ptr + AVX2_DOUBLE_STRIDE * 3);

        __m256d y0 = _mm256_add_pd(x01, x02);
        __m256d y1 = _mm256_add_pd(x11, x12);
        __m256d y2 = _mm256_add_pd(x21, x22);
        __m256d y3 = _mm256_add_pd(x31, x32);

        _mm256_stream_pd(y_ptr, y0);
        _mm256_stream_pd(y_ptr + AVX2_DOUBLE_STRIDE, y1);
        _mm256_stream_pd(y_ptr + AVX2_DOUBLE_STRIDE * 2, y2);
        _mm256_stream_pd(y_ptr + AVX2_DOUBLE_STRIDE * 3, y3);

        x1_ptr += AVX2_DOUBLE_STRIDE * 4;
        x2_ptr += AVX2_DOUBLE_STRIDE * 4;
        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 4;
    }
    if (r >= AVX2_DOUBLE_STRIDE * 2) {
        __m256d x01 = _mm256_load_pd(x1_ptr);
        __m256d x02 = _mm256_load_pd(x2_ptr);
        __m256d x11 = _mm256_load_pd(x1_ptr + AVX2_DOUBLE_STRIDE);
        __m256d x12 = _mm256_load_pd(x2_ptr + AVX2_DOUBLE_STRIDE);

        __m256d y0 = _mm256_add_pd(x01, x02);
        __m256d y1 = _mm256_add_pd(x11, x12);

        _mm256_stream_pd(y_ptr, y0);
        _mm256_stream_pd(y_ptr + AVX2_DOUBLE_STRIDE, y1);

        x1_ptr += AVX2_DOUBLE_STRIDE * 2;
        x2_ptr += AVX2_DOUBLE_STRIDE * 2;
        y_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE * 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE) {
        __m256d x01 = _mm256_load_pd(x1_ptr);
        __m256d x02 = _mm256_load_pd(x2_ptr);

        __m256d y0 = _mm256_add_pd(x01, x02);

        _mm256_stream_pd(y_ptr, y0);

        x1_ptr += AVX2_DOUBLE_STRIDE;
        x2_ptr += AVX2_DOUBLE_STRIDE;
        y_ptr += AVX2_DOUBLE_STRIDE;
        r -= AVX2_DOUBLE_STRIDE;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_pd(r);

        __m256d x1 = _mm256_maskload_pd(x1_ptr, mask);
        __m256d x2 = _mm256_maskload_pd(x2_ptr, mask);

        __m256d y = _mm256_add_pd(x1, x2);

        _mm256_maskstore_pd(y_ptr, mask, y);
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Elementwise::Add(UInt32 n, Array<double>^ x1, Array<double>^ x2, Array<double>^ y) {
    if (n <= 0) {
        return;
    }

    Util::CheckLength(n, x1, x2, y);

    const double* x1_ptr = (const double*)(x1->Ptr.ToPointer());
    const double* x2_ptr = (const double*)(x2->Ptr.ToPointer());
    double* y_ptr = (double*)(y->Ptr.ToPointer());

    int ret = ew_add_d(n, x1_ptr, x2_ptr, y_ptr);

    Util::AssertReturnCode(ret);
}
