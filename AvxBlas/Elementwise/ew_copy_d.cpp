#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"

using namespace System;

#pragma unmanaged

int ew_copy_d(
    const uint n,
    indoubles x_ptr, outdoubles y_ptr) {

    if (x_ptr == y_ptr) {
        return SUCCESS;
    }

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    uint r = n;

    while (r >= AVX2_DOUBLE_STRIDE * 4) {
        __m256d x0 = _mm256_load_pd(x_ptr);
        __m256d x1 = _mm256_load_pd(x_ptr + AVX2_DOUBLE_STRIDE);
        __m256d x2 = _mm256_load_pd(x_ptr + AVX2_DOUBLE_STRIDE * 2);
        __m256d x3 = _mm256_load_pd(x_ptr + AVX2_DOUBLE_STRIDE * 3);

        _mm256_stream_pd(y_ptr, x0);
        _mm256_stream_pd(y_ptr + AVX2_DOUBLE_STRIDE, x1);
        _mm256_stream_pd(y_ptr + AVX2_DOUBLE_STRIDE * 2, x2);
        _mm256_stream_pd(y_ptr + AVX2_DOUBLE_STRIDE * 3, x3);

        x_ptr += AVX2_DOUBLE_STRIDE * 4;
        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 4;
    }
    if (r >= AVX2_DOUBLE_STRIDE * 2) {
        __m256d x0 = _mm256_load_pd(x_ptr);
        __m256d x1 = _mm256_load_pd(x_ptr + AVX2_DOUBLE_STRIDE);

        _mm256_stream_pd(y_ptr, x0);
        _mm256_stream_pd(y_ptr + AVX2_DOUBLE_STRIDE, x1);

        x_ptr += AVX2_DOUBLE_STRIDE * 2;
        y_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE * 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE) {
        __m256d x0 = _mm256_load_pd(x_ptr);

        _mm256_stream_pd(y_ptr, x0);

        x_ptr += AVX2_DOUBLE_STRIDE;
        y_ptr += AVX2_DOUBLE_STRIDE;
        r -= AVX2_DOUBLE_STRIDE;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_pd(r);

        __m256d x = _mm256_maskload_pd(x_ptr, mask);

        _mm256_maskstore_pd(y_ptr, mask, x);
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Elementwise::Copy(UInt32 n, Array<double>^ x, Array<double>^ y) {
    if (n <= 0) {
        return;
    }

    Util::CheckLength(n, x, y);

    const double* x_ptr = (const double*)(x->Ptr.ToPointer());
    double* y_ptr = (double*)(y->Ptr.ToPointer());

    int ret = ew_copy_d(n, x_ptr, y_ptr);

    Util::AssertReturnCode(ret);
}