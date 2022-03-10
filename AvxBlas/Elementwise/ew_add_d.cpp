#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"

using namespace System;

#pragma unmanaged

int ew_add_d(
    unsigned int n,
    const double* __restrict x1_ptr, const double* __restrict x2_ptr, double* __restrict y_ptr) {

#ifdef _DEBUG
    if (((size_t)x1_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)x2_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    while (n >= AVX2_DOUBLE_STRIDE * 4) {
        __m256d x01 = _mm256_load_pd(x1_ptr);
        __m256d x02 = _mm256_load_pd(x2_ptr);
        __m256d x11 = _mm256_load_pd(x1_ptr + 8);
        __m256d x12 = _mm256_load_pd(x2_ptr + 8);
        __m256d x21 = _mm256_load_pd(x1_ptr + 16);
        __m256d x22 = _mm256_load_pd(x2_ptr + 16);
        __m256d x31 = _mm256_load_pd(x1_ptr + 24);
        __m256d x32 = _mm256_load_pd(x2_ptr + 24);

        __m256d y0 = _mm256_add_pd(x01, x02);
        __m256d y1 = _mm256_add_pd(x11, x12);
        __m256d y2 = _mm256_add_pd(x21, x22);
        __m256d y3 = _mm256_add_pd(x31, x32);

        _mm256_stream_pd(y_ptr, y0);
        _mm256_stream_pd(y_ptr + 8, y1);
        _mm256_stream_pd(y_ptr + 16, y2);
        _mm256_stream_pd(y_ptr + 24, y3);

        x1_ptr += AVX2_DOUBLE_STRIDE * 4;
        x2_ptr += AVX2_DOUBLE_STRIDE * 4;
        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        n -= AVX2_DOUBLE_STRIDE * 4;
    }
    if (n >= AVX2_DOUBLE_STRIDE * 2) {
        __m256d x01 = _mm256_load_pd(x1_ptr);
        __m256d x02 = _mm256_load_pd(x2_ptr);
        __m256d x11 = _mm256_load_pd(x1_ptr + 8);
        __m256d x12 = _mm256_load_pd(x2_ptr + 8);

        __m256d y0 = _mm256_add_pd(x01, x02);
        __m256d y1 = _mm256_add_pd(x11, x12);

        _mm256_stream_pd(y_ptr, y0);
        _mm256_stream_pd(y_ptr + 8, y1);

        x1_ptr += AVX2_DOUBLE_STRIDE * 2;
        x2_ptr += AVX2_DOUBLE_STRIDE * 2;
        y_ptr += AVX2_DOUBLE_STRIDE * 2;
        n -= AVX2_DOUBLE_STRIDE * 2;
    }
    if (n >= AVX2_DOUBLE_STRIDE) {
        __m256d x01 = _mm256_load_pd(x1_ptr);
        __m256d x02 = _mm256_load_pd(x2_ptr);

        __m256d y0 = _mm256_add_pd(x01, x02);

        _mm256_stream_pd(y_ptr, y0);

        x1_ptr += AVX2_DOUBLE_STRIDE;
        x2_ptr += AVX2_DOUBLE_STRIDE;
        y_ptr += AVX2_DOUBLE_STRIDE;
        n -= AVX2_DOUBLE_STRIDE;
    }
    if (n > 0) {
        const __m256i mask = _mm256_set_mask(n * 2);

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

    double* x1_ptr = (double*)(x1->Ptr.ToPointer());
    double* x2_ptr = (double*)(x2->Ptr.ToPointer());
    double* y_ptr = (double*)(y->Ptr.ToPointer());

    ew_add_d(n, x1_ptr, x2_ptr, y_ptr);
}
