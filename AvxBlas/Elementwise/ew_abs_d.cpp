#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_ope.cpp"

using namespace System;

#pragma unmanaged

int ew_abs_d(
    unsigned int n, 
    const double* __restrict x_ptr, double* __restrict y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG
    
    while (n >= AVX2_DOUBLE_STRIDE * 4) {
        __m256d x0 = _mm256_load_pd(x_ptr);
        __m256d x1 = _mm256_load_pd(x_ptr + 8);
        __m256d x2 = _mm256_load_pd(x_ptr + 16);
        __m256d x3 = _mm256_load_pd(x_ptr + 24);

        __m256d y0 = _mm256_abs_pd(x0);
        __m256d y1 = _mm256_abs_pd(x1);
        __m256d y2 = _mm256_abs_pd(x2);
        __m256d y3 = _mm256_abs_pd(x3);

        _mm256_stream_pd(y_ptr, y0);
        _mm256_stream_pd(y_ptr + 8, y1);
        _mm256_stream_pd(y_ptr + 16, y2);
        _mm256_stream_pd(y_ptr + 24, y3);

        x_ptr += AVX2_DOUBLE_STRIDE * 4;
        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        n -= AVX2_DOUBLE_STRIDE * 4;
    }
    if (n >= AVX2_DOUBLE_STRIDE * 2) {
        __m256d x0 = _mm256_load_pd(x_ptr);
        __m256d x1 = _mm256_load_pd(x_ptr + 8);

        __m256d y0 = _mm256_abs_pd(x0);
        __m256d y1 = _mm256_abs_pd(x1);

        _mm256_stream_pd(y_ptr, y0);
        _mm256_stream_pd(y_ptr + 8, y1);

        x_ptr += AVX2_DOUBLE_STRIDE * 2;
        y_ptr += AVX2_DOUBLE_STRIDE * 2;
        n -= AVX2_DOUBLE_STRIDE * 2;
    }
    if (n >= AVX2_DOUBLE_STRIDE) {
        __m256d x0 = _mm256_load_pd(x_ptr);

        __m256d y0 = _mm256_abs_pd(x0);

        _mm256_stream_pd(y_ptr, y0);

        x_ptr += AVX2_DOUBLE_STRIDE;
        y_ptr += AVX2_DOUBLE_STRIDE;
        n -= AVX2_DOUBLE_STRIDE;
    }
    if (n > 0) {
        const __m256i mask = _mm256_set_mask(n * 2);

        __m256d x = _mm256_maskload_pd(x_ptr, mask);

        __m256d y = _mm256_abs_pd(x);

        _mm256_maskstore_pd(y_ptr, mask, y);
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Elementwise::Abs(UInt32 n, Array<double>^ x, Array<double>^ y) {
    if (n <= 0) {
        return;
    }
    
    Util::CheckLength(n, x, y);

    double* x_ptr = (double*)(x->Ptr.ToPointer());
    double* y_ptr = (double*)(y->Ptr.ToPointer());

    ew_abs_d(n, x_ptr, y_ptr);
}