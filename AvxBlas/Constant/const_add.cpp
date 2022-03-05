#include "../avxblas.h"
#include "../avxblasutil.h"

using namespace System;

void const_add(
    const unsigned int n, 
    const float* __restrict x_ptr, const float c, float* __restrict y_ptr) {
    
    const unsigned int nb = n & AVX2_FLOAT_BATCH_MASK, nr = n - nb;

    __m256 fillc = _mm256_set1_ps(c);

    for (unsigned int i = 0; i < nb; i += AVX2_FLOAT_STRIDE) {
        __m256 x = _mm256_load_ps(x_ptr + i);

        __m256 y = _mm256_add_ps(x, fillc);

        _mm256_stream_ps(y_ptr + i, y);
    }
    if (nr > 0) {
        const __m256i mask = mm256_mask(nr);

        __m256 x = _mm256_maskload_ps(x_ptr + nb, mask);

        __m256 y = _mm256_add_ps(x, fillc);

        _mm256_maskstore_ps(y_ptr + nb, mask, y);
    }
}

void const_add(
    const unsigned int n, 
    const double* __restrict x_ptr, const double c, double* __restrict y_ptr) {
    
    const unsigned int nb = n & AVX2_DOUBLE_BATCH_MASK, nr = n - nb;

    __m256d fillc = _mm256_set1_pd(c);

    for (unsigned int i = 0; i < nb; i += AVX2_DOUBLE_STRIDE) {
        __m256d x = _mm256_load_pd(x_ptr + i);

        __m256d y = _mm256_add_pd(x, fillc);

        _mm256_stream_pd(y_ptr + i, y);
    }
    if (nr > 0) {
        const __m256i mask = mm256_mask(nr * 2);

        __m256d x = _mm256_maskload_pd(x_ptr + nb, mask);

        __m256d y = _mm256_add_pd(x, fillc);

        _mm256_maskstore_pd(y_ptr + nb, mask, y);
    }
}

void AvxBlas::Constant::Add(UInt32 n, Array<float>^ x, float c, Array<float>^ y) {
    if (n <= 0) {
        return;
    }

    Util::CheckLength(n, x, y);

    float* x_ptr = (float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    const_add(n, x_ptr, c, y_ptr);
}

void AvxBlas::Constant::Add(UInt32 n, Array<double>^ x, double c, Array<double>^ y) {
    if (n <= 0) {
        return;
    }

    Util::CheckLength(n, x, y);

    double* x_ptr = (double*)(x->Ptr.ToPointer());
    double* y_ptr = (double*)(y->Ptr.ToPointer());

    const_add(n, x_ptr, c, y_ptr);
}
