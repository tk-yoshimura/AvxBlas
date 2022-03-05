#include "../avxblas.h"
#include "../avxblasutil.h"

using namespace System;

void ew_abs(
    const unsigned int n, 
    const float* __restrict x_ptr, float* __restrict y_ptr) {
    
    const unsigned int nb = n & AVX2_FLOAT_BATCH_MASK, nr = n - nb;

    union {
        float f;
        unsigned __int32 i;
    }m32;
    m32.i = 0x7FFFFFFFu;

    const __m256 bitmask = _mm256_set1_ps(m32.f);

    for (unsigned int i = 0; i < nb; i += AVX2_FLOAT_STRIDE) {
        __m256 x = _mm256_load_ps(x_ptr + i);

        __m256 y = _mm256_and_ps(bitmask, x);

        _mm256_stream_ps(y_ptr + i, y);
    }
    if (nr > 0) {
        const __m256i mask = mm256_mask(nr);

        __m256 x = _mm256_maskload_ps(x_ptr + nb, mask);

        __m256 y = _mm256_and_ps(bitmask, x);

        _mm256_maskstore_ps(y_ptr + nb, mask, y);
    }
}

void ew_abs(
    const unsigned int n, 
    const double* __restrict x_ptr, double* __restrict y_ptr) {
    
    const unsigned int nb = n & AVX2_DOUBLE_BATCH_MASK, nr = n - nb;

    union {
        double f;
        unsigned __int64 i;
    }m32;
    m32.i = 0x7FFFFFFFFFFFFFFFul;

    const __m256d bitmask = _mm256_set1_pd(m32.f);

    for (unsigned int i = 0; i < nb; i += AVX2_DOUBLE_STRIDE) {
        __m256d x = _mm256_load_pd(x_ptr + i);

        __m256d y = _mm256_and_pd(bitmask, x);

        _mm256_stream_pd(y_ptr + i, y);
    }
    if (nr > 0) {
        const __m256i mask = mm256_mask(nr * 2);

        __m256d x = _mm256_maskload_pd(x_ptr + nb, mask);

        __m256d y = _mm256_and_pd(bitmask, x);

        _mm256_maskstore_pd(y_ptr + nb, mask, y);
    }
}

void AvxBlas::Elementwise::Abs(UInt32 n, Array<float>^ x, Array<float>^ y) {
    if (n <= 0) {
        return;
    }

    Util::CheckLength(n, x, y);

    float* x_ptr = (float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    ew_abs(n, x_ptr, y_ptr);
}

void AvxBlas::Elementwise::Abs(UInt32 n, Array<double>^ x, Array<double>^ y) {
    if (n <= 0) {
        return;
    }
    
    Util::CheckLength(n, x, y);

    double* x_ptr = (double*)(x->Ptr.ToPointer());
    double* y_ptr = (double*)(y->Ptr.ToPointer());

    ew_abs(n, x_ptr, y_ptr);
}