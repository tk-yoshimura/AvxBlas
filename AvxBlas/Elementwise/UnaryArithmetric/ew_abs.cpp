#include "../../AvxBlas.h"
#include "../../AvxBlasUtil.h"

using namespace System;

void abs(
    const unsigned int n, 
    const float* __restrict x_ptr, float* __restrict y_ptr) {
    
    const unsigned int nb = n & ~7u, nr = n - nb;

    union {
        float f;
        unsigned __int32 i;
    }m32;
    m32.i = 0x7FFFFFFFu;

    const __m256 bitmask = _mm256_set1_ps(m32.f);

    for (unsigned int i = 0; i < nb; i += 8) {
        __m256 x = _mm256_load_ps(x_ptr + i);

        __m256 y = _mm256_and_ps(bitmask, x);

        _mm256_stream_ps(y_ptr + i, y);
    }
    if (nr > 0) {
        __m256i mask = AvxBlas::masktable_m256(nr);

        __m256 x = _mm256_maskload_ps(x_ptr + nb, mask);

        __m256 y = _mm256_and_ps(bitmask, x);

        _mm256_maskstore_ps(y_ptr + nb, mask, y);
    }
}

void abs(
    const unsigned int n, 
    const double* __restrict x_ptr, double* __restrict y_ptr) {
    
    const unsigned int nb = n & ~3u, nr = n - nb;

    union {
        double f;
        unsigned __int64 i;
    }m32;
    m32.i = 0x7FFFFFFFFFFFFFFFul;

    const __m256d bitmask = _mm256_set1_pd(m32.f);

    for (unsigned int i = 0; i < nb; i += 4) {
        __m256d x = _mm256_load_pd(x_ptr + i);

        __m256d y = _mm256_and_pd(bitmask, x);

        _mm256_stream_pd(y_ptr + i, y);
    }
    if (nr > 0) {
        __m256i mask = AvxBlas::masktable_m256(nr * 2);

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

    abs(n, x_ptr, y_ptr);
}

void AvxBlas::Elementwise::Abs(UInt32 n, Array<double>^ x, Array<double>^ y) {
    if (n <= 0) {
        return;
    }
    
    Util::CheckLength(n, x, y);

    double* x_ptr = (double*)(x->Ptr.ToPointer());
    double* y_ptr = (double*)(y->Ptr.ToPointer());

    abs(n, x_ptr, y_ptr);
}