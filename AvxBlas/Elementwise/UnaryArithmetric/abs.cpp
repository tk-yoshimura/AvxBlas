#include "../../AvxBlas.h"

using namespace System;

void abs(unsigned int length, const float* __restrict x_ptr, float* __restrict y_ptr) {
    const unsigned int j = length & ~7u, k = length - j;

    union {
        float f;
        unsigned __int32 i;
    }m32;
    m32.i = 0x7FFFFFFFu;

    const __m256 bitmask = _mm256_set1_ps(m32.f);

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x = _mm256_load_ps(x_ptr + i);

        __m256 y = _mm256_and_ps(bitmask, x);

        _mm256_stream_ps(y_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = AvxBlas::masktable_m256(k);

        __m256 x = _mm256_maskload_ps(x_ptr + j, mask);

        __m256 y = _mm256_and_ps(bitmask, x);

        _mm256_maskstore_ps(y_ptr + j, mask, y);
    }
}

void abs(unsigned int length, const double* __restrict x_ptr, double* __restrict y_ptr) {
    const unsigned int j = length & ~3u, k = length - j;

    union {
        double f;
        unsigned __int64 i;
    }m32;
    m32.i = 0x7FFFFFFFFFFFFFFFul;

    const __m256d bitmask = _mm256_set1_pd(m32.f);

    for (unsigned int i = 0; i < j; i += 4) {
        __m256d x = _mm256_load_pd(x_ptr + i);

        __m256d y = _mm256_and_pd(bitmask, x);

        _mm256_stream_pd(y_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = AvxBlas::masktable_m256(k * 2);

        __m256d x = _mm256_maskload_pd(x_ptr + j, mask);

        __m256d y = _mm256_and_pd(bitmask, x);

        _mm256_maskstore_pd(y_ptr + j, mask, y);
    }
}

void AvxBlas::Elementwise::Abs(unsigned int length, Array<float>^ x, Array<float>^ y) {    
    Util::CheckLength(length, x, y);

    float* x_ptr = (float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    abs(length, x_ptr, y_ptr);
}

void AvxBlas::Elementwise::Abs(unsigned int length, Array<double>^ x, Array<double>^ y) {
    Util::CheckLength(length, x, y);

    double* x_ptr = (double*)(x->Ptr.ToPointer());
    double* y_ptr = (double*)(y->Ptr.ToPointer());

    abs(length, x_ptr, y_ptr);
}