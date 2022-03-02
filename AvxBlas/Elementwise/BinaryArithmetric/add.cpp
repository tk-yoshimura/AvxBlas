#include "../../AvxBlas.h"

using namespace System;

void add(unsigned int n, const float* __restrict x1_ptr, const float* __restrict x2_ptr, float* __restrict y_ptr) {
    const unsigned int j = n & ~7u, k = n - j;

    for (unsigned int i = 0; i < j; i += 8) {
        __m256 x1 = _mm256_load_ps(x1_ptr + i);
        __m256 x2 = _mm256_load_ps(x2_ptr + i);

        __m256 y = _mm256_add_ps(x1, x2);

        _mm256_stream_ps(y_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = AvxBlas::masktable_m256(k);

        __m256 x1 = _mm256_maskload_ps(x1_ptr + j, mask);
        __m256 x2 = _mm256_maskload_ps(x2_ptr + j, mask);

        __m256 y = _mm256_add_ps(x1, x2);

        _mm256_maskstore_ps(y_ptr + j, mask, y);
    }
}

void add(unsigned int n, const double* __restrict x1_ptr, const double* __restrict x2_ptr, double* __restrict y_ptr) {
    const unsigned int j = n & ~3u, k = n - j;

    for (unsigned int i = 0; i < j; i += 4) {
        __m256d x1 = _mm256_load_pd(x1_ptr + i);
        __m256d x2 = _mm256_load_pd(x2_ptr + i);

        __m256d y = _mm256_add_pd(x1, x2);

        _mm256_stream_pd(y_ptr + i, y);
    }

    if (k > 0) {
        __m256i mask = AvxBlas::masktable_m256(k * 2);

        __m256d x1 = _mm256_maskload_pd(x1_ptr + j, mask);
        __m256d x2 = _mm256_maskload_pd(x2_ptr + j, mask);

        __m256d y = _mm256_add_pd(x1, x2);

        _mm256_maskstore_pd(y_ptr + j, mask, y);
    }
}

void AvxBlas::Elementwise::Add(unsigned int n, Array<float>^ x1, Array<float>^ x2, Array<float>^ y) {
    Util::CheckLength(n, x1, x2, y);

    float* x1_ptr = (float*)(x1->Ptr.ToPointer());
    float* x2_ptr = (float*)(x2->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    add(n, x1_ptr, x2_ptr, y_ptr);
}

void AvxBlas::Elementwise::Add(unsigned int n, Array<double>^ x1, Array<double>^ x2, Array<double>^ y) {
    Util::CheckLength(n, x1, x2, y);

    double* x1_ptr = (double*)(x1->Ptr.ToPointer());
    double* x2_ptr = (double*)(x2->Ptr.ToPointer());
    double* y_ptr = (double*)(y->Ptr.ToPointer());

    add(n, x1_ptr, x2_ptr, y_ptr);
}
