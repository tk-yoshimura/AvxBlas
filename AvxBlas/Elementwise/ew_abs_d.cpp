#include "../avxblas.h"
#include "../avxblasutil.h"

using namespace System;

void ew_abs_d(
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

void AvxBlas::Elementwise::Abs(UInt32 n, Array<double>^ x, Array<double>^ y) {
    if (n <= 0) {
        return;
    }
    
    Util::CheckLength(n, x, y);

    double* x_ptr = (double*)(x->Ptr.ToPointer());
    double* y_ptr = (double*)(y->Ptr.ToPointer());

    ew_abs_d(n, x_ptr, y_ptr);
}