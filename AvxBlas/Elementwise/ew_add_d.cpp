#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"

using namespace System;

#pragma unmanaged

int ew_add_d(
    const unsigned int n,
    const double* __restrict x1_ptr, const double* __restrict x2_ptr, double* __restrict y_ptr) {

    const unsigned int nb = n & AVX2_DOUBLE_BATCH_MASK, nr = n - nb;

    for (unsigned int i = 0; i < nb; i += AVX2_DOUBLE_STRIDE) {
        __m256d x1 = _mm256_load_pd(x1_ptr + i);
        __m256d x2 = _mm256_load_pd(x2_ptr + i);

        __m256d y = _mm256_add_pd(x1, x2);

        _mm256_stream_pd(y_ptr + i, y);
    }
    if (nr > 0) {
        const __m256i mask = _mm256_set_mask(nr * 2);

        __m256d x1 = _mm256_maskload_pd(x1_ptr + nb, mask);
        __m256d x2 = _mm256_maskload_pd(x2_ptr + nb, mask);

        __m256d y = _mm256_add_pd(x1, x2);

        _mm256_maskstore_pd(y_ptr + nb, mask, y);
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
