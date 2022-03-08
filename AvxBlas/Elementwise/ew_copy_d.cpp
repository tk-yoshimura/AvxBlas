#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"

using namespace System;

#pragma unmanaged

int ew_copy_d(
    const unsigned int n, 
    const double* __restrict x_ptr, double* __restrict y_ptr) {

    if (x_ptr == y_ptr) {
        return SUCCESS;
    }
    
    const unsigned int nb = n & AVX2_DOUBLE_BATCH_MASK, nr = n - nb;

    for (unsigned int i = 0; i < nb; i += AVX2_DOUBLE_STRIDE) {
        __m256d x = _mm256_load_pd(x_ptr + i);

        _mm256_stream_pd(y_ptr + i, x);
    }
    if (nr > 0) {
        const __m256i mask = _mm256_set_mask(nr * 2);

        __m256d x = _mm256_maskload_pd(x_ptr + nb, mask);

        _mm256_maskstore_pd(y_ptr + nb, mask, x);
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Elementwise::Copy(UInt32 n, Array<double>^ x, Array<double>^ y) {
    if (n <= 0) {
        return;
    }
    
    Util::CheckLength(n, x, y);

    double* x_ptr = (double*)(x->Ptr.ToPointer());
    double* y_ptr = (double*)(y->Ptr.ToPointer());

    ew_copy_d(n, x_ptr, y_ptr);
}