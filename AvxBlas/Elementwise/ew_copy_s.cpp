#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"

using namespace System;

#pragma unmanaged

int ew_copy_s(
    const unsigned int n, 
    const float* __restrict x_ptr, float* __restrict y_ptr) {
    
    if (x_ptr == y_ptr) {
        return SUCCESS;
    }
    
    const unsigned int nb = n & AVX2_FLOAT_BATCH_MASK, nr = n - nb;

    for (unsigned int i = 0; i < nb; i += AVX2_FLOAT_STRIDE) {
        __m256 x = _mm256_load_ps(x_ptr + i);

        _mm256_stream_ps(y_ptr + i, x);
    }
    if (nr > 0) {
        const __m256i mask = _mm256_set_mask(nr);

        __m256 x = _mm256_maskload_ps(x_ptr + nb, mask);

        _mm256_maskstore_ps(y_ptr + nb, mask, x);
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Elementwise::Copy(UInt32 n, Array<float>^ x, Array<float>^ y) {
    if (n <= 0) {
        return;
    }

    Util::CheckLength(n, x, y);

    float* x_ptr = (float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    ew_copy_s(n, x_ptr, y_ptr);
}