#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_ope.cpp"

using namespace System;

#pragma unmanaged

int ew_abs_s(
    const unsigned int n, 
    const float* __restrict x_ptr, float* __restrict y_ptr) {
    
    const unsigned int nb = n & AVX2_FLOAT_BATCH_MASK, nr = n - nb;

    for (unsigned int i = 0; i < nb; i += AVX2_FLOAT_STRIDE) {
        __m256 x = _mm256_load_ps(x_ptr + i);

        __m256 y = _mm256_abs_ps(x);

        _mm256_stream_ps(y_ptr + i, y);
    }
    if (nr > 0) {
        const __m256i mask = _mm256_set_mask(nr);

        __m256 x = _mm256_maskload_ps(x_ptr + nb, mask);

        __m256 y = _mm256_abs_ps(x);

        _mm256_maskstore_ps(y_ptr + nb, mask, y);
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Elementwise::Abs(UInt32 n, Array<float>^ x, Array<float>^ y) {
    if (n <= 0) {
        return;
    }

    Util::CheckLength(n, x, y);

    float* x_ptr = (float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    ew_abs_s(n, x_ptr, y_ptr);
}