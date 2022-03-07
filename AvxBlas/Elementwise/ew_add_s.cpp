#include "../avxblas.h"
#include "../avxblasutil.h"

using namespace System;

#pragma unmanaged

int ew_add_s(
    const unsigned int n,
    const float* __restrict x1_ptr, const float* __restrict x2_ptr, float* __restrict y_ptr) {

    const unsigned int nb = n & AVX2_FLOAT_BATCH_MASK, nr = n - nb;

    for (unsigned int i = 0; i < nb; i += AVX2_FLOAT_STRIDE) {
        __m256 x1 = _mm256_load_ps(x1_ptr + i);
        __m256 x2 = _mm256_load_ps(x2_ptr + i);

        __m256 y = _mm256_add_ps(x1, x2);

        _mm256_stream_ps(y_ptr + i, y);
    }
    if (nr > 0) {
        const __m256i mask = _mm256_set_mask(nr);

        __m256 x1 = _mm256_maskload_ps(x1_ptr + nb, mask);
        __m256 x2 = _mm256_maskload_ps(x2_ptr + nb, mask);

        __m256 y = _mm256_add_ps(x1, x2);

        _mm256_maskstore_ps(y_ptr + nb, mask, y);
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Elementwise::Add(UInt32 n, Array<float>^ x1, Array<float>^ x2, Array<float>^ y) {
    if (n <= 0) {
        return;
    }

    Util::CheckLength(n, x1, x2, y);

    float* x1_ptr = (float*)(x1->Ptr.ToPointer());
    float* x2_ptr = (float*)(x2->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    ew_add_s(n, x1_ptr, x2_ptr, y_ptr);
}
