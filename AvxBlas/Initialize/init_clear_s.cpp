#include "../avxblas.h"
#include "../avxblasutil.h"

using namespace System;

#pragma unmanaged

void clear_s(
    const unsigned int n, const float c, 
    float* __restrict y_ptr) {
    
    const unsigned int nb = n & AVX2_FLOAT_BATCH_MASK, nr = n - nb;

    __m256 fillc = _mm256_set1_ps(c);

    for (unsigned int i = 0; i < nb; i += AVX2_FLOAT_STRIDE) {
        _mm256_stream_ps(y_ptr + i, fillc);
    }
    if (nr > 0) {
        const __m256i mask = _mm256_set_mask(nr);

        _mm256_maskstore_ps(y_ptr + nb, mask, fillc);
    }
}

#pragma managed

void AvxBlas::Initialize::Clear(UInt32 n, float c, Array<float>^ y) {
    Util::CheckLength(n, y);

    float* y_ptr = (float*)(y->Ptr.ToPointer());

    clear_s(n, c, y_ptr);
}

void AvxBlas::Initialize::Clear(UInt32 index, UInt32 n, float c, Array<float>^ y) {
    Util::CheckOutOfRange(index, n, y);

    float* y_ptr = (float*)(y->Ptr.ToPointer());
    y_ptr += index;

    for (UInt32 i = index, thr = (index + AVX2_FLOAT_REMAIN_MASK) & AVX2_FLOAT_BATCH_MASK; n > 0 && i < thr; i++, n--) {
        *y_ptr = c;
        y_ptr++;
    }

    clear_s(n, c, y_ptr);
}

void AvxBlas::Initialize::Zeroset(UInt32 n, Array<float>^ y) {
    Clear(n, 0.0f, y);
}

void AvxBlas::Initialize::Zeroset(UInt32 index, UInt32 n, Array<float>^ y) {
    Clear(index, n, 0.0f, y);
}