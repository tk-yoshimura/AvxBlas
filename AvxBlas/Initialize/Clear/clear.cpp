#include "../../AvxBlas.h"

using namespace System;

void clear(unsigned int n, float c, float* __restrict y_ptr) {
    const unsigned int j = n & ~7u, k = n - j;

    __m256 fillc = _mm256_set1_ps(c);

    for (unsigned int i = 0; i < j; i += 8) {
        _mm256_stream_ps(y_ptr + i, fillc);
    }

    if (k > 0) {
        __m256i mask = AvxBlas::masktable_m256(k);

        _mm256_maskstore_ps(y_ptr + j, mask, fillc);
    }
}

void AvxBlas::Initialize::Clear(unsigned int n, float c, Array<float>^ y) {
    Util::CheckLength(n, y);

    float* y_ptr = (float*)(y->Ptr.ToPointer());

    clear(n, c, y_ptr);
}

void AvxBlas::Initialize::Zeroset(unsigned int n, Array<float>^ y) {
    Clear(n, 0, y);
}