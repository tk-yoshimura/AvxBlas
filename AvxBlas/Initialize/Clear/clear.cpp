#include "../../AvxBlas.h"
#include "../../AvxBlasUtil.h"

#include <immintrin.h>

using namespace System;

void clear(
    const unsigned int n, const float c, 
    float* __restrict y_ptr) {
    
    const unsigned int nb = n & AVX2_FLOAT_BATCH_MASK, nr = n - nb;

    __m256 fillc = _mm256_set1_ps(c);

    for (unsigned int i = 0; i < nb; i += AVX2_FLOAT_STRIDE) {
        _mm256_stream_ps(y_ptr + i, fillc);
    }
    if (nr > 0) {
        mm256_mask(const __m256i mask, nr);

        _mm256_maskstore_ps(y_ptr + nb, mask, fillc);
    }
}

void clear(
    const unsigned int n, const double c, 
    double* __restrict y_ptr) {
    
    const unsigned int nb = n & AVX2_DOUBLE_BATCH_MASK, nr = n - nb;

    __m256d fillc = _mm256_set1_pd(c);

    for (unsigned int i = 0; i < nb; i += AVX2_DOUBLE_STRIDE) {
        _mm256_stream_pd(y_ptr + i, fillc);
    }
    if (nr > 0) {
        mm256_mask(const __m256i mask, nr * 2);

        _mm256_maskstore_pd(y_ptr + nb, mask, fillc);
    }
}

void AvxBlas::Initialize::Clear(UInt32 n, float c, Array<float>^ y) {
    Util::CheckLength(n, y);

    float* y_ptr = (float*)(y->Ptr.ToPointer());

    clear(n, c, y_ptr);
}

void AvxBlas::Initialize::Clear(UInt32 index, UInt32 n, float c, Array<float>^ y) {
    Util::CheckOutOfRange(index, n, y);

    float* y_ptr = (float*)(y->Ptr.ToPointer());
    y_ptr += index;

    for (UInt32 i = index, thr = (index + AVX2_FLOAT_REMAIN_MASK) & AVX2_FLOAT_BATCH_MASK; n > 0 && i < thr; i++, n--) {
        *y_ptr = c;
        y_ptr++;
    }

    clear(n, c, y_ptr);
}

void AvxBlas::Initialize::Zeroset(UInt32 n, Array<float>^ y) {
    Clear(n, 0.0f, y);
}

void AvxBlas::Initialize::Zeroset(UInt32 index, UInt32 n, Array<float>^ y) {
    Clear(index, n, 0.0f, y);
}

void AvxBlas::Initialize::Clear(UInt32 n, double c, Array<double>^ y) {
    Util::CheckLength(n, y);

    double* y_ptr = (double*)(y->Ptr.ToPointer());

    clear(n, c, y_ptr);
}

void AvxBlas::Initialize::Clear(UInt32 index, UInt32 n, double c, Array<double>^ y) {
    Util::CheckOutOfRange(index, n, y);

    double* y_ptr = (double*)(y->Ptr.ToPointer());
    y_ptr += index;

    for (UInt32 i = index, thr = (index + AVX2_DOUBLE_REMAIN_MASK) & AVX2_DOUBLE_BATCH_MASK; n > 0 && i < thr; i++, n--) {
        *y_ptr = c;
        y_ptr++;
    }

    clear(n, c, y_ptr);
}

void AvxBlas::Initialize::Zeroset(UInt32 n, Array<double>^ y) {
    Clear(n, 0.0, y);
}

void AvxBlas::Initialize::Zeroset(UInt32 index, UInt32 n, Array<double>^ y) {
    Clear(index, n, 0.0, y);
}