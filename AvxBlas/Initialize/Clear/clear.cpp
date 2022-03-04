#include "../../AvxBlas.h"
#include "../../AvxBlasUtil.h"

using namespace System;

void clear(unsigned int n, float c, float* __restrict y_ptr) {
    const unsigned int j = n & AVX2_FLOAT_BATCH_MASK, k = n - j;

    __m256 fillc = _mm256_set1_ps(c);

    for (unsigned int i = 0; i < j; i += AVX2_FLOAT_STRIDE) {
        _mm256_stream_ps(y_ptr + i, fillc);
    }

    if (k > 0) {
        __m256i mask = AvxBlas::masktable_m256(k);

        _mm256_maskstore_ps(y_ptr + j, mask, fillc);
    }
}

void clear(unsigned int n, double c, double* __restrict y_ptr) {
    const unsigned int j = n & AVX2_DOUBLE_BATCH_MASK, k = n - j;

    __m256d fillc = _mm256_set1_pd(c);

    for (unsigned int i = 0; i < j; i += AVX2_DOUBLE_STRIDE) {
        _mm256_stream_pd(y_ptr + i, fillc);
    }

    if (k > 0) {
        __m256i mask = AvxBlas::masktable_m256(k * 2);

        _mm256_maskstore_pd(y_ptr + j, mask, fillc);
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

    for (UInt32 i = index, thr = (index + 7) / 8 * 8; n > 0 && i < thr; i++, n--) {
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

    for (UInt32 i = index, thr = (index + 3) / 4 * 4; n > 0 && i < thr; i++, n--) {
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