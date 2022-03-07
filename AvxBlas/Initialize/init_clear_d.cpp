#include "../avxblas.h"
#include "../avxblasutil.h"

using namespace System;

#pragma unmanaged

void clear_d(
    const unsigned int n, const double c, 
    double* __restrict y_ptr) {
    
    const unsigned int nb = n & AVX2_DOUBLE_BATCH_MASK, nr = n - nb;

    __m256d fillc = _mm256_set1_pd(c);

    for (unsigned int i = 0; i < nb; i += AVX2_DOUBLE_STRIDE) {
        _mm256_stream_pd(y_ptr + i, fillc);
    }
    if (nr > 0) {
        const __m256i mask = _mm256_set_mask(nr * 2);

        _mm256_maskstore_pd(y_ptr + nb, mask, fillc);
    }
}

#pragma managed

void AvxBlas::Initialize::Clear(UInt32 n, double c, Array<double>^ y) {
    Util::CheckLength(n, y);

    double* y_ptr = (double*)(y->Ptr.ToPointer());

    clear_d(n, c, y_ptr);
}

void AvxBlas::Initialize::Clear(UInt32 index, UInt32 n, double c, Array<double>^ y) {
    Util::CheckOutOfRange(index, n, y);

    double* y_ptr = (double*)(y->Ptr.ToPointer());
    y_ptr += index;

    for (UInt32 i = index, thr = (index + AVX2_DOUBLE_REMAIN_MASK) & AVX2_DOUBLE_BATCH_MASK; n > 0 && i < thr; i++, n--) {
        *y_ptr = c;
        y_ptr++;
    }

    clear_d(n, c, y_ptr);
}

void AvxBlas::Initialize::Zeroset(UInt32 n, Array<double>^ y) {
    Clear(n, 0.0, y);
}

void AvxBlas::Initialize::Zeroset(UInt32 index, UInt32 n, Array<double>^ y) {
    Clear(index, n, 0.0, y);
}