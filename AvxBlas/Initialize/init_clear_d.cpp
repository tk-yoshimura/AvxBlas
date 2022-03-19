#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"

using namespace System;

#pragma unmanaged

int clear_d(
    const unsigned int index, const unsigned int n, const double c,
    double* __restrict y_ptr) {
    
    unsigned int r = n;

    y_ptr += index;
    while (r > 0) {
        if (((size_t)y_ptr % AVX2_ALIGNMENT) == 0) {
            break;
        }

        *y_ptr = c;
        y_ptr++;
        r--;
    }

    const __m256d fillc = _mm256_set1_pd(c);

    while (r >= AVX2_DOUBLE_STRIDE * 4) {
        _mm256_stream_pd(y_ptr, fillc);
        _mm256_stream_pd(y_ptr + AVX2_DOUBLE_STRIDE, fillc);
        _mm256_stream_pd(y_ptr + AVX2_DOUBLE_STRIDE * 2, fillc);
        _mm256_stream_pd(y_ptr + AVX2_DOUBLE_STRIDE * 3, fillc);

        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 4;
    }
    if (r >= AVX2_DOUBLE_STRIDE * 2) {
        _mm256_stream_pd(y_ptr, fillc);
        _mm256_stream_pd(y_ptr + AVX2_DOUBLE_STRIDE, fillc);

        y_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE * 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE) {
        _mm256_stream_pd(y_ptr, fillc);

        y_ptr += AVX2_DOUBLE_STRIDE;
        r -= AVX2_DOUBLE_STRIDE;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_pd(r);

        _mm256_maskstore_pd(y_ptr, mask, fillc);
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Initialize::Clear(UInt32 n, double c, Array<double>^ y) {
    Util::CheckLength(n, y);

    double* y_ptr = (double*)(y->Ptr.ToPointer());

    int ret = clear_d(0, n, c, y_ptr);

    Util::AssertReturnCode(ret);
}

void AvxBlas::Initialize::Clear(UInt32 index, UInt32 n, double c, Array<double>^ y) {
    Util::CheckOutOfRange(index, n, y);

    double* y_ptr = (double*)(y->Ptr.ToPointer());
    
    int ret = clear_d(index, n, c, y_ptr);

    Util::AssertReturnCode(ret);
}

void AvxBlas::Initialize::Zeroset(UInt32 n, Array<double>^ y) {
    Clear(n, 0.0, y);
}

void AvxBlas::Initialize::Zeroset(UInt32 index, UInt32 n, Array<double>^ y) {
    Clear(index, n, 0.0, y);
}