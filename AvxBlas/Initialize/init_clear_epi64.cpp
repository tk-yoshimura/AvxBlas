#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_loadstore_xn_epi64.hpp"

using namespace System;

#pragma unmanaged

int clear_epi64(
    const uint index, const uint n, const long c,
    outulongs y_ptr) {

    uint r = n;

    y_ptr += index;
    while (r > 0) {
        if (((size_t)y_ptr % AVX2_ALIGNMENT) == 0) {
            break;
        }

        *y_ptr = (ulong)c;
        y_ptr++;
        r--;
    }

    uint hi = (uint)(((ulong)c) >> 32), lo = (uint)c;

    const __m256i fillc = _mm256_setr_epi32(hi, lo, hi, lo, hi, lo, hi, lo);

    while (r >= AVX2_EPI64_STRIDE * 4) {
        _mm256_stream_x4_epi64(y_ptr, fillc, fillc, fillc, fillc);

        y_ptr += AVX2_EPI64_STRIDE * 4;
        r -= AVX2_EPI64_STRIDE * 4;
    }
    if (r >= AVX2_EPI64_STRIDE * 2) {
        _mm256_stream_x2_epi64(y_ptr, fillc, fillc);

        y_ptr += AVX2_EPI64_STRIDE * 2;
        r -= AVX2_EPI64_STRIDE * 2;
    }
    if (r >= AVX2_EPI64_STRIDE) {
        _mm256_stream_x1_epi64(y_ptr, fillc);

        y_ptr += AVX2_EPI64_STRIDE;
        r -= AVX2_EPI64_STRIDE;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_pd(r);

        _mm256_maskstore_epi64(y_ptr, mask, fillc);
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Initialize::Clear(UInt32 n, Int64 c, Array<Int64>^ y) {
    Util::CheckLength(n, y);

    ulong* y_ptr = (ulong*)(y->Ptr.ToPointer());

    int ret = clear_epi64(0, n, (long)c, y_ptr);

    Util::AssertReturnCode(ret);
}

void AvxBlas::Initialize::Clear(UInt32 index, UInt32 n, Int64 c, Array<Int64>^ y) {
    Util::CheckOutOfRange(index, n, y);

    ulong* y_ptr = (ulong*)(y->Ptr.ToPointer());

    int ret = clear_epi64(index, n, (long)c, y_ptr);

    Util::AssertReturnCode(ret);
}

void AvxBlas::Initialize::Zeroset(UInt32 n, Array<Int64>^ y) {
    Clear(n, 0L, y);
}

void AvxBlas::Initialize::Zeroset(UInt32 index, UInt32 n, Array<Int64>^ y) {
    Clear(index, n, 0L, y);
}