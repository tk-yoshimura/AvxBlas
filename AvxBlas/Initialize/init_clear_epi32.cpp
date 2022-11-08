#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_loadstore_xn_epi32.hpp"

using namespace System;

#pragma unmanaged

int clear_epi32(
    const uint index, const uint n, const int c,
    outuints y_ptr) {

    uint r = n;

    y_ptr += index;
    while (r > 0) {
        if (((size_t)y_ptr % AVX2_ALIGNMENT) == 0) {
            break;
        }

        *y_ptr = (uint)c;
        y_ptr++;
        r--;
    }

    const __m256i fillc = _mm256_set1_epi32(c);

    while (r >= AVX2_EPI32_STRIDE * 4) {
        _mm256_stream_x4_epi32(y_ptr, fillc, fillc, fillc, fillc);

        y_ptr += AVX2_EPI32_STRIDE * 4;
        r -= AVX2_EPI32_STRIDE * 4;
    }
    if (r >= AVX2_EPI32_STRIDE * 2) {
        _mm256_stream_x2_epi32(y_ptr, fillc, fillc);

        y_ptr += AVX2_EPI32_STRIDE * 2;
        r -= AVX2_EPI32_STRIDE * 2;
    }
    if (r >= AVX2_EPI32_STRIDE) {
        _mm256_stream_x1_epi32(y_ptr, fillc);

        y_ptr += AVX2_EPI32_STRIDE;
        r -= AVX2_EPI32_STRIDE;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps(r);

        _mm256_maskstore_epi32(y_ptr, mask, fillc);
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Initialize::Clear(UInt32 n, Int32 c, Array<Int32>^ y) {
    Util::CheckLength(n, y);

    uint* y_ptr = (uint*)(y->Ptr.ToPointer());

    int ret = clear_epi32(0, n, c, y_ptr);

    Util::AssertReturnCode(ret);
}

void AvxBlas::Initialize::Clear(UInt32 index, UInt32 n, Int32 c, Array<Int32>^ y) {
    Util::CheckOutOfRange(index, n, y);

    uint* y_ptr = (uint*)(y->Ptr.ToPointer());

    int ret = clear_epi32(index, n, c, y_ptr);

    Util::AssertReturnCode(ret);
}

void AvxBlas::Initialize::Zeroset(UInt32 n, Array<Int32>^ y) {
    Clear(n, 0, y);
}

void AvxBlas::Initialize::Zeroset(UInt32 index, UInt32 n, Array<Int32>^ y) {
    Clear(index, n, 0, y);
}