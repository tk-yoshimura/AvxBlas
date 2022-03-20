#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"

using namespace System;

#pragma unmanaged

int clear_s(
    const uint index, const uint n, const float c,
    outfloats y_ptr) {

    uint r = n;

    y_ptr += index;
    while(r > 0) {
        if (((size_t)y_ptr % AVX2_ALIGNMENT) == 0) {
            break;
        }

        *y_ptr = c;
        y_ptr++;
        r--;
    }
    
    const __m256 fillc = _mm256_set1_ps(c);

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_stream_ps(y_ptr, fillc);
        _mm256_stream_ps(y_ptr + AVX2_FLOAT_STRIDE, fillc);
        _mm256_stream_ps(y_ptr + AVX2_FLOAT_STRIDE * 2, fillc);
        _mm256_stream_ps(y_ptr + AVX2_FLOAT_STRIDE * 3, fillc);

        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_stream_ps(y_ptr, fillc);
        _mm256_stream_ps(y_ptr + AVX2_FLOAT_STRIDE, fillc);

        y_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_stream_ps(y_ptr, fillc);

        y_ptr += AVX2_FLOAT_STRIDE;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps(r);

        _mm256_maskstore_ps(y_ptr, mask, fillc);
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Initialize::Clear(UInt32 n, float c, Array<float>^ y) {
    Util::CheckLength(n, y);

    float* y_ptr = (float*)(y->Ptr.ToPointer());

    int ret = clear_s(0, n, c, y_ptr);

    Util::AssertReturnCode(ret);
}

void AvxBlas::Initialize::Clear(UInt32 index, UInt32 n, float c, Array<float>^ y) {
    Util::CheckOutOfRange(index, n, y);

    float* y_ptr = (float*)(y->Ptr.ToPointer());
    
    int ret = clear_s(index, n, c, y_ptr);

    Util::AssertReturnCode(ret);
}

void AvxBlas::Initialize::Zeroset(UInt32 n, Array<float>^ y) {
    Clear(n, 0.0f, y);
}

void AvxBlas::Initialize::Zeroset(UInt32 index, UInt32 n, Array<float>^ y) {
    Clear(index, n, 0.0f, y);
}