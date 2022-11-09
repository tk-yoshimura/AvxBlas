#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_loadstore_xn_s.hpp"

using namespace System;

#pragma unmanaged

int ew_minimum_s(
    const uint n,
    infloats x1_ptr, infloats x2_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (((size_t)x1_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)x2_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x01, x11, x21, x31, x02, x12, x22, x32;
    __m256 y0, y1, y2, y3;

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_load_x4_ps(x1_ptr, x01, x11, x21, x31);
        _mm256_load_x4_ps(x2_ptr, x02, x12, x22, x32);

        y0 = _mm256_min_ps(x01, x02);
        y1 = _mm256_min_ps(x11, x12);
        y2 = _mm256_min_ps(x21, x22);
        y3 = _mm256_min_ps(x31, x32);

        _mm256_stream_x4_ps(y_ptr, y0, y1, y2, y3);

        x1_ptr += AVX2_FLOAT_STRIDE * 4;
        x2_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_load_x2_ps(x1_ptr, x01, x11);
        _mm256_load_x2_ps(x2_ptr, x02, x12);

        y0 = _mm256_min_ps(x01, x02);
        y1 = _mm256_min_ps(x11, x12);

        _mm256_stream_x2_ps(y_ptr, y0, y1);

        x1_ptr += AVX2_FLOAT_STRIDE * 2;
        x2_ptr += AVX2_FLOAT_STRIDE * 2;
        y_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_load_x1_ps(x1_ptr, x01);
        _mm256_load_x1_ps(x2_ptr, x02);

        y0 = _mm256_min_ps(x01, x02);

        _mm256_stream_x1_ps(y_ptr, y0);

        x1_ptr += AVX2_FLOAT_STRIDE;
        x2_ptr += AVX2_FLOAT_STRIDE;
        y_ptr += AVX2_FLOAT_STRIDE;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps(r);

        x01 = _mm256_maskload_ps(x1_ptr, mask);
        x02 = _mm256_maskload_ps(x2_ptr, mask);

        y0 = _mm256_min_ps(x01, x02);

        _mm256_maskstore_ps(y_ptr, mask, y0);
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Elementwise::Minimum(UInt32 n, Array<float>^ x1, Array<float>^ x2, Array<float>^ y) {
    if (n <= 0) {
        return;
    }

    Util::CheckLength(n, x1, x2, y);

    const float* x1_ptr = (const float*)(x1->Ptr.ToPointer());
    const float* x2_ptr = (const float*)(x2->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    int ret = ew_minimum_s(n, x1_ptr, x2_ptr, y_ptr);

    Util::AssertReturnCode(ret);
}
