#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_loadstore_xn_s.hpp"

using namespace System;

#pragma unmanaged

int const_add_s(
    const uint n,
    infloats x_ptr, const float c, outfloats y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    uint r = n;

    const __m256 fillc = _mm256_set1_ps(c);

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        __m256 x0, x1, x2, x3;
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);

        __m256 y0 = _mm256_add_ps(x0, fillc);
        __m256 y1 = _mm256_add_ps(x1, fillc);
        __m256 y2 = _mm256_add_ps(x2, fillc);
        __m256 y3 = _mm256_add_ps(x3, fillc);

        _mm256_stream_x4_ps(y_ptr, x0, x1, x2, x3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 2) {
        __m256 x0, x1;
        _mm256_load_x2_ps(x_ptr, x0, x1);

        __m256 y0 = _mm256_add_ps(x0, fillc);
        __m256 y1 = _mm256_add_ps(x1, fillc);

        _mm256_stream_x2_ps(y_ptr, x0, x1);

        x_ptr += AVX2_FLOAT_STRIDE * 2;
        y_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE) {
        __m256 x0;
        _mm256_load_x1_ps(x_ptr, x0);

        __m256 y0 = _mm256_add_ps(x0, fillc);

        _mm256_stream_x1_ps(y_ptr, x0);

        x_ptr += AVX2_FLOAT_STRIDE;
        y_ptr += AVX2_FLOAT_STRIDE;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps(r);

        __m256 x = _mm256_maskload_ps(x_ptr, mask);

        __m256 y = _mm256_add_ps(x, fillc);

        _mm256_maskstore_ps(y_ptr, mask, y);
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Constant::Add(UInt32 n, Array<float>^ x, float c, Array<float>^ y) {
    if (n <= 0) {
        return;
    }

    Util::CheckLength(n, x, y);

    const float* x_ptr = (const float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    int ret = const_add_s(n, x_ptr, c, y_ptr);

    Util::AssertReturnCode(ret);
}