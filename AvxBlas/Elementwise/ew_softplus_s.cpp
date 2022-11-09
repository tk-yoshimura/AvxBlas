#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_ope_s.hpp"
#include "../Inline/inline_loadstore_xn_s.hpp"

using namespace System;

#pragma unmanaged

__forceinline __m256 _mm256_softplus_ps(__m256 x) {
    const __m256 ones = _mm256_set1_ps(1);

    __m256 y = _mm256_log_ps(_mm256_add_ps(ones, _mm256_exp_ps(x)));

    return y;
}

int ew_softplus_s(
    const uint n,
    infloats x_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x0, x1, x2, x3;
    __m256 y0, y1, y2, y3;

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);

        y0 = _mm256_softplus_ps(x0);
        y1 = _mm256_softplus_ps(x1);
        y2 = _mm256_softplus_ps(x2);
        y3 = _mm256_softplus_ps(x3);

        _mm256_stream_x4_ps(y_ptr, y0, y1, y2, y3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_load_x2_ps(x_ptr, x0, x1);

        y0 = _mm256_softplus_ps(x0);
        y1 = _mm256_softplus_ps(x1);

        _mm256_stream_x2_ps(y_ptr, y0, y1);

        x_ptr += AVX2_FLOAT_STRIDE * 2;
        y_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_load_x1_ps(x_ptr, x0);

        y0 = _mm256_softplus_ps(x0);

        _mm256_stream_x1_ps(y_ptr, y0);

        x_ptr += AVX2_FLOAT_STRIDE;
        y_ptr += AVX2_FLOAT_STRIDE;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps(r);

        x0 = _mm256_maskload_ps(x_ptr, mask);

        y0 = _mm256_softplus_ps(x0);

        _mm256_maskstore_ps(y_ptr, mask, y0);
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Elementwise::Softplus(UInt32 n, Array<float>^ x, Array<float>^ y) {
    if (n <= 0) {
        return;
    }

    Util::CheckLength(n, x, y);

    const float* x_ptr = (const float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    int ret = ew_softplus_s(n, x_ptr, y_ptr);

    Util::AssertReturnCode(ret);
}