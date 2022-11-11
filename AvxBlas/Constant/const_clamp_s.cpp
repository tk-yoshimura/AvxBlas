#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_loadstore_xn_s.hpp"

using namespace System;

#pragma unmanaged

int const_clamp_s(
    const uint n,
    infloats x_ptr, const float cmin, const float cmax, outfloats y_ptr) {

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    const __m256 fillcmin = _mm256_set1_ps(cmin);
    const __m256 fillcmax = _mm256_set1_ps(cmax);

    __m256 x0, x1, x2, x3;
    __m256 y0, y1, y2, y3;

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);

        y0 = _mm256_min_ps(_mm256_max_ps(x0, fillcmin), fillcmax);
        y1 = _mm256_min_ps(_mm256_max_ps(x1, fillcmin), fillcmax);
        y2 = _mm256_min_ps(_mm256_max_ps(x2, fillcmin), fillcmax);
        y3 = _mm256_min_ps(_mm256_max_ps(x3, fillcmin), fillcmax);

        _mm256_stream_x4_ps(y_ptr, y0, y1, y2, y3);

        x_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_load_x2_ps(x_ptr, x0, x1);

        y0 = _mm256_min_ps(_mm256_max_ps(x0, fillcmin), fillcmax);
        y1 = _mm256_min_ps(_mm256_max_ps(x1, fillcmin), fillcmax);

        _mm256_stream_x2_ps(y_ptr, y0, y1);

        x_ptr += AVX2_FLOAT_STRIDE * 2;
        y_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_load_x1_ps(x_ptr, x0);

        y0 = _mm256_min_ps(_mm256_max_ps(x0, fillcmin), fillcmax);

        _mm256_stream_x1_ps(y_ptr, y0);

        x_ptr += AVX2_FLOAT_STRIDE;
        y_ptr += AVX2_FLOAT_STRIDE;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps(r);

        x0 = _mm256_maskload_ps(x_ptr, mask);

        y0 = _mm256_min_ps(_mm256_max_ps(x0, fillcmin), fillcmax);

        _mm256_maskstore_ps(y_ptr, mask, y0);
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Constant::Clamp(UInt32 n, Array<float>^ x, float cmin, float cmax, Array<float>^ y) {
    if (n <= 0) {
        return;
    }

    Util::CheckLength(n, x, y);

    const float* x_ptr = (const float*)(x->Ptr.ToPointer());
    float* y_ptr = (float*)(y->Ptr.ToPointer());

    int ret = const_clamp_s(n, x_ptr, cmin, cmax, y_ptr);

    Util::AssertReturnCode(ret);
}