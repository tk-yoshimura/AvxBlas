#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_loadstore_xn_d.hpp"

using namespace System;

#pragma unmanaged

__forceinline __m256d _mm256_lerp_pd(__m256d xc, __m256d x1, __m256d x2) {
    __m256d y = _mm256_add_pd(x2, _mm256_mul_pd(_mm256_sub_pd(x1, x2), xc));

    return y;
}

int ew_lerp_d(
    const uint n,
    indoubles x1_ptr, indoubles x2_ptr, indoubles x3_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (((size_t)x1_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)x2_ptr % AVX2_ALIGNMENT) != 0 ||
        ((size_t)x3_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256d x01, x11, x21, x31, x02, x12, x22, x32, x03, x13, x23, x33;
    __m256d y0, y1, y2, y3;

    uint r = n;

    while (r >= AVX2_DOUBLE_STRIDE * 4) {
        _mm256_load_x4_pd(x1_ptr, x01, x11, x21, x31);
        _mm256_load_x4_pd(x2_ptr, x02, x12, x22, x32);
        _mm256_load_x4_pd(x3_ptr, x03, x13, x23, x33);

        y0 = _mm256_lerp_pd(x03, x01, x02);
        y1 = _mm256_lerp_pd(x13, x11, x12);
        y2 = _mm256_lerp_pd(x23, x21, x22);
        y3 = _mm256_lerp_pd(x33, x31, x32);

        _mm256_stream_x4_pd(y_ptr, y0, y1, y2, y3);

        x1_ptr += AVX2_DOUBLE_STRIDE * 4;
        x2_ptr += AVX2_DOUBLE_STRIDE * 4;
        x3_ptr += AVX2_DOUBLE_STRIDE * 4;
        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 4;
    }
    if (r >= AVX2_DOUBLE_STRIDE * 2) {
        _mm256_load_x2_pd(x1_ptr, x01, x11);
        _mm256_load_x2_pd(x2_ptr, x02, x12);
        _mm256_load_x2_pd(x3_ptr, x03, x13);

        y0 = _mm256_lerp_pd(x03, x01, x02);
        y1 = _mm256_lerp_pd(x13, x11, x12);

        _mm256_stream_x2_pd(y_ptr, y0, y1);

        x1_ptr += AVX2_DOUBLE_STRIDE * 2;
        x2_ptr += AVX2_DOUBLE_STRIDE * 2;
        x3_ptr += AVX2_DOUBLE_STRIDE * 2;
        y_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE * 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE) {
        _mm256_load_x1_pd(x1_ptr, x01);
        _mm256_load_x1_pd(x2_ptr, x02);
        _mm256_load_x1_pd(x3_ptr, x03);

        y0 = _mm256_lerp_pd(x03, x01, x02);

        _mm256_stream_x1_pd(y_ptr, y0);

        x1_ptr += AVX2_DOUBLE_STRIDE;
        x2_ptr += AVX2_DOUBLE_STRIDE;
        x3_ptr += AVX2_DOUBLE_STRIDE;
        y_ptr += AVX2_DOUBLE_STRIDE;
        r -= AVX2_DOUBLE_STRIDE;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_pd(r);

        x01 = _mm256_maskload_pd(x1_ptr, mask);
        x02 = _mm256_maskload_pd(x2_ptr, mask);
        x03 = _mm256_maskload_pd(x3_ptr, mask);

        y0 = _mm256_lerp_pd(x03, x01, x02);

        _mm256_maskstore_pd(y_ptr, mask, y0);
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Elementwise::Lerp(UInt32 n, Array<double>^ x1, Array<double>^ x2, Array<double>^ x3, Array<double>^ y) {
    if (n <= 0) {
        return;
    }

    Util::CheckLength(n, x1, x2, x3, y);

    const double* x1_ptr = (const double*)(x1->Ptr.ToPointer());
    const double* x2_ptr = (const double*)(x2->Ptr.ToPointer());
    const double* x3_ptr = (const double*)(x3->Ptr.ToPointer());
    double* y_ptr = (double*)(y->Ptr.ToPointer());

    int ret = ew_lerp_d(n, x1_ptr, x2_ptr, x3_ptr, y_ptr);

    Util::AssertReturnCode(ret);
}
