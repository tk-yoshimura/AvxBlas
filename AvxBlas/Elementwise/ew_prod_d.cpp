#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_loadstore_xn_d.hpp"

using namespace System;

#pragma unmanaged

int ew_prod3_d(
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

        y0 = _mm256_mul_pd(_mm256_mul_pd(x01, x02), x03);
        y1 = _mm256_mul_pd(_mm256_mul_pd(x11, x12), x13);
        y2 = _mm256_mul_pd(_mm256_mul_pd(x21, x22), x23);
        y3 = _mm256_mul_pd(_mm256_mul_pd(x31, x32), x33);

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

        y0 = _mm256_mul_pd(_mm256_mul_pd(x01, x02), x03);
        y1 = _mm256_mul_pd(_mm256_mul_pd(x11, x12), x13);

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

        y0 = _mm256_mul_pd(_mm256_mul_pd(x01, x02), x03);

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

        y0 = _mm256_mul_pd(_mm256_mul_pd(x01, x02), x03);

        _mm256_maskstore_pd(y_ptr, mask, y0);
    }

    return SUCCESS;
}

int ew_prod4_d(
    const uint n,
    indoubles x1_ptr, indoubles x2_ptr, indoubles x3_ptr, indoubles x4_ptr, outdoubles y_ptr) {

#ifdef _DEBUG
    if (((size_t)x1_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)x2_ptr % AVX2_ALIGNMENT) != 0 ||
        ((size_t)x3_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)x4_ptr % AVX2_ALIGNMENT) != 0 ||
        ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256d x01, x11, x21, x31, x02, x12, x22, x32, x03, x13, x23, x33, x04, x14, x24, x34;
    __m256d y0, y1, y2, y3;

    uint r = n;

    while (r >= AVX2_DOUBLE_STRIDE * 4) {
        _mm256_load_x4_pd(x1_ptr, x01, x11, x21, x31);
        _mm256_load_x4_pd(x2_ptr, x02, x12, x22, x32);
        _mm256_load_x4_pd(x3_ptr, x03, x13, x23, x33);
        _mm256_load_x4_pd(x4_ptr, x04, x14, x24, x34);

        y0 = _mm256_mul_pd(_mm256_mul_pd(x01, x02), _mm256_mul_pd(x03, x04));
        y1 = _mm256_mul_pd(_mm256_mul_pd(x11, x12), _mm256_mul_pd(x13, x14));
        y2 = _mm256_mul_pd(_mm256_mul_pd(x21, x22), _mm256_mul_pd(x23, x24));
        y3 = _mm256_mul_pd(_mm256_mul_pd(x31, x32), _mm256_mul_pd(x33, x34));

        _mm256_stream_x4_pd(y_ptr, y0, y1, y2, y3);

        x1_ptr += AVX2_DOUBLE_STRIDE * 4;
        x2_ptr += AVX2_DOUBLE_STRIDE * 4;
        x3_ptr += AVX2_DOUBLE_STRIDE * 4;
        x4_ptr += AVX2_DOUBLE_STRIDE * 4;
        y_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 4;
    }
    if (r >= AVX2_DOUBLE_STRIDE * 2) {
        _mm256_load_x2_pd(x1_ptr, x01, x11);
        _mm256_load_x2_pd(x2_ptr, x02, x12);
        _mm256_load_x2_pd(x3_ptr, x03, x13);
        _mm256_load_x2_pd(x4_ptr, x04, x14);

        y0 = _mm256_mul_pd(_mm256_mul_pd(x01, x02), _mm256_mul_pd(x03, x04));
        y1 = _mm256_mul_pd(_mm256_mul_pd(x11, x12), _mm256_mul_pd(x13, x14));

        _mm256_stream_x2_pd(y_ptr, y0, y1);

        x1_ptr += AVX2_DOUBLE_STRIDE * 2;
        x2_ptr += AVX2_DOUBLE_STRIDE * 2;
        x3_ptr += AVX2_DOUBLE_STRIDE * 2;
        x4_ptr += AVX2_DOUBLE_STRIDE * 2;
        y_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE * 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE) {
        _mm256_load_x1_pd(x1_ptr, x01);
        _mm256_load_x1_pd(x2_ptr, x02);
        _mm256_load_x1_pd(x3_ptr, x03);
        _mm256_load_x1_pd(x4_ptr, x04);

        y0 = _mm256_mul_pd(_mm256_mul_pd(x01, x02), _mm256_mul_pd(x03, x04));

        _mm256_stream_x1_pd(y_ptr, y0);

        x1_ptr += AVX2_DOUBLE_STRIDE;
        x2_ptr += AVX2_DOUBLE_STRIDE;
        x3_ptr += AVX2_DOUBLE_STRIDE;
        x4_ptr += AVX2_DOUBLE_STRIDE;
        y_ptr += AVX2_DOUBLE_STRIDE;
        r -= AVX2_DOUBLE_STRIDE;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_pd(r);

        x01 = _mm256_maskload_pd(x1_ptr, mask);
        x02 = _mm256_maskload_pd(x2_ptr, mask);
        x03 = _mm256_maskload_pd(x3_ptr, mask);
        x04 = _mm256_maskload_pd(x4_ptr, mask);

        y0 = _mm256_mul_pd(_mm256_mul_pd(x01, x02), _mm256_mul_pd(x03, x04));

        _mm256_maskstore_pd(y_ptr, mask, y0);
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Elementwise::Prod(UInt32 n, array<Array<double>^>^ xs, Array<double>^ y) {
    if (n <= 0) {
        return;
    }

    if (xs->Length <= 2) {
        if (xs->Length == 0) {
            Initialize::Clear(n, 1, y);
        }
        else if (xs->Length == 1) {
            Elementwise::Copy(n, xs[0], y);
        }
        else if (xs->Length == 2) {
            Elementwise::Mul(n, xs[0], xs[1], y);
        }
        return;
    }

    Util::CheckLength(n, xs);
    Util::CheckLength(n, y);

    double* y_ptr = (double*)(y->Ptr.ToPointer());

    if (xs->Length == 3) {
        const double* x1_ptr = (const double*)(xs[0]->Ptr.ToPointer());
        const double* x2_ptr = (const double*)(xs[1]->Ptr.ToPointer());
        const double* x3_ptr = (const double*)(xs[2]->Ptr.ToPointer());

        int ret = ew_prod3_d(n, x1_ptr, x2_ptr, x3_ptr, y_ptr);
        Util::AssertReturnCode(ret);
        return;
    }

    int r = 0;
    {
        const double* x1_ptr = (const double*)(xs[0]->Ptr.ToPointer());
        const double* x2_ptr = (const double*)(xs[1]->Ptr.ToPointer());
        const double* x3_ptr = (const double*)(xs[2]->Ptr.ToPointer());
        const double* x4_ptr = (const double*)(xs[3]->Ptr.ToPointer());

        int ret = ew_prod4_d(n, x1_ptr, x2_ptr, x3_ptr, x4_ptr, y_ptr);
        Util::AssertReturnCode(ret);
        r += 4;
    }

    while (r < xs->Length) {
        if (xs->Length - r >= 3) {
            const double* x1_ptr = (const double*)(xs[r]->Ptr.ToPointer());
            const double* x2_ptr = (const double*)(xs[r + 1]->Ptr.ToPointer());
            const double* x3_ptr = (const double*)(xs[r + 2]->Ptr.ToPointer());

            int ret = ew_prod4_d(n, x1_ptr, x2_ptr, x3_ptr, y_ptr, y_ptr);
            Util::AssertReturnCode(ret);
            r += 3;
            continue;
        }
        if (xs->Length - r >= 2) {
            const double* x1_ptr = (const double*)(xs[r]->Ptr.ToPointer());
            const double* x2_ptr = (const double*)(xs[r + 1]->Ptr.ToPointer());

            int ret = ew_prod3_d(n, x1_ptr, x2_ptr, y_ptr, y_ptr);
            Util::AssertReturnCode(ret);
            break;
        }
        if (xs->Length - r >= 1) {
            Elementwise::Mul(n, xs[r], y, y);
            break;
        }
    }
}
