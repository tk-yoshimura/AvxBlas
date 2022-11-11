#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_loadstore_xn_s.hpp"

using namespace System;

#pragma unmanaged

int ew_prod3_s(
    const uint n,
    infloats x1_ptr, infloats x2_ptr, infloats x3_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (((size_t)x1_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)x2_ptr % AVX2_ALIGNMENT) != 0 ||
        ((size_t)x3_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x01, x11, x21, x31, x02, x12, x22, x32, x03, x13, x23, x33;
    __m256 y0, y1, y2, y3;

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_load_x4_ps(x1_ptr, x01, x11, x21, x31);
        _mm256_load_x4_ps(x2_ptr, x02, x12, x22, x32);
        _mm256_load_x4_ps(x3_ptr, x03, x13, x23, x33);

        y0 = _mm256_mul_ps(_mm256_mul_ps(x01, x02), x03);
        y1 = _mm256_mul_ps(_mm256_mul_ps(x11, x12), x13);
        y2 = _mm256_mul_ps(_mm256_mul_ps(x21, x22), x23);
        y3 = _mm256_mul_ps(_mm256_mul_ps(x31, x32), x33);

        _mm256_stream_x4_ps(y_ptr, y0, y1, y2, y3);

        x1_ptr += AVX2_FLOAT_STRIDE * 4;
        x2_ptr += AVX2_FLOAT_STRIDE * 4;
        x3_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_load_x2_ps(x1_ptr, x01, x11);
        _mm256_load_x2_ps(x2_ptr, x02, x12);
        _mm256_load_x2_ps(x3_ptr, x03, x13);

        y0 = _mm256_mul_ps(_mm256_mul_ps(x01, x02), x03);
        y1 = _mm256_mul_ps(_mm256_mul_ps(x11, x12), x13);

        _mm256_stream_x2_ps(y_ptr, y0, y1);

        x1_ptr += AVX2_FLOAT_STRIDE * 2;
        x2_ptr += AVX2_FLOAT_STRIDE * 2;
        x3_ptr += AVX2_FLOAT_STRIDE * 2;
        y_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_load_x1_ps(x1_ptr, x01);
        _mm256_load_x1_ps(x2_ptr, x02);
        _mm256_load_x1_ps(x3_ptr, x03);

        y0 = _mm256_mul_ps(_mm256_mul_ps(x01, x02), x03);

        _mm256_stream_x1_ps(y_ptr, y0);

        x1_ptr += AVX2_FLOAT_STRIDE;
        x2_ptr += AVX2_FLOAT_STRIDE;
        x3_ptr += AVX2_FLOAT_STRIDE;
        y_ptr += AVX2_FLOAT_STRIDE;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps(r);

        x01 = _mm256_maskload_ps(x1_ptr, mask);
        x02 = _mm256_maskload_ps(x2_ptr, mask);
        x03 = _mm256_maskload_ps(x3_ptr, mask);

        y0 = _mm256_mul_ps(_mm256_mul_ps(x01, x02), x03);

        _mm256_maskstore_ps(y_ptr, mask, y0);
    }

    return SUCCESS;
}

int ew_prod4_s(
    const uint n,
    infloats x1_ptr, infloats x2_ptr, infloats x3_ptr, infloats x4_ptr, outfloats y_ptr) {

#ifdef _DEBUG
    if (((size_t)x1_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)x2_ptr % AVX2_ALIGNMENT) != 0 ||
        ((size_t)x3_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)x4_ptr % AVX2_ALIGNMENT) != 0 ||
        ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256 x01, x11, x21, x31, x02, x12, x22, x32, x03, x13, x23, x33, x04, x14, x24, x34;
    __m256 y0, y1, y2, y3;

    uint r = n;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_load_x4_ps(x1_ptr, x01, x11, x21, x31);
        _mm256_load_x4_ps(x2_ptr, x02, x12, x22, x32);
        _mm256_load_x4_ps(x3_ptr, x03, x13, x23, x33);
        _mm256_load_x4_ps(x4_ptr, x04, x14, x24, x34);

        y0 = _mm256_mul_ps(_mm256_mul_ps(x01, x02), _mm256_mul_ps(x03, x04));
        y1 = _mm256_mul_ps(_mm256_mul_ps(x11, x12), _mm256_mul_ps(x13, x14));
        y2 = _mm256_mul_ps(_mm256_mul_ps(x21, x22), _mm256_mul_ps(x23, x24));
        y3 = _mm256_mul_ps(_mm256_mul_ps(x31, x32), _mm256_mul_ps(x33, x34));

        _mm256_stream_x4_ps(y_ptr, y0, y1, y2, y3);

        x1_ptr += AVX2_FLOAT_STRIDE * 4;
        x2_ptr += AVX2_FLOAT_STRIDE * 4;
        x3_ptr += AVX2_FLOAT_STRIDE * 4;
        x4_ptr += AVX2_FLOAT_STRIDE * 4;
        y_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_load_x2_ps(x1_ptr, x01, x11);
        _mm256_load_x2_ps(x2_ptr, x02, x12);
        _mm256_load_x2_ps(x3_ptr, x03, x13);
        _mm256_load_x2_ps(x4_ptr, x04, x14);

        y0 = _mm256_mul_ps(_mm256_mul_ps(x01, x02), _mm256_mul_ps(x03, x04));
        y1 = _mm256_mul_ps(_mm256_mul_ps(x11, x12), _mm256_mul_ps(x13, x14));

        _mm256_stream_x2_ps(y_ptr, y0, y1);

        x1_ptr += AVX2_FLOAT_STRIDE * 2;
        x2_ptr += AVX2_FLOAT_STRIDE * 2;
        x3_ptr += AVX2_FLOAT_STRIDE * 2;
        x4_ptr += AVX2_FLOAT_STRIDE * 2;
        y_ptr += AVX2_FLOAT_STRIDE * 2;
        r -= AVX2_FLOAT_STRIDE * 2;
    }
    if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_load_x1_ps(x1_ptr, x01);
        _mm256_load_x1_ps(x2_ptr, x02);
        _mm256_load_x1_ps(x3_ptr, x03);
        _mm256_load_x1_ps(x4_ptr, x04);

        y0 = _mm256_mul_ps(_mm256_mul_ps(x01, x02), _mm256_mul_ps(x03, x04));

        _mm256_stream_x1_ps(y_ptr, y0);

        x1_ptr += AVX2_FLOAT_STRIDE;
        x2_ptr += AVX2_FLOAT_STRIDE;
        x3_ptr += AVX2_FLOAT_STRIDE;
        x4_ptr += AVX2_FLOAT_STRIDE;
        y_ptr += AVX2_FLOAT_STRIDE;
        r -= AVX2_FLOAT_STRIDE;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps(r);

        x01 = _mm256_maskload_ps(x1_ptr, mask);
        x02 = _mm256_maskload_ps(x2_ptr, mask);
        x03 = _mm256_maskload_ps(x3_ptr, mask);
        x04 = _mm256_maskload_ps(x4_ptr, mask);

        y0 = _mm256_mul_ps(_mm256_mul_ps(x01, x02), _mm256_mul_ps(x03, x04));

        _mm256_maskstore_ps(y_ptr, mask, y0);
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Elementwise::Prod(UInt32 n, array<Array<float>^>^ xs, Array<float>^ y) {
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

    float* y_ptr = (float*)(y->Ptr.ToPointer());

    if (xs->Length == 3) {
        const float* x1_ptr = (const float*)(xs[0]->Ptr.ToPointer());
        const float* x2_ptr = (const float*)(xs[1]->Ptr.ToPointer());
        const float* x3_ptr = (const float*)(xs[2]->Ptr.ToPointer());

        int ret = ew_prod3_s(n, x1_ptr, x2_ptr, x3_ptr, y_ptr);
        Util::AssertReturnCode(ret);
        return;
    }

    int r = 0;
    {
        const float* x1_ptr = (const float*)(xs[0]->Ptr.ToPointer());
        const float* x2_ptr = (const float*)(xs[1]->Ptr.ToPointer());
        const float* x3_ptr = (const float*)(xs[2]->Ptr.ToPointer());
        const float* x4_ptr = (const float*)(xs[3]->Ptr.ToPointer());

        int ret = ew_prod4_s(n, x1_ptr, x2_ptr, x3_ptr, x4_ptr, y_ptr);
        Util::AssertReturnCode(ret);
        r += 4;
    }

    while (r < xs->Length) {
        if (xs->Length - r >= 3) {
            const float* x1_ptr = (const float*)(xs[r]->Ptr.ToPointer());
            const float* x2_ptr = (const float*)(xs[r + 1]->Ptr.ToPointer());
            const float* x3_ptr = (const float*)(xs[r + 2]->Ptr.ToPointer());

            int ret = ew_prod4_s(n, x1_ptr, x2_ptr, x3_ptr, y_ptr, y_ptr);
            Util::AssertReturnCode(ret);
            r += 3;
            continue;
        }
        if (xs->Length - r >= 2) {
            const float* x1_ptr = (const float*)(xs[r]->Ptr.ToPointer());
            const float* x2_ptr = (const float*)(xs[r + 1]->Ptr.ToPointer());

            int ret = ew_prod3_s(n, x1_ptr, x2_ptr, y_ptr, y_ptr);
            Util::AssertReturnCode(ret);
            break;
        }
        if (xs->Length - r >= 1) {
            Elementwise::Mul(n, xs[r], y, y);
            break;
        }
    }
}
