#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_loadstore_xn_epi32.hpp"
#include "../Inline/inline_loadstore_xn_epi64.hpp"

using namespace System;

#pragma unmanaged

int cast_int_to_long(
    const uint n,
    inuints x_ptr, outulongs y_ptr) {

    if ((size_t)x_ptr == (size_t)y_ptr) {
        return FAILURE_BADPARAM;
    }

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m128i x0, x1, x2, x3;
    __m256i y0, y1, y2, y3;

    uint r = n;

    while (r >= AVX2_EPI32_STRIDE * 2) {
        x0 = _mm_load_epi32(x_ptr);
        x1 = _mm_load_epi32(x_ptr + AVX2_EPI32_STRIDE / 2);
        x2 = _mm_load_epi32(x_ptr + AVX2_EPI32_STRIDE);
        x3 = _mm_load_epi32(x_ptr + AVX2_EPI32_STRIDE * 3 / 2);

        y0 = _mm256_cvtepi32_epi64(x0);
        y1 = _mm256_cvtepi32_epi64(x1);
        y2 = _mm256_cvtepi32_epi64(x2);
        y3 = _mm256_cvtepi32_epi64(x3);

        _mm256_stream_x4_epi64(y_ptr, y0, y1, y2, y3);

        x_ptr += AVX2_EPI32_STRIDE * 2;
        y_ptr += AVX2_EPI64_STRIDE * 4;
        r -= AVX2_EPI32_STRIDE * 2;
    }
    if (r >= AVX2_EPI32_STRIDE) {
        x0 = _mm_load_epi32(x_ptr);
        x1 = _mm_load_epi32(x_ptr + AVX2_EPI32_STRIDE / 2);

        y0 = _mm256_cvtepi32_epi64(x0);
        y1 = _mm256_cvtepi32_epi64(x1);

        _mm256_stream_x2_epi64(y_ptr, y0, y1);

        x_ptr += AVX2_EPI32_STRIDE;
        y_ptr += AVX2_EPI64_STRIDE * 2;
        r -= AVX2_EPI32_STRIDE;
    }
    if (r >= AVX2_EPI32_STRIDE / 2) {
        x0 = _mm_load_epi32(x_ptr);

        y0 = _mm256_cvtepi32_epi64(x0);

        _mm256_stream_x1_epi64(y_ptr, y0);

        x_ptr += AVX2_EPI32_STRIDE / 2;
        y_ptr += AVX2_EPI64_STRIDE;
        r -= AVX2_EPI32_STRIDE / 2;
    }
    if (r > 0) {
        const __m128i mask_ps = _mm_setmask_ps(r);
        const __m256i mask_pd = _mm256_setmask_pd(r);

        x0 = _mm_maskload_epi32((const int*)x_ptr, mask_ps);

        y0 = _mm256_cvtepi32_epi64(x0);

        _mm256_maskstore_x1_epi64(y_ptr, y0, mask_pd);
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Cast::AsType(UInt32 n, Array<Int32>^ x, Array<Int64>^ y) {
    if (n <= 0) {
        return;
    }

    Util::CheckLength(n, x);
    Util::CheckLength(n, y);

    const uint* x_ptr = (const uint*)(x->Ptr.ToPointer());
    ulong* y_ptr = (ulong*)(y->Ptr.ToPointer());

    int ret = cast_int_to_long(n, x_ptr, y_ptr);

    Util::AssertReturnCode(ret);
}