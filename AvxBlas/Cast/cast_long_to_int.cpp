#include "../avxblas.h"
#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_loadstore_xn_epi32.hpp"
#include "../Inline/inline_loadstore_xn_epi64.hpp"

using namespace System;

#pragma unmanaged

__forceinline __m256i _mm256_cvtepi64x2_epi32(__m256i a, __m256i b) {
    __m256i y = _mm256_castpd_si256(
        _mm256_permute4x64_pd(
            _mm256_castps_pd(_mm256_shuffle_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), 0b10001000)),
            _MM_PERM_DBCA
        )
    );

    return y;
}

int cast_long_to_int(
    const uint n,
    inulongs x_ptr, outuints y_ptr) {

    if ((size_t)x_ptr == (size_t)y_ptr) {
        return FAILURE_BADPARAM;
    }

#ifdef _DEBUG
    if (((size_t)x_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)y_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif // _DEBUG

    __m256i x0, x1, x2, x3;
    __m256i y01, y23;

    uint r = n;

    while (r >= AVX2_EPI64_STRIDE * 4) {
        _mm256_load_x4_epi64(x_ptr, x0, x1, x2, x3);

        y01 = _mm256_cvtepi64x2_epi32(x0, x1);
        y23 = _mm256_cvtepi64x2_epi32(x2, x3);
        
        _mm256_stream_x2_epi32(y_ptr, y01, y23);

        x_ptr += AVX2_EPI64_STRIDE * 4;
        y_ptr += AVX2_EPI32_STRIDE * 2;
        r -= AVX2_EPI64_STRIDE * 4;
    }
    if (r >= AVX2_EPI64_STRIDE * 2) {
        _mm256_load_x2_epi64(x_ptr, x0, x1);

        y01 = _mm256_cvtepi64x2_epi32(x0, x1);

        _mm256_stream_x1_epi32(y_ptr, y01);

        x_ptr += AVX2_EPI64_STRIDE * 2;
        y_ptr += AVX2_EPI32_STRIDE;
        r -= AVX2_EPI64_STRIDE * 2;
    }
    if (r >= AVX2_EPI64_STRIDE) {
        _mm256_load_x1_epi64(x_ptr, x0);

        y01 = _mm256_cvtepi64x2_epi32(x0, _mm256_undefined_si256());

        _mm256_maskstore_x1_epi32(y_ptr, y01, _mm256_setmask_ps(AVX2_EPI64_STRIDE));

        x_ptr += AVX2_EPI64_STRIDE;
        y_ptr += AVX2_EPI32_STRIDE / 2;
        r -= AVX2_EPI64_STRIDE;
    }
    if (r > 0) {
        const __m256i mask_ps = _mm256_setmask_ps(r);
        const __m256i mask_pd = _mm256_setmask_pd(r);

        _mm256_maskload_x1_epi64(x_ptr, x0, mask_pd);

        y01 = _mm256_cvtepi64x2_epi32(x0, _mm256_undefined_si256());

        _mm256_maskstore_x1_epi32(y_ptr, y01, mask_ps);
    }

    return SUCCESS;
}

#pragma managed

void AvxBlas::Cast::AsType(UInt32 n, Array<Int64>^ x, Array<Int32>^ y) {
    if (n <= 0) {
        return;
    }

    Util::CheckLength(n, x);
    Util::CheckLength(n, y);

    const ulong* x_ptr = (const ulong*)(x->Ptr.ToPointer());
    uint* y_ptr = (uint*)(y->Ptr.ToPointer());

    int ret = cast_long_to_int(n, x_ptr, y_ptr);

    Util::AssertReturnCode(ret);
}